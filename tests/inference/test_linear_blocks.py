"""Tests for declared-linear parameter blocks.

`linear=True` is a claim about the model, and a claim that gets exploited
(conjugate-Gaussian solves and, later, GCR sampling) has to be checkable —
otherwise a wrong declaration returns a confident, wrong posterior rather than
an error. These tests cover both halves: that the exported operator really is
the model's linear part, and that a false declaration is caught.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.combinators import SumOperator
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Latent,
    ParameterSpace,
    check_linearity,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import GainOperator, SkyOperator

SKY_A, SKY_B, GAIN = 100.0, 20.0, 1.5


@pytest.fixture
def twin():
    """Two additive sky terms through a gain.

    Linear in either amplitude — and with a NONZERO offset when only one of
    them is the latent, which is what makes the offset worth testing.
    """
    return Pipeline(
        SumOperator(
            SkyOperator(amplitude=jnp.array(SKY_A)),
            SkyOperator(amplitude=jnp.array(SKY_B)),
            names=("sky_a", "sky_b"),
        ),
        GainOperator(gain=jnp.array(GAIN)),
        names=("sum", "gain"),
    )


@pytest.fixture
def linear_space():
    return ParameterSpace.direct(
        "amp", init=SKY_A, into=lambda p: p["sum"]["sky_a"].amplitude, linear=True
    )


class TestLinearOperator:
    def test_offset_is_the_model_at_zero(self, twin, linear_space, template_state):
        block = linear_operator(linear_space, twin, template_state)
        # only sky_b survives when amp = 0
        assert jnp.allclose(block.offset, SKY_B * GAIN)

    def test_offset_plus_forward_reproduces_the_model(self, twin, linear_space, template_state):
        block = linear_operator(linear_space, twin, template_state)
        forward, _ = linear_space.forward_fn(twin, template_state)
        x = jnp.array(37.0)
        assert jnp.allclose(block.offset + block.forward(x), forward({"amp": x}))

    def test_adjoint_satisfies_the_dot_product_identity(self, twin, linear_space, template_state):
        """<A x, y> == <x, A^T y> — the definition of the adjoint, and the only
        cheap way to know the transpose really is the transpose."""
        block = linear_operator(linear_space, twin, template_state)
        x = jnp.array(2.5)
        y = jax.random.normal(jax.random.key(0), block.offset.shape)
        lhs = jnp.sum(block.forward(x) * y)
        rhs = jnp.sum(x * block.adjoint(y))
        assert float(lhs) == pytest.approx(float(rhs), rel=1e-5)

    def test_shape_is_the_latent_shape_not_the_leaf_shape(self, twin, template_state):
        """A linear block is sized by what you infer, not by where it lands —
        the sky-alm case in miniature."""
        space = ParameterSpace.direct(
            "amps", init=jnp.full((3,), SKY_A / 3.0),
            into=lambda p: p["sum"]["sky_a"].amplitude, fn=jnp.sum, linear=True,
        )
        block = linear_operator(space, twin, template_state)
        assert block.shape == (3,)
        assert block.forward(jnp.ones(3)).shape == block.offset.shape
        assert block.adjoint(jnp.ones_like(block.offset)).shape == (3,)

    def test_undeclared_latent_is_refused(self, twin, template_state):
        space = ParameterSpace.direct(
            "amp", init=SKY_A, into=lambda p: p["sum"]["sky_a"].amplitude
        )
        with pytest.raises(ParameterSpaceError, match="linear=True"):
            linear_operator(space, twin, template_state)

    def test_a_false_declaration_is_refused_at_export(self, twin, template_state):
        space = ParameterSpace.direct(
            "log_amp", init=jnp.log(SKY_A), into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=jnp.exp, linear=True,
        )
        with pytest.raises(ParameterSpaceError, match="not affine"):
            linear_operator(space, twin, template_state)


class ComplexCoeffOperator(AbstractOperator):
    """A real observation linear in COMPLEX coefficients — sky alms in miniature.

    Test double, not physics: it exists so the complex adjoint convention is a
    pinned contract rather than folklore.
    """

    provides: ClassVar[tuple[str, ...]] = ("data",)

    coeffs: jax.Array
    matrix: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        return state.with_data(jnp.real(self.matrix @ self.coeffs).reshape(n_time, n_freq))


class TestComplexLatents:
    """Sky alms are complex, so the adjoint convention has to be nailed down."""

    @pytest.fixture
    def complex_twin(self, template_state):
        n_row = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        key = jax.random.key(0)
        matrix = jax.random.normal(key, (n_row, 3)) + 1j * jax.random.normal(
            jax.random.fold_in(key, 1), (n_row, 3)
        )
        return Pipeline(
            ComplexCoeffOperator(coeffs=jnp.zeros(3, dtype=matrix.dtype), matrix=matrix),
            names=("sky",),
        )

    @pytest.fixture
    def complex_space(self):
        return ParameterSpace.direct(
            "coeffs", init=jnp.ones(3, dtype=jnp.complex64),
            into=lambda p: p["sky"].coeffs, linear=True,
        )

    def test_a_complex_block_passes_the_linearity_check(
        self, complex_twin, complex_space, template_state
    ):
        errors = check_linearity(complex_space, complex_twin, template_state)
        assert all(err < 1e-4 for err in errors.values())

    def test_adjoint_is_the_adjoint_of_the_REAL_inner_product(
        self, complex_twin, complex_space, template_state
    ):
        """`adjoint` is `jax.vjp`, which returns the CONJUGATE gradient for
        complex inputs. The identity that therefore holds is

            Re sum(x * adjoint(y))  ==  sum(forward(x) * y)

        — the adjoint with respect to the real inner product, which is exactly
        the pairing a Gaussian likelihood forms. Taking `conj(x)` instead, as
        the sesquilinear convention would, does NOT hold; pinned here so the
        distinction cannot rot into a silent factor.
        """
        block = linear_operator(complex_space, complex_twin, template_state)
        key = jax.random.key(7)
        x = jax.random.normal(key, (3,)) + 1j * jax.random.normal(
            jax.random.fold_in(key, 1), (3,)
        )
        y = jax.random.normal(jax.random.fold_in(key, 2), block.offset.shape)

        paired = float(jnp.sum(block.forward(x) * y))
        assert float(jnp.real(jnp.sum(x * block.adjoint(y)))) == pytest.approx(
            paired, rel=1e-5
        )
        assert float(jnp.real(jnp.sum(jnp.conj(x) * block.adjoint(y)))) != pytest.approx(
            paired, rel=1e-3
        )


def _dense_reference(block, observed, noise_std, prior_std):
    """The same solve, done the expensive, obviously-correct way.

    Builds A column by column over the block's REAL degrees of freedom (real
    and imaginary parts separately when the latent is complex, since the map
    is R-linear but not C-linear), then solves the normal equations densely.
    """
    is_complex = jnp.issubdtype(block.dtype, jnp.complexfloating)
    n = int(jnp.prod(jnp.array(block.shape))) if block.shape else 1

    columns = []
    for index in range(n):
        for unit in (1.0, 1j) if is_complex else (1.0,):
            basis = jnp.zeros(n, dtype=block.dtype).at[index].set(unit)
            columns.append(jnp.ravel(block.forward(basis.reshape(block.shape))))
    A = jnp.stack(columns, axis=1)

    weight = 1.0 / jnp.asarray(noise_std) ** 2
    residual = jnp.ravel(observed - block.offset)
    normal = A.T @ (weight * A) + jnp.eye(A.shape[1]) / jnp.asarray(prior_std) ** 2
    solution = jnp.linalg.solve(normal, A.T @ (weight * residual))
    if is_complex:
        return (solution[0::2] + 1j * solution[1::2]).reshape(block.shape)
    return solution.reshape(block.shape)


class TestWienerSolve:
    """The MAP/Wiener solve the exported operator makes possible.

    The block used here is a per-sample gain: ``A`` is then full column rank,
    so the dense reference is a fair comparison. A block that maps several
    latents into ONE scalar leaf (say via ``jnp.sum``) is rank-one by
    construction — its normal operator is a rank-one matrix plus the prior
    ridge, and the dense solve becomes the unstable one, not CG.
    """

    @pytest.fixture
    def gain_block(self, twin, template_state):
        n_time = template_state.coords.time.shape[0]
        wide = eqx.tree_at(lambda p: p["gain"].gain, twin, jnp.full((n_time,), GAIN))
        space = ParameterSpace.direct(
            "gains", init=jnp.full((n_time,), GAIN),
            into=lambda p: p["gain"].gain, linear=True,
        )
        return linear_operator(space, wide, template_state)

    @pytest.fixture
    def gain_truth(self, template_state):
        n_time = template_state.coords.time.shape[0]
        return GAIN + 0.1 * jnp.arange(n_time, dtype=float)

    def test_matches_a_dense_solve(self, gain_block, gain_truth):
        observed = gain_block.offset + gain_block.forward(gain_truth)
        solved, _ = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=5.0)
        expected = _dense_reference(gain_block, observed, 1.0, 5.0)
        assert jnp.allclose(solved, expected, rtol=1e-3, atol=1e-3)

    def test_recovers_a_noiseless_signal_under_a_weak_prior(self, gain_block, gain_truth):
        observed = gain_block.offset + gain_block.forward(gain_truth)
        solved, _ = wiener_solve(gain_block, observed, noise_std=1e-2, prior_std=1e3)
        assert jnp.allclose(solved, gain_truth, rtol=1e-3)

    def test_matches_a_dense_solve_for_a_complex_block(self, template_state):
        n_row = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        key = jax.random.key(3)
        matrix = jax.random.normal(key, (n_row, 3)) + 1j * jax.random.normal(
            jax.random.fold_in(key, 1), (n_row, 3)
        )
        complex_twin = Pipeline(
            ComplexCoeffOperator(coeffs=jnp.zeros(3, dtype=matrix.dtype), matrix=matrix),
            names=("sky",),
        )
        space = ParameterSpace.direct(
            "coeffs", init=jnp.ones(3, dtype=jnp.complex64),
            into=lambda p: p["sky"].coeffs, linear=True,
        )
        block = linear_operator(space, complex_twin, template_state)
        truth = jnp.array([1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 1.0j])
        observed = block.offset + block.forward(truth)
        solved, _ = wiener_solve(block, observed, noise_std=0.1, prior_std=10.0)
        expected = _dense_reference(block, observed, 0.1, 10.0)
        assert jnp.allclose(solved, expected, rtol=1e-3, atol=1e-3)

    def test_a_complex_block_recovers_a_noiseless_signal(self, template_state):
        n_row = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        matrix = jax.random.normal(jax.random.key(5), (n_row, 3)).astype(float) + 0j
        complex_twin = Pipeline(
            ComplexCoeffOperator(coeffs=jnp.zeros(3, dtype=matrix.dtype), matrix=matrix),
            names=("sky",),
        )
        space = ParameterSpace.direct(
            "coeffs", init=jnp.ones(3, dtype=jnp.complex64),
            into=lambda p: p["sky"].coeffs, linear=True,
        )
        block = linear_operator(space, complex_twin, template_state)
        truth = jnp.array([1.0 + 0j, -0.5 + 0j, 3.0 + 0j])
        observed = block.offset + block.forward(truth)
        solved, _ = wiener_solve(block, observed, noise_std=1e-3, prior_std=1e4)
        assert jnp.allclose(jnp.real(solved), jnp.real(truth), atol=1e-2)

    def test_a_strong_prior_shrinks_towards_zero(self, gain_block, gain_truth):
        observed = gain_block.offset + gain_block.forward(gain_truth)
        loose, _ = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=1e3)
        tight, _ = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=1e-3)
        assert jnp.linalg.norm(tight) < jnp.linalg.norm(loose)

    def test_the_relative_residual_is_reported(self, gain_block, gain_truth):
        observed = gain_block.offset + gain_block.forward(gain_truth)
        _, residual = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=5.0)
        assert float(residual) < 1e-4

    def test_a_prior_is_required(self, gain_block):
        """Without one the normal operator can be singular, and CG would return
        a finite, arbitrary answer rather than complain."""
        with pytest.raises(ParameterSpaceError, match="prior_std"):
            wiener_solve(gain_block, gain_block.offset, noise_std=1.0, prior_std=None)

    def test_solving_is_jittable(self, gain_block, gain_truth):
        observed = gain_block.offset + gain_block.forward(gain_truth)
        run = jax.jit(lambda d: wiener_solve(gain_block, d, noise_std=1.0, prior_std=5.0)[0])
        direct, _ = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=5.0)
        assert jnp.allclose(run(observed), direct, rtol=1e-4)


class TestCheckLinearity:
    def test_a_linear_block_passes(self, twin, linear_space, template_state):
        errors = check_linearity(linear_space, twin, template_state)
        assert all(err < 1e-4 for err in errors.values())

    def test_an_exponential_block_is_caught(self, twin, template_state):
        space = ParameterSpace.direct(
            "log_amp", init=jnp.log(SKY_A), into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=jnp.exp, linear=True,
        )
        with pytest.raises(ParameterSpaceError, match="not affine"):
            check_linearity(space, twin, template_state)

    def test_a_block_linear_only_at_small_scale_is_caught(self, twin, template_state):
        """The reason probes must span extreme scales, not just moderate ones.

        A quadratic with a tiny coefficient is indistinguishable from linear
        near the origin and grossly nonlinear far from it. A probe suite that
        only samples "reasonable" values would sign this off.
        """
        space = ParameterSpace.direct(
            "amp", init=SKY_A, into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=lambda x: x + 1e-8 * x**2, linear=True,
        )
        with pytest.raises(ParameterSpaceError, match="not affine"):
            check_linearity(space, twin, template_state)

    def test_the_same_block_passes_when_only_small_scales_are_probed(
        self, twin, template_state
    ):
        """Companion to the test above: it is the extreme probe that catches it."""
        space = ParameterSpace.direct(
            "amp", init=SKY_A, into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=lambda x: x + 1e-8 * x**2, linear=True,
        )
        errors = check_linearity(space, twin, template_state, scales=(1e-3,))
        assert all(err < 1e-4 for err in errors.values())

    def test_roundoff_at_a_tiny_probe_is_not_mistaken_for_curvature(
        self, twin, linear_space, template_state
    ):
        """Regression: an exactly-linear block probed at 1e-9 of its own scale.

        The relative measure (departure / variation) explodes there — the
        variation is vanishing but roundoff is not — so a relative test alone
        rejects a perfectly linear block. That false positive is worse than no
        check, because the cure users reach for is to switch the check off.
        """
        check_linearity(linear_space, twin, template_state, scales=(1e-9, 1e-6, 1.0))

    def test_unknown_latent_name_is_refused(self, twin, linear_space, template_state):
        with pytest.raises(ParameterSpaceError, match="No latent named"):
            check_linearity(linear_space, twin, template_state, name="nope")

    def test_multi_latent_space_needs_an_explicit_name(self, twin, template_state):
        space = ParameterSpace(
            latents=[
                Latent("amp_a", init=SKY_A, linear=True),
                Latent("amp_b", init=SKY_B, linear=True),
            ],
            bindings=[
                Bind("amp_a", into=lambda p: p["sum"]["sky_a"].amplitude),
                Bind("amp_b", into=lambda p: p["sum"]["sky_b"].amplitude),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="which latent"):
            linear_operator(space, twin, template_state)
        # naming one is enough
        block = linear_operator(space, twin, template_state, name="amp_a")
        assert jnp.allclose(block.offset, SKY_B * GAIN)
