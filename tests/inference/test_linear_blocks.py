"""Tests for declared-linear parameter blocks.

Verified under BOTH float32 and JAX_ENABLE_X64=1: the project documents x64 for
quantitative work, so a suite that only holds at the default precision would
not be testing the mode that matters.

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

from rheplicant import Coordinates, Environment, State
from rheplicant.core.combinators import SumOperator
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Latent,
    ParameterSpace,
    check_linearity,
    gcr_sample,
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
            "coeffs", init=jnp.ones(3) + 0j,
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
            "coeffs", init=jnp.ones(3) + 0j,
            into=lambda p: p["sky"].coeffs, linear=True,
        )
        block = linear_operator(space, complex_twin, template_state)
        truth = jnp.array([1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 1.0j])
        observed = block.offset + block.forward(truth)
        solved, _ = wiener_solve(block, observed, noise_std=0.1, prior_std=10.0)
        expected = _dense_reference(block, observed, 0.1, 10.0)
        assert jnp.allclose(solved, expected, rtol=1e-3, atol=1e-3)

    def test_a_complex_block_recovers_a_noiseless_signal(self, template_state):
        """Both halves of the complex latent, not just the real one.

        An earlier version built the operator from a purely REAL matrix, which
        makes the imaginary half an exact null direction — it could not have
        asserted anything about it, and did not.
        """
        n_row = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        key = jax.random.key(5)
        matrix = jax.random.normal(key, (n_row, 3)) + 1j * jax.random.normal(
            jax.random.fold_in(key, 1), (n_row, 3)
        )
        complex_twin = Pipeline(
            ComplexCoeffOperator(coeffs=jnp.zeros(3, dtype=matrix.dtype), matrix=matrix),
            names=("sky",),
        )
        space = ParameterSpace.direct(
            "coeffs", init=jnp.ones(3) + 0j,
            into=lambda p: p["sky"].coeffs, linear=True,
        )
        block = linear_operator(space, complex_twin, template_state)
        truth = jnp.array([1.0 + 2.0j, -0.5 - 1.25j, 3.0 + 0.75j])
        observed = block.offset + block.forward(truth)
        solved, _ = wiener_solve(block, observed, noise_std=1e-3, prior_std=1e4)
        assert jnp.allclose(jnp.real(solved), jnp.real(truth), atol=2e-2)
        assert jnp.allclose(jnp.imag(solved), jnp.imag(truth), atol=2e-2), (
            "the imaginary half is what the R-linear/C-linear split exists for"
        )

    def test_a_strong_prior_shrinks_towards_zero(self, gain_block, gain_truth):
        observed = gain_block.offset + gain_block.forward(gain_truth)
        loose, _ = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=1e3)
        tight, _ = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=1e-3)
        assert jnp.linalg.norm(tight) < jnp.linalg.norm(loose)

    def test_the_relative_residual_is_reported(self, gain_block, gain_truth):
        observed = gain_block.offset + gain_block.forward(gain_truth)
        _, residual = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=5.0)
        assert float(residual) < 1e-4


    def test_mismatched_data_shape_is_refused(self, gain_block, template_state):
        """Broadcasting a differently-shaped observation would solve a different
        problem and return a perfectly finite answer."""
        wrong = jnp.zeros((gain_block.offset.shape[0],))
        with pytest.raises(ParameterSpaceError, match="different"):
            wiener_solve(gain_block, wrong, noise_std=1.0, prior_std=5.0)

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

    @pytest.fixture
    def saturating_space(self):
        """EXACTLY linear below a knee, grossly nonlinear above it.

        Saturation rather than a tiny quadratic, deliberately. A small
        quadratic's visibility depends on the working precision — float64 sees
        it at probes float32 cannot — so a test built on one enshrines a
        float32 artifact. A knee is exactly linear below it in ANY precision,
        and it is the more realistic failure anyway: ADC compression, amplifier
        saturation, a clipped model.
        """
        knee = 20.0 * SKY_A

        def clip(x):
            over = jnp.abs(x) > knee
            return jnp.where(over, jnp.sign(x) * knee + 0.02 * (x - jnp.sign(x) * knee), x)

        return ParameterSpace.direct(
            "amp", init=SKY_A, into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=clip, linear=True,
        )

    def test_a_block_linear_only_at_small_scale_is_caught(
        self, twin, saturating_space, template_state
    ):
        """The reason probes must span extreme scales, not just moderate ones.

        Swept over MANY keys, not one: an earlier version of this test asserted
        the default key(0) catches it, which held for only about a third of
        keys. A detection that depends on the seed is not a detection.
        """
        for seed in range(25):
            with pytest.raises(ParameterSpaceError, match="not affine"):
                check_linearity(saturating_space, twin, template_state,
                                scales=(1e2, 1e3, 1e4), key=jax.random.key(seed))

    def test_the_same_block_passes_when_only_small_scales_are_probed(
        self, twin, saturating_space, template_state
    ):
        """Companion: it is the EXTREME probe that catches it.

        Asserted as a PAIR on the same block and the same key — small probe
        passes, extreme probe raises — rather than on a numeric threshold. At
        the small probe the departure is float32 roundoff, so any threshold
        tight enough to be meaningful is also flaky.
        """
        for seed in range(25):
            key = jax.random.key(seed)
            check_linearity(saturating_space, twin, template_state,
                            scales=(1e-3, 1e-2), key=key)     # below the knee
            with pytest.raises(ParameterSpaceError, match="not affine"):
                check_linearity(saturating_space, twin, template_state,
                                scales=(1e2, 1e3, 1e4), key=key)

    def test_roundoff_at_a_tiny_probe_is_not_mistaken_for_curvature(
        self, twin, template_state
    ):
        """Regression: a linear block whose arithmetic genuinely rounds.

        The earlier version of this test used a block whose departure was
        EXACTLY 0.0 at every scale, so the relative measure never exploded and
        the absolute floor it claimed to pin never gated anything. This block
        adds and subtracts a large constant, so the departure is real roundoff
        — non-zero, and below the floor — which is what the guard is for.
        """
        # `big` need not track the precision: the departure is ~eps*big and the
        # floor is ~1e4*eps*|prediction|, so eps CANCELS from the comparison and
        # any big between ~1e-2 and ~3e5 exercises the floor in both float32 and
        # float64. Verified in both.
        big = 1e4
        space = ParameterSpace.direct(
            "amp", init=SKY_A, into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=lambda x: (x + big) - big, linear=True,
        )
        errors = check_linearity(space, twin, template_state,
                                 scales=(1e-9, 1e-7, 1e-5))
        # The point is that the RELATIVE measure alone would have rejected this
        # perfectly linear block — the floor is what saves it. Asserted against
        # the same rtol check_linearity uses, so it holds in both precisions
        # (the float32 cancellation is total and the float64 one is mild, so an
        # absolute threshold cannot serve both).
        rtol = 1e4 * float(jnp.finfo(jnp.zeros(()).dtype).eps)
        assert max(errors.values()) > 10 * rtol, (
            f"probe too benign to exercise the floor: {errors} vs rtol {rtol:.1e}"
        )

    def test_a_small_prediction_is_still_checked(self, template_state):
        """Regression: the floor must not be an absolute constant.

        With `noise_floor = 1e4*eps*max(|baseline|, 1.0)` the `1.0` clamp made
        the floor absolute, so ANY block whose prediction is small in the
        pipeline's units passed no matter how nonlinear it was.
        """
        tiny = Pipeline(
            SumOperator(
                SkyOperator(amplitude=jnp.array(1e-6)),
                SkyOperator(amplitude=jnp.array(2e-7)),
                names=("sky_a", "sky_b"),
            ),
            GainOperator(gain=jnp.array(1.0)),
            names=("sum", "gain"),
        )
        space = ParameterSpace.direct(
            "amp", init=1e-6, into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=lambda x: x + 1e6 * x**2, linear=True,   # wildly nonlinear
        )
        with pytest.raises(ParameterSpaceError, match="not affine"):
            check_linearity(space, tiny, template_state)

    def test_a_bright_unrelated_component_does_not_disable_the_check(
        self, template_state
    ):
        """The floor must not be set by the BASELINE alone: the baseline is what
        the other latents contribute, so scaling it up would otherwise exempt
        a nonlinear latent that contributes a small fraction of the signal."""
        space = ParameterSpace.direct(
            "amp", init=1.0, into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=lambda x: x + 0.05 * x**2, linear=True,
        )
        for bright in (1.0, 1e3, 1e6):
            loud = Pipeline(
                SumOperator(
                    SkyOperator(amplitude=jnp.array(1.0)),
                    SkyOperator(amplitude=jnp.array(bright)),
                    names=("sky_a", "sky_b"),
                ),
                GainOperator(gain=jnp.array(1.0)),
                names=("sum", "gain"),
            )
            with pytest.raises(ParameterSpaceError, match="not affine"):
                check_linearity(space, loud, template_state)

    def test_a_nan_probe_counts_as_a_failure(self, twin, template_state):
        """`nan > rtol` is False, so a naive filter reads an unusable probe as
        evidence OF linearity. It must read as failure instead."""
        space = ParameterSpace.direct(
            "amp", init=SKY_A, into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=lambda x: x + jnp.where(jnp.abs(x) > 1e4, jnp.nan, 0.0), linear=True,
        )
        with pytest.raises(ParameterSpaceError, match="not affine"):
            check_linearity(space, twin, template_state)

    def test_probes_are_absolute_when_init_is_zero(self, twin, template_state):
        """The documented sharp edge: max|init| = 0 has no scale to take, so the
        probes fall back to absolute. Pinned because the docs' own examples use
        init=zeros."""
        space = ParameterSpace.direct(
            "amp", init=0.0, into=lambda p: p["sum"]["sky_a"].amplitude,
            linear=True,
        )
        errors = check_linearity(space, twin, template_state)
        assert set(errors) == set((1e-3, 1.0, 1e3))

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


def _dense_posterior(block, noise_std, prior_std):
    """(mean-operator, covariance) built explicitly, for checking the sampler.

    The posterior of a linear-Gaussian block is N(C A^T N^-1 r, C) with
    C = (A^T N^-1 A + S^-1)^-1. Both are formed densely here so the sampler's
    FIRST TWO MOMENTS can be checked, not just its mean — a sampler that gets
    the mean right and the covariance wrong looks perfectly healthy otherwise.
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
    normal = A.T @ (weight * A) + jnp.eye(A.shape[1]) / jnp.asarray(prior_std) ** 2
    return A, weight, jnp.linalg.inv(normal)


class TestGCRSample:
    """A constrained realization is an EXACT posterior draw, so both moments check."""

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
    def observed(self, gain_block, template_state):
        n_time = template_state.coords.time.shape[0]
        truth = GAIN + 0.1 * jnp.arange(n_time, dtype=float)
        return gain_block.offset + gain_block.forward(truth)

    def test_a_sample_is_not_the_mean(self, gain_block, observed):
        mean, _ = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=5.0)
        draw, _ = gcr_sample(gain_block, observed, noise_std=1.0, prior_std=5.0,
                             key=jax.random.key(0))
        assert not jnp.allclose(draw, mean, atol=1e-3)

    def test_the_draw_is_deterministic_given_a_key(self, gain_block, observed):
        kwargs = dict(noise_std=1.0, prior_std=5.0, key=jax.random.key(3))
        first, _ = gcr_sample(gain_block, observed, **kwargs)
        second, _ = gcr_sample(gain_block, observed, **kwargs)
        assert jnp.allclose(first, second)

    def test_different_keys_give_different_draws(self, gain_block, observed):
        a, _ = gcr_sample(gain_block, observed, noise_std=1.0, prior_std=5.0,
                          key=jax.random.key(0))
        b, _ = gcr_sample(gain_block, observed, noise_std=1.0, prior_std=5.0,
                          key=jax.random.key(1))
        assert not jnp.allclose(a, b)

    def test_the_sample_mean_converges_to_the_wiener_mean(self, gain_block, observed):
        mean, _ = wiener_solve(gain_block, observed, noise_std=1.0, prior_std=5.0)
        keys = jax.random.split(jax.random.key(0), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(gain_block, observed, noise_std=1.0, prior_std=5.0,
                                 key=k)[0]
        )(keys)
        _, _, cov = _dense_posterior(gain_block, 1.0, 5.0)
        # standard error of the mean, per component
        sem = jnp.sqrt(jnp.diag(cov) / keys.shape[0])
        assert jnp.all(jnp.abs(draws.mean(axis=0) - mean) < 4.0 * sem)

    def test_the_sample_covariance_matches_the_posterior_covariance(
        self, gain_block, observed
    ):
        """The test that distinguishes a real constrained realization from a
        mean-plus-arbitrary-noise: the second moment has to be right too."""
        _, _, cov = _dense_posterior(gain_block, 1.0, 5.0)
        keys = jax.random.split(jax.random.key(7), 6000)
        draws = jax.vmap(
            lambda k: gcr_sample(gain_block, observed, noise_std=1.0, prior_std=5.0,
                                 key=k)[0]
        )(keys)
        empirical = jnp.cov(draws.T)
        relative = jnp.linalg.norm(empirical - cov) / jnp.linalg.norm(cov)
        assert float(relative) < 0.12, f"covariance off by {float(relative):.3f}"

    def test_a_vanishing_prior_pins_the_draw_at_zero(self, gain_block, observed):
        draw, _ = gcr_sample(gain_block, observed, noise_std=1.0, prior_std=1e-6,
                             key=jax.random.key(0))
        assert jnp.allclose(draw, 0.0, atol=1e-4)

    def test_with_no_data_the_draw_follows_the_PRIOR(self, gain_block):
        """Given data that carries no information (infinite noise), the posterior
        IS the prior — the cleanest check that the fluctuation term is scaled right."""
        keys = jax.random.split(jax.random.key(11), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(gain_block, gain_block.offset, noise_std=1e8,
                                 prior_std=2.0, key=k)[0]
        )(keys)
        assert float(draws.std()) == pytest.approx(2.0, rel=0.05)
        assert abs(float(draws.mean())) < 0.15

    def test_a_complex_block_draws_correctly(self, template_state):
        n_row = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        key = jax.random.key(3)
        matrix = jax.random.normal(key, (n_row, 3)) + 1j * jax.random.normal(
            jax.random.fold_in(key, 1), (n_row, 3)
        )
        twin = Pipeline(
            ComplexCoeffOperator(coeffs=jnp.zeros(3, dtype=matrix.dtype), matrix=matrix),
            names=("sky",),
        )
        space = ParameterSpace.direct(
            "coeffs", init=jnp.ones(3) + 0j,
            into=lambda p: p["sky"].coeffs, linear=True,
        )
        block = linear_operator(space, twin, template_state)
        observed = block.offset + block.forward(jnp.array([1.0 + 2j, -0.5 + 0.25j, 3.0 - 1j]))
        mean, _ = wiener_solve(block, observed, noise_std=0.5, prior_std=4.0)
        keys = jax.random.split(jax.random.key(5), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(block, observed, noise_std=0.5, prior_std=4.0, key=k)[0]
        )(keys)
        assert draws.dtype == block.dtype
        _, _, cov = _dense_posterior(block, 0.5, 4.0)
        sem = jnp.sqrt(jnp.diag(cov)[0::2] / keys.shape[0])
        assert jnp.all(jnp.abs(jnp.real(draws.mean(axis=0) - mean)) < 4.0 * sem)

    def test_a_prior_is_required(self, gain_block, observed):
        with pytest.raises(ParameterSpaceError, match="prior_std"):
            gcr_sample(gain_block, observed, noise_std=1.0, prior_std=None,
                       key=jax.random.key(0))


class DesignMatrixOperator(AbstractOperator):
    """A real observation through an explicit design matrix -- test double.

    Exists to construct arbitrarily ill-conditioned normal operators
    directly, with no external calibration package needed: no physics is
    asserted here, only ``data = matrix @ coeffs``.
    """

    provides: ClassVar[tuple[str, ...]] = ("data",)

    coeffs: jax.Array
    matrix: jax.Array

    def __call__(self, state):
        return state.with_data(self.matrix @ self.coeffs)


class TestGCRSampleIllConditionedConvergence:
    """``gcr_sample``'s CG tolerance has to bound the SOLUTION error, not just
    the residual (see the ``tol``/``require_convergence`` docstrings in
    ``linear.py``). Every other test in this file uses a well-conditioned
    block, which cannot exercise this at all: point-by-point agreement with a
    dense reference says nothing about what happens at a dispatch/convergence
    boundary (boundary-validation.md), and CG on a well-conditioned normal
    operator converges to the same answer at any reasonable ``tol``.

    This block is deliberately as ill-conditioned as the noise-wave
    calibration case that surfaced the bug (``cond(M) ~ 4e8``): one equation
    per channel, overwhelmingly sensitive to the FIRST of three unknowns and
    only residually (``1e-3``, vs. the ``100`` of the sensitive one) coupled
    to the other two. That residual coupling is not incidental: an EXACTLY
    diagonal design (no coupling at all) does NOT reproduce this bug, because
    then CG solves every coordinate independently and the aggregate residual
    can no longer hide a large per-coordinate error in a poorly-constrained
    direction behind a tiny error in a well-constrained one. A real physical
    design matrix (a reflection coefficient touching all three noise-wave
    terms in one equation) is never exactly diagonal, so the mixed case here
    is the representative one, not a special one.
    """

    N_CHANNEL = 16
    NOISE_STD = 0.5
    PRIOR_STD = 100.0
    # One equation per channel: overwhelmingly sensitive to the first unknown,
    # only residually coupled (1e-3 out of 100) to the other two.
    ROW = (100.0, 1e-3, 1e-3)
    TRUE = (250.0, 20.0, -10.0)
    N_DRAWS = 1000

    @pytest.fixture
    def block(self):
        n_param = 3 * self.N_CHANNEL
        row = jnp.array(self.ROW)
        matrix = jnp.zeros((self.N_CHANNEL, n_param))
        for c in range(self.N_CHANNEL):
            matrix = matrix.at[c, 3 * c: 3 * c + 3].set(row)
        state = State(
            coords=Coordinates(time=jnp.arange(1.0), freq=jnp.arange(float(self.N_CHANNEL))),
            env=Environment(temperature=jnp.array(280.0)),
            key=jax.random.key(0),
            meta={"telescope": "test"},
        )
        twin = Pipeline(
            DesignMatrixOperator(coeffs=jnp.zeros(n_param), matrix=matrix),
            names=("design",),
        )
        space = ParameterSpace.direct(
            "coeffs", init=jnp.zeros(n_param),
            into=lambda p: p["design"].coeffs, linear=True,
        )
        return linear_operator(space, twin, state)

    @pytest.fixture
    def observed(self, block):
        # A real (nonzero) signal matters: it is what puts a large component
        # along the well-constrained direction into the right-hand side,
        # which is what lets the aggregate residual norm be dominated by it.
        # observed == block.offset (no signal) does not reproduce the bug.
        true_coeffs = jnp.tile(jnp.array(self.TRUE), self.N_CHANNEL)
        truth = block.offset + block.forward(true_coeffs)
        return truth + self.NOISE_STD * jax.random.normal(jax.random.key(1), truth.shape)

    def test_the_construction_is_genuinely_ill_conditioned(self, block):
        """Pin the fixture's own condition number, so a future edit to the
        constants above cannot silently defang this regression test."""
        _, _, cov = _dense_posterior(block, self.NOISE_STD, self.PRIOR_STD)
        eigvals = jnp.linalg.eigvalsh(jnp.linalg.inv(cov))
        cond = float(eigvals[-1] / eigvals[0])
        assert cond > 1e7, f"fixture is not ill-conditioned enough: cond={cond:.2e}"

    def test_the_unconstrained_directions_are_prior_dominated_in_closed_form(self, block):
        """Two of every three parameters are, for all practical purposes,
        unmeasured (their design-matrix entry is 1e-3 against a row whose
        other entry is 100): the closed-form posterior sigma for them is
        PRIOR_STD, not merely "close to it". Pinning this analytically is
        what makes the draw-based assertions below a real regression test
        rather than a guess at what the "right" number should be."""
        _, _, cov = _dense_posterior(block, self.NOISE_STD, self.PRIOR_STD)
        dense_sigma = jnp.sqrt(jnp.diag(cov)).reshape(self.N_CHANNEL, 3)
        assert jnp.allclose(dense_sigma[:, 1:], self.PRIOR_STD, rtol=1e-4)

    def test_default_tol_recovers_the_prior_in_the_unconstrained_directions(
        self, block, observed
    ):
        """THE regression pin. Library defaults only: no ``tol``/``maxiter``
        passed here at all."""
        keys = jax.random.split(jax.random.key(2), self.N_DRAWS)
        draws, residual = jax.vmap(
            lambda k: gcr_sample(block, observed, noise_std=self.NOISE_STD,
                                 prior_std=self.PRIOR_STD, key=k)
        )(keys)
        sigma = draws.std(axis=0).reshape(self.N_CHANNEL, 3)
        null_sigma = sigma[:, 1:]
        # The standard error of an empirical std from N draws is
        # sigma/sqrt(2N); with N_DRAWS=1000 that is ~2.2% relative, so 10%
        # (~4.5 SE) is generous against MC noise while still ~1000x tighter
        # than the collapse this guards against (see the next test).
        rel_err = jnp.abs(null_sigma - self.PRIOR_STD) / self.PRIOR_STD
        assert jnp.all(jnp.isfinite(residual))
        assert float(jnp.max(rel_err)) < 0.10, (
            "gcr_sample under-reported the posterior width in an "
            f"unconstrained direction: {float(jnp.max(rel_err)):.3f} relative error"
        )

    def test_the_old_default_tol_collapses_the_same_sigma(self, block, observed):
        """Proof the test above has teeth. At the library's PRE-FIX default
        (``tol=1e-6``), ``gcr_sample``'s own ``require_convergence`` guard
        (also left at ITS default, ``1e-3``) does NOT catch this: the
        aggregate relative residual stays far under ``1e-3`` because it is
        dominated by the one well-constrained direction, exactly as described
        in ``linear.py``. Yet the reported sigma in the unconstrained
        directions collapses towards zero instead of the 100 K prior --
        understating the posterior width by orders of magnitude, always in
        the direction of false confidence, and without raising anything."""
        keys = jax.random.split(jax.random.key(2), self.N_DRAWS)
        draws, residual = jax.vmap(
            lambda k: gcr_sample(block, observed, noise_std=self.NOISE_STD,
                                 prior_std=self.PRIOR_STD, key=k, tol=1e-6)
        )(keys)
        assert jnp.all(residual < 1e-3), (
            "the require_convergence guard fired here, which would mean this "
            "no longer demonstrates the silent-failure mode the test exists for"
        )
        sigma = draws.std(axis=0).reshape(self.N_CHANNEL, 3)
        null_sigma = sigma[:, 1:]
        assert float(jnp.mean(null_sigma)) < 0.01 * self.PRIOR_STD, (
            "expected the OLD tol=1e-6 default to collapse the unconstrained "
            f"sigma near zero; got {float(jnp.mean(null_sigma)):.4f} K "
            f"({100 * float(jnp.mean(null_sigma)) / self.PRIOR_STD:.2f}% of prior) -- "
            "if this now passes, either the bug was fixed differently or this "
            "fixture stopped being ill-conditioned enough to show it"
        )


class SpectralTiltOperator(AbstractOperator):
    """Multiply by (nu/nu0)^(-alpha) — a spectral SHAPE, not a scale.

    Test double. It exists because a linear amplitude and a multiplicative gain
    are exactly degenerate, so a Gibbs test built on those two would wander
    along the degeneracy and land nowhere in particular. A tilt separates
    amplitude from shape, which is also the real situation (foreground
    amplitude vs spectral index).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    alpha: jax.Array

    def __call__(self, state):
        nu = state.coords.freq
        return state.with_data(state.data * (nu / nu[0]) ** (-self.alpha))


class TestConditioningOnOtherLatents:
    """A linear block is only linear GIVEN the other parameters, so it has to be
    buildable at their current values — the operation a Gibbs sweep is made of."""

    @pytest.fixture
    def tilted(self):
        return Pipeline(
            SumOperator(
                SkyOperator(amplitude=jnp.array(SKY_A)),
                SkyOperator(amplitude=jnp.array(SKY_B)),
                names=("sky_a", "sky_b"),
            ),
            SpectralTiltOperator(alpha=jnp.array(0.0)),
            names=("sum", "tilt"),
        )

    @pytest.fixture
    def mixed_space(self):
        return ParameterSpace(
            latents=[
                Latent("amp", init=SKY_A, linear=True),
                Latent("alpha", init=0.0),
            ],
            bindings=[
                Bind("amp", into=lambda p: p["sum"]["sky_a"].amplitude),
                Bind("alpha", into=lambda p: p["tilt"].alpha),
            ],
        )

    def test_default_uses_the_declared_initial_values(
        self, tilted, mixed_space, template_state
    ):
        block = linear_operator(mixed_space, tilted, template_state, "amp")
        # alpha init = 0 -> no tilt, so offset is sky_b flat across frequency
        assert jnp.allclose(block.offset, SKY_B)

    def test_at_rebuilds_the_block_elsewhere(self, tilted, mixed_space, template_state):
        nu = template_state.coords.freq
        block = linear_operator(mixed_space, tilted, template_state, "amp",
                                at={"alpha": jnp.array(2.0)})
        expected = SKY_B * (nu / nu[0]) ** (-2.0)
        assert jnp.allclose(block.offset, jnp.broadcast_to(expected, block.offset.shape))

    def test_at_may_be_partial(self, tilted, mixed_space, template_state):
        """Latents absent from `at` keep their declared init."""
        block = linear_operator(mixed_space, tilted, template_state, "amp", at={})
        assert jnp.allclose(block.offset, SKY_B)

    def test_at_rejects_an_unknown_name(self, tilted, mixed_space, template_state):
        with pytest.raises(ParameterSpaceError, match="not a latent"):
            linear_operator(mixed_space, tilted, template_state, "amp",
                            at={"nope": jnp.array(1.0)})

    def test_check_linearity_honours_at(self, tilted, mixed_space, template_state):
        errors = check_linearity(mixed_space, tilted, template_state, "amp",
                                 at={"alpha": jnp.array(2.0)})
        assert all(err < 1e-4 for err in errors.values())

    def test_a_gibbs_sweep_recovers_both_blocks(self, tilted, mixed_space, template_state):
        """The pattern this exists for: alternate an exact conjugate draw of the
        linear block with a conditional update of the nonlinear one."""
        true_amp, true_alpha = 60.0, 1.8
        truth = eqx.tree_at(
            lambda p: (p["sum"]["sky_a"].amplitude, p["tilt"].alpha),
            tilted, (jnp.array(true_amp), jnp.array(true_alpha)),
        )
        observed = truth(template_state).data
        forward, _ = mixed_space.forward_fn(tilted, template_state)

        # Check the linearity claim ONCE, outside the loop.
        check_linearity(mixed_space, tilted, template_state, "amp")

        values = {"amp": jnp.array(1.0), "alpha": jnp.array(0.0)}
        grid = jnp.linspace(-1.0, 4.0, 600)
        key = jax.random.key(0)
        for _ in range(25):
            key, draw_key = jax.random.split(key)
            block = linear_operator(mixed_space, tilted, template_state, "amp",
                                    at=values, check=False)
            amp, _ = gcr_sample(block, observed, noise_std=0.02, prior_std=500.0,
                                key=draw_key)
            values = {**values, "amp": amp}
            chi2 = jax.vmap(
                lambda a, current=values: jnp.sum(
                    (forward({**current, "alpha": a}) - observed) ** 2
                )
            )(grid)
            values = {**values, "alpha": grid[jnp.argmin(chi2)]}

        assert float(values["alpha"]) == pytest.approx(true_alpha, abs=0.02)
        assert float(values["amp"]) == pytest.approx(true_amp, rel=0.02)


class TestPriorMean:
    """A zero-mean prior is wrong for most physical quantities: a noise-wave
    temperature sits near 250 K, not near zero. The prior mean belongs on the
    prior, not smuggled into the model through an affine binding."""

    @pytest.fixture
    def gain_block(self, twin, template_state):
        n_time = template_state.coords.time.shape[0]
        wide = eqx.tree_at(lambda p: p["gain"].gain, twin, jnp.full((n_time,), GAIN))
        space = ParameterSpace.direct(
            "gains", init=jnp.full((n_time,), GAIN),
            into=lambda p: p["gain"].gain, linear=True,
        )
        return linear_operator(space, wide, template_state)

    def test_with_no_data_the_mean_is_the_prior_mean(self, gain_block):
        mean, _ = wiener_solve(gain_block, gain_block.offset, noise_std=1e8,
                               prior_std=1.0, prior_mean=3.0)
        assert jnp.allclose(mean, 3.0, rtol=1e-3)

    def test_with_no_data_draws_centre_on_the_prior_mean(self, gain_block):
        keys = jax.random.split(jax.random.key(2), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(gain_block, gain_block.offset, noise_std=1e8,
                                 prior_std=2.0, prior_mean=3.0, key=k)[0]
        )(keys)
        assert float(draws.mean()) == pytest.approx(3.0, abs=0.15)
        assert float(draws.std()) == pytest.approx(2.0, rel=0.05)

    def test_it_matches_the_affine_binding_workaround(self, twin, template_state):
        """Shifting the PRIOR and shifting the MODEL are the same Gaussian, so
        the two routes must agree — pinned so they cannot drift apart."""
        n_time = template_state.coords.time.shape[0]
        offset_value = 1.2
        wide = eqx.tree_at(lambda p: p["gain"].gain, twin, jnp.full((n_time,), GAIN))
        observed = wide(template_state).data

        shifted_prior = linear_operator(
            ParameterSpace.direct("gains", init=jnp.full((n_time,), GAIN),
                                  into=lambda p: p["gain"].gain, linear=True),
            wide, template_state,
        )
        direct, _ = wiener_solve(shifted_prior, observed, noise_std=0.5,
                                 prior_std=0.3, prior_mean=offset_value)

        shifted_model = linear_operator(
            ParameterSpace.direct("delta", init=jnp.zeros(n_time),
                                  into=lambda p: p["gain"].gain,
                                  fn=lambda d: offset_value + d, linear=True),
            wide, template_state,
        )
        via_fn, _ = wiener_solve(shifted_model, observed, noise_std=0.5, prior_std=0.3)
        assert jnp.allclose(direct, offset_value + via_fn, rtol=1e-4, atol=1e-5)

    def test_a_scalar_prior_mean_broadcasts(self, gain_block):
        mean, _ = wiener_solve(gain_block, gain_block.offset, noise_std=1e8,
                               prior_std=1.0, prior_mean=jnp.zeros(()) + 2.0)
        assert mean.shape == gain_block.shape
        assert jnp.allclose(mean, 2.0, rtol=1e-3)

    def test_default_is_still_zero_mean(self, gain_block):
        mean, _ = wiener_solve(gain_block, gain_block.offset, noise_std=1e8,
                               prior_std=1.0)
        assert jnp.allclose(mean, 0.0, atol=1e-6)
