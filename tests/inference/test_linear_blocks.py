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

from rheplicant.core.combinators import SumOperator
from rheplicant.core.errors import LinearityRefused, ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Latent,
    LinearBlock,
    ParameterSpace,
    check_linearity,
    condition_bound,
    condition_estimate,
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

    @pytest.fixture
    def exponential_space(self):
        """Nonlinear at every probe -- the simplest thing that refuses."""
        return ParameterSpace.direct(
            "log_amp", init=jnp.log(SKY_A), into=lambda p: p["sum"]["sky_a"].amplitude,
            fn=jnp.exp, linear=True,
        )

    def test_the_refusal_is_a_parameter_space_error_and_stays_one(
        self, twin, exponential_space, template_state
    ):
        """The entire back-compatibility claim, in one place.

        The refusal carries numbers now (below), and it does that by being a
        SUBCLASS with an unchanged message -- so every
        ``pytest.raises(ParameterSpaceError, match=...)`` in this suite and
        every ``except ParameterSpaceError`` outside it keeps working, and
        neither had to be touched.  Both halves are asserted here because
        either one alone can be lost by a later edit: re-parenting the class
        breaks the first, rewording the sentence breaks the second, and
        nothing else in this file would notice.
        """
        with pytest.raises(ParameterSpaceError, match="not affine") as caught:
            check_linearity(exponential_space, twin, template_state)
        assert type(caught.value) is LinearityRefused

    def test_the_refusal_carries_the_numbers_its_sentence_only_renders(
        self, twin, exponential_space, template_state
    ):
        """The reason this class exists.

        The passing branch RETURNS ``{scale: error}``; the failing branch used
        to format the same measurement into prose and drop it, so the outcome
        with something to report was the only one a caller could not read.
        Anything downstream wanting the number had to parse the sentence.

        The loop is the load-bearing assertion: it pins that the attribute and
        the sentence are the SAME measurement rendered twice, so the two
        cannot drift into disagreeing about what was measured.
        """
        with pytest.raises(LinearityRefused) as caught:
            check_linearity(exponential_space, twin, template_state)
        refused = caught.value

        assert set(refused.errors) == {1e-3, 1.0, 1e3}
        assert refused.failed
        assert set(refused.failed) <= set(refused.errors)
        for scale, error in refused.errors.items():
            assert f"{scale:g}x -> {error:.2e}" in str(refused)
        assert f"rtol={refused.rtol:.2e}" in str(refused)

    def test_the_carried_numbers_keep_the_TREND_a_single_worst_case_loses(
        self, twin, saturating_space, template_state
    ):
        """Why the whole table is carried, and not a maximum.

        Measured on this block at scales ``(1e-3, 1e2, 1e4)``, seeds 0-5, all
        identical: ``failed == (100.0, 10000.0)`` with the small probe clean at
        ~1e-5.  "Affine until the probe reaches the knee" and "not affine
        anywhere" are different faults with different fixes, and a scalar
        summary cannot tell them apart -- so the summary is not what is
        carried.
        """
        for seed in range(6):
            with pytest.raises(LinearityRefused) as caught:
                check_linearity(saturating_space, twin, template_state,
                                scales=(1e-3, 1e2, 1e4), key=jax.random.key(seed))
            refused = caught.value
            assert 1e-3 not in refused.failed
            assert set(refused.failed) == {1e2, 1e4}
            assert refused.errors[1e-3] < refused.errors[1e4]

    def test_the_refusal_is_importable_from_the_layer_that_raises_it(self):
        """A caller who wants the numbers must be able to name the class.

        ``rheplicant.core`` deliberately does not carry ``ParameterSpaceError``
        -- a parameter space is not a concept ``core`` has -- so
        ``rheplicant.inference`` is where a ``check_linearity`` caller already
        imports from, and an exception reachable only through
        ``rheplicant.core.errors`` would be the one name in this family that
        contradicts that.
        """
        import rheplicant.inference as inference

        assert inference.LinearityRefused is LinearityRefused
        assert "LinearityRefused" in inference.__all__

    def test_the_refusal_cannot_be_raised_without_its_numbers(self):
        """Required keyword arguments, and NOT ``ConfigError.report``'s
        optional payload.

        A ``LinearityRefused`` carrying no measurement is precisely the state
        this class exists to abolish -- the message-only refusal it replaced --
        and a default would let one back in without anything going red.
        """
        with pytest.raises(TypeError):
            LinearityRefused("Latent 'g' ... is not affine in it")

    def test_the_refusal_copies_the_table_it_is_handed(self):
        """A caught exception must not be a live handle on its raiser's state.

        One line of production code (``dict(errors)``), and nothing else in
        the suite would notice it being dropped: ``check_linearity`` hands over
        a dict it is finished with, so the aliasing is invisible until some
        caller sorts or prunes ``errors`` in place.
        """
        mine = {1.0: 0.5}
        refused = LinearityRefused("x", errors=mine, rtol=1e-3, failed=(1.0,))
        mine[1.0] = 0.0
        assert refused.errors == {1.0: 0.5}


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


class OneLoadOperator(AbstractOperator):
    """Three unknowns, of which the data sees only ONE combination.

    The geometry of a single calibration load against ``(T_unc, T_cos,
    T_sin)``: one direction in ℝ³ is pinned by the data and the other two are
    left entirely to the prior. Test double, not physics — it exists to put
    the normal operator ``AᵀN⁻¹A + S⁻¹`` in the deliberately near-singular
    regime that these solvers are advertised to handle.
    """

    provides: ClassVar[tuple[str, ...]] = ("data",)

    coeffs: jax.Array
    direction: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        return state.with_data(
            jnp.full((n_time, n_freq), jnp.sum(self.direction * self.coeffs))
        )


# The one combination the data constrains, and one it is blind to.
SEEN = jnp.array([0.6, 0.8, 0.0])
BLIND = jnp.array([-0.8, 0.6, 0.0])
LOAD_NOISE, LOAD_PRIOR = 0.1, 100.0


class TestUnderDeterminedBlocks:
    """A block the data does not fully identify — the case the prior is for.

    ``AᵀN⁻¹A`` is rank one here, so ``λ_min(M)`` is exactly ``1/prior_std²``
    while ``λ_max(M) ≈ ‖AᵀN⁻¹A‖``: the normal operator's condition number is
    ~3e7 BY DESIGN, not by accident. That matters because CG's stopping rule
    and the ``require_convergence`` guard both measure a relative RESIDUAL,
    and residual and error are related by ``rel_err ≤ κ · rel_residual``. At
    κ=3e7 a residual of 1e-6 certifies nothing whatsoever.

    The failure this pins is silent: the returned draw has a tiny residual, no
    warning, and posterior scatter in the blind directions that is orders of
    magnitude too small — a confidently under-reported uncertainty, which is
    the worst thing a sampler can hand back.
    """

    @pytest.fixture
    def one_load_block(self, template_state):
        pipeline = Pipeline(
            OneLoadOperator(coeffs=jnp.zeros(3), direction=SEEN), names=("load",)
        )
        space = ParameterSpace.direct(
            "coeffs", init=jnp.array([250.0, 5.0, -3.0]),
            into=lambda p: p["load"].coeffs, linear=True,
        )
        return linear_operator(space, pipeline, template_state)

    @pytest.fixture
    def one_load_observed(self, one_load_block):
        truth = jnp.array([250.0, 5.0, -3.0])
        return one_load_block.offset + one_load_block.forward(truth)

    def test_the_default_tolerance_is_refused_rather_than_trusted(
        self, one_load_block, one_load_observed
    ):
        """The reported bug. CG stops at a relative residual of ~1e-7 with the
        blind directions still unresolved; the guard has to notice that a
        residual that small means nothing when κ is this large.

        ``require_convergence`` is passed rather than defaulted: it stopped
        being on by default when κ became a bound. The message says "condition
        bound" rather than "condition number" because that is what the guard
        reads -- :func:`condition_bound`, not :func:`condition_estimate` -- and
        since the switch the wording comes from
        ``bayesmith.exact.solve._conjugate_solve``, which spells the
        distinction the near side only documented."""
        with pytest.raises(RuntimeError, match="condition bound"):
            gcr_sample(one_load_block, one_load_observed, noise_std=LOAD_NOISE,
                       prior_std=LOAD_PRIOR, key=jax.random.key(0),
                       require_convergence=1e-3)

    def test_the_mean_is_refused_once_the_prior_has_something_to_say(
        self, one_load_block, one_load_observed
    ):
        """Not a sampling-only fault — but it takes a nonzero prior mean to show.

        With the default zero-centred prior the right-hand side has NO
        component along the blind directions (``AᵀN⁻¹(d-offset)`` lies in the
        row space, and ``S⁻¹·0`` vanishes), so the answer there is exactly
        zero and CG's untouched zero is exactly right. Give the prior a centre
        — a noise-wave temperature near 250 K, the case ``prior_mean`` exists
        for — and the blind directions must come back at 250; the unguarded
        solve returns ~1e-5 instead.
        """
        with pytest.raises(RuntimeError, match="condition bound"):
            wiener_solve(one_load_block, one_load_observed, noise_std=LOAD_NOISE,
                         prior_std=LOAD_PRIOR, prior_mean=250.0,
                         require_convergence=1e-3)

    def test_the_zero_centred_mean_is_not_refused(
        self, one_load_block, one_load_observed
    ):
        """The other half of that: κ being large is not itself a failure.

        The guard bounds the error by ``κ · residual``, and here the residual
        is exactly zero because there is nothing in the blind directions to
        resolve. A guard that fired on κ alone would reject this correct
        answer.
        """
        mean, _ = wiener_solve(one_load_block, one_load_observed,
                               noise_std=LOAD_NOISE, prior_std=LOAD_PRIOR)
        assert float(mean @ BLIND) == pytest.approx(0.0, abs=1e-3)

    def test_float32_is_refused_however_tight_the_tolerance(
        self, one_load_block, one_load_observed
    ):
        """κ·eps ≈ 4 at single precision, so NO tolerance can make this solve
        accurate. Saying so is the point: the honest answer is a refusal, not
        four thousand iterations that end up equally wrong.

        **Asserted over a spread of keys, not one, and that is a correction.**
        This used to pin ``key(0)`` and read as deterministic. It never was:
        measured across 20 keys, the draw is refused for 15 of them and
        accepted for 5 — because whether ``residual × κ`` clears the target
        depends on the fluctuation drawn, and a draw that lands accurately
        SHOULD be accepted (``test_the_zero_centred_mean_is_not_refused`` is
        that counterweight). Pinning one key made a key-dependent outcome look
        like a property, and the Wave B switch moved which keys fall where
        (12 of 20 after) without changing anything this test is about.

        What is key-independent is the floor itself and the KIND of refusal:
        every refusal here names the precision, never "did not converge",
        because tightening ``tol`` is the wrong advice when the arithmetic
        cannot represent the answer. Both halves hold on either side of the
        seam.
        """
        if jax.config.read("jax_enable_x64"):
            pytest.skip("this is the single-precision floor")

        bound = float(condition_bound(one_load_block, noise_std=LOAD_NOISE,
                                      prior_std=LOAD_PRIOR))
        epsilon = float(jnp.finfo(jnp.float32).eps)
        assert bound * epsilon > 1e-3, (bound, epsilon)

        refusals = []
        for seed in range(20):
            try:
                gcr_sample(one_load_block, one_load_observed,
                           noise_std=LOAD_NOISE, prior_std=LOAD_PRIOR,
                           key=jax.random.key(seed), tol=1e-12, maxiter=5000,
                           require_convergence=1e-3)
            except RuntimeError as refused:
                refusals.append(str(refused))

        assert refusals, (
            "no key was refused at all, so this test can no longer fail for "
            "the reason it exists; κ·eps is "
            f"{bound * epsilon:.2f} against a target of 1e-3"
        )
        assert all("at this precision" in text for text in refusals), (
            "a refusal here advised tightening tol, which cannot help below "
            "the precision floor"
        )

    def test_a_well_conditioned_block_is_not_refused(self, twin, template_state):
        """The default must not fire on healthy solves, and it does not.

        **This docstring used to argue the opposite of what shipped, and both
        halves of the argument were right.** It said: a bound that assumed the
        worst about ``λ_min`` -- all it is entitled to assume without measuring
        is ``λ_min ≥ 1/prior_std²`` -- would report κ~5e7 for THIS block, whose
        true κ is ~1, and would reject every solve in the suite, so the
        conditioning has to be MEASURED. That is correct and it is measured
        here: the bound reads 1.44e+06 against a true κ under 10.

        What it could not know is that the measurement is unsound in the one
        direction a guard cannot afford -- biased high on ``λ_min``, so low on
        κ, so silent where it should speak (bayesmith's port measured 34x on a
        κ=1e4 spectrum and ~700x at 1e7). So κ is a bound now, and the thing
        that gave way is the DEFAULT rather than the soundness: the guard is
        opt-in, and this test is what keeps the default honest.

        The cost is real and is its own test below.
        """
        n_time = template_state.coords.time.shape[0]
        wide = eqx.tree_at(lambda p: p["gain"].gain, twin, jnp.full((n_time,), GAIN))
        block = linear_operator(
            ParameterSpace.direct("gains", init=jnp.full((n_time,), GAIN),
                                  into=lambda p: p["gain"].gain, linear=True),
            wide, template_state,
        )
        observed = block.offset + block.forward(jnp.full((n_time,), GAIN))
        drawn, _ = gcr_sample(block, observed, noise_std=1.0, prior_std=5.0,
                              key=jax.random.key(0))
        assert jnp.all(jnp.isfinite(drawn))

    def test_the_condition_estimate_matches_a_dense_eigenvalue_computation(
        self, one_load_block
    ):
        """The measured κ, checked against LAPACK.

        NOT the number a caller picks a tolerance from -- that is
        ``condition_bound``, which is what ``require_convergence`` reads. This
        one is public because it can SEE a degeneracy the bound cannot, being
        the only one of the two that measures ``λ_min`` instead of flooring
        it. ``TestTheTwoConditionNumbersDivideTheLabour`` below pins that the
        two docstrings keep saying so.
        """
        columns = [one_load_block.forward(jnp.zeros(3).at[i].set(1.0)) for i in range(3)]
        A = jnp.stack([jnp.ravel(c) for c in columns], axis=1)
        dense = A.T @ A / LOAD_NOISE**2 + jnp.eye(3) / LOAD_PRIOR**2
        expected = jnp.linalg.cond(dense)

        estimated = condition_estimate(one_load_block, noise_std=LOAD_NOISE,
                                       prior_std=LOAD_PRIOR)
        assert float(estimated) == pytest.approx(float(expected), rel=0.1)

    def _healthy_block(self, twin, template_state):
        n_time = template_state.coords.time.shape[0]
        wide = eqx.tree_at(lambda p: p["gain"].gain, twin, jnp.full((n_time,), GAIN))
        return linear_operator(
            ParameterSpace.direct("gains", init=jnp.full((n_time,), GAIN),
                                  into=lambda p: p["gain"].gain, linear=True),
            wide, template_state,
        )

    def test_the_bound_is_loose_by_five_decades_on_a_healthy_block(
        self, twin, template_state
    ):
        """The price of soundness, as a number rather than a caveat.

        This asserted ``< 10.0`` while κ was measured, and the measurement was
        right about THIS block: the data constrains every direction, so the
        true λ_min is set by the data and is five decades above the prior
        floor the bound is entitled to assume. The bound cannot see that and
        must not pretend to.

        Pinned as a range rather than a value: the point is the DECADES, and a
        tighter pin would break on any retuning of the fixture.
        """
        block = self._healthy_block(twin, template_state)
        bound = float(condition_bound(block, noise_std=1.0, prior_std=5.0))
        measured = float(condition_estimate(block, noise_std=1.0, prior_std=5.0))

        assert 1e5 < bound < 1e8, bound
        assert measured < 10.0, measured

    def test_a_correct_answer_is_refused_once_the_guard_is_asked_for(
        self, twin, template_state
    ):
        """What being off by default buys, said out loud and with numbers.

        The solve below matches a dense reference to 1e-3 -- it is the same
        block and the same call ``TestWienerSolve.test_matches_a_dense_solve``
        asserts that of. Its residual is 3.6e-08, which is the float32 floor
        and cannot be tightened. Against a bound of 1.44e+06 the error bound is
        5.2e-02, so a 1e-3 target is refused.

        Nothing here is a defect. The bound is right that it cannot PROVE 1e-3;
        the answer is right anyway. A guard that says so is honest and a guard
        that says so by default is unusable, which is the whole trade this
        change made.

        A large bound alone never condemns, and that matters: the healthy block
        above whose CG lands on an exact zero residual passes this same guard.
        It takes a nonzero residual, however small, for the bound to bite.
        """
        n_time = template_state.coords.time.shape[0]
        # `TestWienerSolve.gain_truth`'s own gradient. A CONSTANT truth lands
        # exactly in the operator's range and CG returns it with a residual of
        # exactly zero -- which is the case above, and which no bound can
        # condemn. It takes a truth that needs real iterations to reach the
        # float32 residual floor this test is about.
        truth = GAIN + 0.1 * jnp.arange(n_time, dtype=float)
        block = self._healthy_block(twin, template_state)
        observed = block.offset + block.forward(truth)

        solved, residual = wiener_solve(block, observed, noise_std=1.0,
                                        prior_std=5.0)
        bound = float(condition_bound(block, noise_std=1.0, prior_std=5.0))
        expected = _dense_reference(block, observed, 1.0, 5.0)

        assert 0.0 < float(residual) < 1e-6, float(residual)
        assert float(residual) * bound > 1e-3, (float(residual), bound)
        assert jnp.allclose(solved, expected, rtol=1e-3, atol=1e-3), solved

        with pytest.raises(RuntimeError, match="precision|condition number"):
            wiener_solve(block, observed, noise_std=1.0, prior_std=5.0,
                         require_convergence=1e-3)

    def test_x64_subprocess_recovers_the_prior_in_the_blind_directions(self):
        """The quantitative claim, at the precision that can support it.

        Along a direction the data does not constrain the posterior IS the
        prior, so the scatter of independent draws must come back at
        ``prior_std`` — not at the ~0.03 the unguarded solve reported. Run in a
        fresh interpreter because the suite runs float32 and ``jax_enable_x64``
        is process-global.
        """
        import os
        import subprocess
        import sys

        script = """
import jax, jax.numpy as jnp
assert jax.config.read("jax_enable_x64")
from rheplicant.inference.linear import LinearBlock, gcr_sample

n_row = 32
seen = jnp.array([0.6, 0.8, 0.0])
blind = jnp.array([-0.8, 0.6, 0.0])
forward = lambda x: jnp.full((n_row,), jnp.sum(seen * x))
block = LinearBlock("coeffs", (3,), jnp.float64, jnp.zeros(n_row), forward,
                    lambda y: jax.vjp(forward, jnp.zeros(3))[1](y)[0])
observed = block.forward(jnp.array([250.0, 5.0, -3.0]))

keys = jax.random.split(jax.random.key(0), 600)
draws = jax.vmap(lambda k: gcr_sample(
    block, observed, noise_std=0.1, prior_std=100.0, key=k,
    tol=1e-13, maxiter=5000)[0])(keys)

scatter = float(jnp.std(draws @ blind))
print("blind-direction scatter:", scatter)
assert 80.0 < scatter < 120.0, scatter
"""
        env = {**os.environ, "JAX_ENABLE_X64": "1"}
        done = subprocess.run([sys.executable, "-c", script], env=env,
                              capture_output=True, text=True)
        assert done.returncode == 0, done.stdout + done.stderr


class TestBothConditioningExitsRunTheSamePreconditions:
    """Added 2026-08-28 by the Wave B mutation set, which found the gap.

    Both exits call ``_require_prior_std`` before they compute anything, and
    until this class existed **nothing asserted it**: deleting the call from
    ``condition_bound`` left the whole targeted suite green. It is not a
    silent-answer gap -- without the refusal ``prior_std=None`` still fails,
    at ``jnp.asarray(None) ** 2`` -- so what was unguarded was the MESSAGE,
    which is the whole reason the refusal was written rather than left to
    the array layer. That makes it exactly the kind of guard a mutation set
    finds and a passing suite does not.
    """

    @staticmethod
    def _block():
        operator = jax.random.normal(jax.random.key(0), (8, 3), dtype=jnp.float32)
        return LinearBlock(
            name="x", shape=(3,), dtype=jnp.float32,
            offset=jnp.zeros((8,), dtype=jnp.float32),
            forward=lambda x: operator @ x,
            adjoint=lambda y: operator.T @ y,
        )

    @pytest.mark.parametrize("exit_", [condition_bound, condition_estimate])
    def test_a_missing_prior_is_refused_by_name(self, exit_):
        with pytest.raises(ParameterSpaceError, match="needs prior_std"):
            exit_(self._block(), noise_std=jnp.float32(0.5), prior_std=None)
        # and the refusal names ITS OWN exit, not a shared one -- three exits
        # share this text and only the caller argument tells them apart.
        with pytest.raises(ParameterSpaceError, match=exit_.__name__):
            exit_(self._block(), noise_std=jnp.float32(0.5), prior_std=None)

    def test_the_conditioning_does_not_depend_on_the_data(self):
        """κ is a property of ``AᵀN⁻¹A + S⁻¹``, which holds no data.

        Asserted because of what the mutation set could NOT kill. The adapter
        hands the far side a zero of the prediction's shape when there is no
        ``observed`` to pass, and replacing that with a bare ``None`` survives
        every test -- correctly, because bayesmith's conditioning path never
        reads ``block.data`` (checked by walking the module: neither
        ``condition_bound``, ``condition_estimate``, ``_condition_bound`` nor
        ``normal_operator`` touches it). So that mutant is EQUIVALENT today
        rather than unguarded, and inventing a fixture to kill it would pin a
        mock.

        What is worth pinning is the property that makes the branch matter: if
        the far side ever starts reading ``data`` for conditioning, this goes
        red and the ``None`` stops being defensive.
        """
        block = self._block()
        first = condition_bound(block, noise_std=jnp.float32(0.5), prior_std=3.0)
        second = condition_bound(block, noise_std=jnp.float32(0.5), prior_std=3.0)
        assert float(first) == float(second)
        # the block carries no data at all here, and the number still exists
        assert jnp.isfinite(first)


class TestTheTwoConditionNumbersDivideTheLabour:
    """``condition_bound`` bounds; ``condition_estimate`` measures. Both
    behaviour and prose, because only one of the two rotted.

    The behaviour never broke: ``test_the_bound_is_loose_by_five_decades_on_a
    _healthy_block`` above has always pinned ``bound >> measured``, so an
    implementation swap goes red there. What broke was the SENTENCE. The fix
    that introduced ``condition_bound`` wrote its explanation --
    "an upper bound", "the number to divide an accuracy target by", "now
    ``λ_max · max(prior_variance)``" -- into ``condition_estimate``'s
    docstring, and never took it off. So the two public docstrings ended up
    contradicting each other about the same function: ``condition_bound``
    said "``condition_estimate`` measures κ instead and is biased LOW; a
    tolerance chosen from it is too loose by that bias", while
    ``condition_estimate`` told the reader to divide their accuracy target by
    it.

    That is not a cosmetic defect. ``condition_estimate`` is what the config
    layer's ``condition`` run kind hands back (``config/sections/conjugate.py``),
    so the number a document author receives came with instructions to use it
    in the one way its own implementation warns against -- and the bias runs
    toward silence: measured 33.9x low at a true κ of 1e4 and ~700x at 1e7, so
    a ``tol`` computed from it is too LOOSE by that factor and certifies an
    answer it should have refused.

    Prose has no test, which is why it could sit there. These are that test.
    """

    @staticmethod
    def _doc(function) -> str:
        assert function.__doc__, function.__name__
        return " ".join(function.__doc__.split()).lower()

    def test_only_the_bound_claims_to_be_the_number_to_divide_by(self):
        """The exact claim that was on the wrong function."""
        bound = self._doc(condition_bound)
        estimate = self._doc(condition_estimate)

        assert "number to divide an accuracy target by" in bound
        assert "number to divide an accuracy target by" not in estimate
        # ... and the estimate says so in as many words, rather than merely
        # omitting it: silence would let the claim drift back in unnoticed.
        assert "do not divide an accuracy target by this number" in estimate

    def test_only_the_bound_claims_to_be_an_upper_bound(self):
        assert "upper bound" in self._doc(condition_bound)
        assert "upper bound" not in self._doc(condition_estimate)

    def test_the_estimate_names_its_own_bias_and_its_direction(self):
        """"Biased" alone is not actionable; a reader needs the SIGN.

        Low means the tolerance it suggests is too loose, which is the
        direction that certifies rather than refuses.
        """
        estimate = self._doc(condition_estimate)
        assert "too small" in estimate or "too low" in estimate or "too large" in estimate
        assert "too loose" in estimate

    def test_each_docstring_points_at_the_other(self):
        """Whichever one a reader lands on, the other is one hop away."""
        assert "condition_estimate" in self._doc(condition_bound)
        assert "condition_bound" in self._doc(condition_estimate)

    def test_the_behaviour_the_prose_describes_is_the_behaviour_shipped(
        self, twin, template_state
    ):
        """The anti-vacuity half: the sentences above are checked against the
        numbers, so a docstring pair that agreed with each other while both
        describing the wrong function would still fail here.

        ``condition_bound`` floors ``λ_min`` at ``1/max(prior_variance)``
        instead of measuring it, so on a block whose data constrains every
        direction it must come back the LARGER of the two -- and by decades,
        not by a tie-break.
        """
        n_time = template_state.coords.time.shape[0]
        wide = eqx.tree_at(lambda p: p["gain"].gain, twin, jnp.full((n_time,), GAIN))
        block = linear_operator(
            ParameterSpace.direct("gains", init=jnp.full((n_time,), GAIN),
                                  into=lambda p: p["gain"].gain, linear=True),
            wide, template_state,
        )
        bound = float(condition_bound(block, noise_std=1.0, prior_std=5.0))
        measured = float(condition_estimate(block, noise_std=1.0, prior_std=5.0))
        assert bound > 1e4 * measured, (bound, measured)
