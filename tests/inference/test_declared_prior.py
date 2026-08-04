"""The declared prior has to reach the conjugate exits.

``Latent(prior=...)`` is the package's single statement of what a latent is a
priori. ``to_numpyro_model`` reads it. The conjugate-Gaussian exits — the
Wiener mean, the GCR draw, the reweighted GLS — used to ignore it entirely and
require the same two numbers hand-passed as keywords, so the SAME space could
be given to NUTS and to ``gcr_sample`` and get two different posteriors with
nothing raised.

The measurement that opened this, on a scalar-gain twin: three declarations,
one keyword call, three bit-identical answers. These tests are that
measurement inverted — three declarations must now give three different
answers, each equal to the closed-form Wiener estimate the declaration
implies.

Both halves of the seam are pinned here: that a declaration is used, and that
a declaration the conjugate machinery CANNOT honour (a Half-Normal, a Uniform,
a LogNormal — the last of which duck-types as a Gaussian, holding ``.loc`` and
``.scale``, while being nothing of the sort) is refused rather than quietly
approximated.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from rheplicant.core.combinators import SumOperator
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    HomoscedasticNoise,
    Latent,
    ParameterSpace,
    RadiometerNoise,
    condition_estimate,
    gcr_sample,
    iterative_gls,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import GainOperator, SkyOperator

SKY_A, SKY_B = 100.0, 20.0

#: Deliberately far above the signal, so the LIKELIHOOD is nearly flat and the
#: prior is what decides the answer. A prior that is ignored then shows up as a
#: whole-number discrepancy rather than in the seventh digit.
WEAK_SIGMA = 1e4

TRUE_GAIN = 3.0

#: A radiometer whose fractional level is huge, so ``sigma = |prediction| * f``
#: is as uninformative as :data:`WEAK_SIGMA` — the same "let the prior decide"
#: setup, reached through the PREDICTION-DEPENDENT noise model. That model is
#: the only one that enters ``iterative_gls``'s reweighting loop, and so the
#: only one that exercises the declaration inside a live trace.
WEAK_RADIOMETER = RadiometerNoise(channel_width=1.0, integration_time=1e-8)

#: A radiometer chosen so the likelihood and a ``std=0.4`` prior carry
#: comparable weight: the answer then sits strictly between the data's 3.0 and
#: the prior's centre, so BOTH halves of the solve are load-bearing.
BALANCED_RADIOMETER = RadiometerNoise(channel_width=2.0, integration_time=1.0)


@pytest.fixture
def twin():
    """Two additive sky terms through a gain — linear in the gain, offset zero."""
    return Pipeline(
        SumOperator(
            SkyOperator(amplitude=jnp.array(SKY_A)),
            SkyOperator(amplitude=jnp.array(SKY_B)),
            names=("sky_a", "sky_b"),
        ),
        GainOperator(gain=jnp.array(1.0)),
        names=("sum", "gain"),
    )


def gain_block(twin, state, prior):
    """A one-parameter linear block whose latent carries ``prior``."""
    space = ParameterSpace.direct(
        "gain", init=1.0, into=lambda p: p["gain"].gain, prior=prior, linear=True
    )
    return linear_operator(space, twin, state)


def wiener_by_hand(block, observed, *, mean, std, noise_std=WEAK_SIGMA):
    """The closed-form posterior mean for a SCALAR latent, assembled directly.

    ``x̂ = (Σ aᵢwᵢrᵢ + m/s²) / (Σ aᵢ²wᵢ + 1/s²)`` with ``a = A·1``. Written out
    rather than obtained from the code under test, so it is an independent
    statement of what the declaration means.

    ``noise_std`` may be a scalar or a per-sample array — the weights stay
    inside the sums, which is the same number for a scalar and the right one
    for the varying sigma a reweighted GLS converges to.
    """
    a = block.forward(jnp.array(1.0))
    weight = 1.0 / jnp.asarray(noise_std) ** 2
    residual = observed - block.offset
    curvature = float(jnp.sum(a**2 * weight)) + 1.0 / std**2
    rhs = float(jnp.sum(a * residual * weight)) + mean / std**2
    return rhs / curvature


@pytest.fixture
def observed(twin, template_state):
    """Data from the truth, noise-free — the noise lives in ``WEAK_SIGMA``."""
    block = gain_block(twin, template_state, None)
    return block.offset + block.forward(jnp.array(TRUE_GAIN))


class TestDeclaredPriorReachesTheSolvers:
    """The measured defect, inverted."""

    @pytest.mark.parametrize(
        ("mean", "std"),
        [(1.0, 0.05), (9.9, 1e-4), (-4.0, 2.0)],
    )
    def test_wiener_solve_uses_the_declaration(
        self, twin, template_state, observed, mean, std
    ):
        block = gain_block(twin, template_state, dist.Normal(mean, std))
        estimate, _ = wiener_solve(block, observed, noise_std=WEAK_SIGMA)
        assert float(estimate) == pytest.approx(
            wiener_by_hand(block, observed, mean=mean, std=std), rel=1e-4
        )

    def test_three_declarations_give_three_different_answers(
        self, twin, template_state, observed
    ):
        """The exact table the hearing measured as bit-identical."""
        tight = gain_block(twin, template_state, dist.Normal(1.0, 0.05))
        far = gain_block(twin, template_state, dist.Normal(9.9, 1e-4))
        free = gain_block(twin, template_state, None)

        a = float(wiener_solve(tight, observed, noise_std=WEAK_SIGMA)[0])
        b = float(wiener_solve(far, observed, noise_std=WEAK_SIGMA)[0])
        c = float(
            wiener_solve(free, observed, noise_std=WEAK_SIGMA, prior_std=1.0)[0]
        )

        # N(1.0, 0.05): the prior dominates a nearly flat likelihood.
        assert a == pytest.approx(1.00002, rel=1e-3)
        # N(9.9, 1e-4): a prior this sharp is essentially the answer.
        assert b == pytest.approx(9.9, rel=1e-4)
        # No declaration, prior_std=1 and prior_mean defaulting to zero.
        assert c == pytest.approx(0.01376, rel=1e-2)

        assert abs(a - b) > 1.0 and abs(a - c) > 0.5 and abs(b - c) > 1.0

    def test_gcr_sample_uses_the_declaration(self, twin, template_state, observed):
        """With an uninformative likelihood a draw falls back to the prior —
        the check the ``gcr_sample`` docstring itself names."""
        block = gain_block(twin, template_state, dist.Normal(2.5, 0.4))
        keys = jax.random.split(jax.random.key(0), 512)
        draws = jax.vmap(
            lambda k: gcr_sample(block, observed, noise_std=WEAK_SIGMA, key=k)[0]
        )(keys)
        assert float(jnp.mean(draws)) == pytest.approx(2.5, abs=0.06)
        assert float(jnp.std(draws)) == pytest.approx(0.4, rel=0.1)

    def test_iterative_gls_uses_the_declaration(self, twin, template_state, observed):
        block = gain_block(twin, template_state, dist.Normal(9.9, 1e-4))
        found = iterative_gls(
            block, observed, noise=HomoscedasticNoise(sigma=jnp.array(WEAK_SIGMA))
        )
        assert float(found.solution) == pytest.approx(9.9, rel=1e-4)


class TestTheReweightingLoopReadsTheDeclaration:
    """``iterative_gls`` under the noise model it exists for.

    ``HomoscedasticNoise`` does not depend on the prediction, so there is
    nothing to reweight and ``iterative_gls`` early-returns a single
    ``wiener_solve`` — the loop is never entered. Every claim about the
    declaration reaching ``iterative_gls`` is therefore untested by a
    homoscedastic case. These use ``RadiometerNoise``, whose sigma tracks the
    prediction, which is the whole reason this function exists and the only
    path that resolves the prior once and re-passes it into ``wiener_solve``
    from inside a ``lax.while_loop`` trace.
    """

    def test_a_declared_prior_survives_the_loop(self, twin, template_state, observed):
        """Requirement 1's exact usage: declare the prior, pass no keywords."""
        block = gain_block(twin, template_state, dist.Normal(2.5, 0.4))
        found = iterative_gls(block, observed, noise=BALANCED_RADIOMETER)

        assert bool(found.converged)
        # The closed form at the covariance the run actually converged to.
        assert float(found.solution) == pytest.approx(
            wiener_by_hand(
                block, observed, mean=2.5, std=0.4, noise_std=found.noise_std
            ),
            rel=1e-3,
        )
        # And the prior is doing work: strictly between the data and the prior.
        assert 2.5 < float(found.solution) < TRUE_GAIN

    @pytest.mark.parametrize(
        ("mean", "std"), [(9.9, 1e-4), (1.0, 5e-4), (-4.0, 1e-3)]
    )
    def test_the_declaration_is_what_the_loop_converges_to(
        self, twin, template_state, observed, mean, std
    ):
        """Three declarations, one call, three different answers — the same
        table the hearing measured as bit-identical, through the loop."""
        block = gain_block(twin, template_state, dist.Normal(mean, std))
        found = iterative_gls(block, observed, noise=WEAK_RADIOMETER)
        assert bool(found.converged)
        assert float(found.solution) == pytest.approx(mean, rel=1e-3)

    def test_a_contradicting_keyword_is_still_refused_here(
        self, twin, template_state, observed
    ):
        """Weakening the reconciliation so the loop can run must not weaken the
        refusal: these two numbers still disagree, concretely."""
        block = gain_block(twin, template_state, dist.Normal(2.5, 0.4))
        with pytest.raises(ParameterSpaceError, match="prior_std"):
            iterative_gls(
                block, observed, noise=BALANCED_RADIOMETER, prior_std=1.0
            )

    def test_condition_estimate_uses_the_declaration(self, twin, template_state):
        """κ is what the docstrings tell you to choose ``tol`` from, so it has
        to be the κ of the system the solvers will actually build."""
        block = gain_block(twin, template_state, dist.Normal(0.0, 7.0))
        assert float(condition_estimate(block, noise_std=WEAK_SIGMA)) == pytest.approx(
            float(condition_estimate(block, noise_std=WEAK_SIGMA, prior_std=7.0))
        )


class TestVectorLatents:
    """The shape the module exists for: many coefficients, one declaration."""

    @pytest.fixture
    def amps_block_for(self, twin, template_state):
        def build(prior):
            space = ParameterSpace.direct(
                "amps", init=jnp.zeros(3), into=lambda p: p["sum"]["sky_a"].amplitude,
                prior=prior, fn=jnp.sum, linear=True,
            )
            return linear_operator(space, twin, template_state)

        return build

    @pytest.mark.parametrize(
        "wrap",
        [
            lambda d: d,
            lambda d: d.to_event(1),  # Independent
            lambda d: dist.Normal(2.0, 0.5).expand((3,)),  # ExpandedDistribution
        ],
        ids=["plain", "independent", "expanded"],
    )
    def test_a_per_element_gaussian_is_read_through_its_wrapper(
        self, amps_block_for, wrap
    ):
        """``Independent`` and ``expand`` only re-shape a Normal, so both are
        still the diagonal ``S`` these solves take."""
        declared = wrap(dist.Normal(jnp.full((3,), 2.0), jnp.full((3,), 0.5)))
        block = amps_block_for(declared)
        observed = block.offset + block.forward(jnp.full((3,), 1.0))
        from_declaration, _ = wiener_solve(block, observed, noise_std=WEAK_SIGMA)
        by_keyword, _ = wiener_solve(
            block, observed, noise_std=WEAK_SIGMA, prior_std=0.5, prior_mean=2.0
        )
        assert from_declaration.shape == (3,)
        assert jnp.allclose(from_declaration, by_keyword, rtol=1e-5)
        # The likelihood is nearly flat, so the answer is essentially the prior.
        assert jnp.allclose(from_declaration, 2.0, atol=1e-3)

    def test_a_keyword_of_the_wrong_shape_is_a_disagreement_not_a_crash(
        self, amps_block_for
    ):
        """``jnp.equal`` raises on shapes that do not broadcast. That is still
        a keyword contradicting a declaration, and it gets the message that
        says so rather than a broadcasting TypeError from inside the solver."""
        block = amps_block_for(dist.Normal(jnp.zeros(3), jnp.ones(3)))
        observed = block.offset + block.forward(jnp.full((3,), 1.0))
        with pytest.raises(ParameterSpaceError, match="prior_std"):
            wiener_solve(
                block, observed, noise_std=WEAK_SIGMA, prior_std=jnp.ones(5)
            )

    def test_a_truncated_normal_is_not_unwrapped(self, amps_block_for):
        """It also carries ``.base_dist``, and that base IS a Normal — but the
        truncation is the whole point of declaring it."""
        block = amps_block_for(
            dist.TruncatedNormal(jnp.zeros(3), jnp.ones(3), low=jnp.zeros(3))
        )
        observed = block.offset + block.forward(jnp.full((3,), 1.0))
        with pytest.raises(ParameterSpaceError, match="conjugate"):
            wiener_solve(block, observed, noise_std=WEAK_SIGMA)


class TestSeveralLatents:
    """Which latent's declaration the block carries.

    Every other test here builds its space with ``ParameterSpace.direct``,
    which declares exactly one latent — so ``space.latents[0]`` and the latent
    actually resolved are the same object, and an operator that carried the
    *first-declared* prior instead of the *resolved* one would be
    indistinguishable. A Gibbs sweep over several latents is this module's
    stated reason to exist, and it is exactly where the two come apart: the
    block for ``'gain'`` would silently solve with ``'amp_b'``'s prior.
    """

    @pytest.fixture
    def two_linear_latents(self):
        """``prediction = gain * (SKY_A + sum(amp_b))`` — affine in EITHER,
        given the other, so both may be declared linear and both get an
        operator.

        ``amp_b`` is deliberately a DIFFERENT SHAPE from ``gain`` (a
        3-vector, reduced through ``fn=jnp.sum``), not only a different
        prior: a block that carried ``space.latents[0]``'s ``shape``/
        ``dtype`` instead of the ``Latent`` it was actually asked to resolve
        would then be visibly wrong on its own — the same defect class as
        carrying the wrong ``prior``, one line above in
        :func:`~rheplicant.inference.linear.linear_operator`.

        The two declarations are deliberately far apart and both sharp enough
        to dominate a nearly flat likelihood, so whichever one reaches the
        solve is legible in the first digit of the answer.
        """
        return ParameterSpace(
            latents=(
                # Declared FIRST, and NOT the one asked for below — and a
                # different SHAPE, so carrying this one by mistake is wrong
                # in shape as well as in prior.
                Latent(
                    "amp_b",
                    init=jnp.full((3,), SKY_B / 3),
                    prior=dist.Normal(jnp.full((3,), 20.0), jnp.full((3,), 1e-6)),
                    linear=True,
                ),
                Latent("gain", init=1.0, prior=dist.Normal(7.0, 0.01), linear=True),
            ),
            bindings=(
                Bind("amp_b", into=lambda p: p["sum"]["sky_b"].amplitude, fn=jnp.sum),
                Bind("gain", into=lambda p: p["gain"].gain),
            ),
        )

    def test_each_block_carries_its_own_latents_declaration(
        self, twin, template_state, two_linear_latents
    ):
        gain = linear_operator(two_linear_latents, twin, template_state, "gain")
        amp_b = linear_operator(two_linear_latents, twin, template_state, "amp_b")

        # Carrying `space.latents[0]` (amp_b) instead of the resolved latent
        # would give the 'gain' block a (3,) shape, and/or the 'amp_b' block
        # a () shape — pinned here, not only through the prior below.
        assert gain.shape == ()
        assert amp_b.shape == (3,)

        assert (float(gain.prior.loc), float(gain.prior.scale)) == (7.0, 0.01)
        assert jnp.allclose(amp_b.prior.loc, 20.0)
        assert jnp.allclose(amp_b.prior.scale, 1e-6)

    def test_the_solve_is_driven_by_the_resolved_latents_prior(
        self, twin, template_state, two_linear_latents, observed
    ):
        """Not the first-declared one. With the likelihood nearly flat the
        answer is essentially the declared mean, so carrying ``amp_b``'s
        ``Normal(20.0, 1e-6)`` into the gain solve shows up as 20 where 7
        belongs — a finite, confident, wrong answer, which is the failure this
        module is built to refuse.
        """
        block = linear_operator(two_linear_latents, twin, template_state, "gain")
        estimate, _ = wiener_solve(block, observed, noise_std=WEAK_SIGMA)
        assert float(estimate) == pytest.approx(7.0, rel=1e-3)
        assert float(estimate) == pytest.approx(
            wiener_by_hand(block, observed, mean=7.0, std=0.01), rel=1e-4
        )

    def test_a_contradiction_is_measured_against_the_resolved_latent(
        self, twin, template_state, two_linear_latents, observed
    ):
        """The refusal has to be measured against the right declaration too:
        ``(0.01, 7.0)`` agrees with ``'gain'`` and contradicts ``'amp_b'``.
        Carrying the wrong latent's prior would reverse both verdicts.
        """
        gain = linear_operator(two_linear_latents, twin, template_state, "gain")
        amp_b = linear_operator(two_linear_latents, twin, template_state, "amp_b")

        agrees, _ = wiener_solve(
            gain, observed, noise_std=WEAK_SIGMA, prior_std=0.01, prior_mean=7.0
        )
        assert float(agrees) == pytest.approx(7.0, rel=1e-3)

        with pytest.raises(ParameterSpaceError, match="amp_b"):
            wiener_solve(
                amp_b, observed, noise_std=WEAK_SIGMA, prior_std=0.01, prior_mean=7.0
            )


class TestContradictionIsRefused:
    """One of the two would silently win. Neither may."""

    def test_prior_std_contradicting_the_declaration_raises(
        self, twin, template_state, observed
    ):
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))
        with pytest.raises(ParameterSpaceError, match="prior_std"):
            wiener_solve(block, observed, noise_std=WEAK_SIGMA, prior_std=1.0)

    def test_prior_mean_contradicting_the_declaration_raises(
        self, twin, template_state, observed
    ):
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))
        with pytest.raises(ParameterSpaceError, match="prior_mean"):
            wiener_solve(block, observed, noise_std=WEAK_SIGMA, prior_mean=0.0)

    def test_gcr_sample_refuses_the_contradiction_too(
        self, twin, template_state, observed
    ):
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))
        with pytest.raises(ParameterSpaceError, match="prior_std"):
            gcr_sample(
                block, observed, noise_std=WEAK_SIGMA, prior_std=1.0,
                key=jax.random.key(0),
            )

    def test_a_traced_keyword_cannot_be_reconciled_and_says_so(
        self, twin, template_state, observed
    ):
        """A keyword that is *genuinely* a tracer has no comparison at trace
        time. Refusing beats picking one — this is the same reason the
        convergence guard uses ``eqx.error_if`` rather than a Python ``if``.

        Read together with
        :meth:`test_a_concrete_agreeing_keyword_survives_jit`: what makes the
        value undecidable is that it *is* a tracer, not that some trace happens
        to be live.
        """
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))

        @jax.jit
        def solve(std):
            return wiener_solve(block, observed, noise_std=WEAK_SIGMA, prior_std=std)[0]

        # Names the SUPPLIED side, not the declared one: nothing declared is
        # traced here, so a message blaming latent 'gain''s declared scale
        # would be false about this call and must not pass silently. "traced"
        # alone is satisfied by either side's wording — this pins the side.
        with pytest.raises(ParameterSpaceError, match=r"prior_std= you passed is a traced value"):
            solve(jnp.array(0.05))

    @pytest.mark.parametrize("wrap", [jax.jit, eqx.filter_jit], ids=["jax", "equinox"])
    def test_a_concrete_agreeing_keyword_survives_jit(
        self, twin, template_state, observed, wrap
    ):
        """0.05 is 0.05 whether or not a trace is open.

        Both numbers here are Python floats closed over by the traced function;
        neither is a tracer. Deciding their equality with ``jnp`` stages the
        comparison into whatever trace is live and turns a settled ``True``
        into an unanswerable one, which the reconciliation then reports as
        "one of the two is a traced value" — a message that is simply false
        about this call, attached to a refusal of a call that is correct. The
        jitted answer must equal the unjitted one.
        """
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))

        @wrap
        def solve(data):
            return wiener_solve(
                block, data, noise_std=WEAK_SIGMA, prior_std=0.05, prior_mean=1.0
            )[0]

        assert float(solve(observed)) == pytest.approx(
            float(wiener_solve(block, observed, noise_std=WEAK_SIGMA)[0]), rel=1e-6
        )

    def test_a_float32_declaration_agrees_with_its_python_float_keyword(
        self, twin, template_state, observed
    ):
        """``Normal(jnp.asarray(1.0), jnp.asarray(0.05))`` declares float32.

        ``prior_std=0.05`` is a Python float, i.e. float64. Widened to compare,
        the declared scale reads ``0.05000000074505806`` and the two are not
        equal — so a comparison done in NumPy calls this a contradiction and
        refuses a caller who passed exactly the declared number. The comparison
        has to happen in the working precision, which is what the solve uses.
        """
        block = gain_block(
            twin, template_state, dist.Normal(jnp.asarray(1.0), jnp.asarray(0.05))
        )
        with_keywords, _ = wiener_solve(
            block, observed, noise_std=WEAK_SIGMA, prior_std=0.05, prior_mean=1.0
        )
        without, _ = wiener_solve(block, observed, noise_std=WEAK_SIGMA)
        assert float(with_keywords) == float(without)

        # The same cell with a trace open: a float32 declaration, a float64
        # keyword and a live jit at once, which is where the two failure modes
        # this class pins would have to be caught together.
        @jax.jit
        def solve(data):
            return wiener_solve(
                block, data, noise_std=WEAK_SIGMA, prior_std=0.05, prior_mean=1.0
            )[0]

        assert float(solve(observed)) == pytest.approx(float(without), rel=1e-6)

    @pytest.mark.parametrize(
        "wrap_in",
        [lambda std: [std], lambda std: {"k": std}],
        ids=["list", "dict"],
    )
    def test_a_tracer_inside_a_container_is_undecidable_not_a_contradiction(
        self, twin, template_state, observed, wrap_in
    ):
        """A tracer that never reaches a bare ``isinstance`` check is still a
        tracer. Reporting it as a *disagreement* would name two numbers as
        contradicting when one of them has no value yet.

        Both containers matter, for different reasons: a LIST still lets
        ``jnp.asarray([tracer])`` succeed, so ``bool()`` on the traced result
        raises ``ConcretizationTypeError`` — a bare-``isinstance`` guard is
        saved here by the backstop ``except jax.errors.ConcretizationTypeError``
        clause in :func:`_agrees`, so this case alone would not catch a guard
        that only checks the top-level object. A DICT never reaches that
        backstop the same way: ``jnp.asarray({"k": tracer})`` raises a plain
        ``TypeError`` before any tracer is ever compared, which the *other*
        backstop clause, ``except (TypeError, ValueError): return False``,
        also swallows — but into a settled, invented disagreement rather than
        ``None``. Only scanning the pytree leaves (rather than the object
        itself) catches the dict case.
        """
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))

        @jax.jit
        def solve(std):
            return wiener_solve(
                block, observed, noise_std=WEAK_SIGMA, prior_std=wrap_in(std)
            )[0]

        with pytest.raises(ParameterSpaceError, match="traced"):
            solve(jnp.array(0.05))

    def test_a_concrete_contradicting_keyword_still_raises_under_jit(
        self, twin, template_state, observed
    ):
        """The other half of the same seam. Letting concrete values through a
        live trace must not let a *disagreement* through with them."""
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))

        @jax.jit
        def solve(data):
            return wiener_solve(block, data, noise_std=WEAK_SIGMA, prior_std=1.0)[0]

        with pytest.raises(ParameterSpaceError, match="prior_std"):
            solve(observed)

    def test_a_keyword_that_agrees_is_accepted(self, twin, template_state, observed):
        """Every call site written before the declaration was honoured passes
        exactly the declared numbers. None of them may start raising."""
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))
        with_keywords, _ = wiener_solve(
            block, observed, noise_std=WEAK_SIGMA, prior_std=0.05, prior_mean=1.0
        )
        without, _ = wiener_solve(block, observed, noise_std=WEAK_SIGMA)
        assert float(with_keywords) == float(without)


class TestNonConjugateIsRefused:
    """A prior with no conjugate Gaussian form is an error, not a rounding."""

    @pytest.mark.parametrize(
        "prior",
        [
            dist.HalfNormal(1.0),
            dist.Uniform(0.0, 3.0),
            dist.LogNormal(0.0, 1.0),
            dist.MultivariateNormal(jnp.zeros(1), jnp.eye(1)),
        ],
        ids=["half_normal", "uniform", "log_normal", "multivariate_normal"],
    )
    def test_wiener_solve_refuses(self, twin, template_state, observed, prior):
        init = jnp.zeros(1) if prior.shape() == (1,) else 1.0
        space = ParameterSpace.direct(
            "gain", init=init, into=lambda p: p["gain"].gain,
            prior=prior, fn=jnp.sum, linear=True,
        )
        block = linear_operator(space, twin, template_state)
        with pytest.raises(ParameterSpaceError, match="conjugate"):
            wiener_solve(block, observed, noise_std=WEAK_SIGMA)

    def test_the_message_names_the_exits_that_do_support_it(
        self, twin, template_state, observed
    ):
        block = gain_block(twin, template_state, dist.HalfNormal(1.0))
        with pytest.raises(ParameterSpaceError) as caught:
            wiener_solve(block, observed, noise_std=WEAK_SIGMA)
        message = str(caught.value)
        assert "HalfNormal" in message
        assert "to_numpyro_model" in message

    def test_a_lognormal_is_not_mistaken_for_a_gaussian(
        self, twin, template_state, observed
    ):
        """It holds ``.loc`` and ``.scale``, so attribute duck-typing would take
        it for a Normal and return a finite, confident, wrong posterior."""
        prior = dist.LogNormal(0.0, 1.0)
        assert hasattr(prior, "loc") and hasattr(prior, "scale")
        block = gain_block(twin, template_state, prior)
        with pytest.raises(ParameterSpaceError, match="LogNormal"):
            wiener_solve(block, observed, noise_std=WEAK_SIGMA)


class TestKeywordOnlyUsageIsUnchanged:
    """The escape hatch for prior-free latents, and the guard on it."""

    def test_a_prior_free_latent_still_solves_from_keywords(
        self, twin, template_state, observed
    ):
        block = gain_block(twin, template_state, None)
        estimate, _ = wiener_solve(
            block, observed, noise_std=WEAK_SIGMA, prior_std=2.0, prior_mean=-1.0
        )
        assert float(estimate) == pytest.approx(
            wiener_by_hand(block, observed, mean=-1.0, std=2.0), rel=1e-4
        )

    def test_a_prior_free_latent_still_refuses_a_missing_prior_std(
        self, twin, template_state, observed
    ):
        block = gain_block(twin, template_state, None)
        with pytest.raises(ParameterSpaceError, match="needs prior_std"):
            wiener_solve(block, observed, noise_std=WEAK_SIGMA)

    def test_shape_mismatch_is_still_caught_first(
        self, twin, template_state, observed
    ):
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))
        with pytest.raises(ParameterSpaceError, match="different problem"):
            wiener_solve(block, observed[:2], noise_std=WEAK_SIGMA)
