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

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from rheplicant.core.combinators import SumOperator
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    HomoscedasticNoise,
    ParameterSpace,
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
    """
    a = block.forward(jnp.array(1.0))
    weight = 1.0 / noise_std**2
    residual = observed - block.offset
    curvature = float(jnp.sum(a**2)) * weight + 1.0 / std**2
    rhs = float(jnp.sum(a * residual)) * weight + mean / std**2
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
        """Under jit the comparison has no answer at trace time. Refusing beats
        picking one — this is the same reason the convergence guard uses
        ``eqx.error_if`` rather than a Python ``if``."""
        block = gain_block(twin, template_state, dist.Normal(1.0, 0.05))

        @jax.jit
        def solve(std):
            return wiener_solve(block, observed, noise_std=WEAK_SIGMA, prior_std=std)[0]

        with pytest.raises(ParameterSpaceError, match="traced"):
            solve(jnp.array(0.05))

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
