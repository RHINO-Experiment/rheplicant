"""Tests for the log-space conjugate block under multiplicative noise.

The model is the one this exists for: ``d = g (T_ant + T_nw + tone) (1 + f w)``.
``g`` is bound as ``exp(log_gain)``, so the prediction is NOT affine in
``log_gain`` and the linear machinery refuses it -- while ``log`` of the
prediction is affine in it, whatever the sky is made of, because ``log S``
enters as a constant. Three claims are pinned here and none of them is taken on
trust:

* the summed sky really does land in ``LinearBlock.offset`` (``test_the_offset
  _is_log_of_the_sky``), which is the whole reason an additive sky is no
  obstacle to a log-linear GAIN;
* the log-space sigma really is constant, so the reweighting fixed point
  ``iterative_gls`` performs has nothing left to do;
* the draw is a real constrained realization and not a mean plus arbitrary
  noise -- checked on the SECOND moment against the closed form, the check
  ``test_linear_blocks.py`` calls the one that distinguishes them.

The closed form is available because the block is trivial in log space:
``y = log_gain[t] + log S[f]`` with constant sigma, so the posterior for each
time sample is Gaussian with precision ``n_freq / f^2 + 1 / prior^2``. Every
assertion below compares against that rather than against a recorded number.

``f`` is taken from a real configuration (61 kHz x 1 s -> 4.05e-3, the value
``docs/inference-linear.md`` uses) rather than from a round number, so the
tests run where the approximation is actually used.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import LinearityRefused, ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Block,
    Latent,
    ParameterSpace,
    SamplingPlan,
    gcr_sample,
    wiener_solve,
)
from rheplicant.inference.loglinear import (
    FIRST_ORDER_MAX_FRACTIONAL,
    check_log_linearity,
    log_linear_operator,
    to_log_space,
)
from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    RadiometerNoise,
)
from rheplicant.radio import GainOperator

N_TIME, N_FREQ = 6, 32
TONE_CHANNEL = 7
PRIOR_STD = 10.0

#: 61 kHz x 1 s -- docs/inference-linear.md's own configuration. f = 4.05e-3.
CHANNEL_WIDTH, INTEGRATION_TIME = 61e3, 1.0


class SummedSky(AbstractOperator):
    """``data[t, f] = T_ant[f] + T_nw[f] + tone[f]`` -- a SUM, deliberately.

    log of a sum is not affine in the summands, which is exactly the case that
    was supposed to obstruct a log-linear gain and does not.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    t_ant: jax.Array
    t_nw: jax.Array
    tone: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        row = self.t_ant + self.t_nw + self.tone
        return state.with_data(jnp.broadcast_to(row, (n_time, row.shape[0])))


class AddConstant(AbstractOperator):
    """A known positive level added AFTER the gain -- a receiver-like term.

    Keeps the prediction strictly positive while destroying log-linearity, so
    the affinity check has to be the thing that catches it.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    level: float

    def __call__(self, state):
        return state.with_data(state.data + self.level)


FREQ = jnp.linspace(60e6, 85e6, N_FREQ)
T_ANT = 2000.0 * (FREQ / 60e6) ** -2.5
T_NW = 150.0 * jnp.exp(-jnp.linspace(0.0, 3.0, N_FREQ))
TONE = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(400.0)
SKY = T_ANT + T_NW + TONE

#: Not a constant and not a multiple of anything else here, so a per-time
#: answer carried into the wrong slot cannot pass.
LOG_G = jnp.log(1.5 + 0.05 * jnp.arange(N_TIME, dtype=float))


@pytest.fixture
def state():
    return State(
        coords=Coordinates(time=jnp.arange(N_TIME, dtype=float), freq=FREQ),
        meta={"telescope": "RHINO", "obs_id": "loglinear-000"},
    )


@pytest.fixture
def pipeline():
    return Pipeline(
        SummedSky(t_ant=T_ANT, t_nw=T_NW, tone=TONE),
        GainOperator(gain=jnp.ones(N_TIME)),
        names=("sky", "gain"),
    )


def log_gain_space(linear: bool = False) -> ParameterSpace:
    return ParameterSpace(
        latents=[Latent("log_gain", init=LOG_G, prior=None, linear=linear)],
        bindings=[Bind("log_gain", into=lambda p: p["gain"].gain, fn=jnp.exp)],
    )


def linear_gain_space() -> ParameterSpace:
    """The SAME model with the gain entered directly -- affine, so NOT log-linear."""
    return ParameterSpace(
        latents=[Latent("gain", init=jnp.exp(LOG_G), prior=None)],
        bindings=[Bind("gain", into=lambda p: p["gain"].gain)],
    )


@pytest.fixture
def space():
    return log_gain_space()


@pytest.fixture
def noise():
    return RadiometerNoise(
        channel_width=CHANNEL_WIDTH, integration_time=INTEGRATION_TIME
    )


@pytest.fixture
def observed(space, pipeline, state, noise):
    forward, values0 = space.forward_fn(pipeline, state)
    truth = forward({**values0, "log_gain": LOG_G})
    return noise.realise(truth, key=jax.random.key(0))


def _closed_form(y, sigma):
    """Posterior mean and sd per time sample, for ``y = log_gain[t] + log S[f]``."""
    precision = N_FREQ / sigma**2 + 1.0 / PRIOR_STD**2
    mean = jnp.sum(y - jnp.log(SKY)[None, :], axis=1) / sigma**2 / precision
    return mean, jnp.sqrt(1.0 / precision)


class TestTheBlock:
    def test_the_offset_is_log_of_the_sky(self, space, pipeline, state):
        """The claim that makes an additive sky irrelevant to a log-linear gain."""
        block = log_linear_operator(space, pipeline, state, "log_gain")
        assert jnp.allclose(block.offset, jnp.log(SKY)[None, :], rtol=1e-6)

    def test_the_block_is_affine_where_the_prediction_is_not(
        self, space, pipeline, state
    ):
        """log(prediction) passes the affinity check; the prediction itself fails.

        Both halves matter. Without the second, "it is log-linear" would be
        consistent with the model simply being linear already.
        """
        check_log_linearity(space, pipeline, state, "log_gain")

        from rheplicant.inference.linear import check_linearity

        with pytest.raises(LinearityRefused):
            check_linearity(log_gain_space(linear=True), pipeline, state, "log_gain")

    def test_offset_plus_forward_reproduces_log_of_the_model(
        self, space, pipeline, state
    ):
        block = log_linear_operator(space, pipeline, state, "log_gain")
        forward, values0 = space.forward_fn(pipeline, state)
        assert jnp.allclose(
            block.offset + block.forward(LOG_G),
            jnp.log(forward({**values0, "log_gain": LOG_G})),
            rtol=1e-5,
        )


class TestTheNoiseTransform:
    def test_the_log_space_sigma_is_constant(self, observed, noise):
        """The structural win: nothing left for a reweighting loop to iterate on."""
        _, sigma = to_log_space(observed, noise)
        assert jnp.all(sigma == noise.fractional)
        assert jnp.shape(sigma) == jnp.shape(observed)

    def test_the_leading_order_mean_shift_is_added_back(self, observed, noise):
        """``E[log(1 + f w)] = -f^2/2``, so the log data sits low by a constant."""
        y, _ = to_log_space(observed, noise)
        assert jnp.allclose(y, jnp.log(observed) + noise.fractional**2 / 2, rtol=1e-6)

    def test_a_flagged_sample_may_be_non_positive(self, observed, noise):
        """It was not observed, so it may hold anything and must inform nothing."""
        flags = jnp.zeros((N_TIME, N_FREQ), bool).at[:, 3].set(True)
        y, sigma = to_log_space(observed.at[:, 3].set(-1.0), FlaggedNoise(noise, flags))
        assert jnp.all(jnp.isinf(sigma[:, 3]))
        assert jnp.all(jnp.isfinite(y))

    def test_an_unflagged_non_positive_sample_is_refused(self, observed, noise):
        with pytest.raises(ParameterSpaceError) as caught:
            to_log_space(observed.at[:, 3].set(-1.0), noise)
        assert "NEGATIVE" in str(caught.value)

    def test_additive_noise_is_refused(self, observed):
        """log space buys nothing for a model whose noise is already additive."""
        with pytest.raises(ParameterSpaceError) as caught:
            to_log_space(observed, HomoscedasticNoise(sigma=1.0))
        assert "multiplicative" in str(caught.value)


class TestTheFirstOrderBoundary:
    """Both sides of FIRST_ORDER_MAX_FRACTIONAL, and an extreme beyond it."""

    @staticmethod
    def _noise_with(fractional: float) -> RadiometerNoise:
        return RadiometerNoise(
            channel_width=1.0 / fractional**2, integration_time=1.0
        )

    def test_just_below_the_threshold_is_accepted(self, observed):
        noise = self._noise_with(FIRST_ORDER_MAX_FRACTIONAL * 0.99)
        _, sigma = to_log_space(observed, noise)
        assert jnp.all(jnp.isfinite(sigma))

    def test_just_above_the_threshold_is_refused(self, observed):
        noise = self._noise_with(FIRST_ORDER_MAX_FRACTIONAL * 1.01)
        with pytest.raises(ParameterSpaceError) as caught:
            to_log_space(observed, noise)
        assert "first-order" in str(caught.value)

    def test_the_operating_range_is_far_below_the_threshold(self, noise):
        """The guard must not fire on a real configuration -- if it does, it is
        not a guard against mis-specification but a limit on observing."""
        assert noise.fractional < FIRST_ORDER_MAX_FRACTIONAL / 10


class TestTheExits:
    def test_wiener_matches_the_closed_form(self, space, pipeline, state, observed, noise):
        block = log_linear_operator(space, pipeline, state, "log_gain")
        y, sigma = to_log_space(observed, noise)
        estimate, _ = wiener_solve(block, y, noise_std=sigma, prior_std=PRIOR_STD)
        expected, _ = _closed_form(y, noise.fractional)
        assert jnp.allclose(estimate, expected, rtol=1e-4)

    def test_the_estimate_recovers_the_truth_within_the_noise(
        self, space, pipeline, state, observed, noise
    ):
        block = log_linear_operator(space, pipeline, state, "log_gain")
        y, sigma = to_log_space(observed, noise)
        estimate, _ = wiener_solve(block, y, noise_std=sigma, prior_std=PRIOR_STD)
        _, sd = _closed_form(y, noise.fractional)
        assert jnp.max(jnp.abs(estimate - LOG_G)) < 4.0 * sd

    def test_the_draw_has_the_posterior_SECOND_moment(
        self, space, pipeline, state, observed, noise
    ):
        """What separates a constrained realization from mean-plus-noise.

        4000 draws; the sample sd of each time sample is compared against the
        closed-form posterior sd, with a tolerance set by the sd of an sd
        estimate over that many draws (1/sqrt(2N) ~ 1.1 %, allowed 8 %).
        """
        block = log_linear_operator(space, pipeline, state, "log_gain")
        y, sigma = to_log_space(observed, noise)
        keys = jax.random.split(jax.random.key(4), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(
                block, y, noise_std=sigma, prior_std=PRIOR_STD, key=k
            )[0]
        )(keys)
        _, sd = _closed_form(y, noise.fractional)
        assert jnp.allclose(jnp.std(draws, axis=0), sd, rtol=0.08)

    def test_the_draw_mean_converges_to_the_wiener_mean(
        self, space, pipeline, state, observed, noise
    ):
        block = log_linear_operator(space, pipeline, state, "log_gain")
        y, sigma = to_log_space(observed, noise)
        estimate, _ = wiener_solve(block, y, noise_std=sigma, prior_std=PRIOR_STD)
        keys = jax.random.split(jax.random.key(5), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(
                block, y, noise_std=sigma, prior_std=PRIOR_STD, key=k
            )[0]
        )(keys)
        _, sd = _closed_form(y, noise.fractional)
        assert jnp.all(jnp.abs(jnp.mean(draws, axis=0) - estimate) < 4.0 * sd / 4000**0.5)


class TestThePlan:
    """The third engine, driven through SamplingPlan rather than by hand."""

    @staticmethod
    def _space_with_prior() -> ParameterSpace:
        return ParameterSpace(
            latents=[
                Latent(
                    "log_gain",
                    init=jnp.zeros(N_TIME),
                    prior=dist.Normal(jnp.zeros(N_TIME), PRIOR_STD),
                )
            ],
            bindings=[Bind("log_gain", into=lambda p: p["gain"].gain, fn=jnp.exp)],
        )

    def test_the_plan_recovers_the_truth_through_a_log_conjugate_block(
        self, pipeline, state, observed, noise
    ):
        plan = SamplingPlan(
            self._space_with_prior(), Block("log_gain", engine="log_conjugate")
        )
        assert plan.engines == {("log_gain",): "log_conjugate"}
        estimate = plan.estimate(pipeline, state, observed, noise=noise)
        assert jnp.max(jnp.abs(estimate.values["log_gain"] - LOG_G)) < 0.02

    def test_the_plan_draws_through_a_log_conjugate_block(
        self, pipeline, state, observed, noise
    ):
        plan = SamplingPlan(
            self._space_with_prior(), Block("log_gain", engine="log_conjugate")
        )
        draws = plan.sample(
            pipeline,
            state,
            observed,
            noise=noise,
            key=jax.random.key(9),
            n_sweeps=40,
        )
        assert jnp.max(jnp.abs(draws.mean["log_gain"] - LOG_G)) < 0.02
        # The block is closed-form, so nothing was tuned and the scatter is the
        # posterior's own -- compare against f/sqrt(n_freq), not a pinned number.
        assert jnp.all(draws.std["log_gain"] < 5.0 * noise.fractional / N_FREQ**0.5)

    def test_the_log_engine_gets_its_own_program_slot(
        self, pipeline, state, observed, noise
    ):
        """The cache key carries the engine, so the two spaces cannot cross-serve.

        Serving a log block from a linear block's slot would run one space's
        compiled program on the other's arrays: same names, same tolerances,
        same shapes, and a confident wrong answer with every guard green.
        ``test_conjugate_transition.py`` pins the ``'conjugate'`` half.
        """
        from rheplicant.inference.engines import Conditioning, log_conjugate_estimate

        space = self._space_with_prior()
        forward, values0 = space.forward_fn(pipeline, state)
        y, sigma = to_log_space(observed, noise)
        cond = Conditioning(
            space=space,
            pipeline=pipeline,
            state_template=state,
            observed=observed,
            noise=noise,
            forward=forward,
            log_observed=y,
            log_sigma=sigma,
        )
        programs: dict = {}
        for _ in range(3):
            log_conjugate_estimate(
                cond, ("log_gain",), values0, tol=1e-8, maxiter=None,
                require_convergence=None, programs=programs,
            )
        assert len(programs) == 1, "a key that grows per sweep never hits"
        assert next(iter(programs))[-1] == "log_conjugate"


class TestRefusals:
    def test_a_latent_declared_linear_is_refused_by_name(self, pipeline, state):
        """The two claims exclude each other, so this is caught before probing."""
        with pytest.raises(ParameterSpaceError) as caught:
            log_linear_operator(log_gain_space(linear=True), pipeline, state, "log_gain")
        assert "linear=True" in str(caught.value)
        assert "linear_operator" in str(caught.value)

    def test_a_gain_entered_directly_changes_sign(self, pipeline, state):
        """A directly-entered gain is caught by the SIGN, before affinity is asked.

        Probing an affine gain about zero sends the prediction negative, and
        that is a sharper diagnosis than a departure-from-affinity number: a
        log-linear model predicts exp(affine), which cannot change sign. The
        affinity path is exercised by the next test instead, on a model that
        stays positive.
        """
        with pytest.raises(ParameterSpaceError) as caught:
            check_log_linearity(linear_gain_space(), pipeline, state, "gain")
        message = str(caught.value)
        assert "at the block's zero" in message
        assert "linear_operator" in message
        # The remedy must NOT be the overflow one: nothing has been perturbed
        # here, so sending the reader to look at dtypes would be a confident
        # wrong diagnosis.
        assert "overflow" not in message

    def test_an_additive_term_AFTER_the_gain_breaks_log_linearity(self, space, state):
        """Positive everywhere, and still not log-linear -- so the affinity check
        is what has to catch it.

        Physically the case to know: ``d = g S`` is log-linear in ``log g`` and
        ``d = g S + C`` is not, because ``log`` does not distribute over the
        sum. A receiver term downstream of the gain is enough to take a block
        out of log space, while a sky term UPSTREAM of it (SummedSky, above)
        is not.
        """
        offset_pipeline = Pipeline(
            SummedSky(t_ant=T_ANT, t_nw=T_NW, tone=TONE),
            GainOperator(gain=jnp.ones(N_TIME)),
            AddConstant(level=500.0),
            names=("sky", "gain", "receiver"),
        )
        with pytest.raises(LinearityRefused) as caught:
            check_log_linearity(space, offset_pipeline, state, "log_gain")
        assert "not affine" in str(caught.value)

    def test_an_unnamed_latent_is_refused(self, space, pipeline, state):
        """There is no ``log_linear=True`` to infer from, so nothing is guessed."""
        with pytest.raises(ParameterSpaceError) as caught:
            log_linear_operator(space, pipeline, state)
        assert "name=" in str(caught.value)

    def test_a_probe_that_overflows_the_exponential_says_so(
        self, space, pipeline, state
    ):
        """Not a modelling failure, and the message must not claim it is.

        The map stays exactly affine in log space where ``exp`` of the probe
        leaves the dtype's range; this is why the log check does not inherit
        the linear check's 1e3 probe.
        """
        with pytest.raises(ParameterSpaceError) as caught:
            check_log_linearity(space, pipeline, state, "log_gain", scales=(1e3,))
        message = str(caught.value)
        assert "not finite" in message or "zero" in message
        assert "overflow" in message
