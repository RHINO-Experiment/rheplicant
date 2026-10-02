"""``check_linearity=`` on SamplingPlan: declining the up-front affinity check.

A plan checks each conjugate block's ``linear=True`` claim once, before its
first sweep, at 1e-3, 1 and 1e3 times each latent's prior width. A stage that
saturates is affine below its limit and not above it, so a model holding one
is refused at the outermost probe even when no sample of the data, and no
value the posterior gives weight to, is near the limit.
``check_linearity=False`` is how a caller declines that check, as
``check=False`` is for ``linear_operator`` and ``check: false`` for the
conjugate run kinds.

The fixture is a spectrum scaled per time sample and passed through a clip at
ten prior widths: 5 times by 7 frequencies, so a transposed axis cannot pass.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import LinearityRefused, ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Block,
    Draws,
    Estimate,
    Latent,
    ParameterSpace,
    SamplingPlan,
    check_linearity,
)

N_TIME, N_FREQ = 5, 7
PRIOR_STD = 10.0
NOISE = 0.5
#: Ten prior widths: above every value the data or the posterior reaches, and
#: below the check's outermost probe at a thousand.
CLIP = 100.0

PROFILE = 1.0 + 0.3 * jnp.cos(jnp.linspace(0.0, 2.0, N_FREQ))
TRUE_AMP = jnp.array([4.0, -3.0, 5.5, 2.0, -1.5])


class Spectrum(AbstractOperator):
    """``data[t, f] = (amp[t] + extra[t]) * PROFILE[f]``.

    ``extra`` is a second latent slot that enters exactly as ``amp`` does, so
    a space binding both is degenerate by construction.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    amp: jax.Array
    extra: jax.Array

    def __call__(self, state):
        return state.with_data((self.amp + self.extra)[:, None] * PROFILE[None, :])


class Clip(AbstractOperator):
    """A converter's saturation with nothing else: identity below ``limit``."""

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    limit: float

    def __call__(self, state):
        return state.with_data(jnp.clip(state.data, -self.limit, self.limit))


def _spectrum() -> Spectrum:
    return Spectrum(amp=jnp.zeros(N_TIME), extra=jnp.zeros(N_TIME))


@pytest.fixture
def state():
    return State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float), freq=jnp.linspace(60e6, 85e6, N_FREQ)
        ),
        meta={"telescope": "RHINO", "obs_id": "clip-000"},
    )


@pytest.fixture
def unclipped():
    return Pipeline(_spectrum(), names=("spectrum",))


@pytest.fixture
def clipped():
    return Pipeline(_spectrum(), Clip(limit=CLIP), names=("spectrum", "clip"))


def _latent(name: str) -> Latent:
    return Latent(
        name,
        init=jnp.zeros(N_TIME),
        linear=True,
        prior=dist.Normal(jnp.zeros(N_TIME), PRIOR_STD),
    )


@pytest.fixture
def space():
    return ParameterSpace(
        latents=[_latent("amp")],
        bindings=[Bind("amp", into=lambda p: p["spectrum"].amp)],
    )


@pytest.fixture
def degenerate_space():
    """``amp`` and ``extra`` reach the data only through their sum."""
    return ParameterSpace(
        latents=[_latent("amp"), _latent("extra")],
        bindings=[
            Bind("amp", into=lambda p: p["spectrum"].amp),
            Bind("extra", into=lambda p: p["spectrum"].extra),
        ],
    )


@pytest.fixture
def observed():
    truth = TRUE_AMP[:, None] * PROFILE[None, :]
    return truth + NOISE * jax.random.normal(jax.random.key(3), (N_TIME, N_FREQ))


def _posterior_mean(observed) -> jax.Array:
    """The conjugate posterior mean of ``amp``, per time sample, in closed form."""
    precision = jnp.sum(PROFILE**2) / NOISE**2 + 1.0 / PRIOR_STD**2
    return (observed @ PROFILE) / NOISE**2 / precision


def _sample(plan, pipeline, state, observed, **kwargs) -> Draws:
    return plan.sample(
        pipeline,
        state,
        observed,
        noise=NOISE,
        key=jax.random.key(11),
        n_sweeps=8,
        warmup=4,
        **kwargs,
    )


class TestTheFixture:
    def test_the_data_are_far_below_the_clip(self, observed):
        assert float(jnp.max(jnp.abs(observed))) < 0.1 * CLIP

    def test_the_clip_lies_between_the_inner_and_the_outer_probe(self, space, clipped, state):
        """Affine out to three prior widths, and not at a thousand.

        Both halves are asserted, because the tests below read a refusal as
        "the outer probe reached the clip" and a run as "nothing else is
        wrong with the claim".
        """
        inner = check_linearity(space, clipped, state, names=("amp",), scales=(1e-3, 1.0, 3.0))
        assert max(float(value) for value in inner.values()) < 1e-4
        with pytest.raises(LinearityRefused):
            check_linearity(space, clipped, state, names=("amp",))


class TestTheDefault:
    def test_estimate_refuses_a_block_that_is_affine_only_below_a_clip(
        self, space, clipped, state, observed
    ):
        plan = SamplingPlan(space, Block("amp"))
        with pytest.raises(LinearityRefused):
            plan.estimate(clipped, state, observed, noise=NOISE)

    def test_sample_refuses_it_too(self, space, clipped, state, observed):
        plan = SamplingPlan(space, Block("amp"))
        with pytest.raises(LinearityRefused):
            _sample(plan, clipped, state, observed)

    def test_the_shared_preparation_checks_unless_told_not_to(
        self, space, clipped, state, observed
    ):
        """``_prepare`` is what both exits call, and tests call it with six arguments.

        Its seventh defaults to checking, so a caller that does not know the
        keyword gets the check.
        """
        plan = SamplingPlan(space, Block("amp"))
        with pytest.raises(LinearityRefused):
            plan._prepare(clipped, state, observed, NOISE, False, "probe")
        cond, values = plan._prepare(clipped, state, observed, NOISE, False, "probe", False)
        assert set(values) == {"amp"}
        assert cond.pipeline is clipped

    def test_true_is_the_default_spelled_out(self, space, clipped, state, observed):
        plan = SamplingPlan(space, Block("amp"))
        with pytest.raises(LinearityRefused):
            plan.estimate(clipped, state, observed, noise=NOISE, check_linearity=True)
        with pytest.raises(LinearityRefused):
            _sample(plan, clipped, state, observed, check_linearity=True)


class TestDeclined:
    def test_estimate_gives_the_closed_form_posterior_mean(self, space, clipped, state, observed):
        plan = SamplingPlan(space, Block("amp"))
        estimate = plan.estimate(clipped, state, observed, noise=NOISE, check_linearity=False)
        assert isinstance(estimate, Estimate)
        np.testing.assert_allclose(estimate.values["amp"], _posterior_mean(observed), rtol=1e-4)

    def test_estimate_is_the_unclipped_models_estimate(
        self, space, clipped, unclipped, state, observed
    ):
        """Below the limit the clip is the identity, so nothing else may move."""
        plan = SamplingPlan(space, Block("amp"))
        declined = plan.estimate(clipped, state, observed, noise=NOISE, check_linearity=False)
        reference = plan.estimate(unclipped, state, observed, noise=NOISE)
        np.testing.assert_allclose(declined.values["amp"], reference.values["amp"], rtol=1e-6)
        assert declined.diagnostics.sweeps == reference.diagnostics.sweeps

    def test_sample_draws_what_the_unclipped_model_draws(
        self, space, clipped, unclipped, state, observed
    ):
        plan = SamplingPlan(space, Block("amp"))
        declined = _sample(plan, clipped, state, observed, check_linearity=False)
        reference = _sample(plan, unclipped, state, observed)
        assert declined.n_draw == 4
        np.testing.assert_allclose(declined.samples["amp"], reference.samples["amp"], rtol=1e-6)

    @pytest.mark.parametrize("exit_name", ["estimate", "sample"])
    def test_the_identifiability_check_still_runs(
        self, degenerate_space, clipped, state, observed, exit_name
    ):
        """One check is declined, not both.

        The linearity check runs first, so by default a clipped model that is
        also degenerate reports the clip. With it declined the rank test is
        reached and names the null space.
        """
        plan = SamplingPlan(degenerate_space, Block("amp", "extra"))
        run = {
            "estimate": lambda **kw: plan.estimate(clipped, state, observed, noise=NOISE, **kw),
            "sample": lambda **kw: _sample(plan, clipped, state, observed, **kw),
        }[exit_name]
        with pytest.raises(LinearityRefused):
            run()
        # A literal, so the refusal census records the sentence and not "<computed>".
        assert (N_TIME, 2 * N_TIME) == (5, 10)
        with pytest.raises(ParameterSpaceError, match="nullity 5 of 10"):
            run(check_linearity=False)


class TestTheArgument:
    @pytest.mark.parametrize("value", [None, 0, 1, "once", "false", "skip"])
    def test_estimate_takes_a_bool_and_nothing_else(self, space, clipped, state, observed, value):
        plan = SamplingPlan(space, Block("amp"))
        with pytest.raises(ParameterSpaceError, match="check_linearity") as refusal:
            plan.estimate(clipped, state, observed, noise=NOISE, check_linearity=value)
        assert not isinstance(refusal.value, LinearityRefused)
        assert repr(value) in str(refusal.value)

    @pytest.mark.parametrize("value", [None, 0, 1, "once", "false", "skip"])
    def test_sample_takes_a_bool_and_nothing_else(self, space, clipped, state, observed, value):
        plan = SamplingPlan(space, Block("amp"))
        with pytest.raises(ParameterSpaceError, match="check_linearity") as refusal:
            _sample(plan, clipped, state, observed, check_linearity=value)
        assert not isinstance(refusal.value, LinearityRefused)
        assert repr(value) in str(refusal.value)


class TestTheCurvatureFloor:
    """The certificate's floor is a verified bound, and a declined check verifies nothing.

    Above ``bayesmith.optimize.certify.DENSE_MAX`` latents an estimate
    certifies with the prior precision as a floor on the joint Hessian, on
    the condition that the prediction is affine in every latent. One conjugate
    block met that condition through the check before the first sweep. With
    the check declined nothing has, so there is no floor, for one block as for
    several.
    """

    def test_a_checked_block_has_the_prior_precision_as_its_floor(
        self, space, unclipped, state, observed
    ):
        plan = SamplingPlan(space, Block("amp"))
        cond, _ = plan._prepare(unclipped, state, observed, NOISE, False, "probe")
        assert plan._curvature_floor(cond) == pytest.approx(1.0 / PRIOR_STD**2)

    @pytest.mark.parametrize("model", ["clipped", "unclipped"])
    def test_a_declined_check_gives_no_floor(self, space, state, observed, model, request):
        pipeline = request.getfixturevalue(model)
        plan = SamplingPlan(space, Block("amp"))
        cond, _ = plan._prepare(pipeline, state, observed, NOISE, False, "probe", False)
        assert plan._curvature_floor(cond) is None

    def test_two_declined_blocks_are_not_asked_to_be_jointly_affine(
        self, degenerate_space, clipped, state, observed, monkeypatch
    ):
        """No floor, and no check at the default scales run behind the caller's back."""
        plan = SamplingPlan(degenerate_space, Block("amp"), Block("extra"))
        cond, _ = plan._prepare(clipped, state, observed, NOISE, False, "probe", False)
        asked = []
        monkeypatch.setattr(type(plan), "_jointly_affine", lambda self, c: asked.append(c) or True)
        assert plan._curvature_floor(cond) is None
        assert asked == []

    def test_above_the_dense_limit_a_declined_estimate_does_not_certify(
        self, space, clipped, unclipped, state, observed, monkeypatch
    ):
        """What the missing floor costs, with the limit lowered so five latents exceed it.

        Checked, the estimate certifies on the prior precision. Declined, it
        has no proof to certify with and refuses at ``max_iter``; with
        ``tol=None`` it makes no claim and returns.
        """
        from bayesmith.optimize import certify

        monkeypatch.setattr(certify, "DENSE_MAX", 2)
        plan = SamplingPlan(space, Block("amp"))
        checked = plan.estimate(unclipped, state, observed, noise=NOISE)
        assert checked.diagnostics.converged is True
        with pytest.raises(ParameterSpaceError) as refusal:
            plan.estimate(clipped, state, observed, noise=NOISE, max_iter=6, check_linearity=False)
        assert not isinstance(refusal.value, LinearityRefused)
        unclaimed = plan.estimate(
            clipped, state, observed, noise=NOISE, max_iter=6, tol=None, check_linearity=False
        )
        assert unclaimed.diagnostics.converged is None
        np.testing.assert_allclose(unclaimed.values["amp"], _posterior_mean(observed), rtol=1e-4)
