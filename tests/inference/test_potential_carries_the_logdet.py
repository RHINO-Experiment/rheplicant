"""The conditional potential is a density, so it carries ``sum log sigma``.

This is migration ledger **B1**. ``conditional_potential`` was
``0.5 * chi2 - log_prior``, and ``Conditioning.chi2`` is
``sum r^2/sigma^2`` with sigma evaluated at the current prediction but with
no ``sum log sigma`` beside it. When sigma depends on the prediction that
omission is not a constant, so the gradient engine descended -- and NUTS
sampled -- a different function from the one
:func:`~rheplicant.inference.numpyro_bridge.to_numpyro_model` builds, whose
``dist.Normal(prediction, sigma)`` carries ``-log sigma`` inside
``log_prob``. Same declared model, two targets, no guard between them.

**What is asserted here is an identity, not a landing place.** The potential
must differ from ``-(log likelihood + log prior)`` by a CONSTANT, which the
tests check by comparing differences between two probe points. That is
exact, fixture-independent, and free of the reparameterisation the obvious
test would drag in: the B1 model declares ``w`` with ``mu = exp(w) x``, so a
``Normal`` prior on ``w`` is a ``1/scale`` prior on the recovered scale, and
an argmin pinned against the closed form would be pinning that Jacobian too
(measured: the corrected optimum is 4.7327 against a closed form of 4.8651,
and the 2.7% between them is all prior).

The consequence a reader cares about is asserted separately, and as a SIDE:
the optimum moves off the GLS-type closed form and onto the unbiased one.

Every case here has an anti-vacuity twin under
:class:`~rheplicant.inference.noise.HomoscedasticNoise`, where ``sum log
sigma`` *is* a constant and the identity therefore held before the fix as
well. Without those twins a test asserting "the potential changed" would
pass for any change at all.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.engines import (
    Conditioning,
    _potential_of,
    conditional_potential,
)
from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    NoiseModelLikelihood,
    RadiometerNoise,
)

dist = pytest.importorskip("numpyro.distributions", reason="numpyro not installed")

N = 40
KAPPA = 0.5
W_TRUE = 5.0
PRIOR_STD = 100.0

#: ``RadiometerNoise`` gives ``sigma = prediction / sqrt(channel_width *
#: integration_time)``, so this is the ``sigma = KAPPA * |mu|`` of §三 B1.
PREDICTION_DEPENDENT = RadiometerNoise(
    channel_width=1.0 / KAPPA**2, integration_time=1.0
)


class _Exponential(AbstractOperator):
    """``mu = exp(w) * x`` -- nonlinear in the declared latent, so a GRADIENT block."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    w: jax.Array
    x: jax.Array

    def __call__(self, state):
        return state.with_data(jnp.exp(self.w) * self.x[None, :])


def _model_parts():
    """The model §三 B1 works in, shared by the two routes compared below.

    Returned rather than rebuilt per route on purpose: the claim is that ONE
    declared model reaches two exits, so two exits built from two models would
    not be testing it.
    """
    x = jnp.linspace(1.0, 3.0, N)
    truth = W_TRUE * x
    data = truth + KAPPA * jnp.abs(truth) * jax.random.normal(jax.random.key(3), (N,))
    pipeline = Pipeline(_Exponential(w=jnp.array(1.0), x=x), names=("line",))
    space = ParameterSpace(
        latents=[
            Latent(
                "w",
                init=jnp.array(1.0),
                linear=False,
                prior=dist.Normal(0.0, PRIOR_STD),
            )
        ],
        bindings=[Bind("w", into=lambda p: p["line"].w)],
    )
    template = State(
        data=jnp.zeros((1, N)),
        coords=Coordinates(time=jnp.arange(1.0), freq=jnp.linspace(60e6, 80e6, N)),
    )
    return pipeline, space, template, data


def _conditioning(noise):
    """One gradient block over ``w``, on that model."""
    pipeline, space, template, data = _model_parts()
    forward, _ = space.forward_fn(pipeline, template)
    cond = Conditioning(
        space=space,
        pipeline=pipeline,
        state_template=template,
        observed=data[None, :],
        noise=noise,
        forward=forward,
    )
    return cond, {"w": jnp.array(1.0)}


def _negative_log_joint(cond, probe):
    """``-(log p(d | theta) + log p(theta))`` from the package's own likelihood.

    :class:`NoiseModelLikelihood` is where this repository already spells the
    Gaussian log-density under a noise model, masking included; asking it
    rather than re-writing ``sum log sigma`` here is what keeps the test from
    becoming a second copy of the thing it guards.
    """
    likelihood = NoiseModelLikelihood(cond.noise)
    prediction = cond.forward(probe)
    prior = cond.space.latent("w").prior.log_prob(probe["w"]).sum()
    return -(likelihood(prediction, cond.observed) + prior)


#: Two points far enough apart that the omitted term differs between them by
#: much more than float32 roundoff, and close enough that both are on the same
#: side of the optimum. ``sum log sigma`` differs by ``N * 0.4`` = 16 nats here.
PROBE_A = {"w": jnp.array(1.4)}
PROBE_B = {"w": jnp.array(1.8)}

#: float32. The quantities are ~1e2 and the difference under test is ~16, so a
#: relative tolerance on the DIFFERENCE is what has room; see the module
#: docstring in ``test_gradient_transition.py`` for why x64 is not switched on.
RTOL = 2e-4


class TestThePotentialIsTheNegativeLogJoint:
    """Up to a constant -- checked on differences, so the constant cancels."""

    @pytest.mark.parametrize(
        "noise, label",
        [
            (PREDICTION_DEPENDENT, "sigma depends on the prediction"),
            (HomoscedasticNoise(2.5), "sigma does not (anti-vacuity twin)"),
        ],
    )
    def test_conditional_potential_matches_it(self, noise, label):
        cond, values = _conditioning(noise)
        potential = conditional_potential(cond, ("w",), values)
        theirs = float(potential(PROBE_A)) - float(potential(PROBE_B))
        ours = float(_negative_log_joint(cond, PROBE_A)) - float(
            _negative_log_joint(cond, PROBE_B)
        )
        assert ours == pytest.approx(theirs, rel=RTOL), label

    def test_the_two_probes_actually_separate_the_omitted_term(self):
        """Anti-vacuity for the anti-vacuity twin: the term is not ~0 here.

        If ``sum log sigma`` happened to agree at both probes the identity
        above would hold whether or not the term were present, and every case
        in this file would be decorative.
        """
        cond, _ = _conditioning(PREDICTION_DEPENDENT)
        gap = float(
            jnp.sum(jnp.log(cond.sigma(PROBE_A))) - jnp.sum(jnp.log(cond.sigma(PROBE_B)))
        )
        assert abs(gap) > 1.0, f"the two probes differ by only {gap} nats of log-det"


class TestBothPotentialBuildersTargetOneDistribution:
    """``_potential_of`` is what NUTS gets; ``conditional_potential`` is what Adam gets.

    B1 is a two-exits-two-targets defect, so fixing one builder and not the
    other would reproduce it one layer down rather than close it.
    """

    @pytest.mark.parametrize(
        "noise", [PREDICTION_DEPENDENT, HomoscedasticNoise(2.5)]
    )
    def test_they_agree_value_for_value(self, noise):
        cond, values = _conditioning(noise)
        closed = conditional_potential(cond, ("w",), values)
        lifted = _potential_of(cond, ("w",))
        others = {k: v for k, v in values.items() if k != "w"}
        for probe in (PROBE_A, PROBE_B):
            assert float(lifted(others, probe)) == pytest.approx(
                float(closed(probe)), rel=1e-6
            )


class TestAFlaggedSampleContributesNothing:
    """``log sigma`` at an infinite sigma is ``inf``, and must be masked to zero.

    Taking the limit is not available: ``r^2/sigma^2 -> 0`` but ``log sigma ->
    inf``, so one flagged channel would send the whole potential to ``inf``
    and every draw with it. :class:`NoiseModelLikelihood` already states this;
    the point here is that the potential inherits it.
    """

    def test_the_potential_stays_finite(self):
        flags = jnp.zeros((1, N), dtype=bool).at[0, ::7].set(True)
        cond, values = _conditioning(FlaggedNoise(PREDICTION_DEPENDENT, flags))
        potential = conditional_potential(cond, ("w",), values)
        assert jnp.isfinite(potential(PROBE_A))

    def test_and_the_flagged_samples_are_the_only_difference(self):
        """Masking, not clamping: the unflagged samples must be untouched.

        The expected gap is the flagged samples' own term in the potential's
        own convention -- ``0.5 r^2/sigma^2 + log sigma``, without the
        ``0.5 log 2 pi`` per sample that
        :class:`~rheplicant.inference.noise.NoiseModelLikelihood` carries and
        this potential deliberately does not. Written longhand rather than
        asked of either, because both are under test here.

        That constant is worth knowing by size: a draft of this test used the
        ``NoiseModelLikelihood`` convention and was off by 5.51 nats, which is
        exactly ``6 * 0.5 * log 2 pi`` for the six flagged samples. A
        discrepancy that resolves to a named constant is a convention
        mismatch, not a defect -- and the repair that "fixes" it in the code
        would have removed ``2 pi`` from a Gaussian.

        What this case actually guards is the MASKING, which is independent of
        that convention: flagged samples contribute their own term and no
        other sample is touched.
        """
        flags = jnp.zeros((1, N), dtype=bool).at[0, ::7].set(True)
        bare, values = _conditioning(PREDICTION_DEPENDENT)
        flagged, _ = _conditioning(FlaggedNoise(PREDICTION_DEPENDENT, flags))
        sigma = bare.sigma(PROBE_A)
        residual = (bare.observed - bare.forward(PROBE_A)) / sigma
        per_sample = 0.5 * residual**2 + jnp.log(sigma)
        dropped = float(jnp.sum(jnp.where(flags, per_sample, 0.0)))
        gap = float(conditional_potential(bare, ("w",), values)(PROBE_A)) - float(
            conditional_potential(flagged, ("w",), values)(PROBE_A)
        )
        assert gap == pytest.approx(dropped, rel=1e-3)
        assert int(jnp.sum(flags)) == 6, "the mask must actually flag something"


class TestTheOptimumMovesOntoTheUnbiasedSide:
    """The consequence, stated as a side rather than a value.

    The two closed forms are ``sum d^2/x^2 / sum d/x`` (log-determinant
    dropped, GLS-type) and ``mean(d/x)`` (kept, exactly unbiased), and they
    are 21% apart on this fixture. Which one the potential's optimum sits
    NEAR is the whole of B1; how near is the prior's business.
    """

    def _closed_forms(self, cond):
        x = np.asarray(jnp.linspace(1.0, 3.0, N), dtype=float)
        d = np.asarray(cond.observed[0], dtype=float)
        return float(np.sum(d**2 / x**2) / np.sum(d / x)), float(np.mean(d / x))

    def test_the_two_targets_are_far_apart_on_this_fixture(self):
        cond, _ = _conditioning(PREDICTION_DEPENDENT)
        dropped, kept = self._closed_forms(cond)
        assert abs(dropped - kept) / kept > 0.15, (
            "the fixture no longer separates the two estimators, so nothing "
            "below can tell them apart"
        )

    def test_the_potential_prefers_the_unbiased_one(self):
        from scipy.optimize import minimize_scalar

        cond, values = _conditioning(PREDICTION_DEPENDENT)
        potential = conditional_potential(cond, ("w",), values)
        dropped, kept = self._closed_forms(cond)
        best = float(
            np.exp(
                minimize_scalar(
                    lambda w: float(potential({"w": jnp.array(float(w))})),
                    bracket=(0.5, 1.6, 3.0),
                ).x
            )
        )
        assert abs(best - kept) < abs(best - dropped), (
            f"the optimum {best:.4f} is nearer the GLS-type {dropped:.4f} than "
            f"the unbiased {kept:.4f}, so the log-determinant is still missing"
        )


class TestTheGradientBlockAndTheBridgeAgree:
    """The seam B1 names, asserted directly.

    ``to_numpyro_model``'s observation site is
    ``dist.Normal(prediction, sigma)``, whose ``log_prob`` carries ``-log
    sigma`` automatically -- so the ``nuts`` exit has always sampled the full
    density. The gradient block's potential did not, and nothing compared
    them. This is that comparison: one declared model, two routes to a
    log-density, which must differ by a CONSTANT.

    Measured before the fix, on the prediction-dependent case: 16.0 nats of
    disagreement between the two probes, which is ``N * log(sigma_A /
    sigma_B)`` exactly.
    """

    def _both(self, noise, probe):
        from numpyro.infer.util import log_density

        from rheplicant.inference.numpyro_bridge import to_numpyro_model

        pipeline, space, template, data = _model_parts()
        forward, _ = space.forward_fn(pipeline, template)
        cond = Conditioning(
            space=space,
            pipeline=pipeline,
            state_template=template,
            observed=data[None, :],
            noise=noise,
            forward=forward,
        )
        ours = float(conditional_potential(cond, ("w",), {"w": jnp.array(1.0)})(probe))
        model = to_numpyro_model(pipeline, template, space, noise_std=noise)
        theirs = -float(
            log_density(model, (), {"observed": data[None, :]}, probe)[0]
        )
        return ours, theirs

    @pytest.mark.parametrize(
        "noise, label",
        [
            (PREDICTION_DEPENDENT, "sigma depends on the prediction"),
            (HomoscedasticNoise(2.5), "sigma does not (anti-vacuity twin)"),
        ],
    )
    def test_the_two_routes_differ_by_a_constant(self, noise, label):
        ours_a, theirs_a = self._both(noise, PROBE_A)
        ours_b, theirs_b = self._both(noise, PROBE_B)
        assert (ours_a - ours_b) == pytest.approx(
            theirs_a - theirs_b, rel=RTOL
        ), label

    def test_and_that_constant_is_the_2pi_the_potential_drops(self):
        """Pinning the offset too, so "a constant" cannot hide a second term.

        The potential omits ``0.5 n log 2 pi`` deliberately; the bridge keeps
        it. Anything else appearing in the gap would satisfy the case above
        and fail here.
        """
        ours, theirs = self._both(PREDICTION_DEPENDENT, PROBE_A)
        assert theirs - ours == pytest.approx(
            0.5 * N * float(np.log(2.0 * np.pi)), rel=1e-4
        )
