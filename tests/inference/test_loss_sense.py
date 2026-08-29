"""A scoring function handed to a minimizer must have its optimum downhill.

The failure this file pins: a log-density and an error have the same
signature, ``(prediction, observed) -> scalar``, so ``isinstance(x, Likelihood)``
is ``True`` for both and the ``loss_fn=`` slot accepts either. Minimizing a
log-density descends a function unbounded below, and the loss history -- the
only evidence the user has -- looks like textbook convergence the whole way.

Measured before the guard, on a one-parameter gain fit with truth ``g = 1.0``:

    mean_squared_error       ->  g = +0.9999    loss  2499  ->  0.002617
    GaussianLikelihood(0.05) ->  g = -30.7349   loss -3.2e7 -> -1.3e11

Two mechanisms, and the tests below keep them honest about which one fires.
The declaration (``sense``) is exact and gives the better message, but it is a
whitelist and therefore wrong about code it has not met. The measurement --
score the PERFECT prediction, ``loss_fn(observed, observed)``, and check it is
not better than the score at the start -- needs no declaration at all, and is
what covers a user's own class, a bare lambda, and the one likelihood in the
package that does not declare a sense.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.calibrate import AdamCalibrator, GradientCalibrator
from rheplicant.inference.likelihood import (
    GaussianLikelihood,
    MaskedGaussianLikelihood,
    mean_squared_error,
    sense_of,
)

TRUTH = 1.0
START = 3.0


def forward(params):
    """Asymmetric and every element distinct.

    A constant or symmetric target would let a transposed or partially applied
    prediction score identically to the right one.
    """
    return params["g"] * jnp.arange(1.0, 51.0).reshape(5, 10)


OBSERVED = forward({"g": jnp.array(TRUTH)})
PARAMS0 = {"g": jnp.array(START)}

#: Cheap instances for the refusal tests, which never reach a step: the guard
#: runs before ``lax.scan`` and the step count is irrelevant to them.
CALIBRATORS = [
    pytest.param(GradientCalibrator(learning_rate=1e-4, n_steps=5), id="gradient"),
    pytest.param(AdamCalibrator(learning_rate=1e-2, n_steps=5), id="adam"),
]

#: Settings that actually converge on this fixture, measured rather than
#: guessed. Adam at ``lr=1e-2, n_steps=300`` reaches only g = 1.179 from a
#: start of 3.0 -- Adam's step is roughly ``lr`` per iteration, so the budget
#: has to cover the distance.
CONVERGING = [
    pytest.param(GradientCalibrator(learning_rate=1e-4, n_steps=300), id="gradient"),
    pytest.param(AdamCalibrator(learning_rate=5e-2, n_steps=400), id="adam"),
]


class TestTheControlStillWorks:
    """Before asserting what is refused, assert that the ordinary case is not.

    A guard that refused everything would pass every test below it.
    """

    @pytest.mark.parametrize("calibrator", CONVERGING)
    def test_mean_squared_error_is_accepted_and_finds_the_truth(self, calibrator):
        fit, losses = calibrator.fit(forward, PARAMS0, OBSERVED, loss_fn=mean_squared_error)
        assert float(fit["g"]) == pytest.approx(TRUTH, rel=1e-3)
        assert float(losses[-1]) < float(losses[0])

    def test_a_negated_likelihood_is_accepted_and_finds_the_truth(self):
        """The remedy the error message offers has to actually work.

        Adam only, and that is a finding rather than a convenience. Negating a
        Gaussian log-density does not merely flip the sign: it rescales the
        objective by ``n / (2 sigma^2)`` relative to a *mean* squared error --
        here 50/2 = 25x even at sigma = 1, and 10000x at the sigma = 0.05 the
        original measurement used. A fixed-step descent tuned for MSE diverges
        to NaN on it (measured). Adam normalises by the gradient's own scale
        and is indifferent.

        So the message's advice is correct but incomplete for
        :class:`GradientCalibrator`, and a reader who hits it should know the
        learning rate must come with it.
        """
        likelihood = GaussianLikelihood(jnp.array(1.0))
        fit, _ = AdamCalibrator(learning_rate=5e-2, n_steps=400).fit(
            forward, PARAMS0, OBSERVED, loss_fn=lambda p, o: -likelihood(p, o)
        )
        assert float(fit["g"]) == pytest.approx(TRUTH, rel=1e-3)

    def test_the_fixed_step_descent_really_does_diverge_on_the_unscaled_negation(self):
        """Pinned so the note above is a measurement and not a memory.

        **The observation channel changed when `calibrate` switched, and the
        claim did not** (migration ledger D33, whose triage named this as the
        one test depending on a diverged fit being RETURNED). It used to read
        ``assert not jnp.isfinite(fit["g"])``: the calibrator handed back a
        NaN and this test caught it there. The fit now runs through
        ``bayesmith.optimize.minimize``, which refuses a point whose objective
        is not finite rather than returning it, so the divergence arrives as a
        raise.

        What is asserted is still exactly what the note above needs: that a
        fixed-step descent, at this rate, does not survive the unscaled
        negation. A NaN returned and a refusal raised are the same measurement
        of the same fact.
        """
        likelihood = GaussianLikelihood(jnp.array(1.0))
        with pytest.raises(Exception) as caught:
            GradientCalibrator(learning_rate=1e-4, n_steps=300).fit(
                forward, PARAMS0, OBSERVED, loss_fn=lambda p, o: -likelihood(p, o)
            )
        assert "not finite at the point this fit reached" in str(caught.value), (
            "the fit no longer diverges here, or it diverges into some other "
            "refusal -- either way the note above needs re-measuring"
        )


class TestRefusedByDeclaration:
    @pytest.mark.parametrize("calibrator", CALIBRATORS)
    @pytest.mark.parametrize(
        "likelihood",
        [
            pytest.param(GaussianLikelihood(jnp.array(0.05)), id="gaussian"),
            pytest.param(MaskedGaussianLikelihood(jnp.array(0.05)), id="masked"),
        ],
    )
    def test_a_shipped_likelihood_is_refused_by_name(self, calibrator, likelihood):
        with pytest.raises(ParameterSpaceError, match=r"declares sense='maximize'"):
            calibrator.fit(forward, PARAMS0, OBSERVED, loss_fn=likelihood)

    def test_a_nonsense_declaration_is_refused_rather_than_defaulted(self):
        """A typo must not read as the permissive default.

        ``sense = "maximise"`` (or ``"max"``, or ``"Maximize"``) would sail
        through a ``getattr(..., "minimize")`` and be minimized -- the exact
        failure the attribute was added to prevent, reintroduced by a spelling.
        """

        class Typo:
            sense = "maximise"

            def __call__(self, prediction, observed):
                return jnp.sum((prediction - observed) ** 2)

        with pytest.raises(ParameterSpaceError, match=r"must be 'maximize'"):
            sense_of(Typo())


class TestRefusedByMeasurement:
    """The half a whitelist cannot do.

    Nothing in these cases declares anything; each is caught because scoring
    the perfect prediction comes out ABOVE the score at the starting point.
    """

    @pytest.mark.parametrize("calibrator", CALIBRATORS)
    def test_a_bare_lambda_log_density_is_refused(self, calibrator):
        with pytest.raises(ParameterSpaceError, match=r"scores a PERFECT prediction"):
            calibrator.fit(
                forward,
                PARAMS0,
                OBSERVED,
                loss_fn=lambda p, o: -jnp.sum((p - o) ** 2),
            )

    def test_an_undeclared_user_class_is_refused(self):
        """The case that motivated measuring rather than listing.

        ``NoiseModelLikelihood`` in this package declares no ``sense`` either,
        so this is not a hypothetical about third-party code.
        """

        class UserLikelihood:
            def __call__(self, prediction, observed):
                return -0.5 * jnp.sum((prediction - observed) ** 2)

        with pytest.raises(ParameterSpaceError, match=r"scores a PERFECT prediction"):
            GradientCalibrator(n_steps=5).fit(
                forward, PARAMS0, OBSERVED, loss_fn=UserLikelihood()
            )

    def test_a_non_finite_score_is_refused_rather_than_judged(self):
        """NaN compares False against everything, including ``>``.

        Without this branch the sense test would silently read a NaN score as
        "the perfect prediction is not better, carry on" -- letting through
        precisely the case the guard exists for, whenever it arrives with a
        NaN attached. This is the fourth independent occurrence of that
        pattern in this package's history, which is why it gets its own test
        rather than a comment.
        """
        with pytest.raises(ParameterSpaceError, match=r"not finite at entry"):
            GradientCalibrator(n_steps=5).fit(
                forward,
                PARAMS0,
                OBSERVED,
                loss_fn=lambda p, o: jnp.sum(p - o) * jnp.nan,
            )


class TestTheWrapperSaysThisPackagesRoutes:
    """D11 migrated the detection; the remedy sentence could not go with it.

    ``bayesmith.optimize.check_loss_sense`` ends its declared-sense refusal
    with *"use a density-aware route (`fit`, `nuts`)"*. Neither name exists in
    rheplicant, so pointing a user there is worse than saying nothing, and the
    wrapper swaps that sentence for this package's own.

    **The swap is keyed on the far side's exact wording, which is exactly why
    it needs a test.** If upstream rewords that sentence the substitution stops
    matching, and it stops matching SILENTLY -- the refusal would simply go
    back to naming `fit` and `nuts`, still raising, still the right class,
    still passing every other case in this file.
    """

    def _refusal(self):
        likelihood = GaussianLikelihood(jnp.array(1.0))
        with pytest.raises(ParameterSpaceError) as caught:
            GradientCalibrator(n_steps=5).fit(
                forward, PARAMS0, OBSERVED, loss_fn=likelihood
            )
        return str(caught.value)

    def test_it_names_this_packages_routes(self):
        message = self._refusal()
        assert "numpyro_bridge" in message and "SamplingPlan" in message, message

    def test_and_not_the_far_sides(self):
        """The half with teeth: absence, which the case above cannot see."""
        message = self._refusal()
        assert "`fit`, `nuts`" not in message, (
            "the far side's remedy survived into this package's message -- the "
            "substitution in calibrate.py is keyed on upstream wording that "
            "has changed, and it failed silently"
        )

    def test_the_detection_itself_is_still_the_far_sides(self):
        """Anti-vacuity: the wrapper must not have quietly become a re-write.

        If the substitution were done by re-implementing the message here, both
        cases above would pass while nothing delegated at all. The far side's
        own opening clause is what proves the text came from over there.
        """
        assert "declares sense='maximize'" in self._refusal()


class TestTheGuardCostsNothingAtRunTime:
    def test_it_runs_once_at_entry_and_not_inside_the_scan(self):
        """Counted, because "outside the loop" is the claim the docstring makes.

        The optimizers run under ``lax.scan``; a check that traced per step
        would be re-run on every recompilation and would refuse only after the
        calibrator had been asked to run. Three calls: one from the guard's
        own probe at the start, one for the perfect prediction, and one from
        the first traced step of the loop.

        **The learning rate is spelled out, and that is not decoration.** This
        case used the calibrator's DEFAULT ``1e-2``, at which this fixture
        diverges: measured, ``losses[-1]`` is ``nan`` and the fitted ``g`` is
        not finite. The count assertion could not see it -- it counts
        ``loss_fn`` calls and never looks at the fit -- so the tracing claim
        was being measured on a run that produced NaN. ``1e-4`` is what the
        converging cases at the top of this file use, and the count is
        unchanged by the switch, which is the point: the claim was always
        about tracing, and it should be made on a fit that works.

        Found while probing D33 (a diverged fit refused rather than returned):
        with that refusal applied, this case raised instead of counting.
        """
        calls = []

        def counting_loss(prediction, observed):
            calls.append(1)
            return jnp.mean((prediction - observed) ** 2)

        fit, losses = GradientCalibrator(learning_rate=1e-4, n_steps=50).fit(
            forward, PARAMS0, OBSERVED, loss_fn=counting_loss
        )
        assert jnp.isfinite(fit["g"]), (
            "the fixture diverged again -- the count below would still pass, "
            "which is how this went unnoticed at the default learning rate"
        )
        assert len(calls) < 10, (
            f"loss_fn was called {len(calls)} times for a 50-step fit; the guard "
            "or the loss is being re-entered per step rather than traced once."
        )


def test_jit_still_closes_over_the_calibrator():
    """The guard uses concrete values; it must not break a jitted caller.

    ``jnp.isfinite`` on a tracer raises, so if this guard ever ran under trace
    it would take the whole jitted route down with it. Pinning the working
    arrangement is cheaper than rediscovering why it broke.
    """
    calibrator = GradientCalibrator(n_steps=5)
    fit, _ = calibrator.fit(forward, PARAMS0, OBSERVED, loss_fn=mean_squared_error)
    jitted = jax.jit(lambda p: forward(p))
    assert jnp.all(jnp.isfinite(jitted(fit)))
