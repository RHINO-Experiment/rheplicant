"""The calibrators land on least squares' closed form, which neither package supplies.

`GradientCalibrator.fit` and `AdamCalibrator.fit` delegate their descent to
``bayesmith.optimize.minimize`` as of the Wave C ``calibrate`` switch. That
makes the obvious acceptance test circular: comparing the near side against
the far side now compares a facade with the thing it calls.

**The measurement that was NOT circular could only be taken before the
deletion, and it was.** On this same model, the shipped implementation and
``minimize`` agreed **bitwise** -- both methods, both parameters, and every
one of 120 loss-history entries, ``max|delta| = 0.0``. That is recorded in
``docs/superpowers/specs/2026-08-29-wave-C-calibrate-opening.md`` and cannot be
re-run here, because one side of it no longer exists. Which is the whole
reason it had to be run first.

What CAN be re-run for ever is this: least squares on a straight line has an
exact solution, written here in NumPy from the normal equations. Neither
package supplies it, so a change on either side of the seam that walks away
from the truth fails here regardless of what the other side does.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference import AdamCalibrator, GradientCalibrator
from rheplicant.inference.likelihood import mean_squared_error

N = 32
A_TRUE, B_TRUE = 2.0, -0.5

X = jnp.linspace(0.0, 1.0, N)
OBSERVED = A_TRUE * X + B_TRUE
FORWARD = lambda p: p["a"] * X + p["b"]  # noqa: E731
START = {"a": jnp.array(0.0), "b": jnp.array(0.0)}


def _closed_form():
    """``(a, b)`` from the normal equations, in NumPy, from the algebra.

    Written out rather than taken from either package: an oracle that shared
    an implementation with the thing under test would agree with it while both
    were wrong.
    """
    x = np.asarray(X, dtype=float)
    y = np.asarray(OBSERVED, dtype=float)
    design = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(a), float(b)


@pytest.mark.parametrize(
    "calibrator",
    [
        GradientCalibrator(learning_rate=0.1, n_steps=4000),
        AdamCalibrator(learning_rate=0.05, n_steps=4000),
    ],
    ids=["gradient", "adam"],
)
def test_it_reaches_the_closed_form(calibrator):
    a_true, b_true = _closed_form()
    fitted, losses = calibrator.fit(FORWARD, START, OBSERVED, loss_fn=mean_squared_error)
    assert float(fitted["a"]) == pytest.approx(a_true, abs=1e-3), "slope"
    assert float(fitted["b"]) == pytest.approx(b_true, abs=1e-3), "intercept"
    assert float(losses[-1]) < float(losses[0]), "the loss must have gone down"


def test_the_oracle_is_not_the_starting_point():
    """Anti-vacuity: a fit that never moved would pass the case above otherwise.

    The start is ``a = b = 0`` and the answer is not, by a margin far larger
    than the 1e-3 tolerance -- so reaching it is evidence of a descent rather
    than of an unchanged initial guess.
    """
    a_true, b_true = _closed_form()
    assert abs(a_true - float(START["a"])) > 1.0
    assert abs(b_true - float(START["b"])) > 0.1


def test_the_history_is_one_entry_per_step():
    """The return contract, which the switch had every opportunity to change.

    ``(params_fit, losses)`` with ``losses`` shaped ``(n_steps,)`` is what both
    calibrators promised before the switch; the far side returns a ``Fit``
    named tuple, and the facade unpacks it. A silently shorter history would
    still let every assertion above pass.
    """
    calibrator = AdamCalibrator(learning_rate=0.05, n_steps=137)
    _, losses = calibrator.fit(FORWARD, START, OBSERVED, loss_fn=mean_squared_error)
    assert jnp.shape(losses) == (137,)


class TestEveryKnobReachesTheFarSideExactlyAsWritten:
    """The facade threads eight values through; nothing else checks they arrive.

    This is D50's remedy applied one seam down. A calibrator's settings are now
    ``minimize``'s keyword arguments, and a facade that dropped one would leave
    the far side on its own default -- which for ``beta1`` is 0.9 against a
    declared 0.0, a difference that still converges, still returns finite
    values, and still moves off the starting point. Every assertion elsewhere
    in the suite would stay green.

    Measured: the only test in the suite that sets a non-default beta
    (``test_inference_construction_guards.py``, ``beta1=0.0, beta2=0.0``)
    asserts finiteness and that a step was taken. Both survive the default.
    """

    def _spy(self, monkeypatch):
        seen = {}
        import rheplicant.inference.calibrate as module

        real = module._far_minimize

        def recording(objective, at, **kwargs):
            seen.update(kwargs)
            return real(objective, at, **kwargs)

        monkeypatch.setattr(module, "_far_minimize", recording)
        return seen

    def test_the_gradient_calibrator_passes_its_three(self, monkeypatch):
        seen = self._spy(monkeypatch)
        GradientCalibrator(learning_rate=0.037, n_steps=11).fit(
            FORWARD, START, OBSERVED, loss_fn=mean_squared_error
        )
        assert seen == {"method": "gradient", "steps": 11, "learning_rate": 0.037}, seen

    def test_the_adam_calibrator_passes_its_six(self, monkeypatch):
        seen = self._spy(monkeypatch)
        AdamCalibrator(
            learning_rate=0.041, n_steps=13, beta1=0.11, beta2=0.22, eps=3e-7
        ).fit(FORWARD, START, OBSERVED, loss_fn=mean_squared_error)
        assert seen == {
            "method": "adam",
            "steps": 13,
            "learning_rate": 0.041,
            "beta1": 0.11,
            "beta2": 0.22,
            "eps": 3e-7,
        }, seen

    def test_the_two_methods_are_not_the_same_string(self, monkeypatch):
        """Anti-vacuity for both cases above: `method` actually distinguishes.

        If the facade sent one method for both calibrators, each case above
        would still be checking a self-consistent dictionary.
        """
        gradient = self._spy(monkeypatch)
        GradientCalibrator(n_steps=3).fit(FORWARD, START, OBSERVED)
        first = gradient["method"]
        adam = self._spy(monkeypatch)
        AdamCalibrator(n_steps=3).fit(FORWARD, START, OBSERVED)
        assert first != adam["method"], (first, adam["method"])
