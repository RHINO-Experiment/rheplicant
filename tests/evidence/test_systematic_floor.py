"""Section 9.4. A campaign can out-run its own calibration, silently.

The floor is declared in theta units because the memory has no pipeline -- the
whole architecture is that the raw data and the forward evaluation are gone. The
declaration is the point rather than a limitation: Task 8 measured that the
in-span half of a shared error leaves no trace in any statistic, so what is left
is a stated prior width and the arithmetic of when sigma_N passes under it.

**The plan's floor of 0.20 was above this fixture from the first night.** Its
first test asserts that a four-epoch campaign is still *above* the floor, and
measured, ``camp``'s single epoch already gives widths
``(0.17803, 0.13425)`` -- the loosest of them under 0.20 at N = 1, so
``systematic_floor(_memory(4), {"x": 0.20})`` reports a crossing epoch of 1 and
the test as written could never pass. ``FLOOR`` below is chosen against the
measured widths instead: at N = 4 they are ``(0.08932, 0.06729)``, so 0.05 sits
between the four-epoch width and the four-hundred-epoch one and the crossing
lands where a test can see it.
"""

import math

import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.diagnostics import systematic_floor
from rheplicant.inference.memory import BayesMemory
from tests.evidence import campaign_bank as camp

#: In theta units, and below every width this fixture reaches at N = 4 and above
#: every width it reaches at N = 400. Measured: tightest width 0.06729 at N = 4,
#: 0.00673 at N = 400.
FLOOR = 0.05


def _memory(n_epochs, represents=None):
    memory = BayesMemory(camp.factorization(represents))
    for term in camp.terms(n_epochs, biased=False):
        memory = memory.remember(term)
    return memory


def _widths(n_epochs):
    """The dense route to the same widths, prior included, for the oracle."""
    return np.sqrt(np.diag(camp.posterior(camp.terms(n_epochs, False))[1]))


def test_the_reported_sigma_is_the_dense_posterior_width():
    """The prior is inside it, and that is not a detail.

    ``BayesMemory.fisher()`` excludes the prior's curvature on purpose -- it is
    the *likelihood's* information. A floor is compared against the width a
    result is quoted with, which is the posterior's, so ``systematic_floor``
    adds the prior back by differentiating it. Leaving it out would make every
    sigma larger than the quoted one and the refusal would fire later than it
    should, which is the silent direction.
    """
    report = systematic_floor(_memory(4), {"x": FLOOR})
    assert report["x"]["sigma"] == pytest.approx(float(np.min(_widths(4))), rel=1e-10)


def test_a_young_campaign_is_above_the_floor_and_says_so():
    report = systematic_floor(_memory(4), {"x": FLOOR})
    assert report["x"]["sigma"] > FLOOR
    assert report["x"]["below_floor"] is False
    assert report["x"]["crossing_epoch"] > 4


def test_the_crossing_epoch_is_computed_and_not_quoted():
    """Section 7's rule, one section along: do not quote a crossing you did not compute."""
    report = systematic_floor(_memory(16), {"x": FLOOR})
    predicted = report["x"]["crossing_epoch"]
    actual = next(
        n for n in range(1, 400) if np.min(_widths(n)) < FLOOR
    )
    assert predicted == pytest.approx(actual, rel=0.25)


def test_the_reported_sigma_is_the_tightest_component_not_the_loosest():
    """A vector latent has several widths and the floor is one number.

    The tightest is the one that goes under first, so it is the one a refusal
    must watch. Measured on this fixture at FLOOR = 0.05: the tightest component
    crosses at N = 8 and the loosest at N = 13, so a report keyed on the loosest
    would stay quiet for five nights while a quoted error bar was already below
    the declared systematic.
    """
    tight = next(n for n in range(1, 400) if np.min(_widths(n)) < FLOOR)
    loose = next(n for n in range(1, 400) if np.max(_widths(n)) < FLOOR)
    assert (tight, loose) == (8, 13)
    assert systematic_floor(_memory(8), {"x": FLOOR})["x"]["below_floor"] is True


def test_a_mature_campaign_is_below_the_floor():
    report = systematic_floor(_memory(400), {"x": FLOOR})
    assert report["x"]["below_floor"] is True
    assert report["x"]["sigma"] < FLOOR


def test_audit_refuses_to_report_tighter_than_a_declared_floor():
    with pytest.raises(StateValidationError, match="systematic floor"):
        _memory(400).audit(systematic_floor={"x": FLOOR})


def test_audit_accepts_the_same_campaign_when_the_product_is_represented():
    """The escape hatch section 9.5 names: model it, and the floor no longer binds."""
    memory = _memory(400, represents={"beam_map": ("x",)})
    report = memory.audit(systematic_floor={"x": FLOOR}, modelled=("beam_map",))
    assert report["systematic_floor"]["x"]["below_floor"] is True


def test_declaring_a_product_without_modelling_it_still_refuses():
    """``represents`` alone is not the escape hatch -- ``modelled=`` is.

    The nearest way to over-trust this guard: a campaign that declares which
    latent *would* stand for the beam map, and then never says the beam map is
    among the products it is claiming credit for.
    """
    memory = _memory(400, represents={"beam_map": ("x",)})
    with pytest.raises(StateValidationError, match="systematic floor"):
        memory.audit(systematic_floor={"x": FLOOR})


def test_a_floor_naming_a_latent_the_memory_does_not_have_is_refused():
    with pytest.raises(StateValidationError, match="does not accumulate"):
        _memory(10).audit(systematic_floor={"no_such_latent": 0.1})


def test_modelled_naming_a_product_the_factorization_does_not_declare_is_refused():
    """The typo that would otherwise read as a breach.

    ``modelled=("beam_maps",)`` against ``represents={"beam_map": ...}`` matches
    nothing, so the floor binds and the analyst is shown a refusal about their
    campaign when what is wrong is a letter in an argument.
    """
    memory = _memory(10, represents={"beam_map": ("x",)})
    with pytest.raises(StateValidationError, match="does not declare"):
        memory.audit(systematic_floor={"x": FLOOR}, modelled=("beam_maps",))


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_a_floor_that_is_not_a_finite_positive_width_is_refused(bad):
    """A declared width of nan makes ``sigma > floor`` False and every latent breach.

    So the declaration is checked before it is compared, and ``inf`` with it: a
    floor of ``inf`` puts every campaign below it, which reads as a detection
    and is a typo.
    """
    with pytest.raises(StateValidationError, match="declared floor"):
        systematic_floor(_memory(4), {"x": bad})


def test_an_empty_campaign_is_refused_rather_than_extrapolated():
    """``crossing_epoch`` is ``N (sigma_N / floor)^2``, and N = 0 sends it to 0.

    Which would report that a campaign with no data crossed its systematic floor
    before it started.
    """
    with pytest.raises(StateValidationError, match="at least one epoch"):
        systematic_floor(BayesMemory(camp.factorization()), {"x": FLOOR})


def test_a_nan_width_is_refused_rather_than_compared_away():
    """`x > tol` is False for NaN, so the guard must be written `not (x <= tol)`."""
    import jax.numpy as jnp

    memory = _memory(10)
    poisoned = BayesMemory(
        memory.factorization,
        type(memory.accumulated)(
            factor=memory.accumulated.factor.at[0, 0].set(jnp.nan),
            target=memory.accumulated.target,
            offset=memory.accumulated.offset,
            names=memory.accumulated.names,
            shapes=memory.accumulated.shapes,
        ),
        memory.archive,
    )
    assert math.isnan(systematic_floor(poisoned, {"x": FLOOR})["x"]["sigma"])
    with pytest.raises(StateValidationError, match="systematic floor"):
        poisoned.audit(systematic_floor={"x": FLOOR})
