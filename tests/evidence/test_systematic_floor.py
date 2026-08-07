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
from rheplicant.inference.diagnostics import _tightest_direction, systematic_floor
from rheplicant.inference.memory import BayesMemory, _direction_phrase
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
    """The dense route to the coordinate widths, prior included, for the oracle."""
    return np.sqrt(np.diag(camp.posterior(camp.terms(n_epochs, False))[1]))


def _direction_width(n_epochs):
    """The dense route to the width of the *tightest direction*.

    ``sqrt`` of the smallest eigenvalue of the posterior covariance, which is
    the smallest number the campaign can put an error bar on. It is at most the
    smallest diagonal entry and strictly smaller whenever the posterior is
    correlated at all -- measured on this fixture, whose posterior correlation
    is 0.5131, it is 0.10419 against a tightest coordinate of 0.13425 at N = 1.
    """
    return float(np.sqrt(np.linalg.eigvalsh(camp.posterior(camp.terms(n_epochs, False))[1])[0]))


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
    assert report["x"]["sigma"] == pytest.approx(_direction_width(4), rel=1e-10)


def test_a_young_campaign_is_above_the_floor_and_says_so():
    report = systematic_floor(_memory(4), {"x": FLOOR})
    assert report["x"]["sigma"] > FLOOR
    assert report["x"]["below_floor"] is False
    assert report["x"]["crossing_epoch"] > 4


def test_the_crossing_epoch_is_computed_and_not_quoted():
    """Section 7's rule, one section along: do not quote a crossing you did not compute."""
    report = systematic_floor(_memory(16), {"x": FLOOR})
    predicted = report["x"]["crossing_epoch"]
    actual = next(n for n in range(1, 400) if _direction_width(n) < FLOOR)
    assert predicted == pytest.approx(actual, rel=0.25)


def test_the_reported_sigma_is_the_tightest_direction_not_the_tightest_coordinate():
    """A vector latent has a width in every direction and the floor is one number.

    The *coordinate* widths are the diagonal of the covariance; the smallest
    number the campaign can put an error bar on is the square root of its
    smallest **eigenvalue**, which is below every diagonal entry as soon as the
    posterior is correlated at all. Measured on this fixture at FLOOR = 0.05
    -- posterior correlation 0.5131, direction ``(-0.5059, 0.8626)`` at every
    N because the design repeats -- the tightest direction crosses at N = 5,
    the tightest coordinate at N = 8 and the loosest at N = 13. A report keyed
    on the tightest *coordinate* is quiet for three nights while a quoted error
    bar is already under the declared systematic, and a report keyed on the
    loosest is quiet for eight.
    """
    direction = next(n for n in range(1, 400) if _direction_width(n) < FLOOR)
    tight = next(n for n in range(1, 400) if np.min(_widths(n)) < FLOOR)
    loose = next(n for n in range(1, 400) if np.max(_widths(n)) < FLOOR)
    assert (direction, tight, loose) == (5, 8, 13)
    assert systematic_floor(_memory(5), {"x": FLOOR})["x"]["below_floor"] is True
    reported = systematic_floor(_memory(5), {"x": FLOOR})["x"]
    assert reported["sigma"] < np.min(_widths(5))


class TestTheFloorWatchesADirectionAndNamesIt:
    """The rationale was right and the arithmetic was one basis rotation off.

    The docstring said "the tightest is the first to go under, so it is the one
    a refusal must watch" and the code took ``np.min`` of the *diagonal*. For
    any correlated posterior the smallest eigen-direction is below every
    diagonal entry, so the sentence was true of a quantity the function was not
    computing.

    The fixture below makes the gap unmissable rather than merely present: two
    latents that enter the design almost identically, so their sum is pinned and
    their difference is not. Measured over 200 epochs with a floor of 0.05, the
    coordinate widths are ``(1.402505, 1.403178)`` -- comfortably above -- while
    the tightest direction is ``0.006335``, 7.9 times **below** the floor, along
    ``(0.7073, 0.7069)``. That is the campaign quoting an error bar under its
    own declared systematic while the refusal stays silent, which is the exact
    failure this function exists to prevent.
    """

    N_SAMPLES, SIGMA, PRIOR_STD = 8, 0.5, 2.0
    TRUTH = np.array([1.0, -0.5])

    def _campaign(self, design, n_epochs=200, seed=17):
        """A memory over one latent ``x`` of shape ``(2,)`` with the given design."""
        import jax.numpy as jnp

        from rheplicant.inference.compress import compress_linear
        from rheplicant.inference.factorize import Factorization
        from rheplicant.inference.parameters import Bind, Latent, ParameterSpace

        latent = Latent(
            "x", init=jnp.zeros(2), prior=camp._Normal(0.0, self.PRIOR_STD)
        )
        space = ParameterSpace(
            latents=(latent,), bindings=(Bind("x", into=lambda p: p.x),)
        )
        memory = BayesMemory(Factorization(space))
        rng = np.random.default_rng(seed)
        for e in range(n_epochs):
            data = design @ self.TRUTH + self.SIGMA * rng.normal(size=self.N_SAMPLES)
            memory = memory.remember(
                compress_linear(
                    design={"x": jnp.asarray(design)},
                    observed=jnp.asarray(data),
                    noise_std=self.SIGMA,
                    shapes={"x": (2,)},
                    epoch_id=f"n{e}",
                )
            )
        return memory

    def _near_collinear(self):
        rng = np.random.default_rng(3)
        base = rng.normal(size=self.N_SAMPLES)
        return np.column_stack([base, base + 1e-3 * rng.normal(size=self.N_SAMPLES)])

    def _orthogonal(self):
        """Two columns with zero inner product: the nearest legitimate case.

        With a diagonal prior this makes the posterior covariance diagonal, so
        the smallest eigenvalue **is** the smallest diagonal entry and the fix
        must return exactly what the old code returned. Over-refusing here would
        be as bad as under-refusing there.
        """
        rng = np.random.default_rng(5)
        first = rng.normal(size=self.N_SAMPLES)
        second = rng.normal(size=self.N_SAMPLES)
        second = second - first * (second @ first) / (first @ first)
        return np.column_stack([first, second])

    def test_a_near_collinear_campaign_is_below_the_floor_along_a_combination(self):
        memory = self._campaign(self._near_collinear())
        report = systematic_floor(memory, {"x": FLOOR})["x"]
        assert report["sigma"] == pytest.approx(0.006335, rel=1e-3)
        assert report["below_floor"] is True
        # ...and every coordinate width is more than an order of magnitude above
        # the floor, which is what the old `np.min(diag)` reported.
        widths = np.sqrt(np.diag(np.linalg.inv(_total_information(memory, self.PRIOR_STD))))
        assert np.min(widths) == pytest.approx(1.402505, rel=1e-3)
        assert np.min(widths) > FLOOR

    def test_the_direction_is_reported_so_the_refusal_can_be_acted_on(self):
        """A bare number says "too tight"; the caller needs "too tight in what"."""
        report = systematic_floor(self._campaign(self._near_collinear()), {"x": FLOOR})
        direction = np.asarray(report["x"]["direction"])
        assert direction.shape == (2,)
        assert float(np.linalg.norm(direction)) == pytest.approx(1.0)
        np.testing.assert_allclose(direction, [0.7073, 0.7069], atol=1e-3)

    def test_the_refusal_names_the_combination(self):
        memory = self._campaign(self._near_collinear())
        with pytest.raises(StateValidationError, match="systematic floor") as caught:
            memory.audit(systematic_floor={"x": FLOOR})
        assert "direction" in str(caught.value)
        assert "0.707" in str(caught.value)

    def test_an_uncorrelated_posterior_answers_exactly_as_before(self):
        memory = self._campaign(self._orthogonal(), n_epochs=8)
        report = systematic_floor(memory, {"x": FLOOR})["x"]
        widths = np.sqrt(np.diag(np.linalg.inv(_total_information(memory, self.PRIOR_STD))))
        assert report["sigma"] == pytest.approx(float(np.min(widths)), rel=1e-12)
        # ...and the direction it names is a coordinate axis, not a mixture.
        direction = np.abs(np.asarray(report["direction"]))
        assert float(np.max(direction)) == pytest.approx(1.0, abs=1e-9)


class TestTheDirectionArithmeticOnBlocksNoCampaignCanProduce:
    """Two branches of `_tightest_direction` that the fixtures cannot reach.

    `_posterior_covariance` builds its matrix from a Cholesky factor, so the
    block handed over is positive definite and finite whenever the campaign is.
    Both branches below exist for the day that provenance changes -- and an
    untested branch is how the wrong one of them gets written -- so they are
    exercised where they can be: on the function directly.
    """

    def test_a_block_with_no_finite_entries_reports_nan_rather_than_raising(self):
        """`np.linalg.eigh` on NaN raises `LinAlgError`, which is the wrong error.

        A poisoned campaign must reach `systematic_floor`'s NaN-safe comparison
        and be reported as a breach, not die inside LAPACK with a message about
        convergence.
        """
        width, direction = _tightest_direction(np.array([[np.nan, 0.0], [0.0, 1.0]]))
        assert math.isnan(width)
        assert direction is None

    def test_a_roundoff_negative_eigenvalue_is_zero_and_not_nan(self):
        """Zero is the honest width for a direction the campaign does not constrain.

        `math.sqrt` of a small negative gives `nan`, which the audit message
        describes as a *poisoned* factor -- a confident and wrong diagnosis of a
        block that is merely singular.
        """
        width, direction = _tightest_direction(np.array([[-1e-18, 0.0], [0.0, 1.0]]))
        assert width == 0.0
        assert direction is not None


def test_the_direction_is_omitted_for_a_latent_that_has_only_one():
    """A width-1 latent's only direction is itself, and naming it is noise."""
    assert _direction_phrase("t_rx", np.array([1.0])) == ""
    assert _direction_phrase("x", None) == ""
    assert _direction_phrase("x", np.array([0.6, 0.8])) == (
        " in direction (0.600, 0.800) of x"
    )


def _total_information(memory, prior_std):
    """``F_like + F_prior`` for a single latent with an isotropic normal prior."""
    import jax.numpy as jnp

    return np.asarray(
        memory.fisher(at={"x": jnp.zeros(2)}).matrix
    ) + np.eye(2) / prior_std**2


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
