"""The cheap diagnostics from a document -- the conditioning number first.

``condition`` is the exit a user is supposed to run BEFORE paying for a fit:
kappa is what says how much a solver's residual understates its error, and
``condition_estimate``'s own docstring is blunt about the consequence -- "a
residual of 1e-6 against kappa=1e7 certifies nothing at all" (linear.py:1352).
A diagnostic that returned the same number for a well-posed and an ill-posed
design would be worth nothing, so the two documents here differ by seven
orders of magnitude in kappa and by one line of YAML.

Every number below was measured on this repo at ``cde25c9``, float32 (the
repo's default x32 mode), and is REPRODUCIBLE: ``condition_estimate`` is a
power iteration, but its ``key`` defaults to ``jax.random.key(0)`` inside the
package, so a run with no ``seed:`` returns the same float every time
(measured: two consecutive runs agree bit for bit, which
:meth:`TestTheSeedRunsTheOtherWay.test_the_seed_is_optional` asserts).

``identifiability`` and ``score_directions`` join it below.  Those two need
only a space: neither reads observed data and neither weighs a residual -- a
ParameterSpace, the fit twin and the state are the whole input, which is what
makes them the checks a user runs before paying for a fit.  Every number in
their two classes was measured against
:func:`~tests.config.exit_helpers.diagnostic_document` at ``3b26202``, whose
fit twin is ``data = g * d * gaussian(nu)`` on a 16 x 8 grid: the ``g`` and
``d`` columns of the Jacobian are exactly proportional, so the report reads
n_par 2, n_data 128, rank 1, nullity 1.
"""

import math

import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.document import load_document
from rheplicant.config.sections.runs import parse_runs, run_document
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import identifiability
from tests.config.exit_helpers import (
    IDENTIFIED_PAIR,
    diagnostic_document,
    diagnostic_report,
    diagnostic_rows,
)
from tests.config.test_config_document import synthetic_document

# Two constant-in-time temperatures summed at t_ant_sum, so the prediction is
# jointly affine in BOTH coefficients.  (A gain latent alongside one of them
# would be bilinear, and check_linearity refuses that block by name -- "is not
# affine in them JOINTLY", linear.py:555.)
#
# ORTHOGONAL: the second design column alternates sign channel to channel, so
# the two columns of A are orthogonal and of equal norm -- kappa is exactly 1.
ORTHOGONAL = [[1.0 if channel % 2 == 0 else -1.0] for channel in range(8)]
# DEGENERATE: the second column is the first tilted by 1e-4 per channel, so the
# two are parallel to one part in 1e4 and only the prior holds their difference
# down.  This is the honest ill-conditioned case: lambda_min is 1/prior_std**2
# and lambda_max is set by the data, exactly as linear.py:1356-1359 describes.
DEGENERATE = [[1.0 + 1.0e-4 * channel] for channel in range(8)]

#: A third latent that is declared linear and is not: the prediction is a
#: Gaussian in frequency, so its WIDTH is a knob check_linearity refuses.  It
#: is what gives ``check:`` a lever on this exit -- see
#: :meth:`TestWhatItRefuses.test_check_false_builds_the_block_linearity_refuses`.
WIDTH_LATENT = {"w": {"init": {"value": 5.0, "unit": "MHz"}, "linear": True,
                      "into": "global_signal.width"}}


def condition_document(run, freq_basis=DEGENERATE, sigma=0.5, extra=None):
    """A document with two linear temperatures, a sigma, and NO observed data.

    ``condition_estimate`` takes no ``observed`` and never calls
    ``_check_solve_arguments`` (linear.py:1389-1398), so a document with no
    ``inference.observed:`` is the normal case for this exit -- it is what a
    user runs before the data exists.

    ``sigma`` is the homoscedastic width, which is half of what kappa is made
    of (``lambda_max`` scales as ``1/sigma**2``); ``extra`` adds latents to
    ``inference.parameters`` beside ``a`` and ``b``, for the runs whose
    ``names:`` reach past the affine pair.
    """
    doc = synthetic_document()
    doc["runtime"] = {"seed": 20260806, "seeds": {"zero": 0, "other": 1}}
    doc["model"] = {key: value for key, value in doc["model"].items()
                    if key != "noise"}
    doc["model"]["t_sys_extra"] = [
        {"coeff": {"list": [[100.0]], "unit": "K"},
         "time_basis": {"ones": [16, 1]},
         "freq_basis": {"ones": [8, 1]}},
        {"coeff": {"list": [[10.0]], "unit": "K"},
         "time_basis": {"ones": [16, 1]},
         "freq_basis": {"list": freq_basis}},
    ]
    doc["inference"] = {
        "parameters": {
            "a": {"init": {"list": [[100.0]], "unit": "K"}, "linear": True,
                  "into": "t_sys_extra_1.coeff"},
            "b": {"init": {"list": [[10.0]], "unit": "K"}, "linear": True,
                  "into": "t_sys_extra_2.coeff"},
            **(extra or {}),
        },
        "noise": {"kind": "homoscedastic",
                  "sigma": {"value": sigma, "unit": "K"}},
    }
    doc["runs"] = [run]
    return doc


#: ``name: kappa``, deliberately NOT the kind.  ``runs.py`` names an unnamed
#: run after its kind, so a run named for its kind cannot tell a ``where``
#: built from ``run.name`` -- what the executors spell -- from one built out of
#: ``run.kind``.  Under this name the two are different strings, and the
#: refusal tests read the prefix.
KAPPA = {"name": "kappa", "kind": "condition", "names": ["a", "b"],
         "prior_std": {"a": 100.0, "b": 100.0}}


def kappa_of(run, **document):
    """Run one condition exit and return its product as a Python float."""
    results = run_document(condition_document(run, **document))
    return float(results["kappa"].product)


class TestTheNumberItReports:
    def test_an_orthogonal_block_is_perfectly_conditioned(self):
        assert kappa_of(KAPPA, freq_basis=ORTHOGONAL) == pytest.approx(1.0,
                                                                       rel=1e-3)

    def test_a_nearly_degenerate_block_reports_seven_orders_more(self):
        """The whole point of the exit, in one comparison.

        Measured 1.24e7 against 1.00e0 -- the order of magnitude is pinned,
        the float is not.  An executor that built the block from one latent,
        or that ignored the document's second basis, would land near 1 here.
        """
        ill = kappa_of(KAPPA)
        assert 1.0e6 < ill < 1.0e8
        assert ill / kappa_of(KAPPA, freq_basis=ORTHOGONAL) > 1.0e6

    def test_the_product_is_a_bare_scalar_not_a_tuple(self):
        """condition_estimate returns a scalar array (linear.py:1386-1387).

        Its three siblings return a 2-tuple or a GLSResult; an executor that
        copied one of them and unpacked would raise here, and one that wrapped
        the answer in a dict would fail the shape assertion.
        """
        product = run_document(condition_document(KAPPA))["kappa"].product
        assert product.shape == ()
        assert float(product) > 1.0e6

    def test_the_declared_prior_std_decides_kappa(self):
        """prior_std is the only thing holding the degenerate direction down.

        lambda_min is exactly 1/prior_std**2, so a hundredfold wider prior is
        a ten-thousandfold larger kappa: measured 1.240e3 against 1.239e7.  A
        run that dropped prior_std, or substituted a default of its own, would
        return the same number for both.
        """
        narrow = kappa_of({**KAPPA, "prior_std": {"a": 1.0, "b": 1.0}})
        wide = kappa_of(KAPPA)
        assert narrow == pytest.approx(1.24e3, rel=0.05)
        assert wide / narrow == pytest.approx(1.0e4, rel=0.05)

    def test_names_decides_which_block_is_conditioned(self):
        """The JOINT number is the one a per-block guard cannot produce.

        ``_condition_number``'s own docstring (linear.py:1322-1324): "two
        latents the data barely distinguishes give a well-conditioned operator
        each and a badly conditioned one together".  Measured on the
        degenerate document: ``a`` alone is 1.000000e+00 and so is ``b``,
        while the pair is 1.239e7.  An executor that ignored ``names:`` and
        built over the whole space would report the pair's number for the
        single-latent run.
        """
        alone = kappa_of({**KAPPA, "names": ["a"], "prior_std": {"a": 100.0}})
        assert alone == pytest.approx(1.0, rel=1e-3)
        assert kappa_of(KAPPA) / alone > 1.0e6

    def test_the_documents_sigma_is_half_of_what_kappa_is_made_of(self):
        """lambda_max scales as 1/sigma**2, and lambda_min is the prior's.

        So a tenfold wider noise is a hundredfold smaller kappa: measured
        1.239474e7 at sigma 0.5 K against 1.237803e5 at 5.0 K, a ratio of
        100.13.  This is the assertion that fails when the decided sigma stops
        travelling: an executor calling ``condition_estimate`` with a sigma of
        ones lands at 3.099e6 (measured), which every other test in this
        module accepts -- 1e6 < 3.099e6 < 1e8, and the prior and iteration
        RATIOS are independent of sigma.
        """
        loud = kappa_of(KAPPA, sigma=5.0)
        assert kappa_of(KAPPA) / loud == pytest.approx(100.0, rel=0.05)


class TestTheKnobsReachThePackage:
    def test_iterations_reaches_the_power_iteration(self):
        """One power iteration per end of the spectrum is not twelve.

        POWER_ITERATIONS is 12 (linear.py:102) and the config layer never
        restates it -- iterations: is passed only when declared.  So a
        declared 1 must be VISIBLE in the answer, and it is: the estimate
        lands at 39% of the settled one (4.776e6 against 1.239e7).
        """
        settled = kappa_of(KAPPA)
        one = kappa_of({**KAPPA, "iterations": 1})
        assert one < 0.6 * settled
        assert one / settled == pytest.approx(0.385, rel=0.05)

    def test_declaring_twelve_lands_exactly_where_the_default_does(self):
        """The package default is 12; declaring it must change nothing.

        Together with the test above this pins that the number in the
        document is the number the package receives -- not an off-by-one, not
        a doubling, not a config-layer default of its own.  The second half is
        what keeps the first from being vacuous: the estimate alternates with
        the PARITY of the iteration count on this document (measured:
        1.239474e7 at 12 and at 24, 1.015377e7 at 11 and at 13), so an
        executor that passed ``iterations + 1`` would be caught by an equality
        the reader might otherwise read as untestable.
        """
        settled = kappa_of(KAPPA)
        assert kappa_of({**KAPPA, "iterations": 12}) == settled
        assert kappa_of({**KAPPA, "iterations": 13}) != settled

    def test_a_non_positive_iterations_is_refused(self):
        """Matched on the FLOOR clause, not on the word "iterations".

        The generic sweep names every key this exit takes, ``iterations``
        among them, so a bare ``match="iterations"`` is satisfied by the
        neighbouring refusal as well as by this one.
        """
        with pytest.raises(ConfigError, match=r"iterations: must be >= 1"):
            kappa_of({**KAPPA, "iterations": 0})

    def test_a_non_number_iterations_is_refused_here_not_in_a_trace(self):
        with pytest.raises(ConfigError,
                           match=r"iterations: is a number") as caught:
            kappa_of({**KAPPA, "iterations": "twelve"})
        assert str(caught.value).startswith("runs['kappa']: ")

    def test_iterations_null_is_not_an_off_switch(self):
        """``iterations`` is declared NOT nullable, and that is a decision.

        ``maxiter: null`` and ``require_convergence: null`` are how the
        package spells "no cap" and "no guard"; power iteration has no such
        spelling -- ``iterations=None`` would reach ``range(None)`` inside
        ``extreme_eigenvalues``.  So the coercion spec carries
        ``nullable=False`` and this is the test that fails if it is flipped.
        """
        with pytest.raises(ConfigError, match=r"iterations: is a number"):
            kappa_of({**KAPPA, "iterations": None})


class TestTheSeedRunsTheOtherWay:
    """A29's asymmetry reverses here, and this is the only exit where it does.

    ``gcr_sample``'s key is a REQUIRED keyword-only argument with no default;
    ``condition_estimate``'s is ``key: jax.Array | None = None`` and falls
    back to ``jax.random.key(0)`` (linear.py:1397).  So ``seed:`` is optional
    on a condition run and A29 does not make it required -- schema §4.7.9
    lists ``seed`` among this kind's four keys, and A29's own row names
    ``plan.estimate`` as the exit that refuses one.
    """

    def test_the_seed_is_optional_and_the_estimate_is_reproducible(self):
        """No seed at all is legal here -- and the answer is a fixed number.

        The package's ``key=None`` default is ``jax.random.key(0)``, so two
        runs of the same document agree bit for bit; a config layer that
        invented a key of its own (from the run name, from the clock, from
        ``runtime.seed``) would make this exit's number un-quotable.
        """
        assert 1.0e6 < kappa_of(KAPPA) < 1.0e8
        assert kappa_of(KAPPA) == kappa_of(KAPPA)

    def test_seed_zero_reproduces_the_packages_own_default(self):
        """runtime.seeds.zero is 0, so this must be bit-identical to no seed.

        seed_for returns a declared entry unchanged (config/draws.py:61-62),
        and the package defaults to jax.random.key(0).  Equality here is what
        proves the key is built from the document's number rather than from
        something the executor invented -- the blake2s fallback
        (``_digest(name) ^ runtime.seed``) would land on a different key and a
        different estimate.
        """
        seeded = kappa_of({**KAPPA, "seed": {"from": "runtime.seeds.zero"}})
        assert seeded == kappa_of(KAPPA)

    def test_another_seed_decides_a_different_estimate(self):
        """runtime.seeds.other is 1: a different starting vector, a different
        estimate.  Measured 1.015e7 against 1.239e7 -- same decade, plainly
        not the same number, which is exactly what a power-iteration estimate
        should do and what an ignored seed could not."""
        other = kappa_of({**KAPPA, "seed": {"from": "runtime.seeds.other"}})
        default = kappa_of(KAPPA)
        assert other != default
        assert other / default == pytest.approx(0.819, rel=0.05)

    def test_a_literal_seed_is_refused_by_the_shared_seed_grammar(self):
        """_seed_name demands {from: runtime.seeds.<name>} (draws.py:106-113);
        a bare integer is a realisation provenance.json cannot record."""
        with pytest.raises(ConfigError, match="runtime.seeds") as caught:
            kappa_of({**KAPPA, "seed": 7})
        assert str(caught.value).startswith("runs['kappa']: ")


class TestWhatItRefuses:
    def test_prior_mean_is_refused_because_kappa_has_no_centre(self):
        """Every sibling exit takes prior_mean; this one has nowhere to put it.

        The generic sweep would say "does not take ['prior_mean']", which
        reads as a typo.  The dedicated refusal says why -- and the assertions
        below are what tell the two apart: "centre" appears in no other
        refusal this document can raise, and ``it takes [`` is the sweep's own
        tail, which a reader must NOT be getting here.  The prefix is the
        third: this run is named ``kappa`` and its kind is ``condition``, so
        a ``where`` built out of ``run.kind`` reads differently.
        """
        with pytest.raises(ConfigError, match="centre") as caught:
            kappa_of({**KAPPA, "prior_mean": {"a": 0.0, "b": 0.0}})
        assert "it takes [" not in str(caught.value)
        assert "conjugate.wiener" in str(caught.value)
        assert str(caught.value).startswith("runs['kappa']: ")

    def test_the_centre_refusal_is_heard_before_the_generic_sweep(self):
        """A document wrong BOTH ways hears about the centre first.

        Order is the whole point of putting that check above ``_sweep``, and
        nothing else can observe it: with only ``prior_mean:`` declared, a
        sweep-first executor produces a message that still names the key.
        Declaring ``tol:`` alongside is what makes the two orders produce
        different text -- the sweep would refuse ``['prior_mean', 'tol']``
        together, so a message with no ``tol`` in it is proof of which check
        ran first.
        """
        with pytest.raises(ConfigError, match="centre") as caught:
            kappa_of({**KAPPA, "prior_mean": {"a": 0.0, "b": 0.0},
                      "tol": 1.0e-9})
        assert "tol" not in str(caught.value)

    def test_the_sweep_names_exactly_the_keys_this_exit_takes(self):
        """Schema §4.7.9: names, prior_std, iterations, seed -- and no tol.

        tol belongs to the solves; kappa is what you compute tol FROM.
        ``check`` joins them because it is ``linear_operator``'s own key and
        this exit builds an operator like its three siblings do
        (``conjugate_support._BLOCK_KEYS``).
        """
        with pytest.raises(
                ConfigError,
                match=r"it takes \['check', 'iterations', 'names', "
                      r"'prior_std', 'seed'\]") as caught:
            kappa_of({**KAPPA, "tol": 1.0e-9})
        assert "does not take ['tol']" in str(caught.value)

    def test_condition_needs_no_observed_section_at_all(self):
        """The block and the sigma are the whole input.

        This document declares no inference.observed:, which every other
        conjugate exit refuses -- and this one must not.  An executor calling
        ``_conjugate_block`` at its default ``needs_observed=True`` refuses
        here, naming data the document never had.
        """
        doc = condition_document(KAPPA)
        assert "observed" not in doc["inference"]
        assert float(run_document(doc)["kappa"].product) > 1.0e6

    def test_a_scalar_prior_std_over_a_pair_is_check_A51(self):
        """The per-member rule reaches this exit too, on its own route.

        ``condition`` does not call ``_prior_kwargs`` -- that helper emits
        ``prior_mean`` whenever the document declares it, which
        ``condition_estimate`` has no parameter for -- so its prior travels
        through a call this module is the only test of.  A scalar over a
        two-latent block is check A51's refusal, not a broadcast.
        """
        with pytest.raises(ConfigError, match="wrongly-regularised") as caught:
            kappa_of({**KAPPA, "prior_std": 100.0})
        assert "check A51" in str(caught.value)

    def test_a_prior_std_naming_the_wrong_latents_is_refused(self):
        """The mapping must name every member: S is block-diagonal.

        Reached through the same one-argument call as the test above, and
        with the block's own names in the message -- which is what says the
        BLOCK, not the space, decided what a complete mapping is.
        """
        # "Name every member" is the MAPPING branch's alone.  Matching
        # "prior_std:" or "['a', 'b']" would not do it: _one_prior's scalar
        # branch -- the A51 refusal one `if` below -- carries both, so either
        # anchor is satisfied by the wrong refusal and this test would stop
        # telling the two apart.  (Measured: swapping the mapping branch's
        # text for the scalar one leaves this test green under the looser
        # anchors, and only Task 2's own test notices.)
        with pytest.raises(ConfigError, match="Name every member") as caught:
            kappa_of({**KAPPA, "prior_std": {"a": 100.0}})
        assert "['a', 'b']" in str(caught.value)

    def test_a_prior_free_block_with_no_prior_std_is_the_packages_refusal(
            self):
        """_require_prior_std runs here as it does for the three solves.

        Neither latent in this document declares a ``prior:``, so kappa has
        no lambda_min without one -- and the config layer does not invent a
        width to keep the run going (linear.py:1009).
        """
        with pytest.raises(ParameterSpaceError, match="prior_std"):
            kappa_of({key: value for key, value in KAPPA.items()
                      if key != "prior_std"})

    def test_check_false_builds_the_block_linearity_refuses(self):
        """``check:`` is linear_operator's key and it reaches this exit.

        ``w`` writes into ``global_signal.width``, which the prediction is not
        affine in, so the pair ``(a, w)`` is a false ``linear: true`` claim:
        at the default the operator refuses by name, and ``check: false``
        builds it anyway.  What comes back is NaN (measured, at prior_std 100
        and at 1) -- which is the argument for the default rather than against
        the key: the condition number of a block that is not affine is not a
        number.
        """
        pair = {**KAPPA, "names": ["a", "w"],
                "prior_std": {"a": 100.0, "w": 100.0}}
        with pytest.raises(ParameterSpaceError, match="JOINTLY"):
            kappa_of(pair, extra=WIDTH_LATENT)
        assert math.isnan(kappa_of({**pair, "check": False},
                                   extra=WIDTH_LATENT))


# --- The two diagnostics that need only a space ----------------------------
#
# Their fixture family -- ``diagnostic_document``, ``diagnostic_report``,
# ``diagnostic_rows``, ``DIAGNOSTIC_GAIN``/``_DEPTH``/``_WIDTH`` and the two
# pairs -- lives in ``tests/config/exit_helpers.py``, which plan section 3
# designates the only place any exit document is built.  It was private to
# this module until ``kind: gradient`` needed exactly it: a fixture COPIED
# into a second module is how its one load-bearing property (the ``twin:``
# repair that keeps ``built.twin`` and ``built.inference.fit_twin`` different
# objects) gets re-derived as the deletion form and stops discriminating.


class TestIdentifiability:
    def test_the_report_lands_with_the_measured_rank_and_nullity(self):
        """Two parameters, one identified direction, one null.

        A run that fitted a single latent, or that differentiated the wrong
        twin, cannot produce this quartet.  The singular values are pinned
        too, because the quartet alone does not say that the degeneracy is
        EXACT: a merely near-degenerate pair also reports nullity 1 at the
        shipped rtol of 1e-8, and this fixture claims more than that --
        measured 1.4142136 against 4.52e-17, sixteen orders apart.
        """
        out = diagnostic_report({"kind": "identifiability"})
        assert out.names == ("g", "d")
        assert (out.n_par, out.n_data, out.rank, out.nullity) == (2, 128, 1, 1)
        assert out.singular_values[0] == pytest.approx(1.4142136, rel=1e-5)
        assert abs(float(out.singular_values[1])) < 1.0e-12

    def test_the_fit_twin_is_what_it_differentiates(self):
        """The fixture's discriminating property, asserted rather than assumed.

        ``built.twin`` and ``built.inference.fit_twin`` are the same object
        for every document that declares no ``inference.twin:``, and on such a
        document the difference between the two executors is invisible.  This
        document declares one, and the model twin is then not differentiable
        at all -- which is what makes every assertion in these two classes a
        statement about ``inference.fit_twin``.  Delete the repair from
        :func:`document` and this test is what fails.
        """
        built = load_document(diagnostic_document({"kind": "identifiability"}))
        assert built.twin is not built.inference.fit_twin
        with pytest.raises(ParameterSpaceError, match="NoiseOperator"):
            identifiability(built.inference.space, built.twin, built.state)
        assert diagnostic_report({"kind": "identifiability"}).nullity == 1

    def test_names_narrows_the_report_to_the_named_block(self):
        """The whole space reports n_par 2 / nullity 1; ``g`` alone is 1 / 0.

        An executor that dropped ``names:`` would hand back the first quartet
        here -- the conditional question a Gibbs block faces is answered
        ``yes`` exactly where the joint one is answered ``no``.
        """
        out = diagnostic_report({"kind": "identifiability", "names": ["g"]})
        assert out.names == ("g",)
        assert (out.n_par, out.rank, out.nullity) == (1, 1, 0)

    def test_rtol_reaches_the_package_and_moves_the_rank(self):
        """``g`` and ``w`` are genuinely two directions, and rtol says so.

        Measured singular values 1.2572932 and 0.6474673, so the rank flips
        exactly at ``rtol = 0.514969`` and at nothing else.  ``rtol: 0.6``
        sits between the two with room on both sides: the cutoff it names is
        0.7543759, which is 1.165x the second singular value and 0.600x the
        first -- 16.5% of margin below and 66.7% above, against a float64
        Jacobian whose own roundoff is ~1e-16 relative.  Nothing but the
        declared rtol can produce that, and the report echoes it back.
        """
        loose = diagnostic_report({"kind": "identifiability", "rtol": 0.6},
                       IDENTIFIED_PAIR)
        assert loose.rtol == pytest.approx(0.6)
        assert loose.threshold == pytest.approx(0.7543759, rel=1e-5)
        assert (loose.rank, loose.nullity) == (1, 1)
        assert loose.singular_values == pytest.approx([1.2572932, 0.6474673],
                                                      rel=1e-5)
        tight = diagnostic_report({"kind": "identifiability"}, IDENTIFIED_PAIR)
        assert (tight.rank, tight.nullity) == (2, 0)

    def test_at_moves_the_point_the_jacobian_is_taken_at(self):
        """``d(data)/dg`` is ``d * gaussian``; ``d(data)/dd`` is ``g *
        gaussian``.

        So doubling ``d`` from its declared init of 0.5 doubles the ``g``
        column's norm and does not move the ``d`` column at all.  The
        ASYMMETRY is what makes this a pin on ``at:`` rather than on "some
        number changed" -- a run evaluated at a uniformly rescaled point
        would move both.  ``column_norms`` is also the field the report's own
        docstring forgets to list (identifiability.py:270-292) while the real
        field order inserts it between ``jacobian`` and ``rtol``, so reading
        it by name is the assertion that a positional construction from that
        docstring would fail.
        """
        base = diagnostic_report({"kind": "identifiability"})
        moved = diagnostic_report({"kind": "identifiability",
                                   "at": {"d": 1.0}})
        assert base.column_norms == pytest.approx([3.1501079, 6.3002157],
                                                  rel=1e-5)
        assert moved.column_norms == pytest.approx([6.3002157, 6.3002157],
                                                   rel=1e-5)
        assert moved.column_norms[0] == pytest.approx(
            2.0 * base.column_norms[0], rel=1e-5)
        assert moved.column_norms[1] == pytest.approx(
            base.column_norms[1], rel=1e-5)

    def test_at_reads_the_documents_own_value_grammar(self):
        """``{value:, unit:}`` is what ``inference.observed.<name>.at``
        accepts, and the executor resolves through the same ``resolve_value``
        seam.  An executor that did ``float(node)`` instead would raise on
        this mapping, and one that passed the mapping through unresolved
        would reach jnp.asarray with a dict."""
        moved = diagnostic_report({"kind": "identifiability",
                        "at": {"d": {"value": 1.0, "unit": "K"}}})
        assert moved.column_norms == pytest.approx([6.3002157, 6.3002157],
                                                   rel=1e-5)

    def test_an_empty_at_is_the_report_with_no_at_at_all(self):
        """``{}`` is the right empty, and it is not "no ``at:``" spelled twice.

        ``_at_values`` returns ``{}`` rather than None because both entry
        points accept ``at={}``; what this pins is that an empty mapping is
        legal and inert -- a helper that refused it, or that read ``at: {}``
        as "override every latent with nothing", changes the answer here.
        """
        empty = diagnostic_report({"kind": "identifiability", "at": {}})
        base = diagnostic_report({"kind": "identifiability"})
        assert empty.column_norms == pytest.approx(base.column_norms)
        assert (empty.rank, empty.nullity) == (1, 1)

    def test_at_naming_an_undeclared_latent_is_this_layers_refusal(self):
        """Refused HERE, with the declared names, not as a package error.

        The package refuses it too, so deleting this branch leaves a document
        that still fails -- with a different exception type and without
        ``runs[...]`` to say which run asked.  Both halves are asserted,
        because the type alone is what tells the two apart.
        """
        with pytest.raises(ConfigError, match=r"at: names \['q'\]") as caught:
            run_document(diagnostic_document({"kind": "identifiability",
                                   "at": {"q": 1.0}}))
        assert "it declares ['g', 'd']" in str(caught.value)

    def test_names_naming_an_undeclared_latent_is_the_packages_refusal(self):
        """The asymmetry with ``at:`` above, and it is deliberate.

        ``_names`` validates the SHAPE of the key only; membership is the
        package's own refusal, which names the declared set itself.  A
        layer-level names check would change the exception type on a document
        that is already refused for the right reason, so this test is what
        records the decision rather than leaving it to be "fixed".
        """
        with pytest.raises(ParameterSpaceError,
                           match="not a latent of this space"):
            run_document(diagnostic_document({"kind": "identifiability",
                                   "names": ["q"]}))

    def test_a_float32_document_runs_rather_than_being_refused(self):
        """identifiability() forces x64 process-globally for its own duration
        and casts the selected latents, so the promoted Jacobian is float64
        and the report LANDS on an ordinary float32 document.  Its "even with
        x64" refusal is reserved for a model that pins its OUTPUT dtype with
        an explicit cast.  This test exists so that nobody later "fixes" the
        layer by refusing a float32 run here."""
        doc = diagnostic_document({"kind": "identifiability"})
        assert load_document(doc).runtime.dtype == "float32"
        assert run_document(doc)["identifiability"].product.nullity == 1

    def test_the_sweep_names_exactly_the_three_keys_this_exit_takes(self):
        """``rtols`` is a typo, and the message must say what was meant.

        Anchored on the sweep's own tail rather than on the key: a bare
        ``match="rtol"`` is satisfied by ``rtol: is a number`` and by
        ``rtol: must be >= 0`` as well, and a bare ``match="rtols"`` cannot
        see a key set that has grown or lost a member.
        """
        with pytest.raises(
                ConfigError,
                match=r"it takes \['at', 'names', 'rtol'\]") as caught:
            run_document(diagnostic_document(
                {"kind": "identifiability", "rtols": 0.6}))
        assert "does not take ['rtols']" in str(caught.value)

    def test_without_parameters_it_is_refused(self):
        doc = diagnostic_document({"kind": "identifiability"})
        doc["inference"] = {}
        with pytest.raises(ConfigError, match="inference.parameters"):
            run_document(doc)

    def test_names_is_a_list_even_for_a_block_of_one(self):
        """identifiability reads a bare string as ONE name (:180) while
        score_directions splits it into characters, so this refusal is the
        config layer's own: ``names: g`` means two different things to the
        two kinds, and ``[g]`` means one thing to both.

        The prefix is asserted alongside because this run carries ``name:
        probe`` while its kind is ``identifiability`` -- under that name
        ``runs[0]``, ``runs['identifiability']`` and ``runs['probe']`` are
        three different strings, and only one of them is the contract.
        """
        with pytest.raises(ConfigError,
                           match="non-empty list of latent") as caught:
            run_document(diagnostic_document({"name": "probe",
                                   "kind": "identifiability", "names": "g"}))
        assert str(caught.value).startswith("runs['probe']: ")

    def test_at_is_a_mapping_of_latent_to_value(self):
        """The prefix is asserted here because ``_at_values`` builds its own
        ``where``, and both of its refusals read it.

        Neither ``at:`` refusal said anything about the prefix until this
        line: hard-coding ``where = "runs[0]"`` inside the helper survived the
        whole module (measured), which is the index form ``runs.py`` uses and
        the executors must not.  Under ``name: probe`` the three spellings are
        three different strings.
        """
        with pytest.raises(ConfigError, match="at: is a mapping") as caught:
            run_document(diagnostic_document({"name": "probe",
                                   "kind": "identifiability", "at": ["d"]}))
        assert str(caught.value).startswith("runs['probe']: ")

    def test_the_at_helper_returns_an_empty_mapping_rather_than_none(self):
        """Plan section 3.1 pins ``{}``, and only a direct call can see it.

        Both executors guard with ``if at:`` and None is falsy, so from a
        document the two empties are indistinguishable -- measured: a helper
        returning None survives every other test in this module.  Task 8's
        ``gradient`` is the third caller of this one helper and will be
        written against the annotated ``dict``, so the contract is asserted
        where it is visible instead of being left to the first caller that
        stops guarding.
        """
        from rheplicant.config.sections.diagnostics import _at_values

        (run,) = parse_runs([{"kind": "identifiability"}])
        built = load_document(diagnostic_document({"kind": "identifiability"}))
        assert _at_values(run, built, built.inference.space) == {}

    def test_rtol_is_a_number(self):
        """Anchored on the coercion's own words, not on the key.

        ``match="rtol"`` is satisfied by the sweep as well: drop ``rtol`` from
        this exit's key set and the sweep refuses ``does not take ['rtol']``,
        which contains the key and says nothing at all about its type.
        """
        with pytest.raises(ConfigError, match=r"rtol: is a number"):
            run_document(diagnostic_document({"kind": "identifiability",
                                   "rtol": "loose"}))

    def test_a_negative_rtol_is_refused_at_the_floor(self):
        """A relative tolerance below zero puts every singular value above the
        cutoff, so the rank verdict is vacuous rather than loose.  Matched on
        the floor clause, which the ``is a number`` branch does not carry."""
        with pytest.raises(ConfigError, match=r"rtol: must be >= 0"):
            run_document(diagnostic_document(
                {"kind": "identifiability", "rtol": -0.1}))


class TestScoreDirections:
    def test_the_rows_come_back_in_the_callers_order(self):
        """score_directions returns in the CALLER's order deliberately
        (reduced_basis.py:171-180): jax rebuilds the jacobian dict from its
        flattened, SORTED form, so re-keying the result hands back
        alphabetical names.  Sorted here is ``['d', 'g']``, so asking for
        ``['g', 'd']`` and getting ``['g', 'd']`` is the whole assertion --
        and the reversed ask is what makes it non-vacuous."""
        assert list(diagnostic_rows({"kind": "score_directions",
                          "names": ["g", "d"]})) == ["g", "d"]
        assert list(diagnostic_rows({"kind": "score_directions",
                          "names": ["d", "g"]})) == ["d", "g"]

    def test_no_names_means_the_declared_order_not_the_sorted_one(self):
        """inference.parameters declares g then d; sorted() would say d then
        g.  An executor that filled ``names`` in for an absent key by sorting
        the space would be caught here and nowhere else."""
        assert list(
            diagnostic_rows({"kind": "score_directions"})) == ["g", "d"]

    def test_one_row_per_scalar_degree_of_freedom(self):
        out = diagnostic_rows({"kind": "score_directions",
                               "names": ["g", "d"]})
        assert out["g"].shape == (1, 128)
        assert out["d"].shape == (1, 128)

    def test_at_moves_the_derivative_the_rows_report(self):
        """``d(data)/dd`` is ``g * gaussian``, so evaluating at ``g = 2.0``
        instead of the declared init 1.0 scales the ``d`` row by exactly two:
        measured max|row| 0.9898477 against 1.9796954.  Both are pinned, not
        only the ratio -- a run that dropped ``at:`` returns the first row
        twice, and one that rescaled the whole Jacobian keeps the ratio."""
        base = diagnostic_rows({"kind": "score_directions", "names": ["d"]})
        moved = diagnostic_rows({"kind": "score_directions", "names": ["d"],
                      "at": {"g": 2.0}})
        assert float(np.max(np.abs(base["d"]))) == pytest.approx(0.9898477,
                                                                 rel=1e-5)
        assert float(np.max(np.abs(moved["d"]))) == pytest.approx(1.9796954,
                                                                  rel=1e-5)

    def test_at_naming_an_undeclared_latent_is_refused_here_too(self):
        """The same hole, closed on the other route.

        ``_at_values`` takes the ParameterSpace as its third argument, so an
        executor that skipped the call, or that handed it something without
        ``.names``, loses this refusal on ONE kind and keeps it on the other.
        Two routes, one helper, two tests.
        """
        with pytest.raises(ConfigError, match=r"at: names \['q'\]") as caught:
            run_document(diagnostic_document({"kind": "score_directions",
                                   "at": {"q": 1.0}}))
        assert "it declares ['g', 'd']" in str(caught.value)

    def test_rtol_belongs_to_identifiability_alone(self):
        """The two diagnostics take different key sets, and the sweep says so
        by naming what score_directions does take.  Anchored on that tail:
        ``match="rtol"`` would be satisfied by any refusal mentioning the key,
        including one raised because this exit had started accepting it."""
        with pytest.raises(ConfigError,
                           match=r"it takes \['at', 'names'\]") as caught:
            run_document(diagnostic_document(
                {"kind": "score_directions", "rtol": 0.6}))
        assert "does not take ['rtol']" in str(caught.value)

    def test_names_is_a_list_even_for_a_block_of_one(self):
        with pytest.raises(ConfigError, match="non-empty list of latent"):
            run_document(diagnostic_document(
                {"kind": "score_directions", "names": "g"}))

    def test_without_parameters_it_is_refused(self):
        doc = diagnostic_document({"kind": "score_directions"})
        doc["inference"] = {}
        with pytest.raises(ConfigError, match="inference.parameters"):
            run_document(doc)
