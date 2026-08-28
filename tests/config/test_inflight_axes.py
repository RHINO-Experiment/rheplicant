"""C1 and C2 -- ``inflight/axes.py``.

**Every message here is pinned by EQUALITY on its whole text**, against a
literal written out in this file.  A ``match=`` is a search: it pins one
fragment and leaves every other clause free to be wrong, and Plan 3A's record
is that its surviving mutants lived almost entirely inside refusal text.

**Registry and FINDINGS assertions are subset-shaped only** (``AXIS_CHECKS["C1"]
is _time_axis``, ``{"C1", "C2"} <= report.checks()``).  A length, an exact set or
an insertion index would go red the day Task 7 registers into the other slot,
which is a merge hazard rather than a property.  The same applies to "this
document earns nothing": six of the wave-1 branches land checks that run on the
same documents, so an ``== ()`` here would go red on a correct merge.  Every
such assertion is scoped to :data:`MINE` -- the ids THIS module is about --
through :func:`silent_here`, and the property each test actually carries is the
whole-message pin beside it.
"""

import sys
import time

import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE
from rheplicant.config.inflight import AXIS_CHECKS, axes
from rheplicant.config.inflight.axes import (
    _pointing_finite,
    _time_axis,
    _time_where,
)
from tests.config.inflight_helpers import (
    axis_facts,
    axis_findings,
    axis_only,
    best_ms,
    machine_factor,
)
from tests.config.message_binding import assert_bound_once, modules_carrying
from tests.config.preflight_helpers import UNREADABLE_BEAM, preflight_document

NAN = float("nan")

# --- the documents ---------------------------------------------------------
#
# C1's is `core/coordinates.py`'s own measured example: eight samples 100 s
# apart anchored at unix 1.75e9, where float32 quantises onto a 128 s grid and
# two of the eight have already merged.  Eight samples rather than the plan's
# 262144, deliberately: the property is the ARITHMETIC, the document is 20x
# cheaper to build, and the 262144 case is measured once below for the cost
# claim rather than in every test.

C1_TIME = {"time": {"grid": {"arange": {"start": 1.75e9, "step": 100.0,
                                        "num": 8}, "unit": "s"}}}
C1_FIXED = {"time": {"grid": {"arange": {"start": 0.0, "step": 100.0,
                                         "num": 8}, "unit": "s"}}}
C2_TIME = {"time": {"grid": {"list": [0.0, 2.0, NAN, 6.0], "unit": "s"}}}
C2_TIME_FIXED = {"time": {"grid": {"list": [0.0, 2.0, 4.0, 6.0], "unit": "s"}}}

_LST = [float(i) * 10.0 for i in range(16)]
LST_WITH_NAN = [*_LST[:5], NAN, *_LST[6:]]


#: The bare ids this module is about.  Every "earns nothing" assertion is
#: scoped to these; see the module docstring.
MINE = frozenset({"C1", "C2"})


def _extra(key, values):
    return {"extra": {key: {"value": values, "unit": "deg"}}}


def ids_of(document):
    return {one.check for one in axis_findings(document)}


def silent_here(document):
    """Did this document earn nothing from THIS module's own checks?"""
    return ids_of(document).isdisjoint(MINE)


# --- the messages, whole ---------------------------------------------------

C1_MESSAGE = (
    "coords.time is stored as float32 and reaches 1.75000064e+09, where "
    "consecutive representable numbers are 128 apart — but the closest two "
    "distinct samples on this axis are 128 apart, and coords.time must "
    "resolve its own sampling to at most 0.01 of that. The rounding happens "
    "when the axis is STORED, so no later subtraction recovers it: samples "
    "merge, and every consumer that does arithmetic on the values then reads "
    "the rounded ones — BackendOperator averages merged times into chunk "
    "timestamps wrong by tens of seconds, and CWCalibrationOperator drifts "
    "its tone against them. Neither raises, because their own consistency "
    "checks compare the corrupted values against each other. "
    "read_rhino_observation reports unix seconds (~1.75e9), which quantise "
    "onto a 128 s grid in float32; rheplicant.radio.rhino.to_state therefore "
    "stores time measured from the start of the run and keeps the epoch in "
    "meta['time_epoch_unix_s']. Do the same for a hand-built axis, or enable "
    "float64 (JAX_ENABLE_X64=1, or jax.config.update('jax_enable_x64', "
    "True)). This is the axes pass, which runs after build_observation and "
    "before build_resources. The same guard runs again at Coordinates(...), "
    "which config/document.py reaches only after build_model -- so until now "
    "an axis this arithmetic refuses cost a beam read and a spherical "
    "harmonic transform first, and arrived as a StateValidationError rather "
    "than as this layer's own refusal (check C1)."
)

C2_TIME_MESSAGE = (
    "coords.time holds 1 non-finite value(s), the first at index 2 (0-based). "
    "A sample time is never legitimately NaN or infinite, and it is refused "
    "here rather than left to the resolution check below because NaN compares "
    "False against every bound: a NaN gap is not positive, so it drops out of "
    "the sampling estimate, and an all-NaN axis has no gap left to test at "
    "all and would pass a comparison-based guard untouched. This is the axes "
    "pass, which runs after build_observation and before build_resources. The "
    "same guard runs again at Coordinates(...), which config/document.py "
    "reaches only after build_model -- so until now an axis this arithmetic "
    "refuses cost a beam read and a spherical harmonic transform first, and "
    "arrived as a StateValidationError rather than as this layer's own "
    "refusal (check C2)."
)

#: The invented C2 sentence, with the two holes a subject fills.
_C2_TAIL = (
    " holds 1 non-finite value(s), the first at sample 5 (0-based). Nothing "
    "downstream says so and nothing raises: measured, a NaN at index 5 of a "
    "32-sample lst_deg loads clean and evaluates clean, and schema §6's C2 "
    "row records what follows -- the adjoint returns a finite, "
    "correctly-shaped, identically ZERO map. So a check on the OUTPUT sees no "
    "NaN, a gradient of zero everywhere, and a fit that looks converged. A "
    "sample time, an LST, a pointing and a self-rotation are never "
    "legitimately NaN or infinite (check C2)."
)

LST_MESSAGE = ('observation.extra.lst_deg: coords.extra["lst_deg"]' + _C2_TAIL)
SELFROT_MESSAGE = (
    'observation.extra.selfrot_deg: coords.extra["selfrot_deg"]' + _C2_TAIL)
POINTING_MESSAGE = (
    "observation.pointing: coords.pointing holds 16 non-finite value(s), the "
    "first at sample 0 (0-based)." + _C2_TAIL.split("(0-based).", 1)[1]
)


class TestTheRegistry:
    """Subset forms only -- see the module docstring for why."""

    def test_one_function_carries_C1_and_C2s_time_leg(self):
        assert AXIS_CHECKS["C1"] is _time_axis
        assert AXIS_CHECKS["C2.time"] is _time_axis

    def test_the_other_three_legs_of_C2_are_a_second_function(self):
        assert AXIS_CHECKS["C2.pointing"] is _pointing_finite
        assert AXIS_CHECKS["C2.pointing"] is not AXIS_CHECKS["C2.time"]

    def test_all_three_slots_are_claimed(self):
        assert {"C1", "C2.time", "C2.pointing"} <= set(AXIS_CHECKS)

    def test_the_findings_carry_the_BARE_id(self):
        """``"C2.time"`` is a registry SLOT; a reader looks up ``C2``.

        Two functions decide parts of C2, which is exactly what a dotted slot
        is for (3A's ``A1.runs``/``A1.variants`` precedent), and both must
        report the same id or a document breaking both looks like two rules.

        Subset-shaped, and the SLOT name is asserted absent rather than the
        set asserted equal: ``== {"C2"}`` carries the same property today and
        goes red the day any wave-1 check fires on a NaN axis.
        """
        for document in (preflight_document(observation=C2_TIME),
                         preflight_document(
                             observation=_extra("lst_deg", LST_WITH_NAN))):
            found = ids_of(document)
            assert "C2" in found
            assert "C2.time" not in found and "C2.pointing" not in found


class TestTheModuleIsImportedAtAllAndTheEntryPointSurvivesIt:
    """The registration hazard this package has and ``preflight/`` does not:
    the entry point ``axes()`` and the module ``axes.py`` share a name.

    Both wrong orderings are SILENT, which is why both are pinned here rather
    than described in a comment:

    * the foot-import form ``from ... import axes as _axis_checks`` written
      BELOW ``def axes`` binds the function and never imports the module, so
      C1 and C2 register nowhere and every test that asks for them finds an
      empty registry -- caught by :meth:`test_the_module_really_was_imported`;
    * ``import rheplicant.config.inflight.axes`` written below it shadows the
      entry point with the module, and ``document.py``'s axes hook then raises
      ``'module' object is not callable`` -- caught by
      :meth:`test_the_package_attribute_is_the_entry_point`.
    """

    def test_the_module_really_was_imported(self):
        assert "rheplicant.config.inflight.axes" in sys.modules
        assert {"C1", "C2.time", "C2.pointing"} <= set(AXIS_CHECKS)

    def test_the_package_attribute_is_the_entry_point(self):
        import rheplicant.config.inflight as package
        import rheplicant.config.inflight.axes  # noqa: F401 -- the point

        assert package.axes is axes
        assert callable(package.axes)

    def test_the_hook_still_runs_after_the_submodule_is_imported(self):
        """The end-to-end form of the above: a shadowed entry point makes
        every load raise, and this is the assertion that says so in one line
        rather than through 5000 unrelated failures."""
        assert load_document(preflight_document()) is not None


class TestC1:
    """The time axis the STORED dtype cannot carry."""

    def test_it_fires_on_the_containers_own_measured_example(self):
        found = axis_only(preflight_document(observation=C1_TIME), "C1")
        assert found.severity == REFUSE
        assert found.where == "observation.time.grid"

    def test_the_whole_message(self):
        assert axis_only(preflight_document(observation=C1_TIME),
                         "C1").message == C1_MESSAGE

    def test_the_hoisted_sentence_is_still_bound_where_it_was(self):
        """The words a reader sees are ``core/coordinates.py``'s, interpolated
        rather than copied.  ``coords.time is stored as`` itself is the
        walker's one documented exemption (that clause really does live in two
        modules -- the container's guard and ``CWCalibrationOperator``'s), so
        this asks for the clause that is the CONTAINER's alone."""
        assert modules_carrying(
            "must resolve its own sampling to at most") == ("core/coordinates.py",)

    def test_the_base_document_earns_no_C1(self):
        assert "C1" not in {one.check for one in
                            axis_findings(preflight_document())}

    def test_an_ingested_run_is_sent_to_its_own_key(self):
        """``_time_where`` as a unit, because the branch is otherwise reachable
        only through a real HDF5 recording.  An ingested run has no
        ``observation.time`` at all -- ``build_observation`` refuses the pair
        with "from_file and time: together say two things about one
        recording" -- so a refusal naming ``observation.time.grid`` would send
        the reader to a key their document must not contain."""
        assert _time_where({"observation": {"from_file": {"path": "r.h5"}}}) \
            == "observation.from_file"
        assert _time_where({"observation": {"time": {}}}) \
            == "observation.time.grid"
        assert _time_where({}) == "observation.time.grid"


class TestC2sTimeLeg:
    """A NaN on the time axis, named before any comparison."""

    def test_it_fires_and_carries_C2(self):
        found = axis_only(preflight_document(observation=C2_TIME), "C2")
        assert found.severity == REFUSE
        assert found.where == "observation.time.grid"

    def test_the_whole_message(self):
        assert axis_only(preflight_document(observation=C2_TIME),
                         "C2").message == C2_TIME_MESSAGE

    def test_an_all_NaN_axis_is_refused_too(self):
        """The reason non-finite values are named FIRST.  Every gap on an
        all-NaN axis is NaN, NaN is not ``> 0``, so the smallest DISTINCT gap
        is empty and a comparison-based guard has nothing left to test.  A
        check that asked about the resolution first would let this through."""
        found = axis_only(preflight_document(observation={
            "time": {"grid": {"list": [NAN, NAN, NAN, NAN], "unit": "s"}}}),
            "C2")
        assert found.message.startswith(
            "coords.time holds 4 non-finite value(s), the first at index 0")

    def test_an_integer_axis_is_left_alone(self):
        """``np.spacing`` on an integer promotes to float64 and answers
        5.7e-14 -- the same dtype blindness from the other side.  The hoisted
        guard checks INEXACT dtypes only, and this is the test that the hoist
        did not lose that."""
        assert silent_here(preflight_document(observation={
            "time": {"grid": {"list": [0, 100, 200, 300], "unit": "s"}}}))


class TestC2sOtherThreeLegs:
    """``pointing``, ``lst_deg`` and ``selfrot_deg`` -- missing entirely
    before this plan, and silently: measured, a NaN at index 5 of a 32-sample
    ``lst_deg`` loads clean and evaluates clean."""

    def test_lst_deg(self):
        found = axis_only(preflight_document(
            observation=_extra("lst_deg", LST_WITH_NAN)), "C2")
        assert found.where == "observation.extra.lst_deg"
        assert found.message == LST_MESSAGE

    def test_selfrot_deg(self):
        """The named twin beside ``lst_deg``, guarded rather than recorded."""
        found = axis_only(preflight_document(
            observation=_extra("selfrot_deg", LST_WITH_NAN)), "C2")
        assert found.where == "observation.extra.selfrot_deg"
        assert found.message == SELFROT_MESSAGE

    def test_the_pointing_table(self):
        found = axis_only(preflight_document(observation={
            "pointing": {"mode": "drift", "materialise": ["pointing"],
                         "el_deg": {"value": NAN, "unit": "deg"}}}), "C2")
        assert found.where == "observation.pointing"
        assert found.message == POINTING_MESSAGE

    def test_the_index_is_the_SAMPLE_and_not_the_flat_position(self):
        """``coords.pointing`` is ``(n_time, k)``.  A flat index would report
        1 for a NaN ELEVATION on sample 0 and send the reader to sample 1."""
        assert "the first at sample 0 (0-based)" in axis_only(
            preflight_document(observation={
                "pointing": {"mode": "drift", "materialise": ["pointing"],
                             "el_deg": {"value": NAN, "unit": "deg"}}}),
            "C2").message

    def test_the_where_falls_back_to_the_producer_that_wrote_it(self):
        """``coords.extra['selfrot_deg']`` has two producers and
        ``build_observation`` refuses a document that uses both.  Written
        through ``pointing.materialise`` there is no ``observation.extra`` key
        to name, so the reader is sent to ``observation.pointing``."""
        found = axis_only(preflight_document(observation={
            "pointing": {"mode": "drift", "materialise": ["selfrot_deg"],
                         "selfrot_deg": {"value": NAN, "unit": "deg"}}}), "C2")
        assert found.where == "observation.pointing"

    def test_two_bad_axes_are_two_findings(self):
        """One sentence per line to edit.  ``axis_only`` would fail here, and
        that is the point of it being ``exactly one``."""
        found = axis_findings(preflight_document(
            observation={"extra": {
                "lst_deg": {"value": LST_WITH_NAN, "unit": "deg"},
                "selfrot_deg": {"value": LST_WITH_NAN, "unit": "deg"}}}))
        assert [one.where for one in found if one.check == "C2"] == [
            "observation.extra.lst_deg", "observation.extra.selfrot_deg"]

    def test_an_extra_key_C2_does_not_name_is_left_alone(self):
        """**The trap, and the mutant it kills.**  ``coords.extra`` is an OPEN
        dict: ``observation.extra`` writes whatever the document declares.
        Schema §6's C2 row names four axes and no others, and a user's own
        array is allowed to carry NaN -- a masked weight, a sentinel, a
        deliberately-blanked channel.  Walking the mapping instead of the
        named keys refuses documents that are not wrong.

        Measured: with the walk over ``extra`` itself, this document earns a
        C2 finding at ``observation.extra.my_weights``.
        """
        assert silent_here(preflight_document(
            observation=_extra("my_weights", LST_WITH_NAN)))
        assert "C2" in ids_of(preflight_document(observation={"extra": {
            "my_weights": {"value": LST_WITH_NAN, "unit": "deg"},
            "lst_deg": {"value": LST_WITH_NAN, "unit": "deg"}}})), (
            "the named key beside it must still be decided, or this test "
            "passes because the walk found nothing at all"
        )

    def test_an_integer_extra_is_not_read_as_an_angle_grid(self):
        """``coords.extra`` is an OPEN dict -- ``compile_switching`` writes
        ``receiver_input``, an integer index vector, into it.  The walk is
        over the NAMED keys for that reason, and this is the anti-vacuity
        case: a document with a switch cycle earns nothing here."""
        facts = axis_facts(preflight_document(observation={
            "switching": {"mode": "cycle", "order": ["antenna", "load"],
                          "dwell": 4}}))
        assert "receiver_input" in facts.observation.extra
        assert {one.check for one in axes(facts).findings}.isdisjoint(MINE)


class TestTheyDoNotPreEmptABetterSentence:
    """S4's first half: a document wrong in THIS way and wrong in a way
    something else says better."""

    def test_the_text_pass_still_wins_over_a_bad_time_axis(self):
        """P-1 runs before P-0.5, so an unknown ``model:`` node -- which the
        document's TEXT decides -- is heard first.  A C1 that pre-empted it
        would send the reader to the observation section over a typo in the
        model."""
        with pytest.raises(ConfigError) as raised:
            load_document(preflight_document(model={"gian": {}},
                                             observation=C1_TIME))
        assert "gian" in str(raised.value)
        assert "coords.time is stored as" not in str(raised.value)

    def test_a_pointing_section_the_builder_refuses_never_reaches_C2(self):
        """``compile_pointing`` runs INSIDE ``build_observation``, one line
        before this pass, and says what a bad ``materialise:`` is in its own
        words.  There is nothing for C2 to be about, because there is no
        materialised axis."""
        with pytest.raises(ConfigError) as raised:
            load_document(preflight_document(observation={
                "pointing": {"mode": "drift", "materialise": ["nope"]}}))
        assert "materialise entries are" in str(raised.value)


class TestApplyingTheirOwnAdvice:
    """S4's second half, and 3A shipped three checks that failed it: take the
    fix the message names, apply it, and assert the document LOADS."""

    def test_C1s_remedy_builds(self):
        """*"stores time measured from the start of the run"* -- so the axis
        moves from unix 1.75e9 to 0.0 and everything else stays."""
        assert load_document(preflight_document(
            observation=C1_FIXED)) is not None

    def test_C2s_time_remedy_builds(self):
        assert load_document(preflight_document(
            observation=C2_TIME_FIXED)) is not None

    def test_C2s_pointing_remedy_builds(self):
        assert load_document(preflight_document(
            observation=_extra("lst_deg", _LST))) is not None


class TestThePhaseProperty:
    """The box §5 calls "one real assertion per hoisting task, never a
    lambda": the violation is heard, and the beam is not read.

    Before this plan the beam won -- ``document.py`` reaches
    ``build_resources`` before ``Coordinates(...)`` is ever constructed, and
    measured at ``e0e024a`` an unreadable beam out-ranked five different
    document faults, five for five.
    """

    def test_a_bad_time_axis_beats_an_unreadable_beam(self):
        with pytest.raises(ConfigError) as raised:
            load_document(preflight_document(observation=C1_TIME,
                                             resources=UNREADABLE_BEAM))
        assert str(raised.value) == C1_MESSAGE
        assert "no_such_beam" not in str(raised.value)

    def test_a_NaN_lst_beats_an_unreadable_beam(self):
        with pytest.raises(ConfigError) as raised:
            load_document(preflight_document(
                observation=_extra("lst_deg", LST_WITH_NAN),
                resources=UNREADABLE_BEAM))
        assert str(raised.value) == LST_MESSAGE
        assert "no_such_beam" not in str(raised.value)

    def test_the_refusal_is_this_layers_own_type(self):
        """Today the same arithmetic arrives as a ``StateValidationError``
        from ``core/``, which a caller wrapping ``load_document`` in ``except
        ConfigError`` to report "this document is wrong" does not catch."""
        with pytest.raises(ConfigError):
            load_document(preflight_document(observation=C1_TIME))


class TestOneBindingPerRule:
    """§3.2(h) for THIS module's own literals.

    There is deliberately no shared table (§0.3 C.4): two tasks landing in
    parallel would both edit it, the merge would keep one side, and the rows
    that vanished are exactly the rows nobody is checking any more.
    """

    @pytest.mark.parametrize("literal", [
        "This is the axes pass, which runs after build_observation and "
        "before build_resources.",
        "Nothing downstream says so and nothing raises: measured, a NaN at "
        "index 5 of a 32-sample lst_deg loads clean and evaluates clean",
        "A sample time, an LST, a pointing and a self-rotation are never "
        "legitimately NaN or infinite",
    ])
    def test_each_sentence_this_module_invents_is_bound_once(self, literal):
        assert_bound_once(literal)

    def test_the_hoist_moved_rather_than_copied(self):
        """Two clauses of C1's sentence, each in exactly one module.  A hoist
        that pasted the words into ``inflight/axes.py`` would put both in two,
        and every ``match=`` pin in the repository would still pass."""
        for literal in ("The rounding happens when the axis is STORED",
                        "an all-NaN axis has no gap left to test at all"):
            assert modules_carrying(literal) == ("core/coordinates.py",), literal


class TestTheCost:
    """**Measured on the pass in isolation**, because a fraction-of-a-load
    bound cannot fail for the reason it names: a warm ``load_document`` on the
    worked document is 2.77 ms median with a 3.94 ms maximum (Task 1a's
    measurement), so anything under ~1 ms is inside the spread.

    **And measured NEAR the number.**  A review of Task 1a found all three of
    its cost bounds unable to fail -- a *thousandfold* slowdown of the shared
    ``sweep`` left the suite at exit 0, at margins of x3008 and x30077.  A
    review of Task 1b then found two of the replacements surviving a clean
    ``x10``, so every bound below is now about **six** times its own measured
    best case, which is the largest margin that still dies at ``x10``.  Each
    measurement is written beside its bound so the next task can see what it
    is keeping:

    ==============================  =========  =======
    call                            best       bound
    ==============================  =========  =======
    ``axes`` on the 16-sample doc    0.0138 ms  0.09 ms
    ``axes`` on 262144 samples       0.291 ms   1.8 ms
    ``axes`` on the REFUSING 262144  0.257 ms   1.6 ms
    ==============================  =========  =======

    **Every timing is taken with** :func:`~tests.config.inflight_helpers.best_ms`
    **on an already-built payload**, and that is a correction rather than a
    style.  A single ``perf_counter`` around ``axis_findings`` times the COLD
    payload build as well: measured in one process, 12 repeats of that call are
    ``best 0.0005 s / median 0.0007 s / max 0.107 s`` -- the maximum being the
    FIRST call.  A 0.1 s bound around it therefore passes only when earlier
    tests in the module have warmed the process, and goes red the moment the
    module is run alone, run first, or distributed by ``xdist``: 13 runs out of
    13 in a fresh process.

    **What these bounds cannot see:**

    * the cost of BUILDING the payload -- ``axis_facts`` is ~0.14 ms, and that
      is ``build_runtime`` plus ``build_observation``, not this pass.
    * a check that is slow only on a document none of these are.  The 262144
      sample cell is the largest axis this plan's own traps name, and it is the
      one that matters, but it is not a proof about every document.  A
      projector-and-tone document is measured in ``test_inflight_grids.py``.
    * a cold first call.  Everything here is warm by construction.
    * anything about ``build_resources``, which is what the slot exists to run
      in front of and is 90.9 % of a load's wall time.
    """

    @pytest.mark.parametrize(("num", "bound"), [(16, 0.09), (262144, 1.8)],
                             ids=["worked", "big"])
    def test_the_axes_pass_stays_near_its_measured_cost(self, num, bound):
        document = preflight_document(observation={
            "time": {"grid": {"arange": {"start": 0.0, "step": 2.0,
                                         "num": num}, "unit": "s"}}})
        facts = axis_facts(document)
        axes(facts)  # warm
        assert best_ms(lambda: axes(facts), repeats=30) < bound * machine_factor()

    def test_the_plans_own_hundredth_of_a_second_box(self):
        """§0.1's contract for this slot, kept verbatim and kept KNOWING it is
        two orders of magnitude above the measurement.  It is the number a
        later task must not break; the parametrized test above is the one that
        can actually fail."""
        facts = axis_facts(preflight_document())
        axes(facts)  # warm
        started = time.perf_counter()
        axes(facts)
        assert time.perf_counter() - started < 0.01

    def test_the_pass_refuses_the_big_axis(self):
        """The claim the slot exists for: the 262144-sample float32 axis
        anchored at unix 1.75e9 is refused HERE, by C1, and not two phases and
        one beam read later.  The property only; the cost of refusing it is
        the test below, because a single ``perf_counter`` around
        ``axis_findings`` also times the cold payload build and then fails for
        a reason it does not name."""
        assert "C1" in ids_of(preflight_document(observation={
            "time": {"grid": {"arange": {"start": 1.75e9, "step": 1.0,
                                         "num": 262144}, "unit": "s"}}}))

    def test_refusing_the_big_axis_costs_a_fraction_of_the_load_it_replaces(self):
        """**1.6 ms against a measured 0.257 ms best case.**  The REFUSING
        262144-sample axis and not the clean one parametrized above: the
        refusal path interpolates a 1200-character message, which the clean
        path never pays for, and it is the path this slot exists for.

        Against the **0.151 s** the same document takes to reach the same
        arithmetic today -- a load that has already read a beam and run a
        spherical harmonic transform -- that is two orders of magnitude, and
        the payload build (~2 ms at this size) is outside the timed region
        because it is ``build_observation``'s cost and not this pass's.
        """
        facts = axis_facts(preflight_document(observation={
            "time": {"grid": {"arange": {"start": 1.75e9, "step": 1.0,
                                         "num": 262144}, "unit": "s"}}}))
        assert "C1" in {one.check for one in axes(facts).findings}, (
            "the timed call must be the REFUSING one, or this measures the "
            "clean path the test above already covers"
        )
        assert best_ms(lambda: axes(facts), repeats=30) < 1.6 * machine_factor()
