"""The runner three passes share, the two new slots, and the seams they need.

**This module tests a MECHANISM and registers no real check.**  Plan 3B's
Task 1a ships the machinery and the artefacts six later tasks build on; the
checks that go in the two new registries (C1, C2, C3, A13.grid, C8, and Task
7's five) arrive afterwards.  So the registries are EMPTY here, and every
assertion about them is either driven by a probe this module registers into a
CLEARED registry, or written in a subset form that stays true as later tasks
land.

**The four legal registry assertion forms, and nothing else:**
``AXIS_CHECKS["C1"] is _f`` · ``{"C1", "C2"} <= report.checks()`` ·
``"C8" not in ids`` · ``axis_only(doc, "C1")``.  **Banned**: ``len(...)``,
``set(...) == {...}``, any insertion-index assertion, ``ids(doc) ==
frozenset({...})``, ``len(report.findings) == n``, ``report.refusals()[0]`` --
every one of them is a function of how many tasks have landed, so it is green
in one branch and red after a merge that changed nothing it was about.

**The two clearing fixtures below are module-local on purpose.**  A later task
that asks for one gets ``fixture not found``, which is the answer: what a
later task must do instead is register nothing from a test body and read the
live registries in subset forms only.

**Anti-vacuity is deliberate throughout.**  The registries are empty as this
module lands, so an assertion of the form "every id in ``AXIS_CHECKS`` is a §6
id" is trivially true and stays trivially true if the matcher breaks.  Every
such assertion is paired with one that fails when its matcher stops matching.
"""

import ast
import dataclasses
import pathlib
import statistics
import subprocess
import sys
import time
import warnings

import numpy as np
import pytest

import rheplicant.config.document as document_module
from rheplicant.config.document import ConfiguredRun, _assemble, load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import ConfigWarning, refuse, report, warn
from rheplicant.config.inflight import (
    AXIS_CHECKS,
    BUILT_CHECKS,
    Built,
    axes,
    built,
    register_axes,
    register_built,
)
from rheplicant.config.passes import SLOT
from rheplicant.config.resources import resolved_specs
from tests.config.inflight_helpers import (
    axis_facts,
    axis_findings,
    axis_only,
    best_ms,
    built_findings,
    built_only,
    built_run,
)
from tests.config.message_binding import (
    assert_bound_once,
    exempt_pairs_still_hold,
    modules_carrying,
)
from tests.config.preflight_helpers import UNREADABLE_BEAM, preflight_document
from tests.config.test_config_preflight import _foot_imports

#: ``tests/config/`` -> the repository root.
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DOCUMENT_PY = _ROOT / "src" / "rheplicant" / "config" / "document.py"
_INFLIGHT_DIR = _ROOT / "src" / "rheplicant" / "config" / "inflight"
_PASSES_PY = _ROOT / "src" / "rheplicant" / "config" / "passes.py"


@pytest.fixture
def axis_registry():
    """:data:`AXIS_CHECKS`, EMPTIED for one test and restored afterwards.

    Restored, because a probe left behind leaks into every later
    ``load_document`` in the session -- including other modules' -- and the
    failures land nowhere near the cause.  **Emptied**, because Task 1b and
    Task 7 register real ids into these same dicts at import time: a test that
    registered ``C1`` would then hit "registered twice", and one that asserts a
    refusal's message VERBATIM would read a real check's refusal instead of
    its own.
    """
    saved = dict(AXIS_CHECKS)
    AXIS_CHECKS.clear()
    try:
        yield AXIS_CHECKS
    finally:
        AXIS_CHECKS.clear()
        AXIS_CHECKS.update(saved)


@pytest.fixture
def built_registry():
    """:data:`BUILT_CHECKS`, emptied and restored.  See :func:`axis_registry`."""
    saved = dict(BUILT_CHECKS)
    BUILT_CHECKS.clear()
    try:
        yield BUILT_CHECKS
    finally:
        BUILT_CHECKS.clear()
        BUILT_CHECKS.update(saved)


# ---------------------------------------------------------------------------
# The payload types
# ---------------------------------------------------------------------------


class TestThePayloads:
    """Kills: ``Built`` drifting out of step with ``ConfiguredRun``, under which
    ``Built(*run)`` puts the twin where the state should be and every built
    check reads the wrong object with no error at all."""

    def test_the_two_field_tuples_are_equal(self):
        """``Built(*run)`` is the whole constructor, so the two field lists
        must agree NAME FOR NAME AND IN ORDER.  A positional splat cannot
        notice a reordering: it would put ``state`` in ``twin`` and every
        check reading ``run.twin`` would silently interrogate a ``State``.

        ``ConfiguredRun`` is a ``NamedTuple`` -- ``._fields`` and NOT
        ``dataclasses.fields``, which raises on it.  ``Built`` is a dataclass,
        so its side is ``dataclasses.fields``.  The asymmetry is the reason
        this test exists rather than a comment.
        """
        assert ConfiguredRun._fields == tuple(
            field.name for field in dataclasses.fields(Built))

    def test_configured_run_is_still_a_named_tuple(self):
        """ANTI-VACUITY for the test above.  If ``ConfiguredRun`` stopped being
        a ``NamedTuple``, ``._fields`` would raise rather than compare, and the
        pin would fail loudly -- but if it became a dataclass with a
        ``_fields`` attribute of its own the comparison could go quietly
        wrong.  Say which type it is."""
        assert issubclass(ConfiguredRun, tuple)
        assert hasattr(ConfiguredRun, "_fields")

    def test_the_splat_lands_field_for_field_on_a_real_run(self):
        """The PROPERTY, not the syntactic proxy above: build a real document
        and assert each field of the ``Built`` IS the corresponding field of
        the ``ConfiguredRun``, by identity."""
        run = _assemble(preflight_document())
        payload = Built(*run)
        for name in ConfiguredRun._fields:
            assert getattr(payload, name) is getattr(run, name), name

    def test_the_payloads_are_frozen(self):
        """A check receives the payload and must not be able to edit the run
        out from under the builders that come after it."""
        facts = axis_facts(preflight_document())
        with pytest.raises(dataclasses.FrozenInstanceError):
            facts.runtime = None

    def test_the_axes_payload_carries_no_built_resource(self):
        """**The whole reason the axes slot exists.**  It runs BEFORE
        ``build_resources``, which is 90.9 % of ``load_document``'s wall time,
        so its context's ``resources`` must still be empty.  Kills a helper
        that reached for ``load_document`` and took the pieces off the result
        -- which would pay for the beam and destroy the only property this
        slot has, while every other assertion in this file still passed."""
        facts = axis_facts(preflight_document())
        assert dict(facts.context.resources) == {}
        assert len(facts.context.time) and len(facts.context.freq)

    def test_the_built_payload_carries_an_inference_that_is_never_none(self):
        """§0.3 C.3: ``Built.inference`` is never ``None`` -- ``build_inference``
        returns one even for a document with no ``inference:`` section -- while
        ``.space`` MAY be, and a check reading it stands down rather than
        refusing on "I could not tell"."""
        with_block = built_run(preflight_document())
        assert with_block.inference is not None
        without = built_run(preflight_document(inference=None))
        assert without.inference is not None
        assert without.inference.space is None
        assert with_block.inference.space is not None


# ---------------------------------------------------------------------------
# The registries and the binder
# ---------------------------------------------------------------------------


class TestTheTwoRegistries:
    """Kills: one registry serving two payload types; a bound id that does not
    reach its function; a clash that binds half its ids."""

    def test_a_registered_id_binds_to_its_function(self, axis_registry):
        @register_axes("C1")
        def _one(facts):
            return ()

        assert AXIS_CHECKS["C1"] is _one

    def test_the_two_registries_are_separate_dicts(self, axis_registry,
                                                   built_registry):
        """**Two payload types, two registries.**  One registry would let a
        check be registered in the slot whose payload it cannot read, and the
        symptom would be an ``AttributeError`` wrapped as "in-flight check
        'C8' RAISED AttributeError" -- a stack trace shaped like a user's
        fault.  Kills ``register_built`` writing into ``AXIS_CHECKS``, which is
        a one-word slip and which no other test here would notice."""
        register_axes("C1")(lambda facts: ())
        register_built("C8")(lambda run: ())
        assert "C1" in AXIS_CHECKS and "C1" not in BUILT_CHECKS
        assert "C8" in BUILT_CHECKS and "C8" not in AXIS_CHECKS

    def test_one_id_may_be_claimed_in_each_slot_independently(
            self, axis_registry, built_registry):
        """Separate registries mean separate namespaces.  Stated rather than
        discovered: this is a CONSEQUENCE of the split, and a reader who
        expected a global id space should find the answer here."""
        register_axes("C2")(lambda facts: ())
        register_built("C2")(lambda run: ())
        assert AXIS_CHECKS["C2"] is not BUILT_CHECKS["C2"]

    def test_insertion_order_is_run_order(self, axis_registry):
        """Insertion order IS run order, and ``raise_if_refused`` shows the
        FIRST refusal verbatim -- so the order decides which sentence a user
        reads.  Kills ``for check in sorted(registry)``, which looks tidier and
        reverses exactly this."""
        for check in ("C3", "C2", "C1"):
            register_axes(check)(
                lambda facts, check=check:
                    (refuse(check, "observation.time", f"{check}!"),))
        found = axes(axis_facts(preflight_document())).refusals()
        assert [one.check for one in found] == ["C3", "C2", "C1"]

    def test_a_function_bound_to_several_ids_runs_exactly_once(
            self, axis_registry):
        """One function carries several ids and this plan does it twice: the
        time-axis function is C1 + C2.time, and Task 7's ``_divisible`` is
        C8's two clauses.  Kills a walk with no de-duplication by identity,
        under which the user reads one mistake twice."""
        calls = []

        @register_axes("C1", "C2.time")
        def _both(facts):
            calls.append(1)
            return (refuse("C1", "observation.time", "one."),)

        assert {slot: fn is _both for slot, fn in AXIS_CHECKS.items()} == {
            "C1": True, "C2.time": True}
        found = axes(axis_facts(preflight_document())).refusals()
        assert calls == [1]
        assert [one.check for one in found] == ["C1"]

    def test_two_functions_sharing_a_name_are_not_one_function(
            self, axis_registry):
        """The de-duplication is by IDENTITY and not by ``__name__``.  Every
        lambda is ``<lambda>``; a walk keyed on the name drops the second in
        silence."""
        register_axes("C1")(
            lambda facts: (refuse("C1", "observation.time", "first."),))
        register_axes("C2")(
            lambda facts: (refuse("C2", "observation.time", "second."),))
        found = axes(axis_facts(preflight_document())).refusals()
        assert [one.message for one in found] == ["first.", "second."]

    def test_a_dotted_slot_is_accepted_and_carries_the_bare_id(
            self, axis_registry):
        """``SLOT`` already admits B and C ids, which is why the two new passes
        need no widening of it.  The SLOT may be dotted; ``Finding.check``
        never is."""
        register_axes("A13.grid")(
            lambda facts: (refuse("A13", "model.cw_tone", "one."),))
        assert axes(axis_facts(preflight_document())).checks() == frozenset(
            {"A13"})

    def test_the_slot_pattern_admits_this_plans_ids_and_not_a_bare_suffix(self):
        """ANTI-VACUITY for the pattern itself, and it is the measured trap:
        ``"A12a"`` is NOT a slot.  Registering it raises AT IMPORT, which takes
        ``import rheplicant.config`` down and the whole suite to exit 2 on
        collection -- so it is worth a test rather than a discovery."""
        for good in ("C1", "C2.time", "C8", "A13.grid", "B9", "A50"):
            assert SLOT.fullmatch(good), good
        for bad in ("A12a", "C0", "_mine", "A1.", "D1", "c1"):
            assert not SLOT.fullmatch(bad), bad

    @pytest.mark.parametrize("pair", [("C3", "C1"), ("C1", "C3")],
                             ids=["clash-last", "clash-first"])
    def test_a_clash_on_any_id_refuses_and_binds_none_of_them(
            self, axis_registry, pair):
        """Kills checking only ``checks[0]`` for a clash, kills checking only
        ``checks[-1]``, and kills a non-atomic bind.  **Both orders**, because
        one order closes one direction and leaves its twin open."""
        first = register_axes("C1")(lambda facts: ())
        with pytest.raises(ConfigError, match="registered twice"):
            register_axes(*pair)(lambda facts: ())
        assert set(AXIS_CHECKS) == {"C1"}
        assert AXIS_CHECKS["C1"] is first

    def test_every_id_of_a_variadic_registration_is_validated(
            self, axis_registry):
        """Kills validating ``checks[0]`` alone, under which
        ``@register_axes("C1", "_mine")`` ships a finding tagged
        ``(check _mine).``  The second assertion kills validating as it binds:
        a refusal that leaves half the ids claimed makes the next import of
        the same module report "registered twice" about a module that never
        finished."""
        with pytest.raises(ConfigError, match="not a schema §6 id"):
            register_axes("C1", "_mine")(lambda facts: ())
        assert dict(AXIS_CHECKS) == {}

    def test_the_double_registration_refusal_is_not_an_assert(
            self, axis_registry):
        """``python -O`` strips ``assert``.  Measured on the OTHER registry
        before Plan 3A fixed it: under ``-O`` the second registration won, in
        silence.  This registry is a ``ConfigError`` from the start, and it is
        the SHARED binder, so one test covers all three passes."""
        register_axes("C1")(lambda facts: ())
        with pytest.raises(ConfigError):
            register_axes("C1")(lambda facts: ())

    def test_the_live_registries_hold_only_schema_ids(self):
        """A SUBSET form, deliberately: it is vacuous while the registries are
        empty and load-bearing from Task 1b on, and it stays true however many
        tasks have landed.  Its anti-vacuity partner is
        :meth:`test_the_slot_pattern_admits_this_plans_ids_and_not_a_bare_suffix`.
        """
        for registry in (AXIS_CHECKS, BUILT_CHECKS):
            for slot in registry:
                assert SLOT.fullmatch(slot), slot


class TestEveryRefusalOfTHESEPassesIsPinnedWHOLE:
    """The in-flight half of the label contract, by equality on whole text.

    ``config/passes.py`` is one runner parameterised by two words: the phase
    label a refusal opens with, and the DECORATOR a refusal tells the reader to
    write.  Measured while it was extracted: rewriting all six occurrences of
    ``pre-flight`` to ``in-flight`` in the pre-flight pass left the whole of
    ``tests/config`` at **exit 0**, because every pin that existed was a
    substring beginning after the word.  ``test_config_preflight.py``'s
    ``TestEveryRefusalOfThisPassIsPinnedWHOLE`` closes that side; this closes
    this one.

    **The decorator name is pinned as hard as the label**, and it is the half
    a single ``label`` parameter cannot fix: ``register_axes`` and
    ``register_built`` share the label ``"in-flight"`` and differ here, so
    advice naming the wrong one is advice that cannot be followed (S4).
    """

    def test_the_axes_binder_advises_its_own_decorator(self, axis_registry):
        with pytest.raises(ConfigError) as caught:
            register_axes()(lambda facts: ())
        assert str(caught.value) == (
            "register_axes() takes one or more check ids -- "
            "@register_axes('A30'), or @register_axes('A16', 'A17') when one "
            "function decides several. A registration with no id binds "
            "nothing, so the check it decorates never runs and nothing says "
            "so."
        )

    def test_the_built_binder_advises_its_own_decorator(self, built_registry):
        with pytest.raises(ConfigError) as caught:
            register_built()(lambda run: ())
        assert str(caught.value) == (
            "register_built() takes one or more check ids -- "
            "@register_built('A30'), or @register_built('A16', 'A17') when "
            "one function decides several. A registration with no id binds "
            "nothing, so the check it decorates never runs and nothing says "
            "so."
        )

    def test_an_id_that_is_not_a_string(self, axis_registry):
        with pytest.raises(ConfigError) as caught:
            register_axes(7)(lambda facts: ())
        assert str(caught.value) == (
            "in-flight check id 7 is not a string. @register_axes is called "
            "with its ids -- @register_axes('A30') -- and a bare "
            "@register_axes hands the decorated function in as an id."
        )

    def test_an_id_that_is_not_a_slot(self, built_registry):
        with pytest.raises(ConfigError) as caught:
            register_built("_mine")(lambda run: ())
        assert str(caught.value) == (
            "in-flight check id '_mine' is not a schema §6 id (A1..A52, "
            "B1..B9, C1..C19), optionally with a dotted suffix such as "
            "'A1.runs' when several functions each decide part of one check. "
            "The id is what a Finding carries and what a reader looks up; a "
            "private name here reaches the user as '(check _mine).'"
        )

    def test_one_id_named_twice_in_one_registration(self, axis_registry):
        with pytest.raises(ConfigError) as caught:
            register_axes("C1", "C1")(lambda facts: ())
        assert str(caught.value) == (
            "this registration names 'C1' twice. The variadic form binds one "
            "function to several DIFFERENT ids -- @register_axes('A16', "
            "'A17') -- and a repeated id is a typo for one that is now "
            "claimed by nobody."
        )

    def test_an_id_a_second_function_claims(self, axis_registry):
        """**The two modules are made DIFFERENT on purpose.**  Both functions
        are defined here, so both interpolations read ``test_config_inflight``
        and the sentence cannot tell them apart -- measured, naming the
        NEWCOMER twice (``fn.__module__`` in both holes) leaves this test
        green, and the refusal would then never name the incumbent at all.
        Which is the half a reader needs: "registered twice" is only
        actionable if it says where the first one is.
        """
        def _incumbent(facts):
            return ()

        _incumbent.__module__ = "rheplicant.config.inflight.axes"
        register_axes("C1")(_incumbent)
        with pytest.raises(ConfigError) as caught:
            register_axes("C1")(lambda facts: ())
        assert str(caught.value) == (
            "in-flight check 'C1' is registered twice, by "
            "rheplicant.config.inflight.axes and by test_config_inflight. A "
            "check id has one function, and which of the two would run "
            "depends on import order."
        )

    def test_a_check_that_raises_names_its_own_slot(self, axis_registry):
        """Kills naming whichever slot the loop happened to be on -- the first,
        or the last -- rather than the raiser's, and kills a bare
        ``except: pass``, under which the raising check's findings AND every
        later check's vanish and the document loads."""
        register_axes("C1")(lambda facts: ())

        @register_axes("C2", "C3")
        def _bad(facts):
            raise KeyError("noise")

        with pytest.raises(ConfigError) as caught:
            axes(axis_facts(preflight_document()))
        assert str(caught.value) == (
            "in-flight check 'C2' RAISED KeyError: 'noise'. A check returns "
            "findings and raises nothing -- one that raises aborts the pass "
            "and hides every finding after it, which is the failure the "
            "collect-rather-than-raise design exists to prevent."
        )

    def test_a_where_that_is_not_a_document_path(self, built_registry):
        """``Finding.where`` is a path into the USER'S document **at every
        slot, the built one included** -- which is why the payload carries
        ``document`` at all.  A built check that wrote an object repr or a
        source path would send the reader nowhere."""
        register_built("C8")(lambda run: (refuse("C8", "", "a sentence."),))
        with pytest.raises(ConfigError) as caught:
            built(built_run(preflight_document()))
        assert str(caught.value) == (
            "in-flight check 'C8' emitted where='', which is not a document "
            "path (Path '' is empty or padded with whitespace. A path is "
            "'head' or 'head.step.step', where head names a graph node and "
            "each step is an attribute, optionally with a non-negative "
            "index.). `where` is where the USER types, not where the code "
            "lives."
        )

    def test_a_where_whose_head_is_not_a_section(self, axis_registry):
        """``beam`` is a real NODE and not a section.  The section list is the
        pre-flight package's, borrowed rather than restated -- a second copy
        under ``inflight/`` is the divergence this plan exists to stop -- and
        quoting it here is what says the borrow happened."""
        register_axes("C2")(
            lambda facts: (refuse("C2", "beam", "reserved."),))
        with pytest.raises(ConfigError) as caught:
            axes(axis_facts(preflight_document()))
        assert str(caught.value) == (
            "in-flight check 'C2' emitted where='beam', whose first segment "
            "'beam' is not a document section. The sections are "
            "['schema_version', 'defaults', 'plugins', 'runtime', "
            "'observation', 'resources', 'model', 'variants', 'inference', "
            "'runs', 'outputs', 'campaign']."
        )

    def test_the_BUILT_pass_quotes_the_same_section_list(self, built_registry):
        """The sibling pin, and it is the one that was missing: measured,
        widening ``built()``'s ``sections`` with ``('beam', 'twin')`` left the
        suite at exit 0, because the built pass had only the not-a-path
        variant pinned.  A built check may say ``model.averaging`` and may not
        say ``twin`` -- the payload carries ``document`` precisely so that the
        reader is sent somewhere they can type."""
        register_built("C9")(
            lambda run: (refuse("C9", "twin", "reserved."),))
        with pytest.raises(ConfigError) as caught:
            built(built_run(preflight_document()))
        assert str(caught.value) == (
            "in-flight check 'C9' emitted where='twin', whose first segment "
            "'twin' is not a document section. The sections are "
            "['schema_version', 'defaults', 'plugins', 'runtime', "
            "'observation', 'resources', 'model', 'variants', 'inference', "
            "'runs', 'outputs', 'campaign']."
        )

    @pytest.mark.parametrize("make", [refuse, warn, report],
                             ids=["refuse", "warn", "report"])
    def test_a_bad_where_is_refused_at_every_severity(self, axis_registry,
                                                      make):
        """Kills gating the ``where`` guard on ``severity == "refuse"``.  Task
        7's B9 is warning-shaped, so this is a twin the plan walks into rather
        than a hypothetical one."""
        register_axes("C2")(lambda facts: (make("C2", "beam", "a sentence."),))
        with pytest.raises(ConfigError, match="is not a document section"):
            axes(axis_facts(preflight_document()))

    def test_every_finding_is_checked_and_not_just_the_first(
            self, axis_registry):
        """Kills ``check_where(..., found[0])`` and its twin ``found[-1]``.  A
        check that walks nodes returns one finding per node, so under either
        mutation every finding after the first can carry a source path in
        front of a user."""
        register_axes("C2")(lambda facts: (
            refuse("C2", "observation.time", "a."),
            refuse("C2", "compose.py:262", "b."),
        ))
        with pytest.raises(ConfigError, match="'C2' emitted where="):
            axes(axis_facts(preflight_document()))


class TestThePassesCollect:
    """Kills: raising on the first finding; a generator consumed twice or not
    at all; ``warnings`` reading as refusals."""

    @pytest.mark.parametrize("make", [
        lambda one: (one,),
        lambda one: [one],
        lambda one: iter((one,)),
    ], ids=["tuple", "list", "generator"])
    def test_a_check_may_hand_back_any_iterable(self, axis_registry, make):
        """Kills ``findings.extend(fn(payload))`` followed by a second pass
        over the same exhausted generator."""
        register_axes("C1")(
            lambda facts: make(refuse("C1", "observation.time", "one.")))
        assert len(axes(axis_facts(preflight_document())).refusals()) == 1

    def test_warnings_and_reports_are_collected_without_raising(
            self, built_registry):
        register_built("B9")(lambda run: (
            warn("B9", "resources.projectors.drift", "shared."),
            report("B9", "resources.projectors.drift", "noted."),
        ))
        held = built(built_run(preflight_document()))
        assert held.refusals() == ()
        assert len(held.warnings()) == 1
        assert held.checks() == frozenset({"B9"})

    def test_the_base_document_earns_no_in_flight_finding_of_its_own(self):
        """The REAL registries -- deliberately no clearing fixture.

        A base that is itself a finding makes every later task's "and nothing
        else" assertion inherit an extra id, with nothing recording it.
        Vacuous while the registries are empty and load-bearing from Task 1b
        on, which is exactly when a check registered against the wrong
        document would first be noticed.
        """
        doc = preflight_document()
        assert axis_findings(doc) == ()
        assert built_findings(doc) == ()


class TestTheOnlyHelpers:
    """``axis_only`` / ``built_only`` mirror ``preflight_helpers.only``, and the
    contract that matters is the one a refusals-only reading would break."""

    def test_a_warning_counts_as_the_one(self, axis_registry):
        """§0.3 C.4, and it is not a detail: Task 7's B9 and Task 5's A46 leg 3
        are WARNs, and a ``only`` that filtered to refusals would make every
        test about them unwritable.  Kills exactly that filter."""
        register_axes("C2")(
            lambda facts: (warn("C2", "observation.pointing", "hmm."),))
        found = axis_only(preflight_document(), "C2")
        assert found.message == "hmm."
        assert found.severity == "warn"

    def test_a_report_counts_as_the_one(self, built_registry):
        register_built("B9")(
            lambda run: (report("B9", "model.gain", "noted."),))
        assert built_only(preflight_document(), "B9").severity == "report"

    @pytest.mark.parametrize("count", [0, 2], ids=["none", "twice"])
    def test_more_or_fewer_than_one_is_an_assertion_failure(
            self, axis_registry, count):
        """Kills ``only`` becoming ``found[0]``: a check that fires TWICE on
        one document -- a loop over nodes that forgot to ``break`` -- is a real
        defect that no ``in`` assertion can see."""
        register_axes("C1")(lambda facts: tuple(
            refuse("C1", "observation.time", "x.") for _ in range(count)))
        with pytest.raises(AssertionError, match=f"produced {count} findings"):
            axis_only(preflight_document(), "C1")

    def test_each_only_reads_its_own_pass(self, axis_registry, built_registry):
        """Kills ``built_only`` running the axes pass, which would make every
        Task 7 test read an empty registry and pass by finding nothing.

        The axes probe is a WARN and not a refusal, and that is not a
        convenience: ``built_run`` goes through ``_assemble``, which runs the
        axes hook, so an axes REFUSAL stops the document before the built
        payload exists at all.  That property has a test of its own below.
        """
        register_axes("C1")(
            lambda facts: (warn("C1", "observation.time", "axes."),))
        register_built("C8")(
            lambda run: (refuse("C8", "model.averaging", "built."),))
        doc = preflight_document()
        assert axis_only(doc, "C1").message == "axes."
        assert built_only(doc, "C8").message == "built."
        assert [one.check for one in axis_findings(doc)] == ["C1"]
        assert [one.check for one in built_findings(doc)] == ["C8"]

    def test_a_document_the_axes_pass_refuses_never_reaches_the_built_pass(
            self, axis_registry, built_registry):
        """**The slot order, from the helper's side**, and it is a contract
        Task 7 has to know: ``built_run`` runs ``_assemble``, which runs the
        axes hook and raises there.  So a built-slot test may not reuse a
        document an axes check refuses -- it will never see its own check run,
        and the failure would read as the built check being absent.

        This is also the reason a built check need not restate an axes check's
        precondition: a refused axis cannot reach it.
        """
        register_axes("C1")(
            lambda facts: (refuse("C1", "observation.time", "axes first."),))
        seen = []
        register_built("C8")(lambda run: (seen.append(1), ())[1])
        with pytest.raises(ConfigError, match="axes first."):
            built_run(preflight_document())
        assert seen == []


# ---------------------------------------------------------------------------
# The two hooks in load_document
# ---------------------------------------------------------------------------


def _raise_then_warn() -> dict[str, tuple[int, int]]:
    """``{receiver: (raise_if_refused line, emit_warnings line)}`` in document.py.

    Read off the source rather than exercised, because the property is about
    ORDER between two statements and a behavioural probe can only see it on a
    document that earns both a refusal and a warning from the SAME pass -- a
    document no fixture guarantees, and one a later task could accidentally
    stop producing while the order silently inverted.
    """
    found: dict[str, dict[str, int]] = {}
    for node in ast.walk(ast.parse(_DOCUMENT_PY.read_text())):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.attr in ("raise_if_refused", "emit_warnings")):
            found.setdefault(node.func.value.id, {})[
                node.func.attr] = node.lineno
    return {name: (calls["raise_if_refused"], calls["emit_warnings"])
            for name, calls in found.items()
            if len(calls) == 2}


def _unnamed_hook_calls() -> list[tuple[str, int]]:
    """``(method, line)`` for every hook call whose receiver is NOT a name.

    The blind spot in :func:`_raise_then_warn`, closed rather than recorded.
    That harvester keys on the receiver VARIABLE, so a hook written
    ``priced(...).raise_if_refused()`` -- a receiver that is a ``Call`` -- is
    invisible to it: the pair never enters the dict and the order assertion
    never sees it.

    **Which of the two tests below would have caught it depends on whether
    the hook is an EXISTING one respelled or a NEW one**, and only the first
    was ever guarded.  Measured while Plan 3C's Task 3 added the fourth hook:
    respelling the post-flight hook that way fails
    ``test_all_four_hooks_are_present_and_named`` too, because
    ``priced_report`` then vanishes from the set -- but ADDING a fifth
    ``Report().emit_warnings()`` / ``Report().raise_if_refused()`` pair beside
    the four named ones left **only this test** red, 1 failed / 212 passed.
    That is the hole, and it is the shape a fifth pass would arrive in.

    A receiver that is a name is also what makes the ORDER meaningful at all:
    two chained calls on two separate temporaries are two different reports,
    so "raise before warn" would be a statement about nothing.
    """
    return [(node.func.attr, node.lineno)
            for node in ast.walk(ast.parse(_DOCUMENT_PY.read_text()))
            if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and not isinstance(node.func.value, ast.Name)
                and node.func.attr in ("raise_if_refused", "emit_warnings"))]


class TestEachSlotRaisesBeforeItWarns:
    """§3.2(a), as a pin rather than as a sentence."""

    def test_every_hook_raises_before_it_warns(self):
        """**The pre-flight hook's order is NOT changed by Plan 3B**, and this
        is the one-line pin that says a later refactor cannot move it in
        silence.  A document about to be refused should not also spray
        warnings about lines the user is on their way to change.

        The three later hooks are held to the same rule, so the assertion is
        over every receiver rather than over a named one.
        """
        pairs = _raise_then_warn()
        assert pairs, "no raise/warn pair found in document.py -- see below"
        for receiver, (raised, warned) in sorted(pairs.items()):
            assert raised < warned, (
                f"{receiver}.emit_warnings() runs at line {warned}, BEFORE "
                f"raise_if_refused() at {raised}. A document about to be "
                "refused must not also spray warnings."
            )

    def test_all_four_hooks_are_present_and_named(self):
        """ANTI-VACUITY: the test above passes trivially if the harvester
        stops finding pairs, or if a hook is deleted.  Name the four.

        Plan 3C's post-flight pass is the fourth, and its receiver is
        ``priced_report``.  The name is pinned rather than left free because
        this guard keys on it: a hook whose receiver were spelled anything
        else would fail here, which is the intended way to be told that a
        fifth pass has landed.
        """
        assert set(_raise_then_warn()) == {"report", "axis_report",
                                           "built_report", "priced_report"}

    def test_no_hook_is_called_on_an_unnamed_receiver(self):
        """The blind spot in the harvester, as an assertion.

        A hook whose receiver is a ``Call`` never enters the harvester's dict
        at all, so it escapes the raise-before-warn property the two tests
        above exist to protect.  Measured: a fifth pair added beside the four
        named ones leaves **only this test** red.  It is what makes the
        spelling illegal rather than merely unseen, and it is why the fourth
        hook UPDATED the guard rather than routing around it.
        """
        assert _unnamed_hook_calls() == [], (
            "a raise_if_refused()/emit_warnings() in document.py is called on "
            "a receiver that is not a plain name, so _raise_then_warn cannot "
            "pair it and the raise-before-warn order is unguarded for that "
            "hook. Bind the report to a name first."
        )

    def test_a_warning_from_an_earlier_slot_is_already_out_when_a_later_slot_refuses(
            self, axis_registry, built_registry):
        """§3.2(a)'s other half, **correct rather than tolerated**.

        Across slots the raise-before-warn rule cannot hold: the axes pass has
        already returned by the time ``build_resources`` runs, let alone the
        built pass.  Holding every finding to the end would reinstate the
        round-trip the collect-rather-than-raise design exists to remove, and
        an axes warning is about a line the built refusal does not touch.  No
        later task re-litigates this; it is pinned here.
        """
        register_axes("C2")(
            lambda facts: (warn("C2", "observation.pointing", "early."),))
        register_built("C8")(
            lambda run: (refuse("C8", "model.averaging", "late."),))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ConfigError, match="late."):
                load_document(preflight_document())
        assert [str(one.message) for one in caught
                if issubclass(one.category, ConfigWarning)] == ["early."]


class TestTheHooksArePositioned:
    """Kills: the axes hook sliding below ``build_resources``, which undoes the
    only saving this plan makes; the built hook never being called at all."""

    def test_the_axes_hook_runs_before_build_resources(self, axis_registry,
                                                       monkeypatch):
        """**The one test no later task repeats.**  Tasks 1b and 7 each test
        that their check FIRES; none tests that it fires before the beam, and
        the hook is one call in ``load_document``.  If this is weak, moving
        that call below ``build_resources`` is a green edit that undoes the
        plan."""
        order = []
        real = document_module.build_resources
        monkeypatch.setattr(document_module, "build_resources",
                            lambda *a, **k: (order.append("resources"),
                                             real(*a, **k))[1])
        register_axes("C1")(
            lambda facts: (order.append("axes"), ())[1])
        load_document(preflight_document())
        assert order == ["axes", "resources"]

    def test_the_axes_hook_refuses_before_the_beam_is_read(self, axis_registry):
        """§5's phase property, in its synthetic form.  A document carrying
        ``UNREADABLE_BEAM`` **and** an axes violation refuses with the
        violation, and ``no_such_beam`` does not appear.  Before this plan the
        beam won, five for five.

        SYNTHETIC deliberately -- it registers a probe, so it proves the
        HOOK's position and no real check's phase.  The real assertion, on a
        real check, is Task 1b's.
        """
        register_axes("C1")(lambda facts: (
            refuse("C1", "observation.time", "the time axis is wrong."),))
        with pytest.raises(ConfigError) as caught:
            load_document(preflight_document(resources=UNREADABLE_BEAM))
        assert str(caught.value) == "the time axis is wrong."
        assert "no_such_beam" not in str(caught.value)

    def test_both_payloads_carry_the_VARIANT_APPLIED_document(
            self, axis_registry, built_registry):
        """§0.3 C.3 pins ``Axes.document`` / ``Built.document`` as the
        variant-applied mapping BY NAME, and measured, passing the raw
        ``document`` to either hook left the suite at exit 0.

        This is the first thing that matters about it: every check Task 1b
        registers reads ``facts.document``, so under the raw mapping a
        selected variant's ``resources:``, ``model:`` or ``inference:`` would
        be decided against the base's text -- a check firing on a line the
        user is not running, or silent on the line they are, in either case
        with nothing saying so.
        """
        seen = {}
        register_axes("C1")(
            lambda facts: (seen.__setitem__("axes", facts.document), ())[1])
        register_built("C9")(
            lambda run: (seen.__setitem__("built", run.document), ())[1])
        base = preflight_document()
        assert base["model"]["gain"]["gain"]["value"] != 1.0
        load_document(base, variant="unity_gain")
        for slot in ("axes", "built"):
            assert seen[slot]["model"]["gain"]["gain"]["value"] == 1.0, slot
        assert base["model"]["gain"]["gain"]["value"] != 1.0

    def test_the_built_hook_runs_and_the_beam_wins_against_it(
            self, built_registry):
        """**The ANTI-property, stated rather than implied.**  The built slot
        runs after the money, so a document carrying ``UNREADABLE_BEAM`` and a
        built violation refuses with **the beam**.  A plan claiming otherwise
        would be claiming schema §6's false preamble, and this is the test
        that stops it being claimed by accident."""
        register_built("C8")(
            lambda run: (refuse("C8", "model.averaging", "never reached."),))
        with pytest.raises(ConfigError) as caught:
            load_document(preflight_document(resources=UNREADABLE_BEAM))
        assert "no_such_beam" in str(caught.value)
        assert "never reached." not in str(caught.value)

    def test_the_built_hook_stops_the_load(self, built_registry):
        """The seam ``_assemble`` exists for, from the hook's side."""
        register_built("C8")(
            lambda run: (refuse("C8", "model.averaging", "refused."),))
        with pytest.raises(ConfigError, match="refused."):
            load_document(preflight_document())

    def test_assemble_does_not_run_the_built_pass(self, built_registry):
        """**And from the helper's side, which is the whole reason for the
        split.**  ``Built(*load_document(d))`` cannot observe a built-slot
        REFUSAL, because the hook raises before ``load_document`` returns --
        so a helper built on it could only ever exercise the passing half of
        the slot it exists to test.  Kills ``built_run`` being rewritten to
        call ``load_document``, which is the obvious simplification and which
        would disarm every Task 7 refusal test at once."""
        register_built("C8")(
            lambda run: (refuse("C8", "model.averaging", "refused."),))
        payload = built_run(preflight_document())      # returns
        assert isinstance(payload, Built)
        assert built(payload).refusals()[0].message == "refused."

    def test_the_built_hook_sees_the_assembled_objects(self, built_registry):
        """The payload is built AFTER ``build_inference``, so one payload
        carries the raw twin, the fit twin and the space.  Kills a hook placed
        before it, under which ``.inference`` would be absent."""
        seen = {}

        @register_built("C8")
        def _look(run):
            seen["twin"] = run.twin is not None
            seen["fit"] = run.inference.fit_twin is not None
            seen["resources"] = run.resources is not None
            return ()

        load_document(preflight_document())
        assert seen == {"twin": True, "fit": True, "resources": True}


# ---------------------------------------------------------------------------
# resolved_specs
# ---------------------------------------------------------------------------

#: A beam entry that is well formed, for the cross-kind case.
_BEAM = {"format": "npy", "path": "beam.npy", "nside": 4,
         "normalize": "pixel_sum", "frame": "beam_local"}

#: The six shapes ``_resolved_spec`` raises by name on, each with the literal
#: ``build_resources`` still refuses it with.  Measured at ``e0e024a``.
_MALFORMED = {
    "an-extends-cycle": (
        {"arrays": {"a": {"extends": "b"}, "b": {"extends": "a"}}},
        "resources: these entries extend each other in a loop: "
        "resources.arrays.a -> resources.arrays.b -> resources.arrays.a."),
    "a-self-extend": (
        {"arrays": {"a": {"extends": "a"}}},
        "resources: these entries extend each other in a loop: "
        "resources.arrays.a -> resources.arrays.a."),
    "a-dangling-parent": (
        {"arrays": {"a": {"extends": "nope"}}},
        "resources.arrays.a extends 'nope', which resources.arrays does not "
        "declare."),
    "a-cross-kind-parent": (
        {"arrays": {"a": {"extends": "horn"}}, "beams": {"horn": _BEAM}},
        "resources.arrays.a extends 'horn', which resources.arrays does not "
        "declare. It is declared as resources.beams.horn, and extends: merges "
        "between siblings of the SAME kind only"),
    "a-non-string-extends": (
        {"arrays": {"a": {"extends": 5}}},
        "resources.arrays.a extends 5, which resources.arrays does not "
        "declare."),
    "append-beside-a-sibling-key": (
        {"arrays": {"base": {"value": [1.0, 2.0]},
                    "a": {"extends": "base",
                          "value": {"append": [3.0], "nope": 1}}}},
        "'value': append must be the only key when extending a list; got the "
        "sibling keys ['nope']."),
}


class TestResolvedSpecsIsTotal:
    """**It never raises**, and that is the whole contract.

    A pre-flight check that let one of these escape would be wrapped by the
    pass as *"pre-flight check 'A11' RAISED ConfigError: ..."* -- which
    **aborts the pass and hides every finding after it**, while every existing
    ``match=`` pin still passes, because ``match=`` searches.  A green suite
    and a stack-trace-shaped user message is the worst of the two failures
    available here.
    """

    @pytest.mark.parametrize("case", sorted(_MALFORMED), ids=sorted(_MALFORMED))
    def test_the_malformed_entry_is_dropped_and_nothing_is_raised(self, case):
        section, _ = _MALFORMED[case]
        got = resolved_specs(section)              # must not raise
        assert "resources.arrays.a" not in got, (
            "the malformed entry survived, so a check reading it would decide "
            "on a spec `extends:` never resolved"
        )

    @pytest.mark.parametrize("case", sorted(_MALFORMED), ids=sorted(_MALFORMED))
    def test_build_resources_still_refuses_it_with_its_own_literal(self, case):
        """The other half, and the reason dropping is safe **for these six**:
        ``build_resources`` stays the backstop, saying the right sentence at
        the right phase.  A check that finds an entry missing STANDS DOWN on
        it -- refusing on "I could not tell" refuses documents that build.

        The claim is exactly this list and no wider; see
        :meth:`test_a_shape_the_BUILDER_cannot_name_either_is_recorded_not_claimed`
        for a shape where the backstop is a bare ``TypeError``."""
        section, literal = _MALFORMED[case]
        with pytest.raises(ConfigError) as caught:
            load_document(preflight_document(resources=section))
        assert literal in str(caught.value)

    def test_a_shape_that_is_not_a_ConfigError_at_all_is_still_survived(self):
        """The reason the catch is ``except Exception`` and not ``except
        ConfigError``, as a MEASURED shape rather than a hypothetical one.

        An ``extends:`` chain longer than the interpreter's recursion limit
        raises ``RecursionError`` out of ``_resolved_spec`` -- a
        ``RuntimeError``, not this layer's refusal type.  Under a narrower
        catch that escapes into the pass, which wraps it as *"check 'A11'
        RAISED RecursionError"* and hides every other finding on the document.

        1100 and not 4000: the work is quadratic in the chain length (every
        entry above the limit retries the whole descent), and measured, 4000
        costs 11.7 s against 0.44 s here for the same verdict.
        """
        depth = 1100
        arrays = {f"a{index}": {"extends": f"a{index + 1}"}
                  for index in range(depth)}
        arrays[f"a{depth}"] = {"value": [1.0]}
        got = resolved_specs({"arrays": arrays})       # must not raise
        assert isinstance(got, dict)
        assert got[f"resources.arrays.a{depth}"] == {"value": [1.0]}
        assert len(got) < len(arrays), (
            "nothing was dropped, so this document no longer reaches the "
            "recursion limit and the case has stopped testing its subject"
        )

    def test_a_shape_the_BUILDER_cannot_name_either_is_recorded_not_claimed(self):
        """The docstring's backstop claim, held to what is true.

        A spec whose KEY is not a string resolves here untouched -- there is
        no ``extends:`` to fail on -- and then dies inside ``build_resources``
        as a bare ``TypeError``, not a ``ConfigError``.  So "build_resources
        says the right sentence" is true of the six modelled shapes and NOT of
        every shape, and what this function guarantees is the narrower thing:
        the pass is not aborted.
        """
        section = {"arrays": {"a": {1: "x"}}}
        assert resolved_specs(section) == {"resources.arrays.a": {1: "x"}}
        with pytest.raises(TypeError):
            load_document(preflight_document(resources=section))

    def test_a_well_formed_sibling_survives_a_malformed_entry(self):
        """Only the malformed ENTRY is dropped, not the whole kind and not the
        whole section.  Kills a ``try`` around the whole loop, under which one
        bad entry blinds every check about every other one."""
        section, _ = _MALFORMED["append-beside-a-sibling-key"]
        got = resolved_specs(section)
        assert got["resources.arrays.base"] == {"value": [1.0, 2.0]}
        assert "resources.arrays.a" not in got
        cross, _ = _MALFORMED["a-cross-kind-parent"]
        assert "resources.beams.horn" in resolved_specs(cross)

    @pytest.mark.parametrize("section", [
        None, {}, [], "resources", 7,
        {"arrays": "not a mapping"},
        {"arrays": {"a": "not a mapping"}},
        {"arrays": {"a": None}},
    ], ids=["none", "empty", "list", "string", "int", "kind-not-a-mapping",
            "entry-not-a-mapping", "entry-none"])
    def test_a_shape_build_resources_refuses_is_answered_with_a_mapping(
            self, section):
        """TOTAL means total.  These are shapes ``build_resources`` refuses by
        name, and a reader that has not built yet must not pre-empt that
        sentence -- so the answer is "I have nothing for you", not an
        exception and not a refusal."""
        assert resolved_specs(section) == {} or isinstance(
            resolved_specs(section), dict)
        assert "resources.arrays.a" not in resolved_specs(section)

    def test_the_key_is_the_dotted_string_and_extends_is_applied(self):
        """The key is exactly ``build_resources``' own and exactly
        ``BuiltResources.resources``' -- never the bare name, never a tuple --
        so a caller selects a kind with ``startswith``.  And the spec is the
        one AFTER ``extends:``, which is the measured TRAP this function
        exists to make unnecessary: a check reading the raw text refuses an
        entry whose ``normalize:`` came from its parent."""
        got = resolved_specs({"beams": {
            "parent": _BEAM,
            "child": {"extends": "parent", "nside": 8},
        }})
        assert set(got) == {"resources.beams.parent", "resources.beams.child"}
        child = got["resources.beams.child"]
        assert child["normalize"] == "pixel_sum"   # inherited
        assert child["nside"] == 8                 # overridden
        assert "extends" not in child
        assert [k for k in got if k.startswith("resources.beams.")]

    def test_it_hands_back_a_copy_rather_than_the_users_own_mapping(self):
        """A reader running inside a pass must not be able to edit the
        document it is reporting on."""
        section = {"beams": {"horn": dict(_BEAM)}}
        got = resolved_specs(section)
        got["resources.beams.horn"]["nside"] = 999
        assert section["beams"]["horn"]["nside"] == 4


# ---------------------------------------------------------------------------
# The one-binding walker
# ---------------------------------------------------------------------------


class TestTheMessageBindingWalker:
    """§3.2(h) as a deliverable rather than a review step: for every hoisted
    row, the refusal's literal appears in exactly ONE module under ``src/``.

    Task 1a ships the walker and its anti-vacuity; Tasks 2-6 each parametrize
    a test in THEIR OWN module over THEIR OWN literals.  There is deliberately
    no shared table: it is the one resolution whose failure mode is a
    *passing* test -- two tasks land in parallel, the merge keeps one side,
    and the rows that vanished are exactly the rows nobody is checking.
    """

    def test_a_message_bound_once_passes(self):
        """A sentence this layer is known to hold in exactly one module."""
        assert_bound_once(
            "campaign: is reserved with capability 4 (streaming evidence, "
            "schema §8.2) and refused in v1.")

    def test_a_message_bound_twice_fails(self):
        """ANTI-VACUITY, the direction that matters: the walker must be able
        to say NO.  A walker that reported "1" for everything would pass every
        one-binding test ever written and catch nothing.

        The literal is the measured twin asked under a SHORTER spelling, so
        the exemption -- which is keyed on the whole flattened literal -- does
        not apply to it.  That is deliberate on both counts: it proves the
        walker can say no, AND it proves the exemption is a narrow key rather
        than a blanket pardon for anything resembling it.

        **The count is READ, not written down.**  This literal is exactly the
        one whichever task hoists C1 is deciding, so a hardcoded ``== 2``
        would go red on a correct hoist and the failure would read as the
        walker being wrong.  What is being tested is that the walker reports
        the number it finds and refuses anything above one.
        """
        found = len(modules_carrying("time is stored as"))
        assert found >= 2, (
            "the walker's anti-vacuity case needs a literal that really is "
            "bound more than once; this one no longer is."
        )
        with pytest.raises(AssertionError, match=rf"bound {found} times"):
            assert_bound_once("time is stored as")

    def test_a_message_bound_nowhere_fails(self):
        """The other direction, and it is the failure a drifting test literal
        actually produces: the sentence in the test no longer matches any
        source, so the test is checking nothing while looking rigorous."""
        with pytest.raises(AssertionError, match="bound 0 times"):
            assert_bound_once(
                "no sentence in this package reads anything like this one")

    def test_the_walker_folds_a_message_split_across_lines(self):
        """The reason this is ``ast`` and not ``grep``.  A message in ``src/``
        is a dozen implicitly-concatenated pieces across a dozen lines, so a
        raw substring search for a whole sentence finds NOTHING -- and a
        walker that found nothing for everything would fail every
        ``== 1`` test for the wrong reason and pass every ``>= 1`` one.
        """
        literal = ("A check returns findings and raises nothing -- one that "
                   "raises aborts the pass and hides every finding after it")
        assert modules_carrying(literal) == ("config/passes.py",)

    def test_the_exemption_proves_itself(self):
        """A forgiveness list nobody re-measures is a list of duplications
        nobody is checking -- which is the state this walker exists to end,
        one indirection along.

        This fails the day the pair becomes ONE module (delete the row) or
        grows a third (decide it), which is exactly when the exemption stops
        being true.  ``coords.time is stored as`` is the plan's own named twin
        for C1: two time-axis guards that differ in what they ask, and
        collapsing them belongs to whichever task hoists that rule.
        """
        for literal, (found, allowed) in exempt_pairs_still_hold().items():
            assert set(found) == set(allowed), (
                f"the exemption for {literal!r} claims {sorted(allowed)} and "
                f"the tree now has {sorted(found)}. Re-decide the row rather "
                "than widening it."
            )

    def test_the_extraction_left_one_binding_behind_it(self):
        """Task 1a's own row, and the property it is the first test of: the
        runner MOVED.  Every sentence ``passes.py`` says is said there and
        nowhere else -- a copy left in ``preflight/__init__.py`` would be two
        runners for one property, which is the whole reason for the
        extraction."""
        for literal in (
            "A check returns findings and raises nothing",
            "`where` is where the USER types, not where the code lives.",
            "A check id has one function, and which of the two would run "
            "depends on import order.",
        ):
            assert modules_carrying(literal) == ("config/passes.py",), literal


# ---------------------------------------------------------------------------
# Wiring -- the import block that makes a decorated check a registered one
# ---------------------------------------------------------------------------

#: The package prefix :func:`_foot_imports` is asked about here.  ``preflight``
#: is that function's default; this package has to say its own name.
_INFLIGHT_PACKAGE = "rheplicant.config.inflight"


def _modules_under(directory: pathlib.Path) -> set[str]:
    """Every check module living in ``directory``, by stem.

    A GLOB, never a maintained list, and that is the whole load-bearing
    property of the guard below: a module added under ``inflight/`` after this
    test was written is in this set the moment the file exists, with nobody
    having remembered anything.
    ``test_a_module_added_after_this_test_was_written_is_seen`` drives it on a
    directory this file creates, so the claim is measured rather than asserted
    about the four modules that happen to be here today.
    """
    return {path.stem for path in directory.glob("*.py")
            if path.stem != "__init__"}


def _unwired(directory: pathlib.Path, source: str) -> set[str]:
    """Modules in ``directory`` that ``source``'s import block does not name.

    Both halves of the comparison are derived -- one by glob, one by ``ast``
    -- so neither side is a list that can go stale.  Empty is the only
    acceptable answer for the real package.
    """
    return _modules_under(directory) - _foot_imports(source, _INFLIGHT_PACKAGE)


#: Run in a CHILD process: which modules under ``inflight/`` own a live slot
#: when the package is imported and nothing else has been.
#:
#: **In-process this question cannot be asked at all**, and that is the defect
#: this probe exists to close.  ``test_inflight_grids.py`` imports
#: ``inflight.grids`` to reach its check functions; that import runs the
#: ``@register_axes`` decorators; so by the time any test in a full session
#: reads ``AXIS_CHECKS`` the slots are there whether or not
#: ``inflight/__init__.py`` ever imported the module.  Measured on this branch
#: before the fix: deleting ``grids``' line left the **whole** suite at exit 0
#: with C3, A13.grid and C8 registered nowhere for a user.
#:
#: ``sys.argv[1:]`` is folded into ``present`` so the anti-vacuity case can
#: give the child a module it cannot possibly find a slot for and watch it
#: die -- which is what says the ``assert``s in here are live at all (a child
#: run under ``-O``, or one whose import quietly failed upward, would exit 0
#: with no assertion having been evaluated).
_WIRING_PROBE = """
import importlib
import pathlib
import sys

from rheplicant.config.inflight import AXIS_CHECKS, BUILT_CHECKS

here = pathlib.Path(importlib.import_module(
    "rheplicant.config.inflight").__file__).resolve().parent
present = {p.stem for p in here.glob("*.py") if p.stem != "__init__"}
present |= set(sys.argv[1:])
contributing = {fn.__module__.rsplit(".", 1)[-1]
                for fn in (*AXIS_CHECKS.values(), *BUILT_CHECKS.values())}
assert not present - contributing, (
    "live under inflight/ and own no slot in a fresh process: "
    + repr(sorted(present - contributing)))
assert not contributing - present, (
    "own a slot and are not modules under inflight/: "
    + repr(sorted(contributing - present)))
"""


class TestTheImportBlockCannotRot:
    """A module under ``inflight/`` that the package does not import registers
    nothing, refuses nothing, and stays green.

    **This is the plan's own headline method finding, and it shipped inside
    the plan.**  ``inflight/__init__.py``'s import block is what turns a
    decorated function into a registered check; delete one line of it and the
    module's checks are absent for every user, while every test in that
    module goes on passing, because the test module's own import ran the
    decorators.  Measured on this branch: with ``grids``' line removed,
    ``pytest -n 16`` over the WHOLE suite exited 0 and ``AXIS_CHECKS`` was
    ``['C1', 'C2.time', 'C2.pointing']`` -- C3, A13.grid and C8 registered
    nowhere in production, 7264 tests silent.

    **Structural, not per-module, and that distinction is the fix.**  Two of
    the four modules already carried a hand-written
    ``test_the_module_is_WIRED_and_not_merely_decorated``; the two that did
    not are exactly the two that rotted, and a guard somebody has to remember
    to copy is the thing that failed.  Both tests here derive BOTH sides of
    their comparison -- one side by glob, one by ``ast`` or by
    ``fn.__module__`` -- so a fifth module is covered the moment its file
    exists.

    ``preflight/`` has had this shape since 3A
    (``test_config_preflight.py::TestTheFootImportCannotRot``); this is the
    counterpart it was missing, and it shares that guard's matcher rather
    than restating it.
    """

    def test_every_module_under_inflight_is_named_in_the_import_block(self):
        """PRESENCE IN THE PACKAGE, read off the ``__init__``'s own TEXT.

        Text rather than the live registries on purpose: this is the half no
        sibling's import can fake.  A test that asked the imported package
        would already have the decorators run and would answer "wired" for a
        module the package never imports -- which is precisely how the guard
        that did notice
        (``test_config_surface.py::test_every_check_plan_3b_claims_is_registered``)
        was disarmed in a full session and went red only when run alone.
        """
        unwired = _unwired(
            _INFLIGHT_DIR, (_INFLIGHT_DIR / "__init__.py").read_text())
        assert unwired == set(), (
            f"{sorted(unwired)} live under inflight/ and are imported by "
            "nothing, so their @register_axes / @register_built decorators "
            "never run and their checks are silently absent for every user. "
            "Add an alphabetically-placed line to the import block in "
            "inflight/__init__.py -- ABOVE `def axes`, for the reason that "
            "block's own comment gives."
        )

    def test_the_block_names_no_module_that_is_gone(self):
        """The other direction, which no deletion sweep reaches: a line left
        behind for a module somebody removed.  It would raise ``ImportError``
        at package import -- loudly -- so this is a guard against the *typo*,
        not against silence."""
        declared = _foot_imports(
            (_INFLIGHT_DIR / "__init__.py").read_text(), _INFLIGHT_PACKAGE)
        present = _modules_under(_INFLIGHT_DIR)
        assert declared <= present, (
            f"{sorted(declared - present)} are imported by "
            "inflight/__init__.py and do not exist."
        )

    def test_every_module_under_inflight_contributes_a_slot(self):
        """PRESENCE IS NOT CONTRIBUTION, and it is asked IN A SUBPROCESS.

        The test above reads text, so it cannot tell an import that runs from
        an import that runs and registers nothing -- a module named in the
        block whose ``@register_axes`` calls were commented out passes it.
        ``fn.__module__`` over the live registries answers that, but only in a
        process where nothing else has imported the modules, which is never
        true inside this suite.  Hence the child.

        Both sides are derived there too: the child globs ``inflight/*.py``
        itself rather than carrying a list of the four names, so this test
        needs no edit when a fifth module lands and no edit is what stops it
        rotting.

        A later plan that genuinely wants a helper module under ``inflight/``
        with no check of its own must say so by editing this test -- which is
        the point; the alternative is a silent hole shaped exactly like the
        one that shipped.
        """
        done = subprocess.run(
            [sys.executable, "-c", _WIRING_PROBE],
            capture_output=True, text=True, cwd=str(_ROOT), check=False)
        assert done.returncode == 0, (
            "importing rheplicant.config.inflight does not leave every module "
            "under inflight/ owning a slot. In THIS process the same question "
            "answers 'fine', because the test modules import them "
            "directly.\n" + done.stdout + done.stderr)

    def test_the_subprocess_probe_can_fail(self):
        """ANTI-VACUITY for the child, and it is not ceremony.

        A subprocess guard has two silent failure modes an in-process one does
        not: the child exiting 0 without evaluating anything (``-O`` strips
        ``assert``), and the comparison being trivially true because both
        sides came out empty (an import that failed upward, a glob that
        matched nothing). Handing the child a module name that owns no slot
        must kill it; if this passes, the test above is decoration.
        """
        done = subprocess.run(
            [sys.executable, "-c", _WIRING_PROBE, "no_such_inflight_module"],
            capture_output=True, text=True, cwd=str(_ROOT), check=False)
        assert done.returncode != 0, (
            "the child accepted a module that owns no slot, so its assertions "
            "are not running and the wiring test above proves nothing.\n"
            + done.stdout + done.stderr)
        assert "no_such_inflight_module" in done.stderr, done.stderr

    def test_a_module_added_after_this_test_was_written_is_seen(self, tmp_path):
        """The anti-vacuity that matters for the STATIC half.

        ``_unwired`` returning ``set()`` for the real package is exactly what
        a matcher that found nothing at all would also return, and that is the
        shape 2C shipped -- a discovery-by-prefix guard that matched nothing
        and passed forever.  So the comparison is driven here on a package
        this test builds, with a module the block does not name, and the
        newcomer must be reported.

        Written into ``tmp_path``, never into ``src/`` (R11): a probe module
        created and deleted inside a globbed package went red 1 run in 8 under
        ``-n 16`` and would have flaked every branch at once.
        """
        package = tmp_path / "inflight"
        package.mkdir()
        (package / "__init__.py").write_text(
            "from rheplicant.config.inflight import axes as _axis_checks\n"
            "from rheplicant.config.inflight import grids as _grid_checks\n")
        for stem in ("__init__", "axes", "grids", "newcomer"):
            (package / f"{stem}.py").touch()

        assert _modules_under(package) == {"axes", "grids", "newcomer"}
        assert _unwired(
            package, (package / "__init__.py").read_text()) == {"newcomer"}

    @pytest.mark.parametrize(("source", "expected"), [
        ("from rheplicant.config.inflight import grids as _grid_checks",
         {"grids"}),
        ("from rheplicant.config.inflight import axes as _axis_checks",
         {"axes"}),
        ("from rheplicant.config.inflight import grids, twin",
         {"grids", "twin"}),
        ("from . import grids", {"grids"}),
        ("import rheplicant.config.inflight.grids", {"grids"}),
        ("# from rheplicant.config.inflight import grids as _grid_checks",
         set()),
        ('"""from rheplicant.config.inflight import grids."""', set()),
        ("from rheplicant.config.preflight import document as _d", set()),
        ("from rheplicant.config.passes import binder, sweep", set()),
        ("def axes(facts):\n"
         "    from rheplicant.config.inflight import grids as _g\n", set()),
        ("if True:\n"
         "    from rheplicant.config.inflight import grids\n", set()),
    ], ids=["the-shipped-alias", "the-colliding-name", "several", "relative",
            "import-form", "commented", "in-a-docstring", "the-other-package",
            "the-runner", "in-a-function", "in-a-branch"])
    def test_the_matcher_reads_this_package_and_not_its_neighbours(
            self, source, expected):
        """The matcher is ``test_config_preflight.py``'s, called with this
        package's name -- so what needs checking here is the spellings only
        THIS package produces.

        ``the-colliding-name`` is the one that is specific to ``inflight/``:
        ``axes`` is both a module here and the package's entry point (R13),
        the block's own comment records that two of the three spellings for it
        are silently wrong, and a matcher that resolved the BOUND name rather
        than ``alias.name`` would report ``_axis_checks`` and call the module
        unimported.  ``the-other-package`` and ``the-runner`` are what say the
        prefix argument is doing work: ``preflight/``'s modules and
        ``passes.py`` are head-imported by this ``__init__`` and must not
        count as ``inflight/`` modules -- with the default prefix, the
        ``preflight`` line alone would put ``document`` in the answer and the
        second direction above would go red.
        """
        assert _foot_imports(source, _INFLIGHT_PACKAGE) == expected


# ---------------------------------------------------------------------------
# Scope
# ---------------------------------------------------------------------------

#: Verbs that reach the filesystem.  An in-flight module MAY hold a built
#: object -- that is the point of the slot -- and may NOT go back to disk.
_FILESYSTEM = frozenset({
    "open", "open_code", "read_text", "read_bytes", "iterdir", "exists",
    "listdir", "fromfile", "load", "loadtxt", "connect", "lstat", "stat",
    "getcwd", "rglob", "glob", "walk", "scandir", "mkdir", "unlink",
})

#: Autodiff and decomposition entry points.  These are **Plan 3C's**, without
#: exception; ``jax.eval_shape`` is deliberately absent, because it is in
#: scope and banning it would forbid the one shape-only probe this slot is
#: allowed.
_DIFFERENTIATION = frozenset({
    "jacfwd", "jacrev", "hessian", "jvp", "vjp", "linearize", "grad",
    "value_and_grad", "jacobian", "svd", "eigh", "eig", "lstsq",
})

_OUT_OF_BOUNDS = _FILESYSTEM | _DIFFERENTIATION


def _out_of_bounds_calls(source: str) -> set[str]:
    """Every out-of-bounds call ``source`` writes, **reached or not**.

    ``f()`` and ``x.f()`` alike, by the name at the call site, so it does not
    matter which module the name came from or whether the branch holding it
    ever runs.  That branch-independence is the point: a runtime probe drives
    the pass on documents, so a file read on a branch no document takes is
    invisible to it.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = (func.attr if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None)
        if name in _OUT_OF_BOUNDS:
            found.add(name)
    return found


def _inflight_sources(root: pathlib.Path = _INFLIGHT_DIR) -> list[pathlib.Path]:
    """Every module the two in-flight passes run out of.

    **A GLOB over both paths, never a module list.**  A list is shortened by
    deleting a line, and a module dropped from it is invisible -- which is
    exactly how the pre-flight guards stopped covering ``sweep`` when it moved
    into ``passes.py``.  ``passes.py`` is in both walks because it is the
    runner for all three passes.

    ``root`` is INJECTABLE, and that is a defect fix rather than a
    generalisation.  The discovery anti-vacuity case below used to write a
    probe module into ``src/rheplicant/config/inflight/`` and unlink it;
    measured under ``-n 16``, and even on this module alone, that raced every
    other walk of the same directory -- **1 red in 8 runs**, a
    ``FileNotFoundError`` from a sibling test globbing the directory between
    the write and the unlink, and widened it also took down
    ``tests/config/message_binding.py``, which every branch imports.  Nothing
    in this repository writes into ``src/`` from a test; the probe now writes
    into ``tmp_path`` and the walk is pointed at it, which kills the
    maintained-list mutant exactly as before because a hardcoded list ignores
    the argument.
    """
    return sorted(root.glob("*.py")) + [_PASSES_PY]


class TestTheInFlightBoundary:
    """``inflight/`` MAY hold a built object; it may NOT read a file, evaluate
    the twin, or take a Jacobian.

    **What the chosen method cannot see**, written down because a verification
    method with the same blind spot as the code is this project's recorded
    failure mode:

    * **Calling the twin.**  ``run.twin(run.state)`` is a plain call on an
      attribute and has no distinctive NAME, so no static ban can spell it.
      Neither can a runtime probe with the registries empty.  It is recorded
      here as an uncovered route, and the plan's own stop-and-ask list is what
      stands in front of it: a task that reaches for a forward pass reports
      rather than writes one.
    * **Indirection.**  A module here that calls a helper in ``sections/``
      which itself opens a file passes this walk.  The ban is by the name at
      THIS call site.
    * **Vacuity while the registries are empty.**  The walk reads the two
      modules that exist today; it becomes load-bearing as Tasks 1b and 7 add
      theirs, and because it is a glob it covers them on the day they land
      rather than on the day someone remembers.
    * ``jax.eval_shape`` is deliberately NOT banned -- it is in scope, and a
      ban would forbid the one shape-only probe this slot is allowed.
    """

    def test_no_module_here_reads_a_file_or_takes_a_jacobian(self):
        offenders = {
            path.name: sorted(found)
            for path in _inflight_sources()
            if (found := _out_of_bounds_calls(path.read_text()))
        }
        assert offenders == {}, (
            f"{offenders} are called under inflight/. This slot may hold a "
            "BUILT object and may not go back to disk, evaluate the twin or "
            "take a Jacobian -- a jacfwd here is Plan 3C's work done in Plan "
            "3B's slot. jax.eval_shape is permitted and is deliberately not "
            "on the list."
        )

    def test_the_matcher_sees_an_offence_in_a_module_it_has_never_seen(
            self, tmp_path):
        """ANTI-VACUITY for the MATCHER over a discovered file, as opposed to
        over the inline snippets below."""
        (tmp_path / "later_task.py").write_text(
            "def _check(run):\n"
            "    if run.document.get('never'):\n"
            "        return open('/etc/passwd').read()\n"
            "    return ()\n"
        )
        discovered = sorted(tmp_path.glob("*.py"))
        assert [p.name for p in discovered] == ["later_task.py"]
        assert _out_of_bounds_calls(discovered[0].read_text()) == {"open"}

    def test_the_walk_itself_discovers_a_module_added_after_it_was_written(
            self, tmp_path):
        """ANTI-VACUITY for the **DISCOVERY**, and it is the one a mutation
        campaign proved was missing.

        ``inflight/`` held exactly ONE module when this was written, so every
        subset-shaped assertion about the walk was satisfied by the maintained
        list ``[_INFLIGHT_DIR / "__init__.py", _PASSES_PY]`` -- measured, that
        substitution **survived every other test in this file**.  The property
        only bites once a second module lands, which is precisely when nobody
        is looking.

        So this writes a real module and asserts the walk picks it up.  **It
        writes into ``tmp_path`` and points the walk there**, rather than into
        ``src/``: the first version created and unlinked a file inside the
        package, which raced every other glob of that directory -- 1 red in 8
        runs of this module ALONE, before any ``-n 16``.  A hardcoded list
        ignores ``root`` entirely, so it fails this exactly as it did before.
        """
        probe = tmp_path / "_probe_discovered_by_the_walk.py"
        probe.write_text('"""Written by a test to prove the walk is a '
                         'discovery."""\n')
        assert probe in _inflight_sources(root=tmp_path), (
            "a module added beside the in-flight package is not in the walk, "
            "so the boundary guard is a maintained list and a later task's "
            "modules will not be covered on the day they land."
        )
        probe.unlink()
        assert probe not in _inflight_sources(root=tmp_path)

    def test_both_paths_are_in_the_walk_and_it_is_still_a_glob(self):
        """ANTI-VACUITY for the walk's REACH, and it is a **measured** gap.

        The first assertion kills the glob being narrowed to ``inflight/``
        alone, which would stop covering the runner every check goes through.

        The second is the one a mutation campaign found missing: replacing the
        glob with the maintained list ``[_INFLIGHT_DIR / "__init__.py",
        _PASSES_PY]`` **survived every other test in this file**, because both
        paths really are in that list.  A list is shortened by deleting a
        line, and the module dropped from it is invisible -- which is exactly
        how the pre-flight guards stopped covering ``sweep`` when it moved.
        So the walk must CONTAIN the directory listing, not merely agree with
        it today.

        The two listings below are taken independently, which is only safe
        because **no test writes into ``src/`` any more** -- see
        :func:`_inflight_sources`.  While one did, this was a second race.
        """
        walked = _inflight_sources()
        assert _PASSES_PY in walked
        assert set(_INFLIGHT_DIR.glob("*.py")) <= set(walked), (
            "a module under inflight/ is missing from the walk, so the "
            "boundary guard has become a maintained list rather than a "
            "discovery and a module added by a later task is not covered."
        )
        assert (_INFLIGHT_DIR / "__init__.py") in walked

    @pytest.mark.parametrize(("source", "expected"), [
        ("def f(p):\n    return p.read_bytes()", {"read_bytes"}),
        ("def f(p):\n    return open(p).read()", {"open"}),
        ("def f(p):\n    return np.load(p)", {"load"}),
        ("def f(t):\n    return jax.jacfwd(t)(x)", {"jacfwd"}),
        ("def f(t):\n    return jacrev(t)", {"jacrev"}),
        ("def f(m):\n    return jnp.linalg.svd(m)", {"svd"}),
        ("def f(d):\n    if False:\n        return open(d)\n    return ()",
         {"open"}),
        ("def f(t):\n    return jax.eval_shape(t, s)", set()),
        ("# open(path)", set()),
        ('"""p.read_bytes() is never called here."""', set()),
        ("def f(r):\n    return r.context.shape_scope", set()),
        ("def f(r):\n    return r.twin(r.state)", set()),
    ], ids=["read-bytes", "open", "np-load", "jacfwd", "jacrev", "svd",
            "on-a-branch-never-taken", "eval-shape-PERMITTED", "commented",
            "in-a-docstring", "the-sanctioned-shape-reader",
            "calling-the-twin-NOT-CAUGHT"])
    def test_the_matcher_reads_calls_and_not_mentions_of_them(
            self, source, expected):
        """ANTI-VACUITY for the MATCHER, both directions.

        ``on-a-branch-never-taken`` is the whole point of a static ban: the
        call is unreachable and is still read, which is what a runtime probe
        cannot do.

        ``calling-the-twin-NOT-CAUGHT`` is a **declared blind spot**, written
        as a case rather than as a sentence so that it cannot quietly stop
        being true: ``r.twin(r.state)`` has no distinctive name to ban, and
        banning ``twin`` would forbid every legitimate read of the field the
        payload exists to carry.
        """
        assert _out_of_bounds_calls(source) == expected

    def test_the_package_imports_no_optional_dependency(self):
        """``document.py`` imports this package and ``rheplicant.config``
        imports ``document``, so an in-flight module that dragged in
        ``limtod_jax`` at module scope would put it in every process that so
        much as reads a config.  Read statically, so it covers a module whose
        import this session happens not to have run."""
        banned = {"numpyro", "limtod_jax", "healpy", "h5py", "pyuvdata",
                  "rhino_cal_jax"}
        for path in _inflight_sources():
            for node in ast.walk(ast.parse(path.read_text())):
                if isinstance(node, ast.Import):
                    names = {entry.name.split(".")[0] for entry in node.names}
                elif isinstance(node, ast.ImportFrom):
                    names = {(node.module or "").split(".")[0]}
                else:
                    continue
                assert not (names & banned), f"{path.name} imports {names}"


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------


def _median_ms(call, repeats=200) -> float:
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1e3)
    return statistics.median(samples)




class TestTheCostOfTheTwoSlots:
    """**Measured in isolation, not as a fraction of a load, and the reason is
    a measurement.**

    The plan asks for "the built pass adds under 5 % to a warm
    ``load_document``".  Measured at ``e0e024a`` in this worktree: a warm
    ``load_document`` on the worked document is **2.77 ms median with a 3.94 ms
    maximum** -- a spread of over 40 %.  Five per cent of it is ~0.14 ms, which
    is well INSIDE that spread, so such a bound passes or fails by luck and
    tells nobody anything.  A test that cannot fail for the reason it names is
    worse than no test.

    So both bounds below are on the PASS ITSELF, which is measurable.

    **The bounds are set near the measurement, and that is a correction.**  As
    first written they were 1 ms against a 0.0003 ms median -- a margin of
    **x3008**, under which a *thousandfold* slowdown of the shared ``sweep``
    left the suite at exit 0.  A bound that cannot fail for the reason it
    names is not a test.  Every bound below is now within about **six** times
    its own measured best case, taken with :func:`best_ms`, and the
    measurement is written beside it so the next task can see what it is
    keeping.  Six and not ten, and that too is a correction: a review of Task
    1b applied a clean ``x10`` slowdown of the pass and found both of these
    bounds SURVIVING it, at margins of x11 and x20.  Six is the largest margin
    that still dies at ``x10``.

    **What these bounds cannot see:**

    * the cost of BUILDING the payload.  ``axis_facts`` is 0.137 ms and
      ``built_run`` is 2.8 ms; those are the builders' costs, not the passes',
      and they are what the slot positions are chosen around.
    * a check that is slow on a document none of these are.  These run the
      registries as they stand over the worked document; a per-document cost
      belongs beside the check that has it, which is where
      ``test_inflight_grids.py`` puts its own.
    * a cold first call.  Everything here is warm by construction, and the
      first call of a session is measurably not: ``operator_table()`` alone is
      1.7e-04 s the first time.
    """

    def test_the_axes_pass_costs_a_small_fraction_of_a_millisecond(self):
        """**0.09 ms against a measured 0.0138 ms best case** -- about x6.
        That catches the failures which actually threaten this slot (a payload
        deep-copied per check, a module re-imported per call, a file opened, a
        beam built: all of them milliseconds) AND catches a tenfold regression
        of the runner, which neither the old x3008 margin nor the x11 that
        replaced it did."""
        facts = axis_facts(preflight_document())
        axes(facts)                                    # warm
        assert best_ms(lambda: axes(facts)) < 0.09

    def test_the_built_pass_costs_a_small_fraction_of_a_millisecond(self):
        """**0.02 ms against a measured 0.0032 ms best case** -- about x6.

        **RE-TAKEN at Task 7**, as the previous version of this docstring said
        it would be.  Until then the built registry was empty and 0.002 ms was
        the runner's own overhead and nothing else; Task 7 registers B5, C9,
        A43 and B9, and on the worked document all four reach their stand-down
        in the first few lines -- no tone, no cal load, no second projector --
        so 0.0032 ms is what four early returns and the runner cost together.
        The old bound would fail here for the right reason and the wrong one:
        the pass really is ten times what it was, and it is doing ten times as
        much.

        Six and not ten, for the reason this class's docstring gives: six is
        the largest margin that still dies under a clean ``x10`` slowdown of
        the pass, and a review of Task 1b found the ``x10``/``x20`` bounds
        surviving exactly that mutation.
        """
        run = built_run(preflight_document())
        built(run)                                     # warm
        assert best_ms(lambda: built(run)) < 0.02

    def test_the_axes_pass_is_under_a_hundredth_of_a_second(self):
        """The plan's own §0.1 bound for this slot, on the worked document,
        kept verbatim so that a later task can see which number it must
        keep -- and kept KNOWING it is roughly x700 the measurement (x670 on
        one box, x752 on another; it is a property of the machine, not of the
        pass).

        **This assertion is IMPLIED by the one above, and saying so is the
        point.**  Both now run the same estimator on the same call, so
        ``best_ms(...) < 10.0`` cannot fail while ``best_ms(...) < 0.09``
        passes -- there is no such world, and the x10 and x1000 runs below are
        exactly that shape.  It is here to carry §0.1's contract number where a
        later task will see it, not to add sensitivity.  Before the change it
        did have one independent failure mode: a single-reading outlier.  That
        mode was the flake, not a regression it caught.

        **Taken with :func:`best_ms` rather than from one call, because one
        call measured the box.**  As a single ``perf_counter`` reading it went
        red 1 run in 8 under ``pytest tests/config --no-cov -n 16``, at
        **14.2 ms** against this 10 ms -- while ``best_ms`` puts the same call
        at **0.0149 ms** and the median of 200 at 0.0155 ms.  That is a
        scheduling stall of about x950, not a cost: fifteen sibling xdist
        workers can only ADD time, which is the argument ``best_ms``' own
        docstring makes and the reason the two tests above already take their
        numbers that way.  A guard that goes red 1 run in 8 for reasons that
        are not about the code makes every other red in the suite need triage
        before it can be believed, which is the real cost of leaving it.

        **What it can and cannot catch, measured rather than claimed.**  A
        clean x10 slowdown of the pass leaves this GREEN (0.13-0.15 ms against
        10) and takes the 0.09 ms test above red -- which is that test's
        calibrated job, "six is the largest margin that still dies at x10", and
        which was run rather than trusted.  A x1000 slowdown takes this one red
        at 13.8-13.9 ms, so the contract is not decoration either.  The tight
        bound is the instrument, this one is the contract, and neither is now
        decided by what else the machine is doing.
        """
        facts = axis_facts(preflight_document())
        axes(facts)                                    # warm
        assert best_ms(lambda: axes(facts)) < 10.0     # §0.1's 0.01 s, in ms

    def test_building_the_axes_payload_does_not_pay_for_the_beam(self):
        """The slot's REASON, in time.  ``build_resources`` is 90.9 % of
        ``load_document``; an axes payload that cost the same as a load would
        mean the hook had slid below it.  Generous by design -- this is a
        ratio assertion and ratios are what the class docstring warns about --
        so it is written as an order of magnitude rather than a percentage."""
        doc = preflight_document()
        axis_facts(doc)
        load_document(doc)
        facts_ms = _median_ms(lambda: axis_facts(doc), repeats=20)
        load_ms = _median_ms(lambda: load_document(doc), repeats=20)
        assert facts_ms * 3 < load_ms, (facts_ms, load_ms)


class TestTheHelpersDoNotRollTheirOwnDocument:
    """``inflight_helpers`` DELEGATES to ``preflight_helpers``, so the twin
    repair travels with it and ``test_config_fixture_contract``'s census holds
    it to the same standard as every other helper module."""

    def test_it_defines_no_document_builder(self):
        """Which is why ``_BUILDER_FLOOR`` needs no new row: that table's
        contract is "every helper module that DEFINES a builder must appear",
        and this one builds PAYLOADS out of documents somebody else assembled.
        ``inference_helpers`` is already in the glob with no row for the same
        reason.  Kills a later task adding a ``*_document`` here without the
        row, which would go red in a module that has nothing to do with it."""
        import tests.config.inflight_helpers as helpers

        assert [name for name in vars(helpers) if name.endswith("_document")] == []

    def test_the_projector_patch_acknowledges_float32(self, tmp_path):
        """**A44's condition is ``runtime.jax_enable_x64``** -- absent means
        float32, which means A44 fires BY DEFAULT.  The base document is
        float32 (measured), so without this key every test that reaches for a
        projector would carry an unrelated A44 and every "and nothing else"
        assertion built on one would be about the wrong thing.  Kills the key
        being dropped, or flipped to ``False``, as a tidy-up."""
        pytest.importorskip("limtod_jax")
        from tests.config.inflight_helpers import projector_sections

        section = projector_sections(tmp_path)
        assert section["projectors"]["drift"][
            "acknowledge_float32_sky"] is True
        assert not axis_facts(preflight_document()).runtime.jax_enable_x64

    def test_the_projector_patch_builds(self, tmp_path):
        """The patch is only useful if a document carrying it LOADS -- a
        fixture that refuses would make every test built on it a test of the
        fixture."""
        pytest.importorskip("limtod_jax")
        from tests.config.inflight_helpers import projector_sections

        doc = preflight_document(resources=projector_sections(tmp_path))
        run = built_run(doc, base_dir=str(tmp_path))
        assert run.resources.resources["resources.projectors.drift"] is not None
        assert np.asarray(
            run.resources.resources["resources.beams.horn"]).size
