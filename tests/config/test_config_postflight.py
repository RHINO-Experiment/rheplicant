"""The fourth pass: its payload, its registry, its position, and the record.

**This module tests a MECHANISM and registers no real check.**  Plan 3C's
Task 3 ships the phase and the artefacts Tasks 4, 5 and 6 build on; the checks
that go in the registry (C12, C13, C19, C16, C18) arrive afterwards.  So
:data:`~rheplicant.config.postflight.CHECKS` is EMPTY as this module lands, and
every assertion about it is either driven by a probe registered into a CLEARED
registry or written in a subset form that stays true as later tasks land.

**The four legal registry assertion forms are 3B's**, restated because they
are what keeps this file green across a merge that changed nothing it is
about: ``CHECKS["C12"] is _f`` · ``{"C12"} <= report.checks()`` ·
``"C16" not in ids`` · ``priced_only(doc, "C12")``.  **Banned**: ``len(...)``,
``set(...) == {...}``, any insertion-index assertion,
``len(report.findings) == n``, ``report.refusals()[0]``.

**Anti-vacuity is deliberate throughout**, and one test in here is vacuous
today by construction -- see
:meth:`TestTheDiscoveryMechanism.test_every_module_under_postflight_contributes_a_slot`,
which says so in its own docstring and ships two partners that fail when its
matcher stops matching.

**What this module deliberately does NOT test.**  The receiver name of the
fourth hook in ``document.py`` (``priced_report``) is pinned by
``test_config_inflight.py::test_all_four_hooks_are_present_and_named``, whose
set this task widens to four.  A second copy here would be a second binding of
one contract, and the two would drift.
"""

import copy
import dataclasses
import pathlib
import pkgutil
import subprocess
import sys
import tempfile
import types
import warnings

import pytest

import rheplicant.config.document as document_module
import rheplicant.config.sections.exits as exits_module
from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import ConfigWarning, refuse, report, warn
from rheplicant.config.gating import CHECK_NAMES, Gate, gates
from rheplicant.config.inflight import (
    AXIS_CHECKS,
    BUILT_CHECKS,
    register_axes,
    register_built,
)
from rheplicant.config.postflight import (
    _RESERVED,
    CHECKS,
    Priced,
    _discoverable,
    _reserved,
    priced,
    register,
)
from rheplicant.config.preflight import CHECKS as TEXT_CHECKS
from rheplicant.config.preflight import register as register_text
from rheplicant.config.sections.runs import run_document
from tests.config.inflight_helpers import (
    built_run,
    priced_findings,
    priced_only,
    priced_run,
)
from tests.config.preflight_helpers import preflight_document

#: ``tests/config/`` -> the repository root.
_ROOT = pathlib.Path(__file__).resolve().parents[2]


@pytest.fixture
def priced_registry():
    """:data:`CHECKS`, EMPTIED for one test and restored afterwards.

    Restored, because a probe left behind leaks into every later
    ``load_document`` in the session -- including other modules' -- and the
    failures land nowhere near the cause.  **Emptied**, because Tasks 4, 5 and
    6 register real ids into this same dict at import time: a test that
    registered ``C12`` would then hit "registered twice", and one that asserts
    a refusal's message VERBATIM would read a real check's refusal instead of
    its own.
    """
    saved = dict(CHECKS)
    CHECKS.clear()
    try:
        yield CHECKS
    finally:
        CHECKS.clear()
        CHECKS.update(saved)


@pytest.fixture
def every_registry(priced_registry):
    """All FOUR registries emptied and restored.

    Only the accumulation test needs this, and it needs it for a reason worth
    stating: it asserts the ORDER of four findings across four passes, and a
    real check firing on the base document would land between two of them and
    make the assertion about the wrong list.
    """
    saved = (dict(TEXT_CHECKS), dict(AXIS_CHECKS), dict(BUILT_CHECKS))
    for registry in (TEXT_CHECKS, AXIS_CHECKS, BUILT_CHECKS):
        registry.clear()
    try:
        yield
    finally:
        for registry, was in zip((TEXT_CHECKS, AXIS_CHECKS, BUILT_CHECKS),
                                 saved, strict=True):
            registry.clear()
            registry.update(was)


def _document(**patch):
    """The base document, patched.  One binding for what these tests load."""
    return preflight_document(**patch)


# ---------------------------------------------------------------------------
# The payload
# ---------------------------------------------------------------------------


class TestThePayload:
    """Kills: a two-argument check dying inside ``sweep``'s raise-guard; a
    check writing into the gates every later check reads; a payload whose
    ``run`` carries an empty report while its docstring says otherwise."""

    def test_the_payload_has_exactly_two_fields(self):
        """The field-tuple pin, beside the three ``InferenceBuild`` /
        ``ObservedBuild`` / ``NoiseBuild`` ones.

        **Two, and no more.**  An earlier draft of this contract spelled the
        payload ``Priced(*run, gates)`` -- eight positional fields mirroring
        ``ConfiguredRun`` plus the gates -- which puts ``ConfiguredRun``'s
        field list in a second place that has to be kept in step by hand.
        ``run`` is the whole object, so there is nothing to keep in step.
        """
        assert tuple(field.name for field in dataclasses.fields(Priced)) == (
            "run", "gates")

    def test_the_payload_is_frozen(self):
        """A check receives the payload and must not be able to edit the run
        out from under the pass that is still iterating over checks."""
        payload = priced_run(_document())
        with pytest.raises(dataclasses.FrozenInstanceError):
            payload.run = None

    def test_the_gates_mapping_is_read_only(self):
        """**One mapping is handed to every check.**

        Task 4's natural sloppy write -- ``gates[name] = auto_skipped(...)``
        -- would silently change what a later check sees, in the same file, by
        one drafter, and no assertion about either check's own findings could
        notice.  So the mapping a check is handed is a ``MappingProxyType``.
        """
        payload = priced_run(_document())
        assert isinstance(payload.gates, types.MappingProxyType)
        with pytest.raises(TypeError):
            payload.gates["linearity"] = None

    def test_which_half_of_the_payload_is_actually_frozen(self):
        """M5: "frozen" and "read-only" are not one property; here is which
        half is which, pinned so a future change that opens one of the four
        genuinely-closed writes is caught.

        ``Priced`` is ``frozen=True``, the gates are a ``MappingProxyType``,
        ``InferenceBuild`` is a ``NamedTuple`` and ``Report`` is
        ``frozen=True`` -- all four writes below raise.  But
        ``payload.run.document`` is a plain ``dict`` and
        ``payload.run.inference.checks`` is a plain ``dict`` inside a frozen
        NamedTuple -- freezing stops attribute REBINDING, not mutation of
        what an attribute already points at -- so those two succeed
        silently and the write is visible to every later check and escapes
        onto the ``ConfiguredRun`` the caller holds.  Not asserted here as a
        MUST, only as a fact: this is a pre-existing property of the layer
        below this payload, and the point of the docstring and this test
        together is that a reader learns which half of "frozen" covers what.
        """
        payload = priced_run(_document())
        with pytest.raises(dataclasses.FrozenInstanceError):
            payload.run = None
        with pytest.raises(TypeError):
            payload.gates["linearity"] = None
        with pytest.raises(AttributeError):
            payload.run.inference.space = None
        with pytest.raises(dataclasses.FrozenInstanceError):
            payload.run.report.findings = ()

        payload.run.document["model"]["_written_by_a_check"] = True
        assert payload.run.document["model"]["_written_by_a_check"] is True
        payload.run.inference.checks["_written_by_a_check"] = None
        assert "_written_by_a_check" in payload.run.inference.checks

    def test_the_gates_proxy_costs_more_than_write_access(self):
        """M6: the ``MappingProxyType`` wrapper costs ``deepcopy``, ``asdict``
        and ``hash`` too, not only assignment -- and ``dataclasses.replace``
        is the escape a drafter reaching for ``copy.deepcopy`` needs, named
        because ``preflight_helpers._base()`` deepcopies routinely.

        **``run=None``, deliberately, and not a real ``priced_run(...)``.**
        A real ``ConfiguredRun`` carries its own immutable containers several
        layers down (``rheplicant.core.frozen.FrozenMapping`` among them),
        and ``copy.deepcopy`` on the whole payload trips one of THOSE first
        -- a separate, true fact about the layer below ``run`` and not what
        this test is about.  The claim under test is narrower and isolated
        by handing ``run`` a plain value: it is the ``gates`` proxy itself
        that neither serialising code nor ``hash`` can get past.
        """
        payload = Priced(run=None, gates=gates(None))
        copy.copy(payload)
        replaced = dataclasses.replace(payload, run="something else")
        assert isinstance(replaced.gates, types.MappingProxyType)
        with pytest.raises(TypeError, match="mappingproxy"):
            copy.deepcopy(payload)
        with pytest.raises(TypeError, match="mappingproxy"):
            dataclasses.asdict(payload)
        with pytest.raises(TypeError, match="unhashable"):
            hash(payload)

    def test_the_gates_are_a_copy_of_what_the_caller_passed(self):
        """The proxy wraps a COPY, never the caller's own dict.

        A proxy over the live dict is read-only from the check's side and
        wide open from the builder's: whoever still holds the original writes
        through it and every check downstream reads the change.  Measured here
        rather than asserted, because the difference is invisible in the type.
        """
        source = gates(None)
        payload = Priced(run=None, gates=source)
        source["linearity"] = "not a gate at all"
        assert isinstance(payload.gates["linearity"], Gate)

    def test_gates_that_are_not_a_mapping_are_refused_at_construction(self):
        """A bug in the CALLER, so it is a ``ValueError`` and not a
        ``ConfigError``: a caller who wrapped ``load_document`` in
        ``except ConfigError`` would otherwise swallow it and report the
        document as at fault.  Left unchecked it is a ``TypeError`` from
        inside ``dict()`` naming neither the field nor the fix.
        """
        with pytest.raises(ValueError) as raised:
            Priced(run=None, gates=None)
        assert str(raised.value) == (
            "Priced.gates is the resolved gates -- gating.gates(...)'s "
            "mapping of check name to Gate, cardinality three whatever the "
            "document says; got None."
        )
        assert not isinstance(raised.value, ConfigError)

    def test_a_check_that_writes_into_the_gates_fails_the_pass_by_name(
            self, priced_registry):
        """The property, not the type.  A check that assigns is stopped, and
        the sentence names the check rather than the user's document."""

        @register("C12")
        def _greedy(payload):
            payload.gates["linearity"] = None
            return ()

        with pytest.raises(ConfigError) as raised:
            priced(priced_run(_document()))
        assert str(raised.value).startswith(
            "post-flight check 'C12' RAISED TypeError: ")

    def test_the_gates_carry_the_defaults_a_document_did_not_write(
            self, priced_registry):
        """Cardinality is THREE, always -- a check never writes
        ``gates.get("linearity", <its own default>)``, which is how two
        default tables come to exist and one of them to be wrong."""
        seen = {}

        @register("C12")
        def _reader(payload):
            seen.update(payload.gates)
            return ()

        priced(priced_run(_document()))
        assert set(seen) == CHECK_NAMES
        assert seen["linearity"].state == "refuse"
        assert seen["identifiability"].state == "off"

    def test_the_gates_are_this_documents_and_not_the_default_table(
            self, priced_registry):
        """**The gates are READ OFF THE DOCUMENT**, and this is the only test
        in this file that says so.

        Every other assertion about ``payload.gates`` here is on a document that
        declares no ``inference.checks:`` at all -- and ``gating.gates(None)``
        returns exactly the defaults those assertions read, so a
        ``_priced_payload`` that ignored the document entirely and always
        handed over ``gates(None)`` would pass all of them.  Measured: that
        mutation survives this module without this test.

        Tasks 4, 5 and 6 are ENTIRELY about what a document's ``mode:`` does,
        so a gate that never reflects the document would make every one of
        their gating tests a test of the default table.
        """
        seen = {}

        @register("C12")
        def _reader(payload):
            seen.update(payload.gates)
            return ()

        load_document(_document(inference={"checks": {
            "linearity": {"mode": "warn"},
            "identifiability": {"mode": "report", "rtol": 1e-6},
        }}))
        assert seen["linearity"].state == "warn"
        assert seen["identifiability"].state == "report"
        assert seen["identifiability"].rtol == 1e-6
        assert seen["prior_sensitivity"].state == "off"

    def test_priced_run_takes_a_variant_keyword(self):
        """N7: ``priced_run(document, *, base_dir=None)`` had no ``variant=``
        while :func:`~rheplicant.config.document.load_document` does --
        the one keyword ``built_run`` and ``priced_run`` both lacked while
        ``base_dir`` was present on both.  Threaded to ``_assemble`` exactly
        as ``base_dir`` already was.
        """
        base = _document()
        assert base["model"]["gain"]["gain"]["value"] != 1.0
        payload = priced_run(base, variant="unity_gain")
        assert payload.run.document["model"]["gain"]["gain"]["value"] == 1.0

    def test_the_base_document_earns_no_post_flight_finding_of_its_own(self):
        """The REAL registry -- deliberately no clearing fixture.

        3B ships exactly this for its two passes
        (``test_config_inflight.py:552``) and says why: a base that is itself
        a finding makes every later task's "and nothing else" assertion
        inherit an extra id, with nothing recording it.  Vacuous while
        ``CHECKS`` is empty and load-bearing from Task 4 on -- which is when a
        check written against the wrong document would first be noticed.
        """
        assert priced_findings(_document()) == ()

    def test_the_run_carries_every_finding_earned_before_this_pass(
            self, every_registry):
        """``Priced.run.report`` is pre-flight + axes + built, in run order.

        **This is the contract a priced check leans on to not restate an
        earlier one.**  It is also the half of D-3's docstring line that the
        `Built.report` field alone does not buy: measured, with the field and
        no accumulation, ``built_run(preflight_document()).report`` is
        ``Report(findings=())``.
        """
        register_text("C11")(
            lambda doc: (report("C11", "runtime.dtype", "text."),))
        register_axes("B3")(
            lambda facts: (report("B3", "observation.time", "axes."),))
        register_built("A9")(
            lambda run: (report("A9", "model.gain", "built."),))
        payload = priced_run(_document())
        assert [one.check for one in payload.run.report.findings] == [
            "C11", "B3", "A9"]

    def test_a_priced_check_cannot_see_an_earlier_priced_checks_finding(
            self, priced_registry):
        """B1's correction, as a pin rather than a corrected sentence.

        ``Priced.run.report`` is pre-flight + axes + built -- **NOT this
        pass**.  ``sweep`` accumulates ``priced``'s own findings into a local
        list and returns only once every check has run, so a C13 that runs
        after a C12 in ``sorted(CHECKS)`` order still reads an empty report
        for this pass: two priced checks cannot communicate through
        ``payload.run.report``, in either direction.  A check that needs
        another check's answer must recompute it or the two must be bound to
        one function (``@register('C13', 'C19')``).
        """
        seen = []

        @register("C12")
        def _first(payload):
            return (report("C12", "inference.parameters", "C12's finding."),)

        @register("C13")
        def _second(payload):
            seen.append([one.check for one in payload.run.report.findings])
            return ()

        priced(priced_run(_document()))
        assert seen == [[]]


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


class TestTheRegistry:
    """Kills: a check bound into the wrong pass by an import of the wrong
    ``register``; a two-slot function emitting twice; a run order that is
    whatever the import graph happened to be."""

    def test_a_registered_id_binds_to_its_function(self, priced_registry):
        @register("C12")
        def _one(payload):
            return ()

        assert CHECKS["C12"] is _one

    def test_register_writes_into_this_packages_registry_alone(
            self, priced_registry):
        """``register`` is a SECOND function and not an import of
        ``preflight.register``.

        **The two share a NAME**, so a Task 4/5/6 module that writes
        ``from rheplicant.config.preflight import register`` binds its check
        into the text pass in silence -- where the payload is a document, so
        ``payload.gates`` is a ``KeyError`` laundered into a sentence blaming
        the check author.  ``test_every_module_under_postflight_contributes_a_slot``
        is what catches that in production; this is what says the two
        registries are distinct objects at all.
        """
        register("C12")(lambda payload: ())
        assert "C12" in CHECKS
        assert "C12" not in TEXT_CHECKS
        assert "C12" not in AXIS_CHECKS
        assert "C12" not in BUILT_CHECKS

    def test_the_four_registries_share_no_id_and_no_function(self):
        """The LIVE registries, over whatever has landed.

        Two things are asserted and they fail for different reasons: a
        function object in two registries is one check bound in two passes,
        which means it is handed two payload types and reads one of them
        wrong; a slot id in two registries is two different functions
        answering to one name, after which a reader looking a finding up finds
        the wrong sentence.

        The three in-flight-and-earlier registries are asserted against each
        other too, so this stays the whole-layer statement rather than a
        statement about the newest pass.
        """
        named = {"pre-flight": TEXT_CHECKS, "axes": AXIS_CHECKS,
                 "built": BUILT_CHECKS, "post-flight": CHECKS}
        for one, first in named.items():
            for two, second in named.items():
                if one >= two:
                    continue
                shared_ids = sorted(set(first) & set(second))
                assert not shared_ids, (one, two, shared_ids)
                shared = set(map(id, first.values())) & set(
                    map(id, second.values()))
                assert not shared, (one, two, sorted(
                    fn.__qualname__ for fn in first.values()
                    if id(fn) in shared))

    def test_run_order_is_slot_order_whatever_the_registration_order(
            self, priced_registry):
        """**Not insertion order**, which under discovery is the IMPORT
        GRAPH's rather than the filename's: measured, ``digitising``
        head-importing ``fitting`` gives ``['C12','C13','C19','C16','C18']``
        while no sibling import gives ``['C16','C12','C13','C19','C18']``.
        This repo already deleted a shipped ordering assertion for that reason
        (``8bcf74d``).

        Registered in each of the two orders, and the findings come back
        sorted both times.  Lexicographic, which is not numeric:
        ``sorted(['C9','C12'])`` is ``['C12','C9']``, and a reader expecting
        C9 first should find the answer here rather than in a bug report.
        """
        def _emit(check):
            return lambda payload, check=check: (
                refuse(check, "model.gain", f"{check}!"),)

        for order in (("C16", "C12"), ("C12", "C16")):
            CHECKS.clear()
            for check in order:
                register(check)(_emit(check))
            found = priced(priced_run(_document())).refusals()
            assert [one.check for one in found] == ["C12", "C16"], order

    def test_a_function_bound_to_several_ids_runs_exactly_once(
            self, priced_registry):
        """One function carries several ids and this plan does it twice --
        C13 and C19 both escalate through C14, and Task 5's C16 is one
        function over two thresholds.  Kills a walk with no de-duplication by
        identity, under which the user reads one mistake twice."""
        calls = []

        @register("C13", "C19")
        def _both(payload):
            calls.append(1)
            return (refuse("C13", "inference.parameters", "one."),)

        found = priced(priced_run(_document())).refusals()
        assert calls == [1]
        assert [one.check for one in found] == ["C13"]

    def test_a_check_that_raises_names_itself_and_this_pass(
            self, priced_registry):
        """WHOLE-STRING, because the label is the one word that separates this
        sentence from the three other passes' and a ``match=`` substring
        beginning after it discriminates nothing.  Measured while ``passes.py``
        was written: rewriting every ``pre-flight`` to ``in-flight`` left the
        whole of ``tests/config`` at exit 0."""
        register("C12")(lambda payload: (_ for _ in ()).throw(
            RuntimeError("the Jacobian went missing")))
        with pytest.raises(ConfigError) as raised:
            priced(priced_run(_document()))
        assert str(raised.value) == (
            "post-flight check 'C12' RAISED RuntimeError: the Jacobian went "
            "missing. A check returns findings and raises nothing -- one that "
            "raises aborts the pass and hides every finding after it, which "
            "is the failure the collect-rather-than-raise design exists to "
            "prevent."
        )

    def test_the_where_guard_speaks_in_this_passs_voice(self, priced_registry):
        """``preflight._check_where`` is NOT imported: it is bound with
        ``_LABEL = "pre-flight"`` already closed over, so a post-flight pass
        calling it would report its own defect as a pre-flight one.
        ``passes.sweep`` does the guarding with this pass's label instead, and
        this is the pin that says so."""
        register("C12")(lambda payload: (
            refuse("C12", "src/rheplicant/config/postflight/fitting.py",
                   "wrong."),))
        with pytest.raises(ConfigError) as raised:
            priced(priced_run(_document()))
        assert str(raised.value).startswith("post-flight check 'C12' emitted ")

    def test_a_where_whose_head_is_not_a_section(self, priced_registry):
        """N3: the where-guard's SECOND branch (valid path syntax, head not a
        section), unexercised for this pass until now.

        The test above drives the FIRST branch only -- a slash-separated
        source path is not valid ``head.step.step`` syntax at all, so it
        never reaches the "is the head a section" check.  ``beam.horn`` is
        syntactically a path; ``'beam'`` is simply not one of
        :data:`~rheplicant.config.postflight._DOCUMENT_SECTIONS`'s twelve
        names.  The two sibling passes pin this branch whole
        (``test_config_preflight.py::test_a_where_whose_head_is_not_a_section``,
        ``test_config_inflight.py::test_a_where_whose_head_is_not_a_section``);
        this is this pass's own copy.
        """
        register("C12")(lambda payload: (
            refuse("C12", "beam.horn", "reserved."),))
        with pytest.raises(ConfigError) as raised:
            priced(priced_run(_document()))
        assert str(raised.value) == (
            "post-flight check 'C12' emitted where='beam.horn', whose first "
            "segment 'beam' is not a document section. The sections are "
            "['schema_version', 'defaults', 'plugins', 'runtime', "
            "'observation', 'resources', 'model', 'variants', 'inference', "
            "'runs', 'outputs', 'campaign']."
        )

    def test_the_binder_refusals_open_with_this_passs_label(self):
        """The registration-time refusals, whole.  ``decorator="register"`` is
        this pass's own word: advice naming ``register_built`` cannot be
        followed here."""
        with pytest.raises(ConfigError) as raised:
            register("A12a")
        assert str(raised.value) == (
            "post-flight check id 'A12a' is not a schema §6 id "
            "(A1..A52, B1..B9, C1..C19), optionally with a dotted suffix such "
            "as 'A1.runs' when several functions each decide part of one "
            "check. The id is what a Finding carries and what a reader looks "
            "up; a private name here reaches the user as '(check _mine).'"
        )

    def test_the_binder_names_this_passs_own_decorator(self):
        """``decorator="register"``, WHOLE -- and the slot refusal above
        cannot see it.

        The slot refusal carries the LABEL and no decorator at all, so
        ``_DECORATOR`` was unpinned: measured, ``_DECORATOR = "register_built"``
        left ``test_config_postflight.py`` and ``test_config_inflight.py`` at
        **exit 0** while telling a Task 4/5/6 drafter to write
        ``@register_built``, which binds the check into the BUILT registry in
        silence.  The other three passes pin their own word the same way --
        ``test_config_preflight.py:1044``, ``test_config_inflight.py:356``
        and ``:367``.
        """
        with pytest.raises(ConfigError) as raised:
            register()
        assert str(raised.value) == (
            "register() takes one or more check ids -- @register('A30'), or "
            "@register('A16', 'A17') when one function decides several. A "
            "registration with no id binds nothing, so the check it decorates "
            "never runs and nothing says so."
        )

    def test_a_slot_claimed_twice_names_both_modules(self, priced_registry):
        register("C12")(lambda payload: ())
        with pytest.raises(ConfigError) as raised:
            register("C12")(lambda payload: ())
        assert str(raised.value).startswith(
            "post-flight check 'C12' is registered twice, by ")


# ---------------------------------------------------------------------------
# The hook: where it runs, and what happens when it refuses
# ---------------------------------------------------------------------------


class TestTheHookIsPositioned:
    """Kills: the pass sliding above ``build_inference``, where the space and
    the fit twin do not exist yet; the pass never being called at all; a
    refusal that lets a ``ConfiguredRun`` out of the door anyway."""

    def test_the_pass_runs_after_build_inference(self, priced_registry,
                                                 monkeypatch):
        """It needs the space, the fit twin and the observed data, and all
        three are ``build_inference``'s.  The hook is ONE call in
        ``load_document``; moving it above that builder is a green edit that
        hands every priced check an ``InferenceBuild`` that does not exist."""
        order = []
        real = document_module.build_inference
        monkeypatch.setattr(document_module, "build_inference",
                            lambda *a, **k: (order.append("inference"),
                                             real(*a, **k))[1])
        register("C12")(lambda payload: (order.append("priced"), ())[1])
        load_document(_document())
        assert order == ["inference", "priced"]

    def test_a_refusal_stops_the_load_and_hands_back_no_object(
            self, priced_registry):
        """**Before the return**, so no caller ever holds a ``ConfiguredRun``
        whose priced checks have not run."""
        register("C12")(lambda payload: (
            refuse("C12", "inference.parameters", "not linear."),))
        with pytest.raises(ConfigError, match="not linear."):
            load_document(_document())

    def test_two_refusals_both_arrive(self, priced_registry):
        """Collect, do not raise: a user with two problems sees two.  Read off
        ``raise_if_refused``'s tail, which is the only place the second one
        reaches a user."""
        register("C12")(lambda payload: (
            refuse("C12", "inference.parameters", "first."),))
        register("C16")(lambda payload: (
            refuse("C16", "model.adc", "second."),))
        with pytest.raises(ConfigError) as raised:
            load_document(_document())
        assert str(raised.value) == (
            "first.\n(This document has 1 more refusal, at model.adc.)")

    def test_the_pass_runs_when_the_document_has_no_inference_section(
            self, priced_registry):
        """A document with no latents has ``space is None`` and every gate
        stands down -- but C16 needs no space, and C16 is exactly the check a
        ``kind: forward`` document wants.  A pass that short-circuited on
        ``inference is None`` would lose it."""
        seen = []
        register("C16")(lambda payload: (seen.append(payload.gates["linearity"]),
                                         ())[1])
        load_document(_document(inference=None))
        assert [one.state for one in seen] == ["refuse"]

    def test_a_priced_refusal_never_reaches_the_built_helper(
            self, priced_registry):
        """**The trap this task exists to not fall into.**

        ``inflight_helpers.built_run`` calls ``_assemble`` precisely so a
        raising hook cannot hide a slot's refusing half -- its own docstring
        says a helper built on ``load_document`` could only ever exercise the
        passing half of the slot it exists to test.  Moving the priced hook
        into ``_assemble`` re-arms that failure for every built-slot test and
        for Tasks 4-6's own helpers, and every other test in this file stays
        green while it does.
        """
        register("C12")(lambda payload: (
            refuse("C12", "inference.parameters", "priced refusal."),))
        assert built_run(_document()) is not None
        assert priced_run(_document()) is not None
        assert [one.check for one in priced_findings(_document())] == ["C12"]

    def test_the_accessor_a_later_task_writes_with_sees_this_pass(
            self, priced_registry):
        """``preflight_helpers.only()`` calls ``preflight(document)`` -- the
        TEXT pass alone -- so a Task 4, 5 or 6 assertion written with it
        **passes against an empty implementation**.  Measured on a document
        that fires an axes check: ``preflight_helpers.ids()`` sees ``[]``.

        :func:`~tests.config.inflight_helpers.priced_only` is what those tasks
        write instead, and this is the pin that says it reaches a post-flight
        finding at all.  A WARN deliberately, not a refusal: ``_only`` filters
        across all three severities, and a refusals-only reading would make
        every test about a ``mode: warn`` gate unwritable.
        """
        register("C12")(lambda payload: (
            warn("C12", "inference.parameters", "one warning."),))
        assert priced_only(_document(), "C12").message == "one warning."

    def test_expect_refuse_cannot_catch_a_priced_refusal(self, priced_registry,
                                                         monkeypatch):
        """§2.1's TRAP, as a test rather than as a sentence.

        ``expect:`` is PER RUN and this pass raises out of ``load_document``,
        before any run is executed.  That is correct rather than tolerated: a
        gate is a property of the DOCUMENT, and a run cannot expect a refusal
        of the document that configures it.  The executor is spied on so the
        claim is "never reached" rather than "returned no product".
        """
        reached = []
        monkeypatch.setattr(exits_module, "execute_run",
                            lambda *a, **k: reached.append(1))
        register("C12")(lambda payload: (
            refuse("C12", "inference.parameters", "the gate refuses."),))
        with pytest.raises(ConfigError, match="the gate refuses."):
            run_document(_document(
                runs=[{"kind": "forward", "expect": "refuse"}]))
        assert reached == []


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


class TestTheReportAccumulates:
    """Kills: a concatenation that keeps two passes of four; ``emit_warnings``
    "improved" to walk ``self.findings``, which converts the whole
    ``mode: report`` feature into ``warn`` in one word."""

    def test_every_passs_report_is_kept_and_in_pass_order(self, every_registry):
        """FOUR passes -- pre-flight, axes, built, priced -- one synthetic
        REPORT each.

        **The ids are deliberately not in sorted order.**  ``sorted`` of the
        four is ``['A9', 'B3', 'C11', 'C97']`` and the pass order is
        ``['C11', 'B3', 'A9', 'C97']``, so a walk that happened to sort the
        accumulated findings, or one that concatenated them in the wrong
        order, cannot pass this by luck.

        **The priced id is a synthetic ``C97``, not ``C17`` (MINOR 8, Plan 3C
        fix round).**  ``C17`` is a real, SHIPPED schema id (``sections/
        transforms.py``, ``sections/observed.py``) and D-14 bans any NEW id,
        message, test name or commit subject from being the string ``C17``
        outside a ``C1..C17`` range-literal widening -- this test's registered
        id is a mechanism probe with no relationship to the real check and
        does not need to borrow its name.  ``C97`` is outside every schema
        range (``A1..A52``, ``B1..B9``, ``C1..C19``) so it also cannot be
        mistaken for a future real id.
        """
        register_text("C11")(
            lambda doc: (report("C11", "runtime.dtype", "text."),))
        register_axes("B3")(
            lambda facts: (report("B3", "observation.time", "axes."),))
        register_built("A9")(
            lambda run: (report("A9", "model.gain", "built."),))
        register("C97")(
            lambda payload: (report("C97", "inference.parameters", "priced."),))
        run = load_document(_document())
        assert run.report.checks() == frozenset({"C11", "B3", "A9", "C97"})
        assert [one.check for one in run.report.findings] == [
            "C11", "B3", "A9", "C97"]

    def test_the_built_payload_carries_pre_flight_and_axes(self,
                                                           every_registry):
        """``Built.report`` is everything earned BEFORE the built pass, which
        is the docstring line D-3 ships -- and it is true only once the
        accumulation is in place.  Measured with the field alone and no
        accumulation: ``built_run(preflight_document()).report`` is
        ``Report(findings=())``."""
        register_text("C11")(
            lambda doc: (report("C11", "runtime.dtype", "text."),))
        register_axes("B3")(
            lambda facts: (report("B3", "observation.time", "axes."),))
        assert [one.check for one in built_run(_document()).report.findings
                ] == ["C11", "B3"]

    def test_a_report_finding_never_reaches_warnings_warn(self,
                                                          priced_registry):
        """``mode: report`` exists so a check can record a number without
        interrupting anybody.  ``emit_warnings`` walks ``self.warnings()`` and
        not ``self.findings``; the difference is one word and it converts the
        whole feature into ``warn``."""
        register("C12")(lambda payload: (
            report("C12", "inference.parameters", "recorded, not shouted."),))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run = load_document(_document())
        assert [one for one in caught
                if issubclass(one.category, ConfigWarning)] == []
        assert "C12" in run.report.checks()

    def test_a_warn_finding_does(self, priced_registry):
        """ANTI-VACUITY for the test above: it passes trivially if the hook
        stopped calling ``emit_warnings`` at all, or if the pass stopped
        running."""
        register("C12")(lambda payload: (
            warn("C12", "inference.parameters", "worth saying out loud."),))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_document(_document())
        assert [str(one.message) for one in caught
                if issubclass(one.category, ConfigWarning)] == [
                    "worth saying out loud."]

    def test_the_report_survives_onto_the_object_the_caller_holds(
            self, priced_registry):
        """The record is in memory, on the object the caller already has --
        which is the whole of "3C can gate but cannot durably record".  Kills
        ``report=`` being dropped from the final ``_replace``, under which the
        priced findings are computed, emitted and thrown away."""
        register("C12")(lambda payload: (
            report("C12", "inference.parameters", "kept."),))
        run = load_document(_document())
        assert [one.message for one in run.report.findings] == ["kept."]


# ---------------------------------------------------------------------------
# Discovery -- and what it cannot see
# ---------------------------------------------------------------------------


#: Run in a CHILD process: which modules under ``postflight/`` own a live slot
#: when the package is imported and nothing else has been.
#:
#: **In-process this question cannot be asked at all.**  A test module that
#: imports ``postflight.fitting`` to reach its check functions runs the
#: ``@register`` decorators, so by the time any test in a full session reads
#: ``CHECKS`` the slots are there whether or not the package's own discovery
#: ever imported the module.  Hence the child.
#:
#: ``sys.argv[1:]`` is folded into ``present`` so the anti-vacuity case can
#: give the child a module it cannot possibly find a slot for and watch it die
#: -- which is what says the ``assert``s in here are live at all (a child run
#: under ``-O``, or one whose import quietly failed upward, would exit 0 with
#: no assertion having been evaluated).
_WIRING_PROBE = """
import importlib
import pathlib
import sys

from rheplicant.config.postflight import CHECKS

here = pathlib.Path(importlib.import_module(
    "rheplicant.config.postflight").__file__).resolve().parent
present = {p.stem for p in here.glob("*.py") if p.stem != "__init__"}
present |= set(sys.argv[1:])
contributing = {fn.__module__.rsplit(".", 1)[-1] for fn in CHECKS.values()}
assert not present - contributing, (
    "live under postflight/ and own no slot in a fresh process: "
    + repr(sorted(present - contributing)))
assert not contributing - present, (
    "own a slot and are not modules under postflight/: "
    + repr(sorted(contributing - present)))
"""


def _uncontributing(directory: pathlib.Path,
                    registry: dict) -> set[str]:
    """Module stems under ``directory`` that own no slot in ``registry``.

    Both sides derived -- one by ``pkgutil`` over the directory, one by
    ``fn.__module__`` over the registry -- so neither is a list that can go
    stale.  Shared by the child probe's shape and by the temporary-package
    partner below, so the two cannot disagree about what "contributes" means.
    """
    present = {name for _, name, _ in pkgutil.iter_modules([str(directory)])}
    contributing = {fn.__module__.rsplit(".", 1)[-1]
                    for fn in registry.values()}
    return present - contributing


class TestTheScopeOfThisPackage:
    """The invariant this package's own docstring states and nothing checked.

    ``preflight`` has ``test_config_preflight.py::test_importing_the_pass_
    drags_in_no_optional_dependency``; ``inflight`` has
    ``test_config_inflight.py::test_the_package_imports_no_optional_
    dependency``.  This package had neither, while being the one whose three
    consumer tasks exist in order to take a Jacobian and run a Newton solve.
    """

    def test_importing_the_pass_drags_in_no_optional_dependency(self):
        """``document.py`` imports this package and ``rheplicant.config``
        imports ``document``, so one module-scope ``import numpyro`` under
        ``postflight/`` puts numpyro in every process that so much as reads a
        config.

        A SUBPROCESS and not ``inflight``'s static AST ban, deliberately: a
        priced check MAY import an optional dependency **inside a function**
        -- that is how ``prior_sensitivity`` reaches numpyro at all, and
        ``config/sections/noise.py:195`` is the shipped precedent -- and a
        walk over every ``ast.Import`` node forbids exactly that.  The child
        answers the question the invariant actually asks.
        """
        source = (
            "import sys, rheplicant.config.postflight\n"
            "print(sorted(m for m in ('numpyro', 'limtod_jax', 'healpy', "
            "'h5py', 'pyuvdata', 'rhino_cal_jax') if m in sys.modules))\n"
        )
        done = subprocess.run([sys.executable, "-c", source],
                              capture_output=True, text=True, check=True,
                              cwd=str(_ROOT))
        assert done.stdout.strip() == "[]", done.stdout


class TestTheDiscoveryMechanism:
    """``postflight/`` DISCOVERS its check modules; ``preflight/``'s explicit
    foot import stays as it is.

    The two packages differ because their plans do.  3A's explicit list
    catches the failure it had -- a module nobody imports registers nothing
    and stays green -- at the cost of a line every task must edit, and 3A's
    tasks ran sequentially.  **This plan's Tasks 4, 5 and 6 run in parallel**
    and would each add one line to it; and a docstring-only stub reserved by
    an earlier task turns the contributes-a-slot test RED, so "create the
    modules early and fill them later" is not available either.

    **What discovery cannot see, written down rather than discovered:**

    * **A module dropped into the package is imported whether or not anybody
      meant it.**  That is the opposite of 3A's failure and it is the price:
      there is no list to be incomplete, so there is also no list that has to
      agree.  ``test_every_module_under_postflight_contributes_a_slot`` is the
      only thing between a stray file and a silently-running import, and it
      only speaks about modules that own no slot -- a file that DOES register
      something is admitted with nobody having approved it.
    * **A module whose registration is bound into the WRONG registry.**  It
      owns no post-flight slot, so the walk names it -- but it names it as
      "owns no slot", which reads as "the decorator is missing" rather than
      as "the decorator came from ``preflight``".
      ``test_register_writes_into_this_packages_registry_alone`` is the other
      half.
    * **Import order.**  Discovery is ``sorted()`` over ``pkgutil``, but a
      module that head-imports a sibling registers that sibling's slots first,
      so insertion order is the import graph's.  Nothing here can see that,
      and nothing needs to: run order is ``sorted(CHECKS)`` and
      ``test_run_order_is_slot_order_whatever_the_registration_order`` is what
      says so.
    * **A package with no modules at all.**  Which is exactly today, and the
      contributes-a-slot test is therefore vacuous -- see its own docstring.
    """

    def test_every_module_under_postflight_contributes_a_slot(self):
        """PRESENCE IS NOT CONTRIBUTION, and it is asked IN A SUBPROCESS.

        **VACUOUS TODAY, and deliberately shipped anyway.**  ``postflight/``
        holds only ``__init__.py`` as this task lands, so both sides of the
        comparison are empty and the child exits 0 having compared nothing.
        Said out loud here exactly as 3A's Task 2 said it of
        ``test_every_registered_id_is_a_schema_6_id``: the value of a guard
        that is vacuous on the day it lands is that it is ALREADY THERE on the
        day Task 4 drops ``fitting.py`` in, rather than being remembered then.
        Its two partners below are what say the comparison can fail at all.

        A later plan that genuinely wants a helper module under ``postflight/``
        with no check of its own must say so by editing this test -- which is
        the point; the alternative is a silent hole.
        """
        done = subprocess.run(
            [sys.executable, "-c", _WIRING_PROBE],
            capture_output=True, text=True, cwd=str(_ROOT), check=False)
        assert done.returncode == 0, (
            "importing rheplicant.config.postflight does not leave every "
            "module under postflight/ owning a slot. In THIS process the same "
            "question answers 'fine', because the test modules import them "
            "directly.\n" + done.stdout + done.stderr)

    def test_the_subprocess_probe_can_fail(self):
        """ANTI-VACUITY for the child, and it is not ceremony.

        A subprocess guard has two silent failure modes an in-process one does
        not: the child exiting 0 without evaluating anything (``-O`` strips
        ``assert``), and the comparison being trivially true because both
        sides came out empty -- which is the state this package is actually
        in.  Handing the child a module name that owns no slot must kill it.
        """
        done = subprocess.run(
            [sys.executable, "-c", _WIRING_PROBE, "no_such_priced_module"],
            capture_output=True, text=True, cwd=str(_ROOT), check=False)
        assert done.returncode != 0, (
            "the child accepted a module that owns no slot, so its assertions "
            "are not running and the wiring test above proves nothing.\n"
            + done.stdout + done.stderr)
        assert "no_such_priced_module" in done.stderr, done.stderr

    def test_the_walk_reports_a_module_that_owns_no_slot(self, tmp_path):
        """The anti-vacuity that matters for the WALK, driven on a package
        this test builds.

        ``_uncontributing`` returning ``set()`` for the real package is
        exactly what a matcher that found nothing at all would also return,
        and that is the shape 2C shipped -- a discovery-by-prefix guard that
        matched nothing and passed forever.  So the comparison is driven here
        over a temporary package holding one module that registers and one
        that does not, and the check-less one must be named.

        Written into ``tmp_path``, never into ``src/``: a probe module created
        and deleted inside a globbed package went red 1 run in 8 under
        ``-n 16`` during Plan 3B and would have flaked every branch at once.
        """
        package = tmp_path / "postflight"
        package.mkdir()
        for stem in ("__init__", "fitting", "bystander"):
            (package / f"{stem}.py").touch()

        def _claimed(run):
            return ()

        _claimed.__module__ = "rheplicant.config.postflight.fitting"
        assert _uncontributing(package, {"C12": _claimed}) == {"bystander"}
        assert _uncontributing(package, {}) == {"bystander", "fitting"}

    def test_a_module_named_after_one_of_this_packages_own_names_is_refused(
            self, tmp_path):
        """The failure discovery has that a foot-import list does not.

        Importing a submodule SETS IT as an attribute of the package, so a
        ``postflight/priced.py`` would shadow the entry point ``document.py``
        calls and the hook would raise ``'module' object is not callable``.
        ``inflight/__init__.py`` records that measurement about its own
        ``axes`` and can only defend it with a comment about where the import
        block sits; discovery can refuse it outright, and does.

        Driven on a temporary directory, because the real package holds no
        such module and the raise at import time is otherwise unreachable.

        **The harmless stem is ``bystander`` and NOT a real check module's
        name.**  It used to be ``fitting``, and that went red the day Task 4
        landed: importing a submodule sets it as an attribute of its package,
        so a recomputed ``_reserved()`` holds ``fitting`` and this call
        refused it.  The fix is :data:`~rheplicant.config.postflight._RESERVED`
        -- the snapshot taken before the discovery loop -- and this stem is
        now one no task under this plan will ever create.
        """
        (tmp_path / "bystander.py").touch()
        assert _discoverable([str(tmp_path)]) == ("bystander",)
        (tmp_path / "priced.py").touch()
        with pytest.raises(ConfigError) as raised:
            _discoverable([str(tmp_path)])
        assert str(raised.value) == (
            "postflight/ holds ['priced'], and importing a submodule SETS IT "
            "as an attribute of its package -- so `from "
            "rheplicant.config.postflight import priced` would bind a MODULE "
            "and the hook in document.py would raise \"'module' object is not "
            "callable\". Rename the module; the reserved names are "
            f"{sorted(_RESERVED)}."
        )

    def test_the_reserved_set_is_a_snapshot_taken_before_discovery(self):
        """A check module's own stem must NEVER be reserved.

        Importing ``postflight.fitting`` sets ``postflight.fitting``, which IS
        a global of this module, so a ``_reserved()`` recomputed at any point
        after the loop contains it -- and ``_discoverable`` would then refuse
        `fitting.py`, `digitising.py` and `noise.py`, the three modules this
        plan exists to add.  Measured before :data:`_RESERVED` existed: a
        second ``_discoverable(__path__)`` raised ``postflight/ holds
        ['fitting']``.

        **Kills** ``reserved = _reserved()`` inside ``_discoverable``, which is
        what shipped, and which is invisible while the loop runs exactly once.
        """
        import rheplicant.config.postflight as package

        stems = tuple(sorted(name for _, name, _ in
                             pkgutil.iter_modules(package.__path__)))
        assert stems, "discovery found no module at all -- the pin is vacuous"
        assert set(stems).isdisjoint(_RESERVED), sorted(stems)
        assert set(stems) <= set(_reserved()), (
            "the recomputed set is expected to hold every imported stem -- "
            "that is the defect _RESERVED exists to freeze out")
        assert _discoverable(package.__path__) == stems

    def test_the_found_modules_come_back_sorted(self, tmp_path):
        """N1: ``sorted()`` inside :func:`_discoverable`, pinned on ORDER.

        A single-module fixture cannot see this -- a one-element tuple is
        sorted whether or not the call is there, which is exactly why
        removing ``sorted()`` from ``_discoverable`` left this module at
        **exit 0** before this test existed.  Written out of alphabetical
        order so a walk that happened to preserve filesystem or creation
        order cannot pass by luck.
        """
        for stem in ("zeta", "alpha", "mu"):
            (tmp_path / f"{stem}.py").touch()
        assert _discoverable([str(tmp_path)]) == ("alpha", "mu", "zeta")

    def test_the_reserved_names_are_derived_from_this_module_and_not_listed(
            self):
        """A hand-written list is a list that goes stale.

        Measured: the four-name literal this replaced held ``gates`` and
        ``run`` -- neither of which this package binds, so neither could
        shadow anything -- and MISSED ``sweep``.  A ``postflight/sweep.py``
        makes every ``load_document`` raise ``TypeError: 'module' object is
        not callable``, which is the exact failure the guard exists to
        prevent.
        """
        reserved = _reserved()
        assert {"priced", "register", "sweep", "binder", "CHECKS", "Priced",
                "PostCheck", "Report", "Finding", "Gate"} <= reserved
        assert "gates" not in reserved
        (tmp := pathlib.Path(tempfile.mkdtemp())) and None
        (tmp / "sweep.py").touch()
        with pytest.raises(ConfigError, match="holds \\['sweep'\\]"):
            _discoverable([str(tmp)])
