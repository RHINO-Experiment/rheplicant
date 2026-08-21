"""The verdict, the ledger, and the two things about them that can rot.

Two shapes are guarded here that a "does it work" suite would not reach:

* an assertion on PRESENCE where the defect is ATTRIBUTION -- ``"A30" in
  message`` passes when a tail names the wrong ``where`` and sends the reader
  to the wrong line.  The tail tests below pin the whole string.
* a decision shipped with no test, so reverting it stays green -- the
  ``stacklevel``, the tail's position (after, never before), and ``skip``'s
  absence from ``SEVERITIES`` are each one line of production code and each
  has a test here that fails when the line is reverted.
"""

import dataclasses
import json
import subprocess
import sys
import warnings

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import (
    REFUSE,
    REPORT,
    SEVERITIES,
    WARN,
    ConfigWarning,
    Finding,
    Report,
    refuse,
    report,
    warn,
)
from rheplicant.config.sections import exits
from rheplicant.config.sections.exit_support import (
    DEFERRED_CHECKS,
    EXECUTORS,
    PARSERS,
    PRE_EXECUTORS,
    register,
)
from rheplicant.config.sections.inference import _MODES

_REGISTRIES = (PARSERS, PRE_EXECUTORS, EXECUTORS, DEFERRED_CHECKS)

A = refuse("A2", "model.gian", "model: 'gian' is not a node (check A2).")
B = warn("A41", "resources.arrays.flat", "A literal shadows n_freq (check A41).")
C = refuse("A30", "model.noise", "model.noise: is stochastic (check A30).")
D = report("", "runs[1]", "runs[1]: reused an earlier product.")
E = refuse("A33", "inference.parameters.b", "b and g are both free (check A33).")


class TestTheThreeSeverities:
    """Kills: a fourth severity copied across from ``_MODES``; two
    constructors accidentally bound to one severity; a ``Finding`` accepting
    any string at all."""

    @pytest.mark.parametrize(
        ("make", "expected"),
        [(refuse, REFUSE), (warn, WARN), (report, REPORT)],
        ids=["refuse", "warn", "report"],
    )
    def test_each_constructor_carries_its_own_severity(self, make, expected):
        # Presence would pass with all three bound to REFUSE.  Identity is
        # what separates them.
        one = make("A1", "model", "a sentence.")
        assert one.severity == expected
        assert (one.check, one.where, one.message) == ("A1", "model", "a sentence.")

    def test_skip_is_a_mode_a_document_declares_and_never_a_severity(self):
        """The one design decision in this module, defended three ways.

        ``sections/inference.py:43`` really does parse four modes, so a reader
        who mirrors it gets a ``SEVERITIES`` of four and a ``Finding`` whose
        severity is ``"skip"`` -- which no consumer will ever route.
        """
        assert SEVERITIES == (REFUSE, WARN, REPORT)
        assert len(SEVERITIES) == 3
        assert "skip" not in SEVERITIES
        # ...and it is a real member of the modes, so the omission is a
        # decision rather than an oversight.
        assert "skip" in _MODES and len(_MODES) == 4
        assert set(SEVERITIES) < set(_MODES)

    def test_of_refuses_the_skip_token_by_name(self):
        """Kills ``of`` returning ``()`` for an unrecognised severity, which
        is what a caller who wrote ``report.of("skip")`` would otherwise get:
        an empty tuple that reads as "nothing was skipped".

        The ``isinstance`` line is the second half and kills a different
        implementation: ``of`` raising ``ConfigError``.  ``ConfigError``
        derives from ``ValueError``, so the ``pytest.raises`` above stays
        green under that mutation -- and the reader's ``except ConfigError``
        around ``load_document`` would then report a bug in the caller as a
        fault of their document, which is the same distinction
        ``Finding.__post_init__`` draws below.
        """
        with pytest.raises(ValueError, match="MODE a document declares") as caught:
            Report(findings=(A,)).of("skip")
        assert not isinstance(caught.value, ConfigError)

    def test_the_field_names_and_their_order_are_the_pinned_contract(self):
        """Twelve later tasks build findings against this signature.

        Kills a reordering -- ``message`` before ``where`` -- which every
        other test in this file survives, because they all construct by
        keyword.  The order is what a positional ``Finding(...)``, a ``repr``
        in a failure message and ``dataclasses.astuple`` all read.
        """
        assert [one.name for one in dataclasses.fields(Finding)] == [
            "check", "severity", "where", "message",
        ]

    def test_a_finding_refuses_a_severity_outside_the_three(self):
        with pytest.raises(ValueError, match=r"Finding\.severity is one of"):
            Finding(check="A1", severity="fatal", where="model", message="x.")

    def test_the_bad_severity_is_a_value_error_and_not_a_config_error(self):
        """Kills typing it ``ConfigError``: a caller's ``except ConfigError``
        around ``load_document`` would then report a BUG IN A CHECK as a fault
        of the user's document.

        ``ConfigError`` derives from ``ValueError`` (``config/errors.py``), so
        the ``pytest.raises(ValueError)`` above cannot tell them apart on its
        own and the ``isinstance`` line is the whole discrimination.
        """
        with pytest.raises(ValueError) as caught:
            Finding(check="A1", severity="fatal", where="model", message="x.")
        assert not isinstance(caught.value, ConfigError)

    def test_a_finding_is_frozen_and_hashable(self):
        """Kills dropping ``frozen=True``: one check could then edit another
        check's finding, and ``Report.checks()`` could not put them in a
        frozenset at all."""
        with pytest.raises(dataclasses.FrozenInstanceError):
            A.severity = WARN
        assert hash(A) == hash(refuse("A2", "model.gian", A.message))

    def test_both_dataclasses_carry_slots(self):
        """``slots=True`` is pinned by §3.1 and, unlike ``frozen=True``, no
        behaviour test can see it: dropping it leaves every other test in this
        file green.  A pass that builds thousands of findings pays for the
        per-instance ``__dict__``, and a slotted class also refuses the
        stray attribute a check might set on a finding it did not create.
        """
        assert not hasattr(Finding("A1", REFUSE, "model", "x."), "__dict__")
        assert not hasattr(Report(), "__dict__")

    @pytest.mark.parametrize(
        ("token", "severity"),
        [("refuse", REFUSE), ("warn", WARN), ("report", REPORT)],
        ids=["refuse", "warn", "report"],
    )
    def test_every_severity_route_matches_by_VALUE_and_not_by_identity(
            self, token, severity):
        """Kills ``one.severity is severity`` on any of the routes.

        Every ``Finding`` the rest of this file builds goes through
        ``refuse``/``warn``/``report``, which pass the module constant, so
        identity and equality agree everywhere and the mutant is green in
        every other test here.  It is not a theoretical hazard: a severity
        token read out of a parsed document is equal to :data:`REFUSE` and is
        not it, ``Finding.__post_init__`` admits it (membership tests by
        value), and under the mutant ``of(REFUSE)`` drops it -- so
        ``raise_if_refused`` raises nothing and a refused document runs.  One
        validation route closed, its twin open.

        All three severities and both named accessors, because they are
        SEPARATE call sites: measured, a mutant that leaves ``of`` alone and
        reimplements ``warnings()`` with ``is`` survives a version of this
        test that checks the refusal route only.
        """
        parsed = json.loads(json.dumps({"severity": token}))["severity"]
        assert parsed == severity
        assert parsed is not severity, (
            "this interpreter interned the parsed token, so this test can no "
            "longer tell `==` from `is` -- build the string another way"
        )
        one = Finding(check="A1", severity=parsed, where="model.x",
                      message="model.x: a parsed severity (check A1).")
        held = Report(findings=(one,))
        assert held.of(severity) == (one,)
        assert held.refusals() == ((one,) if severity == REFUSE else ())
        assert held.warnings() == ((one,) if severity == WARN else ())

    def test_a_parsed_refusal_still_stops_the_document(self):
        """The consequence of the test above, said as the user meets it.

        This is the failure the identity comparison actually produces: not a
        wrong tuple from an accessor, but a document that was refused and
        runs anyway.
        """
        parsed = json.loads('{"severity": "refuse"}')["severity"]
        one = Finding(check="A1", severity=parsed, where="model.x",
                      message="model.x: a parsed severity (check A1).")
        with pytest.raises(ConfigError, match="a parsed severity"):
            Report(findings=(one,)).raise_if_refused()


class TestTheReport:
    """Kills: ``of`` sorting or de-duplicating; ``checks()`` admitting the
    empty id; ``__bool__`` collapsing to ``bool(self.refusals())``."""

    def test_of_partitions_by_severity_and_keeps_declaration_order(self):
        # A, C and E are all refusals and are NOT in id order (A2, A30, A33
        # happens to be, so E is placed second here deliberately).
        held = Report(findings=(A, E, B, C, D))
        assert held.of(REFUSE) == (A, E, C)
        assert held.of(WARN) == (B,)
        assert held.of(REPORT) == (D,)

    def test_refusals_and_warnings_are_of_by_another_name(self):
        held = Report(findings=(A, E, B, C, D))
        assert held.refusals() == held.of(REFUSE)
        assert held.warnings() == held.of(WARN)

    def test_checks_names_the_ids_that_fired_and_drops_the_blank_one(self):
        """``D`` carries ``check=""``.  Kills
        ``frozenset(one.check for one in findings)``, which puts ``""`` in the
        set and makes ``"" in report.checks()`` true."""
        held = Report(findings=(A, E, B, C, D))
        assert held.checks() == frozenset({"A2", "A33", "A41", "A30"})
        assert "" not in held.checks()
        # `set(...) == frozenset(...)` is True in Python, so the equality
        # above is blind to `checks()` handing back a MUTABLE set -- which a
        # caller could then add to, editing one report's answer for every
        # other holder of it.  §3.1 pins the return type; this is the line
        # that enforces it.
        assert isinstance(held.checks(), frozenset)

    def test_truthiness_is_any_finding_and_not_any_refusal(self):
        """Kills ``__bool__`` returning ``bool(self.refusals())`` -- under
        which a document with three warnings reads as clean."""
        assert not Report()
        assert Report(findings=(B,))
        assert Report(findings=(B,)).refusals() == ()
        # `D` is `report` severity AND carries `check=""`, so this one line
        # kills two more readings that the `B` case above admits:
        # `bool(self.refusals() or self.warnings())`, under which a
        # report-only ledger -- exactly what Plan 3C's `mode: report` will
        # produce -- reads as clean; and `bool(self.checks())`, under which a
        # finding carrying no schema id does.
        assert Report(findings=(D,))

    def test_a_report_is_frozen_too(self):
        """Kills dropping ``frozen=True`` from ``Report``: the pass hands one
        object to a caller who could then append to another caller's ledger.
        """
        with pytest.raises(dataclasses.FrozenInstanceError):
            Report().findings = (A,)


class TestRaiseIfRefused:
    """Kills: raising on warnings; a prefix on the first message; the tail
    naming ids where it should name places, or naming them out of order; the
    tail appearing when there is nothing to tail; reading the first FINDING
    where it must read the first REFUSAL."""

    def test_a_clean_report_raises_nothing(self):
        assert Report().raise_if_refused() is None

    def test_a_refusal_attaches_the_supplied_cumulative_report(self):
        current = Report(findings=(A,))
        cumulative = Report(findings=(A, E))
        with pytest.raises(ConfigError) as caught:
            current.raise_if_refused(cumulative=cumulative)
        assert caught.value.report is cumulative

    def test_a_report_of_warnings_alone_raises_nothing(self):
        """Kills ``if self.findings: raise`` -- a document with one warning
        would stop loading."""
        assert Report(findings=(B, D)).raise_if_refused() is None

    def test_one_refusal_arrives_with_no_tail_at_all(self):
        with pytest.raises(ConfigError) as caught:
            Report(findings=(A, B, D)).raise_if_refused()
        assert str(caught.value) == A.message

    def test_the_first_REFUSAL_speaks_even_when_a_warning_came_first(self):
        """Kills ``self.findings[0].message``.

        Every other test in this class puts a refusal at index 0 of
        ``findings``, so the wrong reading is green in all of them.  Here ``B``
        is a warning and ``D`` a report, and the message a user must see is
        ``C``'s -- with the tail counting ``E`` alone, not ``B`` and ``D``.
        """
        with pytest.raises(ConfigError) as caught:
            Report(findings=(B, C, D, E)).raise_if_refused()
        assert str(caught.value) == (
            "model.noise: is stochastic (check A30)."
            "\n(This document has 1 more refusal, at inference.parameters.b.)"
        )

    def test_the_tail_counts_the_others_and_names_WHERE_they_are(self):
        """The whole string, because presence cannot tell ``where`` from
        ``check``: ``"A33" in message`` and ``"inference.parameters.b" in
        message`` are both true of a tail that names the wrong one, and a
        reader sent to ``A33`` instead of to a line does not know where to
        type."""
        with pytest.raises(ConfigError) as caught:
            Report(findings=(A, B, C, E)).raise_if_refused()
        assert str(caught.value) == (
            "model: 'gian' is not a node (check A2)."
            "\n(This document has 2 more refusals, at model.noise, "
            "inference.parameters.b.)"
        )

    def test_the_tail_is_singular_for_exactly_one_other(self):
        with pytest.raises(ConfigError) as caught:
            Report(findings=(A, C)).raise_if_refused()
        assert str(caught.value) == (
            "model: 'gian' is not a node (check A2)."
            "\n(This document has 1 more refusal, at model.noise.)"
        )

    @pytest.mark.parametrize("pattern", [
        # Three patterns lifted verbatim from assertions that exist today, so
        # the claim "a moved check keeps its pin" is tested against real pins
        # rather than against a pin written to pass.  Verified at the commit
        # that added this file, by `grep -n 'pytest.raises(ConfigError,
        # match=' tests/config/<module>`:
        r"\{ref:",          # test_config_section_model.py:131 -- and a regex,
                            # so `re.search` rather than `str.__contains__` is
                            # what the tail has to survive.
        "capability 4",     # test_config_document.py:82
        "schema_version",   # test_config_document.py:57
    ], ids=["ref", "capability", "version"])
    def test_a_pinned_pattern_still_matches_through_the_tail(self, pattern):
        """§2.3's "a moved check keeps its pin", tested against real pins.

        **What this kills**: a ``raise_if_refused`` that reproduces the first
        refusal as anything other than itself -- reformatting it, wrapping
        it, truncating it, escaping it, or weaving the tail INTO the sentence
        rather than after it.  The ``\\{ref:`` case is the sharp one: a tail
        spliced mid-message, or any re-escaping, separates the brace from the
        pattern while a plain substring check would not notice.

        **What it does NOT kill, measured**: prepending.  A ``"pre-flight: "``
        prefix keeps all three of these green, because ``pytest.raises(match=)``
        searches.  The prefix mutant exits 1 -- but on
        ``test_one_refusal_arrives_with_no_tail_at_all``,
        ``test_the_first_REFUSAL_speaks_even_when_a_warning_came_first``,
        ``test_the_tail_counts_the_others_and_names_WHERE_they_are`` and
        ``test_the_tail_is_singular_for_exactly_one_other``, which pin whole
        strings.  Those four are why the tail may not move the user's first
        line; this one is why the tail is SAFE to append at all.
        """
        first = refuse("A2", "model.x",
                       "model.x: a {ref: ...} needs capability 4 and "
                       "schema_version 1.")
        with pytest.raises(ConfigError, match=pattern):
            Report(findings=(first, C, E)).raise_if_refused()


class TestEmitWarnings:
    """Kills: emitting ``report``-severity findings; ``stacklevel=2``;
    ``class ConfigWarning(Warning)``."""

    def test_only_the_warn_severity_is_emitted(self):
        """``B`` is a warning, ``A`` a refusal and ``D`` a report.  Kills
        ``for one in self.findings`` -- three warnings instead of one, two of
        them things nobody asked to be interrupted about."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Report(findings=(A, B, D)).emit_warnings()
        assert len(caught) == 1
        assert caught[0].category is ConfigWarning
        assert str(caught[0].message) == B.message

    def test_every_warning_is_said_and_not_just_the_first(self):
        """Kills ``warnings.warn(self.warnings()[0].message, ...)`` and any
        ``break`` in the loop, neither of which
        ``test_only_the_warn_severity_is_emitted`` can see -- it has one
        warning in it."""
        second = warn("A42", "inference.twin", "The fit twin is simulated.")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            Report(findings=(B, A, second)).emit_warnings()
        assert [str(one.message) for one in caught] == [B.message, second.message]

    def test_the_warning_blames_the_caller_of_the_caller(self):
        """``stacklevel=3``, pinned by the frame AND the LINE.

        The chain ``emit_warnings`` aims at is ``user -> load_document ->
        emit_warnings``, so the line a user is shown must be the
        ``stands_for_the_user`` call in THIS file.  Two frames of the chain
        live here, which is why the filename alone decides nothing: measured
        by sweeping ``stacklevel`` 1..5 over a three-frame stand-in of this
        exact shape, the filename assertion passes for **2, 3 and 4 alike**
        (only 1, which blames the emitting module, and 5, which blames
        ``sys``, does the filename kill).  Only the line number separates 2,
        3 and 4, so the wanted line is computed from the stand-in's own
        ``__code__.co_firstlineno`` rather than written down, and moving this
        test in the file cannot break it.  The ``M13``/``M14`` mutants
        (``stacklevel=2`` and ``stacklevel=4``) are both red here.
        """
        def stands_for_load_document(held):
            held.emit_warnings()

        def stands_for_the_user(held):
            stands_for_load_document(held)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            stands_for_the_user(Report(findings=(B,)))
        wanted = stands_for_the_user.__code__.co_firstlineno + 1
        assert (caught[0].filename, caught[0].lineno) == (__file__, wanted)

    def test_a_config_warning_is_catchable_as_a_user_warning(self):
        """Kills ``class ConfigWarning(Warning)``: the six
        ``pytest.warns(UserWarning)`` sites in this repository, and every
        default warning filter, stop seeing this layer."""
        assert issubclass(ConfigWarning, UserWarning)
        with pytest.warns(UserWarning, match="shadows n_freq"):
            Report(findings=(B,)).emit_warnings()


class TestTheExecutorRegistryNoLongerAsserts:
    """The carry-forward ledger item, closed.

    Kills: reverting to ``assert``; a message that names the kind but not the
    two modules claiming it, which leaves the reader with no way to find the
    second registration; a refusal that still lets the second registration
    land.
    """

    @staticmethod
    def probe(run, built, *, results=None):
        return None

    def test_a_second_registration_is_refused_in_the_layers_own_voice(self):
        register("_probe_kind")(self.probe)
        try:
            # The transitional legacy adapter wraps a parse-less binding;
            # ``__wrapped__`` is the executor the caller registered.
            assert EXECUTORS["_probe_kind"].__wrapped__ is self.probe
            with pytest.raises(ConfigError, match="registered twice"):
                register("_probe_kind")(self.probe)
        finally:
            for registry in _REGISTRIES:
                registry.pop("_probe_kind", None)

    def test_the_first_registration_is_the_one_that_survives_the_refusal(self):
        """Kills assigning before raising.  A ``register`` that stored ``fn``
        and *then* refused would leave the table holding the loser of the
        argument, and every assertion about the refusal itself stays green.
        """
        def other(run, built, *, results=None):
            return None

        register("_probe_kind")(self.probe)
        try:
            with pytest.raises(ConfigError):
                register("_probe_kind")(other)
            assert EXECUTORS["_probe_kind"].__wrapped__ is self.probe
        finally:
            for registry in _REGISTRIES:
                registry.pop("_probe_kind", None)

    def test_the_refusal_names_both_claimants(self):
        """Presence of the kind is not enough: with two modules registering
        ``conjugate.wiener`` the reader needs to know WHICH two.

        The incumbent is a REAL executor from a REAL other module --
        ``rheplicant.config.sections.exits`` -- rather than a second function
        defined here, and that is the whole test.  Two functions defined in
        THIS module share one ``__module__``, so ``first.__module__ in
        message`` and ``other.__module__ in message`` collapse into the same
        assertion: measured, a message naming only the incumbent, only the
        challenger, or the same module twice passes all of them.  That is the
        "presence where the defect is attribution" shape this class claims to
        guard, reproduced inside the guard.

        The ``!=`` precondition is what stops it coming back silently, and
        the index comparison pins the ORDER -- incumbent first, challenger
        second -- because a reader chasing a collision needs to know which
        registration is already there and which one just lost.
        """
        def other(run, built, *, results=None):
            return None

        assert exits is not None  # importing exits is what populates EXECUTORS
        first = EXECUTORS["forward"]
        assert first.__module__ != other.__module__, (
            "this test cannot discriminate unless the two claimants live in "
            f"different modules; both are {other.__module__}"
        )
        register("_probe_kind")(first)
        try:
            with pytest.raises(ConfigError) as caught:
                register("_probe_kind")(other)
            message = str(caught.value)
            assert first.__module__ in message
            assert other.__module__ in message
            assert "_probe_kind" in message
            assert message.index(first.__module__) < message.index(other.__module__)
        finally:
            for registry in _REGISTRIES:
                registry.pop("_probe_kind", None)

    def test_the_refusal_survives_python_O(self):
        """The whole reason this changed.  Measured before the change: under
        ``-O`` the second registration WON, silently, and which executor a
        document got depended on import order."""
        source = (
            "from rheplicant.config.sections.exit_support import register\n"
            "def one(run, built, *, results=None): return 1\n"
            "def two(run, built, *, results=None): return 2\n"
            "register('_probe_kind')(one)\n"
            "try:\n"
            "    register('_probe_kind')(two)\n"
            "    print('SHADOWED')\n"
            "except Exception as error:\n"
            "    print(type(error).__name__)\n"
        )
        done = subprocess.run([sys.executable, "-O", "-c", source],
                              capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "ConfigError", done.stdout
