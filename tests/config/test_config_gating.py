"""The gate vocabulary, and the cross-product of a mode and a ``report:``.

Three shapes are guarded here that a "does it work" suite would not reach:

* **the cross-product as a TABLE rather than as three examples.**  §2.3 has
  twelve rows; the ``report:`` column's ``either``, ``—`` and ``ignored`` each
  stand for both values, so the table is EIGHTEEN cells and a twelve-entry
  parametrization silently drops three ``failed=yes`` cells.  The ``failed``
  column's ``—`` does not expand: a gate that does not run has no failure.
* **a hoist that COPIED rather than MOVED.**  ``sections/inference.py`` used
  to raise all seven of these sentences itself.  Two validators for one
  property drift, so the section is asserted to CALL the extraction (with the
  extraction replaced by a spy) and its source is asserted to carry none of
  the seven clauses any more.
* **a message pinned by a substring.**  ``TestChecks``'s four surviving
  ``match=`` pins include ``match="report"`` against a sentence that merely
  contains the word, and a fifth message has no assertion anywhere in the
  tree.  Every one of the seven is pinned below by equality on its WHOLE
  text, against a fixed input.
"""

import ast

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE, REPORT, WARN, Finding
from rheplicant.config.gating import (
    AUTO_SKIP,
    AUTO_SKIP_ID,
    CHECK_ID,
    CHECK_NAMES,
    DEFAULT_MODE,
    MODES,
    OFF,
    STATES,
    Gate,
    auto_skipped,
    check_gates,
    gates,
    verdict,
)
from tests.config.message_binding import assert_bound_once

#: A running gate on the one check that is on by default.  ``state`` and
#: ``record`` are overridden per cell; the rest is scenery.
LINEARITY = Gate(name="linearity", state="refuse", record=False, reason=None,
                 rtol=None)


def _only(section):
    """``check_gates`` over ``section``, asserting it said exactly one thing."""
    found = check_gates(section)
    assert len(found) == 1, found
    return found[0]


class TestTheVocabulary:
    def test_the_two_states_a_document_cannot_write(self):
        """Kills adding ``off`` or ``auto_skip`` to ``MODES`` "for symmetry",
        after which ``{mode: off}`` becomes writable and check A37 -- which
        reads the document's TEXT -- has a case it cannot distinguish from a
        default."""
        assert OFF not in MODES
        assert AUTO_SKIP not in MODES

    def test_the_six_states_are_the_four_plus_those_two(self):
        """The anti-vacuity partner: ``STATES`` could satisfy the test above
        by being short."""
        assert STATES == (*MODES, OFF, AUTO_SKIP)
        assert len(STATES) == 6

    def test_modes_is_a_tuple_and_check_names_is_a_frozenset(self):
        """Kills the container swap Plan 2D lost a task to: ``set(spec) -
        allowed`` needs a set, and the mode-enum refusal interpolates
        ``list(MODES)``, so the ORDER is in a pinned sentence."""
        assert isinstance(MODES, tuple)
        assert MODES == ("refuse", "warn", "report", "skip")
        assert isinstance(CHECK_NAMES, frozenset)

    def test_every_check_name_has_a_default(self):
        """Kills a fourth check name arriving with no default, under which
        ``gates()`` would ``KeyError`` inside ``load_document``."""
        assert set(DEFAULT_MODE) == CHECK_NAMES

    def test_only_linearity_is_on_by_default(self):
        """Kills D-C4 being quietly reversed.  This is the test that makes the
        cost table load-bearing rather than decorative: identifiability is a
        ``jacfwd`` plus a dense SVD and prior sensitivity is that plus two
        Newton solves, on every document that never asked."""
        assert DEFAULT_MODE == {"linearity": "refuse",
                                "identifiability": OFF,
                                "prior_sensitivity": OFF}

    def test_every_check_name_has_an_id_and_the_auto_skip_id_is_not_one(self):
        """Kills C14 being reused as a gate's own id -- see
        ``test_an_auto_skip_carries_C14_and_not_the_gate_s_own_id`` -- AND
        kills all three checks reporting under one shared id, under which
        schema §6's C19 row would be dead and D-14's ``C1..C19`` widening
        would be decoration rather than something any test exercises."""
        assert set(CHECK_ID) == CHECK_NAMES
        assert CHECK_ID == {"linearity": "C12", "identifiability": "C13",
                            "prior_sensitivity": "C19"}
        assert len(set(CHECK_ID.values())) == 3
        assert AUTO_SKIP_ID == "C14"
        assert AUTO_SKIP_ID not in set(CHECK_ID.values())


class TestGate:
    @pytest.mark.parametrize("state", STATES)
    def test_runs_is_false_for_every_non_running_state(self, state):
        """Kills ``state not in ("skip", OFF)`` -- the negation spelling, under
        which ``auto_skip`` RUNS.  No other test touches it, and
        :func:`verdict` leans on ``runs()`` for the whole bottom of §2.3's
        table."""
        expected = state in ("refuse", "warn", "report")
        assert Gate(name="linearity", state=state, record=False, reason=None,
                    rtol=None).runs() is expected

    def test_where_is_the_line_the_user_edits(self):
        assert LINEARITY.where() == "inference.checks.linearity"

    def test_check_gates_and_gate_where_agree_on_the_path(self):
        """One path, one spelling.  ``check_gates`` builds the same string for
        its findings, and a divergence would send a reader to a line that does
        not exist."""
        found = _only({"linearity": "banana"})
        assert found.where == LINEARITY.where()

    def test_auto_skipped_forces_record_false_and_returns_a_new_gate(self):
        """Kills a mutating implementation.  ``Gate`` is a ``NamedTuple``, so
        a mutation would have to rebind the mapping entry -- and the mapping is
        shared by every check in the pass."""
        asked = LINEARITY._replace(record=True)
        skipped = auto_skipped(asked, "no floating latent to differentiate")
        assert skipped == Gate(name="linearity", state=AUTO_SKIP, record=False,
                               reason="no floating latent to differentiate",
                               rtol=None)
        assert asked == Gate(name="linearity", state="refuse", record=True,
                             reason=None, rtol=None)
        assert skipped is not asked

    @pytest.mark.parametrize("empty_reason", ["", None])
    def test_auto_skipped_refuses_an_empty_reason(self, empty_reason):
        """Kills a caller-side bug that would otherwise ship silently.

        ``verdict``'s ``gate.reason or ""`` fallback turns a falsy reason
        into a REPORT finding whose message is the empty string -- an
        auto-skip that says nothing, dressed as a passing check.
        ``auto_skipped`` is called only from ``src/``, never from a
        document's own words, so this is a programmer error and a
        ``ValueError`` here is the right severity: raising turns the bug
        into a traceback at the call site instead of a blank line in a
        report nobody reads closely enough to notice.
        """
        with pytest.raises(ValueError, match="reason must not be empty"):
            auto_skipped(LINEARITY, empty_reason)


class TestGatesAlwaysReturnsThree:
    @pytest.mark.parametrize("section", [
        None,
        {},
        {"linearity": {"mode": "warn"}},
    ])
    def test_gates_always_returns_three(self, section):
        """Kills the ``{}``-shaped return.  A caller that has to write
        ``gates.get("linearity")`` writes ``.get("linearity", <its own
        default>)``, and then there are two default tables."""
        resolved = gates(section)
        assert set(resolved) == CHECK_NAMES
        assert len(resolved) == 3

    def test_an_undeclared_check_carries_its_default(self):
        resolved = gates(None)
        assert resolved["linearity"].state == "refuse"
        assert resolved["identifiability"].state == OFF
        assert resolved["prior_sensitivity"].state == OFF

    def test_a_declaration_wins_over_the_default(self):
        resolved = gates({"identifiability": {"mode": "warn", "rtol": 1e-6,
                                              "report": True}})
        assert resolved["identifiability"] == Gate(
            name="identifiability", state="warn", record=True, reason=None,
            rtol=1e-6)
        assert resolved["linearity"].state == "refuse"

    def test_rtol_is_identifiability_alone(self):
        """The twin: ``rtol`` is refused on the other two by the grammar, so
        the resolver must not carry one across from a document that got past
        it some other way.

        The document carries an ``rtol:`` on ``linearity`` -- illegal by
        :func:`check_gates`, but :func:`gates` never calls it (its own
        docstring: "for a section that passed check_gates"), so an input
        with no ``rtol:`` key at all would read ``None`` whether or not the
        ``if name == "identifiability"`` guard were there to read. This one
        DISCRIMINATES: a resolver that dropped the guard and read
        ``spec.get("rtol")`` unconditionally would return ``1e-8``, not
        ``None``.
        """
        assert gates({"linearity": {"mode": "warn",
                                    "rtol": 1e-8}})["linearity"].rtol is None
        assert gates(None)["identifiability"].rtol is None

    def test_a_non_running_state_never_carries_record(self):
        """``report:`` governs the numbers of a check that RAN.  Kills a
        resolver that hands ``record=True`` to a skipped gate, after which a
        caller reading ``gate.record`` has to ask ``gate.runs()`` first and one
        caller will forget."""
        resolved = gates({"linearity": {"mode": "skip", "reason": "campaign",
                                        "report": True}})
        assert resolved["linearity"].record is False
        assert resolved["linearity"].reason == "campaign"

    def test_a_reason_belongs_to_the_skip_state_alone(self):
        """The document carries a ``reason:`` on a ``warn`` gate -- illegal
        by :func:`check_gates`, but :func:`gates` never calls it, so an
        input with no ``reason:`` key at all would read ``None`` whether or
        not the ``if state == "skip"`` guard were there to read.  This one
        DISCRIMINATES: a resolver that dropped the guard and read
        ``spec.get("reason")`` unconditionally would return ``"x"``, not
        ``None``."""
        assert gates({"linearity": {"mode": "warn",
                                    "reason": "x"}})["linearity"].reason is None


#: §2.3's table, expanded.  ``(state, failed, record)`` -> ``(severity, check)``
#: or ``None``.  TWELVE rows, EIGHTEEN cells: every ``either``/``—``/``ignored``
#: in the ``report:`` column stands for both values, and the ``failed``
#: column's ``—`` does not, because a gate that does not run has no failure.
CROSS_PRODUCT = [
    # refuse: a failure stops the document, whatever report: says
    ("refuse", True, False, (REFUSE, "C12")),
    ("refuse", True, True, (REFUSE, "C12")),
    ("refuse", False, True, (REPORT, "C12")),
    ("refuse", False, False, None),
    # warn: a failure is a ConfigWarning
    ("warn", True, False, (WARN, "C12")),
    ("warn", True, True, (WARN, "C12")),
    ("warn", False, True, (REPORT, "C12")),
    ("warn", False, False, None),
    # report: a failure is recorded and nothing else
    ("report", True, False, (REPORT, "C12")),
    ("report", True, True, (REPORT, "C12")),
    ("report", False, True, (REPORT, "C12")),
    ("report", False, False, None),
    # skip and off: the check did not run, so there is nothing to say
    ("skip", False, False, None),
    ("skip", False, True, None),
    (OFF, False, False, None),
    (OFF, False, True, None),
    # auto_skip: ALWAYS reports, under C14, whatever report: says
    (AUTO_SKIP, False, False, (REPORT, "C14")),
    (AUTO_SKIP, False, True, (REPORT, "C14")),
]


class TestTheCrossProduct:
    def test_the_table_is_eighteen_cells_over_six_states(self):
        """Guard the guard.  A short parametrization is the failure mode here:
        it passes, and the cells it dropped are the ones nobody is checking."""
        assert len(CROSS_PRODUCT) == 18
        assert {cell[0] for cell in CROSS_PRODUCT} == set(STATES)

    @pytest.mark.parametrize("state, failed, record, expected", CROSS_PRODUCT)
    def test_the_cross_product(self, state, failed, record, expected):
        """Kills every one-cell mistake, and it is the test the whole plan
        leans on: a severity chosen inside each check is six chances to
        disagree about what ``mode: warn`` means."""
        gate = Gate(name="linearity", state=state, record=record,
                    reason="undefined here" if state == AUTO_SKIP else None,
                    rtol=None)
        found = verdict(gate, failed=failed,
                        where="inference.parameters.g",
                        message="the margin is 3.2e-01 (check C12).")
        if expected is None:
            assert found is None
        else:
            severity, check = expected
            assert isinstance(found, Finding)
            assert (found.severity, found.check) == (severity, check)
            # `where` is the SUBJECT's path, not the gate's own -- nothing
            # else in this file asserts it, and Tasks 4, 5 and 6 all depend
            # on `verdict` carrying the caller's `where` through unchanged.
            assert found.where == "inference.parameters.g"

    def test_a_failure_with_report_true_is_ONE_finding(self):
        """Kills returning a refusal AND a report, which would double-count in
        ``Report.checks()`` and make ``raise_if_refused``'s "N more refusals"
        tail wrong.  The message carried is the FAILURE sentence."""
        found = verdict(LINEARITY._replace(record=True), failed=True,
                        where="inference.parameters.g",
                        message="the margin is 3.2e-01 (check C12).")
        assert isinstance(found, Finding)
        assert found.severity == REFUSE
        assert found.message == "the margin is 3.2e-01 (check C12)."

    def test_a_pass_with_report_true_carries_the_numbers(self):
        found = verdict(LINEARITY._replace(record=True), failed=False,
                        where="inference.parameters.g",
                        message="margin 3.2e-01 over scales (1e-3, 1, 1e3).")
        assert isinstance(found, Finding)
        assert found.severity == REPORT
        assert found.message == "margin 3.2e-01 over scales (1e-3, 1, 1e3)."

    def test_an_auto_skip_reports_even_when_report_is_false(self):
        """Kills ``record`` gating the auto-skip path -- the silent-loss
        direction.  A check the user asked for and did not get must say so."""
        gate = auto_skipped(LINEARITY._replace(name="identifiability"),
                            "the latent 'a' is complex")
        assert gate.record is False
        found = verdict(gate, failed=False, where="inference.parameters.a",
                        message="unused")
        assert isinstance(found, Finding)
        assert found.severity == REPORT
        assert found.message == "the latent 'a' is complex"
        # Kills `verdict` substituting `gate.where()` for the caller's
        # `where` -- the two are DIFFERENT paths here on purpose, so a
        # substitution shows up as a wrong string, not a passing accident.
        assert found.where == "inference.parameters.a"
        assert found.where != gate.where()
        # Kills `if gate.state == AUTO_SKIP and not failed:` -- `failed`
        # must not change this branch's answer at all.  The mutant's own
        # failure mode is silent loss: a caller that happened to pass
        # `failed=True` would get `None` back instead of the C14 report.
        also_found = verdict(gate, failed=True, where="inference.parameters.a",
                             message="unused")
        assert also_found == found

    def test_an_auto_skip_carries_C14_and_not_the_gate_s_own_id(self):
        """Kills ``check=CHECK_ID[gate.name]`` on that branch, under which a
        user grepping the record for C14 finds nothing and ``Report.checks()``
        claims C13 fired when it did not."""
        gate = auto_skipped(LINEARITY._replace(name="identifiability"),
                            "the latent 'a' is complex")
        found = verdict(gate, failed=False, where="inference.parameters.a",
                        message="unused")
        assert found.check == "C14"
        assert found.check != CHECK_ID["identifiability"]

    def test_a_non_running_gate_is_none_even_when_the_caller_says_failed(self):
        """D-18(c).  Kills a ``verdict`` that reads ``failed`` before
        ``runs()``: the caller should not have called, and turning a caller's
        bug into the document's refusal is the substitution this layer
        exists to prevent."""
        for state in ("skip", OFF):
            gate = Gate(name="linearity", state=state, record=True,
                        reason="campaign" if state == "skip" else None,
                        rtol=None)
            assert verdict(gate, failed=True, where="inference.parameters.g",
                           message="boom") is None

    def test_each_check_reports_under_its_own_id(self):
        for name, check in CHECK_ID.items():
            found = verdict(Gate(name=name, state="report", record=True,
                                 reason=None, rtol=None),
                            failed=True, where="inference.parameters.g",
                            message="x")
            assert found.check == check


#: Every sentence ``check_gates`` says, against the input that produces it.
#: Pinned by equality on the WHOLE string: seven moved out of
#: ``sections/inference.py`` unchanged, and the eighth is this task's new cell.
MESSAGES = [
    (
        "banana",
        "inference.checks: is a mapping; got 'banana'.",
    ),
    (
        {"linearty": {"mode": "warn"}},
        "inference.checks.linearty: 'linearty' is not a check; v1 knows "
        "['identifiability', 'linearity', 'prior_sensitivity'].",
    ),
    (
        {"linearity": "banana"},
        "inference.checks.linearity: is a mapping with mode:; got 'banana'.",
    ),
    (
        {"linearity": {"mode": "warn", "rtol": 1e-8}},
        "inference.checks.linearity: a check: does not take ['rtol']; it "
        "takes ['mode', 'reason', 'report'].",
    ),
    (
        {"linearity": {"mode": "banana"}},
        "inference.checks.linearity.mode: is one of ['refuse', 'warn', "
        "'report', 'skip']; got 'banana'.",
    ),
    (
        {"linearity": {"mode": "skip"}},
        "inference.checks.linearity: mode: skip carries its own reason: "
        "(check A37) -- three unrelated skips sharing one sentence was v0's "
        "mistake.",
    ),
    (
        {"linearity": {"mode": "warn", "reason": "campaign"}},
        "inference.checks.linearity: reason: belongs to mode: skip alone.",
    ),
    (
        {"linearity": {"mode": "skip", "reason": "campaign", "report": True}},
        "inference.checks.linearity: mode: skip and report: true together "
        "ask to record the numbers of a check that will not run. Drop "
        "report:, or drop reason: and change mode: skip to mode: report so "
        "the check runs and has numbers to record (check A1).",
    ),
]


class TestEveryGrammarMessageIsPinnedWHOLE:
    def test_all_eight_decisions_have_a_pin(self):
        """Guard the guard: the docstring lists eight decisions, §3.2(g)
        printed five, and two of the seven that moved were pinned by NOTHING
        in the tree before this file."""
        assert len(MESSAGES) == 8

    @pytest.mark.parametrize("section, expected", MESSAGES)
    def test_the_message_is_this_and_no_other(self, section, expected):
        """Kills a re-wording.  ``TestChecks``' surviving ``match=`` pins are
        not sufficient evidence -- one of the four is ``match="report"``, which
        the mode-enum sentence satisfies with the bare word."""
        assert _only(section).message == expected

    @pytest.mark.parametrize("literal", [
        "mode: skip carries its own reason: (check A37) -- three unrelated "
        "skips sharing one sentence was v0's mistake.",
        "together ask to record the numbers of a check that will not run. "
        "Drop report:, or drop reason: and change mode: skip to mode: "
        "report so the check runs and has numbers to record (check A1).",
    ])
    def test_the_message_is_bound_in_one_module(self, literal):
        """Kills a hoist that copied rather than moved.

        Only these two of the eight clear ``message_binding``'s 40-character
        message floor once interpolations fold to a hole -- the others are
        34-39 characters and the walker cannot see them at all.  Their
        anti-copy partner is
        :meth:`TestTheSectionCallsTheExtraction.test_no_grammar_clause_survives_in_the_section`,
        which reads the section's own source.
        """
        assert_bound_once(literal)


class TestCheckGatesReturnsAndNeverRaises:
    @pytest.mark.parametrize("section", [
        {"linearity": {"mode": "warn"}},
        ["linearity"],
        None,
        {"linearity": "banana"},
        {"linearity": {"mode": "refuse", "rtol": 1e-8}},
    ])
    def test_check_gates_returns_and_never_raises(self, section):
        """Kills the section's ``raise`` being carried across, which would
        abort the pre-flight pass and hide every finding after it.

        The last case is the only shape that reaches the RAISING helper:
        ``check_unknown_keys`` refuses ``rtol`` on ``linearity`` by raising,
        and a ``check_gates`` that did not catch it would take the pass down
        with it.
        """
        found = check_gates(section)
        assert isinstance(found, tuple)
        assert all(isinstance(one, Finding) for one in found)

    def test_a_legal_section_says_nothing(self):
        assert check_gates(None) == ()
        assert check_gates({}) == ()
        assert check_gates({"identifiability": {"mode": "refuse",
                                                "rtol": 1e-8,
                                                "report": True}}) == ()

    #: One case per adjacent pair of ``decide()``'s decisions that a single
    #: input can actually tell apart -- i.e. a reorder of that pair would
    #: change what fires.  ``decide()``'s last three guards (mode not in
    #: MODES / mode == "skip" / mode != "skip") are pairwise mutually
    #: exclusive on ``mode``, so no input can make two of THOSE co-trigger
    #: and a reorder among them is an equivalent mutant -- no test could
    #: kill it without asserting something false.  The pair that spans one
    #: of those guards (A37's ``mode == "skip"`` against the new cell's,
    #: which shares that same precondition) is very much co-triggerable and
    #: is the one the reviewer's mutant actually moved.
    _ADJACENT_DECISION_PAIRS = [
        # (D2) the name is a known check, before (D3) the entry is a
        # mapping: swapped, this would report "is a mapping with mode:; got
        # 'banana'" instead of naming the unknown check.
        (
            {"linearty": "banana"},
            "A1",
            "inference.checks.linearty: 'linearty' is not a check; v1 knows "
            "['identifiability', 'linearity', 'prior_sensitivity'].",
        ),
        # (D3) the entry is a mapping, before (D4) the unknown-key sweep: a
        # dict-convertible non-Mapping (a list of pairs) would clear the key
        # sweep if D4 ran first, so this is the only input that tells the
        # two apart.
        (
            {"linearity": [("mode", "warn"), ("rtol", 1e-8)]},
            "A1",
            "inference.checks.linearity: is a mapping with mode:; got "
            "[('mode', 'warn'), ('rtol', 1e-08)].",
        ),
        # (D4) the unknown-key sweep, before (D5) mode is one of MODES: this
        # entry breaks both at once, and the key sweep must win -- the
        # original regression case.
        (
            {"linearity": {"mode": "banana", "rtol": 1e-8}},
            "A1",
            "inference.checks.linearity: a check: does not take ['rtol']; "
            "it takes ['mode', 'reason', 'report'].",
        ),
        # (D6) mode: skip carries its own reason: (check A37), before (D8)
        # mode: skip is not asked to report: numbers it cannot produce (the
        # new cell): both share the ``mode == "skip"`` precondition, so a
        # missing reason: AND report: true co-trigger them, and A37 must
        # win.  Swapped, this cell would report under A1 instead of A37 --
        # the schema slot Task 2 claims.
        (
            {"linearity": {"mode": "skip", "report": True}},
            "A37",
            "inference.checks.linearity: mode: skip carries its own "
            "reason: (check A37) -- three unrelated skips sharing one "
            "sentence was v0's mistake.",
        ),
    ]

    @pytest.mark.parametrize("section, check, message",
                             _ADJACENT_DECISION_PAIRS)
    def test_check_gates_yields_at_most_one_finding_per_entry(self, section,
                                                              check, message):
        """Kills a ``check_gates`` that collects every violation of an entry,
        AND a reorder of any co-triggerable pair of its decisions.

        ``found[0].message`` is what ``sections/inference.py`` hands back to
        ``ConfigError``, so it must be byte-identical to the sentence that
        section has raised since Plan 2B -- and all four ``match=`` pins in
        ``TestChecks`` stay green either way.  ``found[0].check`` matters as
        much as the message: one of these pairs is a swap that moves a
        finding from A37 to A1, which is invisible to a message-only
        assertion built from the wrong pin.
        """
        found = check_gates(section)
        assert len(found) == 1
        assert found[0].check == check
        assert found[0].message == message

    def test_two_bad_entries_are_two_findings_in_document_order(self):
        """The other half: "one per entry" is not "one per document".  A user
        with two broken entries sees both, which is the whole reason a check
        collects rather than raises."""
        found = check_gates({"linearty": {"mode": "warn"},
                             "linearity": "banana"})
        assert [one.where for one in found] == ["inference.checks.linearty",
                                                "inference.checks.linearity"]

    def test_the_grammar_findings_carry_A1_and_A37(self):
        """The slots ``preflight/gated.py`` claims for this grammar.  A37 is
        its own schema row -- "every ``checks.<name>.mode: skip`` carries its
        own ``reason:``" -- and everything else here is A1's key sweep."""
        by_id = {_only(section).check for section, _ in MESSAGES}
        assert by_id == {"A1", "A37"}
        assert _only({"linearity": {"mode": "skip"}}).check == "A37"

    def test_every_where_is_a_path_into_the_users_document(self):
        for section, _ in MESSAGES:
            assert _only(section).where.startswith("inference.checks")


class TestTheAdviceThisRefusalGives:
    """Take the document the new cell refuses, do what the sentence says, and
    assert the document then passes.  Plan 3A shipped three advice loops, one
    of which produced a clean report with the fault still present."""

    REFUSED = {"linearity": {"mode": "skip", "reason": "campaign",
                             "report": True}}

    def test_the_refusal_is_reached_at_all(self):
        assert _only(self.REFUSED).message.endswith("(check A1).")

    def test_dropping_report_makes_the_document_legal(self):
        advised = {"linearity": {"mode": "skip", "reason": "campaign"}}
        assert check_gates(advised) == ()
        assert gates(advised)["linearity"] == Gate(
            name="linearity", state="skip", record=False, reason="campaign",
            rtol=None)
        assert gates(advised)["linearity"].runs() is False

    def test_the_other_way_out_also_works(self):
        """The sentence names two escapes and both must be followable.

        Built by applying the second escape's own words -- "drop reason: and
        change mode: skip to mode: report" -- to ``REFUSED`` itself, rather
        than writing a fresh literal a reader has to trust matches the
        sentence.  A literal that only changed ``mode:`` and left
        ``reason:`` in place would land on a DIFFERENT refusal (``reason:``
        belongs to ``mode: skip`` alone) -- which is the same advice-loop
        shape this whole check exists to end, one clause over.
        """
        entry = dict(self.REFUSED["linearity"])
        del entry["reason"]
        entry["mode"] = "report"
        advised = {"linearity": entry}
        assert check_gates(advised) == ()
        assert gates(advised)["linearity"] == Gate(
            name="linearity", state="report", record=True, reason=None,
            rtol=None)


#: What :func:`_imported_modules` reports for an ``import_module()`` /
#: ``__import__()`` call whose argument it cannot resolve to a literal
#: string -- ``"a" + ".b"``, an f-string, a variable.  Silently skipping
#: such a call (the walk's previous behaviour) makes
#: ``importlib.import_module("rheplicant.config" + ".sections.inference")``
#: invisible to both :class:`TestTheImportBoundary` guards, because the
#: walk simply never adds anything for that call.  This module name cannot
#: legitimately appear in ``ALLOWED``, so any test that finds it in
#: ``reached`` must fail -- an unresolvable reach is refused, not ignored.
_UNRESOLVABLE_IMPORT = "<unresolvable import_module/__import__ argument>"


def _imported_modules(source: str) -> set[str]:
    """Every module ``source`` reaches for, under any spelling.

    ``import a.b``, ``from a.b import c`` (which reaches ``a.b.c`` when ``c``
    is a module), the relative forms, and ``importlib.import_module("a.b")`` --
    a walk that only read ``ast.Import`` would call a dynamic import invisible
    and report a clean module that drags the builders in at call time.  An
    ``import_module()``/``__import__()`` argument that is not itself a
    literal string -- built with ``+``, an f-string, a variable -- cannot be
    resolved by a static walk either, and is reported as
    :data:`_UNRESOLVABLE_IMPORT` rather than dropped: dropping it is exactly
    how a built argument evades this guard.
    """
    found: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            stem = "." * (node.level or 0) + (node.module or "")
            found.add(stem)
            found.update(f"{stem}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Call):
            call = node.func
            name = call.attr if isinstance(call, ast.Attribute) else getattr(
                call, "id", "")
            if name in ("import_module", "__import__"):
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and isinstance(
                            arg.value, str):
                        found.add(arg.value)
                    else:
                        found.add(_UNRESOLVABLE_IMPORT)
    return found


class TestTheImportBoundary:
    #: What ``gating.py`` may reach inside this package, and nothing else.
    #: ``from a.b import c`` cannot be told from ``import a.b.c`` by a static
    #: walk -- ``c`` may be a submodule or a symbol -- so both spellings are
    #: listed and the comparison is EXACT.  Any new package import fails here,
    #: which is the point: this module sits in front of the pre-flight scope
    #: guard and one convenience import would put the builders behind it.
    ALLOWED = {"rheplicant.config",
               "rheplicant.config.findings",
               "rheplicant.config.errors",
               "rheplicant.config.errors.ConfigError",
               "rheplicant.config.resources",
               "rheplicant.config.resources.check_unknown_keys"}

    def source(self) -> str:
        import pathlib

        import rheplicant.config.gating as module

        return pathlib.Path(module.__file__).read_text()

    def test_gating_imports_no_section_module(self):
        """Kills the transitive reach ``preflight/gated.py`` cannot afford:
        ``sections/inference.py`` imports ``jax.numpy`` and the builders, and
        the pre-flight pass runs in front of all of that on purpose."""
        reached = _imported_modules(self.source())
        offenders = [name for name in reached
                     if "sections" in name.split(".")]
        assert not offenders, offenders
        assert _UNRESOLVABLE_IMPORT not in reached, (
            "gating.py calls import_module()/__import__() with an argument "
            "this walk cannot resolve to a literal module name -- it may or "
            "may not reach sections/inference.py, and this guard refuses to "
            "guess.")

    @pytest.mark.parametrize("spelling", [
        "import rheplicant.config.sections.inference",
        "from rheplicant.config.sections import inference",
        "from rheplicant.config.sections.inference import _MODES",
        "from .sections import inference",
        "from ..config.sections.inference import _MODES",
        "import importlib\nx = importlib.import_module("
        "'rheplicant.config.sections.inference')",
        # The evasion: a BUILT argument, not a literal string.  A matcher
        # that only resolves `ast.Constant` arguments sees nothing here and
        # reports a clean module.
        "import importlib\nx = importlib.import_module("
        "'rheplicant.config' + '.sections.inference')",
    ])
    def test_the_matcher_sees_a_section_import_when_there_is_one(self,
                                                                spelling):
        """The anti-vacuity partner.  A matcher that saw nothing would pass the
        test above against a module that imports the whole builder stack.

        The last case names no module a literal ``"sections"`` substring
        could match -- ``_UNRESOLVABLE_IMPORT`` is what the matcher reports
        instead, and that counts as seeing it: this walk cannot prove such a
        call is safe, so it must not read as invisible.
        """
        reached = _imported_modules(spelling)
        assert (any("sections" in name.split(".") for name in reached)
                or _UNRESOLVABLE_IMPORT in reached), reached

    def test_gating_reaches_only_errors_findings_and_resources(self):
        """The positive form, which also kills a reach into ``passes``,
        ``document`` or ``preflight``.  ``resources`` is on the list for
        ``check_unknown_keys`` alone: §2.5 forbids a fourth hand-rolled
        unknown-key sweep, and that helper lives there."""
        reached = {name for name in _imported_modules(self.source())
                   if name.split(".")[:1] == ["rheplicant"]}
        assert reached <= self.ALLOWED, reached - self.ALLOWED


class TestTheSectionCallsTheExtraction:
    """``sections/inference.py`` keeps refusing on its own path, and keeps no
    second copy of the grammar."""

    def test_the_section_raises_the_first_findings_message_verbatim(self):
        """Kills the section keeping its own copy: with the extraction replaced
        by a spy, a section that still decided for itself would raise its own
        sentence instead of this one.

        The behavioural half of the one-binding rule.  ``message_binding``
        counts MODULES and cannot see a builder that restates a sentence the
        extraction it calls also carries -- so the property pinned here is that
        the section CALLS the extraction at all.
        """
        from rheplicant.config.findings import refuse
        from rheplicant.config.sections import inference as section

        spy = refuse("A1", "inference.checks.linearity", "the spy spoke.")
        original = section.check_gates
        try:
            section.check_gates = lambda _section: (spy,)
            with pytest.raises(ConfigError) as caught:
                section._checks({"linearity": {"mode": "warn"}})
        finally:
            section.check_gates = original
        assert str(caught.value) == "the spy spoke."

    @pytest.mark.parametrize("clause", [
        "inference.checks: is a mapping",
        "is not a check; v1 knows",
        "is a mapping with mode:",
        ".mode: is one of ",
        "mode: skip carries its own reason",
        "reason: belongs to mode: skip",
        "ask to record the numbers of a check",
    ])
    def test_no_grammar_clause_survives_in_the_section(self, clause):
        """The anti-copy half for the six sentences ``message_binding``'s
        40-character floor cannot see."""
        import pathlib

        from rheplicant.config.sections import inference as section

        source = pathlib.Path(section.__file__).read_text()
        assert clause not in source, clause

    def test_the_section_still_records_what_the_document_declared(self):
        """``CheckSpec`` is the DECLARATION and never the effective mode: a
        record that said ``mode: refuse`` for a check nobody configured would
        be a lie about the document, and Plan 4 needs both."""
        from rheplicant.config.sections.inference import _checks

        parsed = _checks({"identifiability": {"mode": "warn", "rtol": 1e-8,
                                              "report": True}})
        assert set(parsed) == {"identifiability"}
        assert parsed["identifiability"].mode == "warn"
        assert parsed["identifiability"].report is True
        assert parsed["identifiability"].rtol == 1e-8
        # ... while the GATE for the same document has all three.
        assert set(gates({"identifiability": {"mode": "warn"}})) == CHECK_NAMES

    @pytest.mark.parametrize("section, expected", MESSAGES)
    def test_the_section_and_the_extraction_say_the_same_thing(self, section,
                                                              expected):
        """The twin, closed by measurement rather than by reading: whatever
        ``check_gates`` refuses, ``_checks`` raises, character for
        character."""
        from rheplicant.config.sections.inference import _checks

        with pytest.raises(ConfigError) as caught:
            _checks(section)
        assert str(caught.value) == expected

    def test_the_section_still_exports_the_two_moved_names(self):
        """``tests/config/test_config_findings.py`` imports ``_MODES`` from the
        section to argue why there are three severities and four modes.  Kills
        the move deleting the name rather than rebinding it."""
        from rheplicant.config.sections.inference import _CHECK_NAMES, _MODES

        assert _MODES is MODES
        assert _CHECK_NAMES is CHECK_NAMES
