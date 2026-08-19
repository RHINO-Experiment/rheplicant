"""``preflight/gated.py`` -- A1.checks, A37 and C18.kind.

**C15 is NOT here.**  It is registered with ``@register_axes`` and its tests
live in ``tests/config/test_inflight_noise_waves.py``; ``preflight_helpers``'
``findings``/``ids``/``refusals``/``only`` all call ``preflight(document)`` --
the TEXT pass alone -- so a C15 test written with ``only(...)`` in this module
would pass against an empty implementation.

**Every message is pinned by equality on its WHOLE text.**  There are 442
``pytest.raises(ConfigError, match=...)`` assertions under ``tests/config/``
and ZERO anchored with ``^`` or ``$``, so a substring pin cannot see a
re-word; Plan 3A's surviving mutants lived almost entirely inside refusal
text.  Seven of the eight sentences below MOVED from
``sections/inference.py`` and two of the seven were pinned by **nothing** in
the tree before this module.

**Set assertions are subset-shaped**, scoped to :data:`MINE`: sibling
branches land checks that run on these same documents, and an ``== ()`` would
go red on a correct merge.
"""

import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE, WARN
from rheplicant.config.gating import check_gates
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight.gated import (
    _DRAWING_TYPES,
    _T2C_GENERATING_TWIN,
    _T2C_NOT_FITTING,
    _checks_grammar,
    _sigma_families,
)
from rheplicant.config.sections.inference import build_inference
from tests.config.exit_helpers import (
    FROZEN,
    HOMOSCEDASTIC,
    MODEL_NOISE,
    RADIOMETER,
)
from tests.config.preflight_helpers import (
    BASE_MODEL,
    CHECKS_SKIP,
    RADIOMETER_DRAWN,
    UNREADABLE_BEAM,
    findings,
    ids,
    only,
    preflight_document,
    repatch,
)

#: The ids THIS module is about.
MINE = frozenset({"A1", "A37", "C18"})


def mine(document) -> frozenset:
    return ids(document) & MINE


def checks_document(checks, **patch):
    """The base document with ``inference.checks:`` set to ``checks``."""
    return preflight_document(inference={"checks": checks}, **patch)


#: "this keyword was not passed", so that ``observed=None`` can mean REMOVE
#: the block -- which is a case C18.kind stands down on and a default of
#: ``None`` could not express.
_UNSET = object()


def sigma_document(drawn, weighed, *, observed=_UNSET, runs=_UNSET,
                   model=_UNSET):
    """A document that DRAWS with ``drawn`` and WEIGHS with ``weighed``.

    ``observed`` is left as the base document's unless a caller says
    otherwise, and the base declares ``from: simulation`` with ``twin: full``
    -- the two conditions C18.kind needs before the two words mean anything.

    **Built through** ``preflight_helpers.repatch`` **and never by assigning
    into the document.**  A depth-1 write of a non-empty ``inference:`` block
    over a delegated one is exactly what
    ``test_config_fixture_contract._rolls_its_own``'s route B catches, and it
    catches it because that write throws away ``exit_helpers._repaired``'s
    ``twin: {without: [noise]}`` -- which would make ``built.twin`` and
    ``built.inference.fit_twin`` the same object for every test in this
    module.  ``repatch`` is the helper that expresses a whole-section
    replacement, including a removal, which a merge cannot.
    """
    document = preflight_document()
    block = dict(document["inference"])
    block["noise"] = weighed
    if observed is not _UNSET:
        if observed is None:
            block.pop("observed", None)
        else:
            block["observed"] = observed
    patch = {"inference": block}
    if model is not _UNSET:
        patch["model"] = model
    elif drawn is not None:
        patch["model"] = {**BASE_MODEL, "noise": drawn}
    if runs is not _UNSET:
        patch["runs"] = runs
    return repatch(document, **patch)


def observed_as(document, **keys):
    """The document's primary ``observed`` record with ``keys`` applied.

    A key whose value is ``None`` is REMOVED, so a test can say "this document
    does not spell its twin out" -- which is the case
    ``sections/observed.py``'s ``full`` default is about.
    """
    record = dict(document["inference"]["observed"])
    for name, value in keys.items():
        if value is None:
            record.pop(name, None)
        else:
            record[name] = value
    return record


# --- the seven moved messages, whole ---------------------------------------
#
# Measured at `e0e024a` in `sections/inference.py`, moved to `gating.py` by
# Task 1, and reached from here through the pre-flight pass.  §3.2(g) prints
# five of them; the two it omits (`_IS_A_MAPPING` and `_ENTRY_IS_A_MAPPING`)
# were pinned by NOTHING in the tree -- `grep -rn "is a mapping with mode"
# tests/` had no match -- so a move that dropped or re-worded either would
# have been silent.
#
# **Defined ABOVE `TestItRunsBeforeTheBeam` (Plan 3C fix round, MINOR 3):**
# that class's own phase test is now parametrised over :data:`MOVED`, and a
# decorator evaluates its arguments at class-body execution time -- moved
# below this block, `MOVED` would not exist yet and collection would fail
# with `NameError` before a single test ran.

_IS_A_MAPPING = "inference.checks: is a mapping; got 'banana'."
_UNKNOWN_NAME = (
    "inference.checks.linearty: 'linearty' is not a check; v1 knows "
    "['identifiability', 'linearity', 'prior_sensitivity'].")
_ENTRY_IS_A_MAPPING = (
    "inference.checks.linearity: is a mapping with mode:; got 'banana'.")
_UNKNOWN_KEY = (
    "inference.checks.linearity: a check: does not take ['rtol']; it takes "
    "['mode', 'reason', 'report'].")
_MODE_ENUM = (
    "inference.checks.linearity.mode: is one of ['refuse', 'warn', 'report', "
    "'skip']; got 'banana'.")
A37_SKIP_WITHOUT_REASON = (
    "inference.checks.linearity: mode: skip carries its own reason: (check "
    "A37) -- three unrelated skips sharing one sentence was v0's mistake.")
_REASON_WITHOUT_SKIP = (
    "inference.checks.linearity: reason: belongs to mode: skip alone.")

#: The one NEW cell (§2.3), which moved from nowhere.
_SKIP_WITH_REPORT = (
    "inference.checks.linearity: mode: skip and report: true together ask to "
    "record the numbers of a check that will not run. Drop report:, or drop "
    "reason: and change mode: skip to mode: report so the check runs and has "
    "numbers to record (check A1).")

#: ``(the section a document writes, the sentence it earns, the id)``.  SEVEN
#: moved rows and one new one.
MOVED = [
    ("banana", _IS_A_MAPPING, "A1"),
    ({"linearty": {"mode": "warn"}}, _UNKNOWN_NAME, "A1"),
    ({"linearity": "banana"}, _ENTRY_IS_A_MAPPING, "A1"),
    ({"linearity": {"mode": "warn", "rtol": 1e-8}}, _UNKNOWN_KEY, "A1"),
    ({"linearity": {"mode": "banana"}}, _MODE_ENUM, "A1"),
    ({"linearity": {"mode": "skip"}}, A37_SKIP_WITHOUT_REASON, "A37"),
    ({"linearity": {"mode": "warn", "reason": "x"}}, _REASON_WITHOUT_SKIP,
     "A1"),
    ({"linearity": {"mode": "skip", "reason": "x", "report": True}},
     _SKIP_WITH_REPORT, "A1"),
]


# --- the phase assertion, which is why this task exists ---------------------


class TestItRunsBeforeTheBeam:
    """Measured before this task, on documents carrying an unreadable beam
    beside one ``checks:`` fault each: **the beam won seven times out of
    seven**.  ``sections/inference.py::_checks`` is reached only after
    ``build_resources``, which is 90.9 % of ``load_document``'s wall time, so
    a user who typed ``mode: sipk`` was told about a file they had not
    touched."""

    @pytest.mark.parametrize("section, message, check", MOVED,
                             ids=[one[2] + "-" + str(index)
                                  for index, one in enumerate(MOVED)])
    def test_a_checks_fault_beats_an_unreadable_beam(self, section, message,
                                                      check):
        """**Kills the whole task being registered and never reached.**

        MINOR 3 (Plan 3C fix round): parametrised over every one of
        :data:`MOVED`'s eight rows, not the one (``A37_SKIP_WITHOUT_REASON``
        alone).  The reviewer measured that the phase property -- a
        ``checks:`` fault beats an unreadable beam -- was driven through
        ``UNREADABLE_BEAM`` for exactly one of the eight rows this pass can
        raise; the coverage this class's own docstring claims (**"the beam
        won seven times out of seven"** before the task, measured against all
        seven of the sentences that then moved here) was never repeated
        against the fix.  **9 for 9** beats ``UNREADABLE_BEAM`` -- each row's
        own sentence is the one raised, and the beam's file name
        (``no_such_beam``) appears in NONE of them.
        """
        document = checks_document(section, resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert str(caught.value) == message
        assert "no_such_beam" not in str(caught.value)

    def test_the_beam_still_refuses_a_document_with_legal_checks(self):
        """The anti-vacuity partner.  Without it, a ``UNREADABLE_BEAM`` that
        had quietly stopped being unreadable would make the test above pass
        for the wrong reason."""
        document = checks_document(CHECKS_SKIP, resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError, match="no_such_beam"):
            load_document(document)


# --- the slots --------------------------------------------------------------


class TestTheSlots:
    def test_one_function_claims_both_a1_checks_and_a37(self):
        """VARIADIC, not stacked.  ``@register("A20") @register("A21")``
        applies BOTTOM-UP, so the stacked form inserts the last decorator's id
        first and insertion order -- which IS run order, which is which
        refusal the user reads -- comes out reversed.  ``sweep``'s
        de-duplication by function IDENTITY is what stops one function bound
        to two slots running twice."""
        assert CHECKS["A1.checks"] is _checks_grammar
        assert CHECKS["A37"] is _checks_grammar

    def test_a1_checks_is_inserted_before_a37(self):
        """The property a stacked pair silently reverses.  **Kills** two
        single-id decorators, which no ``in CHECKS`` assertion can see."""
        slots = list(CHECKS)
        assert slots.index("A1.checks") < slots.index("A37")

    def test_c18_kind_is_the_dotted_slot_and_c18_is_free(self):
        """Two functions, two slots, ONE bare id: the numeric half needs the
        built operators and claims the bare ``C18``.  A task that claimed
        ``C18`` here would take the slot the later one needs, and ``binder``
        refuses a double registration."""
        assert CHECKS["C18.kind"] is _sigma_families
        assert "C18" not in CHECKS

    def test_the_findings_carry_the_bare_ids(self):
        """§3.2(a)'s other half: the SLOT may be dotted, ``Finding.check``
        never is.  ``Report.checks()`` is what a user greps."""
        assert only(checks_document({"linearty": {"mode": "warn"}}),
                    "A1").check == "A1"
        assert only(checks_document({"linearity": {"mode": "skip"}}),
                    "A37").check == "A37"
        assert only(sigma_document(MODEL_NOISE, RADIOMETER),
                    "C18").check == "C18"

    def test_a_check_is_never_run_twice_for_its_two_slots(self):
        """``sweep`` de-duplicates by ``id(fn)``.  **Kills** the guard being
        dropped, whose only symptom is every A1.checks finding appearing
        twice -- which ``only`` sees and an ``in`` assertion does not."""
        only(checks_document({"linearty": {"mode": "warn"}}), "A1")


# --- the seven moved messages, whole ----------------------------------------
#
# The data (the seven-plus-one message constants and :data:`MOVED`) moved
# ABOVE `TestItRunsBeforeTheBeam`, for the reason given there.


class TestTheMovedMessages:
    @pytest.mark.parametrize("section, message, check",
                             MOVED, ids=[one[2] + "-" + str(index)
                                         for index, one in enumerate(MOVED)])
    def test_each_sentence_arrives_through_the_pass_whole(self, section,
                                                          message, check):
        """**Kills** a re-word anywhere in any of the eight."""
        assert only(checks_document(section), check).message == message

    @pytest.mark.parametrize("section, message, check", MOVED,
                             ids=[str(index) for index in range(len(MOVED))])
    def test_the_section_still_refuses_on_its_own_path(self, section, message,
                                                       check):
        """``build_inference`` directly, with no ``load_document`` in front of
        it.

        **Kills** the section's raise being deleted in favour of the pass,
        after which a caller that reaches ``build_inference`` -- every test in
        ``test_config_section_inference.py``, and every future programmatic
        caller -- gets no refusal at all.  §3.2(d): ``_checks`` raises
        ``found[0].message`` verbatim, so the two sentences are the SAME
        sentence and this asserts it by equality rather than trusting it.
        """
        with pytest.raises(ConfigError) as caught:
            build_inference({"checks": section}, twin=None, state=None,
                            observation=None, context=None)
        assert str(caught.value) == message

    def test_check_gates_yields_at_most_one_finding_per_entry(self):
        """So ``found[0]`` is byte-identical to what ``_checks`` raises.  An
        entry wrong in three ways earns the FIRST decision's sentence, in the
        order ``check_gates``' docstring lists -- **kills** a rewrite that
        collected every fault and changed which one a user reads."""
        section = {"linearty": {"mode": "banana", "rtol": 1e-8,
                                "reason": "x"}}
        assert len(check_gates(section)) == 1
        assert check_gates(section)[0].message == _UNKNOWN_NAME

    def test_a_legal_checks_block_earns_nothing(self):
        """The anti-vacuity partner of every row above."""
        assert mine(checks_document(CHECKS_SKIP)) == frozenset()
        assert mine(checks_document({
            "identifiability": {"mode": "refuse", "rtol": 1e-8,
                                "report": True},
            "linearity": {"mode": "warn"},
            "prior_sensitivity": {"mode": "report"}})) == frozenset()

    def test_an_absent_or_unreadable_inference_section_is_silent(self):
        """``inference: 7`` is ``build_inference``'s own refusal, with the
        value the user wrote; answering here would pre-empt it."""
        assert mine(preflight_document(inference=None)) == frozenset()
        assert mine(preflight_document(inference=7)) == frozenset()
        assert mine(preflight_document(inference={"checks": None})) \
            == frozenset()


class TestTheNewCell:
    def test_skip_with_report_true_is_refused(self):
        assert only(checks_document({"linearity": {
            "mode": "skip", "reason": "x", "report": True}}),
            "A1").message == _SKIP_WITH_REPORT

    def test_skip_without_report_is_not(self):
        """**Kills** the cell being written as "refuse any ``report:`` on a
        ``skip``" rather than "refuse ``report: true``" -- measured,
        ``report: false`` beside a skip is coherent and must pass."""
        assert mine(checks_document({"linearity": {
            "mode": "skip", "reason": "x", "report": False}})) == frozenset()
        assert mine(checks_document(CHECKS_SKIP)) == frozenset()

    def test_report_true_on_a_running_mode_is_not_refused(self):
        """The other anti-vacuity direction: ``report: true`` is the whole
        point of the key, and only ``skip`` cannot honour it."""
        for mode in ("refuse", "warn", "report"):
            assert mine(checks_document({
                "linearity": {"mode": mode, "report": True}})) == frozenset()


# --- C18.kind ---------------------------------------------------------------

#: ``(model.noise, inference.noise, does it refuse)`` -- the two refusing rows
#: AND the two agreeing ones.  Parametrised over all four because a table with
#: one row transposed passes every single-cell test written about the other.
FAMILIES = [
    (MODEL_NOISE, HOMOSCEDASTIC, False),
    (RADIOMETER_DRAWN, RADIOMETER, False),
    (RADIOMETER_DRAWN, FROZEN, False),
    (MODEL_NOISE, RADIOMETER, True),
    (MODEL_NOISE, FROZEN, True),
    (RADIOMETER_DRAWN, HOMOSCEDASTIC, True),
]


class TestSigmaFamilies:
    @pytest.mark.parametrize("drawn, weighed, refuses", FAMILIES,
                             ids=[f"{one[0]['type']}-{one[1]['kind']}"
                                  for one in FAMILIES])
    def test_C18_kind_refuses_each_cross_family_cell(self, drawn, weighed,
                                                     refuses):
        """**Kills** a table with one row transposed, which no single-cell
        test can see."""
        document = sigma_document(drawn, weighed)
        if refuses:
            found = only(document, "C18")
            assert found.severity == REFUSE
            assert found.where == "model.noise"
        else:
            assert mine(document) == frozenset()

    def test_the_two_rows_partition_the_four_kinds(self):
        """``_DRAWING_TYPES``' values are DISJOINT and cover every kind
        ``inference.noise`` can weigh with apart from ``none``.  **Kills** a
        kind being added to both rows, after which nothing cross-family
        refuses at all."""
        from rheplicant.config.sections.noise import _KIND_KEYS

        weighing = frozenset().union(*_DRAWING_TYPES.values())
        assert weighing == frozenset(_KIND_KEYS) - frozenset({"none"})
        assert len(weighing) == sum(len(one) for one in
                                    _DRAWING_TYPES.values())

    def test_the_message_names_the_gate_and_the_escape(self):
        """§3.2(i): a refusal that does not name its own off switch is a
        refusal a user cannot act on.  Both escapes are named -- change what
        you draw with, or change what you weigh with -- and each is followed
        literally by an advice-loop test below."""
        found = only(sigma_document(MODEL_NOISE, RADIOMETER), "C18")
        assert "model.noise.type: RadiometerNoiseOperator" in found.message
        assert "inference.noise.kind: homoscedastic" in found.message
        assert found.message.endswith("(check C18).")

    def test_the_symmetric_message_names_the_other_two_escapes(self):
        """The transposed row's own sentence.  **Kills** one row's advice
        being copied onto the other, which would send a
        ``RadiometerNoiseOperator`` user to write the type they already
        have."""
        found = only(sigma_document(RADIOMETER_DRAWN, HOMOSCEDASTIC), "C18")
        assert "model.noise.type: NoiseOperator" in found.message
        assert "inference.noise.kind: radiometer" in found.message


class TestSigmaFamiliesStandsDown:
    def test_C18_kind_stands_down_on_observed_from_file(self, tmp_path):
        """§2.6 item 6: the data came from a file, the twin drew nothing, and
        there is no second sigma to disagree with.  **Kills** the scoping
        decision being dropped, after which every ingested document is
        refused."""
        document = sigma_document(
            MODEL_NOISE, RADIOMETER,
            observed={"file": {"path": str(tmp_path / "d.npy")}})
        assert mine(document) == frozenset()

    def test_C18_kind_stands_down_with_no_observed_at_all(self):
        assert mine(sigma_document(MODEL_NOISE, RADIOMETER,
                                   observed=None)) == frozenset()

    def test_C18_kind_stands_down_on_a_fit_twin_observation(self):
        """**Scoped exactly as the numeric C18 is** (D-10).  On the
        ``twin: fit`` family ``inference.twin.without: [noise]`` has taken the
        drawing operator out of the tree that produced the data, so
        ``model.noise.type`` describes something that provably never touched
        it and a refusal naming it would be a false claim about the document.

        **Measured**: this is what takes the REFUSE census over the thirteen
        shipped ``*_document`` builders from 3 to 0 --
        ``exit_helpers.gls_document``, ``gls_pair_document`` and
        ``fanned_document`` are all ``NoiseOperator`` x radiometer AND all
        ``twin: fit``.
        """
        base = sigma_document(MODEL_NOISE, RADIOMETER)
        document = sigma_document(MODEL_NOISE, RADIOMETER,
                                  observed=observed_as(base, twin="fit"))
        assert mine(document) == frozenset()

    def test_twin_fit_alone_is_not_enough_with_zero_declared_latents(self):
        """MAJOR 5: ``twin: fit`` is not SUFFICIENT for the stand-down above --
        it stands down only when something has actually taken the drawing
        operator out of the fit twin.  ``refuse_stochastic_stages`` is what
        ordinarily forces that (any document that declares a latent), and
        with ZERO latents it never runs at all, so a document can spell
        ``twin: fit`` with no ``inference.twin:`` repair whatsoever -- and
        its fit twin is then the SAME object as the full twin, still
        carrying the draw.

        Before this fix: silent (``preflight ids: []``), and
        ``load_document`` LOADS the document -- ``NoiseOperator`` drew the
        data and ``radiometer`` weighed it, with nobody told.  After it, C18
        reaches this document exactly as it would a ``twin: full`` one, and
        ``load_document`` refuses with C18's own sentence rather than
        returning a silently miscalibrated fit.
        """
        document = repatch(
            preflight_document(),
            inference={"parameters": {}, "noise": RADIOMETER,
                      "observed": {"from": "simulation", "twin": "fit"}})
        found = only(document, "C18")
        assert found.severity == REFUSE
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert str(caught.value) == found.message

    def test_full_is_the_default_so_a_silent_document_is_still_checked(self):
        """The partner: ``sections/observed.py`` defaults ``twin:`` to
        ``full``, so a document that says nothing keeps the check.  **Kills**
        the scoping being written as ``record.get("twin") == "full"``, which
        answers ``None`` for every document that does not spell it out."""
        assert _T2C_GENERATING_TWIN == "full"
        base = sigma_document(MODEL_NOISE, RADIOMETER)
        document = sigma_document(MODEL_NOISE, RADIOMETER,
                                  observed=observed_as(base, twin=None))
        assert only(document, "C18").severity == REFUSE

    @pytest.mark.parametrize("node", [
        {"compose": [{"type": "NoiseOperator",
                      "sigma": {"value": 0.5, "unit": "K"}}]},
        # A `compose:` block that ALSO carries a `type:`.  Without the
        # explicit `"compose" in spec` guard this one reads as a single
        # drawing operator, because `spec.get("type")` answers happily --
        # measured, the other five cells all stand down on the type lookup
        # alone and leave that guard unexercised.
        {"compose": [{"type": "NoiseOperator",
                      "sigma": {"value": 0.5, "unit": "K"}}],
         "type": "NoiseOperator"},
        [{"type": "NoiseOperator", "sigma": {"value": 0.5, "unit": "K"}}],
        7, None, "NoiseOperator", {"python": "rheplicant:NoiseOperator"}])
    def test_C18_kind_stands_down_on_a_composed_noise_node(self, node):
        """**Kills** a bare ``model["noise"]["type"]``, which raises
        ``TypeError`` on a list -- and a check that raises aborts the pass and
        hides every later finding.  A composed node has no single drawing
        operator, and a check that guessed would refuse a document it cannot
        read."""
        document = sigma_document(MODEL_NOISE, RADIOMETER,
                                  model={**BASE_MODEL, "noise": node})
        assert mine(document) == frozenset()

    @pytest.mark.parametrize("model", [
        {"kind": "pipeline", "stages": []},
        # A pipeline model carrying a `noise:` key.  Without the
        # `kind: graph` guard this one is read as a graph node and refused --
        # measured, the bare pipeline above has no `noise:` at all and so
        # stands down on the node lookup, leaving the guard unexercised.
        # `model.kind: pipeline` has no node registry; a `noise:` beside
        # `stages:` is an unknown key the build names itself.
        {"kind": "pipeline", "stages": [], "noise": MODEL_NOISE}])
    def test_C18_kind_stands_down_on_a_pipeline_model(self, model):
        """``preflight/model.py::_nodes`` gates on ``kind: graph`` for the
        same reason: reading a pipeline's keys as graph nodes invents a
        placement the build does not make."""
        assert mine(sigma_document(MODEL_NOISE, RADIOMETER,
                                   model=model)) == frozenset()

    @pytest.mark.parametrize("weighed", [7, "radiometer", {"kind": 7},
                                         {"kind": "banana"}])
    def test_an_unreadable_inference_noise_is_left_to_build_noise(self,
                                                                  weighed):
        """``inference.noise.kind: banana`` is ``build_noise``'s own refusal,
        naming the value and the enum.  Answering here would pre-empt a more
        specific sentence, and RAISING here would abort the pass."""
        assert mine(sigma_document(MODEL_NOISE, weighed)) == frozenset()

    def test_the_primary_record_is_found_by_name_among_several(self):
        """MINOR 7: **kills the ``"primary" in named`` branch being dropped.**
        With two named records, dropping it falls through to ``elif
        len(named) == 1``, which is false with two, and the check goes
        silent about a primary observation it should still have read."""
        document = sigma_document(MODEL_NOISE, RADIOMETER, observed={
            "primary": {"from": "simulation", "twin": "full"},
            "night": {"from": "simulation", "twin": "full"}})
        assert only(document, "C18").severity == REFUSE


class TestTheKindNoneWarning:
    """Row 7, and the only WARN this module has.  **The two stand-down rows
    are evaluated FIRST, and that ordering is the check's contract**: read the
    other way round, ``exit_helpers.diagnostic_document()`` trips this warning
    and the shipped WARN census is ONE rather than zero."""

    def test_a_drawing_operator_with_no_likelihood_warns_on_a_fitting_run(
            self):
        document = sigma_document(MODEL_NOISE, None,
                                   runs=[{"kind": "conjugate.gls"}])
        found = only(document, "C18")
        assert found.severity == WARN
        assert found.where == "model.noise"

    def test_it_stands_down_on_a_forward_only_document(self):
        """The shared non-fitting set includes 4B's cross-run exits too.

        The base document's run is ``kind: forward``, so nothing here fits
        and there is nothing to weigh.
        """
        assert _T2C_NOT_FITTING == frozenset(
            {"forward", "mmodes", "compare", "benchmark"}
        )
        assert mine(sigma_document(MODEL_NOISE, None)) == frozenset()

    def test_an_explicit_kind_none_is_the_same_cell(self):
        """``build_noise`` answers ``NoiseBuild(kind="none")`` for an absent
        section AND for an explicit ``kind: none``, so the two are one cell
        here.  **Kills** a check that read only the absence."""
        document = sigma_document(MODEL_NOISE, {"kind": "none"},
                                   runs=[{"kind": "conjugate.gls"}])
        assert only(document, "C18").severity == WARN

    def test_a_run_expecting_a_refusal_is_not_a_fit(self):
        """``_a30_exits`` drops a run declaring ``expect: refuse`` and a kind
        the run grammar does not know -- both of which ``_kinds``
        deliberately keeps.  **Kills** re-deriving the fitting set from
        ``_kinds``, which would warn about a run written to assert a
        refusal."""
        for run in ({"kind": "conjugate.gls", "expect": "refuse"},
                    {"kind": "banana"}):
            assert mine(sigma_document(MODEL_NOISE, None,
                                       runs=[run])) == frozenset()

    def test_the_warning_names_the_gate_and_the_escape(self):
        document = sigma_document(MODEL_NOISE, None,
                                   runs=[{"kind": "conjugate.gls"}])
        found = only(document, "C18")
        assert "inference.noise: {kind: homoscedastic}" in found.message
        assert "inference.twin.without: [noise]" in found.message
        assert found.message.endswith("(check C18).")

    @pytest.mark.parametrize("observed, why", [
        (None, "no observation at all"),
        ({"file": {"path": "d.npy"}}, "the data came from a file"),
        ({"from": "simulation", "twin": "fit"}, "the fit twin drew it")])
    def test_the_stand_down_rows_are_evaluated_before_the_warning(self,
                                                                  observed,
                                                                  why):
        """**The ordering P14 is about, as a regression test.**

        Read top-to-bottom as the plan's table was originally written, the
        WARN row fires on a document with a drawing operator, no
        ``inference.noise:`` and a fitting run **whatever the observation
        says** -- so an ingested document, whose twin drew nothing at all,
        earns a warning about a sigma it does not have.  Evaluated with the
        two stand-down rows FIRST, it is silent.

        Each row here is a document the WARN must not reach: ``{why}``.
        """
        document = sigma_document(MODEL_NOISE, None, observed=observed,
                                   runs=[{"kind": "conjugate.gls"}])
        assert mine(document) == frozenset(), why

    def test_a_kind_the_run_grammar_does_not_know_is_not_a_fit(self):
        """**Measured, and it corrects P14's own arithmetic.**  P14 predicts
        that ``exit_helpers.diagnostic_document()`` trips this WARN, because
        ``_A30_NOT_FITTING`` holds the four non-fitting exits and its run kind is
        neither.  It does not: its run is ``kind: diagnostics.identifiability``
        and ``sections/runs._KINDS`` holds ``identifiability``, so
        ``_a30_exits`` -- which intersects with that closed enum -- answers
        ``()``.  Reading the fitting set off ``_kinds`` MINUS the complement,
        as P14 assumes, would warn about it.
        """
        from rheplicant.config.preflight.model import _a30_exits
        from tests.config.exit_helpers import diagnostic_document

        document = diagnostic_document({"kind": "diagnostics.identifiability"})
        assert _a30_exits(document) == ()
        assert mine(document) == frozenset()


# --- the advice loop --------------------------------------------------------


class TestApplyingTheAdviceLiterally:
    """**Every refusal names an escape; these take a refused document, apply
    the escape's own words, and assert the id is gone AND the document
    loads.**  Not "sanity-check the wording": in wave 1 a new refusal named
    two escapes and following the SECOND literally produced a second refusal,
    and the reviewer's own prescribed fix was itself an advice loop.  Two
    experts reasoned about it; only the round that RAN it found the bug."""

    def test_dropping_report_clears_the_skip_cell(self):
        """Escape 1 of ``_SKIP_WITH_REPORT``: *"Drop report:"*."""
        before = checks_document({"linearity": {"mode": "skip",
                                                "reason": "x",
                                                "report": True}})
        assert only(before, "A1").message == _SKIP_WITH_REPORT
        after = checks_document({"linearity": {"mode": "skip",
                                               "reason": "x"}})
        assert mine(after) == frozenset()
        load_document(after)

    def test_changing_skip_to_report_clears_the_skip_cell(self):
        """Escape 2, which is the one wave 1's advice loop broke: *"drop
        reason: and change mode: skip to mode: report"*.  Following it must
        not earn ``_REASON_WITHOUT_SKIP`` on the way out."""
        after = checks_document({"linearity": {"mode": "report",
                                               "report": True}})
        assert mine(after) == frozenset()
        load_document(after)

    def test_the_a37_advice_clears_a37(self):
        """A37's own sentence: *"mode: skip carries its own reason:"*."""
        assert only(checks_document({"linearity": {"mode": "skip"}}),
                    "A37").message == A37_SKIP_WITHOUT_REASON
        after = checks_document(CHECKS_SKIP)
        assert mine(after) == frozenset()
        load_document(after)

    def test_drawing_what_you_weigh_clears_C18_kind(self):
        """Escape 1 of the cross-family refusal: *"Write
        model.noise.type: RadiometerNoiseOperator to draw what you weigh"*."""
        assert only(sigma_document(MODEL_NOISE, RADIOMETER),
                    "C18").severity == REFUSE
        after = sigma_document(RADIOMETER_DRAWN, RADIOMETER)
        assert mine(after) == frozenset()
        load_document(after)

    def test_weighing_what_you_draw_clears_C18_kind(self):
        """Escape 2: *"or inference.noise.kind: homoscedastic to weigh what
        you draw"*.  The SECOND escape, run literally -- which is the one
        wave 1 shipped broken."""
        after = sigma_document(MODEL_NOISE, HOMOSCEDASTIC)
        assert mine(after) == frozenset()
        load_document(after)

    def test_the_transposed_rows_advice_clears_it_too(self):
        """Both escapes of the OTHER refusing row, so a copied sentence that
        names the wrong type is caught."""
        assert only(sigma_document(RADIOMETER_DRAWN, HOMOSCEDASTIC),
                    "C18").severity == REFUSE
        for after in (sigma_document(MODEL_NOISE, HOMOSCEDASTIC),
                      sigma_document(RADIOMETER_DRAWN, RADIOMETER)):
            assert mine(after) == frozenset()
            load_document(after)

    def test_declaring_the_likelihood_clears_the_warning(self):
        """The WARN's escape 1: *"Declare inference.noise: {kind:
        homoscedastic}"*."""
        before = sigma_document(MODEL_NOISE, None,
                                runs=[{"kind": "conjugate.gls",
                                       "names": ["g"]}])
        assert only(before, "C18").severity == WARN
        after = sigma_document(MODEL_NOISE, HOMOSCEDASTIC,
                               runs=[{"kind": "conjugate.gls",
                                      "names": ["g"]}])
        assert mine(after) == frozenset()
        load_document(after)

    def test_repairing_the_twin_clears_the_warning(self):
        """The WARN's escape 2, corrected: *"drop model.noise -- and the
        inference.twin.without: [noise] that repairs it -- if this data is
        meant to be noise-free"*.

        **The old wording was an advice loop.**  The WARN fires only when the
        primary observation came out of the FULL twin, and
        ``inference.twin.without:`` shapes the FIT twin alone -- so that key
        cannot change what the WARN is about, and the base document already
        carries it (``exit_helpers._repaired``'s default).  Applying the old
        escape literally left the WARN standing.

        The corrected escape drops ``model.noise`` too, and BOTH edits are
        required together: dropping ``model.noise`` alone while
        ``inference.twin.without: [noise]`` still names it is D-10's own
        trap -- ``AssemblyError without('noise'): no operator sits at
        'noise' in this assembly.`` -- so the ``twin:`` key is dropped
        whole, which is the escape's own second clause.
        """
        before = sigma_document(MODEL_NOISE, None,
                                runs=[{"kind": "conjugate.gls",
                                       "names": ["g"]}])
        after = repatch(
            before,
            model={key: value for key, value in BASE_MODEL.items()
                   if key != "noise"},
            inference={key: value for key, value in before["inference"].items()
                       if key != "twin"})
        assert mine(after) == frozenset()
        load_document(after)


# --- what the pass itself must survive --------------------------------------


class TestNeitherCheckEverRaises:
    """``sweep`` turns any exception out of a check into a ``ConfigError``
    that aborts the whole pass and hides every finding after it."""

    @pytest.mark.parametrize("checks", [
        {"a b": {"mode": "warn"}},
        {"": {"mode": "warn"}},
        {"linearity": {"mode": ["warn"]}},
        [], 7, True, ("linearity",)])
    def test_a_hostile_checks_section_does_not_abort_the_pass(self, checks):
        """**The measured trap.**  ``check_gates`` composes
        ``inference.checks.<name>`` from the KEY, and
        ``parse_path('inference.checks.a b')`` RAISES -- so the ``where``
        guard would turn a document with a hostile check name into
        "pre-flight check 'A1.checks' emitted where=..." and discard every
        other finding in the report.  The finding is re-homed onto its legal
        parent instead, and the MESSAGE still quotes the key the user wrote.

        A NON-string key is a different death: the evidence freeze refuses it
        before any check runs (Task 4's hardened contract, pinned in
        ``tests/bootstrap/test_bootstrap_frozen.py``) -- those three cases
        left this table at Task 10's OI-1 triage and are pinned refusing in
        the test below."""
        found = findings(checks_document(checks))
        for one in found:
            if one.check in ("A1", "A37"):
                assert one.where.startswith("inference.checks")

    @pytest.mark.parametrize(("checks", "key_type"), [
        ({7: {"mode": "warn"}}, "int"),
        ({None: {"mode": "warn"}}, "NoneType"),
        ({(1, 2): {"mode": "warn"}}, "tuple"),
    ], ids=["an-int-key", "a-none-key", "a-tuple-key"])
    def test_a_non_string_check_name_is_refused_by_the_evidence_freeze(
            self, checks, key_type):
        """The boundary Task 4 hardened: ``initial_merge``'s evidence freeze
        requires exact string mapping keys, because the origin tree cannot
        tell a mapping key from a sequence index otherwise.  A YAML scalar
        int key really does reach here (the loader produces ``7``), so this
        is a user route, not only a programmatic one."""
        with pytest.raises(ConfigError) as caught:
            preflight(checks_document(checks))
        assert str(caught.value) == (
            "initial_merge document: unsupported evidence mapping key type "
            f"{key_type}.")

    def test_the_re_homed_finding_still_names_the_key_the_user_wrote(self):
        found = only(checks_document({"a b": {"mode": "warn"}}), "A1")
        assert found.where == "inference.checks"
        assert "'a b' is not a check" in found.message

    @pytest.mark.parametrize("observed", [7, [], "simulation", {"primary": 7},
                                          {"a": {"from": "simulation"},
                                           "b": {"from": "simulation"}}])
    def test_a_hostile_observed_block_does_not_abort_the_pass(self, observed):
        preflight(sigma_document(MODEL_NOISE, RADIOMETER,
                                 observed=observed))

    @pytest.mark.parametrize("patch", [
        {"model": {"noise": {"type": {"a": 1}}}},
        {"inference": {"noise": {"kind": ["x"]}}}])
    def test_a_hostile_drawn_or_weighed_kind_does_not_abort_the_pass(self,
                                                                     patch):
        """MINOR 8: **kills either ``isinstance(..., str)`` guard.**  Without
        the drawn-side one, ``{'a': 1} in _DRAWING_TYPES`` is a bare
        ``TypeError: unhashable type: 'dict'``; without the weighed-side
        one, ``['x'] in agrees`` is the same error over a list.  Both are
        killed only by OTHER modules today -- some sibling check happens to
        refuse first and ``_sigma_families`` is never reached -- which is
        why this task carries its own pin."""
        preflight(preflight_document(**patch))

    def test_the_base_document_earns_nothing_from_this_module(self):
        """THE BASE MUST EARN NO FINDING OF ITS OWN, and it is
        ``NoiseOperator`` x ``homoscedastic`` x ``twin: full`` -- an AGREEING
        cell, reached rather than dodged."""
        assert mine(preflight_document()) == frozenset()
