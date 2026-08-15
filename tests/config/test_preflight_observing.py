"""A14 and A15: the two directions the switch order was never checked in.

Every test names the wrong implementation it kills, because a test that cannot
name one is decoration -- Plan 2C shipped twenty-seven surviving mutants and
every one of them was in a test.

**What the package really does with a wrong row count, measured at ``740d9d1``
and not what this task's brief said.**  The brief claimed a three-row
``gamma_src_re`` under a four-label order "BUILDS".  It does not, on the
pre-flight base document: ``load_document`` reaches ``build_inference`` ->
``build_observed``, simulates the observed data through the twin, and
``rhino_cal_jax``'s switch cycle refuses with *"source_index values span
[0, 3] which is out of range for 3 labels ('0', '1', '2')"*.  What is true, and
is what these checks are for:

* the refusal is **late** -- measured, the same document carrying
  :data:`UNREADABLE_BEAM` reports the beam, every time;
* it names switch LABELS, not the field the user wrote;
* and in the other direction there is no refusal at all: ``{ones: [9, 8]}``
  under a four-label order loads, simulates and runs, with five rows nobody
  ever indexes.
"""

import pytest

from rheplicant.config.findings import REFUSE
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight.observing import _gamma_rows, _switch_order
from tests.config.preflight_helpers import (
    BASE_MODEL,
    BASE_OBSERVATION,
    UNREADABLE_BEAM,
    preflight_document,
)

#: A four-label order: three loads behind the antenna.
ORDER = ["antenna", "ambient", "hot", "noise_source"]

#: The other six fields NoiseWaveOperator requires, none of which A15 reads.
#: Bound once so a test's own noise_wave spec is nothing but the gamma halves.
OTHER_FIELDS = {"t_unc": {"zeros": ["n_freq"]}, "t_cos": {"zeros": ["n_freq"]},
                "t_sin": {"zeros": ["n_freq"]}, "t_rx": {"zeros": ["n_freq"]},
                "gamma_rec_re": {"zeros": ["n_freq"]},
                "gamma_rec_im": {"zeros": ["n_freq"]}}

#: The class name the ``python:`` hatch relocates a noise wave with.
NOISE_WAVE_CLASS = "rheplicant.radio:NoiseWaveOperator"


def switching(order=ORDER, **extra):
    """An observation that switches through ``order``."""
    return {**BASE_OBSERVATION,
            "switching": {"mode": "cycle", "order": list(order), "dwell": 4,
                          **extra}}


def loads(order=ORDER):
    """``model.cal_loads`` matching ``order[1:]``, which A14 wants present."""
    return {label: {"t_load": {"value": 300.0, "unit": "K"}}
            for label in order[1:]}


def noise_wave(re=None, im=None, **extra):
    """A ``model.noise_wave`` spec whose gamma halves are the test's own."""
    gamma = {"zeros": [4, 8]}
    return {**OTHER_FIELDS, "gamma_src_re": re if re is not None else gamma,
            "gamma_src_im": im if im is not None else gamma, **extra}


def ingested(order=("antenna", "internal_load", "heated_load"), **patch):
    """A document whose observation is a RECORDING, section REPLACED.

    Replaced rather than merged, following
    ``test_preflight_model._ingested``: an ingested run declares no ``freq:``
    and no ``time:`` -- the recording carries both, and ``build_observation``
    refuses a document that declares either beside ``from_file:``.  The shape
    is ``tests/config/test_config_document.py:133-151``'s
    (``TestIngestedDocuments.make_document``): three tests build it with
    ``load_document`` and the second runs it, doubling the recorded ones. It
    carries a three-label order and no ``model.cal_loads`` anywhere, and all
    three passed unskipped in this worktree.
    """
    document = preflight_document(**patch)
    document["observation"] = {
        "from_file": {"format": "rhino_hdf5", "path": "obs.hd5f",
                      "freq_unit": "MHz", "settle_seconds": 0.0},
        "switching": {"order": list(order)},
    }
    return document


def pipeline_document(**patch):
    """A ``kind: pipeline`` document, the model section REPLACED not merged.

    ``preflight_document(model={"kind": "pipeline", ...})`` MERGES into
    ``BASE_MODEL``, and the result is refused by "kind: pipeline takes
    stages: and nothing else" -- a document that says nothing about either
    check here. Replaced, and with a switch cycle and no ``cal_loads``, it
    BUILDS: measured, ``load_document`` returns a ``Pipeline`` twin.
    """
    document = preflight_document(**patch)
    document["model"] = {
        "kind": "pipeline",
        "stages": [{"name": "gain", "type": "GainOperator",
                    "gain": {"value": 1.1, "unit": "dimensionless"}}]}
    return document


class TestA14AnOrderWithNoLoadsBehindIt:
    def test_an_order_naming_loads_with_no_cal_loads_is_refused(self):
        """The measured hole: this document builds, simulates and RUNS today
        (probe: ``run_forward`` returns (16, 8) with the cycle in
        ``coords.extra`` and nothing consuming it).

        Kills: not shipping the check at all, which is 2C's shape 3 -- a
        correct decision with no test, so reverting it stays green.
        """
        found = list(_switch_order(preflight_document(observation=switching())))
        assert [finding.check for finding in found] == ["A14"]
        assert found[0].severity == REFUSE

    def test_the_refusal_names_the_labels_the_order_declared(self):
        """Kills a message that says "cal_loads is missing" and stops there.

        The user has to type three specific keys IN ORDER; a refusal that does
        not carry them sends them back to the schema. Attribution, not
        presence: the labels must be the ones after the antenna, so a check
        that quoted the whole order -- antenna included -- fails here.
        """
        found = list(_switch_order(preflight_document(observation=switching())))
        assert "['ambient', 'hot', 'noise_source']" in found[0].message
        assert "'antenna'" not in found[0].message

    def test_the_refusal_carries_the_count_and_both_of_its_fixes(self):
        """Two survivors of the first mutation round, in one cell.

        ``{len(order)}`` -> ``{len(order) - 1}`` gives "gives 3 switch
        positions" under a four-label order -- the wrong count in the refusal
        that exists to state the count. And deleting ", or write switching:
        {mode: none}" leaves the reader one of the two fixes: declaring three
        loads they may not have is not the only way out of this.
        """
        found = list(_switch_order(preflight_document(observation=switching())))
        assert "gives 4 switch positions" in found[0].message
        assert "gives 3 switch positions" not in found[0].message
        assert "switching: {mode: none}" in found[0].message

    def test_the_where_is_the_key_the_user_has_to_add(self):
        """Kills a `where` pointing at the order (which is not what is wrong)
        or at a source path -- Task 2's guard only checks that it starts with a
        section name, so `observation.switching.order` would pass that and
        still send the reader to the line they wrote correctly."""
        found = list(_switch_order(preflight_document(observation=switching())))
        assert found[0].where == "model.cal_loads"

    def test_a_document_that_declares_its_loads_is_not_refused(self):
        """Kills a check that fires on the presence of an order rather than on
        the absence of the loads -- which would refuse every switching
        document in the repository."""
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads()})
        assert list(_switch_order(doc)) == []

    def test_cal_loads_declared_with_the_wrong_shape_still_stands_down(self):
        """``cal_loads: 3`` is A6's sentence ("is a label-keyed mapping"), and
        Task 4's ``A14.cal_loads`` leg stands down on it for the same reason.

        Kills a presence test written ``isinstance(nodes.get("cal_loads"),
        Mapping)``, which would tell this reader to declare a key they have
        declared while a better sentence about it is already queued.
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": 3})
        assert list(_switch_order(doc)) == []

    def test_a_relocated_cal_load_is_a_placement_and_not_a_key(self):
        """The loads may be declared without the key ``cal_loads`` at all.

        Measured: ``bandpass: {python: 'rheplicant.radio:CalLoadOperator',
        t_load: ...}`` beside a switching cycle loads, and the assembly puts
        the operator at the ``cal_loads`` node -- ``twin.lit`` is
        ``('global_signal', 'uniform_sky', 'cal_loads', 'gain', 'noise')``,
        carrying ``cal_loads`` and not ``bandpass``. Kills
        ``"cal_loads" in _nodes(document)``, which refuses that document while
        the package builds it: the same key-instead-of-placement mistake Task
        5's carry-forward records ``_lit`` making, one check over.
        """
        doc = preflight_document(
            observation=switching(),
            model={**BASE_MODEL,
                   "bandpass": {"python": "rheplicant.radio:CalLoadOperator",
                                "t_load": {"value": 300.0, "unit": "K"}}})
        assert list(_switch_order(doc)) == []

    def test_mode_none_declares_no_order_and_is_not_refused(self):
        """Kills a check that reads `order` without reading `mode`: a document
        may carry `switching: {mode: none, order: [...]}`-shaped leftovers, and
        `check_unknown_keys` (switching.py:72) is what refuses those."""
        doc = preflight_document(observation={**BASE_OBSERVATION,
                                              "switching": {"mode": "none"}})
        assert list(_switch_order(doc)) == []

    def test_no_switching_section_at_all_is_not_refused(self):
        """The default (`switching.py:61-62`). Kills a check that treats a
        missing section as an empty order and then as a violated one."""
        assert list(_switch_order(preflight_document())) == []

    def test_a_malformed_order_is_left_to_the_section_that_owns_it(self):
        """`declared_order` already refuses each of these, at
        observation-build time, which is BEFORE build_resources -- measured.
        Kills a check that re-derives the acceptance test and reports a second,
        differently-worded refusal for a document that already has a right one.
        """
        for order in ([], ["antenna"], ["ambient", "hot"],
                      ["antenna", "hot", "hot"], "antenna,hot", [1, 2]):
            doc = preflight_document(observation={
                **BASE_OBSERVATION,
                "switching": {"mode": "cycle", "order": order, "dwell": 4}})
            assert list(_switch_order(doc)) == [], order

    def test_a_key_the_mode_does_not_take_is_the_key_sweeps_own_refusal(self):
        """``{mode: cycle, order: [...], nope: 1}`` is ``check_unknown_keys``'
        sentence inside ``compile_switching`` (``switching.py:72``), and that
        runs in ``build_observation`` -- before the beam.

        Task 4's reader answers "cannot say" for it, and this pins that A14
        inherits the answer rather than reaching past it for the order. Kills
        the reader this task's brief printed, which validates ``order:`` and
        nothing else and would put a cal_loads sentence in front of the one
        that names the typo.
        """
        doc = preflight_document(observation=switching(nope=1))
        assert list(_switch_order(doc)) == []

    def test_a_pipeline_model_is_not_asked_for_a_graph_node(self):
        """Measured: `kind: pipeline` + a switching cycle + no cal_loads
        BUILDS today (`load_document` returns a `Pipeline` twin), and
        correctly -- a pipeline has no graph node ids, so `model.cal_loads` is
        not something it could declare. Kills the obvious implementation,
        `"cal_loads" not in document["model"]`, which refuses a document the
        package accepts."""
        assert list(_switch_order(pipeline_document(
            observation=switching()))) == []

    def test_a_model_that_is_not_a_mapping_keeps_the_builds_own_sentence(self):
        """``_structural`` guarantees ``model:`` is PRESENT, never that it is a
        mapping (Task 4's carry-forward). ``build_model`` answers with the type
        it got; a row of "declare model.cal_loads" one phase earlier names a
        fix that cannot be typed into a string. Kills a check that reads
        ``_nodes`` (which answers ``{}`` here) without asking what the section
        is."""
        assert list(_switch_order(preflight_document(observation=switching(),
                                                     model="graph"))) == []

    def test_an_ingested_run_is_not_asked_for_cal_loads(self):
        """**The false refusal this check would otherwise ship.**

        Measured: ``tests/config/test_config_document.py:133-151``
        (``TestIngestedDocuments.make_document``) is a recording with
        ``switching: {order: [antenna, internal_load, heated_load]}``, a
        ``model:`` of ``{gain: ...}`` and no ``cal_loads``
        anywhere. It loads, and ``run_forward`` doubles the recorded ones --
        three tests there pin it. The order labels the recording's own source
        index (``document.py:85-86`` hands it to ``to_state``); there is no model
        branch for it to fix, because the data was not simulated.

        Kills the implementation this task was handed, which reads the order
        and asks for ``model.cal_loads`` whatever produced the data -- which,
        given Task 4's reader, turns those three tests red. The plan's own
        ``_a14_order``, typed verbatim, stands down here for the WRONG reason
        (it reads a recording's ``order:``-alone grammar as no order at all),
        so this cell alone does not tell the two readers apart --
        ``test_an_ingested_order_counts_the_same_way`` is the one that does.
        """
        assert list(_switch_order(ingested())) == []

    def test_the_finding_carries_its_check_tag(self):
        """Enforced from Task 3 on: every finding ends with ``(check AN).``.
        Kills a message reworded without its tag, which drops the id a reader
        greps for."""
        found = list(_switch_order(preflight_document(observation=switching())))
        assert found[0].message.endswith("(check A14).")


class TestA15TheRowCountOnEveryPath:
    def test_three_rows_under_a_four_label_order_are_refused(self):
        """The measured hole in its late-refusal direction: the package sees
        this only when the twin is evaluated, and then as a switch-cycle error
        about labels.

        Kills: not shipping the check (shape 3).
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        found = list(_gamma_rows(doc))
        assert [finding.where for finding in found] == [
            "model.noise_wave.gamma_src_re", "model.noise_wave.gamma_src_im"]
        assert {finding.check for finding in found} == {"A15"}
        assert {finding.severity for finding in found} == {REFUSE}

    def test_the_refusal_says_which_number_is_which(self):
        """Attribution, not presence -- 2C's shape 1, seven instances. A
        message carrying "4" and "3" passes an `in` test with the two swapped,
        and then tells the reader to change the order rather than the rows.
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        message = list(_gamma_rows(doc))[0].message
        assert "declares 4 sources" in message
        assert "declares 3 rows" in message
        assert "declares 3 sources" not in message

    def test_the_fix_is_the_number_the_document_does_not_have(self):
        """**The refusal is the product, so the fix sentence is pinned too.**

        `Write {n_source} rows` -> `Write {rows} rows` survived the first
        round of mutation: the message then tells the reader to write exactly
        the number they already wrote, and every assertion above still
        passed because they all read the `why` sentence.
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        message = list(_gamma_rows(doc))[0].message
        assert "Write 4 rows in switch order" in message
        assert "Write 3 rows" not in message

    def test_the_recommended_sugar_is_copyable(self):
        """`part:` is `re`/`im` -- `half.rsplit('_', 1)[1]`.

        Kills `[0]`, which recommends
        `{from_switch_order: {..., part: gamma_src}}`: a fix the reader
        cannot paste, from the one sentence whose whole job is to be pasted.
        The two halves must differ, so this reads both.
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        first, second = list(_gamma_rows(doc))
        assert "part: re}}" in first.message
        assert "part: im}}" in second.message
        assert "part: gamma_src" not in first.message

    def test_the_message_opens_with_the_field_the_reader_must_edit(self):
        """`raise_if_refused` (findings.py:159-178) quotes the first
        refusal's MESSAGE and names only the OTHER refusals' `where`, so a
        reader who hits A15 first sees no path unless the message carries
        one -- and there are two routes and three spellings it could be.

        Kills dropping the prefix, which no `in` assertion elsewhere sees.
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        found = list(_gamma_rows(doc))
        assert found[0].message.startswith(
            "model.noise_wave.gamma_src_re: ")
        assert found[1].message.startswith(
            "model.noise_wave.gamma_src_im: ")

    def test_the_why_names_every_label_the_order_declares(self):
        """`{list(order)}` -> `{list(order[1:])}` survived: "declares 4
        sources (['ambient', 'hot', 'noise_source'])" -- four sources beside
        three labels, in the sentence whose subject is the count."""
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        message = list(_gamma_rows(doc))[0].message
        assert ("declares 4 sources (['antenna', 'ambient', 'hot', "
                "'noise_source'])") in message

    def test_too_few_rows_and_too_many_are_told_apart(self):
        """The two directions differ in what the package does, so they differ
        in what the refusal may claim.

        Measured: three rows under a four-label order are refused at call time
        by the switch cycle; NINE rows under the same order load, simulate the
        observed data and run, with five rows nobody indexes. A message
        promising "nothing sees this until the twin is evaluated" is FALSE for
        the second, and one promising "nothing sees it at all" is false for the
        first. Kills a single tail written for whichever direction the author
        happened to measure.
        """
        def message(rows):
            spelling = {"zeros": [rows, 8]}
            doc = preflight_document(
                observation=switching(),
                model={**BASE_MODEL, "cal_loads": loads(),
                       "noise_wave": noise_wave(spelling, spelling)})
            return list(_gamma_rows(doc))[0].message

        assert "until the twin is evaluated" in message(3)
        assert "never used" not in message(3)
        assert "never used" in message(9)
        assert "until the twin is evaluated" not in message(9)

    def test_the_right_row_count_is_not_refused(self):
        """Kills a check that refuses whenever a literal appears."""
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave()})
        assert list(_gamma_rows(doc)) == []

    def test_the_symbol_spelling_is_read_rather_than_refused(self):
        """`{zeros: [n_source, n_freq]}` is right BY CONSTRUCTION and builds at
        (4, 8) -- measured. Kills an implementation that only understands
        literal integers and refuses the spelling the schema recommends."""
        symbolic = {"zeros": ["n_source", "n_freq"]}
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(symbolic,
                                                                 symbolic)})
        assert list(_gamma_rows(doc)) == []

    def test_n_load_is_refused_as_the_off_by_one_it_is(self):
        """`n_load` is `n_source - 1` (symbols.py:73-74), so
        `{zeros: [n_load, n_freq]}` resolves to three rows under a four-label
        order -- measured, and refused late by the switch cycle. Kills an
        implementation that treats any symbol as "correct by construction",
        which is the tempting shortcut and is wrong for exactly one symbol in
        the table."""
        wrong = {"zeros": ["n_load", "n_freq"]}
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(wrong, wrong)})
        found = list(_gamma_rows(doc))
        assert len(found) == 2
        assert "declares 3 rows" in found[0].message

    @pytest.mark.parametrize("spelling, rows", [
        ({"ones": [3, 8]}, 3),
        ({"full": {"shape": [3, 8], "value": 0.0}}, 3),
        ({"list": [[0.0] * 8] * 3}, 3),
        ({"stack": [{"ref": "resources.arrays.row"}] * 3}, 3),
        ({"normal": {"shape": [3, 8],
                     "seed": {"from": "runtime.seeds.g"}}}, 3),
        # The OTHER draw form.  `normal` alone leaves `uniform` untested, and
        # the two are separate branches of the same walk -- 2C's shape 4 in
        # the form this very docstring warns about.
        ({"uniform": {"shape": [3, 8],
                      "seed": {"from": "runtime.seeds.g"}}}, 3),
        # A STRING `axis:` is the noise-sigma modifier, not stack's own
        # argument (refs.py:120-126), so the stack is still on axis 0 and its
        # rows are still its entries -- measured, four such entries build at
        # (4, 8).  Kills `node.get("axis", 0) == 0`, which reads 'time' as a
        # non-zero axis and declines to count a stack it could have counted.
        ({"stack": [{"ref": "resources.arrays.row"}] * 3, "axis": "time"}, 3),
    ], ids=["ones", "full", "list", "stack", "normal", "uniform",
            "stack-under-a-modifier-axis"])
    def test_every_countable_form_is_counted(self, spelling, rows):
        """Each of these declares three rows in its own text -- measured, one
        document per row, every one of them refused late by the switch cycle.

        Kills a check that reads `{zeros: ...}` and nothing else. That is not a
        hypothetical narrowing: `symbols.resolve_shape`'s own docstring records
        that the array forms and the draw forms diverged on exactly this for
        check A41, and "a report that depends on which constructor the writer
        reached for is worse than no report".
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(spelling,
                                                                 spelling)})
        found = list(_gamma_rows(doc))
        assert len(found) == 2
        assert f"declares {rows} rows" in found[0].message

    @pytest.mark.parametrize("spelling", [
        {"ref": "resources.arrays.gamma"},
        {"file": {"path": "gamma.npz", "format": "npz", "key": "g"}},
        {"zeros": ["2 * n_source", 8]},
        {"stack": [{"ref": "resources.arrays.row"}] * 3, "axis": 1},
        {"from_switch_order": {"resource": "resources.s_params", "part": "re"}},
        {"zeros": [True, 8]},
        {"value": 0.0},
        {"zeros": [3]},
        {"normal": {"shape": [3, 8, 2], "seed": {"from": "runtime.seeds.g"}}},
        {"list": []},
        {"stack": []},
        # `axis: true` and `axis: 0.0` DO stack on axis 0 (refs.py:124-126's
        # `mine` test excludes them), so counting them would be arithmetically
        # right and would still pre-empt: the modifier alphabet refuses both
        # in their own words, and a row count in front of that names a fix
        # that is not the fault.  `axis: 'nope'` is the same refusal from the
        # string side.
        {"stack": [{"ref": "resources.arrays.row"}] * 3, "axis": True},
        {"stack": [{"ref": "resources.arrays.row"}] * 3, "axis": 0.0},
        {"stack": [{"ref": "resources.arrays.row"}] * 3, "axis": "nope"},
    ], ids=["ref", "file", "arithmetic-symbol", "stack-on-axis-1",
            "from_switch_order", "bool-is-not-a-row-count", "scalar",
            "one-dimensional", "three-dimensional", "empty-list",
            "empty-stack", "stack-on-a-bool-axis", "stack-on-a-float-axis",
            "stack-on-an-unknown-axis"])
    def test_a_row_count_the_text_does_not_declare_is_left_alone(self,
                                                                 spelling):
        """The boundary of Plan 3A, as a test rather than a sentence.

        Kills a check that reaches for `shape[0]` on anything shaped like a
        list, and a check that "helpfully" resolves a ref. `from_switch_order`
        is in the list for a different reason: its row count is right by
        construction (refs.py:169-188), and a finding there would contradict a
        refusal the value grammar already gets right. `{zeros: [3]}` is the
        one-dimensional spelling, whose "3" is n_freq and not a row count --
        `__check_init__` refuses it by ndim, in its own words.

        `{zeros: [True, 8]}` needs saying out loud: `isinstance(True, int)` is
        True in Python, so a row-count reader written `isinstance(entry, int)`
        reads `True` as **1** and reports "declares 1 rows" for a shape
        `resolve_extent` refuses in its own words (symbols.py:128-132). The
        `not isinstance(x, bool)` half of the guard is what this cell defends,
        and nothing else here reaches it.
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL, "cal_loads": loads(),
                                        "noise_wave": noise_wave(spelling,
                                                                 spelling)})
        assert list(_gamma_rows(doc)) == []

    def test_a_malformed_order_produces_no_row_count_refusal(self):
        """`None` and `()` are DIFFERENT answers from the order reader, and A15
        divides on exactly that.

        `None` means "the text does not say what the order is" -- a non-list, a
        first label that is not `antenna`, a repeated label -- and A15 must
        stand down, because `n_source` is undecidable and `declared_order`
        already refuses it in its own words. `()` means "no switching at all",
        which §3.2(h)1 says IS decidable: `n_source = len(order) or 1`, one
        row.

        Kills `if not order: return` written in place of `if order is None:
        return` -- under which a malformed order silently becomes the
        one-source case and every countable `gamma_src` in the document is
        refused for having the wrong number of rows against a count nothing
        declared. Every other cell in this class supplies a well-formed order
        or none at all, so none of them can see it.
        """
        countable = {"zeros": [3, 8]}
        for bad in (["hot", "ambient"], ["antenna", "hot", "hot"],
                    "antenna,hot", [1, 2]):
            doc = preflight_document(
                observation={**BASE_OBSERVATION,
                             "switching": {"mode": "cycle", "order": bad,
                                           "dwell": 4}},
                model={**BASE_MODEL, "cal_loads": loads(),
                       "noise_wave": noise_wave(countable, countable)})
            assert list(_gamma_rows(doc)) == [], bad
        # ...and the `()` half still decides, or the guard above is just a
        # switched-off check.
        no_switching = preflight_document(
            model={**BASE_MODEL,
                   "noise_wave": noise_wave(countable, countable)})
        assert len(list(_gamma_rows(no_switching))) == 2

    def test_each_half_is_checked_on_its_own(self):
        """One half countable and wrong, the other not countable at all.

        Kills "only gamma_src_re is checked" -- 2C's shape 4, a hole closed on
        one route and left open on its twin, four instances. The `where` is
        what pins it: an implementation that checked `re` and reported the
        finding against `im` passes a length assertion.
        """
        doc = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads(),
                   "noise_wave": noise_wave({"ref": "resources.arrays.gamma"},
                                            {"zeros": [3, 8]})})
        found = list(_gamma_rows(doc))
        assert [finding.where for finding in found] == [
            "model.noise_wave.gamma_src_im"]

    def test_the_replacement_twin_is_checked_too(self):
        """Measured: `inference.twin.replace.noise_wave` with three rows under
        a four-label order reaches `build_node_operator` through
        `twin.py:67-69`, and `fit_twin["noise_wave"].gamma_src_re` comes back
        at (3, 8) while the model twin is at (4, 8) -- the run then dies in the
        fit rather than in the simulation. Kills a check that reads `model:`
        only -- the same shape-4 mutation as the test above, on the route a
        reader is least likely to think of."""
        doc = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads(),
                   "noise_wave": noise_wave()},
            inference={"twin": {"replace": {
                "noise_wave": noise_wave({"zeros": [3, 8]},
                                         {"zeros": [3, 8]})}}})
        found = list(_gamma_rows(doc))
        assert [finding.where for finding in found] == [
            "inference.twin.replace.noise_wave.gamma_src_re",
            "inference.twin.replace.noise_wave.gamma_src_im"]

    def test_a_pipeline_model_gets_no_row_count_for_its_twin_block(self):
        """**A14's own stand-down, which A15 was missing.**

        Measured on this exact document: ``build_fit_twin``
        (``twin.py:46-50``) refuses the WHOLE ``inference.twin:`` block on a
        ``kind: pipeline`` model -- *"inference.twin: repairs a graph
        assembly, and this model is kind: pipeline (Pipeline). A pipeline is
        rebuilt, not repaired"* -- so a row count inside that block answers
        about a block nobody can fix by editing the rows. Before this, A15
        fired on it and was the only refusal the reader got.

        Kills reading ``inference.twin.replace`` without asking what the
        model is: the model half already answers ``{}`` here through
        ``_nodes``, so this guard is the twin half's alone and nothing else
        in this class reaches it.
        """
        doc = pipeline_document(
            observation=switching(),
            inference={"twin": {"replace": {
                "noise_wave": noise_wave({"zeros": [3, 8]},
                                         {"zeros": [3, 8]})}}})
        assert list(_gamma_rows(doc)) == []

    def test_a_noise_wave_key_holding_another_class_is_not_asked(self):
        """**The identity question, which placement cannot answer.**

        ``noise_wave: {type: GainOperator, gain: ..., gamma_src_re: {zeros:
        [3, 8]}}`` LANDS at the ``noise_wave`` node, so a placement-shaped
        site rule asks a ``GainOperator`` for its switch rows -- and measured,
        that was the only refusal the document earned, one phase before
        ``_pick_class`` says "type: 'GainOperator' is not registered at this
        node; it takes ['NoiseWaveOperator']".

        Kills ``_t5_claims(key, spec) == ("noise_wave",)`` as the site rule.
        ``test_a_spec_that_lands_somewhere_else_is_not_asked_about_rows``
        guards the mirror direction and its docstring names this exact
        failure; this is the direction that fired.
        """
        doc = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads(),
                   "noise_wave": {"type": "GainOperator",
                                  "gain": {"value": 1.1,
                                           "unit": "dimensionless"},
                                  "gamma_src_re": {"zeros": [3, 8]},
                                  "gamma_src_im": {"zeros": [3, 8]}}})
        assert list(_gamma_rows(doc)) == []

    def test_the_twin_half_reads_its_key_the_way_the_model_half_does(self):
        """Both routes ask the same two questions -- is the key a graph node,
        and does the class it declares carry ``gamma_src``.

        A replacement is handed to ``build_node_operator(node_id, spec)``
        (``twin.py:67-69``), and a ``NoiseWaveOperator`` reads
        ``coords.extra['receiver_input']`` wherever it sits, so its row count
        is a property of the OPERATOR and not of the node: refusing here
        cannot be a false refusal, because either the replacement assembles
        -- and the rows are wrong -- or the document is refused anyway.
        (Measured: ``replace_node`` on a node the model does not light dies
        with a bare ``KeyError`` naming the lit set, which is late and not
        this check's sentence.)

        Kills reading ``replace["noise_wave"]`` literally while the model
        half resolves the class -- an asymmetry no test could see while the
        two agreed -- and, in the second half, kills dropping the graph-node
        gate, which would emit a ``where`` the path grammar cannot spell and
        take the whole pass down from outside its own ``try``.
        """
        wrong = noise_wave({"zeros": [3, 8]}, {"zeros": [3, 8]},
                           python=NOISE_WAVE_CLASS)
        doc = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads()},
            inference={"twin": {"replace": {"bandpass": wrong}}})
        assert [finding.where for finding in _gamma_rows(doc)] == [
            "inference.twin.replace.bandpass.gamma_src_re",
            "inference.twin.replace.bandpass.gamma_src_im"]
        unspellable = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads()},
            inference={"twin": {"replace": {"not-a-node": wrong}}})
        assert list(_gamma_rows(unspellable)) == []

    def test_a_relocated_noise_wave_is_checked_where_it_lands(self):
        """The THIRD route, which the key-shaped reading misses.

        Measured: `bandpass: {python: 'rheplicant.radio:NoiseWaveOperator',
        ...}` places the operator at the `noise_wave` node -- with four rows it
        loads and `twin['noise_wave'].gamma_src_re` is (4, 8); with three it
        dies in the switch cycle exactly as the plain spelling does. Kills
        `_nodes(document).get("noise_wave")`, which answers nothing here, and
        pins that the site is the PLACEMENT rather than the key.
        """
        wrong = {"zeros": [3, 8]}
        doc = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads(),
                   "bandpass": noise_wave(wrong, wrong,
                                          python=NOISE_WAVE_CLASS)})
        found = list(_gamma_rows(doc))
        assert [finding.where for finding in found] == [
            "model.bandpass.gamma_src_re", "model.bandpass.gamma_src_im"]

    @pytest.mark.parametrize("rows, refusals", [(3, 0), (2, 2)],
                             ids=["right", "wrong"])
    def test_an_ingested_order_counts_the_same_way(self, rows, refusals):
        """**The grammar the plan's own reader gets wrong**, and the only test
        that can see it.

        An ingested run declares ``order:`` ALONE, with no ``mode:`` -- the
        recording carries the cycle (``observation.py:336-348``) -- and
        ``build_observation`` puts that order into the resolution context
        (``observation.py:400-403``, ``switch_order=switching.order``), so
        ``ShapeScope.n_source`` is 3 here exactly as it is for a ``mode:
        cycle`` document with three labels.

        Kills the reader this task's brief printed, which tests ``mode ==
        "cycle"`` and answers ``()`` for a recording: under it the right
        document (three rows) is REFUSED for not having one row, and the wrong
        one is refused with the wrong number in the message. Both A14 cells
        above stay green under that reader -- it stands down on this document
        for the wrong reason -- so this is the cell that decides between the
        two readers.
        """
        spelling = {"zeros": [rows, 8]}
        found = list(_gamma_rows(ingested(
            model={**BASE_MODEL, "noise_wave": noise_wave(spelling,
                                                          spelling)})))
        assert len(found) == refusals
        for finding in found:
            assert "declares 3 sources" in finding.message

    def test_a_spec_that_lands_somewhere_else_is_not_asked_about_rows(self):
        """The same clause in the direction that would over-fire: a
        `gamma_src_re` written under a key whose operator is NOT a noise wave
        lands at that key's own node, where the field is an unknown
        constructor argument and `build_node_operator` says so by name.

        Kills "any spec carrying gamma_src_re is a noise wave", which would
        answer a row count one phase before the sentence that names the real
        fault -- carry-forward rule 1, do not pre-empt a more specific
        refusal.
        """
        doc = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads(),
                   "gain": {"gain": {"value": 1.1, "unit": "dimensionless"},
                            "gamma_src_re": {"zeros": [3, 8]}}})
        assert list(_gamma_rows(doc)) == []


class TestSection415ModeNoneIsTheSameRule:
    def test_three_rows_with_no_switching_are_refused(self):
        """§4.1.5's second sentence: measured, the operator carries three
        sources, finds no switch index at call time and refuses there -- after
        the beam. Kills an implementation that computes the expected count as
        `len(order)`, which is ZERO here and would refuse the correct one-row
        document while accepting this one."""
        doc = preflight_document(model={**BASE_MODEL,
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        found = list(_gamma_rows(doc))
        assert len(found) == 2
        assert "exactly one source" in found[0].message

    def test_one_row_with_no_switching_is_not_refused(self):
        """The correct document, which loads and runs. Kills an off-by-one in
        the other direction."""
        one = {"zeros": [1, 8]}
        doc = preflight_document(model={**BASE_MODEL,
                                        "noise_wave": noise_wave(one, one)})
        assert list(_gamma_rows(doc)) == []

    def test_the_mode_none_finding_is_still_check_a15(self):
        """A decision, pinned: §4.1.5 has no id of its own, and this is A15's
        property at n_source == 1, so it carries A15's id and cites §4.1.5 in
        the message. Kills a later edit that gives it `check=""`, which would
        drop it out of `Report.checks()` and out of the docs table."""
        doc = preflight_document(model={**BASE_MODEL,
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        found = list(_gamma_rows(doc))
        assert {finding.check for finding in found} == {"A15"}
        assert "§4.1.5" in found[0].message

    def test_the_mode_none_tail_is_not_the_silent_one(self):
        """**Three behaviours, three tails, and this cell is the third's.**

        Replacing this branch's tail with the ``rows > n_source`` one survived
        the first mutation round, because
        ``test_too_few_rows_and_too_many_are_told_apart`` only drives the
        switching branch. A ``mode: none`` reader would then be told "Nothing
        refuses this anywhere ... the run comes back finite and confident" --
        false: measured, three rows with no switching reach
        ``NoiseWaveOperator._source_index``, which refuses at call time
        because ``coords.extra['receiver_input']`` is absent.
        """
        doc = preflight_document(model={**BASE_MODEL,
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        message = list(_gamma_rows(doc))[0].message
        assert "no switch index to choose a row with" in message
        assert "never used" not in message
        assert "finite and confident" not in message

    def test_the_mode_none_fix_keeps_both_of_its_halves(self):
        """Gutting the fix to "Write one row (schema §4.1.5), or declare
        switching." survived: only the "§4.1.5" substring was pinned, so the
        gloss that says what mode: none MEANS and the mode: cycle alternative
        could both vanish silently."""
        doc = preflight_document(model={**BASE_MODEL,
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        message = list(_gamma_rows(doc))[0].message
        assert ("schema §4.1.5: mode: none means no cal_loads and a single "
                "gamma_src row") in message
        assert "switching: {mode: cycle, order: [antenna, ...]}" in message

    def test_every_a15_finding_carries_its_check_tag(self):
        """Both branches, because they are two message bodies (Task 3's rule
        applied to a check with a fork in it). Kills a tag appended to the
        switching branch alone."""
        for observation in (switching(), BASE_OBSERVATION):
            doc = preflight_document(observation=observation,
                                     model={**BASE_MODEL, "cal_loads": loads(),
                                            "noise_wave": noise_wave(
                                                {"zeros": [7, 8]},
                                                {"zeros": [7, 8]})})
            found = list(_gamma_rows(doc))
            assert found, observation is switching()
            for finding in found:
                assert finding.message.endswith("(check A15).")


class TestNeitherCheckCanTakeThePassDown:
    """``_structural`` guarantees a section is PRESENT, never that it is a
    MAPPING (Task 4's carry-forward), and inside the pass a ``TypeError`` or a
    ``KeyError`` becomes "check RAISED" and discards every other finding.
    """

    @pytest.mark.parametrize("patch", [
        {"observation": 3},
        {"observation": {**BASE_OBSERVATION, "switching": 3}},
        {"observation": {**BASE_OBSERVATION, "switching": {"mode": []}}},
        {"observation": {**BASE_OBSERVATION,
                         "switching": {"mode": "cycle", "order": ORDER,
                                       "nope": 1}}},
        {"model": 3},
        {"model": {"kind": []}},
        {"observation": switching(), "model": {**BASE_MODEL,
                                               "noise_wave": 3}},
        {"observation": switching(),
         "model": {**BASE_MODEL, "noise_wave": [1, 2]}},
        {"observation": switching(),
         "model": {**BASE_MODEL,
                   "noise_wave": {"gamma_src_re": 3, "gamma_src_im": []}}},
        {"observation": switching(),
         "model": {**BASE_MODEL,
                   "noise_wave": {"gamma_src_re": {"zeros": 3},
                                  "gamma_src_im": {"zeros": []}}}},
        {"observation": switching(),
         "model": {**BASE_MODEL,
                   "noise_wave": {"gamma_src_re": {"zeros": [{}, 8]},
                                  "gamma_src_im": {"list": "rows"}}}},
        {"observation": switching(),
         "model": {**BASE_MODEL,
                   "noise_wave": {"full": 3, "normal": {"shape": 3},
                                  "stack": {}}}},
        {"inference": 3},
        {"inference": {"twin": 3}},
        {"inference": {"twin": {"replace": 3}}},
        {"inference": {"twin": {"replace": {"noise_wave": 3}}}},
        {"inference": {"twin": {"replace": {"noise_wave": []}}}},
        # A key the path grammar cannot spell, on both routes: without the
        # graph-node gate these reach `_check_where`, which raises OUTSIDE
        # the per-check `try` and takes the whole pass down.
        {"observation": switching(),
         "model": {**BASE_MODEL,
                   "bad-key": {"python": NOISE_WAVE_CLASS,
                               "gamma_src_re": {"zeros": [3, 8]},
                               "gamma_src_im": {"zeros": [3, 8]}}}},
        {"observation": switching(),
         "inference": {"twin": {"replace": {
             "bad-key": {"python": NOISE_WAVE_CLASS,
                         "gamma_src_re": {"zeros": [3, 8]},
                         "gamma_src_im": {"zeros": [3, 8]}}}}}},
        {"observation": switching(),
         "model": {**BASE_MODEL, "noise_wave": {"type": [], "gamma_src_re":
                                                {"zeros": [3, 8]}}}},
    ])
    def test_a_hostile_shape_reports_rather_than_raises(self, patch):
        """Kills every ``document["x"]``, ``spec[key]`` and ``shape[0]``
        written without asking what it is holding. The assertion is that the
        PASS answers at all: a check that raises here aborts it and takes
        Tasks 3's, 4's and 5's findings down with it.
        """
        report = preflight(preflight_document(**patch))
        assert isinstance(report.findings, tuple)

    def test_every_where_this_module_emits_is_a_document_path(self):
        """Task 2's ``_check_where`` runs OUTSIDE the per-check ``try``
        (Task 3's carry-forward, rule 1), so a `where` it refuses kills the
        pass rather than reporting the violation. Driving the whole pass on a
        document that fires all three of this module's `where` shapes at once
        is what tests them; calling the functions directly never reaches it.
        """
        wrong = {"zeros": [3, 8]}
        report = preflight(preflight_document(
            observation=switching(),
            model={**BASE_MODEL,
                   "bandpass": noise_wave(wrong, wrong,
                                          python=NOISE_WAVE_CLASS)},
            inference={"twin": {"replace": {
                "noise_wave": noise_wave(wrong, wrong)}}}))
        assert {"A14", "A15"} <= report.checks()
        assert {finding.where for finding in report.refusals()} >= {
            "model.cal_loads",
            "model.bandpass.gamma_src_re",
            "inference.twin.replace.noise_wave.gamma_src_im"}


class TestBothChecksReachThePass:
    def test_the_two_ids_are_bound_to_these_functions(self):
        """Kills a missing decorator, and a decorator carrying the wrong id --
        both of which leave every test above green while the pass itself never
        runs either check."""
        assert CHECKS["A14"] is _switch_order
        assert CHECKS["A15"] is _gamma_rows

    def test_a_document_that_violates_both_reports_both(self):
        """The pass COLLECTS (§2.3): a document with two problems must not cost
        the user two round trips. Kills a check that raises instead of
        returning findings, which would abort the pass and hide the second.
        """
        doc = preflight_document(observation=switching(),
                                 model={**BASE_MODEL,
                                        "noise_wave": noise_wave(
                                            {"zeros": [3, 8]},
                                            {"zeros": [3, 8]})})
        report = preflight(doc)
        assert {"A14", "A15"} <= report.checks()

    def test_the_base_document_earns_neither(self):
        """The fixture's own contract: a check that finds nothing on the base
        has actually looked. Kills a check whose trigger is inverted -- which
        every "is refused" test above would still pass."""
        assert not {"A14", "A15"} & preflight(preflight_document()).checks()

    def test_a_switch_order_wins_against_a_beam_that_cannot_be_read(self):
        # §5's PHASE PROPERTY, this task's one real assertion of it.  Task
        # 2's phase guard registers four synthetic lambdas: it proves the
        # HOOK's position and says nothing about any shipped check.  Nine
        # tasks each own one document that carries a real violation AND an
        # unreadable beam, and the assertion is symmetric -- the violation's
        # own words come back, and `no_such_beam` does NOT.
        from rheplicant.config.document import load_document
        from rheplicant.config.errors import ConfigError

        document = preflight_document(observation=switching(),
                                      resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "check A14" in str(caught.value)
        assert "no_such_beam" not in str(caught.value)

    def test_a_gamma_row_count_wins_against_the_same_beam(self):
        """The other half of the phase property, and the one the plan's thesis
        rests on: before this task the beam won on this document -- measured at
        ``740d9d1``, ``No file at 'no_such_beam.npy'``, with the row count
        reported only when the twin was evaluated.
        """
        from rheplicant.config.document import load_document
        from rheplicant.config.errors import ConfigError

        wrong = {"zeros": [3, 8]}
        document = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads(),
                   "noise_wave": noise_wave(wrong, wrong)},
            resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "check A15" in str(caught.value)
        assert "no_such_beam" not in str(caught.value)
