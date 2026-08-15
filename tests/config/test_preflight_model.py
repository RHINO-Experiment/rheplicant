"""``model:`` checks, decided from text (schema §6 A2-A8, A14, A30-A33).

Tasks 4, 5 and 11 each append a class here.  Every test builds its document
through ``tests/config/preflight_helpers.preflight_document`` -- the one place
a pre-flight document is built (plan §3) -- and asserts on a ``Report``.
Nothing here loads a document or reads a file except the two phase
assertions, whose whole subject is what ``load_document`` does with a beam it
cannot read.  ``build_model`` is called by several tests and **completes an
assembly in exactly one of them** -- ``test_lit_is_what_the_assembly_itself_
reports``, which holds ``_lit``'s claim about ``Assembly.lit`` against a real
assembly rather than against its own docstring; everywhere else the call is
expected to refuse, and comparing that refusal with the pass's is how §2.2's
two call sites are held to one implementation.  Either way it is the SECTION
under test, never the pass: no module under ``preflight/`` may build an
operator, and ``test_config_preflight.py`` enforces that mechanically.

**Why the section is driven as well as the pass.**  §2.2's rule is one name,
one binding, two call sites -- the shared pure function lives in
``sections/compose.py`` and ``sections/model.py``, the pass calls it with text
read off the raw document, and the build keeps calling it.  A test that only
drove the pass would leave the section free to grow a second implementation,
which is the ``_number``-vs-``_whole`` divergence on the 2C ledger.
"""

import re

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError, ResolutionContext
from rheplicant.config.paths import parse_path
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight.model import (
    _a14_cal_load_keys,
    _data_with_sources,
    _double_count,
    _graph_shape,
    _lit,
    _nodes,
    _tone_placement,
    _two_at_one_node,
)
from rheplicant.config.sections.compose import build_model, node_specs
from rheplicant.core.graph import AssemblyError
from rheplicant.radio import CWCalibrationOperator
from rheplicant.radio.graph import RADIO_GRAPH
from tests.config.preflight_helpers import (
    UNREADABLE_BEAM,
    only,
    preflight_document,
)

GAIN = {"gain": {"value": 1.1, "unit": "dimensionless"}}
#: The base document's own global signal, written out rather than read back
#: off ``BASE_MODEL``: half of Task 5's documents REPLACE ``model:`` outright
#: (see :func:`_model_only`), so the constant has to stand on its own, and a
#: reference into the helper's dict would be the same object every document in
#: the session shared.  Measured against ``preflight_helpers.BASE_MODEL`` --
#: ``GlobalSignalOperator``'s fields are ``centre``/``depth``/``width``, and
#: the ``amplitude``/``center_freq`` spelling this plan's draft used is
#: refused by ``build_model`` before any check here is reached.
GLOBAL_SIGNAL = {"depth": {"value": 0.5, "unit": "K"},
                 "centre": {"value": 75.0, "unit": "MHz"},
                 "width": {"value": 5.0, "unit": "MHz"}}
FOREGROUND = {"amplitude": {"value": 2500.0, "unit": "K"},
              "spectral_index": 2.55,
              "ref_freq": {"value": 70.0, "unit": "MHz"}}
SPILL = {"sky_fraction": {"value": 0.1, "unit": "dimensionless"},
         "t_ground": {"value": 300.0, "unit": "K"}}
PICKUP = {"t_ground": {"value": 300.0, "unit": "K"},
          "coupling": {"value": 0.05, "unit": "dimensionless"}}
FILTER = {"type": "FourierBandFilter", "axis": 0, "low": 0.02, "high": 0.5,
          "mode": "extract"}
SIGMA = {"value": 0.5, "unit": "K"}
LOAD = {"t_load": {"value": 300.0, "unit": "K"}}

#: Every model below refuses BEFORE a value node is resolved, so the context
#: needs no grids at all -- measured: all thirteen raise identically under
#: ``ResolutionContext(dtype="float32")`` and ``switch_order=()``.  A context
#: carrying grids would hide a check that had started resolving values.
BARE = ResolutionContext(dtype="float32")

#: The base document's own grids, for the one test that assembles a twin --
#: :meth:`TestTheReadersEveryModelCheckStartsFrom.
#: test_lit_is_what_the_assembly_itself_reports`, which holds :func:`_lit`
#: against what ``Assembly.lit`` reports.  Every other ``build_model`` call
#: here refuses before a value node is resolved and takes :data:`BARE`.  They
#: are ``preflight_helpers``' base ``observation:`` read back as arrays --
#: ``linspace(60, 85, 8) MHz`` and ``arange(0, 32, 2) s``.
GRIDDED = ResolutionContext(freq=jnp.linspace(60e6, 85e6, 8),
                            time=jnp.arange(0.0, 32.0, 2.0), dtype="float32")

#: The check ids this task decides.  A filter on THESE rather than on
#: ``refusals()`` outright: the pass runs every registered check, so an "and
#: nothing else" assertion written over the whole report would go red on
#: whichever later task first fires on one of these documents, naming this
#: module.
MINE = frozenset({"A2", "A3", "A4", "A6", "A7", "A14", "A32"})

#: (id, the model section, a WHOLE CLAUSE that must be IN the message).
#:
#: **The clause is the only pin here that can fail**, and it is why every one
#: of these is a clause rather than a phrase.  The equality assertion below
#: compares the PASS's message with the SECTION's -- and both come from the
#: same shared pure function, so it is ``f(x) == f(x)`` and cannot fail
#: however the message is reworded.  A fragment of two or three words ("is
#: reserved", "(SUM)") survives a rewrite that changed everything a reader
#: acts on.  Each clause below was MEASURED off the live refusal at
#: ``f303af8``, copied character for character.
MOVED = [
    ("A2", {"gian": GAIN},
     "model: 'gian' is not a node of graph 'single-antenna'; known nodes: ["),
    ("A3", {"astro_sum": {}, "gain": GAIN},
     "model.astro_sum: is a junction -- never an operator slot; it "
     "materializes automatically. The switch cycle is observation.switching."),
    ("A3", {"receiver_input": {}, "gain": GAIN},
     "model.receiver_input: is a selector -- never an operator slot; it "
     "materializes automatically. The switch cycle is observation.switching."),
    ("A4", {"beam": {"type": "GainOperator"}, "gain": GAIN},
     "model.beam: is reserved -- no shipped operator registers there; "
     "python: is the route."),
    ("A6", {"gain": [GAIN, GAIN]},
     "model.gain: this node holds a single instance; a list is the shape of "
     "a many node (foregrounds, t_sys_extra, cal_loads, filters)."),
    ("A6", {"foregrounds": FOREGROUND, "gain": GAIN},
     "model.foregrounds: is a non-empty list (SUM); got dict ("),
    ("A6", {"foregrounds": [], "gain": GAIN},
     "model.foregrounds: is a non-empty list (SUM); got list ([])."),
    ("A6", {"filters": FILTER, "gain": GAIN},
     "model.filters: is a non-empty list (CHAIN); got dict ("),
    ("A6", {"cal_loads": [GAIN], "gain": GAIN},
     "model.cal_loads: is a label-keyed mapping (FAN) -- the keys ARE "
     "observation.switching.order[1:], in that order; got list."),
    ("A7", {"noise": {"sigma": SIGMA}, "gain": GAIN},
     "model.noise: 2 classes register at this node (['NoiseOperator', "
     "'RadiometerNoiseOperator']); type: is required."),
    ("A7", {"filters": [{"axis": 0, "low": 0.02, "high": 0.5}],
            "gain": GAIN},
     "model.filters: 3 classes register at this node (['FourierBandFilter', "
     "'SiderealFilter', 'SkySpaceFilter']); type: is required."),
    ("A7", {"flagging": {"threshold": 3.0}, "gain": GAIN},
     "model.flagging: 2 classes register at this node (['FlaggingOperator', "
     "'MomentRFIFlaggingOperator']); type: is required."),
    ("A32", {"beam_spill": SPILL, "ground_pickup": PICKUP, "gain": GAIN},
     "model: beam_spill and ground_pickup both lit describe the ground twice "
     "(the spill term and the pickup term overlap); if that is deliberate, "
     "say so: acknowledge_double_count: true (check A32, decided as D-C13)."),
]

#: ``observation.switching`` declaring a three-position cycle, and the loads
#: that match it.  A14's late leg is the comparison between the two.
CYCLE = {"mode": "cycle", "order": ["antenna", "ambient", "hot"]}
LOADS = {"ambient": LOAD, "hot": {"t_load": {"value": 400.0, "unit": "K"}}}
#: The same two loads, written in the order the cycle does NOT declare.  A
#: stand-down test built on :data:`LOADS` earns no finding whether or not the
#: switch order was read; built on these it earns one the moment it is.
REVERSED_LOADS = {"hot": LOADS["hot"], "ambient": LOAD}

#: A shipped class relocated by ``python:`` -- the ONLY document route to a
#: relocation at all.  Measured: ``at:`` on an entry with no ``python:`` is
#: refused first, at ``compose.py:296-301`` ("at: places an operator that
#: declares no graph node of its own -- a python: operator"), so a
#: ``cw_tone:`` key carrying the shipped class cannot be moved and a check
#: keyed on ``model.cw_tone.at`` would guard a key no document can contain.
PY_GAIN = dict(GAIN, python="rheplicant.radio:GainOperator")
TONE = {"tone_freq": {"value": 70.0, "unit": "MHz"},
        "amplitude": {"value": 1.0, "unit": "K"},
        "line_width": {"value": 0.5, "unit": "MHz"},
        "python": "rheplicant.radio:CWCalibrationOperator"}
#: The same tone written the ordinary way, at its own node: no ``python:``,
#: so nothing relocates it.
BARE_TONE = {key: value for key, value in TONE.items() if key != "python"}
BANDPASS = {"bandpass": {"value": [1.0] * 8, "unit": "dimensionless"}}
#: A shipped class whose ``graph_node`` is neither of the tone's targets.
#: Written under a ``bandpass:`` key it lands at ``adc``, which is what makes
#: it the document where a key and its operator's node come furthest apart.
ADC = {"python": "rheplicant.radio:ADCOperator", "n_bits": 8,
       "scale": {"value": 1.0, "unit": "dimensionless"}}
SKY = {"amplitude": {"value": 10.0, "unit": "K"}}
DATA = {"zeros": ["n_time", "n_freq"]}
NOISE_STAGE = {"type": "NoiseOperator", "sigma": SIGMA}

#: The ids Task 5 decides.  Task 4's :data:`MINE` is its own; an "and nothing
#: else" assertion written over the whole report goes red on whichever later
#: task first fires on one of these documents.
T5_CHECKS = frozenset({"A5", "A8", "A31"})

#: "this key is not in the mapping at all", for a parametrize whose other
#: cells include ``None`` as a VALUE -- ``at: null`` and no ``at:`` are two
#: different documents and ``_single`` treats them alike only by accident of
#: its ``is not None`` test, which is the accident :func:`_t5_claims` mirrors.
_ABSENT = object()


def _with_data(document, data=DATA):
    """``observation.data``, added to a helper-built document by copy."""
    return {**document,
            "observation": {**document["observation"], "data": data}}


def _model_only(model):
    """A document whose ``model:`` is EXACTLY ``model``.

    ``preflight_document`` MERGES one level deep (§3.2(b)), and the base model
    lights ``global_signal`` AND ``uniform_sky`` -- two source nodes -- plus
    ``gain`` and ``noise``.  A31's negative cells and A8's absent-stage cell
    need those gone, and a merge cannot express a removal, so those tests
    replace the section outright rather than patch it.
    """
    return {**preflight_document(), "model": dict(model)}


def _t5_one(model, check):
    """The one finding ``check`` earns on a document patched with ``model``.

    ``only`` and not ``[0]`` of a filtered list (§3.2(b)): a check that fires
    TWICE on one document -- a loop over claims that forgot to stop at the
    first pair -- passes every ``in`` assertion and fails this one.
    """
    return only(preflight_document(model=model), check)


def _t5_refused(document, check):
    """``check``'s refusals on an already-built document, for the empty cases.

    Filtered by id rather than read off ``refusals()``, which is ordered
    across every registered check and is a function of how many tasks have
    landed rather than of this one.
    """
    return [one for one in preflight(document).refusals()
            if one.check == check]


def _findings(model, check=None):
    """This task's findings on a document whose ``model:`` carries ``model``.

    ``check=None`` means "everything Task 4 decides", not "everything the pass
    found": a later task firing on one of these documents would otherwise read
    as this module's defect.
    """
    report = preflight(preflight_document(model=model))
    return [one for one in report.refusals()
            if one.check == check or (check is None and one.check in MINE)]


def _observed(model, observation):
    """The same, with an ``observation:`` patch merged in as well."""
    report = preflight(preflight_document(model=model,
                                          observation=observation))
    return [one for one in report.refusals() if one.check in MINE]


def _ingested(model, switching):
    """The same again, over an INGESTED observation section.

    The section is REPLACED rather than merged: an ingested run declares no
    ``freq:`` and no ``time:`` -- the recording carries both, and
    ``build_observation`` refuses a document that declares either beside
    ``from_file:``.  A merged patch would leave the base document's grids
    there and the test would be asking its question of a document nobody
    could load.
    """
    document = preflight_document(model=model)
    document["observation"] = {
        "meta": {"telescope": "RHINO"},
        "from_file": {"format": "rhino_hdf5", "path": "obs.hd5f",
                      "freq_unit": "MHz", "settle_seconds": 0.0,
                      "thermistor_columns": {"antenna": 0, "ambient": 0,
                                             "hot": 1}},
        "switching": switching,
    }
    return [one for one in preflight(document).refusals()
            if one.check in MINE]


class TestTheGraphShapeChecks:
    """A2, A3, A4, A6, A7 -- moved in front of ``build_resources``."""

    @pytest.mark.parametrize("check, model, fragment", MOVED)
    def test_every_moved_message_is_the_sections_own_word_for_word(
            self, check, model, fragment):
        """The refactor's whole claim, as an assertion -- in TWO parts, and
        only the second one can fail.

        **The equality leg is a tautology and is kept anyway.**  The section
        and the pass both call the SAME shared pure function, so
        ``stripped == str(raised.value)`` is ``f(x) == f(x)``: it holds however
        the message is reworded, and it holds even if the message becomes
        nonsense.  What it DOES pin is the tail rule -- that the pass appends
        ``(check AN).`` and the section does not -- and that the two call sites
        have not been allowed to drift into two implementations, which is the
        ``_number``-vs-``_whole`` divergence on the 2C ledger.

        **The clause leg is the real guard.**  ``fragment`` is a whole clause
        measured off the live refusal, so a rewording that would send the
        reader to a different fix fails here.  A three-word fragment would
        not: "is reserved" survives a message that no longer names ``python:``
        as the route.

        A32 appends no tail, because its message already ends with its own
        citation.
        """
        with pytest.raises(ConfigError) as raised:
            build_model(dict(model), BARE, switch_order=())
        found = _findings(model, check)
        assert len(found) == 1, found
        tail = f" (check {check})."
        message = found[0].message
        stripped = (message[:-len(tail)] if message.endswith(tail)
                    else message)
        assert stripped == str(raised.value)
        assert fragment in message

    def test_an_unknown_node_id_is_named_before_its_kind_is_asked(self):
        # Kills a walk that asks graph.nodes[node_id].kind first: on 'gian'
        # that is a KeyError, which reaches the user as a traceback rather
        # than as this layer's sentence.  `where` is the section, not a node
        # path, because there is no node to point at.
        found = _findings({"gian": GAIN}, "A2")
        assert len(found) == 1
        assert found[0].where == "model"
        assert "'gian'" in found[0].message

    def test_two_unknown_node_ids_are_each_named_in_their_own_sentence(self):
        """Attribution, not presence.

        Both findings carry ``where == "model"`` -- there is no node to point
        at -- so the ONLY thing that distinguishes them is the id quoted in
        the message.  Kills a walk that reports ``next(k for k in specs if k
        not in graph.nodes)`` instead of the key it is standing on, under
        which a document with two typos is told about the first one twice and
        the second typo is invisible.  No other test here carries two unknown
        ids.
        """
        found = _findings({"gian": GAIN, "gaain": GAIN}, "A2")
        assert [one.where for one in found] == ["model", "model"]
        assert "'gian'" in found[0].message and "'gaain'" not in found[0].message
        assert "'gaain'" in found[1].message

    def test_two_placement_problems_both_arrive_and_the_build_raises_the_first(
            self):
        """Collect-versus-raise, and the ORDER, in one assertion.

        The pass exists so that a user with two mistakes sees two (§2.3), and
        the build still shows the first one in the document's own key order --
        it raised the first offending node before this refactor and must
        still.  Kills ``problems[-1][2]`` and ``sorted(problems)`` at the
        build, and ``problems[:1]`` in the pass, none of which any other test
        here can see: every other document carries exactly one problem.
        """
        model = {"gian": GAIN, "astro_sum": {}}
        found = _findings(model)
        assert [one.check for one in found] == ["A2", "A3"]
        with pytest.raises(ConfigError, match="'gian' is not a node"):
            build_model(dict(model), BARE, switch_order=())

    @pytest.mark.parametrize("node, kind", [("astro_sum", "junction"),
                                            ("receiver_input", "selector")])
    def test_a_junction_and_a_selector_are_named_by_their_own_kind(
            self, node, kind):
        # 2C's shape 1: `assert "never an operator slot" in msg` passes with
        # the two kinds swapped, and the reader is told a selector is a
        # junction.  Both cells pin the WORD.
        found = _findings({node: {}, "gain": GAIN}, "A3")
        assert len(found) == 1
        assert found[0].where == f"model.{node}"
        assert f"is a {kind} --" in found[0].message

    def test_a_reserved_node_is_refused_only_when_it_declares_a_type(self):
        """A4's trigger is ``type:``, and the bare form is a different
        refusal.

        Measured: ``model.beam: {}`` is refused inside ``build_node_operator``
        with "no shipped operator registers at this node", which has no §6 id
        and is NOT moved by this task (see the task body).  A check that
        dropped the ``"type" in spec`` condition would report A4 here and the
        reader would go looking for a ``type:`` they never wrote.
        """
        found = _findings({"beam": {"type": "GainOperator"}, "gain": GAIN},
                          "A4")
        assert len(found) == 1
        assert found[0].where == "model.beam"
        assert _findings({"beam": {}, "gain": GAIN}, "A4") == []

    @pytest.mark.parametrize("spec, wanted", [
        (3, "a node spec is a mapping"),
        ("type", "a node spec is a mapping"),
        (["type"], "this node holds a single instance"),
    ], ids=["int", "str", "list"])
    def test_a_reserved_node_whose_spec_is_no_mapping_is_not_asked_for_a_type(
            self, spec, wanted):
        """``"type" in spec`` on a spec that is not a mapping.

        Measured at ``f303af8``, BEFORE this task: ``model: {beam: 3}`` left
        ``build_model`` as a bare ``TypeError: argument of type 'int' is not
        iterable`` -- not this layer's voice at all -- and ``model:
        {beam: 'type'}`` earned A4 because ``'type' in 'type'`` is True, which
        is an accident of Python and not a rule anybody wrote.

        In the pass the first of those is worse than a bad message: a check
        that raises aborts the whole pass (§2.3's TRAP) and every other
        finding on the document is lost.  So the shared function asks
        ``isinstance(spec, Mapping)`` first, and each spelling now reaches the
        refusal that was always right for it -- the node-spec one for the two
        scalars, A6's "holds a single instance" for the list, which is a
        ``many``-shape question and never was A4's.
        """
        assert _findings({"beam": spec, "gain": GAIN}, "A4") == []
        with pytest.raises(ConfigError, match=wanted):
            build_model({"beam": spec, "gain": GAIN}, BARE, switch_order=())

    @pytest.mark.parametrize("model, wanted", [
        ({"gain": [GAIN, GAIN]}, "single instance"),
        ({"foregrounds": FOREGROUND, "gain": GAIN}, "(SUM)"),
        ({"filters": FILTER, "gain": GAIN}, "(CHAIN)"),
        ({"cal_loads": [GAIN], "gain": GAIN}, "(FAN)"),
    ])
    def test_each_many_shape_names_the_shape_it_wanted(self, model, wanted):
        # SUM/CHAIN/FAN swapped is 2C's shape 1 again: "is a non-empty list"
        # is true of both a SUM and a CHAIN, and the reader who is told SUM
        # at `filters` writes the entries in the wrong semantics.
        found = _findings(model, "A6")
        assert len(found) == 1
        assert wanted in found[0].message

    def test_each_shape_problem_names_its_own_node_in_where(self):
        """``where`` is the line to edit, and A6 has two of them on one
        document.

        Kills ``refuse("A6", "model", ...)`` -- which
        ``test_every_where_is_a_path_into_the_document`` accepts, because
        ``"model"`` is a document path -- and kills a hard-coded index.  A14's
        twin of this assertion was pinned from the first commit and A6's was
        not: one route closed, its twin open.
        """
        found = _findings({"gain": [GAIN], "foregrounds": FOREGROUND}, "A6")
        assert [one.where for one in found] == ["model.gain",
                                                "model.foregrounds"]

    def test_a_shape_problem_is_the_only_sentence_that_node_earns(self):
        """One mistake, one sentence -- the ``continue`` after A6.

        ``filters: {a: {...}}`` is a CHAIN given a mapping.  Without the
        ``continue`` the walk goes on to read that mapping as if it were the
        FAN form, asks its one value for a ``type:``, and adds A7 -- a second
        sentence demanding a key on a shape that cannot carry one.  The
        ``cal_loads`` twin of exactly this rule was pinned from the first
        commit (``test_a_load_mapping_of_the_wrong_shape_is_a_shape_problem_
        only``) and the ``filters`` one was not.
        """
        found = _findings({"filters": {"a": {"axis": 0}}, "gain": GAIN})
        assert [one.check for one in found] == ["A6"]

    def test_a_junction_earns_its_own_sentence_and_no_shape_sentence(self):
        # The same rule one clause up: `astro_sum: [GAIN]` is a junction AND a
        # list at a single node.  Deleting the junction/selector `continue`
        # adds A6 to A3 -- "this node holds a single instance" about a node
        # that holds no instance at all, because it is never an operator slot.
        found = _findings({"astro_sum": [GAIN], "gain": GAIN})
        assert [one.check for one in found] == ["A3"]

    def test_a_placement_problem_is_reported_before_a_shape_problem(self):
        """Run order, which ``raise_if_refused`` turns into WHICH sentence the
        user reads first.

        Kills hoisting the placement findings to the end of the walk: the pass
        would then open with A6 while ``build_model`` still raises A2, and the
        two phases would disagree about the same document.  The order test
        above carries two problems from the SAME loop and cannot see it.
        """
        model = {"gian": GAIN, "gain": [GAIN]}
        assert [one.check for one in _findings(model)] == ["A2", "A6"]
        with pytest.raises(ConfigError, match="'gian' is not a node"):
            build_model(dict(model), BARE, switch_order=())

    @pytest.mark.parametrize("model, check_fires", [
        ({"noise": {"sigma": SIGMA}, "gain": GAIN}, True),
        ({"noise": {"type": "NoiseOperator", "sigma": SIGMA}, "gain": GAIN},
         False),
        ({"filters": [{"axis": 0, "low": 0.02, "high": 0.5}], "gain": GAIN},
         True),
        ({"gain": GAIN}, False),
        # `type: None` WRITTEN OUT.  `ambiguous_class_problem` asks
        # `spec.get("type") is not None`, so this must fire exactly as the
        # absent key does -- and a version written `"type" not in spec`
        # would let a document silence A7 by declaring the key empty, then
        # die inside `_pick_class` after every beam had been read.  No other
        # cell here distinguishes the two spellings.
        ({"noise": {"type": None, "sigma": SIGMA}, "gain": GAIN}, True),
    ])
    def test_type_is_required_exactly_where_two_or_more_classes_register(
            self, model, check_fires):
        # Both halves: it fires at the three nodes with a choice (noise,
        # flagging, filters -- measured off operator_table()) and does NOT
        # fire at the twenty-odd with one, nor where `type:` decided it.
        assert bool(_findings(model, "A7")) is check_fires

    def test_a_many_node_sends_the_user_to_the_entry_and_not_the_node(self):
        # `where` is the line to edit.  A check that reported "model.filters"
        # for a three-entry chain leaves the reader to find which entry, and
        # the assertion that would have caught it is this one.
        found = _findings({"filters": [FILTER, {"axis": 0, "low": 0.02,
                                                "high": 0.5}],
                           "gain": GAIN}, "A7")
        assert len(found) == 1
        assert found[0].where == "model.filters[1]"

    @pytest.mark.parametrize("spec", [
        {"python": "rheplicant.radio:NoiseOperator", "sigma": SIGMA},
        {"from": "thermistors", "label": "ambient"},
    ], ids=["python", "from"])
    def test_a_python_or_from_entry_is_never_asked_for_a_type(self, spec):
        # `_pick_class` is not on either route (`build_node_operator` branches
        # to `_python_operator` and `_from_route` before it), so a check that
        # asked anyway would refuse a document the build accepts -- the one
        # direction a pre-flight pass must never be wrong in.
        assert _findings({"noise": spec, "gain": GAIN}, "A7") == []

    def test_a_composed_node_is_asked_at_its_stages_and_not_at_the_node(self):
        """``compose:`` is A7's twin route, and the obvious walk gets it
        backwards.

        Measured at ``f303af8``: a ``noise`` node composing two typed stages
        BUILDS, and one composing an untyped stage is refused by
        ``_pick_class`` with A7's own sentence.  A walk that asked
        ``ambiguous_class_problem`` about the ``{compose: ..., stages: [...]}``
        mapping itself sees no ``type:`` on it and refuses the first of those
        -- a document the build accepts, one phase earlier, which is the one
        direction this pass must never be wrong in.
        """
        typed = {"name": "b", "type": "NoiseOperator", "sigma": SIGMA}
        assert _findings({"gain": GAIN,
                          "noise": {"compose": "cascade",
                                    "stages": [dict(typed, name="a"),
                                               typed]}}, "A7") == []
        found = _findings({"gain": GAIN,
                           "noise": {"compose": "cascade",
                                     "stages": [{"name": "a",
                                                 "sigma": SIGMA}, typed]}},
                          "A7")
        assert len(found) == 1
        assert found[0].where == "model.noise.stages[0]"

    @pytest.mark.parametrize("stages", ["everything", 3, {"a": {}}, None],
                             ids=["a-string", "an-int", "a-mapping", "absent"])
    def test_a_compose_whose_stages_are_not_a_list_is_asked_nothing(
            self, stages):
        """The malformed branch of the same route, and the sentence there is
        the wrong one.

        ``_compose`` refuses the block before it builds any stage, so nothing
        on this document ever reaches ``_pick_class``.  A ``_t4_entries`` that
        fell back to ``[(node_id, spec)]`` asks the COMPOSING mapping for a
        ``type:`` and answers "model.noise: 2 classes register at this node;
        type: is required." -- and a reader who follows it writes ``type:``
        into a ``compose:`` block, which ``_compose`` refuses as an unknown
        key.  Wrong sentence, one phase early: deviation 2's failure mode
        reappearing on the branch deviation 2 did not take.

        A one-entry ``stages:`` is deliberately NOT here: that stage is real
        and really has no ``type:``, so A7 about it is true, and the build
        would say the same thing if a second stage were added.
        """
        spec = {"compose": "cascade"}
        if stages is not None:
            spec = {**spec, "stages": stages}
        assert _findings({"noise": spec, "gain": GAIN}, "A7") == []
        with pytest.raises(ConfigError, match="stages"):
            build_model({"noise": spec, "gain": GAIN}, BARE, switch_order=())

    def test_a_pipeline_model_declares_no_graph_nodes(self):
        # `kind: pipeline` has no node registry.  A `_nodes` that ignored
        # `kind:` would report `stages` as an unknown node id and refuse a
        # legal document.
        document = preflight_document()
        document["model"] = {"kind": "pipeline",
                             "stages": [{"name": "g", "type": "GainOperator",
                                         **GAIN}]}
        found = [one for one in preflight(document).refusals()
                 if one.check in MINE]
        assert found == []

    def test_a_section_level_key_is_not_read_as_a_node(self):
        # The one binding, asserted where it can be seen: `node_specs` is
        # what both callers use, and `acknowledge_double_count` is a
        # section-level key -- a second copy of that rule in preflight/ would
        # refuse it as an unknown node id while the build accepted it.
        model = {"beam_spill": SPILL, "ground_pickup": PICKUP, "gain": GAIN,
                 "acknowledge_double_count": True}
        assert set(node_specs(model)) == {"beam_spill", "ground_pickup",
                                          "gain"}
        assert _findings(model, "A2") == []


class TestTheReadersEveryModelCheckStartsFrom:
    """``_nodes`` and ``_lit``, which Tasks 5 and 11 import rather than
    rewrite (§3.1 rule 1).

    Neither has a caller in this task that could fail without them -- ``_lit``
    has none at all until Task 5's A8 -- so a defect in either is invisible
    to every other test in this module.  That is exactly the "a correct
    decision with no test" shape, and these are the tests that close it.
    """

    @pytest.mark.parametrize("document", [
        {},
        {"model": None},
        {"model": ["gain"]},
        {"model": {"kind": "pipeline", "stages": []}},
    ], ids=["no-model", "model-none", "model-not-a-mapping", "kind-pipeline"])
    def test_nodes_is_empty_for_every_shape_that_declares_none(self, document):
        # Kills `document["model"]` (a KeyError that aborts the pass on a
        # document `_structural` never sees -- Task 5 and Task 11 call these
        # readers directly), and kills a `_nodes` that hands a pipeline's
        # `stages:` back as if it were a node id.
        assert _nodes(document) == {}

    def test_nodes_drops_the_section_level_keys_and_keeps_the_rest(self):
        section = {"gain": GAIN, "kind": "graph",
                   "acknowledge_double_count": True}
        assert _nodes({"model": section}) == {"gain": GAIN}

    def test_a_kind_this_layer_does_not_know_declares_no_nodes_either(self):
        """``!= "graph"`` and not ``== "pipeline"``.

        ``kind: banana`` is ``build_model``'s own refusal ("kind: is 'graph'
        (the default) or 'pipeline'").  A ``_nodes`` written as "everything
        that is not a pipeline has nodes" reads the rest of that section as
        node ids and answers A2 about ``gian`` -- a true statement about a key
        the reader has no reason to look at, in front of the one fault that
        matters.  A wrong kind is refused by nobody in P-1 and that is the
        right answer.
        """
        document = preflight_document()
        document["model"] = {"kind": "banana", "gian": GAIN}
        assert _nodes(document) == {}
        assert [one for one in preflight(document).refusals()
                if one.check in MINE] == []

    @pytest.mark.parametrize("model, replace", [
        ({}, False),
        ({"cw_tone": {"python": "rheplicant:SnapshotOperator", "name": "tap",
                      "at": ["noise_wave", "cw_tone"]}}, False),
        ({"foregrounds": [FOREGROUND]}, False),
        ({"noise": {**PY_GAIN, "snapshot_before": "tap"}}, False),
        ({"global_signal": GLOBAL_SIGNAL, "noise": PY_GAIN}, True),
        ({"global_signal": GLOBAL_SIGNAL,
          "bandpass": dict(PY_GAIN, at=["gain"])}, True),
    ], ids=["the-base-model", "an-at-region", "a-many-node", "a-snapshot",
            "a-python-class-off-its-key", "a-one-element-at-list"])
    def test_lit_is_what_the_assembly_itself_reports(self, model, replace):
        """``_lit``'s docstring claims it reports what ``Assembly.lit`` will,
        so the claim is asserted against a real assembly rather than trusted.

        The region cell discriminates a ``_lit`` written as
        ``set(_nodes(document))``: measured, an ``at: [noise_wave, cw_tone]``
        region lights BOTH covered nodes, and the obvious reading misses
        ``noise_wave``, which is precisely the interior node A8's reachability
        question is about.

        **The last two cells are the ones that shipped wrong**, and they are
        why this test now replaces the model instead of only patching it: a
        merged patch cannot express "and no ``gain:`` key", which is what
        makes a relocation build rather than collide.  Measured at ``1556ff8``:
        ``{global_signal, noise: {python: …GainOperator}}`` assembles with
        ``lit == ('gain', 'global_signal')`` while ``_lit`` said ``noise``, and
        ``{global_signal, bandpass: {…, at: ['gain']}}`` assembles with ``lit
        == ('gain', 'global_signal')`` while ``_lit`` said ``bandpass``.  A
        shipped class does NOT land at the key it is written under -- that is
        the whole reason ``_t5_claims`` exists -- and ``_lit`` trusted the key.
        The three original cells cannot see it: none of them relocates
        anything off its key.

        The snapshot cell is the third: it is the only one whose answer comes
        from the ``snapshot_before:`` clause of ``_t5_claims`` RETURNING
        ``(key,)`` rather than merely standing down, so a guard rewritten to
        return ``()`` loses ``noise`` here and nowhere else.
        """
        document = (_model_only(model) if replace
                    else preflight_document(model=model))
        twin = build_model(document["model"], GRIDDED, switch_order=())
        assert _lit(document) == frozenset(twin.lit)

    def test_lit_drops_a_key_that_is_not_a_graph_node(self):
        # Kills `_lit` returning the keys outright: 'gian' is not a node, and
        # a caller asking "is `beam_spill` lit?" of a set that can contain
        # anything the user typed is asking a different question.
        assert "gian" not in _lit(preflight_document(model={"gian": GAIN}))

    def test_lit_does_not_read_an_at_out_of_a_fan_label(self):
        """A FAN label is a name the user chose, and one of them can be ``at``.

        ``cal_loads: {at: 'noise_wave'}`` under ``order: [antenna, at]`` is a
        load whose switch label is ``at``; the value is that load's spec.  A
        ``_lit`` that ran the ``at:`` reader over every node spec lit
        ``noise_wave`` -- a node nothing in the document places anything at,
        on a document the pass does not refuse and the build refuses for a
        different reason entirely.  §3.1 makes ``_lit`` the binding every
        model check in Tasks 4, 5 and 11 starts from, so lighting a node
        nothing lights is wrong three times over.
        """
        document = preflight_document(
            model={"cal_loads": {"at": "noise_wave"}},
            observation={"switching": {"mode": "cycle",
                                       "order": ["antenna", "at"]}})
        lit = _lit(document)
        assert "cal_loads" in lit
        assert "noise_wave" not in lit

    def test_lit_does_not_read_an_at_out_of_a_compose_block(self):
        # `_single` checks `compose:` BEFORE it pops `at:`, so a block
        # carrying both goes to `_compose`, which refuses `at` as an unknown
        # key ("compose: takes stages: and nothing else").  Nothing is placed
        # at those nodes on any path, so `_lit` must not report them.
        document = preflight_document(model={
            "noise": {"compose": "cascade", "at": ["noise_wave", "cw_tone"],
                      "stages": [{"name": "a", "type": "NoiseOperator",
                                  "sigma": SIGMA},
                                 {"name": "b", "type": "NoiseOperator",
                                  "sigma": SIGMA}]}})
        lit = _lit(document)
        assert "noise" in lit
        assert not ({"noise_wave", "cw_tone"} & lit)
        with pytest.raises(ConfigError, match="compose: takes stages:"):
            build_model(document["model"], BARE, switch_order=())

    @pytest.mark.parametrize("at, extra", [
        ("cw_tone", {"cw_tone"}),
        ("noise_wave", set()),
        (["cw_tone"], {"cw_tone"}),
        (["noise_wave", "cw_tone"], {"noise_wave", "cw_tone"}),
        (["noise_wave", "not_a_node"], {"noise_wave", "cw_tone"}),
        (["noise_wave", 4], set()),
        ({"node": "cw_tone"}, set()),
        (_ABSENT, set()),
    ], ids=["a-string-restating-the-key", "a-string-disagreeing-with-the-key",
            "a-one-element-list", "a-list", "a-list-with-a-stranger",
            "a-list-with-an-int", "a-mapping", "absent"])
    def test_lit_reads_an_at_claim_in_both_spellings_and_no_other(
            self, at, extra):
        """``at:`` is read by ``_lit``, by A5 and by A8 through ONE binding
        (:func:`_t5_claims`), so these cells are what say which shapes it
        honours -- and every expectation below was MEASURED against
        ``build_model`` rather than reasoned.

        Kills a reader that handles only the list form, one that iterates a
        bare string character by character (``'cw_tone'`` would light 'c',
        'w', '_'... none of which is a node), one that keeps a name that is
        not a node, and one that trusts a list of anything.

        **Four of these cells were wrong when this test first shipped**, and
        all four encoded the same false model -- that a key is lit whatever
        its ``at:`` says.  Measured at ``1556ff8``: the two malformed
        spellings and the absent one are documents ``_single`` and ``assemble``
        REFUSE (``at: is a node id or a list of node ids``; ``SnapshotOperator
        declares no graph_node and no At(...) wrapper was given``), so nothing
        is placed and lighting the key reports a placement nobody makes; and a
        string ``at:`` disagreeing with its key is ``_single``'s own refusal
        (``compose.py:303-308``), not check A5's.  The one-element LIST is the
        twin that shows the restatement rule is the STRING spelling's alone:
        ``refuse_misaddressed_region`` returns early below two nodes.

        The stranger cell keeps ``cw_tone`` because a region is where the
        document names nodes that no key does; that document is refused for
        its key, and :func:`_lit` does not make its answer depend on a refusal
        it is not making.
        """
        spec = {"python": "rheplicant:SnapshotOperator", "name": "tap"}
        if at is not _ABSENT:
            spec = {**spec, "at": at}
        lit = (_lit(preflight_document(model={"cw_tone": spec}))
               - _lit(preflight_document()))
        assert lit == extra


class TestTheDoubleCountAcknowledgement:
    """A32 / D-C13, moved and not rewritten."""

    @pytest.mark.parametrize("acknowledgement, refused", [
        (None, True), (True, False), ("yes", True), (1, True),
    ], ids=["absent", "true", "yes", "one"])
    def test_both_ground_terms_lit_need_the_acknowledgement(
            self, acknowledgement, refused):
        # `is True`, not truthiness: 'yes' and 1 are both truthy and neither
        # is the sentence the document has to contain.  The `1` cell is the
        # one that fails a `== True` implementation, because `1 == True`.
        model = {"beam_spill": SPILL, "ground_pickup": PICKUP, "gain": GAIN}
        if acknowledgement is not None:
            model = {**model, "acknowledge_double_count": acknowledgement}
        assert bool(_findings(model, "A32")) is refused

    @pytest.mark.parametrize("model", [
        {"beam_spill": SPILL, "gain": GAIN},
        {"ground_pickup": PICKUP, "gain": GAIN},
    ], ids=["spill-alone", "pickup-alone"])
    def test_one_ground_term_on_its_own_describes_the_ground_once(self, model):
        # Kills `or` for `and`, which refuses every document that models the
        # ground at all -- and `beam_spill` alone is the ordinary way to do
        # it.  No cell above can see that mutation: all of them light both.
        assert _findings(model, "A32") == []

    def test_the_message_cites_its_check_once(self):
        # This message ends "(check A32, decided as D-C13)." already.  A
        # `_double_count` written like `_graph_shape` -- append the tail --
        # produces a sentence citing A32 twice, which is how a reader learns
        # not to trust the citations.
        found = _findings({"beam_spill": SPILL, "ground_pickup": PICKUP,
                           "gain": GAIN}, "A32")
        assert len(found) == 1
        assert found[0].message.count("check A32") == 1
        assert found[0].message.endswith("(check A32, decided as D-C13).")
        assert found[0].where == "model"


class TestTheLoadKeysAgainstTheSwitchOrder:
    """A14's late leg -- §3.2 (i)'s, and it reads as Task 6's.

    Measured for this task: ``declared_order``
    (``sections/switching.py:36-55``) runs inside ``build_observation``, which
    ``document.py`` calls BEFORE ``build_resources``, so A14's first two legs
    already precede the beam and Task 6 hoists neither.  The one late leg is
    the comparison between ``model.cal_loads``' keys and
    ``switching.order[1:]``, which lives in ``_many`` -- this task's file --
    and runs at ``build_model``, one call after the beam.
    """

    def test_load_keys_out_of_order_are_refused_in_the_sections_own_words(
            self):
        found = _observed({"cal_loads": REVERSED_LOADS},
                          {"switching": CYCLE})
        assert len(found) == 1
        assert found[0].check == "A14"
        assert found[0].where == "model.cal_loads"
        with pytest.raises(ConfigError) as raised:
            build_model({"cal_loads": REVERSED_LOADS}, BARE,
                        switch_order=("antenna", "ambient", "hot"))
        assert found[0].message == f"{raised.value} (check A14)."
        assert "expected ['ambient', 'hot'], got ['hot', 'ambient']" in (
            found[0].message)

    def test_loads_declared_with_no_switch_cycle_at_all_are_refused(self):
        # The base document declares no `switching:`, so this is the shape a
        # user reaches by writing `cal_loads:` and forgetting the cycle.  A
        # check that only compared key ORDER would say nothing here, and the
        # load would silently have no switch index.
        found = _findings({"cal_loads": {"ambient": LOAD}}, "A14")
        assert len(found) == 1
        assert "declared without an observation.switching order" in (
            found[0].message)

    def test_loads_that_match_the_order_earn_nothing(self):
        assert _observed({"cal_loads": LOADS}, {"switching": CYCLE}) == []

    @pytest.mark.parametrize("switching", [
        {"mode": "cycle", "order": ["ambient", "hot"]},
        {"mode": "cycle", "order": ["antenna", "hot", "hot"]},
        {"mode": "cycle", "order": "antenna"},
        {"mode": "cycle"},
        {"mode": "spin", "order": ["antenna", "ambient", "hot"]},
        {"mode": [], "order": ["antenna", "ambient", "hot"]},
        {"mode": {"cycle": True}, "order": ["antenna", "ambient", "hot"]},
        {"order": ["antenna", "ambient", "hot"]},
        {"mode": "none", "order": ["antenna", "ambient", "hot"]},
        "cycle",
    ], ids=["antenna-not-first", "a-repeated-label", "order-not-a-list",
            "no-order", "an-unknown-mode", "an-unhashable-mode",
            "a-mapping-mode", "an-order-with-no-mode",
            "an-order-under-mode-none", "switching-not-a-mapping"])
    def test_an_order_this_layer_cannot_read_stands_down(self, switching):
        """Each of these is somebody else's refusal, and every one of them
        already fires before the beam.

        Measured: ``declared_order`` and ``compile_switching`` are reached
        from ``build_observation``, which ``document.py`` calls before
        ``build_resources``.  A check that guessed an order out of a malformed
        ``switching:`` would answer A14 about a cycle the document does not
        declare, and the reader would be sent to ``model.cal_loads`` to fix a
        typo in ``observation.switching``.

        **The loads are OUT OF ORDER against the order these blocks name**,
        which is what makes the cells discriminate at all: with matching loads
        every cell passes whether or not the block was read, and the version
        of this test that shipped first did exactly that.  Six of the ten now
        carry a READABLE ``order:`` beside their fault, so a check that reads
        the order anyway earns A14 here and fails.

        The two unhashable cells are the sharper ones: ``_KEYS.get(mode)`` on
        a list or a mapping is a ``TypeError``, which the pass reports as
        "check 'A14.cal_loads' RAISED" and which costs the document every
        other finding -- §2.3's TRAP, in the module whose own docstring is
        about it.
        """
        assert _observed({"cal_loads": REVERSED_LOADS},
                         {"switching": switching}) == []

    @pytest.mark.parametrize("loads, refused", [
        (LOADS, False),
        (REVERSED_LOADS, True),
    ], ids=["in-order", "out-of-order"])
    def test_an_ingested_run_declares_its_order_in_a_grammar_of_its_own(
            self, loads, refused):
        """The twin grammar, and reading `switching:` with one of them refuses
        a document that BUILDS.

        Measured at ``f303af8``, and found by
        ``test_config_section_model.py::TestThermistorsRoute::
        test_the_document_threads_the_recording_to_the_model_build`` rather
        than by reading: an ingested run declares ``switching: {order: [...]}``
        with **no** ``mode:``, because the recording carries the cycle
        (``observation.py:336-348``); every other run reaches
        ``compile_switching``, where an absent ``mode:`` means ``none`` and an
        ``order:`` beside it is an unknown key.  A single reading answers
        "declared without an observation.switching order" to a run whose order
        is right there, one phase before ``build_observation`` could disagree.

        The ``in-order`` cell is the one that fails that version; the
        ``out-of-order`` cell is what stops the fix from being "stand down on
        every ingested run", which would lose the check on exactly the
        documents ``cal_loads`` is written for.
        """
        found = _ingested({"cal_loads": loads},
                          {"order": ["antenna", "ambient", "hot"]})
        assert bool(found) is refused
        if refused:
            assert found[0].check == "A14"

    def test_an_ingested_run_that_declares_a_mode_is_someone_elses_refusal(
            self):
        # `observation.py:337-344`: an ingested run declares `order:` only,
        # and anything else is that clause's refusal -- which fires in
        # `build_observation`, before the beam.  A check that read the order
        # anyway would answer A14 about a document already refused for a
        # different reason, and the reader would edit `model.cal_loads`.
        assert _ingested({"cal_loads": REVERSED_LOADS},
                         {"mode": "cycle",
                          "order": ["antenna", "ambient", "hot"]}) == []

    def test_a_load_mapping_of_the_wrong_shape_is_a_shape_problem_only(self):
        # A6 and A14 are two sentences about one key, and the build says the
        # shape one first (`many_shape_problem` is called before the order
        # comparison in `_many`).  A pass that emitted both hands the user two
        # sentences for one mistake, the second of them about key ORDER in a
        # value that has no keys.
        found = _observed({"cal_loads": [GAIN]}, {"switching": CYCLE})
        assert [one.check for one in found] == ["A6"]

    def test_the_pass_and_the_build_refuse_the_same_documents(self):
        """One name, one binding, two call sites -- asserted as a table.

        The build is driven directly, so this fails if ``_many`` stops calling
        the shared function or if the pass grows a second reading of the
        switch order.
        """
        order = ("antenna", "ambient", "hot")
        for spec, switch_order, refused in [
                (LOADS, order, False),
                (REVERSED_LOADS, order, True),
                ({"ambient": LOAD}, order, True),
                ({"ambient": LOAD}, (), True)]:
            raised = None
            try:
                build_model({"cal_loads": spec}, BARE,
                            switch_order=switch_order)
            except ConfigError as error:
                raised = str(error)
            switching = ({"mode": "cycle", "order": list(switch_order)}
                         if switch_order else {"mode": "none"})
            found = _observed({"cal_loads": spec}, {"switching": switching})
            assert bool(found) is refused, (spec, switch_order, found)
            if refused:
                assert found[0].message == f"{raised} (check A14)."


class TestTheGraphChecksInThePass:
    """Registration and robustness -- the two ways a whole check dies quietly."""

    @pytest.mark.parametrize("slot", ["A2", "A3", "A4", "A6", "A7"])
    def test_every_id_the_graph_walk_decides_is_a_slot_it_claims(self, slot):
        """§3.1 binds ``_graph_shape`` to five ids, and claiming one of them
        is not enough.

        The registry is what makes a check id have ONE function: measured
        before this, ``register("A3")``, ``("A4")``, ``("A6")`` and ``("A7")``
        from a later module were all accepted, and a contradictory A3 would
        then have put two sentences about one check in one report with import
        order deciding which came first.  ``preflight`` de-duplicates by
        function identity, so five slots still run one walk.
        """
        assert CHECKS[slot] is _graph_shape

    def test_the_other_two_checks_own_their_slots_too(self):
        # A function written and never registered leaves every direct-call
        # test above green.  Subset form, per §3.1 rule 3 -- never `len(CHECKS)`
        # or an exact set, both of which are a function of how many tasks have
        # landed.
        assert CHECKS["A32"] is _double_count
        assert CHECKS["A14.cal_loads"] is _a14_cal_load_keys

    def test_the_graph_checks_reach_the_report_from_one_registration(self):
        # `_graph_shape` is registered under A2 and decides five ids.  This
        # one goes through `preflight` and asks for the ids by name, so a
        # module that is never imported at the foot of `preflight/__init__`
        # fails here rather than passing every direct-call test above.
        model = {"gian": GAIN, "astro_sum": {},
                 "beam": {"type": "GainOperator"}, "gain": [GAIN],
                 "noise": {"sigma": SIGMA},
                 "beam_spill": SPILL, "ground_pickup": PICKUP}
        fired = preflight(preflight_document(model=model)).checks()
        assert {"A2", "A3", "A4", "A6", "A7", "A32"} <= fired

    def test_a_malformed_model_produces_findings_and_never_raises(self):
        # §2.3's TRAP: a check that raises aborts the pass and hides every
        # later finding.  None of these specs is a mapping and one node id
        # is not a string.
        model = {"gain": None, "foregrounds": 3, "beam": "type",
                 "filters": "everything", 4: {}}
        report = preflight(preflight_document(model=model))
        assert report.refusals()

    def test_every_where_is_a_path_into_the_document(self):
        """``Finding.where`` is where the USER types (§3.1, rule 2), and it
        must be spellable by ``config/paths.py``.

        ``parse_path`` and not a ``startswith``: ``preflight._check_where``
        calls it OUTSIDE the per-check ``try``, so a ``where`` the path
        grammar cannot spell kills the whole pass instead of reporting the
        violation, and the user loses every other finding (Task 3 paid for
        this and its carry-forward says so).  Driven over the module's three
        checks, because ``where`` is built four different ways here -- the
        section, a node, a list entry and a compose stage.
        """
        model = {"gian": GAIN, "astro_sum": {}, "gain": [GAIN],
                 "noise": {"compose": "cascade",
                           "stages": [{"name": "a", "sigma": SIGMA},
                                      {"name": "b", "sigma": SIGMA}]},
                 "filters": [{"axis": 0}],
                 "beam_spill": SPILL, "ground_pickup": PICKUP,
                 "cal_loads": {"ambient": LOAD}}
        document = preflight_document(model=model)
        emitted = [*_graph_shape(document), *_double_count(document),
                   *_a14_cal_load_keys(document)]
        assert len(emitted) >= 6, emitted
        assert {"model.noise.stages[0]", "model.filters[0]"} <= {
            one.where for one in emitted}
        for finding in emitted:
            assert parse_path(finding.where)[0] == "model"

    @pytest.mark.parametrize("section", [["gain"], "gain", 3],
                             ids=["list", "str", "int"])
    def test_a_model_that_is_not_a_mapping_is_left_to_its_own_builder(
            self, section):
        """``_structural`` guarantees the SECTION is present, never that it is
        a mapping -- so every reader here needs its own guard, and
        ``_double_count``'s is one no other test touches.

        Measured: without it, ``section.get(...)`` on a list is an
        ``AttributeError``, which ``preflight`` reports as "check 'A32'
        RAISED" and which costs the document every other finding.  ``model:``
        being the wrong type is ``build_model``'s own refusal, with the type
        it got.
        """
        document = preflight_document(model=section)
        found = [one for one in preflight(document).refusals()
                 if one.check in MINE]
        assert found == []

    @pytest.mark.parametrize("section", ["one hour", ["freq"], 3],
                             ids=["str", "list", "int"])
    def test_an_observation_that_is_not_a_mapping_leaves_the_loads_alone(
            self, section):
        # The same hole one section over, and the one A14's leg reads: the
        # switch order comes off `observation:`, and a document whose
        # `observation:` is not a mapping is `build_observation`'s refusal --
        # which precedes the beam anyway.  Without the guard the pass dies
        # with "check 'A14.cal_loads' RAISED AttributeError".
        document = preflight_document(model={"cal_loads": {"ambient": LOAD}},
                                      observation=section)
        found = [one for one in preflight(document).refusals()
                 if one.check == "A14"]
        assert found == []

    @pytest.mark.parametrize("check, model, _fragment", MOVED)
    def test_every_message_ends_with_its_own_check_tag_once(
            self, check, model, _fragment):
        """The convention Task 3 shipped as a test over its own five checks,
        written for this module's.

        Both halves: the tail is THERE (a finding that cites no check leaves a
        reader with nothing to look up) and it is there ONCE (A32's message
        arrives with its own citation, so appending the tail to it cites A32
        twice -- and a `_double_count` copied from `_graph_shape` does exactly
        that).
        """
        found = _findings(model, check)
        assert len(found) == 1
        message = found[0].message
        assert re.search(r"\(check A\d+(, decided as D-C\d+)?\)\.$", message)
        assert message.count(f"check {check}") == 1

    def test_an_unknown_node_wins_against_a_beam_that_cannot_be_read(self):
        # §5's PHASE PROPERTY, this task's real assertion of it.  Task 2's
        # phase guard registers four synthetic lambdas: it proves the HOOK's
        # position and says nothing about any shipped check.
        #
        # Measured at be2027b, BEFORE the pass: this exact document refused
        # with "No file at 'no_such_beam.npy'" -- `build_resources` runs
        # before `build_model`.
        from rheplicant.config.document import load_document

        document = preflight_document(model={"gian": GAIN},
                                      resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "'gian' is not a node" in str(caught.value)
        assert "no_such_beam" not in str(caught.value)

    def test_the_load_keys_win_against_a_beam_that_cannot_be_read(self):
        # The same experiment for A14's late leg, which is the one check in
        # this task that was NOT in front of the beam by the pass alone: it
        # lives inside `_many`, one call after `build_resources`, and §3.2 (i)
        # assigns it here rather than to Task 6.
        from rheplicant.config.document import load_document

        document = preflight_document(
            model={"cal_loads": REVERSED_LOADS},
            observation={"switching": CYCLE},
            resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "the keys are switching.order[1:]" in str(caught.value)
        assert "no_such_beam" not in str(caught.value)


class TestASecondOperatorAtOneNode:
    """A5: the collision, from the keys and the ``at:`` claims."""

    def test_a_one_element_at_list_naming_another_key_is_refused(self):
        """The shape nothing refuses until the assembly.

        ``refuse_misaddressed_region`` returns early for a list shorter than
        two (``paths.py:318-319``), so a one-element ``at:`` naming another
        node passes every config check today and arrives as an
        ``AssemblyError`` after every beam has been read.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "bandpass": dict(PY_GAIN, at=["gain"])}
        assert _t5_one(model, "A5").where == "model.bandpass"

    def test_a_shipped_class_relocated_by_python_collides_at_its_own_node(
            self):
        # No `at:` at all: a `python:` operator lands at its CLASS's
        # graph_node and the key is ignored.  Measured -- this exact model is
        # the `Two operators provided for node 'gain'` AssemblyError quoted in
        # the task body, and a `_t5_claims` that credited the key instead
        # would find no collision here at all.
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "noise": PY_GAIN}
        assert _t5_one(model, "A5").where == "model.noise"

    def test_the_message_names_both_keys_and_the_node(self):
        # 2C's shape 1.  "two operators at one node" with the keys swapped
        # sends the reader to delete the entry that was right.  Three pins:
        # the node, the key that is second, and the key that was first.
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "bandpass": dict(PY_GAIN, at=["gain"])}
        message = _t5_one(model, "A5").message
        assert message.startswith("model.bandpass:")
        assert "node 'gain'" in message
        assert "model.gain already fills" in message

    def test_a_many_node_takes_every_operator_it_is_given(self):
        # `many` is what makes several placements legal.  Two entries under
        # ONE key: this is the shape that would fall to a check counting
        # entries rather than claimants.
        model = {"foregrounds": [FOREGROUND, FOREGROUND], "gain": GAIN}
        assert _t5_refused(preflight_document(model=model), "A5") == []

    def test_two_keys_claiming_one_many_node_is_what_many_means(self):
        """The only cell in this class that reaches A5's ``many`` clause.

        ``foregrounds`` is a MANY node claimed twice -- once by its own key
        and once by a relocated operator -- so this is a real collision of
        claims and not a collision of a node.  Kills ``if len(claimants) > 1:
        refuse`` written without the ``many`` guard, which passes every other
        cell here (all of them single nodes) and refuses the commonest
        multi-source sky in the repository.
        """
        model = {"foregrounds": [FOREGROUND],
                 "gain": dict(PY_GAIN, at=["foregrounds"])}
        assert _t5_refused(preflight_document(model=model), "A5") == []

    @pytest.mark.parametrize("key, at", [("noise", ["gain", "noise"]),
                                         ("bandpass", ["gain", "noise"])])
    def test_a_multi_node_region_is_left_to_the_regions_own_refusals(
            self, key, at):
        """Two claims of the same node is not always check A5.

        Measured, these two documents get ``Node 'gain' is claimed both by the
        region ('gain', 'noise') ...`` and ``A multi-node at: region covering
        ['gain', 'noise'] is written under the key 'bandpass' ...`` -- A47's
        wording and ``_check_disjoint_claims``'.  A5 answering here would name
        the wrong fix in a better voice.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 key: dict(PY_GAIN, at=at)}
        assert _t5_refused(preflight_document(model=model), "A5") == []

    def test_a_snapshot_pins_its_operator_to_the_key_it_is_written_under(self):
        """``snapshot_before:`` overrides a ``python:`` class's own node, and
        the plan's draft did not know it.

        Measured at ``48b359d``: ``_single`` returns ``At(node_id,
        Pipeline(SnapshotOperator, operator))`` for a spec carrying
        ``snapshot_before:`` (``compose.py:319-326``), so a ``python:
        GainOperator`` written under ``noise:`` with a snapshot lands at
        ``noise`` -- and the document BUILDS, with ``gain`` filled once.  A
        ``_t5_claims`` that read ``python:`` first credits it to ``gain``,
        finds a second claimant there, and refuses a document that assembles
        perfectly: the one direction the task body says a pre-flight pass must
        never be wrong in.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "noise": {**PY_GAIN, "snapshot_before": "tap"}}
        assert _t5_refused(preflight_document(model=model), "A5") == []
        twin = build_model(dict(model), BARE, switch_order=())
        assert {"gain", "noise"} <= set(twin.lit)

    def test_an_at_beside_a_snapshot_claims_nothing_at_all(self):
        # `_single` refuses the PAIR outright (`compose.py:289-294`), so
        # nothing is placed anywhere and neither the `at:` node nor the key is
        # claimed.  A check that took the `at:` answers A5 about a collision
        # the build never gets far enough to have.
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "noise": {**PY_GAIN, "at": ["gain"],
                           "snapshot_before": "tap"}}
        assert _t5_refused(preflight_document(model=model), "A5") == []
        with pytest.raises(ConfigError, match="at: and snapshot_before:"):
            build_model(dict(model), BARE, switch_order=())

    def test_a_compose_block_places_at_its_key_whatever_its_at_says(self):
        """``compose:`` is read BEFORE ``at:``, in the section and here.

        ``_single`` dispatches to ``_compose`` on ``compose`` alone, and
        ``_compose`` refuses ``at`` as an unknown key -- measured, ``model.
        noise: compose: takes stages: and nothing else; got ['at'] too.``  A
        ``_t5_claims`` that read the ``at:`` off a composing block credits
        ``cw_tone`` to ``model.noise``, collides with the document's own
        ``cw_tone:`` key, and hands the reader A5's sentence for a document
        whose fault is an unknown key in a ``compose:`` block.  Task 4's
        ``_lit`` closed this hole for lighting; this is its twin for claiming.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "cw_tone": BARE_TONE,
                 "noise": {"compose": "cascade", "at": ["cw_tone"],
                           "stages": [dict(NOISE_STAGE, name="a"),
                                      dict(NOISE_STAGE, name="b")]}}
        assert _t5_refused(preflight_document(model=model), "A5") == []
        with pytest.raises(ConfigError, match="compose: takes stages:"):
            build_model(dict(model), BARE, switch_order=())

    def test_a_fan_label_spelled_at_claims_no_node_of_its_own(self):
        """A FAN label is a name the user chose, and one of them can be ``at``.

        ``cal_loads`` is a MANY node, so ``at`` here is a switch LABEL and its
        value is that load -- Task 4 measured the same trap in ``_lit`` and
        shipped ``test_lit_does_not_read_an_at_out_of_a_fan_label``.  Without
        the ``many`` guard this document claims ``noise_wave`` twice and earns
        a refusal naming a node nothing places anything at.
        """
        model = {"cal_loads": {"at": "noise_wave"}, "noise_wave": {}}
        document = preflight_document(
            model=model,
            observation={"switching": {"mode": "cycle",
                                       "order": ["antenna", "at"]}})
        assert _t5_refused(document, "A5") == []
        with pytest.raises(ConfigError, match="a node spec is a mapping"):
            build_model(dict(model), BARE, switch_order=("antenna", "at"))

    def test_a_key_that_is_not_a_graph_node_places_nothing_to_collide_with(
            self):
        """The first bullet of ``_t5_claims``, pinned rather than reasoned.

        Measured: ``build_model`` refuses an unknown node id outright, so
        nothing an entry under it declares is ever constructed -- however
        convincing its ``python:`` looks.  A ``_t5_claims`` that resolved the
        class anyway credits ``gain``, finds ``model.gain`` there and hands
        the reader A5 for a document whose only fault is a typo in the key.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "gian": PY_GAIN}
        assert _t5_refused(_model_only(model), "A5") == []
        with pytest.raises(ConfigError, match="'gian' is not a node"):
            build_model(dict(model), BARE, switch_order=())

    @pytest.mark.parametrize("target", [
        "rheplicant:SnapshotOperator",
        "rheplicant.core.operator:GainOperator",
        "myproject.radio:GainOperator",
        "rheplicant.radio:GainOperator:extra",
        "rheplicant.radio",
    ], ids=["a-name-radio-does-not-export", "radios-name-off-another-module",
            "a-module-whose-name-ends-in-radio", "two-colons", "no-colon"])
    def test_a_python_target_this_layer_will_not_import_claims_nothing(
            self, target):
        """The trap in the task body's table, as an assertion.

        A ``python:`` target this layer will not import lands at ITS OWN
        class's ``graph_node``, which text cannot resolve -- so crediting the
        KEY with the claim invents a collision on a document that assembles.

        The last four cells defend the parts of the guard the first cannot
        see, and each fails a DIFFERENT loosening of it.  Measured,
        ``SnapshotOperator`` is not in ``rheplicant.radio.__all__``, so a
        ``_t5_radio_class`` that dropped the module test entirely still
        answers ``None`` for cell one.  ``GainOperator`` IS exported by
        ``rheplicant.radio``, so cell two kills the DROPPED module test and
        cell three kills a LOOSENED one (``module.endswith("radio")`` would
        resolve someone else's ``myproject.radio`` to rheplicant's class).
        Cells four and five are the ``count(":") != 1`` clause; both are
        equivalent mutants against the ``__all__`` guard -- ``'GainOperator:
        extra'`` and ``''`` are neither of them exported names -- and they are
        here as the shapes, not as discriminating cells.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "noise": {"python": target, "name": "tap"}}
        assert _t5_refused(preflight_document(model=model), "A5") == []

    def test_the_assemblys_own_refusal_is_still_the_backstop(self):
        """This task adds a refusal; it removes none.

        ``_place_at_node`` (``core/graph.py:910-917``) is what catches the
        collisions text cannot see -- a ``python:`` target outside
        ``rheplicant.radio``, which this layer will not import.  Deleting it
        because "preflight covers it now" leaves those documents building two
        operators onto one node.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "noise": PY_GAIN}
        with pytest.raises(AssemblyError, match="Two operators provided"):
            build_model(dict(model), BARE, switch_order=())

    def test_the_blame_does_not_move_when_the_keys_are_reordered(self):
        """Attribution, not presence -- and the thing that broke it was
        document order.

        Measured on the shipped version: the SAME collision named
        ``model.noise`` when the relocation was written last and
        ``model.gain`` when it was written first, because the message took
        ``keys[1]``.  The second of those sends the reader to delete
        ``model.gain`` -- the entry that is at its own node and the only one
        that is right.  The occupant is knowable from the placements
        (``claim == key``), so this is order-independent by construction.
        """
        for model in ({"gain": GAIN, "noise": PY_GAIN},
                      {"noise": PY_GAIN, "gain": GAIN}):
            found = only(_model_only({"global_signal": GLOBAL_SIGNAL,
                                      **model}), "A5")
            assert found.where == "model.noise"
            assert "model.gain already fills" in found.message

    def test_a_node_claimed_three_times_names_every_entry_that_must_move(self):
        """Cardinality, and the two-round fix this task criticised elsewhere.

        Measured: this document is one ``AssemblyError`` ("Two operators
        provided for node 'gain'") and the user has TWO entries to move.  A
        check emitting one finding per NODE names one of them, the user fixes
        it and comes back for the other; ``keys[0], keys[-1]`` is worse still
        -- it names the last and ``model.bandpass`` never appears at all.  One
        finding per intruder, each at its own ``where``.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "bandpass": dict(PY_GAIN, at=["gain"]),
                 "emi": dict(PY_GAIN, at=["gain"])}
        found = _t5_refused(_model_only(model), "A5")
        assert [one.where for one in found] == ["model.bandpass", "model.emi"]
        assert all("model.gain already fills" in one.message for one in found)
        with pytest.raises(AssemblyError, match="Two operators provided"):
            build_model(dict(model), BARE, switch_order=())

    def test_a_snapshot_written_empty_is_no_snapshot_and_the_at_stands(self):
        """``snapshot_before: null`` and no ``snapshot_before:`` are one
        document to ``_single``, which pops it and tests ``is not None``.

        Measured: this really does place at ``gain`` -- the assembly answers
        "Two operators provided for node 'gain'" -- so the ``at:`` is honoured
        and A5 must fire.  Kills ``"snapshot_before" in spec``, which drops
        the claim here and loses the check on a document that collides.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "noise": {**PY_GAIN, "at": ["gain"],
                           "snapshot_before": None}}
        assert only(_model_only(model), "A5").where == "model.noise"
        with pytest.raises(AssemblyError, match="Two operators provided"):
            build_model(dict(model), BARE, switch_order=())

    def test_a_relocation_onto_an_EMPTY_node_is_not_a_collision(self):
        """A5 counts occupants, and a lone relocation has none to collide
        with.

        Measured: ``{global_signal, noise: {python: ...GainOperator}}``
        BUILDS, with ``lit == ('gain', 'global_signal')`` -- the document has
        one operator and it is at ``gain``.  Kills an occupant read as "the
        node itself, always", under which the single claimant is its own
        intruder and every relocation is refused whether anything is in its
        way or not.  Every other cell in this class writes the collision, so
        none of them can see it.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "noise": PY_GAIN}
        assert _t5_refused(_model_only(model), "A5") == []
        twin = build_model(dict(model), GRIDDED, switch_order=())
        assert set(twin.lit) == {"gain", "global_signal"}

    def test_a_key_whose_spec_is_no_mapping_fills_nothing_to_collide_with(
            self):
        """A ``many`` node's list places at its key; a single node's does not.

        Measured: ``_single`` refuses ``gain: null`` with "a node spec is a
        mapping", so the node is empty and the relocation onto it is the only
        operator there.  A ``_t5_claims`` that let this share the ``many``
        clause's ``return (key,)`` tells the reader that ``model.gain``
        "already fills" a node whose spec the build is about to reject -- and
        this document's real fault is the ``null``.  Found by a mutation round
        after the reviewer's, so it is here as a cell rather than as a claim.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": None,
                 "noise": PY_GAIN}
        assert _t5_refused(_model_only(model), "A5") == []
        with pytest.raises(ConfigError, match="a node spec is a mapping"):
            build_model(dict(model), BARE, switch_order=())

    @pytest.mark.parametrize("spec, wanted", [
        (dict(PY_GAIN, at="gain"), "restates its own key"),
        (dict(GAIN, at=["gain"]), "declares no graph node of its own"),
    ], ids=["a-string-at-disagreeing-with-its-key", "an-at-with-no-python"])
    def test_an_at_single_refuses_more_precisely_than_a5_could(
            self, spec, wanted):
        """``_single`` names the real fault; A5 would name a generic one.

        Both of these are ``at:`` shapes ``_single`` refuses OUTRIGHT, so no
        operator is placed and there is no second occupant.  A ``_t5_claims``
        that took the ``at:`` whenever it parsed would displace a sentence
        about the ``at:`` itself with "puts a second operator at node 'gain'",
        which sends the reader to compose two entries that were never both
        going to be placed.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "bandpass": spec}
        assert _t5_refused(_model_only(model), "A5") == []
        with pytest.raises(ConfigError, match=wanted):
            build_model(dict(model), BARE, switch_order=())

    def test_an_at_naming_a_node_the_graph_does_not_have_never_raises(self):
        """§2.3's TRAP: ``graph.nodes[node]`` on a name out of the document.

        Two entries relocated onto the same stranger is what makes it a
        CLAIM rather than a stray: without the ``node not in graph.nodes``
        guard the walk reaches ``graph.nodes['nope'].many`` and the pass dies
        with "check 'A5' RAISED KeyError".

        The ``gian`` typo is the anti-vacuity leg and the cost: nothing in P-1
        refuses an ``at:`` naming a stranger -- A2 is about KEYS -- so the
        finding a raising A5 would take down with it is a fault about a
        DIFFERENT line, which the reader would never see.
        """
        document = _model_only({"global_signal": GLOBAL_SIGNAL, "gian": GAIN,
                                "bandpass": dict(PY_GAIN, at=["nope"]),
                                "emi": dict(PY_GAIN, at=["nope"])})
        assert _t5_refused(document, "A5") == []
        assert "A2" in preflight(document).checks()


class TestWhereTheToneMayGo:
    """A8: strictly after, and the displacement that is a different error."""

    def test_the_tone_at_its_own_node_is_never_a_violation(self):
        # The ordinary document.  A check that fired on the presence of a
        # cw_tone rather than on its placement refuses every calibrated model
        # in the repository.
        model = {"global_signal": GLOBAL_SIGNAL, "cw_tone": BARE_TONE,
                 "bandpass": BANDPASS, "gain": GAIN}
        assert _t5_refused(preflight_document(model=model), "A8") == []

    @pytest.mark.parametrize("node", ["noise", "emi"])
    def test_the_tone_downstream_of_the_bandpass_is_refused_in_its_own_words(
            self, node):
        """``must_precede_because`` verbatim, and BOTH blocked targets named.

        Paraphrasing the operator's sentence is how the config layer and the
        package start telling a reader two different physics stories; the
        assertion is the string itself.  The second half is 2C's shape 1: a
        message naming only 'bandpass' leaves the reader believing 'gain' is
        fine.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": BANDPASS,
                 "gain": GAIN, node: dict(TONE, at=node)}
        found = _t5_one(model, "A8")
        assert found.where == f"model.{node}"
        assert CWCalibrationOperator.must_precede_because in found.message
        assert "['bandpass', 'gain']" in found.message
        # The class's own home, not the node it was found at: `{node!r}` in
        # that slot would read "its own node is 'emi'", which is where it
        # already is.  The displacement message's `cw_tone` is pinned three
        # cells down; this is its twin.
        assert "'cw_tone'" in found.message
        assert "lights them" in found.message

    def test_the_tone_in_the_bandpass_slot_is_a_different_refusal(self):
        """§2.6 item 1: displacement is not lateness and needs its own words.

        A message telling this reader 'bandpass' is "not reachable" names a
        fix that is not the one -- there IS no bandpass, because the tone is
        standing in it.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "bandpass": dict(TONE, at="bandpass")}
        found = _t5_one(model, "A8")
        assert "IN the 'bandpass' slot" in found.message
        assert "cannot be reached" not in found.message
        assert "'cw_tone'" in found.message

    def test_the_tone_in_the_gain_slot_is_the_same_refusal_as_in_bandpass(
            self):
        """2C's shape 4: a hole closed on one route and left open on its twin.

        ``must_precede`` is ``("bandpass", "gain")`` -- TWO nodes -- and §2.6
        item 1's "displaces the stage" reasoning applies to each of them
        identically.  Every other displacement cell here puts the tone in the
        ``bandpass`` slot, so a check written ``if node == "bandpass"`` passes
        them all.  Pinned on the message, because the DISPLACEMENT refusal and
        the "cannot be reached" one are different sentences naming different
        fixes.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": BANDPASS,
                 "gain": dict(TONE, at="gain")}
        found = _t5_one(model, "A8")
        assert "IN the 'gain' slot" in found.message
        assert "cannot be reached" not in found.message

    def test_a_one_element_at_list_relocates_exactly_as_the_string_does(self):
        """``at:``'s twin spelling, which every other A8 cell here skips.

        Measured: ``refuse_misaddressed_region`` returns early below two nodes
        (``paths.py:318-319``) and ``_single`` builds ``At(('bandpass',), op)``
        -- the same placement the string form makes, and the document builds
        just as silently.  A ``_t5_claims`` reading only ``isinstance(at, str)``
        loses A8 on this whole spelling and every cell above stays green.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "bandpass": dict(TONE, at=["bandpass"])}
        assert "IN the 'bandpass' slot" in _t5_one(model, "A8").message
        twin = build_model(dict(model), BARE, switch_order=())
        assert "cw_tone" not in twin.lit

    def test_a_multi_node_region_carrying_the_tone_is_left_to_the_assembly(
            self):
        """A region is A47's and ``_check_disjoint_claims``', here as in A5.

        Measured: ``at: ['noise', 'emi']`` under ``emi:`` really is refused,
        by ``_check_ordering`` reading the region's LAST node -- so this is a
        deliberate narrowing rather than a hole nobody noticed, and the
        assembly is still the backstop.  A ``_t5_claims`` returning the whole
        region would make A8 answer here in a voice that names one node when
        the document names two.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": BANDPASS,
                 "gain": GAIN, "emi": dict(TONE, at=["noise", "emi"])}
        assert _t5_refused(preflight_document(model=model), "A8") == []
        with pytest.raises(AssemblyError, match="must_precede"):
            build_model(dict(model), BARE, switch_order=())

    def test_the_document_this_refuses_builds_silently_today(self):
        """The hole, as an assertion rather than a claim in a plan.

        Measured at ``48b359d``: this assembles, ``lit`` carries 'bandpass'
        and no 'cw_tone', and the receiver response is gone from the model.
        When the package starts refusing it this test goes red -- which is the
        signal that §2.6 item 1's decision can be revisited.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "bandpass": dict(TONE, at="bandpass")}
        twin = build_model(dict(model), BARE, switch_order=())
        assert "bandpass" in twin.lit
        assert "cw_tone" not in twin.lit

    def test_an_absent_stage_is_still_not_a_violation(self):
        """``fold.py:271-274``'s reasoning, kept.

        The tone at ``adc`` with neither target lit is refused by nothing --
        here or in the package.  ``fold.py:271-274``, verbatim: "A constraint
        is checked only against nodes that are LIT. An absent node contracts
        to identity, so there is no bandpass to pass through and nothing to
        violate -- refusing there would reject a sky-only assembly for the
        sake of a stage it never asked for."  A check that ignored ``lit``
        would do exactly that.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "adc": dict(TONE, at="adc")}
        assert _t5_refused(_model_only(model), "A8") == []

    def test_the_tone_upstream_of_both_targets_is_reachable(self):
        """A8's whole reachability rule, which nothing else here defends.

        The tone sits at its OWN node, upstream of both ``bandpass`` and
        ``gain``, and that is the legal placement.  Kills ``_t5_downstream``
        returning ``set()`` -- under which every tone is "upstream of
        nothing", every negative cell in this class still passes, and the
        check silently refuses the correct document.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": BANDPASS,
                 "gain": GAIN, "cw_tone": dict(TONE, at="cw_tone")}
        assert _t5_refused(preflight_document(model=model), "A8") == []

    def test_a_region_lights_the_nodes_it_covers_but_does_not_key(self):
        """Defends **Task 4's** ``_lit`` ``at:`` union, whose only observer is
        A8.

        A region ``at: [bandpass, gain, noise]`` lights all three even though
        the model has no ``bandpass:`` key, so a tone downstream of it is
        refused and the message must name both covered targets.  Kills ``_lit``
        reading the model's KEYS alone -- a mutation nothing in Task 4's own
        class can see, because every A2/A3/A4/A6/A7 cell keys the node it
        lights.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                 "noise": dict(PY_GAIN, at=["bandpass", "gain", "noise"]),
                 "emi": dict(TONE, at="emi")}
        assert "['bandpass', 'gain']" in _t5_one(model, "A8").message

    def test_a_key_holding_a_relocated_operator_does_not_light_its_own_node(
            self):
        """The document A8 refused while it BUILDS -- ``_lit``'s defect, seen
        from A8.

        ``bandpass:`` here holds an ``ADCOperator``, which lands at ``adc``.
        Measured at ``1556ff8``: the assembly BUILDS with ``lit == ('adc',
        'emi', 'global_signal')`` -- neither ``bandpass`` nor ``gain`` is lit,
        so ``_check_ordering`` sees no violation and is right not to.  A8
        refused it anyway, saying the document "lights" a bandpass it does
        not, because ``_lit`` credited the KEY.  A refusal that asserts a fact
        about the document that is false is worse than a missing check.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": ADC,
                 "emi": dict(TONE, at="emi")}
        assert _t5_refused(_model_only(model), "A8") == []
        twin = build_model(dict(model), GRIDDED, switch_order=())
        assert set(twin.lit) == {"adc", "emi", "global_signal"}

    def test_only_the_targets_the_document_really_lights_are_named(self):
        """The same root cause where the check should still fire, and the
        singular wording that no other cell renders.

        ``bandpass:`` holds a ``GainOperator``, which lands at ``gain``: so
        ``gain`` IS lit, ``bandpass`` is NOT, and the package's own refusal
        names ``'gain'`` alone.  Before the fix A8 said ``['bandpass',
        'gain']`` -- half of it false.  This is also the only cell that
        reaches the ``len(blocked) == 1`` branch, so it kills a message
        hard-coded to the plural.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": PY_GAIN,
                 "emi": dict(TONE, at="emi")}
        found = only(_model_only(model), "A8")
        assert "['gain']" in found.message
        assert "bandpass" not in found.message
        assert "lights it" in found.message
        with pytest.raises(AssemblyError, match="'gain' is not reachable"):
            build_model(dict(model), GRIDDED, switch_order=())

    def test_a_string_at_disagreeing_with_its_key_is_singles_refusal(self):
        """A8 would answer with a fix that is a no-op.

        Measured: ``_single`` refuses this with *"at: 'noise' disagrees with
        the node key -- a single-node at: restates its own key"*
        (``compose.py:303-308``).  A8 arriving first would say "Place it
        upstream: its own node is 'cw_tone'" -- and the key already IS
        ``cw_tone``, so the reader is told to do what the document does.  The
        real fault is the ``at:``.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": BANDPASS,
                 "gain": GAIN, "cw_tone": dict(TONE, at="noise")}
        assert _t5_refused(_model_only(model), "A8") == []
        with pytest.raises(ConfigError, match="restates its own key"):
            build_model(dict(model), GRIDDED, switch_order=())

    def test_the_tones_own_key_can_still_carry_a_relocating_list(self):
        """The twin of the cell above, and the reason the check is keyed on
        ``must_precede`` rather than on any key name.

        A one-element ``at:`` LIST is not held to the restatement rule, so
        ``cw_tone: {python: ..., at: ['noise']}`` really does relocate the
        tone -- ``model.cw_tone.at`` is a key a document CAN contain, which an
        earlier draft of ``_tone_placement``'s docstring denied.  A check that
        stood down for the ``cw_tone`` key would lose this document.
        """
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": BANDPASS,
                 "gain": GAIN, "cw_tone": dict(TONE, at=["noise"])}
        found = only(_model_only(model), "A8")
        assert found.where == "model.cw_tone"
        assert "['bandpass', 'gain']" in found.message

    def test_an_at_naming_a_node_the_graph_does_not_have_never_raises(self):
        # §2.3's TRAP: `_t5_downstream` reaches `graph._out['nope']` and the
        # pass dies with "check 'A8' RAISED KeyError", taking the A2 finding
        # about `gian` -- a different line, and a real fault -- down with it.
        # A LIST, because the string spelling stands down at the
        # key-restatement rule before it ever gets here.
        document = _model_only({"global_signal": GLOBAL_SIGNAL, "gian": GAIN,
                                "bandpass": BANDPASS, "gain": GAIN,
                                "cw_tone": dict(TONE, at=["nope"])})
        assert _t5_refused(document, "A8") == []
        assert "A2" in preflight(document).checks()

    def test_the_constraint_is_read_off_the_class(self):
        # Not written here.  A hard-coded ('bandpass', 'gain') passes every
        # test above and goes stale the first time the operator's own
        # declaration moves.
        assert CWCalibrationOperator.must_precede == ("bandpass", "gain")
        assert CWCalibrationOperator.graph_node == "cw_tone"
        assert CWCalibrationOperator.must_precede_because

    def test_the_tone_is_the_only_shipped_class_with_an_ordering_constraint(
            self):
        """Measured at ``48b359d``: 1 of the 58 names in
        ``rheplicant.radio.__all__``.

        ``_tone_placement`` is written generally -- it asks every entry's class
        for a ``must_precede`` -- so this is not a limitation.  It is the
        trigger: a second such class makes the function's name wrong, and its
        message's "Give the tone its own node" wrong with it, on the same
        commit this goes red.
        """
        import inspect

        import rheplicant.radio as radio
        from rheplicant.core.operator import AbstractOperator

        declaring = [name for name in radio.__all__
                     if isinstance(getattr(radio, name), type)
                     and issubclass(getattr(radio, name), AbstractOperator)
                     and not inspect.isabstract(getattr(radio, name))
                     and getattr(getattr(radio, name), "must_precede", ())]
        assert declaring == ["CWCalibrationOperator"]

    def test_the_assemblys_ordering_refusal_is_still_the_backstop(self):
        # `_check_ordering` catches the relocations text cannot resolve -- a
        # `python:` target from outside `rheplicant.radio`.  It stays.
        model = {"global_signal": GLOBAL_SIGNAL, "bandpass": BANDPASS,
                 "gain": GAIN, "noise": dict(TONE, at="noise")}
        with pytest.raises(AssemblyError, match="must_precede"):
            build_model(dict(model), BARE, switch_order=())


class TestDeclaredDataAndSources:
    """A31: an assembly that generates its own data, handed some."""

    def test_declared_data_with_a_source_model_is_refused_naming_the_sources(
            self):
        # The reader has to know WHICH nodes made the twin a generator --
        # "this model lights sources" leaves them reading 33 node ids.
        document = _with_data(_model_only(
            {"global_signal": GLOBAL_SIGNAL, "gain": GAIN}))
        found = only(document, "A31")
        assert found.where == "observation.data"
        assert "['global_signal']" in found.message

    def test_a_transform_chain_may_declare_its_data(self):
        """Both halves are needed, and this is the half that is easy to lose.

        Measured: a transform-only model on declared data runs to completion
        -- it is the documented route for applying a chain to an array.  A
        check on ``observation.data`` alone refuses it.
        """
        assert _t5_refused(_with_data(_model_only({"gain": GAIN})),
                           "A31") == []

    def test_a_source_key_whose_operator_left_it_is_no_source_at_all(self):
        """The refusal that told a transform chain to delete the data it
        cannot run without.

        ``global_signal:`` is a SOURCE node, but this entry's ``python:``
        puts a ``GainOperator`` there, and measured at ``1556ff8`` the
        assembly places it at ``gain``: ``lit == ('gain',)`` and
        ``has_source`` is **False**.  With ``data=None`` that twin raises "it
        needs caller-supplied state.data to act on", so the array A31 said to
        drop is the one thing making the document runnable.  A31 refused it
        because ``_lit`` credited the key rather than the placement.
        """
        model = {"global_signal": PY_GAIN}
        document = _with_data(_model_only(model))
        assert _t5_refused(document, "A31") == []
        twin = build_model(dict(model), GRIDDED, switch_order=())
        assert set(twin.lit) == {"gain"}
        assert twin.has_source is False

    def test_every_source_the_model_lights_is_named_not_just_the_first(self):
        """Cardinality at A31 -- the twin of A8's "both blocked targets
        named".

        A reader told ``['global_signal']`` on a document that also lights
        ``uniform_sky`` deletes one source and comes back to the same
        refusal.  Kills ``sorted(...)[:1]``, which every other cell here
        passes because every other cell lights exactly one source.
        """
        document = _with_data(_model_only(
            {"global_signal": GLOBAL_SIGNAL, "uniform_sky": SKY,
             "gain": GAIN}))
        assert "['global_signal', 'uniform_sky']" in only(document,
                                                          "A31").message

    def test_a_source_model_with_no_data_is_untouched(self):
        # The other half: sources alone are the ordinary simulating document,
        # and refusing them refuses the package's main use.
        document = preflight_document(
            model={"global_signal": GLOBAL_SIGNAL, "gain": GAIN})
        assert _t5_refused(document, "A31") == []

    def test_a_data_key_written_empty_declares_no_data(self):
        """``data:`` with nothing after it is a document that RUNS.

        Measured at ``48b359d``: ``observation._data`` returns ``None`` for a
        ``None`` node (``observation.py:277-278``) and ``Assembly.__call__``
        refuses only ``state.data is not None`` (``core/graph.py:441-447``),
        so ``data:`` written empty in YAML is exactly no data at all.  A check
        asking ``"data" in section`` refuses a document the package runs,
        which is the one direction this pass must never be wrong in.
        """
        from rheplicant.config.sections.observation import _data

        assert _data(None, BARE, n_time=1, n_freq=1) is None
        document = _with_data(
            _model_only({"global_signal": GLOBAL_SIGNAL, "gain": GAIN}),
            data=None)
        assert _t5_refused(document, "A31") == []

    def test_a_recording_beside_declared_data_keeps_the_sections_own_words(
            self):
        """``from_file`` and ``data`` together is ``build_observation``'s, and
        it already precedes the beam.

        Measured: ``observation.py:315-326`` refuses the pair by name ("the
        recording IS the data") from ``build_observation``, which
        ``document.py:72`` calls before ``build_resources`` at ``:75``.  A31
        answering first would tell this reader that "the twin makes it" --
        true of a simulating document and wrong about a recording, one phase
        earlier than the sentence that is right.
        """
        from rheplicant.config.document import load_document

        document = {**preflight_document(
            model={"global_signal": GLOBAL_SIGNAL, "gain": GAIN}),
            "observation": {"meta": {"telescope": "RHINO"},
                            "from_file": {"format": "rhino_hdf5",
                                          "path": "obs.hd5f",
                                          "freq_unit": "MHz"},
                            "data": DATA}}
        assert _t5_refused(document, "A31") == []
        with pytest.raises(ConfigError, match="from_file and data"):
            load_document(document)

    @pytest.mark.parametrize("model", [
        {"global_signal": GLOBAL_SIGNAL, "gain": GAIN},
        {"gain": GAIN},
        {"gain": GAIN, "noise": {"type": "NoiseOperator", "sigma": SIGMA}},
        {"uniform_sky": SKY},
        {"foregrounds": [FOREGROUND], "gain": GAIN},
        {"atmosphere": {"t_atm": {"value": 3.0, "unit": "K"}}, "gain": GAIN},
    ])
    def test_the_static_source_predicate_is_the_assemblys_own(self, model):
        """``kind == "source"`` over the lit ids IS ``Assembly.has_source``.

        Pinned against the package rather than asserted about the string: this
        pins the PACKAGE's predicate, not this check's.  The ``noise`` row is
        the one that catches "any operator that adds signal": a NoiseOperator
        is a transform and adds no data.
        """
        twin = build_model(dict(model), BARE, switch_order=())
        static = any(RADIO_GRAPH.nodes[key].kind == "source" for key in model)
        assert twin.has_source is static

    def test_the_recording_route_is_recorded_and_not_yet_refused(self):
        """``observation.from_file`` is the same defect and is NOT closed here.

        Measured: ``from_file`` + a source model dies with the identical
        ``AssemblyError``, at the run for ``kind: forward`` and inside
        ``load_document`` for a fitting document.  Widening A31 to reach it
        turns ``tests/config/test_config_document.py:167-179``
        (``test_a_source_twin_against_recorded_data_is_the_assemblys_refusal``)
        red -- that test asserts ``load_document`` SUCCEEDS and pins the
        refusal as the assembly's, deliberately.

        **The exact widening, so the next plan inherits a decision rather than
        a discovery.**  In ``_data_with_sources``, replace::

            if section.get("data") is None:
                return

        with::

            declared = [key for key in ("data", "from_file")
                        if section.get(key) is not None]
            if not declared:
                return

        and read ``declared[0]`` into the ``where`` and the message in place
        of the literal ``data``.  Three lines; the cost is
        ``test_config_document.py:167-179``, which has to move with it.
        """
        document = {**preflight_document(
            model={"global_signal": GLOBAL_SIGNAL, "gain": GAIN}),
            "observation": {"meta": {"telescope": "RHINO"},
                            "from_file": {"format": "rhino_hdf5",
                                          "path": "obs.hd5f",
                                          "freq_unit": "MHz"}}}
        assert _t5_refused(document, "A31") == []

    def test_a_junction_is_not_a_source_however_lit_it_is(self):
        """The cell the parametrized test above cannot supply.

        ``astro_sum`` is a JUNCTION: ``_lit`` reports it, and its node kind is
        not ``source``.  Kills ``if _lit(document): refuse`` and any predicate
        that asks "did the model light anything" rather than "did it light a
        source" -- both of which pass every row above, because every row there
        lights a source exactly when it lights anything at all.

        It also cannot go in that parametrize: ``build_model`` refuses a
        junction given an operator (A3), so the row would die in the fixture
        rather than in the assertion.
        """
        document = _with_data(_model_only({"astro_sum": {}, "gain": GAIN}))
        assert _t5_refused(document, "A31") == []

    def test_an_observation_that_is_not_a_mapping_is_left_to_its_builder(self):
        # `_structural` guarantees the SECTION is present, never that it is a
        # mapping (Task 4's carry-forward).  Without the guard `section.get`
        # is an AttributeError, which `preflight` reports as "check 'A31'
        # RAISED" and which costs the document every other finding.  `None` is
        # not among the shapes: it REMOVES the section, and `_structural`
        # raises for a missing required one before any check is reached.
        for section in (["data"], "data", 3):
            document = preflight_document(observation=section)
            assert _t5_refused(document, "A31") == []


class TestTheReVoicedChecksInThePass:
    """Registration, attribution and the phase, for A5, A8 and A31."""

    def test_each_re_voiced_check_owns_its_slot(self):
        # Three functions written and never registered leave every test above
        # green, because they all go through `preflight`.  Subset form, per
        # §3.1 rule 3 -- never `len(CHECKS)` or an exact set.
        assert CHECKS["A5"] is _two_at_one_node
        assert CHECKS["A8"] is _tone_placement
        assert CHECKS["A31"] is _data_with_sources

    def test_the_three_checks_reach_the_report(self):
        # Registration, again: this one asks for the ids by name on ONE
        # document that trips all three, so a module that is imported but
        # whose decorator was dropped fails here.
        document = _with_data(preflight_document(model={
            "global_signal": GLOBAL_SIGNAL, "gain": GAIN,
            "bandpass": dict(PY_GAIN, at=["gain"]),
            "emi": dict(TONE, at="emi")}))
        assert T5_CHECKS <= preflight(document).checks()

    @pytest.mark.parametrize("check, model, observation", [
        ("A5", {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                "bandpass": dict(PY_GAIN, at=["gain"])}, None),
        ("A8", {"global_signal": GLOBAL_SIGNAL, "bandpass": BANDPASS,
                "gain": GAIN, "emi": dict(TONE, at="emi")}, None),
        ("A8", {"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                "bandpass": dict(TONE, at="bandpass")}, None),
        ("A31", {"global_signal": GLOBAL_SIGNAL, "gain": GAIN},
         {"data": DATA}),
    ], ids=["A5", "A8-unreachable", "A8-displaced", "A31"])
    def test_every_message_ends_with_its_own_check_tag_once(
            self, check, model, observation):
        """The convention Task 3 shipped over its own five checks, written for
        these three -- and A8 has TWO messages, which is why it has two rows.

        Both halves: the tail is THERE (a finding citing no check leaves a
        reader with nothing to look up) and it is there ONCE.
        """
        patch = {"model": model}
        if observation is not None:
            patch["observation"] = observation
        message = only(preflight_document(**patch), check).message
        assert re.search(r"\(check A\d+\)\.$", message)
        assert message.count(f"check {check}") == 1

    def test_every_where_is_a_path_into_the_document(self):
        """``Finding.where`` is where the USER types (§3.1, rule 2), and it
        must be spellable by ``config/paths.py``.

        ``preflight._check_where`` calls ``parse_path`` OUTSIDE the per-check
        ``try``, so a ``where`` the path grammar cannot spell kills the whole
        pass instead of reporting the violation, and the user loses every
        other finding (Task 3 paid for this and its carry-forward says so).
        Driven over all three checks at once, and over both of A8's messages.
        """
        document = _with_data(preflight_document(model={
            "global_signal": GLOBAL_SIGNAL, "gain": GAIN,
            "bandpass": dict(PY_GAIN, at=["gain"]),
            "emi": dict(TONE, at="emi")}))
        emitted = [*_two_at_one_node(document), *_tone_placement(document),
                   *_data_with_sources(document)]
        assert len(emitted) == 3, emitted
        assert {parse_path(one.where)[0] for one in emitted} == {
            "model", "observation"}

    def test_every_node_id_is_a_segment_the_path_grammar_can_spell(self):
        """Why A5 and A8 need no ``_task3_where`` cut-back, as an assertion.

        Both build their ``where`` as ``model.<key>`` and both claim only for
        a key that IS a graph node -- so the segment is a RADIO_GRAPH id, not
        a name the user chose.  Task 3's carry-forward records what a
        hyphenated user name does: ``parse_path`` raises outside the per-check
        ``try`` and the whole pass dies.  Measured, ``parse_path('model.my-
        tap')`` raises today.  If a node id ever stops being an identifier,
        this goes red before a user's document does.
        """
        for node in RADIO_GRAPH.nodes:
            assert parse_path(f"model.{node}") == ("model", node)

    def test_a_model_that_is_not_a_mapping_is_left_to_its_own_builder(self):
        """``_structural`` guarantees the SECTION is present, never that it is
        a mapping (Task 4's carry-forward), so every reader needs its own
        guard.

        What this kills is a RAISING A5 or A8: without the guard ``.items()``
        on a list is an ``AttributeError``, which ``preflight`` reports as
        "check 'A5' RAISED" and which costs the document every other finding.

        It does NOT show A31 surviving, and an earlier version of this comment
        said it did.  Measured: with ``model:`` a list, a string or an int,
        ``_nodes`` is ``{}``, ``_lit`` is empty and A31 cannot fire whatever
        happens -- ``preflight(...).checks()`` is empty for all three.  The
        ``observation.data`` on these documents is inert, and it is kept only
        so the shapes match the rest of the class.
        """
        for section in (["gain"], "gain", 3):
            document = _with_data(preflight_document(model=section))
            assert preflight(document).checks() == frozenset()

    def test_a_malformed_node_spec_produces_findings_and_never_raises(self):
        """§2.3's TRAP one level in, where Task 4's rounds found it.

        Five live raises on one document, each from a guard a draft is likely
        to skip: a spec that is ``None`` and one that is an int; a node id
        that is not a string at all (``_t5_claims`` uses the key as a graph
        lookup); an ``at:`` list holding an int; **an ``at:`` naming a node
        the graph does not have**, which reaches ``graph.nodes['nope']`` in A5
        and ``graph._out['nope']`` in A8; and **a ``python:`` naming an
        attribute ``rheplicant.radio`` does not have**, which is a bare
        ``AttributeError`` out of ``getattr`` without the ``__all__`` test.

        The A2 assertion is the point rather than a bonus: a raising check
        aborts the whole pass, so the finding about ``gian`` -- a fault the
        reader really does have to fix -- is what disappears.
        """
        document = _with_data(preflight_document(model={
            "gain": None, "foregrounds": 3, 4: {}, "gian": GAIN,
            "noise": {"python": "rheplicant.radio:Typo", "name": "t"},
            "bandpass": dict(PY_GAIN, at=["nope"]),
            "emi": dict(PY_GAIN, at=["nope"]),
            "cal_loads": {"at": ["gain", 5]}}))
        assert {"A2", "A31"} <= preflight(document).checks()

    def test_a_second_operator_wins_against_a_beam_that_cannot_be_read(self):
        # §5's PHASE PROPERTY, this task's one real assertion of it.  Task 2's
        # phase guard registers four synthetic lambdas: it proves the HOOK's
        # position and says nothing about any shipped check.  The assertion is
        # symmetric -- the violation's own words come back, and `no_such_beam`
        # does NOT.
        #
        # Measured at be2027b, BEFORE the pass: this exact document refused
        # with "No file at 'no_such_beam.npy'", because `build_resources` runs
        # before `build_model`.
        from rheplicant.config.document import load_document

        document = preflight_document(
            model={"global_signal": GLOBAL_SIGNAL, "gain": GAIN,
                   "bandpass": dict(PY_GAIN, at=["gain"])},
            resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "model.gain already fills" in str(caught.value)
        assert "no_such_beam" not in str(caught.value)
