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
    _double_count,
    _graph_shape,
    _lit,
    _nodes,
)
from rheplicant.config.sections.compose import build_model, node_specs
from tests.config.preflight_helpers import UNREADABLE_BEAM, preflight_document

GAIN = {"gain": {"value": 1.1, "unit": "dimensionless"}}
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

    @pytest.mark.parametrize("patch", [
        {},
        {"cw_tone": {"python": "rheplicant:SnapshotOperator", "name": "tap",
                     "at": ["noise_wave", "cw_tone"]}},
        {"foregrounds": [FOREGROUND]},
    ], ids=["the-base-model", "an-at-region", "a-many-node"])
    def test_lit_is_what_the_assembly_itself_reports(self, patch):
        """``_lit``'s docstring claims it reports what ``Assembly.lit`` will,
        so the claim is asserted against a real assembly rather than trusted.

        The region cell is the one that discriminates: measured, an ``at:
        [noise_wave, cw_tone]`` region lights BOTH covered nodes, and a
        ``_lit`` written as ``set(_nodes(document))`` -- the obvious one --
        misses ``noise_wave``, which is precisely the interior node A8's
        reachability question (Task 5) is about.
        """
        document = preflight_document(model=patch)
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

    @pytest.mark.parametrize("at, wanted", [
        ("noise_wave", {"noise_wave"}),
        (["noise_wave", "cw_tone"], {"noise_wave", "cw_tone"}),
        (["noise_wave", "not_a_node"], {"noise_wave"}),
        (["noise_wave", 4], set()),
        ({"node": "cw_tone"}, set()),
        (None, set()),
    ], ids=["a-string", "a-list", "a-list-with-a-stranger", "a-list-with-an-int",
            "a-mapping", "absent"])
    def test_lit_reads_an_at_claim_in_both_spellings_and_no_other(
            self, at, wanted):
        """``at:`` is read by ``_lit`` (this task) and by Task 5's A5, so it is
        one binding; these cells are what say which shapes it reads.

        Kills a reader that handles only the list form, one that iterates a
        bare string character by character (``'noise_wave'`` would light 'n',
        'o', 'i'... none of which is a node, so the cell would come back
        empty), one that keeps a name that is not a node, and one that trusts
        a list of anything -- a malformed ``at:`` is refused at the build with
        the shape it got, not reinterpreted here.

        The string cell names a node the key is NOT, on purpose: a single-node
        ``at:`` that disagrees with its key is check A5's refusal (Task 5) and
        not this reader's business, and a cell whose ``at:`` restated the key
        would light nothing extra and discriminate nothing.
        """
        spec = {"python": "rheplicant:SnapshotOperator", "name": "tap"}
        if at is not None:
            spec = {**spec, "at": at}
        extra = (_lit(preflight_document(model={"cw_tone": spec}))
                 - _lit(preflight_document()))
        assert extra == wanted | {"cw_tone"}


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
