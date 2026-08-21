"""A41, A42, A52 -- the warning channel's first consumers, and one refusal.

Every test names the wrong implementation it kills.

**Two things measured here that the task body had wrong**, both recorded so
the next reader inherits a decision rather than a discovery:

* ``task-11.md`` and ``task-12.md`` both write ``stochastic_operator``.
  **§3.2 (f) does not** -- it pins ``stochastic_nodes`` and nothing else, and
  Task 11 shipped that name verbatim while binding the per-spec form privately
  as ``_a30_stochastic(node_id, spec, table)``.  §3.1/§3.2 win over any task
  body, so the two task bodies are what need correcting.  This module imports
  Task 11's binding and defines no second predicate.
* A52's reference leg reads the ``model:`` section as TEXT rather than through
  ``_nodes``.  Measured: ``_nodes`` is ``{}`` for a ``kind: pipeline`` model,
  and a pipeline stage referencing a projector with no ``observation.pointing``
  BUILDS today -- so the node-shaped walk the task body describes is blind to
  a live case.
"""

import itertools

import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE, WARN, ConfigWarning
from rheplicant.config.preflight import CHECKS, _check_where, preflight
from rheplicant.config.preflight.values import (
    _A41_COUNTED_FORMS,
    _A41_NESTED_SHAPE_FORMS,
    _A41_SHAPE_FORMS,
    _A42_FORM_KEYS,
    _a41_scope,
    _pointing_none,
    _shadowed_literals,
    _simulated_fit_twin,
)
from rheplicant.config.symbols import literal_shadowing_a_symbol
from tests.config.preflight_helpers import (
    BASE_MODEL,
    BASE_OBSERVATION,
    UNREADABLE_BEAM,
    preflight_document,
    repatch,
)

#: A projector with no beam and no file: engine: matrix takes a matrix and a
#: provenance block and nothing else (kinds/projectors.py), which is why every
#: A52 document here loads in milliseconds.  Measured: a document carrying it
#: and a projector reference with no pointing returns a ``ConfiguredRun``.
PROJECTOR = {"engine": "matrix", "matrix": {"zeros": [16, 12]},
             "provenance": {"built_by": "the test suite", "lat_deg": 0.0}}

#: A sky model with no file behind it, so the ``observed_astro_sky`` documents
#: below build rather than dying on an undeclared ``{ref:}``.
SKY_MODEL = {"kind": "uniform", "amplitude": {"value": 200.0, "unit": "K"},
             "n_pix": 12}

#: model.noise, lit.  Two classes register at that node and BOTH declare
#: ``"key"``, which is why a spec with no ``type:`` is still a certain yes.
NOISE = {"type": "NoiseOperator", "sigma": {"value": 0.5, "unit": "K"}}

#: A replacement operator at ``noise`` that does NOT draw.  The ``python:``
#: hatch and not ``type:``: measured, ``type: GainOperator`` at ``noise`` is
#: refused ("not registered at this node"), so the hatch is the only spelling
#: of a non-drawing replacement the package actually builds.
GAIN_REPLACEMENT = {"python": "rheplicant.radio:GainOperator",
                    "gain": {"value": 1.0, "unit": "dimensionless"}}

#: A four-label switch order and the loads it needs.  Written here rather than
#: imported from ``test_preflight_observing``: that module's ``switching()``
#: is a fixture of Task 6's tests, not a published builder, and
#: ``preflight_helpers`` is the only module this plan lets a document come
#: from.  A14's late leg refuses a cycle whose loads are missing, so the two
#: travel together.
ORDER = ["antenna", "ambient", "hot", "noise_source"]


def switching(order=ORDER):
    """An ``observation:`` patch that switches through ``order``."""
    return {**BASE_OBSERVATION,
            "switching": {"mode": "cycle", "order": list(order), "dwell": 4}}


def loads(order=ORDER):
    """``model.cal_loads`` matching ``order[1:]``, which A14 wants present."""
    return {label: {"t_load": {"value": 300.0, "unit": "K"}}
            for label in order[1:]}


def simulated(twin="fit", **extra):
    """One ``inference.observed`` record, in the flat single-record form."""
    return {"from": "simulation", "twin": twin, **extra}


def no_twin_block(**patch):
    """A document whose ``inference:`` carries no ``twin:`` KEY at all.

    ``preflight_document`` merges one level deep by section (§3.2 (b)), and
    the base document carries ``inference.twin: {without: ["noise"]}`` over a
    stochastic ``model.noise``.  An ``inference=`` patch naming only
    ``observed`` therefore leaves that repair standing -- measured, and pinned
    by :meth:`TestA42DataSimulatedThroughTheTwinTheNoiseLeft
    .test_an_inference_patch_naming_only_observed_keeps_the_bases_repair`.
    A merge cannot express a removal, so this rebuilds the section and hands
    it to ``repatch``, which replaces rather than merges.
    """
    document = preflight_document(**patch)
    inference = {key: value for key, value in document["inference"].items()
                 if key != "twin"}
    return repatch(document, inference=inference)


class TestA41ALiteralThatShadowsAGridLength:
    def test_a_literal_freq_length_in_a_shape_is_warned_about(self):
        """The measured hole: ``{ones: [8]}`` on an eight-channel run builds
        with no warning of any kind, though ``_shadowed`` has recorded the
        fact at every resolution since ``7a86c91``.

        Kills: not shipping the check (2C's shape 3).
        """
        doc = preflight_document(resources={"arrays": {"flat": {"ones": [8]}}})
        found = list(_shadowed_literals(doc))
        assert [(f.check, f.severity, f.where) for f in found] == [
            ("A41", WARN, "resources.arrays.flat")]

    def test_the_whole_message_is_this_and_not_something_like_it(self):
        """Whole-text equality, against a literal rather than against the
        function that produced it.

        Eight of Task 6's nine surviving mutants were in refusal TEXT, and a
        ``match=`` on one fragment leaves every other clause free to be
        wrong. For a validation layer the message IS the product.
        """
        doc = preflight_document(resources={"arrays": {"flat": {"ones": [8]}}})
        [found] = list(_shadowed_literals(doc))
        assert found.message == (
            "resources.arrays.flat: the literal 8 at shape position 0 is "
            "this run's n_freq, which observation.freq.grid declares. Write "
            "'n_freq' there instead -- a copied extent stays right until the "
            "grid moves, and then it is a finite, correctly shaped array of "
            "the wrong length, which no shape check and no finite check can "
            "see (check A41)."
        )

    def test_the_message_names_the_symbol_the_position_and_its_source(self):
        """Attribution, not presence -- 2C's shape 1.  On a (16, 8) shape the
        two positions carry DIFFERENT symbols, so a message built from the
        wrong index sends the reader to rename the wrong axis.  An assertion
        that only checked ``"n_freq" in message`` passes with them swapped.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "gain": {"gain": {"ones": [16, 8]}}})
        found = list(_shadowed_literals(doc))
        assert len(found) == 2
        assert ("the literal 16 at shape position 0 is this run's n_time, "
                "which observation.time.grid declares") in found[0].message
        assert ("the literal 8 at shape position 1 is this run's n_freq, "
                "which observation.freq.grid declares") in found[1].message

    def test_a_shape_written_in_symbols_says_nothing(self):
        """Kills a check that warns about every shape it finds."""
        doc = preflight_document(
            resources={"arrays": {"flat": {"ones": ["n_freq"]}}})
        assert list(_shadowed_literals(doc)) == []

    @pytest.mark.parametrize("spec", [
        {"ones": [8]},
        {"zeros": [8]},
        {"full": {"shape": [8], "value": 1.0}},
        {"normal": {"shape": [8], "seed": {"from": "runtime.seeds.p"}}},
        {"uniform": {"shape": [8], "low": 0.0, "high": 1.0,
                     "seed": {"from": "runtime.seeds.p"}}},
    ], ids=["ones", "zeros", "full", "normal", "uniform"])
    def test_every_form_that_carries_a_shape_is_read(self, spec):
        """All five, and not the two a test happened to ask for.

        ``symbols.resolve_shape``'s docstring records that the array forms and
        the draw forms once diverged on exactly this report, and that "a
        report that depends on which constructor the writer reached for is
        worse than no report".  Kills a walk that reads ``zeros``/``ones`` and
        stops, and one that special-cases ``normal`` and forgets ``full`` and
        ``uniform``.  (No seed is ever resolved here -- this pass reads text
        -- so these need no ``runtime.seeds`` entry.)
        """
        doc = preflight_document(resources={"arrays": {"d": spec}})
        assert [f.where for f in _shadowed_literals(doc)] == \
            ["resources.arrays.d"]

    @pytest.mark.parametrize("spec", [
        {"ones": (8,)},
        {"normal": {"shape": (8,), "seed": {"from": "runtime.seeds.p"}}},
    ], ids=["a-flat-form", "a-nested-form"])
    def test_a_shape_written_as_a_tuple_is_read_like_a_list(self, spec):
        """``symbols.resolve_shape`` accepts ``(list, tuple)``, so a document
        built in Python rather than read from YAML can hold a tuple shape and
        the resolver reports it.

        Both walks, because they are two ``isinstance`` calls and narrowing
        either one is silent about a shape the package itself goes on to flag
        -- two validators for one property, disagreeing.
        """
        doc = preflight_document(resources={"arrays": {"t": spec}})
        assert [f.where for f in _shadowed_literals(doc)] == \
            ["resources.arrays.t"]

    def test_a_list_grid_written_as_a_tuple_is_still_a_length(self):
        """The same widening one level up, on the SCOPE rather than the shape:
        ``{list: (60.0, 70.0, 80.0)}`` resolves through ``jnp.asarray`` like
        its list spelling, so declining it would lose the check on a document
        the package builds."""
        doc = preflight_document(
            observation={**BASE_OBSERVATION,
                         "freq": {"grid": {"list": (60.0, 70.0, 80.0),
                                           "unit": "MHz"}}})
        assert _a41_scope(doc).n_freq == 3

    def test_a_malformed_linspace_is_not_read_as_a_list(self):
        """``{linspace: [60.0, 70.0, 80.0]}`` is a linspace the value grammar
        refuses, not a three-entry axis.

        Kills ``_a41_axis_length`` branching on the VALUE's type instead of on
        the FORM -- which reads that list's length as ``n_freq`` and invents
        an axis out of a shape nothing will ever build.
        """
        doc = preflight_document(
            observation={**BASE_OBSERVATION,
                         "freq": {"grid": {"linspace": [60.0, 70.0, 80.0]}}})
        assert _a41_scope(doc) is None

    def test_a_one_is_never_reported(self):
        """``literal_shadowing_a_symbol`` guards on ``value > 1`` so that
        n_source == 1 -- the default for every non-switching run -- does not
        make every ``[1]`` a finding (symbols.py:172-176).  Kills a
        re-implementation of the predicate instead of a call to it, which is
        the §2.5 rule this check exists inside."""
        doc = preflight_document(resources={"arrays": {"one": {"ones": [1]}}})
        assert list(_shadowed_literals(doc)) == []

    def test_the_switch_order_is_what_n_source_comes_from(self):
        """A four-label order makes 4 a shadowing literal.  Kills a scope
        built with ``n_source=1`` hard-coded, which no other test here would
        see."""
        doc = preflight_document(
            observation=switching(),
            model={**BASE_MODEL, "cal_loads": loads()},
            resources={"arrays": {"g": {"zeros": [4, "n_freq"]}}})
        found = list(_shadowed_literals(doc))
        assert [f.where for f in found] == ["resources.arrays.g"]
        assert "is this run's n_source, which observation.switching.order " \
            "declares" in found[0].message

    def test_a_switching_block_this_pass_cannot_read_declines_entirely(self):
        """``mode: []`` is unhashable and is nobody's refusal yet (Task 4
        measured it as a bare ``TypeError`` one phase later).

        The scope declines rather than falling back to ``n_source = 1``,
        because a fallback would name ``n_source`` about a run whose cycle
        this pass never read.  Kills a scope that reads ``switching`` with its
        own ``mode == "cycle"`` test instead of calling Task 4's reader, and
        kills one that treats an unreadable block as no cycle.
        """
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "switching": {"mode": []}},
            resources={"arrays": {"g": {"zeros": [8]}}})
        assert _a41_scope(doc) is None
        assert list(_shadowed_literals(doc)) == []

    @pytest.mark.parametrize("grid", [
        {"file": {"path": "freq.npz", "format": "npz", "key": "f"}},
        {"ref": "resources.arrays.axis"},
        {"linspace": {"start": 60.0, "stop": 85.0, "num": "n_time",
                      "endpoint": True}, "unit": "MHz"},
        {"zeros": [8], "unit": "MHz"},
    ], ids=["file", "ref", "symbolic-num", "an-array-form"])
    def test_a_freq_grid_this_pass_cannot_measure_produces_no_finding(
            self, grid):
        """THE BOUNDARY of Plan 3A, as a test rather than a sentence: the
        shape is always text, the SCOPE is not.

        Kills every guess -- a scope defaulting to zero, a scope built from
        whichever axis happened to be readable, and a check that resolves a
        ref.  Measured for the third row: a freq grid with ``num: "n_time"``
        builds an EMPTY frequency axis today, so a pass that read ``num`` as
        an extent would warn against ``n_freq == 0``.
        """
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "freq": {"grid": grid}},
            resources={"arrays": {"flat": {"ones": [8]}}})
        assert _a41_scope(doc) is None
        assert list(_shadowed_literals(doc)) == []

    @pytest.mark.parametrize("grid", [
        {"file": {"path": "time.npz", "format": "npz", "key": "t"}},
        {"ref": "resources.arrays.axis"},
        {"arange": {"start": 0.0, "step": 2.0, "num": "n_freq"}, "unit": "s"},
        {"zeros": [16], "unit": "s"},
    ], ids=["file", "ref", "symbolic-num", "an-array-form"])
    def test_a_time_grid_this_pass_cannot_measure_produces_no_finding(
            self, grid):
        """The TWIN of the row above, and the reason it is written out.

        ``_a41_scope`` needs BOTH lengths or neither: a check that decided
        decidability from ``freq`` alone passes every case above and reports
        ``n_freq`` where the resolver reports ``n_time`` on every square grid.
        2C's shape 4 -- a hole closed on one route and left open on its twin.
        """
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "time": {"grid": grid}},
            resources={"arrays": {"flat": {"ones": [8]}}})
        assert _a41_scope(doc) is None
        assert list(_shadowed_literals(doc)) == []

    def test_a_boolean_num_is_not_read_as_a_length(self):
        """``num: true`` is ``1`` in Python and the package refuses it in so
        many words -- measured, "A shape position holds True. Booleans are
        integers in Python and would silently give an extent of 0 or 1; write
        the integer."

        So the scope declines rather than building itself out of a ``True``.
        Kills ``isinstance(num, int)`` written without the bool clause, which
        is the same trap ``literal_shadowing_a_symbol`` keeps its own guard
        for.
        """
        doc = preflight_document(observation={
            **BASE_OBSERVATION,
            "freq": {"grid": {"linspace": {"start": 60.0, "stop": 85.0,
                                           "num": True, "endpoint": True},
                              "unit": "MHz"}}})
        assert _a41_scope(doc) is None

    def test_an_ingested_run_reads_its_grids_off_the_recording(self):
        """``observation.from_file`` puts both axes in an HDF5 file.  Kills a
        scope that reads ``observation.freq`` without noticing there is not
        one."""
        doc = repatch(preflight_document(), observation={
            "from_file": {"path": "night1.h5", "format": "rhino_hdf5"}})
        assert _a41_scope(doc) is None

    def test_an_unselected_variant_is_not_walked(self):
        """A variant's shapes are for a run this document is not describing,
        and unselected-variant text is Task 3's check.  Kills a walk over
        ``document.items()`` with no exclusion, which would warn twice about
        one document and once about a document nobody asked for."""
        doc = preflight_document(variants={"hires": {"resources": {"arrays": {
            "flat": {"ones": [8]}}}}})
        assert list(_shadowed_literals(doc)) == []

    def test_a_name_the_path_grammar_cannot_spell_does_not_kill_the_pass(self):
        """Task 3's rule 1, at this module's own call site.

        ``resources: {arrays: {flat-8: ...}}`` loads today -- no reader
        validates a resource name -- and ``preflight._check_where`` raises
        OUTSIDE the per-check ``try``, so an unspellable ``where`` would kill
        the whole pass and cost the document every other finding.  ``where``
        is cut back to the deepest spellable prefix and the FULL path stays in
        the message, which is what the reader is shown.

        Kills a check that hands its raw path straight to ``Finding.where``.
        """
        doc = preflight_document(
            resources={"arrays": {"flat-8": {"ones": [8]}}})
        [found] = list(_shadowed_literals(doc))
        assert found.where == "resources.arrays"
        assert found.message.startswith("resources.arrays.flat-8: ")
        assert {"A41"} <= preflight(doc).checks()

    @pytest.mark.parametrize(("label", "patch"), [
        ("default", {}),
        ("a-list-freq-grid", {"observation": {
            **BASE_OBSERVATION,
            "freq": {"grid": {"list": [60.0, 70.0, 80.0], "unit": "MHz"}}}}),
        ("a-square-grid", {"observation": {
            **BASE_OBSERVATION,
            "time": {"grid": {"arange": {"start": 0.0, "step": 2.0, "num": 8},
                              "unit": "s"}}}}),
        ("a-four-label-cycle", {"observation": switching(),
                                "model": {**BASE_MODEL,
                                          "cal_loads": loads()}}),
        ("a-modulo-freq-grid", {"observation": {
            **BASE_OBSERVATION,
            "freq": {"grid": {"modulo": {"num": 6, "period": 3},
                              "unit": "MHz"}}}}),
    ])
    def test_the_text_scope_is_the_scope_the_resolver_builds(self, label,
                                                             patch):
        """The only tests here that build a document, and the only ones that
        can say the two agree.  A pass whose ``n_freq`` drifted from
        ``ResolutionContext.shape_scope`` would name the wrong symbol while
        every other test above stayed green, because they all measure the pass
        against itself.

        Four documents and not one: the boundary-validation rule is that both
        methods must agree at the dispatch points, and the points here are the
        three spellings of a counted grid plus the switch order.
        """
        doc = preflight_document(**patch)
        run = load_document(doc)
        mine = _a41_scope(doc)
        built = run.context.shape_scope
        assert (mine.n_time, mine.n_freq, mine.n_source) == (
            built.n_time, built.n_freq, built.n_source), label
        for value in (1, 2, 3, 4, 8, 16):
            assert literal_shadowing_a_symbol(value, mine) == \
                literal_shadowing_a_symbol(value, built), value


class TestA42DataSimulatedThroughTheTwinTheNoiseLeft:
    def test_a_fit_twin_that_drops_noise_is_warned_about(self):
        """The measured hole: this document builds, and the observed array it
        produces is the NOISELESS prediction while ``inference.noise`` is
        asked for a sigma.  Kills not shipping the check."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": simulated()})
        found = list(_simulated_fit_twin(doc))
        assert [(f.check, f.severity, f.where) for f in found] == [
            ("A42", WARN, "inference.observed.primary")]

    def test_the_whole_message_is_this_and_not_something_like_it(self):
        """Whole-text equality, for the reason A41's twin gives."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": simulated()})
        [found] = list(_simulated_fit_twin(doc))
        assert found.message == (
            "inference.observed.primary: from: simulation with twin: fit "
            "simulates this observation through the FIT twin, and "
            "inference.twin: takes ['noise'] out of that twin -- so the data "
            "carries no realisation of ['noise'] while the likelihood is "
            "asked to account for it. examples/radio_digital_twin.py puts the "
            "noise in the DATA and keeps it out of the fit twin: write twin: "
            "full (the default) to simulate through the model twin, or "
            "realise: to put the scatter back where the likelihood's own "
            "sigma can see it (check A42)."
        )

    def test_the_message_names_the_stage_whose_realisation_went(self):
        """§4.7.1 asks for the stages by name.  Kills a generic "the fit twin
        differs from the model twin", which tells the reader nothing they can
        act on."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": simulated()})
        [found] = list(_simulated_fit_twin(doc))
        assert "['noise']" in found.message

    def test_twin_full_is_silent(self):
        """The default.  Kills a check that reads ``from:`` and forgets
        ``twin:``."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": simulated(twin="full")})
        assert list(_simulated_fit_twin(doc)) == []

    def test_an_absent_twin_key_is_the_full_default(self):
        """Kills a check that treats a missing ``twin:`` as ``fit`` -- which
        would warn about the commonest document in the repository.

        It does NOT distinguish ``spec.get("twin", "full")`` from
        ``spec.get("twin")``; measured, no test can, because both answer "not
        fit".  The mirror of ``observed.py:132``'s spelling is a readability
        choice and is claimed as one in that function's docstring, not here.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": {"from": "simulation"}})
        assert list(_simulated_fit_twin(doc)) == []

    def test_an_inference_patch_naming_only_observed_keeps_the_bases_repair(
            self):
        """THE FIXTURE HAZARD, measured and pinned rather than left in a
        comment.

        ``preflight_document`` merges one level deep by section, and the base
        carries ``inference.twin: {without: ["noise"]}`` over a stochastic
        ``model.noise``.  So a patch naming only ``observed`` produces a
        document that still repairs the twin -- and a test meaning "no
        ``inference.twin:`` block" that is written that way is not testing
        what its name says.  When it then fails it reads as "the §3.2 (h)2
        correction is wrong" rather than as a fixture misreading.

        This test asserts the merge semantics directly, so the next reader
        does not have to trust a docstring for them.
        """
        doc = preflight_document(inference={"observed": simulated()})
        assert doc["inference"]["twin"] == {"without": ["noise"]}
        assert [f.check for f in _simulated_fit_twin(doc)] == ["A42"]

    def test_a_null_inference_twin_block_is_silent(self):
        """``inference: {twin: null}`` -- the merge's own way of saying the
        key is not wanted.  Kills a reader that calls ``.get`` on it."""
        doc = preflight_document(model={**BASE_MODEL, "noise": NOISE},
                                 inference={"twin": None,
                                            "observed": simulated()})
        assert list(_simulated_fit_twin(doc)) == []

    def test_twin_fit_with_no_inference_twin_key_at_all_is_silent(self):
        """THE correction to §4.7.1's wording, and the false positive a check
        written from the schema alone ships.

        Measured: ``build_fit_twin`` returns the twin unchanged when the
        section is absent (twin.py:37-38), so ``built.twin is
        built.inference.fit_twin`` is True and the data is byte-identical to
        ``twin: full``'s.  Kills the two-condition implementation the
        schema's sentence describes.

        The document is built by :func:`no_twin_block` and not by an
        ``inference=`` patch, for the reason the test above measures.
        """
        doc = no_twin_block(model={**BASE_MODEL, "noise": NOISE},
                            inference={"observed": simulated()})
        assert "twin" not in doc["inference"]
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_non_stochastic_node_in_without_is_silent(self):
        """``without: ["gain"]`` removes nothing that draws.  Kills a check
        that warns on any non-empty ``without:``."""
        doc = preflight_document(model={**BASE_MODEL, "noise": NOISE},
                                 inference={"twin": {"without": ["gain"]},
                                            "observed": simulated()})
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_stochastic_node_the_model_does_not_light_is_silent(self):
        """``without: ["rfi_field"]`` on a model with no ``rfi_field``.  Kills
        a check that reads ``inference.twin.without`` without asking what
        ``model:`` lights."""
        doc = no_twin_block(model=BASE_MODEL)
        doc = repatch(doc, inference={**doc["inference"],
                                      "twin": {"without": ["rfi_field"]},
                                      "observed": simulated()})
        assert list(_simulated_fit_twin(doc)) == []

    def test_rfi_field_is_found_by_the_class_it_names(self):
        """``RFIOperator`` declares ``"key"`` in ``requires`` and is found the
        same way ``NoiseOperator`` is.  Kills a hard-coded ``{"noise"}`` --
        which is what §6's own wording invites and what goes stale on the next
        stochastic operator (§2.5)."""
        doc = preflight_document(
            model={**BASE_MODEL, "rfi_field": {
                "type": "RFIOperator",
                "amplitude": {"value": 5.0, "unit": "K"},
                "occupancy": 0.1}},
            inference={"twin": {"without": ["rfi_field"]},
                       "observed": simulated()})
        [found] = list(_simulated_fit_twin(doc))
        assert "['rfi_field']" in found.message

    def test_a_replacement_that_still_draws_is_silent(self):
        """``replace: {noise: {type: RadiometerNoiseOperator, ...}}`` swaps
        one stochastic operator for another, so nothing left the data.  Kills
        a check that treats every key of ``replace:`` as a removal."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"replace": {"noise": {
                "type": "RadiometerNoiseOperator",
                "channel_width": {"value": 1.0, "unit": "MHz"},
                "integration_time": {"value": 2.0, "unit": "s"}}}},
                "observed": simulated()})
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_replacement_that_does_not_draw_is_warned_about(self):
        """The TWIN of the row above, and the leg ``stochastic_nodes`` alone
        cannot answer: a ``replace:`` spec is not a model node.

        Measured, this document BUILDS.  The ``python:`` hatch is the only
        spelling of a non-drawing replacement at ``noise`` that does (``type:
        GainOperator`` there is refused, "not registered at this node") AND
        that this pass can name a class for.  Kills a check that reads
        ``without:`` and stops, which is §3.2 (h)2's whole correction dropped.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"replace": {"noise": GAIN_REPLACEMENT}},
                       "observed": simulated()})
        found = list(_simulated_fit_twin(doc))
        assert [f.where for f in found] == ["inference.observed.primary"]
        assert "['noise']" in found[0].message

    def test_a_submodule_spelled_replacement_that_still_draws_is_silent(self):
        """THE LIVE DEFECT the previous version of this test recorded, now
        decided rather than declined -- and asserted the same way either way.

        The old form asserted silence and its docstring said *"``_t5_radio_
        class`` resolves ``rheplicant.radio`` and nothing else; the build
        resolves through ``hatch.import_target``, which imports any module"*.
        That divergence is the one the whole-branch review found in six
        checks, and it is closed at the resolver: measured,
        ``import_target('rheplicant.radio.instrument.noise:NoiseOperator') is
        NoiseOperator``, so the pass now NAMES the class and finds that it
        draws.  The document still earns no warning -- nothing left the twin
        -- but for the right reason, and the assertion below is the same one
        under both regimes, which is why it is the WEAK half of this pair.

        Kills ``_a30_stochastic(...) is None`` read as a removal, which is
        what shipped in ``0cc0516``.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"replace": {"noise": {
                "python": "rheplicant.radio.instrument.noise:NoiseOperator",
                "sigma": {"value": 0.5, "unit": "K"}}}},
                "observed": simulated()})
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_replacement_this_pass_cannot_name_a_class_for_is_silent(self):
        """The STRONG half: a class ``rheplicant.radio`` does not export.

        The resolver imports nothing, so it can only name what
        ``rheplicant.radio.__all__`` already holds -- widening the module
        spelling did not widen that.  ``rheplicant.core.operator:
        SnapshotOperator`` is a real class in an imported module and is not an
        exported radio name, so this pass cannot say whether the replacement
        draws.

        ``_a30_stochastic`` answers ``None`` here for "cannot say" and A42
        must not read that as "the draw was taken out": that reading is a
        warning telling a document its data carries no noise when the pass
        never established anything of the kind.  Unlike the row above, this
        cell DISCRIMINATES -- it is red the moment the ``"python" in
        replacement`` clause of ``took_the_draw_out`` is dropped.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"replace": {"noise": {
                "python": "rheplicant.core.operator:SnapshotOperator"}}},
                "observed": simulated()})
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_submodule_spelled_model_node_is_no_longer_a_blind_spot(self):
        """The MIRROR, and it was a LOST CHECK until the resolver widened.

        Measured before: ``model.noise: {python: '<submodule>:NoiseOperator'}``
        with ``without: [noise]`` and ``twin: fit`` **builds**, the fit twin
        genuinely loses the draw (``"noise" in run.inference.fit_twin.lit`` is
        False while it is True in ``run.twin.lit``) -- and A42 said nothing,
        because ``stochastic_nodes`` could not name the class.  The previous
        version of this test asserted that silence and said in its own
        docstring that it *"goes red the day P-1 gains a wider resolver, and
        whoever widens it should widen the sibling above at the same time"*.
        It went red on exactly that commit; this is the widened form.

        Kills a resolver widened only for the ``replace:`` leg: the two legs
        read the same ``python:`` target through the same function, and a
        widening that reached one of them would leave this half silent.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": {
                "python": "rheplicant.radio.instrument.noise:NoiseOperator",
                "sigma": {"value": 0.5, "unit": "K"}}},
            inference={"twin": {"without": ["noise"]},
                       "observed": simulated()})
        found = list(_simulated_fit_twin(doc))
        assert [one.where for one in found] == ["inference.observed.primary"]
        assert "['noise']" in found[0].message

    def test_the_two_spellings_of_one_class_reach_A42_alike(self):
        """The property the six defects were instances of, on A42's own leg.

        ``rheplicant.radio:NoiseOperator`` and
        ``rheplicant.radio.instrument.noise:NoiseOperator`` are ONE class
        object -- measured, ``import_target`` returns the identical object for
        both -- and the build resolves either.  A pass that answers
        differently about them is answering about the spelling.

        Kills any resolver that special-cases one module name: the two
        documents differ in nothing else, so the comparison cannot be
        satisfied by a rule about ``rheplicant.radio`` in particular.
        """
        def report(target):
            doc = preflight_document(
                model={**BASE_MODEL, "noise": {
                    "python": target, "sigma": {"value": 0.5, "unit": "K"}}},
                inference={"twin": {"without": ["noise"]},
                           "observed": simulated()})
            return [(one.check, one.where, one.message)
                    for one in _simulated_fit_twin(doc)]

        assert report("rheplicant.radio:NoiseOperator") == report(
            "rheplicant.radio.instrument.noise:NoiseOperator")
        assert report("rheplicant.radio:NoiseOperator") != []

    def test_a_replacement_at_a_node_that_never_drew_is_silent(self):
        """``replace: {gain: <a GainOperator>}`` beside a stochastic
        ``model.noise``.  Nothing left the data: ``gain`` never put anything
        in it.

        The TWIN of ``test_a_non_stochastic_node_in_without_is_silent`` -- the
        ``without:`` leg had this test and the ``replace:`` leg did not, which
        is 2C's shape 4 inside one function.  Kills a replace leg written
        without ``node_id in drawing``.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"replace": {"gain": GAIN_REPLACEMENT}},
                       "observed": simulated()})
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_without_written_as_a_tuple_is_read_like_a_list(self):
        """``build_fit_twin`` accepts a tuple there -- measured, this document
        builds and its fit twin has lost ``noise``.

        The same widening this module pins for ``_a41_shapes`` and for the
        ``list:`` grid form, on the third place it matters.  Kills
        ``isinstance(without, list)``, which loses the check on a document the
        package runs.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ("noise",)},
                       "observed": simulated()})
        [found] = list(_simulated_fit_twin(doc))
        assert "['noise']" in found.message

    def test_the_named_records_form_is_read(self):
        """``observed: {primary: ..., night: ...}`` -- 2C's shape 4, a hole
        closed on one route and left open on its twin.  The ``where``s are
        what pin it: a check that reported both against ``primary`` passes a
        count assertion."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": {"primary": simulated(),
                                    "night": simulated(),
                                    "day": simulated(twin="full")}})
        assert [f.where for f in _simulated_fit_twin(doc)] == [
            "inference.observed.primary", "inference.observed.night"]

    def test_a_record_name_the_path_grammar_cannot_spell_is_cut_back(self):
        """Task 3's rule 1 at this module's second call site.  A record named
        ``night-1`` is a document that loads, and its raw path would make
        ``_check_where`` kill the whole pass."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": {"night-1": simulated()}})
        [found] = list(_simulated_fit_twin(doc))
        assert found.where == "inference.observed"
        assert found.message.startswith("inference.observed.night-1: ")

    def test_the_same_node_named_twice_is_said_once(self):
        """``without: [noise, noise]`` is a document, and a message reading
        "takes ['noise', 'noise'] out of that twin" is one the reader has to
        decode.  Kills a walk with no de-duplication."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise", "noise"]},
                       "observed": simulated()})
        [found] = list(_simulated_fit_twin(doc))
        assert "['noise']" in found.message
        assert "'noise', 'noise'" not in found.message

    @pytest.mark.parametrize("record", [
        {"file": {"path": "night1.npz", "format": "npz", "key": "w"}},
        {"file": {"path": "night1.npz", "format": "npz", "key": "w"},
         "twin": "fit"},
    ], ids=["plain", "carrying-a-twin-key"])
    def test_the_file_form_is_silent(self, record):
        """Data read off disk was not simulated through anything.  Kills a
        record reader that treats every mapping as a simulation.

        The second row is what makes the ``from:`` half of the condition
        load-bearing: without it, a check that read only ``twin:`` passes the
        first row (a file record declares no twin) and warns about data no
        twin ever touched.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": {"night": record}})
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_form_key_is_not_read_as_an_observation_name(self):
        """``_A42_FORM_KEYS`` pinned by USE and not only by value.

        The mirror test below compares the constant to the section's own; it
        cannot see a walk that stopped consulting it.  ``observed: {at: {from:
        simulation, twin: fit}}`` is refused by ``build_observed`` in its own
        words ("'at' is not usable as an observation name"), so A42 warning
        about a record called ``at`` would be Task 5's pre-emption defect as
        well as a wrong ``where``.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": {"at": simulated()}})
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_file_headed_section_is_one_record_not_a_mapping_of_names(self):
        """``build_observed`` splits on ``"from" in section or "file" in
        section`` -- BOTH keys -- and this pins the second half of that ``or``.

        The discriminating document has to be one the section itself refuses,
        because the two readings agree on every legal one: a ``file:``-headed
        section that also carries a key named ``primary`` is ONE record under
        the real split (so no top-level ``from:``, so silent) and a MAPPING OF
        NAMES under a split written ``"from" in section`` (so ``primary`` is a
        simulated record, so a warning at a ``where`` that does not exist).
        The test's content is the SPLIT, not the document.
        """
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": {"file": {"path": "night1.npz",
                                             "format": "npz", "key": "w"},
                                    "primary": simulated()}})
        assert list(_simulated_fit_twin(doc)) == []

    def test_a_new_value_form_makes_someone_look_at_the_a41_walk(self):
        """The three A41 form tuples are HAND-WRITTEN, and nothing pinned them.

        ``_A42_FORM_KEYS`` gets a mirror test below and ``_FORM_KEYS`` gets
        one, but the A41 tuples are three separate hand-written subsets of
        ``config.values.VALUE_FORMS`` with no relationship to it asserted
        anywhere.  A nineteenth form landing there -- one that declares a
        shape or a length -- would leave A41 silently blind to it, and every
        test in this module green.

        Two halves.  Each tuple must be a SUBSET of the package's own
        vocabulary (so a typo here is a form that can never match), and the
        set of forms the walk does NOT read is pinned by name -- so adding a
        form to ``VALUE_FORMS`` fails this test and makes someone decide
        whether A41 should read it, rather than deciding by omission.
        """
        from rheplicant.config.values import VALUE_FORMS

        walked = set(_A41_COUNTED_FORMS) | set(_A41_SHAPE_FORMS) | set(
            _A41_NESTED_SHAPE_FORMS)
        assert walked <= set(VALUE_FORMS)
        assert set(VALUE_FORMS) - walked == {
            # Nine forms, and none of them states a length or a shape in the
            # text: a reference, a file, a scalar, a callable, a derived grid,
            # a restacking, a basis fit, and the switch-order sugar.  There is
            # no DELIBERATE omission left -- `modulo` was one until the
            # measurement went the other way (`arrays.py:121` is
            # `jnp.arange(num) % period`, so `num` IS the axis length, and a
            # `{modulo: {num: 6, period: 3}}` freq grid builds `n_freq == 6`),
            # and it is now walked.
            "basis_fit", "file", "from", "from_grid", "from_switch_order",
            "python", "ref", "stack", "value",
        }

    def test_the_record_split_still_mirrors_the_section_that_owns_it(self):
        """This module reads ``inference.observed``'s shape a second time, and
        a second reader is a divergence waiting to happen (§2.2, the
        ``_number``-vs-``_whole`` item).  Pinned against the section's own
        frozenset, so a key added there goes red HERE rather than silently
        turning a form key into an observation name."""
        from rheplicant.config.sections.observed import _FORM_KEYS

        assert _A42_FORM_KEYS == _FORM_KEYS

    def test_the_type_route_of_the_replace_leg_is_unreachable_today(self):
        """The premise that makes one clause of ``took_the_draw_out`` an
        EQUIVALENT MUTANT, asserted rather than assumed.

        A ``replace:`` is only considered at a node ``stochastic_nodes``
        reports, and measured, every class registered at such a node draws --
        so a ``type:`` naming one of that node's own classes always answers
        "still draws" and never reaches the removal branch.  Deleting that
        branch is therefore green today.

        It is kept anyway, because the day a non-drawing class registers at
        ``noise`` or ``rfi_field`` its absence is a LOST check and nothing
        would say so.  This test is what dates that day: it goes red, and
        whoever makes it red can then write the document that kills the
        mutant.
        """
        from rheplicant.config.sections.model import operator_table
        from rheplicant.core.contract import RANDOMNESS

        table = operator_table()
        draws = {node for node, classes in table.items()
                 if any(RANDOMNESS in cls.requires for cls in classes)}
        assert draws == {"noise", "rfi_field"}
        assert all(RANDOMNESS in cls.requires
                   for node in draws for cls in table[node])

    def test_no_shipped_node_has_classes_that_disagree_about_randomness(self):
        """The fact Task 11's ``all(...)`` rests on, defended rather than
        asserted in a docstring -- 2C's shape 2, eleven instances.  Two
        classes register at ``noise`` and both draw; if one ever did not, a
        ``type:``-less spec there would need a tie-break this code does not
        have."""
        from rheplicant.config.sections.model import operator_table
        from rheplicant.core.contract import RANDOMNESS

        mixed = {node: [cls.__name__ for cls in classes]
                 for node, classes in operator_table().items()
                 if len({RANDOMNESS in cls.requires for cls in classes}) > 1}
        assert mixed == {}


class TestA52APointingOfNoneAndAProjectorAnyway:
    def test_a_projector_reference_with_no_pointing_section_is_refused(self):
        """Measured: this document builds today, returning a
        ``ConfiguredRun``.  ``pointing`` absent IS ``{mode: none}``
        (pointing.py:134), so a check that only reads a written ``mode:``
        misses the commonest spelling of the error."""
        doc = preflight_document(
            resources={"projectors": {"p": PROJECTOR}},
            model={**BASE_MODEL, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]})
        found = list(_pointing_none(doc))
        assert [(f.check, f.severity, f.where) for f in found] == [
            ("A52", REFUSE, "model.filters[0].projector")]

    def test_the_whole_reference_message_is_this(self):
        """Whole-text equality on the refusal a user actually reads."""
        doc = preflight_document(
            resources={"projectors": {"p": PROJECTOR}},
            model={**BASE_MODEL, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]})
        [found] = list(_pointing_none(doc))
        assert found.message == (
            "model.filters[0].projector references a projector while "
            "observation.pointing is mode: none -- which is the default when "
            "the section is absent, and the statement that this run has no "
            "pointing at all. A projector turns a sky into what THIS "
            "observation saw, and mode: none says there is no pointing for it "
            "to turn it into. Declare observation.pointing: {mode: "
            "drift|tracked|baked}, or remove the reference (check A52)."
        )

    def test_an_explicitly_written_mode_none_is_refused_the_same_way(self):
        """Kills a check that reads ``observation.pointing`` only when present
        and one that reads it only when absent."""
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": {"mode": "none"}},
            resources={"projectors": {"p": PROJECTOR}},
            model={**BASE_MODEL, "beam_spill": {
                "from": "projector",
                "projector": {"ref": "resources.projectors.p"},
                "t_ground": {"value": 300.0, "unit": "K"}}})
        assert [f.where for f in _pointing_none(doc)] == \
            ["model.beam_spill.projector"]

    def test_observed_astro_sky_is_its_own_trigger(self):
        """§6 names the node AND the reference, and they are two conditions.
        This document lights the node with no projector reference at all, so a
        check that implemented only the reference leg reports nothing.

        The document is refused later for a different reason
        (``SkySourceOperator requires ['projector']``) -- but that refusal is
        ``build_model``'s, which runs AFTER ``build_resources``, so it is not
        a better sentence already said before the beam and Task 5's
        stand-down rule does not reach it.
        """
        doc = preflight_document(model={**BASE_MODEL, "observed_astro_sky": {
            "sky_model": {"ref": "resources.sky_models.s"}}})
        found = list(_pointing_none(doc))
        assert [(f.check, f.severity, f.where) for f in found] == [
            ("A52", REFUSE, "observation.pointing")]

    def test_the_whole_node_message_is_this(self):
        """Whole-text equality on the OTHER of A52's two messages.

        The severity is in the triple above rather than only in the reference
        leg's: a ``warn()`` written where ``refuse()`` belongs turns A52 from
        a refusal into a sentence a user can ignore, and every ``where``-only
        assertion in this class passes with it.
        """
        doc = preflight_document(model={**BASE_MODEL, "observed_astro_sky": {
            "sky_model": {"ref": "resources.sky_models.s"}}})
        [found] = list(_pointing_none(doc))
        assert found.message == (
            "model lights observed_astro_sky, which sees the sky through a "
            "projector, and observation.pointing is mode: none -- which is "
            "the default when the section is absent, and the statement that "
            "this run has no pointing at all. Declare observation.pointing: "
            "{mode: drift|tracked|baked}, or drop the node (check A52)."
        )

    def test_the_document_that_carries_both_triggers_says_both(self):
        """A valid ``observed_astro_sky`` run ALWAYS carries a projector
        reference -- measured, an inline projector is refused ("an object
        field takes a declared resource"), so the two legs necessarily
        coincide on the document that matters.

        Both are emitted, and that is a decision rather than an accident: the
        node leg names the fix (declare a pointing) and the reference leg
        names the LINE, and ``raise_if_refused`` quotes the first and counts
        the rest.  Pinned so that collapsing them later is a deliberate
        change.  Measured: this document builds today.
        """
        doc = preflight_document(
            resources={"projectors": {"p": PROJECTOR},
                       "sky_models": {"s": SKY_MODEL}},
            model={**BASE_MODEL, "observed_astro_sky": {
                "sky_model": {"ref": "resources.sky_models.s"},
                "projector": {"ref": "resources.projectors.p"}}})
        assert [(f.severity, f.where) for f in _pointing_none(doc)] == [
            (REFUSE, "observation.pointing"),
            (REFUSE, "model.observed_astro_sky.projector")]

    def test_every_projector_reference_is_named_not_only_the_first(self):
        """Task 3's rule 3: attribution needs a per-index test on every loop.

        Two filters, two references, two findings with DIFFERENT ``where``s.
        Kills a walk that reports the first hit and returns, and one that
        hard-codes ``[0]`` in the path it builds.
        """
        doc = preflight_document(
            resources={"projectors": {"p": PROJECTOR, "q": PROJECTOR}},
            model={**BASE_MODEL, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}},
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.q"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]})
        assert [f.where for f in _pointing_none(doc)] == [
            "model.filters[0].projector", "model.filters[1].projector"]

    def test_a_pipeline_model_that_references_a_projector_is_refused_too(self):
        """The measured hole in the task body's own design: ``_nodes`` is
        ``{}`` for a ``kind: pipeline`` model, and a pipeline stage that
        references a projector with no ``observation.pointing`` **builds
        today** -- measured, ``load_document`` returns a ``ConfiguredRun``.

        So A52's reference leg reads ``model:`` as text rather than as nodes.
        Kills the node-shaped walk, which is silent here.
        """
        doc = repatch(
            preflight_document(
                resources={"projectors": {"p": PROJECTOR},
                           "sky_models": {"s": SKY_MODEL}},
                inference=None),
            model={"kind": "pipeline", "stages": [
                {"name": "sky", "type": "SkySourceOperator",
                 "sky_model": {"ref": "resources.sky_models.s"},
                 "projector": {"ref": "resources.projectors.p"}}]})
        assert [f.where for f in _pointing_none(doc)] == [
            "model.stages[0].projector"]

    def test_a_fan_label_the_path_grammar_cannot_spell_is_cut_back(self):
        """Task 3's rule 1 at this module's third and last call site.

        A ``many`` node may be a mapping keyed by FAN LABEL, and a label is a
        name the user chose: ``filters: {lo-band: ...}`` is a document that
        reaches this walk.  Its raw path would make ``_check_where`` raise
        outside the per-check ``try`` and cost the document every other
        finding.
        """
        doc = preflight_document(
            resources={"projectors": {"p": PROJECTOR}},
            model={**BASE_MODEL, "filters": {"lo-band": {
                "type": "SkySpaceFilter",
                "projector": {"ref": "resources.projectors.p"},
                "regularization": {"value": 1e-3,
                                    "unit": "dimensionless"}}}})
        [found] = list(_pointing_none(doc))
        assert found.where == "model.filters"
        assert found.message.startswith(
            "model.filters.lo-band.projector references a projector ")

    def test_a_pointing_that_is_not_a_mapping_is_left_to_its_own_refusal(self):
        """``pointing: drift`` (the string, not the block) is a shape
        ``compile_pointing`` refuses in its own words, and reading ``.get`` on
        it here is an ``AttributeError`` -- which the pass turns into "check
        RAISED" and which DISCARDS every other finding (§2.3's TRAP).

        Kills ``(spec or {}).get(...)`` written without the type guard in
        front of it.
        """
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": "drift"},
            resources={"projectors": {"p": PROJECTOR}},
            model={**BASE_MODEL, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]})
        assert list(_pointing_none(doc)) == []

    def test_a_drift_pointing_is_not_refused(self):
        """Kills a check that fires on the projector alone.  Every projector
        document in the repository has a pointing; this is the "did not
        overreach" leg."""
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": {
                "mode": "drift", "az_deg": {"value": 0.0, "unit": "deg"},
                "el_deg": {"value": 90.0, "unit": "deg"},
                "materialise": ["pointing"],
                "lst": {"mode": "uniform_turn", "n_time": "n_time"}}},
            resources={"projectors": {"p": PROJECTOR}},
            model={**BASE_MODEL, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]})
        assert list(_pointing_none(doc)) == []

    def test_a_declared_but_unreferenced_projector_is_not_refused(self):
        """The schema says REFERENCED.  A document may declare a projector for
        a variant that uses it.

        This kills a check that fires on a DECLARATION.  It does not kill one
        that walks ``resources:`` for references, and an earlier version of
        this docstring claimed it did: measured, a ``resources:`` walk finds
        nothing here either, because :data:`PROJECTOR` holds no ``{ref:}`` of
        its own.  The test below is the one that pins the scope.
        """
        doc = preflight_document(resources={"projectors": {"p": PROJECTOR}})
        assert list(_pointing_none(doc)) == []

    def test_a_projector_reference_outside_the_model_is_not_this_check(self):
        """The scope is ``model:``, pinned with a document where the two walks
        DISAGREE -- which the test above cannot do.

        ``resources.arrays.a: {ref: resources.projectors.p}`` puts a projector
        reference in the document and none of it in ``model:``.  A52's subject
        is a model that turns a sky into what THIS observation saw; a resource
        naming another resource is not that, and widening the walk would refuse
        documents on the strength of a reference no operator reads.

        Kills a walk over the whole document, and one over ``resources:``.
        """
        doc = preflight_document(resources={
            "projectors": {"p": PROJECTOR},
            "arrays": {"a": {"ref": "resources.projectors.p"}}})
        assert list(_pointing_none(doc)) == []

    def test_a_mode_this_layer_does_not_own_is_left_to_its_own_refusal(self):
        """``pointing: {mode: null}`` is not ``mode: none``.

        ``compile_pointing`` refuses it accurately -- "observation.pointing:
        mode is one of ['none', 'drift', 'tracked', 'baked']; got None." -- and
        that runs inside ``build_observation``, which is BEFORE
        ``build_resources``, so it already precedes the beam.  A52 answering
        first would tell the reader their document "says mode: none" when it
        says no such thing: Task 5's pre-emption defect, with a false premise
        attached.

        Kills ``((spec or {}).get("mode") or "none")``, which reads a null mode
        as the default.
        """
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": {"mode": None}},
            resources={"projectors": {"p": PROJECTOR}},
            model={**BASE_MODEL, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]})
        assert list(_pointing_none(doc)) == []

    def test_a_reference_to_something_else_is_not_a_projector(self):
        """``resources.projectors_backup.p`` shares every character of the
        prefix except its trailing dot.  Kills a ``startswith`` written
        without the separator, which would refuse a document naming any
        resource whose kind merely begins with "projectors"."""
        doc = preflight_document(model={**BASE_MODEL, "t_sys_extra": [
            {"from": "basis",
             "basis": {"ref": "resources.projectors_backup.p"},
             "coeff": {"zeros": ["n_freq"]}}]})
        assert list(_pointing_none(doc)) == []

    def test_the_site_half_of_a52_is_not_attempted_here(self):
        """§2.6 item 7: the ``{from: site}`` route does not exist -- the
        derivation registry is closed and measured as
        ``['basis_matrix', 'channel_spacing', 'horizon_fraction',
        'interpolate_onto', 'sample_cadence', 'unit_mean_free']`` -- so the
        ``site.lat_deg`` half is §6's, not this task's.

        The content of this test is the ABSENCE: a projector is referenced, it
        carries its own ``lat_deg``, ``observation.site`` declares none, a
        pointing IS declared, and that is not a finding.  Pinned because the
        next reader of the A52 row will otherwise implement the second half
        from the schema and require a key nothing in v1 consumes.
        """
        doc = preflight_document(
            observation={**BASE_OBSERVATION,
                         "site": {"alt_m": {"value": 100.0, "unit": "m"}},
                         "pointing": {"mode": "baked",
                                      "provenance": {"built_by": "a test"}}},
            resources={"projectors": {"p": {
                **PROJECTOR, "lat_deg": {"value": 51.0, "unit": "deg"}}}},
            model={**BASE_MODEL, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]})
        assert list(_pointing_none(doc)) == []

    def test_the_derivation_registry_still_has_no_site_route(self):
        """§2.6 item 7's premise, defended rather than quoted.  The day a
        ``site`` derivation ships is the day someone should look at A52's
        second half; a sentence in a docstring cannot say that and this can.
        """
        from rheplicant.config.derive import _DERIVATIONS

        assert "site" not in _DERIVATIONS


class TestAllThreeChecksReachThePass:
    def test_the_three_ids_are_bound_to_these_functions(self):
        """Kills a missing decorator and a decorator carrying the wrong id --
        either of which leaves every test above green while the pass runs none
        of them."""
        assert CHECKS["A41"] is _shadowed_literals
        assert CHECKS["A42"] is _simulated_fit_twin
        assert CHECKS["A52"] is _pointing_none

    def test_every_finding_this_module_emits_carries_its_check_tag(self):
        """The convention from Task 3 on, over this module's own three checks
        rather than over the registry.  Kills a message that forgot the tail
        and one that doubled it."""
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": {"mode": "none"}},
            resources={"projectors": {"p": PROJECTOR},
                       "arrays": {"flat": {"ones": [8]}}},
            model={**BASE_MODEL, "noise": NOISE, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]},
            inference={"twin": {"without": ["noise"]},
                       "observed": simulated()})
        found = [f for check in (_shadowed_literals, _simulated_fit_twin,
                                 _pointing_none)
                 for f in check(doc)]
        assert {f.check for f in found} == {"A41", "A42", "A52"}
        for finding in found:
            assert finding.message.endswith(f"(check {finding.check}).")
            assert finding.message.count(f"(check {finding.check}).") == 1

    def test_a_warning_does_not_stop_the_pass_or_the_document(self):
        """A41 and A42 are the first ``warn``s this layer emits, and the whole
        point of the severity is that the document still runs.  Kills a check
        that raises, and a ``warn()`` call that was written as ``refuse()``."""
        doc = preflight_document(
            model={**BASE_MODEL, "noise": NOISE},
            inference={"twin": {"without": ["noise"]},
                       "observed": simulated()},
            resources={"arrays": {"flat": {"ones": [8]}}})
        report = preflight(doc)
        assert {"A41", "A42"} <= report.checks()
        assert {f.check for f in report.warnings()} >= {"A41", "A42"}
        assert [f for f in report.refusals() if f.check in {"A41", "A42"}] == []

    def test_the_base_document_earns_none_of_these_three(self):
        """§3.2 (b)'s ``ids`` accessor is "what 'and nothing else' reads", and
        a base that is itself a finding makes it never return empty.  Kills a
        check whose positive branch is the default one."""
        doc = preflight_document()
        assert {"A41", "A42", "A52"} & preflight(doc).checks() == set()

    def test_a_pointing_of_none_wins_against_a_beam_that_cannot_be_read(self):
        """§5's PHASE PROPERTY, this task's one real assertion of it for a
        REFUSAL.

        Task 2's phase guard registers four synthetic lambdas: it proves the
        HOOK's position and says nothing about any shipped check.  Nine tasks
        each own one document that carries a real violation AND an unreadable
        beam, and the assertion is symmetric -- the violation's own words come
        back, and ``no_such_beam`` does NOT.

        A52 is the REFUSAL of this task's three, so it is the one that can
        carry the assertion in this form: A41 and A42 warn, and a warning does
        not out-rank a beam because nothing raises.  The warning form of the
        same property is the test below.
        """
        document = preflight_document(
            resources={**UNREADABLE_BEAM, "projectors": {"p": PROJECTOR}},
            model={**BASE_MODEL, "filters": [
                {"type": "SkySpaceFilter",
                 "projector": {"ref": "resources.projectors.p"},
                 "regularization": {"value": 1e-3,
                                     "unit": "dimensionless"}}]})
        with pytest.raises(ConfigError) as caught:
            load_document(document)
        assert "check A52" in str(caught.value)
        assert "no_such_beam" not in str(caught.value)

    def test_a_warning_reaches_the_user_before_the_beam_is_read(self):
        """The phase property for a WARNING, which cannot be stated the way
        the refusal's is.

        A warning does not out-rank anything: this document still dies on the
        beam.  What the phase buys is that the user is TOLD anyway, and told
        before the spherical harmonic transform runs -- ``load_document``
        calls ``emit_warnings()`` at line 70 and ``build_resources`` at 76.
        Move the hook below ``build_resources``, or delete it, and the beam
        raises first, no ``ConfigWarning`` is ever emitted, and
        ``pytest.warns`` goes red.
        """
        document = preflight_document(
            resources={**UNREADABLE_BEAM,
                       "arrays": {"flat": {"ones": [8]}}})
        with pytest.warns(ConfigWarning, match=r"check A41"), \
                pytest.raises(ConfigError, match="no_such_beam"):
            load_document(document)

    def test_no_check_here_raises_or_emits_an_unspellable_where(self):
        """§2.3's TRAP and Task 3's rule 1, over a product of shapes.

        A ``TypeError`` or ``KeyError`` out of a check becomes "check RAISED"
        and DISCARDS every other finding, so a hostile section costs the user
        the whole report.  ``_structural`` guarantees a section is PRESENT,
        never that it is a mapping -- Task 4's carry-forward measured that
        exact shape as a live crash -- and the same applies one level in, to a
        VALUE used as a dict key.

        ``_check_where`` is run over every finding for the same reason: it
        raises OUTSIDE the per-check ``try``, so a ``where`` built from a name
        the path grammar cannot spell kills the pass just as thoroughly as a
        ``TypeError``, and no single-document test can find the branch that
        builds one.  Measured: ``parse_path('resources.arrays.7')`` raises, so
        the ``isinstance(key, str)`` guards in both walks are load-bearing and
        this is what holds them.

        **The anti-vacuity assertion is PER CHECK, and that is not
        decoration.**  The first form of this battery counted findings in
        total and passed on A41 and A52 alone: no ``model`` cell made ``noise``
        stochastic, so ``_a42_removed`` was empty in every one of its cells and
        A42's record walk was never executed at all.  A total tells you the
        battery found something; it does not tell you which walk was running.
        """
        hostile = [
            None, 3, "text", [], {}, [1, 2], {"a": None}, {3: "int-key"},
            {"ref": 7}, {"ones": 8}, {"ones": [[]]}, {"mode": []},
            "pointing-as-a-string",
            # Shapes that get PAST the first guard of each check, so the
            # battery exercises the walks rather than their early returns.
            {"time": {"grid": {"list": [1, 2, 3]}},
             "freq": {"grid": {"list": [4, 5, 6]}},
             "switching": {"mode": "cycle", "order": [3, None]},
             "pointing": "drift"},
            {"time": {"grid": {"arange": {"num": 4}}},
             "freq": {"grid": {"linspace": {"num": 4}}},
             "pointing": {"mode": None}},
            {"noise": {"type": []}, "gain": 3, "filters": [{"ref": None}],
             "sky-1": {"ref": "resources.projectors.p"},
             "d-1": {"ones": [4, "n_freq"]}, "observed_astro_sky": {},
             "n": {7: {"ones": [4]}}},
            # A model whose `noise` really is stochastic, so `stochastic_nodes`
            # is non-empty and A42's `_a42_records` walk RUNS.
            {"noise": NOISE, "rfi_field": {"type": "RFIOperator"},
             "p": {"ref": "resources.projectors.p"}},
            {"twin": {"without": [["noise"], "noise", 3],
                      "replace": {"noise": [], 3: {}, "rfi_field": {"type": 7}}},
             "observed": {"night-1": {"from": "simulation", "twin": "fit"},
                          "bad": 3, 7: {}}},
        ]
        checks = {"A41": _shadowed_literals, "A42": _simulated_fit_twin,
                  "A52": _pointing_none}
        calls = 0
        emitted = dict.fromkeys(checks, 0)
        for observation, model, inference in itertools.product(hostile,
                                                               repeat=3):
            document = {"schema_version": 1, "runtime": {"seed": 1},
                        "observation": observation, "model": model,
                        "inference": inference, "runs": [{"kind": "forward"}],
                        "resources": model, 7: model}
            for check_id, check in checks.items():
                for finding in check(document):
                    _check_where(check_id, finding)
                    emitted[check_id] += 1
                calls += 1
        assert calls == len(hostile) ** 3 * len(checks)
        # ANTI-VACUITY: a battery whose cells all return at the first
        # `isinstance` proves only that the first `isinstance` is there.
        assert all(emitted.values()), emitted
