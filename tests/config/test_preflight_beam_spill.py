"""Pre-flight A50: the horizon split asked of a projector that cannot answer.

The one INVENTED row of Task 2, and the only row in it whose subject is
currently unguarded rather than merely late.

Leg A has no message anywhere in this package because the failure it is about
produces no message anywhere in this package: a beam cut with ``horizon.mode:
truncate_map`` and then split again by ``beam_spill: {from: projector}`` comes
back finite, correctly shaped, in the right unit -- with the ground term gone.
Leg B does raise, but from ``DriftScanProjector`` and as a
``StateValidationError``, which is a SIBLING of ``ConfigError`` rather than a
subclass (§0.2 C-12), arriving after the beam has been read, analysed and
rotated.

So both messages are written in this layer's voice, both end with ``(check
A50).``, and both are pinned here by equality on their whole text.  There is
no one-binding row for A50 in ``message_binding.py``: a hoist has an original
to be bound once against and an invention does not.

**Three things this module spends as much space on as the two positives**, and
each is a measured trap:

* ``horizon.mode`` is not refused on its own.  ``projector_mask`` beside
  ``beam_spill.from: projector`` is the CORRECT combination and only
  ``truncate_map`` double-counts.
* ``beam_frame`` is never followed.  It is in ``_NOT_WRITABLE`` and already
  refused as a document key by a better sentence; leg B follows
  ``optimizations``.
* ``{from: horizon_fraction, projector: {ref: ...}}`` in a value node has its
  own refusal, and two sentences about one line is what S4 forbids.
"""

import jax.numpy as jnp
import pytest

from rheplicant.config.context import ResolutionContext
from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight.beam_spill import _truncated_beam_spill
from rheplicant.config.resources import build_resources
from rheplicant.core.errors import StateValidationError
from rheplicant.radio import (
    BeamSpillOperator,
    GeneralPointingProjector,
    MatrixProjector,
)
from tests.config.preflight_helpers import (
    BASE_MODEL,
    BASE_OBSERVATION,
    UNREADABLE_BEAM,
    findings,
    ids,
    only,
    preflight_document,
)

#: The check ids this module is about.  **Every "and nothing else" assertion
#: below intersects with this set instead of comparing the whole report**, and
#: that is §0.3 E.11 rather than fastidiousness: :func:`_doc` is
#: ``preflight_document(...)`` patched, so every document here inherits the
#: SHARED base document's tokens -- and five sibling wave-1 branches are
#: adding checks that read those tokens, in files this one cannot see.
#: Measured by the reviewer: registering one stand-in refusal on the real
#: registry turns EIGHT tests in this module red without the intersection.
#: The idiom is ``test_preflight_model.py:110``'s.
#:
#: The whole-report property -- that the shared base document earns nothing at
#: all -- is asserted once, on the real registry, by
#: ``test_config_preflight.py::test_the_base_document_earns_no_finding_of_its_own``.
#: It belongs there and not here.
MINE = frozenset({"A50"})

#: A raw-array beam that earns nothing from A11 or A12.
NPY_BEAM = {"format": "npy", "path": "beam.npy", "nside": 4,
            "normalize": "pixel_sum", "frame": "beam_local"}

#: The same beam, cut at the horizon.  ``truncate_map`` is the ONE mode leg A
#: is about; ``none`` and ``projector_mask`` are the other two and both are
#: legal beside ``from: projector``.
TRUNCATED = {**NPY_BEAM, "horizon": {"mode": "truncate_map"}}

#: A driftscan projector over ``horn``.  ``acknowledge_float32_sky`` keeps A44
#: (a different Task 2 row, on a float32 base document) off every document
#: here, so an ``ids()`` assertion below is about A50 and nothing else.
DRIFTSCAN = {"engine": "driftscan", "beam": {"ref": "resources.beams.horn"},
             "lmax": 8, "lat_deg": {"value": 53.2367, "unit": "deg"},
             "az_deg": {"value": 0.0, "unit": "deg"},
             "el_deg": {"value": 90.0, "unit": "deg"},
             "normalize_beam": True, "acknowledge_float32_sky": True}

#: The same projector asking for the cached rotation -- leg B's subject.
#: ``lst_ref_deg`` is present because A48 would otherwise refuse it first,
#: which would make every leg-B document carry two findings.
CACHED = {**DRIFTSCAN, "optimizations": ["cache_beam_rotation"],
          "lst_ref_deg": {"value": 0.0, "unit": "deg"}}

#: ``model.beam_spill``, taking its fraction off the projector.
SPILL = {"from": "projector",
         "projector": {"ref": "resources.projectors.drift"},
         "t_ground": {"value": 300.0, "unit": "K"}}

#: A pointing this run can actually have.  Without it ``observation.pointing``
#: is ``mode: none`` by default and A52 refuses every projector reference in
#: ``model:`` -- a second finding on every document here, about a fault none
#: of these tests is written for.
POINTING = {"mode": "drift",
            "az_deg": {"value": 0.0, "unit": "deg"},
            "el_deg": {"value": 90.0, "unit": "deg"},
            "materialise": ["pointing"],
            "lst": {"mode": "uniform_turn", "n_time": "n_time",
                    "lst0_deg": {"value": 0.0, "unit": "deg"}}}


def _doc(*, beam=NPY_BEAM, projector=DRIFTSCAN, spill=SPILL, **patch):
    """The base document with one beam, one projector and one spill node."""
    sections = {
        "observation": {**BASE_OBSERVATION, "pointing": POINTING},
        "resources": {"beams": {"horn": beam},
                      "projectors": {"drift": projector}},
        "model": {**BASE_MODEL, **({"beam_spill": spill} if spill else {})},
    }
    sections.update(patch)
    return preflight_document(**sections)


def _refused_saying(document, clause: str) -> bool:
    """Is ``clause`` in **any** refusal the pass produces on ``document``?

    **Not** ``pytest.raises(ConfigError, match=clause)`` around
    ``load_document``.  ``Report.raise_if_refused`` raises ``refusals()[0]``
    and appends only a one-line tail naming where the others are, so a
    ``match=`` on the raised text asserts *"no check registered before mine
    fires on this document"* -- a statement about the REGISTRY, which five
    sibling branches are editing right now, rather than about the document.
    Reading every refusal is the same assertion with the registry taken out
    of it.
    """
    return any(clause in one.message for one in preflight(document).refusals())


@pytest.fixture
def context(tmp_path):
    """A resolution context for the two tests that drive a BUILDER directly.

    ``freq``/``time`` are ``None`` because every refusal reached here fires
    before a grid is read; ``base_dir`` is a fresh directory holding no beam,
    which is what makes ``UNREADABLE_BEAM``'s file genuinely absent.
    """
    return ResolutionContext(freq=None, time=None, dtype="float32",
                             base_dir=str(tmp_path))


class TestTheRegistry:
    def test_A50_is_this_function(self):
        assert CHECKS["A50"] is _truncated_beam_spill

    def test_the_base_document_of_this_module_earns_no_A50(self):
        """The anti-vacuity partner every positive below rests on: the
        working combination -- an untruncated beam, an uncached projector and
        a spill node -- must be silent, or "and A50 is gone" says nothing.

        Scoped to :data:`MINE`.  The stronger claim, that the shared base
        document earns NO finding at all, is
        ``test_config_preflight.py::test_the_base_document_earns_no_finding_of_its_own``'s
        and is asserted there against the real registry."""
        assert ids(_doc()) & MINE == frozenset()


class TestLegATheGroundThatVanishes:
    """``truncate_map`` and ``from: projector`` are one statement said twice.

    Measured at ``ea4839b`` on an nside-8 Gaussian horn at lmax 16, through
    ``build_resources`` and ``BeamSpillOperator.from_projector``: ``f_sky`` is
    0.9510 on the untruncated beam and 0.99924 on the truncated one, so
    ``(1 - f_sky) * 300 K`` falls from 14.7 K to 0.23 K.  Nothing raises.
    """

    def test_it_fires(self):
        found = only(_doc(beam=TRUNCATED), "A50")
        assert found.where == "model.beam_spill"
        assert found.severity == REFUSE

    def test_the_whole_message(self):
        assert only(_doc(beam=TRUNCATED), "A50").message == (
            "model.beam_spill: from: projector takes f_sky off "
            "resources.projectors.drift, whose beam resources.beams.horn is "
            "already cut at the horizon by horizon.mode: truncate_map. "
            "Cutting the beam and splitting it are two spellings of one "
            "physical statement, and writing both does not double the ground "
            "term -- it deletes it: horizon_fraction() over a truncated beam "
            "returns approximately 1.0, so (1 - f_sky) * t_ground goes to "
            "zero with nothing raised anywhere and every shape, unit and "
            "dtype still right. Measured on an nside-8 Gaussian horn at lmax "
            "16: f_sky is 0.9510 untruncated and 0.99924 truncated, which "
            "turns a 14.7 K ground contribution into 0.23 K. Either set "
            "resources.beams.horn's horizon.mode to projector_mask and keep "
            "this node as it is, or keep truncate_map and write sky_fraction: "
            "{ref: resources.beams.horn.sky_fraction} here instead of from: "
            "projector, which is the fraction the truncation already returned "
            "(check A50)."
        )

    @pytest.mark.parametrize("mode", ["none", "projector_mask"])
    def test_the_other_two_horizon_modes_are_left_alone(self, mode):
        """S2's naive mutant, killed: an A50 that refused ``horizon.mode``
        unconditionally refuses ``projector_mask``, which is the CORRECT way
        to have a masked projector and a spill node at once."""
        assert "A50" not in ids(
            _doc(beam={**NPY_BEAM, "horizon": {"mode": mode}}))

    def test_a_projector_mask_beam_beside_a_masking_projector_is_correct(self):
        """The combination the schema recommends, spelled in full: the beam
        declares ``projector_mask``, the projector does the masking, and the
        spill node reads the fraction off it."""
        assert ids(_doc(
            beam={**NPY_BEAM, "horizon": {"mode": "projector_mask"}},
            projector={**DRIFTSCAN, "horizon_mask": True})) & MINE == frozenset()

    def test_a_truncated_beam_with_no_spill_node_earns_nothing(self):
        """The other half of the same mutant.  Truncating a beam is a normal
        thing to do; it is only reading f_sky off the projector AFTERWARDS
        that deletes the ground."""
        assert "A50" not in ids(_doc(beam=TRUNCATED, spill=None))

    def test_a_truncated_beam_under_a_DIFFERENT_projector_earns_nothing(self):
        """The reference chain is followed, not assumed.  Two beams, two
        projectors, and the spill node points at the one whose beam is
        intact."""
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": POINTING},
            resources={"beams": {"horn": NPY_BEAM, "cut": TRUNCATED},
                       "projectors": {
                           "drift": DRIFTSCAN,
                           "other": {**DRIFTSCAN,
                                     "beam": {"ref": "resources.beams.cut"}}}},
            model={**BASE_MODEL, "beam_spill": SPILL})
        assert "A50" not in ids(doc)

    def test_a_horizon_that_is_not_a_mapping_is_a_stand_down(self):
        """``kinds/beams.py`` refuses that shape in its own words; a check
        that guessed past it would pre-empt the better sentence."""
        assert "A50" not in ids(_doc(beam={**NPY_BEAM, "horizon": "yes"}))

    @pytest.mark.parametrize("mode", ["truncate", "TRUNCATE_MAP", None])
    def test_a_mode_that_is_not_truncate_map_is_a_stand_down(self, mode):
        assert "A50" not in ids(
            _doc(beam={**NPY_BEAM, "horizon": {"mode": mode}}))


class TestLegBTheFractionThatCannotBeRead:
    """``optimizations: [cache_beam_rotation]`` and ``from: projector``.

    Measured at ``ea4839b``: ``to_reference_frame()`` folds the mask into the
    cached alms, ``horizon_fraction()`` then raises ``StateValidationError``,
    and through ``load_document`` that arrives 2.5 s in, from inside
    ``build_model``, naming neither the model node nor the ``optimizations:``
    key the reader has to edit.
    """

    def test_it_fires(self):
        found = only(_doc(projector=CACHED), "A50")
        assert found.where == "model.beam_spill"
        assert found.severity == REFUSE

    def test_the_whole_message(self):
        assert only(_doc(projector=CACHED), "A50").message == (
            "model.beam_spill: from: projector takes f_sky off "
            "resources.projectors.drift, which declares optimizations: "
            "[cache_beam_rotation]. to_reference_frame() folds the horizon "
            "mask into that projector's cached alms, so the unmasked "
            "denominator horizon_fraction() divides by is gone and the call "
            "raises a StateValidationError -- from inside build_model, after "
            "the beam has been read, analysed and rotated, and in a class "
            "ConfigError does not catch. Drop cache_beam_rotation from "
            "resources.projectors.drift, or, if this run needs the cached "
            "rotation, declare a second driftscan projector over the same "
            "beam without it and point this node at that one (check A50)."
        )

    def test_a_projector_with_no_optimizations_is_left_alone(self):
        assert "A50" not in ids(_doc(projector=DRIFTSCAN))

    def test_a_cached_projector_with_no_spill_node_is_left_alone(self):
        """Caching the rotation is the optimisation the schema recommends;
        it is only reading f_sky off the result that cannot work."""
        assert "A50" not in ids(_doc(projector=CACHED, spill=None))

    def test_beam_frame_is_never_what_is_followed(self):
        """S2's second naive mutant.  ``beam_frame: reference`` is what the
        OBJECT carries, and a leg B reading it off the document would never
        fire on a legal document at all -- the key is in ``_NOT_WRITABLE``
        and refused outright, by a message that names the optimisation to ask
        for instead.  Leg B follows ``optimizations:``, which is what a
        document may actually write."""
        assert "A50" in ids(_doc(projector=CACHED))
        assert "A50" not in ids(_doc(
            projector={**DRIFTSCAN, "beam_frame": "reference"}))

    def test_leg_B_pre_empts_leg_A_rather_than_joining_it(self):
        """A projector that is BOTH cached and built on a truncated beam
        raises before any fraction is computed, so the vanishing ground term
        is not what the reader would meet.  ``only`` is the assertion: two
        findings for one line is what S4 forbids, and it is exactly what a
        pair of independent ``if`` blocks would produce."""
        found = only(_doc(beam=TRUNCATED, projector=CACHED), "A50")
        assert "cache_beam_rotation" in found.message
        assert "truncate_map" not in found.message


class TestTheEngineGate:
    """§0.3 E.3(3): A50 decides ``engine: driftscan`` only.

    S4's stand-down.  On every other engine ``from_projector`` already refuses
    -- by CLASS NAME, and saying which projector kind does define the cut --
    and that sentence is better than anything this layer could write.
    """

    @pytest.mark.parametrize("engine", [
        pytest.param({"engine": "matrix", "matrix": {"zeros": [16, 12]},
                      "provenance": {"built_by": "the test suite"}},
                     id="matrix"),
        pytest.param({"engine": "general_pointing",
                      "beam": {"ref": "resources.beams.horn"},
                      "lmax": 8, "nside": 4,
                      "lat_deg": {"value": 53.2367, "unit": "deg"},
                      "normalize_beam": True,
                      "acknowledge_float32_sky": True},
                     id="general_pointing"),
        pytest.param({"engine": "nonsense"}, id="not-an-engine-at-all"),
    ])
    def test_no_other_engine_earns_A50_even_on_a_truncated_beam(self, engine):
        assert "A50" not in ids(_doc(beam=TRUNCATED, projector=engine))

    def test_and_the_better_sentence_is_the_one_that_names_the_class(self):
        """The other half of the stand-down: what arrives instead.  A
        ``MatrixProjector`` is built from an array alone, so this costs no
        beam and no spherical harmonic transform."""
        with pytest.raises(StateValidationError) as caught:
            BeamSpillOperator.from_projector(
                MatrixProjector(matrix=jnp.zeros((2, 12))),
                t_ground=jnp.asarray(300.0))
        assert str(caught.value).startswith(
            "MatrixProjector does not expose horizon_fraction()")

    def test_general_pointing_fails_the_same_criterion(self):
        """``from_projector``'s gate is ``hasattr(projector,
        'horizon_fraction')``, and this is that attribute missing -- asserted
        on the CLASS, so no projector has to be built to say it."""
        assert not hasattr(GeneralPointingProjector, "horizon_fraction")


class TestTheValueNodeRouteIsNotSpokenAboutTwice:
    """S4, the task body's own named case.

    ``{from: horizon_fraction, projector: {ref: ...}}`` is the OTHER spelling
    of "read f_sky off a projector", and it already has its own refusal in
    ``kinds/projectors.py::_horizon_fraction`` -- pinned by
    ``test_config_kind_projectors.py::TestOptimizationsAndLstRef::
    test_horizon_fraction_against_a_cached_projector_is_refused``.  A50 says
    nothing about it, in either leg.
    """

    def _derived(self, projector):
        return {"sky_fraction": {"from": "horizon_fraction",
                                 "projector": {"ref":
                                               "resources.projectors.drift"}},
                "t_ground": {"value": 300.0, "unit": "K"}}

    def test_the_derivation_against_a_cached_projector_is_not_A50s_business(
            self):
        assert "A50" not in ids(
            _doc(projector=CACHED, spill=self._derived(CACHED)))

    def test_nor_against_a_truncated_beam(self):
        assert "A50" not in ids(
            _doc(beam=TRUNCATED, spill=self._derived(DRIFTSCAN)))

    def test_a_spill_node_writing_sky_fraction_outright_is_not_either(self):
        """The remedy leg A names, applied: no ``from:`` at all."""
        assert "A50" not in ids(_doc(beam=TRUNCATED, spill={
            "sky_fraction": {"ref": "resources.beams.horn.sky_fraction"},
            "t_ground": {"value": 300.0, "unit": "K"}}))


class TestTakingTheAdvice:
    """R4.  Both legs name two remedies each and all four are followable --
    verified at ``ea4839b`` through ``build_resources`` and
    ``BeamSpillOperator.from_projector``, where each fixed document produces a
    real fraction (0.9510 on the untruncated horn) rather than a different
    refusal."""

    def test_leg_A_remedy_1_switch_the_beam_to_projector_mask(self):
        assert "A50" in ids(_doc(beam=TRUNCATED))
        assert ids(_doc(beam={**NPY_BEAM,
                              "horizon": {"mode": "projector_mask"}})
                   ) & MINE == frozenset()

    def test_leg_A_remedy_2_read_the_truncations_own_second_product(self):
        assert ids(_doc(beam=TRUNCATED, spill={
            "sky_fraction": {"ref": "resources.beams.horn.sky_fraction"},
            "t_ground": {"value": 300.0, "unit": "K"}})) & MINE == frozenset()

    def test_leg_B_remedy_1_drop_the_optimisation(self):
        assert "A50" in ids(_doc(projector=CACHED))
        assert ids(_doc(projector=DRIFTSCAN)) & MINE == frozenset()

    def test_leg_B_remedy_1_may_keep_lst_ref_deg(self):
        """Measured: ``lst_ref_deg`` is a plain ``from_beam_maps`` argument
        and a local-frame projector carrying one builds and answers
        ``horizon_fraction()``.  The remedy does not force a second edit."""
        assert ids(_doc(projector={
            **DRIFTSCAN, "lst_ref_deg": {"value": 0.0, "unit": "deg"}})
        ) & MINE == frozenset()

    def test_leg_B_remedy_2_a_second_uncached_projector(self):
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": POINTING},
            resources={"beams": {"horn": NPY_BEAM},
                       "projectors": {"drift": CACHED, "plain": DRIFTSCAN}},
            model={**BASE_MODEL, "beam_spill": {
                **SPILL, "projector": {"ref": "resources.projectors.plain"}}})
        assert ids(doc) & MINE == frozenset()


class TestTheRouteNodesCannotSee:
    """§0.3 E.10, walked rather than recorded as a false negative.

    ``inference.twin.replace.beam_spill`` reaches the same
    ``build_node_operator`` -> ``_from_route`` -> ``from_projector`` path as
    ``model.beam_spill``, and ``preflight/model.py::_nodes`` cannot see it.
    Walking it is affordable here precisely because A50 is INVENTED: a
    verbatim hoist would have carried ``model.beam_spill`` into the sentence
    and named the wrong section on this route.
    """

    def _replacing(self, spill):
        inference = preflight_document()["inference"]
        return {**inference,
                "twin": {**inference.get("twin", {}), "replace":
                         {"beam_spill": spill}}}

    def test_leg_A_fires_on_the_replace_route(self):
        found = only(_doc(beam=TRUNCATED, spill=None,
                          inference=self._replacing(SPILL)), "A50")
        assert found.where == "inference.twin.replace.beam_spill"
        assert found.message.startswith(
            "inference.twin.replace.beam_spill: from: projector takes f_sky")

    def test_leg_B_fires_on_the_replace_route(self):
        found = only(_doc(projector=CACHED, spill=None,
                          inference=self._replacing(SPILL)), "A50")
        assert found.where == "inference.twin.replace.beam_spill"
        assert "cache_beam_rotation" in found.message

    def test_both_routes_at_once_are_two_findings_and_two_paths(self):
        """A document replacing the node it also declares is two operators
        built from two specs, so it is two faults rather than one said
        twice."""
        doc = _doc(beam=TRUNCATED, inference=self._replacing(SPILL))
        assert [one.where for one in findings(doc) if one.check == "A50"] == [
            "model.beam_spill", "inference.twin.replace.beam_spill"]

    def test_the_compose_route_is_walked_too(self):
        """``compose:`` expands to its stages, which are what reach
        ``build_node_operator``; the composing mapping never does."""
        found = only(_doc(beam=TRUNCATED,
                          spill={"compose": True, "stages": [SPILL]}), "A50")
        assert found.where == "model.beam_spill.stages[0]"
        assert found.message.startswith("model.beam_spill.stages[0]: ")


class TestEveryLayerIsWalked:
    """The ``variants:`` twin.  A50 reads TWO sections -- ``model:`` and
    ``resources:`` -- so a variant can introduce it from either side."""

    def test_a_variant_that_truncates_the_beam_earns_it(self):
        doc = _doc(variants={"cut": {"resources": {"beams": {
            "horn": {"horizon": {"mode": "truncate_map"}}}}}})
        found = only(doc, "A50")
        assert found.message.startswith("variants.cut: model.beam_spill:")

    def test_a_variant_that_adds_the_spill_node_earns_it(self):
        doc = _doc(beam=TRUNCATED, spill=None,
                   variants={"split": {"model": {"beam_spill": SPILL}}})
        found = only(doc, "A50")
        assert found.message.startswith("variants.split: model.beam_spill:")

    def test_the_base_documents_own_fault_is_said_once(self):
        doc = _doc(beam=TRUNCATED,
                   variants={"a": {"runtime": {"jax_enable_x64": False}},
                             "b": {"runtime": {"jax_enable_x64": False}}})
        assert only(doc, "A50").message.startswith("model.beam_spill:")


class TestTheShapesThisCheckDeclinesToRead:
    """A stand-down is silent, and every one of these has a better sentence
    of its own somewhere below this pass."""

    @pytest.mark.parametrize("spill", [
        pytest.param({"sky_fraction": 0.9, "t_ground": 300.0},
                     id="no-from-at-all"),
        pytest.param({"from": "somewhere_else"}, id="another-from-route"),
        pytest.param({"python": "pkg.mod:factory", "from": "projector",
                      "projector": {"ref": "resources.projectors.drift"}},
                     id="the-python-hatch-wins-first"),
        pytest.param({"from": "projector", "projector": "drift"},
                     id="projector-not-a-mapping"),
        pytest.param({"from": "projector", "projector": {"ref": 7}},
                     id="ref-not-a-string"),
        pytest.param({"from": "projector",
                      "projector": {"ref": "resources.beams.horn"}},
                     id="ref-into-the-wrong-kind"),
        pytest.param({"from": "projector",
                      "projector": {"ref": "resources.projectors.absent"}},
                     id="ref-to-an-entry-that-is-not-declared"),
        pytest.param("nonsense", id="node-not-a-mapping"),
    ])
    def test_it_says_nothing(self, spill):
        assert "A50" not in ids(_doc(beam=TRUNCATED, projector=CACHED,
                                     spill=spill))

    def test_a_sub_value_reference_still_resolves_to_its_entry(self):
        """The anti-vacuity partner for the two ``ref``-shaped stand-downs
        above: a reference is cut to three segments, so a legal deeper one is
        still followed rather than dropped."""
        found = only(_doc(beam=TRUNCATED, spill={
            **SPILL,
            "projector": {"ref": "resources.projectors.drift.beam_alms"}}),
            "A50")
        assert "resources.projectors.drift" in found.message

    def test_an_extends_this_layer_cannot_resolve_is_a_stand_down(self):
        """``resolved_specs`` DROPS what it cannot resolve; a check that let
        the ``ConfigError`` out would be wrapped as "check 'A50' RAISED" and
        would hide every other finding on the document."""
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": POINTING},
            resources={"beams": {"horn": NPY_BEAM},
                       "projectors": {"drift": {"extends": "nobody"}}},
            model={**BASE_MODEL, "beam_spill": SPILL})
        assert "A50" not in ids(doc)

    def test_the_extends_copy_of_a_cached_projector_IS_followed(self):
        """S3's ``extends:`` twin, in the direction that matters: the child
        inherits ``optimizations`` and the text of the child says nothing
        about it."""
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": POINTING},
            resources={"beams": {"horn": NPY_BEAM},
                       "projectors": {"base": CACHED,
                                      "drift": {"extends": "base",
                                                "lmax": 16}}},
            model={**BASE_MODEL, "beam_spill": SPILL})
        assert "cache_beam_rotation" in only(doc, "A50").message

    def test_a_hostile_model_section_does_not_abort_the_pass(self):
        doc = _doc(beam=TRUNCATED,
                   spill={"from": "projector", "projector": {"ref": None},
                          "t_ground": [1, 2]})
        assert "A50" not in ids(doc)


class TestThePhaseThisBuys:
    """Both legs, in front of the beam rather than behind it.

    The beam here is the one whose file does not exist, so each assertion is
    symmetric: A50's own words are among the refusals and ``no_such_beam`` is
    nowhere in what the loader raises.  Measured at ``ea4839b``, leg B on a
    readable beam took 2.5 s to reach its ``StateValidationError`` and leg A
    produced no message at all.

    **The positive half reads the report, not the raised exception**, for the
    reason :func:`_refused_saying` gives: ``load_document`` surfaces
    ``refusals()[0]``, so a ``match=`` on it would be an assertion about which
    check registered first -- and five sibling branches are inserting checks
    into that registry right now.  The negative half survives either way: a
    sibling's refusal is not about ``no_such_beam`` either.
    """

    def _unreadable(self, projector):
        return preflight_document(
            observation={**BASE_OBSERVATION, "pointing": POINTING},
            resources={**UNREADABLE_BEAM,
                       "projectors": {"drift": projector}},
            model={**BASE_MODEL, "beam_spill": SPILL})

    def test_leg_B_out_ranks_the_beam(self):
        doc = self._unreadable(CACHED)
        with pytest.raises(ConfigError) as caught:
            load_document(doc)
        assert _refused_saying(doc, "(check A50).")
        assert "no_such_beam" not in str(caught.value)

    def test_leg_A_out_ranks_it_too(self):
        doc = preflight_document(
            observation={**BASE_OBSERVATION, "pointing": POINTING},
            resources={"beams": {"horn": {**UNREADABLE_BEAM["beams"]["horn"],
                                          "horizon": {"mode": "truncate_map"}}},
                       "projectors": {"drift": DRIFTSCAN}},
            model={**BASE_MODEL, "beam_spill": SPILL})
        with pytest.raises(ConfigError) as caught:
            load_document(doc)
        assert _refused_saying(doc, "(check A50).")
        assert "no_such_beam" not in str(caught.value)

    def test_a_document_that_is_fine_still_reaches_the_beam(self, context):
        """The anti-vacuity partner: with the fault removed, the beam is what
        refuses.

        **Driven through ``build_resources`` rather than ``load_document``.**
        This is the one assertion in the class that needs the loader to get
        PAST the pass, so it is the one a sibling check firing on the shared
        base document would silence -- ``load_document`` would raise the
        sibling's sentence and ``no_such_beam`` would never be reached.
        Calling the builder with this document's own ``resources:`` section
        asks the same question with the registry taken out of it, and the
        stand-down half above it is what says the pass is quiet."""
        doc = self._unreadable(DRIFTSCAN)
        assert ids(doc) & MINE == frozenset()
        with pytest.raises(ConfigError) as caught:
            build_resources(doc["resources"], context)
        assert "no_such_beam" in str(caught.value)
