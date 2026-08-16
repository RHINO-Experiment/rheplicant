"""Pre-flight A11, A12, A44 and A48: the beam and projector keys, from text.

Five hoisted refusals and one property: every one of them was already written,
in the right words, in the wrong phase.  What this module asserts is that they
now arrive before ``build_resources`` reads anything, that the words did not
change on the way, and that each of them is written in **exactly one** module
under ``src/``.

The stand-downs get as much room as the positives, deliberately.  A hoist
moves a check EARLIER by construction, so every one of them is a candidate for
pre-empting a more specific refusal, and four of this task's five rows have a
neighbour that speaks first: the beam's format-value refusal, the projector's
engine-validity refusal, the ``_NOT_WRITABLE`` gate and the
unknown-optimisation refusal.
"""

import numpy as np
import pytest

import rheplicant.config.kinds.beams as beams_module
import rheplicant.config.kinds.projectors as projectors_module
from rheplicant.config.context import ResolutionContext
from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE
from rheplicant.config.kinds.beams import _a12_normalize
from rheplicant.config.kinds.projectors import (
    _a12_normalize_beam,
    _a44_float32_sky,
    _a48_lst_ref,
)
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight.resources import (
    _b2_dtype,
    _beam_keys,
    _projector_keys,
)
from rheplicant.config.resources import build_resources, check_unknown_keys
from rheplicant.config.sections.runtime import build_runtime
from rheplicant.config.sections.twin import _TWIN_KEYS
from tests.config.message_binding import assert_bound_once
from tests.config.preflight_helpers import (
    UNREADABLE_BEAM,
    findings,
    ids,
    only,
    preflight_document,
)

#: The check ids this module is about.  Two assertions below intersect with
#: it rather than reading the whole report, for §0.3 E.11's reason: every
#: document here is ``preflight_document(...)`` patched and so carries the
#: SHARED base document's tokens, which five sibling wave-1 branches are
#: adding checks against, in files this one cannot see.
MINE = frozenset({"A11", "A12", "A44", "A48"})

#: A complete raw-array beam.  ``npy`` rather than ``cst`` because the CST
#: branch needs a directory of exports nobody may publish, and every A11 leg
#: is decided from the text either way.
NPY_BEAM = {"format": "npy", "path": "beam.npy", "nside": 4,
            "normalize": "pixel_sum", "frame": "beam_local"}

#: A complete driftscan projector over :data:`NPY_BEAM`.
#: ``acknowledge_float32_sky: true`` is load-bearing and not tidiness: A44's
#: condition is ``runtime.jax_enable_x64``, which the base document does not
#: declare, so without this key EVERY document below would carry an unrelated
#: A44 and every "and nothing else" assertion would be about the wrong thing.
DRIFTSCAN = {"engine": "driftscan", "beam": {"ref": "resources.beams.horn"},
             "lmax": 8, "lat_deg": {"value": 53.2367, "unit": "deg"},
             "az_deg": {"value": 0.0, "unit": "deg"},
             "el_deg": {"value": 90.0, "unit": "deg"},
             "normalize_beam": True, "acknowledge_float32_sky": True}

#: The other engine that reaches the shared prelude.  It takes ``nside:``
#: (which driftscan refuses) and takes neither ``optimizations:`` nor
#: ``lst_ref_deg:`` (which driftscan takes) -- which is the whole reason A44
#: and A48 part company on it.
GENERAL_POINTING = {"engine": "general_pointing",
                    "beam": {"ref": "resources.beams.horn"},
                    "lmax": 8, "nside": 4,
                    "lat_deg": {"value": 53.2367, "unit": "deg"},
                    "normalize_beam": True, "acknowledge_float32_sky": True}

#: ``engine: matrix``: no beam, no file, and an early ``return`` in front of
#: all three projector rows.
MATRIX = {"engine": "matrix", "matrix": {"zeros": [16, 12]},
          "provenance": {"built_by": "the test suite", "lat_deg": 0.0}}


def _without(spec, *keys):
    """``spec`` with ``keys`` removed -- how a test says a key is ABSENT."""
    return {key: value for key, value in spec.items() if key not in keys}


def _beams(**entries):
    """A ``resources:`` patch of beams alone."""
    return {"beams": entries}


def _with_projector(spec, beam=None):
    """A ``resources:`` patch: one beam named ``horn`` and one projector."""
    return {"beams": {"horn": beam or NPY_BEAM}, "projectors": {"drift": spec}}


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
    """A resolution context for the tests that drive a BUILDER directly.

    Module-level rather than per class: three classes below ask what
    ``build_resources`` says about a section, and every refusal they reach
    fires before a grid is read, so one context serves all of them.
    """
    np.save(tmp_path / "beam.npy", np.ones((4, 192)))
    return ResolutionContext(freq=None, time=None, dtype="float32",
                             base_dir=str(tmp_path))


class TestTheRegistry:
    """§3.1: two functions, six slots, and the bare id on every finding."""

    @pytest.mark.parametrize(("slot", "function"), [
        ("A11", _beam_keys),
        ("A12.beam", _beam_keys),
        ("A12.projector", _projector_keys),
        ("A44", _projector_keys),
        ("A48", _projector_keys),
    ])
    def test_each_slot_is_owned_by_the_function_that_decides_it(
            self, slot, function):
        """Identity, not membership.  A slot bound to some OTHER function is
        a slot this module cannot register later, and the failure would be a
        silent absence rather than a clash."""
        assert CHECKS[slot] is function

    def test_a_dotted_slot_still_carries_the_bare_id_to_the_reader(self):
        """``A12.beam`` and ``A12.projector`` are two registry SLOTS deciding
        one check, which is what lets two functions carry A12 at all --
        registering bare ``A12`` twice raises at IMPORT.  What the reader
        greps is the bare id, and every finding from either function has it."""
        beam = preflight_document(
            resources=_beams(horn=_without(NPY_BEAM, "normalize")))
        projector = preflight_document(
            resources=_with_projector(_without(DRIFTSCAN, "normalize_beam")))
        assert only(beam, "A12").check == "A12"
        assert only(projector, "A12").check == "A12"
        assert "A12.beam" not in ids(beam)


class TestEveryHoistedMessageIsPinnedWhole:
    """S1.  ``match=`` searches, so the pins that existed before this task
    left every other clause of these five sentences free to be wrong -- and
    ``test_config_kind_beams.py`` pins A12's beam half by the word
    ``"normalize"`` and one number.  These are equalities on the whole text,
    against a literal in this file."""

    def test_A12_on_a_beam(self):
        assert only(preflight_document(
            resources=_beams(horn=_without(NPY_BEAM, "normalize"))),
            "A12").message == (
            "resources.beams.horn: normalize is required and has no default; "
            "it is one of ['none', 'pixel_sum', 'solid_angle'] and got None. "
            "The output's unit is decided by the PAIR (this key, the "
            "projector's normalize_beam): an unnormalised beam with "
            "normalize_beam: false gives 32838 K on a uniform 200 K sky, and "
            "a unit-pixel-sum beam with the same flag gives 100.42 K against "
            "a 99.79 K sky. Neither half is inferable from the numbers, and "
            "no preset may supply this one."
        )

    def test_A11_leg_1_a_cst_beam_with_no_phi0_deg(self):
        assert only(preflight_document(resources=_beams(horn={
            "format": "cst", "directory": "cst/", "nside": 4,
            "normalize": "pixel_sum"})), "A11").message == (
            "resources.beams.horn: phi0_deg is required for format: cst and "
            "has no default. phi0_deg is the CST azimuth landing on the beam "
            "map's phi = 0 meridian and phi_sense is its handedness -- 'a "
            "fact about the as-built horn, not the file'. They cannot be "
            "defaulted and no preset may supply them, because a MIRRORED beam "
            "passes every integral, every peak and every "
            "azimuthally-symmetric diagnostic unchanged: there is no "
            "numerical symptom, so the only protection is that someone who "
            "knew the horn wrote the value down."
        )

    def test_A11_leg_2_a_uvbeam_declaring_frame(self):
        """Leg 2 refuses ``frame`` while leg 4 REQUIRES it, and that
        asymmetry is the naive mutant this equality exists to kill."""
        assert only(preflight_document(resources=_beams(horn={
            "format": "uvbeam", "path": "horn.beamfits", "nside": 4,
            "normalize": "pixel_sum", "frame": "beam_local"})),
            "A11").message == (
            "resources.beams.horn: ['frame'] are not written for format: "
            "uvbeam. The limTOD bridge carries the azimuth convention itself "
            "-- healpix_phi_to_uvbeam_az is the adapter, pinned by limTOD's "
            "own orientation suite -- and its output is beam_local by "
            "construction, so a declared frame is either redundant or a "
            "contradiction the maps cannot settle."
        )

    def test_A11_leg_3_a_raw_array_declaring_phi0_deg(self):
        assert only(preflight_document(resources=_beams(
            horn={**NPY_BEAM, "phi0_deg": {"value": 0.0, "unit": "deg"}})),
            "A11").message == (
            "resources.beams.horn: ['phi0_deg'] describe how a CST export's "
            "azimuth maps onto the beam-local chart and are meaningless for "
            "format: 'npy'. Requiring them everywhere is the invent-a-value "
            "habit they exist to break. For a raw array the genuinely "
            "unverifiable fact is frame:, which is required instead."
        )

    def test_A11_leg_4_a_raw_array_with_no_frame(self):
        assert only(preflight_document(
            resources=_beams(horn=_without(NPY_BEAM, "frame"))),
            "A11").message == (
            "resources.beams.horn: frame is required for format: 'npy' and is "
            "one of 'beam_local' or 'reference'. It is declared and "
            "unverifiable -- nothing in the array says which chart it was "
            "sampled on."
        )

    def test_A12_on_a_projector(self):
        assert only(preflight_document(
            resources=_with_projector(_without(DRIFTSCAN, "normalize_beam"))),
            "A12").message == (
            "resources.projectors.drift: normalize_beam is required and has "
            "no default. false returns the integral of B.T over the sphere, "
            "which is not a temperature: 32838 K against 200 K on a uniform "
            "200 K sky. The output's unit is decided by this key together "
            "with the beam's own normalize:, so neither can be inferred from "
            "the other."
        )

    def test_A44(self):
        assert only(preflight_document(resources=_with_projector(
            _without(DRIFTSCAN, "acknowledge_float32_sky"))),
            "A44").message == (
            "resources.projectors.drift: engine 'driftscan' is a real sky "
            "engine and this run is in float32. "
            "radio/sky/general_pointing.py:28-32 states that the map/alm "
            "steps carry O(10%) errors in float32 -- larger than every effect "
            "normalize_beam, phi0_deg and phi_sense are required keys for, "
            "and invisible: the maps come back finite, correctly shaped and "
            "plausibly structured. Set runtime.jax_enable_x64: true, or write "
            "acknowledge_float32_sky: true on this entry to say the error is "
            "understood. x64 is process-global and part of the hashed config."
        )

    def test_A48(self):
        assert only(preflight_document(resources=_with_projector(
            {**DRIFTSCAN, "optimizations": ["cache_beam_rotation"]})),
            "A48").message == (
            "resources.projectors.drift: optimizations contains "
            "'cache_beam_rotation', which requires lst_ref_deg. "
            "to_reference_frame() raises without one -- after the beam file "
            "has been read and analysed, which is the whole class of failure "
            "these checks run before. There is no silent default: defaulting "
            "it to lst0_deg would re-anchor the m-mode phases."
        )


class TestEachRuleHasExactlyOneBinding:
    """§2.2, as a command rather than as a review step.

    A hoist that COPIED its message into ``preflight/`` and left the original
    behind would pass every assertion above -- both copies say the same words
    on the day it lands, and then they drift.  The walker reads ``src/`` and
    counts.
    """

    @pytest.mark.parametrize("literal", [
        # A12, the beam half
        "The output's unit is decided by the PAIR (this key, the projector's "
        "normalize_beam)",
        # A11, leg 1
        "because a MIRRORED beam passes every integral",
        # A11, leg 2
        "healpix_phi_to_uvbeam_az is the adapter",
        # A11, leg 3
        "Requiring them everywhere is the invent-a-value habit they exist to "
        "break",
        # A11, leg 4
        "nothing in the array says which chart it was sampled on",
        # A12, the projector half
        "false returns the integral of B.T over the sphere",
        # A44
        "the map/alm steps carry O(10%) errors in float32",
        # A48
        "defaulting it to lst0_deg would re-anchor the m-mode phases",
    ], ids=["A12-beam", "A11-cst", "A11-uvbeam", "A11-raw-refuses",
            "A11-raw-requires-frame", "A12-projector", "A44", "A48"])
    def test_the_refusal_is_written_in_exactly_one_module(self, literal):
        assert_bound_once(literal)


class TestTheBuilderStillAsksTheSameQuestions:
    """R1, in its runnable form: the extraction must change no behaviour.

    ``build_resources`` is still the backstop for an entry the pass never saw
    -- a ``python:`` caller, a kind that grows a beam of its own -- and these
    say so by driving the builder directly.  The A11 ``uvbeam`` leg in
    particular was pinned by nothing that runs without ``pyuvdata`` installed
    (``test_config_kind_beams.py::TestUvbeam`` is behind an
    ``importorskip``), and it needs none: the leg fires before ``_maps_for``
    is reached.
    """

    def test_the_uvbeam_leg_refuses_frame_with_no_pyuvdata_anywhere_near_it(
            self, context):
        """The whole sentence, off the builder, with a path that does not
        exist -- so a leg that had moved BELOW the file read would fail on
        the file instead and this equality would go red."""
        with pytest.raises(ConfigError) as caught:
            build_resources({"beams": {"horn": {
                "format": "uvbeam", "path": "absent.beamfits", "nside": 4,
                "normalize": "pixel_sum", "frame": "beam_local"}}}, context)
        assert str(caught.value) == (
            "resources.beams.horn: ['frame'] are not written for format: "
            "uvbeam. The limTOD bridge carries the azimuth convention itself "
            "-- healpix_phi_to_uvbeam_az is the adapter, pinned by limTOD's "
            "own orientation suite -- and its output is beam_local by "
            "construction, so a declared frame is either redundant or a "
            "contradiction the maps cannot settle."
        )

    def test_the_builder_still_refuses_a_beam_with_no_normalize(self, context):
        with pytest.raises(ConfigError) as caught:
            build_resources(
                {"beams": {"horn": _without(NPY_BEAM, "normalize")}}, context)
        assert str(caught.value) == _a12_normalize(
            "resources.beams.horn", _without(NPY_BEAM, "normalize"))

    def test_the_builder_still_refuses_a_projector_with_no_normalize_beam(
            self, context):
        """No ``limtod_jax`` needed: the refusal is ahead of the beam
        resolution, which is the property being hoisted."""
        spec = _without(DRIFTSCAN, "normalize_beam")
        with pytest.raises(ConfigError) as caught:
            build_resources(_with_projector(spec), context)
        assert str(caught.value) == _a12_normalize_beam(
            "resources.projectors.drift", spec)

    def test_the_builder_still_refuses_float32_with_no_acknowledgement(
            self, context):
        spec = _without(DRIFTSCAN, "acknowledge_float32_sky")
        with pytest.raises(ConfigError) as caught:
            build_resources(_with_projector(spec), context)
        assert str(caught.value) == _a44_float32_sky(
            "resources.projectors.drift", spec, "float32")

    def test_the_builder_still_refuses_cache_beam_rotation_with_no_lst_ref(
            self, context):
        spec = {**DRIFTSCAN, "optimizations": ["cache_beam_rotation"]}
        with pytest.raises(ConfigError) as caught:
            build_resources(_with_projector(spec), context)
        assert str(caught.value) == _a48_lst_ref(
            "resources.projectors.drift", spec)

    def test_the_builder_still_asks_A48_of_general_pointing_too(self, context):
        """**The one place this pass is deliberately QUIETER than the builder
        it hoists from.**  ``build_projector``'s shared prelude asks A48 of
        both engines; the pre-flight row asks it of driftscan only, because
        applying its advice on general_pointing earns ``does not take
        ['lst_ref_deg', 'optimizations']`` (the test below measures that).
        The builder's behaviour is UNCHANGED, and this is what says so."""
        spec = {**GENERAL_POINTING, "optimizations": ["cache_beam_rotation"]}
        with pytest.raises(ConfigError) as caught:
            build_resources(_with_projector(spec), context)
        assert str(caught.value) == _a48_lst_ref(
            "resources.projectors.drift", spec)

    @pytest.mark.parametrize(("module", "helper", "section", "where"), [
        (beams_module, "_a12_normalize",
         lambda: {"beams": {"horn": _without(NPY_BEAM, "normalize")}},
         "resources.beams.horn"),
        (beams_module, "_a11_chart_keys",
         lambda: {"beams": {"horn": _without(NPY_BEAM, "frame")}},
         "resources.beams.horn"),
        (projectors_module, "_a12_normalize_beam",
         lambda: _with_projector(_without(DRIFTSCAN, "normalize_beam")),
         "resources.projectors.drift"),
        (projectors_module, "_a44_float32_sky",
         lambda: _with_projector(
             _without(DRIFTSCAN, "acknowledge_float32_sky")),
         "resources.projectors.drift"),
        (projectors_module, "_a48_lst_ref",
         lambda: _with_projector(
             {**DRIFTSCAN, "optimizations": ["cache_beam_rotation"]}),
         "resources.projectors.drift"),
    ], ids=["A12-beam", "A11", "A12-projector", "A44", "A48"])
    def test_the_builder_CALLS_the_extraction_rather_than_restating_it(
            self, context, monkeypatch, module, helper, section, where):
        """**The property that says this is a hoist and not a copy**, and the
        one assertion every other test in this task is blind to.

        Measured: putting the refusal back INLINE in ``build_beam`` while
        leaving the extracted function in place leaves the whole of
        ``tests/config`` green.  Behaviour is identical, so no behavioural
        test can see it -- and ``message_binding.assert_bound_once`` cannot
        either, because ``modules_carrying`` counts MODULES, not occurrences,
        so a duplicate inside one module collapses to "bound once".  That is
        the shape §2.2's one-binding rule is actually about (two validators
        for one property, drifting), one level below where the walker looks.

        Substituting a sentinel is what makes the call itself observable.
        ``build_beam`` and ``build_projector`` look these helpers up as module
        GLOBALS at call time, so ``monkeypatch.setattr`` on the module reaches
        the call site; an inline restatement ignores the patch and the
        assertion goes red.  The pre-flight pass is unaffected either way --
        it imported these by value at its head -- which is the point: this
        test is about the BUILDER's second opinion, not about the pass.
        """
        monkeypatch.setattr(module, helper,
                            lambda name, *_args, **_kwargs: f"{name}: SENTINEL")
        with pytest.raises(ConfigError) as caught:
            build_resources(section(), context)
        assert str(caught.value) == f"{where}: SENTINEL"


class TestA12IsAMembershipTestAndNotAPresenceOne:
    """``normalize: pixelsum`` is present and is not a convention."""

    @pytest.mark.parametrize("normalize", ["pixelsum", "PIXEL_SUM", "", None,
                                           True, 0])
    def test_a_value_outside_the_three_is_refused_like_an_omission(
            self, normalize):
        doc = preflight_document(
            resources=_beams(horn={**NPY_BEAM, "normalize": normalize}))
        assert only(doc, "A12").where == "resources.beams.horn"

    @pytest.mark.parametrize("normalize", ["none", "pixel_sum", "solid_angle"])
    def test_each_of_the_three_stands_it_down(self, normalize):
        doc = preflight_document(
            resources=_beams(horn={**NPY_BEAM, "normalize": normalize}))
        assert "A12" not in ids(doc)

    def test_normalize_beam_false_is_a_decision_and_not_an_omission(self):
        """**A12's PROJECTOR half is the opposite question**, and the asymmetry
        is deliberate rather than an oversight: on the beam the value must be
        one of three conventions, on the projector any value at all will do,
        because what has no default there is the DECISION and not a spelling
        of it.

        ``normalize_beam: false`` is legal AND load-bearing today --
        ``sections/diagnostics.py`` says ``kind: mmodes`` needs a
        ``normalize_beam: false`` projector -- so a presence test written as
        ``if spec.get("normalize_beam"):`` would refuse a document the package
        depends on.  Measured: that mutant survives BOTH of this task's test
        modules and is killed only by ``test_config_exits_mmodes.py``, which
        is another task's file and was not among this task's listed pins.  One
        line closes it here, where the property is documented.
        """
        doc = preflight_document(resources=_with_projector(
            {**DRIFTSCAN, "normalize_beam": False}))
        assert "A12" not in ids(doc)


class TestA11HasFourLegsAndTheyDisagreeAboutFrame:
    """Leg 2 REFUSES ``frame`` for ``uvbeam``; leg 4 REQUIRES it for every
    other raw array.  Swapping them is the naive implementation, and it is
    the mutant this class exists to kill."""

    def test_cst_requires_both_chart_keys(self):
        for key in ("phi0_deg", "phi_sense"):
            spec = {"format": "cst", "directory": "cst/", "nside": 4,
                    "normalize": "none",
                    "phi0_deg": {"value": 0.0, "unit": "deg"},
                    "phi_sense": "ccw"}
            doc = preflight_document(
                resources=_beams(horn=_without(spec, key)))
            assert key in only(doc, "A11").message

    def test_a_complete_cst_beam_earns_nothing(self):
        doc = preflight_document(resources=_beams(horn={
            "format": "cst", "directory": "cst/", "nside": 4,
            "normalize": "none", "phi0_deg": {"value": 0.0, "unit": "deg"},
            "phi_sense": "ccw"}))
        assert "A11" not in ids(doc)

    def test_cst_does_not_also_demand_frame(self):
        """The mirror of the leg-2/leg-4 swap: a cst beam has no ``frame:``
        and must not be asked for one."""
        doc = preflight_document(resources=_beams(horn={
            "format": "cst", "directory": "cst/", "nside": 4,
            "normalize": "none", "phi0_deg": {"value": 0.0, "unit": "deg"},
            "phi_sense": "ccw"}))
        assert "A11" not in ids(doc)

    @pytest.mark.parametrize("key,value", [
        ("frame", "beam_local"),
        ("phi0_deg", {"value": 0.0, "unit": "deg"}),
        ("phi_sense", "ccw"),
    ])
    def test_uvbeam_refuses_all_three_chart_keys(self, key, value):
        doc = preflight_document(resources=_beams(horn={
            "format": "uvbeam", "path": "horn.beamfits", "nside": 4,
            "normalize": "pixel_sum", key: value}))
        assert key in only(doc, "A11").message

    def test_a_uvbeam_with_none_of_them_earns_nothing(self):
        doc = preflight_document(resources=_beams(horn={
            "format": "uvbeam", "path": "horn.beamfits", "nside": 4,
            "normalize": "pixel_sum"}))
        assert "A11" not in ids(doc)

    @pytest.mark.parametrize("fmt,extra", [
        ("healpix", {"path": "b.fits", "order": "ring", "freq": [1.0]}),
        ("npy", {"path": "b.npy"}),
        ("npz", {"path": "b.npz"}),
        ("inline", {"maps": {"ones": [1, 192]}}),
        ("gaussian", {"fwhm_deg": {"value": 20.0, "unit": "deg"}}),
        ("python", {"python": "pkg.mod:factory"}),
    ])
    def test_every_raw_array_format_requires_frame(self, fmt, extra):
        """Written out rather than driven off ``RAW_ARRAY_FORMATS``, which
        has zero readers and would be a second source of truth: this is the
        ``else`` branch's own membership, format by format."""
        doc = preflight_document(resources=_beams(horn={
            "format": fmt, "nside": 4, "normalize": "none", **extra}))
        assert "frame is required" in only(doc, "A11").message

    def test_the_uvbeam_leg_is_not_the_raw_array_leg(self):
        """The swap, stated as one assertion: a uvbeam WITHOUT ``frame`` is
        clean and a npy WITHOUT ``frame`` is refused.  An implementation that
        exchanged legs 2 and 4 passes every single-format test above and
        fails this one."""
        uvbeam = preflight_document(resources=_beams(horn={
            "format": "uvbeam", "path": "horn.beamfits", "nside": 4,
            "normalize": "pixel_sum"}))
        npy = preflight_document(
            resources=_beams(horn=_without(NPY_BEAM, "frame")))
        assert "A11" not in ids(uvbeam)
        assert "A11" in ids(npy)


class TestTheMatrixEngineIsExemptFromAllThree:
    """§0.3 E.3(1).  ``engine: matrix`` returns at ``kinds/projectors.py``
    before A12's projector half, A44 and A48 -- and the fifteen documents in
    ``test_preflight_values.py`` that carry one all call their check function
    DIRECTLY, so every one of them stays green under an implementation that
    fires on matrix.  Nothing in the tree could see it; this can."""

    def test_a_matrix_projector_earns_no_A12_A44_or_A48_finding(self):
        doc = preflight_document(resources={"projectors": {"baked": MATRIX}})
        assert {"A12", "A44", "A48"}.isdisjoint(ids(doc))

    def test_not_even_when_it_writes_the_keys_that_would_trigger_them(self):
        """A matrix entry declaring ``optimizations`` and no
        ``normalize_beam`` in a float32 run: every trigger present, every row
        silent, because the branch returns first."""
        doc = preflight_document(resources={"projectors": {"baked": {
            **MATRIX, "optimizations": ["cache_beam_rotation"]}}})
        assert {"A12", "A44", "A48"}.isdisjoint(ids(doc))

    def test_an_engine_that_is_not_an_engine_at_all_is_also_left_alone(self):
        """``build_projector``'s first refusal names the three engines that
        exist.  A pass that answered "normalize_beam is required" to
        ``engine: nonsense`` would pre-empt it with a fix that is not the
        fault."""
        doc = preflight_document(resources={"projectors": {"p": {
            "engine": "nonsense"}}})
        assert {"A12", "A44", "A48"}.isdisjoint(ids(doc))


class TestA44FiresByDefault:
    """§0.3 E.3(4).  A44 reads like a conditional key-presence test and is
    not one: its condition lives in ``runtime:``, an absent
    ``jax_enable_x64`` means float32, and float32 is the default."""

    def test_an_unacknowledged_driftscan_projector_in_a_default_run(self):
        doc = preflight_document(resources=_with_projector(
            _without(DRIFTSCAN, "acknowledge_float32_sky")))
        assert only(doc, "A44").where == "resources.projectors.drift"

    def test_general_pointing_earns_it_too(self):
        """S3's named twin.  The two engines part company on A48 and NOT on
        A44 -- both are real sky engines and both do map/alm work."""
        doc = preflight_document(resources=_with_projector(
            _without(GENERAL_POINTING, "acknowledge_float32_sky")))
        assert only(doc, "A44").where == "resources.projectors.drift"

    def test_x64_stands_it_down(self):
        doc = preflight_document(
            runtime={"jax_enable_x64": True},
            resources=_with_projector(
                _without(DRIFTSCAN, "acknowledge_float32_sky")))
        assert "A44" not in ids(doc)

    def test_the_acknowledgement_stands_it_down(self):
        doc = preflight_document(resources=_with_projector(DRIFTSCAN))
        assert "A44" not in ids(doc)

    def test_acknowledge_float32_sky_false_is_not_an_acknowledgement(self):
        doc = preflight_document(resources=_with_projector(
            {**DRIFTSCAN, "acknowledge_float32_sky": False}))
        assert "A44" in ids(doc)

    @pytest.mark.parametrize("runtime", [
        pytest.param({"jax_enable_x64": "yes"}, id="not-a-bool"),
        pytest.param("nonsense", id="not-a-mapping"),
    ])
    def test_a_runtime_this_pass_cannot_read_stands_it_down(self, runtime):
        """``build_runtime`` refuses both of these in its own words, and a
        dtype guessed past either would pre-empt the better sentence."""
        doc = preflight_document(
            runtime=runtime,
            resources=_with_projector(
                _without(DRIFTSCAN, "acknowledge_float32_sky")))
        assert "A44" not in ids(doc)

    def test_the_dtype_this_pass_reads_is_the_one_build_runtime_reports(self):
        """The anti-drift guard on ``_b2_dtype``: the dtype A44 is decided
        against is the one the run will actually be built in.

        **Only the default spelling can be compared this way, and the reason
        is measured**: ``build_runtime({"jax_enable_x64": True})`` RAISES in
        an ordinary test process -- ``runtime.py`` refuses to promise x64 when
        ``jax.config.jax_enable_x64`` is off, because the flag is
        process-global and has to be set before the first array exists.
        Turning it on for one test would turn it on for every test in the
        worker.  The ``true`` half is pinned as a literal in the next test
        instead, which is the honest form: this pass reads the document, and
        the document is all it has.
        """
        section = {"jax_enable_x64": False}
        assert _b2_dtype({"runtime": section}) == build_runtime(section).dtype
        assert _b2_dtype({"runtime": {}}) == build_runtime({}).dtype

    def test_and_x64_true_is_float64(self):
        """The half ``build_runtime`` cannot be asked for here.  Kills a
        ``_b2_dtype`` that answered ``float32`` for both -- under which A44
        would fire on every x64 document in existence."""
        assert _b2_dtype({"runtime": {"jax_enable_x64": True}}) == "float64"


class TestA48IsDriftscanOnly:
    """§0.3 E.3(2), and the live advice loop (R4) behind it."""

    def test_a_driftscan_projector_asking_to_cache_without_an_anchor(self):
        doc = preflight_document(resources=_with_projector(
            {**DRIFTSCAN, "optimizations": ["cache_beam_rotation"]}))
        assert only(doc, "A48").where == "resources.projectors.drift"

    def test_general_pointing_stands_it_down(self):
        """S4's stand-down.  The document is wrong in A48's way AND wrong in
        a way something else says better -- ``general_pointing`` takes
        neither key -- so A48 says nothing and the builder's own sweep says
        the useful thing."""
        doc = preflight_document(resources=_with_projector(
            {**GENERAL_POINTING, "optimizations": ["cache_beam_rotation"]}))
        assert "A48" not in ids(doc)

    def test_and_the_better_sentence_is_what_the_reader_gets(self, tmp_path):
        """The other half of the stand-down: not merely that A48 is silent,
        but that the sentence which arrives instead is the one that names the
        real fault.  Measured -- applying A48's advice HERE earns ``does not
        take ['lst_ref_deg', 'optimizations']``, which is why the row is
        driftscan-only."""
        np.save(tmp_path / "beam.npy", np.ones((4, 192)))
        context = ResolutionContext(freq=None, time=None, dtype="float64",
                                    base_dir=str(tmp_path))
        followed = {**GENERAL_POINTING,
                    "optimizations": ["cache_beam_rotation"],
                    "lst_ref_deg": {"value": 0.0, "unit": "deg"}}
        with pytest.raises(ConfigError) as caught:
            build_resources(_with_projector(followed), context)
        assert str(caught.value).startswith(
            "resources.projectors.drift: engine: general_pointing does not "
            "take ['lst_ref_deg', 'optimizations'];"
        )

    def test_an_unknown_optimisation_stands_it_down(self):
        """``build_projector`` refuses ``bogus`` BEFORE it asks about
        ``lst_ref_deg``, so an A48 that spoke first would name a fix the
        reader cannot apply without fixing the other one anyway."""
        doc = preflight_document(resources=_with_projector(
            {**DRIFTSCAN,
             "optimizations": ["cache_beam_rotation", "bogus"]}))
        assert "A48" not in ids(doc)

    def test_read_horizon_fraction_stands_it_down_too(self):
        """That token has its own redirect: it PRODUCES f_sky and is not an
        optimisation at all."""
        doc = preflight_document(resources=_with_projector(
            {**DRIFTSCAN,
             "optimizations": ["cache_beam_rotation",
                               "read_horizon_fraction"]}))
        assert "A48" not in ids(doc)

    @pytest.mark.parametrize("declared", [
        pytest.param("cache_beam_rotation", id="a-bare-string"),
        pytest.param({"cache_beam_rotation": True}, id="a-mapping"),
    ])
    def test_an_optimizations_that_is_not_a_list_stands_it_down(
            self, declared):
        """``list("cache_beam_rotation")`` is a list of CHARACTERS, which is
        what ``build_projector`` will iterate; the refusal that lands there
        is about ``'c'``, and this pass has nothing better to say."""
        doc = preflight_document(resources=_with_projector(
            {**DRIFTSCAN, "optimizations": declared}))
        assert "A48" not in ids(doc)

    def test_no_optimizations_at_all_is_silent(self):
        doc = preflight_document(resources=_with_projector(DRIFTSCAN))
        assert "A48" not in ids(doc)

    def test_the_not_writable_gate_speaks_first(self):
        """``beam_frame`` is refused ahead of every row in this module, by a
        message that names ``optimizations: [cache_beam_rotation] plus
        lst_ref_deg`` as the thing to ask for instead.  An A48 in front of it
        would tell the reader to add the key the other refusal is about."""
        doc = preflight_document(resources=_with_projector(
            {**DRIFTSCAN, "beam_frame": "reference",
             "optimizations": ["cache_beam_rotation"]}))
        assert {"A12", "A44", "A48"}.isdisjoint(ids(doc))

    def test_taking_its_advice_clears_it(self):
        """R4: the remedy A48 names is followable, and following it leaves
        the entry clean at this phase.

        **That the fixed entry then BUILDS is asserted by a shipped test
        rather than re-asserted here**:
        ``test_config_kind_projectors.py::TestOptimizationsAndLstRef::
        test_horizon_fraction_against_a_cached_projector_is_refused``
        constructs exactly this pair -- ``optimizations:
        [cache_beam_rotation]`` with ``lst_ref_deg`` -- and gets a projector
        back.  Building one costs 6.7 s in a cold process (measured), which
        is over this plan's ~2 s per-test bound, and paying it twice to learn
        the same thing is what the bound is for.
        """
        followed = {**DRIFTSCAN, "optimizations": ["cache_beam_rotation"],
                    "lst_ref_deg": {"value": 0.0, "unit": "deg"}}
        assert "A48" in ids(preflight_document(resources=_with_projector(
            {**DRIFTSCAN, "optimizations": ["cache_beam_rotation"]})))
        assert "A48" not in ids(
            preflight_document(resources=_with_projector(followed)))


class TestTakingTheAdviceClearsTheOtherRows:
    """R4 for the four rows whose remedy is a key on the entry itself."""

    @pytest.mark.parametrize(("check", "broken", "followed"), [
        ("A12", _without(NPY_BEAM, "normalize"),
         {**NPY_BEAM, "normalize": "pixel_sum"}),
        ("A11", _without(NPY_BEAM, "frame"),
         {**NPY_BEAM, "frame": "beam_local"}),
        ("A11", {**NPY_BEAM, "phi0_deg": {"value": 0.0, "unit": "deg"}},
         NPY_BEAM),
    ], ids=["A12-write-normalize", "A11-write-frame", "A11-drop-phi0_deg"])
    def test_a_beam_that_takes_it_earns_nothing(self, check, broken,
                                                followed):
        assert check in ids(preflight_document(resources=_beams(horn=broken)))
        assert check not in ids(
            preflight_document(resources=_beams(horn=followed)))

    @pytest.mark.parametrize(("check", "broken", "followed"), [
        ("A12", _without(DRIFTSCAN, "normalize_beam"), DRIFTSCAN),
        ("A44", _without(DRIFTSCAN, "acknowledge_float32_sky"), DRIFTSCAN),
    ], ids=["A12-write-normalize_beam", "A44-acknowledge"])
    def test_a_projector_that_takes_it_earns_nothing(self, check, broken,
                                                     followed):
        assert check in ids(
            preflight_document(resources=_with_projector(broken)))
        assert check not in ids(
            preflight_document(resources=_with_projector(followed)))

    def test_A44s_other_remedy_is_the_runtime_one_and_it_also_works(self):
        """The message names two fixes and a reader may pick either."""
        assert "A44" not in ids(preflight_document(
            runtime={"jax_enable_x64": True},
            resources=_with_projector(
                _without(DRIFTSCAN, "acknowledge_float32_sky"))))

    def test_a_beam_that_takes_the_advice_actually_builds(self, tmp_path):
        """Not merely quiet: the fixed entry is one ``build_resources``
        accepts."""
        np.save(tmp_path / "beam.npy", np.ones((4, 192)))
        context = ResolutionContext(freq=None, time=None, dtype="float32",
                                    base_dir=str(tmp_path))
        built = build_resources({"beams": {"horn": NPY_BEAM}}, context)
        assert built.resources["resources.beams.horn"].maps.shape == (4, 192)


class TestExtendsIsResolvedBeforeAnythingIsAsked:
    """S3's first named twin, and §2.4's measured TRAP.

    The text of an entry is not the spec its builder sees.  A pre-pass
    reading the raw section refuses an entry whose ``normalize:`` came from
    its parent and misses one whose parent's ``phi0_deg`` was deleted -- both
    directions, both here.
    """

    def test_a_child_inheriting_the_key_is_not_refused(self):
        doc = preflight_document(resources=_beams(
            base=NPY_BEAM,
            child={"extends": "base", "nside": 8}))
        assert "A12" not in ids(doc)
        assert "A11" not in ids(doc)

    def test_a_child_inheriting_a_key_it_must_not_have_IS_refused(self):
        """The other direction: the parent is a legal raw array, the child
        turns it into a ``uvbeam``, and the inherited ``frame:`` is now the
        fault.  A raw read of the child's own two keys sees nothing."""
        doc = preflight_document(resources=_beams(
            base=NPY_BEAM,
            child={"extends": "base", "format": "uvbeam",
                   "path": "horn.beamfits"}))
        assert only(doc, "A11").where == "resources.beams.child"

    def test_a_child_DELETING_the_key_is_refused(self):
        """``~key: null`` deletes.  A raw read of the child sees ``~frame``
        and no ``frame``, and a merge-unaware check would report the parent's
        value as if it survived."""
        doc = preflight_document(resources=_beams(
            base=NPY_BEAM, child={"extends": "base", "~frame": None}))
        assert only(doc, "A11").where == "resources.beams.child"

    def test_a_projector_child_inherits_its_acknowledgement(self):
        doc = preflight_document(resources={
            "beams": {"horn": NPY_BEAM},
            "projectors": {"base": DRIFTSCAN,
                           "drift": {"extends": "base", "lmax": 16}}})
        assert "A44" not in ids(doc)

    def test_a_projector_child_that_deletes_it_earns_A44(self):
        doc = preflight_document(resources={
            "beams": {"horn": NPY_BEAM},
            "projectors": {"base": DRIFTSCAN,
                           "child": {"extends": "base",
                                     "~acknowledge_float32_sky": None}}})
        assert only(doc, "A44").where == "resources.projectors.child"

    def test_an_extends_this_layer_cannot_resolve_is_a_stand_down(
            self, context):
        """``resolved_specs`` DROPS what it cannot resolve rather than
        raising -- a check that let the ``ConfigError`` out would be wrapped
        as "check 'A11' RAISED" and would hide every other finding on the
        document.  ``build_resources`` stays the backstop and says the right
        sentence at the right phase.

        **The backstop half drives the builder, not ``load_document``.**  The
        sentence belongs to ``_resolved_spec`` and is only ever reached if
        nothing refuses the document first, so pinning it through the loader
        would really assert "no sibling check fires on the shared base
        document" -- a claim about a registry five other branches are editing.
        """
        section = _beams(child={"extends": "nobody", "format": "npy",
                                "nside": 4})
        assert MINE.isdisjoint(ids(preflight_document(resources=section)))
        with pytest.raises(ConfigError, match="which resources.beams"):
            build_resources(section, context)


class TestEveryLayerIsWalked:
    """§0.3 F.5(1).  ``resolved_specs`` takes a SECTION, so the layer walk
    belongs to the caller -- and a check that called it once on
    ``document["resources"]`` would close the base route and leave the
    ``variants:`` twin wide open."""

    def test_a_variant_that_introduces_the_fault_is_reported(self):
        doc = preflight_document(
            resources=_beams(horn=NPY_BEAM),
            variants={"no_frame": {"resources": {"beams": {
                "horn": {"~frame": None}}}}})
        found = only(doc, "A11")
        assert found.message.startswith("variants.no_frame: ")
        assert "frame is required" in found.message

    def test_the_base_documents_own_fault_is_said_ONCE(self):
        """``_task3_over_layers``' whole job: a document with one fault and
        two variants must not hand the reader the same sentence three
        times."""
        doc = preflight_document(
            resources=_beams(horn=_without(NPY_BEAM, "frame")),
            variants={"a": {"runtime": {"jax_enable_x64": False}},
                      "b": {"runtime": {"jax_enable_x64": False}}})
        assert only(doc, "A11").message.startswith("resources.beams.horn:")

    def test_a_variant_can_introduce_A44_through_the_runtime_alone(self):
        """The layer walk is not only about ``resources:``.  A44's condition
        lives in ``runtime:``, so a variant that turns x64 OFF makes an
        acknowledged-nowhere projector a fault it did not used to be."""
        doc = preflight_document(
            runtime={"jax_enable_x64": True},
            resources=_with_projector(
                _without(DRIFTSCAN, "acknowledge_float32_sky")),
            variants={"cheap": {"runtime": {"jax_enable_x64": False}}})
        found = only(doc, "A44")
        assert found.message.startswith("variants.cheap: ")


class TestThePhaseThisBuys:
    """The point of the task, as a property rather than as a claim.

    Every document here carries :data:`~tests.config.preflight_helpers.
    UNREADABLE_BEAM` -- a beam whose file does not exist -- so each assertion
    is symmetric: the violation's own words are among the refusals and
    ``no_such_beam`` is nowhere in what the loader raises.  Measured at
    ``ea4839b``, before this module existed, each of them died on the file
    instead, 0.106 s in.

    **The positive half reads the report and the negative half reads the
    raised exception**, for the reason :func:`_refused_saying` gives: the
    loader surfaces ``refusals()[0]``, so a ``match=`` on it would be an
    assertion about which check registered first.  The negative half needs no
    such care -- a sibling's refusal is not about ``no_such_beam`` either, so
    "the beam was never read" survives any registry.
    """

    def test_a_projector_key_out_ranks_the_beam_it_references(self):
        doc = preflight_document(resources={
            **UNREADABLE_BEAM,
            "projectors": {"drift": _without(DRIFTSCAN, "normalize_beam")}})
        with pytest.raises(ConfigError) as caught:
            load_document(doc)
        assert _refused_saying(doc, "normalize_beam is required")
        assert "no_such_beam" not in str(caught.value)

    def test_A44_out_ranks_it_too(self):
        doc = preflight_document(resources={
            **UNREADABLE_BEAM,
            "projectors": {"drift": _without(DRIFTSCAN,
                                             "acknowledge_float32_sky")}})
        with pytest.raises(ConfigError) as caught:
            load_document(doc)
        assert _refused_saying(doc, "acknowledge_float32_sky")
        assert "no_such_beam" not in str(caught.value)

    def test_A48_out_ranks_it_too(self):
        doc = preflight_document(resources={
            **UNREADABLE_BEAM,
            "projectors": {"drift": {**DRIFTSCAN,
                                     "optimizations": ["cache_beam_rotation"]}}})
        with pytest.raises(ConfigError) as caught:
            load_document(doc)
        assert _refused_saying(doc, "which requires lst_ref_deg")
        assert "no_such_beam" not in str(caught.value)

    def test_a_SECOND_beams_missing_key_is_said_before_the_FIRST_is_read(self):
        """``build_resources`` has no validation pre-pass, so today the first
        beam is built entirely before the second is looked at.  Here the
        first beam is the one that cannot be read, and the second one's
        missing ``normalize:`` still wins."""
        doc = preflight_document(resources={"beams": {
            **UNREADABLE_BEAM["beams"],
            "second": _without(NPY_BEAM, "normalize")}})
        with pytest.raises(ConfigError) as caught:
            load_document(doc)
        assert _refused_saying(
            doc, "resources.beams.second: normalize is required")
        assert "no_such_beam" not in str(caught.value)

    def test_every_finding_arrives_together_rather_than_one_per_round_trip(
            self):
        """Collect, do not raise: four faults across two entries, one pass."""
        doc = preflight_document(resources={
            "beams": {"horn": _without(NPY_BEAM, "normalize", "frame")},
            "projectors": {"drift": _without(DRIFTSCAN, "normalize_beam",
                                             "acknowledge_float32_sky")}})
        assert {"A11", "A12", "A44"} <= ids(doc)
        mine = [(one.check, one.where) for one in findings(doc)
                if one.check in {"A11", "A12", "A44"}]
        assert mine == [("A12", "resources.beams.horn"),
                        ("A11", "resources.beams.horn"),
                        ("A12", "resources.projectors.drift"),
                        ("A44", "resources.projectors.drift")]

    def test_every_finding_here_is_a_refusal(self):
        doc = preflight_document(resources={
            "beams": {"horn": _without(NPY_BEAM, "normalize")},
            "projectors": {"drift": _without(DRIFTSCAN, "normalize_beam")}})
        assert [one.severity for one in findings(doc)
                if one.check in {"A11", "A12", "A44", "A48"}] == [REFUSE,
                                                                  REFUSE]


class TestTheShapesThisPassDeclinesToRead:
    """§3.2(c): a check that cannot read what it needs stands down SILENTLY,
    and refusing on "I could not tell" refuses documents that build."""

    @pytest.mark.parametrize("section", [
        pytest.param(None, id="no-resources-section"),
        pytest.param({}, id="empty"),
        pytest.param({"arrays": {"flat": {"ones": ["n_freq"]}}}, id="no-beams"),
        pytest.param({"beams": "nonsense"}, id="beams-not-a-mapping"),
        pytest.param({"beams": {"horn": "nonsense"}}, id="entry-not-a-mapping"),
        pytest.param({"beams": {"horn": {}}}, id="no-format-at-all"),
        pytest.param({"beams": {"horn": {"format": "nonsense"}}},
                     id="a-format-that-is-not-one"),
        pytest.param({"projectors": {"p": {}}}, id="projector-with-no-engine"),
    ])
    def test_it_says_nothing(self, section):
        doc = preflight_document(resources=section)
        assert {"A11", "A12", "A44", "A48"}.isdisjoint(ids(doc))

    def test_an_unknown_format_is_left_to_its_own_refusal(self, context):
        """§0.3 E.3(5).  An A11 that reached its raw-array ``else`` on
        ``format: nonsense`` would answer "frame is required" to a document
        whose fault is the format, pre-empting the message that lists the
        eight formats that exist.

        The better sentence is ``build_beam``'s first question, so it is
        asked of the BUILDER: reached through ``load_document`` it would only
        arrive while no other registered check refuses this document, which
        is a property of the registry rather than of the beam."""
        section = _beams(horn={"format": "nonsense", "nside": 4})
        assert MINE.isdisjoint(ids(preflight_document(resources=section)))
        with pytest.raises(ConfigError, match="the beam formats are"):
            build_resources(section, context)

    def test_a_hostile_resources_section_does_not_abort_the_pass(self):
        """§2.3's TRAP.  A check that RAISED would be wrapped as "pre-flight
        check 'A11' RAISED ..." and would discard every other finding on the
        document -- while every ``match=`` pin in the tree still passed,
        because ``match=`` searches."""
        doc = preflight_document(
            model={"gian": {}},
            resources={"beams": {"horn": {"format": ["npy"], "nside": None}},
                       "projectors": {"p": {"engine": {"driftscan": 1}}}})
        assert "A2" in ids(doc)
        assert {"A11", "A12", "A44", "A48"}.isdisjoint(ids(doc))


class TestTheRouteThisModuleDoesNotWalk:
    """§0.3 E.10, answered for this module: it does NOT walk
    ``inference.twin.replace``, and that is not a false negative.

    That route replaces an OPERATOR, and ``inference.twin`` has no spelling
    for a resource at all -- so a beam or a projector can only ever arrive
    through ``resources:`` on some layer, which is what is walked.  Written
    as an assertion rather than as a sentence in a docstring, because a
    sentence is what goes stale.
    """

    def test_inference_twin_takes_no_resources_block(self):
        assert _TWIN_KEYS == frozenset({"without", "replace"})

    def test_a_resources_block_smuggled_under_the_twin_reaches_no_builder(
            self):
        """And earns nothing from this module either -- the entry is not in
        ``document["resources"]`` on any layer, so there is nothing for
        ``resolved_specs`` to find and nothing to stand down on.

        The refusal half calls the sweep ``sections/twin.py:44`` calls, with
        the arguments it calls it with, rather than reaching it through
        ``load_document``.  That is not a re-implementation -- it is the same
        function on the same inputs -- and it is what keeps the assertion
        about the twin's key set rather than about which check happens to
        refuse this document first."""
        doc = preflight_document(inference={
            **preflight_document()["inference"],
            "twin": {"without": ["noise"],
                     "resources": {"beams": {"ghost": {"format": "npy",
                                                       "nside": 4}}}}})
        assert MINE.isdisjoint(ids(doc))
        with pytest.raises(ConfigError, match=r"twin: does not take"):
            check_unknown_keys("inference.twin",
                               dict(doc["inference"]["twin"]),
                               _TWIN_KEYS, label="twin:")
