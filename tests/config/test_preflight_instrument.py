"""``preflight/instrument.py`` -- A13's text legs, A40 and A47.

**What this module is written against**, because the plan's §0.3 E.8 measured
that nothing in the tree could tell any of these three messages from a rewrite
of it: A13's presence message had **no pin at all**; ``line_width`` was pinned
by ``match="line_width must be > 0"``, twenty-four of its hundred and sixty-
eight characters; ``protect_floor`` by one word of two hundred and forty-three;
A40 and A47 by ``in message`` substrings.  So every message below is pinned by
**equality on its whole text**, against a literal in this file.

The four A13 sentences are this layer's own (they are inventions -- see the
module docstring in ``src`` for why a verbatim hoist was neither available nor
correct), and the A40 and A47 sentences are the section's own with a tail:
:func:`assert_bound_once` is what says the hoisted two were MOVED rather than
copied, and the equality pins are what say the whole sentence is still the one
that arrives.
"""

import pytest

from rheplicant.config.delivery import field_specs, mode_of
from rheplicant.config.errors import ConfigError
from rheplicant.config.preflight import CHECKS, preflight
from rheplicant.config.preflight import instrument as instrument_module
from rheplicant.config.preflight.instrument import (
    _entry_class,
    _instrument_text,
    _region_key,
    _static_fields,
    _text_number,
    _tone_text,
)
from rheplicant.config.sections.model import operator_table
from rheplicant.radio.instrument.calibration import CWCalibrationOperator
from tests.config.message_binding import assert_bound_once, modules_carrying
from tests.config.preflight_helpers import (
    UNREADABLE_BEAM,
    ids,
    only,
    preflight_document,
)

# --- the documents -----------------------------------------------------------

#: A ``cw_tone`` entry that BUILDS on the base document's 60-85 MHz / 8-channel
#: band.  5 MHz sits between ``A13.grid``'s floor (one 3.57 MHz channel) and
#: its ceiling (7.14 MHz), so a test about a TEXT leg is never answered by the
#: grid leg Task 1 shipped -- which is the collision this fixture exists to
#: keep out of every case below.
TONE = {"amplitude": 5000.0,
        "tone_freq": {"value": 70.0, "unit": "MHz"},
        "line_width": {"value": 5.0, "unit": "MHz"}}

#: A well-formed array-producing value node.  ``endpoint:`` is required and has
#: no default (``arrays.py:81``), so omitting it earns ``linspace``'s own
#: refusal instead of A40's.
LINSPACE = {"linspace": {"start": 1.0, "stop": 2.0, "num": 4, "endpoint": True}}

#: The ``python:`` spelling of a shipped class, which is what makes an ``at:``
#: legal at all (``compose.py:296-301``).
PY_GAIN = {"python": "rheplicant.radio:GainOperator"}

#: The ``python:`` spelling of the tone, which places a ``CWCalibrationOperator``
#: at ANY node -- the route that makes the ``cw_tone`` key the wrong thing to
#: key this check on.
PY_TONE = {"python": "rheplicant.radio.instrument.calibration:CWCalibrationOperator"}

#: The three ids this task owns.  Every "and nothing else" assertion in this
#: module intersects with it (R8, §0.3 E.11): five sibling branches are
#: registering checks against the same shared base document and cannot see this
#: file, so an unintersected `ids(doc) == …` is green here and red at merge.
MINE = frozenset({"A13", "A40", "A47"})


def tone_document(**overrides):
    """The base document with a valid ``model.cw_tone``, patched."""
    return preflight_document(model={"cw_tone": {**TONE, **overrides}})


def replace_document(patch, model=None):
    """The base document with ``inference.twin.replace`` carrying ``patch``.

    **The write is NESTED and that is required rather than tidy.**
    ``test_config_fixture_contract._rolls_its_own``'s route B counts a
    depth-1 assignment of a non-empty ``inference`` block as rolling one's
    own document, and it is right to: ``exit_helpers._repaired`` puts
    ``twin: {without: [noise]}`` there, and a module that replaces the block
    hands every one of its tests a document whose two twins are the same
    object.  Measured -- the first draft of this file wrote ``doc["inference"]
    = {**doc["inference"], ...}`` and that census went red on it.  Writing one
    level further in edits the repaired block instead of replacing it.
    """
    doc = preflight_document(model=model if model is not None else {"cw_tone": TONE})
    doc["inference"]["twin"] = {**(doc["inference"].get("twin") or {}),
                                "replace": patch}
    return doc


def variant_document(section, patch):
    """The base document with one variant patching ``section``.

    **The variant twin has two halves and only one of them is the layer
    walk's**, measured rather than taken from plan §0.3 F.5(1)'s wording:

    * ``load_document(doc, variant="twin_route")`` -- ``document.py:77``
      applies the SELECTED variant BEFORE ``preflight(doc)`` runs, so a check
      reading ``document[...]`` already sees the merged mapping.  This half
      needs no layer walk and would pass without one.
    * ``load_document(doc)`` -- no variant requested, so the base mapping is
      what the pass gets and the patch is never read at all.  **This half is
      what ``_task3_over_layers`` buys**, and it is the one that a check
      reading ``document[...]`` loses.

    Each test below says which of the two it proves.
    """
    doc = preflight_document()
    doc["variants"] = {**(doc.get("variants") or {}),
                       "twin_route": {section: patch}}
    return doc


# --- the message literals, quoted whole --------------------------------------

_A13_TAIL = (
    "This is decided from the entry's own words, before build_resources reads "
    "the beam. Today it arrives from inside build_model -- the presence rule "
    "from sections/model.py, the value bounds from "
    "CWCalibrationOperator.__check_init__ as a StateValidationError, which is "
    "a SIBLING of ConfigError rather than a subclass, so `except ConfigError` "
    "does not see it either (check A13)."
)

A13_PRESENCE = (
    "model.cw_tone: a CW calibration tone declares ['line_width'] nowhere, and "
    "CWCalibrationOperator gives it no default on purpose: line_width is the "
    "spectrometer's own channel response in Hz, tone_freq is where the line "
    "sits in the band, and amplitude is the level the tone contributes in "
    "total. A tone the operator has to guess one of those for monitors "
    "nothing, because the gain it is meant to track absorbs it exactly. "
) + _A13_TAIL

A13_WIDTH = (
    "model.cw_tone.line_width: -1e+06 Hz is not above zero. The width is a "
    "scale, not an offset: it divides the frequency offset that the lineshape "
    "is evaluated at, so zero divides by zero and a negative value evaluates "
    "the shape mirrored about the centre before it is normalised. "
) + _A13_TAIL

A13_FLOOR = (
    "model.cw_tone.protect_floor: 2 is outside (0, 1]. It is read as a "
    "fraction of the tone's own peak channel, so 1 protects the peak channel "
    "alone and anything at or below 0 protects the whole band -- which hands "
    "every channel's RFI verdict to a calibrator that touches one line. Above "
    "1 protects nothing at all and the flagger then eats the tone it was told "
    "to keep. "
) + _A13_TAIL

A13_LINESHAPE = (
    "model.cw_tone.lineshape: 'nope' is not a lineshape this operator "
    "evaluates; it takes ['sinc2', 'gaussian']. The two are not "
    "interchangeable spellings of one curve and line_width does not mean the "
    "same thing in each -- for 'sinc2' it is the offset to the first null of a "
    "critically sampled unwindowed FFT, and for 'gaussian' it is the standard "
    "deviation of an apodised polyphase channel. "
) + _A13_TAIL

#: ``delivery.py::_refuse_array_form``'s whole sentence, with this pass's path
#: in front and its tail behind.  The middle is what
#: :func:`assert_bound_once` proves lives in exactly one module.
A40_DELIVERY = (
    "Field 'line_width' is static -- equinox puts it in the treedef, where it "
    "is part of the jit cache key -- and a 'linspace' form produces an array. "
    "Measured, this fails in three different ways depending on the field: "
    "ADCOperator(n_bits=Array(12)) warns 'A JAX array is being set as "
    "static!' and then raises; ForegroundOperator(ref_freq=Array(...)) only "
    "warns, so it constructs, the forward numbers are unchanged, and "
    "filter_grad hands back the static value where a gradient belongs; "
    "FlaggingOperator.threshold has no check at all and detonates later at an "
    "unrelated pytree comparison. Write a single number here, or -- if the "
    "quantity really varies -- bind it to a field that is traced."
)

A40 = (
    f"model.cw_tone.line_width: {A40_DELIVERY} That refusal is "
    "config/delivery.py's own, moved in front of the build: it runs inside "
    "deliver(), once per field of a node being constructed, so on a document "
    "with a beam it used to arrive after the CST directory had been read and "
    "analysed. The form key is written in the document, and whether the field "
    "is static is eqx.field(...) metadata, so neither needs a value resolved "
    "(check A40)."
)

#: ``paths.py::refuse_misaddressed_region``'s whole sentence.
A47_PATHS = (
    "A multi-node at: region covering ['gain', 'noise'] is written under the "
    "key 'bandpass', but a region is addressed in the assembly by its LAST "
    "covered node id -- here 'noise'. assembly['bandpass'] would raise "
    "KeyError, and any into: path with that head would fail as a name error "
    "rather than as the 'landed on static configuration' refusal. Name the "
    "entry 'noise'. (A region is *entered* at its first node, which is where "
    "slot kinds are screened -- that is a different node and a different "
    "check.)"
)

#: The two literals :func:`assert_bound_once` is given, in the form its ``ast``
#: harvest produces: it folds every interpolation to one character, so a
#: sentence quoted with its holes filled in matches **nothing** and the walker
#: reports "bound 0 times" -- which is the shape that passes a ``>= 1``
#: assertion and fails an ``== 1`` one for the wrong reason.  Each is the
#: longest hole-free run of its own message, and each is chosen to be unique to
#: the function that raises it: ``config/paths.py`` says *"a region is
#: addressed by its LAST covered node"* in a SECOND message too
#: (``resolve_path_on``'s walk failure), so the clause below is taken from
#: after that phrase rather than around it.
A40_BOUND = (
    "form produces an array. Measured, this fails in three different ways "
    "depending on the field: ADCOperator(n_bits=Array(12)) warns 'A JAX array "
    "is being set as static!' and then raises; "
    "ForegroundOperator(ref_freq=Array(...)) only warns, so it constructs, the "
    "forward numbers are unchanged, and filter_grad hands back the static "
    "value where a gradient belongs; FlaggingOperator.threshold has no check "
    "at all and detonates later at an unrelated pytree comparison. Write a "
    "single number here, or -- if the quantity really varies -- bind it to a "
    "field that is traced."
)

A47_BOUND = (
    "would raise KeyError, and any into: path with that head would fail as a "
    "name error rather than as the 'landed on static configuration' refusal. "
    "Name the entry"
)

#: This module's own four sentences, hole-free, so a later copy-paste of one
#: into a second module is caught by the same walker the hoists use.  They are
#: inventions, so "exactly one" means ``config/preflight/instrument.py``.
A13_BOUND = (
    "Hz is not above zero. The width is a scale, not an offset: it divides the "
    "frequency offset that the lineshape is evaluated at, so zero divides by "
    "zero and a negative value evaluates the shape mirrored about the centre "
    "before it is normalised.",
    "is outside (0, 1]. It is read as a fraction of the tone's own peak "
    "channel, so 1 protects the peak channel alone and anything at or below 0 "
    "protects the whole band -- which hands every channel's RFI verdict to a "
    "calibrator that touches one line. Above 1 protects nothing at all and the "
    "flagger then eats the tone it was told to keep.",
    "no default on purpose: line_width is the spectrometer's own channel "
    "response in Hz, tone_freq is where the line sits in the band, and "
    "amplitude is the level the tone contributes in total. A tone the operator "
    "has to guess one of those for monitors nothing, because the gain it is "
    "meant to track absorbs it exactly.",
    ". The two are not interchangeable spellings of one curve and line_width "
    "does not mean the same thing in each -- for 'sinc2' it is the offset to "
    "the first null of a critically sampled unwindowed FFT, and for 'gaussian' "
    "it is the standard deviation of an apodised polyphase channel.",
)

A47 = (
    f"model.bandpass: {A47_PATHS} That refusal is config/paths.py's own, moved "
    "in front of the build: compose._single calls it on the line AFTER "
    "build_node_operator, so the operator for the very entry that is "
    "misaddressed is constructed first, and on a region covering a sky node "
    "that means the projector has already been built (check A47)."
)


class TestTheModuleIsWiredIn:
    """The registration itself, in the four legal forms and no other."""

    @pytest.mark.parametrize("slot", ["A13.text", "A40", "A47"])
    def test_each_slot_is_this_modules_own_function(self, slot):
        assert CHECKS[slot] is _instrument_text

    def test_the_three_slots_share_ONE_function_and_therefore_one_layer_walk(self):
        """The cost property, asserted structurally rather than by a clock.

        ``_task3_over_layers`` calls ``apply_variant`` -- a ``deepcopy`` --
        once per declared variant PER REGISTERED CHECK.  Measured cold on the
        shipped guard's own twenty-variant / forty-run document, fresh
        processes, best/median: **22.6 / 23.4 ms** without this module,
        **31.0 / 32.1 ms** with these three ids on one function, and
        **42.7 / 78.3 ms** with them registered separately -- against a 50 ms
        budget, so the separate form is already over on the median.  A timing
        test for that is a flake; the property the timing rests on is that the
        three ids resolve to one object, and THAT cannot flake.
        """
        assert len({id(CHECKS[slot]) for slot in ("A13.text", "A40", "A47")}) == 1

    def test_the_foot_import_is_what_registers_these_three(self):
        """The wiring, named as its own property.

        With all three ids on ONE function there is exactly one wiring point
        for three rules, and deleting the foot-import line leaves **67 of this
        module's tests green** — the test module's own
        ``from …preflight.instrument import …`` runs the ``@register``
        decorators, so importing the module under test hides the fact that the
        package does not.  That is the Task 6 blind spot.  Measured: the
        shared ``TestTheFootImportCannotRot`` catches it in two tests, and
        this module caught it in exactly one — the name-shadowing test below,
        by accident, because it happens to read ``_instrument_checks``.  This
        assertion says the property out loud so a later rewrite of that test
        cannot take three rules dark together.
        """
        import sys

        package = sys.modules["rheplicant.config.preflight"]
        assert package._instrument_checks is instrument_module

    def test_the_module_name_does_not_shadow_an_entry_point(self):
        """Plan §0.3 F.5(8), which cost Task 1 a silent non-registration.

        ``inflight.axes()`` (the function) and ``inflight/axes.py`` (the
        module) collided, and BOTH foot-import spellings were wrong in
        silence: one bound the FUNCTION and registered the checks nowhere with
        nothing failing.  ``instrument`` is not a name
        ``preflight/__init__.py`` defines, so the aliased foot import can bind
        nothing but the module -- and this goes red the day someone adds a
        ``def instrument(...)`` entry point beside it.

        **Read out of ``sys.modules`` and not by importing the name**, because
        this repository already ships one collision of exactly that shape one
        level up: ``import rheplicant.config.preflight as p`` binds the
        FUNCTION ``rheplicant.config.preflight`` that ``config/__init__.py``
        re-exports, not the subpackage.  A test written the obvious way would
        be asserting about the wrong object.
        """
        import sys

        package = sys.modules["rheplicant.config.preflight"]
        assert package._instrument_checks is instrument_module
        assert package.instrument is instrument_module
        assert instrument_module is sys.modules[
            "rheplicant.config.preflight.instrument"]


class TestThePhaseActuallyMoved:
    """§5's box, which is the whole point of the task and had no test.

    *"For each of Tasks 1-6, a document carrying that task's violation AND
    ``preflight_helpers.UNREADABLE_BEAM`` refuses with the VIOLATION, and
    ``no_such_beam`` does not appear in the message."*  The module docstring
    asserts this in prose four times; nothing drove it, so putting the
    pre-flight hook back below ``build_resources`` took nothing red here.

    ``UNREADABLE_BEAM`` is a `resources:` patch naming a beam file that does
    not exist, and reaching it costs 0.115 s and a read — it is the
    expensive-and-broken document that has nothing to do with the violation
    under test.  If the refusal names it, the check ran too late.
    """

    @pytest.mark.parametrize("model, expected", [
        pytest.param({"cw_tone": {k: v for k, v in TONE.items()
                                  if k != "line_width"}},
                     A13_PRESENCE, id="A13-presence"),
        pytest.param({"cw_tone": {**TONE,
                                  "line_width": {"value": -1.0, "unit": "MHz"}}},
                     A13_WIDTH, id="A13-line_width"),
        pytest.param({"cw_tone": {**TONE, "line_width": LINSPACE}},
                     A40, id="A40"),
        pytest.param({"bandpass": dict(PY_GAIN, at=["gain", "noise"])},
                     A47, id="A47"),
    ])
    def test_the_violation_outranks_a_beam_that_cannot_be_read(
            self, model, expected):
        doc = preflight_document(model=model, resources=UNREADABLE_BEAM)
        with pytest.raises(ConfigError) as caught:
            _load(doc)
        assert "no_such_beam" not in str(caught.value)
        assert str(caught.value).startswith(expected)

    def test_the_beam_really_is_unreadable_on_a_document_with_no_violation(self):
        """Anti-vacuity, and it is not optional here: without it every case
        above passes on a beam patch that silently stopped being broken, and
        the class would assert that a refusal beats nothing at all."""
        doc = preflight_document(resources=UNREADABLE_BEAM)
        assert preflight(doc).checks() & MINE == frozenset()
        with pytest.raises(ConfigError, match="no_such_beam"):
            _load(doc)


class TestA13IsPinnedWholeAndSaysTheRightSection:
    """S1 for A13's four text legs, and §0.3 E.10's route ruling."""

    def test_a_tone_missing_line_width_earns_A13_whole(self):
        doc = tone_document()
        del doc["model"]["cw_tone"]["line_width"]
        assert only(doc, "A13").message == A13_PRESENCE

    def test_a_negative_line_width_earns_A13_whole(self):
        found = only(tone_document(line_width={"value": -1.0, "unit": "MHz"}), "A13")
        assert found.message == A13_WIDTH
        assert found.where == "model.cw_tone.line_width"

    def test_a_protect_floor_above_one_earns_A13_whole(self):
        assert only(tone_document(protect_floor=2.0), "A13").message == A13_FLOOR

    def test_an_unknown_lineshape_earns_A13_whole(self):
        assert only(tone_document(lineshape="nope"), "A13").message == A13_LINESHAPE

    @pytest.mark.parametrize("literal", A13_BOUND)
    def test_each_invented_sentence_is_bound_in_exactly_one_module(self, literal):
        """The hoists' walker, turned on this module's own inventions.

        They are inventions, so "exactly one" is
        ``config/preflight/instrument.py`` rather than a section -- and the
        assertion is what catches the next reader copying one of these four
        sentences into ``sections/model.py`` or ``calibration.py`` to "keep
        them in step", which is the two-validators-for-one-property shape the
        rule exists to stop, arriving from the other direction.
        """
        assert_bound_once(literal)
        assert modules_carrying(literal) == ("config/preflight/instrument.py",)

    def test_a_zero_line_width_is_refused_as_well_as_a_negative_one(self):
        """``<= 0`` and not ``< 0``.  Zero is the case the message is written
        about -- it divides by zero rather than mirroring -- and a strict
        ``<`` passes every other cell in this class."""
        assert "A13" in ids(tone_document(line_width=0.0))

    def test_a_protect_floor_of_exactly_one_is_legal(self):
        """``0 < x <= 1``, and 1.0 is the SHIPPED default's legal extreme --
        ``tests/radio/test_cw_lineshape.py:334`` constructs one.  A check
        written ``0 < x < 1`` refuses a document the package builds, which is
        the one direction a pre-flight pass must never be wrong in."""
        assert "A13" not in ids(tone_document(protect_floor=1.0))

    def test_a_protect_floor_of_exactly_zero_is_refused(self):
        assert "A13" in ids(tone_document(protect_floor=0.0))

    @pytest.mark.parametrize("field", ["amplitude", "tone_freq", "line_width"])
    def test_every_required_field_is_a_leg_and_not_only_line_width(self, field):
        """S3's named twin: ``tone_freq`` and ``amplitude`` beside
        ``line_width``.  The set is read off ``field_specs(cls).required``
        rather than written here, so a field that gains a default stops being
        a leg without an edit -- and this parametrization is what says the
        three that have none today are all reached."""
        assert field_specs(CWCalibrationOperator)[field].required
        doc = tone_document()
        del doc["model"]["cw_tone"][field]
        assert only(doc, "A13").message.startswith(
            f"model.cw_tone: a CW calibration tone declares ['{field}'] nowhere,")

    def test_the_replace_route_is_walked_and_names_ITS_OWN_section(self):
        """§0.3 E.10's ruling and its stop-and-ask, in one assertion.

        The shipped sentence hardcodes ``f"model.{node_id}: "``, so on this
        route it names a section the entry is not written in -- measured at
        ``ea4839b``, a ``replace.cw_tone`` missing ``line_width`` is refused
        with *"model.cw_tone: CWCalibrationOperator requires
        ['line_width']."*  A verbatim hoist would have reproduced that one
        phase earlier.  This is the assertion that a rewrite back to the
        shipped wording cannot pass.
        """
        doc = replace_document({"cw_tone": {k: v for k, v in TONE.items()
                                            if k != "line_width"}})
        found = only(doc, "A13")
        assert found.where == "inference.twin.replace.cw_tone"
        assert found.message.startswith(
            "inference.twin.replace.cw_tone: a CW calibration tone declares "
            "['line_width'] nowhere,")
        assert "model.cw_tone" not in found.message

    def test_the_replace_route_is_walked_for_the_value_bounds_too(self):
        doc = replace_document({"cw_tone": {**TONE,
                                            "line_width": {"value": -1.0,
                                                           "unit": "MHz"}}})
        assert only(doc, "A13").where == "inference.twin.replace.cw_tone.line_width"

    def test_the_python_spelling_of_the_tone_is_reached_as_well_as_type(self):
        """S3: the CLASS, not the token.  ``type: CWCalibrationOperator`` and
        the ``python:`` target are two spellings of one class object, and a
        check keyed on the literal ``type:`` misses the one 3A's own tests
        exercise."""
        entry = {"python": "rheplicant.radio:CWCalibrationOperator",
                 **{k: v for k, v in TONE.items() if k != "line_width"}}
        assert "A13" in ids(preflight_document(model={"cw_tone": entry}))

    @pytest.mark.parametrize("entry, expected_where", [
        pytest.param({**PY_TONE, **{k: v for k, v in TONE.items()
                                    if k != "line_width"}},
                     "model.bandpass", id="presence"),
        pytest.param({**PY_TONE, **TONE,
                      "line_width": {"value": -1.0, "unit": "MHz"}},
                     "model.bandpass.line_width", id="line_width"),
        pytest.param({**PY_TONE, **TONE, "lineshape": "nope"},
                     "model.bandpass.lineshape", id="lineshape"),
    ])
    def test_a_tone_placed_at_a_FOREIGN_key_through_python_is_still_read(
            self, entry, expected_where):
        """The check is keyed on the CLASS ALONE, with no ``cw_tone`` gate.

        The ``python:`` hatch places a ``CWCalibrationOperator`` at any node
        and such a document BUILDS — measured, ``model.bandpass: {python:
        '…:CWCalibrationOperator', **TONE}`` loads clean.  Break one of the
        tone's rules there and, before this, the refusal was still a
        ``StateValidationError`` from behind the beam: neither ``A13.grid``
        (which walks the ``cw_tone`` key) nor an earlier draft of this check
        reached it.  A40 never had a node-id gate and always covered that
        document; this is A13 agreeing with it inside one file.
        """
        assert only(preflight_document(model={"bandpass": entry}),
                    "A13").where == expected_where

    def test_a_tone_at_a_foreign_key_that_is_WELL_FORMED_stays_clean(self):
        """The widening's own anti-vacuity: dropping the key gate must not
        start refusing the relocated tone that builds today."""
        doc = preflight_document(model={"bandpass": {**PY_TONE, **TONE}})
        assert ids(doc) & MINE == frozenset()
        assert _load(doc) is not None

    def test_a_NON_tone_class_under_the_cw_tone_key_is_not_given_the_tones_legs(
            self):
        """The other polarity of the same gate, and the one the class check
        buys.  A ``GainOperator`` written under ``cw_tone:`` declares none of
        the tone's required fields; keying on the KEY would hand it *"declares
        ['amplitude', 'line_width', 'tone_freq'] nowhere"* about fields it does
        not have.  The real fault is A5's, and A5 is what fires.

        The existing ``test_the_python_spelling…`` cannot see this: it writes
        ``python:`` UNDER the key ``cw_tone``, so the key alone answers it and
        a check that ignored the resolved class passes.

        **The entry declares no fields at all, and that is the whole point.**
        With ``gain: 1.0`` written beside it, a key-only check stands down on
        ``_unknown_field`` (``gain`` is not one of the tone's fields) and the
        test passes for the wrong reason — measured, that spelling leaves the
        "ignore the class" mutant alive.  A bare ``python:`` entry has nothing
        unknown relative to the tone's spec, so the only thing that can stop
        A13 firing is the resolved class.
        """
        doc = preflight_document(model={"cw_tone": dict(PY_GAIN)})
        assert "A13" not in ids(doc)
        assert "A5" in ids(doc)

    def test_a_tone_under_a_compose_block_is_reached_at_its_stage(self):
        """S3, and §0.3 E.10's "one route of four".  ``_t4_entries`` expands a
        ``compose:`` into stages; the composing mapping itself carries no
        ``line_width`` at all, so a walk that read it would find the tone's
        fields missing on every composed document -- or, reading the node
        spec, find nothing to check."""
        entry = {"compose": "pipeline",
                 "stages": [{k: v for k, v in TONE.items() if k != "amplitude"}]}
        found = only(preflight_document(model={"cw_tone": entry}), "A13")
        assert found.where == "model.cw_tone.stages[0]"

    def test_an_UNSELECTED_variant_that_breaks_the_rule_is_reported(self):
        """The half the layer walk buys, and the only half it buys.

        No variant is requested, so ``_assemble`` merges none of them and the
        base mapping is what the pass is handed.  A check reading
        ``document["model"]`` finds nothing here; the finding exists because
        ``_task3_over_layers`` merged the layer itself, and it names the layer
        in its own sentence because ``raise_if_refused`` quotes the MESSAGE.
        """
        doc = variant_document("model", {"cw_tone": {k: v for k, v in TONE.items()
                                                     if k != "line_width"}})
        found = only(doc, "A13")
        assert found.message.startswith("variants.twin_route: model.cw_tone: ")

    def test_a_SELECTED_variant_is_already_merged_before_the_pass_runs(self):
        """The other half, which the layer walk does NOT buy, driven so the
        two are not confused for one another.

        ``document.py:77`` applies the requested variant and only then calls
        ``preflight(doc)``, so this document is refused by the BASE layer's
        own walk -- the sentence carries no ``variants.`` prefix.  Measured;
        an earlier reading of plan §0.3 F.5(1) had the layer walk buying both.
        """
        doc = variant_document("model", {"cw_tone": {k: v for k, v in TONE.items()
                                                     if k != "line_width"}})
        with pytest.raises(ConfigError) as caught:
            _load(doc, variant="twin_route")
        assert str(caught.value).startswith(A13_PRESENCE)

    def test_the_base_documents_finding_is_not_repeated_per_variant(self):
        """``_task3_over_layers``' de-duplication, driven rather than trusted:
        a document with one fault and one variant that does not touch it must
        hand the reader ONE sentence, and ``only`` is what asserts that."""
        doc = variant_document("runtime", {"seed": 7})
        doc["model"] = {**doc["model"], "cw_tone": {k: v for k, v in TONE.items()
                                                    if k != "line_width"}}
        assert only(doc, "A13").message == A13_PRESENCE


class TestA13ReadsTheNumberTheBuildWouldSee:
    """``_text_number``: the three text spellings, the unit, and the declines."""

    @pytest.mark.parametrize("node, expected", [
        (-1.0, -1.0),
        (-1, -1.0),
        ("-1 MHz", -1e6),
        ({"value": -1.0, "unit": "MHz"}, -1e6),
        ({"value": -1.0}, -1.0),
    ])
    def test_the_unit_is_applied_and_not_dropped(self, node, expected):
        """Kills ``return float(raw)`` written without the conversion.  It is
        not only a magnitude: ``convert_to_canonical`` handles AFFINE units,
        so an offset can move a value across zero and a check that ignored
        the token would answer about a number the build never sees."""
        assert _text_number(node) == pytest.approx(expected)

    def test_an_affine_unit_crosses_zero_the_way_the_build_does(self):
        """``celsius`` converts with an OFFSET of 273.15 (``units.py``'s own
        table), so a negative number in the document is a positive one at the
        field.  A reader that dropped the token would refuse this and a reader
        that only scaled would too."""
        assert _text_number("-1 celsius") == pytest.approx(272.15)

    @pytest.mark.parametrize("node", [
        {"ref": "resources.arrays.w"},
        {"from": "channel_spacing"},
        {"linspace": {"start": 1.0, "stop": 2.0, "num": 4, "endpoint": True}},
        {"value": 1.0, "linspace": {}},
        {"value": 1.0, "as": "traced"},
        {"value": 1.0, "unit": 7},
        "not a shorthand",
        True,
        False,
        {"value": True},
        None,
        [1.0],
    ])
    def test_everything_it_cannot_read_from_text_is_a_stand_down(self, node):
        """§3.2(c): refusing on "I could not tell" refuses documents that
        build.  ``True`` is here twice on purpose -- ``isinstance(True, int)``
        is True, so a bare ``true`` would otherwise arrive as 1.0 and read as
        a legal width, and ``delivery.py::_as_static_float`` has a clause of
        its own naming exactly that."""
        assert _text_number(node) is None

    def test_a_unit_token_the_grammar_rejects_is_left_to_the_unit_grammar(self):
        """`_text_number`'s `except ConfigError: return None`, driven.

        ``canonical_unit`` refuses an unknown token by name, and that sentence
        lists the units this layer converts — it is the one the reader needs.
        Reading the number UNCONVERTED instead would earn A13 *"-1 Hz is not
        above zero"*, a complaint about a quantity the build never computes,
        and it would win because P-1 runs first.
        """
        doc = tone_document(line_width={"value": -1.0, "unit": "banana"})
        assert ids(doc) & MINE == frozenset()
        with pytest.raises(ConfigError, match="Unknown unit 'banana'"):
            _load(doc)

    def test_a_lineshape_this_pass_cannot_read_is_not_quoted_back_at_the_user(
            self):
        """The `isinstance(shape, str)` guard.  Without it a
        ``lineshape: {ref: …}`` earns A13 *"{'ref': …} is not a lineshape"*,
        quoting a mapping at a reader who wrote a reference — and pre-empting
        the delivery layer's own static-str complaint."""
        assert "A13" not in ids(tone_document(lineshape={"ref": "resources.arrays.w"}))

    def test_a_ref_width_is_left_to_the_delivery_layers_own_sentence(self):
        """The stand-down, end to end.  Measured, ``line_width: {ref: ...}``
        earns *"Field 'line_width' is a static float and the value is
        ArrayImpl"* -- so the document this pass declines to read is a
        document the layer refuses anyway, in its own words."""
        doc = preflight_document(
            model={"cw_tone": {**TONE, "line_width": {"ref": "resources.arrays.w"}}},
            resources={"arrays": {"w": {"list": [1.0e6, 2.0e6]}}})
        assert "A13" not in ids(doc)
        assert "A40" not in ids(doc)


class TestA40IsHoistedAndBoundOnce:
    """S1 and §3.2(h) for A40, plus the walk over every static field."""

    def test_an_array_form_on_a_static_field_earns_A40_whole(self):
        found = only(tone_document(line_width=LINSPACE), "A40")
        assert found.message == A40
        assert found.where == "model.cw_tone.line_width"

    def test_the_sentence_is_bound_in_exactly_one_module(self):
        """§2.2's one-binding rule as a command.  A hoist that COPIED the
        words would leave two modules saying one sentence, which is the
        ``_number``-vs-``_whole`` divergence on the 2C ledger."""
        assert_bound_once(A40_BOUND)
        assert modules_carrying(A40_BOUND) == ("config/delivery.py",)

    def test_the_section_still_says_it_too_at_build_time(self):
        """§2.2's third clause: the section KEEPS calling it, as a second
        opinion.  Driven through ``deliver`` directly, so this stays true of
        a document the pre-flight walk declines to reach."""
        from rheplicant.config.delivery import deliver

        spec = field_specs(CWCalibrationOperator)["line_width"]
        with pytest.raises(ConfigError, match="produces an array"):
            deliver([1.0, 2.0], spec, dtype="float32", source="linspace")

    @pytest.mark.parametrize("node_id, cls_name, field, entry", [
        ("adc", "ADCOperator", "n_bits", {"n_bits": LINSPACE, "scale": 1.0}),
        ("cw_tone", "CWCalibrationOperator", "lineshape",
         {**TONE, "lineshape": LINSPACE}),
        ("flagging", "MomentRFIFlaggingOperator", "kernel_shapes",
         {"type": "MomentRFIFlaggingOperator", "kernel_shapes": {"list": [[1, 3]]}}),
    ])
    def test_every_static_field_of_every_node_and_not_only_the_tones(
            self, node_id, cls_name, field, entry):
        """S3's named twin for A40.  Measured over all 28 shipped operator
        classes with ``field_specs`` + ``mode_of``: 28 fields are static (15
        float, 6 int, 5 str, 1 tuple, 1 mapping) and **zero** are
        ``static_bool``.  The three here are an int, a str and a tuple, so a
        check written for ``static_float`` alone dies on two of them."""
        cls = next(one for one in operator_table()[node_id]
                   if one.__name__ == cls_name)
        assert mode_of(field_specs(cls)[field]) != "traced"
        found = only(preflight_document(model={node_id: entry}), "A40")
        assert found.where == f"model.{node_id}.{field}"

    def test_a_traced_field_is_exactly_what_an_array_form_is_FOR(self):
        """The other polarity, and the one an over-eager A40 breaks.
        ``ForegroundOperator.amplitude`` is traced, so ``{linspace: ...}`` on
        it is the grammar working."""
        entry = {"foregrounds": [{"type": "ForegroundOperator",
                                  "amplitude": LINSPACE,
                                  "spectral_index": -2.5,
                                  "ref_freq": {"value": 70.0, "unit": "MHz"}}]}
        assert "A40" not in ids(preflight_document(model=entry))

    def test_a_many_nodes_third_entry_is_reached_at_its_own_index(self):
        """S3, and §0.3 E.10's fourth route.  A ``many`` node is a LIST, so
        the reader is sent to the line to edit rather than to the node."""
        chain = [{"type": "FourierBandFilter", "mode": "extract",
                  "low": 0.0, "high": 0.25},
                 {"type": "FourierBandFilter", "mode": LINSPACE,
                  "low": 0.0, "high": 0.25}]
        found = only(preflight_document(model={"filters": chain}), "A40")
        assert found.where == "model.filters[1].mode"

    def test_the_replace_route_is_walked(self):
        doc = replace_document({"cw_tone": {**TONE, "line_width": LINSPACE}})
        assert only(doc, "A40").where == "inference.twin.replace.cw_tone.line_width"

    def test_a_node_holding_two_form_keys_is_left_to_resolve_values_own_count(
            self):
        """`_array_form`'s ``len(forms) != 1``, driven.

        ``resolve_value`` refuses zero and several form keys by name, and
        *"Value node holds 2 form keys (['linspace', 'value']); exactly one is
        allowed"* names the fault.  Answering A40 here instead reports a
        consequence of the second key — and which consequence depends on
        mapping order, so the message would not even be stable.
        """
        node = {**LINSPACE, "value": 1.0}
        doc = tone_document(line_width=node)
        assert ids(doc) & MINE == frozenset()
        with pytest.raises(ConfigError, match="holds 2 form keys"):
            _load(doc)

    def test_a_compose_block_under_twin_replace_is_not_expanded_into_stages(
            self):
        """`_routes`' third member, driven.

        ``compose:`` is honoured only by ``compose._single``, which the replace
        route bypasses — so there ``compose`` and ``stages`` are unknown
        constructor fields and ``_construct`` refuses them by name.  A walk
        that expanded them would read a stage spec that never reaches a
        constructor, and could answer A40 about it on a document refused for
        the composing key itself.
        """
        doc = replace_document({"cw_tone": {"compose": "pipeline",
                                            "stages": [{**TONE,
                                                        "line_width": LINSPACE}]}})
        assert ids(doc) & MINE == frozenset()
        with pytest.raises(ConfigError,
                           match=r"does not take \['compose', 'stages'\]"):
            _load(doc)

    def test_a_variant_that_breaks_it_is_reported_with_its_layer(self):
        doc = variant_document("model", {"cw_tone": {**TONE,
                                                     "line_width": LINSPACE}})
        assert only(doc, "A40").message.startswith(
            "variants.twin_route: model.cw_tone.line_width: ")

    @pytest.mark.parametrize("node", [
        {"value": 5.0e6},
        {"ref": "resources.arrays.w"},
        {"from": "channel_spacing"},
    ])
    def test_the_five_forms_outside_ARRAY_FORMS_are_not_A40(self, node):
        """Measured, ``set(VALUE_FORMS) - ARRAY_FORMS == {'from',
        'from_switch_order', 'python', 'ref', 'value'}``.  ``{value:}``
        resolves with ``source == "scalar"`` and misses A40 in the BUILD too,
        so agreeing with the build is correct rather than a gap; ``{ref:}``
        and ``{from:}`` deliver a jax ``ArrayImpl`` and earn
        ``_as_static_float``'s type complaint, which is the declared false
        negative recorded in the plan's §7."""
        doc = preflight_document(
            model={"cw_tone": {**TONE, "line_width": node}},
            resources={"arrays": {"w": {"list": [1.0e6, 2.0e6]}}})
        assert "A40" not in ids(doc)


class TestA47IsDecidedThroughT5ClaimsAndNotOffARawAt:
    """§2.3's first named stand-down, and the four shapes it protects."""

    def test_a_misaddressed_region_earns_A47_whole(self):
        doc = preflight_document(model={"bandpass": dict(PY_GAIN,
                                                         at=["gain", "noise"])})
        found = only(doc, "A47")
        assert found.message == A47
        assert found.where == "model.bandpass"

    def test_the_sentence_is_bound_in_exactly_one_module(self):
        assert_bound_once(A47_BOUND)
        assert modules_carrying(A47_BOUND) == ("config/paths.py",)

    def test_the_section_still_says_it_too_at_build_time(self):
        from rheplicant.config.paths import refuse_misaddressed_region

        with pytest.raises(ConfigError, match="is addressed in the assembly"):
            refuse_misaddressed_region("bandpass", ["gain", "noise"])

    @pytest.mark.parametrize("spec, why", [
        ({"at": ["gain", "noise"]},
         "at: with no python: -- compose.py:296 says it better"),
        ({**PY_GAIN, "at": ["gain", "noise"], "snapshot_before": "tap"},
         "at: beside snapshot_before: -- compose.py:289 refuses the PAIR"),
        ({**PY_GAIN, "at": "gain"},
         "the STRING spelling must restate its own key -- compose.py:303"),
    ])
    def test_the_three_shapes_single_refuses_better_are_not_pre_empted(
            self, spec, why):
        """The whole reason A47 goes through ``_t5_claims``.  A P-1 check
        reading ``model.<n>.at`` raw answers about a region on all three, and
        every one of them has a refusal that names the real fault.  This is
        the test the naive implementation dies on."""
        assert "A47" not in ids(preflight_document(model={"bandpass": spec})), why

    def test_a_one_element_at_list_is_legal_and_stays_so(self):
        """``refuse_misaddressed_region`` returns early below two nodes
        (``paths.py:318-319``), so ``{python: ..., at: ['gain']}`` BUILDS --
        measured in ``test_preflight_model.py``'s own A5 cases.  A check
        written ``config_key != at[-1]`` without the length guard refuses it."""
        doc = preflight_document(model={"bandpass": dict(PY_GAIN, at=["gain"])})
        assert "A47" not in ids(doc)

    def test_a_region_written_under_its_LAST_node_is_legal(self):
        doc = preflight_document(model={"noise": dict(PY_GAIN,
                                                      at=["gain", "noise"])})
        assert "A47" not in ids(doc)

    def test_an_at_on_a_many_node_is_a_switch_label_and_not_a_relocation(self):
        """``_t5_placement`` answers a ``many`` node before the shape is asked
        about, so a FAN label a user happened to spell ``at`` is not a
        region."""
        doc = preflight_document(model={"cal_loads": {"at": {"t_load": 300.0}}})
        assert "A47" not in ids(doc)

    def test_an_at_region_under_twin_replace_is_not_A47_at_all(self):
        """§0.3 E.10 for A47: a NON-route, not a false negative.

        ``sections/twin.py:67`` calls ``build_node_operator`` directly,
        bypassing ``compose._single`` -- the only place ``at:`` is honoured --
        so on this route ``at:`` is an unknown constructor field.  Measured:
        the document below is refused with ``_construct``'s *"does not take
        ['at']"*, which is the right sentence, and an A47 here would name a
        relocation the build never makes.
        """
        doc = replace_document({"bandpass": dict(PY_GAIN, at=["gain", "noise"])})
        assert "A47" not in ids(doc)

    def test_a_variant_that_breaks_it_is_reported_with_its_layer(self):
        doc = variant_document("model", {"bandpass": dict(PY_GAIN,
                                                          at=["gain", "noise"])})
        assert only(doc, "A47").message.startswith(
            "variants.twin_route: model.bandpass: ")


class TestTheStandDowns:
    """S4: a document wrong in this check's way AND wrong in a better way."""

    def test_an_unknown_key_beside_a_missing_required_one_is_not_A13(self):
        """``_construct`` sweeps unknown keys BEFORE it looks for missing ones
        (``sections/model.py:151-162``), so the reader sees *"does not take
        ['nope']"* today.  A13 arriving one phase earlier with *"declares
        ['line_width'] nowhere"* would answer a question the user has not got
        to yet.  Kills ``_unknown_field`` deleted."""
        doc = preflight_document(model={"cw_tone": {"amplitude": 5000.0,
                                                    "nope": 1}})
        assert "A13" not in ids(doc)

    def test_an_at_on_the_replace_route_is_left_to_construct_which_names_the_key(
            self):
        """The route flag, and the advice loop it exists to stop.

        ``sections/twin.py:67`` calls ``build_node_operator`` directly, so on
        the replace route ``at:`` is an unknown constructor field and
        ``_construct`` answers *"does not take ['at']"* — the key the user
        actually wrote.  With the ``{at, snapshot_before}`` exemption applied
        unconditionally this module answered A13 instead, and then **applying
        A13's own remedy left the document refused anyway** (R4): the second
        case below is that document, and it must earn nothing from this pass.
        """
        missing = {k: v for k, v in TONE.items() if k != "line_width"}
        pre_empted = replace_document({"cw_tone": {**missing,
                                                   "at": ["gain", "noise"]}})
        assert ids(pre_empted) & MINE == frozenset()
        with pytest.raises(ConfigError, match=r"does not take \['at'\]"):
            _load(pre_empted)
        # A13's advice applied -- declare the field -- and the document is
        # STILL refused, by the sentence that should have come first.
        advised = replace_document({"cw_tone": {**TONE, "at": ["gain", "noise"]}})
        assert ids(advised) & MINE == frozenset()
        with pytest.raises(ConfigError, match=r"does not take \['at'\]"):
            _load(advised)

    def test_an_at_on_the_replace_route_stands_A40_down_too(self):
        """A40 had the identical hole and the identical fix — the flag is
        threaded once and read by both rules."""
        doc = replace_document({"cw_tone": {**TONE, "line_width": LINSPACE,
                                            "at": ["gain", "noise"]}})
        assert ids(doc) & MINE == frozenset()

    def test_an_at_on_the_MODEL_route_is_popped_and_does_not_stand_A13_down(self):
        """The other polarity, and the one a blanket "never exempt" breaks.

        ``compose._single`` pops ``at:`` and ``snapshot_before:`` BEFORE
        ``_construct`` sees the spec (``compose.py:287-288``), so on this route
        they are not unknown keys and the entry must still be read.  Measured:
        this document is refused by ``_construct``'s *"requires ['line_width']"*
        at `ea4839b`, which is exactly the row A13.text moves.
        """
        missing = {k: v for k, v in TONE.items() if k != "line_width"}
        doc = preflight_document(model={"bandpass": {**PY_TONE, **missing,
                                                     "at": ["gain", "noise"]}})
        assert only(doc, "A13").where == "model.bandpass"

    def test_an_unknown_key_beside_an_array_form_is_not_A40(self):
        doc = preflight_document(model={"cw_tone": {**TONE,
                                                    "line_width": LINSPACE,
                                                    "nope": 1}})
        assert "A40" not in ids(doc)

    @pytest.mark.parametrize("declared", ["traced", "banana"])
    def test_an_as_the_field_contradicts_is_not_pre_empted_by_A40(self, declared):
        """``deliver`` reads ``as:`` FIRST (``delivery.py:187-204``), so a
        document making a claim about its own delivery mode is answered about
        that claim: *"This value declares as='traced', but field 'line_width'
        is 'static_float'"*, and *"as='banana' is not a delivery mode"*.  A40
        would name a consequence of the key rather than the key."""
        node = {**LINSPACE, "as": declared}
        assert "A40" not in ids(tone_document(line_width=node))

    def test_an_as_that_AGREES_with_the_field_still_earns_A40(self):
        """The anti-vacuity partner: the stand-down is about a CONTRADICTED
        ``as:``, not about the key being present.  Without this a
        ``_a40_stands_down`` that returned True for any ``as:`` passes."""
        node = {**LINSPACE, "as": "static_float"}
        assert only(tone_document(line_width=node), "A40").where == \
            "model.cw_tone.line_width"

    def test_a_negative_tone_freq_is_left_to_the_grid_leg_that_names_the_band(self):
        """The stand-down §0.3 E.8 does not name and this task measured.

        ``calibration.py:373`` guards ``tone_freq > 0``, but no document
        reaches it: ``A13.grid`` answers first with *"the tone centre spans
        [...] outside this run's observed band [6e+07, 8.5e+07] Hz"*, which
        tells the reader the interval to write in.  A P-1 sign leg would run
        earlier still and replace that with "it is negative".
        """
        doc = tone_document(tone_freq={"value": -70.0, "unit": "MHz"})
        assert "A13" not in ids(doc)
        with pytest.raises(ConfigError, match="outside this run's observed band"):
            _load(doc)


class TestApplyingTheAdviceMakesTheDocumentPass:
    """S4's second half: take each message's own remedy and run it (R4)."""

    def test_A13s_presence_advice_declaring_the_field(self):
        doc = tone_document()
        del doc["model"]["cw_tone"]["line_width"]
        assert "A13" in ids(doc)
        doc["model"]["cw_tone"]["line_width"] = {"value": 5.0, "unit": "MHz"}
        # `ids(doc) == frozenset()` here, and NOT the intersection below, is
        # the banned form R8 / §0.3 E.11 exist for, and it was banned on
        # measurement rather than principle: `tone_document()` is
        # `preflight_document()`, the SHARED base document, and a sibling
        # branch registering a WARN-only check on it -- the shape of Task 5's
        # A46 leg 3 and Task 7's B9, where the document still LOADS and only
        # the report gains an id -- takes this one line red while the other 67
        # tests here stay green.  Demonstrated with a one-check plugin at
        # review; green in this branch, red after merge, in a file the sibling
        # that broke it never opened.
        assert ids(doc) & MINE == frozenset() and _load(doc) is not None

    def test_A13s_width_advice_a_positive_width(self):
        assert "A13" in ids(tone_document(line_width=-1.0))
        assert _load(tone_document(line_width={"value": 5.0, "unit": "MHz"}))

    def test_A13s_floor_advice_a_fraction_of_the_peak(self):
        assert "A13" in ids(tone_document(protect_floor=2.0))
        assert _load(tone_document(protect_floor=0.5))

    def test_A13s_lineshape_advice_one_of_the_two_it_names(self):
        assert "A13" in ids(tone_document(lineshape="nope"))
        assert _load(tone_document(lineshape="gaussian",
                                   line_width={"value": 5.0, "unit": "MHz"}))

    def test_A40s_advice_write_a_single_number_here(self):
        assert "A40" in ids(tone_document(line_width=LINSPACE))
        assert _load(tone_document(line_width={"value": 5.0, "unit": "MHz"}))

    def test_A47s_advice_name_the_entry_after_its_last_covered_node(self):
        """The message says *"Name the entry 'noise'"*; this writes that
        document and asserts it is accepted.  Three of 3A's checks shipped a
        remedy another check refuses, which is what R4 exists to catch."""
        bad = preflight_document(model={"bandpass": dict(PY_GAIN,
                                                         at=["gain", "noise"])})
        assert "A47" in ids(bad)
        good = preflight_document(model={"noise": dict(PY_GAIN,
                                                       at=["gain", "noise"])})
        assert "A47" not in ids(good)


class TestTheEntryWalkResolvesTheClassRatherThanTheToken:
    """``_entry_class``: the three routes, and the two declines."""

    @pytest.mark.parametrize("entry, expected", [
        ({"type": "CWCalibrationOperator"}, "CWCalibrationOperator"),
        ({"python": "rheplicant.radio:CWCalibrationOperator"},
         "CWCalibrationOperator"),
        ({}, "CWCalibrationOperator"),
    ])
    def test_the_three_routes_reach_one_class(self, entry, expected):
        table = operator_table()
        assert _entry_class("cw_tone", entry, table).__name__ == expected

    @pytest.mark.parametrize("node_id, entry", [
        ("noise", {}),
        ("cw_tone", {"type": "NoSuchOperator"}),
        ("cw_tone", {"python": "some.foreign.module:Thing"}),
        ("beam_spill", {"from": "projector"}),
        ("cw_tone", "not a mapping"),
    ])
    def test_the_declines(self, node_id, entry):
        """Each decline hands the question to a check that answers it better:
        A7 for a node registering several classes and naming none, A7 again
        for a ``type:`` that is registered nowhere, §2.4's boundary for a
        foreign ``python:`` target this pass will not import, and
        ``_from_route`` for a ``from:`` entry that never reaches
        ``_construct`` at all."""
        assert _entry_class(node_id, entry, operator_table()) is None

    def test_a_node_registering_several_classes_is_left_to_A7(self):
        doc = preflight_document(model={"noise": {"line_width": LINSPACE}})
        assert "A40" not in ids(doc)
        assert "A7" in ids(doc)


def _load(document, **kwargs):
    """``load_document`` on ``document``.

    A thin wrapper so the advice tests read as one line each, and so the
    import stays function-local: ``load_document`` is on ``preflight/``'s own
    banned-name list and a module-scope import here would be a test asserting
    the opposite of what the scope guard asserts about the code.
    """
    from rheplicant.config.document import load_document

    return load_document(document, **kwargs)


class TestTheRulesAreThreeAndTheyStayThree:
    """The seam the one-registration decision rests on."""

    def test_each_rule_is_one_check_and_the_walker_chains_them(self):
        """The three names plan §3.1 pins each still decide one check.
        Driven, not asserted from the source: each rule is called with a bare
        LAYER and must answer about its own id and no other."""
        layer = preflight_document(model={
            "cw_tone": {**TONE, "protect_floor": 2.0, "lineshape": LINSPACE},
            "bandpass": dict(PY_GAIN, at=["gain", "noise"])})
        assert {one.check for one in _tone_text(layer)} == {"A13"}
        assert {one.check for one in _static_fields(layer)} == {"A40"}
        assert {one.check for one in _region_key(layer)} == {"A47"}

    def test_the_walker_emits_every_rules_findings_on_one_document(self):
        """Anti-vacuity for the chain: a document breaking all three must earn
        all three, or the ``for rule in _RULES`` loop could drop one in
        silence and every single-fault test above would still pass."""
        doc = preflight_document(model={
            "cw_tone": {**TONE, "protect_floor": 2.0, "lineshape": LINSPACE},
            "bandpass": dict(PY_GAIN, at=["gain", "noise"])})
        mine = frozenset({"A13", "A40", "A47"})
        assert preflight(doc).checks() & mine == mine
