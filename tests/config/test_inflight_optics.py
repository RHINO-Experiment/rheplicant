"""A43 and B9 -- ``inflight/optics.py``.

The same two rules ``test_inflight_twin.py`` carries: **every message is
pinned by equality on its whole text**, and **every registry and findings
assertion is subset shaped**, scoped to :data:`MINE`.

**The anti-property, once per ROW rather than once per module.**  Both checks
run after ``build_resources``.  B9's second spherical harmonic transform has
already been PAID by the time B9 mentions it -- and B9 is the row a reader is
likeliest to believe saves something, because its message quotes millisecond
figures.  So ``test_the_beam_wins_against_A43`` and
``test_the_beam_wins_against_B9`` each say what a document that is both wrong
in that row's way and carrying an unreadable beam is refused by: the beam.  A
message here that claimed to save anything would be schema §6's false preamble
in this layer's voice.
"""

import dataclasses
import pathlib
import subprocess
import sys

import pytest

from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE, WARN
from rheplicant.config.inflight import BUILT_CHECKS, built
from rheplicant.config.inflight.optics import (
    _analysing,
    _beam_analysed_twice,
    _protection_cut,
    _tone_survives_flagging,
)
from tests.config.inflight_helpers import (
    best_ms,
    built_findings,
    built_only,
    built_run,
    projector_sections,
)
from tests.config.preflight_helpers import (
    BASE_MODEL,
    UNREADABLE_BEAM,
    preflight_document,
)

_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: The ids this module is about.  Every "and nothing else" assertion is
#: intersected with it (§0.3 E.11).
MINE = frozenset({"A43", "B9"})

#: Twelve values a document key may hold that are not what a check expects.
HOSTILE = ([], {}, set(), None, 3, [[1, 2], [3]], "text", (1, 2),
           {"a": [1]}, [{"b": 2}], True, 0.5)


def mine(document, **kwargs) -> frozenset[str]:
    """The ids from THIS module that fired on ``document``."""
    return frozenset(one.check
                     for one in built_findings(document, **kwargs)) & MINE


# --- the documents ---------------------------------------------------------

#: A 5000 K tone, 3.6 MHz wide, on the base document's 60-85 MHz band.  Its
#: peak channel weight is 0.888945, so the protection cut sits at
#: ``0.01 x 5000 x 0.888945 = 44.4472 K``.
TONE = {"amplitude": {"value": 5000.0, "unit": "K"},
        "tone_freq": {"value": 70.0, "unit": "MHz"},
        "line_width": {"value": 3.6, "unit": "MHz"}}

#: The same tone drifting in BOTH frequency and level.  This is the document
#: on which "max over the frequency axis" and "max over the flat array" give
#: different answers -- 177.159 against 196.844, 11 % apart -- which is what
#: makes the axis choice testable at all.  With a level that does not drift
#: the two are algebraically equal and no document can tell them apart.
DRIFTING = {**TONE, "tone_freq": {"value": 62.0, "unit": "MHz"},
            "drift_rate": 6.0e5, "amplitude_drift_rate": 0.1}

#: A tone whose LEVEL drifts and whose CENTRE does not -- an ordinary
#: document, and the only shape in which the second leg of
#: ``drift_rate != 0.0 or amplitude_drift_rate != 0.0`` is the one that
#: decides.  Its cut is **177.789** against the static path's 44.4472, so a
#: threshold of 100 separates the shipped code from a check that dropped the
#: leg and stood down while the calibrator got eaten.
AMPLITUDE_ONLY = {**TONE, "amplitude_drift_rate": 0.1}

#: The mirror of :data:`AMPLITUDE_ONLY`: the CENTRE drifts and the level does
#: not, which is the only shape in which the FIRST leg of the same ``or`` is
#: the one that decides.  Its cut is **49.2109** against the 28.0445 a check
#: that treated it as static would compute (the tone sitting at its t=0 centre
#: of 62 MHz for the whole run), so a threshold of 35 separates them.
CENTRE_ONLY = {**TONE, "tone_freq": {"value": 62.0, "unit": "MHz"},
               "drift_rate": 6.0e5}

#: The same tone with the drift removed -- what the one-legged check computes
#: on :data:`CENTRE_ONLY`.  Bound so the test can assert the mutant's verdict
#: rather than describe it.
CENTRE_ONLY_FROZEN = {**TONE, "tone_freq": {"value": 62.0, "unit": "MHz"}}

GENERAL_POINTING = {"engine": "general_pointing",
                    "beam": {"ref": "resources.beams.horn"},
                    "lmax": 8, "nside": 4,
                    "lat_deg": {"value": 53.2367, "unit": "deg"},
                    "normalize_beam": True,
                    "acknowledge_float32_sky": True}


def tone_and_flagger(threshold=3.0, tone=None, **model):
    """A ``cw_tone`` and a ``FlaggingOperator`` on the base document."""
    return preflight_document(
        model={**BASE_MODEL, "cw_tone": tone or TONE,
               "flagging": {"type": "FlaggingOperator",
                            "threshold": {
                                "value": threshold, "unit": "adc_count",
                            }},
               **model})


def two_projectors(tmp_path, first=None, second=None):
    """One beam and two projectors over it, written under ``tmp_path``."""
    sections = projector_sections(tmp_path)
    if first is None:
        sections["projectors"]["second"] = dict(sections["projectors"]["drift"],
                                                **(second or {}))
    else:
        sections["projectors"] = {"one": dict(first),
                                  "two": dict(first, **(second or {}))}
    return preflight_document(resources=sections)


# --- the messages, whole ---------------------------------------------------

_A43_TAIL = (
    " bandpass and gain sit on the same trunk between the two and are "
    "DIMENSIONLESS, so neither is multiplied in; adc.scale is the leaf that "
    "carries adc_count/K and it is. Note what adc.scale is: a TRACED field, "
    "so a latent bound into it by inference.bindings overwrites it at bind "
    "time and the twin leaf read here holds only the init. This comparison is "
    "therefore about the run as it was declared, and a fit that frees the ADC "
    "scale can move the threshold out from under the tone without this check "
    "seeing it. And this runs after build_resources: the beam is long since "
    "read, so what the slot buys is the comparison, not a saving (check A43)."
)


def a43_message(threshold, cut, floor, scale="1"):
    """A43's whole sentence, assembled from the four numbers it quotes."""
    return (
        f"model.flagging.threshold is {threshold} and the tone's own "
        f"protection cut sits at {cut} in the same units (protect_floor 0.01 "
        f"x amplitude 5000 K x the peak channel weight, x adc.scale {scale}). "
        "Every channel carrying tone between those two numbers is above the "
        "flagger's threshold and below the tone's own protected set, so the "
        "flagger takes it: those channels are the calibrator's shoulders, and "
        "removing them biases the recovered tone level low while nothing "
        "raises and every shape stays right. docs/contracts.md measures the "
        "unprotected version of this at twelve flagged samples -- 'That is "
        f"the calibrator, gone.' Raise model.flagging.threshold above {cut}, "
        "or lower model.cw_tone.protect_floor so the protected set reaches "
        f"down past it ({floor} or less does it at this amplitude)."
        + _A43_TAIL)


A43_MESSAGE = a43_message("3", "44.4472", "0.000674957")
A43_DRIFTING = a43_message("170", "177.159", "0.00959588")

_B9_TAIL = (
    " Measured: two projectors naming one beam with the same lmax and "
    "beam_iterations analyse it TWICE and their beam_alms come back bitwise "
    "equal and NOT the same array -- so equality cannot tell the two cases "
    "apart and object identity can, which is why this reads the built "
    "projectors rather than the text alone. The cost is one whole analysis "
    "per extra projector: 8.9 -> 16.8 ms warm at nside 4 / lmax 8, against "
    "1.4 s at nside 16 / lmax 31. The second consequence is the one worth "
    "more than the milliseconds: a gradient into the beam map now splits "
    "between two independent leaves rather than accumulating on one "
    "(check B9)."
)

#: The remedy, which on driftscan carries one extra clause: alms have no
#: pixel count, so the resolution the ``beam:`` route infers has to be
#: written. Until A8.6 this constant was ``_B9_NO_REMEDY`` and said there was
#: no edit at all -- true when written, and quotable for as long as nobody
#: re-measured it.
_B9_DRIFTSCAN_REMEDY = (
    "Write beam_alms: {ref: resources.projectors.drift.beam_alms} on the "
    "second entry: measured, that route analyses the beam once and hands both "
    "projectors the same array. On engine: driftscan write nside: too -- alms "
    "carry no pixel count, so the resolution the beam route infers from the "
    "map length has to be declared on this one."
)

B9_DRIFTSCAN = (
    "resources.projectors.second and resources.projectors.drift both analyse "
    "resources.beams.horn at lmax=8 with beam_iterations=3, and this run's "
    "two beam_alms are not the same array -- so the identical spherical "
    "harmonic transform ran 2 times. " + _B9_DRIFTSCAN_REMEDY + _B9_TAIL
)

B9_GENERAL_POINTING = (
    "resources.projectors.two and resources.projectors.one both analyse "
    "resources.beams.horn at lmax=8 with beam_iterations=3, and this run's "
    "two beam_alms are not the same array -- so the identical spherical "
    "harmonic transform ran 2 times. Write beam_alms: {ref: "
    "resources.projectors.one.beam_alms} on the second entry: measured, that "
    "route analyses the beam once and hands both projectors the same array."
    + _B9_TAIL
)


class TestTheRegistry:
    def test_each_id_binds_to_its_own_function(self):
        assert BUILT_CHECKS["A43"] is _tone_survives_flagging
        assert BUILT_CHECKS["B9"] is _beam_analysed_twice

    def test_both_slots_are_claimed(self):
        assert {"A43", "B9"} <= set(BUILT_CHECKS)

    def test_the_module_is_WIRED_and_not_merely_decorated(self):
        """A SUBPROCESS: in this process the import has already happened.

        ``BUILT_CHECKS["A43"] is _tone_survives_flagging`` passes because THIS
        module's own import ran the decorator.  Deleting the foot-import line
        in ``inflight/__init__.py`` leaves every test in this file green and
        the check absent for every user.
        """
        done = subprocess.run(
            [sys.executable, "-c",
             "from rheplicant.config.inflight import BUILT_CHECKS\n"
             "assert {'A43', 'B9'} <= set(BUILT_CHECKS), sorted(BUILT_CHECKS)\n"],
            capture_output=True, text=True, cwd=str(_ROOT), check=False)
        assert done.returncode == 0, (
            "A43/B9 are decorated but not wired: importing the package does "
            "not import inflight/optics.py.\n" + done.stdout + done.stderr)

    def test_the_module_name_collides_with_no_entry_point(self):
        """R13: ``optics`` is not a name ``inflight/__init__.py`` binds, so the
        ordinary aliased foot import works for it -- unlike ``axes``."""
        import rheplicant.config.inflight as package

        assert callable(package.axes) and callable(package.built)
        assert package.optics.__name__ == "rheplicant.config.inflight.optics"

    def test_the_base_document_earns_nothing_here(self):
        assert mine(preflight_document()) == frozenset()


class TestA43:
    """The tone's protection cut against the flagger's threshold."""

    def test_it_fires_and_names_the_flagging_node(self):
        found = built_only(tone_and_flagger(), "A43")
        assert found.severity == REFUSE
        assert found.where == "model.flagging"

    def test_the_whole_message(self):
        assert built_only(tone_and_flagger(), "A43").message == A43_MESSAGE

    def test_the_hook_stops_the_load(self):
        with pytest.raises(ConfigError) as raised:
            load_document(tone_and_flagger())
        assert str(raised.value) == A43_MESSAGE

    def test_a_threshold_above_the_cut_earns_nothing(self):
        """S4, first remedy: raise the threshold, and the document loads.

        ``docs/contracts.md``'s own worked pair is exactly this shape -- a
        5000 K tone under a 1000 K threshold -- and it prints ``flagged: 0``.
        """
        assert mine(tone_and_flagger(threshold=1000.0)) == frozenset()
        assert load_document(tone_and_flagger(threshold=1000.0)) is not None

    def test_a_lower_protect_floor_ALSO_earns_nothing(self):
        """S4, second remedy: the message names a number, and it works.

        The sentence says ``0.000674957 or less does it at this amplitude``;
        this applies it.  Both remedies are exercised because R4 is about
        advice that can be followed rather than advice that was written.
        """
        fixed = tone_and_flagger(tone={**TONE, "protect_floor": 0.0006})
        assert mine(fixed) == frozenset()
        assert load_document(fixed) is not None

    def test_a_tone_with_no_flagger_earns_nothing(self):
        assert mine(preflight_document(
            model={**BASE_MODEL, "cw_tone": TONE})) == frozenset()

    def test_a_flagger_with_no_tone_earns_nothing(self):
        assert mine(preflight_document(
            model={**BASE_MODEL, "flagging": {"type": "FlaggingOperator",
                                              "threshold": {
                                                  "value": 3.0,
                                                  "unit": "adc_count",
                                              }}})) \
            == frozenset()

    def test_NO_ADC_LIT_is_a_real_document_and_not_a_crash(self):
        """§0.3 E.6 ruling 4, as a test.

        ``twin['adc']`` raises ``KeyError: No node named 'adc' in this
        assembly`` on an ordinary ``cw_tone`` + ``flagging`` document --
        measured -- and an implementation that indexes it unguarded turns the
        whole pass into "in-flight check 'A43' RAISED KeyError", which loses
        every finding after it.  An unlit ``adc`` node contracts to identity,
        so the data reaching the flagger is in Kelvin and the scale is 1.0.
        """
        run = built_run(tone_and_flagger())
        assert "adc" not in run.twin.lit
        with pytest.raises(KeyError, match="No node named 'adc'"):
            run.twin["adc"]
        assert "x adc.scale 1)" in built_only(tone_and_flagger(),
                                              "A43").message

    def test_the_adc_scale_is_multiplied_in(self):
        """``adc.scale`` carries adc_count/K and is the ONLY leaf between the
        tone and the flagger that carries a unit.

        With ``scale: 10`` the cut is ten times the Kelvin number and the
        message says so, which is what a mutant dropping the multiplication
        dies on.
        """
        document = tone_and_flagger(threshold=3.0,
                                    adc={"scale": {"value": 10.0,
                                                    "unit": "adc_count/K"},
                                         "n_bits": 12})
        document["inference"]["noise"]["sigma"]["unit"] = "adc_count"
        assert built_only(document, "A43").message == a43_message(
            "3", "444.472", "6.74957e-05", scale="10")

    def test_bandpass_and_gain_are_NOT_multiplied_in(self):
        """The TRAP, and the base document is what makes it testable.

        ``model.gain`` is ``1.1`` in the shipped fixture, so an implementation
        that folded the dimensionless stages in would report ``48.8919``
        rather than ``44.4472`` -- and schema §11 records that as the wrong
        way round: "gain sits before adc on the trunk, so gain is genuinely
        dimensionless and only adc.scale carries adc_count/K".
        """
        run = built_run(tone_and_flagger())
        assert float(run.twin["gain"].gain) == pytest.approx(1.1)
        assert "44.4472" in built_only(tone_and_flagger(), "A43").message
        assert "48.89" not in built_only(tone_and_flagger(), "A43").message

    def test_a_drifting_tone_takes_the_max_over_the_FREQUENCY_axis(self):
        """The TRAP that only a drifting LEVEL can expose.

        ``_weights`` is ``(n_time, n_freq)`` once the tone drifts and
        ``_protection_mask`` takes ``weights.max(axis=-1, keepdims=True)``, so
        the cut is per-sample: ``max_t (protect_floor * level_t * peak_t)``.
        Flattening instead gives ``protect_floor * max_t level_t * max_t
        peak_t``, which is the same number whenever the level is constant --
        so a static tone cannot tell the two apart at all.  On this document
        they are **177.159 and 196.844**, and a threshold between them fires
        one implementation and not the other.
        """
        assert mine(tone_and_flagger(threshold=185.0, tone=DRIFTING)) \
            == frozenset()
        found = built_only(tone_and_flagger(threshold=170.0, tone=DRIFTING),
                           "A43")
        assert found.message == A43_DRIFTING

    def test_a_tone_whose_LEVEL_alone_drifts_is_still_DRIFTING(self):
        """The second leg of ``drift_rate != 0 or amplitude_drift_rate != 0``.

        The shipped ``DRIFTING`` fixture sets BOTH rates, so the ``or``'s
        second operand never decides anything there: dropping it is a
        one-token edit that no test noticed.  On a tone whose centre is fixed
        and whose level climbs, dropping it makes the check compute the STATIC
        cut (44.4472) instead of the real one (177.789) -- so at a threshold
        of 100 the shipped code refuses and the mutant stands down silently
        while the flagger eats the calibrator's shoulders.
        """
        assert "A43" in mine(tone_and_flagger(threshold=100.0,
                                              tone=AMPLITUDE_ONLY))
        assert built_only(tone_and_flagger(threshold=100.0,
                                           tone=AMPLITUDE_ONLY),
                          "A43").message == a43_message(
            "100", "177.789", "0.00562465")
        # and above it, silence -- so the number really is the discriminator
        assert mine(tone_and_flagger(threshold=200.0,
                                     tone=AMPLITUDE_ONLY)) == frozenset()
        # ... while the STATIC tone at the same threshold is silent either way,
        # which is what makes the drifting one the only witness.
        assert mine(tone_and_flagger(threshold=100.0, tone=TONE)) \
            == frozenset()

    def test_a_tone_whose_CENTRE_alone_drifts_is_still_DRIFTING(self):
        """The FIRST leg of the same ``or``, and the twin of the test above.

        Fixing ``amplitude_drift_rate`` with a level-only fixture left
        ``drift_rate`` as the untested operand -- the same R3 shape, one route
        guarded and its sibling open.  Both are now driven, one fixture each.

        On a tone whose centre drifts and whose level does not, dropping
        ``drift_rate != 0.0`` freezes the centre at its t=0 value of 62 MHz
        and computes **28.0445** instead of the real **49.2109** -- so at a
        threshold of 35 the shipped code refuses and the one-legged check
        stands down while the flagger eats the calibrator's shoulders.  The
        frozen tone is loaded here as its own document, so the mutant's
        verdict is asserted rather than described.
        """
        assert "A43" in mine(tone_and_flagger(threshold=35.0,
                                              tone=CENTRE_ONLY))
        assert built_only(tone_and_flagger(threshold=35.0, tone=CENTRE_ONLY),
                          "A43").message == a43_message(
            "35", "49.2109", "0.00711224")
        # what the one-legged check would have compared: the same tone with
        # the drift taken out, which is silent at this threshold.
        assert mine(tone_and_flagger(threshold=35.0,
                                     tone=CENTRE_ONLY_FROZEN)) == frozenset()
        # ... and above the real cut, the drifting one is silent too, so 35 is
        # a discriminator rather than a floor.
        assert mine(tone_and_flagger(threshold=60.0,
                                     tone=CENTRE_ONLY)) == frozenset()

    def test_the_comparison_is_INCLUSIVE_at_the_cut(self):
        """The dispatch boundary, tested at the boundary itself.

        The message's own advice is "Raise …threshold **above** {cut}", so
        ``<=`` is the spelling that makes that advice sufficient: a threshold
        set exactly AT the cut must be accepted, or the remedy is short by one
        ulp and nothing says so.  A strict ``<`` passes every other test in
        this class.
        """
        run = built_run(tone_and_flagger())
        cut = _protection_cut(run.twin["cw_tone"], run.context.freq,
                              run.context.time)
        assert cut == pytest.approx(44.4472, rel=1e-5)
        assert mine(tone_and_flagger(threshold=cut)) == frozenset()
        assert "A43" in mine(tone_and_flagger(threshold=cut * (1 - 1e-12)))

    def test_MomentRFI_has_no_threshold_and_stands_the_check_down(self):
        """§0.3 E.6's first TRAP: ``flagging`` takes two classes.

        ``MomentRFIFlaggingOperator``'s fields are ``config`` and
        ``kernel_shapes`` -- there is no threshold to compare -- and a check
        that reached for ``flagger.threshold`` would raise ``AttributeError``
        inside the pass.

        The document route cannot reach this in an environment without
        MomentRFI: Task 3's A35 refuses ``type: MomentRFIFlaggingOperator`` at
        pre-flight with "MomentRFI is not importable in this environment", so
        the twin is never built.  The operator itself constructs fine (its
        import is deferred to ``__call__``), so the payload is assembled here
        by swapping the flagger on a real run -- which drives the check
        function against the real class rather than a stand-in.
        """
        import dataclasses as dc

        from rheplicant.radio.backend.flagging import (
            FlaggingOperator,
            MomentRFIFlaggingOperator,
        )

        assert not hasattr(MomentRFIFlaggingOperator(), "threshold")
        assert hasattr(FlaggingOperator(threshold=1.0), "threshold")
        run = built_run(tone_and_flagger())
        assert list(_tone_survives_flagging(run))[0].check == "A43"
        swapped = run.twin.replace_node("flagging",
                                        MomentRFIFlaggingOperator())
        run = dc.replace(
            run, twin=swapped,
            inference=run.inference._replace(fit_twin=swapped))
        assert list(_tone_survives_flagging(run)) == []

    def test_the_INFERENCE_TWIN_REPLACE_route_is_walked(self):
        """§0.3 E.10, and the same correction ``twin.py`` records.

        ``Built.twin`` is the RAW twin; ``inference.fit_twin`` is the one
        ``replace:`` rebuilt.  A flagger whose threshold only the fit twin
        carries is compared as the FIT twin holds it, and the finding names
        ``inference.twin.replace.flagging``.
        """
        document = preflight_document(
            model={**BASE_MODEL, "cw_tone": TONE,
                   "flagging": {"type": "FlaggingOperator",
                                "threshold": {"value": 1000.0,
                                               "unit": "adc_count"}}},
            inference={"twin": {"without": ["noise"],
                                "replace": {"flagging": {
                                "type": "FlaggingOperator",
                                "threshold": {"value": 3.0,
                                               "unit": "adc_count"}}}}})
        run = built_run(document)
        assert run.inference.replaced == ("flagging",)
        found = built_only(document, "A43")
        assert found.where == "inference.twin.replace.flagging"
        assert found.message.startswith(
            "inference.twin.replace.flagging.threshold is 3 and")

    def test_the_FIT_twins_adc_scale_is_the_one_compared(self):
        """A ``replace:`` that rebuilds only the ADC still moves the verdict.

        ``adc.scale`` is the one leaf between the tone and the flagger that
        carries a unit, so the twin a comparison is made against decides the
        answer.  Measured here: the raw twin's cut is 44.4472 (scale 1, under
        the 100 threshold, silent) and the fit twin's is 444.472 (scale 10,
        over it), so the finding exists **only** because the fit twin is
        walked -- and dropping that walk kills this test.

        **What this does NOT pin, said out loud:** the third element of the
        de-duplication signature. ``build_fit_twin`` re-assembles, so a
        ``replace:`` of any node mints new objects for all three (measured:
        ``tone shared: False, flagger shared: False, adc shared: False`` on
        this very document) and ``id(adc)`` decides nothing today. That is a
        measured EQUIVALENT mutant and the reason it is kept anyway is written
        beside it in ``optics.py``, not defended by a test that cannot exist.
        """
        document = preflight_document(
            model={**BASE_MODEL, "cw_tone": TONE,
                   "flagging": {"type": "FlaggingOperator",
                                "threshold": {"value": 100.0,
                                               "unit": "adc_count"}},
                   "adc": {"scale": {"value": 1.0,
                                      "unit": "adc_count/K"},
                           "n_bits": 12}},
            inference={"twin": {"without": ["noise"],
                                "replace": {"adc": {"scale": {"value": 10.0,
                                                               "unit": "adc_count/K"},
                                                    "n_bits": 12}}}})
        document["inference"]["noise"]["sigma"]["unit"] = "adc_count"
        run = built_run(document)
        assert run.inference.replaced == ("adc",)
        assert float(run.twin["adc"].scale) == 1.0
        assert float(run.inference.fit_twin["adc"].scale) == 10.0
        assert built_only(document, "A43").message == a43_message(
            "100", "444.472", "0.00224986", scale="10")

    def test_the_pair_is_reported_ONCE_when_nothing_was_replaced(self):
        """The de-duplication, as a property.

        The fit twin is a different object on every document the shipped
        fixture builds (``without: [noise]`` re-assembles), so a walk over
        both twins keyed on the node id would say this twice.  The operators
        are the same objects, and the signature is keyed on their identity.
        """
        run = built_run(tone_and_flagger())
        assert run.inference.fit_twin is not run.twin
        assert run.inference.fit_twin["cw_tone"] is run.twin["cw_tone"]
        built_only(tone_and_flagger(), "A43")

    def test_the_beam_wins_against_A43(self):
        """§5's ANTI-PROPERTY for this task, in this module too.

        The built slot runs after ``build_resources``, so a document that is
        wrong in A43's way AND carries an unreadable beam is refused by the
        beam.  Saying so is the deliverable; claiming the opposite would be
        schema §6's false preamble.
        """
        document = preflight_document(
            resources=UNREADABLE_BEAM,
            model={**BASE_MODEL, "cw_tone": TONE,
                   "flagging": {"type": "FlaggingOperator",
                                "threshold": {"value": 3.0,
                                               "unit": "adc_count"}}})
        with pytest.raises(ConfigError) as raised:
            load_document(document)
        assert "no_such_beam.npy" in str(raised.value)
        assert "check A43" not in str(raised.value)


class TestB9:
    """One beam, two analyses -- a WARN, and the honest framing of it."""

    @pytest.fixture(autouse=True)
    def _needs_limtod(self):
        pytest.importorskip("limtod_jax")

    def test_it_fires_as_a_WARN_and_names_the_second_entry(self, tmp_path):
        found = built_only(two_projectors(tmp_path), "B9",
                           base_dir=str(tmp_path))
        assert found.severity == WARN
        assert found.where == "resources.projectors.second"

    def test_the_whole_driftscan_message(self, tmp_path):
        assert built_only(two_projectors(tmp_path), "B9",
                          base_dir=str(tmp_path)).message == B9_DRIFTSCAN

    def test_a_WARN_does_not_stop_the_load(self, tmp_path):
        """``built_only`` reads all three severities (§0.3 C.4) and a WARN
        counts as the one; the document still loads."""
        assert load_document(two_projectors(tmp_path),
                             base_dir=str(tmp_path)) is not None

    def test_one_projector_earns_nothing(self, tmp_path):
        assert mine(preflight_document(resources=projector_sections(tmp_path)),
                    base_dir=str(tmp_path)) == frozenset()

    def test_a_DIFFERENT_lmax_is_different_work_and_earns_nothing(self,
                                                                 tmp_path):
        document = two_projectors(tmp_path, second={"lmax": 7})
        assert mine(document, base_dir=str(tmp_path)) == frozenset()

    def test_a_DIFFERENT_beam_iterations_is_different_work(self, tmp_path):
        """§0.3 E.6 ruling 1: the key is ``beam_iterations``, not
        ``iterations``.

        ``build_projector`` passes ``int(spec.get("beam_iterations", 3))`` to
        the analysis, so two entries differing in it do different work --
        measured, their ``beam_alms`` are not even equal.  A criterion keyed
        on the plan's ``iterations`` would read ``None`` for both and group
        them anyway.
        """
        document = two_projectors(tmp_path, second={"beam_iterations": 5})
        assert mine(document, base_dir=str(tmp_path)) == frozenset()

    def test_nside_is_NOT_in_the_criterion(self, tmp_path):
        """§0.3 E.6 ruling 1's other half.

        ``nside`` is *explicitly refused* for ``engine: driftscan`` -- measured
        at ``build_projector``'s own "nside is not written for engine:
        driftscan. from_beam_maps() infers it from the map length" -- so a
        criterion naming it could never fire on the engine B9 is mostly about.
        """
        with pytest.raises(ConfigError, match="nside is not written for"):
            load_document(two_projectors(tmp_path, second={"nside": 4}),
                          base_dir=str(tmp_path))
        specs = _analysing(two_projectors(tmp_path))
        assert len(specs) == 2
        for _ref, _lmax, _iterations, _engine in specs.values():
            assert (_ref, _lmax, _iterations) == ("resources.beams.horn", 8, 3)

    def test_two_general_pointing_projectors_differing_ONLY_in_nside_still_fire(
            self, tmp_path):
        """The half of "drop nside" that a driftscan document cannot show.

        On ``engine: driftscan`` the key is refused outright, so including it
        in the criterion changes no verdict there and looks harmless.  On
        ``general_pointing`` it is a legal key that **does not enter the beam
        analysis at all** -- ``build_projector`` calls ``_analyse(name,
        beam.maps, lmax, beam_iterations)`` and passes ``nside`` only to the
        constructor -- so two entries differing in nside alone do the
        identical transform twice, and a criterion carrying nside would put
        them in different groups and say nothing.
        """
        document = two_projectors(tmp_path, first=GENERAL_POINTING,
                                  second={"nside": 8})
        run = built_run(document, base_dir=str(tmp_path))
        one = run.resources.resources["resources.projectors.one"]
        two = run.resources.resources["resources.projectors.two"]
        assert (one.nside, two.nside) == (4, 8)
        assert one.beam_alms is not two.beam_alms
        assert mine(document, base_dir=str(tmp_path)) == {"B9"}

    def test_two_general_pointing_projectors_are_the_named_twin(self, tmp_path):
        """S3: ``general_pointing`` against ``driftscan``.

        Both engines call ``_analyse(name, beam.maps, lmax, beam_iterations)``,
        so both pay twice -- and the criterion is engine-blind for exactly
        that reason.  What differs is the REMEDY, which is why the message is
        assembled rather than fixed.
        """
        document = two_projectors(tmp_path, first=GENERAL_POINTING)
        found = built_only(document, "B9", base_dir=str(tmp_path))
        assert found.severity == WARN
        assert found.message == B9_GENERAL_POINTING

    def test_the_general_pointing_advice_applied_earns_nothing(self, tmp_path):
        """S4/R4: the remedy the message names, applied.

        Measured: ``{ref: resources.projectors.one.beam_alms}`` analyses the
        beam ONCE and hands both projectors the same array (``is``-identical),
        which is the thing bitwise equality could never have told apart.
        """
        document = two_projectors(
            tmp_path, first=GENERAL_POINTING,
            second={"beam_alms": {"ref": "resources.projectors.one.beam_alms"}})
        run = built_run(document, base_dir=str(tmp_path))
        one = run.resources.resources["resources.projectors.one"]
        two = run.resources.resources["resources.projectors.two"]
        assert one.beam_alms is two.beam_alms
        assert mine(document, base_dir=str(tmp_path)) == frozenset()

    def test_the_driftscan_advice_now_names_a_remedy_that_works(self, tmp_path):
        """This test used to assert the OPPOSITE, and the change is the point.

        It was ``test_the_driftscan_advice_is_that_there_ISNT_ONE_and_it_is
        _measured``, and it was right: ``beam_alms:`` was a general_pointing
        key, so the remedy earned ``does not take ['beam_alms']`` two gates
        later -- a live R4 advice loop, and the message said plainly that no
        edit existed. A8.6 opened the route, and the sentence became the last
        thing between a user and a remedy that now works.

        A "measured" note is exactly what this failure mode preys on: the
        measurement stays quotable long after the thing it measured has
        changed, and it reads as more authoritative than prose while doing it.
        What kept it honest was this test -- the measurement was executable,
        so opening the route turned the note red instead of leaving it to be
        found by a reader who trusted it.

        So: apply the advice the message now gives, and assert it builds and
        shares.
        """
        document = two_projectors(tmp_path, second={
            "beam_alms": {"ref": "resources.projectors.drift.beam_alms"},
            "nside": 4,
        })
        run = built_run(document, base_dir=str(tmp_path))
        first = run.resources.resources["resources.projectors.drift"]
        second = run.resources.resources["resources.projectors.second"]

        assert second.beam_alms is first.beam_alms
        # ... and with one analysis shared, B9 has nothing left to report.
        assert mine(document, base_dir=str(tmp_path)) == frozenset()

    def test_the_driftscan_advice_says_what_the_alms_route_additionally_needs(
            self, tmp_path):
        """The one asymmetry that survives: alms carry no pixel count.

        Applying the remedy WITHOUT ``nside:`` is refused, which is why the
        driftscan advice carries a clause the general_pointing advice does
        not. A message that named the remedy and omitted this would send a
        reader into a second refusal -- the R4 loop, one step further along
        than the one the old sentence avoided.
        """
        with pytest.raises(ConfigError, match="nside"):
            load_document(
                two_projectors(tmp_path, second={
                    "beam_alms": {"ref": "resources.projectors.drift.beam_alms"}}),
                base_dir=str(tmp_path))
        assert "write nside: too" in B9_DRIFTSCAN

    def test_an_entry_supplying_its_OWN_beam_alms_analyses_nothing(self,
                                                                  tmp_path):
        """An entry carrying ``beam_alms:`` is excluded from the grouping.

        The mutant this kills is dropping that filter: with a beam_alms taken
        from a ref the identity suppression below hides it, but an entry whose
        alms are written as a literal array is neither identical to the
        other's nor a second analysis -- it is no analysis at all.
        """
        import numpy as np

        alms = np.zeros((2, 45), dtype=np.complex64)
        np.save(tmp_path / "alms.npy", alms)
        document = two_projectors(
            tmp_path, first=GENERAL_POINTING,
            second={"beam_alms": {"file": {"path": "alms.npy",
                                           "format": "npy"},
                                  "dtype": "complex64"}})
        run = built_run(document, base_dir=str(tmp_path))
        one = run.resources.resources["resources.projectors.one"]
        two = run.resources.resources["resources.projectors.two"]
        assert one.beam_alms is not two.beam_alms
        assert mine(document, base_dir=str(tmp_path)) == frozenset()

    def test_two_entries_that_really_DO_share_one_array_earn_nothing(
            self, tmp_path):
        """The identity suppression, killed by a payload rather than a document.

        No document reaches this state today -- the ``{ref:}`` route is
        excluded by the spec filter above -- so the clause would be an
        equivalent mutant if it were only ever driven through documents.  It
        is not: the day ``build_resources`` memoises an identical spec, the
        text criterion still groups the two and only object identity says the
        transform ran once.  This assembles that payload by hand.
        """
        document = two_projectors(tmp_path)
        run = built_run(document, base_dir=str(tmp_path))
        assert list(_beam_analysed_twice(run))[0].check == "B9"
        shared = dict(run.resources.resources)
        shared["resources.projectors.second"] = \
            shared["resources.projectors.drift"]
        run = dataclasses.replace(
            run, resources=run.resources._replace(resources=shared))
        assert list(_beam_analysed_twice(run)) == []

    def test_bitwise_equality_is_NOT_the_criterion(self, tmp_path):
        """The TRAP, stated as a measurement.

        Two analyses of one map with the same statics are deterministic, so
        the two arrays are bitwise EQUAL and not the same object -- equality
        is what the wrong answer looks like too.  An implementation keyed on
        it would also fire on the ``{ref:}`` pair, which is the document that
        took the advice.
        """
        import numpy as np

        run = built_run(two_projectors(tmp_path), base_dir=str(tmp_path))
        one = run.resources.resources["resources.projectors.drift"]
        two = run.resources.resources["resources.projectors.second"]
        assert one.beam_alms is not two.beam_alms
        assert np.array_equal(np.asarray(one.beam_alms),
                              np.asarray(two.beam_alms))

    def test_a_MIXED_engine_group_takes_the_driftscan_sentence(self, tmp_path):
        """A mixed group is advised on the member the warning attaches to.

        One driftscan and one general_pointing over the same beam at the same
        ``lmax``/``beam_iterations`` is a legal document and a real group: both
        analyse, and the pair is not ``is``-identical.

        This used to take the driftscan branch and be told no edit existed --
        conservative, and the docstring argued for it on the grounds that
        overstating would be an R4 loop. With the route open on both engines
        the honest answer is available: advise the entry the warning is on,
        and add the ``nside:`` clause only when that entry is a driftscan.
        Here it is the general_pointing one.
        """
        sections = projector_sections(tmp_path)
        sections["projectors"]["gp"] = dict(GENERAL_POINTING)
        document = preflight_document(resources=sections)
        found = built_only(document, "B9", base_dir=str(tmp_path))
        assert found.severity == WARN
        assert found.where == "resources.projectors.gp"
        # The advice is keyed on the ADVISED entry, which here is the
        # general_pointing one -- so it gets the plain remedy, with no
        # `nside:` clause, because a general_pointing entry taking alms
        # already declares its own nside.
        assert "Write beam_alms:" in found.message
        assert "write nside: too" not in found.message

    def test_the_beam_wins_against_B9(self, tmp_path):
        """§5's ANTI-PROPERTY, for B9 -- the row most likely to be believed to
        save something, because its message quotes millisecond figures.

        It saves nothing.  The two analyses have already been paid by the time
        this pass runs, and a document that is both B9-shaped and carrying an
        unreadable beam is refused by **the beam** -- before either projector
        exists.
        """
        sections = projector_sections(tmp_path)
        sections["projectors"]["second"] = dict(sections["projectors"]["drift"])
        sections["beams"] = dict(UNREADABLE_BEAM["beams"])
        with pytest.raises(ConfigError) as raised:
            load_document(preflight_document(resources=sections),
                          base_dir=str(tmp_path))
        assert "no_such_beam.npy" in str(raised.value)
        assert "check B9" not in str(raised.value)

    def test_the_example_that_must_not_fire_is_not_a_document_at_all(self):
        """§3.2(e) names ``examples/driftscan_mmode.py`` as the pair that
        must not be refused.  Checked, and the answer is stronger than "it is
        a WARN": that file builds its projectors by calling
        ``DriftScanProjector.from_beam_maps`` directly and declares no
        ``resources:`` section, so this check cannot reach it.
        """
        source = (_ROOT / "examples" / "driftscan_mmode.py").read_text()
        assert "from_beam_maps" in source
        assert "resources.projectors" not in source
        assert _analysing({"resources": None}) == {}


class TestTheChecksSurviveAValueThatIsTheWrongPythonType:
    """A document value a check reads may be a list, a dict, a set or ``None``.

    B9 puts three of them into a dict KEY -- an unhashable one raises
    ``TypeError`` inside the check, which ``sweep`` reports as "check 'B9'
    RAISED", aborting the pass and hiding every finding after it while every
    ``match=`` pin in the suite still passes.  So each value is type-checked
    before it is grouped on, and this is what says so.
    """

    @pytest.fixture(scope="module")
    def payload(self):
        return built_run(tone_and_flagger())

    @pytest.mark.parametrize("value", HOSTILE)
    @pytest.mark.parametrize("key", ["model", "resources", "inference"])
    def test_a_hostile_section(self, payload, key, value):
        run = dataclasses.replace(payload,
                                  document={**payload.document, key: value})
        assert isinstance(list(_tone_survives_flagging(run)), list)
        assert isinstance(list(_beam_analysed_twice(run)), list)

    @pytest.mark.parametrize("value", HOSTILE)
    @pytest.mark.parametrize(
        "key", ["engine", "beam", "lmax", "beam_iterations", "beam_alms"])
    def test_a_hostile_projector_key(self, payload, key, value):
        """Every key B9 reads off a resolved spec, including the three it puts
        in a grouping key."""
        spec = {"engine": "driftscan",
                "beam": {"ref": "resources.beams.horn"}, "lmax": 8}
        document = {**payload.document,
                    "resources": {"projectors": {"p": {**spec, key: value},
                                                 "q": {**spec, key: value}}}}
        run = dataclasses.replace(payload, document=document)
        assert isinstance(list(_beam_analysed_twice(run)), list)

    @pytest.mark.parametrize("field", ["twin", "inference", "context",
                                       "resources", "document"])
    def test_a_payload_field_that_is_not_what_it_should_be(self, payload,
                                                           field):
        """``document`` is on this list, and it is the one that bites here.

        ``_analysing`` calls ``document.get("resources")``, so a payload whose
        ``document`` is not a Mapping raises ``AttributeError`` inside B9 --
        which ``sweep`` reports as "in-flight check 'B9' RAISED", aborting the
        pass.  It was missing from this list while its four siblings were
        present, which made the module's own claim to admit hand-built
        payloads false for exactly the field two checks dereference.
        """
        run = dataclasses.replace(payload, **{field: object()})
        assert isinstance(list(_tone_survives_flagging(run)), list)
        assert isinstance(list(_beam_analysed_twice(run)), list)

    @pytest.mark.parametrize("value", HOSTILE)
    def test_a_hostile_DOCUMENT_on_the_payload(self, payload, value):
        """The same sweep with real values rather than a bare ``object()``.

        A list and a dict both reach ``.get`` differently, and ``None`` and
        ``3`` differently again; none is reachable from a real run
        (``run.document`` is always ``_assemble``'s variant-applied Mapping)
        and all are reachable from a hand-built payload.
        """
        run = dataclasses.replace(payload, document=value)
        assert isinstance(list(_tone_survives_flagging(run)), list)
        assert isinstance(list(_beam_analysed_twice(run)), list)


class TestTheCostOfTheseTwoRows:
    """A CALL COUNT first, and a wall clock only where no count expresses it.

    "Under a bound" cannot tell a check that reads a shape from one that runs
    the model on an eight-sample document, and an eight-sample document is
    what every test here uses.  Zero calls can.
    """

    def test_A43_does_not_evaluate_the_twin(self, monkeypatch):
        """The payload is built FIRST: ``built_run`` runs a real forward pass
        of its own (the fixture's ``observed: {from: simulation}``), so a
        counter installed before it could never be about the pass."""
        from rheplicant.core.graph import Assembly

        run = built_run(tone_and_flagger())
        calls = []
        original = Assembly.__call__
        monkeypatch.setattr(
            Assembly, "__call__",
            lambda self, state: (calls.append(1), original(self, state))[1])
        assert "A43" in built(run).checks()
        assert calls == []

    def test_the_tone_operator_itself_is_never_called(self, monkeypatch):
        """The narrower count, and the one that names the line between this
        plan and 3C: A43 reads ``_weights``, which is closed form, and never
        ``CWCalibrationOperator.__call__``, which injects the tone."""
        from rheplicant.radio.instrument.calibration import CWCalibrationOperator

        run = built_run(tone_and_flagger())
        calls = []
        monkeypatch.setattr(CWCalibrationOperator, "__call__",
                            lambda self, state: calls.append(1))
        assert "A43" in built(run).checks()
        assert calls == []

    def test_B9_analyses_no_beam(self, monkeypatch, tmp_path):
        """B9's own anti-property: the check that COUNTS the analyses does not
        add one.  ``_analyse`` is ``build_projector``'s own entry point into
        healpy, so a zero here is the whole property."""
        pytest.importorskip("limtod_jax")
        from rheplicant.config.kinds import projectors

        run = built_run(two_projectors(tmp_path), base_dir=str(tmp_path))
        calls = []
        monkeypatch.setattr(projectors, "_analyse",
                            lambda *a, **k: calls.append(1))
        assert "B9" in built(run).checks()
        assert calls == []

    def test_neither_of_THESE_two_reads_a_file(self, monkeypatch, tmp_path):
        """The dynamic half of §5's Scope box, on a document where BOTH run.

        ``test_config_inflight.py``'s walk bans the filesystem verbs by NAME
        at the call site and writes down that it cannot see indirection -- a
        module here calling a helper in ``sections/`` that itself opens a
        file.  This is the complement, and it only complements anything if the
        rows are actually doing work: the twin module's first cut of this
        assertion drove a document on which three of the four rows stood
        down, and a planted ``open()`` in A43's live path survived it.

        So the document lights a tone, a flagger AND two projectors over one
        beam, and both ids are asserted present before the counter is read.
        """
        import builtins

        sections = projector_sections(tmp_path)
        sections["projectors"]["second"] = dict(sections["projectors"]["drift"])
        document = preflight_document(
            resources=sections,
            model={**BASE_MODEL, "cw_tone": TONE,
                   "flagging": {"type": "FlaggingOperator",
                                "threshold": {"value": 3.0,
                                               "unit": "adc_count"}}})
        run = built_run(document, base_dir=str(tmp_path))
        opened = []
        original = builtins.open
        monkeypatch.setattr(
            builtins, "open",
            lambda *a, **k: (opened.append(a[0]), original(*a, **k))[1])
        fired = built(run).checks()
        assert {"A43", "B9"} <= fired, fired
        assert opened == []

    def test_A43_on_a_tone_document_costs_a_fraction_of_a_millisecond(self):
        """**0.7 ms against a measured 0.122 ms best case** -- about x6.

        The only per-document cost either row has: ``_weights`` is a handful
        of ``jnp`` ops over ``(n_time, n_freq)``, paid once per load of a
        document that lights both a tone and a flagger.  Six and not ten, for
        the reason ``test_config_inflight.py``'s cost class gives: a x10
        slowdown must go red, and it does at six.
        """
        run = built_run(tone_and_flagger(threshold=1000.0))
        built(run)                                        # warm
        assert best_ms(lambda: built(run), repeats=30) < 0.7
