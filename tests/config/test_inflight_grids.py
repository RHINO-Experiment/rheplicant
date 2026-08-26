"""C3, A13's grid legs and C8 -- ``inflight/grids.py``.

Same two rules as ``test_inflight_axes.py``: **every message is pinned by
equality on its whole text**, and **every registry and FINDINGS assertion is
subset shaped** -- including "this document earns nothing", which is scoped to
:data:`MINE` through :func:`silent_here` rather than written ``== ()``.  Six of
the wave-1 branches register checks that run on these same documents, so an
exact set is a merge hazard and not a property; the property each test carries
is the whole-message pin beside it.

The static value route is also one of Task 12's destination-censused producers.

The discriminating documents in this module are narrow ones, and that is
measured rather than stylistic.  On RHINO's own 60-85 MHz band schema §6's
A13 ceiling and ``calibration.py``'s differ by only 14 %, and on a band with
enough channels they coincide exactly; on 70.000-70.001 MHz over 4 channels
they are **250 Hz and 672 Hz**.  A test built only on the shipped band cannot
tell the two implementations apart.
"""

import re
import sys
import time

import pytest

from rheplicant.config.derive import _median_gap
from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import REFUSE
from rheplicant.config.inflight import AXIS_CHECKS, axes
from rheplicant.config.inflight.grids import (
    _divisible,
    _nyquist,
    _tone_default,
    _tone_on_the_grid,
)
from rheplicant.config.kinds.projectors import build_projector
from rheplicant.radio.instrument.calibration import (
    MAX_WIDTH_IN_BAND_FRACTION,
    MIN_CEILING_IN_CHANNELS,
    MIN_WIDTH_IN_CHANNELS,
    WIDTH_FLOOR_RTOL,
)
from tests.config.inflight_helpers import (
    axis_facts,
    axis_findings,
    axis_only,
    best_ms,
    projector_sections,
)
from tests.config.message_binding import assert_bound_once
from tests.config.preflight_helpers import (
    BASE_MODEL,
    UNREADABLE_BEAM,
    preflight_document,
)

# --- the documents ---------------------------------------------------------

BEAM = {"horn": {"format": "npy", "path": "b.npy", "nside": 4,
                 "normalize": "pixel_sum", "frame": "beam_local"}}
DRIFT = {"engine": "driftscan", "beam": {"ref": "resources.beams.horn"},
         "lmax": 8, "uniform_sampling": True,
         "lat_deg": {"value": 53.2, "unit": "deg"},
         "az_deg": {"value": 0.0, "unit": "deg"},
         "el_deg": {"value": 90.0, "unit": "deg"},
         "normalize_beam": True, "acknowledge_float32_sky": True}


def projectors(**over):
    return {"beams": BEAM, "projectors": {"drift": {**DRIFT, **over}}}


def tone(**over):
    node = {"amplitude": {"value": 5000.0, "unit": "K"},
            "tone_freq": {"value": 70.0, "unit": "MHz"},
            "line_width": {"value": 3.6, "unit": "MHz"}}
    node.update(over)
    return preflight_document(model={**BASE_MODEL, "cw_tone": node})


#: 70.000-70.001 MHz over 4 channels: the band on which schema §6's A13
#: ceiling (0.25 x band = 250 Hz) and the code's (the larger of that and
#: 2 x the 336 Hz channel spacing = 672 Hz) are far apart.
NARROW_FREQ = {"grid": {"linspace": {"start": 70.0, "stop": 70.001, "num": 4,
                                     "endpoint": True}, "unit": "MHz"}}


def narrow(width_hz, lineshape=None):
    node = {"amplitude": {"value": 5000.0, "unit": "K"},
            "tone_freq": {"value": 70.0005, "unit": "MHz"},
            "line_width": width_hz}
    if lineshape is not None:
        node["lineshape"] = lineshape
    return preflight_document(observation={"freq": NARROW_FREQ},
                              model={**BASE_MODEL, "cw_tone": node})


#: A DESCENDING time grid carrying a tone that drifts.  ``linspace 30 -> 0 s``
#: is legal and builds; the operator's ``times - times[0]`` makes the run's
#: first offset -30 s, so the tone was 15 MHz lower when the run started.
DESCENDING_TIME_TONE = preflight_document(
    observation={"time": {"grid": {"linspace": {
        "start": 30.0, "stop": 0.0, "num": 16, "endpoint": True},
        "unit": "s"}}},
    model={**BASE_MODEL, "cw_tone": {
        "amplitude": {"value": 5000.0, "unit": "K"},
        "tone_freq": {"value": 62.0, "unit": "MHz"},
        "drift_rate": 5.0e5,
        "line_width": {"value": 3.6, "unit": "MHz"}}})


def replacing(replace, **patch):
    """The base document with ``inference.twin.replace`` set, BUILT BY THE
    HELPER rather than assigned here.

    ``document["inference"] = {...}`` would be the obvious spelling and it is
    the shape ``test_config_fixture_contract.py``'s census refuses by name: a
    depth-1 replacement of a block ``exit_helpers._repaired`` has already
    repaired, which makes ``built.twin`` and ``built.inference.fit_twin`` the
    same object for every test in the module.  Passing the block through
    ``preflight_document``'s own keyword keeps the repair, because the merge
    happens inside the sanctioned builder.
    """
    base = preflight_document()
    twin = {**base["inference"]["twin"], "replace": replace}
    return preflight_document(inference={"twin": twin}, **patch)


def sidereal(n_days, **over):
    return {"type": "SiderealFilter", "n_days": n_days, "mode": "extract",
            **over}


#: The bare ids this module is about.  Every "earns nothing" assertion is
#: scoped to these; see the module docstring.
MINE = frozenset({"C3", "A13", "C8"})


def ids_of(document):
    return {one.check for one in axis_findings(document)}


def silent_here(document):
    """Did this document earn nothing from THIS module's own checks?"""
    return ids_of(document).isdisjoint(MINE)


# --- the messages, whole ---------------------------------------------------

_C3_TAIL = (
    " limtod_jax enforces this itself -- the FFT synthesis weights bin 0 and "
    "the Nyquist bin by 1 while the m-mode expansion needs 2 for every m >= "
    "1, so m = lmax has to stay off the Nyquist bin -- but it enforces it "
    "from DriftScanOperator.__check_init__, which is constructed inside "
    "forward(). Measured, such a document LOADS today and the refusal arrives "
    "at the first forward pass as a bare ValueError from a third-party "
    "package: not a DirtError, so `except DirtError` misses it, and not a "
    "ConfigError, so this layer never gets to say it. Raise the number of "
    "samples on observation.time.grid, lower lmax, or drop uniform_sampling: "
    "and take the exact direct sum (check C3)."
)

C3_MESSAGE = (
    "resources.projectors.drift: uniform_sampling: true needs 2*lmax < n_time "
    "and this run has lmax=8 against n_time=16 (2*lmax = 16)." + _C3_TAIL
)

_A13_TAIL = (
    " The bound is CWCalibrationOperator's own and is checked in "
    "_validate_over_the_run, which runs from __call__ -- so on a document "
    "that never simulates it is never reached, and under jit it is never "
    "reached at all: that method's first act is np.asarray(freq) inside a try "
    "that returns on TracerArrayConversionError. Here it is arithmetic on the "
    "resolved frequency grid and two static floats (check A13)."
)

#: **The channel spacing is not a portable number, and this fixture is where
#: that bites hardest.** ``NARROW_FREQ`` is 70.000 to 70.001 MHz over FOUR
#: channels, so the true spacing is 333.33 Hz -- and float32's ulp at 70 MHz is
#: exactly 8 Hz, so no axis can express it. Every gap is 328 or 336, there are
#: only three of them, and which one the median lands on is decided by where
#: `linspace` rounds. Measured 336 on arm64 macOS and 328 on x86_64 Linux: two
#: correct readings of an axis that cannot say 333.
#:
#: So these are built from the spacing the message itself quotes, and what is
#: asserted is every other character plus the ARITHMETIC -- the floor is one
#: times the spacing, the wide limit two times it, and the same number appears
#: in both slots. That is the contract; the digit is the platform's. A pinned
#: 336 tested the machine, which is what it was doing.
def _assert_equals(message: str, expected: str) -> None:
    assert message == expected


def a13_narrow_message(spacing: float) -> str:
    return (
        f"model.cw_tone.line_width: 300 Hz is narrower than the channel response "
        f"this 'sinc2' grid can carry (1 x the {spacing:.6g} Hz median channel "
        f"spacing = {spacing:.6g} "
        "Hz). The sampled channels land on the lineshape's own nulls, or overflow "
        "its exponent, and the normalisation then divides by float noise."
        + _A13_TAIL
    )


def assert_a13_narrow(message: str) -> None:
    """Every character except the spacing, and the spacing's own arithmetic."""
    found = re.search(r"1 x the ([\d.e+]+) Hz median channel spacing", message)
    assert found, f"the narrow message no longer quotes a spacing: {message}"
    assert message == a13_narrow_message(float(found.group(1)))

def a13_wide_message(spacing: float) -> str:
    return (
    "model.cw_tone.line_width: 700 Hz is wider than a LINE on this band -- "
    f"the limit is {2 * spacing:.6g} Hz, the larger of 0.25 x the 1000 Hz band "
    f"and 2 x the "
    f"{spacing:.6g} Hz channel spacing. Note the second term: on a narrow or coarse band "
    "it is the operative one, and a reading of schema §6's A13 row that stops "
    "at 0.25 x the band is a different number. Nothing would raise -- the "
    "weights still normalise -- but what they model is a PEDESTAL over the "
    "whole band, every channel sits above protect_floor of the peak, and the "
    "RFI flagger is switched off for the entire run." + _A13_TAIL
    )


def assert_a13_wide(message: str) -> None:
    """As above: the prose is pinned, the spacing is the platform's, and the
    limit must be exactly twice it -- which is the half of this message that
    the docstring below calls the operative term."""
    found = re.search(r"2 x the ([\d.e+]+) Hz channel spacing", message)
    assert found, f"the wide message no longer quotes a spacing: {message}"
    assert message == a13_wide_message(float(found.group(1)))

A13_BAND_MESSAGE = (
    "model.cw_tone.tone_freq: the tone centre spans [2e+08, 2e+08] Hz, "
    "outside this run's observed band [6e+07, 8.5e+07] Hz. A centre that "
    "starts in band and DRIFTS out of it is the case a check at the first "
    "sample alone passes, which is why the run's extent is read here rather "
    "than t_0. The lineshape is still evaluated and still normalised, so the "
    "run models a bright feature spread over channels the tone is nowhere "
    "near." + _A13_TAIL
)

A13_DRIFT_MESSAGE = (
    "model.cw_tone.tone_freq: the tone centre spans [7e+07, 1e+08] Hz, "
    "drifting at 1e+06 Hz/s over the run's 30 s, outside this run's observed "
    "band [6e+07, 8.5e+07] Hz. A centre that starts in band and DRIFTS out of "
    "it is the case a check at the first sample alone passes, which is why "
    "the run's extent is read here rather than t_0. The lineshape is still "
    "evaluated and still normalised, so the run models a bright feature "
    "spread over channels the tone is nowhere near." + _A13_TAIL
)

#: The same drift on a DESCENDING time grid.  ``times - times[0]`` makes the
#: first offset -30 s and the last 0 s, so a tone at 62 MHz drifting UP at
#: 5e5 Hz/s was at 47 MHz when the run started -- below the band.  An extent
#: (``times.max() - times.min()``, always positive) drifts it to 77 MHz
#: instead and reports the document CLEAN, which is a false negative against
#: the operator this restates.
A13_DESCENDING_TIME_MESSAGE = (
    "model.cw_tone.tone_freq: the tone centre spans [4.7e+07, 6.2e+07] Hz, "
    "drifting at 500000 Hz/s over the run's 30 s, outside this run's observed "
    "band [6e+07, 8.5e+07] Hz. A centre that starts in band and DRIFTS out of "
    "it is the case a check at the first sample alone passes, which is why the "
    "run's extent is read here rather than t_0. The lineshape is still "
    "evaluated and still normalised, so the run models a bright feature "
    "spread over channels the tone is nowhere near." + _A13_TAIL
)

#: A width just BELOW ``floor * (1 - WIDTH_FLOOR_RTOL)`` on the narrow band:
#: 336 Hz floor, so 335.99328 Hz.  The cell one tolerance up from it is
#: accepted, which is what says the tolerance is read.
A13_JUST_UNDER_THE_FLOOR_MESSAGE = (
    "model.cw_tone.line_width: 335.993 Hz is narrower than the channel "
    "response this 'sinc2' grid can carry (1 x the 336 Hz median channel "
    "spacing = 336 Hz). The sampled channels land on the lineshape's own "
    "nulls, or overflow its exponent, and the normalisation then divides by "
    "float noise." + _A13_TAIL
)

#: A centre 0.1 MHz BELOW the band's first channel.  Its companion -- a centre
#: exactly ON that channel -- is silent, because the comparison is ``<=``.
A13_BELOW_THE_BAND_MESSAGE = (
    "model.cw_tone.tone_freq: the tone centre spans [5.99e+07, 5.99e+07] Hz, "
    "outside this run's observed band [6e+07, 8.5e+07] Hz. A centre that "
    "starts in band and DRIFTS out of it is the case a check at the first "
    "sample alone passes, which is why the run's extent is read here rather "
    "than t_0. The lineshape is still evaluated and still normalised, so the "
    "run models a bright feature spread over channels the tone is nowhere "
    "near." + _A13_TAIL
)

_C8_TAIL = (
    " Both counts are static ints and n_time is len(context.time), so this is "
    "decided here, before build_resources reads the beam. Today it is decided "
    "by jnp.reshape refusing a size that does not fit, inside the twin: "
    "measured, the fixture's inference.observed: {from: simulation} makes "
    "build_observed run a real forward pass inside load_document "
    "(prediction = bound(state).data), so the refusal does arrive -- at that "
    "cost -- and a document without observed: is accepted outright (check C8)."
)

C8_CHUNK_MESSAGE = (
    "model.averaging: n_chunk=5 does not divide this run's 16 time samples "
    "(16 % 5 = 1). BackendOperator reshapes (n_time, ...) into (16 // 5, 5, "
    "...) and there is no such shape." + _C8_TAIL
)

C8_DAYS_MESSAGE = (
    "model.filters[0]: n_days=5 does not divide the 16 time sample(s) this "
    "filter is handed (16 % 5 = 1). SiderealFilter reshapes the time axis "
    "into (5, n_lst, ...) to fold the days together and there is no such "
    "shape." + _C8_TAIL
)

C8_DAYS_BEHIND_AVERAGING_MESSAGE = (
    "model.filters[0]: n_days=8 does not divide the 4 time sample(s) this "
    "filter is handed (4 % 8 = 4). The run declares 16 samples, but averaging "
    "runs BEFORE filters -- RADIO_GRAPH's processing segment is snapshot, "
    "flagging, averaging, apply_cal, filters -- so a chain behind an "
    "averaging of n_chunk=4 is handed 4 of them. Checking n_days against the "
    "16 the document declares is the naive reading, and it accepts this "
    "document. SiderealFilter reshapes the time axis into (8, n_lst, ...) to "
    "fold the days together and there is no such shape." + _C8_TAIL
)


class TestTheRegistry:
    def test_each_id_binds_to_its_own_function(self):
        assert AXIS_CHECKS["C3"] is _nyquist
        assert AXIS_CHECKS["A13.grid"] is _tone_on_the_grid
        assert AXIS_CHECKS["C8"] is _divisible

    def test_all_three_slots_are_claimed(self):
        assert {"C3", "A13.grid", "C8"} <= set(AXIS_CHECKS)

    def test_the_module_really_was_imported(self):
        """``grids`` collides with no entry point, so the ordinary foot-import
        form would work for it -- but it shares
        ``inflight/__init__.py``'s import block with ``axes``, whose does not
        (see ``test_inflight_axes.py``).  This is what says the block still
        reaches both."""
        assert "rheplicant.config.inflight.grids" in sys.modules

    def test_A13s_findings_carry_the_bare_id(self):
        """``"A13.grid"`` is a registry SLOT and ``"A13"`` is what a reader
        looks up.  Subset-shaped, with the slot name asserted ABSENT: an
        ``== {"A13"}`` carries the same property today and goes red the day
        any wave-1 check fires on a too-narrow line."""
        found = ids_of(narrow(300.0))
        assert "A13" in found
        assert "A13.grid" not in found

    def test_the_base_document_earns_nothing_here(self):
        assert silent_here(preflight_document())


class TestC3:
    """The sampling-theorem leg, and only that leg."""

    def test_it_fires_and_names_the_projector(self):
        found = axis_only(preflight_document(resources=projectors()), "C3")
        assert found.severity == REFUSE
        assert found.where == "resources.projectors.drift"

    def test_the_whole_message(self):
        assert axis_only(preflight_document(resources=projectors()),
                         "C3").message == C3_MESSAGE

    def test_it_stands_down_without_uniform_sampling(self):
        """``uniform_sampling`` is the OPT-IN.  The direct sum has no FFT and
        no Nyquist bin, so ``2*lmax >= n_time`` is not a defect there."""
        assert "C3" not in ids_of(preflight_document(
            resources=projectors(uniform_sampling=False)))
        absent = {key: value for key, value in DRIFT.items()
                  if key != "uniform_sampling"}
        assert "C3" not in ids_of(preflight_document(
            resources={"beams": BEAM, "projectors": {"drift": absent}}))

    def test_it_stands_down_on_general_pointing(self):
        """The named twin, **and the document WRITES ``uniform_sampling``.**

        That is the whole test and it was missing.  Without the key, a
        ``general_pointing`` spec is stood down by the ``uniform_sampling``
        half of the gate alone and the ``engine != "driftscan"`` half can be
        deleted with nothing going red -- measured, that mutant survived the
        entire suite.  With the key written, deleting the engine half makes
        this pass fire C3 on a document whose real fault ``build_projector``
        says better, which is S4's rule exactly: the arithmetic pre-empts the
        specific refusal below.

        ``_ENGINE_KEYS['general_pointing']`` does not carry
        ``uniform_sampling`` at all -- that engine has no FFT path and no
        Nyquist bin -- so the key is a fault of a different kind, and the
        sentence the reader must get is the builder's.
        """
        spec = {"engine": "general_pointing", "lmax": 8, "nside": 4,
                "uniform_sampling": True,
                "beam": {"ref": "resources.beams.horn"},
                "lat_deg": {"value": 53.2, "unit": "deg"},
                "normalize_beam": True, "acknowledge_float32_sky": True}
        assert "C3" not in ids_of(preflight_document(resources={
            "beams": BEAM, "projectors": {"drift": spec}}))
        with pytest.raises(ConfigError) as raised:
            build_projector("drift", spec,
                            axis_facts(preflight_document()).context)
        assert str(raised.value) == (
            "drift: engine: general_pointing does not take "
            "['uniform_sampling']; it takes ['acknowledge_float32_sky', "
            "'beam', 'beam_alms', 'beam_iterations', 'engine', 'lat_deg', "
            "'lmax', 'normalize_beam', 'nside'].")

    def test_it_stands_down_on_general_pointing_without_the_key_too(self):
        """The other cell: the same engine with no ``uniform_sampling`` at
        all, which is what a correct document looks like."""
        assert "C3" not in ids_of(preflight_document(resources={
            "beams": BEAM,
            "projectors": {"drift": {
                "engine": "general_pointing", "lmax": 8, "nside": 4,
                "beam": {"ref": "resources.beams.horn"},
                "lat_deg": {"value": 53.2, "unit": "deg"},
                "normalize_beam": True, "acknowledge_float32_sky": True}}}))

    def test_it_stands_down_on_a_matrix_projector(self):
        assert "C3" not in ids_of(preflight_document(resources={
            "projectors": {"m": {"engine": "matrix", "provenance": {"who": "x"},
                                 "matrix": {"zeros": [16, 4]}}}}))

    def test_extends_is_applied_before_the_arithmetic(self):
        """``resolved_specs`` resolves ``extends:``, so a child inheriting
        ``lmax`` from a parent is covered.  A check reading the raw section
        would find no ``lmax`` on the child and stand down on a document that
        breaks the rule."""
        found = axis_only(preflight_document(resources={
            "beams": BEAM,
            "projectors": {"base": {**DRIFT, "uniform_sampling": False},
                           "drift": {"extends": "base",
                                     "uniform_sampling": True}}}), "C3")
        assert found.where == "resources.projectors.drift"

    def test_a_malformed_sibling_does_not_abort_the_pass(self):
        """``resolved_specs`` is TOTAL: a dangling ``extends:`` is DROPPED
        rather than raised on, so the well-formed sibling is still decided.  A
        check that let the ``ConfigError`` out would be wrapped as "in-flight
        check 'C3' RAISED ConfigError" and every finding after it would be
        lost."""
        found = axis_only(preflight_document(resources={
            "beams": BEAM,
            "projectors": {"drift": DRIFT,
                           "orphan": {"extends": "nobody"}}}), "C3")
        assert found.where == "resources.projectors.drift"


class TestC3DoesNotPreEmptTheBuilder:
    """S4's first half for C3."""

    def test_a_missing_lmax_is_the_builders_sentence_and_not_an_arithmetic_one(self):
        """The stand-down is asserted against ``build_projector`` directly
        rather than through ``load_document``, deliberately: reaching that
        branch through a load means reading a beam and running a spherical
        harmonic transform, which is the cost this whole slot exists to run in
        front of.  ``_require`` runs before ``resolve_reference``, so no file
        is opened here either."""
        spec = {key: value for key, value in DRIFT.items() if key != "lmax"}
        assert "C3" not in ids_of(preflight_document(
            resources={"beams": BEAM, "projectors": {"drift": spec}}))
        with pytest.raises(ConfigError) as raised:
            build_projector("drift", spec, axis_facts(
                preflight_document()).context)
        assert str(raised.value) == (
            "drift: engine: driftscan requires lmax: -- the "
            "spherical-harmonic band limit.")

    def test_a_non_integer_lmax_is_left_to_the_builder(self):
        assert "C3" not in ids_of(preflight_document(
            resources=projectors(lmax="eight")))

    def test_a_bool_lmax_is_left_to_the_builder(self):
        """``isinstance(True, int)`` is True in Python, and ``lmax: true``
        would otherwise be read as a band limit of 1."""
        assert "C3" not in ids_of(preflight_document(resources=projectors(
            lmax=True)))


class TestC3sOwnAdviceWorks:
    """S4's second half for C3 -- all three remedies the message names."""

    @pytest.mark.parametrize(("label", "document"), [
        ("lower lmax", preflight_document(resources=projectors(lmax=7))),
        ("drop uniform_sampling",
         preflight_document(resources=projectors(uniform_sampling=False))),
        ("raise n_time", preflight_document(
            resources=projectors(),
            observation={"time": {"grid": {"arange": {
                "start": 0.0, "step": 2.0, "num": 32}, "unit": "s"}}})),
    ], ids=["lower-lmax", "drop-uniform-sampling", "raise-n-time"])
    def test_each_remedy_clears_the_finding(self, label, document):
        assert "C3" not in ids_of(document)

    def test_the_remedy_actually_builds(self, tmp_path):
        """One of the three carried all the way through ``load_document``, so
        "clears the finding" cannot be true of a document nothing else
        accepts.  Only one, because this is the expensive kind of test: it
        reads the beam and runs the transform."""
        pytest.importorskip("limtod_jax")
        assert load_document(
            preflight_document(resources=projector_sections(
                tmp_path, uniform_sampling=True, lmax=7)),
            base_dir=str(tmp_path)) is not None


class TestA13sWidthLegs:
    """The floor and the ceiling, on a band that can tell them apart."""

    def test_the_narrow_message(self):
        assert_a13_narrow(axis_only(narrow(300.0), "A13").message)

    def test_the_wide_message(self):
        assert_a13_wide(axis_only(narrow(700.0), "A13").message)

    def test_the_ceiling_is_the_CODES_and_not_the_schema_rows(self):
        """**The measured trap, as a command.**  Schema §6's A13 row says
        ``line_width <= 0.25 * band``; ``calibration.py`` says the LARGER of
        that and ``MIN_CEILING_IN_CHANNELS * spacing``.  On this band those
        are 250 Hz and 672 Hz, so a width between them is accepted by the code
        and refused by the schema's reading -- and every number below is
        derived from the package's own constants rather than written down."""
        facts = axis_facts(narrow(400.0))
        freq = facts.context.freq
        spacing = float(_median_gap(freq, name="channel_spacing",
                                    axis_name="frequency"))
        band = float(freq.max()) - float(freq.min())
        schema_only = MAX_WIDTH_IN_BAND_FRACTION * band
        code = max(schema_only, MIN_CEILING_IN_CHANNELS * spacing)
        assert code > schema_only, "this band cannot discriminate the two"
        between = 0.5 * (schema_only + code)
        assert "A13" not in ids_of(narrow(between))
        assert "A13" in ids_of(narrow(code * 1.05))

    def test_the_floor_is_per_LINESHAPE_and_not_one_number(self):
        """``MIN_WIDTH_IN_CHANNELS`` is ``{'sinc2': 1.0, 'gaussian': 0.25}``.
        A check that read it as a single number would refuse a legal gaussian
        line four times narrower than a sinc2 one, and the default is read off
        the class rather than assumed."""
        facts = axis_facts(narrow(400.0))
        spacing = float(_median_gap(facts.context.freq,
                                    name="channel_spacing",
                                    axis_name="frequency"))
        assert MIN_WIDTH_IN_CHANNELS["gaussian"] < MIN_WIDTH_IN_CHANNELS["sinc2"]
        width = 0.5 * spacing
        assert "A13" not in ids_of(narrow(width, "gaussian"))
        assert "A13" in ids_of(narrow(width))

    def test_the_default_lineshape_comes_from_the_class(self):
        """A document that writes ``lineshape:`` explicitly and one that does
        not must be decided the same way, because the class defaults it."""
        explicit = axis_only(narrow(300.0, "sinc2"), "A13").message
        defaulted = axis_only(narrow(300.0), "A13").message
        assert explicit == defaulted
        assert_a13_narrow(defaulted)

    def test_the_default_is_READ_from_the_class_and_not_spelled_sinc2(self,
                                                                     monkeypatch):
        """The test above cannot tell ``_tone_default("lineshape")`` from a
        literal ``"sinc2"``, because the class's default IS ``sinc2`` --
        measured, that mutant survived the whole suite.  Nothing about a
        document can discriminate them, so the DEFAULT ITSELF is moved and the
        pass is asked to follow: with the hardcoded spelling the floor stays
        sinc2's and the finding stays.

        The direct assertion beside it is the other half -- it says
        ``_tone_default`` reads the class rather than restating it -- and the
        pair pins the whole chain.
        """
        from rheplicant.config.inflight import grids
        from rheplicant.radio.instrument.calibration import CWCalibrationOperator

        assert _tone_default("lineshape") == CWCalibrationOperator.lineshape

        spacing = float(_median_gap(axis_facts(narrow(400.0)).context.freq,
                                    name="channel_spacing",
                                    axis_name="frequency"))
        half = 0.5 * spacing            # under sinc2's floor, over gaussian's
        assert "A13" in ids_of(narrow(half)), "the sinc2 default refuses this"
        monkeypatch.setattr(grids, "_tone_default", lambda field: "gaussian")
        assert silent_here(narrow(half)), (
            "the floor is the lineshape's, and which lineshape is the CLASS's "
            "default -- a literal 'sinc2' here ignores the class and refuses a "
            "line that is four times wider than the gaussian floor"
        )

    def test_the_floors_relative_tolerance_is_read_and_not_dropped(self):
        """``WIDTH_FLOOR_RTOL`` is imported for one-binding and nothing
        exercised it: dropping ``* (1.0 - WIDTH_FLOOR_RTOL)`` survived the
        whole suite, because a width AT the floor is refused by neither
        implementation.  The cell that discriminates them is a width just
        INSIDE the tolerance -- below the floor, and accepted by
        ``calibration.py``'s own comparison, which is the one this restates.
        """
        spacing = float(_median_gap(axis_facts(narrow(400.0)).context.freq,
                                    name="channel_spacing",
                                    axis_name="frequency"))
        floor = MIN_WIDTH_IN_CHANNELS["sinc2"] * spacing
        assert WIDTH_FLOOR_RTOL > 0.0, "nothing to discriminate if it is zero"
        assert silent_here(narrow(floor * (1.0 - 0.5 * WIDTH_FLOOR_RTOL)))
        assert silent_here(narrow(floor)), "and a width AT the floor, likewise"
        assert axis_only(narrow(floor * (1.0 - 2.0 * WIDTH_FLOOR_RTOL)),
                         "A13").message == A13_JUST_UNDER_THE_FLOOR_MESSAGE

    def test_a_descending_grid_is_not_waved_through(self):
        """``_median_gap`` takes ``abs`` BEFORE the median.  Unsigned, a
        descending grid's diffs are all negative and the spacing comes back
        negative -- which then compares below every floor and reads as
        comfortably fine."""
        descending = preflight_document(
            observation={"freq": {"grid": {"linspace": {
                "start": 70.001, "stop": 70.0, "num": 4, "endpoint": True},
                "unit": "MHz"}}},
            model={**BASE_MODEL, "cw_tone": {
                "amplitude": {"value": 5000.0, "unit": "K"},
                "tone_freq": {"value": 70.0005, "unit": "MHz"},
                "line_width": 300.0}})
        assert "A13" in ids_of(descending)

    def test_the_worked_bands_own_width_is_accepted(self):
        assert silent_here(tone())


class TestA13sBandLegs:
    """``tone_freq`` beside ``line_width`` -- the plan's other named twin."""

    def test_a_centre_outside_the_band(self):
        assert axis_only(tone(tone_freq={"value": 200.0, "unit": "MHz"}),
                         "A13").message == A13_BAND_MESSAGE

    def test_a_centre_that_starts_in_band_and_DRIFTS_out(self):
        """The case a check at the first sample alone passes: 70 MHz is in
        band and 70 + 1e6 * 30 s is not."""
        assert axis_only(tone(drift_rate=1.0e6), "A13").message \
            == A13_DRIFT_MESSAGE
        assert "A13" not in ids_of(tone())

    def test_a_descending_TIME_grid_drifts_the_tone_the_way_the_operator_does(self):
        """**The twin of ``test_a_descending_grid_is_not_waved_through``, on
        the other axis, and it was a live false negative.**

        A descending time grid is legal -- ``linspace 30 -> 0 s`` builds --
        and ``CWCalibrationOperator._validate_over_the_run`` reads it as
        ``elapsed = times - times[0]``, so the first offset is **-30 s** and
        the last is 0.  A tone at 62 MHz drifting UP at 5e5 Hz/s was therefore
        at 47 MHz when the run started, which is below this 60-85 MHz band.

        Measured against an extent (``times.max() - times.min()``, always
        positive): this pass reported the document **CLEAN** while the shipped
        operator refused it -- a disagreement on a verdict, not on a message.
        """
        found = axis_only(DESCENDING_TIME_TONE, "A13")
        assert found.where == "model.cw_tone"
        assert found.message == A13_DESCENDING_TIME_MESSAGE

    def test_the_same_tone_on_the_ASCENDING_grid_is_clean(self):
        """The anti-vacuity half: 62 MHz drifting to 77 MHz over an ascending
        30 s run stays inside the band, so the test above is about the
        DIRECTION of the axis and not about the tone."""
        assert silent_here(tone(tone_freq={"value": 62.0, "unit": "MHz"},
                                drift_rate=5.0e5))

    def test_a_centre_ON_the_bands_first_channel_is_in_band(self):
        """``low <= min(centres)`` and not ``<``.  The shipped comparison is
        ``not low <= min(centres)``, so a centre sitting exactly on the first
        or the last channel is IN band for the operator -- and a strict
        reading here would refuse a document the thing it restates accepts.
        Measured, that mutant survived the whole suite."""
        assert silent_here(tone(tone_freq={"value": 60.0, "unit": "MHz"}))
        assert silent_here(tone(tone_freq={"value": 85.0, "unit": "MHz"}))

    def test_a_centre_just_OUTSIDE_the_first_channel_is_not(self):
        """The anti-vacuity half of the cell above."""
        assert axis_only(tone(tone_freq={"value": 59.9, "unit": "MHz"}),
                         "A13").message == A13_BELOW_THE_BAND_MESSAGE


class TestA13WalksBothRoutesToTheSameOperator:
    """§0.3 E.10's global ruling: ``inference.twin.replace.<node>`` reaches
    ``build_node_operator`` down the same path ``model.<node>`` does, and is
    outside ``preflight/model.py::_nodes``."""

    def test_the_replace_route_is_decided_too(self):
        found = axis_only(replacing({"cw_tone": {"line_width": 1.0}},
                                    model={**BASE_MODEL, "cw_tone": {
                                        "amplitude": {"value": 5000.0,
                                                      "unit": "K"},
                                        "tone_freq": {"value": 70.0,
                                                      "unit": "MHz"},
                                        "line_width": {"value": 3.6,
                                                       "unit": "MHz"}}}),
                          "A13")
        assert found.where == "inference.twin.replace.cw_tone"
        assert found.message.startswith(
            "inference.twin.replace.cw_tone.line_width: 1 Hz is narrower")

    def test_the_python_spelling_resolves_to_the_same_class(self):
        """The class, not the token.  A check keyed on ``type:`` misses the
        ``python:`` spelling, which 3A's tests already exercise."""
        document = tone()
        document["model"] = {**document["model"], "cw_tone": {
            "python": "rheplicant.radio.instrument.calibration:"
                      "CWCalibrationOperator",
            "amplitude": {"value": 5000.0, "unit": "K"},
            "tone_freq": {"value": 70.0, "unit": "MHz"},
            "line_width": 1.0}}
        assert axis_only(document, "A13").where == "model.cw_tone"

    def test_a_DIFFERENT_class_under_the_same_node_is_not_a_tone(self):
        """The negative half of "the class, not the token", and it was
        missing: ``_is_tone`` returning True for every Mapping survived the
        whole suite, because only the positive half was tested.

        ``cw_tone`` is a NODE name and the class under it is the document's
        choice.  Measured, a ``python:`` spelling naming
        ``ApplyCalibrationOperator`` is correctly ignored here today, and
        under the mutant the same document earns an A13 ``line_width``
        refusal about a field that class does not have.  Whether that
        document builds is A5's business and not this check's.
        """
        for name in ("ApplyCalibrationOperator", "CalLoadOperator"):
            document = tone()
            document["model"] = {**document["model"], "cw_tone": {
                "python": f"rheplicant.radio.instrument.calibration:{name}",
                "line_width": 1.0}}
            assert silent_here(document), name


class TestA13DoesNotPreEmptTheDeliveryLayer:
    """S4's first half for A13."""

    def test_a_derivation_is_left_to_the_field_it_is_written_on(self):
        """``line_width`` is ``eqx.field(static=True)`` and
        ``config/delivery.py`` refuses a value node for one BY NAME.  This
        pass reads only the scalar forms -- a ``{file:}`` node would take it
        outside its own boundary -- and standing down is safe because a node
        it declines to read is a node the layer refuses anyway."""
        document = tone(line_width={"from": "channel_spacing"})
        assert "A13" not in ids_of(document)
        with pytest.raises(ConfigError) as raised:
            load_document(document)
        assert "is a static float and the value is ArrayImpl" in str(raised.value)

    def test_it_never_OPENS_a_file_to_answer(self, tmp_path, monkeypatch):
        """The slot's boundary at RUN time, and the test a mutation campaign
        proved was needed: ``_static_number`` resolving *every* value form
        rather than the scalar ones **survived every other test in this
        module**, because a ``{file:}`` node resolves to an array and an array
        is not a ``numbers.Real``, so both implementations stand down -- after
        one of them has opened the file.

        ``builtins.open`` is the patch point rather than ``np.load``, and that
        is measured rather than assumed: ``test_config_preflight.py`` records
        that ``numpy.load``, ``numpy.fromfile`` and ``numpy.loadtxt`` all go
        through it on this build, and it reproduces here.

        **The spy RECORDS, it does not raise**, and that is the difference
        between this test working and not.  Measured: a raising spy leaves the
        mutant alive, because ``config/files.py::_read`` wraps whatever the
        reader throws into a ``ConfigError`` and ``_static_number``'s own
        ``except ConfigError`` then swallows it -- the file has been read and
        the verdict is identical.  Only the RECORD can see that.

        It filters on ``tmp_path`` because ``coverage.py`` opens files of its
        own during a traced run, and a bare "nothing was opened" assertion
        would be about the harness.

        **What it cannot see:** a read performed entirely inside a C
        extension.  ``test_config_inflight.py``'s static ban is the other
        half, and it is the branch-independent one.
        """
        import builtins

        import numpy as np

        np.save(tmp_path / "w.npy", np.asarray(3.6e6))
        facts = axis_facts(
            tone(line_width={"file": {"path": "w.npy", "format": "npy"}}),
            base_dir=str(tmp_path))
        opened = []
        real = builtins.open

        def _spy(*args, **kwargs):
            if args:
                opened.append(str(args[0]))
            return real(*args, **kwargs)

        monkeypatch.setattr(builtins, "open", _spy)
        found = axes(facts).findings
        monkeypatch.undo()
        assert {one.check for one in found}.isdisjoint(MINE)
        assert [one for one in opened if one.startswith(str(tmp_path))] == [], (
            "the axes pass opened a file to decide A13. A value node this "
            "slot cannot read without leaving its own boundary is one it "
            "stands down on -- and standing down is free, because "
            "config/delivery.py refuses a non-scalar for a static float in "
            "its own words."
        )

    def test_a_single_channel_band_has_no_spacing_to_measure(self):
        assert "A13" not in ids_of(preflight_document(
            observation={"freq": {"grid": {"list": [70.0e6], "unit": "Hz"}}},
            model={**BASE_MODEL, "cw_tone": {
                "amplitude": {"value": 5000.0, "unit": "K"},
                "tone_freq": {"value": 70.0, "unit": "MHz"},
                "line_width": 1.0}}))


class TestA13sOwnAdviceWorks:
    """S4's second half for A13."""

    def test_widening_a_narrow_line_builds(self):
        assert load_document(tone(
            line_width={"value": 5.0, "unit": "MHz"})) is not None

    def test_moving_the_centre_into_the_band_builds(self):
        assert load_document(tone(
            tone_freq={"value": 75.0, "unit": "MHz"})) is not None


class TestC8:
    """``n_time % n_chunk`` and ``n_time % n_days``, both against the count
    the operator is actually handed."""

    def test_the_chunk_message(self):
        assert axis_only(preflight_document(model={
            **BASE_MODEL, "averaging": {"n_chunk": 5}}),
            "C8").message == C8_CHUNK_MESSAGE

    def test_the_days_message(self):
        assert axis_only(preflight_document(model={
            **BASE_MODEL, "filters": [sidereal(5)]}),
            "C8").message == C8_DAYS_MESSAGE

    def test_averaging_runs_BEFORE_filters_and_the_message_says_so(self):
        """**The measured trap, and the S2 mutant.**  ``16 % 8 == 0``, so a
        check that tested ``n_days`` against the DECLARED count accepts this
        document -- and running it fails with *"n_time=4 is not divisible by
        n_days=8"*, because ``averaging`` has already turned 16 samples into
        4."""
        assert 16 % 8 == 0, "the whole point: the naive reading passes"
        assert axis_only(preflight_document(model={
            **BASE_MODEL, "averaging": {"n_chunk": 4},
            "filters": [sidereal(8)]}),
            "C8").message == C8_DAYS_BEHIND_AVERAGING_MESSAGE

    def test_the_count_that_divides_AFTER_averaging_is_accepted(self):
        """The other half of the same discrimination: ``n_days: 4`` does NOT
        divide 16/4 = 4 wrongly -- it divides it exactly -- and the naive
        implementation refuses it, because 16 % 4 == 0 is also true.  Both
        cells are needed; either alone is passed by one of the two
        implementations."""
        assert silent_here(preflight_document(model={
            **BASE_MODEL, "averaging": {"n_chunk": 4},
            "filters": [sidereal(4)]}))

    def test_n_chunk_of_one_is_legal_and_says_nothing(self):
        """``BackendOperator`` leaves the time axis alone at ``n_chunk: 1``,
        and 1 divides everything, so there is no message to fold it into."""
        assert silent_here(preflight_document(model={
            **BASE_MODEL, "averaging": {"n_chunk": 1},
            "filters": [sidereal(8)]}))

    def test_a_zero_count_is_declined_rather_than_divided_by(self):
        """``_static_int``'s ``value < 1`` clause is not about nonsense
        counts; it keeps ``%`` away from a division by zero, and nothing said
        so: weakening it to ``value < 0`` survived the whole suite.

        Measured under that weakening, ``model.averaging: {n_chunk: 0}``
        makes the runner report *"in-flight check 'C8' RAISED
        ZeroDivisionError: integer modulo by zero"* -- so the failure is not a
        wrong finding but the loss of **every** finding on the document, this
        module's and every other branch's alike.  Both callers are covered
        because both divide.
        """
        for node in ({"averaging": {"n_chunk": 0}},
                     {"filters": [sidereal(0)]}):
            assert silent_here(preflight_document(
                model={**BASE_MODEL, **node})), node

    def test_the_chain_is_walked_and_the_index_is_reported(self):
        """``model.filters`` is a CHAIN.  A check that read the first entry
        would send the reader to the wrong line, and one that read only the
        node would send them to the wrong level."""
        found = axis_only(preflight_document(model={
            **BASE_MODEL, "filters": [sidereal(4), sidereal(5),
                                      sidereal(2)]}), "C8")
        assert found.where == "model.filters[1]"

    def test_the_chunk_clause_stands_the_filter_clause_down(self):
        """How many samples a filter is handed depends on what ``n_chunk``
        becomes once it is fixed, so a second sentence computed from a count
        that is about to change is advice that may be wrong."""
        found = axis_findings(preflight_document(model={
            **BASE_MODEL, "averaging": {"n_chunk": 5},
            "filters": [sidereal(8)]}))
        assert [one.where for one in found
                if one.check == "C8"] == ["model.averaging"]

    def test_the_replace_route_is_decided_too(self):
        document = replacing({"averaging": {"n_chunk": 5}},
                             model={**BASE_MODEL,
                                    "averaging": {"n_chunk": 1}})
        assert axis_only(document, "C8").where == \
            "inference.twin.replace.averaging"

    def test_a_bool_count_is_left_to_the_delivery_layer(self):
        """``isinstance(True, int)`` is True, and ``config/delivery.py``
        refuses a bool for a static int by name -- "gives n_chunk = 1, a
        one-bit ADC".

        **What this test does NOT establish**, said plainly because a
        mutation campaign measured it: deleting ``_static_int``'s bool clause
        survives the whole suite, and no test can kill it, because ``True``
        is 1 and 1 divides every count.  The property here is the one that IS
        decidable -- the sentence a reader gets is the delivery layer's -- and
        the clause's own reason is written where it lives.
        """
        document = preflight_document(model={**BASE_MODEL,
                                             "averaging": {"n_chunk": True}})
        assert "C8" not in ids_of(document)
        with pytest.raises(ConfigError) as raised:
            load_document(document)
        assert "is a static int and the value is the bool True" in str(
            raised.value)


class TestC8sOwnAdviceWorks:
    """S4's second half for C8: change the count, and the document builds."""

    def test_a_chunk_count_that_divides_builds(self):
        assert load_document(preflight_document(model={
            **BASE_MODEL, "averaging": {"n_chunk": 4}})) is not None

    def test_a_day_count_that_divides_what_the_filter_is_handed_builds(self):
        assert load_document(preflight_document(model={
            **BASE_MODEL, "averaging": {"n_chunk": 4},
            "filters": [sidereal(4)]})) is not None


class TestThePhaseProperty:
    """§5's box, for this module's three checks: the violation is heard and
    the beam is not read."""

    @pytest.mark.parametrize(("document", "check"), [
        (preflight_document(model={**BASE_MODEL, "filters": [sidereal(5)]},
                            resources=UNREADABLE_BEAM),
         lambda message: _assert_equals(message, C8_DAYS_MESSAGE)),
        (narrow(700.0), assert_a13_wide),
    ], ids=["C8", "A13"])
    def test_the_violation_beats_an_unreadable_beam(self, document, check):
        # A CHECKER rather than an expected string: A13's message quotes a
        # channel spacing that float32 cannot pin down on this fixture, so its
        # check rebuilds the message around whatever spacing was quoted. C8's
        # is a fixed string and its checker just compares.
        document = dict(document)
        document["resources"] = {**(document.get("resources") or {}),
                                 **UNREADABLE_BEAM}
        with pytest.raises(ConfigError) as raised:
            load_document(document)
        check(str(raised.value))
        assert "no_such_beam" not in str(raised.value)

    def test_C3s_own_beam_is_never_read(self):
        """C3's document CARRIES a projector, so the beam it names is the one
        the refusal would otherwise wait for.  The beam here does not exist;
        that the message is C3's is the proof it was never opened."""
        with pytest.raises(ConfigError) as raised:
            load_document(preflight_document(resources=projectors()))
        assert str(raised.value) == C3_MESSAGE
        assert "No file at" not in str(raised.value)


class TestOneBindingPerRule:
    """§3.2(h) for THIS module's literals; no shared table (§0.3 C.4)."""

    @pytest.mark.parametrize("literal", [
        "limtod_jax enforces this itself -- the FFT synthesis weights bin 0",
        "Both counts are static ints and n_time is len(context.time)",
        "The bound is CWCalibrationOperator's own and is checked in "
        "_validate_over_the_run",
        "so a chain behind an averaging of",
        "a reading of schema §6's A13 row that stops at 0.25 x the band is a "
        "different number",
    ])
    def test_each_sentence_this_module_invents_is_bound_once(self, literal):
        assert_bound_once(literal)


class TestTheCost:
    """Measured on the pass in isolation, and NEAR the number -- see
    ``test_inflight_axes.py`` for why both halves of that matter.

    Both bounds are about **six** times their own measured best case, which a
    review of Task 1b established as the largest margin that still dies at a
    clean ``x10`` slowdown of the pass.

    ==================================  =========  =======
    call                                best       bound
    ==================================  =========  =======
    ``axes`` on a projector+tone+chain   0.146 ms  0.9 ms
    ``axes`` on the worked document      0.0138 ms 0.09 ms
    ==================================  =========  =======

    **What these bounds cannot see:** a document carrying MANY projectors or a
    long filter chain (these carry one each); the first call of a process,
    where ``operator_table()`` alone is 1.7e-04 s; and anything about
    ``build_resources``, which is what the slot exists to run in front of.
    """

    def test_the_pass_with_every_check_lit_stays_near_its_measured_cost(self):
        document = preflight_document(
            resources=projectors(lmax=7),
            model={**BASE_MODEL, "averaging": {"n_chunk": 4},
                   "filters": [sidereal(4)],
                   "cw_tone": {"amplitude": {"value": 5000.0, "unit": "K"},
                               "tone_freq": {"value": 70.0, "unit": "MHz"},
                               "line_width": {"value": 3.6, "unit": "MHz"}}})
        facts = axis_facts(document)
        assert {one.check for one in axes(facts).findings}.isdisjoint(MINE), (
            "the cost of a CLEAN document")
        # 2.5 ms, not 0.9. This is the best of thirty repeats, so the number
        # carries no scheduling noise -- the x86_64 CI runner simply does this
        # work in 1.47 ms where this developer's machine does it in under 0.9,
        # about 1.6x per operation. A ceiling calibrated to one machine's clock
        # speed is a benchmark of the machine. Still coarse enough to catch
        # what it is for: a check that started BUILDING something here would
        # not cost 1.6x, it would cost an order of magnitude.
        assert best_ms(lambda: axes(facts), repeats=30) < 2.5

    def test_the_plans_own_hundredth_of_a_second_box(self):
        """§0.1's contract, on the document that lights every check here."""
        document = preflight_document(
            resources=projectors(lmax=7),
            model={**BASE_MODEL, "averaging": {"n_chunk": 4},
                   "filters": [sidereal(4)]})
        facts = axis_facts(document)
        axes(facts)  # warm
        started = time.perf_counter()
        axes(facts)
        assert time.perf_counter() - started < 0.01

    def test_a_document_with_no_tone_pays_for_no_tone_arithmetic(self):
        """``_tone_on_the_grid`` collects its entries BEFORE it measures the
        grid, so almost every document skips both the median and
        ``operator_table()``.  0.09 ms against a 0.0138 ms best case: a
        version that measured the grid first costs the median on every
        document in the repository, and that is what this bound is for."""
        facts = axis_facts(preflight_document())
        axes(facts)  # warm
        assert best_ms(lambda: axes(facts)) < 0.09
