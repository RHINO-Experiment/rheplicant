"""CWCalibrationOperator — the continuous-wave calibration tone, as a line.

Elements: "Calibration signals, which are switched in and out of the signal
path on some pre-defined cycle. Each calibration source has its own signal
shape, particularly in frequency, as well as typical power levels, stability,
and additional reflection and noise contributions."

RHINO's central design choice (paper, Sect. 4): a continuous-wave source
injects a known, narrow-band, large-amplitude signal so the overall gain
level is monitored *continuously*, without Dicke switching.

THE MODEL
---------
A monochromatic injection is never observed as a delta on one channel. It is
observed *through the spectrometer*, so what lands in the data is

    T_cw(t, nu_k) = A(t) * w_k(t),      sum_k w_k(t) = 1

with ``w`` the channel response evaluated at the line's offset from each
channel and normalised over the sampled channels, and

    nu_c(t) = tone_freq + drift_rate * (t - t_0)          [Hz]
    A(t)    = amplitude * (1 + amplitude_drift_rate * (t - t_0))   [K]

Normalising over channels is the load-bearing choice: it makes the injected
TOTAL equal to ``amplitude`` whatever the lineshape, whatever the width, and
wherever the line falls between two channels. The tone's level is the one
thing this operator knows; a total that moved with the channelisation would
make the known quantity unknown, which is the whole calibration argument.
The price is that the *peak channel* is no longer ``amplitude`` — a line
sitting halfway between two channels keeps only ~0.42 of it (the classic
half-bin scalloping loss of an unwindowed FFT, -3.8 dB), which is real and is
exactly the bias a delta-on-one-channel model hides.

WHAT IS ESTABLISHED, AND WHAT IS ASSUMED
----------------------------------------
Established in this repository: channel bandwidth equals channel spacing
equals band / n_freq (``docs/sky-to-receiver.md``, ``docs/inference.md``,
``examples/sky_to_noise_wave.py``) — the critically-sampled convention.

NOT established anywhere here, in ``rhino-cal``, or in ``rhino_cal_jax``:
RHINO's spectrometer window or polyphase-filterbank taps, the source's own
linewidth, and its frequency and amplitude stability over a run. Those are
therefore PARAMETERS, not constants:

* ``lineshape`` — ``"sinc2"`` (default) is the response of a critically
  sampled *unwindowed* FFT, the minimal assumption consistent with the
  convention above. ``"gaussian"`` approximates an apodised PFB channel.
  A windowed spectrometer has a WIDER main lobe and far lower sidelobes than
  ``sinc2``, so assuming ``sinc2`` under-protects the core and over-protects
  the tails. Set the shape you have.
* ``line_width`` — no default. It is a property of the spectrometer (and of
  the source), and guessing it silently mis-sizes the protection mask, which
  is the failure this operator exists to avoid.
* ``drift_rate`` / ``amplitude_drift_rate`` — first-order (linear) drift over
  a run, default zero. Linear is an assumption too: it is the leading term of
  any smooth drift over a run short compared with the oscillator's thermal
  time constant, and nothing here establishes that it is short.

WHAT A TONE WITH WIDTH ACTUALLY MEASURES
----------------------------------------
A delta on one channel probes ``b(nu_cw) * g(t)`` — one bandpass value, which
is what the placeholder claimed and what an on-centre ``sinc2`` tone still
delivers exactly. A line with width probes ``sum_k w_k b(nu_k) * g(t)``: the
lineshape-weighted AVERAGE of the bandpass across the line's wings. On a
curved bandpass those are different numbers — measured at 2.37% apart for a
1.5-channel gaussian on a realistically curved band in
``tests/radio/test_cw_lineshape.py`` — and reading the second as the first
biases the recovered bandpass at the tone's channel by the curvature times the
line's second moment. A narrower tone is a sharper probe; that is the trade
against the identifiability point below, which pushes the same way.

ORDERING CONSTRAINT: the tone is combined with the antenna signal *before*
the receiver chain — paper Eq. 6: ``P_rec = g(nu,t) (T_ant + T_nw + T_cw)
+ T_n``. This operator must therefore sit BEFORE the bandpass and gain
operators in a pipeline: the tone tracks g(t) drift only if it passes
through the gain (Eqs. 13-16: delta P_cw ~ g(nu_cw, t)). That is no longer
prose — it is declared as ``must_precede`` and :func:`~rheplicant.core.graph.
assemble` refuses a placement that breaks it.

WHAT THE TONE BUYS, precisely: nothing on its own. Measured with
:func:`~rheplicant.inference.identifiability.identifiability`, a known tone
leaves the nullity of a gain x T_ant model at ``n_time`` whether it is
switched on or off, because a free-per-cell antenna temperature absorbs the
gain sample by sample — including at the tone's own channel. It earns its
keep only against a frequency-SMOOTH ``T_ant`` (nullity 1 -> 0), where a
narrow line is not in the span of the smooth basis and cannot be reabsorbed.

Giving the line a width makes that WORSE, not better, and the direction is
measured in ``tests/radio/test_cw_lineshape.py``: the residual of the tone's
channel profile outside a degree-4 polynomial basis falls by more than an
order of magnitude (0.84 -> 0.038) between a quarter-channel line and a line
at ``MAX_WIDTH_IN_BAND_FRACTION`` of that band, which is the widest line the
width guard admits at all.
A wide line moves *into* the span of the smooth basis it was supposed to be
distinguishable from. Realism here costs leverage; nothing in this module
makes the tone independently sufficient, and no docstring should imply it.

Real physics still to come: the tone's own reflection and noise
contributions, and the switched reference loads used for noise-wave
calibration (GCR draft).
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.coordinates import (
    MAX_TIME_RESOLUTION_IN_SAMPLES as _MAX_TIME_RESOLUTION_IN_SAMPLES,
)
from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.radio.protection import PROTECTED_KEY, protect

#: Lineshapes this operator can evaluate, in the order the refusal names them.
LINESHAPES = ("sinc2", "gaussian")

#: Narrowest line, in channel spacings, each shape may be given on a grid.
#:
#: These are numerical floors, not physics claims, and they differ because the
#: two shapes fail differently below them.
#:
#: ``sinc2`` has zeros at every integer multiple of ``line_width``. Give it a
#: width below one channel and the sampled channels start landing ON those
#: zeros: at half a channel, a line midway between two channels sits at
#: offsets +/-1, +/-3, +/-5 — every one a null — and the normalisation then
#: divides by a sum that is float noise. At one channel exactly (the value a
#: critically-sampled unwindowed FFT actually has) the nearest channel is
#: never further than half a null-width away, so its weight never drops below
#: (2/pi)^2 = 0.405 and the sum is always O(1).
#:
#: ``gaussian`` is evaluated with its peak exponent subtracted, so its largest
#: weight is exactly 1 and the sum is never smaller than 1 — it cannot
#: underflow at any width. Its floor guards the other end: ``(nu/sigma)**2``
#: overflows to ``inf`` for a sigma small enough, and ``inf - inf`` is NaN.
#: A quarter of a channel is FWHM 0.59 channels, already narrower than an
#: unwindowed FFT's 0.886, so nothing below it is a channel response.
MIN_WIDTH_IN_CHANNELS = {"sinc2": 1.0, "gaussian": 0.25}

#: Slack on the width floor, because both sides of it are float32 arithmetic.
#:
#: The canonical ``sinc2`` width is exactly one channel, and the natural way to
#: say that is ``float(freq[1] - freq[0])`` — which on a float32 grid differs
#: from this module's ``median(diff(freq))`` in the seventh digit. Refusing the
#: one width the convention names, over 1e-7, would be absurd. 1e-5 is many
#: orders of magnitude below any width difference that changes a lineshape.
WIDTH_FLOOR_RTOL = 1e-5

#: Widest line, as a fraction of the observed band, that is still a LINE.
#:
#: The floor's mirror, and the other direction the class docstring already
#: names as a silent failure. Past some width the injection stops being a
#: narrow feature and becomes a pedestal across the whole band: every channel
#: lands above ``protect_floor`` of the peak, the protection mask covers the
#: band, and the RFI flagger is switched off for the entire run — genuine RFI
#: surviving into the data, which is the "protect too much" half of the trade.
#:
#: The number is where that starts, at the default ``protect_floor`` of 1e-2,
#: for the worst-case placement (tone at one band edge, channel at the other,
#: an offset of the full band ``B``). For a gaussian of ``sigma = f B`` that
#: channel keeps ``exp(-1/(2 f^2))`` of the peak; for a ``sinc2`` of width
#: ``f B`` its envelope keeps ``(f/pi)^2``:
#:
#:     f       gaussian edge/peak    sinc2 envelope edge/peak
#:     0.25    3.4e-4                6.3e-3      both BELOW the 1e-2 floor
#:     1/3     1.1e-2                1.1e-2      both ABOVE it
#:
#: Both shapes cross the default floor within a percent of each other at
#: ``f = 1/3``; 0.25 is the round value below that crossing, leaving the far
#: side of the band outside the mask by 30x (gaussian) and 1.6x (sinc2).
#:
#: What this does NOT catch, stated because it is measured: a tone nearer the
#: middle of a NARROW band saturates its mask sooner — on the 11-channel band
#: of ``tests/radio/test_cw_lineshape.py`` with the tone on channel 4, a
#: 2-channel gaussian (f = 0.2) already protects all 11. No fraction-of-band
#: rule can express that, because it depends on where the tone sits and on
#: ``protect_floor``. This ceiling refuses a width that is not a line on ANY
#: placement; the level rule owns the rest.
MAX_WIDTH_IN_BAND_FRACTION = 0.25

#: Floor under the ceiling, in channel spacings, so a coarse grid cannot make
#: a one-channel line "too wide".
#:
#: ``MAX_WIDTH_IN_BAND_FRACTION * (high - low)`` is ``0.25 * (n_freq - 1)``
#: channels, which drops BELOW one channel once the grid has four channels or
#: fewer — a 4-channel grid would refuse the critically-sampled width the floor
#: calls canonical. Two channel spacings because an apodised polyphase channel
#: has a wider main lobe than the unwindowed FFT the floor is written for (see
#: ``lineshape`` above), and two is a generous bound on that. It binds only on
#: coarse grids: ``0.25 * (n_freq - 1) >= 2`` from ``n_freq = 9`` up, so on any
#: real spectrometer band the band fraction is the operative limit.
MIN_CEILING_IN_CHANNELS = 2.0

#: Largest fraction of one sample interval that ``coords.time``'s own
#: representable resolution may occupy, before a drift is measured against
#: times that are not there. Re-exported, not redefined: the cut is a property
#: of how ``coords.time`` is STORED rather than of this operator's arithmetic,
#: so it is stated once in :mod:`rheplicant.core.coordinates` — which now
#: refuses such an axis at construction, ahead of every consumer — together
#: with the calibration of the 1e-2 itself.
#:
#: What this operator adds on top of the container's check is the measurement
#: below. On an 11-channel 1 MHz grid, 4 samples 100 s apart, ``drift_rate``
#: 1e4 Hz/s — one channel per sample:
#:
#:     coords.time            elapsed              peak ch     protected/11
#:     [0,100,200,300]        [0,100,200,300]      [4,5,6,7]   1, 1, 1, 1
#:     1.75e9 + the same      [0,128,256,256]      [4,5,7,7]   1, 6, 8, 8
#:
#: Two of the four samples collapse onto the same time, the tone lands in the
#: wrong channel, and the mask blows out from one channel to eight of eleven —
#: this operator's own named silent failure. Nothing raises, nothing is NaN,
#: every shape is right; the same run under ``JAX_ENABLE_X64=1`` gives
#: ``[4,5,6,7]``, so the cause is precision, not logic. Both the injection and
#: the band guard read ``t - t[0]``, so they read the SAME corrupted elapsed
#: values and the band guard cannot see it — subtracting the anchor cannot undo
#: a rounding that already happened at store time.
MAX_TIME_RESOLUTION_IN_SAMPLES = _MAX_TIME_RESOLUTION_IN_SAMPLES


def _static_setting(name: str, array_why: str, traced_why: str):
    """Build a converter for a KNOWN instrument setting: a static scalar float.

    Two things are refused rather than tolerated, for every setting that uses
    this.

    A non-scalar value, because these are single numbers describing the
    injection, and an array would broadcast into the frequency or time axis
    and silently model something the physics does not have.

    A traced or otherwise non-numeric value, because ``static=True`` puts the
    field in the treedef: a tracer there is a leak out of the trace, and an
    array there is unhashable and breaks jit caching. Equinox's
    ``eqx.partition(op, eqx.is_inexact_array)`` then cannot pick the field up
    as a parameter, which is the enforcement.
    """

    def convert(value) -> float:
        shape = jnp.shape(value)
        if shape != ():
            raise StateValidationError(
                f"{name} must be a scalar, got shape {shape}. {array_why}"
            )
        try:
            return float(value)
        except Exception as exc:  # tracer, or anything else with no float value
            raise StateValidationError(
                f"{name} must be a KNOWN, static number, got "
                f"{type(value).__name__}. {traced_why}"
            ) from exc

    return convert


_static_amplitude = _static_setting(
    "amplitude",
    "The CW tone is a single injected level; an array here would broadcast "
    "into the tone's channel and silently model a per-sample or per-channel "
    "tone.",
    "The tone's level is an instrument setting the operator knows, not a "
    "parameter to infer — a tone of unknown amplitude constrains nothing, "
    "because the gain it is meant to track absorbs it exactly.",
)


def _named_setting(name: str):
    """The generic KNOWN-setting converter, with the field's own name in it."""
    return _static_setting(
        name,
        "It describes the injection as a whole, not one sample or one channel.",
        "It is an instrument setting the operator knows, not a parameter to infer.",
    )


class CWCalibrationOperator(AbstractOperator):
    """Inject a CW tone with a lineshape, a width, and a drift.

    Three things beyond the injection itself.

    **It declares its own position.** ``must_precede`` names the stages the
    tone must flow through, and ``assemble()`` enforces it — see the module
    docstring.

    **It protects what it wets.** A narrow bright line is precisely what an
    RFI flagger is built to remove, and flagging sits downstream of this
    operator on the same trunk, so both shipped flaggers erase the tone at
    fraction 1.0 on the first observation. The protection therefore rides in
    ``state.aux[PROTECTED_KEY]``, written HERE — the operator that injected
    the tone is the one that knows where it went — rather than as a flagger
    setting the user has to remember to switch on. A flagger has no way to
    tell a calibration tone from RFI; this operator has no way not to.

    The protected set is every channel where the tone contributes at least
    ``protect_floor`` of its own peak channel. In Kelvin that cut sits at
    ``protect_floor * amplitude * max_k w_k``, which is the number to compare
    against a flagging threshold: with a 5000 K tone at the default 1e-2, the
    protection covers every channel carrying more than about 50 K of tone.
    Both directions are silent failures, which is why the rule is a stated
    level and not a guess: protect too little and the flagger takes the line's
    shoulders, biasing the tone's measured level low; protect too much and
    genuine RFI survives in channels the tone barely touched, which is sky
    thrown away. A level rule rather than a "fraction of the tone's power"
    rule because ``sinc2``'s wings fall off only as 1/x^2 — containing 99% of
    an unwindowed FFT's power takes ~40 channels, nearly all of them carrying
    a tone contribution no flagger would ever have noticed.

    **A drifting tone protects a drifting set.** When ``drift_rate`` is zero
    the mask is a ``(n_freq,)`` channel mask, as before. When the line moves,
    the contaminated channels move with it and the mask becomes a
    ``(n_time, n_freq)`` waterfall — the shape
    :func:`~rheplicant.radio.protection.unflag_protected` already reads for a
    switched calibrator. A channel mask for a drifting tone would protect the
    right channel in the first sample and the wrong one in every other.
    Note ``amplitude_drift_rate`` alone does NOT make the mask 2-D: the
    weights are normalised, so a changing level does not move the line.

    Attributes:
        amplitude: the tone's TOTAL contribution [K-equivalent], summed over
            channels — a known, STATIC scalar, not a differentiable leaf. Note
            this is the total, not the peak channel; see the module docstring.
        tone_freq: tone centre frequency at the first sample [Hz]. Must lie
            inside the observing band for the whole run: a centre outside it
            spreads the line over channels it is nowhere near and calibrates
            nothing.
        line_width: lineshape scale [Hz]. For ``"sinc2"`` this is the offset
            to the first null — one channel spacing for a critically sampled
            unwindowed FFT. For ``"gaussian"`` it is the standard deviation
            (FWHM = 2.355 * line_width). No default: see the module docstring.
            It is the CHANNEL response, not the band, and it is guarded from
            both sides — ``MIN_WIDTH_IN_CHANNELS`` below,
            ``MAX_WIDTH_IN_BAND_FRACTION`` above.
        lineshape: ``"sinc2"`` or ``"gaussian"``.
        drift_rate: centre-frequency drift [Hz/s], linear in ``coords.time``
            from the first sample. Nonzero requires ``coords.time``, and
            requires it to be an axis whose stored precision can express the
            run's own cadence — see ``MAX_TIME_RESOLUTION_IN_SAMPLES``.
        amplitude_drift_rate: FRACTIONAL level drift [1/s], linear from the
            first sample. Nonzero requires ``coords.time``, with the same
            precision condition.
        protect_floor: protect every channel at or above this fraction of the
            tone's peak channel contribution. In (0, 1].
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data", f"aux.{PROTECTED_KEY}")
    graph_node: ClassVar[str] = "cw_tone"
    must_precede: ClassVar[tuple[str, ...]] = ("bandpass", "gain")
    must_precede_because: ClassVar[str] = (
        "The tone tracks g(nu, t) only by passing through it (paper Eqs. 13-16, "
        "delta P_cw ~ g(nu_cw, t)); injected after the gain its response is "
        "exactly 1.0 and it monitors nothing at all."
    )

    amplitude: float = eqx.field(static=True, converter=_static_amplitude)
    tone_freq: float = eqx.field(static=True, converter=_named_setting("tone_freq"))
    line_width: float = eqx.field(static=True, converter=_named_setting("line_width"))
    lineshape: str = eqx.field(static=True, default="sinc2")
    drift_rate: float = eqx.field(
        static=True, default=0.0, converter=_named_setting("drift_rate")
    )
    amplitude_drift_rate: float = eqx.field(
        static=True, default=0.0, converter=_named_setting("amplitude_drift_rate")
    )
    protect_floor: float = eqx.field(
        static=True, default=1e-2, converter=_named_setting("protect_floor")
    )

    def __check_init__(self):
        if self.tone_freq <= 0:
            raise StateValidationError(f"tone_freq must be > 0, got {self.tone_freq}.")
        if self.line_width <= 0:
            raise StateValidationError(
                f"line_width must be > 0, got {self.line_width}. It is the "
                "spectrometer's channel-response scale in Hz; a zero or negative "
                "width has no lineshape to evaluate."
            )
        if self.lineshape not in LINESHAPES:
            raise StateValidationError(
                f"lineshape must be one of {LINESHAPES!r}, got {self.lineshape!r}. "
                "'sinc2' is a critically sampled unwindowed FFT (line_width = the "
                "offset to the first null); 'gaussian' approximates an apodised "
                "polyphase channel (line_width = sigma)."
            )
        if not 0.0 < self.protect_floor <= 1.0:
            raise StateValidationError(
                f"protect_floor must be in (0, 1], got {self.protect_floor}. It is "
                "a fraction of the tone's own peak channel: 1.0 protects the peak "
                "channel only, and 0 would protect the entire band, handing every "
                "channel's RFI verdict to a calibrator that touches one line."
            )

    # -- the injection -------------------------------------------------------

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None:
            raise StateValidationError("CWCalibrationOperator requires state.coords.freq.")
        elapsed = self._elapsed(state)
        self._validate_over_the_run(state.coords.freq, state.coords.time)

        centre = self.tone_freq
        if self.drift_rate != 0.0:
            centre = centre + self.drift_rate * elapsed
        level = self.amplitude
        if self.amplitude_drift_rate != 0.0:
            level = level * (1.0 + self.amplitude_drift_rate * elapsed)

        weights = self._weights(state.coords.freq, centre)
        return state.replace(
            data=state.data + level * weights,
            aux=protect(state.aux, self._protection_mask(weights)),
        )

    def _elapsed(self, state: State) -> jax.Array | None:
        """``(t - t_0)`` as ``(n_time, 1)``, or None when nothing drifts.

        ``coords.time`` is a requirement of the DRIFT, not of the operator, so
        a static tone still runs in a pipeline that carries no time axis at
        all (a single-spectrum fit, say). Asking for it unconditionally would
        break those; not asking for it when it is needed would silently freeze
        the drift at zero, which looks exactly like a stable tone.

        ``state.coords`` itself is not re-checked here: ``__call__`` refused a
        state without coords before this ran, so the disjunct would be a branch
        no input can reach and no test can pin.
        """
        if self.drift_rate == 0.0 and self.amplitude_drift_rate == 0.0:
            return None
        if state.coords.time is None:
            raise StateValidationError(
                "A drifting CW tone needs coords.time: drift_rate="
                f"{self.drift_rate} Hz/s and amplitude_drift_rate="
                f"{self.amplitude_drift_rate} /s are both measured from the first "
                "sample, and without sample times there is nothing to measure them "
                "against. Either supply coords.time or set both rates to zero."
            )
        n_time = state.coords.time.shape[0]
        if state.data.shape[0] != n_time:
            raise StateValidationError(
                f"coords.time has {n_time} time samples but data has "
                f"{state.data.shape[0]}. A drifting tone builds one level and one "
                "centre PER SAMPLE, so mismatched axes would broadcast to the "
                "wrong length instead of failing — a run of a different duration "
                "than the data it was built from."
            )
        return (state.coords.time - state.coords.time[0])[:, None]

    def _weights(self, freq: jax.Array, centre) -> jax.Array:
        """Channel response at ``freq``, normalised to sum to 1 over channels.

        Shaped ``(n_freq,)`` for a fixed centre and ``(n_time, n_freq)`` for a
        drifting one, which is what makes the protection mask pick up the
        right dimensionality for free.
        """
        offset = (freq - centre) / self.line_width
        if self.lineshape == "sinc2":
            shape = jnp.sinc(offset) ** 2
        else:  # "gaussian" — __check_init__ admits nothing else
            # Subtract the peak exponent before exponentiating: the largest
            # weight is then exactly 1, so the sum is >= 1 and no width can
            # underflow every channel to zero and turn the normalisation into
            # 0/0. The ratios, which are all that survive normalisation, are
            # unchanged.
            squared = offset**2
            shape = jnp.exp(-0.5 * (squared - squared.min(axis=-1, keepdims=True)))
        return shape / shape.sum(axis=-1, keepdims=True)

    def _protection_mask(self, weights: jax.Array) -> jax.Array:
        """Every channel at or above ``protect_floor`` of the peak channel.

        ``>=`` rather than a rank cut, deliberately: a line sitting exactly
        between two channels gives them equal weight, and any "keep the best N"
        rule would keep one of the pair and hand the other to the flagger.
        Ties resolve towards protecting more, which is the direction that keeps
        the calibrator whole.
        """
        return weights >= self.protect_floor * weights.max(axis=-1, keepdims=True)

    # -- what can be checked, where it can be checked ------------------------

    def _validate_over_the_run(self, freq: jax.Array, time: jax.Array | None) -> None:
        """Band, width and level checks, over the WHOLE run rather than at t_0.

        The grid arrives per call, so this is the only place these are
        possible. The concreteness escape is the established one (see
        ``DriftScanProjector._validate_uniform_grid``): the frequency and time
        axes are closure constants in every pattern that matters and are still
        readable inside a trace, and when they genuinely are traced arguments
        the checks skip rather than crashing.

        Five silent failures live here.

        ``jnp.sinc`` and ``jnp.exp`` always return SOME number: a tone at
        200 MHz against a 60-85 MHz band still gets normalised weights, and
        the run then models a bright feature that is not the calibrator.

        A tone that starts in band and DRIFTS out of it is worse, because a
        check at the first sample alone passes.

        A line narrower than the grid can carry lands on the lineshape's own
        nulls; a line WIDER than the band stops being a line at all and turns
        the protection mask into a blanket over every channel.

        A time axis whose stored precision cannot express the run's cadence
        makes every elapsed time wrong before any of the above is computed.

        A fractional level drift steep enough to pass through zero turns the
        calibrator into a notch part-way through the run, which is finite,
        correctly shaped and nonsense.
        """
        try:
            channels = np.asarray(freq)
        except jax.errors.TracerArrayConversionError:
            return  # genuinely traced: no values to compare against
        low, high = float(channels.min()), float(channels.max())

        if channels.size > 1:
            spacing = float(np.median(np.abs(np.diff(channels))))
            floor = MIN_WIDTH_IN_CHANNELS[self.lineshape] * spacing
            if self.line_width < floor * (1.0 - WIDTH_FLOOR_RTOL):
                raise StateValidationError(
                    f"line_width {self.line_width:.6g} Hz is narrower than the "
                    f"channel response this {self.lineshape!r} grid can carry "
                    f"({MIN_WIDTH_IN_CHANNELS[self.lineshape]:g} x the "
                    f"{spacing:.6g} Hz channel spacing = {floor:.6g} Hz). The "
                    "sampled channels would land on the lineshape's own nulls, or "
                    "overflow its exponent, and the normalisation would divide by "
                    "float noise. A spectrometer cannot resolve a line this far "
                    "below its own channel width: either widen line_width or "
                    "supply the finer frequency grid the width belongs to."
                )
            ceiling = max(
                MAX_WIDTH_IN_BAND_FRACTION * (high - low),
                MIN_CEILING_IN_CHANNELS * spacing,
            )
            if self.line_width > ceiling:
                raise StateValidationError(
                    f"line_width {self.line_width:.6g} Hz is wider than a LINE on "
                    f"this band ({MAX_WIDTH_IN_BAND_FRACTION:g} x the "
                    f"{high - low:.6g} Hz band, or {MIN_CEILING_IN_CHANNELS:g} x the "
                    f"{spacing:.6g} Hz channel spacing, whichever is larger = "
                    f"{ceiling:.6g} Hz). Nothing would raise: the weights would "
                    "still normalise and the injected total would still be exactly "
                    "the amplitude, but what they model is a PEDESTAL spread over "
                    "the whole band rather than a line. Every channel then sits "
                    "above protect_floor of the peak, so the protection mask covers "
                    "the band and the RFI flagger is switched off for the entire "
                    "run — genuine RFI surviving into the data, which is sky thrown "
                    "away. line_width is the spectrometer's CHANNEL response, not "
                    f"the band: one channel here is {spacing:.6g} Hz."
                )

        first, last = 0.0, 0.0
        if self.drift_rate != 0.0 or self.amplitude_drift_rate != 0.0:
            try:
                times = np.asarray(time)
            except jax.errors.TracerArrayConversionError:
                return  # traced times: the run's extent is unknowable here
            if times.size > 1:
                self._refuse_a_time_axis_it_cannot_resolve(times)
            elapsed = times - times[0]
            first, last = float(elapsed.min()), float(elapsed.max())

        centres = (
            self.tone_freq + self.drift_rate * first,
            self.tone_freq + self.drift_rate * last,
        )
        if not low <= min(centres) or not max(centres) <= high:
            drift = (
                ""
                if self.drift_rate == 0.0
                else f" (drifting at {self.drift_rate:.6g} Hz/s over the run)"
            )
            raise StateValidationError(
                f"the CW tone centre spans [{min(centres):.6g}, {max(centres):.6g}] "
                f"Hz{drift}, which is outside the observed band "
                f"[{low:.6g}, {high:.6g}] Hz. The lineshape would still be "
                "evaluated and still be normalised, so the run would model a "
                "bright feature spread across channels the tone is nowhere near, "
                "tracking the gain of the wrong part of the band."
            )

        levels = (
            1.0 + self.amplitude_drift_rate * first,
            1.0 + self.amplitude_drift_rate * last,
        )
        if min(levels) <= 0.0:
            raise StateValidationError(
                f"amplitude_drift_rate {self.amplitude_drift_rate:.6g} /s takes the "
                f"tone's level to {min(levels):.6g} of its initial value over this "
                "run, so it stops being a tone and becomes a notch part-way "
                "through — a negative injection that the gain would then be fitted "
                "against. A fractional drift is a first-order model of a stable "
                "source; a run long enough to cancel the source is outside it."
            )

    @staticmethod
    def _refuse_a_time_axis_it_cannot_resolve(times: np.ndarray) -> None:
        """Refuse a ``coords.time`` whose STORED precision has eaten the cadence.

        The one check here that is not about the tone at all: it is about the
        axis the drift is measured against.

        Most of what this used to catch alone is now caught earlier, by
        :func:`rheplicant.core.coordinates._refuse_a_time_axis_the_stored_dtype_cannot_carry`,
        which applies the same ratio at the point where ``jnp.asarray`` does the
        rounding — so a unix-second axis never reaches this operator at all.
        Read that function for the shared reasoning: ``np.spacing`` on the
        array's own scalar rather than on a Python float, ``np.abs`` on both the
        peak and the gaps, non-finite values named before any comparison.

        What is still this operator's own is the **smallest gap including
        zero**. The container takes the smallest DISTINCT gap, because it cannot
        tell a genuinely repeated timestamp from a collision and has no business
        refusing the first. This operator can: it subtracts times, so two
        samples sharing an elapsed value means the tone silently stops drifting
        across them, which is precisely this operator's named failure. Rounding
        makes every nonzero gap a multiple of the resolution, so a run that has
        already lost samples to collision reports a smallest gap of exactly
        zero and is refused by the same comparison, with the zero in the message
        saying why.
        """
        resolution = float(np.spacing(np.abs(times).max()))
        cadence = float(np.min(np.abs(np.diff(times))))
        if resolution <= MAX_TIME_RESOLUTION_IN_SAMPLES * cadence:
            return
        raise StateValidationError(
            f"coords.time is stored as {times.dtype} and reaches "
            f"{float(np.abs(times).max()):.9g}, where consecutive representable "
            f"numbers are {resolution:.6g} s apart — but the closest two samples "
            f"in this run are {cadence:.6g} s apart, and this operator requires "
            f"the first to be at most {MAX_TIME_RESOLUTION_IN_SAMPLES:g} of the "
            "second. The rounding happened when the axis was STORED, so (t - t[0]) "
            "cannot recover it: the drift is then computed against elapsed times "
            "that are wrong, samples can collapse onto one another, and nothing "
            "raises — the tone simply lands in the wrong channel and the "
            "protection mask moves with it. read_rhino_observation sets "
            "coords.time from obs.time_s, which is UNIX SECONDS (~1.75e9) and "
            "quantises to a 128 s grid in float32. Either pass sample times "
            "measured from the start of the run, or enable float64 "
            "(JAX_ENABLE_X64=1, or jax.config.update('jax_enable_x64', True))."
        )


class CalLoadOperator(AbstractOperator):
    """Switched calibration load (PLACEHOLDER).

    Elements: "Calibration signals, which are switched in and out of the
    signal path on some pre-defined cycle." The load REPLACES the antenna
    signal on the switching cycle — modeled by the ``receiver_input``
    *selector* node of the canonical graph: provide this operator alongside
    the antenna chain and put the switching cycle in
    ``coords.extra["receiver_input"]`` (0 = antenna, 1 = load, in the
    graph's edge order).

    Real physics to come (GCR draft): warm/hot loads with their own
    reflection coefficients and physical-temperature telemetry.

    ``cal_loads`` is ``many=True`` and feeds only the ``receiver_input``
    selector, so instances compose the way that consumer composes: each one
    becomes its OWN switch position rather than being summed with its siblings.
    Provide two loads alongside the antenna chain and the switch indexes
    0 = antenna, 1 = first load, 2 = second load — the graph's in-edge order,
    then the order they were given, so ``assemble()`` expresses a switching
    cycle of any length with no hand-wiring. How long that cycle has to be for
    ``NoiseWaveOperator``'s temperatures to be identifiable is stated once, in
    :mod:`~rheplicant.radio.instrument.noise_wave` — it is
    ``min(n_src, k) * n_freq`` over the ``k`` FREE temperature families while
    they are free per channel, so three loads are enough only when ``T_rx`` is
    held known, and no fixed number covers a basis parameterization at all.
    Read the order off the assembly (``twin["receiver_input"].names``) rather
    than assuming it: it is the order ``gamma_src``'s rows must match.

    Attributes:
        t_load: load temperature [K], differentiable. Four accepted forms, and
            which axis a 1-D array runs along is a convention this package
            states once rather than guessing:

            * scalar — one temperature for the whole run;
            * ``(n_freq,)`` — per channel. A bare 1-D array is ALWAYS read this
              way, matching
              :class:`~rheplicant.radio.instrument.noise_wave.NoiseWaveOperator`'s
              temperature leaves, so the two cannot disagree on a square grid;
            * ``(n_time, 1)`` — per sample. This is the form a real recording
              takes: a load's physical temperature drifts through a run and is
              logged per sample, not per channel. Spelled as an explicit column
              rather than a bare ``(n_time,)`` for the reason
              :func:`~rheplicant.inference.noise.check_noise_std_axis` gives at
              length — on a square grid ``(n,)`` reads equally well as either
              axis, and NumPy settles it by aligning trailing axes, silently;

            :func:`rheplicant.radio.rhino.cal_load_operators` builds the
            ``(n_time, 1)`` form from a recording's thermistor log.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "cal_loads"

    t_load: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "CalLoadOperator requires state.coords with time and freq axes."
            )
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        if self.t_load.ndim == 0:
            return state.with_data(self.t_load * jnp.ones((n_time, n_freq)))
        if self.t_load.ndim == 1:
            if self.t_load.shape[0] != n_freq:
                raise StateValidationError(
                    f"t_load has {self.t_load.shape[0]} channels but coords.freq "
                    f"has {n_freq}. A 1-D t_load is always read as per-FREQUENCY "
                    f"(the convention NoiseWaveOperator's temperatures use); for "
                    f"a per-SAMPLE temperature pass an explicit ({n_time}, 1) "
                    f"column, which is what a recording's thermistor log gives."
                )
            return state.with_data(jnp.ones((n_time, 1)) * self.t_load[None, :])
        if self.t_load.ndim == 2:
            # `(n_time, 1)` only, not also `(n_time, n_freq)`. The demonstrated
            # need is a load's PHYSICAL temperature, which drifts through a run
            # and is logged per sample -- one number per sample, no spectrum.
            # A load whose spectrum also moved would be a different model and
            # is not one this placeholder has; a narrower guard is easier to
            # widen when that arrives than to narrow after someone relies on it.
            if self.t_load.shape != (n_time, 1):
                raise StateValidationError(
                    f"t_load has shape {tuple(self.t_load.shape)}; a 2-D t_load "
                    f"is the per-SAMPLE form and must be exactly ({n_time}, 1), "
                    f"one temperature per time sample."
                )
            return state.with_data(
                jnp.broadcast_to(self.t_load, (n_time, n_freq))
            )
        raise StateValidationError(
            f"t_load must be scalar, (n_freq,) or (n_time, 1); got "
            f"ndim={self.t_load.ndim}."
        )


class ApplyCalibrationOperator(AbstractOperator):
    """Apply an inferred gain solution: ``data / gain`` (PLACEHOLDER).

    The inverse of :class:`~rheplicant.radio.instrument.gain.GainOperator` — the
    bridge between calibration *inference* (``rheplicant.inference``, which
    produces the gain solution) and calibrated-data *analysis* (filters,
    map-making). Real version: full calibration application — bandpass
    division, noise-wave subtraction, tone-tracked g(t) interpolation.

    Attributes:
        gain: inferred gain — differentiable scalar or ``(n_time,)`` array.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "apply_cal"

    gain: jax.Array

    def __call__(self, state: State) -> State:
        if self.gain.ndim == 0:
            return state.with_data(state.data / self.gain)
        if self.gain.ndim == 1:
            if self.gain.shape[0] != state.data.shape[0]:
                raise StateValidationError(
                    f"gain has {self.gain.shape[0]} samples but data has "
                    f"{state.data.shape[0]} time samples."
                )
            return state.with_data(state.data / self.gain[:, None])
        raise StateValidationError(
            f"{type(self).__name__}: gain must be scalar or 1D, got "
            f"ndim={self.gain.ndim}."
        )
