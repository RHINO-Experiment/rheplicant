"""CWCalibrationOperator — PLACEHOLDER continuous-wave calibration tone.

Elements: "Calibration signals, which are switched in and out of the signal
path on some pre-defined cycle. Each calibration source has its own signal
shape, particularly in frequency, as well as typical power levels, stability,
and additional reflection and noise contributions."

RHINO's central design choice (paper, Sect. 4): a continuous-wave source
injects a known, narrow-band, large-amplitude signal so the overall gain
level is monitored *continuously*, without Dicke switching.

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
delta at one channel is not in the span of the smooth basis and cannot be
reabsorbed. RHINO's antenna temperature is frequency-smooth, so the route is
sound; the tone is not independently useful and no docstring here should
imply it is.

Real physics to come: the tone's spectral shape, drift/stability, its own
reflection and noise contributions, and the switched reference loads used
for noise-wave calibration (GCR draft). The placeholder injects a constant
amplitude into the channel nearest the tone frequency.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.radio.protection import PROTECTED_KEY, protect


def _static_amplitude(value) -> float:
    """A CW tone amplitude is a KNOWN instrument setting, so: static scalar.

    Two things are refused here rather than tolerated.

    A non-scalar amplitude, because the placeholder adds it to ONE channel and
    a ``(n_time,)`` or ``(n_freq,)`` array would broadcast there silently — a
    per-sample tone that the physics does not have and that the calibration
    argument (a reference of *known, fixed* level) does not permit.

    A traced or otherwise non-numeric amplitude, because ``static=True`` puts
    this field in the treedef: a tracer there is a leak out of the trace, and
    an array there is unhashable and breaks jit caching. The deeper reason is
    physical — the whole point of the tone is that its level is known, so
    inferring it would remove the only thing it contributes. Equinox's
    ``eqx.partition(op, eqx.is_inexact_array)`` now cannot pick it up as a
    parameter, which is the enforcement.
    """
    shape = jnp.shape(value)
    if shape != ():
        raise StateValidationError(
            f"amplitude must be a scalar, got shape {shape}. The CW tone is a "
            "single injected level; an array here would broadcast into the tone's "
            "channel and silently model a per-sample or per-channel tone."
        )
    try:
        return float(value)
    except Exception as exc:  # tracer, or anything else with no float value
        raise StateValidationError(
            f"amplitude must be a KNOWN, static number, got {type(value).__name__}. "
            "The tone's level is an instrument setting the operator knows, not a "
            "parameter to infer — a tone of unknown amplitude constrains nothing, "
            "because the gain it is meant to track absorbs it exactly."
        ) from exc


class CWCalibrationOperator(AbstractOperator):
    """Inject a narrow-band CW tone into the nearest frequency channel.

    Two things beyond the injection itself.

    **It declares its own position.** ``must_precede`` names the stages the
    tone must flow through, and ``assemble()`` enforces it — see the module
    docstring.

    **It protects its own channel.** A narrow bright spike is precisely what
    an RFI flagger is built to remove, and flagging sits downstream of this
    operator on the same trunk, so both shipped flaggers erase the tone at
    fraction 1.0 on the first observation. The protection therefore rides in
    ``state.aux[PROTECTED_KEY]``, written HERE — the operator that injected
    the tone is the one that knows which channel it went into — rather than as
    a flagger setting the user has to remember to switch on. A flagger has no
    way to tell a calibration tone from RFI; this operator has no way not to.

    Attributes:
        amplitude: tone amplitude [K-equivalent] — a known, STATIC scalar, not
            a differentiable leaf.
        tone_freq: tone frequency [Hz] (static configuration). Must lie inside
            the observing band: a tone outside it lands on an edge channel by
            ``argmin`` and calibrates nothing.
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
    tone_freq: float = eqx.field(static=True)

    def __check_init__(self):
        if self.tone_freq <= 0:
            raise StateValidationError(f"tone_freq must be > 0, got {self.tone_freq}.")

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None:
            raise StateValidationError("CWCalibrationOperator requires state.coords.freq.")
        self._check_in_band(state.coords.freq)
        channel = jnp.argmin(jnp.abs(state.coords.freq - self.tone_freq))
        mask = jnp.zeros(state.coords.freq.shape, dtype=bool).at[channel].set(True)
        return state.replace(
            data=state.data.at[:, channel].add(self.amplitude),
            aux=protect(state.aux, mask),
        )

    def _check_in_band(self, freq: jax.Array) -> None:
        """``tone_freq`` must be inside the observing band, checked where it can be.

        The band arrives per call, so this is the only place the check is
        possible. ``argmin`` always returns SOME channel: a tone at 200 MHz
        against a 60-85 MHz band lands on the top edge channel, and the run
        then models a bright spike that is not the calibrator and calibrates
        nothing — finite, correctly shaped, and silent.

        Value checks inside ``__call__`` are otherwise against the operator
        rules, so the concreteness escape is the established one (see
        ``DriftScanProjector._validate_uniform_grid``): the frequency axis is a
        closure constant in every pattern that matters and is still readable
        inside a trace, and when it genuinely is a traced argument the check
        skips rather than crashing.
        """
        try:
            values = np.asarray(freq)
        except jax.errors.TracerArrayConversionError:
            return  # genuinely traced: no values to compare against
        low, high = float(values.min()), float(values.max())
        if not low <= self.tone_freq <= high:
            raise StateValidationError(
                f"tone_freq {self.tone_freq:.6g} Hz is outside the observed band "
                f"[{low:.6g}, {high:.6g}] Hz. argmin would still pick a channel — the "
                "nearest edge one — so the tone would be injected into a channel it is "
                "nowhere near, and the run would model a bright spike that tracks the "
                "gain of the wrong part of the band."
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
        t_load: load temperature [K] — differentiable scalar or ``(n_freq,)``.
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
                    f"has {n_freq}."
                )
            return state.with_data(jnp.ones((n_time, 1)) * self.t_load[None, :])
        raise StateValidationError(
            f"t_load must be scalar or (n_freq,), got ndim={self.t_load.ndim}."
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
        raise StateValidationError(f"gain must be scalar or 1D, got ndim={self.gain.ndim}.")
