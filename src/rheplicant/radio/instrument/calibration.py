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
through the gain (Eqs. 13-16: delta P_cw ~ g(nu_cw, t)).

Real physics to come: the tone's spectral shape, drift/stability, its own
reflection and noise contributions, and the switched reference loads used
for noise-wave calibration (GCR draft). The placeholder injects a constant
amplitude into the channel nearest the tone frequency.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class CWCalibrationOperator(AbstractOperator):
    """Inject a narrow-band CW tone into the nearest frequency channel.

    Attributes:
        amplitude: tone amplitude [K-equivalent] — differentiable scalar.
        tone_freq: tone frequency [Hz] (static configuration).
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "cw_tone"

    amplitude: jax.Array
    tone_freq: float = eqx.field(static=True)

    def __check_init__(self):
        if self.tone_freq <= 0:
            raise StateValidationError(f"tone_freq must be > 0, got {self.tone_freq}.")

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None:
            raise StateValidationError("CWCalibrationOperator requires state.coords.freq.")
        channel = jnp.argmin(jnp.abs(state.coords.freq - self.tone_freq))
        return state.with_data(state.data.at[:, channel].add(self.amplitude))


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
