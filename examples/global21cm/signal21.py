"""The 21 cm global signal operator: 21cmVAE through ``global21cm_jax``.

``global21cm_jax.operator.GlobalSignal21cmOperator`` carries the emulator as
a constructor field, which a document cannot build. This subclass-free
wrapper loads the shipped weights by default and exposes one traced field,
``theta``, in the log parametrisation 21cmVAE feeds its network:
``(log10 fstar, log10 Vc, log10 fx, tau, alpha, nu_min, Rmfp)``. Sampling
in the log avoids the ``1/(x ln 10)`` gradient factor that diverges at the
``fx`` floor (``global21cm_jax/emulator.py``).
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
from global21cm_jax.emulator import Global21cmEmulator, to_log_parameters
from global21cm_jax.signal import global_signal_kelvin
from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State

from global21cm import scenario


def theta_log_true() -> tuple[float, ...]:
    """The injected parameters in the log parametrisation."""
    return tuple(float(v) for v in to_log_parameters(jnp.asarray(scenario.THETA_TRUE)))


def prior_box() -> tuple[tuple[float, ...], tuple[float, ...]]:
    """The emulator's training box in the log parametrisation (low, high)."""
    emulator = Global21cmEmulator.load()
    return (
        tuple(float(v) for v in emulator.par_min),
        tuple(float(v) for v in emulator.par_max),
    )


def curve_kelvin(theta_log, freqs_mhz) -> jax.Array:
    """``T21(nu)`` in K for log parameters; batched over leading axes of theta."""
    emulator = _emulator()
    freq_hz = jnp.asarray(freqs_mhz) * 1e6
    one = lambda th: global_signal_kelvin(emulator, th, freq_hz, log_parameters=True)  # noqa: E731
    theta_log = jnp.asarray(theta_log)
    if theta_log.ndim == 1:
        return one(theta_log)
    flat = theta_log.reshape(-1, theta_log.shape[-1])
    return jax.vmap(one)(flat).reshape(theta_log.shape[:-1] + (freq_hz.shape[0],))


_CACHE: dict[str, object] = {}


def _emulator() -> Global21cmEmulator:
    if "emulator" not in _CACHE:
        # The first call may come from inside a trace; the weights must be
        # concrete arrays, not tracers that outlive it.
        with jax.ensure_compile_time_eval():
            _CACHE["emulator"] = Global21cmEmulator.load()
    return _CACHE["emulator"]


class Emulated21cmSignal(AbstractOperator):
    """The emulated global signal, constant in LST, in K.

    Attributes:
        theta: ``(7,)`` log-parametrised 21cmVAE parameters.

    The network weights are not a field: they are read from the module
    cache at call time, so they are constants of the traced model and no
    latent path can reach them.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "global_signal"

    theta: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None or state.coords.time is None:
            raise StateValidationError("Emulated21cmSignal requires coords.time and coords.freq.")
        profile = global_signal_kelvin(
            _emulator(), self.theta, state.coords.freq, log_parameters=True
        )
        n_time = state.coords.time.shape[0]
        return state.with_data(jnp.broadcast_to(profile[None, :], (n_time, profile.shape[0])))



#: Latent names, one per 21cmVAE parameter in PARAMETER_NAMES order.
THETA_NAMES = ("u_fstar", "u_vc", "u_fx", "u_tau", "u_alpha", "u_nu_min", "u_rmfp")


def box_from_unit_normal(u, low, high) -> jax.Array:
    """``theta = low + (high - low) Phi(u)``, the probit map onto the box.

    With ``u ~ Normal(0, 1)`` this puts a uniform prior on ``theta`` over
    the emulator's training box while every sampler works on an unbounded
    variable. A bounded latent with a ``uniform`` prior is not an option
    for the plan's NUTS steps, which do not transform to an unconstrained
    space and were measured to leave the support.
    """
    return jnp.asarray(low) + (jnp.asarray(high) - jnp.asarray(low)) * jax.scipy.stats.norm.cdf(u)


def unit_normal_from_box(theta, low, high) -> jax.Array:
    """Inverse of :func:`box_from_unit_normal`."""
    frac = (jnp.asarray(theta) - jnp.asarray(low)) / (jnp.asarray(high) - jnp.asarray(low))
    return jax.scipy.stats.norm.ppf(frac)


def stack_theta(*values) -> jax.Array:
    """Binding transform: seven standard-normal latents -> the ``(7,)`` log theta.

    The documents declare the parameters as scalars so that every run kind,
    ``nuts`` included, summarises them one number at a time.
    """
    low, high = _box()
    return box_from_unit_normal(jnp.stack([jnp.asarray(v) for v in values]), low, high)


def _box():
    if "box" not in _CACHE:
        with jax.ensure_compile_time_eval():
            emulator = _emulator()
            _CACHE["box"] = (jnp.asarray(emulator.par_min), jnp.asarray(emulator.par_max))
    return _CACHE["box"]


def u_true() -> tuple[float, ...]:
    """The injected parameters as the documents' latents."""
    low, high = _box()
    return tuple(float(v) for v in unit_normal_from_box(jnp.asarray(theta_log_true()), low, high))


class CompressedSignal(AbstractOperator):
    """The 21 cm curve through a compressed likelihood's design matrix.

    Emits ``design @ T21(theta)`` as a ``(1, n_freq)`` array, the model of
    the statistic ``y`` that :mod:`global21cm.collapse` builds by integrating
    the linear foreground out. The row is a whitened statistic, not a
    spectrum; it has the channel count only so that it fits the grid.

    Attributes:
        theta: ``(7,)`` log-parametrised 21cmVAE parameters.
        design: ``(n_freq, n_freq)`` K per K; zero rows beyond the rank.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "global_signal"

    theta: jax.Array
    design: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None or state.coords.time is None:
            raise StateValidationError("CompressedSignal requires coords.time and coords.freq.")
        if state.coords.time.shape[0] != 1 or self.design.shape[1] != state.coords.freq.shape[0]:
            raise StateValidationError(
                f"CompressedSignal: design {self.design.shape} needs a grid of one time "
                f"sample and {self.design.shape[1]} channels, got "
                f"({state.coords.time.shape[0]}, {state.coords.freq.shape[0]})."
            )
        profile = global_signal_kelvin(
            _emulator(), self.theta, state.coords.freq, log_parameters=True
        )
        return state.with_data((self.design @ profile)[None, :])
