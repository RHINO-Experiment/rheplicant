"""BeamSpillOperator — the horizon split of a beam that does not stop at it.

A real beam does not end at the horizon. RHINO's horn puts 1-3 % of its solid
angle below it, and that part is looking at ground, not sky. The antenna
temperature is therefore a weighted mixture::

    T_collected = f_sky * <T_sky>_masked + (1 - f_sky) * T_ground

with ``f_sky`` the above-horizon beam fraction and ``<T_sky>_masked`` the beam
average over the VISIBLE sky — which is exactly what a projector with
``horizon_mask=True, normalize_beam=True`` returns.

Both halves live in one operator ON PURPOSE. Split across two objects — a
weight somewhere and a ``GroundPickupOperator`` somewhere else — the two
numbers can drift apart, and a sky branch weighted by ``f`` while the ground
branch uses ``1 - f'`` is a bias no shape check can see. Here the weights sum
to one by construction.

``f_sky`` IS the horizon mask, measured. Get it from
:meth:`~rheplicant.radio.sky.driftscan.DriftScanProjector.horizon_fraction`,
which computes it from the same band-limited masked beam the forward model
uses, or let :meth:`BeamSpillOperator.from_projector` do both at once. Guessing
it, or taking an ideal pixel-space horizon cut instead, weights one beam by
another beam's solid angle.

Placement (graph v1.4). ``beam_spill`` is the trunk stage of the ASTRO branch:
``beam | observed_astro_sky -> astro_ant_sum -> beam_spill -> t_ant_sum``. The
split applies to the thing that genuinely is a beam integral over the celestial
sphere and to nothing else. The other ``t_ant_sum`` leaves — ``ground_pickup``,
``atmosphere``, ``t_sys_extra`` — are *effective* temperatures by D13's
construction, already carrying whatever beam weighting their author intended;
running them through the split would weight them twice, and would attenuate a
ground term that IS the below-horizon share.

Relation to the two other losses on this path, none of which substitutes for
another (they compose, in this order):

* ``beam_spill`` — what the beam is pointed at. Mixing, no loss: an isotropic
  sky at ``T`` with ground also at ``T`` still gives ``T``.
* :class:`~rheplicant.radio.instrument.antenna_loss.AntennaLossOperator` —
  ohmic dissipation inside the antenna, acting on the whole ``t_ant_sum``.
* ``c_s = (1 - |Gamma|^2)|F|^2`` inside
  :class:`~rheplicant.radio.instrument.noise_wave.NoiseWaveOperator` — the
  impedance mismatch at the receiver input.

The first two share an arithmetic form, ``a * x + (1 - a) * b``, and are
deliberately NOT one operator: they carry independent physical parameters at
different points on the path, and merging them would make an efficiency and a
spill fraction indistinguishable in a fit.

USING IT WITH ``GroundPickupOperator``: this operator already supplies the
below-horizon ground term, so a ``GroundPickupOperator`` alongside it adds a
SECOND, additional one. That is legitimate only if it stands for something else
(a nearby building, a ground screen's own emission); as a model of the same beam
spill it double-counts. Nothing enforces this — both are legal leaves — so it is
the caller's call to make deliberately.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class BeamSpillOperator(AbstractOperator):
    """Mix the horizon-masked sky with the ground the rest of the beam sees.

    Attributes:
        sky_fraction: ``f_sky``, the above-horizon beam fraction; differentiable
            scalar or ``(n_freq,)`` spectrum. ``1.0`` means no spill and makes
            the operator an exact identity.
        t_ground: brightness temperature seen below the horizon [K];
            differentiable scalar or ``(n_freq,)``.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "beam_spill"

    sky_fraction: jax.Array = eqx.field(converter=jnp.asarray)
    t_ground: jax.Array = eqx.field(converter=jnp.asarray)

    @classmethod
    def from_projector(cls, projector, *, t_ground) -> "BeamSpillOperator":
        """Take ``f_sky`` from the projector that will supply the sky.

        The one call that cannot get the weight and the sky average out of step,
        because it reads the fraction off the same beam.

        Args:
            projector: a projector exposing ``horizon_fraction()`` — today
                :class:`~rheplicant.radio.sky.driftscan.DriftScanProjector`,
                on which the horizon cut is a fixed property of the pointing.
            t_ground: brightness temperature below the horizon [K].

        Raises:
            StateValidationError: if the projector has no ``horizon_fraction``.
        """
        if not hasattr(projector, "horizon_fraction"):
            raise StateValidationError(
                f"{type(projector).__name__} does not expose horizon_fraction(), "
                "so the above-horizon beam fraction cannot be read off it. Only "
                "DriftScanProjector defines the cut today (a fixed pointing makes "
                "it time-independent); for a scanning strategy the fraction "
                "varies per sample and needs a per-sample sky_fraction."
            )
        return cls(sky_fraction=projector.horizon_fraction(), t_ground=t_ground)

    def __check_init__(self):
        for name, value in (("sky_fraction", self.sky_fraction),
                            ("t_ground", self.t_ground)):
            if value.ndim > 1:
                raise StateValidationError(
                    f"{name} must be scalar or (n_freq,), got ndim={value.ndim}."
                )
        if (
            self.sky_fraction.ndim == 1
            and self.t_ground.ndim == 1
            and self.sky_fraction.shape != self.t_ground.shape
        ):
            raise StateValidationError(
                f"sky_fraction has {self.sky_fraction.shape[0]} channels but "
                f"t_ground has {self.t_ground.shape[0]}."
            )

    def _check_channels(self, name: str, value: jax.Array, n_freq: int) -> None:
        if value.ndim == 1 and value.shape[0] != n_freq:
            raise StateValidationError(
                f"{name} has {value.shape[0]} channels but data has {n_freq}. "
                "Broadcasting would silently apply the wrong split per channel."
            )

    def __call__(self, state: State) -> State:
        if state.data is None or jnp.asarray(state.data).ndim != 2:
            got = None if state.data is None else jnp.asarray(state.data).shape
            raise StateValidationError(
                f"BeamSpillOperator expects (n_time, n_freq) data; got {got}."
            )
        n_freq = state.data.shape[1]
        self._check_channels("sky_fraction", self.sky_fraction, n_freq)
        self._check_channels("t_ground", self.t_ground, n_freq)
        return state.with_data(
            self.sky_fraction * state.data
            + (1.0 - self.sky_fraction) * self.t_ground
        )
