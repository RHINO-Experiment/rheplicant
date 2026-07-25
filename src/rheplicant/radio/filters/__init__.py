"""Data filters: linear projections onto physically meaningful subspaces.

All filters share :class:`AbstractLinearFilter` semantics —
``mode="extract"`` keeps the projected component, ``mode="remove"``
subtracts it — and run on *calibrated* data as ordinary Pipeline stages::

    analysis = Pipeline(
        SnapshotOperator(name="raw"),
        ApplyCalibrationOperator(gain=gain_solution),
        SiderealFilter(n_days=7, mode="extract"),      # day-repeating structure
    )

- :class:`SiderealFilter` — the day-repeating (sky-locked) subspace.
- :class:`SkySpaceFilter` — map-make through a linear sky projector and
  reproject (Wiener-like; reuses the forward model's projector).
- :class:`FourierBandFilter` — fringe-rate (time) or delay (freq) bands.
"""

from rheplicant.radio.filters.base import AbstractLinearFilter
from rheplicant.radio.filters.fourier import FourierBandFilter
from rheplicant.radio.filters.sidereal import SiderealFilter
from rheplicant.radio.filters.skyspace import SkySpaceFilter

__all__ = [
    "AbstractLinearFilter",
    "FourierBandFilter",
    "SiderealFilter",
    "SkySpaceFilter",
]
