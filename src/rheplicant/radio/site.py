"""Site + epoch -> Local Sidereal Time, as a units-only adapter.

**Thin adapter, deliberately** -- the same seam :mod:`rheplicant.radio.beams`
established: how LST is computed from a site is limTOD's subject
(``limTOD.simulator.generate_LSTs_deg``); this package's job is to hand it the
run's own conventions and place the result on ``coords.extra["lst_deg"]``.
This seam's whole contribution is the epoch: it arrives as unix seconds
(``meta["time_epoch_unix_s"]``, the convention ``radio/rhino.py`` stores) and
leaves as the UTC string limTOD wants, while ``time_s`` passes straight
through -- limTOD's ``time_list`` is *offsets in seconds from the start*,
which is exactly what ``Coordinates.time`` carries.

Decided as D-C2 (2026-08-09): ``site.lon_deg`` and ``site.alt_m`` were recorded
by the config schema and consumed by nothing; this adapter is what makes them
mean something the package can stand behind.
"""

from __future__ import annotations

import datetime

import numpy as np


def _utc_string(epoch_unix_s: float) -> str:
    """The unix epoch as the ``YYYY-MM-DD HH:MM:SS.ffffff`` UTC string limTOD takes."""
    moment = datetime.datetime.fromtimestamp(float(epoch_unix_s), tz=datetime.UTC)
    return moment.strftime("%Y-%m-%d %H:%M:%S.%f")


def lst_grid_deg(
    *,
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    time_s,
    epoch_unix_s: float,
) -> np.ndarray:
    """LST [deg], ``(n_time,)``, for each sample of a run.

    A pass-through to :func:`limTOD.simulator.generate_LSTs_deg`.

    Args:
        lat_deg: site latitude [deg].
        lon_deg: site longitude [deg], east positive.
        alt_m: site altitude [m].
        time_s: ``(n_time,)`` seconds from the start of the run -- the
            ``coords.time`` convention.
        epoch_unix_s: the absolute start of the run [unix s] -- the
            ``meta["time_epoch_unix_s"]`` convention.
    """
    # Imported here, not at module top: limTOD is a hard dependency, but
    # ``limTOD.simulator`` pulls in astropy's time machinery, which is not a
    # cost every ``import rheplicant.radio`` should pay.
    from limTOD.simulator import generate_LSTs_deg

    return np.asarray(
        generate_LSTs_deg(
            float(lat_deg),
            float(lon_deg),
            float(alt_m),
            np.asarray(time_s, dtype=float),
            start_time_utc=_utc_string(epoch_unix_s),
        )
    )
