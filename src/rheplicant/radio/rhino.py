"""RHINO observation HDF5 files as a recording, and as a State.

Two layers on purpose. :func:`read_rhino_observation` produces
:class:`RhinoObservation`, plain numpy that knows nothing about the signal
graph, so a waterfall can be plotted and a switch log inspected without
constructing anything. :func:`to_state` is the separate seam that places it on
the graph.

**The file does not record its own frequency unit, and its two producers
disagree.** ``rhino-cal``'s ``ObservationHandler.save_to_hdf5`` writes an
astropy Quantity in Hz; the ``RHINO_fully_simulated_calibration`` notebook
writes MHz. The reference reader, ``rhino-cal/gcr/data_processing.py``'s
``DataHandler``, defaults to MHz -- wrong for its own simulator's output, and
silent about it, because the consequence is Gamma interpolated onto a band
10^6 away, which then clamps to constant edge values rather than raising.
``freq_unit`` is therefore required here, with no default, and the declaration
is checked against the file's values.

The schema, as both producers write it::

    /sdr/sdr_freqs          (n_freq,)
    /sdr/sdr_times          (n_time,)            unix seconds
    /sdr/sdr_waterfall      (n_time, n_freq)     raw power
    /sdr/max_i_adc          (n_time,)            notebook-written files only
    /sdr/max_q_adc          (n_time,)            notebook-written files only
    /switches/switch_times  (n_switch,)          unix seconds
    /switches/switch_states (n_switch,)          bytes
    /temperatures/temperatures       (n_temp_time, n_column)   CELSIUS
    /temperatures/temperature_times  (n_temp_time,)            unix seconds

``/aux_sdr`` and ``/obs_config`` are ignored. ``save_to_hdf5`` creates
``/aux_sdr/aux_sdr_waterfall`` with a dtype but no ``data=`` and no ``shape=``,
which makes a *scalar* dataset rather than an array; there is nothing there to
read.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from rheplicant.core.errors import DataIngestionError

_FREQ_UNIT_HZ = {"hz": 1.0, "mhz": 1e6}
#: Wide on purpose: this band's job is to catch a 10^6 unit error, not to
#: police which telescope wrote the file.
_PLAUSIBLE_HZ = (1e6, 1e10)
_KELVIN_OFFSET = 273.15


def _require_h5py():
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - exercised by the import guard
        raise ImportError(
            "rheplicant.radio.rhino needs h5py: pip install \"rheplicant[rhino]\""
        ) from exc
    return h5py


@dataclasses.dataclass(frozen=True)
class RhinoObservation:
    """One RHINO recording, in numpy, in Hz / unix seconds / Kelvin.

    Attributes:
        freq_hz: ``(n_freq,)`` channel frequencies [Hz].
        time_s: ``(n_time,)`` sample times [unix seconds].
        waterfall: ``(n_time, n_freq)`` raw power, arbitrary scale.
        switch_label: ``(n_time,)`` per-sample switch state.
        settled: ``(n_time,)`` bool, **True = usable**. Note the polarity: it is
            the opposite of ``aux["flags"]``, which is True-means-bad.
        thermistor_k: switch label -> ``(n_time,)`` physical temperature [K].
            Two labels may share a column and therefore hold equal arrays.
        transitions: the raw ``(times, labels)`` switch log, kept for diagnosis.
        n_leading_dropped: samples that preceded the first transition and were
            dropped, because they have no defined switch state.
        adc_max_i / adc_max_q: ADC monitors, ``None`` when the file has none.
    """

    freq_hz: np.ndarray
    time_s: np.ndarray
    waterfall: np.ndarray
    switch_label: np.ndarray
    settled: np.ndarray
    thermistor_k: dict[str, np.ndarray]
    transitions: tuple[np.ndarray, np.ndarray]
    n_leading_dropped: int
    adc_max_i: np.ndarray | None
    adc_max_q: np.ndarray | None


def _frequencies_in_hz(raw: np.ndarray, freq_unit: str) -> np.ndarray:
    key = str(freq_unit).strip().lower()
    if key not in _FREQ_UNIT_HZ:
        raise DataIngestionError(
            f"freq_unit must be 'Hz' or 'MHz' (case-insensitive); got "
            f"{freq_unit!r}. The file does not record its own unit and its two "
            "known producers disagree, so there is no default to fall back on."
        )
    freq_hz = raw * _FREQ_UNIT_HZ[key]
    lo, hi = _PLAUSIBLE_HZ
    if freq_hz.min() < lo or freq_hz.max() > hi:
        raise DataIngestionError(
            f"declared freq_unit={freq_unit!r}, which puts this file's channels "
            f"at [{freq_hz.min():.6g}, {freq_hz.max():.6g}] Hz -- outside the "
            f"plausible range [{lo:.0e}, {hi:.0e}] Hz. The file's raw values span "
            f"[{raw.min():.6g}, {raw.max():.6g}]; the other unit is likely right."
        )
    return freq_hz


def _expand_switch_log(
    time_s: np.ndarray,
    switch_time: np.ndarray,
    switch_label: np.ndarray,
    settle_seconds: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-sample labels and a settling mask, plus the leading-drop count.

    ``np.searchsorted`` rather than one boolean mask per switch block: a
    four-hour recording holds thousands of transitions, and the reference's
    per-block masking is O(n_switch * n_time).
    """
    if np.any(np.diff(time_s) <= 0):
        raise DataIngestionError(
            "sdr_times are not strictly ascending. The switch log is a list of "
            "transitions, so assigning a state to each sample assumes an "
            "ordered time axis."
        )
    order = np.argsort(switch_time, kind="stable")
    edges, labels = switch_time[order], switch_label[order]

    index = np.searchsorted(edges, time_s, side="right") - 1
    keep = index >= 0
    n_dropped = int((~keep).sum())
    index = index[keep]

    elapsed = time_s[keep] - edges[index]
    return labels[index], elapsed >= settle_seconds, keep, n_dropped


def read_rhino_observation(
    path,
    *,
    freq_unit: str,
    thermistor_columns: Mapping[str, int],
    settle_seconds: float = 5.0,
    thermistor_unit: str = "celsius",
) -> RhinoObservation:
    """Read a RHINO observation HDF5 file.

    Args:
        path: the ``.hd5f`` / ``.hdf5`` file.
        freq_unit: ``"Hz"`` or ``"MHz"``. Required -- see the module docstring.
        thermistor_columns: switch label -> column of ``/temperatures``. Required.
        settle_seconds: samples within this long after a transition are marked
            unsettled. The reference is inconsistent here (5 s in the notebook,
            2 s and 1 s in two rhino-cal functions); 5 s is the most conservative.
        thermistor_unit: ``"celsius"`` (the file's convention) or ``"kelvin"``.
    """
    h5py = _require_h5py()
    path = Path(path)
    with h5py.File(path, "r") as f:
        freq_raw = np.asarray(f["sdr/sdr_freqs"][:], dtype=float)
        time_s = np.asarray(f["sdr/sdr_times"][:], dtype=float)
        waterfall = np.asarray(f["sdr/sdr_waterfall"][:], dtype=float)
        switch_time = np.asarray(f["switches/switch_times"][:], dtype=float)
        switch_raw = f["switches/switch_states"][:]
        temps_raw = np.asarray(f["temperatures/temperatures"][:], dtype=float)
        temp_time = np.asarray(f["temperatures/temperature_times"][:], dtype=float)
        adc_i = np.asarray(f["sdr/max_i_adc"][:], dtype=float) if "sdr/max_i_adc" in f else None
        adc_q = np.asarray(f["sdr/max_q_adc"][:], dtype=float) if "sdr/max_q_adc" in f else None

    if temps_raw.ndim != 2:
        raise DataIngestionError(
            "/temperatures/temperatures must be 2-D (n_temp_time, n_column); got shape "
            f"{temps_raw.shape}. Thermistor columns are addressed by index, so a flat "
            "array would silently map every switch label onto the same reading."
        )
    if temp_time.shape[0] != temps_raw.shape[0]:
        raise DataIngestionError(
            f"/temperatures/temperatures has {temps_raw.shape[0]} rows but "
            f"/temperatures/temperature_times has {temp_time.shape[0]} entries. The "
            "thermistor log is resampled onto the SDR time axis, which needs each row "
            "to carry a timestamp."
        )

    freq_hz = _frequencies_in_hz(freq_raw, freq_unit)
    switch_label_raw = np.array(
        [s.decode() if isinstance(s, bytes) else str(s) for s in switch_raw]
    )
    per_sample_label, settled, keep, n_dropped = _expand_switch_log(
        time_s, switch_time, switch_label_raw, settle_seconds
    )
    time_s = time_s[keep]
    waterfall = waterfall[keep]
    if adc_i is not None:
        adc_i = adc_i[keep]
    if adc_q is not None:
        adc_q = adc_q[keep]

    return RhinoObservation(
        freq_hz=freq_hz,
        time_s=time_s,
        waterfall=waterfall,
        switch_label=per_sample_label,
        settled=settled,
        thermistor_k={},
        transitions=(switch_time, switch_label_raw),
        n_leading_dropped=n_dropped,
        adc_max_i=adc_i,
        adc_max_q=adc_q,
    )
