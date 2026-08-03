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
from collections.abc import Mapping, Sequence
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import DataIngestionError
from rheplicant.core.state import State
from rheplicant.radio.touchstone import _interp_strict

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
            Keyed by the labels actually present in this file's switch log,
            not by every key in the ``thermistor_columns`` map passed to the
            reader -- a label that map declares but this file never switched
            to has no entry here. ``thermistor_k[label]`` raises ``KeyError``
            for such a label; it does not return ``None`` or an empty array.
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
    if freq_hz.size == 0:
        raise DataIngestionError(
            "/sdr/sdr_freqs is empty. There is no band to check the declared "
            "freq_unit against, and nothing for a reflection coefficient to be "
            "interpolated onto."
        )
    if not np.all(np.isfinite(freq_hz)):
        n_bad = int((~np.isfinite(freq_hz)).sum())
        raise DataIngestionError(
            f"/sdr/sdr_freqs holds {n_bad} non-finite channel(s). NaN compares "
            "False against every bound, so it would pass the plausibility check "
            "below untouched and surface only as a silently wrong interpolation. "
            "An infinity would trip that check, but is rejected here too so the "
            "diagnosis names the bad channel rather than blaming the declared unit."
        )
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
    if time_s.size == 0:
        raise DataIngestionError(
            "/sdr/sdr_times is empty, so the recording holds no samples. An empty "
            "recording is never a legitimate read here: it would travel on as a "
            "well-formed State that quietly contributes nothing to a calibration."
        )
    if not np.all(np.isfinite(time_s)):
        n_bad = int((~np.isfinite(time_s)).sum())
        raise DataIngestionError(
            f"/sdr/sdr_times holds {n_bad} non-finite timestamp(s). np.diff across a "
            "NaN yields NaN and `nan <= 0` is False, so the strictly-ascending check "
            "below would pass it; searchsorted then sorts NaN to the end, labelling "
            "that sample with the last switch state and breaking the contiguity the "
            "leading drop depends on."
        )
    ascending = np.diff(time_s) > 0
    if not np.all(ascending):
        bad = int(np.argmin(ascending)) + 1
        raise DataIngestionError(
            f"sdr_times are not strictly ascending (first offender at sample {bad}, "
            "0-based). The switch log is a list of transitions, so assigning a state "
            "to each sample assumes an ordered time axis. Naming the offender matches "
            "the two sibling checks in touchstone.py, which report theirs."
        )
    if switch_time.size == 0:
        raise DataIngestionError(
            "/switches/switch_times is empty, so no sample has a defined switch "
            "state. Every sample would be dropped as leading, leaving an empty "
            "recording that still looks well-formed."
        )
    if switch_label.shape[0] != switch_time.shape[0]:
        raise DataIngestionError(
            f"/switches/switch_times has {switch_time.shape[0]} entries but "
            f"/switches/switch_states has {switch_label.shape[0]}. Sorting the log "
            "indexes the labels by the times' order, so a longer label array loses "
            "its tail silently and `transitions` is handed back as a mismatched pair."
        )
    if not np.all(np.isfinite(switch_time)):
        n_bad = int((~np.isfinite(switch_time)).sum())
        raise DataIngestionError(
            f"/switches/switch_times holds {n_bad} non-finite timestamp(s). This is "
            "the same mechanism the sdr_times guard above names, on the other array "
            "feeding the same searchsorted: NaN sorts to the end, so the corrupted "
            "transition lands after every sample and can never be selected. Its "
            "label then vanishes from the recording and its samples fold into the "
            "neighbouring states -- no exception, nothing dropped, and a switch "
            "position silently missing."
        )
    order = np.argsort(switch_time, kind="stable")
    edges, labels = switch_time[order], switch_label[order]

    index = np.searchsorted(edges, time_s, side="right") - 1
    keep = index >= 0
    if not keep.any():
        raise DataIngestionError(
            f"all {time_s.size} samples precede the first switch transition at "
            f"{edges[0]:.6g}, so none of them has a defined switch state. Check that "
            "the switch log and the SDR share a time origin."
        )
    # `keep` is a contiguous suffix, so every dropped sample really is *leading*
    # and `n_leading_dropped` means what it says. That rests on three preconditions,
    # all enforced above: `edges` is sorted, and `time_s` is both finite and strictly
    # ascending -- together those make `index` non-decreasing. Finiteness is the easy
    # one to overlook: searchsorted sorts NaN to the end, so a single NaN timestamp
    # punches an interior hole in `keep` and the count silently means something else.
    assert np.all(np.diff(keep.astype(np.int8)) >= 0), "keep is not a contiguous suffix"

    n_dropped = int((~keep).sum())
    index = index[keep]

    elapsed = time_s[keep] - edges[index]
    return labels[index], elapsed >= settle_seconds, keep, n_dropped


def _thermistors_in_kelvin(
    time_s: np.ndarray,
    temp_time: np.ndarray,
    temps: np.ndarray,
    labels_present: set[str],
    thermistor_columns: Mapping[str, int],
    thermistor_unit: str,
) -> dict[str, np.ndarray]:
    """Map switch labels onto thermistor columns, in Kelvin, on ``time_s``.

    The reference (``rhino-cal/gcr/data_processing.py``) takes
    ``heated_load_index=1, ambient_load_index=0`` and routes every state
    except ``heated_load`` to the ambient column. Those indices are the
    positional order of ``save_to_hdf5``'s ``save_temps`` argument -- a
    convention shared between writer and reader with nothing in the file to
    enforce it, so a file written with a different order reads back with hot
    and ambient swapped and nothing raises. Requiring the map here makes that
    a declaration, not a default.

    Built for ``labels_present``, not for every key in ``thermistor_columns``:
    a caller may reasonably hold one shared column map covering more loads
    than a given file's switch log uses, so a label the map declares but this
    file never switched to simply has no entry -- see
    :attr:`RhinoObservation.thermistor_k`.

    ``temps`` arrives already checked 2-D with one row per ``temp_time``
    entry (``read_rhino_observation`` does that, ahead of this call, for
    every temperature reading regardless of whether any label maps onto it)
    -- nothing here re-derives that.

    A non-finite reading (NaN or +/-inf) in a used column raises rather than
    propagating. ``_interp_strict`` only guards its *x*-axis (``temp_time``)
    against NaN; the values being interpolated (a thermistor column) get no
    such guard from it, and a linear interpolant spreads one bad row into
    every sample whose bracketing interval touches it -- a dropout wider than
    the dropout itself. Left unchecked, that value flows into T_sys and then
    the noise-wave solve, where nothing left points back at the thermistor
    log it came from. Only the columns a present label actually uses are
    checked, matching the labels-present policy above.

    Real thermistor logs do drop samples, and always raising makes a whole
    file unreadable for one bad reading. If that turns out to be routine on
    real RHINO recordings, the fix is an explicit caller policy -- e.g. an
    ``on_nonfinite: Literal["raise", "flag"] = "raise"`` argument, where
    ``"flag"`` would leave the non-finite samples in the interpolated output
    for the caller to mask, the way ``settled`` already lets a caller mask
    unsettled samples -- not silently interpolating through them by default.

    A second, distinct way this raises: ``_interp_strict``'s range check is
    now strict enough (a tolerance on the order of a microsecond on a
    unix-epoch axis, not the multi-second one it replaced) that a real
    thermistor log stopping even a millisecond short of the last SDR sample
    makes the *whole file* unreadable, not just that one column. That refusal
    is correct -- clamping to the edge reading is exactly the silent failure
    the tolerance fix exists to prevent -- but it is a different failure mode
    from the non-finite one above, and ``on_nonfinite`` would not help here:
    that argument is about a bad value inside the covered range, not about a
    thermistor log that does not cover the SDR axis at all. The remedy is on
    the caller's side, not this function's: trim the SDR time axis to the
    thermistor log's actual coverage before reading, or record a thermistor
    log that spans the full observation window in the first place.
    """
    unit = str(thermistor_unit).strip().lower()
    if unit == "celsius":
        temps_k = temps + _KELVIN_OFFSET
    elif unit == "kelvin":
        temps_k = temps
    else:
        raise DataIngestionError(
            f"thermistor_unit must be 'celsius' or 'kelvin'; got {thermistor_unit!r}."
        )

    missing = sorted(labels_present - set(thermistor_columns))
    if missing:
        raise DataIngestionError(
            f"thermistor_columns has no entry for {missing}, which appear in the "
            f"switch log. Declared: {sorted(thermistor_columns)}."
        )

    out: dict[str, np.ndarray] = {}
    for label in sorted(labels_present):
        column = thermistor_columns[label]
        if not 0 <= column < temps_k.shape[1]:
            raise DataIngestionError(
                f"thermistor_columns[{label!r}] = {column}, but "
                f"/temperatures/temperatures has {temps_k.shape[1]} columns."
            )
        column_values = temps_k[:, column]
        if not np.all(np.isfinite(column_values)):
            bad_index = int(np.flatnonzero(~np.isfinite(column_values))[0])
            raise DataIngestionError(
                f"thermistor_columns[{label!r}] = {column}: /temperatures/temperatures "
                f"has a non-finite reading at row {bad_index} (0-based, indexing "
                "temperature_times). A linear interpolant spreads one bad row into "
                "every sample whose bracketing interval touches it -- a dropout wider "
                "than the dropout itself -- and by the time the value reaches T_sys "
                "and the noise-wave solve there is nothing left pointing back at the "
                "thermistor log it came from."
            )
        out[label] = _interp_strict(
            time_s,
            temp_time,
            column_values,
            what=f"thermistor column {column} for {label!r}",
        )
    return out


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
    if waterfall.ndim != 2:
        raise DataIngestionError(
            "/sdr/sdr_waterfall must be 2-D (n_time, n_freq); got shape "
            f"{waterfall.shape}."
        )
    if waterfall.shape[1] != freq_hz.size:
        raise DataIngestionError(
            f"/sdr/sdr_waterfall has {waterfall.shape[1]} channels but /sdr/sdr_freqs "
            f"has {freq_hz.size}. The row axis is checked by the sample mask, but a "
            "channel-axis mismatch in either direction would survive as a waterfall "
            "silently misaligned with the band it is labelled by."
        )
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
        thermistor_k=_thermistors_in_kelvin(
            time_s,
            temp_time,
            temps_raw,
            set(per_sample_label.tolist()),
            thermistor_columns,
            thermistor_unit,
        ),
        transitions=(switch_time, switch_label_raw),
        n_leading_dropped=n_dropped,
        adc_max_i=adc_i,
        adc_max_q=adc_q,
    )


def to_state(obs: RhinoObservation, *, source_order: Sequence[str]) -> State:
    """Place a recording on the signal graph.

    Args:
        obs: a recording.
        source_order: switch labels in the graph's in-edge order. Read it off
            the assembled twin -- ``assembly["receiver_input"].names`` -- rather
            than assuming it: it is the order ``NoiseWaveOperator``'s
            ``gamma_src`` rows must match, and a transposition there is
            shape-legal and costs tens of kelvin.

    Raises:
        DataIngestionError: if ``source_order`` repeats a label, if the
            recording switches to a label ``source_order`` does not name, or
            if ``obs.settled`` is not boolean.

    The settling mask is **inverted** on the way in. ``aux["flags"]`` is
    True-means-flagged (``radio/backend/flagging.py``, and ``FlaggedNoise``
    consumes it that way) while ``settled`` is True-means-usable. Getting this
    backwards yields a finite, correctly-shaped result that discards every
    good sample and keeps every transient -- nothing about the shape or dtype
    would reveal it.

    ``aux["flags"]`` is also **broadcast to ``(n_time, n_freq)``**, matching
    ``state.data``, even though settling is inherently a per-time quantity --
    every channel of an unsettled sample is unsettled, so the broadcast
    changes nothing about what is being said. The shape is not this
    function's choice; it is set by every consumer: ``FlaggedNoise.std``
    (``inference/noise.py``) raises if ``flags`` disagrees in shape with the
    prediction it masks, ``SkySpaceFilter`` (``radio/filters/skyspace.py``)
    multiplies ``1 - flags`` elementwise against the data, and both
    ``FlaggingOperator`` and ``MomentRFIFlaggingOperator``
    (``radio/backend/flagging.py``) produce and expect ``(n_time, n_freq)``.
    ``obs.settled`` itself stays ``(n_time,)`` on :class:`RhinoObservation`,
    for a caller who wants the per-time form directly.
    """
    order = tuple(source_order)
    duplicates = sorted({label for label in order if order.count(label) > 1})
    if duplicates:
        raise DataIngestionError(
            f"source_order names {duplicates} more than once: {list(order)}. A "
            "repeated label would collapse two switch positions onto the same "
            "index -- shape-legal, and silently wrong."
        )

    lookup = {label: i for i, label in enumerate(order)}
    unknown = sorted(set(obs.switch_label.tolist()) - set(lookup))
    if unknown:
        raise DataIngestionError(
            f"the recording switches to {unknown}, which source_order does not "
            f"name (it lists {list(order)}). Deferring this makes "
            "SwitchCycle.gather return NaN much later, where the cause is no "
            "longer visible."
        )

    if obs.settled.dtype != np.bool_:
        raise DataIngestionError(
            f"obs.settled must be boolean; got dtype {obs.settled.dtype}. `~` on "
            "a non-bool array is a bitwise complement, not a logical negation, "
            "and the result would not recover the intended flag polarity."
        )

    index = np.array([lookup[label] for label in obs.switch_label], dtype=int)
    flags = jnp.broadcast_to(jnp.asarray(~obs.settled)[:, None], obs.waterfall.shape)
    return State(
        data=jnp.asarray(obs.waterfall),
        coords=Coordinates(
            time=obs.time_s,
            freq=obs.freq_hz,
            extra={"receiver_input": jnp.asarray(index)},
        ),
        aux={"flags": flags},
    )
