"""RHINO observation HDF5 files as a recording, and as a State.

Two layers on purpose. :func:`read_rhino_observation` produces
:class:`RhinoObservation`, plain numpy that knows nothing about the signal
graph, so a waterfall can be plotted and a switch log inspected without
constructing anything. :func:`to_state` is the separate seam that places it on
the graph.

The two layers do not use the same time convention, on purpose.
:class:`RhinoObservation` keeps the file's own **unix seconds**;
:func:`to_state` stores **seconds since the first kept sample** and puts the
epoch in ``meta[TIME_EPOCH_META_KEY]``, because ``Coordinates`` stores in
float32 by default and a unix second there is quantised onto a 128 s grid --
argued, and measured, at :func:`to_state`.

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

#: ``state.meta`` key under which :func:`to_state` records the unix second the
#: State's time axis is measured from. ``meta`` rather than ``coords`` on
#: purpose: it is one static number describing the run, not a traced quantity
#: the forward model differentiates through, and ``State``'s own taxonomy puts
#: labels and settings there. The cost is that two observations differ in the
#: jit cache key, which is one recompilation per recording -- the same price
#: ``obs_id`` already pays, and the reason the epoch is a scalar here rather
#: than a per-sample array.
TIME_EPOCH_META_KEY = "time_epoch_unix_s"

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
            **Empty when the reader was called without ``thermistor_columns``**,
            which is not the same statement as a missing label: it says the
            temperatures were never asked for, so the file's thermistor log was
            not read and not judged. Both cases surface as ``KeyError``; the
            caller distinguishes them by what it declared.
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

    **Called only when the reader was given a ``thermistor_columns`` map.** Both
    refusals below are refusals of a whole file, and what they defend does not
    currently reach the signal path: ``to_state`` carries the waterfall, the two
    axes, the switch index and the settling mask, and nothing in ``rheplicant``
    consumes ``thermistor_k``. Running this unconditionally therefore made a
    recording unreadable over a column no operator would have seen. The
    resolution is opt-in, argued in :func:`read_rhino_observation`: a caller who
    declares the map gets every check here, unchanged; a caller who does not
    gets an empty ``thermistor_k`` and a readable waterfall.

    The other direction -- wiring ``thermistor_k`` onto
    ``CalLoadOperator.t_load`` so the defended quantity does reach the signal
    path -- remains open, and is the one worth taking. It is larger than it
    looks: ``t_load`` is a scalar or ``(n_freq,)``, while a load's physical
    temperature is per-SAMPLE, so the operator needs a ``(n_time,)`` case whose
    disambiguation from ``(n_freq,)`` is not free when the two are equal; and
    ``to_state`` returns a State, so wiring an operator from it either changes
    its return type or moves the load temperature into the State for the
    operator to read, which changes that operator's ``requires`` declaration.

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
    thermistor log's actual coverage before reading, record a thermistor log
    that spans the full observation window in the first place, or -- for a
    caller who wanted the waterfall and not the temperatures -- omit
    ``thermistor_columns`` so this function is never reached.
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


def _check_temperature_table(temps_raw: np.ndarray, temp_time: np.ndarray) -> None:
    """Refuse a thermistor log whose table and timestamps do not describe each other.

    Both checks are about the *table*, before any label has been mapped onto a
    column, which is why they run ahead of everything else the reader does: a
    1-D table cannot be addressed by column at all, and a table whose rows do
    not each carry a timestamp cannot be resampled onto the SDR axis. Neither
    is caught later. ``_thermistors_in_kelvin`` documents that it receives a
    table already known 2-D with one row per timestamp; this is where that is
    established.
    """
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


def _check_waterfall(waterfall: np.ndarray, freq_hz: np.ndarray) -> None:
    """Refuse a waterfall that does not lie on the band it is labelled by.

    Takes ``freq_hz`` rather than the raw array, so it runs *after*
    :func:`_frequencies_in_hz`: a file with both a mis-declared ``freq_unit``
    and a bad channel count should be diagnosed by the unit, which is the
    cause a reader can act on. Only the channel axis is checked here -- the row
    axis is checked by the sample mask, which is built from the time axis and
    the switch log rather than from anything in this array.
    """
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


def _keep_samples(array: np.ndarray | None, mask: np.ndarray) -> np.ndarray | None:
    """Apply the leading-drop mask to one per-sample array; pass ``None`` through.

    Named rather than inlined because the invariant is a property of the *set*
    of arrays, not of any one of them: every array indexed by the sample axis
    must be cut by the same mask, or the recording comes back with a per-sample
    label describing a different sample than the waterfall row beside it --
    correctly shaped, and silently misaligned. A per-sample array added later
    (a second ADC monitor, a per-sample RFI flag) is a one-line call here; the
    failure mode of forgetting it is invisible until the leading drop is
    non-zero, which is why ``tests/radio/test_rhino.py`` pins the lengths on a
    file that actually drops samples rather than on the usual one that does not.

    The ``None`` arm is not a convenience: ``max_i_adc``/``max_q_adc`` exist
    only in notebook-written files, so the alternative is the same two-line
    ``if ... is not None`` guard repeated once per optional array.
    """
    return None if array is None else array[mask]


def read_rhino_observation(
    path,
    *,
    freq_unit: str,
    thermistor_columns: Mapping[str, int] | None = None,
    settle_seconds: float = 5.0,
    thermistor_unit: str = "celsius",
) -> RhinoObservation:
    """Read a RHINO observation HDF5 file.

    Args:
        path: the ``.hd5f`` / ``.hdf5`` file.
        freq_unit: ``"Hz"`` or ``"MHz"``. Required -- see the module docstring.
        thermistor_columns: switch label -> column of ``/temperatures``. Omit it
            (or pass ``None``) to skip the thermistor log entirely; see below.
        settle_seconds: samples within this long after a transition are marked
            unsettled. The reference is inconsistent here (5 s in the notebook,
            2 s and 1 s in two rhino-cal functions); 5 s is the most conservative.
        thermistor_unit: ``"celsius"`` (the file's convention) or ``"kelvin"``.
            Unused when ``thermistor_columns`` is omitted.

    **What reaches the signal path, and what does not.** :func:`to_state` places
    ``waterfall`` on ``data``, ``time_s`` on ``coords.time`` (relative -- see
    there), ``freq_hz`` on ``coords.freq``, ``switch_label`` as the integer
    ``coords.extra["receiver_input"]``, and ``settled``, inverted and broadcast,
    on ``aux["flags"]``. Those five are the recording. ``thermistor_k``,
    ``transitions``, ``n_leading_dropped``, ``adc_max_i`` and ``adc_max_q`` are
    **diagnostic**: nothing in ``rheplicant`` consumes them, and a caller that
    wants them reads them off the :class:`RhinoObservation` directly.

    ``thermistor_columns`` is therefore opt-in. ``_thermistors_in_kelvin``
    refuses a thermistor log ending short of the SDR axis, and refuses a
    non-finite reading in a used column; both refusals are argued there and both
    are right for a caller who wants the temperatures. Running them
    unconditionally made a whole recording unreadable over a column nothing
    downstream consumes -- the waterfall, the switch log and the settling mask
    are all intact in such a file, and they are what a forward model needs.

    Omitting the map is not a default guess. The positional convention linking a
    switch label to a ``/temperatures`` column is shared between writer and
    reader with nothing in the file to enforce it, so ``_thermistors_in_kelvin``
    demands a declaration rather than assuming rhino-cal's order; omitting it
    declares that no temperatures are wanted, and ``thermistor_k`` comes back
    empty. When the map IS given, every check it used to make still runs.

    Nothing under ``/temperatures`` is read at all in that case, so a file that
    has no such group -- or a malformed one -- is readable for its waterfall.
    """
    h5py = _require_h5py()
    path = Path(path)
    want_thermistors = thermistor_columns is not None
    temps_raw = temp_time = None
    with h5py.File(path, "r") as f:
        freq_raw = np.asarray(f["sdr/sdr_freqs"][:], dtype=float)
        time_s = np.asarray(f["sdr/sdr_times"][:], dtype=float)
        waterfall = np.asarray(f["sdr/sdr_waterfall"][:], dtype=float)
        switch_time = np.asarray(f["switches/switch_times"][:], dtype=float)
        switch_raw = f["switches/switch_states"][:]
        if want_thermistors:
            temps_raw = np.asarray(f["temperatures/temperatures"][:], dtype=float)
            temp_time = np.asarray(f["temperatures/temperature_times"][:], dtype=float)
        adc_i = np.asarray(f["sdr/max_i_adc"][:], dtype=float) if "sdr/max_i_adc" in f else None
        adc_q = np.asarray(f["sdr/max_q_adc"][:], dtype=float) if "sdr/max_q_adc" in f else None

    # Order is behaviour, not tidiness: a file malformed in two ways at once is
    # diagnosed by whichever of these runs first, and the sequence below reports
    # the cause a reader can act on -- the temperature table before the band it
    # knows nothing about, the declared unit before a channel count that the
    # unit cannot change. The table check keeps its place at the head of that
    # order when it runs at all; it simply does not run for a caller who did not
    # ask for the table.
    if want_thermistors:
        _check_temperature_table(temps_raw, temp_time)
    freq_hz = _frequencies_in_hz(freq_raw, freq_unit)
    _check_waterfall(waterfall, freq_hz)

    switch_label_raw = np.array(
        [s.decode() if isinstance(s, bytes) else str(s) for s in switch_raw]
    )
    per_sample_label, settled, keep, n_dropped = _expand_switch_log(
        time_s, switch_time, switch_label_raw, settle_seconds
    )
    time_s = _keep_samples(time_s, keep)
    waterfall = _keep_samples(waterfall, keep)
    adc_i = _keep_samples(adc_i, keep)
    adc_q = _keep_samples(adc_q, keep)

    return RhinoObservation(
        freq_hz=freq_hz,
        time_s=time_s,
        waterfall=waterfall,
        switch_label=per_sample_label,
        settled=settled,
        thermistor_k=(
            _thermistors_in_kelvin(
                time_s,
                temp_time,
                temps_raw,
                set(per_sample_label.tolist()),
                thermistor_columns,
                thermistor_unit,
            )
            if want_thermistors
            else {}
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

    Returns:
        A State carrying, and only carrying, ``data`` (the waterfall),
        ``coords.time``, ``coords.freq``, ``coords.extra["receiver_input"]``,
        ``aux["flags"]`` and ``meta[TIME_EPOCH_META_KEY]``. Everything else on
        the recording is diagnostic -- see :func:`read_rhino_observation`.

    Raises:
        DataIngestionError: if ``source_order`` repeats a label, if the
            recording switches to a label ``source_order`` does not name, if
            ``obs.settled`` is not boolean, or if the recording holds no
            samples.

    ``coords.time`` is **seconds since the first kept sample**, not unix
    seconds, and ``meta[TIME_EPOCH_META_KEY]`` holds the unix second it is
    measured from -- so ``meta[TIME_EPOCH_META_KEY] + coords.time`` recovers
    ``obs.time_s`` exactly. This is a behaviour change, and it is a fix rather
    than a convenience. :class:`~rheplicant.core.coordinates.Coordinates` stores
    its axes through ``jnp.asarray``, which is float32 unless x64 is enabled,
    and a unix second near 1.75e9 has a float32 resolution of 128 s. Measured on
    six samples at offsets [0, 100, 250, 450, 700, 1000] s from a 1.75e9 epoch,
    with the axis handed over absolute::

        stored offsets  [0, 128, 256, 512, 640, 1024]
        error [s]       [0, +28,  +6, +62,  -60,  +24]

    All six values stay distinct, so no shape, count, dtype or finiteness check
    can see it, while ``BackendOperator``'s chunk timestamps come out wrong by
    tens of seconds and a drifting ``CWCalibrationOperator`` tone lands in the
    wrong channel. Subtracting the epoch *before* the store removes the cause:
    the offsets become small integers, which float32 holds exactly. Detecting it
    afterwards is not possible from the stored values alone, which is why
    ``Coordinates`` refuses such an axis outright rather than repairing it.

    The epoch is the first **kept** sample, not the first sample in the file:
    the leading drop removes samples with no defined switch state, and they are
    not part of the run the State describes.

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
    if obs.time_s.size == 0:
        raise DataIngestionError(
            "the recording holds no samples, so there is no first sample for the "
            "time axis to be measured from. read_rhino_observation refuses an "
            "empty recording ahead of this, so a hand-built RhinoObservation is "
            "the only way here; without this the failure is a bare IndexError "
            "out of obs.time_s[0], naming neither the field nor the reason."
        )

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
    # In numpy float64, deliberately: the subtraction has to happen at the
    # recording's own precision. Handing `obs.time_s` to Coordinates and
    # subtracting afterwards would read the already-rounded values, which is the
    # failure this exists to prevent, not a cheaper way of preventing it.
    epoch = float(obs.time_s[0])
    elapsed = np.asarray(obs.time_s, dtype=np.float64) - obs.time_s[0]
    return State(
        data=jnp.asarray(obs.waterfall),
        coords=Coordinates(
            time=elapsed,
            freq=obs.freq_hz,
            extra={"receiver_input": jnp.asarray(index)},
        ),
        aux={"flags": flags},
        meta={TIME_EPOCH_META_KEY: epoch},
    )
