"""The RHINO observation reader's own surface.

The file schema is fixed by two producers that do not agree with each other --
rhino-cal's ObservationHandler.save_to_hdf5 writes frequencies in Hz, the
RHINO_fully_simulated_calibration notebook writes them in MHz, and the file
records neither. Most of what is tested here is that disagreement being caught
rather than absorbed.
"""

import dataclasses

import numpy as np
import pytest

from rheplicant.core.errors import DataIngestionError

h5py = pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")

from rheplicant.radio.rhino import (  # noqa: E402
    RhinoObservation,
    read_rhino_observation,
    to_state,
)

FREQ_MHZ = np.array([60.0, 70.0, 80.0])
#: Four samples per switch position, 1 s apart, three positions.
TIME_S = np.arange(0.0, 12.0, 1.0) + 1000.0
SWITCH_TIMES = np.array([1000.0, 1004.0, 1008.0])
SWITCH_STATES = [b"antenna", b"internal_load", b"heated_load"]


def make_file(
    path,
    *,
    freqs=FREQ_MHZ,
    times=TIME_S,
    temps=None,
    temp_times=None,
    with_adc=False,
    switch_times=SWITCH_TIMES,
    switch_states=SWITCH_STATES,
    waterfall=None,
):
    n_time, n_freq = len(times), len(freqs)
    if temps is None:
        # column 0 = ambient (20 C), column 1 = hot (100 C)
        temps = np.stack(
            [np.full(n_time, 20.0), np.full(n_time, 100.0)], axis=1
        )
    if temp_times is None:
        # Independent by default only in name -- the thermistor log normally
        # shares the SDR's own time axis. A caller that wants to test a
        # thermistor log spanning something else passes temp_times directly.
        temp_times = times
    if waterfall is None:
        waterfall = np.arange(n_time * n_freq, dtype=float).reshape(n_time, n_freq)
    with h5py.File(path, "w") as f:
        sdr = f.create_group("sdr")
        sdr.create_dataset("sdr_freqs", data=freqs)
        sdr.create_dataset("sdr_times", data=times)
        sdr.create_dataset("sdr_waterfall", data=waterfall)
        if with_adc:
            sdr.create_dataset("max_i_adc", data=np.zeros(n_time))
            sdr.create_dataset("max_q_adc", data=np.ones(n_time))
        sw = f.create_group("switches")
        sw.create_dataset("switch_times", data=np.asarray(switch_times, dtype=float))
        sw.create_dataset("switch_states", data=np.array(switch_states, dtype="S16"))
        tg = f.create_group("temperatures")
        tg.create_dataset("temperature_times", data=np.asarray(temp_times, dtype=float))
        tg.create_dataset("temperatures", data=temps)
    return path


COLUMNS = {"antenna": 0, "internal_load": 0, "heated_load": 1}


def test_reads_the_waterfall_and_converts_mhz_to_hz(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
    )
    assert isinstance(obs, RhinoObservation)
    np.testing.assert_allclose(obs.freq_hz, [60e6, 70e6, 80e6])
    assert obs.waterfall.shape == (len(TIME_S), 3)
    np.testing.assert_allclose(obs.time_s, TIME_S)


def test_declaring_the_wrong_frequency_unit_raises_with_the_actual_range(tmp_path):
    with pytest.raises(DataIngestionError, match="plausible range") as excinfo:
        read_rhino_observation(
            make_file(tmp_path / "hz.hd5f", freqs=FREQ_MHZ * 1e6),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )
    # The message must show what the declaration implies (8e+13 Hz) so the
    # reader can see at a glance that the other unit is the right one.
    assert "8e+13" in str(excinfo.value)


def test_an_unknown_frequency_unit_raises(tmp_path):
    with pytest.raises(DataIngestionError, match="freq_unit"):
        read_rhino_observation(
            make_file(tmp_path / "obs.hd5f"), freq_unit="GHz", thermistor_columns=COLUMNS
        )


def test_a_flat_temperature_array_raises(tmp_path):
    # A 1-D /temperatures cannot be addressed by column, so every switch label
    # would silently resolve to the same reading.
    with pytest.raises(DataIngestionError, match="2-D"):
        read_rhino_observation(
            make_file(tmp_path / "flat.hd5f", temps=np.full(len(TIME_S), 20.0)),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_temperatures_without_a_timestamp_each_raise(tmp_path):
    short = np.stack([np.full(5, 20.0), np.full(5, 100.0)], axis=1)
    with pytest.raises(DataIngestionError, match="temperature_times"):
        read_rhino_observation(
            make_file(tmp_path / "short.hd5f", temps=short),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_switch_labels_are_expanded_per_sample(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    assert obs.switch_label.shape == (12,)
    assert list(obs.switch_label[:4]) == ["antenna"] * 4
    assert list(obs.switch_label[4:8]) == ["internal_load"] * 4
    assert list(obs.switch_label[8:]) == ["heated_load"] * 4


def test_settled_is_false_for_settle_seconds_after_each_transition(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=2.0,
    )
    # transitions at t = 1000, 1004, 1008; samples are 1 s apart from t = 1000.
    expected = np.array([False, False, True, True] * 3)
    np.testing.assert_array_equal(obs.settled, expected)


def test_samples_before_the_first_transition_are_dropped_and_counted(tmp_path):
    early = np.concatenate([[998.0, 999.0], TIME_S])
    obs = read_rhino_observation(
        make_file(tmp_path / "early.hd5f", times=early),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    assert obs.n_leading_dropped == 2
    assert obs.time_s.shape == (12,)
    assert obs.waterfall.shape == (12, 3)
    assert obs.switch_label.shape == (12,)
    assert obs.settled.shape == (12,)


def test_non_finite_frequencies_raise(tmp_path):
    # NaN compares False against both bounds, so it slips through a bare
    # min()/max() plausibility test and reaches the interpolation intact.
    with pytest.raises(DataIngestionError, match="non-finite"):
        read_rhino_observation(
            make_file(tmp_path / "nan.hd5f", freqs=np.array([60.0, np.nan, 80.0])),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_an_empty_frequency_axis_raises(tmp_path):
    # min() of an empty array is a bare numpy ValueError, which callers
    # catching DataIngestionError would not catch.
    with pytest.raises(DataIngestionError, match="empty"):
        read_rhino_observation(
            make_file(tmp_path / "nofreq.hd5f", freqs=np.array([])),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_an_empty_switch_log_raises_rather_than_dropping_every_sample(tmp_path):
    with pytest.raises(DataIngestionError, match="switch_times"):
        read_rhino_observation(
            make_file(
                tmp_path / "noswitch.hd5f", switch_times=np.array([]), switch_states=[]
            ),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_non_finite_sample_times_raise(tmp_path):
    # np.diff across a NaN yields NaN and `nan <= 0` is False, so the strictly
    # ascending guard passes it through -- the same mechanism the frequency
    # axis was hardened against.
    times = np.array([np.nan, 900.0, 1000.0, 1001.0])
    with pytest.raises(DataIngestionError, match="non-finite"):
        read_rhino_observation(
            make_file(tmp_path / "nantime.hd5f", times=times),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_a_nan_timestamp_cannot_punch_an_interior_hole_in_the_leading_drop(tmp_path):
    # Regression: this exact axis produced keep = [1, 0, 1, 1] -- a dropped
    # sample in the *interior*, counted as leading. `keep` must stay a suffix.
    times = np.array([np.nan, 900.0, 1000.0, 1001.0])
    with pytest.raises(DataIngestionError):
        read_rhino_observation(
            make_file(tmp_path / "hole.hd5f", times=times),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
            settle_seconds=0.0,
        )


@pytest.mark.parametrize("n_channels", [2, 7])
def test_waterfall_channel_count_must_match_sdr_freqs(tmp_path, n_channels):
    # Both directions leak: neither too few nor too many channels was caught.
    bad = np.zeros((len(TIME_S), n_channels), dtype=float)
    with pytest.raises(DataIngestionError, match="channel"):
        read_rhino_observation(
            make_file(tmp_path / f"w{n_channels}.hd5f", waterfall=bad),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_more_switch_states_than_switch_times_raise(tmp_path):
    # Silently truncated the third label and handed back a mismatched
    # (2,)/(3,) pair as the public `transitions` field.
    with pytest.raises(DataIngestionError, match="switch_states"):
        read_rhino_observation(
            make_file(
                tmp_path / "extra.hd5f",
                switch_times=np.array([1000.0, 1004.0]),
                switch_states=SWITCH_STATES,
            ),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_an_empty_time_axis_raises(tmp_path):
    with pytest.raises(DataIngestionError, match="no samples"):
        read_rhino_observation(
            make_file(tmp_path / "notime.hd5f", times=np.array([])),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_every_sample_preceding_the_first_transition_raises(tmp_path):
    # Would otherwise return a well-formed but empty recording with
    # n_leading_dropped == n_time.
    with pytest.raises(DataIngestionError, match="precede"):
        read_rhino_observation(
            make_file(tmp_path / "allearly.hd5f", times=np.array([990.0, 991.0, 992.0])),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_non_ascending_sample_times_raise(tmp_path):
    scrambled = TIME_S.copy()
    scrambled[5], scrambled[6] = scrambled[6], scrambled[5]
    with pytest.raises(DataIngestionError, match="ascending"):
        read_rhino_observation(
            make_file(tmp_path / "scrambled.hd5f", times=scrambled),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )


def test_thermistor_columns_are_mapped_and_converted_to_kelvin(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    assert set(obs.thermistor_k) == {"antenna", "internal_load", "heated_load"}
    np.testing.assert_allclose(obs.thermistor_k["internal_load"], 20.0 + 273.15)
    np.testing.assert_allclose(obs.thermistor_k["heated_load"], 100.0 + 273.15)
    # Two labels sharing column 0 hold equal arrays -- that is how the
    # reference's "ambient covers everything but the hot load" rule is
    # written down once the caller has to state it.
    np.testing.assert_allclose(
        obs.thermistor_k["antenna"], obs.thermistor_k["internal_load"]
    )


def test_kelvin_input_is_not_offset_again(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        thermistor_unit="kelvin",
        settle_seconds=0.0,
    )
    np.testing.assert_allclose(obs.thermistor_k["internal_load"], 20.0)


def test_thermistor_unit_is_case_and_whitespace_insensitive(tmp_path):
    # thermistor_unit goes through the same .strip().lower() normalisation as
    # freq_unit, but every other test here passes it already-lowercase, so
    # nothing would fail if that normalisation were deleted. Pin it directly,
    # the way the freq_unit tests already pin theirs by passing "MHz".
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        thermistor_unit="  CeLsIuS  ",
        settle_seconds=0.0,
    )
    np.testing.assert_allclose(obs.thermistor_k["internal_load"], 20.0 + 273.15)


def test_an_unknown_thermistor_unit_raises(tmp_path):
    with pytest.raises(DataIngestionError, match="thermistor_unit"):
        read_rhino_observation(
            make_file(tmp_path / "obs.hd5f"),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
            thermistor_unit="fahrenheit",
        )


def test_a_switch_label_with_no_column_raises_and_names_it(tmp_path):
    with pytest.raises(DataIngestionError, match="heated_load"):
        read_rhino_observation(
            make_file(tmp_path / "obs.hd5f"),
            freq_unit="MHz",
            thermistor_columns={"antenna": 0, "internal_load": 0},
        )


def test_a_thermistor_column_index_out_of_range_raises(tmp_path):
    # Column 5 does not exist -- /temperatures/temperatures has 2 columns in
    # the default fixture. Out of range must raise, not index silently into
    # whatever numpy does with it (wraparound on a negative index; IndexError
    # with no DataIngestionError wrapper on a positive one).
    with pytest.raises(DataIngestionError, match="antenna"):
        read_rhino_observation(
            make_file(tmp_path / "obs.hd5f"),
            freq_unit="MHz",
            thermistor_columns={"antenna": 5, "internal_load": 0, "heated_load": 1},
            settle_seconds=0.0,
        )


def test_a_non_default_column_order_reads_back_correctly(tmp_path):
    # Hot in column 0, ambient in column 1 -- the reverse of what
    # save_to_hdf5's default save_temps order produces. The reference's magic
    # indices would silently swap them; a declared map cannot.
    swapped = np.stack(
        [np.full(len(TIME_S), 100.0), np.full(len(TIME_S), 20.0)], axis=1
    )
    obs = read_rhino_observation(
        make_file(tmp_path / "swapped.hd5f", temps=swapped),
        freq_unit="MHz",
        thermistor_columns={"antenna": 1, "internal_load": 1, "heated_load": 0},
        settle_seconds=0.0,
    )
    np.testing.assert_allclose(obs.thermistor_k["heated_load"], 100.0 + 273.15)
    np.testing.assert_allclose(obs.thermistor_k["internal_load"], 20.0 + 273.15)


def test_every_time_axis_array_keeps_the_same_length_after_the_leading_drop(tmp_path):
    early = np.concatenate([[998.0, 999.0], TIME_S])
    obs = read_rhino_observation(
        make_file(tmp_path / "early.hd5f", times=early),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    n = len(obs.time_s)
    assert n == 12
    assert obs.waterfall.shape[0] == n
    assert obs.switch_label.shape == (n,)
    assert obs.settled.shape == (n,)
    for label, series in obs.thermistor_k.items():
        assert series.shape == (n,), label


def test_adc_monitors_are_passed_through_when_present(tmp_path):
    without = read_rhino_observation(
        make_file(tmp_path / "a.hd5f"), freq_unit="MHz", thermistor_columns=COLUMNS
    )
    assert without.adc_max_i is None and without.adc_max_q is None

    with_adc = read_rhino_observation(
        make_file(tmp_path / "b.hd5f", with_adc=True),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
    )
    assert with_adc.adc_max_i.shape == (12,)
    np.testing.assert_allclose(with_adc.adc_max_q, 1.0)


def test_a_declared_column_never_switched_to_has_no_thermistor_k_entry(tmp_path):
    # thermistor_columns may cover more loads than a given file's switch log
    # uses -- see the rhino.py module docstring. A label that is declared but
    # never appears in this file's switch log gets no entry; that is not an
    # error, since a caller may hold one shared column map across many files.
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns={**COLUMNS, "spare_load": 1},
        settle_seconds=0.0,
    )
    assert "spare_load" not in obs.thermistor_k
    assert set(obs.thermistor_k) == {"antenna", "internal_load", "heated_load"}


def test_thermistor_log_shorter_than_the_sdr_axis_raises_rather_than_clamping(tmp_path):
    # Reproduces the exact bug _interp_strict's tolerance was fixed for. On a
    # real unix-epoch-scale axis, the old tol = 1e-9 * max(span, abs(hi))
    # reduces to ~1e-9 * abs(hi) once abs(hi) dominates the (short) span --
    # about 1.75 s of slack independent of window length -- so a target 1.5 s
    # past the sampled range would have been silently clamped instead of
    # raising. The new two-term tolerance is on the order of a microsecond on
    # this axis, so the same 1.5 s gap must raise instead.
    base = 1_735_000_000.0
    times = np.array([base, base + 11.5])
    obs_path = make_file(
        tmp_path / "short_thermistor_log.hd5f",
        times=times,
        temps=np.array([[20.0], [20.0]]),
        temp_times=np.array([base, base + 10.0]),
        switch_times=np.array([base]),
        switch_states=[b"antenna"],
    )
    with pytest.raises(DataIngestionError, match="outside"):
        read_rhino_observation(
            obs_path,
            freq_unit="MHz",
            thermistor_columns={"antenna": 0},
            settle_seconds=0.0,
        )


def test_a_nan_thermistor_reading_raises_and_names_the_source(tmp_path):
    # _interp_strict only guards its x-axis (temp_time) against NaN -- the
    # column being interpolated gets no such guard from it. A NaN here must
    # not propagate: a linear interpolant would spread it into every sample
    # whose bracketing interval touches it, and by the time it reached T_sys
    # nothing would point back at the thermistor log it came from.
    temps = np.stack(
        [np.full(len(TIME_S), 20.0), np.full(len(TIME_S), 100.0)], axis=1
    )
    temps[2, 1] = np.nan  # heated_load's column (1), source row 2
    with pytest.raises(DataIngestionError, match="heated_load") as excinfo:
        read_rhino_observation(
            make_file(tmp_path / "nan_temp.hd5f", temps=temps),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
            settle_seconds=0.0,
        )
    message = str(excinfo.value)
    assert "= 1" in message  # names the column
    assert "row 2" in message  # names the offending source index


def test_an_infinite_thermistor_reading_raises_and_names_the_source(tmp_path):
    # isfinite, not isnan -- an infinite temperature is equally not a
    # temperature, matching the frequency guard's own reasoning.
    temps = np.stack(
        [np.full(len(TIME_S), 20.0), np.full(len(TIME_S), 100.0)], axis=1
    )
    temps[5, 0] = np.inf  # column 0, shared by antenna and internal_load
    with pytest.raises(DataIngestionError, match="antenna") as excinfo:
        read_rhino_observation(
            make_file(tmp_path / "inf_temp.hd5f", temps=temps),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
            settle_seconds=0.0,
        )
    message = str(excinfo.value)
    assert "= 0" in message
    assert "row 5" in message


def test_a_non_finite_reading_in_an_unused_thermistor_column_does_not_raise(tmp_path):
    # Only the columns a present label actually uses are checked, matching
    # the labels-present policy: column 1 carries a NaN, but this file's
    # switch log never visits a label mapped to it.
    temps = np.stack(
        [np.full(len(TIME_S), 20.0), np.full(len(TIME_S), 100.0)], axis=1
    )
    temps[0, 1] = np.nan
    obs = read_rhino_observation(
        make_file(
            tmp_path / "unused_nan.hd5f",
            temps=temps,
            switch_times=np.array([1000.0]),
            switch_states=[b"antenna"],
        ),
        freq_unit="MHz",
        thermistor_columns={"antenna": 0},
        settle_seconds=0.0,
    )
    assert np.isfinite(obs.thermistor_k["antenna"]).all()


def test_to_state_indexes_sources_and_inverts_the_settling_mask(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=2.0,
    )
    state = to_state(obs, source_order=("antenna", "internal_load", "heated_load"))

    index = np.asarray(state.coords.extra["receiver_input"])
    assert index.shape == (12,)
    assert np.issubdtype(index.dtype, np.integer)
    np.testing.assert_array_equal(index, [0] * 4 + [1] * 4 + [2] * 4)

    np.testing.assert_allclose(np.asarray(state.coords.freq), obs.freq_hz)
    np.testing.assert_allclose(np.asarray(state.coords.time), obs.time_s)
    np.testing.assert_allclose(np.asarray(state.data), obs.waterfall)

    # aux["flags"] is True-means-BAD; settled is True-means-GOOD. With
    # settle_seconds=2.0 on this fixture's 1 s cadence, the split is exact and
    # known (see test_settled_is_false_for_settle_seconds_after_each_transition):
    # 6 of 12 samples settled, 6 not. Pin the count itself, not merely that
    # flags and settled differ somewhere -- a symmetric split like this one is
    # exactly the case where "differs somewhere" cannot distinguish correct
    # inversion from a reversed one, so the element-wise equality below is the
    # check that actually carries the polarity guarantee; the count is the
    # explicit pin the count-based check calls for.
    np.testing.assert_array_equal(np.asarray(state.aux["flags"]), ~obs.settled)
    assert np.asarray(state.aux["flags"]).sum() == 6


def test_to_state_rejects_a_label_outside_source_order(tmp_path):
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
    )
    with pytest.raises(DataIngestionError, match="heated_load"):
        to_state(obs, source_order=("antenna", "internal_load"))


def test_to_state_rejects_a_source_order_with_duplicate_labels(tmp_path):
    # A repeated label collapses two switch positions onto the same index --
    # shape-legal, and NoiseWaveOperator would silently attribute one source's
    # samples to another source's Gamma.
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
    )
    with pytest.raises(DataIngestionError, match="antenna"):
        to_state(
            obs, source_order=("antenna", "antenna", "internal_load", "heated_load")
        )


def test_to_state_rejects_a_non_boolean_settled_array(tmp_path):
    # `~` on a non-bool numpy array is a bitwise complement, not a logical
    # negation -- e.g. ~np.array([1, 0]) == [-2, -1], both nonzero, so a cast
    # back to bool would read as "everything flagged" rather than an inversion.
    # read_rhino_observation always produces bool settled; this guards the
    # to_state seam itself against a hand-built or future-refactored
    # RhinoObservation that does not.
    obs = read_rhino_observation(
        make_file(tmp_path / "obs.hd5f"),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
    )
    bad = dataclasses.replace(obs, settled=obs.settled.astype(int))
    with pytest.raises(DataIngestionError, match="bool"):
        to_state(bad, source_order=("antenna", "internal_load", "heated_load"))


def test_the_public_names_are_reachable_from_the_subpackage():
    from rheplicant import radio
    from rheplicant.radio import RhinoObservation as R
    from rheplicant.radio import read_rhino_observation as r
    from rheplicant.radio import rhino_to_state as t

    assert (R, r, t) == (RhinoObservation, read_rhino_observation, to_state)
    # Reachability alone is not enough -- see the identical note in
    # test_touchstone.py::test_the_public_names_are_reachable_from_the_subpackage:
    # `from pkg import name` resolves through the module namespace, not
    # __all__, so this would still pass even with the trailing `__all__ +=`
    # block for this module deleted.
    assert {"RhinoObservation", "read_rhino_observation", "rhino_to_state"}.issubset(
        radio.__all__
    )
