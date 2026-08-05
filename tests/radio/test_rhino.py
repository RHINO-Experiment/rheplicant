"""The RHINO observation reader's own surface.

The file schema is fixed by two producers that do not agree with each other --
rhino-cal's ObservationHandler.save_to_hdf5 writes frequencies in Hz, the
RHINO_fully_simulated_calibration notebook writes them in MHz, and the file
records neither. Most of what is tested here is that disagreement being caught
rather than absorbed.
"""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import DataIngestionError
from rheplicant.core.state import State

h5py = pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")

from rheplicant.radio.rhino import (  # noqa: E402
    TIME_EPOCH_META_KEY,
    RhinoObservation,
    cal_load_operators,
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


def test_samples_before_the_first_transition_are_dropped_counted_and_cut_everywhere(
    tmp_path,
):
    # Merged from two tests that built this identical fixture: one pinned the
    # drop COUNT plus four arrays' shapes, the other pinned that every array on
    # the sample axis -- thermistors included -- comes back the same length as
    # time_s. They are one claim about one read, and keeping them apart hid
    # that neither covered the two OPTIONAL sample-axis arrays: the ADC
    # monitors are cut by the same mask, and dropping that cut survived the
    # whole suite (the only test that reads them uses a file with nothing to
    # drop, where cut and uncut are the same array).
    early = np.concatenate([[998.0, 999.0], TIME_S])
    obs = read_rhino_observation(
        make_file(tmp_path / "early.hd5f", times=early, with_adc=True),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=0.0,
    )
    assert obs.n_leading_dropped == 2
    assert obs.time_s.shape == (12,)
    n = obs.time_s.size
    assert obs.waterfall.shape == (n, 3)
    assert obs.switch_label.shape == (n,)
    assert obs.settled.shape == (n,)
    assert obs.adc_max_i.shape == (n,)
    assert obs.adc_max_q.shape == (n,)
    for label, series in obs.thermistor_k.items():
        assert series.shape == (n,), label


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


def test_a_non_finite_switch_time_raises_rather_than_losing_its_state(tmp_path):
    # The fourth axis in this package to meet the same mechanism, and the last
    # one to be guarded. sdr_freqs, sdr_times and the interpolation axes all
    # check finiteness because `nan <= 0` and `nan > hi` are both False, so a
    # comparison-based guard never sees a NaN. switch_times reaches the same
    # searchsorted call two lines below the sdr_times guard, and had none.
    #
    # Unguarded, this file read back with NO exception, n_leading_dropped == 0
    # and all 12 samples kept -- but "internal_load" absent from switch_label
    # entirely, its samples folded into the neighbouring states. searchsorted
    # sorts NaN to the end, so the corrupted transition ends up after every
    # sample and can never be selected. A finite, correctly-shaped, silently
    # wrong recording: exactly what this layer exists to prevent.
    corrupted = np.array([1000.0, np.nan, 1008.0])
    with pytest.raises(DataIngestionError, match="non-finite"):
        read_rhino_observation(
            make_file(tmp_path / "nanswitch.hd5f", switch_times=corrupted),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
            settle_seconds=0.0,
        )


@pytest.mark.parametrize("shape", [(len(TIME_S) * 3,), (len(TIME_S), 3, 1)])
def test_a_waterfall_that_is_not_2d_raises(tmp_path, shape):
    # The sibling of the channel-count check below, and the branch nothing
    # pinned: deleting `waterfall.ndim != 2` left the whole suite green. The
    # two directions fail differently and neither is benign. A 1-D waterfall
    # falls through to the channel check, which indexes shape[1] and raises a
    # bare IndexError -- not a DataIngestionError, so a caller catching this
    # package's ingestion error catches nothing. A 3-D one PASSES the channel
    # check outright, because shape[1] is still n_freq, and reads back as a
    # well-formed recording carrying a spare axis.
    bad = np.zeros(shape, dtype=float)
    with pytest.raises(DataIngestionError, match="2-D"):
        read_rhino_observation(
            make_file(tmp_path / "notmatrix.hd5f", waterfall=bad),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
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


@pytest.mark.parametrize("column", [-1, 5])
def test_a_thermistor_column_index_out_of_range_raises(tmp_path, column):
    # /temperatures/temperatures has 2 columns in the default fixture, so both
    # of these are out of range -- and the guard `0 <= column < n` has two
    # independently-failable sides. Only the upper one used to be tested, while
    # the comment claimed both. Narrow it to `column >= n` and -1 stops raising:
    # numpy's negative indexing hands back the LAST column instead, which is a
    # real thermistor reading of the wrong load. Shape-legal, silently wrong.
    with pytest.raises(DataIngestionError, match="antenna"):
        read_rhino_observation(
            make_file(tmp_path / f"obs{column}.hd5f"),
            freq_unit="MHz",
            thermistor_columns={"antenna": column, "internal_load": 0, "heated_load": 1},
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


class TestTheThermistorLogIsOnlyReadWhenItIsAskedFor:
    """A file must not be refused over a column nothing downstream consumes.

    ``_thermistors_in_kelvin`` used to run unconditionally, and its two
    refusals -- a log ending short of the SDR axis, a non-finite reading in a
    used column -- made a whole recording unreadable. Both refusals are right
    for a caller who wants the temperatures. Neither is right for a caller who
    does not, and until a load-temperature consumer exists in ``src/`` nobody
    does: ``to_state`` carries ``data``, ``coords.time``, ``coords.freq``,
    ``coords.extra["receiver_input"]``, ``aux["flags"]`` and the epoch in
    ``meta``, and ``thermistor_k`` reaches no operator anywhere (audited by
    grepping ``src/`` for each field of RhinoObservation).

    So ``thermistor_columns`` is now opt-in. It is still REQUIRED to get
    temperatures -- the map is a declaration the file cannot make for itself,
    which is the argument ``_thermistors_in_kelvin`` gives for demanding it --
    and omitting it says "not these", not "guess".

    Every case below is tested from BOTH sides: the same file read once with
    columns (refused) and once without (read).
    """

    BASE = 1_735_000_000.0

    def _short_log(self, tmp_path):
        """A thermistor log ending 1.5 s before the last SDR sample."""
        return make_file(
            tmp_path / "short_log.hd5f",
            times=np.array([self.BASE, self.BASE + 11.5]),
            temps=np.array([[20.0], [20.0]]),
            temp_times=np.array([self.BASE, self.BASE + 10.0]),
            switch_times=np.array([self.BASE]),
            switch_states=[b"antenna"],
        )

    def _nan_row(self, tmp_path):
        """A NaN in the column ``antenna`` maps onto, at row 1 of 12."""
        temps = np.stack([np.full(len(TIME_S), 20.0), np.full(len(TIME_S), 100.0)], axis=1)
        temps[1, 0] = np.nan
        return make_file(tmp_path / "nan_row.hd5f", temps=temps)

    def _flat_table(self, tmp_path):
        """A 1-D /temperatures, which cannot be addressed by column at all."""
        return make_file(tmp_path / "flat.hd5f", temps=np.full(len(TIME_S), 20.0))

    @pytest.mark.parametrize(
        ("build", "declared", "match"),
        [
            ("_short_log", {"antenna": 0}, "outside"),
            ("_nan_row", COLUMNS, "non-finite"),
            ("_flat_table", COLUMNS, "2-D"),
        ],
    )
    def test_declaring_the_columns_still_refuses_every_bad_log(
        self, tmp_path, build, declared, match
    ):
        with pytest.raises(DataIngestionError, match=match):
            read_rhino_observation(
                getattr(self, build)(tmp_path),
                freq_unit="MHz",
                thermistor_columns=declared,
                settle_seconds=0.0,
            )

    @pytest.mark.parametrize("build", ["_short_log", "_nan_row", "_flat_table"])
    def test_omitting_them_reads_the_same_file(self, tmp_path, build):
        obs = read_rhino_observation(
            getattr(self, build)(tmp_path), freq_unit="MHz", settle_seconds=0.0
        )
        assert obs.thermistor_k == {}
        assert obs.waterfall.shape == (obs.time_s.size, obs.freq_hz.size)
        assert np.isfinite(obs.time_s).all()

    def test_the_signal_path_is_untouched_by_the_omission(self, tmp_path):
        """What is carried must not depend on what was declared. Both reads of
        the SAME file must agree element-by-element on everything ``to_state``
        puts on the graph -- on a fixture whose switch blocks are 4/4/4 but
        whose waterfall and index array are not symmetric."""
        path = make_file(tmp_path / "both.hd5f")
        kwargs = dict(freq_unit="MHz", settle_seconds=2.0)
        with_temps = read_rhino_observation(path, thermistor_columns=COLUMNS, **kwargs)
        without = read_rhino_observation(path, **kwargs)

        assert set(with_temps.thermistor_k) == {"antenna", "internal_load", "heated_load"}
        assert without.thermistor_k == {}

        a = to_state(with_temps, source_order=("antenna", "internal_load", "heated_load"))
        b = to_state(without, source_order=("antenna", "internal_load", "heated_load"))
        for field in ("data", "coords.time", "coords.freq"):
            obj = a if field == "data" else a.coords
            other = b if field == "data" else b.coords
            name = field.split(".")[-1]
            np.testing.assert_array_equal(
                np.asarray(getattr(obj, name)), np.asarray(getattr(other, name))
            )
        np.testing.assert_array_equal(
            np.asarray(a.coords.extra["receiver_input"]),
            np.asarray(b.coords.extra["receiver_input"]),
        )
        np.testing.assert_array_equal(np.asarray(a.aux["flags"]), np.asarray(b.aux["flags"]))
        assert a.meta[TIME_EPOCH_META_KEY] == b.meta[TIME_EPOCH_META_KEY]

    def test_a_file_with_no_temperature_group_at_all_is_readable(self, tmp_path):
        """The stronger form: not merely tolerating a bad log, but not needing
        the datasets to exist. Reading them anyway would refuse this file with a
        raw h5py KeyError over data the caller never asked for."""
        path = tmp_path / "no_temps.hd5f"
        make_file(path)
        with h5py.File(path, "a") as f:
            del f["temperatures"]
        obs = read_rhino_observation(path, freq_unit="MHz", settle_seconds=0.0)
        assert obs.thermistor_k == {}
        assert obs.waterfall.shape == (len(TIME_S), len(FREQ_MHZ))

    def test_an_unmapped_label_is_still_refused_when_columns_are_declared(self, tmp_path):
        """Opting in must not have loosened the declaration it opts into."""
        with pytest.raises(DataIngestionError, match="heated_load"):
            read_rhino_observation(
                make_file(tmp_path / "partial.hd5f"),
                freq_unit="MHz",
                thermistor_columns={"antenna": 0, "internal_load": 0},
            )


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
    # Relative to the first sample, with the epoch in meta -- see
    # TestTheTimeAxisIsStoredFromTheStartOfTheRun below.
    np.testing.assert_allclose(np.asarray(state.coords.time), obs.time_s - obs.time_s[0])
    assert state.meta[TIME_EPOCH_META_KEY] == obs.time_s[0]
    np.testing.assert_allclose(np.asarray(state.data), obs.waterfall)

    # aux["flags"] is True-means-BAD; settled is True-means-GOOD. Every
    # consumer (FlaggedNoise, SkySpaceFilter, both flagging operators) needs
    # flags shaped like the data, not like the per-time settling mask, so
    # settled is broadcast across the frequency axis: every channel of an
    # unsettled sample is unsettled. Assert against that convention --
    # state.data.shape -- rather than a hardcoded (12, 3), so this tracks the
    # fixture rather than freezing today's numbers.
    flags = np.asarray(state.aux["flags"])
    assert flags.shape == np.asarray(state.data).shape

    # With settle_seconds=2.0 on this fixture's 1 s cadence, the split is
    # exact and known (see
    # test_settled_is_false_for_settle_seconds_after_each_transition): 6 of
    # 12 samples settled, 6 not. Pin the count itself, not merely that flags
    # and settled differ somewhere -- a symmetric split like this one is
    # exactly the case where "differs somewhere" cannot distinguish correct
    # inversion from a reversed one, so the element-wise equality below
    # (against the broadcast expectation) is the check that actually carries
    # the polarity guarantee; the count is the explicit pin the exact-count
    # instruction calls for.
    expected = np.broadcast_to((~obs.settled)[:, None], flags.shape)
    np.testing.assert_array_equal(flags, expected)
    assert flags.sum() == 6 * obs.freq_hz.size


def test_to_state_after_a_leading_drop_attributes_and_flags_the_kept_samples(tmp_path):
    # Nothing combined a leading drop with to_state, and the two halves of the
    # seam are exactly where this file has already shipped a Critical bug: a
    # non-finite switch_time silently lost an entire switch state with no
    # exception, nothing dropped and every array the right shape. So anchor on
    # what the forward pass would SEE -- which source each sample is attributed
    # to, which waterfall rows survived, and which cells are flagged -- not on
    # shapes, which would not have caught that bug either.
    #
    # Every axis of this fixture is asymmetric on purpose, so an off-by-the-drop
    # shift, a transposed source order or an inverted mask each change a number
    # rather than only a shape: 3 samples dropped (not 2), blocks of 2/5/4 kept
    # samples (not 4/4/4), 11 samples against 3 channels (not square), 6 flagged
    # against 5 unflagged (not 6/6), and a source_order that is not the order
    # the file switches in.
    kept_times = np.arange(1000.0, 1011.0)
    times = np.concatenate([[995.0, 996.0, 997.0], kept_times])
    waterfall = np.arange(len(times) * 3, dtype=float).reshape(len(times), 3) * 7.0
    obs = read_rhino_observation(
        make_file(
            tmp_path / "drop_then_state.hd5f",
            times=times,
            waterfall=waterfall,
            switch_times=np.array([1000.0, 1002.0, 1007.0]),
            switch_states=[b"internal_load", b"antenna", b"heated_load"],
        ),
        freq_unit="MHz",
        thermistor_columns=COLUMNS,
        settle_seconds=2.0,
    )
    assert obs.n_leading_dropped == 3

    # Named in an order the file never switches in: an index built from the
    # order the labels APPEAR in reads back as [0,0,1,1,1,1,1,2,2,2,2], which
    # is a different array of the same shape and dtype.
    state = to_state(obs, source_order=("heated_load", "antenna", "internal_load"))

    index = np.asarray(state.coords.extra["receiver_input"])
    np.testing.assert_array_equal(index, [2, 2] + [1] * 5 + [0] * 4)

    # The rows that survived are the LAST 11 of 14, not the first 11. Both are
    # (11, 3) and they share no element.
    np.testing.assert_allclose(np.asarray(state.data), waterfall[3:])
    # Measured from the first KEPT sample, not from the first sample in the
    # file: the dropped head is not part of the run the State describes, and an
    # epoch taken before the drop would put the whole axis 5 s off its own zero.
    np.testing.assert_allclose(np.asarray(state.coords.time), kept_times - kept_times[0])
    assert state.meta[TIME_EPOCH_META_KEY] == kept_times[0]

    # Settling is measured from each transition, on the samples that were KEPT:
    # computing it against the head of the un-dropped axis instead shifts every
    # elapsed time by the drop and marks the whole recording unsettled.
    expected_settled = np.array(
        [False, False]  # internal_load, t = 1000, 1001
        + [False, False, True, True, True]  # antenna, t = 1002 .. 1006
        + [False, False, True, True]  # heated_load, t = 1007 .. 1010
    )
    np.testing.assert_array_equal(obs.settled, expected_settled)

    # True-means-bad, broadcast across the 3 channels. The 6/5 split is the
    # point of the fixture: an inverted mask reports 15 flagged cells, not 18.
    flags = np.asarray(state.aux["flags"])
    assert flags.shape == (kept_times.size, obs.freq_hz.size)
    np.testing.assert_array_equal(
        flags, np.broadcast_to(~expected_settled[:, None], flags.shape)
    )
    assert flags.sum() == 6 * obs.freq_hz.size


class TestTheTimeAxisIsStoredFromTheStartOfTheRun:
    """Why ``to_state`` no longer hands ``obs.time_s`` straight to Coordinates.

    ``Coordinates`` stores through ``jnp.asarray`` -- float32 unless x64 is on
    -- and a unix second near 1.75e9 sits on a 128 s grid there. Measured on
    this fixture's six samples at offsets [0, 100, 250, 450, 700, 1000] s::

        offsets asked for   [   0,  100,  250,  450,  700, 1000]
        offsets stored      [   0,  128,  256,  512,  640, 1024]
        error [s]           [   0,  +28,   +6,  +62,  -60,  +24]

    All six remain DISTINCT, so nothing structural signals the loss -- no
    collision, no NaN, the right shape and the right count -- while individual
    timestamps are wrong by up to 62 s. Subtracting the first sample before the
    store removes the cause rather than detecting it: the offsets are then small
    integers, exact in float32, and the absolute epoch survives in ``meta``.

    The public behaviour change: ``state.coords.time`` is SECONDS SINCE THE
    FIRST KEPT SAMPLE, not unix seconds. ``obs.time_s`` on the recording is
    unchanged and still absolute.
    """

    OFFSETS = np.array([0.0, 100.0, 250.0, 450.0, 700.0, 1000.0])
    EPOCH = 1_750_000_000.0
    #: 1 / 3 / 2 samples per switch position -- deliberately not 2 / 2 / 2, so a
    #: mis-attributed sample changes the index array rather than only a shape.
    SWITCHES = EPOCH + np.array([0.0, 100.0, 700.0])
    ORDER = ("antenna", "internal_load", "heated_load")

    def _obs(self, tmp_path):
        return read_rhino_observation(
            make_file(
                tmp_path / "unix.hd5f",
                times=self.EPOCH + self.OFFSETS,
                switch_times=self.SWITCHES,
                switch_states=SWITCH_STATES,
            ),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
            settle_seconds=0.0,
        )

    def test_the_stored_axis_is_exact_and_the_epoch_recovers_the_absolute_time(
        self, tmp_path
    ):
        obs = self._obs(tmp_path)
        state = to_state(obs, source_order=self.ORDER)

        stored = np.asarray(state.coords.time, dtype=np.float64)
        # Exact equality, not allclose: these offsets are integers below 2**24
        # and float32 holds them without error. An allclose here would pass on
        # the very axis this change exists to eliminate.
        np.testing.assert_array_equal(stored, self.OFFSETS)

        epoch = state.meta[TIME_EPOCH_META_KEY]
        assert epoch == self.EPOCH
        np.testing.assert_array_equal(epoch + stored, obs.time_s)

    def test_the_recording_itself_still_carries_absolute_unix_seconds(self, tmp_path):
        """The two layers stay separate: only the State's axis moved."""
        obs = self._obs(tmp_path)
        np.testing.assert_array_equal(obs.time_s, self.EPOCH + self.OFFSETS)

    def test_the_absolute_axis_would_not_have_survived_the_store(self, tmp_path):
        """The measurement above, as an assertion: what the old ``to_state``
        produced is now refused outright by the container it produced it for."""
        from rheplicant.core.coordinates import Coordinates
        from rheplicant.core.errors import StateValidationError

        obs = self._obs(tmp_path)
        with pytest.raises(StateValidationError, match="representable"):
            Coordinates(time=obs.time_s)

        # And what the loss actually was, had it been stored: six distinct
        # values, no NaN, no collision -- and up to 62 s of error.
        import jax.numpy as jnp

        lossy = np.asarray(jnp.asarray(obs.time_s), dtype=np.float64)
        assert len(set(lossy.tolist())) == obs.time_s.size
        assert np.abs(lossy - obs.time_s).max() == 62.0

    def test_averaging_the_relative_axis_gives_the_chunk_times_exactly(self, tmp_path):
        """The stage the defect surfaced in. Chunks of 2 over gaps 100/150/200/
        250/300 give means 50 / 350 / 850 -- three different numbers, so a chunk
        boundary off by one changes every one of them."""
        from rheplicant.radio import BackendOperator

        state = to_state(self._obs(tmp_path), source_order=self.ORDER)
        out = BackendOperator(n_chunk=2)(state)
        np.testing.assert_array_equal(
            np.asarray(out.coords.time, dtype=np.float64), [50.0, 350.0, 850.0]
        )

    def test_a_recording_with_no_samples_is_named_rather_than_indexed(self):
        """``obs.time_s[0]`` on an empty axis is a bare IndexError with nothing
        in it. read_rhino_observation cannot produce one, but a hand-built
        RhinoObservation can, and this seam already guards two other ways of
        hand-building a malformed one."""
        empty = RhinoObservation(
            freq_hz=np.array([60e6]),
            time_s=np.array([]),
            waterfall=np.zeros((0, 1)),
            switch_label=np.array([], dtype="<U8"),
            settled=np.array([], dtype=bool),
            thermistor_k={},
            transitions=(np.array([]), np.array([], dtype="<U8")),
            n_leading_dropped=0,
            adc_max_i=None,
            adc_max_q=None,
        )
        with pytest.raises(DataIngestionError, match="no samples"):
            to_state(empty, source_order=("antenna",))


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


class TestCalLoadOperatorsFromARecording:
    """The route from file to model, which used to stop at ``to_state``.

    ``read_rhino_observation`` parses the thermistor log, interpolates it onto
    the SDR axis, and refuses a recording whose readings are short or
    non-finite. ``to_state`` then dropped it, because a ``State`` has nowhere
    to put a per-load temperature -- so the loads' temperatures were parsed,
    validated and discarded, and the warm/hot-load noise-wave path had no way
    to reach a model from a recording at all.
    """

    #: A DRIFTING temperature log, not the module default. The default holds
    #: each column constant, and a constant column cannot distinguish a
    #: per-sample reading from a per-channel one -- which is the single thing
    #: these tests are about. Both columns drift, by different amounts and in
    #: opposite directions, so a swap between them is visible too.
    TEMPS = np.stack(
        [
            np.linspace(19.0, 23.0, len(TIME_S)),   # ambient, rising
            np.linspace(101.5, 98.0, len(TIME_S)),  # hot, falling
        ],
        axis=1,
    )

    def _obs(self, tmp_path, **kwargs):
        kwargs.setdefault("temps", self.TEMPS)
        return read_rhino_observation(
            make_file(tmp_path / "obs.hd5f", **kwargs),
            freq_unit="MHz",
            thermistor_columns=COLUMNS,
        )

    def test_it_builds_one_operator_per_label_carrying_that_label_temperature(
        self, tmp_path
    ):
        obs = self._obs(tmp_path)
        operators = cal_load_operators(obs)

        assert set(operators) == set(obs.thermistor_k)
        for label, operator in operators.items():
            # An explicit (n_time, 1) column, never a bare (n_time,): on a
            # square grid the bare form reads equally well as per-channel.
            assert operator.t_load.shape == (obs.time_s.shape[0], 1)
            np.testing.assert_allclose(
                np.asarray(operator.t_load)[:, 0],
                obs.thermistor_k[label],
                rtol=1e-6,
            )

    def test_the_temperature_varies_along_TIME_and_not_along_FREQUENCY(
        self, tmp_path
    ):
        """The axis assertion, which is the whole reason for the column shape.

        A per-sample temperature applied per-channel would be finite, correctly
        shaped, and describe a different instrument. Checked on the operator's
        OUTPUT rather than on its leaf, because the leaf's shape is what the
        test above pins and this is about what the operator does with it.
        """
        obs = self._obs(tmp_path)
        label = next(iter(obs.thermistor_k))
        operator = cal_load_operators(obs)[label]

        n_time, n_freq = obs.time_s.shape[0], obs.freq_hz.shape[0]
        state = State(
            data=jnp.zeros((n_time, n_freq)),
            coords=Coordinates(
                time=jnp.asarray(obs.time_s - obs.time_s[0]),
                freq=jnp.asarray(obs.freq_hz),
            ),
        )
        out = np.asarray(operator(state).data)

        # every row constant across channels...
        for row in out:
            assert len(set(row.tolist())) == 1, row
        # ...and the column reproduces the log, which is only a real assertion
        # because the fixture's temperatures are not all equal.
        np.testing.assert_allclose(out[:, 0], obs.thermistor_k[label], rtol=1e-6)
        assert len(set(out[:, 0].tolist())) > 1, "the fixture must vary in time"

    def test_labels_pins_the_switch_order(self, tmp_path):
        """The order is the order ``gamma_src``'s rows must match.

        Insertion order of a dict built from the file is an accident of the
        file; naming the labels is how a caller makes it a declaration.
        """
        obs = self._obs(tmp_path)
        both = [label for label in obs.thermistor_k]
        assert len(both) >= 2, obs.thermistor_k
        forward = list(cal_load_operators(obs, labels=both))
        reverse = list(cal_load_operators(obs, labels=list(reversed(both))))
        assert forward == both
        assert reverse == list(reversed(both))

    def test_a_recording_read_without_thermistor_columns_is_refused(self, tmp_path):
        """Not "returns an empty mapping".

        A caller asking for load operators and getting none back would build a
        model with no loads and no warning, which is the failure this whole
        route exists to remove.
        """
        obs = read_rhino_observation(
            make_file(tmp_path / "obs.hd5f"), freq_unit="MHz"
        )
        assert obs.thermistor_k == {}
        with pytest.raises(DataIngestionError, match="no thermistor temperatures"):
            cal_load_operators(obs)

    def test_a_label_this_file_never_switched_to_is_refused_by_name(self, tmp_path):
        obs = self._obs(tmp_path)
        with pytest.raises(DataIngestionError, match=r"\['nonexistent_load'\]"):
            cal_load_operators(obs, labels=["nonexistent_load"])

    def test_the_two_refusals_say_different_things(self, tmp_path):
        """"Unread log" and "label not in this file" are different mistakes.

        Both are DataIngestionError, and the reader's own docstring says the
        caller distinguishes them by what it declared -- so the two messages
        have to make that possible.
        """
        unread = read_rhino_observation(
            make_file(tmp_path / "a.hd5f", temps=self.TEMPS), freq_unit="MHz"
        )
        read = self._obs(tmp_path)
        with pytest.raises(DataIngestionError) as no_log:
            cal_load_operators(unread)
        with pytest.raises(DataIngestionError) as no_label:
            cal_load_operators(read, labels=["nonexistent_load"])
        assert str(no_log.value) != str(no_label.value)
        assert "no thermistor temperatures" in str(no_log.value)
        assert "no thermistor temperatures" not in str(no_label.value)

