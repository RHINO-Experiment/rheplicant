"""The RHINO observation reader's own surface.

The file schema is fixed by two producers that do not agree with each other --
rhino-cal's ObservationHandler.save_to_hdf5 writes frequencies in Hz, the
RHINO_fully_simulated_calibration notebook writes them in MHz, and the file
records neither. Most of what is tested here is that disagreement being caught
rather than absorbed.
"""

import numpy as np
import pytest

from rheplicant.core.errors import DataIngestionError

h5py = pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")

from rheplicant.radio.rhino import RhinoObservation, read_rhino_observation  # noqa: E402

FREQ_MHZ = np.array([60.0, 70.0, 80.0])
#: Four samples per switch position, 1 s apart, three positions.
TIME_S = np.arange(0.0, 12.0, 1.0) + 1000.0
SWITCH_TIMES = np.array([1000.0, 1004.0, 1008.0])
SWITCH_STATES = [b"antenna", b"internal_load", b"heated_load"]


def make_file(path, *, freqs=FREQ_MHZ, times=TIME_S, temps=None, with_adc=False):
    n_time, n_freq = len(times), len(freqs)
    if temps is None:
        # column 0 = ambient (20 C), column 1 = hot (100 C)
        temps = np.stack(
            [np.full(n_time, 20.0), np.full(n_time, 100.0)], axis=1
        )
    with h5py.File(path, "w") as f:
        sdr = f.create_group("sdr")
        sdr.create_dataset("sdr_freqs", data=freqs)
        sdr.create_dataset("sdr_times", data=times)
        sdr.create_dataset(
            "sdr_waterfall",
            data=np.arange(n_time * n_freq, dtype=float).reshape(n_time, n_freq),
        )
        if with_adc:
            sdr.create_dataset("max_i_adc", data=np.zeros(n_time))
            sdr.create_dataset("max_q_adc", data=np.ones(n_time))
        sw = f.create_group("switches")
        sw.create_dataset("switch_times", data=SWITCH_TIMES)
        sw.create_dataset("switch_states", data=np.array(SWITCH_STATES, dtype="S"))
        tg = f.create_group("temperatures")
        tg.create_dataset("temperature_times", data=times)
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
