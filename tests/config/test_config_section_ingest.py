"""observation.from_file: the recording comes off disk as an object."""

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")

from rheplicant.config import FILE_FORMATS, ConfigError, ResolutionContext  # noqa: E402
from rheplicant.config.sections import ingest as _ingest  # noqa: E402,F401  (registers the reader)
from rheplicant.config.sections.ingest import parse_from_file  # noqa: E402
from rheplicant.radio.rhino import RhinoObservation  # noqa: E402

FREQ_MHZ = np.array([60.0, 70.0, 80.0])
TIME_S = np.arange(0.0, 12.0, 1.0) + 1000.0


def make_file(path):
    """The minimal /sdr + /switches + /temperatures schema rhino.py reads
    (the full builder lives in tests/radio/test_rhino.py:37-86)."""
    with h5py.File(path, "w") as handle:
        handle["/sdr/sdr_freqs"] = FREQ_MHZ
        handle["/sdr/sdr_times"] = TIME_S
        handle["/sdr/sdr_waterfall"] = np.ones((TIME_S.size, FREQ_MHZ.size))
        handle["/switches/switch_times"] = np.array([1000.0, 1004.0, 1008.0])
        handle["/switches/switch_states"] = np.array(
            [b"antenna", b"internal_load", b"heated_load"])
        handle["/temperatures/temperatures"] = np.full((TIME_S.size, 2), 20.0)
        handle["/temperatures/temperature_times"] = TIME_S
    return path


@pytest.fixture()
def context(tmp_path):
    make_file(tmp_path / "obs.hd5f")
    return ResolutionContext(dtype="float32", base_dir=str(tmp_path))


class TestTheReader:
    def test_it_is_registered_as_an_object_reader(self):
        assert "rhino_hdf5" in FILE_FORMATS

    def test_the_file_node_returns_the_observation_object(self, context):
        from rheplicant.config import resolve_value

        resolved = resolve_value(
            {"file": {"path": "obs.hd5f", "format": "rhino_hdf5",
                      "freq_unit": "MHz", "settle_seconds": 0.0}},
            context)
        assert isinstance(resolved.value, RhinoObservation)
        assert resolved.value.freq_hz[0] == pytest.approx(60e6)

    def test_modifiers_are_refused_on_the_object_node(self, context):
        from rheplicant.config import resolve_value

        with pytest.raises(ConfigError, match="modifiers"):
            resolve_value(
                {"file": {"path": "obs.hd5f", "format": "rhino_hdf5",
                          "freq_unit": "MHz"}, "unit": "K"},
                context)

    def test_freq_unit_is_required_with_no_default(self, context):
        from rheplicant.config import resolve_value

        with pytest.raises(ConfigError, match="freq_unit"):
            resolve_value(
                {"file": {"path": "obs.hd5f", "format": "rhino_hdf5"}}, context)


class TestParseFromFile:
    def test_the_observation_and_its_record(self, context):
        obs, record = parse_from_file(
            {"format": "rhino_hdf5", "path": "obs.hd5f", "freq_unit": "MHz",
             "settle_seconds": {"value": 0.0, "unit": "s"}},
            context)
        assert isinstance(obs, RhinoObservation)
        assert set(record) == {"from_file/path", "from_file/sha256"}

    def test_an_unknown_ingestion_format_is_refused(self, context):
        with pytest.raises(ConfigError, match="rhino_hdf5"):
            parse_from_file({"format": "npz", "path": "obs.hd5f"}, context)

    def test_thermistor_columns_reach_the_reader(self, context):
        """Every label in the switch log needs a column -- the reader refuses
        a partial map (rhino.py) -- so the antenna column is declared too."""
        obs, _ = parse_from_file(
            {"format": "rhino_hdf5", "path": "obs.hd5f", "freq_unit": "MHz",
             "settle_seconds": 0.0,
             "thermistor_columns": {"antenna": 0, "internal_load": 0,
                                    "heated_load": 1}},
            context)
        assert {"internal_load", "heated_load"} <= set(obs.thermistor_k)
        assert obs.thermistor_k["internal_load"][0] == pytest.approx(293.15)


class TestBuildObservationIngested:
    def _runtime(self):
        from rheplicant.config.sections.runtime import build_runtime

        return build_runtime({"seed": 3})

    def test_from_file_and_a_declared_axis_are_refused_together(self, context, tmp_path):
        from rheplicant.config.sections.observation import build_observation

        section = {
            "from_file": {"format": "rhino_hdf5", "path": "obs.hd5f",
                          "freq_unit": "MHz", "settle_seconds": 0.0},
            "time": {"grid": {"arange": {"start": 0.0, "step": 1.0, "num": 4},
                              "unit": "s"}},
        }
        with pytest.raises(ConfigError, match="together say two things"):
            build_observation(section, runtime=self._runtime(),
                              base_dir=str(tmp_path))

    def test_an_ingested_run_declares_order_only(self, context, tmp_path):
        from rheplicant.config.sections.observation import build_observation

        section = {
            "from_file": {"format": "rhino_hdf5", "path": "obs.hd5f",
                          "freq_unit": "MHz", "settle_seconds": 0.0},
            "switching": {"mode": "cycle",
                          "order": ["antenna", "internal_load", "heated_load"]},
        }
        with pytest.raises(ConfigError, match="declares order"):
            build_observation(section, runtime=self._runtime(),
                              base_dir=str(tmp_path))
