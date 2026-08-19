"""Addressed observation.pointing -> Coordinates.pointing / coords.extra entries."""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError, ResolutionContext
from rheplicant.config.sections.observation import SiteFacts
from rheplicant.config.sections.pointing import PointingBuild, compile_pointing

N_TIME = 8
TIME_S = jnp.arange(0.0, 16.0, 2.0)
FREQ_HZ = jnp.linspace(60e6, 85e6, 4)
SITE = SiteFacts(lat_deg=53.2367, lon_deg=-2.3085, alt_m=78.0)
NO_SITE = SiteFacts(None, None, None)
EPOCH = 1785312000.0

DRIFT = {
    "mode": "drift",
    "az_deg": {"value": 0.0, "unit": "deg"},
    "el_deg": {"value": 90.0, "unit": "deg"},
    "selfrot_deg": {"value": 0.0, "unit": "deg"},
    "materialise": ["pointing", "selfrot_deg"],
}


@pytest.fixture()
def context():
    return ResolutionContext(freq=FREQ_HZ, time=TIME_S, dtype="float32")


def compile(spec, context, site=SITE, epoch=EPOCH):
    return compile_pointing(spec, context, time_s=TIME_S, epoch_unix_s=epoch,
                            site=site)


class TestModeNone:
    def test_the_default_is_none(self, context):
        build = compile(None, context)
        assert build == PointingBuild(pointing=None, extra={}, provenance={})

    def test_none_takes_no_other_keys(self, context):
        with pytest.raises(ConfigError, match=r"\['az_deg'\]"):
            compile({"mode": "none", "az_deg": 0.0}, context)

    def test_an_unknown_mode_is_refused(self, context):
        with pytest.raises(ConfigError, match="tracked"):
            compile({"mode": "sweep"}, context)


class TestDrift:
    def test_materialise_writes_exactly_what_it_names(self, context):
        build = compile(DRIFT, context)
        assert build.pointing.shape == (N_TIME, 2)
        assert float(build.pointing[0, 1]) == pytest.approx(90.0)
        assert build.extra["selfrot_deg"].shape == (N_TIME,)
        partial = compile({**DRIFT, "materialise": ["pointing"]}, context)
        assert "selfrot_deg" not in partial.extra

    def test_materialise_is_written_not_inferred(self, context):
        spec = {k: v for k, v in DRIFT.items() if k != "materialise"}
        with pytest.raises(ConfigError, match="materialise"):
            compile(spec, context)

    def test_materialise_entries_are_a_closed_table(self, context):
        with pytest.raises(ConfigError, match="pointing.*selfrot_deg"):
            compile({**DRIFT, "materialise": ["azimuth"]}, context)

    def test_angles_must_be_angles(self, context):
        with pytest.raises(ConfigError, match="angle"):
            compile({**DRIFT, "az_deg": {"value": 0.0, "unit": "m"}}, context)


class TestTheLstRoutes:
    def test_uniform_turn_excludes_the_endpoint(self, context):
        spec = {**DRIFT, "lst": {"mode": "uniform_turn", "n_time": "n_time",
                                 "lst0_deg": {"value": 10.0, "unit": "deg"}}}
        lst = compile(spec, context).extra["lst_deg"]
        assert lst.shape == (N_TIME,)
        assert float(lst[0]) == pytest.approx(10.0)
        assert float(lst[-1]) == pytest.approx(10.0 + 360.0 * (N_TIME - 1) / N_TIME)

    def test_uniform_turn_refuses_a_disagreeing_n_time(self, context):
        spec = {**DRIFT, "lst": {"mode": "uniform_turn", "n_time": 7}}
        with pytest.raises(ConfigError, match="n_time"):
            compile(spec, context)

    def test_from_file_reads_the_array(self, context, tmp_path):
        path = tmp_path / "lst.npy"
        np.save(path, np.linspace(0.0, 350.0, N_TIME))
        ctx = ResolutionContext(freq=FREQ_HZ, time=TIME_S, dtype="float32",
                                base_dir=str(tmp_path))
        spec = {**DRIFT, "lst": {"from_file": {"path": "lst.npy", "format": "npy"}}}
        lst = compile(spec, ctx).extra["lst_deg"]
        assert lst.shape == (N_TIME,)

    def test_a_wrong_length_lst_file_is_refused(self, context, tmp_path):
        path = tmp_path / "lst.npy"
        np.save(path, np.zeros(3))
        ctx = ResolutionContext(freq=FREQ_HZ, time=TIME_S, dtype="float32",
                                base_dir=str(tmp_path))
        spec = {**DRIFT, "lst": {"from_file": {"path": "lst.npy", "format": "npy"}}}
        with pytest.raises(ConfigError, match=r"\(n_time,\)"):
            compile(spec, ctx)

    def test_from_site_computes_through_the_adapter(self, context):
        spec = {**DRIFT, "lst": {"mode": "from_site"}}
        lst = compile(spec, context).extra["lst_deg"]
        assert lst.shape == (N_TIME,)
        assert np.all((np.asarray(lst) >= 0.0) & (np.asarray(lst) < 360.0))
        assert float(lst[0]) == pytest.approx(64.6836685570699, abs=0.01)

    def test_lst0_defaults_to_zero(self, context):
        spec = {**DRIFT, "lst": {"mode": "uniform_turn", "n_time": "n_time"}}
        lst = compile(spec, context).extra["lst_deg"]
        assert float(lst[0]) == pytest.approx(0.0)

    def test_from_site_names_what_is_missing(self, context):
        spec = {**DRIFT, "lst": {"mode": "from_site"}}
        with pytest.raises(ConfigError, match="lon_deg"):
            compile(spec, context, site=SiteFacts(53.0, None, 78.0))
        with pytest.raises(ConfigError, match="epoch"):
            compile(spec, context, epoch=None)


class TestTracked:
    def test_the_table_becomes_pointing(self, context):
        table = [[float(i), 45.0] for i in range(N_TIME)]
        spec = {"mode": "tracked", "table": {"list": table, "unit": "deg"},
                "lst": {"mode": "from_site"}}
        build = compile(spec, context)
        assert build.pointing.shape == (N_TIME, 2)
        assert "lst_deg" in build.extra

    def test_lst_is_required(self, context):
        spec = {"mode": "tracked",
                "table": {"list": [[0.0, 45.0]] * N_TIME, "unit": "deg"}}
        with pytest.raises(ConfigError, match="lst"):
            compile(spec, context)

    def test_a_wrong_shape_table_is_refused(self, context):
        spec = {"mode": "tracked", "table": {"list": [0.0] * N_TIME, "unit": "deg"},
                "lst": {"mode": "from_site"}}
        with pytest.raises(ConfigError, match=r"\(n_time, 2\)"):
            compile(spec, context)

    def test_a_declared_selfrot_track_is_written(self, context):
        table = [[0.0, 45.0]] * N_TIME
        spec = {"mode": "tracked", "table": {"list": table, "unit": "deg"},
                "lst": {"mode": "from_site"},
                "selfrot": {"list": [1.0] * N_TIME, "unit": "deg"}}
        build = compile(spec, context)
        assert build.extra["selfrot_deg"].shape == (N_TIME,)
        assert float(build.extra["selfrot_deg"][0]) == pytest.approx(1.0)

    def test_a_wrong_shape_selfrot_is_refused(self, context):
        spec = {"mode": "tracked", "table": {"list": [[0.0, 45.0]] * N_TIME,
                                             "unit": "deg"},
                "lst": {"mode": "from_site"},
                "selfrot": {"list": [1.0, 2.0], "unit": "deg"}}
        with pytest.raises(ConfigError, match=r"\(n_time,\)"):
            compile(spec, context)


class TestBaked:
    def test_provenance_is_required_and_recorded(self, context):
        with pytest.raises(ConfigError, match="provenance"):
            compile({"mode": "baked"}, context)
        build = compile({"mode": "baked",
                         "provenance": {"built_by": "driftscan_v2", "lat_deg": 53.2}},
                        context)
        assert build.pointing is None
        assert build.provenance == {"pointing/built_by": "driftscan_v2",
                                    "pointing/lat_deg": 53.2}
