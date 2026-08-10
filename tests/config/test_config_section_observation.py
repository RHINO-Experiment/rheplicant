"""observation: field loaders -- grids, meta, site, environment, extra, aux, data."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError, ResolutionContext
from rheplicant.config.sections.observation import (
    SiteFacts,
    _aux,
    _data,
    _environment,
    _extra,
    _freq_grid,
    _meta,
    _site,
    _time_facts,
)

FREQ = {"grid": {"linspace": {"start": 60.0, "stop": 85.0, "num": 8, "endpoint": True},
                 "unit": "MHz"}}
TIME = {"grid": {"arange": {"start": 0.0, "step": 2.0, "num": 16}, "unit": "s"}}


@pytest.fixture()
def context():
    return ResolutionContext(dtype="float32")


class TestTheGrids:
    def test_freq_comes_back_in_hz(self, context):
        grid = _freq_grid(FREQ, context)
        assert grid.shape == (8,)
        assert float(grid[0]) == pytest.approx(60e6)

    def test_freq_requires_a_frequency_unit(self, context):
        bad = {"grid": {"linspace": {"start": 60.0, "stop": 85.0, "num": 8,
                                     "endpoint": True}, "unit": "K"}}
        with pytest.raises(ConfigError, match="frequency"):
            _freq_grid(bad, context)
        with pytest.raises(ConfigError, match="frequency"):
            _freq_grid({"grid": {"list": [1.0, 2.0]}}, context)

    def test_freq_requires_grid(self, context):
        with pytest.raises(ConfigError, match="grid"):
            _freq_grid({}, context)

    def test_freq_sweeps_unknown_keys(self, context):
        with pytest.raises(ConfigError, match=r"\['gird'\]"):
            _freq_grid({**FREQ, "gird": 1}, context)

    def test_a_2d_grid_is_refused(self, context):
        with pytest.raises(ConfigError, match="1-D"):
            _freq_grid({"grid": {"zeros": [2, 2], "unit": "MHz"}}, context)

    def test_time_facts(self, context):
        facts = _time_facts(
            {**TIME,
             "epoch": {"value": 1785312000.0, "unit": "unix_s"},
             "integration_time": {"value": 2.0, "unit": "s"},
             "channel_width": {"value": 3.125, "unit": "MHz"}},
            context,
        )
        time_s, epoch, integration, width = facts
        assert time_s.shape == (16,)
        assert float(time_s[1]) == pytest.approx(2.0)
        assert epoch == pytest.approx(1785312000.0)
        assert integration == pytest.approx(2.0)
        assert width == pytest.approx(3.125e6)

    def test_the_epoch_takes_unix_s_and_nothing_else(self, context):
        with pytest.raises(ConfigError, match="unix_s"):
            _time_facts({**TIME, "epoch": {"value": 0.0, "unit": "s"}}, context)

    def test_time_extras_default_to_none(self, context):
        _, epoch, integration, width = _time_facts(TIME, context)
        assert epoch is None and integration is None and width is None


class TestMeta:
    def test_lists_become_tuples_and_scalars_pass(self):
        meta = _meta({"telescope": "RHINO", "chans": [1, 2]})
        assert meta == {"telescope": "RHINO", "chans": (1, 2)}

    def test_an_unhashable_value_is_refused_naming_the_key(self):
        with pytest.raises(ConfigError, match="observation.meta.blob"):
            _meta({"blob": {"a": 1}})

    def test_non_string_keys_are_refused(self):
        with pytest.raises(ConfigError, match="strings"):
            _meta({1: "x"})


class TestSite:
    def test_the_three_facts(self, context):
        site = _site(
            {"lat_deg": {"value": 53.2367, "unit": "deg"},
             "lon_deg": {"value": -2.3085, "unit": "deg"},
             "alt_m": {"value": 78.0, "unit": "m"}},
            context,
        )
        assert site == SiteFacts(lat_deg=pytest.approx(53.2367),
                                 lon_deg=pytest.approx(-2.3085),
                                 alt_m=pytest.approx(78.0))

    def test_everything_is_optional(self, context):
        assert _site({}, context) == SiteFacts(None, None, None)

    def test_dimensions_are_checked(self, context):
        with pytest.raises(ConfigError, match="angle"):
            _site({"lat_deg": {"value": 53.0, "unit": "m"}}, context)
        with pytest.raises(ConfigError, match="length"):
            _site({"alt_m": {"value": 78.0, "unit": "deg"}}, context)


class TestEnvironment:
    def test_celsius_arrives_as_kelvin(self, context):
        env = _environment({"temperature": {"value": 20.0, "unit": "celsius"}}, context)
        assert float(env.temperature) == pytest.approx(293.15)

    def test_humidity_declares_its_unit(self, context):
        env = _environment(
            {"humidity": {"value": 0.4, "unit": "dimensionless"}}, context
        )
        assert float(env.humidity) == pytest.approx(0.4)
        with pytest.raises(ConfigError, match="humidity"):
            _environment({"humidity": 0.4}, context)

    def test_extra_arrays_are_carried(self, context):
        env = _environment(
            {"extra": {"wind": {"list": [1.0, 2.0]}}}, context
        )
        assert env.extra["wind"].shape == (2,)

    def test_no_section_means_no_environment(self, context):
        assert _environment(None, context) is None


class TestExtraAuxData:
    def test_extra_resolves_value_nodes(self, context):
        extra = _extra({"my_switch": {"list": [0, 1, 0, 1]}}, context)
        assert extra["my_switch"].shape == (4,)

    def test_receiver_input_is_reserved_for_switching(self, context):
        with pytest.raises(ConfigError, match="switching"):
            _extra({"receiver_input": {"list": [0, 1]}}, context)

    def test_aux_takes_flags_and_makes_them_boolean(self, context):
        """The array forms cast to the run dtype (arrays.py `_finish`), so a
        0/1 float array is what a document can actually write; the loader
        casts exact 0/1 to bool and refuses anything else."""
        flags = _aux({"flags": {"full": {"shape": [4, 8], "value": 1.0}}},
                     context, n_time=4, n_freq=8)
        assert flags["flags"].dtype == jnp.bool_
        assert bool(flags["flags"][0, 0])
        with pytest.raises(ConfigError, match=r"\['banner'\]"):
            _aux({"banner": 1}, context, n_time=4, n_freq=8)
        with pytest.raises(ConfigError, match="TRUE = BAD"):
            _aux({"flags": {"full": {"shape": [4, 8], "value": 0.5}}},
                 context, n_time=4, n_freq=8)

    def test_flags_shape_is_checked(self, context):
        with pytest.raises(ConfigError, match=r"\(n_time, n_freq\)"):
            _aux({"flags": {"full": {"shape": [3, 8], "value": False}}},
                 context, n_time=4, n_freq=8)

    def test_data_shape_is_checked(self, context):
        data = _data({"zeros": [4, 8]}, context, n_time=4, n_freq=8)
        assert data.shape == (4, 8)
        with pytest.raises(ConfigError, match=r"\(n_time, n_freq\)"):
            _data({"zeros": [3, 8]}, context, n_time=4, n_freq=8)
