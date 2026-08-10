"""model: -> assemble/Pipeline: many shapes, compose, at, snapshot, pipeline."""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError, ResolutionContext
from rheplicant.config.sections.compose import build_model
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State
from rheplicant.radio import GainOperator

FREQ_HZ = jnp.linspace(60e6, 85e6, 8)
TIME_S = jnp.arange(0.0, 32.0, 2.0)
ORDER = ("antenna", "ambient", "hot")

GLOBAL_SIGNAL = {"depth": {"value": 0.5, "unit": "K"},
                 "centre": {"value": 75.0, "unit": "MHz"},
                 "width": {"value": 5.0, "unit": "MHz"}}
GAIN = {"gain": {"value": 1.1, "unit": "dimensionless"}}


@pytest.fixture()
def context():
    return ResolutionContext(freq=FREQ_HZ, time=TIME_S, dtype="float32",
                             switch_order=ORDER)


def state():
    return State(coords=Coordinates(time=TIME_S, freq=FREQ_HZ),
                 key=jax.random.key(0))


class TestTheGraphRoute:
    def test_a_minimal_model_assembles_and_runs(self, context):
        twin = build_model({"global_signal": GLOBAL_SIGNAL, "gain": GAIN},
                           context, switch_order=())
        assert set(twin.lit) == {"global_signal", "gain"}
        out = twin(state())
        assert out.data.shape == (16, 8)

    def test_an_unknown_node_is_refused_listing_the_known_ones(self, context):
        with pytest.raises(ConfigError, match="gian"):
            build_model({"gian": GAIN}, context, switch_order=())

    def test_a_junction_is_never_an_operator_slot(self, context):
        with pytest.raises(ConfigError, match="junction"):
            build_model({"astro_sum": {}, "gain": GAIN}, context,
                        switch_order=())

    def test_a_reserved_node_given_type_is_refused(self, context):
        with pytest.raises(ConfigError, match="reserved"):
            build_model({"beam": {"type": "GainOperator"}, "gain": GAIN},
                        context, switch_order=())

    def test_an_empty_model_is_refused(self, context):
        with pytest.raises(ConfigError, match="model"):
            build_model({}, context, switch_order=())


class TestManyShapes:
    def test_a_list_at_a_non_many_node_is_refused(self, context):
        with pytest.raises(ConfigError, match="single"):
            build_model({"gain": [GAIN, GAIN]}, context, switch_order=())

    def test_a_mapping_at_a_sum_node_is_refused(self, context):
        with pytest.raises(ConfigError, match="list"):
            build_model(
                {"foregrounds": {"amplitude": {"value": 2500.0, "unit": "K"},
                                 "spectral_index": 2.55,
                                 "ref_freq": {"value": 70.0, "unit": "MHz"}},
                 "gain": GAIN},
                context, switch_order=())

    def test_sum_entries_assemble_in_order(self, context):
        entry = {"amplitude": {"value": 2500.0, "unit": "K"},
                 "spectral_index": 2.55,
                 "ref_freq": {"value": 70.0, "unit": "MHz"}}
        twin = build_model({"foregrounds": [entry, entry]}, context,
                           switch_order=())
        assert "foregrounds" in twin.lit
        assert twin.instances  # two instances at one node are recorded

    def test_fan_keys_are_the_switch_order(self, context):
        loads = {"ambient": {"t_load": {"value": 300.0, "unit": "K"}},
                 "hot": {"t_load": {"value": 400.0, "unit": "K"}}}
        twin = build_model({"cal_loads": loads}, context, switch_order=ORDER)
        assert "cal_loads" in twin.lit

    def test_fan_keys_out_of_order_are_refused(self, context):
        loads = {"hot": {"t_load": {"value": 400.0, "unit": "K"}},
                 "ambient": {"t_load": {"value": 300.0, "unit": "K"}}}
        with pytest.raises(ConfigError, match="order"):
            build_model({"cal_loads": loads}, context, switch_order=ORDER)

    def test_cal_loads_without_a_switching_order_is_refused(self, context):
        loads = {"ambient": {"t_load": {"value": 300.0, "unit": "K"}}}
        with pytest.raises(ConfigError, match="switching"):
            build_model({"cal_loads": loads}, context, switch_order=())

    def test_a_chain_entry_needs_its_type(self, context):
        with pytest.raises(ConfigError, match="type:"):
            build_model(
                {"gain": GAIN,
                 "filters": [{"axis": 0, "low": 0.02, "high": 0.5}]},
                context, switch_order=())

    def test_a_filter_chain_builds(self, context):
        twin = build_model(
            {"gain": GAIN,
             "filters": [{"type": "FourierBandFilter", "axis": 0, "low": 0.02,
                          "high": 0.5, "mode": "extract"}]},
            context, switch_order=())
        assert "filters" in twin.lit


class TestCompose:
    def test_cascade_stages_are_addressable_by_name(self, context):
        twin = build_model(
            {"gain": {"compose": "cascade", "stages": [
                {"name": "gain_lna", "type": "GainOperator",
                 "gain": {"value": 1.0, "unit": "dimensionless"}},
                {"name": "gain_backend", "type": "GainOperator",
                 "gain": {"value": 1.1, "unit": "dimensionless"}},
            ]}, "global_signal": GLOBAL_SIGNAL},
            context, switch_order=())
        assert isinstance(twin["gain"]["gain_lna"], GainOperator)

    def test_cascade_at_a_source_node_is_refused(self, context):
        with pytest.raises(ConfigError, match="cascade"):
            build_model(
                {"uniform_sky": {"compose": "cascade", "stages": [
                    {"name": "a", "amplitude": {"value": 1.0, "unit": "K"}}]}},
                context, switch_order=())

    def test_sum_at_a_transform_node_is_refused(self, context):
        with pytest.raises(ConfigError, match="sum"):
            build_model(
                {"gain": {"compose": "sum", "stages": [
                    {"name": "a", "gain": {"value": 1.0,
                                           "unit": "dimensionless"}}]}},
                context, switch_order=())

    def test_stage_names_are_required(self, context):
        stages = [
            {"type": "GainOperator",
             "gain": {"value": 1.0, "unit": "dimensionless"}},
            {"name": "b", "type": "GainOperator",
             "gain": {"value": 1.1, "unit": "dimensionless"}},
        ]
        with pytest.raises(ConfigError, match="name:"):
            build_model({"gain": {"compose": "cascade", "stages": stages}},
                        context, switch_order=())

    def test_compose_without_stages_is_refused(self, context):
        with pytest.raises(ConfigError, match="stages"):
            build_model({"gain": {"compose": "cascade"}}, context,
                        switch_order=())


class TestSnapshotAndAt:
    def test_snapshot_before_preserves_the_raw_data(self, context):
        twin = build_model(
            {"global_signal": GLOBAL_SIGNAL,
             "gain": {**GAIN, "snapshot_before": "raw"}},
            context, switch_order=())
        out = twin(state())
        assert "snapshot/raw" in out.aux
        assert not jnp.allclose(out.aux["snapshot/raw"], out.data)

    def test_at_relocation_places_a_python_operator(self, context):
        twin = build_model(
            {"global_signal": GLOBAL_SIGNAL,
             "snapshot": {"python": "rheplicant:SnapshotOperator",
                          "name": "tap", "at": "snapshot"}},
            context, switch_order=())
        out = twin(state())
        assert "snapshot/tap" in out.aux

    def test_at_with_a_shipped_type_is_refused(self, context):
        with pytest.raises(ConfigError, match="at:"):
            build_model({"gain": {**GAIN, "at": "gain"}}, context,
                        switch_order=())

    def test_a_region_must_be_keyed_by_its_last_node(self, context):
        with pytest.raises(ConfigError, match="LAST"):
            build_model(
                {"global_signal": GLOBAL_SIGNAL,
                 "noise_wave": {"python": "rheplicant:SnapshotOperator",
                                "name": "tap",
                                "at": ["noise_wave", "cw_tone"]}},
                context, switch_order=())


class TestAcknowledgeDoubleCount:
    SPILL = {"sky_fraction": 0.9, "t_ground": {"value": 290.0, "unit": "K"}}
    PICKUP = {"coupling": 0.05, "t_ground": {"value": 290.0, "unit": "K"}}

    def test_both_lit_without_the_acknowledgement_is_refused(self, context):
        with pytest.raises(ConfigError, match="acknowledge_double_count"):
            build_model({"beam_spill": self.SPILL,
                         "ground_pickup": self.PICKUP}, context,
                        switch_order=())

    def test_the_acknowledgement_lets_it_build(self, context):
        # beam_spill is a transform on the astro branch (test_beam_spill.py's
        # TestPlacement: "it lands between the astro entrance and the antenna
        # sum") -- it needs a live upstream astro source or assemble() refuses
        # it as an unsourced branch feeding t_ant_sum, independent of this
        # section's own acknowledge_double_count gate. uniform_sky supplies
        # that source through the reserved (skip-as-identity) beam node.
        twin = build_model(
            {"uniform_sky": {"amplitude": {"value": 1.0, "unit": "K"}},
             "beam_spill": self.SPILL, "ground_pickup": self.PICKUP,
             "acknowledge_double_count": True},
            context, switch_order=())
        assert {"beam_spill", "ground_pickup"} <= set(twin.lit)

    def test_a_truthy_non_true_acknowledgement_is_refused(self, context):
        """The schema's key is literal true -- 'yes' or 1 is not a statement."""
        with pytest.raises(ConfigError, match="acknowledge_double_count"):
            build_model({"beam_spill": self.SPILL,
                         "ground_pickup": self.PICKUP,
                         "acknowledge_double_count": "yes"},
                        context, switch_order=())


class TestKindPipeline:
    STAGES = [
        {"name": "sky", "type": "SkyOperator",
         "amplitude": {"value": 100.0, "unit": "K"}},
        {"name": "gain", "type": "GainOperator",
         "gain": {"value": 1.1, "unit": "dimensionless"}},
    ]

    def test_a_pipeline_builds_and_runs(self, context):
        from rheplicant.radio import SkyOperator

        twin = build_model({"kind": "pipeline", "stages": self.STAGES},
                           context, switch_order=())
        assert isinstance(twin, Pipeline)
        assert isinstance(twin["sky"], SkyOperator)
        out = twin(state())
        assert out.data.shape == (16, 8)

    def test_a_stage_without_a_class_is_refused(self, context):
        with pytest.raises(ConfigError, match="type"):
            build_model({"kind": "pipeline",
                         "stages": [{"name": "sky",
                                     "amplitude": {"value": 1.0, "unit": "K"}}]},
                        context, switch_order=())

    def test_an_unknown_kind_is_refused(self, context):
        with pytest.raises(ConfigError, match="kind"):
            build_model({"kind": "tree", "stages": []}, context,
                        switch_order=())
