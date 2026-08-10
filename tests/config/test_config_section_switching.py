"""observation.switching -> coords.extra['receiver_input'] + the order."""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError, ResolutionContext
from rheplicant.config.sections.switching import SwitchingBuild, compile_switching

ORDER = ["antenna", "ambient", "hot", "noise_source"]


@pytest.fixture()
def context():
    return ResolutionContext(dtype="float32")


class TestModeNone:
    def test_the_default(self, context):
        build = compile_switching(None, context, n_time=8)
        assert build == SwitchingBuild(order=(), receiver_input=None)

    def test_none_takes_no_other_keys(self, context):
        with pytest.raises(ConfigError, match=r"\['order'\]"):
            compile_switching({"mode": "none", "order": ORDER}, context, n_time=8)


class TestTheOrder:
    def test_index_zero_is_the_literal_antenna(self, context):
        with pytest.raises(ConfigError, match="antenna"):
            compile_switching({"mode": "cycle", "order": ["ambient", "hot"]},
                              context, n_time=8)

    def test_labels_appear_once(self, context):
        with pytest.raises(ConfigError, match="once"):
            compile_switching(
                {"mode": "cycle", "order": ["antenna", "hot", "hot"]},
                context, n_time=8)

    def test_order_is_required_for_a_cycle(self, context):
        with pytest.raises(ConfigError, match="order"):
            compile_switching({"mode": "cycle"}, context, n_time=8)


class TestRoundRobin:
    def test_dwell_one(self, context):
        build = compile_switching({"mode": "cycle", "order": ORDER}, context,
                                  n_time=8)
        assert build.order == tuple(ORDER)
        np.testing.assert_array_equal(
            np.asarray(build.receiver_input), [0, 1, 2, 3, 0, 1, 2, 3])
        assert jnp.issubdtype(build.receiver_input.dtype, jnp.integer)

    def test_dwell_two(self, context):
        build = compile_switching(
            {"mode": "cycle", "order": ORDER,
             "dwell": {"value": 2, "unit": "samples"}},
            context, n_time=8)
        np.testing.assert_array_equal(
            np.asarray(build.receiver_input), [0, 0, 1, 1, 2, 2, 3, 3])

    def test_dwell_must_be_a_positive_sample_count(self, context):
        with pytest.raises(ConfigError, match="dwell"):
            compile_switching({"mode": "cycle", "order": ORDER, "dwell": 0},
                              context, n_time=8)
        with pytest.raises(ConfigError, match="samples"):
            compile_switching(
                {"mode": "cycle", "order": ORDER,
                 "dwell": {"value": 2.0, "unit": "s"}},
                context, n_time=8)


class TestExplicitIndex:
    def test_an_index_array_is_taken_as_written(self, context):
        build = compile_switching(
            {"mode": "cycle", "order": ["antenna", "ambient"],
             "index": {"list": [0, 1, 1, 0]}},
            context, n_time=4)
        np.testing.assert_array_equal(np.asarray(build.receiver_input),
                                      [0, 1, 1, 0])

    def test_out_of_range_indices_are_refused(self, context):
        with pytest.raises(ConfigError, match="index"):
            compile_switching(
                {"mode": "cycle", "order": ["antenna", "ambient"],
                 "index": {"list": [0, 2, 0, 0]}},
                context, n_time=4)

    def test_a_wrong_length_index_is_refused(self, context):
        with pytest.raises(ConfigError, match=r"\(n_time,\)"):
            compile_switching(
                {"mode": "cycle", "order": ["antenna", "ambient"],
                 "index": {"list": [0, 1]}},
                context, n_time=4)

    def test_a_fractional_index_is_refused_not_truncated(self, context):
        with pytest.raises(ConfigError, match="non-integer"):
            compile_switching(
                {"mode": "cycle", "order": ["antenna", "ambient"],
                 "index": {"list": [0.0, 0.5, 1.0, 0.0]}},
                context, n_time=4)

    def test_index_and_round_robin_together_are_refused(self, context):
        with pytest.raises(ConfigError, match="one thing"):
            compile_switching(
                {"mode": "cycle", "order": ["antenna", "ambient"],
                 "cycle": "round_robin", "index": {"list": [0, 1, 1, 0]}},
                context, n_time=4)


class TestCycleNone:
    def test_cycle_none_is_refused_with_the_route(self, context):
        with pytest.raises(ConfigError, match=r"switching: \{mode: none\}"):
            compile_switching(
                {"mode": "cycle", "order": ORDER, "cycle": "none"},
                context, n_time=8)
