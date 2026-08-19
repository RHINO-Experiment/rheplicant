"""inference.noise: four kinds, one sigma the likelihood and generator share."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.noise import (
    build_noise,
    decided_noise,
    freeze_sigma,
    freeze_sigmas,
)
from rheplicant.config.sections.observation import build_observation
from rheplicant.config.sections.runtime import build_runtime
from tests.config.inference_helpers import context


def observation(**time_extras):
    section = {
        "freq": {"grid": {"linspace": {"start": 60.0, "stop": 85.0, "num": 8,
                                       "endpoint": True}, "unit": "MHz"}},
        "time": {"grid": {"arange": {"start": 0.0, "step": 2.0, "num": 16},
                          "unit": "s"}, **time_extras},
    }
    build, _ = build_observation(section,
                                 runtime=build_runtime({"seed": 1}))
    return build


FACTS = dict(integration_time={"value": 2.0, "unit": "s"},
             channel_width={"value": 3.125, "unit": "MHz"})


class TestKinds:
    def test_none_is_the_default_and_decides_nothing(self):
        build = build_noise(None, observation=observation(), context=context())
        assert build.kind == "none"
        assert decided_noise(build) is None

    def test_homoscedastic_builds_the_packages_model(self):
        from rheplicant.inference import HomoscedasticNoise

        build = build_noise({"kind": "homoscedastic",
                             "sigma": {"value": 0.5, "unit": "K"}},
                            observation=observation(), context=context())
        assert isinstance(build.model, HomoscedasticNoise)
        assert float(build.model.sigma) == pytest.approx(0.5)

    def test_radiometer_reads_the_observation_facts(self):
        from rheplicant.inference import RadiometerNoise

        build = build_noise({"kind": "radiometer",
                             "channel_width": {"from": "observation"},
                             "integration_time": {"from": "observation"},
                             "include_logdet": True},
                            observation=observation(**FACTS),
                            context=context())
        assert isinstance(build.model, RadiometerNoise)
        assert build.model.channel_width == pytest.approx(3.125e6)
        assert build.model.integration_time == pytest.approx(2.0)
        assert build.include_logdet is True

    def test_from_observation_without_the_declaration_is_refused(self):
        with pytest.raises(ConfigError, match="observation.time.channel_width"):
            build_noise({"kind": "radiometer",
                         "channel_width": {"from": "observation"},
                         "integration_time": {"value": 2.0, "unit": "s"},
                         "include_logdet": True},
                        observation=observation(), context=context())

    def test_explicit_facts_are_dimension_checked_value_nodes(self):
        build = build_noise({"kind": "radiometer",
                             "channel_width": {"value": 3.125, "unit": "MHz"},
                             "integration_time": {"value": 2.0, "unit": "s"},
                             "include_logdet": False},
                            observation=observation(), context=context())
        assert build.model.channel_width == pytest.approx(3.125e6)

    def test_an_explicit_fact_in_the_wrong_dimension_is_refused(self):
        # A width in seconds converts cleanly through resolve_value, so only
        # the dimension check can tell it from a bandwidth.
        with pytest.raises(ConfigError, match="bandwidth"):
            build_noise({"kind": "radiometer",
                         "channel_width": {"value": 2.0, "unit": "s"},
                         "integration_time": {"value": 2.0, "unit": "s"},
                         "include_logdet": True},
                        observation=observation(), context=context())

    def test_an_integer_sigma_lands_in_the_runs_dtype(self):
        build = build_noise({"kind": "homoscedastic", "sigma": 2},
                            observation=observation(), context=context())
        assert build.model.sigma.dtype == jnp.float32

    def test_an_unknown_kind_is_refused_listing_the_four(self):
        with pytest.raises(ConfigError, match="radiometer_frozen"):
            build_noise({"kind": "gaussian"}, observation=observation(),
                        context=context())


class TestIncludeLogdet:
    def test_required_on_radiometer_with_no_default(self):
        with pytest.raises(ConfigError, match="include_logdet"):
            build_noise({"kind": "radiometer",
                         "channel_width": {"value": 3.125, "unit": "MHz"},
                         "integration_time": {"value": 2.0, "unit": "s"}},
                        observation=observation(), context=context())

    def test_refused_on_a_prediction_independent_kind(self):
        with pytest.raises(ConfigError, match="include_logdet"):
            build_noise({"kind": "homoscedastic",
                         "sigma": {"value": 0.5, "unit": "K"},
                         "include_logdet": True},
                        observation=observation(), context=context())

    def test_a_truthy_non_bool_is_refused(self):
        # include_logdet: 1 is a lost declaration, not a yes (A49).
        with pytest.raises(ConfigError, match="include_logdet"):
            build_noise({"kind": "radiometer",
                         "channel_width": {"value": 3.125, "unit": "MHz"},
                         "integration_time": {"value": 2.0, "unit": "s"},
                         "include_logdet": 1},
                        observation=observation(), context=context())


class TestAxis:
    def test_a_1d_sigma_without_axis_is_check_a26(self):
        with pytest.raises(ConfigError, match="axis"):
            build_noise({"kind": "homoscedastic",
                         "sigma": {"ones": ["n_freq"], "unit": "K"}},
                        observation=observation(), context=context())

    def test_axis_freq_makes_a_row(self):
        build = build_noise({"kind": "homoscedastic",
                             "sigma": {"ones": ["n_freq"], "unit": "K"},
                             "axis": "freq"},
                            observation=observation(), context=context())
        assert build.model.sigma.shape == (1, 8)

    def test_axis_time_makes_a_column(self):
        build = build_noise({"kind": "homoscedastic",
                             "sigma": {"ones": ["n_time"], "unit": "K"},
                             "axis": "time"},
                            observation=observation(), context=context())
        assert build.model.sigma.shape == (16, 1)

    def test_axis_on_a_scalar_sigma_is_refused(self):
        with pytest.raises(ConfigError, match="1-D"):
            build_noise({"kind": "homoscedastic",
                         "sigma": {"value": 0.5, "unit": "K"},
                         "axis": "freq"},
                        observation=observation(), context=context())


class TestFlags:
    def test_flags_wrap_the_model_in_flagged_noise(self):
        from rheplicant.inference import FlaggedNoise, HomoscedasticNoise

        section = {
            "freq": {"grid": {"linspace": {"start": 60.0, "stop": 85.0,
                                           "num": 8, "endpoint": True},
                              "unit": "MHz"}},
            "time": {"grid": {"arange": {"start": 0.0, "step": 2.0,
                                         "num": 16}, "unit": "s"}},
            "aux": {"flags": {"zeros": ["n_time", "n_freq"]}},
        }
        obs, _ = build_observation(section, runtime=build_runtime({"seed": 1}))
        build = build_noise({"kind": "homoscedastic",
                             "sigma": {"value": 0.5, "unit": "K"},
                             "flags": {"from": "observation"}},
                            observation=obs, context=context())
        assert isinstance(build.model, FlaggedNoise)
        # base and flags by slot, not just the wrapper type -- swapped
        # arguments would still construct.
        assert isinstance(build.model.base, HomoscedasticNoise)
        assert build.model.flags.shape == (16, 8)

    def test_flags_without_the_aux_declaration_are_refused(self):
        with pytest.raises(ConfigError, match="observation.aux.flags"):
            build_noise({"kind": "homoscedastic",
                         "sigma": {"value": 0.5, "unit": "K"},
                         "flags": {"from": "observation"}},
                        observation=observation(), context=context())


class TestFrozen:
    def frozen(self, **extra):
        # 16 Hz and 1 s, asymmetric on purpose: 1/sqrt(w*tau) is symmetric
        # under a width/tau swap, so equal facts would hide one.
        spec = {"kind": "radiometer_frozen", "source": "observed",
                "channel_width": {"value": 16.0, "unit": "Hz"},
                "integration_time": {"value": 1.0, "unit": "s"}}
        spec.update(extra)
        return build_noise(spec, observation=observation(), context=context())

    def test_the_facts_land_in_their_own_slots(self):
        assert self.frozen().frozen == {
            "source": "observed", "channel_width_hz": 16.0,
            "integration_time_s": 1.0, "floor": 0.0}

    def test_the_source_is_required_by_name(self):
        with pytest.raises(ConfigError, match="prediction_at_init"):
            build_noise({"kind": "radiometer_frozen",
                         "channel_width": {"value": 4.0, "unit": "Hz"},
                         "integration_time": {"value": 4.0, "unit": "s"}},
                        observation=observation(), context=context())

    def test_the_sigma_is_decided_from_the_magnitude(self):
        # 1/sqrt(16*1) = 0.25; a NEGATIVE reference pins |X|, where d(1+fw)
        # and a reused std() would otherwise agree to roundoff (Plan 0's
        # lesson: sample where the wrong implementation differs).
        build = freeze_sigma(self.frozen(),
                             jnp.asarray([[-100.0, 4.0]]))
        assert build.sigma.shape == (1, 2)
        assert float(build.sigma[0, 0]) == pytest.approx(25.0)
        assert float(build.sigma[0, 1]) == pytest.approx(1.0)
        assert decided_noise(build) is build.sigma

    def test_the_floor_clips_the_magnitude_first(self):
        build = freeze_sigma(self.frozen(floor={"value": 8.0, "unit": "K"}),
                             jnp.asarray([[-100.0, 4.0]]))
        assert float(build.sigma[0, 1]) == pytest.approx(2.0)

    def test_an_unfrozen_build_refuses_to_decide(self):
        with pytest.raises(ConfigError, match="frozen"):
            decided_noise(self.frozen())

    def test_freeze_sigmas_decides_one_per_observation(self):
        # 1/sqrt(16*1) = 0.25 on both, off DIFFERENT references: the fan is
        # about which array each sigma is decided FROM, so two references
        # sharing a magnitude could not tell it from freezing once.  The
        # negative entries are deliberate -- |X| is what freeze_sigma takes.
        build = freeze_sigmas(self.frozen(),
                              {"primary": jnp.asarray([[-100.0, 4.0]]),
                               "night": jnp.asarray([[8.0, -40.0]])},
                              primary="primary")
        assert sorted(build.by_observation) == ["night", "primary"]
        assert float(build.by_observation["primary"][0, 0]) == pytest.approx(
            25.0)
        assert float(build.by_observation["primary"][0, 1]) == pytest.approx(
            1.0)
        assert float(build.by_observation["night"][0, 0]) == pytest.approx(2.0)
        assert float(build.by_observation["night"][0, 1]) == pytest.approx(
            10.0)

    def test_the_floor_clips_every_observations_sigma(self):
        # The floor is the half of `freeze_sigma`'s arithmetic that reusing
        # it is meant to buy, and the reuse is a claim no other test here can
        # see: every other `freeze_sigmas` call in this module declares no
        # floor, so an inline reimplementation applying the fractional factor
        # and dropping the clip passed all of tests/config (measured).
        # `floor:` is a legal radiometer_frozen key, so the path is
        # user-reachable on a two-observation document.
        build = freeze_sigmas(self.frozen(floor={"value": 8.0, "unit": "K"}),
                              {"primary": jnp.asarray([[-100.0, 4.0]]),
                               "night": jnp.asarray([[2.0, -40.0]])},
                              primary="primary")
        per = build.by_observation
        # Each reference dips BELOW the floor in one channel: |4.0| -> 8.0 ->
        # 2.0 and |2.0| -> 8.0 -> 2.0, where unclipped they would be 1.0 and
        # 0.5.  Both entries are pinned, so a fan that clipped only the
        # primary's reference fails on night.
        assert float(per["primary"][0, 1]) == pytest.approx(2.0)
        assert float(per["night"][0, 0]) == pytest.approx(2.0)
        # And the channels ABOVE the floor keep their own magnitude, so this
        # cannot be satisfied by clamping every channel to the floor.
        assert float(per["primary"][0, 0]) == pytest.approx(25.0)
        assert float(per["night"][0, 1]) == pytest.approx(10.0)

    def test_the_default_sigma_is_the_primarys_own_array(self):
        # `decided_noise` takes no run, so it must keep answering with the
        # primary's -- and with the SAME array, or `decided_noise(build) is
        # build.sigma` above stops saying anything.  `is`, not allclose:
        # identity is what makes a one-observation document bit-identical.
        build = freeze_sigmas(self.frozen(),
                              {"primary": jnp.asarray([[-100.0, 4.0]]),
                               "night": jnp.asarray([[8.0, -40.0]])},
                              primary="primary")
        assert build.sigma is build.by_observation["primary"]
        assert decided_noise(build) is build.sigma

    def test_one_observation_freezes_exactly_what_freeze_sigma_did(self):
        # The regression claim, as an assertion: with one entry the fan
        # returns the single-reference answer under the single name.
        reference = jnp.asarray([[-100.0, 4.0]])
        one = freeze_sigma(self.frozen(), reference)
        fanned = freeze_sigmas(self.frozen(), {"primary": reference},
                               primary="primary")
        assert list(fanned.by_observation) == ["primary"]
        assert jnp.array_equal(fanned.sigma, one.sigma)

    def test_an_unfanned_build_carries_no_mapping(self):
        # `by_observation` is None, not {} -- `_noise` branches on it, and an
        # empty dict is falsy in exactly the way that makes the two branches
        # agree by accident.
        assert self.frozen().by_observation is None
        assert freeze_sigma(self.frozen(),
                            jnp.asarray([[1.0]])).by_observation is None
