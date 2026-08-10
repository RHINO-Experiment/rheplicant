"""inference.noise: four kinds, one sigma the likelihood and generator share."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.noise import build_noise, decided_noise, freeze_sigma
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
            "integration_time_s": 1.0, "floor_k": 0.0}

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
