"""inference: the section orchestrator -- checks, trainable, truth, sequence."""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.inference import CheckSpec, InferenceBuild, build_inference
from rheplicant.config.sections.observation import build_observation
from rheplicant.config.sections.runtime import build_runtime
from tests.config.inference_helpers import NOISY_MODEL, context, state, twin


def observation(**extras):
    section = {
        "freq": {"grid": {"linspace": {"start": 60.0, "stop": 85.0, "num": 8,
                                       "endpoint": True}, "unit": "MHz"}},
        "time": {"grid": {"arange": {"start": 0.0, "step": 2.0, "num": 16},
                          "unit": "s"}},
    }
    section.update(extras)
    build, _ = build_observation(section, runtime=build_runtime({"seed": 1}))
    return build


def infer(section, model=None, ctx=None):
    ctx = ctx or context()
    return build_inference(section, twin=twin(model, ctx), state=state(),
                           observation=observation(), context=ctx)


PARAMS = {"g": {"init": 1.0, "linear": True, "into": "gain.gain"}}


class TestSequence:
    def test_none_still_yields_a_build_with_the_twin_as_fit_twin(self):
        build = infer(None)
        assert isinstance(build, InferenceBuild)
        assert build.space is None
        assert build.noise.kind == "none"
        assert build.observed is None

    def test_the_whole_section_composes(self):
        build = infer({
            "twin": {"without": ["noise"]},
            "parameters": PARAMS,
            "noise": {"kind": "homoscedastic",
                      "sigma": {"value": 0.5, "unit": "K"}},
            "observed": {"from": "simulation", "at": {"g": 1.5}},
        }, model=NOISY_MODEL)
        assert "noise" not in build.fit_twin.lit
        assert build.space is not None
        assert build.observed.entries["primary"].shape == (16, 8)

    def test_bindings_resolve_against_the_repaired_twin(self):
        build = infer({
            "twin": {"replace": {"gain": {"gain": {"value": 1.0,
                                                   "unit": "dimensionless"}}}},
            "parameters": {"d": {"init": 0.5,
                                 "into": "global_signal.depth"}},
        })
        assert build.replaced == ("gain",)

    def test_a_binding_into_a_removed_node_is_refused(self):
        """Paths resolve against the FIT twin: a latent cannot bind into a
        node the twin repair just removed."""
        with pytest.raises(ConfigError, match=r"noise\.sigma"):
            infer({"twin": {"without": ["noise"]},
                   "parameters": {"s": {"init": 0.1, "into": "noise.sigma"}}},
                  model=NOISY_MODEL)

    def test_a_non_list_bindings_section_is_refused(self):
        """bindings: is a LIST -- a scalar or a dashless single mapping both
        fail as ConfigError, not as a raw TypeError or a blamed first key."""
        with pytest.raises(ConfigError, match="LIST"):
            infer({"parameters": {"g": {"init": 1.0}}, "bindings": 42})
        with pytest.raises(ConfigError, match="LIST"):
            infer({"parameters": {"g": {"init": 1.0}},
                   "bindings": {"latents": ["g"], "into": "gain.gain"}})

    def test_npe_is_plan_2c_by_name(self):
        with pytest.raises(ConfigError, match="2C"):
            infer({"npe": {"bank": {"n_simulations": 8}}})

    def test_unknown_inference_keys_are_swept(self):
        with pytest.raises(ConfigError, match="observations"):
            infer({"observations": {}})


class TestFrozenSequencing:
    NOISE = {"kind": "radiometer_frozen", "source": "observed",
             "channel_width": {"value": 4.0, "unit": "Hz"},
             "integration_time": {"value": 4.0, "unit": "s"}}

    def test_source_observed_freezes_from_the_primary(self):
        build = infer({"parameters": PARAMS, "noise": self.NOISE,
                       "observed": {"from": "simulation"}})
        assert build.noise.sigma is not None
        assert build.noise.sigma.shape == (16, 8)

    def test_source_observed_without_observed_is_refused(self):
        with pytest.raises(ConfigError, match="observed"):
            infer({"parameters": PARAMS, "noise": self.NOISE})

    def test_source_prediction_at_init_evaluates_the_fit_twin_once(self):
        build = infer({"parameters": PARAMS,
                       "noise": {**self.NOISE,
                                 "source": "prediction_at_init"}})
        assert build.noise.sigma is not None

    def test_prediction_at_init_evaluates_the_repaired_twin(self):
        """With the stochastic node repaired away, the frozen sigma comes from
        the FIT twin's deterministic prediction -- |prediction|/sqrt(w*tau)."""
        import jax.numpy as jnp

        build = infer({"twin": {"without": ["noise"]},
                       "parameters": PARAMS,
                       "noise": {**self.NOISE,
                                 "source": "prediction_at_init"}},
                      model=NOISY_MODEL)
        bound = build.space.bind(build.fit_twin,
                                 dict(build.space.initial_values()))
        expected = jnp.abs(bound(state()).data) * 0.25  # 1/sqrt(4 Hz * 4 s)
        assert jnp.allclose(build.noise.sigma, expected)

    def test_source_observed_reads_the_primary_among_several(self):
        """Several named observations: the sigma is decided from the entry
        NAMED primary, not whichever the document happened to list first."""
        import jax.numpy as jnp

        build = infer({"parameters": PARAMS, "noise": self.NOISE,
                       "observed": {"other": {"from": "simulation"},
                                    "primary": {"from": "simulation",
                                                "at": {"g": 1.5}}}})
        assert build.observed.primary == "primary"
        expected = jnp.abs(build.observed.entries["primary"]) * 0.25
        decoy = jnp.abs(build.observed.entries["other"]) * 0.25
        assert jnp.allclose(build.noise.sigma, expected)
        assert not jnp.allclose(build.noise.sigma, decoy)


class TestChecks:
    def test_modes_and_reasons(self):
        build = infer({"checks": {
            "identifiability": {"mode": "refuse", "rtol": 1.0e-8,
                                "report": True},
            "linearity": {"mode": "refuse"},
            "prior_sensitivity": {"mode": "skip", "reason": "campaign"}}})
        assert build.checks["identifiability"] == CheckSpec(
            mode="refuse", report=True, reason=None, rtol=1.0e-8)
        assert build.checks["prior_sensitivity"].reason == "campaign"

    def test_skip_without_its_reason_is_check_a37(self):
        with pytest.raises(ConfigError, match="reason"):
            infer({"checks": {"linearity": {"mode": "skip"}}})

    def test_an_unknown_check_or_mode_is_refused(self):
        with pytest.raises(ConfigError, match="identifiability"):
            infer({"checks": {"stationarity": {"mode": "warn"}}})
        with pytest.raises(ConfigError, match="report"):
            infer({"checks": {"linearity": {"mode": "sometimes"}}})

    def test_rtol_belongs_to_identifiability_alone(self):
        with pytest.raises(ConfigError, match="rtol"):
            infer({"checks": {"linearity": {"mode": "warn", "rtol": 1e-8}}})


class TestTrainable:
    def test_leaves_compile_to_a_filter_spec(self):
        import equinox as eqx

        build = infer({"trainable": {"leaves": ["gain.gain"]}})
        params, _ = eqx.partition(build.fit_twin, build.trainable)
        import jax

        assert len([x for x in jax.tree.leaves(params) if x is not None]) == 1

    def test_nodes_take_every_inexact_leaf_under_them(self):
        import equinox as eqx
        import jax

        build = infer({"trainable": {"nodes": ["global_signal"]}})
        params, _ = eqx.partition(build.fit_twin, build.trainable)
        assert len([x for x in jax.tree.leaves(params)
                    if x is not None]) == 3

    def test_all_true_is_every_inexact_array(self):
        import equinox as eqx

        build = infer({"trainable": {"all": True}})
        assert build.trainable is eqx.is_inexact_array

    def test_all_true_with_a_subset_is_a_contradiction(self):
        with pytest.raises(ConfigError, match="all"):
            infer({"trainable": {"all": True, "leaves": ["gain.gain"]}})

    def test_routes_resolve_against_the_repaired_twin(self):
        """nodes: and leaves: compile on the FIT twin -- a spec built over
        the full twin would not even share the repaired tree's structure."""
        import equinox as eqx
        import jax

        build = infer({"twin": {"without": ["noise"]},
                       "trainable": {"nodes": ["global_signal"],
                                     "leaves": ["gain.gain"]}},
                      model=NOISY_MODEL)
        params, _ = eqx.partition(build.fit_twin, build.trainable)
        assert len([x for x in jax.tree.leaves(params)
                    if x is not None]) == 4

    def test_an_unknown_node_or_leaf_fails_fast(self):
        with pytest.raises(KeyError, match="rfi_field"):
            infer({"trainable": {"nodes": ["rfi_field"]}})
        with pytest.raises(ConfigError):
            infer({"trainable": {"leaves": ["gain.n_bits"]}})


class TestTruth:
    def test_at_wins_and_identity_leaves_derive(self):
        build = infer({
            "parameters": {**PARAMS,
                           "d": {"init": 0.1,
                                 "into": "global_signal.depth"}},
            "observed": {"from": "simulation", "at": {"g": 1.5}},
        })
        assert float(build.truth["g"]) == pytest.approx(1.5)
        assert float(build.truth["d"]) == pytest.approx(0.5)  # the leaf value

    def test_a_transformed_latent_is_omitted_with_its_reason(self):
        build = infer({
            "parameters": {"log_g": {"init": 0.0, "into": "gain.gain",
                                     "transform": "exp"}},
            "observed": {"from": "simulation"},
        })
        assert "log_g" not in build.truth
        assert "transform" in build.truth_omitted["log_g"]

    def test_a_fanned_identity_latent_is_omitted_with_the_fan_reason(self):
        """One latent tied identically into two leaves: no single leaf holds
        its truth, and the omission says so rather than blaming transform
        None."""
        build = infer({
            "parameters": {"d": {"init": 0.5,
                                 "into": ["global_signal.depth",
                                          "gain.gain"]}},
            "observed": {"from": "simulation"},
        })
        assert "d" not in build.truth
        assert "several leaves" in build.truth_omitted["d"]

    def test_the_truth_section_overrides_everything(self):
        build = infer({
            "parameters": PARAMS,
            "observed": {"from": "simulation", "at": {"g": 1.5}},
            "truth": {"g": 1.7},
        })
        assert float(build.truth["g"]) == pytest.approx(1.7)

    def test_a_truth_override_clears_the_omission_record(self):
        """A latent omitted for its transform stops being omitted the moment
        truth: declares it -- one name never sits in both dicts."""
        build = infer({
            "parameters": {"log_g": {"init": 0.1, "into": "gain.gain",
                                     "transform": "exp"}},
            "observed": {"from": "simulation"},
            "truth": {"log_g": 0.3},
        })
        assert float(build.truth["log_g"]) == pytest.approx(0.3)
        assert "log_g" not in build.truth_omitted

    def test_file_data_derives_no_truth(self):
        build = infer({"parameters": PARAMS})
        assert build.truth == {}
