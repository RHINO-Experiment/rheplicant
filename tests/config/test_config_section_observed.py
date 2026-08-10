"""inference.observed: simulation, file, several observations, and the seed."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.noise import build_noise
from rheplicant.config.sections.observed import build_observed
from rheplicant.config.sections.parameters import parse_latents
from rheplicant.config.sections.transforms import build_space
from rheplicant.config.sections.twin import build_fit_twin
from rheplicant.inference import HomoscedasticNoise, RadiometerNoise
from tests.config.inference_helpers import context, state, twin


def harness(model=None, seeds=None, base_dir=None):
    ctx = context(**({"seeds": seeds} if seeds else {}),
                  **({"base_dir": base_dir} if base_dir else {}))
    full = twin(model, ctx)
    space = build_space(
        parse_latents({"g": {"init": 1.0, "linear": True,
                             "into": "gain.gain"}}, ctx),
        None, None, fit_twin=full, replaced=(), context=ctx)
    return ctx, full, space


class _FakeObservation:
    integration_time_s = 2.0
    channel_width_hz = 3.125e6
    aux: dict = {}


def build(spec, *, ctx, full, space, noise=None):
    return build_observed(spec, twin=full, fit_twin=full, space=space,
                          noise=noise or build_noise(
                              None, observation=_FakeObservation(),
                              context=ctx),
                          state=state(), observation=_FakeObservation(),
                          context=ctx)


class TestSimulation:
    def test_at_binds_the_truth_before_evaluating(self):
        ctx, full, space = harness()
        observed = build({"from": "simulation",
                          "at": {"g": 2.0}}, ctx=ctx, full=full, space=space)
        reference = space.bind(full, {"g": jnp.asarray(2.0)})(state()).data
        assert observed.primary == "primary"
        assert jnp.allclose(observed.entries["primary"], reference)
        assert observed.at["primary"]["g"] == pytest.approx(2.0)

    def test_at_an_unknown_latent_is_refused_listing_the_names(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="ghost"):
            build({"from": "simulation", "at": {"ghost": 2.0}},
                  ctx=ctx, full=full, space=space)

    def test_at_without_parameters_is_refused(self):
        ctx, full, _ = harness()
        with pytest.raises(ConfigError, match="parameters"):
            build({"from": "simulation", "at": {"g": 2.0}},
                  ctx=ctx, full=full, space=None)

    def test_twin_must_be_full_or_fit(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="full"):
            build({"from": "simulation", "twin": "both"},
                  ctx=ctx, full=full, space=space)

    def test_twin_fit_evaluates_the_repaired_twin_and_records_it(self):
        # The fit twin here is DISTINGUISHABLE from the full one (a deeper
        # trough), so twin: fit reading the full twin anyway cannot pass.
        ctx, full, space = harness()
        deeper = {"depth": {"value": 0.9, "unit": "K"},
                  "centre": {"value": 75.0, "unit": "MHz"},
                  "width": {"value": 5.0, "unit": "MHz"}}
        fit, _ = build_fit_twin({"replace": {"global_signal": deeper}},
                                full, ctx)
        observed = build_observed(
            {"from": "simulation", "twin": "fit"}, twin=full, fit_twin=fit,
            space=space, noise=build_noise(None,
                                           observation=_FakeObservation(),
                                           context=ctx),
            state=state(), observation=_FakeObservation(), context=ctx)
        init = dict(space.initial_values())
        assert jnp.allclose(observed.entries["primary"],
                            space.bind(fit, init)(state()).data)
        assert not jnp.allclose(observed.entries["primary"],
                                space.bind(full, init)(state()).data)
        assert observed.records["primary"]["twin"] == "fit"

    def test_a_clean_simulation_is_evaluated_at_the_declared_init(self):
        # init g=1.0 differs from the twin's declared gain 1.1, so a build
        # that skips binding the initial values shows up here.
        ctx, full, space = harness()
        observed = build({"from": "simulation"}, ctx=ctx, full=full,
                         space=space)
        reference = space.bind(full, dict(space.initial_values()))(
            state()).data
        assert jnp.allclose(observed.entries["primary"], reference)
        assert not jnp.allclose(observed.entries["primary"],
                                full(state()).data)


class TestRealise:
    def spec(self, kind, seed="runtime.seeds.observed_noise", **extra):
        realise = {"kind": kind, **extra}
        if seed is not None:
            realise["seed"] = {"from": seed}
        return {"from": "simulation", "realise": realise}

    def test_homoscedastic_scatter_is_reproducible_from_its_named_seed(self):
        ctx, full, space = harness(seeds={"observed_noise": 7})
        one = build(self.spec("homoscedastic",
                              sigma={"value": 0.5, "unit": "K"}),
                    ctx=ctx, full=full, space=space)
        two = build(self.spec("homoscedastic",
                              sigma={"value": 0.5, "unit": "K"}),
                    ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert jnp.allclose(one.entries["primary"], two.entries["primary"])
        assert not jnp.allclose(one.entries["primary"],
                                clean.entries["primary"])

    def test_a_different_declared_seed_draws_a_different_scatter(self):
        # Reproducibility alone cannot see a draw that ignores the seed --
        # key(0) twice is also reproducible. Two declared values must differ.
        spec = self.spec("homoscedastic", sigma={"value": 0.5, "unit": "K"})
        ctx7, full7, space7 = harness(seeds={"observed_noise": 7})
        ctx8, full8, space8 = harness(seeds={"observed_noise": 8})
        seven = build(spec, ctx=ctx7, full=full7, space=space7)
        eight = build(spec, ctx=ctx8, full=full8, space=space8)
        assert not jnp.allclose(seven.entries["primary"],
                                eight.entries["primary"])

    def test_the_draw_is_the_packages_realise_at_the_recorded_seed(self):
        # The seam claim, pinned bitwise: the data is NoiseModel.realise at
        # jax.random.key(recorded seed), nothing hand-written beside it.
        ctx, full, space = harness(seeds={"observed_noise": 7})
        observed = build(self.spec("homoscedastic",
                                   sigma={"value": 0.5, "unit": "K"}),
                         ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert observed.records["primary"]["seed"] == 7
        expected = HomoscedasticNoise(jnp.asarray(0.5, dtype=jnp.float32)
                                      ).realise(clean.entries["primary"],
                                                key=jax.random.key(7))
        assert jnp.array_equal(observed.entries["primary"], expected)

    def test_an_undeclared_seed_name_derives_by_blake2s_not_luck(self):
        ctx, full, space = harness()   # runtime.seeds is empty; seed is set
        observed = build(self.spec("homoscedastic",
                                   sigma={"value": 0.5, "unit": "K"}),
                         ctx=ctx, full=full, space=space)
        assert observed.records["primary"]["seed"] is not None

    def test_the_seed_is_required_on_a_drawing_kind(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="seed"):
            build(self.spec("homoscedastic", seed=None,
                            sigma={"value": 0.5, "unit": "K"}),
                  ctx=ctx, full=full, space=space)

    def test_radiometer_scatter_is_multiplicative(self):
        ctx, full, space = harness()
        observed = build(self.spec("radiometer"),
                         ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        ratio = observed.entries["primary"] / clean.entries["primary"]
        fractional = 1.0 / np.sqrt(3.125e6 * 2.0)
        assert float(jnp.max(jnp.abs(ratio - 1.0))) < 6 * fractional

    def test_radiometer_realise_is_the_multiplicative_form_exactly(self):
        # The trough prediction is negative throughout, so the additive
        # d + |d| f w form agrees with d(1 + f w) in ratio MAGNITUDE and
        # differs only in the sign of every perturbation -- which the ratio
        # test above cannot see. Bitwise equality with the package's own
        # realise is the check that can.
        ctx, full, space = harness(seeds={"observed_noise": 11})
        observed = build(self.spec("radiometer"),
                         ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert observed.records["primary"]["seed"] == 11
        expected = RadiometerNoise(3.125e6, 2.0).realise(
            clean.entries["primary"], key=jax.random.key(11))
        assert jnp.array_equal(observed.entries["primary"], expected)

    def test_from_model_draws_with_the_declared_noise_model(self):
        ctx, full, space = harness()
        noise = build_noise({"kind": "homoscedastic",
                             "sigma": {"value": 0.5, "unit": "K"}},
                            observation=_FakeObservation(), context=ctx)
        observed = build(self.spec("from_model"), ctx=ctx, full=full,
                         space=space, noise=noise)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert not jnp.allclose(observed.entries["primary"],
                                clean.entries["primary"])

    def test_from_model_with_kind_none_is_refused_by_name(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="from_model"):
            build(self.spec("from_model"), ctx=ctx, full=full, space=space)

    def test_kind_none_takes_no_seed_and_adds_nothing(self):
        ctx, full, space = harness()
        observed = build({"from": "simulation", "realise": {"kind": "none"}},
                         ctx=ctx, full=full, space=space)
        clean = build({"from": "simulation"}, ctx=ctx, full=full, space=space)
        assert jnp.allclose(observed.entries["primary"],
                            clean.entries["primary"])


class TestFileForm:
    def test_an_npz_lands_shape_checked(self, tmp_path):
        np.savez(tmp_path / "night1.npz",
                 waterfall=np.ones((16, 8), dtype=np.float32))
        ctx, full, space = harness(base_dir=str(tmp_path))
        observed = build({"file": {"path": "night1.npz", "format": "npz",
                                   "key": "waterfall"}},
                         ctx=ctx, full=full, space=space)
        assert observed.entries["primary"].shape == (16, 8)
        assert observed.records["primary"]["from"] == "file"

    def test_a_wrong_shape_is_refused_exactly(self, tmp_path):
        np.savez(tmp_path / "night1.npz",
                 waterfall=np.ones((8, 16), dtype=np.float32))
        ctx, full, space = harness(base_dir=str(tmp_path))
        with pytest.raises(ConfigError, match="16, 8"):
            build({"file": {"path": "night1.npz", "format": "npz",
                            "key": "waterfall"}},
                  ctx=ctx, full=full, space=space)


class TestSeveralObservations:
    def test_named_entries_each_simulate_their_own_truth(self):
        ctx, full, space = harness()
        observed = build({"primary": {"from": "simulation", "at": {"g": 1.1}},
                          "second": {"from": "simulation", "at": {"g": 1.5}}},
                         ctx=ctx, full=full, space=space)
        assert set(observed.entries) == {"primary", "second"}
        assert observed.primary == "primary"
        assert not jnp.allclose(observed.entries["primary"],
                                observed.entries["second"])

    def test_each_entrys_at_is_its_own_record(self):
        # Task 6's truth derivation reads ObservedBuild.at per entry; a dict
        # shared across entries would hand it the LAST entry's truth for all.
        ctx, full, space = harness()
        observed = build({"primary": {"from": "simulation", "at": {"g": 1.1}},
                          "second": {"from": "simulation", "at": {"g": 1.5}},
                          "third": {"from": "simulation"}},
                         ctx=ctx, full=full, space=space)
        assert float(observed.at["primary"]["g"]) == pytest.approx(1.1)
        assert float(observed.at["second"]["g"]) == pytest.approx(1.5)
        assert observed.at["third"] == {}

    def test_without_a_primary_the_default_is_unresolved(self):
        ctx, full, space = harness()
        observed = build({"a": {"from": "simulation"},
                          "b": {"from": "simulation"}},
                         ctx=ctx, full=full, space=space)
        assert observed.primary is None

    def test_an_entry_name_colliding_with_the_grammar_is_refused(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="realise"):
            build({"realise": {"from": "simulation"}},
                  ctx=ctx, full=full, space=space)

    def test_unknown_keys_in_a_simulation_spec_are_swept(self):
        ctx, full, space = harness()
        with pytest.raises(ConfigError, match="sigma"):
            build({"from": "simulation", "sigma": 0.5},
                  ctx=ctx, full=full, space=space)
