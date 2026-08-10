"""inference.parameters: the latent grammar, and the priors that broadcast."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.parameters import parse_latents
from tests.config.inference_helpers import context


def one(name="gain_scale", **keys):
    spec = {"init": 1.0}
    spec.update(keys)
    return {name: spec}


class TestInit:
    def test_a_scalar_init_becomes_a_latent(self):
        parsed = parse_latents(one(), context())
        latent = parsed["gain_scale"].latent
        assert latent.name == "gain_scale"
        assert float(latent.init) == pytest.approx(1.0)
        assert latent.linear is False

    def test_init_is_required(self):
        with pytest.raises(ConfigError, match="init"):
            parse_latents({"gain_scale": {"prior": None}}, context())

    def test_an_array_init_lands_in_the_runs_dtype(self):
        parsed = parse_latents(one(init={"zeros": ["n_freq"], "unit": "K"}),
                               context())
        init = parsed["gain_scale"].latent.init
        assert init.shape == (8,)
        assert init.dtype == jnp.float32

    def test_an_all_zero_init_warns_about_absolute_probes(self):
        with pytest.warns(UserWarning, match="probe"):
            parse_latents(one(init={"zeros": ["n_freq"]}), context())

    def test_an_integer_init_is_coerced_to_the_runs_dtype(self):
        parsed = parse_latents(one(init=2), context())
        init = parsed["gain_scale"].latent.init
        assert init.dtype == jnp.float32
        assert float(init) == pytest.approx(2.0)

    def test_latents_keep_declaration_order(self):
        parsed = parse_latents(
            {"m_mid": {"init": 1.0}, "z_last": {"init": 2.0},
             "a_first": {"init": 3.0}},
            context())
        assert list(parsed) == ["m_mid", "z_last", "a_first"]


class TestPriorSpec:
    def test_a_scalar_normal_prior_broadcasts_to_the_init_shape(self):
        parsed = parse_latents(
            one(init={"zeros": ["n_freq"]},
                prior={"normal": {"loc": 0.0, "scale": 400.0}}),
            context())
        prior = parsed["gain_scale"].latent.prior
        assert prior.shape() == (8,)

    def test_uniform_and_log_normal_build(self):
        parsed = parse_latents(
            {"a": {"init": 0.3, "prior": {"uniform": {"low": 0.05,
                                                      "high": 0.60}}},
             "b": {"init": 1.0, "prior": {"log_normal": {"loc": 0.0,
                                                         "scale": 1.0}}}},
            context())
        assert parsed["a"].latent.prior.shape() == ()
        assert parsed["b"].latent.prior.shape() == ()

    def test_an_array_loc_may_come_from_a_value_node(self):
        parsed = parse_latents(
            one(init={"zeros": ["n_freq"]},
                prior={"normal": {"loc": {"zeros": ["n_freq"]},
                                  "scale": 400.0}}),
            context())
        assert parsed["gain_scale"].latent.prior.shape() == (8,)

    def test_the_python_hatch_builds_any_distribution(self):
        parsed = parse_latents(
            one(prior={"python": "numpyro.distributions:StudentT",
                       "args": {"df": 3.0, "loc": 0.0, "scale": 1.0}}),
            context())
        assert parsed["gain_scale"].latent.prior.shape() == ()

    def test_an_unknown_family_is_refused_listing_the_registry(self):
        with pytest.raises(ConfigError, match="cauchy"):
            parse_latents(one(prior={"cauchy": {"loc": 0.0}}), context())

    def test_two_families_in_one_prior_are_refused(self):
        with pytest.raises(ConfigError, match="exactly one"):
            parse_latents(one(prior={"normal": {"loc": 0.0, "scale": 1.0},
                                     "uniform": {"low": 0.0, "high": 1.0}}),
                          context())

    def test_a_boolean_prior_operand_is_refused(self):
        with pytest.raises(ConfigError, match="loc"):
            parse_latents(one(prior={"normal": {"loc": True, "scale": 1.0}}),
                          context())

    def test_null_is_a_free_latent(self):
        parsed = parse_latents(one(prior=None), context())
        assert parsed["gain_scale"].latent.prior is None


class TestReservedAndRecorded:
    def test_support_and_hyper_are_capability_4(self):
        for key, value in (("support", [0.0, 1.0]), ("hyper", {})):
            with pytest.raises(ConfigError, match="capability 4"):
                parse_latents(one(**{key: value}), context())

    def test_scope_per_epoch_is_capability_4(self):
        with pytest.raises(ConfigError, match="capability 4"):
            parse_latents(one(scope="per_epoch"), context())

    def test_scope_global_is_the_accepted_spelling(self):
        parsed = parse_latents(one(scope="global"), context())
        assert parsed["gain_scale"].latent.scope == "global"

    def test_latex_renames_unit_and_ref_are_recorded(self):
        parsed = parse_latents(
            one(latex="g", renames="old_gain", unit="dimensionless",
                ref=1.5),
            context())
        entry = parsed["gain_scale"]
        assert entry.latex == "g"
        assert entry.renames == ("old_gain",)
        assert entry.unit == "dimensionless"
        assert float(entry.ref) == pytest.approx(1.5)

    def test_ref_lands_in_the_runs_dtype(self):
        parsed = parse_latents(one(ref=2), context())
        ref = parsed["gain_scale"].ref
        assert ref.dtype == jnp.float32
        assert float(ref) == pytest.approx(2.0)

    def test_a_unit_conflicting_with_inits_written_unit_is_refused(self):
        with pytest.raises(ConfigError, match="unit"):
            parse_latents(one(init={"value": 1.0, "unit": "K"},
                              unit="mK"),
                          context())


class TestBindingKeysTravelRaw:
    def test_into_transform_and_fan_are_carried(self):
        parsed = parse_latents(
            one(into="gain.gain", transform="exp", fan="broadcast"),
            context())
        entry = parsed["gain_scale"]
        assert entry.into == ("gain.gain",)
        assert entry.transform == "exp"
        assert entry.fan == "broadcast"

    def test_fan_outside_the_two_modes_is_refused(self):
        with pytest.raises(ConfigError, match="broadcast"):
            parse_latents(one(into="gain.gain", fan="scatter"), context())

    def test_linear_must_be_a_real_bool(self):
        with pytest.raises(ConfigError, match="linear"):
            parse_latents(one(linear="yes"), context())

    def test_unknown_latent_keys_are_swept(self):
        with pytest.raises(ConfigError, match="prior_std"):
            parse_latents(one(prior_std=400.0), context())

    def test_the_section_must_be_a_mapping_of_names(self):
        with pytest.raises(ConfigError, match="mapping"):
            parse_latents([{"init": 1.0}], context())
