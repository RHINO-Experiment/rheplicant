"""The transform registry, and bindings -> ParameterSpace."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.parameters import parse_latents
from rheplicant.config.sections.transforms import build_space, parse_transform
from tests.config.inference_helpers import context, twin


def space_for(parameters, bindings=None, joint_prior=None, replaced=(),
              model=None):
    fit = twin(model)
    return build_space(parse_latents(parameters, context()), bindings,
                       joint_prior, fit_twin=fit, replaced=replaced,
                       context=context()), fit


class TestRegistry:
    def test_named_transforms_map_to_the_documented_callables(self):
        exp, fan = parse_transform("exp", context(), where="t")
        assert float(exp(jnp.asarray(0.0))) == pytest.approx(1.0)
        assert fan is None or fan == "broadcast"

    def test_log_and_sum_are_the_documented_callables(self):
        log, _ = parse_transform("log", context(), where="t")
        assert float(log(jnp.exp(jnp.asarray(1.0)))) == pytest.approx(1.0)
        total, _ = parse_transform("sum", context(), where="t")
        assert float(total(jnp.asarray([1.0, 2.0, 3.0]))) == pytest.approx(6.0)

    def test_split_rows_is_the_distribute_transform(self):
        fn, fan = parse_transform("split_rows", context(), where="t")
        produced = fn(jnp.asarray([1.0, 2.0]))
        assert isinstance(produced, tuple) and len(produced) == 2
        assert fan == "distribute"

    def test_unit_mean_bandpass_is_the_packages_own(self):
        from rheplicant.radio.instrument.receiver import unit_mean_bandpass

        fn, _ = parse_transform("unit_mean_bandpass", context(), where="t")
        free = jnp.full((7,), 1.0)
        assert jnp.allclose(fn(free), unit_mean_bandpass(free))

    def test_affine_takes_scale_and_offset(self):
        fn, _ = parse_transform({"affine": {"scale": 2.0, "offset": 1.0}},
                                context(), where="t")
        assert float(fn(jnp.asarray(3.0))) == pytest.approx(7.0)

    def test_affine_defaults_to_unit_scale_and_zero_offset(self):
        fn, _ = parse_transform({"affine": {"offset": 1.0}}, context(),
                                where="t")
        assert float(fn(jnp.asarray(3.0))) == pytest.approx(4.0)
        fn, _ = parse_transform({"affine": {"scale": 2.0}}, context(),
                                where="t")
        assert float(fn(jnp.asarray(3.0))) == pytest.approx(6.0)

    def test_matmul_applies_a_declared_design(self):
        fn, _ = parse_transform(
            {"matmul": {"design": {"ones": ["n_freq", 2]}}},
            context(), where="t")
        out = fn(jnp.asarray([1.0, 2.0]))
        assert out.shape == (8,)
        assert float(out[0]) == pytest.approx(3.0)

    def test_log_link_basis_is_exp_of_a_basis_expansion(self):
        fn, _ = parse_transform(
            {"log_link_basis": {"kind": "legendre", "n_basis": 3}},
            context(), where="t")
        out = fn(jnp.zeros(3))
        assert out.shape == (8,)
        assert jnp.allclose(out, 1.0)

    def test_log_link_basis_axis_time_reads_the_time_grid(self):
        fn, _ = parse_transform(
            {"log_link_basis": {"kind": "legendre", "n_basis": 3,
                                "axis": "time"}},
            context(), where="t")
        assert fn(jnp.zeros(3)).shape == (16,)

    def test_basis_expand_reads_a_declared_basis_resource(self):
        from rheplicant.config.resources import build_resources

        ctx = context()
        built = build_resources(
            {"bases": {"smooth": {"time": {"kind": "legendre", "n_basis": 2},
                                  "freq": {"kind": "legendre",
                                           "n_basis": 3}}}}, ctx)
        ctx = context(resources=dict(built.resources))
        fn, _ = parse_transform(
            {"basis_expand": {"basis": {"ref": "resources.bases.smooth"}}},
            ctx, where="t")
        assert fn(jnp.zeros((2, 3))).shape == (16, 8)

    def test_basis_expand_refuses_a_ref_that_is_not_a_basis(self):
        ctx = context(resources={"resources.bases.flat": jnp.zeros(3)})
        with pytest.raises(ConfigError, match="not SeparableBasis"):
            parse_transform(
                {"basis_expand": {"basis": {"ref": "resources.bases.flat"}}},
                ctx, where="t")

    def test_beam_analysis_is_plan_2c_by_name(self):
        with pytest.raises(ConfigError, match="2C"):
            parse_transform({"beam_analysis": {"nside": 32, "lmax": 8}},
                            context(), where="t")

    def test_python_requires_a_declared_fan(self):
        with pytest.raises(ConfigError, match="fan"):
            parse_transform({"python": "jax.numpy:exp"}, context(), where="t")

    def test_an_unknown_transform_is_refused_listing_the_registry(self):
        with pytest.raises(ConfigError, match="sinh"):
            parse_transform("sinh", context(), where="t")


class TestBuildSpace:
    def test_the_into_sugar_builds_a_working_space(self):
        space, fit = space_for(
            {"g": {"init": 1.0, "linear": True, "into": "gain.gain"}})
        bound = space.bind(fit, {"g": jnp.asarray(2.0)})
        assert float(bound["gain"].gain) == pytest.approx(2.0)

    def test_a_transform_travels_into_the_bind(self):
        space, fit = space_for(
            {"log_g": {"init": 0.0, "into": "gain.gain",
                       "transform": "exp"}})
        bound = space.bind(fit, {"log_g": jnp.asarray(0.0)})
        assert float(bound["gain"].gain) == pytest.approx(1.0)

    def test_split_rows_distributes_over_two_leaves(self):
        space, fit = space_for(
            {"pair": {"init": {"list": [0.25, 2.0]},
                      "into": ["global_signal.depth", "gain.gain"],
                      "transform": "split_rows"}})
        bound = space.bind(fit, {"pair": jnp.asarray([0.25, 2.0])})
        assert float(bound["global_signal"].depth) == pytest.approx(0.25)
        assert float(bound["gain"].gain) == pytest.approx(2.0)

    def test_a_bindings_entry_spells_the_same_thing_longhand(self):
        space, fit = space_for(
            {"g": {"init": 1.0}},
            bindings=[{"latents": ["g"], "into": "gain.gain"}])
        bound = space.bind(fit, {"g": jnp.asarray(3.0)})
        assert float(bound["gain"].gain) == pytest.approx(3.0)

    def test_a_bindings_entry_joins_two_latents_through_python(self):
        space, fit = space_for(
            {"a": {"init": 1.0}, "b": {"init": 2.0}},
            bindings=[{"latents": ["a", "b"], "into": "gain.gain",
                       "transform": {"python": "jax.numpy:add",
                                     "fan": "broadcast"}}])
        bound = space.bind(fit, {"a": jnp.asarray(2.0),
                                 "b": jnp.asarray(3.0)})
        assert float(bound["gain"].gain) == pytest.approx(5.0)

    def test_latents_keep_declaration_order_not_sorted_order(self):
        space, _ = space_for(
            {"z": {"init": 1.0, "into": "gain.gain"},
             "a": {"init": 1.0, "into": "global_signal.depth"}})
        assert space.names == ("z", "a")

    def test_a_binding_into_an_aliased_node_is_refused_up_front(self):
        class _Forked:
            aliased = ("gain",)

        with pytest.raises(ConfigError, match="more than one place"):
            build_space(parse_latents({"g": {"init": 1.0,
                                             "into": "gain.gain"}},
                                      context()),
                        None, None, fit_twin=_Forked(), replaced=(),
                        context=context())

    def test_into_sugar_and_a_bindings_entry_are_mutually_exclusive(self):
        with pytest.raises(ConfigError, match="mutually exclusive"):
            space_for({"g": {"init": 1.0, "into": "gain.gain"}},
                      bindings=[{"latents": ["g"],
                                 "into": "global_signal.depth"}])

    def test_two_bindings_into_one_leaf_are_refused(self):
        with pytest.raises(ConfigError, match="gain"):
            space_for({"a": {"init": 1.0, "into": "gain.gain"},
                       "b": {"init": 1.0, "into": "gain.gain"}})

    def test_a_binding_into_a_replaced_node_is_check_b8(self):
        with pytest.raises(ConfigError, match="replace"):
            space_for({"g": {"init": 1.0, "into": "gain.gain"}},
                      replaced=("gain",))

    def test_a_bindings_entry_naming_an_undeclared_latent_is_refused(self):
        with pytest.raises(ConfigError, match="ghost"):
            space_for({"g": {"init": 1.0, "into": "gain.gain"}},
                      bindings=[{"latents": ["ghost"],
                                 "into": "global_signal.depth"}])

    def test_a_declared_fan_conflicting_with_the_registry_is_refused(self):
        with pytest.raises(ConfigError, match="distribute"):
            space_for({"pair": {"init": {"list": [1.0, 2.0]},
                                "into": ["global_signal.depth", "gain.gain"],
                                "transform": "split_rows",
                                "fan": "broadcast"}})

    def test_a_transform_without_into_is_refused(self):
        with pytest.raises(ConfigError, match="into"):
            space_for({"g": {"init": 1.0, "transform": "exp"}})

    def test_no_parameters_means_no_space(self):
        fit = twin()
        assert build_space(None, None, None, fit_twin=fit, replaced=(),
                           context=context()) is None

    def test_bindings_without_parameters_are_refused(self):
        with pytest.raises(ConfigError, match="parameters"):
            build_space(None, [{"latents": ["g"], "into": "gain.gain"}],
                        None, fit_twin=twin(), replaced=(),
                        context=context())


class TestJointPrior:
    def test_jeffreys_lands_on_the_space(self):
        from rheplicant.inference import JeffreysPrior

        space, _ = space_for(
            {"a": {"init": 1.0, "into": "gain.gain"},
             "b": {"init": 1.0, "into": "global_signal.depth"}},
            joint_prior={"jeffreys": {"over": ["a", "b"]}})
        assert isinstance(space.joint_prior, JeffreysPrior)
        assert space.joint_prior.over == ("a", "b")

    def test_rank_rtol_travels_onto_the_prior(self):
        space, _ = space_for(
            {"a": {"init": 1.0, "into": "gain.gain"}},
            joint_prior={"jeffreys": {"over": ["a"], "rank_rtol": 0.25}})
        assert space.joint_prior.rank_rtol == pytest.approx(0.25)

    def test_only_jeffreys_exists(self):
        with pytest.raises(ConfigError, match="jeffreys"):
            space_for({"a": {"init": 1.0, "into": "gain.gain"}},
                      joint_prior={"reference": {"over": ["a"]}})

    def test_joint_prior_without_parameters_is_refused(self):
        with pytest.raises(ConfigError, match="parameters"):
            build_space(None, None, {"jeffreys": {"over": ["a"]}},
                        fit_twin=twin(), replaced=(), context=context())
