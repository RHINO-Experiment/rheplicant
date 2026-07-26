"""Tests for the parameter-space layer: Latent, Bind, ParameterSpace.

The three shapes a parameterization can take, all exercised here:

* **direct**   — one latent lands in one leaf unchanged;
* **tied**     — one latent lands in SEVERAL leaves (optionally transformed);
* **derived**  — SEVERAL latents produce one leaf through a function.

Plus the invariant everything downstream rests on: binding never changes the
pipeline's pytree structure.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.radio import GainOperator, SkyOperator

SKY_K = 100.0


@pytest.fixture
def twin():
    """Sky -> two gain stages (so tying has something to tie)."""
    return Pipeline(
        SkyOperator(amplitude=jnp.array(SKY_K)),
        GainOperator(gain=jnp.array(1.0)),
        GainOperator(gain=jnp.array(1.0)),
        names=("sky", "gain_a", "gain_b"),
    )


class TestDeclaration:
    def test_initial_values_are_keyed_by_name(self):
        space = ParameterSpace(
            latents=[Latent("log_gain", init=0.0), Latent("sky_k", init=SKY_K)],
            bindings=[
                Bind("log_gain", into=lambda p: p["gain_a"].gain, fn=jnp.exp),
                Bind("sky_k", into=lambda p: p["sky"].amplitude),
            ],
        )
        values = space.initial_values()
        assert set(values) == {"log_gain", "sky_k"}
        assert values["log_gain"].shape == ()

    def test_init_is_converted_to_an_array(self):
        # plain Python floats are the natural thing to type; they must not stay floats
        assert isinstance(Latent("g", init=1.0).init, jax.Array)

    def test_names_are_reported_in_declaration_order(self):
        space = ParameterSpace(
            latents=[Latent("b", init=0.0), Latent("a", init=0.0)],
            bindings=[
                Bind("b", into=lambda p: p["gain_a"].gain),
                Bind("a", into=lambda p: p["gain_b"].gain),
            ],
        )
        assert space.names == ("b", "a")


class TestBinding:
    def test_direct_placement(self, twin):
        space = ParameterSpace(
            latents=[Latent("g", init=1.0)],
            bindings=[Bind("g", into=lambda p: p["gain_a"].gain)],
        )
        bound = space.bind(twin, {"g": jnp.array(2.5)})
        assert float(bound["gain_a"].gain) == pytest.approx(2.5)
        assert float(bound["gain_b"].gain) == pytest.approx(1.0)  # untouched

    def test_transform_is_applied(self, twin):
        space = ParameterSpace(
            latents=[Latent("log_g", init=0.0)],
            bindings=[Bind("log_g", into=lambda p: p["gain_a"].gain, fn=jnp.exp)],
        )
        bound = space.bind(twin, {"log_g": jnp.array(jnp.log(3.0))})
        assert float(bound["gain_a"].gain) == pytest.approx(3.0)

    def test_one_latent_ties_several_leaves(self, twin):
        """A single scalar value fans out to every selector in `into`."""
        space = ParameterSpace(
            latents=[Latent("log_g", init=0.0)],
            bindings=[
                Bind(
                    "log_g",
                    into=(lambda p: p["gain_a"].gain, lambda p: p["gain_b"].gain),
                    fn=jnp.exp,
                )
            ],
        )
        bound = space.bind(twin, {"log_g": jnp.array(jnp.log(2.0))})
        assert float(bound["gain_a"].gain) == pytest.approx(2.0)
        assert float(bound["gain_b"].gain) == pytest.approx(2.0)

    def test_several_latents_derive_one_leaf(self, twin):
        """Two scalars -> an (n_time,) gain ramp: the low-dim -> high-dim case."""
        n_time = 8
        space = ParameterSpace(
            latents=[Latent("g0", init=1.0), Latent("slope", init=0.0)],
            bindings=[
                Bind(
                    ("g0", "slope"),
                    into=lambda p: p["gain_a"].gain,
                    fn=lambda g0, slope: g0 + slope * jnp.arange(n_time, dtype=float),
                )
            ],
        )
        twin = eqx.tree_at(lambda p: p["gain_a"].gain, twin, jnp.ones(n_time))
        bound = space.bind(twin, {"g0": jnp.array(2.0), "slope": jnp.array(0.5)})
        expected = 2.0 + 0.5 * jnp.arange(n_time, dtype=float)
        assert jnp.allclose(bound["gain_a"].gain, expected)

    def test_fn_may_return_a_tuple_matching_into(self, twin):
        """Returning a tuple addresses each selector separately."""
        space = ParameterSpace(
            latents=[Latent("x", init=1.0)],
            bindings=[
                Bind(
                    "x",
                    into=(lambda p: p["gain_a"].gain, lambda p: p["gain_b"].gain),
                    fn=lambda x: (x, 2.0 * x),
                )
            ],
        )
        bound = space.bind(twin, {"x": jnp.array(3.0)})
        assert float(bound["gain_a"].gain) == pytest.approx(3.0)
        assert float(bound["gain_b"].gain) == pytest.approx(6.0)

    def test_binding_preserves_the_pipeline_structure(self, twin):
        """The invariant every downstream consumer rests on (vmap, jit, ravel_pytree)."""
        space = ParameterSpace(
            latents=[Latent("g", init=1.0)],
            bindings=[Bind("g", into=lambda p: p["gain_a"].gain)],
        )
        bound = space.bind(twin, {"g": jnp.array(2.0)})
        assert jax.tree_util.tree_structure(bound) == jax.tree_util.tree_structure(twin)

    def test_the_original_pipeline_is_not_mutated(self, twin):
        space = ParameterSpace(
            latents=[Latent("g", init=1.0)],
            bindings=[Bind("g", into=lambda p: p["gain_a"].gain)],
        )
        space.bind(twin, {"g": jnp.array(9.0)})
        assert float(twin["gain_a"].gain) == pytest.approx(1.0)

    def test_bind_is_jittable(self, twin):
        space = ParameterSpace(
            latents=[Latent("log_g", init=0.0)],
            bindings=[Bind("log_g", into=lambda p: p["gain_a"].gain, fn=jnp.exp)],
        )

        @eqx.filter_jit
        def run(values):
            return space.bind(twin, values)["gain_a"].gain

        assert float(run({"log_g": jnp.array(0.0)})) == pytest.approx(1.0)


class TestEscapeHatch:
    def test_raw_bind_function(self, twin):
        """Anything the declarative blocks cannot express still has a way through."""
        space = ParameterSpace.raw(
            latents=[Latent("g", init=1.0)],
            bind=lambda pipeline, values: eqx.tree_at(
                lambda p: p["gain_a"].gain, pipeline, values["g"]
            ),
        )
        bound = space.bind(twin, {"g": jnp.array(4.0)})
        assert float(bound["gain_a"].gain) == pytest.approx(4.0)
        assert space.names == ("g",)

    def test_raw_rejects_bindings(self):
        with pytest.raises(Exception, match="bindings"):
            ParameterSpace.raw(
                latents=[Latent("g", init=1.0)],
                bind=lambda pipeline, values: pipeline,
                bindings=[Bind("g", into=lambda p: p["gain_a"].gain)],
            )
