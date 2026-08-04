"""Tests for the parameter-space layer: Latent, Bind, ParameterSpace.

The three shapes a parameterization can take, all exercised here:

* **direct**   — one latent lands in one leaf unchanged;
* **tied**     — one latent lands in SEVERAL leaves (optionally transformed);
* **derived**  — SEVERAL latents produce one leaf through a function.

Plus the invariant everything downstream rests on: binding never changes the
pipeline's pytree structure.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.graph import At, NodeSpec, SignalGraph, assemble
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import AdamCalibrator, Bind, Latent, ParameterSpace
from rheplicant.radio import AntennaLossOperator, GainOperator, SkyOperator

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


class TestDeclarationValidation:
    """Checks that need nothing but the declaration itself."""

    def test_duplicate_names_rejected(self):
        with pytest.raises(ParameterSpaceError, match="unique"):
            ParameterSpace(
                latents=[Latent("g", init=1.0), Latent("g", init=2.0)],
                bindings=[Bind("g", into=lambda p: p["gain_a"].gain)],
            )

    def test_undeclared_latent_rejected(self):
        with pytest.raises(ParameterSpaceError, match="undeclared"):
            ParameterSpace(
                latents=[Latent("g", init=1.0)],
                bindings=[Bind("typo", into=lambda p: p["gain_a"].gain)],
            )

    def test_unbound_latent_rejected(self):
        """A latent nothing binds would sample happily and return the prior."""
        with pytest.raises(ParameterSpaceError, match="never bound"):
            ParameterSpace(
                latents=[Latent("g", init=1.0), Latent("orphan", init=0.0)],
                bindings=[Bind("g", into=lambda p: p["gain_a"].gain)],
            )

    def test_bind_needs_at_least_one_selector(self):
        """Untested guard. Nothing downstream covers it: the dead-latent rule is
        satisfied because the latent IS named in a binding, so a Bind with no
        selectors would validate and quietly reach nothing."""
        with pytest.raises(ParameterSpaceError, match="`into` selector"):
            Bind("g", into=())

    def test_fn_returning_the_wrong_number_of_values_is_caught(self, twin):
        """Untested guard, and the only defence against one binding's values
        landing in another binding's selectors when the mismatches cancel."""
        space = ParameterSpace(
            latents=[Latent("x", init=1.0)],
            bindings=[
                Bind("x",
                     into=(lambda p: p["gain_a"].gain, lambda p: p["gain_b"].gain),
                     fn=lambda x: (x, 2.0 * x, 3.0 * x)),   # 3 values, 2 selectors
            ],
        )
        with pytest.raises(ParameterSpaceError, match="returned 3 values"):
            space.bind(twin, {"x": jnp.array(1.0)})

    def test_identity_bind_needs_exactly_one_latent(self):
        with pytest.raises(ParameterSpaceError, match="exactly one latent"):
            Bind(("a", "b"), into=lambda p: p["gain_a"].gain)

    def test_prior_shape_must_match_init(self):
        prior = pytest.importorskip("numpyro.distributions")
        with pytest.raises(ParameterSpaceError, match="shape"):
            Latent("g", init=1.0, prior=prior.Normal(jnp.zeros(3), 1.0))


class TestFanOut:
    """`fan=` — whether one produced value ties every leaf or one value per leaf.

    Two scalar leaves that a 2-vector reaches by opposite routes, chosen so the
    two readings are numerically far apart rather than merely different:

    * **broadcast** writes the whole ``[2, 5]`` to BOTH — the loss sees it as a
      spectrum and the gain as a per-sample series, so ``pred[0, 0]`` is
      ``2 * 2 = 4``;
    * **distribute** writes ``2`` to the loss and ``5`` to the gain, so
      ``pred[0, 0]`` is ``2 * 5 = 10``.

    A symmetric fixture (equal leaves, or a ``[c, c]`` vector) makes both
    readings agree and blinds every test below.
    """

    N = 2
    V = (2.0, 5.0)

    @pytest.fixture
    def fan_state(self):
        return State(
            coords=Coordinates(
                time=jnp.arange(self.N, dtype=float),
                freq=jnp.array([100.0, 110.0]),
            ),
            data=jnp.ones((self.N, self.N)),
        )

    @pytest.fixture
    def fan_twin(self):
        """t_physical = 0, so the efficiency is a pure multiply and the two
        leaves enter the prediction as a bare product."""
        return Pipeline(
            AntennaLossOperator(
                efficiency=jnp.array(1.0), t_physical=jnp.array(0.0)
            ),
            GainOperator(gain=jnp.array(1.0)),
            names=("loss", "gain"),
        )

    @staticmethod
    def _space(fn, into, init, fan=None):
        return ParameterSpace(
            latents=[Latent("v", init=init)],
            bindings=[Bind("v", into=into, fn=fn, fan=fan)],
        )

    def _predict(self, twin, state, fn, fan=None):
        into = (lambda p: p["loss"].efficiency, lambda p: p["gain"].gain)
        init = jnp.array(self.V)
        space = self._space(fn, into, init, fan)
        return float(space.bind(twin, {"v": init})(state).data[0, 0])

    # ------------------------------------------------- the defect, restated --

    def test_the_python_container_type_alone_decides_the_physics(
        self, fan_twin, fan_state
    ):
        """`v` and `list(v)` are the SAME DATA and mean opposite things.

        This is what `fan=` exists for, and it is still accepted with
        `fan=None` on purpose — `Bind` is public and used in every example, so
        the inference stays the default. What changes is that the intent can
        now be declared and CHECKED, which is the next two tests.
        """
        assert self._predict(fan_twin, fan_state, lambda v: v) == pytest.approx(4.0)
        assert self._predict(fan_twin, fan_state, list) == pytest.approx(10.0)

    # ------------------------------------------------- declared: broadcast --

    def test_a_declared_broadcast_that_produced_a_container_is_refused(
        self, fan_twin, fan_state
    ):
        """The motivating row: the user meant "this whole 2-vector into both
        leaves" and `list(v)` distributed it element-wise instead — finite,
        correctly shaped, and 10.0 where 4.0 was meant."""
        with pytest.raises(ParameterSpaceError, match="fan='broadcast'"):
            self._predict(fan_twin, fan_state, list, fan="broadcast")

    def test_a_declared_broadcast_still_ties_when_it_produced_one_value(
        self, fan_twin, fan_state
    ):
        """The guard's other branch: the declaration agrees, nothing is refused,
        and the answer is the tied one."""
        value = self._predict(fan_twin, fan_state, lambda v: v, fan="broadcast")
        assert value == pytest.approx(4.0)

    # ------------------------------------------------ declared: distribute --

    def test_a_declared_distribute_that_produced_one_value_is_refused(
        self, fan_twin, fan_state
    ):
        """The mirror image: the user meant one value per leaf and got a tie."""
        with pytest.raises(ParameterSpaceError, match="fan='distribute'"):
            self._predict(fan_twin, fan_state, lambda v: v, fan="distribute")

    def test_a_declared_distribute_still_distributes_a_matching_container(
        self, fan_twin, fan_state
    ):
        value = self._predict(fan_twin, fan_state, list, fan="distribute")
        assert value == pytest.approx(10.0)

    def test_a_declared_distribute_of_the_wrong_length_is_refused(
        self, fan_twin, fan_state
    ):
        """The length check is not replaced by the declaration, it is sharpened
        by it — the refusal can now say which count was the declared one."""
        with pytest.raises(ParameterSpaceError, match="returned 3 values"):
            self._predict(
                fan_twin, fan_state, lambda v: [v[0], v[1], v[0]], fan="distribute"
            )

    def test_the_length_check_still_holds_with_no_declaration(
        self, fan_twin, fan_state
    ):
        with pytest.raises(ParameterSpaceError, match="returned 3 values"):
            self._predict(fan_twin, fan_state, lambda v: [v[0], v[1], v[0]])

    # ------------------------------------------------------ one selector --

    def test_a_lone_selector_fed_a_container_warns_rather_than_guessing_silently(
        self, fan_twin, fan_state
    ):
        """`len(produced) == len(into)` is what distinguishes the two modes, and
        at 1 it cannot: a length-1 container satisfies it under either intent.

        Warned rather than refused, and warned rather than left silent. Refused
        would break working code for no correctness gain — a Python list is not
        an array leaf, so unwrapping is the only reading that can produce a
        valid pipeline, and a broadcast of the container would be caught
        downstream as a pytree-structure change. Silent is what it was.
        """
        into = (lambda p: p["gain"].gain,)
        init = jnp.array([3.0])
        space = self._space(lambda v: [v[0]], into, init)
        with pytest.warns(UserWarning, match="one `into` selector"):
            bound = space.bind(fan_twin, {"v": init})
        assert float(bound["gain"].gain) == pytest.approx(3.0)

    def test_declaring_distribute_silences_the_lone_selector_warning(
        self, fan_twin, recwarn
    ):
        into = (lambda p: p["gain"].gain,)
        init = jnp.array([3.0])
        space = self._space(lambda v: [v[0]], into, init, fan="distribute")
        bound = space.bind(fan_twin, {"v": init})
        assert float(bound["gain"].gain) == pytest.approx(3.0)
        assert [w for w in recwarn.list if "into` selector" in str(w.message)] == []

    def test_declaring_broadcast_on_a_lone_selector_refuses_instead(self, fan_twin):
        """The other way out of the ambiguity, and it is a refusal: a container
        is not a leaf value however many selectors there are."""
        into = (lambda p: p["gain"].gain,)
        init = jnp.array([3.0])
        space = self._space(lambda v: [v[0]], into, init, fan="broadcast")
        with pytest.raises(ParameterSpaceError, match="fan='broadcast'"):
            space.bind(fan_twin, {"v": init})

    def test_a_lone_selector_fed_one_value_does_not_warn(self, fan_twin, recwarn):
        """The warning is about the container, not about having one selector."""
        into = (lambda p: p["gain"].gain,)
        init = jnp.array(3.0)
        space = self._space(None, into, init)
        bound = space.bind(fan_twin, {"v": init})
        assert float(bound["gain"].gain) == pytest.approx(3.0)
        assert [w for w in recwarn.list if "into` selector" in str(w.message)] == []

    # ----------------------------------------------------- the declaration --

    def test_an_unknown_fan_is_refused_at_declaration_and_names_both_modes(self):
        with pytest.raises(ParameterSpaceError, match="fan='tie'"):
            Bind("v", into=lambda p: p["gain"].gain, fan="tie")

    def test_fan_defaults_to_absent_so_every_existing_Bind_is_unchanged(self):
        assert Bind("v", into=lambda p: p["gain"].gain).fan is None

    def test_the_new_static_field_left_Bind_a_leafless_pytree(self):
        """`Bind` carries no array leaves — every field is static — so the new
        field lands in the treedef's aux data and changes no leaf count. Pinned
        because `ParameterSpace.bindings` is itself static, which makes that aux
        data part of a jit cache key and therefore required to stay hashable.
        """
        bind = Bind("v", into=lambda p: p["gain"].gain, fan="broadcast")
        leaves, treedef = jax.tree_util.tree_flatten(bind)
        assert leaves == []
        assert hash(treedef) == hash(treedef)
        assert list(Bind.__dataclass_fields__) == ["latents", "into", "fn", "fan"]

    def test_direct_threads_the_fan_through(self, fan_twin, fan_state):
        space = ParameterSpace.direct(
            "v",
            init=jnp.array(self.V),
            into=(lambda p: p["loss"].efficiency, lambda p: p["gain"].gain),
            fn=list,
            fan="broadcast",
        )
        with pytest.raises(ParameterSpaceError, match="fan='broadcast'"):
            space.bind(fan_twin, {"v": jnp.array(self.V)})


class TestPipelineValidation:
    """Checks that need the pipeline — all via jax.eval_shape, so they cost nothing."""

    def test_valid_space_passes(self, twin):
        space = ParameterSpace(
            latents=[Latent("log_g", init=0.0)],
            bindings=[Bind("log_g", into=lambda p: p["gain_a"].gain, fn=jnp.exp)],
        )
        space.validate(twin)  # must not raise

    def test_selector_must_reach_a_real_leaf(self, twin):
        """`graph_node` is static configuration, not an inferable array."""
        space = ParameterSpace(
            latents=[Latent("g", init=1.0)],
            bindings=[Bind("g", into=lambda p: p["gain_a"].graph_node)],
        )
        with pytest.raises(ParameterSpaceError, match="array leaf"):
            space.validate(twin)

    def test_two_bindings_cannot_write_the_same_leaf(self, twin):
        space = ParameterSpace(
            latents=[Latent("a", init=1.0), Latent("b", init=2.0)],
            bindings=[
                Bind("a", into=lambda p: p["gain_a"].gain),
                Bind("b", into=lambda p: p["gain_a"].gain),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="written by more than one"):
            space.validate(twin)

    def test_produced_shape_must_match_the_leaf(self, twin):
        space = ParameterSpace(
            latents=[Latent("g", init=jnp.zeros(3))],
            bindings=[Bind("g", into=lambda p: p["gain_a"].gain)],
        )
        with pytest.raises(ParameterSpaceError, match="shape"):
            space.validate(twin)

    def test_produced_dtype_kind_must_match_the_leaf(self, twin):
        """Writing a complex value into a real leaf is a modelling error, not a cast."""
        space = ParameterSpace(
            latents=[Latent("g", init=1.0)],
            bindings=[Bind("g", into=lambda p: p["gain_a"].gain, fn=lambda x: x + 0j)],
        )
        with pytest.raises(ParameterSpaceError, match="complex"):
            space.validate(twin)

    def test_raw_bind_changing_the_structure_is_caught(self, twin):
        space = ParameterSpace.raw(
            latents=[Latent("g", init=1.0)],
            bind=lambda pipeline, values: Pipeline(
                SkyOperator(amplitude=values["g"]), names=("sky",)
            ),
        )
        with pytest.raises(ParameterSpaceError, match="structure"):
            space.validate(twin)

    def test_derived_binding_validates_against_the_target(self, twin):
        n_time = 8
        twin = eqx.tree_at(lambda p: p["gain_a"].gain, twin, jnp.ones(n_time))
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
        space.validate(twin)  # must not raise


class TestForwardSeam:
    """The seam every inference engine reads: dict of named arrays -> prediction."""

    def test_forward_matches_bind_then_run(self, twin, template_state):
        space = ParameterSpace.direct(
            "log_g", init=0.0, into=lambda p: p["gain_a"].gain, fn=jnp.exp
        )
        forward, values0 = space.forward_fn(twin, template_state)
        values = {"log_g": jnp.array(jnp.log(2.0))}
        assert jnp.allclose(forward(values), space.bind(twin, values)(template_state).data)

    def test_initial_values_are_returned(self, twin, template_state):
        space = ParameterSpace.direct("g", init=1.5, into=lambda p: p["gain_a"].gain)
        _, values0 = space.forward_fn(twin, template_state)
        assert float(values0["g"]) == pytest.approx(1.5)

    def test_an_invalid_space_is_caught_at_build_time(self, twin, template_state):
        space = ParameterSpace.direct(
            "g", init=jnp.zeros(3), into=lambda p: p["gain_a"].gain
        )
        with pytest.raises(ParameterSpaceError, match="shape"):
            space.forward_fn(twin, template_state)

    def test_gradients_reach_every_latent_of_a_derived_binding(self, twin, template_state):
        n_time = template_state.coords.time.shape[0]
        twin = eqx.tree_at(lambda p: p["gain_a"].gain, twin, jnp.ones(n_time))
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
        forward, values0 = space.forward_fn(twin, template_state)
        grads = jax.grad(lambda v: jnp.sum(forward(v) ** 2))(values0)
        assert abs(float(grads["g0"])) > 0.0
        assert abs(float(grads["slope"])) > 0.0

    def test_forward_is_jittable(self, twin, template_state):
        space = ParameterSpace.direct(
            "log_g", init=0.0, into=lambda p: p["gain_a"].gain, fn=jnp.exp
        )
        forward, values0 = space.forward_fn(twin, template_state)
        assert jnp.allclose(eqx.filter_jit(forward)(values0), forward(values0))


class TestCalibratorIntegration:
    """The optimizers need no changes at all: a dict of arrays is a pytree."""

    def test_adam_recovers_a_tied_nonlinear_reparameterization(self, twin, template_state):
        true_gain = 1.1
        truth = eqx.tree_at(
            lambda p: (p["gain_a"].gain, p["gain_b"].gain),
            twin,
            (jnp.array(true_gain), jnp.array(true_gain)),
        )
        observed = truth(template_state).data

        # ONE latent, sampled in log space, driving BOTH gain stages.
        space = ParameterSpace.direct(
            "log_gain",
            init=0.0,
            into=(lambda p: p["gain_a"].gain, lambda p: p["gain_b"].gain),
            fn=jnp.exp,
        )
        forward, values0 = space.forward_fn(twin, template_state)
        fitted, losses = AdamCalibrator(learning_rate=0.01, n_steps=1500).fit(
            forward, values0, observed
        )
        assert float(fitted["log_gain"]) == pytest.approx(float(jnp.log(true_gain)), abs=1e-3)
        assert float(losses[-1]) < float(losses[0])


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

    def test_the_constructor_rejects_bindings_alongside_a_raw_bind(self, twin):
        """Regression. This combination used to CONSTRUCT and VALIDATE cleanly
        while bind() silently ignored every declared Bind — so a latent only
        those bindings reached was sampled without entering the model, and its
        posterior came back as its prior."""
        with pytest.raises(ParameterSpaceError, match="INSTEAD of bindings"):
            ParameterSpace(
                latents=[Latent("a", init=1.0), Latent("b", init=1.0)],
                bindings=[Bind("a", into=lambda p: p["gain_a"].gain),
                          Bind("b", into=lambda p: p["gain_b"].gain)],
                raw_bind=lambda pipeline, values: eqx.tree_at(
                    lambda p: p["gain_a"].gain, pipeline, values["a"]
                ),
            )

    def test_a_raw_bind_that_ignores_a_latent_is_caught(self, twin):
        """The dead-latent rule has to hold for raw binds too. It cannot be read
        off the declaration there, so it is probed: perturb the latent and see
        whether the bound model moves at all."""
        space = ParameterSpace.raw(
            latents=[Latent("used", init=1.0), Latent("forgotten", init=1.0)],
            bind=lambda pipeline, values: eqx.tree_at(
                lambda p: p["gain_a"].gain, pipeline, values["used"]
            ),
        )
        with pytest.raises(ParameterSpaceError, match="does not reach the pipeline"):
            space.validate(twin)

    def test_a_raw_bind_writing_the_wrong_shape_is_caught(self, twin):
        """A treedef encodes neither shape nor dtype, so the structure check
        alone let a scalar be broadcast into an (n_time,) leaf."""
        wide = eqx.tree_at(lambda p: p["gain_a"].gain, twin, jnp.ones(8))
        space = ParameterSpace.raw(
            latents=[Latent("g", init=1.0)],
            bind=lambda pipeline, values: eqx.tree_at(
                lambda p: p["gain_a"].gain, pipeline, values["g"]  # scalar into (8,)
            ),
        )
        with pytest.raises(ParameterSpaceError, match="shape"):
            space.validate(wide)

    def test_a_raw_bind_writing_the_wrong_dtype_kind_is_caught(self, twin):
        space = ParameterSpace.raw(
            latents=[Latent("g", init=1.0)],
            bind=lambda pipeline, values: eqx.tree_at(
                lambda p: p["gain_a"].gain, pipeline, values["g"] + 0j
            ),
        )
        with pytest.raises(ParameterSpaceError, match="complex"):
            space.validate(twin)

    def test_a_correct_raw_bind_still_validates(self, twin):
        space = ParameterSpace.raw(
            latents=[Latent("g", init=1.0)],
            bind=lambda pipeline, values: eqx.tree_at(
                lambda p: p["gain_a"].gain, pipeline, values["g"]
            ),
        )
        space.validate(twin)  # must not raise

    def test_raw_rejects_bindings(self):
        with pytest.raises(Exception, match="bindings"):
            ParameterSpace.raw(
                latents=[Latent("g", init=1.0)],
                bind=lambda pipeline, values: pipeline,
                bindings=[Bind("g", into=lambda p: p["gain_a"].gain)],
            )


# ---------------------------------------------------------------------------
# binding into a node the fold embedded more than once
# ---------------------------------------------------------------------------


class _ForkSrc(AbstractOperator):
    """Toy source: ``value * ones(3)``."""

    graph_node: ClassVar[str | None] = None
    value: jax.Array

    def __call__(self, state):
        return state.with_data(self.value * jnp.ones(3))


class _ForkScale(AbstractOperator):
    """Toy transform, so the fork-rejoin graph has an ordinary node too."""

    graph_node: ClassVar[str | None] = None
    factor: jax.Array

    def __call__(self, state):
        return state.with_data(state.data * self.factor)


@pytest.fixture
def fork_rejoin():
    """``x`` reaches the sink by TWO paths, ``out`` by one.

    ``x -> p -> j`` and ``x -> q -> j``, then ``j -> out``. The fold has no
    way to express the diamond as a tree, so it embeds ``x``'s operator once
    per path; ``out`` sits below the rejoin and is embedded once.
    """
    return SignalGraph(
        "fork-rejoin-binding",
        {
            "x": NodeSpec("source"),
            "p": NodeSpec("transform"),
            "q": NodeSpec("transform"),
            "j": NodeSpec("junction"),
            "out": NodeSpec("transform"),
        },
        [("x", "p"), ("x", "q"), ("p", "j"), ("q", "j"), ("j", "out")],
    )


@pytest.fixture
def forked(fork_rejoin):
    """One ``Src(10)`` at the aliased ``x``, one unit ``Scale`` at ``out``.

    Forward is ``10`` down each path, summed at ``j``, scaled by ``1`` — 20.
    """
    return assemble(
        fork_rejoin,
        At("x", _ForkSrc(value=jnp.array(10.0))),
        At("out", _ForkScale(factor=jnp.array(1.0))),
    )


class TestBindingIntoAnAliasedNode:
    """A node the fold embedded twice cannot be BOUND either, only read.

    :meth:`Assembly.replace_node` already refuses on ``Assembly.aliased``.
    Binding never went through it: ``Bind(into=lambda p: p["x"].value)`` is an
    ``eqx.tree_at`` selector, so ``eqx.tree_at`` rewrote the one copy the
    selector reaches and left the others live in the forward model.

    Measured on the ``forked`` fixture (forward 20.0) before this guard: the
    space validated, and binding ``V=0`` gave 10.0 where 0.0 is correct.
    Finite, correctly shaped, wrong — in the inference path, which is what
    this package is for. Gradients were never the problem: both copies carry
    their own, and the two sum to the true derivative. The pytree is simply a
    correct TWO-parameter model where a one-parameter model was declared,
    which makes ``validate`` a complete gate.
    """

    def test_the_fixture_really_aliases_x_and_not_out(self, forked, template_state):
        assert forked.aliased == ("x",)
        assert jnp.allclose(forked(template_state).data, 20.0)

    def test_binding_into_the_aliased_node_is_refused(self, forked):
        space = ParameterSpace(
            latents=[Latent("V", init=0.0)],
            bindings=[Bind("V", into=lambda p: p["x"].value)],
        )
        with pytest.raises(ParameterSpaceError) as excinfo:
            space.validate(forked)
        message = str(excinfo.value)
        assert "'V'" in message  # the latent
        assert "'x'" in message  # the node

    def test_the_refusal_says_what_breaks_and_what_to_do_instead(self, forked):
        space = ParameterSpace(
            latents=[Latent("V", init=0.0)],
            bindings=[Bind("V", into=lambda p: p["x"].value)],
        )
        with pytest.raises(ParameterSpaceError) as excinfo:
            space.validate(forked)
        message = str(excinfo.value)
        assert "more than one place" in message
        assert "several paths" in message
        assert "downstream" in message

    def test_forward_fn_is_gated_too(self, forked, template_state):
        """The entry point users actually call validates first; it must refuse."""
        space = ParameterSpace(
            latents=[Latent("V", init=0.0)],
            bindings=[Bind("V", into=lambda p: p["x"].value)],
        )
        with pytest.raises(ParameterSpaceError, match="'x'"):
            space.forward_fn(forked, template_state)

    def test_a_selector_spelled_out_to_the_other_copy_is_refused_too(self, forked):
        """The guard is about the node, not about which copy the lambda picks.

        ``p["x"]`` reaches the first copy. A selector written out by hand can
        name the second one, and rewriting THAT leaves the first live — the
        same silent wrong answer from the other end.
        """
        space = ParameterSpace(
            latents=[Latent("V", init=0.0)],
            bindings=[Bind("V", into=lambda p: p.operator.stages[0].branches[1].value)],
        )
        with pytest.raises(ParameterSpaceError, match="'x'"):
            space.validate(forked)

    def test_an_assembly_nested_in_a_larger_pipeline_is_covered(self, forked):
        """The guard finds assemblies wherever they sit, not only at the root.

        ``Pipeline(assembly, post)`` is a perfectly ordinary way to bolt a
        processing stage onto a compiled graph, and ``p["asm"]["x"].value`` is
        the selector it makes natural — so a guard that only looked at the top
        level would leave the same silent wrong answer one wrapper away.
        """
        nested = Pipeline(
            forked, _ForkScale(factor=jnp.array(2.0)), names=("asm", "post")
        )
        space = ParameterSpace(
            latents=[Latent("V", init=0.0)],
            bindings=[Bind("V", into=lambda p: p["asm"]["x"].value)],
        )
        with pytest.raises(ParameterSpaceError, match="'x'"):
            space.validate(nested)

    def test_an_ordinary_node_on_the_same_graph_still_binds(
        self, forked, template_state
    ):
        """The guard must not fire on the shape it is not about.

        ``out`` sits below the rejoin, so the fold embeds it once. Asserted on
        the forward OUTPUT: 20 scaled by 0.5 is 10, and a guard that quietly
        stopped writing would leave it at 20.
        """
        space = ParameterSpace(
            latents=[Latent("F", init=1.0)],
            bindings=[Bind("F", into=lambda p: p["out"].factor)],
        )
        space.validate(forked)  # must not raise
        forward, _ = space.forward_fn(forked, template_state)
        assert jnp.allclose(forward({"F": jnp.array(0.5)}), 10.0)
        assert jnp.allclose(forward({"F": jnp.array(1.0)}), 20.0)

    def test_reading_the_aliased_node_still_works(self, forked):
        """The read is an inspection API and stays one; only the write refuses.

        Pinned here beside the guard so that "make ``__getitem__`` refuse too"
        breaks a test in the same file as the guard it would be tightening.
        """
        assert isinstance(forked["x"], _ForkSrc)
        assert forked["x"].value == 10.0
