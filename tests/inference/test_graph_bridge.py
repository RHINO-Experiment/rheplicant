"""The adapter's contract: what it refuses, and what it brings back.

Two halves, and they are the two halves of a seam.

**Going out** (:func:`~rheplicant.inference.graph_bridge.to_graph`): a
declaration this package can make but the graph cannot spell is refused HERE,
before the graph is built. The rule is not tidiness -- it is that building the
graph destroys the evidence. A sigma vector whose axis the prediction cannot
settle becomes, the moment it is broadcast into a ``Normal``, a perfectly
ordinary sigma array answering a question nobody asked; there is no later point
at which the ambiguity can be noticed.

**Coming back** (:func:`~rheplicant.inference.graph_bridge.translate`):
bayesmith raises bayesmith's exceptions, and this package's exception classes
are a keeping surface. So every refusal is re-raised wearing this package's
class -- and the affine refusal comes back carrying its NUMBERS, not a sentence
someone would have to parse.

The numeric acceptance of the adapter lives in ``tests/seam/``, which is
x64-gated. This file is float32 and about refusals and shapes, which is why it
can be here.
"""

import pickle

import bayesmith
import equinox as eqx
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from bayesmith.errors import AffinityRefused, GraphError, NotGaussian, NotLogLinear

from rheplicant.core.combinators import SumOperator
from rheplicant.core.errors import (
    LinearityRefused,
    ParameterSpaceError,
    StateValidationError,
)
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    FlaggedNoise,
    HomoscedasticNoise,
    Latent,
    ParameterSpace,
)
from rheplicant.inference.graph_bridge import (
    INTERNAL_NAMES,
    OBSERVATION,
    PREDICTION,
    Seam,
    SeamRefusal,
    priors_from_keywords,
    to_graph,
    translate,
)
from rheplicant.radio import GainOperator, SkyOperator

SKY_A, SKY_B, GAIN, N_TIME = 100.0, 20.0, 1.5, 8


@pytest.fixture
def instrument():
    scalar = Pipeline(
        SumOperator(
            SkyOperator(amplitude=jnp.array(SKY_A)),
            SkyOperator(amplitude=jnp.array(SKY_B)),
            names=("sky_a", "sky_b"),
        ),
        GainOperator(gain=jnp.array(GAIN)),
        names=("sum", "gain"),
    )
    return eqx.tree_at(lambda p: p["gain"].gain, scalar, jnp.full((N_TIME,), GAIN))


def gain_space(**overrides):
    """A one-latent space over the per-time gain, with keyword overrides."""
    settings = dict(
        init=jnp.full((N_TIME,), GAIN),
        into=lambda p: p["gain"].gain,
        linear=True,
        prior=dist.Normal(jnp.full((N_TIME,), GAIN), 5.0),
    )
    settings.update(overrides)
    return ParameterSpace.direct("gains", **settings)


@pytest.fixture
def space():
    return gain_space()


@pytest.fixture
def data(instrument, space, template_state):
    forward, _ = space.forward_fn(instrument, template_state)
    return forward({"gains": jnp.full((N_TIME,), GAIN)})


@pytest.fixture
def noise():
    return HomoscedasticNoise(sigma=jnp.array(0.5))


# --------------------------------------------------------------- translate --


class TestTranslateBringsRefusalsBackInThisPackagesClasses:
    """The return path. Each family, and what it becomes."""

    PAYLOAD = dict(
        names=("gains",),
        at="the linearisation point",
        errors={0.001: 1.0, 1.0: 2.0, 1000.0: 3.0},
        weighted={0.001: 3.4e-05, 1.0: 73.0, 1000.0: 4.68e7},
        rtol=1.19e-3,
        weighted_rtol=1e-3,
        failed=(1.0, 1000.0),
    )

    def test_an_affine_refusal_becomes_LinearityRefused(self):
        with pytest.raises(LinearityRefused):
            with translate("linear_operator"):
                raise AffinityRefused("not affine", **self.PAYLOAD)

    def test_the_translated_refusal_carries_the_probe_numbers(self):
        """The whole reason G11 exists, asserted from this side.

        A translation that kept only the sentence would leave a caller parsing
        prose for numbers the probe already had -- which is the defect both
        packages diagnosed independently and repaired in the same shape.
        """
        with pytest.raises(LinearityRefused) as caught:
            with translate("linear_operator"):
                raise AffinityRefused("not affine", **self.PAYLOAD)
        refusal = caught.value
        assert refusal.errors == self.PAYLOAD["errors"]
        assert refusal.rtol == pytest.approx(self.PAYLOAD["rtol"])
        assert refusal.failed == self.PAYLOAD["failed"]

    def test_the_second_criterion_comes_across_too(self):
        """``weighted`` is bayesmith's own criterion and this package's probe
        has no counterpart.

        It still crosses, because the translated exception is where a caller
        can read it without the facade growing a keyword for a capability this
        package does not otherwise publish. ``None`` would be wrong: "not
        measured" and "measured as nothing" are different answers.
        """
        with pytest.raises(LinearityRefused) as caught:
            with translate("linear_operator"):
                raise AffinityRefused("not affine", **self.PAYLOAD)
        assert caught.value.weighted == self.PAYLOAD["weighted"]
        assert caught.value.weighted_rtol == pytest.approx(self.PAYLOAD["weighted_rtol"])

    def test_the_original_is_kept_as_the_cause(self):
        with pytest.raises(LinearityRefused) as caught:
            with translate("linear_operator"):
                raise AffinityRefused("not affine", **self.PAYLOAD)
        assert isinstance(caught.value.__cause__, AffinityRefused)

    def test_a_structural_refusal_becomes_a_ParameterSpaceError_naming_the_site(self):
        with pytest.raises(ParameterSpaceError, match="building the block") as caught:
            with translate("building the block"):
                raise GraphError("duplicate node name 'x'")
        assert isinstance(caught.value, SeamRefusal)
        assert caught.value.site == "building the block"
        assert "duplicate node name" in str(caught.value)

    @pytest.mark.parametrize(
        "verdict",
        [
            NotGaussian("not a Normal", reason="not_normal"),
            NotLogLinear("no log route", reason="noise_additive"),
        ],
        ids=["gaussian", "log-linear"],
    )
    def test_a_blameless_verdict_is_caught_and_reported_rather_than_raised(self, verdict):
        """These two are answers, not errors.

        A caller asks "is there an exact route of this kind here" in order to
        branch, so the verdict ends the block and lands on the ``Seam`` instead
        of propagating. The block's later statements do NOT run -- which is why
        the verdict has to be readable from the ``Seam`` and not from a variable
        the block never assigned.
        """
        reached = []
        with translate("log space") as seam:
            reached.append("before")
            raise verdict
        assert reached == ["before"]
        assert seam.blameless is verdict
        assert seam.refused

    def test_a_block_that_finishes_reports_no_verdict(self):
        """The other half of the pair.

        Without it, a ``Seam`` that reported ``refused`` unconditionally would
        pass every assertion above.
        """
        with translate("log space") as seam:
            pass
        assert seam.blameless is None
        assert not seam.refused

    def test_an_unrelated_exception_is_not_swallowed(self):
        """The seam translates bayesmith's refusals; it is not an except-all.

        A ``ZeroDivisionError`` inside the block is a bug in the block, and a
        context manager that quietly ate it would turn every such bug into a
        silent no-op with a plausible-looking ``Seam``.
        """
        with pytest.raises(ZeroDivisionError):
            with translate("anything"):
                _ = 1 / 0

    def test_a_seam_refusal_survives_a_pickle_round_trip(self):
        """pytest-xdist serialises exceptions across workers.

        A required keyword breaks Python's default ``cls(*args)`` rebuild, so
        without ``__reduce__`` the refusal reaches the report as a ``TypeError``
        about a missing argument -- displacing the error it was reporting, and
        from a stack frame nowhere near it.
        """
        restored = pickle.loads(pickle.dumps(SeamRefusal("boom", site="somewhere")))
        assert isinstance(restored, SeamRefusal)
        assert restored.site == "somewhere"
        assert str(restored) == "boom"

    def test_the_seam_record_is_what_it_says_it_is(self):
        assert Seam(site="x").site == "x"
        assert not Seam(site="x").refused


# ----------------------------------------------------------------- to_graph --


class TestToGraphBuildsThreeLayersAndNothingElse:
    def test_the_nodes_are_the_latents_plus_two_internal_ones(
        self, instrument, space, data, noise, template_state
    ):
        graph = to_graph(space, instrument, template_state, data, noise)
        assert [node.name for node in graph.nodes] == ["gains", PREDICTION, OBSERVATION]

    def test_the_declared_linear_latent_reaches_the_graph_as_a_linear_in_claim(
        self, instrument, space, data, noise, template_state
    ):
        """``linear=True`` here has to become ``linear_in=`` there, or the
        conjugate routes are simply unreachable and every model quietly goes
        to NUTS -- slower, and correct, which is why nothing would report it.
        """
        graph = to_graph(space, instrument, template_state, data, noise)
        prediction = next(node for node in graph.nodes if node.name == PREDICTION)
        assert tuple(prediction.linear_in) == ("gains",)

    def test_an_undeclared_linear_latent_makes_no_claim(
        self, instrument, data, noise, template_state
    ):
        graph = to_graph(
            gain_space(linear=False), instrument, template_state, data, noise
        )
        prediction = next(node for node in graph.nodes if node.name == PREDICTION)
        assert tuple(prediction.linear_in) == ()

    def test_the_observation_sigma_is_the_noise_model_s_own(
        self, instrument, space, data, template_state
    ):
        """Asked of ``noise.std``, never re-derived from its fields.

        Checked by giving the model a sigma no default would produce and
        reading it back off the graph: a second spelling of the radiometer's
        floor or its absolute value would be finite, correctly shaped and
        wrong, which nothing downstream can see.
        """
        graph = to_graph(
            space, instrument, template_state, data, HomoscedasticNoise(sigma=jnp.array(0.137))
        )
        sigma = bayesmith.noise_std_at(graph, {"gains": jnp.full((N_TIME,), GAIN)})
        assert jnp.allclose(sigma[OBSERVATION], 0.137)

    def test_each_latent_gets_its_own_prior_and_not_the_last_one(
        self, instrument, data, noise, template_state
    ):
        """Python's late binding, pinned.

        A closure built inline in the loop would capture the loop variable and
        give every node the LAST latent's prior. The result is finite, the
        right shape, and a different model -- so it is asserted rather than
        trusted, with two latents whose priors are far enough apart to tell.
        """
        two = ParameterSpace(
            latents=(
                Latent(name="amp_a", init=jnp.array(SKY_A), prior=dist.Normal(SKY_A, 1.0)),
                Latent(name="amp_b", init=jnp.array(SKY_B), prior=dist.Normal(SKY_B, 9.0)),
            ),
            bindings=(
                ParameterSpace.direct(
                    "amp_a", init=SKY_A, into=lambda p: p["sum"]["sky_a"].amplitude
                ).bindings[0],
                ParameterSpace.direct(
                    "amp_b", init=SKY_B, into=lambda p: p["sum"]["sky_b"].amplitude
                ).bindings[0],
            ),
        )
        graph = to_graph(two, instrument, template_state, data, noise)
        widths = {
            node.name: float(node.dist_fn().scale)
            for node in graph.nodes
            if node.name in {"amp_a", "amp_b"}
        }
        assert widths == {"amp_a": 1.0, "amp_b": 9.0}


class TestToGraphRefusesWhatTheGraphCannotSpell:
    def test_a_sigma_vector_with_more_than_one_reading(
        self, instrument, space, template_state
    ):
        """The founding pre-validation, on a square prediction grid.

        Past ``to_graph`` the vector has been broadcast into a distribution and
        the reading is settled -- silently, by trailing-axis alignment. So it
        is refused before the graph exists, which is the general rule this
        module's pre-validation follows.
        """
        # A square state built HERE rather than the shared 8x4 one, and not
        # skipped when the shared one is not square: a length-n sigma has only
        # one reading against a rectangular prediction, so the shared fixture
        # cannot exercise this refusal at all -- and a skip standing in for the
        # founding pre-validation would read as a pass.
        square_state = eqx.tree_at(
            lambda state: state.coords.freq,
            template_state,
            jnp.linspace(60e6, 85e6, N_TIME),
        )
        square_space = gain_space()
        forward, _ = square_space.forward_fn(instrument, square_state)
        prediction = forward({"gains": jnp.full((N_TIME,), GAIN)})
        assert prediction.shape == (N_TIME, N_TIME)
        ambiguous = HomoscedasticNoise(sigma=jnp.linspace(0.01, 1.0, N_TIME))
        with pytest.raises(StateValidationError, match="more than one legitimate reading"):
            to_graph(square_space, instrument, square_state, prediction, ambiguous)

    def test_observed_of_the_wrong_shape(
        self, instrument, space, data, noise, template_state
    ):
        with pytest.raises(ParameterSpaceError):
            to_graph(space, instrument, template_state, data[0], noise)

    def test_complex_data(self, instrument, space, data, noise, template_state):
        with pytest.raises(StateValidationError, match="complex `observed`"):
            to_graph(space, instrument, template_state, data + 0j, noise)

    def test_a_thing_that_is_not_a_noise_model(
        self, instrument, space, data, template_state
    ):
        """A bare array reaches ``jnp.asarray`` deep inside and comes back as a
        dtype-object ``TypeError`` naming the wrong layer.

        So it is named here instead, with the remedy -- which is a class, not a
        cast.
        """
        with pytest.raises(ParameterSpaceError, match="HomoscedasticNoise"):
            to_graph(space, instrument, template_state, data, jnp.array(0.5))

    @pytest.mark.parametrize("scope", ["per_epoch", "linked"])
    def test_a_scope_with_no_graph_spelling(
        self, instrument, data, noise, template_state, scope
    ):
        """Refused by NAME, with the wave that will spell it.

        Emitting the graph anyway would marginalise a per-epoch quantity as
        though it were fixed, or turn a Markov chain into independent draws --
        both finite, both a different model, neither reported.
        """
        scoped = ParameterSpace(
            latents=(
                Latent(
                    name="gains",
                    init=jnp.full((N_TIME,), GAIN),
                    prior=dist.Normal(jnp.full((N_TIME,), GAIN), 5.0),
                    scope=scope,
                ),
            ),
            bindings=gain_space().bindings,
        )
        with pytest.raises(ParameterSpaceError, match=scope):
            to_graph(scoped, instrument, template_state, data, noise)

    @pytest.mark.parametrize("name", sorted(INTERNAL_NAMES))
    def test_a_latent_named_like_an_internal_node(
        self, instrument, data, noise, template_state, name
    ):
        space = ParameterSpace.direct(
            name,
            init=jnp.full((N_TIME,), GAIN),
            into=lambda p: p["gain"].gain,
            prior=dist.Normal(jnp.full((N_TIME,), GAIN), 5.0),
        )
        with pytest.raises(ParameterSpaceError, match="internal node names"):
            to_graph(space, instrument, template_state, data, noise)

    def test_a_latent_with_no_prior_anywhere(
        self, instrument, data, noise, template_state
    ):
        with pytest.raises(ParameterSpaceError, match="declares no prior"):
            to_graph(gain_space(prior=None), instrument, template_state, data, noise)

    def test_a_supplied_prior_for_a_latent_that_already_declares_one(
        self, instrument, space, data, noise, template_state
    ):
        with pytest.raises(ParameterSpaceError, match="also declares"):
            to_graph(
                space,
                instrument,
                template_state,
                data,
                noise,
                priors={"gains": dist.Normal(0.0, 1.0)},
            )

    def test_a_supplied_prior_for_a_latent_that_does_not_exist(
        self, instrument, space, data, noise, template_state
    ):
        with pytest.raises(ParameterSpaceError, match="which this space does not declare"):
            to_graph(
                space,
                instrument,
                template_state,
                data,
                noise,
                priors={"gain": dist.Normal(0.0, 1.0)},
            )

    def test_a_complex_latent_whose_prior_is_a_real_normal(self, template_state):
        """Refused, and NOT promoted to ``ComplexNormal``.

        Promotion would have to decide what ``scale`` meant, and the convention
        on the other side is that each half carries ``scale**2`` -- so a silent
        promotion doubles the prior variance the caller declared and reports the
        sqrt(2) as a result.
        """
        from tests.seam.seam_models import ComplexCoeffOperator  # noqa: F401

        pytest.importorskip("bayesmith")
        rows = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        matrix = jnp.ones((rows, 3), dtype=jnp.complex64)
        pipeline = Pipeline(
            ComplexCoeffOperator(coeffs=jnp.zeros(3, dtype=matrix.dtype), matrix=matrix),
            names=("sky",),
        )
        space = ParameterSpace.direct(
            "alm",
            init=jnp.ones(3) + 0j,
            into=lambda p: p["sky"].coeffs,
            prior=dist.Normal(jnp.zeros(3), 1.0),
        )
        forward, _ = space.forward_fn(pipeline, template_state)
        observed = forward({"alm": jnp.ones(3) + 0j})
        with pytest.raises(ParameterSpaceError, match="ComplexNormal"):
            to_graph(
                space,
                pipeline,
                template_state,
                observed,
                HomoscedasticNoise(sigma=jnp.array(0.5)),
            )


# ---------------------------------------------------- the single prior entry --


class TestPriorsFromKeywords:
    def test_a_prior_free_latent_gets_the_supplied_one(self):
        priors = priors_from_keywords(
            gain_space(prior=None), prior_mean=GAIN, prior_std=2.0, caller="here"
        )
        assert set(priors) == {"gains"}
        assert float(priors["gains"].scale.max()) == pytest.approx(2.0)

    def test_a_latent_that_declares_a_prior_is_left_alone(self):
        """Absent from the result, not overwritten with a copy.

        ``to_graph`` refuses a name that is in both places, so a helper that
        echoed the declaration back would make its own output un-passable.
        """
        assert priors_from_keywords(gain_space(), caller="here") == {}

    def test_a_supplied_value_contradicting_the_declaration_is_refused(self):
        """The check that has always guarded these keywords, still guarding
        them on the path to a graph.

        Without it, one of the two silently wins and the other reads like it
        was in force -- and the same declaration reaches NUTS unchanged, so the
        two exits would target different posteriors from one space.
        """
        with pytest.raises(ParameterSpaceError, match="declares"):
            priors_from_keywords(gain_space(), prior_std=0.001, caller="here")

    def test_a_prior_free_latent_with_no_supplied_width_is_refused(self):
        with pytest.raises(ParameterSpaceError, match="no prior for latent"):
            priors_from_keywords(gain_space(prior=None), prior_mean=GAIN, caller="here")

    def test_a_dict_keyword_reaches_the_latent_it_names(self):
        priors = priors_from_keywords(
            gain_space(prior=None),
            prior_mean={"gains": jnp.full((N_TIME,), 3.0)},
            prior_std={"gains": 0.25},
            caller="here",
        )
        assert float(priors["gains"].loc.max()) == pytest.approx(3.0)
        assert float(priors["gains"].scale.max()) == pytest.approx(0.25)


def test_importing_the_adapter_does_not_import_jax_eagerly():
    """A sanity check on the module's own shape, not on jax.

    The adapter imports bayesmith INSIDE its functions so that importing this
    package does not drag in the sibling's whole graph layer, and so a missing
    bayesmith fails at the call that needed it rather than at import of
    something sitting beside it. The cheap way to keep that true is to assert
    the module body holds no top-level bayesmith import.
    """
    import ast
    import pathlib

    import rheplicant.inference.graph_bridge as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text())
    top_level = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    named = {
        getattr(node, "module", None) or "" for node in top_level
    } | {
        alias.name for node in top_level if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert not any(name.startswith("bayesmith") for name in named if name)
    assert not any(name.startswith("numpyro") for name in named if name)


class TestFlaggedNoiseCrossesAsADeclaredMask:
    """G1 landed on the far side, so this seam translates instead of refusing.

    The refusal it replaces, which asserted that flagged noise could not cross
    this seam, is RETIRED -- the triage table's third column, and it needs a
    reason: it pinned a GAP, the gap is closed, and a test that pins an absence
    outlives its subject. What it protected is pinned positively below.

    THIS side spells an unobserved sample ``sigma = inf``. A graph node cannot:
    ``Normal(mu, inf)`` has log-density ``-inf`` everywhere. So the flags cross
    as the node's ``mask`` and the scale stays finite -- the far side's D20,
    reached from the side that has to speak it.
    """

    @staticmethod
    def _flagged(data, channel):
        return FlaggedNoise(
            base=HomoscedasticNoise(sigma=jnp.array(0.5)),
            flags=jnp.zeros(data.shape, dtype=bool).at[:, channel].set(True),
        )

    def test_the_mask_is_the_negation_of_the_flags(
        self, instrument, space, data, template_state
    ):
        """Polarity, against the flagged CHANNEL rather than against a count.

        ``FlaggedNoise.flags`` is True where a sample was FLAGGED; the graph's
        ``mask`` is True where a sample was TAKEN. They are negations, both
        boolean and both the data's shape, so swapping them is invisible to
        every shape and dtype check there is. A count would not catch it either
        unless the flagged fraction happened not to be half.
        """
        graph = to_graph(
            space, instrument, template_state, data, self._flagged(data, 2)
        )
        mask = np.asarray(graph.node("__data__").observed_mask)
        assert mask.shape == tuple(data.shape)
        assert not mask[:, 2].any(), "the flagged channel came through as TAKEN"
        assert mask[:, 0].all() and mask[:, 1].all()

    def test_the_declared_scale_is_finite_where_the_flags_are(
        self, instrument, space, data, template_state
    ):
        """The far side refuses a non-finite scale, by name and on purpose.

        "The sigma expression produced an infinity" and "this sample was
        flagged" need different fixes; the second is now said by the mask, so
        the scale says only the first. The value put there is the BASE model's
        own sigma rather than a placeholder -- it cannot reach any answer, and
        using the instrument's number means the graph reads as the model.
        """
        graph = to_graph(
            space, instrument, template_state, data, self._flagged(data, 2)
        )
        node = graph.node("__data__")
        env = {name: value for name, value in space.initial_values().items()}
        from bayesmith.exact.gaussian import gaussian_parts
        from bayesmith.graph.evaluate import evaluate

        _, scale = gaussian_parts(graph, node, evaluate(graph, env))
        scale = np.asarray(scale)
        assert np.all(np.isfinite(scale)), "an inf reached the declared scale"
        assert np.allclose(scale, 0.5)

    def test_an_unflagged_noise_model_declares_no_mask_at_all(
        self, instrument, space, data, template_state
    ):
        """``None``, not an all-True array: absence is cheaper and is the truth.

        Without this, a translation that handed every graph a full mask would
        pass every assertion above while making a mask the normal case.
        """
        graph = to_graph(
            space,
            instrument,
            template_state,
            data,
            HomoscedasticNoise(sigma=jnp.array(0.5)),
        )
        assert graph.node("__data__").observed_mask is None


class TestAJointPriorCrossesAsADeclaredFactor:
    """G13 landed on the far side, so this seam declares instead of refusing.

    The refusal it replaces — ``to_graph`` telling the caller that a joint
    prior needs a factor site nobody had built yet — is RETIRED, the triage
    table's third column, for the reason the masking one was: it pinned a GAP,
    the gap is closed, and a test that pins an absence does not outlive its
    subject. What it protected is pinned positively here.

    **What makes this seam worth guarding is that dropping it is invisible.**
    A graph built without the declaration has the right nodes, the right
    shapes, the right dtypes; it validates, it samples, and every convergence
    diagnostic is clean. It is simply a different posterior — the likelihood
    alone — which is the sentence the retired refusal used to say out loud.
    So the guard below is not "the field is set" but "the potential moved, by
    this much".
    """

    @staticmethod
    def _covered_space(**overrides):
        """The gain space with its prior handed to a ``JeffreysPrior``.

        A covered latent must NOT declare a ``Latent(prior=...)`` of its own —
        ``ParameterSpace.__check_init__`` refuses that as two priors on one
        quantity, and the far side refuses it a second time by type. So this
        space drops the ``Normal`` the sibling fixtures carry.
        """
        from rheplicant.inference import JeffreysPrior

        return ParameterSpace(
            latents=(Latent(name="gains", init=jnp.full((N_TIME,), GAIN), linear=True),),
            bindings=gain_space().bindings,
            joint_prior=JeffreysPrior(over=("gains",), **overrides),
        )

    def test_the_graph_carries_the_declaration(
        self, instrument, data, noise, template_state
    ):
        graph = to_graph(
            self._covered_space(), instrument, template_state, data, noise
        )
        assert isinstance(graph.joint_prior, bayesmith.JeffreysPrior)
        assert graph.joint_prior.over == ("gains",)

    def test_an_explicit_rank_rtol_crosses_rather_than_defaulting(
        self, instrument, data, noise, template_state
    ):
        """``None`` means the same default on both sides, which is why this
        needs a NON-default value to say anything.

        An explicit ``rank_rtol`` is a caller's decision about where a null
        eigenvalue starts. Dropping it here would leave the rank verdict taken
        at a tolerance nobody asked for — finite, plausible, a different prior,
        and nothing in the graph's shape or dtype to say so.
        """
        graph = to_graph(
            self._covered_space(rank_rtol=1e-5), instrument, template_state, data, noise
        )
        assert graph.joint_prior.rank_rtol == 1e-5

    def test_the_covered_latent_is_declared_flat(
        self, instrument, data, noise, template_state
    ):
        """Improper, and improper by the exact type the far side checks for.

        A covered latent still needs a node — a sampler needs the coordinate —
        but it must not carry a density, because the whole density over the
        block arrives once at the factor. ``_check_against`` refuses anything
        that is not an ``ImproperUniform`` here, so the spelling is part of the
        contract and not a stylistic choice.
        """
        graph = to_graph(
            self._covered_space(), instrument, template_state, data, noise
        )
        from bayesmith.graph.evaluate import apply_probabilistic, evaluate

        values = dict(self._covered_space().initial_values())
        distribution = apply_probabilistic(
            graph, graph.node("gains"), evaluate(graph, values)
        )
        assert isinstance(distribution, dist.ImproperUniform)

    def test_an_uncovered_latent_keeps_its_own_prior(
        self, instrument, space, data, noise, template_state
    ):
        """The flat declaration is for the COVERED ones, and only those.

        Without this, a wiring that declared every latent flat would pass every
        test above and quietly delete the priors of the latents the block does
        not name.
        """
        graph = to_graph(space, instrument, template_state, data, noise)
        from bayesmith.graph.evaluate import apply_probabilistic, evaluate

        values = dict(space.initial_values())
        distribution = apply_probabilistic(
            graph, graph.node("gains"), evaluate(graph, values)
        )
        assert isinstance(distribution, dist.Normal)
        assert graph.joint_prior is None

    # The guard a forgotten declaration would fail -- "the potential moved, and
    # by this much" -- CANNOT live here. `JeffreysPrior.information` refuses
    # ambient float32 by name (D9's third caller), and this file is float32 on
    # purpose. It is in `tests/seam/test_g13_joint_prior.py`, x64-gated, which
    # is where this file's own header says the adapter's numeric acceptance
    # goes. The structural facts above are the half that can be stated here.

    def test_a_supplied_prior_for_a_covered_latent_is_refused(
        self, instrument, data, noise, template_state
    ):
        """``priors=`` and ``over=`` naming the same latent is the same defect
        the declaration-time check refuses, arriving by the other door.

        ``ParameterSpace.__check_init__`` catches ``Latent(prior=...)`` against
        ``over=``; nothing caught a call-site ``priors=`` against it, because
        that dictionary does not exist until ``to_graph`` is called.
        """
        with pytest.raises(ParameterSpaceError, match="two priors on one quantity"):
            to_graph(
                self._covered_space(),
                instrument,
                template_state,
                data,
                noise,
                priors={"gains": dist.Normal(jnp.full((N_TIME,), GAIN), 5.0)},
            )
