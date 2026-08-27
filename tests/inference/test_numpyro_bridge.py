"""Tests for the NumPyro bridge: sites, model construction, MCMC recovery.

Sites are named by their latents, so a re-parameterized model is sampled in
the coordinates it was declared in — `log_gain`, not the tree path of whatever
leaf that logarithm eventually lands in.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro", reason="numpyro not installed")
import numpyro.distributions as dist  # noqa: E402
from numpyro.handlers import seed, substitute, trace  # noqa: E402

from rheplicant.core.errors import ParameterSpaceError, StateValidationError  # noqa: E402
from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    Bind,
    Latent,
    ParameterSpace,
    predict_from_samples,
    to_numpyro_model,
)
from rheplicant.radio import GainOperator, SkyOperator, assemble  # noqa: E402

TRUE_GAIN = 1.1
SKY = 100.0
SIGMA = 0.5


@pytest.fixture
def twin():
    return assemble(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(1.0)),  # model starts mis-calibrated
    )


@pytest.fixture
def space():
    return ParameterSpace.direct(
        "gain", init=1.0, into=lambda p: p["gain"].gain, prior=dist.Normal(1.0, 0.3)
    )


@pytest.fixture
def observed(template_state):
    truth = assemble(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(TRUE_GAIN)),
    )(template_state).data
    noise = SIGMA * jax.random.normal(jax.random.key(99), truth.shape)
    return truth + noise


class TestAJointPriorNeedsFloat64:
    """D25: the refusal, and the sibling that proves it is about precision.

    ``JeffreysPrior.log_density`` is evaluated inside the model body at every
    leapfrog step, and its information matrix is refused at single precision --
    measured on an exactly degenerate block, float32 gives -27.52 where the
    block honestly gives -338.05, in a term the sampler exponentiates.

    The refusal is raised HERE, at construction, rather than left to arrive
    from across the seam. ``jax_enable_x64`` is a tracing-time global and
    NumPyro traces ``model`` long after this function returns, so there is no
    block to open around the arithmetic; what would arrive instead is a
    translated refusal quoting "a Jeffreys information matrix", naming neither
    the ``joint_prior`` nor the document that declared it.

    This module is float32 (it has no x64 fixture), which is what makes the
    condition free here and is why the test lives in this file rather than
    beside the rest of the Jeffreys tests -- ``test_jeffreys_prior.py`` carries
    a module-scope autouse x64 fixture and could not state this at all.
    """

    @staticmethod
    def _covered_space():
        from rheplicant.inference import JeffreysPrior

        return ParameterSpace(
            latents=(Latent(name="gain", init=jnp.array(1.0)),),
            bindings=(Bind("gain", into=lambda p: p["gain"].gain),),
            joint_prior=JeffreysPrior(over=("gain",)),
        )

    def test_the_ambient_precision_is_single_here(self):
        """The condition, stated rather than assumed.

        If this module ever acquires an x64 fixture, the refusal test below
        stops being about anything and would pass by not running its branch.
        """
        assert jnp.result_type(float) == jnp.float32

    def test_a_declared_joint_prior_is_refused_at_construction(
        self, twin, observed, template_state
    ):
        with pytest.raises(StateValidationError) as caught:
            to_numpyro_model(
                twin, template_state, self._covered_space(), SIGMA
            )
        message = str(caught.value)
        # What was declared, and where the number goes wrong -- a refusal that
        # only says "float32" leaves the reader to guess whether it matters.
        assert "JeffreysPrior" in message and "gain" in message
        assert "float32" in message
        assert "310" in message
        # Both routes out, by name.
        assert "jax_enable_x64" in message
        assert "inference.joint_prior" in message

    def test_the_same_space_is_accepted_in_a_float64_session(
        self, twin, observed, template_state
    ):
        """The sibling: the refusal is about the precision, not the prior.

        Without this, a guard that refused every declared ``joint_prior``
        outright would satisfy the test above and delete a working feature.
        The space is rebuilt INSIDE the block, because a space declared outside
        it carries float32 constants in and is refused separately.
        """
        was = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        try:
            model = to_numpyro_model(
                twin, template_state, self._covered_space(), SIGMA
            )
            # `substitute`, not `seed` alone: the covered latent's site is an
            # ImproperUniform and an improper density cannot be sampled from --
            # which is the point of it, and is why a real run passes
            # `init_to_declared`. The value is supplied here for the same
            # reason.
            conditioned = substitute(
                seed(model, rng_seed=0), data={"gain": jnp.array(1.0)}
            )
            sites = trace(conditioned).get_trace(observed)
        finally:
            jax.config.update("jax_enable_x64", was)
        assert "joint_prior" in sites, "the factor site is not in the trace"
        assert np.isfinite(float(sites["joint_prior"]["fn"].log_factor))


class TestModelConstruction:
    def test_a_latent_without_a_prior_is_refused(self, twin, template_state):
        """Free parameters are fine for an optimizer and meaningless in a posterior."""
        space = ParameterSpace.direct("gain", init=1.0, into=lambda p: p["gain"].gain)
        with pytest.raises(ParameterSpaceError, match="no prior"):
            to_numpyro_model(twin, template_state, space, noise_std=1.0)

    def test_the_space_is_validated_against_the_pipeline(self, twin, template_state):
        space = ParameterSpace.direct(
            "gain", init=jnp.zeros(3), into=lambda p: p["gain"].gain,
            prior=dist.Normal(jnp.zeros(3), 1.0),
        )
        with pytest.raises(ParameterSpaceError, match="shape"):
            to_numpyro_model(twin, template_state, space, noise_std=1.0)


class TestModel:
    def test_sites_are_named_after_their_latents(self, twin, space, template_state):
        model = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        tr = trace(seed(model, jax.random.key(0))).get_trace()
        assert "gain" in tr
        assert tr["prediction"]["value"].shape == (8, 4)
        assert tr["obs"]["value"].shape == (8, 4)

    def test_a_reparameterized_site_is_named_for_what_is_sampled(
        self, twin, template_state
    ):
        """The site is `log_gain` — the coordinate NUTS explores — even though
        the value lands in a leaf called `gain`."""
        space = ParameterSpace.direct(
            "log_gain", init=0.0, into=lambda p: p["gain"].gain, fn=jnp.exp,
            prior=dist.Normal(0.0, 0.3),
        )
        model = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        tr = trace(seed(model, jax.random.key(0))).get_trace()
        assert "log_gain" in tr
        assert "gain" not in tr

    def test_sampled_noise_std(self, twin, space, template_state):
        model = to_numpyro_model(
            twin, template_state, space, noise_std=dist.HalfNormal(1.0)
        )
        tr = trace(seed(model, jax.random.key(0))).get_trace()
        assert "noise_std" in tr

    def test_flags_masked_likelihood(self, twin, space, template_state, observed):
        """Corrupting a FLAGGED sample must not change the masked log density."""
        from numpyro.infer.util import log_density

        flags = jnp.zeros(observed.shape, bool).at[0, 0].set(True)
        masked = to_numpyro_model(
            twin, template_state, space, noise_std=SIGMA, flags=flags
        )
        unmasked = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        corrupted = observed.at[0, 0].set(1e6)
        params = {"gain": jnp.array(TRUE_GAIN)}

        def ld(model, obs):
            return float(log_density(model, (), {"observed": obs}, params)[0])

        assert ld(masked, observed) == pytest.approx(ld(masked, corrupted))
        assert ld(unmasked, observed) != pytest.approx(ld(unmasked, corrupted))


class TestMCMCRecovery:
    def test_nuts_recovers_gain(self, twin, space, template_state, observed):
        model = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model), num_warmup=200, num_samples=200,
            progress_bar=False,
        )
        mcmc.run(jax.random.key(0), observed=observed)
        samples = mcmc.get_samples()
        posterior_mean = float(samples["gain"].mean())
        # analytic posterior std ~ sigma / (sky * sqrt(n)) ~ 0.5/(100*sqrt(32)) ~ 1e-3
        assert abs(posterior_mean - TRUE_GAIN) < 0.01
        assert float(samples["gain"].std()) < 0.01

    def test_nuts_recovers_a_tied_reparameterization(self, template_state):
        """Two gain stages, ONE latent, sampled in log space — a model the
        positional-prior scheme could not express at all.

        **1000/1000 rather than 300/300, and the reason is measured.** At
        300/300 this chain's ESS at ``key(0)`` is **28** — the worst of the
        four seeds tried, which give 28, 97, 119 and 126. A chain that thin is
        not sampling the posterior, it is wandering in it, and where it stops
        is decided by the exact floating-point trajectory: the recovered gain
        was within 0.0026 here and **0.72 out** on the x86_64 CI runner, which
        is roughly 160 posterior standard deviations. The assertion below was
        not measuring recovery, it was recording one machine's luck.

        At 1000/1000 the same four seeds give ESS 287 to 450, r-hat at most
        1.006, and a recovery error of at most 1e-4 — an order of magnitude
        inside the tolerance, on every seed, rather than at the mercy of one.

        ESS is asserted BEFORE recovery so that a chain which stops mixing
        again fails as "did not converge" rather than as "wrong number". The
        two have different causes and different fixes, and a bare recovery
        assertion cannot tell them apart.
        """
        def two_stage(gain):
            return Pipeline(
                SkyOperator(amplitude=jnp.array(SKY)),
                GainOperator(gain=jnp.array(gain)),
                GainOperator(gain=jnp.array(gain)),
                names=("sky", "gain_a", "gain_b"),
            )

        data = two_stage(TRUE_GAIN)(template_state).data + SIGMA * jax.random.normal(
            jax.random.key(4), (8, 4)
        )
        space = ParameterSpace.direct(
            "log_gain", init=0.0,
            into=(lambda p: p["gain_a"].gain, lambda p: p["gain_b"].gain),
            fn=jnp.exp, prior=dist.Normal(0.0, 0.5),
        )
        model = to_numpyro_model(two_stage(1.0), template_state, space, noise_std=SIGMA)
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model), num_warmup=1000, num_samples=1000,
            progress_bar=False,
        )
        mcmc.run(jax.random.key(0), observed=data)
        draws = mcmc.get_samples()["log_gain"]
        ess = float(
            numpyro.diagnostics.effective_sample_size(
                np.asarray(draws).reshape(1, -1)
            )
        )
        assert ess > 150, f"the chain barely mixed (ESS {ess:.0f}); see the docstring"
        recovered = float(jnp.exp(draws).mean())
        assert abs(recovered - TRUE_GAIN) < 0.01

    def test_posterior_predictive(self, twin, space, template_state, observed):
        model = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model), num_warmup=100, num_samples=50,
            progress_bar=False,
        )
        mcmc.run(jax.random.key(1), observed=observed)
        preds = predict_from_samples(twin, template_state, space, mcmc.get_samples())
        assert preds.shape == (50, 8, 4)
        # predictions are gain*sky: mean close to the observed mean signal
        assert abs(float(preds.mean()) - SKY * TRUE_GAIN) < 1.0

    def test_posterior_predictive_through_a_derived_binding(self, template_state):
        """vmapping the posterior must survive a latent whose shape differs
        from the leaf it feeds."""
        n_time = template_state.coords.time.shape[0]
        twin = assemble(
            SkyOperator(amplitude=jnp.array(SKY)),
            GainOperator(gain=jnp.ones(n_time)),
        )
        space = ParameterSpace(
            latents=[
                Latent("g0", init=1.0, prior=dist.Normal(1.0, 0.1)),
                Latent("slope", init=0.0, prior=dist.Normal(0.0, 0.05)),
            ],
            bindings=[
                Bind(
                    ("g0", "slope"),
                    into=lambda p: p["gain"].gain,
                    fn=lambda g0, slope: g0 + slope * jnp.arange(n_time, dtype=float),
                )
            ],
        )
        samples = {
            "g0": jnp.linspace(0.9, 1.1, 5),
            "slope": jnp.linspace(-0.01, 0.01, 5),
        }
        preds = predict_from_samples(twin, template_state, space, samples)
        assert preds.shape == (5, n_time, 4)

    def test_predict_wrong_per_sample_shape_rejected(self, template_state):
        """Checking only the NAME let a wrong-shaped stack broadcast into the
        leaf and return a finite, correctly-shaped, wrong predictive."""
        n_time = template_state.coords.time.shape[0]
        twin = assemble(
            SkyOperator(amplitude=jnp.array(SKY)),
            GainOperator(gain=jnp.ones(n_time)),
        )
        space = ParameterSpace.direct(
            "gains", init=jnp.ones(n_time), into=lambda p: p["gain"].gain,
            prior=dist.Normal(jnp.ones(n_time), 0.1),
        )
        with pytest.raises(StateValidationError, match="per-sample shape"):
            predict_from_samples(twin, template_state, space,
                                 {"gains": jnp.ones((5,))})    # scalar per draw

    def test_predict_mismatched_draw_counts_rejected(self, template_state):
        twin = assemble(
            SkyOperator(amplitude=jnp.array(SKY)),
            GainOperator(gain=jnp.array(1.0)),
        )
        space = ParameterSpace(
            latents=[Latent("a", init=1.0, prior=dist.Normal(1.0, 0.1)),
                     Latent("b", init=1.0, prior=dist.Normal(1.0, 0.1))],
            bindings=[Bind(("a", "b"), into=lambda p: p["gain"].gain,
                           fn=lambda a, b: a * b)],
        )
        with pytest.raises(StateValidationError, match="differing numbers of draws"):
            predict_from_samples(twin, template_state, space,
                                 {"a": jnp.ones(5), "b": jnp.ones(7)})

    def test_predict_missing_site_rejected(self, twin, space, template_state):
        with pytest.raises(StateValidationError, match="missing site"):
            predict_from_samples(twin, template_state, space, {"wrong": jnp.zeros(3)})


class TestObservedNoneIsThePriorPredictive:
    """The one thing the two packages spell in OPPOSITE directions.

    Here ``model()`` with no argument has always meant the prior predictive --
    draw the observation, do not condition on anything. On the graph side
    ``model(None)`` means the opposite: use each node's own declared data. The
    facade translates by passing ``{}``, and this class is why that translation
    needs a guard of its own.

    **The graph is built against a placeholder of zeros**, because ``to_graph``
    needs data at build time and ``to_numpyro_model`` is not given any. So a
    translation that passed ``None`` through would condition every
    prior-predictive call on that placeholder and hand back an array of zeros
    wearing the right shape and dtype. Measured: it survives every other test
    in this file -- the shape assertions agree, the site is present, and
    nothing else looks at it.
    """

    def test_no_argument_leaves_the_observation_unconditioned(
        self, twin, space, template_state
    ):
        model = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        tr = trace(seed(model, jax.random.key(0))).get_trace()
        site = tr["obs"]
        assert not site["is_observed"], (
            "the prior-predictive call came back CONDITIONED, which means it was "
            "conditioned on the adapter's placeholder"
        )
        # And the drawn value is a draw, not the placeholder wearing its shape.
        assert float(jnp.max(jnp.abs(site["value"]))) > 0.0

    def test_an_argument_conditions_on_it_and_not_on_the_placeholder(
        self, twin, space, template_state, observed
    ):
        """The other direction, and it is what makes the first one meaningful.

        A facade that ignored its argument and always ran unconditioned would
        satisfy the test above.
        """
        model = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        site = trace(seed(model, jax.random.key(0))).get_trace(observed)["obs"]
        assert site["is_observed"]
        assert jnp.allclose(site["value"], observed)
        # Named separately: `observed` is not all-zero, so this rules out the
        # placeholder specifically rather than by implication.
        assert not jnp.allclose(site["value"], jnp.zeros_like(observed))


class TestASampledSigmaCollidingWithALatent:
    """D27's measurement, turned into a refusal rather than left to NumPyro.

    A sampled ``noise_std`` takes the site ``"noise_std"``, which belongs to no
    ``ParameterSpace``. A space that declares a latent of that name collides
    with it -- loudly, today: NumPyro asserts "all sites must have unique names
    but got `noise_std` duplicated". Loud is not the same as useful. That
    sentence names neither the argument that created the second site nor the
    space that declared the first, and it is a bare ``AssertionError``, while
    this package's exception classes are a keeping surface.
    """

    @staticmethod
    def _colliding_space():
        return ParameterSpace(
            latents=(
                Latent(
                    name="noise_std", init=jnp.array(1.0), prior=dist.Normal(1.0, 0.3)
                ),
            ),
            bindings=(Bind("noise_std", into=lambda p: p["gain"].gain),),
        )

    def test_the_collision_is_refused_by_name(self, twin, template_state):
        # `match=` and not only the assertions below: the pinned-refusal census
        # is derived from `raises(..., match=...)` sites, and Appendix B of the
        # migration plan is what a wave reads to find out which sentences it has
        # just become responsible for. A refusal pinned only by `in message`
        # is invisible to both.
        with pytest.raises(ParameterSpaceError, match="noise_std") as caught:
            to_numpyro_model(
                twin, template_state, self._colliding_space(),
                noise_std=dist.HalfNormal(1.0),
            )
        message = str(caught.value)
        assert "noise_std" in message
        assert "HalfNormal" in message
        # Both quantities named, and both ways out.
        assert "rename the latent" in message
        assert "fixed noise_std" in message

    def test_a_fixed_sigma_beside_that_latent_is_left_alone(
        self, twin, template_state, observed
    ):
        """The sibling: nothing collides when the sigma is not sampled.

        Without it a refusal that fired on the NAME alone would satisfy the
        test above and delete an ordinary, if unfortunately named, latent.
        """
        model = to_numpyro_model(
            twin, template_state, self._colliding_space(), noise_std=SIGMA
        )
        tr = trace(seed(model, jax.random.key(0))).get_trace()
        assert "noise_std" in tr
        assert isinstance(tr["noise_std"]["fn"], dist.Normal)
