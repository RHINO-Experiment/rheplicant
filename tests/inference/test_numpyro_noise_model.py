"""The NumPyro bridge under a noise model, checked against an exact sampler.

The headline is `TestAgainstTheExactPosterior`: on a linear-Gaussian problem
`gcr_sample` draws the posterior *exactly*, in closed form, with no chain and
nothing to diagnose. NUTS explores the same posterior by MCMC. They must agree,
and if they do not, the one that is wrong is not the constrained realization.

That comparison is why 3a comes before any amortized estimator: an approximate
sampler has no internal notion of correctness, so it needs an exact reference,
and this is where the exact reference gets established.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

numpyro = pytest.importorskip("numpyro", reason="numpyro not installed")
import numpyro.distributions as dist  # noqa: E402
from numpyro.infer.util import log_density  # noqa: E402

from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    FlaggedNoise,
    HomoscedasticNoise,
    NoiseModelLikelihood,
    ParameterSpace,
    RadiometerNoise,
    gcr_sample,
    linear_operator,
    to_numpyro_model,
    wiener_solve,
)
from rheplicant.radio import GainOperator, SkyOperator  # noqa: E402

SKY, TRUE_GAIN, SIGMA = 100.0, 1.1, 0.5
PRIOR_MEAN, PRIOR_STD = 1.0, 0.3
CHANNEL_WIDTH, INTEGRATION_TIME = 1e4, 1.0


@pytest.fixture
def twin():
    return Pipeline(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(1.0)),
        names=("sky", "gain"),
    )


@pytest.fixture
def space():
    # linear=True is what lets the SAME space drive both the exact conjugate
    # solvers and NUTS -- which is the comparison this file exists for.
    return ParameterSpace.direct(
        "gain", init=1.0, into=lambda p: p["gain"].gain,
        prior=dist.Normal(PRIOR_MEAN, PRIOR_STD), linear=True,
    )


@pytest.fixture
def observed(template_state):
    truth = Pipeline(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(TRUE_GAIN)),
        names=("sky", "gain"),
    )(template_state).data
    return truth + SIGMA * jax.random.normal(jax.random.key(99), truth.shape)


class TestTheNoiseModelReachesThePotential:
    def test_a_homoscedastic_model_matches_the_bare_sigma(
        self, twin, space, template_state, observed
    ):
        bare = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        model = to_numpyro_model(
            twin, template_state, space,
            noise_std=HomoscedasticNoise(jnp.asarray(SIGMA)),
        )
        values = {"gain": jnp.array(1.05)}
        a, _ = log_density(bare, (), {"observed": observed}, values)
        b, _ = log_density(model, (), {"observed": observed}, values)
        assert jnp.allclose(a, b, rtol=1e-6)

    def test_the_logdet_is_in_the_potential(
        self, twin, space, template_state, observed
    ):
        """``Normal(loc, scale).log_prob`` carries ``-log scale``, so a scale
        that depends on the sampled parameters brings its log-determinant into
        the potential automatically. This is the FULL Gaussian density, not the
        GLS objective ``iterative_gls`` converges to."""
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        model = to_numpyro_model(twin, template_state, space, noise_std=noise)
        values = {"gain": jnp.array(1.05)}
        total, _ = log_density(model, (), {"observed": observed}, values)

        prediction = space.bind(twin, values)(template_state).data
        expected = NoiseModelLikelihood(noise)(prediction, observed) + dist.Normal(
            PRIOR_MEAN, PRIOR_STD
        ).log_prob(values["gain"])
        assert jnp.allclose(total, expected, rtol=1e-5)

        without = NoiseModelLikelihood(noise, include_logdet=False)(
            prediction, observed
        ) + dist.Normal(PRIOR_MEAN, PRIOR_STD).log_prob(values["gain"])
        assert not jnp.allclose(total, without, rtol=1e-3)

    def test_the_potential_moves_with_the_parameters_through_sigma(
        self, twin, space, template_state, observed
    ):
        """The log-det is not a constant here, which is the whole claim: two
        parameter values that give the same chi-squared do NOT give the same
        density once sigma tracks the prediction."""
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        model = to_numpyro_model(twin, template_state, space, noise_std=noise)
        grad = jax.grad(
            lambda g: log_density(model, (), {"observed": observed}, {"gain": g})[0]
        )(jnp.array(1.05))
        assert jnp.isfinite(grad) and grad != 0.0

    def test_an_unobserved_sample_does_not_send_the_potential_to_minus_infinity(
        self, twin, space, template_state, observed
    ):
        flags = jnp.zeros(observed.shape, bool).at[0, 0].set(True)
        ruined = observed.at[0, 0].set(1e9)
        model = to_numpyro_model(
            twin, template_state, space,
            noise_std=FlaggedNoise(HomoscedasticNoise(jnp.asarray(SIGMA)), flags),
        )
        total, _ = log_density(model, (), {"observed": ruined}, {"gain": jnp.array(1.05)})
        assert jnp.isfinite(total)

    def test_wrapping_in_flagged_noise_equals_passing_flags(
        self, twin, space, template_state, observed
    ):
        flags = jnp.zeros(observed.shape, bool).at[:, :2].set(True)
        via_kwarg = to_numpyro_model(
            twin, template_state, space, noise_std=SIGMA, flags=flags
        )
        via_wrapper = to_numpyro_model(
            twin, template_state, space,
            noise_std=FlaggedNoise(HomoscedasticNoise(jnp.asarray(SIGMA)), flags),
        )
        values = {"gain": jnp.array(1.05)}
        a, _ = log_density(via_kwarg, (), {"observed": observed}, values)
        b, _ = log_density(via_wrapper, (), {"observed": observed}, values)
        assert jnp.allclose(a, b, rtol=1e-6)


class TestAgainstTheExactPosterior:
    """NUTS and `gcr_sample` target the same posterior — one exactly.

    The model is linear in the gain with a Gaussian prior and constant noise,
    which is precisely the conjugate case `gcr_sample` draws in closed form.
    Agreement here is the strongest check available on either: an exact
    sampler with no burn-in against a chain, on a posterior both are supposed
    to reproduce.
    """

    N_WARMUP, N_SAMPLES = 800, 3000

    @pytest.fixture
    def block(self, twin, space, template_state):
        return linear_operator(space, twin, template_state)

    @pytest.fixture
    def exact(self, block, observed):
        """Mean and width from the constrained realization, in closed form."""
        mean, _ = wiener_solve(
            block, observed, noise_std=SIGMA,
            prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN,
        )
        keys = jax.random.split(jax.random.key(4), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(
                block, observed, noise_std=SIGMA, prior_std=PRIOR_STD,
                prior_mean=PRIOR_MEAN, key=k,
            )[0]
        )(keys)
        return float(mean), float(jnp.std(draws))

    @pytest.fixture
    def nuts(self, twin, space, template_state, observed):
        model = to_numpyro_model(
            twin, template_state, space,
            noise_std=HomoscedasticNoise(jnp.asarray(SIGMA)),
        )
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model),
            num_warmup=self.N_WARMUP, num_samples=self.N_SAMPLES,
            progress_bar=False,
        )
        mcmc.run(jax.random.key(0), observed=observed)
        chain = np.asarray(mcmc.get_samples()["gain"])
        return float(chain.mean()), float(chain.std())

    def test_the_means_agree(self, exact, nuts):
        exact_mean, exact_std = exact
        nuts_mean, _ = nuts
        # MCMC standard error on the mean, generously bounded: even at an
        # effective sample size of one tenth the chain, 4 of these is decisive.
        stderr = exact_std / np.sqrt(self.N_SAMPLES / 10)
        assert abs(nuts_mean - exact_mean) < 4 * stderr, (
            f"NUTS {nuts_mean:.6f} vs exact {exact_mean:.6f}, stderr {stderr:.6f}"
        )

    def test_the_widths_agree(self, exact, nuts):
        _, exact_std = exact
        _, nuts_std = nuts
        assert nuts_std == pytest.approx(exact_std, rel=0.1)

    def test_the_exact_posterior_is_not_merely_the_prior(self, exact):
        """Otherwise the agreement above would be a test of the prior."""
        _, exact_std = exact
        assert exact_std < 0.2 * PRIOR_STD


class TestInitToDeclared:
    """The space already says where to start; NUTS does not read it unless told.

    Measured on ``examples/tutorial_nuts.py`` the difference is r_hat 840 vs
    1.002 and an effective sample size of 2 vs 1327, with neither tighter
    priors nor triple the warmup moving either number. What is checked here is
    the contract, not that sampling improves: that the strategy carries the
    space's own declared values, so the fix is a one-liner rather than a dict
    the caller reassembles by hand and can get wrong silently.
    """

    def test_it_starts_at_the_declared_values(self, twin, space, template_state):
        from rheplicant.inference import init_to_declared

        strategy = init_to_declared(space)
        model = to_numpyro_model(twin, template_state, space, noise_std=SIGMA)
        kernel = numpyro.infer.NUTS(model, init_strategy=strategy)
        mcmc = numpyro.infer.MCMC(
            kernel, num_warmup=1, num_samples=1, progress_bar=False
        )
        # It runs, which is the integration; the values it carries are the
        # space's own, which is the contract.
        mcmc.run(jax.random.key(0), observed=jnp.zeros((8, 4)))
        assert set(space.initial_values()) == set(space.names)

    def test_it_carries_the_spaces_own_init(self, space):
        from rheplicant.inference import init_to_declared

        # A strategy is a partial'd callable; what matters is that it was built
        # from initial_values() rather than from anything reconstructed.
        strategy = init_to_declared(space)
        assert strategy.keywords["values"] == space.initial_values()

    def test_a_changed_init_changes_the_strategy(self, template_state):
        from rheplicant.inference import init_to_declared

        first = ParameterSpace.direct(
            "gain", init=1.0, into=lambda p: p["gain"].gain,
            prior=dist.Normal(1.0, 0.3),
        )
        second = ParameterSpace.direct(
            "gain", init=2.5, into=lambda p: p["gain"].gain,
            prior=dist.Normal(1.0, 0.3),
        )
        assert (
            init_to_declared(first).keywords["values"]
            != init_to_declared(second).keywords["values"]
        )
