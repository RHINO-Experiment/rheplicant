"""Tests for the NumPyro bridge: sites, model construction, MCMC recovery.

Sites are named by their latents, so a re-parameterized model is sampled in
the coordinates it was declared in — `log_gain`, not the tree path of whatever
leaf that logarithm eventually lands in.
"""

import jax
import jax.numpy as jnp
import pytest

numpyro = pytest.importorskip("numpyro", reason="numpyro not installed")
import numpyro.distributions as dist  # noqa: E402
from numpyro.handlers import seed, trace  # noqa: E402

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
        positional-prior scheme could not express at all."""
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
            numpyro.infer.NUTS(model), num_warmup=300, num_samples=300,
            progress_bar=False,
        )
        mcmc.run(jax.random.key(0), observed=data)
        recovered = float(jnp.exp(mcmc.get_samples()["log_gain"]).mean())
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

    def test_predict_missing_site_rejected(self, twin, space, template_state):
        with pytest.raises(StateValidationError, match="missing site"):
            predict_from_samples(twin, template_state, space, {"wrong": jnp.zeros(3)})
