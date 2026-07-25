"""Tests for the NumPyro bridge: priors, model construction, MCMC recovery."""

import jax
import jax.numpy as jnp
import pytest

numpyro = pytest.importorskip("numpyro", reason="numpyro not installed")
import numpyro.distributions as dist  # noqa: E402
from numpyro.handlers import seed, trace  # noqa: E402

from replicant.core.errors import StateValidationError  # noqa: E402
from replicant.inference import (  # noqa: E402
    predict_from_samples,
    prior_template,
    set_prior,
    to_numpyro_model,
)
from replicant.radio import GainOperator, SkyOperator, assemble  # noqa: E402

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
def priors(twin):
    priors = prior_template(twin)
    return set_prior(priors, lambda p: p["gain"].gain, dist.Normal(1.0, 0.3))


@pytest.fixture
def observed(template_state):
    truth = assemble(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(TRUE_GAIN)),
    )(template_state).data
    noise = SIGMA * jax.random.normal(jax.random.key(99), truth.shape)
    return truth + noise


class TestPriorConstruction:
    def test_template_and_set_prior(self, twin, priors):
        # exactly one distribution attached, structure preserved
        flat = jax.tree_util.tree_flatten_with_path(
            priors, is_leaf=lambda x: isinstance(x, dist.Distribution)
        )[0]
        dists = [d for _, d in flat if isinstance(d, dist.Distribution)]
        assert len(dists) == 1

    def test_no_priors_rejected(self, twin, template_state):
        with pytest.raises(StateValidationError, match="no distributions"):
            to_numpyro_model(twin, template_state, prior_template(twin), noise_std=1.0)

    def test_shape_mismatch_rejected(self, twin, template_state):
        bad = set_prior(
            prior_template(twin), lambda p: p["gain"].gain,
            dist.Normal(jnp.zeros(3), 1.0),  # (3,) vs scalar leaf
        )
        with pytest.raises(StateValidationError, match="shape"):
            to_numpyro_model(twin, template_state, bad, noise_std=1.0)


class TestModel:
    def test_prior_predictive_trace(self, twin, priors, template_state):
        model = to_numpyro_model(twin, template_state, priors, noise_std=SIGMA)
        tr = trace(seed(model, jax.random.key(0))).get_trace()
        assert "gain.gain" in tr  # site named from the tree path
        assert tr["prediction"]["value"].shape == (8, 4)
        assert tr["obs"]["value"].shape == (8, 4)

    def test_sampled_noise_std(self, twin, priors, template_state):
        model = to_numpyro_model(
            twin, template_state, priors, noise_std=dist.HalfNormal(1.0)
        )
        tr = trace(seed(model, jax.random.key(0))).get_trace()
        assert "noise_std" in tr

    def test_flags_masked_likelihood(self, twin, priors, template_state, observed):
        """Corrupting a FLAGGED sample must not change the masked log density."""
        from numpyro.infer.util import log_density

        flags = jnp.zeros(observed.shape, bool).at[0, 0].set(True)
        masked = to_numpyro_model(
            twin, template_state, priors, noise_std=SIGMA, flags=flags
        )
        unmasked = to_numpyro_model(twin, template_state, priors, noise_std=SIGMA)
        corrupted = observed.at[0, 0].set(1e6)
        params = {"gain.gain": jnp.array(TRUE_GAIN)}

        def ld(model, obs):
            return float(log_density(model, (), {"observed": obs}, params)[0])

        assert ld(masked, observed) == pytest.approx(ld(masked, corrupted))
        assert ld(unmasked, observed) != pytest.approx(ld(unmasked, corrupted))


class TestMCMCRecovery:
    def test_nuts_recovers_gain(self, twin, priors, template_state, observed):
        model = to_numpyro_model(twin, template_state, priors, noise_std=SIGMA)
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model), num_warmup=200, num_samples=200,
            progress_bar=False,
        )
        mcmc.run(jax.random.key(0), observed=observed)
        samples = mcmc.get_samples()
        posterior_mean = float(samples["gain.gain"].mean())
        # analytic posterior std ~ sigma / (sky * sqrt(n)) ~ 0.5/(100*sqrt(32)) ~ 1e-3
        assert abs(posterior_mean - TRUE_GAIN) < 0.01
        assert float(samples["gain.gain"].std()) < 0.01

    def test_posterior_predictive(self, twin, priors, template_state, observed):
        model = to_numpyro_model(twin, template_state, priors, noise_std=SIGMA)
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model), num_warmup=100, num_samples=50,
            progress_bar=False,
        )
        mcmc.run(jax.random.key(1), observed=observed)
        preds = predict_from_samples(twin, template_state, priors, mcmc.get_samples())
        assert preds.shape == (50, 8, 4)
        # predictions are gain*sky: mean close to the observed mean signal
        assert abs(float(preds.mean()) - SKY * TRUE_GAIN) < 1.0

    def test_predict_missing_site_rejected(self, twin, priors, template_state):
        with pytest.raises(StateValidationError, match="missing site"):
            predict_from_samples(twin, template_state, priors, {"wrong": jnp.zeros(3)})
