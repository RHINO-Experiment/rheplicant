"""Amortized neural posterior estimation, held to an exactly-known posterior.

An approximate posterior has no internal notion of being wrong: a badly-fitted
`q` returns a smooth, confident, incorrect distribution and reports nothing
amiss. So the substantive test here is not that training runs — it is that on a
linear-Gaussian problem, where `gcr_sample` gives the posterior in closed form,
the network agrees with it.

The problem is deliberately one the estimator cannot fake: the posterior is
about five times narrower than the prior, so an estimator that merely
reproduced the prior it was trained from would fail by a factor of five.

Every constant below was measured rather than guessed, because both of NPE's
failure modes showed up while writing this file and they push in opposite
directions:

===============  =====  ===  ===============
``n_simulations``  steps  K   width / exact
===============  =====  ===  ===============
8192              1500   1   0.88
8192              4000   1   0.84
8192              4000   2   0.60
32768             1500   1   0.98
32768             1500   2   1.07
===============  =====  ===  ===============

*Too few simulations* leaves the width wrong because draws come from the prior,
so only a fraction ``posterior_width / prior_width`` of them land near any given
observation. *Too many steps on a small bank* over-fits, and over-fitting an NPE
makes it too NARROW — a confident, smooth, correctly-centred, wrong answer. The
held-out split in ``train_posterior`` exists to make the second visible, and it
is why the default returns the best validation step rather than the last.
"""

import jax
import jax.numpy as jnp
import rheplicant.inference.npe as rheplicant_npe
import pytest

numpyro = pytest.importorskip("numpyro", reason="numpyro not installed")
import numpyro.distributions as dist  # noqa: E402

from rheplicant import Coordinates, Environment, State  # noqa: E402
from rheplicant.core.errors import ParameterSpaceError, StateValidationError  # noqa: E402
from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.inference import (  # noqa: E402
    HomoscedasticNoise,
    NeuralPosterior,
    ParameterSpace,
    RadiometerNoise,
    gcr_sample,
    linear_operator,
    simulate_pairs,
    train_posterior,
    wiener_solve,
)
from rheplicant.radio import GainOperator, SkyOperator  # noqa: E402

SKY, TRUE_GAIN, SIGMA = 100.0, 1.1, 5.0
PRIOR_MEAN, PRIOR_STD = 1.0, 0.05
N_SIMULATIONS = 32768


# Module-scoped throughout: training the estimator is the expensive part of
# this file, and every test below asks about the SAME trained estimator. A
# function-scoped fixture retrains it five times for no additional coverage.
# The state is built here rather than taken from conftest so the scope can be
# widened without touching a fixture other tests share.


@pytest.fixture(scope="module")
def state():
    return State(
        coords=Coordinates(
            time=jnp.linspace(0.0, 7.0, 8), freq=jnp.linspace(60e6, 85e6, 4)
        ),
        env=Environment(temperature=jnp.array(280.0)),
        key=jax.random.key(0),
    )


def _twin_at(gain):
    return Pipeline(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(gain)),
        names=("sky", "gain"),
    )


@pytest.fixture(scope="module")
def twin():
    return _twin_at(1.0)


@pytest.fixture(scope="module")
def space():
    return ParameterSpace.direct(
        "gain", init=1.0, into=lambda p: p["gain"].gain,
        prior=dist.Normal(PRIOR_MEAN, PRIOR_STD), linear=True,
    )


@pytest.fixture(scope="module")
def noise():
    return HomoscedasticNoise(jnp.asarray(SIGMA))


@pytest.fixture(scope="module")
def observed(state):
    truth = _twin_at(TRUE_GAIN)(state).data
    return truth + SIGMA * jax.random.normal(jax.random.key(99), truth.shape)


@pytest.fixture(scope="module")
def bank(twin, state, space, noise):
    return simulate_pairs(
        twin, state, space,
        noise=noise, key=jax.random.key(0), n_simulations=N_SIMULATIONS,
    )


@pytest.fixture(scope="module")
def exact(twin, state, space, observed):
    """The posterior in closed form: mean from the Wiener solve, width from
    4000 exact constrained realizations."""
    block = linear_operator(space, twin, state)
    mean, _ = wiener_solve(
        block, observed, noise_std=SIGMA,
        prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN,
    )
    draws = jax.vmap(
        lambda k: gcr_sample(
            block, observed, noise_std=SIGMA, prior_std=PRIOR_STD,
            prior_mean=PRIOR_MEAN, key=k,
        )[0]
    )(jax.random.split(jax.random.key(4), 4000))
    return float(mean), float(jnp.std(draws))


@pytest.fixture(scope="module")
def trained(bank):
    thetas, data = bank
    # One component is EXACT for a Gaussian posterior; extra components mostly
    # buy capacity to memorize the bank (see the table above).
    q = NeuralPosterior.create(
        thetas, data, key=jax.random.key(1), n_components=1, width=64, depth=2
    )
    return train_posterior(
        q, thetas, data, key=jax.random.key(2),
        n_steps=2000, batch_size=512, learning_rate=2e-3,
    )


class TestSimulatePairs:
    def test_shapes_are_the_bank(self, bank, state, twin, space):
        thetas, data = bank
        assert thetas.shape == (N_SIMULATIONS, 1)
        assert data.shape[0] == N_SIMULATIONS
        assert data.shape[1:] == space.bind(twin, {"gain": jnp.array(1.0)})(
            state
        ).data.shape

    def test_theta_follows_the_prior(self, bank):
        thetas, _ = bank
        assert float(thetas.mean()) == pytest.approx(PRIOR_MEAN, abs=4e-3)
        assert float(thetas.std()) == pytest.approx(PRIOR_STD, rel=0.05)

    def test_the_scatter_is_the_noise_model(self, bank, twin, state, space):
        """Each simulated datum is prediction + sigma * normal, so the residual
        against its OWN theta has the noise model's width."""
        thetas, data = bank
        clean = jax.vmap(
            lambda t: space.bind(twin, {"gain": t[0]})(state).data
        )(thetas)
        assert float((data - clean).std()) == pytest.approx(SIGMA, rel=0.05)

    def test_a_multiplicative_noise_model_is_honoured(
        self, twin, state, space
    ):
        radiometer = RadiometerNoise(1e4, 1.0)
        thetas, data = simulate_pairs(
            twin, state, space,
            noise=radiometer, key=jax.random.key(1), n_simulations=2048,
        )
        clean = jax.vmap(
            lambda t: space.bind(twin, {"gain": t[0]})(state).data
        )(thetas)
        fractional = ((data - clean) / clean).std()
        assert float(fractional) == pytest.approx(radiometer.fractional, rel=0.06)

    def test_the_floor_is_a_weighting_remedy_and_never_reaches_the_generator(
        self, twin, state, space
    ):
        """The simulator draws with ``realise``, not with ``std``.

        The two are not the same law and this is the case that separates them.
        ``RadiometerNoise.std`` returns ``max(|d|, floor) * f`` -- the floor is
        a remedy for a reweighting iterate crossing zero -- while ``realise``
        draws ``d (1 + f w)`` and applies no floor at all, "because a generator
        has no iterate" (its own docstring). Simulating with ``std`` therefore
        inflates the training bank's scatter by ``floor / |d|`` the moment a
        floor is declared, and trains the network on data no instrument could
        record.

        ``test_a_multiplicative_noise_model_is_honoured`` above cannot see
        this: its predictions are all positive and its floor is zero, so the
        additive and multiplicative laws agree to the last bit there. This one
        chooses a floor a thousand times the prediction, where they disagree by
        that same factor.
        """
        f = RadiometerNoise(1e4, 1.0)
        clean_scale = float(
            jnp.abs(space.bind(twin, {"gain": jnp.asarray(TRUE_GAIN)})(state).data).mean()
        )
        floored = RadiometerNoise(1e4, 1.0, floor=1e3 * clean_scale)

        thetas, data = simulate_pairs(
            twin, state, space,
            noise=floored, key=jax.random.key(3), n_simulations=2048,
        )
        clean = jax.vmap(
            lambda t: space.bind(twin, {"gain": t[0]})(state).data
        )(thetas)
        fractional = float(((data - clean) / clean).std())

        # The generator ignores the floor, so the scatter is the SAME as with
        # no floor: f, not f * floor / |d| (which would be ~1000x larger).
        assert fractional == pytest.approx(f.fractional, rel=0.06), (
            f"fractional scatter {fractional:.4g}; the generator should be "
            f"unaffected by floor= and give {f.fractional:.4g}. A value near "
            f"{f.fractional * 1e3:.4g} means the bank was drawn with std() "
            "instead of realise()."
        )

    def test_a_prior_free_latent_is_refused(self, twin, state, noise):
        space = ParameterSpace.direct("gain", init=1.0, into=lambda p: p["gain"].gain)
        with pytest.raises(ParameterSpaceError, match="no prior"):
            simulate_pairs(
                twin, state, space,
                noise=noise, key=jax.random.key(0), n_simulations=8,
            )

    def test_zero_simulations_is_refused(self, twin, state, space, noise):
        with pytest.raises(StateValidationError, match="positive"):
            simulate_pairs(
                twin, state, space,
                noise=noise, key=jax.random.key(0), n_simulations=0,
            )


class TestNeuralPosteriorConstruction:
    def test_mismatched_bank_halves_are_refused(self, bank):
        thetas, data = bank
        with pytest.raises(StateValidationError, match="same pairs"):
            NeuralPosterior.create(thetas[:10], data[:5], key=jax.random.key(0))

    def test_a_one_dimensional_theta_stack_is_refused(self, bank):
        _, data = bank
        with pytest.raises(StateValidationError, match="n_params"):
            NeuralPosterior.create(jnp.zeros(data.shape[0]), data, key=jax.random.key(0))

    def test_it_is_a_normalized_density(self, bank, observed):
        """A mixture that does not integrate to one is not a posterior. Checked
        by quadrature over a range the trained density is confined to."""
        thetas, data = bank
        q = NeuralPosterior.create(thetas, data, key=jax.random.key(1))
        q, _ = train_posterior(
            q, thetas, data, key=jax.random.key(2), n_steps=600, batch_size=256
        )
        grid = jnp.linspace(PRIOR_MEAN - 8 * PRIOR_STD, PRIOR_MEAN + 8 * PRIOR_STD, 4001)
        density = jnp.exp(
            jax.vmap(lambda t: q.log_prob(jnp.array([t]), observed))(grid)
        )
        assert float(jnp.trapezoid(density, grid)) == pytest.approx(1.0, abs=0.02)


class TestAgainstTheExactPosterior:
    """The only test that can catch a confidently wrong estimator."""



    def test_training_reduces_the_loss(self, trained):
        _, history = trained
        assert float(history.train[-200:].mean()) < float(history.train[:200].mean())
        assert jnp.all(jnp.isfinite(history.train))

    def test_a_held_out_split_is_scored_and_selected_on(self, trained):
        _, history = trained
        assert history.validation.shape == history.train.shape
        assert jnp.all(jnp.isfinite(history.validation))
        best = int(history.best_step)
        assert 1 <= best <= history.validation.shape[0]
        # The returned estimator is the best validation step, so nothing later
        # in the run scored better on data it was not trained on.
        assert float(history.validation[best - 1]) == pytest.approx(
            float(history.validation.min()), rel=1e-6
        )

    def test_the_posterior_mean_matches(self, trained, observed, exact):
        q, _ = trained
        exact_mean, exact_std = exact
        draws = q.sample(observed, jax.random.key(3), 4000)
        assert float(draws.mean()) == pytest.approx(exact_mean, abs=0.35 * exact_std)

    def test_the_posterior_width_matches(self, trained, observed, exact):
        q, _ = trained
        _, exact_std = exact
        draws = q.sample(observed, jax.random.key(3), 4000)
        assert float(draws.std()) == pytest.approx(exact_std, rel=0.35)

    def test_the_answer_is_not_just_the_prior(self, exact):
        """Otherwise both tests above would pass on an estimator that learned
        nothing from the data at all."""
        _, exact_std = exact
        assert exact_std < 0.25 * PRIOR_STD

    def test_it_is_amortized(self, trained, twin, state, space):
        """The point of the exercise: a SECOND observation costs a forward pass,
        with no retraining and no chain."""
        q, _ = trained
        other_truth = 1.05
        fresh = Pipeline(
            SkyOperator(amplitude=jnp.array(SKY)),
            GainOperator(gain=jnp.array(other_truth)),
            names=("sky", "gain"),
        )(state).data
        fresh = fresh + SIGMA * jax.random.normal(jax.random.key(7), fresh.shape)

        block = linear_operator(space, twin, state)
        reference, _ = wiener_solve(
            block, fresh, noise_std=SIGMA,
            prior_std=PRIOR_STD, prior_mean=PRIOR_MEAN,
        )
        draws = q.sample(fresh, jax.random.key(8), 4000)
        assert float(draws.mean()) == pytest.approx(
            float(reference), abs=0.35 * PRIOR_STD
        )


class TestTheRestatedSignaturesStillMatchTheFarSide:
    """`create` and `train_posterior` restate a parameter list they delegate.

    **They have to.** The config layer derives its ``npe:`` grammar from these
    signatures — which keys exist, which are optional, what each defaults to,
    and that ``embed`` belongs to ``create`` and not to ``train``. A forwarding
    ``*args, **kwargs`` is a grammar of nothing: measured, it took seven tests
    in ``tests/config/test_config_section_npe.py::
    TestTheGrammarMatchesTheSignatures`` down at once.

    Restating re-creates the drift that forwarding avoided, so it is guarded
    here rather than hoped for. The far side is free to add a parameter, and
    nothing else in either suite would notice: the config grammar is checked
    against THIS signature, so a near side that fell behind would stay
    self-consistent and silently stop offering the new knob.
    """

    @pytest.mark.parametrize(
        "near, far, skip",
        [
            (
                rheplicant_npe.NeuralPosterior.create,
                "bayesmith.amortize:NeuralPosterior.create",
                (),
            ),
            (
                rheplicant_npe.train_posterior,
                "bayesmith.amortize:train_posterior",
                (),
            ),
        ],
        ids=["create", "train_posterior"],
    )
    def test_parameter_for_parameter_and_default_for_default(self, near, far, skip):
        import importlib
        import inspect

        module_name, _, attr = far.partition(":")
        target = importlib.import_module(module_name)
        for part in attr.split("."):
            target = getattr(target, part)

        ours = inspect.signature(near).parameters
        theirs = inspect.signature(target).parameters
        assert list(ours) == list(theirs), (
            "parameter names or order drifted from the far side; the config "
            "grammar is derived from ours, so it would silently stop matching"
        )
        for name in ours:
            if name in skip:
                continue
            assert ours[name].default == theirs[name].default, (name, ours[name].default, theirs[name].default)
            assert ours[name].kind == theirs[name].kind, (name, ours[name].kind, theirs[name].kind)

    def test_the_comparison_would_notice_an_added_parameter(self):
        """Anti-vacuity: the case above is an equality, so prove it can fail.

        A signature comparison that accidentally compared something with
        itself — the far side's against the far side's — would pass for ever.
        """
        import inspect

        ours = list(inspect.signature(rheplicant_npe.NeuralPosterior.create).parameters)
        assert "min_scale" in ours and "width" in ours, ours
        assert ours != list(
            inspect.signature(rheplicant_npe.train_posterior).parameters
        ), "the two signatures are indistinguishable, so the check above proves little"
