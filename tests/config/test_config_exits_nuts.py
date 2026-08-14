"""kind: nuts, end to end."""

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.nuts import NutsProduct, _run_nuts
from tests.config.exit_helpers import (
    FROZEN,
    PRIOR_FREE,
    RADIOMETER,
    TWO_LATENTS,
    nuts_built,
    nuts_spec,
)


def product(run=None, **document):
    return _run_nuts(nuts_spec(**(run or {})), nuts_built(**document))


class TestTheChainComesBack:
    def test_the_one_latent_document_recovers_its_truth(self):
        """The one number this exit pins.

        Measured on conjugate_document()'s default (CONJUGATE_MODEL +
        ONE_LATENT) at runtime.seeds.chain = 3: mean 1.500021, std 0.000429
        against a truth of 1.5.  Everything else in this module asserts a
        shape, a key, a refusal or a route.
        """
        drawn = product()
        assert isinstance(drawn, NutsProduct)
        assert float(drawn.samples["g"].mean()) == pytest.approx(1.500021,
                                                                 abs=1e-5)
        assert float(drawn.samples["g"].std()) == pytest.approx(0.000429,
                                                                abs=1e-5)

    def test_the_product_carries_the_space_and_NOT_the_prediction(self):
        """The memory trap, asserted as an ABSENCE.

        `get_samples()` also returns the deterministic "prediction" site,
        measured at (200, 16, 8) against g's (200,).  Asserting that "g" is
        present passes just as well when the whole TOD is present too (2C
        shape 5), so this asserts the key SET and names the site.
        """
        drawn = product()
        assert "prediction" not in drawn.samples
        assert set(drawn.samples) == {"g"}

    def test_the_shapes_are_the_predict_contract(self):
        """`n_draw` is the count RETURNED, not the count asked for.

        2C's shipped predict reads `product.n_draw` (diagnostics.py:748) and
        gates its `keep > available` refusal on it (:753), so an `n_draw`
        that names some other number of the chain's is a wrong refusal in
        Task 9 rather than a wrong shape here.

        `num_warmup: 100` against `num_samples: 200` is what makes the
        assertion able to discriminate: under NUTS's own 200/200 the warmup,
        the requested count and the returned count are the same integer, and
        `n_draw = mcmc.num_warmup` passed (measured).  It stays green under
        `n_draw = counts["num_samples"]`, which is unkillable here -- with
        one chain and no thinning, requested and returned ARE equal by
        construction, and only Task 6's `num_chains`/`thinning` can separate
        them.  Recorded for Task 6 alongside `n_chain`.
        """
        drawn = product({"num_warmup": 100})
        assert drawn.n_draw == 200
        assert drawn.n_draw == drawn.samples["g"].shape[0]
        assert drawn.n_chain == 1
        assert drawn.samples["g"].shape == (200,)

    def test_two_latents_come_back_under_their_own_names(self):
        drawn = product(inference=TWO_LATENTS)
        assert set(drawn.samples) == {"d", "a"}
        assert float(drawn.samples["d"].mean()) == pytest.approx(1.19996,
                                                                 abs=1e-4)
        assert float(drawn.samples["a"].mean()) == pytest.approx(11.99974,
                                                                 abs=1e-3)


class TestTheNoiseModelGoesInWhole:
    def test_a_prediction_dependent_sigma_runs(self):
        """`nuts` uses `_noise`, never `_decided_sigma`.

        `inference.noise.kind: radiometer` makes sigma a function of the
        prediction, which every conjugate exit refuses by name (check A27,
        exit_support.py:225).  An executor that reached for `_decided_sigma`
        would raise that refusal here; this run must produce a posterior
        instead.  Measured: g mean 1.500005.
        """
        drawn = product(noise=RADIOMETER)
        assert float(drawn.samples["g"].mean()) == pytest.approx(1.500005,
                                                                 abs=1e-4)

    def test_a_frozen_sigma_array_runs_too(self):
        """The other half of `decided_noise`'s two return shapes.

        `radiometer_frozen` hands `_noise` an ARRAY rather than a NoiseModel,
        and `to_numpyro_model` takes either.  Testing only the model half
        would leave the array half of the same accessor untested.
        """
        drawn = product(noise=FROZEN)
        assert float(drawn.samples["g"].mean()) == pytest.approx(1.5,
                                                                 abs=1e-4)


class TestTheRequiredKeys:
    @pytest.mark.parametrize("key", ["num_warmup", "num_samples"])
    def test_each_count_is_required(self, key):
        """Both legs, not one.

        numpyro declares both keyword-only with no default, so the layer has
        no package default to stand aside for.  Parametrized because a test
        of `num_warmup` alone leaves the `num_samples` leg of the same loop
        untested (2C shape 7).
        """
        with pytest.raises(ConfigError) as caught:
            _run_nuts(nuts_spec(drop=(key,)), nuts_built())
        message = str(caught.value)
        assert message.startswith("runs['chain']:")
        assert f"{key}:" in message
        assert "keyword-only with NO default" in message

    def test_the_seed_is_required(self):
        """A29's fourth member.  The refusal is draws.py's own, worn under
        this run's `where`, so it names the run and names runtime.seeds."""
        with pytest.raises(ConfigError) as caught:
            _run_nuts(nuts_spec(drop=("seed",)), nuts_built())
        assert str(caught.value).startswith("runs['chain']:")
        assert "runtime.seeds" in str(caught.value)

    def test_a_fractional_count_is_refused(self):
        with pytest.raises(ConfigError, match="whole number"):
            _run_nuts(nuts_spec(num_samples=2.5), nuts_built())

    @pytest.mark.parametrize("count", [0, -1])
    def test_a_count_below_one_is_refused(self, count):
        """The `minimum=1` floor, which nothing else in this module reads.

        What it stands in for, measured by driving numpyro directly:
        `num_samples: 0` arrives as `IndexError: index is out of bounds for
        axis 0 with size 0`, and `num_samples: -1` as a bare `AssertionError`
        with NO message at all, naming no run and no key.  Converting exactly
        that into a refusal that names both is what this layer is for, and
        dropping `minimum=1` left all twelve of the other tests green.
        """
        with pytest.raises(ConfigError, match="must be >= 1"):
            _run_nuts(nuts_spec(num_samples=count), nuts_built())

    def test_an_unknown_key_is_swept(self):
        """`step_size` is a real NUTS parameter this layer does not offer.

        The sweep must refuse it rather than let it travel: `_NUTS_KEYS` is
        the whole grammar, and a key that is legal on the package and absent
        from the table is exactly the one a user will try.
        """
        with pytest.raises(ConfigError, match="does not take"):
            _run_nuts(nuts_spec(step_size=0.1), nuts_built())

    def test_a_prior_free_space_is_refused(self):
        with pytest.raises(ConfigError, match="draws a POSTERIOR"):
            _run_nuts(nuts_spec(), nuts_built(parameters=PRIOR_FREE))
