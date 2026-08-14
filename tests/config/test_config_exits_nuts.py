"""kind: nuts, end to end."""

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.nuts import NutsProduct, _run_nuts
from rheplicant.config.sections.runs import run_document
from tests.config.exit_helpers import (
    FROZEN,
    NEEDLE,
    ONE_REF,
    PRIOR_FREE,
    RADIOMETER,
    TWO_LATENTS,
    TWO_REFS,
    WIENER_MODEL,
    nuts_built,
    nuts_document,
    nuts_product,
    nuts_spec,
)


def product(run=None, **document):
    return _run_nuts(nuts_spec(**(run or {})), nuts_built(**document))


def needle(run=None):
    """:data:`NEEDLE` on :data:`WIENER_MODEL`, straight through the executor.

    The model is named rather than left to ``conjugate_document``'s default
    because every number pinned below was measured on this one -- NOT because
    the default breaks the document: measured, ``CONJUGATE_MODEL``'s extra
    fixed ``uniform_sky`` amplitude appears in the prediction and in the
    simulated data alike, and the declared start finds the line under it just
    as well (mean 7.99978e7 against 7.99955e7 here).
    """
    return _run_nuts(nuts_spec(**(run or {})),
                     nuts_built(model=WIENER_MODEL, inference=NEEDLE))


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


class TestTheStartMoves:
    """``init:`` is tested by what the chain DID, not by what parsed.

    A test that ``init: ref`` was accepted proves nothing.  These legs
    together prove the start moved AND that it moved to the ref: the
    posterior contrast shows a different chain, and the spy shows which value
    reached the kernel.  Either alone is satisfiable by the wrong
    implementation -- ``init_to_uniform`` under ``init: ref`` passes the first
    (measured on this document, with the kernel's ``init_strategy`` stripped:
    mean 3.01e7, the same side of every threshold below as the ref's 3.06e7),
    and a kernel built and then thrown away passes the second.
    """

    def test_declared_finds_the_line(self):
        drawn = needle({"init": "declared"})
        assert float(drawn.samples["c"].mean()) == pytest.approx(8.0e7,
                                                                 rel=1e-3)

    def test_declared_is_the_default(self):
        """Silence and ``init: declared`` are the same chain, bit for bit.

        The default is the layer's own choice and not a restatement of
        numpyro's -- numpyro's is ``init_to_uniform`` -- so the branch a
        silent document takes is worth pinning as an identity rather than as
        another approximate mean, which ``init_to_uniform`` would also pass.
        """
        assert float(needle().samples["c"].mean()) == float(
            needle({"init": "declared"}).samples["c"].mean())

    def test_ref_starts_somewhere_else_and_the_chain_shows_it(self):
        drawn = needle({"init": "ref"})
        assert float(drawn.samples["c"].mean()) < 5.0e7

    def test_ref_puts_the_ref_on_the_kernel(self, monkeypatch):
        import numpyro

        captured = {}
        real = numpyro.infer.NUTS

        def spy(model, **kwargs):
            captured.update(kwargs)
            return real(model, **kwargs)

        monkeypatch.setattr(numpyro.infer, "NUTS", spy)
        needle({"init": "ref"})
        values = captured["init_strategy"].keywords["values"]
        assert float(values["c"]) == pytest.approx(6.0e7)

    def test_declared_puts_the_init_on_the_kernel(self, monkeypatch):
        """The twin of the leg above.

        ``init_to_declared(space)`` is ``init_to_value(values=
        space.initial_values())``, so both branches produce the same KIND of
        object and only the numbers differ -- which is why both are asserted
        rather than one being taken on trust.
        """
        import numpyro

        captured = {}
        real = numpyro.infer.NUTS

        def spy(model, **kwargs):
            captured.update(kwargs)
            return real(model, **kwargs)

        monkeypatch.setattr(numpyro.infer, "NUTS", spy)
        needle({"init": "declared"})
        values = captured["init_strategy"].keywords["values"]
        assert float(values["c"]) == pytest.approx(7.8e7)

    def test_each_latent_gets_its_own_ref(self, monkeypatch):
        """The PAIRING, which a one-latent document cannot show.

        On :data:`NEEDLE` the pairing is an identity, so a strategy built by
        zipping ``space.names`` against a separately-ordered list of values
        is right by accident: measured, reversing the values against the
        names leaves the whole config suite green, and sends this document's
        ``{'d': 0.25, 'a': 40.0}`` to the kernel as ``{'d': 40.0, 'a':
        0.25}`` -- finite, correctly shaped, wrong, which is the failure this
        plan is written against.

        The whole mapping is asserted rather than one key, so a strategy that
        carried an extra latent (or dropped one) is caught here too.
        """
        import numpyro

        captured = {}
        real = numpyro.infer.NUTS

        def spy(model, **kwargs):
            captured.update(kwargs)
            return real(model, **kwargs)

        monkeypatch.setattr(numpyro.infer, "NUTS", spy)
        _run_nuts(nuts_spec(init="ref"), nuts_built(inference=TWO_REFS))
        values = captured["init_strategy"].keywords["values"]
        assert {name: float(value) for name, value in values.items()} == {
            "d": 0.25, "a": 40.0}

    def test_ref_without_a_ref_is_refused_by_name(self):
        """ONE_LATENT declares no ref:, and silence is not a fallback."""
        with pytest.raises(ConfigError) as caught:
            _run_nuts(nuts_spec(init="ref"), nuts_built())
        message = str(caught.value)
        assert message.startswith("runs['chain']:")
        assert "['g']" in message
        assert "init: declared" in message

    def test_a_ref_on_only_SOME_latents_is_refused_by_name(self):
        """The mixed document the refusal's own sentence is about.

        ``ONE_LATENT`` above has ZERO refs, so it cannot tell "every latent
        needs one" from "at least one does": measured, a check that refuses
        only when NO latent carries a ref survives the whole config suite,
        and this document then dies inside numpyro with a bare ``KeyError:
        'a'`` -- naming no run, no key: and no way out.

        The latent WITH a ref must not be named: the refusal is about the
        one that has none.
        """
        with pytest.raises(ConfigError) as caught:
            _run_nuts(nuts_spec(init="ref"), nuts_built(inference=ONE_REF))
        message = str(caught.value)
        assert message.startswith("runs['chain']:")
        assert "['a']" in message
        assert "init: declared" in message

    def test_an_unknown_init_is_refused(self):
        """And the refusal names the run, and BOTH ways out.

        Measured, both unpinned before this: a ``where`` built from
        ``run.kind`` rather than ``run.name`` passed (and on a document with
        two ``kind: nuts`` runs it names neither), and so did an enumeration
        with ``ref`` hidden from it -- a refusal that offers a user half the
        grammar.
        """
        with pytest.raises(ConfigError) as caught:
            _run_nuts(nuts_spec(init="uniform"), nuts_built())
        message = str(caught.value)
        assert message.startswith("runs['chain']:")
        assert "init: is one of ['declared', 'ref']" in message


class TestTheKindIsReachableFromADocument:
    """Declared, registered and parseable -- three places to half-ship.

    Every assertion in this module until now called ``_run_nuts`` directly,
    which passes whether or not ``parse_runs`` accepts the kind.
    """

    def test_run_document_runs_it(self):
        drawn = nuts_product()
        assert float(drawn.samples["g"].mean()) == pytest.approx(1.5,
                                                                 abs=1e-3)

    def test_the_run_is_named_and_not_the_kind(self):
        results = run_document(nuts_document())
        assert set(results) == {"chain"}
        assert results["chain"].kind == "nuts"

    def test_a_refusal_can_be_expected(self):
        results = run_document(nuts_document({"expect": "refuse",
                                              "init": "uniform"}))
        assert isinstance(results["chain"].error, ConfigError)
