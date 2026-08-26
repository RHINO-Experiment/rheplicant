"""kind: nuts, end to end."""

import warnings

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.nuts import (
    _MCMC_KEYS,
    _NUTS_KEYS,
    NutsProduct,
    _run_nuts,
)
from rheplicant.config.sections.runs import run_document
from tests.config.exit_helpers import (
    FROZEN,
    ONE_REF,
    PRIOR_FREE,
    RADIOMETER,
    TWO_LATENTS,
    TWO_REFS,
    WIENER_MODEL,
)
from tests.config.posterior_helpers import (
    NEEDLE,
    nuts_built,
    nuts_document,
    nuts_product,
    nuts_spec,
)


def _drive(spec, built):
    """parse -> pre_execute -> execute: the registry's own triple, driven by
    hand for the same reason this file always drove the executor directly.
    Plan 4A Task 9 moved the grammar into the parser, so a SUCCESS path must
    come through the parse seam (the seed travels as its resolved integer).
    """
    from _rheplicant_bootstrap.variants import LayerRef
    from rheplicant.config.sections.exit_support import (
        handler_for,
        parse_run,
    )

    parsed = parse_run(spec, built, index=0,
                       layer=LayerRef(kind="base", name=None, prefix="",
                                      document={}, declared_runs=None))
    handler = handler_for(spec.kind)
    handler.pre_execute(parsed, built, {})
    return handler.execute(parsed, built, {})


def product(run=None, **document):
    return _drive(nuts_spec(**(run or {})), nuts_built(**document))


def needle(run=None):
    """:data:`NEEDLE` on :data:`WIENER_MODEL`, straight through the handler.

    The model is named rather than left to ``conjugate_document``'s default
    because every number pinned below was measured on this one -- NOT because
    the default breaks the document: measured, ``CONJUGATE_MODEL``'s extra
    fixed ``uniform_sky`` amplitude appears in the prediction and in the
    simulated data alike, and the declared start finds the line under it just
    as well (mean 7.99978e7 against 7.99955e7 here).
    """
    return _drive(nuts_spec(**(run or {})),
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
        # Against the TRUTH, at a tolerance the posterior width justifies --
        # not against this machine's chain. Measured 1.500021 here and
        # 1.4999907 on the x86_64 runner: both recover 1.5 to 1e-5, and the
        # `abs=1e-5` pin on the sampler's own mean failed on the second while
        # the recovery it exists to check was fine. The std is ~4.3e-4, so
        # five of those is the band a converged chain's mean lives in.
        mean = float(drawn.samples["g"].mean())
        std = float(drawn.samples["g"].std())
        assert std == pytest.approx(0.000429, rel=0.25), std
        assert abs(mean - 1.5) < 5 * std, (mean, std)

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
        # Against the truths, 1.2 and 12.0, for the reason the one-latent
        # test above records. Measured d = 1.19996 here, 1.1995487 there.
        assert float(drawn.samples["d"].mean()) == pytest.approx(1.2, abs=2e-3)
        assert float(drawn.samples["a"].mean()) == pytest.approx(12.0, abs=2e-2)


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
        this run's `where`, so it names the run and names runtime.seeds.
        Plan 4A Task 9 moved it into the parser."""
        from _rheplicant_bootstrap.variants import LayerRef
        from rheplicant.config.sections.exit_support import parse_run

        with pytest.raises(ConfigError) as caught:
            parse_run(nuts_spec(drop=("seed",)), nuts_built(), index=0,
                      layer=LayerRef(kind="base", name=None, prefix="",
                                     document={}, declared_runs=None))
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
        with pytest.warns(UserWarning, match="divergent"):
            drawn = needle({"init": "ref"})
        assert float(drawn.samples["c"].mean()) < 5.0e7
        assert drawn.divergences > 0

    def test_ref_puts_the_ref_on_the_kernel(self, monkeypatch):
        import numpyro

        captured = {}
        real = numpyro.infer.NUTS

        def spy(model, **kwargs):
            captured.update(kwargs)
            return real(model, **kwargs)

        monkeypatch.setattr(numpyro.infer, "NUTS", spy)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
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
        _drive(nuts_spec(init="ref"), nuts_built(inference=TWO_REFS))
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


class TestTheKnobs:
    def test_thinning_reaches_MCMC(self):
        """`num_samples` counts BEFORE thinning (measured): 200 declared and
        thinning 2 returns 100.  An executor that swept the key and dropped
        it would return 200."""
        drawn = product({"thinning": 2})
        assert drawn.n_draw == 100
        # And `n_draw` is the count RETURNED, not arithmetic on the declared
        # options.  Measured: `n_draw = num_chains * num_samples // thinning`
        # satisfies every count assertion in this module -- 200//2 here,
        # 2*200 below, 200 in Task 4's -- because on all three documents the
        # arithmetic and the stack agree.  Only the stack itself separates
        # them, and it is the property `predict` gates `keep > available` on.
        assert drawn.n_draw == drawn.samples["g"].shape[0]

    def test_num_chains_reaches_MCMC(self):
        """`get_samples()` flattens the chains, so two chains of 200 is 400
        draws under one leading axis -- which is the shape `predict` reads.
        `n_chain` is what says there were two."""
        drawn = product({"num_chains": 2, "chain_method": "sequential"})
        assert drawn.n_chain == 2
        assert drawn.n_draw == 400
        # The same cross-check as above, on the other knob: the count must
        # come off the returned stack, not off `run.options`.
        assert drawn.n_draw == drawn.samples["g"].shape[0]

    def test_n_draw_is_the_stack_and_not_arithmetic_on_the_options(self):
        """The count RETURNED, on the one document where the two differ.

        `n_draw = num_chains * num_samples // thinning` survives every other
        assertion in this module, INCLUDING the two
        `n_draw == samples[...].shape[0]` cross-checks above -- because on
        those documents the arithmetic and the stack agree, so comparing
        them compares a number with itself (measured: that mutant left all
        42 other tests green).

        **numpyro thins per chain, so the floor is taken BEFORE the
        multiply.**  Measured across five documents: 2 chains x 200 samples
        thinned 3 returns `2 * (200 // 3) == 132`, while multiplying first
        gives `(2 * 200) // 3 == 133`.  Any formula that reaches for the
        options is over by up to `num_chains - 1`, and `n_draw` is what 2C's
        `predict` gates `keep > available` on -- so an over-count is a run
        promising draws it has not got, which is a wrong refusal in Task 9
        rather than a wrong shape here.

        Documents where they AGREE (1 chain and any thinning; 2 chains and
        a thinning that divides the count) cannot show this at all, which is
        why the numbers below are 2/200/3 and not the module's usual pair.
        """
        drawn = product({"num_chains": 2, "thinning": 3,
                         "chain_method": "sequential"})
        assert drawn.n_draw == 132
        assert drawn.n_draw == drawn.samples["g"].shape[0]
        assert (2 * 200) // 3 == 133      # what the options alone would say

    def test_target_accept_prob_reaches_NUTS(self):
        """Forward coverage without a spy: a sloppy target DIVERGES.

        Measured on the one-latent document at four seeds: the package
        default (0.8) gives 0 divergences at every one, and 0.2 gives
        200/101/77/52.  An executor that swept the key and never forwarded
        it, or forwarded it to MCMC, would give 0 here.
        """
        assert product().divergences == 0
        with pytest.warns(UserWarning, match="divergent"):
            drawn = product({"target_accept_prob": 0.2})
        assert drawn.divergences > 0

    def test_target_accept_prob_is_NOT_an_MCMC_key(self):
        """Why `_MCMC_KEYS` excludes it, pinned against the package rather
        than asserted about the table.  `"x" not in _MCMC_KEYS` alone is an
        assertion about a tuple; these two `TypeError`s are the reason."""
        import numpyro

        assert "target_accept_prob" not in _MCMC_KEYS
        assert "target_accept_prob" in _NUTS_KEYS
        with pytest.raises(TypeError, match="target_accept_prob"):
            numpyro.infer.MCMC(object(), num_warmup=1, num_samples=1,
                               target_accept_prob=0.8)
        with pytest.raises(TypeError, match="init_strategy"):
            numpyro.infer.MCMC(object(), num_warmup=1, num_samples=1,
                               init_strategy=None)

    def test_no_default_is_restated(self, monkeypatch):
        """A silent document SENDS numpyro the parsed defaults -- its own.

        Plan 4A Task 9 changed WHAT this test can pin: the parser now
        normalizes the six optional knobs into the parsed view, so the
        executor always forwards them.  What must not change is the VALUES:
        measured against numpyro's own signatures, ``num_chains=1``,
        ``chain_method="parallel"``, ``thinning=1``, ``progress_bar=True``
        (and ``target_accept_prob=0.8`` on the kernel) -- the SCHEMA's
        ``sequential``/``false`` pair, the mutant this test was written to
        kill, must still never appear.

        The document is the SILENT one on purpose: :data:`NUTS` declares
        `progress_bar: false`, so it is not silent, and this test builds the
        run without it.
        """
        import numpyro

        captured = {}
        real = numpyro.infer.MCMC

        def spy(kernel, **kwargs):
            captured.update(kwargs)
            return real(kernel, **kwargs)

        monkeypatch.setattr(numpyro.infer, "MCMC", spy)
        drawn = product({"drop": ("progress_bar",)})
        assert captured == {"num_warmup": 200, "num_samples": 200,
                            "num_chains": 1, "chain_method": "parallel",
                            "thinning": 1, "progress_bar": True}, captured
        assert drawn.n_chain == 1
        assert drawn.n_draw == 200

    def test_every_declared_knob_arrives_on_MCMC(self, monkeypatch):
        """The other direction, on the same spy: declared keys DO arrive.

        Without this, `set(captured) == {"num_warmup", "num_samples"}` is
        satisfied by an executor that forwards nothing at all -- which is the
        vacuous half of the pair, and the reason `chain_method` and
        `progress_bar` had no forward coverage before.  Both are declared
        here with values that are NOT numpyro's defaults, so a `_MCMC_KEYS`
        missing either cannot produce this.
        """
        import numpyro

        captured = {}
        real = numpyro.infer.MCMC

        def spy(kernel, **kwargs):
            captured.update(kwargs)
            return real(kernel, **kwargs)

        monkeypatch.setattr(numpyro.infer, "MCMC", spy)
        product({"num_chains": 2, "chain_method": "sequential",
                 "thinning": 2, "progress_bar": False})
        assert captured["num_chains"] == 2
        assert captured["chain_method"] == "sequential"
        assert captured["thinning"] == 2
        assert captured["progress_bar"] is False


class TestTheKnobsAreCHECKED:
    """``_sweep`` checks key NAMES.  Nothing checked VALUES, and it showed.

    Measured on ``nuts_built()`` before this class existed -- every one of
    these reached the user as a raw package error naming no run and no key,
    and one of them did not raise at all:

    ==========================  ==================================================
    ``num_chains: 0``           ``IndexError: tuple index out of range``
    ``num_chains: -1``          ``IndexError: tuple index out of range``
    ``num_chains: 2.5``         ``TypeError: 'float' object cannot be interpreted``
    ``thinning: 1.5``           ``ValueError: thinning must be a positive integer``
    ``chain_method: banana``    ``ValueError: Only supporting the following...``
    ``target_accept_prob: hi``  ``TypeError: unsupported operand type(s) for -``
    ``target_accept_prob: 2.0`` **RUNS SILENTLY**
    ==========================  ==================================================

    Meanwhile ``num_samples: 2.5`` IS refused, by ``_number``, with "is a whole
    number" -- so the layer already owns this shape and applied it to two of the
    six keys.  That is 2C shape 4 (a hole closed on one route and open on its
    twin), shape 7 (raw jax/numpyro errors reaching the user), and shape 9 (the
    escape is through a CALL, so ``grep "raise "`` on this module stays clean).

    The precedent cuts both ways and it was weighed: ``_SAMPLE_PASSTHROUGH``
    already forwards ``warmup``/``max_iter`` unvalidated, so this is not a
    regression -- but ``conjugate.gcr`` DOES validate its own optional count
    (``_number(run, "n_draws", ..., kind=int, minimum=1)``, conjugate.py:325),
    and a NEW knob is where a layer decides which precedent it is following.
    """

    def test_num_chains_must_be_a_whole_number_at_least_one(self):
        for value in (0, -1, 2.5):
            with pytest.raises(ConfigError, match="num_chains"):
                product({"num_chains": value})

    def test_thinning_must_be_a_whole_number_at_least_one(self):
        for value in (0, 1.5):
            with pytest.raises(ConfigError, match="thinning"):
                product({"thinning": value})

    def test_target_accept_prob_must_be_a_number_above_zero(self):
        """``2.0`` is deliberately NOT refused.

        ``minimum=0.0`` is the check ``_number`` can make; an upper bound of 1
        is a statement about numpyro's parametrisation that this layer would be
        restating, and the measured behaviour of ``2.0`` is that it runs.  What
        is refused is the string, which today reaches the user as
        ``TypeError: unsupported operand type(s) for -: 'str' and
        'DynamicJaxprTracer'``.

        The last line is the one that DEFENDS that decision.  Without it,
        adding ``if value > 1.0: raise`` leaves every test in this module
        green (measured) and the docstring above is the only thing saying
        the absence was deliberate -- a correct decision shipped with no
        test, which is the shape this module keeps finding.  Measured:
        ``2.0`` runs to completion, 0 divergences, g mean 1.79 against a
        truth of 1.5 -- badly mixed, because an unreachable target drives
        the step size down, and NOT an error.
        """
        with pytest.raises(ConfigError, match="target_accept_prob"):
            product({"target_accept_prob": "hi"})
        with pytest.raises(ConfigError, match="target_accept_prob"):
            product({"target_accept_prob": -0.5})
        product({"target_accept_prob": 2.0})      # no raise, by decision

    def test_chain_method_must_be_one_of_the_three_numpyro_takes(self):
        """The three words are numpyro's, not this layer's invention, and the
        refusal NAMES them -- ``banana`` today gives
        ``ValueError: Only supporting the following methods...`` with no run
        name and no document key."""
        with pytest.raises(ConfigError, match="chain_method") as caught:
            product({"chain_method": "banana"})
        message = str(caught.value)
        assert "parallel" in message and "sequential" in message
        assert "vectorized" in message
        assert message.startswith("runs['chain']:")
        for legal in ("parallel", "sequential", "vectorized"):
            product({"chain_method": legal})     # no raise

    def test_a_null_knob_is_refused_by_name_on_every_one_of_them(self):
        """The gap between "declared" and "present", which cost a real bug.

        ``_passthrough`` forwards a key when it is PRESENT; a guard that
        reads ``run.options.get(key)`` and tests ``is not None`` calls the
        same key undeclared.  The two disagree on exactly one input, and
        ``chain_method: null`` fell through the gap into ``MCMC(...)`` --
        measured on the shipped executor, directly and through a real
        document, as ``ValueError: Only supporting the following methods to
        draw chains: "sequential", "parallel", or "vectorized"``, which is
        verbatim the error the test above says this layer replaces.

        Every one of the five is asserted rather than the one that broke:
        the other four refused ``null`` correctly at the time, so a test of
        ``chain_method`` alone would not have noticed that this is a SHAPE
        -- a hole closed on one route and left open on its twin, arriving in
        the commit that closed it for the rest.  ``null`` is not
        hypothetical here; this suite already declares one deliberately.
        """
        for key in ("chain_method", "num_chains", "thinning",
                    "target_accept_prob", "progress_bar", "init"):
            with pytest.raises(ConfigError, match=key) as caught:
                product({key: None})
            assert str(caught.value).startswith("runs['chain']:")

    def test_progress_bar_is_true_or_false_and_not_a_truthy_string(self):
        """The fifth knob of this commit, and the one `_number` cannot check.

        Measured on the shipped executor: ``progress_bar: "false"`` runs and
        PRINTS THE BAR -- every non-empty string is truthy, so the document
        says false, the run shows a bar, and nothing says otherwise.  Left
        unchecked it would be four knobs guarded and one not, which is the
        asymmetry the test above is about.
        """
        with pytest.raises(ConfigError, match="progress_bar") as caught:
            product({"progress_bar": "false"})
        assert "true or false" in str(caught.value)


class TestTheDiagnostics:
    def test_r_hat_and_n_eff_land_per_latent(self):
        """Per LATENT, and only the latents -- and each row is ITS latent's.

        `numpyro.diagnostics.summary` over the whole `get_samples(
        group_by_chain=True)` would summarise the deterministic
        "prediction" site too -- 200 x 16 x 8 of it -- and hand back a
        `diagnostics` mapping with a key no latent owns.

        The two PINS are what make the pairing load-bearing.  Measured:
        zipping each latent's name against the OTHER latent's row --
        `zip(list(table)[::-1], table.values())` -- survives a key-set check
        plus `0.9 < r_hat < 1.1` and `n_eff > 10`, because both latents
        satisfy both bounds.  That is 2C shape 5 exactly, and the sibling
        `test_two_latents_come_back_under_their_own_names` avoids it only
        because d ~ 1.2 and a ~ 12.0 are far apart.  So one row is pinned to
        the value MEASURED FOR THAT LATENT.

        The two floats below were measured HERE, on THIS document, at Step
        6.4, and reproduced identically on a second run -- not carried
        forward from the plan, which pinned neither, and not from the
        one-latent document, whose 0.99527 / 80.2 belong to a different
        space.  Measured: `d` 0.9972864 / 57.312, `a` 0.9963239 / 100.081.

        **It is `n_eff` that makes the pairing load-bearing, not `r_hat`.**
        The two `r_hat`s agree to 0.1 %, which is INSIDE the 1 % tolerance
        below, so a swapped pairing would survive that line alone; the two
        `n_eff`s differ by 75 %, so it dies on the next one.  Said out loud
        because a pin that cannot discriminate reads exactly like one that
        can, which is the shape this whole class is written against.
        """
        drawn = product(inference=TWO_LATENTS)
        assert set(drawn.diagnostics) == {"d", "a"}
        for row in drawn.diagnostics.values():
            assert set(row) == {"r_hat", "n_eff"}
            # `float`, not a numpy scalar: `summary` returns numpy, and
            # dropping the `float(...)` calls leaves every numeric assertion
            # in this module green (measured), so `NutsProduct`'s documented
            # `{"r_hat": float, "n_eff": float}` is checked only here.
            assert isinstance(row["r_hat"], float)
            assert isinstance(row["n_eff"], float)
            assert 0.9 < row["r_hat"] < 1.1
            assert row["n_eff"] > 10
        # The `r_hat` pin that stood here is GONE, on the strength of this
        # test's own argument: the two r_hats agree to 0.1 %, inside any
        # tolerance that would admit them both, so it never discriminated a
        # swapped pairing -- and it was platform-fragile as well, measured
        # 0.99729 here against 1.0113 on the x86_64 runner. A pin that cannot
        # discriminate reads exactly like one that can, which is the shape
        # this class is written against; keeping it for another platform to
        # break would have been keeping it for its appearance.
        #
        # `n_eff` is the one that does the work -- the two differ by 75 % --
        # so it stays.
        assert drawn.diagnostics["d"]["n_eff"] == pytest.approx(
            57.31, rel=5e-2)

    def test_the_divergence_count_is_real_and_not_always_zero(self):
        """A count that is 0 on every document a test ever shows it is a
        count nothing has measured (2C shape 3).  Both sides, and the
        unhealthy side is Task 5's needle, which really does diverge.

        The healthy side asserts SILENCE as well as zero, because the
        decision is *conditional* loudness and `pytest.warns` cannot notice
        a warning that always fires: measured, `if divergences:` weakened to
        `if divergences >= 0:` makes every run shout "recorded 0 divergent
        transition(s)" and leaves this module green, there being no
        `filterwarnings = ["error"]` in pyproject.toml.

        `int` is pinned for the same reason `float` is pinned on the
        diagnostics rows above: `diverging.sum()` is a jnp scalar that
        satisfies `== 0`, `> 0`, `== 100` and every f-string alike, so
        `NutsProduct`'s `divergences: int` is a claim only this line checks.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            drawn = product()
        assert drawn.divergences == 0
        assert isinstance(drawn.divergences, int)
        assert not [w for w in caught if "divergent" in str(w.message)], (
            "a healthy run must be silent")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert needle({"init": "ref"}).divergences > 0

    def test_a_diverging_run_says_so_out_loud(self):
        """The recorded decision: warn, do not refuse.  The count is on the
        product either way, and a warning cannot make a legal document
        unrunnable on a host where it converges."""
        with pytest.warns(UserWarning, match="divergent transition"):
            drawn = needle({"init": "ref"})
        assert drawn.divergences > 0

    def test_the_warning_counts_out_of_what_it_actually_counted(self):
        """The DENOMINATOR, which nothing else in this module reads.

        `divergences` is `diverging.sum()`, so `out of N` has to be
        `diverging.size` or the ratio is a lie.  The plan's own arithmetic
        was `num_chains * num_samples`, and thinning is what separates the
        two: measured on this run, `num_samples: 200` with `thinning: 2`
        records `diverging.shape == (100,)` and every one of the 100
        diverges, so the honest sentence is "100 out of 100" and the plan's
        is "100 out of 200" -- a run reported half as bad as it was.

        Written because the correction shipped undefended: reverting the
        denominator to the plan's form left all 39 other tests in this
        module green, every warning assertion matching only "divergent".
        A fix nothing pins is a fix in the state that let the defect exist.
        """
        with pytest.warns(UserWarning, match=r"out of 100\."):
            drawn = product({"thinning": 2, "target_accept_prob": 0.2})
        # `out of 100` above IS the claim -- the denominator is
        # `diverging.size` and not `num_chains * num_samples`. How many of
        # those 100 actually diverged is the sampler's business and moves with
        # the platform: all 100 here, 78 on the x86_64 runner. What must hold
        # is that some did, so the warning fired at all.
        assert 0 < drawn.divergences <= 100, drawn.divergences
        assert drawn.n_draw == 100


class TestTheParserInjectedDefaultsAreThePackagesOwn:
    """Plan 4A Task 9: the six injected defaults equal omission, in draws."""

    def test_explicit_defaults_draw_the_same_chain(self):
        implicit = product()
        explicit = product({"init": "declared", "num_chains": 1,
                            "chain_method": "parallel", "thinning": 1,
                            "progress_bar": True, "target_accept_prob": 0.8})
        assert set(implicit.samples) == set(explicit.samples)
        import numpy as np

        for name, stack in implicit.samples.items():
            assert np.array_equal(np.asarray(stack),
                                  np.asarray(explicit.samples[name])), name
