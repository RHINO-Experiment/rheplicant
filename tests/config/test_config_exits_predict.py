"""kind: predict -- both reuse routes, end to end from a document."""

import jax.numpy as jnp
import numpy as np
import pytest

from _rheplicant_bootstrap.variants import LayerRef
from rheplicant.config import ConfigError
from rheplicant.config.sections.runs import run_document
from tests.config.exit_helpers import conjugate_document
from tests.config.posterior_helpers import NUTS, npe_document
from tests.config.test_config_document import synthetic_document

FORWARD = {"name": "fwd", "kind": "forward"}
FISHER = {"name": "cov", "kind": "fisher"}
SAMPLE = {"name": "chain", "kind": "plan.sample", "blocks": [{"names": ["g"]}],
          "seed": {"from": "runtime.seeds.sample"}, "n_sweeps": 10,
          "warmup": 4, "check_identifiability": False}
ESTIMATE = {"name": "point", "kind": "plan.estimate",
            "blocks": [{"names": ["g"]}], "check_identifiability": False}
OPTIMIZE = {"name": "fit", "kind": "optimize", "optimizer": "gradient",
            "learning_rate": 0.1, "n_steps": 3}
#: The near miss that COLLIDES.  `_gaussian_width` (conjugate.py:195) returns
#: {"fisher": ..., "covariance": ...} and `width: fisher` merges it into the
#: wiener product (:392), so this run's product carries a "covariance" key
#: that is NOT a kind: fisher product -- its covariance is over the CONJUGATE
#: BLOCK's latents.  It is the leg that makes the kind dispatch load-bearing.
WIENER_FISHER = {"name": "w", "kind": "conjugate.wiener", "names": ["g"],
                 "width": "fisher"}
FROM_COV = {"name": "p", "kind": "predict", "reuse": "cov"}
FROM_CHAIN = {"name": "p", "kind": "predict", "reuse": "chain"}

#: The compatibility base layer the parse seam is driven against directly.
_BASE_LAYER = LayerRef(kind="base", name=None, prefix="", document={},
                       declared_runs=None)

# The repair moves the absorption centre from 75 MHz to 65 MHz, which moves
# the widest prediction across the frequency grid (60..85 MHz, 8 points).
TWIN_AT_65 = {"replace": {"global_signal": {
    "depth": {"value": 0.5, "unit": "K"},
    "centre": {"value": 65.0, "unit": "MHz"},
    "width": {"value": 5.0, "unit": "MHz"}}}}


def document(*runs, twin=None, sigma_k=0.05):
    """The estimators' document, with the model's own gain pinned at 1.0.

    The single latent g multiplies the whole prediction, so d(data)/dg is the
    pre-gain signal.  Pinning the model gain at 1.0 -- rather than the 1.1 the
    other exits tests use -- makes the forward run's data EQUAL that
    derivative.  That equality is what lets the predict assertions below pin a
    number the model implies instead of asserting that predict returned
    something.

    ``sigma_k`` is inference.noise's sigma alone.  Nothing else in the document
    reads it -- the observed data is a deterministic forward, with no
    ``realise:`` -- so it moves the Fisher information, and only that.  It is a
    parameter of this builder so that one test can watch the propagated width
    respond to the covariance it was handed.
    """
    doc = synthetic_document()
    doc["model"] = {key: value for key, value in doc["model"].items()
                    if key != "noise"}
    doc["model"]["gain"] = {"gain": {"value": 1.0, "unit": "dimensionless"}}
    doc["runtime"] = {"seed": 20260806, "seeds": {"sample": 11}}
    inference = {
        "parameters": {"g": {"init": 1.0, "linear": True, "into": "gain.gain",
                             "prior": {"normal": {"loc": 1.0,
                                                  "scale": 0.05}}}},
        "noise": {"kind": "homoscedastic",
                  "sigma": {"value": sigma_k, "unit": "K"}},
        "observed": {"from": "simulation", "at": {"g": 1.5}},
    }
    if twin is not None:
        inference["twin"] = twin
    doc["inference"] = inference
    doc["runs"] = list(runs)
    return doc


def _never(*args, **kwargs):
    """The OTHER route's entry point, stubbed to fail loudly if reached."""
    raise AssertionError("the wrong route's entry point was called")


class TestTheCovarianceRoute:
    def test_the_width_is_the_jacobian_the_model_implies(self):
        """The delta-method width must be |d(data)/dg| * sigma_g everywhere.

        With the model gain at 1.0 the forward run's data IS d(data)/dg, so
        this compares predict against the model rather than against itself.
        The one argument swap available here -- passing product["fisher"]
        where product["covariance"] belongs -- returns a number a factor of
        (1/sigma_g)**2 away.  Both are finite, both have the prediction's
        shape, and only the number tells them apart.
        """
        results = run_document(document(FORWARD, FISHER, FROM_COV))
        signal = np.asarray(results["fwd"].product.data)
        sigma_g = float(results["cov"].product["covariance"].sigma("g"))
        width = np.asarray(results["p"].product)
        assert sigma_g == pytest.approx(0.0158725, rel=1e-4)
        assert width.shape == signal.shape == (16, 8)
        assert float(width[0, 4]) == pytest.approx(0.0078557, rel=1e-4)
        assert np.allclose(width, np.abs(signal) * sigma_g, rtol=1e-5)

    def test_the_width_responds_to_the_covariance_it_was_given(self):
        """Doubling inference.noise.sigma doubles the width, exactly.

        The Fisher information is J^T N^-1 J, so its inverse scales as
        sigma**2 and the delta-method width as sigma.  Nothing else in the
        document reads inference.noise.sigma: the observed data is a
        deterministic forward.  So this is the assertion that the propagated
        width is a function OF the covariance -- an executor that returned
        |signal| times any constant, or that returned the prediction, or
        zeros, passes the shape check above and fails here.
        """
        narrow = run_document(document(FISHER, FROM_COV, sigma_k=0.05))
        wide = run_document(document(FISHER, FROM_COV, sigma_k=0.10))
        narrow_width = np.asarray(narrow["p"].product)
        wide_width = np.asarray(wide["p"].product)
        narrow_sigma = float(narrow["cov"].product["covariance"].sigma("g"))
        wide_sigma = float(wide["cov"].product["covariance"].sigma("g"))
        assert wide_sigma / narrow_sigma == pytest.approx(2.0, rel=1e-3)
        assert np.all(wide_width > 0.0)
        assert np.allclose(wide_width, 2.0 * narrow_width, rtol=1e-4)

    def test_the_covariance_route_calls_propagate_covariance(self, monkeypatch):
        """The fisher product reaches propagate_covariance, not the other one.

        Schema 4.7.9's predict row names "push_forward / predict_from_samples"
        and mislabels the pair: propagate_covariance is the only shipped
        function that takes a covariance.  This is that decision as a
        measurement -- the samples route's entry point is stubbed to raise, so
        a predict that sent a fisher product there fails here rather than
        somewhere downstream, and the FlatMatrix assertion pins WHICH half of
        the fisher product was handed over.
        """
        import rheplicant.inference as inference
        from rheplicant.inference.uncertainty import FlatMatrix

        seen = {}
        real = inference.propagate_covariance

        def spy(forward, params, param_cov):
            seen["param_cov"] = param_cov
            seen["params"] = params
            return real(forward, params, param_cov)

        monkeypatch.setattr(inference, "propagate_covariance", spy)
        monkeypatch.setattr(inference, "predict_from_samples", _never)
        results = run_document(document(FISHER, FROM_COV))
        assert isinstance(seen["param_cov"], FlatMatrix)
        assert seen["param_cov"] is results["cov"].product["covariance"]
        assert sorted(seen["params"]) == ["g"]
        assert np.asarray(results["p"].product).shape == (16, 8)

    def test_predict_pushes_the_repaired_fit_twin(self):
        """inference.fit_twin, not built.twin.

        The forward exit evaluates built.twin and so never sees the repair:
        its widest channel stays at index 4 (74.29 MHz, nearest the declared
        75).  A predict written against built.twin would peak there too,
        while the repaired twin peaks at index 1 (63.57 MHz, nearest 65).
        Measured both ways.
        """
        results = run_document(document(FORWARD, FISHER, FROM_COV,
                                        twin=TWIN_AT_65))
        unrepaired = np.asarray(results["fwd"].product.data)
        width = np.asarray(results["p"].product)
        assert int(jnp.argmax(jnp.abs(unrepaired[0]))) == 4
        assert int(jnp.argmax(width[0])) == 1

    def test_n_draw_on_the_covariance_route_is_refused(self):
        """The delta method draws nothing, so n_draw: is refused not ignored.

        `draws nothing` is this branch's phrase alone: the sweep would accept
        n_draw: (it is the one key _PREDICT_KEYS holds), and the samples
        route's own two n_draw refusals say `exceeds` and `must be >= 1`.

        THE LIST IS MATCHED AS ONE ORDERED STRING, and the closing refusal's
        own test matches the identical literal.  Both sentences are built from
        `_DRAW_SOURCES` and both must name it in the SAME order -- two
        orderings for one table in one function is two lists a reader has to
        recognise as the same one.  Membership cannot see that: measured,
        `' / '.join(sorted(_DRAW_SOURCES))` in both messages survived every
        other test in this module.
        """
        with pytest.raises(ConfigError) as caught:
            run_document(document(FISHER, {**FROM_COV, "n_draw": 4}))
        assert type(caught.value) is ConfigError
        message = str(caught.value)
        assert "draws nothing" in message
        assert message.startswith("runs['p']:")
        assert "reuse one of plan.sample / nuts / npe." in message

    def test_an_unknown_key_is_swept(self):
        """n_draws: is the plural typo, and it is the SWEEP that catches it.

        Pinned against the from: refusal below, which fires before the sweep:
        a guard that swept from: instead of naming it would leave that test
        passing on this message.
        """
        with pytest.raises(ConfigError) as caught:
            run_document(document(FISHER, {**FROM_COV, "n_draws": 2}))
        message = str(caught.value)
        assert "does not take ['n_draws']" in message
        assert "second spelling" not in message


class TestTheSamplesRoute:
    def test_every_draw_reaches_the_prediction_and_none_of_it_is_noisy(self):
        """data = g * signal exactly, so a noiseless pushforward reproduces
        every draw to float32 roundoff.

        predict_from_samples is documented noiseless (numpyro_bridge.py:337-338)
        -- the likelihood's own scatter is NOT added back -- and this equality
        is the assertion that says so.  An executor that realised noise, or
        that pushed the posterior mean n times, breaks it.
        """
        results = run_document(document(FORWARD, SAMPLE, FROM_CHAIN))
        signal = np.asarray(results["fwd"].product.data)
        draws = np.asarray(results["chain"].product.samples["g"])
        predictive = np.asarray(results["p"].product)
        assert draws.shape == (6,)          # 10 sweeps, 4 discarded as warmup
        assert predictive.shape == (6, 16, 8)
        assert not np.allclose(predictive, predictive[0], rtol=1e-6)
        assert np.allclose(predictive, draws[:, None, None] * signal,
                           rtol=1e-5)

    def test_the_samples_route_calls_predict_from_samples(self, monkeypatch):
        """The draws reach predict_from_samples, not propagate_covariance.

        The mirror of the covariance route's own binding test.  The stub on
        the other entry point is what makes a swapped call fail HERE, and the
        recorded arguments pin the four the package takes -- the fit twin, the
        state, the space, and the stacked draws keyed by latent name.
        """
        import rheplicant.inference as inference

        seen = {}
        real = inference.predict_from_samples

        def spy(pipeline, state_template, space, samples):
            seen["pipeline"] = pipeline
            seen["state"] = state_template
            seen["space"] = space
            seen["samples"] = samples
            return real(pipeline, state_template, space, samples)

        monkeypatch.setattr(inference, "predict_from_samples", spy)
        monkeypatch.setattr(inference, "propagate_covariance", _never)
        results = run_document(document(SAMPLE, FROM_CHAIN))
        assert sorted(seen["samples"]) == ["g"]
        assert np.asarray(seen["samples"]["g"]).shape == (6,)
        assert np.asarray(results["p"].product).shape == (6, 16, 8)

    def test_the_samples_route_pushes_the_repaired_fit_twin(self):
        """inference.fit_twin, not built.twin -- on THIS route too.

        The covariance route's own twin test cannot see this one: the two
        routes reach for the pipeline separately, and a predict that read
        built.twin here alone left every other test in this file green
        (measured: the mutation survived until this test existed).  The
        forward exit evaluates built.twin, so its widest channel stays at
        index 4 (74.29 MHz, nearest the declared 75) while the repaired twin
        peaks at index 1 (63.57 MHz, nearest 65).
        """
        results = run_document(document(FORWARD, SAMPLE, FROM_CHAIN,
                                        twin=TWIN_AT_65))
        unrepaired = np.asarray(results["fwd"].product.data)
        predictive = np.asarray(results["p"].product)
        assert predictive.shape == (6, 16, 8)
        assert int(jnp.argmax(jnp.abs(unrepaired[0]))) == 4
        assert int(jnp.argmax(jnp.abs(predictive[0, 0]))) == 1

    def test_n_draw_keeps_the_last_draws(self):
        """n_draw: 2 over a 6-draw chain takes the LAST two.

        The chain's own record distinguishes the two ends -- measured -- so an
        executor that sliced from the front, or that ignored n_draw and
        returned all six, fails here.  The final assertion keeps the test
        honest if the chain ever mixes so well that its two ends stop being
        distinguishable.
        """
        results = run_document(document(FORWARD, SAMPLE,
                                        {**FROM_CHAIN, "n_draw": 2}))
        signal = float(np.asarray(results["fwd"].product.data)[0, 4])
        draws = np.asarray(results["chain"].product.samples["g"])
        predictive = np.asarray(results["p"].product)
        assert predictive.shape == (2, 16, 8)
        assert predictive[:, 0, 4] / signal == pytest.approx(draws[-2:],
                                                             rel=1e-4)
        assert not np.allclose(draws[-2:], draws[:2], rtol=1e-3)

    def test_n_draw_equal_to_the_chain_is_the_boundary_that_RUNS(self):
        """``n_draw: 6`` on a six-draw chain is not a refusal.

        The ceiling is ``keep > available``, and ``>=`` is the mutation that
        reads identically to a careless eye: it refuses a document asking for
        exactly the draws it has -- which is what an explicit ``n_draw:``
        equal to the chain length means -- and tells it "6 exceeds the 6
        draws", a sentence that is false on its face.  Measured: ``>=``
        survived every other test in this module on all three sources,
        because the refusal tests ask for one MORE than there is and the
        thinning tests for fewer, so the boundary itself was never evaluated.
        """
        results = run_document(document(SAMPLE, {**FROM_CHAIN, "n_draw": 6}))
        draws = np.asarray(results["chain"].product.samples["g"])
        predictive = np.asarray(results["p"].product)
        assert draws.shape == (6,)
        assert predictive.shape == (6, 16, 8)

    def test_n_draw_beyond_the_chain_names_PLAN_SAMPLES_OWN_reason(self):
        """Renamed and strengthened: the per-kind clause is what makes
        ``_DRAW_SOURCES`` a table rather than a set.

        ``exceeds the N draws`` is boilerplate every source shares, so
        asserting it alone passes on an implementation that told every source
        ``plan.sample``'s sentence -- and on one that told none of them any
        sentence at all.  Measured before this task: mutating the clause to
        ``MUTANT CLAUSE`` survived the whole predict module, and no test or
        page anywhere matched ``discards its warmup``.  The clause below
        belongs to this row, and ``get_samples()`` belongs to ``nuts``'.
        """
        with pytest.raises(ConfigError) as caught:
            run_document(document(SAMPLE, {**FROM_CHAIN, "n_draw": 7}))
        assert type(caught.value) is ConfigError
        message = str(caught.value)
        assert message.startswith("runs['p']:")
        assert "exceeds the 6 draws" in message
        assert "plan.sample discards its warmup before returning" in message
        assert "get_samples()" not in message

    def test_n_draw_is_a_positive_count(self):
        """0 is the one that would pass in silence.

        `stack[-0:]` is the WHOLE stack, so an executor that dropped the
        minimum would answer n_draw: 0 with every draw the chain kept -- a
        finite, correctly-shaped product for a document that asked for none.
        So the floor's own phrase is matched, not the bare key name, which
        every other n_draw refusal in this file also carries.
        """
        with pytest.raises(ConfigError, match="must be >= 1"):
            run_document(document(SAMPLE, {**FROM_CHAIN, "n_draw": 0}))
        with pytest.raises(ConfigError, match="n_draw: is a number"):
            run_document(document(SAMPLE, {**FROM_CHAIN, "n_draw": "all"}))


class TestTheNutsRoute:
    """``reuse:`` naming a ``kind: nuts`` run -- the third leg.

    Built on ``posterior_helpers``' own :data:`NUTS` and ``exit_helpers``'
    ``conjugate_document`` rather than on this module's local ``document()``:
    that builder deletes the model's noise node instead of repairing it away,
    so on a bare ``document()`` ``built.twin`` and ``built.inference.fit_twin``
    are the same object, and a test written against one cannot tell an
    executor reaching for it from an executor reaching for the other.  THE
    QUALIFIER MATTERS: pass ``twin=``, as the two fit-twin tests above do, and
    they separate again -- the sentence was copied from
    ``exit_helpers._repaired`` without it and read as unconditional, which
    those two tests disprove.  The helper's documents keep the two apart with
    no ``twin=`` to remember.

    The class fixture saves ONE chain of the five this class runs: four tests
    build their own document -- the spy, the ``n_draw: 5`` slice, the
    ``n_draw: 201`` refusal and the two-chain layout -- because each needs a
    DIFFERENT one.  That is five ``run_document`` calls and it is NOT five
    chains' worth of time: measured with the class run in isolation, the
    fixture is 1.85 s and the four later tests are 1.01 / 0.43 / 0.31 / 0.30,
    ~3.9 s for the class.  The cold pipeline is the fixture's, and everything
    after it rides that JAX compilation.  The fixture is still worth having:
    it is free, and it is what lets the first test assert a relation inside a
    product that a second run would not reproduce bit-for-bit.
    """

    @pytest.fixture(scope="class")
    @classmethod
    def chain(cls):
        return run_document(conjugate_document(
            NUTS, {"name": "p", "kind": "predict", "reuse": "chain"},
            seeds={"chain": 3}))

    def test_every_draw_reaches_the_prediction_and_none_of_it_is_noisy(
            self, chain):
        """The pushed predictions differ from each other EXACTLY as the draws
        do.

        ``g`` enters at ``gain.gain``, the last node of the twin, so the
        prediction is exactly proportional to it and
        ``predictive[i] / predictive[0]`` must equal ``draws[i] / draws[0]``
        everywhere.  This is a relation INSIDE the product, so it needs no
        forward run to compare against -- which matters here, because
        ``conjugate_document``'s model keeps its stochastic ``noise`` node and
        a ``forward`` run on it is a noise realisation, not the signal.

        Three wrong implementations die here: one that pushed the posterior
        MEAN n times (every row identical -- the third assertion), one that
        realised the likelihood's scatter on top (``predict_from_samples`` is
        documented noiseless, numpyro_bridge.py:337-338 -- the ratio stops
        being exact), and one that pushed the deterministic ``"prediction"``
        site that ``get_samples()`` also returns (wrong shape).
        """
        draws = np.asarray(chain["chain"].product.samples["g"])
        predictive = np.asarray(chain["p"].product)
        assert draws.shape == (200,)
        assert predictive.shape == (200, 16, 8)
        assert not np.allclose(predictive, predictive[0], rtol=1e-6)
        assert np.allclose(predictive / predictive[0],
                           (draws / draws[0])[:, None, None], rtol=1e-4)

    def test_the_prediction_site_never_reaches_predict_from_samples(
            self, monkeypatch):
        """``samples`` carries the latents and NOT the whole TOD.

        ``mcmc.get_samples()`` returns the deterministic ``"prediction"`` site
        beside every latent, and its per-sample shape is the whole data grid
        -- measured on this document, ``g (200,)`` against
        ``prediction (200, 16, 8)``, 128 times the latent's footprint.  The
        assertion is the ABSENCE of that key, by name: asserting that ``"g"``
        is present passes just as well when the TOD is there too.  The spy is
        what makes this a statement about what CROSSED the seam rather than
        about what the product happened to store.
        """
        import rheplicant.inference as inference

        seen = {}
        real = inference.predict_from_samples

        def spy(pipeline, state_template, space, samples):
            seen["samples"] = samples
            return real(pipeline, state_template, space, samples)

        monkeypatch.setattr(inference, "predict_from_samples", spy)
        monkeypatch.setattr(inference, "propagate_covariance", _never)
        run_document(conjugate_document(
            NUTS, {"name": "p", "kind": "predict", "reuse": "chain"},
            seeds={"chain": 3}))
        assert sorted(seen["samples"]) == ["g"]
        assert "prediction" not in seen["samples"]

    def test_n_draw_keeps_the_last_draws_of_the_chain(self):
        """``n_draw: 5`` over a 200-draw chain takes the LAST five.

        The chain's two ends are distinguishable on this document -- the final
        assertion is what keeps this test honest if they ever stop being -- so
        an executor that sliced from the front, or that ignored ``n_draw:``,
        fails here.  A separate ``run_document`` because the class fixture's
        predict declares no ``n_draw:``.
        """
        results = run_document(conjugate_document(
            NUTS, {"name": "p", "kind": "predict", "reuse": "chain",
                   "n_draw": 5},
            seeds={"chain": 3}))
        draws = np.asarray(results["chain"].product.samples["g"])
        predictive = np.asarray(results["p"].product)
        assert predictive.shape == (5, 16, 8)
        assert np.allclose(predictive / predictive[0],
                           (draws[-5:] / draws[-5])[:, None, None], rtol=1e-4)
        assert not np.allclose(draws[-5:], draws[:5], rtol=1e-9)

    def test_n_draw_beyond_the_chain_names_NUTS_OWN_reason(self):
        """The ceiling clause belongs to ``nuts`` alone.

        ``plan.sample``'s clause says "discards its warmup before returning"
        and ``npe``'s says "has no warmup to recover"; neither contains
        ``get_samples()``.  Matching the shared boilerplate ("exceeds the",
        "all there is") would pass on a single hard-coded plan.sample sentence
        told to every source -- which is the implementation this test exists
        to kill, because that sentence is FALSE for npe.
        """
        with pytest.raises(ConfigError) as caught:
            run_document(conjugate_document(
                NUTS, {"name": "p", "kind": "predict", "reuse": "chain",
                       "n_draw": 201},
                seeds={"chain": 3}))
        message = str(caught.value)
        assert message.startswith("runs['p']:")
        assert "exceeds the 200 draws" in message
        assert "get_samples() returns the post-warmup draws alone" in message
        assert "plan.sample discards" not in message

    def test_a_thinned_multi_chain_product_reads_the_LAST_chain(self):
        """Measured, said on the page, and not refused.

        ``get_samples()`` concatenates the chains in order -- measured
        directly on a two-chain NUTS run, ``get_samples()`` is
        ``get_samples(group_by_chain=True)`` reshaped, and ``flat[20:]`` IS
        chain 1 element for element -- so a tail thin of a two-chain product
        reads chain 1 alone.  This test is what makes that a documented
        property of ``predict`` rather than a surprise: the thinning happens
        after the chains are already flat, so ``n_draw: 20`` of 40 is one
        chain and not a pooled 20.

        WHICH assertion kills what, measured rather than reasoned -- because
        the first attempt at this paragraph attributed the front slice to the
        wrong one.  The RATIO assertion is what kills a front slice: under
        ``stack[:keep]`` it ends up comparing ``draws[:20] / draws[0]``
        against ``draws[-20:] / draws[-20]``, and those two normalised
        profiles differ by 1.3e-03 against its own rtol of 1e-04.  Measured:
        that mutation fails the ratio assertion and no other in this test.

        The LAST assertion kills no ``predict`` mutation at all -- ``draws``
        is the PRODUCT's stack and this exit never touches it.  It is the
        precondition that keeps the ratio assertion able to discriminate: if
        the two halves coincided, a front slice and a tail slice would
        predict the same thing and the ratio assertion would pass on both.
        It fails on a product that returned one chain twice or pooled them
        into a repeated block.  Neither assertion can see an interleave; the
        flat/grouped identity that would is not reachable from here, because
        ``NutsProduct`` carries the flat stack alone.

        50 + 20 on two sequential chains rather than the helper's 200 + 200:
        this test is about the LAYOUT of the flat stack, and the layout does
        not get truer with a longer chain.  Measured at ~0.8 s.
        """
        run = {**NUTS, "num_warmup": 50, "num_samples": 20, "num_chains": 2,
               "chain_method": "sequential"}
        results = run_document(conjugate_document(
            run, {"name": "p", "kind": "predict", "reuse": "chain",
                  "n_draw": 20},
            seeds={"chain": 3}))
        product = results["chain"].product
        draws = np.asarray(product.samples["g"])
        predictive = np.asarray(results["p"].product)
        assert product.n_draw == 40 and product.n_chain == 2
        assert draws.shape == (40,)
        assert predictive.shape == (20, 16, 8)
        assert np.allclose(predictive / predictive[0],
                           (draws[-20:] / draws[-20])[:, None, None],
                           rtol=1e-4)
        # Not a claim about predict: `draws` is the PRODUCT's stack.  This is
        # what keeps the ratio assertion above able to see a front slice --
        # with the halves coincident both slice directions predict the same
        # thing -- and it fails on a product that returned one chain twice or
        # pooled them into a repeated block.  Measured: the halves differ by
        # 1.2e-03.
        assert not np.allclose(draws[-20:], draws[:20], rtol=1e-9)


class TestTheNpeRoute:
    """``reuse:`` naming a ``kind: npe`` run -- the fourth leg.

    Task 7's ``posterior_helpers.npe_document`` builds every document here.
    Its run is named **amortized** (not the kind, for the reason :data:`NUTS`
    gives) and its four seeds are 11/12/13/14, so every ``reuse:`` and every
    result key below is ``"amortized"``.

    ``npe={"sample": {"n_draws": 12}}`` on every call.  The helper's own
    section draws 100, and this module pins draw COUNTS and stack SHAPES;
    twelve is a CHOICE and not a minimum -- with one latent the flat draws are
    ``(n, 1)`` and only ``n == 1`` makes that ambiguous with ``(1, n)``, so 2
    would show the unravel reached ``predict`` just as well.  Twelve is cheap,
    leaves room for a ``n_draw: 5`` tail that is visibly not the head, and is
    what every number below was taken at.  The subsection merge keeps
    ``sample``'s seed, which is the point of merging per subsection.

    The class fixture saves ONE pipeline of the four these tests run, and the
    three that build their own document are NOT three more cold pipelines:
    measured with the class run in isolation, the fixture is ~3.5 s and the
    three later tests are 0.42 / 0.29 / 0.24, about 4.5 s for the class (the
    tenths move a little between runs).  The fixture pays the JAX compilation
    and everything after it rides that.  An earlier version of this paragraph
    said the later tests "each pay the full ~2.8 s", which was wrong by ~11x
    in the paragraph written to be honest about cost -- and a reader told a
    cheap test is expensive deletes the wrong test.
    """

    #: The one override every test in this class passes.
    TWELVE = {"sample": {"n_draws": 12}}

    @pytest.fixture(scope="class")
    @classmethod
    def amortized(cls):
        return run_document(npe_document(
            {}, {"name": "p", "kind": "predict", "reuse": "amortized"},
            npe=cls.TWELVE))

    def test_the_unravelled_draws_reach_the_prediction(self, amortized):
        """The npe product's ``samples`` is a MAPPING, and predict reads it.

        ``NeuralPosterior.sample`` returns a flat ``(n_draws, n_params)``
        array; ``predict_from_samples`` takes ``{name: stack}`` and validates
        it against ``space.names``.  Task 8's ``_unravel`` is what bridges the
        two, and this is the assertion that it bridged them for THIS consumer
        rather than only for its own tests -- the shape ``(12, 16, 8)`` is
        reachable no other way.

        THE LAST TWO ASSERTIONS ARE ATTRIBUTION, and without them the counts,
        keys, shapes and ``isfinite`` above are all presence: measured,
        replacing this leg's draws with twelve copies of the last one --
        ``stack[-keep:] * 0 + stack[-1]``, same shape, same key, all finite --
        left the whole module green, and the spy test below could not see it
        either, because the mutation changes precisely the dict the spy
        records.  Both sibling classes carry this pair; this one did not.
        ``g`` enters at ``gain.gain`` here exactly as it does on the nuts
        document, so the prediction is proportional to the draw and the ratio
        relation is exact -- measured, max deviation 1.19e-07, float32
        roundoff.
        """
        product = amortized["amortized"].product
        draws = np.asarray(product.samples["g"])
        predictive = np.asarray(amortized["p"].product)
        assert product.n_draw == 12
        assert sorted(product.samples) == ["g"]
        assert draws.shape == (12,)
        assert predictive.shape == (12, 16, 8)
        assert np.all(np.isfinite(predictive))
        assert not np.allclose(predictive, predictive[0], rtol=1e-6)
        assert np.allclose(predictive / predictive[0],
                           (draws / draws[0])[:, None, None], rtol=1e-4)

    def test_the_npe_leg_calls_predict_from_samples(self, monkeypatch):
        """The mirror of the covariance route's binding test, on the fourth
        leg.

        ``propagate_covariance`` is stubbed to raise, so an executor that
        routed an npe product to the delta method fails HERE and not somewhere
        downstream -- and the recorded arguments pin that the draws crossed
        the seam keyed by latent name rather than as the flat array the
        package returned.
        """
        import rheplicant.inference as inference

        seen = {}
        real = inference.predict_from_samples

        def spy(pipeline, state_template, space, samples):
            seen["samples"] = samples
            return real(pipeline, state_template, space, samples)

        monkeypatch.setattr(inference, "predict_from_samples", spy)
        monkeypatch.setattr(inference, "propagate_covariance", _never)
        run_document(npe_document(
            {}, {"name": "p", "kind": "predict", "reuse": "amortized"},
            npe=self.TWELVE))
        assert sorted(seen["samples"]) == ["g"]
        assert np.asarray(seen["samples"]["g"]).shape == (12,)

    def test_n_draw_thins_the_npe_draws_from_the_tail(self):
        """The third source's ``n_draw:`` is driven at a VALID count.

        ``plan.sample`` is driven at 2, 6 and 7 and ``nuts`` at 5, 20 and 201;
        this class's only other ``n_draw`` test is the ``13 > 12`` refusal,
        which returns BEFORE the slice.  So the thinning branch was never
        entered on this leg: measured, ``keep = available`` for ``npe`` alone
        left the whole module green, and the page promises "the other three
        carry samples, which are pushed through the twin one by one,
        ``n_draw:`` thinning them from the tail".

        The tail, not the front -- the same relation the sibling classes pin,
        against ``draws[-5:]`` rather than ``draws[:5]``, and the last
        assertion is what keeps that distinction visible: twelve independent
        posterior draws do not repeat.
        """
        results = run_document(npe_document(
            {}, {"name": "p", "kind": "predict", "reuse": "amortized",
                 "n_draw": 5},
            npe=self.TWELVE))
        draws = np.asarray(results["amortized"].product.samples["g"])
        predictive = np.asarray(results["p"].product)
        assert draws.shape == (12,)
        assert predictive.shape == (5, 16, 8)
        assert np.allclose(predictive / predictive[0],
                           (draws[-5:] / draws[-5])[:, None, None], rtol=1e-4)
        assert not np.allclose(draws[-5:], draws[:5], rtol=1e-9)

    def test_n_draw_beyond_the_draws_names_NPES_OWN_reason(self):
        """npe has no warmup, so "this is all there is" needs a different why.

        The remedy differs too, and the message has to carry it: a short
        ``plan.sample`` cannot be lengthened after the fact, but an npe
        document raises ``inference.npe.sample.n_draws:`` and draws more.  A
        single shared sentence would send the user looking for a warmup that
        does not exist.
        """
        with pytest.raises(ConfigError) as caught:
            run_document(npe_document(
                {}, {"name": "p", "kind": "predict", "reuse": "amortized",
                     "n_draw": 13},
                npe=self.TWELVE))
        message = str(caught.value)
        assert message.startswith("runs['p']:")
        assert "exceeds the 12 draws" in message
        assert "inference.npe.sample.n_draws:" in message
        assert "no warmup to recover" in message
        assert "get_samples()" not in message


class TestTheReuseGrammar:
    def test_predict_without_reuse_is_refused(self):
        """The bare `is required` would also fit the missing-name refusal
        below, so the phrase matched is the whole clause -- the one that tells
        the user WHICH key to write."""
        with pytest.raises(ConfigError,
                           match=r"reuse: <run name> is required"):
            run_document(document({"kind": "predict"}))

    def test_a_forward_reference_reads_as_a_missing_run(self):
        """Runs execute in declaration order, so reuse only looks backwards."""
        with pytest.raises(ConfigError) as caught:
            run_document(document(FROM_COV, FISHER))
        message = str(caught.value)
        assert "names no earlier run" in message
        assert "by now [] have run" in message

    def test_reusing_a_refused_run_is_refused(self):
        """expect: refuse leaves product None -- there is nothing to push.

        Without this branch the None reaches `None["covariance"]` and the user
        reads a bare TypeError naming no run, so the phrase matched is the one
        that says what was missing rather than the bare word "refused", which
        the expect: refuse vocabulary uses everywhere.
        """
        refused = {**FISHER, "expect": "refuse", "jitter": -1.0}
        with pytest.raises(ConfigError) as caught:
            run_document(document(refused, FROM_COV))
        message = str(caught.value)
        assert "refused" in message
        assert "has no product to read" in message
        assert message.startswith("runs['p']:")

    def test_from_is_refused_by_name_naming_reuse(self):
        """Schema 4.7.9 spells the link both ways; reuse: is the one that
        runs._RUN_KEYS already carries, so from: is refused, not swept.

        `second spelling` belongs to this branch alone -- the sweep's own
        message for an unknown key is pinned above and does not carry it --
        and the refusal has to NAME reuse:, because renaming the key is the
        whole remedy.
        """
        with pytest.raises(ConfigError) as caught:
            run_document(document(FISHER, {**FROM_COV, "from": "cov"}))
        assert type(caught.value) is ConfigError
        message = str(caught.value)
        assert "second spelling" in message
        assert "reuse:" in message
        assert message.startswith("runs['p']:")

    def test_from_is_refused_even_when_it_is_the_only_link(self):
        """from: alone must not read as a missing reuse:.

        `reuse: is required` would send the user to add a key they had
        already written under another name, so the naming refusal comes
        first -- measured here rather than assumed from the source order.
        """
        with pytest.raises(ConfigError, match="second spelling"):
            run_document(document(FISHER, {"name": "p", "kind": "predict",
                                           "from": "cov"}))

    @pytest.mark.parametrize(
        ("earlier", "reuse"),
        [(ESTIMATE, "point"), (FORWARD, "fwd"), (OPTIMIZE, "fit"),
         (WIENER_FISHER, "w")],
    )
    def test_reusing_another_kind_names_the_four_that_work(self, earlier,
                                                           reuse):
        """The dispatch is on the run's KIND, never on its product's shape.

        plan.estimate is the near miss of family -- same estimators, no
        distribution -- while forward carries a State and optimize a
        {params, losses}: three shapes that could not be mistaken for any
        route.  ``conjugate.wiener`` with ``width: fisher`` is the near miss
        that COLLIDES: `_gaussian_width` merges {"fisher", "covariance"} into
        its product (conjugate.py:195, :392), so a predict dispatching on
        `"covariance" in earlier.product` accepts it and returns a finite,
        correctly-shaped delta-method width computed from a covariance over
        the CONJUGATE BLOCK's latents rather than inference.parameters.

        ALL FOUR runnable sources are asserted, not two.  The message named
        two kinds when this test was written and names four now, and a message
        that grows satisfies more substrings than it did: `"fisher" in message
        and "plan.sample" in message` passes on a four-kind message that has
        dropped `nuts` and `npe` from the sentence entirely, which is exactly
        the half-shipped state this task can reach.

        The phrase matched for the BRANCH is "knows how to propagate", which
        belongs to this refusal alone (grep: one hit in src/).  The bare kind
        is not enough -- the refusal's own boilerplate says "predict pushes
        FORWARD either a fisher run's ...", on which the `forward` leg's
        bare-word assertion passed no matter what the message said about the
        run -- and "kind: <name>" is not enough either, because the covariance
        route's n_draw refusal also says "names a kind: fisher run".

        The three draw sources are ALSO matched as one ordered string, the
        identical literal `test_n_draw_on_the_covariance_route_is_refused`
        matches: both sentences are built from `_DRAW_SOURCES` and both must
        name it in the same order.  Membership alone cannot see a reordering
        -- measured, `sorted(_DRAW_SOURCES)` in both messages survived every
        other test in this module.

        AND THE SENTENCE'S OWN COUNT, which is the same trap one turn later.
        Half of it grows with the table (`' / '.join(...)`) and the count was
        hand-written "four" -- measured, "four" -> "three" survived all 39
        tests, and this commit's own `conjugate.gcr` docstring below calls
        that kind the most plausible one for a later plan to promote.  The
        source now derives the count from `len(_DRAW_SOURCES) + 1` so the two
        halves cannot disagree; this assertion is what pins the rendered
        result, so growing the table turns BOTH literals red together and the
        author updates one test rather than discovering a self-contradictory
        message.
        """
        with pytest.raises(ConfigError) as caught:
            run_document(document(earlier, {**FROM_CHAIN, "reuse": reuse}))
        assert type(caught.value) is ConfigError
        message = str(caught.value)
        assert "knows how to propagate" in message
        assert f"kind: {earlier['kind']}" in message
        for source in ("fisher", "plan.sample", "nuts", "npe"):
            assert source in message, source
        assert "draws of a plan.sample / nuts / npe run" in message
        assert "Those are the 4 products this exit knows how to " \
               "propagate." in message
        assert message.startswith("runs['p']:")

    @pytest.mark.parametrize("kind", ["condition", "conjugate.gcr"])
    def test_a_kind_whose_product_HAS_samples_is_still_refused(self, kind):
        """The samples branch dispatches on the kind, not on `.samples`.

        This is 2C's `test_a_future_kind_whose_product_HAS_samples_is_still_
        refused`, rewritten rather than deleted at the moment its own
        docstring anticipated: it asserted that a `nuts`-kinded product was
        refused, and Plan 2D made `nuts` a source.  What the test was ABOUT --
        that a `hasattr(earlier.product, "samples")` refactor is caught -- is
        unchanged, so the kind moves and the assertion stays.

        Two kinds, for two different reasons.  `condition` returns a
        conditioning number and can never be a predict source, so this leg
        will not have to move again.  `conjugate.gcr` is the real near miss:
        its product genuinely IS a stack of posterior draws (a dict under
        `"draws"`, measured), it is the most plausible kind for a later plan
        to promote, and it is refused today because the TABLE does not list
        it -- not because its product looks wrong.  A `hasattr` refactor
        accepts both of these fakes and fails here.

        The handler is driven directly, because a document cannot construct
        a product of one kind and a run of another.  The kind check is
        result-dependent, so it sits in ``pre_execute`` (Plan 4A Task 9's
        deferred boundary).
        """
        from rheplicant.config.document import load_document
        from rheplicant.config.sections.exit_support import (
            handler_for,
            parse_run,
        )
        from rheplicant.config.sections.runs import RunResult, RunSpec

        built = load_document(document(FISHER))

        class Chainlike:
            n_draw = 3
            samples = {"g": jnp.ones((3,))}

        spec = RunSpec(name="p", kind="predict", variant=None, on="primary",
                       expect="ok", options={}, reuse="other")
        results = {"other": RunResult(name="other", kind=kind,
                                      product=Chainlike(), error=None)}
        parsed = parse_run(spec, built, index=1, layer=_BASE_LAYER)
        with pytest.raises(ConfigError) as caught:
            handler_for("predict").pre_execute(parsed, built, results)
        message = str(caught.value)
        assert f"kind: {kind}" in message
        assert "knows how to propagate" in message
        assert message.startswith("runs['p']:")

    def test_the_2c_deferral_table_is_gone(self):
        """predict is the last 2C kind, so the tuple retires with it.

        An empty _KINDS_2C would leave `if kind in ()` in the parser and a
        loop over nothing in the runs tests: both green, both meaningless.
        This asserts absence, not emptiness.
        """
        from rheplicant.config.sections import runs as runs_module

        assert not hasattr(runs_module, "_KINDS_2C")
        assert "predict" in runs_module._KINDS


class TestAVariantMismatchIsRefused:
    """A predict must read the build its own product came from.

    Both routes re-derive their expansion point (the covariance route) or
    their pipeline (the samples route) from THIS run's ``built``, so a
    ``predict`` naming a different ``variant:`` from the run it reuses mixes
    two builds.  ``RunResult`` carries the variant its run was configured on,
    and the comparison sits before the route dispatch, which is why one
    refusal guards both.

    What the refusal is worth is a NUMBER, and the last two tests keep it
    measured rather than asserted, by relabelling the earlier result so the
    guard lets the mixture through: the package catches a variant that moves
    the parameter LAYOUT (uncertainty.py:533) and is silent about the rest --
    a model-only mismatch comes back finite, correctly shaped and 1.1 %
    wrong (0.0014147 against the un-mixed 0.0014307 on the SAME variant: a
    ratio of 0.98883, 1.12 % at the worst channel).  With one latent the
    whole error is the scalar sigma_g(base)/sigma_g(at_65).  1.1 % is
    precisely the error nobody notices, which is the argument for refusing
    it instead of recording it.
    """

    AT_65 = {"at_65": {"model": {"global_signal": {
        "depth": {"value": 0.5, "unit": "K"},
        "centre": {"value": 65.0, "unit": "MHz"},
        "width": {"value": 5.0, "unit": "MHz"}}}}}
    #: A variant that moves the parameter LAYOUT -- the half the package
    #: catches on its own, and the companion measurement at the foot.
    TWO = {"two": {"inference": {"parameters": {
        "g": {"init": 1.0, "linear": True, "into": "gain.gain",
              "prior": {"normal": {"loc": 1.0, "scale": 0.05}}},
        "d": {"init": 0.5, "linear": True, "into": "global_signal.depth",
              "prior": {"normal": {"loc": 0.5, "scale": 0.5}}}}}}}

    @staticmethod
    def _bypass(variants, name):
        """The mixture the guard forbids, reached by lying to the guard.

        The fisher run executes on the BASE build and its RunResult is
        relabelled with the predict's own variant, so the comparison passes
        and everything downstream is exactly what ``run_document`` did before
        this task.  It is the only route left to what the refusal prevents,
        and measuring that is what makes the refusal arguable rather than
        assumed.  ``_replace(variant=...)`` is also a second, structural
        assertion that the field exists: without it this raises
        ``ValueError: Got unexpected field names: ['variant']``.
        """
        from rheplicant.config.document import load_document
        from rheplicant.config.sections.exit_support import (
            handler_for,
            parse_run,
        )
        from rheplicant.config.sections.exits import execute_run
        from rheplicant.config.sections.runs import RunSpec

        doc = document(FISHER)
        doc["variants"] = variants
        cov = execute_run(RunSpec(name="cov", kind="fisher", variant=None,
                                  on="primary", expect="ok", options={}),
                          load_document(doc, variant=None))
        spec = RunSpec(name="p", kind="predict", variant=name, on="primary",
                       expect="ok", options={}, reuse="cov")
        built = load_document(doc, variant=name)
        parsed = parse_run(spec, built, index=1, layer=_BASE_LAYER)
        handler = handler_for("predict")
        results = {"cov": cov._replace(variant=name)}
        handler.pre_execute(parsed, built, results)
        return handler.execute(parsed, built, results)

    def test_the_covariance_route_refuses_a_mismatch(self):
        """Both runs named, both variants named, in the layer's own voice.

        `MIXES TWO BUILDS` belongs to this branch alone -- every neighbouring
        refusal in this file says `names no earlier run`, `has no product to
        read`, `second spelling` or `draws nothing`, and matching the bare
        word `variant` would be satisfied by any of the run grammar's own
        refusals in runs.py.

        THE ATTRIBUTION IS PINNED AS ONE ORDERED SUBSTRING, not as two
        membership checks.  Both variants appear in this message, so
        `"'at_65'" in message and "None" in message` is satisfied just as
        well by a message that has swapped them -- telling the reader their
        `predict` ran on the base and the `fisher` on at_65, which is exactly
        backwards and sends them to edit the wrong line.  Membership cannot
        see that; the ordered form is what kills it.
        """
        doc = document(FISHER, {**FROM_COV, "variant": "at_65"})
        doc["variants"] = self.AT_65
        with pytest.raises(ConfigError) as caught:
            run_document(doc)
        message = str(caught.value)
        assert "MIXES TWO BUILDS" in message
        assert "variant: 'at_65', and reuse: 'cov' ran on variant: None" in \
            message
        assert message.startswith("runs['p']:")

    def test_the_samples_route_refuses_the_same_mismatch(self):
        """The twin route, which no covariance test can reach.

        `predict` has two routes and the comparison sits before the dispatch
        precisely so that one guard serves both; a fix written inside the
        `earlier.kind == "fisher"` branch passes the test above and leaves
        this one open.  Task 9 adds two more routes to the same function,
        which is what makes the placement worth a test of its own.
        """
        doc = document(SAMPLE, {**FROM_CHAIN, "variant": "at_65"})
        doc["variants"] = self.AT_65
        with pytest.raises(ConfigError) as caught:
            run_document(doc)
        assert "MIXES TWO BUILDS" in str(caught.value)
        assert str(caught.value).startswith("runs['p']:")

    def test_the_same_variant_on_both_runs_still_runs(self):
        """The leg a naive `if run.variant or earlier.variant` gets wrong.

        `unity_gain` is synthetic_document's own variant and pins the gain at
        1.0 -- which is what this document already declares -- so running
        both runs on it must reproduce the base document's width to the last
        bit.  A guard that refused whenever EITHER run named a variant fails
        here; a guard comparing the two passes.  Both-None is every other
        test in this file and cannot separate those two implementations.

        ONE OF THE TWO NAMES IS BUILT AT RUNTIME AND IS DELIBERATELY NOT THE
        SAME OBJECT.  Two equal string literals in one module are interned to
        one object, so `is not` holds between them and a guard written
        `if earlier.variant is not run.variant` passes this test while
        spuriously refusing the identical document loaded from YAML --
        measured: `json.loads('{"a":"at_65","b":"at_65"}')` gives
        `d['a'] is d['b']` False, and `yaml.safe_load` the same.  With the
        join below, `is not` FAILS here and `!=` passes, which is what makes
        this test the thing that says `!=` rather than `is not`.
        """
        same = "".join(["unity_", "gain"])   # equal, NOT the same object
        assert same == "unity_gain" and same is not "unity_gain"  # noqa: F632
        both = document({**FISHER, "variant": "unity_gain"},
                        {**FROM_COV, "variant": same})
        neither = document(FISHER, FROM_COV)
        on_variant = np.asarray(run_document(both)["p"].product)
        on_base = np.asarray(run_document(neither)["p"].product)
        assert float(on_variant[0, 4]) == pytest.approx(0.0078557, rel=1e-4)
        assert np.array_equal(on_variant, on_base)

    def test_what_the_refusal_prevents_is_one_per_cent(self):
        """The gap, still a number after the guard closed it.

        Measured both ways: the un-mixed answer on at_65 is 0.0014307 and the
        mixture 0.0014147, a ratio of 0.98883.  `rel=1e-4` on both is tighter
        than the 1.1 % between them, so neither pin can be satisfied by the
        other's number -- which is the whole point of keeping two.  The base
        document's own 0.0078557 is a DIFFERENT model and 5.5x away; it is
        not the baseline for a mixing error and was mistaken for one once.
        """
        doc = document({**FISHER, "variant": "at_65"},
                       {**FROM_COV, "variant": "at_65"})
        doc["variants"] = self.AT_65
        unmixed = np.asarray(run_document(doc)["p"].product)
        mixed = np.asarray(self._bypass(self.AT_65, "at_65"))
        assert unmixed.shape == mixed.shape == (16, 8)
        assert float(unmixed[0, 4]) == pytest.approx(0.0014307, rel=1e-4)
        assert float(mixed[0, 4]) == pytest.approx(0.0014147, rel=1e-4)
        ratio = mixed / unmixed
        assert float(np.max(ratio)) == pytest.approx(0.98883, rel=1e-4)
        assert float(np.max(np.abs(1.0 - ratio))) == pytest.approx(0.01117,
                                                                   rel=1e-3)

    def test_a_variant_that_moves_the_layout_is_the_packages_refusal(self):
        """Underneath the config guard, the package still speaks first.

        Not a ConfigError, and no ``runs['p']:`` prefix -- the companion
        measurement for every config-layer refusal in this file, whose prefix
        and type are what tell them apart from this one.  Reached through the
        bypass, because this layer now refuses that document before the
        package ever sees it: the config refusal is a strict ADDITION, not a
        replacement, and this is the test that says so.
        """
        from rheplicant.core.errors import StateValidationError

        with pytest.raises(StateValidationError) as caught:
            self._bypass(self.TWO, "two")
        message = str(caught.value)
        assert not isinstance(caught.value, ConfigError)
        assert "runs['p']:" not in message
        assert "the flattened orderings differ" in message


class TestTheOptionalDependencyStaysOptional:
    def test_importing_the_config_layer_does_not_import_numpyro(self):
        """The samples route needs numpyro; importing a document must not.

        predict_from_samples calls _require_numpyro (numpyro_bridge.py:340),
        which is why it is imported inside the executor body rather than at
        the head of diagnostics.py.  A module-level import would put numpyro
        on every `import rheplicant.config`, and nothing else in this file
        could see it.  Run in a fresh interpreter, because this worker has
        already imported numpyro through the samples route above.
        """
        import os
        import subprocess
        import sys

        script = (
            "import sys;"
            "import rheplicant.config;"
            "from rheplicant.config.sections import exits;"
            "print(sorted(m for m in ('numpyro', 'rheplicant.inference')"
            " if m in sys.modules))"
        )
        base = {"PATH": "/usr/bin:/bin"}
        if os.environ.get("PYTHONPATH"):
            base["PYTHONPATH"] = os.environ["PYTHONPATH"]
        proc = subprocess.run([sys.executable, "-c", script],
                              capture_output=True, text=True, check=True,
                              env=base)
        assert proc.stdout.strip() == "[]", proc.stdout
