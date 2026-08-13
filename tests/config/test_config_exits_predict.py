"""kind: predict -- both reuse routes, end to end from a document."""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.runs import run_document
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
        """
        with pytest.raises(ConfigError) as caught:
            run_document(document(FISHER, {**FROM_COV, "n_draw": 4}))
        assert type(caught.value) is ConfigError
        message = str(caught.value)
        assert "draws nothing" in message
        assert message.startswith("runs['p']:")
        assert "plan.sample" in message

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

    def test_n_draw_beyond_the_chain_is_refused(self):
        with pytest.raises(ConfigError) as caught:
            run_document(document(SAMPLE, {**FROM_CHAIN, "n_draw": 7}))
        assert type(caught.value) is ConfigError
        message = str(caught.value)
        assert "exceeds the 6 draws" in message
        assert message.startswith("runs['p']:")

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
    def test_reusing_another_kind_names_the_two_that_work(self, earlier,
                                                          reuse):
        """The dispatch is on the run's KIND, never on its product's shape.

        plan.estimate is the near miss of family -- same estimators, no
        distribution -- while forward carries a State and optimize a
        {params, losses}: three shapes that could not be mistaken for either
        route.  ``conjugate.wiener`` with ``width: fisher`` is the near miss
        that COLLIDES: `_gaussian_width` merges {"fisher", "covariance"} into
        its product (conjugate.py:195, :392), so a predict dispatching on
        `"covariance" in earlier.product` accepts it and returns a finite,
        correctly-shaped delta-method width computed from a covariance over
        the CONJUGATE BLOCK's latents rather than inference.parameters.
        Measured: with the three legs above alone that refactor survives the
        whole 1276-test config suite.  Plan 2D's `nuts` product will carry
        `.samples`, so the same temptation exists on the other route.

        The phrase matched is ``kind: <name>`` and not the bare kind, because
        the refusal's own boilerplate says "predict pushes FORWARD either a
        fisher run's ..." -- on which the `forward` leg's bare-word assertion
        passed no matter what the message said about the run.
        """
        with pytest.raises(ConfigError) as caught:
            run_document(document(earlier, {**FROM_CHAIN, "reuse": reuse}))
        assert type(caught.value) is ConfigError
        message = str(caught.value)
        assert f"kind: {earlier['kind']}" in message
        assert "fisher" in message
        assert "plan.sample" in message
        assert message.startswith("runs['p']:")

    def test_a_future_kind_whose_product_HAS_samples_is_still_refused(self):
        """The samples branch dispatches on the kind, not on `.samples`.

        The wiener leg above closes the covariance half of that through a real
        document, because `conjugate.wiener` with `width: fisher` already
        carries a colliding "covariance" key.  Nothing SHIPPED collides on the
        samples side, so a `hasattr(earlier.product, "samples")` refactor is
        today an equivalent mutation -- and stops being one the moment Plan
        2D lands `nuts`, whose product carries draws under exactly that name.
        This is that future, brought forward: the executor is called directly
        with a RunResult of another kind whose product answers `.samples`, and
        it must still refuse by name.
        """
        from rheplicant.config.document import load_document
        from rheplicant.config.sections.diagnostics import _run_predict
        from rheplicant.config.sections.runs import RunResult, RunSpec

        built = load_document(document(FISHER))

        class Chainlike:
            n_draw = 3
            samples = {"g": jnp.ones((3,))}

        spec = RunSpec(name="p", kind="predict", variant=None, on="primary",
                       expect="ok", options={}, reuse="future")
        results = {"future": RunResult(name="future", kind="nuts",
                                       product=Chainlike(), error=None)}
        with pytest.raises(ConfigError) as caught:
            _run_predict(spec, built, results=results)
        message = str(caught.value)
        assert "kind: nuts" in message
        assert "plan.sample" in message
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


class TestAVariantMismatchIsThePACKAGESRefusal:
    """What a predict on a DIFFERENT variant from its fisher actually does.

    ``propagate_covariance`` re-derives the expansion point from THIS run's
    ``built``, and ``RunResult`` carries no variant, so the config layer cannot
    see the mixture.  It is Plan 2D's ledger item rather than a refusal here
    -- and these two tests are what that record is worth: one measures the
    case the package catches, the other measures the case it does not, so the
    gap is a number rather than a sentence.
    """

    @staticmethod
    def _with_variants(run, variants):
        doc = document(FISHER, {**FROM_COV, **run})
        doc["variants"] = variants
        return doc

    def test_a_model_only_variant_mixes_two_builds_in_silence(self):
        """The recorded gap, as a number -- and the number is SMALL.

        The covariance was computed on the base build and the Jacobian on the
        variant's, and the parameter layout is the same on both, so nothing
        refuses.  The baseline that says how wrong that is is the un-mixed
        answer on the SAME variant, not the un-mixed answer on the base
        document -- 0.0078557 belongs to a different model and the 5.5x
        between them is the model difference, not the mixing error.  Measured
        (both runs on at_65): 0.0014307, against 0.0014147 mixed -- a ratio of
        0.98883, and 1.12 % at the worst channel of the grid.  With one latent
        the whole error is the scalar sigma_g(base)/sigma_g(at_65).

        So the danger is not that mixing two builds is wildly wrong; it is
        that it is invisibly wrong, which is the argument for closing the gap
        rather than living with it.  ``rel=1e-3`` is what makes this pin
        discriminate: it is tighter than the 1.1 % shift, so a predict that
        started reading the variant would fail here and have to say so, rather
        than closing the gap by accident.
        """
        doc = self._with_variants(
            {"variant": "at_65"},
            {"at_65": {"model": {"global_signal": {
                "depth": {"value": 0.5, "unit": "K"},
                "centre": {"value": 65.0, "unit": "MHz"},
                "width": {"value": 5.0, "unit": "MHz"}}}}})
        width = np.asarray(run_document(doc)["p"].product)
        assert width.shape == (16, 8)
        assert float(width[0, 4]) == pytest.approx(0.0014147, rel=1e-3)

    def test_a_variant_that_moves_the_layout_is_refused_by_the_package(self):
        """And the refusal is the PACKAGE's, in the package's own voice.

        Not a ConfigError, and it carries no ``runs['p']:`` prefix -- which is
        the companion measurement for every config-layer refusal above: their
        prefix and their type are what tells them apart from this one.  The
        config layer says nothing here on purpose: uncertainty.py:533 names
        both structures, and nothing this layer could add would be more
        specific.
        """
        from rheplicant.core.errors import StateValidationError

        doc = self._with_variants(
            {"variant": "two"},
            {"two": {"inference": {"parameters": {
                "g": {"init": 1.0, "linear": True, "into": "gain.gain",
                      "prior": {"normal": {"loc": 1.0, "scale": 0.05}}},
                "d": {"init": 0.5, "linear": True,
                      "into": "global_signal.depth",
                      "prior": {"normal": {"loc": 0.5, "scale": 0.5}}}}}}})
        with pytest.raises(StateValidationError) as caught:
            run_document(doc)
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
