"""The fisher and optimize exits, end to end from a document."""

import jax
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.runs import run_document
from rheplicant.inference import mean_squared_error
from tests.config.test_config_document import synthetic_document


def half_mse(prediction, observed):
    """Half the package mse -- importable, and observably not the default."""
    return 0.5 * mean_squared_error(prediction, observed)


def document(run, inference=None, model_gain=1.1):
    doc = synthetic_document()
    doc["model"] = {key: value for key, value in doc["model"].items()
                    if key != "noise"}
    doc["model"]["gain"] = {"gain": {"value": model_gain,
                                     "unit": "dimensionless"}}
    doc["inference"] = inference or {
        # prior scale 0.05 makes space=True's narrowing a ~5% effect --
        # measurable in float32 (with scale 10.0 the margin is 1e-6 relative,
        # measured by the pre-execution review). optimize never reads it.
        "parameters": {"g": {"init": 1.0, "linear": True,
                             "into": "gain.gain",
                             "prior": {"normal": {"loc": 1.0,
                                                  "scale": 0.05}}}},
        "noise": {"kind": "homoscedastic",
                  "sigma": {"value": 0.05, "unit": "K"}},
        "observed": {"from": "simulation", "at": {"g": 1.5}},
    }
    doc["runs"] = [run]
    return doc


class TestFisher:
    def test_the_product_is_fisher_and_covariance(self):
        results = run_document(document({"kind": "fisher"}))
        product = results["fisher"].product
        sigma = float(product["covariance"].sigma("g"))
        assert sigma > 0.0
        assert product["fisher"].matrix.shape == (1, 1)

    def test_space_true_adds_the_declared_prior_curvature(self):
        flat = run_document(document({"kind": "fisher"}))
        posterior = run_document(document({"kind": "fisher", "space": True}))
        assert float(posterior["fisher"].product["covariance"].sigma("g")) \
            < float(flat["fisher"].product["covariance"].sigma("g"))

    def test_noise_kind_none_is_refused_naming_the_legal_exits(self):
        doc = document({"kind": "fisher"})
        doc["inference"].pop("noise")
        doc["inference"].pop("observed")
        with pytest.raises(ConfigError, match="forward and optimize"):
            run_document(doc)

    def test_without_parameters_fisher_is_refused(self):
        doc = document({"kind": "fisher"})
        doc["inference"] = {"noise": {"kind": "homoscedastic",
                                      "sigma": {"value": 0.05, "unit": "K"}}}
        with pytest.raises(ConfigError, match="parameters"):
            run_document(doc)

    def test_unknown_keys_are_swept(self):
        with pytest.raises(ConfigError, match="jitters"):
            run_document(document({"kind": "fisher", "jitters": 0.1}))

    def test_fisher_evaluates_the_repaired_fit_twin(self):
        # The output is g * signal, so d(data)/dg is the pre-gain signal:
        # doubling the absorption depth through twin: doubles the Jacobian
        # and halves sigma. A fisher run on the MODEL twin instead of
        # inference.fit_twin cannot see the repair at all.
        flat = run_document(document({"kind": "fisher"}))
        doc = document({"kind": "fisher"})
        doc["inference"]["twin"] = {"replace": {"global_signal": {
            "depth": {"value": 1.0, "unit": "K"},
            "centre": {"value": 75.0, "unit": "MHz"},
            "width": {"value": 5.0, "unit": "MHz"}}}}
        repaired = run_document(doc)
        assert float(repaired["fisher"].product["covariance"].sigma("g")) == \
            pytest.approx(
                0.5 * float(flat["fisher"].product["covariance"].sigma("g")),
                rel=1e-3)

    def test_a_negative_jitter_is_refused(self):
        with pytest.raises(ConfigError, match="jitter"):
            run_document(document({"kind": "fisher", "jitter": -1.0}))

    def test_jitter_is_an_undeclared_prior_of_width_one_over_sqrt(self):
        # F ~ 4e3 here, so against jitter: 1e12 the data is negligible and
        # sigma pins to 1/sqrt(jitter) -- parameter_covariance's own words.
        results = run_document(document({"kind": "fisher", "jitter": 1.0e12}))
        assert float(results["fisher"].product["covariance"].sigma("g")) == \
            pytest.approx(1.0e-6, rel=1e-3)


class TestOptimize:
    RUN = {"kind": "optimize", "optimizer": "gradient",
           "learning_rate": 1.0, "n_steps": 200}

    def test_gradient_descent_recovers_the_truth(self):
        results = run_document(document(self.RUN))
        product = results["optimize"].product
        assert float(product["params"]["g"]) == pytest.approx(1.5, abs=1e-3)
        assert float(product["losses"][-1]) < float(product["losses"][0])

    def test_adam_takes_its_own_knobs(self):
        results = run_document(document({**self.RUN, "optimizer": "adam",
                                         "n_steps": 300, "beta1": 0.8}))
        product = results["optimize"].product
        assert product["losses"].shape == (300,)
        assert float(product["losses"][-1]) < float(product["losses"][0])

    def test_adam_a_huge_eps_visibly_freezes_the_fit(self):
        # eps: 1e6 caps the step at ~learning_rate/eps, so g cannot leave
        # its init -- the one knob whose honouring is visible in the product
        # (defaults reach 1.5 exactly; declared-but-dropped knobs would too).
        results = run_document(document({**self.RUN, "optimizer": "adam",
                                         "n_steps": 300, "eps": 1.0e6}))
        product = results["optimize"].product
        assert float(product["params"]["g"]) == pytest.approx(1.0, abs=1e-3)
        assert float(product["losses"][-1]) > 0.01

    def test_a_non_numeric_knob_is_a_config_refusal(self):
        with pytest.raises(ConfigError, match="n_steps"):
            run_document(document({**self.RUN, "n_steps": "fast"}))
        with pytest.raises(ConfigError, match="learning_rate"):
            run_document(document({**self.RUN, "learning_rate": True}))

    def test_beta1_on_gradient_is_refused(self):
        with pytest.raises(ConfigError, match="beta1"):
            run_document(document({**self.RUN, "beta1": 0.8}))

    def test_optimizer_learning_rate_and_n_steps_are_required(self):
        for missing in ("optimizer", "learning_rate", "n_steps"):
            run = {key: value for key, value in self.RUN.items()
                   if key != missing}
            with pytest.raises(ConfigError, match=missing):
                run_document(document(run))

    def test_the_trainable_route_needs_no_parameters(self):
        doc = document(self.RUN)
        doc["inference"] = {
            "twin": {"replace": {"gain": {"gain": {"value": 1.0,
                                                   "unit": "dimensionless"}}}},
            "trainable": {"leaves": ["gain.gain"]},
            "observed": {"from": "simulation"},
        }
        results = run_document(doc)
        product = results["optimize"].product
        fitted = [x for x in jax.tree.leaves(product["params"])
                  if x is not None]
        assert len(fitted) == 1
        assert abs(float(fitted[0]) - 1.1) < 0.1
        assert float(product["losses"][-1]) < float(product["losses"][0])

    def test_trainable_and_parameters_together_are_ambiguous(self):
        doc = document(self.RUN)
        doc["inference"]["trainable"] = {"leaves": ["gain.gain"]}
        with pytest.raises(ConfigError, match="trainable"):
            run_document(doc)

    def test_optimize_needs_observed_data(self):
        doc = document(self.RUN)
        doc["inference"].pop("observed")
        with pytest.raises(ConfigError, match="observed"):
            run_document(doc)

    def test_a_python_loss_is_imported_not_called(self):
        results = run_document(document(
            {**self.RUN, "loss": {"python":
                                  "rheplicant.inference:mean_squared_error"}}))
        assert float(results["optimize"].product["params"]["g"]) == \
            pytest.approx(1.5, abs=1e-3)

    def test_the_python_loss_is_the_loss_the_fit_minimizes(self):
        # half_mse above scores exactly half of mse everywhere, so the first
        # recorded loss says which function .fit was actually handed --
        # a run that quietly falls back to mse cannot produce the ratio.
        mse = run_document(document(self.RUN))["optimize"].product
        half = run_document(document({**self.RUN, "loss": {
            "python": "tests.config.test_config_exits_estimators:half_mse"}}))
        product = half["optimize"].product
        assert float(product["losses"][0]) == \
            pytest.approx(0.5 * float(mse["losses"][0]), rel=1e-5)
        assert float(product["losses"][-1]) < float(product["losses"][0])
