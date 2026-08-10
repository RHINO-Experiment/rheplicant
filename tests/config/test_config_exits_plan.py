"""plan.estimate / plan.sample from a document: blocks, seeds, warm starts."""

import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.runs import run_document
from tests.config.test_config_document import synthetic_document


def document(run, seeds=None):
    doc = synthetic_document()
    doc["model"] = {key: value for key, value in doc["model"].items()
                    if key != "noise"}
    doc["runtime"] = {"seed": 20260806, "seeds": seeds or {"sample": 11}}
    doc["inference"] = {
        "parameters": {"g": {"init": 1.0, "linear": True,
                             "into": "gain.gain",
                             "prior": {"normal": {"loc": 1.0,
                                                  "scale": 10.0}}}},
        "noise": {"kind": "homoscedastic",
                  "sigma": {"value": 0.05, "unit": "K"}},
        "observed": {"from": "simulation", "at": {"g": 1.5},
                     "realise": {"kind": "homoscedastic",
                                 "sigma": {"value": 0.05, "unit": "K"},
                                 "seed": {"from":
                                          "runtime.seeds.observed_noise"}}},
    }
    doc["runs"] = [run]
    return doc


ESTIMATE = {"kind": "plan.estimate", "blocks": [{"names": ["g"]}],
            "check_identifiability": False}
SAMPLE = {"kind": "plan.sample", "blocks": [{"names": ["g"]}],
          "seed": {"from": "runtime.seeds.sample"}, "n_sweeps": 10,
          "warmup": 4, "check_identifiability": False}
# A NUTS chain, because an exact conjugate draw forgets the init: only a
# gradient block can SEE what a warm start moved.
GRADIENT_SAMPLE = {**SAMPLE, "n_sweeps": 6, "warmup": 2,
                   "blocks": [{"names": ["g"], "engine": "gradient"}]}
WARMED = {**GRADIENT_SAMPLE,
          "warm_start": {"kind": "plan.estimate",
                         "blocks": [{"names": ["g"]}],
                         "move": ["g"]}}


class TestEstimate:
    def test_the_conjugate_block_recovers_the_truth(self):
        results = run_document(document(ESTIMATE))
        estimate = results["plan.estimate"].product
        assert float(estimate.values["g"]) == pytest.approx(1.5, abs=0.05)
        assert estimate.diagnostics.converged is True
        assert estimate.diagnostics.engines[("g",)] == "conjugate"

    def test_a_seed_on_estimate_is_check_a29(self):
        # Matched on "A29", not on "estimate": the generic key sweep would
        # also refuse a seed, so a looser match could not tell the deliberate
        # explanation (placed BEFORE the sweep) from the fallback.
        with pytest.raises(ConfigError, match="A29"):
            run_document(document({**ESTIMATE,
                                   "seed": {"from": "runtime.seeds.sample"}}))

    def test_estimate_takes_no_sample_only_keys(self):
        with pytest.raises(ConfigError, match="warmup"):
            run_document(document({**ESTIMATE, "warmup": 4}))

    def test_blocks_are_required(self):
        run = {key: value for key, value in ESTIMATE.items()
               if key != "blocks"}
        with pytest.raises(ConfigError, match="blocks"):
            run_document(document(run))

    def test_a_block_entry_sweeps_its_own_keys(self):
        with pytest.raises(ConfigError, match="step"):
            run_document(document({**ESTIMATE,
                                   "blocks": [{"names": ["g"], "step": 5}]}))

    def test_declared_knobs_reach_the_plan(self):
        results = run_document(document({**ESTIMATE, "max_iter": 7,
                                         "tol": None}))
        assert results["plan.estimate"].product.diagnostics.sweeps == 7

    def test_the_default_check_identifiability_runs_and_is_recorded(self):
        # The default ("once") RUNS on a float32 twin: identifiability()
        # forces x64 internally and casts the latents, so the promoted
        # Jacobian is float64 and the report lands rather than refusing.
        run = {key: value for key, value in ESTIMATE.items()
               if key != "check_identifiability"}
        results = run_document(document(run))
        d = results["plan.estimate"].product.diagnostics
        assert d.identifiability is not None
        assert d.identifiability.rank == 1

    def test_check_identifiability_false_stores_no_report(self):
        results = run_document(document(ESTIMATE))
        assert results["plan.estimate"].product.diagnostics.identifiability \
            is None

    def test_noise_kind_none_is_refused(self):
        doc = document(ESTIMATE)
        doc["inference"]["noise"] = {"kind": "none"}
        doc["inference"]["observed"] = {"from": "simulation",
                                        "at": {"g": 1.5}}
        with pytest.raises(ConfigError, match="none"):
            run_document(doc)

    def test_expect_refuse_captures_the_packages_own_refusal(self):
        results = run_document(document(
            {**ESTIMATE, "expect": "refuse",
             "blocks": [{"names": ["g", "ghost"]}]}))
        assert results["plan.estimate"].product is None
        assert "ghost" in str(results["plan.estimate"].error)


class TestSample:
    def test_draws_come_back_with_their_diagnostics(self):
        results = run_document(document(SAMPLE))
        draws = results["plan.sample"].product
        assert draws.n_draw == 6
        assert float(draws.mean["g"]) == pytest.approx(1.5, abs=0.2)
        assert draws.diagnostics.rhat is not None

    def test_the_seed_is_required_by_name(self):
        run = {key: value for key, value in SAMPLE.items() if key != "seed"}
        with pytest.raises(ConfigError, match="seed"):
            run_document(document(run))

    def test_n_sweeps_is_required(self):
        run = {key: value for key, value in SAMPLE.items()
               if key != "n_sweeps"}
        with pytest.raises(ConfigError, match="n_sweeps"):
            run_document(document(run))

    def test_the_named_seed_decides_the_draws(self):
        # Same document twice: bitwise the same draws. A different
        # runtime.seeds.sample: different draws. Together these pin that
        # the key the sampler consumes is built from seed_for's answer,
        # not from a constant that happens to converge just as well.
        run = {**SAMPLE, "n_sweeps": 6, "warmup": 2}
        first = run_document(document(run))["plan.sample"].product
        again = run_document(document(run))["plan.sample"].product
        moved = run_document(
            document(run, seeds={"sample": 12}))["plan.sample"].product
        assert np.array_equal(first.samples["g"], again.samples["g"])
        assert not np.array_equal(first.samples["g"], moved.samples["g"])

    def test_check_identifiability_false_stores_no_report(self):
        results = run_document(document(SAMPLE))
        assert results["plan.sample"].product.diagnostics.identifiability \
            is None

    def test_warm_start_moves_only_the_named_inits(self):
        results = run_document(document(
            {**SAMPLE,
             "warm_start": {"kind": "plan.estimate",
                            "blocks": [{"names": ["g"]}],
                            "move": ["g"]}}))
        draws = results["plan.sample"].product
        assert float(draws.mean["g"]) == pytest.approx(1.5, abs=0.2)

    def test_warm_start_moves_the_chain_and_only_the_chain(self):
        # Same key, different start: a NUTS chain whose init the warm
        # start moved is a bitwise-different chain, where a conjugate
        # block would draw the same values from any init and the moved
        # init would be unobservable. The engines pin holds the main run
        # to its own blocks -- the warm start's conjugate estimate must
        # not leak its blocks into the sample.
        cold = run_document(document(GRADIENT_SAMPLE))["plan.sample"].product
        warm = run_document(document(WARMED))["plan.sample"].product
        assert warm.diagnostics.engines[("g",)] == "gradient"
        assert not np.array_equal(cold.samples["g"], warm.samples["g"])

    def test_warm_start_kind_is_plan_estimate_alone(self):
        with pytest.raises(ConfigError, match="plan.estimate"):
            run_document(document(
                {**SAMPLE, "warm_start": {"kind": "plan.sample",
                                          "blocks": [{"names": ["g"]}],
                                          "move": ["g"]}}))

    def test_warm_start_requires_move(self):
        with pytest.raises(ConfigError, match="move"):
            run_document(document(
                {**SAMPLE, "warm_start": {"kind": "plan.estimate",
                                          "blocks": [{"names": ["g"]}]}}))

    def test_move_must_name_declared_latents(self):
        with pytest.raises(ConfigError, match="ghost"):
            run_document(document(
                {**SAMPLE, "warm_start": {"kind": "plan.estimate",
                                          "blocks": [{"names": ["g"]}],
                                          "move": ["ghost"]}}))
