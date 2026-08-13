"""runs: the exit list -- grammar, variants, expect: refuse, and forward."""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.runs import RunSpec, parse_runs, run_document
from tests.config.test_config_document import synthetic_document


def document(*runs, **extra):
    doc = {**synthetic_document(), **extra}
    doc["runs"] = list(runs) if runs else [{"kind": "forward"}]
    return doc


class TestGrammar:
    def test_a_single_mapping_is_one_run_named_by_its_kind(self):
        runs = parse_runs({"kind": "forward"})
        assert runs == (RunSpec(name="forward", kind="forward", variant=None,
                                on="primary", expect="ok", options={}),)

    def test_names_are_required_and_unique_when_there_are_several(self):
        with pytest.raises(ConfigError, match="name"):
            parse_runs([{"kind": "forward"}, {"kind": "forward"}])
        with pytest.raises(ConfigError, match="twice"):
            parse_runs([{"name": "a", "kind": "forward"},
                        {"name": "a", "kind": "forward"}])

    def test_kind_is_required_and_the_table_is_closed(self):
        with pytest.raises(ConfigError, match="kind"):
            parse_runs([{"name": "a"}])
        with pytest.raises(ConfigError, match="plan.estimate"):
            parse_runs([{"kind": "anneal"}])

    def test_the_still_deferred_kinds_are_refused_by_name(self):
        for kind in ("conjugate.wiener", "conjugate.gcr", "conjugate.gls",
                     "gradient", "identifiability", "score_directions",
                     "condition", "mmodes", "predict"):
            with pytest.raises(ConfigError, match="2C"):
                parse_runs([{"kind": kind}])

    def test_nuts_and_npe_are_plan_2d(self):
        for kind in ("nuts", "npe"):
            with pytest.raises(ConfigError, match="2D"):
                parse_runs([{"kind": kind}])

    def test_compare_and_benchmark_are_plan_4(self):
        for kind in ("compare", "benchmark"):
            with pytest.raises(ConfigError, match="Plan 4"):
                parse_runs([{"kind": kind}])

    def test_reuse_is_a_name_on_the_spec(self):
        (run,) = parse_runs([{"kind": "forward", "reuse": "earlier"}])
        assert run.reuse == "earlier"

    def test_expect_is_ok_or_refuse(self):
        with pytest.raises(ConfigError, match="refuse"):
            parse_runs([{"kind": "forward", "expect": "fail"}])

    def test_an_empty_list_declares_no_exit_and_is_refused(self):
        with pytest.raises(ConfigError, match="list of exits"):
            parse_runs([])

    def test_kind_specific_keys_travel_in_options(self):
        (run,) = parse_runs([{"kind": "plan.estimate",
                              "blocks": [{"names": ["g"]}], "tol": 1e-3}])
        assert run.options == {"blocks": [{"names": ["g"]}], "tol": 1e-3}


class TestRunDocument:
    def test_forward_evaluates_the_twin_on_the_state(self):
        results = run_document(document())
        assert set(results) == {"forward"}
        assert results["forward"].product.data.shape == (16, 8)
        assert results["forward"].error is None

    def test_a_variant_run_builds_its_own_configured_run(self):
        results = run_document(document(
            {"name": "base", "kind": "forward"},
            {"name": "unity", "kind": "forward", "variant": "unity_gain"}))
        import jax.numpy as jnp

        assert not jnp.allclose(results["base"].product.data,
                                results["unity"].product.data)

    def test_results_arrive_in_declaration_order(self):
        results = run_document(document(
            {"name": "z_first", "kind": "forward"},
            {"name": "a_second", "kind": "forward"}))
        assert list(results) == ["z_first", "a_second"]

    def test_forward_takes_no_kind_specific_keys(self):
        with pytest.raises(ConfigError, match="n_steps"):
            run_document(document({"kind": "forward", "n_steps": 3}))

    def test_expect_refuse_captures_the_refusal_as_the_product(self):
        doc = document({"kind": "forward", "expect": "refuse"})
        doc["observation"] = {**doc["observation"],
                              "data": {"ones": ["n_time", "n_freq"]}}
        results = run_document(doc)
        assert results["forward"].product is None
        assert "data" in str(results["forward"].error)

    def test_expect_refuse_that_succeeds_is_the_failure(self):
        with pytest.raises(ConfigError, match="SUCCEEDED"):
            run_document(document({"kind": "forward", "expect": "refuse"}))


class TestRequiredness:
    def test_a_document_without_runs_is_refused(self):
        from rheplicant.config.document import load_document

        doc = synthetic_document()
        del doc["runs"]
        with pytest.raises(ConfigError, match="runs"):
            load_document(doc)

    def test_runs_declared_null_is_refused_with_the_grammar_error(self):
        doc = synthetic_document()
        doc["runs"] = None
        with pytest.raises(ConfigError, match="list of exits"):
            run_document(doc)
