from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from _rheplicant_bootstrap.errors import ConfigError
from rheplicant.config.products.extractors import (
    EXTRACTOR_REGISTRY,
    RUN_KIND_SELECTORS,
    extract_run_payload,
    numeric_leaves,
)

RUN_KINDS = (
    "forward",
    "fisher",
    "optimize",
    "plan.estimate",
    "plan.sample",
    "conjugate.wiener",
    "conjugate.gcr",
    "conjugate.gls",
    "condition",
    "identifiability",
    "score_directions",
    "gradient",
    "mmodes",
    "predict",
    "nuts",
    "npe",
    "compare",
    "benchmark",
)


def configured(observed=None, truth=None):
    inference = SimpleNamespace(
        observed=(None if observed is None else SimpleNamespace(entries=observed)),
        truth=truth or {},
        truth_omitted={},
    )
    return SimpleNamespace(inference=inference)


def test_registry_covers_every_declared_predecessor_kind_and_selector():
    assert tuple(RUN_KIND_SELECTORS) == RUN_KINDS
    assert set(EXTRACTOR_REGISTRY) == {
        (kind, selector)
        for kind, selectors in RUN_KIND_SELECTORS.items()
        for selector in selectors
    }
    assert all(selectors for selectors in RUN_KIND_SELECTORS.values())


def test_numeric_flattener_is_stable_structural_and_refuses_objects():
    value = {"z/name": [np.array([3]), 4.0], "a": np.array([1 + 2j])}
    leaves = numeric_leaves(value)
    assert tuple(leaves) == (
        "mapping/n-61",
        "mapping/n-7a2f6e616d65/i-0",
        "mapping/n-7a2f6e616d65/i-1",
    )
    np.testing.assert_array_equal(leaves["mapping/n-61"], np.array([1 + 2j]))
    with pytest.raises(ConfigError, match="numeric"):
        numeric_leaves({"bad": object()})


def test_forward_arrays_aux_and_taps_use_the_recorded_state_only():
    state = SimpleNamespace(
        data=np.array([1.0, 2.0]),
        aux={
            "weights": np.array([3.0]),
            "snapshot/early": np.array([4.0]),
            "snapshot/late": np.array([5.0]),
        },
    )
    built = configured(observed={"primary": np.array([0.5, 1.5])})
    arrays = extract_run_payload("forward", "arrays", state, built)
    assert arrays.encoding == "npz"
    assert set(arrays.value) == {"predicted", "observed/n-7072696d617279"}
    aux = extract_run_payload(
        "forward", "aux", state, built, options=(("keys", ("weights",)),)
    )
    assert set(aux.value) == {"n-77656967687473"}
    taps = extract_run_payload(
        "forward", "taps", state, built, options=(("keys", ("early", "late")),)
    )
    assert set(taps.value) == {"n-6561726c79", "n-6c617465"}
    with pytest.raises(ConfigError, match="missing"):
        extract_run_payload(
            "forward", "taps", state, built, options=(("keys", ("never",)),)
        )


def test_semantic_extractors_do_not_guess_mapping_keys():
    fit = extract_run_payload(
        "optimize",
        "parameters",
        {"params": {"gain": np.array(2.0)}, "losses": np.array([4.0, 1.0])},
        configured(),
    )
    assert set(fit.value) == {"mapping/n-6761696e"}
    losses = extract_run_payload(
        "optimize",
        "losses",
        {"params": {}, "losses": np.array([4.0, 1.0])},
        configured(),
    )
    np.testing.assert_array_equal(losses.value["loss"], np.array([4.0, 1.0]))
    with pytest.raises(ConfigError, match="not compatible"):
        extract_run_payload("gradient", "losses", {"losses": np.array([1])}, configured())
    gradient = extract_run_payload(
        "gradient", "gradients", {"twin.gain": np.array([3.0])}, configured()
    )
    assert tuple(gradient.value) == ("mapping/n-7477696e2e6761696e",)


def test_fisher_covariance_serializes_the_matrix_not_its_live_treedef():
    product = {
        "fisher": SimpleNamespace(matrix=np.eye(2), structure=object(), kind="fisher"),
        "covariance": SimpleNamespace(
            matrix=np.eye(2) * 2,
            structure=object(),
            kind="covariance",
        ),
    }
    extracted = extract_run_payload("fisher", "covariance", product, configured())
    assert tuple(extracted.value) == ("mapping/n-636f76617269616e6365",)
    np.testing.assert_array_equal(
        extracted.value["mapping/n-636f76617269616e6365"],
        np.eye(2) * 2,
    )


@pytest.mark.parametrize("kind", ("plan.sample", "nuts", "npe"))
def test_posterior_extractors_preserve_draws_and_compute_parameters(kind):
    product = SimpleNamespace(
        samples={"gain": np.array([[1.0], [3.0]])},
        n_draw=2,
        train_loss=np.array([2.0, 1.0]),
        validation_loss=np.array([2.5, 1.5]),
        best_step=2,
    )
    draws = extract_run_payload(kind, "draws", product, configured())
    mean = extract_run_payload(kind, "parameters", product, configured())
    np.testing.assert_array_equal(draws.value["mapping/n-6761696e"], [[1.0], [3.0]])
    np.testing.assert_array_equal(mean.value["mapping/n-6761696e"], [2.0])


def test_identifiability_record_is_json_and_arrays_are_explicit_lists():
    report = SimpleNamespace(
        names=("gain",),
        shapes=((1,),),
        spans=((0, 1),),
        n_par=1,
        n_data=2,
        rank=1,
        nullity=0,
        singular_values=np.array([2.0]),
        null_space=np.empty((0, 1)),
        jacobian=np.array([[1.0], [1.0]]),
        column_norms=np.array([2.0]),
        rtol=1e-8,
        threshold=2e-8,
        weakest_identified=1.0,
    )
    extracted = extract_run_payload(
        "identifiability", "identifiability", report, configured()
    )
    assert extracted.encoding == "json"
    assert extracted.value["singular_values"] == [2.0]
    assert extracted.value["rank"] == 1


# --- run_diagnostics: the quality signals, published rather than in-memory ---


def test_every_kind_declaring_run_diagnostics_has_an_extractor():
    # One generic extractor serves the family, so this guard is what catches a
    # kind added to the selector table with no row behind it.
    declared = {
        kind
        for kind, selectors in RUN_KIND_SELECTORS.items()
        if "run_diagnostics" in selectors
    }
    assert declared
    for kind in declared:
        assert (kind, "run_diagnostics") in EXTRACTOR_REGISTRY


def test_run_diagnostics_lifts_the_vocabulary_a_product_carries():
    product = SimpleNamespace(divergences=3, n_draw=200, n_chain=4, samples={"a": [1.0]})
    extracted = EXTRACTOR_REGISTRY[("nuts", "run_diagnostics")](product, None, {})
    assert extracted.encoding == "json"
    # `samples` is bulk data claimed by another selector, not a quality signal.
    assert extracted.value == {"divergences": 3, "n_draw": 200, "n_chain": 4}


def test_run_diagnostics_keeps_nuts_signals_per_latent():
    product = SimpleNamespace(
        divergences=0,
        diagnostics={"depth": {"r_hat": 1.01, "n_eff": 180.0}},
    )
    extracted = EXTRACTOR_REGISTRY[("nuts", "run_diagnostics")](product, None, {})
    assert extracted.value["per_latent"] == {"depth": {"r_hat": 1.01, "n_eff": 180.0}}


def test_run_diagnostics_reads_a_nested_record_field_by_field():
    # plan.* nest a PlanDiagnostics record rather than a mapping of latents.
    nested = SimpleNamespace(chi2=[9.0, 4.0], converged=True, rhat=1.002, sweeps=12)
    product = SimpleNamespace(diagnostics=nested)
    extracted = EXTRACTOR_REGISTRY[("plan.sample", "run_diagnostics")](product, None, {})
    assert extracted.value == {
        "converged": True,
        "rhat": 1.002,
        "chi2": [9.0, 4.0],
        "sweeps": 12,
    }


@pytest.mark.parametrize("bad", (float("nan"), float("inf"), float("-inf")))
def test_a_non_finite_diagnostic_is_recorded_as_null_not_refused(bad):
    # Regression: numpyro reports r_hat/n_eff as NaN for a chain that
    # degenerated, and a run diverging on every transition is exactly the run
    # whose diagnostics someone needs. Refusing would publish nothing for the
    # worst runs. Measured on a real 50/50-divergent NUTS run.
    product = SimpleNamespace(divergences=50, diagnostics={"depth": {"r_hat": bad}})
    extracted = EXTRACTOR_REGISTRY[("nuts", "run_diagnostics")](product, None, {})
    assert extracted.value["per_latent"]["depth"]["r_hat"] is None
    assert extracted.value["divergences"] == 50


def test_a_product_with_no_quality_signals_refuses_so_the_run_is_omitted():
    # The bundle turns this into a recorded omission for that run, not a
    # failure: a forward simulation having no diagnostics is not an error.
    with pytest.raises(ConfigError, match="no diagnostic fields"):
        EXTRACTOR_REGISTRY[("nuts", "run_diagnostics")](
            SimpleNamespace(samples={"a": [1.0]}), None, {}
        )


def test_condition_publishes_its_conditioning_number():
    # `condition`'s product IS the number, not a record carrying one.
    extracted = EXTRACTOR_REGISTRY[("condition", "run_diagnostics")](
        np.float32(1234.5), None, {}
    )
    assert extracted.encoding == "json"
    assert extracted.value["kappa"] == pytest.approx(1234.5)


def test_prediction_bands_reduce_across_the_DRAW_axis_and_carry_a_median():
    """`predict`'s bands are a reduction OF pushforwards, and `median` is one.

    The draw axis is leading, so every statistic is `axis=0`. Checked with a
    stack whose mean and median DIFFER per pixel, because a symmetric fixture
    would pass whichever axis the extractor happened to reduce over and
    whichever central tendency it happened to compute.
    """
    # Three draws over a (2,) grid. Column 0: (1, 2, 9) -> mean 4, median 2.
    # Column 1: (10, 30, 35) -> mean 25, median 30. Neither column's mean
    # equals its median, and the two columns disagree about which is larger.
    prediction = np.array([[1.0, 10.0], [2.0, 30.0], [9.0, 35.0]])
    # The product IS the array: `predict`'s samples route returns
    # `predict_from_samples`' stack verbatim, so there is no wrapper to unwrap.
    bands = extract_run_payload("predict", "prediction_bands", prediction, configured())

    assert bands.encoding == "npz"
    assert set(bands.value) == {"mean", "median", "std", "q025", "q975"}
    np.testing.assert_allclose(bands.value["mean"], np.array([4.0, 25.0]))
    np.testing.assert_allclose(bands.value["median"], np.array([2.0, 30.0]))
    # The grid shape survives and the draw axis is gone.
    assert bands.value["median"].shape == (2,)


def test_prediction_bands_refuse_a_prediction_with_no_draw_axis():
    with pytest.raises(ConfigError, match="leading draw axis"):
        extract_run_payload("predict", "prediction_bands", np.float64(1.0), configured())
