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
