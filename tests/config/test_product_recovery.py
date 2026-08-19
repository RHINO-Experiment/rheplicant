from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from rheplicant.config.products.recovery import recovery_record


def inference(truth, omitted=None):
    return SimpleNamespace(truth=truth, truth_omitted=omitted or {})


def test_recovery_uses_posterior_moments_without_nonfinite_pulls():
    product = SimpleNamespace(
        samples={
            "gain": np.array([1.0, 3.0]),
            "offset": np.array([4.0, 4.0]),
        }
    )
    record = recovery_record(
        "plan.sample",
        product,
        inference({"gain": np.array(2.5), "offset": np.array(5.0)}),
    )
    by_name = {row["name"]: row for row in record["latents"]}
    assert by_name["gain"] == {
        "name": "gain",
        "truth": 2.5,
        "centre": 2.0,
        "sigma": 1.0,
        "absolute_error": 0.5,
        "pull": -0.5,
    }
    assert by_name["offset"]["sigma"] == 0.0
    assert "pull" not in by_name["offset"]
    assert record["omissions"] == [
        {"name": "offset", "reason": "posterior sigma is zero; pull is undefined"}
    ]


def test_recovery_records_missing_truth_and_declared_truth_omissions():
    product = SimpleNamespace(samples={"gain": np.array([1.0, 2.0])})
    record = recovery_record(
        "nuts",
        product,
        inference({}, {"gain": "bound through inference.bindings"}),
    )
    assert record["latents"] == []
    assert record["omissions"] == [
        {"name": "gain", "reason": "bound through inference.bindings"}
    ]


def test_point_recovery_has_no_invented_sigma():
    product = SimpleNamespace(values={"gain": np.array(2.0)})
    record = recovery_record("plan.estimate", product, inference({"gain": np.array(1.5)}))
    assert record["latents"] == [
        {"name": "gain", "truth": 1.5, "centre": 2.0, "absolute_error": 0.5}
    ]
