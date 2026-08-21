from __future__ import annotations

import dataclasses
import hashlib
import json

import jsonschema
import pytest

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import ProductRequest
from rheplicant.config.products import (
    PRODUCT_FORMATS,
    PRODUCT_SELECTORS,
    ProductFile,
    ProductOmission,
    build_product_manifest,
    validate_product_bundle,
)
from rheplicant.config.products.encoding import canonical_product_json
from rheplicant.config.schemas import load_schema


def product_file(path="runs/n-666974/estimates.npz", payload=b"science"):
    return ProductFile(
        relative_path=path,
        payload=payload,
        selector="estimates",
        run="fit",
        kind="optimize",
        format="npz",
        metadata={"mean": {"dtype": "float64", "shape": [2]}},
    )


def test_manifest_is_canonical_hashed_and_matches_packaged_schema():
    file = product_file()
    request = ProductRequest("estimates", "npz", ("fit",), ())
    omission = ProductOmission("estimates", "forward", "forward", "not compatible")
    bundle = build_product_manifest(
        (file,),
        requests=(request,),
        omissions=(omission,),
        component_limit=255,
    )
    value = json.loads(bundle.manifest)
    jsonschema.validate(value, load_schema("products-v1"))
    assert value["format_version"] == 1
    assert value["requests"] == [
        {"format": "npz", "options": {}, "runs": ["fit"], "selector": "estimates"}
    ]
    assert value["files"][0]["sha256"] == hashlib.sha256(file.payload).hexdigest()
    assert value["files"][0]["bytes"] == len(file.payload)
    assert value["omissions"] == [
        {
            "kind": "forward",
            "reason": "not compatible",
            "run": "forward",
            "selector": "estimates",
        }
    ]
    assert bundle.files == (file,)
    validate_product_bundle(bundle, component_limit=255)


def test_manifest_refuses_duplicate_paths_and_disagreeing_payloads():
    first = product_file()
    with pytest.raises(ConfigError, match="duplicate product path"):
        build_product_manifest(
            (first, dataclasses.replace(first, payload=b"other")),
            requests=(ProductRequest("estimates", "npz", (), ()),),
            component_limit=255,
        )
    bundle = build_product_manifest(
        (first,),
        requests=(ProductRequest("estimates", "npz", (), ()),),
        component_limit=255,
    )
    with pytest.raises(ConfigError, match="manifest"):
        validate_product_bundle(
            dataclasses.replace(bundle, files=(dataclasses.replace(first, payload=b"forged"),)),
            component_limit=255,
        )
    forged = json.loads(bundle.manifest)
    forged["requests"][0]["selector"] = "mystery"
    with pytest.raises(ConfigError, match="request"):
        validate_product_bundle(
            dataclasses.replace(bundle, manifest=canonical_product_json(forged)),
            component_limit=255,
        )


def test_manifest_refuses_unknown_selector_format_and_non_json_metadata():
    request = ProductRequest("estimates", "npz", (), ())
    for bad in (
        dataclasses.replace(product_file(), selector="mystery"),
        dataclasses.replace(product_file(), format="pickle"),
        dataclasses.replace(product_file(), metadata={"bad": float("inf")}),
    ):
        with pytest.raises(ConfigError):
            build_product_manifest((bad,), requests=(request,), component_limit=255)


def test_manifest_vocabularies_match_the_live_contract():
    schema = load_schema("products-v1")
    jsonschema.Draft202012Validator.check_schema(schema)
    assert tuple(schema["$defs"]["selector"]["enum"]) == PRODUCT_SELECTORS
    assert tuple(schema["$defs"]["format"]["enum"]) == PRODUCT_FORMATS


def test_product_record_is_frozen_and_payload_is_bytes():
    file = product_file()
    with pytest.raises(dataclasses.FrozenInstanceError):
        file.selector = "arrays"  # type: ignore[misc]
    with pytest.raises(ConfigError, match="payload"):
        build_product_manifest(
            (dataclasses.replace(file, payload=bytearray(b"mutable")),),  # type: ignore[arg-type]
            requests=(ProductRequest("estimates", "npz", (), ()),),
            component_limit=255,
        )
