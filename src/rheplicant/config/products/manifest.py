"""Canonical manifest construction and detached-byte validation."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import ProductRequest

from .encoding import canonical_product_json, validate_relative_product_path
from .types import ProductBundle, ProductFile, ProductOmission

PRODUCT_SELECTORS = (
    "arrays",
    "aux",
    "taps",
    "assembly",
    "estimates",
    "parameters",
    "draws",
    "losses",
    "gradients",
    "covariance",
    "prediction_bands",
    "posterior_predictives",
    "identifiability",
    "scores",
    "recovery",
    "training_history",
    "timings",
    "refusals",
    "signal_paths",
    "compare",
    "benchmark",
    "chains",
    "report",
)
PRODUCT_FORMATS = ("npz", "json", "txt", "svg", "html", "mermaid", "netcdf")


def _text(value: object, *, where: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    if type(value) is not str or not value:
        raise ConfigError(f"{where} must be a non-empty string.")
    return value


def _request_record(request: ProductRequest) -> dict[str, object]:
    if type(request) is not ProductRequest or request.name not in PRODUCT_SELECTORS:
        raise ConfigError("product manifest contains an unknown request selector.")
    if request.format not in PRODUCT_FORMATS:
        raise ConfigError("product manifest contains an unknown request format.")
    if (
        type(request.runs) is not tuple
        or any(type(run) is not str or not run for run in request.runs)
        or len(request.runs) != len(set(request.runs))
    ):
        raise ConfigError("product request runs must be unique non-empty strings.")
    options: dict[str, object] = {}
    for key, value in request.options:
        if type(key) is not str or not key or key in options:
            raise ConfigError("product request options must have unique string keys.")
        options[key] = value
    return {
        "selector": request.name,
        "format": request.format,
        "runs": list(request.runs),
        "options": options,
    }


def _file_record(file: ProductFile, *, component_limit: int) -> dict[str, object]:
    if type(file) is not ProductFile:
        raise ConfigError("product files must be exact ProductFile records.")
    validate_relative_product_path(file.relative_path, component_limit=component_limit)
    if type(file.payload) is not bytes:
        raise ConfigError(f"product payload for {file.relative_path!r} must be immutable bytes.")
    if file.selector not in PRODUCT_SELECTORS:
        raise ConfigError(f"product file {file.relative_path!r} has an unknown selector.")
    if file.format not in PRODUCT_FORMATS:
        raise ConfigError(f"product file {file.relative_path!r} has an unknown format.")
    run = _text(file.run, where="product run", optional=True)
    kind = _text(file.kind, where="product kind", optional=True)
    if not isinstance(file.metadata, Mapping):
        raise ConfigError("product metadata must be a mapping.")
    record = {
        "relative_path": file.relative_path,
        "selector": file.selector,
        "run": run,
        "kind": kind,
        "format": file.format,
        "sha256": hashlib.sha256(file.payload).hexdigest(),
        "bytes": len(file.payload),
        "metadata": dict(file.metadata),
    }
    canonical_product_json(record)
    return record


def _omission_record(omission: ProductOmission) -> dict[str, object]:
    if type(omission) is not ProductOmission or omission.selector not in PRODUCT_SELECTORS:
        raise ConfigError("product omission has an unknown selector.")
    return {
        "selector": omission.selector,
        "run": _text(omission.run, where="omission run", optional=True),
        "kind": _text(omission.kind, where="omission kind", optional=True),
        "reason": _text(omission.reason, where="omission reason"),
    }


def build_product_manifest(
    files: Sequence[ProductFile],
    *,
    requests: Sequence[ProductRequest],
    omissions: Sequence[ProductOmission] = (),
    component_limit: int,
) -> ProductBundle:
    """Build a canonical manifest over already-detached product bytes."""
    frozen_files = tuple(files)
    paths = [file.relative_path for file in frozen_files]
    if len(paths) != len(set(paths)):
        raise ConfigError("duplicate product path in scientific product bundle.")
    tree = {
        "format_version": 1,
        "requests": [_request_record(request) for request in requests],
        "files": [
            _file_record(file, component_limit=component_limit) for file in frozen_files
        ],
        "omissions": [_omission_record(omission) for omission in omissions],
    }
    return ProductBundle(frozen_files, canonical_product_json(tree))


def validate_product_bundle(bundle: ProductBundle, *, component_limit: int) -> None:
    """Refuse a manifest that does not exactly describe its detached files."""
    if type(bundle) is not ProductBundle or type(bundle.manifest) is not bytes:
        raise ConfigError("scientific product bundle has an invalid record type.")
    try:
        value = json.loads(bundle.manifest)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ConfigError("scientific product manifest is not JSON.") from None
    if canonical_product_json(value) != bundle.manifest:
        raise ConfigError("scientific product manifest is not canonical JSON.")
    if type(value) is not dict or set(value) != {
        "format_version",
        "requests",
        "files",
        "omissions",
    }:
        raise ConfigError("scientific product manifest has an invalid root shape.")
    if (
        value["format_version"] != 1
        or type(value["requests"]) is not list
        or type(value["files"]) is not list
        or type(value["omissions"]) is not list
    ):
        raise ConfigError(
            "scientific product manifest has an invalid format version or table."
        )
    for row in value["requests"]:
        if type(row) is not dict or set(row) != {"selector", "format", "runs", "options"}:
            raise ConfigError("scientific product manifest has an invalid request row.")
        if type(row["runs"]) is not list or type(row["options"]) is not dict:
            raise ConfigError("scientific product manifest has an invalid request row.")
        request = ProductRequest(
            row["selector"],
            row["format"],
            tuple(row["runs"]),
            tuple(row["options"].items()),
        )
        if _request_record(request) != row:
            raise ConfigError("scientific product manifest has an invalid request row.")
    requested_selectors = {row["selector"] for row in value["requests"]}
    for row in value["omissions"]:
        if type(row) is not dict or set(row) != {"selector", "run", "kind", "reason"}:
            raise ConfigError("scientific product manifest has an invalid omission row.")
        omission = ProductOmission(row["selector"], row["run"], row["kind"], row["reason"])
        if _omission_record(omission) != row:
            raise ConfigError("scientific product manifest has an invalid omission row.")
    expected = [
        _file_record(file, component_limit=component_limit) for file in bundle.files
    ]
    if value["files"] != expected:
        raise ConfigError("scientific product manifest disagrees with product payloads.")
    if any(row["selector"] not in requested_selectors for row in value["files"]):
        raise ConfigError("scientific product manifest contains an unrequested file.")
    if any(row["selector"] not in requested_selectors for row in value["omissions"]):
        raise ConfigError("scientific product manifest contains an unrequested omission.")
    paths = [file.relative_path for file in bundle.files]
    if len(paths) != len(set(paths)):
        raise ConfigError("duplicate product path in scientific product bundle.")


__all__ = [
    "PRODUCT_FORMATS",
    "PRODUCT_SELECTORS",
    "build_product_manifest",
    "validate_product_bundle",
]
