"""Assemble every requested scientific file into one detached bundle."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from _rheplicant_bootstrap.audit.names import encode_name
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import ProductRequest, ReportRequest

from .encoding import (
    canonical_product_json,
    deterministic_netcdf,
    deterministic_npz,
    validate_relative_product_path,
)
from .extractors import ExtractedProduct, extract_run_payload
from .manifest import build_product_manifest, validate_product_bundle
from .render import materialize_layer_product
from .report import materialize_report
from .types import ProductBundle, ProductFile, ProductOmission

_EXTENSIONS = {"npz": "npz", "json": "json", "txt": "txt", "netcdf": "nc"}
_LAYER_SELECTORS = ("assembly", "signal_paths")
_COMMON_SELECTORS = ("timings", "refusals")


def _configured_for(execution: object, row: object) -> object:
    matches = [
        layer.configured
        for layer in execution.prepared.layers
        if layer.layer.identity == row.parsed.layer.identity
    ]
    if len(matches) != 1:
        raise ConfigError(
            f"scientific product run {row.parsed.name!r} has no unique prepared layer."
        )
    return matches[0]


def _run_path(row: object, selector: str, extension: str) -> str:
    return f"runs/{encode_name(row.parsed.name)}/{selector}.{extension}"


def _timing(row: object) -> ExtractedProduct:
    return ExtractedProduct(
        "json",
        {
            "wall_time_ns": row.wall_time_ns,
            "seconds": row.wall_time_ns / 1_000_000_000,
        },
    )


def _refusal(row: object) -> ExtractedProduct:
    if not row.captured_expected_refusal or row.result is None or row.result.error is None:
        raise ConfigError(
            f"outputs.write.refusals: run {row.parsed.name!r} has no captured refusal."
        )
    error = row.result.error
    payload = (
        f"exception_type: {type(error).__module__}.{type(error).__qualname__}\n"
        f"message: {error}\n"
    )
    return ExtractedProduct("txt", payload)


def _reused_kind(execution: object, row: object) -> str | None:
    reused = getattr(row.parsed, "reuse", None)
    if reused is None:
        return None
    for earlier in execution.runs:
        if earlier.parsed.name == reused:
            return earlier.parsed.kind
    raise ConfigError(
        f"run {row.parsed.name!r} reuses missing execution result {reused!r}."
    )


def _extract(
    execution: object,
    row: object,
    request: ProductRequest,
    configured: object,
) -> ExtractedProduct:
    if request.name == "timings":
        return _timing(row)
    if request.name == "refusals":
        return _refusal(row)
    if row.result is None or row.result.error is not None:
        raise ConfigError(
            f"outputs.write.{request.name}: run {row.parsed.name!r} has no scientific product."
        )
    if row.parsed.kind == "predict":
        source_kind = _reused_kind(execution, row)
        if request.name == "posterior_predictives" and source_kind == "fisher":
            raise ConfigError(
                "a predict run reusing fisher returns delta-method standard deviation, "
                "not a posterior predictive."
            )
        if request.name == "prediction_bands" and source_kind == "fisher":
            return ExtractedProduct("npz", {"std": row.result.product})
    return extract_run_payload(
        row.parsed.kind,
        request.name,
        row.result.product,
        configured,
        options=request.options,
    )


def _materialize_run(
    execution: object,
    row: object,
    request: ProductRequest,
    configured: object,
    *,
    component_limit: int,
) -> tuple[ProductFile, ...]:
    extracted = _extract(execution, row, request, configured)
    metadata = dict(extracted.metadata)
    if request.name == "taps":
        if request.format != "npz" or not isinstance(extracted.value, Mapping):
            raise ConfigError("outputs.write.taps: snapshots require named NPZ arrays.")
        files: list[ProductFile] = []
        for name, value in extracted.value.items():
            payload, array_metadata = deterministic_npz({"value": value})
            path = (
                f"runs/{encode_name(row.parsed.name)}/taps/{name}.npz"
            )
            validate_relative_product_path(path, component_limit=component_limit)
            files.append(
                ProductFile(
                    path,
                    payload,
                    "taps",
                    row.parsed.name,
                    row.parsed.kind,
                    "npz",
                    {"snapshot": name, "arrays": array_metadata},
                )
            )
        return tuple(files)
    if request.name == "chains" and request.format == "netcdf":
        if not isinstance(extracted.value, Mapping):
            raise ConfigError("NetCDF chains require a mapping of named arrays.")
        payload, array_metadata = deterministic_netcdf(extracted.value)
        actual_format = "netcdf"
        metadata["arrays"] = array_metadata
    elif extracted.encoding == "npz":
        if request.format != "npz":
            raise ConfigError(
                f"outputs.write.{request.name}.format: {request.format!r} "
                "does not match its numeric product."
            )
        if not isinstance(extracted.value, Mapping):
            raise ConfigError("NPZ scientific products require named arrays.")
        payload, array_metadata = deterministic_npz(extracted.value)
        actual_format = "npz"
        metadata["arrays"] = array_metadata
    elif extracted.encoding == "json":
        if request.format != "json":
            raise ConfigError(
                f"outputs.write.{request.name}.format: {request.format!r} "
                "does not match its record product."
            )
        payload = canonical_product_json(extracted.value)
        actual_format = "json"
    else:
        if request.format != "txt" or type(extracted.value) is not str:
            raise ConfigError(
                f"outputs.write.{request.name}.format: must be 'txt' for text products."
            )
        payload = extracted.value.encode("utf-8")
        actual_format = "txt"
    path = _run_path(row, request.name, _EXTENSIONS[actual_format])
    validate_relative_product_path(path, component_limit=component_limit)
    return (
        ProductFile(
            path,
            payload,
            request.name,
            row.parsed.name,
            row.parsed.kind,
            actual_format,
            metadata,
        ),
    )


def _selected_rows(execution: object, request: ProductRequest) -> tuple[object, ...]:
    rows_by_name = {row.parsed.name: row for row in execution.runs}
    missing = [name for name in request.runs if name not in rows_by_name]
    if missing:
        raise ConfigError(f"outputs.write.{request.name}.runs: missing executed runs {missing}.")
    if request.runs:
        selected = set(request.runs)
        return tuple(row for row in execution.runs if row.parsed.name in selected)
    return tuple(execution.runs)


def _report_requests(report: ReportRequest) -> tuple[ProductRequest, ...]:
    options = (
        ("columns", report.columns),
        ("reference", report.reference),
        ("relative", report.relative),
    )
    return tuple(
        ProductRequest(
            "report",
            "txt" if format_ == "text" else "json",
            report.rows,
            options,
        )
        for format_ in report.formats
    )


def build_product_bundle(
    execution: object,
    *,
    requests: Sequence[ProductRequest],
    report: ReportRequest | None,
    component_limit: int,
) -> ProductBundle:
    """Materialize requested products without touching the destination tree."""
    if execution.status != "ok":
        raise ConfigError("scientific products require a successful execution record.")
    if not requests and report is None:
        raise ConfigError("scientific product bundle requires a product or report request.")
    files: list[ProductFile] = []
    omissions: list[ProductOmission] = []
    manifest_requests: list[ProductRequest] = list(requests)
    for request in requests:
        if type(request) is not ProductRequest:
            raise ConfigError("scientific product requests must be exact ProductRequest records.")
        if request.name in _LAYER_SELECTORS:
            if request.runs:
                raise ConfigError(
                    f"outputs.write.{request.name}.runs: layer products cannot filter runs."
                )
            for layer in execution.prepared.layers:
                files.extend(
                    materialize_layer_product(
                        layer,
                        request,
                        component_limit=component_limit,
                    )
                )
            continue
        selected = _selected_rows(execution, request)
        emitted = 0
        for row in selected:
            try:
                produced = _materialize_run(
                    execution,
                    row,
                    request,
                    _configured_for(execution, row),
                    component_limit=component_limit,
                )
            except ConfigError as error:
                if request.runs:
                    raise ConfigError(
                        f"outputs.write.{request.name}: run {row.parsed.name!r} "
                        f"(kind: {row.parsed.kind}) cannot produce the requested product: {error}"
                    ) from None
                omissions.append(
                    ProductOmission(
                        request.name,
                        row.parsed.name,
                        row.parsed.kind,
                        str(error),
                    )
                )
                continue
            files.extend(produced)
            emitted += len(produced)
        if emitted == 0 and not request.optional:
            # An invocation request is exempt: a caller asking to keep whatever
            # these runs produce has made no mistake when one of them produces
            # nothing, and every skipped run is already an omission above.
            raise ConfigError(
                f"outputs.write.{request.name}: no executed run can produce this product."
            )
    if report is not None:
        files.extend(materialize_report(execution, report, component_limit=component_limit))
        manifest_requests.extend(_report_requests(report))
    bundle = build_product_manifest(
        tuple(files),
        requests=tuple(manifest_requests),
        omissions=tuple(omissions),
        component_limit=component_limit,
    )
    validate_product_bundle(bundle, component_limit=component_limit)
    return bundle


__all__ = ["build_product_bundle"]
