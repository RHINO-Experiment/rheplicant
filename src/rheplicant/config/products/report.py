"""Deterministic report tables over already-executed run products."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import ReportRequest

from .encoding import canonical_product_json, validate_relative_product_path
from .extractors import numeric_leaves
from .types import ProductFile


def _real_json(value: object, *, where: str) -> object:
    array = np.asarray(value)
    if array.dtype.hasobject or array.dtype.kind not in "biuf":
        raise ConfigError(f"{where} must be a real numeric value.")
    if not np.all(np.isfinite(array)):
        raise ConfigError(f"{where} must be finite.")
    return array.item() if array.ndim == 0 else array.tolist()


def _numeric_record(value: object, *, where: str) -> dict[str, object]:
    return {
        name: _real_json(array, where=f"{where}.{name}")
        for name, array in numeric_leaves(value).items()
    }


def _mean_std(kind: str, product: object) -> tuple[object | None, object | None]:
    if kind == "optimize":
        return product["params"], None
    if kind == "plan.estimate":
        return product.values, None
    if kind in ("plan.sample", "nuts", "npe"):
        means = {
            name: np.mean(np.asarray(stack), axis=0)
            for name, stack in product.samples.items()
        }
        stds = {name: np.std(np.asarray(stack), axis=0) for name, stack in product.samples.items()}
        return means, stds
    if kind == "conjugate.wiener":
        mean = product["mean"]
        covariance = product.get("covariance")
        if covariance is None:
            return mean, None
        diagonal = np.sqrt(np.diag(np.asarray(covariance.matrix)))
        stds: dict[str, object] = {}
        offset = 0
        for name, value in mean.items():
            array = np.asarray(value)
            stds[name] = diagonal[offset:offset + array.size].reshape(array.shape)
            offset += array.size
        if offset != diagonal.size:
            raise ConfigError("report covariance does not match conjugate.wiener mean.")
        return mean, stds
    if kind == "conjugate.gls":
        return product.solution, None
    return None, None


def _statistics(row: object) -> dict[str, object]:
    if row.result is None or row.result.error is not None:
        raise ConfigError(f"report row {row.parsed.name!r} has no successful product.")
    mean, std = _mean_std(row.parsed.kind, row.result.product)
    result: dict[str, object] = {"seconds": row.wall_time_ns / 1_000_000_000}
    if mean is not None:
        result["mean"] = _numeric_record(mean, where=f"report {row.parsed.name} mean")
    if std is not None:
        result["std"] = _numeric_record(std, where=f"report {row.parsed.name} std")
    return result


def _numeric_mapping(value: object, *, where: str) -> dict[str, np.ndarray]:
    if type(value) is not dict:
        raise ConfigError(f"{where} is unavailable.")
    return {name: np.asarray(item) for name, item in value.items()}


def _relative(
    metric: str,
    row: Mapping[str, object],
    reference: Mapping[str, object],
    *,
    row_name: str,
) -> dict[str, object]:
    row_std = _numeric_mapping(row.get("std"), where=f"report row {row_name!r} std")
    ref_std = _numeric_mapping(reference.get("std"), where="report reference std")
    if set(row_std) != set(ref_std):
        raise ConfigError(f"report relative {metric} std structures disagree.")
    result: dict[str, object] = {}
    if metric == "width_ratio":
        for name in row_std:
            if np.any(ref_std[name] == 0):
                raise ConfigError("report width_ratio reference width is zero.")
            result[name] = _real_json(
                row_std[name] / ref_std[name],
                where=f"report width_ratio.{name}",
            )
        return result
    row_mean = _numeric_mapping(row.get("mean"), where=f"report row {row_name!r} mean")
    ref_mean = _numeric_mapping(reference.get("mean"), where="report reference mean")
    if set(row_mean) != set(ref_mean) or set(row_mean) != set(row_std):
        raise ConfigError("report relative mean_sigma structures disagree.")
    for name in row_mean:
        denominator = np.sqrt(np.square(row_std[name]) + np.square(ref_std[name]))
        if np.any(denominator == 0):
            raise ConfigError("report mean_sigma combined width is zero.")
        result[name] = _real_json(
            (row_mean[name] - ref_mean[name]) / denominator,
            where=f"report mean_sigma.{name}",
        )
    return result


def _text_cell(value: object) -> str:
    if type(value) is float:
        return format(value, ".9g")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def materialize_report(
    execution: object,
    request: ReportRequest,
    *,
    component_limit: int,
) -> tuple[ProductFile, ...]:
    """Materialize JSON/text reports without executing any product."""
    rows_by_name = {row.parsed.name: row for row in execution.runs}
    missing = [name for name in request.rows if name not in rows_by_name]
    if missing:
        raise ConfigError(f"outputs.report.rows: missing executed runs {missing}.")
    computed = {name: _statistics(rows_by_name[name]) for name in request.rows}
    for name, statistics in computed.items():
        absent = [column for column in request.columns if column not in statistics]
        if absent:
            raise ConfigError(
                f"outputs.report.columns: row {name!r} has no statistics {absent}."
            )
    reference = None if request.reference is None else computed[request.reference]
    output_rows: list[dict[str, object]] = []
    for name in request.rows:
        statistics = {column: computed[name][column] for column in request.columns}
        relative: dict[str, object] = {}
        if reference is not None:
            for metric in request.relative:
                relative[metric] = _relative(
                    metric,
                    computed[name],
                    reference,
                    row_name=name,
                )
        output_rows.append(
            {
                "name": name,
                "kind": rows_by_name[name].parsed.kind,
                "statistics": statistics,
                "relative": relative,
            }
        )
    tree = {
        "columns": list(request.columns),
        "reference": request.reference,
        "relative": list(request.relative),
        "rows": output_rows,
    }
    files: list[ProductFile] = []
    for format_ in request.formats:
        if format_ == "json":
            path = "report.json"
            payload = canonical_product_json(tree)
            product_format = "json"
        else:
            path = "report.txt"
            headings = ("run", "kind", *request.columns, *request.relative)
            lines = ["\t".join(headings)]
            for output in output_rows:
                cells = [output["name"], output["kind"]]
                cells.extend(
                    _text_cell(output["statistics"][column])
                    for column in request.columns
                )
                cells.extend(
                    _text_cell(output["relative"][metric])
                    for metric in request.relative
                )
                lines.append("\t".join(cells))
            payload = ("\n".join(lines) + "\n").encode("utf-8")
            product_format = "txt"
        validate_relative_product_path(path, component_limit=component_limit)
        files.append(
            ProductFile(
                path,
                payload,
                "report",
                None,
                None,
                product_format,
                {"rows": list(request.rows)},
            )
        )
    return tuple(files)


__all__ = ["materialize_report"]
