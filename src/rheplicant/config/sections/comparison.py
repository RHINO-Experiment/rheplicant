"""Cross-run numeric product comparison."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import NamedTuple

import numpy as np

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import (
    ParsedRun,
    _sweep,
    parsed_options,
    register,
)

_COMPARE_KEYS = frozenset({"of", "metric", "tolerance"})
_METRICS = ("max_rel_diff", "rms", "max_abs")
_DEFERRED = (
    "compare.left_available",
    "compare.right_available",
    "compare.products_compatible",
)


class CompareProduct(NamedTuple):
    left: str
    right: str
    metric: str
    value: float | str
    tolerance: float
    passed: bool
    components: int


class _Leaf(NamedTuple):
    path: str
    shape: tuple[int, ...]
    dtype_class: str
    value: np.ndarray


def _dtype_class(dtype: np.dtype) -> str:
    if dtype.kind == "b":
        return "bool"
    if dtype.kind in "iu":
        return "integer"
    if dtype.kind == "f":
        return "float"
    if dtype.kind == "c":
        return "complex"
    raise ConfigError(f"comparison product has unsupported dtype {dtype}.")


def _numeric_tree(value: object) -> tuple[object, tuple[_Leaf, ...]]:
    leaves: list[_Leaf] = []

    def walk(node: object, path: tuple[str, ...]) -> object:
        if isinstance(node, Mapping):
            if any(type(key) is not str for key in node):
                raise ConfigError("comparison mapping keys must be strings.")
            names = tuple(sorted(node))
            return (
                "mapping",
                names,
                tuple(walk(node[name], (*path, name)) for name in names),
            )
        if isinstance(node, tuple) and hasattr(node, "_fields"):
            names = tuple(node._fields)
            return (
                "namedtuple",
                f"{type(node).__module__}.{type(node).__qualname__}",
                names,
                tuple(walk(getattr(node, name), (*path, name)) for name in names),
            )
        if type(node) in (tuple, list):
            return (
                type(node).__name__,
                len(node),
                tuple(walk(child, (*path, str(index))) for index, child in enumerate(node)),
            )
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            names = tuple(field.name for field in dataclasses.fields(node))
            return (
                "dataclass",
                f"{type(node).__module__}.{type(node).__qualname__}",
                names,
                tuple(walk(getattr(node, name), (*path, name)) for name in names),
            )
        try:
            array = np.asarray(node)
        except Exception:
            raise ConfigError(
                f"comparison product at {'.'.join(path) or '<root>'} is not numeric."
            ) from None
        if array.dtype.hasobject:
            raise ConfigError(
                f"comparison product at {'.'.join(path) or '<root>'} is not numeric."
            )
        kind = _dtype_class(array.dtype)
        if kind in ("float", "complex") and not np.all(np.isfinite(array)):
            raise ConfigError("comparison inputs must contain only finite numbers.")
        index = len(leaves)
        leaves.append(
            _Leaf("/".join(path) or "value", tuple(array.shape), kind, array)
        )
        return ("leaf", index)

    structure = walk(value, ())
    if not leaves:
        raise ConfigError("comparison product contains no numeric leaves.")
    return structure, tuple(leaves)


def compare_products(
    left: object,
    right: object,
    *,
    metric: str,
    tolerance: float,
    left_name: str,
    right_name: str,
) -> CompareProduct:
    """Compare two isomorphic numeric pytrees without ignoring structure."""
    if metric not in _METRICS:
        raise ConfigError(f"comparison metric {metric!r} is not registered.")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float, np.number))
        or not math.isfinite(float(tolerance))
        or float(tolerance) < 0
    ):
        raise ConfigError("comparison tolerance must be finite and non-negative.")
    tolerance = float(tolerance)
    left_structure, left_leaves = _numeric_tree(left)
    right_structure, right_leaves = _numeric_tree(right)
    if left_structure != right_structure:
        raise ConfigError("comparison product structure or mapping keys disagree.")
    components = 0
    maximum = 0.0
    square_sum = 0.0
    infinite = False
    for left_leaf, right_leaf in zip(left_leaves, right_leaves, strict=True):
        if left_leaf.shape != right_leaf.shape:
            raise ConfigError(
                f"comparison shape disagrees at {left_leaf.path}: "
                f"{left_leaf.shape} != {right_leaf.shape}."
            )
        if left_leaf.dtype_class != right_leaf.dtype_class:
            raise ConfigError(
                f"comparison dtype class disagrees at {left_leaf.path}: "
                f"{left_leaf.dtype_class} != {right_leaf.dtype_class}."
            )
        value_dtype = np.complex128 if left_leaf.dtype_class == "complex" else np.float64
        left_value = left_leaf.value.astype(value_dtype, copy=False)
        right_value = right_leaf.value.astype(value_dtype, copy=False)
        difference = np.abs(left_value - right_value).astype(np.float64, copy=False)
        components += int(difference.size)
        if metric == "max_abs":
            if difference.size:
                maximum = max(maximum, float(np.max(difference)))
        elif metric == "rms":
            square_sum += float(np.sum(np.square(difference), dtype=np.float64))
        elif metric == "max_rel_diff":
            one_zero = np.logical_xor(left_value == 0, right_value == 0)
            if bool(np.any(one_zero)):
                infinite = True
                continue
            denominator = np.maximum(np.abs(left_value), np.abs(right_value))
            relative = np.zeros(difference.shape, dtype=np.float64)
            np.divide(difference, denominator, out=relative, where=denominator != 0)
            if relative.size:
                maximum = max(maximum, float(np.max(relative)))
        else:
            raise ConfigError(f"comparison metric {metric!r} is not registered.")
    if components == 0:
        raise ConfigError("comparison product contains empty arrays only.")
    if infinite or not math.isfinite(square_sum):
        value: float | str = "infinity"
        passed = False
    else:
        value = math.sqrt(square_sum / components) if metric == "rms" else maximum
        if not math.isfinite(value):
            value = "infinity"
            passed = False
        else:
            passed = bool(value <= tolerance)
    return CompareProduct(
        left_name,
        right_name,
        metric,
        value,
        tolerance,
        passed,
        components,
    )


def _parse_compare(options: Mapping[str, object], context: object):
    spec = context.spec
    where = f"runs[{spec.name!r}]"
    _sweep(spec, _COMPARE_KEYS)
    if "of" not in options:
        raise ConfigError(f"{where}: of: is required.")
    of = options["of"]
    if (
        type(of) is not list
        or len(of) != 2
        or any(type(name) is not str or not name for name in of)
        or of[0] == of[1]
    ):
        raise ConfigError(f"{where}: of: is a list of two distinct run names.")
    metric = options.get("metric")
    if metric not in _METRICS:
        raise ConfigError(f"{where}: metric: must be one of {list(_METRICS)}.")
    tolerance = options.get("tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(tolerance)
        or tolerance < 0
    ):
        raise ConfigError(f"{where}: tolerance: is a finite non-negative number.")
    normalized = {"of": tuple(of), "metric": metric, "tolerance": float(tolerance)}
    return parsed_options(normalized, resolved={**normalized, "of": list(of)})


def _compare_pre_execute(
    parsed: ParsedRun,
    _configured: object,
    previous: Mapping[str, object],
) -> None:
    for name in parsed.options["of"]:
        if name not in previous:
            raise ConfigError(
                f"runs[{parsed.name!r}]: of: {name!r} names no earlier run; "
                f"by now {sorted(previous)} have run."
            )
        if previous[name].error is not None:
            raise ConfigError(
                f"runs[{parsed.name!r}]: of: {name!r} refused "
                f"({previous[name].error}), so it has no product to compare."
            )


def _comparable_product(result: object) -> object:
    product = result.product
    kind = result.kind
    if kind == "forward":
        return product.data
    if kind == "plan.estimate":
        return product.values
    if kind in ("plan.sample", "nuts"):
        return product.samples
    if kind == "npe":
        return {
            "samples": product.samples,
            "best_step": product.best_step,
            "train_loss": product.train_loss,
            "validation_loss": product.validation_loss,
        }
    if kind == "conjugate.wiener":
        return {
            key: product[key]
            for key in ("mean", "residual", "fisher", "covariance")
            if key in product
        }
    if kind == "conjugate.gcr":
        return {
            key: product[key]
            for key in ("draws", "residual", "noise_std", "gls")
            if key in product and product[key] is not None
        }
    if kind == "conjugate.gls":
        return {
            "noise_std": product.noise_std,
            "solution": product.solution,
            "residual": product.residual,
            "iterations": product.iterations,
            "delta": product.delta,
            "converged": product.converged,
        }
    if kind == "identifiability":
        return {
            "rank": product.rank,
            "nullity": product.nullity,
            "singular_values": product.singular_values,
            "null_space": product.null_space,
            "jacobian": product.jacobian,
            "column_norms": product.column_norms,
        }
    return product


@register(
    "compare",
    parse=_parse_compare,
    pre_execute=_compare_pre_execute,
    deferred_checks=_DEFERRED,
)
def _run_compare(
    run: ParsedRun,
    _configured: object,
    previous: Mapping[str, object],
) -> CompareProduct:
    _sweep(run, _COMPARE_KEYS)
    left_name, right_name = run.options["of"]
    return compare_products(
        _comparable_product(previous[left_name]),
        _comparable_product(previous[right_name]),
        metric=run.options["metric"],
        tolerance=run.options["tolerance"],
        left_name=left_name,
        right_name=right_name,
    )


__all__ = ["CompareProduct", "compare_products"]
