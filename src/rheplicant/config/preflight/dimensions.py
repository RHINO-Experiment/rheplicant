"""A9: every writable numeric destination agrees with the dimension catalog."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping
from typing import Any

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.dimension_catalog import MODEL_FORMULA_BINDINGS
from rheplicant.config.dimensions import (
    _FORMULA_REGISTRY,
    DimensionEnvironment,
    DimensionSpec,
    current_dimension_environment,
    dimension_environment_and_conflicts_for,
    dimension_for,
    dimension_of,
    matching_dimension_rows,
    signature_label,
    signature_token,
)
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.hatch import import_target
from rheplicant.config.preflight import register
from rheplicant.config.resources import _KINDS
from rheplicant.config.sections.model import operator_table
from rheplicant.config.units import canonical_unit

_SHORTHAND_UNIT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s+(\S+)$")


def _qualified(cls: type, field: str | None = None) -> str:
    base = f"{cls.__module__}.{cls.__qualname__}"
    return base if field is None else f"{base}.{field}"


def _selected_class(node_id: str, node: Mapping[str, Any]) -> type | None:
    if "python" in node:
        try:
            selected = import_target(node["python"])
        except ConfigError:
            return None
        return selected if isinstance(selected, type) else None
    classes = operator_table().get(node_id, ())
    declared = node.get("type")
    if declared is None:
        return classes[0] if len(classes) == 1 else None
    return next((cls for cls in classes if cls.__name__ == declared), None)


def _declared_unit(node: Any) -> str | None:
    if isinstance(node, Mapping):
        token = node.get("unit")
        return token if isinstance(token, str) else None
    if isinstance(node, str):
        match = _SHORTHAND_UNIT.fullmatch(node.strip())
        return None if match is None else match.group(1)
    return None


def _raw_value(node: Any) -> Any:
    if isinstance(node, Mapping) and "value" in node:
        return node["value"]
    if isinstance(node, str):
        try:
            return float(node.split(maxsplit=1)[0])
        except (ValueError, IndexError):
            return node
    return node


def _mismatch(
    where: str,
    node: Any,
    token: str,
    expected_token: str,
    expected: DimensionSpec,
) -> Finding | None:
    try:
        actual = dimension_of(canonical_unit(token))
    except ConfigError as error:
        return refuse("A9", where, f"{where}: {error} (check A9).")
    if actual == expected.signature:
        return None
    return refuse(
        "A9",
        where,
        f"{where}: unit {token!r} has dimension {signature_label(actual)}, but this "
        f"destination requires {expected_token}. Use {{value: {_raw_value(node)!r}, "
        f"unit: {expected_token}}} (check A9).",
    )


def _required(where: str, expected_token: str) -> Finding:
    return refuse(
        "A9",
        where,
        f"{where}: requires an explicit unit declaring {expected_token}; this field "
        f"has no unit in its source definition. Use {{value: <number>, unit: "
        f"{expected_token}}} (check A9).",
    )


def _fixed(
    where: str,
    node: Any,
    expected_token: str,
    expected: DimensionSpec,
    *,
    required: bool,
) -> Iterable[Finding]:
    token = _declared_unit(node)
    if token is None:
        if required:
            yield _required(where, expected_token)
        return
    finding = _mismatch(
        where,
        node,
        token,
        expected_token,
        expected,
    )
    if finding is not None:
        yield finding


def _model(document: Mapping[str, Any]) -> Iterable[Finding]:
    model = document.get("model")
    if not isinstance(model, Mapping):
        return
    for node_id, node in model.items():
        if not isinstance(node, Mapping) or "compose" in node:
            continue
        cls = _selected_class(str(node_id), node)
        if cls is None:
            continue
        class_name = _qualified(cls)
        has_formula = class_name in MODEL_FORMULA_BINDINGS or any(
            class_name in formula.producers for formula in _FORMULA_REGISTRY.values()
        )
        if not has_formula:
            yield refuse(
                "A9",
                f"model.{node_id}",
                f"model.{node_id}: {class_name} has no registered dimension formula; "
                "the plugin must call register_dimension_formula with this concrete "
                "class in producers (check A9).",
            )
        writable = {field.name for field in dataclasses.fields(cls) if field.init}
        specs: dict[str, DimensionSpec] = {}
        for field in sorted(writable):
            selector = f"{class_name}.{field}"
            where = f"model.{node_id}.{field}"
            rows = matching_dimension_rows("model_field", selector)
            if not rows:
                yield refuse(
                    "A9",
                    where,
                    f"{where}: {selector} has no dimension catalog row; the plugin "
                    "must call register_dimension for every writable field (check A9).",
                )
                continue
            if len(rows) != 1:
                yield refuse(
                    "A9",
                    where,
                    f"{where}: ambiguous dimension selectors match model_field "
                    f"destination {selector!r} (check A9).",
                )
                continue
            specs[field] = rows[0][1]
        for field, value in node.items():
            if field not in writable:
                continue
            where = f"model.{node_id}.{field}"
            spec = specs.get(field)
            if spec is None:
                continue
            if spec.disposition == "fixed":
                expected_token = signature_token(spec.signature)
                yield from _fixed(
                    where,
                    value,
                    expected_token,
                    spec,
                    required=spec.unit_policy == "required",
                )
            elif spec.disposition == "structural" and _declared_unit(value) is not None:
                yield refuse(
                    "A9",
                    where,
                    f"{where}: is structural and refuses unit:; remove the unit "
                    "declaration (check A9).",
                )


def _walk(node: Any, prefix: str = ""):
    if isinstance(node, Mapping):
        if prefix:
            yield prefix, node
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk(value, path)
    elif isinstance(node, (list, tuple)):
        for index, value in enumerate(node):
            yield from _walk(value, f"{prefix}[{index}]")
    elif prefix:
        yield prefix, node


def _rows_problem(
    where: str,
    domain: str,
    selector: str,
    rows: tuple[tuple[Any, DimensionSpec], ...],
) -> Finding | None:
    if len(rows) <= 1:
        return None
    return refuse(
        "A9",
        where,
        f"{where}: ambiguous dimension selectors match {domain} destination "
        f"{selector!r} (check A9).",
    )


def _validate_live_row(
    where: str,
    node: Any,
    domain: str,
    selector: str,
    rows: tuple[tuple[Any, DimensionSpec], ...],
    environment: DimensionEnvironment,
) -> Iterable[Finding]:
    problem = _rows_problem(where, domain, selector, rows)
    if problem is not None:
        yield problem
        return
    if not rows:
        return
    spec = rows[0][1]
    token = _declared_unit(node)
    if spec.disposition == "fixed":
        yield from _fixed(
            where,
            node,
            signature_token(spec.signature),
            spec,
            required=spec.unit_policy == "required",
        )
        return
    if spec.disposition == "structural":
        if token is not None:
            yield refuse(
                "A9",
                where,
                f"{where}: is structural and refuses unit:; remove the unit "
                "declaration (check A9).",
            )
        return
    if spec.disposition != "contextual" or token is None:
        return
    try:
        expected = dimension_for(
            DestinationDescriptor(selector, domain, where), environment
        )
    except ConfigError as error:
        yield refuse("A9", where, f"{where}: {error} (check A9).")
        return
    if expected is None:
        return
    finding = _mismatch(
        where, node, token, signature_token(expected), DimensionSpec("fixed", expected)
    )
    if finding is not None:
        yield finding


def _config(document: Mapping[str, Any], environment: DimensionEnvironment) -> Iterable[Finding]:
    for path, node in _walk(document):
        rows = matching_dimension_rows("config_path", path)
        yield from _validate_live_row(
            path, node, "config_path", path, rows, environment
        )


def _resource_selectors(kind: str, spec: Mapping[str, Any], path: str):
    builder = _KINDS.get(kind)
    if builder is None:
        return
    root = f"{builder.__module__}.{builder.__qualname__}"
    discriminant = spec.get("format") or spec.get("kind")
    for field_path, value in _walk(spec):
        relative = field_path
        candidates = [f"{root}.{relative}"]
        if isinstance(discriminant, str):
            candidates.insert(0, f"{root}.{discriminant}.{relative}")
        yield f"{path}.{field_path}", candidates, value


def _resources(
    document: Mapping[str, Any], environment: DimensionEnvironment
) -> Iterable[Finding]:
    resources = document.get("resources")
    if not isinstance(resources, Mapping):
        return
    for kind, entries in resources.items():
        if not isinstance(entries, Mapping):
            continue
        for name, spec in entries.items():
            if not isinstance(spec, Mapping):
                continue
            if (
                kind == "sky_models"
                and spec.get("kind") == "maps"
                and isinstance(spec.get("unit"), str)
            ):
                selector = (
                    "rheplicant.config.kinds.sky_models."
                    "build_sky_model.maps.maps"
                )
                rows = matching_dimension_rows("resource_field", selector)
                yield from _validate_live_row(
                    f"resources.{kind}.{name}.maps",
                    {"value": spec.get("maps"), "unit": spec["unit"]},
                    "resource_field",
                    selector,
                    rows,
                    environment,
                )
            for where, candidates, value in _resource_selectors(
                str(kind), spec, f"resources.{kind}.{name}"
            ):
                matched = []
                for candidate in candidates:
                    matched.extend(matching_dimension_rows("resource_field", candidate))
                unique = tuple(dict(matched).items())
                selector = candidates[0]
                yield from _validate_live_row(
                    where,
                    value,
                    "resource_field",
                    selector,
                    unique,
                    environment,
                )


@register("A9")
def _dimensions(document: Mapping[str, Any]) -> Iterable[Finding]:
    environment = current_dimension_environment()
    inferred, conflicts = dimension_environment_and_conflicts_for(document)
    if (
        environment.prediction_dimension is None
        and environment.model_input_dimension is None
        and not environment.latent_dimensions
    ):
        environment = inferred
    for name, signatures in conflicts.items():
        labels = ", ".join(signature_label(value) for value in signatures)
        where = f"inference.parameters.{name}"
        yield refuse(
            "A9",
            where,
            f"{where}: conflicting dimension evidence declares {labels}; "
            "the declaration, prior, and binding must agree (check A9).",
        )
    yield from _model(document)
    yield from _config(document, environment)
    yield from _resources(document, environment)
