"""A9: every writable numeric destination agrees with the dimension catalog."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.dimension_catalog import (
    CONFIG_CONTEXTUAL,
    CONFIG_DIMENSIONS,
    CONFIG_REQUIRED_UNIT,
    CONFIG_SPECIAL,
    MODEL_DIMENSIONS,
    MODEL_FORMULA_BINDINGS,
    RESOURCE_DIMENSIONS,
    RESOURCE_SPECIAL,
)
from rheplicant.config.dimensions import (
    _FORMULA_REGISTRY,
    DimensionEnvironment,
    DimensionSpec,
    current_dimension_environment,
    dimension_environment_for,
    dimension_of,
    dimension_spec_for,
    signature,
    signature_label,
)
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.sections.model import operator_table
from rheplicant.config.units import canonical_unit

_SHORTHAND_UNIT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s+(\S+)$")
_MODEL_TOKENS = dict(MODEL_DIMENSIONS)
_CONFIG_TOKENS = dict(CONFIG_DIMENSIONS)
_RESOURCE_TOKENS = dict(RESOURCE_DIMENSIONS)


def _qualified(cls: type, field: str | None = None) -> str:
    base = f"{cls.__module__}.{cls.__qualname__}"
    return base if field is None else f"{base}.{field}"


def _selected_class(node_id: str, node: Mapping[str, Any]) -> type | None:
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
    except ConfigError:
        return None  # The value grammar owns unknown/malformed-unit wording.
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
        DimensionSpec("fixed", signature(expected_token)),
    )
    if finding is not None:
        yield finding


def _model(document: Mapping[str, Any]) -> Iterable[Finding]:
    model = document.get("model")
    if not isinstance(model, Mapping):
        return
    for node_id, node in model.items():
        if not isinstance(node, Mapping) or any(key in node for key in ("compose", "python")):
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
        for field, value in node.items():
            if field not in writable:
                continue
            selector = f"{class_name}.{field}"
            where = f"model.{node_id}.{field}"
            try:
                spec = dimension_spec_for("model_field", selector)
            except ConfigError:
                yield refuse(
                    "A9",
                    where,
                    f"{where}: {selector} has no dimension catalog row; the plugin "
                    "must call register_dimension for every writable field (check A9).",
                )
                continue
            if spec.disposition == "fixed":
                expected_token = _MODEL_TOKENS.get(selector)
                if expected_token is None:
                    expected_token = next(
                        (
                            token
                            for token in (
                                "Hz",
                                "s",
                                "unix_s",
                                "K",
                                "deg",
                                "m",
                                "ohm",
                                "dimensionless",
                                "count",
                                "samples",
                                "bits",
                                "channels",
                                "cycles",
                                "adc_count",
                                "adc_count/K",
                                "Hz/s",
                                "dimensionless/s",
                                "cycles/samples",
                            )
                            if signature(token) == spec.signature
                        ),
                        repr(spec.signature),
                    )
                yield from _fixed(
                    where,
                    value,
                    expected_token,
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


def _path_matches(pattern: str, path: str) -> bool:
    expression = re.escape(pattern)
    expression = expression.replace(r"\*", r"[^.\[\]]+")
    expression = expression.replace(r"\[\]", r"\[\d+\]")
    return re.fullmatch(expression, path) is not None


def _config(document: Mapping[str, Any], environment: DimensionEnvironment) -> Iterable[Finding]:
    for path, node in _walk(document):
        for selector, expected_token in CONFIG_DIMENSIONS:
            if _path_matches(selector, path):
                yield from _fixed(
                    path,
                    node,
                    expected_token,
                    required=selector in CONFIG_REQUIRED_UNIT,
                )
        for selector, resolver in CONFIG_CONTEXTUAL.items():
            if not _path_matches(selector, path):
                continue
            token = _declared_unit(node)
            if token is None:
                continue
            expected = None
            if resolver == "prediction":
                expected = environment.prediction_dimension
            elif resolver == "model_input":
                expected = environment.model_input_dimension
            elif resolver == "latent":
                names = path.split(".")
                expected = next(
                    (
                        environment.latent_dimensions[name]
                        for name in names
                        if name in environment.latent_dimensions
                    ),
                    None,
                )
            if expected is None:
                continue
            try:
                actual = dimension_of(canonical_unit(token))
            except ConfigError:
                continue
            if actual != expected:
                yield refuse(
                    "A9",
                    path,
                    f"{path}: unit {token!r} does not match its {resolver} dimension "
                    f"{signature_label(expected)} (check A9).",
                )
        for selector, (disposition, _) in CONFIG_SPECIAL.items():
            if (
                disposition == "structural"
                and _path_matches(selector, path)
                and _declared_unit(node) is not None
            ):
                yield refuse(
                    "A9",
                    path,
                    f"{path}: is structural and refuses unit:; remove the unit "
                    "declaration (check A9).",
                )


def _resource_selectors(kind: str, spec: Mapping[str, Any], path: str):
    singular = {
        "arrays": "array",
        "bases": "basis",
        "beams": "beam",
        "projectors": "projector",
        "s_params": "s_param",
        "sky_models": "sky_model",
    }.get(kind)
    if singular is None:
        return
    root = f"rheplicant.config.kinds.{kind}.build_{singular}"
    discriminant = spec.get("format") or spec.get("kind")
    for field_path, value in _walk(spec):
        relative = field_path
        candidates = [f"{root}.{relative}"]
        if isinstance(discriminant, str):
            candidates.insert(0, f"{root}.{discriminant}.{relative}")
        yield f"{path}.{field_path}", candidates, value


def _resources(document: Mapping[str, Any]) -> Iterable[Finding]:
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
                yield from _fixed(
                    f"resources.{kind}.{name}.maps",
                    {"value": spec.get("maps"), "unit": spec["unit"]},
                    "K",
                    required=False,
                )
            for where, candidates, value in _resource_selectors(
                str(kind), spec, f"resources.{kind}.{name}"
            ):
                selector = next((one for one in candidates if one in _RESOURCE_TOKENS), None)
                if selector is not None:
                    yield from _fixed(where, value, _RESOURCE_TOKENS[selector], required=False)
                special = next((one for one in candidates if one in RESOURCE_SPECIAL), None)
                if special is not None and RESOURCE_SPECIAL[special][0] == "structural":
                    if _declared_unit(value) is not None:
                        yield refuse(
                            "A9",
                            where,
                            f"{where}: is structural and refuses unit:; remove the unit "
                            "declaration (check A9).",
                        )


@register("A9")
def _dimensions(document: Mapping[str, Any]) -> Iterable[Finding]:
    environment = current_dimension_environment()
    if (
        environment.prediction_dimension is None
        and environment.model_input_dimension is None
        and not environment.latent_dimensions
    ):
        environment = dimension_environment_for(document)
    yield from _model(document)
    yield from _config(document, environment)
    yield from _resources(document)
