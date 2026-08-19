"""A9: every writable numeric destination agrees with the dimension catalog."""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Iterable, Mapping
from typing import Any

from _rheplicant_bootstrap.path_syntax import longest_legal_prefix
from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.dimension_catalog import MODEL_FORMULA_BINDINGS
from rheplicant.config.dimensions import (
    _FORMULA_REGISTRY,
    DimensionEnvironment,
    DimensionSpec,
    _compose_stage_specs,
    _constructible_operator_class,
    _loaded_operator_target,
    _pipeline_stage_specs,
    _plugin_formulas_for_class,
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
from rheplicant.config.preflight import register
from rheplicant.config.resources import _KINDS
from rheplicant.config.sections.model import operator_table
from rheplicant.config.units import canonical_unit

_SHORTHAND_UNIT = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s+(\S+)$")


def _refuse(where: str, message: str) -> Finding:
    """Keep the full user spelling in the message and a legal path in ``where``."""
    return refuse("A9", longest_legal_prefix(where), message)


def _qualified(cls: type, field: str | None = None) -> str:
    base = f"{cls.__module__}.{cls.__qualname__}"
    return base if field is None else f"{base}.{field}"


def _selected_class(
    node_id: str | None,
    node: Mapping[str, Any],
    table: Mapping[str, tuple[type, ...]],
) -> type | None:
    if "python" in node:
        return _loaded_operator_target(node["python"])
    if node_id is None:
        declared = node.get("type")
        if not isinstance(declared, str):
            return None
        import rheplicant.radio as radio

        return _constructible_operator_class(vars(radio).get(declared))
    classes = table.get(node_id, ())
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
        return _refuse(where, f"{where}: {error} (check A9).")
    if actual == expected.signature:
        return None
    return _refuse(
        where,
        f"{where}: unit {token!r} has dimension {signature_label(actual)}, but this "
        f"destination requires {expected_token}. Use {{value: {_raw_value(node)!r}, "
        f"unit: {expected_token}}} (check A9).",
    )


def _required(where: str, expected_token: str) -> Finding:
    return _refuse(
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


def _operator_entries(
    document: Mapping[str, Any],
    table: Mapping[str, tuple[type, ...]],
) -> Iterable[tuple[str, str | None, Any]]:
    """Every raw spec that the real model/twin builders construct.

    The graph leg delegates its four shapes to the preflight model reader
    shared with the build contract: a normal node, each list/FAN member, and
    each compose stage.  Pipeline stages have no graph node and select their
    exported class directly.  ``twin.replace`` reaches the same single-node
    builder as a graph entry, at its own document path.
    """
    model = document.get("model")
    if isinstance(model, Mapping):
        kind = model.get("kind", "graph")
        if kind == "pipeline":
            for index, entry in enumerate(_pipeline_stage_specs(model)):
                yield f"model.stages[{index}]", None, entry
        elif kind == "graph":
            from rheplicant.config.preflight.model import (
                _nodes,
                _t4_entries,
                _t4_graph,
            )
            from rheplicant.config.sections.compose import many_shape_problem

            graph = _t4_graph()
            nodes = _nodes(document)
            for node_id, spec in nodes.items():
                node = graph.nodes.get(node_id)
                if node is None:
                    continue
                if node.many and many_shape_problem(
                    str(node_id), spec, many=True
                ) is not None:
                    continue
                if isinstance(spec, Mapping) and "compose" in spec:
                    if not _compose_stage_specs(
                        spec, node.kind, str(node_id)
                    ):
                        continue
                for relative, entry in _t4_entries(
                    str(node_id), spec, many=bool(node.many)
                ):
                    yield f"model.{relative}", str(node_id), entry
            # A live plugin table can expose an additional node before the
            # graph's separate placement check accepts it.  Keep A9's plugin
            # completeness contract independent of that check, as it was for
            # the original single-entry walk.
            for node_id, spec in nodes.items():
                if node_id not in graph.nodes and str(node_id) in table:
                    yield f"model.{node_id}", str(node_id), spec

    if not isinstance(model, Mapping) or model.get("kind", "graph") != "graph":
        return
    inference = document.get("inference")
    twin = inference.get("twin") if isinstance(inference, Mapping) else None
    replace = twin.get("replace") if isinstance(twin, Mapping) else None
    if isinstance(replace, Mapping):
        for node_id, entry in replace.items():
            yield f"inference.twin.replace.{node_id}", str(node_id), entry


def _model(document: Mapping[str, Any]) -> Iterable[Finding]:
    table = operator_table()
    for where, node_id, node in _operator_entries(document, table):
        if not isinstance(node, Mapping):
            continue
        cls = _selected_class(node_id, node, table)
        if cls is None:
            continue
        class_name = _qualified(cls)
        binding = MODEL_FORMULA_BINDINGS.get(class_name)
        plugin_formulas = (
            () if binding is not None else _plugin_formulas_for_class(cls)
        )
        missing_formula = (
            not all(name in _FORMULA_REGISTRY for name in binding.formulas)
            if binding is not None
            else not plugin_formulas
        )
        if missing_formula:
            yield _refuse(
                where,
                f"{where}: {class_name} has no registered dimension formula; "
                "the plugin must call register_dimension_formula with this concrete "
                "class in producers (check A9).",
            )
        elif len(plugin_formulas) > 1:
            names = [formula.name for formula in plugin_formulas]
            yield _refuse(
                where,
                f"{where}: {class_name} has ambiguous output formula registrations "
                f"{names}; a plugin class producer must uniquely identify its output "
                "formula. Auxiliary formulas must name their actual helper producer "
                "(or wait for a future explicit operator-formula binding) (check A9).",
            )
        writable = {field.name for field in dataclasses.fields(cls) if field.init}
        specs: dict[str, DimensionSpec] = {}
        for field in sorted(writable):
            selector = f"{class_name}.{field}"
            field_where = f"{where}.{field}"
            rows = matching_dimension_rows("model_field", selector)
            if not rows:
                yield _refuse(
                    field_where,
                    f"{field_where}: {selector} has no dimension catalog row; the plugin "
                    "must call register_dimension for every writable field (check A9).",
                )
                continue
            if len(rows) != 1:
                yield _refuse(
                    field_where,
                    f"{field_where}: ambiguous dimension selectors match model_field "
                    f"destination {selector!r} (check A9).",
                )
                continue
            specs[field] = rows[0][1]
        for field, value in node.items():
            if field not in writable:
                continue
            field_where = f"{where}.{field}"
            spec = specs.get(field)
            if spec is None:
                continue
            if spec.disposition == "fixed":
                expected_token = signature_token(spec.signature)
                yield from _fixed(
                    field_where,
                    value,
                    expected_token,
                    spec,
                    required=spec.unit_policy == "required",
                )
            elif spec.disposition == "structural" and _declared_unit(value) is not None:
                yield _refuse(
                    field_where,
                    f"{field_where}: is structural and refuses unit:; remove the unit "
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
    return _refuse(
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
            yield _refuse(
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
        yield _refuse(where, f"{where}: {error} (check A9).")
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
        yield _refuse(
            where,
            f"{where}: conflicting dimension evidence declares {labels}; "
            "the declaration, prior, and binding must agree (check A9).",
        )
    yield from _model(document)
    yield from _config(document, environment)
    yield from _resources(document, environment)
