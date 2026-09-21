"""Inferring a node's dimensions from the operators a document places.

The half that has to know what an operator is: walking a graph or a pipeline,
resolving each stage to a class, and asking the registry what that class's
formula says. It reads the registry and the algebra and is read by neither.
"""

from __future__ import annotations

import dataclasses
import sys
from collections.abc import Mapping, Sequence
from types import ModuleType

from rheplicant.config.errors import ConfigError

from .dimension_algebra import (
    DimensionSignature,
    signature,
)
from .dimension_environment import (
    DimensionEnvironment,
)
from .dimension_registry import (
    _FORMULA_REGISTRY,
    FormulaRegistration,
    dimension_spec_for,
)


def operator_table() -> dict[str, tuple[type, ...]]:
    """The live built-in/plugin class table, imported lazily to avoid a cycle."""
    from rheplicant.config.sections.model import operator_table as live_operator_table

    return live_operator_table()


def _constructible_operator_class(value: object) -> type | None:
    """A class the model builder can hand to its dataclass field delivery."""
    from rheplicant.core.operator import AbstractOperator

    return (
        value
        if isinstance(value, type)
        and issubclass(value, AbstractOperator)
        and dataclasses.is_dataclass(value)
        else None
    )


def _loaded_operator_target(
    target: object,
) -> type | None:
    """Resolve an already-loaded ``python:`` operator without importing code."""
    if not isinstance(target, str) or target.count(":") != 1 or not all(target.split(":")):
        return None
    module_name, attribute = target.split(":")
    module = sys.modules.get(module_name)
    if not isinstance(module, ModuleType):
        return None
    selected = vars(module).get(attribute)
    return _constructible_operator_class(selected)


def _pipeline_stage_specs(model: Mapping[str, object]) -> tuple[Mapping, ...]:
    """Stages reached by ``_build_pipeline``, or none before its own refusal."""
    from rheplicant.config.sections.compose import (
        pipeline_shape_problem,
        stage_shape_problem,
    )

    if pipeline_shape_problem(model) is not None:
        return ()
    stages = model.get("stages")
    assert isinstance(stages, list)
    if any(
        stage_shape_problem(f"stages[{index}]", stage) is not None
        for index, stage in enumerate(stages)
    ):
        return ()
    return tuple(stages)


def _compose_stage_specs(
    spec: Mapping[str, object], node_kind: str, node_id: str
) -> tuple[Mapping, ...]:
    """Stages reached by ``_compose``, after all of its earlier pure gates."""
    from rheplicant.config.sections.compose import (
        compose_shape_problem,
        stage_shape_problem,
    )

    if compose_shape_problem(node_id, spec, node_kind) is not None:
        return ()
    stages = spec.get("stages")
    assert isinstance(stages, list)
    if any(
        stage_shape_problem(f"{node_id}.stages[{index}]", stage) is not None
        for index, stage in enumerate(stages)
    ):
        return ()
    return tuple(stages)


def _selected_class(
    node_id: str | None,
    node: object,
    table: Mapping[str, Sequence[type]],
) -> type | None:
    if not isinstance(node, Mapping):
        return None
    if "python" in node:
        return _loaded_operator_target(node["python"])
    if node_id is None:
        declared = node.get("type")
        if not isinstance(declared, str):
            return None
        import rheplicant.radio as radio

        return _constructible_operator_class(vars(radio).get(declared))
    choices = table.get(node_id, ())
    declared = node.get("type")
    if declared is None:
        return choices[0] if len(choices) == 1 else None
    return next((choice for choice in choices if choice.__name__ == declared), None)


def _selected_model_classes(
    effective_document: Mapping[str, object],
    *,
    table: Mapping[str, Sequence[type]] | None = None,
) -> tuple[dict[str, tuple[type, ...]], bool]:
    """Return selected classes and whether a declared graph node is unreadable."""
    model = effective_document.get("model")
    if not isinstance(model, Mapping):
        return {}, True
    live_table = operator_table() if table is None else table
    selected: dict[str, tuple[type, ...]] = {}
    kind = model.get("kind", "graph")
    if kind == "pipeline":
        stages = _pipeline_stage_specs(model)
        incomplete = not stages
        for stage in stages:
            name = stage["name"]
            cls = _selected_class(None, stage, live_table)
            if cls is not None:
                selected[str(name)] = (cls,)
            else:
                incomplete = True
        return selected, incomplete
    if kind != "graph":
        return {}, True
    from rheplicant.config.sections.compose import many_shape_problem, node_specs
    from rheplicant.radio.graph import RADIO_GRAPH

    incomplete = False
    for node_id, raw in node_specs(model).items():
        if node_id not in RADIO_GRAPH.nodes:
            incomplete = True
            continue
        node_spec = RADIO_GRAPH.nodes[node_id]
        entries: list[object]
        if node_spec.many:
            if many_shape_problem(str(node_id), raw, many=True) is not None:
                incomplete = True
                continue
            if isinstance(raw, Mapping):
                entries = list(raw.values())
            else:
                assert isinstance(raw, list)
                entries = list(raw)
        elif isinstance(raw, Mapping) and "compose" in raw:
            entries = list(_compose_stage_specs(raw, node_spec.kind, str(node_id)))
            if not entries:
                incomplete = True
                continue
        else:
            entries = [raw]
        classes_list: list[type] = []
        for raw_entry in entries:
            cls = _selected_class(str(node_id), raw_entry, live_table)
            if cls is None:
                incomplete = True
                continue
            classes_list.append(cls)
            if _formula_for_class(cls) is None:
                incomplete = True
        classes = tuple(classes_list)
        if classes:
            selected[str(node_id)] = classes
    return selected, incomplete


def _plugin_formulas_for_class(cls: type) -> tuple[FormulaRegistration, ...]:
    """Every live formula naming an unbound plugin class as its producer."""
    qualified = f"{cls.__module__}.{cls.__qualname__}"
    return tuple(
        formula for formula in _FORMULA_REGISTRY.values() if qualified in formula.producers
    )


def _formula_for_class(cls: type) -> FormulaRegistration | None:
    qualified = f"{cls.__module__}.{cls.__qualname__}"
    from rheplicant.config.dimension_catalog import MODEL_FORMULA_BINDINGS

    binding = MODEL_FORMULA_BINDINGS.get(qualified)
    if binding is not None:
        return _FORMULA_REGISTRY.get(binding.output_formula)
    plugin = _plugin_formulas_for_class(cls)
    return plugin[0] if len(plugin) == 1 else None


def _shared(signatures: Sequence[DimensionSignature | None]) -> DimensionSignature | None:
    known = [value for value in signatures if value is not None]
    if not known or any(value != known[0] for value in known[1:]):
        return None
    return known[0]


def _operator_dimensions(
    cls: type, incoming: DimensionSignature | None
) -> tuple[DimensionSignature | None, DimensionSignature | None]:
    """One live output formula applied to one operator's incoming dimension."""
    formula = _formula_for_class(cls)
    if formula is None:
        return None, None
    input_operand = next((operand for operand in formula.operands if operand.role == "input"), None)
    operator_input = incoming
    if (
        operator_input is None
        and input_operand is not None
        and input_operand.spec.disposition == "fixed"
    ):
        operator_input = input_operand.spec.signature
    output = formula.result.signature if formula.result.disposition == "fixed" else operator_input
    return operator_input, output


def _graph_dimensions(
    selected: Mapping[str, tuple[type, ...]],
) -> tuple[DimensionSignature | None, DimensionSignature | None]:
    from rheplicant.radio.graph import RADIO_GRAPH

    outputs: dict[str, DimensionSignature | None] = {}
    model_input: DimensionSignature | None = None
    for node_id in RADIO_GRAPH._topo:
        incoming = _shared([outputs.get(parent) for parent in RADIO_GRAPH._in[node_id]])
        classes = selected.get(node_id, ())
        if not classes:
            outputs[node_id] = incoming
            continue
        results: list[DimensionSignature | None] = []
        for cls in classes:
            operator_input, output = _operator_dimensions(cls, incoming)
            if model_input is None and operator_input is not None:
                model_input = operator_input
            results.append(output)
        outputs[node_id] = _shared(results)
        if model_input is None and RADIO_GRAPH.nodes[node_id].kind == "source":
            model_input = outputs[node_id]
    prediction = outputs.get(RADIO_GRAPH.sink)
    if prediction is None:
        prediction = next(
            (
                outputs[node]
                for node in reversed(RADIO_GRAPH._topo)
                if outputs.get(node) is not None
            ),
            None,
        )
    return model_input, prediction


def _pipeline_dimensions(
    model: Mapping[str, object], table: Mapping[str, Sequence[type]]
) -> tuple[DimensionSignature | None, DimensionSignature | None]:
    """Infer dimensions by applying live formulas in real pipeline order."""
    stages = _pipeline_stage_specs(model)
    if not stages:
        return None, None
    model_input: DimensionSignature | None = None
    current: DimensionSignature | None = None
    for entry in stages:
        cls = _selected_class(None, entry, table)
        if cls is None:
            return None, None
        operator_input, current = _operator_dimensions(cls, current)
        if model_input is None:
            model_input = operator_input if operator_input is not None else current
        if current is None:
            return model_input, None
    return model_input, current


def _safe_signature(token: object) -> DimensionSignature | None:
    if not isinstance(token, str):
        return None
    try:
        return signature(token)
    except ConfigError:
        return None


def _node_signature(node: object) -> DimensionSignature | None:
    if isinstance(node, Mapping):
        return _safe_signature(node.get("unit"))
    if isinstance(node, str):
        parts = node.strip().split(maxsplit=1)
        if len(parts) == 2:
            try:
                float(parts[0])
            except ValueError:
                return None
            return _safe_signature(parts[1])
    return None


def _binding_target_signature(
    path: object,
    selected: Mapping[str, tuple[type, ...]],
    transform: object,
) -> DimensionSignature | None:
    if not isinstance(path, str):
        return None
    parts = path.split(".")
    if len(parts) < 2:
        return None
    classes = selected.get(parts[0], ())
    signatures: list[DimensionSignature | None] = []
    for cls in classes:
        qualified = f"{cls.__module__}.{cls.__qualname__}.{parts[-1]}"
        try:
            spec = dimension_spec_for("model_field", qualified)
        except ConfigError:
            continue
        signatures.append(spec.signature if spec.disposition == "fixed" else None)
    target = _shared(signatures)
    name = transform if isinstance(transform, str) else None
    if isinstance(transform, Mapping) and len(transform) == 1:
        name = next(iter(transform))
    if name in ("exp", "log", "log_link_basis", "beam_analysis"):
        return signature("dimensionless")
    return target


def _latent_candidates(
    effective_document: Mapping[str, object],
    selected: Mapping[str, tuple[type, ...]],
) -> dict[str, list[DimensionSignature]]:
    inference = effective_document.get("inference")
    if not isinstance(inference, Mapping):
        return {}
    parameters = inference.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    candidates: dict[str, list[DimensionSignature]] = {str(name): [] for name in parameters}
    for name, declaration in parameters.items():
        if not isinstance(declaration, Mapping):
            continue
        found = candidates[str(name)]
        declared = _safe_signature(declaration.get("unit"))
        if declared is not None:
            found.append(declared)
        for key in ("init", "ref"):
            value = _node_signature(declaration.get(key))
            if value is not None:
                found.append(value)
        prior = declaration.get("prior")
        if isinstance(prior, Mapping):
            for family in ("normal", "uniform", "log_normal"):
                body = prior.get(family)
                if isinstance(body, Mapping):
                    for operand in body.values():
                        value = _node_signature(operand)
                        if value is not None:
                            found.append(value)
        into = declaration.get("into")
        paths = (into,) if isinstance(into, str) else into if isinstance(into, list) else ()
        for path in paths:
            value = _binding_target_signature(path, selected, declaration.get("transform"))
            if value is not None:
                found.append(value)
    bindings = inference.get("bindings")
    if isinstance(bindings, list):
        for binding in bindings:
            if not isinstance(binding, Mapping):
                continue
            names = binding.get("latents")
            names = (names,) if isinstance(names, str) else names if isinstance(names, list) else ()
            into = binding.get("into")
            paths = (into,) if isinstance(into, str) else into if isinstance(into, list) else ()
            for name in names:
                if str(name) not in candidates:
                    continue
                for path in paths:
                    value = _binding_target_signature(path, selected, binding.get("transform"))
                    if value is not None:
                        candidates[str(name)].append(value)
    return candidates


def dimension_environment_and_conflicts_for(
    effective_document: Mapping[str, object],
) -> tuple[DimensionEnvironment, dict[str, tuple[DimensionSignature, ...]]]:
    """Infer one document's environment and conflicting latent evidence once."""
    table = operator_table()
    selected, incomplete = _selected_model_classes(effective_document, table=table)
    model = effective_document.get("model")
    if isinstance(model, Mapping) and model.get("kind", "graph") == "pipeline":
        model_input, prediction = _pipeline_dimensions(model, table)
    elif incomplete:
        model_input, prediction = None, None
    else:
        model_input, prediction = _graph_dimensions(selected)
    candidates = _latent_candidates(effective_document, selected)
    environment = DimensionEnvironment(
        latent_dimensions={name: values[0] for name, values in candidates.items() if values},
        prediction_dimension=prediction,
        model_input_dimension=model_input,
    )
    conflicts = {
        name: tuple(dict.fromkeys(values))
        for name, values in candidates.items()
        if len(set(values)) > 1
    }
    return environment, conflicts


def latent_dimension_conflicts_for(
    effective_document: Mapping[str, object],
) -> dict[str, tuple[DimensionSignature, ...]]:
    """Latents whose declarations, priors, and bindings disagree."""
    return dimension_environment_and_conflicts_for(effective_document)[1]
