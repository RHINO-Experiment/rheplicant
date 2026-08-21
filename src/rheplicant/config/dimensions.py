"""Normalized dimension signatures and the closed A9 registries."""

from __future__ import annotations

import dataclasses
import re
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from types import ModuleType
from typing import Literal, TypeAlias

from _rheplicant_bootstrap.types import DestinationDescriptor, DimensionDomain
from rheplicant.config.errors import ConfigError
from rheplicant.config.units import Unit, canonical_unit

PhysicalDimension = tuple[tuple[str, int], ...]
QuantitySignature = tuple[tuple[str, int], ...]


def _canonical_components(
    items: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    totals: Counter[str] = Counter()
    for name, exponent in items:
        if not isinstance(name, str) or not name:
            raise ConfigError(f"dimensions: an atom name is a non-empty string; got {name!r}")
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise ConfigError(
                f"dimensions: an integer exponent is required; got {exponent!r} "
                f"({type(exponent).__name__})."
            )
        totals[name] += exponent
    return tuple(sorted((name, exponent) for name, exponent in totals.items() if exponent))


@dataclass(frozen=True, slots=True)
class DimensionSignature:
    physical: PhysicalDimension
    quantity: QuantitySignature = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical", _canonical_components(self.physical))
        object.__setattr__(self, "quantity", _canonical_components(self.quantity))


DimensionDisposition = Literal["fixed", "contextual", "open", "structural"]
UnitPolicy = Literal["required", "optional", "inherited", "forbidden"]
ContextualResolver = Literal["latent", "prediction", "model_input", "resource", "outer"]
FormulaRule = Literal["product", "same", "affine", "fixed", "radiometer"]


@dataclass(frozen=True, slots=True)
class DimensionSelector:
    domain: DimensionDomain
    selector: str


@dataclass(frozen=True, slots=True)
class DimensionSpec:
    disposition: DimensionDisposition
    signature: DimensionSignature | None = None
    resolver: ContextualResolver | None = None
    unit_policy: UnitPolicy = "optional"
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class FormulaOperand:
    role: str
    spec: DimensionSpec
    exponent: int = 1


@dataclass(frozen=True, slots=True)
class FormulaRegistration:
    name: str
    rule: FormulaRule
    result: DimensionSpec
    operands: tuple[FormulaOperand, ...]
    producers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OperatorFormulaBinding:
    formulas: tuple[str, ...]
    output_formula: str


DimensionFormula: TypeAlias = FormulaRegistration


@dataclass(slots=True)
class DimensionEnvironment:
    latent_dimensions: dict[str, DimensionSignature] = dataclasses.field(default_factory=dict)
    resource_dimensions: dict[str, DimensionSignature | None] = dataclasses.field(
        default_factory=dict
    )
    prediction_dimension: DimensionSignature | None = None
    model_input_dimension: DimensionSignature | None = None


_ATOM_SIGNATURES = {
    "Hz": DimensionSignature((("frequency", 1),)),
    "s": DimensionSignature((("time", 1),)),
    "unix_s": DimensionSignature((("time_epoch", 1),)),
    "K": DimensionSignature((("temperature", 1),)),
    "deg": DimensionSignature((("angle", 1),)),
    "m": DimensionSignature((("length", 1),)),
    "ohm": DimensionSignature((("impedance", 1),)),
    "dimensionless": DimensionSignature(()),
    "count": DimensionSignature((), (("count", 1),)),
    "samples": DimensionSignature((), (("samples", 1),)),
    "bits": DimensionSignature((), (("bits", 1),)),
    "channels": DimensionSignature((), (("channels", 1),)),
    "cycles": DimensionSignature((), (("cycles", 1),)),
    "adc_count": DimensionSignature((("adc_count", 1),)),
}

_DIMENSION_REGISTRY: dict[DimensionSelector, DimensionSpec] = {}
_FORMULA_REGISTRY: dict[str, FormulaRegistration] = {}


@dataclass(slots=True)
class _RegistrySnapshot:
    rows: tuple[tuple[DimensionSelector, DimensionSpec], ...]
    matches: dict[
        tuple[DimensionDomain, str],
        tuple[tuple[DimensionSelector, DimensionSpec], ...],
    ] = dataclasses.field(default_factory=dict)


_ACTIVE_ENVIRONMENT: ContextVar[DimensionEnvironment | None] = ContextVar(
    "rheplicant_dimension_environment", default=None
)
_ACTIVE_REGISTRY_SNAPSHOT: ContextVar[_RegistrySnapshot | None] = ContextVar(
    "rheplicant_dimension_registry_snapshot", default=None
)

_SEGMENT = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\[\])?|\*)\Z")
_RULES = frozenset({"product", "same", "affine", "fixed", "radiometer"})
_DISPOSITIONS = frozenset({"fixed", "contextual", "open", "structural"})
_RESOLVERS = frozenset({"latent", "prediction", "model_input", "resource", "outer"})
_UNIT_POLICIES = frozenset({"required", "optional", "inherited", "forbidden"})


def _normal(items: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return _canonical_components(tuple(items.items()))


def _combine(
    left: tuple[tuple[str, int], ...],
    right: tuple[tuple[str, int], ...],
    factor: int,
) -> tuple[tuple[str, int], ...]:
    merged = Counter()
    merged.update(dict(left))
    for name, exponent in right:
        merged[name] += factor * exponent
    return _normal(merged)


def multiply(left: DimensionSignature, right: DimensionSignature) -> DimensionSignature:
    """Multiply two normalized signatures, cancelling zero exponents."""
    return DimensionSignature(
        _combine(left.physical, right.physical, 1),
        _combine(left.quantity, right.quantity, 1),
    )


def divide(left: DimensionSignature, right: DimensionSignature) -> DimensionSignature:
    """Divide two normalized signatures, cancelling zero exponents."""
    return DimensionSignature(
        _combine(left.physical, right.physical, -1),
        _combine(left.quantity, right.quantity, -1),
    )


def power(value: DimensionSignature, exponent: int) -> DimensionSignature:
    """Raise a signature to a non-fuzzy integer exponent."""
    if not isinstance(exponent, int) or isinstance(exponent, bool):
        raise ConfigError(
            f"dimensions: an integer exponent is required; got {exponent!r} "
            f"({type(exponent).__name__})."
        )
    return DimensionSignature(
        _normal({name: amount * exponent for name, amount in value.physical}),
        _normal({name: amount * exponent for name, amount in value.quantity}),
    )


def dimension_of(unit: Unit | str) -> DimensionSignature:
    """Compute a signature from the existing six-field :class:`Unit`."""
    parsed = canonical_unit(unit) if isinstance(unit, str) else unit
    result = DimensionSignature(())
    for atom in parsed.numerator:
        result = multiply(result, _ATOM_SIGNATURES[atom])
    for atom in parsed.denominator:
        result = divide(result, _ATOM_SIGNATURES[atom])
    return result


def signature(token: str) -> DimensionSignature:
    """Parse one accepted unit spelling into a normalized signature."""
    return dimension_of(canonical_unit(token))


def _valid_selector(selector: str) -> bool:
    return bool(selector) and all(_SEGMENT.fullmatch(part) for part in selector.split("."))


def _validate_spec(spec: DimensionSpec) -> None:
    if spec.disposition not in _DISPOSITIONS:
        raise ConfigError(f"dimensions: unknown disposition {spec.disposition!r}")
    if spec.unit_policy not in _UNIT_POLICIES:
        raise ConfigError(f"dimensions: unknown unit policy {spec.unit_policy!r}")
    if spec.disposition == "fixed":
        if spec.signature is None:
            raise ConfigError("dimensions: a fixed specification requires a signature")
        if spec.resolver is not None:
            raise ConfigError("dimensions: a fixed specification cannot have a resolver")
    if spec.disposition == "contextual":
        if spec.resolver not in _RESOLVERS:
            raise ConfigError("dimensions: a contextual specification requires its resolver")
        if spec.signature is not None:
            raise ConfigError("dimensions: a contextual specification cannot be fixed")
    if spec.disposition in ("open", "structural") and spec.signature is not None:
        raise ConfigError(
            f"dimensions: {spec.disposition} data cannot be called dimensionless or fixed"
        )
    if spec.disposition in ("open", "structural") and spec.resolver is not None:
        raise ConfigError(f"dimensions: {spec.disposition} data cannot have a resolver")
    if spec.disposition == "structural" and spec.unit_policy != "forbidden":
        raise ConfigError("dimensions: structural data has forbidden unit policy")


def build_dimension_spec(
    *,
    dimension: str | None,
    disposition: DimensionDisposition,
    resolver: ContextualResolver | None,
    unit_policy: UnitPolicy | None,
    reason: str | None,
) -> DimensionSpec:
    if unit_policy is None:
        unit_policy = "forbidden" if disposition == "structural" else "optional"
    spec = DimensionSpec(
        disposition=disposition,
        signature=signature(dimension) if dimension is not None else None,
        resolver=resolver,
        unit_policy=unit_policy,
        reason=reason,
    )
    _validate_spec(spec)
    return spec


def register_dimension_spec(selector: DimensionSelector, spec: DimensionSpec) -> None:
    if selector.domain not in ("config_path", "model_field", "resource_field"):
        raise ConfigError(f"dimensions: invalid selector domain {selector.domain!r}")
    if not _valid_selector(selector.selector):
        raise ConfigError(
            f"dimensions: invalid {selector.domain} selector {selector.selector!r}; "
            "use dotted identifiers with '*' for one mapping segment and '[]' "
            "for one list index."
        )
    _validate_spec(spec)
    if selector in _DIMENSION_REGISTRY:
        raise ConfigError(
            f"dimensions: {selector.domain} selector {selector.selector!r} was registered twice"
        )
    _DIMENSION_REGISTRY[selector] = spec


def register_dimension(
    selector: str,
    *,
    domain: DimensionDomain,
    dimension: str | None = None,
    disposition: DimensionDisposition = "fixed",
    resolver: ContextualResolver | None = None,
    unit_policy: UnitPolicy | None = None,
    reason: str | None = None,
) -> None:
    register_dimension_spec(
        DimensionSelector(domain, selector),
        build_dimension_spec(
            dimension=dimension,
            disposition=disposition,
            resolver=resolver,
            unit_policy=unit_policy,
            reason=reason,
        ),
    )


def _segment_matches(pattern: str, actual: str) -> bool:
    if pattern == "*":
        return bool(actual) and "." not in actual and "[" not in actual
    if pattern.endswith("[]"):
        stem = re.escape(pattern[:-2])
        return re.fullmatch(rf"{stem}(?:\[\d+\]|\[\])", actual) is not None
    return pattern == actual


def _selector_matches(pattern: str, actual: str) -> bool:
    wanted = pattern.split(".")
    found = actual.split(".")
    return len(wanted) == len(found) and all(
        _segment_matches(left, right) for left, right in zip(wanted, found, strict=True)
    )


def registered_dimension_rows() -> tuple[tuple[DimensionSelector, DimensionSpec], ...]:
    """A stable snapshot for A9 and independent completeness censuses."""
    active = _ACTIVE_REGISTRY_SNAPSHOT.get()
    return tuple(_DIMENSION_REGISTRY.items()) if active is None else active.rows


def matching_dimension_rows(
    domain: DimensionDomain, selector: str
) -> tuple[tuple[DimensionSelector, DimensionSpec], ...]:
    """Every live exact/wildcard row matching one destination."""
    active = _ACTIVE_REGISTRY_SNAPSHOT.get()
    key = (domain, selector)
    if active is not None and key in active.matches:
        return active.matches[key]
    rows = tuple(
        (registered, spec)
        for registered, spec in registered_dimension_rows()
        if registered.domain == domain and _selector_matches(registered.selector, selector)
    )
    if active is not None:
        active.matches[key] = rows
    return rows


@contextmanager
def using_dimension_registry_snapshot():
    """Bound selector matching to one isolated preflight-pass snapshot."""
    snapshot = _RegistrySnapshot(tuple(_DIMENSION_REGISTRY.items()))
    token = _ACTIVE_REGISTRY_SNAPSHOT.set(snapshot)
    try:
        yield
    finally:
        _ACTIVE_REGISTRY_SNAPSHOT.reset(token)


def dimension_spec_for(
    descriptor_or_domain: DestinationDescriptor | DimensionDomain,
    selector: str | None = None,
) -> DimensionSpec:
    """Return the sole exact/wildcard match, refusing absence and ambiguity."""
    if isinstance(descriptor_or_domain, DestinationDescriptor):
        domain = descriptor_or_domain.domain
        actual = descriptor_or_domain.selector
        where = descriptor_or_domain.document_path
    else:
        domain = descriptor_or_domain
        if selector is None:
            raise TypeError("selector is required with a domain")
        actual = selector
        where = selector
    matches = [spec for _, spec in matching_dimension_rows(domain, actual)]
    if not matches:
        raise ConfigError(
            f"dimensions: no dimension selector matches {domain} destination {where!r}"
        )
    if len(matches) != 1:
        raise ConfigError(
            f"dimensions: ambiguous dimension selectors match {domain} destination {where!r}"
        )
    return matches[0]


def _contextual_signature(
    spec: DimensionSpec,
    descriptor: DestinationDescriptor,
    environment: DimensionEnvironment | None,
    outer: DimensionSignature | None,
) -> DimensionSignature | None:
    if spec.resolver == "outer":
        return outer
    if environment is None:
        return None
    if spec.resolver == "prediction":
        return environment.prediction_dimension
    if spec.resolver == "model_input":
        return environment.model_input_dimension
    if spec.resolver == "latent":
        latent = descriptor.document_path.split(".")
        return next(
            (
                environment.latent_dimensions[name]
                for name in latent
                if name in environment.latent_dimensions
            ),
            None,
        )
    if spec.resolver == "resource":
        return environment.resource_dimensions.get(descriptor.document_path)
    return None


def dimension_for(
    descriptor: DestinationDescriptor,
    environment: DimensionEnvironment | None = None,
    *,
    outer: DimensionSignature | None = None,
) -> DimensionSignature | None:
    spec = dimension_spec_for(descriptor)
    if spec.disposition == "fixed":
        return spec.signature
    if spec.disposition == "contextual":
        return _contextual_signature(spec, descriptor, environment, outer)
    return None


def register_formula_checked(registration: FormulaRegistration) -> None:
    if not registration.name or not registration.name.strip():
        raise ConfigError("dimensions: a formula requires a non-empty name")
    if registration.name in _FORMULA_REGISTRY:
        raise ConfigError(f"dimensions: formula {registration.name!r} was registered twice")
    if registration.rule not in _RULES:
        raise ConfigError(f"dimensions: formula rule {registration.rule!r} is not closed")
    if registration.result.disposition in ("open", "structural"):
        raise ConfigError("dimensions: a formula result cannot be open or structural")
    _validate_spec(registration.result)
    roles = [operand.role for operand in registration.operands]
    if any(not role or not role.strip() for role in roles) or len(roles) != len(set(roles)):
        raise ConfigError("dimensions: formula operands require non-empty unique roles")
    for operand in registration.operands:
        _validate_spec(operand.spec)
        if (
            not isinstance(operand.exponent, int)
            or isinstance(operand.exponent, bool)
            or operand.exponent == 0
        ):
            raise ConfigError(
                "dimensions: every formula operand requires a non-zero integer exponent"
            )
    if any(not producer or not producer.strip() for producer in registration.producers) or len(
        registration.producers
    ) != len(set(registration.producers)):
        raise ConfigError("dimensions: formula producers must be non-empty and unique")
    by_role = {operand.role: operand for operand in registration.operands}
    if registration.rule == "fixed" and registration.result.disposition != "fixed":
        raise ConfigError("dimensions: the fixed rule requires a fixed result")
    if registration.rule == "affine":
        if set(by_role) != {"value", "scale", "offset"}:
            raise ConfigError("dimensions: affine requires value, scale, and offset roles")
        scale = by_role["scale"]
        if scale.spec.disposition != "fixed" or scale.spec.signature != signature(
            "dimensionless"
        ):
            raise ConfigError("dimensions: affine scale must be fixed dimensionless")
        if (
            registration.result.disposition != "contextual"
            or registration.result.resolver != "outer"
        ):
            raise ConfigError("dimensions: affine result must be contextual outer D")
        for role in ("value", "offset"):
            spec = by_role[role].spec
            if spec.disposition != "contextual" or spec.resolver != "outer":
                raise ConfigError(f"dimensions: affine {role} must be contextual outer D")
    if registration.rule == "radiometer":
        expected = {
            "channel_width": signature("Hz"),
            "integration_time": signature("s"),
        }
        if (
            registration.result.disposition != "fixed"
            or registration.result.signature != signature("dimensionless")
        ):
            raise ConfigError("dimensions: radiometer requires a fixed dimensionless result")
        if set(by_role) != set(expected):
            raise ConfigError(
                "dimensions: radiometer requires channel_width and integration_time roles"
            )
        for role, wanted in expected.items():
            operand = by_role[role]
            if (
                operand.spec.disposition != "fixed"
                or operand.spec.signature != wanted
                or operand.exponent != 1
            ):
                raise ConfigError(
                    f"dimensions: radiometer role {role!r} requires {wanted} at exponent 1"
                )
    if registration.rule == "product":
        fixed_signatures = [
            operand.spec.signature
            for operand in registration.operands
            if operand.spec.disposition == "fixed"
        ]
        if any(value is not None and value.quantity for value in fixed_signatures):
            raise ConfigError(
                "dimensions: ordinary product formulas cannot combine quantity tags"
            )
    for existing in _FORMULA_REGISTRY.values():
        for producer in set(existing.producers) & set(registration.producers):
            existing_roles = {operand.role: operand.spec for operand in existing.operands}
            for role in by_role:
                if role in existing_roles:
                    raise ConfigError(
                        f"dimensions: producer {producer!r} role {role!r} is registered "
                        f"by both {existing.name!r} and {registration.name!r}"
                    )
    _FORMULA_REGISTRY[registration.name] = registration


def register_dimension_formula(
    name: str,
    *,
    rule: FormulaRule,
    result: DimensionSpec,
    operands: Sequence[FormulaOperand],
    producers: Sequence[str] = (),
) -> None:
    register_formula_checked(
        FormulaRegistration(name, rule, result, tuple(operands), tuple(producers))
    )


def _one(
    value: DimensionSignature | None, role: str, spec: DimensionSpec
) -> DimensionSignature | None:
    if isinstance(value, DimensionSignature):
        return value
    if value is None and spec.disposition in ("open", "structural"):
        return None
    raise ConfigError(f"dimensions: formula role {role!r} requires exactly one signature")


def _values(value, operand: FormulaOperand) -> tuple[DimensionSignature | None, ...]:
    role = operand.role
    if role.endswith(("[]", ".*")):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(_one(item, role, operand.spec) for item in value)
        raise ConfigError(f"dimensions: formula role {role!r} requires an ordered sequence")
    return (_one(value, role, operand.spec),)


def _expect(role: str, actual: DimensionSignature | None, spec: DimensionSpec) -> None:
    if actual is None:
        return
    if spec.disposition == "fixed" and actual != spec.signature:
        raise ConfigError(
            f"dimensions: formula role {role!r} has {actual}, expected {spec.signature}"
        )


def evaluate_formula(
    name: str,
    values: Mapping[
        str, DimensionSignature | None | Sequence[DimensionSignature | None]
    ],
    *,
    result: DimensionSignature | None = None,
) -> DimensionSignature | None:
    """Evaluate one finite named formula under its closed rule."""
    try:
        formula = _FORMULA_REGISTRY[name]
    except KeyError as error:
        raise ConfigError(f"dimensions: no formula named {name!r}") from error
    declared = {operand.role for operand in formula.operands}
    if set(values) != declared:
        raise ConfigError(
            f"dimensions: formula {name!r} requires roles {sorted(declared)}; got {sorted(values)}"
        )
    if (
        result is not None
        and formula.result.disposition == "fixed"
        and result != formula.result.signature
    ):
        raise ConfigError(
            f"dimensions: formula {name!r} result {result} contradicts registered "
            f"result {formula.result.signature}"
        )
    resolved_result = formula.result.signature if formula.result.disposition == "fixed" else result
    expanded: dict[str, tuple[DimensionSignature | None, ...]] = {}
    for operand in formula.operands:
        signatures = _values(values[operand.role], operand)
        for actual in signatures:
            _expect(operand.role, actual, operand.spec)
        expanded[operand.role] = signatures
    if formula.rule == "product":
        product = DimensionSignature(())
        for operand in formula.operands:
            for actual in expanded[operand.role]:
                if actual is None:
                    raise ConfigError(
                        f"dimensions: product role {operand.role!r} has unknown dimension"
                    )
                if actual.quantity:
                    raise ConfigError(
                        "dimensions: ordinary product formulas cannot combine quantity tags"
                    )
                product = multiply(product, power(actual, operand.exponent))
        if resolved_result is not None and product != resolved_result:
            raise ConfigError(
                f"dimensions: formula {name!r} produces {product}, expected {resolved_result}"
            )
        return product
    if formula.rule == "same":
        peers = [
            actual
            for operand in formula.operands
            if operand.spec.disposition == "contextual"
            for actual in expanded[operand.role]
            if actual is not None
        ]
        if resolved_result is not None:
            peers.append(resolved_result)
        if peers and any(peer != peers[0] for peer in peers[1:]):
            raise ConfigError(f"dimensions: formula {name!r} requires one shared dimension")
        return peers[0] if peers else resolved_result
    if formula.rule == "affine":
        if declared != {"value", "scale", "offset"}:
            raise ConfigError("dimensions: affine requires value, scale, and offset roles")
        scale = expanded["scale"][0]
        if scale != signature("dimensionless"):
            raise ConfigError("dimensions: affine scale must be dimensionless")
        peers = [expanded["value"][0], expanded["offset"][0]]
        if any(peer is None for peer in peers):
            raise ConfigError("dimensions: affine value and offset require known dimensions")
        if resolved_result is not None:
            peers.append(resolved_result)
        if any(peer != peers[0] for peer in peers[1:]):
            raise ConfigError("dimensions: affine value, offset, and result must share D")
        return peers[0]
    if formula.rule == "radiometer":
        if declared != {"channel_width", "integration_time"}:
            raise ConfigError("dimensions: radiometer requires channel_width and integration_time")
        return signature("dimensionless")
    return resolved_result


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
    if (
        not isinstance(target, str)
        or target.count(":") != 1
        or not all(target.split(":"))
    ):
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
            entries = list(
                _compose_stage_specs(raw, node_spec.kind, str(node_id))
            )
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
        formula
        for formula in _FORMULA_REGISTRY.values()
        if qualified in formula.producers
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
    input_operand = next(
        (operand for operand in formula.operands if operand.role == "input"), None
    )
    operator_input = incoming
    if (
        operator_input is None
        and input_operand is not None
        and input_operand.spec.disposition == "fixed"
    ):
        operator_input = input_operand.spec.signature
    output = (
        formula.result.signature
        if formula.result.disposition == "fixed"
        else operator_input
    )
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
    candidates: dict[str, list[DimensionSignature]] = {
        str(name): [] for name in parameters
    }
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
    selected, incomplete = _selected_model_classes(
        effective_document, table=table
    )
    model = effective_document.get("model")
    if isinstance(model, Mapping) and model.get("kind", "graph") == "pipeline":
        model_input, prediction = _pipeline_dimensions(model, table)
    elif incomplete:
        model_input, prediction = None, None
    else:
        model_input, prediction = _graph_dimensions(selected)
    candidates = _latent_candidates(effective_document, selected)
    environment = DimensionEnvironment(
        latent_dimensions={
            name: values[0] for name, values in candidates.items() if values
        },
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


def dimension_environment_for(
    effective_document: Mapping[str, object],
) -> DimensionEnvironment:
    """Infer selected graph/plugin, binding, and latent signatures without I/O."""
    return dimension_environment_and_conflicts_for(effective_document)[0]


def current_dimension_environment() -> DimensionEnvironment:
    """The layer-scoped environment, or a fresh utility-context default."""
    return _ACTIVE_ENVIRONMENT.get() or DimensionEnvironment()


@contextmanager
def using_dimension_environment(environment: DimensionEnvironment):
    """Make one inferred environment available to A9 and construction."""
    token = _ACTIVE_ENVIRONMENT.set(environment)
    try:
        yield
    finally:
        _ACTIVE_ENVIRONMENT.reset(token)


def bind_resource_dimension(
    environment: DimensionEnvironment,
    dotted_name: str,
    signature: DimensionSignature | None,
) -> None:
    if dotted_name in environment.resource_dimensions:
        raise ConfigError(f"dimensions: resource {dotted_name!r} was bound more than once")
    environment.resource_dimensions[dotted_name] = signature


def signature_label(value: DimensionSignature) -> str:
    """Compact human label used by A9 diagnostics."""
    if not value.physical and not value.quantity:
        return "dimensionless"
    if value.quantity and not value.physical and len(value.quantity) == 1:
        name, exponent = value.quantity[0]
        return name if exponent == 1 else f"{name}^{exponent}"
    if len(value.physical) == 1 and not value.quantity:
        name, exponent = value.physical[0]
        return name if exponent == 1 else f"{name}^{exponent}"
    return repr(value)


def describe_signature(value: DimensionSignature) -> str:
    """A signature as a refusal should say it: ``"a length (m)"``.

    The signature already carries the dimension's NAME -- ``length``,
    ``impedance`` -- so this invents nothing; before it existed, a refusal
    printed the dataclass and a user reading ``requires
    DimensionSignature(physical=(('length', 1),), quantity=())`` learned only
    that something was wrong.

    One dimension at exponent one is named with its article, which is the
    wording ``kinds/s_params.py`` was already using where it could raise its
    own refusal. Anything else -- a quotient, or the dimensionless and
    counting signatures that have no physical name at all -- is given as the
    token, because ``a adc_count per temperature`` reads worse than
    ``adc_count/K`` and the token is what the user has to type.
    """
    components = (*value.physical, *value.quantity)
    if len(components) != 1 or components[0][1] != 1:
        return signature_token(value)
    name = components[0][0]
    article = "an" if name[0] in "aeiou" else "a"
    return f"{article} {name} ({signature_token(value)})"


def signature_token(value: DimensionSignature) -> str:
    """The accepted canonical spelling for a catalog signature."""
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
    ):
        if signature(token) == value:
            return token
    return repr(value)
