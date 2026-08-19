"""Normalized dimension signatures and the closed A9 registries."""

from __future__ import annotations

import dataclasses
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, TypeAlias

from _rheplicant_bootstrap.types import DestinationDescriptor, DimensionDomain
from rheplicant.config.errors import ConfigError
from rheplicant.config.units import Unit, canonical_unit

PhysicalDimension = tuple[tuple[str, int], ...]
QuantitySignature = tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class DimensionSignature:
    physical: PhysicalDimension
    quantity: QuantitySignature = ()


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
_ACTIVE_ENVIRONMENT: ContextVar[DimensionEnvironment | None] = ContextVar(
    "rheplicant_dimension_environment", default=None
)

_SEGMENT = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\[\])?|\*)\Z")
_RULES = frozenset({"product", "same", "affine", "fixed", "radiometer"})
_DISPOSITIONS = frozenset({"fixed", "contextual", "open", "structural"})
_RESOLVERS = frozenset({"latent", "prediction", "model_input", "resource", "outer"})
_UNIT_POLICIES = frozenset({"required", "optional", "inherited", "forbidden"})


def _normal(items: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((name, exponent) for name, exponent in items.items() if exponent))


def _combine(
    left: tuple[tuple[str, int], ...],
    right: tuple[tuple[str, int], ...],
    factor: int,
) -> tuple[tuple[str, int], ...]:
    merged = Counter(dict(left))
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
    matches = [
        spec
        for registered, spec in _DIMENSION_REGISTRY.items()
        if registered.domain == domain and _selector_matches(registered.selector, actual)
    ]
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


def _one(value, role: str) -> DimensionSignature:
    if isinstance(value, DimensionSignature):
        return value
    raise ConfigError(f"dimensions: formula role {role!r} requires exactly one signature")


def _values(value, role: str) -> tuple[DimensionSignature, ...]:
    if role.endswith(("[]", ".*")):
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return tuple(_one(item, role) for item in value)
        raise ConfigError(f"dimensions: formula role {role!r} requires an ordered sequence")
    return (_one(value, role),)


def _expect(role: str, actual: DimensionSignature, spec: DimensionSpec) -> None:
    if spec.disposition == "fixed" and actual != spec.signature:
        raise ConfigError(
            f"dimensions: formula role {role!r} has {actual}, expected {spec.signature}"
        )


def evaluate_formula(
    name: str,
    values: Mapping[str, DimensionSignature | Sequence[DimensionSignature]],
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
    resolved_result = formula.result.signature if formula.result.disposition == "fixed" else result
    expanded: dict[str, tuple[DimensionSignature, ...]] = {}
    for operand in formula.operands:
        signatures = _values(values[operand.role], operand.role)
        for actual in signatures:
            _expect(operand.role, actual, operand.spec)
        expanded[operand.role] = signatures
    if formula.rule == "product":
        product = DimensionSignature(())
        for operand in formula.operands:
            for actual in expanded[operand.role]:
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


def dimension_environment_for(
    effective_document: Mapping[str, object],
) -> DimensionEnvironment:
    """Infer declared latent and trunk signatures without I/O."""
    environment = DimensionEnvironment()
    inference = effective_document.get("inference")
    if isinstance(inference, Mapping):
        parameters = inference.get("parameters")
        if isinstance(parameters, Mapping):
            for name, declaration in parameters.items():
                if not isinstance(declaration, Mapping):
                    continue
                for key in ("init", "ref"):
                    node = declaration.get(key)
                    if isinstance(node, Mapping) and isinstance(node.get("unit"), str):
                        environment.latent_dimensions[str(name)] = signature(node["unit"])
                        break
    environment.model_input_dimension = signature("K")
    model = effective_document.get("model")
    adc = model.get("adc") if isinstance(model, Mapping) else None
    environment.prediction_dimension = (
        signature("adc_count") if isinstance(adc, Mapping) else signature("K")
    )
    return environment


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
    if value.quantity and not value.physical and len(value.quantity) == 1:
        name, exponent = value.quantity[0]
        return name if exponent == 1 else f"{name}^{exponent}"
    if len(value.physical) == 1 and not value.quantity:
        name, exponent = value.physical[0]
        return name if exponent == 1 else f"{name}^{exponent}"
    return repr(value)
