"""The closed A9 registries, and what may be registered in them.

A dimension spec, a formula and the selectors that match them to a document's
nodes. Registration is checked on the way in rather than on the way out, so a
registry that accepted a row is a registry whose rows are all valid.

The registries and the active snapshot are MODULE-LEVEL STATE and there is one
of each; `using_dimension_registry_snapshot` is the only supported way to swap
them, and it restores.
"""

from __future__ import annotations

import dataclasses
import re
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal, TypeAlias

from _rheplicant_bootstrap.types import DestinationDescriptor, DimensionDomain
from rheplicant.config.errors import ConfigError

from .dimension_algebra import (
    DimensionSignature,
    multiply,
    power,
    signature,
)

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

_DIMENSION_REGISTRY: dict[DimensionSelector, DimensionSpec] = {}

_FORMULA_REGISTRY: dict[str, FormulaRegistration] = {}


@dataclass(slots=True)
class _RegistrySnapshot:
    rows: tuple[tuple[DimensionSelector, DimensionSpec], ...]
    matches: dict[
        tuple[DimensionDomain, str],
        tuple[tuple[DimensionSelector, DimensionSpec], ...],
    ] = dataclasses.field(default_factory=dict)


_ACTIVE_REGISTRY_SNAPSHOT: ContextVar[_RegistrySnapshot | None] = ContextVar(
    "rheplicant_dimension_registry_snapshot", default=None
)

_SEGMENT = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*(?:\[\])?|\*)\Z")

_RULES = frozenset({"product", "same", "affine", "fixed", "radiometer"})

_DISPOSITIONS = frozenset({"fixed", "contextual", "open", "structural"})

_RESOLVERS = frozenset({"latent", "prediction", "model_input", "resource", "outer"})

_UNIT_POLICIES = frozenset({"required", "optional", "inherited", "forbidden"})


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
        if scale.spec.disposition != "fixed" or scale.spec.signature != signature("dimensionless"):
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
        if registration.result.disposition != "fixed" or registration.result.signature != signature(
            "dimensionless"
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
            raise ConfigError("dimensions: ordinary product formulas cannot combine quantity tags")
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
    values: Mapping[str, DimensionSignature | None | Sequence[DimensionSignature | None]],
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
