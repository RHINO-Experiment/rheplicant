"""What dimensions are currently in force, and the type that says so.

A `DimensionEnvironment` is the resolved answer for one document: every node's
signature, plus the resource bindings that were declared rather than inferred.

It lives below both the inference that PRODUCES one and the query API that
reads one, because putting the type beside either made those two import each
other. `_ACTIVE_ENVIRONMENT` is module-level state and there is exactly one;
`using_dimension_environment` is the only supported way to change it, and it
restores.
"""

from __future__ import annotations

import dataclasses
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

from rheplicant.config.errors import ConfigError

from .dimension_algebra import (  # noqa: F401  # re-exported: importers name this module
    DimensionSignature,
    describe_signature,
    dimension_of,
    divide,
    multiply,
    power,
    signature_label,
    signature_token,
)
from .dimension_registry import (  # noqa: F401  # re-exported: importers name this module
    _DIMENSION_REGISTRY,
    _FORMULA_REGISTRY,
    DimensionSpec,
    FormulaOperand,
    FormulaRegistration,
    FormulaRule,
    OperatorFormulaBinding,
    _selector_matches,
    dimension_spec_for,
    evaluate_formula,
    matching_dimension_rows,
    register_dimension,
    register_dimension_formula,
    register_formula_checked,
    registered_dimension_rows,
    using_dimension_registry_snapshot,
)


@dataclass(slots=True)
class DimensionEnvironment:
    latent_dimensions: dict[str, DimensionSignature] = dataclasses.field(default_factory=dict)
    resource_dimensions: dict[str, DimensionSignature | None] = dataclasses.field(
        default_factory=dict
    )
    prediction_dimension: DimensionSignature | None = None
    model_input_dimension: DimensionSignature | None = None


_ACTIVE_ENVIRONMENT: ContextVar[DimensionEnvironment | None] = ContextVar(
    "rheplicant_dimension_environment", default=None
)


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
