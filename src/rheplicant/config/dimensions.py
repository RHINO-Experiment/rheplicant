"""Normalized dimension signatures and the closed A9 registries.

This module is now a FACADE over the four it was split into, plus the query
API that reads a resolved environment:

* :mod:`.dimension_algebra` -- what a signature is, and the four operations
* :mod:`.dimension_registry` -- the closed A9 registries and what may go in
* :mod:`.dimension_environment` -- a resolved document's dimensions, and the
  one active-environment slot
* :mod:`.dimension_inference` -- reading dimensions off the operators a
  document places

Thirteen modules and eight test files import names from here by this module's
name, so every public name it used to define is re-exported rather than
relocated. New code should import from the module that defines what it wants;
this name stays because breaking thirteen importers to tidy an import line is
not a trade worth making.
"""

from __future__ import annotations

from collections.abc import Mapping

from _rheplicant_bootstrap.types import DestinationDescriptor

from .dimension_algebra import (  # noqa: F401  # re-exported: this module is a facade, see its docstring
    DimensionSignature,
    PhysicalDimension,
    QuantitySignature,
    describe_signature,
    dimension_of,
    divide,
    multiply,
    power,
    signature,
    signature_label,
    signature_token,
)
from .dimension_environment import (  # noqa: F401  # re-exported: this module is a facade, see its docstring
    DimensionEnvironment,
    bind_resource_dimension,
    current_dimension_environment,
    using_dimension_environment,
)
from .dimension_inference import (  # noqa: F401  # re-exported: this module is a facade, see its docstring
    _compose_stage_specs,
    _constructible_operator_class,
    _loaded_operator_target,
    _pipeline_stage_specs,
    _plugin_formulas_for_class,
    dimension_environment_and_conflicts_for,
    latent_dimension_conflicts_for,
    operator_table,
)
from .dimension_registry import (  # noqa: F401  # re-exported: this module is a facade, see its docstring
    _DIMENSION_REGISTRY,
    _FORMULA_REGISTRY,
    ContextualResolver,
    DimensionDisposition,
    DimensionSelector,
    DimensionSpec,
    FormulaOperand,
    FormulaRegistration,
    FormulaRule,
    OperatorFormulaBinding,
    UnitPolicy,
    _selector_matches,
    build_dimension_spec,
    dimension_spec_for,
    evaluate_formula,
    matching_dimension_rows,
    register_dimension,
    register_dimension_formula,
    register_dimension_spec,
    register_formula_checked,
    registered_dimension_rows,
    using_dimension_registry_snapshot,
)


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


def dimension_environment_for(
    effective_document: Mapping[str, object],
) -> DimensionEnvironment:
    """Infer selected graph/plugin, binding, and latent signatures without I/O."""
    return dimension_environment_and_conflicts_for(effective_document)[0]
