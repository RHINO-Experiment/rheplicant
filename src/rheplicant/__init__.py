"""rheplicant: a differentiable scientific pipeline framework built on JAX + Equinox.

Core principle: **everything is an Operator acting on a State.**

- ``rheplicant.core`` — domain-agnostic State / Operator / Pipeline abstractions.
- ``rheplicant.radio`` — generic single-antenna radio telescope operators
  (placeholder physics; RHINO is the eventual target instrument).
- ``rheplicant.inference`` — likelihood / calibration layer, separate from forward models.
"""

from rheplicant.core import (
    AbstractOperator,
    Assembly,
    AssemblyError,
    At,
    Coordinates,
    DirtError,
    Environment,
    FrozenMapping,
    LambdaOperator,
    MissingKeyError,
    Pipeline,
    PipelineError,
    SelectOperator,
    SignalGraph,
    SnapshotOperator,
    State,
    StateValidationError,
    SumOperator,
)

__version__ = "0.1.0"

__all__ = [
    "AbstractOperator",
    "Assembly",
    "AssemblyError",
    "At",
    "SignalGraph",
    "Coordinates",
    "Environment",
    "DirtError",
    "FrozenMapping",
    "LambdaOperator",
    "MissingKeyError",
    "Pipeline",
    "SnapshotOperator",
    "PipelineError",
    "SelectOperator",
    "State",
    "StateValidationError",
    "SumOperator",
    "__version__",
]
