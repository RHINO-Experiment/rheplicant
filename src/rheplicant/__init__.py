"""rheplicant: a differentiable scientific pipeline framework built on JAX + Equinox.

Core principle: **everything is an Operator acting on a State.**

- ``rheplicant.core`` — domain-agnostic State / Operator / Pipeline abstractions.
- ``rheplicant.radio`` — generic single-antenna radio telescope operators
  (placeholder physics; RHINO is the eventual target instrument).
- ``rheplicant.inference`` — likelihood / calibration layer, separate from forward models.
"""

from importlib.metadata import PackageNotFoundError, version

from rheplicant.core import (
    AbstractOperator,
    AmbiguousNodeError,
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

# Single source of truth is pyproject.toml; read it back from the installed
# distribution metadata rather than duplicating the string here.
try:
    __version__ = version("rheplicant")
except PackageNotFoundError:  # running from a source tree with no install
    __version__ = "0.0.0+unknown"

__all__ = [
    "AbstractOperator",
    "AmbiguousNodeError",
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
