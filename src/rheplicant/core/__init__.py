"""Domain-agnostic core abstractions: State, Operator, Pipeline.

This subpackage must never import from ``rheplicant.radio`` or ``rheplicant.inference``,
so it can later be extracted as a standalone framework package.
"""

from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.environment import Environment
from rheplicant.core.errors import (
    DataIngestionError,
    DirtError,
    MissingKeyError,
    PipelineError,
    StateValidationError,
)
from rheplicant.core.frozen import FrozenMapping
from rheplicant.core.graph import (
    Assembly,
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
    get_graph,
    register_graph,
)
from rheplicant.core.operator import AbstractOperator, LambdaOperator, SnapshotOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State

__all__ = [
    "AbstractOperator",
    "Assembly",
    "AssemblyError",
    "At",
    "NodeSpec",
    "SignalGraph",
    "assemble",
    "get_graph",
    "register_graph",
    "Coordinates",
    "Environment",
    "DataIngestionError",
    "DirtError",
    "FrozenMapping",
    "LambdaOperator",
    "MissingKeyError",
    "Pipeline",
    "SnapshotOperator",
    "PipelineError",
    "State",
    "StateValidationError",
    "SelectOperator",
    "SumOperator",
]
