"""Domain-agnostic core abstractions: State, Operator, Pipeline.

This subpackage must never import from ``replicant.radio`` or ``replicant.inference``,
so it can later be extracted as a standalone framework package.
"""

from replicant.core.combinators import SelectOperator, SumOperator
from replicant.core.coordinates import Coordinates
from replicant.core.environment import Environment
from replicant.core.errors import (
    DirtError,
    MissingKeyError,
    PipelineError,
    StateValidationError,
)
from replicant.core.frozen import FrozenMapping
from replicant.core.graph import (
    Assembly,
    AssemblyError,
    At,
    NodeSpec,
    SignalGraph,
    assemble,
    get_graph,
    register_graph,
)
from replicant.core.operator import AbstractOperator, LambdaOperator, SnapshotOperator
from replicant.core.pipeline import Pipeline
from replicant.core.state import State

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
