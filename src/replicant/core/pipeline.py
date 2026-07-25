"""Pipeline: ordered composition of operators — itself an operator.

Because Pipeline satisfies the same ``State -> State`` contract as any other
operator, pipelines nest freely (composite pattern)::

    instrument = Pipeline(beam, receiver, names=("beam", "rx"))
    full = Pipeline(sky, instrument, backend)

Execution is a plain Python loop over heterogeneous stages: under ``jax.jit``
this unrolls into one fused computation, which is exactly right when every
stage is a different operator. (A ``lax.scan`` over a *homogeneous* stack of
identical operators is a different, complementary pattern — deliberately not
built here.)
"""

from collections.abc import Iterator, Sequence

import equinox as eqx

from replicant.core.errors import PipelineError
from replicant.core.operator import AbstractOperator
from replicant.core.state import State


def _auto_names(stages: Sequence[AbstractOperator]) -> tuple[str, ...]:
    """Derive stage names from class names: SkyOperator -> "sky", AddOne -> "addone".

    Duplicates get a 1-based occurrence suffix: ("gain", "gain_2", "gain_3").
    """
    bases = []
    for stage in stages:
        base = type(stage).__name__.lower()
        if base.endswith("operator") and base != "operator":
            base = base[: -len("operator")]
        bases.append(base)
    counts: dict[str, int] = {}
    names = []
    for base in bases:
        counts[base] = counts.get(base, 0) + 1
        names.append(base if counts[base] == 1 else f"{base}_{counts[base]}")
    return tuple(names)


def validate_operators(
    operators: Sequence[AbstractOperator], owner: str
) -> tuple[AbstractOperator, ...]:
    """Shared constructor validation for composite operators (Pipeline, SumOperator)."""
    if not operators:
        raise PipelineError(f"{owner} needs at least one operator.")
    for i, op in enumerate(operators):
        if not isinstance(op, AbstractOperator):
            raise PipelineError(
                f"{owner} operator {i} is {type(op).__name__}, not an AbstractOperator. "
                "Wrap plain functions with LambdaOperator."
            )
    return tuple(operators)


def resolve_names(
    operators: Sequence[AbstractOperator], names: Sequence[str] | None
) -> tuple[str, ...]:
    """Shared name resolution/validation for composite operators."""
    if names is None:
        resolved = _auto_names(operators)
    else:
        resolved = tuple(names)
        if len(resolved) != len(operators):
            raise PipelineError(f"Got {len(resolved)} names for {len(operators)} operators.")
        if not all(isinstance(n, str) for n in resolved):
            raise PipelineError("Operator names must be strings.")
    if len(set(resolved)) != len(resolved):
        raise PipelineError(f"Operator names must be unique, got {resolved}.")
    return resolved


class Pipeline(AbstractOperator):
    """An ordered, named composition of operators.

    Attributes:
        stages: the operators, applied first-to-last.
        names: unique stage names (static). Auto-derived from class names if
            not given; pass ``names=`` for stable, meaningful labels.

    Example::

        pipeline = Pipeline(
            SkyOperator(...), GainOperator(...), NoiseOperator(...),
            names=("sky", "gain", "noise"),
        )
        out = pipeline(state)
        gain_op = pipeline["gain"]
    """

    stages: tuple[AbstractOperator, ...]
    names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *stages: AbstractOperator,
        names: Sequence[str] | None = None,
    ):
        self.stages = validate_operators(stages, "Pipeline")
        self.names = resolve_names(stages, names)

    # -- execution -----------------------------------------------------------

    def __call__(self, state: State) -> State:
        for stage in self.stages:
            state = stage(state)
        return state

    def run_with_intermediates(self, state: State) -> tuple[State, tuple[State, ...]]:
        """Run the pipeline, also returning the state after *every* stage.

        Diagnostics tool: keeps all intermediate states in memory, so use on
        small problems, not inside large jitted optimization loops. This is a
        separate method (not a flag on ``__call__``) so the operator contract
        stays uniform and pipelines keep nesting cleanly.
        """
        intermediates = []
        for stage in self.stages:
            state = stage(state)
            intermediates.append(state)
        return state, tuple(intermediates)

    # -- access --------------------------------------------------------------

    def __getitem__(self, index: int | str) -> AbstractOperator:
        if isinstance(index, str):
            try:
                index = self.names.index(index)
            except ValueError:
                raise KeyError(
                    f"No stage named {index!r}; available: {self.names}"
                ) from None
        return self.stages[index]

    def __len__(self) -> int:
        return len(self.stages)

    def __iter__(self) -> Iterator[AbstractOperator]:
        return iter(self.stages)

    # -- functional updates --------------------------------------------------

    def replace_stage(self, index: int | str, operator: AbstractOperator) -> "Pipeline":
        """Return a new Pipeline with one stage swapped; names are preserved.

        (For surgical edits *inside* a stage — e.g. one parameter — use
        ``eqx.tree_at`` instead of rebuilding.)
        """
        if not isinstance(operator, AbstractOperator):
            raise PipelineError(
                f"Replacement must be an AbstractOperator, got {type(operator).__name__}."
            )
        if isinstance(index, str):
            try:
                index = self.names.index(index)
            except ValueError:
                raise KeyError(
                    f"No stage named {index!r}; available: {self.names}"
                ) from None
        new_stages = list(self.stages)
        new_stages[index] = operator
        return Pipeline(*new_stages, names=self.names)
