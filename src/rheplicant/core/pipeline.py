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

from rheplicant.core.errors import PipelineError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


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


def check_stage_ordering(
    stages: Sequence[AbstractOperator], names: Sequence[str]
) -> None:
    """Enforce ``must_precede`` against THIS SEQUENCE, for composition by hand.

    :func:`~rheplicant.core.graph.assemble` enforces the same declaration by
    reachability on a template, and for a while it was the only thing that did.
    The two routes compile to the same composition, so the refusal was one line
    of call site away from silence::

        assemble(..., At('noise', tone))   AssemblyError: 'bandpass' is not reachable
        Pipeline(sky, band, gain, tone)    no error; the tone's response is 1.0

    A tone injected downstream of the gain it is meant to track has a gain
    response of exactly 1.0 — it monitors nothing, which is the sentence
    ``must_precede_because`` exists to say — and the run converges and reports
    healthy diagnostics.

    **What this checks, and why it is weaker.** ``must_precede`` names nodes in
    a *graph's* vocabulary; a Pipeline is domain-agnostic and has only
    ``names``. So the question it can ask is sequence-local: *if a named stage
    is present here, it must come after me*. A stage that is not present is not
    a violation — the identical rule
    :func:`~rheplicant.core.graph._check_ordering` applies to a node that was
    never lit, and the reason is the same: an absent stage is one there is
    nothing to pass through, and refusing there would reject every partial
    model for the sake of a stage it never asked for.

    Three things it therefore cannot do, all of them pinned in
    ``tests/core/test_ordering.py`` rather than left to be discovered:

    * refuse a constraint naming something no stage is called. ``assemble``
      refuses an unknown node id, because a template has a node list and an
      unenforceable declaration is prose; a Pipeline has no such list, so a typo
      and a legitimately absent stage are the same observation here.
    * see into a nested composite. ``names`` is one level deep; a target inside
      a stage that is itself a Pipeline or a combinator is not a name this
      sequence has.
    * mean anything for the combinators. ``SumOperator`` and ``SelectOperator``
      run their branches in parallel on the same input, so "precede" is not a
      relation between two of them and they deliberately do not call this —
      which is why this takes no ``owner`` argument the way
      :func:`validate_operators` and :func:`resolve_names` do. A sequence is
      the only thing it has anything to say about.

    **Cost.** Called from ``Pipeline.__init__``, which equinox does NOT go
    through when it rebuilds a Module: ``tree_unflatten`` reconstructs directly,
    so a jit trace, a gradient and an ``eqx.tree_at`` edit re-run this zero
    times (measured). It runs where a human composes a pipeline, which is the
    only place the order can be chosen.
    """
    position = {name: i for i, name in enumerate(names)}
    for index, (name, stage) in enumerate(zip(names, stages, strict=True)):
        for target in stage.must_precede:
            at = position.get(target)
            if at is None or at >= index:
                continue
            because = f" {stage.must_precede_because}" if stage.must_precede_because else ""
            raise PipelineError(
                f"Pipeline stage {index} ({type(stage).__name__}, named {name!r}) "
                f"declares must_precede={list(stage.must_precede)}, but {target!r} "
                f"is stage {at} of this sequence — it runs BEFORE this one, so "
                f"nothing this operator contributes ever passes through {target!r}, "
                f"and whatever {target!r} does to the signal is absent from it."
                f"{because} Move it before {target!r}, or drop {target!r} if the "
                "stage genuinely is not there. (This is the sequence-local half of "
                "the constraint assemble() checks by reachability; a stage this "
                "sequence does not contain is not a violation.)"
            )


class Pipeline(AbstractOperator):
    """An ordered, named composition of operators.

    A stage whose physics depends on where it sits declares that with
    :attr:`~rheplicant.core.operator.AbstractOperator.must_precede`, and
    construction refuses an order that breaks it — sequence-locally, which is
    all a domain-agnostic sequence can say. See :func:`check_stage_ordering`
    for what that does and does not cover.

    Attributes:
        stages: the operators, applied first-to-last.
        names: unique stage names (static). Auto-derived from class names if
            not given; pass ``names=`` for stable, meaningful labels. They are
            also the vocabulary ``must_precede`` is read in, so a pipeline whose
            stages carry ordering constraints wants meaningful ones.

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
        # Screen first, then check the physics: what a stage is called has to be
        # settled before what it must precede can name anything.
        check_stage_ordering(self.stages, self.names)

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
