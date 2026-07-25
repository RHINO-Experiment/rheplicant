"""Combinators: parallel composition of operators.

:class:`~replicant.core.pipeline.Pipeline` composes operators *sequentially*;
:class:`SumOperator` composes them *in parallel*, summing their data
contributions. Together they express the typical structure of a physical
forward model, e.g. an antenna temperature assembled from independent
components::

    astro = Pipeline(
        SumOperator(global_signal, foregrounds, point_sources),
        ionosphere,                       # distorts the astrophysical sum
    )
    t_ant = SumOperator(astro, ground_pickup)

Both combinators are themselves operators, so they nest freely.
"""

from collections.abc import Iterator, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant.core.errors import PipelineError, StateValidationError
from replicant.core.operator import AbstractOperator
from replicant.core.pipeline import resolve_names, validate_operators
from replicant.core.state import State


def _tree_add(
    total: Any, contribution: Any, branch_name: str, combinator: str = "SumOperator"
) -> Any:
    """Leafwise sum of two data pytrees with loud structural/shape validation.

    Python ``+`` would concatenate tuples and fail on dicts; silent NumPy
    broadcasting would smear shape-mismatched contributions. Both are trace-time
    (static-shape) checks, so this stays jit-safe.
    """

    total_structure = jax.tree.structure(total)
    if total_structure != jax.tree.structure(contribution):
        raise PipelineError(
            f"{combinator} branch {branch_name!r} produced a data pytree whose structure "
            f"differs from the previous branches: {jax.tree.structure(contribution)} "
            f"vs accumulated {total_structure}."
        )

    def add(a: jax.Array, b: jax.Array) -> jax.Array:
        if jnp.shape(a) != jnp.shape(b):
            raise PipelineError(
                f"{combinator} branch {branch_name!r} produced a contribution of shape "
                f"{jnp.shape(b)}, which does not match the accumulated shape {jnp.shape(a)}. "
                "Branches must produce identical shapes on the shared coordinate grid "
                "(shape-changing operators like BackendOperator cannot be Sum branches)."
            )
        return a + b

    return jax.tree.map(add, total, contribution)


class SumOperator(AbstractOperator):
    """Run source-type branches on the same input state and sum their data.

    Semantics:
        * Every branch receives the *input* context (coords, env, meta) with
          ``data`` stripped to ``None`` — SumOperator is a *source* combinator
          whose branches each produce their own contribution on the shared
          coordinate grid. A branch that tries to read input data fails
          loudly instead of silently depending on caller state.
        * Branches must not change coords/env/meta; the output state carries
          the input context with ``data = sum(branch outputs)``.
        * PRNG: each branch gets its own subkey split off the main chain, so
          stochastic branches draw independent randomness and a single seed
          reproduces the whole sum. The output state carries the advanced key.

    Example::

        sky = SumOperator(
            GlobalSignalOperator(...), ForegroundOperator(...),
            names=("signal", "foregrounds"),
        )
    """

    branches: tuple[AbstractOperator, ...]
    names: tuple[str, ...] = eqx.field(static=True)

    def __init__(
        self,
        *branches: AbstractOperator,
        names: Sequence[str] | None = None,
    ):
        self.branches = validate_operators(branches, "SumOperator")
        self.names = resolve_names(branches, names)

    def __call__(self, state: State) -> State:
        total = None
        for name, branch in zip(self.names, self.branches, strict=True):
            if state.key is not None:
                branch_key, state = state.next_key()
                branch_in = state.replace(key=branch_key, data=None)
            else:
                branch_in = state.replace(data=None)
            contribution = branch(branch_in).data
            if contribution is None:
                raise PipelineError(
                    f"SumOperator branch {name!r} produced no data (state.data is None); "
                    "every branch must be a source-type operator."
                )
            total = contribution if total is None else _tree_add(total, contribution, name)
        return state.with_data(total)

    def __getitem__(self, index: int | str) -> AbstractOperator:
        if isinstance(index, str):
            try:
                index = self.names.index(index)
            except ValueError:
                raise KeyError(
                    f"No branch named {index!r}; available: {self.names}"
                ) from None
        return self.branches[index]

    def __len__(self) -> int:
        return len(self.branches)

    def __iter__(self) -> Iterator[AbstractOperator]:
        return iter(self.branches)

    def replace_branch(self, index: int | str, operator: AbstractOperator) -> "SumOperator":
        """Return a new SumOperator with one branch swapped; names preserved.

        Parity with :meth:`~replicant.core.pipeline.Pipeline.replace_stage`.
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
                    f"No branch named {index!r}; available: {self.names}"
                ) from None
        new_branches = list(self.branches)
        new_branches[index] = operator
        return SumOperator(*new_branches, names=self.names)


class SelectOperator(AbstractOperator):
    """Select between source-type branches per time sample (a switch).

    Models switched signal paths — e.g. calibration loads that REPLACE the
    antenna signal on a pre-defined switching cycle. Every branch runs on the
    input context (data stripped, per-branch PRNG subkeys, context writes
    discarded — exactly :class:`SumOperator`'s branch semantics), and the
    output at each time sample is the contribution of the branch selected by
    the switch state::

        data[t] = contribution[switch[t]][t]

    The switch state is observation configuration, not an operator parameter:
    it is read from ``state.coords.extra[switch_key]`` as an integer array of
    shape ``(n_time,)`` whose values index the branches **in branch order**
    (for graph-assembled selectors: the selector node's in-edge declaration
    order). Out-of-range values select nothing (that sample is zero).

    Attributes:
        branches: the selectable signal paths.
        names: unique branch names (static).
        switch_key: key into ``coords.extra`` holding the switch state
            (static; graph assembly sets it to the selector node id).
    """

    branches: tuple[AbstractOperator, ...]
    names: tuple[str, ...] = eqx.field(static=True)
    switch_key: str = eqx.field(static=True, default="switch_state")

    def __init__(
        self,
        *branches: AbstractOperator,
        names: Sequence[str] | None = None,
        switch_key: str = "switch_state",
    ):
        self.branches = validate_operators(branches, "SelectOperator")
        self.names = resolve_names(branches, names)
        self.switch_key = switch_key

    def _switch_state(self, state: State) -> jax.Array:
        if state.coords is None or state.coords.extra.get(self.switch_key) is None:
            raise StateValidationError(
                f"SelectOperator needs the switch state in "
                f'coords.extra["{self.switch_key}"] — an integer (n_time,) array '
                "whose values index the branches in branch order."
            )
        switch = state.coords.extra[self.switch_key]
        if not jnp.issubdtype(switch.dtype, jnp.integer) or switch.ndim != 1:
            raise StateValidationError(
                f'coords.extra["{self.switch_key}"] must be a 1D integer array, '
                f"got ndim={switch.ndim}, dtype={switch.dtype}."
            )
        return switch

    def __call__(self, state: State) -> State:
        switch = self._switch_state(state)
        total = None
        for index, (name, branch) in enumerate(zip(self.names, self.branches, strict=True)):
            if state.key is not None:
                branch_key, state = state.next_key()
                branch_in = state.replace(key=branch_key, data=None)
            else:
                branch_in = state.replace(data=None)
            contribution = branch(branch_in).data
            if contribution is None:
                raise PipelineError(
                    f"SelectOperator branch {name!r} produced no data (state.data is "
                    "None); every branch must be a source-type operator."
                )
            selected = switch == index

            def apply_mask(leaf, mask=selected, branch=name):
                if leaf.ndim < 1 or leaf.shape[0] != mask.shape[0]:
                    raise StateValidationError(
                        f"SelectOperator branch {branch!r} produced a leaf of shape "
                        f"{jnp.shape(leaf)}, but selection needs a leading time axis "
                        f"of length {mask.shape[0]} (the switch array length)."
                    )
                return leaf * mask.reshape((mask.shape[0],) + (1,) * (leaf.ndim - 1))

            masked = jax.tree.map(apply_mask, contribution)
            total = (
                masked
                if total is None
                else _tree_add(total, masked, name, combinator="SelectOperator")
            )
        return state.with_data(total)

    def __getitem__(self, index: int | str) -> AbstractOperator:
        if isinstance(index, str):
            try:
                index = self.names.index(index)
            except ValueError:
                raise KeyError(
                    f"No branch named {index!r}; available: {self.names}"
                ) from None
        return self.branches[index]

    def __len__(self) -> int:
        return len(self.branches)

    def __iter__(self) -> Iterator[AbstractOperator]:
        return iter(self.branches)
