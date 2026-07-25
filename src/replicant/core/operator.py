"""Operator: the universal transformation interface.

Everything in replicant follows one contract::

    state_out = operator(state_in)          # State -> State, pure

An operator is an ``equinox.Module``: its array-valued fields are traced
pytree leaves (and therefore differentiable parameters "for free"), while
non-array fields are static configuration. Select the trainable parameters
with the standard Equinox idiom::

    params, static = eqx.partition(op, eqx.is_inexact_array)

Design note: :class:`AbstractOperator` is an *interface* (one abstract
method), not a class hierarchy — there are deliberately no intermediate base
classes. Shared behaviour belongs in helper functions and composition
(:class:`~replicant.core.pipeline.Pipeline`), not inheritance.
"""

import abc
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx

from replicant.core.state import State


class AbstractOperator(eqx.Module):
    """A pure, differentiable transformation ``State -> State``.

    Attributes:
        requires: declarative contract (not enforced yet) — dotted State
            paths this operator reads, e.g. ``("data", "coords.freq", "key")``.
        provides: dotted State paths this operator writes, e.g. ``("data",)``.
        graph_node: home node on a SignalGraph template (assembly); ``None``
            means "place explicitly with At(node, op)".

    The requires/provides tuples are documentation today and the hook for a
    future ``pipeline.validate()`` static checker.

    Rules for implementors:
        * Never mutate the input state — return ``state.replace(...)`` /
          ``state.with_data(...)``.
        * Randomness must go through ``subkey, state = state.next_key()`` and
          the *advanced* state must be the one returned.
        * Only structural (shape/dtype) validation inside ``__call__`` —
          value checks would break under jit.
    """

    requires: ClassVar[tuple[str, ...]] = ()
    provides: ClassVar[tuple[str, ...]] = ()

    # Home node on a SignalGraph template (graph-guided assembly); resolved
    # through the MRO so subclasses inherit their base's slot. Documented in
    # the class docstring's Attributes section.
    graph_node: ClassVar[str | None] = None

    @abc.abstractmethod
    def __call__(self, state: State) -> State:
        """Apply this operator to a state, returning a new state."""


class LambdaOperator(AbstractOperator):
    """Wrap a pure function ``State -> State`` as an operator.

    The wrapped function is *static* (part of the module structure, not a
    traced leaf); it hashes by identity, so reuse the same LambdaOperator
    instance rather than re-creating identical lambdas if jit-cache hits
    matter.

    Example::

        clip = LambdaOperator.on_data(lambda d: jnp.clip(d, 0.0, 1.0))
    """

    fn: Callable[[State], State] = eqx.field(static=True)

    def __call__(self, state: State) -> State:
        return self.fn(state)

    @classmethod
    def on_data(cls, fn: Callable[[Any], Any]) -> "LambdaOperator":
        """Lift an ``Array -> Array`` (or pytree -> pytree) function onto ``state.data``."""
        return cls(fn=lambda state: state.with_data(fn(state.data)))


class SnapshotOperator(AbstractOperator):
    """Save the current data into ``aux["snapshot/<name>"]`` (zero-copy).

    Place at the start of a processing pipeline to preserve raw data through
    destructive steps (calibration application, filtering)::

        analysis = Pipeline(SnapshotOperator(name="raw"), apply_cal, sidereal_filter)
        raw = analysis(state).aux["snapshot/raw"]
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("aux",)

    name: str = eqx.field(static=True, default="raw")

    def __call__(self, state: State) -> State:
        return state.checkpoint(self.name)
