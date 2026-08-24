"""Operator: the universal transformation interface.

Everything in rheplicant follows one contract::

    state_out = operator(state_in)          # State -> State, pure

An operator is an ``equinox.Module``: its array-valued fields are traced
pytree leaves (and therefore differentiable parameters "for free"), while
non-array fields are static configuration. Select the trainable parameters
with the standard Equinox idiom::

    params, static = eqx.partition(op, eqx.is_inexact_array)

Design note: :class:`AbstractOperator` is an *interface* (one abstract
method), not a class hierarchy — there are deliberately no intermediate base
classes. Shared behaviour belongs in helper functions and composition
(:class:`~rheplicant.core.pipeline.Pipeline`), not inheritance.
"""

import abc
from collections.abc import Callable
from typing import Any, ClassVar

import equinox as eqx

from rheplicant.core.state import State


class AbstractOperator(eqx.Module):
    """A pure, differentiable transformation ``State -> State``.

    Attributes:
        requires: dotted State paths this operator reads, e.g.
            ``("data", "coords.freq", "key")``. Mostly documentation — with one
            enforced member, ``"key"``: see below.
        provides: dotted State paths this operator writes, e.g. ``("data",)``.
            Documentation.
        graph_node: home node on a SignalGraph template (assembly); ``None``
            means "place explicitly with At(node, op)".
        must_precede: node ids this operator's contribution must flow
            THROUGH — an ordering constraint on the signal path, checked by
            :func:`~rheplicant.core.graph.assemble`.
        must_precede_because: the physics behind that constraint, in one
            sentence, quoted back by the refusal. Empty is allowed and the
            refusal still names the operator, the constraint and the actual
            placement; it just cannot say what the wrong placement costs.

    ``"key"`` in ``requires`` is a **contract, not a note**: it says this
    operator draws randomness through :meth:`~rheplicant.core.state.State.next_key`,
    and the inference layer refuses a model that contains one — a frozen draw
    from the template key would be added to every prediction alike, which is a
    bias no shape check, no linearity check and no rank test can see (the
    corruption is exactly affine and full rank). :mod:`rheplicant.core.contract`
    reads the declaration; :func:`~rheplicant.inference.parameters.refuse_stochastic_stages`
    is the consumer.

    The rest is descriptive, and that is a decision rather than an omission.
    Threading paths forward from a template State and refusing an operator
    whose ``requires`` names an absent one is not implementable against the
    operators shipped here: ``GroundPickupOperator`` declares
    ``"env.temperature"`` and documents a ``t_ground`` fallback for when it is
    missing, so its declaration means "reads if present". And ``provides`` is
    ``("data",)`` on nearly every declaring class, so threading it
    distinguishes nothing the graph's own node kinds do not already say -- the
    measured ratio is in :mod:`rheplicant.core.contract`, which is the one
    place that states it. These
    tuples therefore describe intent and carry exactly one enforced rule; they
    are not a checker waiting to be written.

    ``must_precede`` is NOT one of their consumers and is deliberately a third
    declaration: requires and
    provides speak in State paths, and every operator on the receiver chain
    reads ``"data"`` and writes ``"data"``, so "before the gain" is not a
    sentence that vocabulary can form. Position on the signal path is the
    graph's subject, so the constraint is stated in the graph's nouns (node
    ids) and enforced where the graph is compiled.

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

    # Ordering constraints on the signal path, in the graph's own nouns
    # (node ids). Enforced by rheplicant.core.graph.assemble; documented in
    # the class docstring's Attributes section.
    must_precede: ClassVar[tuple[str, ...]] = ()
    must_precede_because: ClassVar[str] = ""

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
