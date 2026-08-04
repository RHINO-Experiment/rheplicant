"""Reading an assembled tree through the operators' own declarations.

:class:`~rheplicant.core.operator.AbstractOperator` carries two declarative
ClassVars, ``requires`` and ``provides``. This module is where they stop being
prose: it walks a built operator tree and answers *which stages declare a given
State path*, so a caller can refuse a composition on the strength of what its
stages say about themselves rather than on a hard-coded list of classes.

**One path is enforced, and it is** ``"key"``. An operator that names it in
``requires`` draws randomness through
:meth:`~rheplicant.core.state.State.next_key`, and that is a property no shape
check, no linearity check and no rank test can see — which is why it is the
declaration worth consuming. :func:`stages_requiring` is the general form;
:data:`RANDOMNESS` is the path that has a consumer today
(:func:`~rheplicant.inference.parameters.refuse_stochastic_stages`).

**Why the rest is descriptive, deliberately.** The obvious next step —
threading the paths a template ``State`` supplies forward and refusing an
operator whose ``requires`` names one that is absent — is not implementable
against the shipped operator set, and the counter-example is in the package:
``GroundPickupOperator`` declares ``"env.temperature"`` and then documents a
``t_ground`` fallback for when it is missing, so the declaration means "reads
if present", not "needs". A blanket availability rule would refuse a model the
package itself describes as legitimate. ``provides`` is weaker still: 26 of the
31 declaring classes provide exactly ``("data",)``, so threading it forward
distinguishes almost nothing that the graph's own source/transform kinds do not
already say. Both stay as documentation of intent, and the class docstring on
:class:`~rheplicant.core.operator.AbstractOperator` says so in those words
rather than promising a checker.

The walk is by *pytree position*, not by the composite spine
:func:`~rheplicant.core.graph._children` uses. That is on purpose: this is a
safety check, so it must not miss a stage held by a composite type nobody
taught it about. Named composites (:class:`~rheplicant.core.pipeline.Pipeline`,
:class:`~rheplicant.core.combinators.SumOperator`,
:class:`~rheplicant.core.combinators.SelectOperator`) contribute their stage
names to the label so a refusal can quote the graph node id; anything else
contributes structure without a name.
"""

from collections.abc import Iterator

import jax

from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline

#: The one ``requires`` path with an enforced consumer: this operator draws
#: randomness, so a model containing it is not a function of its parameters.
RANDOMNESS = "key"


def _labelled_children(op: AbstractOperator) -> tuple[tuple[str | None, AbstractOperator], ...]:
    """Every operator ``op`` holds directly, with a stage name where one exists.

    A ``None`` label means "structural spine, no name of its own" — an
    :class:`~rheplicant.core.graph.Assembly` wrapping its fold, say. Those
    contribute no segment to the path, so the label a refusal quotes stays the
    stage name a user recognises.
    """
    if isinstance(op, Pipeline):
        return tuple(zip(op.names, op.stages, strict=True))
    if isinstance(op, (SumOperator, SelectOperator)):
        return tuple(zip(op.names, op.branches, strict=True))
    children = jax.tree_util.tree_flatten_with_path(
        op, is_leaf=lambda node: node is not op and isinstance(node, AbstractOperator)
    )[0]
    return tuple((None, child) for _, child in children if isinstance(child, AbstractOperator))


def walk_operators(
    op: AbstractOperator, _label: str = ""
) -> Iterator[tuple[str, AbstractOperator]]:
    """Yield ``(label, operator)`` for ``op`` and every operator nested in it.

    The root's label is ``""``. Nested labels are ``/``-joined stage names, so
    a noise stage inside a summed branch reads ``t_ant/atmosphere``.
    """
    yield _label, op
    for name, child in _labelled_children(op):
        if name is None:
            child_label = _label
        elif _label:
            child_label = f"{_label}/{name}"
        else:
            child_label = name
        yield from walk_operators(child, child_label)


def stages_requiring(
    op: AbstractOperator, path: str
) -> tuple[tuple[str, AbstractOperator], ...]:
    """Every stage of ``op`` whose ``requires`` names the State path ``path``."""
    return tuple((label, stage) for label, stage in walk_operators(op) if path in stage.requires)


def describe_stages(stages: tuple[tuple[str, AbstractOperator], ...]) -> str:
    """``"NoiseOperator at 'noise'"`` — the stage list a refusal quotes."""
    return ", ".join(
        f"{type(stage).__name__} at {label!r}" if label else type(stage).__name__
        for label, stage in stages
    )
