"""Turn a Pipeline into a parametric forward function for inference.

This is the seam between forward modelling and inference: the pipeline stays a
clean instrument description, and inference engines (gradient calibrators,
NumPyro, future neural surrogates) see only ``f(params) -> prediction``.

The mechanism is the standard Equinox partition/combine idiom::

    params, static = eqx.partition(pipeline, filter_spec)
    prediction = eqx.combine(params, static)(state_template).data
"""

from collections.abc import Callable
from typing import Any

import equinox as eqx
import jax

from replicant.core.operator import AbstractOperator
from replicant.core.state import State


def build_forward_fn(
    pipeline: AbstractOperator,
    state_template: State,
    filter_spec: Any = eqx.is_inexact_array,
) -> tuple[Callable[[Any], jax.Array], Any]:
    """Build ``forward(params) -> prediction`` from a pipeline and a template state.

    Args:
        pipeline: any operator (usually a Pipeline) describing the forward model.
        state_template: the input state the forward model is evaluated on
            (coordinates, PRNG key, metadata...). Closed over, held fixed.
        filter_spec: which pipeline leaves are trainable parameters. Default:
            every inexact (floating-point) array. Pass a pytree-of-bools (e.g.
            built with ``jax.tree.map(lambda _: False, pipeline)`` +
            ``eqx.tree_at``) to train a subset.

    Returns:
        ``(forward, params0)``: the forward function and the initial parameter
        pytree extracted from the pipeline. ``forward(params0)`` reproduces
        ``pipeline(state_template).data`` exactly.
    """
    params0, static = eqx.partition(pipeline, filter_spec)

    def forward(params: Any) -> jax.Array:
        model = eqx.combine(params, static)
        return model(state_template).data

    return forward, params0
