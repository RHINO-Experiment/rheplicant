"""resources.arrays: any value node, bound to a name.

There is no package constructor behind this kind and there is nothing to
mirror -- ``find src -name "*array*"`` returns nothing. It exists because the
value grammar deliberately has no expression language, and a schema that
cannot say ``f(g(x), y)`` cannot express the seven reflection coefficients
``examples/gibbs_plan.py:112-119`` builds by nested ``rhino_cal_jax`` calls.

Naming is the answer: bind the inner call to a name, reference it from the
outer one. It is a let-binding, not an expression -- there is no operator, no
precedence and no evaluation order to reason about, and the resulting DAG is
exactly the ``ref`` graph ``resources`` already is.
"""

from typing import Any

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import record_resolved_delivery
from rheplicant.config.dimensions import bind_resource_dimension, dimension_of
from rheplicant.config.resources import register_kind
from rheplicant.config.values import resolve_value


@register_kind("arrays")
def build_array(name: str, spec: dict, context: ResolutionContext) -> Any:
    """Resolve one named value node.

    ``name`` is the dotted ``resources.arrays.<name>`` the caller built this
    under -- every kind builder takes it, because ``_KINDS[kind]`` is called
    the same way for all six kinds, but this builder has no use for it: the
    value node it resolves does not need to know its own name.
    """
    destination = DestinationDescriptor(
        name,
        "resource_field",
        "rheplicant.config.kinds.arrays.build_array.value",
    )
    resolved = resolve_value(
        spec,
        context,
        destination=destination,
    )
    bind_resource_dimension(
        context.dimensions,
        name,
        None if resolved.unit is None else dimension_of(resolved.unit),
    )
    record_resolved_delivery(context, destination, resolved.unit)
    return resolved.value
