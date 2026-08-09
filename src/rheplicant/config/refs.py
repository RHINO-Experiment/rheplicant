"""Forms 5 and 7: reading a named object back, and stacking by name.

``{ref: ...}`` returns the **same Python object**, not a copy. That is not an
optimisation: ``BeamSpillOperator.from_projector`` is documented as "the one
call that cannot get the weight and the sky average out of step", and
``examples/driftscan_mmode.py:84`` hands one engine's ``beam_alms`` to another
so that both see the same analysis -- a loader that rebuilds each reference
passes every shape check and destroys the 2e-16 agreement the comparison
exists to demonstrate. Identity is the contract; a test asserting ``is`` is
the only way to hold it.

Nothing here calls ``jnp.asarray`` on a bare ``ref``, for the same reason and
one more: a beam, a projector and an operator are all legal ``ref`` targets and
none of them is an array. The single modifier exit point in
:func:`rheplicant.config.values.resolve_value` is identity-preserving on an
empty modifier dict, which is what lets ``ref`` be both a form and a pass.

``{from_switch_order: ...}`` is sugar for a ``stack`` over the entries named
by ``observation.switching.order``, **matched by name**. The row order of
``noise_wave.gamma_src`` is fixed by that list, and a transposition there is
shape-legal and costs tens of kelvin.
"""

from typing import Any

import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.units import convert_to_canonical
from rheplicant.config.values import ResolvedValue, register_form

#: `part:` inside a `from_switch_order`. Narrower than the modifier alphabet on
#: purpose: NoiseWaveOperator takes gamma_src_re and gamma_src_im as separate
#: fields, so a switch-ordered stack is written once per half, and `abs` or
#: `angle` there would be a quantity no field reads.
SWITCH_PARTS: tuple[str, ...] = ("re", "im")


def resolve_reference(dotted: str, context: ResolutionContext) -> Any:
    """The object a ``ref`` names -- identically, not a copy.

    Args:
        dotted: ``resources.<kind>.<name>``, optionally followed by one of that
            entry's own attributes.
        context: the scope to resolve against.

    Raises:
        ConfigError: when the name does not start at ``resources.``, when no
            resource carries it, and when the trailing segment is not an
            attribute the named object offers.
    """
    if not isinstance(dotted, str) or not dotted.startswith("resources."):
        raise ConfigError(
            f"ref: {dotted!r} does not name a resource. A reference reads back a "
            "'resources.<kind>.<name>' entry, optionally followed by one of that "
            "entry's own attributes. An axis is read with {from_grid: freq}, and a "
            "quantity computed from other named quantities is itself a "
            "resources.arrays entry -- composition here is by naming, not by nesting."
        )
    if dotted in context.resources:
        return context.resources[dotted]
    head, _, attribute = dotted.rpartition(".")
    if head in context.resources:
        target = context.resources[head]
        if not hasattr(target, attribute):
            offered = sorted(name for name in dir(target) if not name.startswith("_"))
            raise ConfigError(
                f"ref: {head!r} has no {attribute!r}. It offers {offered}. A "
                "sub-value reference exists because one call can produce two products "
                "-- horizon_truncated_beam returns the truncated maps AND the sky "
                "fraction, and the fraction is what BeamSpillOperator wants -- so the "
                "attribute must be one the object really carries."
            )
        return getattr(target, attribute)
    raise ConfigError(
        f"ref: no resource named {dotted!r}. This document declares "
        f"{sorted(context.resources)}. A resource must be declared before it is "
        "referenced; the nearest-sounding name is not a safe guess, because two "
        "beams in one document differ in exactly the ways a comparison is about."
    )


def _delivered(value: Any, modifiers: dict, form: str) -> ResolvedValue:
    """Apply ``unit:`` if one was written, and say which form said it.

    ``unit:`` is the form's own business rather than the modifier pass's --
    it decides what the number means, not what it looks like -- and every form
    in this module ends the same way, so the branch lives here once.
    """
    unit_token = modifiers.get("unit")
    if unit_token is None:
        return ResolvedValue(value, None, form, modifiers)
    converted, unit = convert_to_canonical(value, unit_token)
    return ResolvedValue(converted, unit, form, modifiers)


@register_form("ref")
def _ref(node: dict, context: ResolutionContext, modifiers: dict) -> ResolvedValue:
    return _delivered(resolve_reference(node["ref"], context), modifiers, "ref")


@register_form("stack")
def _stack(node: dict, context: ResolutionContext, modifiers: dict) -> ResolvedValue:
    # `stack` takes its entries under its own key and one flat sibling, `axis`,
    # which is spelled exactly like the noise-sigma modifier -- so it is
    # already admitted by the dispatcher and needs no `arguments=` on the
    # registration. What it does need is to be TAKEN OFF the modifiers on the
    # way out when it is stack's own: `axis: 0` is not one of NOISE_AXES and
    # the single modifier exit point would refuse the node.
    from rheplicant.config.values import resolve_value

    entries = node["stack"]
    if not isinstance(entries, (list, tuple)) or not entries:
        raise ConfigError(
            f"stack: expects a non-empty list of value nodes, got {entries!r}. It is a "
            "container, not a computation -- it has no operator and no result type "
            "other than 'one more axis'."
        )
    # An INTEGER here is stack's own argument; a string ('time' / 'freq' /
    # 'none') is the modifier, and is left in place to be validated as one. A
    # bool is neither: it is excluded so that `axis: true` is refused by the
    # modifier alphabet rather than silently read as axis 1.
    declared = node.get("axis", 0)
    mine = isinstance(declared, int) and not isinstance(declared, bool)
    axis = declared if mine else 0
    onward = modifiers
    if mine:
        onward = {key: value for key, value in modifiers.items() if key != "axis"}
    parts = [jnp.asarray(resolve_value(entry, context).value) for entry in entries]
    return _delivered(jnp.stack(parts, axis=axis), onward, "stack")


@register_form("from_switch_order")
def _from_switch_order(node: dict, context: ResolutionContext, modifiers: dict) -> ResolvedValue:
    spec = node["from_switch_order"]
    if not isinstance(spec, dict) or "resource" not in spec:
        raise ConfigError(
            "from_switch_order: expects {resource: resources.<kind>, part: re|im}. It "
            "is sugar for a stack over the entries observation.switching.order names."
        )
    unknown = sorted(set(spec) - {"resource", "part"})
    if unknown:
        raise ConfigError(f"from_switch_order: unknown key(s) {unknown}; it takes resource, part.")
    part = spec.get("part")
    if part is not None and part not in SWITCH_PARTS:
        raise ConfigError(
            f"from_switch_order: part={part!r}; it takes {list(SWITCH_PARTS)}. The two "
            "halves are separate operator fields (gamma_src_re, gamma_src_im), so "
            "each is stacked in its own node."
        )
    if not context.switch_order:
        raise ConfigError(
            "from_switch_order: this run declares no observation.switching.order, so "
            "there is no order to stack in. With switching.mode: none the model "
            "declares no cal_loads and noise_wave.gamma_src has exactly one row -- "
            "write that row directly."
        )
    prefix = spec["resource"].rstrip(".") + "."
    # `"." not in ...` is what keeps resources.s_params.hot.calibration an
    # attribute of one entry rather than a fourth entry: a prefix match written
    # as startswith alone would count it as a label the order does not name and
    # refuse a document that is right.
    available = {
        name[len(prefix) :]: value
        for name, value in context.resources.items()
        if name.startswith(prefix) and "." not in name[len(prefix) :]
    }
    order = tuple(context.switch_order)
    missing = [label for label in order if label not in available]
    extra = sorted(set(available) - set(order))
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"has no entry for {missing}")
        if extra:
            parts.append(f"carries {extra}, which the order does not name")
        raise ConfigError(
            f"from_switch_order: {spec['resource']} {' and '.join(parts)}. "
            f"switching.order is {list(order)}, and it fixes four things at once: the "
            "switch indices, the order of model.cal_loads, the row order of "
            "noise_wave.gamma_src, and the thermistor_columns labels. Matching by name "
            "rather than by position is what makes a transposition impossible -- a "
            "transposed gamma_src is shape-legal and costs tens of kelvin."
        )
    take = {"re": jnp.real, "im": jnp.imag}.get(part, lambda array: array)
    rows = [take(jnp.asarray(available[label])) for label in order]
    return _delivered(jnp.stack(rows, axis=0), modifiers, "from_switch_order")
