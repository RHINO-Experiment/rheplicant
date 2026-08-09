"""Form 2: the array constructors.

Every shape position goes through :func:`rheplicant.config.symbols.resolve_extent`,
so a grid length is written once and referred to by name. Where a literal
integer happens to equal one of the run's extents the node records it under
``_shadowed`` rather than refusing -- check A41 is a report, because a literal
8 may genuinely be 8, and what it cannot be is *tied* to the grid.
"""

from typing import Any

import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.symbols import literal_shadowing_a_symbol, resolve_extent
from rheplicant.config.units import convert_to_canonical
from rheplicant.config.values import ResolvedValue, register_form


def _shape(spec: Any, context: ResolutionContext, form: str) -> tuple[tuple[int, ...], dict]:
    if not isinstance(spec, (list, tuple)):
        raise ConfigError(
            f"{form}: expects a shape -- a list of integers or shape symbols -- and "
            f"got {spec!r}. A scalar zero is written {{value: 0.0}}."
        )
    scope = context.shape_scope
    shadowed = {
        index: symbol
        for index, entry in enumerate(spec)
        if (symbol := literal_shadowing_a_symbol(entry, scope)) is not None
    }
    return tuple(resolve_extent(entry, scope) for entry in spec), shadowed


def _finish(array, modifiers: dict, form: str, shadowed: dict) -> ResolvedValue:
    unit_token = modifiers.get("unit")
    if unit_token is None:
        return ResolvedValue(array, None, form, {**modifiers, "_shadowed": shadowed})
    converted, unit = convert_to_canonical(array, unit_token)
    return ResolvedValue(converted, unit, form, {**modifiers, "_shadowed": shadowed})


@register_form("zeros")
def _zeros(node, context, modifiers):
    shape, shadowed = _shape(node["zeros"], context, "zeros")
    return _finish(jnp.zeros(shape, dtype=context.dtype), modifiers, "zeros", shadowed)


@register_form("ones")
def _ones(node, context, modifiers):
    shape, shadowed = _shape(node["ones"], context, "ones")
    return _finish(jnp.ones(shape, dtype=context.dtype), modifiers, "ones", shadowed)


@register_form("full")
def _full(node, context, modifiers):
    spec = node["full"]
    _require_keys(spec, {"shape", "value"}, "full")
    shape, shadowed = _shape(spec["shape"], context, "full")
    return _finish(
        jnp.full(shape, spec["value"], dtype=context.dtype), modifiers, "full", shadowed
    )


@register_form("list")
def _list(node, context, modifiers):
    return _finish(jnp.asarray(node["list"], dtype=context.dtype), modifiers, "list", {})


@register_form("linspace")
def _linspace(node, context, modifiers):
    spec = node["linspace"]
    # Before _require_keys, not after: a missing 'endpoint' is the one absence
    # this form has to explain rather than list. _require_keys would report it
    # as "missing ['endpoint']" alongside the other three keys, which reads as
    # a typo and invites the reader to write whichever value makes the error
    # stop -- and the two values differ by a channel in the shipped example.
    if isinstance(spec, dict) and "endpoint" not in spec:
        raise ConfigError(
            "linspace: 'endpoint' is required and has no default. It decides whether "
            "the last sample repeats the first turn -- the sidereal-turn contract "
            "DriftScanProjector(uniform_sampling=True) checks is exactly this key -- "
            "and it changes the channel spacing: linspace(60, 85, 8) spans 25/7 MHz "
            "with the endpoint included and 25/8 without, so a config that declares "
            "channel_width = band/n_freq agrees with one convention and not the "
            "other. Write endpoint: true or endpoint: false."
        )
    _require_keys(spec, {"start", "stop", "num", "endpoint"}, "linspace")
    num = resolve_extent(spec["num"], context.shape_scope)
    array = jnp.linspace(
        float(spec["start"]),
        float(spec["stop"]),
        num,
        endpoint=bool(spec["endpoint"]),
        dtype=context.dtype,
    )
    return _finish(array, modifiers, "linspace", {})


@register_form("arange")
def _arange(node, context, modifiers):
    spec = node["arange"]
    _require_keys(spec, {"start", "step", "num"}, "arange")
    num = resolve_extent(spec["num"], context.shape_scope)
    start, step = float(spec["start"]), float(spec["step"])
    array = start + step * jnp.arange(num, dtype=context.dtype)
    return _finish(array, modifiers, "arange", {})


@register_form("modulo")
def _modulo(node, context, modifiers):
    spec = node["modulo"]
    _require_keys(spec, {"num", "period"}, "modulo")
    num = resolve_extent(spec["num"], context.shape_scope)
    period = resolve_extent(spec["period"], context.shape_scope)
    if period < 1:
        raise ConfigError(f"modulo: period must be >= 1, got {period}.")
    return _finish(jnp.arange(num) % period, modifiers, "modulo", {})


@register_form("from_grid")
def _from_grid(node, context, modifiers):
    axis = node["from_grid"]
    grid = {"freq": context.freq, "time": context.time}.get(axis, ...)
    if grid is ...:
        raise ConfigError(
            f"from_grid: {axis!r} is not one of the run's own axes; they are 'freq' "
            "and 'time'. Anything else is a resource, and a resource is read with "
            "{ref: resources.<kind>.<name>}."
        )
    if grid is None:
        raise ConfigError(
            f"from_grid: {axis!r} names an axis this run does not declare. "
            "observation.freq.grid and observation.time.grid are both required, so "
            "this is reachable only when a value node is resolved before the "
            "observation is."
        )
    return _finish(grid, modifiers, "from_grid", {})


def _require_keys(spec: Any, required: set[str], form: str) -> None:
    if not isinstance(spec, dict):
        raise ConfigError(
            f"{form}: expects a mapping with {sorted(required)}, got "
            f"{type(spec).__name__} ({spec!r})."
        )
    missing = sorted(required - set(spec))
    extra = sorted(set(spec) - required)
    if missing or extra:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if extra:
            parts.append(f"unknown {extra}")
        raise ConfigError(
            f"{form}: {' and '.join(parts)}. Its keys are exactly {sorted(required)}."
        )
