"""``observation.switching`` (schema §4.1.5): one list fixes four orders.

``order`` fixes the switch indices, the order of ``model.cal_loads``, the row
order of ``noise_wave.gamma_src`` (``from_switch_order`` reads it off the
context), and -- for an ingested run -- the thermistor labels. Index 0 is the
reserved literal ``antenna``: the one branch that is not a calibration load.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.values import resolve_value

__all__ = ["SwitchingBuild", "compile_switching", "declared_order"]

_KEYS = {
    "none": frozenset({"mode"}),
    "cycle": frozenset({"mode", "order", "cycle", "dwell", "index"}),
}


class SwitchingBuild(NamedTuple):
    """The declared order and the compiled ``(n_time,)`` integer cycle."""

    order: tuple[str, ...]
    receiver_input: Any


def declared_order(spec: Mapping) -> tuple[str, ...]:
    order = spec.get("order")
    if not isinstance(order, list) or len(order) < 2 \
            or not all(isinstance(label, str) for label in order):
        raise ConfigError(
            "switching: mode: cycle requires order: -- a list of at least two "
            "labels, index 0 the literal 'antenna', the rest the keys of "
            "model.cal_loads in switch order."
        )
    if order[0] != "antenna":
        raise ConfigError(
            f"switching.order[0] is the reserved literal 'antenna'; got "
            f"{order[0]!r}. The antenna chain is not a calibration load, and "
            "index 0 is where NoiseWaveOperator's source index puts it."
        )
    if len(set(order)) != len(order):
        raise ConfigError(
            f"switching.order: every label appears once; got {order!r}."
        )
    return tuple(order)


def compile_switching(spec: Any, context: ResolutionContext, *,
                      n_time: int) -> SwitchingBuild:
    """Compile the switching section against the run's own time axis."""
    if spec is None:
        spec = {"mode": "none"}
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"observation.switching: is a mapping; got {type(spec).__name__}."
        )
    mode = spec.get("mode", "none")
    if mode not in _KEYS:
        raise ConfigError(
            f"observation.switching: mode is 'none' or 'cycle'; got {mode!r}."
        )
    check_unknown_keys("observation.switching", dict(spec), _KEYS[mode],
                       label=f"mode: {mode}")
    if mode == "none":
        return SwitchingBuild(order=(), receiver_input=None)

    order = declared_order(spec)
    n_source = len(order)
    cycle = spec.get("cycle", "from_file" if "index" in spec else "round_robin")
    if cycle == "none":
        raise ConfigError(
            "switching.cycle: 'none' is not a cycle -- a run that never "
            "switches is switching: {mode: none}."
        )
    if cycle not in ("round_robin", "from_file"):
        raise ConfigError(
            f"switching.cycle: is 'round_robin' or 'from_file'; got {cycle!r}."
        )
    if cycle == "round_robin" and "index" in spec:
        raise ConfigError(
            "switching: cycle: round_robin and an explicit index: say two "
            "different things about the same samples -- write one thing."
        )
    if cycle == "from_file":
        if "index" not in spec:
            raise ConfigError(
                "switching: cycle: from_file requires index: -- a value node "
                "holding the (n_time,) integer switch states."
            )
        index = jnp.asarray(
            resolve_value(
                spec["index"],
                context,
                destination=DestinationDescriptor(
                    "observation.switching.index",
                    "config_path",
                    "observation.switching.index",
                ),
            ).value
        )
        if index.shape != (n_time,):
            raise ConfigError(
                f"switching.index: is (n_time,) = ({n_time},); got "
                f"{tuple(index.shape)}."
            )
        if not jnp.issubdtype(index.dtype, jnp.integer):
            if not bool(jnp.all(index == jnp.round(index))):
                raise ConfigError(
                    "switching.index: holds non-integer values -- a switch state "
                    "is a position number, and truncating a fractional one would "
                    "silently reassign samples to the wrong source."
                )
            index = index.astype(jnp.int32)
        low, high = int(index.min()), int(index.max())
        if low < 0 or high >= n_source:
            raise ConfigError(
                f"switching.index: values run {low}..{high} but the order "
                f"declares {n_source} positions (0..{n_source - 1})."
            )
        return SwitchingBuild(order=order, receiver_input=index)

    dwell_node = spec.get("dwell", 1)
    resolved = resolve_value(
        dwell_node,
        context,
        destination=DestinationDescriptor(
            "observation.switching.dwell",
            "config_path",
            "observation.switching.dwell",
        ),
    )
    if resolved.unit is not None and resolved.unit.canonical != "samples":
        raise ConfigError(
            f"switching.dwell: is a sample count (unit: samples); got unit "
            f"{resolved.unit.canonical!r}."
        )
    dwell = resolved.value
    if isinstance(dwell, bool) or int(dwell) != dwell or int(dwell) < 1:
        raise ConfigError(
            f"switching.dwell: is a positive integer number of samples; got "
            f"{dwell!r}."
        )
    index = (jnp.arange(n_time, dtype=jnp.int32) // int(dwell)) % n_source
    return SwitchingBuild(order=order, receiver_input=index)
