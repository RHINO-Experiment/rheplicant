"""inference.parameters: the latent grammar (schema §4.7.2).

Each entry becomes a :class:`rheplicant.inference.Latent` plus the raw
binding keys (``into``/``transform``/``fan``) that Task 3's space builder
resolves against the fit twin.  Priors build numpyro distributions lazily --
numpyro is an extra -- and a scalar family broadcasts to the declared init's
shape, because ``Latent.__check_init__`` refuses ``prior.shape() !=
init.shape`` (``inference/parameters.py:267``) and four levels of braces for
``dist.Normal(jnp.zeros(8), 400.0)`` was v0's mistake, not the user's.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import record_resolved_delivery
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.values import resolve_value

__all__ = ["ParsedLatent", "parse_latents"]

_LATENT_KEYS = frozenset(
    {
        "init",
        "prior",
        "linear",
        "scope",
        "support",
        "hyper",
        "into",
        "transform",
        "fan",
        "ref",
        "unit",
        "latex",
        "renames",
    }
)
_PRIOR_FAMILIES = {
    "normal": ("loc", "scale"),
    "uniform": ("low", "high"),
    "log_normal": ("loc", "scale"),
}
_FAN_MODES = ("broadcast", "distribute")


class ParsedLatent(NamedTuple):
    """One latent, parsed: the Latent itself plus what Task 3 still needs."""

    latent: Any
    into: tuple[str, ...] | None
    transform: Any
    fan: str | None
    unit: str | None
    latex: str | None
    renames: tuple[str, ...]
    ref: Any


def _require_numpyro(where: str):
    try:
        import numpyro.distributions as dist
    except ImportError as exc:
        raise ConfigError(
            f"{where}: declaring a prior needs numpyro: pip install 'rheplicant[numpyro]'."
        ) from exc
    return dist


def _operand(name: str, node: Any, context: ResolutionContext) -> Any:
    if isinstance(node, bool) or not isinstance(node, (int, float, Mapping)):
        raise ConfigError(f"{name}: is a number or a value node; got {node!r}.")
    if isinstance(node, (int, float)):
        return float(node)
    pieces = name.split(".")
    pieces[2] = "*"
    selector = ".".join(pieces)
    destination = DestinationDescriptor(name, "config_path", selector)
    resolved = resolve_value(node, context, destination=destination)
    record_resolved_delivery(context, destination, resolved.unit)
    return resolved.value


def _parse_prior(name: str, spec: Any, init: jnp.ndarray, context: ResolutionContext) -> Any:
    if spec is None:
        return None
    where = f"inference.parameters.{name}.prior"
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{where}: is a mapping naming one family; got {spec!r}.")
    body = dict(spec)
    body.pop("unit", None)
    if "python" in body:
        return resolve_value(
            dict(spec),
            context,
            destination=DestinationDescriptor(
                f"{where}.python",
                "config_path",
                "inference.parameters.*.prior.python",
            ),
        ).value
    families = sorted(set(body) & set(_PRIOR_FAMILIES))
    if len(families) != 1 or set(body) - set(_PRIOR_FAMILIES):
        raise ConfigError(
            f"{where}: takes exactly one family of "
            f"{sorted(_PRIOR_FAMILIES)} (or python:); got "
            f"{sorted(body)}."
        )
    family = families[0]
    args = body[family]
    wanted = _PRIOR_FAMILIES[family]
    if not isinstance(args, Mapping) or set(args) != set(wanted):
        raise ConfigError(
            f"{where}.{family}: takes exactly {list(wanted)}; got "
            f"{sorted(args) if isinstance(args, Mapping) else args!r}."
        )
    dist = _require_numpyro(where)
    operands = [_operand(f"{where}.{family}.{key}", args[key], context) for key in wanted]
    builder = {"normal": dist.Normal, "uniform": dist.Uniform, "log_normal": dist.LogNormal}[family]
    built = builder(*operands)
    shape = jnp.shape(init)
    if tuple(built.shape()) != shape:
        try:
            built = built.expand(shape)
        except Exception as exc:
            raise ConfigError(
                f"{where}: a {family} prior of shape {tuple(built.shape())} "
                f"does not broadcast against init's shape {shape}."
            ) from exc
    return built


def _names(name: str, node: Any, what: str) -> tuple[str, ...]:
    if node is None:
        return ()
    if isinstance(node, str):
        return (node,)
    if isinstance(node, (list, tuple)) and all(isinstance(entry, str) for entry in node):
        return tuple(node)
    raise ConfigError(f"{name}: {what} is a string or a list of strings; got {node!r}.")


def parse_latents(section: Any, context: ResolutionContext) -> dict[str, ParsedLatent]:
    """``inference.parameters`` -> ``{name: ParsedLatent}``, declaration order."""
    from rheplicant.inference import Latent

    if not isinstance(section, Mapping) or not all(isinstance(key, str) for key in section):
        raise ConfigError(
            f"inference.parameters: is a mapping of latent name -> spec; got {section!r}."
        )
    parsed: dict[str, ParsedLatent] = {}
    for name, spec in section.items():
        where = f"inference.parameters.{name}"
        if not isinstance(spec, Mapping):
            raise ConfigError(f"{where}: is a mapping; got {spec!r}.")
        for reserved, capability in (("support", "capability 4"), ("hyper", "capability 4")):
            if reserved in spec:
                raise ConfigError(
                    f"{where}.{reserved}: is reserved with {capability} "
                    "(schema §8.2) and refused in v1."
                )
        check_unknown_keys(where, dict(spec), _LATENT_KEYS, label="a latent:")
        scope = (
            spec["scope"]
            if "scope" in spec
            else context.use_default("inference.parameters[].scope", "global")
        )
        if scope != "global":
            raise ConfigError(
                f"{where}.scope: {scope!r} is reserved with capability 4 "
                "(schema §8.2); v1 knows scope: global only."
            )
        if "init" not in spec:
            raise ConfigError(
                f"{where}: init: is required -- it is the authority on the "
                "latent's shape and dtype (Latent.init, parameters.py:210)."
            )
        init_destination = DestinationDescriptor(
            f"{where}.init", "config_path", "inference.parameters.*.init"
        )
        resolved = resolve_value(
            spec["init"],
            context,
            destination=init_destination,
        )
        init = jnp.asarray(resolved.value, dtype=context.dtype)
        record_resolved_delivery(context, init_destination, resolved.unit)
        written_unit = resolved.modifiers.get("unit")
        unit = spec.get("unit")
        if unit is not None and written_unit is not None and unit != written_unit:
            raise ConfigError(
                f"{where}: unit: {unit!r} conflicts with init's own declared "
                f"unit {written_unit!r}; a latent has one unit."
            )
        if bool(jnp.all(init == 0)):
            warnings.warn(
                f"{where}: an all-zero init makes check_linearity's probe "
                "scales and the gradient step absolute rather than relative "
                "(both fall back to 1.0).",
                UserWarning,
                stacklevel=2,
            )
        linear = (
            spec["linear"]
            if "linear" in spec
            else context.use_default("inference.parameters[].linear", False)
        )
        if not isinstance(linear, bool):
            raise ConfigError(f"{where}.linear: is a bool; got {linear!r}.")
        fan = spec.get("fan")
        if fan is not None and fan not in _FAN_MODES:
            raise ConfigError(f"{where}.fan: {fan!r} is not one of {list(_FAN_MODES)}.")
        latex = spec.get("latex")
        if latex is not None and not isinstance(latex, str):
            raise ConfigError(f"{where}.latex: is a string; got {latex!r}.")
        ref = spec.get("ref")
        if ref is not None:
            ref_destination = DestinationDescriptor(
                f"{where}.ref", "config_path", "inference.parameters.*.ref"
            )
            resolved_ref = resolve_value(ref, context, destination=ref_destination)
            ref = jnp.asarray(resolved_ref.value, dtype=context.dtype)
            record_resolved_delivery(context, ref_destination, resolved_ref.unit)
        parsed[name] = ParsedLatent(
            latent=Latent(
                name,
                init=init,
                prior=_parse_prior(name, spec.get("prior"), init, context),
                linear=linear,
            ),
            into=_names(where, spec.get("into"), "into:") or None,
            transform=spec.get("transform"),
            fan=fan,
            unit=unit if unit is not None else written_unit,
            latex=latex,
            renames=_names(where, spec.get("renames"), "renames:"),
            ref=ref,
        )
    return parsed
