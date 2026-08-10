"""The transform registry, and bindings -> ParameterSpace (schema §4.7.2-3).

``Bind.into`` holds callables (``inference/parameters.py:338``);
``resolve_path_on`` (``config/paths.py:167``) compiles a dotted path into one
and refuses anything that is not an array leaf.  The registry is closed for
the same reason the derivation registry is: every entry names a callable this
package already ships, so a transform is a reference rather than arithmetic.
``beam_analysis`` is Plan 2C's, with its consumers.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.hatch import import_target
from rheplicant.config.paths import (
    parse_path,
    refuse_aliased_target,
    refuse_duplicate_targets,
    resolve_path_on,
)
from rheplicant.config.refs import resolve_reference
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.values import resolve_value

__all__ = ["build_space", "parse_transform"]

_NAMED = ("identity", "exp", "log", "sum", "split_rows", "unit_mean_bandpass")
_MAPPING = ("affine", "matmul", "log_link_basis", "basis_expand", "python")
_BINDING_KEYS = frozenset({"latents", "into", "transform", "fan"})


def _operand(where: str, node: Any, context: ResolutionContext) -> Any:
    if isinstance(node, bool) or not isinstance(node, (int, float, Mapping)):
        raise ConfigError(f"{where}: is a number or a value node; got {node!r}.")
    if isinstance(node, (int, float)):
        return float(node)
    return jnp.asarray(resolve_value(node, context).value)


def parse_transform(spec: Any, context: ResolutionContext, *,
                    where: str) -> tuple[Any, str | None]:
    """A transform spec -> ``(fn, canonical_fan)``; identity -> ``(None, None)``."""
    if spec is None or spec == "identity":
        return None, None
    if isinstance(spec, str):
        if spec == "exp":
            return jnp.exp, "broadcast"
        if spec == "log":
            return jnp.log, "broadcast"
        if spec == "sum":
            return jnp.sum, "broadcast"
        if spec == "split_rows":
            return (lambda v: tuple(v)), "distribute"
        if spec == "unit_mean_bandpass":
            from rheplicant.radio.instrument.receiver import unit_mean_bandpass

            return unit_mean_bandpass, "broadcast"
        raise ConfigError(
            f"{where}: {spec!r} is not a registered transform; the registry "
            f"holds {list(_NAMED)} and the mapping forms {list(_MAPPING)}."
        )
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{where}: is a name or a mapping; got {spec!r}.")
    if "beam_analysis" in spec:
        raise ConfigError(
            f"{where}: beam_analysis arrives with Plan 2C, alongside the "
            "driftscan exits that consume it."
        )
    heads = sorted(set(spec) & set(_MAPPING))
    if len(heads) != 1:
        raise ConfigError(
            f"{where}: a mapping transform names exactly one of "
            f"{list(_MAPPING)}; got {sorted(spec)}."
        )
    head = heads[0]
    if head == "python":
        unknown = sorted(set(spec) - {"python", "fan"})
        if unknown:
            raise ConfigError(f"{where}: python: takes fan: and nothing else; "
                              f"got {unknown} too.")
        fan = spec.get("fan")
        if fan not in ("broadcast", "distribute"):
            raise ConfigError(
                f"{where}: a python: transform must declare its own fan "
                "(broadcast or distribute) -- the registry cannot know what "
                f"an arbitrary callable produces; got {fan!r}."
            )
        return import_target(spec["python"]), fan
    body = spec[head]
    unknown = sorted(set(spec) - {head})
    if unknown:
        raise ConfigError(f"{where}: {head}: stands alone; got {unknown} too.")
    if not isinstance(body, Mapping):
        raise ConfigError(f"{where}.{head}: is a mapping; got {body!r}.")
    if head == "affine":
        check_unknown_keys(where, dict(body), frozenset({"scale", "offset"}),
                           label="affine:")
        scale = _operand(f"{where}.affine.scale", body.get("scale", 1.0),
                         context)
        offset = _operand(f"{where}.affine.offset", body.get("offset", 0.0),
                          context)
        return (lambda v, _s=scale, _o=offset: _s * v + _o), "broadcast"
    if head == "matmul":
        check_unknown_keys(where, dict(body), frozenset({"design"}),
                           label="matmul:")
        if "design" not in body:
            raise ConfigError(f"{where}.matmul: requires design: -- a value "
                              "node for the design matrix.")
        design = jnp.asarray(resolve_value(body["design"], context).value)
        return (lambda c, _d=design: _d @ c), "broadcast"
    if head == "log_link_basis":
        from rheplicant.core.basis import basis_matrix

        check_unknown_keys(where, dict(body),
                           frozenset({"kind", "n_basis", "axis"}),
                           label="log_link_basis:")
        axis = body.get("axis", "freq")
        if axis not in ("freq", "time"):
            raise ConfigError(f"{where}.log_link_basis.axis: is freq or "
                              f"time; got {axis!r}.")
        grid = context.freq if axis == "freq" else context.time
        if grid is None:
            raise ConfigError(f"{where}.log_link_basis: needs the run's "
                              f"{axis} grid, and this context has none.")
        if "kind" not in body or "n_basis" not in body:
            raise ConfigError(f"{where}.log_link_basis: requires kind: and "
                              "n_basis:; n comes from the grid.")
        matrix = basis_matrix(str(body["kind"]), n=int(grid.shape[0]),
                              n_basis=int(body["n_basis"]))
        return (lambda c, _m=matrix: jnp.exp(_m @ c)), "broadcast"
    # basis_expand
    from rheplicant.core.basis import SeparableBasis

    check_unknown_keys(where, dict(body), frozenset({"basis"}),
                       label="basis_expand:")
    reference = body.get("basis")
    if not isinstance(reference, Mapping) or set(reference) != {"ref"}:
        raise ConfigError(
            f"{where}.basis_expand: basis is {{ref: resources.bases.<name>}}; "
            f"got {reference!r}."
        )
    basis = resolve_reference(reference["ref"], context)
    if not isinstance(basis, SeparableBasis):
        raise ConfigError(
            f"{where}.basis_expand: {reference['ref']!r} is "
            f"{type(basis).__name__}, not SeparableBasis."
        )
    return basis.expand, "broadcast"


def _merged_fan(declared: str | None, canonical: str | None,
                where: str) -> str | None:
    if declared is not None and canonical is not None \
            and declared != canonical:
        raise ConfigError(
            f"{where}: fan: {declared} contradicts the transform's own fan "
            f"({canonical}) -- check A38's registry consistency."
        )
    return declared if declared is not None else canonical


def _selectors(where: str, paths: tuple[str, ...], fit_twin: Any,
               replaced: tuple[str, ...], seen: list[str]) -> tuple:
    selectors = []
    for path in paths:
        head = parse_path(path)[0]
        if head in replaced:
            raise ConfigError(
                f"{where}: into: {path!r} targets node {head!r}, which "
                "inference.twin.replace just rebuilt -- the binding would "
                "overwrite the replacement at bind time (check B8). Say one "
                "or the other."
            )
        refuse_aliased_target(path, fit_twin)
        selectors.append(resolve_path_on(path, fit_twin).selector)
        seen.append(path)
    return tuple(selectors)


def _joint_prior(section: Any, names: tuple[str, ...]) -> Any:
    if section is None:
        return None
    from rheplicant.inference import JeffreysPrior

    if not isinstance(section, Mapping) or set(section) != {"jeffreys"}:
        raise ConfigError(
            "inference.joint_prior: names the one joint-prior type the "
            f"package knows -- {{jeffreys: {{over: [...]}}}}; got {section!r}."
        )
    body = section["jeffreys"]
    if not isinstance(body, Mapping):
        raise ConfigError(f"inference.joint_prior.jeffreys: is a mapping; "
                          f"got {body!r}.")
    check_unknown_keys("inference.joint_prior.jeffreys", dict(body),
                       frozenset({"over", "rank_rtol"}), label="jeffreys:")
    if "over" not in body:
        raise ConfigError("inference.joint_prior.jeffreys: requires over: -- "
                          "the latent names it covers.")
    kwargs = {}
    if "rank_rtol" in body:
        kwargs["rank_rtol"] = float(body["rank_rtol"])
    return JeffreysPrior(over=tuple(body["over"]), **kwargs)


def build_space(parsed: Any, bindings: Any, joint_prior: Any, *,
                fit_twin: Any, replaced: tuple[str, ...],
                context: ResolutionContext) -> Any:
    """Parsed latents + bindings + joint_prior -> a ParameterSpace, or None."""
    from rheplicant.inference import Bind, ParameterSpace

    if parsed is None:
        if bindings:
            raise ConfigError(
                "inference.bindings: without inference.parameters binds "
                "nothing -- declare the latents first."
            )
        if joint_prior is not None:
            raise ConfigError(
                "inference.joint_prior: without inference.parameters covers "
                "nothing -- declare the latents first."
            )
        return None
    binds: list[Any] = []
    sugared: set[str] = set()
    seen_paths: list[str] = []
    for name, entry in parsed.items():
        where = f"inference.parameters.{name}"
        if entry.into is None:
            if entry.transform is not None or entry.fan is not None:
                raise ConfigError(
                    f"{where}: transform:/fan: describe a binding, and this "
                    "latent has no into:. Give it one, or move it to "
                    "inference.bindings."
                )
            continue
        fn, canonical = parse_transform(entry.transform, context,
                                        where=f"{where}.transform")
        fan = _merged_fan(entry.fan, canonical, where)
        binds.append(Bind(name,
                          into=_selectors(where, entry.into, fit_twin,
                                          replaced, seen_paths),
                          fn=fn, fan=fan))
        sugared.add(name)
    for index, entry in enumerate(bindings or []):
        where = f"inference.bindings[{index}]"
        if not isinstance(entry, Mapping):
            raise ConfigError(f"{where}: is a mapping; got {entry!r}.")
        check_unknown_keys(where, dict(entry), _BINDING_KEYS,
                           label="a binding:")
        latents = entry.get("latents")
        if isinstance(latents, str):
            latents = [latents]
        if not isinstance(latents, (list, tuple)) or not latents or not all(
                isinstance(item, str) for item in latents):
            raise ConfigError(f"{where}: latents: is a non-empty list of "
                              f"latent names; got {entry.get('latents')!r}.")
        for item in latents:
            if item in sugared:
                raise ConfigError(
                    f"{where}: {item!r} already carries into: on its own "
                    "parameters entry; the two spellings are mutually "
                    "exclusive."
                )
            if item not in parsed:
                raise ConfigError(
                    f"{where}: {item!r} is not a declared latent; "
                    f"inference.parameters declares {sorted(parsed)}."
                )
        into = entry.get("into")
        if isinstance(into, str):
            into = [into]
        if not isinstance(into, (list, tuple)) or not into:
            raise ConfigError(f"{where}: into: is a path or a list of paths; "
                              f"got {entry.get('into')!r}.")
        fn, canonical = parse_transform(entry.get("transform"), context,
                                        where=f"{where}.transform")
        fan = _merged_fan(entry.get("fan"), canonical, where)
        binds.append(Bind(tuple(latents),
                          into=_selectors(where, tuple(into), fit_twin,
                                          replaced, seen_paths),
                          fn=fn, fan=fan))
    refuse_duplicate_targets(seen_paths, fit_twin)
    return ParameterSpace(
        latents=[entry.latent for entry in parsed.values()],
        bindings=binds,
        joint_prior=_joint_prior(joint_prior, tuple(parsed)),
    )
