"""The transform registry, and bindings -> ParameterSpace (schema §4.7.2-3).

``Bind.into`` holds callables (``inference/parameters.py:338``);
``resolve_path_on`` (``config/paths.py:167``) compiles a dotted path into one
and refuses anything that is not an array leaf.  The registry is closed for
the same reason the derivation registry is: every entry names a callable this
package already ships, so a transform is a reference rather than arithmetic.
``beam_analysis`` is the entry that reaches limTOD: it carries a beam's maps
to the ``beam_alms`` a ``DriftScanProjector`` actually holds, so a binding
differentiates d/d(map) rather than d/d(alm).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from _rheplicant_bootstrap.types import DestinationDescriptor
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
from rheplicant.config.values import ResolutionTarget, resolve_operand
from rheplicant.core.errors import ParameterSpaceError

__all__ = ["build_space", "parse_transform"]

_NAMED = ("identity", "exp", "log", "sum", "split_rows", "unit_mean_bandpass")
_MAPPING = ("affine", "matmul", "log_link_basis", "basis_expand",
            "beam_analysis", "python")
_BINDING_KEYS = frozenset({"latents", "into", "transform", "fan"})


def _formula_parent(
    where: str, formula: str, context: ResolutionContext
) -> ResolutionTarget:
    from rheplicant.config import dimensions

    registration = dimensions._FORMULA_REGISTRY[formula]
    document_path = where.rsplit(".", 1)[0]
    return ResolutionTarget(
        DestinationDescriptor(document_path, "config_path", document_path),
        registration.result,
        registration.result.signature,
        None,
        formula_name=formula,
    )


def _operand(
    where: str,
    node: Any,
    context: ResolutionContext,
    *,
    formula: str,
    role: str,
) -> Any:
    if isinstance(node, bool) or not isinstance(node, (int, float, Mapping)):
        raise ConfigError(f"{where}: is a number or a value node; got {node!r}.")
    return jnp.asarray(
        resolve_operand(
            node,
            context,
            parent=_formula_parent(where, formula, context),
            segment=where.rsplit(".", 1)[-1],
            formula=formula,
            role=role,
        ).value
    )


def _whole(where: str, value: Any, minimum: int) -> int:
    """A configuration integer, with ``bool`` refused: ``True`` is not an
    ``nside``, and Python would otherwise let it through as ``1``."""
    if isinstance(value, bool) or not isinstance(value, int) \
            or value < minimum:
        raise ConfigError(
            f"{where}: is an integer >= {minimum}; got {value!r}."
        )
    return int(value)


def _refuse_unreachable_band(where: str, nside: int, lmax: int) -> None:
    """The band limit, in the layer's voice rather than s2fft's.

    ``map2alm_iter`` needs ``lmax >= 2 * nside - 1``, an INEQUALITY -- swept,
    not inferred, at nside 1, 2, 3, 4, 5, 6, 7, 8, 9, 12 and 16: from nside=2
    up (powers of two and not) the lower edge is exactly ``2 * nside - 1``
    and there is no upper edge at all, so at nside=4 every lmax from 7 to 89
    returns alms.  The comparison is therefore ``<`` and never ``!=``: an
    equality would reject documents s2fft accepts, which is how a
    mis-measured constant becomes a bug rather than a nuisance.

    ``nside=1`` is the one place the inequality does not describe: the floor
    would be 1, but the package has no working lmax there at all -- measured,
    every lmax from 0 to 11 fails inside s2fft with "Need at least one array
    to stack".  It gets its own refusal, before the band limit speaks, so the
    promise "this nside admits lmax >= N" is never made where it is false.
    """
    if nside < 2:
        raise ConfigError(
            f"{where}.beam_analysis: nside: {nside} has no lmax that works. "
            "map2alm_iter needs nside >= 2 -- measured, at nside=1 every "
            "lmax from 0 to 11 fails inside s2fft with 'Need at least one "
            "array to stack', including the one the band limit below would "
            "otherwise admit. Raise nside: to 2 or more."
        )
    floor = 2 * nside - 1
    if lmax < floor:
        raise ConfigError(
            f"{where}.beam_analysis: lmax: {lmax} is below the band limit "
            f"nside: {nside} admits. map2alm_iter needs "
            f"lmax >= 2 * nside - 1, so this nside admits lmax >= {floor} "
            "(swept from nside=2 to nside=16, powers of two and not: the "
            "lower edge is exactly 2 * nside - 1 every time, and there is no "
            "upper edge). Below it s2fft's healpix transform fails inside "
            "itself with a shape error naming neither key. Raise lmax: to "
            f"{floor} or more, or lower nside:. An lmax ABOVE the limit is "
            "legal and is not refused here."
        )


def _beam_analysis(where: str, body: Mapping) -> tuple[Any, str]:
    """``{beam_analysis: {nside, lmax, iterations}}`` -> vmapped map2alm_iter.

    The transform that makes a beam-map gradient the quantity the document
    meant.  A ``DriftScanProjector``'s only non-static field is ``beam_alms``
    (``radio/sky/driftscan.py:189-203``), so without this the only binding a
    document can write is ``into: ...projector.beam_alms``, and what comes
    back is d(chi2)/d(alm) rather than d(chi2)/d(map) -- a different quantity
    in a different basis, finite and correctly shaped either way.

    ``map2alm_iter``, not ``map2alm_quad``: the two differ by ``npix/4pi`` and
    are not interchangeable.  ``_iter`` returns true (healpy-convention) alms
    and is what ``from_beam_maps`` feeds into ``beam_alms``
    (``driftscan.py:301``); ``_quad`` returns quadrature alms and is the
    *sky*'s transform (``sky_to_alms``, ``driftscan.py:605-606``).  Picking
    the visible one "silently rescales the beam by npix/4pi"
    (``driftscan.py:269``) -- measured 15.06x against npix/4pi = 15.28 at
    nside=4, finite and correctly shaped and wrong.

    ``map2alm_iter`` is a single-map function -- ``nside``, ``lmax``,
    ``iterations`` and ``npol`` all sit behind a bare ``*``, and ``nside`` and
    ``lmax`` have no defaults -- so the ``(n_freq, n_pix) -> (n_freq, n_alm)``
    shape of the schema's table is the ``jax.vmap``'s, not the function's.
    This is ``driftscan.py:300-302`` verbatim.

    limTOD is imported only after the grammar has been checked, so a
    malformed ``beam_analysis:`` is refused in this layer's voice on an
    install that lacks it rather than raising ``ImportError`` first.
    """
    check_unknown_keys(where, dict(body),
                       frozenset({"nside", "lmax", "iterations"}),
                       label="beam_analysis:")
    missing = sorted({"nside", "lmax"} - set(body))
    if missing:
        raise ConfigError(
            f"{where}.beam_analysis: requires nside: and lmax: -- "
            "map2alm_iter takes both as keywords and defaults neither; "
            f"missing {missing}."
        )
    numbers = {
        key: _whole(f"{where}.beam_analysis.{key}", body[key], minimum)
        for key, minimum in (("nside", 1), ("lmax", 0), ("iterations", 0))
        if key in body
    }
    nside = numbers["nside"]
    lmax = numbers["lmax"]
    _refuse_unreachable_band(where, nside, lmax)
    # What is left is ``iterations``, and only where the document declared it:
    # the package's own default is never restated here.
    extra = {key: value for key, value in numbers.items()
             if key not in ("nside", "lmax")}

    import limtod_jax as ltj

    def analyse(maps: Any) -> Any:
        return ltj.map2alm_iter(maps, nside=nside, lmax=lmax, **extra)

    return jax.vmap(analyse), "broadcast"


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
        scale = _operand(
            f"{where}.affine.scale", body.get("scale", 1.0), context,
            formula="transform_affine", role="scale",
        )
        offset = _operand(
            f"{where}.affine.offset", body.get("offset", 0.0), context,
            formula="transform_affine", role="offset",
        )
        return (lambda v, _s=scale, _o=offset: _s * v + _o), "broadcast"
    if head == "matmul":
        check_unknown_keys(where, dict(body), frozenset({"design"}),
                           label="matmul:")
        if "design" not in body:
            raise ConfigError(f"{where}.matmul: requires design: -- a value "
                              "node for the design matrix.")
        design_where = f"{where}.matmul.design"
        design = jnp.asarray(
            resolve_operand(
                body["design"],
                context,
                parent=_formula_parent(design_where, "matmul", context),
                segment="design",
                formula="matmul",
                role="design",
            ).value
        )
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
    if head == "beam_analysis":
        return _beam_analysis(where, body)
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


def _b4_refuse_unbound_latents(parsed: Any, binds: list, bindings: Any) -> None:
    """A latent declared and bound to nothing, named by its key (check B4).

    ``ParameterSpace.__check_init__`` already refuses this, in two sentences
    that are right about the physics and unusable as advice from a document:
    the first names the LATENT ("the posterior would just return the prior")
    and not the key that declared it, and the second -- the one reached when
    NO latent is bound -- advises ``ParameterSpace.raw(...)``, a Python API no
    YAML author can reach.  Both arrive as ``ParameterSpaceError``, a sibling
    of :class:`~rheplicant.config.errors.ConfigError` rather than a subclass
    (0.2 C-12), so a caller catching config refusals by name misses them.

    So the same set difference is taken HERE, one call earlier, and said with
    the two things the document can act on: the key, and whether
    ``inference.bindings`` is the other place to write the binding.  This is a
    re-voicing and not a second opinion -- the package's own refusal still
    stands behind it for every caller that builds a ``ParameterSpace``
    directly.

    It runs AFTER :func:`refuse_duplicate_targets` deliberately: a document
    wrong in both ways hears about the two bindings fighting over one leaf
    first, which is the more specific fault of the two.
    """
    bound = {name for bind in binds for name in bind.latents}
    dead = sorted(set(parsed) - bound)
    if not dead:
        return
    keys = ", ".join(f"inference.parameters.{name}" for name in dead)
    where = ("an inference.bindings entry -- this document declares "
             "bindings, and none of them names it"
             if bindings else
             "an inference.bindings entry -- this document declares none yet")
    raise ConfigError(
        f"{keys}: declared and bound to nothing, so the fit would sample it "
        "without it ever reaching the model and its posterior would just "
        f"return its prior. Give it into: on its own entry, or name it in "
        f"{where} (check B4)."
    )


def _c17_stochastic_nodes(fit_twin: Any) -> tuple[str, ...]:
    """The fit twin's stages that draw their own randomness, by node id.

    The same detector ``refuse_stochastic_stages`` uses -- the operators' own
    ``requires`` declaration -- asked directly, so the leg that fired is
    decided by a FACT ABOUT THE TWIN rather than by matching words in the
    exception's text.  A message match would rot the first time the package
    rewords a sentence, and silently: the wrong leg's wording would simply
    start appearing.

    Returns the node ids because that is exactly what
    ``inference.twin.without:`` takes, which is the whole point of naming
    them.  ``core.contract`` and not ``inference.parameters``: the detector
    lives in core, and importing the inference layer here would put numpyro
    in ``sys.modules`` at config-import time.
    """
    from rheplicant.core.contract import RANDOMNESS, stages_requiring

    return tuple(node_id for node_id, _ in stages_requiring(fit_twin,
                                                            RANDOMNESS))


def _c17_bound_paths(parsed: Any, bindings: Any,
                     fit_twin: Any) -> tuple[str, ...]:
    """``(document key, path as written, path as the twin spells it)`` rows.

    Both spellings, because ``config/paths.py`` already argues that naming
    only one of them reuses wording at the reader's expense -- and here the
    package's own sentence names NEITHER: its headline refusal is *"Bind for
    ('g',) produces shape () for `into` selector 0"*, which quotes a latent
    tuple and a selector index.  :func:`resolve_path_on` is the one reader
    that holds the two spellings side by side, so it supplies them.
    """
    rows: list[tuple[str, str]] = []
    for name, entry in (parsed or {}).items():
        for path in entry.into or ():
            rows.append((f"inference.parameters.{name}", path))
    for index, entry in enumerate(bindings or []):
        if not isinstance(entry, Mapping):
            continue
        into = entry.get("into")
        if isinstance(into, str):
            into = [into]
        for path in into or ():
            rows.append((f"inference.bindings[{index}]", path))
    spelled = []
    for where, path in rows:
        try:
            keystr = resolve_path_on(path, fit_twin).keystr
        except ConfigError:
            # Unreachable through `build_space`, which resolved every one of
            # these a moment ago -- but this function's whole job is to
            # improve a refusal, so it may not raise a second one on the way.
            keystr = "?"
        spelled.append(f"{where} -> {path!r}, which the twin spells {keystr!r}")
    return tuple(spelled)


def _c17_validate_space(space: Any, fit_twin: Any, parsed: Any,
                        bindings: Any) -> None:
    """``ParameterSpace.validate`` at load time, in this layer's voice (C17).

    Measured: the package ships this check, four call sites call it, and
    **all four are at exit time** -- so a document whose bindings do not fit
    its twin loads clean and fails at the fit, past every resource the layer
    built.  It reads shapes only (``jax.eval_shape``; measured 0.77-1.11 ms,
    live device arrays 15 before and 15 after), so running it at load costs
    the document nothing it will not pay anyway.

    What it adds here, stated exactly, because the plan's list is wider than
    the measurement: :func:`refuse_stochastic_stages`, and the per-binding and
    per-leaf shape/dtype comparisons.  It does **not** add B1, B2 or B3
    through a config document: ``config/paths.py`` refuses an unreachable or
    duplicated target before a ``Bind`` is ever constructed, and
    ``_aliased_leaf_paths`` is measured ``{}`` on a config-built twin, so
    those three legs are unreachable from here and are not claimed.

    **Those two legs get two different sentences, because they are two
    different faults and one wording cannot be true of both.**  The
    shape/dtype legs are a mismatch between the space and the twin, and the
    package names latents and selector positions there.  The stochastic leg
    is a fault in the TWIN ALONE -- the space fits perfectly -- and the
    package names a NODE.  A single wording that says "the space does not fit
    the twin" and "that sentence names latents and selector positions" is
    false on the stochastic leg in both halves, and pastes an innocent
    binding table underneath as though it were evidence.  Worse, the only
    advice surviving from the package there is ``Assembly.without(node_id)``
    -- a Python API no YAML author can reach -- so a re-voicing that adds
    nothing leaves an R4 advice loop where a document remedy exists.

    The twin handed over is the **FIT** twin.  Measured: the full twin raises
    on every document that writes ``inference.twin.without:``, because the
    bindings were resolved against the repaired twin and the dropped node's
    leaves are gone.

    ``space`` may be ``None`` -- a document with no ``inference.parameters``
    has nothing to validate, and ``build_space`` returns ``None`` for it.
    """
    if space is None:
        return
    try:
        space.validate(fit_twin)
    except ParameterSpaceError as exc:
        drawing = _c17_stochastic_nodes(fit_twin)
        if drawing:
            # `validate` runs `refuse_stochastic_stages` FIRST, so a twin with
            # a drawing stage always earns that sentence and never one of the
            # shape ones -- which is what makes this branch exact.
            raise ConfigError(
                "inference: the twin this document fits with still draws its "
                "own randomness, so it is the twin at fault here and not the "
                "parameter space. The package refuses it in its own words: "
                f"{exc} From a document the repair is "
                f"inference.twin.without: {list(drawing)} -- that drops the "
                "stage from the twin the FIT uses while inference.observed's "
                "simulation still defaults to the full twin, so the scatter "
                "goes on entering the data it was written for (check C17)."
            ) from exc
        spelled = "; ".join(_c17_bound_paths(parsed, bindings, fit_twin))
        raise ConfigError(
            "inference: the parameter space this document declares does not "
            "fit the twin it binds into, and the fit would be the first "
            f"thing to say so. The package refuses it in its own words: {exc} "
            "That sentence names latents and selector positions rather than "
            f"document keys, so here is what this document bound: {spelled}. "
            "Each path is written twice -- as the document wrote it, then as "
            "jax.tree_util.keystr spells it, which is the form the package's "
            "other refusals quote (check C17)."
        ) from exc


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
    _b4_refuse_unbound_latents(parsed, binds, bindings)
    return ParameterSpace(
        latents=[entry.latent for entry in parsed.values()],
        bindings=binds,
        joint_prior=_joint_prior(joint_prior, tuple(parsed)),
    )
