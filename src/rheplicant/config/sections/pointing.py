"""``observation.pointing`` (schema §4.1.4): what the loader writes, it was told.

``materialise:`` exists because the two sky engines READ pointing sidecars
differently -- ``DriftScanProjector`` cross-checks ``selfrot_deg`` only if
present, ``GeneralPointingProjector`` zero-defaults it when absent -- so a
loader that guessed either way would silently change which observation a
cross-engine comparison simulated.

``lst:`` has three routes, all landing on ``coords.extra["lst_deg"]`` [deg]:
``mode: uniform_turn`` (the FFT grid, endpoint excluded), ``from_file:``, and
``mode: from_site`` -- D-C2's adapter, which is what ``site.lon_deg`` and
``site.alt_m`` exist to feed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import record_resolved_delivery
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.observation import SiteFacts, _dimensioned
from rheplicant.config.symbols import resolve_extent
from rheplicant.config.values import resolve_value

__all__ = ["PointingBuild", "compile_pointing", "pointing_extra_keys"]

_MODES = ("none", "drift", "tracked", "baked")
_KEYS = {
    "none": frozenset({"mode"}),
    "drift": frozenset({"mode", "az_deg", "el_deg", "selfrot_deg", "materialise", "lst"}),
    "tracked": frozenset({"mode", "table", "lst", "selfrot"}),
    "baked": frozenset({"mode", "provenance"}),
}
_MATERIALISE = ("pointing", "selfrot_deg")
_LST_KEYS = {
    "uniform_turn": frozenset({"mode", "n_time", "lst0_deg"}),
    "from_site": frozenset({"mode"}),
}


class PointingBuild(NamedTuple):
    """What the pointing section writes: the ``Coordinates.pointing`` array
    (or None), the ``coords.extra`` entries it materialised, and the baked
    provenance destined for ``meta``."""

    pointing: Any
    extra: dict[str, Any]
    provenance: dict[str, Any]


def pointing_extra_keys(spec: Any) -> frozenset[str]:
    """The ``coords.extra`` keys :func:`compile_pointing` writes for ``spec``.

    **One binding, two callers** (§2.2): :func:`compile_pointing` below reads
    it to decide what to materialise, and ``config.preflight.ingest`` reads it
    to know which keys a run will carry before anything is built.  The set is
    a four-branch function of ``mode``, ``materialise:``, ``lst:`` and
    ``selfrot:``, and a pre-flight pass that re-derived those four branches
    inline would be a second validator for one property -- the divergence
    §2.2 exists to stop, and the one this layer has already paid for once.

    **``frozenset()`` for a spec this function cannot parse**, and that is a
    stand-down rather than an answer: a non-mapping, an unknown ``mode``, a
    ``materialise:`` that is not a list of the two legal entries are each
    :func:`compile_pointing`'s own refusal, said with the shape it got.  A
    reader that guessed a key set for one of them would be answering about a
    document nobody can run.

    ``mode: none`` and ``mode: baked`` write nothing: the first has no
    geometry and the second has it inside the matrix, where the only record
    is ``provenance:`` and that lands in ``meta``, not in ``coords.extra``.
    """
    if spec is None:
        spec = {"mode": "none"}
    if not isinstance(spec, Mapping):
        return frozenset()
    mode = spec.get("mode", "none")
    keys: set[str] = set()
    if mode == "drift":
        materialise = spec.get("materialise")
        if isinstance(materialise, list) and "selfrot_deg" in materialise:
            keys.add("selfrot_deg")
        if "lst" in spec:
            keys.add("lst_deg")
    elif mode == "tracked":
        if "lst" in spec:
            keys.add("lst_deg")
        if "selfrot" in spec:
            keys.add("selfrot_deg")
    return frozenset(keys)


def _angle(where: str, node: Any, context: ResolutionContext) -> float:
    return float(_dimensioned(where, node, context, dimension="angle", what="an angle").value)


def _lst(spec: Any, context: ResolutionContext, *, time_s, epoch_unix_s, site: SiteFacts):
    n_time = int(time_s.shape[0])
    if not isinstance(spec, Mapping) or not spec:
        raise ConfigError(
            "pointing.lst: is one of {mode: uniform_turn, n_time: ..., "
            "lst0_deg: ...}, {mode: from_site}, or {from_file: {path: ..., "
            "format: ...}}."
        )
    if "from_file" in spec:
        check_unknown_keys(
            "observation.pointing", dict(spec), frozenset({"from_file"}), label="lst:"
        )
        destination = DestinationDescriptor(
            "observation.pointing.lst.from_file",
            "config_path",
            "observation.pointing.lst.from_file",
        )
        resolved = resolve_value(
            {"file": dict(spec["from_file"])},
            context,
            destination=destination,
        )
        if resolved.unit is not None and resolved.unit.dimension != "angle":
            raise ConfigError(
                f"pointing.lst.from_file: declares unit "
                f"{resolved.unit.canonical!r}; LSTs are angles in degrees."
            )
        lst = jnp.asarray(resolved.value)
        if lst.shape != (n_time,):
            raise ConfigError(
                f"pointing.lst.from_file: is (n_time,) = ({n_time},); got {tuple(lst.shape)}."
            )
        record_resolved_delivery(context, destination, resolved.unit)
        return lst
    mode = spec.get("mode")
    if mode == "uniform_turn":
        check_unknown_keys(
            "observation.pointing", dict(spec), _LST_KEYS["uniform_turn"], label="lst:"
        )
        declared = resolve_extent(
            spec["n_time"]
            if "n_time" in spec
            else context.use_default("observation.pointing.lst.n_time", "n_time"),
            context.shape_scope,
        )
        if declared != n_time:
            raise ConfigError(
                f"pointing.lst: n_time is {declared} but the run's time grid "
                f"has {n_time} samples -- a uniform turn is defined over the "
                "run's own axis."
            )
        if "lst0_deg" in spec:
            lst0 = _angle("observation.pointing.lst.lst0_deg", spec["lst0_deg"], context)
        else:
            lst0 = context.use_default("observation.pointing.lst.lst0_deg", 0.0)
        from rheplicant.radio import DriftScanProjector

        return DriftScanProjector.uniform_lst_grid(n_time, lst0)
    if mode == "from_site":
        check_unknown_keys("observation.pointing", dict(spec), _LST_KEYS["from_site"], label="lst:")
        missing = [
            key
            for key, value in (
                ("site.lat_deg", site.lat_deg),
                ("site.lon_deg", site.lon_deg),
                ("site.alt_m", site.alt_m),
                ("time.epoch", epoch_unix_s),
            )
            if value is None
        ]
        if missing:
            raise ConfigError(
                "pointing.lst: mode: from_site computes LSTs from the site "
                "and the epoch (D-C2); this document is missing "
                f"{missing}."
            )
        from rheplicant.radio.site import lst_grid_deg

        return jnp.asarray(
            lst_grid_deg(
                lat_deg=site.lat_deg,
                lon_deg=site.lon_deg,
                alt_m=site.alt_m,
                time_s=time_s,
                epoch_unix_s=epoch_unix_s,
            )
        )
    raise ConfigError(
        f"pointing.lst: mode is 'uniform_turn' or 'from_site' (or write "
        f"from_file: instead); got {mode!r}."
    )


def compile_pointing(
    spec: Any, context: ResolutionContext, *, time_s, epoch_unix_s, site: SiteFacts
) -> PointingBuild:
    """Compile the pointing section against the run's own time axis."""
    if spec is None:
        spec = context.use_default("observation.pointing", {"mode": "none"})
    if not isinstance(spec, Mapping):
        raise ConfigError(f"observation.pointing: is a mapping; got {type(spec).__name__}.")
    mode = (
        spec["mode"] if "mode" in spec else context.use_default("observation.pointing.mode", "none")
    )
    if mode not in _MODES:
        raise ConfigError(f"observation.pointing: mode is one of {list(_MODES)}; got {mode!r}.")
    check_unknown_keys("observation.pointing", dict(spec), _KEYS[mode], label=f"mode: {mode}")
    n_time = int(time_s.shape[0])

    if mode == "none":
        return PointingBuild(pointing=None, extra={}, provenance={})

    if mode == "baked":
        provenance = spec.get("provenance")
        if not isinstance(provenance, Mapping) or not provenance:
            raise ConfigError(
                "observation.pointing: mode: baked requires provenance: -- "
                "the geometry is inside the matrix and cannot be verified, so "
                "who built it and for what site is the ONLY record there is."
            )
        return PointingBuild(
            pointing=None,
            extra={},
            provenance={f"pointing/{key}": value for key, value in provenance.items()},
        )

    extra: dict[str, Any] = {}
    # The single binding: which keys get written is :func:`pointing_extra_keys`'
    # answer here and in the pre-flight pass alike, so the two cannot drift.
    # Read AFTER the grammar refusals above, which is what makes the
    # "cannot parse" stand-down unreachable from this call site.
    keys = pointing_extra_keys(spec)
    if mode == "drift":
        if "materialise" not in spec:
            raise ConfigError(
                "observation.pointing: mode: drift requires materialise: -- "
                "the list of Coordinates fields the loader writes. Written, "
                "not inferred: DriftScanProjector cross-checks selfrot_deg "
                "only if present while GeneralPointingProjector zero-defaults "
                "it, so a guessed sidecar silently changes which observation "
                "a cross-engine comparison simulated."
            )
        materialise = spec["materialise"]
        if not isinstance(materialise, list) or any(
            entry not in _MATERIALISE for entry in materialise
        ):
            raise ConfigError(
                f"observation.pointing: materialise entries are "
                f"{list(_MATERIALISE)}; got {materialise!r}."
            )
        az = (
            _angle("observation.pointing.az_deg", spec["az_deg"], context)
            if "az_deg" in spec
            else context.use_default("observation.pointing.az_deg", 0.0)
        )
        el = (
            _angle("observation.pointing.el_deg", spec["el_deg"], context)
            if "el_deg" in spec
            else context.use_default("observation.pointing.el_deg", 90.0)
        )
        pointing = None
        if "pointing" in materialise:
            pointing = jnp.broadcast_to(jnp.asarray([az, el], dtype=context.dtype), (n_time, 2))
        if "selfrot_deg" in keys:
            if "selfrot_deg" in spec:
                selfrot = _angle("observation.pointing.selfrot_deg", spec["selfrot_deg"], context)
            else:
                selfrot = context.use_default("observation.pointing.selfrot_deg", 0.0)
            extra["selfrot_deg"] = jnp.broadcast_to(
                jnp.asarray(selfrot, dtype=context.dtype), (n_time,)
            )
        if "lst_deg" in keys:
            extra["lst_deg"] = _lst(
                spec["lst"], context, time_s=time_s, epoch_unix_s=epoch_unix_s, site=site
            )
        return PointingBuild(pointing=pointing, extra=extra, provenance={})

    # mode == "tracked"
    if "table" not in spec:
        raise ConfigError(
            "observation.pointing: mode: tracked requires table: -- a value "
            "node of shape (n_time, 2), az/el in degrees."
        )
    resolved = _dimensioned(
        "observation.pointing.table",
        spec["table"],
        context,
        dimension="angle",
        what="the az/el track",
    )
    table = jnp.asarray(resolved.value, dtype=context.dtype)
    if table.shape != (n_time, 2):
        raise ConfigError(
            f"pointing.table: is (n_time, 2) = ({n_time}, 2); got {tuple(table.shape)}."
        )
    if "lst" not in spec:
        raise ConfigError(
            "observation.pointing: mode: tracked requires lst: -- both real "
            "sky engines read coords.extra['lst_deg'] and nothing else in "
            "the package produces it for a tracked scan."
        )
    # `keys` carries lst_deg here by construction -- the refusal above is what
    # makes it certain -- and it is read rather than assumed so that this
    # branch has one source of truth like the drift branch does.
    if "lst_deg" in keys:
        extra["lst_deg"] = _lst(
            spec["lst"], context, time_s=time_s, epoch_unix_s=epoch_unix_s, site=site
        )
    if "selfrot_deg" in keys:
        selfrot = jnp.asarray(
            _dimensioned(
                "observation.pointing.selfrot",
                spec["selfrot"],
                context,
                dimension="angle",
                what="the self-rotation track",
            ).value,
            dtype=context.dtype,
        )
        if selfrot.shape != (n_time,):
            raise ConfigError(
                f"pointing.selfrot: is (n_time,) = ({n_time},); got {tuple(selfrot.shape)}."
            )
        extra["selfrot_deg"] = selfrot
    return PointingBuild(pointing=table, extra=extra, provenance={})
