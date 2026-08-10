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

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.sections.observation import SiteFacts, _dimensioned
from rheplicant.config.symbols import resolve_extent
from rheplicant.config.values import resolve_value

__all__ = ["PointingBuild", "compile_pointing"]

_MODES = ("none", "drift", "tracked", "baked")
_KEYS = {
    "none": frozenset({"mode"}),
    "drift": frozenset({"mode", "az_deg", "el_deg", "selfrot_deg",
                        "materialise", "lst"}),
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


def _angle(where: str, node: Any, context: ResolutionContext) -> float:
    return float(_dimensioned(where, node, context, dimension="angle",
                              what="an angle").value)


def _lst(spec: Any, context: ResolutionContext, *, time_s, epoch_unix_s,
         site: SiteFacts):
    n_time = int(time_s.shape[0])
    if not isinstance(spec, Mapping) or not spec:
        raise ConfigError(
            "pointing.lst: is one of {mode: uniform_turn, n_time: ..., "
            "lst0_deg: ...}, {mode: from_site}, or {from_file: {path: ..., "
            "format: ...}}."
        )
    if "from_file" in spec:
        check_unknown_keys("observation.pointing", dict(spec),
                           frozenset({"from_file"}), label="lst:")
        resolved = resolve_value({"file": dict(spec["from_file"])}, context)
        if resolved.unit is not None and resolved.unit.dimension != "angle":
            raise ConfigError(
                f"pointing.lst.from_file: declares unit "
                f"{resolved.unit.canonical!r}; LSTs are angles in degrees."
            )
        lst = jnp.asarray(resolved.value)
        if lst.shape != (n_time,):
            raise ConfigError(
                f"pointing.lst.from_file: is (n_time,) = ({n_time},); got "
                f"{tuple(lst.shape)}."
            )
        return lst
    mode = spec.get("mode")
    if mode == "uniform_turn":
        check_unknown_keys("observation.pointing", dict(spec),
                           _LST_KEYS["uniform_turn"], label="lst:")
        declared = resolve_extent(spec.get("n_time", "n_time"),
                                  context.shape_scope)
        if declared != n_time:
            raise ConfigError(
                f"pointing.lst: n_time is {declared} but the run's time grid "
                f"has {n_time} samples -- a uniform turn is defined over the "
                "run's own axis."
            )
        lst0 = 0.0
        if "lst0_deg" in spec:
            lst0 = _angle("pointing.lst.lst0_deg", spec["lst0_deg"], context)
        from rheplicant.radio import DriftScanProjector

        return DriftScanProjector.uniform_lst_grid(n_time, lst0)
    if mode == "from_site":
        check_unknown_keys("observation.pointing", dict(spec),
                           _LST_KEYS["from_site"], label="lst:")
        missing = [key for key, value in (
            ("site.lat_deg", site.lat_deg),
            ("site.lon_deg", site.lon_deg),
            ("site.alt_m", site.alt_m),
            ("time.epoch", epoch_unix_s),
        ) if value is None]
        if missing:
            raise ConfigError(
                "pointing.lst: mode: from_site computes LSTs from the site "
                "and the epoch (D-C2); this document is missing "
                f"{missing}."
            )
        from rheplicant.radio.site import lst_grid_deg

        return jnp.asarray(lst_grid_deg(
            lat_deg=site.lat_deg, lon_deg=site.lon_deg, alt_m=site.alt_m,
            time_s=time_s, epoch_unix_s=epoch_unix_s))
    raise ConfigError(
        f"pointing.lst: mode is 'uniform_turn' or 'from_site' (or write "
        f"from_file: instead); got {mode!r}."
    )


def compile_pointing(spec: Any, context: ResolutionContext, *, time_s,
                     epoch_unix_s, site: SiteFacts) -> PointingBuild:
    """Compile the pointing section against the run's own time axis."""
    if spec is None:
        spec = {"mode": "none"}
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"observation.pointing: is a mapping; got {type(spec).__name__}."
        )
    mode = spec.get("mode", "none")
    if mode not in _MODES:
        raise ConfigError(
            f"observation.pointing: mode is one of {list(_MODES)}; got {mode!r}."
        )
    check_unknown_keys("observation.pointing", dict(spec), _KEYS[mode],
                       label=f"mode: {mode}")
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
            pointing=None, extra={},
            provenance={f"pointing/{key}": value
                        for key, value in provenance.items()})

    extra: dict[str, Any] = {}
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
        if (not isinstance(materialise, list)
                or any(entry not in _MATERIALISE for entry in materialise)):
            raise ConfigError(
                f"observation.pointing: materialise entries are "
                f"{list(_MATERIALISE)}; got {materialise!r}."
            )
        az = _angle("pointing.az_deg", spec.get("az_deg", 0.0), context) \
            if "az_deg" in spec else 0.0
        el = _angle("pointing.el_deg", spec["el_deg"], context) \
            if "el_deg" in spec else 90.0
        pointing = None
        if "pointing" in materialise:
            pointing = jnp.broadcast_to(
                jnp.asarray([az, el], dtype=context.dtype), (n_time, 2))
        if "selfrot_deg" in materialise:
            selfrot = 0.0
            if "selfrot_deg" in spec:
                selfrot = _angle("pointing.selfrot_deg", spec["selfrot_deg"],
                                 context)
            extra["selfrot_deg"] = jnp.broadcast_to(
                jnp.asarray(selfrot, dtype=context.dtype), (n_time,))
        if "lst" in spec:
            extra["lst_deg"] = _lst(spec["lst"], context, time_s=time_s,
                                    epoch_unix_s=epoch_unix_s, site=site)
        return PointingBuild(pointing=pointing, extra=extra, provenance={})

    # mode == "tracked"
    if "table" not in spec:
        raise ConfigError(
            "observation.pointing: mode: tracked requires table: -- a value "
            "node of shape (n_time, 2), az/el in degrees."
        )
    resolved = _dimensioned("pointing.table", spec["table"], context,
                            dimension="angle", what="the az/el track")
    table = jnp.asarray(resolved.value, dtype=context.dtype)
    if table.shape != (n_time, 2):
        raise ConfigError(
            f"pointing.table: is (n_time, 2) = ({n_time}, 2); got "
            f"{tuple(table.shape)}."
        )
    if "lst" not in spec:
        raise ConfigError(
            "observation.pointing: mode: tracked requires lst: -- both real "
            "sky engines read coords.extra['lst_deg'] and nothing else in "
            "the package produces it for a tracked scan."
        )
    extra["lst_deg"] = _lst(spec["lst"], context, time_s=time_s,
                            epoch_unix_s=epoch_unix_s, site=site)
    if "selfrot" in spec:
        selfrot = jnp.asarray(
            _dimensioned("pointing.selfrot", spec["selfrot"], context,
                         dimension="angle", what="the self-rotation track"
                         ).value, dtype=context.dtype)
        if selfrot.shape != (n_time,):
            raise ConfigError(
                f"pointing.selfrot: is (n_time,) = ({n_time},); got "
                f"{tuple(selfrot.shape)}."
            )
        extra["selfrot_deg"] = selfrot
    return PointingBuild(pointing=table, extra=extra, provenance={})
