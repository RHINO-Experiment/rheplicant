"""``observation:`` field loaders (schema §4.1.1).

Grids come back in canonical units (Hz, s) because ``resolve_value`` already
converts; what this module adds is the DIMENSION check -- a frequency grid
declared in kelvin is refused by what it means, not by how it is spelled --
and the handful of container rules the core objects enforce too late or not
at all (meta hashability names the key; flags are boolean, TRUE = BAD;
data matches the declared grids).

``pointing:`` and ``switching:`` compile in their own modules; ``from_file:``
in :mod:`rheplicant.config.sections.ingest`; the public ``build_observation``
that orchestrates all of them ships with ingest, because the two forms are
mutually exclusive and one function must own that refusal.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.resources import check_unknown_keys
from rheplicant.config.values import resolve_value
from rheplicant.core.environment import Environment

__all__ = ["ObservationBuild", "SiteFacts", "build_observation"]

_OBSERVATION_KEYS = frozenset(
    {"meta", "freq", "time", "site", "pointing", "switching", "environment",
     "extra", "aux", "data", "from_file"}
)
_FREQ_KEYS = frozenset({"grid"})
_TIME_KEYS = frozenset({"grid", "epoch", "integration_time", "channel_width"})
_SITE_KEYS = frozenset({"lat_deg", "lon_deg", "alt_m"})
_ENVIRONMENT_KEYS = frozenset({"temperature", "humidity", "extra"})
_AUX_KEYS = frozenset({"flags"})


class SiteFacts(NamedTuple):
    """``observation.site``, recorded; ``lat_deg`` fans to projectors (Plan 3
    checks agreement), and all three feed ``lst: {mode: from_site}``."""

    lat_deg: float | None
    lon_deg: float | None
    alt_m: float | None


class ObservationBuild(NamedTuple):
    """Everything the ``observation:`` section produced, before the State."""

    time_s: Any
    freq_hz: Any
    epoch_unix_s: float | None
    integration_time_s: float | None
    channel_width_hz: float | None
    site: SiteFacts
    env: Environment | None
    meta: dict[str, Any]
    aux: dict[str, Any]
    data: Any
    pointing: Any
    extra: dict[str, Any]
    switch_order: tuple[str, ...]
    ingest: Any


def _dimensioned(where: str, node: Any, context: ResolutionContext, *,
                 dimension: str, what: str):
    """Resolve a value node and refuse it by MEANING: the declared unit's
    dimension must be ``dimension``."""
    resolved = resolve_value(node, context)
    unit = resolved.unit
    if unit is None or unit.dimension != dimension:
        got = ("no unit" if unit is None
               else f"unit {unit.canonical!r} (dimension {unit.dimension})")
        raise ConfigError(
            f"{where}: declares {got}; it is {what} and takes a {dimension} "
            "unit."
        )
    return resolved


def _one_d(where: str, value: Any):
    array = jnp.asarray(value)
    if array.ndim != 1:
        raise ConfigError(f"{where}: is 1-D; got shape {tuple(array.shape)}.")
    return array


def _freq_grid(spec: Any, context: ResolutionContext):
    if not isinstance(spec, Mapping) or "grid" not in spec:
        raise ConfigError(
            "observation.freq: requires grid: -- a value node in a frequency "
            "unit; it becomes Coordinates.freq, (n_freq,), Hz."
        )
    check_unknown_keys("observation.freq", dict(spec), _FREQ_KEYS, label="freq:")
    resolved = _dimensioned("observation.freq.grid", spec["grid"], context,
                            dimension="frequency", what="the frequency axis")
    return _one_d("observation.freq.grid", resolved.value)


def _time_facts(spec: Any, context: ResolutionContext):
    """``(time_s, epoch_unix_s, integration_time_s, channel_width_hz)``."""
    if not isinstance(spec, Mapping) or "grid" not in spec:
        raise ConfigError(
            "observation.time: requires grid: -- a value node in a time unit, "
            "RELATIVE to the start of the run (core/coordinates.py documents "
            "the float32 failure this prevents); the absolute epoch is "
            "time.epoch in unix_s."
        )
    check_unknown_keys("observation.time", dict(spec), _TIME_KEYS, label="time:")
    resolved = _dimensioned("observation.time.grid", spec["grid"], context,
                            dimension="time", what="the time axis")
    time_s = _one_d("observation.time.grid", resolved.value)
    epoch = integration = width = None
    if "epoch" in spec:
        epoch = float(_dimensioned(
            "observation.time.epoch", spec["epoch"], context,
            dimension="time_epoch",
            what="an absolute moment (declare unit: unix_s)").value)
    if "integration_time" in spec:
        integration = float(_dimensioned(
            "observation.time.integration_time", spec["integration_time"],
            context, dimension="time", what="a duration").value)
    if "channel_width" in spec:
        width = float(_dimensioned(
            "observation.time.channel_width", spec["channel_width"], context,
            dimension="frequency", what="a bandwidth").value)
    return time_s, epoch, integration, width


def _hashable(key: str, value: Any) -> Any:
    """Coerce a list to a tuple and refuse an unhashable meta value, naming
    ``key``. Shared between ``_meta``'s own sweep and the baked-pointing
    provenance merge in :func:`build_observation` -- provenance values reach
    ``meta`` AFTER ``_meta``'s own gate has already run, so they need the
    same check applied at the merge, not skipped because it ran once."""
    if isinstance(value, list):
        value = tuple(value)
    try:
        hash(value)
    except TypeError:
        raise ConfigError(
            f"observation.meta.{key}: meta values are hashable scalars -- "
            "they key the jit cache (core/frozen.py) -- and "
            f"{type(value).__name__} is not one. Arrays and mappings "
            "belong in observation.extra / environment / aux."
        ) from None
    return value


def _meta(spec: Any) -> dict[str, Any]:
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"observation.meta: is a mapping; got {type(spec).__name__}."
        )
    out: dict[str, Any] = {}
    for key, value in spec.items():
        if not isinstance(key, str):
            raise ConfigError(f"observation.meta keys are strings; got {key!r}.")
        out[key] = _hashable(key, value)
    return out


def _site(spec: Any, context: ResolutionContext) -> SiteFacts:
    if spec is None:
        return SiteFacts(None, None, None)
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"observation.site: is a mapping; got {type(spec).__name__}."
        )
    check_unknown_keys("observation.site", dict(spec), _SITE_KEYS, label="site:")
    values: dict[str, float | None] = {}
    for key, dimension, what in (
        ("lat_deg", "angle", "the site latitude"),
        ("lon_deg", "angle", "the site longitude, east positive"),
        ("alt_m", "length", "the site altitude"),
    ):
        values[key] = None
        if key in spec:
            values[key] = float(_dimensioned(
                f"observation.site.{key}", spec[key], context,
                dimension=dimension, what=what).value)
    return SiteFacts(**values)


def _environment(spec: Any, context: ResolutionContext) -> Environment | None:
    if spec is None:
        return None
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"observation.environment: is a mapping; got {type(spec).__name__}."
        )
    check_unknown_keys("observation.environment", dict(spec), _ENVIRONMENT_KEYS,
                       label="environment:")
    temperature = humidity = None
    if "temperature" in spec:
        temperature = jnp.asarray(
            _dimensioned("observation.environment.temperature",
                         spec["temperature"], context,
                         dimension="temperature", what="a temperature").value,
            dtype=context.dtype)
    if "humidity" in spec:
        # The source states no unit for Environment.humidity, so the config
        # must declare one (schema §4.1.1): relative humidity is
        # 'dimensionless' in this alphabet.
        humidity = jnp.asarray(
            _dimensioned("observation.environment.humidity", spec["humidity"],
                         context, dimension="dimensionless",
                         what="relative humidity (declare unit: "
                              "dimensionless)").value,
            dtype=context.dtype)
    extra: dict[str, Any] = {}
    for key, node in (spec.get("extra") or {}).items():
        extra[key] = jnp.asarray(resolve_value(node, context).value,
                                 dtype=context.dtype)
    return Environment(temperature=temperature, humidity=humidity, extra=extra)


def _extra(spec: Any, context: ResolutionContext) -> dict[str, Any]:
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"observation.extra: is a mapping; got {type(spec).__name__}."
        )
    out: dict[str, Any] = {}
    for key, node in spec.items():
        if key == "receiver_input":
            raise ConfigError(
                "observation.extra.receiver_input: this key is produced by "
                "observation.switching -- declare the cycle there, so the "
                "switch indices, the cal_loads order and the gamma_src rows "
                "cannot disagree."
            )
        out[key] = jnp.asarray(resolve_value(node, context).value)
    return out


def _aux(spec: Any, context: ResolutionContext, *, n_time: int,
         n_freq: int) -> dict[str, Any]:
    if spec is None:
        return {}
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"observation.aux: is a mapping; got {type(spec).__name__}."
        )
    check_unknown_keys("observation.aux", dict(spec), _AUX_KEYS, label="aux:")
    out: dict[str, Any] = {}
    if "flags" in spec:
        flags = jnp.asarray(resolve_value(spec["flags"], context).value)
        if flags.dtype != jnp.bool_:
            # The array forms cast to the run dtype (`_finish`), so a document
            # writes 0/1; exact 0/1 casts to bool, anything else is refused.
            if not bool(jnp.all((flags == 0) | (flags == 1))):
                raise ConfigError(
                    "observation.aux.flags: is boolean, TRUE = BAD (the "
                    "FlaggingOperator convention); a numeric array is accepted "
                    "only when every entry is exactly 0 or 1."
                )
            flags = flags.astype(jnp.bool_)
        if flags.shape != (n_time, n_freq):
            raise ConfigError(
                f"observation.aux.flags: is (n_time, n_freq) = "
                f"({n_time}, {n_freq}); got {tuple(flags.shape)}."
            )
        out["flags"] = flags
    return out


def _data(node: Any, context: ResolutionContext, *, n_time: int, n_freq: int):
    if node is None:
        return None
    data = jnp.asarray(resolve_value(node, context).value)
    if data.shape != (n_time, n_freq):
        raise ConfigError(
            f"observation.data: is (n_time, n_freq) = ({n_time}, {n_freq}); "
            f"got {tuple(data.shape)}."
        )
    return data


def build_observation(section: Any, *, runtime, base_dir: str | None = None):
    """The full ``observation:`` section -> ``(ObservationBuild, context)``.

    The returned context carries the run's grids, dtype, seeds and switch
    order -- everything ``build_resources`` and ``build_model`` resolve
    against (resources are added by ``document.py`` after it builds them).
    """
    from rheplicant.config.sections.ingest import parse_from_file
    from rheplicant.config.sections.pointing import compile_pointing
    from rheplicant.config.sections.switching import (
        SwitchingBuild,
        compile_switching,
        declared_order,
    )

    if not isinstance(section, Mapping):
        raise ConfigError(
            f"observation: is a mapping; got {type(section).__name__}."
        )
    check_unknown_keys("observation", dict(section), _OBSERVATION_KEYS,
                       label="the observation section")
    bootstrap = ResolutionContext(
        dtype=runtime.dtype, base_dir=base_dir, seed=runtime.seed,
        seeds=dict(runtime.seeds))

    ingest = None
    meta = _meta(section.get("meta"))
    if "from_file" in section:
        for key, why in (
            ("freq", "the recording carries the frequency axis"),
            ("time", "the recording carries the time axis"),
            ("data", "the recording IS the data"),
            ("aux", "the recording's settle log becomes aux['flags']"),
        ):
            if key in section:
                raise ConfigError(
                    f"observation: from_file and {key}: together say two "
                    f"things about one recording -- {why}."
                )
        ingest, record = parse_from_file(section["from_file"], bootstrap)
        meta = {**meta, **record}
        import numpy as np

        freq_hz = jnp.asarray(ingest.freq_hz)
        time_s = jnp.asarray(np.asarray(ingest.time_s)
                             - float(ingest.time_s[0]))
        epoch = float(ingest.time_s[0])
        integration = width = None
        switching_spec = section.get("switching")
        if switching_spec is not None:
            extra_keys = sorted(set(switching_spec) - {"order"})
            if extra_keys:
                raise ConfigError(
                    f"observation.switching: an ingested run declares order: "
                    f"only -- the recording carries the cycle; got "
                    f"{extra_keys} too."
                )
            switching = SwitchingBuild(
                order=declared_order(switching_spec), receiver_input=None)
        else:
            switching = SwitchingBuild(order=(), receiver_input=None)
        data = None
        aux: dict[str, Any] = {}
    else:
        freq_hz = _freq_grid(section.get("freq", {}), bootstrap)
        time_s, epoch, integration, width = _time_facts(
            section.get("time", {}), bootstrap)

    grid_context = ResolutionContext(
        freq=freq_hz, time=time_s, dtype=runtime.dtype, base_dir=base_dir,
        seed=runtime.seed, seeds=dict(runtime.seeds))
    n_time, n_freq = int(time_s.shape[0]), int(freq_hz.shape[0])

    site = _site(section.get("site"), grid_context)
    env = _environment(section.get("environment"), grid_context)
    extra = _extra(section.get("extra"), grid_context)
    if "from_file" not in section:
        aux = _aux(section.get("aux"), grid_context, n_time=n_time,
                   n_freq=n_freq)
        data = _data(section.get("data"), grid_context, n_time=n_time,
                     n_freq=n_freq)
        switching = compile_switching(section.get("switching"), grid_context,
                                      n_time=n_time)

    pointing = compile_pointing(section.get("pointing"), grid_context,
                                time_s=time_s, epoch_unix_s=epoch, site=site)
    # Baked-pointing provenance lands in meta AFTER _meta's own hashability
    # gate has already run (compile_pointing's `provenance:` values are
    # arbitrary document data, not swept by `_meta`) -- so the same check
    # applies again here, at the merge, keyed by the `pointing/<key>` name
    # `compile_pointing` already prefixed them with.
    meta = {**meta, **{key: _hashable(key, value)
                       for key, value in pointing.provenance.items()}}
    if epoch is not None:
        meta.setdefault("time_epoch_unix_s", epoch)

    for key in pointing.extra:
        if key in extra:
            raise ConfigError(
                f"observation: {key!r} is written both by pointing "
                f"(materialise/lst) and by observation.extra -- one producer "
                "per key, or the two silently disagree."
            )
    merged_extra = {**extra, **pointing.extra}
    if switching.receiver_input is not None:
        merged_extra["receiver_input"] = switching.receiver_input

    build = ObservationBuild(
        time_s=time_s, freq_hz=freq_hz, epoch_unix_s=epoch,
        integration_time_s=integration, channel_width_hz=width, site=site,
        env=env, meta=meta, aux=aux, data=data, pointing=pointing.pointing,
        extra=merged_extra, switch_order=switching.order, ingest=ingest)
    context = ResolutionContext(
        freq=freq_hz, time=time_s, dtype=runtime.dtype, base_dir=base_dir,
        seed=runtime.seed, seeds=dict(runtime.seeds),
        switch_order=switching.order)
    return build, context
