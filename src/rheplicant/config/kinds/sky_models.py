"""resources.sky_models: what the sky IS, separately from how it is seen.

The sky and the beam meet at exactly one node -- ``model.observed_astro_sky``,
whose two fields are ``sky_model`` and ``projector`` -- and they are declared
apart because they are separately reusable and separately expensive. Giving
the sky its own top-level section would put the beam in two places, which is
the coupling ``{ref: ...}`` identity exists to prevent.

``kind: maps`` builds :class:`~rheplicant.radio.sky.model.MapSky`, and this is the one
place in the config layer that can catch the failure that class's own
docstring names: ``MapSky.__call__`` returns the stored maps and *does not
consult its freq argument* beyond the shape, so maps built for 60-85 MHz
evaluated on a 100-125 MHz grid of the same length come back finite,
plausible and wrong -- and cannot be caught under ``jit``, because the values
are traced and only the shape is static. Here both grids are in hand at once.
"""

from typing import Any

import jax.numpy as jnp
import numpy as np

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.hatch import import_target
from rheplicant.config.resources import register_kind
from rheplicant.config.values import resolve_value
from rheplicant.radio import MapSky, PowerLawSkyModel, UniformSkyModel

SKY_KINDS: tuple[str, ...] = ("uniform", "power_law", "maps", "gdsm", "python")

#: kind -> the keys its builder consumes, "kind" always included. One table
#: rather than a sweep written once per branch: before this table existed,
#: only kind: maps and kind: gdsm swept for unknown keys, and {kind: uniform,
#: ..., spectral_index: 2.5} was accepted with spectral_index silently
#: discarded, because UniformSkyModel's own branch never looked at the
#: spec's key set, only at the keys it happened to read off it.
_ALLOWED_KEYS: dict[str, frozenset[str]] = {
    "uniform": frozenset({"kind", "amplitude", "n_pix"}),
    "power_law": frozenset({"kind", "amplitude", "spectral_index", "ref_freq", "n_pix"}),
    "maps": frozenset({"kind", "maps", "freq", "nside", "order", "frame", "unit"}),
    "gdsm": frozenset({"kind", "nside"}),
    "python": frozenset({"kind", "python", "args", "literal"}),
}


def _traced(spec: dict, key: str, context: ResolutionContext, name: str):
    if key not in spec:
        raise ConfigError(f"{name}: {key!r} is required for kind={spec.get('kind')!r}.")
    return jnp.asarray(resolve_value(spec[key], context).value, dtype=context.dtype)


def _static_int(spec: dict, key: str, name: str) -> int:
    if key not in spec:
        raise ConfigError(f"{name}: {key!r} is required for kind={spec.get('kind')!r}.")
    value = spec[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{name}: {key} must be a plain integer, got {type(value).__name__} "
            f"({value!r}). It is a static field -- an array there is put in the "
            "treedef, where it corrupts the jit cache key."
        )
    return value


def _static_float(spec: dict, key: str, context: ResolutionContext, name: str) -> float:
    if key not in spec:
        raise ConfigError(f"{name}: {key!r} is required for kind={spec.get('kind')!r}.")
    resolved = resolve_value(spec[key], context)
    if resolved.source != "scalar":
        raise ConfigError(
            f"{name}: {key} is a static field and this value is a {resolved.source!r} "
            "form, which produces an array. A static array goes into the treedef and "
            "corrupts the jit cache key -- and on ref_freq specifically it does not "
            "even raise: the operator's `not self.ref_freq > 0` check passes, because "
            "`not Array(True)` is False."
        )
    return float(resolved.value)


def _check_unknown_keys(name: str, kind: str, spec: dict) -> None:
    """Refuse any key ``kind``'s builder does not consume, before dispatch.

    Hoisted out of each branch so every kind gets the same sweep, rather than
    each branch remembering to write its own.
    """
    unknown = sorted(set(spec) - _ALLOWED_KEYS[kind])
    if not unknown:
        return
    if kind == "gdsm":
        raise ConfigError(
            f"{name}: kind: gdsm does not take {unknown}; it takes nside only. The "
            "sky IS the GSM16 model evaluated on the run's own frequency grid -- "
            "there is no amplitude to scale and no grid to declare, because both "
            "are decided by the model and the run."
        )
    raise ConfigError(f"{name}: kind: {kind} does not take {unknown}.")


@register_kind("sky_models")
def build_sky_model(name: str, spec: dict, context: ResolutionContext) -> Any:
    """Build one sky model, discriminated on ``kind``."""
    kind = spec.get("kind")
    if kind is None:
        raise ConfigError(
            f"{name}: 'kind' is required; the kinds are {list(SKY_KINDS)}. What the sky "
            "IS is a modelling choice with no default -- a uniform 200 K sky and a "
            "power law through the same band differ by hundreds of kelvin and both "
            "produce a plausible waterfall."
        )
    if kind not in SKY_KINDS:
        raise ConfigError(
            f"{name}: unknown sky kind {kind!r}; this layer builds {list(SKY_KINDS)}. "
            "A sky the package does not ship goes through kind: python, which names "
            "an AbstractSkyModel subclass and states its own cost."
        )
    _check_unknown_keys(name, kind, spec)
    if kind == "uniform":
        return UniformSkyModel(
            amplitude=_traced(spec, "amplitude", context, name),
            n_pix=_static_int(spec, "n_pix", name),
        )
    if kind == "power_law":
        return PowerLawSkyModel(
            amplitude=_traced(spec, "amplitude", context, name),
            spectral_index=_traced(spec, "spectral_index", context, name),
            ref_freq=_static_float(spec, "ref_freq", context, name),
            n_pix=_static_int(spec, "n_pix", name),
        )
    if kind == "gdsm":
        return _build_gdsm(name, spec, context)
    if kind == "python":
        target = spec.get("python")
        if target is None:
            raise ConfigError(f"{name}: kind: python requires a 'python:' target.")
        args = spec.get("args") or {}
        literal = spec.get("literal") or {}
        # Refuse rather than let literal silently win on overlap, mirroring
        # hatch.py's own python: form (the value-grammar form this kind's
        # branch otherwise duplicates by hand): which one "won" would decide
        # whether a value node was resolved through the grammar or forwarded
        # untouched, and that is not a choice a document should make by
        # writing the same key twice.
        clash = sorted(set(args) & set(literal))
        if clash:
            raise ConfigError(
                f"{name}: {clash} appear in both args and literal. args values are "
                "resolved through the value grammar and literal values are forwarded "
                "untouched, so one argument cannot be both."
            )
        factory = import_target(target)
        arguments = {key: resolve_value(value, context).value for key, value in args.items()}
        arguments.update(literal)
        return factory(**arguments)
    return _build_maps(name, spec, context)


def _build_maps(name: str, spec: dict, context: ResolutionContext) -> MapSky:
    nside = _static_int(spec, "nside", name)
    order = spec.get("order", "ring")
    if order != "ring":
        raise ConfigError(
            f"{name}: order={order!r}. This package reads HEALPix RING throughout -- "
            "cst_beam_maps returns RING, DriftScanProjector documents its nside as "
            "RING -- and there is no reordering step anywhere in src/. A NESTED map "
            "would be read as RING: same shape, same statistics, every pixel in the "
            "wrong place. Reorder it before declaring it."
        )
    maps = jnp.asarray(resolve_value(spec.get("maps", {}), context).value, dtype=context.dtype)
    freq = jnp.asarray(resolve_value(spec.get("freq", {}), context).value)
    expected_pix = 12 * nside * nside
    if maps.ndim != 2:
        raise ConfigError(
            f"{name}: maps must be (n_freq, n_pix) and this is {tuple(maps.shape)}. A "
            "single map is (1, n_pix), not (n_pix,) -- the frequency axis is not "
            "optional, because MapSky's whole contract is which grid the maps belong to."
        )
    if maps.shape[1] != expected_pix:
        raise ConfigError(
            f"{name}: nside={nside} implies 12*nside**2 = {expected_pix} pixels and "
            f"these maps have {maps.shape[1]}. Nothing downstream re-derives nside from "
            "the map length except DriftScanProjector.from_beam_maps, so a mismatch "
            "here becomes a projector built for a different resolution than the sky."
        )
    if context.freq is not None and maps.shape[0] != int(context.freq.shape[0]):
        raise ConfigError(
            f"{name}: the maps have {maps.shape[0]} channels and this run's frequency "
            f"grid has {int(context.freq.shape[0])}. The maps' first axis IS the "
            "frequency axis."
        )
    freq_len = int(freq.shape[0]) if freq.ndim >= 1 else 0
    if freq_len != maps.shape[0]:
        raise ConfigError(
            f"{name}: freq declares {freq_len} channel(s) and maps has "
            f"{maps.shape[0]} in its first axis -- the maps' first axis IS the "
            "frequency axis the maps were built on. MapSky.__check_init__ would "
            "catch the same mismatch, but only once the object exists and as a "
            "StateValidationError; refusing it here, ahead of the band-mismatch "
            "check below, keeps every 'declared against the wrong grid' failure in "
            "one place with one message."
        )
    if context.freq is not None and freq.shape == context.freq.shape:
        if not bool(jnp.allclose(freq, context.freq)):
            raise ConfigError(
                f"{name}: the maps declare the grid "
                f"[{float(freq.min()):.6g}, {float(freq.max()):.6g}] Hz while this run "
                f"observes on [{float(context.freq.min()):.6g}, "
                f"{float(context.freq.max()):.6g}] Hz -- the same NUMBER of channels, "
                "different channels. MapSky returns the stored maps and does not "
                "consult its freq argument beyond the shape: the maps are not "
                "interpolated, so this run would use maps built for another band and "
                "return a smooth, plausible, wrong temperature. Nothing later can see "
                "it, because under jit the values are traced and only the shape is "
                "static. Rebuild the maps on the grid you mean to observe on."
            )
    return MapSky(maps=maps, freq=freq)


def _build_gdsm(name: str, spec: dict, context: ResolutionContext) -> MapSky:
    """GSM16 through ``limTOD.sky_model.GDSM_sky_model``, on the run's own grid.

    The seam takes ONE frequency per call, in MHz (``GDSM_sky_model(*, freq:
    float, nside: int)``, measured against limTOD 1.10.0), so this builder
    loops over the run's grid and stacks. Because the maps are generated on
    that grid rather than declared against one, the band-mismatch refusal
    ``kind: maps`` needs has nothing to catch here.

    The unit is Kelvin: limTOD calls ``GlobalSkyModel16()`` with its defaults,
    and pygdsm's default ``data_unit`` is ``'TCMB'`` with ``include_cmb=False``
    (measured, pygdsm 1.7.1) -- thermodynamic temperature without the CMB
    monopole. pygdsm's other units differ by orders of magnitude, which is why
    the suite pins the 70 MHz sky mean rather than only the shape.
    """
    nside = _static_int(spec, "nside", name)
    if context.freq is None:
        raise ConfigError(
            f"{name}: kind: gdsm needs observation.freq.grid -- the model is "
            "evaluated on the run's own channels, so a run without a frequency "
            "grid has nowhere to evaluate it."
        )
    try:
        import pygdsm  # noqa: F401  -- GDSM_sky_model imports it lazily; surface it eagerly
        from limTOD.sky_model import GDSM_sky_model
    except ImportError as exc:
        raise ConfigError(
            f"{name}: kind: gdsm needs pygdsm, the Global Sky Model package. It is "
            "optional -- it pulls extra packages and downloads sky-model data on "
            "first use -- and it arrives through limTOD's own extra: "
            'pip install "limTOD[gdsm]". Checked here rather than at first use, '
            "because the alternative is a run that fails after everything else "
            "was built."
        ) from exc
    rows = [GDSM_sky_model(freq=float(f) / 1e6, nside=nside) for f in np.asarray(context.freq)]
    # dtype=context.dtype: redundant under JAX's default x32 mode, which already
    # downcasts pygdsm's float64 rows to float32 on `jnp.asarray` regardless of
    # what dtype is requested here -- but it is the one thing enforcing the
    # run's own dtype once x64 is enabled, or if pygdsm's own return dtype ever
    # changes, so it stays rather than being deleted as apparently-dead code.
    return MapSky(maps=jnp.asarray(np.stack(rows), dtype=context.dtype),
                  freq=jnp.asarray(context.freq))
