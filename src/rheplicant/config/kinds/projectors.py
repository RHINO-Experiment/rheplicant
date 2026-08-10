"""resources.projectors: how the sky is seen, with the beam inside it.

Field names are the Python names verbatim, which is the v0 contradiction
removed -- v0 stripped the unit suffixes here while the path grammar kept
them, so one field had two spellings.

Four rules this module exists to enforce, each with a measurement behind it.

``normalize_beam`` has no default and must be written: ``false`` returns
``integral(B . T dOmega)``, which is not a temperature -- 32838 K against
200 K on a uniform 200 K sky.

``beam_frame`` and ``beam_ref_lst_deg`` are **not writable**. They are set only
by ``to_reference_frame()``, and ``__check_init__`` exists to catch a hand-set
pair; a YAML that wrote them would drive the object into exactly the state
that guard is for. The config route is the ordered ``optimizations:`` list.

``beam_iterations`` is ``from_beam_maps(iterations=)`` -- the ``map2alm_iter``
count of the **beam** analysis, which directly sets ``beam_alms``, the
projector's only traced array. It is a constructor argument and **not a stored
field** (verified: ``hasattr(projector, 'iterations')`` is ``False``), so it is
unrecoverable from the built object and the config key is the only place it can
ever be captured. Do not confuse it with the field ``mask_iterations``, which
also defaults to 3 and is a different quantity.

``nside`` may not be written on the ``from_beam_maps`` path: the classmethod
infers it from the map length and passes it itself, so a config that also
passed it raises ``TypeError: got multiple values for keyword argument
'nside'``.
"""

from typing import Any

import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.derive import register_derivation
from rheplicant.config.errors import ConfigError
from rheplicant.config.kinds.beams import _require_healpy
from rheplicant.config.refs import resolve_reference
from rheplicant.config.resources import check_unknown_keys, register_kind
from rheplicant.config.units import canonical_unit
from rheplicant.config.values import ResolvedValue, resolve_value
from rheplicant.radio import DriftScanProjector, GeneralPointingProjector, MatrixProjector

ENGINES: tuple[str, ...] = ("driftscan", "general_pointing", "matrix")
OPTIMIZATIONS: tuple[str, ...] = ("cache_beam_rotation",)
_NOT_WRITABLE: tuple[str, ...] = ("beam_frame", "beam_ref_lst_deg")

#: engine -> every key that engine's branch of build_projector reads,
#: derived from its constructor call (or, for driftscan, the `forwarded`
#: table below it). Checked by :func:`check_unknown_keys`, AFTER every
#: richer, engine-specific refusal already in this module -- the
#: engine-validity check, the _NOT_WRITABLE gate, matrix's provenance
#: requirement, normalize_beam's own message, and driftscan's own nside
#: refusal (which names the classmethod and says "inferred") all run first,
#: so this sweep only ever reports the plain typo none of those messages is
#: about. Running it earlier would shadow the nside message with the
#: generic "does not take" for exactly the key that most needs the specific
#: one -- the same ordering kinds/beams.py established for its own
#: format-specific gates.
_ENGINE_KEYS: dict[str, frozenset[str]] = {
    "driftscan": frozenset({
        "engine", "beam", "lmax", "lat_deg", "az_deg", "el_deg", "normalize_beam",
        "acknowledge_float32_sky", "optimizations", "lst_ref_deg", "beam_iterations",
        "selfrot_deg", "horizon_mask", "apod_deg", "mask_iterations", "uniform_sampling",
        "freq_chunk",
    }),
    "general_pointing": frozenset({
        "engine", "beam", "beam_alms", "lmax", "nside", "lat_deg", "normalize_beam",
        "acknowledge_float32_sky", "beam_iterations",
    }),
    "matrix": frozenset({"engine", "matrix", "provenance"}),
}


def _angle(spec: dict, key: str, context: ResolutionContext, name: str) -> float:
    if key not in spec:
        raise ConfigError(f"{name}: {key!r} is required for this engine.")
    resolved = resolve_value(spec[key], context)
    if resolved.unit is not None and resolved.unit.canonical != "deg":
        raise ConfigError(f"{name}: {key} must be an angle, got {resolved.unit.canonical!r}.")
    return float(resolved.value)


@register_kind("projectors")
def build_projector(name: str, spec: dict, context: ResolutionContext) -> Any:
    """Build one projector, discriminated on ``engine``."""
    engine = spec.get("engine")
    if engine not in ENGINES:
        raise ConfigError(
            f"{name}: engine={engine!r}; the engines are {list(ENGINES)}. The engine is "
            "written rather than inferred from pointing.mode: MatrixProjector has no "
            "geometry to infer from, and one run can legitimately hold several "
            "projectors of different engines over one pointing -- which is what a "
            "cross-engine comparison is."
        )
    offending = sorted(set(_NOT_WRITABLE) & set(spec))
    if offending:
        raise ConfigError(
            f"{name}: {offending} are not writable keys. They are set only by "
            "to_reference_frame(), and DriftScanProjector.__check_init__ exists "
            "specifically to catch a hand-set pair -- writing them here drives the "
            "object into the state that guard is for. Ask for the optimisation "
            "instead: optimizations: [cache_beam_rotation] plus lst_ref_deg."
        )
    if engine == "matrix":
        # `not spec.get("provenance")` rather than `"provenance" not in spec`:
        # provenance: (None), provenance: {} and provenance: "" all satisfy
        # the presence check while recording nothing, which is exactly the
        # state this gate exists to rule out. Any non-empty mapping or
        # non-empty string is accepted -- this layer does not know what a
        # provenance record should contain, only that an empty one is not one.
        if not spec.get("provenance"):
            raise ConfigError(
                f"{name}: engine: matrix requires a provenance: block naming who built "
                "the matrix, at what latitude and over what LST range. The geometry is "
                "inside the matrix and nothing can recover it -- MatrixProjector has "
                "exactly one field. It is unverifiable, which is why it must at least "
                "be recorded. An empty provenance -- None, {}, or a blank string -- "
                "records nothing, so it is refused the same as an absent one."
            )
        check_unknown_keys(name, spec, _ENGINE_KEYS["matrix"], label="engine: matrix")
        return MatrixProjector(matrix=jnp.asarray(resolve_value(spec["matrix"], context).value,
                                                  dtype=context.dtype))

    if "normalize_beam" not in spec:
        raise ConfigError(
            f"{name}: normalize_beam is required and has no default. false returns "
            "the integral of B.T over the sphere, which is not a temperature: 32838 K "
            "against 200 K on a uniform 200 K sky. The output's unit is decided by "
            "this key together with the beam's own normalize:, so neither can be "
            "inferred from the other."
        )
    if context.dtype != "float64" and not spec.get("acknowledge_float32_sky", False):
        raise ConfigError(
            f"{name}: engine {engine!r} is a real sky engine and this run is in "
            f"{context.dtype}. radio/sky/general_pointing.py:28-32 states that the "
            "map/alm steps carry O(10%) errors in float32 -- larger than every effect "
            "normalize_beam, phi0_deg and phi_sense are required keys for, and "
            "invisible: the maps come back finite, correctly shaped and plausibly "
            "structured. Set runtime.jax_enable_x64: true, or write "
            "acknowledge_float32_sky: true on this entry to say the error is "
            "understood. x64 is process-global and part of the hashed config."
        )
    optimizations = list(spec.get("optimizations") or [])
    for entry in optimizations:
        if entry == "read_horizon_fraction":
            raise ConfigError(
                f"{name}: 'read_horizon_fraction' is not an optimisation -- it PRODUCES "
                "f_sky, which beam_spill consumes, and it has exactly one spelling: "
                "{from: horizon_fraction, projector: {ref: ...}}. Where the beam was "
                "truncated, prefer the beam's own sub-value "
                "{ref: resources.beams.<name>.sky_fraction}."
            )
        if entry not in OPTIMIZATIONS:
            raise ConfigError(
                f"{name}: unknown optimisation {entry!r}; they are {list(OPTIMIZATIONS)}."
            )
    if "cache_beam_rotation" in optimizations and "lst_ref_deg" not in spec:
        raise ConfigError(
            f"{name}: optimizations contains 'cache_beam_rotation', which requires "
            "lst_ref_deg. to_reference_frame() raises without one -- after the beam "
            "file has been read and analysed, which is the whole class of failure "
            "these checks run before. There is no silent default: defaulting it to "
            "lst0_deg would re-anchor the m-mode phases."
        )

    beam_alms = None
    if "beam_alms" in spec:
        beam_alms = jnp.asarray(resolve_value(spec["beam_alms"], context).value)
    if engine == "general_pointing":
        check_unknown_keys(
            name, spec, _ENGINE_KEYS["general_pointing"], label="engine: general_pointing"
        )
        if beam_alms is None:
            beam = resolve_reference(spec["beam"]["ref"], context)
            beam_alms = _analyse(
                name, beam.maps, int(spec["lmax"]), int(spec.get("beam_iterations", 3))
            )
        return GeneralPointingProjector(
            beam_alms=beam_alms,
            lat_deg=_angle(spec, "lat_deg", context, name),
            lmax=int(spec["lmax"]),
            nside=int(spec["nside"]),
            normalize_beam=bool(spec["normalize_beam"]),
        )

    if "nside" in spec:
        raise ConfigError(
            f"{name}: nside is not written for engine: driftscan. from_beam_maps() "
            "infers it from the map length -- nside is inferred, not declared -- and "
            "passes it to the constructor itself, so a config that also passed it "
            "raises 'got multiple values for keyword argument nside'. The beam's own "
            "nside: is where the resolution is declared."
        )
    check_unknown_keys(name, spec, _ENGINE_KEYS["driftscan"], label="engine: driftscan")
    beam = resolve_reference(spec["beam"]["ref"], context)
    forwarded = {
        key: spec[key]
        for key in ("selfrot_deg", "horizon_mask", "apod_deg", "mask_iterations",
                    "lst_ref_deg", "uniform_sampling", "freq_chunk")
        if key in spec
    }
    for key in ("selfrot_deg", "apod_deg", "lst_ref_deg"):
        if key in forwarded:
            forwarded[key] = _angle(spec, key, context, name)
    projector = DriftScanProjector.from_beam_maps(
        beam.maps,
        lat_deg=_angle(spec, "lat_deg", context, name),
        az_deg=_angle(spec, "az_deg", context, name),
        el_deg=_angle(spec, "el_deg", context, name),
        lmax=int(spec["lmax"]),
        iterations=int(spec.get("beam_iterations", 3)),
        normalize_beam=bool(spec["normalize_beam"]),
        **forwarded,
    )
    if "cache_beam_rotation" in optimizations:
        projector = projector.to_reference_frame()
    return projector


def _analyse(name: str, maps, lmax: int, iterations: int):
    """Beam maps to packed alms, the way DriftScanProjector.from_beam_maps does.

    Reuses :func:`rheplicant.config.kinds.beams._require_healpy` rather than a
    bare ``import healpy`` -- the general_pointing engine has no beam_alms of
    its own to analyse without it, and that failure belongs at this
    document-facing boundary rather than as a bare ImportError deep inside a
    kind builder.
    """
    import numpy as np

    healpy = _require_healpy(name)
    return jnp.asarray(
        np.stack([
            healpy.map2alm(np.asarray(row), lmax=lmax, iter=iterations)
            for row in np.asarray(maps)
        ])
    )


@register_derivation("horizon_fraction", frozenset({"projector"}))
def _horizon_fraction(node, context, modifiers) -> ResolvedValue:
    """``DriftScanProjector.horizon_fraction()`` -- the above-horizon beam fraction."""
    reference = node.get("projector")
    if not isinstance(reference, dict) or "ref" not in reference:
        raise ConfigError(
            "horizon_fraction: 'projector' must be {ref: resources.projectors.<name>}."
        )
    projector = resolve_reference(reference["ref"], context)
    if getattr(projector, "beam_frame", "local") == "reference":
        raise ConfigError(
            f"horizon_fraction: {reference['ref']!r} was built with "
            "optimizations: [cache_beam_rotation], and horizon_fraction() raises on a "
            "reference-frame projector -- the unmasked denominator is gone. Order "
            "matters and cannot be reordered here: the fraction must be read BEFORE "
            "the rotation is cached. Take it from the beam instead, "
            "{ref: resources.beams.<name>.sky_fraction}, which is the truncation's own "
            "second product."
        )
    return ResolvedValue(
        projector.horizon_fraction(), canonical_unit("dimensionless"), "from", modifiers
    )
