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

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import record_resolved_delivery
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
    "driftscan": frozenset(
        {
            "engine",
            "beam",
            # `beam_alms` and `nside` are the alms route: a driftscan entry may
            # take a pre-computed analysis instead of running one. `nside` is
            # inferred from the map length on the `beam:` route and cannot be
            # inferred from alms, so it is written there and only there -- see
            # the refusal in `build_projector`.
            "beam_alms",
            "nside",
            "lmax",
            "lat_deg",
            "az_deg",
            "el_deg",
            "normalize_beam",
            "acknowledge_float32_sky",
            "optimizations",
            "lst_ref_deg",
            "beam_iterations",
            "selfrot_deg",
            "horizon_mask",
            "apod_deg",
            "mask_iterations",
            "uniform_sampling",
            "freq_chunk",
        }
    ),
    "general_pointing": frozenset(
        {
            "engine",
            "beam",
            "beam_alms",
            "lmax",
            "nside",
            "lat_deg",
            "normalize_beam",
            "acknowledge_float32_sky",
            "beam_iterations",
        }
    ),
    "matrix": frozenset({"engine", "matrix", "provenance"}),
}


def _a12_normalize_beam(name: str, spec: Mapping[str, Any]) -> str | None:
    """Check A12, the projector half: ``normalize_beam`` is written.

    A presence test and nothing more -- ``bool(spec["normalize_beam"])`` takes
    any value, and what has no default is the DECISION, not a particular
    spelling of it.

    Extracted so :mod:`rheplicant.config.preflight.resources` can ask it of
    the document's text before ``build_resources`` reads the beam this
    projector references (§2.2's one-binding rule: the sentence lives here,
    the pass calls this and wraps the answer in a ``Finding``).  The saving is
    real for this row in particular: measured, a projector's missing
    ``normalize_beam`` is noticed only after its beam has been read,
    normalised and truncated.

    Returns:
        The refusal, or ``None`` when the entry is fine.
    """
    if "normalize_beam" in spec:
        return None
    return (
        f"{name}: normalize_beam is required and has no default. false returns "
        "the integral of B.T over the sphere, which is not a temperature: 32838 K "
        "against 200 K on a uniform 200 K sky. The output's unit is decided by "
        "this key together with the beam's own normalize:, so neither can be "
        "inferred from the other."
    )


def _a44_float32_sky(name: str, spec: Mapping[str, Any], dtype: str) -> str | None:
    """Check A44: a real sky engine in float32, unacknowledged.

    **The dtype is an ARGUMENT, not a key of ``spec``.**  A44 reads like a
    conditional key-presence test and is not one: its condition is
    ``runtime.jax_enable_x64``, which lives in another section entirely, and
    an absent ``jax_enable_x64`` means float32 -- so A44 fires **by default**
    on any driftscan or general_pointing entry that does not say
    ``acknowledge_float32_sky: true``.  Taking the dtype in is what lets the
    builder pass ``context.dtype`` and the pre-flight pass pass
    ``RuntimeFacts(...).dtype`` off the document's own ``runtime:`` block,
    without either of them re-deriving the other's.

    Callers must apply the ``engine: matrix`` exemption themselves: this
    function is called from ``build_projector`` AFTER the matrix branch has
    already returned, and a ``MatrixProjector`` reads no beam and does no
    map/alm work, so the O(10%) error the gate is about does not apply to it.

    Returns:
        The refusal, or ``None`` when the entry is fine.
    """
    if dtype == "float64" or spec.get("acknowledge_float32_sky", False):
        return None
    return (
        f"{name}: engine {spec.get('engine')!r} is a real sky engine and this run is in "
        f"{dtype}. radio/sky/general_pointing.py:28-32 states that the "
        "map/alm steps carry O(10%) errors in float32 -- larger than every effect "
        "normalize_beam, phi0_deg and phi_sense are required keys for, and "
        "invisible: the maps come back finite, correctly shaped and plausibly "
        "structured. Set runtime.jax_enable_x64: true, or write "
        "acknowledge_float32_sky: true on this entry to say the error is "
        "understood. x64 is process-global and part of the hashed config."
    )


def _a48_lst_ref(name: str, spec: Mapping[str, Any]) -> str | None:
    """Check A48: ``cache_beam_rotation`` without the ``lst_ref_deg`` it needs.

    Reads ``optimizations`` exactly as :func:`build_projector` does, so that
    the two cannot answer differently on a spelling neither of them models.

    **Its two neighbours run first and this function does not know that.**
    In ``build_projector`` the ``read_horizon_fraction`` redirect and the
    unknown-optimisation refusal are both ahead of this call, so a document
    writing ``optimizations: [cache_beam_rotation, bogus]`` hears about
    ``bogus``.  A caller at another phase that wants the same ordering has to
    arrange it; ``preflight/resources.py`` does, and says so.

    **On ``engine: general_pointing`` this advice is refused two gates later,
    and that loop is left standing.**  Measured: a general_pointing entry
    writing ``optimizations: [cache_beam_rotation]`` earns this sentence, and
    a reader who follows it by adding ``lst_ref_deg`` then earns
    ``engine: general_pointing does not take ['lst_ref_deg',
    'optimizations']`` from :data:`_ENGINE_KEYS`.  That engine takes neither
    key, so there is no edit this sentence could name that would work there.
    The pre-flight row (``preflight/resources.py::_projector_keys``) therefore
    gates on ``driftscan`` and stays silent; closing the loop HERE would
    change a shipped message, which is a decision this task was not given.

    Returns:
        The refusal, or ``None`` when the entry is fine.
    """
    if "cache_beam_rotation" not in list(spec.get("optimizations") or []):
        return None
    if "lst_ref_deg" in spec:
        return None
    return (
        f"{name}: optimizations contains 'cache_beam_rotation', which requires "
        "lst_ref_deg. to_reference_frame() raises without one -- after the beam "
        "file has been read and analysed, which is the whole class of failure "
        "these checks run before. There is no silent default: defaulting it to "
        "lst0_deg would re-anchor the m-mode phases."
    )


def _require(name: str, spec: dict, key: str, engine: str, what: str) -> None:
    if key not in spec:
        raise ConfigError(f"{name}: engine: {engine} requires {key}: -- {what}.")


def _beam_ref(name: str, spec: dict, engine: str) -> dict:
    _require(name, spec, "beam", engine, "a {ref: resources.beams.<name>} mapping")
    beam = spec["beam"]
    if not isinstance(beam, dict) or set(beam) != {"ref"}:
        raise ConfigError(
            f"{name}: beam: is {{ref: resources.beams.<name>}} -- the beam is a "
            f"declared resource, not an inline value; got {beam!r}."
        )
    return beam


def _angle(spec: dict, key: str, context: ResolutionContext, name: str, engine: str) -> float:
    if key not in spec:
        raise ConfigError(f"{name}: {key!r} is required for this engine.")
    destination = DestinationDescriptor(
        f"{name}.{key}",
        "resource_field",
        f"rheplicant.config.kinds.projectors.build_projector.{engine}.{key}",
    )
    resolved = resolve_value(
        spec[key],
        context,
        destination=destination,
    )
    if resolved.unit is not None and resolved.unit.canonical != "deg":
        raise ConfigError(f"{name}: {key} must be an angle, got {resolved.unit.canonical!r}.")
    value = float(resolved.value)
    record_resolved_delivery(context, destination, resolved.unit)
    return value


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
        _require(
            name, spec, "matrix", "matrix", "the (n_data, n_pix) projection matrix as a value node"
        )
        matrix_destination = DestinationDescriptor(
            f"{name}.matrix",
            "resource_field",
            "rheplicant.config.kinds.projectors.build_projector.matrix.matrix",
        )
        resolved = resolve_value(spec["matrix"], context, destination=matrix_destination)
        matrix = jnp.asarray(resolved.value, dtype=context.dtype)
        record_resolved_delivery(context, matrix_destination, resolved.unit)
        return MatrixProjector(matrix=matrix)

    # A12's projector half and A44, in the order they were written in. Both
    # are the pre-flight pass's rows now (preflight/resources.py) and both are
    # still asked here, as the section's own second opinion (§2.2) -- the
    # messages live in the three functions at the head of this module and
    # nowhere else.
    problem = _a12_normalize_beam(name, spec)
    if problem is not None:
        raise ConfigError(problem)
    if "acknowledge_float32_sky" not in spec:
        context.use_default("resources.projectors[].acknowledge_float32_sky", False)
    problem = _a44_float32_sky(name, spec, context.dtype)
    if problem is not None:
        raise ConfigError(problem)
    optimizations = list(
        (
            spec.get("optimizations")
            if "optimizations" in spec
            else context.use_default("resources.projectors[].optimizations", [])
        )
        or []
    )
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
    # A48, after its two neighbours above, deliberately: a document writing
    # both an unknown optimisation and cache_beam_rotation hears about the
    # unknown one, which is the fault it can act on first.
    problem = _a48_lst_ref(name, spec)
    if problem is not None:
        raise ConfigError(problem)

    # `beam:` beside `beam_alms:` is PRECEDENCE, not a conflict, and this was
    # nearly "fixed" into a refusal on the grounds that two sources with one
    # silently discarded is the shape this package refuses elsewhere. It is
    # not that shape: check B9's own advice tells a user to add
    # `beam_alms: {ref: ...}` to an entry that already carries `beam:`, and
    # `tests/config/test_inflight_optics.py` drives exactly that document to
    # show the remedy earns its keep. A refusal here would refuse the layer's
    # own recommendation two gates later -- the R4 advice loop this file
    # exists to avoid.
    if engine == "general_pointing":
        check_unknown_keys(
            name, spec, _ENGINE_KEYS["general_pointing"], label="engine: general_pointing"
        )
        _require(name, spec, "lmax", "general_pointing", "the spherical-harmonic band limit")
        _require(
            name, spec, "nside", "general_pointing", "the HEALPix resolution of the sampling grid"
        )
        beam_alms = None
        if "beam_alms" in spec:
            alms_destination = DestinationDescriptor(
                f"{name}.beam_alms",
                "resource_field",
                "rheplicant.config.kinds.projectors.build_projector.general_pointing.beam_alms",
            )
            resolved = resolve_value(
                spec["beam_alms"], context, destination=alms_destination
            )
            beam_alms = jnp.asarray(resolved.value)
            record_resolved_delivery(context, alms_destination, resolved.unit)
        if beam_alms is None:
            beam = resolve_reference(_beam_ref(name, spec, "general_pointing")["ref"], context)
            iterations = (
                spec["beam_iterations"]
                if "beam_iterations" in spec
                else context.use_default("resources.projectors[].beam_iterations", 3)
            )
            beam_alms = _analyse(name, beam.maps, int(spec["lmax"]), int(iterations))
        return GeneralPointingProjector(
            beam_alms=beam_alms,
            lat_deg=_angle(spec, "lat_deg", context, name, engine),
            lmax=int(spec["lmax"]),
            nside=int(spec["nside"]),
            normalize_beam=bool(spec["normalize_beam"]),
        )

    # Each engine resolves its own beam_alms rather than sharing one prelude,
    # and the duplication is what the destination census requires: it reads
    # this source and resolves `destination=<name>` to the LAST
    # DestinationDescriptor assigned to that name, so one shared call site can
    # only ever register one destination. Two concrete destinations exist, so
    # there are two literal call sites. Measured -- an f-string here, and then
    # a conditional expression, each registered one unresolvable row.
    beam_alms = None
    if "beam_alms" in spec:
        alms_destination = DestinationDescriptor(
            f"{name}.beam_alms",
            "resource_field",
            "rheplicant.config.kinds.projectors.build_projector.driftscan.beam_alms",
        )
        resolved = resolve_value(spec["beam_alms"], context, destination=alms_destination)
        beam_alms = jnp.asarray(resolved.value)
        record_resolved_delivery(context, alms_destination, resolved.unit)

    if beam_alms is None and "nside" in spec:
        raise ConfigError(
            f"{name}: nside is not written for engine: driftscan with beam:. "
            "from_beam_maps() infers it from the map length -- nside is inferred, "
            "not declared -- and passes it to the constructor itself, so a config "
            "that also passed it raises 'got multiple values for keyword argument "
            "nside'. The beam's own nside: is where the resolution is declared. "
            "(With beam_alms: it is the other way round: alms carry no pixel "
            "count, so nside must be written.)"
        )
    check_unknown_keys(name, spec, _ENGINE_KEYS["driftscan"], label="engine: driftscan")
    _require(name, spec, "lmax", "driftscan", "the spherical-harmonic band limit")
    forwarded = {
        key: spec[key]
        for key in (
            "selfrot_deg",
            "horizon_mask",
            "apod_deg",
            "mask_iterations",
            "lst_ref_deg",
            "uniform_sampling",
            "freq_chunk",
        )
        if key in spec
    }
    for key in ("selfrot_deg", "apod_deg", "lst_ref_deg"):
        if key in forwarded:
            forwarded[key] = _angle(spec, key, context, name, engine)
    pointing = {
        "lat_deg": _angle(spec, "lat_deg", context, name, engine),
        "az_deg": _angle(spec, "az_deg", context, name, engine),
        "el_deg": _angle(spec, "el_deg", context, name, engine),
    }
    if beam_alms is not None:
        # The alms route. `from_beam_maps` does exactly two things this
        # constructor call does not: it infers `nside` from the map length and
        # it runs `map2alm_iter`. Everything else it accepts -- selfrot_deg,
        # horizon_mask, apod_deg, mask_iterations, uniform_sampling,
        # freq_chunk, lst_ref_deg -- it forwards to the constructor untouched,
        # so an entry built from alms keeps every one of them. Measured by
        # reading it: the classmethod's body is a shape check, an nside
        # derivation, one vmapped transform, and `cls(...)`.
        if "beam_iterations" in spec:
            raise ConfigError(
                f"{name}: beam_iterations is written beside beam_alms, and there "
                "is no analysis for it to iterate -- it is the healpy `iter` "
                "equivalent of the map-to-alm transform this route skips. The "
                "alms were analysed wherever they came from, at whatever "
                "iteration count was used there. Delete it, or switch to beam: "
                "and let the transform run here."
            )
        _require(
            name, spec, "nside", "driftscan", "the HEALPix resolution the alms are used at"
        )
        projector = DriftScanProjector(
            beam_alms=beam_alms,
            lmax=int(spec["lmax"]),
            nside=int(spec["nside"]),
            normalize_beam=bool(spec["normalize_beam"]),
            **pointing,
            **forwarded,
        )
    else:
        beam = resolve_reference(_beam_ref(name, spec, "driftscan")["ref"], context)
        projector = DriftScanProjector.from_beam_maps(
            beam.maps,
            lmax=int(spec["lmax"]),
            iterations=int(
                spec["beam_iterations"]
                if "beam_iterations" in spec
                else context.use_default("resources.projectors[].beam_iterations", 3)
            ),
            normalize_beam=bool(spec["normalize_beam"]),
            **pointing,
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
        np.stack(
            [
                healpy.map2alm(np.asarray(row), lmax=lmax, iter=iterations)
                for row in np.asarray(maps)
            ]
        )
    )


@register_derivation("horizon_fraction", frozenset({"projector"}))
def _horizon_fraction(node, context, modifiers, target) -> ResolvedValue:
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
