"""resources.beams: a raw (n_freq, n_pix) array, plus what the file cannot say.

A beam is not an object in this package -- ``cst_beam_maps`` returns a bare
``np.ndarray`` and every one of its six arguments is unrecoverable from it. So
this kind returns a small container holding the maps *and* the sky fraction,
because ``horizon_truncated_beam`` produces two products from one call and the
fraction is exactly what ``BeamSpillOperator(sky_fraction=)`` wants. v0
consumed the maps and dropped the fraction, which left the user with
``from: projector`` -- and on a truncated beam that returns approximately 1.0
and silently deletes the ``(1 - f_sky) * T_ground`` term.

Two declarations have no default and no preset may supply them.

``normalize:`` -- the output's unit is decided by the *pair* (beam
normalisation, ``normalize_beam``): 32838 K against 200 K on a uniform 200 K
sky for an unnormalised beam, and 100.42 K against a 99.79 K sky for a
unit-pixel-sum one. Neither half can be inferred from the numbers.

``phi0_deg`` / ``phi_sense`` -- required for ``format: cst`` **only**, and
refused for every other format. ``radio/beams.py`` calls them "a fact about
the as-built horn, not the file". Decision D-C3 rejected the option of letting
a preset supply them marked provisional, for a measurement-shaped reason: a
mirrored beam passes every integral, every peak and every
azimuthally-symmetric diagnostic unchanged. There is no numerical symptom, so
the only protection is that the value was stated by someone who knew it.

``format: uvbeam`` and ``format: healpix`` (D-C7, decided 2026-08-09, after
this module's first draft) both build the raw array from a file, and neither
takes ``phi0_deg``/``phi_sense``: ``uvbeam`` because the limTOD bridge carries
its own azimuth convention, and ``healpix`` because RING-versus-NESTED, not a
CST meridian, is the fact its file cannot state. ``healpix`` still needs
``frame:``, the same as any other raw array.
"""

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import record_resolved_delivery
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import resolve_file_path
from rheplicant.config.hatch import import_target
from rheplicant.config.resources import check_unknown_keys, register_kind
from rheplicant.config.values import resolve_value

BEAM_FORMATS: tuple[str, ...] = (
    "cst",
    "uvbeam",
    "healpix",
    "npy",
    "npz",
    "inline",
    "gaussian",
    "python",
)
BEAM_NORMALIZATIONS: tuple[str, ...] = ("none", "pixel_sum", "solid_angle")
RAW_ARRAY_FORMATS: tuple[str, ...] = (
    "healpix",
    "npy",
    "npz",
    "inline",
    "gaussian",
    "python",
)

_COMMON_FORMAT_KEYS: frozenset[str] = frozenset({"format", "nside", "normalize", "horizon"})
#: format -> every key that format's builder reads, common keys always
#: included. Checked by :func:`check_unknown_keys`, right after the
#: phi0_deg/phi_sense/frame gate below settles which of THOSE three keys
#: belongs where -- so this table deliberately omits phi0_deg/phi_sense from
#: every format but cst, and frame from cst and uvbeam, rather than
#: duplicating the gate's own reasoning. Its job is what the gate does not
#: cover: a plain typo. horizen: for horizon: is the second route to a
#: silently un-truncated beam -- the first is a horizon-truncation
#: implementation that does nothing, which TestHorizonTruncation pins
#: directly; this table is what stops the misspelled key from ever reaching
#: it, on every format, rather than being read as "no horizon declared" and
#: quietly doing nothing.
_FORMAT_KEYS: dict[str, frozenset[str]] = {
    "cst": _COMMON_FORMAT_KEYS | {"directory", "suffix", "phi0_deg", "phi_sense"},
    "uvbeam": _COMMON_FORMAT_KEYS | {"path"},
    "healpix": _COMMON_FORMAT_KEYS | {"path", "order", "freq", "frame"},
    "npy": _COMMON_FORMAT_KEYS | {"path", "key", "frame"},
    "npz": _COMMON_FORMAT_KEYS | {"path", "key", "frame"},
    "inline": _COMMON_FORMAT_KEYS | {"maps", "frame"},
    "gaussian": _COMMON_FORMAT_KEYS | {"fwhm_deg", "sigma_deg", "frame"},
    "python": _COMMON_FORMAT_KEYS | {"python", "args", "literal", "frame"},
}


@dataclasses.dataclass(frozen=True)
class Beam:
    """A beam resource: the maps, and the fraction of sky they see.

    Attributes:
        maps: ``(n_freq, n_pix)`` linear power, HEALPix RING.
        sky_fraction: ``(n_freq,)``. All ones where nothing was truncated --
            written rather than omitted, so ``BeamSpillOperator`` always has a
            fraction to be handed and the "which one did it use" question
            never arises.
        nside: the declared resolution, kept so a projector can check it.
        normalize: the declared convention, kept for the resolved artefact.
    """

    maps: jnp.ndarray
    sky_fraction: jnp.ndarray
    nside: int
    normalize: str


def _a12_normalize(name: str, spec: Mapping[str, Any]) -> str | None:
    """Check A12, the beam half: ``normalize:`` is written and is one of ours.

    **A membership test, not a presence one.**  ``normalize: pixelsum`` is
    present and is not a convention this package has; the pair that decides
    the output's unit is only decided when the value is one of
    :data:`BEAM_NORMALIZATIONS`, so a typo has to be refused the same as an
    omission.

    Extracted so that :mod:`rheplicant.config.preflight.resources` can ask the
    same question of the document's own text, one whole phase earlier --
    ``build_resources`` has no validation pre-pass, so on a two-beam document
    the first beam is read, cast and normalised before this one is asked about
    the second.  The sentence lives here and nowhere else (§2.2's one-binding
    rule); the pass CALLS this and wraps the answer in a ``Finding``.

    Returns:
        The refusal, or ``None`` when the entry is fine.
    """
    normalize = spec.get("normalize")
    if normalize in BEAM_NORMALIZATIONS:
        return None
    return (
        f"{name}: normalize is required and has no default; it is one of "
        f"{list(BEAM_NORMALIZATIONS)} and got {normalize!r}. The output's unit is "
        "decided by the PAIR (this key, the projector's normalize_beam): an "
        "unnormalised beam with normalize_beam: false gives 32838 K on a uniform "
        "200 K sky, and a unit-pixel-sum beam with the same flag gives 100.42 K "
        "against a 99.79 K sky. Neither half is inferable from the numbers, and no "
        "preset may supply this one."
    )


def _a11_chart_keys(name: str, spec: Mapping[str, Any]) -> str | None:
    """Check A11: ``phi0_deg``/``phi_sense``/``frame``, per format. Four legs.

    1. ``format: cst`` REQUIRES ``phi0_deg`` and ``phi_sense``.
    2. ``format: uvbeam`` REFUSES all three, ``frame`` included -- the limTOD
       bridge owns the azimuth convention and its output is beam_local by
       construction.  Leg 2 refusing ``frame`` while leg 4 requires it is the
       asymmetry this check is easiest to get backwards.
    3. every other format REFUSES ``phi0_deg``/``phi_sense``.
    4. every other format REQUIRES ``frame``.

    **Legs 3 and 4 are written out rather than driven off**
    :data:`RAW_ARRAY_FORMATS`.  That tuple has one definition and zero readers
    (measured: ``grep -rn RAW_ARRAY_FORMATS src/ tests/``); it happens to
    equal this ``else`` today, which is exactly why keying the gate on it
    would create a second source of truth that nothing would ever notice
    diverging.

    **Stands down when ``format:`` is not one of** :data:`BEAM_FORMATS`.  The
    format-value refusal above is more specific and is not this check's row,
    and an A11 that reached its ``else`` on ``format: nonsense`` would
    pre-empt it with a sentence about ``frame:``.

    Returns:
        The refusal, or ``None`` when the entry is fine.
    """
    fmt = spec.get("format")
    if fmt not in BEAM_FORMATS:
        return None
    if fmt == "cst":
        for key in ("phi0_deg", "phi_sense"):
            if key not in spec:
                return (
                    f"{name}: {key} is required for format: cst and has no default. "
                    "phi0_deg is the CST azimuth landing on the beam map's phi = 0 "
                    "meridian and phi_sense is its handedness -- 'a fact about the "
                    "as-built horn, not the file'. They cannot be defaulted and no "
                    "preset may supply them, because a MIRRORED beam passes every "
                    "integral, every peak and every azimuthally-symmetric diagnostic "
                    "unchanged: there is no numerical symptom, so the only protection "
                    "is that someone who knew the horn wrote the value down."
                )
        return None
    if fmt == "uvbeam":
        offending = sorted({"phi0_deg", "phi_sense", "frame"} & set(spec))
        if offending:
            return (
                f"{name}: {offending} are not written for format: uvbeam. The limTOD "
                "bridge carries the azimuth convention itself -- "
                "healpix_phi_to_uvbeam_az is the adapter, pinned by limTOD's own "
                "orientation suite -- and its output is beam_local by construction, "
                "so a declared frame is either redundant or a contradiction the "
                "maps cannot settle."
            )
        return None
    offending = sorted({"phi0_deg", "phi_sense"} & set(spec))
    if offending:
        return (
            f"{name}: {offending} describe how a CST export's azimuth maps onto "
            "the beam-local chart and are meaningless for format: "
            f"{fmt!r}. Requiring them everywhere is the invent-a-value habit they "
            "exist to break. For a raw array the genuinely unverifiable fact is "
            "frame:, which is required instead."
        )
    if "frame" not in spec:
        return (
            f"{name}: frame is required for format: {fmt!r} and is one of "
            "'beam_local' or 'reference'. It is declared and unverifiable -- "
            "nothing in the array says which chart it was sampled on."
        )
    return None


@register_kind("beams")
def build_beam(name: str, spec: dict, context: ResolutionContext) -> Beam:
    """Build one beam resource."""
    fmt = spec.get("format")
    if fmt not in BEAM_FORMATS:
        raise ConfigError(
            f"{name}: format={fmt!r}; the beam formats are {list(BEAM_FORMATS)}. "
            "'inline' and 'gaussian' are not conveniences -- two of the shipped "
            "examples build their beam analytically and are designed to run without "
            "the unpublished CST dataset."
        )
    # A12 and A11, in the order they were written in: the pre-flight pass
    # calls these two functions on the document's text and this call is the
    # section's own second opinion (§2.2), so an entry reached some other way
    # -- a `python:` caller, a future kind that composes beams -- is still
    # refused here rather than only where the pass happens to look.
    normalize = spec.get("normalize")
    problem = _a12_normalize(name, spec)
    if problem is not None:
        raise ConfigError(problem)
    problem = _a11_chart_keys(name, spec)
    if problem is not None:
        raise ConfigError(problem)

    # After the gate above, deliberately: that gate already gives phi0_deg,
    # phi_sense and frame their own reasoned, format-specific refusals (why a
    # mirrored CST beam has no numerical symptom; why the uvbeam bridge owns
    # its own convention), and _FORMAT_KEYS omits those three from most
    # formats on purpose. Running this sweep first would shadow those
    # messages with the generic one below for exactly the keys that most
    # need the specific one.
    check_unknown_keys(name, spec, _FORMAT_KEYS[fmt], label=f"format: {fmt}")

    nside = spec.get("nside")
    if isinstance(nside, bool) or not isinstance(nside, int) or nside < 1:
        raise ConfigError(
            f"{name}: nside is required and must be a positive integer, got {nside!r}."
        )
    maps = _maps_for(name, fmt, spec, context, nside)
    expected = 12 * nside * nside
    if maps.ndim != 2 or maps.shape[1] != expected:
        raise ConfigError(
            f"{name}: nside={nside} implies {expected} pixels and these maps are "
            f"{tuple(maps.shape)}. HEALPix RING throughout -- nothing in src/ reorders."
        )
    maps = _normalized(maps, normalize)
    fraction = jnp.ones((maps.shape[0],), dtype=maps.dtype)
    horizon = spec.get("horizon") or {}
    if not isinstance(horizon, dict):
        raise ConfigError(
            f"{name}: horizon: is a mapping (mode, el_deg, apod_deg); got "
            f"{type(horizon).__name__} ({horizon!r})."
        )
    check_unknown_keys(name, horizon, frozenset({"mode", "el_deg", "apod_deg"}),
                       label="horizon:")
    mode = horizon.get("mode", "none")
    if mode == "truncate_map":
        maps, fraction = _truncate(name, maps, horizon, context)
    elif mode not in ("none", "projector_mask"):
        raise ConfigError(
            f"{name}: horizon.mode={mode!r}; it is one of 'none', 'truncate_map', "
            "'projector_mask'."
        )
    return Beam(maps=maps, sky_fraction=fraction, nside=nside, normalize=normalize)


def _maps_for(name: str, fmt: str, spec: dict, context: ResolutionContext, nside: int):
    if fmt == "cst":
        from rheplicant.radio import cst_beam_maps

        directory = spec.get("directory")
        if directory is None:
            raise ConfigError(f"{name}: format: cst requires a directory of per-frequency exports.")
        if context.freq is None:
            raise ConfigError(f"{name}: format: cst needs observation.freq.grid to sample onto.")
        destination = DestinationDescriptor(
            f"{name}.phi0_deg",
            "resource_field",
            "rheplicant.config.kinds.beams.build_beam.cst.phi0_deg",
        )
        resolved = resolve_value(
            spec["phi0_deg"], context, destination=destination
        )
        phi0_deg = float(resolved.value)
        record_resolved_delivery(context, destination, resolved.unit)
        return jnp.asarray(
            cst_beam_maps(
                resolve_file_path(directory, context),
                np.asarray(context.freq),
                nside=nside,
                suffix=spec.get("suffix", ".txt"),
                phi0_deg=phi0_deg,
                phi_sense=spec["phi_sense"],
            ),
            dtype=context.dtype,
        )
    if fmt == "uvbeam":
        return _uvbeam_maps(name, spec, context, nside)
    if fmt == "healpix":
        return _healpix_maps(name, spec, context, nside)
    if fmt in ("npy", "npz"):
        if "path" not in spec:
            raise ConfigError(
                f"{name}: format: {fmt} requires path: -- the file the maps come from."
            )
        node = {"file": {"path": spec["path"], "format": fmt}}
        if "key" in spec:
            node["file"]["key"] = spec["key"]
        destination = DestinationDescriptor(
            f"{name}.maps",
            "resource_field",
            f"rheplicant.config.kinds.beams.build_beam.{fmt}.maps",
        )
        resolved = resolve_value(node, context, destination=destination)
        maps = jnp.asarray(resolved.value, dtype=context.dtype)
        record_resolved_delivery(context, destination, resolved.unit)
        return maps
    if fmt == "inline":
        if "maps" not in spec:
            raise ConfigError(f"{name}: format: inline requires a 'maps' value node.")
        destination = DestinationDescriptor(
            f"{name}.maps",
            "resource_field",
            "rheplicant.config.kinds.beams.build_beam.inline.maps",
        )
        resolved = resolve_value(spec["maps"], context, destination=destination)
        maps = jnp.asarray(resolved.value, dtype=context.dtype)
        record_resolved_delivery(context, destination, resolved.unit)
        return maps
    if fmt == "python":
        if "python" not in spec:
            raise ConfigError(
                f"{name}: format: python requires a 'python:' target "
                "('pkg.mod:callable')."
            )
        args = spec.get("args", {})
        literal = spec.get("literal", {})
        for key, given in (("args", args), ("literal", literal)):
            if not isinstance(given, dict):
                raise ConfigError(
                    f"{name}: {key}: is a mapping of argument name to "
                    + ("value node" if key == "args" else "value")
                    + f", and this one is {type(given).__name__} ({given!r})."
                )
        # Refuse rather than let literal silently win on overlap, mirroring
        # sky_models.py's own kind: python branch and for the same reason:
        # which one "won" would decide whether a value node was resolved
        # through the grammar or forwarded untouched, and that is not a
        # choice a document should make by writing the same key twice.
        clash = sorted(set(args) & set(literal))
        if clash:
            raise ConfigError(
                f"{name}: {clash} appear in both args and literal. args values are "
                "resolved through the value grammar and literal values are forwarded "
                "untouched, so one argument cannot be both."
            )
        from rheplicant.config.values import make_resolution_target

        for key, value in args.items():
            make_resolution_target(
                value,
                DestinationDescriptor(
                    f"{name}.args.{key}",
                    "resource_field",
                    "rheplicant.config.kinds.beams.build_beam.python.args.*",
                ),
                context.dimensions,
            )
        factory = import_target(spec["python"])
        arguments = {}
        for key, value in args.items():
            destination = DestinationDescriptor(
                f"{name}.args.{key}",
                "resource_field",
                "rheplicant.config.kinds.beams.build_beam.python.args.*",
            )
            resolved = resolve_value(
                value,
                context,
                destination=destination,
            )
            arguments[key] = resolved.value
            record_resolved_delivery(context, destination, resolved.unit)
        arguments.update(literal)
        return jnp.asarray(factory(**arguments), dtype=context.dtype)
    return _gaussian(name, spec, context, nside)


def _require_pyuvdata(name: str):
    try:
        import pyuvdata
    except ImportError as exc:
        raise ConfigError(
            f"{name}: format: uvbeam needs pyuvdata, the 'uvbeam' extra: "
            'uv pip install -e ".[uvbeam]". The bridge itself ships with limTOD '
            "(limTOD.uvbeam); only the file reader is missing. Checked before the "
            "path is even resolved, because the alternative is a run that fails "
            "after everything else was built."
        ) from exc
    return pyuvdata


def _uvbeam_maps(name: str, spec: dict, context: ResolutionContext, nside: int):
    """A pyuvdata UVBeam file, sampled per channel through limTOD's bridge.

    ``uvbeam_to_healpix_maps`` takes ONE frequency per call, in MHz (measured
    against limTOD 1.10.0), and returns a ``(n_pix,)`` RING map with pixels
    beyond the beam's zenith-angle coverage at 0.0 -- so this loops over the
    run's grid and stacks. Stokes I only: ``Beam.maps`` is ``(n_freq, n_pix)``
    and the bridge's IQUV multi-row layout does not fit it.
    """
    pyuvdata = _require_pyuvdata(name)
    from limTOD.uvbeam import uvbeam_to_healpix_maps

    if "path" not in spec:
        raise ConfigError(f"{name}: format: uvbeam requires a 'path'.")
    if context.freq is None:
        raise ConfigError(
            f"{name}: format: uvbeam needs observation.freq.grid to sample onto."
        )
    path = resolve_file_path(spec["path"], context)
    try:
        uvb = pyuvdata.UVBeam.from_file(path)
        rows = [
            uvbeam_to_healpix_maps(uvb, freq_MHz=float(f) / 1e6, nside=nside)
            for f in np.asarray(context.freq)
        ]
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"{name}: reading or sampling the UVBeam at {str(path)!r} failed: {exc} "
            "The run's grid must lie inside the file's own frequency axis "
            "(pyuvdata interpolates, it does not extrapolate), and the beam must "
            "cover a regular (az, za) grid over the full azimuth range."
        ) from exc
    return jnp.asarray(np.stack(rows), dtype=context.dtype)


def _healpix_maps(name: str, spec: dict, context: ResolutionContext, nside: int):
    """HEALPix FITS beam maps: the declaration decides, and the header may veto.

    A HEALPix file does not state which chart its maps were sampled on, and an
    export's ``ORDERING`` header is exactly the kind of metadata that survives
    a conversion the data did not -- so RING-versus-NESTED is a required
    declaration (D-C7). Where the file DOES carry an ``ORDERING`` header, a
    declaration that contradicts it is refused rather than trusted: one of the
    two is wrong, this layer cannot know which, and a NESTED map read as RING
    keeps its shape and its statistics and puts every pixel in the wrong
    place. A declared ``nested`` is reordered to RING exactly -- a
    permutation, not an interpolation.

    The frequency-column convention is a declaration too: ``freq:`` states the
    grid the columns were built on, column i belonging to freq[i], and it must
    equal the run's own grid -- the maps are not interpolated, same as
    ``kind: maps`` on the sky side.
    """
    import healpy as hp

    for required in ("path", "order", "freq"):
        if required not in spec:
            raise ConfigError(
                f"{name}: format: healpix requires {required!r}. path names the file; "
                "order declares RING or NESTED, which the file cannot be trusted to "
                "state; freq declares the grid the columns were built on, column i "
                "belonging to freq[i]."
            )
    order = spec["order"]
    if order not in ("ring", "nested"):
        raise ConfigError(f"{name}: order={order!r}; it is 'ring' or 'nested'.")
    path = resolve_file_path(spec["path"], context)
    try:
        raw, header = hp.read_map(path, field=None, nest=None, h=True)
    except ConfigError:
        raise
    except Exception as exc:
        raise ConfigError(
            f"{name}: reading the HEALPix file at {str(path)!r} failed: {exc}"
        ) from exc
    stated = dict(header).get("ORDERING")
    if stated is not None:
        # "NEST" is not reachable through healpy's own writer -- hp.write_map
        # always records the full "RING" / "NESTED" (measured, healpy
        # 1.20.0) -- but the three-letter form is what the HEALPix FITS
        # convention itself documents, and other writers (e.g. HEALPix C++,
        # some CMB pipelines) use it. Mapped here rather than dropped, so a
        # file from one of those is trusted instead of silently falling
        # through to the "no header" branch below.
        stated_order = {"RING": "ring", "NESTED": "nested", "NEST": "nested"}.get(
            str(stated).strip().upper()
        )
        if stated_order is not None and stated_order != order:
            raise ConfigError(
                f"{name}: this entry declares order: {order} while the file's own "
                f"ORDERING header says {stated}. One of them is wrong and this layer "
                "cannot know which -- a NESTED map read as RING keeps its shape and "
                "its statistics and puts every pixel in the wrong place. Fix the "
                "declaration or re-export the file."
            )
    maps = np.atleast_2d(np.asarray(raw))
    if order == "nested":
        maps = np.stack([hp.reorder(row, n2r=True) for row in maps])
    destination = DestinationDescriptor(
        f"{name}.freq",
        "resource_field",
        "rheplicant.config.kinds.beams.build_beam.healpix.freq",
    )
    resolved = resolve_value(spec["freq"], context, destination=destination)
    freq = np.asarray(resolved.value)
    record_resolved_delivery(context, destination, resolved.unit)
    if freq.ndim != 1 or freq.shape[0] != maps.shape[0]:
        raise ConfigError(
            f"{name}: freq declares {tuple(freq.shape)} while the file carries "
            f"{maps.shape[0]} map column(s). Column i is freq[i], so the two lengths "
            "must agree."
        )
    if context.freq is not None:
        run = np.asarray(context.freq)
        if maps.shape[0] != run.shape[0]:
            raise ConfigError(
                f"{name}: the file carries {maps.shape[0]} map column(s) and this "
                f"run's frequency grid has {run.shape[0]} channels. Column i is "
                "freq[i]; there is no interpolation."
            )
        if not np.allclose(freq, run):
            raise ConfigError(
                f"{name}: the maps declare the grid [{freq.min():.6g}, "
                f"{freq.max():.6g}] Hz while this run observes on [{run.min():.6g}, "
                f"{run.max():.6g}] Hz -- the same NUMBER of channels, different "
                "channels. The maps are not interpolated; re-export them on the grid "
                "you mean to observe on."
            )
    return jnp.asarray(maps, dtype=context.dtype)


def _gaussian(name: str, spec: dict, context: ResolutionContext, nside: int):
    """A circular Gaussian in HEALPix RING, on the run's frequency grid."""
    healpy = _require_healpy(name)
    if ("fwhm_deg" in spec) == ("sigma_deg" in spec):
        raise ConfigError(
            f"{name}: format: gaussian takes exactly one of fwhm_deg or sigma_deg. Two "
            "widths for one beam have no defined relationship here, and none is a beam."
        )
    width_key = "fwhm_deg" if "fwhm_deg" in spec else "sigma_deg"
    destination = DestinationDescriptor(
        f"{name}.{width_key}",
        "resource_field",
        f"rheplicant.config.kinds.beams.build_beam.gaussian.{width_key}",
    )
    resolved = resolve_value(spec[width_key], context, destination=destination)
    sigma_deg = jnp.asarray(resolved.value)
    record_resolved_delivery(context, destination, resolved.unit)
    if width_key == "fwhm_deg":
        sigma_deg = sigma_deg / 2.3548200450309493
    n_freq = 1 if context.freq is None else int(context.freq.shape[0])
    widths_raw = jnp.atleast_1d(sigma_deg)
    n_widths = int(widths_raw.shape[0])
    if n_widths not in (1, n_freq):
        raise ConfigError(
            f"{name}: {width_key} declares {n_widths} width(s) and this run has "
            f"{n_freq} channel(s). A per-channel width needs one entry per channel, "
            "or a single entry broadcast to every channel -- not some other count, "
            "which jnp.broadcast_to would otherwise refuse with a bare shape error "
            "naming neither the key nor the run."
        )
    theta, _ = healpy.pix2ang(nside, np.arange(12 * nside * nside))
    theta_deg = jnp.asarray(np.degrees(theta), dtype=context.dtype)
    widths = jnp.broadcast_to(widths_raw, (n_freq,))
    return jnp.exp(-0.5 * (theta_deg[None, :] / widths[:, None]) ** 2).astype(context.dtype)


def _require_healpy(name: str):
    try:  # pragma: no cover - exercised only where healpy is absent
        import healpy
    except ImportError as exc:  # pragma: no cover
        raise ConfigError(
            f"{name}: needs healpy to place pixels on the sphere. It "
            "arrives with limTOD's own dependencies; if it is missing, the install is "
            'incomplete: pip install "limTOD[jax]>=1.10".'
        ) from exc
    return healpy


def _normalized(maps, normalize: str):
    if normalize == "none":
        return maps
    if normalize == "pixel_sum":
        return maps / jnp.sum(maps, axis=1, keepdims=True)
    n_pix = maps.shape[1]
    return maps / (jnp.sum(maps, axis=1, keepdims=True) * (4.0 * jnp.pi / n_pix))


def _horizon_angle(
    name: str,
    horizon: dict,
    key: str,
    default: float,
    context: ResolutionContext,
) -> float:
    defaulted = key not in horizon
    destination = DestinationDescriptor(
        f"{name}.horizon.{key}",
        "resource_field",
        f"rheplicant.config.kinds.beams.build_beam.horizon.{key}",
    )
    resolved = resolve_value(horizon.get(key, default), context, destination=destination)
    value = float(resolved.value)
    record_resolved_delivery(
        context, destination, resolved.unit, defaulted=defaulted
    )
    return value


def _truncate(name: str, maps, horizon: dict, context: ResolutionContext):
    from rheplicant.radio import horizon_truncated_beam

    el_deg = _horizon_angle(name, horizon, "el_deg", 90.0, context)
    if el_deg != 90.0:
        raise ConfigError(
            f"{name}: horizon.el_deg={el_deg}. truncate_map accepts only 90 -- limTOD's "
            "horizon partition is defined at the horizon and nowhere else. For a "
            "different cut, mask in the projector instead (horizon.mode: "
            "projector_mask), which applies it in the horizontal frame."
        )
    truncated, fraction = horizon_truncated_beam(
        np.asarray(maps), el_deg=el_deg,
        apod_deg=_horizon_angle(name, horizon, "apod_deg", 0.0, context)
    )
    return jnp.asarray(truncated), jnp.asarray(fraction)
