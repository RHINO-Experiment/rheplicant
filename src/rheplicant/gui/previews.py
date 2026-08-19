"""Free GUI previews and the visible price of explicit scientific work.

This module reads only the already-parsed YAML mapping.  It never constructs a
twin, opens a resource, imports JAX, or evaluates a model.  When a value cannot
be known from text (notably a file-backed grid), the preview stays unavailable
instead of guessing or reading ahead of the user's explicit action.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class PreviewClass:
    """One of schema §10.3's four, and only four, preview classes."""

    preview_id: Literal["graph", "axes_shapes", "validate", "forward"]
    label: str
    cadence: Literal["continuous", "explicit"]
    priced: bool
    description: str


@dataclass(frozen=True, slots=True)
class AxisPreview:
    """A bounded strip summarising one text-declared axis."""

    axis: Literal["time", "freq"]
    first: tuple[float, ...]
    last: tuple[float, ...]
    count: int
    spacing: float | None
    unit: str | None
    precision_ratio: float | None = None
    precision_ok: bool | None = None


@dataclass(frozen=True, slots=True)
class ShapePreview:
    """One shape symbol whose extent is decidable from the document text."""

    symbol: str
    value: int


@dataclass(frozen=True, slots=True)
class ForwardCost:
    """A deliberately approximate, explicitly labelled forward cost."""

    label: str
    estimated_milliseconds: float | None
    estimated_peak_megabytes: float | None
    n_freq: int | None
    nside: int | None
    lmax: int | None
    optimizations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PreviewProjection:
    """The complete free projection that accompanies an editor snapshot."""

    classes: tuple[PreviewClass, ...]
    axes: tuple[AxisPreview, ...]
    shapes: tuple[ShapePreview, ...]
    forward_cost: ForwardCost
    declared_run_kinds: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _AxisValues:
    first: tuple[float, ...]
    last: tuple[float, ...]
    count: int
    spacing: float | None
    precision_cadence: float | None
    precision_peak: float | None


_CLASSES = (
    PreviewClass(
        "graph",
        "Signal path",
        "continuous",
        False,
        "Text-only graph projection; no Assembly is constructed.",
    ),
    PreviewClass(
        "axes_shapes",
        "Axes and shapes",
        "continuous",
        False,
        "Text-declared axis strips, shape symbols, and time precision.",
    ),
    PreviewClass(
        "validate",
        "Validate",
        "explicit",
        True,
        "Builds the configured layers and runs the priced validation pass.",
    ),
    PreviewClass(
        "forward",
        "Preview forward",
        "explicit",
        True,
        "Runs one forward-only schedule and never a fit.",
    ),
)


def _number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    try:
        found = float(value)
    except OverflowError:
        return None
    return found if math.isfinite(found) else None


def _count(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _linear_values(
    start: float,
    spacing: float | None,
    count: int,
) -> _AxisValues | None:
    step = spacing or 0.0
    first = tuple(start + step * index for index in range(min(3, count)))
    last = tuple(start + step * index for index in range(max(0, count - 3), count))
    if not all(math.isfinite(value) for value in (*first, *last)):
        return None
    cadence = abs(spacing) if spacing not in (None, 0.0) else None
    peak = max(abs(value) for value in (*first, *last))
    return _AxisValues(first, last, count, spacing, cadence, peak)


def _axis_values(node: object) -> _AxisValues | None:
    if not isinstance(node, Mapping):
        return None
    if "list" in node:
        raw = node["list"]
        if isinstance(raw, str | bytes) or not isinstance(raw, Sequence) or not raw:
            return None
        values = tuple(_number(value) for value in raw)
        if any(value is None for value in values):
            return None
        numeric = tuple(value for value in values if value is not None)
        spacing = numeric[1] - numeric[0] if len(numeric) > 1 else None
        if spacing is not None and any(
            not math.isclose(right - left, spacing, rel_tol=1e-12, abs_tol=0.0)
            for left, right in zip(numeric, numeric[1:], strict=False)
        ):
            spacing = None
        gaps = tuple(
            abs(right - left)
            for left, right in zip(numeric, numeric[1:], strict=False)
        )
        distinct = tuple(value for value in gaps if value > 0.0)
        return _AxisValues(
            numeric[:3],
            numeric[-3:],
            len(numeric),
            spacing,
            min(distinct, default=None),
            max(abs(value) for value in numeric),
        )
    for form in ("linspace", "arange", "modulo"):
        if form not in node:
            continue
        spec = node[form]
        if not isinstance(spec, Mapping):
            return None
        num = _count(spec.get("num"))
        if num is None:
            return None
        if form == "linspace":
            start = _number(spec.get("start"))
            stop = _number(spec.get("stop"))
            endpoint = spec.get("endpoint")
            if start is None or stop is None or type(endpoint) is not bool:
                return None
            spacing = None if num == 1 else (stop - start) / (num - 1 if endpoint else num)
            return _linear_values(start, spacing, num)
        if form == "arange":
            start = _number(spec.get("start"))
            spacing = _number(spec.get("step"))
            if start is None or spacing is None:
                return None
            return _linear_values(start, spacing, num)
        period = _count(spec.get("period"))
        if period is None:
            return None
        first = tuple(float(index % period) for index in range(min(3, num)))
        last = tuple(float(index % period) for index in range(max(0, num - 3), num))
        return _AxisValues(first, last, num, None, None, None)
    return None


def _axis(document: Mapping[str, object], name: str) -> AxisPreview | None:
    observation = document.get("observation")
    if not isinstance(observation, Mapping):
        return None
    section = observation.get(name)
    if not isinstance(section, Mapping):
        return None
    grid = section.get("grid")
    resolved = _axis_values(grid)
    if resolved is None:
        return None
    unit = grid.get("unit") if isinstance(grid, Mapping) else None
    unit = unit if isinstance(unit, str) else None
    ratio = ok = None
    if name == "time":
        import numpy as np

        runtime = document.get("runtime")
        x64 = runtime.get("jax_enable_x64") is True if isinstance(runtime, Mapping) else False
        factor = {"s": 1.0, "ms": 1.0e-3}.get(unit.lower()) if unit else None
        if factor is not None:
            dtype = np.float64 if x64 else np.float32
            cadence = resolved.precision_cadence
            peak = resolved.precision_peak
            if cadence is not None and peak is not None:
                with np.errstate(over="ignore", invalid="ignore"):
                    stored_peak = dtype(peak * factor)
                    resolution = abs(float(np.spacing(stored_peak)))
                if (
                    math.isfinite(float(stored_peak))
                    and resolution > 0.0
                    and math.isfinite(resolution)
                ):
                    ratio = cadence * factor / resolution
                    ok = ratio >= 100.0
    return AxisPreview(
        name,  # type: ignore[arg-type]
        resolved.first,
        resolved.last,
        resolved.count,
        resolved.spacing,
        unit,
        ratio,
        ok,
    )


def _source_count(document: Mapping[str, object]) -> int:
    observation = document.get("observation")
    switching = observation.get("switching") if isinstance(observation, Mapping) else None
    if not isinstance(switching, Mapping) or switching.get("mode") == "none":
        return 1
    order = switching.get("order")
    if isinstance(order, list) and order:
        return len(order)
    return 1


def _resource_shapes(document: Mapping[str, object]) -> tuple[ShapePreview, ...]:
    resources = document.get("resources")
    if not isinstance(resources, Mapping):
        return ()
    rows: list[ShapePreview] = []
    for group_name, group in resources.items():
        if not isinstance(group_name, str) or not isinstance(group, Mapping):
            continue
        for name, entry in group.items():
            if not isinstance(name, str) or not isinstance(entry, Mapping):
                continue
            nside = _count(entry.get("nside"))
            lmax = _count(entry.get("lmax"))
            if nside is not None:
                rows.append(
                    ShapePreview(
                        f"resources.{group_name}.{name}.n_pix",
                        12 * nside * nside,
                    )
                )
            if lmax is not None:
                rows.append(
                    ShapePreview(
                        f"resources.{group_name}.{name}.n_alm",
                        (lmax + 1) * (lmax + 2) // 2,
                    )
                )
    return tuple(rows)


def _optimizations(document: Mapping[str, object]) -> tuple[str, ...]:
    resources = document.get("resources")
    projectors = resources.get("projectors") if isinstance(resources, Mapping) else None
    found: list[str] = []
    if isinstance(projectors, Mapping):
        for entry in projectors.values():
            declared = entry.get("optimizations") if isinstance(entry, Mapping) else None
            if isinstance(declared, list):
                found.extend(value for value in declared if isinstance(value, str))
    return tuple(dict.fromkeys(found))


def _max_resource_int(document: Mapping[str, object], key: str) -> int | None:
    resources = document.get("resources")
    if not isinstance(resources, Mapping):
        return None
    values = [
        value
        for group in resources.values()
        if isinstance(group, Mapping)
        for entry in group.values()
        if isinstance(entry, Mapping)
        for value in (_count(entry.get(key)),)
        if value is not None
    ]
    return max(values, default=None)


def _cost(document: Mapping[str, object], n_freq: int | None) -> ForwardCost:
    nside = _max_resource_int(document, "nside")
    lmax = _max_resource_int(document, "lmax")
    optimizations = _optimizations(document)
    if n_freq is None:
        return ForwardCost(
            "Cost unavailable until the document declares complete axes",
            None,
            None,
            None,
            nside,
            lmax,
            optimizations,
        )
    effective_nside = nside or 1
    effective_lmax = lmax if lmax is not None else max(3 * effective_nside - 1, 0)
    scale = (n_freq / 32.0) * (effective_nside / 64.0) ** 2
    band_scale = max((effective_lmax + 1) / 192.0, 1.0 / 192.0)
    scale *= band_scale
    if "cache_beam_rotation" in optimizations:
        scale *= 0.4
    milliseconds = max(0.01, 193.0 * scale)
    peak_megabytes = max(
        0.01,
        114.0 * (n_freq / 32.0) * (effective_nside / 64.0) ** 2 * band_scale,
    )
    label = (
        f"estimated {milliseconds:.2f} ms / {peak_megabytes:.2f} MB "
        f"for {n_freq} channels"
    )
    return ForwardCost(
        label,
        milliseconds,
        peak_megabytes,
        n_freq,
        nside,
        lmax,
        optimizations,
    )


def _run_kinds(document: Mapping[str, object]) -> tuple[str, ...]:
    runs = document.get("runs", ())
    if isinstance(runs, Mapping):
        runs = (runs,)
    if isinstance(runs, str | bytes) or not isinstance(runs, Sequence):
        return ()
    return tuple(
        kind
        for row in runs
        if isinstance(row, Mapping)
        for kind in (row.get("kind"),)
        if isinstance(kind, str)
    )


def project_previews(document: Mapping[str, object]) -> PreviewProjection:
    """Project every free preview and the price label for explicit work."""
    axes = tuple(
        found for name in ("time", "freq") if (found := _axis(document, name)) is not None
    )
    lengths = {row.axis: row.count for row in axes}
    shapes: list[ShapePreview] = []
    if "time" in lengths:
        shapes.append(ShapePreview("n_time", lengths["time"]))
    if "freq" in lengths:
        shapes.append(ShapePreview("n_freq", lengths["freq"]))
    sources = _source_count(document)
    shapes.extend((ShapePreview("n_source", sources), ShapePreview("n_load", sources - 1)))
    shapes.extend(_resource_shapes(document))
    return PreviewProjection(
        _CLASSES,
        axes,
        tuple(shapes),
        _cost(document, lengths.get("freq")),
        _run_kinds(document),
    )


__all__ = [
    "AxisPreview",
    "ForwardCost",
    "PreviewClass",
    "PreviewProjection",
    "ShapePreview",
    "project_previews",
]
