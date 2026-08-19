"""ResolutionContext: everything a value node may resolve against.

It is a frozen dataclass and every widening returns a new one. That is the
package's own rule, and here it earns its keep twice: ``{ref: ...}`` resolves
to the *same Python object*, so a test can assert ``is``; and a resource under
construction cannot see itself, which is what makes a cycle in ``resources``
detectable rather than an infinite recursion.
"""

import dataclasses
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, TypeVar, cast

import jax

from _rheplicant_bootstrap.capture import CaptureService
from _rheplicant_bootstrap.types import LayerIdentity, OriginLookup, TraceSink
from rheplicant.config.dimensions import (
    DimensionEnvironment,
    bind_resource_dimension,
    current_dimension_environment,
    dimension_for,
)
from rheplicant.config.errors import ConfigError
from rheplicant.config.resolution_audit import (
    ResolutionAudit,
    ResolutionAuditSink,
)
from rheplicant.config.symbols import ShapeScope

_ACTIVE_LAYER: ContextVar[LayerIdentity | None] = ContextVar(
    "rheplicant_resolution_layer", default=None
)
_ACTIVE_TRACE: ContextVar[TraceSink | None] = ContextVar(
    "rheplicant_resolution_trace", default=None
)
_ACTIVE_ORIGIN_LOOKUP: ContextVar[OriginLookup | None] = ContextVar(
    "rheplicant_resolution_origin_lookup", default=None
)
_ACTIVE_CAPTURE: ContextVar[CaptureService | None] = ContextVar(
    "rheplicant_resolution_capture", default=None
)

T = TypeVar("T")


@contextmanager
def using_resolution_audit(
    layer: LayerIdentity,
    trace: TraceSink | None,
    origin_lookup: OriginLookup | None,
    capture: CaptureService | None = None,
):
    """Give every context created during one layer its audit authority."""
    layer_token = _ACTIVE_LAYER.set(layer)
    trace_token = _ACTIVE_TRACE.set(trace)
    origin_token = _ACTIVE_ORIGIN_LOOKUP.set(origin_lookup)
    capture_token = _ACTIVE_CAPTURE.set(capture)
    try:
        yield
    finally:
        _ACTIVE_CAPTURE.reset(capture_token)
        _ACTIVE_ORIGIN_LOOKUP.reset(origin_token)
        _ACTIVE_TRACE.reset(trace_token)
        _ACTIVE_LAYER.reset(layer_token)


def _active_layer() -> LayerIdentity:
    return _ACTIVE_LAYER.get() or LayerIdentity("base", None)


def _active_audit() -> ResolutionAudit | None:
    trace = _ACTIVE_TRACE.get()
    required = (
        "record_resource",
        "record_seed",
        "record_gate",
        "record_python_target",
        "record_deletion",
        "snapshot",
    )
    if trace is None or any(not callable(getattr(trace, method, None)) for method in required):
        return None
    return ResolutionAudit(_active_layer(), cast(ResolutionAuditSink, trace))


def current_resolution_audit() -> ResolutionAudit | None:
    """Return the current layer audit during document construction."""
    return _active_audit()


@dataclasses.dataclass(frozen=True)
class ResolutionContext:
    """The scope a value node resolves in.

    Attributes:
        freq: the run's frequency grid, Hz. ``{from_grid: freq}`` reads it and
            ``{from: channel_spacing}`` measures it.
        time: the run's time grid, seconds from the start of the run.
        dtype: ``"float32"`` or ``"float64"`` -- the run's floating dtype.
        base_dir: the directory of the document that mentioned a path.
        roots: extra roots a relative path is tried against, in order.
        seed: the run's root PRNG seed, or None when the run realises nothing.
        seeds: ``runtime.seeds`` -- an open namespace of user-named ints.
        switch_order: ``observation.switching.order``; index 0 is ``antenna``.
        resources: constructed resources by dotted name. ``ref`` returns the
            object stored here, not a copy.
        n_source_override: set when ``switch_order`` is not yet known.
        ingest: the ingested recording (``RhinoObservation``) when
            ``observation.from_file`` was declared; ``from: thermistors``
            reads its thermistor log.
    """

    freq: jax.Array | None = None
    time: jax.Array | None = None
    dtype: str = "float32"
    base_dir: str | None = None
    roots: tuple[str, ...] = ()
    seed: int | None = None
    seeds: dict[str, int] = dataclasses.field(default_factory=dict)
    switch_order: tuple[str, ...] = ()
    resources: dict[str, Any] = dataclasses.field(default_factory=dict)
    n_source_override: int | None = None
    ingest: Any = None
    dimensions: DimensionEnvironment = dataclasses.field(
        default_factory=current_dimension_environment
    )
    layer: LayerIdentity = dataclasses.field(default_factory=_active_layer)
    trace: TraceSink | None = dataclasses.field(default_factory=_ACTIVE_TRACE.get)
    origin_lookup: OriginLookup | None = dataclasses.field(
        default_factory=_ACTIVE_ORIGIN_LOOKUP.get
    )
    capture: CaptureService | None = dataclasses.field(default_factory=_ACTIVE_CAPTURE.get)
    audit: ResolutionAudit | None = dataclasses.field(default_factory=_active_audit)

    @property
    def shape_scope(self) -> ShapeScope:
        """The extents a shape symbol resolves against, taken off the axes."""
        n_source = self.n_source_override
        if n_source is None:
            n_source = len(self.switch_order) or 1
        return ShapeScope(
            n_time=0 if self.time is None else int(self.time.shape[0]),
            n_freq=0 if self.freq is None else int(self.freq.shape[0]),
            n_source=n_source,
            candidates=tuple(
                name
                for name in self.resources
                if name.startswith(("resources.beams.", "resources.projectors."))
            ),
        )

    def use_default(self, path: str, value: T) -> T:
        """Return one chosen default and record it when this load is audited."""
        if self.audit is None:
            return value
        return self.audit.use_default(path, value)

    def with_resource(self, name: str, value: Any) -> "ResolutionContext":
        """A new context carrying one more constructed resource."""
        from _rheplicant_bootstrap.types import DestinationDescriptor

        candidates = [name]
        candidates.extend(
            f"{name}.{attribute}"
            for attribute in ("time", "freq", "maps", "sky_fraction")
            if hasattr(value, attribute)
        )
        for dotted in candidates:
            try:
                found = dimension_for(
                    DestinationDescriptor(dotted, "config_path", dotted),
                    self.dimensions,
                )
            except ConfigError:
                continue
            missing = object()
            existing = self.dimensions.resource_dimensions.get(dotted, missing)
            if existing is missing:
                bind_resource_dimension(self.dimensions, dotted, found)
            elif existing is None:
                self.dimensions.resource_dimensions[dotted] = found
            elif existing != found:
                raise ConfigError(
                    f"dimensions: resource {dotted!r} was rebound with a different dimension"
                )
        return dataclasses.replace(self, resources={**self.resources, name: value})
