"""ResolutionContext: everything a value node may resolve against.

It is a frozen dataclass and every widening returns a new one. That is the
package's own rule, and here it earns its keep twice: ``{ref: ...}`` resolves
to the *same Python object*, so a test can assert ``is``; and a resource under
construction cannot see itself, which is what makes a cycle in ``resources``
detectable rather than an infinite recursion.
"""

import dataclasses
from typing import Any

import jax

from rheplicant.config.symbols import ShapeScope


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

    def with_resource(self, name: str, value: Any) -> "ResolutionContext":
        """A new context carrying one more constructed resource."""
        return dataclasses.replace(self, resources={**self.resources, name: value})
