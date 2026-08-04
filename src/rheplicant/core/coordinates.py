"""Coordinates: the traced coordinate container flowing with a State.

All fields are optional and traced (they are pytree leaves, so they can be
jitted / vmapped / differentiated through). Validation runs at construction and
again on every functional ``replace``.

Angle convention (RHINO family): degrees in public-facing APIs, radians
internally. This module stores whatever it is given — the convention is
enforced by operators, not by the container.

**One value check, and why it lives here.** Validation in this package is
otherwise structural (ndim only) and therefore value-independent and jit-safe.
``time`` is the exception, because storing it is itself lossy: the converter
below calls ``jnp.asarray``, which is float32 unless x64 is enabled, and a
unix-second axis (~1.75e9) has a float32 resolution of 128 s. Samples 100 s
apart merge *at store time*, and no later subtraction can undo it — the
quantities every consumer reads are already the rounded ones, so a consumer's
own consistency checks compare corrupted values against corrupted values and
see nothing wrong. That makes the container the last place the loss is
attributable to anything: one stage later there is only a shorter list of
plausible-looking timestamps. See
:func:`_refuse_a_time_axis_the_stored_dtype_cannot_carry`.
"""

import dataclasses
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import StateValidationError

#: Largest fraction of one sample interval that ``coords.time``'s own
#: representable resolution may occupy.
#:
#: 1e-2 rather than something tighter because a sample is an AVERAGE over its
#: own integration, so its time tag is only meaningful to within one interval
#: to begin with; demanding that the representation error be a hundredth of
#: that leaves two orders of magnitude of headroom below what the axis itself
#: means. At a ratio of 1.0 samples merge outright, which is the measured
#: failure — 100x beyond this cut. Seconds from the start of the run, of the
#: day (86400 -> 3.9e-5 of a 100 s interval) or of the month (2.6e6 -> 2.5e-3)
#: all pass; seconds from the start of the YEAR (3.15e7 -> 2e-2) do not, and
#: should not, because elapsed times there are already wrong by 2 s.
#:
#: **What this costs a long run.** For a uniform axis measured from its own
#: start the peak is ``n_samples * cadence``, so the ratio is
#: ``spacing(n*cadence) / cadence`` — and since ``spacing(x)`` is within a
#: factor of two of ``x * 2**-23``, the cadence very nearly cancels and what the
#: cut constrains is the sample COUNT. The limit for float32 is therefore
#: **of order 1e5 uniform samples**, anywhere in ``[1e-2 * 2**23, 1e-2 * 2**24]``
#: = [8.4e4, 1.7e5]; the exact count depends on where ``n * cadence`` falls
#: inside its binade, so it is not one number (at 1 s cadence it is exactly
#: 2**17 = 131072). A four-hour RHINO run at 1 s is 1.4e4 samples, an order of
#: magnitude clear; the same run at 0.05 s is 2.9e5 and is refused. That is a
#: real limit of fixing the axis by making it RELATIVE rather than float64, and
#: it is stated rather than papered over: relative buys about five decimal
#: orders over a unix epoch, not unlimited range.
#:
#: Stated once, here, because it is a property of how ``coords.time`` is
#: STORED rather than of any one operator's arithmetic:
#: :mod:`rheplicant.radio.instrument.calibration` imports it rather than
#: keeping its own copy.
MAX_TIME_RESOLUTION_IN_SAMPLES = 1e-2


def as_array_or_none(value: Any) -> jax.Array | None:
    """Converter: pass ``None`` through, coerce everything else to a jax array."""
    return None if value is None else jnp.asarray(value)


def _refuse_a_time_axis_the_stored_dtype_cannot_carry(times: jax.Array) -> None:
    """Refuse a ``coords.time`` whose STORED precision has eaten its own cadence.

    Measured, on 8 samples 100 s apart anchored at unix 1.75e9, before this
    check existed::

        stored (float32)  [1750000000, 1750000128, 1750000256, 1750000256,
                           1750000384, 1750000512, 1750000640, 1750000640]
        BackendOperator(n_chunk=2) chunk times
                          [1750000128, 1750000256, 1750000384, 1750000640]
        float64 truth     [1750000050, 1750000250, 1750000450, 1750000650]
        error [s]         [78, 6, -66, -10]

    Two of the eight samples had already merged before the average ran, and
    every chunk timestamp is wrong — by 78 s out of a 100 s cadence at worst.
    Nothing raised, nothing was NaN, every shape was right.

    Four details are load-bearing, and each is a way of getting this wrong:

    * ``np.spacing`` is taken on the array's own scalar, never on
      ``float(...)``. The question is what the STORED dtype can represent, and
      ``np.spacing(float(x))`` answers it for float64 (2.4e-7 s at unix
      seconds) no matter what the array actually holds (128 s in float32) —
      blind to the one thing being guarded.
    * Non-finite values are named FIRST, not left to the comparison. NaN
      compares False against everything, so a NaN gap is not positive and drops
      out of "the smallest distinct gap"; an all-NaN axis then has no gap left
      to test and would sail through untouched.
    * ``np.abs`` on both the peak and the gaps. ``np.spacing`` of a negative
      number is negative, and a negative resolution compares below any positive
      threshold, so an axis anchored on a future epoch would be waved through;
      and it is the MAGNITUDE of the sample interval that has to be resolved,
      which a descending axis has just as much as an ascending one.
    * Only INEXACT dtypes are checked. ``np.spacing`` on an integer promotes to
      float64 and answers 5.7e-14 — the same dtype blindness from the other
      side. An integer axis represents integers exactly, and whatever was
      truncated was truncated before this container saw it.

    The gap taken is the smallest *distinct* one, which is where this
    deliberately stops short of
    ``CWCalibrationOperator._refuse_a_time_axis_it_cannot_resolve``: that one
    counts a zero gap as a refusal, because it subtracts times and a collapsed
    pair silently stops its tone drifting. A container cannot tell a genuine
    repeated timestamp from a collision, and refusing every repeat would make
    it reject data it has no business judging. Nothing is lost on the defect
    this exists for: rounding makes every surviving gap a multiple of the
    representable spacing, so a uniformly quantised axis that has collided
    still shows a smallest distinct gap of one or two grid steps and is refused
    here anyway. What does escape is an axis where one isolated close pair
    merged while the rest stayed coarse — the pre-conversion values are gone by
    the time this runs, and only float64 or a relative axis defends against
    that.

    Traced arrays are stepped over rather than forced: under jit / vmap / grad
    there are no values to compare, and calling ``np.asarray`` on a tracer is
    the error, not the axis.
    """
    try:
        values = np.asarray(times)
    except jax.errors.TracerArrayConversionError:
        return  # genuinely traced: no values to compare against
    if not np.issubdtype(values.dtype, np.inexact) or values.size == 0:
        return

    if not np.all(np.isfinite(values)):
        n_bad = int((~np.isfinite(values)).sum())
        first = int(np.flatnonzero(~np.isfinite(values))[0])
        raise StateValidationError(
            f"coords.time holds {n_bad} non-finite value(s), the first at index "
            f"{first} (0-based). A sample time is never legitimately NaN or "
            "infinite, and it is refused here rather than left to the resolution "
            "check below because NaN compares False against every bound: a NaN "
            "gap is not positive, so it drops out of the sampling estimate, and "
            "an all-NaN axis has no gap left to test at all and would pass a "
            "comparison-based guard untouched."
        )

    gaps = np.abs(np.diff(values))
    distinct = gaps[gaps > 0]
    if distinct.size == 0:
        return  # one sample, or one repeated timestamp: no sampling to resolve

    peak = np.abs(values).max()
    resolution = float(np.spacing(peak))
    cadence = float(distinct.min())
    if resolution <= MAX_TIME_RESOLUTION_IN_SAMPLES * cadence:
        return
    raise StateValidationError(
        f"coords.time is stored as {values.dtype} and reaches {float(peak):.9g}, "
        f"where consecutive representable numbers are {resolution:.6g} apart — but "
        f"the closest two distinct samples on this axis are {cadence:.6g} apart, "
        f"and coords.time must resolve its own sampling to at most "
        f"{MAX_TIME_RESOLUTION_IN_SAMPLES:g} of that. The rounding happens when the "
        "axis is STORED, so no later subtraction recovers it: samples merge, and "
        "every consumer that does arithmetic on the values then reads the rounded "
        "ones — BackendOperator averages merged times into chunk timestamps wrong "
        "by tens of seconds, and CWCalibrationOperator drifts its tone against "
        "them. Neither raises, because their own consistency checks compare the "
        "corrupted values against each other. read_rhino_observation reports unix "
        "seconds (~1.75e9), which quantise onto a 128 s grid in float32; "
        "rheplicant.radio.rhino.to_state therefore stores time measured from the "
        "start of the run and keeps the epoch in meta['time_epoch_unix_s']. Do the "
        "same for a hand-built axis, or enable float64 (JAX_ENABLE_X64=1, or "
        "jax.config.update('jax_enable_x64', True))."
    )


class Coordinates(eqx.Module):
    """Coordinate axes of the data flowing through a pipeline.

    Attributes:
        time: ``(n_time,)`` sample times [seconds].
        freq: ``(n_freq,)`` channel frequencies [Hz].
        pointing: ``(n_time, k)`` pointing coordinates (e.g. alt/az pairs, k=2).
        extra: dict of additional *traced* coordinate arrays (e.g. spatial grids).
    """

    time: jax.Array | None = eqx.field(default=None, converter=as_array_or_none)
    freq: jax.Array | None = eqx.field(default=None, converter=as_array_or_none)
    pointing: jax.Array | None = eqx.field(default=None, converter=as_array_or_none)
    extra: dict[str, Any] = eqx.field(default_factory=dict, converter=dict)

    def __check_init__(self):
        # Structural (shape-rank) checks, jit-safe and value-independent, plus
        # the one value check the module docstring argues for: `time`'s stored
        # precision. Rank first, so a 2-D `time` is diagnosed as a rank error
        # rather than by np.diff along the wrong axis.
        if self.time is not None and self.time.ndim != 1:
            raise StateValidationError(f"coords.time must be 1D, got ndim={self.time.ndim}")
        if self.time is not None:
            _refuse_a_time_axis_the_stored_dtype_cannot_carry(self.time)
        if self.freq is not None and self.freq.ndim != 1:
            raise StateValidationError(f"coords.freq must be 1D, got ndim={self.freq.ndim}")
        if self.pointing is not None and self.pointing.ndim != 2:
            raise StateValidationError(
                f"coords.pointing must be 2D (n_time, k), got ndim={self.pointing.ndim}"
            )
        if not all(isinstance(k, str) for k in self.extra):
            raise StateValidationError("coords.extra keys must be strings")

    def replace(self, **changes: Any) -> "Coordinates":
        """Functional update: return a new Coordinates with ``changes`` applied.

        Re-runs converters and validation (unlike raw ``eqx.tree_at``).
        """
        return dataclasses.replace(self, **changes)
