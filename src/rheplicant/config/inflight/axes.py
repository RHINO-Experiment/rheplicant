"""C1 and C2 -- what the run's own axes decide, in front of the beam.

Two rules live here, and they are the two the resolved grids answer on their
own: the time axis's stored precision (C1), and a non-finite value on any of
the four axes schema §6's C2 row names (C2).

**Why the axes slot and not the text pass.**  Neither is decidable from the
document's text.  ``observation.time.grid`` is a value node -- a
``linspace:``, an ``arange:``, a ``{file: ...}`` -- so what the axis actually
HOLDS exists only once ``build_observation`` has resolved it, and
``observation.pointing``'s ``lst_deg`` and ``selfrot_deg`` are materialised by
``compile_pointing`` inside that same call.  Both are in hand at this slot and
at no earlier one.

**Why not later, which is where they are today.**

* **C1.**  ``core/coordinates.py::_refuse_a_time_axis_the_stored_dtype_cannot_carry``
  is correct and fires dead last.  ``Coordinates(...)`` is constructed at one
  place in ``config/`` -- ``document.py``'s ``_assemble``, **after**
  ``build_model`` -- so a 262 144-sample float32 axis anchored at unix
  1.75e9 costs the beam read and the spherical harmonic transform before the
  refusal.  Measured in this worktree on the worked document: **0.151 s** to
  that refusal, against **0.0003 s** to build this pass's whole payload.  (The
  plan quotes 1.79 s; that is a beam-bearing document's number, and 0.151 s is
  this one's.  Either way the ratio is three orders of magnitude.)  It also
  arrives as a ``StateValidationError``, so a caller wrapping ``load_document``
  in ``except ConfigError`` to report "this document is wrong" does not see it;
  through this slot it is a ``ConfigError`` like every other refusal the layer
  makes.
* **C2.**  Missing entirely, and silently.  Measured: a NaN at index 5 of a
  32-sample ``lst_deg`` loads clean and evaluates clean, and the schema's own
  C2 row records what happens next -- "the adjoint would return a finite,
  correctly-shaped, identically **zero** map".  An output-NaN check sees
  nothing, because there is no NaN in the output.

**One binding, not a second implementation.**  C1's rule and its constant are
:mod:`rheplicant.core.coordinates`'s.  This module imports the function and
calls it; the sentence a user reads is still the one written there, with this
pass's own tail appended saying where it was caught and why that matters.
``tests/config/test_inflight_axes.py`` asserts that tail is bound in exactly
one module and that the hoisted sentence is bound in ``core/coordinates.py``.

**The twin C1 does NOT collapse.**
``CWCalibrationOperator._refuse_a_time_axis_it_cannot_resolve`` is a second
time-axis guard, and ``core/coordinates.py``'s own docstring names it and says
how the two differ: the operator counts a **zero** gap as a refusal, because
it subtracts times and a collapsed pair silently stops its tone drifting,
while the container takes the smallest DISTINCT gap and cannot tell a genuine
repeated timestamp from a collision.  This pass hoists the CONTAINER's, for
three reasons: it is already a module-level pure function, it applies to every
run rather than to a run carrying a ``cw_tone`` node, and it is the only one
of the two with the non-finite branch C2 needs.  The operator's stronger
zero-gap rule is therefore a declared false negative of this pass, recorded in
the plan's §7 -- **and no third guard is written here.**

**What this pass does not walk.**  ``Axes.document`` is the VARIANT-APPLIED
mapping, so unlike a pre-flight check there is no ``variants:`` twin to close:
one layer is selected and it is the one built.  What is NOT covered is the
INGESTED state's own ``coords.time``: ``document.py`` builds that from
``rhino.to_state(observation.ingest, ...)``, whose axis is the recording's,
not the relative ``observation.time_s`` this pass reads.  Recorded rather than
guessed at.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax
import numpy as np

from rheplicant.config.findings import Finding, refuse
from rheplicant.config.inflight import Axes, register_axes
from rheplicant.core.coordinates import (
    _refuse_a_time_axis_the_stored_dtype_cannot_carry,
)
from rheplicant.core.errors import StateValidationError

#: The tail this pass appends to ``core/coordinates.py``'s own sentence.  The
#: hoisted words are NOT restated here -- they are interpolated -- so the rule
#: keeps one binding and this module carries only what the container cannot
#: know: which phase caught it, and what catching it later costs.
_TIME_TAIL = (
    "This is the axes pass, which runs after build_observation and before "
    "build_resources. The same guard runs again at Coordinates(...), which "
    "config/document.py reaches only after build_model -- so until now an "
    "axis this arithmetic refuses cost a beam read and a spherical harmonic "
    "transform first, and arrived as a StateValidationError rather than as "
    "this layer's own refusal"
)

#: The ``coords.extra`` keys schema §6's C2 row names, and no others.
#: ``coords.extra`` is an OPEN dict -- ``observation.extra`` writes whatever
#: the document declares and ``compile_switching`` adds ``receiver_input`` --
#: so the walk is over these NAMED keys rather than over the mapping.  A
#: sweep of everything in ``extra`` would refuse a legitimately-masked user
#: array and would report ``receiver_input``, an integer index vector, as a
#: bad pointing.
_NAMED_EXTRA = ("lst_deg", "selfrot_deg")


def _values(axis: Any):
    """``axis`` as concrete numpy values, or ``None`` if it is traced or absent.

    The tracer branch mirrors the hoisted guard's own: under a trace there are
    no values to compare and calling ``np.asarray`` on a tracer is the error,
    not the axis.  Nothing reaches this slot traced today -- every axis here
    was resolved from the document's text -- and the branch is kept because
    the guard it stands in for keeps its own.

    **MEASURED EQUIVALENT MUTANT, kept anyway and said so rather than left to
    be re-discovered.**  Making the ``except`` re-raise instead of standing
    down survives the whole suite.  Measured with ``coverage`` over all of
    ``tests/config``: the ``except`` line and its ``return None`` are never
    executed at all, and neither is the ``values is None`` arm the two check
    functions take when it fires -- they are the only unreached lines on
    either check's path.  So there is no document that can tell the two
    implementations apart, and there will not be one until something hands
    this slot a traced axis.  The branch stays because ``Axes.context`` is a
    public payload, because the guard this restates keeps its own, and
    because a re-raise would turn a future traced axis into a
    ``TracerArrayConversionError`` escaping a check -- which the runner
    reports as *"in-flight check 'C1' RAISED"* and which loses every other
    finding on the document.
    """
    if axis is None:
        return None
    try:
        return np.asarray(axis)
    except jax.errors.TracerArrayConversionError:
        return None


def _time_where(document: Mapping[str, Any]) -> str:
    """Where the reader must go to change the time axis.

    ``observation.from_file`` for an ingested run -- there is no
    ``observation.time`` on one, ``build_observation`` refuses the pair -- and
    ``observation.time.grid`` otherwise.
    """
    observation = document.get("observation")
    if isinstance(observation, Mapping) and "from_file" in observation:
        return "observation.from_file"
    return "observation.time.grid"


def _non_finite(check: str, where: str, subject: str, values) -> Finding:
    """One refusal about one axis carrying a NaN or an infinity.

    Named FIRST and separately from any comparison, for the reason
    ``core/coordinates.py`` gives about its own axis: NaN compares False
    against every bound, so a NaN drops out of every "the smallest gap" or
    "the largest value" estimate and an all-NaN axis has nothing left to test.
    """
    bad = ~np.isfinite(values)
    # By SAMPLE and not by flat position: `coords.pointing` is (n_time, k), so
    # a flat index would report 1 for a NaN elevation on the first sample and
    # send the reader to the second one. Axis 0 is n_time on all four subjects.
    rows = bad if bad.ndim == 1 else bad.reshape(bad.shape[0], -1).any(axis=1)
    return refuse(
        check, where,
        f"{where}: {subject} holds {int(bad.sum())} non-finite value(s), the "
        f"first at sample {int(np.flatnonzero(rows)[0])} (0-based). Nothing "
        "downstream says so and nothing raises: measured, a NaN at index 5 of "
        "a 32-sample lst_deg loads clean and evaluates clean, and schema §6's "
        "C2 row records what follows -- the adjoint returns a finite, "
        "correctly-shaped, identically ZERO map. So a check on the OUTPUT "
        "sees no NaN, a gradient of zero everywhere, and a fit that looks "
        "converged. A sample time, an LST, a pointing and a self-rotation are "
        f"never legitimately NaN or infinite (check {check}).")


@register_axes("C1", "C2.time")
def _time_axis(facts: Axes) -> Iterable[Finding]:
    """C1 and C2's time leg: the axis the STORED dtype cannot carry.

    One function, two ids, because one call decides both:
    ``_refuse_a_time_axis_the_stored_dtype_cannot_carry`` names non-finite
    values first (that is C2) and then compares the representable spacing at
    the axis's peak against its smallest distinct gap (that is C1).
    ``Finding.check`` carries the BARE id either way -- ``"C1"`` or ``"C2"`` --
    and ``"C2.time"`` is a registry slot, so that ``_pointing_finite`` below
    can claim ``"C2.pointing"`` without either function silently displacing the
    other.

    **The routing is not a second implementation.**  Which of the two legs
    fired is asked with one ``np.isfinite`` on the values that already raised;
    the DECIDING -- both thresholds, both sentences, and the four load-bearing
    details that function's own docstring argues for (``np.spacing`` on the
    array's own scalar and never on ``float(...)``; non-finite named first;
    ``np.abs`` on both the peak and the gaps; inexact dtypes only) -- stays
    where it is.
    """
    times = facts.context.time
    values = _values(times)
    if values is None:
        return
    try:
        _refuse_a_time_axis_the_stored_dtype_cannot_carry(times)
    except StateValidationError as exc:
        check = "C1" if bool(np.all(np.isfinite(values))) else "C2"
        yield refuse(check, _time_where(facts.document),
                     f"{exc} {_TIME_TAIL} (check {check}).")


@register_axes("C2.pointing")
def _pointing_finite(facts: Axes) -> Iterable[Finding]:
    """C2's other three legs: ``pointing``, ``lst_deg`` and ``selfrot_deg``.

    ``observation.pointing`` and the two ``coords.extra`` entries are
    materialised by ``compile_pointing`` inside ``build_observation``, so all
    three are in hand here and none of them is decidable one phase earlier: a
    ``pointing.el_deg`` of NaN, an ``lst:`` read from a file, and an
    ``observation.extra.lst_deg`` written as a value node all arrive as
    resolved arrays and as nothing else.

    Each subject gets its own finding rather than one summary sentence, so a
    document with two bad axes sends the reader to two lines.
    """
    document = facts.document
    yield from _one("observation.pointing", "coords.pointing",
                    facts.observation.pointing)
    extra = facts.observation.extra or {}
    for key in _NAMED_EXTRA:
        yield from _one(_extra_where(document, key),
                        f'coords.extra["{key}"]', extra.get(key))


def _extra_where(document: Mapping[str, Any], key: str) -> str:
    """Which document key produced ``coords.extra[key]``.

    ``build_observation`` refuses a key written by both ``observation.extra``
    and ``observation.pointing`` -- "one producer per key, or the two silently
    disagree" -- so exactly one of the two answers is right, and asking the
    document which is cheaper than threading the producer through the payload.
    """
    observation = document.get("observation")
    extra = observation.get("extra") if isinstance(observation, Mapping) else None
    if isinstance(extra, Mapping) and key in extra:
        return f"observation.extra.{key}"
    return "observation.pointing"


def _one(where: str, subject: str, axis: Any) -> Iterable[Finding]:
    """C2 for one materialised axis, or nothing when it is absent or traced."""
    values = _values(axis)
    if values is None or values.size == 0:
        return
    if not np.issubdtype(values.dtype, np.inexact):
        # An integer axis represents integers exactly and has no non-finite
        # value to hold; `np.isfinite` on it is True everywhere.
        #
        # MEASURED EQUIVALENT MUTANT, kept anyway and said so rather than left
        # to be re-discovered: deleting this line changes no verdict, because
        # the `isfinite` below already answers True for every integer array,
        # and a mutation campaign found the deletion surviving the whole
        # suite. It stays for two reasons -- it says what the guard above is
        # FOR (`coords.extra` carries integer index vectors), and `np.isfinite`
        # raises TypeError rather than answering on a non-numeric dtype, which
        # is one widening of `_NAMED_EXTRA` away from being reachable.
        return
    if bool(np.all(np.isfinite(values))):
        return
    yield _non_finite("C2", where, subject, values)
