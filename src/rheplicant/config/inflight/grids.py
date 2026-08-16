"""C3, A13's grid legs and C8 -- three rules the RESOLVED grids decide.

Split from ``inflight/axes.py`` up front rather than when it got long (§0.3
E.1 pre-names both modules): ``axes.py`` is about what the axes THEMSELVES
hold, and this module is about what other parts of the document say about
them -- a projector's band limit, a tone's width, a chunk count.

**Why the axes slot.**  Every rule here compares a literal int or float in the
document against ``n_time`` or the frequency grid, and neither exists until
``build_observation`` has run.  None of them needs a built resource, so all
three are decided before ``build_resources`` reads the beam.

**What each one costs today, measured in this worktree.**

* **C3.**  ``limtod_jax.driftscan._validate_nyquist`` enforces ``2*lmax <
  n_time`` and is reached only from ``DriftScanOperator.__check_init__``,
  which ``rheplicant.radio.sky.driftscan`` constructs inside ``forward``.
  Measured: a document with ``lmax: 8``, ``uniform_sampling: true`` and a
  16-sample time grid **loads clean**; the refusal arrives at the first
  forward pass, as a bare ``ValueError`` from a third-party package -- not a
  ``DirtError``, so ``except DirtError`` misses it, and not a ``ConfigError``,
  so ``pytest.raises(ConfigError)`` cannot catch it.
* **A13's width legs.**  ``CWCalibrationOperator._validate_over_the_run``
  raises them, from ``__call__``.  Two consequences: on a document with no
  ``inference.observed: {from: simulation}`` they never run during a load at
  all, and **under jit none of them run ever** -- the method's first act is
  ``np.asarray(freq)`` inside a ``try`` that returns on
  ``TracerArrayConversionError``.  That early return is the concreteness
  escape the class documents, and it means the bounds are a trace-time
  courtesy rather than a guarantee.
* **C8.**  Both divisibility rules are ``jnp.reshape``'s, reached by running
  the twin.  Measured: the ``n_chunk: 4`` + ``n_days: 8`` document on 16
  samples IS refused by ``load_document`` today -- because the standard
  fixture carries ``inference.observed: {from: simulation}`` and
  ``build_observed`` does ``prediction = bound(state).data``, **a real
  forward pass inside the load**.  Remove ``observed:`` and the same document
  is accepted.  So what C8 buys is not "a rule nobody checks"; it is the same
  refusal one phase earlier and without the forward pass.

**One binding, and where this module deliberately does not have one.**  C3 and
C8 restate no sentence: both are inventions at this layer, and the packages'
own messages stay where they are.  A13's two width legs DO restate arithmetic
that ``radio/instrument/calibration.py`` also performs -- the constants are
imported so the NUMBERS have one binding, but the comparison is written twice,
because the shipped one is a method on a constructed operator and
``calibration.py`` is outside this task's Files list.  That is recorded rather
than hidden: the divergence to watch is the CEILING, which is
``max(MAX_WIDTH_IN_BAND_FRACTION * band, MIN_CEILING_IN_CHANNELS * spacing)``
in the code and ``0.25 * band`` in schema §6's A13 row.  The second term binds
only on a COARSE grid -- ``0.25 * (n_freq - 1) >= 2`` from nine channels up --
so on any real spectrometer band the two **coincide exactly** and a test built
there cannot tell them apart.  Measured: on 70.000-70.001 MHz over 4 channels
they are **250 Hz** (schema) and **672 Hz** (code) in float32; on the worked
document's 8-channel 60-85 MHz band they are 6.25 MHz and 7.14 MHz, a 14 %
gap.  (The plan says the two coincide on RHINO's own band; on the worked
document's eight channels they do not, and the narrow band is used anyway
because a factor of 2.7 discriminates where 1.14 barely does.)

**Routes.**  ``Axes.document`` is the variant-applied mapping, so there is no
``variants:`` twin here.  What each check walks besides ``model:`` is stated
on the check.
"""

from __future__ import annotations

import dataclasses
import numbers
from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.inflight import Axes, register_axes
from rheplicant.config.resources import resolved_specs

#: The prefix ``resolved_specs`` gives a projector.  Its keys are the dotted
#: strings ``build_resources`` uses, so this is the layer's own spelling and
#: not a second one.
_PROJECTORS = "resources.projectors."

#: The tail C3 appends.  Bound once, and interpolated rather than restated in
#: each branch.
_C3_TAIL = (
    "limtod_jax enforces this itself -- the FFT synthesis weights bin 0 and "
    "the Nyquist bin by 1 while the m-mode expansion needs 2 for every m >= "
    "1, so m = lmax has to stay off the Nyquist bin -- but it enforces it "
    "from DriftScanOperator.__check_init__, which is constructed inside "
    "forward(). Measured, such a document LOADS today and the refusal arrives "
    "at the first forward pass as a bare ValueError from a third-party "
    "package: not a DirtError, so `except DirtError` misses it, and not a "
    "ConfigError, so this layer never gets to say it. Raise the number of "
    "samples on observation.time.grid, lower lmax, or drop uniform_sampling: "
    "and take the exact direct sum"
)

#: The tail both of C8's clauses append.
_C8_TAIL = (
    "Both counts are static ints and n_time is len(context.time), so this is "
    "decided here, before build_resources reads the beam. Today it is decided "
    "by jnp.reshape refusing a size that does not fit, inside the twin: "
    "measured, the fixture's inference.observed: {from: simulation} makes "
    "build_observed run a real forward pass inside load_document "
    "(prediction = bound(state).data), so the refusal does arrive -- at that "
    "cost -- and a document without observed: is accepted outright"
)

#: The tail both of A13's width legs append.
_A13_TAIL = (
    "The bound is CWCalibrationOperator's own and is checked in "
    "_validate_over_the_run, which runs from __call__ -- so on a document "
    "that never simulates it is never reached, and under jit it is never "
    "reached at all: that method's first act is np.asarray(freq) inside a try "
    "that returns on TracerArrayConversionError. Here it is arithmetic on the "
    "resolved frequency grid and two static floats"
)


def _n_time(facts: Axes) -> int:
    return int(facts.context.time.shape[0])


def _routes(document: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """``model:``'s node specs and ``inference.twin.replace``'s, with prefixes.

    §0.3 E.10's global ruling: ``inference.twin.replace.<node>`` reaches
    ``build_node_operator`` down the same path ``model.<node>`` does and is
    outside ``preflight/model.py::_nodes``, so a check that walks only
    ``model:`` guards one route of two.  Every check in this module walks
    both, and each says so in its own docstring.
    """
    from rheplicant.config.preflight.model import _nodes

    routes: list[tuple[str, Mapping[str, Any]]] = [("model", _nodes(document))]
    inference = document.get("inference")
    twin = inference.get("twin") if isinstance(inference, Mapping) else None
    replace = twin.get("replace") if isinstance(twin, Mapping) else None
    if isinstance(replace, Mapping):
        routes.append(("inference.twin.replace", replace))
    return routes


def _static_number(node: Any, context: Any) -> float | None:
    """The number a STATIC field's value node carries, or ``None`` to stand down.

    **Only the scalar forms are resolved**, and that is a scope decision
    rather than a convenience.  ``resolve_value`` on a ``{file: ...}`` node
    opens a file and on a ``{ref: ...}`` node reaches into resources that do
    not exist yet at this slot; both are outside ``inflight/``'s boundary.  A
    bare number, a ``'<number> <unit>'`` shorthand and ``{value: N, unit: U}``
    are resolved with the layer's own :func:`resolve_value`, so the unit
    conversion has one binding and is not restated here.

    Standing down rather than guessing is safe in both directions: the fields
    this is used for are ``eqx.field(static=True)``, and ``config/delivery.py``
    refuses a non-``numbers.Real`` for one by name -- measured,
    ``line_width: {from: channel_spacing}`` is refused with *"Field
    'line_width' is a static float and the value is ArrayImpl"*.  So a node
    this declines to read is a node the layer refuses anyway, in its own
    words.
    """
    from rheplicant.config.values import VALUE_FORMS, resolve_value

    if isinstance(node, bool):
        return None
    if isinstance(node, Mapping):
        if set(node) & set(VALUE_FORMS) != {"value"}:
            return None
    elif not isinstance(node, (int, float, str)):
        return None
    try:
        resolved = resolve_value(node, context)
    except ConfigError:
        return None
    value = resolved.value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None
    return float(value)


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------


@register_axes("C3")
def _nyquist(facts: Axes) -> Iterable[Finding]:
    """C3's sampling-theorem leg: ``2*lmax < n_time`` under ``uniform_sampling``.

    **This is the nyquist leg ONLY.**  C3's other leg -- that the LST grid is
    a uniform full sidereal turn with the endpoint excluded -- is
    ``limtod_jax.check_uniform_grid``'s, and importing ``limtod_jax`` inside
    this pass costs 0.011 s against a 0.01 s budget for the whole slot AND
    puts an optional dependency into every process that reads a config, which
    is §0's import invariant.  It stays a declared false negative, recorded in
    the plan's §7 with that reason.

    ``lmax`` is a literal int in the document: ``build_projector`` does
    ``int(spec["lmax"])`` with no ``{ref:}`` route, so there is nothing to
    resolve and nothing to read.  ``uniform_sampling`` is a driftscan key
    only -- ``_ENGINE_KEYS['general_pointing']`` does not carry it -- so the
    ``general_pointing`` twin has no FFT path and no Nyquist bin, which is
    why this stands down on it rather than guarding it.

    ``resolved_specs`` applies ``extends:`` and is TOTAL, so a projector that
    inherits ``lmax`` from a parent entry is covered and a malformed entry is
    dropped rather than raised on.
    """
    n_time = _n_time(facts)
    for dotted, spec in resolved_specs(facts.document.get("resources")).items():
        if not dotted.startswith(_PROJECTORS):
            continue
        if spec.get("engine") != "driftscan" or not spec.get("uniform_sampling"):
            continue
        lmax = spec.get("lmax")
        if isinstance(lmax, bool) or not isinstance(lmax, int):
            # `build_projector`'s own `int(spec["lmax"])` says what a
            # non-integer lmax is, and a missing one earns "engine: driftscan
            # requires lmax:". Pre-empting either with an arithmetic
            # complaint would be the more specific refusal losing to the less.
            continue
        if 2 * lmax < n_time:
            continue
        yield refuse(
            "C3", dotted,
            f"{dotted}: uniform_sampling: true needs 2*lmax < n_time and this "
            f"run has lmax={lmax} against n_time={n_time} (2*lmax = "
            f"{2 * lmax}). {_C3_TAIL} (check C3).")


# ---------------------------------------------------------------------------
# A13's grid legs
# ---------------------------------------------------------------------------


def _tone_default(field: str) -> Any:
    """One ``CWCalibrationOperator`` field's default, off the class.

    ``lineshape`` decides WHICH floor applies -- ``MIN_WIDTH_IN_CHANNELS`` is
    a dict keyed by it, ``{'sinc2': 1.0, 'gaussian': 0.25}`` -- so a check
    that assumed ``sinc2`` would refuse a legal gaussian line four times too
    narrow to complain about.  Read off the class rather than restated, for
    the same reason the constants are imported.
    """
    from rheplicant.radio.instrument.calibration import CWCalibrationOperator

    for one in dataclasses.fields(CWCalibrationOperator):
        if one.name == field:
            return one.default
    return None


def _is_tone(entry: Any) -> bool:
    """Is this spec the shipped ``CWCalibrationOperator``?

    **The class, not the token** (§0.3 E.9's ruling for A45, and the same hole
    it closes): ``python: 'rheplicant.radio.instrument.calibration:
    CWCalibrationOperator'`` and ``type: CWCalibrationOperator`` are two
    spellings of one class object, and a check keyed on the literal ``type:``
    misses the ``python:`` one that 3A's tests already exercise.
    """
    from rheplicant.config.preflight.model import _t5_radio_class
    from rheplicant.config.sections.model import operator_table
    from rheplicant.radio.instrument.calibration import CWCalibrationOperator

    if not isinstance(entry, Mapping):
        return False
    if "python" in entry:
        return _t5_radio_class(entry) is CWCalibrationOperator
    classes = operator_table().get("cw_tone", ())
    if entry.get("type") is not None:
        return any(cls is CWCalibrationOperator and cls.__name__ == entry["type"]
                   for cls in classes)
    return classes == (CWCalibrationOperator,)


def _tone_entries(document: Mapping[str, Any]) -> Iterable[tuple[str, Mapping]]:
    """``(document path, spec)`` per operator the ``cw_tone`` node declares.

    ``_t4_entries`` expands a ``compose:`` block into its stages, so a tone
    composed with a tap is reached at ``model.cw_tone.stages[0]`` rather than
    being read off the composing mapping, which carries no ``line_width`` at
    all.  ``many=False`` because ``cw_tone`` is not a ``many`` node, and a
    ``CWCalibrationOperator`` written under a different key is A5's refusal,
    not this one's.
    """
    from rheplicant.config.preflight.model import _t4_entries

    for prefix, specs in _routes(document):
        spec = specs.get("cw_tone")
        if spec is None:
            continue
        for path, entry in _t4_entries("cw_tone", spec, many=False):
            if _is_tone(entry):
                yield f"{prefix}.{path}", entry


@register_axes("A13.grid")
def _tone_on_the_grid(facts: Axes) -> Iterable[Finding]:
    """A13's grid legs: the tone's width and its centre, against this band.

    A13's OTHER leg -- ``line_width:`` being present at all -- is decidable
    from text and belongs to the pre-flight pass; this one needs
    ``median(|diff(freq)|)`` and the band's extent, which need the resolved
    grid.

    Walks ``model.cw_tone`` **and** ``inference.twin.replace.cw_tone`` (§0.3
    E.10).  The messages here are this module's own and carry no hardcoded
    section, so unlike A13's presence message they say the right path on both
    routes.

    ``median`` and not ``mean``, via ``config/derive.py::_median_gap``, which
    already takes ``abs`` BEFORE the median: a descending grid is legal
    everywhere in this package and its diffs are all negative, so the unsigned
    version returns a negative spacing that compares below every floor and
    reads as comfortably fine.

    **A descending grid is legal on the TIME axis too**, and the drifted
    centre reads it with the operator's own arithmetic -- ``times - times[0]``
    rather than an extent -- for the same reason from the other side: an
    always-positive extent drifts the tone the wrong way on a descending axis
    and the two disagree on a VERDICT, not on a message.  Both twins are
    tested (``test_a_descending_grid_is_not_waved_through`` for frequency,
    ``test_a_descending_TIME_grid_drifts_the_tone_the_way_the_operator_does``
    for time).
    """
    from rheplicant.config.derive import _median_gap
    from rheplicant.radio.instrument.calibration import (
        MAX_WIDTH_IN_BAND_FRACTION,
        MIN_CEILING_IN_CHANNELS,
        MIN_WIDTH_IN_CHANNELS,
        WIDTH_FLOOR_RTOL,
    )

    # The entries FIRST, so a document with no tone -- which is almost every
    # document -- pays nothing: neither the median nor `operator_table()`.
    entries = list(_tone_entries(facts.document))
    if not entries:
        return
    freq = facts.context.freq
    try:
        spacing = float(_median_gap(freq, name="channel_spacing",
                                    axis_name="frequency"))
    except ConfigError:
        return  # a single-channel band has no spacing; nothing to measure
    low, high = float(freq.min()), float(freq.max())
    times = facts.context.time
    # `times - times[0]` and NOT `times.max() - times.min()`, which is a
    # MEASURED false negative rather than a stylistic difference. The shipped
    # rule this restates is `_validate_over_the_run`'s `elapsed = times -
    # times[0]; first, last = elapsed.min(), elapsed.max()`, and a descending
    # time grid -- legal everywhere in this package -- makes `first` NEGATIVE.
    # An always-positive extent drifts the tone the other way: measured, a
    # 30 -> 0 s grid with `tone_freq: 62 MHz` and `drift_rate: 5e5` over a
    # 60-85 MHz band passed this check CLEAN while the operator refused it.
    offsets = times - times[0]
    first, last = float(offsets.min()), float(offsets.max())

    for where, entry in entries:
        lineshape = entry.get("lineshape", _tone_default("lineshape"))
        width = _static_number(entry.get("line_width"), facts.context)
        if width is not None and lineshape in MIN_WIDTH_IN_CHANNELS:
            floor = MIN_WIDTH_IN_CHANNELS[lineshape] * spacing
            ceiling = max(MAX_WIDTH_IN_BAND_FRACTION * (high - low),
                          MIN_CEILING_IN_CHANNELS * spacing)
            if width < floor * (1.0 - WIDTH_FLOOR_RTOL):
                yield refuse(
                    "A13", where,
                    f"{where}.line_width: {width:.6g} Hz is narrower than the "
                    f"channel response this {lineshape!r} grid can carry "
                    f"({MIN_WIDTH_IN_CHANNELS[lineshape]:g} x the {spacing:.6g} "
                    f"Hz median channel spacing = {floor:.6g} Hz). The sampled "
                    "channels land on the lineshape's own nulls, or overflow "
                    "its exponent, and the normalisation then divides by float "
                    f"noise. {_A13_TAIL} (check A13).")
            elif width > ceiling:
                yield refuse(
                    "A13", where,
                    f"{where}.line_width: {width:.6g} Hz is wider than a LINE "
                    f"on this band -- the limit is {ceiling:.6g} Hz, the larger "
                    f"of {MAX_WIDTH_IN_BAND_FRACTION:g} x the {high - low:.6g} "
                    f"Hz band and {MIN_CEILING_IN_CHANNELS:g} x the "
                    f"{spacing:.6g} Hz channel spacing. Note the second term: "
                    "on a narrow or coarse band it is the operative one, and a "
                    "reading of schema §6's A13 row that stops at 0.25 x the "
                    "band is a different number. Nothing would raise -- the "
                    "weights still normalise -- but what they model is a "
                    "PEDESTAL over the whole band, every channel sits above "
                    "protect_floor of the peak, and the RFI flagger is "
                    f"switched off for the entire run. {_A13_TAIL} (check A13).")
        centre = _static_number(entry.get("tone_freq"), facts.context)
        drift = _static_number(entry.get("drift_rate"), facts.context)
        if centre is None:
            continue
        drift = 0.0 if drift is None else drift
        centres = (centre + drift * first, centre + drift * last)
        # `low <= min(centres)` and not `<`: the shipped comparison is
        # `not low <= min(centres) or not max(centres) <= high`, so a centre
        # sitting exactly ON the first or last channel is IN band for the
        # operator, and a strict reading here would refuse a document the
        # thing it restates accepts.
        if low <= min(centres) and max(centres) <= high:
            continue
        moving = (
            "" if drift == 0.0 else
            f", drifting at {drift:.6g} Hz/s over the run's {last - first:.6g} s")
        yield refuse(
            "A13", where,
            f"{where}.tone_freq: the tone centre spans [{min(centres):.6g}, "
            f"{max(centres):.6g}] Hz{moving}, outside this run's observed band "
            f"[{low:.6g}, {high:.6g}] Hz. A centre that starts in band and "
            "DRIFTS out of it is the case a check at the first sample alone "
            "passes, which is why the run's extent is read here rather than "
            "t_0. The lineshape is still evaluated and still normalised, so "
            "the run models a bright feature spread over channels the tone is "
            f"nowhere near. {_A13_TAIL} (check A13).")


# ---------------------------------------------------------------------------
# C8
# ---------------------------------------------------------------------------


def _static_int(spec: Any, key: str) -> int | None:
    """A static int field's literal, or ``None``.

    ``n_chunk`` and ``n_days`` are both ``eqx.field(static=True)`` ints, and
    ``config/delivery.py::_as_static_int`` refuses a bool for one by name --
    "Python's isinstance(True, int) is True, so this passes the operator's own
    guard and gives n_chunk = 1". This declines a bool for the same reason and
    leaves the sentence to the module that owns it.

    **The ``< 1`` clause is not about nonsense counts; it keeps ``%`` away
    from a division by zero.**  ``n_chunk: 0`` and ``n_days: 0`` are both
    documents a user can write, and the two callers below do ``n_time %
    n_chunk`` and ``after % n_days`` unguarded.  Measured with the clause
    weakened to ``value < 0``: ``model.averaging: {n_chunk: 0}`` makes the
    whole pass raise *"in-flight check 'C8' RAISED ZeroDivisionError: integer
    modulo by zero"*, which is not a wrong finding but the loss of **every**
    finding on that document.  Pinned by
    ``test_a_zero_count_is_declined_rather_than_divided_by``.

    **The bool clause is a MEASURED EQUIVALENT MUTANT and is kept anyway**,
    which is worth a sentence rather than a silent line.  Deleting it changes
    no verdict this function can reach: ``True`` is 1, 1 divides every count,
    and ``False`` is 0, which the ``< 1`` clause already declines -- so a bool
    never reaches a message either way, and a mutation campaign found the
    deletion surviving the whole suite.  It stays because the next clause
    added here (``n_days >= 2`` is the obvious one, and it is
    ``SiderealFilter``'s today) would read ``True`` as a legal count of days,
    and because a reader comparing this with ``delivery.py`` should see the
    two agreeing rather than wonder.
    """
    if not isinstance(spec, Mapping):
        return None
    value = spec.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _chunking(document: Mapping[str, Any]) -> tuple[int, str] | None:
    """The effective ``(n_chunk, where)``, or ``None`` when nothing averages.

    ``inference.twin.replace.averaging`` OVERRIDES ``model.averaging`` for the
    fit twin, so the later route wins here.  A document that declares both
    with DIFFERENT ``n_chunk`` therefore has two twins with two answers, and
    only the fit twin's is checked; that is a declared false negative rather
    than an oversight, and it is recorded in the plan's §7.
    """
    found = None
    for prefix, specs in _routes(document):
        n_chunk = _static_int(specs.get("averaging"), "n_chunk")
        if n_chunk is not None:
            found = (n_chunk, f"{prefix}.averaging")
    return found


def _filters(document: Mapping[str, Any]) -> list[tuple[str, Mapping]]:
    """``(document path, entry)`` for the filter chain that actually runs.

    ``model.filters`` is a **chain** -- a ``many`` node -- so the path carries
    the index (``model.filters[1]``) and every element is walked.  A three
    filter chain with the second one wrong sends the reader to the line to
    edit rather than to the node.  ``inference.twin.replace.filters``
    replaces the chain wholesale when it is present, as for ``averaging``.
    """
    from rheplicant.config.preflight.model import _t4_entries

    found: list[tuple[str, Mapping]] = []
    for prefix, specs in _routes(document):
        spec = specs.get("filters")
        if spec is None:
            continue
        found = [(f"{prefix}.{path}", entry)
                 for path, entry in _t4_entries("filters", spec, many=True)
                 if isinstance(entry, Mapping)]
    return found


@register_axes("C8")
def _divisible(facts: Axes) -> Iterable[Finding]:
    """C8: ``n_time % n_chunk`` and ``n_time % n_days``, both against the truth.

    **The trap, and it is what a naive implementation gets wrong.**
    ``averaging`` runs BEFORE ``filters`` -- ``RADIO_GRAPH``'s processing
    segment is ``snapshot, flagging, averaging, apply_cal, filters`` -- so a
    filter behind an ``averaging: {n_chunk: 4}`` is handed ``n_time //
    n_chunk`` samples, not ``n_time``.  Measured on the worked document:
    ``n_chunk: 4`` and ``n_days: 8`` over 16 samples fails with *"n_time=4 is
    not divisible by n_days=8"* while ``16 % 8 == 0``.  A check that tested
    both clauses against the DECLARED count would pass that document.

    ``n_chunk == 1`` is legal -- ``BackendOperator`` leaves the time axis
    alone -- and divides everything, so it never reaches a message and is not
    folded into one.

    **What this does NOT decide**, each left to the module that owns it:
    ``n_days >= 2`` (``SiderealFilter.__check_init__``, pure text and a
    Task-4-shaped row), and ``0 <= low < high <= 0.5`` for the Fourier band
    filter (schema §6's C8 row names it; it is not a divisibility rule and
    needs no grid).
    """
    n_time = _n_time(facts)
    chunking = _chunking(facts.document)
    after = n_time
    if chunking is not None:
        n_chunk, where = chunking
        if n_time % n_chunk:
            yield refuse(
                "C8", where,
                f"{where}: n_chunk={n_chunk} does not divide this run's "
                f"{n_time} time samples ({n_time} % {n_chunk} = "
                f"{n_time % n_chunk}). BackendOperator reshapes (n_time, ...) "
                f"into ({n_time} // {n_chunk}, {n_chunk}, ...) and there is no "
                f"such shape. {_C8_TAIL} (check C8).")
            # The filter clause STANDS DOWN: how many samples a filter is
            # handed depends on what this document's n_chunk becomes once it
            # is fixed, and a second sentence computed from a count that is
            # about to change is advice that may be wrong.
            return
        after = n_time // n_chunk
    for where, entry in _filters(facts.document):
        n_days = _static_int(entry, "n_days")
        if n_days is None or after % n_days == 0:
            continue
        averaged = "" if after == n_time else (
            f"The run declares {n_time} samples, but averaging runs BEFORE "
            f"filters -- RADIO_GRAPH's processing segment is snapshot, "
            f"flagging, averaging, apply_cal, filters -- so a chain behind an "
            f"averaging of n_chunk={n_time // after} is handed {after} of "
            f"them. Checking n_days against the {n_time} the document declares "
            f"is the naive reading, and it accepts this document. ")
        yield refuse(
            "C8", where,
            f"{where}: n_days={n_days} does not divide the {after} time "
            f"sample(s) this filter is handed ({after} % {n_days} = "
            f"{after % n_days}). {averaged}SiderealFilter reshapes the time "
            f"axis into ({n_days}, n_lst, ...) to fold the days together and "
            f"there is no such shape. {_C8_TAIL} (check C8).")
