"""Pre-flight: what the document's own numbers say about each other.

Three checks, and two of them are the first warnings this layer has ever
emitted.  That is the point of the order they land in: ``ConfigWarning`` and
``Report.warnings`` are a channel, and a channel with no producer is what
check A41 is ABOUT -- the literal-shadows-a-symbol fact has been computed at
every value resolution since ``7a86c91`` (and unified across the array and
draw routes at ``b1d899d``), stored as ``ResolvedValue.modifiers["_shadowed"]``
(``arrays.py:41,43``, ``draws.py:163``), and read by nothing under ``src/``.

**A41 is decided here from the document's TEXT, and this module never touches
that stored modifier.**  "The stored modifier now has a reader" is not what
this ships: it is still consumed by nothing under ``src/``, and giving it one
belongs to the plan that has resolved values to read.  What this module does
is ask the same question of the text, through the same predicate
(:func:`~rheplicant.config.symbols.literal_shadowing_a_symbol`), one phase
earlier.

The boundary is exact.  A shape spec is always literal -- ``resolve_shape``
refuses anything that is not a list -- so what may be missing is the SCOPE.
``n_source`` is always text; ``n_time`` and ``n_freq`` are text only when the
grids declare an integer count.  Both or neither:
``literal_shadowing_a_symbol`` reports the first of ``n_time, n_freq,
n_source`` on a tie, so a half-known scope would name the wrong symbol on
every square grid.

**A42** is worded in schema §4.7.1 as "``from: simulation`` + ``twin: fit``
while ``model`` lights a stochastic node".  Measured, that over-fires:
``build_fit_twin`` (``twin.py:37-38``) returns the twin UNCHANGED when
``inference.twin:`` is absent, so ``twin: fit`` then simulates from the same
object ``twin: full`` does and nothing has been removed from the data.  The
condition implemented here adds the third fact -- the node is named in
``inference.twin.without``, or replaced by an operator that does not draw.

**A52**'s first half only.  Its second (``site.lat_deg`` required iff a
projector reads ``{from: site}``) describes a route v1 does not have: the
derivation registry is closed (``derive.py``) and has no ``site`` entry.  §6
records it as needing a design decision rather than an implementation, and
``test_the_derivation_registry_still_has_no_site_route`` is what makes the day
that changes a day someone looks.

**Two deviations from the task body, both forced by measurement** and both
recorded in the plan's Executor's notes:

* §3.2 (f) pins the shared stochastic predicate as ``stochastic_operator``.
  Task 11 bound it as ``_a30_stochastic(node_id, spec, table)``.  §0's rule is
  that the repository wins, so that is what is imported here and no second
  predicate is written -- which is the whole content of §3.2 (f).
* A52's reference leg reads the ``model:`` section as TEXT rather than through
  ``_nodes``.  Measured: ``_nodes`` is ``{}`` for a ``kind: pipeline`` model
  (it has no node registry), and a pipeline stage referencing a projector with
  no ``observation.pointing`` **builds today**.  ``node_specs`` drops only
  ``kind`` and ``acknowledge_double_count``, neither of which can hold a
  ``{ref:}``, so on a graph model the two walks are measurably identical and
  the text walk is a strict widening.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.findings import Finding, refuse, warn
from rheplicant.config.preflight import register
from rheplicant.config.preflight.document import _task3_where
from rheplicant.config.preflight.model import (
    _a30_stochastic,
    _lit,
    _t4_switch_order,
    stochastic_nodes,
)
from rheplicant.config.sections.model import operator_table
from rheplicant.config.symbols import ShapeScope, literal_shadowing_a_symbol

#: Grid forms that declare an axis LENGTH the document writes down.
#: ``linspace`` and ``arange`` carry it as ``num:``; ``list`` carries it as the
#: length of the list itself.  Deliberately short: every form admitted here is
#: a place a wrong ``n_freq`` could produce a FALSE warning, which is the only
#: way A41 can do harm.  ``modulo`` is the one form left out that could have
#: been in -- an integer ramp modulo a period is a degenerate frequency axis in
#: the same way ``{zeros: [8]}`` is, and both decline.
_A41_COUNTED_FORMS: tuple[str, ...] = ("linspace", "arange", "list")

#: Forms whose shape is the form's own value: ``{zeros: [16, 8]}``.
_A41_SHAPE_FORMS: tuple[str, ...] = ("zeros", "ones")

#: Forms whose shape is nested under ``shape:``.  Read for the reason
#: ``symbols.resolve_shape``'s docstring gives: the array forms and the draw
#: forms once diverged on exactly this report, and "a report that depends on
#: which constructor the writer reached for is worse than no report".
_A41_NESTED_SHAPE_FORMS: tuple[str, ...] = ("full", "normal", "uniform")

#: The keys ``inference.observed``'s grammar owns, so a sibling key is a
#: RECORD NAME rather than a form.  Mirrors ``sections/observed.py``'s own
#: ``_FORM_KEYS``; a test imports that module's frozenset and compares.
_A42_FORM_KEYS: frozenset[str] = frozenset({"from", "twin", "at", "realise",
                                            "file"})

#: A ``ref`` at or below this names a projector.  The trailing dot is
#: load-bearing: ``resources.projectors_backup.p`` shares every other
#: character.
_A52_PREFIX: str = "resources.projectors."


def _a41_axis_length(grid: Any) -> int | None:
    """``len(grid)`` from the text, or ``None`` when the document does not say.

    ``num`` must be a WRITTEN integer: it may also be a shape symbol
    (``resolve_extent``, ``arrays.py:92``), and measured, a freq grid written
    ``{linspace: {..., num: "n_time"}}`` resolves against a scope whose
    ``n_time`` is still 0 and builds an EMPTY frequency axis.  Reading it as
    an extent here would be guessing at a number the loader itself gets from
    somewhere this pass cannot see.

    The branch is on the FORM and not on the value's type: ``{linspace: [...]}``
    is a malformed linspace, not a list form, and reading its length would
    invent an axis out of a shape the value grammar refuses.
    """
    if not isinstance(grid, Mapping):
        return None
    for form in _A41_COUNTED_FORMS:
        if form not in grid:
            continue
        inner = grid[form]
        if form == "list":
            return len(inner) if isinstance(inner, (list, tuple)) else None
        if not isinstance(inner, Mapping):
            return None
        num = inner.get("num")
        return num if isinstance(num, int) and not isinstance(num, bool) else None
    return None


def _a41_scope(document: Mapping[str, Any]) -> ShapeScope | None:
    """The extents a shape symbol resolves against, from the text -- or None.

    ``None`` means "this document does not say", and every caller must then
    emit nothing.  Both grid lengths or neither, because
    :func:`~rheplicant.config.symbols.literal_shadowing_a_symbol` reports the
    FIRST of ``n_time, n_freq, n_source`` that matches: knowing one of the two
    would name the wrong symbol on every square grid, and a warning that names
    the wrong symbol is worse than none -- it is read as authoritative.

    ``n_source`` is ``len(order) or 1``, exactly ``ResolutionContext.
    shape_scope``'s arithmetic (``context.py:55-57``); the override beside it
    has no writer anywhere in ``src/`` (measured), so the two cannot disagree
    through that door.  The order comes from Task 4's
    :func:`~rheplicant.config.preflight.model._t4_switch_order`, which is the
    ONE reader of ``observation.switching`` under ``preflight/`` -- it knows
    the section's two grammars (an ingested run declares ``order:`` with no
    ``mode:``) and it answers ``None`` for a block this layer cannot read.
    That ``None`` propagates: falling back to 1 there would name ``n_source``
    about a cycle nobody read.

    An ingested run needs no clause of its own.  ``observation.from_file`` and
    ``observation.freq`` together are refused by ``build_observation`` ("the
    recording carries the frequency axis"), so an ingested document has
    neither axis block and the loop below declines on the first one.
    """
    observation = document.get("observation")
    if not isinstance(observation, Mapping):
        return None
    lengths: list[int] = []
    for axis in ("time", "freq"):
        block = observation.get(axis)
        if not isinstance(block, Mapping):
            return None
        length = _a41_axis_length(block.get("grid"))
        if length is None:
            return None
        lengths.append(length)
    order = _t4_switch_order(document)
    if order is None:
        return None
    return ShapeScope(n_time=lengths[0], n_freq=lengths[1],
                      n_source=len(order) or 1)


def _a41_shapes(node: Any, where: str) -> list[tuple[str, list]]:
    """``(document path, shape as written)`` for every shape under ``node``.

    The path is the VALUE NODE's, not the shape key's: ``resources.arrays.d``
    rather than ``resources.arrays.d.normal.shape``, because the former is
    what the reader edits and the latter is a spelling detail of whichever
    constructor they reached for.

    ``(list, tuple)`` and not ``list``, because that is what
    ``symbols.resolve_shape`` accepts.  A document built in Python rather than
    read from YAML can hold a tuple, and a narrower reading here would be
    silent about a shape the resolver goes on to report.
    """
    found: list[tuple[str, list]] = []
    if isinstance(node, Mapping):
        for form in _A41_SHAPE_FORMS:
            shape = node.get(form)
            if isinstance(shape, (list, tuple)):
                found.append((where, list(shape)))
        for form in _A41_NESTED_SHAPE_FORMS:
            inner = node.get(form)
            if isinstance(inner, Mapping) and isinstance(inner.get("shape"),
                                                         (list, tuple)):
                found.append((where, list(inner["shape"])))
        for key, value in node.items():
            if isinstance(key, str):
                found.extend(_a41_shapes(value, f"{where}.{key}"))
    elif isinstance(node, (list, tuple)):
        for index, entry in enumerate(node):
            found.extend(_a41_shapes(entry, f"{where}[{index}]"))
    return found


def _a42_records(section: Any) -> tuple[tuple[str, Mapping], ...]:
    """``inference.observed`` -> ``((name, spec), ...)``.

    The discrimination is ``sections/observed.build_observed``'s, mirrored: a
    top-level ``from:`` or ``file:`` is ONE record called ``primary``, and
    anything else is a mapping of names.  A test pins the mirror by importing
    that module's own ``_FORM_KEYS``.

    A name or a spec that section REFUSES is dropped rather than guessed at:
    ``build_observed`` says what is wrong with it in its own words, and A42 --
    a warning about the data a legal document would produce -- has nothing to
    add about one that is not legal.
    """
    if not isinstance(section, Mapping):
        return ()
    if "from" in section or "file" in section:
        return (("primary", section),)
    return tuple((name, spec) for name, spec in section.items()
                 if isinstance(name, str) and name not in _A42_FORM_KEYS
                 and isinstance(spec, Mapping))


# NO SECOND PREDICATE HERE.  §3.2 (f) binds ONE name for "does this node's
# operator class declare `key` in requires" and Task 11 owns it in
# `preflight/model.py` -- shipped as `_a30_stochastic(node_id, spec, table)`
# rather than under §3.2 (f)'s proposed spelling.  A private predicate of this
# module's own is the collision §3.2 (f) predicted, and the two would
# measurably disagree: on `{from: ...}` at `noise` a text-shaped guess says
# "still draws" and `_a30_stochastic` stands down.  `stochastic_nodes` answers
# the `without:` leg; the `replace:` leg needs the per-spec form, because a
# `inference.twin.replace` spec is not a model node and `stochastic_nodes`
# cannot express it -- dropping it would drop §3.2 (h)2's whole correction.


def _a42_removed(document: Mapping[str, Any]) -> tuple[str, ...]:
    """The stochastic nodes ``model:`` lights that the FIT twin does not carry.

    Empty when ``inference.twin:`` is absent, and that is the whole correction
    to §4.7.1's wording: measured, ``twin: fit`` without that section returns
    the model twin itself (``twin.py:37-38``), so nothing has been removed and
    there is nothing to warn about.

    ``_a30_stochastic`` answers ``None`` both for "this replacement does not
    draw" and for "the text cannot say", and this reads either as a removal.
    Measured at ``d229d27``, that costs nothing on any document the package
    builds: at a stochastic node the only ``replace:`` spelling that both
    stops the draw and BUILDS is the ``python:`` hatch, which resolves to a
    class and is decided exactly.  Every spelling that lands in the "cannot
    say" branch is a document ``build_fit_twin`` refuses in its own words --
    ``from:`` ("not a route this node offers"), ``compose:`` and a foreign
    ``type:`` ("not registered at this node") -- so the warning is never the
    only thing the reader is told, and a second predicate to separate the two
    answers is the collision §3.2 (f) exists to prevent.
    """
    inference = document.get("inference")
    if not isinstance(inference, Mapping):
        return ()
    spec = inference.get("twin")
    if not isinstance(spec, Mapping):
        return ()
    drawing = stochastic_nodes(document)
    table = operator_table()
    removed: list[str] = []
    without = spec.get("without")
    if isinstance(without, (list, tuple)):
        removed.extend(node_id for node_id in without
                       if isinstance(node_id, str) and node_id in drawing)
    replace = spec.get("replace")
    if isinstance(replace, Mapping):
        removed.extend(node_id for node_id, replacement in replace.items()
                       if isinstance(node_id, str) and node_id in drawing
                       and _a30_stochastic(node_id, replacement, table) is None)
    return tuple(dict.fromkeys(removed))


def _a52_projector_refs(node: Any, where: str) -> list[str]:
    """Every document path under ``node`` holding ``{ref: resources.projectors.…}``."""
    found: list[str] = []
    if isinstance(node, Mapping):
        target = node.get("ref")
        if isinstance(target, str) and target.startswith(_A52_PREFIX):
            found.append(where)
        for key, value in node.items():
            if isinstance(key, str):
                found.extend(_a52_projector_refs(value, f"{where}.{key}"))
    elif isinstance(node, (list, tuple)):
        for index, entry in enumerate(node):
            found.extend(_a52_projector_refs(entry, f"{where}[{index}]"))
    return found


@register("A41")
def _shadowed_literals(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A41: a literal integer in a shape that equals one of this run's extents.

    A warning, not a refusal, and the reason is ``arrays.py:6-7``'s: a literal
    8 may genuinely be 8, and what it cannot be is TIED to the grid.
    ``variants:`` is not walked -- an unselected variant's shapes belong to a
    run this document is not describing, and unselected-variant text is Task
    3's check.
    """
    findings: list[Finding] = []
    scope = _a41_scope(document)
    if scope is None:
        return findings
    declared_by = {"n_time": "observation.time.grid",
                   "n_freq": "observation.freq.grid",
                   "n_source": "observation.switching.order"}
    for section, block in document.items():
        if not isinstance(section, str) or section == "variants":
            continue
        for path, shape in _a41_shapes(block, section):
            for index, entry in enumerate(shape):
                symbol = literal_shadowing_a_symbol(entry, scope)
                if symbol is None:
                    continue
                findings.append(warn(
                    "A41", _task3_where(path),
                    f"{path}: the literal {entry} at shape position {index} "
                    f"is this run's {symbol}, which {declared_by[symbol]} "
                    f"declares. Write '{symbol}' there instead -- a copied "
                    f"extent stays right until the grid moves, and then it is "
                    f"a finite, correctly shaped array of the wrong length, "
                    f"which no shape check and no finite check can see "
                    f"(check A41)."))
    return findings


@register("A42")
def _simulated_fit_twin(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A42: data simulated through a twin the randomness was taken out of.

    ``spec.get("twin", "full")`` is ``sections/observed.py:132``'s own line,
    character for character, so the two cannot drift apart about what an
    absent key means.  Written that way rather than as ``spec.get("twin")``,
    which no test can tell from it -- both answer "not fit" -- because a
    mirror is worth having even where the mirror image is the same.
    """
    findings: list[Finding] = []
    inference = document.get("inference")
    if not isinstance(inference, Mapping):
        return findings
    removed = _a42_removed(document)
    if not removed:
        return findings
    for name, spec in _a42_records(inference.get("observed")):
        if spec.get("from") != "simulation" or spec.get("twin", "full") != "fit":
            continue
        path = f"inference.observed.{name}"
        findings.append(warn(
            "A42", _task3_where(path),
            f"{path}: from: simulation with twin: fit simulates this "
            f"observation through the FIT twin, and inference.twin: takes "
            f"{list(removed)} out of that twin -- so the data carries no "
            f"realisation of {list(removed)} while the likelihood is asked to "
            f"account for it. examples/radio_digital_twin.py puts the noise "
            f"in the DATA and keeps it out of the fit twin: write twin: full "
            f"(the default) to simulate through the model twin, or realise: "
            f"to put the scatter back where the likelihood's own sigma can "
            f"see it (check A42)."))
    return findings


@register("A52")
def _pointing_none(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A52, first half: a projector in the model and no pointing to use it.

    Two triggers, kept separate because §6 states two conditions and each has
    a document of its own: ``observed_astro_sky`` lit, and a projector
    referenced anywhere in ``model:``.  On the document that carries both --
    which is every ``observed_astro_sky`` run that builds, because an inline
    projector is refused ("an object field takes a declared resource") -- both
    are emitted: the node leg names the fix and the reference leg names the
    line, and ``raise_if_refused`` quotes the first and counts the rest.

    The reference leg reads ``model:`` as TEXT rather than through
    :func:`~rheplicant.config.preflight.model._nodes`, which is ``{}`` for a
    ``kind: pipeline`` model -- and measured, a pipeline stage referencing a
    projector with no pointing builds today.  ``node_specs`` drops only
    ``kind`` and ``acknowledge_double_count``, neither of which can hold a
    ``{ref:}``, so the two walks agree on every graph model.
    """
    findings: list[Finding] = []
    observation = document.get("observation")
    if not isinstance(observation, Mapping):
        return findings
    spec = observation.get("pointing")
    if spec is not None and not isinstance(spec, Mapping):
        return findings
    if (spec or {}).get("mode", "none") != "none":
        return findings
    if "observed_astro_sky" in _lit(document):
        findings.append(refuse(
            "A52", "observation.pointing",
            "model lights observed_astro_sky, which sees the sky through a "
            "projector, and observation.pointing is mode: none -- which is "
            "the default when the section is absent, and the statement that "
            "this run has no pointing at all. Declare observation.pointing: "
            "{mode: drift|tracked|baked}, or drop the node (check A52)."))
    for path in _a52_projector_refs(document.get("model"), "model"):
        findings.append(refuse(
            "A52", _task3_where(path),
            f"{path} references a projector while observation.pointing is "
            "mode: none -- which is the default when the section is absent, "
            "and the statement that this run has no pointing at all. A "
            "projector turns a sky into what THIS observation saw, and mode: "
            "none says there is no pointing for it to turn it into. Declare "
            "observation.pointing: {mode: drift|tracked|baked}, or remove the "
            "reference (check A52)."))
    return findings
