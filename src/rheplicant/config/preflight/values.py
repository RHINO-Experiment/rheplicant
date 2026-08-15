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
(``symbols.py:219``) refuses anything that is not a list or a tuple -- so what
may be missing is the SCOPE.
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

* ``task-11.md`` and ``task-12.md`` both write ``stochastic_operator`` for the
  per-spec stochastic predicate.  **§3.2 (f) does not**: it pins
  ``stochastic_nodes`` and nothing else, and Task 11 shipped that name
  verbatim, binding the per-spec form privately as ``_a30_stochastic(node_id,
  spec, table)``.  §3.1/§3.2 are authoritative over any task body, so the two
  task bodies are what need correcting and the pinned section does not.  This
  module imports Task 11's binding and writes no second predicate.
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
    _t5_radio_class,
    stochastic_nodes,
)
from rheplicant.config.sections.model import operator_table
from rheplicant.config.symbols import ShapeScope, literal_shadowing_a_symbol

#: Grid forms that declare an axis LENGTH the document writes down.
#: ``linspace``, ``arange`` and ``modulo`` carry it as ``num:``; ``list``
#: carries it as the length of the list itself.  Every form admitted is a
#: place a wrong ``n_freq`` could produce a FALSE warning, which is the only
#: way A41 can do harm -- so a form belongs here only when ``num`` (or the
#: list) IS the axis length with no arithmetic in between.  Measured for all
#: four: ``jnp.linspace(..., num)``, ``start + step * jnp.arange(num)``,
#: ``jnp.arange(num) % period`` (``arrays.py:121``) and ``jnp.asarray(list)``
#: each produce exactly that many samples, and a ``{modulo: {num: 6, period:
#: 3}}`` frequency grid builds ``n_freq == 6``.
#:
#: What stays out is the forms whose length the text does NOT state:
#: ``{zeros: [8]}`` and its kin state a SHAPE, and reading a shape as an axis
#: would admit an all-zero frequency axis as a measurement of the run.
_A41_COUNTED_FORMS: tuple[str, ...] = ("linspace", "arange", "modulo", "list")

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
            # DEFENSIVE and, measured, an equivalent mutant: a non-string key
            # would give `model.n.7`, which `parse_path` refuses -- but
            # `_task3_where` cuts every path back to its deepest spellable
            # prefix before it reaches `Finding.where`, so dropping this guard
            # changes no `where` and no verdict, only the path quoted in the
            # message.  It stays because a key nobody can type is not a place
            # to send a reader, and it is recorded as untestable-through-the-
            # pass rather than left looking earned.
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


# NO SECOND STOCHASTIC PREDICATE HERE.  §3.2 (f) binds ONE name for "does this
# node's operator class declare `key` in requires" and Task 11 owns it in
# `preflight/model.py`: `stochastic_nodes` publicly, and the per-spec form
# privately as `_a30_stochastic(node_id, spec, table)`.  (`stochastic_operator`
# is task-11.md's and task-12.md's spelling and appears in neither §3.1 nor
# §3.2, which are authoritative over a task body.)  A private predicate of this
# module's own is the collision §3.2 (f) predicted.  `stochastic_nodes` answers
# the `without:` leg; the `replace:` leg needs the per-spec form, because a
# `inference.twin.replace` spec is not a model node and `stochastic_nodes`
# cannot express it -- dropping it would drop §3.2 (h)2's whole correction.
#
# What the nested helper below adds is NOT a second answer to that question.
# It is a different question -- "did this pass NAME a class at all" -- which
# `_a30_stochastic` deliberately does not distinguish from "it does not draw",
# because in A30's polarity the two are the same answer and in A42's they are
# opposite ones.


def _a42_removed(document: Mapping[str, Any]) -> tuple[str, ...]:
    """The stochastic nodes ``model:`` lights that the FIT twin does not carry.

    Empty when ``inference.twin:`` is absent, and that is the whole correction
    to §4.7.1's wording: measured, ``twin: fit`` without that section returns
    the model twin itself (``twin.py:37-38``), so nothing has been removed and
    there is nothing to warn about.

    **A ``replace:`` counts only when the pass NAMED the replacement's class.**
    ``_a30_stochastic`` answers ``None`` both for "does not draw" and for "the
    text cannot say", and an earlier form of this function read either as a
    removal.  Measured, that was a live false warning: ``_t5_radio_class``
    resolves ``rheplicant.radio`` and nothing else, while the build resolves a
    ``python:`` target through ``hatch.import_target``, which imports any
    module.  So ``replace: {noise: {python:
    'rheplicant.radio.instrument.noise:NoiseOperator'}}`` **builds**, its fit
    twin's ``noise`` IS a ``NoiseOperator`` declaring ``key`` in ``requires``,
    and A42 told that document its data carried no noise realisation and to
    write ``twin: full``.  The docstring that justified the old reading -- "the
    only ``replace:`` spelling that both stops the draw and BUILDS is the
    ``python:`` hatch, which resolves to a class and is decided exactly" -- was
    false in the word *resolves*: two different resolvers, and only one of them
    is P-1's.

    The un-nameable half is a DECLINE and stays one.  Resolving an arbitrary
    module here would import a user's code during pre-flight -- a file read and
    an unbounded cost, which is exactly why ``_t5_radio_class`` is narrow.  The
    same decline costs the mirror-image check: measured, ``model.noise:
    {python: '<submodule>:NoiseOperator'}`` with ``without: [noise]`` builds,
    the fit twin genuinely loses the draw, and ``stochastic_nodes`` is empty,
    so A42 says nothing.  Both directions are pinned by tests; the lost half is
    §6 residue, not a bug this phase can close.
    """
    inference = document.get("inference")
    if not isinstance(inference, Mapping):
        return ()
    spec = inference.get("twin")
    if not isinstance(spec, Mapping):
        return ()
    drawing = stochastic_nodes(document)
    table = operator_table()

    def took_the_draw_out(node_id: str, replacement: Any) -> bool:
        """The pass named a class for ``replacement`` AND it does not draw.

        Nested rather than module level because §3.1 closes this file's
        module-scope names, and because it is one caller's clause rather than
        a property of the layer.
        """
        if _a30_stochastic(node_id, replacement, table) is not None:
            return False                       # it still draws
        if not isinstance(replacement, Mapping):
            return False                       # nothing to name
        if "python" in replacement:
            # `_t5_radio_class` is P-1's resolver and it is narrower than the
            # build's.  A target it cannot reach is "cannot say", never "does
            # not draw".
            return _t5_radio_class(replacement) is not None
        # UNREACHABLE TODAY, and kept because the day it stops being one it
        # is a lost check rather than a loud one: measured, every class
        # registered at a node that draws also draws, so a `type:` naming one
        # of the node's own classes always answers "still draws" at the first
        # clause above.  `test_the_type_route_of_the_replace_leg_is_
        # unreachable_today` is what dates that, rather than this comment.
        declared = replacement.get("type")
        return isinstance(declared, str) and declared in {
            cls.__name__ for cls in table.get(node_id) or ()}

    removed: list[str] = []
    without = spec.get("without")
    if isinstance(without, (list, tuple)):
        removed.extend(node_id for node_id in without
                       if isinstance(node_id, str) and node_id in drawing)
    replace = spec.get("replace")
    if isinstance(replace, Mapping):
        removed.extend(node_id for node_id, replacement in replace.items()
                       if isinstance(node_id, str) and node_id in drawing
                       and took_the_draw_out(node_id, replacement))
    return tuple(dict.fromkeys(removed))


def _a52_projector_refs(node: Any, where: str) -> list[str]:
    """Every document path under ``node`` holding ``{ref: resources.projectors.…}``."""
    found: list[str] = []
    if isinstance(node, Mapping):
        target = node.get("ref")
        if isinstance(target, str) and target.startswith(_A52_PREFIX):
            found.append(where)
        for key, value in node.items():
            # Defensive and equivalent, for the reason `_a41_shapes` gives.
            if isinstance(key, str):
                found.extend(_a52_projector_refs(value, f"{where}.{key}"))
    elif isinstance(node, (list, tuple)):
        for index, entry in enumerate(node):
            found.extend(_a52_projector_refs(entry, f"{where}[{index}]"))
    return found


@register("A41")
def _shadowed_literals(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A41: a literal integer in a shape that equals one of this run's extents.

    **Not a refusal**, for ``arrays.py:6-7``'s reason: a literal 8 may
    genuinely be 8, and what it cannot be is TIED to the grid.

    ``warn`` and not ``report``, though ``arrays.py:6`` and schema §6 both call
    A41 "a report", and the difference is worth stating because this layer has
    three severities where the schema has one word.  ``REPORT`` is "worth
    recording next to the run; not worth interrupting anyone over"
    (``findings.py:74``) and **is consumed by nothing** -- it is the severity
    ``checks.<name>.mode: report`` will read, and that gating is Plan 3C's
    (``findings.py:53-57``).  Shipping A41 at a severity no reader has yet
    would reproduce, at one remove, the defect A41 is about: a fact computed
    and stored for a consumer nobody wrote.  So it interrupts, once, until 3C
    gives the document a way to say it would rather it did not.

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
