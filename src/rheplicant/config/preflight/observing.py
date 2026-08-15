"""Pre-flight: the observation's own text, and what the model owes it.

Two checks, one subject.  ``observation.switching.order`` is the list schema
§4.1.5 says fixes four things at once, and both checks here are about the
places where fixing them is not enforced:

* **A14** -- an order that names calibration loads while the model places
  none.  The other two directions are enforced already and are deliberately
  not copied here: ``declared_order`` (``sections/switching.py:36-55``)
  refuses ``order[0] != "antenna"`` and a repeated label, and it runs inside
  ``build_observation`` (``document.py:72``), which is BEFORE
  ``build_resources`` (``:75``) -- measured, a document with a missing beam
  directory and ``order: [ambient, hot]`` reports the order.  The
  ``model.cal_loads`` key ORDER is late and is **Task 4's**, registered as the
  dotted slot ``A14.cal_loads`` in ``preflight/model.py`` against
  ``compose.cal_load_order_problem``; §3.2 (i) says "the definition of done
  lists both" and that sentence is false -- §5 names neither -- so this
  module's author read the code instead, and it is there.
* **A15**, which schema §6 words as a spelling ("assembled by
  ``from_switch_order`` matched by name") and which is really a property:
  ``noise_wave.gamma_src`` has exactly ``n_source`` rows, where ``n_source``
  is ``len(order) or 1`` -- the same expression
  ``ResolutionContext.shape_scope`` uses at ``context.py:55-57``.  That makes
  §4.1.5's "``mode: none`` requires exactly one row" the same rule at
  ``n_source == 1``, so it is emitted under the same check id with a message
  that cites §4.1.5 (§3.2 (h) 1).

**What the package does with a wrong row count -- measured at ``740d9d1``,
and it is not what this task was handed.**  The brief said a three-row
``gamma_src_re`` under a four-label order "BUILDS".  It does not: on a
document whose ``inference.observed`` simulates its data, ``load_document``
reaches the twin and ``rhino_cal_jax``'s switch cycle refuses with
*"source_index values span [0, 3] which is out of range for 3 labels ('0',
'1', '2')"*.  Three things are true instead, and the messages below say only
these:

* **too few rows** are seen only when the twin is evaluated, and then as an
  error about switch labels rather than about the field the user wrote;
* **too many rows** are seen by nothing at all -- measured, ``{ones: [9, 8]}``
  under a four-label order loads, simulates and runs, with five rows nobody
  indexes;
* with **no order at all**, any count but one reaches
  ``NoiseWaveOperator._source_index``, which refuses at call time because
  there is no switch index to choose a row with.

In every one of those the refusal, if there is one, arrives **after** the beam
is read -- measured, the same documents carrying an unreadable beam report the
beam.  That is what these two checks move.

**The order is read through Task 4's ``_t4_switch_order``, not re-read here.**
``switching:`` has two grammars -- an ingested run (``observation.from_file``)
declares ``order:`` alone with no ``mode:`` (``observation.py:336-348``),
everything else goes through ``compile_switching``, where an absent ``mode:``
means ``none`` -- and a second reader of it is the two-validators shape this
layer has paid for.  §3.1 binds ``_switch_order`` and ``_gamma_rows`` here and
the order reader there; this module imports it, the way
``sections/observed.py:23`` imports ``draws._seed_name``.

``from_switch_order`` is never second-guessed: it stacks over
``context.switch_order`` (``refs.py:169-188``), so its row count cannot be
wrong.  Every other form is checked when -- and only when -- the document's
own text says how many rows it has.  A ``{ref:}``, a ``{file:}`` and an
arithmetic extent such as ``"2 * n_source"`` say nothing this pass may
resolve, and produce no finding: this check declines rather than guesses, so
the only way it can be wrong is by staying silent.  Recorded against 3B.

**Where these two ids reach the pass from.**  ``preflight/__init__.py``'s
foot import is what registers them, and no test in
``test_preflight_observing.py`` can see it missing -- that module imports
``observing`` itself, so deleting the foot import leaves all of them green
(measured).  The guard is
``test_config_preflight.py::TestTheFootImportCannotRot::
test_every_module_under_preflight_is_imported_at_the_foot`` (``:1559``), and
the task body's claim that the two registry tests here are "this task's guard
against the whole module being dead code" is false.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.modifiers import NOISE_AXES
from rheplicant.config.preflight import register
from rheplicant.config.preflight.model import (
    _nodes,
    _t4_graph,
    _t4_switch_order,
    _t5_placement,
    _t5_radio_class,
)
from rheplicant.config.sections.model import _pick_class, operator_table

#: Forms whose shape is the form's own value: ``{zeros: [4, 8]}``.
_A15_SHAPE_FORMS: tuple[str, ...] = ("zeros", "ones")

#: Forms whose shape is nested under ``shape:``: ``{normal: {shape: [4, 8]}}``.
#: Split from the pair above because ``full``/``normal``/``uniform`` take other
#: arguments beside it, and because reading them is what stops A15 depending on
#: which constructor the writer reached for -- the divergence
#: ``symbols.resolve_shape``'s own docstring records for check A41.
_A15_NESTED_SHAPE_FORMS: tuple[str, ...] = ("full", "normal", "uniform")

#: The two operator fields the property is about.  Both, always: a document may
#: spell one in a form this pass can count and the other in one it cannot, and
#: checking only the first leaves the same hole open on its twin.
_A15_HALVES: tuple[str, ...] = ("gamma_src_re", "gamma_src_im")


def _a15_extent(entry: Any, n_source: int) -> int | None:
    """One shape position, as an integer, or ``None`` when the text does not say.

    Three spellings and no more: a literal integer, the symbol ``n_source``
    (which is right by construction) and the symbol ``n_load``, which is
    ``n_source - 1`` (``symbols.py:73-74``) and is the off-by-one this field
    invites -- a document that wrote the number of LOADS where the number of
    SOURCES belongs.  Every other position -- ``"2 * n_source"``, ``"n_freq"``
    -- is ``resolve_extent``'s and needs a ``ShapeScope``, which needs the
    run's grids; those are text-decidable only when the grids are literal, and
    that half is 3B's.  Returning ``None`` there is what keeps this check
    unable to be wrong in the refusing direction.

    ``bool`` is excluded before ``int``, and that is not defensive tidiness:
    ``isinstance(True, int)`` is True in Python, so the obvious spelling reads
    ``{zeros: [True, 8]}`` as one row and reports a count for a shape
    ``resolve_extent`` refuses in its own words (``symbols.py:128-132``).
    """
    if isinstance(entry, bool):
        return None
    if isinstance(entry, int):
        return entry
    if not isinstance(entry, str):
        return None
    if entry == "n_source":
        return n_source
    if entry == "n_load":
        return n_source - 1
    return None


def _a15_stacks_on_axis_zero(node: Mapping[str, Any]) -> bool:
    """Are this ``stack``'s rows its entries?  ``refs.py:120-126``'s rule.

    Three spellings say yes: no ``axis:`` at all, an integer ``0`` (stack's
    own argument), and one of ``NOISE_AXES`` -- ``'time'``/``'freq'``/
    ``'none'`` (``modifiers.py:34``) -- which is the noise-sigma MODIFIER and
    leaves the stack on axis 0.  Measured: four entries under ``axis: 'time'``
    build at ``(4, 8)``, so reading ``node.get("axis", 0) == 0`` declines to
    count a stack it could have counted.

    Everything else stands down, INCLUDING the two spellings that do stack on
    axis 0 -- ``axis: true`` and ``axis: 0.0``, which fail ``refs.py``'s
    ``mine`` test and fall through to zero.  The modifier alphabet then
    refuses them in their own words (*"axis=True is not one of ['time',
    'freq', 'none']"*), and a row count in front of that refusal names a fix
    that is not the fault -- Task 5's pre-emption rule, which is also why A15
    stands down on a pipeline and on a class that is not a noise wave.
    """
    declared = node.get("axis", 0)
    if isinstance(declared, str):
        return declared in NOISE_AXES
    if isinstance(declared, bool):
        return False
    return isinstance(declared, int) and declared == 0


def _a15_shape_rows(shape: Any, n_source: int) -> int | None:
    """``shape[0]`` of a written shape, when that shape is a gamma_src's."""
    if not isinstance(shape, (list, tuple)) or len(shape) != 2:
        return None
    return _a15_extent(shape[0], n_source)


def _a15_declared_rows(node: Any, n_source: int) -> int | None:
    """The row count this value node declares, or ``None``.

    **A shape the text says is not two-dimensional declares no row count.**
    ``gamma_src`` is ``(n_source, n_freq)``, and ``{zeros: [3]}`` is a
    document whose fault is its ndim -- ``__check_init__`` refuses that one by
    name (``noise_wave.py:217-221``), and "write 4 rows in switch order" would
    be a row-count sentence in front of it naming a fix that does not fix it.
    Only the shape FORMS can be asked this: a ``list`` or a ``stack`` says how
    many rows it has and nothing about how deep they are.

    An empty ``list``/``stack`` declares nothing either: both are the value
    grammar's own refusals (``refs.py:113-119`` says a stack is "a container,
    not a computation"), and answering "0 rows" here would do the same thing.
    """
    if not isinstance(node, Mapping):
        return None
    for form in _A15_SHAPE_FORMS:
        if form in node:
            return _a15_shape_rows(node[form], n_source)
    for form in _A15_NESTED_SHAPE_FORMS:
        inner = node.get(form)
        if isinstance(inner, Mapping):
            return _a15_shape_rows(inner.get("shape"), n_source)
    rows = node.get("list")
    if isinstance(rows, (list, tuple)) and rows:
        return len(rows)
    entries = node.get("stack")
    if isinstance(entries, (list, tuple)) and entries:
        return len(entries) if _a15_stacks_on_axis_zero(node) else None
    return None


def _a15_declared_class(node_id: Any, spec: Mapping[str, Any],
                        table: Mapping[str, tuple[type, ...]]) -> Any:
    """The operator class this entry declares, when the text names exactly one.

    ``build_node_operator``'s own dispatch order (``model.py:317-336``),
    read once: ``python:`` first, then ``from:``, then the node's registered
    classes through :func:`~rheplicant.config.sections.model._pick_class` --
    which is CALLED rather than re-implemented, inside a ``try`` the way Task
    4 calls ``declared_order``, because its three refusals (an ambiguous node
    with no ``type:``, a ``type:`` no class answers to, and the deferred
    ``NeuralOperator``) belong to A7, A39 and the build, not to A15.

    A key that is not a graph node answers ``None``: A2 refuses it by name in
    this same report, and a second sentence about it is one too many.
    ``from:`` and ``compose:`` answer ``None`` for the same reason a
    ``compose:`` block has no gamma halves to count -- the class is decided a
    level below where this reads.
    """
    if not isinstance(node_id, str) or node_id not in _t4_graph().nodes:
        return None
    if "python" in spec:
        return _t5_radio_class(spec)
    if "from" in spec or "compose" in spec:
        return None
    classes = table.get(node_id)
    if not classes:
        return None
    try:
        return _pick_class(node_id, classes, spec)
    except ConfigError:
        return None


def _a15_carries_gamma(cls: Any) -> bool:
    """Does this class take both gamma_src halves as constructor fields?

    The IDENTITY question, and it is separate from the placement one on
    purpose.  Measured: ``noise_wave: {type: GainOperator, gain: ...,
    gamma_src_re: {zeros: [3, 8]}}`` lands at the ``noise_wave`` node, so a
    site rule written as a placement asks a ``GainOperator`` for its switch
    rows and offers the only refusal the reader gets -- about a field
    ``GainOperator`` does not accept, one phase before ``_pick_class`` says
    *"type: 'GainOperator' is not registered at this node"*.  A row count is a
    property of the OPERATOR; the node it lands on decides nothing about it.

    ``dataclasses.fields`` is §2.4's class introspection: no operator is
    constructed, and a class that is not a dataclass answers False rather
    than raising inside the pass.
    """
    if not dataclasses.is_dataclass(cls):
        return False
    return set(_A15_HALVES) <= {field.name for field in dataclasses.fields(cls)}


def _a15_sites(document: Mapping[str, Any]) -> list[tuple[str, Mapping]]:
    """Every entry in this document that declares a noise-wave operator.

    Two routes and three spellings, and both routes ask the same two
    questions -- is this key a graph node, and is the class it declares one
    that carries ``gamma_src``:

    * a ``model:`` entry.  Usually the ``noise_wave:`` key; measured,
      ``bandpass: {python: 'rheplicant.radio:NoiseWaveOperator', ...}`` is the
      third spelling and builds, with ``twin['noise_wave'].gamma_src_re`` at
      the rows it declared;
    * an ``inference.twin.replace.<node>`` entry, whose spec reaches the same
      ``build_node_operator`` (``twin.py:67-69``).  Measured, a replacement
      carrying three rows under a four-label order gives a fit twin at
      ``(3, 8)`` beside a model twin at ``(4, 8)``.  Read the SAME way as the
      model half rather than as the literal key ``noise_wave``: two readings
      of one question is the shape this layer has paid for, and neither test
      could tell them apart while they agreed.

    **The twin route needs a graph model.**  ``build_fit_twin``
    (``twin.py:46-50``) refuses the whole ``inference.twin:`` block on a
    ``kind: pipeline`` model -- *"A pipeline is rebuilt, not repaired"* -- so
    a row count inside a block that is about to be rejected wholesale would
    be answering about a document nobody can fix that way.  ``_nodes``
    already answers ``{}`` for such a model, so the model half needs no guard
    of its own; this one is the twin half's.

    A ``noise_wave`` written as ``{compose: {stages: [...]}}``, and a ``kind:
    pipeline`` stage named ``noise_wave``, put the field one level below where
    this reads.  Both are measured genuinely silent rather than crashing, and
    both are recorded HERE and owed to §6's ledger, which as of ``740d9d1``
    records neither.
    """
    sites: list[tuple[str, Mapping]] = []
    table = operator_table()
    for key, spec in _nodes(document).items():
        if isinstance(spec, Mapping) and _a15_carries_gamma(
                _a15_declared_class(key, spec, table)):
            sites.append((f"model.{key}", spec))
    model = document.get("model")
    if not isinstance(model, Mapping) or model.get("kind", "graph") != "graph":
        return sites
    inference = document.get("inference")
    twin = inference.get("twin") if isinstance(inference, Mapping) else None
    replace = twin.get("replace") if isinstance(twin, Mapping) else None
    if isinstance(replace, Mapping):
        for key, spec in replace.items():
            if isinstance(spec, Mapping) and _a15_carries_gamma(
                    _a15_declared_class(key, spec, table)):
                sites.append((f"inference.twin.replace.{key}", spec))
    return sites


@register("A14")
def _switch_order(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A14: an order that names loads requires a model that places them.

    A list, not a generator, for readability alone: ``preflight`` consumes
    each check with ``tuple(fn(document))`` **inside** its own ``try``, so a
    generator's exception is caught and named exactly as an eager one is, and
    either shape is correct.

    **Three documents stand down, and each was measured rather than reasoned
    about**:

    * an INGESTED run.  ``tests/config/test_config_document.py:133-151``
      (``TestIngestedDocuments.make_document``) is a recording with
      ``switching: {order: [antenna, internal_load, heated_load]}``, a
      ``model:`` of ``{gain: ...}`` and no ``cal_loads``
      anywhere; it loads, and ``run_forward`` doubles the recorded ones.  The
      order there labels the recording's own source index
      (``document.py:85-86`` hands it to ``to_state`` as ``source_order``)
      and there is no model branch for it to fix,
      because the data was not simulated.  Refusing it would refuse a
      document the package builds AND runs, and turn three tests red;
    * a ``kind: pipeline`` model, which has no graph node ids at all, so
      ``model.cal_loads`` is not a thing it could declare -- measured, such a
      document with a switching cycle builds;
    * a ``model:`` that is not a mapping.  ``_structural`` guarantees the
      section is PRESENT, never that it is a mapping, and ``build_model``
      answers that one with the type it got.

    And the loads are counted as PLACEMENTS, not as the key ``cal_loads``:
    measured, ``bandpass: {python: 'rheplicant.radio:CalLoadOperator',
    t_load: ...}`` builds with the load at the ``cal_loads`` node --
    ``twin.lit`` carries ``'cal_loads'`` and not ``'bandpass'`` -- so a
    key-shaped test refuses a document that assembles.  A ``cal_loads:``
    written with the wrong SHAPE stands down too -- that is A6's sentence,
    and Task 4's ``A14.cal_loads`` leg stands down on it for the same reason.

    **A FOURTH document stands down, and it was a LIVE FALSE REFUSAL until
    the commit that added this clause.**  This check refuses on ABSENCE, and
    :func:`~rheplicant.config.preflight.model._t5_claims` answers ``()`` both
    for "nothing is placed" and for "the text cannot say where" -- its own
    docstring says so.  Measured, ``bandpass: {python:
    'rheplicant.radio.instrument.calibration:CalLoadOperator', ...}`` was read
    as the first when it is the second: the build resolves that target through
    ``hatch.import_target`` to the very class the exported spelling names, the
    load IS placed at ``cal_loads``, the assembly is identical either way
    (``twin.lit`` equal, verified with this check bypassed) -- and A14 refused
    it.  The same commit widened ``_t5_radio_class`` so every class
    ``rheplicant.radio`` exports resolves under any spelling of its own
    module, which closes that document; what stands down here is what is left:
    a genuinely FOREIGN class, whose ``graph_node`` this pass may not import
    to read.  ``_t5_placement`` answers ``None`` for exactly that entry, and
    one such entry anywhere in ``model:`` stands the whole check down --
    because it is the entry that might be the calibration load, and "no load"
    is a claim this document cannot support.

    **The ``switching: {mode: none}`` alternative is CONDITIONAL, and that is
    an advice loop closed rather than a nicety.**  Measured before the clause
    below: a four-key document with a two-label order and a two-row
    ``gamma_src`` earned A14, whose fix ``switching: {mode: none}`` then
    earned A15 **twice** -- *"this run declares no observation.switching
    order, so it has exactly one source"* -- whose own fix is ``switching:
    {mode: cycle, order: [antenna, ...]}``, and the document that produces is
    ``==`` the one the reader started from.  Both checks are this module's and
    one drafter wrote them.  A14 is quoted first by ``raise_if_refused``, so a
    reader following it never saw the finding that forbids its remedy.
    """
    findings: list[Finding] = []
    order = _t4_switch_order(document)
    # `not order` alone: `_t4_switch_order` returns None when the text does
    # not decide, () for a run that does not switch, and otherwise a tuple of
    # length >= 2 -- `declared_order` rejects a shorter one itself.  A
    # `len(order) < 2` arm here is unreachable, and an unreachable arm reads
    # as a case someone handled.
    if not order:
        return findings
    observation = document.get("observation")
    if isinstance(observation, Mapping) and "from_file" in observation:
        return findings
    model = document.get("model")
    if not isinstance(model, Mapping) or model.get("kind", "graph") != "graph":
        return findings
    placements = [_t5_placement(key, spec)
                  for key, spec in _nodes(document).items()]
    if any(placed is None for placed in placements):
        return findings
    if any(placed == ("cal_loads",) for placed in placements):
        return findings
    findings.append(refuse(
        "A14", "model.cal_loads",
        f"observation.switching.order declares {list(order[1:])} after the "
        f"antenna, and this model places no calibration load at all. One list "
        f"fixes the switch indices, the order of model.cal_loads, the row "
        f"order of noise_wave.gamma_src and the thermistor_columns labels, so "
        f"an order with no loads behind it gives {len(order)} switch "
        f"positions to a model with no calibration branch at all -- and "
        f"nothing refuses that: measured, such a document builds and its twin "
        f"runs, with the cycle in coords.extra['receiver_input'] and no load "
        f"to switch to. Declare model.cal_loads with the keys "
        f"{list(order[1:])} in that order"
        f"{_a14_dropping_the_order(document)} (check A14)."))
    return findings


def _a14_dropping_the_order(document: Mapping[str, Any]) -> str:
    """A14's second alternative -- and what else it costs, when it costs
    something.

    ``switching: {mode: none}`` is a real fix for a document whose only fault
    is an order with no loads behind it.  It is NOT one on a document that
    also writes a ``gamma_src`` row count, because ``n_source`` becomes 1 and
    A15 -- the other check in this module, by the same drafter -- refuses
    every row count but one.  Offering it bare is what made the pair a closed
    loop: measured, A14's fix earned A15 twice and A15's fix restored A14's
    document exactly (``step2 == step0``).

    So the coupled edit is NAMED rather than the alternative being withdrawn:
    a reader who wants no switch cycle can still have one, and now knows what
    else has to move.  The rows are read through A15's own
    :func:`_a15_sites`/:func:`_a15_declared_rows`, not a second reader --
    they answer ``None`` wherever the text does not say how many rows there
    are, and a row count this pass cannot read cannot be coupled to anything.

    **The count is read at ``n_source = 1``, the value it would HAVE**, not at
    the one the document has now.  ``_a15_declared_rows`` resolves the symbols
    ``n_source`` and ``n_load``, so a ``{zeros: ['n_source', 8]}`` follows the
    order down to one row and is not coupled at all; reading it against the
    current order would name a field that needs no edit, which is the fix
    clause telling a reader to write the number they already wrote (Task 6's
    own mutation finding).
    """
    coupled = sorted({
        f"{site}.{half}"
        for site, spec in _a15_sites(document)
        for half in _A15_HALVES
        if (rows := _a15_declared_rows(spec.get(half), 1)) is not None
        and rows != 1})
    if not coupled:
        return ", or write switching: {mode: none}"
    return (", or write switching: {mode: none} AND cut "
            f"{coupled} to a single row each -- mode: none is one source, and "
            "check A15 refuses any other gamma_src row count under it, so "
            "dropping the order alone trades this refusal for that one")


@register("A15")
def _gamma_rows(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A15 and §4.1.5: ``gamma_src`` has exactly ``n_source`` rows, every path.

    Both halves are read, not just ``gamma_src_re``.  ``__check_init__``
    (``noise_wave.py:222-226``) makes a disagreement BETWEEN the two halves
    its own refusal, which is why two findings here are not a contradiction:
    they are two fields, each wrong against the order on its own.
    """
    findings: list[Finding] = []
    order = _t4_switch_order(document)
    if order is None:
        # "The text does not say what the order is" -- a malformed
        # `switching:` block, whose own refusal is `declared_order`'s or
        # `compile_switching`'s and already precedes the beam.  `()` is the
        # different answer that DOES decide: no order is one source
        # (§3.2 (h) 1), and `n_source = len(order) or 1` below is
        # `context.py:55-57`'s own expression.
        return findings
    n_source = len(order) or 1
    for site, spec in _a15_sites(document):
        for half in _A15_HALVES:
            rows = _a15_declared_rows(spec.get(half), n_source)
            if rows is None or rows == n_source:
                continue
            where = f"{site}.{half}"
            findings.append(refuse(
                "A15", where, _a15_message(where, order, rows, n_source,
                                           half)))
    return findings


def _a15_message(where: str, order: tuple[str, ...], rows: int, n_source: int,
                 half: str) -> str:
    """The refusal, whose second half is what the PACKAGE does about it.

    **It opens with the field's own path**, which is not decoration:
    ``Report.raise_if_refused`` (``findings.py:159-178``) raises the first
    refusal's MESSAGE and names only the OTHER refusals' ``where``, so a
    reader who hits this one first would otherwise be told a row count with
    no field -- and there are two routes and three spellings it could be.

    Three tails, because there are three behaviours and a single one would be
    false about two of them (all measured at ``740d9d1``, and the brief this
    task was handed asserted the silent one for every case).
    """
    if order:
        why = (f"observation.switching.order declares {len(order)} sources "
               f"({list(order)}), and this declares {rows} rows.")
        fix = (f"Write {n_source} rows in switch order, or "
               f"{{from_switch_order: {{resource: resources.s_params, part: "
               f"{half.rsplit('_', 1)[1]}}}}}, which stacks them by name.")
        if rows > n_source:
            cost = ("Nothing refuses this anywhere: NoiseWaveOperator checks "
                    "ndim, re/im agreement and n_freq and never n_source, so "
                    "the extra rows are carried, never used, and the run "
                    "comes back finite and confident.")
        else:
            cost = ("NoiseWaveOperator checks ndim, re/im agreement and "
                    "n_freq and never n_source, so nothing sees this until "
                    "the twin is evaluated -- and then as a switch-cycle "
                    "error about labels rather than about the field you "
                    "wrote.")
    else:
        why = ("this run declares no observation.switching order, so it has "
               f"exactly one source, and this declares {rows} rows.")
        fix = ("Write one row (schema §4.1.5: mode: none means no cal_loads "
               "and a single gamma_src row), or declare switching: {mode: "
               "cycle, order: [antenna, ...]}.")
        cost = ("NoiseWaveOperator checks ndim, re/im agreement and n_freq "
                "and never n_source, so nothing sees this until the twin is "
                "evaluated and the operator finds no switch index to choose "
                "a row with.")
    return f"{where}: {why} {fix} {cost} (check A15)."
