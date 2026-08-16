"""A10, A45 and A46: what the recording is asked for, and what the switch reads.

Three checks about one document shape -- a run whose observation comes off
disk -- and each of them is late today for the same reason: the question is
answerable from the document's own text and is currently asked from inside
the reader, behind the whole recording.

* **A10** is a HOIST.  ``sections/ingest.py::freq_unit_problem`` is the one
  binding of the sentence and this pass calls it; the reader keeps calling it
  too.  Measured at ``ea4839b``, a document whose ``from_file:`` omits
  ``freq_unit`` records ``['read_bytes', 'sha256']`` before the refusal
  arrives -- the file is slurped and hashed to miss a one-token key.  **The
  digest is not moved**: it is read today, not only by a later plan, because
  ``sections/ingest.py`` writes ``from_file/sha256`` into the provenance
  record and ``tests/config/test_config_document.py`` asserts it is there.
  What moves is the question.
* **A45** is INVENTED.  ``switch_key:`` names a key in ``coords.extra`` and
  nothing checked that the run writes one.  Measured: with more than one
  source the twin refuses the moment it is evaluated, as a
  ``StateValidationError`` naming ``coords.extra['my_switch']`` and no
  document key; with exactly one source ``_source_index`` returns zeros and
  the wrong key costs nothing visible at all.
* **A46 legs 2 and 3** are INVENTED.  Leg 1 -- ``model.cal_loads``' keys
  against ``switching.order[1:]`` -- is Plan 3A's ``A14.cal_loads`` slot, is
  STRICTER than A46 (ordered equality, not membership), and is not touched
  here.  Leg 2 is the ``label:`` with no thermistor column; leg 3 is the WARN
  for two loads that share one.

**What A46 constrains, and it is not what schema §6 points at.**  Measured,
``sections/model.py``'s thermistors route is
``cal_load_operators(context.ingest, labels=[label])[label]``, so the
``label:`` VALUE is what indexes ``thermistor_k``; the ``cal_loads`` KEY is
``A14.cal_loads``' subject and nothing else.  A crossed document -- keys and
labels swapped between two loads -- LOADS today, and it is not refused here:
it silently swaps which physical temperature each switch position gets, which
is a real defect and a different check's, not this one's.

**Leg 3 compares only the labels a load actually READS.**  The reader demands
a thermistor column for every label the recording's switch log carries, the
antenna included, and the shipped ``/temperatures/temperatures`` has two
columns for three labels -- so sharing is FORCED and ``radio/rhino.py``
documents it as legal.  A leg 3 written over the whole ``thermistor_columns``
map warns on the canonical working document.

**``inference.twin.replace.<node>`` is walked by all three** (§0.3 E.10).
That route reaches the same ``build_node_operator`` and is outside
``preflight/model.py::_nodes``.  Measured with an empty pre-flight report:
``replace.noise_wave.switch_key: nope`` LOADS CLEAN and detonates at fit
time, and ``replace.cal_loads: {from: thermistors, label: ghost}`` reaches
the reader's own ``DataIngestionError``.  A10 walks it by walking every
section that is not ``variants:``.

**None of the three walks the variant LAYERS.**  ``document.py::_assemble``
applies the selected variant and THEN runs this pass, so a variant a user
actually runs is read by all three checks on the merged mapping.  What
``preflight/document.py::_task3_over_layers`` would add is therefore **only
the reporting of faults in variants nobody selected** -- and that is true of
every check in this layer, not just these three.  The un-selected variant is a
RECORDED false negative (§3.2 (c)), with a test per row that names it.

**The cost, measured properly, and the reason for the drop.**  Cold, on the
shipped guard's own document (40 ``plan.sample`` runs, 20 variants, one
``preflight()`` per fresh process, 22 samples each): **22.3 ms unlayered
against 37.8 ms layered, BOTH under §5's 50 ms bound**; the layered walk never
exceeded 42.5 ms.  So the layering is affordable in isolation and this module
must not claim otherwise -- an earlier draft of this paragraph said it took the
guard "from 26 ms to 52.6 ms", which was **machine noise from a box running
six agents at once**, not a property of the layering.  Measured under
``pytest tests/config -n 16``, the same guard reads **52.2 ms with no layering
at all** and fails outright.  ~94 % of the layered walk is ``apply_variant``'s
``deepcopy``, so the real reason the walk is dropped here is arithmetic the
wave owns rather than this task: +15 ms of a 50 ms budget, with ~22 ms already
spent and five sibling branches landing into the same registry.  Restoring it
costs four test edits and becomes cheap the day ``_task3_layers`` is memoised.

**THAT DAY HAS COME, AND THE ARITHMETIC ABOVE IS RETIRED.**  The wave-boundary
fix memoised ``_task3_layers``: one document's layers are built once per pass
and shared by every check that walks them, so a check added to that walk now
costs **zero additional ``apply_variant`` calls** -- measured on this very
guard's document, 210 merges before and 21 after, with the cold pass at 13 ms
against 45 ms.  The "+15 ms" that justified the drop does not exist any more;
the walk here would add only its own per-layer read.

**The orchestrator's ruling, recorded rather than left as silence.**  The cost
objection is dead, so the recorded false negative above should be re-opened --
but restoring the walk is a behaviour change to a merged task's checks, it
needs its own adversarial round, and wave 2 was closing when the trigger fired.
It is therefore DEFERRED to Plan 3C or a dedicated follow-up, and it is the
first thing that should be picked up there.  This paragraph exists because a
pre-registered trigger that fires and produces nothing is worse than no trigger
at all: the condition is met, the reason for the drop is gone, and the only
thing still standing between these three checks and the un-selected variant is
scheduling.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.delivery import field_specs
from rheplicant.config.findings import Finding, refuse, warn
from rheplicant.config.preflight import register
from rheplicant.config.preflight.document import _task3_where
from rheplicant.config.sections.ingest import RHINO_FORMAT, freq_unit_problem
from rheplicant.config.sections.model import operator_table
from rheplicant.config.sections.pointing import pointing_extra_keys

#: The ``NoiseWaveOperator`` field A45 is about.  Named rather than spelled
#: four times, and read off the CLASS rather than off the token: measured over
#: :func:`operator_table`, exactly one node's classes carry it today, and a
#: check keyed on the literal ``type: NoiseWaveOperator`` misses the
#: ``python:`` spelling that Plan 3A's tests already exercise.
_A45_FIELD = "switch_key"

#: The one ``model:`` node whose entries A46 reads.  A FAN node: its spec is a
#: mapping of the user's own load keys, and ``_t4_entries`` is what expands it.
_A46_NODE = "cal_loads"

#: ``model.cal_loads.<key>.from`` that means "read the recording's thermistors".
_A46_ROUTE = "thermistors"


# ---------------------------------------------------------------------------
# The readers the three checks share.  `preflight.model` and
# `preflight.observing` are imported INSIDE the functions and not at this
# module's head, deliberately: the foot-import block in `preflight/__init__.py`
# is alphabetical and `ingest` sits BEFORE `model` and `observing` in it, so a
# head import here would register their ids first and silently reorder
# `CHECKS` -- which IS run order, and which §0.3 C.5 pins alphabetically on
# purpose.  `preflight.model._t4_graph` sets the same precedent for the same
# kind of reason.
# ---------------------------------------------------------------------------


def _a10_file_nodes(node: Any, where: str) -> list[tuple[str, Mapping]]:
    """``(document path, the file spec)`` for every ``{file: ...}`` under ``node``.

    The path is the VALUE NODE's -- ``resources.arrays.rec`` rather than
    ``resources.arrays.rec.file`` -- which is ``preflight/values.py``'s
    ``_a41_shapes`` convention and is what the reader edits.

    A whole-section recursion rather than a fixed list of places, because A10's
    real twin is any of them: measured, ``resources.arrays.<n>: {file:
    {format: rhino_hdf5, path: ...}}`` reaches the same reader behind the same
    ``read_bytes``+``sha256`` and earns the identical refusal.  A check that
    knew only about ``observation.from_file`` would guard one route of two.
    """
    found: list[tuple[str, Mapping]] = []
    if isinstance(node, Mapping):
        spec = node.get("file")
        if isinstance(spec, Mapping):
            found.append((where, spec))
        for key, value in node.items():
            if isinstance(key, str):
                found.extend(_a10_file_nodes(value, f"{where}.{key}"))
    elif isinstance(node, (list, tuple)):
        for index, entry in enumerate(node):
            found.extend(_a10_file_nodes(entry, f"{where}[{index}]"))
    return found


def _a10_sites(document: Mapping[str, Any]) -> list[tuple[str, Mapping]]:
    """Every place in the document that reaches the ``rhino_hdf5`` reader.

    Two spellings.  ``observation.from_file`` is a BARE spec --
    ``parse_from_file`` wraps it in ``{file: ...}`` itself -- and every other
    route is a value node that writes the wrapper out.  Both end at
    ``sections/ingest.py::_read_rhino_hdf5``.

    ``variants:`` is skipped, and that is not tidiness.  ``document.py``
    applies the selected variant BEFORE it runs this pass (measured:
    ``_assemble`` calls ``apply_variant`` and then ``preflight``), and
    ``apply_variant`` leaves ``variants:`` in the merged mapping -- so a walk
    that also descended into the patch would report the selected variant's
    ``from_file:`` TWICE, once at its real path and once inside the patch that
    put it there.
    """
    sites: list[tuple[str, Mapping]] = []
    observation = document.get("observation")
    if isinstance(observation, Mapping):
        from_file = observation.get("from_file")
        if isinstance(from_file, Mapping):
            sites.append(("observation.from_file", from_file))
    for section, block in document.items():
        if isinstance(section, str) and section != "variants":
            sites.extend(_a10_file_nodes(block, section))
    return [(where, spec) for where, spec in sites
            if spec.get("format") == RHINO_FORMAT]


def _a45_written_keys(document: Mapping[str, Any]) -> frozenset[str] | None:
    """The ``coords.extra`` keys this run will carry, or "cannot say".

    Three producers, and reading only the first refuses documents that build:

    * ``observation.extra`` -- the keys the document writes by hand;
    * ``observation.pointing`` -- ``compile_pointing`` materialises
      ``lst_deg`` and ``selfrot_deg`` into ``pointing.extra`` and
      ``build_observation`` merges them in.  Asked through
      ``sections/pointing.py::pointing_extra_keys``, which is the one binding
      of that four-branch rule (§2.2);
    * ``receiver_input``, UNCONDITIONALLY.  Not "only if ``switching.mode:
      cycle``": on an ingested run ``switching.receiver_input`` is ``None`` at
      ``build_observation`` and the key arrives from ``to_state`` afterwards,
      so a mode-gated reading refuses the ingested document this whole module
      is about.

    ``None`` when the observation section or its ``extra:`` is not a mapping
    -- ``sections/observation.py`` refuses both with the shape it got, and a
    guess here would pre-empt a better sentence.
    """
    observation = document.get("observation")
    if not isinstance(observation, Mapping):
        return None
    extra = observation.get("extra")
    if extra is not None and not isinstance(extra, Mapping):
        return None
    keys = {"receiver_input"}
    if isinstance(extra, Mapping):
        keys.update(key for key in extra if isinstance(key, str))
    keys.update(pointing_extra_keys(observation.get("pointing")))
    return frozenset(keys)


def _a45_sites(document: Mapping[str, Any]) -> list[tuple[str, str, Any]]:
    """``(document path, node id, one entry's spec)`` for every operator declared.

    ``preflight/observing.py::_a15_sites`` asks the same two-route question
    for A15 and this is deliberately the same shape, with one widening:
    ``_t4_entries`` expands a ``many`` node's list and FAN entries and a
    ``compose:`` block's stages, so a ``switch_key`` written one level down is
    read rather than silently skipped.  The NODE ID is carried beside the path
    because a stage's path is ``model.noise.stages[0]`` while the class still
    resolves at ``noise``.

    The ``inference.twin.replace`` half needs the graph model guard of its
    own: ``build_fit_twin`` refuses the whole ``inference.twin:`` block on a
    ``kind: pipeline`` model, so a ``switch_key`` inside a block about to be
    rejected wholesale is not this check's sentence.  ``_nodes`` already
    answers ``{}`` for such a model, so the model half needs no guard.
    """
    from rheplicant.config.preflight.model import _nodes, _t4_entries, _t4_graph

    graph = _t4_graph()
    sites: list[tuple[str, str, Any]] = []
    for node_id, spec in _nodes(document).items():
        if node_id not in graph.nodes:
            continue
        for path, entry in _t4_entries(node_id, spec,
                                       many=graph.nodes[node_id].many):
            sites.append((f"model.{path}", node_id, entry))
    model = document.get("model")
    if not isinstance(model, Mapping) or model.get("kind", "graph") != "graph":
        return sites
    inference = document.get("inference")
    twin = inference.get("twin") if isinstance(inference, Mapping) else None
    replace = twin.get("replace") if isinstance(twin, Mapping) else None
    if isinstance(replace, Mapping):
        for node_id, entry in replace.items():
            if isinstance(node_id, str) and node_id in graph.nodes:
                sites.append((f"inference.twin.replace.{node_id}", node_id,
                              entry))
    return sites


def _a45_carries_switch_key(cls: Any) -> bool:
    """Does this class take ``switch_key`` as a constructor field?

    §2.4's class introspection and nothing more: ``field_specs`` reads
    ``dataclasses.fields`` plus the type hints and CONSTRUCTS nothing.  The
    ``is_dataclass`` guard is ``_a15_carries_gamma``'s, for its reason -- a
    class that is not one answers False rather than raising inside the pass,
    where a raise costs the document every other finding.

    **The class, never the token.**  Measured, exactly one node's classes
    carry the field today (``noise_wave``/``NoiseWaveOperator``), and
    ``core/combinators.py::SelectOperator`` carries a ``switch_key`` too --
    defaulted, set by the fold, and not a config-writable node.  Asking the
    class the document declares is what keeps this check off it.
    """
    if not isinstance(cls, type) or not dataclasses.is_dataclass(cls):
        return False
    return _A45_FIELD in field_specs(cls)


def _a46_loads(document: Mapping[str, Any]) -> tuple[list[tuple[str, str]],
                                                  list[tuple[str, str]]]:
    """``(model entries, twin-replace entries)`` reading the thermistor log.

    Each entry is ``(document path, the label: VALUE)``.  The two halves come
    back apart because leg 3 is about the MODEL's loads only: a ``replace:``
    mapping carries at most one ``cal_loads`` operator, so two labels cannot
    share a column inside it, and comparing a replacement against the model's
    own entries would be a claim about two different twins.  That restriction
    is recorded rather than guessed at.

    An entry with no ``from: thermistors``, or with a ``label:`` that is not a
    non-empty string, is skipped: the first reads its ``t_load`` from a value
    node and never touches the recording, and the second is
    ``sections/model.py``'s own refusal, which names the key.
    """
    from rheplicant.config.preflight.model import _nodes, _t4_entries

    def labelled(where: str, spec: Any) -> tuple[str, str] | None:
        if not isinstance(spec, Mapping) or spec.get("from") != _A46_ROUTE:
            return None
        label = spec.get("label")
        if not isinstance(label, str) or not label:
            return None
        return (where, label)

    model_loads: list[tuple[str, str]] = []
    spec = _nodes(document).get(_A46_NODE)
    if spec is not None:
        for path, entry in _t4_entries(_A46_NODE, spec, many=True):
            found = labelled(f"model.{path}", entry)
            if found is not None:
                model_loads.append(found)
    replace_loads: list[tuple[str, str]] = []
    inference = document.get("inference")
    twin = inference.get("twin") if isinstance(inference, Mapping) else None
    replace = twin.get("replace") if isinstance(twin, Mapping) else None
    if isinstance(replace, Mapping):
        found = labelled(f"inference.twin.replace.{_A46_NODE}",
                         replace.get(_A46_NODE))
        if found is not None:
            replace_loads.append(found)
    return model_loads, replace_loads


# ---------------------------------------------------------------------------
# The checks.
# ---------------------------------------------------------------------------


@register("A10")
def _freq_unit(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A10: a ``rhino_hdf5`` recording is asked which unit its channels are in.

    The message is ``sections/ingest.py::freq_unit_problem``'s, verbatim and
    by call -- this pass owns no copy of it (§2.2).

    Walks ``inference.twin.replace`` (§0.3 E.10) and every other section but
    ``variants:``, because the route is a ``{file:}`` value node and one can
    be written anywhere a value node can.
    """
    for where, spec in _a10_sites(document):
        problem = freq_unit_problem(spec)
        if problem is not None:
            yield refuse("A10", _task3_where(where), f"{problem} (check A10).")


@register("A45")
def _switch_key(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A45: ``switch_key:`` names a key ``coords.extra`` will actually carry.

    Walks ``inference.twin.replace`` (§0.3 E.10), and that route is the one
    that matters: measured, ``replace.noise_wave.switch_key: nope`` LOADS
    CLEAN and detonates at fit time -- strictly worse than the ``model:``
    route this check was written for.

    ``operator_table()`` is resolved once, at the top: it costs 1.7e-04 s and
    a document with several noise-wave sites would otherwise pay it per site.
    """
    from rheplicant.config.preflight.observing import _a15_declared_class

    written = _a45_written_keys(document)
    if written is None:
        return
    table = operator_table()
    for path, node_id, entry in _a45_sites(document):
        if not isinstance(entry, Mapping) or _A45_FIELD not in entry:
            continue
        if not _a45_carries_switch_key(_a15_declared_class(node_id, entry,
                                                           table)):
            continue
        value = entry[_A45_FIELD]
        if not isinstance(value, str):
            # `deliver`'s static_str rule refuses a non-string with the type
            # it got; saying "is not a key" about a list is the vaguer of the
            # two sentences.
            continue
        if value in written:
            continue
        yield refuse(
            "A45", _task3_where(f"{path}.{_A45_FIELD}"),
            f"{path}.{_A45_FIELD}: {value!r} is not a key this run writes "
            f"into coords.extra, so the operator has no switch index to read "
            f"-- with more than one source the twin refuses the moment it is "
            f"evaluated, and with one it silently takes the first. This run "
            f"writes {sorted(written)}; observation.extra and "
            f"observation.pointing's materialise:/lst: are where another one "
            f"would come from (check A45).")


@register("A46")
def _thermistor_columns(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A46 legs 2 and 3: the column a load's label has, and the one two share.

    **Leg 2 refuses**, leg 3 **warns**.  Leg 3 must not fire on a document
    ``A14.cal_loads`` already refuses, and it does not need a rule of its own
    for that: ``load_document`` calls ``raise_if_refused()`` before
    ``emit_warnings()``, so a refused document never says the warning out
    loud.  Adding a second ordering rule inside the check would be a second
    binding of that same property.

    Stands down entirely when ``observation.from_file`` is absent or is not a
    ``rhino_hdf5`` recording: the first is ``sections/model.py``'s own
    sentence (*"this document declares no observation.from_file"*), which
    names the route to fix, and the second is ``parse_from_file``'s.

    Walks ``inference.twin.replace.cal_loads`` for leg 2 (§0.3 E.10) --
    measured, that route reaches ``cal_load_operators`` and the reader's own
    ``DataIngestionError`` -- and NOT for leg 3, which is about two loads in
    one twin.
    """
    observation = document.get("observation")
    if not isinstance(observation, Mapping):
        return
    from_file = observation.get("from_file")
    if (not isinstance(from_file, Mapping)
            or from_file.get("format") != RHINO_FORMAT):
        return
    columns = from_file.get("thermistor_columns")
    if columns is not None and not isinstance(columns, Mapping):
        # `sections/ingest.py` refuses the shape with the value it got.
        return
    model_loads, replace_loads = _a46_loads(document)
    for where, label in model_loads + replace_loads:
        if columns is None or label not in columns:
            yield refuse(
                "A46", _task3_where(where),
                f"{where}: label: {label!r} has no entry in "
                f"observation.from_file.thermistor_columns, so this load's "
                f"t_load is asked for a column the recording was never read "
                f"with, and the refusal comes from inside the reader naming "
                f"no key of this document. Declare thermistor_columns with a "
                f"column for {label!r} AND for every other switch label the "
                f"recording visits, the antenna included -- the reader "
                f"refuses a partial map (check A46).")
    if not isinstance(columns, Mapping):
        return
    shared: dict[int, set[str]] = {}
    for _, label in model_loads:
        column = columns.get(label)
        if isinstance(column, bool) or not isinstance(column, int):
            continue
        shared.setdefault(column, set()).add(label)
    for column in sorted(shared):
        labels = sorted(shared[column])
        if len(labels) < 2:
            continue
        yield warn(
            "A46", f"model.{_A46_NODE}",
            f"model.{_A46_NODE}: {labels} share column {column} of "
            f"observation.from_file.thermistor_columns, so their load "
            f"operators carry one physical temperature between them and the "
            f"calibration cannot tell those loads apart. Give each load the "
            f"column its own thermistor was recorded in; a column shared "
            f"with a label no load reads -- the antenna -- is legal and the "
            f"file's own map often forces it (check A46).")
