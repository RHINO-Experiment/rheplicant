"""A13's text legs, A40 and A47 -- three rules the instrument states late.

Every rule here is decided from the document's own words: a required field is
present or it is not, a value node's form key is written out, an ``at:`` list
is a list of node ids.  None of them needs a grid, a resource or a constructed
operator, and all three arrive today from somewhere past ``build_resources``.

**What each one costs today, measured in this worktree at ``ea4839b``.**

* **A13's text legs.**  ``line_width`` missing is
  ``sections/model.py::_construct``'s *"model.cw_tone: CWCalibrationOperator
  requires ['line_width']."*, raised inside ``build_model``; the three value
  bounds are ``CWCalibrationOperator.__check_init__``'s
  ``StateValidationError``, raised while the operator is constructed.  Both
  are behind the beam.
* **A40.**  ``config/delivery.py::_refuse_array_form`` is reached from
  ``deliver``, which runs per field inside ``_construct`` -- so a
  ``{linspace: ...}`` written on a static field is refused after the CST
  directory has been read.
* **A47.**  ``paths.py::refuse_misaddressed_region`` is called by
  ``compose._single`` on the line AFTER ``build_node_operator(node_id, spec,
  context)``, so the operator for the very entry that is misaddressed is
  constructed first.

**One binding, and the two places this module deliberately does not have one.**

A40 and A47 are HOISTS in §2.2's exact shape: ``_refuse_array_form(spec,
source)`` and ``refuse_misaddressed_region(config_key, region)`` are already
module-level pure functions taking plain data, so this module CALLS them, on
data read off the raw document, and converts the ``ConfigError`` they raise
into a ``Finding`` -- the ``preflight/document.py::_variant_text`` precedent.
Neither sentence is copied here; ``tests/config/test_preflight_instrument.py``
asserts each is bound in exactly one module under ``src/``.

**A13's text legs are INVENTIONS in this layer's voice, and that is a recorded
departure from the plan's own "hoist" verdict** (plan §0.3 E.8, E.10).  A
legal hoist needs the sentence extracted into a module-level pure function
where it already lives.  For the presence leg that is
``sections/model.py::_construct``, which plan §0.3 D strikes from this task's
Files list because Task 8 edits the same module in the same wave; the ruling's
own remedy -- "import the literal by name" -- has nothing to import, because
the sentence is an inline f-string with no name.  For the three value bounds it
is ``CWCalibrationOperator.__check_init__``, a method on a constructed
operator, in ``radio/instrument/calibration.py``, which was never in the Files
list at all.  So the choice was between copying two sentences (forbidden) and
writing this layer's own (§2.3's "invented" act).

The second reason is the stronger one and it is measured.  ``_construct``'s
message hardcodes ``f"model.{node_id}: "``, and ``inference.twin.replace.
<node>`` reaches ``build_node_operator`` down the same path
(``sections/twin.py:69``).  Measured here: a ``replace.cw_tone`` with
``line_width`` omitted is refused with *"model.cw_tone: CWCalibrationOperator
requires ['line_width']."* -- naming a section the entry is not written in.  A
verbatim hoist would reproduce that one phase earlier, on a route this pass
walks on purpose.  The messages below carry no hardcoded section and say the
right path on both routes.  ``inflight/grids.py`` took the same decision for
A13's grid legs, for the same reason, one task earlier.

**Routes.**  Every check here walks ``model:`` AND ``inference.twin.replace``
(plan §0.3 E.10) except A47, which walks ``model:`` only because
``inference.twin.replace`` does not honour ``at:`` at all -- that is a
non-route rather than a false negative, and :func:`_routes` says why.  All
three run over the base document and over each declared variant merged on top,
through ``preflight/document.py::_task3_over_layers``: ``load_document``
applies no variant unless one is requested (``document.py:77``), so a
``model:`` an unselected variant patches is otherwise never read at P-1.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register

#: The one route besides ``model:`` that reaches ``build_node_operator``.
_REPLACE = "inference.twin.replace"

#: The tail every A13 text leg appends.  Bound once and interpolated rather
#: than restated per leg, which is what stops four near-copies drifting.
_A13_TAIL = (
    "This is decided from the entry's own words, before build_resources reads "
    "the beam. Today it arrives from inside build_model -- the presence rule "
    "from sections/model.py, the value bounds from "
    "CWCalibrationOperator.__check_init__ as a StateValidationError, which is "
    "a SIBLING of ConfigError rather than a subclass, so `except ConfigError` "
    "does not see it either"
)

#: The tail A40 appends after ``delivery.py``'s own sentence.
_A40_TAIL = (
    "That refusal is config/delivery.py's own, moved in front of the build: it "
    "runs inside deliver(), once per field of a node being constructed, so on "
    "a document with a beam it used to arrive after the CST directory had been "
    "read and analysed. The form key is written in the document, and whether "
    "the field is static is eqx.field(...) metadata, so neither needs a value "
    "resolved"
)

#: The tail A47 appends after ``paths.py``'s own sentence.
_A47_TAIL = (
    "That refusal is config/paths.py's own, moved in front of the build: "
    "compose._single calls it on the line AFTER build_node_operator, so the "
    "operator for the very entry that is misaddressed is constructed first, "
    "and on a region covering a sky node that means the projector has already "
    "been built"
)


# ---------------------------------------------------------------------------
# The routes, and the entries on them
# ---------------------------------------------------------------------------


def _routes(document: Mapping[str, Any]) -> list[tuple[str, Mapping, bool]]:
    """``(prefix, node specs, whether composition keys are honoured)``.

    Plan §0.3 E.10: ``preflight/model.py::_nodes`` reads ``document["model"]``
    and nothing else, but ``inference.twin.replace.<node>`` reaches the same
    ``build_node_operator`` (``sections/twin.py:67-69``), so a check walking
    only ``model:`` guards one route of two.

    **The third member is not decoration.**  The ``model:`` route goes through
    ``compose._single``, which pops ``at:`` and ``snapshot_before:`` and
    dispatches ``compose:`` to its own builder; the replace route calls
    ``build_node_operator`` DIRECTLY, so on it ``at:``, ``snapshot_before:``
    and ``compose:`` are none of them honoured -- they are unknown constructor
    fields and ``_construct`` refuses them by name.  A walk that expanded a
    replace entry's ``compose: {stages: [...]}`` into stages would read specs
    that never reach a constructor and could answer about one of them on a
    document refused for the composing key itself.

    A ``replace:`` that is not a mapping is left alone: ``sections/twin.py:62``
    refuses it with the shape it got, and this pass has nothing better to say.
    """
    from rheplicant.config.preflight.model import _nodes

    routes: list[tuple[str, Mapping, bool]] = [("model", _nodes(document), True)]
    inference = document.get("inference")
    twin = inference.get("twin") if isinstance(inference, Mapping) else None
    replace = twin.get("replace") if isinstance(twin, Mapping) else None
    if isinstance(replace, Mapping):
        routes.append((_REPLACE, replace, False))
    return routes


def _entries(document: Mapping[str, Any]) -> Iterable[tuple[str, str, Any, bool]]:
    """``(where, node id, spec, composed)`` per operator this document builds.

    The fourth member is :func:`_routes`' third, carried to the stand-down that
    needs it.  It is not a convenience: ``_unknown_field`` cannot decide
    whether ``at:`` is an unknown key without knowing which route the entry
    came down, and reading the document again to find out is a second binding
    of the same question.

    ``_t4_entries`` is what expands the two shapes a single key can hold
    several operators in: a ``many`` node's list or FAN mapping, and a
    ``compose:`` block's stages.  Plan §0.3 E.10 names it as the helper the
    naive ``_static_fields(document)`` skips, and the count is four routes to
    one -- ``compose: stages``, the single/``at:`` route, a ``many`` node's
    list and its FAN entries.

    The node id yielded is the KEY, not where the operator lands, because that
    is what the build looks the class up under: every ``build_node_operator``
    call site in ``compose.py`` passes the key (or a FAN label), and
    ``_pick_class`` does ``operator_table().get(node_id)`` with it.  Where the
    operator ends up is ``_t5_claims``' question and only A47 asks it.
    """
    from rheplicant.config.preflight.model import _t4_entries, _t4_graph

    graph = _t4_graph()
    for prefix, specs, composed in _routes(document):
        for key, spec in specs.items():
            if not isinstance(key, str):
                continue
            node = graph.nodes.get(key)
            if composed:
                pairs = _t4_entries(key, spec, many=bool(node and node.many))
            else:
                pairs = [(key, spec)]
            for path, entry in pairs:
                yield f"{prefix}.{path}", key, entry, composed


def _entry_class(node_id: str, entry: Any, table: Mapping[str, Any]) -> type | None:
    """The operator class this entry constructs, or ``None`` to stand down.

    ``table`` is ``operator_table()``, passed IN rather than called here.  It
    is 1.2e-05 s warm and a walk calls this once per entry per layer, so on
    the twenty-variant document the shipped cold-cost guard measures it is the
    difference between two calls and two hundred.  There is no module-level
    memo for the same reason ``operator_table`` has none of its own: it is
    discovered live off ``rheplicant.radio.__all__``, and a cache here would
    be this module deciding that nothing may ever add a class at runtime.

    Three routes, in ``build_node_operator``'s own order.  A ``python:`` entry
    resolves through ``preflight/model.py::_t5_radio_class``, which imports
    nothing and declines a foreign target -- §2.4's boundary, and the decline
    can only lose a check, never invent one.  A ``type:`` entry is matched
    against the classes ``operator_table()`` registers at the node, so a
    ``type:`` naming none of them is left to A7 rather than answered here.  A
    bare entry resolves only when the node registers exactly one class; where
    it registers several and the spec names none, A7's *"N classes register at
    this node; type: is required"* is the better sentence and this stands
    down.

    ``from:`` returns ``None`` outright.  ``_from_route`` does not call
    ``_construct``: it hand-picks one or two fields per route and calls
    ``_field_value`` on them, so the class-plus-field walk below does not
    describe it.  Its three routes carry at most one value node each
    (``beam_spill.t_ground``, ``t_sys_extra.coeff``), and both are TRACED
    fields -- measured with ``mode_of`` -- so no A40 finding is lost today.
    Recorded as a declared restriction rather than guessed at: covering it
    would mean restating ``_from_route``'s routing table here, which is the
    second-binding shape §2.2 exists to stop.
    """
    from rheplicant.config.preflight.model import _t5_radio_class

    if not isinstance(entry, Mapping):
        return None
    if "python" in entry:
        return _t5_radio_class(entry)
    if "from" in entry:
        return None
    classes = table.get(node_id) or ()
    declared = entry.get("type")
    if declared is not None:
        return next((cls for cls in classes if cls.__name__ == declared), None)
    return classes[0] if len(classes) == 1 else None


def _unknown_field(entry: Mapping, specs: Mapping[str, Any],
                   composed: bool) -> bool:
    """Does ``_construct`` refuse this entry for a key before it reads a value?

    ``sections/model.py:151-156`` sweeps unknown keys BEFORE it looks for
    missing required ones and long before it delivers a value, so on an entry
    carrying a typo the reader gets *"does not take ['nope']"* and nothing
    from this module should pre-empt it.

    **``composed`` decides which keys are unknown, and getting that wrong is
    not a nicety -- it was a live advice loop in this module's first commit.**
    ``compose._single`` pops ``at:`` and ``snapshot_before:`` before
    ``_construct`` sees the spec (``compose.py:287-288``), so on the ``model:``
    route they are not unknown and an entry carrying one must still be read
    here.  ``sections/twin.py:67`` calls ``build_node_operator`` DIRECTLY, so
    on the replace route they ARE unknown -- measured, ``replace.cw_tone: {at:
    [...]}`` is refused with *"does not take ['at']"*, naming the key the user
    wrote.  With the exemption applied unconditionally this module answered
    A13 there instead, and then **applying A13's own remedy left the document
    refused anyway** (R4): declare ``line_width`` and ``does not take ['at']``
    arrives regardless.  That is the drift §2.2 predicts when one rule has two
    validators, and it appeared in the first commit.
    """
    from rheplicant.config.sections.model import _NODE_KEYS

    popped = {"at", "snapshot_before"} if composed else frozenset()
    return bool(set(entry) - set(specs) - set(_NODE_KEYS) - popped)


# ---------------------------------------------------------------------------
# A13's text legs
# ---------------------------------------------------------------------------


def _tone_class():
    """``CWCalibrationOperator``, imported where it is used.

    ``preflight/model.py::_t4_graph``'s convention, and for its reason:
    ``import rheplicant.config`` already imports ``rheplicant.radio``
    (``config/kinds/projectors.py:44``), so the import costs nothing at call
    time, and deferring it means this module is not the one that pins that.
    """
    from rheplicant.radio.instrument.calibration import CWCalibrationOperator

    return CWCalibrationOperator


def _text_number(node: Any) -> float | None:
    """The number a static scalar field's value node carries, from TEXT ALONE.

    ``None`` means "this pass cannot read it", which is §3.2(c)'s stand-down
    and not a verdict: ``{ref: ...}``, ``{from: ...}`` and every array form
    need a resolved context, and ``resolve_value`` is on ``preflight/``'s own
    banned list (``test_config_preflight.py::_OUT_OF_SCOPE_NAMES``) precisely
    so a check cannot reach for one.  Standing down is safe in both directions
    here: every field this is read for is ``eqx.field(static=True)``, and
    ``config/delivery.py`` refuses a non-``numbers.Real`` for one by name.

    The three readable spellings are the three that carry their number in the
    text -- a bare number, the ``"<number> <unit>"`` shorthand, and
    ``{value: N}`` with an optional ``unit:``.  **The unit is applied rather
    than ignored**, through the layer's own :func:`convert_to_canonical`, so
    this reads the number ``deliver`` would see: a unit may be affine
    (``degC`` converts with an offset of 273.15), so dropping it can change a
    sign as well as a magnitude.  ``_SHORTHAND`` is imported rather than
    re-written for the same one-binding reason.

    A ``bool`` is declined at both spellings.  ``isinstance(True, int)`` is
    True, so a bare ``true`` would arrive here as 1.0 and read as a legal
    width; ``delivery.py::_as_static_float`` has a clause of its own naming
    that, and this leaves the sentence to it.
    """
    from rheplicant.config.units import convert_to_canonical
    from rheplicant.config.values import _SHORTHAND, VALUE_FORMS

    token: str | None = None
    if isinstance(node, Mapping):
        # `_static_number`'s own test in `inflight/grids.py`: exactly the
        # `value` form and no other, so a `{value: ..., linspace: ...}` node is
        # left to `resolve_value`'s "holds 2 form keys" rather than half-read.
        # `as:` declines too -- `deliver` reads it BEFORE it delivers anything,
        # so a mode the field contradicts is refused there and this pass would
        # be answering about a value that never arrives.
        if set(node) & set(VALUE_FORMS) != {"value"} or "as" in node:
            return None
        raw = node["value"]
        token = node.get("unit")
        if token is not None and not isinstance(token, str):
            return None
    elif isinstance(node, str):
        match = _SHORTHAND.match(node)
        if match is None:
            return None
        raw, token = float(match.group("number")), match.group("unit")
    else:
        raw = node
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None
    if token is None:
        return float(raw)
    try:
        converted, _ = convert_to_canonical(raw, token)
    except ConfigError:
        return None  # the value grammar says what an unusable unit token is
    return float(converted)


def _tone_legs(where: str, entry: Mapping, cls: type,
               specs: Mapping[str, Any], composed: bool) -> Iterable[Finding]:
    """A13's four text legs on one ``CWCalibrationOperator`` entry."""
    from rheplicant.radio.instrument.calibration import LINESHAPES

    if _unknown_field(entry, specs, composed):
        return
    missing = sorted(name for name, spec in specs.items()
                     if spec.required and name not in entry)
    if missing:
        yield refuse(
            "A13", where,
            f"{where}: a CW calibration tone declares {missing} nowhere, and "
            f"{cls.__name__} gives {'them' if len(missing) > 1 else 'it'} no "
            "default on purpose: line_width is the spectrometer's own channel "
            "response in Hz, tone_freq is where the line sits in the band, and "
            "amplitude is the level the tone contributes in total. A tone the "
            "operator has to guess one of those for monitors nothing, because "
            f"the gain it is meant to track absorbs it exactly. {_A13_TAIL} "
            "(check A13).")
        return
    width = _text_number(entry.get("line_width"))
    if width is not None and width <= 0:
        yield refuse(
            "A13", f"{where}.line_width",
            f"{where}.line_width: {width:.6g} Hz is not above zero. The width "
            "is a scale, not an offset: it divides the frequency offset that "
            "the lineshape is evaluated at, so zero divides by zero and a "
            "negative value evaluates the shape mirrored about the centre "
            f"before it is normalised. {_A13_TAIL} (check A13).")
    floor = _text_number(entry.get("protect_floor"))
    if floor is not None and not 0.0 < floor <= 1.0:
        yield refuse(
            "A13", f"{where}.protect_floor",
            f"{where}.protect_floor: {floor:.6g} is outside (0, 1]. It is read "
            "as a fraction of the tone's own peak channel, so 1 protects the "
            "peak channel alone and anything at or below 0 protects the whole "
            "band -- which hands every channel's RFI verdict to a calibrator "
            "that touches one line. Above 1 protects nothing at all and the "
            "flagger then eats the tone it was told to keep. "
            f"{_A13_TAIL} (check A13).")
    shape = entry.get("lineshape")
    if isinstance(shape, str) and shape not in LINESHAPES:
        yield refuse(
            "A13", f"{where}.lineshape",
            f"{where}.lineshape: {shape!r} is not a lineshape this operator "
            f"evaluates; it takes {list(LINESHAPES)}. The two are not "
            "interchangeable spellings of one curve and line_width does not "
            "mean the same thing in each -- for 'sinc2' it is the offset to "
            "the first null of a critically sampled unwindowed FFT, and for "
            "'gaussian' it is the standard deviation of an apodised polyphase "
            f"channel. {_A13_TAIL} (check A13).")


def _tone_text(layer: Mapping[str, Any]) -> Iterable[Finding]:
    """A13's text legs: the tone's required fields, and its three value bounds.

    A13's OTHER legs -- the width against the channel spacing and the band,
    and the drifted centre -- need the resolved frequency and time grids and
    are ``inflight/grids.py``'s ``A13.grid``.  This slot is ``A13.text`` and
    ``Finding.check`` is the bare ``"A13"`` in both, which is 3A's
    ``A1.runs``/``A1.variants`` precedent.

    **Walks ``model:`` AND ``inference.twin.replace``** (plan §0.3 E.10), and
    says the right section on each, which the shipped sentences do not:
    measured, a ``replace.cw_tone`` missing ``line_width`` is refused today
    with *"model.cw_tone: CWCalibrationOperator requires ['line_width']."*

    **And it is keyed on the CLASS ALONE, not on the ``cw_tone`` KEY.**  That
    is not symmetry for its own sake: the ``python:`` hatch places a
    ``CWCalibrationOperator`` at any node and such a document BUILDS, so
    ``model.bandpass: {python: '...:CWCalibrationOperator', line_width:
    -1 MHz}`` was -- measured -- still a ``StateValidationError`` from behind
    the beam, which is the exact row this task exists to move.  Neither
    ``A13.grid`` nor an earlier draft of this check walked it.  A40 has never
    had a node-id gate and always covered that document; A13 now agrees with
    it, in one file.

    **Two stand-downs are load-bearing and each has a test.**

    ``tone_freq <= 0`` is NOT a leg here, although the operator guards it
    (``calibration.py:373``).  Measured at ``ea4839b``, a negative
    ``tone_freq`` never reaches that guard: ``A13.grid`` answers first with
    *"the tone centre spans [...] outside this run's observed band [...]"*,
    which names the band the user has to put the line in.  A P-1 leg would run
    one phase earlier still and replace a sentence carrying the real band with
    one saying only that a frequency is negative.

    An entry with an unknown field is left to ``_construct``'s *"does not take
    [...]"*, which is what the reader sees today and which sweeps keys before
    it looks for missing ones.

    **The CLASS, not the ``type:`` token**, which is the same hole plan §0.3
    E.9 closes for A45: ``python: 'rheplicant.radio.instrument.calibration:
    CWCalibrationOperator'`` and ``type: CWCalibrationOperator`` are two
    spellings of one class object, and 3A's tests already exercise the
    ``python:`` one.  The other polarity matters as much and is easier to get
    wrong: a ``cw_tone: {python: 'rheplicant.radio:GainOperator'}`` entry is
    written under the tone's key and is NOT a tone, so giving it the tone's
    legs would refuse a document that builds.  :func:`_entry_class` answers
    both, so ``inflight/grids.py::_is_tone`` is not imported: ``preflight/``
    and ``inflight/`` are two passes above ``sections/`` and neither imports
    the other (plan §0.3 C.2's reason for putting ``resolved_specs`` in
    ``config/resources.py``).
    """
    from rheplicant.config.delivery import field_specs
    from rheplicant.config.sections.model import operator_table

    tone = _tone_class()
    specs = field_specs(tone)
    table = operator_table()
    for where, node_id, entry, composed in _entries(layer):
        if _entry_class(node_id, entry, table) is tone:
            yield from _tone_legs(where, entry, tone, specs, composed)


# ---------------------------------------------------------------------------
# A40
# ---------------------------------------------------------------------------


def _array_form(node: Any) -> str | None:
    """The array form key this value node writes, or ``None``.

    ``ARRAY_FORMS`` is ``delivery.py``'s own set and is exactly what
    ``deliver`` gates A40 on, so this asks the same question the build asks.
    The five forms OUTSIDE it -- measured, ``set(VALUE_FORMS) - ARRAY_FORMS ==
    {'from', 'from_switch_order', 'python', 'ref', 'value'}`` -- can every one
    of them deliver an array too, and four of them are a declared false
    negative here (plan §7): ``{ref:}`` and ``{from:}`` both arrive as a jax
    ``ArrayImpl`` and earn ``_as_static_float``'s type complaint instead of
    A40's treedef-and-jit-cache-key explanation, and neither can be told from
    text without resolving it.  ``{value: N}`` is not a false negative at all:
    it resolves with ``source == "scalar"``, so the build does not call A40 on
    it either and this agrees with the build.

    Exactly one form key, because that is the only shape ``resolve_value``
    accepts: zero and several are its own refusals (*"holds no form key"*,
    *"holds N form keys"*) and pre-empting either with A40 would name a
    consequence instead of the fault.
    """
    from rheplicant.config.delivery import ARRAY_FORMS
    from rheplicant.config.values import VALUE_FORMS

    if not isinstance(node, Mapping):
        return None
    forms = [key for key in node if key in VALUE_FORMS]
    if len(forms) != 1:
        return None
    return forms[0] if forms[0] in ARRAY_FORMS else None


def _a40_stands_down(node: Mapping, mode: str) -> bool:
    """Does ``deliver`` refuse this value node BEFORE it reaches A40?

    ``delivery.py:187-204`` reads ``as:`` first and the array-form gate
    second, so a document that declares a delivery mode the field contradicts
    -- or one that is not a delivery mode at all -- earns a sentence about its
    own claim.  Measured, ``{linspace: ..., as: traced}`` on ``line_width``
    gives *"This value declares as='traced', but field 'line_width' is
    'static_float'"*, and ``as: banana`` gives *"as='banana' is not a delivery
    mode"*.  Both name the key the reader wrote; A40 would name a consequence
    of it.
    """
    from rheplicant.config.delivery import DELIVERY_MODES

    declared = node.get("as")
    if declared is None:
        return False
    return declared not in DELIVERY_MODES or declared != mode


def _static_fields(layer: Mapping[str, Any]) -> Iterable[Finding]:
    """A40: an array-producing value node written on a static field.

    A HOIST.  ``config/delivery.py::_refuse_array_form(spec, source)`` is
    already the module-level pure function §2.2 asks for, so this calls it and
    converts, and the sentence keeps its single binding in ``delivery.py``.

    **Every static field of every node, not only the tone's** -- the walk is
    ``_entries`` over both routes, so a ``many`` node's third filter and a
    ``compose:`` block's second stage are each reached at their own path.
    Measured with ``field_specs`` + ``mode_of`` over all 28 shipped operator
    classes: 28 fields are static (15 float, 6 int, 5 str, 1 tuple, 1
    mapping), and **zero** are ``static_bool``, which is why ``mode_of``'s
    identity-versus-``isinstance`` trap has no document that can reach it.

    Walks ``model:`` and ``inference.twin.replace`` (plan §0.3 E.10); measured,
    ``replace.cw_tone.line_width: {linspace: ...}`` is A40 on that route too,
    and ``delivery.py``'s sentence names no section, so it reads correctly on
    both.

    **What it does not decide.**  A ``static_tuple`` or ``static_mapping``
    field given a ``{ref:}`` -- outside ``ARRAY_FORMS``, so outside the rule
    this hoists -- still fails later, and measured it fails as a BARE
    ``TypeError`` (*"unhashable type: 'jaxlib._jax.ArrayImpl'"* and *"cannot
    convert dictionary update sequence element #0 to a sequence"*).  Neither
    is a ``DirtError``, so neither is catchable by name; widening A40 to the
    five non-array forms needs them resolved and is recorded in the plan's §7
    rather than guessed at here.
    """
    from rheplicant.config.delivery import _refuse_array_form, field_specs, mode_of
    from rheplicant.config.sections.model import operator_table

    table = operator_table()
    # `field_specs` is 2.5e-05 s and reads `typing.get_type_hints`, so it is
    # memoised per LAYER rather than called per entry: a `many` node's chain
    # of five filters is five entries of one class.  Per layer and not at
    # module scope, for `_entry_class`'s reason -- no state outlives the call.
    by_class: dict[type, Mapping[str, Any]] = {}
    for where, node_id, entry, composed in _entries(layer):
        cls = _entry_class(node_id, entry, table)
        if cls is None:
            continue
        specs = by_class.get(cls)
        if specs is None:
            specs = by_class[cls] = field_specs(cls)
        if _unknown_field(entry, specs, composed):
            continue
        for name, node in entry.items():
            spec = specs.get(name)
            if spec is None:
                continue
            mode = mode_of(spec)
            form = _array_form(node)
            # The form FIRST: it is the only test that answers on a node of
            # any shape, and `_a40_stands_down` reads `as:` off a mapping.  A
            # bare number reaching that would be `AttributeError` inside a
            # check, which `sweep` turns into "check 'A40' RAISED ..." -- one
            # wrong finding traded for every finding on the document.
            if form is None or mode == "traced" or _a40_stands_down(node, mode):
                continue
            try:
                _refuse_array_form(spec, form)
            except ConfigError as exc:
                yield refuse("A40", f"{where}.{name}",
                             f"{where}.{name}: {exc} {_A40_TAIL} (check A40).")


# ---------------------------------------------------------------------------
# A47
# ---------------------------------------------------------------------------


def _region_key(layer: Mapping[str, Any]) -> Iterable[Finding]:
    """A47: a multi-node ``at:`` region written under a key that is not its last.

    A HOIST.  ``config/paths.py::refuse_misaddressed_region(config_key,
    region)`` is already a module-level pure function taking two plain
    arguments, so this calls it and converts.

    **Decided through ``preflight/model.py::_t5_claims``, never off a raw
    ``at:`` list** -- §2.3's first named stand-down, and the reason is that
    ``compose._single`` refuses three other shapes more precisely and
    ``_t5_claims`` already answers ``()`` for all three: an ``at:`` with no
    ``python:``, an ``at:`` beside ``snapshot_before:``, and the STRING
    spelling that must restate its own key.  A P-1 A47 reading ``model.<n>.at``
    directly pre-empts all three.

    **The string spelling cannot be guarded, and that is a stand-down rather
    than a gap.**  ``refuse_misaddressed_region`` returns early below two
    nodes (``paths.py:318-319``), and ``_t5_claims`` answers a string ``at:``
    with either ``()`` (it disagrees with its key, which is ``_single``'s own
    *"a single-node at: restates its own key"*) or a one-tuple (it agrees), so
    the shipped refusal is unreachable through a string and forcing it would
    mean answering where ``_single`` answers better.  The same early return is
    why ``{python: ..., at: ['gain']}`` is legal and stays so.

    **Walks ``model:`` only, and ``inference.twin.replace`` is a NON-route
    rather than a false negative** (plan §0.3 E.10).  ``sections/twin.py:67``
    calls ``build_node_operator`` directly, bypassing ``compose._single``,
    which is the only place ``at:`` is honoured -- so on that route ``at:`` is
    an unknown constructor field and ``_construct`` refuses it by name.
    Measured, and pinned by
    ``test_an_at_region_under_twin_replace_is_not_A47_at_all``.

    ``at:`` on a ``many`` node is a switch LABEL and not a relocation, which
    ``_t5_placement`` answers before the shape is asked about.
    """
    from rheplicant.config.paths import refuse_misaddressed_region
    from rheplicant.config.preflight.model import _nodes, _t5_claims

    for key, spec in _nodes(layer).items():
        if not isinstance(key, str):
            continue
        try:
            refuse_misaddressed_region(key, _t5_claims(key, spec))
        except ConfigError as exc:
            yield refuse("A47", f"model.{key}",
                         f"model.{key}: {exc} {_A47_TAIL} (check A47).")


# ---------------------------------------------------------------------------
# The one registration
# ---------------------------------------------------------------------------


#: The three rules, in the order a reader meets them: what the entry must
#: declare, what it may not declare an array for, and where it may be written.
_RULES = (_tone_text, _static_fields, _region_key)


@register("A13.text", "A40", "A47")
def _instrument_text(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A13's text legs, A40 and A47 -- three rules, one walk over the layers.

    **One function for three slots.**  ``sweep`` de-duplicates by function
    identity precisely so that a function may carry several ids, and every pass
    already does it -- ``preflight/fitting.py``'s ``_blocks`` is A16-A19 and
    ``_prior_gates`` is A20/A21/A23.

    **The cost argument that used to be written here is DEAD, and the table it
    carried was falsified rather than merely aged.**  It said the saving was
    ``_task3_over_layers`` calling ``apply_variant`` -- a ``deepcopy`` -- "once
    per declared variant PER REGISTERED CHECK THAT LAYERS", and tabulated
    22.6 / 31.0 / **42.7 best, 78.3 median** ms for none / one / three
    registrations, concluding that three registrations were already over the
    50 ms budget on a quiet box.  The wave-boundary fix memoised
    ``_task3_layers``: the layers are built once per PASS and shared, so the
    stated mechanism no longer exists.  Re-measured on the same guard's own row
    -- forty runs, twenty variants, fresh processes, these three rules split
    onto three separately registered functions:

    ====================================  ==================  ========
    registration                          cold                merges
    ====================================  ==================  ========
    these three ids on ONE function       12.83-13.54 ms      21
    the same three registered separately  12.74-13.97 ms      21
    ====================================  ==================  ========

    Ten layer walks become twelve and the findings are identical; the
    difference in time is inside the run-to-run spread, and **neither shape
    changes the merge count**.  So the one-function form is no longer load
    bearing for cost.  It is kept because it is still the honest shape -- three
    rules that share one walk, each keeping the name §3.1 pins for it -- and
    because splitting it would be churn in a merged task's checks for no
    measured gain.  A future task that wants them apart may do it; it should
    re-take these numbers rather than trust them.

    The three rules keep the three names plan §3.1 pins for them and each
    still decides one check; what is shared is the walk, not the rule.
    ``Finding.check`` is the bare id in every case, so a reader of the report
    cannot tell -- and a test asserts each id independently through
    ``only(document, ...)``.
    """
    return [finding for rule in _RULES for finding in rule(document)]
