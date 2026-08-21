"""Project one lit single-slot node's settings as labelled, typed controls.

The raw JSON textarea in the node inspector is not replaced by anything here.
It is rendered for every editable node, always, and stays the only place a
value form this module cannot represent can be written. What this adds is a
second, narrower view of the same settings for the twenty nodes that hold
exactly one operator instance: one control per field, carrying the field's
own dimension, unit spellings, enum members and required flag.

Two rules shape everything below.

**Nothing is re-derived.** The fields, their labels, dimensions, units,
choices and defaults are grouped out of the widget catalog, which is already
built live from the config registries. Asking ``field_specs`` again here
would be a second census that could disagree with the first, and the one that
drifts is the one nothing runs.

**Nothing is resolved.** A ``python:`` target names a module to import; a
``from:`` route names a constructor to call. Drawing a form is not a reason
to do either, so both are gates rather than features. This module imports
nothing from the config layer directly -- ``tests/config/test_config_surface``
keeps that list at five files -- and reaches its vocabulary through
``gui/form_catalog.py``, which is one of them.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Literal

from rheplicant.gui.form_catalog import (
    SHORTHAND,
    VALUE_FORMS,
    VALUE_MODIFIERS,
    operator_table,
)
from rheplicant.gui.forms import FormCatalog, WidgetMetadata
from rheplicant.radio.graph import RADIO_GRAPH

#: Distinguishes "the key is not written" from "the key is written as null".
#: ``settings.get(name)`` cannot: both come back ``None``, and only one of
#: them is an empty control the user may fill in.
_ABSENT = object()

#: The keys a quantity control can round-trip, and nothing else. One key
#: beyond these -- a ``scale:``, an ``as:`` -- and the control would have to
#: either drop it or carry it invisibly, so it declines instead.
_ENVELOPE = frozenset({"value", "unit"})

#: Settings keys that are never operator fields.
_TYPE_KEY = "type"

Control = Literal["quantity", "integer", "text", "select", "toggle", "opaque"]


@dataclasses.dataclass(frozen=True, slots=True)
class ValueReading:
    """What one written value is, and the pieces a control can edit.

    ``number`` and ``unit`` are filled in only for the three shapes a typed
    control can represent; every other form reports them as ``None`` rather
    than guessing, because a control that half-understands a value is the one
    that overwrites the half it did not.
    """

    form: str
    number: int | float | None
    unit: str | None


@dataclasses.dataclass(frozen=True, slots=True)
class NodeField:
    """One operator field, as a control and as it is currently written."""

    name: str
    path: str
    label: str
    control: Control
    required: bool
    has_default: bool
    default: object
    dimension: str | None
    unit_policy: str | None
    units: tuple[str, ...]
    choices: tuple[str, ...]
    delivery: str | None
    #: True when the control can represent what is written and may therefore
    #: replace it. False leaves the field read-only and sends the user to the
    #: textarea -- the value is a shape this release has no control for.
    typed: bool
    present: bool
    form: str
    number: int | float | None
    unit: str | None
    written: object


@dataclasses.dataclass(frozen=True, slots=True)
class NodeFieldSet:
    """One node's typed view, or the reason it does not have one."""

    node_id: str
    typed_form: bool
    typed_form_reason: str | None
    type_choices: tuple[str, ...]
    selected_type: str | None
    fields: tuple[NodeField, ...]
    #: Written keys this release has no control for. They do NOT disable the
    #: typed form: ``snapshot_before:`` and ``eqx_leaves:`` are supported keys
    #: with their own existing controls, and refusing the whole form over one
    #: of them would take the typed fields away from every node that uses it.
    extra_keys: tuple[str, ...]
    #: Each candidate type -> the written keys it would leave nowhere to live.
    #: Computed here rather than in the browser because the catalog is what
    #: knows which key belongs to which class, and a confirmation that names
    #: the wrong keys is worse than no confirmation at all.
    removed_by_type: dict[str, tuple[str, ...]]


def classify_value(value: object) -> ValueReading:
    """Read one written value node as a form, a number and a unit.

    Only three shapes are typed: a bare scalar, the ``<number> <unit>``
    shorthand, and the exact ``{value, unit}`` envelope. Everything else --
    another value form, a form carrying a modifier, or a shape the grammar
    does not recognise at all -- reports what it is and no pieces, so the
    caller can leave it to the textarea.

    The final ``unknown`` is the safety net. Silently rewriting a shape this
    module did not recognise into the envelope would be a data loss that
    looks like an edit.
    """
    if value is None:
        return ValueReading("null", None, None)
    # Before the numeric branch: ``bool`` IS ``int``, so ``false`` would
    # otherwise arrive as the number 0, which reads as a legal setting
    # everywhere downstream. ``config/values.py`` keeps a branch here for the
    # same reason and says so at length.
    if isinstance(value, bool):
        return ValueReading("bare", None, None)
    if isinstance(value, (int, float)):
        return ValueReading("bare", value, None)
    if isinstance(value, str):
        found = SHORTHAND.match(value)
        if found is None:
            return ValueReading("bare", None, None)
        return ValueReading("shorthand", float(found["number"]), found["unit"])
    if not isinstance(value, Mapping):
        return ValueReading("unknown", None, None)
    keys = set(value)
    if "value" in keys and keys <= _ENVELOPE:
        number = value["value"]
        unit = value.get("unit")
        if isinstance(number, bool) or not isinstance(number, (int, float)):
            return ValueReading("value", None, None)
        if unit is not None and not isinstance(unit, str):
            return ValueReading("value", None, None)
        return ValueReading("quantity", number, unit)
    forms = [key for key in VALUE_FORMS if key in keys]
    if len(forms) == 1 and keys <= {forms[0], *VALUE_MODIFIERS}:
        return ValueReading(forms[0], None, None)
    return ValueReading("unknown", None, None)


def _removed_by_type(
    *, written: Sequence[str], current: frozenset[str], candidate: frozenset[str]
) -> tuple[str, ...]:
    """The written keys that ``candidate`` has no field for.

    A key the current class does not own either -- ``snapshot_before:``, an
    unknown key -- belongs to the node rather than to its operator, so a
    change of class has no business removing it. A key BOTH classes own
    survives, which is why the candidate's fields are subtracted rather than
    the whole intersection being taken: no node this release can reach has
    such a field, but all three filters own ``mode``, so the case is real and
    arrives with many-node cards.
    """
    return tuple(key for key in written if key in current and key not in candidate)


def _control(widget: WidgetMetadata) -> Control:
    """Which control edits this field.

    Delivery decides first and disposition second, which is why a
    ``static_str`` that is also structural -- ``lineshape``, ``switch_key`` --
    gets a select or a text box rather than the opaque fallback: how a value
    is DELIVERED is what says whether a control can produce it.

    ``toggle`` is unreachable today. The live census of all 66 model fields is
    38 traced, 15 ``static_float``, 6 ``static_int``, 5 ``static_str``, 1
    ``static_mapping``, 1 ``static_tuple`` and ZERO ``static_bool``. It is
    declared so the first boolean field gets a checkbox rather than silently
    falling through to a quantity control it cannot fill.
    """
    if widget.delivery == "static_bool":
        return "toggle"
    if widget.delivery == "static_int":
        return "integer"
    if widget.delivery == "static_str":
        return "select" if widget.choices else "text"
    if widget.delivery in ("static_mapping", "static_tuple"):
        return "opaque"
    if widget.dimension in ("structural", "open"):
        return "opaque"
    return "quantity"


def _typed(control: Control, reading: ValueReading, written: object) -> bool:
    """May this control replace what is written?

    An absent field is always typed -- an empty control is exactly what a
    user needs to fill it in. Otherwise the control has to be able to
    represent the shape that is already there, and the written Python type
    decides that for a bare scalar: a bare number belongs to a numeric
    control, a bare string to a select or a text box, and a bare bool to a
    toggle that does not exist yet.
    """
    if reading.form == "absent":
        return True
    if reading.form in ("quantity", "shorthand"):
        return control == "quantity"
    if reading.form != "bare":
        return False
    if isinstance(written, bool):
        return control == "toggle"
    if isinstance(written, (int, float)):
        return control in ("quantity", "integer")
    if isinstance(written, str):
        return control in ("select", "text")
    return False


def _class_of(selector: str) -> str:
    """The class name in a ``<module>.<Class>.<field>`` dimension selector.

    Compared against ``cls.__name__``, which is what the ``type:`` select
    offers and what a document writes -- so a nested class whose qualname
    carries a dot still matches on its last component.
    """
    return selector.split(".")[-2]


def _class_fields(node_id: str, class_name: str, catalog: FormCatalog) -> tuple[str, ...]:
    """Which field names one class owns at this node, read off the catalog."""
    return tuple(
        widget.path.rsplit(".", 1)[-1]
        for widget in _field_widgets(node_id, catalog)
        if any(_class_of(source.selector) == class_name for source in widget.sources)
    )


def _field_widgets(node_id: str, catalog: FormCatalog) -> tuple[WidgetMetadata, ...]:
    prefix = f"model.{node_id}."
    return tuple(
        widget
        for widget in catalog.widgets
        # Exactly three segments: a node's fields are its immediate children,
        # and `model.*.eqx_leaves` is a different widget that shares the root.
        if widget.path.startswith(prefix) and widget.path.count(".") == 2
        and widget.path.rsplit(".", 1)[-1] not in (_TYPE_KEY, "eqx_leaves")
    )


def _refused(node_id: str, reason: str, type_choices: tuple[str, ...] = ()) -> NodeFieldSet:
    return NodeFieldSet(
        node_id=node_id,
        typed_form=False,
        typed_form_reason=reason,
        type_choices=type_choices,
        selected_type=None,
        fields=(),
        extra_keys=(),
        removed_by_type={choice: () for choice in type_choices},
    )


def project_node_fields(
    node_id: str, settings: object, catalog: FormCatalog
) -> NodeFieldSet:
    """Project one node's settings as typed controls, or say why not.

    The gates are ordered and the first failure supplies the reason, so each
    one may assume the ones above it passed. Gate 5 is a security gate: it
    refuses to resolve a ``python:`` target, and everything after it would
    have to import the module the document names in order to disagree.

    Args:
        node_id: a node of :data:`~rheplicant.radio.graph.RADIO_GRAPH`.
        settings: that node's settings exactly as written, or ``None``.
        catalog: a built widget catalog. Build it ONCE per projection pass --
            it is the whole census, and one per node would rebuild it 33
            times for one document.
    """
    spec = RADIO_GRAPH.nodes.get(node_id)
    if spec is None:
        raise KeyError(f"{node_id!r} is not a node of the shipped graph")

    # 1. A junction adds its live branches and a selector switches between
    #    them; neither holds an operator, so neither has fields.
    if spec.kind in ("junction", "selector"):
        return _refused(node_id, "Automatic junction or selector: not an operator slot.")
    # 2. Ahead of gate 8 on purpose: a reserved node also has no operator, and
    #    "reserved" is the more useful of the two true sentences.
    if spec.reserved:
        return _refused(
            node_id,
            "Reserved graph slot with no shipped operator; configure it through python:.",
        )
    # 3. Asked of the GRAPH, not of the settings. `cal_loads` is a label-keyed
    #    mapping, so a shape test alone would pass it through and then read
    #    its LABELS as field names.
    if spec.many:
        return _refused(node_id, "Many node: one card per instance is not in this release.")
    if settings is not None and not isinstance(settings, Mapping):
        return _refused(node_id, "Node settings are not a mapping; edit them as YAML.")

    written: Mapping[str, object] = settings if isinstance(settings, Mapping) else {}
    if "compose" in written:
        return _refused(node_id, "Composed node: the stages own the fields.")
    # 5. THE SECURITY GATE. Ahead of every remaining key, including `type:`,
    #    so the order the document happened to write them in cannot change
    #    the answer.
    if "python" in written:
        return _refused(node_id, "python: target; its class is not resolved in the browser.")
    if "at" in written:
        return _refused(node_id, "Region placement: at: requires python:.")
    if "from" in written:
        return _refused(
            node_id,
            f"from: {written['from']} is a constructor route, not a field set.",
        )

    classes = operator_table().get(node_id, ())
    if not classes:
        return _refused(
            node_id, "No shipped operator registers at this node; python: is the route."
        )

    type_choices = tuple(cls.__name__ for cls in classes)
    declared = written.get(_TYPE_KEY)
    if isinstance(declared, str):
        # A written type that names no registered class selects nothing: the
        # config layer refuses the document, and showing one class's fields
        # under another class's name would be the wrong half of that answer.
        selected = declared if declared in type_choices else None
    else:
        selected = type_choices[0] if len(type_choices) == 1 else None

    fields = (
        tuple(
            _project_field(widget, written)
            for widget in _field_widgets(node_id, catalog)
            if any(_class_of(source.selector) == selected for source in widget.sources)
        )
        if selected is not None
        else ()
    )
    owned = frozenset(field.name for field in fields)
    named = owned | {_TYPE_KEY}
    by_class = {
        cls.__name__: frozenset(
            _class_fields(node_id, cls.__name__, catalog)
        )
        for cls in classes
    }
    return NodeFieldSet(
        node_id=node_id,
        typed_form=True,
        typed_form_reason=None,
        type_choices=type_choices,
        selected_type=selected,
        fields=fields,
        extra_keys=tuple(key for key in written if key not in named),
        removed_by_type={
            name: _removed_by_type(
                written=tuple(key for key in written if key != _TYPE_KEY),
                current=owned,
                candidate=candidate,
            )
            for name, candidate in by_class.items()
        },
    )


def _project_field(widget: WidgetMetadata, written: Mapping[str, object]) -> NodeField:
    name = widget.path.rsplit(".", 1)[-1]
    value = written.get(name, _ABSENT)
    present = value is not _ABSENT
    reading = classify_value(value) if present else ValueReading("absent", None, None)
    control = _control(widget)
    return NodeField(
        name=name,
        path=widget.path,
        label=widget.label,
        control=control,
        required=widget.required,
        has_default=widget.has_default,
        default=widget.default,
        dimension=widget.dimension,
        unit_policy=widget.unit_policy,
        units=widget.units,
        choices=widget.choices,
        delivery=widget.delivery,
        typed=_typed(control, reading, value),
        present=present,
        form=reading.form,
        number=reading.number,
        unit=reading.unit,
        written=value if present else None,
    )


__all__ = [
    "Control",
    "NodeField",
    "NodeFieldSet",
    "ValueReading",
    "classify_value",
    "project_node_fields",
]
