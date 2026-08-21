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
    FROM_ROUTES,
    SHORTHAND,
    VALUE_FORMS,
    VALUE_MODIFIERS,
    _instance_prefix,
    _is_fan,
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

_SHAPE_REASON = "Node settings are not a mapping; edit them as YAML."

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
    #: The shapes this control can WRITE, in the order a switcher offers them.
    #: Only the three this layer round-trips: re-spelling a value is safe
    #: exactly where the classifier can read the result back.
    forms: tuple[str, ...]
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
    #: Where these settings live inside the node's own settings: empty for a
    #: single-slot node, ``("0",)`` for a list entry, ``("hot",)`` for a FAN
    #: label, ``("stages", "0")`` for a compose stage. A PATH rather than one
    #: key so the browser is handed the address instead of re-deriving it,
    #: which is one fewer place for the FAN and the list to be confused.
    slot: tuple[str, ...] = ()


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


def _forms(widget: WidgetMetadata, control: Control) -> tuple[str, ...]:
    """Which of the three round-trippable shapes this field may be written in.

    A quantity control can always be a bare number or a ``{value, unit}``
    envelope. The ``<number> <unit>`` shorthand needs a unit TOKEN to write,
    so a quotient dimension like ``adc_count/K`` -- an accepted unit but not
    an atom, and therefore with no spelling to offer -- cannot take it.
    Everything else is a bare scalar: an integer or a string has no unit to
    pair with, and re-spelling it buys nothing.
    """
    if control != "quantity" or widget.unit_policy == "forbidden":
        return ("bare",)
    if not widget.units:
        return ("bare", "quantity")
    return ("bare", "shorthand", "quantity")


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
    # `model.gain.`, `model.filters[].` or `model.cal_loads.*.` -- whichever
    # one this node's instances live under. Matching the prefix and then
    # refusing any further dot keeps `model.*.eqx_leaves`, which shares the
    # root, out of every node's field list.
    prefix = f"{_instance_prefix(node_id)}."
    return tuple(
        widget
        for widget in catalog.widgets
        if widget.path.startswith(prefix)
        and "." not in widget.path[len(prefix):]
        and widget.path[len(prefix):] not in (_TYPE_KEY, "eqx_leaves")
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
    #    its LABELS as field names. The node itself has no field set -- it is
    #    the list or the mapping -- and `project_node_instances` is where its
    #    entries get theirs.
    if spec.many:
        return _refused(node_id, "Many node: each instance carries its own fields.")
    if settings is not None and not isinstance(settings, Mapping):
        return _refused(node_id, _SHAPE_REASON)

    written: Mapping[str, object] = settings if isinstance(settings, Mapping) else {}
    return _operator_settings(node_id, written, catalog)


def project_instance_fields(
    node_id: str, settings: object, catalog: FormCatalog
) -> NodeFieldSet:
    """ONE instance of a ``many`` node, as typed controls or a reason.

    Gates 1 to 3 are the node's question and were answered before this was
    called; what arrives here is one operator's settings, so gates 4 to 9 are
    asked of it exactly as they are asked of a single-slot node. Gate 5 in
    particular: a ``python:`` entry inside a chain is still a module the
    document names.
    """
    if settings is not None and not isinstance(settings, Mapping):
        return _refused(node_id, _SHAPE_REASON)
    return _operator_settings(
        node_id, settings if isinstance(settings, Mapping) else {}, catalog
    )


def project_node_instances(
    node_id: str, settings: object, catalog: FormCatalog
) -> tuple[NodeFieldSet, ...]:
    """Every instance of a ``many`` node, in the order the document wrote them.

    Empty for a single-slot node, and empty for a ``many`` node whose settings
    are the wrong shape -- a list where the FAN wants a label mapping, or the
    other way round. That is a refusal ``document._model`` already makes; this
    declines to guess rather than making it twice in different words.
    """
    spec = RADIO_GRAPH.nodes.get(node_id)
    if spec is None or not spec.many:
        return ()
    if _is_fan(node_id):
        if not isinstance(settings, Mapping):
            return ()
        return tuple(
            dataclasses.replace(
                project_instance_fields(node_id, entry, catalog), slot=(str(label),)
            )
            for label, entry in settings.items()
        )
    if isinstance(settings, (str, bytes)) or not isinstance(settings, Sequence):
        return ()
    return tuple(
        dataclasses.replace(
            project_instance_fields(node_id, entry, catalog), slot=(str(index),)
        )
        for index, entry in enumerate(settings)
    )


def from_route_fields(
    node_id: str, settings: object, catalog: FormCatalog
) -> tuple[NodeField, ...]:
    """The keys one ``from:`` route takes, as controls, in the route's order.

    Empty for a node and route the config layer does not pair, so a document
    that names a route no node offers gets the refusal it already had rather
    than a form for something that cannot be built.

    The route table is :data:`~rheplicant.config.sections.model.FROM_ROUTES`,
    which is what ``_from_route`` itself reads, so a form here cannot offer a
    key that function would refuse. Two of the five keys -- ``t_ground`` and
    ``coeff`` -- are the shipped operator's own fields and come out of the
    catalog with their dimensions; the other three are a resource reference or
    a plain label and are described here, because no widget declares them.
    """
    if not isinstance(settings, Mapping):
        return ()
    route = settings.get("from")
    if not isinstance(route, str):
        return ()
    keys = FROM_ROUTES.get((node_id, route))
    if keys is None:
        return ()
    declared = {
        widget.path.rsplit(".", 1)[-1]: widget
        for widget in _field_widgets(node_id, catalog)
    }
    return tuple(
        _project_field(declared[key], settings)
        if key in declared
        else _route_field(node_id, key, settings)
        for key in keys
    )


def _route_field(node_id: str, name: str, settings: Mapping[str, object]) -> NodeField:
    """One ``from:`` key that no widget describes.

    ``projector:`` and ``basis:`` take a resource by ``{ref: ...}`` identity
    and ``label:`` takes a switch label. All three are required -- the route
    refuses without them -- and none is a quantity, so the ``ref`` pair stay
    read-only and point at the textarea while the label is a text box.
    """
    reference = name in ("projector", "basis")
    value = settings.get(name, _ABSENT)
    present = value is not _ABSENT
    reading = classify_value(value) if present else ValueReading("absent", None, None)
    control: Control = "opaque" if reference else "text"
    return NodeField(
        name=name,
        path=f"model.{node_id}.{name}",
        label=name.replace("_", " "),
        control=control,
        required=True,
        has_default=False,
        default=None,
        dimension="structural",
        unit_policy="forbidden",
        units=(),
        choices=(),
        delivery=None,
        forms=("bare",),
        typed=_typed(control, reading, value),
        present=present,
        form=reading.form,
        number=reading.number,
        unit=reading.unit,
        written=value if present else None,
    )


def project_compose_stages(
    node_id: str, settings: object, catalog: FormCatalog
) -> tuple[NodeFieldSet, ...]:
    """Every stage of a composed node, in the order the document wrote them.

    ``compose:`` stacks several operators at ONE node, so a stage is one
    operator's settings exactly as an instance is -- and picks from the same
    classes, because it is the same node. ``name:`` is stripped first: it
    addresses the stage in the path grammar rather than being one of the
    operator's own settings, so offering it as a field or reporting it as an
    unknown key would both be wrong.
    """
    if not isinstance(settings, Mapping) or "compose" not in settings:
        return ()
    stages = settings.get("stages")
    # A Sequence rather than a list: the parser hands out immutable tuples and
    # `document._plain` hands out lists, and a projection that only knew one
    # of them would work through one caller and silently return nothing
    # through the other.
    if isinstance(stages, (str, bytes)) or not isinstance(stages, Sequence):
        return ()
    return tuple(
        dataclasses.replace(
            project_instance_fields(
                node_id,
                {key: value for key, value in stage.items() if key != "name"}
                if isinstance(stage, Mapping)
                else stage,
                catalog,
            ),
            slot=("stages", str(index)),
        )
        for index, stage in enumerate(stages)
    )


def _operator_settings(
    node_id: str, written: Mapping[str, object], catalog: FormCatalog
) -> NodeFieldSet:
    """Gates 4 to 9, and the field set they let through."""
    if "compose" in written:
        return _refused(node_id, "Composed node: the stages own the fields.")
    # THE SECURITY GATE. Ahead of every remaining key, including `type:`, so
    # the order the document happened to write them in cannot change the
    # answer.
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
        forms=_forms(widget, control),
        typed=_typed(control, reading, value),
        present=present,
        form=reading.form,
        number=reading.number,
        unit=reading.unit,
        written=value if present else None,
    )


__all__ = [
    "Control",
    "from_route_fields",
    "project_compose_stages",
    "project_instance_fields",
    "project_node_instances",
    "NodeField",
    "NodeFieldSet",
    "ValueReading",
    "classify_value",
    "project_node_fields",
]
