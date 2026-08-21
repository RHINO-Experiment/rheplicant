"""``model:`` node specs -> operators (schema §4.5.1).

One spec, one operator. The class comes from the node's own registry --
discovered off ``rheplicant.radio.__all__`` by ``graph_node``, the same walk
``radio/graph.py`` runs at import to validate it -- and every field is
delivered through ``field_specs`` + ``deliver``, so static-vs-traced is
decided off the class and an array form landing on a static field is refused
there (check A40). Object fields are the exception ``deliver`` cannot own: a
sky model or a projector is a declared resource taken by ``ref`` IDENTITY,
never a copy, because `beam_spill.from_projector` exists precisely so the
weight and the sky average cannot get out of step.

``eqx_leaves`` (recorded design decision, 2026-08-10): equinox can only
deserialise onto a template (``tree_deserialise_leaves(like=...)``), so the
format is a model-node key -- the node's own declared fields build the
template, the file overwrites its ARRAY leaves, and the statics stay the
document's (measured on this repo's equinox in ``inference/archive.py``: a
template's statics silently win, which is exactly right when the template IS
the declaration).
"""

from __future__ import annotations

import dataclasses
import inspect
from collections.abc import Mapping
from typing import Any

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import (
    canonical_unit_for_delivery,
    declared_or_inferred_mode,
    deliver,
    field_specs,
    origin_for_delivery,
)
from rheplicant.config.errors import ConfigError
from rheplicant.config.files import register_reader
from rheplicant.config.hatch import import_target
from rheplicant.config.refs import resolve_reference
from rheplicant.config.resolution_audit import to_json_value
from rheplicant.config.units import check_field_name_unit
from rheplicant.config.values import ResolvedValue, resolve_value
from rheplicant.core.errors import StateValidationError

__all__ = ["ambiguous_class_problem", "build_node_operator", "operator_table"]

_NODE_KEYS = frozenset({"type", "python", "from", "eqx_leaves"})


def operator_table() -> dict[str, tuple[type, ...]]:
    """Node id -> the shipped operator classes that register there.

    Discovered live off ``rheplicant.radio.__all__`` (the same walk
    ``radio/graph.py:214-224`` validates at import), so a class added to the
    package's surface is addressable from a document with no second table to
    update."""
    import rheplicant.radio as radio
    from rheplicant.core.operator import AbstractOperator

    table: dict[str, list[type]] = {}
    for name in radio.__all__:
        obj = getattr(radio, name)
        if (
            isinstance(obj, type)
            and issubclass(obj, AbstractOperator)
            and not inspect.isabstract(obj)
            and getattr(obj, "graph_node", None)
        ):
            table.setdefault(obj.graph_node, []).append(obj)
    return {node: tuple(classes) for node, classes in table.items()}


def _object_fields(cls: type) -> frozenset[str]:
    """Fields that take a declared resource by identity, not a value node."""
    from rheplicant.radio import SkySourceOperator, SkySpaceFilter

    if cls is SkySourceOperator:
        return frozenset({"sky_model", "projector"})
    if cls is SkySpaceFilter:
        return frozenset({"projector"})
    return frozenset()


def ambiguous_class_problem(node_id: str, classes: tuple[type, ...], spec: Mapping) -> str | None:
    """Check A7: the node registers more than one class and the spec picks none.

    One binding, two callers: :func:`_pick_class` below, and
    ``config.preflight.model._graph_shape``, which asks the same question of
    the raw text one phase earlier.

    Measured off :func:`operator_table` at ``f303af8``, the nodes with a
    choice are exactly three -- ``noise`` (NoiseOperator,
    RadiometerNoiseOperator), ``flagging`` (FlaggingOperator,
    MomentRFIFlaggingOperator) and ``filters`` (FourierBandFilter,
    SiderealFilter, SkySpaceFilter) -- and the table is discovered live off
    ``rheplicant.radio.__all__``, so a fourth needs no edit here.

    ``len(classes) == 1`` rather than ``< 2`` keeps the empty case answering
    exactly what :func:`_pick_class` answered before this function existed.
    It is defensive only, and no test can tell the two spellings apart: both
    callers refuse an empty tuple first -- :func:`build_node_operator`'s ``if
    not classes``, and ``preflight.model._graph_shape``'s.
    """
    if len(classes) == 1 or spec.get("type") is not None:
        return None
    names = [cls.__name__ for cls in classes]
    return (
        f"model.{node_id}: {len(classes)} classes register at this node "
        f"({names}); type: is required."
    )


def _pick_class(node_id: str, classes: tuple[type, ...], spec: Mapping) -> type:
    declared = spec.get("type")
    if declared is None:
        problem = ambiguous_class_problem(node_id, classes, spec)
        if problem is not None:
            raise ConfigError(problem)
        return classes[0]
    if declared == "NeuralOperator":
        raise ConfigError(
            f"model.{node_id}: type: NeuralOperator is deferred with "
            "capability 3 (neural surrogates) -- schema §8.1."
        )
    for cls in classes:
        if cls.__name__ == declared:
            return cls
    raise ConfigError(
        f"model.{node_id}: type: {declared!r} is not registered at this "
        f"node; it takes {[cls.__name__ for cls in classes]}."
    )


def _field_value(node_id: str, cls: type, name: str, node: Any, context: ResolutionContext) -> Any:
    destination = DestinationDescriptor(
        f"model.{node_id}.{name}",
        "model_field",
        f"{cls.__module__}.{cls.__qualname__}.{name}",
    )
    if name in _object_fields(cls):
        if not isinstance(node, Mapping) or set(node) != {"ref"}:
            raise ConfigError(
                f"model.{node_id}.{name}: is {{ref: resources.<kind>.<name>}} "
                "-- an object field takes a declared resource, identically, "
                f"not a copy; got {node!r}."
            )
        return resolve_reference(node["ref"], context)
    if isinstance(node, str):
        # A bare string is a static-str field's natural spelling
        # (mode: extract, lineshape: sinc2, switch_key: receiver_input);
        # resolve_value reserves strings for the '<number> <unit>' shorthand,
        # so the model layer short-circuits them to a scalar ResolvedValue
        # and lets deliver()'s static_str/static_other rules judge the fit.
        resolved = ResolvedValue(node, None, "scalar", {})
    else:
        resolved = resolve_value(node, context, destination=destination)
    if resolved.unit is not None:
        check_field_name_unit(name, resolved.unit)
    field = field_specs(cls)[name]
    delivered = deliver(
        resolved.value,
        field,
        dtype=context.dtype,
        source=resolved.source,
        declared_as=declared_or_inferred_mode(
            context,
            destination,
            field,
            resolved.modifiers.get("as"),
        ),
        destination=destination,
    )
    if context.trace is not None:
        context.trace.record_delivery(
            context.layer,
            destination,
            dtype=context.dtype,
            origin=origin_for_delivery(context, destination),
            unit=canonical_unit_for_delivery(context, destination, resolved.unit),
        )
    return delivered


def _construct(node_id: str, cls: type, spec: Mapping, context: ResolutionContext):
    specs = field_specs(cls)
    unknown = sorted(set(spec) - set(specs) - _NODE_KEYS)
    if unknown:
        raise ConfigError(
            f"model.{node_id}: {cls.__name__} does not take {unknown}; its "
            f"fields are {sorted(specs)}."
        )
    missing = sorted(name for name, field in specs.items() if field.required and name not in spec)
    if missing:
        raise ConfigError(f"model.{node_id}: {cls.__name__} requires {missing}.")
    kwargs = {
        name: _field_value(node_id, cls, name, node, context)
        for name, node in spec.items()
        if name in specs
    }
    declared_fields = {field.name: field for field in dataclasses.fields(cls)}
    for name, field in specs.items():
        if name in kwargs or field.required:
            continue
        declared = declared_fields[name]
        if declared.default is not dataclasses.MISSING:
            chosen = declared.default
        elif declared.default_factory is not dataclasses.MISSING:
            chosen = declared.default_factory()
        else:  # pragma: no cover - FieldSpec.required already proved this
            continue
        try:
            to_json_value(chosen)
        except ConfigError:
            continue
        kwargs[name] = context.use_default(f"model.{node_id}.{name}", chosen)
    operator = cls(**kwargs)
    if "eqx_leaves" in spec:
        operator = _apply_eqx_leaves(node_id, spec["eqx_leaves"], operator, context)
    return operator


def _c7_beam_spill(ref: str, projector: Any, t_ground: Any, context: ResolutionContext):
    """``model.beam_spill: {from: projector}``, refused in this layer's voice.

    The above-horizon fraction has two routes into a document and only one of
    them speaks config.  ``{from: horizon_fraction}`` as a VALUE node reaches
    ``kinds/projectors.py::_horizon_fraction``, which refuses a projector
    built with ``optimizations: [cache_beam_rotation]`` in a ``ConfigError``
    naming both that key and the beam's own
    ``{ref: resources.beams.<n>.sky_fraction}`` alternative.  This route
    reaches the same number through ``BeamSpillOperator.from_projector`` and,
    measured, gets neither: a projector with no ``horizon_fraction()`` earns
    a ``StateValidationError``, and a ``beam_frame='reference'`` one reaches
    ``horizon_fraction()`` itself and earns a second ``StateValidationError``
    from the package, naming no document key in either case.  Both are
    SIBLINGS of ``ConfigError`` rather than subclasses (0.2 C-12), so a test
    written as ``pytest.raises(ConfigError)`` never caught them -- which is
    why nobody noticed.

    Neither sentence is rewritten here.  ``from_projector`` runs first and
    owns the "this class defines no horizon_fraction()" case; when it fails,
    the value-node route is asked the same question, and where that route has
    a config-level sentence -- the cached-rotation one -- that sentence wins.
    Measured, the guard there returns before ``horizon_fraction()`` is
    called, so the refused document pays for no second beam integral, and a
    document that builds never enters this branch at all.
    """
    from rheplicant.radio import BeamSpillOperator

    try:
        return BeamSpillOperator.from_projector(projector, t_ground=t_ground)
    except StateValidationError as exc:
        try:
            resolve_value(
                {"from": "horizon_fraction", "projector": {"ref": ref}},
                context,
                destination=DestinationDescriptor(
                    "model.beam_spill.sky_fraction",
                    "model_field",
                    "rheplicant.radio.instrument.beam_spill.BeamSpillOperator.sky_fraction",
                ),
            )
        except ConfigError as better:
            raise ConfigError(f"model.beam_spill.projector: {better}") from better
        except Exception:  # noqa: BLE001 - measured: a bare AttributeError
            # The value-node route answers a projector with no
            # `horizon_fraction()` with an AttributeError, which is not a
            # DirtError and names nothing. There is no sentence to borrow;
            # `from_projector`'s own, below, is the better one.
            pass
        raise ConfigError(f"model.beam_spill.projector: {exc}") from exc


#: ``(node, route)`` -> the keys that route takes besides ``from:`` itself,
#: in the order a form should offer them. The three branches below read this
#: rather than spelling their key sets twice each, so a fourth reader -- the
#: GUI's ``from:`` form -- cannot drift from what this function accepts.
FROM_ROUTES: dict[tuple[str, str], tuple[str, ...]] = {
    ("beam_spill", "projector"): ("projector", "t_ground"),
    ("t_sys_extra", "basis"): ("basis", "coeff"),
    ("cal_loads", "thermistors"): ("label",),
}


def _unknown_from_keys(node_id: str, route: str, spec: Mapping) -> list[str]:
    return sorted(set(spec) - {"from", *FROM_ROUTES[(node_id, route)]})


def _from_route(node_id: str, spec: Mapping, context: ResolutionContext):
    route = spec["from"]
    if node_id == "beam_spill" and route == "projector":
        from rheplicant.radio import BeamSpillOperator

        unknown = _unknown_from_keys(node_id, route, spec)
        if unknown:
            raise ConfigError(
                f"model.beam_spill: from: projector does not take {unknown}; "
                "it takes projector: {ref: ...} and t_ground:."
            )
        for key in ("projector", "t_ground"):
            if key not in spec:
                raise ConfigError(f"model.beam_spill: from: projector requires {key}:.")
        node = spec["projector"]
        if not isinstance(node, Mapping) or set(node) != {"ref"}:
            raise ConfigError(
                f"model.beam_spill.projector: is {{ref: "
                f"resources.projectors.<name>}}; got {node!r}."
            )
        projector = resolve_reference(node["ref"], context)
        t_ground = _field_value(
            "beam_spill", BeamSpillOperator, "t_ground", spec["t_ground"], context
        )
        return _c7_beam_spill(node["ref"], projector, t_ground, context)
    if node_id == "t_sys_extra" and route == "basis":
        from rheplicant.radio import BasisTemperatureOperator

        unknown = _unknown_from_keys(node_id, route, spec)
        if unknown:
            raise ConfigError(
                f"model.t_sys_extra: from: basis does not take {unknown}; it "
                "takes basis: {ref: ...} and coeff:."
            )
        for key in ("basis", "coeff"):
            if key not in spec:
                raise ConfigError(f"model.t_sys_extra: from: basis requires {key}:.")
        node = spec["basis"]
        if not isinstance(node, Mapping) or set(node) != {"ref"}:
            raise ConfigError(
                f"model.t_sys_extra.basis: is {{ref: resources.bases.<name>}}; got {node!r}."
            )
        basis = resolve_reference(node["ref"], context)
        coeff = _field_value(
            "t_sys_extra", BasisTemperatureOperator, "coeff", spec["coeff"], context
        )
        return BasisTemperatureOperator.from_basis(basis, coeff)
    if node_id == "cal_loads" and route == "thermistors":
        unknown = _unknown_from_keys(node_id, route, spec)
        if unknown:
            raise ConfigError(
                f"model.cal_loads: from: thermistors does not take "
                f"{unknown}; it takes ['from', 'label']."
            )
        label = spec.get("label")
        if not isinstance(label, str) or not label:
            raise ConfigError(
                "model.cal_loads: from: thermistors requires label: -- the "
                "switch label whose thermistor column becomes t_load."
            )
        if context.ingest is None:
            raise ConfigError(
                "model.cal_loads: from: thermistors reads the ingested "
                "recording's thermistor log, and this document declares no "
                "observation.from_file. Declare the recording (with "
                "thermistor_columns), or give t_load a value node."
            )
        from rheplicant.radio.rhino import cal_load_operators

        return cal_load_operators(context.ingest, labels=[label])[label]
    raise ConfigError(
        f"model.{node_id}: from: {route!r} is not a route this node offers. "
        "The shipped routes: beam_spill from: projector, t_sys_extra "
        "from: basis, cal_loads from: thermistors."
    )


def _python_operator(node_id: str, spec: Mapping, context: ResolutionContext):
    from rheplicant.core.operator import AbstractOperator

    if "type" in spec:
        raise ConfigError(
            f"model.{node_id}: python: and type: together say two things "
            "about which class this is -- write one."
        )
    if context.audit is not None:
        context.audit.python_target(f"model.{node_id}.python", spec["python"])
    target = import_target(spec["python"])
    if not (isinstance(target, type) and issubclass(target, AbstractOperator)):
        raise ConfigError(
            f"model.{node_id}: python: {spec['python']!r} is "
            f"{target!r}, not an AbstractOperator subclass -- a model node "
            "constructs an operator."
        )
    remaining = {key: value for key, value in spec.items() if key != "python"}
    return _construct(node_id, target, remaining, context)


def _apply_eqx_leaves(node_id: str, spec: Any, operator, context: ResolutionContext):
    if not isinstance(spec, Mapping) or "path" not in spec:
        raise ConfigError(
            f"model.{node_id}.eqx_leaves: is {{path: ..., sha256: ...}} -- "
            "the node's own declared fields build the template, and the file "
            f"overwrites its array leaves; got {spec!r}."
        )
    unknown = sorted(set(spec) - {"path", "sha256"})
    if unknown:
        raise ConfigError(
            f"model.{node_id}.eqx_leaves: does not take {unknown}; it takes path and sha256."
        )
    file_spec: dict[str, Any] = {
        "path": spec["path"],
        "format": "eqx_leaves",
        "_template": operator,
    }
    if "sha256" in spec:
        file_spec["sha256"] = spec["sha256"]
    return resolve_value(
        {"file": file_spec},
        context,
        destination=DestinationDescriptor(
            f"model.{node_id}.eqx_leaves",
            "config_path",
            "model.*.eqx_leaves",
        ),
    ).value


@register_reader("eqx_leaves", frozenset({"_template"}), array=False)
def _read_eqx_leaves(path, spec: dict):
    """``eqx.tree_deserialise_leaves`` onto the injected template.

    ``array=False``: the return value is the reconstructed operator. The
    template is injected by ``_apply_eqx_leaves``; equinox offers no
    template-free read, so a bare ``file:`` node (or a hand-written
    ``_template:``) is refused with the route.
    """
    from rheplicant.core.operator import AbstractOperator

    template = spec.get("_template")
    if not isinstance(template, AbstractOperator):
        raise ConfigError(
            "file: format eqx_leaves reconstructs operator state onto a "
            "template, so it is read at model.<node>.eqx_leaves -- where the "
            "node's own declared fields build that template. It is not a "
            "bare file: format, and _template is injected by the model "
            "loader, never written in a document."
        )
    import equinox as eqx

    # A HANDLE, not the path. `equinox._serialisation._with_suffix` appends
    # `.eqx` to any path whose suffix is empty, and every file this layer
    # reads arrives as a capture slot -- `capture-00000001`, deliberately
    # extensionless, because the run must read the bytes that were hashed and
    # not the document's own name for them. Handed the path, equinox looks for
    # `capture-00000001.eqx` and raises FileNotFoundError. It passes file
    # objects through untouched, so opening it here is what lets the two
    # designs coexist.
    with open(path, "rb") as handle:
        return eqx.tree_deserialise_leaves(handle, like=template)


def build_node_operator(node_id: str, spec: Any, context: ResolutionContext):
    """One node spec -> one operator (composition keys already stripped)."""
    if not isinstance(spec, Mapping):
        raise ConfigError(
            f"model.{node_id}: a node spec is a mapping of the operator's own "
            f"constructor fields; got {type(spec).__name__} ({spec!r})."
        )
    if "python" in spec:
        return _python_operator(node_id, spec, context)
    if "from" in spec:
        return _from_route(node_id, spec, context)
    classes = operator_table().get(node_id)
    if not classes:
        raise ConfigError(
            f"model.{node_id}: no shipped operator registers at this node; "
            "python: (or at: from another node) is the route."
        )
    cls = _pick_class(node_id, classes, spec)
    return _construct(node_id, cls, spec, context)
