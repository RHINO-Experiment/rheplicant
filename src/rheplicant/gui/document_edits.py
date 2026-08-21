"""Document transformations: one editor command in, one whole document out.

The other half of :mod:`rheplicant.gui.document`, which PROJECTS a document.
This one CHANGES it, and the two are separated because they answer different
questions: a projection reads and never writes, while every function here
returns a complete new YAML string for the caller to accept or discard.

Nothing is edited in place and nothing partial is returned. Each command
loads the authoritative text, transforms a plain copy, dumps it, and projects
the result -- so a command that refuses leaves the caller holding exactly the
bytes it was given.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering import apply_variant
from rheplicant.gui.document import (
    _COMPOSITION_KINDS,
    EditorSnapshot,
    _dump,
    _load,
    _model,
    _plain,
    _project,
    _same_value,
)
from rheplicant.radio.graph import RADIO_GRAPH


def set_node(
    yaml_text: str,
    node_id: str,
    *,
    enabled: bool,
    settings: object | None = None,
    variant: str | None = None,
) -> EditorSnapshot:
    """Enable, edit, or disable one real operator slot in ``model:``."""
    spec = RADIO_GRAPH.nodes.get(node_id)
    if spec is None or spec.kind in _COMPOSITION_KINDS:
        raise ConfigError(f"{node_id!r} is not an operator slot in graph 'single-antenna'.")
    if type(enabled) is not bool:
        raise ConfigError("enabled must be true or false.")
    if not enabled and settings is not None:
        raise ConfigError("A disabled node cannot carry settings.")
    if enabled:
        _validate_node_settings(node_id, settings)

    frozen = _load(yaml_text)
    current = _resolved_model(frozen, variant)
    if (enabled and node_id in current and _same_value(current[node_id], settings)) or (
        not enabled and node_id not in current
    ):
        return _project(yaml_text, frozen)
    plain = _plain(frozen)
    assert isinstance(plain, dict)
    _set_node_on_plain(plain, node_id, enabled=enabled, settings=settings, variant=variant)
    return _finish(yaml_text, frozen, plain)


def _validate_variant(plain: dict[str, object], variant: str | None) -> dict[str, object]:
    if variant is None:
        model = plain.setdefault("model", {})
        if not isinstance(model, dict):
            raise ConfigError("model: must be a mapping.")
        return model
    if not isinstance(variant, str) or not variant:
        raise ConfigError("variant must be a non-empty string or null.")
    variants = plain.get("variants")
    if not isinstance(variants, dict) or variant not in variants:
        known = [] if not isinstance(variants, dict) else list(variants)
        raise ConfigError(f"variant {variant!r} is not declared; known variants: {known}.")
    patch = variants[variant]
    if not isinstance(patch, dict):
        raise ConfigError(f"variant {variant!r}: patch must be a mapping.")
    model = patch.setdefault("model", {})
    if not isinstance(model, dict):
        raise ConfigError(f"variants.{variant}.model: must be a mapping.")
    return model


def _validate_node_settings(node_id: str, settings: object) -> None:
    node = RADIO_GRAPH.nodes[node_id]
    if node.reserved and (
        not isinstance(settings, Mapping) or "python" not in settings
    ):
        raise ConfigError(
            f"model.{node_id}: is reserved; configure it with python: settings."
        )
    if node.many:
        if node_id == "cal_loads":
            if not isinstance(settings, Mapping) or any(
                not isinstance(value, Mapping) for value in settings.values()
            ):
                raise ConfigError("model.cal_loads: settings must be a label-keyed mapping (FAN).")
        elif (
            not isinstance(settings, list)
            or not settings
            or any(not isinstance(value, Mapping) for value in settings)
        ):
            shape = "CHAIN" if node_id == "filters" else "SUM"
            raise ConfigError(f"model.{node_id}: settings must be a non-empty list ({shape}).")
    elif not isinstance(settings, Mapping):
        raise ConfigError("Enabled node settings must be a mapping.")


def _set_node_on_plain(
    plain: dict[str, object],
    node_id: str,
    *,
    enabled: bool,
    settings: object | None,
    variant: str | None,
) -> None:
    model = _validate_variant(plain, variant)
    deletion = f"~{node_id}"
    if enabled:
        model.pop(deletion, None)
        model[node_id] = _plain(settings)
    elif variant is None:
        model.pop(node_id, None)
    else:
        replacement: dict[str, object] = {}
        placed = False
        for key, value in model.items():
            if key in (node_id, deletion):
                if not placed:
                    replacement[deletion] = None
                    placed = True
            else:
                replacement[key] = value
        if not placed:
            replacement[deletion] = None
        model.clear()
        model.update(replacement)


def _finish(
    yaml_text: str,
    frozen: Mapping[str, object],
    plain: dict[str, object],
) -> EditorSnapshot:
    if _same_value(_plain(frozen), plain):
        return _project(yaml_text, frozen)
    rendered = _dump(plain)
    return _project(rendered, _load(rendered))


def set_many_node(
    yaml_text: str,
    node_id: str,
    entries: object,
    *,
    variant: str | None = None,
) -> EditorSnapshot:
    """Configure one SUM/CHAIN list or switching-locked FAN mapping."""
    node = RADIO_GRAPH.nodes.get(node_id)
    if node is None or not node.many:
        raise ConfigError(f"{node_id!r} is not a many node.")
    return set_node(
        yaml_text,
        node_id,
        enabled=True,
        settings=entries,
        variant=variant,
    )


def _resolved_model(document: Mapping[str, object], variant: str | None) -> Mapping[str, object]:
    return _model(document if variant is None else apply_variant(document, variant))


def move_node_instance(
    yaml_text: str,
    node_id: str,
    from_index: int,
    to_index: int,
    *,
    variant: str | None = None,
) -> EditorSnapshot:
    """Move one list-backed many instance while preserving every sibling route."""
    node = RADIO_GRAPH.nodes.get(node_id)
    if node is None or not node.many:
        raise ConfigError(f"{node_id!r} is not a many node.")
    if node_id == "cal_loads":
        raise ConfigError("model.cal_loads FAN order is locked to observation.switching.order.")
    if type(from_index) is not int or type(to_index) is not int:
        raise ConfigError("many-node move indices must be integers.")
    frozen = _load(yaml_text)
    value = _resolved_model(frozen, variant).get(node_id)
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ConfigError(f"model.{node_id}: must be a list before it can be reordered.")
    if not (0 <= from_index < len(value) and 0 <= to_index < len(value)):
        raise ConfigError(f"model.{node_id}: move indices are outside its {len(value)} entries.")
    moved = list(_plain(value))
    moved.insert(to_index, moved.pop(from_index))
    return set_many_node(yaml_text, node_id, moved, variant=variant)


def compose_node(
    yaml_text: str,
    node_id: str,
    compose: str,
    stages: Sequence[Mapping[str, object]],
    *,
    variant: str | None = None,
) -> EditorSnapshot:
    """Replace one single slot with an ordered cascade or sum of named stages."""
    node = RADIO_GRAPH.nodes.get(node_id)
    if node is None or node.kind in _COMPOSITION_KINDS or node.many:
        raise ConfigError(f"{node_id!r} is not a single operator slot.")
    expected = "sum" if node.kind == "source" else "cascade"
    if compose != expected:
        raise ConfigError(f"model.{node_id}: {node.kind} nodes compose with {expected!r}.")
    if (
        isinstance(stages, str | bytes)
        or len(stages) < 2
        or any(not isinstance(stage, Mapping) for stage in stages)
    ):
        raise ConfigError("compose stages must be a list of at least two mappings.")
    detached = [_plain(stage) for stage in stages]
    if any(
        not isinstance(stage, dict) or not isinstance(stage.get("name"), str) for stage in detached
    ):
        raise ConfigError("every composed stage requires a string name.")
    return set_node(
        yaml_text,
        node_id,
        enabled=True,
        settings={"compose": compose, "stages": detached},
        variant=variant,
    )


def _placement_nodes(at: str | Sequence[str]) -> tuple[tuple[str, ...], str | list[str]]:
    if isinstance(at, str):
        nodes = (at,)
        stored: str | list[str] = at
    elif not isinstance(at, str | bytes):
        nodes = tuple(at)
        stored = list(nodes)
    else:  # pragma: no cover - bytes is caught only for typing completeness
        nodes = ()
        stored = []
    if not nodes or any(
        not isinstance(node_id, str) or node_id not in RADIO_GRAPH.nodes for node_id in nodes
    ):
        raise ConfigError("at: must name one or more known graph nodes.")
    if any(
        target not in RADIO_GRAPH._out[source]
        for source, target in zip(nodes, nodes[1:], strict=False)
    ):
        raise ConfigError(f"at: region {list(nodes)} must follow graph edge order.")
    return nodes, stored


def place_node(
    yaml_text: str,
    node_id: str,
    at: str | Sequence[str],
    settings: Mapping[str, object],
    *,
    variant: str | None = None,
) -> EditorSnapshot:
    """Place a ``python:`` operator and auto-address a region by ``at[-1]``."""
    if node_id not in RADIO_GRAPH.nodes or RADIO_GRAPH.nodes[node_id].kind in _COMPOSITION_KINDS:
        raise ConfigError(f"{node_id!r} is not an operator slot.")
    if not isinstance(settings, Mapping) or "python" not in settings:
        raise ConfigError("at: placement requires python: operator settings.")
    if "snapshot_before" in settings:
        raise ConfigError("at: and snapshot_before: cannot be configured together.")
    nodes, stored = _placement_nodes(at)
    destination = nodes[-1]
    if RADIO_GRAPH.nodes[destination].kind in _COMPOSITION_KINDS:
        raise ConfigError(f"at: destination {destination!r} is not an operator slot.")
    frozen = _load(yaml_text)
    resolved = _resolved_model(frozen, variant)
    if destination != node_id and destination in resolved:
        raise ConfigError(
            f"model.{destination}: is already configured; placement would overwrite it."
        )
    plain = _plain(frozen)
    assert isinstance(plain, dict)
    model = _validate_variant(plain, variant)
    replacement = dict(_plain(settings))
    replacement["at"] = stored
    if variant is None:
        relocated: dict[str, object] = {}
        inserted = False
        for key, value in model.items():
            if key == node_id:
                relocated[destination] = replacement
                inserted = True
            elif key != destination:
                relocated[key] = value
        if not inserted:
            relocated[destination] = replacement
        model.clear()
        model.update(relocated)
    else:
        _set_node_on_plain(plain, node_id, enabled=False, settings=None, variant=variant)
        _set_node_on_plain(
            plain,
            destination,
            enabled=True,
            settings=replacement,
            variant=variant,
        )
    return _finish(yaml_text, frozen, plain)


def _request_snapshot_aux(plain: dict[str, object], snapshot_name: str) -> None:
    outputs = plain.setdefault("outputs", {})
    if not isinstance(outputs, dict):
        raise ConfigError("outputs: must be a mapping to request snapshot data.")
    write = outputs.setdefault("write", {})
    if not isinstance(write, dict):
        raise ConfigError("outputs.write: must be a mapping to request snapshot data.")
    aux = write.get("aux")
    key = f"snapshot/{snapshot_name}"
    if aux is True:
        return
    if aux is None:
        write["aux"] = {"keys": [key]}
        return
    if not isinstance(aux, dict):
        raise ConfigError("outputs.write.aux: must be true or a mapping.")
    keys = aux.get("keys")
    if keys is None:
        aux["keys"] = [key]
    elif not isinstance(keys, list) or any(not isinstance(item, str) for item in keys):
        raise ConfigError("outputs.write.aux.keys: must be a list of strings.")
    elif key not in keys:
        keys.append(key)


def set_snapshot_before(
    yaml_text: str,
    node_id: str,
    snapshot_name: str,
    *,
    variant: str | None = None,
) -> EditorSnapshot:
    """Add a pre-processing camera and request its aux product in one edit."""
    node = RADIO_GRAPH.nodes.get(node_id)
    if node is None or node.segment != "processing":
        raise ConfigError(f"model.{node_id}: snapshot_before is a processing-segment action.")
    if not isinstance(snapshot_name, str) or not snapshot_name:
        raise ConfigError("snapshot name must be a non-empty string.")
    frozen = _load(yaml_text)
    current = _resolved_model(frozen, variant).get(node_id)
    if not isinstance(current, Mapping):
        raise ConfigError(f"model.{node_id}: configure the node before adding a snapshot.")
    if "compose" in current or "at" in current:
        raise ConfigError("snapshot_before cannot be combined with compose: or at:.")
    replacement = dict(_plain(current))
    replacement["snapshot_before"] = snapshot_name
    plain = _plain(frozen)
    assert isinstance(plain, dict)
    _set_node_on_plain(
        plain,
        node_id,
        enabled=True,
        settings=replacement,
        variant=variant,
    )
    _request_snapshot_aux(plain, snapshot_name)
    return _finish(yaml_text, frozen, plain)


__all__ = [
    "compose_node",
    "move_node_instance",
    "place_node",
    "set_many_node",
    "set_node",
    "set_snapshot_before",
]
