"""Framework-free YAML transformations for the config editor.

The browser never owns a second scientific state.  It sends YAML text here,
and every content edit receives a new YAML document plus a projection of the
live signal-path graph.  Selection and hover remain view state and therefore
do not invent config keys.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import yaml

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.yaml import safe_load_document
from rheplicant.radio.graph import RADIO_GRAPH

_COMPOSITION_KINDS = frozenset(("junction", "selector"))


@dataclass(frozen=True, slots=True)
class NodeCard:
    """One live graph node projected for a frontend."""

    node_id: str
    label: str
    kind: str
    description: str
    editable: bool
    reserved: bool
    many: bool
    lit: bool


@dataclass(frozen=True, slots=True)
class EditorSnapshot:
    """The complete, serializable result of one document transition."""

    yaml_text: str
    svg: str
    nodes: tuple[NodeCard, ...]
    walk_order: tuple[str, ...]


def _load(yaml_text: str) -> Mapping[str, object]:
    if not isinstance(yaml_text, str):
        raise ConfigError("GUI YAML must be text.")
    try:
        payload = yaml_text.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError("GUI YAML must contain valid UTF-8 text.") from None
    loaded = safe_load_document(payload, source_name="GUI document").value
    if not isinstance(loaded, Mapping):
        raise ConfigError("GUI document root must be a mapping.")
    return loaded


def _model(document: Mapping[str, object]) -> Mapping[str, object]:
    value = document.get("model", {})
    if not isinstance(value, Mapping):
        raise ConfigError("model: must be a mapping.")
    for node_id, settings in value.items():
        if not isinstance(settings, Mapping):
            raise ConfigError(f"model.{node_id}: must be a mapping.")
    return value


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _dump(document: Mapping[str, object]) -> str:
    return yaml.safe_dump(
        _plain(document),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )


def _same_value(left: object, right: object) -> bool:
    """Compare YAML values without collapsing bool/int/float distinctions."""
    if type(left) is not type(right):
        return False
    if isinstance(left, dict) and isinstance(right, dict):
        if len(left) != len(right):
            return False
        for left_key, left_value in left.items():
            matches = [
                right_value
                for right_key, right_value in right.items()
                if _same_value(left_key, right_key)
            ]
            if len(matches) != 1 or not _same_value(left_value, matches[0]):
                return False
        return True
    if isinstance(left, list) and isinstance(right, list):
        return len(left) == len(right) and all(
            _same_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return left == right


def _project(yaml_text: str, document: Mapping[str, object]) -> EditorSnapshot:
    model = _model(document)
    lit = tuple(node_id for node_id in RADIO_GRAPH._topo if node_id in model)
    nodes = tuple(
        NodeCard(
            node_id=node_id,
            label=node_id.replace("_", " "),
            kind=spec.kind,
            description=spec.doc,
            editable=spec.kind not in _COMPOSITION_KINDS,
            reserved=spec.reserved,
            many=spec.many,
            lit=node_id in model,
        )
        for node_id in RADIO_GRAPH._topo
        for spec in (RADIO_GRAPH.nodes[node_id],)
    )
    return EditorSnapshot(
        yaml_text=yaml_text,
        svg=RADIO_GRAPH.to_svg(lit=lit, title="Rheplicant config signal path"),
        nodes=nodes,
        walk_order=RADIO_GRAPH._topo,
    )


def snapshot(yaml_text: str) -> EditorSnapshot:
    """Parse ``yaml_text`` and project it without changing a byte."""
    document = _load(yaml_text)
    return _project(yaml_text, document)


def replace_yaml(yaml_text: str) -> EditorSnapshot:
    """Validate a bidirectional YAML-mirror replacement."""
    return snapshot(yaml_text)


def set_node(
    yaml_text: str,
    node_id: str,
    *,
    enabled: bool,
    settings: Mapping[str, object] | None = None,
) -> EditorSnapshot:
    """Enable, edit, or disable one real operator slot in ``model:``."""
    spec = RADIO_GRAPH.nodes.get(node_id)
    if spec is None or spec.kind in _COMPOSITION_KINDS:
        raise ConfigError(f"{node_id!r} is not an operator slot in graph 'single-antenna'.")
    if type(enabled) is not bool:
        raise ConfigError("enabled must be true or false.")
    if not enabled and settings is not None:
        raise ConfigError("A disabled node cannot carry settings.")
    if enabled and not isinstance(settings, Mapping):
        raise ConfigError("Enabled node settings must be a mapping.")

    frozen = _load(yaml_text)
    current_model = _model(frozen)
    plain = _plain(frozen)
    assert isinstance(plain, dict)
    model = dict(_plain(current_model))

    if enabled:
        replacement = dict(_plain(settings))  # type: ignore[arg-type]
        if node_id in current_model and _same_value(
            _plain(current_model[node_id]), replacement
        ):
            return _project(yaml_text, frozen)
        model[node_id] = replacement
    else:
        if node_id not in current_model:
            return _project(yaml_text, frozen)
        del model[node_id]

    plain["model"] = model
    rendered = _dump(plain)
    return _project(rendered, _load(rendered))


__all__ = [
    "EditorSnapshot",
    "NodeCard",
    "replace_yaml",
    "set_node",
    "snapshot",
]
