"""Bounded YAML composition and plain immutable construction."""

from __future__ import annotations

from dataclasses import dataclass

import yaml
from yaml.constructor import SafeConstructor
from yaml.events import AliasEvent, StreamEndEvent
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze


@dataclass(frozen=True, slots=True)
class YamlLimits:
    input_bytes: int = 16 * 1024 * 1024
    composed_nodes: int = 100_000
    depth: int = 100
    scalar_bytes: int = 1024 * 1024
    alias_events: int = 1_024
    expanded_nodes: int = 250_000


@dataclass(frozen=True, slots=True)
class LoadedYaml:
    value: object
    composed_nodes: int
    expanded_nodes: int
    alias_events: int


@dataclass(frozen=True, slots=True)
class ComposedYaml:
    root: Node | None
    node_count: int
    alias_events: int


class BoundedSafeLoader(yaml.SafeLoader):
    """A SafeLoader that counts resources while nodes are composed."""

    def __init__(self, stream: str, *, source_name: str, limits: YamlLimits) -> None:
        super().__init__(stream)
        self._source_name = source_name
        self._limits = limits
        self._depth = 0
        self._node_count = 0
        self._alias_events = 0

    def _refuse_limit(self, name: str, observed: int, bound: int) -> None:
        raise ConfigError(
            f"{self._source_name}: YAML {name} observed {observed} exceeds limit {bound}."
        )

    def _count_node(self) -> None:
        self._node_count += 1
        if self._node_count > self._limits.composed_nodes:
            self._refuse_limit("composed nodes", self._node_count, self._limits.composed_nodes)

    def compose_node(self, parent: Node | None, index: object) -> Node:
        self._depth += 1
        try:
            if self._depth > self._limits.depth:
                self._refuse_limit("depth", self._depth, self._limits.depth)
            if self.check_event(AliasEvent):
                self._alias_events += 1
                if self._alias_events > self._limits.alias_events:
                    self._refuse_limit(
                        "alias events", self._alias_events, self._limits.alias_events
                    )
            return super().compose_node(parent, index)
        finally:
            self._depth -= 1

    def compose_scalar_node(self, anchor: str | None) -> ScalarNode:
        event = self.peek_event()
        assert hasattr(event, "value")
        scalar_size = len(event.value.encode("utf-8"))
        if scalar_size > self._limits.scalar_bytes:
            self._refuse_limit("scalar bytes", scalar_size, self._limits.scalar_bytes)
        self._count_node()
        return super().compose_scalar_node(anchor)

    def compose_sequence_node(self, anchor: str | None) -> SequenceNode:
        self._count_node()
        return super().compose_sequence_node(anchor)

    def compose_mapping_node(self, anchor: str | None) -> MappingNode:
        self._count_node()
        return super().compose_mapping_node(anchor)


def decode_utf8_yaml(data: bytes, *, source_name: str) -> str:
    """Decode source bytes without normalising any source representation."""
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigError(f"{source_name}: YAML input is not valid UTF-8: {exc}") from exc


def compose_one_bounded(
    text: str, *, source_name: str, limits: YamlLimits
) -> ComposedYaml:
    """Compose at most one YAML document and dispose the loader on every path."""
    loader = BoundedSafeLoader(text, source_name=source_name, limits=limits)
    try:
        loader.get_event()  # StreamStartEvent
        if loader.check_event(StreamEndEvent):
            loader.get_event()
            return ComposedYaml(root=None, node_count=0, alias_events=0)
        root = loader.compose_document()
        if not loader.check_event(StreamEndEvent):
            raise ConfigError(f"{source_name}: expected exactly one YAML document.")
        loader.get_event()
        return ComposedYaml(
            root=root,
            node_count=loader._node_count,
            alias_events=loader._alias_events,
        )
    except ConfigError:
        raise
    except (yaml.YAMLError, RecursionError) as exc:
        raise ConfigError(f"{source_name}: invalid YAML: {exc}") from exc
    finally:
        loader.dispose()


_SCALAR_TAGS = {
    "tag:yaml.org,2002:null": SafeConstructor.construct_yaml_null,
    "tag:yaml.org,2002:bool": SafeConstructor.construct_yaml_bool,
    "tag:yaml.org,2002:int": SafeConstructor.construct_yaml_int,
    "tag:yaml.org,2002:float": SafeConstructor.construct_yaml_float,
    "tag:yaml.org,2002:str": SafeConstructor.construct_yaml_str,
}
_MAP_TAG = "tag:yaml.org,2002:map"
_SEQ_TAG = "tag:yaml.org,2002:seq"
_SCALAR_CONSTRUCTOR = SafeConstructor()


def _mark(mark: object) -> str:
    return f"line {mark.line + 1}, column {mark.column + 1}"


def _unsafe_tag(source_name: str, tag: str, node: Node) -> ConfigError:
    return ConfigError(f"{source_name}: unsafe YAML tag {tag!r} at {_mark(node.start_mark)}.")


def construct_plain_bounded(
    root: Node | None, *, source_name: str, limits: YamlLimits
) -> tuple[object, int]:
    """Construct fresh plain values, refusing recursive aliases and duplicates."""
    if root is None:
        return None, 0

    expanded_nodes = 0
    active: set[int] = set()

    def build(node: Node) -> object:
        nonlocal expanded_nodes
        node_identity = id(node)
        if node_identity in active:
            raise ConfigError(f"{source_name}: recursive alias at {_mark(node.start_mark)}.")
        expanded_nodes += 1
        if expanded_nodes > limits.expanded_nodes:
            raise ConfigError(
                f"{source_name}: YAML expanded nodes observed {expanded_nodes} "
                f"exceeds limit {limits.expanded_nodes}."
            )
        active.add(node_identity)
        try:
            if isinstance(node, ScalarNode):
                constructor = _SCALAR_TAGS.get(node.tag)
                if constructor is None:
                    raise _unsafe_tag(source_name, node.tag, node)
                return constructor(_SCALAR_CONSTRUCTOR, node)
            if isinstance(node, SequenceNode):
                if node.tag != _SEQ_TAG:
                    raise _unsafe_tag(source_name, node.tag, node)
                return [build(child) for child in node.value]
            if isinstance(node, MappingNode):
                if node.tag != _MAP_TAG:
                    raise _unsafe_tag(source_name, node.tag, node)
                result: dict[object, object] = {}
                first_marks: dict[object, object] = {}
                for key_node, value_node in node.value:
                    key = build(key_node)
                    try:
                        duplicate = key in result
                    except TypeError as exc:
                        raise ConfigError(
                            f"{source_name}: mapping key at {_mark(key_node.start_mark)} "
                            "is not scalar."
                        ) from exc
                    if duplicate:
                        raise ConfigError(
                            f"{source_name}: duplicate key {key!r}; first at "
                            f"{_mark(first_marks[key])}, again at {_mark(key_node.start_mark)}."
                        )
                    result[key] = build(value_node)
                    first_marks[key] = key_node.start_mark
                return result
            raise ConfigError(f"{source_name}: unsupported YAML node at {_mark(node.start_mark)}.")
        except (ValueError, TypeError, RecursionError) as exc:
            if isinstance(exc, ConfigError):
                raise
            raise ConfigError(f"{source_name}: invalid YAML scalar: {exc}") from exc
        finally:
            active.remove(node_identity)

    try:
        return build(root), expanded_nodes
    except ConfigError:
        raise
    except (yaml.YAMLError, RecursionError) as exc:
        raise ConfigError(f"{source_name}: invalid YAML: {exc}") from exc


def safe_load_document(
    data: bytes,
    *,
    source_name: str,
    limits: YamlLimits = YamlLimits(),  # noqa: B008 - public API default is specified.
) -> LoadedYaml:
    """Safely compose and construct one bounded YAML document from exact bytes."""
    if len(data) > limits.input_bytes:
        raise ConfigError(
            f"{source_name}: YAML byte count {len(data)} exceeds limit {limits.input_bytes}."
        )
    text = decode_utf8_yaml(data, source_name=source_name)
    composed = compose_one_bounded(text, source_name=source_name, limits=limits)
    value, expanded_nodes = construct_plain_bounded(
        composed.root, source_name=source_name, limits=limits
    )
    return LoadedYaml(
        value=freeze(value),
        composed_nodes=composed.node_count,
        expanded_nodes=expanded_nodes,
        alias_events=composed.alias_events,
    )


__all__ = [
    "BoundedSafeLoader",
    "ComposedYaml",
    "LoadedYaml",
    "YamlLimits",
    "compose_one_bounded",
    "construct_plain_bounded",
    "decode_utf8_yaml",
    "safe_load_document",
]
