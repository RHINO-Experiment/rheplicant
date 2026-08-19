"""Deterministic resolved-YAML encoding for parallel document/origin trees."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import static_isinstance
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import JsonValue, Origin


def _path(parent: str, segment: str | int) -> str:
    if type(segment) is int:
        return f"{parent}[{segment}]"
    return segment if not parent else f"{parent}.{segment}"


def _origin_label(origin: object, *, path: str) -> str:
    if type(origin) is not Origin:
        raise ConfigError(f"resolved YAML origin is missing at {path or '<root>'}.")
    try:
        return Origin(origin.kind, origin.name).render()
    except (TypeError, ValueError):
        raise ConfigError(f"resolved YAML origin is invalid at {path or '<root>'}.") from None


def _string(value: str, *, path: str) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError, UnicodeError):
        raise ConfigError(f"resolved YAML string is invalid at {path or '<root>'}.") from None


def _float(value: float, *, path: str) -> str:
    if not math.isfinite(value):
        raise ConfigError(f"resolved YAML number is not finite at {path or '<root>'}.")
    token = format(value, ".17g").lower()
    if "e" in token:
        mantissa, exponent = token.split("e", 1)
        if "." not in mantissa:
            mantissa += ".0"
        return f"{mantissa}e{int(exponent)}"
    if "." not in token:
        token += ".0"
    return token


def _scalar(value: object, *, path: str) -> str:
    if value is None:
        return "null"
    if type(value) is bool:
        return "true" if value else "false"
    if type(value) is int:
        return str(value)
    if type(value) is float:
        return _float(value, path=path)
    if type(value) is str:
        return _string(value, path=path)
    raise ConfigError(f"resolved YAML contains unsupported scalar at {path or '<root>'}.")


def _container_kind(value: object) -> str | None:
    if static_isinstance(value, Mapping):
        return "mapping"
    if type(value) in (tuple, list):
        return "sequence"
    return None


class ResolvedYamlEncoder:
    """Encode one complete plain-data tree against its exact origin tree."""

    def __init__(self, origins: OriginNode) -> None:
        if type(origins) is not OriginNode:
            raise ConfigError("resolved YAML origins must be an exact OriginNode.")
        self._origins = OriginNode(origins.origin, origins.children)

    def encode(self, document: Mapping[str, JsonValue]) -> bytes:
        if not static_isinstance(document, Mapping):
            raise ConfigError("resolved YAML document must be a mapping.")
        if self._origins.origin is not None:
            raise ConfigError("resolved YAML root origin must be null.")
        keys = self._mapping_keys(document, path="")
        if "_rheplicant_resolved" in keys and keys[-1] != "_rheplicant_resolved":
            raise ConfigError("resolved YAML requires _rheplicant_resolved to be the final key.")
        lines = self._mapping(document, self._origins, indent=0, path="")
        try:
            return ("\n".join(lines) + "\n").encode("utf-8", "strict")
        except UnicodeEncodeError:
            raise ConfigError("resolved YAML contains invalid UTF-8 text.") from None

    @staticmethod
    def _mapping_keys(value: Mapping[object, object], *, path: str) -> tuple[str, ...]:
        try:
            keys = tuple(value)
        except Exception:
            raise ConfigError(
                f"resolved YAML mapping traversal failed at {path or '<root>'}."
            ) from None
        if any(type(key) is not str for key in keys):
            raise ConfigError(f"resolved YAML mapping keys must be strings at {path or '<root>'}.")
        if len(keys) != len(set(keys)):
            raise ConfigError(f"resolved YAML mapping keys collide at {path or '<root>'}.")
        return keys

    @staticmethod
    def _origin_children(
        node: OriginNode, *, expected: Sequence[str | int], path: str
    ) -> Mapping[str | int, OriginNode]:
        try:
            actual = tuple(node.children)
        except Exception:
            raise ConfigError(
                f"resolved YAML origin traversal failed at {path or '<root>'}."
            ) from None
        if actual != tuple(expected):
            raise ConfigError(f"resolved YAML origin shape differs at {path or '<root>'}.")
        return node.children

    def _mapping(
        self,
        value: Mapping[str, object],
        node: OriginNode,
        *,
        indent: int,
        path: str,
    ) -> list[str]:
        keys = self._mapping_keys(value, path=path)
        children = self._origin_children(node, expected=keys, path=path)
        lines: list[str] = []
        prefix = " " * indent
        for key in keys:
            child_path = _path(path, key)
            try:
                child_value = value[key]
                child_node = children[key]
            except Exception:
                raise ConfigError(f"resolved YAML tree access failed at {child_path}.") from None
            label = _origin_label(child_node.origin, path=child_path)
            token = _string(key, path=child_path)
            kind = _container_kind(child_value)
            if kind is None:
                self._require_leaf(child_node, path=child_path)
                lines.append(
                    f"{prefix}{token}: {_scalar(child_value, path=child_path)}  # from {label}"
                )
                continue
            if len(child_value) == 0:
                self._require_leaf(child_node, path=child_path)
                empty = "{}" if kind == "mapping" else "[]"
                lines.append(f"{prefix}{token}: {empty}  # from {label}")
                continue
            lines.append(f"{prefix}{token}:  # from {label}")
            if kind == "mapping":
                lines.extend(
                    self._mapping(child_value, child_node, indent=indent + 2, path=child_path)
                )
            else:
                lines.extend(
                    self._sequence(child_value, child_node, indent=indent + 2, path=child_path)
                )
        return lines

    def _sequence(
        self,
        value: Sequence[object],
        node: OriginNode,
        *,
        indent: int,
        path: str,
    ) -> list[str]:
        indexes = tuple(range(len(value)))
        children = self._origin_children(node, expected=indexes, path=path)
        lines: list[str] = []
        prefix = " " * indent
        for index in indexes:
            child_path = _path(path, index)
            child = value[index]
            child_node = children[index]
            label = _origin_label(child_node.origin, path=child_path)
            kind = _container_kind(child)
            if kind is None:
                self._require_leaf(child_node, path=child_path)
                lines.append(f"{prefix}- {_scalar(child, path=child_path)}  # from {label}")
                continue
            if len(child) == 0:
                self._require_leaf(child_node, path=child_path)
                empty = "{}" if kind == "mapping" else "[]"
                lines.append(f"{prefix}- {empty}  # from {label}")
                continue
            lines.append(f"{prefix}-  # from {label}")
            if kind == "mapping":
                lines.extend(self._mapping(child, child_node, indent=indent + 2, path=child_path))
            else:
                lines.extend(self._sequence(child, child_node, indent=indent + 2, path=child_path))
        return lines

    @staticmethod
    def _require_leaf(node: OriginNode, *, path: str) -> None:
        if node.children:
            raise ConfigError(f"resolved YAML origin shape differs at {path}.")


def dump_resolved_yaml(
    document: Mapping[str, JsonValue],
    origins: OriginNode,
) -> bytes:
    return ResolvedYamlEncoder(origins).encode(document)


__all__ = ["ResolvedYamlEncoder", "dump_resolved_yaml"]
