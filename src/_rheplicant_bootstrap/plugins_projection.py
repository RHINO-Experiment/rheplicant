"""Projecting untrusted plugin metadata into closed, frozen JSON.

The budget, the UTF-8 and JSON validation, and the freezing. Nothing here
knows what a plugin IS: it takes whatever `importlib.metadata` hands back --
which is attacker-influenced, since it is read off the filesystem -- and
either produces a bounded, frozen, canonical value or refuses.
"""

from __future__ import annotations

import json
import keyword
import math
import re
from collections.abc import Mapping, Sequence
from types import (
    MappingProxyType,
)
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import (
    _freeze_evidence_roots,
    static_isinstance,
)
from _rheplicant_bootstrap.types import JsonValue

#: Used only by the distribution-name normaliser below.
_PEP_503_RUN = re.compile(r"[-_.]+")


_METADATA_EVIDENCE_LIMIT = 250_000

_DIRECT_URL_TEXT_LIMIT = 1024 * 1024

_DIRECT_URL_INTEGER_BIT_LIMIT = math.ceil(_DIRECT_URL_TEXT_LIMIT * math.log2(10))

_DIRECT_URL_DEPTH_LIMIT = 100


class _MetadataBudgetExceeded(ConfigError):
    pass


class _MetadataBudget:
    __slots__ = ("_limit", "_used")

    def __init__(self) -> None:
        limit = _METADATA_EVIDENCE_LIMIT
        if type(limit) is not int or limit < 1:
            raise ConfigError("plugins: distribution metadata evidence budget is invalid.")
        self._limit = limit
        self._used = 0

    def consume(self) -> None:
        used = self._used + 1
        if used > self._limit:
            raise _MetadataBudgetExceeded(
                f"plugins: distribution metadata evidence budget exceeds limit {self._limit}."
            )
        self._used = used

    def ensure_remaining(self, count: int) -> None:
        if count > self._limit - self._used:
            raise _MetadataBudgetExceeded(
                f"plugins: distribution metadata evidence budget exceeds limit {self._limit}."
            )


def _validate_utf8_text(value: str, *, where: str) -> None:
    try:
        str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError(f"{where} must contain only valid UTF-8 text.") from None


def _is_utf8_text(value: str) -> bool:
    try:
        str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        return False
    return True


def _canonical_text(value: object, *, where: str, nonempty: bool = True) -> str:
    if not static_isinstance(value, str):
        raise ConfigError(f"{where} must be a string.")
    canonical = str.__str__(value)
    if nonempty and not canonical:
        raise ConfigError(f"{where} must be a non-empty string.")
    _validate_utf8_text(canonical, where=where)
    return canonical


def _canonical_module_name(value: object) -> str:
    if not static_isinstance(value, str):
        raise ConfigError("plugins: entries must be dot-separated Python module names.")
    canonical = str.__str__(value)
    _validate_utf8_text(canonical, where="plugin module name")
    parts = str.split(canonical, ".")
    if not canonical or any(
        not part or not str.isidentifier(part) or keyword.iskeyword(part) for part in parts
    ):
        raise ConfigError("plugins: entries must be non-empty dot-separated Python module names.")
    return canonical


def _normalize_distribution_name(value: object, *, require_normalized: bool) -> str:
    canonical = _canonical_text(value, where="plugin distribution name")
    normalized = _PEP_503_RUN.sub("-", str.lower(canonical))
    if not normalized:
        raise ConfigError("plugin distribution name must be non-empty.")
    if require_normalized and canonical != normalized:
        raise ConfigError("plugin distribution name must be PEP-503-normalized.")
    return normalized


def _validate_json_text(value: str, *, where: str) -> None:
    if len(value) > _DIRECT_URL_TEXT_LIMIT:
        raise ConfigError(f"{where} scalar exceeds the {_DIRECT_URL_TEXT_LIMIT}-byte limit.")
    try:
        encoded = str.encode(value, "utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError(f"{where} must contain only valid UTF-8 text.") from None
    if len(encoded) > _DIRECT_URL_TEXT_LIMIT:
        raise ConfigError(f"{where} scalar exceeds the {_DIRECT_URL_TEXT_LIMIT}-byte limit.")


def _validate_frozen_json(
    value: object,
    *,
    where: str,
    budget: _MetadataBudget | None = None,
    visited_nodes: set[int] | None = None,
    require_frozen: bool = False,
) -> None:
    """Validate the exact output of ``freeze_evidence`` as finite JSON."""
    visited = set() if visited_nodes is None else visited_nodes
    active_containers: set[int] = set()

    def validate(item: object, depth: int) -> None:
        identity = id(item)
        if identity in active_containers:
            raise ConfigError(f"{where} contains cyclic JSON evidence.")
        if identity in visited:
            return
        if depth > _DIRECT_URL_DEPTH_LIMIT:
            raise ConfigError(f"{where} depth exceeds limit {_DIRECT_URL_DEPTH_LIMIT}.")
        if budget is not None:
            budget.consume()
        item_type = type(item)
        if item is None or item_type is bool:
            visited.add(identity)
            return
        if item_type is str:
            _validate_json_text(cast(str, item), where=where)
            visited.add(identity)
            return
        if item_type is int:
            visited.add(identity)
            return
        if item_type is float:
            if not math.isfinite(cast(float, item)):
                raise ConfigError(f"{where} must contain only finite JSON numbers.")
            visited.add(identity)
            return
        is_mapping = (
            item_type is MappingProxyType if require_frozen else static_isinstance(item, Mapping)
        )
        if is_mapping:
            active_containers.add(identity)
            try:
                iterator = iter(item.items())
            except Exception:
                active_containers.remove(identity)
                raise ConfigError(f"{where} JSON mapping traversal failed.") from None
            try:
                while True:
                    try:
                        pair = next(iterator)
                    except StopIteration:
                        break
                    except Exception:
                        raise ConfigError(f"{where} JSON mapping traversal failed.") from None
                    if budget is not None:
                        budget.consume()
                    try:
                        key, child = pair
                    except Exception:
                        raise ConfigError(f"{where} JSON mapping traversal failed.") from None
                    if type(key) is not str:
                        raise ConfigError(f"{where} must have string JSON object keys.")
                    validate(key, depth + 1)
                    validate(child, depth + 1)
            finally:
                active_containers.remove(identity)
            visited.add(identity)
            return
        if type(item) is tuple:
            active_containers.add(identity)
            try:
                for child in tuple.__iter__(item):
                    if budget is not None:
                        budget.consume()
                    validate(child, depth + 1)
            finally:
                active_containers.remove(identity)
            visited.add(identity)
            return
        raise ConfigError(f"{where} contains a value that is not JSON.")

    validate(value, 0)


def _freeze_direct_url(
    value: object,
    *,
    where: str,
    budget: _MetadataBudget | None = None,
) -> Mapping[str, JsonValue]:
    if not static_isinstance(value, Mapping):
        raise ConfigError(f"{where} must be a JSON object.")
    frozen_roots = _freeze_evidence_roots(
        [value],
        where=where,
        text_limit=_DIRECT_URL_TEXT_LIMIT,
        json_only=True,
        integer_bit_limit=_DIRECT_URL_INTEGER_BIT_LIMIT,
        consume=None if budget is None else budget.consume,
    )
    frozen = frozen_roots[0]
    if not static_isinstance(frozen, Mapping):
        raise ConfigError(f"{where} must be a JSON object.")
    _validate_frozen_json(frozen, where=where)
    return cast(Mapping[str, JsonValue], frozen)


def _protocol_value(operation, *, where: str):
    try:
        return operation()
    except Exception:
        raise ConfigError(f"plugins: distribution metadata {where} failed.") from None


def _protocol_items(
    value: object,
    *,
    where: str,
    budget: _MetadataBudget,
):
    iterator = _protocol_value(lambda: iter(value), where=where)
    while True:
        try:
            item = next(iterator)
        except StopIteration:
            return
        except Exception:
            raise ConfigError(f"plugins: distribution metadata {where} failed.") from None
        budget.consume()
        yield item


def _ordered_metadata_sequence(value: object, *, where: str) -> object:
    if (
        static_isinstance(value, (str, bytes))
        or static_isinstance(value, Mapping)
        or not static_isinstance(value, Sequence)
    ):
        raise ConfigError(f"plugins: distribution metadata {where} must be an ordered sequence.")
    return value


def _strict_json_object(text: str, budget: _MetadataBudget) -> Mapping[str, JsonValue]:
    def refuse_constant(_value: str):
        raise ValueError("non-finite JSON number")

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate JSON object key")
            result[key] = value
        return result

    parsed = json.loads(
        text,
        parse_constant=refuse_constant,
        object_pairs_hook=unique_object,
    )
    return _freeze_direct_url(
        parsed,
        where="plugin distribution direct_url",
        budget=budget,
    )
