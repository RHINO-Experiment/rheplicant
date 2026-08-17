"""JAX-free document layering with immutable per-value origin evidence."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence, thaw
from _rheplicant_bootstrap.presets import PresetRequest, PresetSnapshot
from _rheplicant_bootstrap.types import Origin

OriginSegment: TypeAlias = str | int


@dataclass(frozen=True, slots=True)
class OriginNode:
    """One value/container in the origin tree; the document root has no origin."""

    origin: Origin | None
    children: Mapping[OriginSegment, OriginNode]

    def __post_init__(self) -> None:
        if not isinstance(self.children, Mapping):
            raise ConfigError("origin children must be a mapping.")
        canonical: dict[OriginSegment, OriginNode] = {}
        for segment, child in self.children.items():
            if isinstance(segment, bool) or not isinstance(segment, str | int):
                raise ConfigError(
                    f"origin child segment must be str or int; got {segment!r}."
                )
            if not isinstance(child, OriginNode):
                raise ConfigError(f"origin child {segment!r} must be an OriginNode.")
            canonical[segment] = child
        object.__setattr__(self, "children", MappingProxyType(canonical))


@dataclass(frozen=True, slots=True)
class DeletionRecord:
    path: Sequence[OriginSegment]
    origin: Origin

    def __post_init__(self) -> None:
        if isinstance(self.path, str | bytes) or not isinstance(self.path, Sequence):
            raise ConfigError("deletion path must be a sequence of segments.")
        canonical = tuple(self.path)
        if any(
            isinstance(segment, bool) or not isinstance(segment, str | int)
            for segment in canonical
        ):
            raise ConfigError("deletion path segments must be str or int.")
        object.__setattr__(self, "path", canonical)


@dataclass(frozen=True, slots=True)
class MergeResult:
    document: Mapping[str, object]
    origins: OriginNode
    deletions: Sequence[DeletionRecord]

    def __post_init__(self) -> None:
        if not isinstance(self.document, Mapping):
            raise ConfigError("merge result document must be a mapping.")
        if not isinstance(self.origins, OriginNode):
            raise ConfigError("merge result origins must be an OriginNode.")
        if isinstance(self.deletions, str | bytes) or not isinstance(
            self.deletions, Sequence
        ):
            raise ConfigError("merge result deletions must be a sequence.")
        deletions = tuple(self.deletions)
        if any(not isinstance(item, DeletionRecord) for item in deletions):
            raise ConfigError("merge result deletions must contain DeletionRecord values.")
        document = freeze_evidence(self.document, where="merge result document")
        assert isinstance(document, Mapping)
        object.__setattr__(self, "document", document)
        object.__setattr__(self, "deletions", deletions)


def _frozen_children(
    children: Mapping[OriginSegment, OriginNode],
) -> Mapping[OriginSegment, OriginNode]:
    return MappingProxyType(dict(children))


def _origin_node(value: object, origin: Origin) -> OriginNode:
    if isinstance(value, Mapping):
        children = {key: _origin_node(item, origin) for key, item in value.items()}
    elif isinstance(value, list | tuple):
        children = {index: _origin_node(item, origin) for index, item in enumerate(value)}
    else:
        children = {}
    return OriginNode(origin=origin, children=_frozen_children(children))


def _root_node(document: Mapping[str, object], origin: Origin) -> OriginNode:
    return OriginNode(
        origin=None,
        children=_frozen_children(
            {key: _origin_node(value, origin) for key, value in document.items()}
        ),
    )


def initial_merge(document: Mapping[str, object], *, origin: Origin) -> MergeResult:
    if not isinstance(document, Mapping):
        raise ConfigError(
            "initial_merge: document is a mapping; got "
            f"{type(document).__name__} ({document!r})."
        )
    evidence = freeze_evidence(document, where="initial_merge document")
    assert isinstance(evidence, Mapping)
    return MergeResult(
        document=evidence,
        origins=_root_node(evidence, origin),
        deletions=(),
    )


def _append_value(
    *,
    key: str,
    inherited: object,
    inherited_origin: OriginNode,
    given: Mapping[object, object],
    origin: Origin,
) -> tuple[list[object], OriginNode]:
    if set(given) != {"append"}:
        siblings = sorted(str(item) for item in set(given) - {"append"})
        raise ConfigError(
            f"{key!r}: append must be the only key when extending a list; "
            f"got the sibling keys {siblings}."
        )
    appended = given["append"]
    if not isinstance(appended, list | tuple):
        raise ConfigError(
            f"{key!r}: append is a sequence; got "
            f"{type(appended).__name__} ({appended!r})."
        )
    if not isinstance(inherited, list | tuple):
        raise ConfigError(
            f"{key!r} is extended with {{append: ...}} but the inherited value "
            f"is {type(inherited).__name__}, not a list."
        )
    values = [*thaw(inherited), *thaw(appended)]
    inherited_children = dict(inherited_origin.children)
    offset = len(inherited)
    inherited_children.update(
        {
            offset + index: _origin_node(item, origin)
            for index, item in enumerate(appended)
        }
    )
    return values, OriginNode(
        origin=inherited_origin.origin,
        children=_frozen_children(inherited_children),
    )


def _merge_mapping(
    base: Mapping[str, object],
    base_origins: OriginNode,
    patch: Mapping[str, object],
    *,
    origin: Origin,
    prefix: tuple[OriginSegment, ...],
    deletions: list[DeletionRecord],
) -> tuple[dict[str, object], OriginNode]:
    merged = thaw(base)
    assert isinstance(merged, dict)
    children = dict(base_origins.children)
    for key, value in patch.items():
        if not isinstance(key, str):
            raise ConfigError(
                f"layering key at {prefix or ('<root>',)!r} must be a string; "
                f"got {type(key).__name__} ({key!r})."
            )
        if key.startswith("~"):
            if value is not None:
                raise ConfigError(
                    f"{'.'.join(map(str, (*prefix, key)))}: deletion value must be null."
                )
            target = key[1:]
            if not target:
                raise ConfigError("layering deletion key must name a value after '~'.")
            merged.pop(target, None)
            children.pop(target, None)
            deletions.append(DeletionRecord((*prefix, target), origin))
            continue
        inherited = merged.get(key)
        inherited_origin = children.get(key)
        if isinstance(value, Mapping) and "append" in value:
            if inherited_origin is None:
                inherited_origin = _origin_node([], origin)
                inherited = []
            merged[key], children[key] = _append_value(
                key=key,
                inherited=inherited,
                inherited_origin=inherited_origin,
                given=value,
                origin=origin,
            )
            continue
        if (
            isinstance(value, Mapping)
            and isinstance(inherited, Mapping)
            and inherited_origin is not None
        ):
            merged[key], children[key] = _merge_mapping(
                inherited,
                inherited_origin,
                value,
                origin=origin,
                prefix=(*prefix, key),
                deletions=deletions,
            )
            continue
        mutable_value = thaw(value)
        merged[key] = mutable_value
        children[key] = _origin_node(mutable_value, origin)
    return merged, OriginNode(
        origin=base_origins.origin,
        children=_frozen_children(children),
    )


def merge_with_origins(
    parent: MergeResult,
    patch: Mapping[str, object],
    *,
    origin: Origin,
) -> MergeResult:
    if not isinstance(patch, Mapping):
        raise ConfigError(
            "merge_with_origins: patch is a mapping; got "
            f"{type(patch).__name__} ({patch!r})."
        )
    frozen_patch = freeze_evidence(patch, where="merge_with_origins patch")
    assert isinstance(frozen_patch, Mapping)
    deletions = list(parent.deletions)
    document, origins = _merge_mapping(
        parent.document,
        parent.origins,
        frozen_patch,
        origin=origin,
        prefix=(),
        deletions=deletions,
    )
    return MergeResult(
        document=document,
        origins=origins,
        deletions=tuple(deletions),
    )


def origins_at(origins: OriginNode, path: Sequence[OriginSegment]) -> Origin:
    node = origins
    traversed: list[OriginSegment] = []
    for segment in path:
        traversed.append(segment)
        try:
            node = node.children[segment]
        except KeyError:
            raise ConfigError(f"origin path {tuple(traversed)!r} does not exist.") from None
    if node.origin is None:
        raise ConfigError("origin path must identify a document value, not the root.")
    return node.origin


def _compatibility_deepcopy(value: object, memo: dict[int, object] | None = None) -> object:
    """Deep-copy arbitrary public values while converting mapping views to dicts."""
    if memo is None:
        memo = {}
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if isinstance(value, Mapping):
        result: dict[object, object] = {}
        memo[identity] = result
        for key, item in value.items():
            result[copy.deepcopy(key, memo)] = _compatibility_deepcopy(item, memo)
        return result
    if isinstance(value, list):
        result_list: list[object] = []
        memo[identity] = result_list
        result_list.extend(_compatibility_deepcopy(item, memo) for item in value)
        return result_list
    if isinstance(value, tuple):
        result_tuple = tuple(_compatibility_deepcopy(item, memo) for item in value)
        memo[identity] = result_tuple
        return result_tuple
    return copy.deepcopy(value, memo)


def merge_extends(child: dict, parent: dict) -> dict:
    """Deep-merge ``child`` over ``parent`` on the lossless public boundary."""
    for label, given in (("child", child), ("parent", parent)):
        if not isinstance(given, Mapping):
            raise ConfigError(
                f"merge_extends: {label} is a mapping; got "
                f"{type(given).__name__} ({given!r})."
            )
    merged = _compatibility_deepcopy(parent)
    assert isinstance(merged, dict)
    for key, value in child.items():
        if not isinstance(key, str):
            raise ConfigError(
                f"merge_extends: keys are strings; got {type(key).__name__} ({key!r})."
            )
        if key.startswith("~"):
            if value is not None:
                raise ConfigError(f"{key!r}: deletion value must be null.")
            merged.pop(key[1:], None)
            continue
        if isinstance(value, Mapping) and "append" in value:
            if set(value) != {"append"}:
                siblings = sorted(str(item) for item in set(value) - {"append"})
                raise ConfigError(
                    f"{key!r}: append must be the only key when extending a list; "
                    f"got the sibling keys {siblings}."
                )
            appended = value["append"]
            if not isinstance(appended, list | tuple):
                raise ConfigError(
                    f"{key!r}: append is a sequence; got "
                    f"{type(appended).__name__} ({appended!r})."
                )
            inherited = merged.get(key, [])
            if not isinstance(inherited, list):
                raise ConfigError(
                    f"{key!r} is extended with {{append: ...}} but the inherited "
                    f"value is {type(inherited).__name__}, not a list."
                )
            copied = _compatibility_deepcopy(appended)
            assert isinstance(copied, tuple | list)
            merged[key] = [*inherited, *copied]
            continue
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = merge_extends(value, merged[key])
            continue
        merged[key] = _compatibility_deepcopy(value)
    return merged


def recursive_update(base: Mapping, patch: Mapping) -> dict:
    """Return ``patch`` deep-merged over ``base`` without mutating either."""
    for label, given in (("base", base), ("patch", patch)):
        if not isinstance(given, Mapping):
            raise ConfigError(
                f"recursive_update: {label} is a mapping; got "
                f"{type(given).__name__} ({given!r})."
            )
    return merge_extends(patch, base)


def apply_variant(document: Mapping, name: str) -> dict:
    """Return the document with one named one-level variant patch applied."""
    variants = document.get("variants") or {}
    if not isinstance(variants, Mapping):
        raise ConfigError(
            f"variants: is a mapping of name -> patch; got "
            f"{type(variants).__name__} ({variants!r})."
        )
    if not variants:
        raise ConfigError(
            f"variant {name!r} was requested but this document declares no variants."
        )
    if name not in variants:
        raise ConfigError(
            f"variant {name!r} is not declared; this document declares "
            f"{sorted(variants)}."
        )
    patch = variants[name]
    if not isinstance(patch, Mapping):
        raise ConfigError(
            f"variant {name!r}: the patch is a mapping of sections; got "
            f"{type(patch).__name__} ({patch!r})."
        )
    for key in ("variants", "~variants"):
        if key in patch:
            raise ConfigError(
                f"variant {name!r} declares {key!r}. Layering is one level "
                "deep by design: there is no ordering between variants and "
                "no variant builds on another, so a comparison's halves "
                "cannot drift apart through a chain."
            )
    for key in ("schema_version", "~schema_version"):
        if key in patch:
            raise ConfigError(
                f"variant {name!r} touches {key!r}. The version belongs to "
                "the document; a patch that changes -- or deletes -- how the "
                "document is read is not a patch."
            )
    return recursive_update(document, patch)


def _valid_preset_name(name: object) -> str:
    from _rheplicant_bootstrap.presets import validate_preset_name

    return validate_preset_name(name)


def parse_default(raw: object) -> PresetRequest:
    if isinstance(raw, str):
        return PresetRequest(name=_valid_preset_name(raw), only=None)
    if not isinstance(raw, Mapping):
        raise ConfigError(
            "defaults: each entry is a preset name or a {from:, only:} mapping."
        )
    unknown = sorted(str(key) for key in set(raw) - {"from", "only"})
    if unknown:
        raise ConfigError(f"defaults: preset entry has unknown keys {unknown}.")
    if "from" not in raw:
        raise ConfigError("defaults: preset entry requires from:.")
    name = _valid_preset_name(raw["from"])
    if "only" not in raw:
        return PresetRequest(name=name, only=None)
    only = raw["only"]
    if not isinstance(only, list | tuple) or isinstance(only, str):
        raise ConfigError("defaults: only: is a sequence of dotted paths.")
    if not only:
        raise ConfigError("defaults: only: must select at least one path.")
    paths: list[str] = []
    for path in only:
        if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
            raise ConfigError(f"defaults: only: has invalid document path {path!r}.")
        paths.append(path)
    return PresetRequest(name=name, only=tuple(paths))


def _select_only(
    request: PresetRequest, document: Mapping[str, object]
) -> dict[str, object]:
    if request.only is None:
        mutable = thaw(document)
        assert isinstance(mutable, dict)
        return mutable
    paths = [tuple(path.split(".")) for path in request.only]
    terminal = object()
    trie: dict[object, object] = {}
    for parts in paths:
        rendered = ".".join(parts)
        if parts[0] == "model" and len(parts) != 1:
            raise ConfigError(
                f"defaults preset {request.name!r}: select model as a whole; "
                f"{rendered!r} is a partial model selection."
            )
        node = trie
        for part in parts:
            prior = node.get(terminal)
            if isinstance(prior, str):
                raise ConfigError(
                    f"defaults preset {request.name!r}: only paths "
                    f"{prior!r} and {rendered!r} overlap."
                )
            child = node.get(part)
            if child is None:
                child = {}
                node[part] = child
            assert isinstance(child, dict)
            node = child
        if terminal in node:
            raise ConfigError(
                f"defaults preset {request.name!r}: only path {rendered!r} is duplicate."
            )
        if node:
            descendant = node
            while terminal not in descendant:
                descendant = next(
                    child
                    for key, child in descendant.items()
                    if key is not terminal
                )
                assert isinstance(descendant, dict)
            prior = descendant[terminal]
            assert isinstance(prior, str)
            raise ConfigError(
                f"defaults preset {request.name!r}: only paths "
                f"{rendered!r} and {prior!r} overlap."
            )
        node[terminal] = rendered

    selected: dict[str, object] = {}
    for parts in paths:
        value: object = document
        for part in parts:
            if not isinstance(value, Mapping) or part not in value:
                raise ConfigError(
                    f"defaults preset {request.name!r}: only path "
                    f"{'.'.join(parts)!r} does not exist."
                )
            value = value[part]
        cursor = selected
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})  # type: ignore[assignment]
        cursor[parts[-1]] = thaw(value)
    return selected


def _without_key(result: MergeResult, key: str) -> MergeResult:
    document = thaw(result.document)
    assert isinstance(document, dict)
    document.pop(key, None)
    children = dict(result.origins.children)
    children.pop(key, None)
    return MergeResult(
        document=document,
        origins=OriginNode(None, _frozen_children(children)),
        deletions=tuple(result.deletions),
    )


def _replace_key(
    result: MergeResult,
    key: str,
    value: object,
    *,
    origin: Origin,
) -> MergeResult:
    return merge_with_origins(
        _without_key(result, key), {key: value}, origin=origin
    )


def _replace_key_with_node(
    result: MergeResult, key: str, value: object, node: OriginNode
) -> MergeResult:
    without = _without_key(result, key)
    document = thaw(without.document)
    assert isinstance(document, dict)
    document[key] = thaw(value)
    children = dict(without.origins.children)
    children[key] = node
    return MergeResult(
        document=document,
        origins=OriginNode(None, _frozen_children(children)),
        deletions=tuple(result.deletions),
    )


def _apply_user_model(
    result: MergeResult, user_model: object, *, origin: Origin
) -> MergeResult:
    if not isinstance(user_model, Mapping):
        raise ConfigError("model: is a mapping when package presets are layered.")
    if "inherit" not in user_model:
        return _replace_key(result, "model", user_model, origin=origin)
    inherit = user_model["inherit"]
    if not isinstance(inherit, list | tuple) or isinstance(inherit, str):
        raise ConfigError("model.inherit: is a sequence of model node names.")
    candidate = result.document.get("model")
    candidate_node = result.origins.children.get("model")
    if not isinstance(candidate, Mapping) or candidate_node is None:
        candidate = {}
        candidate_node = OriginNode(origin=None, children=_frozen_children({}))
    inherited: dict[str, object] = {}
    inherited_children: dict[str, OriginNode] = {}
    seen: set[str] = set()
    for name in inherit:
        if not isinstance(name, str) or not name:
            raise ConfigError(f"model.inherit: node name {name!r} is not a non-empty string.")
        if name in seen:
            raise ConfigError(f"model.inherit: node {name!r} is repeated.")
        seen.add(name)
        if name not in candidate:
            raise ConfigError(f"model.inherit: candidate node {name!r} is absent.")
        inherited[name] = thaw(candidate[name])
        inherited_children[name] = candidate_node.children[name]
    declared = {key: value for key, value in user_model.items() if key != "inherit"}
    model_origin = candidate_node.origin if inherited else origin
    seed_node = OriginNode(
        origin=model_origin,
        children=_frozen_children(inherited_children),
    )
    seeded = _replace_key_with_node(result, "model", inherited, seed_node)
    return merge_with_origins(seeded, {"model": declared}, origin=origin)


def layer_presets(
    document: Mapping[str, object],
    requests: Sequence[PresetRequest],
    *,
    preset_provider: Callable[[str], PresetSnapshot],
) -> tuple[MergeResult, Sequence[tuple[PresetRequest, PresetSnapshot]]]:
    """Layer selected presets in order and the user document last."""
    if not isinstance(document, Mapping):
        raise ConfigError("configuration document is a mapping before preset layering.")
    result = initial_merge({}, origin=Origin("rheplicant-default"))
    selected: list[tuple[PresetRequest, PresetSnapshot]] = []
    names: set[str] = set()
    for request in requests:
        if request.name in names:
            raise ConfigError(
                f"defaults: package preset {request.name!r} appears more than once."
            )
        names.add(request.name)
        snapshot = preset_provider(request.name)
        chosen = _select_only(request, snapshot.document)
        missing_model = object()
        model = chosen.pop("model", missing_model)
        preset_origin = Origin("preset", request.name)
        result = merge_with_origins(result, chosen, origin=preset_origin)
        if model is not missing_model:
            result = _replace_key(result, "model", model, origin=preset_origin)
        selected.append((request, snapshot))

    user = thaw(document)
    assert isinstance(user, dict)
    has_user_model = "model" in user
    user_model = user.pop("model", None)
    result = merge_with_origins(result, user, origin=Origin("user"))
    if has_user_model:
        result = _apply_user_model(result, user_model, origin=Origin("user"))
    return result, tuple(selected)


__all__ = [
    "DeletionRecord",
    "MergeResult",
    "OriginNode",
    "apply_variant",
    "initial_merge",
    "layer_presets",
    "merge_extends",
    "merge_with_origins",
    "origins_at",
    "parse_default",
    "recursive_update",
]
