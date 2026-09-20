"""Applying a variant patch, and layering the presets underneath a document.

The third of the three subjects ``layering.py`` held (A1, section 3.2). The
merge machinery beside it answers "which value wins when two documents
disagree"; this answers "which documents are being merged, in what order, and
what may a variant patch say at all".

They are separable because the merge does not know what a variant IS -- it is
handed two mappings -- and everything here is about the grammar around that
call: ``~key`` deletions, ``only:`` selection, the preset stack, and the
refusals each of those has.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from types import MappingProxyType

from _rheplicant_bootstrap import layering
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence
from _rheplicant_bootstrap.layering_probe import (
    _mapping_pairs,
)
from _rheplicant_bootstrap.presets import PresetRequest, PresetSnapshot
from _rheplicant_bootstrap.types import Origin


def _apply_variant_mapping_values(
    mapping: Mapping,
) -> dict[str, object]:
    canonical: dict[str, object] = {}
    for key, value in _mapping_pairs(mapping, failure="apply_variant: mapping traversal failed."):
        if not isinstance(key, str):
            raise ConfigError(f"apply_variant: mapping keys are strings; got {type(key).__name__}.")
        exact_key = str.__str__(key)
        if exact_key in canonical:
            raise ConfigError("apply_variant: mapping keys collide after canonicalization.")
        canonical[exact_key] = value
    return canonical


def _validated_variant_patch(name: str, patch: object) -> Mapping:
    if not isinstance(patch, Mapping):
        raise ConfigError(
            f"variant {name!r}: the patch is a mapping of sections; got {type(patch).__name__}."
        )
    patch_values = _apply_variant_mapping_values(patch)
    for key in ("variants", "~variants"):
        if key in patch_values:
            raise ConfigError(
                f"variant {name!r} declares {key!r}. Layering is one level "
                "deep by design: there is no ordering between variants and "
                "no variant builds on another, so a comparison's halves "
                "cannot drift apart through a chain."
            )
    for key in ("schema_version", "~schema_version"):
        if key in patch_values:
            raise ConfigError(
                f"variant {name!r} touches {key!r}. The version belongs to "
                "the document; a patch that changes -- or deletes -- how the "
                "document is read is not a patch."
            )
    return patch


def _apply_variant_values(document: Mapping, name: str) -> dict:
    document_values = _apply_variant_mapping_values(document)
    variants = document_values.get("variants", {})
    if not isinstance(variants, Mapping):
        raise ConfigError(
            f"variants: is a mapping of name -> patch; got {type(variants).__name__}."
        )
    variant_values = _apply_variant_mapping_values(variants)
    if not variant_values:
        raise ConfigError(f"variant {name!r} was requested but this document declares no variants.")
    if name not in variant_values:
        raise ConfigError(
            f"variant {name!r} is not declared; this document declares {sorted(variant_values)}."
        )
    patch = _validated_variant_patch(name, variant_values[name])
    return layering.recursive_update(document, patch)


def _apply_canonical_variant(canonical: layering._CanonicalVariantDocument, name: str) -> dict:
    if canonical._state is not layering._CANONICAL_FRESH:
        raise ConfigError("canonical variant document was already used.")
    canonical._state = layering._CANONICAL_RUNNING
    try:
        if name != canonical._name:
            raise ConfigError("canonical variant name did not match.")
        patch = _validated_variant_patch(name, canonical._patch)
        result = layering.merge_with_origins(
            canonical._parent,
            patch,
            origin=Origin("variant", name),
        )
        returned: dict[str, object] = {}
        canonical._payload = (result, returned)
        canonical._state = layering._CANONICAL_COMPLETED
        return returned
    finally:
        if canonical._state is layering._CANONICAL_RUNNING:
            canonical._payload = None
            canonical._state = layering._CANONICAL_FAILED


def apply_variant(document: Mapping, name: str) -> dict:
    """Return the document with one named one-level variant patch applied."""
    if not isinstance(document, Mapping):
        raise ConfigError(f"variant document is a mapping; got {type(document).__name__}.")
    if not isinstance(name, str):
        raise ConfigError(f"variant name is a string; got {type(name).__name__}.")
    name = str.__str__(name)
    if type(document) is layering._CanonicalVariantDocument:
        return _apply_canonical_variant(document, name)
    return _apply_variant_values(document, name)


def parse_default(raw: object) -> PresetRequest:
    if isinstance(raw, str):
        return PresetRequest(name=raw, only=None)
    if not isinstance(raw, Mapping):
        raise ConfigError("defaults: each entry is a preset name or a {from:, only:} mapping.")
    canonical: dict[str, object] = {}
    for key, value in _mapping_pairs(
        raw, failure="defaults: preset entry mapping traversal failed."
    ):
        if not isinstance(key, str):
            raise ConfigError(f"defaults: preset entry keys are strings; got {type(key).__name__}.")
        exact_key = str.__str__(key)
        if exact_key in canonical:
            raise ConfigError("defaults: preset entry keys collide after canonicalization.")
        canonical[exact_key] = value
    unknown = sorted(set(canonical) - {"from", "only"})
    if unknown:
        raise ConfigError(f"defaults: preset entry has unknown keys {unknown}.")
    if "from" not in canonical:
        raise ConfigError("defaults: preset entry requires from:.")
    name = canonical["from"]
    if "only" not in canonical:
        return PresetRequest(name=name, only=None)  # type: ignore[arg-type]
    only = canonical["only"]
    if not isinstance(only, list | tuple) or isinstance(only, str):
        raise ConfigError("defaults: only: is a sequence of dotted paths.")
    return PresetRequest(name=name, only=only)  # type: ignore[arg-type]


def _select_only(request: PresetRequest, document: Mapping[str, object]) -> dict[str, object]:
    if request.only is None:
        return dict(document)
    paths = [tuple(str.split(path, ".")) for path in request.only]
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
                descendant = next(child for key, child in descendant.items() if key is not terminal)
                assert isinstance(descendant, dict)
            prior = descendant[terminal]
            assert isinstance(prior, str)
            raise ConfigError(
                f"defaults preset {request.name!r}: only paths {rendered!r} and {prior!r} overlap."
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
        cursor[parts[-1]] = value
    return selected


def _without_key(result: layering.MergeResult, key: str) -> layering.MergeResult:
    document = dict(result.document)
    document.pop(key, None)
    children = dict(result.origins.children)
    children.pop(key, None)
    return layering._trusted_merge_result(
        document=MappingProxyType(document),
        origins=layering._trusted_origin_node(None, layering._frozen_children(children)),
        deletions=result.deletions,
    )


def _replace_key(
    result: layering.MergeResult,
    key: str,
    value: object,
    *,
    origin: Origin,
) -> layering.MergeResult:
    return layering.merge_with_origins(_without_key(result, key), {key: value}, origin=origin)


def _replace_key_with_node(
    result: layering.MergeResult, key: str, value: object, node: layering.OriginNode
) -> layering.MergeResult:
    without = _without_key(result, key)
    document = dict(without.document)
    document[key] = value
    children = dict(without.origins.children)
    children[key] = node
    return layering._trusted_merge_result(
        document=MappingProxyType(document),
        origins=layering._trusted_origin_node(None, layering._frozen_children(children)),
        deletions=result.deletions,
    )


def _apply_user_model(
    result: layering.MergeResult, user_model: object, *, origin: Origin
) -> layering.MergeResult:
    if user_model is None:
        return _replace_key(result, "model", None, origin=origin)
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
        candidate_node = layering._trusted_origin_node(
            origin=None, children=layering._frozen_children({})
        )
    inherited: dict[str, object] = {}
    inherited_children: dict[str, layering.OriginNode] = {}
    seen: set[str] = set()
    for name in inherit:
        if not isinstance(name, str) or not name:
            raise ConfigError(f"model.inherit: node name {name!r} is not a non-empty string.")
        if name in seen:
            raise ConfigError(f"model.inherit: node {name!r} is repeated.")
        seen.add(name)
        if name not in candidate:
            raise ConfigError(f"model.inherit: candidate node {name!r} is absent.")
        inherited[name] = candidate[name]
        inherited_children[name] = candidate_node.children[name]
    declared = {key: value for key, value in user_model.items() if key != "inherit"}
    model_origin = candidate_node.origin if inherited else origin
    seed_node = layering._trusted_origin_node(
        origin=model_origin,
        children=layering._frozen_children(inherited_children),
    )
    seeded = _replace_key_with_node(result, "model", inherited, seed_node)
    return layering.merge_with_origins(seeded, {"model": declared}, origin=origin)


def layer_presets(
    document: Mapping[str, object],
    requests: Sequence[PresetRequest],
    *,
    preset_provider: Callable[[str], PresetSnapshot],
) -> tuple[layering.MergeResult, Sequence[tuple[PresetRequest, PresetSnapshot]]]:
    """Layer selected presets in order and the user document last."""
    if not isinstance(document, Mapping):
        raise ConfigError("configuration document is a mapping before preset layering.")
    user_evidence = freeze_evidence(document, where="layer_presets document")
    assert isinstance(user_evidence, Mapping)
    try:
        requests = tuple(requests)
    except Exception:
        raise ConfigError("defaults: request sequence traversal failed.") from None
    result = layering.initial_merge({}, origin=Origin("rheplicant-default"))
    selected: list[tuple[PresetRequest, PresetSnapshot]] = []
    names: set[str] = set()
    for request in requests:
        if not isinstance(request, PresetRequest):
            raise ConfigError(
                "defaults: requests must contain PresetRequest values; got "
                f"{type(request).__name__}."
            )
        if request.name in names:
            raise ConfigError(f"defaults: package preset {request.name!r} appears more than once.")
        names.add(request.name)
        try:
            provided = preset_provider(request.name)
        except ConfigError:
            raise
        except Exception:
            raise ConfigError(f"defaults: preset provider failed for {request.name!r}.") from None
        if type(provided) is not PresetSnapshot:
            raise ConfigError(
                "defaults: preset provider returned "
                f"{type(provided).__name__}; expected PresetSnapshot."
            )
        try:
            snapshot = PresetSnapshot(
                name=provided.name,
                resource=provided.resource,
                input_bytes=provided.input_bytes,
                sha256=provided.sha256,
                document=provided.document,
                expanded_nodes=provided.expanded_nodes,
            )
        except ConfigError:
            raise
        except Exception:
            raise ConfigError(
                f"defaults: preset provider snapshot for {request.name!r} could not be validated."
            ) from None
        if snapshot.name != request.name:
            raise ConfigError(
                f"defaults: preset provider returned {snapshot.name!r} for "
                f"request {request.name!r}."
            )
        chosen = _select_only(request, snapshot.document)
        missing_model = object()
        model = chosen.pop("model", missing_model)
        preset_origin = Origin("preset", request.name)
        result = layering.merge_with_origins(result, chosen, origin=preset_origin)
        if model is not missing_model:
            result = _replace_key(result, "model", model, origin=preset_origin)
        selected.append((request, snapshot))

    user = dict(user_evidence)
    has_user_model = "model" in user
    user_model = user.pop("model", None)
    result = layering.merge_with_origins(result, user, origin=Origin("user"))
    if has_user_model:
        result = _apply_user_model(result, user_model, origin=Origin("user"))
    return result, tuple(selected)
