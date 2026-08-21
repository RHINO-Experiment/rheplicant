"""Canonical one-time variant enumeration and finding attribution."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Protocol, TypeVar

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import freeze_evidence, thaw
from _rheplicant_bootstrap.layering import (
    DeletionRecord,
    MergeResult,
    OriginNode,
    _canonical_variant_document,
    _canonical_variant_parent,
    _DeletionLedger,
    _OverlayMapping,
    _take_canonical_variant_result,
    _trusted_overlay_omit,
    _validate_parallel_origin_tree,
    apply_variant,
)
from _rheplicant_bootstrap.path_syntax import longest_legal_prefix
from _rheplicant_bootstrap.process import validate_variant_process_sections
from _rheplicant_bootstrap.types import LayerIdentity


class AttributableFinding(Protocol):
    where: str
    message: str


F = TypeVar("F", bound=AttributableFinding)


def _trusted_origin_root(children: Mapping[object, OriginNode]) -> OriginNode:
    if type(children) is not _OverlayMapping:
        raise ConfigError(
            "trusted layer origin children must be an exact overlay."
        )
    root = object.__new__(OriginNode)
    object.__setattr__(root, "origin", None)
    object.__setattr__(root, "children", children)
    return root


def _without_variants(
    result: MergeResult,
) -> tuple[
    Mapping[str, object], OriginNode, Sequence[DeletionRecord]
]:
    document = _trusted_overlay_omit(result.document, "variants")
    children = _trusted_overlay_omit(result.origins.children, "variants")
    origins = (
        result.origins
        if children is result.origins.children
        else _trusted_origin_root(children)
    )
    return (
        document,
        origins,
        result.deletions,
    )


@dataclass(frozen=True, slots=True)
class LayerRef:
    kind: Literal["base", "variant"]
    name: str | None
    prefix: str
    document: Mapping[str, object]
    declared_runs: object

    def __post_init__(self) -> None:
        if not isinstance(self.kind, str):
            raise ConfigError(
                f"layer kind must be a string; got {type(self.kind).__name__}."
            )
        kind = str.__str__(self.kind)
        if kind not in ("base", "variant"):
            raise ConfigError(f"layer kind is invalid: {kind!r}.")
        if not isinstance(self.prefix, str):
            raise ConfigError(
                "layer prefix must be a string; got "
                f"{type(self.prefix).__name__}."
            )
        prefix = str.__str__(self.prefix)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "prefix", prefix)
        if kind == "base":
            if self.name is not None or prefix != "":
                raise ConfigError("base layer has null name and empty prefix.")
        else:
            if not isinstance(self.name, str):
                raise ConfigError("variant layer name is a non-empty string.")
            exact_name = str.__str__(self.name)
            if not exact_name:
                raise ConfigError("variant layer name is a non-empty string.")
            object.__setattr__(self, "name", exact_name)
            if prefix != f"variants.{exact_name}":
                raise ConfigError("variant layer prefix does not match its name.")
        if not isinstance(self.document, Mapping):
            raise ConfigError("layer document is a mapping.")
        frozen_document = freeze_evidence(
            self.document, where=f"{self.prefix or 'base'} layer document"
        )
        if not isinstance(frozen_document, Mapping):
            raise ConfigError("layer document is a mapping.")
        if "variants" in frozen_document:
            raise ConfigError("layer document must not retain variants:.")
        frozen_runs = freeze_evidence(
            self.declared_runs,
            where=f"{self.prefix or 'base'} declared runs",
        )
        object.__setattr__(self, "document", frozen_document)
        object.__setattr__(self, "declared_runs", frozen_runs)

    @property
    def identity(self) -> LayerIdentity:
        return LayerIdentity(self.kind, self.name)

    def mutable_document(self) -> dict[str, object]:
        mutable = thaw(self.document)
        if not isinstance(mutable, dict):
            raise ConfigError("layer document did not thaw to a mapping.")
        for section in ("defaults", "plugins", "outputs"):
            mutable.pop(section, None)
        return mutable

    def attribute(self, finding: F) -> F:
        if self.kind == "base":
            return finding
        if not isinstance(finding.where, str) or not isinstance(
            finding.message, str
        ):
            raise ConfigError("finding attribution requires string where/message.")
        return dataclasses.replace(
            finding,
            where=longest_legal_prefix(f"{self.prefix}.{finding.where}"),
            message=f"{self.prefix}: {finding.message}",
        )


def _trusted_layer_ref(
    *,
    kind: Literal["base", "variant"],
    name: str | None,
    prefix: str,
    document: Mapping[str, object],
) -> LayerRef:
    if (
        type(kind) is not str
        or type(prefix) is not str
        or (name is not None and type(name) is not str)
    ):
        raise ConfigError("trusted layer identity fields are invalid.")
    if kind not in ("base", "variant"):
        raise ConfigError("trusted layer identity fields are invalid.")
    if kind == "base":
        if name is not None or prefix != "":
            raise ConfigError("trusted base layer identity is invalid.")
    elif type(name) is not str or not name or prefix != f"variants.{name}":
        raise ConfigError("trusted variant layer identity is invalid.")
    if type(document) is not _OverlayMapping:
        raise ConfigError(
            "trusted layer document must be an exact overlay."
        )
    if "variants" in document:
        raise ConfigError("trusted layer document must not retain variants:.")
    layer = object.__new__(LayerRef)
    object.__setattr__(layer, "kind", kind)
    object.__setattr__(layer, "name", name)
    object.__setattr__(layer, "prefix", prefix)
    object.__setattr__(layer, "document", document)
    object.__setattr__(layer, "declared_runs", document.get("runs"))
    return layer


def _layer_identity_token(
    value: object, *, where: str
) -> tuple[str, str | None]:
    if type(value) is not LayerIdentity:
        raise ConfigError(f"{where} keys must be LayerIdentity values.")
    if not isinstance(value.kind, str):
        raise ConfigError(f"{where} keys contain an invalid layer kind.")
    kind = str.__str__(value.kind)
    if value.name is None:
        name = None
    elif isinstance(value.name, str):
        name = str.__str__(value.name)
    else:
        raise ConfigError(f"{where} keys contain an invalid layer name.")
    if (kind == "base" and name is None) or (
        kind == "variant" and name is not None and bool(name)
    ):
        return kind, name
    raise ConfigError(f"{where} keys contain an invalid layer identity.")


def _canonical_layer_evidence_mapping(
    mapping: Mapping[object, object],
    identities: Sequence[LayerIdentity],
    *,
    where: str,
) -> dict[LayerIdentity, object]:
    if not isinstance(mapping, Mapping):
        raise ConfigError(f"{where} must be a mapping.")
    expected = {
        _layer_identity_token(identity, where=where): identity
        for identity in identities
    }
    observed: dict[tuple[str, str | None], object] = {}
    try:
        iterator = iter(mapping.items())
    except Exception:
        raise ConfigError(f"{where} mapping traversal failed.") from None
    while True:
        try:
            pair = next(iterator)
        except StopIteration:
            break
        except Exception:
            raise ConfigError(f"{where} mapping traversal failed.") from None
        try:
            identity, value = pair
        except Exception:
            raise ConfigError(f"{where} mapping traversal failed.") from None
        token = _layer_identity_token(identity, where=where)
        if token in observed:
            raise ConfigError(f"{where} contains a duplicate layer identity.")
        observed[token] = value
    if observed.keys() != expected.keys():
        raise ConfigError("layer enumeration evidence must match every layer.")
    return {
        identity: observed[token] for token, identity in expected.items()
    }


@dataclass(frozen=True, slots=True)
class LayerEnumeration:
    layers: Sequence[LayerRef]
    origins: Mapping[LayerIdentity, OriginNode]
    deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]]

    def __post_init__(self) -> None:
        if isinstance(self.layers, str | bytes) or not isinstance(
            self.layers, Sequence
        ):
            raise ConfigError("layer enumeration layers must be a sequence.")
        try:
            layers = tuple(self.layers)
        except Exception:
            raise ConfigError(
                "layer enumeration layers sequence traversal failed."
            ) from None
        if any(type(layer) is not LayerRef for layer in layers):
            raise ConfigError(
                "layer enumeration layers must contain LayerRef values."
            )
        layers = tuple(
            LayerRef(
                layer.kind,
                layer.name,
                layer.prefix,
                layer.document,
                layer.declared_runs,
            )
            for layer in layers
        )
        if not layers or layers[0].kind != "base":
            raise ConfigError("layer enumeration begins with exactly one base layer.")
        if any(layer.kind == "base" for layer in layers[1:]):
            raise ConfigError("layer enumeration begins with exactly one base layer.")
        identities = tuple(layer.identity for layer in layers)
        if len(set(identities)) != len(identities):
            raise ConfigError("layer enumeration identities are unique.")
        origins = _canonical_layer_evidence_mapping(
            self.origins,
            identities,
            where="layer enumeration origins",
        )
        deletions = _canonical_layer_evidence_mapping(
            self.deletions,
            identities,
            where="layer enumeration deletions",
        )
        canonical_origins: dict[LayerIdentity, OriginNode] = {}
        canonical_deletions: dict[
            LayerIdentity, Sequence[DeletionRecord]
        ] = {}
        for layer, identity in zip(layers, identities, strict=True):
            origin = origins[identity]
            if type(origin) is not OriginNode:
                raise ConfigError(
                    "layer enumeration origins must contain OriginNode values."
                )
            canonical_origin = OriginNode(origin.origin, origin.children)
            _validate_parallel_origin_tree(layer.document, canonical_origin)
            canonical_origins[identity] = canonical_origin
            rows = deletions[identity]
            if isinstance(rows, str | bytes) or not isinstance(rows, Sequence):
                raise ConfigError(
                    "layer enumeration deletions must contain sequences."
                )
            try:
                frozen_rows = tuple(rows)
            except Exception:
                raise ConfigError(
                    "layer enumeration deletion sequence traversal failed."
                ) from None
            if any(type(row) is not DeletionRecord for row in frozen_rows):
                raise ConfigError(
                    "layer enumeration deletions must contain DeletionRecord values."
                )
            canonical_deletions[identity] = frozen_rows
        object.__setattr__(self, "layers", layers)
        object.__setattr__(
            self,
            "origins",
            MappingProxyType(canonical_origins),
        )
        object.__setattr__(
            self,
            "deletions",
            MappingProxyType(
                {
                    identity: canonical_deletions[identity] for identity in identities
                }
            ),
        )


def _trusted_layer_enumeration(
    layers: tuple[LayerRef, ...],
    origins: Mapping[LayerIdentity, OriginNode],
    deletions: Mapping[LayerIdentity, Sequence[DeletionRecord]],
) -> LayerEnumeration:
    if type(layers) is not tuple or not layers:
        raise ConfigError(
            "trusted layer enumeration requires an exact non-empty layer tuple."
        )
    identities: list[LayerIdentity] = []
    for index, layer in enumerate(layers):
        if (
            type(layer) is not LayerRef
            or type(layer.kind) is not str
            or type(layer.prefix) is not str
            or (layer.name is not None and type(layer.name) is not str)
        ):
            raise ConfigError(
                "trusted layer enumeration layer values are invalid."
            )
        if type(layer.document) is not _OverlayMapping:
            raise ConfigError(
                "trusted layer enumeration documents have invalid exact types."
            )
        if index == 0:
            valid_shape = (
                layer.kind == "base"
                and layer.name is None
                and layer.prefix == ""
            )
        else:
            valid_shape = (
                layer.kind == "variant"
                and bool(layer.name)
                and layer.prefix == f"variants.{layer.name}"
            )
        if not valid_shape:
            raise ConfigError(
                "trusted layer enumeration begins with exactly one base layer."
            )
        identities.append(LayerIdentity(layer.kind, layer.name))
    if type(origins) is not dict or type(deletions) is not dict:
        raise ConfigError(
            "trusted layer enumeration evidence must use exact private maps."
        )
    origin_items = tuple(dict.items(origins))
    deletion_items = tuple(dict.items(deletions))
    if len(origin_items) != len(identities) or len(deletion_items) != len(
        identities
    ):
        raise ConfigError(
            "trusted layer enumeration evidence must match every layer."
        )
    canonical_deletions: dict[
        LayerIdentity, Sequence[DeletionRecord]
    ] = {}
    canonical_origins: dict[LayerIdentity, OriginNode] = {}
    for index, identity in enumerate(identities):
        origin_identity, origin = origin_items[index]
        deletion_identity, rows = deletion_items[index]
        if (
            type(origin_identity) is not LayerIdentity
            or type(deletion_identity) is not LayerIdentity
            or type(origin_identity.kind) is not str
            or type(deletion_identity.kind) is not str
            or (
                origin_identity.name is not None
                and type(origin_identity.name) is not str
            )
            or (
                deletion_identity.name is not None
                and type(deletion_identity.name) is not str
            )
            or origin_identity.kind != identity.kind
            or origin_identity.name != identity.name
            or deletion_identity.kind != identity.kind
            or deletion_identity.name != identity.name
        ):
            raise ConfigError(
                "trusted layer enumeration evidence must match every layer."
            )
        frozen_rows: Sequence[DeletionRecord]
        if type(rows) is _DeletionLedger:
            frozen_rows = rows
        elif type(rows) is tuple and all(
            type(row) is DeletionRecord for row in rows
        ):
            frozen_rows = rows
        else:
            raise ConfigError(
                "trusted layer enumeration deletion values are invalid."
            )
        if type(origin) is not OriginNode:
            raise ConfigError(
                "trusted layer enumeration evidence values are invalid."
            )
        if type(origin.children) is not _OverlayMapping:
            raise ConfigError(
                "trusted layer enumeration origins have invalid exact types."
            )
        canonical_origins[identity] = origin
        canonical_deletions[identity] = frozen_rows
    enumeration = object.__new__(LayerEnumeration)
    object.__setattr__(enumeration, "layers", layers)
    object.__setattr__(
        enumeration,
        "origins",
        MappingProxyType(canonical_origins),
    )
    object.__setattr__(
        enumeration,
        "deletions",
        MappingProxyType(
            {
                identity: canonical_deletions[identity]
                for identity in identities
            }
        ),
    )
    return enumeration


class LayerAttributor:
    """Per-pass base suppression; variant findings never enter the base set."""

    def __init__(self) -> None:
        self._base: frozenset[AttributableFinding] | None = None

    def attribute(
        self, layer: LayerRef, findings: Iterable[F]
    ) -> tuple[F, ...]:
        rows = tuple(findings)
        if layer.kind == "base":
            if self._base is not None:
                raise RuntimeError("base layer may be attributed only once per pass")
            self._base = frozenset(rows)
            return rows
        if self._base is None:
            raise RuntimeError("base layer must be attributed first")
        return tuple(
            layer.attribute(finding)
            for finding in rows
            if finding not in self._base
        )


def _variant_items(document: Mapping[str, object]):
    if "variants" not in document:
        return ()
    variants = document["variants"]
    if not isinstance(variants, Mapping):
        raise ConfigError(
            "variants: is a mapping of name -> patch; got "
            f"{type(variants).__name__}."
        )
    try:
        return tuple(variants.items())
    except Exception:
        raise ConfigError("variants: mapping traversal failed.") from None


def enumerate_layers_once(
    layered_document: Mapping[str, object],
    origins: OriginNode,
    deletions: Sequence[DeletionRecord],
) -> LayerEnumeration:
    """Build one base plus one effective document/evidence tuple per variant."""
    parent = _canonical_variant_parent(
        MergeResult(layered_document, origins, deletions)
    )
    validate_variant_process_sections(parent.document)
    base_document, base_origins, base_deletions = _without_variants(parent)
    base = _trusted_layer_ref(
        kind="base",
        name=None,
        prefix="",
        document=base_document,
    )
    layers: list[LayerRef] = [base]
    layer_origins: dict[LayerIdentity, OriginNode] = {
        base.identity: base_origins
    }
    layer_deletions: dict[LayerIdentity, Sequence[DeletionRecord]] = {
        base.identity: base_deletions
    }
    for raw_name, patch in _variant_items(parent.document):
        if not isinstance(raw_name, str):
            raise ConfigError(
                "variants: names must be non-empty strings; got "
                f"{type(raw_name).__name__}."
            )
        name = str.__str__(raw_name)
        if not name:
            raise ConfigError("variants: names must be non-empty strings; got ''.")
        canonical_document = _canonical_variant_document(parent, name, patch)
        if not isinstance(patch, Mapping):
            # Keep apply_variant as the single owner of this public wording.
            apply_variant(canonical_document, name)
            raise ConfigError(
                f"variants.{name}: compatibility apply accepted a "
                "non-mapping patch."
            )

        returned = apply_variant(canonical_document, name)
        merged = _take_canonical_variant_result(
            canonical_document, returned
        )
        effective_document, effective_origins, effective_deletions = (
            _without_variants(merged)
        )
        layer = _trusted_layer_ref(
            kind="variant",
            name=name,
            prefix=f"variants.{name}",
            document=effective_document,
        )
        layers.append(layer)
        layer_origins[layer.identity] = effective_origins
        layer_deletions[layer.identity] = effective_deletions

    return _trusted_layer_enumeration(
        tuple(layers),
        layer_origins,
        layer_deletions,
    )


enumerate_layers = enumerate_layers_once


__all__ = [
    "AttributableFinding",
    "LayerAttributor",
    "LayerEnumeration",
    "LayerRef",
    "enumerate_layers",
    "enumerate_layers_once",
]
