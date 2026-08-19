from __future__ import annotations

from collections.abc import Mapping

import pytest

import rheplicant.config.orchestration as orchestration
from _rheplicant_bootstrap.audit import AuditTrace
from _rheplicant_bootstrap.audit.resolved import (
    build_resolved_artefacts,
    build_resolved_document,
)
from _rheplicant_bootstrap.audit.types import ResolvedLayerRecord
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import LayerIdentity, Origin, RunDescriptor
from rheplicant.config.errors import ConfigError as RuntimeConfigError
from rheplicant.config.orchestration import prepare_document
from tests.config.test_config_document import synthetic_document

USER = Origin("user")
BASE = LayerIdentity("base", None)


def origin_tree(value, origin=USER):
    if isinstance(value, Mapping):
        return OriginNode(origin, {key: origin_tree(child, origin) for key, child in value.items()})
    if isinstance(value, (list, tuple)):
        return OriginNode(
            origin,
            {index: origin_tree(child, origin) for index, child in enumerate(value)},
        )
    return OriginNode(origin, {})


def root_origins(document):
    nested = origin_tree(document)
    return OriginNode(None, nested.children)


def audit(defaults=()):
    return {
        "presets": (),
        "defaults": defaults,
        "deliveries": (),
        "inputs": (),
        "findings": (),
        "python_targets": (),
        "seeds": (),
        "resources": (),
        "gates": (),
        "parsed_runs": (),
        "deletions": (),
        "switch_map": {},
        "truth": {},
        "truth_omissions": (),
        "x64_required_by": (),
    }


def record(layer=BASE):
    document = {
        "runtime": {"jax_enable_x64": False},
        "model": {"kind": "graph"},
        "runs": [{"name": "fit", "kind": "nuts", "options": {}}],
    }
    descriptor = RunDescriptor(0, "fit", "nuts", None)
    return ResolvedLayerRecord(
        layer,
        document,
        root_origins(document),
        (descriptor,),
        (descriptor,),
        audit(
            (
                {"path": "runtime.platform", "value": "auto"},
                {"path": "runs[].options.thinning", "value": 1},
            )
        ),
    )


def at(node, *segments):
    current = node
    for segment in segments:
        current = current.children[segment]
    return current


def test_builder_returns_parallel_augmented_trees():
    document, origins = build_resolved_document(record())
    assert tuple(document)[-1] == "_rheplicant_resolved"
    assert document["runtime"]["platform"] == "auto"
    assert document["runs"][0]["options"]["thinning"] == 1
    assert at(origins, "runtime", "platform").origin == Origin("rheplicant-default")
    assert at(origins, "runs", 0, "options", "thinning").origin == Origin("rheplicant-default")
    namespace = at(origins, "_rheplicant_resolved")
    assert namespace.origin == Origin("rheplicant-default")


def test_artefacts_validate_both_namespaces_and_emit_completed_layers():
    variant = record(LayerIdentity("variant", "雪"))
    result = build_resolved_artefacts(
        (record(), variant),
        run_names=("fit",),
        variant_names=("雪", "failed"),
        component_limit=255,
    )
    assert [row.relative_path for row in result.files] == [
        "config.resolved.yaml",
        "variants/n-e99baa/config.resolved.yaml",
    ]
    assert [(row.kind, row.document_name) for row in result.path_encodings] == [
        ("run", "fit"),
        ("variant", "雪"),
        ("variant", "failed"),
    ]
    assert all(len(row.sha256) == 64 for row in result.files)


def test_name_limit_refuses_before_serialization(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "_rheplicant_bootstrap.audit.resolved.dump_resolved_yaml",
        lambda *_args: calls.append(True),
    )
    with pytest.raises(ConfigError, match="filesystem limit"):
        build_resolved_artefacts(
            (record(),),
            run_names=("too-long",),
            variant_names=(),
            component_limit=3,
        )
    assert calls == []


def test_emitted_only_namespace_is_refused_as_input():
    row = record()
    document = dict(row.effective_document)
    document["_rheplicant_resolved"] = {}
    bad = ResolvedLayerRecord(
        row.layer,
        document,
        root_origins(document),
        row.declared_runs,
        row.execution_runs,
        row.audit,
    )
    with pytest.raises(ConfigError, match="emitted-only"):
        build_resolved_document(bad)


def test_real_trace_freezes_every_completed_layer_and_emits_its_yaml():
    document = synthetic_document()
    trace = AuditTrace()

    prepared = prepare_document(document, scope="all_layers", trace=trace)
    snapshot = trace.snapshot()

    assert tuple(row.layer for row in snapshot.resolved_layers) == tuple(
        layer.layer.identity for layer in prepared.layers
    )
    assert all(row.audit["defaults"] for row in snapshot.resolved_layers)

    artefacts = build_resolved_artefacts(
        snapshot.resolved_layers,
        run_names=tuple(row.name for row in snapshot.resolved_layers[0].execution_runs),
        variant_names=tuple(document["variants"]),
        component_limit=255,
    )
    variant_encoding = next(
        row.encoded_name for row in artefacts.path_encodings if row.kind == "variant"
    )
    assert tuple(row.relative_path for row in artefacts.files) == (
        "config.resolved.yaml",
        f"variants/{variant_encoding}/config.resolved.yaml",
    )
    assert all(b'"_rheplicant_resolved":' in row.payload for row in artefacts.files)


def test_every_completed_layer_uses_the_same_base_execution_projection():
    document = synthetic_document()
    document["runs"] = [
        {"name": "base", "kind": "forward"},
        {"name": "alt", "kind": "forward", "variant": "v"},
    ]
    document["variants"] = {
        "v": {
            "runtime": {"seed": 5},
            "runs": [{"name": "variant-only", "kind": "forward"}],
        }
    }
    trace = AuditTrace()

    prepare_document(document, scope="all_layers", trace=trace)
    rows = trace.snapshot().resolved_layers

    expected = (
        RunDescriptor(0, "base", "forward", None),
        RunDescriptor(1, "alt", "forward", "v"),
    )
    assert [row.execution_runs for row in rows] == [expected, expected]
    assert rows[0].declared_runs != rows[1].declared_runs
    assert rows[1].declared_runs == (
        RunDescriptor(0, "variant-only", "forward", None),
    )


def test_later_variant_refusal_preserves_only_the_completed_base(monkeypatch):
    document = synthetic_document()
    document["runs"] = [
        {"name": "base", "kind": "forward"},
        {"name": "alt", "kind": "forward", "variant": "v"},
    ]
    document["variants"] = {"v": {"runtime": {"seed": 5}}}
    variant = LayerIdentity("variant", "v")
    trace = AuditTrace()
    real = orchestration._through_priced

    def refuse_variant(configured, *, layer, trace):
        completed = real(configured, layer=layer, trace=trace)
        if layer == variant:
            raise RuntimeConfigError("variant postflight refusal")
        return completed

    monkeypatch.setattr(orchestration, "_through_priced", refuse_variant)
    with pytest.raises(RuntimeConfigError, match="variant postflight refusal"):
        prepare_document(document, scope="all_layers", trace=trace)

    (base,) = trace.snapshot().resolved_layers
    assert base.layer == BASE
    assert base.execution_runs == (
        RunDescriptor(0, "base", "forward", None),
        RunDescriptor(1, "alt", "forward", "v"),
    )


def test_variant_deletions_reach_its_resolved_namespace():
    document = synthetic_document()
    document["variants"] = {"no-noise": {"model": {"~noise": None}}}
    trace = AuditTrace()

    prepare_document(document, scope="all_layers", trace=trace)
    variant = trace.snapshot().resolved_layers[1]
    resolved, _origins = build_resolved_document(variant)

    assert resolved["_rheplicant_resolved"]["deletions"] == (
        {"path": ("model", "noise"), "origin": "variant:n-6e6f2d6e6f697365"},
    )
