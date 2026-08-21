from __future__ import annotations

import json
from types import SimpleNamespace

from _rheplicant_bootstrap.output import ProductRequest
from rheplicant.config.products.render import materialize_layer_product


class FakeAssembly:
    graph_name = "receiver"
    lit = ("antenna", "receiver")
    skipped = ("cable",)
    instances = (("receiver", ("receiver-1", "receiver-2")),)
    materialized = ("sum",)
    aliased = ("antenna",)
    placements = ((('antenna',), "antenna"),)

    def to_svg(self, title=None, theme="light"):
        return f'<svg data-theme="{theme}"><title>{title}</title></svg>'

    def to_html(self, title=None, theme="light"):
        return f"<!doctype html><p>{title}:{theme}</p>"

    def to_mermaid(self, theme="light"):
        return f"%% theme: {theme}\ngraph TD\n"


def layer(kind="variant", name="night/one"):
    return SimpleNamespace(
        layer=SimpleNamespace(kind=kind, name=name),
        configured=SimpleNamespace(twin=FakeAssembly()),
    )


def test_assembly_record_has_no_live_operator_objects():
    files = materialize_layer_product(
        layer("base", None),
        ProductRequest("assembly", "json", (), ()),
        component_limit=255,
    )
    assert [file.relative_path for file in files] == ["layers/base/assembly.json"]
    value = json.loads(files[0].payload)
    assert value == {
        "aliased": ["antenna"],
        "graph_name": "receiver",
        "instances": [["receiver", ["receiver-1", "receiver-2"]]],
        "lit": ["antenna", "receiver"],
        "materialized": ["sum"],
        "placements": [[['antenna'], "antenna"]],
        "skipped": ["cable"],
    }


def test_signal_paths_encode_variant_names_and_emit_each_theme():
    files = materialize_layer_product(
        layer(),
        ProductRequest(
            "signal_paths",
            "svg",
            (),
            (("themes", ("light", "dark")),),
        ),
        component_limit=255,
    )
    assert [file.relative_path for file in files] == [
        "layers/n-6e696768742f6f6e65/signal-path-light.svg",
        "layers/n-6e696768742f6f6e65/signal-path-dark.svg",
    ]
    assert b'data-theme="light"' in files[0].payload
    assert b'data-theme="dark"' in files[1].payload
