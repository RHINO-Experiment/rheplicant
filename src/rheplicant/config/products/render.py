"""Assembly records and signal-path renderings for prepared layers."""

from __future__ import annotations

from _rheplicant_bootstrap.audit.names import encode_name
from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.output import ProductRequest

from .encoding import canonical_product_json, validate_relative_product_path
from .types import ProductFile


def _layer_directory(prepared_layer: object) -> str:
    layer = prepared_layer.layer
    if layer.kind == "base":
        return "layers/base"
    if layer.kind != "variant" or type(layer.name) is not str or not layer.name:
        raise ConfigError("scientific layer product has an invalid layer identity.")
    return f"layers/{encode_name(layer.name)}"


def _assembly_record(assembly: object) -> dict[str, object]:
    return {
        "graph_name": assembly.graph_name,
        "lit": list(assembly.lit),
        "skipped": list(assembly.skipped),
        "instances": [[name, list(instances)] for name, instances in assembly.instances],
        "materialized": list(assembly.materialized),
        "aliased": list(assembly.aliased),
        "placements": [[list(nodes), address] for nodes, address in assembly.placements],
    }


def materialize_layer_product(
    prepared_layer: object,
    request: ProductRequest,
    *,
    component_limit: int,
) -> tuple[ProductFile, ...]:
    """Materialize an assembly record or its existing render methods."""
    root = _layer_directory(prepared_layer)
    assembly = prepared_layer.configured.twin
    layer_name = prepared_layer.layer.name
    if request.name == "assembly":
        path = f"{root}/assembly.json"
        validate_relative_product_path(path, component_limit=component_limit)
        return (
            ProductFile(
                path,
                canonical_product_json(_assembly_record(assembly)),
                "assembly",
                None,
                None,
                "json",
                {"layer": layer_name},
            ),
        )
    if request.name != "signal_paths":
        raise ConfigError(f"outputs.write.{request.name}: is not a layer product.")
    options = dict(request.options)
    themes = tuple(options.get("themes", ("light",)))
    extension = {"svg": "svg", "html": "html", "mermaid": "mmd"}[request.format]
    method = {
        "svg": assembly.to_svg,
        "html": assembly.to_html,
        "mermaid": assembly.to_mermaid,
    }[request.format]
    files: list[ProductFile] = []
    for theme in themes:
        path = f"{root}/signal-path-{theme}.{extension}"
        validate_relative_product_path(path, component_limit=component_limit)
        if request.format == "mermaid":
            rendered = method(theme=theme)
        else:
            rendered = method(title=f"Signal path: {assembly.graph_name}", theme=theme)
        payload = rendered.encode("utf-8")
        if not payload.endswith(b"\n"):
            payload += b"\n"
        files.append(
            ProductFile(
                path,
                payload,
                "signal_paths",
                None,
                None,
                request.format,
                {"layer": layer_name, "theme": theme},
            )
        )
    return tuple(files)


__all__ = ["materialize_layer_product"]
