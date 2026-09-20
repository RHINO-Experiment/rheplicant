"""Pair a JSON document with the subschema that governs each of its nodes.

Two guards over the audit goldens need the same traversal, and they need it for
opposite reasons: the array census asks which item schemas a corpus of goldens
has ever exercised, and the closed-object guard asks which parts of a document
are closed in the first place. Both questions are about the schema location a
value sits at, which is not recoverable from the document alone once ``$ref``
and ``anyOf`` are in play.

Writing the answer out by hand in either guard would make it a copy of the
schema, and a copy is what goes stale. So this walks the two together.
"""

from __future__ import annotations

import re

import jsonschema


def resolve(node, root):
    """Follow a ``$ref``, returning the target and its JSON pointer."""
    ref = node.get("$ref")
    if ref is None:
        return node, None
    if not ref.startswith("#/"):
        raise AssertionError(f"only local references are supported: {ref}")
    parts = ref[2:].split("/")
    target = root
    for part in parts:
        target = target[part]
    return target, "/" + "/".join(parts)


def walk(schema, instance, root, *, pointer="", path=()):
    """Yield ``(instance_path, schema_pointer, subschema, value)`` per node.

    A node governed by an ``anyOf`` is yielded once per branch that accepts it,
    because each branch is a separate claim about the same value. Callers that
    need a single verdict combine the branches themselves.
    """
    resolved, target = resolve(schema, root)
    if target is not None:
        pointer = target
    schema = resolved
    yield path, pointer, schema, instance
    if "anyOf" in schema:
        defs = root.get("$defs", {})
        for index, branch in enumerate(schema["anyOf"]):
            if jsonschema.Draft202012Validator({**branch, "$defs": defs}).is_valid(instance):
                yield from walk(
                    branch, instance, root, pointer=f"{pointer}/anyOf/{index}", path=path
                )
        return
    kind = schema.get("type")
    if kind == "object" and isinstance(instance, dict):
        properties = schema.get("properties", {})
        for key, child in properties.items():
            if key in instance:
                yield from walk(
                    child,
                    instance[key],
                    root,
                    pointer=f"{pointer}/properties/{key}",
                    path=(*path, key),
                )
        for pattern, child in schema.get("patternProperties", {}).items():
            for key, value in instance.items():
                if key not in properties and re.search(pattern, key):
                    yield from walk(
                        child,
                        value,
                        root,
                        pointer=f"{pointer}/patternProperties/{pattern}",
                        path=(*path, key),
                    )
    elif kind == "array" and isinstance(instance, list):
        items = schema.get("items")
        if items:
            for index, element in enumerate(instance):
                yield from walk(
                    items,
                    element,
                    root,
                    pointer=f"{pointer}/items",
                    path=(*path, index),
                )


def declared_array_pointers(node, root, pointer=""):
    """Every array subschema the document declares, by JSON pointer.

    Derived from the schema rather than listed, so an array added later joins
    the census the moment it is declared.
    """
    if isinstance(node, dict):
        if node.get("type") == "array":
            yield pointer
        for key, child in node.items():
            yield from declared_array_pointers(child, root, f"{pointer}/{key}")
    elif isinstance(node, list):
        for index, child in enumerate(node):
            yield from declared_array_pointers(child, root, f"{pointer}/{index}")


def at_pointer(root, pointer):
    """The subschema a JSON pointer names."""
    node = root
    for part in pointer.lstrip("/").split("/"):
        if part == "":
            continue
        node = node[int(part)] if isinstance(node, list) else node[part]
    return node


def object_verdicts(root, instance):
    """Per object path: whether every governing subschema forbids unknown keys,
    and which subschemas those are.

    ``$defs/jsonObject`` is deliberately open -- it is arbitrary captured JSON,
    matched by ``patternProperties: {".*": jsonValue}`` -- so an unknown key
    inside one is accepted by design. Every other object in these two formats
    is closed.

    The distinction is read from the schema rather than listed, and it is not
    decorative: it was invisible until a golden first carried a populated
    ``direct_url``, and the guard asserting closure had been green for the
    whole time no document reached one.

    Both halves are returned because a caller that only skipped the open paths
    would have replaced one blind spot with another: opening a closed ``$def``
    would take its objects out of the census instead of failing it.
    """
    verdicts: dict[tuple, tuple[bool, set[str]]] = {}
    for path, pointer, node, value in walk(root, instance, root):
        if not isinstance(value, dict):
            continue
        if node.get("type") != "object" and "anyOf" in node:
            continue
        closed = node.get("additionalProperties") is False and not node.get("patternProperties")
        was_closed, pointers = verdicts.get(path, (True, set()))
        verdicts[path] = (was_closed and closed, pointers | {pointer})
    return verdicts


__all__ = [
    "at_pointer",
    "object_verdicts",
    "declared_array_pointers",
    "resolve",
    "walk",
]
