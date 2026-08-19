from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import pytest

from _rheplicant_bootstrap.audit.diagnostics import RUN_STATUSES
from _rheplicant_bootstrap.audit.provenance import ARTEFACT_REASONS, STATUSES
from _rheplicant_bootstrap.audit.trace import STAGES
from _rheplicant_bootstrap.types import UNAVAILABLE_REASONS
from rheplicant.config.schemas import load_schema

GOLDEN = Path(__file__).with_name("golden")


def walk_schema(node):
    if isinstance(node, dict):
        yield node
        for child in node.values():
            yield from walk_schema(child)
    elif isinstance(node, list):
        for child in node:
            yield from walk_schema(child)


@pytest.mark.parametrize("name", ("provenance-v1", "diagnostics-v1"))
def test_every_object_schema_is_closed_and_complete(name):
    schema = load_schema(name)
    jsonschema.Draft202012Validator.check_schema(schema)
    for node in walk_schema(schema):
        if node.get("type") == "object":
            assert node["additionalProperties"] is False
            assert set(node["required"]) == set(node["properties"])


@pytest.mark.parametrize("kind", ("provenance", "diagnostics"))
@pytest.mark.parametrize("status", STATUSES)
def test_every_golden_validates_against_packaged_schema(kind, status):
    schema = load_schema(f"{kind}-v1")
    value = json.loads((GOLDEN / f"{kind}-{status}.json").read_bytes())
    jsonschema.validate(value, schema)


def test_schema_vocabularies_are_the_runtime_vocabularies():
    provenance = load_schema("provenance-v1")
    diagnostics = load_schema("diagnostics-v1")
    assert tuple(provenance["properties"]["status"]["enum"]) == STATUSES
    assert tuple(diagnostics["properties"]["status"]["enum"]) == STATUSES
    assert tuple(provenance["$defs"]["boundary"]["properties"]["stage"]["enum"]) == STAGES
    assert tuple(diagnostics["$defs"]["run"]["properties"]["status"]["enum"]) == RUN_STATUSES
    assert tuple(provenance["$defs"]["reason"]["enum"]) == ARTEFACT_REASONS
    assert tuple(provenance["$defs"]["unavailableReason"]["enum"]) == UNAVAILABLE_REASONS


@pytest.mark.parametrize("kind", ("provenance", "diagnostics"))
def test_unknown_properties_are_refused_at_every_present_object_path(kind):
    schema = load_schema(f"{kind}-v1")
    value = json.loads((GOLDEN / f"{kind}-ok.json").read_bytes())

    def object_paths(node, path=()):
        if isinstance(node, dict):
            yield path
            for key, child in node.items():
                yield from object_paths(child, (*path, key))
        elif isinstance(node, list):
            for index, child in enumerate(node):
                yield from object_paths(child, (*path, index))

    def error_tree(error):
        yield error
        for child in error.context:
            yield from error_tree(child)

    for path in object_paths(value):
        mutated = copy.deepcopy(value)
        target = mutated
        for segment in path:
            target = target[segment]
        target["unknown"] = True
        roots = list(jsonschema.Draft202012Validator(schema).iter_errors(mutated))
        errors = [error for root in roots for error in error_tree(root)]
        assert errors
        assert any(tuple(error.absolute_path) == path for error in errors)
