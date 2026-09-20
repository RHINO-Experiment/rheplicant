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
from tests.config.schema_walk import (
    at_pointer,
    declared_array_pointers,
    object_verdicts,
    walk,
)

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


def goldens(kind):
    """Every golden of a kind, taken from the directory rather than listed.

    Parametrizing over ``STATUSES`` was the older spelling and it silently
    excluded any golden whose name is not a status -- which is exactly what
    ``{kind}-populated.json`` is. A golden that no test validates is a file
    that looks like evidence.
    """
    rows = sorted(GOLDEN.glob(f"{kind}-*.json"))
    assert rows, kind
    return rows


GOLDEN_CASES = tuple(
    (path.stem.split("-", 1)[0], path)
    for kind in ("provenance", "diagnostics")
    for path in goldens(kind)
)


@pytest.mark.parametrize(
    ("kind", "path"), GOLDEN_CASES, ids=[path.stem for _kind, path in GOLDEN_CASES]
)
def test_every_golden_validates_against_packaged_schema(kind, path):
    schema = load_schema(f"{kind}-v1")
    jsonschema.validate(json.loads(path.read_bytes()), schema)


def test_the_golden_corpus_covers_every_status_and_the_populated_case():
    """The corpus is read from the directory, so this is what pins its shape.

    Without it, deleting ``provenance-populated.json`` would remove the only
    document that carries an item in most of these arrays, and every test above
    would keep passing over the three that remain.
    """
    for kind in ("provenance", "diagnostics"):
        assert {path.stem for path in goldens(kind)} == {
            f"{kind}-{status}" for status in STATUSES
        } | {f"{kind}-populated"}


def test_the_presets_shape_is_covered_by_a_populated_case():
    """An empty array validates against ANY item type, so `presets: []` proves
    nothing about the shape the producer emits.

    All three goldens carry ``"presets": []``. Measured 2026-08-24: the
    producer at ``entry.py`` was changed from emitting preset NAMES to emitting
    ``{"name", "sha256"}`` objects -- a format_version 1 breaking change
    against a closed schema -- and `tests/bootstrap`, `test_audit_envelopes`,
    `test_resolved_document` and every test in this module stayed GREEN. The
    violation was found by reading the schema, not by running anything.

    This is the missing half: one populated case in each direction, so the
    array's ITEM type is pinned rather than merely its presence. If the preset
    record is ever widened to carry a digest, this test is the one that must be
    updated deliberately -- alongside a schema version bump -- rather than
    discovering afterwards that nothing noticed.

    The argument generalized on 2026-09-20:
    ``test_every_declared_array_is_populated_by_some_golden`` now asks it of
    every array both schemas declare, and ``provenance-populated.json`` answers
    for ``presets`` with a real producer's output rather than a copy edited
    here. What this test still adds is the NEGATIVE direction -- that a widened
    item is refused -- which a populated golden cannot state.
    """
    schema = load_schema("provenance-v1")
    value = json.loads((GOLDEN / "provenance-ok.json").read_bytes())

    populated = copy.deepcopy(value)
    populated["bootstrap"]["presets"] = ["rhino_v1", "rhino_v1_extended"]
    jsonschema.validate(populated, schema)

    widened = copy.deepcopy(value)
    widened["bootstrap"]["presets"] = [
        {"name": "rhino_v1", "sha256": "0" * 64}
    ]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(widened, schema)


def test_schema_vocabularies_are_the_runtime_vocabularies():
    provenance = load_schema("provenance-v1")
    diagnostics = load_schema("diagnostics-v1")
    assert tuple(provenance["properties"]["status"]["enum"]) == STATUSES
    assert tuple(diagnostics["properties"]["status"]["enum"]) == STATUSES
    assert tuple(provenance["$defs"]["boundary"]["properties"]["stage"]["enum"]) == STAGES
    assert tuple(diagnostics["$defs"]["run"]["properties"]["status"]["enum"]) == RUN_STATUSES
    assert tuple(provenance["$defs"]["reason"]["enum"]) == ARTEFACT_REASONS
    assert tuple(provenance["$defs"]["unavailableReason"]["enum"]) == UNAVAILABLE_REASONS


@pytest.mark.parametrize(
    ("kind", "path"), GOLDEN_CASES, ids=[path.stem for _kind, path in GOLDEN_CASES]
)
def test_unknown_properties_are_refused_at_every_present_object_path(kind, path):
    schema = load_schema(f"{kind}-v1")
    value = json.loads(path.read_bytes())

    def error_tree(error):
        yield error
        for child in error.context:
            yield from error_tree(child)

    verdicts = object_verdicts(schema, value)
    closed = tuple(path for path, (shut, _where) in verdicts.items() if shut)
    assert closed

    # An object that is NOT closed has to be a MAP -- no declared properties,
    # keys matched by a pattern -- and never a record with a field list. All
    # three open definitions here (`jsonObject`, `intMap`, `stringMap`) have
    # exactly that shape, so the rule is read off them rather than listed.
    #
    # Without this line the loop below would silently shrink: opening a closed
    # `$def` would take its objects out of the census instead of failing it,
    # which is the same blind spot in a new place.
    for path, (shut, where) in verdicts.items():
        if shut:
            continue
        for pointer in where:
            node = at_pointer(schema, pointer)
            assert not node.get("properties"), (path, pointer)
            assert not node.get("required"), (path, pointer)
            assert node.get("patternProperties"), (path, pointer)

    for path in closed:
        mutated = copy.deepcopy(value)
        target = mutated
        for segment in path:
            target = target[segment]
        target["unknown"] = True
        roots = list(jsonschema.Draft202012Validator(schema).iter_errors(mutated))
        errors = [error for root in roots for error in error_tree(root)]
        assert errors
        assert any(tuple(error.absolute_path) == path for error in errors)

@pytest.mark.parametrize("kind", ("provenance", "diagnostics"))
def test_every_declared_array_is_populated_by_some_golden(kind):
    """An empty array validates against ANY item type, so a corpus of goldens
    whose arrays are all empty checks no item schema at all.

    ``test_the_presets_shape_is_covered_by_a_populated_case`` makes that
    argument for one field. This is the general case. Measured 2026-09-20: of
    the sixteen arrays ``provenance-v1`` declares, exactly one
    (``completed_boundaries``) had ever been seen carrying an item, and of the
    eight in ``diagnostics-v1`` likewise one. The other twenty-two item schemas
    -- every input, capture member, plugin, distribution, python target, seed,
    variant, resource, run, path encoding, preset, finding, gate, deferred
    validation and resolved-variant artefact the producer can emit -- were
    declared and never once exercised.

    The census is derived from the schema rather than listed here, so declaring
    a new array is enough to require a golden that fills it. The one exemption
    is derived too: an array the schema pins at ``maxItems: 0`` is a reserved
    slot with no item type to exercise.

    The assertion is an equality rather than a subset on purpose. A subset
    would stay green if the populated golden were deleted -- those arrays would
    simply stop being reached, and a clause that cannot be reached is the shape
    this repository keeps finding behind a passing test.
    """
    root = load_schema(f"{kind}-v1")
    declared = set(declared_array_pointers(root, root))
    required = {
        pointer for pointer in declared if at_pointer(root, pointer).get("maxItems") != 0
    }

    populated = set()
    for path in goldens(kind):
        value = json.loads(path.read_bytes())
        for _path, pointer, node, item in walk(root, value, root):
            if node.get("type") == "array" and item:
                populated.add(pointer)

    assert populated == required, {
        "declared but never populated": sorted(required - populated),
        "populated but not declared": sorted(populated - required),
    }
