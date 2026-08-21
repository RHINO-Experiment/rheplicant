"""The `json_schema()` accessor: the machine-readable projection of the
config grammar that rheplicant-compute's schema RPC serves verbatim."""

import json

import pytest

from rheplicant.config import (
    ACCEPTED_UNITS,
    DERIVATIONS,
    FILE_FORMATS,
    RESOURCE_KINDS,
    SHAPE_SYMBOLS,
    VALUE_FORMS,
    VALUE_MODIFIERS,
)
from rheplicant.config.dimensions import _FORMULA_REGISTRY
from rheplicant.config.preflight import _REQUIRED, _SECTIONS
from rheplicant.config.schema import json_schema
from rheplicant.config.sections.runs import _KINDS as _EXIT_KINDS


def test_the_top_level_keys_are_exactly_the_six_the_rpc_promises():
    """A key dropped or renamed here is a shape change the RPC consumer sees
    as a KeyError, not as a diff -- so this is asserted as a set, not as a
    handful of membership checks that would miss an extra key."""
    assert set(json_schema()) == {
        "schemaVersion", "sections", "exits", "operators", "transforms",
        "catalogs",
    }


def test_schema_version_is_the_string_one():
    """A string, not an int: the RPC's own consumers compare it to a literal
    "1", and `1 == "1"` is False in every language on the other end of the
    wire."""
    assert json_schema()["schemaVersion"] == "1"
    assert isinstance(json_schema()["schemaVersion"], str)


def test_the_whole_document_round_trips_through_json():
    """The docstring's own claim -- "JSON-serializable" -- checked rather
    than taken on faith.  A tuple or a set anywhere in the tree serializes
    today (json.dumps accepts tuples) but silently changes shape on the way
    (a tuple becomes a JSON array indistinguishable from a list, a set
    raises), so round-tripping and comparing for equality is the guard that
    actually tells the two apart from a plain dict-of-lists."""
    schema = json_schema()
    assert json.loads(json.dumps(schema)) == schema


class TestSections:
    def test_every_entry_is_a_name_required_pair_and_nothing_else(self):
        """Kills an entry that grows a third key (e.g. a description) that
        no consumer asked for and that this schema does not otherwise
        document, and kills an entry that loses `required`."""
        for entry in json_schema()["sections"]:
            assert set(entry) == {"name", "required"}
            assert isinstance(entry["name"], str)
            assert isinstance(entry["required"], bool)

    def test_the_required_section_names_are_exactly_four(self):
        """Pinned as a set equality, not membership: a fifth section
        wrongly marked required would pass a `"runtime" in required` style
        check and still break every consumer that treats the required set
        as exhaustive."""
        sections = json_schema()["sections"]
        required = {entry["name"] for entry in sections if entry["required"]}
        assert required == {"runtime", "observation", "model", "runs"}

    def test_campaign_is_present_and_not_required(self):
        sections = {entry["name"]: entry["required"]
                    for entry in json_schema()["sections"]}
        assert "campaign" in sections
        assert sections["campaign"] is False

    def test_the_section_list_is_derived_from_preflight_s_own_tables(self):
        """Derive, do not re-spell: `_SECTIONS`/`_REQUIRED` are preflight's
        own grammar tables, and this is the guard that a second copy of the
        section list (this schema's) has not drifted from the first."""
        sections = json_schema()["sections"]
        assert [entry["name"] for entry in sections] == list(_SECTIONS)
        assert {entry["name"] for entry in sections if entry["required"]} == (
            set(_REQUIRED)
        )


class TestExits:
    def test_exits_has_exactly_eighteen_entries(self):
        exits = json_schema()["exits"]
        assert len(exits) == 18
        assert len(set(exits)) == 18  # no duplicate exit kind

    def test_exits_includes_forward_nuts_and_plan_sample(self):
        assert {"forward", "nuts", "plan.sample"} <= set(json_schema()["exits"])

    def test_exits_is_the_runs_section_s_own_kind_tuple(self):
        """Derive, do not re-spell: `_KINDS` is the one table `parse_runs`
        actually dispatches on (`sections/runs.py`), so this is the guard
        that the schema's exit list cannot silently diverge from what a
        `runs:` entry may legally name."""
        assert json_schema()["exits"] == list(_EXIT_KINDS)


class TestVocabularies:
    @pytest.mark.parametrize("key", ["operators", "transforms"])
    def test_is_a_nonempty_sorted_list_of_unique_strings(self, key):
        values = json_schema()[key]
        assert values, f"{key} must not be empty"
        assert all(isinstance(v, str) for v in values)
        assert values == sorted(values)
        assert len(values) == len(set(values))

    def test_operators_is_the_sorted_union_of_the_three_value_registries(self):
        """Derive, do not re-spell: operators is documented as the union of
        the value-form, value-modifier and derivation vocabularies, so this
        pins that union rather than a restated literal list that would go
        stale the moment any one registry gains an entry."""
        expected = sorted(
            set(VALUE_FORMS) | set(VALUE_MODIFIERS) | set(DERIVATIONS)
        )
        assert json_schema()["operators"] == expected

    def test_transforms_is_the_sorted_formula_registry(self):
        assert json_schema()["transforms"] == sorted(_FORMULA_REGISTRY)


class TestCatalogs:
    def test_catalogs_has_exactly_the_four_keys(self):
        assert set(json_schema()["catalogs"]) == {
            "acceptedUnits", "resourceKinds", "fileFormats", "shapeSymbols",
        }

    @pytest.mark.parametrize(("key", "source"), [
        ("acceptedUnits", ACCEPTED_UNITS),
        ("resourceKinds", RESOURCE_KINDS),
        ("fileFormats", FILE_FORMATS),
        ("shapeSymbols", SHAPE_SYMBOLS),
    ])
    def test_each_catalog_is_its_own_registry_as_a_list(self, key, source):
        """Derive, do not re-spell, and order-sensitive: these catalogs are
        not sorted (shapeSymbols in particular is grouped by meaning, not by
        the alphabet), so equality against `list(source)` -- not a set
        comparison -- is what would catch a reordering as well as a
        drop/add."""
        catalog = json_schema()["catalogs"][key]
        assert catalog == list(source)
        assert catalog, f"{key} must not be empty"
        assert all(isinstance(v, str) for v in catalog)
