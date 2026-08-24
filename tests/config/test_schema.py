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
from rheplicant.config.errors import ConfigError
from rheplicant.config.preflight import _NOT_YET, _REQUIRED, _RESERVED, _SECTIONS, preflight
from rheplicant.config.schema import json_schema
from rheplicant.config.sections.runs import _KINDS as _EXIT_KINDS
from tests.config.preflight_helpers import preflight_document


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
    def test_every_entry_is_a_name_required_status_triple_and_nothing_else(self):
        """Kills an entry that grows a fourth key (e.g. a description) that
        no consumer asked for and that this schema does not otherwise
        document, and kills an entry that loses `required` or `status`."""
        for entry in json_schema()["sections"]:
            assert set(entry) == {"name", "required", "status"}
            assert isinstance(entry["name"], str)
            assert isinstance(entry["required"], bool)
            assert isinstance(entry["status"], str)

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

    def test_every_status_is_one_of_the_three_known_values(self):
        """`status` is a closed vocabulary -- "accepted", "deferred" or
        "reserved" -- not a free-form string a consumer would have to
        special-case defensively."""
        for entry in json_schema()["sections"]:
            assert entry["status"] in {"accepted", "deferred", "reserved"}

    def test_the_four_refused_sections_carry_a_non_accepted_status(self):
        """`defaults`, `plugins` and `outputs` are refused by `_NOT_YET`
        (`config/preflight/__init__.py`); `campaign` is refused by its own
        clause in the same function, reserved for a later capability rather
        than merely not-yet-implemented. All four must read as something
        other than "accepted", or a form/UI rendered from this schema
        offers a section that can only ever produce an error."""
        statuses = {entry["name"]: entry["status"]
                    for entry in json_schema()["sections"]}
        for name in ("defaults", "plugins", "outputs"):
            assert statuses[name] == "deferred", (name, statuses[name])
        assert statuses["campaign"] == "reserved"

    def test_status_is_computed_from_preflight_s_own_refusal_tables(self):
        """Derive, do not re-spell, weak form: the schema's non-accepted set
        is computed from the very objects `_structural` refuses from --
        `_NOT_YET` (three sections, each routed to a named plan) and
        `_RESERVED` (`campaign`, routed to its own clause) -- and not from a
        fourth list of names written independently in `schema.py`. This is
        the mechanical guard that a name added to, or dropped from, either
        table changes the schema without anyone editing `schema.py` by hand.
        See `TestStatusAgreesWithTheLoader` below for the stronger claim,
        that these tables are what the loader itself actually refuses on --
        this test alone would stay green even if `_structural`'s logic were
        rewired to ignore both tables while `schema.py` kept importing them."""
        sections = json_schema()["sections"]
        non_accepted = {entry["name"] for entry in sections
                         if entry["status"] != "accepted"}
        assert non_accepted == set(_NOT_YET) | set(_RESERVED)
        deferred = {entry["name"] for entry in sections
                    if entry["status"] == "deferred"}
        assert deferred == set(_NOT_YET)
        reserved = {entry["name"] for entry in sections
                    if entry["status"] == "reserved"}
        assert reserved == set(_RESERVED)


class TestStatusAgreesWithTheLoader:
    """The important check (see the task brief): agreement between the
    schema's derived `status` and what `preflight()` -- the real,
    production loader entry point, not a re-implementation of its refusal
    rule -- actually does with a document carrying that section.

    `test_status_is_computed_from_preflight_s_own_refusal_tables` above only
    proves the schema reads the same TABLES `_structural` reads; it would
    stay green even if `_structural`'s own logic stopped consulting those
    tables. Driving an actual `preflight()` call is the stronger claim: the
    sections this schema calls "accepted" truly are accepted by the code
    that decides, and the others truly are refused, and specifically for
    being unknown-to-this-layer, not-yet-read or reserved -- not for some
    unrelated reason -- which is what the message-substring assertions below
    check for.

    `tests.config.preflight_helpers.preflight_document()` builds the one
    valid base document every pre-flight test in this package is a patch of.
    Its own docstring guarantees it "carries all eight sections" this schema
    calls accepted (`schema_version`, `runtime`, `observation`, `resources`,
    `model`, `variants`, `inference`, `runs`) and that the base "MUST EARN NO
    FINDING OF ITS OWN" -- so the unpatched base is exactly the fixture that
    shows the eight accepted sections pass the real loader together, and
    patching in one of the other four is exactly how the fixture reaches
    `_structural`'s refusal for that one section and nothing else.
    """

    def test_the_accepted_sections_are_exactly_what_the_base_document_carries(self):
        assert set(preflight_document()) == {
            entry["name"] for entry in json_schema()["sections"]
            if entry["status"] == "accepted"
        }

    def test_the_unpatched_document_is_accepted_by_the_real_loader(self):
        """No `ConfigError`, no findings: every section this schema calls
        accepted passes `preflight()` when present together, which is the
        loader's own definition of "accepted"."""
        assert preflight(preflight_document()).findings == ()

    @pytest.mark.parametrize("name", ["outputs", "defaults", "plugins", "campaign"])
    def test_a_non_accepted_section_is_refused_by_the_real_loader(self, name):
        """Adding exactly the section this schema calls non-accepted to the
        otherwise-clean base document is what reaches `_structural`'s
        refusal -- the same function, the same call, that `json_schema()`
        never touches."""
        statuses = {entry["name"]: entry["status"]
                    for entry in json_schema()["sections"]}
        assert statuses[name] != "accepted"
        document = preflight_document(**{name: {}})
        with pytest.raises(ConfigError) as caught:
            preflight(document)
        message = str(caught.value)
        if name == "campaign":
            assert statuses[name] == "reserved"
            assert "is reserved" in message
        else:
            assert statuses[name] == "deferred"
            assert "is not read by this layer yet" in message


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
