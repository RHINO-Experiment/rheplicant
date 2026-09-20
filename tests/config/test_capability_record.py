"""``capabilities.json`` and the one resolution check A53 shares with it.

The guard this module exists for is the PARITY one. A record and a notice
answering the same question from two walks is the shape this repository keeps
paying for: nothing renders the two side by side, so they disagree silently
and each looks right on its own. They are one function now, and
:class:`TestTheNoticeAndTheRecordAreOneResolution` is what keeps them one.
"""

from __future__ import annotations

import json

import jsonschema
import pytest

from _rheplicant_bootstrap.audit.bundle import CAPABILITIES_NAME, MERGED_METADATA_PATHS
from _rheplicant_bootstrap.audit.types import ResolvedLayerRecord
from _rheplicant_bootstrap.layering import OriginNode
from _rheplicant_bootstrap.types import LayerIdentity
from rheplicant.config.capability_record import (
    CAPABILITIES_FORMAT_VERSION,
    LEVEL_REASONS,
    at_level,
    capabilities_manifest,
    node_levels,
)
from rheplicant.config.preflight.model import _capability_level
from rheplicant.config.schemas import load_schema
from rheplicant.config.sections.model import operator_table
from rheplicant.core.capability import Maturity
from rheplicant.radio import capabilities

SCHEMA = load_schema("capabilities-v1")


def graph(**nodes):
    return {"model": {"kind": "graph", **nodes}}


def layer(document, *, kind="base", name=None):
    return ResolvedLayerRecord(
        layer=LayerIdentity(kind, name),
        effective_document=document,
        origins=OriginNode(None, {}),
        declared_runs=(),
        execution_runs=(),
        audit={},
    )


class TestTheVocabulariesAreTheRuntimeOnes:
    def test_the_level_enum_is_the_maturity_enum(self):
        """Every value, including ``unavailable``, which no class carries.

        Filtering it out here would be this module deciding what the enum
        means. ``docs/stability.md`` records the reading -- ``Unavailable`` is
        what a SURFACE answers, not a level a class declares -- and the test
        below is what holds it, in the one place it is a fact about the code.
        """
        assert tuple(SCHEMA["$defs"]["level"]["enum"]) == tuple(level.value for level in Maturity)

    def test_the_reason_enum_is_the_runtime_vocabulary(self):
        assert tuple(SCHEMA["$defs"]["levelReason"]["enum"]) == LEVEL_REASONS

    def test_the_declared_format_version_is_the_schemas(self):
        assert SCHEMA["properties"]["format_version"]["const"] == (CAPABILITIES_FORMAT_VERSION)

    def test_every_addressable_operator_declares_a_maturity(self):
        """Why ``LEVEL_REASONS`` has no entry for "the class declares none".

        A third reason would be a vocabulary member nothing can produce, which
        is a clause no test can reach. The condition is impossible instead of
        merely unhandled, and this is the assertion that makes it so: if a
        class ever becomes addressable from a document without declaring a
        ``maturity``, this goes red and the vocabulary question is asked
        deliberately rather than answered by a silent ``null``.
        """
        addressable = {cls.__name__ for row in operator_table().values() for cls in row}
        assert addressable
        assert addressable <= set(capabilities())


class TestEveryReasonIsProducible:
    """The other direction: a vocabulary entry nothing produces is dead too."""

    def test_a_spec_that_is_not_a_mapping_says_so(self):
        rows = node_levels(graph(gain="not a mapping"))
        assert [(row.node_id, row.level, row.level_reason) for row in rows] == [
            ("gain", None, "spec_not_a_mapping")
        ]

    def test_a_type_that_resolves_to_nothing_says_so(self):
        rows = node_levels(graph(gain={"type": "NoSuchOperator"}))
        assert [(row.type, row.level, row.level_reason) for row in rows] == [
            ("NoSuchOperator", None, "unresolved_type")
        ]

    def test_both_reasons_are_reachable(self):
        rows = node_levels(graph(gain="not a mapping", noise={"type": "NoSuchOperator"}))
        assert {row.level_reason for row in rows} == set(LEVEL_REASONS)

    def test_a_node_that_resolves_carries_a_level_and_no_reason(self):
        (row,) = node_levels(graph(gain={"amplitude": 1.0}))
        assert row.type == "GainOperator"
        assert row.level in {level.value for level in Maturity}
        assert row.level_reason is None


class TestTheNoticeAndTheRecordAreOneResolution:
    """A53's message and the published record cannot disagree about a node.

    Not because they are compared at runtime -- they are not -- but because
    they read one function. This is the test that would fail if someone gave
    either of them a walk of its own again.
    """

    @pytest.mark.parametrize(
        "document",
        [
            graph(gain={"amplitude": 1.0}),
            graph(gain={"amplitude": 1.0}, noise={"sigma": 1.0}),
            graph(gain="not a mapping"),
            graph(),
            {},
        ],
    )
    def test_every_node_the_notice_names_is_in_the_record_at_that_level(self, document):
        rows = node_levels(document)
        named = {
            f"{row.node_id} ({row.type})"
            for level in (Maturity.PLACEHOLDER, Maturity.EXPERIMENTAL)
            for row in at_level(rows, level)
        }
        findings = list(_capability_level(document))
        if not named:
            assert findings == []
            return
        (finding,) = findings
        for entry in named:
            assert entry in finding.message

    def test_the_record_also_names_the_maintained_nodes_the_notice_does_not(self):
        """The two are views, not copies, and the views differ on purpose.

        A53 is a notice: silence means nothing to warn about. A record whose
        maintained nodes were missing would be indistinguishable from one
        written before those nodes existed.
        """
        document = graph(gain={"amplitude": 1.0})
        rows = node_levels(document)
        assert [row.node_id for row in rows] == ["gain"]
        assert rows[0].level is not None


class TestTheManifest:
    def test_it_validates_and_carries_one_row_per_resolved_layer(self):
        document = graph(gain={"amplitude": 1.0})
        payload = json.loads(
            capabilities_manifest((layer(document), layer(document, kind="variant", name="v")))
        )
        jsonschema.validate(payload, SCHEMA)
        assert [row["layer"] for row in payload["layers"]] == [
            {"kind": "base", "name": None},
            {"kind": "variant", "name": "v"},
        ]

    def test_a_variant_that_places_a_different_node_gets_its_own_answer(self):
        """Why the record is per LAYER and not per document."""
        payload = json.loads(
            capabilities_manifest(
                (
                    layer(graph(gain={"amplitude": 1.0})),
                    layer(
                        graph(gain={"amplitude": 1.0}, noise={"sigma": 1.0}),
                        kind="variant",
                        name="v",
                    ),
                )
            )
        )
        assert [len(row["nodes"]) for row in payload["layers"]] == [1, 2]

    def test_it_keeps_the_documents_own_node_order(self):
        payload = json.loads(
            capabilities_manifest((layer(graph(noise={"sigma": 1.0}, gain={"amplitude": 1.0})),))
        )
        assert [row["node_id"] for row in payload["layers"][0]["nodes"]] == [
            "noise",
            "gain",
        ]

    def test_it_is_byte_reproducible(self):
        document = graph(gain={"amplitude": 1.0})
        assert capabilities_manifest((layer(document),)) == capabilities_manifest(
            (layer(document),)
        )

    def test_a_layer_that_is_not_an_exact_record_is_refused(self):
        from rheplicant.config.errors import ConfigError

        with pytest.raises(ConfigError, match="exact resolved layers"):
            capabilities_manifest(({"layer": "base", "effective_document": {}},))

    def test_no_layers_is_an_empty_record_rather_than_no_file(self):
        """A published tree that resolved nothing still says so.

        An absent file and a file saying "no layers" are different claims, and
        only the second is one a reader can act on.
        """
        payload = json.loads(capabilities_manifest(()))
        jsonschema.validate(payload, SCHEMA)
        assert payload["layers"] == []


class TestThePublishedNameIsReserved:
    def test_the_name_is_spelled_once(self):
        assert CAPABILITIES_NAME in MERGED_METADATA_PATHS

    def test_it_is_not_in_the_paths_the_transaction_replaces(self):
        """Being in both would refuse its own producer.

        ``RESERVED_BUNDLE_PATHS`` names the three files written last and
        replaced together, and ``merge_bundle_files`` refuses a merge onto one
        of them. ``capabilities.json`` is merged, so it belongs to the other
        tuple -- it is reserved against scientific PRODUCTS, not against the
        transaction.
        """
        from _rheplicant_bootstrap.audit.bundle import RESERVED_BUNDLE_PATHS

        assert CAPABILITIES_NAME not in RESERVED_BUNDLE_PATHS
