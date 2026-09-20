"""Check A53: a document is told when its physics is partly a stand-in.

The severity is the whole design. A placeholder's CONTRACT is real -- shapes,
purity, PRNG consumption, ordering -- so placing one is not a mistake and must
not refuse; it is not a smell either, so it must not warn. What it is, is a
fact about what the numbers mean. ``findings.py`` bound ``REPORT`` for that and
recorded that nothing produced one yet; this is the first.

**One finding per document, naming every node.** The first version emitted one
per node and a four-node document earned four notices -- seventeen of the
twenty-nine shipped operators are placeholders, so a realistic document earned
a column of them. A notice that repeats is a notice a reader learns to skip,
which costs exactly the nodes it was written for.

The tests that matter are not "does it fire" but "does it stay out of the way"
and "does it read the registry rather than a list of its own".
"""

from __future__ import annotations

import sys
import warnings

import pytest

from rheplicant.config.findings import REPORT, Report
from rheplicant.core.capability import Maturity
from rheplicant.radio import capabilities

#: The module object from ``sys.modules``. ``rheplicant.config.preflight``
#: binds a FUNCTION named ``model``, which shadows the submodule of that name,
#: so neither ``import ... .model`` inside a function nor monkeypatch's dotted
#: string form can walk to it -- both were tried and both raised.
_MODEL_PASS = sys.modules["rheplicant.config.preflight.model"]
_capability_level = _MODEL_PASS._capability_level


def _findings(document):
    return list(_capability_level(document))


def _graph(**nodes):
    return {"model": {"kind": "graph", **nodes}}


class TestItSpeaksForTheRightDocuments:
    def test_a_placeholder_node_is_named_with_its_class(self):
        found = _findings(_graph(uniform_sky={"amplitude": 100.0, "n_pix": 12}))
        assert len(found) == 1
        assert found[0].check == "A53"
        assert found[0].where == "model"
        assert "uniform_sky (SkyOperator)" in found[0].message
        assert "placeholder" in found[0].message

    def test_a_document_of_maintained_physics_is_not_reported(self):
        """The half that makes the notice mean something.

        A check that spoke for every document would be noise, and a reader
        would learn to skip it -- which costs exactly the documents it was
        written for.
        """
        assert _findings(_graph(antenna_loss={"efficiency": 0.9})) == []

    def test_four_placeholder_nodes_earn_one_finding_naming_four(self):
        """The design, asserted rather than described."""
        found = _findings(_graph(
            uniform_sky={"amplitude": 1.0, "n_pix": 4},
            gain={"gain": 1.0},
            noise={"type": "NoiseOperator", "sigma": 0.1},
            bandpass={"response": 1.0},
        ))
        assert len(found) == 1, f"{len(found)} findings; one was the point"
        for node in ("uniform_sky", "gain", "noise", "bandpass"):
            assert node in found[0].message, f"{node} is not named"

    def test_experimental_gets_its_own_clause(self):
        """Placeholder and experimental are different kinds, not two depths.

        A placeholder's numbers are a stand-in; an experimental surface's
        numbers may be right and its contract may still move. One wording for
        both would tell the reader the wrong thing about one of them.
        """
        found = _findings(_graph(flagging={"type": "MomentRFIFlaggingOperator"}))
        assert len(found) == 1
        assert "experimental physics" in found[0].message
        assert "stand-in" not in found[0].message


class TestItStaysOutOfTheWay:
    def _report(self):
        return Report(findings=tuple(
            _findings(_graph(uniform_sky={"amplitude": 100.0, "n_pix": 12}))))

    def test_the_severity_is_report_and_never_refuse_or_warn(self):
        assert [one.severity for one in self._report().findings] == [REPORT]

    def test_a_document_placing_a_placeholder_is_not_refused(self):
        """``raise_if_refused`` must pass, or every worked example breaks.

        Seventeen of the shipped operators are placeholders and the tour uses
        them. A notice that refused would make this package unusable for its
        own documentation.
        """
        report = self._report()
        report.raise_if_refused()
        assert report.of(REPORT)

    def test_it_emits_no_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            self._report().emit_warnings()

    def test_it_is_not_a_verdict(self):
        """``verdicts()`` is what "and nothing else" reads, and this is not one."""
        assert self._report().verdicts() == ()


class TestItReadsTheRegistry:
    def test_raising_a_level_silences_the_notice(self, monkeypatch):
        """What makes the check maintenance-free.

        A node whose physics arrives stops being reported in the commit that
        raises its level, with no edit here.
        """
        real = capabilities()
        monkeypatch.setattr(
            _MODEL_PASS, "capabilities",
            lambda: {**real, "SkyOperator": Maturity.MAINTAINED})
        assert _findings(_graph(uniform_sky={"amplitude": 1.0, "n_pix": 4})) == []

    def test_lowering_a_level_starts_one(self, monkeypatch):
        """The other direction, so the test cannot pass by never firing."""
        real = capabilities()
        monkeypatch.setattr(
            _MODEL_PASS, "capabilities",
            lambda: {**real, "AntennaLossOperator": Maturity.PLACEHOLDER})
        found = _findings(_graph(antenna_loss={"efficiency": 0.9}))
        assert len(found) == 1
        assert "AntennaLossOperator" in found[0].message


class TestItDeclinesRatherThanGuesses:
    @pytest.mark.parametrize(
        "document",
        [
            {},
            {"model": "not a mapping"},
            {"model": {"kind": "pipeline", "stages": []}},
            _graph(),
            _graph(uniform_sky="not a mapping"),
            _graph(not_a_node={"x": 1}),
            _graph(uniform_sky={"type": "NoSuchOperator"}),
        ],
        ids=["no model", "model not a mapping", "pipeline kind", "no nodes",
             "spec not a mapping", "unknown node", "unknown type"],
    )
    def test_shapes_that_name_no_resolvable_class_are_silent(self, document):
        """A decline can lose a notice; it must never invent one.

        Each of these is refused elsewhere, by a check whose subject it is.
        Re-reporting them here would put two sentences in front of a reader
        for one mistake, and the second would be about the wrong thing.
        """
        assert _findings(document) == []
