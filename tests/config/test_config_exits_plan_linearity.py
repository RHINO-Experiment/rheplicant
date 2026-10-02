"""``check_linearity:`` on the two plan run kinds, from a document.

A document that digitises declines its linearity gate in writing
(``inference.checks.linearity: {mode: skip, reason: ...}``): a converter clips
at the check's outermost probe whatever the data are. ``SamplingPlan`` runs
the same check itself before its first sweep, so until the plan kinds took
this key such a document loaded and then could not run a plan. The conjugate
kinds have ``check: false`` for the same purpose.

The document is ``preflight_helpers.t5_case`` with the most benign converter
the package builds: twelve bits at one count per kelvin, a peak of 12 counts
against a clip at 2048.
"""

import re
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from rheplicant.config.sections.runs import run_document
from rheplicant.core.errors import LinearityRefused
from tests.config import preflight_helpers as H

ESTIMATE = {"kind": "plan.estimate", "blocks": [{"names": ["g"]}]}
SAMPLE = {
    "kind": "plan.sample",
    "blocks": [{"names": ["g"]}],
    "seed": {"from": "runtime.seeds.sample"},
    "n_sweeps": 10,
    "warmup": 4,
}
WARM = {"kind": "plan.estimate", "blocks": [{"names": ["g"]}], "move": ["g"]}
DECLINED = {"check_linearity": False}


def digitising(run: dict) -> dict:
    """A document with an unsaturated converter and the linearity gate declined."""
    return H.t5_case(
        model=H.t5_model(H.ADC_UNSATURATED),
        runtime={"seeds": {"sample": 11}},
        runs=[run],
    )


class TestTheDefault:
    @pytest.mark.parametrize("run", [ESTIMATE, SAMPLE], ids=["estimate", "sample"])
    def test_a_digitising_document_cannot_run_a_plan(self, run):
        """The gate is declined and the document loads; the plan's own check refuses."""
        with pytest.raises(LinearityRefused):
            run_document(digitising(run))

    @pytest.mark.parametrize("run", [ESTIMATE, SAMPLE], ids=["estimate", "sample"])
    def test_true_is_the_default_written_out(self, run):
        with pytest.raises(LinearityRefused):
            run_document(digitising({**run, "check_linearity": True}))


class TestDeclined:
    def test_the_estimate_runs_and_recovers_the_truth(self):
        results = run_document(digitising({**ESTIMATE, **DECLINED}))
        estimate = results["plan.estimate"].product
        assert estimate.diagnostics.converged is True
        assert estimate.diagnostics.engines[("g",)] == "conjugate"
        assert float(estimate.values["g"]) == pytest.approx(H.TRUTH_G, abs=0.05)

    def test_the_sample_runs_and_recovers_the_truth(self):
        results = run_document(digitising({**SAMPLE, **DECLINED}))
        draws = results["plan.sample"].product
        assert draws.n_draw == 6
        assert float(draws.mean["g"]) == pytest.approx(H.TRUTH_G, abs=0.2)

    def test_the_identifiability_check_still_runs(self):
        """One check is declined. The rank test is not, and its report lands."""
        results = run_document(digitising({**ESTIMATE, **DECLINED}))
        report = results["plan.estimate"].product.diagnostics.identifiability
        assert report is not None
        assert report.rank == 1

    def test_a_warm_start_declines_for_itself(self):
        """The warm start is its own ``estimate()`` call and reads its own key.

        A gradient block has no linear claim to check, so the sampled run
        passes with or without the key and the two documents differ only in
        the warm start.
        """
        gradient = {**SAMPLE, "blocks": [{"names": ["g"], "engine": "gradient"}], "n_sweeps": 6}
        with pytest.raises(LinearityRefused):
            run_document(digitising({**gradient, "warmup": 2, "warm_start": WARM}))
        results = run_document(
            digitising({**gradient, "warmup": 2, "warm_start": {**WARM, **DECLINED}})
        )
        draws = results["plan.sample"].product
        assert draws.diagnostics.engines[("g",)] == "gradient"
        assert np.all(np.isfinite(np.asarray(draws.samples["g"])))


class TestTheValidationPagesSentence:
    """``docs/config-validation.md`` names the run kinds that repeat the check.

    "With the gate declined and the run's key left at its default, the run is
    refused with ``LinearityRefused`` when it starts." The page lists the
    kinds and which key each takes, and the list is read here from the
    loader's own table of the keys each kind accepts, so a kind that gains a
    ``check`` key and is missing from the page fails. The check is reached
    from a document only through ``linear_operator``, behind ``check``, and
    through ``SamplingPlan``, behind ``check_linearity``.
    """

    WIENER: ClassVar[dict] = {"kind": "conjugate.wiener", "names": ["g"], "width": "none"}

    @staticmethod
    def _sentence() -> str:
        page = (Path(__file__).resolve().parents[2] / "docs" / "config-validation.md").read_text()
        flat = " ".join(page.split())
        start = flat.index("The gate decides whether the document loads.")
        return flat[start : flat.index("when it starts.", start)]

    def test_a_conjugate_run_is_refused_at_its_default_and_runs_with_its_key(self):
        with pytest.raises(LinearityRefused):
            run_document(digitising(self.WIENER))
        results = run_document(digitising({**self.WIENER, "check": False}))
        assert results["conjugate.wiener"].error is None

    def test_the_page_names_every_kind_that_takes_either_key(self):
        from rheplicant.config.preflight.document import _task3_allowed_run_options

        allowed = _task3_allowed_run_options()
        by_key = {
            key: sorted(kind for kind, keys in allowed.items() if key in keys)
            for key in ("check", "check_linearity")
        }
        assert by_key == {
            "check": ["condition", "conjugate.gcr", "conjugate.gls", "conjugate.wiener"],
            "check_linearity": ["plan.estimate", "plan.sample"],
        }
        sentence = self._sentence()
        named = set(re.findall(r"`((?:conjugate|plan)\.[a-z]+|condition)`", sentence))
        assert named == set(by_key["check"]) | set(by_key["check_linearity"])
        assert "six run kinds" in sentence
        assert "take `check: false`" in sentence
        assert "take `check_linearity: false`" in sentence
        assert "refused with `LinearityRefused`" in sentence


class TestTheResolvedDocument:
    @pytest.mark.parametrize("run", [ESTIMATE, SAMPLE], ids=["estimate", "sample"])
    def test_the_default_is_resolved_to_true(self, run):
        """An omitted key is recorded as the value the plan is called with."""
        from _rheplicant_bootstrap.variants import LayerRef
        from rheplicant.config.document import load_document
        from rheplicant.config.sections.exit_support import parse_run
        from rheplicant.config.sections.runs import parse_runs

        document = digitising(run)
        built = load_document(document)
        (spec,) = parse_runs(document["runs"])
        parsed = parse_run(
            spec,
            built,
            index=0,
            layer=LayerRef(kind="base", name=None, prefix="", document={}, declared_runs=None),
        )
        assert parsed.options["check_linearity"] is True
        assert parsed.parsed.resolved["check_linearity"] is True
