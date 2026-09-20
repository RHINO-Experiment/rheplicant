"""Two published arrays that a real run never filled.

``tests/config/test_audit_schemas.py`` now proves every array the audit
schemas declare has a golden that EXERCISES its item type. That is a statement
about the schemas and the goldens. It is silent about whether the producer
ever puts a row in one, and the answer for two of them was no.

Measured 2026-09-20 on a real ``rheplicant run``:

* a document declaring one variant published
  ``variants/n-756e6974795f6761696e/config.resolved.yaml``, recorded the
  variant in ``path_encodings``, recorded one row in
  ``artefacts.resolved_variants`` -- and ``provenance.variants`` was ``[]``.
  Three views agreed there was one variant and the fourth said none.
* ``diagnostics.deferred_validations`` is always ``[]``, although
  ``benchmark``, ``compare`` and ``predict`` each declare deferred checks and
  the parsed-run record already carries them.

Both traced to the same cause: ``AuditTrace.record_variant`` and
``AuditTrace.record_deferred_validation`` had **no caller in** ``src/``. The
A10-7 audit line found them and filed them as dead code to delete; the
measurement says the opposite, because the arrays they fill are in a closed
published schema. Deleting them would have meant deleting a declared array and
a ``format_version`` bump, to remove information three other views already had.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from _rheplicant_bootstrap.audit.names import encode_name
from tests.config.test_config_cli import document
from tests.config.test_config_document import synthetic_document


@pytest.fixture
def published_with_a_variant(tmp_path):
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    value = synthetic_document()
    assert value.get("variants"), "the synthetic document no longer declares one"
    value["outputs"] = {"dir": str(target)}
    config = tmp_path / "config.yaml"
    config.write_bytes(yaml.safe_dump(value, sort_keys=False).encode())
    assert main(["run", str(config)]) == 0
    return target, tuple(value["variants"])


def test_provenance_records_every_declared_variant(published_with_a_variant):
    """The row per variant, against the three views that already agreed."""
    target, declared = published_with_a_variant
    provenance = json.loads((target / "provenance.json").read_bytes())

    encodings = [
        row for row in provenance["path_encodings"] if row["kind"] == "variant"
    ]
    assert len(encodings) == len(declared)
    assert len(provenance["artefacts"]["resolved_variants"]) == len(declared)

    rows = provenance["variants"]
    assert len(rows) == len(declared), (
        "provenance.variants disagrees with path_encodings, the artefact table "
        "and the published variant directories about how many variants this "
        f"document has: {rows}"
    )
    for row, name in zip(rows, declared, strict=True):
        assert row["layer"] == {"kind": "variant", "name": name}
        assert row["encoded_name"] == encode_name(name)
        assert row["status"] == "ok"
        assert isinstance(row["resolved_sha256"], str)


def test_the_recorded_digest_is_the_published_variants_own(published_with_a_variant):
    """A digest that is not of the file beside it is worse than no digest."""
    import hashlib

    target, _declared = published_with_a_variant
    provenance = json.loads((target / "provenance.json").read_bytes())
    # Without this line the loop below is empty and the test passes by having
    # nothing to check, which is the state it was written to end.
    assert provenance["variants"]
    for row in provenance["variants"]:
        published = target / "variants" / row["encoded_name"] / "config.resolved.yaml"
        assert published.is_file(), published
        assert row["resolved_sha256"] == hashlib.sha256(published.read_bytes()).hexdigest()


def test_a_document_without_variants_records_none(tmp_path):
    """The other direction, so the row cannot be invented."""
    from _rheplicant_bootstrap.cli import main

    target = tmp_path / "result"
    config = tmp_path / "config.yaml"
    config.write_bytes(
        yaml.safe_dump(document(output=target), sort_keys=False).encode()
    )
    assert main(["run", str(config)]) == 0
    provenance = json.loads((target / "provenance.json").read_bytes())
    assert provenance["variants"] == []
    assert provenance["artefacts"]["resolved_variants"] == []


def test_deferred_validations_are_the_parsed_runs_deferred_checks():
    """One source for a fact recorded twice.

    ``ParsedRunRecord.deferred_checks`` and ``DeferredValidationRecord.checks``
    were the same ``(layer, descriptor, checks)`` triple under two names, and
    only the first had a caller -- so the second's array published empty while
    the information sat in the first. The projection reads the parsed runs now
    and the duplicate record is gone.
    """
    from _rheplicant_bootstrap.audit import AuditTrace
    from _rheplicant_bootstrap.audit.diagnostics import build_diagnostics
    from _rheplicant_bootstrap.types import LayerIdentity

    layer = LayerIdentity("base", None)
    trace = AuditTrace()
    plain = {"index": 0, "name": "fit", "kind": "nuts", "variant": None}
    deferring = {"index": 1, "name": "against", "kind": "compare", "variant": None}
    trace.record_parsed_run(
        layer, {"descriptor": plain, "resolved_options": {}, "deferred_checks": ()}
    )
    trace.record_parsed_run(
        layer,
        {
            "descriptor": deferring,
            "resolved_options": {},
            "deferred_checks": ("compare.left_available", "compare.right_available"),
        },
    )
    rows = build_diagnostics(trace.snapshot(), status="ok")["deferred_validations"]
    assert [row["descriptor"]["name"] for row in rows] == ["against"], (
        "a run with no deferred checks has nothing to defer and must not "
        f"produce a row: {rows}"
    )
    assert rows[0]["checks"] == (
        "compare.left_available",
        "compare.right_available",
    )
    assert rows[0]["layer"] == {"kind": "base", "name": None}


def test_the_run_kinds_that_defer_actually_declare_checks():
    """Guard the guard: if no kind deferred anything, the array would be empty
    for a correct reason and the test above would be pinning a fixture."""
    from rheplicant.config.sections.benchmark import _DEFERRED as BENCHMARK
    from rheplicant.config.sections.comparison import _DEFERRED as COMPARE
    from rheplicant.config.sections.diagnostics import _PREDICT_DEFERRED as PREDICT

    assert BENCHMARK and COMPARE and PREDICT


def test_no_audit_recorder_is_without_a_caller():
    """The census that would have caught both of these.

    A ``record_*`` method nobody calls is an array that publishes empty, and
    neither the schema nor a golden can see it: the schema says the array may
    carry rows, and a golden built from a hand-made snapshot carries whatever
    the fixture recorded. Only the production call sites answer it.
    """
    import re

    root = Path(__file__).resolve().parents[2] / "src"
    trace = (root / "_rheplicant_bootstrap" / "audit" / "trace.py").read_text()
    recorders = set(re.findall(r"^    def (record_\w+)", trace, re.M))
    assert recorders

    sources = [
        path.read_text(encoding="utf-8")
        for path in root.rglob("*.py")
        if path.name != "trace.py"
    ]
    uncalled = sorted(
        name for name in recorders if not any(f".{name}(" in text for text in sources)
    )
    assert not uncalled, (
        f"these audit recorders have no caller in src/: {uncalled}. Either the "
        "array they fill publishes empty for every run, or the recorder is "
        "genuinely dead -- and the two need opposite fixes, so decide which "
        "rather than leaving it"
    )
