"""The streaming evidence layer is reachable from no run kind, and that is a fact.

``campaign:`` is a RESERVED section that raises, not a run kind, and the
eighteen kinds ``runs[].kind:`` accepts reach ``rheplicant.inference``
twenty-eight ways without once touching the evidence stack. That was true
when the migration spec was written and nothing said so, so nothing would
have noticed it changing -- in either direction.

Both directions matter. A nineteenth kind arriving is a decision someone
should make deliberately; and the evidence stack acquiring a config caller
while ``campaign:`` still refuses would mean the layer had been wired in
through a side door.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.runs import _KINDS

#: The six modules that ARE rheplicant's streaming evidence layer. Named
#: rather than pattern-matched: `compress` and `compressed` differ by one
#: character and a glob that meant to catch both has caught one before.
EVIDENCE_MODULES = frozenset(
    {"sqrtinfo", "compress", "compressed", "factorize", "archive", "reduced_basis"}
)

CONFIG_ROOT = pathlib.Path(__file__).resolve().parents[2] / "src" / "rheplicant" / "config"


def _imported_inference_modules(path: pathlib.Path) -> set[str]:
    """Every ``rheplicant.inference.<name>`` this file imports, by AST.

    By AST and not by text, deliberately. ``config/sections/diagnostics.py``
    cites ``reduced_basis.py:159-168`` five times in prose -- naming the bug
    a comparison order reintroduces -- and a ``grep`` for the module name
    reports those as imports. The parse also sees imports deferred inside
    function bodies, which this layer uses throughout to keep jax off the
    import path, and which a scan of the module header would miss entirely.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[:2] == ["rheplicant", "inference"]:
                if len(parts) > 2:
                    found.add(parts[2])
                # `from rheplicant.inference import X` -- X may be a module
                found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if parts[:2] == ["rheplicant", "inference"] and len(parts) > 2:
                    found.add(parts[2])
    return found


def test_the_run_kinds_are_eighteen():
    """A nineteenth is a decision, not a drift.

    Pinned by equality AND spelled out, so the failure names which kind
    arrived rather than only that the count moved.
    """
    assert len(_KINDS) == 18, sorted(_KINDS)
    assert set(_KINDS) == {
        "forward", "fisher", "optimize", "plan.estimate", "plan.sample",
        "conjugate.wiener", "conjugate.gcr", "conjugate.gls", "condition",
        "identifiability", "score_directions", "gradient", "mmodes",
        "predict", "nuts", "npe", "compare", "benchmark",
    }


def test_no_run_kind_is_a_streaming_evidence_kind():
    """None of the eighteen names the capability ``campaign:`` reserves."""
    assert not {kind for kind in _KINDS if "campaign" in kind or "evidence" in kind}


def test_the_config_layer_imports_no_module_of_the_evidence_stack():
    """Measured across the whole layer, not asserted about one file.

    The config layer imports twenty-eight names from ``rheplicant.inference``
    and none of them is from the evidence stack. If that changes, either a
    run kind has grown a streaming face -- in which case ``campaign:``
    should stop refusing -- or the layer has been wired in through a side
    door, which is the one this catches.
    """
    offenders: dict[str, set[str]] = {}
    scanned = 0
    for path in sorted(CONFIG_ROOT.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        scanned += 1
        hits = _imported_inference_modules(path) & EVIDENCE_MODULES
        if hits:
            offenders[str(path.relative_to(CONFIG_ROOT))] = hits
    # The scan must actually have run: an empty result from a wrong root
    # would read exactly like a clean layer.
    assert scanned > 40, scanned
    assert offenders == {}, offenders


def test_the_scan_can_still_fail():
    """Anti-vacuity: the AST walk really does see a deferred import.

    Every import of ``rheplicant.inference`` in this layer is deferred into
    a function body to keep jax off the import path, so a scan that only
    read module headers would report a clean layer forever. Checked on a
    synthetic file rather than by mutating a real one.
    """
    probe = ast.parse(
        "def f():\n"
        "    from rheplicant.inference.sqrtinfo import SqrtInfo\n"
        "    return SqrtInfo\n"
    )
    found: set[str] = set()
    for node in ast.walk(probe):
        if isinstance(node, ast.ImportFrom) and node.module:
            parts = node.module.split(".")
            if parts[:2] == ["rheplicant", "inference"] and len(parts) > 2:
                found.add(parts[2])
    assert found & EVIDENCE_MODULES == {"sqrtinfo"}


def test_campaign_is_still_a_reserved_section_that_refuses():
    """The other half: the layer refuses ``campaign:`` in so many words.

    Driven through the real loader rather than read off ``_RESERVED``,
    because the refusal is spelled twice on purpose (the literal is pinned
    by three guards) and this is the copy a user actually meets.
    """
    from rheplicant.config.preflight import _structural

    with pytest.raises(ConfigError) as refused:
        _structural({"schema_version": 1, "campaign": {"epoch_id": "x"}})
    assert "reserved with capability 4 (streaming evidence" in str(refused.value)


class TestB12sPremiseIsFalse:
    """The migration spec's B12 says `prior_sensitivity` is "only a Python API".

    The row, verbatim: *rheplicant 无对应 run kind，只有 Python API。* Half of
    that is right -- there is no run kind, which
    :func:`test_the_run_kinds_are_eighteen` above pins. The other half is
    wrong: `prior_sensitivity` has a full config face, as a gated CHECK
    rather than a run kind, and it is live.

    The consequence is not cosmetic. B12 is scoped as "implement it here,
    INCLUDING its config face", and bayesmith has no config layer at all --
    so the row cannot be executed as written, and the real question it hides
    is whether the check shape or a new run-kind shape is wanted.

    The paired half of this guard -- that the spec still SAYS it, and is
    still tagged `[R1]` rather than as measured -- lives in bayesmith at
    ``tests/crosscheck/test_spec_claims.py``. Split because ``rheplicant`` is
    installed there with ``--no-deps`` and ``rheplicant.config`` needs yaml;
    measured, ``ModuleNotFoundError: No module named 'yaml'``. Each half runs
    where it is real rather than skipping where it is not.
    """

    def test_the_check_is_declared_with_a_default_and_a_finding_id(self):
        from rheplicant.config.gating import CHECK_ID, CHECK_NAMES, DEFAULT_MODE, OFF

        assert "prior_sensitivity" in CHECK_NAMES
        assert DEFAULT_MODE["prior_sensitivity"] == OFF
        assert CHECK_ID["prior_sensitivity"] == "C19"

    @pytest.mark.parametrize("mode", ["refuse", "warn", "report"])
    def test_every_writable_mode_is_honoured_through_the_real_gate(self, mode):
        """Driven, not read off a constant -- a table can be right and unused."""
        from rheplicant.config.gating import gates

        gate = gates({"prior_sensitivity": {"mode": mode, "report": True}})[
            "prior_sensitivity"
        ]
        assert gate.state == mode
        assert gate.record is True

    def test_skip_carries_its_reason_and_does_not_record(self):
        from rheplicant.config.gating import gates

        gate = gates(
            {"prior_sensitivity": {"mode": "skip", "reason": "not wanted"}}
        )["prior_sensitivity"]
        assert gate.state == "skip"
        assert gate.reason == "not wanted"

    def test_an_invalid_mode_produces_a_named_refusal(self):
        """It VALIDATES, which is what makes it a face rather than a dict.

        ``check_gates`` RETURNS findings rather than raising -- worth stating,
        because a probe that expected an exception here reads the absence of
        one as "validation is missing" and concludes the opposite of the truth.
        """
        from rheplicant.config.gating import check_gates

        findings = check_gates({"prior_sensitivity": {"mode": "nonsense"}})
        assert len(findings) == 1, findings
        assert findings[0].check == "A1"
        assert findings[0].severity == "refuse"
        assert findings[0].where == "inference.checks.prior_sensitivity"
        assert "is one of" in findings[0].message

    def test_the_gate_is_wired_to_the_package_function(self):
        """A gate with nothing behind it would make B12's claim true after all."""
        from rheplicant.config.postflight import fitting
        from rheplicant.inference.sensitivity import prior_sensitivity

        assert callable(prior_sensitivity)
        assert callable(fitting._prior_sensitivity)

    def test_the_config_path_is_reachable_from_the_gui_catalog(self):
        """The face is published, not merely parsed.

        The catalog carries ``inference.checks.*.mode`` as a WILDCARD over
        check names rather than one row per check -- which is why grepping
        the catalog for ``prior_sensitivity`` finds nothing and reads like an
        absence.
        """
        from rheplicant.gui.form_catalog import build_catalog

        paths = {widget.path for widget in build_catalog().widgets}
        assert {"inference.checks.*.mode", "inference.checks.*.report"} <= paths
