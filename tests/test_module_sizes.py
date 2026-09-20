"""Every source file over 800 lines is either split, or justified here.

``CLAUDE.md``'s own style note asks for many small files, 200-400 typical and
800 maximum. Thirty files in ``src/`` are over it. A rule with thirty silent
exceptions is not a rule, so each one is written down with a status and a
reason, and both directions are asserted: a file that grows past the threshold
lands here as a red test, and a file that shrinks below it must leave.

Two statuses, and the difference is a decision rather than a degree.

``SPLIT``
    Stage 1's A1 audit ruled this file should be split and it has not been.
    That is a debt, recorded as a debt. :func:`test_the_unsplit_debt_only_ever_falls`
    is the ratchet: the number may fall and may not rise, so the next person
    to grow one of these past the threshold has to say so.

``SINGLE``
    One file on purpose. The burden is a reason that survives a sceptical
    reader -- "it is cohesive" is not one, and the length floor below is there
    to catch that sentence being pasted in.

What this file does NOT do is assert a line count per file. Pinning 2937 would
make every edit to ``layering.py`` a red test for no benefit, and the number
that matters is the threshold, not the distance past it.
"""

from __future__ import annotations

import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

#: CLAUDE.md's stated maximum. A file at or under it needs no entry.
LIMIT = 800

SPLIT = "split"
SINGLE = "single"

#: path relative to ``src/`` -> (status, why).
LARGE: dict[str, tuple[str, str]] = {
    # --- Stage 1's A1 audit ruled these nine should be split (evidence:
    # .agents/evidence/T-002/audit/ledger_raw.md, "Large files: 9 split").
    "_rheplicant_bootstrap/layering.py": (
        SPLIT,
        "document layering, per-value origin evidence and the variant "
        "merge are three subjects in one file; the origin record alone is a "
        "module",
    ),
    "rheplicant/config/preflight/fitting.py": (
        SPLIT,
        "the checks needing runs[] and inference: together, which is a "
        "join rather than a subject -- the per-run and per-block halves split",
    ),
    "rheplicant/inference/linear.py": (
        SPLIT,
        "checking a linearity claim and exporting the operator are two "
        "jobs; the solve helpers are a third",
    ),
    "_rheplicant_bootstrap/plugins.py": (
        SPLIT,
        "audited import, the closed JSON projection and the refusal "
        "vocabulary are separable, and only the first needs the audit trail",
    ),
    "rheplicant/config/preflight/model.py": (
        SPLIT,
        "schema §6's model checks are a dozen independent passes sharing "
        "one node walk; the walk is the module and the passes are not",
    ),
    "rheplicant/inference/chain.py": (
        SPLIT,
        "the drift model and the recursion that integrates it are "
        "different subjects with different test shapes",
    ),
    "rheplicant/config/dimensions.py": (
        SPLIT,
        "the normalized signature and the closed A9 registries are related by use, not by subject",
    ),
    "rheplicant/core/graph.py": (
        SPLIT,
        "the template grammar and graph-guided assembly are two halves "
        "that only meet at the compiled result",
    ),
    "_rheplicant_bootstrap/output/manager.py": (
        SPLIT,
        "the output grammar, descriptor preflight and A34 lease "
        "management are three, and the lease is the one with its own lifecycle",
    ),
    # --- Grew past the threshold AFTER the audit drew its list.
    "rheplicant/inference/plan.py": (
        SPLIT,
        "crossed 800 with the G1 estimate work and was never assessed "
        "by A1; the point-estimate and draw exits are the natural seam",
    ),
    # --- One file on purpose.
    "_rheplicant_bootstrap/output/transaction.py": (
        SINGLE,
        "one recovery protocol: every step exists to make the others "
        "undoable, and a reader following a failed publication needs all of it "
        "on one screen",
    ),
    "rheplicant/config/orchestration.py": (
        SINGLE,
        "the load is ONE ordered pipeline and its order is the "
        "subject; splitting it would put the sequence in a fourth place that "
        "no test reads",
    ),
    "rheplicant/inference/memory.py": (
        SINGLE,
        "the stored form and what may be asked of it are one contract "
        "-- a reader of either needs the other to know what survives archiving",
    ),
    "_rheplicant_bootstrap/audit/trace.py": (
        SINGLE,
        "append-only storage with a threading contract; the locking "
        "argument is the file and cannot be half-read",
    ),
    "rheplicant/inference/compress.py": (
        SINGLE,
        "one epoch to one likelihood factor, in four routes that share "
        "the same QR and differ only in what they whiten by",
    ),
    "_rheplicant_bootstrap/entry.py": (
        SINGLE,
        "one ordered entry pipeline shared by the CLI and generated "
        "programs; its ORDER is the guarantee, same argument as orchestration",
    ),
    "rheplicant/config/sections/diagnostics.py": (
        SINGLE,
        "the cheap diagnostics are one family with one gating rule, "
        "and the per-check parsers are short",
    ),
    "rheplicant/gui/form_catalog.py": (
        SINGLE,
        "the gateway file: its length IS the config-layer boundary, "
        "because every other GUI module takes its vocabulary from this "
        "__all__ rather than from config (CLAUDE.md)",
    ),
    "rheplicant/config/sections/conjugate.py": (
        SINGLE,
        "one shared opening and one exit per family member; the "
        "opening is what makes the exits comparable",
    ),
    "_rheplicant_bootstrap/process.py": (
        SINGLE,
        "the process-entry and runtime grammar is one closed "
        "vocabulary read before anything is importable",
    ),
    "rheplicant/radio/instrument/calibration.py": (
        SINGLE,
        "the tone, the switched load and the applied solution are one "
        "calibration story and share the lineshape algebra",
    ),
    "rheplicant/inference/graph_bridge.py": (
        SINGLE,
        "the adapter to bayesmith is one seam; splitting it would put "
        "half a translation in each of two files",
    ),
    "rheplicant/inference/engines.py": (
        SINGLE,
        "the engines and the conditioning they share -- separating "
        "them would duplicate the conditioning or export it privately",
    ),
    "rheplicant/inference/uncertainty.py": (
        SINGLE,
        "one propagation argument carried end to end; the intermediate "
        "forms are not independently meaningful",
    ),
    "rheplicant/config/kinds/beams.py": (
        SINGLE,
        "a beam resource is the array plus everything the file cannot "
        "say about it, and the second half only makes sense beside the first",
    ),
    "rheplicant/radio/sky/driftscan.py": (
        SINGLE,
        "the m-mode derivation is one argument from geometry to the "
        "fast path, and its constants are meaningless apart from it",
    ),
    "rheplicant/inference/parameters.py": (
        SINGLE,
        "what is inferred and how it enters the model are two sides of "
        "one declaration and are read together",
    ),
}


def _oversized() -> dict[str, int]:
    """Relative path -> line count, for every file in ``src/`` over the limit."""
    found = {}
    for path in sorted(SRC.rglob("*.py")):
        lines = len(path.read_text(encoding="utf-8").splitlines())
        if lines > LIMIT:
            found[str(path.relative_to(SRC))] = lines
    return found


def test_every_oversized_file_is_accounted_for():
    """Direction one: a file grew past the limit and nobody said anything."""
    missing = sorted(set(_oversized()) - set(LARGE))
    assert not missing, (
        f"these files are over {LIMIT} lines and have no entry: {missing}. "
        "Either split the file, or add it here with a reason a sceptical "
        "reader would accept -- thirty silent exceptions is not a rule"
    )


def test_no_entry_outlives_the_file_it_describes():
    """Direction two: a file shrank, or moved, and its excuse stayed.

    An entry for a file that is no longer large reads as a standing permission
    for it to grow back, and the reason it carries has stopped being true.
    """
    live = _oversized()
    stale = sorted(name for name in LARGE if name not in live)
    present = {str(p.relative_to(SRC)) for p in SRC.rglob("*.py")}
    assert not stale, {
        "no longer over the limit": [n for n in stale if n in present],
        "no longer exists": [n for n in stale if n not in present],
    }


@pytest.mark.parametrize("name", sorted(LARGE), ids=sorted(LARGE))
def test_each_entry_carries_a_real_reason(name):
    """A length floor, because "it is cohesive" is what gets pasted in."""
    status, reason = LARGE[name]
    assert status in (SPLIT, SINGLE), f"{name} has status {status!r}"
    assert len(reason) > 60, (
        f"{name} needs a reason, not a label: {reason!r}. What would a reader "
        "who thinks this file should be split need to hear?"
    )


def test_the_unsplit_debt_only_ever_falls():
    """A ratchet on the files A1 ruled should be split.

    Ten today. The assertion is ``<=``, so splitting one is a green commit and
    nothing has to be edited here except the entry that leaves; growing an
    eleventh file past the limit and marking it SPLIT is a red test that says
    the debt went up, which is the moment to argue for it rather than after.
    """
    debt = sorted(name for name, (status, _) in LARGE.items() if status == SPLIT)
    assert len(debt) <= 10, (
        f"{len(debt)} files are marked for splitting and the recorded debt is "
        f"10: {debt}. Lower the number here only when a file is actually split"
    )
