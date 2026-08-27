"""The pinned-refusal census: how many sentences this layer owes its callers.

Every ``pytest.raises(..., match=...)`` under ``tests/inference/`` is a claim
that a particular refusal says a particular thing. Those sentences are a
KEEPING SURFACE for the bayesmith migration: after a module moves, the refusal
may be raised on the other side of the seam, and the adapter's ``translate``
has to bring it back wearing this package's class and -- where a test pins it
-- this package's words.

So the population has to be known, and it has to be known **per file**, because
a wave switches modules and needs to answer "which pinned sentences did I just
become responsible for". The full listing, pattern by pattern, is Appendix B of
``2026-08-26-one-implementation.md`` in the bayesmith repository; this file is
what keeps that listing from going stale, by making a new or deleted refusal
site fail here with the number to write down.

**Why a count and not a content assertion.** The content assertion already
exists -- it is the ``match=`` itself, in the test that owns the refusal. What
nothing checked before was the SIZE of the population, which is the thing a
migration plan is written against and the thing that drifts silently: a wave
that deletes a module deletes its refusals with it, and a wave that adds one
adds a sentence nobody promised to keep.

The census is derived by parsing, not by a hand-maintained list, for the reason
this codebase keeps re-learning: a second copy of a fact is the copy that goes
stale, because nothing renders the two side by side.
"""

import ast
import collections
from pathlib import Path

import pytest

_DIRECTORY = Path(__file__).resolve().parent

#: Sites per file, measured 2026-08-27 on the P1 batch. Update this table from
#: the failure message rather than by counting by hand, and update Appendix B
#: in the same commit -- the two are one measurement.
CENSUS: dict[str, int] = {
    "test_block_learning_rate.py": 2,
    "test_declared_prior.py": 14,
    "test_fisher_prior.py": 7,
    "test_forward.py": 1,
    "test_gls.py": 3,
    # 14 -> 13 on 2026-08-27: the refusal that pinned the masking GAP retired
    # when G1's wiring landed. What it protected is pinned positively by
    # TestFlaggedNoiseCrossesAsADeclaredMask.
    "test_graph_bridge.py": 13,
    "test_identifiability.py": 13,
    "test_inference_construction_guards.py": 10,
    "test_inference_unpinned_refusals.py": 5,
    "test_jeffreys_prior.py": 13,
    "test_linear_block_as_dict.py": 2,
    "test_linear_blocks.py": 19,
    "test_linear_groups.py": 21,
    "test_loss_sense.py": 5,
    "test_noise_model.py": 3,
    "test_noise_std_axis.py": 18,
    "test_npe.py": 4,
    # 5 -> 6 on 2026-08-27: D27's collision refusal (a sampled noise_std
    # against a latent of that name). NumPyro already refused it with a bare
    # assertion naming neither side; this package now says it first, in its own
    # exception class.
    "test_numpyro_bridge.py": 6,
    "test_parameters.py": 29,
    "test_plan.py": 32,
    "test_prior_sensitivity.py": 9,
    "test_stochastic_twin.py": 4,
    "test_uncertainty.py": 8,
}

#: Which exception classes those sites name, and how often. Carried because the
#: migration's translation rules are written per CLASS: a ``ParameterSpaceError``
#: is what ``translate`` produces from a bayesmith ``StructureError``, while a
#: ``StateValidationError`` is a pre-validation refusal that must never reach
#: the seam at all. A drift between these two populations is a drift in which
#: refusals the adapter is responsible for.
BY_CLASS: dict[str, int] = {
    "ParameterSpaceError": 175,
    "StateValidationError": 58,
    "RuntimeError": 4,
    "Exception": 3,
    "TypeError": 1,
}


def _sites() -> list[tuple[str, int, str, str]]:
    """``(file, line, exception class, pattern)`` for every pinned refusal.

    Parsed rather than executed: a runtime census would only see the sites a
    given environment actually collected, and several modules here stand down
    behind an ``importorskip``. A thinner virtualenv would then report a
    smaller population and this guard would agree with it -- which is the
    silent-shrink failure it exists to catch.
    """
    found = []
    for path in sorted(_DIRECTORY.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            function = node.func
            name = getattr(function, "attr", None) or getattr(function, "id", None)
            if name != "raises":
                continue
            raised = "<expr>"
            if node.args:
                first = node.args[0]
                raised = getattr(first, "id", None) or getattr(first, "attr", None) or "<expr>"
            for keyword in node.keywords:
                if keyword.arg != "match":
                    continue
                try:
                    pattern = ast.literal_eval(keyword.value)
                except ValueError:
                    pattern = "<computed>"
                found.append((path.name, node.lineno, raised, str(pattern)))
    return found


def test_the_census_finds_pinned_refusals_at_all():
    """The self-check: a parser that matched nothing would make every
    assertion below vacuously true.

    Named separately rather than folded into the counts, because "the scan
    broke" and "the population changed" want different fixes and a combined
    assertion reports the first as the second.
    """
    assert len(_sites()) > 100


def test_every_file_pins_the_number_of_refusals_it_used_to():
    counted = collections.Counter(name for name, _, _, _ in _sites())
    drift = {
        name: (CENSUS.get(name), counted.get(name))
        for name in set(CENSUS) | set(counted)
        if CENSUS.get(name) != counted.get(name)
    }
    assert not drift, (
        "the pinned-refusal population moved: {file: (recorded, counted)} = "
        f"{drift}. Update CENSUS above AND Appendix B of "
        "2026-08-26-one-implementation.md in the bayesmith repository -- they are "
        "one measurement, and the appendix is what a wave reads to find out which "
        "sentences it just became responsible for."
    )


def test_the_total_is_the_number_the_plan_records():
    assert sum(CENSUS.values()) == 241


def test_the_exception_classes_are_the_ones_translate_was_written_against():
    counted = collections.Counter(raised for _, _, raised, _ in _sites())
    assert dict(counted) == BY_CLASS, (
        f"the refusal classes moved: counted {dict(counted)}, recorded {BY_CLASS}. "
        "translate() converts by CLASS, so a new class here is a class the seam "
        "has no rule for."
    )


@pytest.mark.parametrize("name", sorted(CENSUS))
def test_each_censused_file_still_exists(name):
    """An entry for a deleted file would keep its count alive on paper.

    The migration deletes test files -- that is iron law 2 -- so the census has
    to notice a row that no longer describes anything, rather than carrying a
    number for a population of zero.
    """
    assert (_DIRECTORY / name).exists()
