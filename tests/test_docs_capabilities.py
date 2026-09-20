"""The capability page is a view of the registry, not a copy of it.

Stage 4.1 asks for a capability matrix and a not-implemented list "generated
from the registry". Generating them once is not the same as keeping them
generated: a table that was correct when written is exactly the thing that
goes stale the first time somebody raises a level, and nothing about a
markdown file says it used to be derived.

The page itself is written at BUILD time, into the gitignored
``docs/_generated/`` beside the graph diagram, so there is no checked-in copy
to go stale. What can still go wrong is the generator: a table that no longer
names what the registry holds, or a level with no explanation. Those are what
this file checks, against a page generated here rather than against one found
on disk -- a test that read the build's output would pass or fail depending on
whether anyone had built the docs.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "docs" / "_generate_capabilities.py"
PAGE = ROOT / "docs" / "capabilities.md"


@pytest.fixture(scope="module")
def generator():
    spec = importlib.util.spec_from_file_location("_generate_capabilities", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generated(generator, tmp_path_factory) -> str:
    """The page as the docs build would write it, written here."""
    target = tmp_path_factory.mktemp("capabilities") / "capabilities.md"
    generator.main(target)
    return target.read_text(encoding="utf-8")


def test_the_generator_runs_and_writes_a_table(generated):
    assert "| Level | Count | What it means |" in generated
    assert "## What is not implemented" in generated


def test_the_page_includes_the_generated_table_rather_than_restating_it():
    """The wrapper carries the prose; the numbers come from the include."""
    text = PAGE.read_text(encoding="utf-8")
    assert "{include} _generated/capabilities.md" in text
    # No level counts written out in the prose. A number here is a number that
    # nothing regenerates, and every count on this page has a generated home.
    import re

    for match in re.finditer(r"\b(\d+)\s+(classes|placeholders?|maintained)\b", text):
        raise AssertionError(
            f"{PAGE.name} states a count in prose: {match.group(0)!r}. The "
            "generated table holds the counts"
        )


def test_every_level_the_registry_can_produce_has_a_meaning(generator):
    """A level with no explanation is a word the reader has to guess at."""
    from rheplicant.core.capability import Maturity

    assert {level.value for level in Maturity} <= set(generator.MEANING)
    assert set(generator.ORDER) == {level.value for level in Maturity}
    for level, meaning in generator.MEANING.items():
        assert len(meaning) > 60, f"{level}'s meaning is a label, not an explanation"


def test_the_generated_page_names_every_class_the_registry_knows(generated):
    """Both directions: the walk and the page cannot disagree about members."""
    import re

    from rheplicant.radio import capabilities

    listed = set(re.findall(r"^- `(\w+)`$", generated, re.M))
    assert listed == set(capabilities())


def test_the_generated_page_names_every_registry_row(generated, generator):
    """The not-implemented half, which is a different table and a different
    source: a capability that is not a class has no ClassVar to walk."""
    from _rheplicant_bootstrap.capability import REGISTRY

    for row in REGISTRY:
        assert f"`{row.name}`" in generated, row.name
