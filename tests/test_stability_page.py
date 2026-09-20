"""``docs/stability.md`` is checked against the code, never the reverse.

The page states numbers: 338 public names, 35 capabilities, 19 of them
placeholders, three contract versions, one bayesmith range. Every one is a
claim a reader will act on, and prose has no guard -- which is the failure
this repository keeps paying for and keeps writing down. So each number is
read back out of the page and compared with the thing it describes.

The direction is the one the plan asks for throughout: the registry, the
snapshots and `pyproject.toml` are the truth, and the page is a VIEW. A page
that disagreed would be corrected; the code would not.

**What is deliberately not checked** is the prose. Whether the explanation of
why a placeholder is not a bug is a good explanation is not decidable here,
and a test that tried would be pinning wording rather than fact.
"""

from __future__ import annotations

import importlib
import pathlib
import re
from collections import Counter

import pytest

from _rheplicant_bootstrap.audit.integrity import INTEGRITY_FORMAT_VERSION
from rheplicant.config.schema import json_schema
from rheplicant.core.capability import Maturity
from rheplicant.inference.archive import _FORMAT_VERSION
from rheplicant.radio import capabilities

ROOT = pathlib.Path(__file__).resolve().parents[1]
PAGE = ROOT / "docs" / "stability.md"

NAMESPACES = (
    "rheplicant", "rheplicant.core", "rheplicant.radio",
    "rheplicant.inference", "rheplicant.config", "rheplicant.gui",
)


@pytest.fixture(scope="module")
def page() -> str:
    assert PAGE.exists(), f"{PAGE} is missing; the stability contract is the page"
    return PAGE.read_text(encoding="utf-8")


def _row(page: str, label: str) -> int:
    """The integer in the table row whose first cell is ``label``."""
    match = re.search(rf"^\|\s*`?{re.escape(label)}`?\s*\|\s*(\d+)\s*\|", page, re.M)
    assert match, f"no table row for {label!r} in {PAGE.name}"
    return int(match.group(1))


@pytest.mark.parametrize("namespace", NAMESPACES, ids=NAMESPACES)
def test_the_page_states_each_namespaces_real_size(page, namespace):
    live = len(importlib.import_module(namespace).__all__)
    assert _row(page, namespace) == live, (
        f"{PAGE.name} says {namespace} has {_row(page, namespace)} public "
        f"names; it has {live}"
    )


def test_the_page_states_the_real_total(page):
    live = sum(len(importlib.import_module(n).__all__) for n in NAMESPACES)
    assert f"**{live} names**" in page, (
        f"{PAGE.name} does not state the real total of {live} public names"
    )


def test_the_page_states_the_real_capability_total(page):
    live = len(capabilities())
    assert f"**{live}** shipped capabilities" in page, (
        f"{PAGE.name} does not state the real capability total of {live}"
    )


@pytest.mark.parametrize(
    "level", [Maturity.MAINTAINED, Maturity.EXPERIMENTAL, Maturity.PLACEHOLDER],
    ids=lambda level: level.name,
)
def test_the_page_states_each_levels_real_count(page, level):
    counts = Counter(capabilities().values())
    label = level.name.capitalize()
    assert _row(page, label) == counts[level], (
        f"{PAGE.name} says {counts[level]} capabilities are {label} in the "
        f"code and {_row(page, label)} on the page"
    )


def test_the_page_never_claims_a_class_is_unavailable(page):
    """The reading the registry is built on, restated where a reader sees it.

    ``Unavailable`` is a surface's answer. If the page ever counted classes at
    that level it would be describing a different model from the one the code
    implements.
    """
    assert not re.search(r"^\|\s*Unavailable\s*\|\s*\d+\s*\|", page, re.M), (
        f"{PAGE.name} counts classes as Unavailable. No class carries that "
        "level: it describes what a SURFACE answers, and the document keys "
        "live in _rheplicant_bootstrap.capability.REGISTRY"
    )


def test_the_page_states_the_real_contract_versions(page):
    for version, what in (
        (f'`"{json_schema()["schemaVersion"]}"`', "the schema version"),
        (f"| `{INTEGRITY_FORMAT_VERSION}` |", "the integrity version"),
        (f"| `{_FORMAT_VERSION}` |", "the archive version"),
    ):
        assert version in page, f"{PAGE.name} does not state {what} as {version}"


def test_the_page_states_the_declared_bayesmith_range(page):
    declared = re.search(r'"(bayesmith>=[^"]+)"', (ROOT / "pyproject.toml").read_text())
    assert declared, "pyproject.toml no longer declares a bayesmith range"
    assert f"`{declared.group(1)}`" in page, (
        f"{PAGE.name} does not state the declared range {declared.group(1)!r}"
    )


def test_the_page_names_the_guards_that_hold_its_claims(page):
    """A page that cites no guard invites the reader to trust the page.

    Each of these files is what makes a section of the page more than an
    assertion, and naming them is how a sceptical reader checks rather than
    believes.
    """
    for guard in (
        "tests/test_import_direction.py",
        "tests/test_public_surface.py",
        "tests/test_cross_package_privates.py",
        "tests/test_bayesmith_floor.py",
    ):
        assert guard in page, f"{PAGE.name} does not name {guard}"
        assert (ROOT / guard).exists(), f"{PAGE.name} names {guard}, which is gone"
