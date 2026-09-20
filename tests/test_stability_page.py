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
import json
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
SCHEMAS = ROOT / "src" / "rheplicant" / "config" / "schemas"

NAMESPACES = (
    "rheplicant",
    "rheplicant.core",
    "rheplicant.radio",
    "rheplicant.inference",
    "rheplicant.config",
    "rheplicant.gui",
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
        f"{PAGE.name} says {namespace} has {_row(page, namespace)} public names; it has {live}"
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
    "level",
    [Maturity.MAINTAINED, Maturity.EXPERIMENTAL, Maturity.PLACEHOLDER],
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


def packaged_schema_versions():
    """Every schema shipped in the wheel that declares a ``format_version``.

    Read from the directory rather than listed, because a list is what this
    guard was before and what it missed: ``products-v1`` shipped a versioned,
    published manifest and the page's table did not mention it at all. Two
    further rows were added by hand on 2026-09-20 and the same omission could
    have happened again the same way.
    """
    rows = {}
    for path in sorted(SCHEMAS.glob("*.schema.json")):
        schema = json.loads(path.read_bytes())
        const = schema.get("properties", {}).get("format_version", {}).get("const")
        if const is not None:
            rows[path.name.removesuffix(".schema.json")] = const
    assert rows, "no packaged schema declares a format_version"
    return rows


def test_the_page_states_the_real_contract_versions(page):
    for version, what in (
        (f'`"{json_schema()["schemaVersion"]}"`', "the schema version"),
        (f"| `{INTEGRITY_FORMAT_VERSION}` |", "the integrity version"),
        (f"| `{_FORMAT_VERSION}` |", "the archive version"),
    ):
        assert version in page, f"{PAGE.name} does not state {what} as {version}"


@pytest.mark.parametrize("name", sorted(packaged_schema_versions()))
def test_the_page_names_every_packaged_schema_and_its_version(page, name):
    """A published format the page does not list is one nobody was told about.

    The row has to name the schema FILE, not just carry the number: every
    version here is currently ``1``, so a table that stated the numbers alone
    would be satisfied by any three rows at all.
    """
    version = packaged_schema_versions()[name]
    assert f"`{name}.schema.json`" in page, (
        f"{PAGE.name} does not name the packaged schema {name}.schema.json"
    )
    row = next((line for line in page.splitlines() if f"`{name}.schema.json`" in line), None)
    assert row is not None
    assert f"`{version}`" in row, (
        f"{PAGE.name}'s row for {name} does not state its version {version}"
    )


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
        "tests/config/test_audit_schemas.py",
    ):
        assert guard in page, f"{PAGE.name} does not name {guard}"
        assert (ROOT / guard).exists(), f"{PAGE.name} names {guard}, which is gone"


def test_the_page_states_the_real_development_status(page):
    """The classifier is a claim about maturity, in two places.

    ``docs/stability.md`` says the package is pre-1.0 and quotes the
    classifier; ``pyproject.toml`` is what PyPI reads. Nothing joined them, so
    raising one would have left the other saying the older thing -- and this
    is exactly the release where somebody is tempted to raise it.

    Alpha is deliberate at 0.9.0 and the page argues it: the boundaries and
    the API are frozen, and 17 of the 29 shipped operator classes are still
    placeholder physics. A classifier describes the whole package, and the
    physics is the part a reader is most likely to trust by mistake.
    """
    declared = re.search(r'"(Development Status :: [^"]+)"', (ROOT / "pyproject.toml").read_text())
    assert declared, "pyproject.toml declares no development-status classifier"
    assert f"`{declared.group(1)}`" in page, (
        f"{PAGE.name} does not quote the declared classifier {declared.group(1)!r}"
    )


#: The page writes its two bootstrap-layer counts as words. Generated rather
#: than listed: a hand-kept table needs an entry added every time a module is
#: split, and the entry that is missing reads as "the page states no count"
#: rather than as "this table is short".
_ONES = (
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen"
).split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_NUMBER_WORDS = {word: n for n, word in enumerate(_ONES)} | {
    (_TENS[t] if o == 0 else f"{_TENS[t]}-{_ONES[o]}"): t * 10 + o
    for t in range(2, 10)
    for o in range(10)
}


def test_the_page_states_the_real_bootstrap_layer_split(page):
    """The one number on this page that was NOT checked, and had gone stale.

    The page opens by saying every number on it is checked against the code.
    That was false here: it said thirty-seven foundation modules while
    ``tests/test_import_direction.py`` pinned forty-four, because splitting
    ``layering.py`` into its seven subjects added seven foundation modules and
    the prose was not in the loop. The guard that knew lived in another file
    and had no reason to read this page.

    Both halves are derived here, from the same walk the import-direction
    ratchet uses, so the page cannot drift from it again.
    """
    from tests.test_import_direction import BOOTSTRAP_COMMAND, _bootstrap_modules

    # The ratchet's OWN walk, not a second one. A stem-based reimplementation
    # written here first gave 46 against the ratchet's 44, because the package
    # has subpackages and `audit/__init__.py` and `output/__init__.py` are not
    # two modules called `__init__`. Two walks of one package is the shape
    # this whole page exists to stop.
    foundation = set(_bootstrap_modules()) - BOOTSTRAP_COMMAND

    stated = re.search(r"([A-Za-z-]+) modules are the \*\*command\s+half\*\*", page, re.S)
    assert stated, "the page no longer states a command-half count"
    assert _NUMBER_WORDS.get(stated.group(1).lower()) == len(BOOTSTRAP_COMMAND), (
        f"the page says {stated.group(1)!r} command modules; the ratchet "
        f"pins {len(BOOTSTRAP_COMMAND)}"
    )

    other = re.search(r"the other ([a-z-]+) are the \*\*foundation\*\*", page)
    assert other, "the page no longer states a foundation count"
    assert _NUMBER_WORDS.get(other.group(1)) == len(foundation), (
        f"the page says {other.group(1)!r} foundation modules; the walk finds "
        f"{len(foundation)}. Splitting a foundation module changes this number"
    )
