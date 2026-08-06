"""The guided tour claims to be a script; this runs it and proves the claim.

``docs/tour.md`` opens with "pasted top to bottom they form a working script".
Nothing was enforcing that. ``myst-parser`` renders code blocks, it does not run
them (unlike ``myst-nb``), so the promise was kept by hand -- and had already
broken: section 3 called ``DriftScanProjector.from_beam_maps(beam_maps, ...)``
with ``beam_maps`` never defined anywhere in the document. A reader pasting the
tour hit a ``NameError`` two thirds of the way down, and the failure was
invisible to the whole test suite.

**Why a subprocess rather than ``exec``.** The claim under test is not "these
statements evaluate" but "a reader who pastes this into a file and runs it gets
a working script", so the test does exactly that: concatenate the fences, write
a file, run it with ``sys.executable``. It also keeps the tour's global effects
-- JAX's x64 flag, the graph registry, imports -- out of the process running the
rest of the suite, which an in-process ``exec`` would not.

**Optional extras.** Two of the sky engines need ``limtod``, an extra rather
than a dependency, so their block cannot run in a default checkout. A block may
opt out with a first-line marker::

    # needs-extra: limtod

and is then omitted -- but only when the import genuinely fails. With the extra
installed the block runs like any other, so the marker cannot rot into a
blanket exemption: a broken snippet behind it still fails for anyone who has
the extra.
"""

import re
import subprocess
import sys
from importlib.util import find_spec
from pathlib import Path

import pytest

TOUR = Path(__file__).resolve().parents[1] / "docs" / "tour.md"

#: ```python fences, in document order.
_FENCE = re.compile(r"^```python\n(.*?)^```", re.MULTILINE | re.DOTALL)

#: ``# needs-extra: <name>`` in a block. The name is the module the code
#: actually IMPORTS, which is not always the pip extra: ``rheplicant[limtod]``
#: installs a distribution whose import is ``limtod_jax``, and a marker naming
#: the extra instead skipped the block on a machine where the import was in
#: fact available -- which is how the ``beam_maps`` bug stayed invisible to a
#: first draft of this very file.
_NEEDS_EXTRA = re.compile(r"^\s*#\s*needs-extra:\s*(\S+)", re.MULTILINE)

#: The imports a marker is allowed to name. A typo would otherwise skip its
#: block forever and silently, since a misspelled module and an uninstalled one
#: are the same thing to ``find_spec``.
_KNOWN_EXTRAS = frozenset({"limtod_jax", "numpyro"})

#: Below this, the fence pattern has stopped matching rather than the tour
#: having shrunk -- a vacuous pass is the failure mode worth naming.
_MIN_BLOCKS = 15


def _blocks() -> list[str]:
    return _FENCE.findall(TOUR.read_text())


def test_the_tour_is_a_script_that_runs(tmp_path: Path) -> None:
    """Paste every python block into one file, in order, and run it."""
    blocks = _blocks()
    assert len(blocks) >= _MIN_BLOCKS, (
        f"Only {len(blocks)} python blocks found in {TOUR.name}; the fence pattern "
        "has probably stopped matching rather than the tour having shrunk."
    )

    parts: list[str] = []
    skipped: list[int] = []
    for number, source in enumerate(blocks, start=1):
        extra = _NEEDS_EXTRA.search(source)
        if extra and find_spec(extra.group(1)) is None:
            skipped.append(number)
            continue
        parts.append(f"# ---- {TOUR.name} block {number} ----\n{source}")

    assert len(skipped) < len(blocks) // 2, (
        f"{len(skipped)} of {len(blocks)} blocks were omitted for missing extras; "
        "this run would prove almost nothing about the tour."
    )

    script = tmp_path / "tour_as_pasted.py"
    script.write_text("\n".join(parts))

    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise AssertionError(
            f"{TOUR.name} is not a runnable script -- pasting it top to bottom fails.\n"
            f"Script written to {script} (the '# ---- block N ----' comments locate "
            f"the failure in the document).\n\n{result.stderr}"
        )


def test_every_needs_extra_marker_names_a_real_optional_import() -> None:
    """A marker with a typo would skip its block forever, and silently."""
    for number, source in enumerate(_blocks(), start=1):
        for name in _NEEDS_EXTRA.findall(source):
            assert name in _KNOWN_EXTRAS, (
                f"{TOUR.name} block {number} declares 'needs-extra: {name}', which is "
                f"not one of {sorted(_KNOWN_EXTRAS)}. If it is a new extra, add it to "
                "_KNOWN_EXTRAS; otherwise it is a typo that would skip the block "
                "forever."
            )


@pytest.mark.parametrize(
    ("anchor", "heading"),
    [
        ("#the-contract", "The contract"),
        ("#composition-cascade-sum-switch", "Composition: cascade, sum, switch"),
        ("#graph-assembly", "Graph assembly"),
        ("#rendering", "Rendering"),
    ],
)
def test_contents_anchors_still_have_their_headings(anchor: str, heading: str) -> None:
    """The tour's contents links to sub-headings that must keep their wording.

    ``myst_heading_anchors = 3`` derives these anchors from the heading text, so
    rewording a ``###`` breaks its own contents link -- and breaks the three
    cross-references from ``index.md``, ``operators.md`` and ``signal-path.md``
    that point at ``#graph-assembly``. Neither raises a build warning.
    """
    text = TOUR.read_text()
    assert f"]({anchor})" in text, f"{anchor} is no longer linked from the contents"
    assert f"### {heading}\n" in text, (
        f"The contents links to {anchor}, which is generated from the heading "
        f"'### {heading}' -- and that heading is gone or reworded."
    )
