"""The shell recipes in the working notes parse as shell.

There are three ways to check a documented command, and they test different
things:

1. **Retype an equivalent.** Tests the retyper's understanding at the moment
   of retyping -- and understanding does not update when the thing it
   describes does. Check B9 shipped a remedy that had stopped existing
   because its note described an equivalent rather than the thing itself.
2. **Copy the block and run it.** Tests what the file says right now, and
   still depends on a person picking the right block.
3. **Extract the block from the file and run it.** Tests the file. Nothing
   between the bytes and the shell.

Only the third is mechanical, so only the third can be a test. This is that,
at the cheapest useful strength: every ``bash`` block in the working notes is
handed to ``bash -n``, which parses without executing. It cannot say a
command does what the prose claims; it can say that an edit to the prose has
not left a recipe a reader cannot run -- an unbalanced quote, a mangled
continuation, a paste that lost a line. Those are the failures a documented
command actually has, and they are invisible to every other test here
because nothing else ever looks at these blocks.

Executing them is deliberately NOT done. Two of them are full test suites.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
_NOTES = ("CLAUDE.md", "AGENTS.md", "README.md")


def _blocks(name: str) -> list[tuple[str, int, str]]:
    """``(file, ordinal, source)`` for every fenced bash block."""
    text = (_ROOT / name).read_text(encoding="utf-8")
    return [
        (name, index, match.group(1))
        for index, match in enumerate(re.finditer(r"```bash\n(.*?)```", text, re.S))
    ]


ALL_BLOCKS = [block for name in _NOTES for block in _blocks(name)]


def test_the_notes_carry_recipes_at_all():
    """ANTI-VACUITY. A regex that matched nothing would make every test
    below pass while checking no recipe -- which is the shape this file
    exists to argue against."""
    assert len(ALL_BLOCKS) >= 8, [(name, index) for name, index, _ in ALL_BLOCKS]
    for name in _NOTES:
        assert _blocks(name), f"{name} has no bash block; the fence style changed"


@pytest.mark.parametrize(
    ("name", "index", "source"),
    ALL_BLOCKS,
    ids=[f"{name}:{index}" for name, index, _ in ALL_BLOCKS],
)
def test_every_documented_shell_recipe_parses(name, index, source):
    bash = shutil.which("bash")
    if bash is None:  # pragma: no cover - bash is present on every dev box
        pytest.skip("no bash on PATH, so a shell recipe cannot be parsed")
    done = subprocess.run(
        [bash, "-n"], input=source, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, (
        f"{name}, bash block {index}, does not parse:\n{done.stderr}\n"
        f"--- the block as the file holds it ---\n{source}"
    )


def test_claude_md_and_agents_md_hold_the_same_recipes():
    """The two files are one document (``test_docs_claims.py`` holds them
    byte-identical), so this adds nothing on its own -- except a failure
    message about RECIPES when the drift is in one, which is the level a
    reader of this file is working at."""
    assert [source for _, _, source in _blocks("CLAUDE.md")] == [
        source for _, _, source in _blocks("AGENTS.md")
    ]
