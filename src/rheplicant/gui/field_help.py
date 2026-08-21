"""One line of help per operator field, read off the class that owns it.

The package documents every model field in its class's ``Attributes:`` block
already -- all 66 of them, measured by
``tests/gui/test_field_help.py`` -- so the inspector reads that rather than
growing a second copy. A second copy is the one that goes stale: nothing
renders it beside the first, so nobody notices when they disagree.

Napoleon parses the same blocks, but only at documentation build time and
only into reStructuredText. This wants the sentence, at run time, in a
process that has no Sphinx.
"""

from __future__ import annotations

import inspect
import re

#: A field line: four spaces, a name, a colon, the sentence. The indent is
#: what separates an attribute from the section header above it and from the
#: next section below.
_FIELD = re.compile(r"^ {4}(\w+)\s*:\s*(.*)$")


def field_help(owner: type) -> dict[str, str]:
    """Field name -> its one-line description, or an empty mapping.

    Only a line that is exactly ``Attributes:`` opens the block: several
    docstrings here mention the word mid-sentence, and reading one of those
    as a header would turn the rest of the prose into attributes.

    A more-indented line continues the sentence above it, which is how a
    description longer than one line is written. The block ends at the first
    non-blank line that is not indented -- the next section, or the end.
    """
    doc = inspect.getdoc(owner)
    if not doc:
        return {}
    lines = doc.splitlines()
    try:
        start = lines.index("Attributes:") + 1
    except ValueError:
        return {}
    found: dict[str, str] = {}
    current: str | None = None
    for line in lines[start:]:
        if not line.strip():
            continue
        if not line.startswith("    "):
            break
        match = _FIELD.match(line)
        if match is not None:
            current = match.group(1)
            found[current] = match.group(2).strip()
        elif current is not None:
            found[current] = f"{found[current]} {line.strip()}".strip()
    return found


__all__ = ["field_help"]
