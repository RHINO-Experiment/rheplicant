"""The TypeScript view of a wire shape must carry every field Python sends.

``gui/react/types.ts`` is hand-written and 29 modules import it, so a field the
projection populates but the interface omits is invisible twice over: the
browser receives it and the compiler never mentions that nothing can read it.
``units`` went missing exactly that way -- declared on ``ProjectedWidget``
(``forms.py:118``), populated on every widget (``forms.py:358``), and absent
from the interface, so a typed consumer was strictly less informed than the
JSON it had just parsed.

The check is field NAMES only. TypeScript spells the value types differently
on purpose (``value: unknown`` for ``object``, ``string | null`` for
``str | None``), and pinning those here would be a second grammar to keep in
step -- which is the problem, not the fix.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from rheplicant.gui import forms

_TYPES_TS = Path(forms.__file__).parent / "react" / "types.ts"

#: Every shape the projection sends that the editor declares a view of. A new
#: dataclass on the wire belongs here the day it gets a TypeScript twin.
_MIRRORED = ["ProjectedWidget", "ProjectedSection", "FormProjection"]


def _typescript_fields(interface: str) -> list[str]:
    source = _TYPES_TS.read_text(encoding="utf-8")
    found = re.search(rf"export interface {interface} \{{(.*?)\n\}}", source, re.S)
    if found is None:
        pytest.fail(f"{_TYPES_TS.name} declares no interface {interface}")
    # `name?:` and `name:` alike; comment lines carry no colon in this position.
    return re.findall(r"^\s*(\w+)\??:", found.group(1), re.M)


@pytest.mark.parametrize("interface", _MIRRORED)
def test_typescript_interface_mirrors_the_dataclass_field_for_field(interface):
    python = [field.name for field in dataclasses.fields(getattr(forms, interface))]
    typescript = _typescript_fields(interface)
    assert sorted(typescript) == sorted(python), (
        f"{interface}: TypeScript is missing {sorted(set(python) - set(typescript))} "
        f"and declares {sorted(set(typescript) - set(python))} that Python does not send"
    )
