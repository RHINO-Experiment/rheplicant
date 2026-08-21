"""How one widget is described: its rule tree, its label, and its view.

Split out of :mod:`rheplicant.gui.form_catalog`, which declares WHICH widgets
exist. These say what each one is called, which of the twelve left-rail views
owns it, and how a visibility condition is written down.

That they need no config vocabulary is the whole reason they could move.
``tests/config/test_config_surface.py`` keeps the list of modules allowed to
import the config layer at FIVE files, so every helper that wants a unit
table, a dimension spec or a node's shape rule stays behind in
``form_catalog`` -- which is why that module is still the largest here.
"""

from __future__ import annotations

from collections.abc import Mapping

from rheplicant.gui.forms import FormRule
from rheplicant.radio.filters.base import _MODES as FILTER_MODES
from rheplicant.radio.graph import RADIO_GRAPH
from rheplicant.radio.instrument.calibration import LINESHAPES


def _rule(path: str, operator: str, expected: object = None) -> FormRule:
    return FormRule(path=path, operator=operator, expected=expected)  # type: ignore[arg-type]


def _all(*rules: FormRule) -> FormRule:
    return FormRule(path=None, operator="all", rules=tuple(rules))


def _any(*rules: FormRule) -> FormRule:
    return FormRule(path=None, operator="any", rules=tuple(rules))


def _label(path: str) -> str:
    tail = path.rsplit(".", 1)[-1].replace("[]", "").replace("*", "entry")
    return tail.replace("_", " ")


def _section_for(path: str) -> str:
    if path.startswith("resources.sky_models") or path.startswith("resources.projectors"):
        return "sky"
    if path.startswith("resources.beams"):
        return "beam"
    if path.startswith("resources"):
        return "resources"
    if path.startswith("model"):
        # A list node's instance prefix spells the node ``filters[]``, which is
        # not a node id: without the trim every field of every list node falls
        # through to the Instrument default, and ``filters`` is `processing`.
        node = path.split(".")[1].removesuffix("[]") if "." in path else ""
        spec = RADIO_GRAPH.nodes.get(node)
        return "backend" if spec is not None and spec.segment == "processing" else "instrument"
    if (
        path.startswith("schema_version")
        or path.startswith("defaults")
        or path.startswith("plugins")
    ):
        return "runtime"
    root = path.split(".", 1)[0].removesuffix("[]")
    # The catch-all hatch-argument destination is a value-form concern rather
    # than a thirteenth view; Resources owns the generic value-node palette.
    return "resources" if root == "python" else root


def _default_value(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _default_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_default_value(item) for item in value)
    return repr(value)


#: Class-path selector -> the members a ``static_str`` field accepts.
#:
#: Keyed by the same selector the dimension registry uses, so the two are
#: comparable by construction: ``tests/gui/test_forms.py`` asserts this table
#: is EXACTLY the live ``model_field`` rows whose disposition is structural
#: and whose reason is ``"enum"``, which makes a new enum field impossible to
#: ship with an empty select.
#:
#: The members are the operators' own constants rather than copies. Both are
#: what the operator validates against -- ``filters/base.py`` and
#: ``instrument/calibration.py`` raise on anything else -- so a select built
#: from them cannot offer a member the constructor would refuse.
_ENUMS: dict[str, tuple[str, ...]] = {
    "rheplicant.radio.filters.fourier.FourierBandFilter.mode": FILTER_MODES,
    "rheplicant.radio.filters.sidereal.SiderealFilter.mode": FILTER_MODES,
    "rheplicant.radio.filters.skyspace.SkySpaceFilter.mode": FILTER_MODES,
    "rheplicant.radio.instrument.calibration.CWCalibrationOperator.lineshape": LINESHAPES,
}
