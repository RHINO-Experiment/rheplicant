"""Closed, framework-free form metadata projected from the live config layer.

The catalog describes widgets; the YAML document remains the only scientific
state.  A projection therefore contains current values and visibility flags,
never a second editable model.  Registry cardinalities are deliberately
guarded: adding a destination, resource kind, exit, output product, or graph
node must make the GUI census fail until its presentation contract is reviewed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal


class CatalogDrift(RuntimeError):
    """The live config surface no longer matches the reviewed GUI catalog."""


RuleOperator = Literal[
    "all",
    "any",
    "not",
    "present",
    "absent",
    "equals",
    "in",
    "contains",
    "absent_or_equals",
    "absent_or_in",
    "any_in",
    "length_gt",
]


@dataclass(frozen=True, slots=True)
class FormRule:
    """A serializable predicate over the YAML document."""

    path: str | None
    operator: RuleOperator
    expected: object = None
    rules: tuple[FormRule, ...] = ()


@dataclass(frozen=True, slots=True)
class SourceRef:
    """One live dimension destination that produced a widget."""

    domain: str
    selector: str


@dataclass(frozen=True, slots=True)
class SectionMetadata:
    """One planned left-rail view."""

    section_id: str
    label: str
    yaml_section: str
    disabled: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class WidgetMetadata:
    """One reviewed config path pattern and its widget contract."""

    path: str
    section: str
    label: str
    widget: str
    choices: tuple[str, ...] = ()
    required: bool = False
    has_default: bool = False
    default: object = None
    required_when: FormRule | None = None
    visible_when: FormRule | None = None
    dimension: str | None = None
    unit_policy: str | None = None
    #: Every spelling the alphabet accepts for :attr:`dimension`, canonical
    #: first, or empty when it has no second spelling to offer or refuses a
    #: unit outright. Derived in the builder; never written at a call site.
    units: tuple[str, ...] = ()
    delivery: str | None = None
    disabled: bool = False
    reason: str | None = None
    sources: tuple[SourceRef, ...] = ()


@dataclass(frozen=True, slots=True)
class FormCatalog:
    """The closed metadata catalog plus its live-registry censuses."""

    sections: tuple[SectionMetadata, ...]
    widgets: tuple[WidgetMetadata, ...]
    resource_kinds: tuple[str, ...]
    run_kinds: tuple[str, ...]
    graph_nodes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ProjectedWidget:
    """One concrete widget state read from the authoritative YAML."""

    path: str
    path_pattern: str
    label: str
    widget: str
    choices: tuple[str, ...]
    visible: bool
    present: bool
    must_decide: bool
    value: object
    dimension: str | None
    unit_policy: str | None
    units: tuple[str, ...]
    delivery: str | None
    disabled: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class ProjectedSection:
    """One planned view and the concrete widgets it owns."""

    section_id: str
    label: str
    disabled: bool
    reason: str | None
    widgets: tuple[ProjectedWidget, ...]


@dataclass(frozen=True, slots=True)
class FormProjection:
    """The complete form view of one parsed YAML mapping."""

    sections: tuple[ProjectedSection, ...]
    missing_required: tuple[str, ...]


_MISSING = object()


def _segments(path: str) -> tuple[str | int | None, ...]:
    """Parse dotted paths; ``*`` is a mapping wildcard and ``[]`` a list one."""
    result: list[str | int | None] = []
    for component in path.split("."):
        if component == "*":
            result.append("*")
        elif component.endswith("[]"):
            result.extend((component[:-2], None))
        elif "[" in component and component.endswith("]"):
            name, raw_index = component[:-1].split("[", 1)
            result.extend((name, int(raw_index)))
        else:
            result.append(component)
    return tuple(result)


def _render(segments: Sequence[str | int | None]) -> str:
    rendered = ""
    for segment in segments:
        if isinstance(segment, int):
            rendered += f"[{segment}]"
        else:
            if rendered:
                rendered += "."
            rendered += "[]" if segment is None else segment
    return rendered


def _expand(
    value: object,
    template: tuple[str | int | None, ...],
    *,
    index: int = 0,
    concrete: tuple[str | int | None, ...] = (),
    bindings: tuple[str | int, ...] = (),
) -> tuple[tuple[str, object, bool, tuple[str | int, ...]], ...]:
    if index == len(template):
        return (
            (
                _render(concrete),
                None if value is _MISSING else value,
                value is not _MISSING,
                bindings,
            ),
        )
    segment = template[index]
    if segment == "*":
        if not isinstance(value, Mapping):
            return ()
        rows = []
        for key, child in value.items():
            if isinstance(key, str):
                rows.extend(
                    _expand(
                        child,
                        template,
                        index=index + 1,
                        concrete=(*concrete, key),
                        bindings=(*bindings, key),
                    )
                )
        return tuple(rows)
    if segment is None:
        if not isinstance(value, list):
            return ()
        rows = []
        for list_index, child in enumerate(value):
            rows.extend(
                _expand(
                    child,
                    template,
                    index=index + 1,
                    concrete=(*concrete, list_index),
                    bindings=(*bindings, list_index),
                )
            )
        return tuple(rows)
    if isinstance(segment, int):
        child = value[segment] if isinstance(value, list) and segment < len(value) else _MISSING
    else:
        child = value.get(segment, _MISSING) if isinstance(value, Mapping) else _MISSING
    return _expand(
        child,
        template,
        index=index + 1,
        concrete=(*concrete, segment),
        bindings=bindings,
    )


def _bind(path: str, bindings: Sequence[str | int]) -> str:
    bound: list[str | int | None] = []
    remaining = iter(bindings)
    for segment in _segments(path):
        bound.append(next(remaining) if segment in ("*", None) else segment)
    return _render(bound)


def _values(document: Mapping[str, object], path: str) -> tuple[tuple[object, bool], ...]:
    return tuple((value, present) for _, value, present, _ in _expand(document, _segments(path)))


def _evaluate(
    rule: FormRule | None,
    document: Mapping[str, object],
    bindings: Sequence[str | int],
) -> bool:
    if rule is None:
        return True
    if rule.operator == "all":
        return all(_evaluate(child, document, bindings) for child in rule.rules)
    if rule.operator == "any":
        return any(_evaluate(child, document, bindings) for child in rule.rules)
    if rule.operator == "not":
        return not _evaluate(rule.rules[0], document, bindings)
    assert rule.path is not None
    bound = _bind(rule.path, bindings)
    found = _values(document, bound)
    actual, present = found[0] if found else (None, False)
    if rule.operator == "present":
        return present
    if rule.operator == "absent":
        return not present
    if rule.operator == "equals":
        return present and type(actual) is type(rule.expected) and actual == rule.expected
    if rule.operator == "in":
        return present and actual in rule.expected  # type: ignore[operator]
    if rule.operator == "contains":
        return (
            present
            and isinstance(actual, (list, tuple, set, frozenset))
            and rule.expected in actual
        )
    if rule.operator == "absent_or_equals":
        return not present or (type(actual) is type(rule.expected) and actual == rule.expected)
    if rule.operator == "absent_or_in":
        return not present or actual in rule.expected  # type: ignore[operator]
    if rule.operator == "any_in":
        pattern = rule.path
        matches = _values(document, pattern)
        return any(present and value in rule.expected for value, present in matches)  # type: ignore[operator]
    if rule.operator == "length_gt":
        return (
            present and isinstance(actual, (Mapping, list, tuple)) and len(actual) > rule.expected
        )  # type: ignore[arg-type,operator]
    raise AssertionError(f"unknown form rule {rule.operator!r}")


def _plain_value(value: object) -> object:
    """Detach parser-owned immutable containers at the GUI API boundary."""
    if isinstance(value, Mapping):
        return {key: _plain_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_value(item) for item in value]
    return value


def widget_catalog() -> FormCatalog:
    """Build and validate a fresh catalog from the registries as they are now."""
    from rheplicant.gui.form_catalog import build_catalog

    found = build_catalog()
    assert_catalog_closed(found)
    return found


def assert_catalog_closed(catalog: FormCatalog | None = None) -> None:
    """Raise when any reviewed live registry has drifted."""
    from rheplicant.gui.form_catalog import build_catalog, catalog_drift

    found = build_catalog() if catalog is None else catalog
    problems = catalog_drift(found)
    if problems:
        raise CatalogDrift("GUI widget catalog drift: " + "; ".join(problems))


def project_forms(document: Mapping[str, object]) -> FormProjection:
    """Project one parsed document through the catalog without mutating it."""
    if not isinstance(document, Mapping):
        raise TypeError("form projection requires a document mapping")
    catalog = widget_catalog()
    by_section: dict[str, list[ProjectedWidget]] = {
        section.section_id: [] for section in catalog.sections
    }
    for metadata in catalog.widgets:
        expanded = _expand(document, _segments(metadata.path))
        for path, value, present, bindings in expanded:
            visible = _evaluate(metadata.visible_when, document, bindings)
            required = metadata.required
            if metadata.required_when is not None:
                required = required or _evaluate(metadata.required_when, document, bindings)
            disabled = metadata.disabled or next(
                section.disabled
                for section in catalog.sections
                if section.section_id == metadata.section
            )
            must_decide = bool(
                visible and not disabled and required and not present and not metadata.has_default
            )
            by_section[metadata.section].append(
                ProjectedWidget(
                    path=path,
                    path_pattern=metadata.path,
                    label=metadata.label,
                    widget=metadata.widget,
                    choices=metadata.choices,
                    visible=visible,
                    present=present,
                    must_decide=must_decide,
                    value=_plain_value(value) if present else None,
                    dimension=metadata.dimension,
                    unit_policy=metadata.unit_policy,
                    units=metadata.units,
                    delivery=metadata.delivery,
                    disabled=disabled,
                    reason=metadata.reason,
                )
            )
    sections = tuple(
        ProjectedSection(
            section.section_id,
            section.label,
            section.disabled,
            section.reason,
            tuple(by_section[section.section_id]),
        )
        for section in catalog.sections
    )
    missing = tuple(
        widget.path for section in sections for widget in section.widgets if widget.must_decide
    )
    return FormProjection(sections=sections, missing_required=missing)


__all__ = [
    "CatalogDrift",
    "FormCatalog",
    "FormProjection",
    "FormRule",
    "ProjectedSection",
    "ProjectedWidget",
    "SectionMetadata",
    "SourceRef",
    "WidgetMetadata",
    "assert_catalog_closed",
    "project_forms",
    "widget_catalog",
]
