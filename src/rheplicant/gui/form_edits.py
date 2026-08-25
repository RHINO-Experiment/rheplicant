"""Closed YAML transitions for one explicitly projected form field."""

from __future__ import annotations

from collections.abc import Mapping

import yaml  # for `safe_dump` only; the LOAD goes through the bounded loader below

from _rheplicant_bootstrap.yaml import safe_load_document
from rheplicant.config import ConfigError
from rheplicant.gui.forms import ProjectedWidget, project_forms
from rheplicant.gui.yaml_values import plain

_EDITABLE_WIDGETS = frozenset({"select", "toggle", "integer", "text", "file"})
_YAML_GUIDANCE = "Edit this value in YAML."


def _load_mapping(yaml_text: str) -> dict[str, object]:
    """The package's bounded loader, then thawed so this module can edit.

    ``yaml.safe_load`` stood here, and the difference is not cosmetic even
    though nothing reachable today exploits it: the bounded loader caps the
    input size and the alias/anchor expansion, which is what keeps a
    billion-laughs document from being expanded before anyone looks at it,
    and it reports every refusal in the package's own words instead of a bare
    ``YAMLError``.

    **Measured, so the change is not sold as a fix for a hole that was
    open**: every caller reaches here through ``session.yaml_text``, and a
    session is only ever built by ``new_session``/``replace_session_yaml`` ->
    ``replace_yaml`` -> ``snapshot`` -> ``document._load``, which is
    ``safe_load_document`` already. So the bound was in place -- by an
    invariant on the CALLERS that nothing stated and no test held. A second
    entry point that had not already loaded the text would have removed it
    silently. Now the bound is here, where the parse is.

    The loader returns a frozen value and this module mutates the result, so
    it is thawed through the same :func:`~rheplicant.gui.yaml_values.plain`
    the rest of the GUI uses.

    **Its message is deliberately thrown away, and that is not laziness.**
    The loader's refusals name the offending construct, which is right for a
    CLI reading a file the user can see and wrong here: this message travels
    back to a browser, so quoting the document reflects whatever was typed
    into the editor. ``test_malformed_or_nonmapping_yaml_is_refused_without
    _echoing_input`` holds that line -- it feeds a document containing a
    marker string and asserts the marker does not come back, and it caught
    exactly this when the swap was first written without the ``except``. So
    the bound is what is gained here; the detail is what is knowingly given
    up for it.
    """
    if not isinstance(yaml_text, str):
        raise ConfigError("Field edit requires a safe YAML mapping document.")
    try:
        payload = yaml_text.encode("utf-8", "strict")
    except UnicodeEncodeError:
        raise ConfigError("Field edit requires a safe YAML mapping document.") from None
    try:
        loaded = safe_load_document(payload, source_name="GUI field edit").value
    except ConfigError:
        raise ConfigError("Field edit requires a safe YAML mapping document.") from None
    document = plain(loaded)
    if not isinstance(document, dict) or not all(
        isinstance(key, str) for key in document
    ):
        raise ConfigError("Field edit requires a safe YAML mapping document.")
    return document


def _matches_pattern(path: str, pattern: str) -> bool:
    components = path.split(".")
    template = pattern.split(".")
    return len(components) == len(template) and all(
        expected == "*" or expected == actual
        for actual, expected in zip(components, template, strict=True)
    )


def _projected_widget(document: Mapping[str, object], path: str) -> ProjectedWidget:
    if not path or "*" in path or "[" in path or "]" in path:
        raise ConfigError(
            f"Field edit path must be one exact concrete mapping path. {_YAML_GUIDANCE}"
        )
    matches = tuple(
        widget
        for section in project_forms(document).sections
        for widget in section.widgets
        if widget.path == path
    )
    if len(matches) != 1:
        raise ConfigError(
            f"Field edit path is not one exact projected field. {_YAML_GUIDANCE}"
        )
    widget = matches[0]
    if not _matches_pattern(path, widget.path_pattern):
        raise ConfigError(
            f"Field edit path is not one unambiguous concrete field. {_YAML_GUIDANCE}"
        )
    if not widget.visible or widget.disabled:
        raise ConfigError(f"This projected field is unavailable. {_YAML_GUIDANCE}")
    if widget.widget not in _EDITABLE_WIDGETS:
        raise ConfigError(f"This field is YAML-only. {_YAML_GUIDANCE}")
    return widget


def _validate_value(widget: ProjectedWidget, value: object) -> None:
    if widget.widget == "select":
        if type(value) is not str or value not in widget.choices:
            raise ConfigError(f"Field {widget.path!r} requires one declared choice.")
        return
    if widget.widget == "toggle":
        if type(value) is not bool:
            raise ConfigError(f"Field {widget.path!r} requires a boolean value.")
        return
    if widget.widget == "integer":
        if type(value) is not int:
            raise ConfigError(f"Field {widget.path!r} requires an integer value.")
        return
    if widget.widget in {"text", "file"} and type(value) is not str:
        noun = "server path string" if widget.widget == "file" else "text value"
        raise ConfigError(f"Field {widget.path!r} requires a {noun}.")


def _parent_mapping(document: dict[str, object], components: tuple[str, ...]) -> dict[str, object]:
    parent = document
    for component in components:
        child = parent.get(component)
        if child is None and component not in parent:
            child = {}
            parent[component] = child
        if not isinstance(child, dict):
            raise ConfigError(
                f"Field edit cannot cross a non-mapping parent. {_YAML_GUIDANCE}"
            )
        parent = child
    return parent


def set_form_value(
    yaml_text: str,
    path: str,
    value: object,
    *,
    remove: bool = False,
) -> str:
    """Return YAML with one closed projected field set or removed."""
    document = _load_mapping(yaml_text)
    widget = _projected_widget(document, path)

    if remove:
        if not widget.present:
            return yaml_text
    else:
        _validate_value(widget, value)
        if widget.present and type(widget.value) is type(value) and widget.value == value:
            return yaml_text

    components = tuple(path.split("."))
    parent = _parent_mapping(document, components[:-1])
    if remove:
        del parent[components[-1]]
    else:
        parent[components[-1]] = value
    return yaml.safe_dump(document, allow_unicode=True, sort_keys=False)


__all__ = ["set_form_value"]
