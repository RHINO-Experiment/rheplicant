"""JAX-free lexical grammar for configuration document paths."""

from __future__ import annotations

import re

from _rheplicant_bootstrap.errors import ConfigError

PATH_STEP = re.compile(
    r"^(?P<name>[A-Za-z_][A-Za-z_0-9]*)?"
    r"(?:\[(?P<index>0|[1-9][0-9]*)\])?$"
)


def _canonical_label(label: object) -> str:
    if not isinstance(label, str):
        raise ConfigError(
            "document path label must be a string; got "
            f"{type(label).__name__}."
        )
    return str.__str__(label)


def is_legal_path(label: object) -> bool:
    """Return whether *label* obeys the shared identifier/index grammar."""
    if not isinstance(label, str):
        return False
    text = str.__str__(label)
    if not text or text != str.strip(text):
        return False
    for piece in str.split(text, "."):
        match = PATH_STEP.fullmatch(piece)
        if match is None:
            return False
        if match.group("name") is None and match.group("index") is None:
            return False
    return True


def longest_legal_prefix(label: str) -> str:
    """Cut *label* back to its longest spellable document-path prefix."""
    text = _canonical_label(label)
    legal: list[str] = []
    for piece in str.split(text, "."):
        match = PATH_STEP.fullmatch(piece)
        if match is None or (
            match.group("name") is None and match.group("index") is None
        ):
            break
        legal.append(piece)
    if legal:
        return ".".join(legal)
    # Every caller supplies a section-headed label.  Keep this branch closed
    # for direct construction without inventing a syntactically valid path.
    raise ConfigError(f"document path label {text!r} has no legal prefix.")


__all__ = ["PATH_STEP", "is_legal_path", "longest_legal_prefix"]
