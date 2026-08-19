"""Immutable editor-session transitions over authoritative YAML text.

The history contains YAML documents, never assembled models or frontend form
state.  Every transition returns a new :class:`EditorSession`; the web adapter
may store that record, but it does not implement a second set of transition
rules.  File access is deliberately limited to the two explicit boundary
functions at the foot of this module.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from _rheplicant_bootstrap.errors import ConfigError
from rheplicant.gui.document import (
    compose_node,
    move_node_instance,
    place_node,
    replace_yaml,
    set_many_node,
    set_node,
    set_snapshot_before,
)
from rheplicant.gui.outputs import set_output_product, set_output_report


class RevisionConflict(ConfigError):
    """An editor command was based on an out-of-date session revision."""

    def __init__(self, expected: object, actual: int) -> None:
        super().__init__(
            f"Editor command expected revision {expected!r}, "
            f"but the current revision is {actual}. Refresh the session and retry."
        )
        self.expected = expected
        self.actual = actual


@dataclass(frozen=True, slots=True)
class EditorSession:
    """One immutable YAML history and its non-scientific lifecycle metadata."""

    history: tuple[str, ...]
    cursor: int
    revision: int
    saved_digest: str
    validated_digest: str | None

    @property
    def yaml_text(self) -> str:
        """The sole current scientific state."""
        return self.history[self.cursor]

    @property
    def dirty(self) -> bool:
        """Whether the exact current YAML bytes differ from the save baseline."""
        return _digest(self.yaml_text) != self.saved_digest

    @property
    def validation_stale(self) -> bool:
        """Whether validation has not been run for these exact YAML bytes."""
        return _digest(self.yaml_text) != self.validated_digest

    @property
    def can_undo(self) -> bool:
        return self.cursor > 0

    @property
    def can_redo(self) -> bool:
        return self.cursor + 1 < len(self.history)


def _digest(yaml_text: str) -> str:
    return sha256(yaml_text.encode("utf-8", "strict")).hexdigest()


def _expect(session: EditorSession, expected_revision: int) -> None:
    if type(expected_revision) is not int or expected_revision != session.revision:
        raise RevisionConflict(expected_revision, session.revision)


def _commit(session: EditorSession, yaml_text: str) -> EditorSession:
    if yaml_text == session.yaml_text:
        return session
    return replace(
        session,
        history=(*session.history[: session.cursor + 1], yaml_text),
        cursor=session.cursor + 1,
        revision=session.revision + 1,
        validated_digest=_digest(yaml_text),
    )


def new_session(yaml_text: str) -> EditorSession:
    """Open validated YAML as a clean session with no validation result."""
    found = replace_yaml(yaml_text)
    return EditorSession(
        history=(found.yaml_text,),
        cursor=0,
        revision=0,
        saved_digest=_digest(found.yaml_text),
        validated_digest=_digest(found.yaml_text),
    )


def replace_session_yaml(
    session: EditorSession,
    yaml_text: str,
    *,
    expected_revision: int,
) -> EditorSession:
    """Commit one validated YAML-mirror edit to ``session``."""
    _expect(session, expected_revision)
    return _commit(session, replace_yaml(yaml_text).yaml_text)


def edit_session_node(
    session: EditorSession,
    node_id: str,
    *,
    enabled: bool,
    expected_revision: int,
    settings: object | None = None,
    variant: str | None = None,
) -> EditorSession:
    """Commit one graph-node transformation against the current YAML."""
    _expect(session, expected_revision)
    found = set_node(
        session.yaml_text,
        node_id,
        enabled=enabled,
        settings=settings,
        variant=variant,
    )
    return _commit(session, found.yaml_text)


def edit_session_many_node(
    session: EditorSession,
    node_id: str,
    entries: object,
    *,
    expected_revision: int,
    variant: str | None = None,
) -> EditorSession:
    """Commit one whole SUM/CHAIN/FAN configuration."""
    _expect(session, expected_revision)
    found = set_many_node(session.yaml_text, node_id, entries, variant=variant)
    return _commit(session, found.yaml_text)


def move_session_node_instance(
    session: EditorSession,
    node_id: str,
    from_index: int,
    to_index: int,
    *,
    expected_revision: int,
    variant: str | None = None,
) -> EditorSession:
    """Commit one ordered list move."""
    _expect(session, expected_revision)
    found = move_node_instance(
        session.yaml_text,
        node_id,
        from_index,
        to_index,
        variant=variant,
    )
    return _commit(session, found.yaml_text)


def compose_session_node(
    session: EditorSession,
    node_id: str,
    compose: str,
    stages: Sequence[Mapping[str, object]],
    *,
    expected_revision: int,
    variant: str | None = None,
) -> EditorSession:
    """Commit an ordered multi-stage composition at one node."""
    _expect(session, expected_revision)
    found = compose_node(
        session.yaml_text,
        node_id,
        compose,
        stages,
        variant=variant,
    )
    return _commit(session, found.yaml_text)


def place_session_node(
    session: EditorSession,
    node_id: str,
    at: str | Sequence[str],
    settings: Mapping[str, object],
    *,
    expected_revision: int,
    variant: str | None = None,
) -> EditorSession:
    """Commit a custom operator placement or region."""
    _expect(session, expected_revision)
    found = place_node(
        session.yaml_text,
        node_id,
        at,
        settings,
        variant=variant,
    )
    return _commit(session, found.yaml_text)


def set_session_snapshot_before(
    session: EditorSession,
    node_id: str,
    snapshot_name: str,
    *,
    expected_revision: int,
    variant: str | None = None,
) -> EditorSession:
    """Commit the processing snapshot and its aux-product request together."""
    _expect(session, expected_revision)
    found = set_snapshot_before(
        session.yaml_text,
        node_id,
        snapshot_name,
        variant=variant,
    )
    return _commit(session, found.yaml_text)


def set_session_output_product(
    session: EditorSession,
    name: str,
    *,
    enabled: bool,
    expected_revision: int,
    format: str | None = None,
    runs: Sequence[str] = (),
    keys: Sequence[str] = (),
    themes: Sequence[str] = (),
) -> EditorSession:
    """Commit one closed scientific-product request."""
    _expect(session, expected_revision)
    yaml_text = set_output_product(
        session.yaml_text,
        name,
        enabled=enabled,
        format=format,
        runs=runs,
        keys=keys,
        themes=themes,
    )
    return _commit(session, yaml_text)


def set_session_output_report(
    session: EditorSession,
    *,
    enabled: bool,
    expected_revision: int,
    rows: Sequence[str] = (),
    columns: Sequence[str] = ("mean", "std", "seconds"),
    reference: str | None = None,
    relative: Sequence[str] = (),
    formats: Sequence[str] = ("text",),
) -> EditorSession:
    """Commit or remove the deterministic report-table request."""
    _expect(session, expected_revision)
    yaml_text = set_output_report(
        session.yaml_text,
        enabled=enabled,
        rows=rows,
        columns=columns,
        reference=reference,
        relative=relative,
        formats=formats,
    )
    return _commit(session, yaml_text)


def undo(session: EditorSession, *, expected_revision: int) -> EditorSession:
    """Move one position back without discarding redo history."""
    _expect(session, expected_revision)
    if not session.can_undo:
        raise ConfigError("Nothing to undo in this editor session.")
    yaml_text = session.history[session.cursor - 1]
    replace_yaml(yaml_text)
    return replace(
        session,
        cursor=session.cursor - 1,
        revision=session.revision + 1,
        validated_digest=_digest(yaml_text),
    )


def redo(session: EditorSession, *, expected_revision: int) -> EditorSession:
    """Move one position forward in retained history."""
    _expect(session, expected_revision)
    if not session.can_redo:
        raise ConfigError("Nothing to redo in this editor session.")
    yaml_text = session.history[session.cursor + 1]
    replace_yaml(yaml_text)
    return replace(
        session,
        cursor=session.cursor + 1,
        revision=session.revision + 1,
        validated_digest=_digest(yaml_text),
    )


def mark_saved(session: EditorSession, *, expected_revision: int) -> EditorSession:
    """Advance the save baseline after an explicit successful save action."""
    _expect(session, expected_revision)
    digest = _digest(session.yaml_text)
    if digest == session.saved_digest:
        return session
    return replace(
        session,
        revision=session.revision + 1,
        saved_digest=digest,
    )


def mark_validated(session: EditorSession, *, expected_revision: int) -> EditorSession:
    """Record that explicit validation covered the exact current YAML."""
    _expect(session, expected_revision)
    digest = _digest(session.yaml_text)
    if digest == session.validated_digest:
        return session
    return replace(
        session,
        revision=session.revision + 1,
        validated_digest=digest,
    )


def load_session_yaml(
    session: EditorSession,
    yaml_text: str,
    *,
    expected_revision: int,
) -> EditorSession:
    """Explicitly load YAML, establishing a clean one-entry history."""
    _expect(session, expected_revision)
    found = replace_yaml(yaml_text)
    return EditorSession(
        history=(found.yaml_text,),
        cursor=0,
        revision=session.revision + 1,
        saved_digest=_digest(found.yaml_text),
        validated_digest=_digest(found.yaml_text),
    )


def _read_utf8(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_utf8(path: Path, yaml_text: str) -> None:
    path.write_text(yaml_text, encoding="utf-8")


def load_session_file(
    session: EditorSession,
    path: str | Path,
    *,
    expected_revision: int,
    read_text: Callable[[Path], str] = _read_utf8,
) -> EditorSession:
    """Read YAML only after an explicit, revision-checked load action."""
    _expect(session, expected_revision)
    yaml_text = read_text(Path(path))
    return load_session_yaml(
        session,
        yaml_text,
        expected_revision=expected_revision,
    )


def save_session_file(
    session: EditorSession,
    path: str | Path,
    *,
    expected_revision: int,
    write_text: Callable[[Path, str], None] = _write_utf8,
) -> EditorSession:
    """Write YAML and mark it clean only after that explicit write succeeds."""
    _expect(session, expected_revision)
    write_text(Path(path), session.yaml_text)
    return mark_saved(session, expected_revision=expected_revision)


__all__ = [
    "compose_session_node",
    "EditorSession",
    "RevisionConflict",
    "edit_session_many_node",
    "edit_session_node",
    "load_session_file",
    "load_session_yaml",
    "mark_saved",
    "mark_validated",
    "move_session_node_instance",
    "new_session",
    "place_session_node",
    "redo",
    "replace_session_yaml",
    "save_session_file",
    "set_session_output_product",
    "set_session_output_report",
    "set_session_snapshot_before",
    "undo",
]
