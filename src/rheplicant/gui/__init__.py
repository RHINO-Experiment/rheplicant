"""YAML-as-truth editor primitives, independent of the chosen web stack."""

from rheplicant.gui.document import (
    EditorSnapshot,
    NodeCard,
    replace_yaml,
    set_node,
    snapshot,
)
from rheplicant.gui.session import (
    EditorSession,
    RevisionConflict,
    edit_session_node,
    load_session_file,
    load_session_yaml,
    mark_saved,
    mark_validated,
    new_session,
    redo,
    replace_session_yaml,
    save_session_file,
    undo,
)

__all__ = [
    "EditorSnapshot",
    "EditorSession",
    "NodeCard",
    "RevisionConflict",
    "edit_session_node",
    "load_session_file",
    "load_session_yaml",
    "mark_saved",
    "mark_validated",
    "new_session",
    "redo",
    "replace_yaml",
    "replace_session_yaml",
    "save_session_file",
    "set_node",
    "snapshot",
    "undo",
]
