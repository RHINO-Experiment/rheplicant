"""YAML-as-truth editor primitives, independent of the chosen web stack."""

from rheplicant.gui.document import (
    EditorSnapshot,
    NodeCard,
    replace_yaml,
    set_node,
    snapshot,
)

__all__ = [
    "EditorSnapshot",
    "NodeCard",
    "replace_yaml",
    "set_node",
    "snapshot",
]
