from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from rheplicant.config import ConfigError
from rheplicant.gui.session import (
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

from .test_document import BASE

EDITED = BASE.replace("gain: 1.0", "gain: 1.25")
SECOND_EDIT = BASE.replace("gain: 1.0", "gain: 1.5")


def test_new_session_is_immutable_clean_and_live_preflight_is_current():
    session = new_session(BASE)

    assert session.yaml_text == BASE
    assert session.history == (BASE,)
    assert session.revision == 0
    assert session.dirty is False
    assert session.validation_stale is False
    assert session.can_undo is False
    assert session.can_redo is False
    with pytest.raises(FrozenInstanceError):
        session.revision = 1  # type: ignore[misc]


def test_validation_staleness_compares_the_exact_content_digest():
    current = new_session(BASE)
    stale = replace(current, validated_digest=None)

    assert stale.validation_stale is True
    refreshed = mark_validated(stale, expected_revision=0)
    assert refreshed.validation_stale is False
    assert refreshed.revision == 1


def test_edit_undo_redo_and_branch_are_immutable_memory_only_transitions():
    original = new_session(BASE)
    first = replace_session_yaml(original, EDITED, expected_revision=0)
    second = replace_session_yaml(first, SECOND_EDIT, expected_revision=1)

    assert original.history == (BASE,)
    assert second.history == (BASE, EDITED, SECOND_EDIT)
    assert second.revision == 2
    assert second.dirty is True
    assert second.can_undo is True
    assert second.can_redo is False

    back = undo(second, expected_revision=2)
    assert back.yaml_text == EDITED
    assert back.revision == 3
    assert back.can_redo is True

    forward = redo(back, expected_revision=3)
    assert forward.yaml_text == SECOND_EDIT
    assert forward.revision == 4

    branch_point = undo(forward, expected_revision=4)
    branched = replace_session_yaml(
        branch_point,
        BASE.replace("gain: 1.0", "gain: 2.0"),
        expected_revision=5,
    )
    assert branched.history == (
        BASE,
        EDITED,
        BASE.replace("gain: 1.0", "gain: 2.0"),
    )
    assert branched.can_redo is False


def test_node_edit_uses_the_current_yaml_and_preserves_noop_revision():
    original = new_session(BASE)
    edited = edit_session_node(
        original,
        "gain",
        enabled=True,
        settings={"type": "GainOperator", "gain": 1.25},
        expected_revision=0,
    )
    noop = edit_session_node(
        edited,
        "gain",
        enabled=True,
        settings={"gain": 1.25, "type": "GainOperator"},
        expected_revision=1,
    )

    assert edited.yaml_text != BASE
    assert edited.revision == 1
    assert noop is edited


def test_saved_and_live_preflight_states_follow_exact_yaml_across_history():
    original = new_session(BASE)
    changed = replace_session_yaml(original, EDITED, expected_revision=0)

    assert changed.dirty is True
    assert changed.validation_stale is False

    saved = mark_saved(changed, expected_revision=1)
    assert saved.revision == 2
    assert saved.dirty is False
    assert saved.validation_stale is False

    back = undo(saved, expected_revision=2)
    assert back.yaml_text == BASE
    assert back.dirty is True
    assert back.validation_stale is False

    forward = redo(back, expected_revision=3)
    assert forward.yaml_text == EDITED
    assert forward.dirty is False
    assert forward.validation_stale is False


def test_explicit_load_is_clean_resets_history_and_runs_live_preflight():
    changed = replace_session_yaml(new_session(BASE), EDITED, expected_revision=0)
    loaded = load_session_yaml(changed, SECOND_EDIT, expected_revision=1)

    assert loaded.yaml_text == SECOND_EDIT
    assert loaded.history == (SECOND_EDIT,)
    assert loaded.revision == 2
    assert loaded.dirty is False
    assert loaded.validation_stale is False
    assert loaded.can_undo is False
    assert loaded.can_redo is False


def test_every_mutating_transition_checks_the_optimistic_revision_first():
    session = new_session(BASE)

    with pytest.raises(RevisionConflict, match="expected revision 4.*current revision is 0"):
        replace_session_yaml(session, EDITED, expected_revision=4)
    with pytest.raises(RevisionConflict):
        edit_session_node(
            session,
            "gain",
            enabled=False,
            expected_revision=1,
        )
    with pytest.raises(RevisionConflict):
        mark_saved(session, expected_revision=True)  # type: ignore[arg-type]

    assert session.yaml_text == BASE
    assert session.revision == 0


def test_undo_and_redo_refuse_missing_history_without_changing_revision():
    session = new_session(BASE)
    with pytest.raises(ConfigError, match="Nothing to undo"):
        undo(session, expected_revision=0)

    changed = replace_session_yaml(session, EDITED, expected_revision=0)
    with pytest.raises(ConfigError, match="Nothing to redo"):
        redo(changed, expected_revision=1)


def test_file_io_exists_only_at_injected_explicit_boundaries():
    reads: list[Path] = []
    writes: list[tuple[Path, str]] = []
    session = replace_session_yaml(new_session(BASE), EDITED, expected_revision=0)

    loaded = load_session_file(
        session,
        Path("loaded.yaml"),
        expected_revision=1,
        read_text=lambda path: reads.append(path) or SECOND_EDIT,
    )
    saved = save_session_file(
        loaded,
        Path("saved.yaml"),
        expected_revision=2,
        write_text=lambda path, text: writes.append((path, text)),
    )

    assert reads == [Path("loaded.yaml")]
    assert writes == [(Path("saved.yaml"), SECOND_EDIT)]
    assert saved.dirty is False


def test_stale_file_boundary_does_no_io_and_failed_write_does_not_mark_clean():
    reads: list[Path] = []
    writes: list[tuple[Path, str]] = []
    dirty = replace_session_yaml(new_session(BASE), EDITED, expected_revision=0)

    with pytest.raises(RevisionConflict):
        load_session_file(
            dirty,
            Path("loaded.yaml"),
            expected_revision=0,
            read_text=lambda path: reads.append(path) or SECOND_EDIT,
        )
    with pytest.raises(OSError, match="disk full"):
        save_session_file(
            dirty,
            Path("saved.yaml"),
            expected_revision=1,
            write_text=lambda path, text: (_ for _ in ()).throw(OSError("disk full")),
        )

    assert reads == []
    assert writes == []
    assert dirty.dirty is True
    assert dirty.revision == 1
