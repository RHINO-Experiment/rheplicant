"""Frozen, JAX-free records shared by output preflight and transactions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from _rheplicant_bootstrap.audit.types import ArtefactMaterialization

StdoutMode: TypeAlias = Literal["none", "summary", "verbose"]
CommandMode: TypeAlias = Literal["validate", "run", "script"]
TransactionPhase: TypeAlias = Literal[
    "preparing",
    "prepared",
    "backup_moved",
    "staging_published",
    "target_durable",
    "backup_removed",
]


@dataclass(frozen=True, slots=True)
class ParsedOutputSection:
    directory: str | None
    clobber: bool
    stdout: StdoutMode
    write_config: bool
    write_provenance: bool
    write_diagnostics: Literal["json"]


@dataclass(frozen=True, slots=True)
class OutputRequest:
    command: CommandMode
    target_path: str | None
    explicit_dir: bool
    clobber: bool
    stdout: StdoutMode
    write_config: bool
    write_provenance: bool
    write_diagnostics: Literal["json"]


@dataclass(frozen=True, slots=True)
class AccessInspection:
    parent_path: str
    effective_uid: int
    owner_uid: int
    mode: int
    access_acl_is_trivial: bool
    default_acl_is_trivial: bool
    reliable: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class AncestorEntryInspection:
    containing_path: str
    child_name: str
    containing_device: int
    containing_inode: int
    child_device: int
    child_inode: int
    child_owner_uid: int
    sticky: bool
    rename_protected: bool
    reliable: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class RecoveryInspection:
    canonical_present: bool
    update_temp_names: tuple[str, ...]
    requires_recovery: bool
    reason: str | None


@dataclass(frozen=True, slots=True)
class OutputMarker:
    format_version: int
    run_directory_id: str


@dataclass(frozen=True, slots=True)
class TargetIdentity:
    exists: bool
    device: int | None
    inode: int | None
    marker_id: str | None


@dataclass(frozen=True, slots=True)
class OutputPathInspection:
    request: OutputRequest
    absolute_target: str
    nearest_existing_ancestor: str
    missing_components: tuple[str, ...]
    parent_path: str
    target_name: str
    target: TargetIdentity
    access: AccessInspection
    ancestry: tuple[AncestorEntryInspection, ...]
    recovery: RecoveryInspection
    component_limit: int


@dataclass(frozen=True, slots=True)
class OutputLease:
    request: OutputRequest
    parent_fd: int
    parent_path: str
    target_name: str
    lock_fd: int
    lock_name: str
    journal_name: str
    ancestry: tuple[AncestorEntryInspection, ...]
    component_limit: int

    def __enter__(self) -> OutputLease:
        from .manager import require_open_output_lease

        require_open_output_lease(self)
        return self

    def __exit__(self, *_exception: object) -> None:
        from .manager import close_output_lease

        close_output_lease(self)


@dataclass(frozen=True, slots=True)
class PublicationLease:
    lease: OutputLease
    component_limit: int


@dataclass(frozen=True, slots=True)
class VerifiedOutputLease:
    publication: PublicationLease
    original: TargetIdentity


@dataclass(frozen=True, slots=True)
class TransactionHandle:
    lease: OutputLease
    publication_lease: PublicationLease
    verified: VerifiedOutputLease | None
    publication: Literal["success", "refused", "error"]
    transaction_id: str
    guarded_target_name: str
    publish_name: str
    journal_name: str
    staging_name: str
    backup_name: str | None
    marker: OutputMarker


@dataclass(frozen=True, slots=True)
class TransactionInterruptionState:
    transaction_id: str | None
    operation: Literal[
        "stage_bundle",
        "replace_staged_metadata",
        "publish_success",
        "publish_failure",
        "discard_staging",
    ]
    handle: TransactionHandle | None
    unreported_materializations: tuple[ArtefactMaterialization, ...]
    exception_type: str
    exception_message: str


@dataclass(frozen=True, slots=True)
class TransactionJournal:
    format_version: int
    transaction_id: str
    publication: Literal["success", "refused", "error"]
    phase: TransactionPhase
    guarded_target_name: str
    publish_name: str
    journal_name: str
    original: TargetIdentity
    staging_name: str
    staging_identity: TargetIdentity | None
    backup_name: str | None
    backup_identity: TargetIdentity | None
    published_identity: TargetIdentity | None
    new_marker_id: str


@dataclass(frozen=True, slots=True)
class RecoveryOutcome:
    action: Literal[
        "none",
        "restored_backup",
        "kept_published",
        "cleaned_backup",
        "cleaned_preparing",
        "completed_failure_publication",
    ]
    target_present: bool
    preserved_names: tuple[str, ...]


__all__ = [
    "AccessInspection",
    "AncestorEntryInspection",
    "CommandMode",
    "OutputLease",
    "OutputMarker",
    "OutputPathInspection",
    "OutputRequest",
    "ParsedOutputSection",
    "PublicationLease",
    "RecoveryInspection",
    "RecoveryOutcome",
    "StdoutMode",
    "TargetIdentity",
    "TransactionHandle",
    "TransactionInterruptionState",
    "TransactionJournal",
    "TransactionPhase",
    "VerifiedOutputLease",
]
