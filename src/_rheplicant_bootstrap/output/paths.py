"""Closed sibling-name codecs and component-budget checks."""

from __future__ import annotations

import hashlib
import os
import re

from _rheplicant_bootstrap.errors import ConfigError

TRANSACTION_PHASES = (
    "preparing",
    "prepared",
    "backup_moved",
    "staging_published",
    "target_durable",
    "backup_removed",
)
_TRANSACTION_ID = re.compile(r"[0-9a-f]{32}")


def target_digest(absolute_target: str) -> str:
    if type(absolute_target) is not str or not os.path.isabs(absolute_target):
        raise ConfigError("output target digest requires an absolute path.")
    return hashlib.sha256(os.fsencode(absolute_target)).hexdigest()


def lock_name(absolute_target: str) -> str:
    return f".rheplicant-lock-{target_digest(absolute_target)}.lock"


def journal_name(absolute_target: str) -> str:
    return f".rheplicant-journal-{target_digest(absolute_target)}.json"


def staging_name(absolute_target: str, transaction_id: str) -> str:
    _require_transaction_id(transaction_id)
    return f".rheplicant-stage-{target_digest(absolute_target)}-{transaction_id}"


def backup_name(absolute_target: str, transaction_id: str) -> str:
    _require_transaction_id(transaction_id)
    return f".rheplicant-backup-{target_digest(absolute_target)}-{transaction_id}"


def journal_temp_name(
    absolute_target: str,
    transaction_id: str,
    phase: str,
) -> str:
    _require_transaction_id(transaction_id)
    if phase not in TRANSACTION_PHASES:
        raise ConfigError("unknown transaction phase.")
    return (
        f".rheplicant-jtmp-{target_digest(absolute_target)}-"
        f"{transaction_id}-{phase}.tmp"
    )


def failure_name(absolute_target: str, publication: str, transaction_id: str) -> str:
    _require_transaction_id(transaction_id)
    if publication not in ("refused", "error"):
        raise ConfigError("failure sibling kind must be refused or error.")
    leaf = os.path.basename(absolute_target)
    return f"{leaf}.{publication}-{transaction_id}"


def decode_journal_temp(absolute_target: str, name: str) -> tuple[str, str] | None:
    prefix = f".rheplicant-jtmp-{target_digest(absolute_target)}-"
    if not name.startswith(prefix) or not name.endswith(".tmp"):
        return None
    body = name[len(prefix) : -4]
    transaction_id, separator, phase = body.partition("-")
    if (
        not separator
        or _TRANSACTION_ID.fullmatch(transaction_id) is None
        or phase not in TRANSACTION_PHASES
    ):
        return None
    return transaction_id, phase


def internal_names(absolute_target: str) -> tuple[str, ...]:
    transaction_id = "f" * 32
    names = [
        lock_name(absolute_target),
        journal_name(absolute_target),
        staging_name(absolute_target, transaction_id),
        backup_name(absolute_target, transaction_id),
        failure_name(absolute_target, "refused", transaction_id),
        failure_name(absolute_target, "error", transaction_id),
    ]
    names.extend(
        journal_temp_name(absolute_target, transaction_id, phase)
        for phase in TRANSACTION_PHASES
    )
    return tuple(names)


def require_component_budget(names: tuple[str, ...], component_limit: int) -> None:
    if type(component_limit) is not int or component_limit <= 0:
        raise ConfigError("output filesystem reported an invalid NAME_MAX.")
    for name in names:
        if type(name) is not str or not name or "/" in name or "\0" in name:
            raise ConfigError("output component is invalid.")
        if len(os.fsencode(name)) > component_limit:
            raise ConfigError(
                f"output component {name!r} exceeds leased NAME_MAX {component_limit}."
            )


def _require_transaction_id(transaction_id: str) -> None:
    if type(transaction_id) is not str or _TRANSACTION_ID.fullmatch(transaction_id) is None:
        raise ConfigError("transaction id must be 32 lowercase hexadecimal characters.")


__all__ = [
    "TRANSACTION_PHASES",
    "backup_name",
    "decode_journal_temp",
    "failure_name",
    "internal_names",
    "journal_name",
    "journal_temp_name",
    "lock_name",
    "require_component_budget",
    "staging_name",
    "target_digest",
]
