"""Closed output grammar, descriptor preflight, and A34 lease management."""

from __future__ import annotations

import dataclasses

# Reached through this module by name in tests and callers, and no longer
# used by this file's own code after the split -- so `ruff --fix` removes a
# plain import. `X as X` is the explicit re-export form it leaves alone.
import fcntl as fcntl
import os
import stat
from collections.abc import Mapping, Sequence
from typing import cast

from _rheplicant_bootstrap.errors import ConfigError
from _rheplicant_bootstrap.frozen import static_isinstance
from _rheplicant_bootstrap.types import SourceInput

from .output_grammar import (
    _COMMANDS,
    _PLAN4B_WRITE,
    _product_request,
    parse_output_grammar,
)
from .output_grammar import _OUTPUT_KEYS as _OUTPUT_KEYS
from .output_grammar import _PLAN4B_TOP as _PLAN4B_TOP
from .output_grammar import _PRODUCT_DEFAULT_FORMATS as _PRODUCT_DEFAULT_FORMATS
from .output_grammar import _PRODUCT_FORMATS as _PRODUCT_FORMATS
from .output_grammar import _REPORT_COLUMNS as _REPORT_COLUMNS
from .output_grammar import _REPORT_FORMATS as _REPORT_FORMATS
from .output_grammar import _REPORT_RELATIVE as _REPORT_RELATIVE
from .output_grammar import _STDOUT as _STDOUT
from .output_grammar import _WRITE_KEYS as _WRITE_KEYS
from .output_grammar import _closed_mapping as _closed_mapping
from .output_grammar import _mapping as _mapping
from .output_grammar import _report_request as _report_request
from .output_grammar import _required_true as _required_true
from .output_grammar import _unique_texts as _unique_texts

# Re-exported so this module's importers keep working. `X as X` is the
# explicit re-export form: a plain import of a name this file does not
# itself use is F401, and `ruff --fix` deletes it whatever the comment
# on the line says.
from .output_lease import _MARKER_ID as _MARKER_ID
from .output_lease import _MARKER_NAME as _MARKER_NAME
from .output_lease import _STATE_LOCK as _STATE_LOCK
from .output_lease import _acquire_lock as _acquire_lock
from .output_lease import _read_owned_marker as _read_owned_marker
from .output_lease import _validate_lock as _validate_lock
from .output_lease import acquire_output_lease as acquire_output_lease
from .output_lease import close_output_lease as close_output_lease
from .output_lease import require_open_output_lease as require_open_output_lease
from .output_lease import revalidate_output_ancestry as revalidate_output_ancestry
from .output_lease import verify_a34_under_lease as verify_a34_under_lease
from .output_lease import verify_publication_under_lease as verify_publication_under_lease
from .output_primitives import (
    _OPEN_DIRECTORY,
    _component_limit,
    _open_child,
    _require_entry,
    _target_identity,
)
from .output_primitives import _same_identity as _same_identity
from .paths import (
    decode_journal_temp,
    internal_names,
    journal_name,
    require_component_budget,
    target_digest,
)
from .paths import lock_name as lock_name
from .platform import OutputPlatform
from .types import (
    AncestorEntryInspection,
    OutputBinding,
    OutputPathInspection,
    OutputRequest,
    ParsedOutputSection,
    ProductRequest,
    RecoveryInspection,
    TargetIdentity,
)
from .types import OutputLease as OutputLease


def _invocation_directory(value: object, *, parsed: ParsedOutputSection) -> str | None:
    """Validate the invocation-level output override, or report its absence.

    The override exists so a caller can place one run's tree without editing
    the document: the bytes that ran stay the bytes the author wrote, which is
    what makes ``config.input.yaml`` and its digest mean anything.  Two rules
    keep it honest.  It refuses rather than replaces an authored
    ``outputs.dir``, because silently discarding a path someone wrote is the
    wrongness this layer exists to catch.  And it must be absolute: a document
    path resolves against the document's own directory, while an invocation
    parameter arrives from a caller whose directory this layer does not know,
    so there is no defensible base to join it to.
    """
    if value is None:
        return None
    if parsed.directory is not None:
        raise ConfigError(
            "outputs_dir: the document already sets outputs.dir; an invocation "
            "override refuses rather than replaces an authored path."
        )
    if not static_isinstance(value, str) or not str.__str__(value):
        raise ConfigError("outputs_dir: must be a non-empty string.")
    directory = str.__str__(value)
    if "\0" in directory:
        raise ConfigError("outputs_dir: contains NUL.")
    if not os.path.isabs(directory):
        raise ConfigError(
            "outputs_dir: must be absolute -- an invocation override has no "
            "document directory to resolve against."
        )
    if os.path.basename(directory) == "":
        raise ConfigError("outputs_dir: must end in a non-empty component.")
    return directory


def _invocation_write(
    value: object, *, parsed: ParsedOutputSection
) -> tuple[ProductRequest, ...] | None:
    """Validate the invocation-level product request, or report its absence.

    The companion to ``_invocation_directory``, and honest for the same reason:
    a program that runs documents on someone's behalf decides *whether* to keep
    the arrays, the same way it decides *where*, and neither decision belongs in
    a document about the science.  Each name is a Plan 4B selector taken at its
    default format, which is what an invocation can express without inventing a
    second grammar; anything finer -- a format, a subset of runs -- is a
    document's job.

    It refuses a document that already asks for products rather than merging
    with it, because a merge would silently produce a tree matching neither
    what the author asked for nor what the caller did.
    """
    if value is None:
        return None
    if parsed.products:
        raise ConfigError(
            "outputs_write: the document already requests products under "
            "outputs.write; an invocation override refuses rather than merges."
        )
    if static_isinstance(value, str) or not static_isinstance(value, Sequence):
        raise ConfigError("outputs_write: must be a sequence of selector names.")
    names: list[str] = []
    for entry in value:
        if not static_isinstance(entry, str) or str.__str__(entry) not in _PLAN4B_WRITE:
            raise ConfigError(f"outputs_write: {entry!r} is not one of {list(_PLAN4B_WRITE)}.")
        name = str.__str__(entry)
        if name in names:
            raise ConfigError(f"outputs_write: {name!r} is requested twice.")
        names.append(name)
    if not names:
        raise ConfigError("outputs_write: must name at least one selector.")
    return tuple(dataclasses.replace(_product_request(name, True), optional=True) for name in names)


def resolve_output_request(
    parsed: ParsedOutputSection,
    *,
    source: SourceInput,
    command: str,
    invocation_dir: str | None = None,
    invocation_write: Sequence[str] | None = None,
) -> OutputRequest:
    """Resolve lexical output path facts without following a symlink."""
    if type(parsed) is not ParsedOutputSection or type(source) is not SourceInput:
        raise ConfigError("output resolution requires exact parsed and source records.")
    if command not in _COMMANDS:
        raise ConfigError(f"unknown output command {command!r}.")
    override = _invocation_directory(invocation_dir, parsed=parsed)
    products = _invocation_write(invocation_write, parsed=parsed)
    explicit = parsed.directory is not None or override is not None
    # The override is already absolute and lexically checked, so it skips the
    # expansion the document form needs; both then meet the same final refusals.
    where = "outputs.dir" if override is None else "outputs_dir"
    raw_target = parsed.directory
    if override is not None:
        target = override
    elif raw_target is None:
        if command == "validate":
            target = None
        elif source.source_path == "<stdin>":
            if command == "run":
                raise ConfigError("outputs.dir: run from stdin requires an explicit directory.")
            target = None
        else:
            stem, _suffix = os.path.splitext(source.source_path)
            target = stem + ".results"
    else:
        try:
            expanded = os.path.expandvars(os.path.expanduser(raw_target))
        except Exception:
            raise ConfigError("outputs.dir: cannot expand path.") from None
        if "\0" in expanded:
            raise ConfigError("outputs.dir: contains NUL.")
        if not expanded or os.path.basename(expanded) == "":
            raise ConfigError("outputs.dir: must end in a non-empty component.")
        target = expanded if os.path.isabs(expanded) else os.path.join(source.base_dir, expanded)
    if target is not None:
        try:
            target = os.path.abspath(target)
        except Exception:
            raise ConfigError(f"{where}: cannot normalize path.") from None
        if target == os.path.abspath(os.sep):
            raise ConfigError(f"{where}: filesystem root is not an output target.")
        if target == os.path.abspath(source.base_dir):
            raise ConfigError(f"{where}: cannot equal the configuration base directory.")
    return OutputRequest(
        command=cast(str, command),
        target_path=target,
        explicit_dir=explicit,
        clobber=parsed.clobber,
        stdout=parsed.stdout,
        write_config=parsed.write_config,
        write_provenance=parsed.write_provenance,
        write_diagnostics=parsed.write_diagnostics,
        products=parsed.products if products is None else products,
        report=parsed.report,
    )


def parse_output_request(
    document: Mapping[str, object],
    *,
    source: SourceInput,
    command: str,
) -> OutputRequest:
    if not static_isinstance(document, Mapping):
        raise ConfigError("document: configuration root must be a mapping.")
    try:
        raw = document.get("outputs", {})
    except Exception:
        raise ConfigError("document: cannot read outputs section.") from None
    return resolve_output_request(
        parse_output_grammar(raw),
        source=source,
        command=command,
    )


def _recovery_inspection(parent_fd: int, absolute_target: str) -> RecoveryInspection:
    canonical = journal_name(absolute_target)
    digest = target_digest(absolute_target)
    prefix = f".rheplicant-jtmp-{digest}-"
    try:
        names = tuple(sorted(os.listdir(parent_fd)))
    except OSError:
        raise ConfigError("cannot inspect output transaction state.") from None
    canonical_present = canonical in names
    candidates = tuple(name for name in names if name.startswith(prefix))
    decoded = tuple(name for name in candidates if decode_journal_temp(absolute_target, name))
    reason = None
    if len(decoded) != len(candidates):
        reason = "illegal transaction update temporary"
    elif len(decoded) > 1:
        reason = "multiple transaction update temporaries"
    elif decoded and not canonical_present:
        reason = "transaction update exists without canonical journal"
    return RecoveryInspection(
        canonical_present,
        candidates,
        canonical_present or bool(candidates),
        reason,
    )


def inspect_output_path(
    request: OutputRequest,
    platform: OutputPlatform,
) -> OutputPathInspection:
    """Return facts from a read-only descriptor walk; perform no mutation."""
    if type(request) is not OutputRequest or request.target_path is None:
        raise ConfigError("output inspection requires an exact request with a target.")
    absolute = request.target_path
    if not os.path.isabs(absolute):
        raise ConfigError("output inspection requires an absolute target.")
    parent_path = os.path.dirname(absolute)
    target_name = os.path.basename(absolute)
    components = tuple(component for component in parent_path.split(os.sep) if component)
    current_fd = os.open(os.sep, _OPEN_DIRECTORY)
    current_path = os.sep
    ancestry: list[AncestorEntryInspection] = []
    missing: tuple[str, ...] = ()
    try:
        for index, component in enumerate(components):
            try:
                before = os.lstat(component, dir_fd=current_fd)
            except FileNotFoundError:
                missing = components[index:]
                break
            except OSError:
                raise ConfigError(f"cannot inspect output ancestor {component!r}.") from None
            if stat.S_ISLNK(before.st_mode):
                raise ConfigError(f"output path contains intermediate symlink {component!r}.")
            if not stat.S_ISDIR(before.st_mode):
                raise ConfigError(f"output ancestor {component!r} is not a directory.")
            entry = platform.inspect_ancestor_entry(
                current_fd,
                current_path,
                component,
                before,
            )
            _require_entry(entry)
            child_fd = _open_child(current_fd, component, before)
            os.close(current_fd)
            current_fd = child_fd
            current_path = os.path.join(current_path, component)
            ancestry.append(entry)
        nearest = current_path
        access = platform.inspect_access(current_fd, current_path)
        limit = _component_limit(current_fd)
        require_component_budget(
            (
                *missing,
                target_name,
                *internal_names(absolute),
            ),
            limit,
        )
        if missing:
            target = TargetIdentity(False, None, None, None)
            recovery = RecoveryInspection(False, (), False, None)
        else:
            target = _target_identity(current_fd, target_name)
            recovery = _recovery_inspection(current_fd, absolute)
        inspection = OutputPathInspection(
            request,
            absolute,
            nearest,
            missing,
            parent_path,
            target_name,
            target,
            access,
            tuple(ancestry),
            recovery,
            limit,
            OutputBinding(platform),
        )
    finally:
        os.close(current_fd)
    return inspection


__all__ = [
    "acquire_output_lease",
    "close_output_lease",
    "inspect_output_path",
    "parse_output_grammar",
    "parse_output_request",
    "require_open_output_lease",
    "resolve_output_request",
    "revalidate_output_ancestry",
    "verify_a34_under_lease",
    "verify_publication_under_lease",
]
