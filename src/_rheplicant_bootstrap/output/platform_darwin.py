"""Darwin fd-only ACL checks and ``renameatx_np(RENAME_EXCL)``."""

from __future__ import annotations

import ctypes
import errno
import os
import stat

from _rheplicant_bootstrap.errors import ConfigError

from .types import AccessInspection, AncestorEntryInspection

_ACL_TYPE_EXTENDED = 0x00000100
_ACL_FIRST_ENTRY = 0
_ACL_NEXT_ENTRY = 1
#: `<sys/acl.h>`: an extended ACE either ALLOWs or DENIEs. Only an ALLOW can
#: hand anyone a right they did not already have from the mode bits.
_ACL_EXTENDED_ALLOW = 1
_ACL_EXTENDED_DENY = 2
#: A pathological ACL will not be walked forever; the ceiling is far above any
#: real one (macOS puts a single ACE on a home directory).
_ACL_ENTRY_LIMIT = 1024
_RENAME_EXCL = 0x00000004


class DarwinOutputPlatform:
    def __init__(self) -> None:
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._rename = getattr(self._libc, "renameatx_np", None)
        if self._rename is not None:
            self._rename.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            self._rename.restype = ctypes.c_int
        self._acl_get_fd = getattr(self._libc, "acl_get_fd_np", None)
        self._acl_get_entry = getattr(self._libc, "acl_get_entry", None)
        self._acl_get_tag_type = getattr(self._libc, "acl_get_tag_type", None)
        self._acl_free = getattr(self._libc, "acl_free", None)
        if self._acl_get_fd is not None:
            self._acl_get_fd.argtypes = (ctypes.c_int, ctypes.c_int)
            self._acl_get_fd.restype = ctypes.c_void_p
        if self._acl_get_entry is not None:
            self._acl_get_entry.argtypes = (
                ctypes.c_void_p,
                ctypes.c_int,
                ctypes.POINTER(ctypes.c_void_p),
            )
            self._acl_get_entry.restype = ctypes.c_int
        if self._acl_get_tag_type is not None:
            self._acl_get_tag_type.argtypes = (
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int),
            )
            self._acl_get_tag_type.restype = ctypes.c_int
        if self._acl_free is not None:
            self._acl_free.argtypes = (ctypes.c_void_p,)
            self._acl_free.restype = ctypes.c_int

    def _access_acl(self, directory_fd: int) -> tuple[bool, bool, bool, str | None]:
        """Inspect one directory's extended ACL.

        Returns ``(trivial, grants_others, reliable, reason)``.

        ``trivial`` is the ACL-sense fact: no extended entries at all.
        ``grants_others`` is the question every caller actually asks -- whether
        the ACL hands anyone a right the mode bits did not already give them --
        and the two are NOT the same. A DENY-only ACL is non-trivial and grants
        nothing, which is precisely what macOS puts on every home directory
        (``group:everyone deny delete``): protection ADDED, never removed.
        Reading "non-trivial" as "unsafe" refused every project under ``~``.

        Anything unreadable reports ``grants_others`` True as well as
        ``reliable`` False, so a caller that ignores one still fails closed.
        """
        if None in (self._acl_get_fd, self._acl_get_entry, self._acl_get_tag_type, self._acl_free):
            return False, True, False, "cannot verify access control: ACL symbols are unavailable"
        ctypes.set_errno(0)
        handle = self._acl_get_fd(directory_fd, _ACL_TYPE_EXTENDED)
        if not handle:
            error = ctypes.get_errno()
            # Darwin reports ENOENT when an inode has no extended ACL.  That
            # is the ordinary, trivially protected case rather than an
            # inability to inspect the descriptor.
            if error == errno.ENOENT:
                return True, False, True, None
            return False, True, False, (
                f"cannot verify access control: acl_get_fd_np errno {error}"
            )
        try:
            return self._walk_entries(handle)
        finally:
            self._acl_free(handle)

    def _walk_entries(self, handle: object) -> tuple[bool, bool, bool, str | None]:
        """Classify every ACE, stopping at the first that grants something."""
        entry = ctypes.c_void_p()
        selector = _ACL_FIRST_ENTRY
        seen = 0
        while seen < _ACL_ENTRY_LIMIT:
            ctypes.set_errno(0)
            result = self._acl_get_entry(handle, selector, ctypes.byref(entry))
            if result != 0 or not entry.value:
                # A clean end of list. `EINVAL`/`ENOENT` are how Darwin spells
                # "no more entries"; anything else means the walk itself failed
                # and nothing may be concluded from a partial read.
                if result != 0 and ctypes.get_errno() not in (0, errno.EINVAL, errno.ENOENT):
                    return False, True, False, (
                        "cannot verify access control: acl_get_entry errno "
                        f"{ctypes.get_errno()}"
                    )
                return seen == 0, False, True, None if seen == 0 else "deny-only access ACL"
            tag = ctypes.c_int()
            ctypes.set_errno(0)
            if self._acl_get_tag_type(entry, ctypes.byref(tag)) != 0:
                return False, True, False, (
                    "cannot verify access control: acl_get_tag_type errno "
                    f"{ctypes.get_errno()}"
                )
            if tag.value != _ACL_EXTENDED_DENY:
                # An ALLOW ACE, or a tag this code does not recognise. Either
                # way it may grant something, and a guess is not available.
                return False, True, True, "access ACL grants rights beyond the mode bits"
            seen += 1
            selector = _ACL_NEXT_ENTRY
        return False, True, True, "access ACL is too long to verify"

    def inspect_access(self, directory_fd: int, parent_path: str) -> AccessInspection:
        try:
            row = os.fstat(directory_fd)
        except OSError as error:
            return AccessInspection(
                parent_path, os.geteuid(), -1, 0, False, True, True, False, str(error)
            )
        trivial, grants_others, reliable, reason = self._access_acl(directory_fd)
        return AccessInspection(
            parent_path,
            os.geteuid(),
            row.st_uid,
            stat.S_IMODE(row.st_mode),
            trivial,
            True,
            grants_others,
            reliable,
            reason,
        )

    def inspect_ancestor_entry(
        self,
        containing_fd: int,
        containing_path: str,
        child_name: str,
        child_stat: os.stat_result,
    ) -> AncestorEntryInspection:
        containing = os.fstat(containing_fd)
        access = self.inspect_access(containing_fd, containing_path)
        sticky = bool(containing.st_mode & stat.S_ISVTX)
        writable = bool(stat.S_IMODE(containing.st_mode) & 0o022)
        rename_protected = (
            access.reliable
            and not access.access_acl_grants_others
            and (not writable or (sticky and child_stat.st_uid == os.geteuid()))
        )
        reason = access.reason
        if access.reliable and not rename_protected:
            reason = "ancestor entry can be renamed by a non-owner"
        return AncestorEntryInspection(
            containing_path,
            child_name,
            containing.st_dev,
            containing.st_ino,
            child_stat.st_dev,
            child_stat.st_ino,
            child_stat.st_uid,
            sticky,
            rename_protected,
            access.reliable,
            reason,
        )

    def verify_rename_noreplace_available(self, directory_fd: int) -> None:
        if self._rename is None:
            raise ConfigError("atomic no-replace rename is unavailable.")
        try:
            os.fstatvfs(directory_fd)
        except OSError:
            raise ConfigError("cannot verify atomic no-replace rename support.") from None

    def rename_noreplace(
        self,
        old_dir_fd: int,
        old_name: str,
        new_dir_fd: int,
        new_name: str,
    ) -> None:
        self.verify_rename_noreplace_available(old_dir_fd)
        ctypes.set_errno(0)
        result = self._rename(
            old_dir_fd,
            os.fsencode(old_name),
            new_dir_fd,
            os.fsencode(new_name),
            _RENAME_EXCL,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(error, os.strerror(error), new_name)
            raise OSError(error, os.strerror(error), new_name)


__all__ = ["DarwinOutputPlatform"]
