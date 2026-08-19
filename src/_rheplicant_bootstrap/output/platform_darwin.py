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
        if self._acl_free is not None:
            self._acl_free.argtypes = (ctypes.c_void_p,)
            self._acl_free.restype = ctypes.c_int

    def _access_acl(self, directory_fd: int) -> tuple[bool, bool, str | None]:
        if None in (self._acl_get_fd, self._acl_get_entry, self._acl_free):
            return False, False, "cannot verify access control: ACL symbols are unavailable"
        ctypes.set_errno(0)
        handle = self._acl_get_fd(directory_fd, _ACL_TYPE_EXTENDED)
        if not handle:
            error = ctypes.get_errno()
            return False, False, f"cannot verify access control: acl_get_fd_np errno {error}"
        try:
            entry = ctypes.c_void_p()
            ctypes.set_errno(0)
            result = self._acl_get_entry(handle, _ACL_FIRST_ENTRY, ctypes.byref(entry))
            if result == 0 and entry.value:
                return False, True, "non-trivial access ACL"
            if result != 0 and ctypes.get_errno() not in (0, errno.EINVAL, errno.ENOENT):
                return False, False, (
                    "cannot verify access control: acl_get_entry errno "
                    f"{ctypes.get_errno()}"
                )
            return True, True, None
        finally:
            self._acl_free(handle)

    def inspect_access(self, directory_fd: int, parent_path: str) -> AccessInspection:
        try:
            row = os.fstat(directory_fd)
        except OSError as error:
            return AccessInspection(
                parent_path, os.geteuid(), -1, 0, False, True, False, str(error)
            )
        trivial, reliable, reason = self._access_acl(directory_fd)
        return AccessInspection(
            parent_path,
            os.geteuid(),
            row.st_uid,
            stat.S_IMODE(row.st_mode),
            trivial,
            True,
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
            and access.access_acl_is_trivial
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
