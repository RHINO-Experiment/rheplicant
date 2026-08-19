"""Linux fd-only ACL checks and ``renameat2(RENAME_NOREPLACE)``."""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import stat

from _rheplicant_bootstrap.errors import ConfigError

from .types import AccessInspection, AncestorEntryInspection

_RENAME_NOREPLACE = 1


class LinuxOutputPlatform:
    def __init__(self) -> None:
        self._libc = ctypes.CDLL(None, use_errno=True)
        self._renameat2 = getattr(self._libc, "renameat2", None)
        if self._renameat2 is not None:
            self._renameat2.argtypes = (
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_int,
                ctypes.c_char_p,
                ctypes.c_uint,
            )
            self._renameat2.restype = ctypes.c_int
        acl_library = ctypes.util.find_library("acl")
        self._acl = (
            None if acl_library is None else ctypes.CDLL(acl_library, use_errno=True)
        )
        self._acl_get_fd = None if self._acl is None else getattr(self._acl, "acl_get_fd", None)
        self._acl_equiv_mode = (
            None if self._acl is None else getattr(self._acl, "acl_equiv_mode", None)
        )
        self._acl_free = None if self._acl is None else getattr(self._acl, "acl_free", None)
        if self._acl_get_fd is not None:
            self._acl_get_fd.argtypes = (ctypes.c_int,)
            self._acl_get_fd.restype = ctypes.c_void_p
        if self._acl_equiv_mode is not None:
            self._acl_equiv_mode.argtypes = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint))
            self._acl_equiv_mode.restype = ctypes.c_int
        if self._acl_free is not None:
            self._acl_free.argtypes = (ctypes.c_void_p,)
            self._acl_free.restype = ctypes.c_int

    def _access_acl(self, directory_fd: int) -> tuple[bool, bool, str | None]:
        if None in (self._acl_get_fd, self._acl_equiv_mode, self._acl_free):
            return False, False, "cannot verify access control: libacl is unavailable"
        ctypes.set_errno(0)
        handle = self._acl_get_fd(directory_fd)
        if not handle:
            error = ctypes.get_errno()
            return False, False, f"cannot verify access control: acl_get_fd errno {error}"
        try:
            mode = ctypes.c_uint()
            ctypes.set_errno(0)
            result = self._acl_equiv_mode(handle, ctypes.byref(mode))
            if result == 0:
                return True, True, None
            if result == 1:
                return False, True, "non-trivial access ACL"
            error = ctypes.get_errno()
            return False, False, f"cannot verify access control: acl_equiv_mode errno {error}"
        finally:
            self._acl_free(handle)

    @staticmethod
    def _default_acl(directory_fd: int) -> tuple[bool, bool, str | None]:
        try:
            os.getxattr(directory_fd, "system.posix_acl_default")
        except OSError as error:
            if error.errno in (errno.ENODATA, getattr(errno, "ENOATTR", errno.ENODATA)):
                return True, True, None
            return False, False, f"cannot verify default access control: errno {error.errno}"
        return False, True, "non-trivial default ACL"

    def inspect_access(self, directory_fd: int, parent_path: str) -> AccessInspection:
        try:
            row = os.fstat(directory_fd)
        except OSError as error:
            return AccessInspection(
                parent_path, os.geteuid(), -1, 0, False, False, False, str(error)
            )
        access_trivial, access_reliable, access_reason = self._access_acl(directory_fd)
        default_trivial, default_reliable, default_reason = self._default_acl(directory_fd)
        reliable = access_reliable and default_reliable
        reason = access_reason or default_reason
        return AccessInspection(
            parent_path,
            os.geteuid(),
            row.st_uid,
            stat.S_IMODE(row.st_mode),
            access_trivial,
            default_trivial,
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
            and access.default_acl_is_trivial
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
        if self._renameat2 is None:
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
        result = self._renameat2(
            old_dir_fd,
            os.fsencode(old_name),
            new_dir_fd,
            os.fsencode(new_name),
            _RENAME_NOREPLACE,
        )
        if result != 0:
            error = ctypes.get_errno()
            if error == errno.EEXIST:
                raise FileExistsError(error, os.strerror(error), new_name)
            raise OSError(error, os.strerror(error), new_name)


__all__ = ["LinuxOutputPlatform"]
