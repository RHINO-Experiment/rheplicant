"""A stored format version is checked type-exactly, or it is not checked.

Both readers compared with ``!=`` alone. In Python ``3.0 != 3`` is False and
``True != 1`` is False, so a manifest storing ``"format_version": 3.0`` -- which
any JSON writer may produce for a whole number, and ``json.dumps(3.0)`` does --
was accepted as version 3, and the binary beside it was then parsed against a
byte layout nothing had verified. That is the one failure the version exists
to prevent.

``isinstance`` does not close it either: ``isinstance(True, int)`` is True.
The exact form is ``type(x) is int``, which is what
``_rheplicant_bootstrap/audit/integrity.py`` already used for the manifest's
``dict`` and the file list's ``list``. The version was the one field in that
module still compared loosely.

**Why this is a contract test rather than a unit test.** Both numbers are
external contracts: the integrity manifest is read by anyone verifying an
audit bundle, and the archive manifest is read by a different rheplicant than
the one that wrote it. A reader that accepts a version it does not understand
is worse than one that refuses, because the refusal is the only thing standing
between a changed byte layout and silently wrong numbers.
"""

from __future__ import annotations

import json

import pytest

from _rheplicant_bootstrap.audit.integrity import (
    INTEGRITY_FORMAT_VERSION,
    INTEGRITY_NAME,
)

#: The values that are ``==`` to an integer version and are not that version.
#: ``True`` is only equal to 1, so it is only a probe for a version of 1.
LOOSE = (
    pytest.param(lambda n: float(n), id="float"),
    pytest.param(lambda n: True if n == 1 else None, id="bool"),
    pytest.param(lambda n: str(n), id="str"),
)


class TestTheIntegrityManifest:
    def _verify(self, manifest):
        """``verify_tree`` does no IO: it takes the tree as path -> bytes."""
        from _rheplicant_bootstrap.audit.integrity import verify_tree

        return verify_tree({INTEGRITY_NAME: json.dumps(manifest).encode()})

    def test_the_declared_version_is_accepted(self):
        """Anti-vacuity: the shape below must be refusable, not just refused.

        Without this, a reader that refused EVERYTHING would pass every other
        test in this class.
        """
        problems = self._verify(
            {"format_version": INTEGRITY_FORMAT_VERSION, "files": []}
        )
        assert not [p for p in problems if "format_version" in p], problems

    @pytest.mark.parametrize("coerce", LOOSE)
    def test_a_loosely_equal_version_is_refused(self, coerce):
        value = coerce(INTEGRITY_FORMAT_VERSION)
        if value is None:
            pytest.skip("bool is only == to 1")
        problems = self._verify({"format_version": value, "files": []})
        assert any("format_version" in problem for problem in problems), (
            f"{value!r} ({type(value).__name__}) was accepted as version "
            f"{INTEGRITY_FORMAT_VERSION}. `==` alone does not separate them, "
            f"and {INTEGRITY_NAME} is read by anyone verifying a bundle"
        )


class TestTheArchiveManifest:
    """The archive's version guards a BYTE LAYOUT, which is the stronger case.

    A wrong-but-accepted version here does not produce a missing field; it
    produces a read at the wrong offset, which returns numbers.
    """

    def _read(self, tmp_path, version):
        from rheplicant.core.errors import StateValidationError
        from rheplicant.inference import archive

        # The manifest is the archive path with a .json suffix -- read from
        # `_manifest_path` rather than assumed, which cost one red run.
        target = tmp_path / "memory.bin"
        target.write_bytes(b"\x00" * 32)
        target.with_suffix(".json").write_text(
            json.dumps({"format_version": version, "leaves": []})
        )
        return archive, target, StateValidationError

    @pytest.mark.parametrize("coerce", LOOSE)
    def test_a_loosely_equal_version_is_refused(self, tmp_path, coerce):
        from rheplicant.inference.archive import _FORMAT_VERSION

        value = coerce(_FORMAT_VERSION)
        if value is None:
            pytest.skip("bool is only == to 1")
        archive, target, StateValidationError = self._read(tmp_path, value)
        with pytest.raises(StateValidationError) as raised:
            # `factorization` is never reached: the manifest's version is
            # checked first, which is the ordering this refusal depends on.
            archive.load_memory(target, None)
        assert "format_version" in str(raised.value) or "format version" in str(
            raised.value
        ), (
            f"{value!r} was not refused for being a "
            f"{type(value).__name__}: {raised.value}"
        )

    def test_the_refusal_names_the_type_rather_than_the_number(self, tmp_path):
        """A reader who wrote ``3.0`` is not helped by "expected 3, got 3".

        The message has to say which PROPERTY failed, or the next thing they
        try is writing ``3.0`` again.
        """
        from rheplicant.inference.archive import _FORMAT_VERSION

        archive, target, StateValidationError = self._read(
            tmp_path, float(_FORMAT_VERSION)
        )
        with pytest.raises(StateValidationError) as raised:
            archive.load_memory(target, None)
        assert "float" in str(raised.value), raised.value
