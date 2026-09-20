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
        problems = self._verify({"format_version": INTEGRITY_FORMAT_VERSION, "files": []})
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
        assert "format_version" in str(raised.value) or "format version" in str(raised.value), (
            f"{value!r} was not refused for being a {type(value).__name__}: {raised.value}"
        )

    def test_the_refusal_names_the_type_rather_than_the_number(self, tmp_path):
        """A reader who wrote ``3.0`` is not helped by "expected 3, got 3".

        The message has to say which PROPERTY failed, or the next thing they
        try is writing ``3.0`` again.
        """
        from rheplicant.inference.archive import _FORMAT_VERSION

        archive, target, StateValidationError = self._read(tmp_path, float(_FORMAT_VERSION))
        with pytest.raises(StateValidationError) as raised:
            archive.load_memory(target, None)
        assert "float" in str(raised.value), raised.value


def test_no_stored_version_is_compared_loosely_anywhere():
    """The census: every ``format_version`` comparison in ``src/`` is type-exact.

    The three readers above were found one at a time. This is the check that
    would have found all of them at once, and the one that catches the fourth:
    a comparison of a stored ``format_version`` must be preceded by a
    ``type(...) is not int`` clause guarding it.

    Read as TEXT rather than by AST on purpose. The clauses are spread across
    boolean chains of five and six terms, and what matters is whether the type
    guard is present in the same condition -- a property of the written
    condition, which is what a reviewer reads.
    """
    import pathlib
    import re

    src = pathlib.Path(__file__).resolve().parents[1] / "src"
    loose = []
    for path in sorted(src.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in re.finditer(
            # Four spellings, not two. The first version of this census saw
            # only a subscript and a `.get(...)`, so `row.format_version != 1`
            # and a bare `format_version != X` were invisible to it -- and
            # `output/transaction.py` was carrying exactly the defect this
            # file exists to find, unguarded, while the census read green.
            # That is the census's own failure mode: a matcher narrower than
            # the thing it guards.
            r"^.*\[.format_version.\]\s*!=\s*"
            r"|^.*\.get\(.format_version.\)\s*!=\s*"
            r"|^.*\.format_version\s*!=\s*"
            # A bare name, wherever it sits on the line: the guarded
            # comparison in entry.py is the SECOND term of its condition, so a
            # line-anchored pattern missed it. The lookbehind is what keeps
            # this branch from re-matching the two spellings above.
            r"|^.*(?<![\w.])format_version\s*!=\s*"
            # Reversed operands and getattr. Neither is used for a
            # format_version today, but `1 != x` IS this codebase's style in
            # four other places, so the next one written that way would be
            # invisible to the census that exists to find it -- which is how
            # the two branches above came to be missing.
            r"|^.*!=\s*[\w.\[\]\"']*format_version"
            r"|^.*getattr\([^\n]*format_version[^\n]*\)\s*!=\s*",
            text,
            re.M,
        ):
            line_no = text[: match.start()].count("\n")
            window = "\n".join(text.splitlines()[max(0, line_no - 25) : line_no + 1])
            # The guard must be ON format_version. Asking only whether the
            # window contains "type(" was the first version of this check and
            # it was VACUOUS: every one of these conditions type-checks its
            # neighbouring fields -- `type(value) is not dict`,
            # `type(value["requests"]) is not list` -- so the substring was
            # always present and the census passed with all three fixes
            # reverted. Measured, not reasoned: reverting one and re-running
            # is what showed it.
            # Line-scoped, so the nested parenthesis in
            # `type(manifest.get("format_version"))` does not stop the match --
            # a `[^)]*` pattern cannot cross it, which cost one red run. The
            # window is 25 lines because archive.py's guard sits above a long
            # refusal message.
            if not re.search(r"type\([^\n]*format_version[^\n]*is\s+not\s+int", window):
                loose.append(f"{path.relative_to(src)}:{line_no + 1}")
    assert not loose, (
        f"these compare a stored format_version without a type check in the "
        f"same condition: {loose}. `1.0 != 1` and `True != 1` are both False, "
        "so the version would be accepted and the payload read against a shape "
        "nothing verified"
    )
