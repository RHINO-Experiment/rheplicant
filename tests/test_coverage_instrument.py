"""Nothing in this suite may switch the coverage instrument off.

``coverage`` installs its ``CTracer`` through ``sys.settrace``. A test that
sets its own tracer and then hands back ``None`` does not "clean up" -- it
UNINSTALLS coverage from that thread for the remainder of the session, and
neither pytest-cov nor the run's exit code says a word.

This is not hypothetical. Two tests did exactly that, and the cost was
measured before the fix:

* serial coverage was a strict SUBSET of parallel -- 82.35 % against 88.36 %,
  1825 statements, with ``ser_only == 0`` for all 238 files;
* the loss was not scattered: **1982 consecutive tests** recorded nothing,
  starting at the offending test and ending only where
  ``tests/config/inflight_helpers.py::best_ms`` happens to call
  ``collector.resume()`` -- which re-installs the tracer by accident, for its
  own unrelated reasons;
* every affected file was under ``_rheplicant_bootstrap/**`` or
  ``rheplicant/config/**``, and the window's extent was an accident of
  alphabetical collection order;
* ``plugins.py`` fell from 698 covered statements to 123 -- and the 123 were
  all module level, so not one line of it was recorded by any TEST.

The last point is why this is a defect and not a reporting curiosity: for
1982 tests the coverage gate could not fail, which is this repo's recurring
shape -- a guard that has stopped being able to fail.

Scanned by AST, not by text, and the first draft here is why. A regex for the
offending call matched this module's own docstring and matched the comment
left at the fix site explaining the bug -- prose about the mistake is
indistinguishable from the mistake to a text scan, and a guard that must
exclude the file describing it is one edit from excluding the file committing
it. The AST sees calls.

An autouse fixture comparing ``sys.gettrace()`` around every test was the
other option and is worse: it would run ten thousand times to catch a two-line
mistake, and would be perturbing the very thing it measures.
"""

from __future__ import annotations

import ast
import pathlib

_HERE = pathlib.Path(__file__).parent


def _sources() -> list[pathlib.Path]:
    return sorted(_HERE.rglob("*.py"))


def _settrace_calls(tree: ast.AST):
    """Every ``*.settrace(...)`` and ``settrace(...)`` call in one module."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name == "settrace":
            yield node


def _hands_back_none(call: ast.Call) -> bool:
    return (
        len(call.args) == 1
        and isinstance(call.args[0], ast.Constant)
        and call.args[0].value is None
    )


def test_no_test_hands_the_tracer_back_as_none():
    """The guard itself."""
    offenders = [
        f"{path.relative_to(_HERE)}:{call.lineno}"
        for path in _sources()
        for call in _settrace_calls(ast.parse(path.read_text()))
        if _hands_back_none(call)
    ]
    assert offenders == [], (
        "these lines uninstall coverage's tracer for the rest of the session "
        "rather than restoring what was there:\n  " + "\n  ".join(offenders)
        + "\nCapture `previous = sys.gettrace()` before installing your own "
        "and restore THAT. See this module's docstring for what it cost."
    )


def test_the_scan_still_finds_the_settrace_calls_it_guards():
    """ANTI-VACUITY, and the half that matters.

    The guard above passes trivially the day nothing matches -- a renamed
    directory, a moved test, a regex that stopped working. It is a scan for
    the ABSENCE of a pattern, so it has no failure of its own to prove it is
    still reading anything. This asserts the scan can still see the calls it
    exists to police.
    """
    found = [
        f"{path.relative_to(_HERE)}:{call.lineno}"
        for path in _sources()
        for call in _settrace_calls(ast.parse(path.read_text()))
    ]
    assert len(found) >= 4, (
        f"the scan found {len(found)} settrace() calls in tests/, against the "
        "four that were there when this was written (two installs and two "
        "restores). The guard above asserts an ABSENCE and so has no failure "
        "of its own to prove it still reads anything. Either the calls moved, "
        f"or the walk stopped matching: {found}"
    )
