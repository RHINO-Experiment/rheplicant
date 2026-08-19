"""The pass, its registry, and the position in `load_document` that is the point.

**The one test here no later task repeats** is :class:`TestThePhaseGuard`.
Tasks 3-12 each test that their check FIRES; none of them tests that it fires
before ``build_resources``, and the hook is one call in ``load_document``.  If
the assertions there are weak, moving that call below ``build_resources`` is a
green edit that undoes the whole plan.  The four cases here are SYNTHETIC --
they register their own lambdas -- so they prove the hook's POSITION and no
real check's phase; the ten real assertions the definition of done asks for
are Tasks 3-12's, one each.

**Anti-vacuity is deliberate throughout.**  ``CHECKS`` is empty when this
module lands, so an assertion of the form "every id in ``CHECKS`` is a §6 id"
is trivially true and stays trivially true if the §6 extractor breaks.  Every
such assertion here is paired with one that fails when its matcher stops
matching -- the shape ``test_config_fixture_contract.py:413`` uses, and the
shape 2C shipped without and paid for.
"""

import ast
import importlib
import io
import os
import pathlib
import re
import sqlite3
import subprocess
import sys
import tarfile
import time
import warnings
from collections.abc import Callable
from typing import NamedTuple

import pytest

import rheplicant.config as config_package
import rheplicant.config.document as document_module
import rheplicant.config.sections.runtime as runtime_module
import rheplicant.config.values as values_module
from _rheplicant_bootstrap.layering import initial_merge
from _rheplicant_bootstrap.types import Origin
from _rheplicant_bootstrap.variants import enumerate_layers_once
from rheplicant.config.document import load_document
from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import ConfigWarning, refuse, report, warn
from rheplicant.config.preflight import (
    _SECTIONS,
    CHECKS,
    _structural,
    preflight,
    register,
)
from rheplicant.config.sections.runs import run_document
from rheplicant.core.operator import AbstractOperator
from rheplicant.radio import NoiseOperator
from tests.config.preflight_helpers import (
    BASE_MODEL,
    UNREADABLE_BEAM,
    findings,
    ids,
    only,
    preflight_document,
    refusals,
)

#: ``tests/config/``.
_HERE = pathlib.Path(__file__).resolve().parent
#: ``tests/config/`` -> the repository root.
_ROOT = _HERE.parents[1]
_SCHEMA = (_ROOT / "docs" / "superpowers" / "specs"
           / "2026-08-09-rheplicant-config-schema-v1.md")

#: The preflight PACKAGE's directory, resolved through ``importlib`` rather
#: than through ``from rheplicant.config import preflight``.  Task 13 exports
#: the pre-flight FUNCTION from ``config/__init__.py``, and that binding
#: shadows the subpackage attribute: after it, ``from rheplicant.config import
#: preflight`` returns the FUNCTION, so ``.__file__`` would raise
#: ``AttributeError`` at import and this module would stop collecting entirely
#: -- on the last task of the plan.  Only ``sys.modules[...]`` and
#: ``importlib.import_module(...)`` return the module.
_PREFLIGHT_DIR = pathlib.Path(
    importlib.import_module("rheplicant.config.preflight").__file__
).resolve().parent

#: ``config/passes.py`` -- the runner Plan 3B extracted out of this package's
#: ``__init__``, so that the three passes this layer now has share one
#: de-duplication, one raise-guard and one ``where`` guard.
_PASSES = pathlib.Path(
    importlib.import_module("rheplicant.config.passes").__file__
).resolve()


def _preflight_sources() -> list[pathlib.Path]:
    """Every module the pre-flight pass's own code lives in.

    **The two static guards below walked ``_PREFLIGHT_DIR.glob("*.py")``, and
    that walk stopped covering the code they were written for the moment
    ``sweep`` and ``_check_where`` moved into ``config/passes.py``.**  The
    extraction is the kind of edit that makes a guard vacuous without making
    it red: every module still under ``preflight/`` still passes, and the file
    that now runs every check is simply not looked at.

    ``passes.py`` is held to P-1's boundary rather than to a weaker one on
    purpose.  It is the runner for the two IN-FLIGHT passes as well, and those
    may hold a built object -- but ``passes.py`` itself only reads a registry
    and calls functions, so the strictest of the three boundaries is the one
    it can meet, and meeting it is what keeps the pre-flight pass's guarantee
    true after the move.
    """
    return sorted(_PREFLIGHT_DIR.glob("*.py")) + [_PASSES]


@pytest.fixture
def registry():
    """``CHECKS``, EMPTIED for one test and restored afterwards.

    Restored, because a probe check left behind leaks into every later
    ``load_document`` in the session -- including other modules' -- and the
    failures land nowhere near the cause.

    **Emptied**, because Tasks 3-12 register real ids into this same dict at
    import time.  A test that registers ``A2`` would then hit "registered
    twice", and one that asserts a refusal's message VERBATIM would read a
    real check's refusal instead of its own.

    It is module-local on purpose: a later task that asks for it gets
    ``fixture 'registry' not found``.  What every later task must do instead
    is register nothing from a test body and assert on the live registry only
    in subset forms.

    The tests that must see the REAL registry -- the schema-id census, the
    cost, the four boundary guards, and the anti-vacuity beam test --
    deliberately do not take this fixture.
    """
    saved = dict(CHECKS)
    CHECKS.clear()
    try:
        yield CHECKS
    finally:
        CHECKS.clear()
        CHECKS.update(saved)


#: Every check id schema §6 declares, in table order -- as a LITERAL.
#: ``docs/superpowers/`` is gitignored (``.gitignore:52``, confirmed with
#: ``git check-ignore -v``), so a worktree or a ``git archive`` tree has no
#: copy of the spec and a test that read it would error with
#: ``FileNotFoundError`` rather than fail honestly.  The cross-check against
#: the real file is :func:`_schema_ids_from_the_spec`, which SKIPS when it is
#: absent; where it is absent the literal is all that holds the census.
_SCHEMA_IDS: tuple[str, ...] = (
    tuple(f"A{n}" for n in range(1, 53))
    + tuple(f"B{n}" for n in range(1, 10))
    + tuple(f"C{n}" for n in range(1, 20))
)


def _schema_ids() -> list[str]:
    """Every check id schema §6 declares, in table order."""
    return list(_SCHEMA_IDS)


def _schema_ids_from_the_spec() -> list[str] | None:
    """The same list read off the spec -- or ``None`` when it is not here."""
    if not _SCHEMA.is_file():
        return None
    _, marker, after = _SCHEMA.read_text().partition(
        "\n## 6. Validation: every check before anything expensive\n")
    assert marker, "schema §6's heading has moved"
    body, _, _ = after.partition("\n## ")
    return re.findall(r"^\|\s*([ABC]\d+)\s*\|", body, re.M)


def _foot_imports(source: str,
                  package: str = "rheplicant.config.preflight") -> set[str]:
    """The modules ``preflight/__init__.py``'s foot import names.

    **``package`` is a parameter because ``inflight/`` needs the same answer
    about its own block, and two copies of this matcher would be two
    validators for one property.**  That is the ``_number``-vs-``_whole``
    divergence the layer's one-binding rule exists to prevent, one level up
    from a message: the correction below -- ``tree.body`` rather than
    ``ast.walk`` -- was measured and applied *here*, and a second copy written
    from memory in ``test_config_inflight.py`` would have been the pre-
    correction version, guarding nothing while reading as though it did.
    ``tests/config/test_config_inflight.py::TestTheImportBlockCannotRot``
    calls this with ``"rheplicant.config.inflight"`` and keeps its own
    anti-vacuity cases for the spellings only that package can produce.

    Read with ``ast`` rather than ``grep``: ``from ... import document`` and
    ``from ... import document, model`` and an aliased ``as _document_checks``
    all resolve, and a name inside a comment or a docstring does not -- which
    a grep tripwire cannot say, and 2D shipped one that counted four comment
    lines as four code lines.

    **``alias.name``, never the bound name.**  Every task's foot import is
    ALIASED, because the bare name collides with ``_structural``'s and
    ``preflight``'s ``document`` parameter and gives F811 twice.  A matcher
    reading ``entry.asname or entry.name`` would report ``_document_checks``
    and the completeness test would say the module is imported by nothing.

    ``node.level == 1`` is accepted as well as ``node.module``, so the
    relative spelling ``from . import document`` resolves too.

    **MODULE-LEVEL statements only, and that is a correction measured after it
    bit.**  This walked the whole tree, so an import inside ANY function body
    answered for the foot: a one-line ``from rheplicant.config.preflight import
    document`` inside ``preflight()`` -- added to drop the layer memo -- made
    deleting ``document``'s own foot import green across the whole of
    ``tests/config``, in the very file whose job is to notice that.  A foot
    import is a module-level statement by definition; anything else is a call-
    time import that does not run at package import and therefore registers
    nothing at the moment this guard is about.  ``tree.body`` is the whole fix,
    and ``test_the_matcher_reads_the_import_and_not_a_mention_of_one``'s
    ``in-a-function`` case is what keeps it.
    """
    found = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and (
                (node.module or "").startswith(package)
                or (node.level == 1 and node.module is None)):
            found.update(entry.name for entry in node.names)
        elif isinstance(node, ast.Import):
            for entry in node.names:
                if entry.name.startswith(package + "."):
                    found.add(entry.name.rsplit(".", 1)[-1])
    return found


#: Names no module under ``preflight/`` may import, under any spelling.  Each
#: is an entry point OUT of P-1: the five builders and the file resolver read
#: files or construct operators, ``resolve_value`` resolves a value node
#: (§2.4's boundary), ``resolve_path_on`` needs a built twin, and the three
#: document-level entry points would close a cycle as well as leave the phase.
#: Matched by the imported NAME rather than by module, because
#: ``config/resources.py`` also exports ``check_unknown_keys``, which §2.5
#: requires every check to use.
#:
#: ``build_runtime`` is on the list because ``load_document`` calls it and a
#: draft of this guard left it off -- a check importing and calling it passed
#: every assertion in this module.
_OUT_OF_SCOPE_NAMES = frozenset({
    "build_resources", "build_model", "build_observation", "build_inference",
    "build_runtime",
    "resolve_value", "resolve_path_on", "resolve_file_path",
    "load_document", "run_document", "run_forward",
})
#: Modules no module under ``preflight/`` may import at all.  ``document``
#: imports this package for the hook, so importing it back closes a cycle;
#: **the umbrella package is banned for a sharper reason** -- it re-exports
#: ``resolve_value``, ``build_resources``, ``load_document`` and
#: ``run_document`` as attributes of its own, so ``import rheplicant.config as
#: cfg`` followed by ``cfg.resolve_value(...)`` needs no banned NAME and no
#: alias trick.  A module here imports the specific module it wants
#: (``from rheplicant.config.errors import ConfigError``), never the package.
_OUT_OF_SCOPE_MODULES = frozenset({"rheplicant.config",
                                   "rheplicant.config.document"})

#: Filesystem verbs a runtime patch takes away DIRECTLY: verb -> the target
#: :func:`_forbid_the_filesystem` rebinds.  Read by that function, so the
#: table and the patches cannot drift apart.
#:
#: ``open_code``, ``statvfs``, ``rglob`` and ``connect`` were added after a
#: reviewer's probe module walked out of P-1 through them with the whole suite
#: green -- measured, exit 0, and a SQLite database left on disk under
#: ``preflight/``.  Two of the routes needed no cleverness at all:
#: ``io.open_code`` is a C-level opener that rebinding ``builtins.open`` does
#: not touch, and ``sqlite3.connect`` opens its path inside the extension.
#: The probe's other three -- ``os.getcwd``, ``os.readlink`` and ``os.lstat``
#: -- are closed on the STATIC ban only, and :data:`_STATIC_ONLY_CALLS` says
#: why with the measurement.
_FILESYSTEM_PATCHES: dict[str, str] = {
    "open": "builtins.open",
    "open_code": "io.open_code",
    "read_text": "pathlib.Path.read_text",
    "read_bytes": "pathlib.Path.read_bytes",
    "iterdir": "pathlib.Path.iterdir",
    "glob": "pathlib.Path.glob",
    "rglob": "pathlib.Path.rglob",
    "listdir": "os.listdir",
    "scandir": "os.scandir",
    "stat": "os.stat",
    "statvfs": "os.statvfs",
    "connect": "sqlite3.connect",
}

#: Filesystem verbs with no patch of their own, and the patched verb that IS
#: their floor -- each with an anti-vacuity case of its own in
#: :data:`_READING_CASES`, so the floor is measured rather than asserted.
_FLOORED_UNDER: dict[str, str] = {
    "walk": "scandir",
    "exists": "stat",
    "is_file": "stat",
    "is_dir": "stat",
}

#: Filesystem verbs on the STATIC ban that no runtime patch claims, each with
#: the reason -- because "banned and patched by nothing" is exactly the state
#: ``lstat`` and ``rglob`` were in when a reviewer's probe walked out through
#: them, and the difference between a hole and a decision is whether the
#: reason is written down and measured.
#:
#: Two kinds of reason, and the second is the interesting one.
#:
#: The five READERS come from modules ``preflight/`` may not import at all, so
#: there is no call to drive and a patch would be a claim about code that
#: cannot be written.
#:
#: The three METADATA syscalls are the ones ``coverage.py`` itself runs.
#: Measured: with ``os.lstat``, ``os.readlink`` and ``os.getcwd`` patched, a
#: coverage run of this class fails **22 of its own tests** --
#: ``coverage.inorout.should_trace`` canonicalises every newly traced file
#: through ``coverage.files.abs_file``, which is ``os.path.abspath(os.path.
#: realpath(path))``, which is exactly those three.  The guard would then fail
#: the SUITE rather than the pass under test, and which tests it failed would
#: depend on which files coverage had already seen -- a flake whose cause is a
#: guard.  They stay on the static ban, where they are branch-independent and
#: cost nothing, and the reviewer's three ``os`` routes are closed there.
_STATIC_ONLY_CALLS: dict[str, str] = {
    "fromfile": "numpy; goes through builtins.open, which IS patched",
    "loadtxt": "numpy; goes through builtins.open, which IS patched",
    "genfromtxt": "numpy; goes through builtins.open, which IS patched",
    "read_map": "healpy, which §0 forbids importing under preflight/",
    "read_alm": "healpy, which §0 forbids importing under preflight/",
    "lstat": "coverage.files.abs_file runs it on every newly traced file",
    "readlink": "coverage.files.abs_file runs it on every newly traced file",
    "getcwd": "coverage.files.abs_file runs it on every newly traced file",
}

#: The filesystem half of the static ban: every verb above, from whichever
#: table.  Written as a UNION rather than as a literal so that adding a patch
#: without adding the ban -- or the reverse, which is what shipped -- is not
#: expressible.
_FILESYSTEM_CALLS = (frozenset(_FILESYSTEM_PATCHES)
                     | frozenset(_FLOORED_UNDER)
                     | frozenset(_STATIC_ONLY_CALLS))

#: Called names no module under ``preflight/`` may write, whatever it imported
#: to get them.  Filesystem verbs first, then the entry points out of P-1.
#: This is the STATIC half of the boundary and its whole point is that it is
#: branch-independent: it reads code that never runs, which is exactly what
#: the runtime patches below cannot do.
_OUT_OF_SCOPE_CALLS = _FILESYSTEM_CALLS | _OUT_OF_SCOPE_NAMES


def _out_of_scope_imports(source: str) -> set[str]:
    """Every out-of-scope name or module ``source`` imports, at any depth.

    ``ast.walk``, so a lazy import inside a function is read the same as one
    at module scope -- a check that defers ``build_model`` to call time has
    left P-1 just as thoroughly.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.module in _OUT_OF_SCOPE_MODULES:
                found.add(node.module)
            # `from rheplicant import config` binds the package under a name
            # of its own, which is the same evasion as `import
            # rheplicant.config` and has a different AST shape.
            found.update(f"{node.module}.{entry.name}"
                         for entry in node.names
                         if f"{node.module}.{entry.name}"
                         in _OUT_OF_SCOPE_MODULES)
            found.update(entry.name for entry in node.names
                         if entry.name in _OUT_OF_SCOPE_NAMES)
        elif isinstance(node, ast.Import):
            found.update(entry.name for entry in node.names
                         if entry.name in _OUT_OF_SCOPE_MODULES)
    return found


def _out_of_scope_calls(source: str) -> set[str]:
    """Every out-of-scope call ``source`` writes, reached or not.

    ``f()`` and ``x.f()`` alike, by the name at the call site, so it does not
    matter which module the name came from or whether the branch holding it
    is ever taken.

    Blind, and both blind spots are the reason the runtime patches stay:
    a call reached through a variable (``fn = getattr(Path, "read_text");
    fn(p)``) and a read performed by a module ``preflight/`` merely CALLS.
    The runtime patches see both and are blind to the branch; this sees the
    branch and is blind to those.  Neither alone is the guard.
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id in _OUT_OF_SCOPE_CALLS:
            found.add(node.func.id)
        elif (isinstance(node.func, ast.Attribute)
                and node.func.attr in _OUT_OF_SCOPE_CALLS):
            found.add(node.func.attr)
    return found


class _GuardTripped(BaseException):
    """The pass left P-1: it read a file, built an operator or resolved a value.

    **A ``BaseException``, and that is the whole guard.**  :func:`preflight`
    wraps a raising check in ``except Exception`` -- and so does any
    ``except Exception`` or ``except AssertionError`` a CHECK might write.  A
    check that catches the guard and reports a re-worded finding of its own
    ("the beam file could not be read.") therefore evades every assertion made
    about the findings, however that assertion is phrased: measured, with
    these guards raising ``AssertionError``, a check reaching
    ``Path.read_text`` through a variable -- the static call ban's own
    documented blind spot -- and reporting it in its own words passed this
    whole module at exit 0.

    ``BaseException`` is unswallowable by any check whatever it reports, so
    the three branch guards below need no assertion about findings at all.
    That is what keeps them honest as Plan 3A lands: :data:`_BRANCH_DOCUMENTS`
    lights the branches Tasks 3-12's checks read, so a branch that lights a
    check lights its legitimate REFUSAL too -- ``a-typed-model-node`` earns
    A39's, by design -- and an assertion that the branch documents are refusal
    free would make every later task's correct check read as a scope violation
    here.
    """


#: The targets :func:`_forbid_the_filesystem` rebinds, in the spelling
#: ``monkeypatch.setattr`` takes.  ``pathlib.Path.open`` and ``os.open`` are
#: here and NOT in :data:`_FILESYSTEM_PATCHES`, because the verb they answer
#: for -- ``open`` -- already is: they are second targets for one banned name,
#: not a sixteenth verb.
_FILESYSTEM_TARGETS: tuple[str, ...] = (
    "builtins.open", "pathlib.Path.open", "os.open",
    *sorted(set(_FILESYSTEM_PATCHES.values()) - {"builtins.open"}),
)


def _forbid_the_filesystem(monkeypatch) -> None:
    """Take away every Python-level filesystem API in :data:`_FILESYSTEM_TARGETS`.

    The four an obvious draft carries (``builtins.open``,
    ``Path.open/read_text/read_bytes``) leave the directory walks wide open,
    and a beam is a DIRECTORY of files.  ``os.open`` is a second floor under
    ``builtins.open``; ``os.stat`` is the floor under ``Path.exists``,
    ``Path.is_file`` and ``os.path.exists`` -- measured, patching it catches
    all three -- which is the shape a check asking "is the beam file there?"
    would take.

    **The list is :data:`_FILESYSTEM_PATCHES`, not a literal here**, so the
    static ban and the runtime patches cannot name different routes.  A
    version that kept them apart shipped with ``lstat`` and ``rglob`` banned
    statically and patched by nothing, and with ``io.open_code``,
    ``sqlite3.connect``, ``os.getcwd``, ``os.statvfs`` and ``os.readlink``
    in neither half -- through which a reviewer's probe module read 15 671
    bytes and created a SQLite database on every document, with this whole
    module at exit 0.  Where a verb cannot be patched, it says so in
    :data:`_STATIC_ONLY_CALLS` with the measurement, which is a decision;
    absent from both lists is a hole.

    **Two of the patches are redundant on this interpreter, measured.**
    Deleting the ``Path.iterdir`` and ``Path.glob`` patches leaves every test
    in this module green, because CPython 3.12's ``Path.iterdir`` delegates to
    ``os.listdir`` and its ``glob`` to ``os.scandir``, both of which are
    patched too.  They are kept because that delegation is an implementation
    detail that has already been rewritten once between versions -- but the
    two anti-vacuity cases named for them prove the FLOOR patch works, not
    these two, and saying otherwise would be a claim no run defends.
    """
    def refuse_to_open(*args, **kwargs):
        raise _GuardTripped("the pre-flight pass touched the filesystem")

    for target in _FILESYSTEM_TARGETS:
        monkeypatch.setattr(target, refuse_to_open)


def _forbid_operators(monkeypatch) -> None:
    """Take away ``eqx.Module`` construction for the duration of a call.

    ``equinox``'s metaclass is where ``Cls(...)`` lands for every Module, and
    ``AbstractOperator`` is one -- measured, patching
    ``type(AbstractOperator).__call__`` catches ``NoiseOperator(sigma=1.0)``
    and leaves ``parse_path`` untouched.
    """
    def refuse_to_build(cls, *args, **kwargs):
        raise _GuardTripped(
            f"the pre-flight pass constructed {cls.__name__}")

    monkeypatch.setattr(type(AbstractOperator), "__call__", refuse_to_build)


#: ``(verb, a call that reaches it)`` -- one case per name the filesystem ban
#: carries, driven through the pass with the patches on.  Module level rather
#: than inline in the ``parametrize``, so that
#: ``test_every_banned_verb_has_a_case_that_trips_the_guard`` can read the
#: COVERAGE of the ban off it: a verb banned with no case here is a claim with
#: no measurement behind it, which is how ``lstat`` and ``rglob`` came to be
#: banned statically and patched by nothing.
#:
#: ``open`` has three cases because it has three targets -- ``builtins.open``,
#: ``Path.open`` and ``os.open`` are three distinct entry points to one verb,
#: and a draft that patched the first alone left the other two open.
_READING_CASES: tuple[tuple[str, Callable[[], object]], ...] = (
    ("open", lambda: open(__file__).close()),
    ("open", lambda: pathlib.Path(__file__).open().close()),
    ("open", lambda: os.close(os.open(__file__, os.O_RDONLY))),
    ("open_code", lambda: io.open_code(__file__).close()),
    ("read_text", lambda: pathlib.Path(__file__).read_text()),
    ("read_bytes", lambda: pathlib.Path(__file__).read_bytes()),
    ("iterdir", lambda: list(pathlib.Path(__file__).parent.iterdir())),
    ("glob", lambda: list(pathlib.Path(__file__).parent.glob("*.py"))),
    ("rglob", lambda: list(pathlib.Path(__file__).parent.rglob("*.py"))),
    ("walk", lambda: next(pathlib.Path(__file__).parent.walk())),
    ("listdir", lambda: os.listdir(str(_HERE))),
    ("scandir", lambda: list(os.scandir(str(_HERE)))),
    ("stat", lambda: os.stat(__file__)),
    ("statvfs", lambda: os.statvfs(str(_HERE))),
    ("connect", lambda: sqlite3.connect(":memory:").close()),
    ("exists", lambda: pathlib.Path(__file__).exists()),
    ("is_file", lambda: pathlib.Path(__file__).is_file()),
    ("is_dir", lambda: pathlib.Path(__file__).parent.is_dir()),
)
_READING_IDS = ["builtins-open", "path-open", "os-open", "io-open-code",
                "path-read-text", "path-read-bytes", "path-iterdir",
                "path-glob", "path-rglob", "path-walk", "os-listdir",
                "os-scandir", "os-stat", "os-statvfs", "sqlite3-connect",
                "path-exists", "path-is-file", "path-is-dir"]


#: ``(home module, name)`` for value resolution and the five builders
#: ``load_document`` calls.  ``build_runtime`` is here for the same reason it
#: is in :data:`_OUT_OF_SCOPE_NAMES`: a draft left it off and a check that
#: imported and called it passed every assertion in this module.
_OUT_OF_SCOPE_CALLABLES = (
    ("rheplicant.config.values", "resolve_value"),
    ("rheplicant.config.resources", "build_resources"),
    ("rheplicant.config.sections.compose", "build_model"),
    ("rheplicant.config.sections.observation", "build_observation"),
    ("rheplicant.config.sections.inference", "build_inference"),
    ("rheplicant.config.sections.runtime", "build_runtime"),
)


def _forbid_the_builders(monkeypatch) -> None:
    """Take away value resolution and the five builders -- EVERY reference.

    Patching the home module alone is not enough and the hole is not
    theoretical: ``rheplicant/config/__init__.py`` re-exports
    ``resolve_value``, so ``import rheplicant.config as cfg`` followed by
    ``cfg.resolve_value(...)`` reaches a SECOND reference the home-module
    patch never touches -- measured, a probe check calling it that way was
    seen 0 times by the old guard and 1 time by a patch on the re-export.

    So the walk is over ``sys.modules``: every module holding an attribute
    that IS the real function gets it replaced, which covers the re-export,
    any future alias, and the ``importlib.import_module(...).name`` route as
    well, because they all resolve to a module attribute at call time.

    What it still cannot see: a reference stashed under a DIFFERENT name
    before the patch (``_RESOLVE = resolve_value`` at module scope, or one
    held in a table).  That needs an evasion of
    :func:`_out_of_scope_imports` as well, which is why the two guards ship
    together.
    """
    def refuse_to_build(*args, **kwargs):
        raise _GuardTripped("the pre-flight pass left P-1")

    for home, name in _OUT_OF_SCOPE_CALLABLES:
        real = getattr(importlib.import_module(home), name)
        for module in list(sys.modules.values()):
            if module is None:
                continue
            try:
                held = getattr(module, name, None)
            except Exception:                       # a lazy module attribute
                continue
            if held is real:
                monkeypatch.setattr(module, name, refuse_to_build)


#: The documents the three runtime guards below are driven over, and the
#: reason there is a SET of them rather than one.
#:
#: A draft ran each guard on ``preflight_document()`` alone.  A check that
#: reads a file only on a branch the base document does not take -- and every
#: interesting one does, because a beam is only read when ``resources.beams``
#: is there -- was then invisible to all three, measured: a probe module
#: reading ``observation.from_file`` and walking ``resources.beams``
#: directories passed the whole module at exit 0.
#:
#: These patches light the branches Tasks 3-12's checks actually take.  It is
#: a covering set, not a proof: a branch nobody thought of is still a branch
#: nobody drives.  That is why the STATIC call guard exists beside these --
#: it reads the branch whether or not any document takes it.
_BRANCH_DOCUMENTS = (
    ("base", {}),
    ("observation-from-file",
     {"observation": {"from_file": {"format": "rhino_hdf5",
                                    "path": "obs.hd5f"}}}),
    ("resources-beam", {"resources": UNREADABLE_BEAM}),
    ("resources-absent", {"resources": None}),
    ("a-file-value-node",
     {"observation": {"freq": {"grid": {"file": "grid.npy"}, "unit": "MHz"}}}),
    ("a-ref-value-node",
     {"model": {"gain": {"gain": {"ref": "resources.arrays.flat"}}}}),
    ("a-python-resource",
     {"resources": {"arrays": {"made": {"python": "numpy.ones",
                                        "args": [8]}}}}),
    ("a-typed-model-node",
     {"model": {"bandpass": {"type": "NeuralOperator", "path": "net.eqx"}}}),
    ("inference-absent", {"inference": None}),
    ("an-npe-block", {"inference": {"npe": {"bank": {"seed": 1, "n": 8}}}}),
    ("several-runs",
     {"runs": [{"kind": "forward", "name": "a"},
               {"kind": "plan.estimate", "name": "b",
                "blocks": [{"names": ["g"]}]}]}),
    # The path §5's 0.05 s is thinnest on, and it was in NO row of this table
    # -- `several-runs` carries `plan.estimate` and nothing carried
    # `plan.sample`, so the branch that used to drag 43 modules of
    # `rheplicant.inference` into the first call was driven by neither the
    # cost test nor any of the three runtime scope guards.
    ("a-plan-sample-run",
     {"runs": [{"kind": "plan.sample", "name": "s",
                "seed": "runtime.seeds.a", "n_sweeps": 8, "warmup": 2,
                "blocks": [{"names": ["g"], "engine": "conjugate"}]}]}),
    ("a-variant-applied",
     {"variants": {"other": {"model": {"gain": {"gain": {"value": 2.0,
                                                         "unit": "dimensionless"}}}}}}),
)
_BRANCH_IDS = [name for name, _ in _BRANCH_DOCUMENTS]
_BRANCH_PATCHES = [patch for _, patch in _BRANCH_DOCUMENTS]

#: The branches that must stay in the set.  A FLOOR, in the shape
#: ``test_config_fixture_contract._BUILDER_FLOOR`` already uses here: the
#: guards are only as good as the documents they are driven over, so pruning
#: the list is a way of covering less while staying green, and a two-line edit
#: would otherwise do it silently.
_BRANCH_FLOOR = frozenset({
    "base", "observation-from-file", "resources-beam", "a-file-value-node",
    "a-ref-value-node", "a-python-resource", "a-typed-model-node",
    "several-runs", "a-plan-sample-run",
})


#: The child of :meth:`TestTheCostAndTheBoundary.
#: test_the_pass_raises_no_audit_event_that_reaches_the_os`.
#:
#: **The pass is run over every branch document TWICE and the hook is armed
#: only on the second lap.**  A check's deferred imports read ``.py`` files
#: through CPython's own opener, so a hook armed on the first lap reports the
#: import machinery -- ``fitting.py`` alone defers four.  Warming first is
#: what makes an ``open`` event mean "this check reads a file", and it is why
#: this guard is blind to a read that happens once and never again.  A
#: MODULE-SCOPE import is the other subprocess test's half.
_AUDIT_CHILD = '''
import sys

WATCHED = ("open", "sqlite3.connect", "os.listdir", "os.scandir",
           "pathlib.Path.rglob", "pathlib.Path.walk", "socket.connect",
           "subprocess.Popen", "os.mkdir", "os.remove", "os.rename")
SEEN = []
CONTROL = []
ARMED = [False]
LABEL = [""]


def hook(event, args):
    if ARMED[0] and event in WATCHED:
        (CONTROL if LABEL[0] == "control" else SEEN).append(
            LABEL[0] + ": " + event + " " + repr(args)[:160])


sys.addaudithook(hook)

from rheplicant.config.preflight import preflight
from tests.config.preflight_helpers import preflight_document

BRANCHES = __BRANCHES__
BUILT = [(name, preflight_document(**patch)) for name, patch in BRANCHES]
for _, document in BUILT:            # lap one: warm every deferred import
    preflight(document)
for name, document in BUILT:         # lap two: the measurement
    LABEL[0] = name
    ARMED[0] = True
    try:
        preflight(document)
    finally:
        ARMED[0] = False

LABEL[0] = "control"
ARMED[0] = True
try:
    open("/dev/null").close()
finally:
    ARMED[0] = False

for line in SEEN:
    print(line)
print("ANTI-VACUITY-OK" if CONTROL else "ANTI-VACUITY-BLIND")
'''


class TestTheStructuralSweepMoved:
    """Kills: a message reworded in the move; ``_structural`` running AFTER
    the registered checks; a copy of ``_sweep`` left behind in
    ``document.py`` (§2.2's "one name, one binding")."""

    def test_the_section_list_is_the_twelve_it_always_was(self):
        """The data moved too, so it is pinned against a literal rather than
        against itself.  Kills a section quietly dropped in the move -- after
        which that section stops being refused and nothing says so."""
        assert _SECTIONS == (
            "schema_version", "defaults", "plugins", "runtime", "observation",
            "resources", "model", "variants", "inference", "runs", "outputs",
            "campaign")

    @pytest.mark.parametrize(("patch", "expected"), [
        ({"observations": {}},
         "This document declares ['observations']; the sections are "
         "['schema_version', 'defaults', 'plugins', 'runtime', 'observation', "
         "'resources', 'model', 'variants', 'inference', 'runs', 'outputs', "
         "'campaign']."),
        ({"campaign": {}},
         "campaign: is reserved with capability 4 (streaming evidence, "
         "schema §8.2) and refused in v1."),
        ({"outputs": {}},
         "outputs: is not read by this layer yet -- it arrives with Plan 4 "
         "(outputs, provenance, the CLI)."),
        ({"defaults": {}},
         "defaults: is not read by this layer yet -- it arrives with Plan 4 "
         "(presets are YAML files, and the CLI is where YAML first comes off "
         "disk)."),
        ({"plugins": {}},
         "plugins: is not read by this layer yet -- it arrives with Plan 4 "
         "(plugin import belongs to the process entry point)."),
        ({"schema_version": 2},
         "schema_version: 1 is required (got 2); it is what lets a later "
         "loader read an older document on purpose rather than by luck."),
        ({"schema_version": True},
         "schema_version: 1 is required (got True); it is what lets a later "
         "loader read an older document on purpose rather than by luck."),
        ({"runtime": None},
         "This document is missing ['runtime']; schema_version, runtime, "
         "observation, model and runs are required."),
        ({"observation": None},
         "This document is missing ['observation']; schema_version, runtime, "
         "observation, model and runs are required."),
        ({"model": None},
         "This document is missing ['model']; schema_version, runtime, "
         "observation, model and runs are required."),
        ({"runs": None},
         "This document is missing ['runs']; schema_version, runtime, "
         "observation, model and runs are required."),
    ], ids=["unknown-section", "campaign", "outputs", "defaults", "plugins",
            "version-two", "version-true", "missing-runtime",
            "missing-observation", "missing-model", "missing-runs"])
    def test_every_message_survived_the_move_verbatim(self, patch, expected):
        """Equality, not ``match=``.  Five of these are pinned by
        ``test_config_document.py`` with one-word patterns (``"observations"``,
        ``"capability 4"``, ``"Plan 4"``, ``"schema_version"``, ``"model"``),
        all of which stay green through a complete rewording -- so the
        substring pins cannot be the guard that the move was faithful.

        All FOUR required sections are driven, not just ``model``: a
        ``_REQUIRED`` that lost ``runs`` in the move is invisible to a
        one-section case.
        """
        with pytest.raises(ConfigError) as caught:
            _structural(preflight_document(**patch))
        assert str(caught.value) == expected

    @pytest.mark.parametrize(("patch", "wins"), [
        ({"observations": {}, "schema_version": 2}, "This document declares"),
        ({"observations": {}, "campaign": {}}, "This document declares"),
        ({"observations": {}, "outputs": {}}, "This document declares"),
        ({"campaign": {}, "outputs": {}}, "campaign:"),
        ({"campaign": {}, "schema_version": 2}, "campaign:"),
        ({"campaign": {}, "model": None}, "campaign:"),
        ({"outputs": {}, "defaults": {}}, "outputs:"),
        ({"defaults": {}, "plugins": {}}, "defaults:"),
        ({"outputs": {}, "schema_version": 2}, "outputs:"),
        ({"plugins": {}, "model": None}, "plugins:"),
        ({"schema_version": 2, "model": None}, "schema_version:"),
    ], ids=["unknown-beats-version", "unknown-beats-campaign",
            "unknown-beats-deferred", "campaign-beats-deferred",
            "campaign-beats-version", "campaign-beats-missing",
            "outputs-beats-defaults", "defaults-beats-plugins",
            "deferred-beats-version", "deferred-beats-missing",
            "version-beats-missing"])
    def test_the_clause_order_survived_the_move_too(self, patch, wins):
        """A message pinned verbatim says nothing about WHICH message a
        document with two problems is shown, and every case in the test above
        carries exactly one.

        Three mutations survived without this, all exit 0: hoisting the
        ``schema_version`` clause above the unknown-section one, hoisting
        missing-required above everything, and reordering ``_NOT_YET`` (whose
        iteration order is what decides ``outputs:`` over ``defaults:``).
        Each changes the first sentence a user reads on a document with two
        problems, and the docstring's claim that ``_structural`` "IS
        ``document._sweep``, moved here whole" is false under any of them.
        """
        with pytest.raises(ConfigError) as caught:
            _structural(preflight_document(**patch))
        assert str(caught.value).startswith(wins)

    def test_structural_runs_before_any_registered_check(self, registry):
        """Kills ``_structural`` called after the loop, or not at all -- under
        which a check reading ``document["model"]`` on a document that has no
        ``model:`` raises ``KeyError`` and reaches the user as a crash."""
        ran = []

        @register("A2")
        def _watcher(document):
            ran.append(document)
            return ()

        with pytest.raises(ConfigError, match="is missing"):
            preflight(preflight_document(model=None))
        assert ran == []

    def test_document_no_longer_carries_a_sweep_of_its_own(self):
        """§2.2: one name, one binding.  Kills leaving ``_sweep`` behind as a
        second, now-unreachable copy that a later reader edits."""
        assert not hasattr(document_module, "_sweep")
        assert not hasattr(document_module, "_SECTIONS")
        assert not hasattr(document_module, "_NOT_YET")
        assert not hasattr(document_module, "_REQUIRED")


class TestTheRegistry:
    """Kills: an id that is not a §6 id reaching a user as ``(check _mine)``;
    a silent double registration; the run order becoming sorted rather than
    declared; a variadic registration that binds or validates only its first
    id; a §6 extractor that matches nothing and makes every id assertion
    vacuous."""

    def test_a_registered_id_binds_to_its_function(self, registry):
        @register("A2")
        def _one(document):
            return ()

        assert CHECKS["A2"] is _one

    def test_an_id_that_is_not_a_schema_id_is_refused(self, registry):
        with pytest.raises(ConfigError, match="not a schema §6 id"):
            register("_mine")(lambda document: ())

    def test_every_id_of_a_variadic_registration_is_validated(self, registry):
        """Kills validating ``checks[0]`` alone -- under which
        ``@register("A16", "A17", "A18", "_mine")`` ships a finding tagged
        ``(check _mine).``  The second assertion kills validating as it binds:
        a refusal that leaves half the ids claimed makes the next import of
        the same module report "registered twice" about a module that never
        finished."""
        with pytest.raises(ConfigError, match="not a schema §6 id"):
            register("A20", "_mine")(lambda document: ())
        assert dict(CHECKS) == {}

    def test_a_bare_register_with_no_id_is_refused(self, registry):
        """``@register`` without parentheses hands the FUNCTION in as an id,
        and ``@register()`` binds nothing at all and runs never.  Both are
        one-character slips a later task can make, and neither is an error
        without this."""
        with pytest.raises(ConfigError, match="takes one or more check ids"):
            register()(lambda document: ())
        with pytest.raises(ConfigError, match="@register"):
            register(lambda document: ())

    def test_one_id_named_twice_in_one_registration_is_refused(self, registry):
        with pytest.raises(ConfigError, match="names 'A20' twice"):
            register("A20", "A20")(lambda document: ())
        assert dict(CHECKS) == {}

    def test_a_second_registration_of_one_id_is_refused_not_asserted(
            self, registry):
        register("A2")(lambda document: ())
        with pytest.raises(ConfigError, match="registered twice"):
            register("A2")(lambda document: ())

    @pytest.mark.parametrize("pair", [("A23", "A21"), ("A21", "A23")],
                             ids=["clash-last", "clash-first"])
    def test_a_clash_on_any_id_refuses_and_binds_none_of_them(
            self, registry, pair):
        """Kills checking only ``checks[0]`` for a clash, kills checking only
        ``checks[-1]``, and kills a non-atomic bind.

        **Both orders, because one order closes one direction and leaves its
        twin open** -- measured, a version with ``("A23", "A21")`` alone let
        ``for check in checks[-1:]`` through, under which
        ``register("A21", "A23")`` silently rebinds ``A21`` to a second
        function and which of the two runs depends on import order.
        """
        first = register("A21")(lambda document: ())
        with pytest.raises(ConfigError, match="registered twice"):
            register(*pair)(lambda document: ())
        assert set(CHECKS) == {"A21"}
        assert CHECKS["A21"] is first

    def test_the_double_registration_refusal_survives_python_O(self):
        """§2.5's ledger item, at this registry.  Measured on the OTHER one
        before Task 1 fixed it: under ``-O`` the second registration won."""
        source = (
            "from rheplicant.config.preflight import CHECKS, register\n"
            # Cleared for the same reason the `registry` fixture clears: from
            # Task 3 onward `A2` is already bound at import and the FIRST
            # registration below would be the one that raised.
            "CHECKS.clear()\n"
            "register('A2')(lambda document: ())\n"
            "try:\n"
            "    register('A2')(lambda document: ())\n"
            "    print('SHADOWED')\n"
            "except Exception as error:\n"
            "    print(type(error).__name__)\n"
        )
        done = subprocess.run([sys.executable, "-O", "-c", source],
                              capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "ConfigError", done.stdout

    def test_insertion_order_is_run_order(self, registry):
        """§2.6 item 4 rests on this: A20 and A21 must be registered before
        A23, because the first refusal is the one a user reads and A23's would
        contradict A20's.  Kills ``for check in sorted(CHECKS)``, which looks
        tidier and reverses exactly that triple."""
        for check in ("A23", "A21", "A20"):
            register(check)(
                lambda document, check=check:
                    (refuse(check, "inference.parameters.g", f"{check}!"),))
        found = preflight(preflight_document()).refusals()
        assert [one.check for one in found] == ["A23", "A21", "A20"]

    def test_a_function_bound_to_several_ids_runs_exactly_once(self, registry):
        """§3.1: ``CHECKS`` is many-to-one -- ``_blocks`` is A16+A17+A18+A19,
        ``_prior_gates`` is A20+A21+A23, ``_counts`` is A24+A25, ``_decided``
        is A27+A28.  Kills a walk of ``CHECKS`` with no de-duplication by
        identity, under which every one of those runs once per id and the user
        reads one mistake two to four times.  The first assertion kills a
        ``register`` that bound ``checks[0]`` alone, under which the other two
        would pass unchanged."""
        calls = []

        @register("A20", "A21", "A23")
        def _three(document):
            calls.append(1)
            return (refuse("A20", "inference.parameters.g", "one."),)

        assert {slot: fn is _three for slot, fn in CHECKS.items()} == {
            "A20": True, "A21": True, "A23": True}
        document = preflight_document()
        found = preflight(document).refusals()
        assert calls == [1] * len(_enumeration(document).layers)
        assert [one.check for one in found] == ["A20"]

    def test_two_functions_sharing_a_name_are_not_one_function(self, registry):
        """The de-duplication is by IDENTITY and not by ``__name__``.  Two
        lambdas are both ``<lambda>``; a walk keyed on the name drops the
        second silently.  (``test_two_refusals_both_arrive`` below fails on
        the same mutation; this one says why in its title.)"""
        register("A2")(lambda document: (refuse("A2", "model", "first."),))
        register("A3")(lambda document: (refuse("A3", "model", "second."),))
        found = preflight(preflight_document()).refusals()
        assert [one.check for one in found] == ["A2", "A3"]

    def test_a_dotted_slot_is_accepted_and_a_private_name_is_not(self, registry):
        """§3.2 (a): three functions each decide part of A1, so they cannot
        all claim the slot ``"A1"``.  Kills a regex that forgot the suffix --
        under which Task 3's module raises at import and every
        ``import rheplicant.config`` in the repo fails."""
        register("A1.runs")(lambda document: ())
        register("A1.variants")(lambda document: ())
        assert set(CHECKS) == {"A1.runs", "A1.variants"}
        with pytest.raises(ConfigError, match="not a schema §6 id"):
            register("A1.")(lambda document: ())

    def test_a_dotted_slot_still_carries_the_bare_id_to_the_reader(
            self, registry):
        """§3.2 (a)'s other half: the SLOT may be dotted, the
        ``Finding.check`` never is.  ``Report.checks()`` is what a user greps
        and what every later task's ``ids(document)`` assertion reads."""
        register("A1.runs")(
            lambda document: (refuse("A1", "runs[0]", "one."),))
        assert preflight(preflight_document()).checks() == frozenset({"A1"})

    def test_every_registered_id_is_a_schema_6_id(self):
        """Vacuous today (``CHECKS`` is empty) and load-bearing from Task 3.

        §3.2(a): a registry key is a SLOT and may carry a dotted suffix, so
        what is validated is the part before the first ``.``.  A test that
        rejected the dotted form would turn red at Task 3 and read as Task 3's
        fault.  Its anti-vacuity partner is the next test.

        **BLOCKER 1 (Plan 3C fix round): this covered ONE registry of four
        until this fix, and it was the one where a wrong id could never
        arrive unseen.** ``binder``'s ``SLOT`` regex (``passes.py``) validates
        the SHAPE of an id, not the range -- measured, ``A99``, ``B77`` and
        ``C42`` all ``fullmatch`` it, though the refusal it raises for a
        malformed id claims the range is enforced (``A1..A52, B1..B9,
        C1..C19)``.  Only ``preflight.CHECKS`` (the pre-flight registry) was
        ever compared against the real id list here; ``inflight.AXIS_CHECKS``
        and ``inflight.BUILT_CHECKS`` were checked against the regex alone,
        and ``postflight.CHECKS`` -- a DISCOVERED registry, precisely where a
        wrong id arrives with nobody reading the module that registers it --
        was checked by nothing at all.  Measured before this fix: a
        ``@register("C16", "C42")`` planted in ``postflight/digitising.py``,
        or a ``@register_axes("C15", "C42")`` in ``inflight/noise_waves.py``,
        both left ``tests/config`` at ``EXIT=0`` (4988 / 4984 passed); a
        ``@register("C18.kind", "C42")`` planted in ``preflight/gated.py`` was
        the only one of the three that was killed.  Walking all four
        registries here is what makes this test the census §5 already claims
        it is.
        """
        from rheplicant.config.inflight import AXIS_CHECKS, BUILT_CHECKS
        from rheplicant.config.postflight import CHECKS as PRICED_CHECKS

        for name, registry in (("pre-flight", CHECKS),
                               ("axes", AXIS_CHECKS),
                               ("built", BUILT_CHECKS),
                               ("post-flight", PRICED_CHECKS)):
            bare = {slot.split(".", 1)[0] for slot in registry}
            assert bare <= set(_schema_ids()), (
                name, sorted(bare - set(_schema_ids())))

    def test_the_schema_extractor_still_finds_the_table(self):
        """The partner.  Kills the §6 heading being renamed, the table losing
        its pipe form, or the id regex drifting -- each of which would let the
        literal drift away from the spec unnoticed.

        SKIPS rather than fails when the spec is not in the tree:
        ``docs/superpowers/`` is gitignored (``.gitignore:52``), so a worktree
        and a ``git archive`` tree have no copy and a hard read would error
        with ``FileNotFoundError``, which says nothing true about the ids.
        Where the spec is absent the literal below is all that runs; where it
        is present, this test holds the two to each other."""
        found = _schema_ids()
        assert len(found) == 80, found[:5]
        assert len(set(found)) == len(found), "an id is declared twice"
        assert found[0] == "A1" and "A52" in found and "C17" in found
        from_spec = _schema_ids_from_the_spec()
        if from_spec is None:
            pytest.skip("schema §6 spec absent -- docs/superpowers/ is gitignored")
        assert from_spec == found


class TestEveryRefusalOfThisPassIsPinnedWHOLE:
    """The eight sentences this pass says about a CHECK rather than about a
    document, pinned by equality on their whole text.

    **Why this class exists, measured.**  Plan 3B's Task 1 turns this pass's
    machinery into ``config/passes.py``, parameterised by the phase word
    (``"pre-flight"``) and by the decorator's name (``"register"``), so that
    the in-flight passes share one implementation instead of forking a second
    one.  Every pin that stood between that extraction and a wrong word was a
    SUBSTRING beginning AFTER the word: ``assert "'A3' RAISED KeyError" in
    str(...)``, ``match="is not a document section"``, ``match="takes one or
    more check ids"``, ``match="names 'A20' twice"``.  Measured before this
    class was written: rewriting all six occurrences of ``pre-flight`` to
    ``in-flight`` in ``preflight/__init__.py`` left ``tests/config`` at
    **exit 0**.  So did rewriting ``@register`` to ``@register_axes`` in the
    two sentences that advise it.  A pass whose every refusal opened with the
    wrong word satisfied the whole suite.

    Eight and not six: the two that never carried the phase word carry the
    DECORATOR's name instead, and a reader told to write ``@register_axes``
    into a pre-flight module is misdirected exactly as badly.

    **What this class cannot see.**  It pins the text of a refusal, not that
    the refusal is REACHED -- the surrounding classes are what drive each
    branch -- and it says nothing about the in-flight passes' own wording,
    which is ``test_config_inflight.py``'s to pin.
    """

    def test_a_registration_with_no_id_at_all(self, registry):
        with pytest.raises(ConfigError) as caught:
            register()(lambda document: ())
        assert str(caught.value) == (
            "register() takes one or more check ids -- @register('A30'), or "
            "@register('A16', 'A17') when one function decides several. A "
            "registration with no id binds nothing, so the check it decorates "
            "never runs and nothing says so."
        )

    def test_an_id_that_is_not_a_string(self, registry):
        """``7`` rather than the bare ``@register``'s function, whose ``repr``
        carries an address and cannot be pinned by equality at all."""
        with pytest.raises(ConfigError) as caught:
            register(7)(lambda document: ())
        assert str(caught.value) == (
            "pre-flight check id 7 is not a string. @register is called with "
            "its ids -- @register('A30') -- and a bare @register hands the "
            "decorated function in as an id."
        )

    def test_an_id_that_is_not_a_slot(self, registry):
        with pytest.raises(ConfigError) as caught:
            register("_mine")(lambda document: ())
        assert str(caught.value) == (
            "pre-flight check id '_mine' is not a schema §6 id (A1..A52, "
            "B1..B9, C1..C19), optionally with a dotted suffix such as "
            "'A1.runs' when several functions each decide part of one check. "
            "The id is what a Finding carries and what a reader looks up; a "
            "private name here reaches the user as '(check _mine).'"
        )

    def test_one_id_named_twice_in_one_registration(self, registry):
        with pytest.raises(ConfigError) as caught:
            register("A20", "A20")(lambda document: ())
        assert str(caught.value) == (
            "this registration names 'A20' twice. The variadic form binds one "
            "function to several DIFFERENT ids -- @register('A16', 'A17') -- "
            "and a repeated id is a typo for one that is now claimed by "
            "nobody."
        )

    def test_an_id_a_second_function_claims(self, registry):
        """The two module names are this module's, twice, because both lambdas
        are defined here.  They are an interpolation rather than wording, and
        they are written out anyway: the sentence is only useful if it names
        the two modules truthfully, and an equality pin is what says it does.

        ``test_config_preflight`` and not ``tests.config.test_config_preflight``
        -- measured: pytest imports this file under its bare stem, so a literal
        written from the import path in the test header is wrong."""
        register("A2")(lambda document: ())
        with pytest.raises(ConfigError) as caught:
            register("A2")(lambda document: ())
        assert str(caught.value) == (
            "pre-flight check 'A2' is registered twice, by "
            "test_config_preflight and by test_config_preflight. A check id "
            "has one function, and which of the two would run depends on "
            "import order."
        )

    def test_a_check_that_raises(self, registry):
        @register("A3")
        def _bad(document):
            raise KeyError("noise")

        with pytest.raises(ConfigError) as caught:
            preflight(preflight_document())
        assert str(caught.value) == (
            "pre-flight check 'A3' RAISED KeyError: 'noise'. A check returns "
            "findings and raises nothing -- one that raises aborts the pass "
            "and hides every finding after it, which is the failure the "
            "collect-rather-than-raise design exists to prevent."
        )

    def test_a_where_that_is_not_a_document_path(self, registry):
        """``parse_path``'s own sentence is quoted INSIDE this one, so the pin
        holds the nesting too: a version that dropped the parenthesised cause
        would tell a reader their ``where`` is wrong and not why."""
        register("A2")(lambda document: (refuse("A2", "", "a sentence."),))
        with pytest.raises(ConfigError) as caught:
            preflight(preflight_document())
        assert str(caught.value) == (
            "pre-flight check 'A2' emitted where='', which is not a document "
            "path (Path '' is empty or padded with whitespace. A path is "
            "'head' or 'head.step.step', where head names a graph node and "
            "each step is an attribute, optionally with a non-negative "
            "index.). `where` is where the USER types, not where the code "
            "lives."
        )

    def test_a_where_whose_head_is_not_a_section(self, registry):
        register("A4")(lambda document: (refuse("A4", "beam", "reserved."),))
        with pytest.raises(ConfigError) as caught:
            preflight(preflight_document())
        assert str(caught.value) == (
            "pre-flight check 'A4' emitted where='beam', whose first segment "
            "'beam' is not a document section. The sections are "
            "['schema_version', 'defaults', 'plugins', 'runtime', "
            "'observation', 'resources', 'model', 'variants', 'inference', "
            "'runs', 'outputs', 'campaign']."
        )


class TestThePassCollects:
    """Kills: raising on the first finding; a check that raises truncating the
    pass in silence; a check whose return type is a generator being consumed
    twice or not at all."""

    def test_two_refusals_both_arrive(self, registry):
        register("A2")(lambda document: (refuse("A2", "model", "first."),))
        register("A3")(lambda document: (refuse("A3", "model", "second."),))
        found = preflight(preflight_document()).refusals()
        assert [one.message for one in found] == ["first.", "second."]

    def test_a_check_that_raises_fails_the_pass_loudly_and_names_itself(
            self, registry):
        """The §2.3 TRAP.  Kills a bare ``except: pass`` around the call --
        under which the raising check's findings AND every later check's
        vanish, and the document loads."""
        register("A2")(lambda document: (refuse("A2", "model", "first."),))

        @register("A3")
        def _bad(document):
            raise KeyError("noise")

        register("A4")(lambda document: (refuse("A4", "model", "third."),))
        with pytest.raises(ConfigError) as caught:
            preflight(preflight_document())
        assert "'A3' RAISED KeyError" in str(caught.value)
        # ...and it is not mistaken for a refusal about the document.
        assert "returns findings and raises nothing" in str(caught.value)

    def test_the_raising_check_is_named_by_its_own_slot(self, registry):
        """Kills naming whichever slot the loop happened to be on -- the
        first, or the last -- rather than the raiser's."""
        register("A2")(lambda document: ())

        @register("A16", "A17")
        def _bad(document):
            raise ValueError("nope")

        with pytest.raises(ConfigError, match="'A16' RAISED ValueError"):
            preflight(preflight_document())

    @pytest.mark.parametrize("make", [
        lambda one: (one,),
        lambda one: [one],
        lambda one: iter((one,)),
    ], ids=["tuple", "list", "generator"])
    def test_a_check_may_hand_back_any_iterable(self, registry, make):
        """``Check`` is typed ``-> Iterable[Finding]`` and Tasks 3-12 will
        write all three.  Kills ``findings.extend(fn(document))`` followed by
        a second pass over the same exhausted generator."""
        register("A2")(lambda document: make(refuse("A2", "model", "one.")))
        assert len(preflight(preflight_document()).refusals()) == 1

    def test_warnings_and_reports_are_collected_without_raising(self, registry):
        register("A41")(lambda document: (
            warn("A41", "resources.arrays.flat", "shadowed."),
            report("A41", "resources.arrays.flat", "noted."),
        ))
        held = preflight(preflight_document())
        assert held.refusals() == ()
        assert len(held.warnings()) == 1
        assert held.checks() == frozenset({"A41"})

    def test_the_base_document_earns_no_finding_of_its_own(self):
        """The REAL registry -- deliberately no ``registry`` fixture.

        §3.2(b) offers ``ids(document)`` as "what 'and nothing else' reads".
        A base that is itself a finding makes that accessor never return
        empty, with nothing recording it, and every later task's
        ``ids(...) == {"A30"}`` inherits the extra id.  Vacuous at Task 2
        (``CHECKS`` is empty) and load-bearing from Task 3 on.  Measured
        before ``preflight_helpers``' ``observed.twin: full`` landed: the
        untouched base would earn A42 at ``inference.observed.primary``.
        """
        assert preflight(preflight_document()).findings == ()


class TestTheOneDocumentBuilder:
    """``preflight_helpers``' API, which Tasks 3-12 all build on.

    A fixture layer that cannot discriminate disarms every test above it at
    once -- 2D measured 86 of 90 going blind that way -- so the four
    properties every later task assumes are pinned here rather than trusted.
    """

    def test_a_mapping_patch_merges_into_the_section(self):
        """§3.2(b): ONE LEVEL DEEP.  Kills a merge that replaces the section --
        under which ``preflight_document(model={"gian": {}})`` hands every
        model check a document with one node in it, and a check that finds
        nothing has not looked."""
        model = preflight_document(model={"gian": {}})["model"]
        assert "gian" in model
        assert set(BASE_MODEL) <= set(model)
        assert set(BASE_MODEL) == {"global_signal", "uniform_sky", "gain",
                                   "noise"}

    def test_none_removes_a_section_and_a_non_mapping_replaces_it(self):
        """The two other patch forms.  ``None`` is how a check that fires on
        an ABSENT section is reached; the replacement form is how ``runs=[...]``
        gets a list where a merge makes no sense."""
        assert "inference" not in preflight_document(inference=None)
        assert preflight_document(runs=[{"kind": "forward", "name": "a"}])[
            "runs"] == [{"kind": "forward", "name": "a"}]

    def test_each_call_hands_back_a_fresh_document(self):
        """Kills a module-level base handed out by reference: one test's patch
        would then travel into every later test in the session, and the
        failures land nowhere near the cause."""
        first = preflight_document()
        first["model"]["gian"] = {}
        first["inference"]["parameters"]["g"]["init"] = 99.0
        second = preflight_document()
        assert "gian" not in second["model"]
        assert second["inference"]["parameters"]["g"]["init"] == 1.0

    def test_only_refuses_a_check_that_fired_more_or_less_than_once(
            self, registry):
        """§3.2(b)'s reason for :func:`only` existing.  Kills it becoming
        ``found[0]``: a check that fires TWICE on one document -- a loop over
        nodes that forgot to ``break`` -- is a real defect that no ``in``
        assertion can see, and a plain ``[0]`` passes through it."""
        register("A2")(lambda document: (
            refuse("A2", "model.gain", "once."),
            refuse("A2", "model.noise", "twice."),
        ))
        doc = preflight_document()
        with pytest.raises(AssertionError, match="produced 2 findings"):
            only(doc, "A2")
        with pytest.raises(AssertionError, match="produced 0 findings"):
            only(doc, "A3")

    def test_the_accessors_read_the_pass_and_not_each_other(self, registry):
        """``findings`` is everything in run order, ``refusals`` the subset
        that stops the run, ``ids`` the bare check ids, ``only`` the one
        finding.  Kills ``refusals`` returning every finding -- under which a
        later task's warning-shaped check would read as a refusal."""
        register("A2")(lambda document: (refuse("A2", "model", "no."),))
        register("A41")(lambda document: (warn("A41", "model", "hmm."),))
        doc = preflight_document()
        assert [one.message for one in findings(doc)] == ["no.", "hmm."]
        assert [one.message for one in refusals(doc)] == ["no."]
        assert ids(doc) == frozenset({"A2", "A41"})
        assert only(doc, "A41").message == "hmm."


class TestTheWhereShape:
    """Kills: a check putting a ``src/`` path, a bare node name, or an empty
    string in front of a user.  §3.1's second non-negotiable rule."""

    @pytest.mark.parametrize("bad", [
        "src/rheplicant/config/sections/model.py",
        "compose.py:262",
        "",
    ], ids=["source-path", "source-and-line", "empty"])
    def test_a_where_that_is_not_a_document_path_is_refused(self, registry, bad):
        register("A2")(lambda document: (refuse("A2", bad, "a sentence."),))
        with pytest.raises(ConfigError) as caught:
            preflight(preflight_document())
        assert "'A2' emitted where=" in str(caught.value)

    @pytest.mark.parametrize("bad", [
        "beam", "models.gain", "run", "variant", "inferences.parameters.g",
        "Model.noise",
    ], ids=["a-node-id", "a-plural-section", "a-truncated-section",
            "a-singular-section", "a-pluralised-section", "wrong-case"])
    def test_a_where_whose_head_is_not_a_section_is_refused(
            self, registry, bad):
        """``beam`` is a real NODE and not a section; a check that wrote the
        node id alone would send the reader looking for a top-level key that
        does not exist.

        **The five NEAR MISSES are the discriminating cases**, and ``beam``
        alone is not.  Two mutations survived a version with only ``beam``:
        admitting any head of five characters or more (``models``,
        ``variant``, ``inferences`` all pass that), and testing membership
        against the JOINED section names as a substring (``run`` is inside
        ``runtime``, ``variant`` inside ``variants``).  Both send a reader to
        a top-level key that does not exist, off a one-character slip.
        """
        register("A4")(lambda document: (refuse("A4", bad, "reserved."),))
        with pytest.raises(ConfigError, match="is not a document section"):
            preflight(preflight_document())

    @pytest.mark.parametrize("position", [0, 1, 2],
                             ids=["first", "middle", "last"])
    def test_every_finding_is_checked_and_not_just_the_first(
            self, registry, position):
        """Kills ``_check_where(check, found[0])`` and its twin
        ``found[-1]``, both of which survived a version of this class where
        every case registered a check returning exactly ONE finding.

        A check that walks nodes returns one finding per node, so under either
        mutation every finding after the first (or before the last) can carry
        a ``src/`` path in front of a user.
        """
        wheres = ["model.gain", "model.gain", "model.gain"]
        wheres[position] = "compose.py:262"
        register("A4")(lambda document: tuple(
            refuse("A4", one, "a sentence.") for one in wheres))
        with pytest.raises(ConfigError, match="'A4' emitted where="):
            preflight(preflight_document())

    @pytest.mark.parametrize("make", [refuse, warn, report],
                             ids=["refuse", "warn", "report"])
    def test_a_bad_where_is_refused_at_every_severity(self, registry, make):
        """Kills gating the check on ``severity == "refuse"``, which survived
        a version where every case here used :func:`refuse`.  **Task 12's A41
        is warning-shaped**, so this is a twin the plan walks into rather than
        a hypothetical one."""
        register("A41")(lambda document: (make("A41", "beam", "a sentence."),))
        with pytest.raises(ConfigError, match="is not a document section"):
            preflight(preflight_document())

    def test_the_check_that_emitted_a_bad_where_is_named(self, registry):
        """``_check_where`` takes the SLOT as well as the finding: a refusal
        that does not name the culprit leaves the reader a message with no
        author.  Kills passing ``finding.check`` instead -- which is the bare
        id and is empty for a finding that carries none."""
        register("A1.runs")(lambda document: (refuse("", "beam", "no."),))
        with pytest.raises(ConfigError, match=r"check 'A1\.runs' emitted"):
            preflight(preflight_document())

    @pytest.mark.parametrize("good", [
        "model", "model.noise", "runs[2].blocks[0]", "inference.parameters.g",
        "observation.freq.grid", "resources.beams.horn", "runtime.seeds",
        "variants.unity_gain",
    ], ids=["section", "node", "subscripted", "latent", "grid", "resource",
            "seeds", "variant"])
    def test_every_real_document_path_passes(self, registry, good):
        """The other direction, and it is not decoration: a head check written
        as ``where.split(".")[0]`` rejects ``runs[2]`` -- measured,
        ``parse_path`` is what handles the subscript -- and a guard that
        refuses valid paths would be worked around rather than fixed."""
        register("A2")(lambda document: (refuse("A2", good, "a sentence."),))
        assert len(preflight(preflight_document()).refusals()) == 1


class TestThePhaseGuard:
    """The measurement this whole task exists for.

    At ``be2027b``, on a document carrying an unreadable beam, the beam's
    refusal beat an unknown ``model:`` node, a junction given an operator, a
    ``flagging`` with no ``type:``, a ``NeuralOperator`` type and a
    ``scope: per_epoch`` latent -- five for five.  Every assertion below is
    the same experiment with the hook in place.

    Kills, all four one-line edits: the ``preflight`` call moved below
    ``build_resources``; the call deleted; ``raise_if_refused`` swapped for
    ``emit_warnings``; and the pass run on the RAW document rather than the
    variant-applied one.

    These four cases are SYNTHETIC -- the ids and wording mirror what Tasks 3,
    4, 7 and 11 will emit, but the functions are this test's, because
    ``CHECKS`` is empty when this module lands.  They are the hook's position
    and no real check's phase; the definition of done's ten real assertions
    are Tasks 3-12's, one each.
    """

    @pytest.mark.parametrize(("check", "where", "message"), [
        ("A2", "model.gian",
         "model: 'gian' is not a node of graph 'single-antenna' (check A2)."),
        ("A39", "model.bandpass",
         "model.bandpass.type: NeuralOperator is deferred with capability 3 "
         "(neural surrogates) -- schema §8.1 (check A39)."),
        ("A30", "inference.twin",
         "inference.twin: the fit twin keeps a stochastic stage (check A30)."),
        ("A16", "runs[0].blocks[0]",
         "runs[0].blocks[0]: 'g' appears in no block (check A16)."),
    ], ids=["model-node", "capability-key", "fit-twin", "block"])
    def test_a_registered_refusal_precedes_the_beam_read(
            self, registry, check, where, message):
        """Four shapes of ``where``, because the hook is one call and a test on
        one shape would not notice a guard that special-cased ``model.``.

        Equality and not ``in``: it says the beam's own refusal is not what
        came back without a second assertion that would only look like one.
        """
        register(check)(lambda document: (refuse(check, where, message),))
        with pytest.raises(ConfigError) as caught:
            load_document(preflight_document(resources=UNREADABLE_BEAM))
        assert str(caught.value) == message

    @pytest.mark.parametrize(("patch", "fragment"), [
        ({"campaign": {}}, "capability 4"),
        ({"observations": {}}, "This document declares ['observations']"),
    ], ids=["campaign", "unknown-section"])
    def test_the_structural_half_precedes_it_too(self, patch, fragment):
        """These two already won at ``be2027b`` -- ``_sweep`` was the one thing
        before ``build_resources``.  Kills the move putting ``_structural``
        behind the hook, or behind ``build_runtime``."""
        with pytest.raises(ConfigError, match=re.escape(fragment)):
            load_document(preflight_document(resources=UNREADABLE_BEAM,
                                             **patch))

    def test_the_beam_still_refuses_when_nothing_else_does(self):
        """ANTI-VACUITY, and it is the assertion the whole class rests on.

        If ``UNREADABLE_BEAM`` ever became readable -- a file appearing at that
        name, a default filled in, ``build_resources`` learning to defer --
        every test above would pass with no pre-flight pass at all, because
        there would be nothing for the refusal to beat.  Measured when this was
        written: 0.115 s to reach this refusal, and it is ``build_resources``
        that raises it.
        """
        with pytest.raises(ConfigError, match="No file at 'no_such_beam.npy'"):
            load_document(preflight_document(resources=UNREADABLE_BEAM))

    def test_a_warning_is_emitted_and_a_refusal_still_wins(self, registry):
        """Kills the hook calling ``emit_warnings`` BEFORE
        ``raise_if_refused`` -- under which a document about to be refused
        first sprays warnings about lines the user is on their way to
        change."""
        register("A41")(lambda document: (
            warn("A41", "resources.arrays.flat", "shadowed."),))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            load_document(preflight_document())
        assert [str(one.message) for one in caught].count("shadowed.") == 1
        assert caught[0].category is ConfigWarning

        register("A2")(lambda document: (refuse("A2", "model", "no."),))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with pytest.raises(ConfigError, match=r"^no\.$"):
                load_document(preflight_document())
        assert not [one for one in caught if one.category is ConfigWarning]

    def test_the_pass_sees_the_variant_applied_document(self, registry):
        """§2.1: P-1 runs on the variant-applied document and nothing else.
        Kills hooking ``preflight`` above ``apply_variant`` -- under which a
        variant could smuggle a violation past every check, which is exactly
        the hole ``test_config_document.py:107`` closed for the sweep."""
        seen = []

        @register("A2")
        def _watch(document):
            seen.append(float(document["model"]["gain"]["gain"]["value"]))
            return ()

        document = preflight_document()
        load_document(document, variant="unity_gain")
        assert seen == [1.0] * len(_enumeration(document).layers)
        assert 1.1 not in seen

    def test_run_document_inherits_the_hook_and_adds_none_of_its_own(
            self, registry):
        """One attributed fan, each layer's own content, however many runs.

        Re-pinned at Task 10: the per-variant memoised ``load_document`` loop
        is gone, and with it the pass running once per configured variant.
        The orchestration runs the text pass ONCE over the canonical layers
        before any build, so the count is the layer count -- and the values
        are the layers' OWN, in canonical order, which is what keeps "the
        pass ran on the unpatched document" dead: the variant layer's content
        is the second entry, and the base's own value cannot stand in for it.

        Kills both directions, as before: a hook that never fires makes it
        nought, and one that re-reads a single document makes the two entries
        equal.
        """
        seen = []

        @register("A2")
        def _watch(document):
            seen.append(float(document["model"]["gain"]["gain"]["value"]))
            return ()

        doc = preflight_document(runs=[
            {"kind": "forward", "name": "a"},
            {"kind": "forward", "name": "b"},
            {"kind": "forward", "name": "c", "variant": "unity_gain"},
        ])
        run_document(doc)
        assert seen == [1.1, 1.0]


class TestTheCostAndTheBoundary:
    """§0.1's contract and §2.4's, mechanically rather than in prose.

    Five guards, each with an anti-vacuity partner, and **what the five
    together cannot see is written down** -- a verification method with the
    same blind spot as the code is this project's recorded failure mode, and a
    draft of this class had three such spots at once.

    **The runtime guards run over a SET of documents** (:data:`_BRANCH_
    DOCUMENTS`), not over the base alone.  A draft ran each on
    ``preflight_document()``, which lights none of the branches an interesting
    check takes; measured, a probe module reading ``observation.from_file``
    and walking ``resources.beams`` directories passed the whole module at
    exit 0.

    * **the filesystem** -- every Python-level API in
      :data:`_FILESYSTEM_TARGETS` taken away, and that list is DERIVED from
      the static ban rather than written beside it
      (:func:`_forbid_the_filesystem` says why).  Measured on this build with
      all of them patched, ``numpy.load``, ``numpy.fromfile`` and
      ``numpy.loadtxt`` are **all caught** -- every one of them goes through
      ``builtins.open``.
    * **the audit hook** -- a subprocess runs the pass with
      ``sys.addaudithook`` armed and fails on ``open``, ``sqlite3.connect``,
      ``os.listdir``, ``os.scandir`` or ``pathlib.Path.rglob``.  This is the
      half that sees a read performed ENTIRELY INSIDE A C EXTENSION, which no
      rebinding of a Python name can.

    **What the filesystem halves cannot see, restated because the sentence
    that stood here was false.**  It said the one C-extension read that
    escapes "is ``h5py.File``", stopped by the no-import rule -- but neither
    ``io`` nor ``sqlite3`` was on any list, and a reviewer's probe module
    walked out through ``io.open_code`` and ``sqlite3.connect`` with this
    whole module at exit 0.  The true statement is a pair of complementary
    blind spots:

    * the PATCHES are blind to any opener that does not go through a name
      they rebind -- an arbitrary C extension, and a third-party one nobody
      here has enumerated.  The audit hook sees those, because CPython raises
      ``open`` from ``_io`` itself.
    * the AUDIT HOOK is blind to metadata-only syscalls: measured on this
      build, ``os.stat``, ``os.lstat``, ``os.statvfs``, ``os.readlink``,
      ``os.getcwd`` and ``Path.exists`` raise **no audit event at all**.  The
      patches see ``stat``, ``statvfs`` and the three ``Path`` predicates; on
      ``lstat``, ``readlink`` and ``getcwd`` NEITHER runtime half can see,
      because ``coverage.py`` runs those three itself on every newly traced
      file and patching them fails the suite instead of the pass.  Those
      three are closed by the STATIC ban alone, which is branch-independent
      and reads them whether a document takes the branch or not; the
      reviewer's probe used all three and the static ban is what caught it.
    * the AUDIT HOOK is blind, too, to a read that happens only on the FIRST
      call, because it warms every deferred import first.
    * BOTH are blind to a C library that opens a file without CPython's own
      opener: measured, ``h5py.File`` on a real ``.h5`` raises no ``open``
      event and touches no patched name.  ``h5py`` is on §0's no-import list,
      which is what stops it -- so that claim is still true, it was just
      never the only one.
    * **operator construction** -- ``equinox``'s metaclass ``__call__``, so
      every ``eqx.Module`` built anywhere in the process during the pass
      trips.  Blind to an operator built by C code (there is none) and, by
      design, NOT to class introspection, which §2.4 puts in scope.
    * **value-node resolution and the five builders** -- ``resolve_value``,
      ``build_resources``, ``build_model``, ``build_observation``,
      ``build_inference`` and ``build_runtime``, patched in EVERY module that
      holds a reference to one rather than in the home module alone (see
      :func:`_forbid_the_builders` for the re-export that made the narrow
      version useless).  Blind to a reference stashed under another name
      before the patch, which needs an evasion of the import ban as well.
    * **the static bans** -- no out-of-scope import
      (:func:`_out_of_scope_imports`, including the umbrella package) and no
      out-of-scope call (:func:`_out_of_scope_calls`).  Branch-independent,
      and blind to a call reached through a variable and to a read performed
      by a module ``preflight/`` merely calls -- which the runtime patches
      see.  The two halves have disjoint blind spots and neither is the
      guard on its own.
    * **the optional dependencies** -- a subprocess import, which is
      transitive and sees module scope only.

    What none of them sees: a check that RE-IMPLEMENTS value resolution
    inline rather than calling it (§2.5's "do not re-implement" failure).

    "Runs a forward pass" needs a twin, and a twin is an ``eqx.Module``, so the
    operator guard covers it; a check cannot be handed an already-built twin
    because ``preflight``'s only argument is a ``Mapping``.
    """

    def test_the_branch_set_cannot_be_pruned_to_cover_less(self):
        """The three runtime guards are driven over :data:`_BRANCH_DOCUMENTS`,
        so deleting a row covers less and stays green -- measured, shrinking
        the set to the base alone is a two-line edit and nothing else in this
        module notices.

        Also: a row whose patch changes nothing is a row that adds no branch,
        which is the other way to look plentiful while covering one document.
        """
        assert set(_BRANCH_IDS) == {name for name, _ in _BRANCH_DOCUMENTS}
        assert len(_BRANCH_PATCHES) == len(_BRANCH_DOCUMENTS)
        assert _BRANCH_FLOOR <= set(_BRANCH_IDS), (
            f"{sorted(_BRANCH_FLOOR - set(_BRANCH_IDS))} left the branch set. "
            "Each names a shape a Plan 3A check reads; a guard that never "
            "sees it cannot see a check that reads a file only there."
        )
        base = preflight_document()
        for name, patch in _BRANCH_DOCUMENTS:
            built = preflight_document(**patch)
            if name == "base":
                assert built == base
            else:
                assert built != base, f"{name} patches nothing"

    def test_the_pass_costs_less_than_a_twentieth_of_a_second(self):
        """§0.1's number, against ``load_document``'s measured 1.536 s on a toy
        beam.

        The threshold is the plan's contract, not the measurement: with
        ``CHECKS`` empty the pass is microseconds, and the headroom is what
        keeps this from flaking on a loaded box.

        **What 0.05 s does and does not kill, measured rather than assumed.**
        It kills a check that reads and analyses a beam (§2.7: 1.397 s, and a
        real CST directory is worse) and a check that loads a document from
        cold.  It does **not** kill a check that CONSTRUCTS an operator --
        ``build_model`` on this document is about a millisecond on a warm
        process -- nor one that opens a small file.  Those two halves are
        closed by the guards below, not by this number.

        **And it does not kill an IMPORT, which is what made this test the
        weaker half of a pair.**  The warm-up line below is not incidental:
        it hides the cost of every deferred import a check makes, and one of
        them was 43 modules and 21 ms.  ``TestTheColdCostOnARealDocument`` is
        the half that sees it, and this one keeps the per-call number on the
        document the plan is written around.
        """
        doc = preflight_document()
        preflight(doc)                       # warm the import graph
        start = time.perf_counter()
        preflight(doc)
        assert time.perf_counter() - start < 0.05

    @pytest.mark.parametrize("patch", _BRANCH_PATCHES, ids=_BRANCH_IDS)
    def test_the_pass_touches_no_file(self, monkeypatch, patch):
        """§2.4's boundary, enforced rather than described.

        **Once per branch document, and that is the fix for a measured hole.**
        A version of this driving ``preflight_document()`` alone saw nothing
        when a probe module read ``observation.from_file`` and walked
        ``resources.beams`` directories -- because the base document declares
        neither, so the branch never ran.  The documents in
        :data:`_BRANCH_DOCUMENTS` light the branches Tasks 3-12 will read.

        The document is built and the patches are undone before the assertion,
        so a failure here is reported by a pytest that can still read its own
        source.
        """
        doc = preflight_document(**patch)
        _forbid_the_filesystem(monkeypatch)
        try:
            # The assertion is that this returns.  A guard fires as
            # `_GuardTripped`, a BaseException no check can swallow and no
            # findings assertion has to look for -- see that class.
            preflight(doc)
        finally:
            monkeypatch.undo()

    @pytest.mark.parametrize(("verb", "read"), _READING_CASES,
                             ids=_READING_IDS)
    def test_that_guard_can_still_see_each_way_of_reading(
            self, registry, monkeypatch, verb, read):
        """ANTI-VACUITY, one case per banned verb.  Without it, a target that
        stopped being the one the code reaches -- ``open`` moving inside a C
        extension, ``Path.read_text`` bypassed, a new ``os`` spelling -- makes
        the guard green forever, and a four-patch draft would pass the four it
        happens to name while the directory, metadata and existence routes
        ran wide open.

        ``verb`` is carried so that
        :meth:`test_every_banned_verb_has_a_case_that_trips_the_guard` can
        assert this list COVERS the ban.  A verb banned with no case here is
        a claim with no measurement behind it, which is exactly how ``lstat``
        and ``rglob`` came to be banned and patched by nothing.
        """
        @register("A2")
        def _reader(document):
            read()
            return ()

        doc = preflight_document()
        _forbid_the_filesystem(monkeypatch)
        try:
            with pytest.raises(_GuardTripped):
                preflight(doc)
        finally:
            monkeypatch.undo()

    def test_every_banned_verb_has_a_case_that_trips_the_guard(self):
        """The anti-vacuity list is only as good as its coverage of the ban.

        Kills a verb added to :data:`_FILESYSTEM_PATCHES` or
        :data:`_FLOORED_UNDER` with no case above -- under which the ban reads
        as wider than the measurement behind it, which is the shape this whole
        class exists to refuse -- and a case driven for a verb nothing bans.

        :data:`_STATIC_ONLY_CALLS` is excluded and says why in its own
        comment: those verbs come from modules ``preflight/`` may not import
        at all, so there is no call to drive.
        """
        cases = {verb for verb, _ in _READING_CASES}
        wanted = frozenset(_FILESYSTEM_PATCHES) | frozenset(_FLOORED_UNDER)
        assert wanted <= cases, (
            f"{sorted(wanted - cases)} are banned and no case above drives "
            "them, so nothing measures that the ban is enforced."
        )
        assert cases <= wanted, (
            f"{sorted(cases - wanted)} are driven above and banned by "
            "nothing."
        )
        assert len(_READING_IDS) == len(_READING_CASES)

    def test_a_check_cannot_swallow_a_guard_and_report_it_in_its_own_words(
            self, registry, monkeypatch):
        """ANTI-VACUITY for :class:`_GuardTripped`'s whole reason to exist.

        The route this closes is a check that CATCHES the guard rather than
        letting it out, and reports a finding of its own -- under which the
        pass returns normally and no assertion about findings can name what
        went wrong, because the check chose the words.  Both halves of the
        evasion are here: `Path.read_text` reached through a variable, which
        is the static call ban's own documented blind spot, and an `except
        Exception` that would have caught an `AssertionError`.

        Measured: with the guards raising `AssertionError` this check passed
        the whole module at exit 0.  A `BaseException` is not catchable by
        `except Exception`, so it escapes the check, escapes `preflight`'s own
        wrapper, and lands here."""
        @register("A2")
        def _swallower(document):
            reader = pathlib.Path.read_text          # not a Call node
            try:
                reader(pathlib.Path(__file__))
            except Exception:
                return (refuse("A2", "model",
                               "the beam file could not be read."),)
            return ()

        doc = preflight_document()
        _forbid_the_filesystem(monkeypatch)
        try:
            with pytest.raises(_GuardTripped):
                preflight(doc)
        finally:
            monkeypatch.undo()

    @pytest.mark.parametrize("patch", _BRANCH_PATCHES, ids=_BRANCH_IDS)
    def test_the_pass_builds_no_operator(self, monkeypatch, patch):
        """§2.4 again, on the half no timing can reach: ``build_model`` on this
        document is about a millisecond, so the 0.05 s budget above says
        nothing about it.  ``equinox``'s metaclass ``__call__`` is where every
        ``eqx.Module`` -- and so every ``AbstractOperator`` -- is constructed,
        measured, so taking it away for the duration of the pass is the whole
        check.  Once per branch document, for the reason above."""
        doc = preflight_document(**patch)
        _forbid_operators(monkeypatch)
        try:
            preflight(doc)          # returns, or `_GuardTripped` escapes
        finally:
            monkeypatch.undo()

    def test_that_guard_can_still_see_an_operator_being_built(
            self, registry, monkeypatch):
        """ANTI-VACUITY: measured, a check constructing a ``NoiseOperator``
        trips it.  Kills the metaclass moving, or ``AbstractOperator`` ceasing
        to be an ``eqx.Module``, either of which makes the guard above green
        for every possible check."""
        @register("A2")
        def _builder(document):
            NoiseOperator(sigma=1.0)
            return ()

        doc = preflight_document()
        _forbid_operators(monkeypatch)
        try:
            with pytest.raises(_GuardTripped):
                preflight(doc)
        finally:
            monkeypatch.undo()

    @pytest.mark.parametrize("patch", _BRANCH_PATCHES, ids=_BRANCH_IDS)
    def test_the_pass_resolves_no_value_node_and_builds_no_section(
            self, monkeypatch, patch):
        """§2.4's TRAP: ``observation.freq.grid`` may be ``{value: [...]}`` or
        ``{file: ...}`` or ``{ref: ...}``, and a check that wants the resolved
        number has left P-1.  ``resolve_value`` and the five builders are the
        entry points out; taking them away -- in every module that holds one --
        is what says the pass never reached one.  Once per branch document, and
        two of those documents carry exactly the ``{file:}`` and ``{ref:}``
        grids the TRAP is about."""
        doc = preflight_document(**patch)
        _forbid_the_builders(monkeypatch)
        try:
            preflight(doc)          # returns, or `_GuardTripped` escapes
        finally:
            monkeypatch.undo()

    @pytest.mark.parametrize("reach", [
        lambda: values_module.resolve_value(1.0, None),
        lambda: config_package.resolve_value(1.0, None),
        lambda: importlib.import_module(
            "rheplicant.config.values").resolve_value(1.0, None),
        lambda: runtime_module.build_runtime({"seed": 1}),
        lambda: config_package.build_resources({}, None),
    ], ids=["the-home-module", "the-package-re-export", "via-importlib",
            "build-runtime", "build-resources-re-exported"])
    def test_that_guard_can_still_see_a_builder_reached_any_way(
            self, registry, monkeypatch, reach):
        """ANTI-VACUITY, one case per route out.

        The second case is the measured hole: ``rheplicant/config/__init__.py``
        re-exports ``resolve_value``, and a guard patching only
        ``rheplicant.config.values`` saw a probe call it **0 times** through
        ``cfg.resolve_value(...)`` while a patch on the re-export saw it once.
        The fourth is the other one: ``build_runtime`` was absent from the
        guard's target list and from the import ban, so a check importing and
        calling it passed the whole module.

        ``resolve_value(1.0, None)`` is ``values.py:89``'s real two-argument
        signature, so the call is one a check could actually write.
        """
        @register("A2")
        def _resolver(document):
            reach()
            return ()

        doc = preflight_document()
        _forbid_the_builders(monkeypatch)
        try:
            with pytest.raises(_GuardTripped):
                preflight(doc)
        finally:
            monkeypatch.undo()

    def test_the_two_static_guards_still_walk_the_runner_they_were_written_for(
            self):
        """ANTI-VACUITY for the walk itself, and it is a measured hole.

        Both static guards below iterate :func:`_preflight_sources`.  Before
        Plan 3B they iterated ``_PREFLIGHT_DIR.glob("*.py")`` directly, and
        ``sweep`` and ``_check_where`` -- the two functions that run every
        check and validate every ``where`` -- then moved into
        ``config/passes.py``, which that glob does not reach.  Neither guard
        would have gone red; they would simply have stopped looking at the
        code they exist to look at.

        So this asserts the two things a shrinking walk breaks: the runner is
        IN the walk, and the walk is still a GLOB over ``preflight/`` rather
        than a list somebody maintains -- a list is shortened by deleting a
        line, and a module dropped from it is invisible.
        """
        walked = _preflight_sources()
        assert _PASSES in walked, (
            "config/passes.py is not in the walk, so the two static guards "
            "below no longer cover sweep() or check_where() -- the functions "
            "that call every check and validate every `where`."
        )
        assert set(_PREFLIGHT_DIR.glob("*.py")) <= set(walked), (
            "a module under preflight/ is missing from the walk, so the guards "
            "have become a maintained list rather than a discovery."
        )
        assert (_PREFLIGHT_DIR / "__init__.py") in walked

    def test_no_module_here_imports_its_way_out_of_the_phase(self):
        """The static half, which the monkeypatches cannot reach: a module that
        writes ``from rheplicant.config.values import resolve_value`` at its
        head holds the function by value and the patch above never sees the
        call.  ``rheplicant.config.document`` is banned outright -- it imports
        this package for the hook, so importing it back closes a cycle."""
        offenders = {
            path.name: sorted(found)
            for path in _preflight_sources()
            if (found := _out_of_scope_imports(path.read_text()))
        }
        assert offenders == {}, (
            f"{offenders} are imported under preflight/. Each is an entry "
            "point out of P-1: a builder, the file resolver, the value-node "
            "resolver, a path resolved against a built twin, or the document "
            "module itself (which imports this package and would close a "
            "cycle)."
        )

    def test_no_check_replays_variants_or_references_deleted_walkers(self):
        """The precondition :data:`_COLD_COST_CHILD`'s merge counter rests on,
        which was an unasserted claim in a docstring until a review walked
        through it.

        The counter patches ``rheplicant.config.layering``'s ATTRIBUTE, so a
        caller that writes ``from rheplicant.config.layering import
        apply_variant`` at its head holds the function by value and every merge
        it makes is invisible.  Measured: an eleventh merge site spelled that
        way adds 21 real deep merges to the guard's own document and leaves
        ``test_the_layers_are_built_once_per_declared_variant`` reading 21, with
        the whole module green.  That is the same hole as a check merging the
        variants itself -- which the count DOES catch -- arriving through the
        one door it cannot see.

        Module-level statements only: the two legitimate callers
        (``_task3_layers`` and ``_variant_text``) import it inside the function,
        which is what makes the attribute read at call time.

        **What this sees, and what it does not -- because it reads like a twin
        of ``_foot_imports`` and is not one.**  It catches
        ``from rheplicant.config.layering import apply_variant`` written as a
        bare module-level statement, which is how anyone would actually write
        it.  It does NOT catch that import wrapped in ``try:``/``if``, nor
        ``import rheplicant.config.layering`` followed by attribute access at
        call time (which is fine anyway -- the attribute is read late, so the
        counter sees it).  The wrapped form is contrived, and the asymmetry is
        recorded rather than closed because the two guards FAIL IN OPPOSITE
        DIRECTIONS: a spelling ``_foot_imports`` cannot see makes a module read
        as un-imported and turns that guard RED, while a spelling this one
        cannot see reads as no offender and leaves it GREEN.  Fail-safe there,
        fail-open here.  A reader who assumes both are exhaustive because they
        share a shape would be wrong about this one.
        """
        deleted = {
            "_task3_over_layers",
            "_task3_where",
            "_task3_layers",
            "_task3_build_layers",
            "_task3_forget_layers",
        }

        def violations(source: str) -> set[str]:
            found: set[str] = set()
            for node in ast.walk(ast.parse(source)):
                if isinstance(node, ast.Name) and node.id in deleted:
                    found.add(node.id)
                if isinstance(node, ast.Attribute) and node.attr in deleted:
                    found.add(node.attr)
                if isinstance(node, ast.ImportFrom):
                    found.update(
                        alias.name for alias in node.names if alias.name in deleted
                    )
                if isinstance(node, ast.Call):
                    called = node.func
                    if (
                        isinstance(called, ast.Name)
                        and called.id == "apply_variant"
                    ) or (
                        isinstance(called, ast.Attribute)
                        and called.attr == "apply_variant"
                    ):
                        found.add("apply_variant()")
            return found

        assert violations(
            "from x import _task3_layers\n"
            "def check(d):\n"
            "    return layering.apply_variant(d, 'x')\n"
        ) == {"_task3_layers", "apply_variant()"}
        offenders = {
            path.name: sorted(found)
            for path in _preflight_sources()
            if (found := violations(path.read_text()))
        }
        assert offenders == {}, (
            f"{offenders} retain a deleted layer walker/path helper or replay "
            "apply_variant inside a check. The pass driver owns the sole "
            "enumeration and attribution."
        )

    @pytest.mark.parametrize(("source", "expected"), [
        ("from rheplicant.config.sections.compose import build_model",
         {"build_model"}),
        ("from rheplicant.config.values import resolve_value as _rv",
         {"resolve_value"}),
        ("from rheplicant.config import load_document",
         {"load_document", "rheplicant.config"}),
        ("import rheplicant.config.document", {"rheplicant.config.document"}),
        ("from rheplicant.config.document import ConfiguredRun",
         {"rheplicant.config.document"}),
        ("def f():\n    from rheplicant.config.resources import build_resources",
         {"build_resources"}),
        ("from rheplicant.config.sections.runtime import build_runtime",
         {"build_runtime"}),
        ("import rheplicant.config", {"rheplicant.config"}),
        ("import rheplicant.config as cfg", {"rheplicant.config"}),
        ("from rheplicant import config", {"rheplicant.config"}),
        ("# from rheplicant.config.values import resolve_value", set()),
        ('"""build_model is not imported here."""', set()),
        ("from rheplicant.config.resources import check_unknown_keys", set()),
        ("from rheplicant.config.paths import parse_path", set()),
        ("from rheplicant.config.findings import refuse", set()),
    ], ids=["builder", "aliased-resolver", "via-the-package-re-export",
            "the-document-module", "a-name-from-the-document-module",
            "deferred-inside-a-function", "build-runtime",
            "the-umbrella-package", "the-umbrella-package-aliased",
            "the-umbrella-package-from-rheplicant", "commented",
            "in-a-docstring", "the-sanctioned-sweep",
            "the-sanctioned-path-parser", "the-sanctioned-findings"])
    def test_that_import_guard_reads_imports_and_not_mentions_of_them(
            self, source, expected):
        """ANTI-VACUITY, and the reason this is ``ast`` and not ``grep``.  2D
        shipped a tripwire that greped four strings, got four hits from a
        COMMENT, and counted code that could have been deleted.

        The three umbrella-package cases are the measured hole: the package
        re-exports ``resolve_value``, ``build_resources``, ``load_document``
        and ``run_document``, so ``import rheplicant.config as cfg`` reaches
        every one of them while naming none.  All three spellings resolve to
        different AST shapes and a draft caught none of them.

        The last four are the other direction: §2.5 REQUIRES every check to
        use ``check_unknown_keys``, which lives in the same module as
        ``build_resources``, so the ban is by imported NAME and a
        module-shaped ban would forbid the helper the plan mandates.
        """
        assert _out_of_scope_imports(source) == expected

    def test_no_module_here_calls_its_way_out_of_the_phase(self):
        """The BRANCH-INDEPENDENT half, and the one a runtime patch cannot be.

        Every guard above drives the pass on documents, so a filesystem read
        on a branch no document takes is invisible to all of them -- measured:
        a probe module reading ``observation.from_file`` and walking
        ``resources.beams`` directories passed this whole module at exit 0
        while three runtime guards watched.  This one reads the call whether
        the branch runs or not.
        """
        offenders = {
            path.name: sorted(found)
            for path in _preflight_sources()
            if (found := _out_of_scope_calls(path.read_text()))
        }
        assert offenders == {}, (
            f"{offenders} are called under preflight/. P-1 reads the "
            "document's text, RADIO_GRAPH and operator CLASSES, and nothing "
            "else: no file is opened, listed or stat-ed, no value node is "
            "resolved, no section is built. If one of these names is a "
            "false positive -- a local helper that happens to share a name "
            "-- that is a stop-and-ask, not a rename."
        )

    @pytest.mark.parametrize(("source", "expected"), [
        ("def f(p):\n    open(p).read()", {"open"}),
        ("def f(p):\n    return p.read_text()", {"read_text"}),
        ("def f(p):\n    return list(p.iterdir())", {"iterdir"}),
        ("def f(p):\n    return p.exists()", {"exists"}),
        ("def f(p):\n    return os.listdir(p)", {"listdir"}),
        ("def f(p):\n    return numpy.fromfile(p)", {"fromfile"}),
        ("def f(p):\n    return io.open_code(p)", {"open_code"}),
        ("def f(p):\n    return sqlite3.connect(p)", {"connect"}),
        ("def f(p):\n    return os.lstat(p).st_size", {"lstat"}),
        ("def f(p):\n    return list(p.rglob('*.npy'))", {"rglob"}),
        ("def f(p):\n    return os.getcwd()", {"getcwd"}),
        ("def f(p):\n    return h5py.File(p)", set()),
        ("def f(d):\n    return cfg.resolve_value(d, None)",
         {"resolve_value"}),
        ("def f(d):\n    return build_runtime(d)", {"build_runtime"}),
        ("def f(d):\n    if False:\n        return open(d)\n    return ()",
         {"open"}),
        ("# open(path)", set()),
        ('"""p.read_text() is never called here."""', set()),
        ("def f(d):\n    return check_unknown_keys('m', d, frozenset())",
         set()),
        ("def f(d):\n    return parse_path(d)[0]", set()),
        ("def f(d):\n    return dataclasses.fields(d)", set()),
    ], ids=["open", "read-text", "iterdir", "exists", "os-listdir",
            "numpy-fromfile", "io-open-code", "sqlite3-connect", "os-lstat",
            "path-rglob", "os-getcwd", "h5py-File-NOT-caught",
            "resolve-value", "build-runtime", "on-a-branch-never-taken",
            "commented", "in-a-docstring", "the-sanctioned-sweep",
            "the-sanctioned-path-parser", "class-introspection"])
    def test_that_call_guard_reads_calls_and_not_mentions_of_them(
            self, source, expected):
        """ANTI-VACUITY, both directions.

        ``on-a-branch-never-taken`` is the whole point: the call is
        unreachable and is still read, which is what the runtime patches
        cannot do.

        ``io-open-code``, ``sqlite3-connect``, ``os-lstat``, ``path-rglob``
        and ``os-getcwd`` are five the reviewer's probe module used.  The
        first two escaped BOTH halves; the last three escaped the runtime
        half only, which is what "the two halves disagreed" meant and what
        :meth:`test_the_two_halves_of_the_boundary_name_the_same_routes` now
        makes unshippable.

        ``h5py-File-NOT-caught`` is a declared blind spot, written as a case
        rather than as a sentence: ``File`` is too generic a name to ban
        without false positives, so an ``h5py.File`` read is caught by neither
        this guard, nor the filesystem patches, nor the audit hook --
        measured, on a real ``.h5`` it raises no ``open`` audit event at all.
        It is stopped instead by §0's rule that no module here may import
        ``h5py``, which the subprocess test below enforces.

        The last three are the other direction: §2.5's mandated helpers and
        §2.4's sanctioned class introspection must not be flagged.
        """
        assert _out_of_scope_calls(source) == expected

    def test_the_two_halves_of_the_boundary_name_the_same_routes(self):
        """The static ban and the runtime patches must cover ONE set of names.

        This is the assertion whose absence shipped the hole: ``lstat`` and
        ``rglob`` were on the static call ban and on no patch, and
        ``open_code``, ``connect``, ``statvfs``, ``readlink`` and ``getcwd``
        were on neither -- so "the two halves have disjoint blind spots and
        neither is the guard on its own" was a claim about two lists that had
        drifted apart.

        Kills a patch added without its ban, a ban added without its patch or
        a declared floor, and a floor pointing at a name nothing patches.
        """
        assert _FILESYSTEM_CALLS <= _OUT_OF_SCOPE_CALLS
        assert not frozenset(_FILESYSTEM_PATCHES) & frozenset(_FLOORED_UNDER), (
            "a verb cannot both have a patch of its own and be floored under "
            "another one -- one of the two entries is stale."
        )
        assert set(_FLOORED_UNDER.values()) <= set(_FILESYSTEM_PATCHES), (
            f"{sorted(set(_FLOORED_UNDER.values()) - set(_FILESYSTEM_PATCHES))}"
            " are named as floors and are patched by nothing."
        )
        patched = {target.rsplit(".", 1)[-1] for target in _FILESYSTEM_TARGETS}
        assert patched == set(_FILESYSTEM_PATCHES), (
            f"{sorted(patched ^ set(_FILESYSTEM_PATCHES))}: the targets "
            "actually rebound and the verbs the ban names have drifted apart."
        )
        assert not (frozenset(_STATIC_ONLY_CALLS)
                    & (frozenset(_FILESYSTEM_PATCHES)
                       | frozenset(_FLOORED_UNDER))), (
            "a verb cannot be both patched and declared un-patchable."
        )
        assert all(reason.strip() for reason in _STATIC_ONLY_CALLS.values()), (
            "every verb banned with no runtime partner states WHY, because "
            "'banned and patched by nothing' with no reason beside it is the "
            "state lstat and rglob were in when the probe walked out."
        )

    def test_the_pass_raises_no_audit_event_that_reaches_the_os(self):
        """The half no rebinding of a Python name can be, in a subprocess.

        ``sys.addaudithook`` is where CPython itself announces a read, so it
        sees an opener that lives ENTIRELY inside a C extension --
        ``io.open_code`` and ``sqlite3.connect``, measured, reach the OS
        without touching ``builtins.open`` and were invisible to every patch
        in this module.  A reviewer's probe used both, read 15 671 bytes and
        created a SQLite database on every document, with this whole module
        at exit 0.

        A subprocess, because an audit hook cannot be removed once added --
        that is the point of the API -- so installing one in the test process
        would leave it in front of every later test in the session.

        **Anti-vacuity is inside the child**: it arms the hook, performs one
        ``open`` of its own, and reports that it saw it.  A hook that stopped
        firing -- a Python build with auditing compiled out, an exception
        swallowed in the child, a typo in an event name -- would otherwise
        make this green forever, which is 2C's discovery-by-prefix guard one
        level down.
        """
        source = _AUDIT_CHILD.replace("__BRANCHES__", repr(_BRANCH_DOCUMENTS))
        done = subprocess.run([sys.executable, "-c", source],
                              capture_output=True, text=True, cwd=str(_ROOT))
        assert done.returncode == 0, done.stdout + done.stderr
        lines = done.stdout.strip().splitlines()
        assert lines and lines[-1] == "ANTI-VACUITY-OK", (
            "the child's own control read raised no audit event, so this "
            f"guard could not have seen one either: {done.stdout!r}"
        )
        assert lines[:-1] == [], (
            "the pre-flight pass reached the OS on these documents:\n"
            + "\n".join(lines[:-1])
        )

    def test_importing_the_pass_drags_in_no_optional_dependency(self):
        """§0's invariant, at the module that could break it.  ``preflight`` is
        imported by ``document``, which is imported by ``rheplicant.config`` --
        so one ``import healpy`` at the head of a check module puts healpy in
        every process that reads a config.  Measured at ``be2027b``: none of
        the six is present."""
        source = (
            "import sys, rheplicant.config.preflight\n"
            "print(sorted(m for m in ('numpyro', 'limtod_jax', 'healpy', "
            "'h5py', 'pyuvdata', 'rhino_cal_jax') if m in sys.modules))\n"
        )
        done = subprocess.run([sys.executable, "-c", source],
                              capture_output=True, text=True, check=True)
        assert done.stdout.strip() == "[]", done.stdout


#: The child of :class:`TestTheColdCostOnARealDocument`.  A fresh process and
#: ONE ``preflight()`` call, because the cost this is about is a deferred
#: import and a second call cannot see one.
#:
#: **``apply_variant`` is counted here, and the count -- not the clock -- is
#: what the class asserts on.**  The patch goes in before ``preflight`` is
#: imported, but that is belt and braces: every caller of ``apply_variant``
#: under ``preflight/`` imports it INSIDE the function, so the module attribute
#: is read at call time and a patch installed at any point before the pass is
#: seen.  Its cost is one Python frame per call -- twenty-one of them on this
#: document -- which is microseconds against a pass measured in milliseconds,
#: so the same child can carry the clock and the counter without the counter
#: moving the clock.
_COLD_COST_CHILD = '''
import sys, time

import _rheplicant_bootstrap.variants as _variants

_MERGES = []
_real_apply_variant = _variants.apply_variant


def _counting_apply_variant(document, name):
    _MERGES.append(name)
    return _real_apply_variant(document, name)


_variants.apply_variant = _counting_apply_variant

from rheplicant.config.preflight import preflight
from tests.config.preflight_helpers import BASE_MODEL, preflight_document

RUN = {"kind": "plan.sample", "seed": "runtime.seeds.a", "n_sweeps": 8,
       "warmup": 2, "blocks": [{"names": ["g"], "engine": "conjugate"}]}
runs = [dict(RUN, name="s%d" % i) for i in range(__RUNS__)]
variants = {"v%d" % i: {"model": {"gain": {"gain": {
    "value": float(i), "unit": "dimensionless"}}}}
    for i in range(__VARIANTS__)}
model = dict(BASE_MODEL)
model["bandpass"] = {"bandpass": {"ones": ["n_freq"]}}
document = preflight_document(runs=runs, variants=variants, model=model)

start = time.perf_counter()
report = preflight(document)
print(time.perf_counter() - start)
print(len(report.findings))
print(" ".join(sorted(name for name in sys.modules
                      if name == "rheplicant.inference"
                      or name.startswith("rheplicant.inference.")
                      or name in ("numpyro", "healpy", "h5py"))))
print(len(_MERGES))
print(len(document.get("variants") or {}))
'''


#: How many fresh processes the wall-clock backstop takes its minimum over.
#: Three, because a review measured one child in twenty stalling to 55.48 ms on
#: a 14 ms pass under ``-n 16`` plus eight CPU hogs: at that rate three
#: independent children all stalling is about one run in eight thousand, and
#: each child costs ~0.43 s.  It is not five because this is a backstop and the
#: instrument beside it is exact.
_COLD_CHILDREN = 3


class _ColdRun(NamedTuple):
    """One child process's five prints, named rather than positional.

    The child grew two prints when the count became the instrument, and a
    caller unpacking a bare tuple would have taken the wrong one silently.
    """

    #: Seconds for ONE ``preflight()`` call in a fresh process.
    cold: float
    #: How many findings that call produced -- the anti-vacuity read.
    findings: int
    #: Modules from the inference layer that reached ``sys.modules``.
    dragged: list[str]
    #: How many times ``apply_variant`` ran during the pass.
    merges: int
    #: How many variants the child's document declares.
    variants: int


class TestTheColdCostOnARealDocument:
    """§5's 0.05 s, COLD, on the document that used to breach it.

    **Both shipped cost tests measured warm, on documents that take neither
    expensive path.**  ``test_the_pass_costs_less_than_a_twentieth_of_a
    _second`` calls ``preflight(doc)`` once to warm the import graph before it
    starts the clock; ``test_config_surface``'s runs inside a warm session on
    a ``conjugate.wiener`` document.  Neither declares a ``kind: plan.sample``
    run, and :data:`_BRANCH_DOCUMENTS` had no ``plan.sample`` row either -- so
    the one branch whose real margin was 2x was the one branch nothing drove.

    Measured, cold, one call per fresh process, before the fix: the worked
    document **1.6 ms** (31x margin); one ``kind: plan.sample`` run **23 ms**
    (2.2x); that document plus 20 variants **32 ms** (1.6x); 40 runs, 20
    variants and a fatter model **46 ms** (1.1x).  The cause was one deferred
    ``from rheplicant.inference import MIN_DRAWS``: 43 modules on the first
    call, for one integer.  After it (``fitting._T9_MIN_DRAWS``): 1.6 ms,
    3.4 ms, 12 ms and **26 ms**, and ``sys.modules`` gains six config modules
    and nothing from ``rheplicant.inference`` at all.

    **What WAS still linear in the number of checks, and what fixed it.**
    ``document._task3_over_layers`` runs the pass once per DECLARED VARIANT,
    which §2.1 says is the right granularity -- a variant IS a different
    document -- and 88-94 % of what remained after the ``MIN_DRAWS`` fix was
    ``apply_variant``'s deep merge.  It was called once per declared variant
    **per check that layers**, and the number of such checks grew: four at
    ``ea4839b``, ELEVEN at the wave-1 tip -- ten ``_task3_over_layers`` call
    sites plus ``_variant_text``, which merged every variant a second time to
    run ``_structural`` over it.  TEN of the eleven run on this child's
    document, because ``noise``'s walk is gated off on it.
    Measured on that document, which declares twenty-one variants:
    210 merges and a 45 ms pass, against a 50 ms budget it then breached 3
    runs in 5 under ``pytest tests/config -n 16`` (52.4, 71.7, 60.4 ms).

    The layer memo's successor is Task 4's canonical enumeration
    (``_rheplicant_bootstrap.variants.enumerate_layers_once``): **21
    merges**, and the merge count does
    not move when an eleventh layering check is added.  The 13 ms that number
    carried when the memo landed is stale since Plan 4A's Tasks 3-5 hardened
    the evidence pipeline: ~58 ms quiet, measured 2026-08-19 (the test
    below's docstring carries the decomposition).

    **So the instrument here is the merge COUNT, not the clock.**
    ``test_the_layers_are_built_once_per_declared_variant`` is the assertion
    that can see the thing this class is about; the wall clock below is a
    backstop, and says so.
    """

    def _run(self, runs: int, variants: int) -> _ColdRun:
        source = (_COLD_COST_CHILD.replace("__RUNS__", str(runs))
                  .replace("__VARIANTS__", str(variants)))
        done = subprocess.run([sys.executable, "-c", source],
                              capture_output=True, text=True, cwd=str(_ROOT))
        assert done.returncode == 0, done.stdout + done.stderr
        cold, found, dragged, merges, declared = done.stdout.split("\n")[:5]
        return _ColdRun(float(cold), int(found), dragged.split(),
                        int(merges), int(declared))

    def test_the_layers_are_built_once_per_declared_variant(self):
        """THE instrument: a call count, which no load on this box can move.

        The property is that one pass merges each declared variant ONCE --
        not once per variant per check that walks layers.  It is exact, it is
        the thing that regressed, and unlike the wall clock below it cannot be
        made to pass by a quiet machine or to fail by a loud one.

        **R9, both halves, run rather than asserted.**  Reverting
        ``_task3_layers`` to the eager walk takes this to 210 and the assertion
        red.  Adding an eleventh check that walks layers THROUGH
        ``_task3_over_layers`` leaves it at 21 and green -- which is the
        property, not a hole: the count is meant to be independent of how many
        checks layer.  Adding one that calls ``apply_variant`` itself, the way
        ``_variant_text`` used to, takes it to 42 and red, which is the
        regression a reader of this test most needs caught.

        Both anti-vacuity reads are here because either one alone is passable
        by an empty document: a child whose document declared no variants
        would assert ``0 == 0``, and one that earned no findings would be
        counting merges nothing looked at.
        """
        result = self._run(runs=40, variants=20)
        assert result.variants == 21, (
            f"the child's document declares {result.variants} variants, not "
            "the base fixture's one plus this child's twenty -- the count "
            "below would be asserting about a document nobody wrote"
        )
        assert result.findings >= 40, (
            f"the child's document earned {result.findings} findings, fewer "
            "than one per run -- it is not exercising the checks whose cost "
            "this measures"
        )
        assert result.merges > 0
        assert result.merges == result.variants, (
            f"one pass merged {result.merges} variants for "
            f"{result.variants} declared. apply_variant is a deep merge of "
            "the whole document; it belongs to the pass, not to each check "
            "that walks layers. Measured before preflight/document.py's "
            "layer memo: 210, at the ten merge sites this document reaches."
        )

    def test_a_cold_pass_on_forty_runs_and_twenty_variants_is_under_the_budget(
            self):
        """§5's cold-pass budget, as a BACKSTOP -- not as this class's
        instrument.

        A one-shot wall clock, in a subprocess spawned from an xdist worker,
        on a box running its own suite at ``-n 16``, measures the box as much
        as the pass.  Measured: at ``ea4839b`` -- with nothing this plan wrote
        present -- it went red 2 runs in 5 under ``-n 16`` and 1 in 9 running
        serially, at 52.2 ms against a 21-23 ms pass; at the wave-1 tip, 3 in
        5.  Four reviewers hit it independently.  So the number that decides
        whether the layer walk regressed is
        ``test_the_layers_are_built_once_per_declared_variant``'s count, and
        this one is here to catch a cost that arrives from somewhere the count
        cannot see -- another 43-module deferred import, a check that reads a
        file -- rather than to police a few milliseconds.

        **The bound was re-measured at 2026-08-19 (Plan 4A's OI-1 triage) and
        moved to 150 ms.**  §5's 50 ms predates the evidence-hardened
        enumeration: the strict ``freeze_evidence``/origin pipeline from Tasks
        3-5 (five review rounds, the audit trail's foundation) took this exact
        pass from 16.5 ms at ``ac807dc`` to 119 ms at Task 3's terminal, back
        to 42 ms at Task 4, and to ~58 ms quiet since Task 5 -- measured as the
        minimum of three fresh processes on this box, with the merge count
        exact (21) and nothing dragged from ``rheplicant.inference``.  The
        hardening is the contract, so the bound moved to honest numbers rather
        than the contract to the stale bound; the distributed cost (thaw ~15
        ms + sweeps ~30 ms + enumerate ~10 ms over 22 layers) has no single
        accidental sink, and recovering it is a recorded optimization
        opportunity, not a triage edit.

        **The number is the MINIMUM over :data:`_COLD_CHILDREN` fresh
        processes, and one reading is not enough -- measured, not assumed.**  A
        review spawned twenty cold children under ``-n 16`` plus eight CPU-bound
        processes and read ``min 13.77, median 14.22, max 55.48 ms``: one child
        in twenty stalled past the bound on a 14 ms measurement.  A single
        reading would still be deciding this verdict on scheduler noise, which
        is the defect this whole commit exists to remove.  Contention can only
        ADD time, so the minimum is the least-noise estimator -- the argument
        ``inflight_helpers.best_ms`` makes, and the correction
        ``test_preflight_depends_cost.py::_COST_CHILD`` already shipped for the
        identical shape.

        **Across CHILDREN and not across passes inside one child, which is not
        interchangeable here.**  ``_COST_CHILD`` can take its minimum over five
        calls because what it times is a per-layer walk.  This class times a
        COLD pass: the regression it was built for was a deferred
        ``from rheplicant.inference import MIN_DRAWS`` costing 43 modules on the
        FIRST call and nothing after, so a minimum over passes in one process
        would report the warm number and never see it again.  Every sample here
        is a first call in its own interpreter.

        **Its measured sensitivity, so that nobody reads it as more than it
        is:** making ``apply_variant`` ten times slower while leaving the call
        count alone lands this at 43-46 ms and it stays GREEN.  A cost that
        arrives per merge is the count's blind spot AND nearly this one's; what
        this catches is a cost that arrives in bulk, which is the shape every
        regression this class has actually seen has had.

        ANTI-VACUITY is the finding count, on EVERY child: a document that
        earns nothing has not run the checks whose cost is the subject, and
        this one earns one A24 per ``plan.sample`` run.  ``>=`` and not ``==``
        because the child's document is ``preflight_document()``, the shared
        base fixture, and a sibling task registering a check that fires on 40
        ``plan.sample`` runs would take an exact count red in a file it never
        opened (R8).
        """
        results = [self._run(runs=40, variants=20)
                   for _ in range(_COLD_CHILDREN)]
        for result in results:
            assert result.findings >= 40, (
                f"a child's document earned {result.findings} findings, fewer "
                "than one per run -- it is not exercising the checks whose "
                "cost this measures"
            )
        best = min(result.cold for result in results)
        assert best < 0.15, (
            f"the fastest of {_COLD_CHILDREN} cold passes on 40 plan.sample "
            f"runs and 20 variants took {best * 1000:.1f} ms against the "
            "re-measured 150 ms. This is the backstop, so read the merge "
            "count first: the evidence-hardened enumeration has cost ~58 ms "
            "quiet since Plan 4A's Task 5 (measured 2026-08-19)."
        )

    def test_the_cold_pass_drags_in_no_part_of_the_inference_layer(self):
        """The MECHANISM, and the half that cannot flake on a loaded box.

        A time assertion with a 2x margin says "fast enough today"; this says
        "the 43 modules are gone", which is the thing that was wrong.  Kills
        the deferred ``from rheplicant.inference import MIN_DRAWS`` coming
        back, and any sibling of it -- ``rheplicant.inference.plan``, which a
        draft reached for instead, costs exactly the same 43 because
        importing the submodule runs the package ``__init__`` first.
        """
        result = self._run(runs=40, variants=20)
        assert result.dragged == [], (
            f"{result.dragged} reached sys.modules during one pre-flight "
            "pass. The pass reads text; nothing under rheplicant.inference "
            "is text."
        )


def _enumeration(document):
    merged = initial_merge(document, origin=Origin("user"))
    return enumerate_layers_once(
        merged.document, merged.origins, merged.deletions
    )


def _prefixes(document) -> list[str]:
    """The canonical layer prefixes for ``document``."""
    return [layer.prefix for layer in _enumeration(document).layers]


class TestCanonicalLayerEnumeration:
    """The pass-scoped canonical enumerator has no cross-pass memo."""

    def _document(self, **variants):
        return preflight_document(
            variants={name: {"runtime": {"seed": seed}}
                      for name, seed in variants.items()})

    def test_each_enumeration_returns_a_fresh_frozen_record(self):
        document = self._document(a=1, b=2)
        first = _enumeration(document)
        second = _enumeration(document)
        assert first is not second
        assert tuple(layer.prefix for layer in first.layers) == tuple(
            layer.prefix for layer in second.layers
        )

    def test_a_second_document_gets_its_own_layers(self):
        """One entry, and it is replaced rather than shared.

        Kills a memo that answers the first document it ever saw for every
        document after it -- which passes the test above and every cost
        assertion in this file.
        """
        first = self._document(a=1)
        second = self._document(b=2, c=3)
        assert "variants.a" in _prefixes(first)
        assert "variants.a" not in _prefixes(second)
        assert {"variants.b", "variants.c"} <= set(_prefixes(second))
        assert "variants.a" in _prefixes(first)

    def test_a_fresh_enumeration_reads_a_mutated_document(self):
        document = self._document(a=1)
        before = _prefixes(document)
        document["variants"]["b"] = {"runtime": {"seed": 2}}
        assert "variants.b" not in before
        assert "variants.b" in _prefixes(document)

    def test_an_unused_variant_is_seen_by_every_registered_check(self):
        """Layer fan-out belongs to the driver, not to selected checks."""
        document = preflight_document(
            variants={"unused": {"model": {"ghost": {}}}}
        )
        found = [one for one in findings(document)
                 if one.check == "A2"
                 and one.where == "variants.unused.model"]
        assert len(found) == 1
        assert found[0].message.startswith(
            "variants.unused: model: 'ghost' is not a node"
        )

    def test_a_document_edited_between_two_passes_is_read_afresh(self):
        """And the limit stops at the pass boundary, which is not optional.

        R4's shape -- earn a refusal, write the remedy the message advises
        into the document, ask again -- edits a document in place between two
        ``preflight`` calls, and the suite is full of it;
        ``test_preflight_instrument.py:970`` is one and went red against a memo
        that outlived the pass.  ``preflight`` drops the entry at the head of
        every pass, so the second read is of the document as it is now.

        Asserted on the finding's own ``where`` rather than on the id set:
        ``A1`` is on the shared base document already (§0.3 E.11), so
        ``"A1" not in ids(doc)`` would be false for reasons that have nothing
        to do with this variant.
        """
        document = preflight_document(variants={"v": {"campaign": {"of": 1}}})
        assert any(finding.where == "variants.v.campaign"
                   for finding in findings(document)), (
            "the variant this test edits earns nothing, so the second read "
            "below could not tell a fresh walk from a stale one"
        )
        del document["variants"]["v"]["campaign"]
        assert not any(finding.where == "variants.v.campaign"
                       for finding in findings(document)), (
            "a second pass answered from the first pass's layers: the "
            "document was edited between them and preflight() is supposed to "
            "drop the layer memo at the head of every pass"
        )
        assert _prefixes(document) == [""] + [
            f"variants.{name}" for name in document["variants"]]


#: A placeholder inside a harvested message: where an f-string interpolated
#: something.  Not ``{}``, because several shipped messages contain literal
#: braces, and not a word, because a word could be text.
_HOLE = "…"

#: The shortest run of characters this guard is willing to call a message.
#: Below it a literal is a key name, a ``kind:`` word or a format fragment --
#: things that move for reasons that are not rewordings, and that would make
#: the guard fire on every ordinary refactor until somebody deleted it.
_MESSAGE_FLOOR = 40


def _message_texts(source: str) -> set[str]:
    """Every message-shaped string literal in ``source``, whitespace-flattened.

    A message is ONE ``ast`` node even when it is written as a dozen
    implicitly concatenated pieces across a dozen lines -- CPython folds
    adjacent string literals in the parser, and an f-string mixed in among
    them folds into a single ``JoinedStr``.  So this needs no line joining of
    its own, and it cannot be fooled by a re-wrap.

    Interpolations become :data:`_HOLE`, so ``f"got {value!r}"`` compares
    equal across a change of variable name and unequal across a change of
    words.  ``ast.walk`` visits a ``JoinedStr``'s literal parts as well as the
    whole, which is deliberate: a fragment is pinned as well as the sentence,
    so a message that is later assembled from clauses still has its pieces
    checked.

    DOCSTRINGS ARE EXCLUDED.  They are the layer's reasoning, they are edited
    constantly and on purpose, and including them would make this guard fire
    on every commit until somebody deleted it -- which is the one outcome a
    standing guard must not have.
    """
    tree = ast.parse(source)
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef)):
            first = node.body[0] if node.body else None
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    found = set()
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value
        elif isinstance(node, ast.JoinedStr):
            text = "".join(
                part.value if (isinstance(part, ast.Constant)
                               and isinstance(part.value, str)) else _HOLE
                for part in node.values)
        else:
            continue
        flat = re.sub(r"\s+", " ", text).strip()
        if len(flat) >= _MESSAGE_FLOOR:
            found.add(flat)
    return found


#: The base this plan's messages are compared against: ``main`` at the commit
#: the plan was written on.
_BASE_COMMIT = "be2027b"

#: How many messages the base tree must yield before this guard believes its
#: own harvest.  Measured at ``be2027b``: **907**.  A floor rather than the
#: number, because the base tree never changes but the harvester might, and a
#: guard whose corpus quietly became empty would pass forever -- 2C's
#: discovery-by-prefix shape, at the scale of a whole package.
_BASE_MESSAGE_FLOOR = 800

#: Base messages that are deliberately no longer SOURCE LITERALS, and the
#: test that pins the sentence they became.  Every entry needs one, because
#: "it moved into a clause" is exactly what a rewording would also claim.
#:
#: All three are A28's, and all three are the same sentence at three
#: granularities (the whole ``JoinedStr`` and two of its folded parts).  It is
#: assembled from ``conjugate._A28_GLS_CLAUSES`` now rather than written out,
#: because ``npe`` needed different words for two of its fragments and
#: ``conjugate.gls`` had to keep ``be2027b``'s to the character.
_ASSEMBLED_ELSEWHERE: dict[str, str] = {
    literal: "test_config_conjugate_shared.py::"
             "test_the_gls_refusal_is_be2027b_verbatim"
    for literal in (
        "runs[" + _HOLE + "]: kind: " + _HOLE + " solves for the covariance a "
        "PREDICTION-DEPENDENT sigma implies, so it reads inference.noise as a "
        "model; inference.noise.kind: " + _HOLE + " decides its sigma into an "
        "array before any run sees it, and a decided array has no fixed point "
        "to iterate (check A28). Declare inference.noise.kind: radiometer to "
        "iterate the rule, or run kind: conjugate.wiener, which is what a "
        "decided sigma wants.",

        "solves for the covariance a PREDICTION-DEPENDENT sigma implies, so "
        "it reads inference.noise as a model; inference.noise.kind:",

        "decides its sigma into an array before any run sees it, and a "
        "decided array has no fixed point to iterate (check A28). Declare "
        "inference.noise.kind: radiometer to iterate the rule, or run kind: "
        "conjugate.wiener, which is what a decided sigma wants.",
    )
}

#: Compatibility diagnostics intentionally hardened by committed Task 3
#: (``2dc8904``): callback-controlled repr text was removed while the static
#: sentence and concrete type remain.  These are not Task 4 message moves.
_TASK3_SAFE_CORRECTIONS: dict[str, str] = {
    literal: "test_config_preflight.py::"
             "test_task3_compatibility_diagnostics_keep_static_whole_strings"
    for literal in (
        "recursive_update: … is a mapping; got … (…).",
        "variants: is a mapping of name -> patch; got … (…).",
        "variant …: the patch is a mapping of sections; got … (…).",
    )
}

#: Base messages CORRECTED by a plan that names them, and the equality pin on
#: the sentence that replaced each.  **Separate from
#: :data:`_ASSEMBLED_ELSEWHERE` on purpose**: "this sentence is assembled from
#: clauses now" and "this sentence was deliberately reworded" are different
#: claims, and only the first is what that list's docstring describes.  Routing
#: a correction through the assembly list would make the guard's own words
#: false about its own contents, which is the shape of defect this whole class
#: exists to catch.
#:
#: One entry, and its authority is written down rather than assumed: Plan 3B
#: §0.2 C-10 rules that ``sections/observed.py`` compared a file's shape
#: against the GRIDS while citing "check C11", and names Task 8 as the fixer.
#: Measured with ``averaging: {n_chunk: 4}`` on (16, 8) grids -- prediction
#: (4, 8) -- the shipped code accepted the (16, 8) file and refused the (4, 8)
#: one, the exact inverse of the rule its own sentence states.  The clause that
#: was right is kept character for character; "this run's grids say" becomes
#: "this run's fit twin predicts".
_CORRECTED_BY_PLAN: dict[str, str] = {
    _HOLE + ": the file holds shape " + _HOLE + "; this run's grids say "
    + _HOLE
    + ". Exactly -- broadcast-compatible is the dangerous case (check C11).":
        "test_config_section_observed.py::"
        "test_the_refusal_names_the_prediction_and_keeps_the_clause_that_was_right",
}


class TestNoMovedMessageWasReworded:
    """§5's *"every check moved rather than written keeps its message
    verbatim"*, as a guard over the whole layer rather than as a sentence.

    **What shipped instead, and what it could not see.**  The plan's verbatim
    guard equality-pins ``_structural``'s five sweep messages and nothing
    else, so the other ~30 moved messages were held only by the ``match=``
    substrings that happened to already exist -- and ``match=`` is a search.
    A28's ``conjugate.gls`` sentence was reworded from *"so it reads
    inference.noise as a model"* to *"as a RULE"* and from *"a decided array
    has no fixed point to iterate"* to *"is not a rule"*, its own test was
    rewritten from substrings to full equality on the NEW words, and the
    whole suite stayed green.  3A's §2.3 designated exactly four messages
    CORRECTED (A39's) and called a fifth a stop-and-ask; Plan 3B §0.2 C-10
    designates a **fifth**, C11's, naming Task 8 as its fixer and the defect
    it repairs (a shape compared against the grids while the sentence claims
    to be about the prediction).  So the standing count is **A39's four plus
    C11, and a SIXTH is a stop-and-ask.**  The five are not interchangeable
    and are not in one list: A28's three live in
    :data:`_ASSEMBLED_ELSEWHERE` because they became clauses, C11's lives in
    :data:`_CORRECTED_BY_PLAN` because it was reworded on purpose, and each
    entry names the equality pin on the sentence that replaced it.

    **The guard.**  Harvest every message-shaped string literal from
    ``src/rheplicant/config/`` at :data:`_BASE_COMMIT` and from the tree as it
    stands, and require the base to be a SUBSET.  Additions are free -- this
    plan writes many new messages -- and a disappearance is a rewording, a
    deletion, or a move into a clause, all three of which are things a reader
    of §2.3 must be told about.

    Run against ``be2027b``, this finds **3** missing literals out of 907, and
    all three are A28's -- which is both the measurement that the layer's
    other messages survived and the reason to believe the harvester works.

    **What it cannot see**, stated because a verification method with the same
    blind spot as the code is this project's recorded failure mode:

    * a message that survives as TEXT and stops being REACHED.  Nothing here
      runs the pass.  The reviewer's 863-document differential is what
      measured that (no refusal disappeared); the per-check tests are what
      keep it true.
    * a change under :data:`_MESSAGE_FLOOR` characters, and a change to a
      DOCSTRING, both excluded on purpose above.
    * a fragment that becomes an f-string placeholder holding different
      words -- which is precisely how A28 escaped, and which is why
      :data:`_ASSEMBLED_ELSEWHERE` demands a named equality pin per entry
      rather than being a list of forgiven strings.
    """

    def _base_sources(self) -> dict[str, str]:
        """``src/rheplicant/config/`` at :data:`_BASE_COMMIT`, from the object
        store -- one ``git archive``, no checkout, no network."""
        try:
            done = subprocess.run(
                ["git", "archive", "--format=tar", _BASE_COMMIT,
                 "src/rheplicant/config"],
                cwd=str(_ROOT), capture_output=True)
        except OSError as error:                       # pragma: no cover
            pytest.skip(f"git is not runnable here: {error}")
        if done.returncode != 0:                       # pragma: no cover
            pytest.skip(
                f"{_BASE_COMMIT} is not in this repository "
                f"({done.stderr.decode(errors='replace').strip()}), so the "
                "pre-move text cannot be read"
            )
        found = {}
        with tarfile.open(fileobj=io.BytesIO(done.stdout)) as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith(".py"):
                    handle = archive.extractfile(member)
                    found[member.name] = handle.read().decode()
        return found

    def test_the_harvest_is_not_empty_and_reads_a_message_it_should(self):
        """ANTI-VACUITY, and it is the assertion the class rests on.

        Two ways this guard could pass while checking nothing: the base tree
        comes back empty (a broken ``git archive``, a renamed directory), or
        the harvester stops recognising a message (an ``ast`` change, a
        docstring filter that swallows everything).  Both are silent, and
        both make ``base <= head`` trivially true.
        """
        base = self._base_sources()
        assert base, "git archive returned no python files at all"
        harvested = set()
        for source in base.values():
            harvested |= _message_texts(source)
        assert len(harvested) >= _BASE_MESSAGE_FLOOR, (
            f"{len(harvested)} messages harvested from {_BASE_COMMIT}, "
            f"against {_BASE_MESSAGE_FLOOR} expected -- the harvester has "
            "stopped seeing messages and this guard is checking nothing."
        )
        assert ("campaign: is reserved with capability 4 (streaming evidence, "
                "schema §8.2) and refused in v1.") in harvested, (
            "a message this plan is known to have MOVED verbatim is not in "
            "the harvest, so the harvest is not reading messages."
        )

    def test_every_message_the_layer_shipped_at_be2027b_still_exists(self):
        """The guard itself.

        Kills the A28 rewording, and any sibling of it: a moved check whose
        sentence is edited while its ``match=`` pins go on passing.
        """
        base = set()
        for source in self._base_sources().values():
            base |= _message_texts(source)
        head = set()
        roots = [(_ROOT / "src" / "rheplicant" / "config"),
                 (_ROOT / "src" / "_rheplicant_bootstrap")]
        for root in roots:
            for path in sorted(root.rglob("*.py")):
                head |= _message_texts(path.read_text())

        missing = (base - head - set(_ASSEMBLED_ELSEWHERE)
                   - set(_TASK3_SAFE_CORRECTIONS)
                   - set(_CORRECTED_BY_PLAN))
        assert missing == set(), (
            f"{len(missing)} message(s) this layer shipped at {_BASE_COMMIT} "
            "are gone. A MOVED check keeps its message verbatim. Five are "
            "designated CORRECTED so far -- A39's four (3A §2.3) and C11's "
            "(3B §0.2 C-10) -- so a SIXTH is a stop-and-ask. If a plan names "
            "yours, add it to _CORRECTED_BY_PLAN with the test that pins the "
            "replacement by EQUALITY; if the sentence is merely assembled "
            "from clauses now rather than written out, add it to "
            "_ASSEMBLED_ELSEWHERE the same way. The two are different claims "
            "and are deliberately not one list.\n\n"
            + "\n\n".join(sorted(missing))
        )

    def test_every_forgiven_message_names_a_pin_that_exists(self):
        """Both forgiveness lists are only as good as the tests they name.

        A forgiveness list whose pins have been deleted is a list of
        rewordings nobody is checking -- which is the state this class was
        written to end, one indirection along.

        It walks :data:`_ASSEMBLED_ELSEWHERE` **and**
        :data:`_CORRECTED_BY_PLAN`: splitting the lists to keep their two
        claims apart would be a loosening rather than a tightening if only
        one of them were still checked here.
        """
        forgiven = {
            **_ASSEMBLED_ELSEWHERE,
            **_TASK3_SAFE_CORRECTIONS,
            **_CORRECTED_BY_PLAN,
        }
        assert len(forgiven) == (len(_ASSEMBLED_ELSEWHERE)
                                 + len(_TASK3_SAFE_CORRECTIONS)
                                 + len(_CORRECTED_BY_PLAN)), (
            "a literal is forgiven by BOTH lists, so one of the two claims "
            "about it -- 'assembled from clauses' and 'reworded on purpose' "
            "-- is untrue and nothing here says which."
        )
        for literal, pin in forgiven.items():
            module, _, name = pin.partition("::")
            path = _HERE / module
            assert path.is_file(), f"{pin} names no module ({literal[:60]}...)"
            tree = ast.parse(path.read_text())
            defined = {node.name for node in ast.walk(tree)
                       if isinstance(node, ast.FunctionDef)}
            assert name in defined, (
                f"{pin} is the only thing standing between this message and "
                f"a silent rewording, and it does not exist: {literal[:60]}..."
            )

    def test_task3_compatibility_diagnostics_keep_static_whole_strings(self):
        """Pin Task 3's approved removal of callback-controlled repr text."""
        from _rheplicant_bootstrap.layering import apply_variant, recursive_update

        cases = (
            (lambda: recursive_update([], {}),
             "recursive_update: base is a mapping; got list."),
            (lambda: apply_variant({"variants": []}, "v"),
             "variants: is a mapping of name -> patch; got list."),
            (lambda: apply_variant({"variants": {"v": []}}, "v"),
             "variant 'v': the patch is a mapping of sections; got list."),
        )
        for callback, expected in cases:
            with pytest.raises(ConfigError) as caught:
                callback()
            assert str(caught.value) == expected


class TestTheFootImportCannotRot:
    """A module under ``preflight/`` that nobody imports registers nothing and
    stays green -- which is 2C's discovery-by-prefix guard that matched
    nothing, one level down.  Kills exactly that."""

    def test_every_module_under_preflight_is_imported_at_the_foot(self):
        present = {path.stem for path in _PREFLIGHT_DIR.glob("*.py")
                   if path.stem != "__init__"}
        declared = _foot_imports((_PREFLIGHT_DIR / "__init__.py").read_text())
        assert present <= declared, (
            f"{sorted(present - declared)} live under preflight/ and are "
            "imported by nothing, so their @register decorators never run "
            "and their checks are silently absent. Add them to the foot "
            "import in preflight/__init__.py."
        )
        assert declared <= present, (
            f"{sorted(declared - present)} are imported at the foot and do "
            "not exist."
        )
        # This second direction lost one case when `_foot_imports` narrowed to
        # module-level statements: a name imported only inside a function, for
        # a module that does not exist, is no longer in `declared` and is no
        # longer reported here.  Immaterial -- a module that does not exist
        # registers no checks, and the import raises `ImportError` the first
        # time the function runs -- but it is a real change in what this line
        # covers, and it belongs here rather than in a later bug report.

    def test_every_module_under_preflight_contributes_a_slot(self):
        """PRESENCE IS NOT CONTRIBUTION, and the test above only reads
        presence.

        Measured: a module added under ``preflight/`` AND named in the foot
        import, whose body registers nothing at all, passes the whole suite
        unchanged -- which is 2C's discovery-by-prefix guard that matched
        nothing, one level further down than the test above reaches.  The
        import list can be complete while the registry is missing a whole
        file's worth of checks, and the only symptom is that documents stop
        being refused.

        ``fn.__module__`` rather than the foot import: the question is which
        module the LIVE registry's functions came from, so a module whose
        ``@register`` calls were commented out is caught even though its
        import still runs.

        A later plan that genuinely wants a helper module here with no check
        of its own must say so by editing this test, which is the point --
        the alternative is a silent hole shaped exactly like the one that
        shipped.
        """
        present = {path.stem for path in _PREFLIGHT_DIR.glob("*.py")
                   if path.stem != "__init__"}
        contributing = {fn.__module__.rsplit(".", 1)[-1]
                        for fn in CHECKS.values()}
        assert present <= contributing, (
            f"{sorted(present - contributing)} live under preflight/ and own "
            "no slot in CHECKS, so nothing they contain ever runs and the "
            "foot-import test above stays green anyway. Every module here "
            "registers at least one check."
        )
        assert contributing <= present, (
            f"{sorted(contributing - present)} own a slot and are not "
            "modules under preflight/."
        )

    @pytest.mark.parametrize(("source", "expected"), [
        ("from rheplicant.config.preflight import document", {"document"}),
        ("from rheplicant.config.preflight import document, model",
         {"document", "model"}),
        ("from rheplicant.config.preflight import document as _d", {"document"}),
        ("from rheplicant.config.preflight import document as _document_checks",
         {"document"}),
        ("from . import document", {"document"}),
        ("import rheplicant.config.preflight.document", {"document"}),
        ("# from rheplicant.config.preflight import document", set()),
        ('"""from rheplicant.config.preflight import document."""', set()),
        ("from rheplicant.config.findings import Finding", set()),
        ("def preflight(document):\n"
         "    from rheplicant.config.preflight import document as _d\n", set()),
        ("if True:\n"
         "    from rheplicant.config.preflight import document\n", set()),
    ], ids=["plain", "several", "aliased", "the-shipped-alias", "relative",
            "import-form", "commented", "in-a-docstring", "another-package",
            "in-a-function", "in-a-branch"])
    def test_the_matcher_reads_the_import_and_not_a_mention_of_one(
            self, source, expected):
        """ANTI-VACUITY, and the reason this is ``ast`` and not ``grep``.

        Every case here is a mutation the test above would otherwise pass
        with -- the commented and docstring forms make a missing import read
        as present, and the last one makes an unrelated import read as a check
        module.

        **``in-a-function`` is the one that shipped broken.**  A call-time
        import inside ``preflight()`` satisfied this matcher and made deleting
        a real foot import green across the whole suite; it is spelled here
        exactly as it was written.  ``in-a-branch`` is its sibling: a
        conditional import is module-level in the AST but not unconditional at
        package import, and a guard that counted it would answer "imported" for
        a module that may not be.
        """
        assert _foot_imports(source) == expected
