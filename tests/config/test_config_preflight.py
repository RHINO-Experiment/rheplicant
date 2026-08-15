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
import os
import pathlib
import re
import subprocess
import sys
import time
import warnings

import pytest

import rheplicant.config as config_package
import rheplicant.config.document as document_module
import rheplicant.config.sections.runtime as runtime_module
import rheplicant.config.values as values_module
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
    + tuple(f"C{n}" for n in range(1, 18))
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


def _foot_imports(source: str) -> set[str]:
    """The modules ``preflight/__init__.py``'s foot import names.

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
    """
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and (
                (node.module or "").startswith("rheplicant.config.preflight")
                or (node.level == 1 and node.module is None)):
            found.update(entry.name for entry in node.names)
        elif isinstance(node, ast.Import):
            for entry in node.names:
                if entry.name.startswith("rheplicant.config.preflight."):
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

#: Called names no module under ``preflight/`` may write, whatever it imported
#: to get them.  Filesystem verbs first, then the entry points out of P-1.
#: This is the STATIC half of the boundary and its whole point is that it is
#: branch-independent: it reads code that never runs, which is exactly what
#: the runtime patches below cannot do.
_OUT_OF_SCOPE_CALLS = frozenset({
    "open", "read_text", "read_bytes", "iterdir", "glob", "rglob", "walk",
    "listdir", "scandir", "stat", "lstat", "exists", "is_file", "is_dir",
    "fromfile", "loadtxt", "genfromtxt", "read_map", "read_alm",
    *_OUT_OF_SCOPE_NAMES,
})


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


def _forbid_the_filesystem(monkeypatch) -> None:
    """Take away every Python-level filesystem API, all ten of them.

    The four an obvious draft carries (``builtins.open``,
    ``Path.open/read_text/read_bytes``) leave the directory walks wide open,
    and a beam is a DIRECTORY of files.  ``os.open`` is the floor under
    ``builtins.open``; ``os.stat`` is the floor under ``Path.exists``,
    ``Path.is_file`` and ``os.path.exists`` -- measured, patching it catches
    all three -- which is the shape a check asking "is the beam file there?"
    would take.

    **Two of the ten are redundant on this interpreter, measured.**  Deleting
    the ``Path.iterdir`` and ``Path.glob`` patches leaves every test in this
    module green, because CPython 3.12's ``Path.iterdir`` delegates to
    ``os.listdir`` and its ``glob`` to ``os.scandir``, both of which are
    patched below.  They are kept because that delegation is an implementation
    detail that has already been rewritten once between versions -- but the
    two anti-vacuity cases named for them prove the FLOOR patch works, not
    these two, and saying otherwise would be a claim no run defends.
    """
    def refuse_to_open(*args, **kwargs):
        raise _GuardTripped("the pre-flight pass touched the filesystem")

    monkeypatch.setattr("builtins.open", refuse_to_open)
    monkeypatch.setattr(pathlib.Path, "open", refuse_to_open)
    monkeypatch.setattr(pathlib.Path, "read_text", refuse_to_open)
    monkeypatch.setattr(pathlib.Path, "read_bytes", refuse_to_open)
    monkeypatch.setattr(pathlib.Path, "iterdir", refuse_to_open)
    monkeypatch.setattr(pathlib.Path, "glob", refuse_to_open)
    monkeypatch.setattr(os, "open", refuse_to_open)
    monkeypatch.setattr(os, "listdir", refuse_to_open)
    monkeypatch.setattr(os, "scandir", refuse_to_open)
    monkeypatch.setattr(os, "stat", refuse_to_open)


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
    "several-runs",
})


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
        found = preflight(preflight_document()).refusals()
        assert calls == [1]
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
        fault.  Its anti-vacuity partner is the next test."""
        assert {slot.split(".", 1)[0] for slot in CHECKS} <= set(_schema_ids())

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
        assert len(found) == 78, found[:5]
        assert len(set(found)) == len(found), "an id is declared twice"
        assert found[0] == "A1" and "A52" in found and "C17" in found
        from_spec = _schema_ids_from_the_spec()
        if from_spec is None:
            pytest.skip("schema §6 spec absent -- docs/superpowers/ is gitignored")
        assert from_spec == found


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

        load_document(preflight_document(), variant="unity_gain")
        assert seen == [1.0]        # 1.1 is the unpatched value

    def test_run_document_inherits_the_hook_and_adds_none_of_its_own(
            self, registry):
        """Measured: three runs across two distinct variants run the pass
        TWICE -- once per variant-applied document, which is what
        ``configured(run.variant)`` memoises (``runs.py:152-156``).

        Kills both directions.  A ``preflight`` call added to ``run_document``
        makes it three or four and runs the pass on the UNPATCHED document,
        which is a document no variant describes; a hook that never fires makes
        it nought.
        """
        seen = []

        @register("A2")
        def _watch(document):
            seen.append(1)
            return ()

        doc = preflight_document(runs=[
            {"kind": "forward", "name": "a"},
            {"kind": "forward", "name": "b"},
            {"kind": "forward", "name": "c", "variant": "unity_gain"},
        ])
        run_document(doc)
        assert len(seen) == 2


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

    * **the filesystem** -- ten Python-level APIs taken away.  The blind spot
      is a read that reaches the OS entirely inside a C extension, and it is
      NARROWER than a draft of this list claimed: measured on this build with
      all ten patched, ``numpy.load``, ``numpy.fromfile`` and
      ``numpy.loadtxt`` are **all caught** -- every one of them goes through
      ``builtins.open`` -- and the one that is **not** is ``h5py.File``.
      ``h5py`` is on §0's no-import list, which is what stops it; the numpy
      names are on the STATIC call ban below anyway, because that ban reads a
      branch no document takes and these patches cannot.
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

    @pytest.mark.parametrize("read", [
        lambda: open(__file__).close(),
        lambda: pathlib.Path(__file__).open().close(),
        lambda: pathlib.Path(__file__).read_text(),
        lambda: pathlib.Path(__file__).read_bytes(),
        lambda: list(pathlib.Path(__file__).parent.iterdir()),
        lambda: list(pathlib.Path(__file__).parent.glob("*.py")),
        lambda: os.close(os.open(__file__, os.O_RDONLY)),
        lambda: os.listdir(str(_HERE)),
        lambda: list(os.scandir(str(_HERE))),
        lambda: pathlib.Path(__file__).exists(),
    ], ids=["builtins-open", "path-open", "path-read-text", "path-read-bytes",
            "path-iterdir", "path-glob", "os-open", "os-listdir", "os-scandir",
            "path-exists"])
    def test_that_guard_can_still_see_each_way_of_reading(
            self, registry, monkeypatch, read):
        """ANTI-VACUITY, one case per patched API.  Without it, a target that
        stopped being the one the code reaches -- ``open`` moving inside a C
        extension, ``Path.read_text`` bypassed, a new ``os`` spelling -- makes
        the guard green forever, and a four-patch draft would pass the four it
        happens to name while the six directory and existence routes ran wide
        open."""
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

    def test_no_module_here_imports_its_way_out_of_the_phase(self):
        """The static half, which the monkeypatches cannot reach: a module that
        writes ``from rheplicant.config.values import resolve_value`` at its
        head holds the function by value and the patch above never sees the
        call.  ``rheplicant.config.document`` is banned outright -- it imports
        this package for the hook, so importing it back closes a cycle."""
        offenders = {
            path.name: sorted(found)
            for path in sorted(_PREFLIGHT_DIR.glob("*.py"))
            if (found := _out_of_scope_imports(path.read_text()))
        }
        assert offenders == {}, (
            f"{offenders} are imported under preflight/. Each is an entry "
            "point out of P-1: a builder, the file resolver, the value-node "
            "resolver, a path resolved against a built twin, or the document "
            "module itself (which imports this package and would close a "
            "cycle)."
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
            for path in sorted(_PREFLIGHT_DIR.glob("*.py"))
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
            "numpy-fromfile", "h5py-File-NOT-caught", "resolve-value",
            "build-runtime", "on-a-branch-never-taken", "commented",
            "in-a-docstring", "the-sanctioned-sweep",
            "the-sanctioned-path-parser", "class-introspection"])
    def test_that_call_guard_reads_calls_and_not_mentions_of_them(
            self, source, expected):
        """ANTI-VACUITY, both directions.

        ``on-a-branch-never-taken`` is the whole point: the call is
        unreachable and is still read, which is what the runtime patches
        cannot do.

        ``h5py-File-NOT-caught`` is the declared blind spot, written as a case
        rather than as a sentence: ``File`` is too generic a name to ban
        without false positives, so an ``h5py.File`` read is caught by neither
        this guard nor the filesystem patches -- measured, with all ten
        patched it opens the file anyway, and it is the only one of the four
        C-extension readers probed that does.  It is stopped instead by §0's
        rule that no module here may import ``h5py`` at all, which the
        subprocess test below enforces.

        The last three are the other direction: §2.5's mandated helpers and
        §2.4's sanctioned class introspection must not be flagged.
        """
        assert _out_of_scope_calls(source) == expected

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
    ], ids=["plain", "several", "aliased", "the-shipped-alias", "relative",
            "import-form", "commented", "in-a-docstring", "another-package"])
    def test_the_matcher_reads_the_import_and_not_a_mention_of_one(
            self, source, expected):
        """ANTI-VACUITY, and the reason this is ``ast`` and not ``grep``.

        Every case here is a mutation the test above would otherwise pass
        with -- the commented and docstring forms make a missing import read
        as present, and the last one makes an unrelated import read as a check
        module.
        """
        assert _foot_imports(source) == expected
