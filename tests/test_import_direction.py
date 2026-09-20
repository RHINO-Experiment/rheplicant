"""Which package may import which, for every package, in both directions.

``tests/core/test_layering.py`` guards one rule -- core stays domain-agnostic
-- by scanning core's own files for four forbidden prefixes. It is the right
rule and it covers one package out of six; nothing said what the other five
may do, and nothing said that ``_rheplicant_bootstrap``, which sits under all
of them, may import none of them.

Measured by AST over ``src/``, including function-local imports, which matter
here: several modules import lazily to break a cycle at module scope, and an
edge that exists only inside a function is still an edge. A text scan would
also see the four prefixes inside a docstring; this does not.

**Dynamic imports are read too, and the first version of this file did not
read them.** It asserted that ``_rheplicant_bootstrap`` imports nothing from
this project, and it PASSED -- while
``execution_environment.py::prepare_execution_environment`` called
``importlib.import_module("rheplicant.config.orchestration")``. An ``ast``
walk over ``Import`` and ``ImportFrom`` cannot see that, so the guard was
green about a property that was false in exactly the way it could not look.
Stage 1 had the edge on record as A1-2; what was missing was a check that
could fail on it. Literal string arguments to ``import_module`` and
``__import__`` now count as edges.

A dynamic import whose argument is NOT a literal cannot be resolved here at
all, and three exist: the plugin loader, ``config/hatch.py``'s target import
and postflight's discovery. Each takes a name from a document or from the
filesystem, so there is no edge to check -- what governs them is the audited
plugin protocol and ``import_target``'s own refusals, not this file.

The stack, measured rather than declared:

    _rheplicant_bootstrap      nothing statically; ONE pinned dynamic edge
    rheplicant.core            -> bootstrap
    rheplicant.radio           -> core
    rheplicant.inference       -> core
    rheplicant.config          -> bootstrap, core, radio, inference
    rheplicant.gui             -> bootstrap, core, config, radio

Two properties of that shape are worth naming because they are easy to lose
and nothing else asserts them:

* **radio and inference do not import each other.** A forward model and a
  likelihood layer are siblings; the moment one reaches for the other, the
  only way to use either is to have both.
* **gui does not import inference.** It reaches config, and config reaches
  inference. A GUI that imported the likelihood layer directly would pull the
  sampler into a web server for the sake of a form.

**Asserted in both directions**, like the config boundary and the private-name
allowlist. An allowed edge nobody uses is deleted, because a standing
permission is how a dependency arrives with nothing in the review to say it
began.
"""

from __future__ import annotations

import ast
import collections
import pathlib

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1] / "src"

#: The bottom of the stack. It is read before the package is importable, so it
#: may not depend on the package at all -- that is what makes it the bottom
#: rather than merely the first.
BOOTSTRAP = "_rheplicant_bootstrap"

#: package -> the packages it may import. Both directions are asserted, so
#: this is the whole truth and not a ceiling.
ALLOWED: dict[str, frozenset[str]] = {
    BOOTSTRAP: frozenset({"rheplicant.config"}),  # dynamic only; see BOOTSTRAP_UPWARD
    "rheplicant": frozenset({"rheplicant.core"}),
    "rheplicant.core": frozenset({BOOTSTRAP}),
    "rheplicant.radio": frozenset({"rheplicant.core"}),
    "rheplicant.inference": frozenset({"rheplicant.core"}),
    "rheplicant.config": frozenset(
        {
            BOOTSTRAP,
            "rheplicant.core",
            "rheplicant.radio",
            "rheplicant.inference",
        }
    ),
    "rheplicant.gui": frozenset(
        {
            BOOTSTRAP,
            "rheplicant.core",
            "rheplicant.config",
            "rheplicant.radio",
        }
    ),
}

#: WHICH bootstrap modules are the command half (A1-2).
#:
#: The prose below has named two layers since this file was written and never
#: said which module is in which, so the only checkable part of A1-2 was the
#: single upward edge. This is the membership, and it is the non-derivable
#: half: everything else about these modules -- their size, their imports --
#: can be measured, and which side of the seam they belong on cannot.
#:
#: The seam is what a module costs AT IMPORT. A foundation module is read
#: before ``rheplicant`` is importable and must stay that way; a command
#: module drives the package once it is, and may reach for it at call time.
BOOTSTRAP_COMMAND: frozenset[str] = frozenset(
    {
        BOOTSTRAP,
        f"{BOOTSTRAP}.__main__",
        f"{BOOTSTRAP}.cli",
        f"{BOOTSTRAP}.entry",
        f"{BOOTSTRAP}.execution_environment",
        f"{BOOTSTRAP}.gui_worker",
        f"{BOOTSTRAP}.script",
    }
)

#: The package root is in the command half by ROLE and not by cost, and it is
#: the one member a foundation module may import.
#:
#: It has to be: importing any submodule runs it first, so a rule saying no
#: foundation module may reach the command half would be false the moment
#: anything imported anything. What makes the exemption safe is that the root
#: defers BOTH of its own imports into function bodies --
#: ``test_the_package_root_costs_nothing_at_import`` is the assertion that
#: keeps it true, so the exemption proves itself rather than being asserted.
BOOTSTRAP_ROOT_IS_FREE = BOOTSTRAP

#: The bootstrap's one upward edge, and the only one it may have.
#:
#: The bootstrap is two layers in one package (A1-2): a foundation that is read
#: before ``rheplicant`` is importable, and a COMMAND half that drives the
#: package once it is. The command half cannot import ``config`` at module
#: scope without dragging the whole package -- and JAX -- into a bare
#: ``--help``, so it reaches for it at call time instead. That is the inversion
#: being pinned: allowed, singular, and named here so a second one is a red
#: test rather than a precedent.
BOOTSTRAP_UPWARD: dict[tuple[str, str], str] = {
    (
        "_rheplicant_bootstrap/execution_environment.py",
        "rheplicant.config.orchestration",
    ): "establish_runtime() hands back the orchestration module the command "
    "half then drives; importing it at module scope would put JAX behind "
    "`rheplicant --help`, which tests/config/test_entry_order.py forbids",
}

#: The one cycle this project accepts, from the inversion above.
PINNED_CYCLE = (BOOTSTRAP, "rheplicant.config")

#: Edges that are allowed and carry a cost worth stating at the edge itself.
NOTED = {
    (
        "rheplicant.core",
        BOOTSTRAP,
    ): "core.errors and core.capability re-export from the bootstrap so the "
    "layers above import them the ordinary way; two edges, and DESIGN.md's "
    "'core graduates by moving one directory' means moving these two with it",
    (
        "rheplicant.gui",
        "rheplicant.radio",
    ): "the GUI server loads JAX through this edge (A1-10). It is allowed and "
    "it is not free: a form that needs operator vocabulary pays for the "
    "array library to answer it",
}


def _package_of(module: str) -> str:
    parts = [part for part in module.split(".") if part != "__init__"]
    if not parts:
        return ""
    if parts[0] == BOOTSTRAP:
        return BOOTSTRAP
    if parts[0] == "rheplicant":
        return f"rheplicant.{parts[1]}" if len(parts) > 1 else "rheplicant"
    return ""


def _edges() -> dict[str, frozenset[str]]:
    """``package -> the packages it imports``, over every module in ``src/``."""
    found: dict[str, set[str]] = collections.defaultdict(set)
    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        here = _package_of(relative[:-3].replace("/", "."))
        if not here:
            continue
        found.setdefault(here, set())
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                there = _package_of(module)
                if there and there != here:
                    found[here].add(there)
            for module in _dynamic_targets(node):
                there = _package_of(module)
                if there and there != here:
                    found[here].add(there)
    return {key: frozenset(value) for key, value in found.items()}


def _dynamic_targets(node: ast.AST) -> list[str]:
    """Literal module names handed to ``import_module`` or ``__import__``.

    A non-literal argument yields nothing: it cannot be resolved without
    running the program, and the three that exist take a name from a document
    or the filesystem rather than naming a package at all.
    """
    if not isinstance(node, ast.Call):
        return []
    function = node.func
    name = (
        function.attr
        if isinstance(function, ast.Attribute)
        else function.id
        if isinstance(function, ast.Name)
        else ""
    )
    if name not in {"import_module", "__import__"} or not node.args:
        return []
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return [first.value]
    return []


def test_the_bottom_of_the_stack_imports_nothing_at_module_scope():
    """No STATIC edge out of the bootstrap, in any direction.

    This is the property the stack rests on: the CLI entry point reads the
    bootstrap before ``rheplicant`` is importable, and
    ``tests/config/test_entry_order.py`` depends on that to keep JAX out of a
    bare ``--help``. A module-scope import of any part of the package would
    break it at import time, which is why the one real edge is deferred to a
    call and pinned separately below.
    """
    static: set[str] = set()
    for path in sorted((SRC / BOOTSTRAP).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                if _package_of(module) not in ("", BOOTSTRAP):
                    static.add(f"{path.relative_to(SRC)} -> {module}")
    assert not static, (
        f"{BOOTSTRAP} imports the package at module scope: {sorted(static)}. "
        "It is read before `rheplicant` is importable, so this is not a "
        "layering preference -- it is what makes the entry point work"
    )


def test_the_bootstrap_has_exactly_one_upward_edge():
    """And it is the pinned one, at the pinned call site.

    The first version of this file asserted that the bootstrap imports nothing
    from the project and passed, because it read only ``Import`` nodes and the
    edge is an ``importlib.import_module`` with a literal argument. The
    assertion was green about something false. What makes it able to fail now
    is reading that call; what makes it USEFUL is naming the one edge, so a
    second one is a red test rather than a precedent set by the first.
    """
    found = set()
    for path in sorted((SRC / BOOTSTRAP).rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            for module in _dynamic_targets(node):
                if _package_of(module) not in ("", BOOTSTRAP):
                    found.add((str(path.relative_to(SRC)), module))
    assert found == set(BOOTSTRAP_UPWARD), {
        "unpinned": sorted(found - set(BOOTSTRAP_UPWARD)),
        "pinned and gone": sorted(set(BOOTSTRAP_UPWARD) - found),
    }
    for reason in BOOTSTRAP_UPWARD.values():
        assert len(reason) > 60, f"an inversion needs a reason, not a label: {reason!r}"


@pytest.mark.parametrize("package", sorted(ALLOWED), ids=sorted(ALLOWED))
def test_each_package_imports_exactly_what_it_is_allowed_to(package):
    """Both directions: no new edge, and no allowance nobody uses."""
    live = _edges().get(package, frozenset())
    allowed = ALLOWED[package]
    new = sorted(live - allowed)
    unused = sorted(allowed - live)
    assert not new, (
        f"{package} now imports {new}, which the stack does not allow. If the "
        "dependency is right, the layering changed and this table should say "
        "so in the same commit"
    )
    assert not unused, (
        f"{package} is allowed to import {unused} and does not. Delete the "
        "allowance: a permission nobody uses is how a dependency arrives with "
        "nothing in the review to say it began"
    )


def test_no_two_packages_import_each_other():
    """A cycle makes two packages one, whatever the directory listing says.

    Checked separately from the table because the table could allow one by
    accident -- two entries, each naming the other, each looking reasonable
    alone.

    One is accepted and named: ``config`` imports the bootstrap at module
    scope, and the bootstrap's command half imports ``config.orchestration``
    at call time. That is the inversion A1-2 recorded, and it was invisible to
    this file until the scanner learned to read dynamic imports.
    """
    live = _edges()
    cycles = sorted(
        {
            tuple(sorted((here, there)))
            for here, reached in live.items()
            for there in reached
            if here in live.get(there, frozenset())
        }
    )
    assert cycles == [PINNED_CYCLE], (
        f"the package cycles are {cycles}; the only one this project accepts "
        f"is {PINNED_CYCLE}, which is the bootstrap's command half driving the "
        "package at call time (see BOOTSTRAP_UPWARD). A cycle makes two "
        "packages one, whatever the directory listing says"
    )


def test_the_two_domain_layers_stay_siblings():
    """radio and inference do not import each other.

    A forward model and a likelihood layer are siblings. The moment one
    reaches for the other, using either means having both -- and the seam this
    package is built around, an instrument model that a Bayesian layer can be
    pointed at, stops being a seam.
    """
    live = _edges()
    assert "rheplicant.inference" not in live.get("rheplicant.radio", frozenset()), (
        "rheplicant.radio now imports rheplicant.inference. The forward model "
        "must be usable without the likelihood layer; this edge means it is not"
    )
    assert "rheplicant.radio" not in live.get("rheplicant.inference", frozenset()), (
        "rheplicant.inference now imports rheplicant.radio. The likelihood "
        "layer must be pointable at any instrument model, and an import of "
        "this one says it is pointable at exactly one"
    )


def test_every_noted_edge_is_real_and_allowed():
    """A note on an edge that no longer exists is worse than no note.

    It reads as a live cost and sends a reader looking for something that is
    not there.
    """
    live = _edges()
    for (here, there), reason in NOTED.items():
        assert there in ALLOWED.get(here, frozenset()), (
            f"{here} -> {there} carries a note but is not an allowed edge"
        )
        assert there in live.get(here, frozenset()), (
            f"{here} -> {there} carries a note and no longer exists"
        )
        assert len(reason) > 40, f"{here} -> {there} needs a reason, not a label"


def test_the_table_covers_every_package_that_exists():
    """A package absent from the table is a package with no rule.

    Without this, a new subpackage would be governed by nothing and the suite
    would stay green -- the same shape as a census that picks its own
    population.
    """
    live = sorted(_edges())
    missing = sorted(set(live) - set(ALLOWED))
    assert not missing, (
        f"{missing} exist in src/ and have no entry in the table, so nothing "
        "says what they may import"
    )


def _bootstrap_modules() -> dict[str, pathlib.Path]:
    """Every module in the bootstrap package, by dotted name."""
    found = {}
    for path in sorted((SRC / BOOTSTRAP).rglob("*.py")):
        parts = list(path.relative_to(SRC).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        found[".".join(parts)] = path
    return found


def _module_scope_imports(path: pathlib.Path) -> set[str]:
    """Modules imported at MODULE scope, which is what an import costs.

    Does not descend into function bodies -- an import there is deferred and
    costs nothing until the function runs, which is the whole mechanism the
    seam below relies on. A class body is module scope and is descended into.

    **Every ``from X import Y`` contributes ``X.Y`` as well as ``X``**, and the
    first version of this helper did not. It recorded only ``node.module``, so
    ``from _rheplicant_bootstrap import cli`` read as an import of the package
    ROOT -- which is the one member the rule below exempts. Probed by injecting
    exactly that line into a foundation module: the guard stayed GREEN. A
    matcher that cannot see the most natural spelling of the thing it forbids
    is the failure this repository keeps writing down.
    """
    targets: set[str] = set()

    def walk(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            if isinstance(child, ast.ImportFrom) and child.module and not child.level:
                targets.add(child.module)
                targets.update(f"{child.module}.{alias.name}" for alias in child.names)
            elif isinstance(child, ast.Import):
                targets.update(alias.name for alias in child.names)
            walk(child)

    walk(ast.parse(path.read_text(encoding="utf-8")))
    return targets


def test_the_command_half_is_named_and_covers_the_package():
    """Both directions, like every other table here.

    A module added to the bootstrap without being classified fails, which is
    the point: the seam is what the package is FOR, and a new module lands on
    one side of it whether or not anyone said so. Measured 2026-09-20, exactly
    one module had arrived since A1 drew the line (``capability``, the
    JAX-free level vocabulary the GUI reads) and it is foundation.
    """
    live = set(_bootstrap_modules())
    assert BOOTSTRAP_COMMAND <= live, {"named and gone": sorted(BOOTSTRAP_COMMAND - live)}
    foundation = live - BOOTSTRAP_COMMAND
    assert foundation, "the whole package cannot be the command half"
    # 44, up from A1's 37: splitting `layering.py` into its three subjects
    # plus the defensive copy's three pieces added seven foundation modules
    # on 2026-09-20 (section 3.2). The command half did not move, which is the
    # point of pinning the two numbers apart.
    assert len(BOOTSTRAP_COMMAND) == 7 and len(foundation) == 44, (
        f"A1-2 recorded a 7-module command half; this tree has "
        f"{len(BOOTSTRAP_COMMAND)} and {len(foundation)} foundation modules. "
        "Reclassify deliberately rather than letting the number drift"
    )


def test_no_foundation_module_imports_the_command_half():
    """The direction that makes the seam real.

    Without it, ``BOOTSTRAP_COMMAND`` is a comment. Measured before it was
    written: zero such edges, so this pins a property the tree already has
    rather than asking for work.
    """
    modules = _bootstrap_modules()
    reachable = BOOTSTRAP_COMMAND - {BOOTSTRAP_ROOT_IS_FREE}
    offences = []
    for name, path in modules.items():
        if name in BOOTSTRAP_COMMAND:
            continue
        for target in sorted(_module_scope_imports(path) & reachable):
            offences.append((name, target))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            for target in _dynamic_targets(node):
                if target in reachable:
                    offences.append((name, target))
    assert not offences, (
        f"foundation modules reaching the command half: {sorted(set(offences))}. "
        "The foundation is read before `rheplicant` is importable; a command "
        "module is what drives the package once it is"
    )


def test_the_package_root_costs_nothing_at_import():
    """What makes the root's exemption above safe rather than assumed.

    Importing any bootstrap submodule runs the root first, so if the root
    imported its own command modules at module scope, every foundation module
    would pay for the CLI. Both of its entry points defer into function
    bodies; this is the assertion that keeps them deferred.
    """
    root = _bootstrap_modules()[BOOTSTRAP]
    at_module_scope = _module_scope_imports(root)
    project = sorted(
        name
        for name in at_module_scope
        if _package_of(name) or name.split(".")[0] in (BOOTSTRAP, "rheplicant")
    )
    assert not project, (
        f"{BOOTSTRAP}/__init__.py imports {project} at module scope. Both of "
        "its entry points defer, which is what lets the foundation import the "
        "root without paying for the command half"
    )


def test_no_document_section_set_is_spelled_twice():
    """A10-2: one constant per rule, asserted over the literal COLLECTIONS.

    The process-owned sections had three spellings in ``src/`` and the preset
    sections two -- in different orders and different container types, which
    is how two copies of one rule stop looking like one rule. Both are public
    constants in the bootstrap now, and this keeps a fourth from appearing.

    Matched by ``ast`` on tuple, list and set literals whose members are
    exactly the set, NOT by looking for the member strings in the file. The
    first version did the latter and named six modules, every one a false
    positive: a file that mentions every document section for some other
    reason contains all five words without restating the rule. Grepping a
    name answers "does this string appear", never "is this the same rule".
    """
    from _rheplicant_bootstrap.presets import PRESET_SECTIONS
    from _rheplicant_bootstrap.process import PROCESS_SECTIONS

    owners = {
        "PROCESS_SECTIONS": (frozenset(PROCESS_SECTIONS), f"{BOOTSTRAP}/process.py"),
        "PRESET_SECTIONS": (frozenset(PRESET_SECTIONS), f"{BOOTSTRAP}/presets.py"),
    }
    offenders: dict[str, list[str]] = {}
    for path in sorted(SRC.rglob("*.py")):
        relative = str(path.relative_to(SRC))
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
                continue
            members = [
                element.value
                for element in node.elts
                if isinstance(element, ast.Constant) and isinstance(element.value, str)
            ]
            if len(members) != len(node.elts):
                continue
            for name, (expected, home) in owners.items():
                if relative != home and frozenset(members) == expected:
                    offenders.setdefault(name, []).append(f"{relative}:{node.lineno}")
    assert not offenders, (
        f"these modules write out a section set instead of importing it: "
        f"{offenders}. Import the constant -- two copies of one rule is how a "
        "new section reaches one of them and not the other"
    )
