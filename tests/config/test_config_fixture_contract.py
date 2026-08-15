"""One place builds a test document, and this is what says so.

``tests/config/exit_helpers.py`` was designated "the only place any is built"
during Plan 2C, in prose.  Prose did not hold: four test modules written after
it landed still build their own document on top of
``test_config_document.synthetic_document()``, and every test on those
documents lost the same thing.

**What is lost.** :func:`~tests.config.exit_helpers._repaired` keeps
``synthetic_document``'s stochastic ``noise`` node in ``model:`` and repairs
it away in ``inference.twin.without:``, so ``built.twin`` and
``built.inference.fit_twin`` stay different objects.  A builder that deletes
the node from ``model:`` instead gets the same prediction and the same
observed data -- and makes the two twins the SAME OBJECT, after which no test
on that document can tell an executor that reaches for one from an executor
that reaches for the other.

**Measured on this tree, at Plan 2D's Task 10.**  Forcing ``build_fit_twin``
to the identity -- the runtime equivalent of every executor reading
``built.twin`` -- and running the four modules gives, as
``module: collected / detect``::

    predict 40/11  plan 20/0  estimators 21/2  diagnostics 48/12

(predict's eleven are 9 failures and 2 class-fixture errors.)  **25 of 129.**
Split by which document each test ran on rather than by module -- the split is
what the number is about, and it was taken by spying on each module's own
builder while the mutation was in place:

* **93 tests run on a rolled-own document and 4 of them notice** -- so
  **89 cannot**.  The four are hand-written compensations, and they are in two
  of the four modules: predict's two ``..._pushes_the_repaired_fit_twin``
  tests pass ``twin=TWIN_AT_65``, and estimators'
  ``test_fisher_evaluates_the_repaired_fit_twin`` and
  ``test_the_trainable_route_needs_no_parameters`` write an
  ``inference.twin.replace`` into the document by hand.  ``plan`` and
  ``diagnostics`` buy none of it back.
* **36 run on a shared ``exit_helpers`` / ``posterior_helpers`` document in
  those same modules and 21 notice** -- including all 12 of diagnostics'
  detections, which are ``TestIdentifiability``'s and
  ``TestScoreDirections``' **12 of 25**.

2C measured "~82" and this plan's own draft measured 16 of 115, both before
Plan 2D's Task 9 added fourteen tests to the predict module.  The numbers
above supersede them and were re-derived here rather than transcribed.

**Why an allow-list rather than a migration.** The repair needs the ``noise``
node back in ``model:``, and that changes what these documents MEAN, not just
where they are built.

Counted rather than generalised: **exactly ONE of the four modules runs
``forward`` runs** -- ``test_config_exits_predict.py``, which declares
``FORWARD = {"name": "fwd", "kind": "forward"}`` at its line 13 and drives it
at six sites, and pins those products as the NOISELESS signal.  ``exits.py``
evaluates ``built.twin(built.state)`` with no fit/full opt-down of any kind,
so putting the ``noise`` node back makes those products noise realisations and
moves every number that module pins (0.0078557, 0.0014147, and the ``argmax``
indices 4 and 1 that four of its assertions read off a forward product).
``test_config_exits_plan.py`` and ``test_config_exits_diagnostics.py`` contain
the word ``forward`` nowhere at all, and ``test_config_exits_estimators.py``
only inside a refusal string (``match="forward and optimize"``).  **An earlier
draft of this docstring said "three of the four"; measured, it is one.**

The scope argument is what carries the decision, and it does not need that
generalisation: migrating means re-deriving one module's pinned numbers from
the fit twin AND re-checking 93 tests across four modules for a twin identity
they were never written against, in the last task-pair of an eleven-task plan.
2C's own fixture repair was safe because it could verify the OBSERVED arrays
bit-identical (``observed.twin: fit`` is a documented opt-down); a forward run
offers nothing to hold constant.  So the four are recorded, the list is
asserted to shrink and never to grow, and the migration is Plan 3's.

**The migration recipe, so it need not be re-derived.**  One module at a time:
record every ``forward`` product the module pins; move the builder into
``exit_helpers`` with ``model=`` carrying
:data:`~tests.config.exit_helpers.MODEL_NOISE`; re-run.  A pinned number that
moves was measured on the MODEL twin and has to be re-derived from the fit
twin -- not nudged into tolerance.

**How the counts here were taken, so they can be re-taken.**  A pytest plugin
that (a) rebinds ``rheplicant.config.sections.inference.build_fit_twin`` to a
wrapper which calls the real one for its refusals and returns ``(twin, ())``,
and (b) wraps each module's own builder to record which test called it; then,
per module, the tests that ran on the local builder and did NOT fail.  **The
rebind must name the IMPORTING module**: ``inference.py`` imports the function
by value (``from ...sections.twin import build_fit_twin``), so patching
``sections.twin`` instead leaves the live call site untouched and measures
nothing, with no error -- verified, the detecting test stays green.  The
plugin is not shipped: it measures the state Plan 3 is going to remove.

**The matcher works at MODULE scope, and that is a correction.**  Its first
draft required the ``synthetic_document()`` call and the ``["inference"]``
write to be in the SAME function.  Measured: hoisting predict's three fixture
lines into a ``_base()`` helper -- a pure refactor, its own 40 tests still at
exit 0 and its 27 blind tests still blind -- dropped the module out of the
census, and the guard then reported it as *retired* and told the editor to
delete its row.  A guard that instructs you to stop watching a module that
still offends is worse than no guard, so the pairing is now per FILE.  What
that also buys, in one change: a module-level ``DOC = synthetic_document()``
edited at module scope (the walk was over ``FunctionDef`` only), a qualified
``test_config_document.synthetic_document()``, an aliased import, and a
builder split across any number of functions.

**What it still cannot see** -- recorded because a matcher whose reach is
guessed is a matcher that reports absence as innocence:

* a module that imports a rolled-own builder from ANOTHER test module.  The
  offence is recorded against the module that DEFINES it, which is where the
  fix goes, so this is a naming gap rather than a hole.
* a document built with no literal ``"inference"`` key at all -- a variable
  key (``doc[key] = ...``), a ``**`` of a block assembled elsewhere, a
  comprehension over section names.
* a factory reached by anything other than its own name: ``functools.partial``,
  ``globals()[...]``, a plain rebinding (``factory = synthetic_document``), or
  a document arriving as a parameter and edited by a fixture that never calls
  a factory itself.  A qualified call, an aliased import and a method call all
  DO resolve -- ``mod.f()`` and ``self.f()`` are read by their attribute name.

The property test in the second class is what covers the helper modules
themselves against all of these, because it reads the built object rather
than the source.
"""

import ast
import importlib
import inspect
import pathlib

import pytest

from rheplicant.config.document import load_document
from tests.config.test_config_document import synthetic_document

#: ``tests/config/``.
_HERE = pathlib.Path(__file__).resolve().parent

#: The raw fixture, by name.  Building an ``inference:`` block on THIS is the
#: offence 2C measured.
_RAW = "synthetic_document"

#: The module that defines the fixture, DERIVED from the function rather than
#: named, so renaming the file cannot open a hole.  It is sanctioned for one
#: reason that applies to nothing else: a test OF the fixture has to build on
#: the fixture, and it cannot import a document from the helper modules
#: instead without inverting the dependency -- ``exit_helpers`` imports
#: ``synthetic_document``, never the other way.  Measured, its one such site
#: (``test_an_inference_section_builds_on_the_run``) writes
#: ``twin: {without: [noise]}`` by hand and so keeps the two twins distinct;
#: measured again with that one line dropped, they collapse.  The exemption is
#: therefore safe and one line thick, which is worth knowing before a second
#: document is added there.
_FIXTURE_HOME = pathlib.Path(inspect.getsourcefile(synthetic_document)).name

#: The four modules that build their own document anyway, with the number of
#: their tests that consequently cannot tell the two twins apart.  Measured
#: when Plan 2D landed, by the recipe in this module's docstring (an earlier
#: draft named a commit SHA, which an amend then made unresolvable); the
#: values are
#: COMMENTS -- no assertion reads them, and they move whenever a test is added
#: to one of these modules -- while the keys are the contract.  They are
#: recorded at all because "86 of 90" travelled through two plans as prose and
#: was two different numbers by the time anyone re-took it.
_KNOWN_OFFENDERS = {
    "test_config_exits_predict.py": 27,      # of 29 on its own document;
                                             # the other 11 of its 40 run on
                                             # shared ones and 9 of those bite
    "test_config_exits_plan.py": 20,         # of 20; none notice
    "test_config_exits_estimators.py": 19,   # of 21; 2 declare a twin
    "test_config_exits_diagnostics.py": 23,  # of the 23 on its own
                                             # condition_document; the OTHER
                                             # 25 are shared and 12 notice
}

#: The list may SHRINK and may never GROW.  Without this, the ceiling test is
#: silenced by adding one line to the table above -- measured, a fifth module
#: plus its row is exit 0 -- and "the next fan-out does not add a fifth", the
#: whole contract, would rest on the advisory prose in a failure message.
_OFFENDER_CEILING = 4

#: How many ``*_document`` builders each helper module defines, measured here.
#: A FLOOR, not an equality: adding a builder is free and the property test
#: covers it on the day it lands.  Every helper module that defines a builder
#: must appear, which is what stops a row being deleted to shrink the walk.
#: ``preflight_helpers`` joined at Plan 3A's Task 2 with one: its
#: ``preflight_document`` DELEGATES to ``exit_helpers.conjugate_document``, so
#: the repair travels with it and the property below holds it to the same
#: standard as a builder that rolls its own -- measured when the row landed,
#: and measured again without the row, where this file goes red naming
#: ``['preflight_helpers']``.
_BUILDER_FLOOR = {"exit_helpers": 8, "posterior_helpers": 4,
                  "preflight_helpers": 1}

#: A run every builder in every helper module accepts as its first argument.
#: The property below is about the document's two TWINS, which no run touches,
#: so any run does; builders that take none are called with none.
_FORWARD = {"kind": "forward"}


def _helper_modules() -> dict[str, object]:
    """The modules a document may be built in, DISCOVERED by name.

    ``tests/config/*_helpers.py``.  Two of the three define documents today
    (``exit_helpers`` and ``posterior_helpers``, split apart by Plan 2D's
    Task 7 for the 800-line ceiling); ``inference_helpers`` builds twins and
    states and no document at all, and is included because the rule is the
    file name, not a list someone maintains.

    **Discovered rather than listed, and that is load-bearing.**  A tuple of
    module objects can be shortened: dropping ``posterior_helpers`` from it
    takes four builders out of the property walk below, which is exactly the
    failure this task exists to prevent, and with a hand-written floor table
    the same two-line edit removes the assertion that would have caught it.
    A glob cannot be shortened without deleting or renaming a file.
    """
    return {path.stem: importlib.import_module(f"tests.config.{path.stem}")
            for path in sorted(_HERE.glob("*_helpers.py"))}


def _sanctioned() -> set[str]:
    """The files that may build a test document."""
    return {f"{stem}.py" for stem in _helper_modules()} | {_FIXTURE_HOME}


def _factories() -> frozenset[str]:
    """Every helper name that hands back a document, derived not listed.

    ``*_document`` only.  The ``*_built`` siblings are NOT factories for this
    purpose: they return the loaded object, and measured,
    ``nuts_built()["inference"] = ...`` raises ``TypeError: 'ConfiguredRun'
    object does not support item assignment``.  A route-B case written
    against one would pin a shape that cannot occur.
    """
    names = set()
    for module in _helper_modules().values():
        names.update(name for name, value in vars(module).items()
                     if inspect.isfunction(value)
                     and value.__module__ == module.__name__
                     and name.endswith("_document"))
    return frozenset(names)


def _sites(node, where: str = "<module>"):
    """Every node in ``node``, paired with the function that encloses it."""
    for child in ast.iter_child_nodes(node):
        inner = (child.name
                 if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                 else where)
        yield child, inner
        yield from _sites(child, inner)


def _called(tree: ast.Module) -> set[str]:
    """Every name called in the module, aliases resolved.

    ``f()``, ``mod.f()`` and ``from x import f as g`` + ``g()`` all count.
    The last is why the import table is read: an alias is a one-line evasion
    of a name match, and nothing else in this file would notice it.
    """
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for entry in node.names:
                if entry.asname:
                    aliases[entry.asname] = entry.name.rsplit(".", 1)[-1]
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            names.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            names.add(node.func.attr)
    return names | {aliases[name] for name in names if name in aliases}


def _depth_of(target) -> int | None:
    """``doc["inference"]`` -> 1, ``doc["inference"]["twin"]`` -> 2."""
    layers = 0
    probe = target
    while isinstance(probe, ast.Subscript):
        if (isinstance(probe.slice, ast.Constant)
                and probe.slice.value == "inference"):
            return layers + 1
        layers += 1
        probe = probe.value
    return None


def _writes(tree: ast.Module) -> list[tuple[str, str, int, bool]]:
    """``(kind, enclosing function, depth, writes an empty block)`` per site.

    Four kinds, because there are four ways in this repository's own idiom to
    put an ``inference:`` block somewhere: assignment (at any nesting depth),
    deletion, a ``{**base, "inference": ...}`` literal -- which
    ``test_config_document.py:91`` already uses -- and ``dict.update``.
    """
    found = []
    for node, where in _sites(tree):
        targets = []
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, (ast.AugAssign, ast.AnnAssign)):
            targets = [node.target]
        for target in targets:
            depth = _depth_of(target)
            if depth is not None:
                value = getattr(node, "value", None)
                blank = isinstance(value, ast.Dict) and not value.keys
                found.append(("assign", where, depth, blank))
        if isinstance(node, ast.Delete):
            for target in node.targets:
                depth = _depth_of(target)
                if depth is not None:
                    found.append(("delete", where, depth, False))
        if isinstance(node, ast.Dict) and any(key is None for key in node.keys):
            if any(isinstance(key, ast.Constant) and key.value == "inference"
                   for key in node.keys):
                found.append(("star", where, 1, False))
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "update"):
            for argument in node.args:
                if isinstance(argument, ast.Dict) and any(
                        isinstance(key, ast.Constant)
                        and key.value == "inference"
                        for key in argument.keys):
                    found.append(("update", where, 1, False))
    return found


def _rolls_its_own(path: pathlib.Path) -> list[str]:
    """Where in ``path`` a test document is built instead of imported.

    Returns the enclosing function names (or ``"<module>"``), so the evidence
    survives being reported.  **The pairing is per FILE, not per function** --
    see this module's docstring for the refactor that made it so.

    Two routes, because there are two ways to end up with an unrepaired
    ``inference:`` block and closing one would leave its twin open:

    **Route A** -- the module calls :data:`_RAW` anywhere and writes an
    ``inference`` block anywhere.  On the raw fixture there is no repaired
    block to damage: any block the module puts there is its own, at any
    nesting depth and by any of the four write forms.

    **Route B** -- the module calls a helper builder and REPLACES the block it
    got back: a depth-1 assignment of something non-empty.  Measured,
    ``doc = conjugate_document(...)`` then ``doc["inference"] = ONE_LATENT``
    makes the two twins the same object just as thoroughly as route A does, so
    a docstring claiming that a module which "merely imports a helper and
    edits the result keeps the repair" is false -- an earlier draft of this
    one said exactly that.

    **Depth and emptiness are what keep route B honest**, and both are
    measured rather than assumed:

    * a NESTED write (``doc["inference"]["twin"] = ...``) edits inside the
      repaired block rather than replacing it, and flagging it would be
      backwards: ``test_config_exits_estimators.py:88`` is that shape and it
      exists to BUY the discrimination BACK, while
      ``test_config_exits_conjugate.py:407`` and four sites in
      ``test_config_exits_npe.py`` use it to reach a refusal.
    * ``doc["inference"] = {}`` blanks the block to reach the "no
      inference.parameters" refusal -- both copies of
      ``test_without_parameters_it_is_refused`` do exactly this -- and hands
      the document straight to ``run_document``.  A blank is not a
      rolled-own block, and neither is ``del doc["inference"]``.
    """
    tree = ast.parse(path.read_text())
    writes = _writes(tree)
    if not writes:
        return []
    called = _called(tree)
    if _RAW in called:
        return sorted({where for _, where, _, _ in writes})
    if called & _factories():
        return sorted({where for kind, where, depth, blank in writes
                       if kind in ("assign", "star", "update")
                       and depth == 1 and not blank})
    return []


def _census() -> dict[str, list[str]]:
    return {path.name: found
            for path in sorted(_HERE.glob("*.py"))
            if (found := _rolls_its_own(path))}


def _builders() -> dict[str, dict[str, object]]:
    """Every ``*_document`` each helper module defines, BY MODULE.

    Keyed by module rather than flattened, because the failure this whole
    task exists to prevent is a walk that silently covers one module: a flat
    mapping can be short by four and still look plentiful.
    """
    return {stem: {name: value for name, value in sorted(vars(module).items())
                   if name.endswith("_document")
                   and inspect.isfunction(value)
                   and value.__module__ == module.__name__}
            for stem, module in _helper_modules().items()}


def _build(builder) -> object:
    """``builder`` driven with a forward run if it takes one, then loaded.

    Two of the twelve builders take no argument at all
    (``trio_npe_document`` and ``joint_prior_document``, both
    ``posterior_helpers``', both deliberately parameterless), so a walk that
    passed a run to everything would die with a ``TypeError`` on exactly the
    module this guard was widened to reach.
    """
    try:
        inspect.signature(builder).bind(_FORWARD)
    except TypeError:
        return load_document(builder())
    return load_document(builder(_FORWARD))


class TestOnlyOnePlaceBuildsADocument:
    def test_the_matcher_finds_the_builders_it_is_named_for(self):
        """Anti-vacuity: the matcher still matches.

        Every other assertion in this class is of the form "nothing new was
        found".  A matcher that stopped matching -- ``synthetic_document``
        renamed, the write form changed, a module moved out of the glob --
        makes all of them true and this module a green no-op.  2C's
        ``test_the_kind_tables_are_pairwise_disjoint`` carries the same guard
        as ``len(tables) >= 2`` (``test_config_exit_support.py:83``).

        The two sanctioned builders are the anchor, and NOT the offender
        count: Plan 3 retires the four, and an anti-vacuity assertion that
        went red on the day the suite got clean would be read as "the matcher
        broke" when it means the opposite.  It is a superset assertion for the
        same reason -- ``exit_helpers`` is allowed to grow a third.
        """
        census = _census()
        assert set(census.get("exit_helpers.py", [])) >= {
            "conjugate_document", "diagnostic_document"}, census

    @pytest.mark.parametrize("source, expected", [
        # --- route A: the raw fixture, built on ---------------------------
        # The refactor that broke the first draft: call in one function,
        # write in another.
        ("""
def _base():
    doc = synthetic_document()
    doc["model"] = {}
    return doc


def document():
    doc = _base()
    doc["inference"] = {"parameters": {}}
    return doc
""", ["document"]),
        # Module scope, which a FunctionDef walk cannot see at all.
        ("""
DOC = synthetic_document()
DOC["inference"] = {"parameters": {}}
""", ["<module>"]),
        # Qualified call.
        ("""
def document():
    doc = test_config_document.synthetic_document()
    doc["inference"] = {"parameters": {}}
    return doc
""", ["document"]),
        # Aliased import.
        ("""
from tests.config.test_config_document import synthetic_document as raw


def document():
    doc = raw()
    doc["inference"] = {"parameters": {}}
    return doc
""", ["document"]),
        # The `{**base, "inference": ...}` literal.
        ("""
def document():
    return {**synthetic_document(), "inference": {"parameters": {}}}
""", ["document"]),
        # dict.update.
        ("""
def document():
    doc = synthetic_document()
    doc.update({"inference": {"parameters": {}}})
    return doc
""", ["document"]),
        # Nested, on the raw fixture: there is no repaired block to preserve,
        # so depth does not save it.
        ("""
def document():
    doc = synthetic_document()
    doc["inference"]["twin"] = {"replace": {}}
    return doc
""", ["document"]),
        # The raw fixture used without an inference block of its own.
        ("""
def test_it_loads():
    assert load_document(synthetic_document())
""", []),
        # --- route B: a helper document, replaced -------------------------
        ("""
def variant():
    doc = conjugate_document({"kind": "forward"})
    doc["inference"] = ONE_LATENT
    return doc
""", ["variant"]),
        # No `return` required -- a fixture that yields, or a test that
        # replaces the block inline, loses the repair just the same.
        ("""
def test_variant():
    doc = conjugate_document({"kind": "forward"})
    doc["inference"] = ONE_LATENT
    run_document(doc)
""", ["test_variant"]),
        # Not route B: blanking to reach the refusal.
        ("""
def test_without_parameters_it_is_refused():
    doc = diagnostic_document({"kind": "identifiability"})
    doc["inference"] = {}
    with pytest.raises(ConfigError):
        run_document(doc)
""", []),
        # Not route B: editing INSIDE the repaired block.
        ("""
def test_fisher_evaluates_the_repaired_fit_twin():
    doc = diagnostic_document({"kind": "fisher"})
    doc["inference"]["twin"] = {"replace": {}}
    run_document(doc)
""", []),
        # Not route B: deleting a key, or the block.
        ("""
def test_the_seed_is_required():
    doc = npe_document()
    del doc["inference"]["npe"]["bank"]["seed"]
    run_document(doc)
""", []),
        # Not an offence at all: a helper is used as it stands.
        ("""
def test_it_runs():
    results = run_document(conjugate_document({"kind": "forward"}))
    assert results
""", []),
    ], ids=["raw-split-across-functions", "raw-at-module-scope",
            "raw-called-qualified", "raw-imported-under-an-alias",
            "raw-star-unpacked", "raw-via-update", "raw-written-nested",
            "raw-with-no-block", "helper-block-replaced",
            "helper-block-replaced-inline", "helper-block-blanked",
            "helper-block-edited-inside", "helper-key-deleted",
            "helper-used-as-is"])
    def test_the_matcher_reads_both_routes_and_neither_more(
            self, tmp_path, source, expected):
        """The matcher's reach, pinned on synthetic sources.

        Every case here is a mutation this module would otherwise ship green,
        and the first eight are one bug: the pairing used to be per function,
        so hoisting the fixture call into a helper -- a refactor with no
        semantic content -- took a module out of the census and made the
        allow-list guard demand its row be deleted.

        The last six are the other direction.  Dropping route B's depth or
        emptiness conditions turns the two blanking tests, the two
        compensations and five refusal-reaching deletions into "offenders",
        and sends a reader to move a builder that does not exist.
        """
        path = tmp_path / "test_probe.py"
        path.write_text(source)
        assert _rolls_its_own(path) == expected

    def test_no_module_beyond_the_recorded_four_rolls_its_own(self):
        """The contract, as the only thing it can usefully be: a ceiling.

        DISCOVERED, not listed: a test that named the modules it checks would
        need editing whenever a module was added, and that edit is the same
        one that silently drops a new offender.  What is listed is the four
        that already offend, and the assertion is about everything that is
        NOT on that list -- plus, first, that the list itself has not grown,
        because a fifth module and a fifth row is otherwise a green edit.

        A module here that is neither an allow-listed offender nor sanctioned
        means a test document was built somewhere it may not be, and every
        test in it is running on a document where ``built.twin`` and
        ``built.inference.fit_twin`` may be the same object.  Move the builder
        into a helper module rather than adding a row.
        """
        assert len(_KNOWN_OFFENDERS) <= _OFFENDER_CEILING, (
            f"_KNOWN_OFFENDERS has {len(_KNOWN_OFFENDERS)} rows against a "
            f"ceiling of {_OFFENDER_CEILING}: {sorted(_KNOWN_OFFENDERS)}. "
            "The list records what Plan 2C left behind and may only shrink. "
            "A new module builds its documents in a helper module; it does "
            "not buy an exemption by adding a row here."
        )
        census = _census()
        sanctioned = _sanctioned()
        unexpected = sorted(set(census) - set(_KNOWN_OFFENDERS) - sanctioned)
        assert not unexpected, (
            f"{unexpected} build their own test document. "
            f"{sorted(sanctioned)} are the only ones that may -- see "
            "exit_helpers._repaired for what a rolled-own builder costs every "
            "test in the module. Add the document to a helper module instead."
        )

    def test_the_allow_list_shrinks_and_never_goes_stale(self):
        """The other direction: a retired offender must leave the list.

        An allow-list that only ever grows is a list nobody prunes, and a
        stale row silently re-permits a module that had been fixed.  So each
        row is asserted to still be true.

        **What this test cannot tell you, and must therefore not claim.**  A
        module leaves the census either because it was migrated -- the row
        should go -- or because it drifted out of the matcher's reach while
        still building its own document, in which case deleting the row is
        the defect and the matcher is what needs the edit.  The first draft
        of this message asserted the former and was wrong: measured, a pure
        refactor of ``test_config_exits_predict.py`` produced exactly this
        failure while all 27 of its blind tests stayed blind.  The message
        now names both readings and the check that separates them.
        """
        census = _census()
        gone = sorted(name for name in _KNOWN_OFFENDERS if name not in census)
        assert not gone, (
            f"{gone} no longer match as building their own document. TWO "
            "readings, and they need opposite fixes. If the module was "
            "migrated to a helper module, delete its row -- a stale row "
            "re-permits a module that had been fixed. If it still builds a "
            "document and the matcher merely stopped seeing it, the matcher "
            "is what to fix: check with `grep -n synthetic_document` and by "
            "loading one of its documents and comparing built.twin with "
            "built.inference.fit_twin. Deleting the row on that reading "
            "retires a module that still loses the discrimination."
        )


class TestTheSharedDocumentsCanTellTheTwoTwinsApart:
    def test_every_helper_module_that_builds_documents_declares_a_floor(self):
        """The floor table cannot be pruned to shrink the walk.

        The per-module floors below are what make the property test's
        coverage measurable, so deleting a row is a way of covering less
        while staying green.  Every discovered helper module that defines a
        builder must therefore appear in the table, and every row must name a
        module that exists.  ``inference_helpers`` defines no document at all
        and correctly needs no row.
        """
        by_module = _builders()
        missing = sorted(stem for stem, found in by_module.items()
                         if found and stem not in _BUILDER_FLOOR)
        assert not missing, (
            f"{missing} define document builders but have no row in "
            "_BUILDER_FLOOR, so nothing measures whether the property test "
            "still walks them. Add the row with the count you measured."
        )
        stale = sorted(set(_BUILDER_FLOOR) - set(by_module))
        assert not stale, (
            f"{stale} have rows in _BUILDER_FLOOR but are not "
            "tests/config/*_helpers.py modules any more."
        )

    def test_every_builder_in_every_helper_module_keeps_the_repair(self):
        """The PROPERTY, not the syntactic proxy above.

        :func:`_rolls_its_own` reads syntax; this reads the built object.  A
        future edit that kept the shape and dropped the repair -- deleting
        ``MODEL_NOISE`` from ``CONJUGATE_MODEL``, or making ``_repaired``
        ``setdefault`` something else -- passes every assertion in the class
        above and disarms the whole exit suite, which is the failure
        ``_repaired``'s own docstring measured at 74 tests.

        Discovered by walking every helper module's namespace for its own
        ``*_document`` functions, so a builder added by a later plan is
        covered on the day it lands rather than on the day someone remembers.
        The ``__module__`` test is what makes it "their own" -- and the walk
        is over ``tests/config/*_helpers.py`` rather than a written tuple
        because ``posterior_helpers``' builders carry ITS ``__module__``, so
        a walk that lost that one name would drop every ``kind: nuts`` and
        ``kind: npe`` document and stay green.

        **A builder that DELEGATES is held to the same property, not to a
        weaker one.**  All four of ``posterior_helpers``' builders reach their
        document through ``exit_helpers.conjugate_document`` rather than
        through ``synthetic_document``, so none of them appears in the census
        above and no syntactic rule says anything about them.  What is
        asserted of them is what is asserted of a builder that rolls its own
        from scratch: load the document and the two twins are different
        objects.  Delegation is thereby a way of keeping the repair, not an
        exemption from it -- and the day a delegating builder overrides
        ``inference:`` with a block of its own, this is what says so.

        The floors are per module and are this tree's measured counts (8 and
        4, twelve in all).  A single global floor is the assertion this task
        was written to avoid: at ``>= 5`` it stays green with a whole module
        missing from the walk.
        """
        by_module = _builders()
        for module, floor in _BUILDER_FLOOR.items():
            found = by_module.get(module, {})
            assert len(found) >= floor, (
                f"{module} defines {len(found)} document builders, below the "
                f"{floor} measured when this guard was written: "
                f"{sorted(found)}. A builder that left this module is a "
                "builder this property no longer covers -- if it moved, move "
                "the floor with it; if it went, say so here."
            )
        same = []
        for found in by_module.values():
            for name, builder in found.items():
                built = _build(builder)
                if built.twin is built.inference.fit_twin:
                    same.append(name)
        assert not same, (
            f"{sorted(same)} build a document whose built.twin IS its "
            "inference.fit_twin, so no test on it can tell an executor that "
            "reaches for one from an executor that reaches for the other. "
            "See exit_helpers._repaired."
        )
