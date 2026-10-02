"""The declared bayesmith range, checked by capability and by policy, not by version.

``pyproject.toml`` declares ``bayesmith>=0.10,<0.11``, and that range holds two
numbers which this file keeps apart.

* The **capability floor**, :data:`CAPABILITY_FLOOR`, is the highest release
  whose surface this package uses. Each level has one case below asking
  whether what that level bought is reachable: a name, a signature, or for 0.6
  a behaviour. ``CLAUDE.md`` states what each level buys; the cases turn those
  statements into assertions, so a floor that silently drops a name anywhere
  below the top fails here rather than at a call site three modules away.

  How an install below a level fails differs by level, measured against
  source exports of the tags on 2026-09-19. Below 0.5, ``import
  rheplicant.inference`` fails outright, because ``bayesmith.marginal``, which
  it imports, first ships in 0.5. The 0.4 and 0.5 capabilities are keyword
  arguments on names that already existed, so taken alone each is a
  ``TypeError`` at the call on the release below. A 0.5 install imports, and
  differs from 0.6 in behaviour only. On 0.10 the shape returns to the 0.5
  one: ``bayesmith.optimize.certify`` does not exist below it, so ``import
  rheplicant.inference.plan`` fails outright there.
* The **declared range** starts at 0.10 and the capability floor now sits at
  the same number, which it did not before: the convergence certificate
  ``plan.py`` stops on lives in ``bayesmith.optimize.certify``, which 0.10
  added. The range is closed at the next minor because a pre-1.0 minor may
  move the deep module paths this package imports. 0.10 moved one:
  ``bayesmith.optimize`` became a package. The older imports survived it,
  which is the point of closing the range rather than evidence that closing it
  was unnecessary.

**No case reads the installed version.** For most of this file's history
bayesmith was installed editable from ``../bayesmith``, and an editable install
reports the version its metadata was written with: 0.2.0 against 0.5.0 source
on 2026-08-28, and still 0.2.0 against 0.9.0 source on 2026-09-19. The
checkout now installs bayesmith from its local 0.10.0 wheel, where the metadata
is right, but a guard that holds in one kind of environment only is what this
file replaced. Whether a capability is reachable holds in every environment.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pytest

#: The highest bayesmith release whose surface this package uses. Raise it, and
#: add the matching ``test_the_<level>_surface_is_reachable`` case, in the same
#: commit that starts relying on something a later release added.
CAPABILITY_FLOOR = "0.10"

bayesmith = pytest.importorskip("bayesmith", reason="bayesmith not installed")


def test_the_0_2_surface_is_reachable():
    """``first_fit`` and ``exact.loglinear`` -- what ``partition.py`` and
    ``loglinear.py`` import."""
    from bayesmith.dispatch.factor import first_fit  # noqa: F401
    from bayesmith.exact import loglinear  # noqa: F401


def test_the_0_3_surface_is_reachable():
    """``AffinityRefused``'s structured PAYLOAD and ``ComplexNormal`` -- what
    ``graph_bridge.py`` needs. Both are new in 0.3: a 0.2 install has neither
    ``bayesmith.distributions`` nor ``AffinityRefused``, so this case fails at
    its import statement on 0.2. The payload assertion below therefore cannot
    fail on its own against any tagged release; it is kept as a statement of
    what 0.3 bought.

    **The payload is what is asserted, not the name**, and the first version
    of this case got that wrong: it asserted ``hasattr(AffinityRefused,
    "__mro__")``, which is true of every class in Python and could not have
    failed for any reason. Asking whether it could is what found it -- and the
    answer arrived sideways, because removing the name cannot demonstrate the
    case either: ``AffinityRefused`` is re-exported from ``bayesmith``'s top
    level, so deleting it makes the whole package unimportable and the module
    skips instead. The name's existence was never the capability; the fields
    are.
    """
    from bayesmith.distributions import ComplexNormal  # noqa: F401
    from bayesmith.errors import AffinityRefused, StructureError

    assert issubclass(AffinityRefused, StructureError), (
        "the narrow catch is half of what 0.3 buys -- an `except "
        "StructureError` must keep catching it"
    )
    payload = {"names", "at", "errors", "weighted", "rtol", "weighted_rtol", "failed"}
    missing = payload - set(inspect.signature(AffinityRefused).parameters)
    assert not missing, (
        f"AffinityRefused is missing {sorted(missing)}: the installed bayesmith "
        "defines the class without the structured payload graph_bridge.py "
        "passes. No tagged release does that: 0.2 has no AffinityRefused at "
        "all, and every release from 0.3.0 to 0.9.0 has the full payload"
    )


def test_the_0_4_surface_is_reachable():
    """``observe(..., mask=)`` and the node field ``Probabilistic.observed_mask``
    -- how ``graph_bridge.py`` presents a ``FlaggedNoise``: it passes
    ``mask=`` to ``bayesmith.observe``, and the graph carries it on the node.

    Both arrived in 0.4. On a 0.3 install ``observe`` exists without ``mask``,
    so the adapter's call raises ``TypeError``.

    The previous version of this case had the story backwards. The first
    version searched ``bayesmith.exact`` for the name ``observed_mask`` and
    found it on ``gaussian.Probabilistic``, which is the node class imported
    from ``bayesmith.graph.nodes``, so it had found the 0.4 field. It was
    rewritten as a "homonym" into an import of the FUNCTION
    ``bayesmith.marginal.observed_mask``. That function is not what 0.4 added:
    it has existed since 0.2 as ``bayesmith.evidence.compress.observed_mask``,
    and ``bayesmith.marginal`` first ships in 0.5, so the rewrite failed on
    0.4.0, a release that has the capability.
    """
    import dataclasses

    from bayesmith.graph.nodes import Probabilistic

    assert "mask" in inspect.signature(bayesmith.observe).parameters, (
        "bayesmith.observe takes no `mask`, so the installed bayesmith is below "
        "the 0.4 level and the adapter's FlaggedNoise call raises TypeError"
    )
    assert "observed_mask" in {field.name for field in dataclasses.fields(Probabilistic)}, (
        "Probabilistic has no `observed_mask` field, so the graph cannot carry "
        "a flag mask; the installed bayesmith is below the 0.4 level"
    )


def test_the_0_5_surface_is_reachable():
    """``local_block(..., priors=True)`` -- G15's third block constructor,
    which ``uncertainty.fisher_information(space=...)`` delegates its prior
    curvature to. On a 0.4 install ``local_block`` imports and the call raises
    ``TypeError: unexpected keyword argument 'priors'``."""
    from bayesmith.diagnose.local import local_block

    assert "priors" in inspect.signature(local_block).parameters, (
        "local_block has no `priors` parameter, so the installed bayesmith is "
        "below the 0.5 level however its metadata is labelled"
    )


def _frozen_chain_mean(process_std):
    """``marginal.chain.smooth`` on a frozen (``phi = 1``) chain, in float32.

    Synthetic blocks rather than ``tests/evidence/chain_bank``: the bank builds
    its blocks through ``compress_linear``, which refuses float32 by design,
    and this file runs in the default float32 session.
    """
    import jax.numpy as jnp
    import numpy as np
    from bayesmith.marginal.chain import smooth

    from rheplicant.inference.chain import LinearGaussianTransition

    rng = np.random.default_rng(0)
    n_epochs, width = 16, 3  # two theta columns and one chain column
    factor = np.triu(rng.normal(size=(n_epochs, width, width))) + 3.0 * np.eye(width)
    blocks = (
        jnp.asarray(factor, jnp.float32),
        jnp.asarray(rng.normal(size=(n_epochs, width)), jnp.float32),
        jnp.zeros((n_epochs,), jnp.float32),
    )
    transition = LinearGaussianTransition(
        phi=jnp.eye(1, dtype=jnp.float32),
        process_std=jnp.full((1,), process_std, jnp.float32),
        initial_std=jnp.ones(1, jnp.float32),
    )
    theta = {"a": jnp.float32(0.4), "b": jnp.float32(-1.1)}
    mean, variance = smooth(blocks, transition, theta, ("a", "b"), ((), ()))
    return np.asarray(mean), np.asarray(variance)


def test_the_0_6_surface_is_reachable():
    """``marginal.chain.smooth`` assembled as a square root -- what
    ``inference.chain.smooth`` delegates to.

    No name arrived in 0.6, so the case asks for the behaviour. Up to 0.5 the
    far smoother inverted the explicit precision and paid ``kappa(F)`` where a
    square root pays ``sqrt(kappa(F))``. Measured in float32 on these blocks:
    at ``process_std = 1e-5`` the 0.5 source returns ``nan`` while 0.6.0 and
    0.9.0 return the same finite answer. A 0.5 install imports ``smooth`` fine.

    The convergence bound is 0.1 posterior std, not tighter, because in
    float32 most of the shift between 1e-4 and 1e-5 is roundoff: 2e-3 std on
    these blocks, and up to 1.7e-2 std across 200 random block sets (about
    1e-5 in float64). The ``isfinite`` assertion is what separates 0.5.
    """
    import numpy as np

    loose_mean, loose_variance = _frozen_chain_mean(1e-4)
    stiff_mean, stiff_variance = _frozen_chain_mean(1e-5)
    for label, values in (("mean", stiff_mean), ("variance", stiff_variance)):
        assert np.all(np.isfinite(values)), (
            f"smooth returned a non-finite {label} on a frozen chain at "
            "process_std=1e-5, which is the explicit-precision spelling bayesmith "
            "replaced in 0.6; the installed bayesmith is below the 0.6 level"
        )
    assert np.all(stiff_variance > 0.0)
    scale = float(np.sqrt(loose_variance.min()))
    shift = float(np.abs(stiff_mean - loose_mean).max())
    assert shift < 0.1 * scale, (
        f"the smoothed mean moved {shift:.3e} (posterior std {scale:.3e}) between "
        "process_std 1e-4 and 1e-5, where a frozen chain has converged"
    )


def test_the_0_10_surface_is_reachable():
    """``bayesmith.optimize.certify`` -- the convergence certificate
    ``inference.plan`` stops an estimate on, and the ``certify=``/``floor=``/
    ``polish=`` seam on ``minimize``.

    This is the level that turned ``bayesmith.optimize`` from a module into a
    package. On 0.9 it is a module with no ``certify`` submodule, so this case
    fails at its import statement there -- and so does ``import
    rheplicant.inference.plan``, which has no local copy to fall back on since
    the module was lifted upstream in 0.10.

    **What is asserted is the refusal rule, not the name.** The module's one
    load-bearing property is which curvature floors a verdict may rest on:
    ``dense`` and ``supplied`` are proven lower bounds on the smallest
    eigenvalue, and a *probed* floor is a lower bound on nothing. A build that
    admitted ``"probe"`` here would certify points that have not converged,
    and ``plan.py`` would report them as converged. That cannot be seen from a
    signature, so a signature check would pass on exactly the install this
    package must refuse.

    The set is asserted by membership rather than by equality, so a later
    release may add a floor it has proven without failing this case. What may
    not change is that a probe is not one.
    """
    from bayesmith.optimize import Fit, certify, minimize

    assert "dense" in certify.PROVEN_FLOORS, (
        f"PROVEN_FLOORS is {set(certify.PROVEN_FLOORS)!r} and does not admit "
        "'dense': plan.py certifies small models on the assembled Hessian's "
        "own floor, and nothing else it has would certify them"
    )
    assert "probe" not in certify.PROVEN_FLOORS, (
        f"PROVEN_FLOORS is {set(certify.PROVEN_FLOORS)!r} and admits 'probe': "
        "a probed curvature floor is an estimate, not a lower bound, so a "
        "decrement resting on one certifies nothing. This installed bayesmith "
        "would let SamplingPlan.estimate report unconverged points as converged"
    )
    taken = set(inspect.signature(minimize).parameters)
    missing = {"certify", "floor", "polish"} - taken
    assert not missing, (
        f"bayesmith.optimize.minimize takes no {sorted(missing)}, so the "
        "installed bayesmith is below the 0.10 level however it is labelled"
    )
    carried = {"certificate", "limit", "polished"} - set(Fit._fields)
    assert not carried, (
        f"bayesmith.optimize.Fit has no {sorted(carried)}, so a fit cannot "
        "report what certified it; the installed bayesmith is below 0.10"
    )


def _levels_with_a_case():
    pattern = re.compile(r"test_the_(\d+)_(\d+)_surface_is_reachable")
    return sorted(
        (int(match.group(1)), int(match.group(2)))
        for match in map(pattern.fullmatch, globals())
        if match
    )


def _declared_range():
    text = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    found = re.findall(r'"bayesmith>=(\d+)\.(\d+)(?:\.\d+)?,<(\d+)\.(\d+)"', text)
    assert len(found) == 1, (
        "pyproject.toml must declare bayesmith exactly once, as a closed range "
        f"'bayesmith>=X.Y,<X.Z'; found {found!r}"
    )
    low_major, low_minor, high_major, high_minor = map(int, found[0])
    return (low_major, low_minor), (high_major, high_minor)


def test_the_capability_floor_is_the_highest_level_with_a_case():
    """A level with a case above the recorded floor, or a floor with no case,
    is a number nothing checks -- the state this file was written to end."""
    floor = tuple(int(part) for part in CAPABILITY_FLOOR.split("."))
    levels = _levels_with_a_case()
    assert levels, "no capability case found"
    assert levels[-1] == floor, (
        f"CAPABILITY_FLOOR is {CAPABILITY_FLOOR} but the highest level with a case "
        f"is {'.'.join(map(str, levels[-1]))}; raise the floor with its case"
    )


def test_the_declared_range_covers_the_floor_and_closes_at_the_next_minor():
    floor = tuple(int(part) for part in CAPABILITY_FLOOR.split("."))
    low, high = _declared_range()
    assert low >= floor, (
        f"pyproject admits bayesmith {low[0]}.{low[1]}, below the capability "
        f"floor {CAPABILITY_FLOOR} this package's code needs"
    )
    assert high == (low[0], low[1] + 1), (
        f"the range should close at the next minor after {low[0]}.{low[1]}, "
        f"not at {high[0]}.{high[1]}: a pre-1.0 bayesmith minor may move the "
        "deep module paths this package imports"
    )


def test_the_delegation_surface_for_the_gradient_engine_exists():
    """``engines._adam`` is NOT delegated, and this is half of why.

    The reason recorded in that function is a LEVEL, not a capability gap:
    bayesmith 0.10's ``minimize`` grew every keyword the delegation would
    need, and upstream's own stability page calls the descent engine inside it
    Experimental (reference implementation). Both halves of that sentence are
    facts about another package, and a decision resting on facts nobody checks
    is a decision that quietly stops matching its reason.

    This half is checkable from the installed wheel and always runs. The other
    half needs upstream's prose and is below.
    """
    import inspect

    from bayesmith.optimize import minimize

    accepted = set(inspect.signature(minimize).parameters)
    assert {"step_sizes", "certify", "floor", "polish"} <= accepted, (
        "engines._adam's docstring says delegation is technically possible "
        f"because minimize accepts these; it accepts {sorted(accepted)}"
    )


def test_upstream_still_calls_its_descent_engine_experimental():
    """The other half, and the trigger to reopen the question.

    If upstream raises this surface to Maintained, the objection to delegating
    ``engines._adam`` is gone and someone should measure whether delegation
    preserves the A5-2 behaviour. Nothing else in this checkout would notice,
    which is why the watch is here rather than in a comment.

    It needs the sibling checkout, because the wheel ships no documentation --
    checked 2026-09-20, ``importlib.metadata.files("bayesmith")`` lists no
    markdown at all. A skipping guard is not a passing one, so the message
    says what it stood down on rather than disappearing into a dot.
    """
    from tests.config.wheel_support import BAYESMITH_CHECKOUT

    page = BAYESMITH_CHECKOUT / "docs" / "stability.md"
    if not page.is_file():
        pytest.skip(
            f"{page} is absent, so upstream's declared level for the descent "
            "engine cannot be read here. The decision not to delegate "
            "engines._adam rests on it; re-check by hand against the "
            "bayesmith release this venv installs"
        )

    def cells(line):
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    # Matched on the SUBJECT cell, not on the line. Two rows of that page
    # contain the phrase "descent engine inside" -- one is the summary row
    # whose second cell is a LIST OF SURFACES -- and taking the first match
    # read that list as the level. It then failed for the wrong reason, which
    # is the better of the two ways a mis-aimed matcher can go.
    rows = [
        row
        for row in page.read_text(encoding="utf-8").splitlines()
        if len(cells(row)) >= 2 and "descent engine inside" in cells(row)[0]
    ]
    assert len(rows) == 1, (
        f"{page} has {len(rows)} rows whose subject is the descent engine "
        "inside minimize; engines._adam's recorded reason names exactly one"
    )
    # The LEVEL CELL, not the row. Searching the whole line for "Experimental"
    # is what this assertion did first, and the row's own prose says "Treat it
    # as a working reference" and "the ENGINE being a reference" -- so a row
    # whose level had been raised to Maintained still contained the word, and
    # the guard stayed green on exactly the change it exists to catch.
    level = cells(rows[0])[1]
    assert "Experimental" in level, (
        "upstream now calls the descent engine inside minimize "
        f"{level!r} rather than Experimental. engines._adam declined to "
        "delegate because of that level -- reopen the question"
    )


#: What `docs/stability.md` calls a deliberate reference implementation, and
#: the test file in this repository that holds each one in agreement.
#:
#: Written here rather than only on the page because a table of promises with
#: nothing checking it is the shape this repository keeps paying for: the
#: local symbol can be deleted or renamed, and the file that compares it can
#: be retired, and the page would go on saying both are there.
#: Each row is ``local symbol -> (the file, the ENTRY POINT that comparison
#: goes through)``.
#:
#: The first three rows named files in bayesmith's ``tests/crosscheck/`` until
#: bayesmith's ``61d4644`` removed that directory, and this guard skipped for
#: them from that commit on. The two files moved to ``tests/crosscheck/`` here.
#:
#: The entry point is the part that makes this checkable. A cross-check need
#: not name the private symbol it exercises -- ``test_linear.py`` compares
#: through ``check_linearity`` and never writes ``_worse`` -- so asserting
#: that the file names the SYMBOL would be false. Asserting that it names the
#: entry point is true, and is the thing a reader would look for.
REFERENCE_IMPLEMENTATIONS = {
    "rheplicant.inference.sqrtinfo:SqrtInfo": (
        "tests/crosscheck/test_sqrtinfo_agrees.py",
        "SqrtInfo",
    ),
    "rheplicant.inference.sqrtinfo:marginalise": (
        "tests/crosscheck/test_sqrtinfo_agrees.py",
        "marginalise",
    ),
    "rheplicant.inference.linear:_worse": (
        "tests/crosscheck/test_linear.py",
        "check_linearity",
    ),
    # NOT a comparison with bayesmith, and the row said it was until
    # 2026-09-21. `_zeta_joint` exists on both sides, but `_joint_covariance`
    # -- the only thing that uses it here -- has no counterpart upstream, so
    # there is nothing to compare against. It is held by a dense oracle.
    "rheplicant.inference.chain_recursion:_zeta_joint": (
        "tests/evidence/test_chain_smoother.py",
        "_joint_covariance",
    ),
}


@pytest.mark.parametrize("target", sorted(REFERENCE_IMPLEMENTATIONS))
def test_every_labelled_reference_implementation_is_here(target):
    """The near half: the symbol the page names exists and is importable."""
    import importlib

    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(module_name)
    assert hasattr(module, attribute), (
        f"docs/stability.md calls {target} a deliberate reference implementation and it is gone"
    )


@pytest.mark.parametrize("target", sorted(REFERENCE_IMPLEMENTATIONS))
def test_every_labelled_reference_implementation_is_still_held(target):
    """The far half, which is the half that makes the label mean anything.

    A copy kept "because a cross-check holds it" and no cross-check is just a
    copy.

    **Asserting the file EXISTS is not asserting it checks anything**, and
    that was this test until 2026-09-21. `_zeta_joint` was recorded as held by
    `test_provenance.py`; that file exists, so this passed, and it is a
    per-symbol provenance table that never mentions `_zeta_joint` and compares
    no arithmetic at all. Existence is the cheapest possible proxy for the
    claim and it was wrong about a quarter of the table.

    So the entry point is read out of the file too. Every file is in this
    repository, so nothing here can skip.
    """
    filename, entry = REFERENCE_IMPLEMENTATIONS[target]
    path = pathlib.Path(__file__).resolve().parents[1] / filename

    assert path.is_file(), (
        f"docs/stability.md names {filename} as what holds {target} in "
        "agreement, and it is not there"
    )
    assert entry in path.read_text(encoding="utf-8"), (
        f"{filename} is named as what holds {target} in agreement, and it "
        f"does not mention {entry!r} -- the entry point that comparison is "
        "supposed to go through. A file that exists is not a file that checks."
    )


def test_the_page_lists_exactly_the_labelled_reference_implementations():
    """Both directions, so the page and this table cannot drift apart."""
    root = pathlib.Path(__file__).resolve().parents[1]
    page = (root / "docs" / "stability.md").read_text(encoding="utf-8")
    section = page.split("## Deliberate reference implementations", 1)
    assert len(section) == 2, "docs/stability.md no longer has that section"
    body = section[1].split("## ", 1)[0]
    for filename, _ in REFERENCE_IMPLEMENTATIONS.values():
        assert filename in body, f"the page does not name {filename}"
    for target in REFERENCE_IMPLEMENTATIONS:
        module_name, _, attribute = target.partition(":")
        assert attribute in body or module_name in body, target


#: bayesmith modules imported for a type or an error class. They are not
#: delegations and the page's table does not list them.
NOT_A_DELEGATION = frozenset({"bayesmith.errors", "bayesmith.distributions"})


def _bayesmith_imports() -> dict[str, set[str]]:
    """``local module -> the bayesmith modules it imports``, read out of ``src/``."""
    package = pathlib.Path(__file__).resolve().parents[1] / "src" / "rheplicant" / "inference"
    found: dict[str, set[str]] = {}
    for path in sorted(package.glob("*.py")):
        modules: set[str] = set()
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                modules.add(node.module)
            elif isinstance(node, ast.Import):
                modules.update(alias.name for alias in node.names)
        modules = {name for name in modules if name.split(".")[0] == "bayesmith"}
        modules -= NOT_A_DELEGATION
        if modules:
            found[path.stem] = modules
    return found


def _delegation_rows() -> list[tuple[set[str], set[str]]]:
    """``(local modules, bayesmith names)`` for each row of the page's table."""
    root = pathlib.Path(__file__).resolve().parents[1]
    page = (root / "docs" / "bayesmith.md").read_text(encoding="utf-8")
    section = page.split("## What is delegated", 1)
    assert len(section) == 2, "docs/bayesmith.md no longer has that section"
    rows = []
    for line in section[1].split("## ", 1)[0].splitlines():
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != 3:
            continue
        far = set(re.findall(r"`(bayesmith[\w.]*)`", cells[2]))
        if far:
            rows.append((set(re.findall(r"`([a-z_]+)`", cells[1])), far))
    return rows


def _names(listed: str, imported: str) -> bool:
    """A row names an import by the module or by a member of it."""
    return listed == imported or listed.startswith(imported + ".")


def test_the_page_lists_every_bayesmith_module_this_package_imports():
    """The delegation table is the imports, so a new one cannot go unlisted.

    The table had six rows while 21 modules imported bayesmith: it was written
    when the delegation was small and nothing compared it with ``src/`` after.
    """
    imports = _bayesmith_imports()
    assert len(imports) >= 15, sorted(imports)
    rows = _delegation_rows()
    missing = [
        f"{module}: {imported}"
        for module, modules in sorted(imports.items())
        for imported in sorted(modules)
        if not any(
            module in local and any(_names(listed, imported) for listed in far)
            for local, far in rows
        )
    ]
    assert not missing, (
        "docs/bayesmith.md's delegation table does not list these imports:\n  "
        + "\n  ".join(missing)
    )


def test_every_row_of_the_delegation_table_is_a_real_import():
    """The other direction: a delegation that was removed leaves the table."""
    imports = _bayesmith_imports()
    rows = _delegation_rows()
    assert len(rows) >= 12, rows
    stale = []
    for local, far in rows:
        assert local, f"a row names no module of rheplicant.inference: {far}"
        for module in sorted(local):
            if module not in imports:
                stale.append(f"{module} imports nothing from bayesmith")
        for listed in sorted(far):
            if not any(
                _names(listed, imported) for module in local for imported in imports.get(module, ())
            ):
                stale.append(f"{sorted(local)} do not import {listed}")
    assert not stale, "docs/bayesmith.md's delegation table is stale:\n  " + "\n  ".join(stale)
