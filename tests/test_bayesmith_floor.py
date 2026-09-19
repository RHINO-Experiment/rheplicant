"""The declared bayesmith range, checked by capability and by policy, not by version.

``pyproject.toml`` declares ``bayesmith>=0.9,<0.10``, and that range holds two
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
  differs from 0.6 in behaviour only.
* The **declared range** starts at 0.9 although the code needs nothing newer
  than 0.6, because the stable baseline relies on bayesmith 0.9's stability
  contract and is tested only against 0.9. It is closed at the next minor,
  because a pre-1.0 minor may move the deep module paths this package imports.

**No case reads the installed version.** For most of this file's history
bayesmith was installed editable from ``../bayesmith``, and an editable install
reports the version its metadata was written with: 0.2.0 against 0.5.0 source
on 2026-08-28, and still 0.2.0 against 0.9.0 source on 2026-09-19. The
checkout now installs bayesmith from its local 0.9.0 wheel, where the metadata
is right, but a guard that holds in one kind of environment only is what this
file replaced. Whether a capability is reachable holds in every environment.
"""

from __future__ import annotations

import inspect
import pathlib
import re

import pytest

#: The highest bayesmith release whose surface this package uses. Raise it, and
#: add the matching ``test_the_<level>_surface_is_reachable`` case, in the same
#: commit that starts relying on something a later release added.
CAPABILITY_FLOOR = "0.6"

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
    payload = {"names", "at", "errors", "weighted", "rtol", "weighted_rtol",
               "failed"}
    missing = payload - set(inspect.signature(AffinityRefused).parameters)
    assert not missing, (
        f"AffinityRefused is missing {sorted(missing)}, so the installed "
        "bayesmith carries the name without the structured payload -- which "
        "is exactly the 0.2-install failure the >=0.3 half of the floor exists "
        "to turn into a resolution error rather than a TypeError at the call"
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
