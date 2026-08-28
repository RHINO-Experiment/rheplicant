"""The declared ``bayesmith>=0.5`` floor, checked by CAPABILITY not by version.

**Written because the floor cannot be checked the obvious way here, and that
is not a temporary accident.** ``pyproject.toml`` declares ``bayesmith>=0.5``,
and a resolver installing from PyPI would enforce it. This checkout does not
go through a resolver: ``CLAUDE.md`` records that bayesmith is held **editable
from ``../bayesmith`` with ``--no-deps``**, deliberately, because the two
repositories are developed against each other and a released pin would freeze
the seam mid-programme.

The consequence, measured 2026-08-28: ``bayesmith.__version__`` reports
**0.2.0** while ``../bayesmith/pyproject.toml`` says **0.5.0**. The metadata
was written when the editable install was made and no version bump since has
refreshed it. The CODE is 0.5's code -- an editable install is the working
tree -- so nothing is broken, but the label is two releases stale and any
guard reading it would be reading fiction. Nothing in ``src/`` or ``tests/``
does read it, which is how this went unnoticed.

So the floor is asserted the only way that is true in every environment: by
asking whether the surface each level was raised FOR is actually reachable.
``CLAUDE.md`` states what each level buys, and those statements are what this
file turns into assertions -- one per level, so a floor that silently drops
names anywhere below the top still fails here rather than at some call site
three modules away.

This is the second of the two closing questions the migration handover asks --
"what does this green guard depend on that you have never varied?" -- answered
for the dependency itself. What was never varied was the installed bayesmith.
"""

from __future__ import annotations

import inspect

import pytest

bayesmith = pytest.importorskip("bayesmith", reason="bayesmith not installed")


def test_the_0_2_surface_is_reachable():
    """``first_fit`` and ``exact.loglinear`` -- what ``partition.py`` and
    ``loglinear.py`` import."""
    from bayesmith.dispatch.factor import first_fit  # noqa: F401
    from bayesmith.exact import loglinear  # noqa: F401


def test_the_0_3_surface_is_reachable():
    """``AffinityRefused``'s structured PAYLOAD and ``ComplexNormal`` -- what
    ``graph_bridge.py`` needs. A 0.2 install satisfies the import statements of
    this pair and then fails at the CALL, which is the shape a floor exists to
    turn into a resolution error.

    **So the payload is what is asserted, not the name**, and the first version
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
    """``observed_mask`` -- how the adapter presents a ``FlaggedNoise``.

    **The first version of this test passed for the wrong reason**, and the
    correction is worth keeping visible. It searched ``bayesmith.exact``'s
    ``gaussian`` and ``precision`` for the name, in the module or in any
    signature, and went green -- because ``gaussian.Probabilistic`` happens to
    take a PARAMETER called ``observed_mask``. The function the floor is about
    lives in ``bayesmith.marginal``, which the search never looked at. A guard
    that hunts a name across several modules will find a homonym sooner or
    later; this one asks the single question it means.
    """
    from bayesmith.marginal import observed_mask

    assert callable(observed_mask)
    assert "precision" in inspect.signature(observed_mask).parameters


def test_the_0_5_surface_is_reachable():
    """``local_block(..., priors=True)`` -- G15's third block constructor,
    which ``uncertainty.fisher_information(space=...)`` delegates its prior
    curvature to. A 0.4 install imports fine and raises ``TypeError:
    unexpected keyword argument 'priors'`` at the call."""
    from bayesmith.diagnose.local import local_block

    assert "priors" in inspect.signature(local_block).parameters, (
        "local_block has no `priors` parameter, so the installed bayesmith is "
        "below the declared >=0.5 floor however its metadata is labelled"
    )


def test_the_declared_floor_and_this_file_name_the_same_level():
    """A floor raised in ``pyproject.toml`` without a case here would be a
    number nothing checks -- which is the state this file was written to end.
    """
    import pathlib
    import re

    text = (pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    found = re.search(r'"bayesmith>=([0-9.]+)"', text)
    assert found, "pyproject.toml no longer declares a bayesmith floor"
    declared = found.group(1)
    assert f"test_the_{declared.replace('.', '_')}_surface_is_reachable" in globals(), (
        f"pyproject declares bayesmith>={declared} and this file has no case "
        f"for that level; add one naming what the level buys"
    )
