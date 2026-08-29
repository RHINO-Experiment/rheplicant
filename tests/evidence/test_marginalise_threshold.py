"""The divergence guard reads a scale that is not the block's, and here is the proof.

:func:`~rheplicant.inference.sqrtinfo.marginalise` refuses to integrate out a
block that does not constrain one of its own directions -- correctly, because
the true answer is ``+inf`` and finite arithmetic returns a large plausible
number instead. The refusal is right. **The threshold it fires at is not.**

``sqrtinfo.py`` compares the block's own pivots against
``sqrt(eps) * max(pivots)``, where the ``max`` runs over **every column of the
joint term**, including columns that are not being integrated out. Its comment
gives the reason:

    the rows are whitened data, so their scale is the epoch's ``1/sigma``

That holds when one ``sigma`` describes the whole epoch. It fails for a
heteroscedastic epoch, and it fails for any term whose global columns are
scaled differently from the block's -- and then the verdict on a block depends
on numbers that have nothing to do with it.

**What this file pins is the defect, not the fix.** The refusal is shipped
behaviour: moving the threshold would admit inputs that are refused today, so
it is a decision with a blast radius rather than a repair (registered as
**D67**). These are ``xfail(strict=True)``, so they pass while the defect
stands and **go red the moment the threshold is corrected** -- at which point
the markers come off and they become ordinary guards.

**Neither this package nor bayesmith could see it.** The only tests of this
refusal zero a column outright, so the pivot is exactly ``0.0`` and any
threshold whatsoever catches it. Nothing exercised a pivot that is small but
legitimate. The evidence fixtures are homoscedastic at order 0.2-0.5, several
decades from the boundary.

**bayesmith ships the same rule**, byte for byte, and refuses at the same
boundary. This package's ``compress_linear`` avoids it only by routing around
it -- it does its own inline QR and never calls ``marginalise`` -- which is why
delegating that function would newly subject the path to a guard this package
already ships and already misplaces.

**In float32 the false-refusal zone is about four and a half decades wider**,
because the floor carries ``sqrt(eps)``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.sqrtinfo import SqrtInfo, marginalise

#: ``0.5 * log(2 * pi)``: the exact marginal of a unit-precision direction with
#: a zero target, which is what ``p`` is in every term this file builds.
_UNIT_MARGINAL = 0.5 * float(np.log(2.0 * np.pi))


def _term(unrelated_scale):
    """Two names, and ``p`` is byte-identical whatever ``unrelated_scale`` is.

    ``p`` is a unit-precision, perfectly well-constrained direction -- pivot
    exactly 1.0. ``g`` is a global column that is never integrated out and that
    the answer for ``p`` does not depend on.
    """
    return SqrtInfo(
        factor=jnp.array([[unrelated_scale, 0.0], [0.0, 1.0]]),
        target=jnp.array([0.0, 1.0]),
        offset=jnp.array(0.0),
        names=("g", "p"),
        shapes=((), ()),
    )


def test_the_blocks_own_pivot_is_the_same_number_in_every_case():
    """The premise. Without this the tests below could be reading a real change."""
    for scale in (1e0, 1e6, 1e8, 1e10):
        pivots = np.abs(np.diag(np.asarray(_term(scale).factor)))
        assert pivots[1] == 1.0, (
            f"the fixture is wrong: at unrelated_scale={scale:g} the block's own "
            f"pivot is {pivots[1]!r} rather than 1.0, so a refusal below would "
            f"not prove what this file says it proves"
        )


@pytest.mark.parametrize("unrelated_scale", (1e0, 1e3, 1e6, 1e7))
def test_a_well_constrained_block_is_admitted_below_the_boundary(unrelated_scale):
    """The unmarked half, and it is what keeps the marked half honest.

    A change that made the marked cases pass by refusing *less* carefully would
    have to keep these exact too.
    """
    out = marginalise(_term(unrelated_scale), ["p"])
    assert float(out.offset) == pytest.approx(_UNIT_MARGINAL, abs=1e-12)


@pytest.mark.xfail(
    strict=True,
    reason="D67: the floor is sqrt(eps) * max(pivots) over ALL columns, tested "
    "against the block's own pivot, so an unrelated column's scale decides "
    "whether a well-constrained block is admitted. Fix by scaling the floor to "
    "the block's own columns; then remove this marker.",
)
@pytest.mark.parametrize("unrelated_scale", (1e8, 1e9, 1e10))
def test_an_unrelated_column_must_not_decide_the_blocks_fate(unrelated_scale):
    """``p`` is unchanged and its integral converges. It is refused anyway."""
    out = marginalise(_term(unrelated_scale), ["p"])
    assert float(out.offset) == pytest.approx(_UNIT_MARGINAL, abs=1e-12)


def test_the_refusal_itself_is_right_and_this_file_does_not_ask_for_it_back():
    """A genuinely unconstrained block must still be refused.

    Said out loud because everything above argues the guard fires too eagerly,
    and the obvious wrong repair is to loosen it until it stops firing.
    """
    unconstrained = SqrtInfo(
        factor=jnp.array([[1.0, 0.0], [0.0, 0.0]]),
        target=jnp.array([1.0, 0.0]),
        offset=jnp.array(0.0),
        names=("g", "p"),
        shapes=((), ()),
    )
    with pytest.raises(StateValidationError, match="does not constrain"):
        marginalise(unconstrained, ["p"])
