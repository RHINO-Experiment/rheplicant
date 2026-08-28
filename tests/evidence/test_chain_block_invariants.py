"""The stored-block guard compared a quantity that ``combine`` does not preserve.

``ChainMemory.__init__`` checks every stored block against the epoch the archive
names, by comparing three coefficients from :func:`_quadratic_form`. Its
docstring says they are "what a stored block and the epoch it came from have in
common, and all that they have in common". Two of the three are. The third,
``target @ target + offset``, is not.

``_square_block`` stores an epoch as ``combine(null, info)``, and ``combine``
re-triangularises: the part of the residual that no quadratic form in ``x`` can
express becomes the corner ``rho``, which leaves ``target`` and is paid into
``offset`` as ``-rho^2/2``. So ``z.z`` falls by ``rho^2`` while ``offset`` falls
by ``rho^2/2``, and their SUM falls by ``1.5 * rho^2`` -- a preserved quantity
only when ``rho`` is zero.

**How it surfaced.** CI (linux/x86-64) went red on
``test_a_rank_deficient_night_is_not_an_error_and_the_prior_is_the_floor`` with
"quadratic forms differ by 2.081e+01, against a band of 1.849e-06", while the
same commit was green here (darwin/arm64). The fixture is an exactly collinear
design, so its QR has a zero pivot, and *where LAPACK puts it* is the platform
difference: here column 1 reduces to 1.36e-16 and the null direction stays
inside the kept block, so ``rho`` is empty and the wrong formula agrees with
the right one. There, it reduces to exactly 0, the mass moves to the corner,
and ``rho^2 = 14.62``. 1.5 times that is 21.93 -- the reported discrepancy.

The block was correct on both machines. The guard was wrong on one of them, and
it is the kind of wrong that reads as a data-integrity failure: a
``StateValidationError`` saying the campaign has been shuffled.

Nothing here depends on the platform. A ``rho`` the corner has to carry is a
property of the arithmetic, and the fixtures below build one directly.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.chain import (
    ChainMemory,
    _quadratic_form,
    _square_block,
    ornstein_uhlenbeck,
)
from rheplicant.inference.compressed import QuadraticLikelihood
from rheplicant.inference.sqrtinfo import SqrtInfo
from tests.evidence import chain_bank as bank

ORDER = (*bank.THETA_NAMES, bank.ZETA_NAME)
SHAPES = tuple(() for _ in ORDER)
WIDTH = len(ORDER)


def info_with(factor, target, offset=0.0):
    return SqrtInfo(
        factor=jnp.asarray(np.asarray(factor, dtype=float)),
        target=jnp.asarray(np.asarray(target, dtype=float)),
        offset=jnp.asarray(float(offset)),
        names=ORDER,
        shapes=SHAPES,
    )


#: A factor with an exactly zero ROW and target mass sitting on it. That mass is
#: not expressible as a quadratic form in x, so `combine` must move it to the
#: corner -- which is the state CI reached by accident and this reaches on
#: purpose. Deliberately round numbers: rho^2 comes out exactly 25.
ZERO_ROW = info_with(
    np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 3.0]])[:WIDTH, :WIDTH],
    np.array([1.0, 5.0, 2.0])[:WIDTH],
)

#: Exactly collinear columns -- the CI fixture's shape, built rather than
#: sampled. rho^2 comes out exactly 16.
COLLINEAR = info_with(
    np.array([[1.0, 2.0, 0.5], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])[:WIDTH, :WIDTH],
    np.array([3.0, 1.0, 4.0])[:WIDTH],
)

#: The control. A full-rank factor leaves the corner empty, which is why the
#: defect went unseen: with rho = 0 the wrong formula and the right one agree.
FULL_RANK = info_with(
    np.triu(np.arange(1, WIDTH * WIDTH + 1, dtype=float).reshape(WIDTH, WIDTH)),
    np.arange(1.0, WIDTH + 1.0),
)


def corner_squared(info):
    """``rho^2`` that ``_square_block`` paid into the offset for this epoch."""
    return 2.0 * float(info.offset - _square_block(info, ORDER).offset)


class TestTheFixturesReachTheStateTheyClaimTo:
    """Asked first, because every assertion below is vacuous without it.

    A guard that fires only when the corner is non-empty cannot be exercised by
    a fixture whose corner is empty -- and that is exactly how this defect
    survived: every fixture in the evidence suite that reached this code path
    had rho = 0 on the machine it was written on.
    """

    @pytest.mark.parametrize(
        ("info", "expected"), [(ZERO_ROW, 25.0), (COLLINEAR, 16.0)]
    )
    def test_the_rank_deficient_fixtures_leave_a_corner(self, info, expected):
        assert corner_squared(info) == pytest.approx(expected, rel=1e-9)

    def test_the_full_rank_control_leaves_none(self):
        assert corner_squared(FULL_RANK) == pytest.approx(0.0, abs=1e-12)


class TestTheComparedCoefficientsAreInvariant:
    """All three, under the ``combine`` that ``_square_block`` stores through."""

    @pytest.mark.parametrize("info", [ZERO_ROW, COLLINEAR, FULL_RANK])
    def test_every_coefficient_survives_being_stored(self, info):
        square = _square_block(info, ORDER)
        found = _quadratic_form(square.factor, square.target, square.offset)
        want = _quadratic_form(info.factor, info.target, info.offset)
        for index, label in enumerate(("gram", "cross term", "constant")):
            assert np.allclose(found[index], want[index], rtol=1e-9, atol=1e-9), (
                f"the {label} moved when the epoch was stored: "
                f"{found[index]} vs {want[index]}"
            )

    @pytest.mark.parametrize("info", [ZERO_ROW, COLLINEAR])
    def test_the_constant_is_the_log_densitys_own_and_not_z_dot_z_plus_offset(
        self, info
    ):
        """The two candidates, separated by a fixture that can tell them apart.

        ``offset - z.z/2`` is the constant term of ``offset - ||Rx - z||^2/2``.
        ``z.z + offset`` is not a term of anything; it agreed with the truth on
        every fixture that had rho = 0, which was all of them.
        """
        square = _square_block(info, ORDER)

        def wrong(term):
            return float(term.target @ term.target) + float(term.offset)

        def right(term):
            return float(term.offset) - 0.5 * float(term.target @ term.target)

        assert right(square) == pytest.approx(right(info), rel=1e-9)
        assert wrong(square) != pytest.approx(wrong(info), rel=1e-3)
        assert wrong(square) - wrong(info) == pytest.approx(
            -1.5 * corner_squared(info), rel=1e-9
        )


class TestTheGuardItself:
    def _memory(self):
        return ChainMemory(bank.factorization(ornstein_uhlenbeck(tau=5.0, sigma=1.0)))

    def _term(self, info, epoch_id="corner"):
        return QuadraticLikelihood(
            info=info,
            epoch_id=epoch_id,
            n_observed=bank.N_SAMPLES,
            residual_chi2=jnp.asarray(0.0),
            residual_dof=0,
        )

    @pytest.mark.parametrize("info", [ZERO_ROW, COLLINEAR, FULL_RANK])
    def test_an_epoch_whose_corner_is_not_empty_is_still_its_own_block(self, info):
        """The regression. This raised `StateValidationError` on CI, naming a
        shuffled campaign, for a block that was correct."""
        memory = self._memory().remember(self._term(info))
        assert tuple(memory.epoch_ids) == ("corner",)

    def test_a_genuinely_foreign_block_is_still_refused(self):
        """The sibling that keeps the fix from being a way to switch the guard
        off. A block whose Gram is another epoch's must still be caught."""
        memory = self._memory().remember(self._term(COLLINEAR))
        factors, targets, offsets = memory.stacked
        other = _square_block(FULL_RANK, ORDER)
        poisoned = (
            factors.at[0].set(other.factor),
            targets.at[0].set(other.target),
            offsets.at[0].set(other.offset),
        )
        with pytest.raises(StateValidationError, match="quadratic forms differ"):
            ChainMemory(memory.factorization, poisoned, memory._epochs)
