"""``held_out_z`` against a leave-one-out posterior assembled from scratch.

The existing checks in ``test_held_out.py`` are good at what they do and two of
them are **not oracles**, which matters now that this function is a delegation
candidate:

* ``test_the_score_is_the_chi_square_of_the_held_out_residual`` reads ``z``,
  ``chi2`` and ``dof`` off the **same returned object** and checks they are
  consistent with each other. Any implementation that computed all three the
  same wrong way satisfies it.
* ``test_a_repeated_design_makes_the_held_out_score_blind`` compares
  ``held_out_z(clean)`` against ``held_out_z(biased)``. Both operands come from
  the implementation under test, so the claim survives any implementation that
  is merely *self-consistent* -- including one that returns a constant. Its
  sibling ``test_the_blindness_tightens_as_the_priors_share_falls`` is the half
  with teeth, because a constant would not scale as ``1/N``.

Neither is a defect while the arithmetic is this package's own. Both become one
the moment it is bayesmith's, because then the two sides of the comparison are
the same code (the migration ledger's **D61**: a test comparing far against far
passes forever).

**The oracle here is a different derivation, not a second spelling.**
``diagnostics.held_out_z`` forms the campaign total once in ``(F, b)`` form and
subtracts one epoch's contribution per row -- ``O(N)``, and its docstring
explains why it does not downdate the QR. This file assembles
``F_{-e}`` and ``b_{-e}`` from the ``N - 1`` remaining terms **directly**, in
dense NumPy, for every ``e``. Same quantity, ``O(N^2)`` arithmetic, no
subtraction anywhere -- so a sign error or a stale-total bug in the subtraction
path has nowhere to hide.
"""

import numpy as np
import pytest

from rheplicant.inference.diagnostics import held_out_z
from tests.evidence import campaign_bank as camp


def _oracle(terms, prior_fisher, prior_mean=None):
    """``(chi2, dof, z)`` per epoch, assembled from the others without subtracting.

    ``m = R_e mu_{-e} - z_e`` is ``N(0, I + R_e Sigma_{-e} R_e^T)``, so
    ``m^T V^-1 m`` is chi-square on ``rank(R_e)`` degrees of freedom.
    """
    factors = [np.asarray(t.info.factor, dtype=float) for t in terms]
    targets = [np.asarray(t.info.target, dtype=float) for t in terms]
    prior = np.asarray(prior_fisher, dtype=float)
    width = prior.shape[0]
    mean = np.zeros(width) if prior_mean is None else np.asarray(prior_mean, float)

    out = []
    for held in range(len(terms)):
        fisher = prior.copy()
        rhs = prior @ mean
        for other, (r, z) in enumerate(zip(factors, targets, strict=True)):
            if other == held:
                continue
            fisher = fisher + r.T @ r
            rhs = rhs + r.T @ z
        mu = np.linalg.solve(fisher, rhs)
        covariance = np.linalg.inv(fisher)
        r, z = factors[held], targets[held]
        residual = r @ mu - z
        spread = np.eye(r.shape[0]) + r @ covariance @ r.T
        chi2 = float(residual @ np.linalg.solve(spread, residual))
        dof = int(r.shape[0])
        out.append((chi2, dof, (chi2 - dof) / float(np.sqrt(2.0 * dof))))
    return out


@pytest.mark.parametrize("n_epochs", (2, 5, 12))
@pytest.mark.parametrize("biased", (False, True))
def test_the_scores_are_the_leave_one_out_chi_squares(n_epochs, biased):
    """Every epoch, against a posterior built from the others and nothing else."""
    terms = camp.terms(n_epochs, biased=biased)
    prior = camp.prior_fisher()
    rows = held_out_z(terms, prior)
    expected = _oracle(terms, prior)
    assert len(rows) == len(expected)
    for row, (chi2, dof, z) in zip(rows, expected, strict=True):
        assert row.dof == dof
        assert row.chi2 == pytest.approx(chi2, rel=1e-9), (
            f"epoch {row.epoch_id!r}: chi2 {row.chi2!r} against a directly "
            f"assembled leave-one-out posterior's {chi2!r}"
        )
        assert row.z == pytest.approx(z, rel=1e-9)


def test_the_oracle_disagrees_with_a_plausible_wrong_answer():
    """Anti-vacuity: the oracle is sharp enough to reject a near miss.

    Leaving the held-out epoch IN the posterior is the obvious way to get this
    wrong, and it is a change the rest of the suite barely notices. If the
    oracle could not tell that apart, the tests above would be decoration.
    """
    terms = camp.terms(6, biased=False)
    prior = camp.prior_fisher()
    honest = _oracle(terms, prior)

    factors = [np.asarray(t.info.factor, float) for t in terms]
    targets = [np.asarray(t.info.target, float) for t in terms]
    fisher = prior + sum(r.T @ r for r in factors)
    rhs = sum(r.T @ z for r, z in zip(factors, targets, strict=True))
    mu = np.linalg.solve(fisher, rhs)
    covariance = np.linalg.inv(fisher)
    for index, (r, z) in enumerate(zip(factors, targets, strict=True)):
        residual = r @ mu - z
        spread = np.eye(r.shape[0]) + r @ covariance @ r.T
        leaky = float(residual @ np.linalg.solve(spread, residual))
        assert leaky != pytest.approx(honest[index][0], rel=1e-6), (
            f"epoch {index}: leaving the epoch in gives {leaky!r} and leaving "
            f"it out gives {honest[index][0]!r}. The oracle cannot separate "
            f"the two, so it cannot certify the implementation either."
        )


def test_a_non_zero_prior_mean_reaches_the_scores():
    """The prior-mean path, against the same independent assembly.

    ``held_out_z``'s third argument is optional and every other test in this
    directory leaves it out.
    """
    terms = camp.terms(5, biased=False)
    prior = camp.prior_fisher()
    mean = np.array([0.7, -0.3])[: prior.shape[0]]
    rows = held_out_z(terms, prior, mean)
    expected = _oracle(terms, prior, mean)
    for row, (chi2, _dof, z) in zip(rows, expected, strict=True):
        assert row.chi2 == pytest.approx(chi2, rel=1e-9)
        assert row.z == pytest.approx(z, rel=1e-9)

    without = held_out_z(terms, prior)
    assert [r.chi2 for r in rows] != pytest.approx(
        [r.chi2 for r in without], rel=1e-6
    ), "the prior mean moved nothing, so this test cannot see it being dropped"
