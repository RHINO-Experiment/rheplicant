"""Section 5 requirement 4 and validation item 12.10.

An invisible direction must appear as a collapsed eigenvalue WITH A NAME
ATTACHED, and compression must refuse rather than proceed. The distinguishing
property of this failure is that everything else looks right: the basis
reproduces every draw in the bank, the residual chi-square is fine, the term is
finite and well conditioned, and the marginal on the one parameter the campaign
exists to measure has reverted to its prior.

The basis under test here is a plain SVD at n_S = 3, which is where this
fixture's deletion is real -- measured r_t21 = 0.565 on this seed, and gone by
n_S = 4. `test_reduced_basis_build.py` carries the whole table and the reason.
A refusal test built at n_S = 8 would pass by construction and would go on
passing after the refusal was deleted.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.reduced_basis import (
    ReducedBasis,
    basis_fidelity,
    build_reduced_basis,
    orthonormalise,
    score_directions,
    select_svd,
)
from tests.evidence import rhino_bank

#: The whitened bank's numerical rank -- see `test_reduced_basis_build.py`.
#: `build_reduced_basis` refuses above it, so this is the largest honest basis.
BANK_RANK = 6


@pytest.fixture(scope="module")
def bank():
    return rhino_bank.bank(jax.random.key(12), n_draws=400)


@pytest.fixture(scope="module")
def scores():
    return score_directions(
        rhino_bank.space(), rhino_bank.pipeline(), rhino_bank.state()
    )


def _svd_only(bank, n_basis=3):
    """`rows` are raw; dividing the whitened rows back out is safe here only
    because RadiometerNoise on a positive prediction leaves no zero weight."""
    weight = rhino_bank.weight()
    predict, values = rhino_bank.forward()
    return ReducedBasis(
        rows=orthonormalise(select_svd(bank * weight, n_basis)) / weight,
        weight=weight,
        predict=predict,
        reference=predict(values),
    )


def _seeded(bank, n_basis=BANK_RANK):
    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=bank,
        n_basis=n_basis,
    )


def test_a_seeded_direction_has_zero_residual(bank, scores):
    report = basis_fidelity(_seeded(bank), scores)
    for name, residual in report.residuals.items():
        assert residual < 1e-10, f"{name} -> {residual}"


def test_the_fisher_rendering_names_its_rows_in_flatten_order(bank, scores):
    """Plan A's BayesMemory.fisher() bug, in the place it would recur.

    The fixture declares ("t21_depth", "index", "amplitude", "running"); jax
    flattens a dict in sorted order. A matrix built in declared order and
    labelled in flatten order is wrong by a permutation that is the identity
    for an alphabetical fixture -- which is why this one is not.
    """
    report = basis_fidelity(_seeded(bank), scores)
    assert report.full.names == ("amplitude", "index", "running", "t21_depth")
    assert report.full.kind == "fisher"
    diagonal = np.diag(np.asarray(report.full.matrix))
    # amplitude carries ~200 K of signal, t21_depth ~0.2 K: if the rows were in
    # declared order these two would be swapped, and both are strictly positive
    # so no sign or finiteness check could notice.
    assert diagonal[0] > diagonal[3]


def test_a_deleted_direction_is_a_collapsed_eigenvalue_with_a_name(bank, scores):
    """12.10, the diagnostic half."""
    report = basis_fidelity(_svd_only(bank, 3), scores)
    assert report.residuals["t21_depth"] > 0.15
    # `.block` is a SUB-MATRIX, (1, 1) for a scalar latent -- `float()` of it
    # raises rather than silently taking an element, which is the right shape of
    # refusal and the reason for the explicit `.item()`.
    full = np.asarray(report.full.block("t21_depth")).item()
    projected = np.asarray(report.projected.block("t21_depth")).item()
    assert projected < 0.98 * full
    assert report.worst()[0] == "t21_depth"


def test_compression_refuses_a_basis_that_omits_the_t21_score(bank, scores):
    """12.10, the refusal half. The tolerance is declared, not discovered."""
    report = basis_fidelity(_svd_only(bank, 3), scores)
    with pytest.raises(StateValidationError, match="t21_depth"):
        report.refuse_above(0.01)


def test_the_refusal_fires_at_the_tolerance_and_not_a_hair_above_it(bank, scores):
    """The refusal has to be a threshold, not a mood.

    Measured either side of the worst residual on this basis: 1 % above it is
    silent, 1 % below it raises. Without this the test above would still pass
    against a `refuse_above` that ignored its argument and always raised, which
    is the over-refusing half of the same bug.
    """
    report = basis_fidelity(_svd_only(bank, 3), scores)
    worst = report.worst()[1]
    report.refuse_above(worst * 1.01)
    with pytest.raises(StateValidationError, match="t21_depth"):
        report.refuse_above(worst * 0.99)


def test_a_tolerance_a_good_basis_meets_does_not_fire(bank, scores):
    """A guard that over-refuses is its own bug."""
    basis_fidelity(_seeded(bank), scores).refuse_above(0.01)


def test_a_latent_the_prediction_does_not_respond_to_is_named_not_divided_by_zero(
    bank, scores
):
    blind = {**scores, "dead": jnp.zeros((1, rhino_bank.N_FREQ))}
    report = basis_fidelity(_seeded(bank), blind)
    assert np.isnan(report.residuals["dead"])
    with pytest.raises(StateValidationError, match="does not respond"):
        report.refuse_above(0.5)


def test_a_dead_direction_is_reported_as_a_model_fault_not_a_truncation(bank, scores):
    """The two refusals are separate messages on purpose.

    Reporting a derivative that is identically zero as a truncation would send
    the caller to raise n_S against a direction that does not exist. `worst()`
    sorts nan first for the same reason: it is the finding, not a missing value.
    """
    blind = {**scores, "dead": jnp.zeros((1, rhino_bank.N_FREQ))}
    report = basis_fidelity(_seeded(bank), blind)
    assert report.worst()[0] == "dead"
    with pytest.raises(StateValidationError, match="identifiability"):
        report.refuse_above(0.5)


def test_a_non_finite_score_is_refused_rather_than_scored(bank, scores):
    """NaN defeats every comparison-based guard, so it is caught before one runs.

    `norm(nan) / norm(nan)` is nan, `nan > tolerance` is False, and the refusal
    would pass a broken forward model straight through while reporting a
    perfectly conditioned Fisher.
    """
    broken = {**scores, "index": scores["index"].at[0, 3].set(jnp.nan)}
    with pytest.raises(StateValidationError, match="broken forward model"):
        basis_fidelity(_seeded(bank), broken)


def test_the_projected_fisher_never_exceeds_the_full_one(bank, scores):
    """A projection removes information; it cannot add any."""
    report = basis_fidelity(_svd_only(bank, 5), scores)
    difference = np.asarray(report.full.matrix) - np.asarray(report.projected.matrix)
    assert np.linalg.eigvalsh(difference).min() > -1e-8
