"""The container, and the one choice the whole of section 5 turns on: the metric.

A fixture whose sigma is constant cannot see any of this -- the N^-1 metric and
unweighted L2 are then the same inner product, every assertion below holds
vacuously, and the tests pass while the code is wrong. So the weight here spans
a factor of 20, and `test_a_constant_sigma_would_make_this_file_vacuous` says so
in executable form.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.reduced_basis import ReducedBasis, orthonormalise

N_DATA = 64


def _weight(spread=20.0):
    """1/sigma over a band whose noise varies by `spread` end to end."""
    sigma = jnp.linspace(1.0, spread, N_DATA)
    return 1.0 / sigma


def _basis(rows, weight=None, orthonormal=True):
    """A basis over `rows`, which are RAW (unwhitened) -- see the class docstring.

    The helper orthonormalises in the whitened metric and divides back out,
    which is legitimate here only because `_weight()` is strictly positive.
    `build_reduced_basis` cannot do that and uses the transform instead.
    """
    weight = _weight() if weight is None else weight
    raw = orthonormalise(rows * weight) / weight if orthonormal else rows
    return ReducedBasis(
        rows=raw,
        weight=weight,
        predict=lambda values: jnp.zeros(N_DATA),
        reference=jnp.zeros(N_DATA),
        orthonormal=orthonormal,
    )


def _snapshots(key, n):
    return jax.random.normal(key, (n, N_DATA))


def test_orthonormalise_gives_rows_orthonormal_in_the_inverse_noise_metric():
    basis = _basis(_snapshots(jax.random.key(0), 5))
    np.testing.assert_allclose(np.asarray(basis.gram()), np.eye(5), atol=1e-12)


def test_a_constant_sigma_would_make_this_file_vacuous():
    """The metric claim is only testable where the two metrics differ."""
    rows = _snapshots(jax.random.key(1), 4)
    flat = _basis(rows, weight=jnp.ones(N_DATA))
    graded = _basis(rows)
    # Under a flat weight the whitened rows ARE the raw rows, so an
    # L2-orthonormalisation and an N^-1 one coincide. Under the graded weight
    # they do not, and that difference is what every later test measures.
    np.testing.assert_allclose(
        np.asarray(flat.whitened), np.asarray(orthonormalise(rows)), atol=1e-12
    )
    assert not np.allclose(
        np.asarray(graded.whitened), np.asarray(orthonormalise(rows)), atol=1e-3
    )


def test_the_gram_factor_is_the_identity_after_orthonormalisation():
    basis = _basis(_snapshots(jax.random.key(2), 4))
    np.testing.assert_allclose(np.abs(np.asarray(basis.factor)), np.eye(4), atol=1e-12)


def test_project_then_expand_reproduces_anything_in_the_span():
    rows = _snapshots(jax.random.key(3), 4)
    basis = _basis(rows)
    inside = jnp.einsum("k,kd->d", jnp.array([1.0, -2.0, 0.5, 3.0]), rows)
    coefficients = basis.project(inside)
    np.testing.assert_allclose(
        np.asarray(basis.expand(coefficients)),
        np.asarray(basis.whiten(inside)),
        atol=1e-10,
    )
    assert float(basis.residual_fraction(inside)) < 1e-12


def test_out_of_span_content_stays_in_the_residual():
    rows = _snapshots(jax.random.key(4), 3)
    basis = _basis(rows)
    outside = jax.random.normal(jax.random.key(5), (N_DATA,))
    fraction = float(basis.residual_fraction(outside))
    assert 0.5 < fraction < 1.0
    # and the residual really is N^-1-orthogonal to the span, which is the
    # property the whole of section 5 requirement 1 rests on
    residual = basis.whiten(outside) - basis.expand(basis.project(outside))
    np.testing.assert_allclose(
        np.asarray(basis.whitened @ residual), np.zeros(3), atol=1e-10
    )


def test_raw_snapshots_are_ill_conditioned_and_orthonormalising_repairs_it():
    """Section 5 requirement 5, in the direction that is reproducible.

    The spec quotes measured kappa(G) = 1.4e8 at n_S = 4 and 2e16 at n_S = 16
    for RAW greedy/EIM snapshots. Those digits are a property of that bank, not
    of this fixture, so what is pinned here is the structural claim they were
    evidence for: a nearly-collinear candidate set has a Gram matrix no float64
    quadratic form survives, and orthonormalising is what makes it a projector.
    """
    base = _snapshots(jax.random.key(6), 1)
    rows = jnp.concatenate(
        [base + 1e-7 * _snapshots(jax.random.key(7 + i), 1) for i in range(4)], axis=0
    )
    raw = _basis(rows, orthonormal=False)
    fixed = _basis(rows, orthonormal=True)
    assert raw.condition() > 1e10
    assert fixed.condition() < 10.0


def test_a_flagged_sample_carries_zero_weight_and_does_not_enter_the_metric():
    """The raw rows keep their values; the whitened ones go to zero there.

    Storing the raw rows is what lets an epoch whose flags differ from the
    reference's use this dictionary at all -- `whitened / weight` would be
    infinite at exactly the samples the reference could not see.
    """
    weight = _weight().at[7].set(0.0)
    rows = _snapshots(jax.random.key(8), 3)
    basis = ReducedBasis(
        rows=rows,
        weight=weight,
        predict=lambda values: jnp.zeros(N_DATA),
        reference=jnp.zeros(N_DATA),
        orthonormal=False,
    )
    assert float(basis.whitened[:, 7] @ basis.whitened[:, 7]) == 0.0
    assert float(basis.rows[0, 7]) == float(rows[0, 7])


def test_a_non_finite_row_at_a_flagged_sample_is_still_a_usable_basis():
    """The legitimate case the finiteness refusal must not swallow.

    `0.0 * inf` is `nan`, so whitening by a plain multiply would turn a row that
    is infinite at a sample the reference could not see into a NaN, poison the
    whole Gram matrix through it, and then be refused by a message saying it
    fires only where the weight is non-zero. Whitening selects first.
    """
    weight = _weight().at[7].set(0.0)
    rows = _snapshots(jax.random.key(9), 3).at[1, 7].set(jnp.inf)
    basis = ReducedBasis(
        rows=rows,
        weight=weight,
        predict=lambda values: jnp.zeros(N_DATA),
        reference=jnp.zeros(N_DATA),
        orthonormal=False,
    )
    assert float(basis.whitened[1, 7]) == 0.0
    assert bool(jnp.all(jnp.isfinite(basis.gram())))


def test_a_non_finite_row_where_the_data_was_observed_is_refused():
    """The other side of the same guard: here there is no mask to hide behind."""
    with pytest.raises(StateValidationError, match="non-finite entry"):
        ReducedBasis(
            rows=_snapshots(jax.random.key(10), 3).at[1, 7].set(jnp.nan),
            weight=_weight(),
            predict=lambda values: jnp.zeros(N_DATA),
            reference=jnp.zeros(N_DATA),
            orthonormal=False,
        )


def test_more_basis_vectors_than_data_samples_is_refused():
    with pytest.raises(StateValidationError, match="more directions than"):
        ReducedBasis(
            rows=jnp.zeros((5, 3)),
            weight=jnp.ones(3),
            predict=lambda values: jnp.zeros(3),
            reference=jnp.zeros(3),
        )


def test_a_non_finite_weight_is_refused_rather_than_encoded():
    """`inf` is FlaggedNoise's encoding for sigma, never for 1/sigma."""
    with pytest.raises(StateValidationError, match="weight"):
        ReducedBasis(
            rows=jnp.zeros((1, 3)),
            weight=jnp.array([1.0, jnp.inf, 1.0]),
            predict=lambda values: jnp.zeros(3),
            reference=jnp.zeros(3),
        )
