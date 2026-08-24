"""Tests for the matrix-free spectral diagnostics.

These pin the numbers a convergence guard is only as good as. The operators
here are dense matrices wrapped as callables, so every estimate can be checked
against ``jnp.linalg.eigvalsh`` — the point being that the power iteration is
matrix-free, not that it is doing something the dense routine could not.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.conditioning import (
    extreme_eigenvalues,
    largest_eigenvalue,
    tree_norm,
)


def _operator_with_spectrum(eigenvalues, seed=0):
    """A symmetric positive-definite operator with exactly this spectrum."""
    n = len(eigenvalues)
    basis, _ = jnp.linalg.qr(jax.random.normal(jax.random.key(seed), (n, n)))
    matrix = basis @ jnp.diag(jnp.asarray(eigenvalues, dtype=jnp.zeros(1).dtype)) @ basis.T
    return (lambda x: matrix @ x), matrix


class TestExtremeEigenvalues:
    @pytest.mark.parametrize(
        ("spectrum", "label"),
        [
            ([1.0, 3.0, 7.0, 20.0], "distinct"),
            ([451.0, 451.0, 1e-4], "degenerate top, tiny bottom"),
            ([1e-4, 1e-4, 451.0], "the under-determined block's own spectrum"),
            ([5.0, 5.0, 5.0], "fully degenerate"),
            ([2.0, 1e3, 1e6], "wide dynamic range"),
        ],
    )
    def test_it_recovers_a_known_spectrum(self, spectrum, label):
        operator, matrix = _operator_with_spectrum(spectrum)
        exact = jnp.linalg.eigvalsh(matrix)
        largest, smallest = extreme_eigenvalues(
            operator, jnp.zeros(len(spectrum)), jax.random.key(1), 30
        )
        assert float(largest) == pytest.approx(float(exact[-1]), rel=1e-3), label
        # lam_min is a difference of two numbers the size of lam_max, so it is
        # only ever resolved to that scale -- which is precisely why callers
        # floor it with an independent bound rather than trusting it near zero.
        assert float(smallest) == pytest.approx(
            float(exact[0]), abs=1e-3 * float(exact[-1])
        ), label

    def test_the_top_estimate_never_exceeds_the_truth(self):
        """Power iteration approaches lam_max from below.

        Worth pinning: it means a condition number built from it is an
        UNDER-estimate, so the guard errs towards accepting rather than towards
        crying wolf, and the two-sided estimate has to be trusted accordingly.
        """
        operator, matrix = _operator_with_spectrum([1.0, 4.0, 9.0])
        exact = float(jnp.linalg.eigvalsh(matrix)[-1])
        for iterations in (1, 2, 3, 5, 10):
            estimate = float(
                largest_eigenvalue(operator, jnp.zeros(3), jax.random.key(2), iterations)
            )
            assert estimate <= exact * (1 + 1e-5)

    def test_it_settles_within_a_few_iterations(self):
        """The default iteration count is a margin, not a requirement."""
        operator, matrix = _operator_with_spectrum([1e-4, 1e-4, 451.0])
        exact = float(jnp.linalg.eigvalsh(matrix)[-1])
        settled = float(
            largest_eigenvalue(operator, jnp.zeros(3), jax.random.key(3), 3)
        )
        assert settled == pytest.approx(exact, rel=1e-3)

    def test_it_works_on_a_pytree_domain(self):
        """A complex latent is carried as a (real, imag) tuple, so the domain
        is a pytree rather than an array."""
        def operator(parts):
            return (3.0 * parts[0], 0.5 * parts[1])

        largest, smallest = extreme_eigenvalues(
            operator, (jnp.zeros(4), jnp.zeros(4)), jax.random.key(4), 30
        )
        assert float(largest) == pytest.approx(3.0, rel=1e-3)
        assert float(smallest) == pytest.approx(0.5, rel=1e-2)


class TestTreeNorm:
    def test_it_matches_a_flat_euclidean_norm(self):
        parts = (jnp.array([3.0, 4.0]), jnp.array([12.0]))
        assert float(tree_norm(parts)) == pytest.approx(13.0)

    def test_it_survives_entries_that_would_overflow_when_squared(self):
        """Squaring 1e20 overflows float32; the norm must still come back
        finite, because it is the only convergence signal these solvers give
        and it is needed most when the scaling is worst."""
        parts = jnp.array([1e20, 1e20], dtype=jnp.float32)
        assert jnp.isfinite(tree_norm(parts))
        assert float(tree_norm(parts)) == pytest.approx(1e20 * jnp.sqrt(2.0), rel=1e-5)
