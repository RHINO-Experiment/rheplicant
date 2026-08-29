import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference.sqrtinfo import SqrtInfo


def test_log_prob_matches_the_dense_quadratic_it_encodes():
    factor = jnp.array([[2.0, 0.5], [0.0, 1.5]])
    target = jnp.array([1.0, -0.5])
    term = SqrtInfo(
        factor=factor, target=target, offset=jnp.array(3.0),
        names=("a", "b"), shapes=((), ()),
    )
    values = {"a": jnp.array(0.3), "b": jnp.array(-1.2)}

    expected = 3.0 - 0.5 * float(
        jnp.sum((factor @ jnp.array([0.3, -1.2]) - target) ** 2)
    )
    assert float(term.log_prob(values)) == pytest.approx(expected, abs=1e-12)


def test_log_prob_ravels_multi_element_latents_in_declared_order():
    factor = jnp.eye(3)
    term = SqrtInfo(
        factor=factor, target=jnp.zeros(3), offset=jnp.array(0.0),
        names=("vec", "scalar"), shapes=((2,), ()),
    )
    values = {"vec": jnp.array([1.0, 2.0]), "scalar": jnp.array(3.0)}
    assert float(term.log_prob(values)) == pytest.approx(-0.5 * 14.0, abs=1e-12)


def test_a_value_of_the_wrong_shape_is_refused():
    term = SqrtInfo(
        factor=jnp.eye(2), target=jnp.zeros(2), offset=jnp.array(0.0),
        names=("a", "b"), shapes=((), ()),
    )
    with pytest.raises(Exception, match="shape"):
        term.log_prob({"a": jnp.zeros(3), "b": jnp.array(0.0)})


def _random_term(key, width, rows, names=("a", "b", "c")):
    kf, kt, ko = jax.random.split(key, 3)
    return SqrtInfo(
        factor=jax.random.normal(kf, (rows, width)),
        target=jax.random.normal(kt, (rows,)),
        offset=jax.random.normal(ko, ()),
        names=names[:width],
        shapes=((),) * width,
    )


def _values(key, names):
    draws = jax.random.normal(key, (len(names),))
    return {name: draws[i] for i, name in enumerate(names)}


def test_combine_adds_the_two_log_densities_exactly():
    a = _random_term(jax.random.key(0), width=3, rows=2)
    b = _random_term(jax.random.key(1), width=3, rows=4)
    values = _values(jax.random.key(2), ("a", "b", "c"))

    combined = SqrtInfo.combine(a, b)
    assert float(combined.log_prob(values)) == pytest.approx(
        float(a.log_prob(values)) + float(b.log_prob(values)), abs=1e-10
    )


def test_combine_is_order_invariant():
    a = _random_term(jax.random.key(3), width=3, rows=2)
    b = _random_term(jax.random.key(4), width=3, rows=5)
    values = _values(jax.random.key(5), ("a", "b", "c"))
    assert float(SqrtInfo.combine(a, b).log_prob(values)) == pytest.approx(
        float(SqrtInfo.combine(b, a).log_prob(values)), abs=1e-10
    )


def test_combine_is_associative():
    a = _random_term(jax.random.key(6), width=3, rows=2)
    b = _random_term(jax.random.key(7), width=3, rows=2)
    c = _random_term(jax.random.key(8), width=3, rows=2)
    values = _values(jax.random.key(9), ("a", "b", "c"))
    left = SqrtInfo.combine(SqrtInfo.combine(a, b), c)
    right = SqrtInfo.combine(a, SqrtInfo.combine(b, c))
    assert float(left.log_prob(values)) == pytest.approx(
        float(right.log_prob(values)), abs=1e-10
    )


def test_the_null_term_is_the_identity_of_combine():
    a = _random_term(jax.random.key(10), width=3, rows=2)
    null = SqrtInfo.null(names=("a", "b", "c"), shapes=((), (), ()))
    values = _values(jax.random.key(11), ("a", "b", "c"))
    assert float(SqrtInfo.combine(a, null).log_prob(values)) == pytest.approx(
        float(a.log_prob(values)), abs=1e-12
    )


def test_combine_refuses_terms_over_different_latents():
    a = _random_term(jax.random.key(12), width=2, rows=2, names=("a", "b"))
    b = _random_term(jax.random.key(13), width=2, rows=2, names=("a", "c"))
    with pytest.raises(Exception, match="over different latents"):
        SqrtInfo.combine(a, b)


def test_fisher_is_R_transpose_R_and_stays_positive_semidefinite():
    a = _random_term(jax.random.key(14), width=3, rows=1)
    fisher = a.fisher()
    assert fisher.shape == (3, 3)
    assert np.linalg.eigvalsh(np.asarray(fisher)).min() >= -1e-12
    # rank-1 epoch: exactly one nonzero eigenvalue
    assert int(np.linalg.matrix_rank(np.asarray(fisher), tol=1e-9)) == 1


def test_a_thousand_epochs_stay_positive_definite():
    """The regression for explicit-Schur accumulation going indefinite."""
    running = SqrtInfo.null(names=("a", "b", "c"), shapes=((), (), ()))
    key = jax.random.key(15)
    scale = jnp.array([1.0, 1e-3, 1e-5])  # graded sensitivity
    for step in range(1000):
        sub = jax.random.fold_in(key, step)
        row = jax.random.normal(sub, (1, 3)) * scale
        running = SqrtInfo.combine(
            running,
            SqrtInfo(
                factor=row, target=jnp.zeros(1), offset=jnp.array(0.0),
                names=("a", "b", "c"), shapes=((), (), ()),
            ),
        )
    eigenvalues = np.linalg.eigvalsh(np.asarray(running.fisher()))
    assert eigenvalues.min() > 0.0


def test_dropping_the_qr_corner_would_be_wrong_and_this_pins_it():
    """Guard against a 'simplification' that discards rho^2."""
    a = _random_term(jax.random.key(16), width=2, rows=3, names=("a", "b"))
    b = _random_term(jax.random.key(17), width=2, rows=3, names=("a", "b"))
    combined = SqrtInfo.combine(a, b)
    # rho^2 > 0 whenever the stack over-determines x, which 6 rows over 2
    # columns always does.
    assert float(combined.offset) < float(a.offset + b.offset)


class TestAComplexFactorIsRefused:
    """`SqrtInfo` is a BILINEAR form, so a complex factor is a silent wrong answer.

    Migration ledger **D62**, mirroring **D46**, which the owner ruled for
    bayesmith on 2026-08-28 and placed at the same chokepoint. Two entries were
    needed rather than one because ``SqrtInfo`` is a D12 stay-container: the
    far side's array-level functions RETURN this class, so the near side's
    never becomes the far side's and the refusal could not arrive with a
    delegation. Both packages guard their own constructor; that is two doors,
    not one rule written twice.

    ``log_prob`` computes ``sum(residual**2)`` -- that is ``r^T r`` -- and
    ``fisher()`` computes ``factor.T @ factor`` with no conjugate. A complex
    QR's ``Q`` is UNITARY: it preserves ``r^H r``, not ``r^T r``. So the object
    silently answers a different question than the one its construction implies.
    """

    def test_it_is_refused_at_construction(self):
        with pytest.raises(Exception, match="real by construction"):
            SqrtInfo(
                factor=jnp.array([[1j]]),
                target=jnp.array([0j]),
                offset=jnp.array(0.0),
                names=("c",),
                shapes=((),),
            )

    def test_each_complex_part_is_named(self):
        """The message lists WHICH parts are complex, because a caller with a
        real factor and a complex target needs to know which one to fix."""
        with pytest.raises(Exception) as caught:
            SqrtInfo(
                factor=jnp.array([[1.0]]),
                target=jnp.array([1j]),
                offset=jnp.array(0.0),
                names=("c",),
                shapes=((),),
            )
        message = str(caught.value)
        assert "complex target" in message, message
        assert "factor and" not in message, "a real factor must not be blamed"

    def test_a_real_factor_is_still_accepted(self):
        """ANTI-VACUITY. A guard that refused every factor would pass the two
        cases above and take the whole evidence layer with it."""
        term = SqrtInfo(
            factor=jnp.array([[1.0]]),
            target=jnp.array([0.5]),
            offset=jnp.array(0.0),
            names=("c",),
            shapes=((),),
        )
        assert float(term.fisher()[0, 0]) == pytest.approx(1.0)

    def test_the_arithmetic_the_refusal_protects(self):
        """The SIBLING assertion: pin the numbers, because the refusal makes
        the behaviour they describe unreachable.

        Without this, the refusal's message cites a measurement that nothing
        can check any more, and a future reader has no way to tell whether the
        two forms really did disagree or whether somebody wrote a plausible
        number into a docstring. Computed here in NumPy, no `SqrtInfo`
        involved.
        """
        r1, r2 = np.array([[1j]]), np.array([[1.0 + 0j]])
        bilinear = (r1.T @ r1 + r2.T @ r2)[0, 0]
        unitary = (r1.conj().T @ r1 + r2.conj().T @ r2)[0, 0]
        assert bilinear == 0, bilinear
        assert unitary == 2, unitary
        assert bilinear != unitary, (
            "the two forms agree on this fixture, so it cannot demonstrate "
            "what the refusal exists for -- pick a fixture where they differ"
        )
