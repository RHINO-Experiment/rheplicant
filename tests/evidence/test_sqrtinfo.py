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
