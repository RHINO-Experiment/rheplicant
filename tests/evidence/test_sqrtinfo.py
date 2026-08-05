import jax.numpy as jnp
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
