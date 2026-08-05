import jax.numpy as jnp
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compressed import CompressedLikelihood, QuadraticLikelihood
from rheplicant.inference.sqrtinfo import SqrtInfo


def _term(**overrides):
    kwargs = dict(
        info=SqrtInfo(
            factor=jnp.eye(2), target=jnp.zeros(2), offset=jnp.array(0.0),
            names=("a", "b"), shapes=((), ()),
        ),
        epoch_id="night-001",
        n_observed=1024,
        exact=True,
        include_logdet=True,
        noise_frozen_at="none",
    )
    kwargs.update(overrides)
    return QuadraticLikelihood(**kwargs)


def test_it_satisfies_the_protocol():
    assert isinstance(_term(), CompressedLikelihood)


def test_calling_it_gives_the_log_likelihood():
    values = {"a": jnp.array(1.0), "b": jnp.array(2.0)}
    assert float(_term()(values)) == pytest.approx(-2.5, abs=1e-12)


def test_it_reports_the_latents_it_is_over():
    assert _term().latents == ("a", "b")


def test_an_approximate_term_must_declare_a_support():
    with pytest.raises(StateValidationError, match="support"):
        _term(exact=False)


def test_an_exact_term_must_not_declare_a_support():
    with pytest.raises(StateValidationError, match="exact"):
        _term(support={"a": (-1.0, 1.0)})


def test_a_float32_term_is_refused_at_construction():
    with pytest.raises(StateValidationError, match="float64"):
        _term(
            info=SqrtInfo(
                factor=jnp.eye(2, dtype=jnp.float32),
                target=jnp.zeros(2, dtype=jnp.float32),
                offset=jnp.array(0.0, dtype=jnp.float32),
                names=("a", "b"), shapes=((), ()),
            )
        )


def test_the_prior_share_defaults_to_zero_because_terms_are_prior_free():
    assert _term().prior_share == (0, 1)


def test_a_prior_share_that_is_not_a_rational_is_refused():
    with pytest.raises(StateValidationError, match="integers"):
        _term(prior_share=(0.25, 1.0))


def test_an_estimator_key_groups_the_provenance_that_must_not_be_mixed():
    assert _term().estimator == ("full", "none")
    assert _term(include_logdet=False).estimator == ("gls", "none")
    assert _term(noise_frozen_at="gls").estimator == ("full", "gls")
