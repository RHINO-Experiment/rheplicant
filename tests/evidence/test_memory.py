import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.compressed import QuadraticLikelihood
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory
from rheplicant.inference.sqrtinfo import SqrtInfo


class _Normal:
    """Minimal duck-typed prior: only log_prob is used by BayesMemory."""

    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        return -0.5 * (((x - self.loc) / self.scale) ** 2 + jnp.log(2 * jnp.pi * self.scale**2))


def _factorization():
    latents = (
        Latent("depth", init=-0.5, prior=_Normal(0.0, 1.0)),
        Latent("width", init=1.0, prior=_Normal(0.0, 2.0)),
    )
    space = ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )
    return Factorization(space)


def _term(epoch_id, key=None, rows=1):
    names, shapes = ("depth", "width"), ((), ())
    if key is None:
        factor, target = jnp.zeros((rows, 2)), jnp.zeros(rows)
    else:
        kf, kt = jax.random.split(key)
        factor = jax.random.normal(kf, (rows, 2))
        target = jax.random.normal(kt, (rows,))
    return QuadraticLikelihood(
        info=SqrtInfo(factor=factor, target=target, offset=jnp.array(0.0),
                      names=names, shapes=shapes),
        epoch_id=epoch_id, n_observed=1024,
    )


def test_an_empty_memory_returns_exactly_the_prior():
    memory = BayesMemory(_factorization())
    values = {"depth": jnp.array(0.3), "width": jnp.array(-1.1)}
    expected = float(_Normal(0.0, 1.0).log_prob(0.3) + _Normal(0.0, 2.0).log_prob(-1.1))
    assert float(memory.log_posterior(values)) == pytest.approx(expected, abs=1e-12)
    assert float(memory.log_likelihood(values)) == 0.0


@pytest.mark.parametrize("n_epochs", [1, 2, 10, 1000])
def test_null_terms_leave_the_posterior_exactly_equal_to_the_prior(n_epochs):
    """The prior-count pin: N null terms must not multiply the prior N times."""
    memory = BayesMemory(_factorization())
    for index in range(n_epochs):
        memory = memory.remember(_term(f"null-{index}"))
    values = {"depth": jnp.array(0.3), "width": jnp.array(-1.1)}
    expected = float(_Normal(0.0, 1.0).log_prob(0.3) + _Normal(0.0, 2.0).log_prob(-1.1))
    assert float(memory.log_posterior(values)) == pytest.approx(expected, abs=1e-10)


def test_remembering_accumulates_the_log_likelihood():
    terms = [_term(f"n{i}", key=jax.random.key(i)) for i in range(5)]
    memory = BayesMemory(_factorization())
    for term in terms:
        memory = memory.remember(term)
    values = {"depth": jnp.array(0.3), "width": jnp.array(-1.1)}
    assert float(memory.log_likelihood(values)) == pytest.approx(
        sum(float(term(values)) for term in terms), abs=1e-10
    )


def test_the_same_night_cannot_be_remembered_twice_by_accident():
    memory = BayesMemory(_factorization()).remember(_term("night-001", jax.random.key(0)))
    with pytest.raises(StateValidationError, match="already"):
        memory.remember(_term("night-001", jax.random.key(1)))


def test_a_deliberate_duplicate_is_allowed_and_doubles_the_information():
    term = _term("night-001", jax.random.key(0))
    once = BayesMemory(_factorization()).remember(term)
    twice = once.remember(term, duplicate=True)
    assert float(np.trace(np.asarray(twice.fisher().matrix))) == pytest.approx(
        2 * float(np.trace(np.asarray(once.fisher().matrix))), rel=1e-10
    )


def test_two_estimators_cannot_be_summed():
    memory = BayesMemory(_factorization()).remember(_term("a", jax.random.key(0)))
    gls = QuadraticLikelihood(
        info=_term("b", jax.random.key(1)).info,
        epoch_id="b", n_observed=1024, include_logdet=False,
    )
    with pytest.raises(StateValidationError, match="estimator"):
        memory.remember(gls)


def test_a_term_over_different_latents_is_refused():
    memory = BayesMemory(_factorization())
    other = QuadraticLikelihood(
        info=SqrtInfo(factor=jnp.zeros((1, 1)), target=jnp.zeros(1),
                      offset=jnp.array(0.0), names=("depth",), shapes=((),)),
        epoch_id="x", n_observed=1,
    )
    with pytest.raises(StateValidationError, match="different latents"):
        memory.remember(other)


def test_a_term_carrying_a_prior_share_is_refused_on_the_streaming_path():
    memory = BayesMemory(_factorization())
    tempered = QuadraticLikelihood(
        info=_term("a", jax.random.key(0)).info,
        epoch_id="a", n_observed=1024, prior_share=(1, 300),
    )
    with pytest.raises(StateValidationError, match="prior-free"):
        memory.remember(tempered)


def test_the_treedef_does_not_change_as_epochs_are_remembered():
    """The compile-cost guard: one trace for a whole campaign, not one per epoch."""
    memory = BayesMemory(_factorization())
    first = jax.tree.structure(memory.accumulated)
    for index in range(20):
        memory = memory.remember(_term(f"n{index}", jax.random.key(index)))
    assert jax.tree.structure(memory.accumulated) == first


def test_fisher_is_named_and_refuses_to_pretend_to_be_a_covariance():
    memory = BayesMemory(_factorization()).remember(_term("a", jax.random.key(0), rows=2))
    fisher = memory.fisher()
    assert fisher.kind == "fisher"
    assert fisher.names == ("depth", "width")
    with pytest.raises(StateValidationError, match="not a standard deviation"):
        fisher.sigma("depth")


def test_fisher_spans_index_the_flat_vector_not_the_latent_list():
    """A (4,) latent covers four columns, not one.

    ``spans`` is what ``block`` and ``sigma`` slice with, so an off-by-one here
    reports a neighbouring parameter's number under this parameter's name --
    finite, correctly shaped, and wrong. Pinned against a non-scalar latent
    because every other test in this file uses scalars, where a span-per-latent
    bug and a span-per-element bug agree.
    """
    latents = (
        Latent("depth", init=-0.5, prior=_Normal(0.0, 1.0)),
        Latent("alms", init=jnp.zeros(4), prior=_Normal(0.0, 2.0)),
    )
    space = ParameterSpace(
        latents=latents,
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )
    fisher = BayesMemory(Factorization(space)).fisher()
    assert fisher.spans == ((0, 1), (1, 5))
    assert fisher.matrix.shape == (5, 5)
    assert fisher.span("alms") == (1, 5)
    assert fisher.block("alms").shape == (4, 4)


def test_audit_reports_the_epoch_count_and_the_conditioning():
    memory = BayesMemory(_factorization())
    for index in range(4):
        memory = memory.remember(_term(f"n{index}", jax.random.key(index), rows=2))
    report = memory.audit()
    assert report["n_epochs"] == 4
    assert report["estimator"] == ("full", "none")
    assert report["prior_shares_sum"] == 0
    assert np.isfinite(report["fisher_lambda_min"])
    assert np.isfinite(report["fisher_condition"])
