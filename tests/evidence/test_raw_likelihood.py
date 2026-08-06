"""T0: the likelihood with the data still in it.

Every assertion here is on an ABSOLUTE log-density. A tier that is right up to
an additive constant is wrong: Plan A shipped two such bugs (a dropped QR corner
and a missing -sum(log(std))), both invisible in a posterior's shape and fatal
to any evidence.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compress import compress_linear
from rheplicant.inference.compressed import RawLikelihood
from rheplicant.inference.memory import BayesMemory
from tests.evidence.test_memory import _factorization


def _linear_oracle(design, observed, sigma, x):
    resid = np.asarray(observed) - np.asarray(design) @ np.asarray(x)
    return -0.5 * float(
        np.sum(resid**2) / sigma**2 + len(resid) * np.log(2 * np.pi * sigma**2)
    )


def _term(design, observed, sigma, epoch_id="e0", **kwargs):
    return RawLikelihood(
        predict=lambda values: design @ values["x"],
        observed=observed,
        sigma=sigma,
        names=("x",),
        epoch_id=epoch_id,
        **kwargs,
    )


def test_t0_is_the_gaussian_log_density_it_claims_to_be():
    design = jax.random.normal(jax.random.key(0), (40, 3))
    observed = jax.random.normal(jax.random.key(1), (40,))
    term = _term(design, observed, 0.1)
    for probe in ([0.0, 0.0, 0.0], [0.4, -1.1, 2.0]):
        x = jnp.asarray(probe)
        assert float(term({"x": x})) == pytest.approx(
            _linear_oracle(design, observed, 0.1, x), rel=1e-12
        )


def test_t0_and_t2_agree_absolutely_on_a_linear_model():
    """The pin that makes T0 usable as an oracle for the tier ladder."""
    design = jax.random.normal(jax.random.key(2), (40, 3))
    observed = jax.random.normal(jax.random.key(3), (40,))
    raw = _term(design, observed, 0.1)
    quadratic = compress_linear(
        design={"x": design}, observed=observed, noise_std=0.1,
        shapes={"x": (3,)}, epoch_id="e0",
    )
    for seed in range(4):
        probe = {"x": jax.random.normal(jax.random.key(seed + 10), (3,))}
        assert float(raw(probe)) == pytest.approx(float(quadratic(probe)), rel=1e-11)


def test_a_flagged_sample_leaves_a_finite_term_and_a_smaller_n_observed():
    """D21's trap: sigma = inf must not take the normalisation with it."""
    design = jax.random.normal(jax.random.key(4), (8, 2))
    sigma = jnp.array([0.1, jnp.inf, 0.1, 0.1, 0.1, 0.1, jnp.inf, 0.1])
    term = _term(design, jax.random.normal(jax.random.key(5), (8,)), sigma)
    assert term.n_observed == 6
    assert np.isfinite(float(term({"x": jnp.zeros(2)})))


def test_a_nan_under_a_flag_does_not_reach_the_density():
    """`0.0 * nan` is `nan`. The mask must SELECT, never weight.

    This test can fail, which was checked rather than assumed. On this fixture
    the four candidate implementations give, measured: `jnp.where(seen, r, 0.0)`
    -- what ships -- `-150.0`; a float weight `r * jnp.where(seen, 1.0, 0.0)`,
    which is the shape Plan A actually shipped, `nan`; no mask at all, `r /
    sigma**2` with `sigma = inf`, `nan`.

    The one variant it does NOT catch is `r * seen` with `seen` left boolean:
    that also gives `-150.0`, because XLA lowers a float-by-bool product as a
    select and never forms `0.0 * nan`. So the bool form is accidentally
    correct here and would stop being so the moment anyone wrote
    `seen.astype(float)`. Named rather than relied on.

    The gradient is asserted alongside the value because this package's
    recurring defect is a forward pass that is insensitive to something the
    gradient is not. It is finite for the reason that matters: the NaN sits in
    `observed`, a constant, so the `where`'s VJP hands the flagged slot a zero
    cotangent instead of differentiating a NaN.
    """
    design = jnp.ones((4, 1))
    observed = jnp.array([1.0, jnp.nan, 1.0, 1.0])
    sigma = jnp.array([0.1, jnp.inf, 0.1, 0.1])
    term = _term(design, observed, sigma)
    assert np.isfinite(float(term({"x": jnp.zeros(1)})))
    gradient = jax.grad(term)({"x": jnp.zeros(1)})
    assert np.all(np.isfinite(np.asarray(gradient["x"])))


def test_dropping_the_normalisation_is_gls_and_says_so():
    design = jax.random.normal(jax.random.key(6), (12, 2))
    observed = jax.random.normal(jax.random.key(7), (12,))
    full = _term(design, observed, 0.1)
    gls = _term(design, observed, 0.1, include_logdet=False)
    assert full.estimator == ("full", "none")
    assert gls.estimator == ("gls", "none")
    assert float(full({"x": jnp.zeros(2)})) - float(gls({"x": jnp.zeros(2)})) == (
        pytest.approx(-0.5 * 12 * float(jnp.log(2 * jnp.pi * 0.01)), rel=1e-12)
    )


def test_a_memory_refuses_to_remember_the_oracle():
    memory = BayesMemory(_factorization())
    term = RawLikelihood(
        predict=lambda values: jnp.zeros(4),
        observed=jnp.zeros(4),
        sigma=jnp.full(4, 0.1),
        names=("depth", "width"),
        epoch_id="e0",
    )
    with pytest.raises(StateValidationError, match="oracle"):
        memory.remember(term)


def test_the_archive_refuses_a_term_it_cannot_describe(tmp_path):
    from rheplicant.inference.archive import save_memory

    memory = BayesMemory(_factorization())
    term = RawLikelihood(
        predict=lambda values: jnp.zeros(4), observed=jnp.zeros(4),
        sigma=jnp.full(4, 0.1), names=("depth", "width"), epoch_id="e0",
    )
    memory = BayesMemory(memory.factorization, memory.accumulated, (term,))
    with pytest.raises(StateValidationError, match="reconstruction spec"):
        save_memory(memory, tmp_path / "campaign.eqx")
