"""Adaptive-tempering sequential Monte Carlo over the seven 21 cm latents.

The reference posterior every sampler is checked against, and the headline
posterior of the scores. Neither bayesmith nor numpyro ships an SMC or
tempering sampler (checked in this environment), so this is a port of the
independent reviewer's ``smcL.py`` (review round 2), unchanged in method:

* particles start as prior draws ``u ~ Normal(0, I)``;
* the inverse temperature steps from 0 to 1, each step chosen by bisection
  so the incremental weights keep an effective sample size of ``ess_frac N``;
* after each resample (systematic), ``n_move`` random-walk Metropolis moves
  with the particles' own covariance, scale adapted towards 30 % acceptance;
* ``log Z`` accumulates the mean incremental weight of each step, so the
  relative mass of modes is set by the likelihood, not by which mode a chain
  happened to start in.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

from global21cm import collapse, signal21

DIM = len(signal21.THETA_NAMES)


def evaluator(design, freqs):
    """jitted ``(u (N, 7), y) -> -chi^2 / 2``; compiled once for every ``y``."""
    low, high = (jnp.asarray(v) for v in signal21.prior_box())
    d, s0, freqs = jnp.asarray(design), collapse.SIGMA0, jnp.asarray(freqs)

    @jax.jit
    def evaluate(u, target):
        curves = signal21.curve_kelvin(signal21.box_from_unit_normal(u, low, high), freqs)
        return -0.5 * jnp.sum((curves @ d.T - target[None, :]) ** 2, axis=1) / s0**2

    return evaluate


def log_likelihood(design, y, freqs):
    """``u (N, 7) -> -chi^2 / 2`` of the compressed statistic ``y``."""
    evaluate, target = evaluator(design, freqs), jnp.asarray(y)
    return lambda u: evaluate(u, target)


def _tempered(loglik, step):
    """``step * loglik`` with ``-inf`` kept (``0 * -inf`` would be nan)."""
    return np.where(np.isfinite(loglik), step * loglik, -np.inf)


def _increment(loglik, beta, ess_frac, n):
    def ess(step):
        w = np.exp(_tempered(loglik, step) - np.max(_tempered(loglik, step)))
        return w.sum() ** 2 / (w**2).sum()

    if ess(1.0 - beta) >= ess_frac * n:
        return 1.0 - beta
    lo, hi = 0.0, 1.0 - beta
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if ess(mid) >= ess_frac * n else (lo, mid)
    # No positive step keeps the target ESS when part of the particles have
    # log L = -inf (a hard constraint): take the rest of the way at once.
    return lo if lo > 0.0 else 1.0 - beta


def _move(u, loglik, beta, evaluate, rng, n_move):
    chol = np.linalg.cholesky(np.cov(u, rowvar=False) + 1e-12 * np.eye(DIM))
    scale, accepted = 2.38 / np.sqrt(DIM), 0.0
    for _ in range(n_move):
        proposal = u + scale * rng.standard_normal(u.shape) @ chol.T
        proposed = np.asarray(evaluate(proposal))
        log_a = beta * (proposed - loglik) - 0.5 * (np.sum(proposal**2, 1) - np.sum(u**2, 1))
        accept = np.log(rng.random(u.shape[0])) < log_a
        u = np.where(accept[:, None], proposal, u)
        loglik = np.where(accept, proposed, loglik)
        accepted += accept.mean()
        scale *= np.exp(accept.mean() - 0.3)
    return u, loglik, accepted / n_move


def run(evaluate, seed: int, n: int = 20000, n_move: int = 30, ess_frac: float = 0.6) -> dict:
    """Particles ``u`` (equal weight), their log likelihood, and ``log Z``."""
    rng = np.random.default_rng(seed)
    u = rng.standard_normal((n, DIM))
    loglik = np.asarray(evaluate(u))
    beta, log_z, steps, acceptance = 0.0, 0.0, 0, []
    while beta < 1.0:
        step = _increment(loglik, beta, ess_frac, n)
        log_w = _tempered(loglik, step)
        log_z += float(logsumexp(log_w) - np.log(n))
        w = np.exp(log_w - log_w.max())
        w /= w.sum()
        index = np.minimum(np.searchsorted(np.cumsum(w), (rng.random() + np.arange(n)) / n), n - 1)
        u, loglik, beta, steps = u[index], loglik[index], beta + step, steps + 1
        u, loglik, rate = _move(u, loglik, beta, evaluate, rng, n_move)
        acceptance.append(float(rate))
    return {"u": u, "loglik": loglik, "log_z": log_z, "steps": steps, "acceptance": acceptance}
