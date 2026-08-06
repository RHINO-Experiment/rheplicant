"""A linked-scope campaign, and the dense oracle everything else is measured against.

Not a test file -- imported by the chain's test modules, which is why the repo root
has to be on sys.path (`python -m pytest`, never bare `pytest`).

Three numbers are chosen against specific ways a chain test goes quiet:

* `PHI = 0.8`. At `PHI = 0` the chain IS a bag -- every transition block becomes
  a plain prior on the next epoch's zeta, the filter degenerates to
  `compress_linear` per epoch, and every bug in the augmentation is invisible.
* `Q = 0.5`, `P0 = 1.0`, which is deliberately NOT the stationary variance
  `Q / (1 - PHI^2) = 1.389`. A stationary chain has one variance for every
  epoch, so an implementation that used `P0` where it meant `Q` -- or dropped
  one of the two log-determinants into the other -- would agree with the oracle.
* Two globals declared `("t_rx", "gain_slope")`. Sorted order is
  `("gain_slope", "t_rx")`, so anything that builds a matrix in declared order
  and labels it in flatten order is wrong by a permutation with somewhere to
  show. That permutation is the identity for an alphabetical fixture, which is
  how Plan A's `BayesMemory.fisher()` bug survived its own tests.

`C_e` carries a `+1.0` offset so the per-epoch response to zeta has a nonzero
mean: a zero-mean random `C_e` would make the campaign's sensitivity to the
chain grow as sqrt(N) rather than N, and the transition would matter less than
it should.
"""

import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.inference.compress import compress_linear
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace
from rheplicant.inference.sqrtinfo import SqrtInfo

N_EPOCHS, N_SAMPLES = 6, 4
PHI, PROCESS_VAR, INITIAL_VAR, SIGMA = 0.8, 0.5, 1.0, 0.3
INITIAL_MEAN = 0.0

THETA_NAMES = ("t_rx", "gain_slope")  # NOT sorted: see the module docstring
ZETA_NAME = "t_rx_drift"
N_THETA, N_ZETA = len(THETA_NAMES), 1
WIDTH = N_THETA + N_ZETA

TRUTH = {"t_rx": 1.0, "gain_slope": -0.5}
PRIOR_STD = {"t_rx": 2.0, "gain_slope": 2.0}

PROCESS_STD = float(np.sqrt(PROCESS_VAR))
INITIAL_STD = float(np.sqrt(INITIAL_VAR))


class _Normal:
    """Minimal duck-typed prior: `log_prob` and `scale` are all that is read."""

    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        return -0.5 * (
            ((x - self.loc) / self.scale) ** 2 + jnp.log(2 * jnp.pi * self.scale**2)
        )


def design(seed=0):
    """`(A, C, d)` -- per-epoch theta design, zeta design and data."""
    keys = jax.random.split(jax.random.key(seed), 3 * N_EPOCHS)
    A = [
        np.asarray(jax.random.normal(keys[e], (N_SAMPLES, N_THETA)))
        for e in range(N_EPOCHS)
    ]
    C = [
        np.asarray(jax.random.normal(keys[N_EPOCHS + e], (N_SAMPLES, 1))) + 1.0
        for e in range(N_EPOCHS)
    ]
    d = [
        np.asarray(jax.random.normal(keys[2 * N_EPOCHS + e], (N_SAMPLES,)))
        for e in range(N_EPOCHS)
    ]
    return A, C, d


def space():
    latents = [
        Latent(
            name,
            init=jnp.asarray(TRUTH[name]),
            prior=_Normal(TRUTH[name], PRIOR_STD[name]),
        )
        for name in THETA_NAMES
    ] + [
        Latent(
            ZETA_NAME,
            init=jnp.asarray(0.0),
            prior=_Normal(0.0, INITIAL_STD),
            scope="linked",
        )
    ]
    return ParameterSpace(
        latents=tuple(latents),
        bindings=tuple(
            Bind(latent.name, into=lambda p, n=latent.name: getattr(p, n))
            for latent in latents
        ),
    )


def factorization(transition):
    return Factorization(space(), linked={ZETA_NAME: transition})


def blocks(seed=0):
    """One square `(WIDTH, WIDTH)` SqrtInfo per epoch over `(*THETA_NAMES, zeta)`.

    Square rather than `(rank, WIDTH)` because `lax.scan` needs one shape for
    every iteration. `SqrtInfo.combine(null, info)` is the padding: it is the
    same QR the accumulator uses, so the offset it produces is the one the
    filter will consume, corner included.
    """
    A, C, d = design(seed)
    out = []
    for e in range(N_EPOCHS):
        term = compress_linear(
            design={
                "t_rx": A[e][:, :1],
                "gain_slope": A[e][:, 1:],
                ZETA_NAME: C[e],
            },
            observed=jnp.asarray(d[e]),
            noise_std=SIGMA,
            shapes={"t_rx": (), "gain_slope": (), ZETA_NAME: ()},
            epoch_id=f"e{e}",
        )
        square = SqrtInfo.combine(
            SqrtInfo.null(term.info.names, term.info.shapes), term.info
        )
        out.append(square)
    return tuple(out)


def stacked(seed=0):
    """`(factor (N, w, w), target (N, w), offset (N,))` -- what the scan eats."""
    infos = blocks(seed)
    return (
        jnp.stack([info.factor for info in infos]),
        jnp.stack([info.target for info in infos]),
        jnp.stack([jnp.asarray(info.offset) for info in infos]),
    )


def zeta_covariance(
    phi=PHI, process_var=PROCESS_VAR, initial_var=INITIAL_VAR, n=N_EPOCHS
):
    """`Sigma_zeta` for `zeta_{e+1} = phi zeta_e + w`, `var(zeta_1) = initial_var`."""
    var = np.zeros(n)
    var[0] = initial_var
    for e in range(1, n):
        var[e] = phi**2 * var[e - 1] + process_var
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov[i, j] = phi ** abs(i - j) * var[min(i, j)]
    return cov


def oracle(theta, seed=0, phi=PHI, process_var=PROCESS_VAR, initial_var=INITIAL_VAR):
    """`log p(d_1:N | theta)` with zeta_1:N integrated ANALYTICALLY.

    The whole campaign as one multivariate normal:
    `d ~ N(A theta + C E[zeta], sigma^2 I + C Sigma_zeta C^T)`. No recursion, no
    square roots, no shared code with `chain.py` -- which is the point. A bug
    that lives in the recursion cannot live here too.
    """
    A, C, d = design(seed)
    theta = np.asarray(theta, dtype=float)
    n_data = N_EPOCHS * N_SAMPLES
    big_a = np.zeros((n_data, N_THETA))
    big_c = np.zeros((n_data, N_EPOCHS))
    for e in range(N_EPOCHS):
        rows = slice(e * N_SAMPLES, (e + 1) * N_SAMPLES)
        big_a[rows, :] = A[e]
        big_c[rows, e] = C[e][:, 0]
    mean_zeta = np.array([phi**e * INITIAL_MEAN for e in range(N_EPOCHS)])
    cov = SIGMA**2 * np.eye(n_data) + big_c @ zeta_covariance(
        phi, process_var, initial_var
    ) @ big_c.T
    resid = np.concatenate(d) - big_a @ theta - big_c @ mean_zeta
    _, logdet = np.linalg.slogdet(cov)
    return float(
        -0.5 * resid @ np.linalg.solve(cov, resid)
        - 0.5 * (logdet + n_data * np.log(2 * np.pi))
    )


def oracle_zeta_posterior(theta, seed=0):
    """`(mean (N,), cov (N, N))` for zeta_1:N given all the data and theta."""
    A, C, d = design(seed)
    theta = np.asarray(theta, dtype=float)
    n_data = N_EPOCHS * N_SAMPLES
    big_a = np.zeros((n_data, N_THETA))
    big_c = np.zeros((n_data, N_EPOCHS))
    for e in range(N_EPOCHS):
        rows = slice(e * N_SAMPLES, (e + 1) * N_SAMPLES)
        big_a[rows, :] = A[e]
        big_c[rows, e] = C[e][:, 0]
    prior = zeta_covariance()
    mean_zeta = np.array([PHI**e * INITIAL_MEAN for e in range(N_EPOCHS)])
    cov = SIGMA**2 * np.eye(n_data) + big_c @ prior @ big_c.T
    gain = prior @ big_c.T @ np.linalg.inv(cov)
    resid = np.concatenate(d) - big_a @ theta - big_c @ mean_zeta
    return mean_zeta + gain @ resid, prior - gain @ big_c @ prior


PROBES = ([0.0, 0.0], [0.4, -1.1], [2.0, 3.0], [-5.0, 0.7])
