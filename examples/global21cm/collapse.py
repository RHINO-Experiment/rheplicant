"""Integrate the linear foreground out exactly, leaving a 21 cm likelihood.

Both foreground models are linear with a Gaussian (demo B) or flat (demo A)
prior, and the 21 cm signal is the same curve ``c(theta)`` at every LST:
``d = J a + T c(theta) + n`` with ``T`` the ``(n_data, n_freq)`` tiling
matrix and ``n ~ Normal(0, sigma^2 I)``. Marginalising ``a ~ Normal(m, S)``,

    d ~ Normal(J m + T c,  C),   C = sigma^2 I + J S J^T,

so, up to a constant in theta, ``log L(theta) = b^T c - c^T G c / 2`` with
``G = T^T C^-1 T`` and ``b = T^T C^-1 (d - J m)``. With ``G = V diag(g) V^T``
that is a unit-variance Gaussian in the compressed statistic

    y = sigma0 g^-1/2 V^T b,   model  D c = sigma0 g^1/2 V^T c,

which is what the documents fit (:class:`global21cm.signal21.CompressedSignal`
with noise ``sigma0``). ``y`` is a fixed linear map ``S`` of the data, kept
for drawing new noise realisations. Directions with ``g`` below ``RCOND``
times the largest carry no information on the signal and are dropped; the
statistic is zero-padded to ``n_freq`` so it fits the channel grid.

Demo B's ``C`` is never formed: :class:`GaussianMarginal` holds a triangular
square root of it (see there), because ``J S J^T`` alone has eigenvalues up
to ~3e12 K^2 against ``sigma^2 = 1e-4 K^2``, and rounding in that Gram
matrix is as large as the noise term it is added to.

Demo A's flat prior is the limit ``S -> inf``, where ``C^-1`` becomes the
per-LST projector off the basis: ``G = n_time P_perp / sigma^2``. The oracle
has no foreground uncertainty: ``C = sigma^2 I`` and ``J m`` is the truth.

This is done here, outside the configuration layer, because the layer has no
route that integrates a linear block out of the likelihood (README).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import scipy.linalg as sla
from scipy.linalg import lapack

from global21cm import scenario

#: Eigenvalues of ``G`` below this fraction of the largest are dropped.
RCOND = 1e-12
#: The documents' noise scale for the compressed statistic, K.
SIGMA0 = scenario.NOISE_SIGMA_K
#: LAPACK block size of the stacked-triangle QR (``tpqrt``).
TPQRT_BLOCK = 64


@dataclass(frozen=True)
class Collapsed:
    """The compressed likelihood: ``y ~ Normal(D c(theta), SIGMA0^2 I)``."""

    design: np.ndarray  # (n_freq, n_freq), rows beyond the kept rank are zero
    data: np.ndarray  # (n_freq,)
    transform: np.ndarray  # (n_freq, n_data): y = transform @ (vec(d) - offset)
    offset: np.ndarray  # (n_data,)
    rank: int
    fit_residual: np.ndarray  # (n_data,): sigma^2 C^-1 (vec(d) - offset)
    fit_signal: np.ndarray  # (n_data, n_freq): sigma^2 C^-1 T
    centred: np.ndarray  # (n_data,): vec(d) - offset
    dof: int  # degrees of freedom of the marginal chi^2
    info: dict = field(default_factory=dict, compare=False)

    def residual(self, curve) -> np.ndarray:
        """Data minus the foreground's posterior mean and ``T curve``, frequency-major.

        For a Gaussian linear model this is ``sigma^2 C^-1 (d - J m - T c)``.
        """
        return self.fit_residual - self.fit_signal @ np.asarray(curve)

    def marginal_chi2(self, curve) -> float:
        """``r^T C^-1 r`` with ``r = d - J m - T c``: the whole waterfall's misfit.

        Chi-squared with :attr:`dof` degrees of freedom when the model is
        right, whatever direction the misfit takes; the compressed statistic
        sees only the directions the signal can move.
        """
        tiled = np.repeat(np.asarray(curve), self.centred.size // np.size(curve))
        return float((self.centred - tiled) @ self.residual(curve)) / scenario.NOISE_SIGMA_K**2

    def statistic(self, waterfall) -> np.ndarray:
        """``y`` for another waterfall ``(n_time, n_freq)`` (a noise realisation)."""
        return self.transform @ (vec(waterfall) - self.offset)

    def chi2(self, curves, y=None) -> np.ndarray:
        """``||D c - y||^2 / SIGMA0^2`` for curves ``(N, n_freq)``."""
        y = self.data if y is None else y
        r = np.atleast_2d(curves) @ self.design.T - y[None, :]
        return np.sum(r**2, axis=1) / SIGMA0**2


def vec(waterfall) -> np.ndarray:
    """``(n_time, n_freq)`` -> frequency-major vector, the Jacobians' row order."""
    return np.asarray(waterfall, dtype=np.float64).T.reshape(-1)


def _compress(g_matrix, cinv_t, residual, offset, n_freq, cinv_residual, dof) -> Collapsed:
    """From ``G``, ``C^-1 T`` and ``C^-1 (d - J m)`` to :class:`Collapsed`."""
    eigval, eigvec = np.linalg.eigh(0.5 * (g_matrix + g_matrix.T))
    keep = eigval > RCOND * eigval.max()
    g, v = eigval[keep], eigvec[:, keep]
    transform = SIGMA0 * (v / np.sqrt(g)[None, :]).T @ cinv_t.T
    design = SIGMA0 * (v * np.sqrt(g)[None, :]).T
    pad = n_freq - g.size
    return Collapsed(
        design=np.vstack([design, np.zeros((pad, n_freq))]),
        data=np.concatenate([transform @ residual, np.zeros(pad)]),
        transform=np.vstack([transform, np.zeros((pad, transform.shape[1]))]),
        offset=offset,
        rank=int(g.size),
        fit_residual=scenario.NOISE_SIGMA_K**2 * cinv_residual,
        fit_signal=scenario.NOISE_SIGMA_K**2 * cinv_t,
        centred=residual,
        dof=int(dof),
    )


def oracle(waterfall, fg_truth) -> Collapsed:
    """The foreground known exactly."""
    n_time, n_freq = np.shape(waterfall)
    sigma2 = scenario.NOISE_SIGMA_K**2
    cinv_t = np.kron(np.eye(n_freq), np.ones((n_time, 1))) / sigma2
    offset = vec(fg_truth)
    g = np.eye(n_freq) * n_time / sigma2
    residual = vec(waterfall) - offset
    return _compress(g, cinv_t, residual, offset, n_freq, residual / sigma2, residual.size)


def per_lst_basis(waterfall, basis) -> Collapsed:
    """Demo A: every LST's spectrum free on ``basis`` (flat prior)."""
    n_time, n_freq = np.shape(waterfall)
    sigma2 = scenario.NOISE_SIGMA_K**2
    q, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64))
    perp = np.eye(n_freq) - q @ q.T
    cinv_t = np.kron(perp, np.ones((n_time, 1))) / sigma2
    offset = np.zeros(n_time * n_freq)
    projected = vec(np.asarray(waterfall) @ perp) / sigma2
    dof = n_time * (n_freq - q.shape[1])
    return _compress(n_time * perp / sigma2, cinv_t, vec(waterfall), offset, n_freq, projected, dof)


@dataclass(frozen=True, eq=False)
class GaussianMarginal:
    """``C = sigma^2 I + J S J^T`` for ``a ~ Normal(loc, diag(scale^2))``, through a square root.

    With ``B = J S^1/2``, one Householder QR ``B^T = Q0 R0`` gives the
    triangular :attr:`root` ``R0`` with ``R0^T R0 = B B^T``; for each noise
    level the QR of the stacked triangles ``[R0; sigma I]`` (LAPACK ``tpqrt``)
    gives :attr:`factor` ``R`` with ``R^T R = C``. Then

        C^-1 x = R^-1 R^-T x,   log det C = 2 sum log|R_ii|,
        E[a | d] = m + S J^T C^-1 (d - J m).

    Both QRs are backward stable in ``B``: the result is exact for a ``B``
    perturbed by ``eps ||B||``. Forming ``B B^T`` perturbs ``C`` by
    ``eps ||B||^2`` instead, which for demo B exceeds ``sigma^2``.
    :meth:`at` reuses ``R0`` for another noise level.
    """

    jac: np.ndarray  # (n_data, n_par)
    loc: np.ndarray  # (n_par,)
    scale: np.ndarray  # (n_par,)
    root: np.ndarray  # (n_data, n_data) upper triangular, R0^T R0 = J S J^T
    sigma: float
    factor: np.ndarray  # (n_data, n_data) upper triangular, R^T R = C

    @classmethod
    def build(cls, jac, loc, scale, sigma: float = SIGMA0) -> GaussianMarginal:
        jac = np.asarray(jac, dtype=np.float64)
        scale = np.asarray(scale, dtype=np.float64)
        root = _gram_root(jac * scale[None, :])
        loc = np.asarray(loc, dtype=np.float64)
        return cls(jac, loc, scale, root, float(sigma), _noise_root(root, sigma))

    def at(self, sigma: float) -> GaussianMarginal:
        """The same prior at noise ``sigma``."""
        return replace(self, sigma=float(sigma), factor=_noise_root(self.root, sigma))

    def whiten(self, x) -> np.ndarray:
        """``R^-T x``, so that ``x^T C^-1 x = ||R^-T x||^2``."""
        return sla.solve_triangular(self.factor, x, trans="T", lower=False)

    def solve(self, x) -> np.ndarray:
        """``C^-1 x`` for ``x`` of shape ``(n_data,)`` or ``(n_data, k)``."""
        return sla.solve_triangular(self.factor, self.whiten(x), lower=False)

    @property
    def logdet(self) -> float:
        """``log det C``."""
        return 2.0 * float(np.sum(np.log(np.abs(np.diag(self.factor)))))

    def posterior_mean(self, data) -> np.ndarray:
        """``E[a | d] = m + S J^T C^-1 (d - J m)`` for ``data = vec(d)``."""
        centred = np.asarray(data, dtype=np.float64) - self.jac @ self.loc
        return self.loc + self.scale**2 * (self.jac.T @ self.solve(centred))

    def log_evidence(self, data) -> float:
        """``log N(d; J m, C)`` for ``data = vec(d)``."""
        white = self.whiten(np.asarray(data, dtype=np.float64) - self.jac @ self.loc)
        return -0.5 * (float(white @ white) + self.logdet + white.size * np.log(2.0 * np.pi))


def _gram_root(b) -> np.ndarray:
    """Upper-triangular ``R0``, ``(n, n)``, with ``R0^T R0 = b b^T`` for ``b`` ``(n, p)``."""
    n = b.shape[0]
    tri = sla.qr(b.T, mode="r")[0]  # (p, n): upper trapezoidal, and zero below row n
    root = np.zeros((n, n))
    rows = min(tri.shape[0], n)
    root[:rows] = np.triu(tri[:rows])
    return root


def _noise_root(root, sigma: float) -> np.ndarray:
    """Upper-triangular ``R``, ``R^T R = root^T root + sigma^2 I``: QR of ``[root; sigma I]``."""
    if not sigma > 0.0:
        raise ValueError(f"noise sigma must be positive, got {sigma!r}.")
    n = root.shape[0]
    a, _, _, info = lapack.dtpqrt(n, min(TPQRT_BLOCK, n), np.asfortranarray(root),
                                  np.asfortranarray(sigma * np.eye(n)))  # fmt: skip
    if info != 0:
        raise np.linalg.LinAlgError(f"tpqrt returned info = {info}.")
    return np.triu(a)


def gaussian_prior(waterfall, jac, loc, scale, sigma: float = SIGMA0) -> Collapsed:
    """Demo B: ``a ~ Normal(loc, diag(scale^2))`` through the Jacobian ``jac``, noise ``sigma``."""
    return from_marginal(waterfall, GaussianMarginal.build(jac, loc, scale, sigma))


def from_marginal(waterfall, marginal: GaussianMarginal) -> Collapsed:
    """:func:`gaussian_prior` from an already factorised :class:`GaussianMarginal`.

    ``fit_residual`` and ``fit_signal`` keep the documents' normalisation
    ``SIGMA0^2 C^-1``, so :meth:`Collapsed.residual` is the data minus the
    posterior-mean fit only at ``sigma = SIGMA0``; the statistic, the design
    and :meth:`Collapsed.marginal_chi2` hold at any ``sigma``.
    """
    n_time, n_freq = np.shape(waterfall)
    tiling = np.kron(np.eye(n_freq), np.ones((n_time, 1)))
    offset = marginal.jac @ marginal.loc
    residual = vec(waterfall) - offset
    solved = marginal.solve(np.column_stack([tiling, residual]))
    cinv_t, cinv_r = solved[:, :-1], solved[:, -1]
    return _compress(tiling.T @ cinv_t, cinv_t, residual, offset, n_freq, cinv_r, residual.size)


def amplitude_forecast(collapsed: Collapsed, shape, noiseless) -> dict[str, float]:
    """sigma and bias of a free scale ``a`` on the true curve, ``a = 1`` at the truth.

    Sampler-free: ``sigma(a)^-2 = ||D t||^2 / SIGMA0^2``, and the bias is the
    estimate from the noiseless waterfall minus 1, which is what model
    misfit alone does to it.
    """
    column = collapsed.design @ np.asarray(shape)
    norm = float(column @ column)
    estimate = float(column @ collapsed.statistic(noiseless)) / norm
    sigma = SIGMA0 / np.sqrt(norm)
    return {"sigma": float(sigma), "bias": estimate - 1.0, "bias_sigma": (estimate - 1.0) / sigma}


def save(collapsed: Collapsed, directory, name: str) -> None:
    """``<name>_data.npy`` (the documents' observation) and ``<name>_collapsed.npz``."""
    np.save(directory / f"{name}_data.npy", collapsed.data[None, :])  # the (1, n_freq) grid
    np.savez(directory / f"{name}_collapsed.npz", design=collapsed.design, data=collapsed.data,
             transform=collapsed.transform, offset=collapsed.offset, rank=collapsed.rank,
             fit_residual=collapsed.fit_residual, fit_signal=collapsed.fit_signal,
             centred=collapsed.centred, dof=collapsed.dof)  # fmt: skip


def load(directory, name: str) -> Collapsed:
    with np.load(directory / f"{name}_collapsed.npz") as z:
        parts = {k: z[k] for k in z.files}
    return Collapsed(**{**parts, "rank": int(parts["rank"]), "dof": int(parts["dof"])})


