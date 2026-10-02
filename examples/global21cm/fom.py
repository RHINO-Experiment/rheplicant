"""Figures of merit for how well a 21 cm curve was extracted.

Pure numpy functions of posterior draws ``T`` of the curve, shape
``(S, F)`` for ``S`` draws on ``F`` frequencies, the draws' parameters, and
the injected truth. Nothing here knows about foregrounds or samplers.

**Error.** ``SER = sqrt(||t||^2 / E_s ||T_s - t||^2)``, where the mean
squared error splits exactly into bias and spread,
``E_s ||T_s - t||^2 = ||mean - t||^2 + tr Cov`` (population covariance).
SER alone has no natural threshold: the prior itself scores above 1 when
its curves are on average closer to the truth than zero is. So every
posterior is also scored against the prior's own draws:
``SER / SER_prior`` and the variance ratio ``tr Cov_post / tr Cov_prior``.

**Calibration.** Three gated checks with fixed degrees of freedom:

* ``z^2 = (mean_u - u_true)^T Cov_u^-1 (mean_u - u_true)`` in the seven
  unit-normal latents (chi-squared with 7 degrees of freedom for a
  calibrated Gaussian posterior), and its tail probability;
* the posterior quantile of the true trough depth;
* the posterior quantile of the true trough frequency (each quantile is
  uniform over realisations for a calibrated posterior).

A posterior is called calibrated when the z^2 tail probability is at least
``ALPHA`` and both trough quantiles lie in ``[ALPHA/2, 1 - ALPHA/2]``. The
efficiency ``eta = sqrt(tr Cov_oracle / tr Cov)`` is only meaningful then.
Band coverage per channel (:func:`coverage`) is reported but is not a
gate: on one realisation it is weak, so ``scoring.realisations`` measures
it over many noise realisations.

Degenerate inputs are answered, not refused: a single draw or a zero-spread
posterior has no covariance, so its ``z^2`` is 0 on the truth and ``inf``
off it; ``t = 0`` makes SER 0 for any error and ``nan`` for none.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import stats

#: Squared errors below ``F (ROUNDOFF_ULPS eps scale)^2`` are float64
#: roundoff: the mean of identical draws differs from each in the last bit.
ROUNDOFF_ULPS = 64
#: Two-sided level of the calibration checks.
ALPHA = 0.01


@dataclass(frozen=True)
class Extraction:
    """The error scores of one posterior against one truth."""

    ser: float
    mse: float
    bias2: float
    variance: float
    bias_fraction: float
    coverage68: float
    coverage95: float

    def as_dict(self) -> dict:
        return asdict(self)


def _check(draws, truth) -> tuple[np.ndarray, np.ndarray]:
    draws = np.atleast_2d(np.asarray(draws, dtype=np.float64))
    truth = np.asarray(truth, dtype=np.float64)
    if truth.ndim != 1 or draws.shape[1] != truth.shape[0]:
        raise ValueError(f"draws {draws.shape} and truth {truth.shape} are not (S, F) and (F,).")
    if not (np.all(np.isfinite(draws)) and np.all(np.isfinite(truth))):
        raise ValueError("draws and truth must be finite.")
    return draws, truth


def roundoff_floor(draws, truth) -> float:
    """The squared-error level below which a difference is float64 roundoff."""
    scale = max(float(np.abs(draws).max()), float(np.abs(truth).max()))
    return truth.shape[0] * (ROUNDOFF_ULPS * np.finfo(np.float64).eps * scale) ** 2


def mse_decomposition(draws, truth) -> tuple[float, float, float]:
    """``(mse, bias^2, variance)``; either term at or below the roundoff floor is 0."""
    draws, truth = _check(draws, truth)
    floor = roundoff_floor(draws, truth)
    mean = draws.mean(axis=0)
    bias2 = float(np.sum((mean - truth) ** 2))
    variance = float(np.sum((draws - mean) ** 2) / draws.shape[0])
    bias2 = 0.0 if bias2 <= floor else bias2
    variance = 0.0 if variance <= floor else variance
    return bias2 + variance, bias2, variance


def signal_to_error(draws, truth) -> float:
    """SER of the module docstring."""
    mse, _, _ = mse_decomposition(draws, truth)
    signal = float(np.sum(np.asarray(truth, dtype=np.float64) ** 2))
    if mse == 0.0:
        return float("nan") if signal == 0.0 else float("inf")
    return float(np.sqrt(signal / mse))


def coverage(draws, truth, level: float) -> float:
    """Fraction of channels whose truth lies in the central ``level`` band."""
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}.")
    draws, truth = _check(draws, truth)
    tail = 50.0 * (1.0 - level)
    low, high = np.percentile(draws, [tail, 100.0 - tail], axis=0)
    return float(np.mean((truth >= low) & (truth <= high)))


def extraction(draws, truth) -> Extraction:
    """The error scores of the module docstring for one posterior."""
    mse, bias2, variance = mse_decomposition(draws, truth)
    return Extraction(
        ser=signal_to_error(draws, truth),
        mse=mse,
        bias2=bias2,
        variance=variance,
        bias_fraction=bias2 / mse if mse > 0.0 else 0.0,
        coverage68=coverage(draws, truth, 0.68),
        coverage95=coverage(draws, truth, 0.95),
    )


def spread(draws) -> float:
    """``tr Cov`` of the draws, 0 at or below the roundoff floor."""
    draws = np.atleast_2d(np.asarray(draws, dtype=np.float64))
    mean = draws.mean(axis=0)
    trace = float(np.sum(draws.var(axis=0)))
    return 0.0 if trace <= roundoff_floor(draws, mean) else trace


def relative_to_prior(posterior, prior, truth) -> dict[str, float]:
    """``SER / SER_prior`` and ``tr Cov_post / tr Cov_prior``."""
    ser_post, ser_prior = signal_to_error(posterior, truth), signal_to_error(prior, truth)
    return {"ser_over_prior": ser_post / ser_prior, "variance_over_prior": spread(posterior) / spread(prior)}


def parameter_z2(u_draws, u_true) -> tuple[float, float]:
    """``(z^2, tail probability)`` of the truth in the latent space, 7 dof for 7 latents."""
    u = np.atleast_2d(np.asarray(u_draws, dtype=np.float64))
    offset = u.mean(axis=0) - np.asarray(u_true, dtype=np.float64)
    if u.shape[0] < 2 or spread(u) == 0.0:
        z2 = 0.0 if float(offset @ offset) <= roundoff_floor(u, u.mean(axis=0)) else float("inf")
    else:
        z2 = float(offset @ np.linalg.solve(np.cov(u, rowvar=False, ddof=0), offset))
    return z2, float(stats.chi2.sf(z2, df=u.shape[1]))


def trough(curves, freqs) -> tuple[np.ndarray, np.ndarray]:
    """``(depth, frequency)`` of each curve's minimum, ``(S,)`` each."""
    curves = np.atleast_2d(np.asarray(curves, dtype=np.float64))
    index = np.argmin(curves, axis=1)
    return curves[np.arange(curves.shape[0]), index], np.asarray(freqs)[index]


def quantile_of(value: float, samples) -> float:
    """Fraction of ``samples`` below ``value`` (ties count half)."""
    samples = np.asarray(samples, dtype=np.float64)
    return float(np.mean(samples < value) + 0.5 * np.mean(samples == value))


def calibrated(z2_tail: float, depth_quantile: float, position_quantile: float) -> bool:
    """The calibration test of the module docstring."""
    inside = lambda q: ALPHA / 2 <= q <= 1 - ALPHA / 2  # noqa: E731
    return z2_tail >= ALPHA and inside(depth_quantile) and inside(position_quantile)


def efficiency(oracle_draws, joint_draws) -> float:
    """``eta = sqrt(tr Cov_oracle / tr Cov_joint)``; ``nan`` when neither has spread."""
    tr_oracle, tr_joint = spread(oracle_draws), spread(joint_draws)
    if tr_joint == 0.0:
        return float("nan") if tr_oracle == 0.0 else float("inf")
    return float(np.sqrt(tr_oracle / tr_joint))


def perpendicular_fraction(columns, signal, rcond: float = 1e-12) -> float:
    """``||P_perp t|| / ||t||``: the share of the signal no foreground can mimic.

    ``columns`` is ``(D, J)``, the directions the foreground model can move
    the data in; same quantity as MERS's
    ``Global_21cm_extraction.projection_in_fg_null_space``.
    """
    columns = np.asarray(columns, dtype=np.float64)
    signal = np.asarray(signal, dtype=np.float64)
    norm = float(np.linalg.norm(signal))
    if norm == 0.0:
        return float("nan")
    u, s, _ = np.linalg.svd(columns, full_matrices=False)
    basis = u[:, s > rcond * s.max()] if s.size and s.max() > 0.0 else u[:, :0]
    residual = signal - basis @ (basis.T @ signal)
    return float(np.linalg.norm(residual) / norm)
