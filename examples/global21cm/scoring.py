"""Scores of one posterior: against the truth, the prior, the data and the noise.

The headline posterior of every model is the tempered SMC of
:mod:`global21cm.smc` on the compressed likelihood; the NUTS runs the
documents make are compared with it. Four groups of numbers:

* **error** (:func:`score`): SER, relative to the prior, bias, and the
  calibration quantities of :mod:`global21cm.fom`;
* **goodness of fit** (:func:`goodness_of_fit`): the compressed chi^2 at the
  posterior's best particle against its rank, the whole waterfall's marginal
  chi^2 at the posterior-mean curve against its degrees of freedom, and a
  posterior-predictive p-value;
* **forecast** (:func:`laplace_trace`): the Laplace curve covariance at the
  truth, which gives the sampler-free efficiency;
* **coverage** (:func:`realisations`): fresh noise on the noiseless
  waterfall, an SMC posterior for each realisation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from scipy import stats

from global21cm import collapse, fom, scenario, signal21, smc

#: Frequency step of the grid the trough depth and position are read on.
FINE_STEP_MHZ = 0.25
#: Tail probability below which a goodness-of-fit check fails.
GOF_ALPHA = 0.01
N_PPC = 2000


def fine_freqs(case: scenario.Scenario) -> np.ndarray:
    return np.arange(case.freq_start_mhz, case.freq_stop_mhz + 1e-9, FINE_STEP_MHZ)


def score(draws: dict, truth: dict, prior: dict, freqs_fine) -> dict:
    """Error and calibration of one posterior; ``draws`` holds ``u``, ``curves``, ``fine``."""
    depth, where = fom.trough(draws["fine"], freqs_fine)
    q_depth = fom.quantile_of(truth["depth"], depth)
    q_where = fom.quantile_of(truth["where"], where)
    z2, tail = fom.parameter_z2(draws["u"], truth["u"])
    return {
        **fom.extraction(draws["curves"], truth["curve"]).as_dict(),
        **fom.relative_to_prior(draws["curves"], prior["curves"], truth["curve"]),
        "z2_theta": z2,
        "z2_tail": tail,
        "depth_quantile": q_depth,
        "position_quantile": q_where,
        "depth_mk": [float(np.percentile(depth, p) * 1e3) for p in (16, 50, 84)],
        "position_mhz": [float(np.percentile(where, p)) for p in (16, 50, 84)],
        "calibrated_signal": fom.calibrated(tail, q_depth, q_where),
    }


def goodness_of_fit(collapsed: collapse.Collapsed, draws: dict, loglik, seed: int) -> dict:
    """Three checks that look at the whole fit, not only along the signal."""
    best = int(np.argmax(loglik))
    chi2_best = float(collapsed.chi2(draws["curves"][best])[0])
    dof_compressed = collapsed.rank - len(signal21.THETA_NAMES)
    marginal = collapsed.marginal_chi2(draws["curves"].mean(axis=0))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(draws["curves"]), size=min(N_PPC, len(draws["curves"])), replace=False)
    model = draws["curves"][pick] @ collapsed.design.T
    noise = collapse.SIGMA0 * rng.standard_normal(model.shape)
    noise[:, collapsed.rank :] = 0.0  # the padded rows carry no data
    observed = np.sum((collapsed.data[None, :] - model) ** 2, axis=1)
    replicated = np.sum(noise**2, axis=1)
    out = {
        "chi2_compressed_best": chi2_best,
        "dof_compressed": dof_compressed,
        # With rank <= 7 the seven parameters can absorb the whole statistic:
        # there is nothing left to test, and the check is reported as absent.
        "p_compressed": float(stats.chi2.sf(chi2_best, dof_compressed)) if dof_compressed > 0 else None,
        "chi2_marginal": marginal,
        "dof_marginal": collapsed.dof,
        "p_marginal": float(stats.chi2.sf(marginal, collapsed.dof)),
        "p_predictive": float(np.mean(replicated >= observed)),
    }
    out["passes"] = passes(out)
    return out


def passes(checks: dict) -> bool:
    """Every available p-value at least ``GOF_ALPHA``; an absent one is skipped."""
    values = [checks[k] for k in ("p_compressed", "p_marginal", "p_predictive") if checks[k] is not None]
    return bool(values) and all(v >= GOF_ALPHA for v in values)


def laplace_trace(design, freqs, u_true) -> float:
    """``tr[J F^-1 J^T]``: the Laplace curve covariance at the truth, K^2.

    ``J = dT21/du`` at the true latents and ``F = J^T G J + I`` with
    ``G = D^T D / SIGMA0^2``, the compressed likelihood's precision.
    """
    low, high = (jnp.asarray(v) for v in signal21.prior_box())
    curve = lambda u: signal21.curve_kelvin(signal21.box_from_unit_normal(u, low, high), jnp.asarray(freqs))  # noqa: E731
    jac = np.asarray(jax.jacfwd(curve)(jnp.asarray(u_true)))
    g = np.asarray(design).T @ np.asarray(design) / collapse.SIGMA0**2
    fisher = jac.T @ g @ jac + np.eye(jac.shape[1])
    return float(np.trace(jac @ np.linalg.solve(fisher, jac.T)))


def realisation(collapsed: collapse.Collapsed, noiseless, rng) -> np.ndarray:
    """The statistic of the noiseless waterfall plus one fresh noise draw.

    Not ``collapsed.data`` plus noise: the data already carry one noise
    realisation, and adding another centres every realisation on it.
    """
    noise = scenario.NOISE_SIGMA_K * rng.standard_normal(np.shape(noiseless))
    return collapsed.statistic(np.asarray(noiseless) + noise)


def realisations(collapsed, truth, noiseless, freqs, fine, n_real: int, seed: int) -> dict:
    """Coverage over fresh noise on the noiseless waterfall, SMC per realisation."""
    rng = np.random.default_rng(seed)
    column = collapsed.design @ truth["curve"]
    norm = float(column @ column)
    rows = {"cov68": [], "cov95": [], "depth_q": [], "amp_z": []}
    evaluate = smc.evaluator(collapsed.design, freqs)
    box = tuple(np.asarray(v) for v in signal21.prior_box())
    for i in range(n_real):
        y = realisation(collapsed, noiseless, rng)
        particles = smc.run(lambda u, y=jnp.asarray(y): evaluate(u, y), seed + 1 + i, n=4000, n_move=20)
        theta = np.asarray(signal21.box_from_unit_normal(particles["u"], *box))
        curves = np.asarray(signal21.curve_kelvin(theta, freqs))
        pit = np.mean(curves < truth["curve"][None, :], axis=0)
        depth = fom.trough(np.asarray(signal21.curve_kelvin(theta, fine)), fine)[0]
        rows["cov68"].append(np.mean((pit >= 0.16) & (pit <= 0.84)))
        rows["cov95"].append(np.mean((pit >= 0.025) & (pit <= 0.975)))
        rows["depth_q"].append(fom.quantile_of(truth["depth"], depth))
        rows["amp_z"].append((float(column @ y) / norm - 1.0) * np.sqrt(norm) / collapse.SIGMA0)
    return summarise_realisations(rows)


def summarise_realisations(rows: dict) -> dict:
    """Means, standard errors and deficits in units of the standard error."""
    n = len(rows["cov68"])
    out = {"n": n}
    for key, nominal in (("cov68", 0.68), ("cov95", 0.95)):
        values = np.asarray(rows[key], dtype=np.float64)
        se = float(values.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
        out[key] = {"mean": float(values.mean()), "se": se,
                    "deficit_se": (nominal - float(values.mean())) / se if se else float("nan")}  # fmt: skip
    q, z = np.asarray(rows["depth_q"]), np.abs(np.asarray(rows["amp_z"]))
    out["depth_in_central68"] = float(np.mean(np.abs(q - 0.5) <= 0.34))
    out["depth_in_central95"] = float(np.mean(np.abs(q - 0.5) <= 0.475))
    out["amplitude_within_1sigma"] = float(np.mean(z <= 1.0))
    out["amplitude_within_2sigma"] = float(np.mean(z <= 2.0))
    return out
