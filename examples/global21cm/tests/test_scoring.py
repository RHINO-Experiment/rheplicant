"""The scores of one posterior, on synthetic inputs whose answers are known."""

import numpy as np
import pytest
from scipy import stats

from global21cm import collapse, fom, scenario, scoring, signal21

SIGMA = scenario.NOISE_SIGMA_K


@pytest.mark.parametrize(
    "checks, expected",
    [
        ({"p_compressed": 0.5, "p_marginal": 0.5, "p_predictive": 0.5}, True),
        ({"p_compressed": 0.001, "p_marginal": 0.5, "p_predictive": 0.5}, False),
        ({"p_compressed": 0.5, "p_marginal": 0.001, "p_predictive": 0.5}, False),
        ({"p_compressed": 0.5, "p_marginal": 0.5, "p_predictive": 0.001}, False),
        ({"p_compressed": None, "p_marginal": 0.5, "p_predictive": 0.5}, True),
        ({"p_compressed": None, "p_marginal": 0.001, "p_predictive": 0.5}, False),
        ({"p_compressed": 0.01, "p_marginal": 0.01, "p_predictive": 0.01}, True),
        # A literal just below the level, not a multiple of GOF_ALPHA: the value is pinned.
        ({"p_compressed": 0.009, "p_marginal": 0.5, "p_predictive": 0.5}, False),
        ({"p_compressed": None, "p_marginal": None, "p_predictive": None}, False),
    ],
)
def test_gate_needs_every_available_check(checks, expected):
    assert scoring.passes(checks) is expected


def test_summary_of_synthetic_realisations():
    rows = {
        "cov68": [0.6, 0.7, 0.8, 0.5],
        "cov95": [1.0, 0.9, 0.95, 0.95],
        "depth_q": [0.5, 0.1, 0.9, 0.99],   # |q - 0.5| = 0, 0.4, 0.4, 0.49
        "amp_z": [0.5, -1.5, 2.5, 1.0],
    }
    out = scoring.summarise_realisations(rows)
    se68 = np.std(rows["cov68"], ddof=1) / 2
    assert out["cov68"]["mean"] == pytest.approx(0.65)
    assert out["cov68"]["se"] == pytest.approx(se68)
    assert out["cov68"]["deficit_se"] == pytest.approx((0.68 - 0.65) / se68)
    assert out["cov95"]["mean"] == pytest.approx(0.95)
    assert out["depth_in_central68"] == 0.25
    assert out["depth_in_central95"] == 0.75
    assert out["amplitude_within_1sigma"] == 0.5  # 0.5 and 1.0, the edge counted in
    assert out["amplitude_within_2sigma"] == 0.75
    assert out["n"] == 4


def test_summary_central_bands_at_their_edges():
    rows = {"cov68": [0.7] * 5, "cov95": [0.95] * 5, "amp_z": [0.0] * 5,
            "depth_q": [0.17, 0.83, 0.15, 0.03, 0.985]}  # |q - 0.5| = 0.33, 0.33, 0.35, 0.47, 0.485
    out = scoring.summarise_realisations(rows)
    assert out["depth_in_central68"] == 0.4  # half-width 0.34
    assert out["depth_in_central95"] == 0.8  # half-width 0.475


def test_realisation_noise_has_the_documents_scale():
    # The oracle's statistic has unit-variance rows in units of SIGMA0: fresh
    # noise of the waterfall's amplitude must come out with standard deviation SIGMA0.
    rng = np.random.default_rng(3)
    noiseless = rng.normal(size=(8, 12))
    item = collapse.oracle(noiseless, np.zeros_like(noiseless))
    draws = np.array([scoring.realisation(item, noiseless, rng) for _ in range(400)])
    assert np.std(draws - item.statistic(noiseless)) == pytest.approx(collapse.SIGMA0, rel=0.05)


def _flat_item(seed=7, n_time=8, n_freq=6, n_basis=3):
    rng = np.random.default_rng(seed)
    return collapse.per_lst_basis(SIGMA * rng.normal(size=(n_time, n_freq)), rng.normal(size=(n_freq, n_basis)))


def test_gof_marginal_p_value_uses_the_waterfall_dof():
    # 8 LSTs x (6 channels - 3 basis columns) = 24 degrees of freedom, not the rank 3.
    item = _flat_item()
    gof = scoring.goodness_of_fit(item, {"curves": np.zeros((50, 6))}, np.zeros(50), seed=1)
    assert gof["dof_marginal"] == 8 * 3
    assert gof["p_marginal"] == pytest.approx(stats.chi2.sf(gof["chi2_marginal"], 8 * 3), rel=1e-12)


def test_gof_scores_the_highest_likelihood_particle():
    item = _flat_item()
    curves = np.outer(np.linspace(0.0, 1.0, 5), np.ones(6)) * SIGMA
    chi2 = item.chi2(curves)
    assert chi2.min() < chi2.max()
    gof = scoring.goodness_of_fit(item, {"curves": curves}, -0.5 * chi2, seed=1)
    assert gof["chi2_compressed_best"] == pytest.approx(chi2.min(), rel=1e-12)


# ---- score(): troughs placed on the fine grid by construction --------------
FINE = np.arange(60.0, 80.01, 0.25)
N_DRAW = 14
DEPTHS = -(0.100 + 0.005 * np.arange(N_DRAW))  # -0.100 ... -0.165 K
WHERE_INDEX = 40 + (5 * np.arange(N_DRAW)) % N_DRAW  # a permutation of 40 ... 53
TRUE_CURVE = np.array([-0.05, -0.15, -0.10, -0.02])


def _score_inputs(u_offset, true_depth=-0.1525, true_where=71.1):
    """Draws whose trough depths, positions and latent covariance are known exactly."""
    fine = np.zeros((N_DRAW, FINE.size))
    fine[np.arange(N_DRAW), WHERE_INDEX] = DEPTHS
    # u = mean +/- sqrt(7) e_i: population covariance I, so z^2 = |mean - u_true|^2.
    u_true = np.linspace(-0.3, 0.3, 7)
    signs = np.sqrt(7.0) * np.concatenate([np.eye(7), -np.eye(7)])
    offsets = 0.01 * np.concatenate([np.eye(4), -np.eye(4)])
    draws = {"u": u_true + u_offset + signs, "fine": fine,
             "curves": TRUE_CURVE + np.concatenate([offsets, np.zeros((6, 4))])}  # fmt: skip
    prior = {"curves": TRUE_CURVE + 3.0 * offsets}
    truth = {"u": u_true, "curve": TRUE_CURVE, "depth": true_depth, "where": true_where}
    return draws, truth, prior


def test_score_reads_the_troughs_from_the_fine_curves():
    draws, truth, prior = _score_inputs(np.r_[0.5, np.zeros(6)])
    out = scoring.score(draws, truth, prior, FINE)
    assert out["depth_quantile"] == 3 / 14  # -0.155, -0.160, -0.165 lie below -0.1525
    assert out["position_quantile"] == 5 / 14  # 70.00 ... 71.00 MHz lie below 71.1
    assert out["depth_mk"] == pytest.approx([np.percentile(DEPTHS, p) * 1e3 for p in (16, 50, 84)], rel=1e-12)
    wheres = FINE[WHERE_INDEX]
    assert out["position_mhz"] == pytest.approx([np.percentile(wheres, p) for p in (16, 50, 84)], rel=1e-12)


def test_score_measures_error_against_the_truth_and_the_prior():
    draws, truth, prior = _score_inputs(np.r_[0.5, np.zeros(6)])
    out = scoring.score(draws, truth, prior, FINE)
    assert out == {**out, **fom.extraction(draws["curves"], TRUE_CURVE).as_dict()}
    # Draws +/- 0.01 on four channels, the prior +/- 0.03, both centred on the
    # truth, the prior over 8 draws and the posterior over 14.
    assert out["bias2"] == pytest.approx(0.0, abs=1e-30)
    assert out["variance_over_prior"] == pytest.approx((8 * 1e-4 / 14) / (8 * 9e-4 / 8), rel=1e-12)


def test_score_calibration_uses_the_latent_z2_and_both_quantiles():
    draws, truth, prior = _score_inputs(np.r_[0.5, np.zeros(6)])
    out = scoring.score(draws, truth, prior, FINE)
    assert out["z2_theta"] == pytest.approx(0.25, rel=1e-12)
    assert out["z2_tail"] == pytest.approx(stats.chi2.sf(0.25, 7), rel=1e-12)
    assert out["calibrated_signal"] is True
    # A tail of 0.007 fails the z^2 gate although both quantiles are central,
    # and lies inside [ALPHA/2, 1 - ALPHA/2]: a swapped argument would pass it.
    far = np.sqrt(stats.chi2.isf(0.007, 7))
    out = scoring.score(*_score_inputs(np.r_[far, np.zeros(6)]), FINE)
    assert out["z2_tail"] == pytest.approx(0.007, rel=1e-9)
    assert out["calibrated_signal"] is False
    assert scoring.score(*_score_inputs(np.zeros(7), true_depth=-0.2), FINE)["calibrated_signal"] is False
    assert scoring.score(*_score_inputs(np.zeros(7), true_where=79.0), FINE)["calibrated_signal"] is False


# ---- laplace_trace(): against a finite-difference Jacobian -----------------
@pytest.fixture(scope="module")
def curve_jacobian():
    """``(freqs, u_true, J)`` with ``J = dT21/du`` by central differences, not by jax."""
    freqs = np.linspace(50.0, 100.0, 12)
    u = np.asarray(signal21.u_true())
    low, high = (np.asarray(v) for v in signal21.prior_box())
    curve = lambda v: np.asarray(signal21.curve_kelvin(signal21.box_from_unit_normal(v, low, high), freqs))  # noqa: E731
    h = 1e-5
    jac = np.stack([(curve(u + h * e) - curve(u - h * e)) / (2 * h) for e in np.eye(7)], axis=1)
    return freqs, u, jac


def test_laplace_trace_without_data_is_the_prior_trace(curve_jacobian):
    # A zero design carries no information: F = I and the trace is tr(J J^T).
    freqs, u, jac = curve_jacobian
    assert scoring.laplace_trace(np.zeros((12, 12)), freqs, u) == pytest.approx(np.trace(jac @ jac.T), rel=1e-7)


def test_laplace_trace_is_the_curve_covariance_of_the_fisher_matrix(curve_jacobian):
    freqs, u, jac = curve_jacobian
    design = 0.01 * np.random.default_rng(0).normal(size=(12, 12))
    g = design.T @ design / collapse.SIGMA0**2
    expected = np.trace(jac @ np.linalg.solve(jac.T @ g @ jac + np.eye(7), jac.T))
    assert scoring.laplace_trace(design, freqs, u) == pytest.approx(expected, rel=1e-7)
