"""The oracle is the one thing here with nothing above it, so it gets two routes.

Route one is the analytic marginal in `chain_bank.oracle`. Route two integrates
`zeta` numerically on a two-epoch, one-sample cut-down, where a product grid is
affordable. They share no code.

The last two tests are about the FIXTURE rather than the oracle: a chain whose
`PHI` is 0, or whose `Q` equals its stationary variance, cannot see the bugs
section 6 exists to prevent, and this file says so in executable form so that
nobody later "simplifies" the constants.
"""

import numpy as np
import pytest

from tests.evidence import chain_bank as bank


def _grid_oracle(theta, phi, process_var, initial_var, points=1201, span=9.0):
    """log p(d | theta) for a 2-epoch, 1-sample chain, by direct integration."""
    a = np.array([[1.3], [-0.7]])  # (epoch, theta_0)
    c = np.array([0.9, 1.4])
    d = np.array([0.6, -0.2])
    sigma = 0.3
    grid = np.linspace(-span, span, points)
    step = grid[1] - grid[0]

    def normal(x, mean, var):
        return np.exp(-0.5 * (x - mean) ** 2 / var) / np.sqrt(2 * np.pi * var)

    total = 0.0
    for z1 in grid:
        w1 = normal(z1, 0.0, initial_var) * normal(
            d[0], a[0, 0] * theta + c[0] * z1, sigma**2
        )
        if w1 == 0.0:
            continue
        z2 = grid
        w2 = normal(z2, phi * z1, process_var) * normal(
            d[1], a[1, 0] * theta + c[1] * z2, sigma**2
        )
        total += w1 * np.sum(w2) * step
    return float(np.log(total * step))


def _analytic_two_epoch(theta, phi, process_var, initial_var):
    a = np.array([[1.3], [-0.7]])
    c = np.array([0.9, 1.4])
    d = np.array([0.6, -0.2])
    sigma = 0.3
    prior = np.array(
        [
            [initial_var, phi * initial_var],
            [phi * initial_var, phi**2 * initial_var + process_var],
        ]
    )
    cov = sigma**2 * np.eye(2) + np.diag(c) @ prior @ np.diag(c)
    resid = d - a[:, 0] * theta
    _, logdet = np.linalg.slogdet(cov)
    return float(
        -0.5 * resid @ np.linalg.solve(cov, resid)
        - 0.5 * (logdet + 2 * np.log(2 * np.pi))
    )


@pytest.mark.parametrize("theta", [0.0, 0.7, -1.4])
def test_the_analytic_marginal_agrees_with_direct_integration(theta):
    """Two routes to the same number, sharing no code."""
    grid = _grid_oracle(theta, bank.PHI, bank.PROCESS_VAR, bank.INITIAL_VAR)
    analytic = _analytic_two_epoch(
        theta, bank.PHI, bank.PROCESS_VAR, bank.INITIAL_VAR
    )
    assert grid == pytest.approx(analytic, abs=1e-6)


def test_the_campaign_oracle_is_finite_and_peaks_near_the_truth():
    at_truth = bank.oracle([bank.TRUTH["t_rx"], bank.TRUTH["gain_slope"]])
    assert np.isfinite(at_truth)
    for probe in ([20.0, 0.0], [0.0, -30.0]):
        assert bank.oracle(probe) < at_truth


def test_zeta_covariance_satisfies_its_own_recursion():
    cov = bank.zeta_covariance()
    for e in range(bank.N_EPOCHS - 1):
        assert cov[e + 1, e + 1] == pytest.approx(
            bank.PHI**2 * cov[e, e] + bank.PROCESS_VAR, rel=1e-12
        )
        assert cov[e, e + 1] == pytest.approx(bank.PHI * cov[e, e], rel=1e-12)


def test_phi_zero_would_make_every_chain_test_a_bag_test():
    """The fixture's PHI is what stops this file's siblings passing vacuously."""
    linked = bank.zeta_covariance(phi=bank.PHI)
    unlinked = bank.zeta_covariance(phi=0.0)
    assert np.allclose(unlinked, np.diag(np.diag(unlinked)))
    off_diagonal = np.abs(linked - np.diag(np.diag(linked))).max()
    assert off_diagonal > 0.5  # measured 0.8 at PHI = 0.8, INITIAL_VAR = 1


def test_the_chain_is_deliberately_not_stationary():
    """`Q == stationary variance` would hide a swap of the two log-determinants."""
    stationary = bank.PROCESS_VAR / (1.0 - bank.PHI**2)
    assert abs(stationary - bank.INITIAL_VAR) > 0.3  # measured 0.389
    variances = np.diag(bank.zeta_covariance())
    assert variances[0] != pytest.approx(variances[-1], rel=1e-3)
