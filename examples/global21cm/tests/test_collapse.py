"""The compressed likelihood against the dense marginal likelihood it replaces.

On a small random problem the exact log marginal likelihood
``log N(d; J m + T c, sigma^2 I + J S J^T)`` is computed densely for several
curves ``c``; the compressed ``-chi2 / 2`` must differ from it by one
constant, the same for every curve.

Demo B's square root (:class:`collapse.GaussianMarginal`) is checked against
a 50-digit reference on a problem as ill-conditioned as demo B's, over noise
1e-4 .. 1 K and prior widths 1e-3 .. 1e3 times the base, where forming
``J S J^T`` loses the answer or fails.
"""

import mpmath
import numpy as np
import pytest
from scipy import stats

from global21cm import collapse, scenario

N_TIME, N_FREQ, N_PAR = 6, 5, 7
SIGMA = scenario.NOISE_SIGMA_K


def _tiling():
    return np.kron(np.eye(N_FREQ), np.ones((N_TIME, 1)))


def _dense_loglike(data_vec, mean, cov):
    return stats.multivariate_normal(mean, cov, allow_singular=False).logpdf(data_vec)


@pytest.fixture
def problem():
    rng = np.random.default_rng(0)
    jac = rng.normal(size=(N_TIME * N_FREQ, N_PAR))
    loc, scale = rng.normal(size=N_PAR), 0.5 + rng.random(N_PAR)
    waterfall = rng.normal(size=(N_TIME, N_FREQ))
    curves = 0.1 * rng.normal(size=(4, N_FREQ))
    return jac, loc, scale, waterfall, curves


def _constant_offset(compressed_ll, dense_ll):
    diff = np.asarray(dense_ll) - np.asarray(compressed_ll)
    return np.ptp(diff) / np.abs(diff).max()


def test_gaussian_prior_matches_the_dense_marginal(problem):
    jac, loc, scale, waterfall, curves = problem
    item = collapse.gaussian_prior(waterfall, jac, loc, scale)
    cov = SIGMA**2 * np.eye(jac.shape[0]) + (jac * scale**2) @ jac.T
    dense = [_dense_loglike(collapse.vec(waterfall), jac @ loc + _tiling() @ c, cov) for c in curves]
    assert _constant_offset(-0.5 * item.chi2(curves), dense) < 1e-10


def test_flat_prior_is_the_wide_gaussian_limit(problem):
    _, _, _, waterfall, curves = problem
    basis = np.random.default_rng(1).normal(size=(N_FREQ, 2))
    jac = np.kron(np.eye(N_TIME), basis)  # per-LST coefficients, time-major...
    order = np.arange(N_TIME * N_FREQ).reshape(N_TIME, N_FREQ).T.reshape(-1)
    jac = jac[order]  # ...rows reordered to frequency-major
    flat = collapse.per_lst_basis(waterfall, basis)

    def gap(width):
        wide = collapse.gaussian_prior(waterfall, jac, np.zeros(jac.shape[1]), np.full(jac.shape[1], width))
        return np.ptp(-0.5 * wide.chi2(curves) + 0.5 * flat.chi2(curves))

    # The two methods agree at the boundary S -> inf, and approach it as 1 / width^2.
    # Measured: 7.27e-3, 7.27e-5, 7.27e-7 at widths 10, 100, 1000, against chi^2 up to 4e4;
    # at 1e6 the basis directions fall below RCOND and the gap is rounding, 6e-12. Forming
    # sigma^2 I + width^2 J J^T instead gave 7.75e-5 at width 100 and 7.4e-3 at 1000.
    gaps = [gap(w) for w in (10.0, 100.0, 1000.0)]
    assert gaps[0] > 1e-3
    assert gaps[0] / gaps[1] == pytest.approx(100.0, rel=1e-3)
    assert gaps[1] / gaps[2] == pytest.approx(100.0, rel=1e-3)
    assert gap(1e6) < 1e-12 * np.abs(flat.chi2(curves)).max()


def test_oracle_is_the_waterfall_minus_the_truth(problem):
    _, _, _, waterfall, curves = problem
    fg = np.ones_like(waterfall)
    item = collapse.oracle(waterfall, fg)
    direct = np.array([np.sum((waterfall - fg - c[None, :]) ** 2) for c in curves]) / SIGMA**2
    diff = item.chi2(curves) - direct
    assert np.ptp(diff) < 1e-8 * np.abs(direct).max()


def test_statistic_of_the_same_waterfall_is_the_data(problem):
    jac, loc, scale, waterfall, _ = problem
    item = collapse.gaussian_prior(waterfall, jac, loc, scale)
    np.testing.assert_allclose(item.statistic(waterfall), item.data, atol=1e-12)


def test_residual_is_data_minus_posterior_mean_fit(problem):
    jac, loc, scale, waterfall, curves = problem
    item = collapse.gaussian_prior(waterfall, jac, loc, scale)
    cov = SIGMA**2 * np.eye(jac.shape[0]) + (jac * scale**2) @ jac.T
    shifted = collapse.vec(waterfall) - _tiling() @ curves[0]
    mean = loc + scale**2 * (jac.T @ np.linalg.solve(cov, shifted - jac @ loc))
    np.testing.assert_allclose(item.residual(curves[0]), shifted - jac @ mean, atol=1e-10)


def test_amplitude_forecast_is_unbiased_on_a_noiseless_model(problem):
    jac, loc, scale, _, curves = problem
    noiseless = (jac @ loc + _tiling() @ curves[0]).reshape(N_FREQ, N_TIME).T
    item = collapse.gaussian_prior(noiseless, jac, loc, scale)
    forecast = collapse.amplitude_forecast(item, curves[0], noiseless)
    assert abs(forecast["bias"]) < 1e-9 and forecast["sigma"] > 0.0


def test_amplitude_sigma_is_the_matched_filter_error():
    rng = np.random.default_rng(0)
    curve = 0.1 * rng.normal(size=N_FREQ)
    fg = 100.0 + rng.random((N_TIME, N_FREQ))
    noiseless = fg + curve[None, :]
    forecast = collapse.amplitude_forecast(collapse.oracle(noiseless, fg), curve, noiseless)
    assert forecast["sigma"] == pytest.approx(SIGMA / np.sqrt(N_TIME * curve @ curve), rel=1e-10)


def test_amplitude_bias_reads_the_noiseless_waterfall_not_the_data(problem):
    jac, loc, scale, _, curves = problem
    noiseless = (jac @ loc + _tiling() @ curves[0]).reshape(N_FREQ, N_TIME).T
    observed = noiseless + 50 * SIGMA * np.random.default_rng(1).normal(size=noiseless.shape)
    item = collapse.gaussian_prior(observed, jac, loc, scale)
    assert abs(collapse.amplitude_forecast(item, curves[0], noiseless)["bias"]) < 1e-9
    # The observed statistic is far from the noiseless one, so reading it would show.
    column = item.design @ curves[0]
    assert abs(float(column @ item.data) / float(column @ column) - 1.0) > 0.1


def test_save_and_load_round_trip(problem, tmp_path):
    jac, loc, scale, waterfall, curves = problem
    item = collapse.gaussian_prior(waterfall, jac, loc, scale)
    collapse.save(item, tmp_path, "x")
    back = collapse.load(tmp_path, "x")
    np.testing.assert_array_equal(back.chi2(curves), item.chi2(curves))
    assert (back.rank, back.dof) == (item.rank, item.dof)
    assert back.marginal_chi2(curves[0]) == item.marginal_chi2(curves[0])
    np.testing.assert_array_equal(np.load(tmp_path / "x_data.npy"), item.data[None, :])


def test_marginal_chi2_is_the_dense_mahalanobis_distance(problem):
    jac, loc, scale, waterfall, curves = problem
    item = collapse.gaussian_prior(waterfall, jac, loc, scale)
    cov = SIGMA**2 * np.eye(jac.shape[0]) + (jac * scale**2) @ jac.T
    r = collapse.vec(waterfall) - jac @ loc - _tiling() @ curves[1]
    assert item.marginal_chi2(curves[1]) == pytest.approx(r @ np.linalg.solve(cov, r), rel=1e-9)
    assert item.dof == jac.shape[0]


def test_flat_prior_marginal_chi2_counts_the_projected_dof(problem):
    _, _, _, waterfall, curves = problem
    basis = np.random.default_rng(1).normal(size=(N_FREQ, 2))
    item = collapse.per_lst_basis(waterfall, basis)
    q, _ = np.linalg.qr(basis)
    perp = np.eye(N_FREQ) - q @ q.T
    direct = np.sum(((waterfall - curves[2][None, :]) @ perp) ** 2) / SIGMA**2
    assert item.marginal_chi2(curves[2]) == pytest.approx(direct, rel=1e-9)
    assert item.dof == N_TIME * (N_FREQ - 2)


# ------------------------------------------------------- demo B's square root --
ILL_TIME, ILL_FREQ, ILL_PAR = 4, 3, 20
EPS = np.finfo(np.float64).eps


@pytest.fixture(scope="module")
def ill():
    """``J S^1/2`` with singular values 1e6 .. 1e-4 at unit width (demo B's range), 4 LSTs x 3 channels."""
    rng = np.random.default_rng(12)
    n = ILL_TIME * ILL_FREQ
    left = np.linalg.qr(rng.normal(size=(n, n)))[0]
    right = np.linalg.qr(rng.normal(size=(ILL_PAR, n)))[0]
    scale = 0.5 + rng.random(ILL_PAR)
    jac = ((left * np.logspace(6, -4, n)[None, :]) @ right.T) / scale[None, :]
    loc = rng.normal(size=ILL_PAR)
    curve = 0.1 * rng.normal(size=ILL_FREQ)
    tiling = np.kron(np.eye(ILL_FREQ), np.ones((ILL_TIME, 1)))
    data = jac @ (loc + 3.0 * scale * rng.normal(size=ILL_PAR)) + tiling @ curve + 0.01 * rng.normal(size=n)
    return {"jac": jac, "loc": loc, "scale": scale, "curve": curve, "tiling": tiling, "data": data,
            "waterfall": data.reshape(ILL_FREQ, ILL_TIME).T}  # fmt: skip


def _reference(ill, width, sigma) -> dict:
    """G, b, log evidence, posterior mean and sigma(amp) at 50 digits, from the float64 inputs."""
    jac, n = ill["jac"], ill["data"].size
    with mpmath.workdps(50):
        j = mpmath.matrix(jac.tolist())
        var = [mpmath.mpf(float(width * v)) ** 2 for v in ill["scale"]]
        js = mpmath.matrix(n, jac.shape[1])
        for i in range(n):
            for k in range(jac.shape[1]):
                js[i, k] = j[i, k] * var[k]
        cov = js * j.T + mpmath.mpf(sigma) ** 2 * mpmath.eye(n)
        inverse = mpmath.inverse(cov)
        tiling = mpmath.matrix(ill["tiling"].tolist())
        r = mpmath.matrix((ill["data"] - jac @ ill["loc"]).tolist())
        g, b = tiling.T * inverse * tiling, tiling.T * inverse * r
        log_z = -((r.T * inverse * r)[0] + mpmath.log(mpmath.det(cov)) + n * mpmath.log(2 * mpmath.pi)) / 2
        mean = mpmath.matrix(ill["loc"].tolist()) + js.T * inverse * r
        t = mpmath.matrix(ill["curve"].tolist())
        sigma_amp = 1 / mpmath.sqrt((t.T * g * t)[0])
        as_array = lambda m: np.array(m.tolist(), dtype=np.float64).reshape(m.rows, m.cols)  # noqa: E731
        return {"G": as_array(g), "b": as_array(b).ravel(), "log_z": float(log_z),
                "mean": as_array(mean).ravel(), "sigma_amp": float(sigma_amp)}  # fmt: skip


def _relative(got, want) -> float:
    return float(np.linalg.norm(np.asarray(got) - np.asarray(want)) / np.linalg.norm(want))


@pytest.mark.parametrize("width", [1e-3, 1.0, 1e3])
@pytest.mark.parametrize("sigma", [1e-4, 1e-2, 1.0])
def test_gaussian_marginal_matches_a_50_digit_reference(ill, width, sigma):
    scale = width * ill["scale"]
    ref = _reference(ill, width, sigma)
    item = collapse.gaussian_prior(ill["waterfall"], ill["jac"], ill["loc"], scale, sigma=sigma)
    moved = collapse.GaussianMarginal.build(ill["jac"], ill["loc"], scale).at(sigma)
    got = {"G": item.design.T @ item.design / collapse.SIGMA0**2,  # the design D and the statistic y
           "b": item.design.T @ item.data / collapse.SIGMA0**2,  # enter through these invariants
           "mean": moved.posterior_mean(ill["data"]),
           "sigma_amp": collapse.amplitude_forecast(item, ill["curve"], ill["waterfall"])["sigma"]}  # fmt: skip
    # Backward stable in J S^1/2: errors up to ~eps kappa, kappa = ||J S^1/2|| / sigma
    # (1e3 .. 1e13 here). Measured worst over the nine cells: 12 eps kappa, and 1.2e-6.
    # Forming J S J^T errs by ~eps kappa^2: 21 % in b at kappa 1e8, or Cholesky fails.
    kappa = np.linalg.norm(ill["jac"] * scale[None, :], 2) / sigma
    tolerance = min(1e-5, 100.0 * EPS * kappa)
    assert item.rank == ILL_FREQ
    for key, value in got.items():
        assert _relative(value, ref[key]) < tolerance, key
    log_z = moved.log_evidence(ill["data"])
    assert abs(log_z - ref["log_z"]) < tolerance * max(abs(ref["log_z"]), ill["data"].size)


def test_gaussian_prior_does_not_depend_on_the_column_order(ill):
    """Demo B's hook failed its Cholesky for 4 of 5 random column orders at 10 mK."""
    base = collapse.GaussianMarginal.build(ill["jac"], ill["loc"], ill["scale"])
    item = collapse.from_marginal(ill["waterfall"], base)
    rng = np.random.default_rng(3)
    for _ in range(5):
        order = rng.permutation(ILL_PAR)
        other = collapse.GaussianMarginal.build(ill["jac"][:, order], ill["loc"][order], ill["scale"][order])
        shuffled = collapse.from_marginal(ill["waterfall"], other)
        np.testing.assert_allclose(shuffled.design.T @ shuffled.design, item.design.T @ item.design, rtol=1e-7)
        np.testing.assert_allclose(shuffled.design.T @ shuffled.data, item.design.T @ item.data, rtol=1e-7)
        back = np.empty(ILL_PAR)
        back[order] = other.posterior_mean(ill["data"])
        np.testing.assert_allclose(back, base.posterior_mean(ill["data"]), rtol=1e-7, atol=1e-9)
        assert other.log_evidence(ill["data"]) == pytest.approx(base.log_evidence(ill["data"]), rel=1e-7)


def test_moving_the_noise_reuses_the_prior_root(ill):
    built = collapse.GaussianMarginal.build(ill["jac"], ill["loc"], ill["scale"], sigma=0.3)
    moved = collapse.GaussianMarginal.build(ill["jac"], ill["loc"], ill["scale"]).at(0.3)
    assert moved.sigma == built.sigma == 0.3
    np.testing.assert_array_equal(moved.root, built.root)
    np.testing.assert_allclose(moved.factor.T @ moved.factor, built.root.T @ built.root + 0.09 * np.eye(12),
                               rtol=1e-12, atol=1e-12 * np.abs(built.root).max() ** 2)  # fmt: skip


@pytest.mark.parametrize("sigma", [0.0, -0.01, float("nan")])
def test_gaussian_marginal_refuses_a_noise_that_is_not_positive(ill, sigma):
    with pytest.raises(ValueError, match="positive"):
        collapse.GaussianMarginal.build(ill["jac"], ill["loc"], ill["scale"], sigma=sigma)


def test_fewer_parameters_than_data_is_the_dense_marginal(problem):
    """``p < n``: the root is trapezoidal, padded; against the dense covariance at 10 mK and 1 K."""
    jac, loc, scale, waterfall, _ = problem
    data = collapse.vec(waterfall)
    for sigma in (SIGMA, 1.0):  # at 1e-4 K the dense solve is the less accurate side
        marginal = collapse.GaussianMarginal.build(jac, loc, scale, sigma)
        cov = sigma**2 * np.eye(jac.shape[0]) + (jac * scale**2) @ jac.T
        r = data - jac @ loc
        dense = -0.5 * (r @ np.linalg.solve(cov, r) + np.linalg.slogdet(cov)[1] + r.size * np.log(2 * np.pi))
        assert marginal.log_evidence(data) == pytest.approx(dense, rel=1e-10)
        np.testing.assert_allclose(marginal.solve(r), np.linalg.solve(cov, r), rtol=1e-8)
