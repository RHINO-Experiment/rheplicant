"""The sensitivity sweep's algebra against direct computations.

* Demo B's forecast at any noise (``collapse.GaussianMarginal`` moved to
  that noise) against a dense solve of ``sigma^2 I + J S J^T`` at 1e-4 K,
  1e-2 K and 1 K, and the rescaling identity
  ``C(sigma, S) = (sigma/sigma0)^2 C(sigma0, S sigma0^2/sigma^2)`` through
  ``collapse.gaussian_prior``.
* The same forecast on an ill-conditioned problem against a 50-digit reference.
* Demo A's order re-chosen by the documents' rule at each noise level.
* The radiometer formula against a hand computation.
* A flat prior's sigma(amp) is proportional to the noise, equal to the
  projector formula, and its crossings are the closed-form ones.
"""

import mpmath
import numpy as np
import pytest
from global21cm import collapse, scenario, sensitivity, strategies
from global21cm.foreground_beamconv import moment_basis, orthonormal_span

N_TIME, N_FREQ, N_PAR = 6, 5, 7
SIGMA0 = scenario.NOISE_SIGMA_K


def _tiling(n_time=N_TIME, n_freq=N_FREQ):
    return np.kron(np.eye(n_freq), np.ones((n_time, 1)))


def _waterfall(vector, n_time=N_TIME):
    """Frequency-major vector -> ``(n_time, n_freq)``, the inverse of ``collapse.vec``."""
    return np.asarray(vector).reshape(-1, n_time).T


@pytest.fixture
def problem():
    """A foreground off its prior centre plus an out-of-model misfit: the bias is not zero."""
    rng = np.random.default_rng(11)
    jac = rng.normal(size=(N_TIME * N_FREQ, N_PAR))
    loc, scale = rng.normal(size=N_PAR), 0.5 + rng.random(N_PAR)
    shape = 0.1 * rng.normal(size=N_FREQ)
    truth = loc + 2.0 * scale * rng.normal(size=N_PAR)
    noiseless = _waterfall(
        jac @ truth + _tiling() @ shape + 0.01 * rng.normal(size=N_TIME * N_FREQ)
    )
    waterfall = noiseless + SIGMA0 * rng.normal(size=noiseless.shape)
    return {
        "jac": jac,
        "loc": loc,
        "scale": scale,
        "shape": shape,
        "noiseless": noiseless,
        "waterfall": waterfall,
    }


def _dense(problem, sigma):
    """sigma(amp) and bias straight from ``C = sigma^2 I + J S J^T``."""
    jac, scale = problem["jac"], problem["scale"]
    cov = sigma**2 * np.eye(jac.shape[0]) + (jac * scale**2) @ jac.T
    tiled = _tiling() @ problem["shape"]
    whitened = np.linalg.solve(cov, tiled)
    info = tiled @ whitened
    residual = collapse.vec(problem["noiseless"]) - jac @ problem["loc"]
    return 1.0 / np.sqrt(info), whitened @ residual / info - 1.0


def _marginal(problem):
    return collapse.GaussianMarginal.build(problem["jac"], problem["loc"], problem["scale"])


@pytest.mark.parametrize("sigma", [1e-4, 1e-2, 1.0])
def test_gaussian_forecast_matches_the_dense_forecast(problem, sigma):
    forecast = sensitivity.gaussian_forecast(
        _marginal(problem), problem["waterfall"], problem["shape"], problem["noiseless"]
    )
    sigma_amp, bias = forecast(sigma)
    dense_sigma, dense_bias = _dense(problem, sigma)
    assert abs(bias) > 1e-3  # the check has something to compare
    assert sigma_amp == pytest.approx(dense_sigma, rel=1e-7)
    assert bias == pytest.approx(dense_bias, rel=1e-7, abs=1e-9)
    residual = collapse.vec(problem["noiseless"]) - problem["jac"] @ problem["loc"]
    direct = sensitivity.direct_forecast(
        _marginal(problem).at(sigma), _tiling() @ problem["shape"], residual
    )
    assert direct == pytest.approx((dense_sigma, dense_bias), rel=1e-7)


@pytest.mark.parametrize("sigma", [1e-4, 1e-2, 1.0])
def test_the_rescaling_identity_holds_through_the_hook(problem, sigma):
    """``gaussian_prior`` at ``sigma`` equals it at ``SIGMA0`` with the prior scaled by ``sigma0/sigma``."""
    ratio = sigma / SIGMA0
    args = (problem["waterfall"], problem["jac"], problem["loc"])
    at_sigma = collapse.gaussian_prior(*args, problem["scale"], sigma=sigma)
    rescaled = collapse.gaussian_prior(*args, problem["scale"] / ratio)
    one = collapse.amplitude_forecast(at_sigma, problem["shape"], problem["noiseless"])
    other = collapse.amplitude_forecast(rescaled, problem["shape"], problem["noiseless"])
    # Measured: agreement to 1e-12 or better. Leaving the prior unscaled moves sigma(amp)
    # by 7e-7 at 1e-4 K and 6e-3 at 1 K, and the bias by 0.2 at 1 K.
    assert one["sigma"] == pytest.approx(other["sigma"] * ratio, rel=1e-9)
    assert one["bias"] == pytest.approx(other["bias"], rel=1e-9, abs=1e-12)


def _ill_conditioned():
    """``J S^1/2`` with singular values 1e6 .. 1e-4 on 4 LSTs x 3 channels (demo B's range)."""
    n_time, n_freq, n_par = 4, 3, 20
    rng = np.random.default_rng(12)
    left = np.linalg.qr(rng.normal(size=(n_time * n_freq, n_time * n_freq)))[0]
    right = np.linalg.qr(rng.normal(size=(n_par, n_time * n_freq)))[0]
    js = (left * np.logspace(6, -4, n_time * n_freq)[None, :]) @ right.T
    return js, rng.normal(size=n_freq), rng.normal(size=n_time * n_freq), (n_time, n_freq)


def _reference_50_digits(js, tiled, residual, sigma):
    with mpmath.workdps(50):
        j = mpmath.matrix(js.tolist())
        cov = j * j.T + mpmath.mpf(sigma) ** 2 * mpmath.eye(j.rows)
        whitened = mpmath.lu_solve(cov, mpmath.matrix(tiled.tolist()))
        info = sum(whitened[i] * tiled[i] for i in range(j.rows))
        cross = sum(whitened[i] * residual[i] for i in range(j.rows))
        return float(1 / mpmath.sqrt(info)), float(cross / info - 1)


@pytest.mark.parametrize("sigma", [1e-4, SIGMA0, 1.0])
def test_demo_b_forecast_survives_the_conditioning_that_broke_the_gram_route(sigma):
    """The Gram route was 0.9 % off in sigma(amp) here at 10 mK and failed at 1e-4 K."""
    js, shape, residual, (n_time, n_freq) = _ill_conditioned()
    tiled = _tiling(n_time, n_freq) @ shape
    reference = _reference_50_digits(js, tiled, residual, sigma)
    marginal = collapse.GaussianMarginal.build(js, np.zeros(js.shape[1]), np.ones(js.shape[1]))
    noiseless = _waterfall(residual, n_time)
    forecast = sensitivity.gaussian_forecast(marginal, noiseless, shape, noiseless)
    # Measured against the reference: 7.9e-8 at 1e-4 K (||J S^1/2|| / sigma = 1e10),
    # 1.5e-10 at 10 mK, 5e-12 at 1 K; float64 allows ~eps 1e10 = 2e-6 at the first.
    assert forecast(sigma) == pytest.approx(reference, rel=1e-6)
    assert sensitivity.direct_forecast(marginal.at(sigma), tiled, residual) == pytest.approx(
        reference, rel=1e-6
    )


def test_checks_record_shuffled_columns_and_the_compression(problem):
    marginal = _marginal(problem)
    model = {k: problem[k] for k in ("jac", "loc", "scale")} | {"marginal": marginal}
    sim = {
        "waterfall": problem["waterfall"],
        "noiseless": problem["noiseless"],
        "t21_truth": problem["shape"],
    }
    forecast = sensitivity.gaussian_forecast(
        marginal, problem["waterfall"], problem["shape"], problem["noiseless"]
    )
    residual = collapse.vec(problem["noiseless"]) - problem["jac"] @ problem["loc"]
    out = sensitivity.checks(sim, model, forecast, _tiling() @ problem["shape"], residual)
    shuffled = out["shuffled_columns_at_sigma0"]
    assert shuffled["n"] == len(shuffled["results"]) == sensitivity.N_ORDERINGS
    assert shuffled["sigma_amp_max_rel_diff"] < 1e-10 and shuffled["bias_max_abs_diff"] < 1e-10
    rows = out["compressed_vs_direct"]
    assert [r["noise_k"] for r in rows] == list(sensitivity.NOISE_GRID_K)
    assert max(abs(r["sigma_amp_rel_diff"]) for r in rows) < 1e-9


# ------------------------------------------------------------- demo A's order --
FREQS = np.linspace(50.0, 110.0, 12)


def test_order_is_re_chosen_by_the_documents_rule_at_each_noise(monkeypatch):
    """A second moment at 2.5 chi^2 per LST at 10 mK: needed below ~10 mK, not above."""
    shape = -0.15 * np.exp(-0.5 * ((FREQS - 72.0) / 10.0) ** 2)
    bank = np.stack([np.zeros(FREQS.size), shape])  # the bank holds the true curve
    monkeypatch.setattr(strategies, "prior_bank", lambda f: (np.zeros((2, 7)), bank))
    rng = np.random.default_rng(0)
    q = orthonormal_span(moment_basis(FREQS, -2.55, 2, 70.0))
    amp = 1000.0 * (1.0 + 0.1 * rng.random(50))
    noiseless = (
        amp[:, None] * q[:, 0][None, :]
        + np.sqrt(2.5) * SIGMA0 * q[:, 1][None, :]
        + shape[None, :]
    )
    sim = {
        "noiseless": noiseless,
        "waterfall": noiseless + SIGMA0 * rng.normal(size=noiseless.shape),
        "t21_truth": shape,
    }
    arrays = {"waterfall": sim["waterfall"], "beam_spectra": np.zeros((12, 1)), "freqs_mhz": FREQS}
    literal = {"order": "bic_among_passing", "moments": [1, 3], "beam_terms": [0, 0],
               "beta0": -2.55, "nu_ref_mhz": 70.0}  # fmt: skip
    rows = sensitivity.order_sweep(arrays, literal, sim, 1.0, lambda s: (s, 0.0))
    chosen = {r["noise_k"]: r["K"] for r in rows}
    # Plain BIC takes K = 1 at 10 mK, where it fails the fit check; min chi^2 takes K = 3.
    assert (chosen[3e-4], chosen[SIGMA0], chosen[0.1]) == (2, 2, 1)
    assert all(r["passes"] for r in rows)


def test_radiometer_formula_by_hand():
    # sqrt(2e6 Hz * 898 s) = 42379.2402; 5000 K / 42379.2402 = 0.1179823 K;
    # (0.1179823 / 0.01)^2 = 139.198 days for a fixed T_sys of 5000 K (a hand
    # case; the pipeline takes T_sys from the noiseless sky, test below).
    one_day = sensitivity.radiometer_sigma(5000.0, 2e6, 898.0)
    assert one_day == pytest.approx(0.1179823, rel=1e-6)
    assert sensitivity.days_to_reach(one_day, 0.01) == pytest.approx(139.198, rel=1e-5)


def test_radiometer_summary_of_a_two_lst_waterfall():
    freqs = np.array([60.0, 70.0, 80.0])  # 10 MHz channels; 70 MHz is the reference
    sky = np.array([[2000.0, 1000.0, 500.0], [2000.0, 3000.0, 500.0]])
    out = sensitivity.radiometer(sky, freqs, sigma=0.01)
    per_day = np.sqrt(1e7 * scenario.TIME_STEP_S)
    assert out["reference_channel_mhz"] == 70.0
    assert out["sigma_1day_reference_k"] == pytest.approx(
        2000.0 / per_day, rel=1e-12
    )  # median of 1000, 3000
    assert out["days_reference"]["min_over_lst"] == pytest.approx(
        (1000.0 / per_day / 0.01) ** 2, rel=1e-12
    )
    assert out["sigma_band_factor"] == pytest.approx(4.0, rel=1e-12)  # 2000 K against 500 K
    assert out["sigma_lst_factor_reference"] == pytest.approx(3.0, rel=1e-12)


def test_radiometer_refuses_a_non_uniform_grid():
    with pytest.raises(ValueError, match="not uniform"):
        sensitivity.radiometer(np.ones((2, 3)), [60.0, 70.0, 85.0])


def _flat(problem):
    basis = np.random.default_rng(13).normal(size=(N_FREQ, 2))
    return collapse.per_lst_basis(problem["waterfall"], basis), basis


def test_flat_prior_sigma_amp_is_proportional_to_the_noise(problem):
    item, basis = _flat(problem)
    forecast = sensitivity.flat_forecast(item, problem["shape"], problem["noiseless"])
    q, _ = np.linalg.qr(basis)
    shape = problem["shape"]
    projected = shape @ (np.eye(N_FREQ) - q @ q.T) @ shape
    rows = [forecast(s) for s in (1e-4, 3e-3, 1.0)]
    for sigma, (sigma_amp, bias) in zip((1e-4, 3e-3, 1.0), rows, strict=True):
        assert sigma_amp == pytest.approx(sigma / np.sqrt(N_TIME * projected), rel=1e-10)
        assert bias == rows[0][1]
    ratios = [
        sigma_amp / sigma for sigma, (sigma_amp, _) in zip((1e-4, 3e-3, 1.0), rows, strict=True)
    ]
    assert np.ptp(ratios) <= 1e-14 * ratios[0]


def test_flat_prior_crossings_are_the_closed_form_ones(problem):
    item, _ = _flat(problem)
    forecast = sensitivity.flat_forecast(item, problem["shape"], problem["noiseless"])
    sigma_amp, bias = forecast(SIGMA0)
    detect = sensitivity.crossings(forecast, lambda sa, b: sensitivity.DETECTION * sa)
    misfit = sensitivity.crossings(forecast, lambda sa, b: abs(b) / sa)
    assert detect == pytest.approx([SIGMA0 / (sensitivity.DETECTION * sigma_amp)], rel=1e-9)
    assert misfit == pytest.approx([SIGMA0 * abs(bias) / sigma_amp], rel=1e-9)


def test_per_lst_basis_is_read_back_from_the_collapse(problem):
    item, basis = _flat(problem)
    found, same = sensitivity.per_lst_basis(item, N_TIME)
    q, _ = np.linalg.qr(basis)
    assert same and found.shape == (N_FREQ, 2)
    np.testing.assert_allclose(found @ found.T, q @ q.T, atol=1e-12)


def test_flat_prior_effective_fraction_is_the_perpendicular_fraction(problem):
    item, basis = _flat(problem)
    fg = problem["noiseless"] - problem["shape"][None, :]
    oracle = sensitivity.flat_forecast(
        collapse.oracle(problem["waterfall"], fg), problem["shape"], problem["noiseless"]
    )
    forecast = sensitivity.flat_forecast(item, problem["shape"], problem["noiseless"])
    rows = sensitivity.summarise(forecast, 0.05, oracle)["sweep"]
    expected = sensitivity.fom.perpendicular_fraction(basis, problem["shape"])
    assert [r["noise_k"] for r in rows] == list(sensitivity.NOISE_GRID_K)
    for r in rows:
        assert r["effective_perp_fraction"] == pytest.approx(expected, rel=1e-10)
        assert r["days_70mhz"] == pytest.approx((0.05 / r["noise_k"]) ** 2, rel=1e-12)
