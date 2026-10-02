"""The documents' strategy hooks: order choice, refusals, the chains' start."""

import mpmath
import numpy as np
import pytest
from scipy import stats

from global21cm import collapse, physical_inputs, scenario, scoring, strategies
from global21cm.foreground_beamconv import moment_basis, orthonormal_span

FREQS = np.linspace(50.0, 110.0, 12)
N_TIME = 8


@pytest.fixture
def waterfall():
    rng = np.random.default_rng(3)
    power = (FREQS / 70.0) ** -2.55
    return 1000.0 * (1.0 + 0.1 * rng.random(N_TIME))[:, None] * power[None, :] + 0.01 * rng.normal(size=(N_TIME, FREQS.size))


@pytest.fixture
def spectra():
    return np.linalg.qr(np.random.default_rng(4).normal(size=(FREQS.size, 3)))[0]


@pytest.fixture
def small_bank(monkeypatch):
    rng = np.random.default_rng(5)
    u = rng.normal(size=(64, 7))
    curves = 0.05 * rng.normal(size=(64, FREQS.size))
    monkeypatch.setattr(strategies, "prior_bank", lambda freqs: (u, curves))
    return u, curves


def test_explicit_order_is_honoured(waterfall, spectra, small_bank):
    item = strategies.per_lst_moments(waterfall, spectra, FREQS, [2, 1], [3, 8], [0, 4], -2.55, 70.0)
    assert item.info["model_order"]["chosen"] == {"K": 2, "beam_terms": 1}
    assert item.dof == N_TIME * (FREQS.size - 2)


@pytest.mark.parametrize("rule", strategies.ORDER_RULES)
def test_the_hook_records_the_rule_and_its_choice(waterfall, spectra, small_bank, rule):
    item = strategies.per_lst_moments(waterfall, spectra, FREQS, rule, [1, 3], [0, 2], -2.55, 70.0)
    record = item.info["model_order"]
    assert (record["rule"], record["fit_check_p"]) == (rule, strategies.FIT_CHECK_P)
    assert record["chosen"] == strategies.choose_order(record["table"], rule)
    assert {(r["K"], r["beam_terms"]) for r in record["table"]} >= {(1, 0), (3, 2)}
    assert item.dof == N_TIME * (FREQS.size - record["chosen"]["n_basis"])


def test_order_rows_carry_the_declared_penalty_and_p_value(monkeypatch):
    rng = np.random.default_rng(3)
    waterfall = 1000.0 * (FREQS / 70.0) ** -2.55 * (1 + 0.1 * rng.random(N_TIME))[:, None]
    monkeypatch.setattr(strategies, "prior_bank", lambda f: (rng.normal(size=(16, 7)), 0.05 * rng.normal(size=(16, 12))))
    spectra = np.linalg.qr(rng.normal(size=(12, 3)))[0]
    record = strategies.model_order(waterfall, spectra, FREQS, "bic_among_passing", [1, 3], [0, 2], -2.55, 70.0)
    for row in record["table"]:
        n_par = N_TIME * row["n_basis"] + 7  # every LST's coefficients, and the 21 cm parameters
        assert row["dof"] == waterfall.size - n_par
        assert row["bic"] == pytest.approx(row["chi2_min"] + n_par * np.log(waterfall.size), rel=1e-12)
        assert row["p"] == pytest.approx(stats.chi2.sf(row["chi2_min"], row["dof"]), rel=1e-12)
        assert row["passes"] == (row["p"] >= strategies.FIT_CHECK_P)


def _moment_sky(misfits=(2.5,), n_time=50, seed=0):
    """Every LST: one moment times a varying amplitude, plus moment ``k + 2`` at ``misfits[k]`` chi^2 per LST.

    With the default, K = 1 fails the fit check (p ~ 1e-5 .. 1e-3) yet has the lowest
    BIC, since the second moment's 50 coefficients cost 50 ln(600) = 320 and buy ~175
    of chi^2.
    """
    rng = np.random.default_rng(seed)
    q = orthonormal_span(moment_basis(FREQS, -2.55, 1 + len(misfits), 70.0))
    amp = 1000.0 * (1.0 + 0.1 * rng.random(n_time))
    sky = amp[:, None] * q[:, 0][None, :]
    for k, misfit in enumerate(misfits):
        sky = sky + np.sqrt(misfit) * collapse.SIGMA0 * q[:, k + 1][None, :]
    return sky + collapse.SIGMA0 * rng.normal(size=sky.shape)


@pytest.fixture
def zero_bank(monkeypatch):
    monkeypatch.setattr(strategies, "prior_bank", lambda f: (np.zeros((4, 7)), np.zeros((4, FREQS.size))))


def _order(waterfall, rule, moments=(1, 4), sigma=collapse.SIGMA0):
    return strategies.model_order(waterfall, np.zeros((FREQS.size, 1)), FREQS, rule, list(moments), [0, 0],
                                  -2.55, 70.0, sigma=sigma)  # fmt: skip


def test_the_fit_check_excludes_a_failing_low_order_candidate(zero_bank):
    waterfall = _moment_sky()
    gated, plain = _order(waterfall, "bic_among_passing"), _order(waterfall, "bic")
    rows = {r["K"]: r for r in gated["table"]}
    assert not rows[1]["passes"] and rows[1]["bic"] == min(r["bic"] for r in rows.values())
    assert plain["chosen"]["K"] == 1  # BIC alone takes the order the data reject
    assert gated["chosen"]["K"] == 2 and gated["chosen"]["passes"]  # the lowest BIC that fits
    # Lower chi^2 keeps falling with K (it is what min chi^2 would pick); BIC does not.
    assert rows[4]["chi2_min"] < rows[2]["chi2_min"] and rows[4]["passes"]


def test_when_nothing_passes_the_rule_takes_the_lowest_bic(zero_bank):
    """Moments 2 and 3 at 20 and 2.5 chi^2 per LST; the grid stops at K = 2, so nothing fits."""
    record = _order(_moment_sky((20.0, 2.5)), "bic_among_passing", moments=(1, 2))
    rows = {r["K"]: r for r in record["table"]}
    assert record["n_passing"] == 0 and rows[1]["p"] < rows[2]["p"] < 1e-4  # measured 1.7e-6
    assert rows[2]["bic"] < rows[1]["bic"]
    assert record["chosen"] == rows[2] and record["chosen"]["passes"] is False


def test_bic_keeps_the_order_the_data_need(monkeypatch):
    """One moment exactly: BIC keeps K = 1 where min chi^2 would take the widest basis."""
    rng = np.random.default_rng(3)
    sky = 1000.0 * (FREQS / 70.0) ** -2.55 * (1 + 0.1 * rng.random(8))[:, None]
    waterfall = sky + collapse.SIGMA0 * rng.normal(size=sky.shape)
    monkeypatch.setattr(strategies, "prior_bank", lambda f: (np.zeros((4, 7)), np.zeros((4, 12))))
    for rule in strategies.ORDER_RULES:
        item = strategies.per_lst_moments(waterfall, np.zeros((12, 1)), FREQS, rule, [1, 3], [0, 0], -2.55, 70.0)
        assert item.info["model_order"]["chosen"]["K"] == 1


def test_choose_order_on_hand_made_rows():
    # The lowest BIC (K = 4) fails; the highest p (K = 3) is not the lowest passing BIC (K = 2).
    rows = [{"K": 1, "bic": 10.0, "p": 1e-6, "passes": False}, {"K": 2, "bic": 12.0, "p": 0.3, "passes": True},
            {"K": 3, "bic": 15.0, "p": 0.5, "passes": True}, {"K": 4, "bic": 8.0, "p": 0.005, "passes": False}]  # fmt: skip
    assert strategies.choose_order(rows, "bic")["K"] == 4
    assert strategies.choose_order(rows, "bic_among_passing")["K"] == 2
    failing = [dict(r, passes=False) for r in rows]
    assert strategies.choose_order(failing, "bic_among_passing")["K"] == 4
    with pytest.raises(ValueError, match="order rule"):
        strategies.choose_order(rows, "aic")


def test_order_table_takes_chi2_at_the_noise_it_is_given(zero_bank):
    waterfall = _moment_sky()
    at_sigma0, doubled = _order(waterfall, "bic"), _order(waterfall, "bic", sigma=2 * collapse.SIGMA0)
    for a, b in zip(at_sigma0["table"], doubled["table"], strict=True):
        assert b["chi2_min"] == pytest.approx(a["chi2_min"] / 4, rel=1e-12)
        assert b["p"] == pytest.approx(stats.chi2.sf(b["chi2_min"], b["dof"]), rel=1e-12)


def test_the_basis_needs_a_moment(zero_bank):
    with pytest.raises(ValueError, match="at least one moment"):
        _order(_moment_sky(), "bic", moments=(0, 2))


def test_design_checked_refuses_a_foreign_observation(waterfall):
    item = collapse.oracle(waterfall, np.zeros_like(waterfall))
    np.testing.assert_array_equal(strategies.design_checked(item, item.data[None, :]), item.design)
    with pytest.raises(ValueError, match="prepare"):
        strategies.design_checked(item, item.data[None, :] * 1.001)


def test_gaussian_sky_refuses_an_unknown_template(waterfall):
    with pytest.raises(ValueError, match="gsm2008"):
        strategies.gaussian_sky(waterfall, None, FREQS, 16, 47, "haslam", 1.0, 0.6, 0.05, 10.0, 0.0, 3)


def test_start_point_weighs_the_prior(small_bank):
    u, curves = small_bank
    # Two draws fit equally well; the one nearer the prior's centre must win.
    fit = np.zeros(FREQS.size)
    curves[:] = 1.0
    curves[10] = curves[20] = fit
    u[10], u[20] = np.full(7, 2.0), np.full(7, 0.5)
    item = collapse.oracle(np.zeros((N_TIME, FREQS.size)), np.zeros((N_TIME, FREQS.size)))
    np.testing.assert_array_equal(strategies.start_point(item, FREQS), u[20])
    assert strategies.component(u[20], 3) == u[20, 3]


def test_realisations_are_centred_on_the_noiseless_statistic(waterfall):
    noiseless = waterfall.copy()
    observed = noiseless + 5 * scenario.NOISE_SIGMA_K  # a data set far from the noiseless one
    item = collapse.oracle(observed, np.zeros_like(observed))
    rng = np.random.default_rng(6)
    mean = np.mean([scoring.realisation(item, noiseless, rng) for _ in range(400)], axis=0)
    target = item.statistic(noiseless)
    tolerance = 4 * collapse.SIGMA0 / np.sqrt(400)
    assert np.abs(mean - target).max() < tolerance
    assert np.abs(mean - item.data).max() > 10 * tolerance


def test_predictive_replicates_leave_the_padded_rows_empty():
    from scipy import stats

    rng = np.random.default_rng(7)
    item = collapse.per_lst_basis(rng.normal(size=(N_TIME, 6)), rng.normal(size=(6, 3)))
    assert item.rank == 3  # three padded rows
    data = np.zeros(6)
    data[0] = np.sqrt(6.0) * collapse.SIGMA0  # observed chi^2 = 6 for a zero curve
    item = collapse.Collapsed(**{**item.__dict__, "data": data})
    gof = scoring.goodness_of_fit(item, {"curves": np.zeros((2000, 6))}, np.zeros(2000), seed=1)
    # Replicates are chi^2 with rank = 3 dof: P(>= 6) = 0.112; with 6 dof it would be 0.423.
    assert gof["p_predictive"] == pytest.approx(stats.chi2.sf(6.0, 3), abs=0.03)
    # rank 3 < 7 parameters: the compressed test is absent, not run on 1 dof.
    assert gof["dof_compressed"] == -4 and gof["p_compressed"] is None


def test_global_amplitude_column_scales_the_template():
    from global21cm import physical_inputs

    rng = np.random.default_rng(8)
    n_pix, freqs = 12, np.linspace(50.0, 110.0, 4)
    response = rng.random((freqs.size, N_TIME, n_pix))
    template = (np.full(n_pix, -2.6), 100.0 + rng.random(n_pix), -0.1)
    point = (template[0], template[2], 2.0 * template[1])  # a_t differs from a_g
    wide = physical_inputs.Widths(1.0, 0.6, 0.05, 10.0, 1.0)
    jac = physical_inputs.jacobian(response, point, template, freqs, wide)
    assert jac.shape[1] == 2 * n_pix + 3
    np.testing.assert_allclose(jac[:, -2], jac[:, :n_pix] @ template[1], rtol=1e-12)
    np.testing.assert_allclose(jac[:, -1], jac[:, n_pix : 2 * n_pix] @ point[2], rtol=1e-12)
    post = np.zeros(2 * n_pix + 3)
    post[-2] = -0.4  # the template 40 % too bright
    (beta, c, amplitude), _ = physical_inputs._step(post, point, template, wide)
    # A-map and global scale add: 0 (the A-map posterior here) + (-0.4) a_g.
    np.testing.assert_allclose(amplitude, -0.4 * template[1], rtol=1e-12)


def _small_physical(seed=9):
    rng = np.random.default_rng(seed)
    n_pix, freqs = 12, np.linspace(50.0, 110.0, 4)
    response = rng.random((freqs.size, N_TIME, n_pix))
    template = (np.full(n_pix, -2.6) + 0.1 * rng.normal(size=n_pix), 100.0 + rng.random(n_pix), -0.1)
    widths = physical_inputs.Widths(1.0, 0.6, 0.05, 10.0, 0.0)
    truth = (template[0] + 0.05, 1.3 * template[1])
    sky = np.einsum("ftp,fp->tf", response, physical_inputs.spectra_at(truth[0], -0.1, freqs) * truth[1][None, :])
    return response, sky + collapse.SIGMA0 * rng.normal(size=sky.shape), freqs, template, widths


def test_relinearisation_steps_to_the_dense_posterior_mean():
    response, waterfall, freqs, template, widths = _small_physical()
    point = (template[0].copy(), template[2], template[1].copy())
    jac = physical_inputs.jacobian(response, point, template, freqs, widths)
    loc, scale = physical_inputs._prior(template, point, widths)
    cov = collapse.SIGMA0**2 * np.eye(jac.shape[0]) + (jac * scale**2) @ jac.T
    data = collapse.vec(waterfall)
    dense = loc + scale**2 * (jac.T @ np.linalg.solve(cov, data - jac @ loc))
    (beta, c, _), _ = physical_inputs._step(dense, point, template, widths)
    model = physical_inputs.linear_model(response, waterfall, freqs, template, widths, 1)
    # Measured 4e-8 apart: the dense solve of a covariance with condition ~2e13 is the
    # less accurate side. The step itself moves beta by up to 0.06 and c by 1.6e-3.
    np.testing.assert_allclose(model["beta_t"], beta, rtol=1e-6)
    assert model["c_t"] == pytest.approx(c, rel=1e-6)
    assert np.abs(beta - template[0]).max() > 1e-2
    # The evidence at the final point, with no 21 cm signal, against 50 digits: a float64
    # dense solve of this covariance differs from both by 5e-5.
    assert physical_inputs.log_evidence(model, waterfall) == pytest.approx(
        _log_evidence_50_digits(model["jac"], model["loc"], model["scale"], data), rel=1e-9)


def _log_evidence_50_digits(jac, loc, scale, data) -> float:
    with mpmath.workdps(50):
        j = mpmath.matrix(jac.tolist())
        var = mpmath.diag([mpmath.mpf(float(v)) ** 2 for v in scale])
        cov = j * var * j.T + mpmath.mpf(collapse.SIGMA0) ** 2 * mpmath.eye(j.rows)
        r = mpmath.matrix((data - jac @ loc).tolist())
        quad = (r.T * mpmath.lu_solve(cov, r))[0]
        return float(-(quad + mpmath.log(mpmath.det(cov)) + j.rows * mpmath.log(2 * mpmath.pi)) / 2)
