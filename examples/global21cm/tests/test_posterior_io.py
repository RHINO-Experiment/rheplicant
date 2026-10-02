"""Reading run trees and turning latents into curves, on tiny synthetic trees."""

import json

import numpy as np
import pytest
from scipy import stats

from global21cm import posterior_io as io
from global21cm import signal21


def test_names_are_encoded_as_n_dash_utf8_hex():
    assert io.encode("posterior") == "n-706f73746572696f72"  # a run directory of the shipped trees
    assert io.encode("u_vc") == "n-755f7663"
    assert io.encode("é") == "n-c3a9"


def test_run_dir_refuses_a_run_that_was_not_published(tmp_path):
    with pytest.raises(FileNotFoundError, match="rheplicant run"):
        io.run_dir(tmp_path, "posterior")
    (tmp_path / "runs" / "n-706f73746572696f72").mkdir(parents=True)
    assert io.run_dir(tmp_path, "posterior") == tmp_path / "runs" / "n-706f73746572696f72"


def _tree(tmp_path, run="posterior", arrays=None, diagnostics=None, seconds=None):
    path = tmp_path / "runs" / io.encode(run)
    path.mkdir(parents=True)
    if arrays is not None:
        np.savez(path / "draws.npz", **arrays)
    if diagnostics is not None:
        (path / "run_diagnostics.json").write_text(json.dumps(diagnostics))
    if seconds is not None:
        (path / "timings.json").write_text(json.dumps({"seconds": seconds}))
    return tmp_path


def test_load_mapping_decodes_the_latent_names(tmp_path):
    tree = _tree(tmp_path, arrays={"mapping/" + io.encode("u_vc"): np.arange(3.0),
                                   "mapping/" + io.encode("é"): np.ones(2), "other/raw": np.zeros(1)})  # fmt: skip
    out = io.load_mapping(tree, "posterior", "draws")
    assert set(out) == {"u_vc", "é", "other/raw"}  # an unencoded key is kept whole
    np.testing.assert_array_equal(out["u_vc"], np.arange(3.0))


def test_unit_draws_follow_the_signal_parameter_order():
    # Stored in reverse, each latent's draws equal to its index in THETA_NAMES.
    mapping = {name: np.full(3, float(i)) for i, name in reversed(list(enumerate(signal21.THETA_NAMES)))}
    u = io.unit_draws(mapping)
    assert u.shape == (3, 7)
    np.testing.assert_array_equal(u[0], np.arange(7.0))


def test_theta_is_the_probit_map_onto_the_box():
    low, high = (np.asarray(v) for v in signal21.prior_box())
    u = np.random.default_rng(2).normal(size=(5, 7))
    theta = io.theta_from_unit(u)
    np.testing.assert_allclose(theta, low + (high - low) * stats.norm.cdf(u), rtol=1e-12)
    np.testing.assert_allclose(io.theta_from_unit(np.zeros((1, 7)))[0], (low + high) / 2, rtol=1e-12)
    np.testing.assert_allclose(signal21.unit_normal_from_box(theta, low, high), u, rtol=1e-9, atol=1e-9)


def test_curves_are_evaluated_in_batches_without_reordering(monkeypatch):
    # A cheap stand-in for the emulator: row i of the result must be draw i's curve.
    calls = []

    def fake(theta, freqs):
        calls.append(len(theta))
        return theta[:, :1] + np.asarray(freqs)[None, :]

    monkeypatch.setattr(signal21, "curve_kelvin", fake)
    theta = np.arange(12001.0)[:, None] * np.ones(7)
    freqs = np.array([50.0, 60.0])
    out = io.curves(theta, freqs)
    np.testing.assert_array_equal(out, theta[:, :1] + freqs[None, :])
    assert calls == [5000, 5000, 2001]


def test_one_theta_gives_one_curve():
    theta = np.asarray(signal21.theta_log_true())
    freqs = np.linspace(50.0, 100.0, 6)
    out = io.curves(theta, freqs)
    assert out.shape == (1, 6)
    np.testing.assert_allclose(out[0], np.asarray(signal21.curve_kelvin(theta, freqs)), rtol=1e-12)


def test_nuts_summary_takes_the_worst_latent(tmp_path):
    diagnostics = {"divergences": 3, "n_chain": 4, "n_draw": 400,
                   "per_latent": {"u_vc": {"r_hat": 1.01, "n_eff": 300.0},
                                  "u_fx": {"r_hat": 1.30, "n_eff": 900.0},
                                  "u_tau": {"r_hat": 1.00, "n_eff": 12.5}}}  # fmt: skip
    tree = _tree(tmp_path, run="stress", diagnostics=diagnostics, seconds=12.25)
    assert io.nuts_summary(tree, "stress") == {"chains": 4, "draws": 400, "r_hat_max": 1.30, "n_eff_min": 12.5,
                                                "divergences": 3, "seconds": 12.25}  # fmt: skip
