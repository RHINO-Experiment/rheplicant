"""The decisions ``analyse.py`` makes around the scores, on synthetic inputs.

No SMC, no run tree and no simulation output is read: each function is fed
the few numbers or files it decides on.
"""

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from global21cm import analyse, scenario, scoring, strategies


@pytest.mark.parametrize(
    "signal, fit, expected",
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_calibrated_needs_the_signal_and_the_fit(signal, fit, expected):
    assert analyse.calibrated_verdict(signal, {"passes": fit}) is expected


def _entry(calibrated, laplace_trace, trace, core_trace):
    return {"calibrated": calibrated, "laplace_trace": laplace_trace,
            "tail": {"trace": trace, "core_trace": core_trace}}  # fmt: skip


ORACLE_CURVES = np.array([[0.0, 0.0], [2.0, 0.0]])  # tr Cov = 1
MODEL_CURVES = np.array([[0.0, 0.0], [4.0, 0.0]])  # tr Cov = 4


def test_eta_when_both_posteriors_are_calibrated():
    oracle = _entry(True, 1.0, 1.0, 0.25)
    model = _entry(True, 4.0, 16.0, 4.0)
    out = analyse.efficiencies(model, oracle, MODEL_CURVES, ORACLE_CURVES)
    assert out["eta_smc"] == pytest.approx(0.5, rel=1e-12)
    # The oracle's whole spread over the model's core, not the oracle's core.
    assert out["eta_smc_core"] == pytest.approx(0.5, rel=1e-12)
    assert out["eta_laplace"] == pytest.approx(0.5, rel=1e-12)


@pytest.mark.parametrize("model_ok, oracle_ok", [(False, True), (True, False), (False, False)])
def test_eta_needs_both_posteriors_calibrated(model_ok, oracle_ok):
    out = analyse.efficiencies(_entry(model_ok, 4.0, 16.0, 4.0), _entry(oracle_ok, 1.0, 1.0, 0.25),
                               MODEL_CURVES, ORACLE_CURVES)  # fmt: skip
    assert out["eta_smc"] is None and out["eta_smc_core"] is None
    assert out["eta_laplace"] == pytest.approx(0.5, rel=1e-12)  # sampler-free, always reported


def test_score_scenario_gives_each_model_its_own_efficiency(monkeypatch, tmp_path):
    # Everything but the eta block is stubbed: which curves and which verdicts reach it.
    np.save(tmp_path / "noiseless.npy", np.zeros((2, 3)))
    (tmp_path / "prepare.json").write_text("{}")
    case = SimpleNamespace(freqs_mhz=lambda: np.arange(3.0), freq_start_mhz=0.0, freq_stop_mhz=2.0,
                           sim_dir=tmp_path)  # fmt: skip
    monkeypatch.setattr(strategies, "prior_bank", lambda freqs: (np.zeros((4, 7)), np.zeros((4, 3))))
    monkeypatch.setattr(analyse, "_truth", lambda case, fine: {})
    monkeypatch.setattr(analyse, "_draws", lambda u, freqs, fine: {})
    monkeypatch.setattr(scoring, "score", lambda *args: {})
    curves = {"oracle": ORACLE_CURVES, "beamconv": MODEL_CURVES, "physical": np.array([[0.0, 0.0], [8.0, 0.0]])}
    ok = {"oracle": True, "beamconv": False, "physical": True}
    monkeypatch.setattr(analyse, "score_model",
                        lambda case, model, suffix, context: (_entry(ok[model], 1.0, 1.0, 1.0),
                                                              {"curves": curves[model]}))  # fmt: skip
    report, _ = analyse.score_scenario(case, quick=False)
    assert report["beamconv"]["eta_smc"] is None and report["beamconv"]["eta_smc_core"] is None
    assert report["physical"]["eta_smc"] == pytest.approx(0.25, rel=1e-12)
    assert "eta_smc" not in report["oracle"]


class TestTailDecomposition:
    def test_the_core_keeps_the_nearest_ninety_percent(self):
        # 20 curves: 16 at 0, two at +/-a, two far at +/-b. The core of 18 keeps
        # the zeros and the two near ones: tr Cov = F 2 a^2 / 18.
        a, b, n_freq = 1.0, 10.0, 3
        level = np.r_[np.zeros(16), a, -a, b, -b]
        curves = level[:, None] * np.ones(n_freq)
        out = analyse.tail_decomposition({"curves": curves, "fine": curves}, np.arange(float(n_freq)))
        assert out["core_trace"] == pytest.approx(n_freq * 2 * a**2 / 18, rel=1e-12)
        assert out["trace"] == pytest.approx(n_freq * 2 * (a**2 + b**2) / 20, rel=1e-12)

    def test_weak_signal_fraction_and_depth_quantiles(self):
        depths = np.array([-0.01, -0.049, -0.051, -0.1, -0.2])
        fine = np.zeros((5, 4))
        fine[:, 2] = depths
        out = analyse.tail_decomposition({"curves": fine, "fine": fine}, np.arange(4.0))
        assert out["weak_fraction"] == 0.4  # troughs shallower than 50 mK
        assert list(out["depth_quantiles_mk"]) == [1, 5, 50, 95, 99]
        assert out["depth_quantiles_mk"][50] == pytest.approx(-51.0, rel=1e-12)


class TestCheckInputs:
    @pytest.fixture
    def layout(self, tmp_path, monkeypatch):
        here = tmp_path.resolve()
        monkeypatch.setattr(scenario, "HERE", here)
        tree, sim = here / "results" / "oracle", here / "results" / "sim"
        tree.mkdir(parents=True)
        sim.mkdir(parents=True)
        read, written = sim / "waterfall.npy", sim / "oracle_data.npy"
        read.write_bytes(b"read by the run")
        written.write_bytes(b"written by prepare")
        digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()  # noqa: E731
        inputs = [{"path": str(read), "sha256": digest(read)}]
        (tree / "provenance.json").write_text(json.dumps({"inputs": inputs}))
        (sim / "prepare.json").write_text(json.dumps({"written": {"oracle_data.npy": digest(written)}}))
        return tree, SimpleNamespace(sim_dir=sim), {"read": read, "written": written}, digest

    def test_unchanged_inputs_are_recorded_relative_to_the_example(self, layout):
        tree, case, files, digest = layout
        assert analyse.check_inputs([tree], [case]) == {
            "results/sim/waterfall.npy": digest(files["read"]),
            "results/sim/oracle_data.npy": digest(files["written"]),
        }

    @pytest.mark.parametrize("changed", ["read", "written"])
    def test_a_changed_input_is_refused(self, layout, changed):
        tree, case, files, _ = layout
        files[changed].write_bytes(b"changed after the run")
        with pytest.raises(RuntimeError, match=files[changed].name):
            analyse.check_inputs([tree], [case])
