"""The ablation's rows, on the documents and on a tiny synthetic linear model.

``ablation.main`` is not run here: it needs the simulation and a minute of
drift-scan projections. What is tested is what it is built from.
"""

import numpy as np
import pytest
import yaml

from global21cm import ablation, collapse, physical_inputs, scenario, strategies

#: ``physical_inputs.Widths`` field for each key of the document's literal.
FIELD_OF = {"amplitude_frac": "amplitude", "index_sigma": "index", "curvature_sigma": "curvature",
            "global_amplitude_sigma": "global_amplitude", "global_index_sigma": "global_index"}  # fmt: skip


def _literal(document) -> dict:
    return yaml.safe_load(document.read_text())["resources"]["arrays"]["foreground"]["literal"]


def test_the_chosen_row_is_the_documents_prior():
    literal = _literal(scenario.HERE / "physical.yaml")
    kind, widths, relinearise = ablation.rows()["chosen"]
    assert kind == "gsm" and literal["template"] == "gsm2008"
    assert widths._asdict() == {FIELD_OF[key]: float(literal[key]) for key in FIELD_OF}
    assert relinearise == literal["relinearise"]


def test_a_changed_document_moves_the_chosen_row(tmp_path):
    document = yaml.safe_load((scenario.HERE / "physical.yaml").read_text())
    literal = document["resources"]["arrays"]["foreground"]["literal"]
    literal.update(amplitude_frac=0.3, index_sigma=0.7, curvature_sigma=0.2, global_amplitude_sigma=0.0,
                   global_index_sigma=1.0, relinearise=1)  # fmt: skip
    path = tmp_path / "physical.yaml"
    path.write_text(yaml.safe_dump(document))
    assert ablation.rows(path)["chosen"] == ("gsm", physical_inputs.Widths(0.3, 0.7, 0.2, 0.0, 1.0), 1)
    assert {k: v for k, v in ablation.rows(path).items() if k != "chosen"} == ablation.FIXED_ROWS


def test_the_chosen_row_is_what_the_strategy_builds(monkeypatch):
    # gaussian_sky reads the same literal: whatever Widths it hands the linear
    # model must be the ablation's, field by field.
    class Built(Exception):
        pass

    def capture(response, waterfall, freqs, template, widths, relinearise):
        raise Built(widths, relinearise)

    monkeypatch.setattr(physical_inputs, "gsm_template", lambda nside: None)
    monkeypatch.setattr(physical_inputs, "response", lambda beam_alm, nside, lmax: None)
    monkeypatch.setattr(physical_inputs, "linear_model", capture)
    literal = _literal(scenario.HERE / "physical.yaml")
    with pytest.raises(Built) as built:
        strategies.gaussian_sky(np.zeros((2, 3)), np.zeros(1), np.arange(3.0), **literal)
    _, widths, relinearise = ablation.rows()["chosen"]
    assert built.value.args == (widths, relinearise)


def test_a_template_other_than_gsm_is_refused(tmp_path):
    document = yaml.safe_load((scenario.HERE / "physical.yaml").read_text())
    document["resources"]["arrays"]["foreground"]["literal"]["template"] = "haslam"
    path = tmp_path / "physical.yaml"
    path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="haslam"):
        ablation.shipped_prior(path)


def test_template_offsets_on_synthetic_maps():
    # Pixel k: GSM's amplitude is (1 + k/10) times the truth's and its index
    # offset is -0.1 k, for k = 0 ... 10 (medians 1.5 and -0.5).
    k = np.arange(11.0)
    truth = (np.full(11, -2.5), np.full(11, 100.0), -0.1)
    gsm = (truth[0] - 0.1 * k, truth[1] * (1 + k / 10), -0.12)
    out = ablation.template_offsets(gsm, truth)
    assert out["gsm_amplitude_over_truth"]["median"] == pytest.approx(1.5, rel=1e-12)
    assert out["gsm_amplitude_over_truth"]["p16"] == pytest.approx(1.16, rel=1e-12)
    assert out["gsm_amplitude_over_truth"]["p84"] == pytest.approx(1.84, rel=1e-12)
    index = out["gsm_index_minus_truth"]
    assert index["median"] == pytest.approx(-0.5, rel=1e-12)
    assert index["median_abs"] == pytest.approx(0.5, rel=1e-12)
    assert (index["p16"], index["p84"]) == (pytest.approx(-0.84, rel=1e-12), pytest.approx(-0.16, rel=1e-12))


def test_template_offsets_are_signed_gsm_minus_truth():
    truth = (np.array([-2.5, -2.5, -2.5]), np.ones(3), -0.1)
    gsm = (np.array([-2.3, -2.4, -2.8]), np.ones(3), -0.1)  # dbeta = +0.2, +0.1, -0.3
    index = ablation.template_offsets(gsm, truth)["gsm_index_minus_truth"]
    assert index["median"] == pytest.approx(0.1, rel=1e-12)
    assert index["median_abs"] == pytest.approx(0.2, rel=1e-12)


def test_row_misfit_is_zero_on_a_noiseless_model_at_the_truth(monkeypatch):
    # A waterfall the linear model makes exactly, plus the true curve: the
    # compressed chi^2 at the true curve is roundoff, and far from it at zero.
    rng = np.random.default_rng(8)
    n_pix, n_time, freqs = 12, 5, np.linspace(50.0, 110.0, 6)
    response = rng.random((freqs.size, n_time, n_pix))
    template = (np.full(n_pix, -2.6), 100.0 + rng.random(n_pix), -0.1)
    widths = physical_inputs.Widths(1.0, 0.6, 0.05)
    truth = -0.1 * np.exp(-0.5 * ((freqs - 75.0) / 10.0) ** 2)
    model = physical_inputs.linear_model(response, np.zeros((n_time, freqs.size)), freqs, template, widths, 0)
    tiling = np.kron(np.eye(freqs.size), np.ones((n_time, 1)))
    waterfall = (model["jac"] @ model["loc"] + tiling @ truth).reshape(freqs.size, n_time).T
    monkeypatch.setattr(strategies, "prior_bank", lambda f: (np.zeros((2, 7)), np.tile(truth, (2, 1))))
    row = ablation._row(response, waterfall, freqs, template, widths, 0, truth)
    item = collapse.gaussian_prior(waterfall, model["jac"], model["loc"], model["scale"])
    assert row["chi2_at_truth"] < 1e-8 * float(item.chi2(np.zeros_like(truth))[0])
    assert row["rank"] == item.rank
    assert row["bank_fit"]["chi2_min"] == pytest.approx(row["chi2_at_truth"], abs=1e-8)
