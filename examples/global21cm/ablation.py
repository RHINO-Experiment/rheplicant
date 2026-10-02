"""Where demo B's misfit comes from: its prior, one change at a time.

Every row fits demo B's linear model (``physical_inputs``, ``collapse``) to
the main scenario's noiseless waterfall (the NSIDE 64 truth plus the true
21 cm curve) and reports the compressed chi^2 at the true curve. With no
noise that number is pure model misfit in the directions the signal can
move; ``strategies.bank_fit`` on the same model gives the truth-free fit
check the prior selection uses. The rows:

* ``gsm_1.0_0.6_0.05``: the round-2 prior (GSM template, per-pixel widths);
* the same with one width widened at a time, then all three;
* ``truth_template_wide``: the true NSIDE 16 maps as template, wide widths,
  no relinearisation: what pixelisation and lmax 47 alone cost;
* ``chosen``: the shipped prior, read from ``physical.yaml``'s foreground
  literal (:func:`shipped_prior`), at NSIDE 16 / lmax 47 and at NSIDE 32 /
  lmax 63.

It also records GSM's template against the truth's at NSIDE 16
(:func:`template_offsets`): the 70 MHz amplitude as a ratio and the
spectral index as a difference ``beta_GSM - beta_true``.

Run from the repository root after simulate.py (writes
``results/analysis/ablation.json``):

    PYTHONPATH=examples .venv/bin/python -m global21cm.ablation
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from global21cm import collapse, foreground_physical as phys, instrument, physical_inputs, scenario, strategies  # noqa: E402

W = physical_inputs.Widths
#: ``(template, widths, relinearise)`` of each fixed row; :func:`rows` adds
#: ``chosen`` from the document, so the shipped prior is not spelled here.
FIXED_ROWS = {
    "gsm_1.0_0.6_0.05": ("gsm", W(1.0, 0.6, 0.05), 3),
    "gsm_curvature_1": ("gsm", W(1.0, 0.6, 1.0), 3),
    "gsm_index_3": ("gsm", W(1.0, 3.0, 0.05), 3),
    "gsm_amplitude_10": ("gsm", W(10.0, 0.6, 0.05), 3),
    "gsm_all_wide": ("gsm", W(10.0, 3.0, 1.0), 3),
    "truth_template_wide": ("truth", W(10.0, 3.0, 1.0), 0),
}
DOCUMENT = scenario.HERE / "physical.yaml"


def shipped_prior(document: Path = DOCUMENT) -> tuple[str, physical_inputs.Widths, int]:
    """``(template, widths, relinearise)`` of demo B's prior, from the document's foreground literal.

    The literal's keys are the ones :func:`global21cm.strategies.gaussian_sky`
    receives, so a change of the document's prior moves this row with it.
    """
    literal = yaml.safe_load(Path(document).read_text())["resources"]["arrays"]["foreground"]["literal"]
    if literal["template"] != "gsm2008":
        raise ValueError(f"{document}: template {literal['template']!r}; the ablation's rows use 'gsm2008'.")
    widths = W(float(literal["amplitude_frac"]), float(literal["index_sigma"]), float(literal["curvature_sigma"]),
               float(literal["global_amplitude_sigma"]), float(literal["global_index_sigma"]))  # fmt: skip
    return "gsm", widths, int(literal["relinearise"])


def rows(document: Path = DOCUMENT) -> dict:
    """The fixed rows, then ``chosen``: the prior ``document`` ships."""
    return {**FIXED_ROWS, "chosen": shipped_prior(document)}


def template_offsets(gsm, truth) -> dict:
    """GSM's ``(beta, a_70, c)`` template against the truth's, pixel by pixel at one NSIDE."""
    ratio = np.asarray(gsm[1]) / np.asarray(truth[1])
    dbeta = np.asarray(gsm[0]) - np.asarray(truth[0])
    return {"gsm_amplitude_over_truth": {"median": float(np.median(ratio)),
                                         "p16": float(np.percentile(ratio, 16)),
                                         "p84": float(np.percentile(ratio, 84))},
            "gsm_index_minus_truth": {"median": float(np.median(dbeta)),
                                      "median_abs": float(np.median(np.abs(dbeta))),
                                      "p16": float(np.percentile(dbeta, 16)),
                                      "p84": float(np.percentile(dbeta, 84))}}  # fmt: skip


def _row(response, waterfall, freqs, template, widths, relinearise, truth) -> dict:
    model = physical_inputs.linear_model(response, waterfall, freqs, template, widths, relinearise)
    item = collapse.gaussian_prior(waterfall, model["jac"], model["loc"], model["scale"])
    return {"chi2_at_truth": float(item.chi2(truth)[0]), "rank": item.rank,
            "bank_fit": strategies.bank_fit(item, freqs)}  # fmt: skip


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    case, start = scenario.MAIN, time.perf_counter()
    freqs = case.freqs_mhz()
    sim = {n: np.load(case.sim_dir / f"{n}.npy") for n in ("noiseless", "t21_truth", "t408_fit", "beta_fit")}
    beams = instrument.beam_maps(case, scenario.NSIDE_SIM)
    truth_template = (sim["beta_fit"], np.asarray(phys.amplitude_at_nu0(sim["t408_fit"], sim["beta_fit"])),
                      scenario.CURVATURE_TRUE)  # fmt: skip
    record, table = {}, rows()
    for nside, lmax, names in ((16, 47, list(table)), (32, 63, ["chosen"])):
        response = instrument.drift_response(instrument.beam_alms(beams, lmax), nside, lmax, scenario.lst_deg())
        gsm = physical_inputs.gsm_template(nside)
        for name in names:
            kind, widths, relinearise = table[name]
            template = gsm if kind == "gsm" else truth_template
            if kind == "truth" and nside != scenario.NSIDE_FIT:
                continue
            key = f"{name}@{nside}/{lmax}"
            record[key] = _row(response, sim["noiseless"], freqs, template, widths, relinearise, sim["t21_truth"])
            print(key, json.dumps(record[key]), flush=True)
    record.update(template_offsets(physical_inputs.gsm_template(scenario.NSIDE_FIT), truth_template))
    record["seconds"] = round(time.perf_counter() - start, 1)
    out = scenario.RESULTS / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    (out / "ablation.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
