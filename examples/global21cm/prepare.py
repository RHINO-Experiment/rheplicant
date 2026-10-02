"""Build each document's observation from its own strategy, and record it.

For one scenario, and for each of ``oracle.yaml``, ``beamconv.yaml`` and
``physical.yaml``, this reads the document's ``resources.arrays.foreground``
entry (the stress variant merged in for ``stress``), calls the same
:mod:`global21cm.strategies` function the document names with the same
arguments, and writes into ``results/sim*/``:

* ``<model>_data.npy``, the document's ``inference.observed`` file, which
  the document checks against its own strategy when it loads
  (``strategies.design_checked``);
* ``<model>_collapsed.npz``, the compressed likelihood for ``analyse.py``;
* ``prepare.json``: demo A's model-order table (chi^2, p-value, fit-check
  flag and BIC of every candidate, and the rule's choice), demo B's
  prior-structure grid (evidence and fit check of each, and whether the
  document's literal is the selection rule's choice), the sampler-free
  amplitude forecasts, and the sha256 of every file written.

Both selections share one fit check, ``strategies.FIT_CHECK_P``.

Run from the repository root after simulate.py:

    PYTHONPATH=examples .venv/bin/python -m global21cm.prepare main
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from global21cm import collapse, physical_inputs, scenario, strategies  # noqa: E402

MODELS = ("oracle", "beamconv", "physical")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _merge(base: dict, patch: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in patch.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def document_spec(model: str, case: scenario.Scenario) -> tuple:
    """``(function, arrays, literal)`` of the document's foreground entry."""
    document = yaml.safe_load((scenario.HERE / f"{model}.yaml").read_text())
    if case.name != "main":
        document = _merge(document, document["variants"][case.name])
    entry = document["resources"]["arrays"]["foreground"]
    name = entry["python"].split(":")[1]
    function = getattr(strategies, name)
    arrays = {key: np.load(scenario.HERE / node["file"]["path"]) for key, node in entry["args"].items()}
    return function, arrays, dict(entry.get("literal", {}))


def prior_selection(arrays: dict, literal: dict) -> list[dict]:
    """Every prior structure in the declared grid: fit check at the bank's best, and evidence.

    The rule is the highest evidence among the structures whose best
    prior-bank curve fits the compressed statistic at
    ``p >= strategies.FIT_CHECK_P`` (:func:`best_structure`). The evidence
    alone is dominated by the thousands of foreground directions and is
    blind to the few the signal moves; the fit check is the one that sees
    those.
    """
    freqs = np.asarray(arrays["freqs_mhz"])
    gsm = physical_inputs.gsm_template(int(literal["nside"]))
    response = physical_inputs.response(arrays["beam_alm"], int(literal["nside"]), int(literal["lmax"]))
    rows = []
    for fa, ga, gb in physical_inputs.PRIOR_GRID:
        widths = physical_inputs.Widths(fa, float(literal["index_sigma"]), float(literal["curvature_sigma"]), ga, gb)
        model = physical_inputs.linear_model(response, arrays["waterfall"], freqs, gsm, widths,
                                             int(literal["relinearise"]))  # fmt: skip
        item = collapse.from_marginal(arrays["waterfall"], model["marginal"])
        fit = strategies.bank_fit(item, freqs)
        rows.append({"amplitude_frac": fa, "global_amplitude_sigma": ga, "global_index_sigma": gb,
                     "log_evidence": physical_inputs.log_evidence(model, arrays["waterfall"]),
                     "bank_chi2": fit["chi2_min"], "bank_dof": fit["dof"], "bank_p": fit["p"]})  # fmt: skip
    return rows


def best_structure(grid: list[dict]) -> dict | None:
    """The rule's structure: the highest evidence among the rows that pass the fit check."""
    passing = [r for r in grid if r["bank_p"] is not None and r["bank_p"] >= strategies.FIT_CHECK_P]
    return max(passing, key=lambda r: r["log_evidence"]) if passing else None


def _summary(model: str, item: collapse.Collapsed, sim: dict) -> dict:
    out = {"rank": item.rank, "dof": item.dof,
           **collapse.amplitude_forecast(item, sim["t21_truth"], sim["noiseless"]),
           "marginal_chi2_truth": item.marginal_chi2(sim["t21_truth"]),
           "compressed_chi2_truth": float(item.chi2(sim["t21_truth"])[0])}  # fmt: skip
    info = dict(item.info)
    if model == "beamconv":
        out["model_order"] = info["model_order"]
    if model == "physical":
        out["linearisation"] = info
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="build one scenario's observations")
    parser.add_argument("scenario", choices=sorted(scenario.SCENARIOS))
    case = scenario.SCENARIOS[parser.parse_args().scenario]
    start, out = time.perf_counter(), case.sim_dir
    sim = {n: np.load(out / f"{n}.npy") for n in ("noiseless", "t21_truth")}
    record: dict = {"scenario": case.name, "models": {}, "written": {}}
    for model in MODELS:
        t0 = time.perf_counter()
        function, arrays, literal = document_spec(model, case)
        item = function(**arrays, **literal)
        collapse.save(item, out, model)
        record["models"][model] = {**_summary(model, item, sim), "literal": literal,
                                   "seconds": round(time.perf_counter() - t0, 1)}  # fmt: skip
        if model == "physical":
            grid = prior_selection(arrays, literal)
            best = best_structure(grid)
            chosen = (literal["amplitude_frac"], literal["global_amplitude_sigma"], literal["global_index_sigma"])
            record["models"][model]["prior_selection"] = {"grid": grid, "best": best,
                "document_is_best": best is not None and chosen == (best["amplitude_frac"],
                best["global_amplitude_sigma"], best["global_index_sigma"])}  # fmt: skip
        for suffix in ("_data.npy", "_collapsed.npz"):
            record["written"][model + suffix] = sha256(out / f"{model}{suffix}")
    record["seconds"] = round(time.perf_counter() - start, 1)
    (out / "prepare.json").write_text(json.dumps(record, indent=2) + "\n")
    for model, entry in record["models"].items():
        brief = {k: v for k, v in entry.items() if k not in ("model_order", "linearisation", "prior_selection")}
        print(model, json.dumps(brief))
    order = record["models"]["beamconv"]["model_order"]
    for row in order["table"]:
        print("order", json.dumps(row))
    print("order rule", order["rule"], "chosen", json.dumps(order["chosen"]))
    selection = record["models"]["physical"]["prior_selection"]
    for row in selection["grid"]:
        print("prior", json.dumps(row))
    print("best", json.dumps(selection["best"]), "document_is_best", selection["document_is_best"])


if __name__ == "__main__":
    main()
