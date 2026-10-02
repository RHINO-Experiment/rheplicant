"""Score every posterior of both scenarios and draw the figures.

Run from the repository root after prepare.py and the three documents:

    PYTHONPATH=examples .venv/bin/python -m global21cm.analyse          # default runs
    PYTHONPATH=examples .venv/bin/python -m global21cm.analyse --quick  # *_quick runs

The headline posterior of each model is a tempered SMC (four seeds pooled);
the documents' NUTS runs are compared with it. Before anything is scored,
every file a run tree records in ``provenance.json`` and every file
``prepare.json`` records is hashed again, and a mismatch is refused.

Writes ``results/analysis[_quick]/fom.json``, saves the scored particles to
``posteriors.npz`` beside it (:func:`global21cm.plots.save_posteriors`), draws
every figure from those files (:func:`global21cm.plots.render`, which
``python -m global21cm.plots`` reruns without any SMC), and prints the tables
the README quotes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402

from global21cm import collapse, fom, plots, scenario, scoring, signal21, smc, strategies  # noqa: E402
from global21cm import posterior_io as io  # noqa: E402
from global21cm.prepare import sha256  # noqa: E402
from global21cm.report import print_tables  # noqa: E402

MODELS = ("oracle", "beamconv", "physical")
RUN_OF = {"main": "posterior", "stress": "stress"}
N_REALISATIONS = {"main": 100, "stress": 50}
SMC_SEEDS = (1, 2, 3, 4)
#: The central fraction of posterior curves (by distance from the median curve)
#: the core of the efficiency is measured on.
CORE = 0.9


def check_inputs(trees, cases) -> dict:
    """sha256 of every input the run trees and prepare.json name, refused if stale."""
    seen = {}
    for tree in trees:
        for entry in json.loads((tree / "provenance.json").read_text())["inputs"]:
            seen[entry["path"]] = entry["sha256"]
    for case in cases:
        for name, digest in json.loads((case.sim_dir / "prepare.json").read_text())["written"].items():
            seen[str(case.sim_dir / name)] = digest
    stale = [p for p, digest in seen.items() if sha256(p) != digest]
    if stale:
        raise RuntimeError(f"these inputs changed after the runs that read them: {stale}")
    return {str(Path(p).resolve().relative_to(scenario.HERE)): digest for p, digest in seen.items()}


def _truth(case, fine) -> dict:
    theta = np.asarray(signal21.theta_log_true())
    depth, where = fom.trough(io.curves(theta[None, :], fine), fine)
    return {"u": np.asarray(signal21.u_true()), "curve": np.load(case.sim_dir / "t21_truth.npy"),
            "depth": float(depth[0]), "where": float(where[0])}  # fmt: skip


def _draws(u, freqs, fine) -> dict:
    theta = io.theta_from_unit(u)
    return {"u": u, "curves": io.curves(theta, freqs), "fine": io.curves(theta, fine)}


def _smc(collapsed, freqs) -> tuple[np.ndarray, np.ndarray, list, np.ndarray]:
    """Pooled particles, their log likelihood, per-seed summaries, and each particle's seed."""
    runs = [smc.run(smc.log_likelihood(collapsed.design, collapsed.data, freqs), seed) for seed in SMC_SEEDS]
    return (np.concatenate([r["u"] for r in runs]), np.concatenate([r["loglik"] for r in runs]),
            [{"log_z": r["log_z"], "steps": r["steps"],
              "trace": fom.spread(io.curves(io.theta_from_unit(r["u"]), freqs))} for r in runs],
            np.concatenate([np.full(len(r["u"]), seed) for r, seed in zip(runs, SMC_SEEDS)]))  # fmt: skip


def tail_decomposition(draws, fine) -> dict:
    """How much of the curve spread sits in a weak-signal tail."""
    curves = draws["curves"]
    distance = np.sum((curves - np.median(curves, axis=0)) ** 2, axis=1)
    core = curves[np.argsort(distance)[: int(CORE * len(curves))]]
    depth = fom.trough(draws["fine"], fine)[0]
    return {"trace": fom.spread(curves), "core_trace": fom.spread(core),
            "weak_fraction": float(np.mean(depth > -0.05)),
            "depth_quantiles_mk": {q: float(np.percentile(depth, q) * 1e3) for q in (1, 5, 50, 95, 99)}}  # fmt: skip


def calibrated_verdict(calibrated_signal: bool, gof: dict) -> bool:
    """Calibrated: the signal passes :func:`fom.calibrated` and the whole fit passes the GoF gate."""
    return bool(calibrated_signal and gof["passes"])


def efficiencies(entry: dict, oracle: dict, curves, oracle_curves) -> dict:
    """``eta_laplace`` always; ``eta_smc`` and its core only when both posteriors are calibrated.

    The core efficiency sets the oracle's whole spread against this model's
    central ``CORE`` fraction (:func:`tail_decomposition`).
    """
    both = entry["calibrated"] and oracle["calibrated"]
    return {"eta_laplace": float(np.sqrt(oracle["laplace_trace"] / entry["laplace_trace"])),
            "eta_smc": fom.efficiency(oracle_curves, curves) if both else None,
            "eta_smc_core": (float(np.sqrt(oracle["tail"]["trace"] / entry["tail"]["core_trace"]))
                             if both else None)}  # fmt: skip


def _compare(nuts, ref) -> dict:
    sd = ref["curves"].std(axis=0)
    shift = np.abs(nuts["curves"].mean(axis=0) - ref["curves"].mean(axis=0)) / np.where(sd > 0, sd, np.inf)
    return {"spread_nuts_over_smc": fom.spread(nuts["curves"]) / fom.spread(ref["curves"]),
            "mean_shift_max_sd": float(shift.max()),
            "high_nu_min_fraction": {"nuts": float(np.mean(nuts["u"][:, 5] > 0)),
                                     "smc": float(np.mean(ref["u"][:, 5] > 0))}}  # fmt: skip


def score_model(case, model, suffix, context) -> dict:
    freqs, fine, truth, prior = context["freqs"], context["fine"], context["truth"], context["prior"]
    collapsed = collapse.load(case.sim_dir, model)
    tree = scenario.RESULTS / f"{model}{suffix}"
    nuts = _draws(io.unit_draws(io.load_mapping(tree, RUN_OF[case.name], "draws")), freqs, fine)
    t0 = time.perf_counter()
    u, loglik, runs, seeds = _smc(collapsed, freqs)
    ref = _draws(u, freqs, fine)
    smc_seconds = time.perf_counter() - t0
    gof = scoring.goodness_of_fit(collapsed, ref, loglik, seed=5)
    entry = {
        **scoring.score(ref, truth, prior, fine),
        "gof": gof,
        "smc": {"runs": runs, "seconds": round(smc_seconds, 1), "particles": int(len(u))},
        "nuts": {**scoring.score(nuts, truth, prior, fine), **io.nuts_summary(tree, RUN_OF[case.name])},
        "nuts_vs_smc": _compare(nuts, ref),
        "forecast": context["prepared"]["models"][model],
        "laplace_trace": scoring.laplace_trace(collapsed.design, freqs, truth["u"]),
        "tail": tail_decomposition(ref, fine),
    }
    entry["calibrated"] = calibrated_verdict(entry["calibrated_signal"], gof)
    t0 = time.perf_counter()
    entry["realisations"] = scoring.realisations(collapsed, truth, context["noiseless"], freqs, fine,
                                                 N_REALISATIONS[case.name], seed=11)  # fmt: skip
    entry["realisations"]["seconds"] = round(time.perf_counter() - t0, 1)
    # The scored draws plus what the figures need to redraw them without SMC.
    return entry, {**ref, "loglik": loglik, "seed": seeds, "nuts_u": nuts["u"]}


def score_scenario(case: scenario.Scenario, quick: bool) -> tuple[dict, dict]:
    freqs, fine = case.freqs_mhz(), scoring.fine_freqs(case)
    bank_u, _ = strategies.prior_bank(tuple(float(v) for v in freqs))
    context = {"freqs": freqs, "fine": fine, "truth": _truth(case, fine),
               "prior": _draws(bank_u[:4000], freqs, fine),
               "noiseless": np.load(case.sim_dir / "noiseless.npy"),
               "prepared": json.loads((case.sim_dir / "prepare.json").read_text())}  # fmt: skip
    report = {"prior": scoring.score(context["prior"], context["truth"], context["prior"], fine)}
    shown = {"prior": context["prior"]}
    for model in MODELS:
        report[model], shown[model] = score_model(case, model, "_quick" if quick else "", context)
    for model in ("beamconv", "physical"):
        report[model] = {**report[model], **efficiencies(report[model], report["oracle"], shown[model]["curves"],
                                                         shown["oracle"]["curves"])}  # fmt: skip
    return report, {"shown": shown, "truth": context["truth"], "freqs": freqs}


def posterior_record(case: scenario.Scenario, context: dict) -> dict:
    """The arrays :func:`global21cm.plots.save_posteriors` writes for one scenario."""
    shown = context["shown"]
    return {"freqs": context["freqs"], "fine": scoring.fine_freqs(case), "prior_u": shown["prior"]["u"],
            "models": {m: {k: shown[m][k] for k in plots.PARTICLE_FIELDS} for m in MODELS}}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="score the *_quick run trees")
    quick = parser.parse_args().quick
    suffix = "_quick" if quick else ""
    cases = (scenario.MAIN, scenario.STRESS)
    hashes = check_inputs([scenario.RESULTS / f"{m}{suffix}" for m in MODELS], cases)
    out = scenario.RESULTS / f"analysis{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    reports, contexts = {"input_sha256": hashes}, {}
    for case in cases:
        reports[case.name], contexts[case.name] = score_scenario(case, quick)
    (out / "fom.json").write_text(json.dumps(reports, indent=2, default=float) + "\n")
    plots.save_posteriors(out / plots.POSTERIORS, {c.name: posterior_record(c, contexts[c.name]) for c in cases})
    plots.render(out, quick)
    print_tables({k: v for k, v in reports.items() if k != "input_sha256"})


if __name__ == "__main__":
    main()
