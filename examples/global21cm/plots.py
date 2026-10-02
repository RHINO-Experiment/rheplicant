"""Draw every figure from saved products, without running any sampler.

Run from the repository root after ``analyse.py`` (which also calls
:func:`render` itself):

    PYTHONPATH=examples .venv/bin/python -m global21cm.plots          # results/analysis/
    PYTHONPATH=examples .venv/bin/python -m global21cm.plots --quick  # results/analysis_quick/

Inputs, all written by earlier steps: ``fom.json`` and ``posteriors.npz``
(``analyse.py``), the simulation arrays and ``prepare.json`` in
``results/sim*/`` (``simulate.py``, ``prepare.py``), and
``results/analysis/ablation.json`` (``ablation.py``; the quick analysis
reads the same file). The emulator is evaluated to turn saved latents into
curves; nothing is sampled or trained.

Two checks run before anything is drawn, and either mismatch is refused:
the simulation files ``fom.json`` recorded are hashed again, and the trough
depths recomputed from the saved particles must equal the ones ``fom.json``
scored. A figure therefore cannot show particles other than the scored ones.

``posteriors.npz`` holds, for scenario ``s`` in main/stress and model ``m``
in oracle/beamconv/physical:

* ``s__freqs`` (F,) channel centres in MHz and ``s__fine`` the grid the
  trough is read on (``scoring.fine_freqs``);
* ``s__prior__u`` (4000, 7): the prior draws the scores compare against;
* ``s__m__u`` (N, 7), ``s__m__loglik`` (N,), ``s__m__seed`` (N,): the pooled
  SMC particles in unit-normal latents, their log likelihood and SMC seed;
* ``s__m__nuts_u`` (M, 7): the document's NUTS draws in the same latents.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from global21cm_jax.emulator import Global21cmEmulator  # noqa: E402

from global21cm import collapse, figures, fom, scenario, signal21, strategies  # noqa: E402
from global21cm import posterior_io as io  # noqa: E402
from global21cm.prepare import best_structure, sha256  # noqa: E402

POSTERIORS = "posteriors.npz"
MODELS = ("oracle", "beamconv", "physical")
KEYS = ("prior", *MODELS)
PARTICLE_FIELDS = ("u", "loglik", "seed", "nuts_u")
CASES = {c.name: c for c in (scenario.MAIN, scenario.STRESS)}
#: Relative tolerance of the recomputed trough depths against ``fom.json``.
RTOL = 1e-12
#: Rows of the fine-grid curves evaluated at once (a multiple of io.curves' batch).
CHUNK = 20000


# ------------------------------------------------------------ posteriors.npz --
def save_posteriors(path: Path, records: dict) -> None:
    """Write ``records`` (``{scenario: {freqs, fine, prior_u, models: {m: fields}}}``) under the pinned keys."""
    arrays = {}
    for name, record in records.items():
        arrays[f"{name}__freqs"] = np.asarray(record["freqs"], dtype=np.float64)
        arrays[f"{name}__fine"] = np.asarray(record["fine"], dtype=np.float64)
        arrays[f"{name}__prior__u"] = np.asarray(record["prior_u"], dtype=np.float64)
        for model, fields in record["models"].items():
            n = len(fields["u"])
            for field in PARTICLE_FIELDS:
                value = np.asarray(fields[field], dtype=np.int64 if field == "seed" else np.float64)
                if field != "nuts_u" and value.shape[0] != n:
                    raise ValueError(f"{name}/{model}: {field} has {value.shape[0]} rows, u has {n}.")
                arrays[f"{name}__{model}__{field}"] = value
    tmp = Path(path).with_name(Path(path).name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def load_posteriors(path: Path) -> dict:
    """The nested dict :func:`save_posteriors` wrote; refused if any pinned key is missing."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: run `python -m global21cm.analyse` first.")
    with np.load(path) as data:
        flat = {k: np.asarray(data[k]) for k in data.files}
    records: dict = {}
    for key, value in flat.items():
        parts = key.split("__")
        record = records.setdefault(parts[0], {"models": {}})
        if len(parts) == 2:
            record[parts[1]] = value
        elif parts[1] == "prior":
            record["prior_u"] = value
        else:
            record["models"].setdefault(parts[1], {})[parts[2]] = value
    for name, record in records.items():
        missing = [f for f in ("freqs", "fine", "prior_u") if f not in record]
        missing += [f"{m}__{f}" for m in MODELS for f in PARTICLE_FIELDS if f not in record["models"].get(m, {})]
        if missing:
            raise KeyError(f"{path}: scenario {name} lacks {missing}.")
    return records


# ----------------------------------------------------------------- derived --
def _theta(u) -> np.ndarray:
    return io.theta_from_unit(np.asarray(u))


def _trough(theta, fine) -> tuple[np.ndarray, np.ndarray]:
    """``fom.trough`` on the fine grid, in chunks so the curves are never all held."""
    parts = [fom.trough(io.curves(theta[i : i + CHUNK], fine), fine) for i in range(0, len(theta), CHUNK)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def _posterior(u, freqs, fine) -> dict:
    theta = _theta(u)
    depth, where = _trough(theta, fine)
    return {"u": np.asarray(u), "curves": io.curves(theta, freqs), "depth": depth, "where": where}


def derive(case: scenario.Scenario, record: dict) -> dict:
    """Curves, trough depth (K) and frequency (MHz) of every posterior and NUTS run."""
    freqs, fine = record["freqs"], record["fine"]
    truth_depth, truth_where = fom.trough(io.curves(np.asarray(signal21.theta_log_true())[None, :], fine), fine)
    return {
        "freqs": freqs,
        "truth": np.load(case.sim_dir / "t21_truth.npy"),
        "truth_trough": (float(truth_depth[0]), float(truth_where[0])),
        "smc": {"prior": _posterior(record["prior_u"], freqs, fine),
                **{m: _posterior(record["models"][m]["u"], freqs, fine) for m in MODELS}},
        "nuts": {m: _posterior(record["models"][m]["nuts_u"], freqs, fine) for m in MODELS},
    }  # fmt: skip


# ------------------------------------------------------------------ checks --
def _relative(path) -> str:
    """The key ``fom.json`` records a path under: relative to this directory when inside it."""
    resolved = Path(path).resolve()
    return str(resolved.relative_to(scenario.HERE)) if resolved.is_relative_to(scenario.HERE) else str(resolved)


def verify_inputs(report: dict, paths) -> None:
    """Refuse when a file ``fom.json`` hashed has changed, or was never hashed."""
    recorded = report["input_sha256"]
    keys = {_relative(p): p for p in paths}
    unknown = sorted(k for k in keys if k not in recorded)
    stale = sorted(k for k, p in keys.items() if k in recorded and sha256(p) != recorded[k])
    if unknown or stale:
        raise RuntimeError(f"fom.json does not vouch for these inputs: unrecorded {unknown}, changed {stale}")


def verify_particles(report: dict, derived: dict) -> None:
    """Refuse when the saved particles do not reproduce ``fom.json``'s trough depths."""
    bad = []
    for name, d in derived.items():
        scored = {k: report[name][k]["depth_mk"] for k in KEYS}
        scored |= {f"{m} NUTS": report[name][m]["nuts"]["depth_mk"] for m in MODELS}
        drawn = {k: d["smc"][k]["depth"] for k in KEYS} | {f"{m} NUTS": d["nuts"][m]["depth"] for m in MODELS}
        for key, depth in drawn.items():
            again = [float(np.percentile(depth, p) * 1e3) for p in (16, 50, 84)]
            if not np.allclose(again, scored[key], rtol=RTOL, atol=0.0):
                bad.append(f"{name}/{key}: {again} vs {scored[key]}")
    if bad:
        raise RuntimeError("posteriors.npz does not match fom.json:\n" + "\n".join(bad))


# ----------------------------------------------------------- figure inputs --
def overview_inputs() -> dict:
    out = {}
    for name, case in CASES.items():
        sim = {n: np.load(case.sim_dir / f"{n}.npy") for n in ("waterfall", "noiseless", "fg_truth", "t21_truth")}
        out[name] = {"freqs": case.freqs_mhz(), "lst": scenario.lst_deg(), "waterfall": sim["waterfall"],
                     "foreground": sim["fg_truth"], "signal": sim["t21_truth"],
                     "noise": sim["waterfall"] - sim["noiseless"], "beam": case.beam,
                     "band": (case.freq_start_mhz, case.freq_stop_mhz)}  # fmt: skip
    return out


def residual_rows(derived: dict) -> list[dict]:
    """Data minus the posterior-mean model, and recovered minus true foreground, per scenario."""
    rows = []
    for name, case in CASES.items():
        waterfall, fg_truth = (np.load(case.sim_dir / f"{n}.npy") for n in ("waterfall", "fg_truth"))
        fits, fgs = {}, {}
        for model in ("beamconv", "physical"):
            curve = derived[name]["smc"][model]["curves"].mean(axis=0)
            residual = collapse.load(case.sim_dir, model).residual(curve)
            residual = residual.reshape(waterfall.shape[1], waterfall.shape[0]).T
            fits[figures.LABELS[model]] = residual
            fgs[figures.LABELS[model]] = waterfall - curve[None, :] - residual - fg_truth
        freqs = derived[name]["freqs"]
        rows.append({"title": f"{name}: data - model", "freqs": freqs, "panels": fits})
        rows.append({"title": f"{name}: fg recovered - true", "freqs": freqs, "panels": fgs})
    return rows


def recovery_inputs(derived: dict) -> dict:
    return {n: {"freqs": d["freqs"], "truth": d["truth"], "curves": {k: d["smc"][k]["curves"] for k in KEYS}}
            for n, d in derived.items()}  # fmt: skip


def native_spacing(band) -> float:
    """The widest gap of the emulator's own frequency grid inside ``band``, MHz.

    ``signal21.curve_kelvin`` interpolates that grid linearly, so a curve's
    minimum sits at (the fine-grid point nearest) one of its nodes, and a
    histogram of trough frequencies finer than this gap is a comb.
    """
    nodes = np.sort(np.asarray(Global21cmEmulator.load().frequencies_mhz, dtype=np.float64))
    return float(np.diff(nodes[(nodes >= band[0]) & (nodes <= band[1])]).max())


def trough_inputs(derived: dict, records: dict) -> dict:
    out = {}
    for n, d in derived.items():
        band, fine = (d["freqs"][0], d["freqs"][-1]), records[n]["fine"]
        out[n] = {"truth": (d["truth_trough"][0] * 1e3, d["truth_trough"][1]),
                  "depth": {k: d["smc"][k]["depth"] * 1e3 for k in KEYS},
                  "where": {k: d["smc"][k]["where"] for k in KEYS}, "band": band,
                  "step": float(fine[1] - fine[0]), "smooth_mhz": 0.5 * native_spacing(band)}  # fmt: skip
    return out


def corner_inputs(derived: dict) -> dict:
    return {k: derived["main"]["smc"][k]["u"] for k in KEYS}


def summary_inputs(report: dict) -> dict:
    """SER / prior, eta and realisation coverage from ``fom.json`` alone."""
    rows = [(n, m) for n in CASES for m in MODELS]
    ser = [{"scenario": n, "key": m, "value": report[n][m]["ser_over_prior"],
            "calibrated": report[n][m]["calibrated"]} for n, m in rows]  # fmt: skip
    eta = [{"scenario": n, "key": m, "values": {"SMC, all curves": report[n][m]["eta_smc"],
                                                "SMC, central 90 %": report[n][m]["eta_smc_core"],
                                                "Laplace": report[n][m]["eta_laplace"]}}
           for n, m in rows if m != "oracle"]  # fmt: skip
    coverage = [{"scenario": n, "key": m, "n": report[n][m]["realisations"]["n"],
                 **{c: (report[n][m]["realisations"][c]["mean"], report[n][m]["realisations"][c]["se"])
                    for c in ("cov68", "cov95")}} for n, m in rows]  # fmt: skip
    return {"ser": ser, "eta": eta, "coverage": coverage}


def _ablation_label(key: str) -> str:
    """``gsm_1.0_0.6_0.05@16/47`` -> the widths of ``ablation.rows()`` in words."""
    from global21cm.ablation import rows

    name, resolution = key.split("@")
    kind, w, _ = rows()[name]
    nside, lmax = resolution.split("/")
    parts = [f"FA {w.amplitude:g}", f"FB {w.index:g}", f"SC {w.curvature:g}"]
    parts += [f"GA {w.global_amplitude:g}"] if w.global_amplitude else []
    source = "GSM" if kind == "gsm" else "true maps"
    return f"{source}: {', '.join(parts)}; NSIDE {nside}, lmax {lmax}"


def selection_inputs(report: dict, ablation: dict) -> dict:
    forecast = {n: report[n]["beamconv"]["forecast"] for n in CASES}
    order = {n: {**f["model_order"], **{k: f["literal"][k] for k in ("moments", "beam_terms")}}
             for n, f in forecast.items()}  # fmt: skip
    prior = {n: report[n]["physical"]["forecast"]["prior_selection"] for n in CASES}
    entries = {k: v for k, v in ablation.items() if isinstance(v, dict) and "chi2_at_truth" in v}
    # Bold marks the shipped document only: its prior at its own NSIDE and lmax,
    # not the same prior at another resolution.
    shipped = None
    if entries:
        literal = report["main"]["physical"]["forecast"]["literal"]
        shipped = f"chosen@{literal['nside']}/{literal['lmax']}"
        if shipped not in entries:
            raise RuntimeError(f"ablation.json has no row {shipped!r}, the shipped prior.")
    rows = [{"label": _ablation_label(k), "chi2_truth": v["chi2_at_truth"], "chi2_bank": v["bank_fit"]["chi2_min"],
             "rank": v["rank"], "chosen": k == shipped} for k, v in entries.items()]  # fmt: skip
    for name, block in prior.items():
        if best_structure(block["grid"]) != block["best"]:
            raise RuntimeError(f"{name}: the chosen prior is not the best-evidence cell with p >= "
                               f"{strategies.FIT_CHECK_P}.")  # fmt: skip
    for name, block in order.items():
        if strategies.choose_order(block["table"], block["rule"]) != block["chosen"]:
            raise RuntimeError(f"{name}: demo A's recorded order is not rule {block['rule']!r}'s choice.")
    return {"order": order, "prior": prior, "ablation": rows}


def nuts_inputs(derived: dict, report: dict) -> dict:
    out = {}
    for name, d in derived.items():
        stats = {}
        for m in MODELS:
            nuts, cmp = report[name][m]["nuts"], report[name][m]["nuts_vs_smc"]
            high = cmp["high_nu_min_fraction"]
            stats[m] = {"r_hat": nuts["r_hat_max"], "divergences": nuts["divergences"],
                        "spread": cmp["spread_nuts_over_smc"], "high_nu_min": (high["nuts"], high["smc"])}  # fmt: skip
        out[name] = {"freqs": d["freqs"], "truth": d["truth"], "stats": stats,
                     "smc": {m: d["smc"][m]["curves"] for m in MODELS},
                     "nuts": {m: d["nuts"][m]["curves"] for m in MODELS}}  # fmt: skip
    return out


# ------------------------------------------------------------------ render --
def render(out_dir: Path, quick: bool = False) -> list[Path]:
    """Every PNG of ``out_dir`` from its ``fom.json`` and ``posteriors.npz``; returns the paths."""
    out_dir = Path(out_dir)
    report = json.loads((out_dir / "fom.json").read_text())
    ablation = json.loads((scenario.RESULTS / "analysis" / "ablation.json").read_text())
    names = ("waterfall.npy", "fg_truth.npy", *(f"{m}_collapsed.npz" for m in MODELS))
    verify_inputs(report, [c.sim_dir / n for c in CASES.values() for n in names])
    records = load_posteriors(out_dir / POSTERIORS)
    derived = {name: derive(CASES[name], records[name]) for name in CASES}
    verify_particles(report, derived)
    runs = "quick runs" if quick else "default runs"
    paths = {name: out_dir / f"{name}.png" for name in (
        "scenario_overview", "signal_recovery", "trough", "corner", "residual_waterfalls",
        "fom_summary", "model_selection", "nuts_vs_smc")}  # fmt: skip
    figures.scenario_overview(overview_inputs(), paths["scenario_overview"])
    figures.signal_recovery(recovery_inputs(derived), paths["signal_recovery"])
    figures.trough(trough_inputs(derived, records), paths["trough"])
    figures.corner(corner_inputs(derived), np.asarray(signal21.u_true()), paths["corner"])
    figures.residual_waterfalls(residual_rows(derived), paths["residual_waterfalls"])
    figures.fom_summary(summary_inputs(report), paths["fom_summary"])
    figures.model_selection(selection_inputs(report, ablation), paths["model_selection"], strategies.FIT_CHECK_P)
    figures.nuts_vs_smc(nuts_inputs(derived, report), paths["nuts_vs_smc"], runs)
    return list(paths.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="draw results/analysis_quick/")
    quick = parser.parse_args().quick
    for path in render(scenario.RESULTS / f"analysis{'_quick' if quick else ''}", quick):
        print(path.relative_to(scenario.HERE), flush=True)


if __name__ == "__main__":
    main()
