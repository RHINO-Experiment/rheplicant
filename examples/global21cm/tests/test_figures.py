"""The figure layer on synthetic inputs: contour levels, posteriors.npz, the refusals, every PNG."""

import copy

import numpy as np
import pytest

from global21cm import figures, plots, strategies

RNG = np.random.default_rng(3)
KEYS = ("prior", "oracle", "beamconv", "physical")
MODELS = KEYS[1:]


# ------------------------------------------------------------ contour levels --
def test_gaussian_grid_levels_sit_where_the_density_is_one_minus_the_mass():
    """For a 2-D Gaussian the HPD region of mass p ends where the density is (1 - p) of its peak."""
    axis = np.linspace(-6.0, 6.0, 1201)
    x, y = np.meshgrid(axis, axis)
    density = np.exp(-0.5 * (x**2 + y**2))
    levels = figures.mass_levels(density, (0.68, 0.95)) / density.max()
    np.testing.assert_allclose(levels, [0.32, 0.05], rtol=5e-3)


def test_smoothed_68_percent_region_encloses_68_percent_of_the_samples():
    """Anisotropic, so a density indexed [x, y] instead of [y, x] encloses the wrong fraction."""
    samples = RNG.standard_normal((200_000, 2)) * np.array([1.0, 0.35])
    edges = np.linspace(-5.0, 5.0, 61)
    _, _, density, counts = figures.density_2d(samples[:, 0], samples[:, 1], [(-5.0, 5.0)] * 2, bins=60, smooth=1.2)
    ix, iy = (np.clip(np.digitize(samples[:, k], edges) - 1, 0, 59) for k in (0, 1))
    (level,) = figures.mass_levels(density, (0.68,), counts)
    assert abs(float(np.mean(density[iy, ix] >= level)) - 0.68) < 0.005
    # Ranked by the smoothed density but weighted by itself, the region is wider than 68 % of the samples.
    (level,) = figures.mass_levels(density, (0.68,))
    assert float(np.mean(density[iy, ix] >= level)) > 0.70


def test_levels_fall_as_the_mass_grows_and_hold_at_least_that_mass():
    density = RNG.random((30, 30))
    masses = (0.2, 0.68, 0.95, 1.0)
    levels = figures.mass_levels(density, masses)
    assert np.all(np.diff(levels) <= 0.0)
    for mass, level in zip(masses, levels, strict=True):
        assert density[density >= level].sum() / density.sum() >= mass - 1e-12


@pytest.mark.parametrize("masses, density", [((0.0,), np.ones(4)), ((1.2,), np.ones(4)), ((0.5,), -np.ones(4)),
                                             ((0.5,), np.zeros(4))])  # fmt: skip
def test_mass_levels_refuses_what_has_no_answer(masses, density):
    with pytest.raises(ValueError):
        figures.mass_levels(density, masses)


# ------------------------------------------------------------ posteriors.npz --
def _records(n: int = 12, m: int = 5) -> dict:
    out = {}
    for name, n_freq in (("main", 4), ("stress", 3)):
        models = {k: {"u": RNG.standard_normal((n, 7)), "loglik": RNG.standard_normal(n),
                      "seed": np.repeat([1, 2, 3, 4], n // 4), "nuts_u": RNG.standard_normal((m, 7))}
                  for k in MODELS}  # fmt: skip
        out[name] = {"freqs": np.linspace(45.0, 135.0, n_freq), "fine": np.arange(45.0, 135.1, 0.25),
                     "prior_u": RNG.standard_normal((8, 7)), "models": models}  # fmt: skip
    return out


def test_posteriors_round_trip_under_the_pinned_keys(tmp_path):
    records, path = _records(), tmp_path / plots.POSTERIORS
    plots.save_posteriors(path, records)
    with np.load(path) as raw:
        keys, seed_kind = set(raw.files), raw["main__oracle__seed"].dtype.kind
    pinned = {f"{s}__{f}" for s in records for f in ("freqs", "fine")} | {f"{s}__prior__u" for s in records}
    pinned |= {f"{s}__{m}__{f}" for s in records for m in MODELS for f in ("u", "loglik", "seed", "nuts_u")}
    assert keys == pinned and seed_kind == "i"
    loaded = plots.load_posteriors(path)
    for s, record in records.items():
        for field in ("freqs", "fine", "prior_u"):
            np.testing.assert_array_equal(loaded[s][field], record[field])
        for m in MODELS:
            for field, value in record["models"][m].items():
                np.testing.assert_array_equal(loaded[s]["models"][m][field], value)


def test_a_missing_key_or_file_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError):
        plots.load_posteriors(tmp_path / "absent.npz")
    records, path = _records(), tmp_path / plots.POSTERIORS
    plots.save_posteriors(path, records)
    with np.load(path) as raw:
        kept = {k: raw[k] for k in raw.files if k != "stress__physical__loglik"}
    np.savez(path, **kept)
    with pytest.raises(KeyError, match="physical__loglik"):
        plots.load_posteriors(path)


def test_rows_of_unequal_length_are_refused(tmp_path):
    records = _records()
    records["main"]["models"]["oracle"]["loglik"] = np.zeros(3)
    with pytest.raises(ValueError, match="loglik"):
        plots.save_posteriors(tmp_path / plots.POSTERIORS, records)


# ------------------------------------------------------------------ refusals --
def _scored(derived: dict) -> dict:
    q = lambda d: [float(np.percentile(d, p) * 1e3) for p in (16, 50, 84)]  # noqa: E731
    return {n: {**{k: {"depth_mk": q(d["smc"][k]["depth"])} for k in KEYS},
                **{m: {"depth_mk": q(d["smc"][m]["depth"]), "nuts": {"depth_mk": q(d["nuts"][m]["depth"])}}
                   for m in MODELS}} for n, d in derived.items()}  # fmt: skip


def test_particles_that_do_not_reproduce_the_scored_depths_are_refused():
    derived = {"main": {"smc": {k: {"depth": RNG.normal(-0.15, 0.01, 50)} for k in KEYS},
                        "nuts": {m: {"depth": RNG.normal(-0.15, 0.01, 20)} for m in MODELS}}}  # fmt: skip
    report = _scored(derived)
    plots.verify_particles(report, derived)
    derived["main"]["nuts"]["physical"]["depth"] = derived["main"]["nuts"]["physical"]["depth"] + 1e-9
    with pytest.raises(RuntimeError, match="physical NUTS"):
        plots.verify_particles(report, derived)


def test_inputs_fom_json_did_not_hash_or_that_changed_are_refused(tmp_path):
    path = tmp_path / "waterfall.npy"
    np.save(path, np.arange(3.0))
    with pytest.raises(RuntimeError, match="unrecorded"):
        plots.verify_inputs({"input_sha256": {}}, [path])
    with pytest.raises(RuntimeError, match="changed"):
        plots.verify_inputs({"input_sha256": {str(path.resolve()): "0" * 64}}, [path])
    plots.verify_inputs({"input_sha256": {str(path.resolve()): plots.sha256(path)}}, [path])


def _report(grid, best, table, chosen):
    order = {"rule": "bic_among_passing", "table": table, "chosen": chosen}
    forecast = {"model_order": order, "literal": {"moments": [1, 8], "beam_terms": [0, 4]}}
    return {n: {"beamconv": {"forecast": copy.deepcopy(forecast)},
                "physical": {"forecast": {"prior_selection": {"grid": grid, "best": best}}}}
            for n in plots.CASES}  # fmt: skip


# A failing K = 1 with the lowest BIC, the passing K = 3 the rule's choice, then K = 2.
ORDER_TABLE = [{"K": 1, "beam_terms": 0, "bic": 90.0, "p": 1e-6, "passes": False},
               {"K": 2, "beam_terms": 0, "bic": 120.0, "p": 0.3, "passes": True},
               {"K": 3, "beam_terms": 0, "bic": 100.0, "p": 0.5, "passes": True}]  # fmt: skip


def test_a_recorded_prior_choice_that_breaks_the_selection_rule_is_refused():
    grid = [{"log_evidence": 5.0, "bank_p": 1e-9}, {"log_evidence": 3.0, "bank_p": 0.2},
            {"log_evidence": 1.0, "bank_p": 0.9}]  # fmt: skip
    report = _report(grid, grid[1], ORDER_TABLE, ORDER_TABLE[2])
    assert plots.selection_inputs(report, {})["prior"]["main"]["best"] is grid[1]
    report["stress"]["physical"]["forecast"]["prior_selection"]["best"] = grid[2]
    with pytest.raises(RuntimeError, match="stress"):
        plots.selection_inputs(report, {})


@pytest.mark.parametrize("wrong", [0, 1])
def test_a_recorded_order_that_breaks_the_rule_is_refused(wrong):
    grid = [{"log_evidence": 3.0, "bank_p": 0.2}]
    report = _report(grid, grid[0], ORDER_TABLE, ORDER_TABLE[2])
    assert plots.selection_inputs(report, {})["order"]["main"]["chosen"] == ORDER_TABLE[2]
    # The lowest BIC fails the fit check (0); the other passing row has a higher BIC (1).
    report["main"]["beamconv"]["forecast"]["model_order"]["chosen"] = ORDER_TABLE[wrong]
    with pytest.raises(RuntimeError, match="main: demo A"):
        plots.selection_inputs(report, {})


def test_ablation_labels_come_from_the_ablation_rows():
    assert plots._ablation_label("chosen@32/63") == "GSM: FA 3, FB 0.6, SC 0.05, GA 10; NSIDE 32, lmax 63"


# ------------------------------------------------------------------- figures --
F = np.linspace(45.0, 135.0, 6)
TRUTH = -0.15 * np.exp(-0.5 * ((F - 72.0) / 10.0) ** 2)


def _curves(n: int = 40) -> np.ndarray:
    return TRUTH[None, :] + RNG.normal(0.0, 0.01, (n, F.size))


def _overview():
    lst = 360.0 * np.arange(8) / 8
    sky = 1e3 * (F[None, :] / 70.0) ** -2.5 * (1.0 + 0.2 * np.cos(np.radians(lst))[:, None])
    return {"main": {"freqs": F, "lst": lst, "waterfall": sky, "foreground": sky, "signal": TRUTH,
                     "noise": RNG.normal(0.0, 0.01, sky.shape), "beam": "gaussian", "band": (45.0, 135.0)}}  # fmt: skip


def _trough():
    where = {k: RNG.choice(np.arange(45.0, 135.1, 0.25), 300) for k in KEYS}
    return {"main": {"truth": (-152.1, 72.0), "depth": {k: RNG.normal(-150.0, 30.0, 300) for k in KEYS},
                     "where": where, "band": (45.0, 135.0), "step": 0.25, "smooth_mhz": 0.6}}  # fmt: skip


def _summary():
    rows = [(s, k) for s in ("main", "stress") for k in MODELS]
    return {"ser": [{"scenario": s, "key": k, "value": 2.0, "calibrated": k != "beamconv"} for s, k in rows],
            "eta": [{"scenario": s, "key": k, "values": {"SMC, all curves": None if s == "stress" else 0.01,
                                                         "SMC, central 90 %": 0.02, "Laplace": 0.05}}
                    for s, k in rows if k != "oracle"],
            "coverage": [{"scenario": s, "key": k, "n": 10, "cov68": (0.0 if k == "beamconv" else 0.7, 0.02),
                          "cov95": (0.95, 0.01)} for s, k in rows]}  # fmt: skip


def _selection():
    # K = 1 fails its fit check with the lowest BIC; the chosen K = 2 x 0 is the lowest that passes.
    table = [{"K": k, "beam_terms": b, "bic": 100.0 + 50 * k + 10 * b - 500 * (k == 1),
              "p": 1e-6 if k == 1 else 0.2 + 0.1 * b, "passes": k > 1}
             for k in (1, 2, 3) for b in (0, 1) if (k, b) != (3, 1)]  # fmt: skip
    grid = [{"amplitude_frac": fa, "global_amplitude_sigma": ga, "global_index_sigma": 0.0,
             "log_evidence": 10.0 * fa + ga, "bank_p": 0.5 if fa > 1 else 1e-5} for fa in (1.0, 3.0) for ga in (0.0, 10.0)]  # fmt: skip
    order = {"rule": "bic_among_passing", "table": table, "chosen": table[2], "moments": [1, 3], "beam_terms": [0, 1]}
    ablation = [{"label": "GSM: FA 1", "chi2_truth": 240.0, "chi2_bank": 244.0, "rank": 46, "chosen": False},
                {"label": "GSM: FA 3", "chi2_truth": 13.0, "chi2_bank": 14.6, "rank": 45, "chosen": True}]  # fmt: skip
    return {"order": {"main": order}, "prior": {"main": {"grid": grid, "best": grid[-1]}}, "ablation": ablation}


def _nuts():
    stats = {"r_hat": 1.01, "divergences": 0, "spread": 0.9, "high_nu_min": (0.4, 0.39)}
    return {"main": {"freqs": F, "truth": TRUTH, "smc": {m: _curves() for m in MODELS},
                     "nuts": {m: _curves() for m in MODELS}, "stats": {m: stats for m in MODELS}}}  # fmt: skip


FIGURES = {
    "scenario_overview": lambda path: figures.scenario_overview(_overview(), path),
    "signal_recovery": lambda path: figures.signal_recovery(
        {"main": {"freqs": F, "truth": TRUTH, "curves": {k: _curves() for k in KEYS}}}, path),
    "trough": lambda path: figures.trough(_trough(), path),
    "corner": lambda path: figures.corner({k: RNG.standard_normal((400, 3)) for k in KEYS}, np.zeros(3), path,
                                          labels=figures.U_LABELS[:3]),  # fmt: skip
    "residual_waterfalls": lambda path: figures.residual_waterfalls(
        [{"title": "main: data - model", "freqs": F, "panels": {"A": RNG.normal(0, 0.01, (8, 6)),
                                                                 "B": RNG.normal(0, 0.01, (8, 6))}}], path),  # fmt: skip
    "fom_summary": lambda path: figures.fom_summary(_summary(), path),
    "model_selection": lambda path: figures.model_selection(_selection(), path),
    "nuts_vs_smc": lambda path: figures.nuts_vs_smc(_nuts(), path),
}


def test_the_selection_fixture_obeys_the_rule():
    order = _selection()["order"]["main"]
    assert strategies.choose_order(order["table"], order["rule"]) == order["chosen"]
    assert min(order["table"], key=lambda r: r["bic"]) != order["chosen"]


@pytest.mark.parametrize("name", sorted(FIGURES))
def test_every_figure_writes_a_png(name, tmp_path):
    path = tmp_path / f"{name}.png"
    FIGURES[name](path)
    assert path.stat().st_size > 10_000
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
