"""How each strategy's amplitude forecast moves with the noise level, sampler-free.

Three questions per scenario (main, stress) and model (oracle, demo A, demo B),
each strategy read from its document the way ``prepare.py`` reads it:

1. **Radiometer equivalence.** The simulation adds white noise of
   ``scenario.NOISE_SIGMA_K`` per sample, the same everywhere. A radiometer
   whose system temperature is the noiseless sky (receiver temperature
   neglected) has ``sigma_1day(nu, t) = T_sky(nu, t) / sqrt(dnu tau)`` in one
   LST bin of ``tau = scenario.TIME_STEP_S`` per sidereal day, and reaches
   ``sigma`` after ``(sigma_1day / sigma)^2`` days.
2. **sigma(amp) and bias(amp) against the noise.** ``collapse`` fixes the
   noise at ``SIGMA0`` for the documents. A flat prior (oracle, demo A) has
   no other scale, so sigma(amp) scales as ``sigma / sigma0`` and the bias
   does not move. Demo B's Gaussian prior does have one: its forecast at
   each ``sigma`` is ``collapse.amplitude_forecast`` of the documents' own
   compression (``collapse.from_marginal``) with the prior's
   :class:`~global21cm.collapse.GaussianMarginal` moved to that noise
   (``at(sigma)``), which never forms ``J S J^T``. Structure is held fixed:
   demo B's widths, template and linearisation point, all as chosen at the
   documents' noise. Demo A's order is also re-chosen at each level by the
   documents' rule (``strategies.model_order`` at that noise), on the
   simulated noise realisation scaled to that level.
3. **The a-priori predictor** ``||P_perp t|| / ||t||``
   (``fom.perpendicular_fraction``) for each strategy's foreground columns.

The JSON also records demo B's forecast at ``SIGMA0`` for random column
orders of its Jacobian (the result must not depend on them) and the
compression's forecast against the direct ``t^T C^-1 t`` at every noise
level of the table.

Run from the repository root after prepare.py (writes
``results/analysis/sensitivity.json`` and ``sensitivity.png``; measured at
271 s, 191 s main and 80 s stress, with a machine load average of 4 to 6):

    PYTHONPATH=examples .venv/bin/python -m global21cm.sensitivity
"""

from __future__ import annotations

import functools
import argparse
import json
import time
from collections.abc import Callable

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
from scipy import optimize  # noqa: E402

from global21cm import collapse, fom, physical_inputs, prepare, scenario, strategies  # noqa: E402

SIGMA0 = collapse.SIGMA0
#: The noise levels of the tables, K.
NOISE_GRID_K = (3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
#: The dense grid for crossings, K; the figure shows it up to PLOT_MAX_K.
CURVE_GRID_K = np.logspace(-4.0, 1.0, 201)
PLOT_MAX_K = 1.0
#: Detection threshold on 1 / sigma(amp).
DETECTION = 5.0
REFERENCE_MHZ = 70.0
#: Random column orderings of demo B's Jacobian tried at SIGMA0.
N_ORDERINGS, ORDERING_SEED = 5, 0
#: Relative cut-offs at which demo B's raw Jacobian span is measured.
RCOND_SCAN = (1e-6, 1e-8, 1e-10, 1e-12, 1e-14)
Forecast = Callable[[float], tuple[float, float]]


# ------------------------------------------------------------ radiometer --
def radiometer_sigma(t_sys, dnu_hz: float, tau_s: float) -> np.ndarray:
    """One day's radiometer noise ``T_sys / sqrt(dnu tau)``, K."""
    return np.asarray(t_sys, dtype=np.float64) / np.sqrt(dnu_hz * tau_s)


def days_to_reach(sigma_1day, sigma: float) -> np.ndarray:
    """Days of integration at which ``sigma_1day / sqrt(days)`` equals ``sigma``."""
    return (np.asarray(sigma_1day, dtype=np.float64) / sigma) ** 2


def radiometer(noiseless, freqs_mhz, sigma: float = SIGMA0) -> dict:
    """Item 1 for one scenario: the days at which the radiometer noise is ``sigma``."""
    freqs = np.asarray(freqs_mhz, dtype=np.float64)
    step = np.diff(freqs)
    if not np.allclose(step, step[0]):
        raise ValueError("the channel grid is not uniform; dnu is not one number.")
    one_day = radiometer_sigma(noiseless, step[0] * 1e6, scenario.TIME_STEP_S)  # (n_time, n_freq)
    median = np.median(one_day, axis=0)
    days, every = days_to_reach(median, sigma), days_to_reach(one_day, sigma)
    ref = int(np.argmin(np.abs(freqs - REFERENCE_MHZ)))
    return {
        "t_sys": "noiseless sky waterfall; receiver temperature neglected",
        "channel_width_mhz": float(step[0]),
        "tau_s": scenario.TIME_STEP_S,
        "noise_k": sigma,
        "reference_channel_mhz": float(freqs[ref]),
        "sigma_1day_reference_k": float(median[ref]),
        "days_reference": {
            "median_over_lst": float(days[ref]),
            "min_over_lst": float(every[:, ref].min()),
            "max_over_lst": float(every[:, ref].max()),
        },
        "days_band": {
            "min": float(days.min()),
            "min_at_mhz": float(freqs[np.argmin(days)]),
            "max": float(days.max()),
            "max_at_mhz": float(freqs[np.argmax(days)]),
        },
        "days_waterfall": {"min": float(every.min()), "max": float(every.max())},
        "sigma_band_factor": float(median.max() / median.min()),
        "sigma_lst_factor_reference": float(one_day[:, ref].max() / one_day[:, ref].min()),
        "freqs_mhz": freqs.tolist(),
        "sigma_1day_median_k": median.tolist(),
        "sigma_1day_min_k": one_day.min(axis=0).tolist(),
        "sigma_1day_max_k": one_day.max(axis=0).tolist(),
    }


# --------------------------------------------------------------- forecasts --
def flat_forecast(item: collapse.Collapsed, shape, noiseless) -> Forecast:
    """sigma(amp), bias at any noise for a strategy whose prior is flat (or absent)."""
    base = collapse.amplitude_forecast(item, shape, noiseless)
    return lambda sigma: (base["sigma"] * sigma / SIGMA0, base["bias"])


def gaussian_forecast(marginal: collapse.GaussianMarginal, waterfall, shape, noiseless) -> Forecast:
    """sigma(amp), bias at any noise for demo B: its prior and linearisation fixed, the noise moved.

    The documents' compression (``collapse.from_marginal``) of ``waterfall``
    with ``marginal`` at each noise level; memoised, since the crossings
    revisit the curve's grid.
    """

    @functools.cache
    def forecast(sigma: float) -> tuple[float, float]:
        item = collapse.from_marginal(waterfall, marginal.at(sigma))
        base = collapse.amplitude_forecast(item, shape, noiseless)
        return base["sigma"], base["bias"]

    return forecast


def direct_forecast(marginal: collapse.GaussianMarginal, tiled, residual) -> tuple[float, float]:
    """``(t^T C^-1 t)^-1/2`` and ``t^T C^-1 r / t^T C^-1 t - 1`` at ``marginal``'s noise.

    The same forecast without the compression's eigenvalue cut (``collapse.RCOND``).
    """
    cinv_t = marginal.solve(tiled)
    info = float(tiled @ cinv_t)
    return 1.0 / np.sqrt(info), float(residual @ cinv_t) / info - 1.0


# ------------------------------------------------------------------ sweeps --
def row(
    sigma: float, forecast: tuple[float, float], sigma_1day_ref: float, oracle_sigma: float
) -> dict:
    """One noise level: forecast, significance, misfit ratio, days at 70 MHz, soft P_perp fraction.

    ``effective_perp_fraction`` is sigma(amp) of the oracle over this one's:
    ``||P_perp t|| / ||t||`` exactly for a flat prior, its noise-weighted
    analogue for a Gaussian one.
    """
    sigma_amp, bias = forecast
    return {
        "noise_k": sigma,
        "days_70mhz": float(days_to_reach(sigma_1day_ref, sigma)),
        "sigma_amp": float(sigma_amp),
        "bias": float(bias),
        "detection": float(1.0 / sigma_amp),
        "abs_bias_over_sigma": float(abs(bias) / sigma_amp),
        "effective_perp_fraction": float(oracle_sigma / sigma_amp),
    }


def crossings(
    forecast: Forecast, metric: Callable[[float, float], float], grid=CURVE_GRID_K
) -> list[float]:
    """Noise levels in ``grid``'s range where ``metric(sigma_amp, bias)`` crosses 1.

    Sign changes on the grid, each refined by brentq in log noise. The grid
    itself is evaluated at its own values, which a memoised forecast has seen.
    """

    def log_metric(x: float) -> float:
        return float(np.log(metric(*forecast(float(np.exp(x))))))

    x = np.log(np.asarray(grid))
    y = np.array([float(np.log(metric(*forecast(float(s))))) for s in np.asarray(grid)])
    idx = np.nonzero(np.sign(y[:-1]) * np.sign(y[1:]) < 0)[0]
    return [float(np.exp(optimize.brentq(log_metric, x[i], x[i + 1], xtol=1e-12))) for i in idx]


def summarise(forecast: Forecast, sigma_1day_ref: float, oracle: Forecast) -> dict:
    """The table rows, the dense curve and both crossings of one strategy."""
    rows = [row(s, forecast(s), sigma_1day_ref, oracle(s)[0]) for s in NOISE_GRID_K]
    curve = np.array([forecast(float(s)) for s in CURVE_GRID_K])
    marks = {
        "bias_equals_sigma": crossings(forecast, lambda sa, b: max(abs(b), 1e-300) / sa),
        "detection_5": crossings(forecast, lambda sa, b: DETECTION * sa),
    }
    at = lambda s: {"noise_k": s, "days_70mhz": float(days_to_reach(sigma_1day_ref, s))}  # noqa: E731
    return {
        "sweep": rows,
        "crossings": {key: [at(s) for s in found] for key, found in marks.items()},
        "curve": {"sigma_amp": curve[:, 0].tolist(), "abs_bias": np.abs(curve[:, 1]).tolist()},
    }


def order_sweep(
    arrays: dict, literal: dict, sim: dict, sigma_1day_ref: float, oracle: Forecast
) -> list[dict]:
    """Demo A's order re-chosen at each noise level by the documents' rule.

    The simulated realisation is rescaled to each level, and
    ``strategies.model_order`` takes chi^2 at that noise, so its p-values,
    fit check and BIC are the ones the documents' rule would see there.
    """
    noise = sim["waterfall"] - sim["noiseless"]
    keys = ("beam_spectra", "freqs_mhz")
    knobs = ("order", "moments", "beam_terms", "beta0", "nu_ref_mhz")
    out = []
    for sigma in NOISE_GRID_K:
        waterfall = sim["noiseless"] + (sigma / SIGMA0) * noise
        record = strategies.model_order(
            waterfall, *(arrays[k] for k in keys), *(literal[k] for k in knobs), sigma=sigma
        )
        best = record["chosen"]
        args = {**arrays, "waterfall": waterfall}
        item = strategies.per_lst_moments(
            **args, **{**literal, "order": [best["K"], best["beam_terms"]]}
        )
        forecast = flat_forecast(item, sim["t21_truth"], sim["noiseless"])(sigma)
        out.append(
            {
                **row(sigma, forecast, sigma_1day_ref, oracle(sigma)[0]),
                "K": best["K"],
                "beam_terms": best["beam_terms"],
                "n_basis": best["n_basis"],
                "p": best["p"],
                "passes": best["passes"],
                "n_passing": record["n_passing"],
                "perp_fraction": spectral_perp(item, sim["t21_truth"]),
            }
        )
    return out


# ---------------------------------------------------------- P_perp (item 3) --
def per_lst_basis(item: collapse.Collapsed, n_time: int) -> tuple[np.ndarray, bool]:
    """Demo A's orthonormal basis, read from the strategy's own output, and whether it
    is the same at every LST (``fit_signal`` is ``kron(P_perp, 1_time)``)."""
    blocks = item.fit_signal.reshape(-1, n_time, item.fit_signal.shape[1])
    same = bool(np.array_equal(blocks, np.repeat(blocks[:, :1, :], n_time, axis=1)))
    evals, evecs = np.linalg.eigh(np.eye(blocks.shape[0]) - blocks[:, 0, :])
    return evecs[:, evals > 0.5], same


def spectral_perp(item: collapse.Collapsed, shape) -> float:
    basis, _ = per_lst_basis(item, item.centred.size // np.size(shape))
    return fom.perpendicular_fraction(basis, shape)


def beamconv_perp(item: collapse.Collapsed, shape, n_time: int) -> dict:
    """The fraction of one spectrum and of the tiled waterfall; equal when the basis is shared."""
    basis, same = per_lst_basis(item, n_time)
    tiled = np.repeat(np.asarray(shape), n_time)
    return {
        "n_basis": int(basis.shape[1]),
        "same_basis_every_lst": same,
        "spectral": fom.perpendicular_fraction(basis, shape),
        "data_space": fom.perpendicular_fraction(np.kron(basis, np.eye(n_time)), tiled),
    }


def physical_perp(jac, tiled, n_time: int) -> dict:
    """Demo B's raw Jacobian span against the data space.

    ``lst_nyquist_leakage`` is the norm of ``J`` along each channel's
    alternating-LST pattern, relative to ``||J||``: the lmax-47 drift scan
    cannot make that pattern, and the time-constant signal has none of it.
    """
    u, s, _ = np.linalg.svd(jac, full_matrices=False)
    scan = {}
    for rcond in RCOND_SCAN:
        keep = u[:, s > rcond * s.max()]
        rest = np.linalg.norm(tiled - keep @ (keep.T @ tiled)) / np.linalg.norm(tiled)
        scan[f"{rcond:g}"] = {"rank": int(keep.shape[1]), "fraction": float(rest)}
    pattern = ((-1.0) ** np.arange(n_time))[:, None] / np.sqrt(n_time)
    nyquist = np.kron(np.eye(jac.shape[0] // n_time), pattern)
    return {
        "n_data": int(jac.shape[0]),
        "n_columns": int(jac.shape[1]),
        "singular_max": float(s.max()),
        "singular_min": float(s.min()),
        "fom_default_rcond": fom.perpendicular_fraction(jac, tiled),
        "rcond_scan": scan,
        "lst_nyquist_leakage": float(np.linalg.norm(nyquist.T @ jac) / np.linalg.norm(jac)),
    }


# ------------------------------------------------------------- per scenario --
def physical_model(arrays: dict, literal: dict) -> dict:
    """``strategies.gaussian_sky``'s linear model, kept: the hook returns only the collapse.

    The caller checks the rebuilt collapse against the document's observed
    file (``strategies.design_checked``), so a drift from the hook refuses.
    ``model["marginal"]`` is the prior's factorisation at ``SIGMA0``.
    """
    if literal["template"] != "gsm2008":
        raise ValueError(f"template {literal['template']!r}: only 'gsm2008' is implemented.")
    nside, lmax = int(literal["nside"]), int(literal["lmax"])
    keys = (
        "amplitude_frac",
        "index_sigma",
        "curvature_sigma",
        "global_amplitude_sigma",
        "global_index_sigma",
    )
    widths = physical_inputs.Widths(*(float(literal[k]) for k in keys))
    response = physical_inputs.response(np.asarray(arrays["beam_alm"]), nside, lmax)
    waterfall = np.asarray(arrays["waterfall"], dtype=np.float64)
    freqs = np.asarray(arrays["freqs_mhz"], dtype=np.float64)
    template = physical_inputs.gsm_template(nside)
    return physical_inputs.linear_model(
        response, waterfall, freqs, template, widths, int(literal["relinearise"])
    )


def checks(sim: dict, model: dict, forecast: Forecast, tiled, residual) -> dict:
    """Demo B's compressed forecast against the direct one, and on shuffled column orders."""
    marginal = model["marginal"]
    grid = []
    for sigma in NOISE_GRID_K:
        compressed, direct = forecast(sigma), direct_forecast(marginal.at(sigma), tiled, residual)
        grid.append(
            {
                "noise_k": sigma,
                "compressed": list(compressed),
                "direct": list(direct),
                "sigma_amp_rel_diff": compressed[0] / direct[0] - 1.0,
                "bias_abs_diff": abs(compressed[1] - direct[1]),
            }
        )
    rng, shuffled = np.random.default_rng(ORDERING_SEED), []
    for _ in range(N_ORDERINGS):
        order = rng.permutation(model["jac"].shape[1])
        parts = (np.asarray(model[k])[..., order] for k in ("jac", "loc", "scale"))
        item = collapse.gaussian_prior(sim["waterfall"], *parts)
        base = collapse.amplitude_forecast(item, sim["t21_truth"], sim["noiseless"])
        shuffled.append([base["sigma"], base["bias"]])
    ref = forecast(SIGMA0)
    return {
        "compressed_vs_direct": grid,
        "shuffled_columns_at_sigma0": {
            "n": N_ORDERINGS,
            "results": shuffled,
            "sigma_amp_max_rel_diff": max(abs(v[0] / ref[0] - 1.0) for v in shuffled),
            "bias_max_abs_diff": max(abs(v[1] - ref[1]) for v in shuffled),
        },
    }


def checked(model: str, case: scenario.Scenario, item: collapse.Collapsed) -> collapse.Collapsed:
    """``item``, after checking it is the statistic the document observes."""
    strategies.design_checked(item, np.load(case.sim_dir / f"{model}_data.npy"))
    return item


def scenario_record(case: scenario.Scenario) -> dict:
    start = time.perf_counter()
    names = ("waterfall", "noiseless", "t21_truth", "freqs_mhz")
    sim = {n: np.load(case.sim_dir / f"{n}.npy") for n in names}
    n_time, shape = sim["waterfall"].shape[0], sim["t21_truth"]
    tiled = np.repeat(shape, n_time)
    radio = radiometer(sim["noiseless"], sim["freqs_mhz"])
    ref = radio["sigma_1day_reference_k"]
    specs = {m: prepare.document_spec(m, case) for m in prepare.MODELS}
    items = {
        m: checked(m, case, fn(**arrays, **lit))
        for m, (fn, arrays, lit) in specs.items()
        if m != "physical"
    }
    oracle = flat_forecast(items["oracle"], shape, sim["noiseless"])
    _, arrays_b, literal_b = specs["physical"]
    model = physical_model(arrays_b, literal_b)
    checked("physical", case, collapse.from_marginal(sim["waterfall"], model["marginal"]))
    residual = collapse.vec(sim["noiseless"]) - model["jac"] @ model["loc"]
    physical = gaussian_forecast(model["marginal"], sim["waterfall"], shape, sim["noiseless"])
    _, arrays_a, literal_a = specs["beamconv"]
    chosen = items["beamconv"].info["model_order"]["chosen"]
    orders = order_sweep(arrays_a, literal_a, sim, ref, oracle)
    at_sigma0 = next(r for r in orders if r["noise_k"] == SIGMA0)
    models = {
        "oracle": {
            "structure": "foreground known",
            **summarise(oracle, ref, oracle),
            "perp": {"fraction": fom.perpendicular_fraction(np.zeros((tiled.size, 0)), tiled)},
        },
        "beamconv": {
            "structure": {"K": chosen["K"], "beam_terms": chosen["beam_terms"],
                          "rule": literal_a["order"]},  # fmt: skip
            **summarise(flat_forecast(items["beamconv"], shape, sim["noiseless"]), ref, oracle),
            "order_sweep": orders,
            "order_at_sigma0_matches_document": (at_sigma0["K"], at_sigma0["beam_terms"])
            == (chosen["K"], chosen["beam_terms"]),
            "perp": beamconv_perp(items["beamconv"], shape, n_time),
        },
        "physical": {
            "structure": {k: literal_b[k] for k in sorted(literal_b)},
            "route": "collapse.GaussianMarginal: QR of (J S^1/2)^T, then tpqrt of [R0; sigma I]",
            **summarise(physical, ref, oracle),
            "checks": checks(sim, model, physical, tiled, residual),
            "perp": physical_perp(model["jac"], tiled, n_time),
        },
    }
    return {"radiometer": radio, "models": models, "seconds": round(time.perf_counter() - start, 1)}


# ------------------------------------------------------------------ figure --
INK, MUTED, GRID = "#1f1f1d", "#6b6a66", "#dddddd"
SCENARIO_INK = {"main": INK, "stress": "#8c8b87"}
TITLES = {"main": "main: 45-135 MHz, Gaussian beam", "stress": "stress: 55-85 MHz, HornWet beams"}
SIGMA_AMP = r"$\sigma(\mathrm{amp})$"


def _style(ax) -> None:
    ax.grid(color=GRID, linewidth=0.5, which="major")
    ax.tick_params(labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _days_axis(ax, sigma_1day_ref: float, channel_mhz: float) -> None:
    """A top axis reading the noise (mK) as days of integration at the reference channel."""

    def to_days(mk):
        return (sigma_1day_ref / (np.maximum(mk, 1e-12) * 1e-3)) ** 2

    def to_mk(days):
        return 1e3 * sigma_1day_ref / np.sqrt(np.maximum(days, 1e-12))

    top = ax.secondary_xaxis("top", functions=(to_days, to_mk))
    top.set_xlabel(
        f"days of integration for this noise at {channel_mhz:.0f} MHz", fontsize=9, color=MUTED
    )
    top.tick_params(labelsize=8, colors=MUTED)


def _sensitivity_panel(ax, name: str, entry: dict, colours: dict) -> None:
    keep = CURVE_GRID_K <= PLOT_MAX_K
    noise_mk = CURVE_GRID_K[keep] * 1e3
    for key in ("oracle", "beamconv", "physical"):
        curve = {k: np.asarray(v)[keep] for k, v in entry["models"][key]["curve"].items()}
        ax.plot(noise_mk, curve["sigma_amp"], color=colours[key], linewidth=2)
        if key != "oracle":  # the oracle's bias is float64 roundoff, ~1e-13
            ax.plot(
                noise_mk,
                curve["abs_bias"],
                color=colours[key],
                linewidth=1.5,
                linestyle=(0, (4, 2)),
            )
    orders = entry["models"]["beamconv"]["order_sweep"]
    noise = [1e3 * r["noise_k"] for r in orders]
    ax.plot(
        noise,
        [r["sigma_amp"] for r in orders],
        linestyle="none",
        marker="o",
        markersize=7,
        color=colours["beamconv"],
        markeredgecolor="white",
        markeredgewidth=1.5,
    )
    ax.plot(
        noise,
        [abs(r["bias"]) for r in orders],
        linestyle="none",
        marker="o",
        markersize=7,
        markerfacecolor="white",
        markeredgecolor=colours["beamconv"],
        markeredgewidth=1.5,
    )
    ax.axhline(1.0 / DETECTION, color=MUTED, linewidth=1)
    ax.annotate(
        "5 sigma",
        (1.0, 1.0 / DETECTION),
        xycoords=("axes fraction", "data"),
        xytext=(4, 0),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
        va="center",
    )
    ax.axvline(SIGMA0 * 1e3, color=MUTED, linewidth=1)
    ax.annotate(
        "documents' noise",
        (SIGMA0 * 1e3, 1.0),
        xycoords=("data", "axes fraction"),
        xytext=(5, -6),
        textcoords="offset points",
        fontsize=8,
        color=MUTED,
        rotation=90,
        ha="left",
        va="top",
    )
    ax.set(xscale="log", yscale="log", xlim=(noise_mk[0], noise_mk[-1]), ylim=(2e-5, 1e3))
    ax.set_xlabel("simulated noise per sample (mK)", fontsize=9)
    ax.set_ylabel(f"{SIGMA_AMP} (solid) and |bias(amp)| (dashed)", fontsize=9)
    ax.set_title(TITLES[name], fontsize=10, loc="left", pad=34)
    radio = entry["radiometer"]
    _days_axis(ax, radio["sigma_1day_reference_k"], radio["reference_channel_mhz"])
    _style(ax)


def _radiometer_panel(ax, record: dict) -> None:
    for name, entry in record.items():
        radio = entry["radiometer"]
        freqs = np.asarray(radio["freqs_mhz"])
        median, low, high = (
            np.asarray(radio[k]) * 1e3
            for k in ("sigma_1day_median_k", "sigma_1day_min_k", "sigma_1day_max_k")
        )
        colour, days = SCENARIO_INK[name], radio["days_reference"]["median_over_lst"]
        ax.fill_between(freqs, low, high, color=colour, alpha=0.12, linewidth=0)
        width = radio["channel_width_mhz"]
        ax.plot(
            freqs,
            median,
            color=colour,
            linewidth=2,
            label=f"{name}, {width:.0f} MHz channels: one day",
        )
        ax.plot(
            freqs,
            median / np.sqrt(days),
            color=colour,
            linewidth=1.5,
            linestyle=(0, (4, 2)),
            label=f"{name}: {days:.0f} days (10 mK at {radio['reference_channel_mhz']:.0f} MHz)",
        )
    ax.axhline(SIGMA0 * 1e3, color=MUTED, linewidth=1.5, label="simulated: 10 mK at every channel")
    ax.set(yscale="log")
    ax.set_xlabel("frequency (MHz)", fontsize=9)
    ax.set_ylabel("radiometer noise per LST bin (mK)", fontsize=9)
    ax.set_title(
        "radiometer noise, T_sys = noiseless sky\n(line: LST median; band: LST min to max)",
        fontsize=10,
        loc="left",
        pad=8,
    )
    _style(ax)


def figure(record: dict, path) -> None:
    """Both scenarios' sweeps (top), the radiometer noise and the legends (bottom).

    Drawn at ``figures.WIDTH`` and ``figures.DPI`` under ``figures.RC``, like
    every other figure of the example. ``record`` is ``sensitivity.json``'s
    ``scenarios`` block, so the figure can be redrawn from the JSON alone.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from global21cm.figures import COLOURS, DPI, LABELS, RC, WIDTH

    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 8.6), layout="constrained")
        for ax, name in zip(axes[0], ("main", "stress"), strict=True):
            _sensitivity_panel(ax, name, record[name], COLOURS)
        _radiometer_panel(axes[1, 0], record)
        _legend_panel(axes[1, 1], axes[1, 0], COLOURS, LABELS)
        fig.savefig(path, dpi=DPI)
        plt.close(fig)


def _legend_panel(ax, radiometer_ax, colours: dict, labels: dict) -> None:
    """The fourth cell: the sweep panels' legend above the radiometer panel's."""
    from matplotlib.lines import Line2D

    ax.axis("off")
    blue = colours["beamconv"]
    handles = [
        Line2D([], [], color=colours[k], linewidth=2, label=labels[k])
        for k in ("oracle", "beamconv", "physical")
    ]
    handles += [
        Line2D([], [], color=INK, linewidth=2, label=SIGMA_AMP),
        Line2D(
            [],
            [],
            color=INK,
            linewidth=1.5,
            linestyle=(0, (4, 2)),
            label="|bias(amp)| on noiseless data (A and B)",
        ),
        Line2D(
            [],
            [],
            color=blue,
            linestyle="none",
            marker="o",
            markeredgecolor="white",
            markersize=7,
            label=f"A, order re-chosen at that noise: {SIGMA_AMP}",
        ),
        Line2D(
            [],
            [],
            color=blue,
            linestyle="none",
            marker="o",
            markerfacecolor="white",
            markersize=7,
            markeredgewidth=1.5,
            label="A, order re-chosen: |bias|",
        ),
    ]
    top = ax.legend(
        handles=handles,
        loc="upper left",
        fontsize=8,
        frameon=False,
        title="top panels",
        title_fontsize=8.5,
        alignment="left",
    )
    ax.add_artist(top)
    ax.legend(
        *radiometer_ax.get_legend_handles_labels(),
        loc="lower left",
        fontsize=8,
        frameon=False,
        title="radiometer panel (left)",
        title_fontsize=8.5,
        alignment="left",
    )


def main() -> None:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args()
    start = time.perf_counter()
    record = {}
    for name in ("main", "stress"):
        record[name] = scenario_record(scenario.SCENARIOS[name])
        print(name, record[name]["seconds"], "s", flush=True)
    out = scenario.RESULTS / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    figure(record, out / "sensitivity.png")
    meta = {
        "noise_grid_k": list(NOISE_GRID_K),
        "curve_grid_k": CURVE_GRID_K.tolist(),
        "sigma0_k": SIGMA0,
        "detection": DETECTION,
        "reference_mhz": REFERENCE_MHZ,
        "seconds": round(time.perf_counter() - start, 1),
    }
    text = json.dumps({**meta, "scenarios": record}, indent=1)
    (out / "sensitivity.json").write_text(text + "\n")
    print("total", meta["seconds"], "s")


if __name__ == "__main__":
    main()
