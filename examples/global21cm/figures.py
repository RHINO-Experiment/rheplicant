"""Every figure of the example, as pure functions: arrays in, one PNG out.

Nothing here reads a file; :mod:`global21cm.plots` loads the products and
builds the inputs. One fixed categorical order names the posteriors in every
figure: prior grey, oracle aqua, demo A blue, demo B orange, the truth black.
Magnitudes use one-hue ramps (violet for the sky data, blue for demo A's grid,
orange for demo B's), and signed residuals a blue-grey-red diverging map. Figures are
drawn 9.5 in wide at 200 dpi, so they stay legible when a README shows them
at about 900 px.
"""

from __future__ import annotations

import functools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402
from scipy.ndimage import gaussian_filter, gaussian_filter1d  # noqa: E402

COLOURS = {"prior": "#8c8b87", "oracle": "#1baf7a", "beamconv": "#2a78d6", "physical": "#eb6834"}
LABELS = {
    "prior": "prior only",
    "oracle": "oracle (foreground known)",
    "beamconv": "A: beam-convolved",
    "physical": "B: physical",
}
SHORT = {"prior": "prior", "oracle": "oracle", "beamconv": "A", "physical": "B"}
INK, INK2, MUTED, GRID, AXIS, WASH = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#f1f0ec"
FOREGROUND = "#4a3aa7"  # violet: the sky, a colour no posterior uses
#: The data waterfall's ramp, ending in FOREGROUND so the sky keeps one hue in every figure.
VIOLETS = ("#ecebf6", "#cfcaec", "#aaa1dc", "#8174c8", "#5f50b5", "#4a3aa7", "#342777", "#221a4f")
BLUES = ("#f3f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#104281", "#0d366b")
ORANGES = ("#fef3ee", "#fbd9c9", "#f7b597", "#f28e62", "#eb6834", "#c9501f", "#9e3c14", "#6f290c")
#: The four LST spectra of the overview; they stop at a mid-grey so none is the truth's INK.
LST_GREYS = ("#c3c2b7", "#a09f98", "#77766f", "#52514e")
DIVERGING = LinearSegmentedColormap.from_list("blue_grey_red", ["#184f95", "#6da7ec", "#f0efec", "#ec8a89", "#a3201f"])
PARAM_LABELS = (
    r"$\log_{10} f_*$", r"$\log_{10} V_c$", r"$\log_{10} f_X$", r"$\tau$",
    r"$\alpha$", r"$\nu_{\min}$", r"$R_{\rm mfp}$",
)  # fmt: skip
#: The corner plot's axes: unit-normal latents u = Phi^-1((theta - low) / (high - low)).
U_LABELS = tuple(f"u of {p}" for p in PARAM_LABELS)
BANDS = (2.5, 16.0, 50.0, 84.0, 97.5)
ALPHA95, ALPHA68 = 0.16, 0.38
#: The NUTS band outlines, dashed so they stay apart from the solid truth line.
NUTS_DASH = (0, (4, 2))
WIDTH, DPI = 9.5, 200
RC = {
    "font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "legend.frameon": False, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
    "xtick.major.size": 3, "ytick.major.size": 3, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "hatch.linewidth": 0.6, "axes.titlelocation": "left",
}  # fmt: skip


def _figure(func):
    """Draw ``func`` under the shared style."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with plt.rc_context(RC):
            return func(*args, **kwargs)

    return wrapper


def _save(fig, path: Path) -> None:
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def _style(ax) -> None:
    """Hairline grid behind the data, no top or right spine."""
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _sci(value: float) -> str:
    """``23400`` -> ``2.3\\times10^{4}`` for mathtext."""
    exponent = int(np.floor(np.log10(abs(value))))
    return rf"{value / 10**exponent:.1f}\times10^{{{exponent}}}"


def _compact(value: float) -> str:
    if value < 10:
        return f"{value:.2g}"
    if value < 1000:
        return f"{value:.0f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def _ink_on(rgb) -> str:
    """White text on a dark fill, ink on a light one."""
    r, g, b = to_rgb(rgb)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.45 else INK


def _log_ticks(ax, axis: str, ticks) -> None:
    """Plain-number major ticks on a log axis, no minor labels."""
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(ticks))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())


# --------------------------------------------------------- contour levels --
def mass_levels(density, masses, weights=None) -> np.ndarray:
    """Density thresholds whose super-level sets hold the given fractions of the mass.

    Cells are ranked by ``density``; the mass of a cell is ``weights`` (the
    density itself by default). For each ``p`` in ``masses`` the level is the
    largest ``t`` such that the cells with ``density >= t`` hold at least ``p``
    of the total mass: the highest-density region of mass ``p``. Passing the
    raw histogram counts as ``weights`` for a smoothed ``density`` makes the
    region hold ``p`` of the samples, whatever the smoothing. Returned in the
    order of ``masses``.
    """
    masses = np.asarray(masses, dtype=np.float64)
    if np.any((masses <= 0.0) | (masses > 1.0)):
        raise ValueError(f"masses must lie in (0, 1], got {masses}.")
    density = np.ravel(np.asarray(density, dtype=np.float64))
    weights = density if weights is None else np.ravel(np.asarray(weights, dtype=np.float64))
    if weights.shape != density.shape or density.size == 0:
        raise ValueError(f"density {density.shape} and weights {weights.shape} must be equal and non-empty.")
    if np.any(density < 0.0) or np.any(weights < 0.0) or not weights.sum() > 0.0:
        raise ValueError("density and weights must be non-negative, with a positive total weight.")
    order = np.argsort(density, kind="stable")[::-1]
    cumulative = np.cumsum(weights[order]) / weights.sum()
    return density[order][np.minimum(np.searchsorted(cumulative, masses), density.size - 1)]


def density_2d(x, y, extent, bins: int = 40, smooth: float = 1.2) -> tuple[np.ndarray, ...]:
    """``(x centres, y centres, smoothed density[y, x], counts[y, x])`` of a 2-D histogram.

    The density is the histogram smoothed by a Gaussian of ``smooth`` bins;
    the counts are the raw histogram, the weights :func:`mass_levels` needs
    for regions that hold a given fraction of the samples.
    """
    hist, xe, ye = np.histogram2d(x, y, bins=bins, range=extent)
    return 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), gaussian_filter(hist.T, smooth), hist.T


def _extent(values, span: float) -> tuple[float, float]:
    """A grid range holding all but 0.2 % of ``values``, padded, inside ``[-span, span]``."""
    lo, hi = np.percentile(values, [0.1, 99.9])
    pad = 0.3 * (hi - lo) + 0.02
    return max(lo - pad, -span), min(hi + pad, span)


# --------------------------------------------------------------- overview --
def _waterfall_panel(ax, s, norm, title):
    freqs, half = s["freqs"], 0.5 * (s["freqs"][1] - s["freqs"][0])
    image = ax.imshow(s["waterfall"], aspect="auto", norm=norm, interpolation="nearest",
                      cmap=LinearSegmentedColormap.from_list("violets", VIOLETS),
                      extent=(freqs[0] - half, freqs[-1] + half, 360.0, 0.0))  # fmt: skip
    for colour, row in zip(LST_GREYS, _lst_rows(s), strict=True):
        ax.plot(freqs[-1] + half, s["lst"][row], marker="<", ms=7, color=colour, mec="white", clip_on=False)
    ax.set(title=title, xlabel="frequency [MHz]", ylabel="LST [deg]")
    return image


def _lst_rows(s) -> list[int]:
    n = len(s["lst"])
    return [0, n // 4, n // 2, 3 * n // 4]


def _spectra_panel(ax, s):
    for colour, row in zip(LST_GREYS, _lst_rows(s), strict=True):
        ax.loglog(s["freqs"], s["waterfall"][row], color=colour, lw=1.6, label=f"LST {s['lst'][row]:.0f} deg")
    lo, hi = s["band"]
    _log_ticks(ax, "x", sorted({lo, hi} | {f for f in (60, 70, 80, 100) if lo + 3 < f < hi - 3}))
    ax.set(title="spectra at four LSTs", xlabel="frequency [MHz]", ylabel="T [K]")
    ax.legend(loc="lower left")
    _style(ax)


def _scales_panel(ax, s):
    freqs, fg, signal = s["freqs"], s["foreground"], np.abs(s["signal"])
    rms = s["noise"].std(axis=0)
    ax.fill_between(freqs, fg.min(axis=0), fg.max(axis=0), color=FOREGROUND, alpha=0.18, lw=0)
    ax.semilogy(freqs, np.median(fg, axis=0), color=FOREGROUND, lw=1.6, label="foreground (median, LST range)")
    ax.semilogy(freqs, signal, color=INK, lw=1.6, label="|21 cm truth|")
    ax.semilogy(freqs, rms, color=MUTED, lw=1.2, label="noise rms, one sample")
    ax.semilogy(freqs, rms / np.sqrt(fg.shape[0]), color=AXIS, lw=1.2, label=f"noise rms, mean of {fg.shape[0]} LSTs")
    at = int(np.argmin(s["signal"]))
    ratio = np.median(fg, axis=0)[at] / signal[at]
    ax.annotate(rf"foreground / signal $\approx {_sci(ratio)}$" + f"\nat {freqs[at]:.0f} MHz",
                (freqs[at], signal[at]), xytext=(0.5, 0.5), textcoords="axes fraction", fontsize=8,
                color=INK2, ha="center", arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})  # fmt: skip
    ax.set_ylim(1e-4, 5e4)
    ax.set(title="temperature scales", xlabel="frequency [MHz]", ylabel="|T| [K]")
    _style(ax)


@_figure
def scenario_overview(scenarios: dict, path: Path) -> None:
    """The simulated data of each scenario: waterfall, LST spectra, and the three temperature scales.

    ``scenarios`` maps a name to ``freqs`` (F,) MHz, ``lst`` (T,) deg, ``band``,
    ``beam``, and in K: ``waterfall``, ``foreground`` and ``noise`` (T, F) and
    ``signal`` (F,).
    """
    names = list(scenarios)
    fig, axes = plt.subplots(len(names), 3, figsize=(WIDTH, 3.1 * len(names)), layout="constrained",
                             squeeze=False, gridspec_kw={"width_ratios": (1.0, 1.0, 1.1)})  # fmt: skip
    fig.get_layout_engine().set(wspace=0.07)
    norm = LogNorm(min(float(s["waterfall"].min()) for s in scenarios.values()),
                   max(float(s["waterfall"].max()) for s in scenarios.values()))  # fmt: skip
    beams = {"gaussian": "Gaussian beam", "hornwet": "HornWet beams"}
    for r, name in enumerate(names):
        s = scenarios[name]
        image = _waterfall_panel(axes[r, 0], s, norm, f"{name}: data, {beams.get(s['beam'], s['beam'])}")
        fig.colorbar(image, ax=axes[r, 0], label="T [K]", fraction=0.08, pad=0.06)
        _spectra_panel(axes[r, 1], s)
        _scales_panel(axes[r, 2], s)
    fig.legend(*axes[0, 2].get_legend_handles_labels(), loc="outside lower right", ncol=4)
    _save(fig, path)


# -------------------------------------------------------- signal recovery --
def _bands(ax, x, draws, colour) -> None:
    """95 and 68 % bands and the median of ``draws`` (S, F) along ``x``."""
    lo95, lo68, mid, hi68, hi95 = np.percentile(draws, BANDS, axis=0)
    ax.fill_between(x, lo95, hi95, color=colour, alpha=ALPHA95, lw=0)
    ax.fill_between(x, lo68, hi68, color=colour, alpha=ALPHA68, lw=0)
    ax.plot(x, mid, color=colour, lw=1.6)


def _symmetric_limit(*draws) -> float:
    return 1.1 * max(float(np.max(np.abs(np.percentile(d, [2.5, 97.5], axis=0)))) for d in draws) + 1e-3


def _band_legend(fig) -> None:
    handles = [Patch(color=MUTED, alpha=ALPHA95, lw=0), Patch(color=MUTED, alpha=ALPHA68, lw=0),
               Line2D([], [], color=MUTED, lw=1.6), Line2D([], [], color=INK, lw=1.2)]  # fmt: skip
    fig.legend(handles, ["95 % band", "68 % band", "median", "truth"], loc="outside upper center", ncol=4)


@_figure
def signal_recovery(scenarios: dict, path: Path) -> None:
    """Posterior bands of the curve and of (curve - truth), two rows per scenario.

    ``scenarios`` maps a name to ``freqs`` (F,), ``truth`` (F,) K and
    ``curves`` ``{posterior: (S, F) K}``.
    """
    names, keys = list(scenarios), list(next(iter(scenarios.values()))["curves"])
    fig, axes = plt.subplots(2 * len(names), len(keys), figsize=(WIDTH, 2.05 * 2 * len(names)),
                             layout="constrained", squeeze=False)  # fmt: skip
    for r, name in enumerate(names):
        s = scenarios[name]
        freqs, truth = s["freqs"], s["truth"] * 1e3
        top, bottom = axes[2 * r], axes[2 * r + 1]
        low = min(float(np.percentile(s["curves"][k] * 1e3, 2.5, axis=0).min()) for k in keys)
        high = max(float(np.percentile(s["curves"][k] * 1e3, 97.5, axis=0).max()) for k in keys)
        for c, key in enumerate(keys):
            draws = s["curves"][key] * 1e3
            _bands(top[c], freqs, draws, COLOURS[key])
            top[c].plot(freqs, truth, color=INK, lw=1.2)
            top[c].set_ylim(low - 10, max(high, truth.max()) + 10)
            _bands(bottom[c], freqs, draws - truth[None, :], COLOURS[key])
            bottom[c].axhline(0.0, color=INK, lw=1.2)
            limit = _symmetric_limit(draws - truth[None, :])
            bottom[c].set_ylim(-limit, limit)
            bottom[c].set_xlabel("frequency [MHz]")
            top[c].set_title(f"{name}: {LABELS[key]}", fontsize=8.5)
            for ax in (top[c], bottom[c]):
                ax.set_xlim(freqs[0], freqs[-1])
                _style(ax)
            top[c].tick_params(labelbottom=False)
            if c:
                top[c].set_yticklabels([])
        top[0].set_ylabel(r"$T_{21}$ [mK]")
        bottom[0].set_ylabel("curve - truth [mK]")
    _band_legend(fig)
    _save(fig, path)


# ----------------------------------------------------------------- trough --
def _density_line(ax, values, edges, key, smooth_bins: float) -> None:
    density, _ = np.histogram(values, bins=edges, density=True)
    if smooth_bins > 0:
        density = gaussian_filter1d(density, smooth_bins, mode="reflect")
    if key == "prior":
        ax.stairs(density, edges, fill=True, color=COLOURS[key], alpha=0.3, lw=0)
    else:
        ax.stairs(density, edges, color=COLOURS[key], lw=1.4)


def _weak_note(ax, values: dict, weak: float, right: float) -> None:
    ax.axvspan(weak, right, color=WASH, lw=0, zorder=0)
    lines = ["weak signal", f"(> {weak:.0f} mK):"]
    lines += [f"{SHORT[k]} {100 * np.mean(v > weak):.1f} %" for k, v in values.items()]
    ax.text(weak + 3, 0.97, "\n".join(lines), transform=ax.get_xaxis_transform(), va="top", fontsize=7.5,
            color=INK2)  # fmt: skip


def _trough_panel(ax, values: dict, edges, truth, unit: str, smooth_bins: float = 0.0) -> None:
    for key, v in values.items():
        _density_line(ax, v, edges, key, smooth_bins)
    ax.axvline(truth, color=INK, lw=1.2)
    ax.annotate(f"truth {truth:.1f} {unit}", (truth, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-3, -3), textcoords="offset points", va="top", ha="right", fontsize=7.5)  # fmt: skip
    ax.set_yscale("log")
    ax.set_xlim(edges[0], edges[-1])
    _style(ax)


@_figure
def trough(scenarios: dict, path: Path, weak_mk: float = -50.0) -> None:
    """Posterior densities of the trough depth and frequency, on a log density axis so thin tails show.

    ``scenarios`` maps a name to ``truth`` ``(depth mK, frequency MHz)``,
    ``depth`` and ``where`` ``{posterior: (S,)}`` in mK and MHz, ``band``,
    ``step`` (the grid the trough was read on, MHz) and ``smooth_mhz``, the
    Gaussian width the frequency histogram is smoothed with.
    """
    names = list(scenarios)
    fig, axes = plt.subplots(len(names), 2, figsize=(WIDTH, 3.0 * len(names)), layout="constrained",
                             squeeze=False, gridspec_kw={"width_ratios": (1.25, 1.0)})  # fmt: skip
    for r, name in enumerate(names):
        s = scenarios[name]
        low = min(float(np.percentile(v, 0.05)) for v in s["depth"].values())
        edges = np.arange(np.floor(low / 5) * 5 - 5, 2.0 + 1e-9, 2.0)
        _trough_panel(axes[r, 0], s["depth"], edges, s["truth"][0], "mK")
        _weak_note(axes[r, 0], s["depth"], weak_mk, edges[-1])
        (lo, hi), step = s["band"], s["step"]
        edges = np.arange(lo - step / 2, hi + step, step)
        _trough_panel(axes[r, 1], s["where"], edges, s["truth"][1], "MHz", s["smooth_mhz"] / step)
        axes[r, 0].set(xlabel="trough depth [mK]", ylabel="density [1/mK]", title=f"{name}: trough depth")
        axes[r, 1].set(xlabel="trough frequency [MHz]", ylabel="density [1/MHz]",
                       title=f"{name}: trough frequency (smoothed, sigma {s['smooth_mhz']:.2f} MHz)")  # fmt: skip
        top = max(ax.get_ylim()[1] for ax in axes[r])
        for ax in axes[r]:
            ax.set_ylim(3e-6, top * 3)
    keys = list(scenarios[names[0]]["depth"])
    handles = [Patch(color=COLOURS[k], alpha=0.3, lw=0) if k == "prior" else Line2D([], [], color=COLOURS[k], lw=1.4)
               for k in keys] + [Line2D([], [], color=INK, lw=1.2)]  # fmt: skip
    fig.legend(handles, [LABELS[k] for k in keys] + ["truth"], loc="outside upper center", ncol=len(keys) + 1)
    _save(fig, path)


# ----------------------------------------------------------------- corner --
def _diagonal(ax, draws: dict, i: int, truth, span: float) -> None:
    for key, u in draws.items():
        lo, hi = (-span, span) if key == "prior" else _extent(u[:, i], span)
        hist, edges = np.histogram(u[:, i], bins=60, range=(lo, hi))
        density = gaussian_filter1d(hist.astype(float), 2.5 if key == "prior" else 1.2)
        centres = 0.5 * (edges[1:] + edges[:-1])
        if key == "prior":
            ax.fill_between(centres, density / density.max(), color=WASH, lw=0)
            ax.plot(centres, density / density.max(), color=AXIS, lw=0.8)
        else:
            ax.plot(centres, density / density.max(), color=COLOURS[key], lw=1.4)
    ax.axvline(truth[i], color=INK, lw=1.0)
    ax.set_ylim(0, 1.08)
    ax.set_yticks([])


def _pair(ax, draws: dict, i: int, j: int, truth, span: float) -> None:
    for key, u in draws.items():
        extent = [(-span, span)] * 2 if key == "prior" else [_extent(u[:, j], span), _extent(u[:, i], span)]
        xc, yc, density, counts = density_2d(u[:, j], u[:, i], extent, *((30, 1.8) if key == "prior" else (40, 1.2)))
        l68, l95 = mass_levels(density, (0.68, 0.95), counts)
        if key == "prior":
            ax.contourf(xc, yc, density, levels=[l95, l68, density.max() * 1.01], colors=["#f4f3f0", "#e6e5e0"])
        else:
            ax.contour(xc, yc, density, levels=[l95, l68], colors=COLOURS[key], linewidths=[0.8, 1.5], zorder=3)
    ax.plot(truth[j], truth[i], marker="+", ms=8, mew=1.3, color=INK, zorder=2)


@_figure
def corner(draws: dict, truth, path: Path, labels=U_LABELS, span: float = 3.5) -> None:
    """Smoothed 68 / 95 % regions of every parameter pair, 1-D densities on the diagonal.

    ``draws`` maps a posterior to ``(S, d)`` unit-normal latents; ``"prior"``, if
    present, is drawn as a light filled region. ``truth`` is ``(d,)``.
    """
    n = len(truth)
    fig, axes = plt.subplots(n, n, figsize=(WIDTH, WIDTH), squeeze=False)
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.07, top=0.99, wspace=0.07, hspace=0.07)
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            if j > i:
                ax.axis("off")
                continue
            _diagonal(ax, draws, i, truth, span) if i == j else _pair(ax, draws, i, j, truth, span)
            ax.set_xlim(-span, span)
            ax.set_xticks([-2, 0, 2])
            if i != j:
                ax.set_ylim(-span, span)
                ax.set_yticks([-2, 0, 2])
            ax.tick_params(labelsize=7, labelbottom=i == n - 1, labelleft=j == 0 and i > 0)
            ax.set_xlabel(labels[j] if i == n - 1 else "", fontsize=8.5)
            ax.set_ylabel(labels[i] if j == 0 and i > 0 else "", fontsize=8.5)
    handles = [Patch(color="#e6e5e0")] + [Line2D([], [], color=COLOURS[k], lw=1.5) for k in draws if k != "prior"]
    handles.append(Line2D([], [], color=INK, marker="+", ls="", ms=8, mew=1.3))
    names = [LABELS["prior"]] + [LABELS[k] for k in draws if k != "prior"] + ["truth"]
    fig.legend(handles, names, loc="upper right", bbox_to_anchor=(0.98, 0.97), fontsize=9)
    fig.text(0.98, 0.72, "Main scenario. 68 % (thick) and 95 % (thin)\n"
             "regions, smoothed histograms of the SMC particles.\n"
             "Axes: unit-normal latents,\nu = Phi^-1((theta - min) / (max - min));\n"
             "the prior is N(0, 1) in each.", ha="right", va="top", fontsize=8, color=INK2)  # fmt: skip
    _save(fig, path)


# ---------------------------------------------------- residual waterfalls --
@_figure
def residual_waterfalls(rows: list, path: Path) -> None:
    """Rows of ``(n_time, n_freq)`` residual maps in K, drawn in mK; one symmetric colour scale per row.

    Each row is ``{"title", "freqs", "panels": {label: array}}``.
    """
    n_col = len(rows[0]["panels"])
    fig, axes = plt.subplots(len(rows), n_col, figsize=(WIDTH, 2.3 * len(rows)), layout="constrained", squeeze=False)
    for r, row in enumerate(rows):
        panels = {k: np.asarray(v) * 1e3 for k, v in row["panels"].items()}
        limit = max(float(np.percentile(np.abs(p), 99)) for p in panels.values())
        freqs = row["freqs"]
        half = 0.5 * (freqs[1] - freqs[0])
        for c, (title, panel) in enumerate(panels.items()):
            ax = axes[r, c]
            image = ax.imshow(panel, aspect="auto", cmap=DIVERGING, vmin=-limit, vmax=limit, interpolation="nearest",
                              extent=(freqs[0] - half, freqs[-1] + half, 360.0, 0.0))  # fmt: skip
            ax.text(0.98, 0.04, f"rms {np.sqrt(np.mean(panel**2)):.1f} mK", transform=ax.transAxes, ha="right",
                    fontsize=7.5, bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": "none",
                                        "alpha": 0.85})  # fmt: skip
            ax.set_title(title if r == 0 else "", loc="center")
            ax.set_ylabel(f"{row['title']}\nLST [deg]" if c == 0 else "")
            ax.set_xlabel("frequency [MHz]" if r == len(rows) - 1 else "")
            ax.set_yticks([0, 90, 180, 270, 360])
        fig.colorbar(image, ax=axes[r, :], label="mK", fraction=0.04, pad=0.01)
    _save(fig, path)


# ------------------------------------------------------------ FoM summary --
def _rows_y(rows) -> np.ndarray:
    """Top-to-bottom positions, with a gap between scenarios."""
    y, out, last = 0.0, [], None
    for row in rows:
        y += 0.0 if last is None else (1.0 if row["scenario"] == last else 1.6)
        out.append(y)
        last = row["scenario"]
    return np.asarray(out)


def _category_axis(ax, rows, ys, with_n: bool = False) -> None:
    labels = [f"{r['scenario']} · {SHORT[r['key']]}" + (f" (n={r['n']})" if with_n else "") for r in rows]
    ax.set_yticks(ys, labels)
    ax.set_ylim(ys[-1] + 0.7, ys[0] - 0.7)
    ax.grid(axis="y", visible=False)


def _ser_panel(ax, rows) -> None:
    ys = _rows_y(rows)
    for y, row in zip(ys, rows, strict=True):
        colour = COLOURS[row["key"]]
        ax.plot([1.0, row["value"]], [y, y], color=colour, lw=1.2, alpha=0.5)
        ax.plot(row["value"], y, "o", ms=8, mec=colour, mew=1.6, mfc=colour if row["calibrated"] else "white")
        ax.annotate(f"{row['value']:.3g}", (row["value"], y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK2)  # fmt: skip
    ax.axvline(1.0, color=INK, lw=1.0)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 1e3)
    _log_ticks(ax, "x", [0.3, 1, 3, 10, 30, 100, 300, 1000])
    ax.set(title="SER / SER(prior)  (prior = 1)", xlabel="SER relative to the prior")
    handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=MUTED, mec=MUTED),
               Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec=MUTED, mew=1.4)]  # fmt: skip
    ax.legend(handles, ["calibrated", "not calibrated"], loc="lower right")
    _category_axis(ax, rows, ys)


def _eta_panel(ax, rows) -> None:
    ys, markers = _rows_y(rows), ("o", "s", "D")
    for y, row in zip(ys, rows, strict=True):
        for k, (name, value) in enumerate(row["values"].items()):
            if value is None:
                ax.text(1.1e-3, y + 0.3 * (k - 1), f"{name}: withheld, not calibrated", fontsize=7, color=INK2,
                        va="center")  # fmt: skip
                continue
            ax.plot(value, y + 0.3 * (k - 1), markers[k], ms=6.5, color=COLOURS[row["key"]], mec="white", mew=0.8)
            ax.annotate(f"{value:.3g}", (value, y + 0.3 * (k - 1)), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=7, color=INK2)  # fmt: skip
    names = list(rows[0]["values"])
    ax.legend([Line2D([], [], marker=m, ls="", ms=6.5, color=MUTED) for m in markers], names, loc="lower right")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 0.3)
    _log_ticks(ax, "x", [0.001, 0.003, 0.01, 0.03, 0.1, 0.3])
    ax.set(title="efficiency against the oracle",
           xlabel=r"$\eta = \sqrt{\mathrm{tr\,Cov_{oracle}} / \mathrm{tr\,Cov}}$")
    _category_axis(ax, rows, ys)


def _coverage_panel(ax, rows, key: str, nominal: float) -> None:
    ys = _rows_y(rows)
    shown = [r[key][0] for r in rows if r[key][0] > 0]
    lo = max(0.0, min(m - 3 * se for m, se in (r[key] for r in rows if r[key][0] > 0)) - 0.02)
    hi = min(1.0, max(max(shown), nominal) + 0.04)
    for y, row in zip(ys, rows, strict=True):
        mean, se = row[key]
        if mean < lo:
            ax.annotate(f"{mean:.2f} (off scale)", (lo, y), xytext=(4, 0), textcoords="offset points",
                        va="center", fontsize=7.5, color=INK2)  # fmt: skip
            continue
        ax.errorbar(mean, y, xerr=se, fmt="o", ms=6.5, color=COLOURS[row["key"]], elinewidth=1.6, capsize=0)
    ax.axvline(nominal, color=INK, lw=1.0)
    ax.annotate(f"nominal {nominal:g}", (nominal, 1.0), xycoords=("data", "axes fraction"), xytext=(3, -2),
                textcoords="offset points", va="top", fontsize=7.5)  # fmt: skip
    ax.set_xlim(lo, hi)
    ax.set(title=f"{100 * nominal:.0f} % band coverage over noise realisations", xlabel="mean coverage, ±1 SE")
    _category_axis(ax, rows, ys, with_n=True)


@_figure
def fom_summary(summary: dict, path: Path) -> None:
    """SER / prior, eta and realisation coverage, from ``fom.json`` alone.

    ``summary`` holds lists ``ser`` (scenario, key, value, calibrated), ``eta``
    (scenario, key, ``values`` by estimator) and ``coverage`` (scenario, key, n,
    ``cov68`` and ``cov95`` as ``(mean, se)``).
    """
    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, 6.6), layout="constrained")
    _ser_panel(axes[0, 0], summary["ser"])
    _eta_panel(axes[0, 1], summary["eta"])
    _coverage_panel(axes[1, 0], summary["coverage"], "cov68", 0.68)
    _coverage_panel(axes[1, 1], summary["coverage"], "cov95", 0.95)
    for ax in axes.flat:
        _style(ax)
        ax.grid(axis="y", visible=False)
    _save(fig, path)


# -------------------------------------------------------- model selection --
def _cell_grid(ax, values, texts, cmap, norm, xlabels, ylabels, failed=None) -> None:
    """A heatmap of ``values[y, x]`` (nan = absent) with ``texts[y][x]`` in each cell; ``failed`` cells hatched."""
    ax.imshow(np.ma.masked_invalid(values), cmap=cmap, norm=norm, aspect="auto", origin="lower")
    for (y, x), value in np.ndenumerate(values):
        ink = INK2 if np.isnan(value) else _ink_on(cmap(norm(value)))
        if np.isnan(value) or (failed is not None and failed[y, x]):
            hatch, edge = ("xx", AXIS) if np.isnan(value) else ("///", "white" if ink == "white" else MUTED)
            ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, lw=0, hatch=hatch, edgecolor=edge))
        box = None if np.isnan(value) else {"boxstyle": "square,pad=0.15", "fc": cmap(norm(value)), "ec": "none"}
        ax.text(x, y, "not\nfitted" if np.isnan(value) else texts[y][x], ha="center", va="center", fontsize=7,
                color=ink, bbox=box)  # fmt: skip
    ax.set_xticks(range(len(xlabels)), xlabels)
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def _mark(ax, x, y) -> None:
    ax.add_patch(Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, edgecolor=INK, lw=2.2))


def _order_grid(ax, order, name, norm, cmap, pass_p: float) -> None:
    ks = range(order["moments"][0], order["moments"][1] + 1)
    bs = range(order["beam_terms"][0], order["beam_terms"][1] + 1)
    cells = {(r["K"], r["beam_terms"]): r for r in order["table"]}
    chosen = order["chosen"]
    low = min(r["bic"] for r in order["table"])
    values = np.array([[max(cells[k, b]["bic"] - low, norm.vmin) if (k, b) in cells else np.nan for k in ks]
                       for b in bs])  # fmt: skip
    texts = [[_bic_text(cells[k, b], chosen) if (k, b) in cells else "" for k in ks] for b in bs]
    failed = np.array([[(k, b) in cells and cells[k, b]["p"] < pass_p for k in ks] for b in bs])
    _cell_grid(ax, values, texts, cmap, norm, [str(k) for k in ks], [str(b) for b in bs], failed)
    _mark(ax, list(ks).index(chosen["K"]), list(bs).index(chosen["beam_terms"]))
    verdict = "" if chosen["passes"] else ": no candidate passes, lowest BIC"
    ax.set(xlabel="K, moments per LST", ylabel="beam spectra", title=f"A, {name}{verdict}")


def _bic_text(row, chosen) -> str:
    """``BIC - BIC(chosen)`` and the fit-check p of one order-grid cell."""
    if row == chosen:
        head = "chosen"
    else:
        delta = row["bic"] - chosen["bic"]
        head = ("+" if delta >= 0 else "-") + _compact(abs(delta))
    return head + "\n" + _p_text(row["p"])


def _p_text(p: float) -> str:
    return f"p {p:.2g}" if p >= 1e-3 else "p<0.001"


def _structure(row) -> tuple[float, float]:
    return row["global_amplitude_sigma"], row["global_index_sigma"]


def _structure_label(ga: float, gb: float) -> str:
    parts = [f"scale {ga:g}"] * bool(ga) + [f"index {gb:g}"] * bool(gb)
    return "global " + " + ".join(parts) if parts else "no global"


def _prior_grid(ax, prior, name, norm, cmap, pass_p: float) -> None:
    grid, best = prior["grid"], prior["best"]
    fas = sorted({r["amplitude_frac"] for r in grid})
    structures = list(dict.fromkeys(_structure(r) for r in grid))
    cells = {(r["amplitude_frac"], _structure(r)): r for r in grid}
    top = max(r["log_evidence"] for r in grid)
    values = np.array([[max(top - cells[fa, s]["log_evidence"], norm.vmin) for fa in fas] for s in structures])
    texts = [[_evidence_text(cells[fa, s], best) for fa in fas] for s in structures]
    failed = np.array([[cells[fa, s]["bank_p"] < pass_p for fa in fas] for s in structures])
    _cell_grid(ax, values, texts, cmap, norm, [f"{fa:g}" for fa in fas], [_structure_label(*s) for s in structures],
               failed)  # fmt: skip
    _mark(ax, fas.index(best["amplitude_frac"]), structures.index(_structure(best)))
    ax.set(xlabel="FA, per-pixel amplitude width (fraction of the template)", title=f"B, {name}")


def _evidence_text(row, best) -> str:
    """``ln Z - ln Z(chosen)`` and the fit-check p of one prior-grid cell."""
    if row is best or row == best:
        head = "chosen"
    else:
        delta = row["log_evidence"] - best["log_evidence"]
        head = f"{delta:+.1f}" if abs(delta) < 10 else f"{delta:+.0f}"
    return head + "\n" + _p_text(row["bank_p"])


def _ablation_panel(ax, rows) -> None:
    ys = np.arange(len(rows))
    colour = COLOURS["physical"]
    for y, row in zip(ys, rows, strict=True):
        ax.plot([row["chi2_truth"], row["chi2_bank"]], [y, y], color=colour, lw=1.0, alpha=0.5)
        ax.plot(row["rank"], y, "|", ms=12, mew=1.6, color=INK2)
        ax.plot(row["chi2_truth"], y, "o", ms=5.5, color=colour)
        ax.plot(row["chi2_bank"], y, "o", ms=9, mfc="none", mec=colour, mew=1.4)
    ax.set_yticks(ys, [r["label"] for r in rows])
    for tick, row in zip(ax.get_yticklabels(), rows):
        tick.set_fontweight("bold" if row["chosen"] else "normal")
    ax.set_ylim(len(rows) - 0.4, -0.6)
    ax.set_xscale("log")
    ax.set(xlabel="compressed chi2 on the noiseless waterfall (main)",
           title="B's misfit, one prior change at a time (ablation.json); bold: the shipped prior")  # fmt: skip
    handles = [Line2D([], [], marker="o", ls="", ms=5.5, color=colour),
               Line2D([], [], marker="o", ls="", ms=9, mfc="none", mec=colour, mew=1.4),
               Line2D([], [], marker="|", ls="", ms=12, mew=1.6, color=INK2)]  # fmt: skip
    ax.legend(handles, ["at the true curve", "minimised over the prior bank", "rank (mean chi2 of noise alone)"],
              loc="upper left", fontsize=7.5)  # fmt: skip
    _style(ax)
    ax.grid(axis="y", visible=False)


@_figure
def model_selection(selection: dict, path: Path, pass_p: float = 0.01) -> None:
    """Demo A's order grid, demo B's prior-structure grid, and B's ablation rows.

    ``selection`` holds ``order`` and ``prior`` ``{scenario: fom.json block}``
    (``order`` blocks carry ``moments`` and ``beam_terms`` ranges) and
    ``ablation`` rows (label, chi2_truth, chi2_bank, rank, chosen). Cells
    whose fit check has ``p < pass_p`` are hatched in both grids.
    """
    fig = plt.figure(figsize=(WIDTH, 10.6), layout="constrained")
    top, middle, bottom = fig.subfigures(3, 1, height_ratios=(1.0, 0.9, 1.05))
    # Both grids are darker where the candidate is worse: higher BIC, lower evidence.
    blues = LinearSegmentedColormap.from_list("b", BLUES)
    oranges = LinearSegmentedColormap.from_list("o", ORANGES)
    spans = [r["bic"] - min(q["bic"] for q in o["table"]) for o in selection["order"].values() for r in o["table"]]
    bic_norm = LogNorm(1e2, max(1e3, *spans))
    tops = {n: max(r["log_evidence"] for r in p["grid"]) for n, p in selection["prior"].items()}
    gaps = [tops[n] - r["log_evidence"] for n, p in selection["prior"].items() for r in p["grid"]]
    z_norm = LogNorm(1.0, max(10.0, *gaps))
    top.suptitle("Demo A's basis order: cell text BIC - BIC(chosen) and the whole-waterfall chi2 p; "
                 f"hatched: p < {pass_p:g}, fails the fit check", x=0.01, ha="left", fontsize=9.5)  # fmt: skip
    axes = top.subplots(1, len(selection["order"]), squeeze=False)[0]
    for ax, (name, order) in zip(axes, selection["order"].items()):
        _order_grid(ax, order, name, bic_norm, blues, pass_p)
    top.colorbar(plt.cm.ScalarMappable(bic_norm, blues), ax=axes, label="BIC - BIC(lowest)", fraction=0.04)
    middle.suptitle("Demo B's prior structure: cell text ln Z - ln Z(chosen) and the fit-check p; "
                    f"hatched: p < {pass_p:g}, fails the fit check", x=0.01, ha="left", fontsize=9.5)  # fmt: skip
    axes = middle.subplots(1, len(selection["prior"]), squeeze=False)[0]
    for ax, (name, prior) in zip(axes, selection["prior"].items()):
        _prior_grid(ax, prior, name, z_norm, oranges, pass_p)
    middle.colorbar(plt.cm.ScalarMappable(z_norm, oranges), ax=axes, label="ln Z(max) - ln Z", fraction=0.04)
    _ablation_panel(bottom.subplots(), selection["ablation"])
    _save(fig, path)


# ------------------------------------------------------------ NUTS vs SMC --
def _nuts_panel(ax, freqs, truth, smc, nuts, stats, key) -> None:
    _bands(ax, freqs, smc - truth[None, :], COLOURS[key])
    lo95, lo68, _, hi68, hi95 = np.percentile(nuts - truth[None, :], BANDS, axis=0)
    for curve, width in ((lo68, 1.3), (hi68, 1.3), (lo95, 0.7), (hi95, 0.7)):
        ax.plot(freqs, curve, color=INK2, lw=width, ls=NUTS_DASH)
    ax.axhline(0.0, color=INK, lw=1.0)
    limit = _symmetric_limit(smc - truth[None, :], nuts - truth[None, :])
    ax.set_ylim(-limit, limit)
    ax.set_xlim(freqs[0], freqs[-1])
    text = (f"R-hat {stats['r_hat']:.3f}, {stats['divergences']} div., spread {stats['spread']:.3g}\n"
            f"u(nu_min) > 0: NUTS {stats['high_nu_min'][0]:.2f}, SMC {stats['high_nu_min'][1]:.3f}")  # fmt: skip
    ax.text(0.0, 1.02, text, transform=ax.transAxes, va="bottom", fontsize=7, color=INK2).set_in_layout(False)
    _style(ax)


@_figure
def nuts_vs_smc(scenarios: dict, path: Path, runs: str = "default runs") -> None:
    """The documents' NUTS bands (outlines) against the headline SMC bands (fills), as curve - truth.

    ``scenarios`` maps a name to ``freqs``, ``truth`` (F,) K, ``smc`` and ``nuts``
    ``{model: (S, F) K}`` and ``stats`` ``{model: {r_hat, divergences, spread, high_nu_min}}``.
    """
    names = list(scenarios)
    keys = list(scenarios[names[0]]["smc"])
    fig, axes = plt.subplots(len(names), len(keys), figsize=(WIDTH, 2.9 * len(names)), layout="constrained",
                             squeeze=False)  # fmt: skip
    for r, name in enumerate(names):
        s = scenarios[name]
        truth = s["truth"] * 1e3
        for c, key in enumerate(keys):
            _nuts_panel(axes[r, c], s["freqs"], truth, s["smc"][key] * 1e3, s["nuts"][key] * 1e3, s["stats"][key], key)
            axes[r, c].set_title(f"{name}: {LABELS[key]}", fontsize=8.5, pad=24)
            axes[r, c].set_xlabel("frequency [MHz]" if r == len(names) - 1 else "")
        axes[r, 0].set_ylabel("curve - truth [mK]")
    handles = [Patch(color=MUTED, alpha=ALPHA95, lw=0), Patch(color=MUTED, alpha=ALPHA68, lw=0),
               Line2D([], [], color=MUTED, lw=1.6),
               Line2D([], [], color=INK2, lw=0.7, ls=NUTS_DASH),
               Line2D([], [], color=INK2, lw=1.3, ls=NUTS_DASH),
               Line2D([], [], color=INK, lw=1.0)]  # fmt: skip
    names = ["SMC 95 %", "SMC 68 %", "SMC median", f"NUTS 95 % ({runs})", "NUTS 68 %", "truth"]
    fig.legend(handles, names, loc="outside upper center", ncol=6)
    _save(fig, path)
