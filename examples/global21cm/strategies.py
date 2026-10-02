"""The two foreground strategies and the oracle, as ``resources.arrays`` hooks.

Each document names its strategy here, with the data it reads as ``file:``
arguments (hashed into ``provenance.json``) and its choices as ``literal:``:

* :func:`known_foreground`: the oracle, the true foreground subtracted;
* :func:`per_lst_moments`: demo A, every LST's spectrum free on a moment
  basis, the order chosen by a named rule (:data:`ORDER_RULES`);
* :func:`gaussian_sky`: demo B, NSIDE 16 sky maps with GSM2008-centred
  priors through the drift scan.

Each returns a :class:`~global21cm.collapse.Collapsed`: the foreground
integrated out exactly. :func:`design_checked` hands its design to the
21 cm operator and refuses unless the document's observed file is this
strategy's statistic of this waterfall, and :func:`start_point` starts the
chains at the prior-bank draw of highest posterior (this strategy's
compressed likelihood times the unit-normal prior). ``prepare.py`` calls
the same functions with the same arguments, read from the documents.

A candidate model passes the fit check when its chi^2 has
``p >= FIT_CHECK_P``; demo A's order rule and demo B's prior selection
(``prepare.py``) both use it.
"""

from __future__ import annotations

import functools

import numpy as np
from scipy import stats

from global21cm import collapse, physical_inputs, signal21
from global21cm.foreground_beamconv import moment_basis, orthonormal_span

#: Prior draws searched for each candidate's chi^2 minimum and the chains' start.
N_BANK = 40000
BANK_SEED = 7
#: The fit check: a candidate passes when its chi^2 has p >= FIT_CHECK_P.
FIT_CHECK_P = 0.01
#: Demo A's order rules. ``bic``: the lowest BIC over the grid.
#: ``bic_among_passing``: the lowest BIC among the candidates that pass the
#: fit check, or over the whole grid when none passes (recorded ``passes: false``).
ORDER_RULES = ("bic", "bic_among_passing")


@functools.lru_cache(maxsize=4)
def prior_bank(freqs: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray]:
    """``(u, curves)``: ``N_BANK`` prior draws and their curves on ``freqs``."""
    u = np.random.default_rng(BANK_SEED).standard_normal((N_BANK, len(signal21.THETA_NAMES)))
    low, high = (np.asarray(v) for v in signal21.prior_box())
    theta = np.asarray(signal21.box_from_unit_normal(u, low, high))
    curves = np.concatenate([np.asarray(signal21.curve_kelvin(theta[i : i + 5000], np.asarray(freqs)))
                             for i in range(0, N_BANK, 5000)])  # fmt: skip
    return u, curves


def _freqs(freqs_mhz) -> tuple[float, ...]:
    return tuple(float(v) for v in np.asarray(freqs_mhz))


def known_foreground(waterfall, fg_truth) -> collapse.Collapsed:
    """The oracle: ``d - fg_truth`` is signal plus noise."""
    return collapse.oracle(np.asarray(waterfall), np.asarray(fg_truth))


def _moment_rows(waterfall, freqs, spectra, moments, beam_terms, beta0, nu_ref, sigma) -> list[dict]:
    """Every (K, beam terms): chi^2 minimised over the prior bank, its p-value, and BIC.

    chi^2 is the whole waterfall's at noise ``sigma``, with ``n_time n_basis``
    coefficients and the 7 signal parameters fitted, so its p-value is taken
    on ``n_data - n_time n_basis - 7`` degrees of freedom and the BIC penalty
    counts the same ``n_time n_basis + 7`` parameters.
    """
    n_time, n_freq = waterfall.shape
    _, curves = prior_bank(_freqs(freqs))
    rows = []
    for n_beam in range(beam_terms[0], beam_terms[1] + 1):
        for k in range(moments[0], moments[1] + 1):
            n_basis = k * max(n_beam, 1)
            if n_basis > n_freq - 3:
                continue
            q = orthonormal_span(moment_basis(freqs, beta0, k, nu_ref, spectra[:, :n_beam] if n_beam else None))
            perp = np.eye(n_freq) - q @ q.T
            chi2 = (np.sum((waterfall @ perp) ** 2) - 2 * curves @ (perp @ waterfall.sum(axis=0))
                    + n_time * np.einsum("nf,fg,ng->n", curves, perp, curves))  # fmt: skip
            best = float(chi2.min()) / sigma**2
            n_par = n_time * n_basis + len(signal21.THETA_NAMES)
            dof = waterfall.size - n_par
            p = float(stats.chi2.sf(best, dof))
            rows.append({"K": k, "beam_terms": n_beam, "n_basis": n_basis, "chi2_min": best, "dof": dof, "p": p,
                         "passes": p >= FIT_CHECK_P, "chi2_per_datum": best / waterfall.size,
                         "bic": best + n_par * np.log(waterfall.size)})  # fmt: skip
    return rows


def choose_order(rows: list[dict], rule: str) -> dict:
    """The row of ``rows`` (:func:`_moment_rows`) that ``rule`` picks; see :data:`ORDER_RULES`."""
    if rule not in ORDER_RULES:
        raise ValueError(f"order rule {rule!r}: expected one of {ORDER_RULES} or [K, beam_terms].")
    passing = [r for r in rows if r["passes"]] if rule == "bic_among_passing" else []
    return min(passing or rows, key=lambda r: r["bic"])


def model_order(waterfall, beam_spectra, freqs_mhz, rule, moments, beam_terms, beta0, nu_ref_mhz,
                sigma: float = collapse.SIGMA0) -> dict:  # fmt: skip
    """Demo A's order table over ``moments`` x ``beam_terms`` at noise ``sigma``, and ``rule``'s choice."""
    if int(moments[0]) < 1:
        raise ValueError(f"moments {moments!r}: the basis needs at least one moment.")
    waterfall, freqs, spectra = (np.asarray(v, dtype=np.float64) for v in (waterfall, freqs_mhz, beam_spectra))
    rows = _moment_rows(waterfall, freqs, spectra, moments, beam_terms, float(beta0), float(nu_ref_mhz), sigma)
    return {"rule": rule, "fit_check_p": FIT_CHECK_P, "table": rows, "chosen": choose_order(rows, rule),
            "n_passing": sum(r["passes"] for r in rows)}  # fmt: skip


def per_lst_moments(waterfall, beam_spectra, freqs_mhz, order, moments, beam_terms,
                    beta0, nu_ref_mhz) -> collapse.Collapsed:  # fmt: skip
    """Demo A. ``order`` is a rule of :data:`ORDER_RULES` (searching ``moments`` x ``beam_terms``) or ``[K, n_beam]``."""
    waterfall, freqs, spectra = (np.asarray(v, dtype=np.float64) for v in (waterfall, freqs_mhz, beam_spectra))
    if isinstance(order, str):
        record = model_order(waterfall, spectra, freqs, order, moments, beam_terms, beta0, nu_ref_mhz)
        k, n_beam = record["chosen"]["K"], record["chosen"]["beam_terms"]
    else:
        k, n_beam = (int(v) for v in order)
        record = {"table": [], "chosen": {"K": k, "beam_terms": n_beam}}
    basis = moment_basis(freqs, float(beta0), k, float(nu_ref_mhz), spectra[:, :n_beam] if n_beam else None)
    item = collapse.per_lst_basis(waterfall, orthonormal_span(basis))
    item.info.update({"model_order": record})
    return item


def gaussian_sky(waterfall, beam_alm, freqs_mhz, nside, lmax, template, amplitude_frac, index_sigma,
                 curvature_sigma, global_amplitude_sigma, global_index_sigma,
                 relinearise) -> collapse.Collapsed:  # fmt: skip
    """Demo B. ``template`` must be ``"gsm2008"``; widths as in :class:`physical_inputs.Widths`."""
    if template != "gsm2008":
        raise ValueError(f"template {template!r}: only 'gsm2008' is implemented.")
    waterfall, freqs = np.asarray(waterfall, dtype=np.float64), np.asarray(freqs_mhz, dtype=np.float64)
    gsm = physical_inputs.gsm_template(int(nside))
    response = physical_inputs.response(np.asarray(beam_alm), int(nside), int(lmax))
    widths = physical_inputs.Widths(float(amplitude_frac), float(index_sigma), float(curvature_sigma),
                                    float(global_amplitude_sigma), float(global_index_sigma))  # fmt: skip
    model = physical_inputs.linear_model(response, waterfall, freqs, gsm, widths, int(relinearise))
    item = collapse.from_marginal(waterfall, model["marginal"])
    item.info.update({"gsm_curvature": gsm[2], "c_t": model["c_t"], "relinearisation": model["steps"],
                      "log_evidence_no_signal": physical_inputs.log_evidence(model, waterfall)})  # fmt: skip
    return item


def bank_fit(collapsed: collapse.Collapsed, freqs_mhz) -> dict:
    """A truth-free, sampler-free goodness of fit: the compressed chi^2 minimised over the bank."""
    _, curves = prior_bank(_freqs(freqs_mhz))
    chi2 = float(collapsed.chi2(curves).min())
    dof = collapsed.rank - len(signal21.THETA_NAMES)
    return {"chi2_min": chi2, "dof": dof, "p": float(stats.chi2.sf(chi2, dof)) if dof > 0 else None}


def design_checked(collapsed: collapse.Collapsed, observed) -> np.ndarray:
    """The design, after checking the observed file is this strategy's statistic."""
    observed = np.asarray(observed, dtype=np.float64).reshape(-1)
    scale = max(float(np.abs(collapsed.data).max()), 1e-300)
    if observed.shape != collapsed.data.shape or np.abs(observed - collapsed.data).max() > 1e-9 * scale:
        raise ValueError(
            "inference.observed is not the statistic this document's strategy makes of this "
            "waterfall; re-run `python -m global21cm.prepare` after changing either."
        )
    return collapsed.design


def start_point(collapsed: collapse.Collapsed, freqs_mhz) -> np.ndarray:
    """The prior-bank draw of highest posterior under ``collapsed``, ``(7,)``.

    The 21 cm posterior is multimodal; chains started at the box centre were
    measured to settle in a mode 270 log-units below the truth's on the main
    oracle. The bank is a global search at one matrix-vector product per draw.
    """
    u, curves = prior_bank(_freqs(freqs_mhz))
    log_post = -0.5 * collapsed.chi2(curves) - 0.5 * np.sum(u**2, axis=1)
    return u[int(np.argmax(log_post))]


def component(start, index) -> float:
    """``start[index]``, spelled with keywords for the ``python:`` hatch."""
    return float(np.asarray(start)[int(index)])
