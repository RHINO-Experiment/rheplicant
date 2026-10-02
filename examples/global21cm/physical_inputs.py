"""Demo B's linear foreground: GSM template, linearisation and Jacobian.

The template is independent of the truth: GSM2008 (``pygdsm``) at 45, 70,
135 and 408 MHz, CMB removed, rotated to equatorial and averaged to the fit
NSIDE. Per pixel the MERS law's index ``beta_g`` and curvature are solved
from the 45, 135 and 408 MHz maps; ``a_g`` is the 70 MHz map; ``c_g`` is the
median curvature over the sky. The priors are

    A ~ Normal(a_g, (FA a_g)^2) + s a_g,  s ~ Normal(0, GA^2),
    beta ~ Normal(beta_g, FB^2) (+ a global offset, width GB),  c ~ Normal(c_g, SC^2),

with ``beta`` and ``c`` entering to first order about a linearisation point
``(beta_t, c_t)``: the index through the map ``B = A_t (beta - beta_t)`` and
the curvature through one global column ``A_t E dQ``. The global scale ``s``
exists because GSM's 70 MHz amplitude is 1.78 times the truth's (median over
pixels; 1.41 to 2.23 over the central 68 %): independent per-pixel priors
around it bind the beam-averaged sky to a wrong value. The structure (FA, GA, GB) is the literal hand-written in
``physical.yaml``; ``prepare.py`` evaluates every structure in ``PRIOR_GRID``
and records whether the literal is the selection rule's choice
(``document_is_best``): the highest evidence among structures that pass the
fit check (``strategies.FIT_CHECK_P``).

The linearisation point starts at the GSM values and is moved ``relinearise``
times to the posterior mean of the fit with the 21 cm signal set to zero
(Gauss-Newton). The point, and the amplitude ``A_t`` inside the index prior's
width, therefore come from the data: this is empirical Bayes. The prior
centres stay on GSM throughout. Every posterior mean and the evidence go
through :class:`global21cm.collapse.GaussianMarginal` at the documents' noise.
"""

from __future__ import annotations

import functools
from typing import NamedTuple

import healpy as hp
import numpy as np

from global21cm import foreground_physical as phys
from global21cm import instrument, scenario
from global21cm.collapse import GaussianMarginal, vec
from global21cm.physical_operator import moment_jacobian

GSM_FREQS_MHZ = (45.0, 70.0, 135.0, 408.0)
T_CMB = 2.725
BETA_RANGE = (-3.8, -1.5)
#: Declared grid of prior structures (prepare.py records each): the per-pixel
#: amplitude width FA, and whether a free global amplitude scale (GA = 10)
#: and a free global index offset (GB = 1) are added. FB = 0.6 and SC = 0.05.
PRIOR_GRID = tuple(
    (fa, ga, gb)
    for ga, gb in ((0.0, 0.0), (10.0, 0.0), (10.0, 1.0))
    for fa in (0.1, 0.3, 1.0, 3.0, 10.0)
)


@functools.lru_cache(maxsize=2)
def gsm_template(nside: int) -> tuple[np.ndarray, np.ndarray, float]:
    """``(beta_g, a_g, c_g)`` from GSM2008 at ``nside``, equatorial."""
    from pygdsm import GlobalSkyModel

    maps = GlobalSkyModel(freq_unit="MHz").generate(list(GSM_FREQS_MHZ))
    rotator = hp.Rotator(coord=["G", "C"])
    t45, t70, t135, t408 = (hp.ud_grade(rotator.rotate_map_pixel(m), nside) - T_CMB for m in maps)
    x = lambda nu: np.log(nu / scenario.NU_HASLAM_MHZ)  # noqa: E731
    q = lambda nu: x(nu) * np.log(nu / scenario.NU_CURVATURE_MHZ)  # noqa: E731
    system = np.array([[x(45.0), q(45.0)], [x(135.0), q(135.0)]])
    beta, curvature = np.linalg.solve(system, np.stack([np.log(t45 / t408), np.log(t135 / t408)]))
    return beta, t70, float(np.median(curvature))


def spectra_at(beta, curvature, freqs) -> np.ndarray:
    """``E(p, nu)``, the MERS law at ``(beta, c)`` relative to 70 MHz."""
    log_ratio, q_shift = (np.asarray(v) for v in phys.spectral_design(freqs))
    return np.exp(beta[None, :] * log_ratio[:, None] + curvature * q_shift[:, None])


class Widths(NamedTuple):
    """The prior's widths; a global width of 0 leaves that column out."""

    amplitude: float  # FA, per pixel, fraction of the template
    index: float  # FB, per pixel
    curvature: float  # SC, global
    global_amplitude: float = 0.0  # GA: common fractional scale of the whole template
    global_index: float = 0.0  # GB: common index offset of the whole sky


def jacobian(response, point, template, freqs, widths: Widths) -> np.ndarray:
    """``(n_data, 2 n_pix + 1 + globals)``: A and B maps, curvature, then the global columns.

    The global amplitude column is the A-map Jacobian applied to the template
    (``A += s a_g``); the global index column is the B-map Jacobian applied to
    the current amplitude (``B += d_beta a_t``).
    """
    beta_t, c_t, a_t = point
    spectra = spectra_at(beta_t, c_t, freqs)
    maps = moment_jacobian(response, spectra, np.log(freqs / scenario.NU0_MHZ))
    n_pix = beta_t.size
    q_shift = np.asarray(phys.spectral_design(freqs)[1])
    columns = [maps, np.einsum("ftp,fp->ft", response, spectra * a_t[None, :] * q_shift[:, None]).reshape(-1, 1)]
    if widths.global_amplitude > 0:
        columns.append((maps[:, :n_pix] @ template[1]).reshape(-1, 1))
    if widths.global_index > 0:
        columns.append((maps[:, n_pix:] @ a_t).reshape(-1, 1))
    return np.hstack(columns)


def _prior(template, point, widths: Widths):
    (beta_g, a_g, c_g), (beta_t, c_t, a_t) = template, point
    loc = [a_g, a_t * (beta_g - beta_t), [c_g - c_t]]
    scale = [widths.amplitude * np.abs(a_g), widths.index * np.abs(a_t), [widths.curvature]]
    for width in (widths.global_amplitude, widths.global_index):
        if width > 0:
            loc.append([0.0])
            scale.append([width])
    return np.concatenate(loc), np.concatenate(scale)


def _step(post, point, template, widths: Widths):
    """The next linearisation point from a posterior mean of the parameters."""
    beta_t, c_t, a_t = point
    n_pix = beta_t.size
    amplitude, first, extra = post[:n_pix], post[n_pix : 2 * n_pix], list(post[2 * n_pix :])
    dc = extra.pop(0)
    if widths.global_amplitude > 0:
        amplitude = amplitude + extra.pop(0) * template[1]
    if widths.global_index > 0:
        first = first + extra.pop(0) * a_t
    safe = np.where(np.abs(amplitude) > 1e-6, amplitude, 1e-6)
    beta = np.clip(beta_t + first / safe, *BETA_RANGE)
    return (beta, c_t + dc, amplitude), [float(np.median(np.abs(first / safe))), float(dc)]


def linear_model(response, waterfall, freqs, template, widths: Widths, relinearise: int) -> dict:
    """Jacobian, prior, and their :class:`GaussianMarginal`, at the point after ``relinearise`` steps."""
    point = (template[0].copy(), template[2], template[1].copy())
    data, steps = vec(waterfall), []
    for step in range(relinearise + 1):
        jac = jacobian(response, point, template, freqs, widths)
        loc, scale = _prior(template, point, widths)
        marginal = GaussianMarginal.build(jac, loc, scale)
        if step == relinearise:
            break
        point, record = _step(marginal.posterior_mean(data), point, template, widths)
        steps.append(record)
    return {"jac": jac, "loc": loc, "scale": scale, "marginal": marginal, "beta_t": point[0],
            "c_t": float(point[1]), "steps": steps}  # fmt: skip


def log_evidence(model: dict, waterfall) -> float:
    """``log N(d; J m, sigma^2 I + J S J^T)`` with no 21 cm signal (type-II likelihood)."""
    return model["marginal"].log_evidence(vec(waterfall))


def response(beam_alm, nside: int, lmax: int) -> np.ndarray:
    """The fit's drift-scan matrix with the known beam's alms."""
    alms = np.asarray(beam_alm)
    have = hp.Alm.getlmax(alms.shape[-1])
    if have != lmax:
        alms = np.stack([hp.resize_alm(a, have, have, lmax, lmax) for a in alms])
    return instrument.drift_response(alms, nside, lmax, scenario.lst_deg())
