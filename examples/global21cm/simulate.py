"""Simulate the drift-scan waterfall of one scenario.

Truth: the MERS synchrotron sky (Haslam 408 MHz, the CNN spectral-index map
and the curvature law) at NSIDE 64, seen through the scenario's chromatic
beam by limtod_jax's drift scan at lmax 159, plus the 21cmVAE global signal,
plus white noise. ``truth.json`` records a band-limit convergence check: the
rms change when the same maps are projected at lmax 127 and at 191.

Writes ``results/sim/`` (main) or ``results/sim_stress/``.

Run from the repository root:

    MERS_DATA_DIR=... PYTHONPATH=examples .venv/bin/python -m global21cm.simulate main
    MERS_DATA_DIR=... HORNWET_DIR=... PYTHONPATH=examples \
        .venv/bin/python -m global21cm.simulate stress
"""

from __future__ import annotations

import argparse
import json
import time

import jax

jax.config.update("jax_enable_x64", True)

import healpy as hp  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from global21cm import foreground_physical as phys  # noqa: E402
from global21cm import instrument, scenario  # noqa: E402
from global21cm.foreground_beamconv import beam_svd_spectra  # noqa: E402

#: How many beam SVD spectra to store; demo A uses the first few.
N_BEAM_SPECTRA = 8


def signal_21cm(freqs_mhz) -> np.ndarray:
    """The injected 21 cm curve, K."""
    from global21cm_jax.emulator import Global21cmEmulator
    from global21cm_jax.signal import global_signal_kelvin

    emulator = Global21cmEmulator.load()
    theta = jnp.asarray(scenario.THETA_TRUE)
    return np.asarray(global_signal_kelvin(emulator, theta, jnp.asarray(freqs_mhz) * 1e6))


def foreground(t408, beta, beams, freqs, nside: int, lmax: int) -> np.ndarray:
    """Beam-convolved MERS foreground ``(n_time, n_freq)`` K at ``(nside, lmax)``."""
    sky = np.asarray(phys.mers_synchrotron(t408, beta, freqs)).T
    alms = instrument.beam_alms(beams, lmax)
    return instrument.drift_waterfall(sky, alms, nside, lmax, scenario.lst_deg())


#: Band-limits the truth is recomputed at for the convergence record.
CONVERGENCE_LMAX = (127, 191)  # s2fft needs lmax >= 2 nside - 1 = 127


def convergence(t408, beta, beams, freqs, reference) -> dict[str, float]:
    """rms change of the truth, mK, when the same maps are projected at another lmax."""
    out = {}
    for lmax in CONVERGENCE_LMAX:
        lower = foreground(t408, beta, beams, freqs, scenario.NSIDE_SIM, lmax)
        out[f"lmax{lmax}_rms_mk"] = float(1e3 * np.sqrt(np.mean((lower - reference) ** 2)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="simulate one scenario")
    parser.add_argument("scenario", choices=sorted(scenario.SCENARIOS))
    case = scenario.SCENARIOS[parser.parse_args().scenario]
    start = time.perf_counter()
    freqs = case.freqs_mhz()
    t408, beta = phys.load_mers_maps(scenario.NSIDE_SIM)
    beams = instrument.beam_maps(case, scenario.NSIDE_SIM)
    fg = foreground(t408, beta, beams, freqs, scenario.NSIDE_SIM, scenario.LMAX_SIM)
    t21 = signal_21cm(freqs)
    noiseless = fg + t21[None, :]
    rng = np.random.default_rng(scenario.NOISE_SEED)
    waterfall = noiseless + scenario.NOISE_SIGMA_K * rng.standard_normal(noiseless.shape)
    arrays = {
        "waterfall": waterfall,
        "noiseless": noiseless,
        "fg_truth": fg,
        "t21_truth": t21,
        "freqs_mhz": freqs,
        "beam_svd_spectra": beam_svd_spectra(beams, N_BEAM_SPECTRA),
        "t408_fit": hp.ud_grade(t408, scenario.NSIDE_FIT),
        "beta_fit": hp.ud_grade(beta, scenario.NSIDE_FIT),
    }
    case.sim_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(case.sim_dir / f"{name}.npy", np.asarray(value, dtype=np.float64))
    # Complex: the fit's beam, analysed at its own band-limit.
    np.save(case.sim_dir / "beam_alm_fit.npy", np.asarray(instrument.beam_alms(beams, scenario.LMAX_FIT)))
    trough = int(np.argmin(t21))
    record = {
        "scenario": case.name,
        "beam": case.beam,
        "band_mhz": [case.freq_start_mhz, case.freq_stop_mhz, case.n_freq],
        "theta_true": list(scenario.THETA_TRUE),
        "noise_sigma_k": scenario.NOISE_SIGMA_K,
        "nside_sim": scenario.NSIDE_SIM,
        "lmax_sim": scenario.LMAX_SIM,
        "n_time": scenario.N_TIME,
        "trough_mhz": float(freqs[trough]),
        "trough_k": float(t21[trough]),
        "foreground_k_range": [float(fg.min()), float(fg.max())],
        "convergence": convergence(t408, beta, beams, freqs, fg),
        "seconds": round(time.perf_counter() - start, 1),
    }
    (case.sim_dir / "truth.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
