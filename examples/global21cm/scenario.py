"""The simulated observations the demonstrations fit, as constants.

Two scenarios share one code path and differ only in the fields of
:class:`Scenario`:

* ``main``: 45-135 MHz, an analytic chromatic Gaussian beam. The band is
  wide enough that the absorption trough is not a smooth piece of the
  foreground's spectral span.
* ``stress``: RHINO's 55-85 MHz band with the simulated ``HornWet`` horn
  beams. Kept as a labelled stress case: there the beam's chromaticity makes
  the signal and the foreground nearly degenerate (README, "Results").

Every number the simulator, the plugin, the analysis and the tests share
lives here, so the YAML documents can be checked against it
(``tests/test_documents.py``) rather than trusted to agree.

Units: frequencies in MHz, temperatures in K, angles in degrees.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


@dataclass(frozen=True)
class Scenario:
    """One simulated observation."""

    name: str
    freq_start_mhz: float
    freq_stop_mhz: float
    n_freq: int
    beam: str  # "gaussian" or "hornwet"

    @property
    def sim_dir(self) -> Path:
        return RESULTS / ("sim" if self.name == "main" else f"sim_{self.name}")

    def freqs_mhz(self) -> np.ndarray:
        """The channel centres, MHz."""
        return np.linspace(self.freq_start_mhz, self.freq_stop_mhz, self.n_freq)


MAIN = Scenario("main", 45.0, 135.0, 46, "gaussian")  # 2 MHz channels
STRESS = Scenario("stress", 55.0, 85.0, 31, "hornwet")  # 1 MHz channels
SCENARIOS = {s.name: s for s in (MAIN, STRESS)}

# --------------------------------------------------------------- the grid --
N_TIME = 96  # LST samples over one full sidereal turn; 2 * 47 < 96
SIDEREAL_DAY_S = 86164.0905
TIME_STEP_S = SIDEREAL_DAY_S / N_TIME
LST0_DEG = 0.0

# ------------------------------------------------------------------- site --
LAT_DEG = 53.2  # RHINO (Jodrell Bank), zenith pointing, drift scan
AZ_DEG = 0.0
EL_DEG = 90.0

# ------------------------------------------------------------ resolutions --
NSIDE_SIM = 64  # truth sky and beam
# 2.5 NSIDE rather than 3 NSIDE - 1: at NSIDE 64 the projection changes by
# 1.1 (main) and 3.1 (stress) mK rms from lmax 127 to 159, but by 15.8 and
# 9.5 mK from 159 to 191, the multipoles at the HEALPix band edge being the
# poorly sampled ones (results/sim*/truth.json, "convergence").
LMAX_SIM = 159
NSIDE_FIT = 16  # demo B's sky maps
LMAX_FIT = 3 * NSIDE_FIT - 1  # 47

# ------------------------------------------------------ foreground physics --
NU_HASLAM_MHZ = 408.0
MONOPOLE_OFFSET_K = 8.9  # CMB + extragalactic background, MERS fg_model.py:135
CURVATURE_TRUE = -0.10  # MERS fg_model.py:73
NU_CURVATURE_MHZ = (23000.0 - 408.0) / 2.0 + 408.0  # 11704 MHz, MERS fg_model.py:71
NU0_MHZ = 70.0  # reference frequency of demo B's amplitude map

# Demo A moment basis
BETA0_MOMENT = -2.55
NU_REF_MOMENT_MHZ = 70.0

# Analytic beam: FWHM 30 deg at 70 MHz, scaling as 70 MHz / nu.
GAUSSIAN_FWHM70_DEG = 30.0

# ------------------------------------------------------------- 21 cm truth --
# 21cmVAE parameters in PARAMETER_NAMES order:
# (fstar, Vc [km/s], fx, tau, alpha, nu_min [keV], Rmfp [Mpc]).
# Inside the emulator's training box; the trough is at 72 MHz, depth 0.152 K.
THETA_TRUE = (0.1, 20.0, 0.5, 0.07, 1.25, 0.5, 30.0)

# ------------------------------------------------------------------ noise --
# White, homoscedastic, additive: 10 mK per sample in both scenarios, taken
# as constant across the waterfall. With the noiseless sky as the system
# temperature (receiver neglected) and one 898 s LST bin, 10 mK is the
# radiometer noise after 15.0 days of 2 MHz channels at 69 MHz (main) and
# 26.9 days of 1 MHz channels at 70 MHz (stress), medians over LST
# (results/analysis/sensitivity.json, radiometer.days_reference).
NOISE_SIGMA_K = 0.01
NOISE_SEED = 20260923

# ------------------------------------------------------------- data paths --
MERS_DATA_ENV = "MERS_DATA_DIR"
HORNWET_ENV = "HORNWET_DIR"
HASLAM_FILE = "haslam408_dsds_Remazeilles2014.fits"
BETA_FILE = "cnn56arcmin_beta.npy"


def lst_deg() -> np.ndarray:
    """A uniform full-turn LST grid, endpoint excluded."""
    return LST0_DEG + 360.0 * np.arange(N_TIME) / N_TIME


def mers_data_dir() -> Path:
    """The MERS input-map directory, from ``$MERS_DATA_DIR``; refused if absent."""
    raw = os.environ.get(MERS_DATA_ENV)
    if not raw:
        raise FileNotFoundError(
            f"${MERS_DATA_ENV} is not set. Point it at MERS's data/ directory, "
            f"the one holding {HASLAM_FILE} and {BETA_FILE}."
        )
    path = Path(raw).expanduser()
    missing = [name for name in (HASLAM_FILE, BETA_FILE) if not (path / name).is_file()]
    if missing:
        raise FileNotFoundError(f"${MERS_DATA_ENV}={path} does not contain {missing}.")
    return path


def hornwet_dir() -> Path:
    """The RHINO HornWet beam directory from ``$HORNWET_DIR``; refused if absent."""
    raw = os.environ.get(HORNWET_ENV)
    path = Path(raw).expanduser() if raw else None
    if path is None or not path.is_dir():
        raise FileNotFoundError(
            f"${HORNWET_ENV} does not name a directory; the stress scenario needs "
            "the HornWet{f}.fits beam files."
        )
    return path
