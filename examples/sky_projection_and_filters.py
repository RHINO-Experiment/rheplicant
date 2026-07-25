"""Modular sky + calibrated-data analysis — end-to-end demo.

Demonstrates the three integration seams:

1. modular sky: SkySourceOperator = PowerLawSkyModel x MatrixProjector
   (the matrix stands in for limTOD's generate_sky2sys_projection output);
2. analysis on calibrated data: snapshot -> apply calibration -> filters;
3. sky-space filtering reusing the SAME projector via its adjoint (CG).

Run:  uv run python examples/sky_projection_and_filters.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant import Coordinates, Pipeline, SnapshotOperator, State
from replicant.radio import (
    ApplyCalibrationOperator,
    GainOperator,
    MatrixProjector,
    NoiseOperator,
    PowerLawSkyModel,
    SiderealFilter,
    SkySourceOperator,
    SkySpaceFilter,
)

N_DAYS, N_LST, N_FREQ, N_PIX = 4, 32, 8, 20
N_TIME = N_DAYS * N_LST

# --------------------------------------------------------------- forward ---
# Projection matrix: one sidereal day of drift-scan pointings, repeated daily
# (in real use: limTOD.generate_sky2sys_projection, computed offline once).
key = jax.random.key(0)
one_day = jax.nn.relu(jax.random.normal(key, (N_LST, N_PIX)))
projection = jnp.tile(one_day, (N_DAYS, 1)) / N_PIX

sky_source = SkySourceOperator(
    sky_model=PowerLawSkyModel(
        amplitude=50.0 * jax.random.uniform(jax.random.key(1), (N_PIX,)),
        spectral_index=jnp.array(2.5),
        ref_freq=70e6,
        n_pix=N_PIX,
    ),
    projector=MatrixProjector(matrix=projection),
)

twin = Pipeline(
    sky_source,
    GainOperator(gain=jnp.array(1.1)),
    NoiseOperator(sigma=jnp.array(0.02)),
    names=("sky", "gain", "noise"),
)

state = State(
    coords=Coordinates(time=jnp.arange(float(N_TIME)), freq=jnp.linspace(60e6, 85e6, N_FREQ)),
    key=jax.random.key(42),
)
observation = eqx.filter_jit(twin)(state)
print(f"observed waterfall: {observation.data.shape}")

# -------------------------------------------------------------- analysis ---
analysis = Pipeline(
    SnapshotOperator(name="raw"),
    ApplyCalibrationOperator(gain=jnp.array(1.1)),        # inferred elsewhere
    names=("snap", "cal"),
)
calibrated = analysis(observation)

# Sidereal filter: the day-repeating structure IS the sky signal here.
sky_locked = SiderealFilter(n_days=N_DAYS, mode="extract")(calibrated)
residual = SiderealFilter(n_days=N_DAYS, mode="remove")(calibrated)
print(f"sidereal-repeating rms: {jnp.std(sky_locked.data):8.3f}   "
      f"non-repeating rms: {jnp.std(residual.data):.3f}")

# Sky-space filter: project onto sky space through the SAME projector.
sky_filter = SkySpaceFilter(
    projector=sky_source.projector, regularization=jnp.array(1e-4),
    cg_maxiter=200, mode="extract",
)
sky_part = sky_filter(calibrated)
truth = sky_source(state).data
err = jnp.linalg.norm(sky_part.data - truth) / jnp.linalg.norm(truth)
print(f"sky-space extraction vs true sky signal: relative error {err:.3%}")

raw = calibrated.aux["snapshot/raw"]
print(f"raw data preserved through the whole analysis: {raw.shape}")
