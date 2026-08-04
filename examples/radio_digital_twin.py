"""Radio telescope digital twin — end-to-end demo, graph-assembled.

Composition is implicit in the canonical signal-path graph: provide the
operators, and ``assemble`` lights up the connected sub-path they induce and
compiles it to the equivalent Pipeline/SumOperator nesting (see
``rheplicant/radio/graph.py`` for the topology and DESIGN.md D11 for the rules).

Then the inference seam works unchanged: gradient calibration of the gain
against "observed" data, without touching the forward model's internals.

Run:  uv run python examples/radio_digital_twin.py
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant import Coordinates, Environment, State
from rheplicant.inference import GradientCalibrator, build_forward_fn
from rheplicant.radio import (
    ADCOperator,
    AtmosphericEmissionOperator,
    BackendOperator,
    BeamOperator,
    CWCalibrationOperator,
    EMIOperator,
    FlaggingOperator,
    ForegroundOperator,
    GainOperator,
    GlobalSignalOperator,
    GroundPickupOperator,
    IonosphereOperator,
    NoiseOperator,
    NoiseWaveOperator,
    PointSourceOperator,
    ReceiverOperator,
    RFIOperator,
    assemble,
)

N_TIME, N_FREQ = 128, 32
# Channel spacing of the grid below. For the CW tone's default 'sinc2'
# lineshape this is the first-null offset of a critically sampled unwindowed
# FFT — the width has no default because it is a property of the spectrometer.
CHANNEL_HZ = 25e6 / (N_FREQ - 1)

# ------------------------------------------------------------ observation --
state = State(
    coords=Coordinates(
        time=jnp.linspace(0.0, 60.0, N_TIME),           # seconds
        freq=jnp.linspace(60e6, 85e6, N_FREQ),          # Hz (demo values)
    ),
    env=Environment(temperature=jnp.array(280.0)),      # couples into ground pickup
    key=jax.random.key(0),
    meta={"telescope": "generic-dish", "obs_id": "demo-001"},
)

# Provide the elements; the signal-path graph decides how they compose.
twin = assemble(
    GlobalSignalOperator(depth=jnp.array(0.2), centre=jnp.array(72e6),
                         width=jnp.array(5e6)),
    ForegroundOperator(amplitude=jnp.array(1.0e3), spectral_index=jnp.array(2.5),
                       ref_freq=70e6),
    PointSourceOperator(level=jnp.array(2.0)),
    IonosphereOperator(delta=jnp.array(0.01), ref_freq=70e6),
    RFIOperator(amplitude=jnp.array(2.0e3), occupancy=0.01),   # pre-beam field
    BeamOperator(solid_angle=jnp.array(0.8)),
    GroundPickupOperator(coupling=jnp.array(0.01),             # post-beam effective
                         t_ground=jnp.array(300.0)),
    AtmosphericEmissionOperator(t_atm=jnp.array(150.0)),       # t_ant_sum branch
    NoiseWaveOperator(t_unc=jnp.array(1.0), t_cos=jnp.array(0.5),
                      t_sin=jnp.array(0.2), t_rx=jnp.array(2.0),
                      gamma_src_re=jnp.full((1, N_FREQ), 0.05),
                      gamma_src_im=jnp.full((1, N_FREQ), 0.02),
                      gamma_rec_re=jnp.zeros(N_FREQ),
                      gamma_rec_im=jnp.zeros(N_FREQ)),
    CWCalibrationOperator(
        amplitude=jnp.array(500.0), tone_freq=80e6, line_width=CHANNEL_HZ
    ),
    ReceiverOperator(bandpass=jnp.ones(N_FREQ)),
    GainOperator(gain=jnp.array(1.1)),                         # the "true" gain
    NoiseOperator(sigma=jnp.array(0.5)),
    EMIOperator(amplitude=jnp.array(0.3), period=8),
    ADCOperator(scale=jnp.array(1.0), n_bits=14),
    FlaggingOperator(threshold=2.5e3),
    BackendOperator(n_chunk=4),
)
print(twin)

observation = eqx.filter_jit(twin)(state)
print(f"simulated waterfall: {observation.data.shape}  "
      f"(flags: {int(observation.aux['flags'].sum())} samples, "
      f"meta preserved: {observation.meta['telescope']!r})")

# --------------------------------------------------------------- inference --
# Mis-calibrated model: same instrument, wrong gain. Train ONLY the gain —
# node-id access works on the assembly regardless of fold nesting.
model = twin.replace_node("gain", GainOperator(gain=jnp.array(1.0)))
spec = jax.tree.map(lambda _: False, model)
spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
forward, params0 = build_forward_fn(model, state, filter_spec=spec)

calibrator = GradientCalibrator(learning_rate=2e-7, n_steps=200)
params_fit, losses = calibrator.fit(forward, params0, observation.data)

fitted_gain = jax.tree.leaves(params_fit)[0]
print(f"calibration: gain 1.000 -> {float(fitted_gain):.4f}  (truth 1.100)")
print(f"loss: {float(losses[0]):.3e} -> {float(losses[-1]):.3e}")
