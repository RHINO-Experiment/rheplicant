"""Drift scan the fast way: the m-mode engine — end-to-end demo.

A drift scan points at one fixed spot on the sky and lets the Earth do the
scanning. The general engine does not know that, so it recomputes the full
beam rotation for every single sample; the m-mode engine rotates ONCE and
turns the rest of the sidereal day into per-m phases. Same numbers to
float64 roundoff, a fraction of the work.

Six things happen below:

1. build a drift-scan twin from a HEALPix beam MAP (no healpy needed);
2. check it against the general engine — this is an optimization, not an
   approximation;
3. turn on the two fast-path opt-ins and time all three;
4. read off the m-modes, which is what a drift scan actually measures;
5. hoist the sky analysis out of the loop, for when the twin is being fitted;
6. take a gradient through the whole twin — the reason any of this is in JAX.

The sky engine is a dependency, not an extra -- limTOD >= 1.10, which a plain
`pip install rheplicant` brings in from PyPI.
Run:  uv run --frozen python examples/driftscan_mmode.py   (~60 s: mostly JIT)
"""

import time

import jax

jax.config.update("jax_enable_x64", True)  # quantitative work wants float64

import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from rheplicant import Coordinates, State  # noqa: E402
from rheplicant.radio import SkySourceOperator, assemble  # noqa: E402
from rheplicant.radio.sky import DriftScanProjector, GeneralPointingProjector  # noqa: E402
from rheplicant.radio.sky.model import AbstractSkyModel  # noqa: E402

NSIDE, N_FREQ = 16, 4
LMAX = 3 * NSIDE - 1
N_TIME = 4 * (LMAX + 1)  # FFT synthesis needs 2*lmax < n_time (sampling theorem)
N_PIX = 12 * NSIDE**2
LAT_DEG, AZ_DEG, EL_DEG = 53.2, 0.0, 90.0  # a zenith drift scan


class MapSky(AbstractSkyModel):
    """Fixed brightness maps — stand-in for GSM/foreground models."""

    maps: jax.Array

    def __call__(self, freq: jax.Array) -> jax.Array:
        return self.maps


# ------------------------------------------------------------ 1. the twin ---
# A Gaussian beam and a lumpy sky, both as ordinary HEALPix maps.
key = jax.random.key(0)
theta = jnp.arccos(1.0 - 2.0 * (jnp.arange(N_PIX) + 0.5) / N_PIX)  # ~ring colatitude
beam_maps = jnp.stack([jnp.exp(-0.5 * (theta / (0.25 + 0.02 * f)) ** 2)
                       for f in range(N_FREQ)])
beam_maps = beam_maps / beam_maps.sum(axis=1, keepdims=True)     # unit pixel sum
sky_maps = 100.0 + 20.0 * jax.random.normal(key, (N_FREQ, N_PIX))

# from_beam_maps runs the beam analysis in JAX, so the beam MAP stays
# differentiable and the sky's quadrature transform cannot be picked by
# mistake (it would silently rescale the beam by npix/4pi).
drift = DriftScanProjector.from_beam_maps(
    beam_maps, lat_deg=LAT_DEG, az_deg=AZ_DEG, el_deg=EL_DEG, lmax=LMAX,
)

# The LST grid the FFT path wants: a FULL sidereal turn, endpoint excluded.
lst_deg = DriftScanProjector.uniform_lst_grid(N_TIME)
coords = Coordinates(time=jnp.arange(float(N_TIME)),
                     freq=jnp.linspace(60e6, 85e6, N_FREQ),
                     extra={"lst_deg": lst_deg})
state = State(coords=coords, key=jax.random.key(42), meta={"scan": "drift"})

twin = assemble(SkySourceOperator(sky_model=MapSky(sky_maps), projector=drift))
observation = eqx.filter_jit(twin)(state)
print(f"drift-scan waterfall: {observation.data.shape}  "
      f"({observation.data.mean():.1f} K mean)")

# ------------------------------------- 2. the same physics, cross-checked ---
# The general engine reads the pointing per sample, so hand it a constant one.
generic = GeneralPointingProjector(beam_alms=drift.beam_alms, lat_deg=LAT_DEG,
                                lmax=LMAX, nside=NSIDE)
generic_coords = coords.replace(
    pointing=jnp.stack([jnp.full(N_TIME, AZ_DEG), jnp.full(N_TIME, EL_DEG)], -1),
    extra={"lst_deg": lst_deg, "selfrot_deg": jnp.zeros(N_TIME)},
)
reference = eqx.filter_jit(lambda p, s: p.forward(s, generic_coords))(generic, sky_maps)
worst = jnp.max(jnp.abs(observation.data - reference)) / jnp.max(jnp.abs(reference))
print(f"m-mode vs general engine: {worst:.2e} relative — float64 roundoff")

# ------------------------------------------------- 3. the two fast paths ---
# to_reference_frame() pays the O(lmax^3) rotation ONCE, outside any loop;
# uniform_sampling routes the time synthesis through real FFTs.
cached = drift.to_reference_frame(lst_ref_deg=float(lst_deg[0]))
fastest = DriftScanProjector.from_beam_maps(
    beam_maps, lat_deg=LAT_DEG, az_deg=AZ_DEG, el_deg=EL_DEG, lmax=LMAX,
    uniform_sampling=True,
).to_reference_frame(lst_ref_deg=float(lst_deg[0]))

run_drift = eqx.filter_jit(lambda p, s: p.forward(s, coords))
run_generic = eqx.filter_jit(lambda p, s: p.forward(s, generic_coords))


def bench(fn, projector, repeats=5):
    jax.block_until_ready(fn(projector, sky_maps))          # compile first
    t0 = time.perf_counter()
    for _ in range(repeats):
        jax.block_until_ready(fn(projector, sky_maps))
    return (time.perf_counter() - t0) / repeats


t_generic = bench(run_generic, generic)
for label, projector in (("m-mode", drift), ("+ cached beam", cached),
                         ("+ cached beam + FFT", fastest)):
    dt = bench(run_drift, projector)
    print(f"{label:>22}: {dt * 1e3:7.2f} ms   ({t_generic / dt:5.1f}x faster)")
print(f"{'general engine':>22}: {t_generic * 1e3:7.2f} ms")

# --------------------------------------------- 4. what a drift scan sees ---
# The TOD of a drift scan is periodic over a sidereal day, so it IS a Fourier
# series: the m-modes are its coefficients, and only the first few matter.
mmodes = cached.mmodes(sky_maps, coords)          # (n_freq, lmax + 1) complex
power = jnp.abs(mmodes[0])
print(f"m-modes: {mmodes.shape}, "
      f"|V_m|/|V_0| falls to {power[5] / power[0]:.1e} by m=5")

# ----------------------------------------- 5. hoist the sky out of the loop ---
# forward() analyses the sky maps into alms on every call. With a FIXED sky
# that becomes the dominant cost as resolution grows — at nside 64 / lmax 191
# / 32 channels it is 91 % of the runtime and 99.5 % of the peak memory, and
# skipping it takes a forward from 163 ms to 1.1 ms. At THIS example's toy
# size the analysis is cheap and the hoist buys nothing measurable; it is the
# call to reach for when the twin is inside a fit, not a free win everywhere.
sky_alms = eqx.filter_jit(drift.sky_to_alms)(sky_maps)
tod_from_alms = eqx.filter_jit(lambda p, a: p.forward_alms(a, coords))(fastest, sky_alms)
agree = jnp.max(jnp.abs(tod_from_alms - observation.data)) / jnp.max(
    jnp.abs(observation.data)
)
print(f"forward_alms agrees with forward to {agree:.0e} "
      f"(same operator, sky analysed once)")

# ------------------------------------------------- 6. why it is all JAX ---
# The beam map is a differentiable leaf: this gradient is what a beam-model
# fit descends. Note it stays on the BEAM-LOCAL projector — to_reference_frame()
# would move the gradient to the reference-frame alms — so the beam rotation
# cannot be cached here, which is exactly why the sky hoist above matters.
def loss(maps: jax.Array) -> jax.Array:
    projector = DriftScanProjector.from_beam_maps(
        maps, lat_deg=LAT_DEG, az_deg=AZ_DEG, el_deg=EL_DEG, lmax=LMAX,
    )
    return jnp.sum((projector.forward_alms(sky_alms, coords) - reference) ** 2)


grad_fn = eqx.filter_jit(jax.grad(loss))
wrong_beam = beam_maps * 1.05                      # a 5 % mis-scaled beam
print(f"d(chi^2)/d(beam map): {grad_fn(wrong_beam).shape}, "
      f"|grad| = {jnp.linalg.norm(grad_fn(wrong_beam)):.3e} off the truth, "
      f"{jnp.linalg.norm(grad_fn(beam_maps)):.1e} at it")
