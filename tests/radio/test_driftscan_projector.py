"""DriftScanProjector: the m-mode fast path vs the general native projector.

Tolerance strategy mirrors ``TestNativeLimTODProjector``: this suite runs in
default float32, but the drift-vs-native comparison is TIGHTER than the
native-vs-numpy oracle one (1e-3 vs 5e-2) because both projectors share the
same s2fft map<->alm transforms — the f32 transform error cancels and only
the rotation kernels differ (Risbo, f32-stable). The float64 roundoff
statement (1e-6 vs numpy limTOD) lives in ``test_x64_subprocess`` and in the
limtod_jax suite.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.radio.sky import DriftScanProjector, NativeLimTODProjector

N_TIME, N_FREQ = 8, 2


@pytest.fixture
def key():
    return jax.random.key(0)


def dot(a, b):
    return jnp.sum(a * b)


class TestDriftScanProjector:
    NSIDE = 4
    LMAX = 11
    N_PIX_HP = 12 * NSIDE**2
    N_ALM = (LMAX + 1) * (LMAX + 2) // 2
    LAT, AZ, EL, SELFROT = 53.2, 30.0, 50.0, 10.0

    @pytest.fixture
    def drift_coords(self, coords):
        # Irregular LST steps incl. a wrap past 360 deg — a drift scan is
        # defined by fixed pointing, not by uniform sampling.
        return coords.replace(
            freq=coords.freq[:N_FREQ],
            extra={"lst_deg": jnp.asarray([12.0, 61.3, 155.0, 262.9, 300.1, 330.0, 359.9, 401.6])},
        )

    def _projector(self, key, **kwargs):
        pytest.importorskip("limtod_jax")
        re, im = jax.random.normal(key, (2, N_FREQ, self.N_ALM))
        im = im.at[:, : self.LMAX + 1].set(0.0)  # valid real-field alms
        return DriftScanProjector(
            beam_alms=re + 1j * im,
            lat_deg=self.LAT,
            az_deg=self.AZ,
            el_deg=self.EL,
            lmax=self.LMAX,
            nside=self.NSIDE,
            selfrot_deg=self.SELFROT,
            **kwargs,
        )

    def _native_twin(self, proj, **kwargs):
        """NativeLimTODProjector with the same beams (the ground truth here)."""
        return NativeLimTODProjector(
            beam_alms=proj.beam_alms, lat_deg=self.LAT,
            lmax=self.LMAX, nside=self.NSIDE, **kwargs,
        )

    def _native_coords(self, drift_coords):
        """The drift pointing expressed in the general projector's contract."""
        lst = drift_coords.extra["lst_deg"]
        n_t = lst.shape[0]
        return drift_coords.replace(
            pointing=jnp.stack(
                [jnp.full(n_t, self.AZ), jnp.full(n_t, self.EL)], axis=-1
            ),
            extra={"lst_deg": lst, "selfrot_deg": jnp.full(n_t, self.SELFROT)},
        )

    # -------------------------------------------------------------- forward
    @pytest.mark.parametrize("normalize", [False, True])
    def test_matches_native_projector(self, key, drift_coords, normalize):
        proj = self._projector(key, normalize_beam=normalize)
        native = self._native_twin(proj, normalize_beam=normalize)
        sky = jax.random.uniform(jax.random.key(1), (N_FREQ, self.N_PIX_HP))
        out = proj.forward(sky, drift_coords)
        ref = native.forward(sky, self._native_coords(drift_coords))
        assert out.shape == ref.shape == (N_TIME, N_FREQ)
        rel = jnp.max(jnp.abs(out - ref)) / jnp.max(jnp.abs(ref))
        assert rel < 1e-3, f"drift vs native rel err {rel:.2e}"

    def test_explicit_lst_ref_changes_nothing(self, key, drift_coords):
        """The reference LST only re-anchors phases; the TOD is invariant.

        lst_ref_deg is a STATIC eqx field — not a pytree leaf — so variants
        are built with dataclasses.replace (re-runs validation), never
        eqx.tree_at (which only addresses leaves)."""
        import dataclasses

        sky = jax.random.uniform(jax.random.key(2), (N_FREQ, self.N_PIX_HP))
        base = self._projector(key).forward(sky, drift_coords)
        shifted = dataclasses.replace(self._projector(key), lst_ref_deg=237.5)
        out = shifted.forward(sky, drift_coords)
        rel = jnp.max(jnp.abs(out - base)) / jnp.max(jnp.abs(base))
        assert rel < 1e-3, f"lst_ref must be a gauge choice, moved TOD by {rel:.2e}"

    # -------------------------------------------------------------- adjoint
    @pytest.mark.parametrize("normalize", [False, True])
    def test_adjoint_dot_identity(self, key, drift_coords, normalize):
        proj = self._projector(key, normalize_beam=normalize)
        sky = jax.random.uniform(jax.random.key(3), (N_FREQ, self.N_PIX_HP))
        tod = jax.random.normal(jax.random.key(4), (N_TIME, N_FREQ))
        lhs = dot(proj.forward(sky, drift_coords), tod)
        rhs = dot(sky, proj.adjoint(tod, drift_coords))
        assert jnp.abs(lhs - rhs) / jnp.abs(lhs) < 1e-3

    def test_adjoint_matches_native(self, key, drift_coords):
        proj = self._projector(key)
        native = self._native_twin(proj)
        tod = jax.random.normal(jax.random.key(5), (N_TIME, N_FREQ))
        ours = proj.adjoint(tod, drift_coords)
        ref = native.adjoint(tod, self._native_coords(drift_coords))
        assert ours.shape == ref.shape == (N_FREQ, self.N_PIX_HP)
        rel = jnp.max(jnp.abs(ours - ref)) / jnp.max(jnp.abs(ref))
        assert rel < 1e-3, f"adjoint drift vs native rel err {rel:.2e}"

    # -------------------------------------------------------------- m-modes
    def test_mmodes_shape_and_forward_consistency(self, key, drift_coords):
        """mmodes are the Fourier coefficients of the forward TOD: resynthesize
        the TOD from them at the coords' phases and compare."""
        pytest.importorskip("limtod_jax")
        import limtod_jax as ltj

        proj = self._projector(key)
        sky = jax.random.uniform(jax.random.key(6), (N_FREQ, self.N_PIX_HP))
        vm = proj.mmodes(sky, drift_coords)
        assert vm.shape == (N_FREQ, self.LMAX + 1)
        assert jnp.iscomplexobj(vm)
        lst = drift_coords.extra["lst_deg"]
        dphi = jnp.deg2rad(lst - lst[0])
        resynth = jnp.stack([ltj.tod_from_mmodes(vm[f], dphi) for f in range(N_FREQ)]).T
        out = proj.forward(sky, drift_coords)
        rel = jnp.max(jnp.abs(resynth - out)) / jnp.max(jnp.abs(out))
        assert rel < 1e-5, f"mmodes/forward inconsistency {rel:.2e}"

    # -------------------------------------------------- mask, grad, validation
    def test_horizon_mask_smoke(self, key, drift_coords):
        """Masked projector runs and differs from the unmasked one (low
        elevation + broad random beam: the cut removes real response).
        Static-field variants via dataclasses.replace (not eqx.tree_at)."""
        import dataclasses

        proj = self._projector(key)
        low = dataclasses.replace(proj, el_deg=10.0)
        masked = dataclasses.replace(
            proj, el_deg=10.0, horizon_mask=True, apod_deg=5.0
        )
        sky = jax.random.uniform(jax.random.key(7), (N_FREQ, self.N_PIX_HP))
        out_m = masked.forward(sky, drift_coords)
        out_u = low.forward(sky, drift_coords)
        assert jnp.all(jnp.isfinite(out_m))
        rel = jnp.max(jnp.abs(out_m - out_u)) / jnp.max(jnp.abs(out_u))
        assert rel > 1e-3, f"mask changed nothing ({rel:.2e}) — silently dropped?"

    def test_grad_and_jit(self, key, drift_coords):
        import equinox as eqx

        proj = self._projector(key)
        sky = jax.random.uniform(jax.random.key(8), (N_FREQ, self.N_PIX_HP))

        out_jit = eqx.filter_jit(lambda p, s: p.forward(s, drift_coords))(proj, sky)
        assert jnp.allclose(out_jit, proj.forward(sky, drift_coords), rtol=1e-5)

        g_sky = jax.grad(lambda s: jnp.sum(proj.forward(s, drift_coords) ** 2))(sky)
        assert bool(jnp.all(jnp.isfinite(g_sky)))

        def loss_beam(beams):
            p = eqx.tree_at(lambda q: q.beam_alms, proj, beams)
            return jnp.sum(p.forward(sky, drift_coords) ** 2)

        g_beam = jax.grad(loss_beam, holomorphic=False)(proj.beam_alms)
        assert bool(jnp.all(jnp.isfinite(jnp.abs(g_beam))))

    def test_validation(self, key, coords, drift_coords):
        proj = self._projector(key)
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))
        with pytest.raises(StateValidationError, match="lst_deg"):
            proj.forward(sky, coords)  # base coords: no lst_deg
        with pytest.raises(StateValidationError, match="sky"):
            proj.forward(jnp.ones((N_FREQ, self.N_PIX_HP - 1)), drift_coords)
        with pytest.raises(StateValidationError, match="tod"):
            proj.adjoint(jnp.ones((N_TIME + 1, N_FREQ)), drift_coords)
        with pytest.raises(StateValidationError, match="beam_alms"):
            DriftScanProjector(
                beam_alms=jnp.zeros((N_FREQ, self.N_ALM + 1), dtype=jnp.complex64),
                lat_deg=self.LAT, az_deg=self.AZ, el_deg=self.EL,
                lmax=self.LMAX, nside=self.NSIDE,
            )

    # ------------------------------------------------------ float64 statement
    def test_x64_subprocess(self):
        """Roundoff-level statement in a fresh x64 interpreter: forward ==
        numpy limTOD generate_TOD_sky on the drift scan to 1e-6, and the
        adjoint dot identity to 1e-10 (mirrors the native projector's
        subprocess test; x64 cannot be flipped mid-process)."""
        import os
        import subprocess
        import sys

        pytest.importorskip("limtod_jax")
        pytest.importorskip("limTOD.simulator")
        pytest.importorskip("healpy")

        script = f"""
import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np
from rheplicant import Coordinates
from rheplicant.radio.sky import DriftScanProjector
from limTOD.simulator import generate_TOD_sky

assert jax.config.read("jax_enable_x64")
nside, lmax, n_time = {self.NSIDE}, {self.LMAX}, {N_TIME}
lat, az, el, sr = {self.LAT}, {self.AZ}, {self.EL}, {self.SELFROT}
npix = 12 * nside**2
rng = np.random.default_rng(0)
beam_maps = rng.random((2, npix))
sky_maps = rng.random((2, npix))
lst = np.array([12.0, 61.3, 155.0, 262.9, 300.1, 330.0, 359.9, 401.6])
coords = Coordinates(
    time=jnp.arange(n_time, dtype=float),
    freq=jnp.array([60e6, 70e6]),
    extra={{"lst_deg": jnp.asarray(lst)}},
)
proj = DriftScanProjector(
    beam_alms=jnp.asarray(np.stack([hp.map2alm(b, lmax=lmax) for b in beam_maps])),
    lat_deg=lat, az_deg=az, el_deg=el, selfrot_deg=sr, lmax=lmax, nside=nside,
)
out = np.asarray(proj.forward(jnp.asarray(sky_maps), coords))
for f in range(2):
    direct = generate_TOD_sky(
        beam_maps[f], sky_maps[f], lst, lat,
        np.full(n_time, az), np.full(n_time, el), np.full(n_time, sr),
        truncate_frac_thres=0.0,
    )
    rel = np.max(np.abs(out[:, f] - direct)) / np.max(np.abs(direct))
    assert rel < 1e-6, f"freq {{f}}: rel {{rel:.3e}}"
sky = jnp.asarray(rng.random((2, npix)))
tod = jnp.asarray(rng.standard_normal((n_time, 2)))
lhs = float(jnp.sum(proj.forward(sky, coords) * tod))
rhs = float(jnp.sum(sky * proj.adjoint(tod, coords)))
assert abs(lhs - rhs) / abs(lhs) < 1e-10, (lhs, rhs)
print("x64 drift-scan projector: OK")
"""
        env = dict(os.environ, JAX_ENABLE_X64="1")
        result = subprocess.run(
            [sys.executable, "-c", script], env=env, capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
        assert "OK" in result.stdout
