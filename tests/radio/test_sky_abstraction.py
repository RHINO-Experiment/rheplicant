"""Tests for the modular sky abstraction: SkyModel x SkyProjector x SkySourceOperator."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from replicant import Pipeline, State
from replicant.core.errors import StateValidationError
from replicant.radio import GainOperator
from replicant.radio.sky import (
    LimTODProjector,
    MatrixProjector,
    MModeProjector,
    NativeLimTODProjector,
    PowerLawSkyModel,
    SkySourceOperator,
    UniformSkyModel,
)

N_TIME, N_FREQ, N_PIX = 8, 4, 6


@pytest.fixture
def key():
    return jax.random.key(0)


def dot(a, b):
    return jnp.sum(a * b)


class TestSkyModels:
    def test_uniform_shape_and_value(self):
        sky = UniformSkyModel(amplitude=jnp.array(5.0), n_pix=N_PIX)(jnp.linspace(60e6, 85e6, 4))
        assert sky.shape == (4, N_PIX)
        assert jnp.all(sky == 5.0)

    def test_power_law(self):
        freq = jnp.linspace(60e6, 85e6, N_FREQ)
        model = PowerLawSkyModel(
            amplitude=jnp.arange(1.0, N_PIX + 1.0),
            spectral_index=jnp.array(2.5),
            ref_freq=70e6,
            n_pix=N_PIX,
        )
        sky = model(freq)
        assert sky.shape == (N_FREQ, N_PIX)
        expected = (freq[2] / 70e6) ** (-2.5) * 3.0
        assert jnp.allclose(sky[2, 2], expected)

    def test_validation(self):
        with pytest.raises(StateValidationError, match="n_pix"):
            UniformSkyModel(amplitude=jnp.array(1.0), n_pix=0)
        with pytest.raises(StateValidationError, match="ref_freq"):
            PowerLawSkyModel(
                amplitude=jnp.array(1.0), spectral_index=jnp.array(2.5),
                ref_freq=-1.0, n_pix=N_PIX,
            )


class TestMatrixProjector:
    @pytest.mark.parametrize("shape", [(N_TIME, N_PIX), (N_FREQ, N_TIME, N_PIX)])
    def test_forward_shape(self, key, shape, coords):
        A = jax.random.normal(key, shape)
        sky = jnp.ones((N_FREQ, N_PIX))
        out = MatrixProjector(matrix=A).forward(sky, coords)
        assert out.shape == (N_TIME, N_FREQ)

    @pytest.mark.parametrize("shape", [(N_TIME, N_PIX), (N_FREQ, N_TIME, N_PIX)])
    def test_adjoint_dot_identity(self, key, shape, coords):
        """<A x, y> == <x, A^T y> — the property SkySpaceFilter relies on."""
        k1, k2, k3 = jax.random.split(key, 3)
        proj = MatrixProjector(matrix=jax.random.normal(k1, shape))
        x = jax.random.normal(k2, (N_FREQ, N_PIX))
        y = jax.random.normal(k3, (N_TIME, N_FREQ))
        lhs = dot(proj.forward(x, coords), y)
        rhs = dot(x, proj.adjoint(y, coords))
        assert jnp.allclose(lhs, rhs, rtol=1e-5)

    def test_pixel_mismatch_raises(self, coords):
        proj = MatrixProjector(matrix=jnp.ones((N_TIME, N_PIX)))
        with pytest.raises(StateValidationError, match="pixels"):
            proj.forward(jnp.ones((N_FREQ, N_PIX + 1)), coords)

    def test_bad_ndim_rejected(self):
        with pytest.raises(StateValidationError, match="matrix"):
            MatrixProjector(matrix=jnp.ones(N_PIX))


class TestMModeProjector:
    @pytest.fixture
    def transfer(self, key):
        re, im = jax.random.normal(key, (2, N_FREQ, N_TIME, N_PIX))
        return re + 1j * im

    def test_forward_real_and_shaped(self, transfer, coords):
        out = MModeProjector(transfer=transfer).forward(jnp.ones((N_FREQ, N_PIX)), coords)
        assert out.shape == (N_TIME, N_FREQ)
        assert jnp.isrealobj(out)

    def test_adjoint_dot_identity(self, transfer, key, coords):
        proj = MModeProjector(transfer=transfer)
        k1, k2 = jax.random.split(key)
        x = jax.random.normal(k1, (N_FREQ, N_PIX))
        y = jax.random.normal(k2, (N_TIME, N_FREQ))
        lhs = dot(proj.forward(x, coords), y)
        rhs = dot(x, proj.adjoint(y, coords))
        assert jnp.allclose(lhs, rhs, rtol=1e-4)

    def test_nm_mismatch_raises(self, transfer, coords):
        bad = MModeProjector(transfer=transfer[:, :-1, :])
        with pytest.raises(StateValidationError, match="n_m"):
            bad.forward(jnp.ones((N_FREQ, N_PIX)), coords)


class TestSkySourceOperator:
    @pytest.fixture
    def source(self, key):
        return SkySourceOperator(
            sky_model=PowerLawSkyModel(
                amplitude=jnp.ones(N_PIX), spectral_index=jnp.array(2.5),
                ref_freq=70e6, n_pix=N_PIX,
            ),
            projector=MatrixProjector(matrix=jax.random.normal(key, (N_TIME, N_PIX))),
        )

    def test_produces_data(self, source, template_state):
        out = source(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)

    def test_jit_and_grad_reach_sky_params(self, source, template_state):
        pipe = Pipeline(source, GainOperator(gain=jnp.array(1.1)), names=("sky", "gain"))
        out = eqx.filter_jit(pipe)(template_state)

        def loss(pipe):
            return jnp.mean((pipe(template_state).data - out.data * 1.05) ** 2)

        grads = eqx.filter_grad(loss)(pipe)
        g_index = grads["sky"].sky_model.spectral_index
        g_amp = grads["sky"].sky_model.amplitude
        assert jnp.isfinite(g_index) and g_index != 0
        assert jnp.all(jnp.isfinite(g_amp)) and jnp.any(g_amp != 0)

    def test_projector_swaps_independently(self, source, template_state, key):
        """The modularity contract: same sky model, different engine."""
        re, im = jax.random.normal(key, (2, N_FREQ, N_TIME, N_PIX))
        swapped = eqx.tree_at(
            lambda s: s.projector, source, MModeProjector(transfer=re + 1j * im)
        )
        out = swapped(template_state)
        assert out.data.shape == (N_TIME, N_FREQ)
        assert isinstance(swapped.sky_model, PowerLawSkyModel)  # sky untouched

    def test_type_validation(self, key):
        with pytest.raises(StateValidationError, match="sky_model"):
            SkySourceOperator(
                sky_model=jnp.ones(3),
                projector=MatrixProjector(matrix=jnp.ones((N_TIME, N_PIX))),
            )
        with pytest.raises(StateValidationError, match="projector"):
            SkySourceOperator(
                sky_model=UniformSkyModel(amplitude=jnp.array(1.0), n_pix=N_PIX),
                projector=jnp.ones(3),
            )

    def test_requires_coords(self, source):
        with pytest.raises(StateValidationError, match="coords"):
            source(State())


class TestLimTODProjector:
    def test_validates_coordinate_requirements(self, template_state):
        proj = LimTODProjector(beam_maps=jnp.ones((N_FREQ, N_PIX)), lat_deg=53.2)
        with pytest.raises(StateValidationError, match="pointing"):
            proj.forward(jnp.ones((N_FREQ, N_PIX)), template_state.coords)

    def test_oracle_matches_limtod(self, template_state):
        """End-to-end oracle test — runs only where limTOD is installed."""
        limtod = pytest.importorskip("limTOD.simulator")
        import numpy as np

        nside = 4
        n_pix = 12 * nside**2
        rng = np.random.default_rng(0)
        beam = jnp.asarray(rng.random((2, n_pix)))
        sky = jnp.asarray(rng.random((2, n_pix)))
        coords = template_state.coords.replace(
            freq=template_state.coords.freq[:2],
            pointing=jnp.tile(jnp.array([[0.0, 90.0]]), (N_TIME, 1)),
            extra={"lst_deg": jnp.linspace(0.0, 30.0, N_TIME)},
        )
        proj = LimTODProjector(beam_maps=beam, lat_deg=53.2)
        out = proj.forward(sky, coords)
        direct = limtod.generate_TOD_sky(
            np.asarray(beam[0]), np.asarray(sky[0]),
            np.linspace(0.0, 30.0, N_TIME), 53.2,
            np.zeros(N_TIME), np.full(N_TIME, 90.0), np.zeros(N_TIME),
        )
        assert out.shape == (N_TIME, 2)
        assert jnp.allclose(out[:, 0], jnp.asarray(direct), rtol=1e-5)


class TestNativeLimTODProjector:
    """The port-contract endpoint: pure-JAX, differentiable, exact adjoint.

    replicant-telescope's suite runs in default float32, so oracle/adjoint tolerances
    here are f32-scale (1e-4); the float64 1e-6 guarantees live in the
    limtod_jax package's own suite (which enables x64).
    """

    NSIDE = 4
    LMAX = 11
    N_PIX_HP = 12 * NSIDE**2
    N_ALM = (LMAX + 1) * (LMAX + 2) // 2
    LAT = 53.2

    @pytest.fixture
    def obs_coords(self, coords):
        import numpy as np

        az = jnp.asarray(np.array([0.0, 45.0, 123.4, -42.3, 80.0, 0.0, 200.0, 10.0]))
        el = jnp.asarray(np.array([90.0, 60.0, 5.0, 41.0, 30.0, 90.0, 70.0, 45.0]))
        return coords.replace(
            freq=coords.freq[:2],
            pointing=jnp.stack([az, el], axis=-1),
            extra={
                "lst_deg": jnp.linspace(0.0, 300.0, N_TIME),
                "selfrot_deg": jnp.zeros(N_TIME),
            },
        )

    def _random_projector(self, key, **kwargs):
        re, im = jax.random.normal(key, (2, 2, self.N_ALM))
        # Valid REAL-FIELD packed alms: the m=0 coefficients (the first
        # lmax+1 packed entries) must be real, as hp.map2alm produces them.
        # Arbitrary imaginary parts there fall outside the representable
        # space and forward/adjoint are only each other's transpose on it.
        im = im.at[:, : self.LMAX + 1].set(0.0)
        return NativeLimTODProjector(
            beam_alms=re + 1j * im,
            lat_deg=self.LAT,
            lmax=self.LMAX,
            nside=self.NSIDE,
            **kwargs,
        )

    def test_validates_coordinate_requirements(self, template_state, key):
        proj = self._random_projector(key)
        with pytest.raises(StateValidationError, match="pointing"):
            proj.forward(jnp.ones((2, self.N_PIX_HP)), template_state.coords)

    def test_validates_shapes(self, key):
        with pytest.raises(StateValidationError, match="beam_alms"):
            NativeLimTODProjector(
                beam_alms=jnp.zeros((2, self.N_ALM + 1), dtype=jnp.complex64),
                lat_deg=self.LAT, lmax=self.LMAX, nside=self.NSIDE,
            )

    def test_oracle_matches_limtod_linear_chain(self, obs_coords):
        """forward == numpy generate_TOD_sky with truncate_frac_thres=0.

        The native port is the linear chain; numpy limTOD's default
        ``truncate_frac_thres=1e-10`` is a nonlinear cleanup outside the
        port contract, so the oracle disables it.

        Tolerance note: this suite runs in float32, where s2fft's healpix
        map->alm transform (Price-McEwen recursion) carries ~1% error even
        at lmax=11 — so 5e-2 here is a WIRING guard (swapped angles or a
        transposed axis show up as O(100%)). The float64 1e-6 statement is
        made by ``test_oracle_x64_subprocess`` and the limtod_jax suite.
        """
        pytest.importorskip("limtod_jax")
        limtod = pytest.importorskip("limTOD.simulator")
        hp = pytest.importorskip("healpy")
        import numpy as np

        rng = np.random.default_rng(0)
        beam_maps = rng.random((2, self.N_PIX_HP))
        sky_maps = rng.random((2, self.N_PIX_HP))
        beam_alms = jnp.asarray(
            np.stack([hp.map2alm(b, lmax=self.LMAX) for b in beam_maps])
        )
        proj = NativeLimTODProjector(
            beam_alms=beam_alms, lat_deg=self.LAT, lmax=self.LMAX, nside=self.NSIDE
        )
        out = proj.forward(jnp.asarray(sky_maps), obs_coords)
        assert out.shape == (N_TIME, 2)

        az = np.asarray(obs_coords.pointing[:, 0])
        el = np.asarray(obs_coords.pointing[:, 1])
        lst = np.asarray(obs_coords.extra["lst_deg"])
        for f in range(2):
            direct = limtod.generate_TOD_sky(
                beam_maps[f], sky_maps[f], lst, self.LAT, az, el,
                np.zeros(N_TIME), truncate_frac_thres=0.0,
            )
            assert jnp.allclose(
                out[:, f], jnp.asarray(direct), rtol=5e-2
            ), f"freq {f}: {np.max(np.abs(np.asarray(out[:, f]) - direct))}"

    def test_oracle_x64_subprocess(self):
        """Full-precision end-to-end wiring proof: 1e-6 vs the oracle in x64.

        replicant-telescope's suite runs float32 (and flipping jax_enable_x64 mid-
        process is global), so the float64 acceptance criterion of the port
        contract is checked in a fresh interpreter with JAX_ENABLE_X64=1.
        Covers the whole adapter path: coords -> zyz angles -> quadrature
        sky alms -> Wigner rotation -> TOD, plus the adjoint dot identity.
        """
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
from replicant import Coordinates
from replicant.radio.sky import NativeLimTODProjector
from limTOD.simulator import generate_TOD_sky

assert jax.config.read("jax_enable_x64")
nside, lmax, lat, n_time = {self.NSIDE}, {self.LMAX}, {self.LAT}, 8
npix = 12 * nside**2
rng = np.random.default_rng(0)
beam_maps = rng.random((2, npix))
sky_maps = rng.random((2, npix))
az = np.array([0.0, 45.0, 123.4, -42.3, 80.0, 0.0, 200.0, 10.0])
el = np.array([90.0, 60.0, 5.0, 41.0, 30.0, 90.0, 70.0, 45.0])
lst = np.linspace(0.0, 300.0, n_time)
coords = Coordinates(
    time=jnp.arange(n_time, dtype=float),
    freq=jnp.array([60e6, 70e6]),
    pointing=jnp.stack([jnp.asarray(az), jnp.asarray(el)], axis=-1),
    extra={{"lst_deg": jnp.asarray(lst)}},
)
proj = NativeLimTODProjector(
    beam_alms=jnp.asarray(np.stack([hp.map2alm(b, lmax=lmax) for b in beam_maps])),
    lat_deg=lat, lmax=lmax, nside=nside,
)
out = np.asarray(proj.forward(jnp.asarray(sky_maps), coords))
worst = 0.0
for f in range(2):
    direct = generate_TOD_sky(
        beam_maps[f], sky_maps[f], lst, lat, az, el, np.zeros(n_time),
        truncate_frac_thres=0.0,
    )
    worst = max(worst, float(np.max(np.abs(out[:, f] - direct)) / np.max(np.abs(direct))))
assert worst < 1e-6, f"oracle rel err {{worst:.3e}}"

y = jnp.asarray(rng.standard_normal((n_time, 2)))
x = jnp.asarray(sky_maps)
lhs = float(jnp.sum(proj.forward(x, coords) * y))
rhs = float(jnp.sum(x * proj.adjoint(y, coords)))
assert abs(lhs - rhs) / abs(lhs) < 1e-10, f"dot test {{lhs}} vs {{rhs}}"
print(f"X64 OK worst_rel={{worst:.3e}}")
"""
        env = dict(os.environ, JAX_ENABLE_X64="1")
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, env=env, timeout=600,
        )
        assert result.returncode == 0, result.stderr[-2000:]
        assert "X64 OK" in result.stdout

    @pytest.mark.parametrize("normalize", [False, True])
    def test_adjoint_dot_identity(self, obs_coords, key, normalize):
        """<forward(x), y> == <x, adjoint(y)> in plain pixel/TOD dots."""
        pytest.importorskip("limtod_jax")
        k1, k2, k3 = jax.random.split(key, 3)
        proj = self._random_projector(k1, normalize_beam=normalize)
        x = jax.random.normal(k2, (2, self.N_PIX_HP))
        y = jax.random.normal(k3, (N_TIME, 2))
        lhs = dot(proj.forward(x, obs_coords), y)
        rhs = dot(x, proj.adjoint(y, obs_coords))
        assert jnp.allclose(lhs, rhs, rtol=1e-4)

    def test_grad_through_forward(self, obs_coords, key):
        pytest.importorskip("limtod_jax")
        k1, k2 = jax.random.split(key)
        proj = self._random_projector(k1)
        sky = jax.random.normal(k2, (2, self.N_PIX_HP))

        g_sky = jax.grad(lambda s: jnp.sum(proj.forward(s, obs_coords) ** 2))(sky)
        assert jnp.all(jnp.isfinite(g_sky)) and jnp.any(g_sky != 0)

        def beam_loss(beam_alms):
            p = eqx.tree_at(lambda q: q.beam_alms, proj, beam_alms)
            return jnp.sum(p.forward(sky, obs_coords) ** 2)

        g_beam = jax.grad(beam_loss)(proj.beam_alms)
        assert jnp.all(jnp.isfinite(g_beam.real)) and jnp.any(g_beam != 0)

    def test_jit_forward(self, obs_coords, key):
        pytest.importorskip("limtod_jax")
        proj = self._random_projector(key)
        sky = jnp.ones((2, self.N_PIX_HP))
        f = eqx.filter_jit(proj.forward)
        out1 = f(sky, obs_coords)
        out2 = f(2.0 * sky, obs_coords)
        assert out1.shape == out2.shape == (N_TIME, 2)

    def test_skyspace_filter_composition(self, obs_coords, key):
        """The adjoint consumer: CG map-making runs and stays finite."""
        pytest.importorskip("limtod_jax")
        from replicant import State
        from replicant.radio.filters import SkySpaceFilter

        k1, k2 = jax.random.split(key)
        proj = self._random_projector(k1)
        filt = SkySpaceFilter(
            projector=proj, regularization=jnp.array(1e-2), cg_maxiter=8
        )
        data = jax.random.normal(k2, (N_TIME, 2))
        state = State(coords=obs_coords)
        projected = filt.project(data, state)
        assert projected.shape == data.shape
        assert jnp.all(jnp.isfinite(projected))
