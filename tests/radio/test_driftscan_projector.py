"""DriftScanProjector: the m-mode fast path vs the general-pointing projector.

Tolerance strategy mirrors ``TestGeneralPointingProjector``: this suite runs in
default float32, but the drift-vs-general comparison is TIGHTER than the
general-vs-numpy oracle one (1e-3 vs 5e-2) because both projectors share the
same s2fft map<->alm transforms — the f32 transform error cancels and only
the rotation kernels differ (Risbo, f32-stable). The float64 roundoff
statement (1e-6 vs numpy limTOD) lives in ``test_x64_subprocess`` and in the
limtod_jax suite.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates
from rheplicant.core.errors import StateValidationError
from rheplicant.radio.sky import DriftScanProjector, GeneralPointingProjector

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

    def _general_twin(self, proj, **kwargs):
        """GeneralPointingProjector with the same beams (the ground truth here)."""
        return GeneralPointingProjector(
            beam_alms=proj.beam_alms, lat_deg=self.LAT,
            lmax=self.LMAX, nside=self.NSIDE, **kwargs,
        )

    def _general_coords(self, drift_coords):
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
        general = self._general_twin(proj, normalize_beam=normalize)
        sky = jax.random.uniform(jax.random.key(1), (N_FREQ, self.N_PIX_HP))
        out = proj.forward(sky, drift_coords)
        ref = general.forward(sky, self._general_coords(drift_coords))
        assert out.shape == ref.shape == (N_TIME, N_FREQ)
        rel = jnp.max(jnp.abs(out - ref)) / jnp.max(jnp.abs(ref))
        assert rel < 1e-3, f"drift vs general rel err {rel:.2e}"

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
        general = self._general_twin(proj)
        tod = jax.random.normal(jax.random.key(5), (N_TIME, N_FREQ))
        ours = proj.adjoint(tod, drift_coords)
        ref = general.adjoint(tod, self._general_coords(drift_coords))
        assert ours.shape == ref.shape == (N_FREQ, self.N_PIX_HP)
        rel = jnp.max(jnp.abs(ours - ref)) / jnp.max(jnp.abs(ref))
        assert rel < 1e-3, f"adjoint drift vs general rel err {rel:.2e}"

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

    # ------------------------------------------------ ignored-input guard
    def test_accepts_agreeing_native_coords(self, key, drift_coords):
        """Reusing the general projector's coords is THE way to switch engines.

        The guard must not punish it: constant pointing that agrees with the
        projector's own is exactly the legitimate case.
        """
        proj = self._projector(key)
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))
        assert jnp.allclose(
            proj.forward(sky, self._general_coords(drift_coords)),
            proj.forward(sky, drift_coords),
        )

    @pytest.mark.parametrize(
        "kind, match",
        [("scanning_az", "azimuth"), ("wrong_el", "elevation"),
         ("varying_selfrot", "selfrot")],
    )
    def test_rejects_disagreeing_pointing(self, key, drift_coords, kind, match):
        """Coords carrying a pointing this projector would DISCARD must fail.

        Otherwise a scan handed to the drift engine silently simulates a
        different observation: finite, correctly shaped, and wrong.
        """
        proj = self._projector(key)
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))
        n_t = drift_coords.extra["lst_deg"].shape[0]
        base = self._general_coords(drift_coords)
        if kind == "scanning_az":
            bad = base.replace(pointing=base.pointing.at[:, 0].set(
                jnp.linspace(0.0, 90.0, n_t)))
        elif kind == "wrong_el":
            bad = base.replace(pointing=base.pointing.at[:, 1].set(self.EL + 20.0))
        else:
            bad = base.replace(extra={**base.extra,
                                      "selfrot_deg": jnp.linspace(0.0, 30.0, n_t)})
        with pytest.raises(StateValidationError, match=match):
            proj.forward(sky, bad)
        with pytest.raises(StateValidationError, match=match):
            proj.adjoint(jnp.ones((n_t, N_FREQ)), bad)

    def test_traced_pointing_is_not_inspected(self, key, drift_coords):
        """Tracers carry no values, so the guard steps aside instead of crashing."""
        import equinox as eqx

        proj = self._projector(key)
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))
        n_t = drift_coords.extra["lst_deg"].shape[0]

        @eqx.filter_jit
        def run(pointing):
            return proj.forward(sky, drift_coords.replace(pointing=pointing))

        scanning = jnp.stack(
            [jnp.linspace(0.0, 90.0, n_t), jnp.full(n_t, self.EL)], axis=-1
        )
        assert bool(jnp.all(jnp.isfinite(run(scanning))))

    # ------------------------------------------------------ friendly builders
    def test_from_beam_maps_uses_the_true_alm_transform(self, key, drift_coords):
        """from_beam_maps must analyse the beam the way healpy does.

        The footgun it exists to remove: the quadrature transform the SKY uses
        is off by npix/4pi for the beam, so the wrong choice is silently
        wrong rather than loudly broken.
        """
        import numpy as np

        hp = pytest.importorskip("healpy")
        ltj = pytest.importorskip("limtod_jax")
        rng = jax.random.normal(key, (N_FREQ, self.N_PIX_HP))
        beam_maps = jnp.exp(-0.5 * rng**2)  # positive, beam-like
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))

        proj = DriftScanProjector.from_beam_maps(
            beam_maps, lat_deg=self.LAT, az_deg=self.AZ, el_deg=self.EL,
            lmax=self.LMAX, selfrot_deg=self.SELFROT,
        )
        assert proj.nside == self.NSIDE  # inferred from the map length

        healpy_alms = jnp.asarray(
            [hp.map2alm(b, lmax=self.LMAX, iter=3) for b in np.asarray(beam_maps)]
        )
        reference = DriftScanProjector(
            beam_alms=healpy_alms, lat_deg=self.LAT, az_deg=self.AZ,
            el_deg=self.EL, lmax=self.LMAX, nside=self.NSIDE,
            selfrot_deg=self.SELFROT,
        )
        assert jnp.allclose(proj.forward(sky, drift_coords),
                            reference.forward(sky, drift_coords), rtol=5e-2)

        # ... and the transform the sky uses would NOT have done (npix/4pi).
        quad = DriftScanProjector(
            beam_alms=jax.vmap(
                lambda m: ltj.map2alm_quad(m, nside=self.NSIDE, lmax=self.LMAX)
            )(beam_maps),
            lat_deg=self.LAT, az_deg=self.AZ, el_deg=self.EL,
            lmax=self.LMAX, nside=self.NSIDE, selfrot_deg=self.SELFROT,
        )
        assert not jnp.allclose(quad.forward(sky, drift_coords),
                                proj.forward(sky, drift_coords), rtol=0.5)

    def test_from_beam_maps_is_differentiable_in_the_map(self, key, drift_coords):
        """The point of doing the analysis in JAX: gradients reach the beam MAP."""
        pytest.importorskip("limtod_jax")
        beam_maps = jnp.exp(-0.5 * jax.random.normal(key, (N_FREQ, self.N_PIX_HP)) ** 2)
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))

        def loss(maps):
            proj = DriftScanProjector.from_beam_maps(
                maps, lat_deg=self.LAT, az_deg=self.AZ, el_deg=self.EL,
                lmax=self.LMAX,
            )
            return jnp.sum(proj.forward(sky, drift_coords) ** 2)

        grad = jax.grad(loss)(beam_maps)
        assert grad.shape == beam_maps.shape
        assert bool(jnp.all(jnp.isfinite(grad))) and bool(jnp.any(grad != 0.0))

    def test_from_beam_maps_validation(self):
        with pytest.raises(StateValidationError, match="n_freq, n_pix"):
            DriftScanProjector.from_beam_maps(
                jnp.ones(self.N_PIX_HP), lat_deg=self.LAT, az_deg=self.AZ,
                el_deg=self.EL, lmax=self.LMAX,
            )
        with pytest.raises(StateValidationError, match="HEALPix"):
            DriftScanProjector.from_beam_maps(
                jnp.ones((N_FREQ, 100)), lat_deg=self.LAT, az_deg=self.AZ,
                el_deg=self.EL, lmax=self.LMAX,
            )

    def test_uniform_lst_grid_satisfies_the_fft_contract(self, key):
        """The helper's grid is accepted; the natural linspace mistake is not."""
        from rheplicant import Coordinates

        pytest.importorskip("limtod_jax")
        n_t = 4 * (self.LMAX + 1)
        proj = self._projector(key, uniform_sampling=True)
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))
        base = Coordinates(time=jnp.arange(float(n_t)), freq=jnp.ones(N_FREQ))

        good = DriftScanProjector.uniform_lst_grid(n_t, lst0_deg=12.0)
        assert good.shape == (n_t,)
        out = proj.forward(sky, base.replace(extra={"lst_deg": good}))
        assert bool(jnp.all(jnp.isfinite(out)))

        # linspace INCLUDES the endpoint: a turn plus one step, not a turn.
        endpoint = jnp.linspace(12.0, 372.0, n_t)
        with pytest.raises(StateValidationError, match="uniform_lst_grid"):
            proj.forward(sky, base.replace(extra={"lst_deg": endpoint}))

    # ------------------------------------- cached reference beam + FFT sampling
    @pytest.fixture
    def uniform_coords(self, coords):
        """A uniform full-turn LST grid with 2*lmax < n_time (FFT contract)."""
        n_t = 4 * (self.LMAX + 1)
        return coords.replace(
            freq=coords.freq[:N_FREQ],
            extra={"lst_deg": jnp.asarray(
                12.0 + 360.0 * jnp.arange(n_t) / n_t
            )},
        )

    # A reference LST deliberately UNLIKE lst_deg[0] (= 12.0 in both fixtures):
    # with ref == lst[0] the projector cannot distinguish self.lst_ref_deg from
    # coords lst[0], so a "use lst[0] regardless" bug is invisible — the exact
    # degeneracy that let an equivalent mutation survive limTOD's suite once.
    REF_OFF = 237.5

    @pytest.mark.parametrize("lst_ref", [12.0, REF_OFF])
    def test_to_reference_frame_is_equivalent(self, key, drift_coords, lst_ref):
        """Caching the rotation must not change any output, and must actually
        remove the rotation (the cached alms differ from the beam-local ones).

        Run at a reference LST equal to AND far from the first sample: the TOD
        is reference-independent, so both must also match the general
        GeneralPointingProjector, which knows nothing about m-mode references."""
        proj = self._projector(key, lst_ref_deg=lst_ref)
        cached = proj.to_reference_frame()
        assert cached.beam_frame == "reference" and proj.beam_frame == "local"
        assert not jnp.allclose(cached.beam_alms, proj.beam_alms)  # rotation baked in

        sky = jax.random.uniform(jax.random.key(20), (N_FREQ, self.N_PIX_HP))
        tod = jax.random.normal(jax.random.key(21), (N_TIME, N_FREQ))
        for a, b in (
            (proj.forward(sky, drift_coords), cached.forward(sky, drift_coords)),
            (proj.adjoint(tod, drift_coords), cached.adjoint(tod, drift_coords)),
            (proj.mmodes(sky, drift_coords), cached.mmodes(sky, drift_coords)),
        ):
            rel = jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a))
            assert rel < 1e-5, f"cached path diverged: {rel:.2e}"

        # independent ground truth: the reference choice is a gauge, so both
        # the local and the cached projector must reproduce the general path
        general = self._general_twin(proj)
        ref_tod = general.forward(sky, self._general_coords(drift_coords))
        for got in (proj.forward(sky, drift_coords), cached.forward(sky, drift_coords)):
            rel = jnp.max(jnp.abs(got - ref_tod)) / jnp.max(jnp.abs(ref_tod))
            assert rel < 1e-3, f"lst_ref={lst_ref} broke the gauge: {rel:.2e}"

    def test_to_reference_frame_folds_in_the_mask(self, key, drift_coords):
        """A masked local projector must cache the MASKED beam (and clear the
        flag so the mask is not applied twice)."""
        import dataclasses

        masked = dataclasses.replace(
            self._projector(key), el_deg=10.0, horizon_mask=True, apod_deg=5.0,
            lst_ref_deg=12.0,
        )
        cached = masked.to_reference_frame()
        assert cached.horizon_mask is False  # folded into the alms
        sky = jax.random.uniform(jax.random.key(22), (N_FREQ, self.N_PIX_HP))
        rel = jnp.max(
            jnp.abs(masked.forward(sky, drift_coords) - cached.forward(sky, drift_coords))
        ) / jnp.max(jnp.abs(masked.forward(sky, drift_coords)))
        assert rel < 1e-5, f"mask lost or double-applied: {rel:.2e}"

    def test_to_reference_frame_is_differentiable(self, key, drift_coords):
        """Full-JAX principle: the precompute is itself traceable, so a caller
        that DOES want beam-local gradients can differentiate through it — and
        gets the SAME gradient as the uncached projector. Asserting only
        finite-and-nonzero would pass for any wrong-but-finite gradient, so the
        equality against the local path is the load-bearing check."""
        import dataclasses

        proj = self._projector(key, lst_ref_deg=self.REF_OFF)
        sky = jax.random.uniform(jax.random.key(23), (N_FREQ, self.N_PIX_HP))

        def loss_cached(beams):
            p = dataclasses.replace(proj, beam_alms=beams)
            return jnp.sum(p.to_reference_frame().forward(sky, drift_coords) ** 2)

        def loss_local(beams):
            p = dataclasses.replace(proj, beam_alms=beams)
            return jnp.sum(p.forward(sky, drift_coords) ** 2)

        g = jax.grad(loss_cached, holomorphic=False)(proj.beam_alms)
        g_local = jax.grad(loss_local, holomorphic=False)(proj.beam_alms)
        assert bool(jnp.all(jnp.isfinite(jnp.abs(g))))
        assert float(jnp.max(jnp.abs(g))) > 0.0
        rel = jnp.max(jnp.abs(g - g_local)) / jnp.max(jnp.abs(g_local))
        assert rel < 1e-4, f"beam-local gradient changed by caching: {rel:.2e}"

    def test_reference_frame_validation(self, key):
        import dataclasses

        proj = self._projector(key)
        with pytest.raises(StateValidationError, match="lst_ref_deg"):
            dataclasses.replace(proj, beam_frame="reference")
        # isolate the horizon_mask guard: give a CONSISTENT anchor so the
        # (earlier, more fundamental) anchor invariant does not fire instead
        with pytest.raises(StateValidationError, match="horizon_mask"):
            dataclasses.replace(
                proj, beam_frame="reference", lst_ref_deg=12.0,
                beam_ref_lst_deg=12.0, horizon_mask=True,
            )
        with pytest.raises(StateValidationError, match="beam_frame"):
            dataclasses.replace(proj, beam_frame="celestial")
        with pytest.raises(StateValidationError, match="reference LST"):
            proj.to_reference_frame()  # no lst_ref_deg anywhere
        cached = proj.to_reference_frame(lst_ref_deg=12.0)
        assert cached.to_reference_frame() is cached  # idempotent
        with pytest.raises(StateValidationError, match="already cached"):
            cached.to_reference_frame(lst_ref_deg=99.0)

    def test_cached_beam_cannot_be_re_anchored_by_field_edit(self, key, drift_coords):
        """Editing lst_ref_deg on a cached projector would measure the phases
        from a reference the baked-in rotation does not correspond to — a
        silently wrong TOD. The beam_ref_lst_deg pair makes it loud, and
        going through to_reference_frame on the LOCAL projector is the way."""
        import dataclasses

        proj = self._projector(key, lst_ref_deg=12.0)
        cached = proj.to_reference_frame()
        assert cached.beam_ref_lst_deg == cached.lst_ref_deg == 12.0

        with pytest.raises(StateValidationError, match="anchored at"):
            dataclasses.replace(cached, lst_ref_deg=self.REF_OFF)
        # dropping the reference entirely trips the earlier explicit-ref guard
        with pytest.raises(StateValidationError, match="requires an explicit"):
            dataclasses.replace(cached, lst_ref_deg=None, beam_frame="reference")
        # a hand-built "reference" projector without the anchor is rejected too
        with pytest.raises(StateValidationError, match="anchored at"):
            dataclasses.replace(proj, beam_frame="reference", lst_ref_deg=12.0)

        # the legitimate route works and agrees with the local path
        re_anchored = proj.to_reference_frame(lst_ref_deg=self.REF_OFF)
        sky = jax.random.uniform(jax.random.key(31), (N_FREQ, self.N_PIX_HP))
        local = dataclasses.replace(proj, lst_ref_deg=self.REF_OFF)
        a, b = re_anchored.forward(sky, drift_coords), local.forward(sky, drift_coords)
        assert float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))) < 1e-5

    @pytest.mark.parametrize("lst_ref", [12.0, REF_OFF])
    @pytest.mark.parametrize("normalize", [False, True])
    def test_uniform_sampling_matches_direct(
        self, key, uniform_coords, normalize, lst_ref
    ):
        """The FFT route must agree with the direct sum, forward and adjoint.

        Also run with a reference LST far from the first sample: shifting the
        reference only shifts phase0, so the grid stays uniform and the result
        must be unchanged — otherwise phase0 is being mishandled."""
        import dataclasses

        proj = self._projector(key, normalize_beam=normalize, lst_ref_deg=lst_ref)
        fft = dataclasses.replace(proj, uniform_sampling=True)
        n_t = uniform_coords.extra["lst_deg"].shape[0]
        sky = jax.random.uniform(jax.random.key(24), (N_FREQ, self.N_PIX_HP))
        tod = jax.random.normal(jax.random.key(25), (n_t, N_FREQ))
        for a, b in (
            (proj.forward(sky, uniform_coords), fft.forward(sky, uniform_coords)),
            (proj.adjoint(tod, uniform_coords), fft.adjoint(tod, uniform_coords)),
        ):
            rel = jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(a))
            assert rel < 1e-4, f"FFT route diverged: {rel:.2e}"

    def test_uniform_sampling_adjoint_dot_identity(self, key, uniform_coords):
        import dataclasses

        fft = dataclasses.replace(
            self._projector(key, lst_ref_deg=12.0), uniform_sampling=True
        )
        n_t = uniform_coords.extra["lst_deg"].shape[0]
        sky = jax.random.uniform(jax.random.key(26), (N_FREQ, self.N_PIX_HP))
        tod = jax.random.normal(jax.random.key(27), (n_t, N_FREQ))
        lhs = dot(fft.forward(sky, uniform_coords), tod)
        rhs = dot(sky, fft.adjoint(tod, uniform_coords))
        assert jnp.abs(lhs - rhs) / jnp.abs(lhs) < 1e-3

    def test_uniform_sampling_rejects_irregular_grid(self, key, uniform_coords):
        """An irregular grid must fail loudly rather than silently return a
        wrong TOD. Long enough (2*lmax < n_time) that the Nyquist guard is not
        what fires — this is the uniformity check itself."""
        import dataclasses

        fft = dataclasses.replace(
            self._projector(key, lst_ref_deg=12.0), uniform_sampling=True
        )
        lst = uniform_coords.extra["lst_deg"]
        jittered = uniform_coords.replace(
            extra={"lst_deg": lst.at[len(lst) // 3].add(3.0)}  # one bad sample
        )
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))
        with pytest.raises(ValueError, match="uniform grid"):
            fft.forward(sky, jittered)
        # a uniform grid covering only HALF a turn is equally invalid
        half = uniform_coords.replace(
            extra={"lst_deg": 12.0 + 180.0 * jnp.arange(len(lst)) / len(lst)}
        )
        with pytest.raises(ValueError, match="uniform grid"):
            fft.forward(sky, half)

    def test_uniform_sampling_rejects_bad_grid_under_jit(self, key, uniform_coords):
        """The regression that matters most: an eager-only guard was bypassed
        by ANY jit wrapping — including a compile-time-constant grid, because
        deriving dphi inside a trace produces a tracer — and a uniform
        HALF-turn grid (the normal shape of a real observation) then returned
        a silently 74%-wrong TOD. The adapter now validates the RAW lst_deg,
        which is still concrete there, so this raises at trace time."""
        import dataclasses

        import equinox as eqx

        proj = self._projector(key, lst_ref_deg=12.0)
        fft = dataclasses.replace(proj, uniform_sampling=True)
        n_t = uniform_coords.extra["lst_deg"].shape[0]
        half = uniform_coords.replace(
            extra={"lst_deg": 12.0 + 180.0 * jnp.arange(n_t) / n_t}
        )
        sky = jax.random.uniform(jax.random.key(30), (N_FREQ, self.N_PIX_HP))

        # Layer 1 — coords closed over (the normal likelihood pattern): the
        # RAW lst_deg is still concrete inside the trace, so this is a clear
        # error at trace time, before anything compiles.
        with pytest.raises(ValueError, match="uniform grid"):
            jax.jit(lambda s: fft.forward(s, half))(sky)
        with pytest.raises(ValueError, match="uniform grid"):
            jax.jit(lambda t: fft.adjoint(t, half))(jnp.ones((n_t, N_FREQ)))

        # Layer 2 — coords passed as a TRACED argument: the values genuinely
        # cannot be read, so no exception is possible. limtod_jax's pure-JAX
        # guard makes the result NaN instead of a plausible wrong TOD.
        out_bad = eqx.filter_jit(lambda p, s, c: p.forward(s, c))(fft, sky, half)
        assert bool(jnp.all(jnp.isnan(out_bad))), "traced violation must poison"

        # the valid grid still compiles and matches the direct sum, both ways
        ref = proj.forward(sky, uniform_coords)
        for out in (
            jax.jit(lambda s: fft.forward(s, uniform_coords))(sky),
            eqx.filter_jit(lambda p, s, c: p.forward(s, c))(fft, sky, uniform_coords),
        ):
            assert float(jnp.max(jnp.abs(out - ref)) / jnp.max(jnp.abs(ref))) < 1e-4

    def test_mmodes_rejects_normalize_beam(self, key, drift_coords):
        """mmodes() is the spectrum of the UN-normalized TOD; combined with
        normalize_beam=True it would silently not be the spectrum of
        forward() (measured ~18x off), so the combination is rejected."""
        proj = self._projector(key, normalize_beam=True)
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))
        with pytest.raises(StateValidationError, match="normalize_beam"):
            proj.mmodes(sky, drift_coords)
        # and forward/adjoint still work with normalization
        assert jnp.all(jnp.isfinite(proj.forward(sky, drift_coords)))

    def test_fast_combo_jit_vmap_grad(self, key, uniform_coords):
        """Both opt-ins together must stay jit/vmap/grad-safe (the rheplicant
        full-JAX guarantee) and still match the plain projector."""
        import dataclasses

        import equinox as eqx

        plain = self._projector(key, lst_ref_deg=12.0)
        fast = dataclasses.replace(
            plain, uniform_sampling=True
        ).to_reference_frame()
        sky = jax.random.uniform(jax.random.key(28), (N_FREQ, self.N_PIX_HP))

        # Max-norm relative comparisons throughout: a TOD crosses zero, and an
        # ELEMENTWISE rtol (jnp.allclose) is unsatisfiable there — the error
        # floor follows the array scale, not the individual element.
        def relmax(a, b):
            return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))

        ref = plain.forward(sky, uniform_coords)
        rel = relmax(fast.forward(sky, uniform_coords), ref)
        assert rel < 1e-4, f"fast combo diverged: {rel:.2e}"  # measured 7e-7 in f32

        out_jit = eqx.filter_jit(lambda p, s: p.forward(s, uniform_coords))(fast, sky)
        # measured 1e-7 in f32 — identical to the plain projector's jit delta,
        # i.e. ordinary fusion noise, nothing FFT-specific
        assert relmax(out_jit, fast.forward(sky, uniform_coords)) < 1e-5

        skies = jnp.stack([sky, 2.0 * sky])
        batched = jax.vmap(lambda s: fast.forward(s, uniform_coords))(skies)
        assert batched.shape == (2,) + ref.shape
        assert relmax(batched[1], 2.0 * batched[0]) < 1e-5

        g = jax.grad(lambda s: jnp.sum(fast.forward(s, uniform_coords) ** 2))(sky)
        assert bool(jnp.all(jnp.isfinite(g)))

    @pytest.mark.parametrize("n_t", [4 * (LMAX + 1), 2 * LMAX + 1, 257])
    @pytest.mark.parametrize("lst0", [0.0, 12.0, 237.5])
    def test_uniform_lst_grid_satisfies_the_contract(self, key, n_t, lst0):
        """The helper's whole reason to exist: whatever it returns must be
        accepted by the FFT path, and the endpoint-inclusive grid people reach
        for instead must be rejected. Odd and even n_t, at the Nyquist
        boundary, and with a nonzero first LST."""
        import dataclasses

        import limtod_jax as ltj
        import numpy as np

        grid = DriftScanProjector.uniform_lst_grid(n_t, lst0)
        assert grid.shape == (n_t,)
        assert float(grid[0]) == pytest.approx(lst0)
        # a FULL turn: the excluded endpoint means the last sample is one step
        # short of lst0 + 360, so the wrap-around gap equals every other gap.
        # Tolerance follows the GRID's dtype, not float64: this suite runs in
        # float32, where a ~360-degree grid carries ~360*eps of representation
        # error and steps like 360/23 are not representable at all (360/48 is,
        # which is why an f64-shaped rtol passes for some n_t and not others).
        eps = float(np.finfo(np.asarray(grid).dtype).eps)
        atol = 8.0 * eps * 360.0
        steps = np.diff(np.asarray(grid, dtype=np.float64))
        wrap = lst0 + 360.0 - float(grid[-1])
        np.testing.assert_allclose(steps, 360.0 / n_t, atol=atol, rtol=0)
        assert wrap == pytest.approx(360.0 / n_t, abs=atol)

        # limtod_jax must accept it, and reject linspace's endpoint-inclusive
        # sibling — the regression this helper exists to prevent
        ltj.check_uniform_grid(np.deg2rad(np.asarray(grid) - lst0))
        endpoint_inclusive = np.linspace(lst0, lst0 + 360.0, n_t)
        with pytest.raises(ValueError, match="uniform grid"):
            ltj.check_uniform_grid(np.deg2rad(endpoint_inclusive - lst0))

        # and end to end through the projector, where 2*lmax < n_t allows it
        if 2 * self.LMAX < n_t:
            fft = dataclasses.replace(
                self._projector(key, lst_ref_deg=lst0), uniform_sampling=True
            )
            plain = dataclasses.replace(fft, uniform_sampling=False)
            coords = Coordinates(
                time=jnp.arange(n_t, dtype=float),
                freq=jnp.array([60e6, 70e6]),
                extra={"lst_deg": grid},
            )
            sky = jax.random.uniform(jax.random.key(40), (N_FREQ, self.N_PIX_HP))
            a, b = fft.forward(sky, coords), plain.forward(sky, coords)
            assert float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))) < 1e-4
            bad = coords.replace(extra={"lst_deg": jnp.asarray(endpoint_inclusive)})
            with pytest.raises(ValueError, match="uniform grid"):
                fft.forward(sky, bad)

    def test_uniform_sampling_requires_limtod_17(self, key, uniform_coords, monkeypatch):
        """An outdated limTOD must fail at the boundary with a clear message,
        not with an AttributeError deep inside a traced call: the FFT path and
        the public grid check only exist from limTOD 1.7."""
        import dataclasses

        import limtod_jax

        fft = dataclasses.replace(
            self._projector(key, lst_ref_deg=12.0), uniform_sampling=True
        )
        sky = jnp.ones((N_FREQ, self.N_PIX_HP))
        monkeypatch.delattr(limtod_jax, "check_uniform_grid", raising=False)
        with pytest.raises(ImportError, match="limTOD 1.7"):
            fft.forward(sky, uniform_coords)
        # the plain path is unaffected by the missing 1.7 symbol
        plain = dataclasses.replace(fft, uniform_sampling=False)
        assert jnp.all(jnp.isfinite(plain.forward(sky, uniform_coords)))

    # ------------------------------------------------------ float64 statement
    def test_x64_subprocess(self):
        """Roundoff-level statement in a fresh x64 interpreter: forward ==
        numpy limTOD generate_TOD_sky on the drift scan to 1e-6, and the
        adjoint dot identity to 1e-10 (mirrors the general projector's
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

    # ------------------------------------------------- pre-analysed sky alms
    def test_forward_alms_equals_forward(self, key, drift_coords):
        """The alm entry point must be the same operator, not a variant.

        forward() is now a wrapper over it, so a discrepancy here would mean
        the hoist changed the physics rather than just when the analysis runs.
        """
        proj = self._projector(key)
        sky = jax.random.uniform(jax.random.key(20), (N_FREQ, self.N_PIX_HP))
        alms = proj.sky_to_alms(sky)
        assert alms.shape == (N_FREQ, self.N_ALM)
        assert jnp.allclose(
            proj.forward_alms(alms, drift_coords),
            proj.forward(sky, drift_coords),
            rtol=1e-6,
        )
        assert jnp.allclose(
            proj.mmodes_alms(alms, drift_coords),
            proj.mmodes(sky, drift_coords),
            rtol=1e-6,
        )

    def test_forward_alms_is_differentiable_in_the_alms(self, key, drift_coords):
        """The point of the entry point: fit against a sky held in harmonic
        space without paying the analysis on every evaluation."""
        proj = self._projector(key)
        sky = jax.random.uniform(jax.random.key(21), (N_FREQ, self.N_PIX_HP))
        alms = proj.sky_to_alms(sky)

        def loss(a):
            return jnp.sum(jnp.abs(proj.forward_alms(a, drift_coords)) ** 2)

        grad = jax.grad(loss)(alms)
        assert grad.shape == alms.shape
        assert bool(jnp.all(jnp.isfinite(jnp.abs(grad)))) and bool(jnp.any(grad != 0))

    def test_sky_alms_shape_is_validated(self, key, drift_coords):
        """Quadrature alms of the wrong length must not be silently broadcast."""
        proj = self._projector(key)
        with pytest.raises(StateValidationError, match="sky_alms"):
            proj.forward_alms(
                jnp.zeros((N_FREQ, self.N_ALM + 1), dtype=jnp.complex64), drift_coords
            )
        with pytest.raises(StateValidationError, match="sky_alms"):
            proj.mmodes_alms(
                jnp.zeros((N_FREQ + 1, self.N_ALM), dtype=jnp.complex64), drift_coords
            )

    # ------------------------------------------------------ frequency chunking
    @pytest.mark.parametrize("chunk", [1, 2])
    def test_freq_chunk_changes_nothing_but_the_schedule(self, key, drift_coords, chunk):
        """Chunking is a memory/time trade; the numbers must be untouched —
        forward, adjoint and mmodes alike, since a map-making iteration mixes
        the first two and would otherwise drift.

        Compared on the scale of the array, as everywhere else in this file:
        vmap and lax.map reduce in different orders, so elements near zero
        differ at f32 roundoff without anything being wrong.
        """
        import dataclasses

        proj = self._projector(key)
        chunked = dataclasses.replace(proj, freq_chunk=chunk)
        sky = jax.random.uniform(jax.random.key(30), (N_FREQ, self.N_PIX_HP))
        tod = jax.random.normal(jax.random.key(31), (N_TIME, N_FREQ))
        alms = proj.sky_to_alms(sky)
        for a, b in (
            (chunked.forward_alms(alms, drift_coords),
             proj.forward_alms(alms, drift_coords)),
            (chunked.adjoint(tod, drift_coords), proj.adjoint(tod, drift_coords)),
            (chunked.mmodes_alms(alms, drift_coords),
             proj.mmodes_alms(alms, drift_coords)),
        ):
            rel = jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b))
            assert rel < 1e-5, f"freq_chunk={chunk} moved values by {rel:.2e}"
