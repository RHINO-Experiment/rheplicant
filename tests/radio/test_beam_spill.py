"""BeamSpillOperator and the horizon fraction it applies.

The headline test is ``test_the_split_reproduces_a_painted_ground_sky``: at
latitude 90 the local horizon coincides with the celestial equator and stops
moving with LST, so a celestial map can hold the ground and the projector can
be asked for the answer directly. Everything else here exists to pin the three
conventions that make that closure work.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("limtod_jax", reason="limTOD[jax] not installed")
ltj = pytest.importorskip("limtod_jax")
hp = pytest.importorskip("healpy")

from rheplicant import Coordinates, State  # noqa: E402
from rheplicant.core.errors import StateValidationError  # noqa: E402
from rheplicant.radio import (  # noqa: E402
    AtmosphericEmissionOperator,
    BeamSpillOperator,
    CalLoadOperator,
    GroundPickupOperator,
    SkyOperator,
    SkySourceOperator,
    assemble,
)
from rheplicant.radio.sky import DriftScanProjector  # noqa: E402
from rheplicant.radio.sky.model import AbstractSkyModel  # noqa: E402

N_TIME = 6
T_SKY, T_GROUND = 3000.0, 290.0
X64 = jax.config.read("jax_enable_x64")


class MapSky(AbstractSkyModel):
    maps: jax.Array

    def __call__(self, freq: jax.Array) -> jax.Array:
        return self.maps


def gaussian_beam(nside: int, fwhm_sigma_deg: float = 35.0, floor: float = 0.02):
    """A main lobe plus a sidelobe floor, so the below-horizon share is real."""
    theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    beam = np.exp(-0.5 * (theta / np.deg2rad(fwhm_sigma_deg)) ** 2) + floor
    return jnp.asarray(beam)[None, :], theta


def polar_projector(nside: int, beam, *, horizon_mask: bool, apod_deg: float = 0.0):
    """Latitude 90, zenith pointing: the horizon IS the celestial equator."""
    return DriftScanProjector.from_beam_maps(
        beam, lat_deg=90.0, az_deg=0.0, el_deg=90.0, lmax=3 * nside - 1,
        normalize_beam=True, horizon_mask=horizon_mask, apod_deg=apod_deg,
    )


def polar_coords(nside: int):
    return Coordinates(
        time=jnp.arange(float(N_TIME)), freq=jnp.array([70e6]),
        extra={"lst_deg": 360.0 * jnp.arange(N_TIME) / N_TIME},
    )


def painted_sky(theta):
    """T_SKY above, T_GROUND below, and the horizon ring HALF of each.

    A pixel centred exactly on the horizon is half sky and half ground; a
    reference that gave it entirely to one side would be testing a convention
    rather than the physics.
    """
    on_horizon = np.isclose(theta, np.pi / 2)
    painted = np.where(theta < np.pi / 2 - 1e-9, T_SKY, T_GROUND)
    painted[on_horizon] = 0.5 * (T_SKY + T_GROUND)
    return jnp.asarray(painted)[None, :]


class TestClosureAgainstPaintedGround:
    @pytest.mark.parametrize("nside", [8, 16])
    def test_the_split_reproduces_a_painted_ground_sky(self, nside):
        """The whole construction, end to end: masked sky average + f_sky
        weight + ground must equal what the unmasked beam collects when the
        ground is actually in the map. Closes a ~200 K bias to milli-kelvin."""
        beam, theta = gaussian_beam(nside)
        coords = polar_coords(nside)
        uniform = jnp.full((1, hp.nside2npix(nside)), T_SKY)

        exact = polar_projector(nside, beam, horizon_mask=False).forward(
            painted_sky(theta), coords
        )
        masked = polar_projector(nside, beam, horizon_mask=True).forward(
            uniform, coords
        )
        spill = BeamSpillOperator.from_projector(
            polar_projector(nside, beam, horizon_mask=False),
            t_ground=jnp.array(T_GROUND),
        )
        modelled = spill(State(data=masked, coords=coords)).data
        assert float(jnp.max(jnp.abs(modelled - exact))) < 0.1

    def test_ignoring_the_spill_is_a_two_hundred_kelvin_error(self):
        """What the split is worth: the same configuration with no split at
        all. Without this the closure above could be passing on a beam whose
        spill is negligible."""
        nside = 8
        beam, theta = gaussian_beam(nside)
        coords = polar_coords(nside)
        exact = polar_projector(nside, beam, horizon_mask=False).forward(
            painted_sky(theta), coords
        )
        naive = polar_projector(nside, beam, horizon_mask=False).forward(
            jnp.full((1, hp.nside2npix(nside)), T_SKY), coords
        )
        assert float(jnp.mean(naive - exact)) > 150.0


class TestTheHorizonRingCountsHalf:
    """``horizon_weights`` uses a strict ``el > 0``; the ring on the horizon is
    half sky and half ground, and getting that wrong is the dominant error."""

    def test_ring_ordering_mirrors_the_hemispheres_exactly(self):
        """The identity the half-weight rests on: in RING ordering the
        strict-below indicator is the strict-above one reversed."""
        for nside in (4, 8, 16):
            above = np.asarray(ltj.horizon_weights(nside, 0.0))
            theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
            below = (theta > np.pi / 2 + 1e-12).astype(float)
            np.testing.assert_array_equal(above[::-1], below)

    def test_the_strict_cut_drops_a_whole_ring(self):
        nside = 16
        above = np.asarray(ltj.horizon_weights(nside, 0.0))
        theta, _ = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        on_horizon = np.isclose(theta, np.pi / 2)
        assert on_horizon.sum() == 4 * nside
        assert np.all(above[on_horizon] == 0.0), "strict el > 0 drops the ring"

        half = 0.5 * (above + 1.0 - above[::-1])
        np.testing.assert_array_equal(np.unique(half), [0.0, 0.5, 1.0])
        assert np.all(half[on_horizon] == 0.5)

    @pytest.mark.parametrize("nside", [8, 16])
    def test_the_one_sided_conventions_are_symmetrically_wrong(self, nside):
        """Counting the ring as all-sky or as nothing costs the same in
        opposite directions, and the error halves with nside -- the signature
        of a miscounted ring rather than of anything harmonic. The half-weight
        that ``horizon_fraction`` uses sits between them, at ~0."""
        beam, theta = gaussian_beam(nside)
        coords = polar_coords(nside)
        uniform = jnp.full((1, hp.nside2npix(nside)), T_SKY)
        exact = float(polar_projector(nside, beam, horizon_mask=False)
                      .forward(painted_sky(theta), coords).mean())
        masked = float(polar_projector(nside, beam, horizon_mask=True)
                       .forward(uniform, coords).mean())

        beam_map = np.asarray(ltj.alm2map(
            ltj.map2alm_iter(beam[0], nside=nside, lmax=3 * nside - 1),
            nside=nside, lmax=3 * nside - 1,
        ))
        above = np.asarray(ltj.horizon_weights(nside, 0.0))
        on = np.isclose(theta, np.pi / 2)

        def bias(ring_weight):
            w = np.where(on, ring_weight, above)
            f = float((beam_map * w).sum() / beam_map.sum())
            return f * masked + (1.0 - f) * T_GROUND - exact

        strict, inclusive = bias(0.0), bias(1.0)
        assert strict < -1.0 and inclusive > 1.0
        assert abs(strict + inclusive) < 0.2 * abs(strict), "should be symmetric"

        f_used = float(polar_projector(nside, beam, horizon_mask=False)
                       .horizon_fraction()[0])
        assert abs(f_used * masked + (1.0 - f_used) * T_GROUND - exact) < 0.1


class TestHorizonFraction:
    def test_a_narrow_zenith_beam_sees_almost_only_sky(self):
        nside = 8
        beam, _ = gaussian_beam(nside, fwhm_sigma_deg=8.0, floor=0.0)
        f = polar_projector(nside, beam, horizon_mask=False).horizon_fraction()
        assert 0.99 < float(f[0]) <= 1.0

    def test_an_isotropic_beam_splits_the_sphere_in_half(self):
        """No harmonic subtlety left: half the solid angle is above the
        horizon, exactly, and the equal-area pixel partition must say so."""
        nside = 8
        beam = jnp.ones((1, hp.nside2npix(nside)))
        f = polar_projector(nside, beam, horizon_mask=False).horizon_fraction()
        assert abs(float(f[0]) - 0.5) < 1e-6

    def test_apodization_does_not_move_it(self):
        """f_sky is a partition of the sphere; a tapered region is not one.
        apod_deg belongs to the masked sky AVERAGE, not here."""
        nside = 8
        beam, _ = gaussian_beam(nside)
        sharp = polar_projector(nside, beam, horizon_mask=True, apod_deg=0.0)
        taper = polar_projector(nside, beam, horizon_mask=True, apod_deg=5.0)
        assert jnp.allclose(sharp.horizon_fraction(), taper.horizon_fraction())

    def test_it_is_refused_on_a_cached_reference_frame_projector(self):
        nside = 8
        beam, _ = gaussian_beam(nside)
        cached = polar_projector(nside, beam, horizon_mask=True).to_reference_frame(
            lst_ref_deg=0.0
        )
        with pytest.raises(StateValidationError, match="beam_frame='reference'"):
            cached.horizon_fraction()

    def test_from_projector_refuses_a_projector_without_the_method(self):
        class NoHorizon:
            pass

        with pytest.raises(StateValidationError, match="horizon_fraction"):
            BeamSpillOperator.from_projector(NoHorizon(), t_ground=jnp.array(290.0))


class TestTheOperator:
    @pytest.fixture
    def coords(self):
        return Coordinates(time=jnp.arange(float(N_TIME)),
                           freq=jnp.linspace(60e6, 85e6, 4))

    def test_no_spill_is_the_identity(self, coords):
        data = jnp.full((N_TIME, 4), 1234.0)
        op = BeamSpillOperator(sky_fraction=jnp.array(1.0),
                               t_ground=jnp.array(T_GROUND))
        assert jnp.array_equal(op(State(data=data, coords=coords)).data, data)

    @pytest.mark.parametrize("fraction", [0.0, 0.5, 0.9, 1.0])
    def test_it_mixes_without_losing_anything(self, coords, fraction):
        """A spill is a mixture, not a loss: with sky and ground at the same
        temperature the output is that temperature for any fraction. This is
        what separates it from AntennaLossOperator's identical arithmetic --
        there the second term is the antenna's own emission, here it is another
        part of the same sky-plus-ground scene."""
        data = jnp.full((N_TIME, 4), T_GROUND)
        op = BeamSpillOperator(sky_fraction=jnp.array(fraction),
                               t_ground=jnp.array(T_GROUND))
        assert jnp.allclose(op(State(data=data, coords=coords)).data, T_GROUND)

    def test_a_per_channel_fraction_acts_per_channel(self, coords):
        f = jnp.linspace(0.90, 1.00, 4)
        data = jnp.full((N_TIME, 4), T_SKY)
        op = BeamSpillOperator(sky_fraction=f, t_ground=jnp.array(T_GROUND))
        expected = f * T_SKY + (1.0 - f) * T_GROUND
        assert jnp.allclose(op(State(data=data, coords=coords)).data,
                            expected[None, :])

    def test_a_fraction_with_the_wrong_channel_count_is_refused(self, coords):
        op = BeamSpillOperator(sky_fraction=jnp.ones(5),
                               t_ground=jnp.array(T_GROUND))
        with pytest.raises(StateValidationError, match="channels"):
            op(State(data=jnp.zeros((N_TIME, 4)), coords=coords))

    def test_disagreeing_spectra_are_refused_at_construction(self):
        with pytest.raises(StateValidationError, match="channels"):
            BeamSpillOperator(sky_fraction=jnp.ones(4), t_ground=jnp.ones(5))

    def test_it_jits_and_differentiates_in_both_leaves(self, coords):
        data = jnp.full((N_TIME, 4), T_SKY)

        def loss(f, t_g):
            op = BeamSpillOperator(sky_fraction=f, t_ground=t_g)
            return jnp.sum(op(State(data=data, coords=coords)).data ** 2)

        d_f, d_g = jax.jit(jax.grad(loss, argnums=(0, 1)))(
            jnp.array(0.93), jnp.array(T_GROUND)
        )
        assert jnp.isfinite(d_f) and d_f != 0.0
        assert jnp.isfinite(d_g) and d_g != 0.0


class TestPlacement:
    """The split belongs to the astro branch and to nothing else."""

    @pytest.fixture
    def coords(self):
        return Coordinates(time=jnp.arange(float(N_TIME)),
                           freq=jnp.linspace(60e6, 85e6, 4))

    def test_it_lands_between_the_astro_entrance_and_the_antenna_sum(self, coords):
        nside = 4
        beam, _ = gaussian_beam(nside)
        sky = jnp.full((1, hp.nside2npix(nside)), T_SKY)
        twin = assemble(
            SkySourceOperator(
                sky_model=MapSky(sky),
                projector=polar_projector(nside, beam, horizon_mask=True),
            ),
            BeamSpillOperator(sky_fraction=jnp.array(0.9),
                              t_ground=jnp.array(T_GROUND)),
        )
        assert twin.lit == ("observed_astro_sky", "beam_spill")
        assert "astro_ant_sum" in twin.skipped

    def test_the_other_antenna_temperature_leaves_are_not_split(self, coords):
        """ground_pickup, atmosphere and t_sys_extra are effective temperatures
        by D13's construction. Splitting them would weight them twice -- and
        ground_pickup in particular IS a below-horizon share already."""
        split = BeamSpillOperator(sky_fraction=jnp.array(0.5),
                                  t_ground=jnp.array(0.0))
        leaves = (
            SkyOperator(amplitude=jnp.array(1000.0)),
            GroundPickupOperator(coupling=jnp.array(0.1),
                                 t_ground=jnp.array(200.0)),
            AtmosphericEmissionOperator(t_atm=jnp.array(4.0)),
        )
        state = State(coords=coords)
        out = assemble(*leaves, split)(state).data
        # uniform_sky feeds astro_sum -> ... -> beam, so it goes through the
        # split; the other two join at t_ant_sum, downstream of it.
        assert jnp.allclose(out, 0.5 * 1000.0 + 0.1 * 200.0 + 4.0, atol=1e-3)

    def test_the_calibration_loads_are_downstream_of_it(self, coords):
        switch = jnp.arange(N_TIME) % 2
        twin = assemble(
            SkyOperator(amplitude=jnp.array(1000.0)),
            BeamSpillOperator(sky_fraction=jnp.array(0.5),
                              t_ground=jnp.array(0.0)),
            CalLoadOperator(t_load=jnp.array(400.0)),
        )
        out = twin(State(coords=coords.replace(
            extra={"receiver_input": switch}))).data
        assert jnp.allclose(out[switch == 1], 400.0, atol=1e-3)
        assert jnp.allclose(out[switch == 0], 500.0, atol=1e-3)
