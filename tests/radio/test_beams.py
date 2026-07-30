"""CST far-field exports -> HEALPix beam maps.

Two layers. The synthetic tests build CST files whose answer is known in closed
form and always run. The RHINO tests run against the real horn when
``~/Dataspace/RHINO/CST_beams`` is present, and check the invariants that a
convention error would break: a directivity integrates to 4*pi, its boresight
sits at the pole, and the below-horizon fraction survives the resampling.
"""

from pathlib import Path

import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.radio import (
    cst_beam_maps,
    cst_frequency_table,
    horizon_truncated_beam,
    read_cst_farfield,
)

hp = pytest.importorskip("healpy", reason="healpy comes with rheplicant[limtod]")
pytest.importorskip("scipy")

RHINO_BEAMS = Path("~/Dataspace/RHINO/CST_beams/HornDryGround").expanduser()
requires_rhino = pytest.mark.skipif(
    not RHINO_BEAMS.is_dir(), reason=f"RHINO CST beams not present at {RHINO_BEAMS}"
)

THETA_STEP, PHI_STEP = 2.0, 5.0


def synthetic_directivity(theta_deg, phi_deg, *, sigma_deg, az_depth):
    """A Gaussian main lobe with genuine azimuthal structure, as directivity.

    Two properties the tests below rely on:

    * the azimuthal modulation carries a ``sin(theta)`` factor, so it vanishes
      at the pole. A far-field pattern must be single-valued at ``theta = 0``;
      a modulation that survives there makes "where is the peak?" meaningless.
    * it is ``sin(phi)``, ODD about ``phi = 0``, so reversing the azimuth
      handedness is detectable. ``cos(phi)`` would be its own mirror image and
      the ``phi_sense`` test would pass on a no-op.

    Normalized so ``int D dOmega == 4*pi`` on its own quadrature, which is what
    makes the integral test a statement about the reader rather than about the
    test's own arithmetic.
    """
    theta = np.deg2rad(theta_deg)[:, None]
    phi = np.deg2rad(phi_deg)[None, :]
    pattern = np.exp(-0.5 * (theta / np.deg2rad(sigma_deg)) ** 2) * (
        1.0 + az_depth * np.sin(theta) * np.sin(phi)
    )
    weight = np.sin(theta) * np.deg2rad(THETA_STEP) * np.deg2rad(PHI_STEP)
    return 4.0 * np.pi * pattern / float((pattern * weight).sum())


def write_cst(path, *, sigma_deg=15.0, az_depth=0.5):
    theta_deg = np.arange(0.0, 180.0 + THETA_STEP, THETA_STEP)
    phi_deg = np.arange(0.0, 360.0, PHI_STEP)
    directivity = synthetic_directivity(
        theta_deg, phi_deg, sigma_deg=sigma_deg, az_depth=az_depth
    )
    rows = []
    for j, phi in enumerate(phi_deg):          # theta runs fastest, as CST writes
        for i, theta in enumerate(theta_deg):
            dbi = 10.0 * np.log10(directivity[i, j])
            rows.append(f"{theta:10.3f} {phi:10.3f} {dbi:22.14e} 0 0 0 0 0")
    path.write_text(
        "Theta [deg.]  Phi [deg.]  Abs(Dir.)[dBi]  Abs(Theta)[dBi]  "
        "Phase(Theta)[deg.]  Abs(Phi)[dBi]  Phase(Phi)[deg.]  Ax.Ratio[dB]\n"
        + "-" * 100 + "\n" + "\n".join(rows) + "\n"
    )
    return theta_deg, phi_deg, directivity


class TestReader:
    def test_it_recovers_the_grid_and_the_linear_power(self, tmp_path):
        """dBi in, linear power out -- the quantity the projectors integrate."""
        path = tmp_path / "Horn70.txt"
        theta_deg, phi_deg, directivity = write_cst(path)
        got_theta, got_phi, got_d = read_cst_farfield(path)
        np.testing.assert_allclose(got_theta, theta_deg)
        np.testing.assert_allclose(got_phi, phi_deg)
        np.testing.assert_allclose(got_d, directivity, rtol=1e-6)

    def test_the_reshape_is_theta_fastest_not_phi_fastest(self, tmp_path):
        """CST writes theta fastest within each phi block. Reshaping the other
        way is shape-legal and gives a different beam, so the choice needs a
        discriminating check rather than a comment.

        The discriminator: ``theta = 0`` is a single direction, so row 0 of the
        correct grid is constant in phi. Under the transposed reading row 0 is a
        phi-slice at many thetas, and a beam with any radial structure at all
        makes that visibly non-constant."""
        path = tmp_path / "Horn70.txt"
        write_cst(path, az_depth=0.9)
        _, _, got = read_cst_farfield(path)
        np.testing.assert_allclose(got[0], got[0, 0], rtol=1e-9)

        flat = np.loadtxt(path, skiprows=2)[:, 2]
        transposed = 10.0 ** (flat.reshape(got.shape[0], got.shape[1]) / 10.0)
        assert not np.allclose(transposed[0], transposed[0, 0], rtol=1e-3), (
            "the transposed reading must be distinguishable, or this test is "
            "not testing anything"
        )

    def test_an_incomplete_grid_is_refused(self, tmp_path):
        path = tmp_path / "Horn70.txt"
        write_cst(path)
        lines = path.read_text().splitlines()
        path.write_text("\n".join(lines[:-3]) + "\n")   # drop three samples
        with pytest.raises(StateValidationError, match="do not fill"):
            read_cst_farfield(path)

    def test_a_table_without_the_directivity_column_is_refused(self, tmp_path):
        path = tmp_path / "Horn70.txt"
        path.write_text("h\n---\n0.0 0.0\n1.0 0.0\n")
        with pytest.raises(StateValidationError, match="columns"):
            read_cst_farfield(path)


class TestFrequencyTable:
    def test_frequencies_come_from_the_trailing_number_in_megahertz(self, tmp_path):
        for name in ("Horn70.txt", "Horn70.5.txt", "Horn85.txt", "notes.txt"):
            write_cst(tmp_path / name) if name != "notes.txt" else (
                (tmp_path / name).write_text("no frequency here\n")
            )
        table = cst_frequency_table(tmp_path)
        assert sorted(table) == [70e6, 70.5e6, 85e6]

    def test_an_empty_directory_is_refused(self, tmp_path):
        with pytest.raises(StateValidationError, match="No CST exports"):
            cst_frequency_table(tmp_path)


class TestHealpixSampling:
    NSIDE = 32

    def test_a_directivity_still_integrates_to_four_pi(self, tmp_path):
        """The single strongest check on the whole path: dBi conversion, the
        (theta, phi) mapping and the HEALPix sampling all have to be right for
        the solid-angle integral to survive."""
        write_cst(tmp_path / "Horn70.txt")
        maps = cst_beam_maps(tmp_path, [70e6], nside=self.NSIDE)
        integral = maps[0].sum() * 4.0 * np.pi / hp.nside2npix(self.NSIDE)
        assert abs(integral / (4.0 * np.pi) - 1.0) < 0.01

    def test_the_boresight_lands_on_the_pole(self, tmp_path):
        write_cst(tmp_path / "Horn70.txt")
        maps = cst_beam_maps(tmp_path, [70e6], nside=self.NSIDE)
        theta, _ = hp.pix2ang(self.NSIDE, int(np.argmax(maps[0])))
        assert np.rad2deg(theta) < 2.0

    def test_phi_sense_flips_the_azimuthal_structure(self, tmp_path):
        """The handedness is not derivable from the file, so it is a knob; this
        pins that the knob does something, and does the RIGHT something -- a
        reflection about phi = 0, not a relabelling."""
        write_cst(tmp_path / "Horn70.txt", az_depth=0.9)
        ccw = cst_beam_maps(tmp_path, [70e6], nside=self.NSIDE, phi_sense="ccw")[0]
        cw = cst_beam_maps(tmp_path, [70e6], nside=self.NSIDE, phi_sense="cw")[0]
        assert not np.allclose(ccw, cw)

        theta, phi = hp.pix2ang(self.NSIDE, np.arange(hp.nside2npix(self.NSIDE)))
        mirrored = hp.ang2pix(self.NSIDE, theta, (2.0 * np.pi - phi) % (2.0 * np.pi))
        band = theta > np.deg2rad(10.0)   # the pole's pixels are their own mirror
        np.testing.assert_allclose(ccw[mirrored][band], cw[band], rtol=2e-2)

    def test_phi0_rotates_the_pattern(self, tmp_path):
        write_cst(tmp_path / "Horn70.txt", az_depth=0.9)
        base = cst_beam_maps(tmp_path, [70e6], nside=self.NSIDE)[0]
        turned = cst_beam_maps(tmp_path, [70e6], nside=self.NSIDE, phi0_deg=180.0)[0]
        assert not np.allclose(base, turned)
        assert abs(base.sum() - turned.sum()) / base.sum() < 1e-3

    def test_an_unknown_phi_sense_is_refused(self, tmp_path):
        write_cst(tmp_path / "Horn70.txt")
        with pytest.raises(StateValidationError, match="phi_sense"):
            cst_beam_maps(tmp_path, [70e6], nside=8, phi_sense="widdershins")


class TestFrequencyInterpolation:
    def test_a_frequency_on_the_grid_reproduces_its_own_file(self, tmp_path):
        write_cst(tmp_path / "Horn60.txt", sigma_deg=20.0)
        write_cst(tmp_path / "Horn80.txt", sigma_deg=12.0)
        both = cst_beam_maps(tmp_path, [60e6, 80e6], nside=16)
        only60 = cst_beam_maps(tmp_path, [60e6], nside=16)[0]
        np.testing.assert_allclose(both[0], only60, rtol=1e-12)
        assert not np.allclose(both[0], both[1]), "the two files must differ"

    def test_a_midpoint_is_the_average_of_its_neighbours(self, tmp_path):
        write_cst(tmp_path / "Horn60.txt", sigma_deg=20.0)
        write_cst(tmp_path / "Horn80.txt", sigma_deg=12.0)
        ends = cst_beam_maps(tmp_path, [60e6, 80e6], nside=16)
        mid = cst_beam_maps(tmp_path, [70e6], nside=16)[0]
        np.testing.assert_allclose(mid, 0.5 * (ends[0] + ends[1]), rtol=1e-10)

    def test_extrapolation_beyond_the_simulated_band_is_refused(self, tmp_path):
        write_cst(tmp_path / "Horn60.txt")
        write_cst(tmp_path / "Horn80.txt")
        with pytest.raises(StateValidationError, match="covers only"):
            cst_beam_maps(tmp_path, [55e6], nside=8)
        with pytest.raises(StateValidationError, match="covers only"):
            cst_beam_maps(tmp_path, [85e6], nside=8)


@requires_rhino
class TestTheRhinoHorn:
    """Against the real CST export, not a synthetic stand-in."""

    NSIDE = 32

    def test_the_directivity_normalization_survives_resampling(self):
        maps = cst_beam_maps(RHINO_BEAMS, [70e6], nside=self.NSIDE)
        integral = maps[0].sum() * 4.0 * np.pi / hp.nside2npix(self.NSIDE)
        # The raw file's own quadrature is 0.38% off 4*pi; resampling must not
        # add materially to that.
        assert abs(integral / (4.0 * np.pi) - 1.0) < 0.01

    def test_the_boresight_is_at_the_pole_and_the_gain_is_the_files(self):
        maps = cst_beam_maps(RHINO_BEAMS, [70e6], nside=self.NSIDE)
        theta, _ = hp.pix2ang(self.NSIDE, int(np.argmax(maps[0])))
        assert np.rad2deg(theta) < 2.0
        peak_dbi = 10.0 * np.log10(maps[0].max())
        assert 12.0 < peak_dbi < 15.0, f"RHINO's horn is ~14 dBi, got {peak_dbi:.2f}"

    def test_the_below_horizon_fraction_matches_the_raw_export(self):
        """A beam-local theta > 90 deg is the horn's back response. It is small
        (~3%) but not zero, and a sampling error would move it."""
        _, _, directivity = read_cst_farfield(cst_frequency_table(RHINO_BEAMS)[70e6])
        theta_deg = np.arange(directivity.shape[0], dtype=float)
        weight = np.sin(np.deg2rad(theta_deg))[:, None]
        raw = float((directivity * weight)[theta_deg > 90.0].sum()
                    / (directivity * weight).sum())

        maps = cst_beam_maps(RHINO_BEAMS, [70e6], nside=self.NSIDE)[0]
        theta_hp, _ = hp.pix2ang(self.NSIDE, np.arange(hp.nside2npix(self.NSIDE)))
        sampled = float(maps[theta_hp > np.pi / 2].sum() / maps.sum())
        assert abs(sampled - raw) < 0.005, f"raw {raw:.4f} vs sampled {sampled:.4f}"

    def test_the_beam_narrows_with_frequency(self):
        """A horn's beam narrows as the wavelength shrinks. Cheap, and it would
        catch a frequency table read in the wrong order."""
        maps = cst_beam_maps(RHINO_BEAMS, [60e6, 85e6], nside=self.NSIDE)
        theta_hp, _ = hp.pix2ang(self.NSIDE, np.arange(hp.nside2npix(self.NSIDE)))
        widths = [
            float((m * theta_hp).sum() / m.sum()) for m in maps
        ]
        assert widths[1] < widths[0], f"85 MHz should be narrower, got {widths}"


class TestHorizonTruncation:
    """The adapter's own surface only.

    What the horizon cut IS -- the partition weights, the half-counted horizon
    ring, the zenith-only exactness, the painted-ground closure -- is limTOD's
    subject and is locked in ``tests/limtod_jax/test_horizon_partition.py``.
    Re-testing it here would duplicate a moving target across two repos. What
    belongs here is the seam: that the call reaches limTOD, that nside is
    inferred from the maps, and that a stale install says so.
    """

    NSIDE = 16

    def beam(self):
        theta, _ = hp.pix2ang(self.NSIDE, np.arange(hp.nside2npix(self.NSIDE)))
        return (np.exp(-0.5 * (theta / np.deg2rad(35.0)) ** 2) + 0.02)[None, :]

    def test_it_infers_nside_and_returns_maps_with_a_fraction(self):
        maps, fraction = horizon_truncated_beam(self.beam())
        assert maps.shape == (1, hp.nside2npix(self.NSIDE))
        assert fraction.shape == (1,)
        assert 0.0 < float(fraction[0]) < 1.0

    def test_a_single_map_is_accepted(self):
        maps, fraction = horizon_truncated_beam(self.beam()[0])
        assert maps.shape == (1, hp.nside2npix(self.NSIDE))
        assert fraction.shape == (1,)

    def test_an_invalid_map_length_is_refused_here(self):
        """The one guard this side owns, because it is about the argument this
        side adds: nside is inferred, so a bad length has to be caught before
        limTOD is handed a wrong one."""
        with pytest.raises(StateValidationError, match="HEALPix"):
            horizon_truncated_beam(np.ones((1, 100)))

    def test_it_reaches_limtods_own_physics(self):
        """A value check thin enough not to duplicate limTOD, sharp enough to
        fail if the call went somewhere else: an isotropic beam divides the
        sphere exactly in half."""
        _, fraction = horizon_truncated_beam(np.ones((1, hp.nside2npix(self.NSIDE))))
        assert abs(float(fraction[0]) - 0.5) < 1e-12

    def test_a_tilted_pointing_is_refused_by_limtod(self):
        """Not re-raised as a rheplicant error: it is limTOD's physics
        constraint, and its message names limTOD's own alternative."""
        with pytest.raises(ValueError, match="zenith"):
            horizon_truncated_beam(self.beam(), el_deg=45.0)

    def test_an_outdated_limtod_is_named_at_the_boundary(self, monkeypatch):
        import limtod_jax

        monkeypatch.delattr(limtod_jax, "horizon_truncated_beam", raising=False)
        with pytest.raises(ImportError, match="limTOD >= 1.9"):
            horizon_truncated_beam(self.beam())
