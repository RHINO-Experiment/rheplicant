"""The beam adapters' own surface only.

What a CST export *is*, and what the horizon cut *is*, are limTOD's subject and
are locked upstream — in ``tests/test_cstbeam.py`` and
``tests/limtod_jax/test_horizon_partition.py`` respectively. Re-testing the
grid reshape, the phi handedness, the frequency interpolation or the horizon
partition here would duplicate a moving target across two repositories (D20,
D25).

What belongs here is the seam:

* that the call reaches limTOD at all;
* the argument this side adds — Hz in, MHz across the boundary, and ``nside``
  inferred from maps that already carry it;
* that a limTOD without the feature is named at the boundary rather than
  failing on an AttributeError three calls deeper;

plus RHINO's actual horn, which is this package's subject and not limTOD's.
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
pytest.importorskip("limTOD.cstbeam", reason="the CST reader lives in limTOD")

RHINO_BEAMS = Path("~/Dataspace/RHINO/CST_beams/HornDryGround").expanduser()
requires_rhino = pytest.mark.skipif(
    not RHINO_BEAMS.is_dir(), reason=f"RHINO CST beams not present at {RHINO_BEAMS}"
)

THETA_STEP, PHI_STEP = 2.0, 5.0


def write_cst(path, *, sigma_deg=15.0):
    """A minimal well-formed export — enough to exercise the seam, no more.

    The pattern's properties do not matter here; the ones that do (azimuthal
    structure odd in phi, vanishing at the pole) are what limTOD's own tests
    are built on.
    """
    theta_deg = np.arange(0.0, 180.0 + THETA_STEP, THETA_STEP)
    phi_deg = np.arange(0.0, 360.0, PHI_STEP)
    pattern = np.exp(
        -0.5 * (np.deg2rad(theta_deg)[:, None] / np.deg2rad(sigma_deg)) ** 2
    ) * np.ones_like(phi_deg)[None, :]
    rows = [
        f"{theta:10.3f} {phi:10.3f} {10.0 * np.log10(pattern[i, j]):22.14e} 0 0 0 0 0"
        for j, phi in enumerate(phi_deg)      # theta runs fastest, as CST writes
        for i, theta in enumerate(theta_deg)
    ]
    path.write_text(
        "Theta [deg.]  Phi [deg.]  Abs(Dir.)[dBi]  Abs(Theta)[dBi]  "
        "Phase(Theta)[deg.]  Abs(Phi)[dBi]  Phase(Phi)[deg.]  Ax.Ratio[dB]\n"
        + "-" * 100 + "\n" + "\n".join(rows) + "\n"
    )


class TestTheCstSeam:
    NSIDE = 8

    def test_the_reader_reaches_limtod(self, tmp_path):
        write_cst(tmp_path / "Horn70.txt")
        theta_deg, phi_deg, directivity = read_cst_farfield(tmp_path / "Horn70.txt")
        assert directivity.shape == (theta_deg.size, phi_deg.size)
        assert np.all(directivity > 0.0)     # linear power, not dB

    def test_the_frequency_table_is_in_HERTZ_on_this_side(self, tmp_path):
        """limTOD keys the table in MHz, as it does everywhere; this package
        speaks Hz, because that is what ``Coordinates.freq`` carries. The
        conversion is the seam's whole contribution, so it is what gets tested.
        """
        write_cst(tmp_path / "HornDry70.5.txt")
        write_cst(tmp_path / "HornDry71.txt")
        assert sorted(cst_frequency_table(tmp_path)) == [70.5e6, 71.0e6]

        from limTOD.cstbeam import cst_frequency_table as upstream

        assert sorted(upstream(tmp_path)) == [70.5, 71.0]

    def test_maps_are_requested_in_hertz(self, tmp_path):
        write_cst(tmp_path / "Horn70.txt")
        maps = cst_beam_maps(tmp_path, [70e6], nside=self.NSIDE)
        assert maps.shape == (1, hp.nside2npix(self.NSIDE))

        from limTOD.cstbeam import cst_beam_maps as upstream

        np.testing.assert_allclose(
            maps, upstream(tmp_path, [70.0], nside=self.NSIDE), rtol=1e-12
        )

    def test_a_band_stated_in_hertz_is_the_band_enforced(self, tmp_path):
        """A unit slip at this seam would turn 70 MHz into 70 Hz and refuse
        every legitimate request — or, worse, accept an illegitimate one."""
        write_cst(tmp_path / "Horn60.txt")
        write_cst(tmp_path / "Horn80.txt")
        assert cst_beam_maps(tmp_path, [70e6], nside=self.NSIDE).shape[0] == 1
        with pytest.raises(ValueError, match="covers only"):
            cst_beam_maps(tmp_path, [85e6], nside=self.NSIDE)

    def test_a_limtod_without_the_reader_is_named_at_the_boundary(self, monkeypatch):
        import builtins

        real_import = builtins.__import__

        def blocked(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "limTOD" and fromlist and "cstbeam" in fromlist:
                raise ImportError("no cstbeam")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", blocked)
        with pytest.raises(ImportError, match="limTOD.cstbeam"):
            read_cst_farfield("irrelevant.txt")


@requires_rhino
class TestTheRhinoHorn:
    """Against the real CST export, not a synthetic stand-in.

    RHINO's horn is this package's subject, so these stay here even though the
    reader does not.
    """

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
        widths = [float((m * theta_hp).sum() / m.sum()) for m in maps]
        assert widths[1] < widths[0], f"85 MHz should be narrower, got {widths}"


class TestHorizonTruncation:
    """The adapter's own surface only.

    What the horizon cut IS -- the partition weights, the half-counted horizon
    ring, the zenith-only exactness, the painted-ground closure -- is limTOD's
    subject and is locked in ``tests/limtod_jax/test_horizon_partition.py``.
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
