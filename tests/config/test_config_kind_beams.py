"""resources.beams: addressed formats, required declarations, and one sub-value."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.resources import build_resources


@pytest.fixture
def context(tmp_path):
    np.save(tmp_path / "beam.npy", np.ones((4, 192)))
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 4), time=jnp.arange(8.0), dtype="float32",
        base_dir=str(tmp_path),
    )


def _beam(**overrides):
    spec = {"format": "npy", "path": "beam.npy", "nside": 4, "normalize": "pixel_sum",
            "frame": "beam_local"}
    spec.update(overrides)
    # None strips a key entirely rather than leaving it in the spec with a
    # None value: format: gaussian/inline/python callers override path=None
    # to say "this format has no path", and since resources.beams now sweeps
    # for unknown keys per format (none of those three take path:), a stray
    # {"path": None} would be refused as an unrecognised key rather than
    # simply ignored -- which is exactly backwards for a helper whose job is
    # building a VALID spec.
    spec = {key: value for key, value in spec.items() if value is not None}
    return {"beams": {"horn": spec}}


def _uvbeam(path, **overrides):
    spec = {"format": "uvbeam", "path": path, "nside": 4, "normalize": "pixel_sum"}
    spec.update(overrides)
    return {"beams": {"horn": spec}}


def _healpix(path, freq=None, **overrides):
    spec = {"format": "healpix", "path": path, "nside": 4, "normalize": "none",
            "frame": "beam_local", "order": "ring",
            "freq": freq if freq is not None else {"from_grid": "freq"}}
    spec.update(overrides)
    return {"beams": {"horn": spec}}


class TestTheFormats:
    def test_npy(self, context):
        built = build_resources(_beam(), context)
        assert built.resources["resources.beams.horn"].maps.shape == (4, 192)

    def test_inline_takes_a_value_node(self, context):
        """driftscan_mmode.py:57-60 and sky_to_noise_wave.py:104-107 both build
        the beam analytically and are designed to run without the unpublished
        CST dataset. v0 excluded beams from the python: hatch, so BOTH scripts
        died before step 1 -- the first thing a new user does is run a shipped
        example on a machine with no beam files."""
        built = build_resources(
            _beam(format="inline", path=None,
                  maps={"full": {"shape": ["n_freq", 192], "value": 1.0},
                        "unit": "dimensionless"}),
            context,
        )
        assert built.resources["resources.beams.horn"].maps.shape == (4, 192)

    def test_gaussian_takes_a_width(self, context):
        built = build_resources(
            _beam(format="gaussian", path=None, fwhm_deg={"value": 30.0, "unit": "deg"}), context
        )
        maps = built.resources["resources.beams.horn"].maps
        assert maps.shape == (4, 192)
        assert float(maps.max()) > float(maps.min())  # it is not flat

    def test_fwhm_deg_uses_the_standard_gaussian_constant(self, context):
        """Pins 2.3548200450309493 (2*sqrt(2*ln2)) rather than leaving it free
        to drift: fwhm_deg: 30 must produce exactly the same beam as
        sigma_deg: 30 / 2.3548200450309493, not merely a beam of similar shape."""
        via_fwhm = build_resources(
            _beam(format="gaussian", path=None, fwhm_deg={"value": 30.0, "unit": "deg"}), context
        )
        via_sigma = build_resources(
            _beam(format="gaussian", path=None,
                  sigma_deg={"value": 30.0 / 2.3548200450309493, "unit": "deg"}),
            context,
        )
        assert np.allclose(
            np.asarray(via_fwhm.resources["resources.beams.horn"].maps),
            np.asarray(via_sigma.resources["resources.beams.horn"].maps),
        )

    def test_a_per_channel_width_of_the_wrong_length_is_refused(self, context):
        """sigma_deg: [10, 20] on a 4-channel run must name both lengths and
        refuse before jnp.broadcast_to gets a chance to raise its own bare
        shape error, which names neither the key nor the run."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                _beam(format="gaussian", path=None,
                      sigma_deg={"list": [10.0, 20.0], "unit": "deg"}),
                context,
            )
        message = str(excinfo.value)
        assert "2" in message
        assert "4" in message

    def test_python(self, context):
        built = build_resources(
            _beam(format="python", path=None, python="jax.numpy:ones",
                  literal={"shape": [4, 192]}),
            context,
        )
        assert built.resources["resources.beams.horn"].maps.shape == (4, 192)

    def test_python_args_and_literal_may_not_share_a_key(self, context):
        """Mirrors sky_models.py's kind: python clash refusal: which one
        "won" would decide whether a value node was resolved through the
        grammar or forwarded untouched, and a document should not make that
        choice by writing the same key twice."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                _beam(format="python", path=None, python="jax.numpy:ones",
                      args={"shape": {"list": [4, 192]}}, literal={"shape": [4, 192]}),
                context,
            )
        assert "shape" in str(excinfo.value)


class TestTheTwoDeclarationsWithNoDefault:
    def test_normalize_is_required(self, context):
        """The output's unit is decided by the PAIR (beam normalisation,
        normalize_beam). A user who lifts normalize_beam: false from a preset
        built for a unit-sum beam and applies it to a raw CST beam is off by
        about 1.6e4 with every shape correct."""
        spec = _beam()
        del spec["beams"]["horn"]["normalize"]
        with pytest.raises(ConfigError) as excinfo:
            build_resources(spec, context)
        message = str(excinfo.value)
        assert "normalize" in message
        assert "32838" in message or "1.6e4" in message

    def test_frame_is_required_for_a_raw_array(self, context):
        spec = _beam()
        del spec["beams"]["horn"]["frame"]
        with pytest.raises(ConfigError, match="frame"):
            build_resources(spec, context)

    def test_phi0_and_phi_sense_are_required_for_cst_only(self, context):
        """D-C3: they are 'a fact about the as-built horn, not the file', so
        they cannot live in a preset -- and a mirrored beam passes every
        integral, every peak and every azimuthally-symmetric diagnostic
        UNCHANGED, so there is no numerical symptom to fall back on."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_beam(format="cst", path=None, directory="cst/"), context)
        message = str(excinfo.value)
        assert "phi0_deg" in message
        assert "phi_sense" in message
        assert "mirror" in message.lower()

    def test_they_are_refused_on_a_raw_array(self, context):
        """v0 required them for every format, which is exactly the
        invent-a-value habit the requirement exists to break."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_beam(phi0_deg={"value": 0.0, "unit": "deg"}), context)
        assert "phi0_deg" in str(excinfo.value)
        assert "CST" in str(excinfo.value)


class TestNormalisationIsApplied:
    def test_pixel_sum_makes_each_channel_sum_to_one(self, context):
        built = build_resources(_beam(normalize="pixel_sum"), context)
        sums = built.resources["resources.beams.horn"].maps.sum(axis=1)
        assert [float(v) for v in sums] == pytest.approx([1.0] * 4)

    def test_none_leaves_the_maps_alone(self, context):
        built = build_resources(_beam(normalize="none"), context)
        assert float(built.resources["resources.beams.horn"].maps.sum()) == pytest.approx(4 * 192)

    def test_solid_angle_makes_each_channel_integrate_to_one(self, context):
        """Kills the drops-the-4*pi/n_pix-factor mutant: pixel_sum alone
        would also make this pass if the factor were silently 1."""
        built = build_resources(_beam(normalize="solid_angle"), context)
        maps = built.resources["resources.beams.horn"].maps
        n_pix = maps.shape[1]
        integral = maps.sum(axis=1) * (4.0 * jnp.pi / n_pix)
        assert [float(v) for v in integral] == pytest.approx([1.0] * 4)


class TestHorizonTruncation:
    def test_it_exposes_both_products(self, context):
        """radio/beams.py:150 horizon_truncated_beam returns (maps, fraction)
        -- one call, two products -- and the fraction is exactly what
        BeamSpillOperator(sky_fraction=) wants. v0 consumed the maps and
        dropped the fraction, leaving the user with from: projector, which on
        a truncated beam returns about 1.0 and silently deletes the
        (1 - f_sky) * T_ground term.

        A no-op _truncate (return the maps and an all-ones fraction
        unchanged) would pass a shape-only assertion, so shape alone is not
        pinned here. On the all-ones fixture, normalize: pixel_sum makes
        each channel sum to 1.0 BEFORE truncation, so a fraction strictly
        below 1.0 and a post-truncation sum strictly below 1.0 both require
        the horizon cut to have actually removed weight -- and MEASURED: at
        nside 4 the strictly-south pixels are indices 104: (88 of them),
        which el_deg: 90 (boresight at zenith) must zero."""
        pytest.importorskip("limtod_jax")
        built = build_resources(
            _beam(horizon={"mode": "truncate_map", "el_deg": 90.0, "apod_deg": 0.0}), context
        )
        beam = built.resources["resources.beams.horn"]
        assert beam.maps.shape == (4, 192)
        assert beam.sky_fraction.shape == (4,)
        assert float(beam.sky_fraction.max()) < 1.0
        # pixel_sum made every channel sum to 1.0 before truncation removed weight.
        assert float(beam.maps.sum(axis=1).max()) < 1.0
        assert float(jnp.abs(beam.maps[:, 104:]).max()) == 0.0

    def test_apod_deg_is_forwarded_and_changes_the_result(self, context):
        """apod_deg widens the horizon cut with a cosine taper instead of a
        hard edge. If it were dropped on the way to horizon_truncated_beam,
        an apod_deg: 20 run would be numerically identical to apod_deg: 0 --
        MEASURED to differ (max abs diff ~2.8e-3 on this fixture)."""
        pytest.importorskip("limtod_jax")
        sharp = build_resources(
            _beam(horizon={"mode": "truncate_map", "el_deg": 90.0, "apod_deg": 0.0}), context
        )
        tapered = build_resources(
            _beam(horizon={"mode": "truncate_map", "el_deg": 90.0, "apod_deg": 20.0}), context
        )
        sharp_maps = sharp.resources["resources.beams.horn"].maps
        tapered_maps = tapered.resources["resources.beams.horn"].maps
        assert not np.allclose(np.asarray(sharp_maps), np.asarray(tapered_maps))

    def test_el_deg_other_than_90_is_refused(self, context):
        """Check C6: limTOD supports only 90."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_beam(horizon={"mode": "truncate_map", "el_deg": 80.0}), context)
        message = str(excinfo.value)
        assert "80" in message
        assert "90" in message


class TestTheSubValues:
    def test_maps_and_sky_fraction_are_both_referenceable(self, context):
        built = build_resources(_beam(), context)
        beam = built.resources["resources.beams.horn"]
        assert hasattr(beam, "maps")
        assert hasattr(beam, "sky_fraction")

    def test_sky_fraction_is_all_ones_when_nothing_was_truncated(self, context):
        built = build_resources(_beam(), context)
        fraction = built.resources["resources.beams.horn"].sky_fraction
        assert [float(v) for v in fraction] == pytest.approx([1.0] * 4)


class TestUvbeam:
    """format: uvbeam -- a pyuvdata UVBeam file, sampled through limTOD's bridge."""

    @pytest.fixture
    def beamfits(self, tmp_path):
        pytest.importorskip("pyuvdata")
        from pyuvdata.analytic_beam import GaussianBeam

        az = np.linspace(0.0, 2 * np.pi, 73)[:-1]
        za = np.linspace(0.0, np.deg2rad(90.0), 41)
        uvb = GaussianBeam(sigma=np.deg2rad(20.0)).to_uvbeam(
            freq_array=np.array([55e6, 90e6]), axis1_array=az, axis2_array=za,
            beam_type="efield",
        )
        uvb.write_beamfits(str(tmp_path / "horn.beamfits"))
        return "horn.beamfits"

    def test_it_reads_and_samples_onto_the_runs_grid(self, context, beamfits):
        """The efield construction is deliberate. MEASURED against pyuvdata
        3.2.6: a POWER beam built with UVBeam.new does not round-trip through
        beamfits -- the writer lands the frequency axis on the FITS IF slot
        and the reader refuses the file as multi-spectral-window."""
        built = build_resources(_uvbeam(beamfits), context)
        assert built.resources["resources.beams.horn"].maps.shape == (4, 192)

    def test_the_peak_is_at_the_pole_and_below_horizon_is_exactly_zero(
        self, context, beamfits
    ):
        """RING pixel 0 sits at the pole the beam points at, and the za grid
        stops at 90 deg, so the bridge's fill_value=0.0 makes every strictly-
        south pixel (indices 104: at nside 4) exactly zero. A wrong azimuth or
        zenith convention moves the peak off the pole; a wrong fill leaves
        ghost power below the horizon."""
        built = build_resources(_uvbeam(beamfits, normalize="none"), context)
        maps = built.resources["resources.beams.horn"].maps
        assert int(jnp.argmax(maps[0])) == 0
        assert float(jnp.abs(maps[:, 104:]).max()) == 0.0

    @pytest.mark.parametrize("key,value", [
        ("frame", "beam_local"),
        ("phi0_deg", {"value": 0.0, "unit": "deg"}),
        ("phi_sense", "ccw"),
    ])
    def test_the_chart_keys_are_refused(self, key, value, context, beamfits):
        """The bridge carries the azimuth convention itself; a declared chart
        is either redundant or a contradiction the maps cannot settle."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_uvbeam(beamfits, **{key: value}), context)
        assert key in str(excinfo.value)

    def test_an_out_of_band_run_grid_is_refused(self, context, beamfits):
        """MEASURED: pyuvdata's own interpolation raises ValueError('at least
        one interpolation frequency is outside of the UVBeam freq_array
        range') when the run's grid (100-125 MHz here) falls outside the
        file's own axis (55-90 MHz, this fixture) -- caught and rewrapped
        rather than left as a bare pyuvdata ValueError."""
        out_of_band = dataclasses.replace(context, freq=jnp.linspace(100e6, 125e6, 4))
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_uvbeam(beamfits), out_of_band)
        assert "does not extrapolate" in str(excinfo.value)

    def test_without_pyuvdata_the_refusal_names_the_extra(self, monkeypatch, context):
        """No importorskip, deliberately: the dependency check must fire
        before the path is even resolved, so a nonexistent path proves the
        ordering."""
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "pyuvdata" or name.startswith("pyuvdata."):
                raise ImportError("blocked for the test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_uvbeam("absent.beamfits"), context)
        message = str(excinfo.value)
        assert "pyuvdata" in message
        assert "uvbeam" in message


class TestHealpixFormat:
    """format: healpix -- declared ordering, header veto, declared grid."""

    @pytest.fixture
    def ring_file(self, tmp_path):
        import healpy as hp

        maps = np.stack([np.arange(192.0) + 1000.0 * i for i in range(4)])
        hp.write_map(str(tmp_path / "maps.fits"), maps)
        return "maps.fits"

    def test_ring_maps_are_read_as_written(self, context, ring_file):
        built = build_resources(_healpix(ring_file), context)
        maps = built.resources["resources.beams.horn"].maps
        assert maps.shape == (4, 192)
        assert float(maps[1, 0]) == pytest.approx(1000.0)

    def test_nested_is_reordered_exactly(self, context, tmp_path):
        """arange makes a wrong or missing reorder visible at almost every
        pixel; a permutation test on a constant map would pass under both."""
        import healpy as hp

        ring_original = np.stack([np.arange(192.0) + 1000.0 * i for i in range(4)])
        nested = np.stack([hp.reorder(row, r2n=True) for row in ring_original])
        hp.write_map(str(tmp_path / "nested.fits"), nested, nest=True)
        built = build_resources(_healpix("nested.fits", order="nested"), context)
        maps = built.resources["resources.beams.horn"].maps
        assert np.allclose(np.asarray(maps), ring_original)

    def test_a_declaration_contradicting_the_header_is_refused(self, context, ring_file):
        """One of the two is wrong and this layer cannot know which."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_healpix(ring_file, order="nested"), context)
        message = str(excinfo.value)
        assert "ORDERING" in message
        assert "nested" in message

    def test_a_missing_ordering_header_trusts_the_declaration(
        self, context, ring_file, monkeypatch
    ):
        """Not reachable through healpy's own writer, which always records
        ORDERING (measured, healpy 1.20.0) -- this simulates a file from a
        writer that omits it, via monkeypatching healpy.read_map itself
        rather than constructing a headerless FITS file by hand. Pins that
        the declared order: is trusted, rather than refused, when there is
        nothing in the file to check it against."""
        import healpy as hp

        real_read_map = hp.read_map

        def _headerless(path, **kwargs):
            raw, header = real_read_map(path, **kwargs)
            return raw, [(key, value) for key, value in header if key != "ORDERING"]

        monkeypatch.setattr(hp, "read_map", _headerless)
        built = build_resources(_healpix(ring_file), context)
        maps = built.resources["resources.beams.horn"].maps
        assert maps.shape == (4, 192)

    def test_a_single_column_file_on_a_one_channel_run_is_two_dimensional(self, tmp_path):
        """MEASURED (healpy 1.20.0): a single-column HEALPix file reads back
        as (n_pix,), not (1, n_pix) -- np.atleast_2d in _healpix_maps is what
        keeps Beam.maps's frequency axis from silently disappearing on the
        one-channel case."""
        import healpy as hp

        single = np.arange(192.0) + 500.0
        hp.write_map(str(tmp_path / "single.fits"), single)
        one_channel = ResolutionContext(
            freq=jnp.asarray([70e6]), time=jnp.arange(8.0), dtype="float32",
            base_dir=str(tmp_path),
        )
        built = build_resources(_healpix("single.fits"), one_channel)
        maps = built.resources["resources.beams.horn"].maps
        assert maps.shape == (1, 192)

    def test_order_is_required(self, context, ring_file):
        spec = _healpix(ring_file)
        del spec["beams"]["horn"]["order"]
        with pytest.raises(ConfigError, match="order"):
            build_resources(spec, context)

    def test_freq_is_required(self, context, ring_file):
        spec = _healpix(ring_file)
        del spec["beams"]["horn"]["freq"]
        with pytest.raises(ConfigError, match="freq"):
            build_resources(spec, context)

    def test_frame_is_required_like_any_raw_array(self, context, ring_file):
        spec = _healpix(ring_file)
        del spec["beams"]["horn"]["frame"]
        with pytest.raises(ConfigError, match="frame"):
            build_resources(spec, context)

    def test_a_grid_mismatch_is_refused(self, context, ring_file):
        """Same channel count, different band -- the failure MapSky's docstring
        names, on the beam side."""
        other = [float(v) for v in np.linspace(100e6, 125e6, 4)]
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_healpix(ring_file, freq={"list": other, "unit": "Hz"}),
                            context)
        assert "different channels" in str(excinfo.value)


class TestUnknownKeysAreRefused:
    """A per-format allowed-keys sweep, so a typo is refused rather than
    silently ignored. horizon.mode: truncate_map is the case that matters
    most: TestHorizonTruncation pins that an implementation of _truncate
    which does nothing would be caught (both products actually change); this
    class pins the OTHER way a horizon declaration can silently do nothing --
    the key itself never reaching build_beam's horizon.get('mode', 'none')
    at all, because it was spelled horizen:."""

    def test_a_horizon_typo_is_refused(self, context):
        spec = _beam()
        spec["beams"]["horn"]["horizen"] = {"mode": "truncate_map"}
        with pytest.raises(ConfigError, match="horizen"):
            build_resources(spec, context)

    def test_a_cst_suffix_typo_is_refused(self, context):
        """sufix: -- not suffix: -- is silently unread by spec.get('suffix',
        '.txt') today: the typo's value is discarded and the default '.txt'
        is used instead, with no error anywhere."""
        spec = _beam(format="cst", path=None, frame=None, directory="cst/",
                      phi0_deg={"value": 0.0, "unit": "deg"}, phi_sense="ccw")
        spec["beams"]["horn"]["sufix"] = ".txt"
        with pytest.raises(ConfigError, match="sufix"):
            build_resources(spec, context)


class TestTheHorizonInnerKeys:
    """horizon: is swept like every other mapping: a misspelled apod_dg was
    silently dropped, shipping a hard-edged cut where a taper was declared."""

    def test_a_misspelled_inner_key_is_refused(self, context):
        section = _beam(horizon={"mode": "truncate_map", "apod_dg": 3.0})
        with pytest.raises(ConfigError) as excinfo:
            build_resources(section, context)
        message = str(excinfo.value)
        assert "apod_dg" in message
        assert "apod_deg" in message  # the allowed list names the correct spelling

    def test_a_non_mapping_horizon_is_refused(self, context):
        with pytest.raises(ConfigError, match="horizon"):
            build_resources(_beam(horizon="truncate_map"), context)


class TestPresenceRefusals:
    def test_npy_without_path_is_refused_by_name(self, context):
        with pytest.raises(ConfigError, match=r"format: npy requires path"):
            build_resources(_beam(path=None), context)

    def test_python_without_target_is_refused_by_name(self, context):
        """Start from the file's existing VALID format: python spec (copy it
        from that test) and drop the python: key."""
        with pytest.raises(ConfigError, match=r"format: python requires"):
            build_resources(_beam(format="python", path=None), context)

    def test_a_list_args_is_refused_as_not_a_mapping(self, context):
        with pytest.raises(ConfigError, match="mapping of argument name"):
            build_resources(
                _beam(format="python", path=None, python="numpy:ones",
                      args=[12]),
                context)
