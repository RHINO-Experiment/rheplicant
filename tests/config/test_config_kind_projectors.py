"""resources.projectors: three engines, and the keys a YAML may not write."""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.resources import build_resources

pytest.importorskip("limtod_jax")


@pytest.fixture
def context(tmp_path):
    np.save(tmp_path / "beam.npy", np.ones((2, 192)))
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 2), time=jnp.arange(8.0), dtype="float64",
        base_dir=str(tmp_path),
    )


def _doc(**overrides):
    projector = {
        "engine": "driftscan",
        "beam": {"ref": "resources.beams.horn"},
        "lmax": 8,
        "lat_deg": {"value": 53.2367, "unit": "deg"},
        "az_deg": {"value": 0.0, "unit": "deg"},
        "el_deg": {"value": 90.0, "unit": "deg"},
        "normalize_beam": True,
        "acknowledge_float32_sky": False,
    }
    projector.update(overrides)
    return {
        "beams": {"horn": {"format": "npy", "path": "beam.npy", "nside": 4,
                           "normalize": "pixel_sum", "frame": "beam_local"}},
        "projectors": {"drift": projector},
    }


def _matrix_doc(provenance):
    return {"projectors": {"baked": {
        "engine": "matrix",
        "matrix": {"full": {"shape": [8, 192], "value": 1.0}},
        "provenance": provenance,
    }}}


class TestTheEngines:
    def test_driftscan_goes_through_from_beam_maps(self, context):
        """from_beam_maps() is the CORRECT analysis: it sets beam_alms, the
        projector's only traced array. Every other field is static."""
        built = build_resources(_doc(), context)
        projector = built.resources["resources.projectors.drift"]
        assert projector.beam_alms.shape == (2, 45)  # (lmax+1)(lmax+2)/2 at lmax 8
        assert projector.nside == 4  # inferred by the classmethod, not written

    def test_general_pointing_can_share_the_drift_engines_alms(self, context):
        """examples/driftscan_mmode.py:84 hands drift.beam_alms to the general
        engine so both see the SAME analysis. Re-analysing the beam with a
        different transform destroys the 2e-16 agreement the comparison
        exists to demonstrate."""
        doc = _doc()
        doc["projectors"]["general"] = {
            "engine": "general_pointing",
            "beam_alms": {"ref": "resources.projectors.drift.beam_alms"},
            "lmax": 8, "nside": 4,
            "lat_deg": {"value": 53.2367, "unit": "deg"},
            "normalize_beam": True,
        }
        built = build_resources(doc, context)
        drift = built.resources["resources.projectors.drift"]
        general = built.resources["resources.projectors.general"]
        assert general.beam_alms is drift.beam_alms

    def test_general_pointing_without_beam_alms_uses_analyse(self, context):
        """No ``beam_alms:`` written -- the branch falls back to ``_analyse()``
        on the beam's own maps, a path the sharing test above never exercises
        (it always supplies ``beam_alms`` by reference). MEASURED:
        ``_analyse``'s ``healpy.map2alm`` and ``from_beam_maps``'s
        ``limtod_jax.map2alm_iter`` agree to within float32 roundoff on this
        fixture (max abs diff ~2e-4 of a ~0.018 alm magnitude here, x64
        disabled in this venv; the reviewer's x64 run measured 3.4e-18) --
        pinning that ``_analyse`` is a faithful re-implementation of the SAME
        beam analysis ``from_beam_maps`` performs, not merely one that
        happens to return the right shape."""
        doc = _doc()
        doc["projectors"]["general"] = {
            "engine": "general_pointing",
            "beam": {"ref": "resources.beams.horn"},
            "lmax": 8, "nside": 4,
            "lat_deg": {"value": 53.2367, "unit": "deg"},
            "normalize_beam": True,
        }
        built = build_resources(doc, context)
        drift = built.resources["resources.projectors.drift"]
        general = built.resources["resources.projectors.general"]
        assert jnp.allclose(general.beam_alms, drift.beam_alms, atol=5e-4)

    def test_matrix_requires_provenance(self, context):
        doc = {"projectors": {"baked": {"engine": "matrix",
                                        "matrix": {"full": {"shape": [8, 192], "value": 1.0}}}}}
        with pytest.raises(ConfigError) as excinfo:
            build_resources(doc, context)
        assert "provenance" in str(excinfo.value)


class TestTheKeysAYamlMayNotWrite:
    @pytest.mark.parametrize("key,value", [("beam_frame", "reference"), ("beam_ref_lst_deg", 0.0)])
    def test_they_are_refused_and_the_route_is_named(self, key, value, context):
        """__check_init__ exists to catch a hand-set pair; a YAML that wrote
        them would drive the object into the state that guard exists to catch."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_doc(**{key: value}), context)
        message = str(excinfo.value)
        assert key in message
        assert "cache_beam_rotation" in message

    def test_nside_is_refused_on_the_from_beam_maps_path(self, context):
        """MEASURED: passing nside through **kwargs gives
        TypeError: ... got multiple values for keyword argument 'nside'."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_doc(nside=4), context)
        assert "nside" in str(excinfo.value)
        assert "inferred" in str(excinfo.value)


class TestNormalizeBeam:
    def test_it_is_required_with_no_default(self, context):
        doc = _doc()
        del doc["projectors"]["drift"]["normalize_beam"]
        with pytest.raises(ConfigError) as excinfo:
            build_resources(doc, context)
        message = str(excinfo.value)
        assert "normalize_beam" in message
        assert "32838" in message


class TestCheckA44:
    def test_a_real_sky_engine_in_float32_needs_an_acknowledgement(self, tmp_path):
        """radio/sky/general_pointing.py:28-32 -- 'the map/alm steps carry
        O(10%) errors in float32'. A 10% error on the beam-weighted sky is
        larger than every effect normalize_beam, phi0_deg and phi_sense are
        required keys for, and it is invisible: the maps come back finite,
        correctly shaped and plausibly structured."""
        np.save(tmp_path / "beam.npy", np.ones((2, 192)))
        context = ResolutionContext(freq=jnp.linspace(60e6, 85e6, 2), time=jnp.arange(8.0),
                                    dtype="float32", base_dir=str(tmp_path))
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_doc(), context)
        message = str(excinfo.value)
        assert "jax_enable_x64" in message
        assert "10%" in message or "O(10" in message
        assert "acknowledge_float32_sky" in message

    def test_the_acknowledgement_lets_it_through(self, tmp_path):
        np.save(tmp_path / "beam.npy", np.ones((2, 192)))
        context = ResolutionContext(freq=jnp.linspace(60e6, 85e6, 2), time=jnp.arange(8.0),
                                    dtype="float32", base_dir=str(tmp_path))
        built = build_resources(_doc(acknowledge_float32_sky=True), context)
        assert "resources.projectors.drift" in built.resources


class TestOptimizationsAndLstRef:
    def test_cache_beam_rotation_requires_lst_ref_deg(self, context):
        """Check A48: to_reference_frame() raises without one -- AFTER the
        beam file has been read and analysed, which is the class of failure
        the pre-flight checks exist to prevent."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_doc(optimizations=["cache_beam_rotation"]), context)
        message = str(excinfo.value)
        assert "lst_ref_deg" in message
        assert "analysed" in message or "analysis" in message

    def test_read_horizon_fraction_is_not_an_optimisation(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_doc(optimizations=["read_horizon_fraction"]), context)
        message = str(excinfo.value)
        assert "from: horizon_fraction" in message

    def test_an_unknown_optimisation_is_refused_listing_the_known_ones(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(_doc(optimizations=["bogus"]), context)
        message = str(excinfo.value)
        assert "bogus" in message
        assert "cache_beam_rotation" in message

    def test_horizon_fraction_against_a_cached_projector_is_refused(self, context):
        """C7: horizon_fraction() raises on a beam_frame='reference'
        projector, because the unmasked denominator is gone. Order matters:
        call it BEFORE to_reference_frame(), never after."""
        from rheplicant.config.values import resolve_value

        built = build_resources(
            _doc(optimizations=["cache_beam_rotation"],
                 lst_ref_deg={"value": 0.0, "unit": "deg"}),
            context,
        )
        scoped = context
        for name, value in built.resources.items():
            scoped = scoped.with_resource(name, value)
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from": "horizon_fraction",
                 "projector": {"ref": "resources.projectors.drift"}},
                scoped,
            )
        assert "cache_beam_rotation" in str(excinfo.value)


class TestTheEngineKeys:
    """A per-engine allowed-keys sweep (the house pattern from
    kinds/beams.py's own TestUnknownKeysAreRefused), placed AFTER every
    richer, engine-specific refusal above -- the nside-on-driftscan check in
    particular already has its own message ('inferred'), and running the
    sweep before it would shadow that message with the generic 'does not
    take' for exactly the key that most needs the specific one."""

    def test_a_typo_on_driftscan_is_refused(self, context):
        with pytest.raises(ConfigError, match="lmaxx"):
            build_resources(_doc(lmaxx=8), context)

    def test_a_driftscan_only_key_on_general_pointing_is_refused(self, context):
        """az_deg is real -- it is driftscan's own fixed-pointing field -- but
        general_pointing reads pointing from coords, per sample, and has no
        field to put it in."""
        doc = _doc()
        doc["projectors"]["general"] = {
            "engine": "general_pointing",
            "beam_alms": {"ref": "resources.projectors.drift.beam_alms"},
            "lmax": 8, "nside": 4,
            "lat_deg": {"value": 53.2367, "unit": "deg"},
            "normalize_beam": True,
            "az_deg": {"value": 0.0, "unit": "deg"},
        }
        with pytest.raises(ConfigError, match="az_deg"):
            build_resources(doc, context)


class TestAngleUnits:
    def test_lat_deg_with_a_non_angle_unit_is_refused(self, context):
        with pytest.raises(ConfigError, match="must be an angle"):
            build_resources(_doc(lat_deg={"value": 1.0, "unit": "Hz"}), context)


class TestBeamIterationsIsForwarded:
    def test_it_changes_beam_alms(self, context):
        """``iterations=`` reaches ``map2alm_iter`` directly, and it is a
        constructor argument rather than a stored field -- so a build_projector
        that silently hard-coded 3 (the default) would still return an object
        with a plausible, correctly shaped ``beam_alms``, and nothing about the
        object itself could ever reveal the mistake. MEASURED on this fixture:
        beam_iterations 1 vs the default 3 differ (max abs diff ~6e-4 here,
        x64 disabled in this venv)."""
        default = build_resources(_doc(), context)
        one_iteration = build_resources(_doc(beam_iterations=1), context)
        a_default = default.resources["resources.projectors.drift"].beam_alms
        a_one = one_iteration.resources["resources.projectors.drift"].beam_alms
        assert not jnp.allclose(a_default, a_one)


class TestHorizonFractionHappyPath:
    def test_on_an_uncached_projector_the_fraction_is_a_real_share(self, context):
        """The refusal on a cached (``beam_frame="reference"``) projector is
        pinned above; this pins the happy path it guards, which a
        return-``ones``-shaped mutant would otherwise pass silently -- every
        other assertion in this module either checks a refusal or checks
        identity/shape, never the VALUE ``horizon_fraction()`` actually
        returns. Pinned loosely (a window, not exact figures; MEASURED ~0.5 on
        this fixture's uniform beam) because the precise number is a
        limtod_jax implementation detail, not this layer's contract."""
        from rheplicant.config.values import resolve_value

        built = build_resources(_doc(), context)
        scoped = context
        for name, value in built.resources.items():
            scoped = scoped.with_resource(name, value)
        result = resolve_value(
            {"from": "horizon_fraction", "projector": {"ref": "resources.projectors.drift"}},
            scoped,
        )
        assert result.value.shape == (2,)  # this fixture's n_freq
        assert bool(jnp.all(result.value > 0.3))
        assert bool(jnp.all(result.value < 0.7))
        assert result.unit.canonical == "dimensionless"


class TestProvenance:
    """``provenance:`` exists to force a record naming who built the matrix,
    at what latitude and over what LST range -- a gate that only checks
    PRESENCE, not content, lets ``provenance: null`` (or ``{}``, or ``""``)
    through: each satisfies ``'provenance' in spec`` while recording nothing,
    which is exactly the state the gate exists to rule out."""

    def test_none_is_refused(self, context):
        with pytest.raises(ConfigError, match="provenance"):
            build_resources(_matrix_doc(None), context)

    def test_empty_mapping_is_refused(self, context):
        with pytest.raises(ConfigError, match="provenance"):
            build_resources(_matrix_doc({}), context)

    def test_a_non_empty_mapping_is_accepted(self, context):
        built = build_resources(_matrix_doc({"built_by": "test"}), context)
        assert "resources.projectors.baked" in built.resources


class TestMatrixEngine:
    def test_is_exempt_from_the_float32_gate(self, tmp_path):
        """Check A44's dtype gate lives in the shared prelude AFTER the
        matrix branch already returned: ``MatrixProjector`` reads no beam and
        does no map/alm work, so the O(10%) float32 error the gate warns
        about does not apply to it. Pins that scoping -- a matrix entry in a
        float32 context, with no ``acknowledge_float32_sky:``, must still
        succeed. The dtype assertion below documents the built object's
        dtype in that context; it is not a live pin on projectors.py's own
        ``dtype=context.dtype`` cast -- the ``full:`` value node already
        delivers a float32 array here (the value grammar casts to
        ``context.dtype`` itself), so a dropped cast in this kind's own code
        would not be caught by this assertion alone."""
        float32_context = ResolutionContext(
            freq=jnp.linspace(60e6, 85e6, 2), time=jnp.arange(8.0),
            dtype="float32", base_dir=str(tmp_path),
        )
        built = build_resources(_matrix_doc({"built_by": "test"}), float32_context)
        projector = built.resources["resources.projectors.baked"]
        assert projector.matrix.dtype == jnp.float32  # documents the built dtype
