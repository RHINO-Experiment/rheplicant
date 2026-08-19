"""resources.sky_models: addressed kinds, and the grid a MapSky is pinned to."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.kinds import sky_models as sky_models_module
from rheplicant.config.kinds.sky_models import build_sky_model
from rheplicant.config.resources import build_resources
from rheplicant.radio import MapSky, PowerLawSkyModel, UniformSkyModel


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 4), time=jnp.arange(8.0), dtype="float32"
    )


class TestTheFourKinds:
    def test_uniform(self, context):
        built = build_resources(
            {"sky_models": {"flat": {"kind": "uniform", "amplitude": {"value": 200.0, "unit": "K"},
                                     "n_pix": 192}}},
            context,
        )
        sky = built.resources["resources.sky_models.flat"]
        assert isinstance(sky, UniformSkyModel)
        assert sky.n_pix == 192 and type(sky.n_pix) is int

    def test_power_law(self, context):
        built = build_resources(
            {"sky_models": {"fg": {"kind": "power_law",
                                   "amplitude": {"value": 300.0, "unit": "K"},
                                   "spectral_index": 2.5,
                                   "ref_freq": {"value": 70.0, "unit": "MHz"},
                                   "n_pix": 192}}},
            context,
        )
        sky = built.resources["resources.sky_models.fg"]
        assert isinstance(sky, PowerLawSkyModel)
        assert sky.ref_freq == pytest.approx(7e7)
        assert type(sky.ref_freq) is float  # static, and a static array corrupts the jit key

    def test_maps_from_a_value_node(self, context):
        """The maps in driftscan_mmode.py:61 and sky_to_noise_wave.py:109-112
        are DRAWN, not read, and will never be on disk -- which is why MapSky
        ships with a value-node constructor as well as a file one."""
        built = build_resources(
            {"sky_models": {"drawn": {"kind": "maps",
                                      "maps": {"full": {"shape": ["n_freq", 192], "value": 100.0},
                                               "unit": "K"},
                                      "freq": {"from_grid": "freq"},
                                      "nside": 4}}},
            context,
        )
        sky = built.resources["resources.sky_models.drawn"]
        assert isinstance(sky, MapSky)
        assert sky.maps.shape == (4, 192)

    def test_python(self, context):
        built = build_resources(
            {"sky_models": {"custom": {"kind": "python",
                                       "python": "rheplicant.radio:UniformSkyModel",
                                       "args": {"amplitude": {"value": 10.0, "unit": "K"},
                                                "n_pix": 12}}}},
            context,
        )
        sky = built.resources["resources.sky_models.custom"]
        assert isinstance(sky, UniformSkyModel)
        # Kills the raw-args mutant: equinox does not enforce field annotations
        # at runtime, so a builder that forwarded args unresolved would still
        # construct a UniformSkyModel -- just one carrying a value node instead
        # of a number.
        assert float(sky.amplitude) == 10.0
        assert sky.n_pix == 12

    def test_uniform_amplitude_is_cast_to_the_context_dtype(self, context):
        """{value: 200} is a bare Python int; if _traced's dtype cast were ever
        dropped, amplitude would come back as whatever dtype jnp.asarray infers
        for a Python int rather than the context's own float32."""
        built = build_resources(
            {"sky_models": {"flat": {"kind": "uniform", "amplitude": {"value": 200}, "n_pix": 12}}},
            context,
        )
        sky = built.resources["resources.sky_models.flat"]
        assert sky.amplitude.dtype == jnp.float32

    def test_uniform_stray_key_is_refused(self, context):
        """Check item 1: before the per-kind key table, uniform/power_law/python
        had no unknown-key sweep at all, so a stray key such as spectral_index
        on a uniform sky was silently discarded rather than refused."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "uniform",
                                        "amplitude": {"value": 200.0, "unit": "K"},
                                        "n_pix": 192,
                                        "spectral_index": 2.5}}},
                context,
            )
        assert "spectral_index" in str(excinfo.value)

    def test_python_stray_key_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "python",
                                        "python": "rheplicant.radio:UniformSkyModel",
                                        "args": {"amplitude": {"value": 10.0, "unit": "K"},
                                                 "n_pix": 12},
                                        "amplitude": {"value": 5.0, "unit": "K"}}}},
                context,
            )
        assert "amplitude" in str(excinfo.value)

    def test_python_args_literal_clash_is_refused(self, context):
        """args and literal disagreeing on which key won was the risk this
        refusal removes: args values are resolved through the value grammar
        and literal values are forwarded untouched, so the same key in both
        would silently decide which happened."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "python",
                                        "python": "rheplicant.radio:UniformSkyModel",
                                        "args": {"amplitude": {"value": 10.0, "unit": "K"}},
                                        "literal": {"amplitude": 5.0, "n_pix": 12}}}},
                context,
            )
        assert "amplitude" in str(excinfo.value)

    def test_python_argument_targets_are_validated_before_import(
        self, context, monkeypatch
    ):
        imported = []
        monkeypatch.setattr(
            sky_models_module,
            "import_target",
            lambda target: imported.append(target) or (lambda **kwargs: object()),
        )
        with pytest.raises(ConfigError, match="unit"):
            build_resources(
                {
                    "sky_models": {
                        "bad": {
                            "kind": "python",
                            "python": "probe.module:factory",
                            "args": {"bad": {"value": 1.0, "unit": 7}},
                        }
                    }
                },
                context,
            )
        assert imported == []


class TestStaticFieldGuards:
    """The two static-field guards in _static_int / _static_float."""

    def test_n_pix_as_a_bool_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "uniform",
                                        "amplitude": {"value": 200.0, "unit": "K"},
                                        "n_pix": True}}},
                context,
            )
        message = str(excinfo.value)
        assert "plain integer" in message
        assert "bool" in message

    def test_ref_freq_as_a_list_form_is_refused(self, context):
        """resolve_value({"list": [70.0]}, ...).source == "list", not "scalar" --
        exactly the array-producing form _static_float exists to catch."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "power_law",
                                        "amplitude": {"value": 300.0, "unit": "K"},
                                        "spectral_index": 2.5,
                                        "ref_freq": {"list": [70.0], "unit": "MHz"},
                                        "n_pix": 192}}},
                context,
            )
        message = str(excinfo.value)
        assert "treedef" in message
        assert "jit" in message


class TestTheGridCheck:
    def test_n_pix_must_match_12_nside_squared(self, context):
        """Check C5. MapSky.__call__ ignores its freq argument beyond the
        shape, so a map at the wrong resolution is finite, correctly shaped
        and wrong -- and cannot be caught under jit, because the values are
        traced and only the shape is static."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "maps",
                                        "maps": {"full": {"shape": ["n_freq", 100], "value": 1.0},
                                                 "unit": "K"},
                                        "freq": {"from_grid": "freq"},
                                        "nside": 4}}},
                context,
            )
        message = str(excinfo.value)
        assert "100" in message  # what was found
        assert "192" in message  # 12 * 4**2, what nside implies

    def test_the_first_axis_must_be_the_runs_frequency_axis(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "maps",
                                        "maps": {"full": {"shape": [3, 192], "value": 1.0},
                                                 "unit": "K"},
                                        "freq": {"from_grid": "freq"},
                                        "nside": 4}}},
                context,
            )
        message = str(excinfo.value)
        assert "3" in message
        assert "4" in message

    def test_the_declared_freq_length_must_match_the_maps(self, context):
        """This layer's own check, ahead of MapSky's: the maps have the run's
        own channel count (4, so the check above this one passes) but the
        declared freq names only 3 -- not a shape MapSky.__check_init__ would
        ever see, since this is refused before MapSky is constructed."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "maps",
                                        "maps": {"full": {"shape": ["n_freq", 192], "value": 1.0},
                                                 "unit": "K"},
                                        "freq": {"list": [60e6, 70e6, 80e6], "unit": "Hz"},
                                        "nside": 4}}},
                context,
            )
        message = str(excinfo.value)
        assert "4" in message
        assert "3" in message

    def test_maps_must_be_two_dimensional(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "maps",
                                        "maps": {"full": {"shape": [192], "value": 1.0},
                                                 "unit": "K"},
                                        "freq": {"from_grid": "freq"},
                                        "nside": 4}}},
                context,
            )
        message = str(excinfo.value)
        assert "(n_freq, n_pix)" in message
        assert "192" in message

    def test_order_other_than_ring_is_refused(self, context):
        """The one message nothing else in this layer can catch: a NESTED map
        read as RING is the same shape, the same statistics, and every pixel
        in the wrong place."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "maps",
                                        "maps": {"full": {"shape": ["n_freq", 192], "value": 1.0},
                                                 "unit": "K"},
                                        "freq": {"from_grid": "freq"},
                                        "nside": 4,
                                        "order": "nested"}}},
                context,
            )
        message = str(excinfo.value)
        assert "nested" in message
        assert "RING" in message

    def test_maps_stray_key_is_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "maps",
                                        "maps": {"full": {"shape": ["n_freq", 192], "value": 1.0},
                                                 "unit": "K"},
                                        "freq": {"from_grid": "freq"},
                                        "nside": 4,
                                        "amplitude": {"value": 1.0, "unit": "K"}}}},
                context,
            )
        assert "amplitude" in str(excinfo.value)

    def test_a_map_declared_against_another_grid_is_refused_before_tracing(self, context):
        """The failure MapSky's own docstring says a config layer exists to
        catch: maps built for 60-85 MHz evaluated on a 100-125 MHz grid of the
        SAME length. MapSky cannot see it; this can, because it holds both."""
        other = jnp.linspace(100e6, 125e6, 4)
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"bad": {"kind": "maps",
                                        "maps": {"full": {"shape": ["n_freq", 192], "value": 1.0},
                                                 "unit": "K"},
                                        "freq": {"list": [float(v) for v in other], "unit": "Hz"},
                                        "nside": 4}}},
                context,
            )
        message = str(excinfo.value)
        assert "60" in message or "6e+07" in message or "60000000" in message
        assert "not interpolated" in message or "interpolat" in message


class TestRefusals:
    def test_an_unknown_kind_lists_the_five(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources({"sky_models": {"x": {"kind": "gsm"}}}, context)
        message = str(excinfo.value)
        for kind in ("uniform", "power_law", "maps", "gdsm", "python"):
            assert kind in message, kind

    def test_a_missing_kind_is_refused(self, context):
        with pytest.raises(ConfigError, match="kind"):
            build_resources({"sky_models": {"x": {"amplitude": 1.0}}}, context)


class TestGdsm:
    """kind: gdsm -- limTOD.sky_model.GDSM_sky_model, evaluated on the run's own grid."""

    @pytest.fixture
    def two_channels(self):
        return ResolutionContext(freq=jnp.asarray([70e6, 80e6]), time=jnp.arange(8.0),
                                 dtype="float32")

    def test_it_builds_a_mapsky_on_the_runs_own_grid(self, two_channels):
        pytest.importorskip("pygdsm")
        built = build_resources({"sky_models": {"real": {"kind": "gdsm", "nside": 8}}},
                                two_channels)
        sky = built.resources["resources.sky_models.real"]
        assert isinstance(sky, MapSky)
        assert sky.maps.shape == (2, 768)
        assert bool(jnp.allclose(sky.freq, two_channels.freq))

    def test_the_sky_has_structure_and_the_channels_differ(self, two_channels):
        """GSM16 at 70 MHz has structure (measured at nside 8: std ~ 2000 K on
        a mean of ~2800 K) and a synchrotron-steep spectrum. A flat map means
        the model call was replaced by a constant; identical channels mean the
        frequency loop collapsed to one call."""
        pytest.importorskip("pygdsm")
        built = build_resources({"sky_models": {"real": {"kind": "gdsm", "nside": 8}}},
                                two_channels)
        maps = built.resources["resources.sky_models.real"].maps
        assert float(maps[0].std()) > 0.0
        assert not bool(jnp.allclose(maps[0], maps[1]))

    def test_the_70_mhz_mean_is_in_kelvin(self):
        """Measured: GDSM_sky_model(freq=70.0, nside=8).mean() is 2757.6 K.
        The pin is wide -- (1000, 10000) -- because pygdsm's fit may move
        between releases; what it catches is a unit change (TCMB in K against
        MJysr differ by orders of magnitude) and an Hz-for-MHz slip in the
        conversion this builder owns."""
        pytest.importorskip("pygdsm")
        context = ResolutionContext(freq=jnp.asarray([70e6]), time=jnp.arange(8.0),
                                    dtype="float32")
        built = build_resources({"sky_models": {"real": {"kind": "gdsm", "nside": 8}}},
                                context)
        mean = float(built.resources["resources.sky_models.real"].maps.mean())
        assert 1000.0 < mean < 10000.0

    def test_it_requires_a_frequency_grid(self):
        """Checked BEFORE the pygdsm import, so this test needs no pygdsm."""
        context = ResolutionContext(freq=None, time=jnp.arange(8.0), dtype="float32")
        with pytest.raises(ConfigError, match="freq"):
            build_resources({"sky_models": {"real": {"kind": "gdsm", "nside": 8}}}, context)

    def test_stray_keys_are_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"sky_models": {"real": {"kind": "gdsm", "nside": 8,
                                         "amplitude": {"value": 1.0, "unit": "K"}}}},
                context,
            )
        assert "amplitude" in str(excinfo.value)

    def test_without_pygdsm_the_refusal_names_the_extra(self, monkeypatch, context):
        """Check A35's shape, same as the cal extra in Task 8: the alternative
        is a run that fails at first use, after everything else was built."""
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "pygdsm" or name.startswith("pygdsm."):
                raise ImportError("blocked for the test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(ConfigError) as excinfo:
            build_resources({"sky_models": {"real": {"kind": "gdsm", "nside": 8}}}, context)
        message = str(excinfo.value)
        assert "pygdsm" in message
        assert "limTOD[gdsm]" in message


class TestThePythonKindTypeChecks:
    def test_a_list_args_is_refused_as_not_a_mapping(self, context):
        spec = {"kind": "python", "python": "numpy:ones", "args": [3]}
        with pytest.raises(ConfigError, match="mapping of argument name"):
            build_sky_model("resources.sky_models.s", spec, context)
