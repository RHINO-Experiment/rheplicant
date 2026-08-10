"""model node specs -> operators: delivery, object fields, routes, eqx_leaves."""

import dataclasses

import equinox as eqx
import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError, ResolutionContext
from rheplicant.config.sections.model import (
    build_node_operator,
    operator_table,
)
from rheplicant.radio import (
    ADCOperator,
    BeamSpillOperator,
    GainOperator,
    GlobalSignalOperator,
    MatrixProjector,
    NoiseOperator,
    RadiometerNoiseOperator,
    SkySourceOperator,
    UniformSkyModel,
)

FREQ_HZ = jnp.linspace(60e6, 85e6, 8)
TIME_S = jnp.arange(0.0, 32.0, 2.0)


@pytest.fixture()
def context():
    return ResolutionContext(freq=FREQ_HZ, time=TIME_S, dtype="float32")


class TestTheTable:
    def test_every_shipped_operator_is_discoverable(self):
        table = operator_table()
        assert table["gain"] == (GainOperator,)
        assert set(cls.__name__ for cls in table["noise"]) == {
            "NoiseOperator", "RadiometerNoiseOperator"}
        assert len(table["flagging"]) == 2
        assert len(table["filters"]) == 3
        assert "astro_sum" not in table          # junctions register nothing
        assert "beam" not in table               # reserved: no shipped class


class TestFieldDelivery:
    def test_traced_and_static_fields_arrive_by_kind(self, context):
        op = build_node_operator(
            "adc",
            {"scale": {"value": 0.25, "unit": "adc_count/K"}, "n_bits": 12},
            context)
        assert isinstance(op, ADCOperator)
        assert op.n_bits == 12 and isinstance(op.n_bits, int)
        assert op.scale.dtype == jnp.float32

    def test_units_convert_on_the_way_in(self, context):
        op = build_node_operator(
            "global_signal",
            {"depth": {"value": 0.5, "unit": "K"},
             "centre": {"value": 75.0, "unit": "MHz"},
             "width": {"value": 5.0, "unit": "MHz"}},
            context)
        assert isinstance(op, GlobalSignalOperator)
        assert float(op.centre) == pytest.approx(75e6)

    def test_an_unknown_field_is_refused_listing_the_real_ones(self, context):
        with pytest.raises(ConfigError, match="depth"):
            build_node_operator("global_signal", {"dept": 0.5}, context)

    def test_a_missing_required_field_is_refused_by_name(self, context):
        with pytest.raises(ConfigError, match="width"):
            build_node_operator(
                "global_signal",
                {"depth": {"value": 0.5, "unit": "K"},
                 "centre": {"value": 75.0, "unit": "MHz"}},
                context)

    # No wrong-suffix test here: the suffix table (`_NAME_SUFFIX_DIMENSION`,
    # units.py:65) holds only `_deg`/`_m`, and no shipped operator field ends
    # in either -- the `check_field_name_unit` call in `_field_value` is
    # future-proofing, exercised by the units module's own tests.


class TestTypeSelection:
    def test_type_is_required_where_two_classes_register(self, context):
        with pytest.raises(ConfigError, match="type:"):
            build_node_operator("noise", {"sigma": {"value": 0.5, "unit": "K"}},
                                context)

    def test_type_picks_the_class(self, context):
        op = build_node_operator(
            "noise", {"type": "NoiseOperator",
                      "sigma": {"value": 0.5, "unit": "K"}}, context)
        assert isinstance(op, NoiseOperator)
        radiometer = build_node_operator(
            "noise", {"type": "RadiometerNoiseOperator",
                      "channel_width": {"value": 3.125, "unit": "MHz"},
                      "integration_time": {"value": 2.0, "unit": "s"}},
            context)
        assert isinstance(radiometer, RadiometerNoiseOperator)

    def test_an_unregistered_type_is_refused_listing_the_choices(self, context):
        with pytest.raises(ConfigError, match="NoiseOperator"):
            build_node_operator("noise", {"type": "GainOperator"}, context)

    def test_neural_operator_names_the_deferred_capability(self, context):
        with pytest.raises(ConfigError, match="capability 3"):
            build_node_operator("bandpass", {"type": "NeuralOperator"}, context)


class TestObjectFields:
    def _context_with_sky(self, context):
        sky = UniformSkyModel(amplitude=jnp.array(10.0), n_pix=12)
        projector = MatrixProjector(matrix=jnp.ones((4, 12)))
        return (context
                .with_resource("resources.sky_models.sky", sky)
                .with_resource("resources.projectors.p", projector))

    def test_object_fields_arrive_by_identity(self, context):
        ctx = self._context_with_sky(context)
        op = build_node_operator(
            "observed_astro_sky",
            {"sky_model": {"ref": "resources.sky_models.sky"},
             "projector": {"ref": "resources.projectors.p"}},
            ctx)
        assert isinstance(op, SkySourceOperator)
        assert op.sky_model is ctx.resources["resources.sky_models.sky"]

    def test_an_inline_object_field_is_refused(self, context):
        with pytest.raises(ConfigError, match=r"\{ref:"):
            build_node_operator(
                "observed_astro_sky",
                {"sky_model": {"value": 10.0}, "projector": {"value": 1.0}},
                context)


class TestFromRoutes:
    def test_beam_spill_from_projector(self, context):
        class WithFraction(eqx.Module):
            def horizon_fraction(self):
                return jnp.array(0.83)

        ctx = context.with_resource("resources.projectors.p", WithFraction())
        op = build_node_operator(
            "beam_spill",
            {"from": "projector", "projector": {"ref": "resources.projectors.p"},
             "t_ground": {"value": 290.0, "unit": "K"}},
            ctx)
        assert isinstance(op, BeamSpillOperator)
        assert float(op.sky_fraction) == pytest.approx(0.83)

    def test_beam_spill_from_projector_requires_t_ground(self, context):
        class WithFraction(eqx.Module):
            def horizon_fraction(self):
                return jnp.array(0.5)

        ctx = context.with_resource("resources.projectors.p", WithFraction())
        with pytest.raises(ConfigError, match="t_ground"):
            build_node_operator(
                "beam_spill",
                {"from": "projector",
                 "projector": {"ref": "resources.projectors.p"}},
                ctx)

    def test_t_sys_extra_from_basis(self, context):
        # BASIS_KINDS (core/basis.py) is ("legendre", "polynomial", "fourier")
        # -- not "chebyshev", which the reviewer's sketch named. Following the
        # real kinds table, per the one rule: fix the test, not the code.
        from rheplicant.config.kinds.bases import build_basis
        from rheplicant.radio import BasisTemperatureOperator

        basis = build_basis(
            "resources.bases.b",
            {"time": {"kind": "legendre", "n_basis": 2},
             "freq": {"kind": "legendre", "n_basis": 3}},
            context)
        ctx = context.with_resource("resources.bases.b", basis)
        op = build_node_operator(
            "t_sys_extra",
            {"from": "basis", "basis": {"ref": "resources.bases.b"},
             "coeff": {"zeros": [2, 3], "unit": "K"}},
            ctx)
        assert isinstance(op, BasisTemperatureOperator)
        assert op.coeff.shape == (2, 3)

    def test_an_unknown_route_is_refused(self, context):
        with pytest.raises(ConfigError, match="from:"):
            build_node_operator("gain", {"from": "nowhere"}, context)


class TestThermistorsRoute:
    h5py = pytest.importorskip("h5py",
                               reason="h5py comes with rheplicant[rhino]")

    def _ingested(self, tmp_path, ctx):
        """Widen the module's context FIXTURE VALUE (ctx) with the recording
        -- the fixture is not callable from a method body."""
        from rheplicant.config.sections.ingest import parse_from_file
        from tests.config.test_config_section_ingest import make_file

        make_file(tmp_path / "obs.hd5f")
        bootstrap = ResolutionContext(base_dir=str(tmp_path))
        obs, _ = parse_from_file(
            {"format": "rhino_hdf5", "path": "obs.hd5f", "freq_unit": "MHz",
             "settle_seconds": 0.0,
             "thermistor_columns": {"antenna": 0, "internal_load": 0,
                                    "heated_load": 1}},
            bootstrap)
        return dataclasses.replace(ctx, ingest=obs)

    def test_a_thermistor_column_becomes_t_load(self, tmp_path, context):
        operator = build_node_operator(
            "cal_loads", {"from": "thermistors", "label": "internal_load"},
            self._ingested(tmp_path, context))
        assert operator.t_load.ndim == 2
        assert operator.t_load.shape[1] == 1
        assert float(operator.t_load[0, 0]) == pytest.approx(293.15)

    def test_without_an_ingested_recording_it_is_refused(self, context):
        with pytest.raises(ConfigError, match="from_file"):
            build_node_operator(
                "cal_loads", {"from": "thermistors", "label": "ambient"},
                context)

    def test_label_is_required(self, tmp_path, context):
        with pytest.raises(ConfigError, match="label"):
            build_node_operator("cal_loads", {"from": "thermistors"},
                                self._ingested(tmp_path, context))

    def test_a_label_with_no_thermistor_is_the_readers_own_refusal(
            self, tmp_path, context):
        from rheplicant.core.errors import DataIngestionError

        with pytest.raises(DataIngestionError):
            build_node_operator(
                "cal_loads", {"from": "thermistors", "label": "ghost_load"},
                self._ingested(tmp_path, context))

    def test_an_empty_label_is_this_layers_refusal_not_the_readers(
            self, tmp_path, context):
        with pytest.raises(ConfigError, match="label"):
            build_node_operator(
                "cal_loads", {"from": "thermistors", "label": ""},
                self._ingested(tmp_path, context))

    def test_an_unknown_key_is_swept(self, tmp_path, context):
        with pytest.raises(ConfigError, match="smoothing"):
            build_node_operator(
                "cal_loads", {"from": "thermistors", "label": "internal_load",
                              "smoothing": 3},
                self._ingested(tmp_path, context))

    def test_the_document_threads_the_recording_to_the_model_build(
            self, tmp_path):
        """load_document -> build_model: the widening in document.py is what
        puts the recording on the context this route reads."""
        from rheplicant.config import load_document
        from rheplicant.radio import CalLoadOperator
        from tests.config.test_config_section_ingest import make_file

        make_file(tmp_path / "obs.hd5f")
        run = load_document({
            "schema_version": 1,
            "runtime": {"seed": 1},
            "observation": {
                "from_file": {"format": "rhino_hdf5", "path": "obs.hd5f",
                              "freq_unit": "MHz", "settle_seconds": 0.0,
                              "thermistor_columns": {"antenna": 0,
                                                     "internal_load": 0,
                                                     "heated_load": 1}},
                "switching": {"order": ["antenna", "internal_load",
                                        "heated_load"]},
            },
            "model": {
                "gain": {"gain": {"value": 2.0, "unit": "dimensionless"}},
                "cal_loads": {
                    "internal_load": {"from": "thermistors",
                                      "label": "internal_load"},
                    "heated_load": {"from": "thermistors",
                                    "label": "heated_load"},
                },
            },
            "runs": [{"kind": "forward"}],
        }, base_dir=str(tmp_path))
        operator = run.twin["cal_loads_1"]
        assert isinstance(operator, CalLoadOperator)
        assert operator.t_load.shape[1] == 1
        assert float(operator.t_load[0, 0]) == pytest.approx(293.15)


class TestThePythonHatch:
    def test_a_dotted_operator_class_constructs(self, context):
        op = build_node_operator(
            "gain",
            {"python": "rheplicant.radio:GainOperator",
             "gain": {"value": 1.1, "unit": "dimensionless"}},
            context)
        assert isinstance(op, GainOperator)

    def test_a_non_operator_target_is_refused(self, context):
        with pytest.raises(ConfigError, match="AbstractOperator"):
            build_node_operator("gain", {"python": "numpy:ones"}, context)

    def test_python_and_type_together_are_refused(self, context):
        with pytest.raises(ConfigError, match="one"):
            build_node_operator(
                "gain",
                {"python": "rheplicant.radio:GainOperator",
                 "type": "GainOperator",
                 "gain": {"value": 1.0, "unit": "dimensionless"}},
                context)


class TestEqxLeaves:
    def test_arrays_come_from_the_file_and_statics_from_the_document(
            self, context, tmp_path):
        saved = ADCOperator(scale=jnp.array(0.75), n_bits=14)
        path = tmp_path / "adc.eqx"
        eqx.tree_serialise_leaves(path, saved)
        ctx = ResolutionContext(freq=FREQ_HZ, time=TIME_S, dtype="float32",
                                base_dir=str(tmp_path))
        op = build_node_operator(
            "adc",
            {"scale": {"value": 0.25, "unit": "adc_count/K"}, "n_bits": 12,
             "eqx_leaves": {"path": "adc.eqx"}},
            ctx)
        assert float(op.scale) == pytest.approx(0.75)   # array: from the FILE
        assert op.n_bits == 12                          # static: from the DOCUMENT

    def test_a_wrong_sha256_is_refused(self, context, tmp_path):
        # files.py's mismatch refusal reads "{path} hashes to {digest}, and
        # this reference declares {declared}." -- it names the two digests
        # rather than the literal word "sha256" (confirmed against
        # tests/config/test_config_files.py::TestTheHash, which matches on
        # the declared value's own text, "0000", for the same refusal). The
        # one rule: fix the test to the real message, not the code to a
        # match string written from memory.
        saved = ADCOperator(scale=jnp.array(0.75), n_bits=14)
        path = tmp_path / "adc.eqx"
        eqx.tree_serialise_leaves(path, saved)
        ctx = ResolutionContext(freq=FREQ_HZ, time=TIME_S, dtype="float32",
                                base_dir=str(tmp_path))
        with pytest.raises(ConfigError, match="hashes to"):
            build_node_operator(
                "adc",
                {"scale": {"value": 0.25, "unit": "adc_count/K"}, "n_bits": 12,
                 "eqx_leaves": {"path": "adc.eqx", "sha256": "0" * 64}},
                ctx)

    def test_a_bare_file_node_is_routed_to_the_model_key(self, tmp_path):
        """The file must EXIST: files.py resolves the path and hashes it
        before any reader runs, so the reader's route-refusal is only
        reachable on a real file."""
        from rheplicant.config import resolve_value

        eqx.tree_serialise_leaves(
            tmp_path / "x.eqx", ADCOperator(scale=jnp.asarray(0.5), n_bits=8))
        ctx = ResolutionContext(freq=FREQ_HZ, time=TIME_S, dtype="float32",
                                base_dir=str(tmp_path))
        with pytest.raises(ConfigError, match=r"model\.<node>\.eqx_leaves"):
            resolve_value({"file": {"path": "x.eqx", "format": "eqx_leaves"}},
                          ctx)

    def test_unknown_keys_on_the_leaves_spec_are_refused(self, context):
        with pytest.raises(ConfigError, match=r"\['like'\]"):
            build_node_operator(
                "adc",
                {"scale": {"value": 0.25, "unit": "adc_count/K"}, "n_bits": 12,
                 "eqx_leaves": {"path": "x.eqx", "like": "template"}},
                context)
