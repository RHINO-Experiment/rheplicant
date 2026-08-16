"""model node specs -> operators: delivery, object fields, routes, eqx_leaves."""

import dataclasses
import types

import equinox as eqx
import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError, ResolutionContext
from rheplicant.config.sections.model import (
    build_node_operator,
    operator_table,
)
from rheplicant.config.sections.twin import build_fit_twin
from rheplicant.config.values import resolve_value
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


def _twin_with_beam_spill(ctx):
    """A twin carrying a ``beam_spill`` node, so ``replace:`` has one to hit.

    ``replace_node`` needs the node to already be lit; the operator it starts
    with is irrelevant to the refusal under test, so it is the cheapest legal
    one -- an explicit ``sky_fraction:``, which takes no projector at all.
    """
    from rheplicant.config.sections.compose import build_model

    return build_model(
        {"global_signal": {"depth": {"value": 0.5, "unit": "K"},
                           "centre": {"value": 75.0, "unit": "MHz"},
                           "width": {"value": 5.0, "unit": "MHz"}},
         "beam_spill": {"sky_fraction": {"value": 0.9,
                                         "unit": "dimensionless"},
                        "t_ground": {"value": 290.0, "unit": "K"}}},
        ctx, switch_order=())


class _NoFraction(eqx.Module):
    """A projector class that defines no ``horizon_fraction()`` at all."""


class _ReferenceFrame(eqx.Module):
    """A driftscan-shaped projector whose rotation has been cached.

    ``optimizations: [cache_beam_rotation]`` sets ``beam_frame='reference'``,
    and the real ``DriftScanProjector.horizon_fraction()`` then raises a
    ``StateValidationError`` -- the unmasked denominator has been folded away.
    The stand-in raises the same class for the same reason, so that neither
    route is being tested against a beam this test does not need.
    """

    beam_frame: str = eqx.field(static=True, default="reference")

    def horizon_fraction(self):
        from rheplicant.core.errors import StateValidationError

        raise StateValidationError(
            "horizon_fraction() needs the unmasked beam, but this projector "
            "has beam_frame='reference'."
        )


class TestBeamSpillFromProjectorSpeaksConfig:
    """Check C7's second route, re-voiced.

    ``{from: horizon_fraction}`` as a VALUE node and
    ``model.beam_spill: {from: projector}`` reach the same number, and
    measured at ``e0e024a`` only the first spoke config:

    ======================  ==============================  =================
    projector               model.beam_spill route          value-node route
    ======================  ==============================  =================
    ``MatrixProjector``     ``StateValidationError``        ``AttributeError``
    no ``horizon_fraction`` ``StateValidationError``        ``AttributeError``
    ``beam_frame`` cached   reaches ``horizon_fraction()``  ``ConfigError``
    ======================  ==============================  =================

    ``StateValidationError`` is a SIBLING of ``ConfigError`` rather than a
    subclass (0.2 C-12), so ``pytest.raises(ConfigError)`` never caught the
    left-hand column -- which is why nobody noticed.  The right-hand column's
    third row is the sentence that names ``optimizations:`` and the beam's own
    ``{ref: resources.beams.<n>.sky_fraction}``, and it is CALLED here rather
    than copied, so it keeps exactly one home in ``kinds/projectors.py``.
    """

    def _ctx(self, context, projector):
        return context.with_resource("resources.projectors.p", projector)

    def _build(self, ctx):
        return build_node_operator(
            "beam_spill",
            {"from": "projector",
             "projector": {"ref": "resources.projectors.p"},
             "t_ground": {"value": 290.0, "unit": "K"}},
            ctx)

    def test_a_cached_rotation_earns_the_value_routes_own_sentence(self,
                                                                   context):
        """The row that had NO config-level guard on this route at all."""
        with pytest.raises(ConfigError) as caught:
            self._build(self._ctx(context, _ReferenceFrame()))
        message = str(caught.value)
        assert message.startswith("model.beam_spill.projector: "
                                  "horizon_fraction: ")
        assert "optimizations: [cache_beam_rotation]" in message
        assert "{ref: resources.beams.<name>.sky_fraction}" in message

    def test_the_two_routes_say_the_same_thing_about_a_cached_rotation(
            self, context):
        """The two guards agree at the boundary, asserted rather than assumed.

        This is the property that makes calling the owner better than
        restating its condition: the model route's sentence is the value
        route's sentence, character for character, under one added key.
        A copy would pass on the day it was written and drift afterwards.
        """
        ctx = self._ctx(context, _ReferenceFrame())
        with pytest.raises(ConfigError) as by_model:
            self._build(ctx)
        with pytest.raises(ConfigError) as by_value:
            resolve_value({"from": "horizon_fraction",
                           "projector": {"ref": "resources.projectors.p"}},
                          ctx)
        assert str(by_model.value) == (
            f"model.beam_spill.projector: {by_value.value}")

    def test_a_projector_without_the_method_is_refused_as_a_ConfigError(
            self, context):
        """The class-naming sentence, kept -- and now catchable.

        The value-node route answers this one with a bare ``AttributeError``
        naming nothing (0.3 E.7), so there is no better sentence to borrow
        and ``from_projector``'s own is carried through instead.
        """
        with pytest.raises(ConfigError) as caught:
            self._build(self._ctx(context, _NoFraction()))
        assert str(caught.value) == (
            "model.beam_spill.projector: _NoFraction does not expose "
            "horizon_fraction(), so the above-horizon beam fraction cannot "
            "be read off it. Only DriftScanProjector defines the cut today "
            "(a fixed pointing makes it time-independent); for a scanning "
            "strategy the fraction varies per sample and needs a per-sample "
            "sky_fraction."
        )

    def test_a_matrix_projector_is_refused_by_name_and_not_by_AttributeError(
            self, context):
        """S3's named twin: the value-node route beside the model route.

        Measured: the value-node route raises ``'MatrixProjector' object has
        no attribute 'horizon_fraction'``, which is not a ``DirtError`` and
        names no document key. The model route no longer does anything of
        the sort. The false negative on the OTHER route is recorded in the
        task report rather than repaired here -- ``kinds/projectors.py`` is
        outside this task's files.
        """
        ctx = self._ctx(context, MatrixProjector(matrix=jnp.ones((4, 12))))
        with pytest.raises(ConfigError, match="MatrixProjector does not "
                                              "expose horizon_fraction"):
            self._build(ctx)
        with pytest.raises(AttributeError):
            resolve_value({"from": "horizon_fraction",
                           "projector": {"ref": "resources.projectors.p"}},
                          ctx)

    def test_the_replace_route_gets_the_same_refusal_as_the_model_route(
            self, context):
        """0.3 E.10, closed for C7 rather than recorded as a false negative.

        ``inference.twin.replace.<node>`` reaches ``build_node_operator``
        (``sections/twin.py:69``), which is the route ``preflight/model.py``'s
        ``_nodes()`` cannot see and the hole that ruling is about.  This fix
        does not walk ``model:`` as text -- it sits INSIDE ``_from_route``,
        below ``build_node_operator`` -- so both routes are covered by
        construction and neither can drift from the other.

        Asserted rather than argued, because "it is the same function" is
        exactly the claim that stops being true when someone adds a branch.
        """
        ctx = self._ctx(context, _ReferenceFrame())
        spec = {"from": "projector",
                "projector": {"ref": "resources.projectors.p"},
                "t_ground": {"value": 290.0, "unit": "K"}}
        with pytest.raises(ConfigError) as by_model:
            build_node_operator("beam_spill", spec, ctx)
        with pytest.raises(ConfigError) as by_replace:
            build_fit_twin({"replace": {"beam_spill": spec}},
                           _twin_with_beam_spill(ctx), ctx)
        assert str(by_replace.value) == str(by_model.value)

    def test_applying_the_refusals_own_advice_makes_the_document_build(
            self, context):
        """S4's second half for C7, which was missing.

        Both C7 sentences carry advice -- *"Take it from the beam instead,
        ``{ref: resources.beams.<name>.sky_fraction}``"* on the cached-rotation
        leg, and *"needs a per-sample sky_fraction"* on the wrong-class one.
        R4's four known loops on this plan were all found by RUNNING the
        advice, so a row nobody runs it on is where a fifth would hide. This
        runs the cached-rotation one: the refused document, then the same
        node written the way the sentence says, which must build.
        """
        beam = types.SimpleNamespace(maps=jnp.ones((8, 12)),
                                     sky_fraction=jnp.full((8,), 0.83))
        ctx = self._ctx(context, _ReferenceFrame()).with_resource(
            "resources.beams.horn", beam)
        with pytest.raises(ConfigError, match="sky_fraction"):
            self._build(ctx)
        operator = build_node_operator(
            "beam_spill",
            {"sky_fraction": {"ref": "resources.beams.horn.sky_fraction"},
             "t_ground": {"value": 290.0, "unit": "K"}},
            ctx)
        assert isinstance(operator, BeamSpillOperator)
        assert operator.sky_fraction.shape == (8,)

    def test_a_failure_that_is_not_the_packages_propagates_as_itself(
            self, context):
        """The handler catches ``StateValidationError`` and nothing wider.

        C-12 is the stated reason the handler exists, so the class it names
        is load-bearing. Widening it to ``except Exception`` left every test
        green while an unrelated failure -- an ``OSError`` from a beam read
        inside ``from_projector``, say -- would be re-labelled
        ``model.beam_spill.projector: ...`` as though the document were at
        fault.
        """
        class _Exploding(eqx.Module):
            def horizon_fraction(self):
                raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            self._build(self._ctx(context, _Exploding()))

    def test_the_package_classes_are_siblings_of_ConfigError(self):
        """0.2 C-12 measured here, because it is this row's whole cause."""
        from rheplicant.core.errors import DirtError, StateValidationError

        assert not issubclass(StateValidationError, ConfigError)
        assert issubclass(StateValidationError, DirtError)
        assert issubclass(ConfigError, DirtError)

    def test_a_working_projector_is_untouched_and_pays_for_no_second_read(
            self, context):
        """The happy path does not enter the re-voicing branch at all.

        ``horizon_fraction()`` is a beam integral on a real projector, so a
        guard that ran it twice to decide whether to complain would be a
        real cost on every document that builds. Counting the calls is the
        only assertion that can tell.
        """
        calls = []

        class _Counting(eqx.Module):
            def horizon_fraction(self):
                calls.append(1)
                return jnp.array(0.83)

        op = self._build(self._ctx(context, _Counting()))
        assert isinstance(op, BeamSpillOperator)
        assert float(op.sky_fraction) == pytest.approx(0.83)
        assert len(calls) == 1


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
