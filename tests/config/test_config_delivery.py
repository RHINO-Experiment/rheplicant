"""Static vs traced: the destination field decides, and the decision is measured."""

import dataclasses
from collections.abc import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.delivery import FieldSpec, deliver, field_specs
from rheplicant.core.frozen import FrozenMapping
from rheplicant.core.operator import SnapshotOperator
from rheplicant.radio.backend.flagging import MomentRFIFlaggingOperator
from rheplicant.radio.instrument.adc import ADCOperator
from rheplicant.radio.instrument.antenna_loss import AntennaLossOperator
from rheplicant.radio.sky.foregrounds import ForegroundOperator


class TestReadingTheFieldMetadata:
    def test_a_static_int_is_recognised(self):
        spec = field_specs(ADCOperator)["n_bits"]
        assert spec.static is True
        assert spec.annotation is int

    def test_a_traced_array_is_recognised(self):
        spec = field_specs(ADCOperator)["scale"]
        assert spec.static is False

    def test_a_converter_is_reported(self):
        """equinox runs the converter one line BEFORE the static-array check
        (equinox/_module/_module.py:415-417), which is why
        CWCalibrationOperator accepts a 0-d array with no warning. A caller
        that wants to know whether the destination will clean up after it has
        to be able to see the converter."""
        assert field_specs(AntennaLossOperator)["efficiency"].converter is jnp.asarray

    def test_every_operator_class_in_the_package_can_be_inspected(self):
        """72 eqx.Module subclasses, 257 fields, 0 string annotations today --
        but four modules use `from __future__ import annotations`, so the
        helper resolves hints rather than reading f.type."""
        from rheplicant.core.operator import AbstractOperator

        seen = 0
        stack = [AbstractOperator]
        while stack:
            cls = stack.pop()
            stack.extend(cls.__subclasses__())
            if not dataclasses.is_dataclass(cls):
                continue
            specs = field_specs(cls)
            assert isinstance(specs, dict)
            seen += 1
        assert seen > 20, seen

    def test_a_default_factory_counts_as_a_default(self):
        """Catches a `required` that only tests f.default:
        MomentRFIFlaggingOperator.config has default=MISSING and
        default_factory=FrozenMapping, so checking f.default alone calls an
        optional field required and every document without it is refused."""
        assert field_specs(MomentRFIFlaggingOperator)["config"].required is False
        assert field_specs(ADCOperator)["n_bits"].required is True


class TestTracedDeliveryForcesAFloatingDtype:
    def test_an_integer_scalar_becomes_a_float_array(self):
        """MEASURED: AntennaLossOperator(efficiency=1) stores int32, and
        eqx.partition(op, eqx.is_inexact_array) then returns [] -- the field
        is silently untrainable, with no warning and an identical forward
        pass. A YAML `1` must not do that."""
        spec = field_specs(AntennaLossOperator)["efficiency"]
        value = deliver(1, spec, dtype="float32")
        assert jnp.issubdtype(value.dtype, jnp.floating)

    def test_the_delivered_field_is_trainable(self):
        spec = field_specs(AntennaLossOperator)["efficiency"]
        op = AntennaLossOperator(
            efficiency=deliver(1, spec, dtype="float32"),
            t_physical=deliver(
                300, field_specs(AntennaLossOperator)["t_physical"], dtype="float32"
            ),
        )
        params, _ = eqx.partition(op, eqx.is_inexact_array)
        assert jax.tree_util.tree_leaves(params), "delivered fields must be trainable"

    def test_the_run_dtype_is_honoured(self):
        spec = field_specs(AntennaLossOperator)["efficiency"]
        assert deliver(1.0, spec, dtype="float32").dtype == jnp.float32

    def test_a_complex_value_keeps_its_complex_dtype(self):
        """Forcing float on a reflection coefficient would delete the phase."""
        spec = field_specs(AntennaLossOperator)["efficiency"]
        value = deliver(1 + 2j, spec, dtype="float32")
        assert jnp.issubdtype(value.dtype, jnp.complexfloating)

    def test_a_boolean_mask_stays_boolean(self):
        """Catches deletion of the bool branch, which would send a mask
        through astype(dtype) and hand back 1.0/0.0. A float mask still
        indexes and still multiplies, so the forward pass looks right, but
        `~mask` becomes -2.0 and the flag is inverted rather than negated."""
        spec = field_specs(ForegroundOperator)["amplitude"]
        value = deliver([True, False], spec, dtype="float32")
        assert jnp.issubdtype(value.dtype, jnp.bool_)


class TestStaticDelivery:
    def test_a_static_int_arrives_as_a_python_int(self):
        spec = field_specs(ADCOperator)["n_bits"]
        value = deliver(12, spec, dtype="float32")
        assert value == 12 and type(value) is int

    def test_a_bool_on_a_static_int_is_refused(self):
        """isinstance(True, int) is True, so ADCOperator(n_bits=True)
        constructs and gives a 1-bit ADC. The annotation is `int`, not `bool`,
        and the two are told apart by annotation identity, not isinstance."""
        spec = field_specs(ADCOperator)["n_bits"]
        with pytest.raises(ConfigError) as excinfo:
            deliver(True, spec, dtype="float32")
        message = str(excinfo.value)
        assert "n_bits" in message
        assert "bool" in message

    def test_a_float_on_a_static_int_is_refused_here_rather_than_downstream(self):
        """ADCOperator's own guard raises 'n_bits must be a positive int, got
        8.0' -- but only after the operator has been built, which for a beam
        node is after the CST directory has been read and analysed."""
        spec = field_specs(ADCOperator)["n_bits"]
        with pytest.raises(ConfigError, match="n_bits"):
            deliver(8.0, spec, dtype="float32")

    def test_a_static_float_arrives_as_a_python_float(self):
        spec = field_specs(ForegroundOperator)["ref_freq"]
        value = deliver(1.4e8, spec, dtype="float32")
        assert type(value) is float

    def test_a_list_on_a_static_tuple_field_becomes_a_tuple(self):
        """MEASURED: a list of lists on a static tuple field constructs fine
        and makes the whole module unhashable -- `hash(op)` raises
        TypeError: unhashable type: 'list'. A YAML sequence is a list."""
        spec = FieldSpec("kernel_shapes", tuple, True, None, True)
        value = deliver([[3, 3], [5, 5]], spec, dtype="float32")
        assert value == ((3, 3), (5, 5))
        assert hash(value)

    def test_a_mapping_on_a_static_mapping_field_becomes_a_frozen_mapping(self):
        from collections.abc import Mapping

        spec = FieldSpec("config", Mapping, True, None, True)
        value = deliver({"a": 1}, spec, dtype="float32")
        assert isinstance(value, FrozenMapping)

    def test_an_int_on_a_static_float_field_arrives_as_a_python_float(self):
        """Catches `_as_static_float` returning the value without float().
        The plan's other static-float test passes 1.4e8, which is already a
        float and so cannot tell the two apart. A YAML `300` on a float field
        is an int, and an int that reaches a float field is the same silent
        int32 story one layer up."""
        spec = field_specs(ForegroundOperator)["ref_freq"]
        value = deliver(300, spec, dtype="float32")
        assert type(value) is float and value == 300.0

    def test_a_static_str_takes_a_str_and_refuses_a_number(self):
        """Catches `_as_static_str` passing anything through. SnapshotOperator
        .name lands in the treedef, so a number there is a second jit cache
        entry that renders under a label nobody wrote."""
        spec = field_specs(SnapshotOperator)["name"]
        assert deliver("raw", spec, dtype="float32") == "raw"
        with pytest.raises(ConfigError, match="name"):
            deliver(3, spec, dtype="float32")

    def test_a_static_bool_takes_only_a_bool(self):
        """Catches `_as_static_bool` passing anything through. Truthiness is
        the trap this exists for: `deliver(0.0, ...)` would become a valid
        False and `deliver("false", ...)` a valid True."""
        spec = FieldSpec("apply", bool, True, None, True)
        assert deliver(True, spec, dtype="float32") is True
        with pytest.raises(ConfigError, match="apply"):
            deliver("false", spec, dtype="float32")

    def test_a_field_this_grammar_cannot_build_is_passed_through(self):
        """Catches `mode_of` letting an unclassifiable static field fall
        through to "traced", which would call jnp.asarray on a callable or a
        nested Module and raise somewhere with no field name in it."""
        spec = FieldSpec("fn", Callable, True, None, True)
        assert deliver(len, spec, dtype="float32") is len


class TestCheckA40:
    def test_an_array_shaped_form_on_a_static_field_is_refused_by_name(self):
        """MEASURED: ADCOperator(n_bits=jnp.asarray(12)) warns 'A JAX array is
        being set as static!' then raises. ForegroundOperator(ref_freq=array)
        only warns -- it constructs, and filter_grad then hands back 1.4e+08
        in the gradient slot, which reads exactly like a gradient."""
        spec = field_specs(ForegroundOperator)["ref_freq"]
        with pytest.raises(ConfigError) as excinfo:
            deliver(1.4e8, spec, dtype="float32", source="linspace")
        message = str(excinfo.value)
        assert "ref_freq" in message  # the field
        assert "linspace" in message  # the form that cannot land here
        assert "static" in message  # why
        assert "jit" in message  # what it corrupts

    def test_a_scalar_form_on_a_static_field_is_accepted(self):
        spec = field_specs(ForegroundOperator)["ref_freq"]
        assert deliver(1.4e8, spec, dtype="float32", source="scalar") == 1.4e8

    def test_an_array_form_on_a_traced_field_is_what_traced_fields_are_for(self):
        """Catches an A40 that drops its `mode != "traced"` guard and refuses
        array forms everywhere. That mutation passes every other test in this
        file -- they all aim array forms at static fields -- while making a
        linspace-valued amplitude, the ordinary case, unwritable."""
        spec = field_specs(ForegroundOperator)["amplitude"]
        value = deliver([1.0, 2.0], spec, dtype="float32", source="linspace")
        assert value.shape == (2,)


class TestTheDeclaredAsModifier:
    def test_a_matching_as_is_accepted(self):
        spec = field_specs(ADCOperator)["n_bits"]
        assert deliver(12, spec, dtype="float32", declared_as="static_int") == 12

    def test_a_contradicted_as_is_refused_and_names_both(self):
        spec = field_specs(ADCOperator)["n_bits"]
        with pytest.raises(ConfigError) as excinfo:
            deliver(12, spec, dtype="float32", declared_as="traced")
        message = str(excinfo.value)
        assert "traced" in message  # what the document said
        assert "static_int" in message  # what the field says
        assert "n_bits" in message

    def test_an_unknown_as_is_refused_and_the_set_is_listed(self):
        spec = field_specs(ADCOperator)["n_bits"]
        with pytest.raises(ConfigError) as excinfo:
            deliver(12, spec, dtype="float32", declared_as="int")
        message = str(excinfo.value)
        for token in ("traced", "static_int", "static_float", "static_str"):
            assert token in message


class TestNumpyNeverReachesAField:
    def test_a_numpy_scalar_on_a_static_int_is_refused(self):
        """MEASURED: np.int64(8) is rejected by ADCOperator's isinstance guard
        AND trips the equinox array warning, because numpy.generic is in
        equinox's _ARRAY_TYPES."""
        import numpy as np

        spec = field_specs(ADCOperator)["n_bits"]
        with pytest.raises(ConfigError, match="n_bits"):
            deliver(np.int64(8), spec, dtype="float32")

    def test_a_numpy_scalar_on_a_static_float_is_refused(self):
        """The int case above cannot see _reject_numpy at all: deleting the
        check still leaves isinstance(np.int64(8), int) == False, so the
        generic int guard raises and the test passes on a broken
        implementation. The float field is where the check is load-bearing --
        isinstance(np.float64(8.0), float) is True, so without _reject_numpy a
        numpy scalar sails into the treedef and trips equinox's array warning
        at construction instead."""
        import numpy as np

        spec = field_specs(ForegroundOperator)["ref_freq"]
        with pytest.raises(ConfigError, match="ref_freq"):
            deliver(np.float64(8.0), spec, dtype="float32")
