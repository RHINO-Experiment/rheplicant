"""resources.arrays: composition by naming, which is why there is no expression language."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.resources import build_resources
from rheplicant.config.values import resolve_value


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 4), time=jnp.arange(8.0), dtype="float32"
    )


class TestANamedValueNode:
    def test_any_value_node_may_be_named(self, context):
        built = build_resources({"arrays": {"band": {"from_grid": "freq"}}}, context)
        assert built.resources["resources.arrays.band"].shape == (4,)

    def test_a_named_python_call_may_take_another_named_entry_as_an_argument(self, context):
        """This is v1's whole answer to 'the schema cannot express f(g(x), y)':
        composition is by naming, not by nesting. It unblocks
        examples/gibbs_plan.py:112-119 and sky_to_noise_wave.py:158-165, seven
        reflection coefficients built by nested rhino_cal_jax calls.

        DEVIATION from the plan's text: the plan's example target is
        ``jax.numpy:multiply``, called with ``args: {x1: ..., x2: ...}``. But
        ``jax.numpy.multiply`` is a ``jax.numpy.ufunc``, whose ``__call__``
        signature is ``(*args, out=None, where=None)`` -- it takes no keyword
        named ``x1``/``x2`` at all (confirmed against plain ``numpy`` too), so
        it cannot be reached through ``hatch._call``, which calls
        ``attribute(**keywords)`` (``src/rheplicant/config/hatch.py:164``).
        ``jax.numpy:dot`` is a plain Python-level function with real keyword
        parameters ``a``/``b`` and gives the identical result for a vector
        times a scalar, so it stands in for the plan's example without
        changing what the test demonstrates: a named python call taking
        another named entry as one of its arguments."""
        built = build_resources(
            {
                "arrays": {
                    "inner": {"list": [1.0, 2.0, 3.0, 4.0]},
                    "outer": {
                        "python": "jax.numpy:dot",
                        "args": {"a": {"ref": "resources.arrays.inner"}, "b": {"value": 2.0}},
                    },
                }
            },
            context,
        )
        assert [float(v) for v in built.resources["resources.arrays.outer"]] == pytest.approx(
            [2.0, 4.0, 6.0, 8.0]
        )
        # Pin identity, not values: jnp.dot returns the numerically right
        # answer whether 'a' was handed the SAME array 'inner' built or a
        # deep copy of it, so the assertion above cannot catch a
        # deep-copying resolve_reference mutant. Resolve the same
        # {ref: resources.arrays.inner} node the 'a' argument used, on its
        # own, and check the object that comes back is the one 'inner'
        # already built -- not merely an equal one.
        scoped = context.with_resource(
            "resources.arrays.inner", built.resources["resources.arrays.inner"]
        )
        resolved_again = resolve_value({"ref": "resources.arrays.inner"}, scoped).value
        assert resolved_again is built.resources["resources.arrays.inner"]

    def test_the_full_modifier_set_applies(self, context):
        """DEVIATION from the plan's text: the plan's example node is
        ``{list: [1+2j], unit: dimensionless, part: re}``. But
        ``arrays._list`` casts the literal to ``context.dtype`` ('float32'
        here) before any modifier runs, and ``jnp.asarray([1+2j],
        dtype='float32')`` raises ``TypeError`` -- this is a pre-existing,
        documented departure (see
        ``tests/config/test_config_modifiers.py::TestPart
        .test_each_part_of_a_complex_value``'s own docstring, MEASURED
        there). A complex value normally arrives here from ``ref:``,
        ``python:`` or ``file:``, and that test file already establishes the
        fix: a ``{value: ...}`` scalar node stands in for those, which is
        what this test uses too."""
        built = build_resources(
            {"arrays": {"g": {"value": 1.0 + 2.0j, "unit": "dimensionless", "part": "re"}}},
            context,
        )
        assert not jnp.iscomplexobj(built.resources["resources.arrays.g"])

    def test_an_entry_with_no_form_is_refused_as_a_value_node(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources({"arrays": {"g": {"unit": "K"}}}, context)
        assert "form key" in str(excinfo.value)
