"""resources.bases, and the basis_fit value form."""

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.kinds.bases import build_basis
from rheplicant.config.resources import build_resources
from rheplicant.config.values import resolve_value
from rheplicant.core.basis import SeparableBasis


@pytest.fixture
def context():
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 8), time=jnp.arange(16.0), dtype="float32"
    )


class TestBuildingABasis:
    def test_it_builds_a_separable_basis_from_the_runs_own_grids(self, context):
        built = build_resources(
            {"bases": {"t_ant": {"time": {"kind": "legendre", "n_basis": 3},
                                 "freq": {"kind": "legendre", "n_basis": 2}}}},
            context,
        )
        basis = built.resources["resources.bases.t_ant"]
        assert isinstance(basis, SeparableBasis)
        assert basis.shape == (16, 8)
        assert basis.coeff_shape == (3, 2)

    def test_n_is_never_written(self, context):
        """radio/t_sys.py: a basis built for another band 'would return a
        smooth, plausible, wrong temperature'. Taking n from the grid is what
        makes that structurally impossible."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"bases": {"b": {"time": {"kind": "legendre", "n_basis": 3, "n": 16},
                                 "freq": {"kind": "legendre", "n_basis": 2}}}},
                context,
            )
        message = str(excinfo.value)
        assert "'n'" in message or '"n"' in message
        assert "grid" in message

    def test_a_file_route_is_refused_with_its_reason_and_its_alternative(self, context):
        """§7's named refusal, and open question 11.8 recommends keeping it.
        A copied built_for: block is exactly what a copied basis comes with."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"bases": {"b": {"time": {"file": {"path": "design.npy", "format": "npy"}},
                                 "freq": {"kind": "legendre", "n_basis": 2}}}},
                context,
            )
        message = str(excinfo.value)
        assert "kind" in message
        assert "n_basis" in message
        assert "python:" in message

    def test_both_axes_are_required(self, context):
        with pytest.raises(ConfigError, match="freq"):
            build_resources({"bases": {"b": {"time": {"kind": "legendre", "n_basis": 3}}}}, context)

    def test_the_three_kinds_are_the_packages_own(self, context):
        from rheplicant.core.basis import BASIS_KINDS

        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"bases": {"b": {"time": {"kind": "chebyshev", "n_basis": 3},
                                 "freq": {"kind": "legendre", "n_basis": 2}}}},
                context,
            )
        message = str(excinfo.value)
        assert "chebyshev" in message  # what was asked, not only what is available
        for kind in BASIS_KINDS:
            assert kind in message, kind

    def test_a_missing_grid_is_refused_by_name(self):
        """Kills the mutant that drops the `if grid is None` guard: with the
        axis check gone, `int(None.shape[0])` would raise AttributeError
        instead of a ConfigError naming the axis and the grid it needs."""
        no_time = ResolutionContext(freq=jnp.linspace(60e6, 85e6, 8), time=None, dtype="float32")
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"bases": {"b": {"time": {"kind": "legendre", "n_basis": 2},
                                 "freq": {"kind": "legendre", "n_basis": 2}}}},
                no_time,
            )
        assert "observation.time.grid" in str(excinfo.value)


class TestBasisFit:
    def test_it_fits_a_field_onto_a_named_basis(self, context):
        built = build_resources(
            {"bases": {"b": {"time": {"kind": "legendre", "n_basis": 3},
                             "freq": {"kind": "legendre", "n_basis": 2}}}},
            context,
        )
        scoped = context
        for name, value in built.resources.items():
            scoped = scoped.with_resource(name, value)
        # coeff_shape is (3, 2) -- deliberately NOT square. core/basis.py's own
        # docstring names a swapped pair of design matrices as shape-legal and
        # silently transposed whenever the two axes happen to match; a square
        # result here would let exactly that mutant (basis.fit(field).T) pass
        # this test unnoticed. The field also varies along time only, so the
        # fitted coefficients are asymmetric in content as well as shape.
        field_values = [[float(t) for _ in range(8)] for t in range(16)]
        got = resolve_value(
            {"basis_fit": {"basis": {"ref": "resources.bases.b"},
                           "field": {"list": field_values, "unit": "K"}},
             "unit": "K"},
            scoped,
        )
        assert got.value.shape == (3, 2)
        assert got.unit.canonical == "K"
        assert float(got.value[0, 0]) != 0.0  # the constant function is first for every kind

    def test_the_basis_must_be_a_ref_not_inline(self, context):
        """The docstring's identity argument: a basis inline would be a SECOND
        object built for the same grid, not the one everything else fitting
        on it shares -- so an inline mapping is refused rather than accepted
        and silently rebuilt."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"basis_fit": {"basis": {"value": 5.0}, "field": {"value": 5.0}}},
                context,
            )
        assert "must be a {ref:" in str(excinfo.value)

    def test_a_ref_to_something_that_is_not_a_basis_is_refused_by_type(self, context):
        """Load-bearing for the same identity argument: `ref` happily resolves
        to any resource, so this form has to check what it got rather than
        trust the name. The message names the actual type -- this is the
        check whose wording used to read 'is a ArrayImpl', grammatically
        wrong in a way that suggested nobody had actually triggered it."""
        built = build_resources({"arrays": {"a": {"list": [1.0, 2.0]}}}, context)
        scoped = context.with_resource("resources.arrays.a", built.resources["resources.arrays.a"])
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"basis_fit": {"basis": {"ref": "resources.arrays.a"}, "field": {"value": 5.0}}},
                scoped,
            )
        message = str(excinfo.value)
        assert "resources.arrays.a" in message
        assert "SeparableBasis" in message
        assert type(built.resources["resources.arrays.a"]).__name__ in message  # "ArrayImpl"

    def test_a_missing_field_is_refused_with_the_expects_shape(self, context):
        """Pins the specific 'missing' variant of the split refusal (item 6):
        {basis} alone is missing 'field', which is not the same failure as an
        unknown key or a non-mapping node, and the three now read differently."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_value({"basis_fit": {"basis": {"ref": "resources.bases.b"}}}, context)
        message = str(excinfo.value)
        assert "expects" in message
        assert "missing ['field']" in message

    def test_the_form_is_registered_so_the_grammar_is_now_complete(self):
        from rheplicant.config.values import _RESOLVERS, VALUE_FORMS

        assert set(VALUE_FORMS) - {"value"} == set(_RESOLVERS)


class TestBasisFitRefusals:
    """_basis_fit's own refusals, pinned (the 1B review found them untested)."""

    def test_a_non_mapping_spec_is_refused(self, context):
        with pytest.raises(ConfigError, match="basis_fit: expects"):
            resolve_value({"basis_fit": [1, 2]}, context)

    def test_an_unknown_key_is_refused(self, context):
        node = {"basis_fit": {"basis": {"ref": "resources.bases.b"},
                              "field": {"zeros": [2, 2]}, "amplitude": 1.0}}
        with pytest.raises(ConfigError, match=r"does not take \['amplitude'\]"):
            resolve_value(node, context)


class TestTheAxisRefusalsCarryTheResourceName:
    def test_a_bad_axis_spec_names_the_resource(self, context):
        spec = {"time": "chebyshev", "freq": {"kind": "chebyshev", "n_basis": 3}}
        with pytest.raises(ConfigError) as excinfo:
            build_basis("resources.bases.mine", spec, context)
        assert "resources.bases.mine" in str(excinfo.value)

    def test_the_axis_sweep_names_the_resource(self, context):
        spec = {
            "time": {"kind": "chebyshev", "n_basis": 3, "stray": 1},
            "freq": {"kind": "chebyshev", "n_basis": 3},
        }
        with pytest.raises(ConfigError) as excinfo:
            build_basis("resources.bases.mine", spec, context)
        message = str(excinfo.value)
        assert "resources.bases.mine" in message
        assert "stray" in message
