"""resources.s_params: three kinds, and the component a 1-port file cannot give."""

import dataclasses

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.context import ResolutionContext
from rheplicant.config.resources import build_resources
from rheplicant.config.values import resolve_value

_S1P = """! a one-port device
# HZ S RI R 50
60000000  0.10  0.00
85000000  0.20  0.00
"""

#: A two-port fixture whose s11 (0.10->0.20) and s22 (0.70->0.80) sweeps
#: differ, so flipped: true (which swaps s11<->s22) is verifiable directly:
#: reading component: s11 with flipped: true must return the s22 numbers.
_S2P = """! a two-port device
# HZ S RI R 50
60000000  0.10  0.00  0.30  0.00  0.40  0.00  0.70  0.00
85000000  0.20  0.00  0.30  0.00  0.40  0.00  0.80  0.00
"""

# The 4-point run grid interpolates linearly between the file's two
# frequencies, so every test against it can pin the same two arrays.
_EXPECTED_S11 = [0.1, 0.1333333, 0.1666667, 0.2]
_EXPECTED_S22 = [0.7, 0.7333333, 0.7666667, 0.8]


def _real_parts(array):
    return [complex(v).real for v in array]


@pytest.fixture
def context(tmp_path):
    (tmp_path / "horn.s1p").write_text(_S1P)
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 4), time=jnp.arange(8.0), dtype="float32",
        base_dir=str(tmp_path),
    )


@pytest.fixture
def two_port_context(tmp_path):
    (tmp_path / "horn.s2p").write_text(_S2P)
    return ResolutionContext(
        freq=jnp.linspace(60e6, 85e6, 4), time=jnp.arange(8.0), dtype="float32",
        base_dir=str(tmp_path),
    )


class TestTouchstone:
    def test_it_reads_and_interpolates_onto_the_runs_grid(self, context):
        built = build_resources(
            {"s_params": {"horn": {"kind": "touchstone",
                                   "file": {"path": "horn.s1p", "format": "touchstone"},
                                   "component": "s11", "onto": "freq"}}},
            context,
        )
        gamma = built.resources["resources.s_params.horn"]
        assert gamma.shape == (4,)
        assert jnp.iscomplexobj(gamma)
        assert _real_parts(gamma) == pytest.approx(_EXPECTED_S11, rel=1e-4)

    def test_the_three_parts_are_exposed(self, context):
        built = build_resources(
            {"s_params": {"horn": {"kind": "touchstone",
                                   "file": {"path": "horn.s1p", "format": "touchstone"},
                                   "component": "s11", "onto": "freq"}},
             "arrays": {"re": {"ref": "resources.s_params.horn", "part": "re"}}},
            context,
        )
        assert not jnp.iscomplexobj(built.resources["resources.arrays.re"])

    def test_asking_a_one_port_file_for_s21_is_refused_by_the_packages_own_guard(self, context):
        """touchstone.py:67-74 -- 'a zero would read as a perfectly isolated
        port'. The refusal belongs to the reader; this layer must not swallow it."""
        from rheplicant.core.errors import DataIngestionError

        with pytest.raises((ConfigError, DataIngestionError)):
            build_resources(
                {"s_params": {"horn": {"kind": "touchstone",
                                       "file": {"path": "horn.s1p", "format": "touchstone"},
                                       "component": "s21", "onto": "freq"}}},
                context,
            )

    def test_an_unknown_component_lists_the_four(self, context):
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"s_params": {"horn": {"kind": "touchstone",
                                       "file": {"path": "horn.s1p", "format": "touchstone"},
                                       "component": "s13", "onto": "freq"}}},
                context,
            )
        message = str(excinfo.value)
        for component in ("s11", "s12", "s21", "s22"):
            assert component in message

    def test_flipped_is_recorded_because_it_cannot_be_recovered(self, context):
        """The WIRING -- whether the device genuinely was connected to the
        VNA reversed -- is unverifiable after the fact: Touchstone has
        exactly three fields and the flip is applied into `s` before one is
        even constructed, so a config that records flipped: true cannot be
        checked against the file it came from. The PLUMBING is a different
        claim and IS verifiable -- see TestFlippedIsVerifiable below, which
        pins it directly with a 2-port fixture whose s11 and s22 sweeps
        differ. This test only checks that flipped: False on a 1-port file
        still builds (flipped: True there is refused by read_touchstone's
        own guard -- port reversal exchanges two ports and a 1-port file
        has one)."""
        built = build_resources(
            {"s_params": {"horn": {"kind": "touchstone",
                                   "file": {"path": "horn.s1p", "format": "touchstone"},
                                   "component": "s11", "flipped": False, "onto": "freq"}}},
            context,
        )
        assert "resources.s_params.horn" in built.resources


class TestFlippedIsVerifiable:
    """The PLUMBING -- whether writing flipped: on this entry actually
    reaches read_touchstone(flipped=) and swaps the matrix -- is a
    different claim from the WIRING (see the docstring above), and IS
    verifiable: read_touchstone applies the port reversal into `s` before
    Touchstone is even constructed, so a 2-port fixture with distinct
    s11/s22 sweeps proves it directly."""

    def _s11(self, context, *, flipped):
        entry = {"kind": "touchstone",
                  "file": {"path": "horn.s2p", "format": "touchstone"},
                  "component": "s11", "onto": "freq"}
        if flipped is not None:
            entry["flipped"] = flipped
        built = build_resources({"s_params": {"horn": entry}}, context)
        return built.resources["resources.s_params.horn"]

    def test_flipped_true_reads_the_original_s22_sweep(self, two_port_context):
        gamma = self._s11(two_port_context, flipped=True)
        assert _real_parts(gamma) == pytest.approx(_EXPECTED_S22, rel=1e-4)

    def test_flipped_absent_reads_the_original_s11_sweep(self, two_port_context):
        gamma = self._s11(two_port_context, flipped=None)
        assert _real_parts(gamma) == pytest.approx(_EXPECTED_S11, rel=1e-4)


class TestAllowExtrapolation:
    """A run grid wider than the file's own sweep: refused by default
    (_interp_strict's own tolerance-band guard), accepted and clamped to
    the edge values with allow_extrapolation: true."""

    @pytest.fixture
    def wide_context(self, tmp_path):
        (tmp_path / "horn.s1p").write_text(_S1P)
        return ResolutionContext(
            freq=jnp.array([50e6, 95e6]), time=jnp.arange(8.0), dtype="float32",
            base_dir=str(tmp_path),
        )

    def test_a_wider_run_grid_is_refused_by_default(self, wide_context):
        from rheplicant.core.errors import DataIngestionError

        with pytest.raises((ConfigError, DataIngestionError)):
            build_resources(
                {"s_params": {"horn": {"kind": "touchstone",
                                       "file": {"path": "horn.s1p", "format": "touchstone"},
                                       "component": "s11", "onto": "freq"}}},
                wide_context,
            )

    def test_allow_extrapolation_clamps_to_the_edge_values(self, wide_context):
        built = build_resources(
            {"s_params": {"horn": {"kind": "touchstone",
                                   "file": {"path": "horn.s1p", "format": "touchstone"},
                                   "component": "s11", "onto": "freq",
                                   "allow_extrapolation": True}}},
            wide_context,
        )
        gamma = built.resources["resources.s_params.horn"]
        assert _real_parts(gamma) == pytest.approx([0.1, 0.2], rel=1e-4)


class TestTheObjectReader:
    """format: touchstone is registered array=False (files.py): the file:
    value node hands back the Touchstone object itself, not a jnp-wrapped
    array, and refuses any modifiers written on that node -- a modifier
    describes what an array's numbers ARE and a Touchstone is not one."""

    def test_the_file_node_returns_a_touchstone_object(self, context):
        from rheplicant.radio.touchstone import Touchstone

        resolved = resolve_value(
            {"file": {"path": "horn.s1p", "format": "touchstone"}}, context
        )
        assert isinstance(resolved.value, Touchstone)

    def test_modifiers_on_an_object_valued_file_node_are_refused(self, context):
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"file": {"path": "horn.s1p", "format": "touchstone"}, "unit": "dimensionless"},
                context,
            )
        message = str(excinfo.value)
        assert "modifiers" in message
        assert "Touchstone" in message


class TestInterpolateOntoDerivation:
    """The 'interpolate_onto' derivation (form 6, `from:`) -- distinct from
    kind: touchstone, which never returns a raw Touchstone: every
    resources.s_params entry is already the interpolated array. This
    derivation is how a value node elsewhere in a document reads a
    Touchstone and interpolates it in one step, without a resources.s_params
    entry at all."""

    def test_the_happy_path_pins_the_same_values_as_kind_touchstone(self, context):
        resolved = resolve_value(
            {"from": "interpolate_onto",
             "of": {"file": {"path": "horn.s1p", "format": "touchstone"}},
             "component": "s11"},
            context,
        )
        assert _real_parts(resolved.value) == pytest.approx(_EXPECTED_S11, rel=1e-4)

    def test_of_resolving_to_an_array_is_refused_and_names_the_type(self, context):
        probe = jnp.array([1.0, 2.0, 3.0, 4.0])
        with_array = dataclasses.replace(context, resources={"resources.arrays.probe": probe})
        with pytest.raises(ConfigError) as excinfo:
            resolve_value(
                {"from": "interpolate_onto", "of": {"ref": "resources.arrays.probe"},
                 "component": "s11"},
                with_array,
            )
        message = str(excinfo.value)
        assert "Touchstone" in message
        assert type(probe).__name__ in message


class TestTheCalKinds:
    def test_termination_needs_the_cal_extra(self, context):
        pytest.importorskip("rhino_cal_jax")
        built = build_resources(
            {"s_params": {"open": {"kind": "termination", "termination": "open",
                                   "z0": {"value": 50.0, "unit": "ohm"}, "n": "n_freq"}}},
            context,
        )
        assert built.resources["resources.s_params.open"].shape == (4,)

    def test_resistive_requires_an_impedance(self, context):
        pytest.importorskip("rhino_cal_jax")
        with pytest.raises(ConfigError, match="impedance"):
            build_resources(
                {"s_params": {"r": {"kind": "termination", "termination": "resistive",
                                    "z0": {"value": 50.0, "unit": "ohm"}, "n": "n_freq"}}},
                context,
            )

    def test_a_missing_optional_dependency_names_the_extra(self, monkeypatch, context):
        """Check A35: a 40-minute failure at the first import is what this
        exists to prevent."""
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name.startswith("rhino_cal_jax"):
                raise ImportError("blocked for the test")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _blocked)
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"s_params": {"open": {"kind": "termination", "termination": "open",
                                       "z0": {"value": 50.0, "unit": "ohm"}, "n": "n_freq"}}},
                context,
            )
        message = str(excinfo.value)
        assert "cal" in message
        assert "rhino-cal-jax" in message


class TestTerminationNeedsFreq:
    """kind: termination defaults n: to the shape symbol 'n_freq', which
    resolves to 0 with no frequency grid declared -- termination_gamma(
    n_freq=0) used to return a silently empty array with no complaint,
    mirroring the observation.freq.grid requirement kind: cable already
    had."""

    def test_termination_without_a_frequency_grid_is_refused(self):
        pytest.importorskip("rhino_cal_jax")
        freqless = ResolutionContext(freq=None, time=jnp.arange(8.0), dtype="float32")
        with pytest.raises(ConfigError, match="freq"):
            build_resources(
                {"s_params": {"open": {"kind": "termination", "termination": "open",
                                       "z0": {"value": 50.0, "unit": "ohm"}}}},
                freqless,
            )


class TestUnitValidation:
    """z0/impedance must be an impedance (or unitless); cable length must be
    a length (or unitless) -- mirrors kinds/projectors.py's own _angle: a
    value node with no unit token is trusted to already be canonical, one
    WITH a unit token must carry the right one."""

    def test_z0_must_be_an_impedance(self, context):
        pytest.importorskip("rhino_cal_jax")
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"s_params": {"open": {"kind": "termination", "termination": "open",
                                       "z0": {"value": 50.0, "unit": "K"}, "n": "n_freq"}}},
                context,
            )
        message = str(excinfo.value)
        assert "z0" in message
        assert "impedance" in message

    def test_impedance_must_be_an_impedance(self, context):
        pytest.importorskip("rhino_cal_jax")
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"s_params": {"r": {"kind": "termination", "termination": "resistive",
                                    "impedance": {"value": 75.0, "unit": "K"},
                                    "z0": {"value": 50.0, "unit": "ohm"}, "n": "n_freq"}}},
                context,
            )
        assert "impedance" in str(excinfo.value)

    def test_cable_length_must_be_a_length(self, context):
        pytest.importorskip("rhino_cal_jax")
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"s_params": {
                    "open": {"kind": "termination", "termination": "open",
                             "z0": {"value": 50.0, "unit": "ohm"}, "n": "n_freq"},
                    "cable": {"kind": "cable", "behind": {"ref": "resources.s_params.open"},
                              "length": {"value": 1.0, "unit": "K"}},
                }},
                context,
            )
        message = str(excinfo.value)
        assert "length" in message
        assert "a length" in message


class TestCable:
    def test_cable_builds_behind_a_termination_with_a_bare_length(self, context):
        """length: 1.5 is a bare number -- no unit token -- so it must pass
        _dimensioned's unitless branch; velocity_factor and loss are both
        omitted, so both must fall back to their 1.0 default rather than
        requiring a value node on every cable."""
        pytest.importorskip("rhino_cal_jax")
        built = build_resources(
            {"s_params": {
                "open": {"kind": "termination", "termination": "open",
                         "z0": {"value": 50.0, "unit": "ohm"}, "n": "n_freq"},
                "cable": {"kind": "cable", "behind": {"ref": "resources.s_params.open"},
                          "length": 1.5},
            }},
            context,
        )
        gamma = built.resources["resources.s_params.cable"]
        assert gamma.shape == (4,)
        assert jnp.iscomplexobj(gamma)


class TestZ0LivesWhereItIsRead:
    def test_a_touchstone_entry_does_not_take_z0(self, context):
        """Touchstone.z0 is parsed and never read by any other module;
        termination_gamma(z0=) IS read. So the key exists in exactly one kind."""
        with pytest.raises(ConfigError) as excinfo:
            build_resources(
                {"s_params": {"horn": {"kind": "touchstone",
                                       "file": {"path": "horn.s1p", "format": "touchstone"},
                                       "component": "s11", "onto": "freq",
                                       "z0": {"value": 50.0, "unit": "ohm"}}}},
                context,
            )
        assert "z0" in str(excinfo.value)


class TestUnknownKeysAreRefused:
    """Task 8's own sweep addition: termination and cable had no
    check_unknown_keys coverage in the plan's own code, unlike
    kind: touchstone's richer, hand-written unknown-key refusal."""

    def test_a_stray_key_on_termination_is_refused(self, context):
        pytest.importorskip("rhino_cal_jax")
        with pytest.raises(ConfigError, match="component"):
            build_resources(
                {"s_params": {"open": {"kind": "termination", "termination": "open",
                                       "z0": {"value": 50.0, "unit": "ohm"}, "n": "n_freq",
                                       "component": "s11"}}},
                context,
            )

    def test_a_stray_key_on_cable_is_refused(self, context):
        pytest.importorskip("rhino_cal_jax")
        with pytest.raises(ConfigError, match="z0"):
            build_resources(
                {"s_params": {
                    "open": {"kind": "termination", "termination": "open",
                             "z0": {"value": 50.0, "unit": "ohm"}, "n": "n_freq"},
                    "cable": {"kind": "cable", "behind": {"ref": "resources.s_params.open"},
                              "length": {"value": 1.0, "unit": "m"},
                              "z0": {"value": 50.0, "unit": "ohm"}},
                }},
                context,
            )
