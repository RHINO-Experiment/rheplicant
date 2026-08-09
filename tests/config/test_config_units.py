"""The unit alphabet, the quotient grammar, and the field-name cross-check."""

import math

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.units import (
    ACCEPTED_UNITS,
    canonical_unit,
    check_field_name_unit,
    convert_to_canonical,
)


class TestTheAtomicAlphabet:
    @pytest.mark.parametrize(
        "declared,value,expected,canonical",
        [
            ("Hz", 1.0, 1.0, "Hz"),
            ("kHz", 1.0, 1e3, "Hz"),
            ("MHz", 60.0, 6e7, "Hz"),
            ("GHz", 1.4, 1.4e9, "Hz"),
            ("s", 2.0, 2.0, "s"),
            ("ms", 500.0, 0.5, "s"),
            ("K", 290.0, 290.0, "K"),
            ("celsius", 16.85, 290.0, "K"),
            ("deg", 90.0, 90.0, "deg"),
            ("rad", math.pi, 180.0, "deg"),
            ("m", 2.0, 2.0, "m"),
            ("ohm", 50.0, 50.0, "ohm"),
            ("dimensionless", 0.97, 0.97, "dimensionless"),
        ],
    )
    def test_a_declared_unit_converts_to_its_canonical_form(
        self, declared, value, expected, canonical
    ):
        got, unit = convert_to_canonical(value, declared)
        assert got == pytest.approx(expected)
        assert unit.canonical == canonical

    def test_the_lookup_is_case_insensitive(self):
        """radio/rhino.py:130 lower-cases and radio/touchstone.py:137
        upper-cases; the two readers disagree, so this layer accepts either
        rather than inventing a third convention that contradicts one of them."""
        assert canonical_unit("mhz").canonical == "Hz"
        assert canonical_unit("MHZ").canonical == "Hz"
        assert canonical_unit("MHz").canonical == "Hz"


class TestTheQuotientGrammar:
    def test_a_quotient_converts_by_dividing_the_denominator(self):
        got, unit = convert_to_canonical(2.0, "adc_count/K")
        assert got == pytest.approx(2.0)
        assert unit.canonical == "adc_count/K"

    def test_a_quotient_carries_each_side_s_own_factor(self):
        got, unit = convert_to_canonical(1.0, "MHz/s")
        assert got == pytest.approx(1e6)
        assert unit.canonical == "Hz/s"

    def test_a_product_is_accepted(self):
        assert canonical_unit("K*s").canonical == "K*s"

    def test_cycles_per_sample_is_expressible(self):
        assert canonical_unit("cycles/samples").canonical == "cycles/samples"

    def test_a_second_slash_is_refused(self):
        with pytest.raises(ConfigError, match="one '/'"):
            canonical_unit("K/s/s")

    def test_an_exponent_is_refused_and_the_grammar_is_named(self):
        with pytest.raises(ConfigError) as excinfo:
            canonical_unit("m^2")
        message = str(excinfo.value)
        assert "m^2" in message  # what was found
        assert "product" in message  # the grammar it is not
        assert "resources.arrays" in message  # the remedy


class TestAffineUnitsCannotCompose:
    def test_celsius_inside_a_quotient_is_refused(self):
        """A unit with an offset is affine, and an affine unit has no meaning
        as a factor in a product: 2 celsius/s is not 275.15 K/s. The offset
        exists for exactly one atom and it may only appear alone."""
        with pytest.raises(ConfigError) as excinfo:
            canonical_unit("celsius/s")
        message = str(excinfo.value)
        assert "celsius" in message
        assert "273.15" in message  # the offset that cannot be distributed
        assert "K/s" in message  # the remedy


class TestAnUnknownTokenIsRefused:
    def test_the_accepted_set_is_listed(self):
        with pytest.raises(ConfigError) as excinfo:
            canonical_unit("Jy")
        message = str(excinfo.value)
        assert "Jy" in message
        for token in ("Hz", "K", "deg", "dimensionless"):
            assert token in message
        assert "python:" in message  # the escape route

    def test_the_accepted_set_is_the_one_the_module_exports(self):
        """A message that lists a hand-written set goes stale the day the table
        changes. This asserts the message is DERIVED from the table."""
        with pytest.raises(ConfigError) as excinfo:
            canonical_unit("Jy")
        message = str(excinfo.value)
        for token in ACCEPTED_UNITS:
            assert token in message, token


class TestTheFieldNameCrossCheck:
    def test_a_deg_suffixed_field_declared_in_radians_is_accepted(self):
        """The suffix names the unit the field is STORED in, and the value is
        converted before it is stored, so declaring radians is legal."""
        check_field_name_unit("lat_deg", canonical_unit("rad"))

    def test_a_deg_suffixed_field_declared_in_kelvin_is_refused(self):
        with pytest.raises(ConfigError) as excinfo:
            check_field_name_unit("lat_deg", canonical_unit("K"))
        message = str(excinfo.value)
        assert "lat_deg" in message
        assert "_deg" in message
        assert "K" in message

    def test_an_unsuffixed_field_is_not_second_guessed(self):
        """adc.scale, gain.gain, apply_cal.gain, flagging.threshold and
        filters[].regularization carry no unit anywhere in the source. The
        config declares one and it is recorded as a declaration; there is
        nothing to cross-check it against."""
        check_field_name_unit("scale", canonical_unit("adc_count/K"))
