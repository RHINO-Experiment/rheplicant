"""The shape-symbol table: n_freq, n_freq - 1, 2 * n_time, and nothing else."""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.symbols import SHAPE_SYMBOLS, ShapeScope, resolve_extent


@pytest.fixture
def scope():
    return ShapeScope(n_time=64, n_freq=8, n_source=4)


class TestTheClosedTable:
    def test_every_symbol_in_the_table_resolves(self, scope):
        """n_pix and n_alm are in the table but have no global source, so they
        are covered by their own test below rather than skipped silently."""
        assert resolve_extent("n_time", scope) == 64
        assert resolve_extent("n_freq", scope) == 8
        assert resolve_extent("n_source", scope) == 4

    def test_n_load_is_derived_from_n_source(self, scope):
        """schema 2.1.3: n_load = n_source - 1. Deriving it is what makes a
        cal_loads list that disagrees with the switch order impossible."""
        assert resolve_extent("n_load", scope) == 3

    def test_a_plain_integer_passes_through(self, scope):
        assert resolve_extent(12, scope) == 12

    def test_a_bool_is_refused_rather_than_read_as_its_integer(self, scope):
        """MEASURED: delete the bool branch, or run it after the int branch,
        and every other test in this file still passes. True IS an int in
        Python, so a True in a shape position would resolve to an extent of 1
        and give a length-1 axis -- finite, correctly-shaped, and wrong."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_extent(True, scope)
        assert "True" in str(excinfo.value)


class TestTheArithmeticIsAnOffsetNotAnExpression:
    @pytest.mark.parametrize(
        "text,expected",
        [("n_freq - 1", 7), ("n_freq + 1", 9), ("2 * n_time", 128), ("3*n_source", 12)],
    )
    def test_the_three_legal_forms(self, text, expected, scope):
        assert resolve_extent(text, scope) == expected

    def test_the_multiple_binds_to_the_symbol_not_to_the_offset(self, scope):
        """MEASURED: move the multiply after the offset -- mult * (extent +
        offset) instead of mult * extent + offset -- and every other test in this
        file still passes, because no other fixture uses both at once.
        '2 * n_freq - 1' is 15 under the reading the module states and 14 under
        the other, and an axis one short is a correctly-shaped wrong array."""
        assert resolve_extent("2 * n_freq - 1", scope) == 15

    @pytest.mark.parametrize(
        "text", ["n_freq * n_time", "n_freq / 2", "n_freq ** 2", "n_freq - n_source"]
    )
    def test_anything_else_is_refused(self, text, scope):
        with pytest.raises(ConfigError, match="not an expression language"):
            resolve_extent(text, scope)


class TestAnUnknownSymbol:
    def test_it_is_refused_and_the_table_is_listed(self, scope):
        with pytest.raises(ConfigError) as excinfo:
            resolve_extent("n_channels", scope)
        message = str(excinfo.value)
        assert "n_channels" in message
        for symbol in SHAPE_SYMBOLS:
            assert symbol in message, symbol


class TestNPixAndNAlmRefuseRatherThanGuess:
    def test_n_pix_without_an_nside_in_scope_is_refused(self, scope):
        """schema 2.1.3 says '12 * nside**2 of the named beam/sky' and never
        says WHICH. In a document with two beams at different nside there is
        no answer, and a guessed nside yields a correctly-shaped HEALPix map
        at the wrong resolution -- finite, plausible, wrong."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_extent("n_pix", scope)
        message = str(excinfo.value)
        assert "n_pix" in message
        assert "nside" in message  # what would have had to be guessed
        assert "12 * nside**2" in message  # the literal remedy

    def test_n_pix_resolves_where_the_entry_declares_its_own_nside(self, scope):
        assert resolve_extent("n_pix", scope.within(nside=8)) == 768

    def test_n_alm_resolves_where_the_entry_declares_its_own_lmax(self, scope):
        assert resolve_extent("n_alm", scope.within(lmax=23)) == 300

    def test_n_alm_without_an_lmax_is_refused_in_its_own_words(self, scope):
        """MEASURED: the n_alm half of the refusal is otherwise uncovered. Hard-
        wire `key` to "nside", or narrow the refusal itself to n_pix so n_alm
        falls through to a guessed lmax, and every other test in this file still
        passes. Asserting only a substring the two halves share would pass
        whichever one ran; what a reader needs is which declaration is missing,
        because an lmax written as an nside is not a smaller version of the
        right fix, it is a different one."""
        with pytest.raises(ConfigError) as excinfo:
            resolve_extent("n_alm", scope)
        message = str(excinfo.value)
        assert "n_alm" in message
        assert "lmax" in message  # what would have had to be guessed
        assert "(lmax+1)(lmax+2)/2" in message  # the literal remedy
        assert "nside" not in message  # not the other half's remedy

    def test_within_leaves_the_scope_it_was_called_on_alone(self, scope):
        """MEASURED: make ShapeScope mutable and have within() setattr and
        return self, and every other test in this file still passes -- each
        reads the extent back off the object within() handed it. The scope is
        threaded down a document one entry at a time, so an nside that outlives
        its entry is an nside silently applied to the next one, and a HEALPix
        map at the wrong resolution comes back finite and correctly shaped."""
        inner = scope.within(nside=8)
        assert resolve_extent("n_pix", inner) == 768
        assert inner is not scope
        assert scope.nside is None
        with pytest.raises(ConfigError):
            resolve_extent("n_pix", scope)

    def test_the_refusal_names_the_entries_that_could_have_been_meant(self):
        scope = ShapeScope(
            n_time=64,
            n_freq=8,
            n_source=4,
            candidates=("resources.beams.horn", "resources.beams.gauss"),
        )
        with pytest.raises(ConfigError) as excinfo:
            resolve_extent("n_pix", scope)
        message = str(excinfo.value)
        assert "resources.beams.horn" in message
        assert "resources.beams.gauss" in message


class TestAHardCodedExtentIsFlagged:
    def test_an_integer_equal_to_a_known_extent_is_reported(self, scope):
        """Check A41. Not a refusal -- a literal 8 may legitimately be 8 --
        but the copies must be tied together somewhere, and the report is what
        ties them."""
        from rheplicant.config.symbols import literal_shadowing_a_symbol

        assert literal_shadowing_a_symbol(8, scope) == "n_freq"
        assert literal_shadowing_a_symbol(64, scope) == "n_time"
        assert literal_shadowing_a_symbol(7, scope) is None

    def test_a_literal_1_is_not_reported_against_an_unswitched_n_source(self):
        """MEASURED: drop the `value > 1` guard and every other test in this file
        still passes, because the fixture switches. n_source defaults to 1
        for a document that does not, and 1 is the commonest literal in any
        shape -- reporting every one of them as shadowing n_source is how a
        check that fires on everything gets switched off."""
        from rheplicant.config.symbols import literal_shadowing_a_symbol

        quiet = ShapeScope(n_time=64, n_freq=8, n_source=1)
        assert literal_shadowing_a_symbol(1, quiet) is None
        assert literal_shadowing_a_symbol(8, quiet) == "n_freq"

    def test_a_float_equal_to_an_extent_is_not_reported(self, scope):
        """MEASURED: drop `not isinstance(value, int)` and 8.0 is reported as
        shadowing n_freq while every other test in this file still passes. A41
        is about a hand-copied integer extent; a float in a shape position is a
        different fault, refused by resolve_extent, and naming it here would put
        the wrong remedy in the report.

        A string is deliberately NOT asserted alongside it: the isinstance
        guard returns before the comparison either way, so both the guarded and
        the unguarded implementation answer None and the assertion would sit
        where two candidate implementations coincide."""
        from rheplicant.config.symbols import literal_shadowing_a_symbol

        assert literal_shadowing_a_symbol(8.0, scope) is None
