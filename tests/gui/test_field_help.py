"""Help text for a field comes from the operator that owns it.

Every one of the 66 model fields already carries an ``Attributes:`` line in
its class docstring -- measured, not hoped for -- so the inspector reads the
documentation the package already maintains rather than growing a second
copy that nothing renders and nobody proofreads.
"""

from __future__ import annotations

import inspect
import textwrap

from rheplicant.config.delivery import field_specs
from rheplicant.config.sections.model import operator_table
from rheplicant.gui.field_help import field_help


class Documented:
    """One line.

    A paragraph that mentions attributes: casually, and must not be read as
    the section header below.

    Attributes:
        alpha: the first one.
        beta: the second one, whose sentence runs on
            to a continuation line.
        gamma: the third.
    """


class NoAttributes:
    """One line, and nothing else."""


class SectionAfter:
    """One line.

    Attributes:
        alpha: the only one.

    Raises:
        ValueError: never.
    """


class TestTheAttributesParser:
    def test_it_reads_one_line_per_field(self):
        assert field_help(Documented)["alpha"] == "the first one."
        assert field_help(Documented)["gamma"] == "the third."

    def test_a_continuation_line_joins_the_sentence_it_belongs_to(self):
        assert field_help(Documented)["beta"] == (
            "the second one, whose sentence runs on to a continuation line."
        )

    def test_a_class_with_no_attributes_block_offers_nothing(self):
        assert field_help(NoAttributes) == {}

    def test_the_block_ends_where_the_next_section_starts(self):
        """``Raises:`` is a sibling section, not a fourth attribute."""
        assert list(field_help(SectionAfter)) == ["alpha"]

    def test_only_the_section_header_starts_the_block(self):
        """The prose above says the word; a mid-sentence mention must not be
        read as a header, or everything after it becomes an attribute."""
        assert "casually" not in " ".join(field_help(Documented))

    def test_an_object_without_a_docstring_is_not_an_error(self):
        assert field_help(type("Bare", (), {})) == {}

    def test_the_block_needs_the_conventional_summary_line_above_it(self):
        """A boundary of the parser, and of ``inspect.cleandoc`` beneath it.

        ``cleandoc`` leaves the FIRST line alone and dedents the rest by their
        COMMON indent. In a conventional docstring that common indent is zero
        -- ``Attributes:`` itself sits at column 0 -- so the four spaces in
        front of each field survive and mark them. In a docstring whose whole
        body is the block, the common indent IS four, so cleandoc removes
        exactly the indent that distinguishes a field from a section header
        and there is nothing left to read.

        Relaxing the parser to accept unindented fields would make the next
        section header (``Raises:``) look like a field, which is worse. Every
        operator here writes the conventional shape, and the census below is
        what keeps that true."""
        class Terse:
            pass

        Terse.__doc__ = textwrap.dedent(
            """\
            Attributes:
                only: one.
            """
        )
        assert inspect.getdoc(Terse).splitlines() == ["Attributes:", "only: one."]
        assert field_help(Terse) == {}


class TestEveryModelFieldIsDocumented:
    def test_all_sixty_six_of_them(self):
        """The census this feature rests on. A new field with no
        ``Attributes:`` line turns the inspector's help text into a blank, so
        this goes red instead."""
        undocumented = [
            f"{cls.__name__}.{name}"
            for classes in operator_table().values()
            for cls in classes
            for name in field_specs(cls)
            if not field_help(cls).get(name)
        ]

        assert undocumented == []

    def test_the_count_is_the_sixty_six_the_catalog_declares(self):
        total = sum(
            len(field_specs(cls)) for classes in operator_table().values() for cls in classes
        )

        assert total == 66
