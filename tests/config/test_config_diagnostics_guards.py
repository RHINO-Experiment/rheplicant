"""What both diagnostics refuse about ``names:``, ``at:`` and ``rtol:``.

A sibling of ``test_config_exits_diagnostics.py`` rather than more of it:
that module reached 796 of this repository's 800-line ceiling with Task 6's
``condition`` tests and Task 7's two classes, and these guards are the half
of Task 7 that is about the SHARED helpers rather than about either exit's
answer.  Both modules import their fixtures from
``tests/config/exit_helpers.py``, so there is one ``diagnostic_document()``
and one measured model -- Task 8 moved them there when ``kind: gradient``
became their third caller.

The whole class is parametrized over BOTH kinds on purpose.  ``_names`` and
``_at_values`` are one implementation serving two package entry points that
disagree about every malformed shape, and on the shapes below it is
``score_directions`` -- the one with no guard of its own -- that escapes the
layer's single-ConfigError contract.  A guard tested on one route only is
how a hole gets closed on one side and left open on its twin.
"""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.runs import run_document
from tests.config.exit_helpers import (
    IDENTIFIED_PAIR,
    diagnostic_document,
    diagnostic_report,
)

BOTH = ("identifiability", "score_directions")


@pytest.mark.parametrize("kind", BOTH)
class TestTheNamesGuardStandsForBothKinds:
    @pytest.mark.parametrize("names", ([], "g", [1], [["g"]], {"g": 1}, 3))
    def test_a_malformed_names_is_one_config_error(self, kind, names):
        """Measured, with the guard's ``or not names`` / ``or not all(...)``
        legs removed: ``names: []`` reaches a concatenate of nothing inside
        ``score_directions`` and raises ``ValueError: at least one array or
        dtype is required``; ``names: [[g]]`` raises ``TypeError: unhashable
        type: 'list'``; ``names: [1]`` degrades to a ParameterSpaceError
        naming no run.  ``identifiability`` guards itself against the empty
        list and would still refuse -- which is exactly why every shape is
        asked of both kinds rather than of the one that happens to be safe.
        """
        with pytest.raises(ConfigError, match="non-empty list of latent"):
            run_document(diagnostic_document({"kind": kind, "names": names}))

    def test_a_repeated_latent_is_refused_before_either_package_sees_it(
            self, kind):
        """``identifiability`` refuses ``['g', 'g']`` itself, because two
        copies of one latent are exactly degenerate; ``score_directions``
        returns ONE key for the two-name ask, silently, so the measured
        product for ``[g, d, g]`` has two keys for a three-name list.  A
        caller zipping that against their own list is off by one -- the
        permutation bug ``reduced_basis.py:171-180`` is named after, reached
        from the far side.  Only this layer can refuse it for both kinds.
        """
        with pytest.raises(ConfigError,
                           match=r"lists \['g'\] more than once") as caught:
            run_document(diagnostic_document(
                {"kind": kind, "names": ["g", "d", "g"]}))
        assert "off by one" in str(caught.value)

    def test_a_declared_null_at_is_refused_rather_than_ignored(self, kind):
        """``at:`` with an empty YAML value is a key the user wrote.

        ``_names`` refuses ``names:`` written that way ("got None"), and
        before this ``at:`` read ``run.options.get("at")`` and folded a
        declared null into the absent case -- a document that says one thing
        and does another, with nothing to notice.  An ABSENT ``at:`` is
        still ``{}``, which is plan section 3.1 and the helper test in the
        sibling module.
        """
        with pytest.raises(ConfigError, match="at: is a mapping") as caught:
            run_document(diagnostic_document({"kind": kind, "at": None}))
        assert "got None" in str(caught.value)


class TestTheRankToleranceHasBothBounds:
    def test_an_rtol_of_one_is_refused_at_the_ceiling(self):
        """The mirror of the floor, and it is the arithmetic that says so.

        The cutoff is ``rtol * s_max``, so at 1 the largest singular value
        sits AT the threshold and nothing is above it: measured rank 0 /
        nullity 2 at ``rtol: 1.0`` on the identified pair AND on the
        degenerate one -- one answer for two different models, which is the
        definition of a vacuous verdict.  Matched on the ceiling clause,
        which neither ``rtol: is a number`` nor ``rtol: must be >= 0``
        carries.
        """
        with pytest.raises(ConfigError, match=r"rtol: must be < 1") as caught:
            run_document(diagnostic_document(
                {"kind": "identifiability", "rtol": 1.0}))
        assert "runs['identifiability']: " in str(caught.value)

    def test_an_rtol_above_one_is_refused_by_the_same_clause(self):
        with pytest.raises(ConfigError, match=r"rtol: must be < 1"):
            run_document(diagnostic_document(
                {"kind": "identifiability", "rtol": 2.0}))

    def test_the_ceiling_is_exclusive_and_0_999_still_discriminates(self):
        """The other side of the boundary, which is what keeps the ceiling
        from being a round number chosen for looks: at 0.999 the identified
        pair still reads rank 1 / nullity 1 rather than rank 0, so the
        refusal begins exactly where the arithmetic goes vacuous.  A ceiling
        clamped low -- or a ``> 1.0`` written where ``>= 1.0`` belongs --
        fails one of these two tests."""
        near = diagnostic_report({"kind": "identifiability", "rtol": 0.999},
                      IDENTIFIED_PAIR)
        assert (near.rank, near.nullity) == (1, 1)
        assert near.rtol == pytest.approx(0.999)
