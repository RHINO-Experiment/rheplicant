"""The "N of M are still placeholders" claim is counted, not remembered.

Three places state how much of ``rheplicant.radio`` is real physics: the
package docstring, the README's Status section, and every operator's own
docstring. The first two are prose and rot silently; the third is the ground
truth, because it is written by whoever last touched the body.

So this module derives the census from the operator docstrings and pins the
number the prose quotes. The failure it exists to prevent is not a wrong count
-- nobody is harmed by 17 vs 18 -- it is the shape of the claim that was there
before: a blanket "every operator is a trivial-but-runnable placeholder",
written when it was true, left standing while six operators and an ingestion
layer became real. A reader who believed it would not have trusted the sky
engines, and a reader who caught it once would stop trusting the rest of the
sentence too.
"""

import inspect
import re
from pathlib import Path

import pytest

import rheplicant.radio as radio
from rheplicant.core.capability import Maturity
from rheplicant.core.operator import AbstractOperator
from rheplicant.radio import at_level, capability_classes

#: Wording by which a class declares its own body a stand-in. This reads
#: PROSE, and it is now the only thing here that does: the membership lists
#: below are derived from the ``maturity`` ClassVar each class declares, so
#: this regex no longer decides anything. It checks that the docstring a
#: reader sees AGREES with the level the package publishes, in both
#: directions, which is the direction the plan asks for -- docs checked
#: against the registry, never the reverse.
_PLACEHOLDER_WORDING = re.compile(
    r"placeholder|toy|trivial|stand-?in|not (?:the )?real|deliberately simpl|simplest",
    re.IGNORECASE,
)


def _concrete_operators() -> dict[str, type]:
    """Every concrete OPERATOR class the package exports, by name.

    A subset of :func:`rheplicant.radio.capability_classes`, which also holds
    the sky models and projectors. The counts this module pins say "operator
    classes" in the prose they check, so the counted population stays
    operators; the prose CHECKS below cover every capability.
    """
    return {
        name: cls for name, cls in capability_classes().items() if issubclass(cls, AbstractOperator)
    }


#: Classes whose body is a stand-in, and the rest. Both are now DERIVED from
#: the levels the classes declare, not written out here.
#:
#: They used to be two hand-maintained frozensets, and keeping them was real
#: work with a real argument attached to individual names -- which is where
#: the reasoning went when they were replaced. ``AtmosphericEmissionOperator``
#: and ``GroundPickupOperator`` are the case worth remembering: both were
#: promoted to REAL on the grounds that their contract and placement are real,
#: and both were moved back, because that criterion does not SEPARATE the two
#: lists (``ReceiverOperator`` and ``GainOperator`` have load-bearing contracts
#: and stand-in bodies too). Whatever divides them has to be a claim about the
#: BODY. Each class now states that claim in its own docstring -- "The BODY is
#: a placeholder" -- and declares the matching level beside ``requires`` and
#: ``provides``.
#:
#: That history is also why the derivation is worth having. When the capability
#: registry was first written its level for those two was taken from the
#: docstring's SUMMARY line, which had been cleaned when they were promoted,
#: so the registry said MAINTAINED while this census said placeholder and
#: nothing compared them. Two criteria, two answers, no failure.
PLACEHOLDER = frozenset(
    name for name, cls in _concrete_operators().items() if cls.maturity is Maturity.PLACEHOLDER
)
REAL = frozenset(_concrete_operators()) - PLACEHOLDER

#: Every capability, not just the operators, for the prose checks.
ALL_PLACEHOLDER = at_level(Maturity.PLACEHOLDER)
ALL_OTHER = frozenset(capability_classes()) - ALL_PLACEHOLDER


class TestCensus:
    def test_the_shared_walk_sees_what_a_direct_walk_sees(self):
        """The derivation in ``src/`` agrees with the obvious one.

        The lists are derived now, so "no operator is unclassified" is true by
        construction and asserting it would prove nothing. What is worth
        asserting is the part construction does NOT give: that
        ``capability_classes()`` -- the one walk every view shares -- really
        covers the exported operators, rather than filtering some of them out
        on its way. If it ever narrowed, every view would narrow with it,
        silently and together, which is the failure mode a shared derivation
        buys in exchange for the drift it removes.
        """
        direct = {
            name: obj
            for name in radio.__all__
            if inspect.isclass(obj := getattr(radio, name, None))
            and issubclass(obj, AbstractOperator)
            and not inspect.isabstract(obj)
        }
        assert set(_concrete_operators()) == set(direct), {
            "shared walk missed": sorted(set(direct) - set(_concrete_operators())),
            "shared walk invented": sorted(set(_concrete_operators()) - set(direct)),
        }
        assert not (PLACEHOLDER & REAL)

    @pytest.mark.parametrize("name", sorted(ALL_PLACEHOLDER))
    def test_placeholder_operators_say_so(self, name):
        doc = inspect.getdoc(capability_classes()[name]) or ""
        # Necessary, not sufficient. A docstring can contain the word while
        # DENYING that it applies -- "the real flagger behind the placeholder
        # above" matched here for a year while describing a permanent
        # integration. This direction cannot catch that; only reading can, and
        # `test_real_operators_do_not_hedge` is the half that has teeth.
        assert _PLACEHOLDER_WORDING.search(doc), (
            f"{name} is listed as a placeholder but its docstring no longer "
            f"says so. If the physics is real now, move it to REAL and update "
            f"the counts in rheplicant/radio/__init__.py and README.md."
        )

    @pytest.mark.parametrize("name", sorted(ALL_OTHER))
    def test_real_operators_do_not_hedge(self, name):
        """The direction that catches a stale caveat, not a stale count.

        An operator whose body became real while its docstring kept the
        placeholder sentence reads as untrustworthy to exactly the reader who
        checks -- which is the failure this whole module is about, one level
        down.
        """
        doc = inspect.getdoc(capability_classes()[name]) or ""
        assert not _PLACEHOLDER_WORDING.search(doc), (
            f"{name} is listed as real physics but its docstring still hedges. "
            f"Either the caveat is stale and should go, or the operator belongs "
            f"in PLACEHOLDER."
        )


class TestProseAgrees:
    """The number in the prose is the number in the code.

    Both files are checked for the same literal pair, so a count corrected in
    one place and not the other fails rather than leaving the two disagreeing
    -- which is how the claim drifted the first time.
    """

    @pytest.mark.parametrize(
        "relative_path",
        ["src/rheplicant/radio/__init__.py", "README.md"],
    )
    def test_quoted_counts_match_the_census(self, relative_path):
        root = Path(__file__).resolve().parents[2]
        text = (root / relative_path).read_text(encoding="utf-8")
        expected = rf"{len(PLACEHOLDER)}\s+of\s+the\s+{len(PLACEHOLDER | REAL)}\b"
        # Allow the line break the wrapped prose puts inside the phrase.
        assert re.search(expected.replace(r"\s+", r"[\s\n]+"), text), (
            f"{relative_path} does not state "
            f"'{len(PLACEHOLDER)} of the {len(PLACEHOLDER | REAL)}' concrete "
            f"operator classes; the census says it should."
        )

    @pytest.mark.parametrize(
        "relative_path",
        ["src/rheplicant/radio/__init__.py", "README.md"],
    )
    def test_the_complement_is_stated_correctly_where_it_is_stated(self, relative_path):
        """The OTHER count -- the half that drifted while its partner held.

        ``15 of the 29`` was pinned above and stayed right; the sentence after
        it said "the other twelve" for as long as the census said fourteen,
        because nothing derived the complement. Pinning one number and
        remembering its complement is not two guards, it is one guard and one
        opportunity.

        Stated where it is stated: a file that does not use the phrase is not
        required to, and this asserts nothing about it. Adding the phrase to a
        file brings it under the guard automatically.
        """
        root = Path(__file__).resolve().parents[2]
        text = (root / relative_path).read_text(encoding="utf-8")
        words = {
            10: "ten",
            11: "eleven",
            12: "twelve",
            13: "thirteen",
            14: "fourteen",
            15: "fifteen",
            16: "sixteen",
            17: "seventeen",
            18: "eighteen",
        }
        assert len(REAL) in words, f"{len(REAL)} real operators -- extend the number words above"
        stated = re.search(r"[Tt]he other[\s\n]+([a-z]+)", text)
        if stated is None:
            pytest.skip(f"{relative_path} does not state the complement")
        assert stated.group(1) == words[len(REAL)], (
            f"{relative_path} says 'the other {stated.group(1)}'; the census "
            f"has {len(REAL)} operators outside PLACEHOLDER "
            f"({words[len(REAL)]})."
        )
