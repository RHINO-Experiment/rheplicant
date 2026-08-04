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
from rheplicant.core.operator import AbstractOperator

#: Wording by which an operator declares its own body a stand-in. This is a
#: proxy -- it reads prose -- so the test pins the resulting membership lists
#: rather than the count alone: a docstring that picks up "trivial" in some
#: unrelated sentence moves a name between lists and must be looked at, not
#: silently absorbed by a count that happens to stay put.
_PLACEHOLDER_WORDING = re.compile(
    r"placeholder|toy|trivial|stand-?in|not (?:the )?real|deliberately simpl|simplest",
    re.IGNORECASE,
)

#: Operators whose own docstring declares the body a stand-in. Three of these
#: are load-bearing regardless (``ReceiverOperator``, ``GainOperator``,
#: ``CalLoadOperator``): their *shape* is the identifiability convention, the
#: plan's engine derivation and ``must_precede``. "Placeholder" is a claim
#: about the body, never about the contract.
PLACEHOLDER = frozenset({
    "ADCOperator",
    "ApplyCalibrationOperator",
    "BackendOperator",
    "BeamOperator",
    "CalLoadOperator",
    "EMIOperator",
    "FlaggingOperator",
    "ForegroundOperator",
    "GainOperator",
    "GlobalSignalOperator",
    "IonosphereOperator",
    "MomentRFIFlaggingOperator",
    "NoiseOperator",
    "PointSourceOperator",
    "RFIOperator",
    "ReceiverOperator",
    "SkyOperator",
})

#: Operators that carry no such wording. Adding a name here is a claim that
#: the physics is real, and the burden is a docstring that says what it does.
REAL = frozenset({
    "AntennaLossOperator",
    "AtmosphericEmissionOperator",
    "BasisTemperatureOperator",
    "BeamSpillOperator",
    "CWCalibrationOperator",
    "FourierBandFilter",
    "GroundPickupOperator",
    "NeuralOperator",
    "NoiseWaveOperator",
    "SiderealFilter",
    "SkySourceOperator",
    "SkySpaceFilter",
})


def _concrete_operators() -> dict[str, type]:
    """Every concrete operator class the package exports, by name."""
    found = {}
    for name in radio.__all__:
        obj = getattr(radio, name)
        if not (inspect.isclass(obj) and issubclass(obj, AbstractOperator)):
            continue
        if inspect.isabstract(obj) or name.startswith("Abstract"):
            continue
        found[name] = obj
    return found


class TestCensus:
    def test_the_two_lists_are_the_exported_operators(self):
        """No operator is unclassified, and none is classified twice.

        This is what makes the count below mean something: a new operator
        lands in neither list and fails here, so the census cannot quietly
        stop covering the package it claims to describe.
        """
        exported = set(_concrete_operators())
        assert not (PLACEHOLDER & REAL), PLACEHOLDER & REAL
        assert PLACEHOLDER | REAL == exported, {
            "unclassified": sorted(exported - (PLACEHOLDER | REAL)),
            "listed but not exported": sorted((PLACEHOLDER | REAL) - exported),
        }

    @pytest.mark.parametrize("name", sorted(PLACEHOLDER))
    def test_placeholder_operators_say_so(self, name):
        doc = inspect.getdoc(_concrete_operators()[name]) or ""
        assert _PLACEHOLDER_WORDING.search(doc), (
            f"{name} is listed as a placeholder but its docstring no longer "
            f"says so. If the physics is real now, move it to REAL and update "
            f"the counts in rheplicant/radio/__init__.py and README.md."
        )

    @pytest.mark.parametrize("name", sorted(REAL))
    def test_real_operators_do_not_hedge(self, name):
        """The direction that catches a stale caveat, not a stale count.

        An operator whose body became real while its docstring kept the
        placeholder sentence reads as untrustworthy to exactly the reader who
        checks -- which is the failure this whole module is about, one level
        down.
        """
        doc = inspect.getdoc(_concrete_operators()[name]) or ""
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
