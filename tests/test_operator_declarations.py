"""Package-wide audit: the one enforced declaration is true of every operator.

`refuse_stochastic_stages` reads `'key' in requires` and refuses the model. That
is only a guard if the declaration is honest, and honesty is not something the
guard itself can check — there is no numerical symptom of a frozen draw, which
is the premise of the whole thing. So it is checked here instead, mechanically,
the way `tests/core/test_layering.py` checks the layering rule: an operator
whose source consumes the PRNG must say so, and one that says so must consume
it.

Deliberately package-wide rather than under `tests/core`: the contract is
declared in core and honoured in radio, so neither directory owns it.
"""

import inspect

import pytest

import rheplicant  # noqa: F401  (populates __subclasses__)
import rheplicant.inference  # noqa: F401
import rheplicant.radio  # noqa: F401
from rheplicant.core.combinators import SelectOperator, SumOperator
from rheplicant.core.contract import RANDOMNESS
from rheplicant.core.operator import AbstractOperator

# These two call `next_key` to SPLIT a subkey per branch so that stochastic
# branches draw independently. They consume no randomness of their own and
# correctly declare none — the exemption is the distinction, not a waiver.
SPLITTERS = (SumOperator, SelectOperator)

#: The shipped operators that draw. Pinned so that adding a third is a
#: conversation rather than a silent widening of what inference refuses.
KNOWN_STOCHASTIC = {"NoiseOperator", "RFIOperator", "RadiometerNoiseOperator"}


def shipped_operators() -> list[type]:
    """Every concrete AbstractOperator subclass defined inside the package."""
    found: dict[str, type] = {}
    stack = [AbstractOperator]
    while stack:
        for sub in stack.pop().__subclasses__():
            key = f"{sub.__module__}.{sub.__qualname__}"
            if key in found:
                continue
            found[key] = sub
            stack.append(sub)
    return [
        klass
        for klass in found.values()
        if klass.__module__.startswith("rheplicant.")
        and not inspect.isabstract(klass)
    ]


def test_the_audit_actually_sees_the_package():
    """A sweep that found nothing would pass every assertion below."""
    names = {klass.__name__ for klass in shipped_operators()}
    assert len(names) > 25, names
    assert {"NoiseOperator", "GainOperator", "SkyOperator"} <= names


@pytest.mark.parametrize(
    "klass", shipped_operators(), ids=lambda k: f"{k.__module__.split('.')[-1]}.{k.__name__}"
)
def test_drawing_and_declaring_agree(klass):
    if issubclass(klass, SPLITTERS):
        assert RANDOMNESS not in klass.requires, (
            f"{klass.__name__} splits keys for its branches but does not draw; "
            "declaring 'key' would have inference refuse every composite model."
        )
        return
    try:
        source = inspect.getsource(klass)
    except OSError:  # pragma: no cover - only for classes with no source file
        pytest.skip("no source available")
    draws = "next_key(" in source
    declares = RANDOMNESS in klass.requires
    assert draws == declares, (
        f"{klass.__module__}.{klass.__name__}: consumes the PRNG = {draws}, but "
        f"declares {RANDOMNESS!r} in requires = {declares}. The inference layer "
        "refuses a model on the strength of that declaration, so a drawing "
        "operator that stays silent is invisible to every exit and biases the "
        "fit; a declaration with no draw refuses a model that is fine."
    )


def test_the_stochastic_census_is_the_one_that_is_pinned():
    drawing = {k.__name__ for k in shipped_operators() if RANDOMNESS in k.requires}
    assert drawing == KNOWN_STOCHASTIC, (
        f"the set of operators inference refuses changed: {drawing} vs "
        f"{KNOWN_STOCHASTIC}. If you added a stochastic operator this is the "
        "expected failure — add it above, and check that the exits' refusal "
        "message still reads correctly for it."
    )
