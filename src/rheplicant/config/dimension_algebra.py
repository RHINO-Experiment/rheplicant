"""The signature algebra: what a physical dimension IS.

Exponents over the seven base atoms, normalised so that equality is exact, and
the four operations that combine them. Nothing here knows about documents,
operators or registries -- it is the arithmetic the rest of the layer agrees
in, which is why it is the module nothing else in the family imports back.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rheplicant.config.errors import ConfigError
from rheplicant.config.units import Unit, canonical_unit

PhysicalDimension = tuple[tuple[str, int], ...]

QuantitySignature = tuple[tuple[str, int], ...]


def _canonical_components(
    items: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    totals: Counter[str] = Counter()
    for name, exponent in items:
        if not isinstance(name, str) or not name:
            raise ConfigError(f"dimensions: an atom name is a non-empty string; got {name!r}")
        if not isinstance(exponent, int) or isinstance(exponent, bool):
            raise ConfigError(
                f"dimensions: an integer exponent is required; got {exponent!r} "
                f"({type(exponent).__name__})."
            )
        totals[name] += exponent
    return tuple(sorted((name, exponent) for name, exponent in totals.items() if exponent))


@dataclass(frozen=True, slots=True)
class DimensionSignature:
    physical: PhysicalDimension
    quantity: QuantitySignature = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "physical", _canonical_components(self.physical))
        object.__setattr__(self, "quantity", _canonical_components(self.quantity))


_ATOM_SIGNATURES = {
    "Hz": DimensionSignature((("frequency", 1),)),
    "s": DimensionSignature((("time", 1),)),
    "unix_s": DimensionSignature((("time_epoch", 1),)),
    "K": DimensionSignature((("temperature", 1),)),
    "deg": DimensionSignature((("angle", 1),)),
    "m": DimensionSignature((("length", 1),)),
    "ohm": DimensionSignature((("impedance", 1),)),
    "dimensionless": DimensionSignature(()),
    "count": DimensionSignature((), (("count", 1),)),
    "samples": DimensionSignature((), (("samples", 1),)),
    "bits": DimensionSignature((), (("bits", 1),)),
    "channels": DimensionSignature((), (("channels", 1),)),
    "cycles": DimensionSignature((), (("cycles", 1),)),
    "adc_count": DimensionSignature((("adc_count", 1),)),
}


def _normal(items: Mapping[str, int]) -> tuple[tuple[str, int], ...]:
    return _canonical_components(tuple(items.items()))


def _combine(
    left: tuple[tuple[str, int], ...],
    right: tuple[tuple[str, int], ...],
    factor: int,
) -> tuple[tuple[str, int], ...]:
    merged = Counter()
    merged.update(dict(left))
    for name, exponent in right:
        merged[name] += factor * exponent
    return _normal(merged)


def multiply(left: DimensionSignature, right: DimensionSignature) -> DimensionSignature:
    """Multiply two normalized signatures, cancelling zero exponents."""
    return DimensionSignature(
        _combine(left.physical, right.physical, 1),
        _combine(left.quantity, right.quantity, 1),
    )


def divide(left: DimensionSignature, right: DimensionSignature) -> DimensionSignature:
    """Divide two normalized signatures, cancelling zero exponents."""
    return DimensionSignature(
        _combine(left.physical, right.physical, -1),
        _combine(left.quantity, right.quantity, -1),
    )


def power(value: DimensionSignature, exponent: int) -> DimensionSignature:
    """Raise a signature to a non-fuzzy integer exponent."""
    if not isinstance(exponent, int) or isinstance(exponent, bool):
        raise ConfigError(
            f"dimensions: an integer exponent is required; got {exponent!r} "
            f"({type(exponent).__name__})."
        )
    return DimensionSignature(
        _normal({name: amount * exponent for name, amount in value.physical}),
        _normal({name: amount * exponent for name, amount in value.quantity}),
    )


def dimension_of(unit: Unit | str) -> DimensionSignature:
    """Compute a signature from the existing six-field ``Unit``."""
    parsed = canonical_unit(unit) if isinstance(unit, str) else unit
    result = DimensionSignature(())
    for atom in parsed.numerator:
        result = multiply(result, _ATOM_SIGNATURES[atom])
    for atom in parsed.denominator:
        result = divide(result, _ATOM_SIGNATURES[atom])
    return result


def signature(token: str) -> DimensionSignature:
    """Parse one accepted unit spelling into a normalized signature."""
    return dimension_of(canonical_unit(token))


def signature_label(value: DimensionSignature) -> str:
    """Compact human label used by A9 diagnostics."""
    if not value.physical and not value.quantity:
        return "dimensionless"
    if value.quantity and not value.physical and len(value.quantity) == 1:
        name, exponent = value.quantity[0]
        return name if exponent == 1 else f"{name}^{exponent}"
    if len(value.physical) == 1 and not value.quantity:
        name, exponent = value.physical[0]
        return name if exponent == 1 else f"{name}^{exponent}"
    return repr(value)


def describe_signature(value: DimensionSignature) -> str:
    """A signature as a refusal should say it: ``"a length (m)"``.

    The signature already carries the dimension's NAME -- ``length``,
    ``impedance`` -- so this invents nothing; before it existed, a refusal
    printed the dataclass and a user reading ``requires
    DimensionSignature(physical=(('length', 1),), quantity=())`` learned only
    that something was wrong.

    One dimension at exponent one is named with its article, which is the
    wording ``kinds/s_params.py`` was already using where it could raise its
    own refusal. Anything else -- a quotient, or the dimensionless and
    counting signatures that have no physical name at all -- is given as the
    token, because ``a adc_count per temperature`` reads worse than
    ``adc_count/K`` and the token is what the user has to type.
    """
    components = (*value.physical, *value.quantity)
    if len(components) != 1 or components[0][1] != 1:
        return signature_token(value)
    name = components[0][0]
    article = "an" if name[0] in "aeiou" else "a"
    return f"{article} {name} ({signature_token(value)})"


def signature_token(value: DimensionSignature) -> str:
    """The accepted canonical spelling for a catalog signature."""
    for token in (
        "Hz",
        "s",
        "unix_s",
        "K",
        "deg",
        "m",
        "ohm",
        "dimensionless",
        "count",
        "samples",
        "bits",
        "channels",
        "cycles",
        "adc_count",
        "adc_count/K",
        "Hz/s",
        "dimensionless/s",
        "cycles/samples",
    ):
        if signature(token) == value:
            return token
    return repr(value)
