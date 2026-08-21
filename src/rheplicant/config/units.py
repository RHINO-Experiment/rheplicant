"""Units: a closed alphabet, a quotient grammar, and conversion on read.

Field names in this package already carry their unit -- ``lat_deg``,
``apod_deg``, ``lst_ref_deg`` -- so a config cannot put the unit in the key.
It goes in the value, and the redundancy becomes a free consistency check
instead of a contradiction.

The alphabet is deliberately small. Every token is either a canonical unit the
schema names, or a spelling one of the package's two existing readers already
accepts (``radio/rhino.py:70`` ``{"hz", "mhz"}``; ``radio/touchstone.py:40``
``{"HZ", "KHZ", "MHZ", "GHZ"}``). A wider table is a place to be quietly
wrong: an unconvertible unit is better refused by name than passed through,
because a factor of 1e6 on a frequency grid produces a finite,
correctly-shaped, wrong answer and nothing downstream can tell.

Compound units exist for one measured reason. Five fields carry no unit
anywhere in the source -- ``adc.scale``, ``gain.gain``, ``apply_cal.gain``,
``flagging.threshold``, ``filters[].regularization`` -- and decision D-C1
declares the post-``gain`` trunk to be ``adc_count``. That makes
``adc_count/K x K = adc_count`` an identity a validator can check, which is
what promotes the unit rule from a spelling convention to a real one.
"""

import math
from typing import NamedTuple

from rheplicant.config.errors import ConfigError


class _Atom(NamedTuple):
    canonical: str
    dimension: str
    factor: float
    offset: float = 0.0


#: Keyed by the lower-cased spelling; the value carries the canonical one.
_ATOMS: dict[str, _Atom] = {
    "hz": _Atom("Hz", "frequency", 1.0),
    "khz": _Atom("Hz", "frequency", 1e3),
    "mhz": _Atom("Hz", "frequency", 1e6),
    "ghz": _Atom("Hz", "frequency", 1e9),
    "s": _Atom("s", "time", 1.0),
    "ms": _Atom("s", "time", 1e-3),
    "unix_s": _Atom("unix_s", "time_epoch", 1.0),
    "k": _Atom("K", "temperature", 1.0),
    "celsius": _Atom("K", "temperature", 1.0, 273.15),
    "deg": _Atom("deg", "angle", 1.0),
    "rad": _Atom("deg", "angle", 180.0 / math.pi),
    "m": _Atom("m", "length", 1.0),
    "ohm": _Atom("ohm", "impedance", 1.0),
    "dimensionless": _Atom("dimensionless", "dimensionless", 1.0),
    "count": _Atom("count", "count", 1.0),
    "samples": _Atom("samples", "samples", 1.0),
    "bits": _Atom("bits", "bits", 1.0),
    "channels": _Atom("channels", "channels", 1.0),
    "cycles": _Atom("cycles", "cycles", 1.0),
    "adc_count": _Atom("adc_count", "adc_count", 1.0),
}

#: Every accepted spelling, in canonical form, for messages and for callers.
ACCEPTED_UNITS: tuple[str, ...] = tuple(dict.fromkeys(atom.canonical for atom in _ATOMS.values()))

#: Field-name suffix -> the dimension the stored value is in. Only the two the
#: schema names: a wider table invents a claim about fields it has not read.
_NAME_SUFFIX_DIMENSION: dict[str, str] = {"_deg": "angle", "_m": "length"}


class Unit(NamedTuple):
    """A parsed unit: what to multiply by, and what the result is called.

    A bullet list rather than an ``Attributes:`` section, as ``GLSResult`` in
    ``inference/gls.py`` already is: autodoc documents a NamedTuple's field
    aliases itself, so napoleon's second copy is a duplicate object description
    and sphinx warns on every field.

    * ``canonical`` -- the canonical spelling, e.g. ``"Hz"`` or ``"adc_count/K"``.
    * ``factor`` -- multiply a declared value by this to reach canonical.
    * ``offset`` -- add after scaling. Non-zero only for a bare affine atom
      (``celsius``), which is why an affine atom may not compose.
    * ``numerator`` -- canonical atom spellings above the line.
    * ``denominator`` -- canonical atom spellings below it.
    * ``dimension`` -- the single dimension when this unit is one atom, else None.
    """

    canonical: str
    factor: float
    offset: float
    numerator: tuple[str, ...]
    denominator: tuple[str, ...]
    dimension: str | None


# A9 derives its normalized signature exclusively from ``numerator`` and
# ``denominator`` above. Keep this compatibility record at six fields: a
# cached seventh dimension field would create two authorities that can drift.


def _atom(token: str) -> _Atom:
    found = _ATOMS.get(token.strip().lower())
    if found is None:
        raise ConfigError(
            f"Unknown unit {token!r}; this layer converts {list(ACCEPTED_UNITS)}. "
            "The table is small on purpose -- every token in it is either a "
            "canonical unit or a spelling one of the package's own file readers "
            "already accepts, and a unit it cannot convert is better refused than "
            "passed through, because a wrong factor produces a finite, "
            "correctly-shaped, wrong answer that nothing downstream can detect. "
            "Convert the value yourself and declare the canonical unit, or build "
            "the quantity through the python: hatch, which states its own cost."
        )
    return found


def canonical_unit(token: str) -> Unit:
    """Parse a unit token into a :class:`Unit`.

    Args:
        token: an atom (``"MHz"``), a product (``"K*s"``) or a quotient with at
            most one ``/`` (``"adc_count/K"``, ``"Hz/s"``).

    Raises:
        ConfigError: on an unknown atom, a second ``/``, an exponent or any
            other syntax -- this is a unit alphabet, not an expression
            language, and the boundary is the same one §2.3 draws for values.
    """
    text = str(token).strip()
    if not text:
        raise ConfigError(
            "A unit token is empty. Every dimensional value declares its unit; "
            "a value that genuinely has none declares 'dimensionless' or 'count', "
            "which is a statement rather than an omission."
        )
    if text.count("/") > 1:
        raise ConfigError(
            f"Unit {token!r} has {text.count('/')} '/' characters; this grammar "
            "allows at most one '/'. Write the whole denominator after it as a "
            "product -- 'K/s*s' rather than 'K/s/s' -- or, if the quantity really "
            "needs an expression, build it as a resources.arrays entry."
        )
    for illegal in ("^", "**", "(", ")"):
        if illegal in text:
            raise ConfigError(
                f"Unit {token!r} uses {illegal!r}. This is a product and quotient "
                "over a closed alphabet, not an expression language: there are no "
                "exponents, no parentheses and no precedence to reason about. "
                f"Write the atoms out ('m*m' for an area), or name the quantity as "
                "a resources.arrays entry and reference it."
            )

    head, slash, tail = text.partition("/")
    if slash and not head.strip():
        raise ConfigError(
            f"Unit {token!r} has nothing above the '/'. Write the numerator "
            "explicitly -- '1/s' has no atom '1' here, so use 'cycles/s' or "
            "'dimensionless/s', whichever states what the quantity is."
        )
    numerator = head.split("*")
    denominator = tail.split("*") if slash else []
    for segment in (*numerator, *denominator):
        if not segment.strip():
            raise ConfigError(
                f"Unit {token!r} has an empty segment -- split on '/' and '*' it "
                f"gives {[*numerator, *denominator]}, and one of those is blank. "
                "A stray or doubled separator is almost always a template that "
                "expanded to nothing, and dropping it silently would turn that "
                "into a unit which reads as valid: 'K/' would become 'K', and "
                "every value under it would be wrong by whatever the denominator "
                "should have been -- finite, correctly-shaped, and undetectable "
                "downstream. Write the missing atom out, or delete the separator."
            )

    # Resolve every atom before applying any rule, so an unknown token is
    # reported as one and the canonical form below -- which the affine refusal
    # hands back as its suggested remedy -- is built only from known atoms.
    numerator_atoms = [_atom(part) for part in numerator]
    denominator_atoms = [_atom(part) for part in denominator]
    compound = len(numerator_atoms) + len(denominator_atoms) > 1
    up = tuple(atom.canonical for atom in numerator_atoms)
    down = tuple(atom.canonical for atom in denominator_atoms)
    # Every canonical spelling is itself a key of _ATOMS, so this string always
    # parses back to an equal Unit. That is what lets a refusal quote it.
    canonical = "*".join(up) + (("/" + "*".join(down)) if down else "")

    for raw, atom in zip(numerator, numerator_atoms, strict=True):
        if atom.offset and compound:
            raise ConfigError(
                f"Unit {token!r} uses {raw.strip()!r} inside a compound. That atom "
                f"is affine -- it converts with an offset of {atom.offset} -- and an "
                f"affine unit has no meaning as a factor: 2 {text} is not "
                f"{2 + atom.offset} {canonical}. Declare the compound in "
                f"{atom.canonical} (here: {canonical}), and convert the offset "
                "where the quantity is still a scalar."
            )
    for raw, atom in zip(denominator, denominator_atoms, strict=True):
        if atom.offset:
            raise ConfigError(
                f"Unit {token!r} uses the affine atom {raw.strip()!r} as a "
                f"denominator. Dividing by a unit with an offset of {atom.offset} "
                f"is not defined; use {atom.canonical} below the line."
            )

    factor = 1.0
    for atom in numerator_atoms:
        factor *= atom.factor
    for atom in denominator_atoms:
        factor /= atom.factor

    # A compound is never affine: the loop above refused every way of getting
    # an offset into one, so the sole numerator atom is the only source left.
    offset = numerator_atoms[0].offset if not compound else 0.0
    dimension = numerator_atoms[0].dimension if not compound else None
    return Unit(canonical, factor, offset, up, down, dimension)


def convert_to_canonical(value, token: str):
    """Convert ``value`` from ``token`` into canonical units.

    Returns:
        ``(converted, unit)``. ``converted`` keeps ``value``'s own type where
        the conversion is exact (factor 1, offset 0), so a Python ``int``
        destined for a static field is not turned into a float on the way.
    """
    unit = canonical_unit(token)
    # Exact equality, deliberately, not a lazy `math.isclose`. Every factor
    # here is a product and quotient of the table's own literals, and x/x is
    # exactly 1.0 in IEEE-754 for any finite x -- so a unit that scales by
    # nothing lands ON 1.0 rather than near it ('ms*kHz' is exactly 1.0, and
    # no pair of atoms in the table lands near 1.0 without landing on it). A
    # tolerance here would be its own fuzz: it would return a genuinely
    # scaled value unscaled.
    if unit.factor == 1.0 and unit.offset == 0.0:
        return value, unit
    return value * unit.factor + unit.offset, unit


def check_field_name_unit(field_name: str, unit: Unit) -> None:
    """Cross-check a unit-suffixed Python field name against a declared unit.

    ``lat_deg`` stores degrees, so declaring radians is legal -- the value is
    converted before it is stored and the *canonical* unit is what the suffix
    describes. Declaring kelvin is not.

    Raises:
        ConfigError: when the field's suffix names a dimension and the
            canonical unit is in a different one.
    """
    for suffix, dimension in _NAME_SUFFIX_DIMENSION.items():
        if not field_name.endswith(suffix):
            continue
        if unit.dimension != dimension:
            raise ConfigError(
                f"Field {field_name!r} ends in {suffix!r}, so it stores a "
                f"{dimension} and the package reads it as one -- but the value "
                f"declares {unit.canonical!r}. The suffix is the field's own "
                "statement about what it holds and nothing converts between the "
                "two later, so the number would be used as if it were a "
                f"{dimension}. Declare a {dimension} unit (any spelling this "
                "layer converts), or write the value against the field that "
                "really holds it."
            )
    return None
