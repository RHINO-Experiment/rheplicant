"""Shape symbols: a closed table, an integer offset, and nothing more.

``examples/radio_digital_twin.py:71-78`` writes the frequency grid length by
hand five times in one 90-line script with nothing tying the copies together.
A symbol table fixes that without opening the door a value grammar closes:
there is no operator, no precedence and no evaluation order here, only a name,
an optional integer multiple and an optional integer offset.

``n_pix`` and ``n_alm`` are the exception, and deliberately so -- see
:func:`resolve_extent`.
"""

import dataclasses
import re

from rheplicant.config.errors import ConfigError

#: The closed table. Adding to it is a schema change, not a convenience.
SHAPE_SYMBOLS: tuple[str, ...] = (
    "n_time",
    "n_freq",
    "n_source",
    "n_pix",
    "n_alm",
    "n_load",
)

_FORM = re.compile(
    r"""^\s*
    (?:(?P<mult>\d+)\s*\*\s*)?
    (?P<symbol>[A-Za-z_][A-Za-z_0-9]*)
    (?:\s*(?P<sign>[+-])\s*(?P<offset>\d+))?
    \s*$""",
    re.VERBOSE,
)


@dataclasses.dataclass(frozen=True)
class ShapeScope:
    """The extents a shape symbol may resolve against.

    Attributes:
        n_time: ``len(observation.time.grid)``.
        n_freq: ``len(observation.freq.grid)``.
        n_source: ``len(observation.switching.order)``; 1 when not switching.
        nside: set only inside an entry that declares its own; ``n_pix``
            resolves from it and refuses without it.
        lmax: likewise for ``n_alm``.
        candidates: dotted names of entries that DO declare an ``nside`` or
            ``lmax``, quoted in the refusal so the reader is told what could
            have been meant instead of being told to guess.
    """

    n_time: int
    n_freq: int
    n_source: int = 1
    nside: int | None = None
    lmax: int | None = None
    candidates: tuple[str, ...] = ()

    def within(self, **extents) -> "ShapeScope":
        """A new scope with per-entry extents added. Never mutates."""
        return dataclasses.replace(self, **extents)


def _extent(symbol: str, scope: ShapeScope) -> int:
    if symbol == "n_time":
        return scope.n_time
    if symbol == "n_freq":
        return scope.n_freq
    if symbol == "n_source":
        return scope.n_source
    if symbol == "n_load":
        return scope.n_source - 1
    if symbol in ("n_pix", "n_alm"):
        declared = scope.nside if symbol == "n_pix" else scope.lmax
        key = "nside" if symbol == "n_pix" else "lmax"
        if declared is None:
            named = (
                " The entries that declare one here are "
                + ", ".join(repr(name) for name in scope.candidates)
                + "."
                if scope.candidates
                else ""
            )
            raise ConfigError(
                f"{symbol!r} has no value in this position: it is "
                f"{'12 * nside**2' if symbol == 'n_pix' else '(lmax+1)(lmax+2)/2'} "
                f"of a NAMED beam, sky or projector, and nothing here declares a "
                f"{key}.{named} A document may hold several at different {key}, so "
                f"there is no single one to fall back on -- and the wrong one is not "
                f"detectable downstream: the map comes back finite, correctly shaped "
                f"and at the wrong resolution. Write the {key} on this entry, or "
                f"write the integer extent out."
            )
        return (
            12 * declared * declared if symbol == "n_pix" else (declared + 1) * (declared + 2) // 2
        )
    raise ConfigError(
        f"Unknown shape symbol {symbol!r}; the table is {list(SHAPE_SYMBOLS)}. It is "
        "closed: a symbol resolves against the run's own axes or against an entry's "
        "declared resolution, and a name that resolves against neither would have to "
        "be guessed from context. Write the integer extent, or name the quantity as a "
        "resources.arrays entry and reference its shape."
    )


def resolve_extent(value, scope: ShapeScope) -> int:
    """Resolve one integer position of a shape.

    Args:
        value: a Python ``int``, a bare symbol, or ``"<int> * <symbol>"`` /
            ``"<symbol> +|- <int>"``. The multiple and the offset are each
            optional and may both appear, in which case the multiple binds to
            the symbol: ``"2 * n_freq - 1"`` is ``2 * n_freq`` minus one, not
            twice ``n_freq - 1``. That is a fixed combination rule, not
            precedence in the expression-language sense this module's refusals
            disclaim -- there is still no operator to apply, nothing to nest
            and no evaluation order to reason about. It is written down here
            rather than left to be inferred from an example.
        scope: the extents in force at this position.

    Raises:
        ConfigError: on an unknown symbol, on ``n_pix``/``n_alm`` with nothing
            to resolve against, or on anything the two legal arithmetic forms
            do not cover.
    """
    if isinstance(value, bool):
        raise ConfigError(
            f"A shape position holds {value!r}. Booleans are integers in Python and "
            "would silently give an extent of 0 or 1; write the integer."
        )
    if isinstance(value, int):
        return value
    text = str(value)
    match = _FORM.match(text)
    if match is None:
        raise ConfigError(
            f"Shape position {text!r} is not a symbol with an integer offset. The "
            "legal forms are 'n_freq', 'n_freq - 1', 'n_freq + 1' and '2 * n_time'; "
            "this is a symbol table, not an expression language -- there is no "
            "operator, no precedence and no evaluation order here. Compute the extent "
            "and write the integer, or name the quantity as a resources.arrays entry."
        )
    extent = _extent(match.group("symbol"), scope)
    if match.group("mult"):
        extent *= int(match.group("mult"))
    if match.group("offset"):
        delta = int(match.group("offset"))
        extent += delta if match.group("sign") == "+" else -delta
    return extent


def literal_shadowing_a_symbol(value, scope: ShapeScope) -> str | None:
    """The symbol a literal integer in a shape position equals, if any.

    Check A41 in the schema, and a report rather than a refusal: a literal 8
    may genuinely be 8. What it cannot be is *tied* to the grid, which is the
    whole failure -- five hand-copied grid lengths in one 90-line script.

    Where two extents are equal the symbol reported is the first in
    ``SHAPE_SYMBOLS`` order, which is what the loop below walks. On a tie both
    answers are true, so this is a stated convention rather than a correctness
    claim -- but it is stated, so a later reordering is a deliberate change
    rather than an accident.
    """
    # The bool check is redundant today: True and False are 1 and 0, and
    # neither survives the `value > 1` guard below. It is kept because that
    # guard exists for an unrelated reason -- silencing the n_source == 1
    # default -- and would take this protection with it if it were ever
    # relaxed to also report 0 or 1.
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    for symbol in ("n_time", "n_freq", "n_source"):
        if value == _extent(symbol, scope) and value > 1:
            return symbol
    return None


def resolve_shape(
    spec, scope: ShapeScope, *, form: str, instead: str
) -> tuple[tuple[int, ...], dict[int, str]]:
    """Resolve a whole shape, and report the literals that shadow a symbol.

    The resolution and the check A41 report are one function because they are
    one pass over one list, and because splitting them let them diverge: the
    array constructors paired them and the draw forms did not, so
    ``{zeros: [8]}`` reported a hand-copied grid length and
    ``{normal: {shape: [8]}}`` said nothing about the same 8. A report that
    depends on which constructor the writer reached for is worse than no
    report -- it is read as authoritative.

    It lives here rather than in either caller because both halves consult
    :data:`SHAPE_SYMBOLS` and the scope, which is this module's subject, and
    because the alternative on offer was ``draws`` importing a private name
    out of ``arrays``.

    Args:
        spec: the shape as written -- a list of integers and shape symbols.
        scope: the extents in force at this position.
        form: the form key, quoted in the refusal.
        instead: what to write if this is a scalar rather than a shape. It is
            the one clause that is genuinely the caller's to say and not this
            module's -- a scalar zero is ``{value: 0.0}`` and a scalar draw is
            an empty shape, and neither is deducible from the other. Required
            and undefaulted, so a form added later states it rather than
            inheriting whichever caller happened to be written first.

    Returns:
        ``(extents, shadowed)``. ``shadowed`` maps position -> the symbol a
        literal integer there happens to equal, and is empty for a shape
        written entirely in symbols.

    Raises:
        ConfigError: when ``spec`` is not a list, and from
            :func:`resolve_extent` for anything in it that is not a shape
            position.
    """
    if not isinstance(spec, (list, tuple)):
        raise ConfigError(
            f"{form}: expects a shape -- a list of integers or shape symbols -- and "
            f"got {spec!r}. {instead}"
        )
    shadowed = {
        index: symbol
        for index, entry in enumerate(spec)
        if (symbol := literal_shadowing_a_symbol(entry, scope)) is not None
    }
    return tuple(resolve_extent(entry, scope) for entry in spec), shadowed
