"""The value grammar: eight forms, one dispatcher, one result type.

Resolution knows nothing about the destination. It turns a fragment of a
document into a canonical-unit number and a record of which form produced it;
:mod:`rheplicant.config.delivery` then decides how that reaches a field. The
split is not tidiness -- the form is only answerable from the document and the
representation is only answerable from the target class, so a function taking
both could be tested against neither.
"""

import re
from collections.abc import Callable
from typing import Any, NamedTuple

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.modifiers import apply_modifiers
from rheplicant.config.units import Unit, convert_to_canonical

#: Every form key. A mapping value node holds exactly one of these.
VALUE_FORMS: tuple[str, ...] = (
    "value",
    "zeros",
    "ones",
    "full",
    "list",
    "linspace",
    "arange",
    "modulo",
    "from_grid",
    "basis_fit",
    "normal",
    "uniform",
    "file",
    "ref",
    "from",
    "stack",
    "from_switch_order",
    "python",
)

#: The nine modifier keys -- the schema's eight plus the delivery declaration ``as:``.
VALUE_MODIFIERS: tuple[str, ...] = (
    "unit",
    "dtype",
    "as",
    "axis",
    "column",
    "scale",
    "offset",
    "part",
    "normalize",
)

_SHORTHAND = re.compile(
    r"""^\s*
    (?P<number>[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)
    \s+
    (?P<unit>[A-Za-z_][A-Za-z_0-9]*(?:\s*[*/]\s*[A-Za-z_][A-Za-z_0-9]*)*)
    \s*$""",
    re.VERBOSE,
)


class ResolvedValue(NamedTuple):
    """What a value node resolved to, before any destination is considered.

    A bullet list rather than an ``Attributes:`` section, for the reason given
    on :class:`rheplicant.config.units.Unit`: napoleon's copy of a NamedTuple's
    fields duplicates the one autodoc already emits.

    * ``value`` -- the canonical-unit value: a Python scalar, a ``jnp`` array,
      or an arbitrary object for ``ref`` and ``python``.
    * ``unit`` -- the parsed unit, or ``None`` for a bare number.
    * ``source`` -- the form key that produced it. Check A40 reads this, and
      every refusal quotes it.
    * ``modifiers`` -- the modifier keys as written, for the caller that needs
      ``axis:`` (which is recorded, never applied). No default: a mutable
      default on a NamedTuple is shared between every instance, and every
      construction site here has the dict to hand anyway.
    """

    value: Any
    unit: Unit | None
    source: str
    modifiers: dict[str, Any]


def resolve_value(node: Any, context: ResolutionContext) -> ResolvedValue:
    """Resolve one value node.

    Args:
        node: a bare number, a ``"<number> <unit>"`` shorthand string, or a
            mapping holding exactly one form key plus any modifiers.
        context: the scope to resolve against.

    Raises:
        ConfigError: on zero or several form keys, on an unknown key beside a
            form, and on anything a form's own resolver refuses.
    """
    # MEASURED: this branch is redundant today -- bool IS int in Python, so a
    # bool already reaches the branch below and comes back as the same object
    # by the same expression. It is kept because it is the place any coercion
    # would be written: the moment the numeric branch normalises anything
    # (float(node) for a dtype, int(node) for a count) a bool would follow it
    # and a false in the document would arrive as 0, which reads as a legal
    # value everywhere downstream. Stating the type here costs one branch and
    # makes that a deliberate change rather than a side effect.
    if isinstance(node, bool):
        return ResolvedValue(node, None, "scalar", {})
    if isinstance(node, (int, float, complex)):
        return ResolvedValue(node, None, "scalar", {})
    if isinstance(node, str):
        return _resolve_shorthand(node)
    if not isinstance(node, dict):
        raise ConfigError(
            f"A value node is {type(node).__name__} ({node!r}). It must be a number, "
            "a '<number> <unit>' string, or a mapping holding exactly one of "
            f"{list(VALUE_FORMS)}. A bare list is not a value node -- write "
            "{list: [...]} so the form is stated rather than inferred from the "
            "Python type, which a YAML sequence and a stack share."
        )

    forms = [key for key in node if key in VALUE_FORMS]
    # Read before the sweep below, because a form may take its arguments as
    # siblings rather than nested under its own key. Only answerable when
    # exactly one form is present; with none or several the node is refused
    # either way, and the empty default reproduces the sweep as it was.
    arguments = _FORM_ARGUMENTS.get(forms[0], frozenset()) if len(forms) == 1 else frozenset()
    unknown = (
        []
        if arguments is None
        else [
            key
            for key in node
            if key not in VALUE_FORMS and key not in VALUE_MODIFIERS and key not in arguments
        ]
    )
    if unknown:
        matches = {key: [m for m in VALUE_MODIFIERS if m.startswith(key[:3])] for key in unknown}
        near = [
            f"{key!r} (did you mean one of {close}?)" if close else repr(key)
            for key, close in matches.items()
        ]
        also = f", and {forms[0]}: also takes {sorted(arguments)}" if arguments else ""
        raise ConfigError(
            f"Value node has unknown key(s) {', '.join(near)}. The forms are "
            f"{list(VALUE_FORMS)} and the modifiers are {list(VALUE_MODIFIERS)}{also}. An "
            "unrecognised key is refused rather than ignored: a mistyped modifier is "
            "silently dropped by any loader that reads only the keys it knows, and "
            "the run then differs from the document by exactly the thing that key "
            "was there to say."
        )
    if not forms:
        raise ConfigError(
            f"Value node {sorted(node)} holds no form key; one of {list(VALUE_FORMS)} "
            "is required. Modifiers describe a value, they do not make one."
        )
    if len(forms) > 1:
        raise ConfigError(
            f"Value node holds {len(forms)} form keys ({sorted(forms)}); exactly one "
            "is allowed. Two forms in one node have no defined order and no defined "
            "result. Name one of them as a resources.arrays entry and reference it "
            "with {ref: ...}, which is how this grammar composes."
        )

    form = forms[0]
    modifiers = {key: node[key] for key in node if key in VALUE_MODIFIERS}
    if form == "value":
        resolved = _resolve_scalar(node, modifiers)
    else:
        resolver = _RESOLVERS.get(form)
        if resolver is None:  # pragma: no cover - every declared form registers a resolver
            raise ConfigError(
                f"Form {form!r} is declared in the grammar but no resolver is registered "
                f"for it. Registered: {sorted(_RESOLVERS)}."
            )
        resolved = resolver(node, context, modifiers)
    # One exit, deliberately, rather than a call on each branch. Every form's
    # result passes through the same modifier order, and a form added in a
    # later task cannot opt out of it by forgetting to call -- which is the
    # failure the two-call-site version invites, and an invisible one: the
    # value stays finite and correctly shaped, it is simply not what the
    # document declared. The resolver's own `modifiers` dict is read back off
    # `resolved` rather than reused, because a form may have added to it.
    return resolved._replace(
        value=apply_modifiers(resolved.value, resolved.modifiers, form=resolved.source)
    )


def _resolve_scalar(node: dict, modifiers: dict) -> ResolvedValue:
    raw = node["value"]
    unit_token = modifiers.get("unit")
    if unit_token is None:
        return ResolvedValue(raw, None, "scalar", modifiers)
    converted, unit = convert_to_canonical(raw, unit_token)
    return ResolvedValue(converted, unit, "scalar", modifiers)


def _resolve_shorthand(text: str) -> ResolvedValue:
    match = _SHORTHAND.match(text)
    if match is None:
        raise ConfigError(
            f"{text!r} is not a '<number> <unit>' shorthand. The shorthand is exactly "
            "one number, whitespace, and one unit token -- '290 K', '60 MHz', "
            "'2.5 adc_count/K'. Anything else goes in the longhand form, "
            "{value: <number>, unit: <token>}, where there is nowhere for a stray "
            "word to hide."
        )
    number = float(match.group("number"))
    converted, unit = convert_to_canonical(number, match.group("unit"))
    return ResolvedValue(converted, unit, "scalar", {"unit": match.group("unit")})


#: Form key -> resolver. Populated by the modules that own each form, in the
#: shape core/graph.py:350 established: registration is an expression, the key
#: comes off the thing registered, and the refusal lists what is known.
_RESOLVERS: dict[str, Any] = {}

#: Form key -> the sibling keys that form takes beside the modifiers. Forms
#: 1-4 and 7 nest their arguments under their own key and so take none, which
#: is the default. ``from`` is registered with None, because schema 2.1.7
#: spells a derivation flat -- ``{from: <name>, ...arguments...}`` -- and WHICH
#: arguments are legal depends on which derivation was named, so the
#: dispatcher cannot know them. None means the form refuses its own strays,
#: and :func:`rheplicant.config.derive._from` does, naming the derivation and
#: the arguments it takes rather than the whole grammar. The guarantee is
#: unchanged either way: no key a document wrote is ignored.
_FORM_ARGUMENTS: dict[str, frozenset[str] | None] = {}


def register_form(
    name: str, *, arguments: frozenset[str] | None = frozenset()
) -> Callable[[Any], Any]:
    """Register a resolver for one value form. Returns the function.

    Args:
        name: the form key.
        arguments: the sibling keys this form takes beside the modifiers, or
            None when the legal siblings depend on the node's own content, in
            which case the form is responsible for refusing its own strays.
    """

    def _register(fn):
        _RESOLVERS[name] = fn
        _FORM_ARGUMENTS[name] = arguments
        return fn

    return _register


# Imported for its side effect, at the very bottom and nowhere else: the one
# place a circular import is deliberate and safe, because `arrays` imports only
# names already defined above it.
from rheplicant.config import arrays as _arrays  # noqa: E402,F401  (registers form 2)
from rheplicant.config import derive as _derive  # noqa: E402,F401  (registers form 6)
from rheplicant.config import draws as _draws  # noqa: E402,F401  (registers form 3)
from rheplicant.config import files as _files  # noqa: E402,F401  (registers form 4)
from rheplicant.config import hatch as _hatch  # noqa: E402,F401  (registers form 8)
from rheplicant.config import refs as _refs  # noqa: E402,F401  (registers forms 5 and 7)
