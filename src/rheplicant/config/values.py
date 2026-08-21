"""The value grammar: eight forms, one dispatcher, one result type.

Resolution knows nothing about the destination. It turns a fragment of a
document into a canonical-unit number and a record of which form produced it;
:mod:`rheplicant.config.delivery` then decides how that reaches a field. The
split is not tidiness -- the form is only answerable from the document and the
representation is only answerable from the target class, so a function taking
both could be tested against neither.
"""

import dataclasses
import inspect
import re
from collections.abc import Callable
from typing import Any, NamedTuple

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.dimensions import (
    DimensionEnvironment,
    DimensionSignature,
    DimensionSpec,
    describe_signature,
    dimension_for,
    dimension_of,
    dimension_spec_for,
)
from rheplicant.config.errors import ConfigError
from rheplicant.config.modifiers import apply_modifiers
from rheplicant.config.units import Unit, canonical_unit, convert_to_canonical

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


@dataclasses.dataclass(frozen=True, slots=True)
class ResolutionTarget:
    """The catalog row and concrete document address governing one value."""

    destination: DestinationDescriptor
    spec: DimensionSpec
    expected: DimensionSignature | None
    explicit_unit: Unit | None
    inherits_outer_unit: bool = False
    formula_name: str | None = None
    formula_role: str = "result"

    def operand(
        self,
        node: object,
        segment: str | int,
        *,
        formula: str,
        role: str,
        environment: DimensionEnvironment,
        destination: DestinationDescriptor | None = None,
    ) -> "ResolutionTarget":
        return target_for_formula_operand(
            self,
            node=node,
            segment=segment,
            formula=formula,
            role=role,
            environment=environment,
            destination=destination,
        )


def _declared_unit(node: object) -> str | None:
    if isinstance(node, str):
        match = _SHORTHAND.match(node)
        return None if match is None else match.group("unit")
    if isinstance(node, dict):
        if "unit" not in node:
            return None
        token = node["unit"]
        if not isinstance(token, str):
            raise ConfigError(
                f"dimensions: unit modifier is a string; got {token!r} ({type(token).__name__})"
            )
        return token
    return None


def validate_declared_unit(node: object, target: ResolutionTarget) -> ResolutionTarget:
    """Validate only the node's declaration, before a resolver can do I/O."""
    token = _declared_unit(node)
    if target.spec.disposition == "structural" and token is not None:
        raise ConfigError(f"dimensions: {target.destination.document_path!r} is structural")
    if target.spec.unit_policy == "forbidden" and token is not None:
        raise ConfigError(f"dimensions: {target.destination.document_path!r} forbids unit")
    if target.spec.unit_policy == "required" and token is None and not target.inherits_outer_unit:
        raise ConfigError(f"dimensions: {target.destination.document_path!r} requires unit")
    unit = canonical_unit(token) if token is not None else target.explicit_unit
    if unit is not None and target.expected is not None and dimension_of(unit) != target.expected:
        raise ConfigError(
            f"dimensions: {target.destination.document_path!r} declares {token!r}, "
            f"but requires {describe_signature(target.expected)}"
        )
    return dataclasses.replace(target, explicit_unit=unit)


def make_resolution_target(
    node: object, destination: DestinationDescriptor, environment: DimensionEnvironment
) -> ResolutionTarget:
    spec = dimension_spec_for(destination)
    expected = dimension_for(destination, environment)
    return validate_declared_unit(node, ResolutionTarget(destination, spec, expected, None))


def target_for_formula_operand(
    parent: ResolutionTarget,
    *,
    node: object,
    segment: str | int,
    formula: str,
    role: str,
    environment: DimensionEnvironment,
    destination: DestinationDescriptor | None,
) -> ResolutionTarget:
    from rheplicant.config import dimensions

    try:
        registration = dimensions._FORMULA_REGISTRY[formula]
    except KeyError as error:
        raise ConfigError(f"dimensions: no formula named {formula!r}") from error
    matches = [candidate for candidate in registration.operands if candidate.role == role]
    if len(matches) != 1:
        raise ConfigError(f"dimensions: formula {formula}.{role} is not registered")
    operand = matches[0]
    addressed = parent.destination.nested(segment)
    if operand.spec.resolver == "outer":
        if destination is not None:
            raise ConfigError(f"dimensions: {formula}.{role} inherits its outer destination")
        return validate_declared_unit(
            node,
            dataclasses.replace(
                parent,
                destination=addressed,
                inherits_outer_unit=True,
                formula_name=formula,
                formula_role=role,
            ),
        )
    if operand.spec.disposition in ("open", "structural"):
        if destination is None:
            raise ConfigError(f"dimensions: {formula}.{role} requires an explicit destination")
        addressed = dataclasses.replace(destination, document_path=addressed.document_path)
        declared = dimension_spec_for(addressed)
        agreement = (
            declared.disposition,
            declared.signature,
            declared.resolver,
            declared.unit_policy,
        ) == (
            operand.spec.disposition,
            operand.spec.signature,
            operand.spec.resolver,
            operand.spec.unit_policy,
        )
        if not agreement:
            raise ConfigError(f"dimensions: {formula}.{role} disagrees with catalog")
    elif destination is not None:
        raise ConfigError(f"dimensions: fixed role {formula}.{role} is formula-addressed")
    expected = (
        operand.spec.signature
        if operand.spec.disposition == "fixed"
        else dimension_for(addressed, environment, outer=parent.expected)
    )
    return validate_declared_unit(
        node,
        ResolutionTarget(
            addressed,
            operand.spec,
            expected,
            None,
            formula_name=formula,
            formula_role=role,
        ),
    )


def resolve_value(
    node: Any,
    context: ResolutionContext,
    *,
    destination: DestinationDescriptor | None = None,
) -> ResolvedValue:
    """Resolve one value node.

    Args:
        node: a bare number, a ``"<number> <unit>"`` shorthand string, or a
            mapping holding exactly one form key plus any modifiers.
        context: the scope to resolve against.

    Raises:
        ConfigError: on zero or several form keys, on an unknown key beside a
            form, and on anything a form's own resolver refuses.
    """
    target = (
        make_resolution_target(node, destination, context.dimensions)
        if destination is not None
        else None
    )
    return resolve_registered_form(node, context, target)


def resolve_operand(
    node: object,
    context: ResolutionContext,
    parent: ResolutionTarget,
    *,
    segment: str | int,
    formula: str,
    role: str,
    destination: DestinationDescriptor | None = None,
) -> ResolvedValue:
    if parent is None:  # compatibility for direct utility form calls
        return resolve_registered_form(node, context, None)
    target = parent.operand(
        node,
        segment,
        formula=formula,
        role=role,
        environment=context.dimensions,
        destination=destination,
    )
    return resolve_registered_form(node, context, target)


def resolve_registered_form(
    node: Any,
    context: ResolutionContext,
    target: ResolutionTarget | None,
) -> ResolvedValue:
    """The sole value-form dispatcher, after destination validation."""
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
        resolved = resolver(node, context, modifiers, target)
    # One exit, deliberately, rather than a call on each branch. Every form's
    # result passes through the same modifier order, and a form added in a
    # later task cannot opt out of it by forgetting to call -- which is the
    # failure the two-call-site version invites, and an invisible one: the
    # value stays finite and correctly shaped, it is simply not what the
    # document declared. The resolver's own `modifiers` dict is read back off
    # `resolved` rather than reused, because a form may have added to it.
    return resolved._replace(
        value=apply_modifiers(
            resolved.value,
            resolved.modifiers,
            form=resolved.source,
            context=context,
        )
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
        # The compatibility adapter makes every registered resolver obey the
        # destination-aware four-argument protocol while preserving external
        # third-party forms written for the historical three arguments.
        try:
            inspect.signature(fn).bind_partial(None, None, None, None)
        except TypeError:
            takes_target = False
        else:
            takes_target = True

        def _with_target(node, context, modifiers, target):
            if takes_target:
                return fn(node, context, modifiers, target)
            return fn(node, context, modifiers)

        _RESOLVERS[name] = _with_target
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
