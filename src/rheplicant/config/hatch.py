"""Form 8: the one escape hatch, and the cost it states.

Everything the value grammar deliberately cannot do goes here: elementwise
arithmetic beyond ``scale``/``offset``, trig, powers, logs, nested calls with
positional operands, a pipeline topology outside ``RADIO_GRAPH``.
``30*cos(linspace(0,3,8))`` needs an evaluator, precedence and a namespace,
and that is a programming language rather than a schema.

Composition of *named* quantities does not need the hatch:
``resources.arrays.<name>`` binds a value node to a name and ``{ref: ...}``
reads it back, which gives a DAG of named quantities and nothing more.

**The cost, stated rather than implied.** The run is no longer reproducible
from the config alone -- the hash covers the *string*, not the code. The
object must be importable in the run process and, inside the gradient path,
jax-traceable. On a static field it is hashed by identity, so an equivalent
re-created function misses the jit cache. With ``raw_bind``,
``ParameterSpace.validate``'s per-selector checks cannot run at all.

**What a document is trusted to do.** :mod:`rheplicant.config.files` records
the same assumption next to path resolution; this is the sharper half of it,
so it belongs beside the function that acts on it rather than in one place for
both. A document that reaches this form runs arbitrary named code in the
process that loaded it, with that process's privileges -- there is no sandbox
here and no allow-list of importable packages, and neither would be honest
about what the form is for. **Importing alone is sufficient.** Measured:
:func:`import_target` calls ``importlib.import_module``, which executes the
named module's body whether or not anything is subsequently called, so
``{python: "pkg.mod:anything"}`` runs ``pkg/mod.py`` even with no ``args:``
key -- resolving a document is enough, and a loader that only *resolves* a
document in order to report on it has already executed it. The assumption
underneath is the one ``files`` states: whoever wrote the document is whoever
is running the pipeline, in which case the document can do nothing its author
could not do at a shell.

That assumption stops holding the moment a document arrives from somewhere
else -- a shared root, a CI artefact, a collaborator's YAML. Two consequences
follow and are worth writing down. The top-level ``plugins:`` key this
module's own refusals point at -- whose loader arrives with the document
loader, in Plan 1B -- is not a declaration of intent but a list of module
bodies to execute, because importing for the registration side effect is the
whole mechanism by which a plugin registers; and those imports have to happen
before any value node is resolved, since resolution is what needs the names.
Reading the ``plugins:`` of an unfamiliar document is therefore reading code
that will run first, unconditionally, whatever else the document says. And a
published ``config.resolved.yaml`` records the target *string* --
``_python: "pkg.mod:fn"`` -- which says which code ran and nothing whatever
about what that code was, so the artefact is evidence of provenance and not of
behaviour. Two runs whose resolved configs match byte for byte can have done
different things if the named package changed between them; the file hash a
``file:`` entry records has no counterpart here, because there is no single
artefact to hash. The remedy, when a document is not your own, is to read its
``plugins:`` and its ``python:`` targets before loading it, in a process that
can afford what they do. This is recorded here rather than in an issue because
it is a property of the form and not a defect in it: an escape hatch that
could be made safe would not be an escape hatch.

**Calling is spelled by the document, not inferred from the object.** Writing
``args:`` or ``literal:`` -- either of them, even empty -- calls the attribute;
writing neither delivers the attribute itself. The alternative considered and
rejected was "call it if it turns out to be callable", which reads well on
``{python: "math:pi"}`` and has two defects. It decides what a node *means*
from a property of code the document does not contain, which is the exact
reproducibility cost this module exists to state rather than spread; and it
leaves no spelling at all for handing over a named function, which
``core/operator.py:117`` needs -- ``LambdaOperator.fn`` is a static
``Callable[[State], State]`` and :mod:`rheplicant.config.delivery` records
that this hatch is the only route to a field like it. Under this rule the two
intents are one key apart and both are visible in the document:
``{python: "pkg:fn"}`` hands over ``fn``, ``{python: "pkg:fn", args: {}}``
hands over ``fn()``.
"""

import importlib
from typing import Any

from _rheplicant_bootstrap.types import DestinationDescriptor
from rheplicant.config.context import ResolutionContext
from rheplicant.config.delivery import record_resolved_delivery
from rheplicant.config.errors import ConfigError
from rheplicant.config.units import convert_to_canonical
from rheplicant.config.values import ResolutionTarget, ResolvedValue, register_form


def import_target(target: str) -> Any:
    """Import ``"package.module:attribute"`` and return the attribute.

    Raises:
        ConfigError: on a target that is not one colon-separated pair, on a
            module that cannot be imported, and on an attribute the module
            does not carry.
    """
    if not isinstance(target, str) or target.count(":") != 1 or not all(target.split(":")):
        raise ConfigError(
            f"python: {target!r} is not a target. The form is "
            "'package.module:attribute' -- one colon, a module on the left and an "
            "attribute on the right. A dotted name alone is ambiguous: "
            "'jax.numpy.zeros' could be the attribute 'zeros' of module 'jax.numpy' "
            "or the attribute 'numpy.zeros' of package 'jax', and the two fail "
            "differently."
        )
    module_name, attribute = target.split(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ConfigError(_import_failure_message(module_name, exc)) from exc
    try:
        return getattr(module, attribute)
    except AttributeError as exc:
        offered = sorted(name for name in dir(module) if not name.startswith("_"))
        raise ConfigError(
            f"python: {module_name!r} has no attribute {attribute!r}. It offers {offered}."
        ) from exc


def _import_failure_message(module_name: str, exc: ImportError) -> str:
    """Say which of the two import failures happened, because the remedies differ.

    "The module is not installed" and "the module is installed and its own
    body raised" are one exception type apart and nothing else distinguishes
    them in the traceback a document's author sees. Answering both with
    "cannot import X -- name it under plugins:" is worse than vague: adding X
    to ``plugins:`` changes nothing at all when X was found and *its*
    dependency was not, so the advice sends the reader to edit the one key
    that cannot help.

    The two are told apart by ``ModuleNotFoundError.name``, which is the
    module the interpreter failed to find rather than the one it was asked
    for. Importing ``a.b.c`` when ``a`` is absent reports ``name='a'``, so the
    test is a prefix match and not equality. A plain ``ImportError``, or one
    carrying no name, falls to the second message: it is the safe default,
    because it names the real exception instead of prescribing a fix.
    """
    missing = getattr(exc, "name", None)
    absent = isinstance(exc, ModuleNotFoundError) and missing is not None
    if absent and (module_name == missing or module_name.startswith(f"{missing}.")):
        return (
            f"python: cannot import {module_name!r} ({exc}). The object must be "
            "importable in the process that runs this config. If it lives in a "
            "package that has to be imported first for a registry name to resolve, "
            "name that package under the document's top-level plugins: key."
        )
    return (
        f"python: {module_name!r} was found, and importing it failed: "
        f"{type(exc).__name__}: {exc}. The target is spelled correctly and this "
        "document is not what is wrong -- the module's own import did not "
        "complete, so the fix is in that module or in the environment it expects, "
        "and naming it under plugins: would change nothing."
    )


def _call(attribute: Any, keywords: dict, target: str) -> Any:
    """Call the imported attribute, and let nothing out of it without the target.

    The wrapper :func:`rheplicant.config.files._read` applies to a reader and
    :func:`rheplicant.config.derive._package_guard` to a derivation, for the
    same reason and with the same catch-``Exception`` argument: a guard that
    enumerates the types it expects reads every other shape as success. Here
    the range is wider than either, not narrower -- this is the one form whose
    callee is chosen by the document, so its failure modes are not a set this
    module could list even in principle. ``TypeError`` for a signature the
    ``args:`` keys do not fit is merely the most common.
    """
    try:
        return attribute(**keywords)
    except Exception as exc:
        raise ConfigError(
            f"python: calling {target} raised {type(exc).__name__}: {exc}. That "
            f"message is the callee's, and it knows nothing about the document. The "
            f"keyword(s) it was called with were {sorted(keywords)}, taken from args: "
            "and literal:; the hatch passes keywords only, so a callee whose operands "
            "are positional-only cannot be reached through it directly."
        ) from exc


# `args` and `literal` are flat siblings of `python:` rather than nested under
# it, so the dispatcher would refuse them as unknown keys unless it is told.
# A set and not None: which siblings are legal here is fixed by the grammar and
# does not depend on the node's own content, which is what separates this from
# `from` -- there the legal arguments follow from WHICH derivation was named,
# so only the form can know them. Registering the set rather than refusing
# strays inside `_python` keeps one copy of that check: the dispatcher's
# refusal already names them ("and python: also takes ['args', 'literal']"),
# and a second check here would be unreachable behind it.
@register_form("python", arguments=frozenset({"args", "literal"}))
def _python(
    node: dict,
    context: ResolutionContext,
    modifiers: dict,
    resolution_target: ResolutionTarget | None,
) -> ResolvedValue:
    from rheplicant.config.values import resolve_operand

    target = node["python"]
    args = node.get("args", {})
    literal = node.get("literal", {})
    for name, given in (("args", args), ("literal", literal)):
        if not isinstance(given, dict):
            raise ConfigError(
                f"python: {name}: is a mapping of argument name to "
                + ("value node" if name == "args" else "value")
                + f", and this one is {type(given).__name__} ({given!r}). The hatch "
                "passes keywords only, so there is nowhere for a bare list to go."
            )
    clash = sorted(set(args) & set(literal))
    if clash:
        raise ConfigError(
            f"python: {clash} appear in both args and literal. args values are resolved "
            "through the value grammar and literal values are forwarded untouched, so "
            "one argument cannot be both -- and which one won would decide whether a "
            "{file: ...} was read or passed through as a dict."
        )

    argument_targets = {}
    if resolution_target is not None:
        for name, spec in args.items():
            argument_targets[name] = resolution_target.operand(
                spec,
                f"args.{name}",
                formula="python",
                role="args.*",
                environment=context.dimensions,
                destination=DestinationDescriptor(
                    f"python.args.{name}", "config_path", "python.args.*"
                ),
            )

    # Import AFTER the node has been checked, deliberately. Importing a module
    # runs its body, so a node this layer is going to refuse anyway must be
    # refused before it can have that effect.
    if context.audit is not None:
        document_path = (
            "python" if resolution_target is None else resolution_target.destination.document_path
        )
        context.audit.python_target(document_path, target)
    attribute = import_target(target)
    keywords = {}
    for name, spec in args.items():
        resolved = resolve_operand(
            spec,
            context,
            parent=resolution_target,
            segment=f"args.{name}",
            formula="python",
            role="args.*",
            destination=DestinationDescriptor(
                f"python.args.{name}", "config_path", "python.args.*"
            ),
        )
        keywords[name] = resolved.value
        if resolution_target is not None:
            record_resolved_delivery(
                context,
                argument_targets[name].destination,
                resolved.unit,
            )
    keywords.update(literal)
    # Presence of the KEY, not truth of its value: `args: {}` is how a document
    # spells a call that takes no arguments, and it is the only spelling there
    # is, so reading it as "no args were written" would leave that intent with
    # none. See the module docstring for why the rule is syntactic.
    called = "args" in node or "literal" in node
    value = _call(attribute, keywords, target) if called else attribute

    carried = {**modifiers, "_python": target}
    unit_token = modifiers.get("unit")
    if unit_token is None:
        return ResolvedValue(value, None, "python", carried)
    converted, unit = convert_to_canonical(value, unit_token)
    return ResolvedValue(converted, unit, "python", carried)
