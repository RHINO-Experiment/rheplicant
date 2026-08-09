"""The resources DAG: build each named entry once, and hand back the object.

``{ref: ...}`` is object identity, not a copy, and that is a physics
requirement rather than an optimisation:
``BeamSpillOperator.from_projector`` is documented as "the one call that
cannot get the weight and the sky average out of step", and check B9 asks
whether two projectors nominally sharing a beam actually share the array. A
loader that rebuilds each reference passes every shape check and decouples
them silently.

Build order comes from the reference graph rather than from the document's
own order, so a reader does not have to know which entry to write first. A
cycle is refused by name -- ``resources.arrays`` is a let-binding, and a
let-binding that refers to itself has no value.

``extends:`` deep-merges over a sibling **of the same kind**: mappings merge,
lists replace, ``{append: [...]}`` extends, ``~key: null`` deletes. Lists
replacing rather than merging is the rule that forces a whole comparison into
one document -- split across two files, the halves can silently disagree in
exactly the keys the comparison is about.
"""

import copy
import dataclasses
from collections.abc import Callable
from typing import Any, NamedTuple

from rheplicant.config.context import ResolutionContext
from rheplicant.config.errors import ConfigError
from rheplicant.config.registry import LiveNames

#: kind -> builder(name, spec, context) -> object
_KINDS: dict[str, Callable[[str, dict, ResolutionContext], Any]] = {}


def register_kind(name: str):
    """Register a resource-kind builder. Returns the function."""

    def _register(fn):
        _KINDS[name] = fn
        return fn

    return _register


#: Every registered kind, live. Tasks 3-8 add to it by importing their modules.
RESOURCE_KINDS = LiveNames(_KINDS)

_REF_PREFIX = "resources."


class BuiltResources(NamedTuple):
    """What a ``resources:`` section produced.

    Attributes:
        resources: dotted name -> object, ready to put in a
            :class:`~rheplicant.config.context.ResolutionContext`.
        shared_objects: groups of names that ended up as one object, for
            ``config.resolved.yaml``'s ``shared_objects:`` map. Grouped by
            ``id()``, which treats interned scalars (small ints, ``None``,
            short strings) as "shared" even when they were built
            independently -- a kind builder must hand back arrays or other
            real objects, never a bare Python scalar, or this map over-reports.
        order: the order entries were built in, for provenance.
    """

    resources: dict[str, Any]
    shared_objects: tuple[frozenset[str], ...]
    order: tuple[str, ...]


def merge_extends(child: dict, parent: dict) -> dict:
    """Deep-merge ``child`` over ``parent``.

    Mappings merge; lists replace; ``{"append": [...]}`` extends the parent's
    list; a ``~key`` entry deletes ``key``.
    """
    merged = copy.deepcopy(parent)
    for key, value in child.items():
        if key.startswith("~"):
            merged.pop(key[1:], None)
            continue
        if isinstance(value, dict) and "append" in value:
            if set(value) != {"append"}:
                siblings = sorted(set(value) - {"append"})
                raise ConfigError(
                    f"{key!r}: append must be the only key when extending a list; "
                    f"got the sibling keys {siblings}."
                )
            base = merged.get(key, [])
            if not isinstance(base, list):
                raise ConfigError(
                    f"{key!r} is extended with {{append: ...}} but the inherited value "
                    f"is {type(base).__name__}, not a list."
                )
            # deepcopy the appended items too -- splicing them in by reference
            # would let a caller's later mutation of the merge RESULT reach
            # back into the document they passed in as `child`.
            merged[key] = [*base, *copy.deepcopy(value["append"])]
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_extends(value, merged[key])
            continue
        merged[key] = copy.deepcopy(value)
    return merged


def _referenced_names(node: Any) -> set[str]:
    """Every ``resources.<kind>.<name>`` a spec mentions, at any depth."""
    found: set[str] = set()
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "ref" and isinstance(value, str) and value.startswith(_REF_PREFIX):
                parts = value.split(".")
                if len(parts) >= 3:
                    found.add(".".join(parts[:3]))
            else:
                found |= _referenced_names(value)
    elif isinstance(node, (list, tuple)):
        for item in node:
            found |= _referenced_names(item)
    elif isinstance(node, str) and node.startswith(_REF_PREFIX):
        parts = node.split(".")
        if len(parts) >= 3:
            found.add(".".join(parts[:3]))
    return found


def _resolved_spec(
    dotted: str,
    specs: dict[str, dict],
    kind_of: dict[str, str],
    done: dict[str, dict],
    chain: list[str],
) -> dict:
    """Resolve one entry's ``extends:`` chain, recursively and memoised.

    A single pass over the document in declaration order merges a child over
    whatever its parent's *spec happens to hold at that point in the loop* --
    if the child is declared before its own parent (``a extends b, b extends
    c``, written in that order), the parent has not been merged with the
    grandparent yet, so the child silently loses the grandparent's keys and
    keeps a stray ``extends: 'c'`` in the spec handed to the builder. Resolving
    recursively (a parent is fully resolved before its child merges over it)
    and memoising in ``done`` fixes both: a chain resolves regardless of
    declaration order, and ``extends`` is stripped on both sides of every
    merge, so the spec a builder sees never carries it.

    ``chain`` is the entries currently being resolved, in order -- threading
    it through the recursion is what turns an ``extends:`` cycle into a named
    refusal instead of infinite recursion.
    """
    if dotted in done:
        return done[dotted]
    if dotted in chain:
        loop = chain[chain.index(dotted):] + [dotted]
        raise ConfigError(
            "resources: these entries extend each other in a loop: "
            + " -> ".join(loop)
            + ". extends: is a textual merge over a sibling, so a loop has no "
            "base document to start from. Break it by making one entry "
            "self-contained."
        )
    spec = specs[dotted]
    parent_name = spec.get("extends")
    if parent_name is None:
        done[dotted] = spec
        return spec
    parent = f"resources.{kind_of[dotted]}.{parent_name}"
    if parent not in specs:
        other = [name for name in specs if name.endswith(f".{parent_name}")]
        raise ConfigError(
            f"{dotted} extends {parent_name!r}, which resources.{kind_of[dotted]} "
            f"does not declare."
            + (
                f" It is declared as {other[0]}, and extends: merges between "
                "siblings of the SAME kind only -- the keys of two kinds describe "
                "different constructors, so a merge across them would produce an "
                "entry no builder can read."
                if other
                else ""
            )
        )
    base = _resolved_spec(parent, specs, kind_of, done, chain + [dotted])
    merged = merge_extends(
        {k: v for k, v in spec.items() if k != "extends"},
        {k: v for k, v in base.items() if k != "extends"},
    )
    done[dotted] = merged
    return merged


def build_resources(section: dict, context: ResolutionContext) -> BuiltResources:
    """Build every entry of a ``resources:`` section, once each, in dependency order.

    Raises:
        ConfigError: on an unknown kind, an ``extends:`` across kinds, a
            reference to an entry no kind declares, or a cycle.
    """
    if not isinstance(section, dict):
        raise ConfigError(
            f"resources: must be a mapping of kind -> name -> entry, got "
            f"{type(section).__name__}."
        )
    unknown = sorted(set(section) - set(_KINDS))
    if unknown:
        raise ConfigError(
            f"resources: unknown kind(s) {unknown}; the kinds are {sorted(_KINDS)}. Each "
            "names a family of objects the package constructs, so a kind this layer "
            "does not know has no constructor to call. A quantity that is just an "
            "array belongs under resources.arrays."
        )

    specs: dict[str, dict] = {}
    kind_of: dict[str, str] = {}
    for kind, entries in section.items():
        if not isinstance(entries, dict):
            raise ConfigError(f"resources.{kind}: must be a mapping of name -> entry.")
        for name, spec in entries.items():
            dotted = f"resources.{kind}.{name}"
            if not isinstance(spec, dict):
                raise ConfigError(
                    f"{dotted}: must be a mapping, got {type(spec).__name__}. Every "
                    "resources.<kind>.<name> entry is the arguments to that kind's "
                    "builder, and a builder reads its arguments off a mapping's keys."
                )
            specs[dotted] = spec
            kind_of[dotted] = kind

    # `extends:` first: it is a textual merge between siblings, so it happens
    # before anything is built and cannot depend on a constructed object.
    # Resolved recursively (see `_resolved_spec`) rather than in one pass over
    # the document, so a child declared before its own parent still merges
    # over the parent's fully-resolved spec, and an extends: cycle is refused
    # by name instead of merging over an incomplete parent.
    resolved: dict[str, dict] = {}
    for dotted in list(specs):
        specs[dotted] = _resolved_spec(dotted, specs, kind_of, resolved, [])

    dependencies = {dotted: set(_referenced_names(spec)) for dotted, spec in specs.items()}
    for dotted, needs in dependencies.items():
        missing = sorted(needs - set(specs))
        if missing:
            raise ConfigError(
                f"{dotted} references {missing}, which this document does not declare. "
                f"It declares {sorted(specs)}. A resource is built before it is read, "
                "so a reference to a name that does not exist has nothing to wait for."
            )

    built: dict[str, Any] = {}
    order: list[str] = []
    building: list[str] = []

    def _build(dotted: str) -> Any:
        if dotted in built:
            return built[dotted]
        if dotted in building:
            loop = building[building.index(dotted):] + [dotted]
            raise ConfigError(
                "resources: these entries reference each other in a loop: "
                + " -> ".join(loop)
                + ". resources.<kind>.<name> is a let-binding -- a name bound to a "
                "value -- and a binding that refers to itself has no value to bind. "
                "Break the loop by naming the shared quantity as a third entry both "
                "sides reference."
            )
        building.append(dotted)
        spec = specs[dotted]
        for needed in sorted(dependencies[dotted]):
            _build(needed)
        # A deliberate snapshot of everything built so far, not an incremental
        # add -- `context.with_resource` is the one-more-resource affordance;
        # this builder wants "every sibling built before me", all at once.
        scoped = dataclasses.replace(context, resources=dict(built))
        built[dotted] = _KINDS[kind_of[dotted]](dotted, spec, scoped)
        order.append(dotted)
        building.pop()
        return built[dotted]

    for dotted in specs:
        _build(dotted)

    groups: dict[int, set[str]] = {}
    for dotted, value in built.items():
        groups.setdefault(id(value), set()).add(dotted)
    shared = tuple(frozenset(names) for names in groups.values() if len(names) > 1)
    return BuiltResources(built, shared, tuple(order))
