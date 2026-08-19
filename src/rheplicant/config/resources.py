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

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from _rheplicant_bootstrap.layering import merge_extends
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


#: Every registered kind, live. Kind modules add to it on import.
RESOURCE_KINDS = LiveNames(_KINDS)

_REF_PREFIX = "resources."


def check_unknown_keys(
    name: str,
    spec: dict,
    allowed: frozenset[str],
    *,
    label: str,
    note: str = "",
    hints: Mapping[str, str] | None = None,
) -> None:
    """Refuse any key ``label`` does not consume, naming what it does.

    Shared rather than reimplemented per kind, because the shape it exists to
    rule out has already appeared twice independently:
    ``kinds/sky_models.py``'s own ``_check_unknown_keys`` and (before this
    helper) the same sweep written by hand in ``kinds/beams.py``. Before
    either existed, a branch that only read the keys it happened to consume
    left a stray sibling key -- ``{kind: uniform, ..., spectral_index: 2.5}``
    -- silently discarded rather than refused. A discriminated union (``kind:``,
    ``format:``) is exactly the shape where that happens, because each branch
    is naturally written to read its own keys and nothing else, and nothing
    forces it to also account for what is left over.

    Args:
        name: the entry's dotted name, quoted first in the refusal.
        spec: the mapping under inspection.
        allowed: every key this consumer reads.
        label: what the sweep speaks for (``"kind: touchstone"``).
        note: a sentence appended to every refusal from this call site.
        hints: per-key sentences appended only when that key is among the
            unknown ones -- the z0-belongs-to-termination redirect, without
            the call site hand-rolling its own sweep to say it.
    """
    unknown = sorted(set(spec) - allowed)
    if not unknown:
        return
    tail = ""
    if hints:
        applicable = " ".join(hints[key] for key in unknown if key in hints)
        if applicable:
            tail += " " + applicable
    if note:
        tail += " " + note
    raise ConfigError(
        f"{name}: {label} does not take {unknown}; it takes {sorted(allowed)}.{tail}"
    )


class BuiltResources(NamedTuple):
    """What a ``resources:`` section produced.

    * ``resources`` — dotted name -> object, ready to put in a
      :class:`~rheplicant.config.context.ResolutionContext`.
    * ``shared_objects`` — groups of names that ended up as one object, for
      ``config.resolved.yaml``'s ``shared_objects:`` map. Grouped by
      ``id()``, which treats interned scalars (small ints, ``None``, short
      strings) as "shared" even when they were built independently -- a kind
      builder must hand back arrays or other real objects, never a bare
      Python scalar, or this map over-reports.
    * ``order`` — the order entries were built in, for provenance.
    """

    resources: dict[str, Any]
    shared_objects: tuple[frozenset[str], ...]
    order: tuple[str, ...]


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


def resolved_specs(section: Mapping[str, Any] | None) -> dict[str, dict]:
    """``resources:`` -> ``{"resources.<kind>.<name>": spec-after-``extends:``}``.

    **THE KEY IS THE DOTTED STRING**, exactly :func:`build_resources`' own and
    exactly :attr:`BuiltResources.resources`' -- never the bare name, never a
    tuple.  Callers select a kind with
    ``k.startswith("resources.projectors.")``.

    **TOTAL: IT NEVER RAISES**, and that is the whole reason this function
    exists rather than each reader calling ``_resolved_spec`` itself.
    Measured, ``_resolved_spec`` raises by name on six malformed shapes: an
    ``extends:`` cycle, a self-extend, a dangling parent, a cross-kind parent,
    a non-string ``extends:``, and ``{append: ...}`` beside a sibling key.  A
    pre-flight check that let one escape would be wrapped by the pass as
    *"pre-flight check 'A11' RAISED ConfigError: ..."* -- which **aborts the
    pass and hides every finding after it**, while every existing ``match=``
    pin still passes, because ``match=`` searches.  A green suite and a
    stack-trace-shaped user message is the worst of the two failures available
    here.

    So each malformed entry is **DROPPED** from the mapping.  A check that
    finds an entry missing **stands down** on it: refusing on "I could not
    tell" refuses documents that build.

    **What the backstop is, stated narrowly.**  For the six shapes above,
    :func:`build_resources` says the right sentence at the right phase -- each
    is a ``ConfigError`` naming the entry.  For a shape it does not model, the
    backstop is only *the builder's own exception*, and that may not be this
    layer's voice at all: measured, ``{arrays: {a: {1: 'x'}}}`` -- a spec whose
    KEY is not a string -- passes through here untouched and dies inside
    ``build_resources`` as a bare ``TypeError: 'int' object is not
    subscriptable``.  What this function guarantees is narrower than "the
    right sentence": it is that **the PASS is not aborted**, so every other
    finding on the document still reaches the user.

    **It reads ONE LAYER.**  A check must not call it once on
    ``document["resources"]`` and stop -- that closes the base route and leaves
    the ``variants:`` twin open.  Walk the layers with
    ``preflight/document.py::_task3_over_layers`` and call this per layer::

        return _task3_over_layers(document, lambda layer: _per_layer(
            resolved_specs(layer.get("resources"))))

    **The walk belongs to the CALLER, and the reason is not an import cycle.**
    An earlier draft of this docstring said a head import of
    ``_task3_over_layers`` here would close one; a reviewer tried it and it
    imports cleanly from all four entry points, ruff green.  The real reason
    is placement: this module sits BELOW both passes so that both can read it
    without either importing the other, and layering is a ``preflight/``
    concept -- the axes pass is handed a document that has already had its
    variant applied and has no layers to walk at all.

    Args:
        section: a layer's ``resources:`` block, or ``None``/anything else --
            a non-mapping is a shape :func:`build_resources` refuses, and a
            reader that has not built yet must not pre-empt that sentence.

    Returns:
        Dotted name -> the resolved spec, for every entry that resolves.  A
        SHALLOW copy per entry, so a caller cannot edit the user's document
        through it; nested values are still shared, as they are with
        :func:`build_resources`.
    """
    if not isinstance(section, Mapping):
        return {}
    specs: dict[str, dict] = {}
    kind_of: dict[str, str] = {}
    for kind, entries in section.items():
        if not isinstance(kind, str) or not isinstance(entries, Mapping):
            continue
        for name, spec in entries.items():
            if not isinstance(name, str) or not isinstance(spec, Mapping):
                continue
            dotted = f"resources.{kind}.{name}"
            specs[dotted] = dict(spec)
            kind_of[dotted] = kind

    done: dict[str, dict] = {}
    out: dict[str, dict] = {}
    for dotted in specs:
        try:
            out[dotted] = _resolved_spec(dotted, specs, kind_of, done, [])
        except Exception:  # noqa: BLE001 -- see below
            # `except Exception` and not `except ConfigError`, deliberately,
            # and there is a MEASURED seventh shape rather than a hypothetical
            # one: an `extends:` chain 4000 entries long raises RecursionError
            # out of `_resolved_spec` -- a RuntimeError, not a ConfigError, and
            # not one an enumeration of expected types would have listed. Under
            # `except ConfigError` that aborts the whole pass and hides every
            # finding on the document. Here the unresolvable entries are
            # dropped -- 992 of 4001 resolve -- and the pass runs on.
            continue
    return out


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
        # Bind derived fixed outputs only after the builder has returned.
        # The environment object is shared by every dependency snapshot, so
        # the next resource can resolve a sub-value's dimension immediately.
        scoped.with_resource(dotted, built[dotted])
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


# Imported for its side effect, at the very bottom and nowhere else: `kinds`
# imports `register_kind` from this module, so importing it any earlier would
# be circular. A caller who only imports `resources` still gets every
# registered kind, because this import runs as part of importing this module.
from rheplicant.config import kinds as _kinds  # noqa: E402,F401  (populates the registry)
