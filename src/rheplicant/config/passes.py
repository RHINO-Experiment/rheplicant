"""One runner, shared by every pass this layer has.

**Why this module exists.**  Plan 3A shipped a single pass -- ``preflight/``,
the text pass -- and wrote its registry, its de-duplication, its
raise-guard and its ``where`` guard inside that package's ``__init__``.  Plan
3B adds two more passes (the axes pass, in front of ``build_resources``, and
the built pass, when ``load_document`` is ready to return), and three copies
of one runner is the ``_number``-vs-``_whole`` divergence on the 2C ledger:
two validators for one property, disagreeing, each with its own tests, and
nothing comparing them.  So the machinery moves here **whole** and the three
passes differ only in the registry they hand it and the two words it says.

**This is an extraction and not a rewrite.**  Every refusal below is Plan
3A's, character for character, with two interpolations where a constant word
used to be:

* ``label`` -- the phase word that OPENS a refusal about a check.
  ``preflight`` passes ``"pre-flight"``, so its messages stay byte-identical
  to 3A's and ``tests/config/test_config_preflight.py`` passes unchanged; the
  two new passes pass ``"in-flight"``.
* ``decorator`` -- the name a reader is told to WRITE.  Three registries mean
  three decorators (``register``, ``register_axes``, ``register_built``), and
  advice naming the wrong one is advice that cannot be followed.  It is a
  second parameter and not a second use of ``label`` because ``register_axes``
  and ``register_built`` share the label ``"in-flight"`` and differ here.

**How that is defended.**  A substring pin cannot see either word: measured
while this module was written, rewriting all six occurrences of ``pre-flight``
in ``preflight/__init__.py`` to ``in-flight`` left the whole of
``tests/config`` at **exit 0**, and so did rewriting the ``@register``
advisories.  ``test_config_preflight.py``'s
``TestEveryRefusalOfThisPassIsPinnedWHOLE`` is the eight equality pins that
close it, and ``test_config_inflight.py`` pins the in-flight spellings.

**What is NOT here.**  ``preflight``'s ``_structural`` stays in ``preflight/``:
it raises rather than collects, it is about the document's top level rather
than about a check, and the in-flight passes run after it has already held.
``_SECTIONS`` stays there too and is passed in, because it is the pre-flight
package's own knowledge of the document's shape and this module has no
business owning a list of section names.

**Scope.**  This module reads a registry and a payload and calls functions.
It opens no file, builds nothing and imports no optional dependency, so it is
inside every pass's boundary at once; ``tests/config/test_config_preflight.py``
and ``tests/config/test_config_inflight.py`` both walk it.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, Report
from rheplicant.config.paths import parse_path

__all__ = ["SLOT", "Check", "Registry", "binder", "check_where", "sweep"]

#: One check: a payload in, findings out, nothing raised.  The payload is the
#: document for the text pass, an ``Axes`` for the axes pass and a ``Built``
#: for the built pass, which is why this is ``Any`` rather than a mapping --
#: the registries are separate precisely so a check cannot be bound in a slot
#: whose payload it cannot read.
Check = Callable[[Any], Iterable[Finding]]

#: id -> the function.  **Insertion order IS run order** in every pass.
Registry = dict[str, Check]

#: A registry SLOT: a schema §6 id, optionally with a dotted suffix.  Moved
#: from ``preflight/__init__.py::_T2_SLOT`` unchanged.  It already admits ``B``
#: and ``C`` ids, which is why the two new passes need no widening of it --
#: measured, ``"C1"``, ``"C2.time"`` and ``"A13.grid"`` all ``fullmatch`` and
#: ``"A12a"`` does not (a bare letter suffix is not a dotted slot).
SLOT = re.compile(r"[ABC][1-9][0-9]*(\.[a-z_]+)?")


def binder(registry: Registry, *checks: str, label: str,
           decorator: str) -> Callable[[Check], Check]:
    """Bind one or more check ids to one function, in ``registry``.

    Plan 3A's ``preflight.register``, with the registry and the two words as
    arguments.  Raises ``ConfigError`` -- never an ``assert``, which
    ``python -O`` strips.  That is the ledger item closed at
    ``sections/exit_support.py:31``, where the strip was measured to let the
    second registration win in silence.

    A SLOT is a schema §6 id, optionally with a dotted suffix (``"A1.runs"``,
    ``"C2.time"``) when several functions each decide part of one check.
    ``Finding.check`` stays the bare id.

    VARIADIC, not stacked.  ``@register("A20") @register("A21")`` applies
    BOTTOM-UP, so the stacked form inserts the last decorator's id first and
    insertion order -- which IS run order -- comes out reversed.

    Every id is validated before any is bound, and every id is checked for a
    clash before any is bound: a registration that failed half way through
    would leave a module's ids partly claimed, after which the next import of
    that module reports "registered twice" about a module that never finished.

    Args:
        registry: the dict to bind into.  Passed rather than closed over, so
            that one implementation serves three passes.
        checks: the slot ids this function claims.
        label: the phase word a refusal opens with (``"pre-flight"``).
        decorator: the name a refusal tells the reader to write
            (``"register"``, ``"register_axes"``, ``"register_built"``).

    Raises:
        ConfigError: no id at all; an id that is not a string (the decorator
            written without its parentheses); an id that is not a schema §6
            slot; one id named twice in one call; or an id another function
            already claims.
    """
    if not checks:
        raise ConfigError(
            f"{decorator}() takes one or more check ids -- "
            f"@{decorator}('A30'), or @{decorator}('A16', 'A17') when one "
            "function decides several. A registration with no id binds "
            "nothing, so the check it decorates never runs and nothing says "
            "so."
        )
    for check in checks:
        if not isinstance(check, str):
            raise ConfigError(
                f"{label} check id {check!r} is not a string. @{decorator} is "
                f"called with its ids -- @{decorator}('A30') -- and a bare "
                f"@{decorator} hands the decorated function in as an id."
            )
        if not SLOT.fullmatch(check):
            raise ConfigError(
                f"{label} check id {check!r} is not a schema §6 id "
                "(A1..A52, B1..B9, C1..C19), optionally with a dotted "
                "suffix such as 'A1.runs' when several functions each "
                "decide part of one check. The id is what a Finding "
                "carries and what a reader looks up; a private name here "
                "reaches the user as '(check _mine).'"
            )
    for check in checks:
        if checks.count(check) > 1:
            raise ConfigError(
                f"this registration names {check!r} twice. The variadic form "
                f"binds one function to several DIFFERENT ids -- "
                f"@{decorator}('A16', 'A17') -- and a repeated id is a typo "
                "for one that is now claimed by nobody."
            )

    def bind(fn: Check) -> Check:
        for check in checks:
            if check in registry:
                raise ConfigError(
                    f"{label} check {check!r} is registered twice, by "
                    f"{registry[check].__module__} and by {fn.__module__}. A "
                    "check id has one function, and which of the two would "
                    "run depends on import order."
                )
        for check in checks:
            registry[check] = fn
        return fn

    return bind


def sweep(registry: Registry, payload: Any, *, label: str,
          sections: tuple[str, ...]) -> Report:
    """Run every check in ``registry`` over ``payload``, once each, and collect.

    Plan 3A's ``preflight`` body below ``_structural``, with the registry, the
    payload and the label as arguments.

    A check RETURNS findings and raises nothing, so a user with four errors
    sees four.  A check that raises would abort the pass and hide every later
    finding, so this catches that and says which check did it rather than
    letting the pass truncate in silence.

    Args:
        registry: the pass's slots, in insertion order.
        payload: whatever this pass's checks read -- the document, an
            ``Axes``, a ``Built``.
        label: the phase word the two refusals below open with.
        sections: the document's top-level section names, for the ``where``
            guard.  Passed rather than imported: this module has no business
            owning the document's shape, and the guard's own message quotes
            the list.
    """
    findings: list[Finding] = []
    seen: set[int] = set()
    for check, fn in registry.items():
        # ONE FUNCTION CARRIES SEVERAL IDS and every pass does it: in
        # `preflight/` `_blocks` is A16-A19, `_prior_gates` is A20/A21/A23,
        # `_counts` is A24+A25, `_decided` is A27+A28; in `inflight/` the
        # time-axis function is C1+C2.time.  Iterating naively calls `_blocks`
        # FOUR times and emits every finding four times.
        #
        # By `id`, not by name: two module-level checks can share a
        # `__name__` (every lambda is `<lambda>`), and a walk keyed on the
        # name would drop the second in silence.  `id` cannot be recycled
        # here because `registry` holds a reference to every function for the
        # whole loop.
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        try:
            found = tuple(fn(payload))
        except Exception as error:
            raise ConfigError(
                f"{label} check {check!r} RAISED "
                f"{type(error).__name__}: {error}. A check returns findings "
                "and raises nothing -- one that raises aborts the pass and "
                "hides every finding after it, which is the failure the "
                "collect-rather-than-raise design exists to prevent."
            ) from error
        for finding in found:
            check_where(label, check, finding, sections)
            findings.append(finding)
    return Report(findings=tuple(findings))


def check_where(label: str, check: str, finding: Finding,
                sections: tuple[str, ...]) -> None:
    """``Finding.where`` is a path into the USER'S document, never into ``src/``.

    Plan 3A's ``preflight._check_where``, with the label and the section names
    as arguments, and true of every slot including the built one: the payload
    carries ``document`` precisely so that a built-slot check can say
    ``model.averaging`` rather than a source path or an object repr.

    Not a test, because a ``where`` is often computed from the document and a
    test on a fixed set of documents cannot see the branch that builds a bad
    one.  ``parse_path`` (``config/paths.py:38``) is the layer's own path
    grammar and already refuses a source path: measured,
    ``parse_path('src/rheplicant/config/model.py')`` raises and
    ``parse_path('runs[2].blocks[0]')`` returns ``('runs', 2, 'blocks', 0)``,
    which is the head this needs.

    Takes the registry SLOT rather than ``finding.check``: the slot is what
    names the culprit, and ``finding.check`` is empty for a finding that
    carries no id at all.
    """
    try:
        head = parse_path(finding.where)[0]
    except ConfigError as error:
        raise ConfigError(
            f"{label} check {check!r} emitted where={finding.where!r}, "
            f"which is not a document path ({error}). `where` is where the "
            "USER types, not where the code lives."
        ) from error
    if head not in sections:
        raise ConfigError(
            f"{label} check {check!r} emitted where={finding.where!r}, "
            f"whose first segment {head!r} is not a document section. The "
            f"sections are {list(sections)}."
        )
