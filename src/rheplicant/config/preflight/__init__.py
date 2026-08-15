"""Every check decidable from the document's text, run before the money.

**Why a pass and not a check.**  Schema §6's premise is that its checks run
"before any beam is analysed".  Measured at ``be2027b``, on a document
carrying an unreadable beam alongside one violation each, the beam's refusal
won for an unknown ``model:`` node (A2), a junction given an operator (A3), a
``flagging`` with no ``type:`` (A7), a ``NeuralOperator`` type (A39) and a
``scope: per_epoch`` latent (A39) -- five for five, every one of them a
complete ``ConfigError`` with a good message, every one of them arriving after
the file it had nothing to do with.  The only two refusals that won were
``document._sweep``'s own, because ``_sweep`` was the one thing running before
``build_resources``.  This module is where the rest of them go.

**Collect, do not raise.**  A check RETURNS an iterable of
:class:`~rheplicant.config.findings.Finding` and raises nothing, so a user with
four errors sees four.  A check that raises would abort the pass and hide every
later finding, so :func:`preflight` catches that and says which check did it
rather than letting the pass truncate in silence.

**Except structurally.**  Section names, ``schema_version``, the required
sections and the deferred ones must hold before any other check can assume the
document's shape, so :func:`_structural` still raises immediately, and every
registered check may assume the sections it reads are present or absent --
never malformed at the top level.  ``_structural`` IS ``document._sweep``,
moved here whole: its five messages are unchanged to the character, because
``tests/config/test_config_document.py`` pins them.

**Three sources and no fourth** (§2.4): the document mapping, ``RADIO_GRAPH``,
and operator classes resolved by name.  Resolving a class is an attribute
lookup; constructing one is not in scope, no file is read, no value node is
resolved, no twin is built.  ``tests/config/test_config_preflight.py`` enforces
that mechanically and in five ways: ten filesystem APIs taken away;
``equinox``'s metaclass ``__call__`` taken away; ``resolve_value`` and the
five builders taken away in every module that holds a reference to one; and
two static bans -- on importing any of them and on writing any of their calls
-- which are what see a branch no test document takes.  Each has an
anti-vacuity partner, and what the five together cannot see is written down
beside them.

**No module in this package may import an optional dependency at module
scope** -- ``numpyro``, ``limtod_jax``, ``healpy``, ``h5py``, ``pyuvdata``,
``rhino_cal_jax``.  ``document.py`` imports this package and
``rheplicant.config`` imports ``document``, so a pre-flight pass that dragged
one in would put it in every process that so much as reads a config.  Asserted
in a subprocess by the same test module.

**The direction of the dependency** (§2.2): this package sits ABOVE
``sections/`` and calls into it.  ``sections/`` never imports ``preflight/``,
and no module here may import ``rheplicant.config.document`` -- that one closes
a cycle, since ``document`` imports this package's ``preflight`` at its head.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, Report
from rheplicant.config.paths import parse_path

__all__ = ["CHECKS", "preflight", "register"]

#: One check: a document in, findings out, nothing raised.
Check = Callable[[Mapping[str, Any]], Iterable[Finding]]

#: A registry SLOT: a schema §6 id, optionally with a dotted suffix.  Named
#: for this task rather than for a check, per §3.1's rule about a private
#: helper that is not check-shaped: no later task writes this file, so the
#: prefix costs nothing and keeps the convention unbroken.
_T2_SLOT = re.compile(r"[ABC][1-9][0-9]*(\.[a-z_]+)?")

# --- what `document._sweep` knew, moved with it -----------------------------
# These three were `config/document.py:31-40` and are read by `_structural`
# alone.  Measured before the move: nothing outside `document.py` imported any
# of them (`grep -rn "_SECTIONS\|_NOT_YET\|_REQUIRED" src/ tests/` -> only
# document.py), so the move is closed.  They cannot be imported back from
# `document.py` either: that module imports this one for the hook.
_SECTIONS = ("schema_version", "defaults", "plugins", "runtime", "observation",
             "resources", "model", "variants", "inference", "runs", "outputs",
             "campaign")
_NOT_YET = {
    "outputs": "Plan 4 (outputs, provenance, the CLI)",
    "defaults": "Plan 4 (presets are YAML files, and the CLI is where YAML "
                "first comes off disk)",
    "plugins": "Plan 4 (plugin import belongs to the process entry point)",
}
_REQUIRED = ("runtime", "observation", "model", "runs")

#: id -> the function.  **Insertion order IS run order**, which is what
#: §2.6's "A20 and A21 before A23" rests on: the first refusal is the one a
#: user reads, and A23's would contradict A20's if it came first.
CHECKS: dict[str, Check] = {}


def register(*checks: str) -> Callable[[Check], Check]:
    """Bind one or more check ids to one function.

    Raises ``ConfigError`` -- never an ``assert``, which ``python -O`` strips.
    That is the ledger item Task 1 closed at ``sections/exit_support.py:31``,
    where the strip was measured to let the second registration win in
    silence; this registry is written the same way from the start.

    A SLOT is a schema §6 id, optionally with a dotted suffix (``"A1.runs"``)
    when several functions each decide part of one check (§3.2 (a)).
    ``Finding.check`` stays the bare id.

    VARIADIC, not stacked.  ``@register("A20") @register("A21")`` applies
    BOTTOM-UP, so the stacked form inserts A23 first and ``CHECKS`` insertion
    order -- which IS run order, and which §2.6 item 4's "A20 and A21 before
    A23" rests on -- comes out reversed.

    Every id is validated before any is bound, and every id is checked for a
    clash before any is bound: a registration that failed half way through
    would leave a module's ids partly claimed, after which the next import of
    that module reports "registered twice" about a module that never finished.

    Raises:
        ConfigError: no id at all; an id that is not a string (the
            ``@register`` written without its parentheses); an id that is not
            a schema §6 slot; one id named twice in one call; or an id another
            function already claims.
    """
    if not checks:
        raise ConfigError(
            "register() takes one or more check ids -- @register('A30'), or "
            "@register('A16', 'A17') when one function decides several. A "
            "registration with no id binds nothing, so the check it decorates "
            "never runs and nothing says so."
        )
    for check in checks:
        if not isinstance(check, str):
            raise ConfigError(
                f"pre-flight check id {check!r} is not a string. @register is "
                "called with its ids -- @register('A30') -- and a bare "
                "@register hands the decorated function in as an id."
            )
        if not _T2_SLOT.fullmatch(check):
            raise ConfigError(
                f"pre-flight check id {check!r} is not a schema §6 id "
                "(A1..A52, B1..B9, C1..C17), optionally with a dotted "
                "suffix such as 'A1.runs' when several functions each "
                "decide part of one check. The id is what a Finding "
                "carries and what a reader looks up; a private name here "
                "reaches the user as '(check _mine).'"
            )
    for check in checks:
        if checks.count(check) > 1:
            raise ConfigError(
                f"this registration names {check!r} twice. The variadic form "
                "binds one function to several DIFFERENT ids -- "
                "@register('A16', 'A17') -- and a repeated id is a typo for "
                "one that is now claimed by nobody."
            )

    def bind(fn: Check) -> Check:
        for check in checks:
            if check in CHECKS:
                raise ConfigError(
                    f"pre-flight check {check!r} is registered twice, by "
                    f"{CHECKS[check].__module__} and by {fn.__module__}. A "
                    "check id has one function, and which of the two would "
                    "run depends on import order."
                )
        for check in checks:
            CHECKS[check] = fn
        return fn

    return bind


def _structural(document: Mapping[str, Any]) -> None:
    """Section names, ``schema_version``, required and deferred sections.

    RAISES, and runs before :data:`CHECKS`, so every registered check may
    assume the document's top level is well formed.  **This is
    ``config/document.py``'s ``_sweep``, moved** -- every message is byte for
    byte what it was, because ``tests/config/test_config_document.py`` pins
    five of them by ``match=`` and
    ``tests/config/test_config_preflight.py`` pins all of them by equality.
    """
    unknown = sorted(set(document) - set(_SECTIONS))
    if unknown:
        raise ConfigError(
            f"This document declares {unknown}; the sections are "
            f"{list(_SECTIONS)}."
        )
    if "campaign" in document:
        raise ConfigError(
            "campaign: is reserved with capability 4 (streaming evidence, "
            "schema §8.2) and refused in v1."
        )
    for section, route in _NOT_YET.items():
        if section in document:
            raise ConfigError(
                f"{section}: is not read by this layer yet -- it arrives with "
                f"{route}."
            )
    version = document.get("schema_version")
    if version != 1 or isinstance(version, bool):
        raise ConfigError(
            f"schema_version: 1 is required (got {version!r}); it is what "
            "lets a later loader read an older document on purpose rather "
            "than by luck."
        )
    missing = [section for section in _REQUIRED if section not in document]
    if missing:
        raise ConfigError(
            f"This document is missing {missing}; schema_version, runtime, "
            "observation, model and runs are required."
        )


def preflight(document: Mapping[str, Any]) -> Report:
    """Every check decidable from the document's text.

    Never reads a file, never builds an operator, never resolves a value node.
    Structural problems raise; everything else is collected, in registration
    order, and handed back for the caller to raise or emit.
    """
    _structural(document)
    findings: list[Finding] = []
    seen: set[int] = set()
    for check, fn in CHECKS.items():
        # ONE FUNCTION CARRIES SEVERAL IDS (§3.1) and five tasks do it:
        # `_blocks` is A16-A19, `_prior_gates` is A20/A21/A23, `_counts` is
        # A24+A25, `_decided` is A27+A28.  Iterating `CHECKS` naively calls
        # `_blocks` FOUR times and emits every finding four times.
        #
        # By `id`, not by name: two module-level checks can share a
        # `__name__` (every lambda is `<lambda>`), and a walk keyed on the
        # name would drop the second in silence.  `id` cannot be recycled
        # here because `CHECKS` holds a reference to every function for the
        # whole loop.
        if id(fn) in seen:
            continue
        seen.add(id(fn))
        try:
            found = tuple(fn(document))
        except Exception as error:
            raise ConfigError(
                f"pre-flight check {check!r} RAISED "
                f"{type(error).__name__}: {error}. A check returns findings "
                "and raises nothing -- one that raises aborts the pass and "
                "hides every finding after it, which is the failure the "
                "collect-rather-than-raise design exists to prevent."
            ) from error
        for finding in found:
            _check_where(check, finding)
            findings.append(finding)
    return Report(findings=tuple(findings))


def _check_where(check: str, finding: Finding) -> None:
    """``Finding.where`` is a path into the USER'S document, never into ``src/``.

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
            f"pre-flight check {check!r} emitted where={finding.where!r}, "
            f"which is not a document path ({error}). `where` is where the "
            "USER types, not where the code lives."
        ) from error
    if head not in _SECTIONS:
        raise ConfigError(
            f"pre-flight check {check!r} emitted where={finding.where!r}, "
            f"whose first segment {head!r} is not a document section. The "
            f"sections are {list(_SECTIONS)}."
        )


# Importing the check modules is what registers their ids.  The import sits at
# the foot rather than the head because they import this module's `register`;
# `sections/exits.py` closes the same loop the same way.  The completeness
# test in tests/config/test_config_preflight.py is what keeps it from rotting
# -- a module here that nobody imports registers nothing and stays green.
#
# Tasks 3-12 each add ONE ALIASED line of their own here and NOTHING else in
# this file:
#
#     from rheplicant.config.preflight import model as _model_checks  # noqa: F401,E402
#
# Aliased because `_structural` and `preflight` both take a parameter named
# `document`, so a foot import binding the bare name gives F811 at both `def`
# lines.
#
# `document` registers A1.runs, A1.variants, A1.horizon, A38 and A39.
# `model` registers A2 (which decides A2, A3, A4, A6 and A7), A32 and
# A14.cal_loads.

from rheplicant.config.preflight import document as _document_checks  # noqa: E402,F401
from rheplicant.config.preflight import fitting as _fitting_checks  # noqa: E402,F401
from rheplicant.config.preflight import model as _model_checks  # noqa: E402,F401
from rheplicant.config.preflight import observing as _observing_checks  # noqa: E402,F401
