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

from collections.abc import Callable, Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, Report
from rheplicant.config.passes import binder, check_where, sweep

__all__ = ["CHECKS", "preflight", "register"]

#: One check: a document in, findings out, nothing raised.
Check = Callable[[Mapping[str, Any]], Iterable[Finding]]

#: The phase word every refusal ABOUT A CHECK opens with, and the name a
#: reader is told to write.  Both are arguments to ``passes.py``'s runner now
#: rather than constants inside it, because three passes share that runner and
#: differ in exactly these two words.  **They are pinned by equality on whole
#: messages** in ``test_config_preflight.py::
#: TestEveryRefusalOfThisPassIsPinnedWHOLE`` -- measured, every pin that
#: existed before was a substring beginning after the word, and rewriting all
#: six occurrences left ``tests/config`` at exit 0.
_LABEL = "pre-flight"
_DECORATOR = "register"

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
    """Bind one or more check ids to one function, in :data:`CHECKS`.

    The body is :func:`~rheplicant.config.passes.binder`, which is this
    function as Plan 3A wrote it with the registry and the two words passed
    in.  Its five refusals, its validate-all-before-binding-any order and its
    variadic-not-stacked contract are unchanged and documented there; what
    stays here is the binding of THIS pass's registry and THIS pass's words,
    so that every message it says is byte-identical to 3A's.
    """
    return binder(CHECKS, *checks, label=_LABEL, decorator=_DECORATOR)


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
    return sweep(CHECKS, document, label=_LABEL, sections=_SECTIONS)


def _check_where(check: str, finding: Finding) -> None:
    """This pass's ``where`` guard: :func:`~rheplicant.config.passes.check_where`
    with this pass's label and section names already bound.

    **Kept as a name, with its Plan 3A two-argument signature**, for the same
    reason :func:`register` is: the IMPLEMENTATION moved to ``passes.py`` and
    lives in exactly one place, while the spelling a caller uses stays put.
    It is not a second copy -- there is no message here, only a binding of the
    two words this pass says.

    Measured when the extraction landed: ``tests/config/test_preflight_values.py``
    calls this directly, in a 3-way hostile-document product that drives every
    check's ``where`` through the guard OUTSIDE the pass.  Deleting the name
    took that module to a collection ``ImportError`` -- pytest exit **2**, not
    1 -- which is the shape that reads as "the suite is broken" rather than as
    "a test failed".
    """
    check_where(_LABEL, check, finding, _SECTIONS)


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
# WHICH MODULE CLAIMS WHICH SLOT IS NOT WRITTEN HERE, AND THAT IS THE POINT.
# `sorted(CHECKS)` is the answer and it cannot go stale; a list in this
# comment can, and did.  An earlier version said `model` "registers A2 (which
# decides A2, A3, A4, A6 and A7)" -- false, and consequentially so: A2, A3,
# A4, A6 and A7 are five registry SLOTS bound to one function variadically
# (`model.py:201`), which is what stops a later function claiming one of
# them.  Plan 3A's Task 13 predicted 29 slots and 27 bare ids by counting
# from that sentence, against a measured 34 and 31.  Count from `CHECKS`:
#
#     {slot.split(".")[0] for slot in CHECKS}   # the bare schema §6 ids
#
# A dotted slot (`A1.runs`) is several functions deciding parts of one check
# (§3.2 (a)); `Finding.check` is always the bare id.

from rheplicant.config.preflight import beam_spill as _beam_spill_checks  # noqa: E402,F401
from rheplicant.config.preflight import document as _document_checks  # noqa: E402,F401
from rheplicant.config.preflight import fitting as _fitting_checks  # noqa: E402,F401
from rheplicant.config.preflight import model as _model_checks  # noqa: E402,F401
from rheplicant.config.preflight import observing as _observing_checks  # noqa: E402,F401
from rheplicant.config.preflight import resources as _resources_checks  # noqa: E402,F401
from rheplicant.config.preflight import values as _values_checks  # noqa: E402,F401
