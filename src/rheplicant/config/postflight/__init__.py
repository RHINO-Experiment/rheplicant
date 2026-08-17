"""The pass whose checks cost a forward pass, a Jacobian or a Newton solve.

``preflight/`` decides everything a document's TEXT decides and ``inflight/``
adds the two slots that need the resolved grids and the assembled twin.  A
third kind of check is decidable by neither: the ones that have to **run** the
thing -- ``check_linearity``'s ``len(scales) + 1`` forward passes per linear
claim, ``identifiability``'s ``jacfwd`` plus a dense SVD,
``prior_sensitivity``'s two Newton solves on top of that.  This package is
where those go.

**Why a separate package and not a third registry inside ``inflight/``.**
``inflight/``'s module docstring declares a package-level invariant --
*"``jax.eval_shape`` is permitted; evaluating the twin is not.  A ``jacfwd``, a
``jacrev``, an SVD or a real forward pass belongs to Plan 3C"* -- and
``tests/config/test_config_inflight.py::TestTheInFlightBoundary`` enforces it
with a static AST ban over ``inflight/*.py``.  These checks cross that line by
design.  Weakening a shipped guard to admit the code it was written to exclude
is the anti-pattern this project keeps recording, so the code moves instead.

**Why a third registry and not ``BUILT_CHECKS``.**  It is the same TYPE
argument ``inflight/__init__.py`` already makes for itself: a priced check
needs the resolved gates and a built check does not, so one registry would let
a check be bound in a slot whose payload it cannot read, and the symptom would
be an ``AttributeError`` wrapped by :func:`~rheplicant.config.passes.sweep`
into "post-flight check 'C12' RAISED AttributeError" -- a stack trace shaped
like a user's fault.  Corroborating rather than decisive: the built pass is
bounded at ``best_ms < 0.02`` ms while ``check_linearity`` warm is 4.7-6.5 ms
on the worked document, so a priced check in ``BUILT_CHECKS`` turns that bound
into a lie by a factor of a few hundred.

**The runner is shared.**  :mod:`rheplicant.config.passes` holds the binder,
the sweep and the ``where`` guard, so this pass and the three before it have
one de-duplication, one raise-guard and one path guard between them rather
than four that drift.  What this module supplies is the registry, the payload
type, the two words its refusals say, and the discovery that replaces a
foot-import list.

**ONE argument, because ``sweep`` calls ``fn(payload)``.**  The gates travel
ON the payload rather than beside it (:class:`Priced`), so :func:`priced`
mirrors ``axes(facts)`` and ``built(run)`` and a check is
``def _linearity(run)`` reading ``run.gates``.  A two-argument check would die
as ``TypeError: _linearity() missing 1 required positional argument: 'gates'``
laundered by ``sweep``'s raise-guard into a sentence blaming the check author
-- which is exactly the failure the registry split exists to prevent.

**Run order is ``sorted(CHECKS)``, and that is not decoration.**  ``sweep``
iterates INSERTION order, and under discovery insertion order is the import
graph's rather than the filename's: measured, a ``digitising`` that
head-imports ``fitting`` gives ``['C12','C13','C19','C16','C18']`` while no
sibling import gives ``['C16','C12','C13','C19','C18']``.  This repository has
already paid for that once -- commit ``8bcf74d`` deleted a shipped ordering
assertion after measuring ``beam_spill`` sorting first and registering fourth.
The dict is sorted at the call site, which keeps the property and adds no
second runner.

**TRAP -- ``expect: refuse`` cannot catch a refusal from this pass.**
``expect:`` is per RUN (``sections/runs.py``, consumed by
``sections/exits.py::execute_run``) and this pass raises out of
``load_document``, before any run is executed.  A document whose only run
carries ``expect: refuse`` and whose ``linearity`` gate refuses gets a
``ConfigError`` out of ``load_document``, not a captured refusal.  **That is
correct rather than a limitation**: a gate is a property of the DOCUMENT, and
a run cannot expect a refusal of the document that configures it.
``tests/config/test_config_postflight.py::test_expect_refuse_cannot_catch_a_priced_refusal``
is the pin, and it spies on the executor so the claim is "never reached"
rather than "returned no product".

**Scope.**  A module here MAY evaluate the twin, take a Jacobian and run an
SVD -- that is the whole point -- and may NOT import
``rheplicant.config.document``, which imports this package for the hook and
would close a cycle.  It must not import an optional dependency at module
scope either: ``rheplicant.config`` reaches this package through
``document.py``, so a module-scope ``numpyro`` here would land in every
process that so much as reads a config.
"""

from __future__ import annotations

import dataclasses
import importlib
import pkgutil
from collections.abc import Callable, Iterable, Mapping
from types import MappingProxyType
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, Report
from rheplicant.config.gating import Gate
from rheplicant.config.passes import Registry, binder, sweep
from rheplicant.config.preflight import _SECTIONS

__all__ = ["CHECKS", "PostCheck", "Priced", "priced", "register"]

#: The phase word every refusal ABOUT A CHECK opens with, and the name a
#: reader is told to write.  Both are arguments to ``passes.py``'s runner
#: rather than constants inside it, because four passes share that runner and
#: differ in exactly these two words.  Pinned by equality on whole messages in
#: ``test_config_postflight.py`` -- a substring pin beginning after the word
#: cannot see either, measured when ``passes.py`` was extracted.
_LABEL = "post-flight"
_DECORATOR = "register"

#: The document's top-level section names, borrowed from ``preflight`` rather
#: than restated.  ``Finding.where`` is a path into the USER'S document at
#: every slot -- this one included -- so all four passes validate it against
#: the same list.
#:
#: **``_check_where`` is deliberately NOT imported beside it.**
#: ``preflight._check_where`` is bound with ``_LABEL = "pre-flight"`` already
#: closed over, so a post-flight pass calling it would report its own defect
#: as a pre-flight one.  ``passes.sweep`` does the guarding with this pass's
#: own label instead, so nothing but the section names needs importing.
_DOCUMENT_SECTIONS = _SECTIONS

#: Module stems under this package that would SHADOW one of its own names.
#: Importing a submodule sets it as an attribute of its package, so a
#: ``postflight/priced.py`` would make ``from ... import priced`` bind a
#: module and ``document.py``'s hook raise "'module' object is not callable".
#: ``inflight/__init__.py`` records exactly that measurement about its own
#: ``axes`` and can only defend it with a comment about where its import block
#: sits; discovery can refuse it outright.
_RESERVED = frozenset({"gates", "priced", "register", "run"})


@dataclasses.dataclass(frozen=True, slots=True)
class Priced:
    """What a check that costs money is handed: the built run, and the gates.

    **Two fields, not ten.**  An earlier draft of this contract spelled it
    ``Priced(*run, gates)`` -- ``ConfiguredRun``'s seven fields plus the
    gates -- which puts that field list in a second place somebody has to keep
    in step by hand.  :class:`~rheplicant.config.inflight.Built` pays that
    price for a reason (it is spliced positionally out of a ``ConfiguredRun``
    and a test pins the two tuples equal); this payload is built from a whole
    ``ConfiguredRun`` and has nothing to mirror.
    """

    #: The ``ConfiguredRun``, carrying **the report earned by every earlier
    #: pass** -- pre-flight, then axes, then built, in run order.  A priced
    #: check can therefore read what an earlier check already said and need
    #: not restate it.
    #:
    #: Typed ``Any`` and not ``ConfiguredRun``: ``document.py`` imports this
    #: package for the hook, so a real annotation would close the cycle.
    run: Any
    #: The three gates, defaults applied, from
    #: :func:`~rheplicant.config.gating.gates` -- **read-only, and a COPY**.
    #: One mapping is handed to every check in the pass, so a check that
    #: assigned into it would silently change what a later check sees; and a
    #: proxy over the CALLER's dict would leave the same hole open from the
    #: other side.  ``gating.gates()``'s own return type is unchanged; the
    #: wrapping happens here, at the one place a payload is built.
    gates: Mapping[str, Gate]

    def __post_init__(self) -> None:
        # Wrapped in `__post_init__` rather than at the call site so that
        # there is no way to construct a `Priced` whose gates a check can
        # write into -- including the ones Tasks 4-6 build in their own test
        # helpers, which is where the sloppy write would otherwise arrive.
        if not isinstance(self.gates, Mapping):
            # A bug in the CALLER, never the document's fault, so a
            # ValueError and not a ConfigError: a caller who wrapped
            # `load_document` in `except ConfigError` would otherwise swallow
            # it and report the document as at fault.  (`ConfigError` IS a
            # `ValueError`, so the test asserts `not isinstance(...,
            # ConfigError)` rather than the exception type alone.)
            raise ValueError(
                "Priced.gates is the resolved gates -- gating.gates(...)'s "
                "mapping of check name to Gate, cardinality three whatever "
                f"the document says; got {self.gates!r}."
            )
        object.__setattr__(self, "gates",
                           MappingProxyType(dict(self.gates)))


#: One post-flight check: a :class:`Priced` in, findings out, nothing raised.
#: ONE argument -- see this module's docstring for the ``TypeError`` the
#: two-argument spelling produces and for how ``sweep`` launders it.
PostCheck = Callable[[Priced], Iterable[Finding]]

#: Slot -> the function.  **Run order is ``sorted(CHECKS)``**, not insertion
#: order; :func:`priced` sorts at the call site and this module's docstring
#: records the measurement that makes that necessary.
CHECKS: Registry = {}


def register(*checks: str) -> Callable[[PostCheck], PostCheck]:
    """Bind one or more check ids to one function, in :data:`CHECKS`.

    The body is :func:`~rheplicant.config.passes.binder`; its five refusals,
    its validate-all-before-binding-any order and its variadic-not-stacked
    contract are documented there.  The decorated function takes a
    :class:`Priced`.

    **A SECOND function and deliberately not an import of
    ``preflight.register``.**  Two registries that can receive each other's
    checks fail silently -- 3A measured that exact shadowing against
    ``sections/exit_support.py``'s ``register(kind)``.  The consequence worth
    knowing is that this one SHARES ITS NAME with ``preflight.register``, so a
    module here that imports the wrong one binds its check into the text pass,
    where the payload is a document and ``run.gates`` is an
    ``AttributeError``.  ``test_config_postflight.py``'s
    ``test_every_module_under_postflight_contributes_a_slot`` is what catches
    that in production, because such a module owns no slot here.
    """
    return binder(CHECKS, *checks, label=_LABEL, decorator=_DECORATOR)


def _discoverable(path: Iterable[str]) -> tuple[str, ...]:
    """Every module stem under ``path``, sorted -- or a refusal for a shadow.

    Split out from the loop below so that the refusal is reachable from a
    test: the real package holds no reserved name, so a check written inline
    at import time could never be driven.

    Raises:
        ConfigError: a module here is named after one of this package's own
            public names.  See :data:`_RESERVED`.
    """
    found = tuple(sorted(name for _, name, _ in pkgutil.iter_modules(path)))
    clash = [name for name in found if name in _RESERVED]
    if clash:
        raise ConfigError(
            f"postflight/ holds {clash}, and importing a submodule SETS IT as "
            "an attribute of its package -- so `from "
            "rheplicant.config.postflight import priced` would bind a MODULE "
            "and the hook in document.py would raise \"'module' object is not "
            "callable\". Rename the module; the reserved names are "
            f"{sorted(_RESERVED)}."
        )
    return found


# Importing the check modules is what registers their ids -- and unlike
# `preflight/` and `inflight/`, there is NO LIST here.  This package
# DISCOVERS its modules, for one reason: Plan 3C's Tasks 4, 5 and 6 run in
# parallel and would each add a line to a shared foot-import list, which is
# three conflicts on one line; and "create the modules early and fill them
# later" is not available either, because a docstring-only stub owns no slot
# and turns the contributes-a-slot test red.
#
# Discovery cannot have 3A's failure -- there is no list to be incomplete.
# It CAN have the opposite one, and that is written down in
# `test_config_postflight.py::TestTheDiscoveryMechanism`'s docstring rather
# than left to be found: a module dropped in here is imported whether or not
# anybody meant it.
#
# WHICH MODULE CLAIMS WHICH SLOT IS NOT WRITTEN HERE.  `sorted(CHECKS)` is the
# answer and it cannot go stale.
for _found in _discoverable(__path__):
    importlib.import_module(f"{__name__}.{_found}")


def priced(run: Priced) -> Report:
    """Every check that has to run the thing it is deciding about.

    Runs after ``build_inference`` and before ``load_document`` returns, so
    one payload carries the raw twin, the fit twin, the space and the observed
    data -- and so no caller ever holds a ``ConfiguredRun`` whose priced
    checks have not run.  **It saves nothing**: the beam is long since read,
    and schema §6's preamble ("all run before any beam is analysed") is false
    about every check registered here.  What it buys is that the checks run at
    all, in this layer's voice, instead of detonating later inside a fit.

    Structural problems have already been raised by ``preflight`` and the two
    in-flight passes have already refused what they can, so every check here
    may assume the document's top level is well formed and its grids and its
    twin are built.

    ``dict(sorted(...))`` and not a second runner: see this module's docstring
    for why insertion order is the import graph's here.
    """
    return sweep(dict(sorted(CHECKS.items())), run, label=_LABEL,
                 sections=_DOCUMENT_SECTIONS)
