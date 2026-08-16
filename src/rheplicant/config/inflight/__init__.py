"""The two passes that run inside ``load_document`` rather than in front of it.

``preflight/`` decides everything a document's TEXT decides, before anything is
built.  Two kinds of check cannot be decided there, and until this package they
were decided nowhere useful:

* **the axes pass (P-0.5)** runs immediately after ``build_observation`` and
  **before ``build_resources``**, so it sees the resolved time and frequency
  grids, the switch order and the materialised pointing -- and it sees them
  before the money.  ``build_resources`` is **90.9 %** of ``load_document``'s
  wall time (1.397 s of 1.536 s, toy nside-16 beam); a slot that costs a
  hundredth of that and can refuse a bad time axis is worth having.  A refusal
  that today costs a beam read and a spherical harmonic transform costs a
  fraction of a millisecond here.
* **the built pass (P-1.5)** runs when ``load_document`` is ready to return,
  after ``build_inference``, so one payload carries the raw twin, the fit twin
  and the space.  **It saves nothing**, and saying so is the point: schema §6's
  preamble ("all run before any beam is analysed") is FALSE for the
  twin-shaped rows, and a task that writes that sentence about a built-slot
  check is repeating the mistake this package's position exists to record.
  What it buys is that the checks run at all, in this layer's voice, against
  the assembled objects, instead of detonating later inside a fit.

**Two registries, not one, and that is a type argument.**  A check bound in
the axes slot is handed an :class:`Axes` and a check bound in the built slot is
handed a :class:`Built`.  One registry would let a check be registered in the
slot whose payload it cannot read, and the symptom would be an
``AttributeError`` wrapped by :func:`~rheplicant.config.passes.sweep` into
"in-flight check 'C8' RAISED AttributeError" -- a stack trace shaped like a
user's fault.

**The runner is shared.**  :mod:`rheplicant.config.passes` holds the binder,
the sweep and the ``where`` guard, so these two passes and ``preflight`` have
one de-duplication, one raise-guard and one path guard between them rather
than three that drift.  The only things this module supplies are the two
registries, the two payload types and the two words those refusals open with.

**Each slot raises before it warns, and an earlier slot's warning may already
have been shown when a later slot refuses.**  ``document.py`` argues that a
document about to be refused should not also spray warnings, and across slots
that cannot be honoured -- the axes pass has already returned before
``build_resources`` runs.  This is correct rather than tolerated: an axes
warning is about a line the built refusal does not touch, and suppressing it
would mean holding every finding until the end, which is the round-trip the
collect-rather-than-raise design exists to remove.

**Scope.**  A module here MAY hold a built object -- that is the whole point --
and may NOT read a file, evaluate the twin, or take a Jacobian.
``jax.eval_shape`` is permitted; evaluating the twin is not.  A ``jacfwd``, a
``jacrev``, an SVD or a real forward pass belongs to Plan 3C.
``tests/config/test_config_inflight.py`` walks this package and
``config/passes.py`` together for exactly that, and writes down what the walk
cannot see.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Mapping
from typing import Any

from rheplicant.config.findings import Report
from rheplicant.config.passes import Check, Registry, binder, sweep
from rheplicant.config.preflight import _SECTIONS

__all__ = ["AXIS_CHECKS", "BUILT_CHECKS", "Axes", "Built", "axes", "built",
           "register_axes", "register_built"]

#: The phase word both passes' refusals-about-a-check open with.  ONE word for
#: two passes, deliberately: a reader who sees it wants to know that the
#: complaint is about a check rather than about their document, and which of
#: the two in-flight slots it came from is what the ``decorator`` below says.
_LABEL = "in-flight"

#: The document's top-level section names, borrowed from ``preflight`` rather
#: than restated.  ``Finding.where`` is a path into the USER'S document at
#: every slot -- the built one included -- so all three passes validate it
#: against the same list, and a second copy here would be the
#: ``_number``-vs-``_whole`` divergence in its purest form.  The import is
#: one-directional: ``preflight`` never imports this package, and
#: ``document.py`` already imports ``preflight`` at its head, so nothing is
#: dragged in that a configured run does not already pay for.
_DOCUMENT_SECTIONS = _SECTIONS


@dataclasses.dataclass(frozen=True, slots=True)
class Axes:
    """What exists after ``build_observation`` and before ``build_resources``.

    **No resource is built**, which is what makes an axes check cost
    microseconds: the beam is still unread when these run.
    """

    #: The VARIANT-APPLIED mapping, so a check reports a path the reader can
    #: actually go and edit.
    document: Mapping[str, Any]
    #: ``RuntimeFacts`` -- the seed and the dtype.
    runtime: Any
    #: ``ObservationBuild`` -- ``time_s``, ``freq_hz``, site, switch order,
    #: pointing and ``extra``.
    observation: Any
    #: ``ResolutionContext`` with ``freq`` and ``time`` RESOLVED and
    #: ``resources`` still empty.  ``context.shape_scope`` is the layer's own
    #: reader for ``n_time``, ``n_freq`` and ``n_source``.
    context: Any


@dataclasses.dataclass(frozen=True, slots=True)
class Built:
    """What exists when ``load_document`` is ready to return.

    **The field names and their order are** :class:`ConfiguredRun`'s, exactly,
    so that ``Built(*run)`` is the whole constructor and no test keeps two
    field lists in step by hand.
    ``test_config_inflight.py::test_the_two_field_tuples_are_equal`` pins the
    two tuples; the day ``ConfiguredRun`` grows a field, that test names it
    rather than a positional argument landing in the wrong slot in silence.
    ``ConfiguredRun`` is a ``NamedTuple``, so the pin reads ``._fields`` --
    ``dataclasses.fields`` raises on it.
    """

    #: The VARIANT-APPLIED mapping, as in :class:`Axes`.
    document: Mapping[str, Any]
    runtime: Any
    state: Any
    #: The RAW twin -- ``model:`` as declared, before ``inference.twin``'s
    #: ``without:``/``replace:`` are applied.
    twin: Any
    #: ``InferenceBuild``.  **Never ``None``** -- ``build_inference`` returns
    #: one for a document with no ``inference:`` section at all.  Its
    #: ``.space`` MAY be ``None``, and a check that reads it stands down when
    #: it is rather than refusing on "I could not tell".
    inference: Any
    #: ``BuiltResources``.
    resources: Any
    #: ``ResolutionContext``, now carrying the built resources and the ingest.
    context: Any


#: Slot -> the function, for the axes pass.  **Insertion order IS run order**,
#: and ``raise_if_refused`` shows the first refusal verbatim.
AXIS_CHECKS: Registry = {}

#: Slot -> the function, for the built pass.  SEPARATE from
#: :data:`AXIS_CHECKS`: two payload types, two registries.
BUILT_CHECKS: Registry = {}


def register_axes(*checks: str) -> Callable[[Check], Check]:
    """Bind one or more check ids to one function, in :data:`AXIS_CHECKS`.

    The body is :func:`~rheplicant.config.passes.binder`; its five refusals,
    its validate-all-before-binding-any order and its variadic-not-stacked
    contract are documented there.  The decorated function takes an
    :class:`Axes`.
    """
    return binder(AXIS_CHECKS, *checks, label=_LABEL,
                  decorator="register_axes")


def register_built(*checks: str) -> Callable[[Check], Check]:
    """Bind one or more check ids to one function, in :data:`BUILT_CHECKS`.

    As :func:`register_axes`, for the later slot.  The decorated function
    takes a :class:`Built`.
    """
    return binder(BUILT_CHECKS, *checks, label=_LABEL,
                  decorator="register_built")


# Importing the check modules is what registers their ids, exactly as
# `preflight/__init__.py`'s foot does -- and unlike that one, this import CANNOT
# sit at the foot.  **Measured, and it is silent both ways.**  This package
# defines `axes` as its entry point AND holds a module called `axes.py`:
#
# * `from rheplicant.config.inflight import axes as _axis_checks` placed BELOW
#   `def axes` binds the FUNCTION and never imports the submodule at all --
#   `_handle_fromlist` finds the attribute already there and stops -- so
#   `inflight/axes.py` would never run and its checks would never register,
#   with nothing going red.
# * `import rheplicant.config.inflight.axes` placed below it imports the
#   submodule and then SETS IT as the package's `axes` attribute, shadowing
#   the entry point -- after which `document.py`'s
#   `from rheplicant.config.inflight import ... axes ...` binds a module and
#   the axes hook raises "'module' object is not callable".
#
# Placed HERE, above the two `def`s, the attribute does not exist yet, so the
# submodule really is imported and really is registered; the `def axes` below
# then rebinds the package attribute to the entry point, while
# `sys.modules['rheplicant.config.inflight.axes']` keeps the module for anyone
# who imports it by path.  A later `import rheplicant.config.inflight.axes`
# does NOT re-set the attribute (measured: the parent attribute is written
# once, at first load), so the entry point stays callable.
#
# A task adding a module here inserts ONE alphabetically-placed line and
# nothing else.  WHICH MODULE CLAIMS WHICH SLOT IS NOT WRITTEN HERE:
# `sorted(AXIS_CHECKS)` is the answer and it cannot go stale.
from rheplicant.config.inflight import axes as _axis_checks  # noqa: E402,F401
from rheplicant.config.inflight import grids as _grid_checks  # noqa: E402,F401


def axes(facts: Axes) -> Report:
    """Every check decidable from the resolved grids and nothing built.

    Runs no builder and reads no file.  Structural problems have already been
    raised by ``preflight``, so every check here may assume the document's top
    level is well formed.
    """
    return sweep(AXIS_CHECKS, facts, label=_LABEL,
                 sections=_DOCUMENT_SECTIONS)


def built(run: Built) -> Report:
    """Every check that needs the assembled objects, run before the fit.

    ``jax.eval_shape`` is permitted here; evaluating the twin is not.
    """
    return sweep(BUILT_CHECKS, run, label=_LABEL,
                 sections=_DOCUMENT_SECTIONS)
