"""Pre-flight: the beam and projector keys, decided before an entry is built.

Four rows, two functions, one subject: ``resources.beams`` and
``resources.projectors`` are the only two kinds whose builders read a file and
run a spherical harmonic transform, and **``build_resources`` has no
validation pre-pass**.  Measured at ``ea4839b``: a document whose projector
omits ``normalize_beam`` and whose beam is a path that does not exist dies on
the missing file, 0.106 s in, saying nothing about the key -- and on a
two-beam document the first beam is read, cast and normalised before the
second is asked about its ``normalize:``.  That is the class of failure this
module moves, and it is the plan's largest single saving.

**What is saved, stated narrowly** (the plan's own instruction not to claim a
saving nobody measured).  A11 and A12's beam half already fire before
``_maps_for`` on a SINGLE-beam document -- ``build_beam`` asks its two
questions first -- so for that shape the gain is only that the answer arrives
with every other text finding rather than alone.  The gain that is real is
A12's projector half, A44 and A48, each of which sits behind the referenced
beam's whole read-normalise-truncate cycle today, and any document declaring
more than one entry, where the entries before the bad one are built in full.

**Every message here is a HOIST and none of them is written in this module.**
Each of the five refusals stayed in the section that owns it, refactored into
a module-level pure function taking plain data
(``kinds/beams.py::_a12_normalize``, ``::_a11_chart_keys``,
``kinds/projectors.py::_a12_normalize_beam``, ``::_a44_float32_sky``,
``::_a48_lst_ref``); this module imports those functions and turns their
answers into findings, and the builders keep calling them as their own second
opinion.  ``tests/config/message_binding.py::assert_bound_once`` is what says
so mechanically, and ``test_preflight_resources.py`` parametrizes it over
these five literals.

**``extends:`` is why nothing here reads the raw section.**  The text of an
entry is not the spec its builder sees: ``config/resources.py`` resolves
``extends:`` before a kind is called, so a pre-pass reading
``document["resources"]["beams"]["a"]`` raw refuses an entry whose
``normalize:`` came from its parent and misses one whose parent's
``phi0_deg`` was deleted.  :func:`~rheplicant.config.resources.resolved_specs`
is the layer's own reader and it never raises -- a malformed entry is DROPPED
and ``build_resources`` stays the backstop, so an entry that is missing from
its mapping is one this pass **stands down on**.

**And it is called once per LAYER.**  ``resolved_specs`` takes a section, not
a document, and the walk belongs to the caller:
``preflight/document.py::_task3_over_layers`` is what makes a ``variants:``
copy of a bad beam earn the same sentence, prefixed with the variant that
introduced it.  A check reading ``document["resources"]`` closes one route and
leaves that twin wide open, which is why 3A's ``A1.horizon`` already walks
this very section by layer.

**Where this module does NOT look, and why it is not a false negative.**
``inference.twin.replace.<node>`` reaches ``build_node_operator`` and is the
route ``preflight/model.py::_nodes`` cannot see (§0.3 E.10).  It replaces an
OPERATOR, not a resource: there is no ``resources:`` block under
``inference.twin`` and no spelling of one, so a beam or projector entry can
only ever arrive through ``resources:`` on some layer, which is exactly what
is walked above.  ``test_preflight_resources.py::
TestTheRouteThisModuleDoesNotWalk`` is the assertion, not this sentence.

**Ordering.**  Each function emits its findings in the order the builder asks
its questions -- A12 before A11 on a beam, A12 before A44 before A48 on a
projector -- so the first refusal a user reads is the one ``build_resources``
would have said, one phase earlier.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from _rheplicant_bootstrap.path_syntax import longest_legal_prefix
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.kinds.beams import (
    BEAM_FORMATS,
    _a11_chart_keys,
    _a12_normalize,
)
from rheplicant.config.kinds.projectors import (
    _NOT_WRITABLE,
    ENGINES,
    OPTIMIZATIONS,
    _a12_normalize_beam,
    _a44_float32_sky,
    _a48_lst_ref,
)
from rheplicant.config.preflight import register
from rheplicant.config.resources import resolved_specs
from rheplicant.config.sections.runtime import RuntimeFacts

#: The dotted prefixes :func:`~rheplicant.config.resources.resolved_specs`
#: keys by.  It answers ``{"resources.<kind>.<name>": spec}`` for EVERY kind
#: the document declares, so a reader that wants one kind selects on the
#: prefix rather than reaching back into the section.
_B2_BEAMS = "resources.beams."
_B2_PROJECTORS = "resources.projectors."

#: The engines whose branch of ``build_projector`` reaches the shared prelude
#: A12's projector half and A44 live in.  **Derived from**
#: :data:`~rheplicant.config.kinds.projectors.ENGINES` rather than restated:
#: ``engine: matrix`` returns before all three of this module's projector rows
#: (a ``MatrixProjector`` reads no beam and has exactly one field), and a
#: hand-written pair here would be a second source of truth for a list that is
#: one line away.
_B2_SKY_ENGINES: tuple[str, ...] = tuple(
    engine for engine in ENGINES if engine != "matrix")


def _b2_dtype(layer: Mapping[str, Any]) -> str | None:
    """This layer's traced dtype, from ``runtime.jax_enable_x64``, or ``None``.

    ``None`` means "the text does not say", and every caller stands down on
    it: ``runtime:`` not being a mapping and ``jax_enable_x64`` not being a
    bool are both refusals ``build_runtime`` writes in its own words, and a
    dtype guessed past either of them would pre-empt the better sentence.

    :class:`~rheplicant.config.sections.runtime.RuntimeFacts` is CONSTRUCTED
    with three placeholder fields so that its own ``dtype`` property answers,
    rather than the ternary being written a second time here.  The property
    reads ``jax_enable_x64`` and nothing else (measured), so the placeholders
    cannot reach the answer; what they buy is that the day the mapping from
    the flag to the dtype changes, it changes in one place.
    ``test_preflight_resources.py::test_the_dtype_this_pass_reads_is_the_one_
    build_runtime_reports`` is the guard on that claim.
    """
    runtime = layer.get("runtime")
    if not isinstance(runtime, Mapping):
        return None
    x64 = runtime.get("jax_enable_x64", False)
    if not isinstance(x64, bool):
        return None
    return RuntimeFacts(jax_enable_x64=x64, platform="auto", seed=None,
                        seeds={}).dtype


def _b2_beams_in(layer: Mapping[str, Any]) -> Iterable[Finding]:
    """A12's beam half and A11, over one layer's ``resources.beams``."""
    for dotted, spec in resolved_specs(layer.get("resources")).items():
        if not dotted.startswith(_B2_BEAMS):
            continue
        # The format-value refusal is `build_beam`'s first question and is not
        # one of this task's rows. Standing down on an unknown format is what
        # keeps A11 from reaching its raw-array `else` on `format: nonsense`
        # and answering "frame is required" to a document whose fault is the
        # format itself.
        if spec.get("format") not in BEAM_FORMATS:
            continue
        where = longest_legal_prefix(dotted)
        problem = _a12_normalize(dotted, spec)
        if problem is not None:
            yield refuse("A12", where, problem)
        problem = _a11_chart_keys(dotted, spec)
        if problem is not None:
            yield refuse("A11", where, problem)


def _b2_projectors_in(layer: Mapping[str, Any]) -> Iterable[Finding]:
    """A12's projector half, A44 and A48, over one layer's projectors."""
    dtype = _b2_dtype(layer)
    for dotted, spec in resolved_specs(layer.get("resources")).items():
        if not dotted.startswith(_B2_PROJECTORS):
            continue
        engine = spec.get("engine")
        # `engine: matrix` and an engine that is not an engine at all are both
        # stand-downs, and for opposite reasons: the matrix branch RETURNS
        # before all three rows below, and an unknown engine earns
        # `build_projector`'s own first refusal, which names the three that
        # exist.
        if engine not in _B2_SKY_ENGINES:
            continue
        # `beam_frame`/`beam_ref_lst_deg` are refused ahead of every row here,
        # by a message that names the optimisation to ask for instead. A48 in
        # particular would otherwise pre-empt it on a document writing both.
        if set(_NOT_WRITABLE) & set(spec):
            continue
        where = longest_legal_prefix(dotted)
        problem = _a12_normalize_beam(dotted, spec)
        if problem is not None:
            yield refuse("A12", where, problem)
        if dtype is not None:
            problem = _a44_float32_sky(dotted, spec, dtype)
            if problem is not None:
                yield refuse("A44", where, problem)
        if engine == "driftscan" and _b2_a48_is_this_entrys_first_fault(spec):
            problem = _a48_lst_ref(dotted, spec)
            if problem is not None:
                yield refuse("A48", where, problem)


def _b2_a48_is_this_entrys_first_fault(spec: Mapping[str, Any]) -> bool:
    """Is A48 the first thing ``build_projector`` would say about this entry?

    Two things run ahead of A48 there and both say something more specific:
    the ``read_horizon_fraction`` redirect (that token PRODUCES f_sky and is
    not an optimisation) and the unknown-optimisation refusal, which lists the
    ones that exist.  Neither is one of this task's rows, so A48 stands down
    whenever ``optimizations`` holds anything ``OPTIMIZATIONS`` does not.

    A non-list ``optimizations:`` stands down too.  ``build_projector`` runs
    ``list(spec.get("optimizations") or [])`` over it, so a string is read
    character by character and a mapping key by key -- and the refusal that
    lands is the unknown-optimisation one, about ``'c'`` or about a key.  This
    pass has nothing better to say about either.
    """
    declared = spec.get("optimizations")
    if declared is None:
        return True
    if not isinstance(declared, list):
        return False
    return all(entry in OPTIMIZATIONS for entry in declared)


@register("A11", "A12.beam")
def _beam_keys(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Checks A11 and A12: what a ``resources.beams`` entry must and must not say.

    * **A12** -- ``normalize:`` is written AND is one of
      :data:`~rheplicant.config.kinds.beams.BEAM_NORMALIZATIONS`.  A
      membership test: ``normalize: pixelsum`` is present and is not a
      convention, and the pair that decides the output's unit is undecided
      either way.
    * **A11** -- ``phi0_deg``/``phi_sense`` for ``format: cst`` and refused
      otherwise, ``frame:`` for every raw-array format, and the ``uvbeam``
      special case that refuses ``frame`` as well as the other two.

    Registered under both ids **variadically**, in one call, because one walk
    answers both and ``sweep`` de-duplicates by function identity.  The dotted
    slot ``A12.beam`` is what leaves ``A12.projector`` free for
    :func:`_projector_keys` -- two functions may not both claim bare ``A12``,
    and every ``Finding`` either of them emits carries the bare id anyway.
    """
    return _b2_beams_in(document)


@register("A12.projector", "A44", "A48")
def _projector_keys(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Checks A12, A44 and A48: the three keys a projector entry owes.

    * **A12** -- ``normalize_beam`` is written; it has no default and the
      output's unit is the pair (it, the beam's own ``normalize:``).
    * **A44** -- a real sky engine in float32 with no
      ``acknowledge_float32_sky: true``.  Its condition is
      ``runtime.jax_enable_x64``, which is ABSENT by default, so this row
      fires on any unacknowledged driftscan or general_pointing entry in an
      ordinary document rather than on an exotic one.
    * **A48** -- ``optimizations: [cache_beam_rotation]`` with no
      ``lst_ref_deg``.  **On ``engine: driftscan`` only.**  The shared prelude
      in ``build_projector`` asks it of general_pointing too, but that engine
      takes neither key: applying A48's advice there earns ``engine:
      general_pointing does not take ['lst_ref_deg', 'optimizations']``
      (measured), so a P-1 A48 on that engine would be advice a reader cannot
      follow.  The narrower gate is deliberate and is the one place this pass
      is quieter than the builder it hoists from.

    ``engine: matrix`` is exempt from all three -- the matrix branch returns
    before every one of them.
    """
    return _b2_projectors_in(document)
