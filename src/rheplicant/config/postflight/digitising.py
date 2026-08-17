"""C16 -- ADC saturation: what the digitiser clipped, and what that costs a fit.

``ADCOperator`` (``radio/instrument/adc.py``) is ``clip(data * scale, -limit,
limit)`` with ``limit = 2 ** (n_bits - 1)``.  A sample that lands ON the clip
has zero gradient there -- ``jnp.clip`` is flat outside its bounds -- so a
latent whose only route to the prediction passes through a saturated ADC
sample is invisible to a gradient-based fit even though the document loads
and the forward pass returns numbers.  This check measures how much of a run
saturates and, when that risk is real, refuses instead of noting it.

**Ungated (Slot C16).**  ``inference.checks`` knows three names --
``linearity``, ``identifiability``, ``prior_sensitivity`` -- and
``adc_saturation`` is not a fourth.  This check does not read
``payload.gates`` at all; it reaches ``payload.gates`` only because every
:class:`~rheplicant.config.postflight.PostCheck` takes the same
:class:`~rheplicant.config.postflight.Priced` payload.

**The probe.**  A forward pass through the real twin never exposes the
per-sample value the clip saw -- only the clipped output survives.  So this
module runs ONE extra forward pass, through a SUBCLASS of ``ADCOperator``
that records ``abs(data * scale)`` into a closure and then calls
``super().__call__`` -- the rest of the pipeline sees exactly the real ADC's
output, because the subclass still clips.  The subclass inherits
``graph_node = "adc"`` (a ``ClassVar``, resolved through the MRO), so
``Assembly.replace_node`` accepts it exactly where it refuses a bare
``eqx.Module`` -- a plain, non-``AbstractOperator`` class fails
``replace_node``'s own ``validate_operators`` call (``core/pipeline.py``)
with ``PipelineError: replace_node operator 0 is Bare, not an
AbstractOperator`` (measured: ``tests/core/test_assembly_editing.py``'s
``Bare`` fixture, through ``assemble``'s ``owner="assemble"`` spelling of the
same check).  ``_check_slot_kinds`` (``core/fold.py``) -- the guard
``assemble()`` runs at BUILD time, over the whole graph -- never runs here at
all: ``replace_node`` swaps one already-assembled node and does not
re-fold. This is sound only because the call is EAGER: nothing here runs
under ``jit``, so ``state.data`` is a concrete array and the closure
captures a real number rather than a tracer that outlives its trace.

**The escalator is a conjunction, not a second refusal.**  ``fraction > 0``
alone earns at most a WARN once the fraction clears
:data:`_T5_WARN_ABOVE`; it becomes a REFUSE either because the fraction alone
clears :data:`_T5_REFUSE_ABOVE`, or because -- inside the narrow band between
the two thresholds -- a bound latent's ``into:`` reaches a node upstream of
``adc``.  Measured on ``RADIO_GRAPH``: 27 of its 33 nodes are upstream of
``adc``, ``gain`` among them, so "refuse whenever a latent binds upstream of
adc" -- read without the ``fraction > 0`` half of the conjunction -- refuses
a document that never saturates at all.  That reading is exactly what this
module's tests are built to catch; see
``tests/config/test_postflight_digitising.py``.

**Reused rather than restated.**  ``preflight/model.py::_t11_bindings``
already walks both routes a latent reaches a node by (``inference.parameters.
<n>.into`` and ``inference.bindings[].into``), already normalises a string-or
-list ``into:`` and already wraps ``parse_path`` in a ``try/except
ConfigError`` so a malformed path stands the reader down rather than aborting
the whole pass.  This module imports it rather than re-implementing the same
normalisation a second time with its own chance to disagree.

**No optional dependency, and no cycle, at module scope.**  ``RADIO_GRAPH``
and ``ADCOperator`` live under ``rheplicant.radio``, whose package
``__init__`` imports ``radio/rhino.py`` and friends; importing any
``rheplicant.radio`` submodule at THIS module's top level would run that
whole package body in every process that so much as reads a config.
``config/sections/compose.py`` already imports ``RADIO_GRAPH`` inside a
function for the same reason, and this module does the same.

**A known false negative: ``adc.scale: 0.0`` and ``adc.scale: -1.0``.**
Both load cleanly and earn no C16 finding at all (measured).  Taken to its
limit, this check's own advice -- "lower ``adc.scale``" -- produces exactly
this: a scale small enough (zero, or negative) that ``_saturation_stats``
sees no sample reach the clip limit, because the digitiser now discards
signal rather than merely rescaling it.  ``fraction <= 0`` in
:func:`_adc_saturation` cannot distinguish "this document does not
saturate" from "this document digitises nothing usable" -- both read as
silence.  A6/A7/A40 in pre-flight are what would need to refuse a
non-positive ``adc.scale`` on physical grounds; this check does not, and
this paragraph is that gap recorded rather than silently shipped.

**A shipped interaction with C12: any document that lights ``adc`` and
declares ``linear: true`` on a latent bound upstream of it is refused at C12
by default, whether or not it saturates.**  ``linearity`` is the one gate
``gating.DEFAULT_MODE`` leaves at ``refuse``, and ``check_linearity``
(``inference/linear.py``, ``DEFAULT_SCALES = (1e-3, 1.0, 1e3)``) probes each
linear claim at a THOUSAND times the latent's own scale -- where a converter
that never clips at ``1x`` clips hard.  Measured on the most benign ADC this
package can build (``model.adc: {scale: 1.0, n_bits: 12}`` on the base
document, achieved peak 12.116166 ``adc_count`` against a 2048 ``adc_count``
clip limit, so the real forward pass clips NOTHING): C12 still refuses
``inference.parameters.g``, departure 5.32e+00 at ``1000x`` against
``rtol=1.19e-03``, with ``0.001x`` and ``1x`` both exactly 0.  **That refusal
is correct** -- a converter is a deliberate non-linearity and the prediction
really is not affine in ``g`` across the probed range -- so neither check is
weakened for the other.  The consequence is for the AUTHOR of such a
document: it must decline the claim in writing,
``inference.checks.linearity: {mode: skip, reason: "..."}``, which is the
escape C12's own message names.  ``tests/config/preflight_helpers.py``'s
``t5_case``/``T5_LINEARITY_DECLINED`` is this package's own worked example.
The two checks were drafted in parallel and this interaction was invisible
to both; it is recorded here rather than left for the next document to
discover.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from rheplicant.config import findings
from rheplicant.config.findings import Finding
from rheplicant.config.postflight import Priced, register

#: Below this, no clipping happened at all -- silence.  ``0.0`` and not some
#: small positive epsilon: the fraction is exact (a ratio of two integers,
#: the clipped count and ``n``), so there is no float noise to guard against
#: at the bottom edge, only at the boundary between a saturated sample and a
#: nearly-saturated one -- which is :func:`_saturation_stats`'s ``>=``, not
#: this constant.
_T5_WARN_ABOVE: float = 0.0

#: Above this, the fraction alone earns a REFUSE, whatever the twin binds
#: upstream of ``adc``.  AT OR BELOW it (and above :data:`_T5_WARN_ABOVE`),
#: the escalator decides: WARN with no bound latent upstream of ``adc``,
#: REFUSE with one.  The base document's smallest non-zero fraction is
#: ``1/128 == 0.0078125``, almost 8x this value, which is why every base-grid
#: saturating document in this module's tests refuses outright and a
#: dedicated wide grid (``WIDE_GRID``, in the test module) is what makes the
#: WARN band and the escalator's own work observable at all.
_T5_REFUSE_ABOVE: float = 0.001


def _t5_limit(n_bits: int) -> float:
    """The ADC's own clip limit -- ONE place, not two literals kept in step.

    ``ADCOperator.__call__`` (``radio/instrument/adc.py``) computes this same
    expression to build the ``jnp.clip`` bounds.  Rather than pin
    ``2.0 ** (n_bits - 1)`` a second time here and trust the two not to
    drift, ``test_postflight_digitising.py`` asserts this function agrees
    with the OPERATOR's own observed clip point, over ``n_bits`` 1..24, by
    driving a real ``ADCOperator`` with an input far outside its range and
    reading back where the output actually saturates.
    """
    return 2.0 ** (n_bits - 1)


def _t5_target_scale(peak: float, scale: float, limit: float) -> float:
    """The ``adc.scale`` boundary this run's own peak would clear AT.

    ``peak == raw_peak * scale`` (:func:`_saturation_stats`'s ``captured``
    is already scaled), so ``raw_peak == peak / scale`` and the boundary
    where a lowered scale would put the achieved peak exactly on the clip
    limit is ``limit / raw_peak``.  Named a boundary and not a safe value on
    purpose: :func:`_saturation_stats` counts a sample AT the limit as
    saturated (``>=``, not ``>``), so a caller who sets ``adc.scale`` to
    exactly this number reproduces the boundary cell, not silence -- the
    message below says "below", not "at or below".
    """
    return limit / (peak / scale)


def _t5_target_bits(peak: float, n_bits: int) -> int:
    """The smallest ``n_bits`` at or above ``node.n_bits`` whose clip limit
    clears ``peak`` OUTRIGHT (``2 ** (n_bits - 1) > peak``, strictly).

    Walked one bit at a time rather than solved with ``log2`` and rounded:
    :func:`_t5_limit` is the single place this module keeps the clip-limit
    arithmetic, and driving it in a loop is one more use of that same
    function rather than a second, float-rounding-prone route to the same
    number.  ``peak`` is always positive here -- this is only ever called
    from the branch that already found a saturated sample, and a saturated
    sample's scaled magnitude is at least ``limit > 0``.
    """
    target = n_bits
    while _t5_limit(target) <= peak:
        target += 1
    return target


def _saturation_probe(node: Any) -> tuple[Any, list]:
    """A subclass instance of ``node``'s class, plus the list it records into.

    ``node`` is the real ``ADCOperator`` this document's ``model.adc``
    resolved to.  The returned operator carries the SAME ``scale`` and
    ``n_bits`` and still clips -- ``__call__`` ends with
    ``super().__call__(state)`` -- so swapping it in for one forward pass
    changes nothing about what the rest of the twin sees.  What it adds is
    one entry in ``captured``: ``abs(data * scale)``, the value the clip
    acted on, read BEFORE the clip discards it.

    The recording list is a closure and not an ``eqx`` field on purpose --
    an ``eqx.Module`` is a frozen dataclass, so there is no field to
    mutate into, and the whole point is a plain Python list this function's
    caller can read back after the (eager) call returns.
    """
    from rheplicant.radio.instrument.adc import ADCOperator

    captured: list[jax.Array] = []

    class _RecordingADC(ADCOperator):
        def __call__(self, state):
            captured.append(jnp.abs(state.data * self.scale))
            return super().__call__(state)

    return _RecordingADC(scale=node.scale, n_bits=node.n_bits), captured


def _saturation_stats(twin: Any, state: Any, node: Any) -> tuple[float, float, int]:
    """``(peak, fraction, n)`` from ONE forward pass of the FULL twin.

    ``twin`` must be ``payload.run.twin`` -- the raw twin, carrying every
    node ``model:`` lit, ``noise`` included.  ``payload.run.inference.
    fit_twin`` has ``noise`` removed (``inference.twin.without: [noise]``,
    the fit twin's whole reason to differ from the raw one), so probing it
    instead would measure a model nobody's forward pass actually runs and
    would silently under- or over-report how much saturates.

    ``fraction`` counts a sample as saturated at ``>=`` the clip limit, not
    ``>``: the boundary tests in ``test_postflight_digitising.py`` are built
    on a scale that lands a sample at EXACTLY the limit, and a sample sitting
    exactly on the flat part of ``jnp.clip`` has a dead gradient exactly like
    one further out.
    """
    probe, captured = _saturation_probe(node)
    twin.replace_node("adc", probe)(state)
    scaled = captured[0]
    limit = _t5_limit(node.n_bits)
    peak = float(jnp.max(scaled))
    fraction = float(jnp.mean(scaled >= limit))
    return peak, fraction, int(scaled.size)


def _upstream_of_adc() -> frozenset[str]:
    """Every ``RADIO_GRAPH`` node whose signal can reach ``adc``.

    Walked off the graph's PUBLIC ``edges`` tuple -- ``(src, dst)`` pairs --
    rather than its private ``_in`` table, which is ``SignalGraph``'s own
    bookkeeping and not this package's to reach into.  Recomputed on every
    call rather than cached at import time: the whole point of the function
    -scope import above is that ``RADIO_GRAPH`` is not available at module
    load, and 33 nodes / 32 edges is not a cost worth a cache for.
    """
    from rheplicant.radio.graph import RADIO_GRAPH

    predecessors: dict[str, list[str]] = {node: [] for node in RADIO_GRAPH.nodes}
    for src, dst in RADIO_GRAPH.edges:
        predecessors[dst].append(src)
    seen: set[str] = set()
    frontier = [node for node in predecessors.get("adc", ())]
    while frontier:
        current = frontier.pop()
        if current in seen:
            continue
        seen.add(current)
        frontier.extend(predecessors.get(current, ()))
    return frozenset(seen)


def _escalating_latents(document: Mapping[str, Any],
                        upstream: frozenset[str]) -> frozenset[str]:
    """The bound latent NAMES whose ``into:`` reaches a node in ``upstream``.

    Delegates the actual walk -- both ``inference.parameters.<n>.into`` and
    ``inference.bindings[].into``, both spellings of ``into:`` (a string or a
    list of strings), each path wrapped in its own ``parse_path`` try/except
    -- to ``preflight/model.py::_t11_bindings`` rather than re-implementing
    it: that function already returns the normalised HEAD of every ``into:``
    path, which is exactly what "reaches a node" needs and exactly the
    normalisation a second copy here could drift from.
    """
    from rheplicant.config.preflight.model import _t11_bindings

    hit: set[str] = set()
    for _where, names, heads, _transform in _t11_bindings(document):
        if any(head in upstream for head in heads):
            hit.update(names)
    return frozenset(hit)


def _t5_message(node: Any, *, peak: float, fraction: float, n: int,
                latents: frozenset[str]) -> str:
    """The one sentence both severities share -- WARN never names a latent,
    because ``latents`` is only ever non-empty on the branch that already
    refuses (see :func:`_adc_saturation`).

    **The advice names a TARGET, not just the two knobs.**  "Lower
    ``adc.scale`` or raise ``adc.n_bits``" alone tells a reader which two
    fields to touch and not by how much; :func:`_t5_target_scale` and
    :func:`_t5_target_bits` turn THIS run's own achieved peak into the two
    numbers that would actually clear it.  This is not a second refusal
    waiting to happen (STANDING-RULES.md section D): both targets are
    derived from ``peak`` alone, which is already in the message, and
    applying either literally (:func:`_saturation_stats` re-run on the
    result) clears the check -- see
    ``test_advice_lowering_scale_clears_the_refusal`` and
    ``test_advice_raising_n_bits_clears_the_refusal``.
    """
    scale = float(node.scale)
    limit = _t5_limit(node.n_bits)
    body = (
        f"model.adc (adc.scale={scale:.6g}, "
        f"adc.n_bits={node.n_bits}) clipped {fraction:.4%} of {n} samples "
        f"this run; the achieved peak was {peak:.6g} adc_count against a "
        f"clip limit of {limit:.6g} adc_count."
    )
    if latents:
        body += (
            " That clip sits upstream of the bound latent(s) "
            f"{sorted(latents)}, so their gradient is dead wherever a "
            "sample saturates."
        )
    target_scale = _t5_target_scale(peak, scale, limit)
    target_bits = _t5_target_bits(peak, node.n_bits)
    body += (
        f" Lower adc.scale below {target_scale:.6g} or raise adc.n_bits to "
        f"at least {target_bits} so fewer samples saturate (check C16)."
    )
    return body


@register("C16")
def _adc_saturation(payload: Priced) -> Iterable[Finding]:
    """C16: how much of a run the digitiser clipped, and whether a bound
    latent's gradient died in it.

    **Stands down** (yields nothing) when ``model:`` lights no ``adc`` --
    read off the DOCUMENT rather than the twin, so this never has to
    interpret an exception to learn "absent".  ``payload.run.twin["adc"]``
    is still guarded by its own ``except KeyError`` beneath that.  **The
    document read is a FAST PATH, not the only guard**: on every document
    this module's tests build, the document and the twin agree about
    whether ``adc`` is lit, so deleting the document read and relying on the
    ``KeyError`` guard alone leaves every test in this module passing --
    that is why it is not distinguished from the ``KeyError`` guard by its
    own test.  It exists to avoid interpreting an exception (raised from
    inside the fold, several calls removed from this function) to learn
    something the document itself already says directly; the ``KeyError``
    guard is what keeps a document/twin disagreement, should one ever arise,
    from turning into an unhandled exception that ``passes.sweep`` would
    launder into a message blaming this check
    rather than the mismatch.

    A ``adc`` node is ``many=False`` on ``RADIO_GRAPH`` (measured:
    ``radio/graph.py``'s ``NodeSpec`` entry for it never sets ``many=True``,
    and ``NodeSpec.many`` defaults to ``False``), so
    ``Assembly.__getitem__`` can never raise ``AmbiguousNodeError`` for this
    node -- that exception is for a node carrying SEVERAL instances, which
    ``adc`` structurally cannot.  An earlier draft of this check's contract
    asked for a stand-down on that error; it is not reachable here and is
    not written.

    **Also stands down when ``model.adc`` resolved to a COMPOSED node.**  A
    legal, pre-flight-accepted ``model.adc: {compose: cascade, stages: [...]}``
    builds a ``Pipeline`` (``core/pipeline.py``) at ``adc``, not a bare
    ``ADCOperator``, and a ``Pipeline`` carries no ``.scale`` --
    :func:`_saturation_stats` reading ``node.scale`` off one raised
    ``AttributeError`` straight through ``sweep``, which turned into a
    ``ConfigError`` that aborted the ENTIRE post-flight pass and hid every
    other priced check's findings on that document.  The
    ``isinstance(node, ADCOperator)`` guard below is what this check has
    instead: this measures the placeholder ``ADCOperator``'s own gradient
    dead-zone (see this module's docstring), which is not a claim this check
    can make about an arbitrary composed pipeline, so it says nothing rather
    than misreport.  ``_RecordingADC`` (:func:`_saturation_probe`) is an
    ``ADCOperator`` *subclass*, so a plain, uncomposed ``adc`` is unaffected
    by this guard.

    **Does not consult ``payload.gates`` at all** -- C16 is ungated (see
    this module's docstring) -- and does not pre-empt A6/A7/A40, which
    refuse a malformed ``model.adc`` (no ``type:``, a non-numeric
    ``scale:``) in pre-flight, long before this pass runs.
    """
    from rheplicant.radio.instrument.adc import ADCOperator

    model = payload.run.document.get("model")
    if not isinstance(model, Mapping) or "adc" not in model:
        return
    try:
        node = payload.run.twin["adc"]
    except KeyError:
        return
    if not isinstance(node, ADCOperator):
        return

    peak, fraction, n = _saturation_stats(payload.run.twin, payload.run.state,
                                          node)
    if fraction <= _T5_WARN_ABOVE:
        return

    latents = _escalating_latents(payload.run.document, _upstream_of_adc())
    message = _t5_message(node, peak=peak, fraction=fraction, n=n,
                          latents=latents)
    where = "model.adc"
    if fraction > _T5_REFUSE_ABOVE or latents:
        yield findings.refuse("C16", where, message)
    else:
        yield findings.warn("C16", where, message)
