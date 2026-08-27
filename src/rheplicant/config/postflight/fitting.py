"""C12, C13 and C19 -- the three checks a document may be asked to pay for.

Each of the three runs the thing it is deciding about, which is why they live
in this pass and not in ``inflight/``, whose module docstring bans a
``jacfwd``, a ``jacrev``, an SVD or a real forward pass with a static AST
check.

**What each one costs, structurally.**  The load-bearing statement is this
one and not a timing table (plan §0.1's own caution: 3A recorded
``identifiability`` as superlinear and a re-measurement on a toy model made it
flat -- both are true, the growth is in the forward pass rather than in the
SVD at these sizes):

    C12 costs ``len(scales) + 1`` forward passes **per linear latent**.  C13
    costs one ``jacfwd`` plus a dense ``(n_data, n_par)`` SVD, and grows with
    the number of separately bound latents (tracing) and with ``n_data``; it
    does **not** grow with a single latent's size.  C19 costs C13 plus two
    Newton solves.

Re-measured cold on the worked document at ``0030724``: ``load_document``
0.67-0.72 s; ``check_linearity`` 0.163 s (one latent) / 0.307 s (two);
``identifiability`` 0.421 s / 0.542 s; ``prior_sensitivity`` **2.40 s /
3.16 s**, and 0.076-0.099 s warm.  ``prior_sensitivity`` already CONTAINS
``identifiability`` (``sensitivity.py``'s ``_refuse_rank_deficient``), so C19
is not additive with C13 -- it subsumes it.  That is why
``prior_sensitivity`` and ``identifiability`` default to ``off`` and only
``linearity`` defaults to ``refuse``, and why
``tests/config/test_postflight_fitting.py`` lets exactly ONE of its tests pay
the cold C19 cost (``test_C19_reports_the_shift_the_priors_caused``, on the
one-latent base document) while every other drives a counting stub or stands
down before the call.

**TWO dtype predicates, and that is this module's whole reason for existing.**
Measured:

* ``identifiability`` and ``prior_sensitivity`` both call
  ``_check_differentiable`` and refuse **complex** *and* **non-floating**, in
  two distinct sentences -- so :func:`_undifferentiable` gates C13 and C19;
* ``check_linearity`` calls ``_require_inexact`` and refuses **non-floating
  only**; a complex latent is fine by it -- so :func:`_unlinearisable` gates
  C12.

A module that bound one predicate for all three would either lose C12 on a
legitimate complex latent or let a complex latent reach ``identifiability``
and turn an auto-skip into a ``ParameterSpaceError`` from inside the package,
which is the substitution this whole layer exists to prevent.

**WARNING, measured, for anyone driving these predicates on a hand-built
space.**  ``check_linearity`` runs ``_isolate`` BEFORE ``_require_inexact``
(``linear.py:517`` then ``:518``; the ``names=`` branch is the second pair at
``:538`` and ``:539``), and ``_isolate`` validates the space against the
pipeline.  So on a space whose latent binds DIRECTLY into a real leaf, the
refusal a complex or integer latent earns from C12 is a **bind** message --
*"Bind for ('g',) produces complex values for `into` selector 0, but that
leaf is float."* -- which names neither the check nor the gate, and which step
5's ``except ParameterSpaceError`` below re-voices as C12's own failure
sentence.  ``_require_inexact``'s own *"a linear block must be
floating-point or complex"* is unreachable on such a space.

**No config document can build a complex or a non-floating latent today**, and
the C14 machinery here is written for the day one can.  There is exactly one
``Latent(`` construction site under ``config/`` (``sections/parameters.py``)
and a few lines above it every ``init`` is cast to ``context.dtype``, which
``RuntimeFacts.dtype`` restricts to ``float32``/``float64``: a
``dtype: complex64`` init is accepted and silently cast (a ``ComplexWarning``,
not a raise), ``dtype: int32`` is refused by the dtype grammar
(``modifiers.DTYPES`` names four dtypes and no integer), a latent spec takes
no ``dtype:`` key of its own, and ``transform:`` changes the BOUND value and
never ``latent.init``.  The test module carries one end-to-end test that goes
red the day that stops holding, beside four unit tests over a doctored build.

**One shape for all three functions**, and the order matters:

1. read the gate; ``if not gate.runs(): return``;
2. read the subject off the BUILD, never off the document's text;
3. if the subject is empty, stand down -- a document with no latents gives
   C13 nothing and one with no ``linear: true`` gives C12 nothing;
4. if the dtype predicate says the check is undefined here, emit ONE C14 and
   return;
5. call the package function inside ``try``, catching **only**
   ``ParameterSpaceError`` and ``StateValidationError``;
6. compose the sentence and hand it to ``gating.verdict``.

**Step 5 is where a bare ``except Exception`` wants to go.**  It must not: a
``TypeError`` out of a package function is a bug in THIS layer and must not be
re-voiced as a document fault.  ``passes.sweep`` already catches a raising
check and names its slot, so an unexpected exception is reported either way --
with the slot that caused it rather than dressed up as a refusal of a line the
user wrote correctly.

**``rtol`` is ``identifiability``'s alone.**  ``check_linearity`` has an
``rtol=`` too and ``inference.checks.linearity`` cannot express one, so C12
passes none.  ``identifiability``'s is **keyword-only with a default of
1e-08** and ``rtol=None`` raises ``TypeError: unsupported operand type(s) for
*: 'NoneType' and 'float'`` -- so it is passed only when the document wrote
one (:func:`_rtol`), never as ``rtol=gate.rtol``.

**Every C13 and C19 message says ``inference.checks.identifiability`` /
``inference.checks.prior_sensitivity``** and never ``runs[].
check_identifiability``, which is a ``SamplingPlan`` passthrough (four tuples
in ``sections/exits.py``, read by ``inference/plan.py``'s own refusal) and has
nothing to do with these gates.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax.numpy as jnp

from rheplicant.config.findings import Departure, Finding
from rheplicant.config.gating import Gate, auto_skipped, verdict
from rheplicant.config.postflight import Priced, register
from rheplicant.config.sections.noise import decided_noise
from rheplicant.core.errors import (
    LinearityRefused,
    ParameterSpaceError,
    StateValidationError,
)

#: The document path a finding about the whole parameter space points at.  A
#: finding about ONE latent points at ``{_PARAMETERS}.{name}`` instead --
#: :func:`_where` is the single binding of that rule, so the two spellings
#: cannot drift.
_PARAMETERS = "inference.parameters"

#: The two escapes every gated message here names (§3.2 i), phrased so that
#: the same sentence is true of a refusal, a warning and a passing record: a
#: reader of any of the three learns what the check costs to downgrade and
#: what it costs to decline.  **Neither clause names the gate's CURRENT
#: mode** -- "instead of refuse" would be false read against a gate already
#: at ``mode: report`` (a REPORT-severity failure there is not being
#: downgraded FROM a refusal; ``gating.verdict``'s REPORT branch never
#: raises), so the wording below is deliberately mode-agnostic: it is true
#: whether the reader's gate is at ``refuse``, ``warn`` or ``report``.
#: ``mode: skip`` is written WITH its ``reason:`` because check A37 refuses
#: one without, and WITHOUT ``report:`` because ``{mode: skip, report:
#: true}`` is refused by name in pre-flight -- advice that earns a second
#: refusal is the advice loop this project keeps paying for, and
#: ``TestTheAdviceLoop`` applies both of these literally.
_ESCAPE = ("Write {where}: {{mode: warn}} so a failure here warns rather "
           "than blocks the load, or {{mode: skip, reason: \"...\"}} to "
           "decline the check and record why.")

#: The clause C13 and C19 carry and C12 does not: ``linearity`` is the one
#: check on by default (``gating.DEFAULT_MODE``), so only the other two can
#: tell a reader that this DOCUMENT asked for the cost.
_TURNED_ON = ("{name} is off by default and this document turned it on at "
              "{where}. ")


def _tagged(check: str, message: str) -> str:
    """``message`` with ``(check CN).`` APPENDED -- and never twice.

    Several messages this layer ships already end in their own tag, and a
    doubled tail is a defect rather than a cosmetic one: it is what a reader
    sees first.
    """
    tail = f"(check {check})."
    return message if message.endswith(tail) else f"{message} {tail}"


def _where(names: Iterable[str]) -> str:
    """The SUBJECT's document path: one latent's line, or the block.

    Never ``gate.where()`` -- a reader told a check failed needs the line that
    caused it, and the gate is one hop away in the message itself.
    """
    named = tuple(names)
    return f"{_PARAMETERS}.{named[0]}" if len(named) == 1 else _PARAMETERS


def _escape(gate: Gate) -> str:
    """The tail of every C12/C13/C19 message: the gate, and both ways out."""
    lead = ("" if gate.name == "linearity"
            else _TURNED_ON.format(name=gate.name, where=gate.where()))
    return lead + _ESCAPE.format(where=gate.where())


def _dtypes(space: Any, names: Iterable[str] | None) -> dict[str, str]:
    """``{latent: dtype}`` for the selected latents, in declaration order."""
    if space is None:
        return {}
    selected = tuple(space.names) if names is None else tuple(names)
    return {name: str(space.latent(name).init.dtype) for name in selected}


def _undifferentiable(space: Any,
                      names: Iterable[str] | None = None) -> dict[str, str]:
    """``{latent: dtype}`` for every latent C13 and C19 cannot be asked about.

    Mirrors ``identifiability._check_differentiable``: **complex OR
    non-floating**.  A complex block has 2n real degrees of freedom and its
    rank over C is not the number anybody wants; an integer latent is not a
    continuous parameter at all.

    Not the same predicate as :func:`_unlinearisable`, and the difference is
    one word: ``floating`` here, ``inexact`` there.
    """
    return {name: dtype for name, dtype in _dtypes(space, names).items()
            if not jnp.issubdtype(jnp.dtype(dtype), jnp.floating)}


def _unlinearisable(space: Any,
                    names: Iterable[str] | None = None) -> dict[str, str]:
    """``{latent: dtype}`` for every latent C12 cannot be asked about.

    Mirrors ``linear._require_inexact``: **non-floating only**.  A complex
    latent IS a legitimate linear block -- a linear operator over C is still a
    linear operator -- so it is absent from this mapping and present in
    :func:`_undifferentiable`'s.
    """
    return {name: dtype for name, dtype in _dtypes(space, names).items()
            if not jnp.issubdtype(jnp.dtype(dtype), jnp.inexact)}


def _departure(measured: Mapping[str, Mapping[float, float]],
               names: Iterable[str]) -> Departure | None:
    """The measured per-scale departures for ``names``, as C12's own numbers.

    Nested tuples, scales ascending, in ``names``' order -- the shape
    :attr:`~rheplicant.config.findings.Finding.departure` declares, and the
    reason it is tuples is that a ``Finding`` is frozen and hashable.

    A latent with no measurement is ABSENT rather than present-and-empty.
    ``check_linearity`` can refuse for reasons that carry no numbers at all --
    a ``StateValidationError``, or the bind refusal this module's docstring
    warns about, which arrives as a plain ``ParameterSpaceError`` -- and an
    empty row would read as "measured, and found nothing".

    ``None`` when nothing was measured, which is not a table of zeros: a
    latent that really IS affine measures 0.0 at every scale, and that is a
    result rather than the absence of one.

    A NON-FINITE departure is kept.  Measured on the shipped refusing
    document, ``w`` gives ``nan`` at all three scales -- the prediction's own
    arithmetic is unusable there, which ``_affinity_errors`` counts as a
    failure precisely because ``nan > rtol`` is False and a naive comparison
    would read it as evidence of linearity.  Filtering it here would leave a
    table that appears to have probed fewer scales than it did, so whatever
    encodes this downstream needs an answer for a non-finite float, and that
    answer is not zero.

    **The error is NOT coerced with ``float()``, and that is load-bearing.**
    Since D16 axis 5 a departure may arrive as an ``Unresolved`` -- a float
    subclass meaning "the roundoff floor declined to judge this", whose string
    says so.  ``float()`` keeps the value and destroys the marker, so the
    attribute would report a bare number while the message beside it rendered
    ``unresolved:``, and the two would describe the same run differently.
    ``test_C12_carries_its_numbers_on_the_PASSING_branch`` compares them and
    caught exactly that.  The coercion was there for hashability, which a
    float subclass already has, so it was never buying anything.

    This module does not import the inference package (the layer boundary),
    and does not need to: the value passes through, and nothing here asks what
    type it is.
    """
    table = tuple(
        (name, tuple((float(scale), error)
                     for scale, error in sorted(measured[name].items())))
        for name in names if name in measured
    )
    return table or None


def _auto_skip(gate: Gate, blocked: Mapping[str, str],
               reason: str) -> Iterable[Finding]:
    """The ONE C14 a gate that cannot be decided here emits.

    ``gating.auto_skipped`` returns a NEW gate, kept locally: one gates
    mapping is handed to every check in the pass, so writing into it would
    silently change what a later check sees (and is a ``TypeError`` today).
    The C14 REPORT ``verdict`` emits is the ONLY channel by which anything
    else learns this happened.

    **It advises no fix, deliberately.**  Every other message here names its
    escape; a generated auto-skip has none to give -- there is nothing in
    ``inference.checks:`` for the reader to change, and telling them to write
    ``{mode: skip}`` for a check that already did not run is an advice loop
    with a straight face.
    """
    named = ", ".join(f"{name} ({dtype})" for name, dtype in blocked.items())
    message = _tagged("C14", (
        f"{gate.where()} asked for {gate.name} and it is not defined on this "
        f"document: {named}. {reason} The check was skipped automatically and "
        "its reason recorded here; nothing you wrote needs changing."))
    found = verdict(auto_skipped(gate, message), failed=False,
                    where=_where(blocked), message=message)
    if found is not None:
        yield found


def _differentiability_stand_down(gate: Gate,
                                  space: Any) -> Iterable[Finding] | None:
    """C13's and C19's shared step 4, so the two cannot drift apart.

    Returns the C14 findings, or ``None`` when every selected latent carries a
    real derivative and the check may go ahead.

    **Two sentences and not one**, because a complex dtype is ALSO
    non-floating and the package's non-floating message embeds the dtype name:
    a single reason saying "complex or non-floating" would be satisfied by
    either branch and would tell a reader with a ``complex64`` latent to go
    and look for an integer.
    """
    blocked = _undifferentiable(space)
    if not blocked:
        return None
    complexes = {name: dtype for name, dtype in blocked.items()
                 if jnp.issubdtype(jnp.dtype(dtype), jnp.complexfloating)}
    if complexes:
        return _auto_skip(gate, complexes, (
            "The prediction is real, so the map from complex coefficients to "
            "data is R-linear but not C-linear and its rank over C is not the "
            "number this check is about -- declare the real and imaginary "
            "parts as separate latents if you want it asked."))
    return _auto_skip(gate, blocked, (
        "A latent that is not floating-point has no derivative for the "
        "prediction to be taken with respect to, and this check is a "
        "statement about a Jacobian."))


def _rtol(gate: Gate) -> dict[str, float]:
    """``{"rtol": ...}`` when the document wrote one, and ``{}`` otherwise.

    D-12: ``identifiability``'s ``rtol`` is keyword-only with a default of
    ``1e-08``, and ``rtol=None`` raises ``TypeError: unsupported operand
    type(s) for *: 'NoneType' and 'float'`` -- which is deliberately outside
    this module's except set and would therefore escape as ``post-flight check
    'C13' RAISED TypeError`` on **every** default C13 run.
    """
    return {} if gate.rtol is None else {"rtol": gate.rtol}


@register("C12")
def _linearity(payload: Priced) -> Iterable[Finding]:
    """C12 -- ``check_linearity`` once per latent declared ``linear: true``.

    ``name=`` per latent and never a bare ``check_linearity(space, ...)``:
    measured, the bare call raises *"Latent 'a' is not declared linear=True"*
    when only one latent is declared and silently checks ONE of them when
    several are.

    ``fit_twin`` and never ``payload.run.twin``: the twin the likelihood
    predicts with is the one whose linearity is being claimed, and the raw
    twin still carries the stochastic ``noise`` node
    ``inference.twin.without`` removed.
    """
    gate = payload.gates["linearity"]
    if not gate.runs():
        return
    space = payload.run.inference.space
    if space is None:
        return
    claimed = tuple(name for name in space.names if space.latent(name).linear)
    if not claimed:
        return
    blocked = _unlinearisable(space, claimed)
    if blocked:
        yield from _auto_skip(gate, blocked, (
            "A latent that is not floating-point or complex carries no "
            "derivative, so there is no linearization for the prediction to "
            "be compared against."))
        return

    from rheplicant.inference.linear import check_linearity

    # ONE table, written by BOTH outcomes.  ``check_linearity`` RETURNS the
    # per-scale departures when the latent really is affine -- every value
    # 0.0 -- and ``LinearityRefused`` carries the same measurement when it is
    # not.  Until it did, the branch with something to report was the only one
    # with nothing structured to report it with, and the numbers survived
    # solely inside a sentence.
    measured: dict[str, dict[float, float]] = {}
    failures: list[tuple[str, str]] = []
    for name in claimed:
        try:
            measured[name] = check_linearity(space, payload.run.inference.fit_twin,
                                             payload.run.state, name=name)
        except (ParameterSpaceError, StateValidationError) as refused:
            failures.append((name, str(refused)))
            # Only this ONE refusal measured anything.  The others -- a
            # StateValidationError, or the bind refusal this module's
            # docstring warns about -- reach here with no numbers at all, and
            # `_departure` leaves those latents out rather than showing zeros.
            if isinstance(refused, LinearityRefused):
                measured[name] = refused.errors

    if failures:
        refused_names = [name for name, _ in failures]
        detail = " ".join(f"{name}: {sentence}"
                          for name, sentence in failures)
        message = _tagged("C12", (
            f"{_where(refused_names)}: the prediction is not "
            "affine in a latent this document declares linear: true, so the "
            "claim does not hold and every conjugate exit built on it is "
            f"solving the wrong problem. check_linearity refuses it in its "
            f"own words -- {detail} {_escape(gate)}"))
        found = verdict(gate, failed=True,
                        where=_where(refused_names),
                        message=message,
                        departure=_departure(measured, refused_names))
        if found is not None:
            yield found
        return

    detail = "; ".join(
        f"{name}: " + ", ".join(f"{scale:g}x -> {error:.2e}"
                                for scale, error in sorted(errors.items()))
        for name, errors in measured.items())
    message = _tagged("C12", (
        f"{_where(claimed)}: check_linearity holds for every latent this "
        f"document declares linear: true -- relative departure from each "
        f"one's own linearization at {detail}. {_escape(gate)}"))
    found = verdict(gate, failed=False, where=_where(claimed), message=message,
                    departure=_departure(measured, claimed))
    if found is not None:
        yield found


@register("C13")
def _identifiability(payload: Priced) -> Iterable[Finding]:
    """C13 -- the joint Jacobian's rank, over the WHOLE space at once.

    Jointly and not block by block: the answer is routinely *yes* for every
    block of a partition whose joint model is degenerate, which is the whole
    reason ``identifiability`` takes the joint by default.
    """
    gate = payload.gates["identifiability"]
    if not gate.runs():
        return
    space = payload.run.inference.space
    if space is None:
        return
    stood_down = _differentiability_stand_down(gate, space)
    if stood_down is not None:
        yield from stood_down
        return

    from rheplicant.inference.identifiability import identifiability

    subject = _where(space.names)
    try:
        report = identifiability(space, payload.run.inference.fit_twin,
                                 payload.run.state, **_rtol(gate))
    except (ParameterSpaceError, StateValidationError) as refused:
        message = _tagged("C13", (
            f"{subject}: identifiability could not be decided for this "
            f"document. The package refuses it in its own words: {refused} "
            f"{_escape(gate)}"))
        found = verdict(gate, failed=True, where=subject, message=message)
        if found is not None:
            yield found
        return

    numbers = (f"rank {report.rank} of {report.n_par} parameters over "
               f"{report.n_data} data points, nullity {report.nullity}, at "
               f"rtol {report.rtol:.0e}")
    if report.nullity:
        shares = ", ".join(
            f"{name} {share:.2f}"
            for name, share in sorted(report.participation(0).items(),
                                      key=lambda item: -item[1]))
        message = _tagged("C13", (
            f"{subject}: the data cannot see {report.nullity} direction(s) of "
            f"this parameter space -- {numbers}. The worst one is carried by "
            f"{shares}, so those latents trade off against each other and a "
            "fit will return a finite, correctly-shaped answer in which they "
            f"are not separately determined. {_escape(gate)}"))
    else:
        message = _tagged("C13", (
            f"{subject}: identifiability finds {numbers} -- weakest "
            f"identified direction {report.weakest_identified:.3e} of the "
            f"strongest. {_escape(gate)}"))
    found = verdict(gate, failed=bool(report.nullity), where=subject,
                    message=message)
    if found is not None:
        yield found


@register("C19")
def _prior_sensitivity(payload: Priced) -> Iterable[Finding]:
    """C19 -- how far the declared priors moved the mode, in posterior sigmas.

    **Two stand-downs of its own**, and neither is C13's.  ``observed`` is
    ``None`` on a document with no ``inference:`` section at all and its
    ``primary`` can be ``None`` beside real entries, and ``decided_noise``
    returns ``None`` for ``kind: none`` -- which ``as_noise_model`` inside the
    package turns into a ``TypeError``, a bug in this layer wearing a
    document's clothes.

    The threshold is the package's own :data:`CRITERION_SHIFT` (0.1 sigma),
    imported rather than restated: 0.1 sigma moves a 68 % interval's endpoints
    by a tenth of its half-width, which shifts a reported central value
    without visibly changing the error bar, and a second constant here would
    be a second thing to retune.
    """
    gate = payload.gates["prior_sensitivity"]
    if not gate.runs():
        return
    inference = payload.run.inference
    space = inference.space
    if space is None:
        return
    observed = inference.observed
    if observed is None or observed.primary is None:
        return
    noise = decided_noise(inference.noise)
    if noise is None:
        return
    stood_down = _differentiability_stand_down(gate, space)
    if stood_down is not None:
        yield from stood_down
        return

    from rheplicant.inference.sensitivity import (
        CRITERION_SHIFT,
        prior_sensitivity,
    )

    subject = _where(space.names)
    try:
        report = prior_sensitivity(space, inference.fit_twin, payload.run.state,
                                   observed.entries[observed.primary], noise)
    except (ParameterSpaceError, StateValidationError) as refused:
        message = _tagged("C19", (
            f"{subject}: prior sensitivity could not be decided for this "
            f"document. The package refuses it in its own words: {refused} "
            f"{_escape(gate)}"))
        found = verdict(gate, failed=True, where=subject, message=message)
        if found is not None:
            yield found
        return

    name, index, shift = report.worst
    criterion = float(
        report.for_latent(name)["criterion_std"].ravel()[index])
    failed = abs(shift) >= CRITERION_SHIFT
    numbers = (f"the declared priors move the mode of {name}[{index}] by "
               f"{shift:+.3e} posterior sigma, the largest of this space's "
               f"{report.n_par} parameter(s), against the "
               f"{CRITERION_SHIFT:.3e} sigma at which a shift is large enough "
               "to change a published central value without visibly widening "
               "its error bar")
    tail = (f"That element reaches {CRITERION_SHIFT:g} sigma at a prior width "
            f"of {criterion:.3e}; compare it with the prior this document "
            "declares.")
    message = _tagged("C19", (
        f"{subject}: {numbers}. {tail} {_escape(gate)}"))
    found = verdict(gate, failed=failed, where=subject, message=message)
    if found is not None:
        yield found
