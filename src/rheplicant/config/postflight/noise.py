"""C18, the numeric half: the sigma that drew the data and the sigma that
weighs it, made to agree.

D-C17, the handover's own words: *"a run can carry two sigmas... nothing in
the package keeps them equal, and if they drift the fit is weighted by a
number that did not generate the data -- finite, correctly shaped, wrong, and
invisible to every diagnostic. The config layer is the one thing that can
close that."* :class:`~rheplicant.radio.instrument.noise.RadiometerNoiseOperator`'s
own class docstring says the duty travels with the class and names Plan 3.
Measured before this task: every disagreement was accepted, including a
factor of ten.

**The two-word half and the numeric half are two different slots sharing one
bare id.** ``preflight/gated.py::_sigma_families`` claims the registry slot
``C18.kind`` and decides whatever is legible from the document's TEXT alone --
``model.noise.type`` against ``inference.noise.kind``, as two spellings of one
family or two different noise models entirely. This module claims the bare
slot ``C18`` and decides the other half: given a document whose two sides
agree on WHICH noise model, do they also agree on the NUMBER? That needs the
built twin and the built likelihood, so it cannot run before the beam is
read -- which is why it lives here rather than beside its sibling.

**Why this check does not also referee a family mismatch.** A drawn class
and a weighed kind that name two different noise models entirely are already
:func:`~rheplicant.config.preflight.gated._sigma_families`'s subject, and a
REFUSE there halts the pipeline at ``load_document``'s very first pass -- this
module never runs on such a document at all. Where the two tables disagree
about what counts as "the same family" (today they do not), this check still
declines: it answers one question -- do the two sides of a MATCHING family
agree on the number -- and a mismatched pairing that reaches it is a
different check's subject, not a gap in coverage this one should quietly
widen to fill.

**One case is NOT covered by that stand-down, and is recorded here rather
than fixed -- owed to Task 7.** A ``python:`` target relocated onto a model
key other than ``noise`` can still bind to the GRAPH NODE ``noise``.
``_sigma_families`` reads the document's TEXT by KEY (``model.noise.type``)
and cannot see such a document at all, so its REFUSE never fires on it --
but :func:`_t6_drawn` reads the BUILT twin by node id and does see it. This
module then runs on a document its sibling never touched, and its message
still says ``model.noise`` even though the document's own text never wrote
that key -- the same wrong-``where`` shape BLOCKER 3 closed for
``inference.twin.replace:``. This is a real gap and not a "this check
declines" case; it is recorded here because no document in this suite
constructs a relocated ``python:`` target to measure it against.

**Stand-down, in the order it is decided (contract step order, and it
matters -- reading it any other way changes which reason a user is told):**

1. The document's PRIMARY observation is not ``from: simulation`` at all (an
   ingested or file-form observation carries no second sigma to disagree
   with), or there is no single primary to read (:func:`_t6_generating_twin`).
2. The twin that actually drew that primary's data -- the FULL twin when
   ``observed.twin`` is ``full`` (the shipped default), the FIT twin when it
   is ``fit`` -- does not itself carry the ``noise`` node: either because it
   was never lit at all, or because it was genuinely repaired out of the fit
   twin (``inference.twin.without: [noise]`` or an equivalent ``replace:``).
   Indexing the twin by node id answers this directly (:func:`_t6_drawn`) --
   see the function's own docstring for why an ``AmbiguousNodeError`` and a
   bare ``AssemblyError`` both fold into the same stand-down.
3. ``inference.noise:`` was never declared, or was declared ``kind: none``.
   Task 2's ``C18.kind`` already warns on this document when it is a fitting
   run; a second sentence about the same absence is not this check's to add.

   **This is also why a WARN from ``C18.kind`` and a REFUSE from this
   check's bare ``C18`` can never co-fire on one document.** ``C18.kind``'s
   only WARN row (``preflight/gated.py::_sigma_families``) fires exactly
   when ``inference.noise`` is undeclared or ``kind: none`` and the run is a
   fitting one -- precisely this stand-down's own condition -- and
   ``C18.kind`` calls ``refuse()``/``warn()`` directly rather than through a
   gate's ``verdict()``, so nothing downstream can promote or demote either
   into the other. A document with ``weighed.kind == "none"`` cannot also
   earn this check's REFUSE: the line just below says so in words, and
   (measured) the two family guards further down -- ``weighed.kind not in
   _T6_RADIOMETER_KINDS`` and ``weighed.kind != "homoscedastic"``, neither
   of which contains ``"none"`` -- hold the same line on their own even if
   this one is deleted. Three lines carry one invariant between them; this
   is the one written in words, and the one this task's own mutation
   testing named.
4. The drawn class and the weighed kind belong to two different noise model
   families. See above.

**Why the comparison reads ``.fractional`` and ``.sigma`` off the objects
themselves, and never a formula.** ``grep -rn "channel_width \\* .*integration_
time" src/`` finds ``1 / sqrt(channel_width * integration_time)`` already
written three times -- in
:meth:`~rheplicant.radio.instrument.noise.RadiometerNoiseOperator.fractional`,
:meth:`~rheplicant.inference.noise.RadiometerNoise.fractional`, and
``config/sections/noise.py``'s :func:`~rheplicant.config.sections.noise.freeze_sigma`.
A fourth copy in a validator is a validator that agrees with a formula rather
than with the code -- if either shipped property is ever wrong, a hand-rolled
copy here would still say the document is fine. For ``radiometer_frozen``,
whose ``NoiseBuild.frozen`` dict carries the facts but not an object, the
facts are handed BACK to the likelihood's own class
(:class:`~rheplicant.inference.noise.RadiometerNoise`) so the SAME property
answers, rather than re-deriving the formula a fourth time.

**Why the compared quantity is the fractional scatter and not the two
fields.** ``(channel_width: 1 MHz, integration_time: 2 s)`` on the operator
and ``(channel_width: 2 s, integration_time: 1 MHz)`` on the likelihood --
values SWAPPED, units kept -- give the identical fractional scatter
``f = 0.0007071067811865475``, and that document is correct: it is one
physical statement written in two harmless orders, because multiplication
commutes. A field-by-field comparison would refuse it for a difference that
does not exist. ``f`` is the number that actually reaches the residuals'
weights; ``f`` is what must agree.

**``floor`` does not participate in the comparison.**
:class:`~rheplicant.inference.noise.RadiometerNoise` takes a ``floor`` and
:class:`~rheplicant.radio.instrument.noise.RadiometerNoiseOperator` does not,
deliberately: ``RadiometerNoise.realise`` does not apply it either -- grep
``deliberately not applied here`` in ``inference/noise.py`` -- because a floor
is "a remedy for a reweighting iterate crossing zero, and a generator has no
iterate." A document that declares one is not disagreeing with itself, and
the refusal message says so explicitly whenever one is declared, so a reader
is not sent looking for a difference this check chose to ignore.

**Known scope edges, recorded here rather than widened into -- owed to
Task 7.**

* A multi-record ``inference.observed:`` whose PRIMARY is ``from: file``
  while a SIBLING record is ``from: simulation`` stands this check down
  entirely: :func:`_t6_generating_twin` reads only ``observed.primary``, and
  a primary that did not come from the twin is contract step 1's stand-down
  regardless of what a sibling record did. ``preflight/gated.py::
  _t2c_generated`` reads the same way (primary alone), so this is a scope
  the two C18 slots share, not a divergence between them.
* **BLOCKER 2 (Plan 3C fix round), recorded rather than widened -- owed to
  Plan 4.** A multi-record ``inference.observed:`` with TWO OR MORE named
  records and NONE of them literally named ``primary`` resolves no primary at
  all -- ``ObservedBuild.primary`` is ``None`` -- and :func:`_t6_generating_twin`
  returns ``None`` on it exactly as it does on a document with no ``observed:``
  section. **Both C18 bindings stand down together**: this numeric check and
  ``preflight/gated.py::_t2c_generated`` (the family check, ``C18.kind``) read
  ``observed.primary`` the same way, so the two-vantage-point agreement this
  design relies on is not broken by this gap -- both are silent, together,
  precisely because neither has a primary to read. Measured:

  .. code-block:: text

      single-record (``primary``)                REFUSED
      named ``'primary'``                         REFUSED
      named ``'alpha'`` ALONE (only record)        REFUSED
      named ``'alpha'`` + ``'beta'`` (no ``'primary'``)  LOADS, report=[]

  The trigger is a record NAME, not a document property -- the fourth row can
  carry the same tenfold sigma disagreement measured elsewhere in this module
  and load clean, because the primary-resolution rule this check and its
  sibling both depend on has nothing to resolve. **Decision: record it, do
  not widen the readers** -- widening C18 to per-record is a contract change
  that needs its own adversarial pass, which this fix round does not have.
  See ``preflight/gated.py::_t2c_generated``'s own docstring for the matching
  entry and the plan's §6 residues for the standing decision record.
* :attr:`~rheplicant.config.sections.noise.NoiseBuild.by_observation` (the
  per-observation frozen sigma ``radiometer_frozen`` with
  ``source: observed`` produces, one entry per named observation) is never
  read here -- this check reads ``weighed.frozen``, which
  :func:`~rheplicant.config.sections.noise.freeze_sigmas` documents as
  staying the PRIMARY's own facts. That is the same primary
  :func:`_t6_generating_twin` reads, so the two stay aligned on a
  multi-record document even though this module never opens
  ``by_observation`` itself.
* **MAJOR 2 (Plan 3C fix round), recorded rather than widened -- owed to
  Task 7.** A composed ``model.noise: {compose: cascade, stages: [...]}``
  resolves to a ``Pipeline`` (``core/pipeline.py``) at the ``noise`` node, not
  a bare :class:`~rheplicant.radio.instrument.noise.NoiseOperator` or
  :class:`~rheplicant.radio.instrument.noise.RadiometerNoiseOperator`. Neither
  ``isinstance`` guard in :func:`_t6_sigma_agreement` matches a ``Pipeline``,
  so this check stands down entirely and silently on a composed draw, exactly
  as ``postflight/digitising.py`` documents at length for a composed
  ``model.adc`` (see that module's docstring, "Also stands down when
  ``model.adc`` resolved to a COMPOSED node") -- a ``Pipeline`` carries no
  ``.sigma``/``.fractional`` for either check to read, and answering here
  would be a claim about an arbitrary composed pipeline this check cannot
  make. **This is not a false negative on a document that happens to be
  correct.** Measured: a cascade of two ``NoiseOperator(sigma=0.05 K)``
  stages draws this document's data at an effective sigma of ``~0.075 K``
  (``std(drawn - noiseless) == 0.07492221891880035``) while a single, flat
  ``inference.noise: {kind: homoscedastic, sigma: 0.05 K}`` weighs it at
  ``0.05 K`` -- a **41% mis-weighting**, invisible to every diagnostic, on a
  document ``load_document`` accepts clean. Composing the twin's noise stages
  and composing the likelihood's weight are two independent authoring
  actions with no shared grammar tying them together, so there is no single
  number this check could compare without first deciding what "the sigma of
  a composed noise pipeline" even means -- a widening this fix round does not
  have the scope to make. Recorded here, and in the plan's §6 residues, as
  scope this check knowingly does not cover, matching ``digitising.py``'s own
  precedent for a composed node stand-down.

**Vacuity, measured rather than assumed.** Mutated to an unconditional
``return ()``, cache cleared, this module's own test file still passes 34 of
its 52 tests -- expected of the negative half of any stand-down suite, not a
defect in it, but worth writing down so a future green run is not mistaken
for proof this check does anything. The 18 that fail are the ones that carry
the weight: a real disagreement asserted to ``severity == "refuse"``, or a
pinned message or ``where``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax.numpy as jnp

from rheplicant.config.findings import Finding, refuse
from rheplicant.config.postflight import Priced, register
from rheplicant.core.errors import AssemblyError

#: The one node this check is about (§2.6 item 8).  Restated as a literal
#: rather than imported from ``preflight/gated.py``: the two checks read two
#: different objects (a document's text there, a built twin here), so a
#: shared binding would let a mutation of either module's spelling look like
#: it silently repaired both, and neither test would notice.
_T6_NOISE_NODE = "noise"

#: The two ``inference.noise.kind`` spellings a
#: ``RadiometerNoiseOperator`` draw can legitimately be weighed against --
#: mirrors ``preflight/gated.py::_DRAWING_TYPES``'s own row for the same
#: class, by NAME there (the document's text) and by ``isinstance`` here (the
#: built operator).
_T6_RADIOMETER_KINDS: frozenset[str] = frozenset({"radiometer",
                                                   "radiometer_frozen"})

#: What ``inference.noise`` says when it says nothing -- ``build_noise``'s own
#: answer for an absent section and for an explicit ``kind: none`` alike.
_T6_NO_WEIGHT: str = "none"

#: The FULL twin is the default drawing twin (``sections/observed.py``'s own
#: default for ``observed.twin:``) -- a document that says nothing about which
#: twin drew its data still gets this check.
_T6_GENERATING_TWIN: str = "full"

#: Relative tolerance on the compared quantity (the fractional scatter for
#: the radiometer family, the sigma array for the homoscedastic one).
#:
#: **Measured, not assumed** -- both routes give EXACTLY 0.0 relative error
#: between the two independently-built objects, over the sweeps below:
#:
#: * Radiometer route: ``RadiometerNoiseOperator.channel_width`` /
#:   ``.integration_time`` and ``RadiometerNoise.channel_width`` /
#:   ``.integration_time`` are BOTH ``eqx.field(static=True)`` Python
#:   ``float``s, untouched by ``context.dtype`` -- ``context.dtype`` only
#:   reaches a TRACED leaf (``delivery.py::_as_traced``), and a static field
#:   is never one. Measured directly against the shipped classes, across four
#:   unit spellings of the same ``(1 MHz, 2 s)`` pair (``Hz``, ``kHz*ms``,
#:   ``GHz``, ``MHz*ms``): both ``.fractional`` properties read
#:   ``0.0007071067811865475`` on every spelling, ``rel = 0.0`` throughout.
#:   Command::
#:
#:       .venv/bin/python -c "
#:       from rheplicant.config.context import ResolutionContext
#:       from rheplicant.config.values import resolve_value
#:       from rheplicant.radio.instrument.noise import RadiometerNoiseOperator
#:       from rheplicant.inference.noise import RadiometerNoise
#:       ctx = ResolutionContext(dtype='float32')
#:       canon = lambda n: resolve_value(n, ctx).value
#:       pairs = [({'value': 1.0, 'unit': 'MHz'}, {'value': 2.0, 'unit': 's'}),
#:                ({'value': 1000.0, 'unit': 'kHz'}, {'value': 2000.0, 'unit': 'ms'}),
#:                ({'value': 0.001, 'unit': 'GHz'}, {'value': 2.0, 'unit': 's'}),
#:                ({'value': 1.0, 'unit': 'MHz'}, {'value': 2000.0, 'unit': 'ms'})]
#:       for dn, tn in pairs:
#:           dnu, tau = float(canon(dn)), float(canon(tn))
#:           op = RadiometerNoiseOperator(channel_width=dnu, integration_time=tau)
#:           like = RadiometerNoise(channel_width=dnu, integration_time=tau)
#:           print(op.fractional, like.fractional)
#:       "
#:
#: * Homoscedastic route: ``NoiseOperator.sigma`` (a traced leaf, cast to
#:   ``context.dtype`` by ``delivery.py::_as_traced``) and
#:   ``HomoscedasticNoise.sigma`` (``jnp.asarray(..., dtype=context.dtype)`` in
#:   ``sections/noise.py::build_noise``) are BOTH ``float32`` by this
#:   package's default -- so this route, unlike the radiometer one, genuinely
#:   passes through the downcast the tolerance exists to guard.  Measured by
#:   building both objects through the real ``build_model`` /
#:   ``sections/noise.py::build_noise`` pipeline (not a hand-rolled cast) over
#:   2000 random values in K and 2000 more spelled in ``celsius`` (``_ATOMS``'
#:   two temperature atoms -- ``mK`` is not one of them and is refused): 4000
#:   pairs, 0 disagreements, worst ``rel = 0.0``.  Command::
#:
#:       .venv/bin/python -c "
#:       import random
#:       from rheplicant.config.context import ResolutionContext
#:       from rheplicant.config.sections.compose import build_model
#:       from rheplicant.config.sections.noise import build_noise
#:       from rheplicant.config.sections.observation import ObservationBuild
#:       ctx = ResolutionContext(dtype='float32')
#:       obs = ObservationBuild(**{f: None for f in ObservationBuild._fields})
#:       def pair(value, unit):
#:           twin = build_model({'noise': {'type': 'NoiseOperator',
#:                                          'sigma': {'value': value, 'unit': unit}}},
#:                               ctx, switch_order=())
#:           weighed = build_noise({'kind': 'homoscedastic',
#:                                   'sigma': {'value': value, 'unit': unit}},
#:                                  observation=obs, context=ctx)
#:           return twin['noise'].sigma, weighed.model.sigma
#:       random.seed(1)
#:       worst = 0.0
#:       for _ in range(2000):
#:           a, b = pair(random.uniform(0.001, 500.0), 'K')
#:           worst = max(worst, abs(float(a) - float(b)))
#:       for _ in range(2000):
#:           a, b = pair(random.uniform(-273.0, 500.0), 'celsius')
#:           worst = max(worst, abs(float(a) - float(b)))
#:       print(worst)
#:       "
#:
#: **What this module's own tests can and cannot say about 1e-9.**  Both
#: measured routes give ``rel = 0.0`` exactly, so every test in this file that
#: exercises an agreeing pair would pass identically at ``_T6_RTOL = 0.0`` or
#: ``_T6_RTOL = 1e-12``: measured directly, mutating this constant to either
#: value leaves the whole of ``tests/config/test_postflight_noise.py`` green.
#: ``test_the_constant_is_far_below_a_real_disagreement`` (``_T6_RTOL <
#: 1e-6``) is the ONLY thing in this suite that discriminates the literal
#: ``1e-9`` at all, and it is a restatement of the constant rather than a
#: behavioural claim about it -- so read this comment as recording where
#: ``1e-9`` sits (far below the ten-fold disagreement this module's tests
#: construct, and comfortably inside the open interval ``(0, 1e-6)`` nothing
#: here narrows further), not as a derivation of why THIS value and not
#: another one in that interval was chosen.  No measurement in this file
#: exhibits a nonzero disagreement on either route for ``_T6_RTOL`` to be
#: guarding against; if one is ever found, it belongs in this comment as a
#: number, not as a plausible-sounding mechanism.
_T6_RTOL: float = 1e-9


def _t6_generating_twin(payload: Priced) -> Any | None:
    """The Assembly that actually drew the primary observation's data.

    ``None`` when there is no such assembly to read: the document has no
    ``inference.observed:`` at all, there is no single primary among several
    named observations, or the primary's own record is not
    ``from: simulation`` -- an ingested or file-form observation carries no
    second sigma to disagree with (contract step 1).

    **Reads the BUILT payload's own ``ObservedBuild.records``, never the raw
    document's text.**  ``records[primary]["twin"]`` is ``build_observed``'s
    own record of which twin it actually used (``"full"`` or ``"fit"``,
    ``sections/observed.py::_one``), defaulted the same way the builder
    defaults it -- so this reads as one voice with the thing it is describing
    rather than re-deriving the default from a second copy of the grammar.
    """
    observed = payload.run.inference.observed
    if observed is None:
        return None
    primary = observed.primary
    if primary is None:
        return None
    record = observed.records.get(primary)
    if not isinstance(record, Mapping) or record.get("from") != "simulation":
        return None
    choice = record.get("twin", _T6_GENERATING_TWIN)
    if choice == _T6_GENERATING_TWIN:
        return payload.run.twin
    if _T6_NOISE_NODE in (payload.run.inference.replaced or ()):
        # `inference.twin.replace: {noise: ...}` swapped the FIT twin's own
        # drawing node for a document-declared operator -- the operator that
        # actually drew this data is THAT replacement, not `model.noise`, and
        # a refusal built from `_t6_drawn` would go on to name `model.noise`
        # (both in `where` and in the message) for a document whose
        # `model.noise` may already agree with `inference.noise` exactly.
        # `payload.run.inference.replaced` is `build_fit_twin`'s own record
        # of which node ids `inference.twin.replace:` swapped
        # (`sections/twin.py::build_fit_twin`, ``tuple(replace)`` off the
        # declared mapping) -- reading it here is one voice with the builder
        # that did the swapping, aligned with the text pass's own
        # `preflight/gated.py::_t2c_repaired`, which already excludes this
        # same case from `C18.kind`.  Standing down entirely, not reading the
        # replacement's own sigma: the replacement is a document-declared
        # node spec with no guarantee it is even a noise model, and no
        # attempt is made to divine what the user meant by declaring it.
        return None
    return payload.run.inference.fit_twin


def _t6_drawn(payload: Priced) -> Any | None:
    """The operator at ``noise`` in the twin that actually drew the data.

    ``None`` when there is no twin to read at all (:func:`_t6_generating_twin`
    stood down), or when that twin does not carry the node: unlit entirely, a
    ``many`` node holding more than one instance (``AmbiguousNodeError``), or
    -- unreachable through ``__getitem__`` today but caught all the same,
    since :class:`~rheplicant.core.errors.AmbiguousNodeError` IS a bare
    :class:`~rheplicant.core.errors.AssemblyError` by inheritance and the
    two-member except-set below is what lets one clause cover both without
    also swallowing an unrelated failure this check has no business hiding.
    ``noise``'s own template entry declares ``many=False``
    (``rheplicant.radio.graph.RADIO_GRAPH.nodes["noise"]``), so the ambiguous
    branch is defensive rather than reachable by any document this check's
    own test suite can construct -- the same shape D-20 records for Task 5's
    ``adc``.
    """
    twin = _t6_generating_twin(payload)
    if twin is None:
        return None
    try:
        return twin[_T6_NOISE_NODE]
    except (KeyError, AssemblyError):
        return None


def _t6_agrees(drawn_value: Any, weighed_value: Any) -> bool:
    """``True`` within :data:`_T6_RTOL`, broadcasting -- never ``is``/``==``.

    ``jnp.allclose`` rather than a bare ``==``: a ``(1, n_freq)`` weighed
    sigma and a scalar drawn one can legitimately agree (contract step 5,
    the homoscedastic row), and comparing an array against a scalar with
    ``==`` returns an ARRAY of booleans whose truth value ``if`` cannot read
    -- ``ValueError: The truth value of an array...`` for more than one
    element, and a silently wrong single-element read for exactly one.
    ``bool(...)`` on the ``allclose`` result is what keeps this a real
    boolean either way.
    """
    return bool(jnp.allclose(jnp.asarray(drawn_value), jnp.asarray(weighed_value),
                             rtol=_T6_RTOL, atol=0.0))


def _t6_unwrapped(model: Any) -> Any:
    """The noise model itself, with any ``flags:`` wrapper taken off.

    ``inference.noise.flags: {from: observation}`` wraps the built model in
    :class:`~rheplicant.inference.noise.FlaggedNoise`, which carries neither
    ``.sigma`` nor ``.fractional`` nor ``.floor``; reading any of them through
    the wrapper is an ``AttributeError`` `sweep` launders into a sentence
    blaming the user for a document that is correct.  Flags say a sample was
    not OBSERVED, and ``FlaggedNoise.realise`` is the wrapped model's draw
    unchanged (its own docstring, verified against the shipped class), so the
    number this check compares is the base model's.
    """
    from rheplicant.inference import FlaggedNoise

    while isinstance(model, FlaggedNoise):
        model = model.base
    return model


def _t6_radiometer_fractional(kind: str, weighed: Any) -> float:
    """The likelihood side's fractional scatter, for either radiometer kind.

    ``kind: radiometer`` already built a
    :class:`~rheplicant.inference.noise.RadiometerNoise` (``weighed.model``,
    unwrapped by :func:`_t6_unwrapped` in case ``flags:`` put a
    :class:`~rheplicant.inference.noise.FlaggedNoise` around it), so its own
    ``.fractional`` answers directly.  ``kind: radiometer_frozen`` built no
    such object -- ``NoiseBuild.frozen`` carries the facts
    (``channel_width_hz``, ``integration_time_s``) and nothing else, and
    ``_KIND_KEYS["radiometer_frozen"]`` (``config/sections/noise.py``) does
    not even accept a ``flags:`` key -- so one is built here FROM those facts
    and its ``.fractional`` is read the same way, rather than re-writing
    ``1 / sqrt(dnu * tau)`` a fourth time.

    Imported at function scope, matching ``config/sections/noise.py:195``'s
    own precedent for this exact class.
    """
    from rheplicant.inference import RadiometerNoise

    if kind == "radiometer":
        return float(_t6_unwrapped(weighed.model).fractional)
    facts = weighed.frozen or {}
    return float(RadiometerNoise(
        channel_width=facts["channel_width_hz"],
        integration_time=facts["integration_time_s"]).fractional)


def _t6_radiometer_floor(kind: str, weighed: Any) -> float:
    """The floor declared on the likelihood side, for either radiometer kind.

    ``0.0`` (the shipped default, and therefore "none declared") when there
    is none. Read off ``weighed.model.floor`` for ``kind: radiometer`` --
    unwrapped the same way :func:`_t6_radiometer_fractional` is, since
    ``flags:`` wraps the same object this reads -- and off
    ``NoiseBuild.frozen["floor_k"]`` for ``kind: radiometer_frozen`` -- the
    two places :func:`~rheplicant.config.sections.noise.build_noise` puts it,
    mirroring :func:`_t6_radiometer_fractional`'s own split.
    """
    if kind == "radiometer":
        return float(_t6_unwrapped(weighed.model).floor)
    facts = weighed.frozen or {}
    return float(facts.get("floor_k", 0.0))


def _t6_floor_clause(floor: float) -> str:
    """The sentence appended when a floor is declared, or ``""``.

    Named so a disagreement's message can mention the floor without the
    reader going to look for a difference this check deliberately excluded
    from the comparison -- see this module's own docstring for why.
    """
    if floor <= 0.0:
        return ""
    return (
        f" inference.noise.floor: {floor!r} K is declared and takes no part "
        "in this comparison -- RadiometerNoiseOperator applies no floor, "
        "deliberately: a floor is a remedy for a reweighting iterate "
        "crossing zero, and a generator has no iterate."
    )


def _t6_radiometer(kind: str, drawn: Any, weighed: Any) -> Iterable[Finding]:
    """The ``RadiometerNoiseOperator`` family: compare the fractional scatter.

    Never the two fields -- ``(1 MHz, 2 s)`` on the operator and
    ``(2 s, 1 MHz)`` on the likelihood give the identical fractional scatter
    and are not a disagreement; see this module's own docstring.
    """
    drawn_f = float(drawn.fractional)
    weighed_f = _t6_radiometer_fractional(kind, weighed)
    if _t6_agrees(drawn_f, weighed_f):
        return ()
    floor_clause = _t6_floor_clause(_t6_radiometer_floor(kind, weighed))
    return (refuse("C18", f"model.{_T6_NOISE_NODE}", (
        f"model.{_T6_NOISE_NODE} draws this document's data at a fractional "
        f"scatter (1 / sqrt(channel_width * integration_time)) of "
        f"{drawn_f!r}, and inference.noise (kind: {kind}) weighs it at a "
        f"different fractional scatter of {weighed_f!r}. The fit is weighted "
        "against a scatter its own data does not have, and it returns a "
        "finite, correctly-shaped answer whose error bars are wrong by "
        "whatever the two differ by. Make the two agree -- change "
        f"model.{_T6_NOISE_NODE}'s channel_width/integration_time, or "
        "inference.noise's -- so the same physical bandwidth and "
        f"integration time reach both sides.{floor_clause} (check C18)."
    )),)


def _t6_homoscedastic(drawn: Any, weighed: Any) -> Iterable[Finding]:
    """The ``NoiseOperator`` family: compare sigma, broadcasting.

    ``weighed.model`` is unwrapped by :func:`_t6_unwrapped` first: ``flags:
    {from: observation}`` puts a
    :class:`~rheplicant.inference.noise.FlaggedNoise` around the
    :class:`~rheplicant.inference.noise.HomoscedasticNoise` this reads, and
    ``FlaggedNoise`` carries no ``.sigma`` of its own.
    """
    drawn_sigma = drawn.sigma
    weighed_sigma = _t6_unwrapped(weighed.model).sigma
    if _t6_agrees(drawn_sigma, weighed_sigma):
        return ()
    return (refuse("C18", f"model.{_T6_NOISE_NODE}", (
        f"model.{_T6_NOISE_NODE} draws this document's data with sigma = "
        f"{jnp.asarray(drawn_sigma)!r}, and inference.noise (kind: "
        f"homoscedastic) weighs it with a different sigma = "
        f"{jnp.asarray(weighed_sigma)!r}. The fit is weighted against a "
        "scatter its own data does not have, and it returns a finite, "
        "correctly-shaped answer whose error bars are wrong by whatever the "
        f"two differ by. Make the two agree -- change model.{_T6_NOISE_NODE}"
        ".sigma, or inference.noise.sigma -- so both sides declare the same "
        "number (check C18)."
    )),)


@register("C18")
def _t6_sigma_agreement(payload: Priced) -> Iterable[Finding]:
    """C18, the numeric half: does the twin's own drawn sigma agree with the
    likelihood's own weighed sigma?

    See this module's docstring for the full stand-down order and for why a
    family mismatch is never this check's to name -- that is
    ``preflight/gated.py``'s ``C18.kind``, and a REFUSE there halts
    ``load_document`` long before this pass runs at all.

    Yields at most one finding.
    """
    drawn = _t6_drawn(payload)
    if drawn is None:
        return ()
    weighed = payload.run.inference.noise
    if weighed.kind == _T6_NO_WEIGHT:
        return ()

    from rheplicant.radio import NoiseOperator, RadiometerNoiseOperator

    if isinstance(drawn, RadiometerNoiseOperator):
        if weighed.kind not in _T6_RADIOMETER_KINDS:
            return ()
        return _t6_radiometer(weighed.kind, drawn, weighed)
    if isinstance(drawn, NoiseOperator):
        if weighed.kind != "homoscedastic":
            return ()
        return _t6_homoscedastic(drawn, weighed)
    return ()
