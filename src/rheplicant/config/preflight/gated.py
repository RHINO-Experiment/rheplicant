"""The ``checks:`` grammar, in front of the beam -- and the two-word sigma
cross-check that never needed anything built.

**Why this module exists at all.**  ``inference.checks:``'s grammar was
enforced by ``sections/inference.py::_checks``, which ``load_document`` reaches
only after ``build_resources``.  Measured before this task, on documents
carrying an unreadable beam beside one ``checks:`` fault each: **the beam won
seven times out of seven**.  A user who typed ``mode: sipk`` was told about a
file they had not touched.  Task 1 moved the grammar itself into
:mod:`rheplicant.config.gating`, where it is a function returning findings
rather than a raise; this module is the slot that runs it before the money.

**Three slots, two functions.**

* :func:`_checks_grammar` claims ``A1.checks`` and ``A37`` **variadically**.
  Two stacked single-id decorators apply bottom-up and would insert ``A37``
  first, reversing run order -- which is which refusal a user reads -- and
  ``sweep``'s de-duplication by function identity is what stops one function
  bound to two slots running twice.
* :func:`_sigma_families` claims ``C18.kind``.

``A1.checks`` and not ``A1``: ``A1.runs``, ``A1.variants`` and ``A1.horizon``
are already bound, and §3.2(a) allows a dotted suffix when several functions
each decide part of one check.  ``Finding.check`` stays the bare ``"A1"``.

``C18.kind`` and not ``C18``: the family mismatch is decidable from two words
of text and belongs in front of the beam; the numeric comparison needs the
built operators and is a later task's ``C18``.  **Two functions, two slots,
one bare id.**

**This module imports** :mod:`rheplicant.config.gating` **and never
``sections/``.**  ``gating`` imports ``errors``, ``findings`` and
``resources`` and nothing else from this package, precisely so that a
pre-flight slot can reach the grammar without dragging ``jax.numpy`` and the
builders behind the scope guard.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse, warn
from rheplicant.config.gating import check_gates
from rheplicant.config.paths import parse_path
from rheplicant.config.preflight import register
from rheplicant.config.preflight.model import _A30_NOT_FITTING, _a30_exits

#: Drawing operator class name -> the ``inference.noise.kind`` values that
#: weigh with the same noise model.  A frozenset per row and not one string:
#: ``radiometer`` and ``radiometer_frozen`` are two spellings of one sigma
#: (the frozen one is decided from the data rather than the prediction, and
#: both scale as ``1/sqrt(channel_width * integration_time)``), and a
#: one-string table would refuse a document that is right.
#:
#: A ``model.noise.type`` this table does not know stands the check down: the
#: two rows are the operators ``inference.noise`` has a counterpart for, and
#: guessing about a third would refuse a document this check cannot read.
_DRAWING_TYPES: dict[str, frozenset[str]] = {
    "NoiseOperator": frozenset({"homoscedastic"}),
    "RadiometerNoiseOperator": frozenset({"radiometer", "radiometer_frozen"}),
}

#: What ``inference.noise`` says when it says nothing.  ``build_noise``
#: answers ``NoiseBuild(kind="none")`` for an absent section AND for an
#: explicit ``kind: none``, so the two are one cell here.
_T2C_NO_WEIGHT: frozenset[str] = frozenset({"none"})

#: The one node C18 is about (§2.6 item 8).  ``stochastic_nodes`` also returns
#: ``rfi_field``, and ``inference.noise`` has no ``rfi`` counterpart -- there
#: is no second number to disagree with.  The widening belongs to the plan
#: that gives RFI a likelihood.
_T2C_NOISE_NODE: str = "noise"

#: **Scoped exactly as the numeric C18 is** (D-10): the drawing operator is
#: only the generator when the data came out of the FULL twin, OR out of a
#: ``twin: fit`` that :func:`_t2c_repaired` finds was never actually repaired
#: (MAJOR 5: with zero declared latents, nothing forces that repair, and a
#: bare ``twin: fit`` is then the full twin under another name).  Where the
#: repair really did take ``inference.twin.without: [noise]`` (or an
#: equivalent ``replace:``) out of the tree that produced the data,
#: ``model.noise.type`` describes something that provably never touched it,
#: and a refusal naming it would be a claim about the document that is false.
#: ``full`` is ``sections/observed.py``'s own default, so a document that
#: says nothing still gets the check.
_T2C_GENERATING_TWIN: str = "full"


def _t2_where(finding: Finding) -> Finding:
    """The same finding with a ``where`` this pass's guard can accept.

    ``check_gates`` composes ``inference.checks.<name>`` from the KEY a
    document wrote, and a document may write any YAML scalar there --
    measured, ``parse_path('inference.checks.7')`` and
    ``parse_path('inference.checks.a b')`` both RAISE.  ``passes.check_where``
    turns that into a ``ConfigError`` that aborts the whole pre-flight pass
    and hides every finding after it, which is the exact failure the
    collect-rather-than-raise design exists to prevent.

    So an unusable path is re-homed onto its PARENT, which is always legal and
    is still the line the reader edits.  The MESSAGE is untouched and still
    quotes the key the user wrote, so nothing is lost from what they are told.

    ``sections/inference.py::_checks`` needs none of this: it raises the
    message and never looks at the ``where``.
    """
    try:
        parse_path(finding.where)
    except ConfigError:
        return dataclasses.replace(finding, where="inference.checks")
    return finding


@register("A1.checks", "A37")
def _checks_grammar(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Checks A1 (the ``checks:`` grammar) and A37 (a skip's own reason).

    Reads ``inference.checks``, hands it to
    :func:`~rheplicant.config.gating.check_gates`, and yields what comes back.
    **That is the whole function.**  It writes no message of its own: every
    sentence is ``gating``'s, and the seven that moved there are byte-identical
    to what ``sections/inference.py`` raised at ``e0e024a``.  One binding per
    sentence rather than two that drift is §2.2 of Plan 3A's own instruction,
    and it is what makes the move a constraint rather than a rewrite.

    Absent, ``None`` and a non-mapping ``inference:`` all mean "nothing to
    say" -- ``inference: 7`` is ``build_inference``'s own refusal, with the
    value the user wrote.
    """
    section = document.get("inference")
    if not isinstance(section, Mapping):
        return ()
    return tuple(_t2_where(one) for one in check_gates(section.get("checks")))


def _t2c_drawn(document: Mapping[str, Any]) -> str | None:
    """The class name at ``model.noise``, or ``None`` when there is not
    exactly one.

    ``None`` for an absent node, for a ``model:`` that is not a graph, for a
    ``compose:`` block and for a list -- a composed node has no single drawing
    operator, and a check that guessed would refuse a document it cannot read.
    A bare ``model["noise"]["type"]`` raises ``TypeError`` on a list, and a
    check that raises aborts the pass and hides every later finding.
    """
    model = document.get("model")
    if not isinstance(model, Mapping) or model.get("kind", "graph") != "graph":
        return None
    spec = model.get(_T2C_NOISE_NODE)
    if not isinstance(spec, Mapping) or "compose" in spec:
        return None
    declared = spec.get("type")
    return declared if isinstance(declared, str) else None


def _t2c_weighed(document: Mapping[str, Any]) -> str | None:
    """``inference.noise.kind``, or ``None`` when the document does not say.

    An absent ``inference.noise:`` and an explicit ``kind: none`` are one
    answer here, as they are in ``build_noise`` -- both give
    ``NoiseBuild(kind="none")`` -- so both come back as a member of
    :data:`_T2C_NO_WEIGHT` rather than as two cells.
    """
    section = document.get("inference")
    if not isinstance(section, Mapping):
        return None
    noise = section.get("noise")
    if noise is None:
        return "none"
    if not isinstance(noise, Mapping):
        return None
    kind = noise.get("kind")
    return kind if isinstance(kind, str) else None


def _t2c_generated(document: Mapping[str, Any]) -> bool:
    """Did the TWIN generate this document's primary observation?

    Two conditions, and both are §2.6's and D-10's rather than this check's:

    * ``from: simulation`` -- when the data came from a file the twin drew
      nothing and there is no second sigma to disagree with (§2.6 item 6);
    * ``twin: full``, OR a ``twin: fit`` whose ``inference.twin:`` has NOT
      actually taken the drawing operator out of the tree
      (:func:`_t2c_repaired`).  ``twin: fit`` alone is not sufficient: a
      document declaring no latent at all never reaches
      ``refuse_stochastic_stages``, so it can spell ``twin: fit`` with no
      ``inference.twin:`` repair, and its fit twin is then the SAME object as
      the full twin -- still carrying the draw (D-10, and see
      :data:`_T2C_GENERATING_TWIN`).

    The primary is picked exactly as ``build_observed`` picks it: the
    single-record form is ``primary``; otherwise a record literally named
    ``primary``, else the only one, else none at all.

    **This function's own CONTRACT is frozen at "a single primary-alone
    bool" and this fix (A1.2) does not touch it.**
    ``tests/config/test_postflight_noise.py`` imports and calls this exact
    function to compare the two C18 vantage points (this text check and the
    built ``_t6_generating_twin`` in ``postflight/noise.py``), and that
    module is out of scope for A1.2 -- so widening what THIS function returns
    would silently change what that other test file asserts.  The widening
    lives beside it instead, in :func:`_t2c_generating_records`, which this
    function now feeds through the shared :func:`_t2c_record_generated`.

    **BLOCKER 2 (Plan 3C fix round) -- the multi-named-no-``primary`` case,
    fixed for C18.kind by A1.2 and for the bare (numeric) C18 since.**  Two
    or more NAMED records, none of them literally called ``primary``, resolve
    no primary at all, and THIS function still returns ``False`` on that case
    exactly as it does on a document with no ``observed:`` section
    whatsoever -- unchanged, per the contract note above. Measured, before
    A1.2:

    .. code-block:: text

        single-record (``primary``)                REFUSED
        named ``'primary'``                         REFUSED
        named ``'alpha'`` ALONE (only record)        REFUSED
        named ``'alpha'`` + ``'beta'`` (no ``'primary'``)  LOADS, report=[]

    The trigger is a record NAME, not a physical property of the document --
    the fourth row can carry the exact same tenfold sigma disagreement as the
    first three and load clean, because the primary-resolution rule that
    every other check relies on has nothing to resolve.  **This is not a
    principled boundary; it is an accident of the primary-resolution rule**,
    and it is the same shape D-C17 exists to name: *"finite, correctly
    shaped, wrong, and invisible to every diagnostic."*

    **A1.2's decision: widen the READER that decides which records C18.kind
    checks, not this function.**  :func:`_sigma_families` no longer calls
    THIS function to decide whether to run at all; it calls
    :func:`_t2c_generating_records`, which reproduces this function's exact
    single-record answer on every row above except the fourth, and on the
    fourth now runs the per-record test once per named record instead of
    returning nothing.  The two C18 vantage points agree everywhere they did
    before and, since the numeric half was widened to match, on the fourth row
    as well: ``postflight/noise.py::_t6_generating_records`` is that function,
    built the same way and sitting beside its own single-primary reader for
    the same reason this one sits beside THIS function.  For one release they
    differed here -- C18.kind reported the fourth row per record and the
    numeric C18 was silent on it -- and that is recorded in the other module's
    docstring rather than only in a plan.
    """
    section = document.get("inference")
    if not isinstance(section, Mapping):
        return False
    observed = section.get("observed")
    if not isinstance(observed, Mapping):
        return False
    if "from" in observed or "file" in observed:
        record: Any = observed
    else:
        named = {name: spec for name, spec in observed.items()
                 if isinstance(spec, Mapping)}
        if "primary" in named:
            record = named["primary"]
        elif len(named) == 1:
            record = next(iter(named.values()))
        else:
            return False
    return _t2c_record_generated(record, section)


def _t2c_record_generated(record: Mapping[str, Any],
                          section: Mapping[str, Any]) -> bool:
    """Did the TWIN generate THIS record specifically?

    The per-record half of :func:`_t2c_generated`'s own test (``from:
    simulation``, then :data:`_T2C_GENERATING_TWIN` or an unrepaired ``twin:
    fit``), pulled out so :func:`_t2c_generating_records` can run it once per
    NAMED record without re-deriving it, and so :func:`_t2c_generated` keeps
    answering exactly as it always has by calling this on the one record it
    resolves.
    """
    if record.get("from") != "simulation":
        return False
    if record.get("twin", _T2C_GENERATING_TWIN) == _T2C_GENERATING_TWIN:
        return True
    return not _t2c_repaired(section)


def _t2c_generating_records(
        document: Mapping[str, Any]) -> tuple[str | None, ...]:
    """Which record(s) :func:`_sigma_families` must check -- BLOCKER 2's own
    fix, threaded WITHOUT changing :func:`_t2c_generated`'s contract (see its
    docstring for why that function itself stays untouched).

    Returns one entry per record to check, so a finding can name which one
    disagreed:

    * every case :func:`_t2c_generated` already resolves to a SINGLE record
      -- the single-record legacy form, a literally-named ``primary``, or the
      lone named record -- widens to nothing: ``(None,)`` when the twin
      generated it, ``()`` otherwise.  The ``None`` name carries no
      record-naming text, so the finding this earns is byte-identical to the
      one C18.kind shipped before A1.2.
    * two or more NAMED records with none of them literally called
      ``primary`` -- the case that used to fall through to
      :func:`_t2c_generated`'s own ``return False``, standing this check down
      exactly as an absent ``observed:`` does.  This runs
      :func:`_t2c_record_generated` once per named record instead, and
      returns the NAME of every one the twin generated: zero, one, or all of
      them.  A record from a file, or one whose fit twin was actually
      repaired, is simply not in the tuple -- there is no second sigma to
      disagree with for that one, exactly as :func:`_t2c_generated` reads a
      single such record.
    """
    section = document.get("inference")
    if not isinstance(section, Mapping):
        return ()
    observed = section.get("observed")
    if not isinstance(observed, Mapping):
        return ()
    if "from" in observed or "file" in observed:
        return (None,) if _t2c_generated(document) else ()
    named = {name: spec for name, spec in observed.items()
             if isinstance(spec, Mapping)}
    if "primary" in named or len(named) == 1:
        return (None,) if _t2c_generated(document) else ()
    return tuple(name for name, record in named.items()
                 if _t2c_record_generated(record, section))


def _t2c_repaired(section: Mapping[str, Any]) -> bool:
    """Does ``inference.twin:`` take the noise node OUT of the fit twin?

    A ``twin: fit`` observation is silent only when SOMETHING has actually
    taken the drawing operator out of the tree that produced the data.  With
    zero declared latents, ``refuse_stochastic_stages`` -- which is what
    ordinarily forces a ``without:``/``replace:`` onto a fit twin that still
    draws -- never runs (it is reached only while building a
    ``ParameterSpace`` over a declared latent), so a document CAN spell
    ``twin: fit`` with no ``inference.twin:`` at all, and its fit twin is
    then the SAME object as the full twin: still carrying the draw.
    """
    twin = section.get("twin")
    if not isinstance(twin, Mapping):
        return False
    without = twin.get("without")
    without = [without] if isinstance(without, str) else without
    if isinstance(without, (list, tuple)) and _T2C_NOISE_NODE in without:
        return True
    replace = twin.get("replace")
    return isinstance(replace, Mapping) and _T2C_NOISE_NODE in replace


def _t2c_subject(name: str | None) -> str:
    """The finding message's own noun phrase for WHOSE data disagreed.

    ``None`` is every :func:`_t2c_generating_records` subject that resolves
    to a single record with nothing to widen -- the single-record legacy
    form, a literal ``primary``, or the lone named record -- and reads
    exactly as the sentence did before A1.2: *"this document's data"*.  A
    NAME is BLOCKER 2's own widened case: two or more named records, none
    literally called ``primary``, so the sentence must say WHICH one it read.
    """
    if name is None:
        return "this document's data"
    return f"the data of observed record {name!r}"


@register("C18.kind")
def _sigma_families(document: Mapping[str, Any]) -> Iterable[Finding]:
    """Check C18, the half decidable from two words: the drawing operator and
    the likelihood's ``kind:`` are the same noise MODEL.

    The numeric half -- do the two sigmas agree to a tolerance -- needs the
    built operators and is a later task's bare ``C18``.  This one is
    ``model.noise.type`` against ``inference.noise.kind`` and nothing else, so
    it runs before the beam is read.

    **The two stand-down rows are evaluated FIRST, and that ordering is the
    check's contract rather than an implementation detail.**  Read the other
    way round, the WARN row fires on any document with a drawing operator, no
    ``inference.noise:`` and a fitting run **whatever the observation says** --
    so an INGESTED document, whose twin drew nothing at all, earns a warning
    about a sigma it has not got.

    *The plan's own example of that was measured and does not hold, and the
    correction matters because it is the other half of the same reading.*
    The review predicts ``exit_helpers.diagnostic_document()`` trips the WARN,
    on the grounds that ``_A30_NOT_FITTING`` is ``{'forward', 'mmodes'}`` and
    its run is neither.  It does not: its run is
    ``kind: diagnostics.identifiability`` and ``sections/runs._KINDS`` holds
    ``identifiability``, so :func:`_a30_exits` -- which intersects with that
    closed enum -- answers ``()``.  Reading the fitting set off ``_kinds``
    minus the complement, which is what the review assumes, WOULD warn about
    it.  Both readings are closed here: the fitting set is ``_a30_exits``' and
    the stand-down rows come first.

    Measured with both in place: **no builder in ``tests/config/*_helpers.py``
    earns anything from this check at all** -- REFUSE census 0, WARN census 0
    over all thirteen.

    **MINOR 9, recorded rather than fixed: this check can pre-empt A26/A49 on
    the very block it reads, and that is accepted.**  ``gated`` foot-imports
    before ``noise``, and ``sweep`` runs the registered checks in that
    insertion order, so on ``{kind: radiometer_frozen, include_logdet:
    true}`` a user is told to change ``model.noise.type`` (C18's family
    mismatch, if the drawing operator does not already agree) while their
    actual fault is the stray ``include_logdet:`` A49 would have named.  Both
    are real errors about the same document -- this is message ORDERING, not
    a wrong answer -- and reordering the two modules is out of scope here (it
    would move every other check's position in the report too).  It is also
    the reason ``tests/config/test_preflight_noise.py::
    TestTheRowsArriveBeforeTheBeam``'s two ``include_logdet`` documents give
    ``model.noise`` the matching drawing operator rather than the base
    document's default: that fixture WORKS AROUND this ordering rather than
    resolving it.

    **A1.2: widened to one call per record when ``observed:`` names two or
    more records and none of them is literally ``primary`` (BLOCKER 2, see
    :func:`_t2c_generating_records`), so "yields at most one finding" is now
    true only per RECORD.**  Every document this module shipped before A1.2
    still resolves to exactly one subject (``None``, unnamed) and still
    yields at most one finding; a document that trips BLOCKER 2's own case
    can now yield one finding per named record the twin generated, each
    naming the record it is about.
    """
    drawn = _t2c_drawn(document)
    if drawn not in _DRAWING_TYPES:
        return ()
    subjects = _t2c_generating_records(document)
    if not subjects:
        return ()
    where = f"model.{_T2C_NOISE_NODE}"
    agrees = _DRAWING_TYPES[drawn]
    weighed = _t2c_weighed(document)
    if weighed is None or weighed in agrees:
        # `None` is a kind this check cannot read -- `build_noise`'s own
        # refusal names it with the value the user wrote, and answering here
        # would pre-empt a more specific sentence.
        return ()
    if weighed in _T2C_NO_WEIGHT:
        fitting = _a30_exits(document)
        if not fitting:
            return ()
        return tuple(warn("C18", where, (
            f"model.{_T2C_NOISE_NODE} draws {_t2c_subject(name)} with "
            f"{drawn}, and inference.noise: says nothing -- so {list(fitting)} "
            "weighs every channel equally over data that carries noise. The "
            "fit still returns a finite, correctly-shaped answer; its error "
            "bars are the ones an unweighted least squares gives. Declare "
            f"inference.noise: {{kind: {sorted(agrees)[0]}}} to weigh what "
            f"you draw, or drop model.{_T2C_NOISE_NODE} -- and the "
            f"inference.twin.without: [{_T2C_NOISE_NODE}] that repairs it -- "
            "if this data is meant to be noise-free (check C18)."))
            for name in subjects)
    other = sorted(
        name for name, kinds in _DRAWING_TYPES.items() if weighed in kinds)
    if not other:
        # A kind no operator in `_DRAWING_TYPES` produces.  Today that is only
        # a kind `build_noise` itself refuses -- `inference.noise.kind:
        # banana`, whose own sentence names the enum and the value -- and
        # answering here would pre-empt it.  Standing down is also what stops
        # the escape sentence below naming `other[0]` out of an EMPTY list:
        # measured, that is an `IndexError` inside the check, which `sweep`
        # turns into "check 'C18.kind' RAISED" and which discards every other
        # finding in the report.
        return ()
    return tuple(refuse("C18", where, (
        f"model.{_T2C_NOISE_NODE} draws {_t2c_subject(name)} with {drawn}, "
        f"and inference.noise.kind: {weighed} weighs the likelihood with a "
        "different noise model. They are not two spellings of one sigma, so "
        "there is no number to compare: the fit is weighted against a scatter "
        "its own data does not have, and it returns a finite, "
        "correctly-shaped answer whose error bars are wrong by whatever the "
        f"two models differ by. Write model.{_T2C_NOISE_NODE}.type: "
        f"{other[0]} to draw what you weigh, or inference.noise.kind: "
        f"{sorted(agrees)[0]} to weigh what you draw (check C18)."))
        for name in subjects)


# `_A30_NOT_FITTING` is imported rather than re-derived (§3.1's one-name-one-
# binding rule) and reaches this module through `_a30_exits`, which is the
# narrowed reader: it drops a kind the run grammar does not know and a run
# declaring `expect: refuse`, both of which `_kinds` deliberately keeps.  The
# name is bound here so that a mutation of the complement is visibly this
# check's dependency and so `ruff` does not read the import as unused.
_T2C_NOT_FITTING: frozenset[str] = _A30_NOT_FITTING
