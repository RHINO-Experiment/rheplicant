"""A43 and B9 -- the calibrator the flagger eats, and one beam analysed twice.

Split from ``inflight/twin.py`` up front rather than when it got long (§0.3
E.1 pre-names both modules, and §3.2(g) names this one by path): ``twin.py``
is about the SWITCH and the temperature leaves hanging off it, and this module
is about the two things that pass through an aperture -- a tone the RFI
flagger can see, and a beam two projectors each pay to analyse.

**Neither row saves anything and neither message says it does.**  The built
pass runs when ``load_document`` is ready to return, so B9's second analysis
has already been PAID by the time B9 mentions it.  What the slot buys is that
the comparison happens at all, in this layer's voice: the number B9 reports is
the cost of the run that just loaded, and the next load is where the saving
would land.

**What each one needs the built objects for.**

* **A43** compares ``protect_floor * amplitude * max_k w_k`` -- the Kelvin
  level of the tone's protection cut -- against ``flagging.threshold``,
  through ``adc.scale``.  ``_weights`` is closed form on ``context.freq`` and
  four static scalars, so this is arithmetic on the resolved grid and **not a
  forward pass**; the twin is never called.  Measured warm at 0.068 ms.
* **B9** groups projectors by the beam they name and the two numbers that
  decide the analysis, and then asks the BUILT projectors whether their
  ``beam_alms`` are the same object.  That last question is the one only this
  slot can ask, and it is what tells one analysis from two -- bitwise equality
  cannot, because two analyses of one map with the same statics are
  deterministic and come back equal.

**One binding.**  Neither row restates a shipped sentence: A43's comparison
exists nowhere in the package (``calibration.py`` states the RULE in prose --
"that is the number to compare against a flagging threshold" -- and compares
nothing), and B9 has no shipped text at all.  The constants and the weights
are IMPORTED from ``radio/instrument/calibration.py`` so the arithmetic has
one binding; only the comparison is written here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.findings import Finding, refuse, warn
from rheplicant.config.inflight import Built, register_built
from rheplicant.config.resources import resolved_specs

#: The prefix ``resolved_specs`` gives a projector.  Its keys are the dotted
#: strings ``build_resources`` uses, so this is the layer's own spelling.
_PROJECTORS = "resources.projectors."

#: ``from_beam_maps(iterations=)``'s default and ``_analyse``'s, read as one
#: number rather than two: ``build_projector`` writes
#: ``int(spec.get("beam_iterations", 3))`` on both engine branches.
_DEFAULT_BEAM_ITERATIONS = 3

#: The tail A43 appends.  ``adc.scale`` is the only leaf on the trunk between
#: the tone and the flagger that carries a UNIT -- ``bandpass`` and ``gain``
#: sit there too and are dimensionless (schema §11's own correction: "gain
#: sits before adc on the trunk, so gain is genuinely dimensionless and only
#: adc.scale carries adc_count/K") -- so it is the only one multiplied in.
_A43_TAIL = (
    "bandpass and gain sit on the same trunk between the two and are "
    "DIMENSIONLESS, so neither is multiplied in; adc.scale is the leaf that "
    "carries adc_count/K and it is. Note what adc.scale is: a TRACED field, "
    "so a latent bound into it by inference.bindings overwrites it at bind "
    "time and the twin leaf read here holds only the init. This comparison is "
    "therefore about the run as it was declared, and a fit that frees the ADC "
    "scale can move the threshold out from under the tone without this check "
    "seeing it. And this runs after build_resources: the beam is long since "
    "read, so what the slot buys is the comparison, not a saving"
)

#: The tail B9 appends.
_B9_TAIL = (
    "Measured: two projectors naming one beam with the same lmax and "
    "beam_iterations analyse it TWICE and their beam_alms come back bitwise "
    "equal and NOT the same array -- so equality cannot tell the two cases "
    "apart and object identity can, which is why this reads the built "
    "projectors rather than the text alone. The cost is one whole analysis "
    "per extra projector: 8.9 -> 16.8 ms warm at nside 4 / lmax 8, against "
    "1.4 s at nside 16 / lmax 31. The second consequence is the one worth "
    "more than the milliseconds: a gradient into the beam map now splits "
    "between two independent leaves rather than accumulating on one"
)


def _twins(run: Built) -> list[tuple[Any, tuple[str, ...]]]:
    """``inflight/twin.py``'s route walk, imported rather than restated.

    A FUNCTION-scope import, deliberately.  A head import of a sibling under
    ``inflight/`` registers that sibling's checks first -- measured at wave
    1's merge, where a module sorting first in the foot block had its own
    check land at index 29 -- so the import order of this package would then
    be decided here rather than by ``__init__``'s block.  Nothing about this
    module's answers depends on that, and neither should the registry's shape.
    """
    from rheplicant.config.inflight.twin import _twins as routes

    return routes(run)


def _where(replaced: tuple[str, ...], node_id: str) -> str:
    """``inflight/twin.py``'s document-key resolver, imported the same way."""
    from rheplicant.config.inflight.twin import _where as document_key

    return document_key(replaced, node_id)


def _node(twin: Any, node_id: str) -> Any:
    """The single operator at ``node_id``, or ``None``.

    ``lit`` first, so a document that never lights the node costs an ``in``.
    A node carrying SEVERAL instances answers ``None`` rather than raising
    ``AmbiguousNodeError``: neither ``cw_tone`` nor ``flagging`` nor ``adc`` is
    a ``many`` node, so the branch is defensive rather than expected, and a
    check that raises loses every finding after it (``sweep`` wraps it as
    "check RAISED", which aborts the pass while every ``match=`` pin in the
    suite still passes).
    """
    if node_id not in (getattr(twin, "lit", None) or ()):
        return None
    if dict(getattr(twin, "instances", ()) or ()).get(node_id) is not None:
        return None
    return twin[node_id]


# ---------------------------------------------------------------------------
# A43
# ---------------------------------------------------------------------------


def _protection_cut(tone: Any, freq: Any, time: Any) -> float | None:
    """``protect_floor * level * max_k w_k`` in Kelvin, worst case over the run.

    The peak of the tone's own protection cut: every channel at or above this
    level is protected from the flagger, and every channel BELOW it is not.

    ``_weights`` is :class:`CWCalibrationOperator`'s own and is imported rather
    than restated, so the lineshape, the normalisation and the ``sinc2``/
    ``gaussian`` split have one binding.  With ``drift_rate`` zero it returns
    ``(n_freq,)``; with a drift it returns ``(n_time, n_freq)`` and **the max
    is taken over the FREQUENCY axis**, which is what
    ``_protection_mask`` does (``weights.max(axis=-1, keepdims=True)``) -- a
    max over the flat array would answer a different question on a drifting
    tone and the two agree on every static one, so a test on a static tone
    cannot tell them apart.

    ``amplitude_drift_rate`` enters through ``level``: the mask is computed
    from the weights alone and is level-independent, but the KELVIN value at
    the mask's own boundary is ``protect_floor * level(t) * max_k w_k(t)``, and
    a drifting level moves it.  The worst case over the run is the number that
    decides whether any sample has an unprotected channel above the threshold.
    """
    from rheplicant.radio.instrument.calibration import CWCalibrationOperator

    if not isinstance(tone, CWCalibrationOperator):
        return None
    drifting = tone.drift_rate != 0.0 or tone.amplitude_drift_rate != 0.0
    elapsed = None
    if drifting:
        if time is None:
            return None
        elapsed = (time - time[0])[:, None]
    centre = tone.tone_freq
    level = tone.amplitude
    if elapsed is not None:
        centre = centre + tone.drift_rate * elapsed
        level = level * (1.0 + tone.amplitude_drift_rate * elapsed)
    weights = tone._weights(freq, centre)
    peak = weights.max(axis=-1, keepdims=True)
    return float((tone.protect_floor * level * peak).max())


@register_built("A43")
def _tone_survives_flagging(run: Built) -> Iterable[Finding]:
    """A43: the flagger's threshold sits below the tone's protection cut.

    The rule is ``calibration.py``'s own, stated there in prose and compared
    nowhere: the protected set is every channel at or above ``protect_floor``
    of the tone's peak channel, "which is the number to compare against a
    flagging threshold".  A channel carrying tone ABOVE the flagger's
    threshold but BELOW that cut is flagged -- it is the calibrator's own
    shoulder, removed as RFI -- and ``docs/contracts.md`` measures the
    unprotected version of the same failure at twelve flagged samples: "That
    is the calibrator, gone."

    **``FlaggingOperator`` only.**  ``flagging`` takes two classes and
    ``MomentRFIFlaggingOperator`` has no ``threshold`` field at all (measured:
    its fields are ``config`` and ``kernel_shapes``), so there is no number to
    compare and this stands down on it.  Read as ``getattr(..., "threshold")``
    rather than by class name so a third flagger with a threshold is covered
    the day it lands.

    **No ``adc`` lit is a real document, not a defect.**  ``twin['adc']``
    raises ``KeyError`` on an ordinary ``cw_tone`` + ``flagging`` document
    (measured), and an unlit ``adc`` node contracts to identity -- the data
    reaching the flagger is in Kelvin -- so the scale is 1.0 and the
    comparison still means something.

    **No forward pass.**  ``_weights`` is closed form on ``context.freq`` and
    four static scalars.  §0.1 puts a Jacobian, an SVD and a real evaluation
    of the twin in Plan 3C; this reaches for none of them.

    Walks the raw twin and the fit twin (``twin.py::_twins``), so a tone or a
    flagger that ``inference.twin.replace`` rebuilt is compared as the FIT
    twin actually holds it.  The pair is de-duplicated by operator identity,
    which is what keeps a document with no ``replace:`` at all from hearing
    the same sentence twice: measured, a node the fit twin did not rebuild is
    the same object in both.
    """
    freq = getattr(run.context, "freq", None)
    if freq is None:
        return
    time = getattr(run.context, "time", None)
    seen: set[tuple[int, int, int]] = set()
    for twin, replaced in _twins(run):
        tone = _node(twin, "cw_tone")
        flagger = _node(twin, "flagging")
        if tone is None or flagger is None:
            continue
        threshold = getattr(flagger, "threshold", None)
        if threshold is None or isinstance(threshold, bool):
            continue
        adc = _node(twin, "adc")
        # The third element is a MEASURED EQUIVALENT MUTANT today, and it is
        # kept with the reason written down rather than deleted (§0.3 F.5(10)
        # is the precedent).  `build_fit_twin` RE-ASSEMBLES, so a `replace:`
        # of any node mints new objects for ALL of them -- measured on a
        # document replacing only `adc`: `tone shared: False, flagger shared:
        # False, adc shared: False` -- and a fit twin built by `without:`
        # alone shares all three.  So there is no document today in which the
        # first two match and the third does not, and dropping `id(adc)`
        # changes no verdict.  It stays because that is a property of HOW the
        # fit twin is built, not of what this check needs: `Assembly.
        # replace_node` swaps one operator through `eqx.tree_at` and leaves
        # the others shared, so the day the fit twin is built that way the
        # adc is the one leaf of the three that can move alone -- and it is
        # the one carrying the unit.
        signature = (id(tone), id(flagger), id(adc))
        if signature in seen:
            continue
        seen.add(signature)
        scale = 1.0 if adc is None else float(getattr(adc, "scale", 1.0))
        cut = _protection_cut(tone, freq, time)
        if cut is None:
            continue
        counts = cut * scale
        if counts <= float(threshold):
            continue
        where = _where(replaced, "flagging")
        yield refuse(
            "A43", where,
            f"{where}.threshold is {float(threshold):.6g} and the tone's own "
            f"protection cut sits at {counts:.6g} in the same units "
            f"(protect_floor {tone.protect_floor:.6g} x amplitude "
            f"{tone.amplitude:.6g} K x the peak channel weight, x adc.scale "
            f"{scale:.6g}). Every channel carrying tone between those two "
            "numbers is above the flagger's threshold and below the tone's "
            "own protected set, so the flagger takes it: those channels are "
            "the calibrator's shoulders, and removing them biases the "
            "recovered tone level low while nothing raises and every shape "
            "stays right. docs/contracts.md measures the unprotected version "
            "of this at twelve flagged samples -- 'That is the calibrator, "
            f"gone.' Raise {where}.threshold above {counts:.6g}, or lower "
            "model.cw_tone.protect_floor so the protected set reaches down "
            f"past it ({float(threshold) / counts * tone.protect_floor:.6g} "
            f"or less does it at this amplitude). {_A43_TAIL} (check A43).")


# ---------------------------------------------------------------------------
# B9
# ---------------------------------------------------------------------------


def _analysing(document: Mapping[str, Any]) -> dict[str, tuple[str, int, int, str]]:
    """``projector name -> (beam ref, lmax, beam_iterations, engine)`` for the analysers.

    ``resolved_specs`` applies ``extends:`` and is TOTAL, so an entry that
    inherits its ``lmax`` from a parent is covered and a malformed one is
    dropped rather than raised on.

    **Every value is type-checked before it is put in a key**, and that is not
    defensiveness for its own sake: a document may write ``lmax: [8]`` or
    ``beam: {ref: {a: 1}}``, and an unhashable value in a dict key raises
    ``TypeError`` inside the check -- which ``sweep`` reports as "check 'B9'
    RAISED", aborting the pass and hiding every finding after it while every
    ``match=`` pin in the suite still passes.  An entry whose numbers this
    cannot read is simply not grouped: ``build_projector`` has already refused
    the ones that are genuinely wrong, in its own words.

    **An entry carrying ``beam_alms:`` is excluded**, because it does not
    analyse: ``build_projector`` resolves that value node and skips
    ``_analyse`` entirely.  That is the route this check advises, so counting
    it would make the advice fire on the documents that took it.
    """
    found: dict[str, tuple[str, int, int, str]] = {}
    # `isinstance` on the payload's own `document` too: a hand-assembled
    # payload is a supported caller, and `.get` on a list raises INSIDE the
    # check, which `sweep` reports as "check 'B9' RAISED".
    if not isinstance(document, Mapping):
        return found
    for dotted, spec in resolved_specs(document.get("resources")).items():
        if not dotted.startswith(_PROJECTORS) or not isinstance(spec, Mapping):
            continue
        engine = spec.get("engine")
        if engine not in ("driftscan", "general_pointing"):
            continue
        if "beam_alms" in spec:
            continue
        beam = spec.get("beam")
        ref = beam.get("ref") if isinstance(beam, Mapping) else None
        lmax = spec.get("lmax")
        iterations = spec.get("beam_iterations", _DEFAULT_BEAM_ITERATIONS)
        if not isinstance(ref, str):
            continue
        if isinstance(lmax, bool) or not isinstance(lmax, int):
            continue
        if isinstance(iterations, bool) or not isinstance(iterations, int):
            continue
        found[dotted] = (ref, lmax, iterations, engine)
    return found


@register_built("B9")
def _beam_analysed_twice(run: Built) -> Iterable[Finding]:
    """B9: two projectors nominally sharing a beam do not share the array.

    **A WARN, on §3.2(e)'s reading, and the wording does not lead with "wrong
    answer".**  ``examples/driftscan_mmode.py`` builds two projectors from one
    beam on purpose -- they differ in ``uniform_sampling`` -- so two projectors
    over one beam is a legitimate thing to want, and the honest framing is one
    analysis paid twice plus a gradient into the beam map that splits.  (That
    example is a Python script rather than a config document: it calls
    ``DriftScanProjector.from_beam_maps`` directly and declares no
    ``resources.projectors``, so this check cannot reach it at all.  Checked.)

    **The criterion is identical ``lmax`` and ``beam_iterations``, and
    ``nside`` is not in it** (§0.3 E.6).  ``nside`` is *explicitly refused* for
    ``engine: driftscan`` -- ``from_beam_maps`` infers it from the map length
    -- so a criterion naming it would never fire on the engine it is mostly
    about; and the key is ``beam_iterations``, not ``iterations``.  Those two
    numbers are exactly what ``_analyse`` is called with, so equal pairs mean
    identical work.  Measured: change ``beam_iterations`` and the two
    ``beam_alms`` are no longer even equal.

    **The criterion is COMPLETE for driftscan, and the mechanism is worth
    stating rather than asserting.**  ``driftscan.py:299`` picks its backend
    with ``ltj = _limtod_jax(bool(kwargs.get("uniform_sampling", False)))``,
    and ``_limtod_jax`` returns the same ``limtod_jax`` module on both
    branches -- it only GATES on a feature.  So
    ``ltj.map2alm_iter(m, nside=nside, lmax=lmax, iterations=iterations)`` is
    byte-identical work whatever ``uniform_sampling`` says, and every other
    key ``build_projector`` forwards (``selfrot_deg``, ``horizon_mask``,
    ``apod_deg``, ``mask_iterations``, ``lst_ref_deg``, ``freq_chunk``) goes to
    the CONSTRUCTOR, after the analysis.  ``(beam, lmax, beam_iterations)`` is
    therefore the whole of what decides the transform, and two entries
    differing only in the rest -- which is exactly what
    ``examples/driftscan_mmode.py``'s pair differs in -- really are doing one
    analysis twice.

    **Then the built objects decide.**  A group of two is only reported when
    the projectors' ``beam_alms`` are NOT the same object.  Bitwise equality
    is not the criterion and could not be: two analyses of one map with the
    same statics are deterministic, so equality is what the wrong answer looks
    like too.

    **The remedy exists on one engine and not on the other, and the message
    says which** (R4).  ``beam_alms:`` is a ``general_pointing`` key only:
    measured, a driftscan entry carrying it earns *"engine: driftscan does not
    take ['beam_alms']"*, so for two driftscan projectors there is no edit
    this sentence could name.  Saying so plainly is the deliverable.
    """
    specs = _analysing(run.document)
    if len(specs) < 2:
        return
    resources = getattr(run.resources, "resources", None) or {}
    groups: dict[tuple[str, int, int], list[str]] = {}
    for dotted, (ref, lmax, iterations, _engine) in specs.items():
        groups.setdefault((ref, lmax, iterations), []).append(dotted)
    for (ref, lmax, iterations), members in groups.items():
        if len(members) < 2:
            continue
        alms = [getattr(resources.get(dotted), "beam_alms", None)
                for dotted in members]
        if any(one is None for one in alms):
            continue
        if all(one is alms[0] for one in alms[1:]):
            continue  # already one analysis: nothing was paid twice
        # The engine of the entry being ADVISED, not the set in the group:
        # the advice tells that entry to write `beam_alms:`, so what it
        # additionally needs is a property of it alone. A mixed pair advised
        # on its general_pointing member needs no `nside:`; the same pair
        # advised on its driftscan member does.
        advised_engine = specs[members[1]][3]
        yield warn(
            "B9", members[1],
            f"{members[1]} and {members[0]} both analyse {ref} at lmax="
            f"{lmax} with beam_iterations={iterations}, and this run's two "
            "beam_alms are not the same array -- so the identical spherical "
            f"harmonic transform ran {len(members)} times. "
            f"{_remedy(advised_engine, members[0])} {_B9_TAIL} (check B9).")


def _remedy(advised_engine: str, first: str) -> str:
    """B9's advice, which exists on one engine and not on the other.

    Both engines take ``beam_alms:``, so the second entry can point at the
    first's and the analysis runs once -- measured, that route gives
    ``is``-identity.

    **This used to be two branches, and the second one said there was no
    edit at all.**  That was true when it was written: ``beam_alms:`` was a
    ``general_pointing`` key, and a driftscan entry carrying it earned
    ``engine: driftscan does not take ['beam_alms']`` two gates later, so
    advising it would have been the R4 loop this file exists to avoid.  A8.6
    opened the route, and the sentence became the last thing standing between
    a user and a remedy that now exists -- which is the failure mode a
    "measured" note is most prone to: the measurement stays quotable long
    after the thing it measured has changed.  ``DriftScanProjector`` always
    carried the field; only the config grammar did not.

    The one asymmetry left is real and is stated in the advice rather than
    branched on: alms carry no pixel count, so a driftscan entry taking them
    must write ``nside:``, where the ``beam:`` route infers it from the map
    length.

    The clause turns on the engine of the entry being ADVISED rather than on
    the set of engines in the group, which is what makes a MIXED pair -- one
    driftscan and one general_pointing over the same beam, legal, and driven
    by ``test_a_MIXED_engine_group…`` -- come out right rather than merely
    safe. Advised on its general_pointing member it needs no ``nside:``;
    advised on its driftscan member it does. The old code sent every mixed
    group down the driftscan branch and called that conservative, which it
    was, because the branch said there was nothing to do.
    """
    advice = (f"Write beam_alms: {{ref: {first}.beam_alms}} on the second "
              "entry: measured, that route analyses the beam once and hands "
              "both projectors the same array.")
    if advised_engine == "driftscan":
        return advice + (" On engine: driftscan write nside: too -- alms "
                         "carry no pixel count, so the resolution the beam "
                         "route infers from the map length has to be "
                         "declared on this one.")
    return advice
