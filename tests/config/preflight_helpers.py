"""The one document every pre-flight test is a patch of.

``tests/config/exit_helpers.py`` is "the only place an exit document is
built"; this is the same contract for the pre-flight pass, and it is a
DELEGATION rather than a second builder.  :func:`preflight_document` calls
``exit_helpers.conjugate_document``, so the twin repair (``exit_helpers``'
``_repaired``: the stochastic ``noise`` node stays in ``model:`` and is
repaired away in ``inference.twin.without:``) comes with it, and
``tests/config/test_config_fixture_contract.py``'s property walk holds this
module to the same standard as every other -- its row in ``_BUILDER_FLOOR``
is what makes the walk cover it.

**Why delegate rather than build a minimal dict.**  A pre-flight check reads
``model:``, ``inference:``, ``runs:``, ``observation:`` and ``resources:``.
A hand-rolled document with three of those would make every check that reads
the other two vacuously clean, and the test would still be green.  The
delegated document carries all eight sections (measured: ``inference``,
``model``, ``observation``, ``resources``, ``runs``, ``runtime``,
``schema_version``, ``variants``), so a check that finds nothing has actually
looked.

**The patch is one level deep, deliberately.**  ``preflight_document(model={
"gian": {}})`` ADDS a node to the base model; it does not replace the block.
That is what almost every check wants -- a valid document with one thing wrong
in it -- and it is what keeps the rest of the document able to discriminate.
A value of ``None`` deletes the section, which is how a check that fires on an
ABSENT section is reached.
"""

import copy

from rheplicant.config.findings import Finding
from rheplicant.config.preflight import preflight
from tests.config.exit_helpers import (
    HOMOSCEDASTIC,
    ONE_LATENT,
    conjugate_document,
)

#: A ``resources:`` patch whose beam file does not exist.  Loading a document
#: carrying it reaches ``build_resources`` (``document.py:75``, measured at
#: Task 3; ``:104`` before Task 2 put the hook above it) and refuses
#: with ``No file at 'no_such_beam.npy'.`` -- measured, 0.115 s to that
#: refusal.  It is what the phase guard puts in front of the pass: a document
#: that is expensive-and-broken in a way that has nothing to do with the
#: violation under test.
UNREADABLE_BEAM = {
    "beams": {"horn": {"format": "npy", "path": "no_such_beam.npy",
                       "nside": 4, "normalize": "pixel_sum",
                       "frame": "beam_local"}},
}

#: THE BASE MUST EARN NO FINDING OF ITS OWN.  ``exit_helpers._repaired``
#: writes ``observed.twin: fit`` beside ``twin.without: [noise]`` whenever the
#: observed block does not name a twin of its own, and that PAIR is A42's
#: shipped condition (plan §3.2(h)2): data simulated through a twin the
#: stochastic node left.  §3.2(b) offers :func:`ids` as "what 'and nothing
#: else' reads"; a base that is itself a finding makes that accessor never
#: return empty, with nothing recording it.  Declaring ``twin: full`` here is
#: what keeps ``_repaired`` from supplying ``fit`` -- both of its keys are
#: DEFAULTS, so a caller that names one keeps it.  ``inference.twin.without``
#: STAYS: §3.2(g)'s twin-repair contract rides on that key, not on
#: ``observed.twin``, and the fixture-contract guard's ``twin is not
#: fit_twin`` is unaffected (measured: False either way).
_OBSERVED = {**ONE_LATENT["observed"], "twin": "full"}
_INFERENCE = {**ONE_LATENT, "observed": _OBSERVED}


def _base() -> dict:
    """The untouched document, fresh and UNSHARED, each call.

    Built through ``conjugate_document``'s ``inference=`` keyword rather than
    by writing ``doc["inference"]`` afterwards: the second form is a depth-1
    replacement of a delegated block, which is exactly what
    ``test_config_fixture_contract._rolls_its_own``'s route B is written to
    catch, and being *sanctioned* for it is weaker than not doing it.
    ``_repaired`` still runs over the block, so ``twin: {without: [noise]}``
    arrives the usual way.

    **Deep-copied, and that is not belt-and-braces.**  Measured:
    ``conjugate_document`` copies its inference block one level
    (``_repaired`` does ``dict(block)``), so without this
    ``preflight_document()["inference"]["parameters"] is
    exit_helpers.ONE_LATENT["parameters"]`` -- a test that edits a latent in
    place rewrites the module constant every other document in the session is
    built from, and the failures land nowhere near the cause.  Tasks 3-12 all
    patch documents; the copy is what makes "a valid document with one thing
    wrong in it" true of one document rather than of the process.
    """
    return copy.deepcopy(conjugate_document({"kind": "forward"},
                                            inference=_INFERENCE))


#: The base document's ``model:`` and ``observation:`` sections, so a test can
#: write ``model={**BASE_MODEL, "noise": NOISE}``.  Named BASE_* rather than
#: MODEL/OBSERVATION because ``tests/config/inference_helpers.py:20`` already
#: binds a MODEL with different contents, and a second one under
#: ``tests/config/`` is the shadowing ``exit_helpers.py:54-58`` named its own
#: CONJUGATE_MODEL to avoid.
BASE_MODEL = dict(_base()["model"])
BASE_OBSERVATION = dict(_base()["observation"])

#: A ``model:`` lighting a node whose class declares randomness -- A30's
#: subject.  **Measured EQUAL to** :data:`BASE_MODEL`: every document this
#: module delegates to already carries ``exit_helpers.MODEL_NOISE`` at
#: ``noise`` (``{type: NoiseOperator}``), because ``_repaired``'s whole point
#: is a stochastic node in ``model:`` repaired away in
#: ``inference.twin.without:``.  So this RENAMES the base model rather than
#: adding to it, and it is bound anyway for one reason: a test written against
#: ``BASE_MODEL`` would not say what its subject is, and
#: ``test_the_base_model_is_the_one_A30_is_about`` is what goes red the day
#: the helper stops carrying the node -- under which every A30 positive test
#: would keep passing while testing nothing at all.  Plan §4's Task 11 spells
#: it ``{**BASE_MODEL, "noise": {...}}``; written that way it is a second
#: literal that must track ``MODEL_NOISE`` by hand, which is the duplication
#: this file exists to avoid.
STOCHASTIC_MODEL = dict(BASE_MODEL)

#: ``model.noise`` under ``RadiometerNoiseOperator``, WITHOUT the ``type:``
#: key -- callers supply that, because ``inference.twin.replace`` and
#: ``model:`` want the same fields under different spellings.  The key set is
#: load-bearing: measured with ``dataclasses.fields``,
#: ``RadiometerNoiseOperator`` takes exactly ``channel_width`` and
#: ``integration_time``, and a third key here would be refused by the delivery
#: layer while a missing one would refuse the REPLACEMENT rather than
#: exercising A30's ``replace:`` leg.
RADIOMETER_NODE = {
    "channel_width": {"value": 1.0, "unit": "MHz"},
    "integration_time": {"value": 2.0, "unit": "s"},
}

#: The base model with the receiver's ``bandpass`` node lit as well as its
#: ``gain`` -- the pair A33 is about.  ``ReceiverOperator``'s only field is
#: ``bandpass`` (measured with ``dataclasses.fields``).
BANDPASS_MODEL = {**BASE_MODEL,
                  "bandpass": {"bandpass": {"ones": ["n_freq"]}}}

#: An ``inference:`` patch freeing one latent into ``bandpass`` and one into
#: ``gain``, with no identifiability convention on either -- A33's document.
#: The latents are deliberately DIFFERENT names: one latent written into both
#: leaves is one parameter and no null direction, which is a case A33 must
#: stand down on rather than refuse.
BANDPASS_AND_GAIN = {
    "parameters": {"b": {"init": {"ones": ["n_freq"]},
                         "into": "bandpass.bandpass"},
                   "g": {"init": 1.0, "into": "gain.gain"}},
    "noise": HOMOSCEDASTIC,
}


def preflight_document(**patch):
    """The valid base document, with ``patch`` merged one level deep.

    Each keyword is a section name.  A mapping is merged INTO that section
    (so ``model={"gian": {}}`` keeps ``global_signal``, ``uniform_sky``,
    ``gain`` and ``noise``); anything else replaces it; ``None`` removes it.

    Takes no positional argument, which is load-bearing:
    ``test_config_fixture_contract._build`` probes the signature for a
    NO-ARGUMENT call first and falls back to its own ``_FORWARD`` run only
    when that raises ``TypeError``.  ``(**patch)`` binds with no arguments, so
    the property walk builds the base document and holds it to the twin-repair
    contract rather than overriding the run.

    The outcome for THIS builder is the same under either probe order --
    measured, ``bind({"kind": "forward"})`` against ``(**patch)`` raises "too
    many positional arguments" -- but the order itself is not free, and Plan
    3A's Task 3 reversed it: five builders in ``exit_helpers`` and
    ``posterior_helpers`` merge the run they are handed OVER their own
    default, so driving them with a bare forward run left their default's
    ``names:`` and ``seed:`` on a ``kind: forward``, which ``A1.runs`` now
    sweeps.
    """
    doc = _base()
    for section, value in patch.items():
        if value is None:
            doc.pop(section, None)
        elif isinstance(value, dict) and isinstance(doc.get(section), dict):
            doc[section] = {**doc[section], **value}
        else:
            doc[section] = value
    return doc


def repatch(document, **sections):
    """``document`` with whole SECTIONS replaced, by copy and without loading.

    Deliberately NOT :func:`preflight_document`'s one-level-deep merge: this
    is what a product test uses to say a key is ABSENT, and a merge cannot
    express a removal.  ``None`` removes the section outright, as it does
    there.

    It exists for cost.  :func:`preflight_document` deep-copies a delegated
    document -- measured at 0.6 ms -- which a cartesian product of tens of
    thousands of cells turns into a minute of fixture around a second of
    subject.  Built ONCE and repatched per cell, the same product costs the
    subject and nothing else.

    It lives HERE and not in a test module for the reason §3 gives: a
    document assembled beside the tests that read it is outside
    ``test_config_fixture_contract``'s census, and 2D shipped that census
    after four modules rolled their own builder and 86 of 90 tests went
    blind.  No ``_document`` suffix, because that suffix is what
    ``_builders()`` discovers and the property walk then drives with no
    argument -- and this one takes a document.
    """
    patched = dict(document)
    for name, value in sections.items():
        if value is None:
            patched.pop(name, None)
        else:
            patched[name] = value
    return patched


# --- Plan 3C Task 2's constants ---------------------------------------------
#
# Appended at the foot and each named for its OWN subject, per §3.2(f): a
# generic name here is a name the next task reaches for and quietly widens.
# Nothing above this line is edited.

#: An ``inference.checks:`` patch declaring one legal skip -- the shape A37 is
#: about when its ``reason:`` is taken away, and the shape the phase assertion
#: puts BESIDE :data:`UNREADABLE_BEAM`.  ``linearity`` and not one of the other
#: two because it is the only check on by default (``gating.DEFAULT_MODE``), so
#: a skip of it is a document that actually changes what runs.
CHECKS_SKIP = {"linearity": {"mode": "skip", "reason": "campaign"}}

#: ``model.noise`` drawn by the radiometer operator -- C18.kind's other
#: drawing type.  :data:`RADIOMETER_NODE` above is the same fields WITHOUT the
#: ``type:``, because ``inference.twin.replace`` and ``model:`` want them under
#: different spellings; this is the ``model:`` spelling.
RADIOMETER_DRAWN = {"type": "RadiometerNoiseOperator", **RADIOMETER_NODE}

#: Three switch positions, so ``context.shape_scope.n_source`` is 3 rather
#: than the base document's 1.  C15's ``min(n_source, k)`` cap is invisible on
#: a one-load document, where the product is ``min(1, k) * n_freq`` whatever
#: ``k`` is.
NOISE_WAVE_SWITCHING = {"mode": "cycle", "order": ["antenna", "ambient",
                                                   "hot"]}

#: The two calibration loads :data:`NOISE_WAVE_SWITCHING`'s order names.
NOISE_WAVE_LOADS = {"ambient": {"t_load": {"value": 300.0, "unit": "K"}},
                    "hot": {"t_load": {"value": 400.0, "unit": "K"}}}

#: A ``model:`` PATCH lighting the ``noise_wave`` node -- C15's subject.  A
#: patch and not a whole model, because ``preflight_document`` merges one
#: level deep and a test that wants a basis beside it writes
#: ``{**NOISE_WAVE_MODEL, "t_sys_extra": NOISE_WAVE_BASIS}``.
#:
#: The four temperatures are written out separately rather than defaulted: C15
#: counts which of them a LATENT frees, and a node that omitted one would make
#: the freed-set assertions read against a field that is not there.
NOISE_WAVE_MODEL = {
    "noise_wave": {"type": "NoiseWaveOperator",
                   "t_unc": {"value": 1.0, "unit": "K"},
                   "t_cos": {"value": 1.0, "unit": "K"},
                   "t_sin": {"value": 1.0, "unit": "K"},
                   "t_rx": {"value": 1.0, "unit": "K"},
                   "gamma_src_re": {"zeros": ["n_source", "n_freq"]},
                   "gamma_src_im": {"zeros": ["n_source", "n_freq"]},
                   "gamma_rec_re": {"zeros": ["n_freq"]},
                   "gamma_rec_im": {"zeros": ["n_freq"]}},
}

#: A ``model.t_sys_extra`` entry of the type C15 declines under.  **Its
#: ``graph_node`` is ``t_sys_extra`` and NOT ``noise_wave``** (measured), which
#: is why the detector cannot look under the node the check is otherwise about.
#:
#: **A LIST, not a label-keyed mapping** (MAJOR 4 fix).  ``t_sys_extra`` IS a
#: ``many`` node, but only ``cal_loads`` is FAN-shaped (a label-keyed
#: mapping) -- ``foregrounds``, ``t_sys_extra`` and ``filters`` are
#: SUM/CHAIN-shaped and take a non-empty LIST instead
#: (``many_shape_problem``, ``sections/compose.py``).  Measured: the old
#: mapping-shaped fixture was refused at check A6 --
#: ``model.t_sys_extra: is a non-empty list (SUM); got dict`` -- and every
#: C15 test built on it passed only because ``axis_findings`` reaches the
#: axes pass alone and never runs the text pass A6 lives in.
#:
#: **Every field here is a real value node and the class is real**, so a
#: document built on this fixture is one ``load_document`` accepts.
#: ``time_basis``/``freq_basis`` are ordinary array leaves, not object
#: fields (``sections/model.py::_object_fields`` names only ``sky_model``
#: and ``projector``), so they cannot take a ``{ref: resources.bases.<n>}``
#: -- they are written as literal ``ones`` arrays sized off the run's own
#: ``n_time``/``n_freq`` symbols, so they build against whatever grid the
#: calling document declares.  ``coeff``'s shape, ``(2, 3)``, is exactly
#: what those two design matrices take: 2 time functions by 3 frequency
#: ones (``BasisTemperatureOperator.__check_init__``).
NOISE_WAVE_BASIS = [
    {"type": "BasisTemperatureOperator",
     "coeff": {"zeros": [2, 3], "unit": "K"},
     "time_basis": {"ones": ["n_time", 2]},
     "freq_basis": {"ones": ["n_freq", 3]}},
]


def findings(document) -> tuple[Finding, ...]:
    """Everything the pass found on ``document``, in run order."""
    return preflight(document).findings


def refusals(document) -> tuple[Finding, ...]:
    """The findings on ``document`` that stop it being run, in run order."""
    return preflight(document).refusals()


def ids(document) -> frozenset[str]:
    """The check ids that fired -- what an "and nothing else" assertion reads."""
    return preflight(document).checks()


def only(document, check: str) -> Finding:
    """The one finding ``check`` produced, asserting there is exactly one.

    A test that reaches for ``[0]`` of a filtered list passes when a check
    fires TWICE on one document, which is a real defect (a loop over nodes
    that forgot to break) and one no ``in`` assertion can see.
    """
    found = [one for one in findings(document) if one.check == check]
    assert len(found) == 1, (
        f"{check} produced {len(found)} findings on this document, not one: "
        f"{[one.where for one in found]}"
    )
    return found[0]


# --- Plan 3C Task 4's constants ---------------------------------------------
#
# C12/C13/C19 and the C14 auto-skip.  Appended at the foot and each named for
# its OWN subject, per §3.2(f).  Every name here begins ``LINEAR_``,
# ``NONLINEAR_``, ``COMPLEX_``, ``INTEGER_`` or ``T4_`` so that Tasks 5 and 6,
# which append to this same file, cannot collide with one.  Nothing above this
# line is edited.

#: :data:`~tests.config.exit_helpers.NONLINEAR_LATENT`'s ``w`` with the claim
#: TAKEN AWAY -- the same document, the same non-affine ``global_signal.width``,
#: and no ``linear: true`` for C12 to check.  It is the anti-vacuity partner of
#: every C12 refusal test: the refusal must come from the DECLARATION and not
#: from the model, and a check that refused this document would be refusing
#: physics nobody claimed anything about.
NONLINEAR_NOT_DECLARED = {
    "parameters": {"w": {"init": 5.0, "into": "global_signal.width",
                         "prior": {"normal": {"loc": 5.0, "scale": 1.0}}}},
    "noise": HOMOSCEDASTIC,
    "observed": {"from": "simulation", "at": {"w": 6.0}},
}

#: Two latents SUMMED into one leaf -- an exactly degenerate pair, by
#: construction rather than by luck.  ``gain.gain = g1 + g2``, so the data
#: constrains the sum and nothing else: measured, ``rank 1`` of ``n_par 2``
#: over 128 data points, ``nullity 1``, ``participation(0) == {'g1': 0.5,
#: 'g2': 0.5}``.
#:
#: **Neither latent is declared ``linear: true``**, deliberately: ``linearity``
#: is the one check on by default, so a ``linear: true`` here would put a C12
#: finding beside every C13 assertion and make "and nothing else" read against
#: the wrong list.  The prediction IS affine in both -- the claim is simply not
#: made.
#:
#: **A WHOLE inference block, for** :func:`repatch` **and not**
#: :func:`preflight_document`: the latter merges one level deep, so the base's
#: ``g`` would survive beside ``g1``/``g2`` and the pair would no longer be the
#: whole space.  That is why ``twin: {without: [noise]}`` is written out here --
#: ``exit_helpers._repaired`` supplies it on the delegated path and a repatch
#: goes round it.
T4_DEGENERATE_PAIR = {
    "parameters": {
        "g1": {"init": 0.55, "prior": {"normal": {"loc": 0.55, "scale": 0.5}}},
        "g2": {"init": 0.55, "prior": {"normal": {"loc": 0.55, "scale": 0.5}}},
    },
    "bindings": [{"latents": ["g1", "g2"], "into": "gain.gain",
                  "transform": {"python": "jax.numpy:add",
                                "fan": "broadcast"}}],
    "noise": HOMOSCEDASTIC,
    "observed": {"from": "simulation"},
    "twin": {"without": ["noise"]},
}

#: The base document's inference block with ``observed:`` REMOVED -- the one
#: shape on which ``inference.observed`` is ``None`` while ``inference.space``
#: is not.  C19 needs data and stands down here; C12 and C13 do not and still
#: run, which is what makes the stand-down C19's own rather than the document's.
#:
#: Derived from :func:`_base` rather than written out, so the twin repair and
#: the noise block travel with it.  **For** :func:`repatch`: a one-level merge
#: cannot express a removal (:func:`preflight_document`'s own docstring).
T4_NO_OBSERVED_INFERENCE = {key: value
                            for key, value in _base()["inference"].items()
                            if key != "observed"}

#: The base document's inference block with its ONE observation replaced by
#: TWO named ones, neither called ``primary``.  ``sections/observed.py:266-271``
#: names a ``primary`` entry, falls back to the single entry when there is
#: exactly one, and otherwise leaves ``ObservedBuild.primary`` **None** -- so
#: this is the shape on which ``observed`` is not ``None`` and its ``primary``
#: is, which is the second half of C19's stand-down and the only one a
#: document can reach.  Measured: ``entries == ['day', 'night']``,
#: ``primary is None``.
#:
#: Without the ``primary is None`` half of the guard, C19 evaluates
#: ``observed.entries[None]`` and dies as ``post-flight check 'C19' RAISED
#: KeyError: None`` -- laundered blame.  **For** :func:`repatch`.
T4_TWO_NAMED_OBSERVATIONS = {
    **{key: value for key, value in _base()["inference"].items()
       if key != "observed"},
    "observed": {"night": dict(_base()["inference"]["observed"]),
                 "day": dict(_base()["inference"]["observed"])},
}

#: An ``inference:`` PATCH turning the noise off.  ``decided_noise`` returns
#: ``None`` for ``kind: none`` (``sections/noise.py:318``) and
#: ``as_noise_model(None, ...)`` is a ``TypeError`` out of the package, so C19
#: stands down here rather than handing it over.  A patch and not a block: the
#: one-level merge keeps the base's ``g``, its priors and its observed data, so
#: the ONLY thing that differs from the passing document is the noise.
T4_NOISE_NONE = {"noise": {"kind": "none"}}

#: An ``inference:`` patch whose latent asks for a COMPLEX init.  It is
#: **accepted**, and that is the whole point: ``sections/parameters.py:162``
#: casts every ``init`` to ``context.dtype``, which ``RuntimeFacts.dtype``
#: restricts to ``float32``/``float64``, so the complex value arrives as a
#: ``float32`` latent with a ``ComplexWarning`` and nothing downstream ever
#: sees a complex dtype.  ``dtype: int32`` is not even writable --
#: ``modifiers.DTYPES`` holds four names and none of them is an integer.
#:
#: This is the fixture behind the one end-to-end C14 test: it is what goes red
#: the day the config layer starts admitting a complex latent, at which point
#: the four unit tests over a doctored build stop being hypothetical.
COMPLEX_INIT_LATENT = {
    "parameters": {"g": {"init": {"value": 1.0, "dtype": "complex64"},
                         "linear": True, "into": "gain.gain",
                         "prior": {"normal": {"loc": 1.0, "scale": 0.5}}}},
}

#: All three gates at ``mode: refuse`` -- the anti-vacuity partner of the
#: call-count property.  With no ``inference.checks:`` at all the counts are
#: ``(one per linear latent, 0, 0)``; with this they are ``(n, 1, 1)``, and a
#: default table quietly reversed cannot satisfy both.
T4_CHECKS_ALL_REFUSE = {"linearity": {"mode": "refuse"},
                        "identifiability": {"mode": "refuse"},
                        "prior_sensitivity": {"mode": "refuse"}}

#: ``identifiability`` turned on at ``mode: refuse``.  Off by default, so
#: every C13 test has to write something.
T4_CHECKS_IDENTIFIABILITY_REFUSE = {"identifiability": {"mode": "refuse"}}

#: ``identifiability`` turned on and asked to record its numbers on a PASS.
T4_CHECKS_IDENTIFIABILITY_REPORT = {"identifiability": {"mode": "report",
                                                        "report": True}}

#: ``identifiability`` declined in writing -- the shape
#: :meth:`~rheplicant.config.gating.Gate.runs` is false for with a ``reason:``
#: the record keeps.  Used to show that one gate standing down does not silence
#: another.
T4_CHECKS_IDENTIFIABILITY_SKIP = {
    "identifiability": {"mode": "skip",
                        "reason": "the joint rank is checked by hand"}}

#: ``identifiability`` with its own ``rtol:`` -- the ONLY check whose entry may
#: carry one (``gating.check_gates``'s ``allowed`` set).  ``1e-2`` is four
#: decades above ``DEFAULT_RANK_RTOL`` and is chosen to be visibly not the
#: default in the recorded numbers.
T4_CHECKS_IDENTIFIABILITY_RTOL = {"identifiability": {"mode": "report",
                                                      "report": True,
                                                      "rtol": 1e-2}}

#: ``prior_sensitivity`` turned on and asked to record its numbers on a PASS.
#: **This is the one fixture in Task 4 that pays the real cold cost** of the
#: two Newton solves; every other C19 test either stands down before the call
#: or drives a stub.
T4_CHECKS_PRIOR_SENSITIVITY_REPORT = {
    "prior_sensitivity": {"mode": "report", "report": True}}

#: ``linearity`` downgraded -- the first escape C12's own refusal names, applied
#: literally by the advice-loop test.
T4_CHECKS_LINEARITY_WARN = {"linearity": {"mode": "warn"}}

#: ``linearity`` declined in writing -- the second escape C12's refusal names.
#: Distinct from :data:`CHECKS_SKIP` (Task 2's) only in the ``reason:``, and
#: kept apart because the advice-loop test asserts the sentence a reader would
#: actually have copied out of the refusal.
T4_CHECKS_LINEARITY_SKIP = {
    "linearity": {"mode": "skip",
                  "reason": "this block's linearity is checked in the campaign "
                            "notebook"}}

#: ``linearity`` asked to record its margins on a PASS.  Without ``report:
#: true`` a passing check says nothing at all (§2.3's table, rows 3, 6 and 9),
#: so this is the only way the margins reach the record.
T4_CHECKS_LINEARITY_REPORT = {"linearity": {"mode": "refuse", "report": True}}
