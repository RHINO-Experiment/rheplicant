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
