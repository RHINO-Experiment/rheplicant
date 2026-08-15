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
from tests.config.exit_helpers import ONE_LATENT, conjugate_document

#: A ``resources:`` patch whose beam file does not exist.  Loading a document
#: carrying it reaches ``build_resources`` (``document.py:104``) and refuses
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


def preflight_document(**patch):
    """The valid base document, with ``patch`` merged one level deep.

    Each keyword is a section name.  A mapping is merged INTO that section
    (so ``model={"gian": {}}`` keeps ``global_signal``, ``uniform_sky``,
    ``gain`` and ``noise``); anything else replaces it; ``None`` removes it.

    Takes no positional argument, which is load-bearing:
    ``test_config_fixture_contract._build`` binds its own default run against
    the signature first and falls back to a NO-ARGUMENT call when that raises
    ``TypeError`` -- measured, ``bind({"kind": "forward"})`` against
    ``(**patch)`` raises "too many positional arguments", so the property walk
    builds the base document and holds it to the twin-repair contract rather
    than overriding the run.
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
