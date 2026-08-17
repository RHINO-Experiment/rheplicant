"""The one place an in-flight payload is built.

``exit_helpers.py`` is "the only place an exit document is built" and
``preflight_helpers.py`` is the same contract for the text pass; this is it for
the two in-flight passes, and like ``preflight_helpers`` it is a **DELEGATION
rather than a second builder**.  Every document here comes from
:func:`~tests.config.preflight_helpers.preflight_document`, so the twin repair
(``exit_helpers._repaired``) travels with it and
``tests/config/test_config_fixture_contract.py``'s census holds this module to
the same standard as every other.

**This module defines no ``*_document`` builder**, deliberately.  It builds
PAYLOADS out of documents somebody else assembled, so
``test_config_fixture_contract.py``'s ``_BUILDER_FLOOR`` needs no new row --
that table's contract is "every helper module that DEFINES a builder must
appear", and ``inference_helpers`` is already in the glob with no row for the
same reason.  The module still joins the census automatically through the
``*_helpers.py`` glob, which is the point.

**Why ``built_run`` calls ``_assemble`` and not ``load_document``.**  The built
hook calls ``raise_if_refused()`` before ``load_document`` returns, so
``Built(*load_document(d))`` **cannot observe a built-slot refusal at all** --
the document that earns one never comes back.  Most built-slot rows are
refusals, so a helper built on ``load_document`` could only ever exercise the
passing half of the slot it exists to test.  A test about the HOOK -- that a
refusal really does stop the load -- calls ``load_document`` under
``pytest.raises`` instead, and ``test_config_inflight.py`` ships that pair.
"""

import time

import numpy as np

from rheplicant.config.document import _assemble, _carrying, _priced_payload
from rheplicant.config.findings import Finding
from rheplicant.config.inflight import Axes, Built, axes, built
from rheplicant.config.postflight import Priced, priced
from rheplicant.config.sections.observation import build_observation
from rheplicant.config.sections.runtime import build_runtime


def best_ms(call, repeats: int = 100) -> float:
    """The FASTEST of ``repeats`` runs of ``call``, in milliseconds.

    **The minimum and not the median**, and that is what makes a bound set
    NEAR the measurement safe: contention from fifteen sibling ``-n 16``
    workers can only ADD time, so it moves the median and the maximum and
    leaves the best case alone.  Measured on an idle machine the two agree to
    3 % (``axes`` on the worked document: 0.0132 ms best, 0.0135 ms median),
    so nothing is given up by taking it.

    It lives here rather than in one test module because three of them time
    these passes, and a review of Task 1a found every cost bound in the first
    of them unable to fail -- a *thousandfold* slowdown of the shared
    ``sweep`` left the suite at exit 0, at margins up to x30077.  One helper
    is what lets the three agree on how the number is taken.
    """
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        call()
        samples.append((time.perf_counter() - started) * 1e3)
    return min(samples)


def axis_facts(document, *, base_dir=None) -> Axes:
    """``build_runtime`` + ``build_observation``, and nothing else.

    **NO resource is built**, which is what makes an axes test cost
    microseconds rather than the 90.9 % of ``load_document`` that
    ``build_resources`` is.  A helper that reached for ``load_document`` and
    then took the pieces off the result would silently pay for the beam and
    destroy the only property this slot has.

    ``base_dir`` is additive to the signature §3.1 pins: an ingested
    ``observation.from_file`` and :func:`projector_sections`' beam are both
    relative paths, and without it they resolve against the process's cwd.
    The pinned positional call ``axis_facts(document)`` is unchanged.
    """
    runtime = build_runtime(document["runtime"])
    observation, context = build_observation(document["observation"],
                                             runtime=runtime,
                                             base_dir=base_dir)
    return Axes(document=document, runtime=runtime, observation=observation,
                context=context)


def built_run(document, *, base_dir=None) -> Built:
    """Everything ``load_document`` builds, as a :class:`Built`, WITHOUT the hook.

    Raises whatever the document earns from the pre-flight pass, the axes pass
    or any builder -- so a test about a document that must BUILD calls this,
    and a test about a document those earlier phases must refuse wraps it in
    ``pytest.raises``.  What it does NOT do is run the built pass, which is
    what makes a built-slot refusal observable at all (see the module
    docstring).
    """
    return Built(*_assemble(document, base_dir=base_dir))


def axis_findings(document, *, base_dir=None) -> tuple[Finding, ...]:
    """Everything the axes pass found, in run order."""
    return axes(axis_facts(document, base_dir=base_dir)).findings


def built_findings(document, *, base_dir=None) -> tuple[Finding, ...]:
    """Everything the built pass found, in run order."""
    return built(built_run(document, base_dir=base_dir)).findings


def _only(found, check: str, where: str) -> Finding:
    """:func:`~tests.config.preflight_helpers.only`'s body, shared by the two.

    **Filters on ``one.check`` across ALL THREE severities and asserts exactly
    one.**  A WARN counts as the one: several rows this layer ships are
    warnings, and a refusals-only reading makes every test about them
    unwritable.

    Exactly one, and not ``[0]`` of a filtered list: a check that fires TWICE
    on one document -- a loop over nodes that forgot to ``break`` -- is a real
    defect, and no ``in`` assertion can see it.
    """
    mine = [one for one in found if one.check == check]
    assert len(mine) == 1, (
        f"{check} produced {len(mine)} findings on this document in the "
        f"{where} pass, not one: {[one.where for one in mine]}"
    )
    return mine[0]


def axis_only(document, check: str, *, base_dir=None) -> Finding:
    """The one finding ``check`` produced in the axes pass."""
    return _only(axis_findings(document, base_dir=base_dir), check, "axes")


def built_only(document, check: str, *, base_dir=None) -> Finding:
    """The one finding ``check`` produced in the built pass."""
    return _only(built_findings(document, base_dir=base_dir), check, "built")


def priced_run(document, *, base_dir=None) -> Priced:
    """The post-flight payload, WITHOUT the hook that would raise on it.

    ``_assemble`` for the same reason :func:`built_run` uses it -- the priced
    hook calls ``raise_if_refused()`` before ``load_document`` returns, so a
    payload taken off ``load_document``'s result could only ever exercise the
    passing half of the slot it exists to test, and most priced rows are
    refusals.  A test about the HOOK -- that a refusal really does stop the
    load -- calls ``load_document`` under ``pytest.raises`` instead.

    **The built pass IS run here, and collected rather than raised**, because
    ``Priced.run.report`` promises pre-flight + axes + built and a payload
    whose report stopped at the axes pass would make that promise false in
    exactly the tests written to check it.  ``document.py``'s own
    :func:`~rheplicant.config.document._carrying` and
    :func:`~rheplicant.config.document._priced_payload` are reused rather than
    restated: which section the gates come from is one contract, and a second
    reading of it here is how the two come to disagree.
    """
    run = _assemble(document, base_dir=base_dir)
    return _priced_payload(_carrying(run, built(Built(*run))))


def priced_findings(document, *, base_dir=None) -> tuple[Finding, ...]:
    """Everything the post-flight pass found, in slot order."""
    return priced(priced_run(document, base_dir=base_dir)).findings


def priced_only(document, check: str, *, base_dir=None) -> Finding:
    """The one finding ``check`` produced in the post-flight pass."""
    return _only(priced_findings(document, base_dir=base_dir), check,
                 "post-flight")


def projector_sections(tmp_path, **overrides) -> dict:
    """A ``resources:`` patch: one small npy beam and one driftscan projector.

    nside 4 (192 pixels), lmax 8, two channels -- the smallest thing that
    actually goes through ``from_beam_maps``.  Callers write::

        preflight_document(resources=projector_sections(tmp_path))

    and pass ``base_dir=str(tmp_path)`` to whichever helper they then use.

    **NOT a ``*_document`` builder and deliberately not named like one**: the
    fixture census drives every ``*_document`` with no argument, and this one
    needs a directory to write the beam into.

    **It writes ``acknowledge_float32_sky: true``, and that is load-bearing.**
    A44's condition is ``runtime.jax_enable_x64`` -- absent means float32,
    which means A44 fires **by default**.  The base document is float32, so
    without this key every test that reaches for a projector would carry an
    unrelated A44 finding, and every ``and nothing else`` assertion built on
    one would be about the wrong thing.  It is ``true`` here because these
    documents exist to test something else; a test whose subject IS A44 sets
    it ``false`` through ``overrides``.

    Guarded by ``pytest.importorskip("limtod_jax")`` at its CALL SITES rather
    than here: importing this module must not skip a test that never asks for
    a projector.
    """
    np.save(tmp_path / "beam.npy", np.ones((2, 192)))
    projector = {
        "engine": "driftscan",
        "beam": {"ref": "resources.beams.horn"},
        "lmax": 8,
        "lat_deg": {"value": 53.2367, "unit": "deg"},
        "az_deg": {"value": 0.0, "unit": "deg"},
        "el_deg": {"value": 90.0, "unit": "deg"},
        "normalize_beam": True,
        "acknowledge_float32_sky": True,
    }
    projector.update(overrides)
    return {
        "beams": {"horn": {"format": "npy", "path": "beam.npy", "nside": 4,
                           "normalize": "pixel_sum", "frame": "beam_local"}},
        "projectors": {"drift": projector},
    }
