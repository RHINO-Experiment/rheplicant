"""A29: which kinds must carry a seed, and which subsections.

Short, and its own module because it was the piece that put the counts file
at 796 lines -- four short of the limit, which is not a margin.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register

from .fitting_vocabulary import (
    _runs,
)

#: Which ``runs[].kind`` needs a seed on the RUN.  ``npe`` is absent on
#: purpose: it draws four times and declares its seeds per subsection in
#: ``inference.npe:`` (``npe._seeded``, ``exits.py::_SAMPLE_KEYS``), and a run-level
#: ``seed:`` on it is refused rather than required.  ``condition`` is absent
#: on purpose too: ``_CONDITION_KEYS`` (``conjugate.py``) carries ``seed``
#: and ``_run_condition`` reads it only ``if "seed" in run.options``
# : (``conjugate.py::_a51_condition_takes_no_prior_mean``), so it is OPTIONAL there and correctly
# outside A29 --
#: ``condition_estimate``'s ``key`` defaults internally, which that function's
#: own docstring argues at ``conjugate.py::_parse_gcr``.
_A29_SEEDED_KINDS: frozenset[str] = frozenset({"plan.sample", "conjugate.gcr", "nuts"})

#: The subsections of ``inference.npe:`` that declare a seed, in the order
#: ``parse_npe`` (``sections/npe.py::_train``) reads them.  ``embed:`` is the one
#: member of ``_NPE_KEYS`` (``sections/npe.py``) missing here, and it is missing
#: because it declares no seed -- measured, ``_seeded`` is called at
# : ``inference/npe.py::NeuralPosterior.sample``, ``inference/npe.py``,
# ``inference/npe.py::train_posterior`` and ``inference/npe.py::train_posterior`` and nowhere else.
# (It is
#: written second in that frozenset, which orders nothing; the order below is
#: ``parse_npe``'s.)
_A29_NPE_SUBSECTIONS: tuple[str, ...] = ("bank", "create", "train", "sample")


@register("A29")
def _seeds(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A29: the seed asymmetry, decided from the document's text.

    A MOVE, not a rewrite.  All five routes already refuse, all five already
    say ``ConfigError``, and all five messages are reproduced here by CALLING
    the function that holds them rather than by restating it -- three of them
    lifted to module level in their own sections for exactly that purpose
    (``exits._a29_estimate_takes_no_seed``,
    ``conjugate._a29_gcr_needs_a_seed``, ``npe._a29_npe_takes_no_run_seed``),
    and the other two through ``draws._seed_name`` (``inference/npe.py::simulate_pairs``), which was
    already pure: it takes a dict and a prefix, refuses a missing key, a
    literal seed and a name outside ``runtime.seeds.``, and resolves nothing.
    The RESOLUTION (``seed_for``) stays where the context is.

    **Not byte for byte on every route**, and the difference is §3.2(c)'s.
    The three lifted refusals already carry ``(check A29)`` mid-sentence and
    are re-emitted verbatim; the two that come through ``_seed_name`` carry
    no tag at all, so those two -- and every ``inference.npe.<sub>`` finding
    -- get ``" (check A29)."`` APPENDED. Appending is safe because measured,
    all 442 ``pytest.raises(ConfigError, match=...)`` assertions in
    ``tests/config/`` are searches and none is anchored with ``$``.

    What moves is the phase.  ``nuts``'s seed is checked at ``nuts.py::_parse_nuts``,
    i.e. AFTER ``to_numpyro_model`` is built at ``nuts.py::_parse_nuts`` -- the most expensive
    object that executor makes is constructed before the cheapest key on the
    run is looked at -- and all five sit behind ``build_resources``, which is
    90.9 % of ``load_document`` (§2.7).

    Two things are deliberately NOT A29's, both measured:
    ``inference.observed.<name>.realise.seed`` (``observed.py::_predicted_shape``) and a
    ``{normal: {seed: ...}}`` value node (``draws.py::_resolve_operand``) also go through
    ``_seed_name``, and neither is a ``runs[].kind`` -- schema §6's A29 row
    is about the run kinds and the npe subsections, and answering for the
    other two here would put this check in front of two grammars that own
    their own sentences.

    **And one leg of A29 is NOT hoisted, which is a hole in this plan's own
    thesis rather than a decision about wording.**  An ABSENT
    ``inference.npe.<sub>`` is left to ``npe._subsection`` (``draws.py::_draw``),
    whose sentence is that the subsection is required rather than that a seed
    is missing -- but ``parse_npe`` runs inside ``build_inference``, which is
    after ``build_resources``, so that refusal still costs the beam.  Closing
    it means A29 answering for a section the user has not written, which is a
    worse sentence.  **This docstring is the record**, and §6's residue list
    is where it was copied to; an earlier wording ("recorded for §6's ledger")
    read as a citation to a §6 entry that did not exist at the time.
    """
    from rheplicant.config.draws import _seed_name
    from rheplicant.config.sections.conjugate import _a29_gcr_needs_a_seed
    from rheplicant.config.sections.exits import _a29_estimate_takes_no_seed
    from rheplicant.config.sections.npe import _a29_npe_takes_no_run_seed
    from rheplicant.config.sections.runs import _RUN_KEYS

    for index, run in enumerate(_runs(document)):
        kind = run.get("kind")
        if not isinstance(kind, str):
            continue
        # `expect: refuse` is an assertion ABOUT the refusal and a P-1 one
        # cannot be captured -- `_prior_gates`' docstring argues it at length.
        if run.get("expect") == "refuse":
            continue
        where = f"runs[{index}]"
        named = f"runs[{run['name']!r}]"
        # `RunSpec.options` is exactly this (`runs.py::_one`), and
        # `_RUN_KEYS` is imported rather than restated -- measured, it is
        # {expect, kind, name, on, reuse, variant}, and a sixth key added
        # there would otherwise reach `_seed_name` as an option here while
        # travelling on the spec at P3.
        options = {key: value for key, value in run.items() if key not in _RUN_KEYS}
        # THE GATE AND `_seed_name` ARE ALTERNATIVES, NOT A PAIR.
        # `conjugate.gcr` is in both `_A29_SEEDED_KINDS` and the gate chain,
        # so a seedless gcr run would be described TWICE -- once by
        # `_a29_gcr_needs_a_seed` and once by `_seed_name`, in two different
        # voices, for one missing key.  `gated` is what makes the bespoke
        # refusal the only one the user reads.  It is set only when the gate
        # FIRED, so a gcr run with a seed of the wrong FORM still reaches
        # `_seed_name`, which is the leg that decides form.
        gated = False
        gate = (
            _a29_estimate_takes_no_seed
            if kind == "plan.estimate"
            else _a29_gcr_needs_a_seed
            if kind == "conjugate.gcr"
            else _a29_npe_takes_no_run_seed
            if kind == "npe"
            else None
        )
        if gate is not None:
            try:
                gate(named, options)
            except ConfigError as refusal:
                # No `(check A29)` tail: all three of these messages already
                # carry one mid-sentence, and §3.2(c) appends only when the
                # message does not.
                yield refuse("A29", where, str(refusal))
                gated = True
        if kind in _A29_SEEDED_KINDS and not gated:
            try:
                _seed_name(options, named)
            except ConfigError as refusal:
                yield refuse("A29", where, f"{refusal} (check A29).")

    inference = document.get("inference")
    npe = inference.get("npe") if isinstance(inference, Mapping) else None
    # NOT gated on a `kind: npe` run being declared, and the gate that was
    # here rested on the same false premise Task 9 removed from `_counts` one
    # function down. `inference.py::build_inference` is `npe =
    # parse_npe(section["npe"], context) if "npe" in section else None` --
    # unconditional on `runs:` -- so the seed IS read, at P2, and
    # `build_inference` (`preflight/document.py`) runs AFTER `build_resources`
    # (`:75`). Measured on a document whose only run is `kind: forward` and
    # whose `inference.npe:` is otherwise complete: a `train:` with no `seed`
    # was silent here and refused by `_seeded` at P2, and with UNREADABLE_BEAM
    # in the same document the BEAM spoke first -- the one outcome this plan
    # exists to stop. A literal `seed: 7` behaves the same way. Deleting the
    # gate refuses no document the package runs: the same document with a
    # complete section still BUILDS, measured.
    if not isinstance(npe, Mapping):
        return
    for subsection in _A29_NPE_SUBSECTIONS:
        body = npe.get(subsection)
        # An ABSENT or malformed subsection is `npe._subsection`'s
        # (`:230-243`), whose sentence is that the subsection is required
        # rather than that a seed is missing.  Standing down leaves the
        # reader the fault they actually have -- and leaves that leg behind
        # the beam, which the docstring records as a hole.
        if not isinstance(body, Mapping):
            continue
        where = f"inference.npe.{subsection}"
        try:
            _seed_name(dict(body), where)
        except ConfigError as refusal:
            yield refuse("A29", where, f"{refusal} (check A29).")
