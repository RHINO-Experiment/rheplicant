"""A20, A23 and T8: priors, joint blocks and traded keys.

What a latent needs declared before a solve can use it, which blocks may be
jointly solved, and the calibrator keys T8 trades. The prior gates are the
long one and they are the reason this is its own module.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register

from .fitting_engines import (
    _engine_of,
)
from .fitting_vocabulary import (
    _T7_GRADIENT,
    _latents,
    _runs,
    _t7_entries,
    _t7_names,
)


def _a20_joint_over(document: Mapping[str, Any]) -> tuple[str, ...]:
    """The latent names ``inference.joint_prior`` covers, or ``()``.

    ``{jeffreys: {over: ...}}`` is the grammar and the only one:
    ``transforms._joint_prior`` (``preflight/document.py::_task3_horizon_in``) refuses anything
    else, and
    ``{kind: jeffreys, names: [...]}`` is refused by name there.  So coverage
    is text in the document and nothing has to be built to read it.

    **``tuple(over)`` is what the package does, and this mirrors it rather
    than narrowing it.**  ``_joint_prior`` ends
    ``JeffreysPrior(over=tuple(body["over"]), **kwargs)``
    (``preflight/document.py::_variant_text``), so FIVE
    shapes besides a list of strings build a real prior -- measured through
    ``run_document``: ``over: da`` and ``over: 'd'`` (a bare YAML scalar,
    split into characters), ``over: {d: 1, a: 2}`` and ``over: {d: 1}``
    (ordinary YAML, iterated as its keys) and ``over: ('d', 'a')``.  An
    earlier version of this reader required a ``list`` and answered ``()`` for
    all five, so ``over: da`` on a ``kind: plan.*`` document reached
    ``ParameterSpaceError: ... no block would step it at all`` **behind the
    beam**, which is the one thing A20 exists to prevent.  Reading what the
    package reads is what closes that.

    Two guards, and both are about the pass rather than about the grammar.
    ``tuple()`` raises ``TypeError`` on ``over: 7``, and inside the pass a
    ``TypeError`` becomes "check A20 RAISED" and discards every other
    finding.  And a member that is not a string cannot be looked up in the
    latents without risking an unhashable key, so a mixed ``over:`` reads as
    no coverage and ``JeffreysPrior.validate_against`` -- which names WHICH
    member is wrong, as A20 does not -- keeps that document.
    """
    inference = document.get("inference")
    if not isinstance(inference, Mapping):
        return ()
    joint = inference.get("joint_prior")
    if not isinstance(joint, Mapping):
        return ()
    body = joint.get("jeffreys")
    if not isinstance(body, Mapping):
        return ()
    try:
        names = tuple(body.get("over"))
    except TypeError:
        return ()
    if not all(isinstance(name, str) for name in names):
        return ()
    return names


def _a23_latents(document: Mapping[str, Any]) -> dict[str, Any]:
    """:func:`_latents`, minus every latent whose BODY the grammar refuses.

    ``_latents`` (Task 7) keeps the NAME of a latent whose body is not a
    mapping and reads the body as ``{}``, deliberately and rightly for A16:
    dropping it would make a block that covers that latent look like a block
    over a name ``inference.parameters`` never declared.  **A23 must not
    inherit that**, and did until this was written.  A ``{}`` body has no
    ``prior:``, so ``w: 7`` reads as prior-free and A23 answers *"declare a
    prior:"* in front of ``sections/parameters.py``'s *"inference.parameters.w:
    is a mapping; got 7"* -- which is the fault the reader actually has.

    ``if body`` is the whole filter and it needs no second reader of
    ``inference.parameters``: measured, every body that reads as empty here
    is one the grammar refuses on its own terms -- ``w: 7``, ``w: ['x']`` and
    ``w: 'oops'`` as *"is a mapping; got ..."*, and a genuinely empty
    ``w: {}`` as *"init: is required"*, because ``init:`` is required of every
    latent.  So an empty body is never a latent A23 could be right about.
    """
    return {name: body for name, body in _latents(document).items() if body}


def _a23_prior_free(
    latents: Mapping[str, Any], names: Iterable[str], covered: tuple[str, ...] = ()
) -> list[str]:
    """The names among ``names`` that declare no prior this route accepts.

    ``Latent.prior`` comes from ``spec.get("prior")`` and from nowhere else
    (``inference/parameters.py::refuse_stochastic_stages``; ``_parse_prior`` returns None for a
    missing key at
    ``inference/parameters.py`` and for no other value), so "does this latent declare a prior"
    is a text question -- and a prior that is present but malformed is a
    DECLARED prior, refused by ``_parse_prior`` in its own words.
    ``covered`` is non-empty only for the ``nuts`` route, which is the one
    route that counts ``inference.joint_prior`` coverage as a prior --
    ``to_numpyro_model`` accepts a covered latent (``numpyro_bridge.py``)
    and ``simulate_pairs`` does not (``inference/npe.py::simulate_pairs``).

    **``names`` must already be names the document DECLARES WELL** -- the
    keys of :func:`_a23_latents`, or a subset of them.  An absent latent reads
    as prior-free here, and on the ``plan.sample`` leg -- the one route whose
    names come from a block rather than from ``inference.parameters`` -- that
    would put an A23 refusal beside A16's *"names 'zzz', which
    inference.parameters does not declare"*: one typo, two refusals, two
    different fixes.  The caller filters; this function cannot, because on the
    other three routes ``names`` IS the declaration.
    """
    return [
        name for name in names if latents.get(name, {}).get("prior") is None and name not in covered
    ]


#: What ``kind: optimize`` and ``kind: plan.estimate`` -- the two exits A23's
#: bare fix clause offers -- accept between them.  IMPORTED rather than
#: restated, for §2.2's one-name-one-binding reason and for a sharper one: a
#: key added to either executor and not here would put a key in A23's "drop
#: these" list that the exit takes, which is the same defect as leaving one
#: out, in the other direction.
def _t8_calibrator_keys() -> frozenset[str]:
    """``_OPTIMIZE_KEYS | _ESTIMATE_KEYS``, deferred the way ``_ENGINES`` is."""
    from rheplicant.config.sections.exits import _ESTIMATE_KEYS, _OPTIMIZE_KEYS

    return frozenset(_OPTIMIZE_KEYS) | frozenset(_ESTIMATE_KEYS)


def _t8_traded_keys(run: Mapping[str, Any]) -> list[str]:
    """The run's own option keys that NEITHER calibrator exit takes.

    A23's bare *"or run one of those"* is an instruction to change ``kind:``
    and nothing else, and on a ``kind: plan.sample`` run that carries
    ``seed:``, ``n_sweeps:`` and ``warmup:`` the result is a document check
    A29 or check A1 refuses -- whose own remedy is to change ``kind:`` back.
    Measured before the fix, in three steps: A23 fired, its ``kind:
    plan.estimate`` fix earned A29's *"Drop it, or make this run
    plan.sample"*, that fix restored the document exactly, and ``step2 ==
    step0`` was True.

    ``_RUN_KEYS`` is what ``RunSpec.options`` excludes (``runs.py::_one``),
    imported for the reason :func:`_seeds` imports it: a sixth key added
    there would otherwise read as an option here.
    """
    from rheplicant.config.sections.runs import _RUN_KEYS

    # Both sets are read ONCE, not once per key: the comprehension runs over
    # a user's mapping and a call inside it would be one import lookup per
    # key on every A23 refusal.
    calibrator = _t8_calibrator_keys()
    return sorted(
        key
        for key in run
        if isinstance(key, str) and key not in _RUN_KEYS and key not in calibrator
    )


def _a23_message(
    named: str,
    kind: str,
    missing: list[str],
    because: str,
    covered: tuple[str, ...],
    traded: list[str],
) -> str:
    """A23's refusal: one shape, and three clauses the document decides.

    **The verb is per route.**  ``kind: fisher`` with ``space: true`` does not
    draw anything -- ``fisher_information`` computes a posterior precision
    (``uncertainty.py::_declared_gaussian_priors``) -- so a shared *"draws a POSTERIOR"* is false
    on one of the four routes, and measured, nothing in the suite could tell.

    **The fix clause is the one that can send a reader into another
    refusal.**  *"or run one of those"* names the calibrator exits, and
    ``kind: plan.estimate`` and ``kind: plan.sample`` are both refused beside
    an ``inference.joint_prior`` by A20 -- so on a covered document the advice
    trades this refusal for that one, which is §2.6 item 4's contradiction
    running the other way.  And where a NAMED latent is itself covered (only
    ``kind: npe`` reaches that, because it ignores coverage by design)
    *"declare a prior:"* is refused by A22, so that branch says so and offers
    the exit that does read coverage as a prior.

    **``traded`` closes the third loop, and it is the live one.**  The two
    joint-prior branches above name what they must not be followed with;
    the bare branch did not, and its own exits refuse the KEYS the run is
    carrying.  Only the first-listed remedy -- *"give each one a prior:"* --
    terminated.  So where the run carries keys neither calibrator exit takes,
    the alternative names them; where it carries none, the bare sentence
    stands, because an edit list nobody needs is noise in a refusal that is
    already four clauses long.
    """
    verb = "computes a POSTERIOR precision" if kind == "fisher" else "draws a POSTERIOR"
    overlap = [name for name in missing if name in covered]
    if not covered and traded:
        # `seed:` is called out separately because it is the one key whose
        # refusal is NOT A1's: `document._TASK3_SPOKEN_FOR` hands
        # `plan.estimate` + `seed` to A29, so a clause naming A1 alone would
        # send the reader to a check that stays silent on exactly the key
        # this loop turned on.
        if traded == ["seed"]:
            instead = "check A29's"
        elif "seed" in traded:
            instead = "check A1's, and check A29's on seed:"
        else:
            instead = "check A1's"
        fix = (
            "A prior-free latent is a free parameter, which the calibrator "
            "exits (kind: optimize, kind: plan.estimate) fit and a "
            "posterior cannot. Give each one a prior:, or run one of those "
            f"AND drop {traded} with the kind -- neither exit takes "
            f"{'them' if len(traded) > 1 else 'it'}, so changing kind: "
            f"alone trades this refusal for {instead}"
        )
    elif not covered:
        fix = (
            "A prior-free latent is a free parameter, which the calibrator "
            "exits (kind: optimize, kind: plan.estimate) fit and a "
            "posterior cannot. Give each one a prior:, or run one of those"
        )
    elif overlap:
        fix = (
            "A prior-free latent is a free parameter, which a posterior "
            f"cannot fit. inference.joint_prior already covers {overlap}, "
            "and a latent may not be covered AND declare a prior: of its "
            "own (check A22), so declaring one is not the fix here: run "
            "kind: nuts, which reads joint-prior coverage as a prior, or "
            "drop inference.joint_prior and give every latent a prior: of "
            "its own. Not kind: plan.estimate or kind: plan.sample -- A20 "
            "refuses both beside a joint prior"
        )
    else:
        fix = (
            "A prior-free latent is a free parameter, which a posterior "
            "cannot fit. Give each one a prior:, or add it to "
            "inference.joint_prior.over, which already covers "
            f"{list(covered)}. Not kind: plan.estimate or kind: "
            "plan.sample -- A20 refuses both beside a joint prior, so "
            "switching would trade this refusal for that one"
        )
    return (
        f"{named}: kind: {kind} {verb}, and inference.parameters declares "
        f"{missing} with no prior: {because}. {fix} (check A23)."
    )


@register("A20", "A21", "A23")
def _prior_gates(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A20, A21 and A23 -- and A20/A21 make two of A23's four legs moot.

    **The order inside this function is the decision §2.6 item 4 records, and
    it is structural rather than positional.**  A20 refuses
    ``inference.joint_prior`` beside ANY ``kind: plan.*``
    (``plan_settings.py::_halves``, unconditional -- ``_refuse_split_joint_prior``
    chooses only its wording from whether the partition splits the prior) and
    A21 refuses it beside ``fisher`` with ``space: true``
    (``uncertainty.py::_declared_gaussian_priors`` -- ``inference/``, not
    ``config/sections/``).  So a run refused by either ``continue``s and never
    reaches its A23 leg: the joint-prior branches of A23's
    ``plan.sample``-gradient and ``fisher(space=)`` legs are UNREACHABLE, not
    merely later.  An implementation that ordered them the other way would
    tell a ``joint_prior`` + ``plan.sample`` document that its latents
    "declare no prior", contradicting A20's refusal of the same document, and
    the two refusals name different edits.

    ``nuts`` and ``npe`` keep their own gate at
    ``posterior_support._sampled_space`` (``uncertainty.py::_named_spans``) as the P3 second
    opinion; this function does not call it and does not change it -- it
    needs a BUILT space (``uncertainty.py::_named_spans``), which P-1 may not make.

    **A run declaring ``expect: refuse`` is left alone**, for the reason
    :func:`_blocks` gives and with a document to point at: ``execute_run``
    (``exits.py::_SAMPLE_DEFAULTS``) runs such a run's executor and captures its error as
    the run's product (``exits.py::_plan_default``), and a P-1 refusal makes the whole
    document unloadable, so the assertion could never be made.
    ``posterior_helpers.joint_prior_document`` is exactly that document --
    ``kind: npe`` under ``expect: refuse`` beside a ``kind: nuts`` that runs,
    over ONE joint-prior space -- and it is what
    ``test_config_exits_npe.py::TestThePriorGate`` reads.  This clause was not
    in the task this function was written from; without it that class goes
    red.  The guard is per RUN, so the nuts run of that same document is
    still read.

    **The joint prior is only this pass's business when the PACKAGE would
    accept it.**  ``JeffreysPrior.validate_against`` refuses an ``over:``
    naming a latent the space does not declare, and A22 refuses a covered
    latent that also declares its own ``prior:`` -- and both of those name
    WHICH latent is wrong, which A20's sentence does not.  A20 in front of
    either would send the reader to ``kind: nuts``, where the real fault is
    waiting unchanged.  So ``covered`` is emptied unless every name in it is
    declared and prior-free, which stands A20 and A21 down and costs A23
    nothing: an undeclared name is in no A23 iteration, and a covered latent
    that declares its own prior is not prior-free.
    """
    latents = _a23_latents(document)
    covered = _a20_joint_over(document)
    if not all(name in latents and latents[name].get("prior") is None for name in covered):
        covered = ()
    for index, run in enumerate(_runs(document)):
        kind = run.get("kind")
        if not isinstance(kind, str):
            continue
        if run.get("expect") == "refuse":
            continue
        where = f"runs[{index}]"
        named = f"runs[{run['name']!r}]"

        if covered and kind.startswith("plan."):
            yield refuse(
                "A20",
                where,
                f"{named}: inference.joint_prior covers {list(covered)}, and "
                f"kind: {kind} does not evaluate a joint prior -- each block's "
                "conditional is built from the latent's OWN prior:, and a "
                "covered latent declares none, so the density contributes "
                "exactly zero. The sweep would run, settle, and report a "
                "converged chi-squared computed entirely from blocks that "
                "never saw the prior. kind: nuts is the exit that evaluates "
                "it; use that, or drop inference.joint_prior (check A20).",
            )
            continue

        if covered and kind == "fisher" and run.get("space") is True:
            yield refuse(
                "A21",
                where,
                f"{named}: inference.joint_prior covers {list(covered)}, and "
                "space: true means 'add the declared priors' curvature to "
                "this matrix'. A Jeffreys prior is DEFINED as sqrt(det of "
                "that matrix), so adding it would put it inside its own "
                "definition: what comes back is not the posterior precision "
                "it would be labelled as, and it is finite, symmetric and "
                "positive definite, so nothing downstream would say "
                "otherwise. Drop space: true -- the likelihood Fisher is what "
                "the prior is built from -- or read the posterior with "
                "kind: nuts (check A21).",
            )
            continue

        if kind == "nuts":
            missing = _a23_prior_free(latents, latents, covered)
            # "no inference.joint_prior COVERS them", never "this document
            # declares none": `covered` is emptied above for a joint prior
            # the package would refuse, and a document that declares an
            # `over: [zzz]` does declare one.  The earlier wording was false
            # on exactly that document.
            because = (
                "and the inference.joint_prior this document declares "
                f"covers {list(covered)} and not them"
                if covered
                else "and no inference.joint_prior covers them"
            )
        elif kind == "npe":
            missing = _a23_prior_free(latents, latents)
            because = (
                "and kind: npe SIMULATES a bank from each latent's OWN "
                "prior, consulting inference.joint_prior not at all"
            )
        elif kind == "fisher" and run.get("space") is True:
            missing = _a23_prior_free(latents, latents)
            # The package's own way out, which A23 owes the reader as well as
            # the calibrator exits: `uncertainty.py::_declared_gaussian_priors` says drop `space=`
            # and what comes back is exactly the likelihood matrix.
            because = (
                "and space: true asks for a posterior precision, which "
                "a prior-free latent has no row of -- drop space: and "
                "what comes back is the likelihood Fisher, which is "
                "that same matrix without the priors in it"
            )
        elif kind == "plan.sample":
            # `_t7_entries`, not `run.get("blocks") or ()`: a `blocks: 5`
            # raises on iteration and a `blocks: "nope"` iterates into
            # characters, and a malformed list is one `exits._blocks`
            # (`:181-207`) refuses in its own words.  `warm_start.blocks` is
            # NOT a site here, and that is measured rather than forgotten:
            # `require_priors` is called from `SamplingPlan.sample`
            # (`plan.py::SamplingPlan._partition`) and a warm start is `.estimate()`d
            # (`exits.py::_ESTIMATE_DEFAULTS`).
            #
            # `== _T7_GRADIENT` and never `!= _T7_CONJUGATE`: `_engine_of`
            # answers `""` for a block whose engine cannot be derived (A18's
            # case) and returns a DECLARED engine unvalidated, so the
            # complement also selects a mixed block and an `engine: banana`
            # one -- putting an A23 refusal beside A18's "mixes
            # declared-linear latents" and beside the enum clause's "asks for
            # engine: 'banana'", about a block whose engine nobody knows.
            missing = sorted(
                {
                    name
                    for entry in (_t7_entries(run.get("blocks")) or ())
                    if _engine_of(entry, latents) == _T7_GRADIENT
                    for name in _a23_prior_free(
                        latents, [one for one in (_t7_names(entry) or ()) if one in latents]
                    )
                }
            )
            because = (
                "and a block stepped by the gradient engine needs a "
                "prior on every member -- the potential is flat in a "
                "prior-free latent and the chain wanders without any "
                "diagnostic saying so"
            )
        else:
            continue

        if missing:
            yield refuse(
                "A23",
                where,
                _a23_message(named, kind, missing, because, covered, _t8_traded_keys(run)),
            )
