"""T7's engine checks, and A16's partition.

Which engine a block gets, whether it may be warm-started, how many steps it
takes, and the refusals for a partition that cannot be honoured. `_blocks` and
`_engine_of` are here rather than in the vocabulary because they reach into
these checks rather than the other way round -- measured before moving.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register

from .fitting_vocabulary import (
    _ENGINES,
    _T7_ADDITIVE_KINDS,
    _T7_CONJUGATE,
    _T7_GRADIENT,
    _T7_LOG_CONJUGATE,
    _T7_NOISE_ADDITIVE,
    _T7_NOISE_NEITHER,
    _latents,
    _runs,
    _t7_entries,
    _t7_names,
)

#: What an uncovered latent costs, per SITE.  Two sentences rather than one
#: because the main partition and a ``warm_start``'s are not the same claim:
#: a latent no ``runs[].blocks`` entry covers is frozen for the whole run,
#: while one no ``warm_start.blocks`` entry covers is frozen only for the warm
#: estimate -- the run's own blocks still update it.  Sharing the formatter
#: without sharing the reasoning is how a message becomes false on its second
#: caller, which is the defect class Task 6 shipped eight of.
_A16_FROZEN: dict[str, str] = {
    "blocks": (
        "An omitted latent is silently frozen at its declared init for "
        "the whole run -- the sweep converges, the joint chi-squared "
        "settles, and nothing anywhere reports that a parameter you "
        "declared was never inferred."
    ),
    "warm_start.blocks": (
        "warm_start builds a SamplingPlan of its own over the same space "
        "(exits.py:287), so an omitted latent sits at its declared init "
        "for the whole warm estimate, and warm_start.move: can only carry "
        "over a value that estimate produced."
    ),
}


def _log_route_refusal_text(noise: Any) -> str | None:
    """``log_route_refusal``'s verdict on ``inference.noise``, from its TEXT.

    ``None`` means either "the log route exists" or "the text cannot say";
    both let the block through, and the package decides the second at P3.

    * ``kind:`` in :data:`_T7_ADDITIVE_KINDS` -> ``noise_additive``.
    * ``kind: radiometer`` with a ``floor:`` whose number the text carries and
      which is not ``<= 0`` -> ``noise_neither``. ``not <= 0`` rather than
      ``> 0``, as G4 writes it in the package, so a NaN floor is refused and
      not routed. The number is read by ``preflight/instrument.py``'s
      :func:`_text_number`, which APPLIES the unit: ``celsius`` is affine, and
      ``{value: 0, unit: celsius}`` is a floor of 273.15 K.
    * Anything else -- no noise, ``kind: none``, an unknown kind, a floor
      written as ``{ref: ...}`` -- stands down. Those are other checks' and
      ``build_noise``'s to refuse, and a second voice here would give one
      fault two refusals.

    ``flags:`` does not enter: ``log_route_refusal`` unwraps a
    ``FlaggedNoise`` and judges the base model.
    """
    from rheplicant.config.preflight.instrument import _text_number

    if not isinstance(noise, Mapping):
        return None
    kind = noise.get("kind")
    if not isinstance(kind, str):
        return None
    if kind in _T7_ADDITIVE_KINDS:
        return _T7_NOISE_ADDITIVE
    if kind != "radiometer" or "floor" not in noise:
        return None
    floor = _text_number(noise["floor"])
    if floor is None:
        return None
    return _T7_NOISE_NEITHER if not floor <= 0.0 else None


def _log_route_message(
    named: str, site: str, position: int, noise: Mapping[str, Any], reason: str
) -> str:
    """Why this ``engine: log_conjugate`` block has no log route."""
    opening = f"{named}: {site}[{position}] asks for engine: log_conjugate, and "
    if reason == _T7_NOISE_NEITHER:
        floor = noise["floor"]
        said = (
            f"{floor.get('value')} {floor.get('unit', '')}".strip()
            if isinstance(floor, Mapping)
            else repr(floor)
        )
        return (
            opening + f"inference.noise.floor declares {said} on kind: "
            "radiometer, so sigma = f * max(|mu|, floor): proportional to the "
            "prediction above the floor and constant below it. The log route "
            "uses sigma = f on every sample, which is the declared likelihood "
            "only where the floor never binds, so a floored noise has no "
            "closed-form log route (log_route_refusal: noise_neither). Declare "
            "engine: gradient for this block, which evaluates the declared "
            "likelihood, or drop inference.noise.floor if the prediction cannot "
            "approach zero (check A19)."
        )
    return (
        opening + f"inference.noise is kind: {noise.get('kind')}, whose sigma "
        "does not scale with the prediction. The log route solves against "
        "log(data) under a multiplicative noise d = mu (1 + f w); applied to a "
        "noise that is already additive it states a different likelihood from "
        "the one declared (log_route_refusal: noise_additive). Declare another "
        "engine for this block, or inference.noise.kind: radiometer if the "
        "noise is multiplicative (check A19)."
    )


def _a18_linear(latents: Mapping[str, Any], name: str) -> bool:
    """``Latent(linear=)`` as the document writes it.

    ``sections/parameters.py::parse_latents`` reads ``spec.get("linear", False)`` and
    ``sections/parameters.py::parse_latents`` refuses a non-bool, and ``Latent``'s own default is
    False
    (``inference/parameters.py::Latent``: ``linear: bool = eqx.field(static=True,
    default=False)``).  So a latent that declares nothing is non-linear, an
    undeclared name is non-linear, and a latent that declares ``linear: "yes"``
    is refused at P2 rather than here -- this function is not the grammar
    check for the key.
    """
    body = latents.get(name)
    return isinstance(body, Mapping) and body.get("linear") is True


def _engine_of(block: Mapping[str, Any], latents: Mapping[str, Any]) -> str:
    """The engine a block takes, derived from text -- ``plan_settings.py::split_rhat``.

    **Mirrored, line for line, from** ``SamplingPlan._engine_of``
    (``plan_settings.py::split_rhat``, verified with ``inspect.getsourcelines``):

    * ``plan_settings.py::split_rhat`` partition the names by ``Latent.linear`` -> :func:`_a18_linear`;
    * ``plan_settings.py::split_rhat`` ``if block.engine is None`` -> the ``declared is None`` branch;
    * ``plan_settings.py::split_rhat`` mixed-with-no-override is UNDERIVABLE -> ``""`` here,
      because the package raises there and a pre-flight check may not
      (§2.3's TRAP: a check that raises aborts the pass and hides every
      later finding);
    * ``plan_settings.py::split_rhat`` ``engine = CONJUGATE if linear else GRADIENT`` -> verbatim,
      through :data:`_T7_CONJUGATE` and :data:`_T7_GRADIENT`;
    * ``plan_settings.py::split_rhat`` the override wins, unvalidated -> returned as declared, so
      :func:`_blocks` can refuse an engine outside :data:`_ENGINES` itself;
    * ``plan_settings.py::split_rhat`` conjugate-over-non-linear is A19's and stays in
      :func:`_blocks`, because it is a REFUSAL and this function returns a
      string;
    * ``plan_results.py::Block`` conjugate-plus-steps is A17's, same reason.

    **The risk this mirroring carries is drift**: the package can change
    ``_engine_of`` and every test written against our messages stays green
    while the pass and the run disagree about which engine a block takes --
    which is worse than the gap it closes, because a wrong engine is refused
    at P-1 for a reason the run would not have given.  What catches it is
    ``test_the_pass_agrees_with_the_package_on_every_case``: it drives this
    function AND ``Block(...)`` + ``SamplingPlan(space, *blocks)`` over the
    same thirteen ``(latents, blocks)`` cases and asserts the two agree on
    every one.  A drift in ``plan.py`` turns that test red on the case it
    drifted on.

    Returns ``""`` when the block's engine cannot be derived -- A18's case,
    which :func:`_blocks` turns into the refusal.  A declared engine is
    returned even when it is not in :data:`_ENGINES`, so the caller can refuse
    it by name; ``""`` and an unknown string are therefore different answers
    and the caller distinguishes them.
    """
    names = _t7_names(block) or ()
    declared = block.get("engine")
    if declared is not None:
        return declared if isinstance(declared, str) else ""
    linear = [name for name in names if _a18_linear(latents, name)]
    other = [name for name in names if not _a18_linear(latents, name)]
    if other and linear:
        return ""
    return _T7_CONJUGATE if linear else _T7_GRADIENT


def _t7_warm_start(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The ``warm_start`` whose ``blocks:`` the executor would actually READ.

    ``_run_plan`` reaches ``warm_start.blocks`` (``exits.py::_ESTIMATE_DEFAULTS``) only after
    five earlier refusals, and three of them are this check's business.  Each
    of the three means the warm block list is **never read at all**, so a
    partition answer about it is an answer about a structure the package
    discards -- and it is the only sentence the reader gets, because none of
    the three is hoisted to P-1 by anything:

    * the run's own ``blocks:`` grammar (``exits.py::_SAMPLE_KEYS``, into
      ``_blocks``/``exits.py::_parse_optimize``).  Measured: a ``plan.sample`` with no
      ``blocks:`` at all, and one with ``blocks: "nope"``, both used to earn
      *"runs['s']: warm_start.blocks: does not cover [...]"* -- telling a user
      who omitted a key about a different key, and never about theirs;
    * ``warm_start.kind`` (``exits.py::_WARM_KEYS``): anything but
      ``plan.estimate`` and the whole warm start is refused;
    * ``warm_start.move`` (``exits.py::_ESTIMATE_PASSTHROUGH``): required, and without it the
      warm start never runs.

    **``n_sweeps`` (``exits.py::_SAMPLE_KEYS``) and the seed (``exits.py::_SAMPLE_KEYS``) are NOT
    gates**, and that is the line rather than an omission.  They are
    independent run keys, not the warm start's own grammar: a document
    missing one AND carrying a broken warm partition has two faults the user
    must fix either way, and reporting both is what collect-rather-than-raise
    is for (§2.3).  Gating on them would trade one round trip for another.

    ``warm_start`` is read on ``plan.sample`` only, because ``_ESTIMATE_KEYS``
    (``exits.py::_parse_optimize``) does not take it and Task 3's ``A1.runs`` already
    refuses it at P-1 -- measured: *"kind: plan.estimate does not take
    ['warm_start']"*.
    """
    if run.get("kind") != "plan.sample":
        return None
    warm = run.get("warm_start")
    if not isinstance(warm, Mapping):
        return None
    if warm.get("kind") != "plan.estimate":
        return None
    move = warm.get("move")
    if not isinstance(move, list) or not move or not all(isinstance(name, str) for name in move):
        return None
    return warm


def _t7_sites(run: Mapping[str, Any]) -> tuple[tuple[str, tuple], ...]:
    """Every ``blocks:`` list in this run that reaches a ``SamplingPlan``.

    **Two, not one.**  ``exits.py::_ESTIMATE_DEFAULTS`` builds ``_blocks(f"{where}:
    warm_start", warm.get("blocks"))`` and hands the result to
    ``SamplingPlan(space, *warm_blocks)`` -- the same constructor, over the
    same space, refused by the same four rules, at the same P3 behind the
    same beam.  A16-A19 written on ``runs[].blocks`` alone would guard one
    route and leave its identical sibling open.

    The warm site is reached only when the main one was READABLE -- the first
    of :func:`_t7_warm_start`'s three gates, kept here because it is the same
    ``_t7_entries`` answer the main site already needed.
    """
    sites: list[tuple[str, tuple]] = []
    entries = _t7_entries(run.get("blocks"))
    if entries is None:
        return ()
    sites.append(("blocks", entries))
    warm = _t7_warm_start(run)
    if warm is not None:
        warm_entries = _t7_entries(warm.get("blocks"))
        if warm_entries is not None:
            sites.append(("warm_start.blocks", warm_entries))
    return tuple(sites)


def _t7_step_count(steps: Any) -> bool:
    """Is this a value ``Block`` would have accepted as inner steps?

    ``plan.py``'s own predicate: a positive ``int`` that is not a
    ``bool``.  A17 needs it because its sentence is a COUNTERFACTUAL -- *"so
    steps: 5 would be silently ignored"* -- and that claim is false of every
    value the package refuses first.  Measured: ``Block('d', 'a', steps=0)``
    never reaches ``SamplingPlan`` at all, and neither does ``steps=True`` or
    ``steps='5'``; each is refused as *"inner steps must be a positive int"*.
    """
    return isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1


def _a16_partition(
    named: str,
    listed: str,
    site: str,
    entries: tuple[Mapping[str, Any], ...],
    latents: Mapping[str, Any],
) -> Iterable[Finding]:
    """A16: the partition, in the order ``plan_settings.py::_not_converged_message`` settles it.

    Three legs, one id.  The schema row (line 1193) describes two of them --
    "every latent appears in exactly one block" -- and the third, a block
    naming a name ``inference.parameters`` never declared, is
    ``plan_settings.py::_not_converged_message``'s and is refused FIRST there (the covered pair is
    ``plan_settings.py::_not_converged_message`` and ``plan_settings.py::_not_converged_message``).  A fourth shape,
    one name written twice inside ONE block, is ``Block._check``'s
    (``plan.py``) rather than the plan's and carries the same id: it
    is the same property (each latent in exactly one place) one level in.
    Task 13 records the wording.

    ``listed`` is the ``where`` of the block LIST; ``site`` is how the message
    spells it (``blocks`` or ``warm_start.blocks``).
    """
    owner: dict[str, int] = {}
    for position, entry in enumerate(entries):
        block_where = f"{listed}[{position}]"
        seen: set[str] = set()
        for name in _t7_names(entry) or ():
            if name in seen:
                yield refuse(
                    "A16",
                    block_where,
                    f"{named}: {site}[{position}].names lists {name!r} twice, "
                    "and two copies of one latent in a block are exactly "
                    "degenerate with each other -- the block's normal "
                    "operator is singular in a direction that says nothing "
                    "about the model, and the answer has one entry per name, "
                    "so one copy's result silently overwrites the other's "
                    "(check A16).",
                )
                continue
            seen.add(name)
            if name not in latents:
                yield refuse(
                    "A16",
                    block_where,
                    f"{named}: {site}[{position}] names {name!r}, which "
                    "inference.parameters does not declare; it declares "
                    f"{list(latents)}. A block over a name nobody declared "
                    "updates nothing and leaves the latent it was meant to "
                    "cover sitting at its declared init (check A16).",
                )
                continue
            if name in owner:
                yield refuse(
                    "A16",
                    block_where,
                    f"{named}: {name!r} is in {site}[{owner[name]}] and in "
                    f"{site}[{position}]. A Gibbs sweep updates each block "
                    "against the conditional that holds when it runs, so the "
                    "second update solves a conditional the first one just "
                    "invalidated -- and every diagnostic reports the second's "
                    "answer as if the first had never happened. Put each "
                    "latent in exactly one block; to update two together, put "
                    "them in ONE block (check A16).",
                )
                continue
            owner[name] = position
    missing = [name for name in latents if name not in owner]
    if missing:
        yield refuse(
            "A16",
            listed,
            f"{named}: {site}: does not cover {missing}; every latent "
            "inference.parameters declares must be in exactly one block. "
            f"{_A16_FROZEN[site]} Add it to a block, or drop it from "
            "inference.parameters (check A16).",
        )


def _a17_message(named: str, site: str, position: int, steps: Any) -> str:
    """A17's refusal, whose middle clause depends on the VALUE.

    *"would be silently ignored"* is a claim about what the package would do
    without this check, and for a ``steps:`` the package refuses outright it
    is false -- so is the fix clause that sends the reader to ``engine:
    gradient``, which would leave the run refused for the second reason.
    Measured, ``plan.py``: ``steps=0``, ``steps=True``, ``steps='5'``
    and ``steps=1.5`` are all refused before a plan is settled at all.
    """
    head = (
        f"{named}: {site}[{position}] is solved by the conjugate engine, "
        "which has no inner steps, so steps: "
    )
    why = (
        "A conjugate block's estimate is one Wiener solve and its draw is "
        "one exact constrained realization -- there is no step count to "
        "tune, which is the whole advantage. "
    )
    if _t7_step_count(steps):
        return (
            f"{head}{steps!r} would be silently ignored. {why}Drop "
            "steps:, or declare engine: gradient if a gradient step was "
            "what you meant (check A17)."
        )
    return (
        f"{head}{steps!r} is not a knob it has. {why}Drop steps:. Moving "
        f"to engine: gradient would not rescue {steps!r} either -- inner "
        "steps are a positive int on every engine (plan.py:360), so "
        "the block would be refused a second time (check A17)."
    )


def _t7_engines(
    named: str,
    listed: str,
    site: str,
    entries: tuple[Mapping[str, Any], ...],
    latents: Mapping[str, Any],
    *,
    derive: bool,
    noise: Any = None,
) -> Iterable[Finding]:
    """A17, A18, A19 and the engine enum, in the order ``plan.py`` decides them.

    **``derive`` is False when the partition is wrong, and only the ENUM runs
    then.**  A17, A18 and A19 all read the latents a name resolves to, so on a
    broken partition they answer about names that do not exist -- an
    undeclared name reads as non-linear (:func:`_a18_linear` returns False for
    an absent latent) and produces an A18 *"mixes linear with non-linear"*
    refusal naming a latent nobody declared.  That is ``plan_settings.py::_not_converged_message``'s
    own argument: *"a block naming an undeclared latent cannot have its engine
    derived at all, so the partition is settled first"*.

    The enum is not one of those.  ``engine:`` is checked by ``Block._check``
    (``plan.py``), which the package runs on EVERY block before
    ``SamplingPlan`` settles anything -- it reads the block alone and no
    latent at all.  Suppressing it behind the partition would cost a user with
    both faults a second round trip, which is what §2.3's collect-rather-than-
    raise design exists to prevent.
    """
    for position, entry in enumerate(entries):
        block_where = f"{listed}[{position}]"
        names = _t7_names(entry) or ()
        declared = entry.get("engine")

        # The enum, closed, under A19 -- the row for an explicit `engine:`
        # the block cannot have.  It was id-less (`check=""`), which the
        # audit trace refuses: `validate` printed "finding.check must be a
        # non-empty string." in place of this sentence.  It is not decoration:
        # `_engine_of` returns a declared engine unvalidated, so without this
        # an `engine: banana` block would reach the A17 test as
        # `engine == _T7_CONJUGATE` -> False and be silently accepted by this pass
        # while `plan.py` refuses it at P3.
        #
        # `isinstance` BEFORE `in`: `_ENGINES` is a frozenset, and `["x"] in
        # frozenset` raises TypeError on the unhashable list -- which inside
        # the pass is "check A16 RAISED" and costs the report every other
        # finding.  `Block._check` refuses a non-string engine by the same
        # clause, measured: "asks for engine=5; the engines are [...]".
        if declared is not None and not (isinstance(declared, str) and declared in _ENGINES):
            yield refuse(
                "A19",
                block_where,
                f"{named}: {site}[{position}] asks for engine: {declared!r}; "
                f"the engines are {sorted(_ENGINES)}. Leave engine: out and "
                "it is derived from linear: true on each member, which is the "
                "normal case -- an explicit engine is an override (check A19).",
            )
            continue

        # The log route, which reads the block's declared engine and the
        # noise and no latent, so it runs whether or not the partition is
        # right, as the enum does. `to_log_space` refuses the same noises at
        # P3 through `log_route_refusal`; before this, such a document passed
        # `rheplicant validate` and failed `rheplicant run` with a traceback.
        #
        # Filed under A19 -- an engine override whose precondition the
        # declaration does not meet -- and NOT id-less like the enum above:
        # measured, an id-less finding reaches `rheplicant validate` as
        # "finding.check must be a non-empty string." from the audit trace
        # (`_rheplicant_bootstrap/audit/trace.py`), exit 2 with the message
        # lost, which is what `engine: banana` does today.
        if declared == _T7_LOG_CONJUGATE:
            reason = _log_route_refusal_text(noise)
            if reason is not None:
                yield refuse(
                    "A19", block_where, _log_route_message(named, site, position, noise, reason)
                )
                continue
        if not derive:
            continue

        engine = _engine_of(entry, latents)
        if engine == "":
            linear = [name for name in names if _a18_linear(latents, name)]
            other = [name for name in names if not _a18_linear(latents, name)]
            yield refuse(
                "A18",
                block_where,
                f"{named}: {site}[{position}] mixes declared-linear latents "
                f"{linear} with non-linear ones {other}, so which engine it "
                "takes cannot be derived. A conjugate solve needs the whole "
                "block affine; a gradient step does not exploit the linear "
                "members' structure at all, which for a high-dimensional "
                "linear block is the difference between tractable and "
                "hopeless. Split them into separate blocks, or declare "
                "engine: gradient to step the whole block by gradient "
                "deliberately (check A18).",
            )
            continue

        # A19 before A17, because `plan.py` does: `:662-671` is inside the
        # override branch and `:673-681` is after it, so a MIXED block asking
        # for `engine: conjugate` with `steps:` is refused as A19 and never
        # reaches A17.  Measured, on that exact block: "Block('d', 'w') asks
        # for engine='conjugate', but ['w'] are not declared linear=True."
        if declared == _T7_CONJUGATE:
            other = [name for name in names if not _a18_linear(latents, name)]
            if other:
                yield refuse(
                    "A19",
                    block_where,
                    f"{named}: {site}[{position}] asks for engine: conjugate, "
                    f"but {other} do not declare linear: true. The conjugate "
                    "machinery solves (A^T N^-1 A + S^-1)x = b, which is the "
                    "posterior only if the prediction really is affine in the "
                    "block -- and that claim belongs in the latent's "
                    "declaration, where check_linearity verifies it, not in a "
                    "run that asserts it. Declare linear: true and the claim "
                    "will be checked; leave it out and this block is stepped "
                    "by gradient (check A19).",
                )
                continue

        # A17, as §2.6 item 3 decided it: "conjugate-ENGINE block", not the
        # schema's "all-linear block".  Measured both ways: an all-linear
        # block declared `engine: gradient` with `steps: 5` is ACCEPTED by the
        # package (SamplingPlan(('d','a'):gradient, ('w'):gradient)), and
        # `engine: conjugate` on a partly-linear block trips A19 above.
        if engine == _T7_CONJUGATE and entry.get("steps") is not None:
            yield refuse(
                "A17", block_where, _a17_message(named, site, position, entry.get("steps"))
            )


@register("A16", "A17", "A18", "A19")
def _blocks(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A16-A19: a ``plan.*`` run's blocks, against the declared latents.

    The first check in the layer that needs ``runs[]`` and ``inference:`` at
    the same time.  Registered under four ids because ``Report.checks()`` is
    "the ids that fired" and a reader asking "did A17 run" must get an answer;
    ``preflight()`` calls this function once.

    **A run declaring ``expect: refuse`` is left alone**, and that is not
    politeness.  ``execute_run`` (``exits.py::_plan_default``) runs such a run's
    executor and CAPTURES its error as the run's product -- the run is an
    assertion ABOUT the refusal.  A P-1 refusal makes the whole document
    unloadable, so the assertion could never be made; measured,
    ``test_config_exits_plan.py::TestEstimate.test_noise_kind_none_is_refused`` is exactly that
    document (``blocks: [{names: [g, ghost]}]`` under ``expect: refuse``) and
    it asserts the captured error names ``ghost``.

    Only ``plan.*`` kinds are read.  ``blocks:`` is a ``plan.*`` option, so a
    ``kind: fisher`` run carrying a stray one is Task 3's ``A1.runs`` hole and
    not A16's; refusing it here would give the user two refusals for one typo.

    **Variant layers are not walked.**  ``load_document`` calls the pass on
    the variant-APPLIED document (``preflight/document.py``), so a selected variant
    is read; an unselected one is A1's row and Task 3's ``_variant_text``, not
    this one's.
    """
    latents = _latents(document)
    inference = document.get("inference")
    noise = inference.get("noise") if isinstance(inference, Mapping) else None
    for index, run in enumerate(_runs(document)):
        kind = run.get("kind")
        if not (isinstance(kind, str) and kind.startswith("plan.")):
            continue
        if run.get("expect") == "refuse":
            continue
        named = f"runs[{run['name']!r}]"
        for site, entries in _t7_sites(run):
            listed = f"runs[{index}].{site}"
            # The partition FIRST, and no engine DERIVED when it is wrong --
            # `_t7_engines`' own docstring says which clause survives that and
            # why the enum is the one that does.
            partition = list(_a16_partition(named, listed, site, entries, latents))
            yield from partition
            yield from _t7_engines(
                named, listed, site, entries, latents, derive=not partition, noise=noise
            )
