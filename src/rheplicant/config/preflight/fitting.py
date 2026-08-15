"""The checks that need ``runs[]`` and ``inference:`` at the same time.

No function in this layer sees both today (the plan's finding 2):
``parse_runs`` sees ``runs`` alone and inspects five keys -- every other run
key travels untouched in ``RunSpec.options`` (``runs.py:113-117``) and is
swept for the first time inside the executor at P3; ``build_inference`` sees
``inference`` alone; ``run_document`` (``runs.py:138``) holds both and hands
them to ``load_document`` one at a time, calling ``parse_runs`` at
``runs.py:149`` -- BEFORE the pass, which is why every test of a
``runs``-shaped check here drives ``load_document`` rather than
``run_document`` (§2.1).

Everything here is text: latent names, ``linear:`` flags, block membership,
kind strings, integers.  No function in this module resolves a value node,
builds an operator, reads a file or constructs a ParameterSpace.

**Nothing from ``rheplicant.inference`` is imported at module scope, and that
is forced rather than chosen.**  The task this module was written from said to
write ``from rheplicant.inference.engines import CONJUGATE, ENGINES,
GRADIENT`` at the head, having measured that ``numpyro`` stays out of
``sys.modules``.  It does -- and that is not the invariant the repository
actually holds this layer to.  ``rheplicant/inference/__init__.py`` re-exports
the whole layer eagerly, so reaching ``...engines`` imports it, and
``test_config_exits_predict.py:1046-1073`` runs ``import rheplicant.config;
from rheplicant.config.sections import exits`` in a fresh interpreter and
asserts that BOTH ``numpyro`` and ``rheplicant.inference`` are absent.
Measured: at ``9ee99af`` that probe prints ``[]``, and with the head import
here it prints ``['rheplicant.inference']`` and the test fails.
``sections/exits.py`` defers ``from rheplicant.inference import Block`` into
its function bodies for the same reason.  So the two engine names are written
out and :func:`test_the_engine_enum_is_the_packages_own` holds them to the
package's own ``ENGINES``.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register

__all__: list[str] = []

#: ``engines.CONJUGATE`` and ``engines.GRADIENT`` (``engines.py:62``, ``:66``),
#: written out because this module may not import that package -- see the
#: module docstring for the guard that measures it.
_T7_CONJUGATE: str = "conjugate"
_T7_GRADIENT: str = "gradient"

#: The engines a block may ask for, closed.  ``_BLOCK_KEYS``
#: (``exits.py:164``) accepts ANY string for ``engine:``, so today
#: ``engine: banana`` reaches the user as a ParameterSpaceError from
#: ``Block._check`` (``plan.py:353-359``) -- measured.
#:
#: A copy of ``ENGINES`` (``engines.py:69``), which is the thing this plan
#: warns against: a closed set written out stops being closed the day a third
#: engine ships, and this pass would then refuse a block the package accepts.
#: What stops that is not the import -- which the guard above forbids -- but
#: ``test_the_engine_enum_is_the_packages_own``, which imports ``CONJUGATE``,
#: ``GRADIENT`` and ``ENGINES`` in the TEST and asserts all three against
#: these three names.  A third engine turns that test red.
_ENGINES: frozenset[str] = frozenset({_T7_CONJUGATE, _T7_GRADIENT})

#: What an uncovered latent costs, per SITE.  Two sentences rather than one
#: because the main partition and a ``warm_start``'s are not the same claim:
#: a latent no ``runs[].blocks`` entry covers is frozen for the whole run,
#: while one no ``warm_start.blocks`` entry covers is frozen only for the warm
#: estimate -- the run's own blocks still update it.  Sharing the formatter
#: without sharing the reasoning is how a message becomes false on its second
#: caller, which is the defect class Task 6 shipped eight of.
_A16_FROZEN: dict[str, str] = {
    "blocks": ("An omitted latent is silently frozen at its declared init for "
               "the whole run -- the sweep converges, the joint chi-squared "
               "settles, and nothing anywhere reports that a parameter you "
               "declared was never inferred."),
    "warm_start.blocks": (
        "warm_start builds a SamplingPlan of its own over the same space "
        "(exits.py:268), so an omitted latent sits at its declared init for "
        "the whole warm estimate, and warm_start.move: can only carry over a "
        "value that estimate produced."),
}


def _latents(document: Mapping[str, Any]) -> dict[str, Any]:
    """``inference.parameters`` as the document writes it, or ``{}``.

    The KEYS are the space's latent names, in declaration order -- measured
    through ``load_document``: ``tuple(document["inference"]["parameters"]) ==
    tuple(built.inference.space.names)``.  ``fan:`` and ``transform:``
    describe the BINDING (``sections/transforms.py:357`` appends one ``Bind``
    per entry and creates no latent) and ``hyper:`` is refused as capability 4
    (``sections/parameters.py:144-148``), so nothing in v1 splits one
    declaration into two latents.  A16 rests entirely on that and
    ``test_the_space_names_are_the_declared_parameter_keys`` is what keeps it
    true.

    A latent whose body is not a mapping keeps its NAME and reads as ``{}``.
    Dropping it would make it look undeclared to A16, which would refuse a
    block that names it -- a wrong refusal caused by a malformed neighbour.
    ``sections/parameters.py:140-141`` refuses the malformed body itself, at
    P2.
    """
    inference = document.get("inference")
    if not isinstance(inference, Mapping):
        return {}
    parameters = inference.get("parameters")
    if not isinstance(parameters, Mapping):
        return {}
    return {name: (body if isinstance(body, Mapping) else {})
            for name, body in parameters.items() if isinstance(name, str)}


def _runs(document: Mapping[str, Any]) -> tuple[dict, ...]:
    """``runs:`` as a tuple of mappings, INDEX-PRESERVING, with ``name`` filled.

    Three things this does, each of which a later task would otherwise redo:

    * a single mapping is wrapped in a list, the way ``parse_runs``
      (``runs.py:122-123``) does, so ``runs: {kind: forward}`` is one run here
      too;
    * a malformed entry becomes ``{}`` rather than being dropped, so
      ``_runs(document)[i]`` is always ``runs[i]`` of the document and a
      ``Finding.where`` of ``runs[2]`` points where the user must type.  Such
      an entry carries NO ``name`` -- read it with ``.get`` or behind a test
      on ``kind``;
    * ``name`` is filled in by ``parse_runs``' own rule -- ``entry.get("name")``
      when it is a string, else the kind (``runs.py:115``) -- because every
      refusal in this layer is prefixed ``runs['<name>']:`` and three tasks
      would otherwise each re-derive it.  A NEW dict is built; the caller's
      document is never mutated.
    """
    section = document.get("runs")
    if isinstance(section, Mapping):
        section = [section]
    if not isinstance(section, list):
        return ()
    filled: list[dict] = []
    for entry in section:
        if not isinstance(entry, Mapping):
            filled.append({})
            continue
        kind = entry.get("kind")
        name = entry.get("name")
        if not isinstance(name, str):
            name = kind if isinstance(kind, str) else ""
        filled.append({**entry, "name": name})
    return tuple(filled)


def _kinds(document: Mapping[str, Any]) -> frozenset[str]:
    """Every ``runs[].kind`` the document declares, as a set.

    For the checks that ask "does this document run ANY exit of shape X" --
    A20's ``plan.*`` family, A21's ``fisher``, A30's "is a fitting exit
    declared at all".  A check that needs the run's own index or options
    walks :func:`_runs` instead.
    """
    return frozenset(run["kind"] for run in _runs(document)
                     if isinstance(run.get("kind"), str))


def _t7_names(entry: Mapping[str, Any]) -> tuple[str, ...] | None:
    """``names:`` when the grammar accepts it, and ``None`` when it does not.

    ``exits.py:198-202``'s own three tests -- a list, non-empty, all strings
    -- because everything else is that refusal's, in its own words: *"blocks[0]
    .names is a non-empty list of latent names"*.

    ``None`` and not ``()``, and the difference is the point.  The brief's
    spelling was ``for name in (entry.get("names") or ())``, which reads
    ``names: "d"`` as the one-name list ``['d']`` (a string is iterable) and
    **raises TypeError** on ``names: 5`` -- and inside the pass a TypeError
    becomes "check A16 RAISED" and discards every other finding in the
    report.
    """
    names = entry.get("names")
    if not isinstance(names, list) or not names:
        return None
    if not all(isinstance(name, str) for name in names):
        return None
    return tuple(names)


def _t7_entries(node: Any) -> tuple[Mapping[str, Any], ...] | None:
    """A ``blocks:`` list this check may read, or ``None`` to stand down.

    ``exits._blocks`` (``exits.py:180-202``) refuses four shapes before a
    ``Block`` is ever built: a ``blocks:`` that is not a list, an empty one,
    an entry that is not a mapping, and a malformed ``names:``.  Each has a
    sentence naming the fault; a partition answer in front of one would say
    *"blocks: does not cover ['d', 'a', 'w']"* -- true, useless, and offering
    a fix ("add it to a block") that is not the fault.

    **All or nothing per list.**  One malformed entry makes the whole
    partition undecidable: the names it would have owned are unknown, so
    every OTHER entry's coverage answer would be computed against a set that
    is missing them.
    """
    if not isinstance(node, list) or not node:
        return None
    for entry in node:
        if not isinstance(entry, Mapping) or _t7_names(entry) is None:
            return None
    return tuple(node)


def _a18_linear(latents: Mapping[str, Any], name: str) -> bool:
    """``Latent(linear=)`` as the document writes it.

    ``sections/parameters.py:178`` reads ``spec.get("linear", False)`` and
    ``:179-180`` refuses a non-bool, and ``Latent``'s own default is False
    (``inference/parameters.py:255``: ``linear: bool = eqx.field(static=True,
    default=False)``).  So a latent that declares nothing is non-linear, an
    undeclared name is non-linear, and a latent that declares ``linear: "yes"``
    is refused at P2 rather than here -- this function is not the grammar
    check for the key.
    """
    body = latents.get(name)
    return isinstance(body, Mapping) and body.get("linear") is True


def _engine_of(block: Mapping[str, Any], latents: Mapping[str, Any]) -> str:
    """The engine a block takes, derived from text -- ``plan.py:643-682``.

    **Mirrored, line for line, from** ``SamplingPlan._engine_of``
    (``plan.py:643-682``, verified with ``inspect.getsourcelines``):

    * ``:645-646`` partition the names by ``Latent.linear`` -> :func:`_a18_linear`;
    * ``:648`` ``if block.engine is None`` -> the ``declared is None`` branch;
    * ``:649-658`` mixed-with-no-override is UNDERIVABLE -> ``""`` here,
      because the package raises there and a pre-flight check may not
      (§2.3's TRAP: a check that raises aborts the pass and hides every
      later finding);
    * ``:659`` ``engine = CONJUGATE if linear else GRADIENT`` -> verbatim,
      through :data:`_T7_CONJUGATE` and :data:`_T7_GRADIENT`;
    * ``:660-661`` the override wins, unvalidated -> returned as declared, so
      :func:`_blocks` can refuse an engine outside :data:`_ENGINES` itself;
    * ``:662-671`` conjugate-over-non-linear is A19's and stays in
      :func:`_blocks`, because it is a REFUSAL and this function returns a
      string;
    * ``:673-681`` conjugate-plus-steps is A17's, same reason.

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


def _t7_sites(run: Mapping[str, Any]) -> tuple[tuple[str, tuple], ...]:
    """Every ``blocks:`` list in this run that reaches a ``SamplingPlan``.

    **Two, not one.**  ``exits.py:268`` builds ``_blocks(f"{where}:
    warm_start", warm.get("blocks"))`` and hands the result to
    ``SamplingPlan(space, *warm_blocks)`` -- the same constructor, over the
    same space, refused by the same four rules, at the same P3 behind the
    same beam.  A16-A19 written on ``runs[].blocks`` alone would guard one
    route and leave its identical sibling open.

    **``warm_start`` is read on ``plan.sample`` only**, because
    ``_ESTIMATE_KEYS`` (``exits.py:165``) does not take it and Task 3's
    ``A1.runs`` already refuses it at P-1 -- measured: *"kind: plan.estimate
    does not take ['warm_start']"*.  Reading it on every ``plan.*`` run would
    put a partition refusal about an unread block list in front of the
    refusal that names the real fault.
    """
    sites: list[tuple[str, tuple]] = []
    entries = _t7_entries(run.get("blocks"))
    if entries is not None:
        sites.append(("blocks", entries))
    if run.get("kind") == "plan.sample":
        warm = run.get("warm_start")
        if isinstance(warm, Mapping):
            entries = _t7_entries(warm.get("blocks"))
            if entries is not None:
                sites.append(("warm_start.blocks", entries))
    return tuple(sites)


def _t7_step_count(steps: Any) -> bool:
    """Is this a value ``Block`` would have accepted as inner steps?

    ``plan.py:360-368``'s own predicate: a positive ``int`` that is not a
    ``bool``.  A17 needs it because its sentence is a COUNTERFACTUAL -- *"so
    steps: 5 would be silently ignored"* -- and that claim is false of every
    value the package refuses first.  Measured: ``Block('d', 'a', steps=0)``
    never reaches ``SamplingPlan`` at all, and neither does ``steps=True`` or
    ``steps='5'``; each is refused as *"inner steps must be a positive int"*.
    """
    return isinstance(steps, int) and not isinstance(steps, bool) and steps >= 1


def _a16_partition(named: str, listed: str, site: str,
                   entries: tuple[Mapping[str, Any], ...],
                   latents: Mapping[str, Any]) -> Iterable[Finding]:
    """A16: the partition, in the order ``plan.py:545-586`` settles it.

    Three legs, one id.  The schema row (line 1193) describes two of them --
    "every latent appears in exactly one block" -- and the third, a block
    naming a name ``inference.parameters`` never declared, is
    ``plan.py:545-558``'s and is refused FIRST there.  A fourth shape,
    one name written twice inside ONE block, is ``Block._check``'s
    (``plan.py:344-352``) rather than the plan's and carries the same id: it
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
                    "A16", block_where,
                    f"{named}: {site}[{position}].names lists {name!r} twice, "
                    "and two copies of one latent in a block are exactly "
                    "degenerate with each other -- the block's normal "
                    "operator is singular in a direction that says nothing "
                    "about the model, and the answer has one entry per name, "
                    "so one copy's result silently overwrites the other's "
                    "(check A16).")
                continue
            seen.add(name)
            if name not in latents:
                yield refuse(
                    "A16", block_where,
                    f"{named}: {site}[{position}] names {name!r}, which "
                    "inference.parameters does not declare; it declares "
                    f"{list(latents)}. A block over a name nobody declared "
                    "updates nothing and leaves the latent it was meant to "
                    "cover sitting at its declared init (check A16).")
                continue
            if name in owner:
                yield refuse(
                    "A16", block_where,
                    f"{named}: {name!r} is in {site}[{owner[name]}] and in "
                    f"{site}[{position}]. A Gibbs sweep updates each block "
                    "against the conditional that holds when it runs, so the "
                    "second update solves a conditional the first one just "
                    "invalidated -- and every diagnostic reports the second's "
                    "answer as if the first had never happened. Put each "
                    "latent in exactly one block; to update two together, put "
                    "them in ONE block (check A16).")
                continue
            owner[name] = position
    missing = [name for name in latents if name not in owner]
    if missing:
        yield refuse(
            "A16", listed,
            f"{named}: {site}: does not cover {missing}; every latent "
            "inference.parameters declares must be in exactly one block. "
            f"{_A16_FROZEN[site]} Add it to a block, or drop it from "
            "inference.parameters (check A16).")


def _a17_message(named: str, site: str, position: int, steps: Any) -> str:
    """A17's refusal, whose middle clause depends on the VALUE.

    *"would be silently ignored"* is a claim about what the package would do
    without this check, and for a ``steps:`` the package refuses outright it
    is false -- so is the fix clause that sends the reader to ``engine:
    gradient``, which would leave the run refused for the second reason.
    Measured, ``plan.py:360-368``: ``steps=0``, ``steps=True``, ``steps='5'``
    and ``steps=1.5`` are all refused before a plan is settled at all.
    """
    head = (f"{named}: {site}[{position}] is solved by the conjugate engine, "
            "which has no inner steps, so steps: ")
    why = ("A conjugate block's estimate is one Wiener solve and its draw is "
           "one exact constrained realization -- there is no step count to "
           "tune, which is the whole advantage. ")
    if _t7_step_count(steps):
        return (f"{head}{steps!r} would be silently ignored. {why}Drop "
                "steps:, or declare engine: gradient if a gradient step was "
                "what you meant (check A17).")
    return (f"{head}{steps!r} is not a knob it has. {why}Drop steps:. Moving "
            f"to engine: gradient would not rescue {steps!r} either -- inner "
            "steps are a positive int on every engine (plan.py:360-368), so "
            "the block would be refused a second time (check A17).")


def _t7_engines(named: str, listed: str, site: str,
                entries: tuple[Mapping[str, Any], ...],
                latents: Mapping[str, Any], *,
                derive: bool) -> Iterable[Finding]:
    """A17, A18, A19 and the engine enum, in the order ``plan.py`` decides them.

    **``derive`` is False when the partition is wrong, and only the ENUM runs
    then.**  A17, A18 and A19 all read the latents a name resolves to, so on a
    broken partition they answer about names that do not exist -- an
    undeclared name reads as non-linear (:func:`_a18_linear` returns False for
    an absent latent) and produces an A18 *"mixes linear with non-linear"*
    refusal naming a latent nobody declared.  That is ``plan.py:539-541``'s
    own argument: *"a block naming an undeclared latent cannot have its engine
    derived at all, so the partition is settled first"*.

    The enum is not one of those.  ``engine:`` is checked by ``Block._check``
    (``plan.py:353-359``), which the package runs on EVERY block before
    ``SamplingPlan`` settles anything -- it reads the block alone and no
    latent at all.  Suppressing it behind the partition would cost a user with
    both faults a second round trip, which is what §2.3's collect-rather-than-
    raise design exists to prevent.
    """
    for position, entry in enumerate(entries):
        block_where = f"{listed}[{position}]"
        names = _t7_names(entry) or ()
        declared = entry.get("engine")

        # The enum, closed.  `check=""` -- schema §6 gives this no row of its
        # own, and §3.1 allows an id-less Finding.  It is not decoration:
        # `_engine_of` returns a declared engine unvalidated, so without this
        # an `engine: banana` block would reach the A17 test as
        # `engine == _T7_CONJUGATE` -> False and be silently accepted by this pass
        # while `plan.py:353-359` refuses it at P3.
        #
        # `isinstance` BEFORE `in`: `_ENGINES` is a frozenset, and `["x"] in
        # frozenset` raises TypeError on the unhashable list -- which inside
        # the pass is "check A16 RAISED" and costs the report every other
        # finding.  `Block._check` refuses a non-string engine by the same
        # clause, measured: "asks for engine=5; the engines are [...]".
        if declared is not None and not (isinstance(declared, str)
                                         and declared in _ENGINES):
            yield refuse(
                "", block_where,
                f"{named}: {site}[{position}] asks for engine: {declared!r}; "
                f"the engines are {sorted(_ENGINES)}. Leave engine: out and "
                "it is derived from linear: true on each member, which is the "
                "normal case -- an explicit engine is an override.")
            continue
        if not derive:
            continue

        engine = _engine_of(entry, latents)
        if engine == "":
            linear = [name for name in names if _a18_linear(latents, name)]
            other = [name for name in names if not _a18_linear(latents, name)]
            yield refuse(
                "A18", block_where,
                f"{named}: {site}[{position}] mixes declared-linear latents "
                f"{linear} with non-linear ones {other}, so which engine it "
                "takes cannot be derived. A conjugate solve needs the whole "
                "block affine; a gradient step does not exploit the linear "
                "members' structure at all, which for a high-dimensional "
                "linear block is the difference between tractable and "
                "hopeless. Split them into separate blocks, or declare "
                "engine: gradient to step the whole block by gradient "
                "deliberately (check A18).")
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
                    "A19", block_where,
                    f"{named}: {site}[{position}] asks for engine: conjugate, "
                    f"but {other} do not declare linear: true. The conjugate "
                    "machinery solves (A^T N^-1 A + S^-1)x = b, which is the "
                    "posterior only if the prediction really is affine in the "
                    "block -- and that claim belongs in the latent's "
                    "declaration, where check_linearity verifies it, not in a "
                    "run that asserts it. Declare linear: true and the claim "
                    "will be checked; leave it out and this block is stepped "
                    "by gradient (check A19).")
                continue

        # A17, as §2.6 item 3 decided it: "conjugate-ENGINE block", not the
        # schema's "all-linear block".  Measured both ways: an all-linear
        # block declared `engine: gradient` with `steps: 5` is ACCEPTED by the
        # package (SamplingPlan(('d','a'):gradient, ('w'):gradient)), and
        # `engine: conjugate` on a partly-linear block trips A19 above.
        if engine == _T7_CONJUGATE and entry.get("steps") is not None:
            yield refuse("A17", block_where,
                         _a17_message(named, site, position,
                                      entry.get("steps")))


@register("A16", "A17", "A18", "A19")
def _blocks(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A16-A19: a ``plan.*`` run's blocks, against the declared latents.

    The first check in the layer that needs ``runs[]`` and ``inference:`` at
    the same time.  Registered under four ids because ``Report.checks()`` is
    "the ids that fired" and a reader asking "did A17 run" must get an answer;
    ``preflight()`` calls this function once.

    **A run declaring ``expect: refuse`` is left alone**, and that is not
    politeness.  ``execute_run`` (``exits.py:292-302``) runs such a run's
    executor and CAPTURES its error as the run's product -- the run is an
    assertion ABOUT the refusal.  A P-1 refusal makes the whole document
    unloadable, so the assertion could never be made; measured,
    ``tests/config/test_config_exits_plan.py:107-112`` is exactly that
    document (``blocks: [{names: [g, ghost]}]`` under ``expect: refuse``) and
    it asserts the captured error names ``ghost``.

    Only ``plan.*`` kinds are read.  ``blocks:`` is a ``plan.*`` option, so a
    ``kind: fisher`` run carrying a stray one is Task 3's ``A1.runs`` hole and
    not A16's; refusing it here would give the user two refusals for one typo.

    **Variant layers are not walked.**  ``load_document`` calls the pass on
    the variant-APPLIED document (``document.py:68``), so a selected variant
    is read; an unselected one is A1's row and Task 3's ``_variant_text``, not
    this one's.
    """
    latents = _latents(document)
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
            partition = list(_a16_partition(named, listed, site, entries,
                                            latents))
            yield from partition
            yield from _t7_engines(named, listed, site, entries, latents,
                                   derive=not partition)


# --- Task 8: the prior gates, and the seed asymmetry ------------------------


def _a20_joint_over(document: Mapping[str, Any]) -> tuple[str, ...]:
    """The latent names ``inference.joint_prior`` covers, or ``()``.

    ``{jeffreys: {over: [...]}}`` is the grammar and the only one:
    ``transforms._joint_prior`` (``:298-321``) refuses anything else, and
    ``{kind: jeffreys, names: [...]}`` is refused by name there.  So coverage
    is a list of strings in the document and nothing has to be built to read
    it.

    Every level is type-tested rather than subscripted.  ``_structural``
    guarantees a SECTION is present, never that it is a mapping (Task 4's
    measurement), and ``over:`` is user text that reaches this function
    before ``JeffreysPrior`` ever sees it -- inside the pass a ``TypeError``
    becomes "check A20 RAISED" and discards every other finding.  A shape the
    grammar refuses reads as no coverage, which stands this check down and
    leaves the sentence to ``_joint_prior``.

    **One hole, recorded rather than closed.**  ``over: da`` -- a bare YAML
    scalar -- is read by ``_joint_prior`` as ``tuple("da") == ('d', 'a')``
    (measured), so such a document does reach ``JeffreysPrior`` and this
    reader answers ``()`` for it.  Splitting a string into characters is an
    accident of ``tuple()`` rather than a grammar, and A20 telling somebody
    who wrote ``over: da`` that their joint prior "covers ['d', 'a']" would
    name a coverage they did not write.  A non-string MEMBER of a real list
    is dropped for the same reason and the rest of the list is still read:
    ``over: [d, 7]`` covers ``['d']`` here, and which name is wrong is
    ``JeffreysPrior.validate_against``'s sentence.
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
    over = body.get("over")
    if not isinstance(over, list):
        return ()
    return tuple(name for name in over if isinstance(name, str))


def _a23_prior_free(latents: Mapping[str, Any], names: Iterable[str],
                    covered: tuple[str, ...] = ()) -> list[str]:
    """The names among ``names`` that declare no prior this route accepts.

    ``Latent.prior`` comes from ``spec.get("prior")`` and from nowhere else
    (``parameters.py:195``; ``_parse_prior`` returns None for a missing key at
    ``:69-70``), so "does this latent declare a prior" is a text question.
    ``covered`` is non-empty only for the ``nuts`` route, which is the one
    route that counts ``inference.joint_prior`` coverage as a prior --
    ``to_numpyro_model`` accepts a covered latent (``numpyro_bridge.py:66-79``)
    and ``simulate_pairs`` does not (``npe.py:111-118``).

    **``names`` must already be names the document DECLARES.**  An absent
    latent reads as prior-free here, and on the ``plan.sample`` leg -- the one
    route whose names come from a block rather than from
    ``inference.parameters`` -- that would put an A23 refusal beside A16's
    *"names 'zzz', which inference.parameters does not declare"*: one typo,
    two refusals, two different fixes.  The caller filters; this function
    cannot, because on the other three routes ``names`` IS the declaration.
    """
    return [name for name in names
            if latents.get(name, {}).get("prior") is None
            and name not in covered]


@register("A20", "A21", "A23")
def _prior_gates(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A20, A21 and A23 -- and A20/A21 make two of A23's four legs moot.

    **The order inside this function is the decision §2.6 item 4 records, and
    it is structural rather than positional.**  A20 refuses
    ``inference.joint_prior`` beside ANY ``kind: plan.*``
    (``plan.py:588-641``, unconditional -- ``_refuse_split_joint_prior``
    chooses only its wording from whether the partition splits the prior) and
    A21 refuses it beside ``fisher`` with ``space: true``
    (``rheplicant/inference/uncertainty.py:313-326`` -- ``inference/``, not
    ``config/sections/``).  So a run refused by either ``continue``s and never
    reaches its A23 leg: the joint-prior branches of A23's
    ``plan.sample``-gradient and ``fisher(space=)`` legs are UNREACHABLE, not
    merely later.  An implementation that ordered them the other way would
    tell a ``joint_prior`` + ``plan.sample`` document that its latents
    "declare no prior", contradicting A20's refusal of the same document, and
    the two refusals name different edits.

    ``nuts`` and ``npe`` keep their own gate at
    ``posterior_support._sampled_space`` (``:68-149``) as the P3 second
    opinion; this function does not call it and does not change it -- it
    needs a BUILT space (``:76``), which P-1 may not make.

    **A run declaring ``expect: refuse`` is left alone**, for the reason
    :func:`_blocks` gives and with a document to point at:
    ``execute_run`` (``exits.py:293-303``) captures such a run's error as its
    product, and a P-1 refusal makes the whole document unloadable, so the
    assertion could never be made.  ``posterior_helpers.joint_prior_document``
    is exactly that document -- ``kind: npe`` under ``expect: refuse`` beside
    a ``kind: nuts`` that runs, over ONE joint-prior space -- and it is what
    ``test_config_exits_npe.py::TestThePriorGate`` reads.  This clause was not
    in the task this function was written from; without it that class goes
    red.  The guard is per RUN, so the nuts run of that same document is
    still read.
    """
    latents = _latents(document)
    covered = _a20_joint_over(document)
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
                "A20", where,
                f"{named}: inference.joint_prior covers {list(covered)}, and "
                f"kind: {kind} does not evaluate a joint prior -- each block's "
                "conditional is built from the latent's OWN prior:, and a "
                "covered latent declares none, so the density contributes "
                "exactly zero. The sweep would run, settle, and report a "
                "converged chi-squared computed entirely from blocks that "
                "never saw the prior. kind: nuts is the exit that evaluates "
                "it; use that, or drop inference.joint_prior (check A20).")
            continue

        if covered and kind == "fisher" and run.get("space") is True:
            yield refuse(
                "A21", where,
                f"{named}: inference.joint_prior covers {list(covered)}, and "
                "space: true means 'add the declared priors' curvature to "
                "this matrix'. A Jeffreys prior is DEFINED as sqrt(det of "
                "that matrix), so adding it would put it inside its own "
                "definition: what comes back is not the posterior precision "
                "it would be labelled as, and it is finite, symmetric and "
                "positive definite, so nothing downstream would say "
                "otherwise. Drop space: true -- the likelihood Fisher is what "
                "the prior is built from -- or read the posterior with "
                "kind: nuts (check A21).")
            continue

        if kind == "nuts":
            missing = _a23_prior_free(latents, latents, covered)
            because = ("and no inference.joint_prior covering them"
                       if covered else "and this document declares no "
                                       "inference.joint_prior")
        elif kind == "npe":
            missing = _a23_prior_free(latents, latents)
            because = ("and kind: npe SIMULATES a bank from each latent's OWN "
                       "prior, consulting inference.joint_prior not at all")
        elif kind == "fisher" and run.get("space") is True:
            missing = _a23_prior_free(latents, latents)
            because = ("and space: true asks for a posterior precision, which "
                       "a prior-free latent has no row of")
        elif kind == "plan.sample":
            # `_t7_entries`, not `run.get("blocks") or ()`: a `blocks: 5`
            # raises on iteration and a `blocks: "nope"` iterates into
            # characters, and a malformed list is one `exits._blocks`
            # (`:180-202`) refuses in its own words.  `warm_start.blocks` is
            # NOT a site here, and that is measured rather than forgotten:
            # `require_priors` is called from `SamplingPlan.sample`
            # (`plan.py:1064-1066`) and a warm start is `.estimate()`d.
            missing = sorted({
                name
                for entry in (_t7_entries(run.get("blocks")) or ())
                if _engine_of(entry, latents) == _T7_GRADIENT
                for name in _a23_prior_free(
                    latents,
                    [one for one in (_t7_names(entry) or ())
                     if one in latents])})
            because = ("and a block stepped by the gradient engine needs a "
                       "prior on every member -- the potential is flat in a "
                       "prior-free latent and the chain wanders without any "
                       "diagnostic saying so")
        else:
            continue

        if missing:
            yield refuse(
                "A23", where,
                f"{named}: kind: {kind} draws a POSTERIOR, and "
                f"inference.parameters declares {missing} with no prior: "
                f"{because}. A prior-free latent is a free parameter, which "
                "the calibrator exits (kind: optimize, kind: plan.estimate) "
                "fit and a posterior cannot. Give each one a prior:, or run "
                "one of those (check A23).")


#: Which ``runs[].kind`` needs a seed on the RUN.  ``npe`` is absent on
#: purpose: it draws four times and declares its seeds per subsection in
#: ``inference.npe:`` (``npe.py:216-228``), and a run-level ``seed:`` on it is
#: refused rather than required.  ``condition`` is absent on purpose too: it
#: takes an OPTIONAL seed (``conjugate.py:568-572``) and is correctly outside
#: A29.
_A29_SEEDED_KINDS: frozenset[str] = frozenset({"plan.sample", "conjugate.gcr",
                                               "nuts"})

#: The subsections of ``inference.npe:`` that declare a seed, in the order
#: ``parse_npe`` (``npe.py:364-368``) reads them.  ``embed:`` is the fifth
#: member of ``_NPE_KEYS`` and is absent here because it declares none --
#: measured, ``_seeded`` is called at ``npe.py:240``, ``:251``, ``:263`` and
#: ``:283`` and nowhere else.
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
    and the other two through ``draws._seed_name`` (``:98-121``), which was
    already pure: it takes a dict and a prefix, refuses a missing key, a
    literal seed and a name outside ``runtime.seeds.``, and resolves nothing.
    The RESOLUTION (``seed_for``) stays where the context is.

    What moves is the phase.  ``nuts``'s seed is checked at ``nuts.py:300``,
    i.e. AFTER ``to_numpyro_model`` is built at ``:287`` -- the most expensive
    object that executor makes is constructed before the cheapest key on the
    run is looked at -- and all five sit behind ``build_resources``, which is
    90.9 % of ``load_document`` (§2.7).

    Two things are deliberately NOT A29's, both measured:
    ``inference.observed.<name>.realise.seed`` (``observed.py:63``) and a
    ``{normal: {seed: ...}}`` value node (``draws.py:144``) also go through
    ``_seed_name``, and neither is a ``runs[].kind`` -- schema §6's A29 row
    is about the run kinds and the npe subsections, and answering for the
    other two here would put this check in front of two grammars that own
    their own sentences.
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
        # `RunSpec.options` is exactly this (`runs.py:113-114`), and
        # `_RUN_KEYS` is imported rather than restated -- measured, it is
        # {expect, kind, name, on, reuse, variant}, and a sixth key added
        # there would otherwise reach `_seed_name` as an option here while
        # travelling on the spec at P3.
        options = {key: value for key, value in run.items()
                   if key not in _RUN_KEYS}
        # THE GATE AND `_seed_name` ARE ALTERNATIVES, NOT A PAIR.
        # `conjugate.gcr` is in both `_A29_SEEDED_KINDS` and the gate chain,
        # so a seedless gcr run would be described TWICE -- once by
        # `_a29_gcr_needs_a_seed` and once by `_seed_name`, in two different
        # voices, for one missing key.  `gated` is what makes the bespoke
        # refusal the only one the user reads.  It is set only when the gate
        # FIRED, so a gcr run with a seed of the wrong FORM still reaches
        # `_seed_name`, which is the leg that decides form.
        gated = False
        gate = (_a29_estimate_takes_no_seed if kind == "plan.estimate"
                else _a29_gcr_needs_a_seed if kind == "conjugate.gcr"
                else _a29_npe_takes_no_run_seed if kind == "npe"
                else None)
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
    if not isinstance(npe, Mapping) or "npe" not in _kinds(document):
        return
    for subsection in _A29_NPE_SUBSECTIONS:
        body = npe.get(subsection)
        # An ABSENT or malformed subsection is `npe._subsection`'s
        # (``:199-207``), whose sentence is that the subsection is required
        # rather than that a seed is missing.  Standing down leaves the
        # reader the fault they actually have.
        if not isinstance(body, Mapping):
            continue
        where = f"inference.npe.{subsection}"
        try:
            _seed_name(dict(body), where)
        except ConfigError as refusal:
            yield refuse("A29", where, f"{refusal} (check A29).")
