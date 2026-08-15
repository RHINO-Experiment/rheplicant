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
