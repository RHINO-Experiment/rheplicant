"""The checks that need ``runs[]`` and ``inference:`` at the same time.

No function in this layer sees both today (the plan's finding 2):
``parse_runs`` sees ``runs`` alone and inspects the SIX keys of
``_RUN_KEYS`` (``runs.py:25``: ``name``, ``kind``, ``variant``, ``on``,
``reuse``, ``expect``; the plan's "five" was inherited and is wrong, and
``_one`` validates all six) -- every other run key travels untouched in
``RunSpec.options`` (``runs.py:113-117``) and is
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
from types import SimpleNamespace
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.sections.exit_support import _number
from rheplicant.config.sections.transforms import _whole

__all__: list[str] = []

#: ``engines.CONJUGATE`` and ``engines.GRADIENT`` (``engines.py:62``, ``:66``),
#: written out because this module may not import that package -- see the
#: module docstring for the guard that measures it.
_T7_CONJUGATE: str = "conjugate"
_T7_GRADIENT: str = "gradient"

#: The engines a block may ask for, closed.  ``_BLOCK_KEYS``
#: (``exits.py:165``) accepts ANY string for ``engine:``, so today
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

#: The keys a block entry takes -- ``_BLOCK_KEYS`` (``exits.py:165``), copied
#: for the same reason :data:`_ENGINES` is: reaching it means importing
#: ``sections/exits``, which foot-imports ``conjugate``, ``diagnostics``,
#: ``npe`` and ``nuts``, and measured that adds ~30 ms to every ``import
#: rheplicant.config`` for one frozenset.  ``test_the_block_key_set_is_the_
#: packages_own`` imports ``exits._BLOCK_KEYS`` in the TEST and pins it, so a
#: fifth block key turns that red rather than leaving this check reading an
#: entry the grammar rejects.
_T7_BLOCK_KEYS: frozenset[str] = frozenset({"names", "steps", "engine",
                                            "learning_rate"})

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
        "(exits.py:287-288), so an omitted latent sits at its declared init "
        "for the whole warm estimate, and warm_start.move: can only carry "
        "over a value that estimate produced."),
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
    * ``name`` is filled in by ``parse_runs``' own rule -- the entry's
      ``name`` when it has one, else the kind (``runs.py:115``) -- because
      every refusal in this layer is prefixed ``runs['<name>']:`` and three
      tasks would otherwise each re-derive it.  A NEW dict is built; the
      caller's document is never mutated.

      **``runs.py:115``'s test is ``is not None``, not "is a string"**, and
      the difference is one document: a non-string ``name:`` is REFUSED at
      ``runs.py:101-102`` (*"name: is a string; got 7"*) rather than
      defaulted, so ``parse_runs`` has no such run to name.  This function
      does fill it -- ``name: 7`` is prefixed ``runs['<kind>']`` -- because
      ``parse_runs`` runs only on the ``run_document`` path (``runs.py:149``)
      and a ``load_document`` caller would otherwise get a check that
      declined on a document nothing else refuses.  The cost is a prefix
      naming a run the user did not write, on a document already refused for
      its ``name:``; recorded rather than repaired, because repairing it is
      a stand-down that loses the block checks on that document entirely.
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

    ``exits.py:198-203``'s own three tests -- a list, non-empty, all strings
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

    ``exits._blocks`` (``exits.py:181-207``) refuses **five** shapes before a
    ``Block`` is ever built, and all five are mirrored here: a ``blocks:``
    that is not a list and an empty one (``:184-186``), an entry that is not a
    mapping (``:189-191``), an entry carrying a key a block does not take
    (``:192-197``), and a malformed ``names:`` (``:198-203``).  Each has a
    sentence naming the fault; a partition answer in front of one would say
    *"blocks: does not cover ['d', 'a', 'w']"* -- true, useless, and offering
    a fix ("add it to a block") that is not the fault.

    The unknown-key one was missed in the first draft of this function, whose
    docstring said "four shapes": measured, ``blocks: [{names: [d], step: 5}]``
    passed the four and reached the engine derivation, so a document whose
    real fault is a typo'd ``step:`` was answered with a coverage sentence.

    **All or nothing per list.**  One malformed entry makes the whole
    partition undecidable: the names it would have owned are unknown, so
    every OTHER entry's coverage answer would be computed against a set that
    is missing them.
    """
    if not isinstance(node, list) or not node:
        return None
    for entry in node:
        if not isinstance(entry, Mapping):
            return None
        if set(entry) - _T7_BLOCK_KEYS:
            return None
        if _t7_names(entry) is None:
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


def _t7_warm_start(run: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The ``warm_start`` whose ``blocks:`` the executor would actually READ.

    ``_run_plan`` reaches ``warm_start.blocks`` (``exits.py:287``) only after
    five earlier refusals, and three of them are this check's business.  Each
    of the three means the warm block list is **never read at all**, so a
    partition answer about it is an answer about a structure the package
    discards -- and it is the only sentence the reader gets, because none of
    the three is hoisted to P-1 by anything:

    * the run's own ``blocks:`` grammar (``exits.py:250``, into
      ``_blocks``/``:181-207``).  Measured: a ``plan.sample`` with no
      ``blocks:`` at all, and one with ``blocks: "nope"``, both used to earn
      *"runs['s']: warm_start.blocks: does not cover [...]"* -- telling a user
      who omitted a key about a different key, and never about theirs;
    * ``warm_start.kind`` (``exits.py:268-271``): anything but
      ``plan.estimate`` and the whole warm start is refused;
    * ``warm_start.move`` (``exits.py:273-278``): required, and without it the
      warm start never runs.

    **``n_sweeps`` (``exits.py:255-256``) and the seed (``:257``) are NOT
    gates**, and that is the line rather than an omission.  They are
    independent run keys, not the warm start's own grammar: a document
    missing one AND carrying a broken warm partition has two faults the user
    must fix either way, and reporting both is what collect-rather-than-raise
    is for (§2.3).  Gating on them would trade one round trip for another.

    ``warm_start`` is read on ``plan.sample`` only, because ``_ESTIMATE_KEYS``
    (``exits.py:166``) does not take it and Task 3's ``A1.runs`` already
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
    if not isinstance(move, list) or not move or not all(
            isinstance(name, str) for name in move):
        return None
    return warm


def _t7_sites(run: Mapping[str, Any]) -> tuple[tuple[str, tuple], ...]:
    """Every ``blocks:`` list in this run that reaches a ``SamplingPlan``.

    **Two, not one.**  ``exits.py:287-288`` builds ``_blocks(f"{where}:
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
    """A16: the partition, in the order ``plan.py:544-586`` settles it.

    Three legs, one id.  The schema row (line 1193) describes two of them --
    "every latent appears in exactly one block" -- and the third, a block
    naming a name ``inference.parameters`` never declared, is
    ``plan.py:545-558``'s and is refused FIRST there (the covered pair is
    ``:560-574`` and ``:576-584``).  A fourth shape,
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
    refusal naming a latent nobody declared.  That is ``plan.py:541-542``'s
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
    politeness.  ``execute_run`` (``exits.py:311-321``) runs such a run's
    executor and CAPTURES its error as the run's product -- the run is an
    assertion ABOUT the refusal.  A P-1 refusal makes the whole document
    unloadable, so the assertion could never be made; measured,
    ``tests/config/test_config_exits_plan.py:108-113`` is exactly that
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

    ``{jeffreys: {over: ...}}`` is the grammar and the only one:
    ``transforms._joint_prior`` (``:298-320``) refuses anything else, and
    ``{kind: jeffreys, names: [...]}`` is refused by name there.  So coverage
    is text in the document and nothing has to be built to read it.

    **``tuple(over)`` is what the package does, and this mirrors it rather
    than narrowing it.**  ``_joint_prior`` ends
    ``JeffreysPrior(over=tuple(body["over"]), **kwargs)`` (``:320``), so FIVE
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


def _a23_prior_free(latents: Mapping[str, Any], names: Iterable[str],
                    covered: tuple[str, ...] = ()) -> list[str]:
    """The names among ``names`` that declare no prior this route accepts.

    ``Latent.prior`` comes from ``spec.get("prior")`` and from nowhere else
    (``parameters.py:195``; ``_parse_prior`` returns None for a missing key at
    ``:71-72`` and for no other value), so "does this latent declare a prior"
    is a text question -- and a prior that is present but malformed is a
    DECLARED prior, refused by ``_parse_prior`` in its own words.
    ``covered`` is non-empty only for the ``nuts`` route, which is the one
    route that counts ``inference.joint_prior`` coverage as a prior --
    ``to_numpyro_model`` accepts a covered latent (``numpyro_bridge.py:68-73``)
    and ``simulate_pairs`` does not (``npe.py:111-117``).

    **``names`` must already be names the document DECLARES WELL** -- the
    keys of :func:`_a23_latents`, or a subset of them.  An absent latent reads
    as prior-free here, and on the ``plan.sample`` leg -- the one route whose
    names come from a block rather than from ``inference.parameters`` -- that
    would put an A23 refusal beside A16's *"names 'zzz', which
    inference.parameters does not declare"*: one typo, two refusals, two
    different fixes.  The caller filters; this function cannot, because on the
    other three routes ``names`` IS the declaration.
    """
    return [name for name in names
            if latents.get(name, {}).get("prior") is None
            and name not in covered]


def _a23_message(named: str, kind: str, missing: list[str], because: str,
                 covered: tuple[str, ...]) -> str:
    """A23's refusal: one shape, and three clauses the document decides.

    **The verb is per route.**  ``kind: fisher`` with ``space: true`` does not
    draw anything -- ``fisher_information`` computes a posterior precision
    (``uncertainty.py:346-357``) -- so a shared *"draws a POSTERIOR"* is false
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
    """
    verb = ("computes a POSTERIOR precision" if kind == "fisher"
            else "draws a POSTERIOR")
    overlap = [name for name in missing if name in covered]
    if not covered:
        fix = ("A prior-free latent is a free parameter, which the calibrator "
               "exits (kind: optimize, kind: plan.estimate) fit and a "
               "posterior cannot. Give each one a prior:, or run one of those")
    elif overlap:
        fix = ("A prior-free latent is a free parameter, which a posterior "
               f"cannot fit. inference.joint_prior already covers {overlap}, "
               "and a latent may not be covered AND declare a prior: of its "
               "own (check A22), so declaring one is not the fix here: run "
               "kind: nuts, which reads joint-prior coverage as a prior, or "
               "drop inference.joint_prior and give every latent a prior: of "
               "its own. Not kind: plan.estimate or kind: plan.sample -- A20 "
               "refuses both beside a joint prior")
    else:
        fix = ("A prior-free latent is a free parameter, which a posterior "
               "cannot fit. Give each one a prior:, or add it to "
               "inference.joint_prior.over, which already covers "
               f"{list(covered)}. Not kind: plan.estimate or kind: "
               "plan.sample -- A20 refuses both beside a joint prior, so "
               "switching would trade this refusal for that one")
    return (f"{named}: kind: {kind} {verb}, and inference.parameters declares "
            f"{missing} with no prior: {because}. {fix} (check A23).")


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
    ``posterior_support._sampled_space`` (``:68-150``) as the P3 second
    opinion; this function does not call it and does not change it -- it
    needs a BUILT space (``:77``), which P-1 may not make.

    **A run declaring ``expect: refuse`` is left alone**, for the reason
    :func:`_blocks` gives and with a document to point at: ``execute_run``
    (``exits.py:301``) runs such a run's executor and captures its error as
    the run's product (``:314-316``), and a P-1 refusal makes the whole
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
    if not all(name in latents and latents[name].get("prior") is None
               for name in covered):
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
            # "no inference.joint_prior COVERS them", never "this document
            # declares none": `covered` is emptied above for a joint prior
            # the package would refuse, and a document that declares an
            # `over: [zzz]` does declare one.  The earlier wording was false
            # on exactly that document.
            because = ("and the inference.joint_prior this document declares "
                       f"covers {list(covered)} and not them"
                       if covered else "and no inference.joint_prior covers "
                                       "them")
        elif kind == "npe":
            missing = _a23_prior_free(latents, latents)
            because = ("and kind: npe SIMULATES a bank from each latent's OWN "
                       "prior, consulting inference.joint_prior not at all")
        elif kind == "fisher" and run.get("space") is True:
            missing = _a23_prior_free(latents, latents)
            # The package's own way out, which A23 owes the reader as well as
            # the calibrator exits: `uncertainty.py:354` says drop `space=`
            # and what comes back is exactly the likelihood matrix.
            because = ("and space: true asks for a posterior precision, which "
                       "a prior-free latent has no row of -- drop space: and "
                       "what comes back is the likelihood Fisher, which is "
                       "that same matrix without the priors in it")
        elif kind == "plan.sample":
            # `_t7_entries`, not `run.get("blocks") or ()`: a `blocks: 5`
            # raises on iteration and a `blocks: "nope"` iterates into
            # characters, and a malformed list is one `exits._blocks`
            # (`:181-207`) refuses in its own words.  `warm_start.blocks` is
            # NOT a site here, and that is measured rather than forgotten:
            # `require_priors` is called from `SamplingPlan.sample`
            # (`plan.py:1064-1066`) and a warm start is `.estimate()`d
            # (`exits.py:287-288`).
            #
            # `== _T7_GRADIENT` and never `!= _T7_CONJUGATE`: `_engine_of`
            # answers `""` for a block whose engine cannot be derived (A18's
            # case) and returns a DECLARED engine unvalidated, so the
            # complement also selects a mixed block and an `engine: banana`
            # one -- putting an A23 refusal beside A18's "mixes
            # declared-linear latents" and beside the enum clause's "asks for
            # engine: 'banana'", about a block whose engine nobody knows.
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
            yield refuse("A23", where,
                         _a23_message(named, kind, missing, because, covered))


#: Which ``runs[].kind`` needs a seed on the RUN.  ``npe`` is absent on
#: purpose: it draws four times and declares its seeds per subsection in
#: ``inference.npe:`` (``npe._seeded``, ``:246-258``), and a run-level
#: ``seed:`` on it is refused rather than required.  ``condition`` is absent
#: on purpose too: ``_CONDITION_KEYS`` (``conjugate.py:108``) carries ``seed``
#: and ``_run_condition`` reads it only ``if "seed" in run.options``
#: (``:685-690``), so it is OPTIONAL there and correctly outside A29 --
#: ``condition_estimate``'s ``key`` defaults internally, which that function's
#: own docstring argues at ``:645-649``.
_A29_SEEDED_KINDS: frozenset[str] = frozenset({"plan.sample", "conjugate.gcr",
                                               "nuts"})

#: The subsections of ``inference.npe:`` that declare a seed, in the order
#: ``parse_npe`` (``npe.py:394-398``) reads them.  ``embed:`` is the one
#: member of ``_NPE_KEYS`` (``npe.py:108``) missing here, and it is missing
#: because it declares no seed -- measured, ``_seeded`` is called at
#: ``npe.py:270``, ``:281``, ``:293`` and ``:313`` and nowhere else.  (It is
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
    and the other two through ``draws._seed_name`` (``:98-121``), which was
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

    **And one leg of A29 is NOT hoisted, which is a hole in this plan's own
    thesis rather than a decision about wording.**  An ABSENT
    ``inference.npe.<sub>`` is left to ``npe._subsection`` (``:230-243``),
    whose sentence is that the subsection is required rather than that a seed
    is missing -- but ``parse_npe`` runs inside ``build_inference``, which is
    after ``build_resources``, so that refusal still costs the beam.  Closing
    it means A29 answering for a section the user has not written, which is a
    worse sentence; recorded for §6's ledger.
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


# --- Task 9: the counts a run declares, and the six knobs A25 never named ---


#: ``CHECK_ONCE`` (``plan.py:148``) and ``CHECK_EACH_SWEEP`` (``:151``),
#: written out for the reason :data:`_ENGINES` is: this module may not import
#: ``rheplicant.inference`` at scope, and a module-level constant cannot defer
#: an import the way a function body can.  ``MIN_DRAWS`` is the one name that
#: is IMPORTED rather than written -- plan §2.5 requires it by name -- and it
#: can be, because it is wanted inside :func:`_counts`, where a deferred
#: import costs one ``sys.modules`` lookup.
#: ``test_check_identifiability_is_a_closed_enum_read_from_the_package``
#: imports both names in the TEST, and
#: ``test_the_package_guard_this_enum_mirrors_is_still_that_guard`` reads
#: ``plan.py:708``'s expression itself, so a third accepted mode turns those
#: red rather than leaving this pass refusing a document the package runs.
_T9_CHECK_ONCE: str = "once"
_T9_CHECK_EACH_SWEEP: str = "each_sweep"

#: What ``check_identifiability:`` takes, in the package's own order.
#:
#: **A TUPLE and not a frozenset, and neither reason is style.**  ``x in
#: frozenset`` RAISES ``TypeError`` on an unhashable ``x`` --
#: ``check_identifiability: [once]`` is a document a user can write, and
#: inside the pass a ``TypeError`` becomes "check A25 RAISED" and discards
#: every other finding.  And ``0 in frozenset({False, ...})`` is ``True``,
#: because ``hash(0) == hash(False)``, while ``plan.py:708`` tests ``check is
#: not False`` -- an IDENTITY -- and so refuses ``check_identifiability: 0``.
#: A frozenset would therefore accept a document the package refuses AND
#: crash on another.  :func:`_a25_check_mode` mirrors the package's two-part
#: test rather than this tuple's membership; the tuple is what the test pins.
_A25_CHECK_MODES: tuple[Any, ...] = (False, _T9_CHECK_ONCE,
                                     _T9_CHECK_EACH_SWEEP)

#: ``(key, kind, minimum)`` per ``runs[].kind``, for every numeric knob that
#: reaches the package.  ``sorted(set(_ESTIMATE_PASSTHROUGH) |
#: set(_SAMPLE_PASSTHROUGH))`` (``exits.py:175-178``) is eight names --
#: ``check_identifiability``, ``max_iter``, ``min_sweeps``, ``rhat_max``,
#: ``solve_guard``, ``solve_tol``, ``tol``, ``warmup`` -- of which A25's
#: schema row names two, and ``n_sweeps`` is the ninth: it reaches ``_number``
#: at ``exits.py:297`` with NO ``minimum=``, so today ``n_sweeps: 0`` is the
#: package's sentence at P3.  ``check_identifiability`` is the one of the
#: eight that is not numeric, and :data:`_A25_CHECK_MODES` has it.
#:
#: The floors are the package's own: ``plan.py:900`` (``max_iter >= 1``),
#: ``:909-911`` (``1 <= min_sweeps <= max_iter``), ``:1043`` (``n_sweeps >=
#: 1``), ``:1048`` (``warmup >= 0``), ``nuts.py:282`` (both nuts counts
#: ``>= 1``).  The three tolerances carry ``0.0`` because nothing in
#: ``config/`` refuses a negative one -- grepped: ``solve_tol`` and
#: ``solve_guard`` appear only in the key sets and the passthrough tuples --
#: so it is forwarded raw into a solver whose bound is an
#: ``equinox.error_if`` (``engines.py:286``, ``linear.py:1504``), inside jit.
#: ``rhat_max`` carries ``0.0`` and NOT a strictly-positive floor: see
#: :func:`_counts`' residues.
_A25_KNOBS: dict[str, tuple[tuple[str, type, float | None], ...]] = {
    "plan.estimate": (("max_iter", int, 1), ("min_sweeps", int, 1),
                      ("tol", float, 0.0), ("solve_tol", float, 0.0),
                      ("solve_guard", float, 0.0)),
    "plan.sample": (("n_sweeps", int, 1), ("warmup", int, 0),
                    ("rhat_max", float, 0.0), ("solve_tol", float, 0.0),
                    ("solve_guard", float, 0.0)),
    "nuts": (("num_samples", int, 1), ("num_warmup", int, 1)),
}

#: The one row of :data:`_A25_KNOBS` the package does not read
#: unconditionally.  ``plan.py:909`` gates ``min_sweeps`` on ``tol is not
#: None`` and ``:943-946`` short-circuits on the same test, so beside ``tol:
#: null`` a ``min_sweeps`` is forwarded, validated by nothing and consulted
#: by nothing: ``min_sweeps: 0`` with ``tol: null`` RUNS.  Checking it anyway
#: would refuse a document the package runs, for a knob that does nothing --
#: the defect Task 5 shipped twice.  The task body's table applied this row
#: unconditionally while its own measurement table said it must not.
_A25_TOL_GATED: frozenset[str] = frozenset({"min_sweeps"})

#: ``inference.npe``'s two counts, and the subsection each lives in.  They go
#: through ``transforms._whole`` (``:50``), not ``_number``: that is the
#: binding ``npe._count`` (``:240-244``) already uses, and a second reading
#: here would be the ``_number``-vs-``_whole`` divergence the 2C ledger names.
_A25_NPE_COUNTS: tuple[tuple[str, str], ...] = (("bank", "n_simulations"),
                                                ("sample", "n_draws"))


def _t9_whole_number(value: Any) -> bool:
    """Is this an ``int`` the package would treat as a count?

    ``bool`` refused, because ``isinstance(True, int)`` is True and
    ``plan.py:900`` and ``:1043`` both read ``isinstance(..., int)`` beside a
    comparison -- so ``max_iter: true`` would sail through a bare
    ``isinstance`` here and then arrive as the count ``1``.  Bound once and
    used by :func:`_a24_kept_draws` and by the ``min_sweeps``/``max_iter``
    pair, which are the two places this module does arithmetic on user text.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _a25_check_mode(mode: Any) -> bool:
    """``plan.py:708``'s own test, mirrored and negated -- ``check is not
    False and check not in (CHECK_ONCE, CHECK_EACH_SWEEP)``.

    Identity on ``False`` and ``==`` membership over a TUPLE for the two
    strings, both because the package does it that way and both because the
    obvious spelling is wrong: see :data:`_A25_CHECK_MODES` for the ``0`` and
    the unhashable list a frozenset gets wrong in opposite directions.
    """
    return mode is False or mode in _A25_CHECK_MODES[1:]


def _a25_bounded(where: str, name: str, key: str, value: Any, *,
                 kind: type, minimum: float | None) -> Finding | None:
    """``exit_support._number``, with its refusal turned into a Finding.

    ``_number`` (``exit_support.py:73-98``) reads exactly one attribute off
    the object it is handed -- ``run.name``, for the ``runs['<name>']:``
    prefix, at ``:77``, ``:89`` and ``:95`` -- so a namespace carrying that
    name is the whole adapter.  **``name`` is the BARE run name, never the
    formatted ``runs['fit']``**: ``_number`` BUILDS the prefix itself, so
    handing it the finished string makes every A25 message read
    ``runs["runs['fit']"]:``.  Calling it rather than restating its two type
    refusals is what keeps ONE binding for "is this a whole number": the
    ``_number``-vs-``_whole`` divergence on the 2C ledger is two validators
    for one property disagreeing, and a third written here would be that
    defect with a new name.

    It also fixes a message.  ``plan.py:900`` reads ``not isinstance(
    max_iter, int) or max_iter < 1`` and reports only the second half, so
    ``max_iter: 2.5`` reaches the user as *"estimate() needs max_iter >= 1,
    got 2.5"* -- false, since 2.5 IS >= 1, and the real fault is the
    ``isinstance``.  ``_number``'s own wording for that case says what is
    wrong: *"is a whole number; got 2.5. It counts, so 2 and 2.5 are
    different runs"*.
    """
    try:
        _number(SimpleNamespace(name=name), key, value, kind=kind,
                minimum=minimum)
    except ConfigError as refusal:
        return refuse("A25", where, f"{refusal} (check A25).")
    return None


def _a25_bounds(where: str, name: str, prefix: str,
                rows: tuple[tuple[str, type, float | None], ...],
                spec: Mapping[str, Any]) -> list[tuple[str, Finding]]:
    """Every A25 finding ``rows`` earns on ``spec``, with the key that earned
    it -- the caller needs the key, because A24 is computed from two of them.

    An ABSENT key takes the package's default, which this layer does not
    restate.  A ``null`` is skipped too, and that is a RESIDUE rather than a
    rule: ``null`` is the package's own off-switch on exactly three of these
    rows -- ``tol`` (no convergence test at all, ``plan.py:874-878``),
    ``solve_guard`` (no condition-number estimate, ``:884-887``) and
    ``warmup`` (the ``n_sweeps // 2`` default, ``:1047``) -- and on the rest
    it is a typo the package refuses in its own voice: ``max_iter: null``
    reaches ``plan.py:900``'s ``isinstance`` and ``n_sweeps: null`` reaches
    ``_number`` at ``exits.py:297``.  Telling the three apart needs a fourth
    column on every row and a measurement per key, so the whole of ``null``
    is left where it is today.
    ``test_a_null_on_a_row_where_null_is_NOT_legal_is_left_to_the_package``
    is what stops that becoming a claim nothing defends.
    """
    live = spec.get("tol", 1) is not None
    found: list[tuple[str, Finding]] = []
    for key, kind_of, minimum in rows:
        if key not in spec or spec[key] is None:
            continue
        if key in _A25_TOL_GATED and not live:
            continue
        finding = _a25_bounded(where, name, f"{prefix}{key}", spec[key],
                               kind=kind_of, minimum=minimum)
        if finding is not None:
            found.append((key, finding))
    return found


def _a25_sites(run: Mapping[str, Any]) -> tuple[tuple[str, str, Mapping], ...]:
    """Every mapping on this run whose counts reach a ``SamplingPlan``.

    **Two, not one.**  ``exits.py:288-290`` calls ``SamplingPlan(space,
    *warm_blocks).estimate(..., **_passthrough(warm, _ESTIMATE_PASSTHROUGH))``
    -- so ``max_iter``, ``tol``, ``min_sweeps``, ``check_identifiability``,
    ``solve_tol`` and ``solve_guard`` are read off the WARM mapping and meet
    the same guards in the same method, at the same P3 behind the same beam.
    A25 written on ``runs[]`` alone would guard one route and leave its
    identical sibling open, which is the shape Task 7 found on ``blocks:``.

    Each entry is ``(the _A25_KNOBS row set, the message prefix, the
    mapping)``.  The warm site takes ``plan.estimate``'s rows because
    ``_passthrough`` hands it to ``estimate()``, and ``n_sweeps`` is not a
    ``_WARM_KEYS`` member at all, so no A24 arithmetic follows it.

    ``_t7_warm_start`` decides whether the executor reaches the warm start at
    all -- IMPORTED from Task 7 rather than re-derived, because two
    independently written "would this warm start be read?" predicates is the
    two-validators shape one function over.  Its gates are the executor's own
    (``exits.py:259-285``): anything they reject means the run is refused
    before ``_passthrough`` is evaluated, and an A25 about a mapping the
    package discards would be the only sentence the reader gets.
    """
    kind = run.get("kind")
    if not isinstance(kind, str) or kind not in _A25_KNOBS:
        return ()
    sites: list[tuple[str, str, Mapping]] = [(kind, "", run)]
    warm = _t7_warm_start(run)
    if warm is not None:
        sites.append(("plan.estimate", "warm_start.", warm))
    return tuple(sites)


def _a24_kept_draws(options: Mapping[str, Any]) -> tuple[int, int] | None:
    """``(kept, warmup)`` for a ``plan.sample`` run, or None if undecidable.

    **``n_sweeps // 2`` is RESTATED from ``plan.py:1047``**, because the
    config layer forwards ``warmup`` only when the document declares it
    (``_passthrough``, ``exit_support.py:223``), so there is nothing to ask.
    A restated default drifts silently: were the package to change it, every
    message this pass writes would still read "N sweeps minus M warmup" with
    the wrong M, and every test here would stay green because they restate it
    too.  ``test_the_restated_default_is_still_the_packages_own`` reads the
    expression back out of the package's source and is what catches that.

    ``plan.py:1047`` reads ``warmup is None`` rather than "was warmup
    declared", so a written-out ``warmup: null`` takes the default here as
    well -- and :func:`_counts`' "the default" clause is on the same test
    rather than on ``"warmup" in run``.

    Returns None when either count is not a whole number -- :func:`_counts`
    has already emitted A25 for that, and a second finding computed from a
    value it just refused would name a draw count nobody asked for.

    **The TYPE half is not the whole of it.**  A FLOOR violation gets through
    this guard: ``n_sweeps: -3`` is an int, so this returns ``(-1, -2)`` and
    A24 would read "keep -1 draw(s) (-3 sweeps minus -2 warmup)" beside the
    A25 that already refused the -3.  :func:`_counts` carries
    ``refused_counts`` for exactly that, and the reason it is there rather
    than here is that a floor is ``_number``'s to state, and this function
    would have to restate one to see it.
    """
    sweeps = options.get("n_sweeps")
    if not _t9_whole_number(sweeps):
        return None
    declared = options.get("warmup")
    if declared is None:
        warmup = sweeps // 2
    elif not _t9_whole_number(declared):
        return None
    else:
        warmup = declared
    return sweeps - warmup, warmup


@register("A24", "A25")
def _counts(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A24 and A25: every count a run declares, checked where it is written.

    A25's schema row names four clauses and this plan's §1 adds "the six
    passthrough keys A25 does not name".  Counted from ``exits.py:175-178``:
    ``sorted(set(_ESTIMATE_PASSTHROUGH) | set(_SAMPLE_PASSTHROUGH))`` is
    eight names, A25's row names ``max_iter`` and ``min_sweeps``, so the six
    are ``check_identifiability``, ``rhat_max``, ``solve_guard``,
    ``solve_tol``, ``tol`` and ``warmup``.  ``n_sweeps`` is the ninth and
    reaches ``_number`` at ``exits.py:297`` with no ``minimum=``.

    **A run declaring ``expect: refuse`` is NOT left alone**, and that is the
    one place this check departs from :func:`_blocks` and
    :func:`_prior_gates` deliberately.  Those two stand down because a real
    document in this repository would otherwise lose the assertion it exists
    to make (``test_config_exits_plan.py:108-113``,
    ``posterior_helpers.joint_prior_document``).  No document expects a count
    refusal, and the layer's own policy test says the general shape is the
    other way round: ``test_config_section_runs.py``'s
    ``test_a_text_decidable_refusal_is_not_a_runs_to_expect`` records that a
    P-1 refusal raising out of ``run_document`` rather than being captured is
    the CORRECT shape, because ``expect: refuse`` is for a run that fails
    when it RUNS.

    Residues, all three named so that no later task reads this as complete:

    * **``rhat_max: 0.0`` stays legal.**  It decides a chain's convergence
      verdict and no value of it is refused today.  This check closes the
      type and the negative half; ``0.0`` cannot be closed with ``_number``,
      whose ``minimum=`` is inclusive (``exit_support.py:93``: ``not value >=
      minimum``), and a strictly-positive floor written here would be a
      second validator for a bound ``_number`` owns.  A threshold nobody can
      satisfy is a warning rather than a refusal, and the warning channel's
      first consumers are Task 12's.
    * **``null`` on a row where ``null`` is not the package's own
      off-switch** -- see :func:`_a25_bounds`.
    * **``nuts``'s other three numeric knobs** -- ``num_chains``,
      ``thinning`` and ``target_accept_prob`` -- are outside A25's schema row
      and outside this plan's §1 wording.  They ARE checked, by ``_number``,
      at ``nuts.py:230-238``, which is P3 and behind the beam; hoisting them
      is a widening rather than a hole this check leaves in a rule it states.

    **Variant layers are not walked**, the same way :func:`_blocks` does not.
    ``load_document`` calls the pass on the variant-APPLIED document
    (``document.py:68``), so a SELECTED variant's counts are read here like
    any others.  An unselected one is not: Task 3's ``_variant_text``
    (``preflight/document.py:343-353``) re-runs ``_structural`` per layer and
    says in its own docstring that "the model interior of an unselected
    variant therefore stays open here", which is the same residue one section
    along and is §6's rather than this check's.
    """
    for index, run in enumerate(_runs(document)):
        sites = _a25_sites(run)
        if not sites:
            continue
        where = f"runs[{index}]"
        name = run["name"]
        named = f"runs[{name!r}]"
        refused_counts = False
        for rows_kind, prefix, spec in sites:
            site = where if not prefix else f"{where}.warm_start"
            for key, finding in _a25_bounds(site, name, prefix,
                                            _A25_KNOBS[rows_kind], spec):
                yield finding
                # A24 is COMPUTED from these two, so a value A25 has just
                # refused must not be arithmetic for a second finding.  Only
                # the RUN's own pair counts: the warm start declares neither.
                if not prefix and key in ("n_sweeps", "warmup"):
                    refused_counts = True

            # Only the two ``plan.*`` kinds take it.  ``_NUTS_KEYS``
            # (``nuts.py:103-108``) does not carry the key, so Task 3's
            # ``A1.runs`` already refuses it there by name, and a second
            # answer here would be two refusals in two voices for one typo.
            if (rows_kind.startswith("plan.")
                    and "check_identifiability" in spec):
                mode = spec["check_identifiability"]
                if not _a25_check_mode(mode):
                    yield refuse(
                        "A25", site,
                        f"{named}: {prefix}check_identifiability: is false, "
                        "'once' (before the first sweep) or 'each_sweep' (at "
                        f"every parameter tuple visited); got {mode!r}. There "
                        "is no size heuristic here on purpose: the cost is a "
                        "dense Jacobian and an SVD, so which of the three a "
                        "run wants is a decision the document makes "
                        "(check A25).")

            if rows_kind == "plan.estimate":
                # Gated on ``tol``, and the gate is the package's:
                # ``plan.py:909`` reads ``tol is not None and ...``, because
                # with no convergence test there is no floor for
                # ``min_sweeps`` to raise.  ``:910`` is ``not 1 <=
                # min_sweeps <= max_iter``, so EQUALITY is legal and only
                # ``floor > cap`` is not.
                floor = spec.get("min_sweeps")
                cap = spec.get("max_iter")
                if (spec.get("tol", 1) is not None
                        and _t9_whole_number(floor)
                        and _t9_whole_number(cap)
                        and floor > cap):
                    yield refuse(
                        "A25", site,
                        f"{named}: {prefix}min_sweeps: {floor} is above "
                        f"{prefix}max_iter: {cap}, so the convergence test is "
                        "never consulted -- the run always exhausts max_iter "
                        "and always refuses, including on a model it "
                        "converged on at sweep two. Lower min_sweeps, raise "
                        f"max_iter, or declare {prefix}tol: null to run a "
                        "fixed number of sweeps with no verdict (check A25).")

        if run["kind"] == "plan.sample" and not refused_counts:
            kept = _a24_kept_draws(run)
            if kept is not None:
                # Imported HERE and not at the module head.
                # ``rheplicant/inference/__init__.py`` re-exports the layer
                # eagerly, so a head import puts ``rheplicant.inference`` in
                # ``sys.modules`` on every ``import rheplicant.config`` --
                # which ``test_config_exits_predict.py:1046-1073`` forbids by
                # name, in a fresh interpreter.  ``sections/exits.py`` defers
                # ``from rheplicant.inference import Block`` for the same
                # reason.  Deferred rather than written out because plan §2.5
                # names this one constant: "do not write the literal".
                from rheplicant.inference import MIN_DRAWS

                if kept[0] < MIN_DRAWS:
                    default = ("" if run.get("warmup") is not None
                               else ", the default n_sweeps // 2")
                    yield refuse(
                        "A24", where,
                        f"{named}: this run would keep {kept[0]} draw(s) "
                        f"({run['n_sweeps']} sweeps minus {kept[1]} warmup"
                        f"{default}), and a split-r_hat needs at least "
                        f"{MIN_DRAWS} -- two halves of two. Below that the "
                        "mixing diagnostic is not weak, it is undefined, and "
                        "a run whose only convergence evidence is undefined "
                        "is the silent answer this exit exists to refuse. "
                        "Raise n_sweeps or lower warmup (check A24).")

    inference = document.get("inference")
    npe = inference.get("npe") if isinstance(inference, Mapping) else None
    # The section may sit on a document whose runs do not use it, and a count
    # nothing will read is not a fault.
    if not isinstance(npe, Mapping) or "npe" not in _kinds(document):
        return
    for subsection, key in _A25_NPE_COUNTS:
        body = npe.get(subsection)
        # An ABSENT or malformed subsection is ``npe._subsection``'s
        # (``:209``) and a MISSING count is ``npe._count``'s
        # (``:240-242``), whose sentences say the subsection or the key is
        # required rather than that a number is out of range.
        if not isinstance(body, Mapping) or key not in body:
            continue
        where = f"inference.npe.{subsection}.{key}"
        try:
            _whole(where, body[key], 1)
        except ConfigError as refusal:
            yield refuse("A25", where, f"{refusal} (check A25).")


# --- Task 10: the (kind, noise.kind) table, and the clause npe was denied ---


#: Every ``inference.noise.kind`` ``build_noise`` accepts.  A COPY of
#: ``sections/noise.py``'s ``_KIND_KEYS`` keys (``noise.py:34-41``), and the
#: copy is CHECKED -- ``test_the_noise_kinds_are_the_ones_build_noise_accepts``
#: asserts the two are equal, so a fifth kind added there and forgotten here
#: is a red test rather than a check that silently stands down on every
#: document declaring it.  Bound once, and as a frozenset (plan §3.1): 2D lost
#: a task to one drafter's frozenset meeting another's tuple inside
#: ``set(spec) - allowed``.
#:
#: **Every membership test against it is guarded by ``isinstance(..., str)``
#: first**, and that is not defensive typing.  ``inference.noise.kind:
#: [radiometer]`` is a document a user can write; ``['radiometer'] in
#: frozenset(...)`` raises ``TypeError`` on the unhashable list, and inside
#: the pass a ``TypeError`` becomes "check A27 RAISED" and discards every
#: other finding (§2.3's TRAP).  Task 9 met the same shape on
#: ``check_identifiability`` and answered it by making that container a
#: tuple; §3.1 pins THIS one as a frozenset, so the guard goes on the test.
_NOISE_KINDS: frozenset[str] = frozenset(
    {"none", "homoscedastic", "radiometer", "radiometer_frozen"})

#: ``inference.noise.kind`` -> what ``decided_noise`` (``noise.py:235-247``)
#: hands an exit, decided from the WORD and nothing else.
#:
#: * ``absent``  -- ``kind: none``; ``decided_noise`` returns None and
#:   ``_noise`` (``exit_support.py:205-209``) refuses in its own words.
#:   Neither A27 nor A28 may speak here: A28's sentence would tell a document
#:   that declares no sigma that it "decides its sigma into an array", which
#:   is precisely what ``test_noise_kind_none_keeps_the_shared_refusal``
#:   (``test_config_conjugate_shared.py:483``) exists to prevent.
#: * ``decided`` -- a NoiseModel whose ``std`` ignores its argument's values
#:   by contract, so an exit wanting an array evaluates it and an exit
#:   wanting a rule takes it.  Refused by neither.
#: * ``iterated`` -- a NoiseModel whose sigma depends on the prediction.
#:   Check A27.
#: * ``array``   -- not a model at all.  Check A28.
#:
#: Measured: ``HomoscedasticNoise.depends_on_prediction`` is False and
#: ``RadiometerNoise.depends_on_prediction`` is True, both as CLASS
#: attributes, and ``FlaggedNoise`` forwards its inner model's -- so a
#: ``flags:`` entry never moves a row.
#: ``test_the_shape_table_matches_the_noise_classes`` reads the two class
#: attributes back rather than trusting this comment.
_T10_NOISE_SHAPE: dict[str, str] = {
    "none": "absent",
    "homoscedastic": "decided",
    "radiometer": "iterated",
    "radiometer_frozen": "array",
}

#: The exits that ALWAYS resolve a decided sigma array, so A27 is theirs
#: unconditionally.  This is ``conjugate_support._DECIDES_SIGMA_HERE``'s
#: membership (``conjugate_support.py:62``, read at ``:135``).
#: ``conjugate.gcr`` is deliberately NOT here: it reaches ``_decided_sigma``
#: only under ``noise_from: declared`` (``conjugate.py:380``, ``:426``,
#: ``:435``) and has a third way out that costs it nothing, so it is branched
#: on in the body and hears its own sentence.
_T10_DECIDES_SIGMA: frozenset[str] = frozenset({"conjugate.wiener",
                                                "condition"})

#: The ``runs[].kind``\ s that read ``inference.noise`` as a RULE and
#: therefore reach ``_decided_model``.  **THREE, not two, and the third was
#: measured rather than read**: ``_decided_model`` has two CALL SITES in
#: ``src`` (``conjugate.py``, inside ``_gls_result``; ``npe.py``, inside
#: ``_simulate_bank``), but ``_gls_result`` has two callers of its own --
#: ``_run_gls`` and ``_draw_sigma`` -- and ``_draw_sigma`` is ``kind:
#: conjugate.gcr``'s ``noise_from: gls`` route.  Counting call sites per
#: MODULE gives two and is the count this plan's §6 table carries; counting
#: run KINDS gives three, and the third earned ``conjugate.gls``'s sentence
#: (measured: a gcr run with a frozen sigma was told to *"run kind:
#: conjugate.wiener"* when ``noise_from: declared`` is one key and RUNS).
#: ``test_every_exit_that_reads_the_rule_has_a_branch`` walks that call graph
#: with ``ast`` rather than reading module stems, because the stem form is
#: exactly what hid this.
#:
#: ``conjugate.gcr`` is the one CONDITIONAL member -- it reaches the rule
#: only under ``noise_from: gls``, and under ``noise_from: declared`` it
#: resolves a decided sigma like ``conjugate.wiener`` does -- so the body
#: tests the key as well as the kind.
_T10_ITERATES: frozenset[str] = frozenset({"conjugate.gls", "npe",
                                           "conjugate.gcr"})


@register("A27", "A28")
def _decided(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A27 and A28: the ``(runs[].kind, inference.noise.kind)`` table.

    Two words of text decide both.  Today they are decided at P3, inside
    ``_decided_sigma`` (``exit_support.py:227-270``) and ``_decided_model``
    (``:273-317``), after ``build_resources`` has read and analysed every
    beam -- which is 90.9 % of ``load_document``'s wall time on a toy
    nside-16 beam (§2.7).

    **The runtime refusals stay.**  They are not the same predicate: this one
    reads two strings, theirs reads ``isinstance(noise, NoiseModel)`` and
    ``depends_on_prediction`` off a BUILT object, and only the second one can
    see a fanned ``by_observation`` mapping (``exit_support.py:201-202``) or
    a noise this layer has not finished resolving.  The two agree on every
    document v1 can express, and :data:`_T10_NOISE_SHAPE`'s own test is what
    keeps them agreeing -- so this is the second opinion plan §2.2 sanctions,
    not the copy it forbids.

    Returns findings and raises nothing (§2.3): an unknown ``noise.kind`` is
    left to ``build_noise`` (``noise.py:112-116``), which names the
    vocabulary, rather than becoming a ``KeyError`` out of this table, and
    both membership tests are guarded by ``isinstance(..., str)`` because a
    frozenset raises on an unhashable left operand.

    **``expect: refuse`` does NOT stand this check down**, where it does
    stand :func:`_blocks` and :func:`_prior_gates` down.  Those two have a
    real document each that would otherwise lose the assertion it exists to
    make; measured, none of the five ``expect: refuse`` documents in
    ``tests/config/`` is A27/A28 shaped, and
    ``test_config_section_runs.py:88`` records that a P-1 refusal raising out
    of ``run_document`` rather than being captured is the correct shape.

    **``where`` names the run, not the noise, and that is a decision.**  The
    user may fix either end.  The pass names the run because a run index is
    unambiguous where "which of my four runs made this a problem" is not, and
    every message carries ``inference.noise.kind: <kind>`` verbatim so the
    other end is named too.

    **One ordering this hoist REVERSES, recorded rather than gated.**  The
    ``inference.noise:`` grammar is ``build_noise``'s, at P2, so on a
    document with a readable beam it used to speak before A27 did.  Measured,
    ``kind: radiometer`` with no ``include_logdet`` earns *"is required for a
    prediction-dependent noise model and has no default"* and a stray key
    earns ``check_unknown_keys``' sentence; beside a ``conjugate.wiener`` run
    both now arrive after A27.  Standing down for them would mean
    re-implementing ``_KIND_KEYS``' sweep here, which §2.5 forbids and which
    would be a second validator for a grammar ``build_noise`` owns; hoisting
    that grammar to P-1 as well is the real answer and belongs to whichever
    plan takes A49.  ``test_a_noise_wrong_in_its_GRAMMAR_too_still_hears_
    A27`` is what keeps this paragraph a measurement rather than a claim.
    """
    section = document.get("inference")
    noise = section.get("noise") if isinstance(section, Mapping) else None
    kind = noise.get("kind") if isinstance(noise, Mapping) else None
    if not isinstance(kind, str) or kind not in _NOISE_KINDS:
        return ()
    shape = _T10_NOISE_SHAPE[kind]
    if shape not in ("iterated", "array"):
        return ()

    findings: list[Finding] = []
    for index, entry in enumerate(_runs(document)):
        exit_kind = entry.get("kind")
        if not isinstance(exit_kind, str):
            continue
        # `_runs` filled `name` by `parse_runs`' own rule (`runs.py:115`: an
        # unnamed run is named after its kind), and every executor's message
        # spells the prefix `runs['<that name>']:`.
        # `test_config_exits_gls.py:422` asserts startswith("runs['gls']: "),
        # so the INDEX form would be a red test and a message that does not
        # match the one the user gets from the executor.
        name = entry["name"]
        where = f"runs[{index}].kind"

        if shape == "iterated" and exit_kind in _T10_DECIDES_SIGMA:
            findings.append(refuse("A27", where, (
                f"runs[{name!r}]: kind: {exit_kind} takes a DECIDED sigma "
                f"array, and inference.noise.kind: {kind} makes sigma a "
                "function of the prediction -- which a conjugate solve has "
                "not got, because the prediction is what it solves for "
                "(linear.py:1031). Two routes run this noise: kind: "
                "conjugate.gls iterates the covariance it implies, or "
                "inference.noise.kind: radiometer_frozen decides the sigma "
                "once and keeps this exit (check A27).")))
        elif (shape == "iterated" and exit_kind == "conjugate.gcr"
                and entry.get("noise_from", "declared") != "gls"):
            findings.append(refuse("A27", where, (
                f"runs[{name!r}]: inference.noise.kind: {kind} has a sigma "
                "that depends on the prediction, and a conjugate draw has no "
                "prediction to evaluate it at -- the prediction is what it "
                "draws. Declare noise_from: gls, which runs iterative_gls "
                "first and draws at the covariance it converges to, or "
                "inference.noise.kind: radiometer_frozen, which decides one "
                "sigma array up front (check A27).")))
        elif shape == "array" and exit_kind == "conjugate.gls":
            findings.append(refuse("A28", where, (
                f"runs[{name!r}]: kind: conjugate.gls solves for the "
                "covariance a PREDICTION-DEPENDENT sigma implies, so it "
                "reads inference.noise as a RULE; inference.noise.kind: "
                f"{kind} decides its sigma into an array before any run sees "
                "it, and a decided array is not a rule. Declare "
                "inference.noise.kind: radiometer to iterate the rule, or "
                "run kind: conjugate.wiener, which is what a decided sigma "
                "wants (check A28).")))
        elif (shape == "array" and exit_kind == "conjugate.gcr"
                and entry.get("noise_from", "declared") == "gls"):
            # The third caller of `_decided_model`, reached through
            # `_gls_result` rather than by a call site of its own -- and the
            # one whose fix is a KEY rather than an exit.  Its `instead`
            # clause names A27, because A27's gcr sentence offers
            # `noise_from: gls` and `radiometer_frozen` as alternatives and a
            # user who takes both arrives exactly here.
            findings.append(refuse("A28", where, (
                f"runs[{name!r}]: kind: conjugate.gcr under noise_from: gls "
                "runs iterative_gls first and draws at the covariance it "
                "converges to, so it reads inference.noise as a RULE; "
                f"inference.noise.kind: {kind} decides its sigma into an "
                "array before any run sees it, and a decided array is not a "
                "rule. Drop noise_from: gls: the declared route draws at "
                "that array directly, which is what a frozen sigma is for "
                "-- and noise_from: gls is check A27's answer for "
                "inference.noise.kind: radiometer, so declaring both asks a "
                "reweighting to find a fixed point in a number that is "
                "already fixed (check A28).")))
        elif shape == "array" and exit_kind == "npe":
            findings.append(refuse("A28", where, (
                f"runs[{name!r}]: kind: npe SIMULATES a bank of (theta, "
                "data) pairs and draws the noise for each one, so it reads "
                "inference.noise as a RULE; inference.noise.kind: "
                f"{kind} decides its sigma into an array before any run sees "
                "it, and a decided array is not a rule. Declare "
                "inference.noise.kind: radiometer or homoscedastic -- either "
                "is a rule simulate_pairs can draw from. There is no "
                "amortized-posterior exit that takes a decided array, so the "
                "sigma is what has to change (check A28).")))
    return tuple(findings)
