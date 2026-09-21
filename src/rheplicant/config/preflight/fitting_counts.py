"""T9, A24 and A25: the numbers a run is allowed to carry.

Draw counts, sweep counts, tolerances and check modes. All of these are
bounds on integers and floats rather than statements about the model, which is
why they read as one subject despite three check ids.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from types import SimpleNamespace
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.findings import Finding, refuse
from rheplicant.config.preflight import register
from rheplicant.config.sections.exit_support import _number
from rheplicant.config.sections.transforms import _whole

from .fitting_engines import (
    _t7_warm_start,
)
from .fitting_vocabulary import (
    _runs,
)

#: ``CHECK_ONCE`` (``plan.py``) and ``CHECK_EACH_SWEEP`` (``plan.py``),
#: written out for the reason :data:`_ENGINES` is: this module may not import
#: ``rheplicant.inference`` at scope, and a module-level constant cannot defer
#: an import the way a function body can.
#: ``test_check_identifiability_is_a_closed_enum_read_from_the_package``
#: imports both names in the TEST, and
#: ``test_the_package_guard_this_enum_mirrors_is_still_that_guard`` reads
#: ``plan_results.py::Block``'s expression itself, so a third accepted mode turns those
#: red rather than leaving this pass refusing a document the package runs.
_T9_CHECK_ONCE: str = "once"

_T9_CHECK_EACH_SWEEP: str = "each_sweep"

#: ``MIN_DRAWS`` (``plan.py``), ``MIN_SWEEPS`` and ``DEFAULT_MAX_ITER``,
#: written out for the same reason and against plan §2.5's *"do not write the
#: literal"* -- **which is a COST decision, measured, not a style one.**
#:
#: §2.5 has this pass import ``MIN_DRAWS`` from ``rheplicant.inference`` on the
#: grounds that a deferred import costs one ``sys.modules`` lookup.  It does
#: not.  Measured, cold, one ``preflight()`` per fresh process: the worked
#: document is **1.6 ms** and a document whose only difference is a ``kind:
#: plan.sample`` run is **23 ms**, because ``rheplicant/inference/__init__.py``
#: re-exports the layer eagerly and the first call drags **43 modules** in --
#: 37 more than the worked document needs -- against §5's whole budget of
#: 50 ms.  ``rheplicant.inference.plan`` is not cheaper: importing the
#: SUBMODULE runs the package ``__init__`` first, so ``MIN_SWEEPS`` and
#: ``DEFAULT_MAX_ITER`` cost exactly the same 43.  Three integers are not worth
#: a factor of fourteen on the one path §5's number is thinnest on, and a
#: pre-flight pass that pays a tenth of what it saves is not one (§0.1).
#:
#: What replaces the import is the same thing that stands behind
#: :data:`_ENGINES` and the two modes above: a test that imports the REAL name
#: and asserts equality (``test_the_three_plan_counts_are_the_packages_own``).
#: A drift in the package turns that red here rather than leaving this pass
#: refusing a document the package runs.
_T9_MIN_DRAWS: int = 4

_T9_MIN_SWEEPS: int = 3

_T9_DEFAULT_MAX_ITER: int = 100

#: ``EARLIEST_CONVERGED_SWEEP`` from ``inference/plan.py``, the first sweep at
#: which ``SamplingPlan.estimate`` can report converged: its stop rule needs
#: two consecutive sweep-to-sweep changes within tol, and the first change is
#: sweep 2 against sweep 1 (T-002 A5-1). Written out for the same measured
#: reason as the three above, and held to the package's value by the same
#: test, ``test_the_three_plan_counts_are_the_packages_own``;
#: ``test_no_name_from_the_inference_layer_is_imported_at_all`` refuses the
#: deferred import that would otherwise spell it.
_T9_EARLIEST_CONVERGED_SWEEP: int = 3

#: What ``check_identifiability:`` takes, in the package's own order.
#:
#: **A TUPLE and not a frozenset, and neither reason is style.**  ``x in
#: frozenset`` RAISES ``TypeError`` on an unhashable ``x`` --
#: ``check_identifiability: [once]`` is a document a user can write, and
#: inside the pass a ``TypeError`` becomes "check A25 RAISED" and discards
#: every other finding.  And ``0 in frozenset({False, ...})`` is ``True``,
#: because ``hash(0) == hash(False)``, while ``plan_results.py::Block`` tests ``check is
#: not False`` -- an IDENTITY -- and so refuses ``check_identifiability: 0``.
#: A frozenset would therefore accept a document the package refuses AND
#: crash on another.  :func:`_a25_check_mode` mirrors the package's two-part
#: test rather than this tuple's membership; the tuple is what the test pins.
_A25_CHECK_MODES: tuple[Any, ...] = (False, _T9_CHECK_ONCE, _T9_CHECK_EACH_SWEEP)

#: ``(key, kind, minimum)`` per ``runs[].kind``, for every numeric knob that
#: reaches the package.  ``sorted(set(_ESTIMATE_PASSTHROUGH) |
#: set(_SAMPLE_PASSTHROUGH))`` (``exits.py::_parse_optimize``) is eight names --
#: ``check_identifiability``, ``max_iter``, ``min_sweeps``, ``rhat_max``,
#: ``solve_guard``, ``solve_tol``, ``tol``, ``warmup`` -- of which A25's
#: schema row names two, and ``n_sweeps`` is the ninth: it reaches ``_number``
#: at ``exits.py::_SAMPLE_DEFAULTS`` with NO ``minimum=``, so today ``n_sweeps: 0`` is the
#: package's sentence at P3.  ``check_identifiability`` is the one of the
#: eight that is not numeric, and :data:`_A25_CHECK_MODES` has it.
#:
#: The floors are the package's own: ``plan_results.py::Estimate`` (``max_iter >= 1``),
# : ``plan_results.py::Estimate.names`` (``1 <= min_sweeps <= max_iter``),
# ``plan.py::SamplingPlan._partition`` (``n_sweeps >=
# : 1``), ``plan.py::SamplingPlan._partition`` (``warmup >= 0``), ``nuts.py::_parse_nuts`` (both
# nuts counts
#: ``>= 1``).  The three tolerances carry ``0.0`` because nothing in
#: ``config/`` refuses a negative one -- grepped: ``solve_tol`` and
#: ``solve_guard`` appear only in the key sets and the passthrough tuples --
#: so it is forwarded raw into a solver whose bound is the ``eqx.error_if`` at
#: ``linear_solve.py::wiener_solve``, inside jit -- ``engines.py::
#: _monitor_programs`` is the DOCSTRING that describes how that surfaces, not
#: the guard itself.
#: ``rhat_max`` carries ``0.0`` and NOT a strictly-positive floor: see
#: :func:`_counts`' residues.
_A25_KNOBS: dict[str, tuple[tuple[str, type, float | None], ...]] = {
    "plan.estimate": (
        ("max_iter", int, 1),
        ("min_sweeps", int, 1),
        ("tol", float, 0.0),
        ("solve_tol", float, 0.0),
        ("solve_guard", float, 0.0),
    ),
    "plan.sample": (
        ("n_sweeps", int, 1),
        ("warmup", int, 0),
        ("rhat_max", float, 0.0),
        ("solve_tol", float, 0.0),
        ("solve_guard", float, 0.0),
    ),
    "nuts": (("num_samples", int, 1), ("num_warmup", int, 1)),
}

#: The one row of :data:`_A25_KNOBS` the package does not read
#: unconditionally.  ``plan_results.py::Estimate.names`` gates ``min_sweeps`` on ``tol is not
#: None`` and ``plan_results.py::Draws.std`` short-circuits on the same test, so beside ``tol:
#: null`` a ``min_sweeps`` is forwarded, validated by nothing and consulted
#: by nothing: ``min_sweeps: 0`` with ``tol: null`` RUNS.  Checking it anyway
#: would refuse a document the package runs, for a knob that does nothing --
#: the defect Task 5 shipped twice.  The task body's table applied this row
#: unconditionally while its own measurement table said it must not.
_A25_TOL_GATED: frozenset[str] = frozenset({"min_sweeps"})

#: The rows where ``null`` is the package's OWN off-switch rather than a typo,
#: DERIVED rather than judged: they are exactly the ``SamplingPlan.estimate``
#: and ``.sample`` parameters whose annotation admits ``None`` -- ``tol:
#: float | None`` (no convergence test at all, ``plan_results.py::PlanDiagnostics``),
#: ``solve_guard: float | None`` (no condition-number estimate, ``plan_results.py::PlanResult``)
#: and ``warmup: int | None`` (the ``n_sweeps // 2`` default, ``plan.py::SamplingPlan._partition``).
#: ``test_the_nullable_rows_are_the_packages_own_optional_parameters`` reads
#: those signatures back out, so a fourth optional parameter turns it red.
#:
#: On every other row a ``null`` is REFUSED here, and the two that most needed
#: it are the two the package handles worst: ``rhat_max`` is annotated
# : ``float`` and reaches ``plan.py::SamplingPlan._refuse_split_joint_prior``'s ``bool(rhat <=
# rhat_max)`` -- a
#: ``TypeError`` naming no run and no key, raised AFTER the whole chain has
#: been drawn -- and ``solve_tol`` is annotated ``float`` and reaches the CG
#: solver as ``tol=None``.  ``max_iter``, ``min_sweeps``, ``n_sweeps`` and the
#: two ``nuts`` counts the package does refuse in its own voice, so for those
#: five this only moves the phase.
_A25_NULLABLE: frozenset[str] = frozenset({"tol", "solve_guard", "warmup"})

#: The two counts ``inference.npe:`` REQUIRES, and the subsection each lives
#: in.  Two of seven, not two of two: ``_whole`` guards seven whole-number
# : knobs in that section (``sections/npe.py::_positive``, ``sections/npe.py::_subsection``,
# ``sections/npe.py``, ``sections/npe.py::_seeded``), and
#: these are the two ``npe._count`` (``sections/npe.py::_positive``) refuses when absent as well
#: as when out of range.  The other five are optional and are Plan 3B's row.
#:
#: They go through ``transforms._whole`` (``sections/npe.py``), not ``_number``: that is
#: the binding ``npe._count`` already uses, and a second reading here would be
#: the ``_number``-vs-``_whole`` divergence the 2C ledger names.
_A25_NPE_COUNTS: tuple[tuple[str, str], ...] = (("bank", "n_simulations"), ("sample", "n_draws"))


def _t9_whole_number(value: Any) -> bool:
    """Is this an ``int`` the package would treat as a count?

    ``bool`` refused, because ``isinstance(True, int)`` is True.  What that
    buys is NOT that a boolean would otherwise reach the package as the count
    ``1`` -- ``_number`` refuses it first, as *"max_iter: is a number; got
    True"*, measured -- but that a boolean would otherwise reach the
    ARITHMETIC in this module: ``min_sweeps: true`` beside ``max_iter: 0``
    makes ``True > 0`` and yields a spurious second refusal, *"min_sweeps:
    True is above max_iter: 0"*, beside the two ``_number`` already wrote.
    ``test_a_boolean_count_earns_no_SECOND_message_about_itself`` is that
    document.  Bound once and used by :func:`_a24_kept_draws` and by the
    ``min_sweeps``/``max_iter`` pair, which are the two places this module
    does arithmetic on user text.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _a25_check_mode(mode: Any) -> bool:
    """``plan_results.py::Block``'s own test, mirrored and negated -- ``check is not
    False and check not in (CHECK_ONCE, CHECK_EACH_SWEEP)``.

    Identity on ``False`` and ``==`` membership over a TUPLE for the two
    strings, both because the package does it that way and both because the
    obvious spelling is wrong: see :data:`_A25_CHECK_MODES` for the ``0`` and
    the unhashable list a frozenset gets wrong in opposite directions.
    """
    return mode is False or mode in _A25_CHECK_MODES[1:]


def _a25_bounded(
    where: str, name: str, key: str, value: Any, *, kind: type, minimum: float | None
) -> Finding | None:
    """``exit_support._number``, with its refusal turned into a Finding.

    ``_number`` (``exit_support.py::ParsedOptions``) reads exactly one attribute off
    the object it is handed -- ``run.name``, for the ``runs['<name>']:``
    prefix, at ``exit_support.py::RunParseContext``, ``exit_support.py::ParsedRun`` and
    ``exit_support.py::ParsedRun`` -- so a namespace carrying that
    name is the whole adapter.  **``name`` is the BARE run name, never the
    formatted ``runs['fit']``**: ``_number`` BUILDS the prefix itself, so
    handing it the finished string makes every A25 message read
    ``runs["runs['fit']"]:``.  Calling it rather than restating its two type
    refusals is what keeps ONE binding for "is this a whole number": the
    ``_number``-vs-``_whole`` divergence on the 2C ledger is two validators
    for one property disagreeing, and a third written here would be that
    defect with a new name.

    It also fixes a message.  ``plan_results.py::Estimate`` reads ``not isinstance(
    max_iter, int) or max_iter < 1`` and reports only the second half, so
    ``max_iter: 2.5`` reaches the user as *"estimate() needs max_iter >= 1,
    got 2.5"* -- false, since 2.5 IS >= 1, and the real fault is the
    ``isinstance``.  ``_number``'s own wording for that case says what is
    wrong: *"is a whole number; got 2.5. It counts, so 2 and 2.5 are
    different runs"*.

    **The one value ``_number`` cannot be ASKED about, and it aborts the whole
    pass.**  That same whole-number refusal formats ``kind(value)``
    (``exit_support.py::ParsedRun``) to show what the count would round to, and
    ``int(float('inf'))`` raises ``OverflowError`` while ``int(float('nan'))``
    raises ``ValueError``.  Neither is a ``ConfigError``, so neither is caught
    below; inside the pass it becomes "check A24 RAISED" and **every other
    finding in the report is discarded** -- measured, an infinite ``n_sweeps``
    beside two ordinary A16 violations loses both of them, while the same
    document with a finite count reports all three.  It arrives from an
    ordinary document: PyYAML 6.0.3 resolves ``.inf``, ``-.inf`` and ``.nan``,
    and ``json.loads`` turns ``1e400`` into ``float('inf')``.

    So a non-finite value on an ``int`` row is refused HERE, in a sentence
    that opens the way ``_number``'s does and then says the thing ``_number``
    could not.  ``float`` rows need no guard -- ``kind(value)`` is
    ``float(value)`` there and never raises -- and what a ``float`` row does
    with ``inf`` is a residue :func:`_counts` records rather than a bound this
    layer invents.
    """
    if kind is int and isinstance(value, float) and not math.isfinite(value):
        return refuse(
            "A25",
            where,
            f"runs[{name!r}]: {key}: is a whole number; got {value!r}, and "
            f"there is no integer {value!r} rounds to. It reaches this "
            "document from ordinary YAML and JSON -- .inf, -.inf and .nan "
            "are resolved values and 1e400 parses to infinity (check A25).",
        )
    try:
        _number(SimpleNamespace(name=name), key, value, kind=kind, minimum=minimum)
    except ConfigError as refusal:
        return refuse("A25", where, f"{refusal} (check A25).")
    return None


def _a25_bounds(
    where: str,
    name: str,
    prefix: str,
    rows: tuple[tuple[str, type, float | None], ...],
    spec: Mapping[str, Any],
) -> list[tuple[str, Finding]]:
    """Every A25 finding ``rows`` earns on ``spec``, with the key that earned
    it -- the caller needs the key, because A24 is computed from two of them.

    An ABSENT key takes the package's default, which this layer does not
    restate.  A ``null`` is skipped on :data:`_A25_NULLABLE`'s three rows and
    REFUSED everywhere else -- an earlier form skipped every ``null`` and
    justified it with "on the rest it is a typo the package refuses in its own
    voice", which is false of two of the seven: ``rhat_max: null`` is a
    ``TypeError`` from ``plan.py::SamplingPlan._refuse_split_joint_prior`` after the chain has been
    drawn, and
    ``solve_tol: null`` is a ``TypeError`` from inside the solver.  Both are
    annotated ``float`` rather than ``float | None``, which is where
    :data:`_A25_NULLABLE` comes from and why it is derived rather than judged.

    ``live`` is read off ``spec``, never off the run: on a warm start the two
    genuinely differ, because ``tol`` is not a ``_SAMPLE_KEYS`` member at all
    (``exits.py::_parse_optimize``), so a ``plan.sample`` carrying a ``warm_start`` with
    ``tol: null`` has a live run-``tol`` and a dead warm one.
    """
    live = spec.get("tol", 1) is not None
    found: list[tuple[str, Finding]] = []
    for key, kind_of, minimum in rows:
        if key not in spec:
            continue
        if spec[key] is None and key in _A25_NULLABLE:
            continue
        if key in _A25_TOL_GATED and not live:
            continue
        finding = _a25_bounded(
            where, name, f"{prefix}{key}", spec[key], kind=kind_of, minimum=minimum
        )
        if finding is not None:
            found.append((key, finding))
    return found


def _a25_pair_message(
    named: str, prefix: str, spec: Mapping[str, Any], floor: int, cap: int
) -> str:
    """A25's sweep-count refusal, naming only what the document actually wrote.

    Two clauses, one finding. ``1 <= min_sweeps <= max_iter`` is the package's
    own guard; ``max_iter >= EARLIEST_CONVERGED_SWEEP`` is what its stop rule
    needs to be able to pass at all (T-002 A5-1). A document can break
    either or both, and when it breaks both it gets one message that names
    both and one piece of advice that satisfies both: raise ``max_iter`` to
    the larger of the two floors.

    Either half of the pair may be the PACKAGE's default, and a message that
    spelled a default as though the user had typed it is the "hard-coded
    value the user never wrote" shape Task 7 shipped: ``max_iter: 1`` with no
    ``min_sweeps`` would otherwise read *"min_sweeps: 3 is above max_iter: 1"*
    and send the reader looking for a key that is not in their document.  The
    fix clause says "declare a lower min_sweeps" rather than "lower
    min_sweeps" for the same reason -- there may be nothing there to lower.
    The cap in the second clause is always written, since the default is
    far above the earliest verdict.
    """
    floor_said = (
        f"{prefix}min_sweeps: {floor}"
        if "min_sweeps" in spec
        else f"min_sweeps, which defaults to {floor},"
    )
    cap_said = (
        f"{prefix}max_iter: {cap}" if "max_iter" in spec else f"max_iter, which defaults to {cap}"
    )
    earliest = _T9_EARLIEST_CONVERGED_SWEEP
    clauses = []
    if floor > cap:
        clauses.append(
            f"{floor_said} is above {cap_said}, so the convergence test is never consulted"
        )
    if cap < earliest:
        clauses.append(
            f"{cap_said} is below {earliest}, the earliest sweep at which a "
            "verdict can come: the test needs two consecutive sweep-to-sweep "
            "changes within tol, and the first is sweep 2 against sweep 1"
        )
    if cap < earliest:
        advice = f"Raise max_iter to at least {max(floor, earliest)}, or declare {prefix}tol: null"
    else:
        advice = f"Declare a lower min_sweeps, raise max_iter, or declare {prefix}tol: null"
    return (
        f"{named}: " + "; and ".join(clauses) + " -- the run always exhausts "
        "max_iter and always refuses, including on a model that had already "
        f"settled. {advice} to run a fixed number of sweeps with no verdict "
        "(check A25)."
    )


def _a25_sites(run: Mapping[str, Any]) -> tuple[tuple[str, str, Mapping], ...]:
    """Every mapping on this run whose counts reach a ``SamplingPlan``.

    **Two, not one.**  ``exits.py::_ESTIMATE_DEFAULTS`` calls ``SamplingPlan(space,
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
    two-validators shape one function over.  It implements **three of the
    executor's five** gates (``exits.py::_WARM_KEYS``): a mapping (``exits.py::_WARM_KEYS``),
    ``kind: plan.estimate`` (``exits.py::_WARM_KEYS``) and a usable ``move:``
    (``exits.py::_ESTIMATE_PASSTHROUGH``).  It
    does NOT implement the unknown-key sweep (``exits.py::_WARM_KEYS``) or the test that
    ``move`` names a DECLARED latent (``exits.py::_SAMPLE_PASSTHROUGH``), so a warm start carrying
    a typo'd key, or ``move: [ghost]``, still earns A25 here for a mapping the
    executor discards.  That is a residue rather than an oversight: the A25
    sentence is true of the value the user wrote either way, unlike a
    partition answer about a block list nobody reads, and closing it means a
    second reader of ``move`` against ``inference.parameters``.
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

    **``n_sweeps // 2`` is RESTATED from ``plan.py::SamplingPlan._partition``**, because the
    config layer forwards ``warmup`` only when the document declares it
    (``_passthrough``, ``exit_support.py::_legacy_freeze_parse``), so there is nothing to ask.
    A restated default drifts silently: were the package to change it, every
    message this pass writes would still read "N sweeps minus M warmup" with
    the wrong M, and every test here would stay green because they restate it
    too.  ``test_the_restated_default_is_still_the_packages_own`` reads the
    expression back out of the package's source and is what catches that.

    ``plan.py::SamplingPlan._partition`` reads ``warmup is None`` rather than "was warmup
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
    passthrough keys A25 does not name".  Counted from ``exits.py::_parse_optimize``:
    ``sorted(set(_ESTIMATE_PASSTHROUGH) | set(_SAMPLE_PASSTHROUGH))`` is
    eight names, A25's row names ``max_iter`` and ``min_sweeps``, so the six
    are ``check_identifiability``, ``rhat_max``, ``solve_guard``,
    ``solve_tol``, ``tol`` and ``warmup``.  ``n_sweeps`` is the ninth and
    reaches ``_number`` at ``exits.py::_SAMPLE_DEFAULTS`` with no ``minimum=``.

    **A run declaring ``expect: refuse`` is NOT left alone**, and that is the
    one place this check departs from :func:`_blocks` and
    :func:`_prior_gates` deliberately.  Those two stand down because a real
    document in this repository would otherwise lose the assertion it exists
    to make (``test_config_exits_plan.py::TestEstimate.test_noise_kind_none_is_refused``,
    ``posterior_helpers.joint_prior_document``).  No document expects a count
    refusal, and the layer's own policy test says the general shape is the
    other way round: ``test_config_section_runs.py``'s
    ``test_a_text_decidable_refusal_is_not_a_runs_to_expect`` records that a
    P-1 refusal raising out of ``run_document`` rather than being captured is
    the CORRECT shape, because ``expect: refuse`` is for a run that fails
    when it RUNS.

    Residues, all named so that no later task reads this as complete:

    * **A threshold nobody can fail or satisfy stays legal on a ``float``
      row** -- ``rhat_max: 0.0``, and equally ``rhat_max: .inf``,
      ``tol: .inf`` and ``solve_guard: .inf``.  ``rhat_max`` decides a chain's
      convergence verdict and no value of it is refused today; this check
      closes the type and the negative half, but ``0.0`` cannot be closed with
      ``_number``, whose ``minimum=`` is inclusive (``exit_support.py::ParsedRun``:
      ``not value >= minimum``), and neither an exclusive floor nor a ceiling
      written here would be anything but a second validator for a bound
      ``_number`` owns.  A threshold nobody can fail is a warning rather than
      a refusal, and the warning channel's first consumers are Task 12's.
    * **``nuts``'s other three numeric knobs** -- ``num_chains``,
      ``thinning`` and ``target_accept_prob`` -- are outside A25's schema row
      and outside this plan's §1 wording.  They ARE checked, by ``_number``,
      at ``nuts.py::_init_strategy``, which is P3 and behind the beam; hoisting them
      is a widening rather than a hole this check leaves in a rule it states.
    * **A warm start the executor refuses for a key it cannot read** still
      earns A25 for the counts inside it -- :func:`_a25_sites` says which two
      of the executor's five gates are not mirrored, and why.

    The ``null`` residue is CLOSED: :data:`_A25_NULLABLE` is derived from the
    package's own optional parameters and every other ``null`` is refused.

    **Variant layers are not walked**, the same way :func:`_blocks` does not.
    ``load_document`` calls the pass on the variant-APPLIED document
    (``preflight/document.py``), so a SELECTED variant's counts are read here like
    any others.  An unselected one is not: Task 3's ``_variant_text``
    (``preflight/document.py::_variant_text``) re-runs ``_structural`` per layer and
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
            for key, finding in _a25_bounds(site, name, prefix, _A25_KNOBS[rows_kind], spec):
                yield finding
                # A24 is COMPUTED from these two, so a value A25 has just
                # refused must not be arithmetic for a second finding.  Only
                # the RUN's own pair counts: the warm start declares neither.
                if not prefix and key in ("n_sweeps", "warmup"):
                    refused_counts = True

            # Only the two ``plan.*`` kinds take it.  ``_NUTS_KEYS``
            # (``nuts.py``) does not carry the key, so Task 3's
            # ``A1.runs`` already refuses it there by name, and a second
            # answer here would be two refusals in two voices for one typo.
            if rows_kind.startswith("plan.") and "check_identifiability" in spec:
                mode = spec["check_identifiability"]
                if not _a25_check_mode(mode):
                    yield refuse(
                        "A25",
                        site,
                        f"{named}: {prefix}check_identifiability: is false, "
                        "'once' (before the first sweep) or 'each_sweep' (at "
                        f"every parameter tuple visited); got {mode!r}. There "
                        "is no size heuristic here on purpose: the cost is a "
                        "dense Jacobian and an SVD, so which of the three a "
                        "run wants is a decision the document makes "
                        "(check A25).",
                    )

            if rows_kind == "plan.estimate":
                # Gated on ``tol``, and the gate is the package's:
                # ``plan_results.py::Estimate.names`` reads ``tol is not None and ...``, because
                # with no convergence test there is no floor for
                # ``min_sweeps`` to raise.  ``plan_results.py::Estimate.names`` is ``not 1 <=
                # min_sweeps <= max_iter``, so EQUALITY is legal and only
                # ``floor > cap`` is not.  ``spec``, never ``run``: on a warm
                # start the two `tol`s genuinely differ, because `tol` is not
                # a `_SAMPLE_KEYS` member (`exits.py::_parse_optimize`).
                #
                # THE DEFAULTS ARE THE PACKAGE'S.  An earlier form read
                # `spec.get("min_sweeps")` and `spec.get("max_iter")` and so
                # fired only when BOTH keys were written -- which skips
                # `max_iter: 1` and `max_iter: 2`, ordinary documents that
                # `plan_results.py::Estimate.names` refuses against MIN_SWEEPS = 3, and
                # `min_sweeps: 101` against DEFAULT_MAX_ITER = 100.  That is
                # one of the four clauses A25's own schema row names.  They
                # are WRITTEN OUT rather than deferred-imported: measured, a
                # deferred `from rheplicant.inference.plan import ...` drags
                # 43 modules into the first call and takes this pass from
                # 1.6 ms to 23 ms -- see `_T9_MIN_DRAWS`, which carries the
                # measurement and the test that keeps the three honest.
                #
                # The second clause is the stop rule's own floor: with a tol
                # no run converges before `_T9_EARLIEST_CONVERGED_SWEEP`, so
                # a cap below it always exhausts and refuses at P3, whatever
                # `min_sweeps` says. Same gate, same site, ONE finding.
                floor = spec.get("min_sweeps", _T9_MIN_SWEEPS)
                cap = spec.get("max_iter", _T9_DEFAULT_MAX_ITER)
                if (
                    spec.get("tol", 1) is not None
                    and _t9_whole_number(floor)
                    and _t9_whole_number(cap)
                    and (floor > cap or cap < _T9_EARLIEST_CONVERGED_SWEEP)
                ):
                    yield refuse("A25", site, _a25_pair_message(named, prefix, spec, floor, cap))

        if run["kind"] == "plan.sample" and not refused_counts:
            kept = _a24_kept_draws(run)
            if kept is not None:
                # `_T9_MIN_DRAWS`, not `from rheplicant.inference import
                # MIN_DRAWS`.  A head import is forbidden outright --
                # `rheplicant/inference/__init__.py` re-exports the layer
                # eagerly and
                # `
                # test_config_exits_predict.py::test_a_variant_that_moves_the_layout_is_the_packages_refusal`  # noqa: E501
                # refuses
                # it by name in a fresh interpreter -- and the DEFERRED import
                # plan §2.5 called for is what made this the one path §5's
                # 0.05 s could not hold: 43 modules and 21 ms, on the first
                # call, for one integer.  The constant carries the numbers.
                if kept[0] < _T9_MIN_DRAWS:
                    default = "" if run.get("warmup") is not None else ", the default n_sweeps // 2"
                    yield refuse(
                        "A24",
                        where,
                        f"{named}: this run would keep {kept[0]} draw(s) "
                        f"({run['n_sweeps']} sweeps minus {kept[1]} warmup"
                        f"{default}), and a split-r_hat needs at least "
                        f"{_T9_MIN_DRAWS} -- two halves of two. Below that the "
                        "mixing diagnostic is not weak, it is undefined, and "
                        "a run whose only convergence evidence is undefined "
                        "is the silent answer this exit exists to refuse. "
                        "Raise n_sweeps or lower warmup (check A24).",
                    )

    inference = document.get("inference")
    npe = inference.get("npe") if isinstance(inference, Mapping) else None
    # NOT gated on a `kind: npe` run being declared, and the earlier gate that
    # was ("a count nothing will read is not a fault") rested on a claim that
    # is false. `inference.py::build_inference` is `npe = parse_npe(section["npe"],
    # context) if "npe" in section else None` -- unconditional on `runs:` -- so
    # the count IS read, at P2, and `build_inference` (`preflight/document.py`) runs
    # AFTER `build_resources` (`:75`). Measured: `n_simulations: 0` beside
    # `runs: [{kind: forward}]` was silent here and refused by the package, and
    # with UNREADABLE_BEAM in the same document the BEAM spoke first -- which
    # is the one outcome this plan exists to stop.
    if not isinstance(npe, Mapping):
        return
    for subsection, key in _A25_NPE_COUNTS:
        body = npe.get(subsection)
        # An ABSENT or malformed subsection is ``npe._subsection``'s
        # (``preflight/document.py``) and a MISSING count is ``npe._count``'s
        # (``preflight/document.py::_task3_horizon_in``),
        # whose sentences say the subsection or the key is required rather
        # than that a number is out of range.
        if not isinstance(body, Mapping) or key not in body:
            continue
        where = f"inference.npe.{subsection}.{key}"
        try:
            _whole(where, body[key], 1)
        except ConfigError as refusal:
            yield refuse("A25", where, f"{refusal} (check A25).")
