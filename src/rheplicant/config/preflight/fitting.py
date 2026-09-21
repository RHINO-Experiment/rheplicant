"""The checks that need ``runs[]`` and ``inference:`` at the same time.

No function in this layer sees both today (the plan's finding 2):
``parse_runs`` sees ``runs`` alone and inspects the SIX keys of
``_RUN_KEYS`` (``runs.py::_RUN_KEYS``: ``name``, ``kind``, ``variant``, ``on``,
``reuse``, ``expect``; the plan's "five" was inherited and is wrong, and
``_one`` validates all six) -- every other run key travels untouched in
``RunSpec.options`` (``runs.py::_one``) and is
swept for the first time inside the executor at P3; ``build_inference`` sees
``inference`` alone; ``run_document`` (``runs.py::parse_runs``) holds both and hands
them to ``load_document`` one at a time, calling ``parse_runs`` at
``runs.py::parse_runs`` -- BEFORE the pass, which is why every test of a
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
``test_config_exits_predict.py::TestAVariantMismatchIsRefused.test_a_variant_that_moves_the_layout_is_the_packages_refusal``
runs ``import rheplicant.config;
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

from .fitting_counts import _A25_CHECK_MODES as _A25_CHECK_MODES
from .fitting_counts import _A25_KNOBS as _A25_KNOBS
from .fitting_counts import _A25_NPE_COUNTS as _A25_NPE_COUNTS
from .fitting_counts import _A25_NULLABLE as _A25_NULLABLE
from .fitting_counts import _A25_TOL_GATED as _A25_TOL_GATED
from .fitting_counts import _T9_CHECK_EACH_SWEEP as _T9_CHECK_EACH_SWEEP
from .fitting_counts import _T9_CHECK_ONCE as _T9_CHECK_ONCE
from .fitting_counts import _T9_DEFAULT_MAX_ITER as _T9_DEFAULT_MAX_ITER
from .fitting_counts import _T9_EARLIEST_CONVERGED_SWEEP as _T9_EARLIEST_CONVERGED_SWEEP
from .fitting_counts import _T9_MIN_DRAWS as _T9_MIN_DRAWS
from .fitting_counts import _T9_MIN_SWEEPS as _T9_MIN_SWEEPS
from .fitting_counts import _a24_kept_draws as _a24_kept_draws
from .fitting_counts import _a25_bounded as _a25_bounded
from .fitting_counts import _a25_bounds as _a25_bounds
from .fitting_counts import _a25_check_mode as _a25_check_mode
from .fitting_counts import _a25_pair_message as _a25_pair_message
from .fitting_counts import _a25_sites as _a25_sites
from .fitting_counts import _counts as _counts
from .fitting_counts import _t9_whole_number as _t9_whole_number

# Re-exported so this module's importers keep working. `X as X` is the
# explicit re-export form: a plain import of a name this file does not
# itself use is F401, and `ruff --fix` deletes it whatever the comment
# on the line says.
from .fitting_engines import _A16_FROZEN as _A16_FROZEN
from .fitting_engines import _a16_partition as _a16_partition
from .fitting_engines import _a17_message as _a17_message
from .fitting_engines import _a18_linear as _a18_linear
from .fitting_engines import _blocks as _blocks
from .fitting_engines import _engine_of as _engine_of
from .fitting_engines import _log_route_message as _log_route_message
from .fitting_engines import _log_route_refusal_text as _log_route_refusal_text
from .fitting_engines import _t7_engines as _t7_engines
from .fitting_engines import _t7_sites as _t7_sites
from .fitting_engines import _t7_step_count as _t7_step_count
from .fitting_engines import _t7_warm_start as _t7_warm_start
from .fitting_priors import _a20_joint_over as _a20_joint_over
from .fitting_priors import _a23_latents as _a23_latents
from .fitting_priors import _a23_message as _a23_message
from .fitting_priors import _a23_prior_free as _a23_prior_free
from .fitting_priors import _prior_gates as _prior_gates
from .fitting_priors import _t8_calibrator_keys as _t8_calibrator_keys
from .fitting_priors import _t8_traded_keys as _t8_traded_keys
from .fitting_seeds import _A29_NPE_SUBSECTIONS as _A29_NPE_SUBSECTIONS
from .fitting_seeds import _A29_SEEDED_KINDS as _A29_SEEDED_KINDS
from .fitting_seeds import _seeds as _seeds
from .fitting_vocabulary import _ENGINES as _ENGINES
from .fitting_vocabulary import _LOG_ROUTE_REASONS as _LOG_ROUTE_REASONS
from .fitting_vocabulary import _T7_ADDITIVE_KINDS as _T7_ADDITIVE_KINDS
from .fitting_vocabulary import _T7_BLOCK_KEYS as _T7_BLOCK_KEYS
from .fitting_vocabulary import _T7_CONJUGATE as _T7_CONJUGATE
from .fitting_vocabulary import _T7_GRADIENT as _T7_GRADIENT
from .fitting_vocabulary import _T7_LOG_CONJUGATE as _T7_LOG_CONJUGATE
from .fitting_vocabulary import _T7_NOISE_ADDITIVE as _T7_NOISE_ADDITIVE
from .fitting_vocabulary import _T7_NOISE_NEITHER as _T7_NOISE_NEITHER
from .fitting_vocabulary import _kinds as _kinds
from .fitting_vocabulary import _latents as _latents
from .fitting_vocabulary import (
    _runs,
)
from .fitting_vocabulary import _t7_entries as _t7_entries
from .fitting_vocabulary import _t7_names as _t7_names

__all__: list[str] = []


# --- Task 8: the prior gates, and the seed asymmetry ------------------------


# --- Task 9: the counts a run declares, and the six knobs A25 never named ---


# --- Task 10: the (kind, noise.kind) table, and the clause npe was denied ---


#: Every ``inference.noise.kind`` ``build_noise`` accepts.  A COPY of
#: ``sections/noise.py``'s ``_KIND_KEYS`` keys (``sections/noise.py``), and the
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
    {"none", "homoscedastic", "radiometer", "radiometer_frozen"}
)

#: ``inference.noise.kind`` -> what ``decided_noise`` (``sections/noise.py::build_noise``)
#: hands an exit, decided from the WORD and nothing else.
#:
#: * ``absent``  -- ``kind: none``; ``decided_noise`` returns None and
#:   ``_noise`` (``exit_support.py::parsed_options``) refuses in its own words.
#:   Neither A27 nor A28 may speak here: A28's sentence would tell a document
#:   that declares no sigma that it "decides its sigma into an array", which
#:   is precisely what ``test_noise_kind_none_keeps_the_shared_refusal``
# :
# (``test_config_conjugate_shared.py::test_a_decided_array_is_refused_naming_conjugate_wiener``)
# exists to prevent.
#: * ``decided`` -- a NoiseModel whose ``std`` ignores its argument's values
#:   by contract, so an exit wanting an array evaluates it and an exit
#:   wanting a rule takes it.  Refused by neither.
#: * ``iterated`` -- a NoiseModel whose sigma depends on the prediction.
#:   Check A27.
#: * ``array``   -- not a model at all.  Check A28.
#:
#: Measured: ``HomoscedasticNoise.depends_on_prediction`` is False and
#: ``RadiometerNoise.depends_on_prediction`` is True, both as CLASS
# : attributes (``inference/noise.py::HomoscedasticNoise``,
# ``inference/noise.py::RadiometerNoise``), and ``FlaggedNoise``
#: forwards its inner model's through a ``@property``
# : (``inference/noise.py::FlaggedNoise.depends_on_prediction``) -- so a ``flags:`` entry never
# moves a
#: row.  ``test_the_shape_table_matches_the_noise_classes`` reads all THREE
#: back rather than trusting this comment: the two class attributes
#: directly, and the forwarding by constructing a ``FlaggedNoise`` over each
#: of them, because a claim about a ``@property`` cannot be read off the
#: class the way a ``ClassVar`` can.
#:
#: **What no test pins, and cannot be pinned from inside this module.**  The
#: four messages interpolate ``{kind}`` rather than naming a noise kind, and
#: hard-coding ``radiometer`` in the ``iterated`` sentences or
#: ``radiometer_frozen`` in the ``array`` ones is an EQUIVALENT mutant today
#: -- exactly one kind maps to each shape, so the literal and the
#: interpolation cannot differ.  It stops being equivalent the moment a fifth
#: kind is filed on either row, which is precisely the change
#: ``test_the_noise_kinds_are_the_ones_build_noise_accepts`` exists to make
#: ergonomic, and the message would then name the wrong kind to the user.
#: Recorded here because the only test that could kill it would have to
#: forge a fifth kind that ``_KIND_KEYS`` does not carry, which that same
#: equality test forbids.
_T10_NOISE_SHAPE: dict[str, str] = {
    "none": "absent",
    "homoscedastic": "decided",
    "radiometer": "iterated",
    "radiometer_frozen": "array",
}

#: The exits that ALWAYS resolve a decided sigma array, so A27 is theirs
#: unconditionally.  This is ``conjugate_support._DECIDES_SIGMA_HERE``'s
#: membership (``conjugate_support.py``, read at ``conjugate_support.py::_conjugate_block``).
#: ``conjugate.gcr`` is deliberately NOT here: it reaches ``_decided_sigma``
#: only under ``noise_from: declared`` (``conjugate.py::_wiener_plan``, ``conjugate.py::_gcr_plan``,
#: ``conjugate.py::_gcr_plan``) and has a third way out that costs it nothing, so it is branched
#: on in the body and hears its own sentence.
_T10_DECIDES_SIGMA: frozenset[str] = frozenset({"conjugate.wiener", "condition"})

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
#: resolves a decided sigma like ``conjugate.wiener`` does -- so its leg in
#: the body tests the key as well as the kind.
#:
#: **What this constant does and does not guarantee.**  It GATES A28's three
#: legs, so dropping a member silences that kind behaviourally and a real
#: document says so.  It does NOT select the leg: each one is a literal
#: ``exit_kind == "<kind>"`` inside the gate, because the three sentences
#: differ and that difference is the whole of this task.  So a member ADDED
#: here with no leg written changes nothing at run time; what catches that is
#: ``test_every_exit_that_reads_the_rule_has_a_branch``, which greps this
#: module's source for one such literal per member.  Two guards, and they
#: fail in opposite directions on purpose.
_T10_ITERATES: frozenset[str] = frozenset({"conjugate.gls", "npe", "conjugate.gcr"})


@register("A27", "A28")
def _decided(document: Mapping[str, Any]) -> Iterable[Finding]:
    """A27 and A28: the ``(runs[].kind, inference.noise.kind)`` table.

    Two words of text decide both.  Today they are decided at P3, inside
    ``_decided_sigma`` (``exit_support.py``) and ``_decided_model``
    (``exit_support.py::_adapt_legacy_executor``), after ``build_resources`` has read and analysed
    every
    beam -- which is 90.9 % of ``load_document``'s wall time on a toy
    nside-16 beam (§2.7).

    **The runtime refusals stay.**  They are not the same predicate: this one
    reads two strings, theirs reads ``isinstance(noise, NoiseModel)`` and
    ``depends_on_prediction`` off a BUILT object, and only the second one can
    see a fanned ``by_observation`` mapping (``exit_support.py::parsed_options``) or
    a noise this layer has not finished resolving.  The two agree on every
    document v1 can express, and :data:`_T10_NOISE_SHAPE`'s own test is what
    keeps them agreeing -- so this is the second opinion plan §2.2 sanctions,
    not the copy it forbids.

    Returns findings and raises nothing (§2.3): an unknown ``noise.kind`` is
    left to ``build_noise`` (``sections/noise.py::_a26_sigma_axis_problem``), which names the
    vocabulary, rather than becoming a ``KeyError`` out of this table, and
    both membership tests are guarded by ``isinstance(..., str)`` because a
    frozenset raises on an unhashable left operand.

    **``expect: refuse`` does NOT stand this check down**, where it does
    stand :func:`_blocks` and :func:`_prior_gates` down.  Those two have a
    real document each that would otherwise lose the assertion it exists to
    make; measured, none of the five ``expect: refuse`` documents in
    ``tests/config/`` is A27/A28 shaped, and
    ``test_config_section_runs.py::TestRunDocument`` records that a P-1 refusal raising out
    of ``run_document`` rather than being captured is the correct shape.

    **``where`` names the run, not the noise, and that is a decision.**  The
    user may fix either end.  The pass names the run because a run index is
    unambiguous where "which of my four runs made this a problem" is not, and
    every message carries ``inference.noise.kind: <kind>`` verbatim so the
    other end is named too.

    **One ordering this hoist REVERSES, recorded rather than gated -- and
    Plan 3B took half of it back.**  The ``inference.noise:`` grammar is
    ``build_noise``'s, at P2, so on a document with a readable beam it used to
    speak before A27 did.  Standing down for it would mean re-implementing
    ``_KIND_KEYS``' sweep here, which §2.5 forbids and which would be a second
    validator for a grammar ``build_noise`` owns; hoisting that grammar to
    P-1 as well was named as the real answer, and **Plan 3B's A49 did exactly
    that for the ``include_logdet`` leg** (``preflight/noise.py::_a49_in``).

    So the position today, measured on a ``conjugate.wiener`` run rather than
    inferred:

    * ``kind: radiometer`` with no ``include_logdet`` now earns **A27 and A49
      in one report**, A27 first -- the round trip is gone, and the second
      sentence is *"include_logdet: is required for a prediction-dependent
      noise model and has no default"* in A49's voice at P-1 rather than
      ``build_noise``'s at P2;
    * a **stray key** still earns nothing here.  A49's hoist is gated on
      ``include_logdet`` being one of the unknown keys, deliberately, so that
      a typo'd ``flors:`` is not claimed under A49's id; that sentence still
      arrives from ``check_unknown_keys`` at P2, after A27.

    ``test_a_noise_wrong_in_its_GRAMMAR_too_still_hears_A27`` is what keeps
    this paragraph a measurement rather than a claim -- and note that it calls
    :func:`_t10_decided` directly, so no registration can move it.
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
        # `_runs` filled `name` by `parse_runs`' own rule (`runs.py::_one`: an
        # unnamed run is named after its kind), and every executor's message
        # spells the prefix `runs['<that name>']:`.
        # `test_config_exits_gls.py::TestWhatReachesTheLoop.test_the_solver_knobs_reach_the_loop`
        # asserts startswith("runs['gls']: "),
        # so the INDEX form would be a red test and a message that does not
        # match the one the user gets from the executor.
        name = entry["name"]
        where = f"runs[{index}].kind"

        if shape == "iterated" and exit_kind in _T10_DECIDES_SIGMA:
            findings.append(
                refuse(
                    "A27",
                    where,
                    (
                        f"runs[{name!r}]: kind: {exit_kind} takes a DECIDED sigma "
                        f"array, and inference.noise.kind: {kind} makes sigma a "
                        "function of the prediction -- which a conjugate solve has "
                        "not got, because the prediction is what it solves for "
                        "(linear.py:1031). Two routes run this noise: kind: "
                        "conjugate.gls iterates the covariance it implies, or "
                        "inference.noise.kind: radiometer_frozen decides the sigma "
                        "once and keeps this exit (check A27)."
                    ),
                )
            )
        elif (
            shape == "iterated"
            and exit_kind == "conjugate.gcr"
            and entry.get("noise_from", "declared") != "gls"
        ):
            findings.append(
                refuse(
                    "A27",
                    where,
                    (
                        f"runs[{name!r}]: inference.noise.kind: {kind} has a sigma "
                        "that depends on the prediction, and a conjugate draw has no "
                        "prediction to evaluate it at -- the prediction is what it "
                        "draws. Declare noise_from: gls, which runs iterative_gls "
                        "first and draws at the covariance it converges to, or "
                        "inference.noise.kind: radiometer_frozen, which decides one "
                        "sigma array up front (check A27)."
                    ),
                )
            )
        # A28's three legs sit BEHIND `_T10_ITERATES`, so the constant is read
        # by the body rather than only asserted about.  Before this gate
        # existed `grep -rn _T10_ITERATES src/` returned one line -- its own
        # definition -- and dropping `conjugate.gcr` from it moved no
        # behavioural test at all: a set that documents the body without
        # constraining it is the docstring-nobody-defends shape, in a constant.
        # The per-kind test stays a literal inside, because each leg's SENTENCE
        # is different and that is the whole of this task; a member added to
        # the set with no leg here is caught by
        # `test_every_exit_that_reads_the_rule_has_a_branch`, which greps this
        # source for `exit_kind == "<kind>"` per member.
        elif shape == "array" and exit_kind in _T10_ITERATES:
            if exit_kind == "conjugate.gls":
                # A MOVE, so the words are `be2027b`'s: *"as a model"* and
                # *"has no fixed point to iterate"*, not *"as a RULE"* and
                # *"is not a rule"*. Plan §2.3 designates FOUR corrected
                # messages (A39's) and this is not one of them -- the false
                # A28 sentence §1 licenses is the `npe` leg below, which is
                # the only one whose words this task chose.
                findings.append(
                    refuse(
                        "A28",
                        where,
                        (
                            f"runs[{name!r}]: kind: conjugate.gls solves for the "
                            "covariance a PREDICTION-DEPENDENT sigma implies, so it "
                            "reads inference.noise as a model; inference.noise.kind: "
                            f"{kind} decides its sigma into an array before any run "
                            "sees it, and a decided array has no fixed point to "
                            "iterate. Declare "
                            "inference.noise.kind: radiometer to iterate the rule, or "
                            "run kind: conjugate.wiener, which is what a decided "
                            "sigma wants (check A28)."
                        ),
                    )
                )
            elif exit_kind == "npe":
                findings.append(
                    refuse(
                        "A28",
                        where,
                        (
                            f"runs[{name!r}]: kind: npe SIMULATES a bank of (theta, "
                            "data) pairs and draws the noise for each one, so it "
                            "reads inference.noise as a RULE; inference.noise.kind: "
                            f"{kind} decides its sigma into an array before any run "
                            "sees it, and a decided array is not a rule. Declare "
                            "inference.noise.kind: radiometer or homoscedastic -- "
                            "either is a rule simulate_pairs can draw from. There is "
                            "no amortized-posterior exit that takes a decided array, "
                            "so the sigma is what has to change (check A28)."
                        ),
                    )
                )
            elif exit_kind == "conjugate.gcr" and entry.get("noise_from", "declared") == "gls":
                # The third caller of `_decided_model`, reached through
                # `_gls_result` rather than by a call site of its own -- and
                # the one whose fix is a KEY rather than an exit.  Its
                # `instead` clause names A27, because A27's gcr sentence
                # offers `noise_from: gls` and `radiometer_frozen` as
                # alternatives and a user who takes both arrives exactly here.
                findings.append(
                    refuse(
                        "A28",
                        where,
                        (
                            f"runs[{name!r}]: kind: conjugate.gcr under noise_from: "
                            "gls runs iterative_gls first and draws at the covariance "
                            "it converges to, so it reads inference.noise as a model; "
                            f"inference.noise.kind: {kind} decides its sigma into an "
                            "array before any run sees it, and a decided array has no "
                            "fixed point to iterate. Drop noise_from: gls: the "
                            "declared route draws "
                            "at that array directly, which is what a frozen sigma is "
                            "for -- and noise_from: gls is check A27's answer for "
                            "inference.noise.kind: radiometer, so declaring both asks "
                            "a reweighting to find a fixed point in a number that is "
                            "already fixed (check A28)."
                        ),
                    )
                )
    return tuple(findings)
