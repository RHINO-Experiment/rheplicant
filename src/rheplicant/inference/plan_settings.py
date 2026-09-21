"""The knobs a plan stops on, and the arithmetic of stopping.

Every default and floor a sweep is judged against, plus the convergence
certificate, the gap step and the split-Rhat. These are read by the plan and
by nothing else in this package, and they carry the DOCUMENTED defaults -- a
document that omits a knob gets the value from here.
"""

import dataclasses
import math
from typing import Any

import jax
import numpy as np
from bayesmith.optimize import certify

from rheplicant.core.errors import ParameterSpaceError

#: ``check_identifiability="once"`` — the rank test runs at the starting values.
CHECK_ONCE: str = "once"

#: ``check_identifiability="each_sweep"`` — at every parameter tuple visited.
CHECK_EACH_SWEEP: str = "each_sweep"

#: Sweep cap for :meth:`~rheplicant.inference.plan.SamplingPlan.estimate`.
DEFAULT_MAX_ITER: int = 100

#: Relative CHANGE in the JOINT negative log posterior below which a point
#: estimate has converged, where ``f`` is
#: :meth:`~rheplicant.inference.engines.Conditioning.neg_log_posterior`. The
#: test is ``|f[k] - f[k-1]| <= t * max(|f[k]|, 1)`` on two consecutive
#: sweep-to-sweep changes, with ``t = max(tol, OBJECTIVE_FLOOR_EPS * eps)`` and
#: ``eps`` the machine epsilon of the objective's dtype (see
#: :data:`OBJECTIVE_FLOOR_EPS`). The name predates this rule; until T-002 it was
#: a tolerance on the DECREASE of the joint chi-squared.
#:
#: **Why the objective and not chi-squared.** A block update minimises its
#: conditional of ``f``, so ``f`` is what a sweep descends. Chi-squared is not:
#: with a prior the MAP is not the chi-squared minimum, and a sweep that moves
#: towards the MAP raises chi-squared. The old rule read any such rise as
#: convergence; the T-002 verifier (A5-1) measured runs certified 1 to 15
#: posterior sigma from the exact MAP that way, including a plan of two
#: conjugate blocks with no gradient block.
#:
#: **Why a change in both directions.** For Gaussian noise that does not
#: depend on the prediction every block update is an exact conditional
#: minimisation, so ``f`` cannot rise beyond the arithmetic's noise; a larger
#: rise means a block did not do what its engine says, or the noise is
#: prediction-dependent and the conjugate engine's frozen-sigma solve has a
#: fixed point that is not a minimum of ``f`` (see the module docstring).
#: Neither is convergence.
#:
#: **What this cannot see.** A change is a certificate of distance only when
#: the sweep makes progress. For two blocks with posterior correlation ``r``
#: a sweep shrinks the objective gap by about ``r**4``, so a tolerance on the
#: change bounds the Mahalanobis distance to the MAP by about
#: ``sqrt(2 t |f| / (1 - r**4))``, and ``|f|`` is about ``N / 2`` for ``N``
#: data: the T-002 review measured two conjugate blocks passing this test 0.6
#: posterior sigma from the MAP at ``N = 1e6`` in float64 and 16.5 sigma in
#: float32. So this test certifies nothing on its own: it selects the sweeps
#: at which the Newton decrement (:data:`DEFAULT_GAP_TOL`) is computed, and
#: the decrement decides.
DEFAULT_CHI2_TOL: float = 1e-8

#: A point estimate's certificate, in nats of the joint objective ``f``:
#: ``converged=True`` needs the Newton decrement at the returned point,
#: ``lambda2 = g^T H^-1 g`` over every latent, bounded by ``2 * gap_tol``
#: (T-002 reviews, HIGH).
#:
#: **Nats and sigma.** Near a minimum ``f`` is locally quadratic, and
#: ``sqrt(lambda2)`` is the Mahalanobis distance to it under the posterior
#: precision (the Hessian of ``f``); the gap in nats is ``lambda2 / 2``. So
#: ``0.5 * 0.1**2 = 0.005`` nats is 0.1 posterior sigma, and ``gap_tol = g``
#: certifies ``sqrt(2 g)`` sigma, whatever the number of data and however
#: the blocks are correlated.
#:
#: **What decides and what only schedules.** The decrement is one gradient
#: and a conjugate-gradient solve on Hessian-vector products
#: (one of the two programs
#: ``_monitor_programs`` returns). It runs only
#: on a sweep that passes the cheap tests: the change within ``tol``, and a
#: gap pre-screen (``_gap_step``) that extrapolates the decreases at their
#: estimated contraction, or a decrease below what the arithmetic resolves.
#: Those tests pick the candidate; they cannot certify, because the
#: contraction read from decreases is the fastest mode still moving, and the
#: second review measured a slow mode hidden under a fast one passing a gap
#: test 0.5 to 1.0 sigma off.
DEFAULT_GAP_TOL: float = 0.5 * 0.1**2

#: Sweeps taken before the convergence test is consulted at all. The first
#: steps of a coordinate descent can be nearly stationary without being near the
#: minimum — the same reason
#: :data:`~rheplicant.inference.gls.MIN_REWEIGHTS` exists, at a LOWER count
#: because a sweep here is several block solves rather than one.
#:
#: This read "at a third the count" until 2026-08-28, and the two constants are
#: 3 against 5. Corrected to the relation that holds rather than to a ratio,
#: because the ratio is not the design — what is, is that a sweep costs more
#: than a reweight and so fewer of them are spent before asking.
MIN_SWEEPS: int = 3

#: The floor under a point estimate's relative tolerance, in units of the
#: objective's machine epsilon: the stop rule uses
#: ``max(tol, OBJECTIVE_FLOOR_EPS * eps)``.
#:
#: Without it the default ``tol = 1e-8`` sits below float32's epsilon
#: (1.19e-7), and a float32 run can pass the test only if two consecutive
#: sweeps reproduce the objective to the last bit. Conjugate solves at
#: ``solve_tol = 1e-6`` do not: on the motivating bilinear model
#: (``tests/inference/test_plan.py``, float32, 400 sweeps recorded) the
#: objective at its plateau moves by tens of ulps a sweep, which is the trace
#: :data:`~bayesmith.optimize.certify.OBJECTIVE_FLOOR_EPS` was measured on.
#: In float64 the floor is 1.4e-14 and ``tol`` governs.
OBJECTIVE_FLOOR_EPS: int = certify.OBJECTIVE_FLOOR_EPS

#: How many consecutive sweep-to-sweep changes of the objective the stop rule
#: needs within the effective tolerance. See :func:`_settled`.
_SETTLED_CHANGES: int = certify.SETTLED_CHANGES

#: The first sweep at which :meth:`~rheplicant.inference.plan.SamplingPlan.estimate` can report
#: converged, whatever ``min_sweeps`` says below it. The changes are counted
#: between sweep OUTPUTS, never from the starting values, so
#: ``_SETTLED_CHANGES`` changes need one more sweep than that. A run with
#: a ``tol`` and a ``max_iter`` below this always refuses; the config
#: layer's pre-flight check A25 refuses such a document before it runs.
EARLIEST_CONVERGED_SWEEP: int = _SETTLED_CHANGES + 1

#: The resolution of a sweep-to-sweep change of the objective, in units of
#: machine epsilon: ``RESOLUTION_EPS * eps * sqrt(sum (s0 + s1)**2)`` over the
#: objective's terms, where ``s`` is each term's rounding magnitude (see
#: ``_monitor_programs``). The gap
#: pre-screen treats a change below it as no change; nothing certifies on it.
#:
#: The change is taken as a sum of per-term differences, so the constant parts
#: of ``f`` (every prior's normalizer) and the bulk of the chi-squared sum
#: cancel term by term instead of costing ``eps * |f|``. The multiple is
#: :data:`~bayesmith.optimize.certify.RESOLUTION_EPS` and was measured on
#: this package's collinear templates.
RESOLUTION_EPS: float = certify.RESOLUTION_EPS

#: Split-``r_hat`` above which a run's draws are reported unmixed. 1.05 rather
#: than the modern 1.01 because this is ``r_hat`` of a single scalar summary of
#: a single chain, where 1.01 is noise-dominated at the draw counts a Gibbs
#: sweep over an expensive forward model can afford.
DEFAULT_RHAT_MAX: float = 1.05

#: Fewest post-warmup draws a split-``r_hat`` can be computed from at all: two
#: halves of two. Below this the diagnostic is not weak, it is undefined.
MIN_DRAWS: int = 4

#: Null directions named in a refusal before it says "and N more". Enough to see
#: the pattern, few enough to read.
_DIRECTIONS_SHOWN: int = 4

#: The stop rule's two halves, from
#: :mod:`~bayesmith.optimize.certify`: whether the objective's last
#: ``_SETTLED_CHANGES`` changes are within a tolerance, and that
#: tolerance floored at the dtype's resolution. The changes counted are
#: between sweep OUTPUTS, never from the starting values, so the earliest a
#: trace can settle is :data:`EARLIEST_CONVERGED_SWEEP`.
_settled = certify.settled

_effective_tol = certify.effective_tol

#: The gap PRE-SCREEN, from :mod:`~bayesmith.optimize.certify`: it picks
#: the sweeps at which the Newton decrement is computed and certifies
#: nothing. ``_certify`` is what decides.
_GapState = certify.GapState

_gap_step = certify.gap_step

#: The key of a point estimate's monitor in the run's ``programs`` cache. A
#: 1-tuple of a string cannot equal a gradient block's ``(names, steps,
#: adapting)``, a conjugate block's 6-tuple or an estimate transition's
#: ``("estimate", names, steps, learning_rate)``.
_MONITOR_TAG: tuple[str] = ("monitor",)

#: The key of the Newton decrement's program in the same cache.
_DECREMENT_TAG: tuple[str] = ("decrement",)

#: How a closed-form block's CG tolerance is tightened when the sweep shows
#: its solves are inexact (a rise of the objective the arithmetic resolves,
#: or a candidate stop the decrement refuses), and where that stops, from
#: :mod:`~bayesmith.optimize.certify`. Measured here: the bilinear basis
#: fixture's worst case certifies at 1e-8, six digits above the float64
#: floor. See :meth:`~rheplicant.inference.plan.SamplingPlan.estimate`, ``solve_tol``.
_SOLVE_TOL_STEP = certify.SOLVE_TOL_STEP

_solve_tol_floor = certify.solve_tol_floor


@dataclasses.dataclass(frozen=True)
class _Attempt:
    """One certificate: the sweep it was taken at, and what it measured.

    ``measured`` is a :class:`~bayesmith.optimize.certify.Decrement`, whose
    ``estimate`` and ``distance`` are in posterior sigma because the Hessian
    of the joint negative log posterior IS the posterior precision.
    """

    sweep: int
    measured: Any
    certified: bool


def _certify(
    programs: dict[Any, Any],
    cond: Any,
    values: dict[str, jax.Array],
    gap_tol: float,
    sweep: int,
    floor: float | None = None,
) -> _Attempt:
    """The Newton decrement at ``values``, and whether it certifies ``gap_tol``.

    The objective is :meth:`Conditioning.neg_log_posterior` over every
    latent, so the decrement is the distance to the MAP of the model the
    sweep is descending, in posterior sigma; ``gap_tol`` nats stands for
    ``sqrt(2 gap_tol)`` of them. The verdict reads the upper bound the solve's
    residual and the curvature floor allow, never the estimate alone, so an
    inexact solve can only make it refuse — see
    :func:`~bayesmith.optimize.certify.decrement`.

    ``floor`` is ``_curvature_floor``'s, and matters only
    for a model with more latents than
    :data:`~bayesmith.optimize.certify.DENSE_MAX`, where the Hessian is not
    formed and its smallest eigenvalue has to come from somewhere.

    The program is built once per run and cached in ``programs`` beside the
    block transitions, because compiling it every candidate stop would cost
    more than the solve.
    """
    limit = math.sqrt(2.0 * gap_tol)
    program = programs.get(_DECREMENT_TAG)
    if program is None:
        program = programs[_DECREMENT_TAG] = certify.decrement_program(
            cond.neg_log_posterior, values, floor=floor, limit=limit
        )
    measured = certify.decrement(None, values, program=program)
    return _Attempt(sweep, measured, measured.certifies(limit))


def _at_this_size(measured: Any) -> str:
    """What a decrement with no PROVEN curvature floor adds to the refusal.

    The bound divides by a lower bound on the joint Hessian's smallest
    eigenvalue, and this package certifies only where that number is a proof:
    the formed Hessian's own eigenvalue below
    :data:`~bayesmith.optimize.certify.DENSE_MAX` latents, or the prior
    precision where ``_curvature_floor``'s conditions hold.
    Above that limit and outside those conditions the floor is a Lanczos
    probe's, which is an estimate — measured, it sits ABOVE the smallest
    eigenvalue for about one random spectrum in fifty — so the run refuses
    however small the distance looks, and says what would make it a proof.
    """
    if measured.proven:
        return ""
    return (
        f"Its curvature floor came from {measured.floor_source}, not from a proof, so "
        "nothing here can certify a distance: this model has more than "
        f"{certify.DENSE_MAX} latents, so the Hessian is not formed, and the plan "
        "cannot claim the prior precision as a floor either. Three things give a "
        "proof: ONE conjugate block with a Normal prior on every latent and a sigma "
        "that does not depend on the prediction, which makes the prior precision a "
        f"floor; fewer than {certify.DENSE_MAX} latents, which forms the Hessian and "
        "measures it; or float64 (JAX_ENABLE_X64=1) where the precision is what "
        "blocks the dense path. "
    )


def _not_converged_message(
    *,
    max_iter: int,
    tol: float,
    gap_tol: float,
    effective: float,
    changed: bool,
    objective: list[float],
    chi2: list[float],
    contraction: float | None,
    gap: float | None,
    rise: tuple[int, float] | None,
    attempt: _Attempt | None,
    solve_tol: float,
    dtype: Any,
    hidden: str,
) -> str:
    """:meth:`~rheplicant.inference.plan.SamplingPlan.estimate`'s refusal at ``max_iter``.

    Two shapes. If a sweep passed the pre-screen and the Newton decrement was
    computed, the message is about the decrement: the distance it measured,
    what its conjugate gradients did, and the CG tolerance the closed-form
    blocks ended at. Otherwise the run never settled, and the message names
    the change test, the contraction and the gap it last estimated. Either
    way it names the last rise of the objective beyond its resolution if the
    last ten sweeps had one: an inner solve's noise, which an exact block
    update does not make.

    The headline changes for one case. A model whose decrement has no proven
    curvature floor (above
    :data:`~bayesmith.optimize.certify.DENSE_MAX` latents, outside
    ``_curvature_floor``'s conditions) has not failed to
    converge — nothing here can say whether it has. The refusal says that
    instead, and :func:`_at_this_size` says what would change it.
    """
    limit = math.sqrt(2.0 * gap_tol)
    unsure = attempt is not None and not attempt.measured.proven
    headline = (
        "SamplingPlan.estimate cannot certify this estimate at this size and precision"
        if unsure
        else "SamplingPlan.estimate did not converge"
    )
    opening = (
        f"{headline}: after {max_iter} sweeps the JOINT "
        f"negative log posterior is still changing by "
        f"{objective[-1] - objective[-2]:.3g} per sweep (objective = "
        f"{objective[-1]:.6g}, chi2 = {chi2[-1]:.6g}). "
    )
    if rise is not None and rise[0] > max_iter - 10:
        opening += (
            f"It rose by {rise[1]:.3g} nats at sweep {rise[0]}, beyond its "
            "resolution, which an exact block update cannot do: an inner solve "
            "is inexact at this precision (solve_tol, or the dtype). "
        )
    if attempt is not None:
        measured = attempt.measured
        where = (
            f"between {measured.estimate:.3g} and {measured.distance:.3g}"
            if math.isfinite(measured.distance)
            else f"at least {measured.estimate:.3g} (its residual is too large to "
            "bound it from above)"
        )
        return opening + (
            f"The last candidate stop, sweep {attempt.sweep}, was not certified: the "
            "Newton decrement of the joint objective "
            f"{certify.STATUS_SAID[measured.status]}"
            f" after {measured.products} Hessian-vector product(s) and puts the "
            f"point {where} "
            f"posterior sigma from the objective's minimum, against {limit:.3g} "
            f"(gap_tol = {gap_tol:g} nats). The sweep's fixed point "
            "is not that minimum when an inner solve is inexact (the closed-form "
            f"blocks ended at solve_tol = {solve_tol:g}), when sigma depends on the "
            "prediction and a conjugate block freezes it, when a block is solved in "
            f"log space, or in {np.dtype(dtype).name} when the objective's gradient "
            "is below its rounding. "
            + _at_this_size(measured)
            + "Run in float64 (JAX_ENABLE_X64=1), group the "
            "correlated latents into ONE Block, raise max_iter, or pass tol=None to "
            "accept an unconverged answer."
        )
    said_rho = "not estimable" if contraction is None else f"{contraction:.4g}"
    if changed:
        opening += (
            "The change passed tol, but no sweep after it passed the gap "
            f"pre-screen (gap_tol = {gap_tol:g} nats). "
        )
    else:
        opening += (
            "A verdict needs two consecutive sweep-to-sweep changes each within "
            f"{effective:.3g} of it, relative (tol={tol:g}, floored at "
            f"{OBJECTIVE_FLOOR_EPS} machine epsilons of the objective's dtype), then "
            f"the Newton decrement within {limit:.3g} posterior sigma. "
        )
    left = "" if gap is None else f", leaving a gap of about {gap:.3g} nats"
    rounding = (
        ""
        if np.dtype(dtype).itemsize >= 8
        else f"In {np.dtype(dtype).name} the objective's own rounding can keep it "
        "moving, and a gradient below that rounding cannot be descended at all: "
        "run in float64 (JAX_ENABLE_X64=1). "
    )
    return opening + (
        f"The decrease contracts by {said_rho} per sweep{left}. "
        + hidden
        + rounding
        + "Slow convergence here means the blocks are correlated — group the "
        "correlated latents into ONE Block, which resolves them in a single "
        "solve, or raise max_iter. identifiability(space, pipeline, state, "
        "names=...) reports how much the partition is costing. Pass tol=None to "
        "accept an unconverged answer."
    )


def _halves(values: np.ndarray) -> np.ndarray:
    """The two halves a split-``r_hat`` compares, stacked ``(2, n // 2)``.

    The first ``n // 2`` draws and the LAST ``n // 2``, so an odd-length trace
    drops its middle draw rather than handing one half an extra one.

    Written ``values[values.size - half:]`` and not ``values[-half:]``. Those
    are the same slice for every positive ``half`` and are NOT the same slice
    when ``half`` is 0: ``values[-0:]`` is ``values[0:]``, the whole trace. The
    minimum :data:`~rheplicant.inference.plan_settings.MIN_DRAWS` imposes makes that unreachable
    through
    :func:`split_rhat` today, which is exactly why it is written correctly here
    — a slice that is only right because a caller upstream never passes the
    length that breaks it is a bug waiting for someone to lower a constant. Its
    symptom was a length-1 trace compared against itself and reported by numpy
    as "all input arrays must have the same shape", naming neither this
    function nor the length that caused it.
    """
    half = values.size // 2
    return np.stack([values[:half], values[values.size - half :]])


def split_rhat(trace: Any) -> float:
    """Split-``r_hat`` of a one-dimensional trace.

    The standard single-chain mixing diagnostic: cut the trace in half, treat
    the halves as two chains, and compare the variance between them against the
    variance within them. 1.0 is perfect agreement; anything much above says the
    two halves are describing different distributions, which for a Gibbs run
    means it has not reached stationarity.

    Applied here to the JOINT chi-squared trace, which is the whole point — a
    per-block quantity would be blind to exactly the cross-block degeneracy this
    module exists to catch.

    A trace with no variance within its halves has nothing to mix: identical
    halves are reported as 1.0, and halves that are each constant at *different*
    values as ``inf``, which is the honest reading of a chain that moved once
    and stopped.

    **A trace too short to halve is refused, not answered.**
    ``SamplingPlan.sample`` enforces :data:`~rheplicant.inference.plan_settings.MIN_DRAWS` on the
    draws it keeps
    and this function is public and exported, so it enforces the same minimum
    rather than trusting its one in-package caller. What it refuses used to be
    returned: two halves of one have no within-half variance, so ``ddof=1`` gave
    ``nan`` under a numpy ``RuntimeWarning`` nothing here surfaces. That nan is
    worse than either bare exception below it, because **nan defeats a
    comparison in both directions** — ``rhat <= rhat_max`` is False and
    ``rhat > rhat_max`` is False too, so a threshold guard reads an undefined
    diagnostic as whichever answer the caller happened to test for.

    Args:
        trace: any array-like; flattened first, so its shape does not matter.

    Returns:
        The split-``r_hat``, or ``inf`` for halves each constant at different
        values.

    Raises:
        ParameterSpaceError: if fewer than
            :data:`~rheplicant.inference.plan_settings.MIN_DRAWS` values are given.
    """
    values = np.asarray(trace, dtype=np.float64).ravel()
    if values.size < MIN_DRAWS:
        raise ParameterSpaceError(
            f"split_rhat was given {values.size} value(s) and a split-r_hat needs at "
            f"least {MIN_DRAWS} — two halves of two. Below that the mixing diagnostic "
            "is not weak, it is undefined: halves of one have no variance within them "
            "to divide by, so the answer came back as nan rather than as this refusal "
            "— and a nan passes no threshold test in either direction, which makes an "
            "undefined diagnostic read as whichever verdict the caller tested for. "
            "SamplingPlan.sample refuses the same count on (n_sweeps - warmup); this "
            "is that refusal, for the trace you brought yourself."
        )
    halves = _halves(values)
    half = halves.shape[1]
    within = float(np.mean(np.var(halves, axis=1, ddof=1)))
    between = float(half * np.var(np.mean(halves, axis=1), ddof=1))
    if within <= 0.0:
        return 1.0 if between <= 0.0 else float("inf")
    estimated = (half - 1) / half * within + between / half
    return float(np.sqrt(estimated / within))
