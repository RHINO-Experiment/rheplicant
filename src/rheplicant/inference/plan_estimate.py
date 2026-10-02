"""Driving a plan to a point estimate.

The longest thing `SamplingPlan` does, and the first half of the seam the
ruling named: "the point-estimate and draw exits are the natural seam".

These are NOT pure moves -- `self` became `plan` and the methods left
behind forward here -- which is why they are the only two functions in
this split whose source fingerprint changes on purpose. The class keeps
its signatures and its docstrings; only the bodies live here.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from bayesmith.optimize import certify

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.engines import (
    CLOSED_FORM,
    CONJUGATE,
    DEFAULT_GRADIENT_STEPS,
    LOG_CONJUGATE,
    Conditioning,
    _monitor_programs,
    conditional_potential,
    conjugate_draw,
    conjugate_estimate,
    gradient_draw,
    gradient_estimate,
    log_conjugate_draw,
    log_conjugate_estimate,
)

from .plan_results import (
    Estimate,
    PlanDiagnostics,
)
from .plan_results import PlanResult as PlanResult
from .plan_settings import _DECREMENT_TAG as _DECREMENT_TAG
from .plan_settings import (
    _MONITOR_TAG,
    _SOLVE_TOL_STEP,
    CHECK_EACH_SWEEP,
    CHECK_ONCE,
    DEFAULT_CHI2_TOL,
    DEFAULT_GAP_TOL,
    DEFAULT_MAX_ITER,
    MIN_SWEEPS,
    RESOLUTION_EPS,
    _certify,
    _effective_tol,
    _gap_step,
    _GapState,
    _not_converged_message,
    _settled,
    _solve_tol_floor,
)
from .plan_settings import _SETTLED_CHANGES as _SETTLED_CHANGES
from .plan_settings import EARLIEST_CONVERGED_SWEEP as EARLIEST_CONVERGED_SWEEP
from .plan_settings import OBJECTIVE_FLOOR_EPS as OBJECTIVE_FLOOR_EPS
from .plan_settings import _at_this_size as _at_this_size
from .plan_settings import _Attempt as _Attempt
from .plan_settings import _halves as _halves


def advance(
    plan,
    cond: Conditioning,
    values: dict[str, jax.Array],
    *,
    draw: bool,
    key: jax.Array | None,
    adapt: bool,
    solve_tol: float,
    solve_guard: float | None,
    tuning: dict[tuple[str, ...], Any],
    residuals: dict[tuple[str, ...], float],
    programs: dict[Any, Any],
) -> dict[str, jax.Array]:
    """One sweep: every block updated in declaration order, in place of nothing.

    The one place the two exits diverge, and they diverge by which pair of
    engine functions is called. Everything above and below this line —
    conditioning, partitioning, sigma, the joint chi-squared, the
    identifiability check — is shared, which is the claim the module
    docstring makes.

    ``programs`` is the run's compiled-transition cache, created once per
    run and threaded through every sweep. It is an argument rather than
    state on the plan because a plan is an immutable declaration that may be
    run twice against different data; a cache living on it would outlive the
    conditioning it was compiled for.
    """
    for index, (block, engine) in enumerate(plan._assign):
        block_key = None if key is None else jax.random.fold_in(key, index)
        if engine in CLOSED_FORM:
            log = engine == LOG_CONJUGATE
            if draw:
                run = log_conjugate_draw if log else conjugate_draw
            else:
                run = log_conjugate_estimate if log else conjugate_estimate
            extra = {"key": block_key} if draw else {}
            values, recorded = run(
                cond,
                block.names,
                values,
                tol=solve_tol,
                maxiter=None,
                require_convergence=solve_guard,
                programs=programs,
                **extra,
            )
            residuals[block.names] = float(recorded)
        else:
            steps = DEFAULT_GRADIENT_STEPS if block.steps is None else block.steps
            if draw:
                values, tuning[block.names] = gradient_draw(
                    cond,
                    block.names,
                    values,
                    key=block_key,
                    steps=steps,
                    tuning=tuning.get(block.names),
                    adapt=adapt,
                    programs=programs,
                )
                potential = conditional_potential(cond, block.names, values)
                residuals[block.names] = float(
                    potential({name: values[name] for name in block.names})
                )
            else:
                values, potential = gradient_estimate(
                    cond,
                    block.names,
                    values,
                    steps=steps,
                    programs=programs,
                    **(
                        {}
                        if block.learning_rate is None
                        else {"learning_rate": block.learning_rate}
                    ),
                )
                residuals[block.names] = float(potential)
    return values


def run_estimate(
    plan,
    pipeline: AbstractOperator,
    state_template: State,
    observed: jax.Array,
    *,
    noise: Any,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float | None = DEFAULT_CHI2_TOL,
    min_sweeps: int = MIN_SWEEPS,
    check_identifiability: Any = CHECK_ONCE,
    check_linearity: bool = True,
    solve_tol: float = 1e-6,
    solve_guard: float | None = None,
    gap_tol: float = DEFAULT_GAP_TOL,
) -> Estimate:
    """Best fit: block-coordinate descent to a fixed point of the whole model.

    Every block is updated to its conditional best — a Wiener solve for a
    conjugate block; for a gradient one, Adam on the conditional posterior
    followed by Newton steps that remove Adam's step-size floor (see
    :func:`~rheplicant.inference.engines.gradient_estimate`) — and the
    sweep repeats until the point is certified near the minimum of the
    **joint** negative log posterior ``f``, whose minimum is the MAP:

    * **the certificate**: the Newton decrement of ``f`` over every
      latent, ``sqrt(g^T H^-1 g)``, at most ``sqrt(2 gap_tol)`` posterior
      sigma (0.1 by default) once the error its conjugate gradients may
      have left is added (see :data:`~rheplicant.inference.plan_settings.DEFAULT_GAP_TOL` and
      ``_certify``).
      It does not grow with the number of data, and it sees every mode,
      the slow ones included.
    * **the schedule**: the decrement is computed only on a sweep whose
      last two changes of ``f`` are within the effective tolerance
      ``max(tol, OBJECTIVE_FLOOR_EPS * eps)``, relative to ``|f|``, and
      whose decrease passes the gap pre-screen (``_gap_step``) or is
      below what the arithmetic resolves. After a refusal the next
      candidate waits twice as many candidates as the last, so a run
      pays O(log max_iter) decrements.

    The joint chi-squared is recorded but not tested, because with a
    prior it can rise while the run approaches the MAP. A rise of ``f``
    beyond its resolution is never convergence.

    **Inexact inner solves.** A conjugate block solved to ``solve_tol``
    moves the sweep's fixed point off the MAP by an amount that grows as
    the blocks become correlated: measured on the bilinear basis fixture,
    0.11 posterior sigma at ``solve_tol = 1e-6``. When a sweep shows it (a
    rise of ``f`` the arithmetic resolves, or a candidate the decrement
    refuses) the closed-form blocks' tolerance is divided by
    ``1 / _SOLVE_TOL_STEP`` down to a floor set by the dtype, and the
    value the run ended at is recorded as
    :attr:`~rheplicant.inference.plan_results.PlanDiagnostics.solve_tol`. A model that certifies at
    the
    caller's ``solve_tol`` is never tightened.

    **Migration (T-002 reviews).** Until the certificate, the change test
    alone decided, and a tolerance relative to ``|f|`` certifies a
    distance that grows as ``sqrt(N)``: measured, 0.6 posterior sigma at
    ``N = 1e6`` in float64 and 16.5 in float32. ``tol`` keeps its meaning
    and default, so a caller's ``tol`` still does what it did; what is new
    is that it no longer suffices, and runs that used to report converged
    may now run longer or refuse.

    **What the certificate cannot see.** It measures the distance to the
    minimum of a locally quadratic model of ``f`` at the returned point.
    Far from quadratic, where the curvature changes over a posterior
    sigma, the decrement is a local statement. A direction of
    non-positive curvature, an iteration that does not reach its residual
    within its cap (:data:`~bayesmith.optimize.certify.MAXITER`),
    or a residual too large to bound the decrement certifies nothing, and
    the run refuses at ``max_iter`` saying which. In float32 the gradient
    of ``f`` is itself rounded, so a model whose posterior sigma is small
    against the latents' magnitude can stay above the threshold at every
    sweep; the refusal names float64 as the remedy.

    Args:
        pipeline: the forward model.
        state_template: the state it is evaluated on.
        observed: the data. Refused unless shaped exactly like the
            prediction.
        noise: a :class:`~rheplicant.inference.noise.NoiseModel`, or a bare
            sigma (wrapped as
            :class:`~rheplicant.inference.noise.HomoscedasticNoise`).
        max_iter: sweep cap. With a ``tol``, a verdict needs
            :data:`~rheplicant.inference.plan_settings.EARLIEST_CONVERGED_SWEEP` (3) sweeps (see
            ``min_sweeps``), so ``max_iter`` of 1 or 2 can never converge
            and always refuses.
        tol: relative change in the joint negative log posterior below
            which the run has converged, required on two consecutive
            sweep-to-sweep changes — see
            :data:`~rheplicant.inference.plan_settings.DEFAULT_CHI2_TOL`. It is
            floored at :data:`~rheplicant.inference.plan_settings.OBJECTIVE_FLOOR_EPS` machine
            epsilons of the
            objective's dtype, and the value applied is recorded as
            :attr:`~rheplicant.inference.plan_results.PlanDiagnostics.effective_tol`. ``None`` runs
            exactly
            ``max_iter`` sweeps and makes no convergence claim at all — the
            only way to get an answer back without one.
        min_sweeps: sweeps taken before the test is consulted. The test
            compares sweep 2 with sweep 1 and sweep 3 with sweep 2 at the
            earliest (the starting values are not a sweep's output), so
            the earliest verdict is at sweep ``max(min_sweeps,
            EARLIEST_CONVERGED_SWEEP)``: ``min_sweeps`` of 1, 2 and 3
            behave alike.
        check_identifiability: ``"once"``, ``"each_sweep"`` or ``False``. See
            the module docstring; a point estimate is the exit that needs it
            most, because it has no other diagnostic.
        check_linearity: ``True`` checks each closed-form block's claim
            once, before the first sweep:
            :func:`~rheplicant.inference.linear.check_linearity` for a
            conjugate block and
            :func:`~rheplicant.inference.loglinear.check_log_linearity` for a
            log-conjugate one, at their default probe scales. ``False`` skips
            both. The linear check's outermost probe is a thousand prior
            widths out, so a model with a stage that saturates (a converter
            that clips) is refused there even when the data and the posterior
            are far below the limit. A block is solved as the affine map
            tangent to the model at the block's zero, so with the check
            skipped the answer is the model's own only where the model equals
            that map.
        solve_tol: CG tolerance for conjugate blocks, at the start: the run
            tightens it when its solves are inexact (see above).
        solve_guard: bound on each conjugate solve's relative ERROR, as for
            :func:`~rheplicant.inference.linear_solve.wiener_solve`. ``None`` skips
            the condition-number estimate, which is what a 10^6-coefficient
            block wants — see that function's own note on the bargain.
        gap_tol: the certificate's threshold, in nats of the joint
            negative log posterior: the Newton decrement must be at most
            ``2 gap_tol`` — see :data:`~rheplicant.inference.plan_settings.DEFAULT_GAP_TOL`. The
            last
            decrement's upper bound is recorded as
            :attr:`~rheplicant.inference.plan_results.PlanDiagnostics.distance_bound`, in posterior
            sigma.
            Not consulted when ``tol`` is ``None``.

    Returns:
        An :class:`~rheplicant.inference.plan_results.Estimate`.

    Raises:
        ParameterSpaceError: if the model is not identified; if ``observed``
            is mis-shaped; or if the joint negative log posterior has not
            settled within ``max_iter`` sweeps. That last one is an error rather
            than a flag *here* and a flag rather than an error at
            :meth:`~rheplicant.inference.plan.SamplingPlan.sample`, and the asymmetry is deliberate:
            a chain has
            ``r_hat`` to scream with, and a point estimate has nothing.
    """
    if not isinstance(max_iter, int) or max_iter < 1:
        raise ParameterSpaceError(
            f"estimate() needs max_iter >= 1, got {max_iter!r}. Zero sweeps returns "
            "the declared initial values with a converged-looking chi-squared trace "
            "of length one."
        )
    # Gated on `tol`, because with no convergence test there is no floor for
    # `min_sweeps` to raise and a run of two sweeps asking for no verdict is
    # a perfectly ordinary thing to want.
    if tol is not None and (not isinstance(min_sweeps, int) or not 1 <= min_sweeps <= max_iter):
        raise ParameterSpaceError(
            f"estimate() needs 1 <= min_sweeps <= max_iter, got {min_sweeps!r} and "
            f"{max_iter!r}. A min_sweeps above the cap means the test is never "
            "consulted, so the run always exhausts max_iter and always refuses — "
            "including on a model that had already settled."
        )
    cond, values = plan._prepare(
        pipeline,
        state_template,
        observed,
        noise,
        check_identifiability,
        "SamplingPlan.estimate",
        check_linearity,
    )
    report = None
    residuals: dict[tuple[str, ...], float] = {}
    programs: dict[Any, Any] = {}
    monitor = programs.get(_MONITOR_TAG)
    if monitor is None:
        monitor = programs[_MONITOR_TAG] = _monitor_programs(cond, RESOLUTION_EPS)
    measure, change = monitor
    chi2_now, objective_now, terms, scales = measure(values)
    chi2 = [float(chi2_now)]
    objective = [float(objective_now)]
    effective = None if tol is None else _effective_tol(tol, objective_now)
    gap_state, gap, rho, resolution = _GapState(), None, None, None
    measured, last_rise, changed = None, None, False
    attempt, attempts, wait, backoff = None, 0, 0, 1
    tightened = solve_tol
    closed = any(engine in CLOSED_FORM for _, engine in plan._assign)
    # Named apart from the sweep's `floor`, which is the CG TOLERANCE's:
    # the two are both floors and neither is the other's. Asked only where
    # it can matter, since below DENSE_MAX the decrement measures the
    # curvature itself and proving a floor would cost a linearity check
    # this run has no use for.
    curvature = (
        plan._curvature_floor(cond) if certify.real_size(values) > certify.DENSE_MAX else None
    )
    converged = None if tol is None else False
    # "once" is "due now, and never again"; "each_sweep" is "due every time".
    due, repeat = (
        check_identifiability is not False,
        (check_identifiability == CHECK_EACH_SWEEP),
    )

    for sweep in range(1, max_iter + 1):
        if due:
            report = plan._identifiable(cond, values, "SamplingPlan.estimate")
            due = repeat
        values = plan._update(
            cond,
            values,
            draw=False,
            key=None,
            adapt=False,
            solve_tol=tightened,
            solve_guard=solve_guard,
            tuning={},
            residuals=residuals,
            programs=programs,
        )
        chi2_now, objective_now, following, following_scales = measure(values)
        decrease, resolution = change(terms, scales, following, following_scales)
        terms, scales = following, following_scales
        decrease, resolution = float(decrease), float(resolution)
        chi2.append(float(chi2_now))
        objective.append(float(objective_now))
        gap_state, screened, gap, rho = _gap_step(gap_state, decrease, resolution, gap_tol)
        if gap_state.contraction is not None:
            measured = gap_state.contraction
        floor = _solve_tol_floor(objective_now)
        if decrease < -resolution:
            last_rise = (sweep, -decrease)
            # An exact block update cannot raise the objective, so a rise
            # the arithmetic resolves is an inexact inner solve: tighten
            # the closed-form blocks' CG before it keeps the change test
            # from ever settling. Measured on the bilinear basis fixture at
            # noise 0.28, float64: rises of 4e-5 nats at solve_tol = 1e-6
            # held every sweep off the pre-screen for 3000 sweeps.
            if closed:
                tightened = max(tightened * _SOLVE_TOL_STEP, floor)
        changed = effective is not None and _settled(objective[1:], effective)
        # The schedule: the change settled, and the gap pre-screen passed
        # or the decrease sank below what the arithmetic resolves (a rise
        # it resolves is neither). The first decrease is measured from the
        # starting values, which are not a sweep's output, so it seeds the
        # contraction and schedules nothing.
        quiet = sweep >= 2 and (screened or abs(decrease) <= resolution)
        if not (changed and sweep >= min_sweeps and quiet):
            continue
        if wait > 0:
            wait -= 1
            continue
        attempt = _certify(programs, cond, values, gap_tol, sweep, curvature)
        attempts += 1
        if attempt.certified:
            converged = True
            break
        # Not certified. If the plan has closed-form blocks, their CG
        # tolerance is the likeliest reason the sweep's fixed point is off
        # the objective's minimum: tighten it, and back off before asking
        # again so a stalled run pays O(log max_iter) decrements.
        if closed:
            tightened = max(tightened * _SOLVE_TOL_STEP, floor)
        wait, backoff = backoff, 2 * backoff

    if converged is False:
        worst = max(
            (residuals[block.names] for block, engine in plan._assign if engine == CONJUGATE),
            default=None,
        )
        hidden = (
            "Note what this does NOT show up in: every conjugate block's own CG "
            f"residual is {worst:.3g} or better, because a per-block residual is "
            "computed from the block and converges at every sweep of an alternation "
            "that is going nowhere. "
            if worst is not None
            else ""
        )
        raise ParameterSpaceError(
            _not_converged_message(
                max_iter=max_iter,
                tol=tol,
                gap_tol=gap_tol,
                effective=effective,
                changed=changed,
                objective=objective,
                chi2=chi2,
                contraction=measured,
                gap=gap,
                rise=last_rise,
                attempt=attempt,
                solve_tol=tightened,
                dtype=jnp.result_type(objective_now),
                hidden=hidden,
            )
        )

    return Estimate(
        values=values,
        diagnostics=PlanDiagnostics(
            chi2=np.asarray(chi2, dtype=np.float64),
            sweeps=len(chi2) - 1,
            converged=converged,
            engines=dict(plan.engines),
            block_residuals=dict(residuals),
            identifiability=report,
            noise_depends_on_prediction=bool(cond.noise.depends_on_prediction),
            objective=np.asarray(objective, dtype=np.float64),
            effective_tol=effective,
            contraction=rho,
            distance_bound=None if attempt is None else attempt.measured.distance,
            certificate_iterations=(None if attempt is None else attempt.measured.products),
            certificate_attempts=attempts,
            solve_tol=tightened,
            floor_source=None if attempt is None else attempt.measured.floor_source,
        ),
    )
