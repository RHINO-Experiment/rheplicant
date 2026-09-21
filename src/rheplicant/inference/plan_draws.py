"""Driving a plan to a set of draws.

The second half of the same seam. It is far shorter than the estimate
exit and shares almost nothing with it beyond the plan itself, which is
what makes them two exits rather than one with a flag.
"""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.engines import (
    GRADIENT,
    require_priors,
)

from .plan_results import (
    Draws,
    PlanDiagnostics,
)
from .plan_results import PlanResult as PlanResult
from .plan_settings import _DECREMENT_TAG as _DECREMENT_TAG
from .plan_settings import _SETTLED_CHANGES as _SETTLED_CHANGES
from .plan_settings import (
    CHECK_EACH_SWEEP,
    CHECK_ONCE,
    DEFAULT_RHAT_MAX,
    MIN_DRAWS,
    split_rhat,
)
from .plan_settings import EARLIEST_CONVERGED_SWEEP as EARLIEST_CONVERGED_SWEEP
from .plan_settings import OBJECTIVE_FLOOR_EPS as OBJECTIVE_FLOOR_EPS
from .plan_settings import _at_this_size as _at_this_size
from .plan_settings import _Attempt as _Attempt
from .plan_settings import _halves as _halves


def run_sample(
    plan,
    pipeline: AbstractOperator,
    state_template: State,
    observed: jax.Array,
    *,
    noise: Any,
    key: jax.Array,
    n_sweeps: int,
    warmup: int | None = None,
    check_identifiability: Any = CHECK_ONCE,
    rhat_max: float = DEFAULT_RHAT_MAX,
    solve_tol: float = 1e-6,
    solve_guard: float | None = None,
) -> Draws:
    """Posterior draws: a Gibbs sweep over the same partition.

    Each conjugate block is drawn EXACTLY by
    :func:`~rheplicant.inference.linear_solve.gcr_sample`, so a plan of conjugate
    blocks is an exact Gibbs sampler with nothing tuned. A gradient block
    takes ``steps`` NUTS steps instead, which makes the whole scheme
    Metropolis-within-Gibbs — see :class:`~rheplicant.inference.plan_results.Block`'s ``steps`` and
    :func:`~rheplicant.inference.engines.gradient_draw`.

    Args:
        pipeline, state_template, observed, noise: as for
            :meth:`~rheplicant.inference.plan.SamplingPlan.estimate`.
        key: PRNG key. Required — that is the point of this being a separate
            method rather than ``estimate(key=...)``.
        n_sweeps: total sweeps, warmup included.
        warmup: sweeps discarded. Defaults to half of ``n_sweeps``. NUTS
            tuning for gradient blocks adapts through warmup and is **frozen**
            afterwards, because a kernel that keeps adapting from the states
            it visits is no longer a valid transition.
        check_identifiability: as for :meth:`~rheplicant.inference.plan.SamplingPlan.estimate`.
        rhat_max: split-``r_hat`` of the post-warmup joint chi-squared above
            which :attr:`~rheplicant.inference.plan_results.PlanDiagnostics.converged` is ``False``.
            Reported,
            not raised: unlike a point estimate, a chain hands you the
            diagnostic along with the draws, and throwing away expensive
            draws over a scalar summary would be the worse trade.
        solve_tol, solve_guard: as for :meth:`~rheplicant.inference.plan.SamplingPlan.estimate`.

    Returns:
        A :class:`~rheplicant.inference.plan_results.Draws`. **Read ``diagnostics.rhat``.** The
        measured
        difference between a non-identified gain and the same model with an
        identifying tone is 1.824 against 1.002.

    Raises:
        ParameterSpaceError: if the model is not identified; if ``observed``
            is mis-shaped; if ``n_sweeps`` or ``warmup`` is not a sensible
            count; if fewer than :data:`~rheplicant.inference.plan_settings.MIN_DRAWS` draws would
            be kept; or if
            a gradient block has a member with no declared prior.
    """
    if not isinstance(n_sweeps, int) or n_sweeps < 1:
        raise ParameterSpaceError(f"sample() needs n_sweeps >= 1, got {n_sweeps!r}.")
    warmup = n_sweeps // 2 if warmup is None else warmup
    if not isinstance(warmup, int) or warmup < 0:
        raise ParameterSpaceError(
            f"sample() needs warmup >= 0, got {warmup!r}. A negative warmup would "
            "index the chi-squared trace from the end and report r_hat over draws "
            "that were never kept."
        )
    n_draw = n_sweeps - warmup
    if n_draw < MIN_DRAWS:
        raise ParameterSpaceError(
            f"sample() would keep {n_draw} draw(s) ({n_sweeps} sweeps minus "
            f"{warmup} warmup), and a split-r_hat needs at least {MIN_DRAWS} — two "
            "halves of two. Below that the mixing diagnostic is not weak, it is "
            "undefined, and a run whose only convergence evidence is undefined is "
            "exactly the silent answer this plan exists to refuse. Raise n_sweeps or "
            "lower warmup."
        )
    for block, engine in plan._assign:
        if engine == GRADIENT:
            require_priors(plan.space, block.names, block.label)

    cond, values = plan._prepare(
        pipeline,
        state_template,
        observed,
        noise,
        check_identifiability,
        "SamplingPlan.sample",
    )
    report = None
    residuals: dict[tuple[str, ...], float] = {}
    tuning: dict[tuple[str, ...], Any] = {}
    programs: dict[Any, Any] = {}
    chi2: list[float] = []
    kept: dict[str, list[jax.Array]] = {name: [] for name in plan.space.names}
    due, repeat = (
        check_identifiability is not False,
        (check_identifiability == CHECK_EACH_SWEEP),
    )

    for sweep in range(n_sweeps):
        if due:
            report = plan._identifiable(cond, values, "SamplingPlan.sample")
            due = repeat
        values = plan._update(
            cond,
            values,
            draw=True,
            key=jax.random.fold_in(key, sweep),
            adapt=sweep < warmup,
            solve_tol=solve_tol,
            solve_guard=solve_guard,
            tuning=tuning,
            residuals=residuals,
            programs=programs,
        )
        chi2.append(float(cond.chi2(values)))
        if sweep >= warmup:
            for name in plan.space.names:
                kept[name].append(values[name])

    rhat = split_rhat(chi2[warmup:])
    return Draws(
        samples={name: jnp.stack(stack) for name, stack in kept.items()},
        diagnostics=PlanDiagnostics(
            chi2=np.asarray(chi2, dtype=np.float64),
            sweeps=n_sweeps,
            converged=bool(rhat <= rhat_max),
            engines=dict(plan.engines),
            block_residuals=dict(residuals),
            identifiability=report,
            noise_depends_on_prediction=bool(cond.noise.depends_on_prediction),
            warmup=warmup,
            rhat=rhat,
        ),
    )
