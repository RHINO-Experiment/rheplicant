"""How a plan divides its latents, and what each piece is solved with.

The partition, the refusal for a joint prior split across blocks, the
per-block engine and the curvature floor. These decide the SHAPE of the
work before either exit runs, and both exits ask them the same questions.

Lifted from `SamplingPlan` rather than moved: `self` became `plan` and the
methods forward here, so the class keeps its signatures and docstrings.
"""

import math
from typing import Any

import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.engines import (
    CONJUGATE,
    GRADIENT,
    LOG_CONJUGATE,
    Conditioning,
)
from rheplicant.inference.likelihood import (
    check_observed_shape,
)
from rheplicant.inference.linear import _gaussian_parameters, check_linearity
from rheplicant.inference.loglinear import (
    check_log_linearity,
    to_log_space,
)
from rheplicant.inference.uncertainty import (
    as_noise_model,
)

from .plan_results import (
    Block,
)
from .plan_results import PlanResult as PlanResult
from .plan_settings import _DECREMENT_TAG as _DECREMENT_TAG
from .plan_settings import _SETTLED_CHANGES as _SETTLED_CHANGES
from .plan_settings import (
    CHECK_EACH_SWEEP,
    CHECK_ONCE,
)
from .plan_settings import EARLIEST_CONVERGED_SWEEP as EARLIEST_CONVERGED_SWEEP
from .plan_settings import OBJECTIVE_FLOOR_EPS as OBJECTIVE_FLOOR_EPS
from .plan_settings import _at_this_size as _at_this_size
from .plan_settings import _Attempt as _Attempt
from .plan_settings import _halves as _halves


def partition_of(plan) -> tuple[tuple[Block, str], ...]:
    """Check the partition, then derive each block's engine.

    Order matters: a block naming an undeclared latent cannot have its
    engine derived at all, so the partition is settled first.
    """
    declared = set(plan.space.names)
    unknown = [
        (block, name) for block in plan.blocks for name in block.names if name not in declared
    ]
    if unknown:
        listed = ", ".join(f"{name!r} in Block{b.names}" for b, name in unknown)
        raise ParameterSpaceError(
            f"This plan names {listed}, which the space does not declare; its latents "
            f"are {list(plan.space.names)}. A block over a name nobody declared "
            "updates nothing and leaves the latent it was meant to cover sitting at "
            "its initial value."
        )

    owner: dict[str, Block] = {}
    for block in plan.blocks:
        for name in block.names:
            if name in owner:
                raise ParameterSpaceError(
                    f"Latent {name!r} is in more than one block of this plan "
                    f"(Block{owner[name].names} and Block{block.names}). A Gibbs sweep "
                    "updates each block against the conditional that holds when it "
                    "runs, so the second update would be solving a conditional the "
                    "first one just invalidated — and every diagnostic would report "
                    "the second's answer as if the first had never happened. Put each "
                    "latent in exactly one block; to update two together, put them in "
                    "ONE block: Block('gain', 't_ant')."
                )
            owner[name] = block

    missing = [name for name in plan.space.names if name not in owner]
    if missing:
        raise ParameterSpaceError(
            f"This plan does not cover latent(s) {missing}: every latent of the space "
            "must be in exactly one block. An omitted latent is silently frozen at its "
            "declared init for the whole run — the sweep converges, the joint "
            "chi-squared settles, and nothing anywhere reports that a parameter you "
            "declared was never inferred. Add it to a block, or drop it from the space."
        )
    plan._refuse_split_joint_prior(owner)
    return tuple((block, plan._engine_of(block)) for block in plan.blocks)


def refuse_split_joint_prior(plan, owner: dict[str, "Block"]) -> None:
    """Refuse any plan over a space carrying a joint prior.

    A plan does not read one. ``engines._log_prior`` builds each block's
    conditional from ``Latent.prior`` alone, and a latent covered by a joint
    prior declares ``prior=None`` — so the density contributes exactly
    nothing. Measured, on a two-latent block the partition would otherwise
    accept: the conditional potential is IDENTICAL with the prior declared
    and without it, at every point, while ``0.5 logdet I`` ranges over 1.20
    nats across the same points. The sweep then runs and reports a converged
    chi-squared.

    This refusal used to fire only on a partition that SPLIT the block, and
    its own advice — "put the whole block in ONE Block" — led straight into
    the silent case. The refusal is therefore unconditional, and it names
    the exit that does evaluate the prior.

    A :class:`~rheplicant.inference.priors.JeffreysPrior` is ONE density
    over its whole ``over=`` block — ``sqrt(det I)`` of the joint
    information matrix, which is not the product of the sub-blocks'
    determinants and does not factorise into a term per latent. So a sweep
    whose blocks split it has no conditional to give either block: whatever
    each one steps against, the two are not conditionals of a common joint
    density, and the sweep has no invariant distribution to converge to. It
    would nevertheless run, settle, and report a converged chi-squared,
    because every per-block number is computed from the block.

    Called after the cover check, so every name in ``over`` is known to
    have exactly one owning block.
    """
    joint = plan.space.joint_prior
    if joint is None:
        return
    placed = ", ".join(f"{name!r} in Block{owner[name].names}" for name in joint.over)
    split = len({owner[name].names for name in joint.over}) > 1
    why = (
        "and this partition splits it across blocks, so neither block would "
        "even be stepping a conditional of a common density"
        if split
        else "and no block would step it at all"
    )
    raise ParameterSpaceError(
        f"This space declares {type(joint).__name__}(over={list(joint.over)}) "
        f"({placed}), {why}. SamplingPlan does not evaluate a joint prior: each "
        "block's conditional is built from Latent.prior, and a covered latent "
        "declares none, so the density would contribute exactly zero — measured, "
        "the conditional potential is identical with the declaration and without "
        "it, while 0.5 logdet I ranges over 1.20 nats across the same points. The "
        "sweep would run, settle, and report a converged chi-squared computed "
        "entirely from blocks that never saw the prior. to_numpyro_model is the "
        "exit that evaluates it; use that, or drop the joint prior from the space."
    )


def engine_of(plan, block: Block) -> str:
    """Derive the block's engine from the declaration, or honour the override."""
    linear = [name for name in block.names if plan.space.latent(name).linear]
    other = [name for name in block.names if not plan.space.latent(name).linear]

    if block.engine == LOG_CONJUGATE:
        if linear:
            raise ParameterSpaceError(
                f"Block{block.names} asks for engine='log_conjugate', but {linear} "
                "are declared linear=True — the prediction is affine in them, and "
                "then log(prediction) is not. The two claims exclude each other, "
                "so one of them is wrong: drop linear=True if the latent really "
                "enters through an exponential, or drop engine='log_conjugate' and "
                "let the conjugate engine be derived."
            )
        if block.steps is not None:
            raise ParameterSpaceError(
                f"Block{block.names} is solved in closed form in log space, which "
                f"has no inner steps, so steps={block.steps} would be silently "
                "ignored. Drop steps=, or say engine='gradient' if a gradient step "
                "was what you meant."
            )
        return LOG_CONJUGATE

    if block.engine is None:
        if other and linear:
            raise ParameterSpaceError(
                f"Block{block.names} mixes declared-linear latents {linear} with "
                f"non-linear ones {other}, so which engine it takes cannot be derived. "
                "A conjugate solve needs the whole block affine; a gradient step does "
                "not exploit the linear members' structure at all, which for a "
                "high-dimensional linear block is the difference between tractable and "
                "hopeless. Split them into separate blocks, or say "
                "engine='gradient' to step the whole block by gradient deliberately."
            )
        engine = CONJUGATE if linear else GRADIENT
    else:
        engine = block.engine
        if engine == CONJUGATE and other:
            raise ParameterSpaceError(
                f"Block{block.names} asks for engine='conjugate', but {other} are not "
                "declared linear=True. The conjugate machinery solves "
                "(A^T N^-1 A + S^-1)x = b, which is the posterior only if the "
                "prediction really is affine in the block — and that claim belongs in "
                "the Latent declaration, where check_linearity verifies it, not in a "
                "plan that asserts it. Declare linear=True and the claim will be "
                "checked; leave it undeclared and this block is stepped by gradient."
            )

    if engine == CONJUGATE and block.steps is not None:
        raise ParameterSpaceError(
            f"Block{block.names} is solved by the conjugate engine, which has no inner "
            f"steps, so steps={block.steps} would be silently ignored. A conjugate "
            "block's estimate is one Wiener solve and its draw is one exact "
            "constrained realization — there is no step count to tune, which is the "
            "whole advantage. Drop steps=, or say engine='gradient' if a gradient step "
            "was what you meant."
        )
    return engine


def curvature_floor(plan, cond: Conditioning) -> float | None:
    """A VERIFIED lower bound on the joint Hessian's smallest eigenvalue, or
    ``None`` when this plan's model gives none.

    For a prediction the model is affine in JOINTLY, under a noise that
    does not depend on it, the joint objective is
    ``0.5 |N^-1/2 (d - J x)|^2 + 0.5 (x - m)^T P (x - m)`` plus constants,
    so ``H = J^T N^-1 J + P`` and ``H >= P >= min(1 / sigma_prior^2)``:
    the prior precision is a floor, whatever the data. Each condition is
    checked rather than assumed:

    * **jointly affine.** Every latent declared ``linear=True``, so the
      claim exists to be checked at all, and then: one conjugate block is
      enough on its own, since :meth:`_prepare` has already verified that
      block's joint linearity. With more than one block, each was
      verified alone, and two conditionally affine blocks are not jointly
      affine — the bilinear ``gain * sky`` is the standing example — so
      the same check is asked of the union, and a refusal means no
      floor.
    * **a sigma that does not depend on the prediction**, so the
      log-determinant is a constant and not curvature.
    * **a Normal prior on every latent**, since a prior that is not
      Gaussian has no constant curvature to floor with. The floor is the
      smallest precision, which is the WIDEST scale.

    It is consulted only above
    :data:`~bayesmith.optimize.certify.DENSE_MAX` latents, where the
    Hessian is not formed. Returning ``None`` is not a failure: it means
    the decrement falls back to a probe, which never certifies, and the
    run says so (:func:`_at_this_size`).
    """
    if bool(cond.noise.depends_on_prediction):
        return None
    floor = math.inf
    for name in plan.space.names:
        latent = plan.space.latent(name)
        if not latent.linear:
            return None
        gaussian = _gaussian_parameters(latent.prior)
        if gaussian is None:
            return None
        widest = float(jnp.max(jnp.abs(jnp.asarray(gaussian[1]))))
        if not math.isfinite(widest) or widest <= 0.0:
            return None
        floor = min(floor, 1.0 / widest**2)
    if not math.isfinite(floor):
        return None
    single = len(plan._assign) == 1 and plan._assign[0][1] == CONJUGATE
    if not single and not plan._jointly_affine(cond):
        return None
    return floor


def prepare_conditioning(
    plan,
    pipeline: AbstractOperator,
    state_template: State,
    observed: jax.Array,
    noise: Any,
    check: Any,
    exit_name: str,
) -> tuple[Conditioning, dict[str, jax.Array]]:
    """Everything both exits do before their first sweep.

    Builds the forward function ONCE, refuses a mis-shaped ``observed``,
    and checks each conjugate block's linearity claim once — the bargain
    :func:`~rheplicant.inference.linear.gcr_sample` recommends for a sweep,
    which is what lets every rebuild inside the loop pass ``check=False``.
    """
    if check is not False and check not in (CHECK_ONCE, CHECK_EACH_SWEEP):
        raise ParameterSpaceError(
            f"{exit_name} was given check_identifiability={check!r}; it takes "
            f"{CHECK_ONCE!r} (before the first sweep), {CHECK_EACH_SWEEP!r} (at every "
            "parameter tuple visited, which is cheap for a small model and strictly "
            "more informative, since identifiability is a local property of a "
            "nonlinear model), or False. There is no size heuristic here on purpose: "
            "the cost is a dense Jacobian and SVD, n_data x n_par float64 words, and "
            "which side of that trade you are on is yours to say."
        )
    forward, values0 = plan.space.forward_fn(pipeline, state_template)
    check_observed_shape(
        jnp.shape(jax.eval_shape(forward, values0)),
        observed,
        predictor="this plan's model",
    )
    normalized = as_noise_model(noise)
    log_observed, log_sigma = None, None
    if any(engine == LOG_CONJUGATE for _, engine in plan._assign):
        log_observed, log_sigma = to_log_space(observed, normalized)

    cond = Conditioning(
        space=plan.space,
        pipeline=pipeline,
        state_template=state_template,
        observed=observed,
        noise=normalized,
        forward=forward,
        log_observed=log_observed,
        log_sigma=log_sigma,
    )
    for block, engine in plan._assign:
        if engine == CONJUGATE:
            check_linearity(plan.space, pipeline, state_template, names=block.names, at=values0)
        elif engine == LOG_CONJUGATE:
            check_log_linearity(plan.space, pipeline, state_template, names=block.names, at=values0)
    return cond, values0
