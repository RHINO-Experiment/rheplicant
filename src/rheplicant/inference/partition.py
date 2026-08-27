"""Deriving the partition, rather than being handed one.

:class:`~rheplicant.inference.plan.SamplingPlan` takes a partition and checks
it. This module produces one, from the model itself::

    plan = SamplingPlan.automatic(space, pipeline, state)

    # the same thing, with the blocks in reach to inspect or amend
    blocks = auto_blocks(space, pipeline, state)
    plan = SamplingPlan(space, *blocks)

The rule is the one a reader would guess — conjugate blocks for the latents
declared ``linear=True``, one gradient block for everything else — plus the
one refinement without which it is wrong on the models this package exists for.

**Why "every linear latent in one block" is wrong.** ``Latent(..., linear=True)``
is a claim about ONE latent: the prediction is affine in it *with the others
held fixed*. A conjugate block over several of them makes a strictly stronger
claim — that the prediction is affine in them JOINTLY — and for a multilinear
model that claim is false while every member's own declaration is true. A gain
and an antenna temperature are each affine given the other; their product is
affine in neither pair member's company. The correct partition puts each factor
of the multilinear form in its own conjugate block, and all of *one* factor's
latents together in that block, which is the arrangement
:func:`~rheplicant.inference.linear.linear_operator` groups for.

Grouping them wrongly is not a silent error — the plan's first sweep hands
every conjugate block to
:func:`~rheplicant.inference.linear.check_linearity`, which probes the joint
map and refuses. But a refusal is not a partition, and producing one is this
module's whole job.

**What is probed, and why PAIRS settle it.** For a group of latents each
already known to be affine on its own, every diagonal block of the group's
Hessian vanishes. Joint affinity is exactly the statement that the whole
Hessian vanishes, so what remains to establish is that every *off-diagonal*
block does — a question about pairs. Probing the C(n, 2) pairs therefore
settles a property of all 2^n - n - 1 subsets, and no subset larger than two
has to be tried. The pairwise verdicts define a graph, "these two may share a
block", and the partition is a colouring of it.

The colouring is first-fit in the space's own declaration order: each latent
joins the first existing group whose every member it is compatible with, and
otherwise opens a new one. Deterministic, and on a multilinear model it
recovers the factors — but first-fit is not guaranteed to find the *fewest*
groups, and a needless extra group costs mixing rather than correctness. A
caller who knows better writes the blocks by hand; that route did not go away.

**The loop itself is bayesmith's** —
:func:`bayesmith.dispatch.factor.first_fit`, the same function its own
graph-native ``factor_partition`` colours with. What stays here is the probing
(this package's probes read a ``Pipeline`` through a ``ParameterSpace``;
bayesmith's read a ``Graph``) — the RULE deciding what the probes' verdicts
mean is one piece of code with one home, so the two packages cannot drift into
partitioning by different logic.

**The cost is probes, and it is the caller's to accept.** One
:func:`~rheplicant.inference.linear.check_linearity` per declared-linear latent,
then at most one per pair — each a linearization plus one forward evaluation
per entry in ``scales``. For the handful of latents a partition is written over
this is cheap; it is quadratic, so for very many it is not. Nothing here
guesses from a size, for the reason
:meth:`~rheplicant.inference.plan.SamplingPlan.estimate`'s
``check_identifiability`` does not either.

No pair is probed twice, and no cache is needed to say so: the groups are
disjoint, so a latent meets any other latent in at most one of them, and the
``all()`` short-circuits the moment a group is ruled out. A memo here would be
a cache that cannot be hit — which reads as an optimization and is one line of
machinery standing in for a fact about the loop.

**What this does NOT decide.** Whether the partition it found is one the model
can support: two conjugate blocks that are coupled is precisely the
configuration whose degenerate case
:mod:`rheplicant.inference.plan` documents as converging quietly onto an
arbitrary point thousands of kelvin from the truth, with every per-block guard
green. The guard that sees it is
:func:`~rheplicant.inference.identifiability.identifiability`, which both plan
exits run by default. Auto-partitioning does not make that check optional, and
this module deliberately does not run it early — a partition is not a reason to
believe a model.
"""

import warnings
from collections.abc import Sequence
from typing import Any

import jax
from bayesmith.dispatch.factor import first_fit

from rheplicant.core.errors import (
    LinearityRefused,
    LogSpaceUnavailable,
    ParameterSpaceError,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.engines import LOG_CONJUGATE
from rheplicant.inference.linear import DEFAULT_SCALES, check_linearity
from rheplicant.inference.loglinear import (
    FIRST_ORDER_MAX_FRACTIONAL,
    LOG_DEFAULT_SCALES,
    check_log_linearity,
    has_log_linear_block,
    log_route_refusal,
)
from rheplicant.inference.parameters import ParameterSpace
from rheplicant.inference.plan import Block


class UncheckedLogRouteWarning(UserWarning):
    """A latent looked log-linear, and no noise model was there to confirm it.

    Whether a log-conjugate block exists is half a question about the
    prediction (is ``log(prediction)`` affine?) and half a question about the
    likelihood (does taking logs SIMPLIFY this noise, and is its fractional
    level inside the first-order ceiling?). :func:`auto_blocks` can answer the
    first from the model alone; the second needs ``noise=``.

    A warning rather than a refusal, because there is nothing wrong with the
    model: the caller simply did not supply what the second half needs, and
    the conservative verdict -- gradient -- is always sound. It is not silent
    for the same reason the partition is conservative: this module's job is to
    produce a partition its own solvers will accept, and before 2026-08-27 it
    could hand out a ``log_conjugate`` block that ``to_log_space`` then
    refused.
    """


def _may_share_a_block(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    first: str,
    second: str,
    *,
    at: dict[str, jax.Array] | None,
    scales: Sequence[float],
    rtol: float | None,
    log: bool,
    noise: Any | None = None,
) -> bool:
    """Whether the prediction — or ``log`` of it — is affine in this PAIR, jointly.

    Only :class:`~rheplicant.core.errors.LinearityRefused` and, in log space,
    :class:`~rheplicant.core.errors.LogSpaceUnavailable` are read as "no". Their
    siblings under ``ParameterSpaceError`` — a latent of integer dtype, a name
    the space does not declare — are conditions under which the question was
    never asked, and swallowing them here would report a broken declaration as
    an interesting multilinear coupling.
    """
    refusals: tuple[type[Exception], ...] = (
        (LinearityRefused, LogSpaceUnavailable) if log else (LinearityRefused,)
    )
    check = check_log_linearity if log else check_linearity
    # `noise=` only for the linear check: `check_log_linearity` asks about
    # `log(prediction)`, whose noise the transform is what REMOVES, so there is
    # no sigma column for it to be judged in.
    extra = {} if log else {"noise": noise}
    try:
        check(
            space,
            pipeline,
            state_template,
            names=(first, second),
            at=at,
            scales=scales,
            rtol=rtol,
            **extra,
        )
    except refusals:
        return False
    return True





def auto_blocks(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    *,
    at: dict[str, jax.Array] | None = None,
    noise: Any | None = None,
    steps: int | None = None,
    learning_rate: float | None = None,
    scales: Sequence[float] = DEFAULT_SCALES,
    log_scales: Sequence[float] = LOG_DEFAULT_SCALES,
    rtol: float | None = None,
) -> tuple[Block, ...]:
    """Derive a partition: closed-form blocks by factor, one gradient block.

    See the module docstring for why latents are grouped by a pairwise probe
    rather than swept into a single block, and for why pairs settle it.

    The blocks come back closed-form first, gradient last. Order is a Gibbs
    sweep's visiting order and does not affect the stationary distribution, but
    it does affect the first sweeps: a gradient block visited first takes its
    NUTS steps against linear latents still sitting at their declared init —
    often a zero sky — and adapts its step size to that conditional, whereas one
    visited last conditions on an exact draw.

    Args:
        space, pipeline, state_template: the model to partition. Unlike
            :class:`~rheplicant.inference.plan.SamplingPlan`, which needs only
            the declaration, this needs the model: which latents may share a
            block, and which have a log-linear one at all, are properties of the
            prediction and of nothing declared.
        at: values for the latents outside each probed pair, as for
            :func:`~rheplicant.inference.linear.check_linearity`. Defaults to
            the declared initial values, which is where the plan's own
            first-sweep check evaluates the same claim.
        noise: the model's noise. Read twice, for two different questions.
            **For the affinity check** it enables the second criterion — the
            departure in units of sigma (D16 axis 3) — which is what tells a
            curvature that is small against the signal from one that is small
            against the noise. Omitted, only the relative criterion applies.
            **A log-conjugate block is a claim about the
            LIKELIHOOD, not only about the prediction**, so whether one exists
            cannot be settled without it: taking logs simplifies a
            multiplicative noise and merely restates an additive one, and the
            first-order equivalence holds only up to
            :data:`~rheplicant.inference.loglinear.FIRST_ORDER_MAX_FRACTIONAL`.
            Given one, those two refusals are applied HERE, and a latent they
            reject is filed to the gradient block — the same verdict, the same
            constant, simply reached before a partition is handed out rather
            than at the first sweep. Omitted, no log-conjugate block is claimed
            at all and :class:`UncheckedLogRouteWarning` names any latent that
            would have qualified on its prediction alone.
        steps, learning_rate: passed to the gradient block. They have no
            meaning without one, so a space with no gradient latents refuses
            them rather than accepting a tuning that reaches nothing.
        scales: probe magnitudes for the LINEAR checks.
        log_scales: probe magnitudes for the LOG-linear checks, which are a
            separate argument rather than the same one. Feeding the linear
            default's ``1e3`` entry to a log probe sends it through an
            exponential that overflows, the check refuses, and a genuinely
            log-linear latent is filed as gradient — a misclassification that
            would cost a conjugate block and report nothing. See
            :data:`~rheplicant.inference.loglinear.LOG_DEFAULT_SCALES`.
        rtol: forwarded to every probe.

    Returns:
        Blocks, closed-form first, ready to splat into a
        :class:`~rheplicant.inference.plan.SamplingPlan`.

    Raises:
        LinearityRefused: if a latent declared ``linear=True`` is not in fact
            affine in itself. Checked before any pair is tried, so that
            "these two may not share a block" always means a coupling between
            two sound declarations and never one broken declaration poisoning
            every pair it appears in.
        ParameterSpaceError: if ``steps`` or ``learning_rate`` is given for a
            gradient block this space has no latents for.
    """
    declared = tuple(name for name in space.names if space.latent(name).linear)
    candidates = tuple(name for name in space.names if not space.latent(name).linear)

    for name in declared:
        # names=(name,), NOT name=name: the plan's first-sweep re-check probes
        # every block through the GROUP path, whose probe points differ from
        # the single-latent path's. Measured: a marginally-curved latent
        # passed discovery through one path and was refused by the plan
        # through the other -- a partition that survives its own re-check is
        # one probed at the same points the re-check will use.
        check_linearity(
            space, pipeline, state_template, names=(name,), at=at,
            scales=scales, rtol=rtol, noise=noise,
        )

    # The noise settles the log question before any probe is worth running:
    # an additive noise, or an f above the first-order ceiling, means there is
    # no log route however affine `log(prediction)` turns out to be.
    refusal = None if noise is None else log_route_refusal(noise)
    log_candidates = (
        ()
        if refusal is not None
        else tuple(
            name
            for name in candidates
            if has_log_linear_block(
                space, pipeline, state_template, names=(name,), at=at,
                scales=log_scales, rtol=rtol,
            )
        )
    )
    if noise is None and log_candidates:
        # Probed, qualified on the prediction, and NOT claimed -- because the
        # half of the question that needs the noise was never asked. Filing
        # them as log blocks anyway is what produced a partition `to_log_space`
        # refuses; saying nothing would make that silent instead of wrong.
        warnings.warn(
            f"auto_blocks() found {list(log_candidates)} log-linear in the "
            "PREDICTION, but was given no noise= model, so it cannot tell "
            "whether a log route exists: log space simplifies a multiplicative "
            "noise and merely restates an additive one, and the first-order "
            "equivalence holds only up to "
            f"f = {FIRST_ORDER_MAX_FRACTIONAL}. They are in the gradient block. "
            "Pass noise= to have them checked and, if they qualify, solved in "
            "closed form.",
            UncheckedLogRouteWarning,
            stacklevel=2,
        )
    log_linear = () if noise is None else log_candidates
    log_set = set(log_linear)
    other = tuple(name for name in candidates if name not in log_set)

    if not other and (steps is not None or learning_rate is not None):
        given = ", ".join(
            f"{key}={value!r}"
            for key, value in (("steps", steps), ("learning_rate", learning_rate))
            if value is not None
        )
        raise ParameterSpaceError(
            f"auto_blocks() was given {given}, which belongs to a gradient block, but "
            f"every latent this space declares ({list(space.names)}) is solved in "
            "closed form — declared linear, or found log-linear by probe — so there is "
            "no gradient block for it to reach. Accepting it would report a sampler "
            "tuned by a number that tuned nothing. Drop the argument, or drop "
            "linear=True from the latent that is not really affine."
        )

    def linear_pair(first: str, second: str) -> bool:
        return _may_share_a_block(
            space, pipeline, state_template, first, second,
            at=at, scales=scales, rtol=rtol, log=False, noise=noise,
        )

    def log_pair(first: str, second: str) -> bool:
        return _may_share_a_block(
            space, pipeline, state_template, first, second,
            at=at, scales=log_scales, rtol=rtol, log=True,
        )

    blocks = [Block(*group) for group in first_fit(declared, linear_pair)]
    blocks += [
        Block(*group, engine=LOG_CONJUGATE) for group in first_fit(log_linear, log_pair)
    ]
    if other:
        blocks.append(Block(*other, steps=steps, learning_rate=learning_rate))
    return tuple(blocks)


__all__ = ["UncheckedLogRouteWarning", "auto_blocks"]
