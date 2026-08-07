"""Sampling plans: one declared partition, two exits — a point estimate and a draw.

Everything under :mod:`rheplicant.inference` up to here builds *one* block's
answer. :func:`~rheplicant.inference.linear.wiener_solve` is a linear-Gaussian
block's posterior mean and :func:`~rheplicant.inference.linear.gcr_sample` is an
exact draw from the same conditional, sharing one private solve that differs by
a single argument. This module promotes that: a
:class:`SamplingPlan` says how the whole space is partitioned into blocks, and
then

.. code-block:: python

    plan = SamplingPlan(
        space,
        Block("t_nw", "t_ant"),        # one conjugate solve over both
        Block("gain"),                 # another
        Block("beam_fwhm", steps=20),  # not linear -> gradient engine
    )

    est   = plan.estimate(twin, state, observed, noise=noise)
    draws = plan.sample(twin, state, observed, noise=noise, key=k, n_sweeps=200)

Two methods, not a mode flag. ``key=None | k`` is the right *implementation* and
the wrong *interface*: a caller's intent is "give me the best fit" or "give me
draws", not "here is a PRNG key". Making them two methods also makes the invalid
combinations unrepresentable rather than validated — ``key`` is required on
:meth:`~SamplingPlan.sample` and absent from :meth:`~SamplingPlan.estimate`, so
"asked for samples and forgot the key" cannot be written down; ``n_sweeps`` and
``warmup`` belong to one and ``max_iter`` and ``tol`` to the other because they
mean nothing to the other. And the layer below already names the two exits
differently, so two methods continue an idiom rather than adding a third.

**The engine is derived, never restated.** ``Latent(..., linear=True)`` already
says which exit a latent takes, so ``Block("t_nw", "t_ant")`` needs no
``engine=``: a block whose members are all declared linear is solved by the
conjugate machinery, anything else is stepped by gradient. An explicit
``engine=`` is an override for the one case that is genuinely ambiguous — a
block mixing declared-linear and non-linear latents, which is an error unless
the caller says to downgrade the whole block to gradient.

**The partition is checked, and the check is the point.** Every latent of the
space in exactly one block: a latent the plan forgets would sit at its initial
value while every other number in the run looked healthy, and a latent in two
blocks would be updated twice per sweep against a conditional that no longer
holds. Both are refused by name.

**What this exists to prevent.** A hand-rolled alternating solve over a bilinear
``gain x T_ant`` model, with a free antenna temperature per (time, frequency)
cell, lands thousands of kelvin from the truth while every guard this package
ships reports green — CG residual ~1e-7, per-block condition number ~1.47,
``check_linearity`` passing at every sweep because each conditional genuinely
*is* affine. Nothing in the sweep is wrong. The **partition** is, and no
per-block number is entitled to notice: a residual and a condition number are
both computed from the block being solved.

"Thousands" rather than a number, because the distance is the *initial offset*
carried along the null direction, not a property of the model: 27 K from a
1 %-off start, 2962 K from a 100 %-off start, and the guards read alike in
both (``tests/inference/test_degenerate_partition.py``). Iterating does not
help either — the answer at five sweeps and at two hundred agrees to four
figures, because the solve reaches the solution manifold at once and then has
nowhere left to move.

Two things here can notice, and both are on by default.
:func:`~rheplicant.inference.identifiability.identifiability` sees across
blocks and refuses the model before a sweep runs, naming the degenerate
directions by latent; and the convergence monitor is the **joint** chi-squared
at the current parameter tuple across sweeps, never a per-block residual — which
is precisely the number that read ~1e-7 on an answer thousands of kelvin wrong.

**The identifiability check costs a dense Jacobian and a dense SVD**, ``n_data x
n_par`` float64 words, so ``check_identifiability=`` is the caller's explicit
choice and not a size heuristic. ``"once"`` (the default) checks before the
first sweep; ``"each_sweep"`` checks at every parameter tuple the run visits,
which for a small model is cheap and strictly more informative — a nonlinear
model's identifiability is a property of *where you are*, so a check only at the
start misses a degeneracy that opens up near the parameters you actually reach.
``False`` skips it, which is also how a complex latent (which the rank test
cannot analyse) or a 10^6-coefficient sky block (which it cannot afford) gets
through. Switching between these by size would be the "guess instead of refuse"
this package rejects.

**Both exits check it, and the point estimate is the more dangerous one.** The
instinct is that only sampling needs an identifiability guard. It is backwards:
a chain at least has ``r_hat`` to scream with — 1.824 for a non-identified gain
against 1.002 with an identifying tone — while a point estimate has no
diagnostic at all and CG converges quietly onto an arbitrary point of the null
space.

**Exactness, stated rather than hidden.** A linear-Gaussian block's GCR draw is
an exact conditional draw, so Gibbs over conjugate blocks is an exact sampler.
The moment one block takes a finite number of NUTS steps the scheme becomes
Metropolis-within-Gibbs: still valid, still targeting the right stationary
distribution, and no longer exact — the inner step count now affects mixing.
``Block(..., steps=20)`` looks like a performance knob and is a statistical
assumption. See :func:`~rheplicant.inference.engines.gradient_draw`.

**Relationship to iterative_gls.**
:func:`~rheplicant.inference.gls.iterative_gls` is a fixed-point loop over a
prediction-dependent noise model for ONE block; a plan is a loop over blocks.
They are not nested here, and deliberately: a plan re-evaluates sigma at the
current joint prediction before every block update, so for a
:class:`~rheplicant.inference.noise.RadiometerNoise` the sweep IS the
reweighting iteration. Nesting ``iterative_gls`` inside a block would run the
same fixed point twice, one inside the other, at the product of their costs.
The consequence to know is statistical rather than numerical: freezing sigma
inside a draw makes it an exact draw from a linear-Gaussian conditional *at that
covariance*, which is not the full model's conditional when sigma depends on the
prediction — the same GLS-versus-full-likelihood difference
:mod:`rheplicant.inference.noise` gives in closed form.
:attr:`PlanDiagnostics.noise_depends_on_prediction` records whether that
applied.
"""

import dataclasses
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.engines import (
    CONJUGATE,
    DEFAULT_GRADIENT_STEPS,
    ENGINES,
    GRADIENT,
    Conditioning,
    conditional_potential,
    conjugate_draw,
    conjugate_estimate,
    gradient_draw,
    gradient_estimate,
    require_priors,
)
from rheplicant.inference.identifiability import (
    IdentifiabilityReport,
    identifiability,
)
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.linear import check_linearity
from rheplicant.inference.parameters import ParameterSpace
from rheplicant.inference.uncertainty import as_noise_model

#: ``check_identifiability="once"`` — the rank test runs at the starting values.
CHECK_ONCE: str = "once"

#: ``check_identifiability="each_sweep"`` — at every parameter tuple visited.
CHECK_EACH_SWEEP: str = "each_sweep"

#: Sweep cap for :meth:`SamplingPlan.estimate`.
DEFAULT_MAX_ITER: int = 100

#: Relative PROGRESS in the JOINT chi-squared below which a point estimate has
#: converged — ``chi2[k-1] - chi2[k]``, relative to ``max(|chi2|, 1)``.
#:
#: A decrease rather than a change, and that is the whole design of the test.
#: Block-coordinate descent cannot increase the objective, so once a sweep stops
#: reducing it there is nothing left to reach. Testing ``|chi2[k] - chi2[k-1]|``
#: instead walks into the trap
#: :func:`~rheplicant.inference.gls.iterative_gls` documents for its own
#: ``reweight_tol``: consecutive sweeps differ by roughly the inner solver's own
#: noise whatever the outer iteration is doing, so a threshold below that floor
#: measures CG and never passes. Measured on the motivating model in float32:
#: the plateau sits at chi2 = 2.7e-3 and jitters by 1.2e-3 a sweep, so a
#: converged run would have been refused for 300 sweeps and counting.
DEFAULT_CHI2_TOL: float = 1e-8

#: Sweeps taken before the convergence test is consulted at all. The first
#: steps of a coordinate descent can be nearly stationary without being near the
#: minimum — the same reason
#: :data:`~rheplicant.inference.gls.MIN_REWEIGHTS` exists, at a third the count
#: because a sweep here is several block solves rather than one.
MIN_SWEEPS: int = 3

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


def _halves(values: np.ndarray) -> np.ndarray:
    """The two halves a split-``r_hat`` compares, stacked ``(2, n // 2)``.

    The first ``n // 2`` draws and the LAST ``n // 2``, so an odd-length trace
    drops its middle draw rather than handing one half an extra one.

    Written ``values[values.size - half:]`` and not ``values[-half:]``. Those
    are the same slice for every positive ``half`` and are NOT the same slice
    when ``half`` is 0: ``values[-0:]`` is ``values[0:]``, the whole trace. The
    minimum :data:`MIN_DRAWS` imposes makes that unreachable through
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
    :meth:`SamplingPlan.sample` enforces :data:`MIN_DRAWS` on the draws it keeps
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
        ParameterSpaceError: if fewer than :data:`MIN_DRAWS` values are given.
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


@dataclasses.dataclass(frozen=True, init=False)
class Block:
    """One group of latents, updated together, by one engine.

    ``Block("t_nw", "t_ant")`` puts two latents in one conjugate solve;
    ``Block("gain")`` is a block of one. Which engine a block takes is
    **derived** from ``Latent(..., linear=True)`` and is not restated here —
    see :class:`SamplingPlan` for the derivation and for when ``engine=`` is a
    legitimate override.

    Attributes:
        names: the latents in this block, in the caller's own order.
        steps: inner steps for a **gradient** block — Adam steps at
            :meth:`SamplingPlan.estimate`, NUTS steps at
            :meth:`SamplingPlan.sample`. ``None`` takes
            :data:`~rheplicant.inference.engines.DEFAULT_GRADIENT_STEPS`.

            This reads as a performance knob and it is a **statistical
            assumption**: a conjugate block's draw is an exact conditional draw,
            so a plan of conjugate blocks is an exact Gibbs sampler, while a
            finite number of NUTS steps is a transition that merely *leaves* the
            conditional invariant. The scheme is then Metropolis-within-Gibbs —
            valid, and with mixing that depends on this number. Giving it to a
            conjugate block is an error rather than an ignored argument, because
            a conjugate solve has no inner steps for it to mean.
        engine: ``"conjugate"``, ``"gradient"``, or ``None`` to derive. An
            override, not the norm.
    """

    names: tuple[str, ...]
    steps: int | None
    engine: str | None

    def __init__(
        self, *names: str, steps: int | None = None, engine: str | None = None
    ) -> None:
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "engine", engine)
        self._check()

    def _check(self) -> None:
        if not self.names:
            raise ParameterSpaceError(
                "Block() needs at least one latent name. An empty block is updated every "
                "sweep and changes nothing, so a plan holding one runs, converges, and "
                "reports a partition that does not cover the space it claims to."
            )
        wrong = [name for name in self.names if not isinstance(name, str)]
        if wrong:
            raise ParameterSpaceError(
                f"Block() takes latent NAMES, got {wrong}. Blocks are declared over the "
                "names a ParameterSpace uses, not over Latent objects or values — "
                "Block('gain'), not Block(space.latent('gain'))."
            )
        repeated = sorted({name for name in self.names if self.names.count(name) > 1})
        if repeated:
            raise ParameterSpaceError(
                f"Block{self.names} lists {repeated} more than once. Two copies of one "
                "latent in a block are exactly degenerate with each other, so the block's "
                "normal operator is singular in a direction that says nothing about the "
                "model — and the {name: array} answer has one entry per name, so one "
                "copy's result would silently overwrite the other's."
            )
        if self.engine is not None and self.engine not in ENGINES:
            raise ParameterSpaceError(
                f"Block{self.names} asks for engine={self.engine!r}; the engines are "
                f"{list(ENGINES)}. Leave engine=None and it is derived from "
                "Latent(..., linear=True), which is the normal case — an explicit engine "
                "is an override."
            )
        if self.steps is not None and (
            not isinstance(self.steps, int) or isinstance(self.steps, bool) or self.steps < 1
        ):
            raise ParameterSpaceError(
                f"Block{self.names} asks for steps={self.steps!r}; inner steps must be a "
                "positive int. steps=0 would leave the block at its current value every "
                "sweep, which is a latent excluded from the inference while the partition "
                "check still reports it covered."
            )

    @property
    def label(self) -> str:
        """How a message names this block."""
        return "(" + ", ".join(repr(name) for name in self.names) + ")"


@dataclasses.dataclass(frozen=True)
class PlanDiagnostics:
    """What a run measured, shared by both exits.

    Attributes:
        chi2: the JOINT chi-squared, one entry per sweep — the monitored
            quantity. For :meth:`SamplingPlan.estimate` it decreases towards a
            fixed point; for :meth:`SamplingPlan.sample` it fluctuates around a
            stationary value, which is what :attr:`rhat` tests.
        sweeps: sweeps actually run.
        converged: for a point estimate, whether the joint chi-squared settled
            within ``tol`` (``None`` when the test was disabled). For a draw,
            whether :attr:`rhat` came in under the caller's threshold. **False
            here means the answer is not what it looks like** — the same reading
            as :attr:`~rheplicant.inference.gls.GLSResult.converged`.
        engines: which engine each block took, keyed by the block's ``names``.
        block_residuals: each conjugate block's last relative CG residual, and
            each gradient block's last conditional potential. Recorded because
            it is worth having and **not** because it is a verdict: these are
            the numbers that read ~1e-7 on an answer thousands of kelvin
            wrong. Read :attr:`chi2`.
        identifiability: the last rank report taken, or ``None`` when the check
            was disabled.
        noise_depends_on_prediction: whether sigma was re-evaluated at the
            prediction each block update. When ``True`` a draw is exact for the
            linear-Gaussian conditional at the frozen covariance, which is not
            the full model's conditional — see the module docstring.
        warmup: sweeps discarded before collecting draws (``None`` for a point
            estimate).
        rhat: split-``r_hat`` of the post-warmup joint chi-squared (``None`` for
            a point estimate).
    """

    chi2: np.ndarray
    sweeps: int
    converged: bool | None
    engines: dict[tuple[str, ...], str]
    block_residuals: dict[tuple[str, ...], float]
    identifiability: IdentifiabilityReport | None
    noise_depends_on_prediction: bool
    warmup: int | None = None
    rhat: float | None = None


class PlanResult(Protocol):
    """What both exits guarantee: diagnostics, and answers keyed by latent name.

    One result currency in two shapes rather than four unrelated ones. Whatever
    a plan returns, ``result.diagnostics`` is a :class:`PlanDiagnostics` and
    ``result.names`` are the latents — so a caller can log, compare or assert on
    a run without knowing which exit produced it. What differs is what the
    answer *is*, which is the honest difference: a point estimate has values, a
    sampling run has draws.
    """

    diagnostics: PlanDiagnostics

    @property
    def names(self) -> tuple[str, ...]: ...


@dataclasses.dataclass(frozen=True)
class Estimate:
    """A point estimate: one value per latent, plus what the run measured.

    Attributes:
        values: ``{name: array}``, in the space's declaration order. Keyed by
            name so the physical names survive the solve — a caller never slices
            an anonymous stacked vector and never has to get an offset right.
        diagnostics: see :class:`PlanDiagnostics`.
    """

    values: dict[str, jax.Array]
    diagnostics: PlanDiagnostics

    @property
    def names(self) -> tuple[str, ...]:
        """The latents, in the order the plan's space declares them."""
        return tuple(self.values)


@dataclasses.dataclass(frozen=True)
class Draws:
    """Posterior draws: a stack per latent, plus what the run measured.

    Attributes:
        samples: ``{name: (n_draw, *latent.shape)}``, warmup already discarded.
        diagnostics: see :class:`PlanDiagnostics`. Read
            :attr:`~PlanDiagnostics.rhat` before believing :attr:`mean`.
    """

    samples: dict[str, jax.Array]
    diagnostics: PlanDiagnostics

    @property
    def names(self) -> tuple[str, ...]:
        """The latents, in the order the plan's space declares them."""
        return tuple(self.samples)

    @property
    def n_draw(self) -> int:
        """How many draws were kept."""
        return int(jnp.shape(next(iter(self.samples.values())))[0])

    @property
    def mean(self) -> dict[str, jax.Array]:
        """Posterior mean per latent — comparable with :attr:`Estimate.values`."""
        return {name: jnp.mean(stack, axis=0) for name, stack in self.samples.items()}

    @property
    def std(self) -> dict[str, jax.Array]:
        """Posterior standard deviation per latent."""
        return {name: jnp.std(stack, axis=0) for name, stack in self.samples.items()}


class SamplingPlan:
    """A partition of a parameter space into blocks, with two exits.

    See the module docstring for the design and for the measured failure this
    exists to prevent.

    Args:
        space: the parameter declaration this plan partitions.
        *blocks: the :class:`Block` s, in the order a sweep visits them.

    Raises:
        ParameterSpaceError: if no blocks are given; if a block names something
            the space does not declare; if a latent appears in more than one
            block; if a latent appears in none; if a block mixes declared-linear
            and non-linear latents without an explicit ``engine=``; if
            ``engine="conjugate"`` is asked for a block with a non-linear
            member; or if ``steps=`` is given to a conjugate block.
    """

    def __init__(self, space: ParameterSpace, *blocks: Block) -> None:
        if not blocks:
            raise ParameterSpaceError(
                "A SamplingPlan needs at least one Block. With none, every sweep updates "
                "nothing, the joint chi-squared never moves, and the run converges "
                f"immediately at the declared initial values — for a space of "
                f"{list(space.names)}, an answer that is entirely the starting guess."
            )
        self.space = space
        self.blocks = tuple(blocks)
        self._assign = self._partition()
        self.engines = {block.names: engine for block, engine in self._assign}

    # ------------------------------------------------------------ declaring --

    def _partition(self) -> tuple[tuple[Block, str], ...]:
        """Check the partition, then derive each block's engine.

        Order matters: a block naming an undeclared latent cannot have its
        engine derived at all, so the partition is settled first.
        """
        declared = set(self.space.names)
        unknown = [
            (block, name)
            for block in self.blocks
            for name in block.names
            if name not in declared
        ]
        if unknown:
            listed = ", ".join(f"{name!r} in Block{b.names}" for b, name in unknown)
            raise ParameterSpaceError(
                f"This plan names {listed}, which the space does not declare; its latents "
                f"are {list(self.space.names)}. A block over a name nobody declared "
                "updates nothing and leaves the latent it was meant to cover sitting at "
                "its initial value."
            )

        owner: dict[str, Block] = {}
        for block in self.blocks:
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

        missing = [name for name in self.space.names if name not in owner]
        if missing:
            raise ParameterSpaceError(
                f"This plan does not cover latent(s) {missing}: every latent of the space "
                "must be in exactly one block. An omitted latent is silently frozen at its "
                "declared init for the whole run — the sweep converges, the joint "
                "chi-squared settles, and nothing anywhere reports that a parameter you "
                "declared was never inferred. Add it to a block, or drop it from the space."
            )
        self._refuse_split_joint_prior(owner)
        return tuple((block, self._engine_of(block)) for block in self.blocks)

    def _refuse_split_joint_prior(self, owner: dict[str, "Block"]) -> None:
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
        joint = self.space.joint_prior
        if joint is None:
            return
        placed = ", ".join(
            f"{name!r} in Block{owner[name].names}" for name in joint.over
        )
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

    def _engine_of(self, block: Block) -> str:
        """Derive the block's engine from the declaration, or honour the override."""
        linear = [name for name in block.names if self.space.latent(name).linear]
        other = [name for name in block.names if not self.space.latent(name).linear]

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

    def __repr__(self) -> str:
        listed = ", ".join(
            f"{block.label}:{engine}" for block, engine in self._assign
        )
        return f"SamplingPlan({listed})"

    # -------------------------------------------------------------- running --

    def _prepare(
        self,
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
        forward, values0 = self.space.forward_fn(pipeline, state_template)
        check_observed_shape(
            jnp.shape(jax.eval_shape(forward, values0)),
            observed,
            predictor="this plan's model",
        )
        cond = Conditioning(
            space=self.space,
            pipeline=pipeline,
            state_template=state_template,
            observed=observed,
            noise=as_noise_model(noise),
            forward=forward,
        )
        for block, engine in self._assign:
            if engine == CONJUGATE:
                check_linearity(
                    self.space, pipeline, state_template, names=block.names, at=values0
                )
        return cond, values0

    def _identifiable(
        self, cond: Conditioning, values: dict[str, jax.Array], exit_name: str
    ) -> IdentifiabilityReport:
        """Refuse a model whose joint Jacobian has a null space, naming it.

        The refusal names the degenerate directions as combinations of
        **latents**, which is what
        :meth:`~rheplicant.inference.identifiability.IdentifiabilityReport.participation`
        reports and the whole reason it reports by name: "you have 8 blind
        directions" tells a user they have a problem and nothing about which.
        """
        report = identifiability(
            self.space, cond.pipeline, cond.state_template, at=values
        )
        if report.nullity == 0:
            return report

        lines = []
        for index in range(min(report.nullity, _DIRECTIONS_SHOWN)):
            share = report.participation(index)
            carried = sorted(share.items(), key=lambda item: -item[1])
            lines.append(
                f"  direction {index}: "
                + ", ".join(f"{name} {value:.2f}" for name, value in carried if value > 1e-3)
            )
        more = report.nullity - len(lines)
        if more > 0:
            lines.append(f"  ... and {more} more")

        raise ParameterSpaceError(
            f"{exit_name} refuses this model: its joint Jacobian has nullity "
            f"{report.nullity} of {report.n_par} parameters, so that many independent "
            "directions leave the prediction unchanged and any answer along them is "
            "arbitrary. No per-block guard can see this — a residual and a condition "
            "number are both computed from the block being solved — so the run would "
            "otherwise converge quietly onto one arbitrary point of the null space. The "
            "degenerate directions, as shares of each latent:\n"
            + "\n".join(lines)
            + "\nRe-parameterize (a smooth basis in place of one free parameter per cell "
            "is the usual repair), add data that breaks the degeneracy, or pass "
            "check_identifiability=False if you have another reason to believe the model. "
            "identifiability(space, pipeline, state) reports the same thing in full."
        )

    def _update(
        self,
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
        for index, (block, engine) in enumerate(self._assign):
            block_key = None if key is None else jax.random.fold_in(key, index)
            if engine == CONJUGATE:
                run = conjugate_draw if draw else conjugate_estimate
                extra = {"key": block_key} if draw else {}
                values, recorded = run(
                    cond, block.names, values,
                    tol=solve_tol, maxiter=None, require_convergence=solve_guard,
                    programs=programs, **extra,
                )
                residuals[block.names] = float(recorded)
            else:
                steps = DEFAULT_GRADIENT_STEPS if block.steps is None else block.steps
                if draw:
                    values, tuning[block.names] = gradient_draw(
                        cond, block.names, values, key=block_key, steps=steps,
                        tuning=tuning.get(block.names), adapt=adapt,
                        programs=programs,
                    )
                    potential = conditional_potential(cond, block.names, values)
                    residuals[block.names] = float(
                        potential({name: values[name] for name in block.names})
                    )
                else:
                    values, potential = gradient_estimate(
                        cond, block.names, values, steps=steps
                    )
                    residuals[block.names] = float(potential)
        return values

    # ---------------------------------------------------------- exit: point --

    def estimate(
        self,
        pipeline: AbstractOperator,
        state_template: State,
        observed: jax.Array,
        *,
        noise: Any,
        max_iter: int = DEFAULT_MAX_ITER,
        tol: float | None = DEFAULT_CHI2_TOL,
        min_sweeps: int = MIN_SWEEPS,
        check_identifiability: Any = CHECK_ONCE,
        solve_tol: float = 1e-6,
        solve_guard: float | None = 1e-3,
    ) -> Estimate:
        """Best fit: block-coordinate descent to a fixed point of the whole model.

        Every block is updated to its conditional best — a Wiener solve for a
        conjugate block, Adam on the conditional posterior for a gradient one —
        and the sweep repeats until the **joint** chi-squared stops moving.

        Args:
            pipeline: the forward model.
            state_template: the state it is evaluated on.
            observed: the data. Refused unless shaped exactly like the
                prediction.
            noise: a :class:`~rheplicant.inference.noise.NoiseModel`, or a bare
                sigma (wrapped as
                :class:`~rheplicant.inference.noise.HomoscedasticNoise`).
            max_iter: sweep cap.
            tol: relative PROGRESS in the joint chi-squared below which the run
                has converged — see :data:`DEFAULT_CHI2_TOL` for why it is a
                decrease and not a change. ``None`` runs exactly ``max_iter``
                sweeps and makes no convergence claim at all — the only way to
                get an answer back without one.
            min_sweeps: sweeps taken before the test is consulted.
            check_identifiability: ``"once"``, ``"each_sweep"`` or ``False``. See
                the module docstring; a point estimate is the exit that needs it
                most, because it has no other diagnostic.
            solve_tol: CG tolerance for conjugate blocks.
            solve_guard: bound on each conjugate solve's relative ERROR, as for
                :func:`~rheplicant.inference.linear.wiener_solve`. ``None`` skips
                the condition-number estimate, which is what a 10^6-coefficient
                block wants — see that function's own note on the bargain.

        Returns:
            An :class:`Estimate`.

        Raises:
            ParameterSpaceError: if the model is not identified; if ``observed``
                is mis-shaped; or if the joint chi-squared has not converged
                within ``max_iter`` sweeps. That last one is an error rather
                than a flag *here* and a flag rather than an error at
                :meth:`sample`, and the asymmetry is deliberate: a chain has
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
        if tol is not None and (
            not isinstance(min_sweeps, int) or not 1 <= min_sweeps <= max_iter
        ):
            raise ParameterSpaceError(
                f"estimate() needs 1 <= min_sweeps <= max_iter, got {min_sweeps!r} and "
                f"{max_iter!r}. A min_sweeps above the cap means the test is never "
                "consulted, so the run always exhausts max_iter and always refuses — "
                "including on a model it converged on at sweep two."
            )
        cond, values = self._prepare(
            pipeline, state_template, observed, noise, check_identifiability,
            "SamplingPlan.estimate",
        )
        report = None
        residuals: dict[tuple[str, ...], float] = {}
        programs: dict[Any, Any] = {}
        chi2 = [float(cond.chi2(values))]
        converged = None if tol is None else False
        # "once" is "due now, and never again"; "each_sweep" is "due every time".
        due, repeat = check_identifiability is not False, (
            check_identifiability == CHECK_EACH_SWEEP
        )

        for sweep in range(1, max_iter + 1):
            if due:
                report = self._identifiable(cond, values, "SamplingPlan.estimate")
                due = repeat
            values = self._update(
                cond, values, draw=False, key=None, adapt=False,
                solve_tol=solve_tol, solve_guard=solve_guard,
                tuning={}, residuals=residuals, programs=programs,
            )
            chi2.append(float(cond.chi2(values)))
            progress = chi2[-2] - chi2[-1]
            if (
                tol is not None
                and sweep >= min_sweeps
                and progress <= tol * max(abs(chi2[-1]), 1.0)
            ):
                converged = True
                break

        if converged is False:
            worst = max(
                (residuals[block.names] for block, engine in self._assign
                 if engine == CONJUGATE),
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
                f"SamplingPlan.estimate did not converge: after {max_iter} sweeps the "
                f"JOINT chi-squared is still falling by "
                f"{chi2[-2] - chi2[-1]:.6g} per sweep (chi2 = {chi2[-1]:.6g}), which "
                f"is above tol={tol:g}. " + hidden + "Slow convergence here "
                "means the blocks are correlated — group the correlated latents into ONE "
                "Block, which resolves them in a single solve, or raise max_iter. "
                "identifiability(space, pipeline, state, names=...) reports how much the "
                "partition is costing. Pass tol=None to accept an unconverged answer."
            )

        return Estimate(
            values=values,
            diagnostics=PlanDiagnostics(
                chi2=np.asarray(chi2, dtype=np.float64),
                sweeps=len(chi2) - 1,
                converged=converged,
                engines=dict(self.engines),
                block_residuals=dict(residuals),
                identifiability=report,
                noise_depends_on_prediction=bool(cond.noise.depends_on_prediction),
            ),
        )

    # --------------------------------------------------------- exit: sample --

    def sample(
        self,
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
        solve_guard: float | None = 1e-3,
    ) -> Draws:
        """Posterior draws: a Gibbs sweep over the same partition.

        Each conjugate block is drawn EXACTLY by
        :func:`~rheplicant.inference.linear.gcr_sample`, so a plan of conjugate
        blocks is an exact Gibbs sampler with nothing tuned. A gradient block
        takes ``steps`` NUTS steps instead, which makes the whole scheme
        Metropolis-within-Gibbs — see :class:`Block`'s ``steps`` and
        :func:`~rheplicant.inference.engines.gradient_draw`.

        Args:
            pipeline, state_template, observed, noise: as for :meth:`estimate`.
            key: PRNG key. Required — that is the point of this being a separate
                method rather than ``estimate(key=...)``.
            n_sweeps: total sweeps, warmup included.
            warmup: sweeps discarded. Defaults to half of ``n_sweeps``. NUTS
                tuning for gradient blocks adapts through warmup and is **frozen**
                afterwards, because a kernel that keeps adapting from the states
                it visits is no longer a valid transition.
            check_identifiability: as for :meth:`estimate`.
            rhat_max: split-``r_hat`` of the post-warmup joint chi-squared above
                which :attr:`PlanDiagnostics.converged` is ``False``. Reported,
                not raised: unlike a point estimate, a chain hands you the
                diagnostic along with the draws, and throwing away expensive
                draws over a scalar summary would be the worse trade.
            solve_tol, solve_guard: as for :meth:`estimate`.

        Returns:
            A :class:`Draws`. **Read ``diagnostics.rhat``.** The measured
            difference between a non-identified gain and the same model with an
            identifying tone is 1.824 against 1.002.

        Raises:
            ParameterSpaceError: if the model is not identified; if ``observed``
                is mis-shaped; if ``n_sweeps`` or ``warmup`` is not a sensible
                count; if fewer than :data:`MIN_DRAWS` draws would be kept; or if
                a gradient block has a member with no declared prior.
        """
        if not isinstance(n_sweeps, int) or n_sweeps < 1:
            raise ParameterSpaceError(
                f"sample() needs n_sweeps >= 1, got {n_sweeps!r}."
            )
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
        for block, engine in self._assign:
            if engine == GRADIENT:
                require_priors(self.space, block.names, block.label)

        cond, values = self._prepare(
            pipeline, state_template, observed, noise, check_identifiability,
            "SamplingPlan.sample",
        )
        report = None
        residuals: dict[tuple[str, ...], float] = {}
        tuning: dict[tuple[str, ...], Any] = {}
        programs: dict[Any, Any] = {}
        chi2: list[float] = []
        kept: dict[str, list[jax.Array]] = {name: [] for name in self.space.names}
        due, repeat = check_identifiability is not False, (
            check_identifiability == CHECK_EACH_SWEEP
        )

        for sweep in range(n_sweeps):
            if due:
                report = self._identifiable(cond, values, "SamplingPlan.sample")
                due = repeat
            values = self._update(
                cond, values, draw=True, key=jax.random.fold_in(key, sweep),
                adapt=sweep < warmup, solve_tol=solve_tol, solve_guard=solve_guard,
                tuning=tuning, residuals=residuals, programs=programs,
            )
            chi2.append(float(cond.chi2(values)))
            if sweep >= warmup:
                for name in self.space.names:
                    kept[name].append(values[name])

        rhat = split_rhat(chi2[warmup:])
        return Draws(
            samples={name: jnp.stack(stack) for name, stack in kept.items()},
            diagnostics=PlanDiagnostics(
                chi2=np.asarray(chi2, dtype=np.float64),
                sweeps=n_sweeps,
                converged=bool(rhat <= rhat_max),
                engines=dict(self.engines),
                block_residuals=dict(residuals),
                identifiability=report,
                noise_depends_on_prediction=bool(cond.noise.depends_on_prediction),
                warmup=warmup,
                rhat=rhat,
            ),
        )


__all__ = [
    "CHECK_EACH_SWEEP",
    "CHECK_ONCE",
    "DEFAULT_CHI2_TOL",
    "DEFAULT_MAX_ITER",
    "DEFAULT_RHAT_MAX",
    "MIN_DRAWS",
    "MIN_SWEEPS",
    "Block",
    "Draws",
    "Estimate",
    "PlanDiagnostics",
    "PlanResult",
    "SamplingPlan",
    "split_rhat",
]
