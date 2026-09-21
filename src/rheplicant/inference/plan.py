"""Sampling plans: one declared partition, two exits — a point estimate and a draw.

Everything under :mod:`rheplicant.inference` up to here builds *one* block's
answer. :func:`~rheplicant.inference.linear_solve.wiener_solve` is a linear-Gaussian
block's posterior mean and :func:`~rheplicant.inference.linear_solve.gcr_sample` is an
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
``SamplingPlan.sample`` and absent from :meth:`~rheplicant.inference.plan.SamplingPlan.estimate`, so
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
directions by latent; and the convergence monitor is a **joint** quantity at
the current parameter tuple across sweeps, never a per-block residual — which
is precisely the number that read ~1e-7 on an answer thousands of kelvin wrong.
For :meth:`~rheplicant.inference.plan.SamplingPlan.estimate` that quantity is the joint negative log
posterior (:meth:`~rheplicant.inference.engines.Conditioning.neg_log_posterior`),
the objective every block update descends; the joint chi-squared is recorded
beside it, and ``SamplingPlan.sample`` tests its mixing on the chi-squared
trace.

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

**The gradient blocks used to omit the same term, by a different route, and
that half went unsaid for longer.** A conjugate block freezes sigma to keep
its step linear; a gradient block does not freeze it -- sigma is re-evaluated
at the current prediction inside
:func:`~rheplicant.inference.engines.conditional_potential` -- but that
potential was ``0.5 * chi2 - log_prior``, and ``chi2`` carries no
``sum log sigma``. So when sigma depended on the prediction, a gradient block
targeted the density divided by ``prod sigma(theta)``: the GLS-flavoured
posterior again, not the full one. The contrast that made it visible is
:func:`~rheplicant.inference.numpyro_bridge.to_numpyro_model`, whose
observation site is a ``Normal`` whose ``log_prob`` carries ``-log scale``
automatically -- so the ``nuts`` exit sampled the full density while a
gradient block sampled the GLS-flavoured one, from the same declared model.
It is the same distinction
:class:`~rheplicant.inference.memory.BayesMemory` refuses to mix under its
``estimator`` field, which is why it was written down here rather than left to
be discovered.

**Closed 2026-08-28** (migration ledger B1).
:meth:`~rheplicant.inference.engines.Conditioning.neg_log_likelihood` is now
``0.5 * chi2 + log_determinant``, and both potential builders take it -- the
single-argument one the optimiser gets and the lifted one NUTS gets, since
fixing one alone would have rebuilt the same two-targets defect a layer down.
``chi2`` itself is deliberately unchanged: it is reported goodness of fit and
the sampler's mixing trace, and a number that changed units the moment a
noise model started reading its argument would be worse than the omission it
replaced. The point estimate's stop rule reads the full objective, log
determinant included, since T-002 (A5-1).

The gradient block moved from **6.248269** to **5.004059** on the fixture
below, against an unbiased closed form of 5.104641 -- so onto the unbiased
side, 2.0% from the closed form where it had been 22% away. That remaining
2.0% is the prior and not a residue of B1: the fixture declares ``w`` with
``mu = exp(w) x``, so a ``Normal`` prior on ``w`` is a ``1/scale`` prior on
the recovered scale. ``tests/inference/test_potential_carries_the_logdet.py``
asserts the identity rather than the landing place for exactly that reason,
and compares the two routes across the seam named above.

One declaration cannot be honoured by a plan and is refused instead of
overridden: ``inference.noise.include_logdet: false`` selects generalized
least squares, which is a point estimator and is not a posterior, so
``plan.estimate`` and ``plan.sample`` decline it by name
(``config/sections/exits.py::_a49_is_not_honourable_by_a_plan``). Threading it
down would have made one word mean two things at the two exits, which is the
defect B1 IS.

**It is the BLOCK TYPE that decides, not the exit** -- worth stating plainly,
because this paragraph and the migration spec both first described it as
something ``plan.sample`` does. Measured across both packages on
``mu = w x`` with ``sigma = 0.5 |mu|``, n=40, prior ``N(0, 100)``, where the
two closed forms are **5.104641** (log-determinant kept, the unbiased one)
and **6.258841** (dropped, GLS-flavoured):

======================================  ==========  ===================
exit                                    lands at    which side
======================================  ==========  ===================
``plan.estimate``, CONJUGATE block      5.104558    unbiased
``plan.estimate``, GRADIENT block       6.248269    GLS-flavoured
  *the same, after B1 closed*           5.004059    unbiased
======================================  ==========  ===================

So ``estimate`` showed it too, and a conjugate block never did: freezing sigma
per inner solve puts its fixed point on the unbiased side, which is the same
argument :func:`~rheplicant.inference.gls.iterative_gls` makes for itself.
The ratio was ``1.2261`` against ``(1 + f^2) = 1.25`` at ``f = 0.5``, the gap
being finite-sample scatter at n=40. The difference is ``O(f^2)`` with
``f = 1/sqrt(delta_nu tau)`` and was small in most regimes -- but it was
attached to which ENGINE ran, and reading it as a property of one exit is
what left the estimate path unexamined for as long as it was.

The first two rows are kept at their measured values rather than dropped. The
conjugate row still holds; the gradient row is what the defect looked like,
and a fix whose record does not say what it changed cannot be checked.

The numbers are bayesmith's cross-check
(``tests/crosscheck/test_dispatch.py``, recorded in
``docs/migration/plan.md``), where the port's own conjugate estimate agrees
with the first row to **9e-12** and its nonlinear path declines to give a
point estimate at all.
"""

from typing import Any

import jax

from rheplicant.core.errors import LinearityRefused, ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.engines import (
    Conditioning,
)
from rheplicant.inference.identifiability import (
    IdentifiabilityReport,
    identifiability,
)
from rheplicant.inference.linear import check_linearity
from rheplicant.inference.parameters import ParameterSpace

from .plan_blocks import (
    curvature_floor,
    engine_of,
    partition_of,
    prepare_conditioning,
    refuse_split_joint_prior,
)
from .plan_draws import (
    run_sample,
)
from .plan_estimate import (
    advance,
    run_estimate,
)
from .plan_results import (
    Block,
    Draws,
    Estimate,
    PlanDiagnostics,
)
from .plan_results import PlanResult as PlanResult
from .plan_settings import _DECREMENT_TAG as _DECREMENT_TAG
from .plan_settings import (
    _DIRECTIONS_SHOWN,
    CHECK_EACH_SWEEP,
    CHECK_ONCE,
    DEFAULT_CHI2_TOL,
    DEFAULT_GAP_TOL,
    DEFAULT_MAX_ITER,
    DEFAULT_RHAT_MAX,
    MIN_DRAWS,
    MIN_SWEEPS,
    RESOLUTION_EPS,
    split_rhat,
)
from .plan_settings import _SETTLED_CHANGES as _SETTLED_CHANGES

# Re-exported so this module's importers keep working. `X as X` is the
# explicit re-export form: a plain import of a name this file does not
# itself use is F401, and `ruff --fix` deletes it whatever the comment
# on the line says.
from .plan_settings import EARLIEST_CONVERGED_SWEEP as EARLIEST_CONVERGED_SWEEP
from .plan_settings import OBJECTIVE_FLOOR_EPS as OBJECTIVE_FLOOR_EPS
from .plan_settings import _at_this_size as _at_this_size
from .plan_settings import _Attempt as _Attempt
from .plan_settings import _halves as _halves
from .plan_settings import _not_converged_message as _not_converged_message


class SamplingPlan:
    """A partition of a parameter space into blocks, with two exits.

    See the module docstring for the design and for the measured failure this
    exists to prevent.

    Args:
        space: the parameter declaration this plan partitions.
        *blocks: the :class:`~rheplicant.inference.plan_results.Block` s, in the order a sweep
        visits them.

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

    @classmethod
    def automatic(
        cls,
        space: ParameterSpace,
        pipeline: AbstractOperator,
        state_template: State,
        **options: Any,
    ) -> "SamplingPlan":
        """The same plan, over a partition derived from the model.

        :func:`~rheplicant.inference.partition.auto_blocks` groups the
        declared-linear latents into as many conjugate blocks as the model
        needs — one per factor of a multilinear form, since latents that are
        each affine alone need not be affine together — and puts everything
        else in one gradient block. ``**options`` are that function's; it owns
        the vocabulary, so ``steps=`` and the probe's tolerances are spelled
        once rather than restated here.

        This needs the pipeline, which ``__init__`` does not: which latents may
        share a conjugate block is a property of the prediction, and no
        declaration carries it.

        Args:
            space: the parameter declaration to partition.
            pipeline, state_template: the model, probed to find the partition.
            **options: forwarded to
                :func:`~rheplicant.inference.partition.auto_blocks`.

        Returns:
            A plan whose blocks were derived. Nothing else differs — the
            partition is checked, and each conjugate block's joint linearity
            re-verified at the first sweep, exactly as for a declared one.
        """
        from rheplicant.inference.partition import auto_blocks

        return cls(space, *auto_blocks(space, pipeline, state_template, **options))

    # ------------------------------------------------------------ declaring --

    def _partition(self) -> tuple[tuple[Block, str], ...]:
        """Check the partition, then derive each block's engine.

        Order matters: a block naming an undeclared latent cannot have its
        engine derived at all, so the partition is settled first.
        """
        return partition_of(self)

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
        return refuse_split_joint_prior(self, owner)

    def _engine_of(self, block: Block) -> str:
        """Derive the block's engine from the declaration, or honour the override."""
        return engine_of(self, block)

    def __repr__(self) -> str:
        listed = ", ".join(f"{block.label}:{engine}" for block, engine in self._assign)
        return f"SamplingPlan({listed})"

    # -------------------------------------------------------------- running --

    def _curvature_floor(self, cond: Conditioning) -> float | None:
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
        return curvature_floor(self, cond)

    def _jointly_affine(self, cond: Conditioning) -> bool:
        """Whether the prediction is affine in ALL latents at once.

        The check :meth:`_prepare` runs per conjugate block, asked of the
        union and answered rather than raised: its refusal is this plan's
        answer, not this plan's failure.
        """
        try:
            check_linearity(
                self.space,
                cond.pipeline,
                cond.state_template,
                names=self.space.names,
            )
        except LinearityRefused:
            return False
        return True

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
        :func:`~rheplicant.inference.linear_solve.gcr_sample` recommends for a sweep,
        which is what lets every rebuild inside the loop pass ``check=False``.
        """
        return prepare_conditioning(
            self, pipeline, state_template, observed, noise, check, exit_name
        )

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
        report = identifiability(self.space, cond.pipeline, cond.state_template, at=values)
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
        return advance(
            self,
            cond,
            values,
            draw=draw,
            key=key,
            adapt=adapt,
            solve_tol=solve_tol,
            solve_guard=solve_guard,
            tuning=tuning,
            residuals=residuals,
            programs=programs,
        )

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
        :attr:`~rheplicant.inference.plan_results.PlanDiagnostics.solve_tol`. A model that certifies
        at the
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
                :attr:`~rheplicant.inference.plan_results.PlanDiagnostics.effective_tol`. ``None``
                runs exactly
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
                :attr:`~rheplicant.inference.plan_results.PlanDiagnostics.distance_bound`, in
                posterior sigma.
                Not consulted when ``tol`` is ``None``.

        Returns:
            An :class:`~rheplicant.inference.plan_results.Estimate`.

        Raises:
            ParameterSpaceError: if the model is not identified; if ``observed``
                is mis-shaped; or if the joint negative log posterior has not
                settled within ``max_iter`` sweeps. That last one is an error rather
                than a flag *here* and a flag rather than an error at
                :meth:`sample`, and the asymmetry is deliberate: a chain has
                ``r_hat`` to scream with, and a point estimate has nothing.
        """
        return run_estimate(
            self,
            pipeline,
            state_template,
            observed,
            noise=noise,
            max_iter=max_iter,
            tol=tol,
            min_sweeps=min_sweeps,
            check_identifiability=check_identifiability,
            solve_tol=solve_tol,
            solve_guard=solve_guard,
            gap_tol=gap_tol,
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
        solve_guard: float | None = None,
    ) -> Draws:
        """Posterior draws: a Gibbs sweep over the same partition.

        Each conjugate block is drawn EXACTLY by
        :func:`~rheplicant.inference.linear_solve.gcr_sample`, so a plan of conjugate
        blocks is an exact Gibbs sampler with nothing tuned. A gradient block
        takes ``steps`` NUTS steps instead, which makes the whole scheme
        Metropolis-within-Gibbs — see :class:`~rheplicant.inference.plan_results.Block`'s ``steps``
        and
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
                which :attr:`~rheplicant.inference.plan_results.PlanDiagnostics.converged` is
                ``False``. Reported,
                not raised: unlike a point estimate, a chain hands you the
                diagnostic along with the draws, and throwing away expensive
                draws over a scalar summary would be the worse trade.
            solve_tol, solve_guard: as for :meth:`estimate`.

        Returns:
            A :class:`~rheplicant.inference.plan_results.Draws`. **Read ``diagnostics.rhat``.** The
            measured
            difference between a non-identified gain and the same model with an
            identifying tone is 1.824 against 1.002.

        Raises:
            ParameterSpaceError: if the model is not identified; if ``observed``
                is mis-shaped; if ``n_sweeps`` or ``warmup`` is not a sensible
                count; if fewer than :data:`~rheplicant.inference.plan_settings.MIN_DRAWS` draws
                would be kept; or if
                a gradient block has a member with no declared prior.
        """
        return run_sample(
            self,
            pipeline,
            state_template,
            observed,
            noise=noise,
            key=key,
            n_sweeps=n_sweeps,
            warmup=warmup,
            check_identifiability=check_identifiability,
            rhat_max=rhat_max,
            solve_tol=solve_tol,
            solve_guard=solve_guard,
        )


__all__ = [
    "CHECK_EACH_SWEEP",
    "CHECK_ONCE",
    "DEFAULT_CHI2_TOL",
    "DEFAULT_GAP_TOL",
    "DEFAULT_MAX_ITER",
    "DEFAULT_RHAT_MAX",
    "EARLIEST_CONVERGED_SWEEP",
    "MIN_DRAWS",
    "MIN_SWEEPS",
    "OBJECTIVE_FLOOR_EPS",
    "RESOLUTION_EPS",
    "Block",
    "Draws",
    "Estimate",
    "PlanDiagnostics",
    "PlanResult",
    "SamplingPlan",
    "split_rhat",
]
