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
directions by latent; and the convergence monitor is a **joint** quantity at
the current parameter tuple across sweeps, never a per-block residual — which
is precisely the number that read ~1e-7 on an answer thousands of kelvin wrong.
For :meth:`SamplingPlan.estimate` that quantity is the joint negative log
posterior (:meth:`~rheplicant.inference.engines.Conditioning.neg_log_posterior`),
the objective every block update descends; the joint chi-squared is recorded
beside it, and :meth:`SamplingPlan.sample` tests its mixing on the chi-squared
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
:class:`~rheplicant.inference.compressed.BayesMemory` refuses to mix under its
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

import dataclasses
import math
from typing import Any, Protocol

import jax
import jax.numpy as jnp
import numpy as np
from bayesmith.optimize import certify

from rheplicant.core.errors import LinearityRefused, ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.engines import (
    CLOSED_FORM,
    CONJUGATE,
    DEFAULT_GRADIENT_STEPS,
    ENGINES,
    GRADIENT,
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
    require_priors,
)
from rheplicant.inference.identifiability import (
    IdentifiabilityReport,
    identifiability,
)
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.linear import _gaussian_parameters, check_linearity
from rheplicant.inference.loglinear import check_log_linearity, to_log_space
from rheplicant.inference.parameters import ParameterSpace
from rheplicant.inference.uncertainty import as_noise_model

#: ``check_identifiability="once"`` — the rank test runs at the starting values.
CHECK_ONCE: str = "once"

#: ``check_identifiability="each_sweep"`` — at every parameter tuple visited.
CHECK_EACH_SWEEP: str = "each_sweep"

#: Sweep cap for :meth:`SamplingPlan.estimate`.
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
#: (:func:`~rheplicant.inference.engines._decrement_program`). It runs only
#: on a sweep that passes the cheap tests: the change within ``tol``, and a
#: gap pre-screen (:func:`_gap_step`) that extrapolates the decreases at their
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

#: The first sweep at which :meth:`SamplingPlan.estimate` can report
#: converged, whatever ``min_sweeps`` says below it. The changes are counted
#: between sweep OUTPUTS, never from the starting values, so
#: :data:`_SETTLED_CHANGES` changes need one more sweep than that. A run with
#: a ``tol`` and a ``max_iter`` below this always refuses; the config
#: layer's pre-flight check A25 refuses such a document before it runs.
EARLIEST_CONVERGED_SWEEP: int = _SETTLED_CHANGES + 1

#: The resolution of a sweep-to-sweep change of the objective, in units of
#: machine epsilon: ``RESOLUTION_EPS * eps * sqrt(sum (s0 + s1)**2)`` over the
#: objective's terms, where ``s`` is each term's rounding magnitude (see
#: :func:`~rheplicant.inference.engines._monitor_programs`). The gap
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
#: :data:`_SETTLED_CHANGES` changes are within a tolerance, and that
#: tolerance floored at the dtype's resolution. The changes counted are
#: between sweep OUTPUTS, never from the starting values, so the earliest a
#: trace can settle is :data:`EARLIEST_CONVERGED_SWEEP`.
_settled = certify.settled
_effective_tol = certify.effective_tol


#: The gap PRE-SCREEN, from :mod:`~bayesmith.optimize.certify`: it picks
#: the sweeps at which the Newton decrement is computed and certifies
#: nothing. :func:`_certify` is what decides.
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
#: floor. See :meth:`SamplingPlan.estimate`, ``solve_tol``.
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


def _certify(programs: dict[Any, Any], cond: Any, values: dict[str, jax.Array],
             gap_tol: float, sweep: int, floor: float | None = None) -> _Attempt:
    """The Newton decrement at ``values``, and whether it certifies ``gap_tol``.

    The objective is :meth:`Conditioning.neg_log_posterior` over every
    latent, so the decrement is the distance to the MAP of the model the
    sweep is descending, in posterior sigma; ``gap_tol`` nats stands for
    ``sqrt(2 gap_tol)`` of them. The verdict reads the upper bound the solve's
    residual and the curvature floor allow, never the estimate alone, so an
    inexact solve can only make it refuse — see
    :func:`~bayesmith.optimize.certify.decrement`.

    ``floor`` is :meth:`SamplingPlan._curvature_floor`'s, and matters only
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
    precision where :meth:`SamplingPlan._curvature_floor`'s conditions hold.
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
    *, max_iter: int, tol: float, gap_tol: float, effective: float,
    changed: bool, objective: list[float], chi2: list[float],
    contraction: float | None, gap: float | None,
    rise: tuple[int, float] | None, attempt: _Attempt | None, solve_tol: float,
    dtype: Any, hidden: str,
) -> str:
    """:meth:`SamplingPlan.estimate`'s refusal at ``max_iter``.

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
    :meth:`SamplingPlan._curvature_floor`'s conditions) has not failed to
    converge — nothing here can say whether it has. The refusal says that
    instead, and :func:`_at_this_size` says what would change it.
    """
    limit = math.sqrt(2.0 * gap_tol)
    unsure = attempt is not None and not attempt.measured.proven
    headline = (
        "SamplingPlan.estimate cannot certify this estimate at this size and "
        "precision"
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
            "is below its rounding. " + _at_this_size(measured)
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
        f"The decrease contracts by {said_rho} per sweep{left}. " + hidden + rounding
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
        learning_rate: Adam step size for a **gradient** block at
            :meth:`SamplingPlan.estimate`, as a fraction of ``max|init|``.
            ``None`` takes
            :data:`~rheplicant.inference.engines.DEFAULT_LEARNING_RATE`.
            It sets how far the Adam steps travel in a sweep; the Newton steps
            that follow them set the precision, so the answer does not carry
            a floor proportional to it.
            It has no meaning at :meth:`SamplingPlan.sample`, where NUTS adapts
            its own step size, and none for a conjugate block, which has no
            iterate to step. Giving it to a conjugate block is an error rather
            than an ignored argument, for the same reason ``steps`` is.
    """

    names: tuple[str, ...]
    steps: int | None
    engine: str | None
    learning_rate: float | None

    def __init__(
        self,
        *names: str,
        steps: int | None = None,
        engine: str | None = None,
        learning_rate: float | None = None,
    ) -> None:
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "engine", engine)
        object.__setattr__(self, "learning_rate", learning_rate)
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
        if self.learning_rate is not None:
            if not self.learning_rate > 0.0:  # `not >` so a NaN is refused too
                raise ParameterSpaceError(
                    f"learning_rate must be > 0, got {self.learning_rate}. It is a "
                    "fraction of max|init|, not an absolute step."
                )
            if self.engine in CLOSED_FORM:
                raise ParameterSpaceError(
                    f"Block(..., engine={self.engine!r}, learning_rate=...) is a "
                    "contradiction: that block is solved in closed form and has no "
                    "iterate for a step size to scale. Drop learning_rate, or drop "
                    f"engine={self.engine!r} if the block is meant to be gradient."
                )

    @property
    def label(self) -> str:
        """How a message names this block."""
        return "(" + ", ".join(repr(name) for name in self.names) + ")"


@dataclasses.dataclass(frozen=True)
class PlanDiagnostics:
    """What a run measured, shared by both exits.

    Attributes:
        chi2: the JOINT chi-squared, one entry per sweep, the first at the
            starting values. For :meth:`SamplingPlan.estimate` it is reported
            data and not the stop rule: with a prior it can RISE as the run
            approaches the MAP. For :meth:`SamplingPlan.sample` it fluctuates
            around a stationary value, which is what :attr:`rhat` tests.
        sweeps: sweeps actually run.
        converged: for a point estimate, whether the Newton decrement at the
            returned point certified it within ``gap_tol`` of the objective's
            minimum (see :data:`DEFAULT_GAP_TOL`; ``None`` when the test was
            disabled). For a draw,
            whether :attr:`rhat` came in under the caller's threshold. **False
            here means the answer is not what it looks like** — the same
            reading as
            :attr:`~rheplicant.inference.gls.GLSResult.converged`.
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
        objective: the JOINT negative log posterior (up to a constant), one
            entry per sweep aligned with :attr:`chi2`, the first at the
            starting values — the quantity a point estimate's stop rule tests.
            ``None`` for a draw.
        effective_tol: the relative tolerance the stop rule applied,
            ``max(tol, OBJECTIVE_FLOOR_EPS * eps)`` for the objective's dtype
            (see :data:`OBJECTIVE_FLOOR_EPS`). ``None`` for a draw and for a
            point estimate run with ``tol=None``.
        contraction: the per-sweep contraction of the objective's decrease the
            gap pre-screen used at the last sweep (see :func:`_gap_step`), or
            ``None`` when it had none. ``None`` for a draw.
        distance_bound: the last Newton decrement's upper bound, in posterior
            sigma: ``sqrt(g^T H^-1 g)`` at the returned point plus the error
            its solve may have left (see :func:`_certify`), so
            at most ``sqrt(2 gap_tol)`` when :attr:`converged` is ``True``.
            ``inf`` when the residual could not bound it; ``None`` when no
            decrement was computed, and for a draw.
        certificate_iterations: the Hessian-vector products the last decrement
            took: its conjugate-gradient iterations, or one per latent where
            it formed the Hessian instead. ``None`` as for
            :attr:`distance_bound`.
        certificate_attempts: how many decrements the run computed (0 when no
            sweep was a candidate). ``None`` for a draw.
        solve_tol: the closed-form blocks' CG tolerance at the end of the run,
            after any tightening (see :meth:`SamplingPlan.estimate`). ``None``
            for a draw.
        floor_source: where the last certificate's curvature floor came from —
            ``"dense"`` (the formed Hessian's own smallest eigenvalue),
            ``"supplied"`` (this plan's prior-precision floor, see
            :meth:`SamplingPlan._curvature_floor`), ``"probe"`` (a Lanczos
            estimate, which never certifies) or ``"none"``. ``None`` as for
            :attr:`distance_bound`.

    :attr:`objective` and the fields after it are not among the fields the
    config layer copies into a run's diagnostics record
    (``config/products/extractors.py::DIAGNOSTIC_FIELDS``), so that record's
    format is unchanged by them.
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
    objective: np.ndarray | None = None
    effective_tol: float | None = None
    contraction: float | None = None
    distance_bound: float | None = None
    certificate_iterations: int | None = None
    certificate_attempts: int | None = None
    solve_tol: float | None = None
    floor_source: str | None = None


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

    def __repr__(self) -> str:
        listed = ", ".join(
            f"{block.label}:{engine}" for block, engine in self._assign
        )
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
        if bool(cond.noise.depends_on_prediction):
            return None
        floor = math.inf
        for name in self.space.names:
            latent = self.space.latent(name)
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
        single = len(self._assign) == 1 and self._assign[0][1] == CONJUGATE
        if not single and not self._jointly_affine(cond):
            return None
        return floor

    def _jointly_affine(self, cond: Conditioning) -> bool:
        """Whether the prediction is affine in ALL latents at once.

        The check :meth:`_prepare` runs per conjugate block, asked of the
        union and answered rather than raised: its refusal is this plan's
        answer, not this plan's failure.
        """
        try:
            check_linearity(
                self.space, cond.pipeline, cond.state_template,
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
        normalized = as_noise_model(noise)
        log_observed, log_sigma = None, None
        if any(engine == LOG_CONJUGATE for _, engine in self._assign):
            log_observed, log_sigma = to_log_space(observed, normalized)

        cond = Conditioning(
            space=self.space,
            pipeline=pipeline,
            state_template=state_template,
            observed=observed,
            noise=normalized,
            forward=forward,
            log_observed=log_observed,
            log_sigma=log_sigma,
        )
        for block, engine in self._assign:
            if engine == CONJUGATE:
                check_linearity(
                    self.space, pipeline, state_template, names=block.names, at=values0
                )
            elif engine == LOG_CONJUGATE:
                check_log_linearity(
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
            if engine in CLOSED_FORM:
                log = engine == LOG_CONJUGATE
                if draw:
                    run = log_conjugate_draw if log else conjugate_draw
                else:
                    run = log_conjugate_estimate if log else conjugate_estimate
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
                        cond, block.names, values, steps=steps, programs=programs,
                        **({} if block.learning_rate is None
                           else {"learning_rate": block.learning_rate}),
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
          have left is added (see :data:`DEFAULT_GAP_TOL` and :func:`_certify`).
          It does not grow with the number of data, and it sees every mode,
          the slow ones included.
        * **the schedule**: the decrement is computed only on a sweep whose
          last two changes of ``f`` are within the effective tolerance
          ``max(tol, OBJECTIVE_FLOOR_EPS * eps)``, relative to ``|f|``, and
          whose decrease passes the gap pre-screen (:func:`_gap_step`) or is
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
        :attr:`PlanDiagnostics.solve_tol`. A model that certifies at the
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
                :data:`EARLIEST_CONVERGED_SWEEP` (3) sweeps (see
                ``min_sweeps``), so ``max_iter`` of 1 or 2 can never converge
                and always refuses.
            tol: relative change in the joint negative log posterior below
                which the run has converged, required on two consecutive
                sweep-to-sweep changes — see :data:`DEFAULT_CHI2_TOL`. It is
                floored at :data:`OBJECTIVE_FLOOR_EPS` machine epsilons of the
                objective's dtype, and the value applied is recorded as
                :attr:`PlanDiagnostics.effective_tol`. ``None`` runs exactly
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
                :func:`~rheplicant.inference.linear.wiener_solve`. ``None`` skips
                the condition-number estimate, which is what a 10^6-coefficient
                block wants — see that function's own note on the bargain.
            gap_tol: the certificate's threshold, in nats of the joint
                negative log posterior: the Newton decrement must be at most
                ``2 gap_tol`` — see :data:`DEFAULT_GAP_TOL`. The last
                decrement's upper bound is recorded as
                :attr:`PlanDiagnostics.distance_bound`, in posterior sigma.
                Not consulted when ``tol`` is ``None``.

        Returns:
            An :class:`Estimate`.

        Raises:
            ParameterSpaceError: if the model is not identified; if ``observed``
                is mis-shaped; or if the joint negative log posterior has not
                settled within ``max_iter`` sweeps. That last one is an error rather
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
                "including on a model that had already settled."
            )
        cond, values = self._prepare(
            pipeline, state_template, observed, noise, check_identifiability,
            "SamplingPlan.estimate",
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
        closed = any(engine in CLOSED_FORM for _, engine in self._assign)
        # Named apart from the sweep's `floor`, which is the CG TOLERANCE's:
        # the two are both floors and neither is the other's. Asked only where
        # it can matter, since below DENSE_MAX the decrement measures the
        # curvature itself and proving a floor would cost a linearity check
        # this run has no use for.
        curvature = (
            self._curvature_floor(cond)
            if certify.real_size(values) > certify.DENSE_MAX
            else None
        )
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
                solve_tol=tightened, solve_guard=solve_guard,
                tuning={}, residuals=residuals, programs=programs,
            )
            chi2_now, objective_now, following, following_scales = measure(values)
            decrease, resolution = change(terms, scales, following, following_scales)
            terms, scales = following, following_scales
            decrease, resolution = float(decrease), float(resolution)
            chi2.append(float(chi2_now))
            objective.append(float(objective_now))
            gap_state, screened, gap, rho = _gap_step(
                gap_state, decrease, resolution, gap_tol
            )
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
                _not_converged_message(
                    max_iter=max_iter, tol=tol, gap_tol=gap_tol,
                    effective=effective, changed=changed, objective=objective,
                    chi2=chi2, contraction=measured, gap=gap,
                    rise=last_rise, attempt=attempt, solve_tol=tightened,
                    dtype=jnp.result_type(objective_now), hidden=hidden,
                )
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
                objective=np.asarray(objective, dtype=np.float64),
                effective_tol=effective,
                contraction=rho,
                distance_bound=None if attempt is None else attempt.measured.distance,
                certificate_iterations=(
                    None if attempt is None else attempt.measured.products
                ),
                certificate_attempts=attempts,
                solve_tol=tightened,
                floor_source=None if attempt is None else attempt.measured.floor_source,
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
        solve_guard: float | None = None,
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
