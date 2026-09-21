"""What a plan HANDS BACK.

A block, the per-sweep diagnostics, and the three result shapes: a plan
result, a point estimate and a set of draws. Separated because every consumer
of this package reads these and almost none of them constructs a plan.
"""

import dataclasses
from typing import Protocol

import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.engines import (
    CLOSED_FORM,
    ENGINES,
)
from rheplicant.inference.identifiability import (
    IdentifiabilityReport,
)


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
            ``SamplingPlan.sample``. ``None`` takes
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
            It has no meaning at ``SamplingPlan.sample``, where NUTS adapts
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
            approaches the MAP. For ``SamplingPlan.sample`` it fluctuates
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
            gap pre-screen used at the last sweep (see ``_gap_step``), or
            ``None`` when it had none. ``None`` for a draw.
        distance_bound: the last Newton decrement's upper bound, in posterior
            sigma: ``sqrt(g^T H^-1 g)`` at the returned point plus the error
            its solve may have left (see ``_certify``), so
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
            ``_curvature_floor``), ``"probe"`` (a Lanczos
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
