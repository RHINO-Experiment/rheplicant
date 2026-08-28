"""SkySpaceFilter: map-make onto the sky, then reproject.

The JAX form of limTOD's ``wiener_filter_map`` / ``HPW_mapmaking``: solve the
regularised normal equations for the sky map best explaining the data,

    (A^T N^-1 A + lam I) m = A^T N^-1 d,

with A any *linear* :class:`~rheplicant.radio.sky.projection.AbstractSkyProjector`
(forward + adjoint) — the SAME object that generates the sky term in the
forward model. ``extract`` returns ``A m`` (the sky-locked component of the
data), ``remove`` returns the sky-subtracted residual.

The solve uses matrix-free conjugate gradients
(``jax.scipy.sparse.linalg.cg``, built on ``lax.custom_linear_solve``), so
the whole filter is differentiable — filter transfer functions can be
marginalised in inference.

**Convergence has to be asked for, and CG will not volunteer it.** JAX's ``cg``
returns ``(x, None)``: there is no status to read, so a solve that ran out of
iterations comes back looking exactly like one that converged — and
``mode="remove"`` subtracts it from the data all the same. ``require_convergence``
turns that silence into a refusal.

**A residual is not an accuracy**, which is why the knob is stated as an error.
The two differ by the condition number of ``A^T W A + lam I``, and for map-making
that number is large *by construction*: sky pixels the scan never touched are
held by the ridge alone, so lam is exactly the bottom of the spectrum while the
top is set by the data. CG then stops on a tiny residual with those pixels still
at their starting value. So the guard estimates kappa by power iteration
(:mod:`rheplicant.core.conditioning`) and limits ``kappa * residual``, the bound
on the relative error. It is off by default: unlike a Wiener solve, which runs
once, a filter runs on every evaluation of the signal path, and the bound costs
a fixed ``POWER_ITERATIONS`` unrolled applications of the normal operator on
top of the one the residual needs.

Noise weighting: if ``state.aux["flags"]`` exists (e.g. from
MomentRFI flagging), flagged samples get zero weight in ``N^-1``.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.conditioning import (
    POWER_ITERATIONS,
    largest_eigenvalue,
    tree_norm,
)
from rheplicant.core.errors import StateValidationError
from rheplicant.core.state import State
from rheplicant.radio.backend.flagging import FLAGS_KEY
from rheplicant.radio.filters.base import AbstractLinearFilter
from rheplicant.radio.sky.projection import AbstractSkyProjector


class SkySpaceFilter(AbstractLinearFilter):
    """Wiener-like sky projection filter built on a linear sky projector.

    Attributes:
        projector: linear sky projector supplying ``forward``/``adjoint``.
        regularization: ridge strength lam (differentiable scalar; acts as a
            white prior inverse-variance, stabilising unseen pixels).
        cg_tol: conjugate-gradient tolerance (static).
        cg_maxiter: conjugate-gradient iteration cap (static).
        mode: ``"extract"`` (sky-locked component) or ``"remove"`` (residual).
        require_convergence: bound on the solve's relative ERROR, or ``None``
            (the default) to run unchecked as before; see the module docstring
            for why an error and not a residual, and what checking costs.
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    projector: AbstractSkyProjector
    regularization: jax.Array
    cg_tol: float = eqx.field(static=True, default=1e-8)
    cg_maxiter: int = eqx.field(static=True, default=100)
    mode: str = eqx.field(static=True, default="remove")
    require_convergence: float | None = eqx.field(static=True, default=None)

    def __check_init__(self):
        if not isinstance(self.projector, AbstractSkyProjector):
            raise StateValidationError(
                f"projector must be an AbstractSkyProjector, got {type(self.projector).__name__}."
            )
        if not isinstance(self.cg_maxiter, int) or self.cg_maxiter < 1:
            raise StateValidationError(
                f"cg_maxiter must be a positive int, got {self.cg_maxiter!r}."
            )
        # bool is an int, and True would read as "guard at a relative error of
        # 1" -- a target so loose it certifies nothing while looking switched on.
        if self.require_convergence is not None and (
            isinstance(self.require_convergence, bool)
            or not isinstance(self.require_convergence, (int, float))
            or not self.require_convergence > 0
        ):
            raise StateValidationError(
                "require_convergence must be a positive number or None, got "
                f"{self.require_convergence!r}."
            )

    def project(self, data: jax.Array, state: State) -> jax.Array:
        coords = state.coords
        flags = state.aux.get(FLAGS_KEY)
        weights = 1.0 - flags.astype(data.dtype) if flags is not None else jnp.ones_like(data)

        def normal_op(sky: jax.Array) -> jax.Array:
            tod = self.projector.forward(sky, coords)
            back = self.projector.adjoint(weights * tod, coords)
            return back + self.regularization * sky

        rhs = self.projector.adjoint(weights * data, coords)
        sky_hat, _ = jax.scipy.sparse.linalg.cg(
            normal_op, rhs, tol=self.cg_tol, maxiter=self.cg_maxiter
        )
        if self.require_convergence is not None:
            sky_hat = self._checked(sky_hat, normal_op, rhs)
        return self.projector.forward(sky_hat, coords)

    def _checked(self, sky_hat, normal_op, rhs):
        """``sky_hat`` again, refusing if its relative error cannot be bounded.

        The residual costs one more application of ``normal_op`` because JAX's
        ``cg`` discards it; kappa costs ``POWER_ITERATIONS`` more. The verdict
        is on ``kappa * residual`` — see the module docstring for why the
        residual alone decides nothing here.

        ``eqx.error_if`` rather than a Python ``if``, because the comparison is
        on traced values and this filter is meant to run under ``jit``.

        **kappa is bounded, not estimated, and the difference is the whole
        safety of this guard.** ``lam_min(A^T W A + lam I) >= min(lam)``
        exactly, because ``A^T W A`` is positive semi-definite -- so
        ``lam_max / min(lam)`` bounds kappa from ABOVE and the error bound
        built on it can only be too large, never too small. A refusal and a
        pass are both sound.

        This function shipped measuring ``lam_min`` with
        ``extreme_eigenvalues`` and flooring the measurement, which gave a
        kappa that was a LOWER bound: on the five-pixel fixture with one pixel
        the scan never touches and a ridge of 1e-8, the true kappa is about
        5e+09 and twelve iterations reported 3.0e+04. That case was refused
        anyway, so nothing here was observably wrong -- but a guard whose
        number leans toward silence is one defect away from being useless, and
        ``inference/linear.py::condition_bound`` carries the same correction
        and the measurements that forced it.
        """
        misfit = normal_op(sky_hat) - rhs
        residual = tree_norm(misfit) / jnp.maximum(tree_norm(rhs), 1e-30)
        largest = largest_eigenvalue(
            normal_op, rhs, jax.random.key(0), POWER_ITERATIONS
        )
        # A^T W A is positive semi-definite, so lam_min cannot fall below the
        # ridge however little of the sky the scan actually saw. Taking the
        # smallest entry keeps the floor a true lower bound for a per-pixel lam.
        floor = jnp.min(self.regularization)
        kappa = largest / floor
        bad = jnp.logical_or(
            ~jnp.isfinite(residual), residual * kappa > self.require_convergence
        )

        # Below kappa * eps the arithmetic cannot represent an answer that
        # accurate, and the natural response to the other message -- tighten
        # cg_tol, raise cg_maxiter -- burns iterations to arrive somewhere
        # equally wrong. Worth its own verdict, because the remedy is different.
        epsilon = float(jnp.finfo(jnp.asarray(rhs).dtype).eps)
        unreachable = kappa * epsilon > self.require_convergence

        sky_hat = eqx.error_if(
            sky_hat,
            jnp.logical_and(bad, unreachable),
            "SkySpaceFilter cannot reach require_convergence at this precision: "
            "the normal operator's condition number times the machine epsilon "
            "already exceeds it, so no cg_tol or cg_maxiter will help. This is "
            "the usual signature of a sky the scan does not constrain -- pixels "
            "no sample touched are held by the ridge alone. Enable "
            "jax_enable_x64, or raise regularization, which bounds the "
            "conditioning (kappa is about ||A^T W A|| / regularization).",
        )
        return eqx.error_if(
            sky_hat,
            jnp.logical_and(bad, ~unreachable),
            "SkySpaceFilter's map-making CG did not converge: the relative "
            "residual times the normal operator's condition number -- the bound "
            "on the RELATIVE ERROR, which is what require_convergence limits -- "
            "exceeds it. The residual alone looks converged; it is not, along "
            "the directions the ridge dominates. Pass cg_tol about "
            "require_convergence/kappa with a cg_maxiter to match, or raise "
            "regularization.",
        )
