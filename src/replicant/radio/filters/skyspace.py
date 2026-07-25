"""SkySpaceFilter: map-make onto the sky, then reproject.

The JAX form of limTOD's ``wiener_filter_map`` / ``HPW_mapmaking``: solve the
regularised normal equations for the sky map best explaining the data,

    (A^T N^-1 A + lam I) m = A^T N^-1 d,

with A any *linear* :class:`~replicant.radio.sky.projection.AbstractSkyProjector`
(forward + adjoint) — the SAME object that generates the sky term in the
forward model. ``extract`` returns ``A m`` (the sky-locked component of the
data), ``remove`` returns the sky-subtracted residual.

The solve uses matrix-free conjugate gradients
(``jax.scipy.sparse.linalg.cg``, built on ``lax.custom_linear_solve``), so
the whole filter is differentiable — filter transfer functions can be
marginalised in inference.

Noise weighting: if ``state.aux["flags"]`` exists (e.g. from
MomentRFI flagging), flagged samples get zero weight in ``N^-1``.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant.core.errors import StateValidationError
from replicant.core.state import State
from replicant.radio.filters.base import AbstractLinearFilter
from replicant.radio.sky.projection import AbstractSkyProjector


class SkySpaceFilter(AbstractLinearFilter):
    """Wiener-like sky projection filter built on a linear sky projector.

    Attributes:
        projector: linear sky projector supplying ``forward``/``adjoint``.
        regularization: ridge strength lam (differentiable scalar; acts as a
            white prior inverse-variance, stabilising unseen pixels).
        cg_tol: conjugate-gradient tolerance (static).
        cg_maxiter: conjugate-gradient iteration cap (static).
        mode: ``"extract"`` (sky-locked component) or ``"remove"`` (residual).
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "coords")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    projector: AbstractSkyProjector
    regularization: jax.Array
    cg_tol: float = eqx.field(static=True, default=1e-8)
    cg_maxiter: int = eqx.field(static=True, default=100)
    mode: str = eqx.field(static=True, default="remove")

    def __check_init__(self):
        if not isinstance(self.projector, AbstractSkyProjector):
            raise StateValidationError(
                f"projector must be an AbstractSkyProjector, got {type(self.projector).__name__}."
            )
        if not isinstance(self.cg_maxiter, int) or self.cg_maxiter < 1:
            raise StateValidationError(
                f"cg_maxiter must be a positive int, got {self.cg_maxiter!r}."
            )

    def project(self, data: jax.Array, state: State) -> jax.Array:
        coords = state.coords
        flags = state.aux.get("flags")
        weights = 1.0 - flags.astype(data.dtype) if flags is not None else jnp.ones_like(data)

        def normal_op(sky: jax.Array) -> jax.Array:
            tod = self.projector.forward(sky, coords)
            back = self.projector.adjoint(weights * tod, coords)
            return back + self.regularization * sky

        rhs = self.projector.adjoint(weights * data, coords)
        sky_hat, _ = jax.scipy.sparse.linalg.cg(
            normal_op, rhs, tol=self.cg_tol, maxiter=self.cg_maxiter
        )
        return self.projector.forward(sky_hat, coords)
