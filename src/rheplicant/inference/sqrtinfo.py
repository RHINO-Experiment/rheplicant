"""The square-root information form: a log-quadratic that cannot go indefinite.

A term is ``[R | z]`` with ``log L(x) = -0.5 * ||R x - z||^2 + offset``, so the
Fisher information it carries is ``F = R^T R`` -- positive semi-definite by
construction rather than by hope. Three consequences the accumulation layer
depends on:

* **Rank deficiency is representable and cheap.** One epoch rarely constrains
  every global parameter; its ``R`` simply has fewer rows than columns. The
  equivalent statement in ``(F, b)`` form is ``F v = 0``, which survives
  addition but not the sequence of explicit Schur complements a filter needs.
* **The working condition number is the square root.** ``kappa(R) =
  sqrt(kappa(F))``, which is what keeps a thousand-epoch accumulation inside
  float64. Accumulating ``F`` directly and taking explicit Schur complements
  goes indefinite in float64 on a realistic near-degenerate campaign.
* **Accumulation is a QR.** Stacking two factors vertically and
  re-triangularising *is* the sum of the two quadratic forms, so
  order-invariance and associativity hold to roundoff by construction.

The class knows nothing about epochs, priors or pipelines. That is deliberate:
it keeps the numerics separable from the model machinery, as
:mod:`rheplicant.inference.conditioning` does for spectral diagnostics.
"""

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError


class SqrtInfo(eqx.Module):
    """``log L(x) = -0.5 ||R x - z||^2 + offset`` over a flat, named vector.

    Attributes:
        factor: ``(r, n)`` array ``R``. ``r < n`` means the term constrains
            only an ``r``-dimensional subspace -- the normal case for a single
            epoch, not an error.
        target: ``(r,)`` array ``z``.
        offset: scalar; every part of the log-density that does not depend on
            the latents -- the residual chi-square about the storage origin and
            the (masked) Gaussian normalisation.
        names: the latents this term is over, in the order they are ravelled.
        shapes: each latent's shape, in the same order. ``()`` for a scalar.
    """

    factor: jax.Array
    target: jax.Array
    offset: jax.Array
    names: tuple[str, ...] = eqx.field(static=True)
    shapes: tuple[tuple[int, ...], ...] = eqx.field(static=True)

    def __check_init__(self):
        if self.factor.ndim != 2:
            raise StateValidationError(
                f"SqrtInfo.factor must be 2-D (r, n); got shape {self.factor.shape}."
            )
        if self.target.shape != (self.factor.shape[0],):
            raise StateValidationError(
                f"SqrtInfo.target must have one entry per row of factor: factor is "
                f"{self.factor.shape}, target is {self.target.shape}."
            )
        if len(self.names) != len(self.shapes):
            raise StateValidationError(
                f"SqrtInfo has {len(self.names)} names but {len(self.shapes)} shapes."
            )
        if self.width != self.factor.shape[1]:
            raise StateValidationError(
                f"SqrtInfo.factor has {self.factor.shape[1]} columns but the named "
                f"latents {list(self.names)} ravel to {self.width} values."
            )

    @property
    def width(self) -> int:
        """Number of columns the named latents ravel to."""
        return sum(_size(shape) for shape in self.shapes)

    def ravel(self, values: dict[str, jax.Array]) -> jax.Array:
        """Flatten ``{name: array}`` into this term's column order."""
        parts = []
        for name, shape in zip(self.names, self.shapes, strict=True):
            if name not in values:
                raise StateValidationError(
                    f"This term is over {list(self.names)}; no value was given for "
                    f"{name!r}."
                )
            leaf = jnp.asarray(values[name])
            if leaf.shape != shape:
                raise StateValidationError(
                    f"Latent {name!r} has shape {shape} in this term but "
                    f"{leaf.shape} was supplied."
                )
            parts.append(jnp.ravel(leaf))
        return jnp.concatenate(parts) if parts else jnp.zeros(0)

    def log_prob(self, values: dict[str, jax.Array]) -> jax.Array:
        """The log-density this term encodes, at the given latent values."""
        resid = self.factor @ self.ravel(values) - self.target
        return self.offset - 0.5 * jnp.sum(resid**2)

    @classmethod
    def null(
        cls,
        names: tuple[str, ...],
        shapes: tuple[tuple[int, ...], ...],
    ) -> "SqrtInfo":
        """A term that says nothing -- the identity of :meth:`combine`.

        Square rather than zero-row so the accumulator's pytree keeps a fixed
        treedef across a whole campaign, which is what stops ``jit`` retracing
        once per epoch.
        """
        width = sum(_size(shape) for shape in shapes)
        return cls(
            factor=jnp.zeros((width, width)),
            target=jnp.zeros(width),
            offset=jnp.zeros(()),
            names=names,
            shapes=shapes,
        )

    @classmethod
    def combine(cls, first: "SqrtInfo", second: "SqrtInfo") -> "SqrtInfo":
        """The term whose log-density is the sum of the two given ones.

        Stack the augmented factors and re-triangularise. Writing ``y = [x; -1]``
        so that ``[R | z] y = R x - z``, the stacked product has the same norm
        as its triangular factor because ``Q`` has orthonormal columns::

            ||R_a x - z_a||^2 + ||R_b x - z_b||^2 = ||R_tot x - z_tot||^2 + rho^2

        ``rho`` is the corner of the triangular factor -- the part of the two
        residuals that no single quadratic form in ``x`` can express. It is a
        constant, so it belongs in the offset; dropping it leaves every
        combined term wrong by an amount that grows with the campaign and is
        invisible in the posterior's shape.
        """
        if first.names != second.names or first.shapes != second.shapes:
            raise StateValidationError(
                "Cannot combine two terms over different latents: "
                f"{list(first.names)} vs {list(second.names)}. A ledger of terms "
                "declared against different parameter sets is not a likelihood."
            )
        width = first.factor.shape[1]
        stacked = jnp.concatenate(
            [
                jnp.concatenate([first.factor, first.target[:, None]], axis=1),
                jnp.concatenate([second.factor, second.target[:, None]], axis=1),
            ],
            axis=0,
        )
        upper = jnp.linalg.qr(stacked, mode="r")
        keep = min(upper.shape[0], width)
        corner = upper[keep:, width]
        return cls(
            factor=upper[:keep, :width],
            target=upper[:keep, width],
            offset=first.offset + second.offset - 0.5 * jnp.sum(corner**2),
            names=first.names,
            shapes=first.shapes,
        )

    def fisher(self) -> jax.Array:
        """``F = R^T R`` -- the Fisher information this term carries.

        May legitimately be singular: a single epoch usually constrains only a
        subspace, and only the campaign total plus the prior need be positive
        definite.
        """
        return self.factor.T @ self.factor


def _size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size
