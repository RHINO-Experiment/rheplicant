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


def _size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size
