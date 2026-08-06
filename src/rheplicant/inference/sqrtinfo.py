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

from collections.abc import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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


def marginalise(info: SqrtInfo, block: Sequence[str]) -> SqrtInfo:
    """Integrate named latents out of a square-root information form, exactly.

    Permute the block's columns first, re-triangularise, and drop the leading
    rows and columns. That drop **is** the Schur complement, and the Gaussian
    integral over the block contributes exactly

    ``+ (n_block/2) log(2 pi)  -  sum log|R_bb,ii|  -  0.5 rho^2``

    and nothing else. In particular it does **not** contribute the block's own
    prior normalisation: whoever appended the prior rows owns
    ``-sum(log(std)) - (n/2) log(2 pi)``, and the two ``2 pi`` halves cancel
    while ``sum(log(std))`` has nothing to cancel against. Plan A shipped that
    second term missing -- 1.07 nats for three nuisances at ``std=0.7``, 27.47
    for twenty-five at ``std=3``, and **exactly zero at** ``std=1``, which is how
    a probe built on unit priors passed. A constant is invisible in a
    posterior's shape, so the tests for this function compare absolute
    log-densities against a dense oracle and use a non-unit prior.

    Written once here rather than at each caller because section 6's chain
    filter marginalises ``zeta_e`` by the same three lines, and two copies of a
    constant is how one of them gets fixed.

    Marginalising **every** name is legal and returns a zero-width term --
    ``factor`` shape ``(0, 0)``, ``names=()`` -- whose ``log_prob({})`` is the
    marginal likelihood. That is what T1 does at each ``theta`` when it
    integrates out ``phi_e``, so refusing it would mean writing the nuisance and
    no-nuisance paths twice. Marginalising **nothing** is likewise legal and is
    the identity on the log-density: the re-triangularisation folds any excess
    rows into ``rho`` and the offset absorbs it, which is
    :meth:`SqrtInfo.combine`'s arithmetic with one term.

    Args:
        info: the joint form, prior rows already appended by the caller.
        block: which names to integrate out.

    Returns:
        A term over ``info``'s remaining names, in their original relative
        order, whose log-density is the integral of ``info``'s over ``block``.

    Raises:
        StateValidationError: if a name is repeated or not in ``info``, or if
            the block is not constrained -- an unconstrained direction makes the
            integral divergent, and finite arithmetic returns a large plausible
            number for it rather than an infinity anyone would notice.
    """
    block = tuple(block)
    if len(set(block)) != len(block):
        raise StateValidationError(
            f"marginalise was given {list(block)}, which names a latent twice. "
            "Integrating the same block out twice is not defined. Pass each name "
            "once."
        )
    unknown = [name for name in block if name not in info.names]
    if unknown:
        raise StateValidationError(
            f"This term is not over {unknown}; it is over {list(info.names)}. "
            "Marginalise a name the term actually carries, or combine the terms "
            "that do carry it first."
        )

    spans: dict[str, range] = {}
    position = 0
    for name, shape in zip(info.names, info.shapes, strict=True):
        size = _size(shape)
        spans[name] = range(position, position + size)
        position += size
    kept = tuple(name for name in info.names if name not in block)
    columns = [column for name in block for column in spans[name]]
    columns += [column for name in kept for column in spans[name]]
    n_block = sum(len(spans[name]) for name in block)
    width = info.factor.shape[1]

    upper = jnp.linalg.qr(
        jnp.concatenate(
            [info.factor[:, jnp.asarray(columns, dtype=int)], info.target[:, None]],
            axis=1,
        ),
        mode="r",
    )
    if n_block:
        if upper.shape[0] < n_block:
            raise StateValidationError(
                f"This term has {upper.shape[0]} independent rows but {n_block} "
                f"columns in the block {list(block)}, so the block does not "
                "constrain itself and the integral over it diverges. A per-epoch "
                "latent is integrated exactly once, which is why condition C3 "
                "requires its prior to be part of the model rather than an "
                "optional regulariser: append the prior rows before marginalising."
            )
        pivots = jnp.abs(jnp.diag(upper)[:n_block])
        # Compared against the LARGEST pivot, not against an absolute floor: the
        # rows are whitened data, so their scale is the epoch's 1/sigma and an
        # absolute threshold would refuse a well-constrained low-noise block and
        # wave through a badly-constrained high-noise one.
        scale = float(jnp.max(jnp.abs(jnp.diag(upper))))
        floor = float(np.sqrt(np.finfo(np.asarray(upper).dtype).eps)) * scale
        if bool(jnp.any(pivots <= floor)):
            raise StateValidationError(
                f"The block {list(block)} does not constrain one of its own "
                "directions, so the Gaussian integral over it diverges and the "
                "marginal would come back as +inf -- finite arithmetic gives a "
                "large plausible number instead, which is worse, because nothing "
                "downstream tests for it. Give the block a proper prior "
                "(condition C3) and append its rows before marginalising."
            )
        log_pivots = jnp.sum(jnp.log(pivots))
    else:
        log_pivots = jnp.zeros(())

    keep = min(upper.shape[0], width)
    # The part of the residual no quadratic form in the retained columns can
    # express. A constant, so it belongs in the offset; dropped, every term is
    # wrong by an amount that grows with the campaign and changes no gradient.
    corner = upper[keep:, width]
    constant = (
        0.5 * n_block * jnp.log(2.0 * jnp.pi) - log_pivots - 0.5 * jnp.sum(corner**2)
    )
    return SqrtInfo(
        factor=upper[n_block:keep, n_block:width],
        target=upper[n_block:keep, width],
        offset=info.offset + constant,
        names=kept,
        shapes=tuple(info.shapes[info.names.index(name)] for name in kept),
    )


def _size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size
