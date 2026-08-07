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


def marginalise_arrays(
    factor: jax.Array,
    target: jax.Array,
    offset: jax.Array,
    n_block: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    """The Schur complement in square-root form, with no Python control flow.

    The block is the **leading** ``n_block`` columns; permuting is the caller's
    job, because a caller that already knows its layout should not pay for a
    name lookup once per epoch inside a ``lax.scan``.

    This exists because :func:`marginalise` cannot be traced. It concretises
    twice -- ``float(jnp.max(...))`` for the pivot scale and
    ``np.finfo(...)`` on a materialised dtype for the floor -- and the chain
    filter evaluates this arithmetic **inside** the theta likelihood, under
    ``jax.lax.scan``, differentiating it with respect to the transition's own
    parameters. Measured on the checked path, on a ``(6, 4)`` term::

        eager  0.2497233081987414
        jit    ConcretizationTypeError
        grad   ConcretizationTypeError

    ``grad`` is the half that matters. A correlation time that is inferred
    rather than pinned at compression time is differentiated on every leapfrog
    step, so a ``marginalise`` that could be jitted but not differentiated would
    still be unusable here.

    **The refusal is not weakened, it is moved -- and this function cannot make
    it.** What :func:`marginalise` catches is a block that does not constrain
    itself, which makes the integral divergent. Under a trace that judgement is
    unavailable: it needs a comparison against a value, and there is no value.
    What that costs, measured on one term with its block column rescaled:

    ========================  ===============  ==================
    block column              this function    :func:`marginalise`
    ========================  ===============  ==================
    healthy (scale 1)         +0.203 nats      accepted
    rounding-scale (1e-10)    **+23.23 nats**  refused
    identically zero          ``+inf``         refused
    scaled by ``nan``         ``nan``          refused
    scaled by ``inf``         ``nan``          refused
    ========================  ===============  ==================

    The zero column is the easy half: ``-log|pivot|`` is a true ``+inf`` and any
    finiteness check downstream catches it. The dangerous half is the
    rounding-scale row, which is what a genuinely near-degenerate night looks
    like -- finite, the right sign, the right order of magnitude for a good
    night's evidence, growing as ``-log(pivot)`` (27.8 at 1e-12) and unbounded
    in principle. Nothing downstream tests for that.

    **The last two rows are new, and the third column used to read "accepted".**
    The table stopped at the zero column and so did the checked path's own test;
    the threshold there is relative to ``max(pivots)``, so one ``nan`` anywhere
    in the term made the threshold ``nan`` and every comparison against it
    False. Both are pinned by
    ``test_the_kernel_cannot_see_what_the_checked_path_refuses``, which reads
    ``pivots`` at all five scales, and by
    ``test_a_poisoned_block_is_refused_rather_than_marginalised_to_nan``.

    What it hands back instead is the evidence: ``pivots``, as data. An eager
    caller judges them (:func:`marginalise` does). A chain does not need to,
    because :class:`~rheplicant.inference.chain.LinearGaussianTransition`
    refuses a non-positive ``process_std`` or ``initial_std`` at construction
    and those rows are what constrain every ``zeta_e`` -- one eager check at
    declaration instead of one traced check per epoch. A caller that has
    neither owns the gap, and should look at ``pivots`` itself.

    Args:
        factor: ``(r, n)`` array ``R``, block columns first.
        target: ``(r,)`` array ``z``.
        offset: scalar; the constant this term already carries.
        n_block: how many leading columns to integrate out. ``0`` is legal and
            is the identity on the density -- the re-triangularisation folds any
            excess rows into the corner and the offset absorbs it.

    Returns:
        ``(factor, target, offset, pivots)`` -- the retained form, the offset
        with the Gaussian integral's constant folded in, and ``|diag(R)|`` of
        the re-triangularisation so a checked caller can test it.
    """
    width = factor.shape[1]
    upper = jnp.linalg.qr(jnp.concatenate([factor, target[:, None]], axis=1), mode="r")
    keep = min(upper.shape[0], width)
    # The part of the residual no quadratic form in the retained columns can
    # express. A constant, so it belongs in the offset.
    corner = upper[keep:, width]
    pivots = jnp.abs(jnp.diag(upper))
    constant = (
        0.5 * n_block * jnp.log(2.0 * jnp.pi)
        - jnp.sum(jnp.log(pivots[:n_block]))
        - 0.5 * jnp.sum(corner**2)
    )
    return (
        upper[n_block:keep, n_block:width],
        upper[n_block:keep, width],
        offset + constant,
        pivots,
    )


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

    Written once here rather than at each caller because the chain filter
    marginalises ``zeta_e`` by the same three lines, and two copies of a
    constant is how one of them gets fixed.

    The arithmetic itself lives in :func:`marginalise_arrays`, which takes arrays
    and can be traced; this function is that call plus the two checks, which
    cannot -- **calling this one under** ``jit`` **or** ``grad`` **raises**
    ``ConcretizationTypeError``, and :func:`marginalise_arrays` is what to reach
    for there. The chain filter calls the kernel directly from inside a
    ``lax.scan``. There is still exactly one copy of the constant, and
    ``test_the_kernel_and_the_checked_path_return_the_same_numbers`` compares the
    two paths element-wise rather than trusting this sentence.

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
        StateValidationError: if a name is repeated or not in ``info``; if the
            term's re-triangularisation is not finite -- the degeneracy test
            below is relative to the largest pivot, so one ``nan`` or ``inf``
            anywhere in the term used to make the threshold ``nan`` and the
            answer "accepted", returning a ``SqrtInfo`` whose offset was ``nan``
            past a ``__check_init__`` that validates shapes only; or if the
            block is not constrained -- an unconstrained direction makes the
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
    permuted = info.factor[:, jnp.asarray(columns, dtype=int)]

    if n_block:
        # A shape fact, available before any arithmetic: the re-triangularised
        # factor has min(rows, width + 1) rows -- width + 1, not width, because
        # the target column is part of the matrix being factorised -- and fewer
        # than n_block of them means the block cannot constrain its own
        # directions. Writing `width` here would refuse a term with exactly
        # `width` independent rows and a full-rank block, which is the normal
        # case for a square padded epoch block.
        if min(permuted.shape[0], width + 1) < n_block:
            raise StateValidationError(
                f"This term has {min(permuted.shape[0], width + 1)} independent "
                f"rows but {n_block} columns in the block {list(block)}, so the "
                "block does not constrain itself and the integral over it "
                "diverges. A per-epoch latent is integrated exactly once, which "
                "is why condition C3 requires its prior to be part of the model "
                "rather than an optional regulariser: append the prior rows "
                "before marginalising."
            )

    factor, target, offset, pivots = marginalise_arrays(
        permuted, info.target, info.offset, n_block
    )

    if n_block:
        # Finiteness FIRST, and it is not defensive padding: the comparison
        # below is relative to `max(pivots)`, so a single `nan` or `inf` anywhere
        # in this term makes the threshold itself `nan` and every comparison
        # against it False. Measured on `_block_scaled`, which is the term this
        # module's own tests use: `scale=0` refused, `scale=nan` ACCEPTED with
        # offset `nan`, `scale=inf` ACCEPTED with offset `nan` -- and
        # `SqrtInfo.__check_init__` checks shapes only, so nothing between here
        # and a campaign total would have said a word.
        #
        # BOTH ends, because a guard written NaN-safely can still be defeated
        # from the other one: `inf > 0` is True, so a threshold of `inf` admits
        # every pivot there is.
        # `bool(jnp.all(...))` rather than `np.all(np.asarray(...))`: both
        # concretise, and only the first raises `ConcretizationTypeError` under
        # a trace. `test_the_checked_path_refuses_under_a_trace_rather_than_
        # skipping_its_guard` pins that error by name, and a guard added in
        # front of the others must not change which one a traced caller sees.
        if not bool(jnp.all(jnp.isfinite(pivots))):
            raise StateValidationError(
                f"This term's re-triangularisation is not finite: its pivots are "
                f"{np.asarray(pivots)}, so the marginal over {list(block)} would "
                "come back as a SqrtInfo carrying nan -- which __check_init__ "
                "does not test for, and which loses every comparison a campaign "
                "audit could make about it. The stored factor or target already "
                "carried nan or inf before this call. Recompress that epoch."
            )
        # Compared against the LARGEST pivot, not against an absolute floor: the
        # rows are whitened data, so their scale is the epoch's 1/sigma and an
        # absolute threshold would refuse a well-constrained low-noise block and
        # wave through a badly-constrained high-noise one.
        #
        # These lines are why this function cannot be traced, and why the
        # arithmetic above it can.
        scale = float(jnp.max(pivots))
        floor = float(np.sqrt(np.finfo(permuted.dtype).eps)) * scale
        # `not all(> floor)` rather than `any(<= floor)`: the two are the same
        # for finite numbers and not for `nan`, which loses both comparisons.
        # The finiteness check above makes that unreachable today; it is written
        # this way anyway, because it is the shape three other refusals in this
        # subsystem needed and the fourth is how one of them gets written the
        # weak way again.
        if not bool(jnp.all(pivots[:n_block] > floor)):
            raise StateValidationError(
                f"The block {list(block)} does not constrain one of its own "
                "directions, so the Gaussian integral over it diverges and the "
                "marginal would come back as +inf -- finite arithmetic gives a "
                "large plausible number instead, which is worse, because nothing "
                "downstream tests for it. Give the block a proper prior "
                "(condition C3) and append its rows before marginalising."
            )

    return SqrtInfo(
        factor=factor,
        target=target,
        offset=offset,
        names=kept,
        shapes=tuple(info.shapes[info.names.index(name)] for name in kept),
    )


def _size(shape: tuple[int, ...]) -> int:
    size = 1
    for dim in shape:
        size *= int(dim)
    return size
