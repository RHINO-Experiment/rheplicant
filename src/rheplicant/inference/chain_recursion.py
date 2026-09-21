"""The recursion that integrates the drift model.

The forward marginal, its log-likelihood, and the backward smoother. These
have a different test shape from the model they integrate: a transition is
checked one step at a time against an analytic form, and a recursion is
checked over a whole chain against a dense solve.
"""

from typing import Any

import jax
import jax.numpy as jnp
from bayesmith.errors import StructureError as _FarStructureError
from bayesmith.marginal.chain import chain_marginal as _far_chain_marginal
from bayesmith.marginal.chain import smooth as _far_smooth

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.sqrtinfo import SqrtInfo


def _check_block_width(
    factors: jax.Array,
    names: tuple[str, ...],
    n_theta: int,
    n_zeta: int,
) -> int:
    """How wide the stored blocks are, checked against what is claimed of them.

    One copy, called by the filter and by the smoother, because both of them
    slice at ``n_theta`` to separate theta's columns from the chain's: a block
    of the wrong width makes that slice take one for the other, and what comes
    back is finite, plausible, and a quantity nothing downstream re-derives.
    """
    width = n_theta + n_zeta
    if factors.shape[-1] != width:
        raise StateValidationError(
            f"The stored blocks are {factors.shape[-1]} columns wide but "
            f"{list(names)} plus a width-{n_zeta} chain is {width}. The blocks are "
            "a quadratic form in a specific ordered vector; reading them against a "
            "different one is not a rename, it is a different model."
        )
    return width


def chain_marginal(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> SqrtInfo:
    """``zeta_1:N`` integrated out exactly, leaving a quadratic form in theta.

    Args:
        blocks: ``(factor (N, w, w), target (N, w), offset (N,))`` -- one square
            per-epoch joint form over ``(*names, zeta)``, ``zeta``'s columns
            **last**. Square rather than ragged because ``lax.scan`` needs one
            shape per iteration; ``SqrtInfo.combine(SqrtInfo.null(...), info)``
            is the padding, and it is the same QR the accumulator uses, so the
            offset it produces is the one this consumes, corner included.
        transition: a :class:`LinearGaussianTransition` or a
            :class:`HyperTransition`. Resolved once, here, against ``values`` --
            which is what makes an inferred correlation time inferred rather
            than pinned at compression time.
        values: the global latents. Read for the transition's hyperparameters
            and for the returned form's evaluation point.
        names: the global latents, in the same column order the blocks were
            built in.
        shapes: each of those latents' shapes, in the same order.

    Returns:
        A :class:`~rheplicant.inference.sqrtinfo.SqrtInfo` over ``names`` whose
        ``log_prob`` is ``log p(d_1:N | theta)``. Prior-free, like every other
        stored factor in this layer.

    **Why the scan stops one short, and what that is *not* about.** The plan this
    was built from warned that scanning over all ``N`` blocks and then
    marginalising once more would integrate a ``zeta_{N+1}`` no data constrained
    and come back "finite and wrong by one transition's worth of constants".
    Measured, by writing it that way: it comes back **exact**, 4.5e-13 from the
    dense oracle at all four probes -- because the extra transition density
    integrates to one over its own argument, so its normalisation and the extra
    marginalisation's constant cancel term for term. The reason to stop one short
    is therefore cost, one QR per call, not correctness. What the extra step does
    change is the *count* of each constant, five transitions becoming six and six
    marginalisations becoming seven, which is what
    ``test_the_transition_normalisation_is_the_whole_density_not_half_of_it`` and
    ``test_the_marginalisation_constant_is_carried`` notice and the exactness
    tests cannot.
    """
    try:
        far = _far_chain_marginal(blocks, transition, values, names, shapes)
    except _FarStructureError as exc:
        # `_check_block_width` is the only refusal on this path, and the two
        # sides' messages are byte-identical (231 characters, measured), so
        # preserving `str(exc)` reproduces this package's text exactly. Only
        # the class is translated: `StructureError` and `StateValidationError`
        # share nothing below `ValueError`, and this module's callers pin
        # `StateValidationError` by name in twenty places.
        raise StateValidationError(str(exc)) from None
    # Re-wrapped rather than returned: the far container is a different class
    # with the same five fields, and it spells the curvature `information()`
    # where this package spells it `fisher()` -- `ChainMemory.fisher()` would
    # break on the far object. The five fields transfer bitwise (measured 0.0
    # on factor, target and offset at all four `chain_bank.PROBES`, on the
    # scalar bank and on wide chains of width 2 and 3), and near's
    # `__check_init__` re-runs on the way in, which is where this package's
    # own refusals -- the complex-value one among them -- get their say.
    return SqrtInfo(
        factor=far.factor,
        target=far.target,
        offset=far.offset,
        names=far.names,
        shapes=far.shapes,
    )


def chain_log_likelihood(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> jax.Array:
    """``log p(d_1:N | theta)``, the chain integrated out exactly. No prior."""
    return chain_marginal(blocks, transition, values, names, shapes).log_prob(values)


def _zeta_joint(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[jax.Array, jax.Array, int, int]:
    """The block-tridiagonal joint over ``zeta_1:N``, assembled and triangularised.

    Three kinds of row go in and nothing else: each epoch's stored rows with
    ``theta`` moved to the right-hand side, ``zeta_1``'s prior, and one coupling
    ``Q^-1/2 (zeta_{e+1} - phi zeta_e)`` per transition. The offsets are not
    read -- a constant cannot move a mean or a covariance -- which is why this
    returns no offset and :func:`smooth` reports no density.

    The QR always has at least ``T + 1`` rows to work with: the assembled matrix
    has ``N n_theta + 2 N n_zeta`` of them against ``T = N n_zeta`` columns, so
    the slices below never under-run, for any ``N >= 1``.

    Returns:
        ``(triangular (T, T), rhs (T,), n_epochs, n_zeta)``.
    """
    factors, targets, _ = blocks
    resolved = transition.at(values)
    n_zeta = resolved.width
    n_theta = sum(int(jnp.zeros(shape).size) for shape in shapes)
    _check_block_width(factors, names, n_theta, n_zeta)
    n_epochs = int(factors.shape[0])
    total = n_epochs * n_zeta
    theta = (
        jnp.concatenate([jnp.ravel(jnp.asarray(values[name])) for name in names])
        if names
        else jnp.zeros(0)
    )

    rows, rhs = [], []
    # Each epoch's evidence. theta is CONDITIONED on, not marginalised: it moves
    # to the right-hand side rather than becoming more columns.
    for e in range(n_epochs):
        rows.append(
            jnp.zeros((factors.shape[1], total))
            .at[:, e * n_zeta : (e + 1) * n_zeta]
            .set(factors[e][:, n_theta:])
        )
        rhs.append(targets[e] - factors[e][:, :n_theta] @ theta)

    # zeta_1's prior.
    rows.append(jnp.zeros((n_zeta, total)).at[:, :n_zeta].set(jnp.diag(1.0 / resolved.initial_std)))
    rhs.append(resolved.initial_mean / resolved.initial_std)

    # The couplings. `diag(1/q) @ phi`, not `phi @ diag(1/q)` -- the same line
    # the filter's transition rows are built from, and the same one that no
    # scalar and no equal-spread fixture can tell apart.
    inverse_process = 1.0 / resolved.process_std
    for e in range(n_epochs - 1):
        coupling = jnp.zeros((n_zeta, total))
        coupling = coupling.at[:, e * n_zeta : (e + 1) * n_zeta].set(
            -inverse_process[:, None] * resolved.phi
        )
        coupling = coupling.at[:, (e + 1) * n_zeta : (e + 2) * n_zeta].set(
            jnp.diag(inverse_process)
        )
        rows.append(coupling)
        rhs.append(jnp.zeros(n_zeta))

    upper = jnp.linalg.qr(
        jnp.concatenate([jnp.concatenate(rows, axis=0), jnp.concatenate(rhs)[:, None]], axis=1),
        mode="r",
    )
    return upper[:total, :total], upper[:total, total], n_epochs, n_zeta


def _joint_covariance(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> jax.Array:
    """The WHOLE ``cov(zeta_1:N | d_1:N, theta)``, cross-epoch blocks included.

    Private, and :func:`smooth` returns only its diagonal, for the reason that
    function's docstring gives. It exists because the diagonal is the weaker
    claim: a joint solve that coupled the epochs in the wrong direction, or
    coupled them twice, can reproduce every variance and still get
    ``cov(zeta_2, zeta_5)`` wrong, and no per-epoch diagnostic would notice.
    ``tests/evidence/test_chain_smoother.py`` pins the full matrix against the
    dense oracle's, at both chain widths.
    """
    triangular, _, n_epochs, n_zeta = _zeta_joint(blocks, transition, values, names, shapes)
    inverse = jax.scipy.linalg.solve_triangular(triangular, jnp.eye(n_epochs * n_zeta), lower=False)
    return inverse @ inverse.T


def smooth(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> tuple[jax.Array, jax.Array]:
    """``p(zeta_e | d_1:N, theta)`` for every epoch -- mean and variance.

    ``theta`` is **conditioned on**, not marginalised: the question a smoother
    answers is "given this receiver model, what did the drift do?", and
    marginalising theta would answer a different one with the same shapes.

    **Delegated, and it was the last of the six G3 names to go.** The far
    implementation inverted the explicit precision -- paying ``kappa(F)`` where
    a square root pays ``sqrt(kappa(F))`` -- and on a frozen chain returned
    exactly twice the right answer and then ``nan``. That is fixed upstream as
    of ``bayesmith`` 0.6.0, which assembles the information square root the way
    this package always did; measured after the repair, the two agree
    **bitwise** at every stiffness from 1e-1 down to 1e-9.
    ``tests/evidence/test_chain_smoother_conditioning.py`` is what would notice
    a regression, and it asserts properties -- positive variances, and the
    limit a freezing chain must reach -- rather than comparing the two sides,
    so it keeps its teeth now that only one implementation is left.

    ``_zeta_joint`` stays because ``_joint_covariance`` still uses it
    and the far side has no counterpart to that. Both are checked against
    ``chain_bank``'s dense oracle rather than against each other, so the
    remaining near-side copy is a reference the tests hold, not a second
    answer in the shipping path.

    **How, and why not the classical backward pass.** Section 6 names an RTS
    smoother. What this computes is the same quantity -- the exact smoothed
    marginals -- by assembling the block-tridiagonal joint information form over
    ``zeta_1:N`` (each epoch's stored rows with theta substituted, the initial
    prior, and the transition couplings) and triangularising it once. The
    arithmetic is then :class:`~rheplicant.inference.sqrtinfo.SqrtInfo`'s and the
    transition rows are the filter's, so there is one implementation of the
    algebra rather than two; the failure mode of two is that one of them gets
    fixed. It is an offline diagnostic and not on the sampling path, so what
    this costs buys the absence of a second numerical route.

    **What it costs is ``O((N n_zeta)^2)``, not ``O(N n_zeta^2)``.** The
    variances are the row norms of ``R^-1``, and ``R^-1`` is a dense
    ``T``-by-``T`` triangular solve however sparse ``R`` is -- 8 MB of float64
    at a thousand epochs of a scalar chain, 128 MB at four thousand. Affordable
    offline and not on the filter's ``O(1)`` carry, which is the whole reason
    these are two functions.

    **The covariance does not depend on ``theta``, and the mean does.** For a
    linear-Gaussian chain the posterior spread of the drift is a property of the
    designs, the noise and the transition alone, so a test that pinned only the
    covariance would be blind to every error in how ``theta`` is substituted --
    which is why ``tests/evidence/test_chain_smoother.py`` pins the mean at four
    probes and the full covariance once.

    Returns:
        ``(mean (N, n_zeta), variance (N, n_zeta))``. Variances rather than full
        per-epoch covariances because that is what the diagnostics read and
        because a full one invites the reader to believe the cross-epoch blocks
        are in there; they are not returned, though the joint form has them and
        ``_joint_covariance`` is where the tests get at them.
    """
    try:
        return _far_smooth(blocks, transition, values, names, shapes)
    except _FarStructureError as exc:
        # `_check_block_width` is the only refusal on this path and the two
        # sides' messages are byte-identical, so preserving `str(exc)`
        # reproduces this package's text exactly. Only the class translates.
        raise StateValidationError(str(exc)) from None
