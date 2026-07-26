"""Declared-linear parameter blocks: check the claim, then export the operator.

Some parameters enter the forward model **linearly** — sky ``alm``
coefficients, noise-wave amplitudes, any component whose contribution is a
matrix acting on it. Those blocks are also the big ones: a sky at ``lmax=191``
across 32 channels is ~10⁶ real degrees of freedom, where gradient-based
samplers are hopeless but a conjugate-Gaussian solve is exactly right.

Declaring ``Latent(..., linear=True)`` promises that, holding every other
latent fixed, the prediction is an **affine** function of this one::

    prediction(x) = A x + b

Two things follow. First, the promise is *checkable*, and this module checks
it before anything exploits it — :func:`check_linearity` compares the model
against its own linearization. A false declaration would otherwise produce a
confident, wrong posterior instead of an error.

Second, ``A`` and ``Aᵀ`` are available without ever forming a matrix:
``jax.linearize`` gives the forward action and ``jax.vjp`` the adjoint, at the
cost of one trace. :func:`linear_operator` packages them as a
:class:`LinearBlock`, which is the whole interface a conjugate-Gaussian
sampler needs.

**Probe at extreme scales.** :func:`check_linearity` probes at 10⁻³, 1 and 10³
times the latent's own magnitude, because near-linearity is scale-dependent:
``x + εx²`` is indistinguishable from linear near the origin and grossly
nonlinear far from it. A probe suite that only samples "reasonable" values
signs off on exactly the blocks that will fail in a sampler's tails.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.parameters import ParameterSpace

DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1.0, 1e3)


@dataclasses.dataclass(frozen=True)
class LinearBlock:
    """The affine action of one latent on the prediction: ``A x + offset``.

    Deliberately a plain dataclass rather than an ``eqx.Module``: this is a
    derived linear-algebra *handle*, not a differentiable model. ``forward``
    and ``adjoint`` are closures over a traced computation, so the block is
    something you build where you need it, not a pytree to carry around.

    Attributes:
        name: the latent this block belongs to.
        shape: shape of ``x``.
        dtype: dtype of ``x``.
        offset: ``prediction(0)`` — everything the other parameters contribute.
        forward: ``x -> A x``, from ``jax.linearize``.
        adjoint: ``y -> Aᵀ y``, from ``jax.vjp``, shaped like ``x``.

    Adjoint convention, which matters as soon as ``x`` is complex (sky
    ``alm`` coefficients are): ``adjoint`` is exactly ``jax.vjp``, and JAX
    returns the *conjugate* gradient for complex inputs. The identity that
    holds is therefore the one over the **real** inner product::

        Re sum(x * adjoint(y))  ==  sum(forward(x) * y)

    and NOT the sesquilinear ``sum(conj(x) * adjoint(y))``. The real pairing is
    the one a Gaussian likelihood forms, so this is the useful convention as
    well as the honest one; ``tests/inference/test_linear_blocks.py`` pins both
    halves so the distinction cannot rot into a silent factor.
    """

    name: str
    shape: tuple[int, ...]
    dtype: Any
    offset: jax.Array
    forward: Callable[[jax.Array], jax.Array]
    adjoint: Callable[[jax.Array], jax.Array]


def _resolve_name(space: ParameterSpace, name: str | None) -> str:
    """Pick the linear latent to work on, insisting the declaration exists."""
    if name is None:
        declared = [latent.name for latent in space.latents if latent.linear]
        if not declared:
            raise ParameterSpaceError(
                "No latent in this space is declared linear. A block is only usable as a "
                "linear operator once you assert it — declare it with linear=True, and the "
                "assertion will be checked."
            )
        if len(declared) > 1:
            raise ParameterSpaceError(
                f"This space has several linear latents {declared}; say which latent you mean "
                "by passing name=."
            )
        return declared[0]
    if not space.latent(name).linear:
        raise ParameterSpaceError(
            f"Latent {name!r} is not declared linear=True, so its linear operator is not "
            "meaningful. Declare it, and the claim will be checked."
        )
    return name


def _isolate(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str,
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
    """``g(x) = prediction with latent `name` set to x``, plus a zero of its shape."""
    forward, values0 = space.forward_fn(pipeline, state_template)
    latent = space.latent(name)

    def g(x: jax.Array) -> jax.Array:
        return forward({**values0, name: x})

    return g, jnp.zeros_like(latent.init)


def check_linearity(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
    key: jax.Array | None = None,
) -> dict[float, float]:
    """Verify that the prediction really is affine in one latent.

    Compares the model against its own linearization at zero, at several
    magnitudes of probe. Costs one linearization plus one forward evaluation
    per scale.

    Args:
        space, pipeline, state_template: the model under test.
        name: which latent. Optional when exactly one is declared linear.
        scales: probe magnitudes, as multiples of the latent's own scale.
            The default spans six orders of magnitude on purpose — see the
            module docstring.
        rtol: tolerance on the relative departure from affinity. Default:
            ``1e4 * eps`` of the prediction dtype, which leaves room for
            accumulated roundoff in a long reduction without admitting real
            curvature.
        key: PRNG key for the probes. Fixed by default, so the check is
            reproducible.

    Returns:
        ``{scale: relative error}`` — useful for reporting how linear a block
        is, not only whether it passes.

    Raises:
        ParameterSpaceError: if any scale departs from affinity by more
            than ``rtol``.
    """
    name = _resolve_name(space, name)
    g, zero = _isolate(space, pipeline, state_template, name)
    latent = space.latent(name)
    if not jnp.issubdtype(latent.init.dtype, jnp.inexact):
        raise ParameterSpaceError(
            f"Latent {name!r} has dtype {latent.init.dtype}; a linear block must be "
            "floating-point or complex."
        )

    baseline, tangent = jax.linearize(g, zero)
    if rtol is None:
        rtol = 1e4 * float(jnp.finfo(baseline.dtype).eps)
    key = jax.random.key(0) if key is None else key

    magnitude = float(jnp.max(jnp.abs(latent.init)))
    if magnitude == 0.0:
        magnitude = 1.0

    # A departure smaller than the arithmetic's own noise floor is not evidence
    # of curvature. Without this the relative measure below explodes for SMALL
    # probes — where the variation is tiny but roundoff is not — and rejects
    # perfectly linear blocks. That false positive is worse than no check at
    # all, because the cure users reach for is to switch the check off.
    noise_floor = 1e4 * float(jnp.finfo(baseline.dtype).eps) * float(
        jnp.maximum(jnp.max(jnp.abs(baseline)), 1.0)
    )

    errors: dict[float, float] = {}
    departures: dict[float, float] = {}
    for index, scale in enumerate(scales):
        probe = magnitude * scale * jax.random.normal(
            jax.random.fold_in(key, index), latent.init.shape, dtype=latent.init.dtype
        )
        actual = g(probe)
        predicted = baseline + tangent(probe)
        # Measure against the VARIATION, not the total: a large constant offset
        # would otherwise hide a completely nonlinear response.
        variation = jnp.max(jnp.abs(actual - baseline))
        departure = jnp.max(jnp.abs(actual - predicted))
        departures[scale] = float(departure)
        errors[scale] = float(departure / jnp.maximum(variation, 1e-30))

    failed = {
        scale: err
        for scale, err in errors.items()
        if err > rtol and departures[scale] > noise_floor
    }
    if failed:
        detail = ", ".join(f"{scale:g}x -> {err:.2e}" for scale, err in errors.items())
        raise ParameterSpaceError(
            f"Latent {name!r} is declared linear=True, but the prediction is not affine in "
            f"it: departure from its own linearization exceeds rtol={rtol:.2e} (and the "
            f"{noise_floor:.2e} roundoff floor) at {sorted(failed)} times the latent's scale "
            f"({detail}). Either drop the declaration, or re-parameterize so the model really "
            "is linear in this block."
        )
    return errors


def linear_operator(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    check: bool = True,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
) -> LinearBlock:
    """Export ``A``, ``Aᵀ`` and the offset for a declared-linear latent.

    No matrix is ever formed: ``A`` comes from ``jax.linearize`` and ``Aᵀ``
    from ``jax.vjp``, so a 10⁶-dimensional block costs the same as one forward
    evaluation per application. That is what makes conjugate-Gaussian solves
    tractable here — see :func:`wiener_solve`.

    Args:
        space, pipeline, state_template: the model.
        name: which latent. Optional when exactly one is declared linear.
        check: verify the linearity claim first (:func:`check_linearity`).
            Leave it on. Turning it off costs three forward evaluations less
            and buys a class of silent, confident errors.
        scales, rtol: forwarded to :func:`check_linearity`.
    """
    name = _resolve_name(space, name)
    if check:
        check_linearity(space, pipeline, state_template, name, scales=scales, rtol=rtol)
    g, zero = _isolate(space, pipeline, state_template, name)
    latent = space.latent(name)

    offset, tangent = jax.linearize(g, zero)
    _, pullback = jax.vjp(g, zero)

    return LinearBlock(
        name=name,
        shape=latent.init.shape,
        dtype=latent.init.dtype,
        offset=offset,
        forward=tangent,
        adjoint=lambda y: pullback(y)[0],
    )


def _real_parts(block: LinearBlock) -> tuple[Callable, Callable]:
    """Convert between a latent and its real degrees of freedom.

    A complex latent is carried as ``(real, imag)``. This is not bookkeeping
    pedantry: ``prediction`` is real, so the map from complex coefficients to
    data is **ℝ-linear but not ℂ-linear**, and a Krylov method run over ℂ
    would be solving a different problem. Splitting makes the vector space the
    one the objective actually lives on.
    """
    is_complex = jnp.issubdtype(block.dtype, jnp.complexfloating)
    if is_complex:
        return (
            lambda x: (jnp.real(x), jnp.imag(x)),
            lambda parts: parts[0] + 1j * parts[1],
        )
    return (lambda x: x, lambda parts: parts)


def wiener_solve(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any,
    tol: float = 1e-6,
    maxiter: int | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Posterior mean of a linear-Gaussian block — the Wiener filter, by CG.

    With ``d = A x + offset + n``, ``n ~ N(0, N)`` and ``x ~ N(0, S)``::

        x̂ = (AᵀN⁻¹A + S⁻¹)⁻¹ AᵀN⁻¹ (d - offset)

    solved with conjugate gradients, so the normal operator is only ever
    *applied*, never formed. Each iteration costs one JVP and one VJP through
    the forward model — which is why a block with 10⁶ degrees of freedom is
    tractable at all.

    The normal operator and the right-hand side are both obtained as gradients
    of the objective itself rather than assembled from ``A`` and ``Aᵀ`` by
    hand. That is not a shortcut: it makes the operator symmetric positive
    definite *by construction* over the real degrees of freedom, with no
    adjoint-convention arithmetic left to get wrong for complex latents.

    This is the posterior **mean**, not a sample. Drawing constrained
    realizations adds a fluctuation term to the right-hand side; that sampler
    is the next thing this operator is for.

    Args:
        block: from :func:`linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise_std: noise standard deviation — scalar or broadcastable to the
            data.
        prior_std: prior standard deviation on the latent — scalar or
            broadcastable to it. Required: without a prior the normal operator
            can be singular, and CG would return a finite, arbitrary answer
            instead of complaining.
        tol: CG tolerance.
        maxiter: CG iteration cap. ``None`` lets JAX choose.

    Returns:
        ``(x̂, relative_residual)``. The residual is
        ``‖M x̂ - b‖ / ‖b‖`` over the real degrees of freedom — check it, since
        CG reports no convergence status of its own.
    """
    if prior_std is None:
        raise ParameterSpaceError(
            "wiener_solve needs prior_std: with no prior the normal operator "
            "AᵀN⁻¹A can be singular, and CG would return a finite, arbitrary answer rather "
            "than fail. Pass a large prior_std for an effectively flat prior."
        )
    if jnp.issubdtype(jnp.asarray(block.offset).dtype, jnp.complexfloating):
        raise ParameterSpaceError(
            "wiener_solve expects a real-valued prediction; this block's offset is complex."
        )

    split, join = _real_parts(block)
    weight = 1.0 / jnp.asarray(noise_std) ** 2
    inverse_prior = 1.0 / jnp.asarray(prior_std) ** 2
    residual_data = observed - block.offset
    zero = split(jnp.zeros(block.shape, dtype=block.dtype))

    def half_chi2(parts):
        return 0.5 * jnp.sum(weight * block.forward(join(parts)) ** 2)

    def cross_term(parts):
        return jnp.sum(weight * block.forward(join(parts)) * residual_data)

    normal_from_data = jax.grad(half_chi2)

    def normal(parts):
        curvature = normal_from_data(parts)
        return jax.tree.map(lambda c, p: c + inverse_prior * p, curvature, parts)

    rhs = jax.grad(cross_term)(zero)
    solution, _ = jax.scipy.sparse.linalg.cg(normal, rhs, tol=tol, maxiter=maxiter)

    def norm(parts):
        return jnp.sqrt(sum(jnp.sum(leaf**2) for leaf in jax.tree.leaves(parts)))

    misfit = jax.tree.map(lambda a, b: a - b, normal(solution), rhs)
    return join(solution), norm(misfit) / jnp.maximum(norm(rhs), 1e-30)
