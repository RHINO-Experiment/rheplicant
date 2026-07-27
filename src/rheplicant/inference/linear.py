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
:class:`LinearBlock`, which is the whole interface the conjugate-Gaussian
routines here need: :func:`wiener_solve` for the posterior mean and
:func:`gcr_sample` for an exact posterior draw.

Because the block is affine only *given* the other latents, both take ``at=``
to rebuild it wherever those currently are — which is what makes a Gibbs
sweep possible: draw the linear block exactly, update the nonlinear ones
however you like, repeat.

**Probe at extreme scales.** :func:`check_linearity` probes at 10⁻³, 1 and 10³
times the latent's own magnitude, because near-linearity is scale-dependent:
``x + εx²`` is indistinguishable from linear near the origin and grossly
nonlinear far from it. A probe suite that only samples "reasonable" values
signs off on exactly the blocks that will fail in a sampler's tails.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

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
    at: dict[str, jax.Array] | None = None,
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
    """``g(x) = prediction with latent `name` set to x``, plus a zero of its shape.

    ``at`` fixes the OTHER latents. A block is only linear *given* them, so a
    Gibbs sweep has to rebuild it wherever they currently are; without ``at``
    the block would silently keep describing the model at its declared starting
    point, which is right exactly once.
    """
    forward, values0 = space.forward_fn(pipeline, state_template)
    if at:
        unknown = [key for key in at if key not in space.names]
        if unknown:
            raise ParameterSpaceError(
                f"`at` names {unknown}, which is not a latent of this space; declared: "
                f"{list(space.names)}."
            )
        values0 = {**values0, **at}
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
    at: dict[str, jax.Array] | None = None,
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
        at: values for the OTHER latents. Linearity is a claim *given* them, so
            check it where the sampler will actually be. Defaults to the
            declared initial values.
        scales: probe magnitudes, as multiples of the latent's own scale,
            taken from ``max|init|``. The default spans six orders of
            magnitude on purpose — see the module docstring. NOTE: an
            all-zero ``init`` has no scale to take, so it falls back to 1.0
            and the probes become absolute. If your latent lives at 1e6 (sky
            alms in kelvin, say), give a representative ``init`` or pass
            ``scales`` explicitly — otherwise the sweep never reaches the
            regime the sampler will actually explore.
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
    g, zero = _isolate(space, pipeline, state_template, name, at)
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

    epsilon = float(jnp.finfo(baseline.dtype).eps)
    errors: dict[float, float] = {}
    verdicts: dict[float, bool] = {}
    for index, scale in enumerate(scales):
        probe = magnitude * scale * jax.random.normal(
            jax.random.fold_in(key, index), latent.init.shape, dtype=latent.init.dtype
        )
        actual = g(probe)
        predicted = baseline + tangent(probe)
        # Measure against the VARIATION, not the total: a large constant offset
        # would otherwise hide a completely nonlinear response.
        variation = float(jnp.max(jnp.abs(actual - baseline)))
        departure = float(jnp.max(jnp.abs(actual - predicted)))
        errors[scale] = departure / max(variation, 1e-300)

        # A departure smaller than the arithmetic's OWN noise floor is not
        # evidence of curvature; without this the relative measure explodes at
        # small probes, where the variation is vanishing but roundoff is not,
        # and rejects perfectly linear blocks. The floor is set by the magnitudes
        # actually being differenced AT THIS PROBE — not by a constant, and not
        # by the baseline alone. A constant floor would silently exempt every
        # model whose prediction is small in its own units, and a baseline-only
        # floor would let an unrelated bright component disable the check.
        floor = 1e4 * epsilon * max(
            float(jnp.max(jnp.abs(actual))), float(jnp.max(jnp.abs(baseline)))
        )
        # NaN must count as a FAILURE, not a pass: `nan > rtol` is False, so a
        # naive comparison treats an unusable probe as evidence of linearity.
        finite = np.isfinite(errors[scale]) and np.isfinite(departure)
        verdicts[scale] = (not finite) or (errors[scale] > rtol and departure > floor)

    failed = sorted(scale for scale, bad in verdicts.items() if bad)
    if failed:
        detail = ", ".join(f"{scale:g}x -> {err:.2e}" for scale, err in errors.items())
        raise ParameterSpaceError(
            f"Latent {name!r} is declared linear=True, but the prediction is not affine in "
            f"it: departure from its own linearization exceeds rtol={rtol:.2e} (above the "
            f"per-probe roundoff floor) at {failed} times the latent's scale "
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
    at: dict[str, jax.Array] | None = None,
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
        at: values for the OTHER latents, fixing where the block is built.
            Defaults to the declared initial values — right exactly once, so a
            Gibbs sweep must pass the current values here every sweep.
        check: verify the linearity claim first (:func:`check_linearity`).
            Leave it on. Turning it off costs three forward evaluations less
            and buys a class of silent, confident errors.
        scales, rtol: forwarded to :func:`check_linearity`.
    """
    name = _resolve_name(space, name)
    if check:
        check_linearity(space, pipeline, state_template, name, at=at,
                        scales=scales, rtol=rtol)
    g, zero = _isolate(space, pipeline, state_template, name, at)
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


def _check_solve_arguments(
    block: LinearBlock, observed: jax.Array, prior_std: Any, caller: str
) -> None:
    """Shared preconditions for the mean and the draw."""
    if jnp.shape(observed) != jnp.shape(block.offset):
        raise ParameterSpaceError(
            f"observed has shape {jnp.shape(observed)} but this block predicts "
            f"{jnp.shape(block.offset)}. Broadcasting these would solve a different "
            "problem and return a perfectly finite answer."
        )
    if prior_std is None:
        raise ParameterSpaceError(
            f"{caller} needs prior_std: with no prior the normal operator AᵀN⁻¹A can be "
            "singular, and CG would return a finite, arbitrary answer rather than fail. "
            "Pass a large prior_std for an effectively flat prior."
        )
    if jnp.issubdtype(jnp.asarray(block.offset).dtype, jnp.complexfloating):
        raise ParameterSpaceError(
            f"{caller} expects a real-valued prediction; this block's offset is complex."
        )


def wiener_solve(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any,
    prior_mean: Any = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = 1e-3,
) -> tuple[jax.Array, jax.Array]:
    """Posterior mean of a linear-Gaussian block — the Wiener filter, by CG.

    With ``d = A x + offset + n``, ``n ~ N(0, N)`` and ``x ~ N(m, S)``::

        x̂ = (AᵀN⁻¹A + S⁻¹)⁻¹ [AᵀN⁻¹ (d - offset) + S⁻¹m]

    solved with conjugate gradients, so the normal operator is only ever
    *applied*, never formed. Each iteration costs one JVP and one VJP through
    the forward model — which is why a block with 10⁶ degrees of freedom is
    tractable at all.

    The normal operator and the right-hand side are both obtained as gradients
    of the objective itself rather than assembled from ``A`` and ``Aᵀ`` by
    hand. That is not a shortcut: it makes the operator symmetric positive
    definite *by construction* over the real degrees of freedom, with no
    adjoint-convention arithmetic left to get wrong for complex latents.

    This is the posterior **mean**, not a sample. For a draw, see
    :func:`gcr_sample`, which adds a fluctuation term to this same right-hand
    side and costs exactly the same solve.

    Args:
        block: from :func:`linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise_std: noise standard deviation — scalar or broadcastable to the
            data.
        prior_std: prior standard deviation on the latent — scalar or
            broadcastable to it. Required: without a prior the normal operator
            can be singular, and CG would return a finite, arbitrary answer
            instead of complaining.
        prior_mean: centre of the prior. Defaults to zero, which is wrong for
            most physical quantities — a noise-wave temperature sits near
            250 K. Equivalent to an affine binding that adds the same offset,
            but says what it means.
        tol: CG tolerance.
        maxiter: CG iteration cap. ``None`` lets JAX choose.
        require_convergence: raise if the relative residual exceeds this.
            Defaults to ``1e-3``; ``None`` disables the guard and returns
            whatever CG produced. On by default because jax's ``cg`` reports no
            convergence status, so an unconverged solve otherwise comes back
            looking exactly like a converged one.

    Returns:
        ``(x̂, relative_residual)``, the residual being ``‖M x̂ - b‖ / ‖b‖``
        over the real degrees of freedom.
    """
    _check_solve_arguments(block, observed, prior_std, "wiener_solve")
    return _conjugate_solve(
        block, observed, noise_std=noise_std, prior_std=prior_std,
        prior_mean=prior_mean, tol=tol, maxiter=maxiter, key=None,
        require_convergence=require_convergence,
    )


def _norm(parts) -> jax.Array:
    """Euclidean norm, scaled so it survives float32.

    Squaring first overflows for entries beyond ~1.8e19, which turns the only
    convergence signal these solvers give into inf/inf = NaN exactly when the
    problem is badly scaled and the answer is most likely wrong.
    """
    leaves = [leaf for leaf in jax.tree.leaves(parts) if eqx.is_array(leaf)]
    if not leaves:  # pragma: no cover - defensive
        return jnp.array(0.0)
    biggest = jnp.max(jnp.stack([jnp.max(jnp.abs(leaf)) for leaf in leaves]))
    biggest = jnp.where(biggest > 0, biggest, 1.0)
    total = sum(jnp.sum((leaf / biggest) ** 2) for leaf in leaves)
    return biggest * jnp.sqrt(total)


def _conjugate_solve(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any,
    prior_mean: Any,
    tol: float,
    maxiter: int | None,
    key: jax.Array | None,
    require_convergence: float | None,
) -> tuple[jax.Array, jax.Array]:
    """Shared machinery for the posterior mean and for a posterior draw.

    Both solve ``(AᵀN⁻¹A + S⁻¹) x = b`` by CG over the latent's real degrees of
    freedom. They differ only in ``b``: the mean uses ``AᵀN⁻¹(d - offset)``,
    a draw adds the two fluctuation terms. ``key=None`` selects the mean.
    """
    split, join = _real_parts(block)
    weight = 1.0 / jnp.asarray(noise_std) ** 2
    prior_variance = jnp.asarray(prior_std) ** 2
    residual_data = observed - block.offset
    zero = split(jnp.zeros(block.shape, dtype=block.dtype))
    centre = split(
        jnp.zeros(block.shape, dtype=block.dtype)
        if prior_mean is None
        else jnp.broadcast_to(
            jnp.asarray(prior_mean, dtype=block.dtype), block.shape
        )
    )

    def half_chi2(parts):
        return 0.5 * jnp.sum(weight * block.forward(join(parts)) ** 2)

    def pair_with(vector):
        """``Aᵀ vector`` in real coordinates, as the gradient of a real pairing.

        Taking it as a gradient rather than calling ``block.adjoint`` is what
        keeps the real/complex conventions from ever entering: ``jax.grad`` of a
        real scalar is by construction the adjoint of the real inner product,
        which is the pairing every term here lives in.
        """
        return jax.grad(lambda parts: jnp.sum(block.forward(join(parts)) * vector))(zero)

    def normal(parts):
        curvature = jax.grad(half_chi2)(parts)
        return jax.tree.map(lambda c, p: c + p / prior_variance, curvature, parts)

    # S^-1 m: a zero-mean prior is wrong for most physical quantities (a
    # noise-wave temperature sits near 250 K, not near zero), and shifting the
    # prior is not the same act as shifting the model even though the two give
    # the same Gaussian.
    rhs = jax.tree.map(
        lambda base, m: base + m / prior_variance,
        pair_with(weight * residual_data),
        centre,
    )

    if key is not None:
        # Constrained realization: the two fluctuation terms whose covariances
        # sum to the normal operator itself, which is exactly why the solve
        # comes out distributed as the posterior rather than merely centred on
        # its mean.  b = AᵀN⁻¹(d-offset) + AᵀN⁻¹ᐟ²ω₁ + S⁻¹ᐟ²ω₂
        data_key, prior_key = jax.random.split(key)
        omega_data = jax.random.normal(
            data_key, jnp.shape(residual_data), dtype=jnp.result_type(residual_data)
        )
        omega_prior = jax.tree.map(
            lambda leaf, k: jax.random.normal(k, leaf.shape, dtype=leaf.dtype),
            zero,
            _split_like(prior_key, zero),
        )
        rhs = jax.tree.map(
            lambda base, from_data, from_prior: (
                base + from_data + from_prior / jnp.sqrt(prior_variance)
            ),
            rhs,
            pair_with(jnp.sqrt(weight) * omega_data),
            omega_prior,
        )

    solution, _ = jax.scipy.sparse.linalg.cg(normal, rhs, tol=tol, maxiter=maxiter)
    misfit = jax.tree.map(lambda a, b: a - b, normal(solution), rhs)
    residual = _norm(misfit) / jnp.maximum(_norm(rhs), 1e-30)
    if require_convergence is not None:
        # jax's cg reports no convergence status of its own, so an unconverged
        # solve otherwise comes back looking like any other answer. eqx.error_if
        # fires under jit, where a Python `if` on a traced value cannot.
        solution = eqx.error_if(
            solution,
            jnp.logical_or(~jnp.isfinite(residual), residual > require_convergence),
            "wiener_solve/gcr_sample did not converge: the relative residual exceeds "
            "require_convergence. Raise maxiter, loosen tol, or condition the problem "
            "(a stronger prior is usually the honest fix).",
        )
    return join(solution), residual


def _split_like(key: jax.Array, template) -> Any:
    """One independent key per leaf of ``template``, same structure."""
    leaves, treedef = jax.tree.flatten(template)
    return jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))


def gcr_sample(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any,
    key: jax.Array,
    prior_mean: Any = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = 1e-3,
) -> tuple[jax.Array, jax.Array]:
    """Draw an EXACT posterior sample of a linear-Gaussian block.

    The constrained-realization (GCR) identity: solve the same system
    :func:`wiener_solve` does, but with two white-noise terms added to the
    right-hand side::

        (AᵀN⁻¹A + S⁻¹) x = AᵀN⁻¹(d - offset) + S⁻¹m + AᵀN⁻¹ᐟ² ω₁ + S⁻¹ᐟ² ω₂

    with ``ω₁``, ``ω₂`` standard normal on the data and on the latent. The
    right-hand side then has the posterior-mean numerator as its mean and
    covariance ``AᵀN⁻¹A + S⁻¹`` — the operator itself — so ``x = M⁻¹b`` has the posterior
    mean and covariance ``M⁻¹M M⁻¹ = M⁻¹`` exactly. Not an approximation and
    not a Markov chain: every call is an independent draw, with no burn-in and
    nothing to diagnose for convergence.

    This is what makes a 10⁶-dimensional block samplable at all. It costs one
    CG solve — the same as the mean — because the fluctuation enters the
    right-hand side, never the operator.

    In a Gibbs scheme, this draws the linear block conditional on the nonlinear
    parameters; rebuild the block with
    :func:`linear_operator(..., check=False)` each sweep, having checked the
    linearity claim once outside the loop.

    Args:
        block: from :func:`linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise_std: noise standard deviation — scalar or broadcastable to the data.
        prior_std: prior standard deviation on the latent. Required, as for
            :func:`wiener_solve`. For a complex latent this is the width of the
            real and imaginary parts independently.
        key: PRNG key. ``vmap`` over split keys for many independent draws.
        prior_mean: centre of the prior; defaults to zero. With uninformative
            data the draws fall back to ``N(prior_mean, prior_std²)``, which is
            the check that it is wired in correctly.
        tol: CG tolerance.
        maxiter: CG iteration cap.
        require_convergence: as for :func:`wiener_solve`.

    Returns:
        ``(x, relative_residual)``. An unconverged CG returns a draw from the
        WRONG distribution, so ``require_convergence`` is on by default here
        too — see :func:`wiener_solve`.
    """
    _check_solve_arguments(block, observed, prior_std, "gcr_sample")
    return _conjugate_solve(
        block, observed, noise_std=noise_std, prior_std=prior_std,
        prior_mean=prior_mean, tol=tol, maxiter=maxiter, key=key,
        require_convergence=require_convergence,
    )
