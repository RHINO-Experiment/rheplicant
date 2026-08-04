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

**Where the prior comes from.** ``S`` is read off ``Latent(prior=...)`` — the
same declaration :func:`~rheplicant.inference.numpyro_bridge.to_numpyro_model`
reads, so one space handed to NUTS and to :func:`gcr_sample` targets one
posterior. The ``prior_std=`` / ``prior_mean=`` keywords remain for a
prior-free latent, but a keyword that *contradicts* a declaration is refused
rather than allowed to win, and a declared prior with no conjugate Gaussian
form is refused rather than approximated by its first two moments. Both would
otherwise be a finite, confident posterior for a model nobody declared, which
is the failure mode every guard in this module is placed against.

**Probe at extreme scales.** :func:`check_linearity` probes at 10⁻³, 1 and 10³
times the latent's own magnitude, because near-linearity is scale-dependent:
``x + εx²`` is indistinguishable from linear near the origin and grossly
nonlinear far from it. A probe suite that only samples "reasonable" values
signs off on exactly the blocks that will fail in a sampler's tails.

**A residual is not an accuracy.** The solvers here are iterative, and what an
iterative method can cheaply report is ``‖M x - b‖``, not ``‖x - x*‖``. The two
differ by the condition number of ``M = AᵀN⁻¹A + S⁻¹``, and κ is large here by
*design*: whenever the data does not fully identify the block — which is the
case the prior is for — ``λ_min(M)`` is exactly ``1/prior_std²`` and κ runs to
1e6 and beyond. A solve can then sit at a relative residual of 1e-7 with the
prior-dominated directions untouched, and a draw comes back with almost no
scatter where it should have carried the whole prior width. So the guard on
these solves bounds the *error*, ``κ · residual``, and :func:`condition_estimate`
exposes κ for choosing ``tol``.
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
from rheplicant.inference.conditioning import extreme_eigenvalues, tree_norm
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.parameters import ParameterSpace

DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1.0, 1e3)

#: Power-iteration steps per end of the spectrum in :func:`condition_estimate`.
#: Both ends typically settle within three; this leaves margin at a fixed cost
#: of ``2 * POWER_ITERATIONS`` operator applications per guarded solve.
POWER_ITERATIONS: int = 12

#: ``jax.ensure_compile_time_eval``, under the name of what it does at the one
#: place it is used: run this comparison **now**, on the constants in hand,
#: rather than emitting it into whatever trace happens to be open. See
#: :func:`_agrees`.
_RIGHT_NOW = jax.ensure_compile_time_eval


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
        prior: the latent's declared prior, carried through from the
            :class:`~rheplicant.inference.parameters.Latent`. ``None`` for a
            prior-free latent, and for a block assembled by hand. It is what
            lets :func:`wiener_solve` and :func:`gcr_sample` read ``S`` off the
            declaration instead of making the caller hand-pass — and hand-sync
            — the same two numbers at every exit.

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
    prior: Any = None


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
        prior=latent.prior,
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


def _numpyro_distributions() -> Any:
    """numpyro's distribution module, or ``None`` when it is not installed.

    Imported here rather than at module scope because numpyro is an optional
    extra and this module is usable without it — a prior-free linear block
    solves from keywords alone.
    """
    try:
        import numpyro.distributions as distributions
    except ImportError:  # pragma: no cover - numpyro is an optional extra
        return None
    return distributions


def _gaussian_parameters(prior: Any) -> tuple[Any, Any] | None:
    """``(loc, scale)`` if ``prior`` is a Gaussian **on the latent itself**.

    ``None`` otherwise — including for distributions that merely look like one.
    Identification is by TYPE, never by attribute, and that is the whole point:
    ``numpyro.distributions.LogNormal`` carries ``.loc`` and ``.scale`` and even
    a ``.base_dist`` that *is* a ``Normal``, while being a Gaussian in ``log x``
    and not in ``x``. Duck-typing on ``.loc``/``.scale`` would read those two
    numbers off it and return a finite, confident posterior for a
    parameterization nobody declared, which is exactly the failure this module
    exists to refuse.

    ``Independent`` and ``ExpandedDistribution`` are unwrapped because both
    only re-shape a base distribution; ``TransformedDistribution`` and the
    truncations are not, because both change what the distribution *is*.
    """
    distributions = _numpyro_distributions()
    if distributions is None:  # pragma: no cover - numpyro is an optional extra
        return None
    if isinstance(prior, (distributions.Independent, distributions.ExpandedDistribution)):
        return _gaussian_parameters(prior.base_dist)
    if isinstance(prior, distributions.Normal):
        return prior.loc, prior.scale
    return None


def _holds_a_tracer(value: Any) -> bool:
    """Whether ``value`` carries a tracer anywhere inside it.

    Asked of the pytree *leaves* rather than of the object itself, so a tracer
    wrapped in a list or a tuple is still recognised as one. The alternative —
    letting the comparison run and reading the failure — cannot tell a tracer
    apart from a shape mismatch, because ``TracerArrayConversionError`` is a
    ``TypeError``; an unanswerable comparison would then be reported as a
    settled *disagreement*, which is the one verdict this must never invent.
    """
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _agrees(supplied: Any, declared: Any) -> bool | None:
    """Whether two prior parameters are the same number. ``None``: undecidable.

    Only a genuine tracer is undecidable. Two concrete numbers are the same two
    numbers whether or not some enclosing ``jit`` or ``lax.while_loop`` happens
    to be tracing — so the comparison is evaluated *here*, on the constants in
    hand, rather than staged into that trace. Staged, ``bool()`` raises on the
    result, a settled ``True`` comes back as unanswerable, and
    :func:`_reconcile` refuses a correct call while blaming a tracer that does
    not exist. That is not hypothetical:
    :func:`~rheplicant.inference.gls.iterative_gls` resolves the prior once and
    re-passes it into :func:`wiener_solve` from *inside* its reweighting loop,
    so this guard meets a live trace on every iteration of the one function it
    was written to serve.

    The comparison itself stays in ``jnp``, which canonicalizes both sides to
    the working precision. Comparing in NumPy instead would widen a declared
    ``float32`` scale to ``float64`` and call ``prior_std=0.05`` a
    contradiction of ``Normal(jnp.asarray(1.0), jnp.asarray(0.05))``, whose
    scale reads ``0.05000000074505806`` once widened — the same false refusal,
    moved rather than removed.
    """
    if _holds_a_tracer(supplied) or _holds_a_tracer(declared):
        return None
    try:
        with _RIGHT_NOW():
            return bool(jnp.all(jnp.asarray(supplied) == jnp.asarray(declared)))
    except jax.errors.ConcretizationTypeError:
        # Unreachable given the check above, and kept regardless: an
        # undecidable comparison has to reach the caller as undecidable, never
        # as a verdict.
        return None
    except (TypeError, ValueError):
        # Shapes that do not even broadcast are a disagreement, not a crash.
        return False


def _reconcile(
    keyword: str, field: str, supplied: Any, declared: Any, block: LinearBlock, caller: str
) -> Any:
    """The supplied keyword, or the declared value — never a silent choice."""
    if supplied is None:
        return declared
    verdict = _agrees(supplied, declared)
    if verdict is None:
        side = (
            f"the {keyword}= you passed is"
            if _holds_a_tracer(supplied)
            else f"latent {block.name!r}'s declared {field} is"
        )
        raise ParameterSpaceError(
            f"{caller} cannot check the {keyword}= it was given against the prior latent "
            f"{block.name!r} declares: {side} a traced value, so what the two are cannot be "
            "known until the trace runs. Pass one or the other, not both: whichever lost "
            "would still look like it was in force. (Two CONCRETE values are compared "
            "normally, jit or no jit — being inside a trace is not itself the problem.)"
        )
    if not verdict:
        raise ParameterSpaceError(
            f"{caller} was given {keyword}={supplied!r}, but latent {block.name!r} declares "
            f"prior={type(block.prior).__name__}(..., {field}={declared!r}) in its "
            "ParameterSpace. One of the two would silently win and the other would be a "
            "number you believed was in force — and that same declaration reaches "
            "to_numpyro_model unchanged, so this exit and NUTS would then target different "
            f"posteriors from one space. Drop {keyword}= and let the declaration drive the "
            "solve, or change the declaration."
        )
    return supplied


def _resolve_prior(
    block: LinearBlock, prior_mean: Any, prior_std: Any, caller: str
) -> tuple[Any, Any]:
    """Fill ``prior_mean``/``prior_std`` from the latent's declaration.

    A block with no declared prior passes straight through — that is the escape
    hatch for a prior-free latent, which the optimizers use and which
    ``prior_std=`` alone is enough for.
    """
    if block.prior is None:
        return prior_mean, prior_std
    gaussian = _gaussian_parameters(block.prior)
    if gaussian is None:
        raise ParameterSpaceError(
            f"{caller} is a conjugate-Gaussian solve, but latent {block.name!r} declares a "
            f"{type(block.prior).__name__} prior, which has no conjugate Gaussian form. "
            "These exits solve (AᵀN⁻¹A + S⁻¹)x = b, and S⁻¹ only exists as a matrix for a "
            "Gaussian S; substituting the distribution's mean and variance would return a "
            "finite, confident posterior for a prior you did not declare — narrower than "
            "the truth wherever the declared prior is skewed or bounded. Sample this space "
            "with to_numpyro_model + NUTS instead, which honours the prior as written, or "
            "declare a numpyro Normal here and keep the conjugate exits."
        )
    loc, scale = gaussian
    return (
        _reconcile("prior_mean", "loc", prior_mean, loc, block, caller),
        _reconcile("prior_std", "scale", prior_std, scale, block, caller),
    )


def _require_prior_std(prior_std: Any, caller: str) -> None:
    """No prior at all leaves AᵀN⁻¹A free to be singular."""
    if prior_std is None:
        raise ParameterSpaceError(
            f"{caller} needs prior_std: with no prior the normal operator AᵀN⁻¹A can be "
            "singular, and CG would return a finite, arbitrary answer rather than fail. "
            "Pass a large prior_std for an effectively flat prior, or declare "
            "Latent(prior=dist.Normal(...)) and it will be read from there."
        )


def _check_solve_arguments(
    block: LinearBlock,
    observed: jax.Array,
    prior_mean: Any,
    prior_std: Any,
    caller: str,
) -> tuple[Any, Any]:
    """Shared preconditions for the mean and the draw, plus the resolved prior.

    Returns the ``(prior_mean, prior_std)`` the solve should actually use: the
    keywords when they were given, the latent's declaration when they were not,
    and an exception when the two disagree.
    """
    check_observed_shape(jnp.shape(block.offset), observed)
    prior_mean, prior_std = _resolve_prior(block, prior_mean, prior_std, caller)
    _require_prior_std(prior_std, caller)
    if jnp.issubdtype(jnp.asarray(block.offset).dtype, jnp.complexfloating):
        raise ParameterSpaceError(
            f"{caller} expects a real-valued prediction; this block's offset is complex."
        )
    return prior_mean, prior_std


def wiener_solve(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any = None,
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
            broadcastable to it. **Defaults to the latent's declared prior**;
            required only when there is none, because without a prior the
            normal operator can be singular and CG would return a finite,
            arbitrary answer instead of complaining. Passing a value that
            contradicts the declaration raises rather than one silently
            winning — see the note below.
        prior_mean: centre of the prior. Defaults to the declared prior's
            location, and to zero when nothing is declared — which is wrong for
            most physical quantities, a noise-wave temperature sitting near
            250 K. Equivalent to an affine binding that adds the same offset,
            but says what it means.
        tol: CG tolerance — a bound on the relative RESIDUAL, which is not the
            same as accuracy. See the note on conditioning below.
        maxiter: CG iteration cap. ``None`` lets JAX choose.
        require_convergence: raise unless the relative ERROR can be bounded by
            this. Defaults to ``1e-3``; ``None`` disables the guard and returns
            whatever CG produced. On by default because jax's ``cg`` reports no
            convergence status, so an unconverged solve otherwise comes back
            looking exactly like a converged one.

            The bound is ``κ · relative_residual``, with ``κ`` estimated by
            :func:`condition_estimate`. Guarding on the residual alone would
            certify nothing in the regime that matters — see below — so this
            costs ``2 · POWER_ITERATIONS`` extra operator applications. That is
            not free: on a well-conditioned block, where CG itself converges in
            a few iterations, it roughly DOUBLES the solve. In a Gibbs sweep,
            where the conditioning barely moves from sweep to sweep, call
            :func:`condition_estimate` once outside the loop, choose ``tol``
            from it, and pass ``require_convergence=None`` inside — the same
            bargain :func:`linear_operator`'s ``check`` offers.

    Returns:
        ``(x̂, relative_residual)``, the residual being ``‖M x̂ - b‖ / ‖b‖``
        over the real degrees of freedom. Note that this is the residual, not
        the error; multiply by :func:`condition_estimate` for the error bound.

    Note:
        **Conditioning, and why ``tol`` is not accuracy.** Residual and error
        differ by the condition number of ``M = AᵀN⁻¹A + S⁻¹``::

            ‖x̂ - x*‖ / ‖x*‖  ≤  κ(M) · ‖M x̂ - b‖ / ‖b‖

        For a block the data does not fully identify — one calibration load
        against three unknowns, a flagged channel, a short integration — the
        prior is the only thing holding the blind directions down, so
        ``λ_min(M)`` is exactly ``1/prior_std²`` and ``κ ≈ ‖AᵀN⁻¹A‖ · prior_std²``
        runs to 1e6 and beyond. At κ=1e7 the default ``tol=1e-6`` bounds the
        relative error by 10: no digits at all. CG stops on a residual that
        looks converged, having left the prior-dominated directions at their
        starting value, and the draw comes back with far too little scatter.

        This is exactly the regime these solvers exist for, so the guard is on
        by default and the accuracy target is stated as an error, not a
        residual. To solve rather than refuse, pass ``tol ≈
        require_convergence / κ`` with a ``maxiter`` to match. Past ``κ · eps``
        no tolerance helps and only precision does; the guard says so in its
        own words.

    Note:
        **Where S comes from.** ``Latent(prior=dist.Normal(m, s))`` is the
        package's one statement of what a latent is a priori, and it is the
        statement ``to_numpyro_model`` reads. So it is the statement this solve
        reads too: declare it once and both exits target the same posterior.
        The keywords remain, for a prior-free latent and for overriding a
        declaration you are deliberately solving away from — but a keyword that
        *contradicts* a declaration raises, because the alternative is one of
        the two silently winning and the two exits quietly disagreeing. A
        declared prior with no conjugate Gaussian form (a Half-Normal, a
        Uniform) raises here as well; NUTS is where that space belongs.
    """
    prior_mean, prior_std = _check_solve_arguments(
        block, observed, prior_mean, prior_std, "wiener_solve"
    )
    return _conjugate_solve(
        block, observed, noise_std=noise_std, prior_std=prior_std,
        prior_mean=prior_mean, tol=tol, maxiter=maxiter, key=None,
        require_convergence=require_convergence,
    )


def _normal_operator(block: LinearBlock, weight, prior_variance) -> Callable:
    """``x -> (AᵀN⁻¹A + S⁻¹) x`` over the latent's real degrees of freedom.

    The curvature half is taken as a gradient rather than assembled from
    ``A`` and ``Aᵀ``, which makes it symmetric positive definite by
    construction with no adjoint convention left to get wrong.
    """
    split, join = _real_parts(block)

    def half_chi2(parts):
        return 0.5 * jnp.sum(weight * block.forward(join(parts)) ** 2)

    def normal(parts):
        curvature = jax.grad(half_chi2)(parts)
        return jax.tree.map(lambda c, p: c + p / prior_variance, curvature, parts)

    return normal


def _condition_number(
    block: LinearBlock, weight, prior_variance, key, iterations: int
) -> jax.Array:
    """Estimated ``κ`` of ``AᵀN⁻¹A + S⁻¹``."""
    split, _ = _real_parts(block)
    template = split(jnp.zeros(block.shape, dtype=block.dtype))
    largest, smallest = extreme_eigenvalues(
        _normal_operator(block, weight, prior_variance), template, key, iterations
    )
    # AᵀN⁻¹A is positive semi-definite, so λ_min can never fall below the
    # prior's own curvature however rank-deficient the data is.
    floor = 1.0 / jnp.max(jnp.asarray(prior_variance))
    return largest / jnp.maximum(smallest, floor)


def condition_estimate(
    block: LinearBlock,
    *,
    noise_std: Any,
    prior_std: Any = None,
    iterations: int = POWER_ITERATIONS,
    key: jax.Array | None = None,
) -> jax.Array:
    """Condition number of the normal operator this block would be solved with.

    ``κ(AᵀN⁻¹A + S⁻¹)`` is the number that says how much a solver's residual
    understates its error: for a solution ``x`` with relative residual ``r``,

        ‖x - x*‖ / ‖x*‖  ≤  κ · r

    so a residual of 1e-6 against κ=1e7 certifies nothing at all. Use it to
    pick ``tol`` for :func:`wiener_solve` and :func:`gcr_sample`: for a target
    relative accuracy ``a``, ask for roughly ``tol = a / κ``.

    Large κ is not a defect here, it is the design: for a block the data does
    not fully identify, ``λ_min`` is exactly ``1/prior_std²`` while ``λ_max``
    is set by the data, so κ grows with how much better the data constrains
    one direction than the prior constrains another.

    Costs ``2 · iterations`` applications of the normal operator — each the
    same JVP-plus-VJP a CG iteration costs — and no matrix is ever formed.

    Args:
        block: from :func:`linear_operator`.
        noise_std, prior_std: as for :func:`wiener_solve`, ``prior_std``
            included — it defaults to the latent's declared prior, so the κ
            reported here is the κ of the system those solves will build rather
            than of a system nobody solves.
        iterations: power-iteration steps per end of the spectrum. The default
            is comfortable; the estimate typically settles within three.
        key: PRNG key for the starting vectors. Fixed by default, so the
            estimate is reproducible.

    Returns:
        The estimated condition number, as a scalar array.
    """
    _, prior_std = _resolve_prior(block, None, prior_std, "condition_estimate")
    _require_prior_std(prior_std, "condition_estimate")
    return _condition_number(
        block,
        1.0 / jnp.asarray(noise_std) ** 2,
        jnp.asarray(prior_std) ** 2,
        jax.random.key(0) if key is None else key,
        iterations,
    )


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

    def pair_with(vector):
        """``Aᵀ vector`` in real coordinates, as the gradient of a real pairing.

        Taking it as a gradient rather than calling ``block.adjoint`` is what
        keeps the real/complex conventions from ever entering: ``jax.grad`` of a
        real scalar is by construction the adjoint of the real inner product,
        which is the pairing every term here lives in.
        """
        return jax.grad(lambda parts: jnp.sum(block.forward(join(parts)) * vector))(zero)

    normal = _normal_operator(block, weight, prior_variance)

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
    residual = tree_norm(misfit) / jnp.maximum(tree_norm(rhs), 1e-30)
    if require_convergence is not None:
        # jax's cg reports no convergence status of its own, so an unconverged
        # solve otherwise comes back looking like any other answer. eqx.error_if
        # fires under jit, where a Python `if` on a traced value cannot.
        #
        # The residual ALONE cannot decide this. Error and residual differ by
        # the condition number, and for a block the data does not fully
        # identify κ is enormous by construction — λ_min is exactly the prior's
        # 1/prior_std² — so CG stops on a tiny residual with the prior-dominated
        # directions still at their starting value, and hands back a draw whose
        # posterior scatter there is orders of magnitude too small. Guarding on
        # the residual certifies precisely nothing in the one regime these
        # solvers exist to serve.
        kappa = _condition_number(
            block, weight, prior_variance, jax.random.key(0), POWER_ITERATIONS
        )
        error_bound = residual * kappa
        bad = jnp.logical_or(~jnp.isfinite(residual), error_bound > require_convergence)

        # Below κ·eps no tolerance can help: the arithmetic itself cannot
        # represent the answer that accurately. Worth its own message, because
        # the remedy is precision, and the natural response to the other
        # message — tighten tol, raise maxiter — burns a great many iterations
        # here to arrive at an equally wrong answer.
        epsilon = float(jnp.finfo(jnp.asarray(block.offset).dtype).eps)
        unreachable = kappa * epsilon > require_convergence

        solution = eqx.error_if(
            solution,
            jnp.logical_and(bad, unreachable),
            "wiener_solve/gcr_sample cannot reach require_convergence at this "
            "precision: the normal operator's condition number times the machine "
            "epsilon already exceeds it, so no tol or maxiter will help. This is "
            "the usual signature of a block the data does not identify. Enable "
            "jax_enable_x64, or strengthen the prior (prior_std bounds the "
            "conditioning: κ ≈ ‖AᵀN⁻¹A‖·prior_std²). condition_estimate() reports "
            "the number.",
        )
        solution = eqx.error_if(
            solution,
            jnp.logical_and(bad, ~unreachable),
            "wiener_solve/gcr_sample did not converge: the relative residual times "
            "the normal operator's condition number — the bound on the RELATIVE "
            "ERROR, which is what require_convergence limits — exceeds it. The "
            "residual alone looks converged; it is not, along the directions the "
            "prior dominates. Pass tol ≈ require_convergence/κ with a maxiter to "
            "match, or strengthen the prior. condition_estimate() reports κ.",
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
    prior_std: Any = None,
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
    linearity claim once outside the loop. The conditioning guard is worth
    hoisting the same way: :func:`condition_estimate` once to fix ``tol``, then
    ``require_convergence=None`` in the loop. What you must NOT do is leave
    ``tol`` at its default and the guard off — that is the combination this
    module returned a silently over-confident posterior for.

    Args:
        block: from :func:`linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise_std: noise standard deviation — scalar or broadcastable to the data.
        prior_std: prior standard deviation on the latent. Defaults to the
            latent's declared prior, as for :func:`wiener_solve`, and required
            only when there is none. For a complex latent this is the width of
            the real and imaginary parts independently.
        key: PRNG key. ``vmap`` over split keys for many independent draws.
        prior_mean: centre of the prior; defaults to the declared prior's
            location, and to zero when nothing is declared. With uninformative
            data the draws fall back to ``N(prior_mean, prior_std²)``, which is
            the check that it is wired in correctly.
        tol: CG tolerance — a bound on the residual, not on the accuracy.
        maxiter: CG iteration cap.
        require_convergence: as for :func:`wiener_solve`, including the
            conditioning note there, which a draw is MORE exposed to than the
            mean. The fluctuation term ``S⁻¹ᐟ²ω₂`` puts weight on every
            direction of the latent by construction, including the ones the
            data is blind to — so a draw always has something to resolve where
            the operator is worst conditioned, whereas the mean does only when
            ``prior_mean`` is nonzero.

    Returns:
        ``(x, relative_residual)``. An unconverged CG returns a draw from the
        WRONG distribution — and a distribution that is too NARROW, since the
        directions left unresolved are the prior-dominated ones that should
        have carried the most scatter — so ``require_convergence`` is on by
        default here too.

    Note:
        ``S`` is read off ``Latent(prior=...)`` when the keywords are omitted;
        see the corresponding note on :func:`wiener_solve` for what that does
        and does not permit. It matters more here than for the mean: with a
        declared prior ignored, the fluctuation term ``S⁻¹ᐟ²ω₂`` is drawn at
        the wrong width, so every draw is wrong in the one direction the mean
        can be right in.
    """
    prior_mean, prior_std = _check_solve_arguments(
        block, observed, prior_mean, prior_std, "gcr_sample"
    )
    return _conjugate_solve(
        block, observed, noise_std=noise_std, prior_std=prior_std,
        prior_mean=prior_mean, tol=tol, maxiter=maxiter, key=key,
        require_convergence=require_convergence,
    )
