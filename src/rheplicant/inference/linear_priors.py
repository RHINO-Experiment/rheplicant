"""Turning declared priors into the Gaussian parameters a solve needs.

Reconciling a NumPyro distribution with an explicit mean and standard
deviation, per member and per group, and refusing the combinations that have
no conjugate meaning. Separated because a prior is resolved once and read by
every solve, and because its failures are about the DOCUMENT rather than
about the arithmetic.
"""

from typing import Any

import jax
import jax.numpy as jnp

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.noise import NoiseModel

from .linear_block import (
    _RIGHT_NOW,
    LinearBlock,
)


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
    keyword: str, field: str, supplied: Any, declared: Any, name: str, prior: Any, caller: str
) -> Any:
    """The supplied keyword, or the declared value — never a silent choice."""
    if supplied is None:
        return declared
    verdict = _agrees(supplied, declared)
    if verdict is None:
        side = (
            f"the {keyword}= you passed is"
            if _holds_a_tracer(supplied)
            else f"latent {name!r}'s declared {field} is"
        )
        raise ParameterSpaceError(
            f"{caller} cannot check the {keyword}= it was given against the prior latent "
            f"{name!r} declares: {side} a traced value, so what the two are cannot be "
            "known until the trace runs. Pass one or the other, not both: whichever lost "
            "would still look like it was in force. (Two CONCRETE values are compared "
            "normally, jit or no jit — being inside a trace is not itself the problem.)"
        )
    if not verdict:
        raise ParameterSpaceError(
            f"{caller} was given {keyword}={supplied!r}, but latent {name!r} declares "
            f"prior={type(prior).__name__}(..., {field}={declared!r}) in its "
            "ParameterSpace. One of the two would silently win and the other would be a "
            "number you believed was in force — and that same declaration reaches "
            "to_numpyro_model unchanged, so this exit and NUTS would then target different "
            f"posteriors from one space. Drop {keyword}= and let the declaration drive the "
            "solve, or change the declaration."
        )
    return supplied


def _resolve_one_prior(
    name: str, prior: Any, prior_mean: Any, prior_std: Any, caller: str
) -> tuple[Any, Any]:
    """Fill ``prior_mean``/``prior_std`` from one latent's declaration.

    A latent with no declared prior passes straight through — that is the escape
    hatch for a prior-free latent, which the optimizers use and which
    ``prior_std=`` alone is enough for.
    """
    if prior is None:
        return prior_mean, prior_std
    gaussian = _gaussian_parameters(prior)
    if gaussian is None:
        raise ParameterSpaceError(
            f"{caller} is a conjugate-Gaussian solve, but latent {name!r} declares a "
            f"{type(prior).__name__} prior, which has no conjugate Gaussian form. "
            "These exits solve (AᵀN⁻¹A + S⁻¹)x = b, and S⁻¹ only exists as a matrix for a "
            "Gaussian S; substituting the distribution's mean and variance would return a "
            "finite, confident posterior for a prior you did not declare — narrower than "
            "the truth wherever the declared prior is skewed or bounded. Sample this space "
            "with to_numpyro_model + NUTS instead, which honours the prior as written, or "
            "declare a numpyro Normal here and keep the conjugate exits."
        )
    loc, scale = gaussian
    return (
        _reconcile("prior_mean", "loc", prior_mean, loc, name, prior, caller),
        _reconcile("prior_std", "scale", prior_std, scale, name, prior, caller),
    )


def _group_priors(block: LinearBlock) -> dict[str, Any]:
    """The declared prior of each member of a group.

    A hand-assembled grouped block may carry ``prior=None``, meaning no member
    declares one; anything else has to be a dict covering every member, because
    a single distribution object standing for a whole group is a statement about
    latents in different units that nobody made, and dropping it silently would
    solve at whatever ``prior_std=`` happened to say.
    """
    if block.prior is None:
        return dict.fromkeys(block.names)
    if isinstance(block.prior, dict) and set(block.prior) == set(block.names):
        return block.prior
    raise ParameterSpaceError(
        f"This block groups {list(block.names)}, so its `prior` must be a dict with one "
        f"entry per member (use None for a prior-free one); it holds "
        f"{type(block.prior).__name__}"
        + (f" keyed by {sorted(block.prior)}" if isinstance(block.prior, dict) else "")
        + ". S is block-diagonal over the group and each member contributes its own "
        "block, so there is no reading under which one declaration covers all of them."
    )


def _per_member(keyword: str, value: Any, block: LinearBlock, caller: str) -> dict[str, Any]:
    """A grouped keyword, split by member. ``None`` everywhere when not given."""
    if value is None:
        return dict.fromkeys(block.names)
    if not isinstance(value, dict):
        raise ParameterSpaceError(
            f"{caller} was given {keyword}={value!r} for a block grouping "
            f"{list(block.names)}, but a grouped block has one prior PER LATENT — S is "
            "block-diagonal, not a multiple of the identity. These latents are routinely "
            "in different units (a noise-wave temperature in kelvin, a gain of order one), "
            "so one number spread across all of them is a prior nobody declared, and it "
            "would come back as a finite, confidently wrong posterior. Pass a dict keyed "
            f"by latent name — {keyword}={{{block.names[0]!r}: ...}} — or omit it and let "
            "each latent's own Latent(prior=...) drive the solve."
        )
    unknown = [key for key in value if key not in block.names]
    if unknown:
        raise ParameterSpaceError(
            f"{caller} was given {keyword} for {unknown}, which this block does not group; "
            f"it holds {list(block.names)}. The entry would be silently dropped and the "
            "latent it names solved at some other prior entirely."
        )
    return {member: value.get(member) for member in block.names}


def _resolve_prior(
    block: LinearBlock, prior_mean: Any, prior_std: Any, caller: str
) -> tuple[Any, Any]:
    """Fill ``prior_mean``/``prior_std`` from the block's declaration(s).

    For a group the resolution is per member and independent — each one takes
    its keyword if it was given, its declaration if it was not, and raises if
    the two disagree — so a group mixing a declared latent with a prior-free one
    is honoured rather than refused wholesale, and it is
    :func:`_require_prior_std` that names any member left with nothing at all.
    """
    if not block.grouped:
        return _resolve_one_prior(block.name, block.prior, prior_mean, prior_std, caller)

    priors = _group_priors(block)
    means = _per_member("prior_mean", prior_mean, block, caller)
    stds = _per_member("prior_std", prior_std, block, caller)
    resolved_mean: dict[str, Any] = {}
    resolved_std: dict[str, Any] = {}
    for member in block.names:
        resolved_mean[member], resolved_std[member] = _resolve_one_prior(
            member, priors[member], means[member], stds[member], caller
        )
    return resolved_mean, resolved_std


def _require_prior_std(block: LinearBlock, prior_std: Any, caller: str) -> None:
    """No prior at all leaves AᵀN⁻¹A free to be singular."""
    if block.grouped:
        missing = [member for member in block.names if prior_std[member] is None]
        if not missing:
            return
        detail = (
            f"needs a prior_std for {missing} — the other members of this block have one, "
            "which does not help: "
        )
    else:
        if prior_std is not None:
            return
        detail = "needs prior_std: "
    raise ParameterSpaceError(
        f"{caller} {detail}with no prior the normal operator AᵀN⁻¹A can be "
        "singular, and CG would return a finite, arbitrary answer rather than fail. "
        "Pass a large prior_std for an effectively flat prior, or declare "
        "Latent(prior=dist.Normal(...)) and it will be read from there."
    )


def _refuse_a_noise_model_at_the_conjugate_seam(noise_std: Any, caller: str) -> None:
    """Say why a :class:`NoiseModel` does not belong here, instead of TypeError.

    ``check_noise_std_axis`` accepts a noise model -- it has to, since every
    other exit in the package passes one. This module does not: the conjugate
    solves take ``1 / sigma**2`` from a plain array. Without this refusal a
    model reaches ``jnp.asarray`` and comes back as
    ``TypeError: Value 'HomoscedasticNoise(sigma=weak_f32[])' with dtype
    object is not a valid JAX array type``, which names the wrong layer and
    reads like a bug in the package rather than a wrong argument.

    A prediction-dependent model gets the longer sentence because it is not a
    packaging problem: a conjugate solve has no prediction to evaluate one at,
    the prediction being what it solves for. Freezing sigma at some parameter
    tuple is a real choice with a statistical consequence -- see
    :mod:`rheplicant.inference.plan` -- and it belongs to the caller who knows
    which tuple, never to a silent unwrap here.
    """
    if not isinstance(noise_std, NoiseModel):
        return
    if getattr(noise_std, "depends_on_prediction", False):
        raise ParameterSpaceError(
            f"{caller} was given {type(noise_std).__name__}, whose sigma depends on "
            "the prediction — but a conjugate solve has no prediction to evaluate it "
            "at, because the prediction is what it solves for. Freeze it yourself at "
            "the parameter tuple you mean (`noise.std(prediction)`) and pass that "
            "array, which also makes explicit that the result is an exact draw at "
            "THAT covariance and not from the full model's conditional. A "
            "SamplingPlan does this per sweep."
        )
    raise ParameterSpaceError(
        f"{caller} takes a plain sigma array, not a {type(noise_std).__name__}. "
        "The conjugate solves compute 1/sigma**2 directly; pass `noise.std(...)`, "
        "or the sigma you built the model from."
    )
