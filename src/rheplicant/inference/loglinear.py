"""Multiplicative noise, taken to logs: a conjugate block where there was none.

:class:`~rheplicant.inference.noise.RadiometerNoise` generates
``d = mu (1 + f w)`` with ``f = 1 / sqrt(delta_nu tau)`` — the multiplicative
form the radiometer equation actually states. Take logs::

    log d = log mu + log(1 + f w)

and ``log(1 + f w) -> N(0, f^2)`` to first order. Two things follow, and the
second is the one that is easy to miss.

**A block whose ``log mu`` is affine becomes conjugate.** ``mu = exp(A x)`` is
not affine in ``x``, so :func:`~rheplicant.inference.linear.linear_operator`
refuses it and the only route is a gradient block — NUTS, and inside a
:class:`~rheplicant.inference.plan.SamplingPlan` a gradient block's potential
carries no ``sum log sigma``, so it targets the GLS-flavoured posterior rather
than the full one. In log space the same block is an ordinary
:class:`~rheplicant.inference.linear.LinearBlock`, so it gets
:func:`~rheplicant.inference.linear.wiener_solve` and an exact
:func:`~rheplicant.inference.linear.gcr_sample` draw instead.

**And the log-space sigma does not depend on the prediction.**
``Var[log(1 + f w)]`` is a function of ``f`` alone — measured at prediction
magnitudes 1, 1e3 and 1e6 it moves in the fifth significant figure, which is
the Monte Carlo floor of the measurement. So the reweighting that
:func:`~rheplicant.inference.gls.iterative_gls` exists to perform has nothing
left to do for that block: one solve, not a fixed point, and
``depends_on_prediction`` is genuinely ``False`` rather than approximately so.
The whole GLS-versus-full-likelihood distinction that
:mod:`rheplicant.inference.noise` sets out is a consequence of sigma tracking
the prediction, and in log space it does not.

**A summed sky is no obstacle, which is the fact that makes this useful here.**
For ``d = g (T_ant + T_nw + tone)``, ``log`` of the sum is not affine in the
sky coefficients — but the GAIN block does not need it to be. Conditional on
the sky, ``log d = log g + log S`` with ``log S`` a known constant, and a known
constant added to the prediction is exactly what
:attr:`~rheplicant.inference.linear.LinearBlock.offset` holds. So the gain is
log-linear whatever the sky is made of, and the sky block stays an ordinary
linear block in the original space. Each block takes the space its own
conditional is affine in.

**The approximation, and its size.** First order is not exact:
``E[log(1 + f w)] = -f^2 / 2`` to leading order, and ``Var`` exceeds ``f^2``.
Both are corrected or bounded here rather than left implicit —
:func:`to_log_space` adds the ``f^2 / 2`` back, and ``f`` above
:data:`FIRST_ORDER_MAX_FRACTIONAL` is refused. Measured, over 2e7 draws:

======  ==================  ==============================
``f``   ``Var / f^2 - 1``   ``mean / (-f^2 / 2)``
======  ==================  ==============================
0.001   below the floor     1.00
0.004   below the floor     1.00
0.06    0.0088              1.006
0.10    0.0258              1.016
0.30    0.3983              1.185
======  ==================  ==============================

The operating range is the top of that table and not the bottom: ``f`` for the
configurations in this repository runs 4.0e-4 (``3.125e6 x 2.0``) to 4.1e-3
(``61e3 x 1.0``), where the mean shift is 8e-8 to 8.2e-6 and the variance
error is below what 2e7 draws can resolve. The refusal at 0.06 is therefore
not a limit anyone meets by observing; it fires on a noise model that has been
mis-specified, which is the only way to reach it.

**Why positivity is checked and not left to arithmetic.** ``log`` of a
non-positive sample is NaN, and NaN fails every comparison — so a NaN
departure in :func:`~rheplicant.inference.linear.check_linearity`'s
``departure > rtol`` test reads as *passing*. A guard that cannot fail is the
defect this package hunts, so both exits here refuse a non-positive value by
name before any log is taken. Flagged samples are exempt: an unobserved sample
informs nothing, may hold anything, and is carried through at infinite sigma.
"""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from bayesmith.exact.loglinear import (
    FIRST_ORDER_MAX_FRACTIONAL,
    multiplicative_log_data,
)

from rheplicant.core.errors import (
    LinearityRefused,
    LogSpaceUnavailable,
    ParameterSpaceError,
)
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.linear import (
    LinearBlock,
    _affinity_errors,
    _group_probe,
    _isolate,
    _isolate_group,
    _require_inexact,
    _resolve_names,
    _single_probe,
)
from rheplicant.inference.noise import FlaggedNoise, NoiseModel
from rheplicant.inference.parameters import ParameterSpace

#: Largest fractional noise for which the first-order log-space equivalence is
#: used — BAYESMITH's constant, re-exported rather than re-spelled. The number
#: (0.06, where the log-space variance taken as ``f^2`` is 1 % wrong) and the
#: measurement behind it live in ``bayesmith.exact.loglinear``, which is also
#: where the transform arithmetic below comes from: one threshold, one shift,
#: one home, two paradigms reading it.

#: Probe magnitudes for the log-space affinity check, as multiples of the
#: latent's own scale. Deliberately NOT
#: :data:`~rheplicant.inference.linear.DEFAULT_SCALES`, whose top entry is
#: ``1e3``: here the probe is fed through an exponential before the log is
#: taken, and ``exp`` overflows above about 88 in float32. A probe 1000x a
#: log-latent's scale therefore measures the dtype, not the model — the map is
#: still exactly affine in log space where its exponential is not
#: representable. The span kept is four decades of probe, which is what the
#: linear check's own docstring says the sweep is for.
LOG_DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0)


#: The named reasons a noise model has no log route.
#:
#: The vocabulary is deliberately the same two names bayesmith's
#: ``NOT_LOG_LINEAR_REASONS`` uses for the same two verdicts, so a reader
#: comparing the packages is comparing words as well as behaviour.
LOG_ROUTE_REFUSALS: frozenset[str] = frozenset(
    {"noise_additive", "fractional_too_large"}
)


def log_route_refusal(noise: Any) -> str | None:
    """Which of :data:`LOG_ROUTE_REFUSALS` applies to ``noise``, or ``None``.

    **The predicate, extracted so there is exactly one of it.** Whether a log
    route exists is asked in two places — here, at PARTITION time by
    :func:`~rheplicant.inference.partition.auto_blocks`, and at SOLVE time by
    :func:`to_log_space`, which raises the full refusal — and the two must not
    be able to disagree. Before this function they could: `auto_blocks` took no
    noise model at all, so it could produce a ``log_conjugate`` block that
    ``to_log_space`` then refused, which
    :mod:`rheplicant.inference.partition`'s own docstring names as the failure
    that module exists to prevent. Measured on 2026-08-27 by D17's dual-run
    protocol, against bayesmith's graph-side probe, which reads the noise when
    it partitions.

    Returns a REASON rather than raising, because at partition time "no log
    route here" is a blameless verdict that routes the latent to a gradient
    block; the raise belongs where a caller asked for the transform by name.
    """
    if isinstance(noise, FlaggedNoise):
        noise = noise.base
    fractional = getattr(noise, "fractional", None)
    if fractional is None:
        return "noise_additive"
    if float(fractional) > FIRST_ORDER_MAX_FRACTIONAL:
        return "fractional_too_large"
    return None


def _fraction_and_flags(noise: Any) -> tuple[float, jax.Array | None]:
    """The multiplicative level ``f``, and the flags if the model carries any.

    Refuses anything whose noise is not multiplicative in the prediction, since
    for those the log transform is not a change of variables that buys
    anything — it is simply a different, wrong, likelihood.

    The two refusals are decided by :func:`log_route_refusal` and only WORDED
    here: one predicate, two consumers, so the partition-time verdict and this
    one cannot drift apart.
    """
    flags = None
    if isinstance(noise, FlaggedNoise):
        flags, noise = noise.flags, noise.base

    refusal = log_route_refusal(noise)
    fractional = getattr(noise, "fractional", None)
    if refusal == "noise_additive":
        raise ParameterSpaceError(
            f"to_log_space() needs a multiplicative noise model — one generating "
            f"d = mu (1 + f w) — and {type(noise).__name__} is not one. The log "
            "transform turns a multiplicative noise into an additive, "
            "prediction-independent one; applied to a model whose noise is ALREADY "
            "additive it does not simplify anything, it just states a different "
            "likelihood from the one declared. RadiometerNoise is the model this "
            "reads, optionally wrapped in FlaggedNoise."
        )

    fractional = float(fractional)
    if refusal == "fractional_too_large":
        raise ParameterSpaceError(
            f"This noise model has f = {fractional:.4g}, above the "
            f"{FIRST_ORDER_MAX_FRACTIONAL} at which the first-order log-space "
            "equivalence is still good to 1 %. log(1 + f w) is treated here as "
            "N(-f^2/2, f^2); measured, its variance is 0.9 % above f^2 at f = 0.06 "
            "and 40 % above at f = 0.3, where a 90 % interval covers 83 %. For "
            "reference, f = 1/sqrt(delta_nu tau) is 4.1e-3 for a 61 kHz channel at "
            "1 s — two orders of magnitude below this — so reaching it means "
            "channel_width or integration_time is not what was intended."
        )
    return fractional, flags


def _resolve_log_name(space: ParameterSpace, name: str | None) -> str:
    """The latent to build a log-linear block over, insisting it is named.

    Deliberately NOT
    :func:`~rheplicant.inference.linear._resolve_name`, which infers "the one
    declared ``linear=True``" and refuses anything else. A log-linear latent is
    precisely one that is not: if the prediction were affine in it, ``log`` of
    the prediction would not be. There is no declaration to read here, so there
    is nothing to infer from and the caller says which latent.
    """
    if name is None:
        raise ParameterSpaceError(
            "log_linear_operator() needs name= for one latent, or names= for a group. "
            "The linear machinery can infer 'the one declared linear' from "
            "Latent(linear=True); log-linearity has no such declaration, so there is "
            "nothing to infer from and the latent has to be named."
        )
    if name not in space.names:
        raise ParameterSpaceError(
            f"{name!r} is not a latent of this space; declared: {list(space.names)}."
        )
    if space.latent(name).linear:
        raise ParameterSpaceError(
            f"Latent {name!r} is declared linear=True, which says the prediction is "
            "affine in it — and then log(prediction) is not, so it has no log-linear "
            "block. The two claims exclude each other: exp of an affine function is "
            "affine only where it is constant. Use linear_operator for this latent, or "
            "drop linear=True if the model really enters through an exponential."
        )
    return name


#: What a zero or non-finite value means at the block's ZERO: nothing has been
#: perturbed yet, so it is the model, and a prediction that vanishes when the
#: block is zeroed is the signature of a LINEAR one.
_ZERO_AT_ORIGIN = (
    "Nothing has been perturbed at this point, so this is the model itself: a "
    "prediction that vanishes when the block is set to zero is what a LINEAR block "
    "does — mu = A 0 = 0 — and a log-linear one predicts exp(affine), which is never "
    "zero. This latent almost certainly belongs in linear_operator. (If the model "
    "really is exp(affine), check that the latent is bound THROUGH the exponential, "
    "Bind(..., fn=jnp.exp), and not directly onto the gain.)"
)

#: What it means at a probe: the exponential left the dtype's range, which is
#: arithmetic rather than a statement about the model.
_ZERO_AT_PROBE = (
    "That is normally the exponential leaving the dtype's range rather than anything "
    "wrong with the model — exp overflows above about 88 in float32 and 709 in "
    "float64, so a probe several times a log-latent's own scale can reach it while "
    "the map stays exactly affine in log space. Probe at smaller `scales`, give the "
    "latent a representative `init` if its scale is being taken from the all-zero "
    "fallback, or enable float64."
)


def _resolve_log_names(
    space: ParameterSpace, names: Sequence[str] | str
) -> tuple[str, ...]:
    """The group form of :func:`_resolve_log_name`.

    Reuses :func:`~rheplicant.inference.linear._resolve_names` for every
    structural check and turns off only its ``linear=True`` requirement, then
    applies the opposite one: a member declared linear is a contradiction here
    for the reason :func:`_resolve_log_name` gives.
    """
    selected = _resolve_names(space, names, require_linear=False)
    declared = [name for name in selected if space.latent(name).linear]
    if declared:
        raise ParameterSpaceError(
            f"Latent(s) {declared} are declared linear=True, which says the prediction "
            "is affine in them — and then log(prediction) is not, so they have no "
            "log-linear block. The two claims exclude each other: exp of an affine "
            "function is affine only where it is constant. Use linear_operator for "
            "these, or drop linear=True if the model really enters through an "
            "exponential."
        )
    return selected


def _refuse_non_positive(
    values: jax.Array, where: str, exempt: jax.Array | None, zero_remedy: str
) -> None:
    """Refuse a value ``log`` would turn into NaN or an infinity, by cause.

    The causes call for different things, so they are separated rather than
    merged into "non-positive":

    * **negative** — the prediction changes sign, so it is not ``exp`` of
      anything and the block is not log-linear. The same answer wherever it is
      found, which is why this branch needs no remedy from the caller.
    * **exactly zero, or not finite** — ambiguous on its own, so the caller
      supplies the reading: :data:`_ZERO_AT_ORIGIN` where nothing has been
      perturbed, :data:`_ZERO_AT_PROBE` where a probe has. Guessing one of
      those would put a confident wrong diagnosis in front of a reader — a
      linear gain zeroes its prediction at the origin with no overflow in
      sight, and the overflow message would send them to look at dtypes.

    All of it is refused rather than propagated because NaN fails every
    comparison: a NaN departure reads as PASSING
    :func:`~rheplicant.inference.linear.check_linearity`'s ``departure > rtol``
    test, and a NaN residual reads as a converged solve.
    """
    considered = jnp.ones_like(values, dtype=bool) if exempt is None else ~exempt
    total = int(jnp.size(values))

    negative = (values < 0) & considered
    count = int(jnp.sum(negative))
    if count:
        worst = float(jnp.min(jnp.where(negative, values, jnp.inf)))
        raise LogSpaceUnavailable(
            f"{where}: {count} of {total} unflagged values are NEGATIVE (smallest "
            f"{worst:.6g}). A log-linear model predicts exp(affine), which is positive "
            "everywhere, so a sign change says this block is not log-linear — or, for "
            "data, that samples which should be flagged are not. log() of a negative "
            "value is NaN, and NaN fails every comparison, so this is refused rather "
            "than propagated into a check that would then read as passing."
        )

    unusable = (~jnp.isfinite(values) | (values == 0)) & considered
    count = int(jnp.sum(unusable))
    if count:
        raise LogSpaceUnavailable(
            f"{where}: {count} of {total} unflagged values are zero or not finite, so "
            f"log() of them is not a number this can work with. {zero_remedy}"
        )


def check_log_linearity(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
    scales: Sequence[float] = LOG_DEFAULT_SCALES,
    rtol: float | None = None,
    key: jax.Array | None = None,
) -> dict[float, float]:
    """Verify that ``log(prediction)`` is affine in one latent — or in a group.

    The log-space counterpart of
    :func:`~rheplicant.inference.linear.check_linearity`, asking the same
    question of ``log`` of the same map, at the same probe points — the probe
    scheme is imported rather than restated, so the two cannot drift into
    probing different models while both reporting on "linearity".

    Args:
        space, pipeline, state_template: the model under test.
        name: which latent. Optional when exactly one is declared linear.
        names: several, checked **jointly**, as for the linear check. A gain and
            a sky are jointly log-linear no more than they are jointly linear.
        at: values for the latents OUTSIDE the block. For a log-linear gain this
            is where the sky goes, and it matters: ``log S`` is the block's
            offset, so the claim is made *given* it.
        scales, rtol, key: as for
            :func:`~rheplicant.inference.linear.check_linearity`.

    Returns:
        ``{scale: relative error}``, as the linear check returns.

    Raises:
        ParameterSpaceError: if both spellings are given, or if the prediction
            is non-positive anywhere the log would be taken.
        LinearityRefused: if any scale departs from affinity by more than
            ``rtol``.
    """
    if name is not None and names is not None:
        raise ParameterSpaceError(
            "check_log_linearity() takes name= for one latent or names= for a group, "
            "not both."
        )
    key = jax.random.key(0) if key is None else key

    if names is None:
        name = _resolve_log_name(space, name)
        g, zero = _isolate(space, pipeline, state_template, name, at)
        _require_inexact(space, (name,))
        probe_at: Callable[[int, float], Any] = _single_probe(space, name, key)
        subject = (
            f"log(prediction) is not affine in latent {name!r}, so it has no log-linear "
            "block"
        )
        scale_of = "the latent's scale"
    else:
        selected = _resolve_log_names(space, names)
        g, zero = _isolate_group(space, pipeline, state_template, selected, at)
        _require_inexact(space, selected)
        probe_at = _group_probe(space, selected, key)
        subject = (
            f"log(prediction) is not affine in latents {list(selected)} JOINTLY, so "
            "they have no shared log-linear block"
        )
        scale_of = "each latent's own scale"

    _refuse_non_positive(
        g(zero), "check_log_linearity, at the block's zero", None, _ZERO_AT_ORIGIN
    )
    for index, scale in enumerate(scales):
        _refuse_non_positive(
            g(probe_at(index, scale)),
            f"check_log_linearity, at a probe of {scale:g}x {scale_of}",
            None,
            _ZERO_AT_PROBE,
        )

    def g_log(x: Any) -> jax.Array:
        return jnp.log(g(x))

    # No noise column here, and that is the point of log space rather than an
    # omission: the transform's whole purpose is to leave a noise that no
    # longer depends on the prediction, so 'the departure in units of sigma'
    # is a question about the ORIGINAL likelihood, not this one.
    errors, _weighted, failed, rtol = _affinity_errors(
        g_log, zero, probe_at, scales, rtol
    )
    if failed:
        detail = ", ".join(f"{scale:g}x -> {err:.2e}" for scale, err in errors.items())
        raise LinearityRefused(
            f"{subject}: departure from its own linearization exceeds rtol={rtol:.2e} "
            f"at {failed} times {scale_of} ({detail}). A log-linear block is one whose "
            "prediction is exp(affine) — a gain entered as exp(log_gain) is, a gain "
            "entered directly is not (it is affine already, so use linear_operator). "
            "Check which side of the exponential this latent is bound on.",
            errors=errors,
            rtol=rtol,
            failed=failed,
        )
    return errors


def log_linear_operator(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
    check: bool = True,
    scales: Sequence[float] = LOG_DEFAULT_SCALES,
    rtol: float | None = None,
) -> LinearBlock:
    """Export ``log(prediction)`` as a linear block, for data taken to logs.

    The returned :class:`~rheplicant.inference.linear.LinearBlock` is an
    ordinary one — nothing downstream knows or needs to know that its ``offset``
    is ``log`` of something. Feed it to
    :func:`~rheplicant.inference.linear.wiener_solve` or
    :func:`~rheplicant.inference.linear.gcr_sample` with the data and sigma that
    :func:`to_log_space` returns::

        block = log_linear_operator(space, twin, state, "log_gain", at=values)
        y, sigma = to_log_space(observed, noise)
        draw, info = gcr_sample(block, y, noise_std=sigma, prior_std=P, key=key)

    ``offset`` is ``log`` of the prediction with the block at zero — for a gain
    bound as ``exp(log_gain)``, that is ``log`` of the sky, which is why a
    summed sky costs this nothing.

    Args:
        space, pipeline, state_template: the model.
        name, names: one latent or a group, as for
            :func:`~rheplicant.inference.linear.linear_operator`.
        at: values for the latents outside the block. A log-linear gain block is
            log-linear *given* the sky, so a sweep must pass the current sky
            here every sweep.
        check: verify the claim first with :func:`check_log_linearity`. This is
            the eager path — it converts values to Python floats, so a caller
            rebuilding the block inside a jitted sweep passes ``check=False``
            and checks once outside, exactly as the linear machinery does.
        scales, rtol: forwarded to the check.

    Returns:
        A :class:`~rheplicant.inference.linear.LinearBlock` over log space.
    """
    if check:
        check_log_linearity(
            space,
            pipeline,
            state_template,
            name,
            names=names,
            at=at,
            scales=scales,
            rtol=rtol,
        )

    if names is None:
        name = _resolve_log_name(space, name)
        g, zero = _isolate(space, pipeline, state_template, name, at)
        latent = space.latent(name)
        shape: Any = latent.init.shape
        dtype: Any = latent.init.dtype
        prior: Any = latent.prior
        block_name: Any = name
    else:
        selected = _resolve_log_names(space, names)
        g, zero = _isolate_group(space, pipeline, state_template, selected, at)
        shape = {member: space.latent(member).init.shape for member in selected}
        dtype = {member: space.latent(member).init.dtype for member in selected}
        prior = {member: space.latent(member).prior for member in selected}
        block_name = selected

    def g_log(x: Any) -> jax.Array:
        return jnp.log(g(x))

    offset, tangent = jax.linearize(g_log, zero)
    _, pullback = jax.vjp(g_log, zero)

    return LinearBlock(
        name=block_name,
        shape=shape,
        dtype=dtype,
        offset=offset,
        forward=tangent,
        adjoint=lambda y: pullback(y)[0],
        prior=prior,
    )


def has_log_linear_block(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
    scales: Sequence[float] = LOG_DEFAULT_SCALES,
    rtol: float | None = None,
) -> bool:
    """The same question as :func:`check_log_linearity`, answered rather than raised.

    Discovery needs a verdict: :func:`~rheplicant.inference.partition.auto_blocks`
    asks this of every latent that is not already declared linear, and a "no" is
    an ordinary answer rather than a fault. :func:`check_log_linearity` stays the
    exit that refuses, because a caller who has *asserted* log-linearity wants
    the departure numbers and the remedy, not a ``False``.

    Exactly two refusals are read as "no" — a departure from affinity
    (:class:`~rheplicant.core.errors.LinearityRefused`) and a prediction ``log``
    cannot be taken of (:class:`~rheplicant.core.errors.LogSpaceUnavailable`).
    Anything else propagates: a latent of integer dtype or a name the space does
    not declare means the question was never asked, and filing that as "not
    log-linear" would route a broken declaration to a gradient block silently.
    """
    try:
        check_log_linearity(
            space,
            pipeline,
            state_template,
            name,
            names=names,
            at=at,
            scales=scales,
            rtol=rtol,
        )
    except (LinearityRefused, LogSpaceUnavailable):
        return False
    return True


def to_log_space(
    observed: jax.Array, noise: NoiseModel
) -> tuple[jax.Array, jax.Array]:
    """Take the data to logs, and hand back the sigma that goes with it.

    One function for both because they must agree: the transformed data and the
    log-space sigma are two halves of one claim about the noise, and a caller
    that took the log itself and reached for ``f`` separately could get the
    ``f^2 / 2`` shift on one and not the other.

    ``log d = log mu + log(1 + f w)`` and ``E[log(1 + f w)] = -f^2 / 2``, so the
    log data sits low by a CONSTANT — the same for every sample, independent of
    the prediction — and adding it back is exact arithmetic rather than an
    estimate. Measured, the leading-order shift accounts for the true mean to
    0.6 % at f = 0.06 and to better than 0.5 % below it.

    Args:
        observed: the data, as recorded. Strictly positive wherever it is not
            flagged.
        noise: a multiplicative model —
            :class:`~rheplicant.inference.noise.RadiometerNoise`, optionally
            wrapped in :class:`~rheplicant.inference.noise.FlaggedNoise`.

    Returns:
        ``(y, sigma)``. ``y`` is ``log(observed) + f^2 / 2``; ``sigma`` is ``f``
        broadcast to the data's shape, carrying ``inf`` at flagged samples so
        they keep the zero weight the flags gave them. ``sigma`` is a plain
        array and not a ``NoiseModel``, which is what the conjugate solvers
        accept — and it is CONSTANT, which is the point: no reweighting loop.

    Raises:
        ParameterSpaceError: if the noise is not multiplicative; if its ``f`` is
            above :data:`FIRST_ORDER_MAX_FRACTIONAL`; or if an unflagged sample
            is non-positive.
    """
    fractional, flags = _fraction_and_flags(noise)
    observed = jnp.asarray(observed)
    if flags is not None and jnp.shape(flags) != jnp.shape(observed):
        raise ParameterSpaceError(
            f"flags shape {jnp.shape(flags)} does not match the data shape "
            f"{jnp.shape(observed)}."
        )
    _refuse_non_positive(
        observed,
        "to_log_space",
        flags,
        "A recorded sample of exactly zero, or one that is not finite, is not "
        "something a log-space likelihood can weight. If those samples were not "
        "observed, they belong in FlaggedNoise's flags, where an infinite sigma "
        "gives them a clean zero weight and this exit passes them through.",
    )

    safe = observed if flags is None else jnp.where(flags, 1.0, observed)
    # The shift and the sigma are bayesmith's arithmetic — one statement of
    # the first-order transform for both packages. What stays here is what is
    # this package's own: the NoiseModel reading above, and the flag handling
    # below, which bayesmith's graph paradigm has no counterpart for.
    y, sigma = multiplicative_log_data(safe, fractional)
    if flags is not None:
        sigma = jnp.where(flags, jnp.inf, sigma)
    return y, sigma


__all__ = [
    "FIRST_ORDER_MAX_FRACTIONAL",
    "LOG_DEFAULT_SCALES",
    "LOG_ROUTE_REFUSALS",
    "check_log_linearity",
    "has_log_linear_block",
    "log_route_refusal",
    "log_linear_operator",
    "to_log_space",
]
