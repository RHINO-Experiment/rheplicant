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

**One block may hold several latents.** ``linear_operator(..., names=("t_nw",
"t_ant"))`` exports the joint operator over a *group*, whose ``x`` is a
``{name: array}`` dict rather than one array — and whose solve returns the same
dict, so the physical names survive instead of the caller slicing an anonymous
stacked vector and getting the offsets right by hand. Nothing is concatenated:
the group's domain is a pytree, ``cg`` already solves over pytrees, and the
prior is block-diagonal by construction because each latent's ``S`` sits on its
own leaf.

Grouping is not cosmetic. Two latents the data cannot tell apart are solved
*together* in one CG here, whereas alternating between them as two blocks
converges at the rate of their correlation — and reports a per-block residual
of 1e-7 and a per-block ``κ`` of 1 the whole way down, because both numbers are
computed from the block and neither can see across the partition. The joint
``κ`` this module reports for the grouped block *can*; so can
:func:`~rheplicant.inference.identifiability.identifiability`, which is the
right instrument for choosing the partition in the first place. And a group
whose members are only *pairwise* linear — a gain against an antenna
temperature — is refused by :func:`check_linearity`, which probes the joint map
and finds it bilinear, not affine.

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

from collections.abc import Sequence
from typing import Any

import jax
from bayesmith.exact.linearity import (
    WEIGHTED_RTOL,
)

from rheplicant.core.errors import LinearityRefused, ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.parameters import ParameterSpace

from .linear_block import _RIGHT_NOW as _RIGHT_NOW
from .linear_block import (
    DEFAULT_SCALES,
    LinearBlock,
)
from .linear_block import _domain_centre as _domain_centre
from .linear_priors import _agrees as _agrees
from .linear_priors import _gaussian_parameters as _gaussian_parameters
from .linear_priors import _group_priors as _group_priors
from .linear_priors import _holds_a_tracer as _holds_a_tracer
from .linear_priors import _numpyro_distributions as _numpyro_distributions
from .linear_priors import _per_member as _per_member
from .linear_priors import _reconcile as _reconcile
from .linear_priors import (
    _refuse_a_noise_model_at_the_conjugate_seam as _refuse_a_noise_model_at_the_conjugate_seam,
)
from .linear_priors import _require_prior_std as _require_prior_std
from .linear_priors import _resolve_one_prior as _resolve_one_prior
from .linear_priors import _resolve_prior as _resolve_prior
from .linear_probe import (
    _BOTH_SPELLINGS,
    DEFAULT_AT_POINTS,
    _affinity_errors,
    _group_probe,
    _isolate,
    _isolate_group,
    _prior_at_points,
    _require_inexact,
    _resolve_name,
    _resolve_names,
    _single_probe,
    _values_at,
    _worse,
)
from .linear_probe import _is_complex as _is_complex
from .linear_probe import _magnitude as _magnitude
from .linear_probe import _probe_anchor as _probe_anchor
from .linear_probe import _reported as _reported

# Re-exported so this module's importers keep working. `X as X` is the
# explicit re-export form: a plain import of a name this file does not
# itself use is F401, and `ruff --fix` deletes it whatever the comment
# on the line says.
from .linear_solve import _OBSERVED as _OBSERVED
from .linear_solve import _as_far_block as _as_far_block
from .linear_solve import _check_solve_arguments as _check_solve_arguments
from .linear_solve import _far_precision as _far_precision
from .linear_solve import _from_far_domain as _from_far_domain
from .linear_solve import condition_bound as condition_bound
from .linear_solve import condition_estimate as condition_estimate
from .linear_solve import gcr_sample as gcr_sample
from .linear_solve import wiener_solve as wiener_solve


def check_linearity(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
    at_points: Sequence[dict[str, jax.Array]] | None = None,
    noise: Any | None = None,
    key: jax.Array | None = None,
) -> dict[float, float]:
    """Verify that the prediction really is affine in one latent — or in a group.

    Compares the model against its own linearization at zero, at several
    magnitudes of probe. Costs one linearization plus one forward evaluation
    per scale.

    Args:
        space, pipeline, state_template: the model under test.
        name: which latent. Optional when exactly one is declared linear.
        names: several latents, checked **jointly** — the claim a grouped
            :func:`linear_operator` block makes. Mutually exclusive with
            ``name``. This is strictly stronger than checking each in turn, and
            the difference is the whole reason a bilinear model needs more than
            one block: a gain and an antenna temperature are each affine given
            the other, and their product is not affine in the pair, so a group
            holding both is refused here rather than solved as if it were
            linear.
        at: values for the latents OUTSIDE the block. Linearity is a claim
            *given* them, so check it where the sampler will actually be.
            Defaults to the declared initial values.
        at_points: the outside values to check at, in full. Defaults to ``at``
            plus ``DEFAULT_AT_POINTS - 1`` draws from those latents' own
            priors (D16 axis 2, ruled 2026-08-27). **Passing a single point is
            how a check becomes a moderate-parameter probe**, which is the
            failure mode ``boundary-validation.md`` exists to prevent; do it
            only when the model is used at exactly one outside value. Measured
            before the default moved: a model affine in ``u`` exactly when the
            outside latent sits at its declared init reported 0.0 at every
            scale and was accepted. A latent with no Gaussian prior keeps its
            ``at`` value at every point -- a free parameter has no
            distribution to draw from.
        scales: probe magnitudes, as multiples of the latent's own scale,
            taken from its **prior width** — per latent, for a group, since two
            latents in one block are routinely in different units. The default
            spans six orders of magnitude on purpose — see the module
            docstring. The prior is where the sampler will actually go, which
            is why it and not ``init`` sets the magnitude: an all-zero ``init``
            is the ordinary declaration for a sky, and anchoring on it made the
            probes absolute. A latent with no Gaussian prior to read falls back
            to ``max|init|``, and to 1.0 if that is zero too.
        rtol: tolerance on the relative departure from affinity. Default:
            ``1e4 * eps`` of the prediction dtype, which leaves room for
            accumulated roundoff in a long reduction without admitting real
            curvature.
        noise: the model's noise, enabling a SECOND criterion — the departure
            in units of sigma, against
            :data:`~bayesmith.exact.linearity.WEIGHTED_RTOL` (D16 axis 3,
            ruled 2026-08-27). "Small" has to be small compared to something,
            and a departure far under ``rtol`` can still be many noise widths
            wide, which is the regime a conjugate solve gets wrong. Measured
            on one such model, the relative column reads ``0.000e+00`` at
            every probe while the weighted one reads ``6.262e-02``. Omitted,
            only the relative criterion applies and the verdict is the one
            this function gave before the axis moved — a weaker check, not a
            different one, so no warning: unlike the log route, nothing here
            makes a positive claim that a missing noise model would render
            unsafe.
        key: PRNG key for the probes. Fixed by default, so the check is
            reproducible. For a group the per-latent sub-keys are folded in by
            position in the SORTED names, so permuting ``names`` probes the
            model at the same points and returns the same verdict.

    Returns:
        ``{scale: relative error}`` — useful for reporting how linear a block
        is, not only whether it passes.

    Raises:
        ParameterSpaceError: if ``name`` and ``names`` are both given.
        LinearityRefused: if any scale departs from affinity by more than
            ``rtol``. It IS a ``ParameterSpaceError`` -- an existing
            ``except ParameterSpaceError`` needs no change -- and it carries
            the same per-scale numbers this returns on the passing branch, so
            a caller can report the departure instead of quoting the sentence.
    """
    if name is not None and names is not None:
        raise ParameterSpaceError(f"check_linearity() {_BOTH_SPELLINGS}")
    key = jax.random.key(0) if key is None else key

    if names is None:
        name = _resolve_name(space, name)
        selected = (name,)
        _require_inexact(space, selected)
        probe_at = _single_probe(space, name, key)

        def isolate_at(point, _n=name):
            return _isolate(space, pipeline, state_template, _n, point)

        subject = f"Latent {name!r} is declared linear=True, but the prediction is not affine in it"
        scale_of = "the latent's scale"
        remedy = (
            "Either drop the declaration, or re-parameterize so the model really is "
            "linear in this block."
        )
    else:
        selected = _resolve_names(space, names)
        _require_inexact(space, selected)
        probe_at = _group_probe(space, selected, key)

        def isolate_at(point, _s=selected):
            return _isolate_group(space, pipeline, state_template, _s, point)

        subject = (
            f"Latents {list(selected)} are each declared linear=True, but the prediction "
            "is not affine in them JOINTLY"
        )
        scale_of = "each latent's own scale"
        remedy = (
            "Each conditional of a bilinear model is affine on its own, which is why "
            "this is not caught one latent at a time — and why these two cannot share "
            "one linear block. Split them into separate blocks and alternate, or "
            "re-parameterize so the joint map really is affine. "
            "identifiability(space, pipeline, state) will tell you what the split "
            "costs before you choose it."
        )

    # D16 axis 2: the claim is "affine GIVEN the outside latents", so it is
    # checked at more than the one value they were declared at. The verdicts
    # are merged per scale by the WORSE of them -- the axis-4 ruling, which
    # keeps this function's published return shape one level deep.
    points = (
        list(at_points)
        if at_points is not None
        else _prior_at_points(space, selected, _values_at(space, {}, at), DEFAULT_AT_POINTS, key)
    )
    merged: dict[float, float] = {}
    merged_weighted: dict[float, float] | None = None
    failed_scales: list[float] = []
    for point in points:
        g, zero = isolate_at(point)
        errors, weighted, failed, rtol = _affinity_errors(g, zero, probe_at, scales, rtol, noise)
        for scale, value in errors.items():
            merged[scale] = value if scale not in merged else _worse(merged[scale], value)
        if weighted is not None:
            merged_weighted = merged_weighted or {}
            for scale, value in weighted.items():
                merged_weighted[scale] = (
                    value
                    if scale not in merged_weighted
                    else (_worse(merged_weighted[scale], value))
                )
        failed_scales.extend(f for f in failed if f not in failed_scales)
    errors, weighted, failed = merged, merged_weighted, sorted(failed_scales)
    if failed:
        detail = ", ".join(f"{scale:g}x -> {err:.2e}" for scale, err in errors.items())
        # The second criterion's own column, and only when it was asked. Both
        # are reported because the guard is a DISJUNCTION: one number against
        # one threshold is unreadable half the time, since a reader sees a
        # value under the tolerance printed beside a refusal and concludes the
        # guard is broken.
        weighted_detail = (
            ""
            if weighted is None
            else (
                "; in units of sigma against weighted_rtol="
                f"{WEIGHTED_RTOL:.2e}: "
                + ", ".join(f"{scale:g}x -> {err:.2e}" for scale, err in weighted.items())
            )
        )
        # The subclass, and the SAME sentence: `detail` renders the numbers for
        # a reader, and `errors=` hands the same numbers to a caller that has
        # to do something with them. Rendering them only would leave parsing
        # this string as the one route to a number this function already has.
        raise LinearityRefused(
            f"{subject}: departure from its own linearization exceeds rtol={rtol:.2e} "
            f"(above the per-probe roundoff floor) at {failed} times {scale_of} "
            f"({detail}){weighted_detail}. {remedy}",
            errors=errors,
            rtol=rtol,
            failed=failed,
            weighted=weighted,
            weighted_rtol=None if weighted is None else WEIGHTED_RTOL,
        )
    return errors


def linear_operator(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
    check: bool = True,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
) -> LinearBlock:
    """Export ``A``, ``Aᵀ`` and the offset for a declared-linear latent — or a group.

    No matrix is ever formed: ``A`` comes from ``jax.linearize`` and ``Aᵀ``
    from ``jax.vjp``, so a 10⁶-dimensional block costs the same as one forward
    evaluation per application. That is what makes conjugate-Gaussian solves
    tractable here — see :func:`wiener_solve`.

    Args:
        space, pipeline, state_template: the model.
        name: which latent. Optional when exactly one is declared linear. The
            block's ``x`` is then one array, and so is the solve's answer.
        names: several latents, exported as ONE block. Mutually exclusive with
            ``name``. The block's ``x`` is then a ``{name: array}`` dict — and
            so is the answer, which is the point: the physical names survive the
            solve instead of the caller slicing an anonymous stacked vector.
            ``names=("gain",)`` is a legitimate group of one, and is how a
            partition can hold one-latent and many-latent blocks without the
            caller special-casing either.

            Solving a group JOINTLY is not the same as alternating over its
            members: two latents the data barely tells apart are resolved in one
            CG here, where alternation converges at the rate of their
            correlation while reporting a converged residual and a κ of ~1 at
            every step. The joint κ that
            :func:`condition_estimate` reports for this block is the honest one.
        at: values for the latents OUTSIDE the block, fixing where it is built.
            Defaults to the declared initial values — right exactly once, so a
            Gibbs sweep must pass the current values here every sweep.
        check: verify the linearity claim first (:func:`check_linearity`).
            Leave it on. Turning it off costs three forward evaluations less
            and buys a class of silent, confident errors. For a group the claim
            checked is JOINT affinity, which a bilinear pair fails.
        scales, rtol: forwarded to :func:`check_linearity`.

    Raises:
        ParameterSpaceError: if both ``name`` and ``names`` are given; if
            ``names`` is empty, repeats a latent, or names an undeclared or
            non-linear one; or if the linearity claim fails.
    """
    if name is not None and names is not None:
        raise ParameterSpaceError(f"linear_operator() {_BOTH_SPELLINGS}")

    if names is None:
        name = _resolve_name(space, name)
        if check:
            check_linearity(space, pipeline, state_template, name, at=at, scales=scales, rtol=rtol)
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

    selected = _resolve_names(space, names)
    if check:
        check_linearity(
            space, pipeline, state_template, names=selected, at=at, scales=scales, rtol=rtol
        )
    g, zero = _isolate_group(space, pipeline, state_template, selected, at)

    offset, tangent = jax.linearize(g, zero)
    _, pullback = jax.vjp(g, zero)

    return LinearBlock(
        name=selected,
        shape={member: space.latent(member).init.shape for member in selected},
        dtype={member: space.latent(member).init.dtype for member in selected},
        offset=offset,
        forward=tangent,
        adjoint=lambda y: pullback(y)[0],
        prior={member: space.latent(member).prior for member in selected},
    )
