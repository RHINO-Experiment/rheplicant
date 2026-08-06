"""Turn one epoch's data into a factor of the campaign likelihood.

Plan A ships one method, and it is the one with an exact oracle: a model
affine in every latent, Gaussian noise, and linear-Gaussian per-epoch
nuisances. The stored term is then a *sufficient statistic* -- no anchor, no
validity region, order-invariant -- which is what lets the exactness pin compare
a streamed campaign against a batch solve to roundoff.

Two pieces of arithmetic carry the whole module.

**Marginalisation is a QR drop.** Order the columns nuisance-first, append the
nuisance prior's whitened rows, and triangularise. The leading ``n_phi`` rows
and columns are then the nuisance block; dropping them *is* the Schur
complement, exactly. Two constants come out of that drop and BOTH belong in the
offset: ``-sum(log|diag|)`` off the discarded block, and ``-sum(log(std))``,
the nuisance prior's own Gaussian normalisation. The ``(n_phi/2) log(2 pi)``
that the Gaussian integral contributes cancels against the prior's copy of the
same factor; the ``sum(log(std))`` has nothing to cancel against. Omitting it
leaves every term wrong by a constant that grows with the nuisance count --
measured at 1.07 for three nuisances at ``std=0.7``, 27.5 for twenty-five at
``std=3`` -- and a constant is invisible in the posterior's shape, so gradients
and curvature look perfect while the evidence is wrong.

**The normalisation is masked.** ``FlaggedNoise`` encodes a flagged sample as
``sigma = inf``. Its inverse variance is a clean zero, but ``log(2 pi sigma^2)``
is ``+inf``, so an unmasked normalisation makes one flagged channel send the
whole term -- and then the whole campaign -- to ``-inf``, with every gradient
NaN. Only finite-sigma samples are summed, and ``n_observed`` records how many
there were.
"""

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compressed import (
    COEFFICIENTS,
    QuadraticLikelihood,
    ReducedBasisLikelihood,
)
from rheplicant.inference.sqrtinfo import SqrtInfo, marginalise
from rheplicant.inference.uncertainty import as_noise_model


def _stack(design: Mapping[str, jax.Array], names: tuple[str, ...]) -> jax.Array:
    """Column-concatenate the per-latent design blocks in declared order."""
    return jnp.concatenate([jnp.atleast_2d(design[name]) for name in names], axis=1)


def _refuse_traced_sigma(sigma: jax.Array, caller: str) -> None:
    """The jit guard, narrowed to sigma. Shared by both compressors.

    It tests whether *sigma* is traced, not whether the data is: ``n_observed``
    is a Python int recorded on the term and written to the archive manifest, so
    under a trace the number of unflagged samples is not known and the term
    could only be built by inventing one. Testing the data instead would refuse
    ``jax.grad`` and ``jax.vmap`` over ``observed``, which work and are useful --
    that narrowing is the fix Plan A had to make after shipping the broad
    version, and it is why there is now one copy of this guard rather than two
    that can drift apart.

    Left unguarded the symptom was a raw ``TracerBoolConversionError`` out of the
    NaN check on the line below, which names neither the cause nor the remedy.
    """
    if isinstance(sigma, jax.core.Tracer):
        raise StateValidationError(
            f"{caller} cannot run under jit: the flag pattern is traced, and "
            "n_observed is STATIC provenance -- a Python int recorded on the term "
            "and written to the archive manifest, not an array. Under a trace the "
            "number of unflagged samples is not known, so the term could only be "
            "built by inventing one.\n\n"
            "What does work, measured: jax.grad and jax.vmap over `observed`, "
            "because a concrete noise model keeps sigma concrete and only the data "
            "is traced. What does not: a FlaggedNoise whose flags are themselves "
            "traced. Compression is a once-per-epoch step AROUND a jitted forward "
            "model, not a step to put inside one -- jit the model that produces "
            "the prediction and the data, then call this on the results."
        )


def _epoch_weights(
    noise: Any, reference: jax.Array, observed: jax.Array, caller: str
) -> tuple[jax.Array, jax.Array, jax.Array, int]:
    """``(sigma, seen, weight, n_observed)`` for one epoch, with both guards.

    ``inf`` and NaN are separated here rather than downstream. ``inf`` is this
    package's encoding of "not observed"; NaN is a broken noise model, and the
    two are the same only to ``jnp.isfinite``. Left to that test a NaN sigma
    would be counted as a flag, silently shrinking ``n_observed`` and building
    the epoch's statistics on the rest -- finite, self-consistent, and quietly
    short of data.
    """
    sigma = jnp.broadcast_to(noise.std(reference), observed.shape)
    _refuse_traced_sigma(sigma, caller)
    if bool(jnp.any(jnp.isnan(sigma))):
        raise StateValidationError(
            f"{caller}: the noise model produced NaN for at least one sample. "
            "`inf` is this package's encoding of 'not observed' (FlaggedNoise); "
            "NaN is a broken noise model, and the two are only the same to "
            "`jnp.isfinite`. Left to that test the sample would be counted as "
            "flagged, silently shrinking n_observed and building the epoch's "
            "statistics on the rest -- finite, self-consistent, and quietly short "
            "of data. Fix the noise model, or mask the sample explicitly with "
            "FlaggedNoise."
        )
    seen = jnp.isfinite(sigma)
    weight = jnp.where(seen, 1.0 / jnp.where(seen, sigma, 1.0), 0.0)
    return sigma, seen, weight, int(jnp.sum(seen))


def compress_linear(
    design: Mapping[str, jax.Array],
    observed: jax.Array,
    noise_std: Any,
    shapes: Mapping[str, tuple[int, ...]],
    epoch_id: str,
    offset_prediction: jax.Array | None = None,
    nuisance_design: Mapping[str, jax.Array] | None = None,
    nuisance_prior_std: Mapping[str, Any] | None = None,
    nuisance_prior_mean: Mapping[str, jax.Array] | None = None,
    nuisance_shapes: Mapping[str, tuple[int, ...]] | None = None,
) -> QuadraticLikelihood:
    """Compress one epoch of a linear-Gaussian model into a sufficient statistic.

    Args:
        design: ``{global latent: (n_data, n_i) block}``. Column order follows
            iteration order of this mapping, which is also the term's latent
            order.
        observed: ``(n_data,)`` data for this epoch.
        noise_std: a scalar, an array, or a ``NoiseModel``. A
            prediction-dependent model must already have been resolved -- pass
            ``iterative_gls(...).noise_std``, and record that in the term.
        shapes: each global latent's shape.
        epoch_id: the recording's data hash.
        offset_prediction: the model's constant part, subtracted before
            compression. Storing the statistics about this origin rather than
            about zero is an exact change of variable, and it is what keeps the
            residual chi-square from being the time-bandwidth product.
        nuisance_design: ``{nuisance latent: (n_data, n_j) block}`` for the
            latents integrated out here.
        nuisance_prior_std: each nuisance latent's prior standard deviation.
            Required for every entry of ``nuisance_design``.
        nuisance_prior_mean: each nuisance latent's prior mean; zero if absent.
        nuisance_shapes: each nuisance latent's shape.

    Returns:
        A prior-free, exact
        :class:`~rheplicant.inference.compressed.QuadraticLikelihood` over the
        global latents.
    """
    names = tuple(design)
    if not names:
        raise StateValidationError("compress_linear needs at least one design block.")
    nuisance_design = dict(nuisance_design or {})
    nuisance_names = tuple(nuisance_design)
    nuisance_shapes = dict(nuisance_shapes or {})
    nuisance_prior_std = dict(nuisance_prior_std or {})
    nuisance_prior_mean = dict(nuisance_prior_mean or {})
    missing = [name for name in nuisance_names if name not in nuisance_prior_std]
    if missing:
        raise StateValidationError(
            f"Nuisance latent(s) {missing} are integrated out here, so their prior is "
            "part of the model rather than an optional regulariser. Give each one a "
            "nuisance_prior_std."
        )

    noise = as_noise_model(noise_std)
    prediction = (
        jnp.zeros_like(observed) if offset_prediction is None else offset_prediction
    )
    sigma, seen, weight, n_observed = _epoch_weights(
        noise, prediction, observed, "compress_linear"
    )

    # SELECT on `seen`, never multiply by a zero weight. A flagged sample is
    # exactly where a NaN lives -- that is usually why it was flagged -- and
    # `0.0 * nan` is `nan`, so multiplying propagates the value the mask exists
    # to discard. The poison then reaches `target` and `offset` while `factor`
    # stays finite, so every density is NaN while `audit()` reports a
    # well-conditioned campaign: measured lambda_min 94.06, condition 7.11,
    # all_exact True, log_likelihood nan. Once combined into the accumulator it
    # is irreversible. Every other masked path in the package already selects
    # (noise.py, engines.py, likelihood.py); this was the one copy that did not.
    masked = jnp.where(seen, observed - prediction, 0.0)
    residual = masked * weight
    global_block = jnp.where(seen[:, None], _stack(design, names), 0.0) * weight[:, None]
    width = global_block.shape[1]

    if nuisance_names:
        nuisance_block = (
            jnp.where(seen[:, None], _stack(nuisance_design, nuisance_names), 0.0)
            * weight[:, None]
        )
        n_nuisance = nuisance_block.shape[1]
        prior_rows, prior_target, prior_log_std = [], [], []
        for name in nuisance_names:
            size = int(jnp.zeros(nuisance_shapes.get(name, ())).size) or 1
            std = jnp.broadcast_to(jnp.asarray(nuisance_prior_std[name]), (size,))
            mean = jnp.broadcast_to(
                jnp.asarray(nuisance_prior_mean.get(name, 0.0)), (size,)
            )
            prior_rows.append(jnp.diag(1.0 / std))
            prior_target.append(mean / std)
            prior_log_std.append(jnp.sum(jnp.log(std)))
        prior_block = jax.scipy.linalg.block_diag(*prior_rows)
        augmented = jnp.concatenate(
            [
                jnp.concatenate(
                    [nuisance_block, global_block, residual[:, None]], axis=1
                ),
                jnp.concatenate(
                    [
                        prior_block,
                        jnp.zeros((n_nuisance, width)),
                        jnp.concatenate(prior_target)[:, None],
                    ],
                    axis=1,
                ),
            ],
            axis=0,
        )
        upper = jnp.linalg.qr(augmented, mode="r")
        nuisance_diag = jnp.abs(jnp.diag(upper)[:n_nuisance])
        # Both constants, and the second is the one an implementation forgets:
        # the Gaussian integral over p gives -sum(log|R_pp,ii|), and the prior
        # it was integrated against carries its own -sum(log(std)).
        marginal_logdet = -jnp.sum(jnp.log(nuisance_diag)) - sum(prior_log_std)
        upper = upper[n_nuisance:, n_nuisance:]
    else:
        augmented = jnp.concatenate([global_block, residual[:, None]], axis=1)
        upper = jnp.linalg.qr(augmented, mode="r")
        marginal_logdet = jnp.array(0.0)

    keep = min(upper.shape[0], width)
    corner = upper[keep:, width]
    normalisation = -0.5 * jnp.sum(
        jnp.where(seen, jnp.log(2.0 * jnp.pi * jnp.where(seen, sigma, 1.0) ** 2), 0.0)
    )
    info = SqrtInfo(
        factor=upper[:keep, :width],
        target=upper[:keep, width],
        offset=normalisation + marginal_logdet - 0.5 * jnp.sum(corner**2),
        names=names,
        shapes=tuple(shapes[name] for name in names),
    )
    return QuadraticLikelihood(
        info=info, epoch_id=epoch_id, n_observed=n_observed, exact=True
    )


def compress_reduced_basis(
    basis: Any,
    observed: jax.Array,
    noise: Any,
    epoch_id: str,
    *,
    support: dict[str, tuple[float, float]] | None = None,
    noise_frozen_at: str = "none",
    nuisance_design: Mapping[str, jax.Array] | None = None,
    nuisance_prior_std: Mapping[str, Any] | None = None,
    nuisance_prior_mean: Mapping[str, jax.Array] | None = None,
    nuisance_shapes: Mapping[str, tuple[int, ...]] | None = None,
    nuisance_tolerance: float = 1e-6,
) -> ReducedBasisLikelihood:
    """Compress one epoch against a shared reduced basis -- section 4.1's T1.

    The prediction is expanded in the basis and the epoch's likelihood becomes a
    quadratic in the coefficient vector. That is ``compress_linear``'s QR with
    ``S_e^T`` in place of the design matrix, so the rank-deficiency handling, the
    masked normalisation and the retained corner term are the same three lines
    rather than a second implementation of them.

    **The metric is the epoch's, the dictionary is the campaign's.** ``S_e`` is
    the shared rows whitened by *this* epoch's sigma, so the stored form is the
    exact likelihood of the model ``S^T c`` under ``N_e``. The coefficient map
    ``c(theta)`` was built in the reference metric and is shared, which is what
    makes evaluation ``O(n_S^2)`` per epoch instead of ``O(n_S n_data)``; the
    mismatch that buys is what the bias budget measures per epoch.

    **Sigma comes from the basis's reference prediction, not from this epoch's
    data.** That looks as though it should make this call jittable where
    ``compress_linear`` is not -- ``rows^T c_ref`` is concrete numbers carried on
    the dictionary, so the flag pattern is known before the data arrives. It does
    not, and the reason is omnistaging: inside a ``jax.jit`` trace *every*
    ``jnp`` call is staged into the jaxpr whether or not its operands are traced,
    so ``broadcast_to`` of a concrete sigma still returns a tracer and the guard
    fires. Under ``jax.grad`` and ``jax.vmap`` it does not, because those traces
    lift only values that are actually traced -- which is the same split
    ``compress_linear`` has, reached by a different route, and the reason the
    guard tests sigma rather than the data.

    Args:
        basis: the shared
            :class:`~rheplicant.inference.reduced_basis.ReducedBasis`.
        observed: this epoch's data.
        noise: this epoch's :class:`~rheplicant.inference.noise.NoiseModel`,
            evaluated at the basis's reference prediction. Under
            ``RadiometerNoise`` that evaluation *is* the frozen-N step of
            section 8, and ``noise_frozen_at`` is how the term says so.
        epoch_id: the recording's data hash.
        support: overrides the basis's. Refused if neither supplies one.
        noise_frozen_at: provenance. ``"none"`` for a genuinely fixed
            covariance, ``"reference"`` for a sigma frozen at the basis's
            reference prediction, ``"gls"`` for one from
            :func:`~rheplicant.inference.gls.iterative_gls`. Two terms with
            different values are different estimators and ``BayesMemory``
            refuses to sum them.
        nuisance_design: ``{nuisance latent: (n_data, n_j) block}`` for the
            per-epoch latents integrated out here. Every column must lie inside
            the basis -- pass the same block to ``build_reduced_basis`` as
            ``extra_directions=``.
        nuisance_prior_std: each nuisance latent's prior standard deviation.
            Required for every entry of ``nuisance_design``: the block is
            integrated exactly once and an improper integral is not a
            likelihood, which is condition C3.
        nuisance_prior_mean: each nuisance latent's prior mean; zero if absent.
        nuisance_shapes: each nuisance latent's shape. Defaults to the design
            block's column count as a flat vector.
        nuisance_tolerance: how far outside the basis a nuisance column may lie
            before it is refused, as a fraction of its own ``N^-1`` norm.

    Returns:
        A prior-free, approximate
        :class:`~rheplicant.inference.compressed.ReducedBasisLikelihood` over
        the basis coefficients.
    """
    observed = jnp.ravel(jnp.asarray(observed))
    reference = basis.rows.T @ basis.c_ref
    sigma, seen, weight, n_observed = _epoch_weights(
        noise, reference, observed, "compress_reduced_basis"
    )
    support = support if support is not None else basis.support
    if not support:
        raise StateValidationError(
            "compress_reduced_basis needs a support: the region the training bank "
            "populated, which is the only region this basis's fidelity was ever "
            "measured in. Pass support= to build_reduced_basis so it travels with "
            "the dictionary, or to this call."
        )

    # SELECT on `seen`, never multiply by a zero weight: a flagged sample is
    # usually flagged BECAUSE it holds a NaN, and `0.0 * nan` is `nan`. The
    # poison reaches `target` and `offset` while `factor` stays finite, so every
    # density is NaN while audit() reports a well-conditioned campaign.
    residual = jnp.where(seen, observed - reference, 0.0) * weight
    design = jnp.where(seen[None, :], basis.rows, 0.0).T * weight[:, None]
    width = design.shape[1]
    # The masked normalisation, +200.738 nats on the RHINO fixture. A pure
    # offset, which is why it is compared absolutely and never inferred from a
    # posterior's shape.
    normalisation = -0.5 * jnp.sum(
        jnp.where(seen, jnp.log(2.0 * jnp.pi * jnp.where(seen, sigma, 1.0) ** 2), 0.0)
    )

    nuisance_design = dict(nuisance_design or {})
    nuisance_names = tuple(nuisance_design)
    nuisance_shapes = dict(nuisance_shapes or {})
    nuisance_prior_std = dict(nuisance_prior_std or {})
    nuisance_prior_mean = dict(nuisance_prior_mean or {})
    missing = [name for name in nuisance_names if name not in nuisance_prior_std]
    if missing:
        raise StateValidationError(
            f"Nuisance latent(s) {missing} are integrated out here, so their prior "
            "is part of the model rather than an optional regulariser (condition "
            "C3): without it the integral over the block diverges, and finite "
            "arithmetic returns a large plausible number rather than an infinity "
            "anything downstream would notice. Give each one a nuisance_prior_std."
        )

    columns, priors, targets, log_std, shapes = [], [], [], [], []
    for name in nuisance_names:
        block = jnp.reshape(jnp.asarray(nuisance_design[name]), (observed.shape[0], -1))
        # Section 4.2(b): the basis must SPAN the phi directions, or the epoch's
        # nuisance is being integrated against a model that cannot express it and
        # the residual silently reappears as signal -- with the chi-square, the
        # conditioning and the bank reproduction all clean.
        for index in range(block.shape[1]):
            fraction = float(basis.residual_fraction(block[:, index]))
            if not fraction < nuisance_tolerance:
                raise StateValidationError(
                    f"Nuisance column {name}[{index}] lies {fraction:.3e} outside "
                    f"this basis, against a tolerance of {nuisance_tolerance}. "
                    "Section 4.2(b) integrates phi_e by expanding it in the SAME "
                    "dictionary, so a direction the basis cannot represent is one "
                    "the marginalisation cannot remove -- it would reappear as "
                    "signal. Pass the nuisance design to build_reduced_basis as "
                    "extra_directions=."
                )
        size = int(block.shape[1])
        shapes.append(nuisance_shapes.get(name, (size,)))
        std = jnp.broadcast_to(jnp.asarray(nuisance_prior_std[name]), (size,))
        mean = jnp.broadcast_to(jnp.asarray(nuisance_prior_mean.get(name, 0.0)), (size,))
        # Select, then multiply. Same rule as the residual above.
        columns.append(jnp.where(seen[:, None], block, 0.0) * weight[:, None])
        priors.append(jnp.diag(1.0 / std))
        targets.append(mean / std)
        log_std.append(jnp.sum(jnp.log(std)))

    if nuisance_names:
        nuisance_block = jnp.concatenate(columns, axis=1)
        n_nuisance = nuisance_block.shape[1]
        # Triangularise the data rows first so the stored joint is O((n_phi +
        # n_S)^2) rather than O(n_data * n_S) -- storing the raw rows would
        # reintroduce exactly the n_data dependence this tier removes. The prior
        # rows are appended afterwards and carry their own normalisation, which
        # is the caller's half of the constant `marginalise` deliberately does
        # not supply.
        upper = jnp.linalg.qr(
            jnp.concatenate([nuisance_block, design, residual[:, None]], axis=1),
            mode="r",
        )
        joint = SqrtInfo(
            factor=jnp.concatenate(
                [
                    upper[:, : n_nuisance + width],
                    jnp.concatenate(
                        [
                            jax.scipy.linalg.block_diag(*priors),
                            jnp.zeros((n_nuisance, width)),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            ),
            target=jnp.concatenate(
                [upper[:, n_nuisance + width], jnp.concatenate(targets)]
            ),
            offset=(
                normalisation
                - sum(log_std)
                - 0.5 * n_nuisance * jnp.log(2.0 * jnp.pi)
            ),
            names=(*nuisance_names, COEFFICIENTS),
            shapes=(*shapes, (width,)),
        )
        info = marginalise(joint, nuisance_names)
    else:
        upper = jnp.linalg.qr(
            jnp.concatenate([design, residual[:, None]], axis=1), mode="r"
        )
        keep = min(upper.shape[0], width)
        # The corner is the part of the residual no coefficient can reach. It is
        # a constant in theta, so leaving it out changes no gradient and no
        # curvature and every density by the same amount -- measured at -51.321
        # nats on the RHINO fixture, which is why Plan A's dropped corner
        # survived a probe built on shapes.
        corner = upper[keep:, width]
        joint = None
        info = SqrtInfo(
            factor=upper[:keep, :width],
            target=upper[:keep, width],
            offset=normalisation - 0.5 * jnp.sum(corner**2),
            names=(COEFFICIENTS,),
            shapes=((width,),),
        )

    return ReducedBasisLikelihood(
        basis=basis,
        info=info,
        joint=joint,
        epoch_id=epoch_id,
        n_observed=n_observed,
        support=dict(support),
        nuisance_names=nuisance_names,
        nuisance_shapes=tuple(shapes),
        noise_frozen_at=noise_frozen_at,
    )


def compress(
    observed: jax.Array,
    epoch_id: str,
    *,
    design: Mapping[str, jax.Array] | None = None,
    basis: Any = None,
    noise: Any = None,
    noise_std: Any = None,
    shapes: Mapping[str, tuple[int, ...]] | None = None,
    **tier_arguments: Any,
) -> Any:
    """Route one epoch to the tier that can represent it -- section 4's ladder.

    Two methods exist today and the choice is structural rather than a
    heuristic: a model affine in every latent has a sufficient statistic (T2),
    and one that is not needs a dictionary to be expanded in (T1). Anything
    else -- an emulator (T3), a Monte Carlo integral (T4) -- is refused by name
    rather than approximated by whichever of these two is nearer, because a
    finite plausible number from the wrong tier is exactly the failure the tier
    ladder exists to prevent.

    Being a dispatcher makes this subject to boundary validation: the tiers must
    agree at every threshold and at extreme parameter values, which
    ``tests/evidence/test_tier_boundaries.py`` checks by calling each tier
    **directly** rather than through this function -- routing one input to one
    method can only show the function's own continuity, never that two methods
    agree.

    Args:
        observed: this epoch's data.
        epoch_id: the recording's data hash.
        design: ``{latent: block}`` -- the claim that the model is affine in
            every latent. Routes to :func:`compress_linear`.
        basis: a :class:`~rheplicant.inference.reduced_basis.ReducedBasis` --
            the claim that the prediction is expanded in this dictionary. Routes
            to :func:`compress_reduced_basis`.
        noise: the noise model, for the reduced route.
        noise_std: the noise scale, for the linear route.
        shapes: each latent's shape, for the linear route.
        **tier_arguments: passed through to whichever tier was chosen, so a
            keyword the tier does not take is a TypeError naming that tier
            rather than a silently ignored argument here.

    Raises:
        StateValidationError: if neither a design nor a basis is given, or if
            both are.
    """
    if design is not None and basis is not None:
        raise StateValidationError(
            "compress was given both a design and a basis. They are different "
            "claims about the same model -- 'affine in every latent' and 'expanded "
            "in this dictionary' -- and silently preferring one would hide a "
            "modelling disagreement inside a routing decision. Pass exactly one."
        )
    if design is not None:
        return compress_linear(
            design=design,
            observed=observed,
            noise_std=noise_std,
            shapes=shapes or {},
            epoch_id=epoch_id,
            **tier_arguments,
        )
    if basis is not None:
        return compress_reduced_basis(
            basis, observed=observed, noise=noise, epoch_id=epoch_id, **tier_arguments
        )
    raise StateValidationError(
        "compress needs either design= (T2: the model is affine in every latent, "
        "and the term is a sufficient statistic) or basis= (T1: the prediction is "
        "expanded in a reduced basis). T3 (EmulatedLikelihood) and T4 "
        "(SampledLikelihood) are named in the design and are not built, so a model "
        "that is neither is refused here rather than routed to whichever of the two "
        "is nearer."
    )
