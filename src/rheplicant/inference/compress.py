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
from rheplicant.inference.compressed import QuadraticLikelihood
from rheplicant.inference.sqrtinfo import SqrtInfo
from rheplicant.inference.uncertainty import as_noise_model


def _stack(design: Mapping[str, jax.Array], names: tuple[str, ...]) -> jax.Array:
    """Column-concatenate the per-latent design blocks in declared order."""
    return jnp.concatenate([jnp.atleast_2d(design[name]) for name in names], axis=1)


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
    sigma = jnp.broadcast_to(noise.std(prediction), observed.shape)
    if bool(jnp.any(jnp.isnan(sigma))):
        raise StateValidationError(
            "noise_std produced NaN for at least one sample. `inf` is this "
            "package's encoding of 'not observed' (FlaggedNoise); NaN is a "
            "broken noise model, and the two are only the same to "
            "`jnp.isfinite`. Left to that test the sample would be counted as "
            "flagged, silently shrinking n_observed and building the epoch's "
            "sufficient statistic on the rest -- finite, self-consistent, and "
            "quietly short of data. Fix the noise model, or mask the sample "
            "explicitly with FlaggedNoise."
        )
    seen = jnp.isfinite(sigma)
    weight = jnp.where(seen, 1.0 / jnp.where(seen, sigma, 1.0), 0.0)
    n_observed = int(jnp.sum(seen))

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
