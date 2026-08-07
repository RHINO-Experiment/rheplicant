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
    RawLikelihood,
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


def _reject_bad_templates(
    templates: Mapping[str, jax.Array], seen: jax.Array
) -> None:
    """A template is model input, and this is the one place it is checked.

    Shape and finiteness together, because they are the same guard: both say
    "this array cannot be projected onto, and here is which one and why".
    Splitting them would leave two messages to keep in step and two places for a
    third caller to miss one.

    **Finiteness, and why refusing beats sanitising.** The projection selected
    on ``norm > 0.0``, and that comparison is **False for NaN**. So a NaN
    anywhere in an unflagged sample took the same branch as a template lying
    entirely inside the design's column space, and both were reported as
    exactly ``0.0`` -- the value :func:`_residual_summary` documents as the
    genuine null. Measured on this fixture's whitened expression: healthy
    ``norm=1.3136``, projection ``-0.4957``; all-zero ``norm=0.0000``,
    projection ``+0.0000``; one NaN ``norm=nan``, projection ``+0.0000``. End
    to end over 200 epochs carrying a real common mode, one NaN took
    ``gain_ripple`` from ``z = 30.49`` to ``z = 0.0`` while ``chi2_z`` stayed at
    18.542. Nothing downstream could see it: ``0.0`` passes every finiteness
    guard there is, and ``QuadraticLikelihood.__check_init__`` checks
    ``info.factor``, ``info.target`` and ``info.offset`` for dtype without ever
    reading ``template_projections``.

    ``inf`` is refused by the same line and for the same reason, not as a
    tidiness measure: it looks like the loud case and is not. ``projector @
    column`` is infinite too, ``inf - inf`` is NaN, so an infinite template
    reaches ``norm=nan`` and reports ``+0.0000`` by exactly the route a NaN
    does.

    **Only unflagged samples.** A template built from the product that caused
    the flag is non-finite precisely where the data is, which is the normal
    case and is what ``test_a_flagged_epoch_summarises_what_it_saw_rather_than_a_nan``
    pins. ``jnp.where(seen, ...)`` removes those before any arithmetic, so
    refusing them would be a second bug in the other direction.

    **Shape.** Never checked at all before this. A scalar broadcast to a
    constant vector and was reported as a legitimate projection of the named
    systematic -- on one epoch of this fixture the real ``gain_ripple`` gives
    ``-0.4957`` and a scalar ``1.0`` gives ``-1.1330``, a louder detection of
    something nobody supplied, and ``3.0`` gives the same number because a
    constant vector has one direction whatever its length. A wrong length
    raised ``Incompatible shapes for broadcasting: shapes=[(8,), (5,), ()]``,
    out of jax, naming no template and no remedy.

    A template is caller-supplied *model* input, not data, so the remedy is the
    caller's: fix the array, or flag the sample. Neither is something this
    function may choose on the caller's behalf.
    """
    for name, template in templates.items():
        # `bool` on a concrete array: `_refuse_traced_sigma` has already refused
        # jit, and grad/vmap trace `observed`, never a template. A caller who
        # maps over templates gets a tracer error here rather than a wrong
        # answer, which is the safe side of that trap.
        column = jnp.ravel(jnp.asarray(template))
        if column.shape != seen.shape:
            raise StateValidationError(
                f"Template {name!r} has {column.size} entries, but this epoch has "
                f"{seen.size} samples, so it is not a usable systematic template. "
                "A scalar is the trap this catches: it broadcasts to a constant "
                "vector and is reported as a projection of the named systematic, "
                "which on a real epoch reads louder than the true template does. "
                "Pass the template sampled on the epoch's own time-frequency "
                "grid, in the model's units."
            )
        if bool(jnp.any(seen & ~jnp.isfinite(column))):
            bad = [int(i) for i in jnp.flatnonzero(seen & ~jnp.isfinite(column))]
            raise StateValidationError(
                f"Template {name!r} is not a usable systematic template: it holds "
                f"a non-finite value at unflagged sample(s) {bad[:8]}. This is "
                "refused rather than repaired because the only repair available "
                "here reports the projection as exactly 0.0, and 0.0 is this "
                "summary's documented value for a template lying entirely inside "
                "the design's column space -- a genuine null. Measured, one NaN "
                "turned a 30.5-sigma detection into z = 0.0 with the chi-square "
                "half of the same summary still firing. A template is model "
                "input, not data: fix the bad channel, or mask that sample with "
                "FlaggedNoise, after which the template may be non-finite there "
                "and this epoch summarises what it saw."
            )


def _residual_summary(
    whitened_design: jax.Array,
    whitened_residual: jax.Array,
    seen: jax.Array,
    weight: jax.Array,
    templates: Mapping[str, jax.Array] | None,
) -> tuple[jax.Array, int, tuple[str, ...], jax.Array | None]:
    """Section 9.3's ~100 bytes: chi-square, DOF, and named template projections.

    The residual is taken **after** the epoch's own best fit, so what is left is
    the part no value of theta could have absorbed. That is the half of a
    coherent error a diagnostic can see: the in-span half biases theta
    identically in every epoch and leaves nothing here at all, which is why
    sections 9.4 and 9.5 are refusals based on a declaration rather than reports
    based on the data.

    Under the null the projections are standard normal -- the whitened residual
    has unit-variance components by construction -- so a campaign's mean over
    epochs is a z-score at sqrt(N) with no calibration step.

    Templates are projected onto the part of themselves that is also out of
    span. A template lying entirely inside the design's column space projects to
    exactly zero and says so, rather than reporting a small number that reads
    like a null result. That reading of ``0.0`` is exclusive, which is what
    :func:`_reject_bad_templates` buys: a non-finite or wrongly shaped template
    is refused by name here rather than arriving downstream wearing the null's
    own value.

    **``whitened_design`` must carry every column the epoch fits, nuisances
    included.** The plan wrote this call with the *global* block alone, which is
    wrong wherever ``nuisance_design=`` is used and wrong in a way no shape
    would catch: the nuisance's contribution stays in the residual, so the
    chi-square is inflated by whatever the nuisance explained while the DOF is
    over-counted by its rank. Measured over 4000 clean epochs of a two-global
    design with a three-column nuisance: including the nuisance gives
    ``dof = 3`` and a mean chi-square of ``3.0000``, and excluding it gives
    ``dof = 6`` against a mean chi-square of ``28.44`` -- a nine-sigma-per-epoch
    detection of nothing at all, on data with no fault in it.

    The projector is the flat-prior one: everything the columns *could* explain
    is removed, where the marginalisation itself is taken against a proper
    prior. That is the conservative direction -- it can only make a template
    quieter, never louder -- and it is what makes the null exact rather than
    prior-dependent.
    """
    projector = whitened_design @ jnp.linalg.pinv(whitened_design)
    perpendicular = whitened_residual - projector @ whitened_residual
    chi2 = jnp.sum(perpendicular**2)
    dof = int(jnp.sum(seen)) - int(jnp.linalg.matrix_rank(whitened_design))
    names = tuple(templates or ())
    if not names:
        return chi2, dof, (), None
    # Before any arithmetic, so that `norm > 0.0` below decides ONE question.
    _reject_bad_templates(templates, seen)
    rows = []
    for name in names:
        # SELECT on `seen` before weighting: a template is supplied in the
        # model's own units and a flagged sample is usually flagged because it
        # holds a NaN, and `0.0 * nan` is `nan`.
        column = jnp.where(seen, jnp.ravel(jnp.asarray(templates[name])), 0.0) * weight
        column = column - projector @ column
        norm = jnp.linalg.norm(column)
        # `norm > 0` is now the in-span test and nothing else -- the guard above
        # has already refused every template that could reach here non-finite.
        # It used to be both, and `norm > 0.0` is False for NaN, so a broken
        # template and a null result were reported as the same 0.0.
        safe = jnp.where(norm > 0.0, norm, 1.0)
        rows.append(jnp.where(norm > 0.0, column @ perpendicular / safe, 0.0))
    return chi2, dof, names, jnp.stack(rows)


def _sorted_inputs(inputs: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    """``{product: hash}`` as a sorted tuple of pairs -- hashable, and static.

    Sorted so that two epochs built from the same products compare equal
    whatever order the caller happened to write them in. Section 9.5's refusal
    is an equality test on these pairs, and an order-dependent one would let two
    nights that share a calibration solution be summed as independent.
    """
    return tuple(sorted((inputs or {}).items()))


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
    templates: Mapping[str, jax.Array] | None = None,
    inputs: Mapping[str, str] | None = None,
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
        templates: ``{name: (n_data,) shape}`` -- named systematic templates the
            epoch's residual is projected onto, in the model's own units.
            Section 9.3, and it can only be done here: ``audit()`` runs after
            the recording is archived, and the residual is gone by then.
        inputs: ``{product: content hash}`` for the shared products this epoch
            was built from -- one calibration solution, one beam model, one flag
            table. Section 9.5's conditional-independence refusal reads these,
            and a campaign that has forgotten which nights shared a solution
            will cheerfully sum them.

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

    fitted_block = global_block
    if nuisance_names:
        nuisance_block = (
            jnp.where(seen[:, None], _stack(nuisance_design, nuisance_names), 0.0)
            * weight[:, None]
        )
        # Every column the epoch fits, for section 9.3's projector. A nuisance
        # marginalised here is part of this night's best fit, so leaving it out
        # would report as unexplained a residual the model did explain.
        fitted_block = jnp.concatenate([global_block, nuisance_block], axis=1)
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
    chi2, dof, template_names, projections = _residual_summary(
        fitted_block, residual, seen, weight, templates
    )
    return QuadraticLikelihood(
        info=info,
        epoch_id=epoch_id,
        n_observed=n_observed,
        exact=True,
        residual_chi2=chi2,
        template_projections=projections,
        residual_dof=dof,
        template_names=template_names,
        inputs=_sorted_inputs(inputs),
    )


def _reference_values(basis: Any) -> dict[str, jax.Array]:
    """The latent values the basis's reference prediction was taken at.

    Read off the basis rather than re-derived, so that section 7's bias
    gradient and the storage origin every epoch's statistics are centred on are
    the same point by construction rather than by two callers agreeing to use
    the same one. ``ReducedBasis`` records them for exactly this reason.
    """
    if not basis.reference_values:
        raise StateValidationError(
            "This basis records no reference_values, so it was built by hand "
            "rather than by build_reduced_basis. The bias gradient must be taken "
            "at the point the storage origin was taken at; taken anywhere else it "
            "is a finite, plausible number about nothing -- it would report the "
            "slope of the compression error somewhere the epoch's statistics are "
            "not centred, which is neither the truncation nor the metric "
            "mismatch. Pass reference_values= when constructing the basis."
        )
    return dict(basis.reference_values)


def _evaluate(info: SqrtInfo, basis: Any, values: dict[str, jax.Array]) -> jax.Array:
    """T1's density from the pieces, before the term object exists."""
    return info.log_prob({COEFFICIENTS: basis.coefficients(values) - basis.c_ref})


def _support_corners(
    values: Mapping[str, jax.Array], support: Mapping[str, tuple[float, float]]
) -> list[dict[str, jax.Array]]:
    """The centre, and each latent pushed to each end of its declared box.

    ``2 n_theta + 1`` probes rather than the ``2^n_theta`` corners: the cost is
    linear in the parameter count, and a per-axis extreme is what a frozen
    covariance is actually sensitive to -- sigma is proportional to the
    prediction, so it is the latent that scales the prediction that moves it,
    not a joint excursion of all of them. Named here rather than inlined so the
    boundary-validation task can use the same points.

    **The edge is broadcast to the latent's own shape, not substituted for it.**
    A declared box is one interval per *latent*, so pushing a latent of shape
    ``(3,)`` to an edge means all three components at that edge. Writing the
    bare scalar in its place changes the probe's shape, and the failure is not
    a wrong number -- it is ``matmul input operand 1 must have ndim at least
    1`` out of the forward model, raised from inside section 8's frozen-noise
    measurement, several frames from anything the caller wrote. Every latent in
    ``tests/evidence/rhino_bank.py`` is a scalar, where ``broadcast_to(edge,
    ())`` and ``asarray(edge)`` are the same array, which is why this survived
    until a boundary fixture declared a vector latent.
    """
    probes = [dict(values)]
    for name, (low, high) in support.items():
        if name not in values:
            continue
        probes.extend(
            {**values, name: jnp.broadcast_to(jnp.asarray(edge), jnp.shape(values[name]))}
            for edge in (low, high)
        )
    return probes


def _frozen_noise_residual(
    basis: Any,
    observed: jax.Array,
    sigma: jax.Array,
    noise: Any,
    values: Mapping[str, jax.Array],
    support: Mapping[str, tuple[float, float]],
    epoch_id: str,
) -> jax.Array:
    """What freezing ``N`` cost this epoch, in nats, over the declared support.

    Section 8 makes the correction mandatory *with a measured residual*, and
    recording ``noise_frozen_at`` is provenance rather than detection: it says
    the covariance was frozen, not what freezing did.

    **Frozen T0 against live T0, not T1 against live T0.** The projection error
    is section 7's, measured as a gradient and budgeted there; folding it in
    here would inflate this number by something a larger ``n_basis`` fixes and
    make the refusal below name the wrong remedy. Measured on the RHINO fixture
    under a constant sigma, where freezing costs exactly nothing: this returns
    ``0.0`` while T1 still differs from the raw likelihood by 8.2e-6 nats.

    Computed for **every** noise model rather than behind a branch on
    ``depends_on_prediction``. A branch would make the zero a skipped code path
    instead of an arithmetic fact, and the test for that claim would then pass
    with the whole measurement deleted.

    **Returned as a traced scalar, and stored as a dynamic leaf.** Casting it to
    a Python float here would concretise the data, which breaks ``jax.grad`` and
    ``jax.vmap`` over ``observed`` -- the two traces this module documents as
    working, and which ``_refuse_traced_sigma`` was deliberately narrowed to
    keep. Measured: the float cast turned both of
    ``test_reduced_basis_likelihood.py``'s trace pins into a
    ``ConcretizationTypeError``. So the reduction stays in ``jnp``, and only the
    tolerance comparison -- which genuinely needs a concrete answer -- asks for
    one.
    """
    frozen = RawLikelihood(
        predict=basis.predict,
        observed=observed,
        sigma=sigma,
        names=tuple(values),
        epoch_id=epoch_id,
    )
    gaps = []
    for probe in _support_corners(values, support):
        live = RawLikelihood(
            predict=basis.predict,
            observed=observed,
            sigma=jnp.broadcast_to(
                noise.std(basis.predict(probe)), jnp.shape(observed)
            ),
            names=tuple(values),
            epoch_id=epoch_id,
        )
        gaps.append(jnp.abs(frozen(probe) - live(probe)))
    # `jnp.max`, never the builtin `max` over floats: with a NaN among the gaps
    # the builtin's answer depends on which element the NaN happened to be, so
    # the same broken epoch would refuse or pass according to the iteration
    # order of `support`. `jnp.max` propagates the NaN; `jnp.nanmax` would hide
    # it, which is the opposite of what this measurement is for.
    return jnp.max(jnp.stack(gaps))


def _phi_marginal_raw_density(
    basis: Any,
    observed: jax.Array,
    seen: jax.Array,
    weight: jax.Array,
    nuisance_block: jax.Array,
    prior_block: jax.Array,
    prior_target: jax.Array,
) -> Any:
    """The oracle for an epoch that integrated a nuisance out, up to a constant.

    T1's stored ``info`` is a function of theta alone -- ``phi_e`` was
    marginalised out of it -- so comparing it against a raw density that still
    carries ``phi_e`` would measure the marginalisation rather than the
    compression, and the gradient section 7 stores would be about the wrong
    thing entirely. The honest oracle is the raw likelihood with the *same*
    block integrated out against the *same* prior.

    Marginalising a Gaussian over ``phi`` leaves
    ``-0.5 ||(I - P) y(theta)||^2`` plus ``-sum(log|diag R_M|)`` and the prior's
    own normalisation, where ``M`` stacks the whitened nuisance columns on the
    prior rows and ``P`` projects onto its range. Only the first term depends on
    theta; ``M`` is built from the design and the prior, neither of which moves.
    So the constants are dropped here and the returned callable is the raw
    marginal **up to an additive constant** -- which is exactly enough, because
    the only thing taken of it is a theta-gradient. Writing it any other way
    would mean re-deriving two normalisations that cancel.
    """
    stacked = jnp.concatenate([nuisance_block, prior_block], axis=0)
    orthonormal = jnp.linalg.qr(stacked, mode="reduced")[0]

    def density(values: dict[str, jax.Array]) -> jax.Array:
        prediction = jnp.reshape(basis.predict(values), observed.shape)
        # Select on `seen`, then multiply. Same rule as everywhere else here:
        # a flagged sample is usually flagged BECAUSE it holds a NaN, and
        # `0.0 * nan` is `nan`.
        residual = jnp.where(seen, observed - prediction, 0.0) * weight
        target = jnp.concatenate([residual, prior_target])
        return -0.5 * jnp.sum((target - orthonormal @ (orthonormal.T @ target)) ** 2)

    return density


def compress_reduced_basis(
    basis: Any,
    observed: jax.Array,
    noise: Any,
    epoch_id: str,
    *,
    support: dict[str, tuple[float, float]] | None = None,
    noise_frozen_at: str = "none",
    frozen_tolerance: float | None = None,
    nuisance_design: Mapping[str, jax.Array] | None = None,
    nuisance_prior_std: Mapping[str, Any] | None = None,
    nuisance_prior_mean: Mapping[str, jax.Array] | None = None,
    nuisance_shapes: Mapping[str, tuple[int, ...]] | None = None,
    nuisance_tolerance: float = 1e-6,
    templates: Mapping[str, jax.Array] | None = None,
    inputs: Mapping[str, str] | None = None,
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

    **The bias gradient is taken here because here is where the oracle still
    exists.** Section 7 budgets the theta-*gradient* of the compression error,
    not its magnitude, and evaluating that needs the epoch's raw data. This is
    the last moment the raw data is in hand, so ``bias_gradient`` is computed
    before the return and stored as ``n_theta`` floats;
    :meth:`~rheplicant.inference.memory.BayesMemory.audit` turns the campaign's
    sum of them into a bias per named direction.

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
            refuses to sum them. Not overridden for a noise model that does not
            depend on the prediction: a sigma computed by ``iterative_gls`` and
            handed back as ``HomoscedasticNoise`` is constant *and* frozen at
            the GLS solution, and ``"gls"`` is the true answer there.
        frozen_tolerance: refuse when freezing the covariance costs more than
            this many nats anywhere in ``support``. ``None`` measures and
            records without refusing.
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
        templates: ``{name: (n_data,) shape}`` -- named systematic templates the
            epoch's residual is projected onto, in the model's own units
            (section 9.3). Here "the epoch's best fit" is the best the
            *dictionary* can do, so a template lying inside the basis's span
            projects to zero: at T1 the residual summary is blind to exactly
            what the basis can absorb, which is a wider blindness than T2's and
            is the honest reading of the number.
        inputs: ``{product: content hash}`` for this epoch's shared input
            products. Section 9.5.

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

    # The storage origin, read once. Everything measured below -- the frozen-N
    # residual and section 7's bias gradient -- is anchored here, so it is
    # resolved before any of it rather than twice in two places.
    values = _reference_values(basis)

    # Section 8, mandatory WITH A MEASURED RESIDUAL. Under RadiometerNoise sigma
    # tracks the prediction (D21), so N^-1 depends on theta and the stored
    # statistics are not constants until it is frozen -- and D23 records that
    # the converged answer is then generalized least squares, not the maximum of
    # the full Gaussian likelihood. Freezing is exact at the anchor and costs
    # something away from it; that cost is measured here at the extremes of the
    # declared support, not bounded by an argument.
    frozen_residual = _frozen_noise_residual(
        basis, observed, sigma, noise, values, support, epoch_id
    )
    if frozen_tolerance is not None and isinstance(frozen_residual, jax.core.Tracer):
        raise StateValidationError(
            "frozen_tolerance cannot be enforced under a trace over the data: the "
            "residual is the epoch's own log-density gap, so it is traced too, and "
            "a comparison against it has no concrete answer to branch on. "
            "jax.grad and jax.vmap over `observed` do work here and are pinned -- "
            "what does not is refusing inside one. Drop frozen_tolerance and read "
            "term.frozen_noise_residual afterwards; the measurement is still taken "
            "and still stored."
        )
    # `not <=`, never `>`: a zero sigma makes the quadratic -inf and the
    # log-determinant +inf, so the density is NaN -- and `nan > tolerance` is
    # False, which is the one answer that lets a broken epoch past.
    if frozen_tolerance is not None and not bool(frozen_residual <= frozen_tolerance):
        raise StateValidationError(
            f"Freezing the covariance costs {float(frozen_residual):.6g} nats over the "
            f"declared support, against a tolerance of {frozen_tolerance}. Under "
            "RadiometerNoise sigma is a function of the very quantity being "
            "inferred (D21), so a frozen N makes the stored statistics constants "
            "at the price of a theta-dependent error -- and D23 records that the "
            "converged answer is then generalized least squares, not the maximum "
            "of the full Gaussian likelihood. This number is the freezing cost "
            "alone; the projection error is section 7's bias budget and a larger "
            "n_basis is its remedy, not this one. Narrow the support, re-anchor "
            "the basis nearer the accumulated posterior, or declare a larger "
            "tolerance knowing this is a coherent error across every epoch."
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

    nuisance_block = jnp.concatenate(columns, axis=1) if columns else None
    if nuisance_names:
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

    # Section 7: what the posterior sees of the compression error is its
    # theta-GRADIENT. A constant offset has exactly zero effect; a small
    # theta-dependent tilt has unbounded effect, and because one basis, one
    # instrument model and one estimator serve every epoch the tilt is coherent,
    # so the induced bias -F^-1 grad(delta) is N-independent while sigma_N falls
    # as N^-1/2 and the ratio grows as sqrt(N). One JVP and n_theta floats, and
    # it can only be taken here -- the oracle needs the raw data, and the next
    # line of the campaign releases it.
    bias_names = tuple(sorted(values))
    if nuisance_names:
        oracle = _phi_marginal_raw_density(
            basis,
            observed,
            seen,
            weight,
            nuisance_block,
            jax.scipy.linalg.block_diag(*priors),
            jnp.concatenate(targets),
        )
    else:
        oracle = RawLikelihood(
            predict=basis.predict,
            observed=observed,
            sigma=sigma,
            names=tuple(values),
            epoch_id=epoch_id,
            noise_frozen_at=noise_frozen_at,
        )

    def _fidelity_residual(chosen: dict[str, jax.Array]) -> jax.Array:
        probe = {**values, **chosen}
        return _evaluate(info, basis, probe) - oracle(probe)

    gradient = jax.grad(_fidelity_residual)(dict(values))
    chi2, dof, template_names, projections = _residual_summary(
        design if nuisance_block is None
        else jnp.concatenate([design, nuisance_block], axis=1),
        residual,
        seen,
        weight,
        templates,
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
        frozen_noise_residual=frozen_residual,
        bias_gradient=jnp.concatenate(
            [jnp.ravel(gradient[name]) for name in bias_names]
        ),
        bias_names=bias_names,
        residual_chi2=chi2,
        template_projections=projections,
        residual_dof=dof,
        template_names=template_names,
        inputs=_sorted_inputs(inputs),
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
