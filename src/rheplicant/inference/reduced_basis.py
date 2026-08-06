"""A reduced basis over PARAMETER space -- not the smooth basis over the data grid.

Two different objects share the word. :mod:`rheplicant.core.basis` holds
:class:`~rheplicant.core.basis.SeparableBasis`: Legendre/Fourier columns in time
and frequency, the repair an identifiability refusal names (D28). This module
holds the other one: a dictionary of *snapshots of the prediction itself*, in
which ``mu_e(theta) ~ sum_k c_k(theta) s_k`` -- the reduced-basis surrogate of
Field, Galley, Hesthaven, Kaye & Tiglio (2014, PRX 4, 031006). ROQ proper
(Canizares et al. 2013, 2015) retains model calls at EIM nodes and is a
different scheme; the special case ``s_k = d mu / d theta_k`` at a fiducial is
MOPED (Heavens, Jimenez & Lahav 2000) and score compression (Alsing & Wandelt
2018).

**The metric is applied once, at construction.** The rows are kept in the
model's own units and the whitened copy ``rows * weight`` is derived beside
them, so "orthonormal in the ``N^-1`` metric the likelihood uses" is just
"orthonormal", the Gram matrix is the identity, and the projector is a matrix
product. That is not a convenience: with any other convention the projector is
not self-adjoint in the likelihood's own inner product, the truncation residual
stops being orthogonal to the span, and the score at the truth acquires a term
that does not vanish -- a *bias*, not a loss of sensitivity. Both copies are
stored because neither alone is enough: the projector needs the whitened one,
and an epoch whose flag pattern differs from the reference's needs the raw one,
since ``whitened / weight`` is infinite at exactly the samples the reference
could not see. There is no inverse back to unwhitened data there, and that is
what ``weight = 0`` means rather than a limitation.

**Selection and basis are different things.** ``select_svd`` and
``select_greedy`` choose *candidates*; :func:`orthonormalise` turns candidates
into a basis. Storing raw candidates is what gives a Gram matrix no float64
quadratic form survives -- ``c^T G c`` then returns a finite and occasionally
negative number.

**The science direction is seeded, not hoped for.** Singular values of a bank of
prior draws order modes by prior-induced amplitude, and at RHINO's band the
foreground spread (~200 K) sits three to four orders above the 21 cm trough
(~0.2 K), so the direction the campaign exists to measure is the last retained
and the first dropped. ``score_directions`` puts ``d mu / d theta_j`` for every
named global latent into the candidate set first, which repairs it by
construction rather than by choosing ``n_S`` large enough.
"""

import hashlib
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import solve_triangular

from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference.uncertainty import FlatMatrix, _named_spans


def orthonormal_transform(
    candidates: jax.Array, rtol: float | None = None
) -> tuple[jax.Array, tuple[int, ...]]:
    """``(M, kept)`` with ``M @ candidates`` orthonormal, in candidates' own metric.

    Modified Gram-Schmidt with one reorthogonalisation pass, in numpy, because
    the *transform* is what the caller needs and a QR only hands back the
    result. ``M`` is what lets the same combination be applied to the **raw**
    rows: a basis has to be usable by an epoch whose flag pattern differs from
    the reference's, and ``whitened / weight`` is infinite at exactly the
    samples the reference could not see.

    Order is preserved -- ``span(row_1..row_j)`` is nested -- which is the
    property seeding depends on: a score direction placed first survives
    whatever the later candidates do.

    Args:
        candidates: ``(k, n_data)`` rows, already whitened.
        rtol: drop a direction whose residual norm falls below
            ``rtol * max_row_norm``. Defaults to ``sqrt(eps)``.

    Returns:
        ``(M, kept)``: ``M`` is ``(r, k)`` with ``r <= k``, and ``kept`` names
        which candidate indices survived. ``r < k`` means the candidate set was
        rank-deficient -- a fact about the model, which the caller decides about.
    """
    whitened = np.asarray(candidates, dtype=np.float64)
    count = whitened.shape[0]
    if rtol is None:
        rtol = float(np.sqrt(np.finfo(np.float64).eps))
    scale = float(np.max(np.linalg.norm(whitened, axis=1))) if count else 0.0
    cut = rtol * scale
    vectors = np.zeros((0, whitened.shape[1]))
    transform = np.zeros((0, count))
    kept: list[int] = []
    for index in range(count):
        vector, row = whitened[index].copy(), np.eye(count)[index].copy()
        for _ in range(2):  # twice is enough: MGS alone loses orthogonality here
            for position in range(len(kept)):
                overlap = float(vectors[position] @ vector)
                vector = vector - overlap * vectors[position]
                row = row - overlap * transform[position]
        norm = float(np.linalg.norm(vector))
        if norm <= cut:
            continue
        vectors = np.vstack([vectors, vector / norm])
        transform = np.vstack([transform, row / norm])
        kept.append(index)
    return jnp.asarray(transform), tuple(kept)


def orthonormalise(candidates: jax.Array, rtol: float | None = None) -> jax.Array:
    """The orthonormal rows themselves, when the transform is not needed."""
    transform, _ = orthonormal_transform(candidates, rtol)
    return transform @ candidates


def score_directions(
    space: Any,
    pipeline: Any,
    state_template: Any,
    names: Sequence[str] | None = None,
    at: dict[str, jax.Array] | None = None,
) -> dict[str, jax.Array]:
    """``d mu / d theta_j``, one row per scalar degree of freedom, per named latent.

    Section 5 requirement 3, and the repair for the SVD's blindness: singular
    values of a prior bank order modes by prior-induced amplitude, so a
    direction whose amplitude is a thousandth of the foreground's is the last
    retained and the first dropped. Seeding these makes each parameter's
    signature present by construction, at any ``n_S``.

    Built on :meth:`~rheplicant.inference.parameters.ParameterSpace.forward_fn`
    rather than :func:`~rheplicant.inference.forward.build_forward_fn`, which
    returns a ghost pipeline whose leaves carry no latent names -- and "for
    every *named* global latent" is the whole requirement.

    Args:
        space: the :class:`~rheplicant.inference.parameters.ParameterSpace`.
            Validated against the pipeline by ``forward_fn``.
        pipeline: the forward model.
        state_template: the state it is evaluated on.
        names: which latents. ``None`` means all of them.
        at: where to differentiate. Latents are a local property of a nonlinear
            model, so this is the question "what does the prediction look like
            *here*"; defaults to the declared initial values.

    Returns:
        ``{name: (size, n_data)}``, ravelled over the data axis, in the order
        ``names`` asked for -- or the space's *declared* order. A latent of
        shape ``(4,)`` contributes four rows.

    Raises:
        ParameterSpaceError: if ``names`` or ``at`` mentions a latent the space
            does not declare.
    """
    forward, values = space.forward_fn(pipeline, state_template)
    values = dict(values)
    for supplied, label in ((at or {}, "at"), ({n: None for n in names or ()}, "names")):
        unknown = [name for name in supplied if name not in values]
        if unknown:
            raise ParameterSpaceError(
                f"score_directions was given {label}={sorted(unknown)}, which this "
                f"space has not declared. Declared: {sorted(values)}."
            )
    values.update({name: jnp.asarray(value) for name, value in (at or {}).items()})

    selected = tuple(values) if names is None else tuple(names)
    fixed = {name: value for name, value in values.items() if name not in selected}

    def only(chosen: dict[str, jax.Array]) -> jax.Array:
        return jnp.ravel(forward({**fixed, **chosen}))

    jacobian = jax.jacfwd(only)({name: values[name] for name in selected})
    # Iterate `selected`, never `jacobian.items()`. jax rebuilds a dict from its
    # flattened form, which is SORTED, so returning the jacobian's own order
    # would silently hand back alphabetical names -- and every consumer that
    # zips this against a declared-order list would then be wrong by a
    # permutation that is the identity exactly when the latents happen to be
    # named alphabetically. That is Plan A's BayesMemory.fisher() bug verbatim.
    return {
        name: jnp.reshape(jacobian[name], (jacobian[name].shape[0], -1)).T
        for name in selected
    }


class ReducedBasis(eqx.Module):
    """A dictionary of directions in the data space, shared by every epoch.

    Attributes:
        rows: ``(n_S, n_data)`` directions in the model's own units. Stored raw
            rather than only whitened because an epoch's flag pattern need not
            match the reference's, and ``whitened / weight`` is infinite at
            exactly the samples the reference could not see.
        weight: ``(n_data,)`` reference ``1/sigma``, exactly ``0`` where the
            reference epoch had nothing to say. The *reference* metric: each
            epoch stores its own ``G_e`` against these rows rather than
            re-orthonormalising, so that ``c(theta)`` is one map for the whole
            campaign and evaluation costs ``O(n_S^2)`` per epoch instead of
            ``O(n_S * n_data)``. The mismatch that buys is measured, per epoch,
            by the bias budget (section 7).
        whitened: ``rows * weight`` where the weight is non-zero and exactly
            ``0.0`` where it is not -- a select, not a product, because
            ``0.0 * inf`` is ``nan``. Orthonormal when :attr:`orthonormal`,
            which is the supported case. Derived at construction and kept as a
            leaf because every projection reads it.
        factor: ``(n_S, n_S)`` upper-triangular ``R`` with ``R^T R = G``. The
            identity to roundoff once orthonormalised; kept as a leaf so a
            deliberately un-orthonormalised basis is still evaluable and its
            conditioning measurable.
        c_ref: coefficients of the reference prediction. The storage origin --
            every epoch's statistics are residual-centred on ``expand(c_ref)``,
            an exact change of variable, which is what keeps the residual
            chi-square from being the time-bandwidth product.
        predict: ``values -> prediction``. Static, and the reason a basis is
            *code plus numbers*: section 10's behaviour fingerprint exists
            because these two halves can desynchronise with no shape or dtype
            change.
        seeded: names of the latents whose score directions were placed first.
        orthonormal: whether the rows were orthonormalised.
        support: ``{latent: (low, high)}`` -- the region the training bank
            populated. Section 5 requirement 6: the *projection* error is
            uniformly bounded over the prior box, but the fitted coefficient map
            is a surrogate whose error is measured on draws from the same prior,
            so it is a prior-weighted in-distribution average and blind to a
            sparsely-sampled corner the accumulated posterior may concentrate
            in. The guard is therefore worded as *training-bank coverage*, and
            it lives on the basis rather than on the caller because "the region
            the bank populated" is a fact about the bank.
        reference_values: the latent values ``reference`` was taken at. Recorded
            so the bias gradient (section 7) and the storage origin are the same
            point by construction, rather than two callers agreeing to use the
            same one. A **dynamic** field, unlike ``seeded`` and ``support``
            beside it: these are arrays, and equinox puts a static field into
            the *treedef*, where array ``__eq__`` decides treedef equality. With
            a scalar latent that silently works; with a latent of shape ``(4,)``
            -- which ``score_directions`` explicitly supports -- comparing two
            treedefs raises "the truth value of an array with more than one
            element is ambiguous", measured. Equinox says so itself, with "A JAX
            array is being set as static", and ``tests/core/test_basis.py`` pins
            the same rule for ``Bind.fn``.
    """

    rows: jax.Array
    weight: jax.Array
    whitened: jax.Array
    factor: jax.Array
    c_ref: jax.Array
    predict: Callable[[dict[str, jax.Array]], jax.Array] = eqx.field(static=True)
    seeded: tuple[str, ...] = eqx.field(static=True)
    orthonormal: bool = eqx.field(static=True)
    support: dict[str, tuple[float, float]] | None = eqx.field(static=True)
    reference_values: dict[str, jax.Array]

    def __init__(
        self,
        rows: jax.Array,
        weight: jax.Array,
        predict: Callable[[dict[str, jax.Array]], jax.Array],
        reference: jax.Array,
        seeded: Sequence[str] = (),
        orthonormal: bool = True,
        support: dict[str, tuple[float, float]] | None = None,
        reference_values: dict[str, jax.Array] | None = None,
    ):
        self.rows = jnp.asarray(rows)
        self.weight = jnp.asarray(weight)
        # Validated HERE and not in `__check_init__`, unlike the rest of the
        # package. `__check_init__` runs after `__init__` returns, and the two
        # derived leaves below are built out of exactly the quantities being
        # checked: a `(5, 3)` row set gives a `(3, 5)` QR factor and dies inside
        # `solve_triangular` with jax's own "inconsistent size for core
        # dimension" -- measured -- before any refusal here could be raised. A
        # refusal that the failure mode outruns is not a refusal.
        _refuse_shapes_no_projector_can_use(self.rows, self.weight)
        # Select on the weight and multiply second, never multiply alone:
        # `0.0 * inf` and `0.0 * nan` are both `nan` -- measured -- so a raw row
        # carrying a non-finite value at a sample the reference could not see
        # would whiten to NaN, poison `gram()` and every projection downstream,
        # and be refused two lines later by a message saying it fires only where
        # the weight is non-zero. The weight is exactly `0.0` there, checked
        # above, so this is the definition of "not observed" rather than a
        # repair applied to it.
        self.whitened = jnp.where(self.weight > 0.0, self.rows, 0.0) * self.weight
        _refuse_a_direction_that_is_not_there(self.whitened)
        self.predict = predict
        self.seeded = tuple(seeded)
        self.orthonormal = bool(orthonormal)
        self.support = None if support is None else dict(support)
        self.reference_values = {
            name: jnp.asarray(value) for name, value in (reference_values or {}).items()
        }
        self.factor = jnp.linalg.qr(self.whitened.T, mode="r")
        self.c_ref = self._project_whitened(self.weight * jnp.ravel(reference))

    # --------------------------------------------------------------- geometry --

    @property
    def n_basis(self) -> int:
        """``n_S`` -- how many directions this dictionary holds."""
        return int(self.whitened.shape[0])

    def gram(self) -> jax.Array:
        """``G = S N^-1 S^T`` in the reference metric. The identity, if orthonormal."""
        return self.whitened @ self.whitened.T

    def condition(self) -> float:
        """``kappa(G)``. Above ``1/eps`` the quadratic form is not computable."""
        return float(jnp.linalg.cond(self.gram()))

    def whiten(self, prediction: Any) -> jax.Array:
        """``w * mu``, ravelled -- the same vector space the rows live in."""
        return self.weight * jnp.ravel(jnp.asarray(prediction))

    def _project_whitened(self, vector: jax.Array) -> jax.Array:
        rhs = self.whitened @ vector
        return solve_triangular(
            self.factor, solve_triangular(self.factor.T, rhs, lower=True), lower=False
        )

    def project(self, prediction: Any) -> jax.Array:
        """Coefficients of the ``N^-1``-orthogonal projection of ``prediction``.

        Solving with ``G`` rather than taking a bare inner product is what keeps
        this an *orthogonal* projector when the rows are not orthonormal. When
        they are, ``G`` is the identity and the solve is free.
        """
        return self._project_whitened(self.whiten(prediction))

    def expand(self, coefficients: jax.Array) -> jax.Array:
        """``S^T c`` -- back to the whitened data vector, not to raw data.

        There is no way back to raw data at a flagged sample, where the weight
        is exactly zero. That is what "not observed" means.
        """
        return self.whitened.T @ coefficients

    def residual_fraction(self, direction: Any) -> jax.Array:
        """``||(I - Pi) d|| / ||d||`` in the ``N^-1`` metric -- section 5's ``r_j``.

        ``nan`` when the direction is identically zero in this metric: the
        prediction does not respond to whatever produced it, so there is no
        direction to be faithful to. ``basis_fidelity`` names that case rather
        than dividing by zero.
        """
        vector = self.whiten(direction)
        norm = jnp.linalg.norm(vector)
        residual = vector - self.expand(self._project_whitened(vector))
        # `where` twice, never a multiply: the inner one keeps the division
        # itself finite, because `0.0 * nan` is `nan` and a masked-out `0/0`
        # would poison the selected branch as well.
        safe = jnp.where(norm > 0.0, norm, 1.0)
        return jnp.where(norm > 0.0, jnp.linalg.norm(residual) / safe, jnp.nan)

    def coefficients(self, values: dict[str, jax.Array]) -> jax.Array:
        """``c(theta)`` -- the live half of the surrogate."""
        return self.project(self.predict(values))

    def fingerprint(self) -> str:
        """Content hash of the stored numbers, for the identity refusals.

        Two terms compressed against different dictionaries are quadratic forms
        in different vectors; summing them is not a likelihood. This is what
        ``BayesMemory`` compares. It deliberately does NOT cover ``predict``:
        the live half needs section 10's behaviour canary, which is a different
        check with a different failure mode.
        """
        payload = b"".join(
            np.asarray(leaf).tobytes()
            for leaf in (self.whitened, self.weight, self.c_ref)
        )
        return hashlib.sha256(payload).hexdigest()[:16]


def numerical_rank(whitened_bank: jax.Array) -> int:
    """Largest ``k`` with ``s_k / s_0 > sqrt(eps)`` -- section 5 requirement 5.

    Not a tuning knob. Beyond this cut the Gram matrix of the retained set is
    numerically singular in float64, and ``c^T G c`` returns a finite,
    occasionally negative number rather than raising: the spec records measured
    ``kappa(G) = 1.4e8`` at ``n_S = 4`` and ``2e16`` at ``n_S = 16`` for raw
    snapshots. ``sqrt(eps)`` rather than ``eps`` because the quadratic form
    squares the conditioning -- that is the same reason section 3 stores ``R``
    instead of ``F``.
    """
    singular = np.asarray(jnp.linalg.svd(whitened_bank, compute_uv=False))
    if singular.size == 0 or singular[0] == 0.0:
        return 0
    cut = float(np.sqrt(np.finfo(singular.dtype).eps))
    return int(np.sum(singular / singular[0] > cut))


def select_svd(whitened_bank: jax.Array, count: int) -> jax.Array:
    """The ``count`` leading right singular directions of the bank.

    **Candidates, not a basis.** They happen to be orthonormal here, which is a
    property of the SVD rather than of the pipeline; :func:`orthonormalise` is
    still what makes that a claim the code depends on.
    """
    if count <= 0:
        return jnp.zeros((0, whitened_bank.shape[1]))
    return jnp.linalg.svd(whitened_bank, full_matrices=False)[2][:count]


def select_greedy(whitened_bank: jax.Array, count: int) -> jax.Array:
    """Greedy EIM-style selection: the worst-represented draw, repeatedly.

    Returns rows *of the bank*, in the order chosen -- the selection step of
    Field/Galley/Puerrer, and nothing more. Orthonormalising them is a separate
    call because storing the raw picks is what makes ``G`` unusable.
    """
    bank = np.asarray(whitened_bank)
    if count <= 0:
        return jnp.zeros((0, bank.shape[1]))
    chosen: list[int] = []
    residual = bank.copy()
    for _ in range(min(count, bank.shape[0])):
        index = int(np.argmax(np.linalg.norm(residual, axis=1)))
        chosen.append(index)
        direction = residual[index]
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            break
        direction = direction / norm
        residual = residual - np.outer(residual @ direction, direction)
    return jnp.asarray(bank[chosen])


def build_reduced_basis(
    space: Any,
    pipeline: Any,
    state_template: Any,
    *,
    noise: Any,
    bank: jax.Array,
    n_basis: int,
    at: dict[str, jax.Array] | None = None,
    names: Sequence[str] | None = None,
    method: str = "svd",
    seed_scores: bool = True,
    support: dict[str, tuple[float, float]] | None = None,
) -> ReducedBasis:
    """Score directions first, bank directions after, orthonormalised once.

    The order is the substance. Seeding puts every named latent's signature in
    the span at any ``n_basis``; the bank then completes it with whatever else
    the prior actually produces, chosen on the *residual* after the scores so
    the leading singular direction does not simply restate the mean.

    Measured on the four-latent RHINO fixture (60-85 MHz, 128 channels, 400
    draws), the residual fraction of the ``t21_depth`` score direction against a
    plain SVD basis is ``0.562`` at ``n_S = 3``; seeding that one direction at
    the same ``n_S = 3`` takes it to ``1.5e-16``. The repair is complete, not
    incremental, and it does not depend on choosing ``n_S`` large enough.

    **Where the deletion stops depends on the bank, and both numbers are real.**
    That fixture recovers the direction unseeded by ``n_S = 4``, because four
    near-linear latents span their own tangent space once four vectors are
    allowed. A richer pre-planning bank measured ``0.3147`` at ``n_S = 3``,
    ``0.0289`` at 5, ``0.0040`` at 8 and ``0.0000`` only at 13. Seeding is what
    makes the answer independent of which of those a campaign happens to have.

    Args:
        space, pipeline, state_template: the model.
        noise: the epoch's :class:`~rheplicant.inference.noise.NoiseModel`,
            evaluated at the reference prediction to give the metric. Under
            ``RadiometerNoise`` this is the frozen-N step of section 8, and the
            caller owes it a ``noise_frozen_at`` provenance downstream.
        bank: ``(n_draws, ...)`` predictions at draws from the prior. The
            *training bank*, whose extent is the ``support`` a T1 term claims.
        n_basis: ``n_S``. Refused above the bank's numerical rank.
        at: where the reference prediction and the scores are taken.
        names: which latents to seed. ``None`` means all of them.
        method: ``"svd"`` or ``"greedy"`` for the non-seeded remainder.
        seed_scores: off only to *build the failure* the tests pin.
        support: ``{latent: (low, high)}`` covered by ``bank``. Carried onto
            every term compressed against this basis, so that "the region the
            bank populated" is recorded where it was known rather than
            re-promised by each caller.

    Raises:
        StateValidationError: if ``n_basis`` is below the number of score rows,
            above the bank's numerical rank, or if the score rows are themselves
            linearly dependent.
    """
    forward, values = space.forward_fn(pipeline, state_template)
    values = {**values, **(at or {})}
    reference = jnp.ravel(forward(values))
    sigma = jnp.ravel(jnp.broadcast_to(noise.std(reference), reference.shape))
    seen = jnp.isfinite(sigma) & (sigma > 0.0)
    weight = jnp.where(seen, 1.0 / jnp.where(seen, sigma, 1.0), 0.0)

    flat_bank = jnp.reshape(jnp.asarray(bank), (jnp.asarray(bank).shape[0], -1))
    whitened_bank = _whiten(flat_bank, weight)
    rank = numerical_rank(whitened_bank)
    if n_basis > rank:
        raise StateValidationError(
            f"n_basis={n_basis} exceeds the whitened bank's numerical rank {rank} "
            "(the largest k with s_k/s_0 > sqrt(eps)). Above it the retained Gram "
            "matrix is singular in float64 and c^T G c returns a finite, sometimes "
            "negative number instead of raising. Draw a bank that actually varies "
            "in more directions, or accept the rank as the answer to how many "
            "directions this prior has."
        )

    seeds = jnp.zeros((0, whitened_bank.shape[1]))
    seeded_names: tuple[str, ...] = ()
    if seed_scores:
        scores = score_directions(space, pipeline, state_template, names=names, at=values)
        seeded_names = tuple(scores)
        seeds = _whiten(
            jnp.concatenate([scores[name] for name in seeded_names], axis=0), weight
        )
        if seeds.shape[0] > n_basis:
            raise StateValidationError(
                f"n_basis={n_basis} is smaller than the {seeds.shape[0]} score "
                f"directions of {list(seeded_names)}. Seeding is not a suggestion "
                "that improves a truncation -- it is what puts each parameter's "
                "signature in the span at all, and a basis with room for only some "
                "of them silently deletes the rest. Raise n_basis to at least "
                f"{seeds.shape[0]}, or name fewer latents."
            )
        seed_transform, seed_kept = orthonormal_transform(seeds)
        if len(seed_kept) < seeds.shape[0]:
            raise StateValidationError(
                f"The score directions of {list(seeded_names)} are linearly "
                "dependent, so the basis cannot hold them all and the data cannot "
                "tell those latents apart in the first place. Run "
                "identifiability(space, pipeline, state) and read "
                "report.participation(0) -- it names which combination is blind."
            )
        orthonormal_seeds = seed_transform @ seeds
        whitened_bank = (
            whitened_bank - (whitened_bank @ orthonormal_seeds.T) @ orthonormal_seeds
        )

    remainder = n_basis - seeds.shape[0]
    if method == "svd":
        extra = select_svd(whitened_bank, remainder)
    elif method == "greedy":
        extra = select_greedy(whitened_bank, remainder)
    else:
        raise StateValidationError(
            f"method must be 'svd' or 'greedy', got {method!r}."
        )

    # The SAME combination is applied to the whitened candidates and to their
    # raw counterparts, rather than dividing the orthonormal rows by the weight.
    # The two agree wherever the reference observed and differ by `inf` where it
    # did not, and an epoch whose flags differ from the reference's is the
    # normal case rather than the exception.
    safe = jnp.where(weight > 0.0, weight, 1.0)
    candidates = jnp.concatenate([seeds, extra], axis=0)
    raw_candidates = candidates / safe
    transform, kept = orthonormal_transform(candidates)
    if len(kept) < candidates.shape[0]:
        raise StateValidationError(
            f"Only {len(kept)} of {candidates.shape[0]} candidate directions are "
            "independent to sqrt(eps). Storing the rest gives a Gram matrix that is "
            "singular in float64, after which c^T G c returns a finite and "
            f"sometimes negative number rather than raising. Lower n_basis to "
            f"{len(kept)}, or draw a bank that varies in more directions."
        )
    return ReducedBasis(
        rows=transform @ raw_candidates,
        weight=weight,
        predict=forward,
        reference=reference,
        seeded=seeded_names,
        support=support,
        reference_values=values,
    )


class FidelityReport(eqx.Module):
    """What a basis retains of each named latent's signature.

    Attributes:
        residuals: ``{name: r_j}`` with
            ``r_j = ||(I - Pi) dmu/dtheta_j||_{N^-1} / ||dmu/dtheta_j||_{N^-1}``.
            ``nan`` where the prediction does not respond to the latent at all.
        full: the Fisher of the score directions against the data, named rows.
        projected: the same Fisher after projection onto the basis. The
            difference between the two is what the truncation cost, and D14's
            named-row rendering is what attaches a *latent's name* to a
            collapsed eigenvalue -- a scalar fidelity number names no culprit,
            which is the whole reason section 5 requirement 4 exists.
    """

    residuals: dict[str, float] = eqx.field(static=True)
    full: FlatMatrix
    projected: FlatMatrix

    def worst(self) -> tuple[str, float]:
        """``(name, r_j)`` of the least faithful direction. ``nan`` sorts first."""
        items = sorted(
            self.residuals.items(),
            key=lambda item: (not np.isnan(item[1]), -item[1]),
        )
        return items[0]

    def refuse_above(self, tolerance: float) -> None:
        """Raise if any direction is worse than a declared tolerance.

        Two failures, deliberately separate messages. A direction the basis
        cannot represent is a truncation to fix; a direction that does not exist
        is a model to fix, and reporting the second as the first would send the
        caller to raise ``n_S`` against a derivative that is identically zero.
        """
        absent = sorted(
            name for name, value in self.residuals.items() if np.isnan(value)
        )
        if absent:
            raise StateValidationError(
                f"The prediction does not respond to {absent} at this point: "
                "dmu/dtheta is identically zero, so there is no direction for a "
                "basis to be faithful to and r_j is 0/0 rather than large. This is "
                "a statement about the model, not the basis. Run "
                "identifiability(space, pipeline, state), or move the expansion "
                "point with at= if the derivative merely happens to vanish here."
            )
        bad = {
            name: value for name, value in self.residuals.items() if value > tolerance
        }
        if bad:
            listed = ", ".join(
                f"{name}={value:.4f}" for name, value in sorted(bad.items())
            )
            raise StateValidationError(
                f"This basis loses {listed} against a tolerance of {tolerance}. "
                "Compression is refused: the stored term would have a collapsed "
                "Fisher eigenvalue along that direction and its marginal would "
                "revert toward the prior, while every other diagnostic -- residual "
                "chi-square, conditioning, bank reproduction -- stayed clean. "
                "Seed the score direction (build_reduced_basis does this by "
                "default), or raise n_basis; a plain SVD orders modes by "
                "prior-induced amplitude, so a signal three orders below the "
                "foreground is the last retained and the first dropped."
            )


def basis_fidelity(basis: ReducedBasis, scores: dict[str, jax.Array]) -> FidelityReport:
    """Per-direction fidelity, plus the two named Fishers -- section 5 requirement 4.

    Args:
        basis: the dictionary under test.
        scores: ``{name: (size, n_data)}`` from :func:`score_directions`.

    Returns:
        A :class:`FidelityReport`. Its matrices are rendered in **flatten**
        order, derived from the actual flattening of a template rather than from
        the order ``scores`` happens to iterate in -- ``jax`` sorts a dict's
        keys, and a matrix built in one order and labelled in the other is wrong
        by a permutation that is the identity exactly when the latents are named
        alphabetically.
    """
    template = {
        name: jnp.zeros((rows.shape[0],) if rows.shape[0] > 1 else ())
        for name, rows in scores.items()
    }
    names, spans, shapes = _named_spans(template)
    stacked = _whiten(
        jnp.concatenate([scores[name] for name in names], axis=0), basis.weight
    )
    # `_whiten` selects first, so a non-finite entry at a sample the reference
    # could not see is already an exact 0.0 here and this fires only for the case
    # the message names. Ordered before anything comparison-based, because NaN
    # defeats every comparison: `norm(nan)/norm(nan)` is nan, `nan > tolerance`
    # is False, and `refuse_above` would wave a broken forward model through
    # while reporting a perfectly conditioned Fisher.
    if not bool(jnp.all(jnp.isfinite(stacked))):
        raise StateValidationError(
            "A score direction is non-finite where the reference observed. That is "
            "a broken forward model, not a truncation to report: dmu/dtheta cannot "
            "be NaN at a sample the data constrains. Sanitising it here would turn "
            "the fault into a plausible fidelity number, so it is refused instead."
        )

    coefficients = jax.vmap(basis._project_whitened)(stacked)
    residuals = stacked - coefficients @ basis.whitened
    norms = jnp.linalg.norm(stacked, axis=1)
    # `where` twice, never a multiply: the inner one keeps the division finite,
    # since a masked-out `0/0` is nan and `0.0 * nan` is nan too.
    safe = jnp.where(norms > 0.0, norms, 1.0)
    per_row = np.asarray(
        jnp.where(norms > 0.0, jnp.linalg.norm(residuals, axis=1) / safe, jnp.nan)
    )
    # `np.max` and NOT `np.nanmax`: a multi-component latent with one dead
    # component IS a dead direction, and hiding it behind its live components is
    # the failure this whole function exists to make visible.
    fractions = {
        name: float(np.max(per_row[start:stop]))
        for name, (start, stop) in zip(names, spans, strict=True)
    }

    structure = jax.tree.structure(template)
    return FidelityReport(
        residuals=fractions,
        full=FlatMatrix(
            matrix=stacked @ stacked.T,
            structure=structure,
            kind="fisher",
            names=names,
            spans=spans,
            shapes=shapes,
        ),
        projected=FlatMatrix(
            matrix=coefficients @ basis.gram() @ coefficients.T,
            structure=structure,
            kind="fisher",
            names=names,
            spans=spans,
            shapes=shapes,
        ),
    )


def _whiten(rows: jax.Array, weight: jax.Array) -> jax.Array:
    """``rows * weight``, selecting on the weight first and multiplying second.

    Never a bare product. A score direction or a bank draw may be non-finite at
    a sample the reference could not see -- that is what a flag is -- and
    ``0.0 * inf`` and ``0.0 * nan`` are both ``nan``, which then survives
    ``orthonormal_transform`` silently: ``nan <= cut`` is ``False``, so the row
    is *kept* with NaN entries rather than dropped, and every projection built
    from it is NaN while the Gram matrix stays perfectly well conditioned.
    """
    return jnp.where(weight > 0.0, rows, 0.0) * weight


def _refuse_shapes_no_projector_can_use(rows: jax.Array, weight: jax.Array) -> None:
    """The structural checks, before anything is derived from them."""
    if rows.ndim != 2:
        raise StateValidationError(
            f"ReducedBasis.rows must be (n_S, n_data); got {rows.shape}."
        )
    n_basis, n_data = rows.shape
    if weight.shape != (n_data,):
        raise StateValidationError(
            f"ReducedBasis.weight is {weight.shape} but the rows are over "
            f"{n_data} samples."
        )
    if n_basis > n_data:
        raise StateValidationError(
            f"This basis has {n_basis} rows over {n_data} samples -- more "
            "directions than the data can distinguish. Reduce n_S, or admit "
            "that the epoch does not constrain them: the extra rows are not a "
            "richer model, they are a Gram matrix that is singular by counting."
        )
    if not bool(jnp.all(jnp.isfinite(weight) & (weight >= 0.0))):
        raise StateValidationError(
            "ReducedBasis.weight must be finite and non-negative. It is "
            "`1/sigma`, so a flagged sample is `0.0` here -- `inf` is the "
            "encoding on sigma itself (FlaggedNoise), never on its reciprocal, "
            "and a NaN would make every density NaN while leaving the Gram "
            "matrix perfectly well conditioned. Build it by selecting on "
            "isfinite(sigma) before dividing, never by dividing and hoping."
        )


def _refuse_a_direction_that_is_not_there(whitened: jax.Array) -> None:
    """Run on the whitened rows, so it fires only where the weight is non-zero.

    A non-finite raw row at a *flagged* sample is legal and reaches here as an
    exact ``0.0``: the reference could not see that sample, so no direction has
    a component there to be non-finite about.
    """
    if not bool(jnp.all(jnp.isfinite(whitened))):
        raise StateValidationError(
            "ReducedBasis.rows holds a non-finite entry where the weight is "
            "non-zero. A basis direction that is NaN or inf where the data was "
            "observed makes every projection NaN, and `0.0 * nan` is `nan`, so "
            "no downstream mask can undo it."
        )
