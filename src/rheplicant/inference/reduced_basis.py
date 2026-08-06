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

from rheplicant.core.errors import StateValidationError


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
        whitened: ``rows * weight``. Orthonormal when :attr:`orthonormal`,
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
            same one.
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
    reference_values: dict[str, jax.Array] = eqx.field(static=True)

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
        self.reference_values = dict(reference_values or {})
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
