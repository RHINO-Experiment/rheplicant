"""One epoch's data, compressed into a factor of the campaign likelihood.

The protocol mirrors :class:`~rheplicant.inference.likelihood.Likelihood`
one seam further along: that one scores a *prediction* against data, this one
has already absorbed the data and scores the *latents*. The name says
"likelihood" because that is the invariant the whole accumulation layer rests
on -- a stored term contains **no factor of the prior**, so the prior can be
applied exactly once, at the end, by whoever holds the
:class:`~rheplicant.inference.factorize.Factorization`.

Provenance travels with the term because two of the package's existing
distinctions become silent bugs the moment terms are summed. Under
``RadiometerNoise`` the noise level is a function of the prediction (D21), so a
term built with the covariance frozen at the ``iterative_gls`` solution encodes
*generalized least squares*, while one built with the log-determinant live
encodes the full Gaussian posterior (D23). They are different estimators.
Adding them produces a finite, correctly-shaped, meaningless number, so
:attr:`QuadraticLikelihood.estimator` exists and the memory refuses to mix.
"""

from collections.abc import Callable
from fractions import Fraction
from typing import Any, Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.sqrtinfo import SqrtInfo


@runtime_checkable
class CompressedLikelihood(Protocol):
    """Contract: ``logL = term(values)``, over the global latents alone.

    Every member below is one :class:`~rheplicant.inference.memory.BayesMemory`
    actually reads, and the list is that long for a reason. An earlier version
    declared only ``latents``, ``epoch_id``, ``estimator`` and ``__call__``,
    which made the protocol a **narrower** claim than the code relied on: a term
    satisfying it passed ``isinstance``, passed ``remember``, contributed to the
    accumulated density, and then raised ``AttributeError: 'X' object has no
    attribute 'n_observed'`` from ``audit()`` -- a diagnostic, reached long
    after the term had already been folded irreversibly into the QR.

    ``exact`` and ``n_observed`` are read by ``audit()``; ``prior_share`` by
    ``remember``'s tempering refusal. A published contract that omits them is
    an invitation to write a class that cannot work.
    """

    @property
    def latents(self) -> tuple[str, ...]: ...

    @property
    def epoch_id(self) -> str: ...

    @property
    def estimator(self) -> tuple[str, ...]: ...

    @property
    def n_observed(self) -> int: ...

    @property
    def exact(self) -> bool: ...

    @property
    def prior_share(self) -> tuple[int, int]: ...

    def __call__(self, values: dict[str, jax.Array]) -> jax.Array: ...


#: What :class:`~rheplicant.inference.memory.BayesMemory` reads off a term.
#: Checked by name at ``remember`` so an incomplete term is refused at the door
#: rather than at a diagnostic, and listed here so the two cannot drift.
REQUIRED_TERM_MEMBERS = (
    "latents",
    "epoch_id",
    "estimator",
    "n_observed",
    "exact",
    "prior_share",
)


class QuadraticLikelihood(eqx.Module):
    """A log-quadratic factor: a :class:`~rheplicant.inference.sqrtinfo.SqrtInfo`
    plus what it means.

    Attributes:
        info: the square-root information form carrying the numbers.
        epoch_id: the recording's identity, supplied by the ingestion layer --
            the data hash, not a filename. This is what makes appending the
            same night twice refusable.
        n_observed: unflagged samples that went into it. Needed to interpret
            the residual chi-square, and the natural weight if a
            Consensus-Monte-Carlo share is ever assigned.
        exact: whether the term is a sufficient statistic (all latents linear,
            noise frozen) or an expansion. An expansion is only meaningful
            inside its ``support``, which is why one is required.
        support: ``{latent: (low, high)}`` where an approximate term is
            trustworthy. ``None`` for an exact term, and required otherwise.
        include_logdet: whether the Gaussian normalisation was kept. ``False``
            is generalized least squares (D21), not a cheaper version of the
            same estimator.
        noise_frozen_at: ``"none"`` for a genuinely fixed covariance, or the
            procedure that produced the frozen one (``"gls"``).
        prior_share: ``(numerator, denominator)`` of the prior tempering this
            term carries. ``(0, 1)`` -- no prior at all -- is the invariant, and
            the only value the streaming path produces. Stored as integers so
            that "the shares sum to one" is a statement that can be true for
            any N and any summation routine, which a float equality is not.
    """

    info: SqrtInfo
    epoch_id: str = eqx.field(static=True)
    n_observed: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True, default=True)
    support: dict[str, tuple[float, float]] | None = eqx.field(static=True, default=None)
    include_logdet: bool = eqx.field(static=True, default=True)
    noise_frozen_at: str = eqx.field(static=True, default="none")
    prior_share: tuple[int, int] = eqx.field(static=True, default=(0, 1))

    def __check_init__(self):
        for name, leaf in (
            ("factor", self.info.factor),
            ("target", self.info.target),
            ("offset", self.info.offset),
        ):
            if jnp.asarray(leaf).dtype != jnp.float64:
                raise StateValidationError(
                    f"QuadraticLikelihood.info.{name} is {jnp.asarray(leaf).dtype}, not "
                    "float64. Under RadiometerNoise the term's offset is the "
                    "time-bandwidth product while the difference of interest is the "
                    "sample count, so in float32 the quadratic form is annihilated "
                    "rather than merely imprecise. Enable x64: "
                    'jax.config.update("jax_enable_x64", True).'
                )
        if self.exact and self.support is not None:
            raise StateValidationError(
                "An exact term is a sufficient statistic, valid everywhere, so a "
                "support would be a claim about nothing. Drop the support, or set "
                "exact=False."
            )
        if not self.exact and self.support is None:
            raise StateValidationError(
                "An approximate term needs a support: it is an expansion, and the "
                "region it was built for is the only place its numbers mean "
                "anything. The accumulated posterior can and does leave that region "
                "as a campaign grows, and nothing in the numbers says so."
            )
        numerator, denominator = self.prior_share
        if not isinstance(numerator, int) or not isinstance(denominator, int):
            raise StateValidationError(
                f"prior_share must be a pair of integers (numerator, denominator); "
                f"got {self.prior_share!r}. Floats are refused because 'the shares "
                "sum to exactly 1' is then a predicate whose truth depends on the "
                "summation routine."
            )
        if denominator <= 0 or numerator < 0 or numerator > denominator:
            raise StateValidationError(
                f"prior_share {self.prior_share} is not a fraction in [0, 1]."
            )

    @property
    def latents(self) -> tuple[str, ...]:
        """The global latents this term is a function of."""
        return self.info.names

    @property
    def estimator(self) -> tuple[str, ...]:
        """What must match before two terms may be summed."""
        return ("full" if self.include_logdet else "gls", self.noise_frozen_at)

    @property
    def share(self) -> Fraction:
        """The prior share as an exact rational."""
        return Fraction(*self.prior_share)

    def __call__(self, values: dict[str, jax.Array]) -> jax.Array:
        return self.info.log_prob(values)


class RawLikelihood(eqx.Module):
    """T0 -- the epoch's likelihood with the raw data still inside it.

    **This tier deliberately does not compress.** It holds ``observed`` and a
    live ``predict``, so it defeats every one of the four bottlenecks §0 lists
    and has no place in a campaign. It exists because D24's posture requires it:
    an approximate posterior is only trustworthy where an exact one exists, and
    from T1 upward "exact" is defined as "agrees with this, absolutely, at every
    probe". §12.12's boundary validation is not writable without it.

    Two refusals keep it out of the places it must not reach.
    :attr:`info` raises rather than returning a quadratic form, so
    :meth:`~rheplicant.inference.memory.BayesMemory.remember` -- which folds a
    term into the running QR -- stops at the door instead of at a diagnostic.
    :func:`~rheplicant.inference.archive.save_memory` refuses it for the same
    reason: ``predict`` is a Python callable, and
    ``eqx.tree_serialise_leaves`` would take it from whatever template it was
    handed, so a reloaded T0 would evaluate a *different model* against the same
    data with no error and no warning.

    The masked normalisation is D21's, not a variant of it: a flagged sample is
    ``sigma = inf``, whose inverse variance is a clean zero but whose
    ``log(2 pi sigma^2)`` is ``+inf``. Only finite-sigma samples are summed. The
    residual is SELECTED on that same mask before it is divided, never weighted
    by a zero afterwards -- a flagged sample is usually flagged *because* it
    holds a NaN, and ``0.0 * nan`` is ``nan``.

    Attributes:
        predict: ``values -> prediction``, shaped like ``observed``. Static: it
            is code, not data, and it is what makes this tier unarchivable.
        observed: the epoch's data.
        sigma: the noise standard deviation, already resolved to an array.
            ``inf`` marks an unobserved sample.
        names: the latents ``predict`` consumes.
        epoch_id: the recording's data hash.
        include_logdet: whether the Gaussian normalisation is kept. ``False``
            is generalized least squares (D21), a different estimator rather
            than a cheaper version of this one.
        noise_frozen_at: ``"none"`` for a genuinely fixed covariance, or the
            procedure that produced the frozen one.
    """

    predict: Callable[[dict[str, jax.Array]], jax.Array] = eqx.field(static=True)
    observed: jax.Array
    sigma: jax.Array
    names: tuple[str, ...] = eqx.field(static=True)
    epoch_id: str = eqx.field(static=True)
    n_observed: int = eqx.field(static=True)
    exact: bool = eqx.field(static=True)
    include_logdet: bool = eqx.field(static=True)
    noise_frozen_at: str = eqx.field(static=True)
    prior_share: tuple[int, int] = eqx.field(static=True)

    def __init__(
        self,
        predict: Callable[[dict[str, jax.Array]], jax.Array],
        observed: jax.Array,
        sigma: Any,
        names: tuple[str, ...],
        epoch_id: str,
        include_logdet: bool = True,
        noise_frozen_at: str = "none",
    ):
        self.predict = predict
        self.observed = jnp.asarray(observed)
        self.sigma = jnp.broadcast_to(jnp.asarray(sigma), self.observed.shape)
        self.names = tuple(names)
        self.epoch_id = epoch_id
        self.n_observed = int(jnp.sum(jnp.isfinite(self.sigma)))
        self.exact = True
        self.include_logdet = bool(include_logdet)
        self.noise_frozen_at = noise_frozen_at
        self.prior_share = (0, 1)

    @property
    def latents(self) -> tuple[str, ...]:
        return self.names

    @property
    def estimator(self) -> tuple[str, ...]:
        return ("full" if self.include_logdet else "gls", self.noise_frozen_at)

    @property
    def info(self) -> SqrtInfo:
        raise StateValidationError(
            f"Term {self.epoch_id!r} is a RawLikelihood -- T0, the oracle. It keeps "
            "the raw data and a live forward model, so it is not a quadratic form "
            "and it is not storable: accumulating it would mean the campaign never "
            "released a byte, and archiving it would serialise the arrays while "
            "silently taking `predict` from the load-time template. Compress the "
            "epoch (compress_linear for a linear model, compress_reduced_basis "
            "otherwise) and remember that. Use this one to check the result."
        )

    def __call__(self, values: dict[str, jax.Array]) -> jax.Array:
        seen = jnp.isfinite(self.sigma)
        safe = jnp.where(seen, self.sigma, 1.0)
        prediction = jnp.reshape(self.predict(values), self.observed.shape)
        residual = jnp.where(seen, self.observed - prediction, 0.0) / safe
        quadratic = -0.5 * jnp.sum(residual**2)
        if not self.include_logdet:
            return quadratic
        return quadratic - 0.5 * jnp.sum(
            jnp.where(seen, jnp.log(2.0 * jnp.pi * safe**2), 0.0)
        )
