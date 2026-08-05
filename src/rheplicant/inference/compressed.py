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
:attr:`estimator` exists and the memory refuses to mix.
"""

from fractions import Fraction
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.sqrtinfo import SqrtInfo


@runtime_checkable
class CompressedLikelihood(Protocol):
    """Contract: ``logL = term(values)``, over the global latents alone."""

    @property
    def latents(self) -> tuple[str, ...]: ...

    @property
    def epoch_id(self) -> str: ...

    @property
    def estimator(self) -> tuple[str, ...]: ...

    def __call__(self, values: dict[str, jax.Array]) -> jax.Array: ...


class QuadraticLikelihood(eqx.Module):
    """A log-quadratic factor: a :class:`SqrtInfo` plus what it means.

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
