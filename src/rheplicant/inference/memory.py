"""What the twin remembers of a campaign after the recordings are archived.

The invariant is one sentence: **the memory holds likelihood factors and
exactly one prior.** Everything else in this module exists to make that
impossible to violate by accident. Stored terms are prior-free
(:class:`~rheplicant.inference.compressed.QuadraticLikelihood`), the prior
lives on the :class:`~rheplicant.inference.factorize.Factorization`'s global
latents, and the two accessors below differ by exactly one application of it.

Three refusals are worth the words they cost, because each one otherwise
produces a smooth, correctly-shaped, over-confident answer:

* **The same night twice.** A retried epoch appended again adds its
  information a second time; the posterior stays centred and narrows by
  ``sqrt(2)`` for a duplicated batch. Terms therefore carry the recording's
  data hash and ``remember`` refuses a repeat, in the posture D17 takes on
  ``BeamSpillOperator`` plus ``GroundPickupOperator``: legitimate
  double-counting is a choice made deliberately, with ``duplicate=True``.
* **Two estimators.** Full-likelihood and GLS terms (D21/D23) are different
  estimators; their sum is neither.
* **A tempered term.** A term carrying a share of the prior breaks the
  invariant, and then ``log_posterior`` would apply the prior twice.

The accumulator's pytree keeps a fixed treedef for the life of a campaign.
That is not tidiness: an ``eqx.filter_jit``-ed log-density over a memory whose
child count grows with N retraces once per epoch, which is measurable well
before the thousand epochs the design targets.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compressed import CompressedLikelihood
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.sqrtinfo import SqrtInfo
from rheplicant.inference.uncertainty import FlatMatrix, _named_spans


class BayesMemory(eqx.Module):
    """Accumulated evidence from a campaign, sampled without the raw data.

    Attributes:
        factorization: the single declaration -- which latents are global, and
            the campaign's only prior.
        accumulated: the running
            :class:`~rheplicant.inference.sqrtinfo.SqrtInfo`. Fixed treedef.
        archive: the terms as remembered, kept for diagnostics, re-anchoring
            and the smoother. Never re-summed on the sampling path.
    """

    factorization: Factorization
    accumulated: SqrtInfo
    archive: tuple[CompressedLikelihood, ...] = eqx.field(default=())

    def __init__(
        self,
        factorization: Factorization,
        accumulated: SqrtInfo | None = None,
        archive: tuple[CompressedLikelihood, ...] = (),
    ):
        self.factorization = factorization
        self.accumulated = (
            SqrtInfo.null(factorization.global_names, factorization.global_shapes)
            if accumulated is None
            else accumulated
        )
        self.archive = tuple(archive)

    # ------------------------------------------------------------ accumulate --

    def remember(
        self, term: CompressedLikelihood, duplicate: bool = False
    ) -> "BayesMemory":
        """A new memory holding this term as well. The original is unchanged.

        Args:
            term: one epoch's compressed likelihood.
            duplicate: allow an ``epoch_id`` already present. Off by default,
                because the common cause is a retried run, and the effect is a
                posterior that narrows for no reason.
        """
        self._reject_bad_term(term, duplicate)
        return BayesMemory(
            factorization=self.factorization,
            accumulated=SqrtInfo.combine(self.accumulated, term.info),
            archive=self.archive + (term,),
        )

    def _reject_bad_term(self, term: CompressedLikelihood, duplicate: bool) -> None:
        if tuple(term.latents) != self.factorization.global_names:
            raise StateValidationError(
                f"This term is over different latents from the memory: term has "
                f"{list(term.latents)}, memory accumulates "
                f"{list(self.factorization.global_names)}. Summing them is not a "
                "likelihood."
            )
        if getattr(term, "prior_share", (0, 1))[0] != 0:
            raise StateValidationError(
                f"Term {term.epoch_id!r} carries prior_share={term.prior_share}, but "
                "a streaming memory stores prior-free factors: log_posterior applies "
                "the prior exactly once, so a tempered term would apply it twice. "
                "Divide the temper back out at compression, or use the batch "
                "consensus path."
            )
        if self.archive:
            held = self.archive[0].estimator
            if term.estimator != held:
                raise StateValidationError(
                    f"Term {term.epoch_id!r} was built with estimator {term.estimator} "
                    f"but this memory holds {held}. Generalized least squares and the "
                    "full Gaussian likelihood are different estimators (D21/D23); "
                    "their sum is neither."
                )
        if not duplicate:
            seen = {held.epoch_id for held in self.archive}
            if term.epoch_id in seen:
                raise StateValidationError(
                    f"Epoch {term.epoch_id!r} is already in this memory. Adding it "
                    "again would count its data twice, narrowing the posterior with "
                    "nothing to show for it. Pass duplicate=True if that is genuinely "
                    "what you mean."
                )

    # --------------------------------------------------------------- densities --

    def log_likelihood(self, values: dict[str, jax.Array]) -> jax.Array:
        """Sum of the stored factors. No prior."""
        return self.accumulated.log_prob(values)

    def log_posterior(self, values: dict[str, jax.Array]) -> jax.Array:
        """The stored factors plus the prior, applied exactly once."""
        total = self.log_likelihood(values)
        for name, prior in self.factorization.global_priors.items():
            total = total + jnp.sum(prior.log_prob(values[name]))
        return total

    # ------------------------------------------------------------ diagnostics --

    def fisher(self) -> FlatMatrix:
        """``sum_e F_e`` over the stored terms, with named rows.

        Excludes the prior's curvature, so it may legitimately be singular at
        small N: a single epoch usually constrains only a subspace, and that is
        exactly what the square-root form is for.

        **Permuted into flatten order, not left in declared order.** A
        :class:`~rheplicant.inference.uncertainty.FlatMatrix` carries the
        treedef its rows were flattened against, and ``jax`` sorts a dict's
        keys, while ``SqrtInfo``'s columns follow the order the latents were
        *declared* in. For a space declared ``("width", "depth")`` the two
        disagree, and returning the raw accumulator would hand back a matrix
        whose ``structure`` field describes an ordering the numbers do not
        have. Measured on that space with per-latent information 9 and 49:
        unpermuted, ``matrix`` reads ``diag(9, 49)`` against
        ``names=("width", "depth")`` while ``structure`` says
        ``{'depth', 'width'}``.

        Nothing downstream returns wrong numbers today -- ``sigma`` and
        ``block`` read ``names`` and ``spans`` together, and
        ``propagate_covariance`` catches the mismatch on its ``_named_spans``
        check. But it catches it as "computed for {'width': (), 'depth': ()}
        but params is {'depth': (), 'width': ()}", which reads as a shape
        disagreement between two identical shapes. Permuting here removes the
        cause instead: every other named matrix in the package derives its
        names from the actual flattening, deliberately "rather than from an
        assumption about dict ordering", and this was the one place that did
        not.
        """
        names = self.factorization.global_names
        shapes = self.factorization.global_shapes
        template = {
            name: jnp.zeros(shape) for name, shape in zip(names, shapes, strict=True)
        }
        flat_names, flat_spans, flat_shapes = _named_spans(template)

        declared: dict[str, range] = {}
        offset = 0
        for name, shape in zip(names, shapes, strict=True):
            size = int(jnp.zeros(shape).size)
            declared[name] = range(offset, offset + size)
            offset += size
        order = jnp.asarray(
            [column for name in flat_names for column in declared[name]], dtype=int
        )
        return FlatMatrix(
            matrix=self.accumulated.fisher()[jnp.ix_(order, order)],
            structure=jax.tree.structure(template),
            kind="fisher",
            names=flat_names,
            spans=flat_spans,
            shapes=flat_shapes,
        )

    def to_numpyro_model(self, **unsupported: Any):
        """A NumPyro model that samples the global latents against this memory.

        Unlike :func:`~rheplicant.inference.numpyro_bridge.to_numpyro_model`
        there is no pipeline, no observed data and no noise model here: the
        terms already absorbed all three. Passing ``noise_std=`` is therefore
        refused rather than ignored -- silently ignoring it would let a caller
        believe they had changed the likelihood.
        """
        if unsupported:
            raise StateValidationError(
                f"BayesMemory.to_numpyro_model takes no {sorted(unsupported)} -- the "
                "noise model, the data and the forward evaluation are already inside "
                "the stored terms. Changing the noise now would mean recompressing."
            )
        try:
            import numpyro
        except ImportError as exc:  # pragma: no cover - exercised by the import guard
            raise ImportError(
                'BayesMemory.to_numpyro_model needs numpyro: pip install '
                '"rheplicant[numpyro]"'
            ) from exc

        priors = self.factorization.global_priors
        accumulated = self.accumulated

        def model():
            values = {
                name: numpyro.sample(name, prior) for name, prior in priors.items()
            }
            numpyro.factor("campaign", accumulated.log_prob(values))

        return model

    def audit(self) -> dict[str, Any]:
        """What the memory can say about its own trustworthiness."""
        fisher = jnp.asarray(self.accumulated.fisher())
        eigenvalues = jnp.linalg.eigvalsh(fisher)
        largest = float(eigenvalues[-1])
        smallest = float(eigenvalues[0])
        return {
            "n_epochs": len(self.archive),
            "epoch_ids": tuple(term.epoch_id for term in self.archive),
            "estimator": self.archive[0].estimator if self.archive else None,
            "prior_shares_sum": sum(
                (getattr(term, "share", 0) for term in self.archive), start=0
            ),
            "n_observed": sum(term.n_observed for term in self.archive),
            "all_exact": all(term.exact for term in self.archive),
            "fisher_lambda_min": smallest,
            "fisher_condition": float("inf") if smallest <= 0 else largest / smallest,
        }
