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

**There are two accumulators, not one.** A T2 term's stored numbers are a
quadratic form in ``theta``; a T1 term's are a quadratic form in the *basis
coefficients*, because ``c(theta)`` is nonlinear and that is the whole reason
the tier exists. Summing the two would be the same error as summing terms over
different latents, one level down, so they are kept apart and
:meth:`BayesMemory.log_likelihood` adds their densities rather than their
factors. ``SqrtInfo.combine`` refuses the mixture by name in any case, which is
what makes the routing defence in depth rather than the only defence.

Each accumulator's pytree keeps a fixed treedef for the life of a campaign.
That is not tidiness: an ``eqx.filter_jit``-ed log-density over a pytree whose
child count grows with N retraces once per epoch, which is measurable well
before the thousand epochs the design targets. Note what is fixed and what is
not -- ``archive`` gains one term per epoch by design, so it is the *density
path* (the two accumulators plus the shared dictionary) that keeps its shape,
and :meth:`BayesMemory.to_numpyro_model` closes over exactly that rather than
over ``self``.

**The archive is one pytree leaf, not one per term, and the reason is not the
sum.** An earlier version of this sentence said the archive "is never re-summed
on the sampling path", which was true and beside the point: nothing summed it,
and it still cost O(N) per call. Equinox wraps every non-magic bound method as a
``BoundMethod``, which is a ``Module`` *with* a dataclass ``__init__``, and that
constructor flattens ``(args, kwargs)`` -- ``self`` among them -- to check each
leaf for a jax-transformed function. So ``memory.log_likelihood(v)`` paid for
every array in every stored term before executing a line of its own body --
measured at 1,000 / 2,000 / 4,000 epochs, **1.43 / 3.18 / 7.24 ms** -- and a
NUTS chain pays that once per leapfrog step.

Holding the terms behind :class:`_Archive`, which is not a registered pytree
node and is therefore a single opaque leaf, makes the memory's leaf count
independent of the campaign's length: 12,007 leaves at 4,000 epochs before, 8 at
any length after. The same three sizes then measure **0.13 / 0.16 / 0.15 ms**,
which is flat and is also nearly free -- ``accumulated.log_prob`` on the same
values costs 0.109 ms at N = 4,000, so what the memory's own wrapper still adds
is 0.025 ms and a constant. ``remember`` fell from 2.42 / 4.51 / 10.22 ms per
epoch to 0.32 / 0.36 / 0.37, and a 4,000-epoch campaign from 27.0 s to 1.7 s.

``eqx.field(static=True)`` would also have taken the terms out of the leaf list,
and would have been wrong: a static field goes into the *treedef*, where array
``__eq__`` decides treedef equality. Equinox warns "A JAX array is being set as
static" for exactly that, and both ``ReducedBasis.reference_values`` and
:attr:`BayesMemory.basis` carry the same note. An opaque leaf keeps the arrays on
the dynamic side, where their identity rather than their contents is compared.

The one thing that becomes explicit rather than automatic is serialisation:
``eqx.tree_serialise_leaves`` walks leaves it recognises and skips one it does
not, silently, so :mod:`rheplicant.inference.archive` now writes
``(memory, tuple(memory.archive))`` and says so in its format version.
"""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compressed import (
    COEFFICIENTS,
    REQUIRED_TERM_MEMBERS,
    CompressedLikelihood,
)
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.sqrtinfo import SqrtInfo
from rheplicant.inference.uncertainty import FlatMatrix, _named_spans


def _is_reduced(term: Any) -> bool:
    """Whether this term's numbers are a quadratic in the BASIS COEFFICIENTS.

    Asked of the *stored form* rather than of the class, because that is the
    fact which decides what the term can be added to: a quadratic in ``c``
    cannot join an accumulator over ``theta``, whatever either object is called.
    ``hasattr(term, "basis")`` would be the same question asked of a spelling,
    and would answer ``True`` for anything that happened to carry that name.

    A tier whose ``info`` refuses -- T0 keeps its raw data and is not a
    quadratic form in any space -- is not this one. The refusal is swallowed
    here and re-raised immediately by the quadratic path, which reads ``info``
    on the next line and whose message names the remedy.
    """
    try:
        info = term.info
    except StateValidationError:
        return False
    return isinstance(info, SqrtInfo) and tuple(info.names) == (COEFFICIENTS,)


class _Archive:
    """The remembered terms, held as ONE pytree leaf rather than N.

    Deliberately **not** registered with :mod:`jax.tree_util`, which is the whole
    mechanism: an unregistered object is a leaf, so flattening a
    :class:`BayesMemory` costs the same whether it holds one epoch or ten
    thousand. Registering it -- or storing the tuple directly, as this did --
    puts three arrays per term into the leaf list, and every bound-method call
    on the memory walks all of them (see this module's docstring).

    ``__slots__`` is not a micro-optimisation either: it is what stops a caller
    quietly attaching state to an object that the surrounding
    :class:`~equinox.Module` believes is immutable.

    Membership is answered from ``ids`` rather than by rebuilding
    ``{held.epoch_id for held in terms}`` on every ``remember``. That rebuild is
    a second, independent O(N) per call which survives the leaf-count fix
    untouched -- it is plain Python and has nothing to do with pytrees -- and
    **it is still O(N) here.** ``frozenset | {x}`` copies; nothing in the
    standard library gives a persistent set with O(1) insertion, and the
    alternative that would (a dict shared down the chain, extended in place)
    cannot be reconciled with the immutability ``remember`` promises: ``m1``
    must be unchanged by ``m2 = m1.remember(t)``, and two remembers off the same
    ``m1`` would then see each other's terms.

    What the union buys is a smaller constant, measured on the same 10,000
    ids: 0.0490 vs 0.0211 ms at N = 4,000 and 0.1694 vs 0.0891 ms at N = 10,000,
    so a little over half. It is kept because it is free to write, and it is
    **not** claimed to fix anything: against a ``remember`` that costs 0.37
    ms/epoch flat after the leaf fix, the residual is 6 % at N = 4,000 and 24 %
    at N = 10,000, or about 0.45 s spread over a ten-thousand-epoch campaign.
    That is the measurement on which this was judged acceptable and left alone.
    """

    __slots__ = ("terms", "ids")

    def __init__(
        self,
        terms: tuple[CompressedLikelihood, ...] = (),
        ids: frozenset[str] | None = None,
    ):
        self.terms = tuple(terms)
        self.ids = (
            frozenset(term.epoch_id for term in self.terms) if ids is None else ids
        )

    def appended(self, term: CompressedLikelihood) -> "_Archive":
        """A new archive holding ``term`` last. The original is unchanged."""
        return _Archive(self.terms + (term,), self.ids | {term.epoch_id})

    # `is`, never `==`, and that is the whole care in these two methods.
    # `eqx.filter_jit` partitions leaves into arrays and everything else, so an
    # opaque leaf lands on the STATIC side, where this `__eq__` decides whether
    # a cached trace is reused. Comparing the terms by value would call `==` on
    # arrays -- the elementwise-array-instead-of-bool trap that `static=True` was
    # rejected for -- while the default identity `__eq__` on the archive itself
    # is too coarse the other way: two separately built empty archives would
    # compare unequal and retrace a density path whose shape never changed
    # (`tests/evidence/test_compress_dispatch.py` measures exactly that, five
    # traces for five epochs). Term identity is the honest middle: terms are
    # immutable and never rebuilt in place, so same objects means same numbers.
    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is _Archive
            and len(self.terms) == len(other.terms)
            # `strict=` is safe rather than redundant: the length check above is
            # what makes it unreachable, and it stays honest if that ever moves.
            and all(
                mine is theirs
                for mine, theirs in zip(self.terms, other.terms, strict=True)
            )
        )

    def __hash__(self) -> int:
        # Consistent with the above by being coarser than it: equal archives hold
        # the same terms, hence the same count and the same ids.
        return hash((len(self.terms), self.ids))


class BayesMemory(eqx.Module):
    """Accumulated evidence from a campaign, sampled without the raw data.

    Attributes:
        factorization: the single declaration -- which latents are global, and
            the campaign's only prior.
        accumulated: the running
            :class:`~rheplicant.inference.sqrtinfo.SqrtInfo`. Fixed treedef.
        archive: the terms as remembered, kept for diagnostics, re-anchoring
            and the smoother. The one part of the memory that legitimately grows
            with the campaign -- and therefore the one held behind
            :class:`_Archive`, so that growing costs one leaf rather than N.
            A read-only property, because the stored object is not the tuple:
            every reader here iterates in Python (``len``, ``archive[0]``, a
            comprehension over the terms), so the property hands back the plain
            tuple those readers already expect and the wrapper stays an
            implementation detail of the flattening.
        coefficients: the second running
            :class:`~rheplicant.inference.sqrtinfo.SqrtInfo`, over the reduced
            basis coefficients. ``None`` until the first T1 term arrives.
        basis: the shared
            :class:`~rheplicant.inference.reduced_basis.ReducedBasis` that
            ``coefficients`` is a quadratic form in. ``None`` alongside it. A
            **dynamic** field: it holds arrays, and equinox puts a static field
            into the treedef, where array ``__eq__`` decides treedef equality --
            it warns "A JAX array is being set as static" for exactly this, and
            ``ReducedBasis.reference_values`` carries the same note for the same
            reason.

    New fields go last, with defaults: Plan A's tests construct this
    positionally as ``BayesMemory(factorization, accumulated, archive)``, and
    that third argument is still a plain tuple of terms.
    """

    factorization: Factorization
    accumulated: SqrtInfo
    _archive: Any = eqx.field(default=None)
    coefficients: SqrtInfo | None = eqx.field(default=None)
    basis: Any = eqx.field(default=None)

    def __init__(
        self,
        factorization: Factorization,
        accumulated: SqrtInfo | None = None,
        archive: tuple[CompressedLikelihood, ...] | _Archive = (),
        coefficients: SqrtInfo | None = None,
        basis: Any = None,
    ):
        self.factorization = factorization
        self.accumulated = (
            SqrtInfo.null(factorization.global_names, factorization.global_shapes)
            if accumulated is None
            else accumulated
        )
        # An `_Archive` passes straight through: rebuilding one from its own
        # terms would re-derive the id set, which is the O(N) this exists to
        # avoid, and `remember` hands over an archive it has already extended.
        self._archive = archive if isinstance(archive, _Archive) else _Archive(archive)
        self.coefficients = coefficients
        self.basis = basis

    @property
    def archive(self) -> tuple[CompressedLikelihood, ...]:
        """The stored terms, oldest first."""
        return self._archive.terms

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
        if _is_reduced(term):
            return self._remember_reduced(term)
        return BayesMemory(
            factorization=self.factorization,
            accumulated=SqrtInfo.combine(self.accumulated, term.info),
            archive=self._archive.appended(term),
            coefficients=self.coefficients,
            basis=self.basis,
        )

    def _remember_reduced(self, term: CompressedLikelihood) -> "BayesMemory":
        """Fold a T1 term into the SECOND accumulator, the one over ``c``.

        A reduced-basis term is a quadratic in the basis coefficients, not in
        theta -- ``c(theta)`` is nonlinear, which is why the tier exists -- so
        it cannot join the theta accumulator. It can still be accumulated by the
        same QR, in its own space, which is what keeps the density path's
        treedef fixed for the life of a campaign: Plan A measured 60 traces and
        3.66 s for 60 successive remembers into a jitted density over a growing
        pytree, against 1 trace for this shape.

        The dictionary is compared by content hash, not by identity. Two terms
        compressed against different dictionaries are quadratic forms in
        different ordered vectors of the same length, so their sum has the right
        shape, the right conditioning and no meaning -- and nothing downstream
        would ever notice, because the numbers stay finite and the archive still
        lists both epochs.
        """
        if (
            self.basis is not None
            and self.basis.fingerprint() != term.basis.fingerprint()
        ):
            raise StateValidationError(
                f"Term {term.epoch_id!r} was compressed against a different "
                "dictionary from the one this memory already holds. The stored "
                "numbers are a quadratic form in a SPECIFIC ordered coefficient "
                "vector, so adding two of them is not a likelihood -- it is the "
                "same error as summing terms over different latents, one level "
                "down, and it is silent because both sums have the same shape. "
                "Recompress the epoch against the campaign's basis, or start a "
                "second memory."
            )
        running = (
            SqrtInfo.null(term.info.names, term.info.shapes)
            if self.coefficients is None
            else self.coefficients
        )
        return BayesMemory(
            factorization=self.factorization,
            accumulated=self.accumulated,
            archive=self._archive.appended(term),
            coefficients=SqrtInfo.combine(running, term.info),
            basis=term.basis,
        )

    def _reject_bad_term(self, term: CompressedLikelihood, duplicate: bool) -> None:
        absent = [name for name in REQUIRED_TERM_MEMBERS if not hasattr(term, name)]
        if absent:
            raise StateValidationError(
                f"This term is missing {absent}, which BayesMemory reads. "
                f"CompressedLikelihood requires {list(REQUIRED_TERM_MEMBERS)} plus "
                "__call__. Checked here rather than where each is first read, "
                "because accumulation is a QR: a term admitted now is folded "
                "irreversibly into the running factor, and the omission would "
                "surface later as an AttributeError out of audit() or "
                "save_memory(), by which time the campaign already depends on it."
            )
        declared = self.factorization.global_names
        if _is_reduced(term):
            # A T1 term's columns are the basis coefficients, so they cannot be
            # compared against the campaign's latents at all. What it does claim
            # is which latents its coefficient map CONSUMES, and a subset is
            # legal: a basis that seeded three of four latents simply says
            # nothing about the fourth, which is a rank-deficient epoch and the
            # normal case rather than an error. A name the memory never declared
            # is a different model, and that is what is refused.
            stray = [name for name in term.latents if name not in declared]
            if stray:
                raise StateValidationError(
                    f"Term {term.epoch_id!r} expands its prediction in latents "
                    f"{stray}, which this memory does not declare; it accumulates "
                    f"{list(declared)}. The coefficient map would be evaluated at "
                    "values the sampler never produces."
                )
        elif tuple(term.latents) != declared:
            raise StateValidationError(
                f"This term is over different latents from the memory: term has "
                f"{list(term.latents)}, memory accumulates "
                f"{list(declared)}. Summing them is not a likelihood."
            )
        if getattr(term, "prior_share", (0, 1))[0] != 0:
            raise StateValidationError(
                f"Term {term.epoch_id!r} carries prior_share={term.prior_share}, but "
                "a streaming memory stores prior-free factors: log_posterior applies "
                "the prior exactly once, so a tempered term would apply it twice. "
                "Divide the temper back out at compression, or use the batch "
                "consensus path."
            )
        held_terms = self._archive.terms
        if held_terms:
            held = held_terms[0].estimator
            if term.estimator != held:
                raise StateValidationError(
                    f"Term {term.epoch_id!r} was built with estimator {term.estimator} "
                    f"but this memory holds {held}. Generalized least squares and the "
                    "full Gaussian likelihood are different estimators (D21/D23); "
                    "their sum is neither."
                )
        if not duplicate and term.epoch_id in self._archive.ids:
            raise StateValidationError(
                f"Epoch {term.epoch_id!r} is already in this memory. Adding it "
                "again would count its data twice, narrowing the posterior with "
                "nothing to show for it. Pass duplicate=True if that is genuinely "
                "what you mean."
            )

    # --------------------------------------------------------------- densities --

    def log_likelihood(self, values: dict[str, jax.Array]) -> jax.Array:
        """Sum of the stored factors, quadratic and reduced-basis alike. No prior.

        The two accumulators are added as *densities*, one forward call apart:
        the reduced half is a quadratic in ``c(theta) - c_ref``, so evaluating
        it costs one pass through the model to get ``c`` and then ``O(n_S^2)``,
        independent of how many epochs went into it.
        """
        total = self.accumulated.log_prob(values)
        if self.coefficients is not None:
            shift = self.basis.coefficients(values) - self.basis.c_ref
            total = total + self.coefficients.log_prob(
                {self.coefficients.names[0]: shift}
            )
        return total

    def log_posterior(self, values: dict[str, jax.Array]) -> jax.Array:
        """The stored factors plus the prior, applied exactly once."""
        total = self.log_likelihood(values)
        for name, prior in self.factorization.global_priors.items():
            total = total + jnp.sum(prior.log_prob(values[name]))
        return total

    # ------------------------------------------------------------ diagnostics --

    def _flatten_layout(
        self,
    ) -> tuple[
        dict[str, jax.Array],
        tuple[str, ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[int, ...], ...],
        jax.Array,
    ]:
        """``(template, names, spans, shapes, order)`` for the global latents.

        ``order`` permutes a matrix built in **declared** column order -- which
        is what ``SqrtInfo`` carries -- into **flatten** order, which is what
        every named matrix in this package is labelled in. One copy, read by
        both :meth:`fisher` and :meth:`audit`, so the two cannot disagree about
        which is which: the permutation is the identity exactly when the
        latents happen to be alphabetical, so a second copy that drifted would
        pass every test written against an alphabetical fixture.
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
        return template, flat_names, flat_spans, flat_shapes, order

    def _permuted(self, matrix: jax.Array) -> jax.Array:
        """A declared-order square matrix, re-indexed into flatten order."""
        order = self._flatten_layout()[4]
        return matrix[jnp.ix_(order, order)]

    def _theta_fisher(self, at: dict[str, jax.Array] | None) -> jax.Array:
        """``sum_e F_e`` over theta, in DECLARED order.

        A reduced-basis term's information lives in coefficient space, and
        ``c(theta)`` is nonlinear, so pulling it back needs a point. That is why
        ``at`` is required here rather than defaulted: a single fixed Jacobian
        would be a linearisation nobody declared, and it would be invisible --
        the returned matrix is finite, symmetric and positive semi-definite
        whichever point it was taken at.
        """
        fisher = jnp.asarray(self.accumulated.fisher())
        if self.coefficients is None:
            return fisher
        if at is None:
            raise StateValidationError(
                "This memory holds reduced-basis terms, whose information is a "
                "quadratic in the basis coefficients rather than in theta. Pulling "
                "it back needs a point, because c(theta) is nonlinear -- that "
                "nonlinearity is the reason the tier exists, and a default point "
                "would be a linearisation nobody declared. Pass at=, e.g. "
                "at=dict(memory.basis.reference_values) for the storage origin."
            )
        jacobian = jax.jacfwd(self.basis.coefficients)(dict(at))
        columns = jnp.concatenate(
            [
                jnp.reshape(jacobian[name], (jacobian[name].shape[0], -1))
                for name in self.factorization.global_names
            ],
            axis=1,
        )
        return fisher + columns.T @ jnp.asarray(self.coefficients.fisher()) @ columns

    def fisher(self, at: dict[str, jax.Array] | None = None) -> FlatMatrix:
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

        Args:
            at: where to pull a reduced-basis term's coefficient-space
                information back into theta. Required once the memory holds
                one, and refused rather than defaulted -- see
                ``_theta_fisher``. Ignored for a memory of T2 terms alone,
                whose information is already a quadratic in theta.
        """
        template, flat_names, flat_spans, flat_shapes, _ = self._flatten_layout()
        return FlatMatrix(
            matrix=self._permuted(self._theta_fisher(at)),
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

        **Both accumulators reach the factor.** A memory holding only T1 terms
        has an empty theta accumulator, so a model built on ``accumulated``
        alone would sample a smooth, finite, perfectly well-behaved posterior
        that is exactly the prior -- no error, no warning, and no data. The
        closure is over the density path (both accumulators and the dictionary)
        and deliberately not over ``self``: ``archive`` grows with the campaign,
        and capturing it would retrace the sampler's log-density once per epoch.
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
        coefficients = self.coefficients
        basis = self.basis

        def model():
            values = {
                name: numpyro.sample(name, prior) for name, prior in priors.items()
            }
            total = accumulated.log_prob(values)
            if coefficients is not None:
                total = total + coefficients.log_prob(
                    {coefficients.names[0]: basis.coefficients(values) - basis.c_ref}
                )
            numpyro.factor("campaign", total)

        return model

    def _bias_report(
        self, at: dict[str, jax.Array] | None
    ) -> tuple[dict[str, float], tuple[str, ...]]:
        """``({name: |bias| / sigma_N}, unconstrained)`` -- section 7's budget.

        **Marginalised, not raw.** ``bias = F^-1 sum_e grad(delta_e)`` propagates
        the tilt through the parameter correlations, and the width it is divided
        by is ``sqrt(diag(F^-1))`` -- the error bar left after the foreground
        latents are integrated out, which is the one a result is quoted with. A
        raw gradient amplitude is a derivative of nats with respect to whatever
        units a latent happens to carry, and it ranks the directions
        differently: measured over eight RHINO epochs it calls ``running`` 9.4x
        more compromised than ``t21_depth``, while the marginalised ratio calls
        ``t21_depth`` 4.0x worse than ``running``. The orderings are reversed,
        and only the second one is a scientific error.

        A direction whose width is exactly zero has a ratio of ``0/0``. That is
        a young campaign, not a failure -- an epoch flagged end to end has a
        stored factor of exactly zero and constrains nothing at all -- so it is
        named rather than refused. A multi-component latent with one such
        component is named too: hiding a dead component behind its live ones is
        the failure this whole section exists to make visible.
        """
        gradients = [
            (term.bias_names, term.bias_gradient)
            for term in self.archive
            if getattr(term, "bias_gradient", None) is not None
        ]
        if not gradients:
            return {}, ()

        _, names, spans, _, _ = self._flatten_layout()
        stray = {stored for stored, _ in gradients if tuple(stored) != names}
        if stray:
            raise StateValidationError(
                f"A stored bias_names is {sorted(stray)[0]} but this memory "
                f"accumulates {list(names)}. Summing the two gradients would be "
                "wrong by a permutation or short by a block, and it is silent in "
                "both cases because the shapes still match -- which is Plan A's "
                "fisher() bug one layer along. Recompress the epoch against a "
                "basis over this memory's latents."
            )

        covariance = jnp.linalg.pinv(self._permuted(self._theta_fisher(at)))
        bias = covariance @ jnp.sum(
            jnp.stack([gradient for _, gradient in gradients]), axis=0
        )
        width = jnp.sqrt(jnp.diag(covariance))
        ratios: dict[str, float] = {}
        unconstrained: list[str] = []
        for name, (start, stop) in zip(names, spans, strict=True):
            widths = np.asarray(width[start:stop])
            if not np.all(widths > 0.0):
                unconstrained.append(name)
                continue
            ratios[name] = float(
                np.max(np.abs(np.asarray(bias[start:stop])) / widths)
            )
        return ratios, tuple(unconstrained)

    def audit(
        self,
        at: dict[str, jax.Array] | None = None,
        bias_tolerance: float | None = None,
    ) -> dict[str, Any]:
        """What the memory can say about its own trustworthiness.

        ``fisher_lambda_min`` and ``fisher_condition`` describe the **theta**
        accumulator alone, and there is no honest way to merge the two: the
        second accumulator's curvature is in coefficient space, and mapping it
        back through ``dc/dtheta`` would be a Fisher at one point rather than a
        property of the stored numbers. So it gets its own pair, and the
        dictionary it is a form in is named. Reporting only the first for an
        archive holding both would say ``fisher_lambda_min = 0`` and
        ``fisher_condition = inf`` for a T1-only campaign, which reads as a
        degenerate memory when the memory is simply not quadratic in theta.

        ``bias_over_sigma`` is section 7's budget, per named direction, and
        ``unconstrained`` names the directions whose ratio is ``0/0``.

        Args:
            at: where to pull the coefficient-space information back into theta
                for the bias ratio. Defaults to the basis's recorded
                ``reference_values`` -- and that default is the point, not a
                convenience: the stored gradients were taken at the storage
                origin, so a Fisher taken anywhere else makes the ratio a
                quotient of two different linearisations. :meth:`fisher`
                refuses to default it for exactly the opposite reason -- there
                the question has no privileged point.
            bias_tolerance: refuse when any **constrained** direction's
                ``|bias| / sigma_N`` exceeds this. Directions the campaign does
                not yet constrain are listed under ``"unconstrained"`` instead
                of refused: their ratio is ``0/0``, and treating that as a
                failure would refuse every young campaign.
        """
        if at is None and self.basis is not None and self.basis.reference_values:
            at = dict(self.basis.reference_values)
        ratios, unconstrained = self._bias_report(at)
        if bias_tolerance is not None:
            # `not <=`, never `>`: NaN loses every comparison, so `ratio >
            # tolerance` is False for a poisoned campaign and the guard waves it
            # through while reporting the NaN in the same dict.
            bad = {n: r for n, r in ratios.items() if not r <= bias_tolerance}
            if bad:
                listed = ", ".join(f"{n}={r:.3e}" for n, r in sorted(bad.items()))
                raise StateValidationError(
                    f"Compression bias exceeds the declared budget: {listed} "
                    f"against {bias_tolerance}. This is the theta-GRADIENT of the "
                    "fidelity residual, not its magnitude: a constant offset has "
                    "exactly zero effect on the posterior, and because one basis "
                    "and one instrument model serve every epoch the tilt is "
                    "coherent, so the bias is N-independent while sigma_N falls as "
                    "N^-1/2 and the ratio above grows as sqrt(N). Raise n_basis, "
                    "re-anchor the basis nearer the accumulated posterior, or "
                    "declare a larger budget knowing what it costs."
                )

        fisher = jnp.asarray(self.accumulated.fisher())
        eigenvalues = jnp.linalg.eigvalsh(fisher)
        largest = float(eigenvalues[-1])
        smallest = float(eigenvalues[0])
        coefficient_lambda_min: float | None = None
        coefficient_condition: float | None = None
        if self.coefficients is not None:
            spectrum = jnp.linalg.eigvalsh(jnp.asarray(self.coefficients.fisher()))
            coefficient_lambda_min = float(spectrum[0])
            coefficient_condition = (
                float("inf")
                if coefficient_lambda_min <= 0
                else float(spectrum[-1]) / coefficient_lambda_min
            )
        return {
            "n_epochs": len(self.archive),
            "n_reduced": sum(1 for term in self.archive if _is_reduced(term)),
            "epoch_ids": tuple(term.epoch_id for term in self.archive),
            "estimator": self.archive[0].estimator if self.archive else None,
            "prior_shares_sum": sum(
                (getattr(term, "share", 0) for term in self.archive), start=0
            ),
            "n_observed": sum(term.n_observed for term in self.archive),
            "all_exact": all(term.exact for term in self.archive),
            "fisher_lambda_min": smallest,
            "fisher_condition": float("inf") if smallest <= 0 else largest / smallest,
            "basis_fingerprint": None if self.basis is None else self.basis.fingerprint(),
            "coefficient_lambda_min": coefficient_lambda_min,
            "coefficient_condition": coefficient_condition,
            "bias_over_sigma": ratios,
            "unconstrained": unconstrained,
        }
