"""A nuisance that drifts across epochs, and the recursion that integrates it out.

A ``per_epoch`` latent is re-drawn every night and integrated away inside its own
epoch. A ``linked`` one is not: it is a Markov chain, and condition C1b says that
declaring one ``per_epoch`` marginalises a single physical fluctuation N times
against independent priors, injecting information that is not there. The exact
alternative is this module.

**The recursion.** Carry a joint square-root information factor over
``(theta, zeta_e)``. Fold in an epoch by stacking its rows and re-triangularising
-- :meth:`~rheplicant.inference.sqrtinfo.SqrtInfo.combine`'s arithmetic. Advance
to the next epoch by widening to ``(theta, zeta_e, zeta_{e+1})``, appending the
transition's rows, and marginalising ``zeta_e``: permute it first,
re-triangularise, drop row and column. That drop **is** the Schur complement, in
square root, which is what keeps a thousand-epoch accumulation inside float64
where the explicit ``(F, b)`` form goes indefinite. ``theta`` is never
marginalised, so what comes back is ``log p(d_1:N | theta)`` exactly.

**Two sub-scopes, because "linear-Gaussian" is not enough.** An OU with an
inferred correlation time is still linear-Gaussian, so a caveat phrased that way
is satisfied while its claim fails: ``Q(theta)``, ``phi(theta)`` and the Schur
complement all become functions of theta, and a filter run at compression time
pins them silently. The distinction lives in the *type*: a
:class:`LinearGaussianTransition` holds numbers and the theta posterior is exact
under filtering; a :class:`HyperTransition` holds a builder and is resolved
**inside** the theta likelihood, so the whole recursion is a differentiable
``lax.scan`` over the stored per-epoch blocks. One code path serves both, because
the recursion is traceable either way -- which is also why the fixed case is
validated by the same tests rather than by a second implementation of the same
arithmetic.

**The constant bookkeeping is not optional, and it is where this module can be
wrong while looking right.** Six constants reach the answer; the recursion's
shape, gradient and curvature are correct without any of them. Measured on
``tests/evidence/chain_bank.py`` at ``theta = (0.4, -1.1)``, the cost of
dropping one:

====================================================  ========
dropped                                               nats
====================================================  ========
initial ``zeta`` prior normalisation                  +0.9189
per-transition ``-0.5 logdet(2 pi Q)``, five of them  +2.8618
the spec's shorthand for it (``0.5 logdet Q^-1``)     +4.5947
marginalisation constant, six of them                 +7.2619
the fold corner ``-0.5 rho^2``, six of them           +45.9502
the masked data normalisation                         -6.8408
====================================================  ========

Two of those belong to nobody else. The corner is
:meth:`~rheplicant.inference.sqrtinfo.SqrtInfo.combine`'s and the data
normalisation is :mod:`~rheplicant.inference.compress`'s, and a reader will
assume the rest are handled elsewhere too; the **initial prior normalisation**
and the **final marginalisation** are new here.

Note also what does *not* appear: the marginalisation's own corner is exactly
zero in this recursion, because that QR is square and ``upper[keep:, width]`` is
a length-zero slice. Measured through the filter, deleting it moves the answer by
0.0 nats bit for bit. A test asserting it matters would pass vacuously, so
``tests/evidence/test_chain_filter.py`` pins the zero instead and pins the
**fold's** corner as the one that is worth 45.95.
"""

from collections.abc import Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.sqrtinfo import SqrtInfo

from .chain_blocks import (
    _CHAIN_REPEAT_REMEDY,
    _reject_a_foreign_stack,
    _square_block,
)
from .chain_blocks import _column_spans as _column_spans
from .chain_blocks import _quadratic_form as _quadratic_form
from .chain_blocks import _reject_a_foreign_block as _reject_a_foreign_block
from .chain_recursion import _check_block_width as _check_block_width
from .chain_recursion import _joint_covariance as _joint_covariance
from .chain_recursion import _zeta_joint as _zeta_joint
from .chain_recursion import (
    chain_log_likelihood,
    chain_marginal,
)

# Re-exported so this module's importers keep working. `X as X` is the
# explicit re-export form: a plain import of a name this file does not
# itself use is F401, and `ruff --fix` deletes it whatever the comment
# on the line says.
from .chain_recursion import smooth as smooth

# Re-exported so this module's importers keep working. `X as X` is the
# explicit re-export form; a plain import of a name this file does not
# itself use is F401 and `ruff --fix` deletes it.
from .chain_transition import HyperTransition as HyperTransition
from .chain_transition import LinearGaussianTransition as LinearGaussianTransition
from .chain_transition import ornstein_uhlenbeck as ornstein_uhlenbeck


class _Epochs:
    """The remembered terms and their ids, as ONE pytree leaf rather than N.

    The same device :class:`~rheplicant.inference.memory._Archive` uses and for
    the same measured reason: equinox wraps every bound method as a ``Module``
    whose constructor flattens ``self``, so a memory holding N terms as pytree
    children pays for every array in every term before executing a line of its
    own body. Unregistered means leaf.
    """

    __slots__ = ("terms", "ids")

    def __init__(self, terms: Sequence[Any] = (), ids: frozenset[str] | None = None):
        self.terms = tuple(terms)
        self.ids = frozenset(term.epoch_id for term in self.terms) if ids is None else ids

    def appended(self, term: Any) -> "_Epochs":
        """A new record holding ``term`` last. The original is unchanged."""
        return _Epochs(self.terms + (term,), self.ids | {term.epoch_id})

    # `is`, never `==`: an opaque leaf lands on `filter_jit`'s static side, where
    # this decides whether a cached trace is reused, and comparing terms by value
    # would call `==` on arrays.
    def __eq__(self, other: Any) -> bool:
        return (
            type(other) is _Epochs
            and len(self.terms) == len(other.terms)
            and all(a is b for a, b in zip(self.terms, other.terms, strict=True))
        )

    def __hash__(self) -> int:
        return hash((len(self.terms), self.ids))


class ChainMemory(eqx.Module):
    """A campaign whose nuisance drifts across epochs. **Ordered.**

    The difference from :class:`~rheplicant.inference.memory.BayesMemory` is one
    sentence: a bag is exchangeable and a chain is not, so a bag can fold each
    term into a running QR and forget it, while a chain must keep the per-epoch
    blocks and run the recursion. Section 6 puts that distinction in the type
    rather than in a flag, and the two refusals are symmetric --
    ``BayesMemory.remember`` refuses a term carrying a linked latent's columns,
    and this one requires it.

    **The stack grows, and a jitted density therefore retraces once per night.**
    Section 11's compile-cost measurement applies to the bag's fixed-treedef
    accumulator; a chain cannot have one, because section 6 spends O(N) *work per
    likelihood call* by design -- that is what buys an exact inferred correlation
    time. Measured: one trace per ``remember`` and none thereafter. During a NUTS
    run N is fixed, so the cost is one compilation, not one per step.

    **Ordered is a property of the type, not of ``remember``.** The stack is
    what the recursion reads and the archive is what names it, and until
    :func:`_reject_a_foreign_stack` existed the constructor related the two in
    no way at all -- a reversed archive over an unchanged stack, a reversed
    stack under an unchanged archive, six blocks with two epochs and six blocks
    with none were all accepted, and the first two answer with a plausible
    number. See that function for what each one costs.

    Attributes:
        factorization: the single declaration. Its ``linked`` entry supplies the
            transition, and ``__check_init__`` has already refused a transition
            built from anything that is not global.
        stacked: ``(factor (N, w, w), target (N, w), offset (N,))``, epochs in
            the order they were remembered, ``zeta``'s columns last. Checked at
            construction against the archive, block by block.
    """

    factorization: Any
    stacked: tuple[jax.Array, jax.Array, jax.Array]
    _epochs: Any = eqx.field(default=None)

    def __init__(
        self,
        factorization: Any,
        stacked: tuple[jax.Array, jax.Array, jax.Array] | None = None,
        epochs: Any = (),
    ):
        self.factorization = factorization
        if len(factorization.linked_names) != 1:
            raise StateValidationError(
                f"ChainMemory carries exactly one linked latent; this factorization "
                f"declares {list(factorization.linked_names)}. Two independent "
                "chains are two memories -- accumulating them together would need "
                "one joint transition, which is a different model from two, and "
                "silently so."
            )
        width = self._width(factorization)
        self.stacked = (
            (jnp.zeros((0, width, width)), jnp.zeros((0, width)), jnp.zeros((0,)))
            if stacked is None
            else stacked
        )
        self._epochs = epochs if isinstance(epochs, _Epochs) else _Epochs(epochs)
        _reject_a_foreign_stack(self.stacked, self._epochs.terms, self.column_order, width)

    @staticmethod
    def _width(factorization: Any) -> int:
        globals_width = sum(int(jnp.zeros(shape).size) for shape in factorization.global_shapes)
        transition = factorization.linked[factorization.linked_names[0]]
        return globals_width + transition.width

    @property
    def linked_name(self) -> str:
        """The one latent that is a Markov chain across epochs."""
        return self.factorization.linked_names[0]

    @property
    def transition(self) -> Any:
        """Its transition -- fixed, or a builder resolved inside the likelihood."""
        return self.factorization.linked[self.linked_name]

    @property
    def archive(self) -> tuple[Any, ...]:
        """The stored terms, oldest first."""
        return self._epochs.terms

    @property
    def epoch_ids(self) -> tuple[str, ...]:
        """The recordings' data hashes, in the order they were remembered."""
        return tuple(term.epoch_id for term in self._epochs.terms)

    @property
    def column_order(self) -> tuple[str, ...]:
        """What a stored block is a quadratic form in, ``zeta`` last."""
        return self.factorization.global_names + (self.linked_name,)

    def remember(
        self, term: Any, duplicate: bool = False, shared_inputs: bool = False
    ) -> "ChainMemory":
        """A new memory holding this epoch **last**. The original is unchanged.

        Order is the content here, not a convenience: epoch *e*'s drift is
        correlated with *e-1*'s and not with *e+3*'s, so appending out of order
        is a different model rather than the same one shuffled. Measured on
        ``tests/evidence/chain_bank.py``, swapping two adjacent epochs moves the
        campaign's log-likelihood by 0.0752 nats; a bag's ``remember`` moves it
        by roundoff, which is what its own tests pin. Small in absolute terms
        and 1e12 times the recursion's own 9.1e-13 disagreement with the dense
        oracle, which is the comparison that makes it evidence.

        **``duplicate=True`` is refused here, where the bag takes it**, and the
        asymmetry is the same one the two types exist to carry. For a bag it
        means "count this recording twice": the terms are exchangeable, the
        result is a well-defined posterior that is too narrow by a known amount,
        and a caller who wrote it meant it. A chain has no such reading. The
        repeat lands **last**, so it does not say "``e1`` twice", it says "``e1``
        happened, then ``e2``, then ``e1`` again" -- one night's drift correlated
        with itself across two transitions. Measured at ``PROBES[0]``:
        ``('e0', 'e1', 'e2')`` is -58.9892 nats, appending ``e1`` gives
        -68.6127, and the same double count placed in order gives -68.4998. The
        0.1129 nats between the last two is the part no bag can produce, and it
        is larger than the 0.0752 above. There is no flag value that means
        "twice, in the right place", because a chain has no right place for a
        night that happened once: use :class:`BayesMemory` if the epochs really
        are exchangeable, or give the second recording its own ``epoch_id``.

        Args:
            term: one epoch's compressed likelihood, over this memory's globals
                **and** its linked latent.
            duplicate: refused. Present so that a caller who reaches for the
                bag's flag gets a refusal that says why rather than a
                ``TypeError`` about a keyword.
            shared_inputs: allow an input product this memory already holds under
                the same hash. Section 9.5, and it is the *bag*'s rule reused
                rather than restated: a chain already says the epochs are
                dependent through ``zeta``, and a shared calibration solution is
                a second dependence the chain does not model.
        """
        from rheplicant.inference.memory import reject_bad_term

        if duplicate:
            raise StateValidationError(
                "ChainMemory.remember does not take duplicate=True. A bag counts "
                "a recording twice and stays a posterior over the same model; a "
                "chain appends the repeat at the end, which says the same night "
                "happened again after the ones that followed it and its drift is "
                "correlated with itself through the transition. Measured on "
                "tests/evidence/chain_bank.py, that costs 0.1129 nats beyond the "
                "double count itself. Use BayesMemory if the epochs are "
                "exchangeable, or give the second recording its own epoch_id."
            )
        reject_bad_term(
            term,
            self._epochs.terms,
            self._epochs.ids,
            duplicate,
            self._latents_ok,
            self.factorization.represents,
            shared_inputs,
            _CHAIN_REPEAT_REMEDY,
        )
        square = _square_block(term.info, self.column_order)
        factors, targets, offsets = self.stacked
        return ChainMemory(
            self.factorization,
            (
                jnp.concatenate([factors, square.factor[None]], axis=0),
                jnp.concatenate([targets, square.target[None]], axis=0),
                jnp.concatenate([offsets, jnp.asarray(square.offset)[None]], axis=0),
            ),
            self._epochs.appended(term),
        )

    def _declared_widths(self) -> dict[str, int]:
        """``{column: how many entries it contributes}``, ``zeta`` last.

        The globals' widths come from the factorization's shapes and the chain's
        from the transition, which is the same arithmetic :meth:`_width` does --
        written once here so the admission check and the stack cannot disagree
        about how wide a block should be.
        """
        widths = {
            name: int(jnp.zeros(shape).size)
            for name, shape in zip(
                self.factorization.global_names,
                self.factorization.global_shapes,
                strict=True,
            )
        }
        widths[self.linked_name] = self.transition.width
        return widths

    def _latents_ok(self, term: Any) -> None:
        """A chain's half of the admission rules -- the mirror of the bag's.

        Four questions, and it used to ask two. Present and no strays were
        checked; **every declared global present**, and **at the declared
        width**, were not, and each has a stored block that reaches the
        accumulator and dies there without naming anything.

        A term over a *subset* -- an epoch compressed with
        ``design={"t_rx": ..., "t_rx_drift": ...}`` against a factorization that
        also declares ``gain_slope`` -- reached ``_square_block``, whose
        ``columns[name]`` lookup is keyed on the term's own names, and raised a
        bare ``KeyError('gain_slope')``. That is a night where one global was
        not exercised, not a typo. ``BayesMemory._latents_ok`` has the mirror
        check (``tuple(term.latents) != declared``) and this half did not.

        A linked latent declared at width 1 and carried at width 2 reached
        ``jnp.concatenate`` and raised ``TypeError: ... shapes (0, 3, 3),
        (1, 4, 4)``. :func:`_check_block_width` says that properly, and existed
        only on the read path, where the blocks are already stacked and the
        damage is already in them.

        **A set, not a tuple.** ``compress_linear`` names the columns in the
        caller's dict order, and a night written drift-first is the same
        information in a differently ordered vector -- ``_square_block``
        permutes it, and comparing ordered tuples here would refuse the
        campaign ``test_a_term_whose_columns_arrive_in_another_order...``
        measures against the dense oracle.

        **Exact, where the bag's is not.** A reduced-basis term legitimately
        expands in a subset of the latents, and ``BayesMemory._latents_ok`` has
        a branch for exactly that. It cannot be reached from here: a T1 term's
        stored columns are ``(COEFFICIENTS,)``, so the linked-latent refusal
        below turns it away first. The two rules do not collide, and this one
        must not be softened to imitate the other.
        """
        from rheplicant.inference.memory import _stored_names

        stored = _stored_names(term)
        if self.linked_name not in stored:
            raise StateValidationError(
                f"Term {term.epoch_id!r} is over {list(stored)}, which does not "
                f"include the linked latent {self.linked_name!r}. A chain memory "
                "integrates that latent out itself, across epochs -- a term that "
                "already marginalised it against an independent prior has spent "
                "the correlation, and folding it in here would add the chain's "
                "prior a second time. Compress the epoch with the linked latent "
                "among its design blocks, or use BayesMemory."
            )
        declared = set(self.column_order)
        stray = [name for name in stored if name not in declared]
        if stray:
            raise StateValidationError(
                f"Term {term.epoch_id!r} is over {stray}, which this memory does "
                f"not declare; it accumulates {list(self.column_order)}."
            )
        missing = [name for name in self.column_order if name not in stored]
        if missing:
            raise StateValidationError(
                f"Term {term.epoch_id!r} is over {list(stored)} and does not carry "
                f"{missing}, which this memory declares; it accumulates "
                f"{list(self.column_order)} and every block must be square in all "
                "of them. A night that did not exercise one global is still a "
                "quadratic form in it -- an all-zero design block, which says "
                "the epoch constrains that latent not at all and is the normal "
                "rank-deficient case the square-root form exists to carry. "
                "Compress the epoch with a design block for each declared "
                "latent, or accumulate it in a BayesMemory over the latents it "
                "does have."
            )
        widths = self._declared_widths()
        stored_widths = {
            name: int(jnp.zeros(shape).size)
            for name, shape in zip(term.info.names, term.info.shapes, strict=True)
        }
        wrong = [
            (name, stored_widths[name], widths[name])
            for name in self.column_order
            if stored_widths[name] != widths[name]
        ]
        if wrong:
            detail = ", ".join(
                f"{name!r} carries {got} column(s) where this memory declares {want}"
                for name, got, want in wrong
            )
            raise StateValidationError(
                f"Term {term.epoch_id!r} does not have the declared widths: "
                f"{detail}. The stored blocks are a quadratic form in one "
                "specific ordered vector, and the filter slices at the "
                "globals' width to separate theta's columns from the chain's -- "
                "a block of another width makes that slice take one for the "
                "other. Checked at admission because the accumulator stacks "
                "these into a single array: unguarded, the symptom was a raw "
                "TypeError out of jnp.concatenate naming only two shapes."
            )

    def log_likelihood(self, values: dict[str, jax.Array]) -> jax.Array:
        """``log p(d_1:N | theta)`` with the chain integrated out. No prior."""
        return chain_log_likelihood(
            self.stacked,
            self.transition,
            values,
            names=self.factorization.global_names,
            shapes=self.factorization.global_shapes,
        )

    def log_posterior(self, values: dict[str, jax.Array]) -> jax.Array:
        """The chain's likelihood plus the prior, applied exactly once."""
        total = self.log_likelihood(values)
        for name, prior in self.factorization.global_priors.items():
            total = total + jnp.sum(prior.log_prob(values[name]))
        return total

    def marginal(self, values: dict[str, jax.Array]) -> SqrtInfo:
        """The campaign's quadratic form in theta, at these transition values."""
        return chain_marginal(
            self.stacked,
            self.transition,
            values,
            names=self.factorization.global_names,
            shapes=self.factorization.global_shapes,
        )

    def fisher(self, at: dict[str, jax.Array]):
        """``sum_e F_e`` after the chain is integrated out, with named rows.

        ``at`` is required rather than defaulted, for the reason
        :meth:`~rheplicant.inference.memory.BayesMemory.fisher` refuses to
        default it one layer along: with a :class:`HyperTransition` the marginal
        curvature is a function of theta, and a fixed default point would be a
        linearisation nobody declared and nothing could see -- the matrix comes
        back finite, symmetric and PSD whichever point it was taken at.

        The permutation into flatten order is
        :meth:`~rheplicant.inference.memory.BayesMemory.fisher`'s, reached by
        wrapping the marginal in a throwaway bag rather than by building a
        ``FlatMatrix`` here. A second copy would reintroduce Plan A's own bug
        invisibly: ``chain_marginal`` returns columns in *declared* order, and
        the two orders coincide exactly when the latents are alphabetical.
        """
        from rheplicant.inference.memory import BayesMemory

        return BayesMemory(self.factorization, self.marginal(at)).fisher()

    def to_numpyro_model(self, **unsupported: Any):
        """Sample the globals against this chain. Refuses a ``noise_std=``.

        The closure is over the stacked blocks and the transition -- the density
        path -- and not over ``self``, which also holds the archive. That archive
        grows with the campaign, but the stack does too (deviation 12), so what
        this buys is one retrace per ``remember`` rather than none: N is fixed
        for the whole of a sampling run, and the compilation is paid once.
        """
        from rheplicant.inference.memory import BayesMemory

        stacked = self.stacked
        transition = self.transition
        names = self.factorization.global_names
        shapes = self.factorization.global_shapes

        def density(values: dict[str, jax.Array]) -> jax.Array:
            return chain_log_likelihood(stacked, transition, values, names=names, shapes=shapes)

        return BayesMemory(
            self.factorization,
            SqrtInfo.null(names, shapes),
        )._numpyro_model(density, **unsupported)
