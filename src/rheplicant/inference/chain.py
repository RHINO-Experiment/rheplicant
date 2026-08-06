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

===================================================  ==============
dropped                                              nats
===================================================  ==============
initial ``zeta`` prior normalisation                 +0.9189
per-transition ``-0.5 logdet(2 pi Q)``, five of them +2.8618
the spec's shorthand for it (``0.5 logdet Q^-1``)    +4.5947
marginalisation constant, six of them                +7.2619
the fold corner ``-0.5 rho^2``, six of them          +45.9502
the masked data normalisation                        -6.8408
===================================================  ==============

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

from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.sqrtinfo import SqrtInfo, marginalise_arrays


class LinearGaussianTransition(eqx.Module):
    """``zeta_{e+1} = phi zeta_e + w``, ``w ~ N(0, diag(process_std)^2)``.

    Attributes:
        phi: ``(n, n)``. A full matrix, because a multi-component drift can
            rotate; the process and initial spreads are diagonal because a
            correlated *innovation* is a modelling claim nobody has made and a
            silently-accepted full covariance would need its own Cholesky
            refusal.
        process_std: ``(n,)``, strictly positive.
        initial_std: ``(n,)``, strictly positive -- ``sd(zeta_1)``.
        initial_mean: ``(n,)``. Zero unless declared.
        hyper: empty for a fixed transition. Present so that
            :class:`~rheplicant.inference.factorize.Factorization` can ask one
            question of either type.

    **Positivity is checked here and nowhere else, on purpose.** The rows this
    class contributes are what constrain every ``zeta_e``, so a strictly
    positive spread makes each marginalisation's block full-rank *by
    construction* -- and that is what lets the filter call
    :func:`~rheplicant.inference.sqrtinfo.marginalise_arrays` inside a
    ``lax.scan`` instead of the checked
    :func:`~rheplicant.inference.sqrtinfo.marginalise`, which concretises and
    therefore cannot be traced or differentiated. One eager check at
    declaration, not one traced check per epoch of a thousand.

    A traced spread is **not** checked, and cannot be: a
    :class:`HyperTransition` builds these blocks from theta, and under NUTS
    theta goes wherever it likes. Parameterise the builder so that positivity is
    structural -- return ``jnp.exp(log_sigma)``, never a raw sampled scale --
    which is why this class takes standard deviations rather than a covariance.
    """

    phi: jax.Array
    process_std: jax.Array
    initial_std: jax.Array
    initial_mean: jax.Array
    hyper: tuple[str, ...] = eqx.field(static=True, default=())

    def __init__(
        self,
        phi: Any,
        process_std: Any,
        initial_std: Any,
        initial_mean: Any = None,
        hyper: Sequence[str] = (),
    ):
        self.process_std = jnp.atleast_1d(jnp.asarray(process_std))
        self.initial_std = jnp.atleast_1d(jnp.asarray(initial_std))
        self.phi = jnp.atleast_2d(jnp.asarray(phi))
        self.initial_mean = (
            jnp.zeros_like(self.initial_std)
            if initial_mean is None
            else jnp.broadcast_to(
                jnp.atleast_1d(jnp.asarray(initial_mean)), self.initial_std.shape
            )
        )
        self.hyper = tuple(hyper)

    def __check_init__(self):
        width = int(self.process_std.shape[0])
        if self.phi.shape != (width, width):
            raise StateValidationError(
                f"phi is {self.phi.shape} but process_std has {width} "
                "component(s), so the transition maps the chain into a space of a "
                "different size. Broadcasting one into the other would build a "
                "chain nobody declared."
            )
        if self.initial_std.shape != (width,):
            raise StateValidationError(
                f"initial_std is {self.initial_std.shape} but process_std is "
                f"{self.process_std.shape}; they describe the same chain."
            )
        for name, spread in (
            ("process_std", self.process_std),
            ("initial_std", self.initial_std),
        ):
            # A traced spread cannot be judged here and must not be pretended
            # about -- see the class docstring.
            if isinstance(spread, jax.core.Tracer):
                continue
            # `not (... > 0)` rather than `... <= 0`, because every comparison
            # against NaN is False and the second form waves a NaN spread
            # through. `isfinite` is separate rather than folded in for the same
            # reason: `inf > 0` is True, and an infinite spread makes
            # `1 / process_std` zero, which is a transition row of zeros -- the
            # density comes back -inf from the log-determinant, a thousand epochs
            # after the declaration that caused it.
            if not bool(jnp.all(jnp.isfinite(spread) & (spread > 0.0))):
                raise StateValidationError(
                    f"{name} must be finite and strictly positive; got {spread}. "
                    "These rows are what constrain zeta at each marginalisation, so "
                    "a zero leaves the Gaussian integral over that epoch divergent "
                    "-- and inside a lax.scan finite arithmetic returns a large "
                    "plausible number for it rather than an infinity anyone would "
                    "notice. A chain that genuinely does not move is "
                    "process_std=1e-9, not 0.0; one that is effectively unlinked is "
                    "1e12, not inf, which would zero the transition rows and send "
                    "the whole campaign's density to -inf; and a quantity that is "
                    'constant across the campaign is scope="global".'
                )

    @property
    def width(self) -> int:
        """How many components the chain carries."""
        return int(self.process_std.shape[0])

    def at(self, values: dict[str, jax.Array]) -> "LinearGaussianTransition":
        """Itself. A fixed transition does not depend on theta -- that is the claim."""
        return self


def ornstein_uhlenbeck(
    tau: Any, sigma: Any, width: int = 1, hyper: Sequence[str] = ()
) -> LinearGaussianTransition:
    """A stationary OU chain: correlation time ``tau`` in epochs, spread ``sigma``.

    ``phi = exp(-1/tau)`` and ``process_std = sigma sqrt(1 - phi^2)``, so
    ``var(zeta_{e+1}) = phi^2 var + Q`` returns ``sigma^2`` when it starts there
    -- stationarity is arithmetic here, not an assumption, and
    ``tests/evidence/test_transition.py`` pins it.

    A **function**, not a class: section 11's sketch writes
    ``OrnsteinUhlenbeck(...)`` as though it were a type, but the type the filter
    consumes -- and the type a :class:`HyperTransition` builder must return --
    is :class:`LinearGaussianTransition`. An OU is a way of constructing one, and
    this package spells constructors in lower case.
    """
    decay = jnp.exp(-1.0 / jnp.asarray(tau))
    return LinearGaussianTransition(
        phi=decay * jnp.eye(width),
        process_std=jnp.broadcast_to(
            jnp.asarray(sigma) * jnp.sqrt(1.0 - decay**2), (width,)
        ),
        initial_std=jnp.broadcast_to(jnp.asarray(sigma), (width,)),
        hyper=hyper,
    )


class HyperTransition(eqx.Module):
    """A transition whose blocks are functions of theta -- section 6's ``linked_hyper``.

    Attributes:
        build: ``{global latent: value} -> LinearGaussianTransition``. Static: it
            is code. Called **inside** the theta likelihood, so it must be
            traceable, and it must return blocks that are positive for every
            theta the sampler can reach -- ``exp`` of a sampled log-scale, not a
            sampled scale.
        hyper: which global latents ``build`` reads. Declared rather than
            inferred, because
            :class:`~rheplicant.inference.factorize.Factorization` checks them
            and a closure's free variables are not inspectable.
        width: how many components the chain carries. Declared for the same
            reason the shape of anything else is: the filter needs it before it
            has a value to look at.
    """

    build: Callable[[dict[str, jax.Array]], LinearGaussianTransition] = eqx.field(
        static=True
    )
    hyper: tuple[str, ...] = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def at(self, values: dict[str, jax.Array]) -> LinearGaussianTransition:
        """Resolve the blocks at these latent values."""
        missing = [name for name in self.hyper if name not in values]
        if missing:
            raise StateValidationError(
                f"This transition is built from {list(self.hyper)}; no value was "
                f"given for {missing}. A linked_hyper chain is evaluated inside the "
                "theta likelihood precisely so those values are current -- "
                "resolving it against a default would be the compression-time "
                "pinning the sub-scope exists to prevent."
            )
        resolved = self.build({name: values[name] for name in self.hyper})
        if resolved.width != self.width:
            raise StateValidationError(
                f"This transition declares width {self.width} but its builder "
                f"returned {resolved.width}. The stored per-epoch blocks were "
                "shaped against the declared number and cannot be re-cut now."
            )
        return resolved


def _initial_log_norm(transition: LinearGaussianTransition) -> jax.Array:
    """``-0.5 logdet(2 pi P0)`` -- the prior on ``zeta_1``.

    A module-level function rather than three inline terms so that a test can
    delete exactly this constant and measure what it was worth. It belongs to
    nobody else: the per-epoch blocks know nothing about the chain, and
    :func:`~rheplicant.inference.sqrtinfo.marginalise_arrays` carries only the
    integral's own constant. Measured cost of dropping it: +0.9189 nats, which is
    exactly ``0.5 log(2 pi)`` for a scalar chain at ``P0 = 1``.
    """
    return -0.5 * transition.width * jnp.log(2.0 * jnp.pi) - jnp.sum(
        jnp.log(transition.initial_std)
    )


def _transition_log_norm(transition: LinearGaussianTransition) -> jax.Array:
    """``-0.5 logdet(2 pi Q)`` -- one per augmentation.

    Section 6 names only ``0.5 logdet Q^-1``, and the ``2 pi`` half is not
    optional: it cancels against the marginalisation's ``+0.5 n log(2 pi)``,
    which is why the shorthand reads plausible. Keeping only the log-determinant
    while calling :func:`~rheplicant.inference.sqrtinfo.marginalise_arrays`
    leaves ``+0.5 log(2 pi)`` per transition -- measured +4.5947 nats over this
    fixture's five transitions, so +918 nats over a thousand-epoch campaign, and
    no effect on any posterior mean, width or gradient. Dropping the whole term
    instead costs +2.8618.
    """
    return -0.5 * transition.width * jnp.log(2.0 * jnp.pi) - jnp.sum(
        jnp.log(transition.process_std)
    )


def _fold(
    factor: jax.Array,
    target: jax.Array,
    offset: jax.Array,
    block: tuple[jax.Array, jax.Array, jax.Array],
    width: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Add one epoch's evidence to the running joint form.

    :meth:`~rheplicant.inference.sqrtinfo.SqrtInfo.combine`'s arithmetic on raw
    arrays, because a ``lax.scan`` carry cannot afford a name lookup and a
    re-validation per epoch. The corner is the largest of the six constants --
    measured +45.9502 nats over six epochs, and it grows with the campaign.
    """
    block_factor, block_target, block_offset = block
    upper = jnp.linalg.qr(
        jnp.concatenate(
            [
                jnp.concatenate([factor, target[:, None]], axis=1),
                jnp.concatenate([block_factor, block_target[:, None]], axis=1),
            ],
            axis=0,
        ),
        mode="r",
    )
    keep = min(upper.shape[0], width)
    corner = upper[keep:, width]
    return (
        upper[:keep, :width],
        upper[:keep, width],
        offset + block_offset - 0.5 * jnp.sum(corner**2),
    )


def chain_marginal(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> SqrtInfo:
    """``zeta_1:N`` integrated out exactly, leaving a quadratic form in theta.

    Args:
        blocks: ``(factor (N, w, w), target (N, w), offset (N,))`` -- one square
            per-epoch joint form over ``(*names, zeta)``, ``zeta``'s columns
            **last**. Square rather than ragged because ``lax.scan`` needs one
            shape per iteration; ``SqrtInfo.combine(SqrtInfo.null(...), info)``
            is the padding, and it is the same QR the accumulator uses, so the
            offset it produces is the one this consumes, corner included.
        transition: a :class:`LinearGaussianTransition` or a
            :class:`HyperTransition`. Resolved once, here, against ``values`` --
            which is what makes an inferred correlation time inferred rather
            than pinned at compression time.
        values: the global latents. Read for the transition's hyperparameters
            and for the returned form's evaluation point.
        names: the global latents, in the same column order the blocks were
            built in.
        shapes: each of those latents' shapes, in the same order.

    Returns:
        A :class:`~rheplicant.inference.sqrtinfo.SqrtInfo` over ``names`` whose
        ``log_prob`` is ``log p(d_1:N | theta)``. Prior-free, like every other
        stored factor in this layer.

    **Why the scan stops one short, and what that is *not* about.** The plan this
    was built from warned that scanning over all ``N`` blocks and then
    marginalising once more would integrate a ``zeta_{N+1}`` no data constrained
    and come back "finite and wrong by one transition's worth of constants".
    Measured, by writing it that way: it comes back **exact**, 4.5e-13 from the
    dense oracle at all four probes -- because the extra transition density
    integrates to one over its own argument, so its normalisation and the extra
    marginalisation's constant cancel term for term. The reason to stop one short
    is therefore cost, one QR per call, not correctness. What the extra step does
    change is the *count* of each constant, five transitions becoming six and six
    marginalisations becoming seven, which is what
    ``test_the_transition_normalisation_is_the_whole_density_not_half_of_it`` and
    ``test_the_marginalisation_constant_is_carried`` notice and the exactness
    tests cannot.
    """
    factors, targets, offsets = blocks
    resolved = transition.at(values)
    n_zeta = resolved.width
    n_theta = sum(int(jnp.zeros(shape).size) for shape in shapes)
    width = n_theta + n_zeta
    if factors.shape[-1] != width:
        raise StateValidationError(
            f"The stored blocks are {factors.shape[-1]} columns wide but "
            f"{list(names)} plus a width-{n_zeta} chain is {width}. The blocks are "
            "a quadratic form in a specific ordered vector; reading them against a "
            "different one is not a rename, it is a different model."
        )

    inverse_process = 1.0 / resolved.process_std
    inverse_initial = 1.0 / resolved.initial_std

    # zeta_1's prior, and nothing else: theta's prior lives on the
    # Factorization, and a stored factor is prior-free in theta by construction.
    carry_factor = (
        jnp.zeros((width, width)).at[n_theta:, n_theta:].set(jnp.diag(inverse_initial))
    )
    carry_target = (
        jnp.zeros(width).at[n_theta:].set(inverse_initial * resolved.initial_mean)
    )
    carry_offset = _initial_log_norm(resolved)

    # `[zeta_e | theta | zeta_{e+1}]`: marginalise_arrays takes the block first.
    augment_order = jnp.asarray(
        list(range(n_theta, width))
        + list(range(n_theta))
        + list(range(width, width + n_zeta)),
        dtype=int,
    )
    final_order = jnp.asarray(
        list(range(n_theta, width)) + list(range(n_theta)), dtype=int
    )
    # `Q^-1/2 (zeta_{e+1} - phi zeta_e)`, with zero response to theta: condition
    # C1b's locality, written into the rows rather than assumed. The scaling is
    # `diag(1/q) @ phi`, not `phi @ diag(1/q)`; the two coincide for a scalar
    # chain, which is why a wide chain is what tests the ordering.
    transition_rows = jnp.concatenate(
        [
            jnp.zeros((n_zeta, n_theta)),
            -inverse_process[:, None] * resolved.phi,
            jnp.diag(inverse_process),
        ],
        axis=1,
    )
    transition_constant = _transition_log_norm(resolved)

    def step(carry, block):
        factor, target, offset = _fold(*carry, block, width)
        widened = jnp.concatenate(
            [factor, jnp.zeros((factor.shape[0], n_zeta))], axis=1
        )
        joint = jnp.concatenate([widened, transition_rows], axis=0)[:, augment_order]
        joint_target = jnp.concatenate([target, jnp.zeros(n_zeta)])
        factor, target, offset, _ = marginalise_arrays(
            joint, joint_target, offset + transition_constant, n_zeta
        )
        return (factor, target, offset), None

    # Every epoch but the last is folded and then advanced; the last is folded
    # and then integrated out, because it has no successor to hand the chain to.
    (factor, target, offset), _ = jax.lax.scan(
        step,
        (carry_factor, carry_target, carry_offset),
        (factors[:-1], targets[:-1], offsets[:-1]),
    )
    factor, target, offset = _fold(
        factor, target, offset, (factors[-1], targets[-1], offsets[-1]), width
    )
    factor, target, offset, _ = marginalise_arrays(
        factor[:, final_order], target, offset, n_zeta
    )
    return SqrtInfo(
        factor=factor, target=target, offset=offset, names=names, shapes=shapes
    )


def chain_log_likelihood(
    blocks: tuple[jax.Array, jax.Array, jax.Array],
    transition: Any,
    values: dict[str, jax.Array],
    names: tuple[str, ...],
    shapes: tuple[tuple[int, ...], ...],
) -> jax.Array:
    """``log p(d_1:N | theta)``, the chain integrated out exactly. No prior."""
    return chain_marginal(blocks, transition, values, names, shapes).log_prob(values)


def _square_block(info: SqrtInfo, order: tuple[str, ...]) -> SqrtInfo:
    """One epoch's joint form, permuted into ``order`` and padded to square.

    Square because ``lax.scan`` needs one shape per iteration, and by
    ``combine(null, info)`` rather than by ``jnp.pad`` because that is the QR the
    accumulator already uses: the offset it returns is the one the filter
    consumes, corner included. Padding with zeros would produce the same factor
    and a **different offset**, which is exactly the class of error section 6 is
    most exposed to -- the fold's corner is the largest of the six constants at
    +45.95 nats over this fixture's six epochs.
    """
    if tuple(info.names) != order:
        shapes = dict(zip(info.names, info.shapes, strict=True))
        columns: dict[str, range] = {}
        position = 0
        for name, shape in zip(info.names, info.shapes, strict=True):
            size = int(jnp.zeros(shape).size)
            columns[name] = range(position, position + size)
            position += size
        permutation = jnp.asarray(
            [column for name in order for column in columns[name]], dtype=int
        )
        info = SqrtInfo(
            factor=info.factor[:, permutation],
            target=info.target,
            offset=info.offset,
            names=order,
            shapes=tuple(shapes[name] for name in order),
        )
    return SqrtInfo.combine(SqrtInfo.null(info.names, info.shapes), info)


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
        self.ids = (
            frozenset(term.epoch_id for term in self.terms) if ids is None else ids
        )

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

    Attributes:
        factorization: the single declaration. Its ``linked`` entry supplies the
            transition, and ``__check_init__`` has already refused a transition
            built from anything that is not global.
        stacked: ``(factor (N, w, w), target (N, w), offset (N,))``, epochs in
            the order they were remembered, ``zeta``'s columns last.
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

    @staticmethod
    def _width(factorization: Any) -> int:
        globals_width = sum(
            int(jnp.zeros(shape).size) for shape in factorization.global_shapes
        )
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

    def remember(self, term: Any, duplicate: bool = False) -> "ChainMemory":
        """A new memory holding this epoch **last**. The original is unchanged.

        Order is the content here, not a convenience: epoch *e*'s drift is
        correlated with *e-1*'s and not with *e+3*'s, so appending out of order
        is a different model rather than the same one shuffled. Measured on
        ``tests/evidence/chain_bank.py``, swapping two adjacent epochs moves the
        campaign's log-likelihood by 0.3675 nats; a bag's ``remember`` moves it
        by roundoff, which is what its own tests pin.

        Args:
            term: one epoch's compressed likelihood, over this memory's globals
                **and** its linked latent.
            duplicate: allow an ``epoch_id`` already present. Off by default,
                because the common cause is a retried run.
        """
        from rheplicant.inference.memory import reject_bad_term

        reject_bad_term(
            term,
            self._epochs.terms,
            self._epochs.ids,
            duplicate,
            self._latents_ok,
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

    def _latents_ok(self, term: Any) -> None:
        """A chain's half of the admission rules -- the mirror of the bag's."""
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
            return chain_log_likelihood(
                stacked, transition, values, names=names, shapes=shapes
            )

        return BayesMemory(
            self.factorization,
            SqrtInfo.null(names, shapes),
        )._numpyro_model(density, **unsupported)
