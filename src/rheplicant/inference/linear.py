"""Declared-linear parameter blocks: check the claim, then export the operator.

Some parameters enter the forward model **linearly** — sky ``alm``
coefficients, noise-wave amplitudes, any component whose contribution is a
matrix acting on it. Those blocks are also the big ones: a sky at ``lmax=191``
across 32 channels is ~10⁶ real degrees of freedom, where gradient-based
samplers are hopeless but a conjugate-Gaussian solve is exactly right.

Declaring ``Latent(..., linear=True)`` promises that, holding every other
latent fixed, the prediction is an **affine** function of this one::

    prediction(x) = A x + b

Two things follow. First, the promise is *checkable*, and this module checks
it before anything exploits it — :func:`check_linearity` compares the model
against its own linearization. A false declaration would otherwise produce a
confident, wrong posterior instead of an error.

Second, ``A`` and ``Aᵀ`` are available without ever forming a matrix:
``jax.linearize`` gives the forward action and ``jax.vjp`` the adjoint, at the
cost of one trace. :func:`linear_operator` packages them as a
:class:`LinearBlock`, which is the whole interface the conjugate-Gaussian
routines here need: :func:`wiener_solve` for the posterior mean and
:func:`gcr_sample` for an exact posterior draw.

Because the block is affine only *given* the other latents, both take ``at=``
to rebuild it wherever those currently are — which is what makes a Gibbs
sweep possible: draw the linear block exactly, update the nonlinear ones
however you like, repeat.

**One block may hold several latents.** ``linear_operator(..., names=("t_nw",
"t_ant"))`` exports the joint operator over a *group*, whose ``x`` is a
``{name: array}`` dict rather than one array — and whose solve returns the same
dict, so the physical names survive instead of the caller slicing an anonymous
stacked vector and getting the offsets right by hand. Nothing is concatenated:
the group's domain is a pytree, ``cg`` already solves over pytrees, and the
prior is block-diagonal by construction because each latent's ``S`` sits on its
own leaf.

Grouping is not cosmetic. Two latents the data cannot tell apart are solved
*together* in one CG here, whereas alternating between them as two blocks
converges at the rate of their correlation — and reports a per-block residual
of 1e-7 and a per-block ``κ`` of 1 the whole way down, because both numbers are
computed from the block and neither can see across the partition. The joint
``κ`` this module reports for the grouped block *can*; so can
:func:`~rheplicant.inference.identifiability.identifiability`, which is the
right instrument for choosing the partition in the first place. And a group
whose members are only *pairwise* linear — a gain against an antenna
temperature — is refused by :func:`check_linearity`, which probes the joint map
and finds it bilinear, not affine.

**Where the prior comes from.** ``S`` is read off ``Latent(prior=...)`` — the
same declaration :func:`~rheplicant.inference.numpyro_bridge.to_numpyro_model`
reads, so one space handed to NUTS and to :func:`gcr_sample` targets one
posterior. The ``prior_std=`` / ``prior_mean=`` keywords remain for a
prior-free latent, but a keyword that *contradicts* a declaration is refused
rather than allowed to win, and a declared prior with no conjugate Gaussian
form is refused rather than approximated by its first two moments. Both would
otherwise be a finite, confident posterior for a model nobody declared, which
is the failure mode every guard in this module is placed against.

**Probe at extreme scales.** :func:`check_linearity` probes at 10⁻³, 1 and 10³
times the latent's own magnitude, because near-linearity is scale-dependent:
``x + εx²`` is indistinguishable from linear near the origin and grossly
nonlinear far from it. A probe suite that only samples "reasonable" values
signs off on exactly the blocks that will fail in a sampler's tails.

**A residual is not an accuracy.** The solvers here are iterative, and what an
iterative method can cheaply report is ``‖M x - b‖``, not ``‖x - x*‖``. The two
differ by the condition number of ``M = AᵀN⁻¹A + S⁻¹``, and κ is large here by
*design*: whenever the data does not fully identify the block — which is the
case the prior is for — ``λ_min(M)`` is exactly ``1/prior_std²`` and κ runs to
1e6 and beyond. A solve can then sit at a relative residual of 1e-7 with the
prior-dominated directions untouched, and a draw comes back with almost no
scatter where it should have carried the whole prior width. So the guard on
these solves bounds the *error*, ``κ · residual``, and :func:`condition_estimate`
exposes κ for choosing ``tol``.
"""

import dataclasses
from collections.abc import Callable, Sequence
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.conditioning import extreme_eigenvalues, tree_norm
from rheplicant.inference.likelihood import check_observed_shape
from rheplicant.inference.noise import NoiseModel, check_noise_std_axis
from rheplicant.inference.parameters import ParameterSpace

DEFAULT_SCALES: tuple[float, ...] = (1e-3, 1.0, 1e3)

#: Power-iteration steps per end of the spectrum in :func:`condition_estimate`.
#: Both ends typically settle within three; this leaves margin at a fixed cost
#: of ``2 * POWER_ITERATIONS`` operator applications per guarded solve.
POWER_ITERATIONS: int = 12

#: ``jax.ensure_compile_time_eval``, under the name of what it does at the one
#: place it is used: run this comparison **now**, on the constants in hand,
#: rather than emitting it into whatever trace happens to be open. See
#: :func:`_agrees`.
_RIGHT_NOW = jax.ensure_compile_time_eval


@dataclasses.dataclass(frozen=True)
class LinearBlock:
    """The affine action of one latent on the prediction: ``A x + offset``.

    Deliberately a plain dataclass rather than an ``eqx.Module``: this is a
    derived linear-algebra *handle*, not a differentiable model. ``forward``
    and ``adjoint`` are closures over a traced computation, so the block is
    something you build where you need it, not a pytree to carry around.

    A block may hold ONE latent or a GROUP of them, and the difference is
    carried by ``name``: a ``str`` for one, a ``tuple[str, ...]`` for a group.
    Everything else follows from what ``x`` then is. For one latent ``x`` is an
    array and ``shape``/``dtype``/``prior`` describe it directly; for a group
    ``x`` is a ``{name: array}`` dict and each of the three is a dict keyed the
    same way — the shape of a pytree being a pytree of shapes, which is the
    reading ``jax.eval_shape`` already uses.

    That is the whole of the generalization, and it is deliberately NOT a
    concatenation over real degrees of freedom. Every solve in this module runs
    on ``jax.tree.map`` and ``jax.scipy.sparse.linalg.cg``, both of which take
    pytrees; keeping the group a pytree means there is no offset arithmetic to
    invert, no ordering to state beyond JAX's own, and ``S`` is block-diagonal
    because each latent's variance sits on its own leaf rather than being
    spliced into a stacked vector at the right index.

    Attributes:
        name: the latent this block belongs to — or, for a group, the tuple of
            them in the caller's own order. :attr:`names` normalizes the two.
        shape: shape of ``x``; for a group, ``{name: shape}``.
        dtype: dtype of ``x``; for a group, ``{name: dtype}``.
        offset: ``prediction(0)`` — everything the other parameters contribute.
            For a group, everything OUTSIDE the group contributes.
        forward: ``x -> A x``, from ``jax.linearize``.
        adjoint: ``y -> Aᵀ y``, from ``jax.vjp``, shaped like ``x``.
        prior: the latent's declared prior, carried through from the
            :class:`~rheplicant.inference.parameters.Latent`. ``None`` for a
            prior-free latent, and for a block assembled by hand; for a group,
            ``{name: prior}`` with a ``None`` per prior-free member. It is what
            lets :func:`wiener_solve` and :func:`gcr_sample` read ``S`` off the
            declaration instead of making the caller hand-pass — and hand-sync
            — the same two numbers at every exit.

    Adjoint convention, which matters as soon as ``x`` is complex (sky
    ``alm`` coefficients are): ``adjoint`` is exactly ``jax.vjp``, and JAX
    returns the *conjugate* gradient for complex inputs. The identity that
    holds is therefore the one over the **real** inner product::

        Re sum(x * adjoint(y))  ==  sum(forward(x) * y)

    and NOT the sesquilinear ``sum(conj(x) * adjoint(y))``. The real pairing is
    the one a Gaussian likelihood forms, so this is the useful convention as
    well as the honest one; ``tests/inference/test_linear_blocks.py`` pins both
    halves so the distinction cannot rot into a silent factor.
    """

    name: str | tuple[str, ...]
    shape: tuple[int, ...] | dict[str, tuple[int, ...]]
    dtype: Any
    offset: jax.Array
    forward: Callable[[Any], jax.Array]
    adjoint: Callable[[jax.Array], Any]
    prior: Any = None

    @property
    def grouped(self) -> bool:
        """Whether this block holds several latents at once."""
        return isinstance(self.name, tuple)

    @property
    def names(self) -> tuple[str, ...]:
        """The latents in this block, in the caller's order — one, or several."""
        return self.name if isinstance(self.name, tuple) else (self.name,)

    def as_dict(self, x: Any) -> dict[str, Any]:
        """``x`` as the ``{name: array}`` mapping every consumer downstream reads.

        A solve returns this block's own domain — a bare array for a ``name=``
        block, a ``{name: array}`` dict for a ``names=`` group — and only the
        second is the shape anything else takes. ``space.forward_fn``'s
        ``forward``, :meth:`~rheplicant.inference.parameters.ParameterSpace.bind`,
        :func:`~rheplicant.inference.uncertainty.fisher_information`,
        :func:`~rheplicant.inference.identifiability.identifiability`'s ``at=``,
        :func:`linear_operator`'s ``at=`` and
        :func:`~rheplicant.inference.engines.conditional_potential` all index by
        latent name, and all six raise on the bare form — with six *different*
        exceptions, none of which names the actual mistake
        (``TypeError: JAX does not support string indexing; got idx='gain'`` is
        the friendliest of them, and it arrives from inside a trace).

        So this is the wrap, and it is deliberately **idempotent over the two
        spellings**: the same one call is correct whether the block was built
        with ``name=`` or with ``names=``, which is what lets calling code stop
        caring which it was. It returns a new dict; the block is untouched.

        Raises:
            ParameterSpaceError: for a group, if ``x`` is not a dict with one
                entry per member — that is someone else's solution, and
                wrapping it would put an array under a name it does not belong
                to.
        """
        if not self.grouped:
            return {self.name: x}
        if isinstance(x, dict) and set(x) == set(self.names):
            return dict(x)
        raise ParameterSpaceError(
            f"This block groups {list(self.names)}, so its solution is already a dict "
            f"with one entry per member and as_dict() has nothing to wrap; it was given "
            f"{type(x).__name__}"
            + (f" keyed by {sorted(x)}" if isinstance(x, dict) else "")
            + ". A bare array here is another block's answer, and wrapping it would file "
            "it under a name it does not belong to."
        )


#: Refusal shared by every entry point that takes both spellings. Named rather
#: than repeated so the two exits cannot drift into saying different things.
_BOTH_SPELLINGS = (
    "takes name= OR names=, not both. `name='gain'` builds a block whose x is one "
    "array; `names=('gain',)` builds a group of one, whose x is {'gain': array} and "
    "whose solve comes back as a dict. Both are legitimate and they are not "
    "interchangeable, so which one you meant cannot be guessed."
)


def _is_complex(dtype: Any) -> bool:
    return bool(jnp.issubdtype(dtype, jnp.complexfloating))


def _resolve_name(space: ParameterSpace, name: str | None) -> str:
    """Pick the linear latent to work on, insisting the declaration exists."""
    if name is None:
        declared = [latent.name for latent in space.latents if latent.linear]
        if not declared:
            raise ParameterSpaceError(
                "No latent in this space is declared linear. A block is only usable as a "
                "linear operator once you assert it — declare it with linear=True, and the "
                "assertion will be checked."
            )
        if len(declared) > 1:
            raise ParameterSpaceError(
                f"This space has several linear latents {declared}; say which latent you mean "
                "by passing name=."
            )
        return declared[0]
    if not space.latent(name).linear:
        raise ParameterSpaceError(
            f"Latent {name!r} is not declared linear=True, so its linear operator is not "
            "meaningful. Declare it, and the claim will be checked."
        )
    return name


def _resolve_names(space: ParameterSpace, names: Sequence[str] | str) -> tuple[str, ...]:
    """The latents to put in one block, in the caller's own order.

    A bare string is one name, the same "one or many" convention
    :class:`~rheplicant.inference.parameters.Bind` and
    :func:`~rheplicant.inference.identifiability.identifiability` use. Without
    it, ``names="gain"`` iterates into ``('g', 'a', 'i', 'n')``.
    """
    selected = (names,) if isinstance(names, str) else tuple(names)
    if not selected:
        raise ParameterSpaceError(
            "linear_operator() needs at least one latent name; names=() would build a "
            "block with no parameters, whose normal operator is empty and whose solve "
            "returns {} with a residual of zero — which reads as a converged answer to "
            "a question nobody asked. Pass name= for one latent."
        )
    unknown = [name for name in selected if name not in space.names]
    if unknown:
        raise ParameterSpaceError(
            f"`names` contains {unknown}, which is not a latent of this space; declared: "
            f"{list(space.names)}."
        )
    repeated = sorted({name for name in selected if selected.count(name) > 1})
    if repeated:
        raise ParameterSpaceError(
            f"`names` lists {repeated} more than once. Two copies of one latent are "
            "exactly degenerate with each other, so the group's normal operator would be "
            "singular in a direction that says nothing whatever about the model — and "
            "the {name: array} solution has one entry per name, so one copy's answer "
            "would silently overwrite the other's."
        )
    not_linear = [name for name in selected if not space.latent(name).linear]
    if not_linear:
        raise ParameterSpaceError(
            f"Latent(s) {not_linear} are not declared linear=True, so their linear "
            "operator is not meaningful. Declare them, and the claim will be checked — "
            "jointly, which is stricter than one at a time: a gain and an antenna "
            "temperature are each affine given the other and bilinear together."
        )
    return selected


def _values_at(
    space: ParameterSpace, values0: dict[str, jax.Array], at: dict[str, jax.Array] | None
) -> dict[str, jax.Array]:
    """``values0`` with ``at`` laid over it, refusing a name the space never declared."""
    if not at:
        return values0
    unknown = [key for key in at if key not in space.names]
    if unknown:
        raise ParameterSpaceError(
            f"`at` names {unknown}, which is not a latent of this space; declared: "
            f"{list(space.names)}."
        )
    return {**values0, **at}


def _isolate(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str,
    at: dict[str, jax.Array] | None = None,
) -> tuple[Callable[[jax.Array], jax.Array], jax.Array]:
    """``g(x) = prediction with latent `name` set to x``, plus a zero of its shape.

    ``at`` fixes the OTHER latents. A block is only linear *given* them, so a
    Gibbs sweep has to rebuild it wherever they currently are; without ``at``
    the block would silently keep describing the model at its declared starting
    point, which is right exactly once.
    """
    forward, values0 = space.forward_fn(pipeline, state_template)
    values0 = _values_at(space, values0, at)
    latent = space.latent(name)

    def g(x: jax.Array) -> jax.Array:
        return forward({**values0, name: x})

    return g, jnp.zeros_like(latent.init)


def _isolate_group(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    names: tuple[str, ...],
    at: dict[str, jax.Array] | None = None,
) -> tuple[Callable[[dict[str, jax.Array]], jax.Array], dict[str, jax.Array]]:
    """``g(x) = prediction with the whole group set to x``, plus a zero of it.

    The group's ``x`` is a ``{name: array}`` dict, so ``g`` is a function of a
    pytree and ``jax.linearize``/``jax.vjp`` hand back a JVP and a VJP over that
    pytree — which is exactly the domain the rest of this module solves in.

    ``at`` fixes the latents OUTSIDE the group, as for :func:`_isolate`. A value
    it supplies for a latent that IS in the group is overridden by ``x`` and so
    has no effect, which is the same for the group as for a single block: the
    map is affine, so where it is linearized does not change it.
    """
    forward, values0 = space.forward_fn(pipeline, state_template)
    values0 = _values_at(space, values0, at)

    def g(x: dict[str, jax.Array]) -> jax.Array:
        return forward({**values0, **x})

    return g, {name: jnp.zeros_like(space.latent(name).init) for name in names}


def _require_inexact(space: ParameterSpace, names: Sequence[str]) -> None:
    """Every latent in a linear block must carry a derivative worth taking."""
    for name in names:
        dtype = space.latent(name).init.dtype
        if not jnp.issubdtype(dtype, jnp.inexact):
            raise ParameterSpaceError(
                f"Latent {name!r} has dtype {dtype}; a linear block must be "
                "floating-point or complex."
            )


def _magnitude(latent: Any) -> float:
    """The latent's own scale, with the documented fallback for an all-zero init."""
    magnitude = float(jnp.max(jnp.abs(latent.init)))
    return magnitude if magnitude != 0.0 else 1.0


def _affinity_errors(
    g: Callable[[Any], jax.Array],
    zero: Any,
    probe_at: Callable[[int, float], Any],
    scales: Sequence[float],
    rtol: float | None,
) -> tuple[dict[float, float], list[float], float]:
    """Compare a map against its own linearization at zero, probe by probe.

    Shared verbatim by the single-latent and the grouped check, which differ
    only in what a probe *is* — an array, or a ``{name: array}`` dict. Every
    number below is computed from ``g``, ``zero`` and the probe alone, so the
    two paths cannot drift into measuring different things.
    """
    baseline, tangent = jax.linearize(g, zero)
    if rtol is None:
        rtol = 1e4 * float(jnp.finfo(baseline.dtype).eps)

    epsilon = float(jnp.finfo(baseline.dtype).eps)
    errors: dict[float, float] = {}
    verdicts: dict[float, bool] = {}
    for index, scale in enumerate(scales):
        probe = probe_at(index, scale)
        actual = g(probe)
        predicted = baseline + tangent(probe)
        # Measure against the VARIATION, not the total: a large constant offset
        # would otherwise hide a completely nonlinear response.
        variation = float(jnp.max(jnp.abs(actual - baseline)))
        departure = float(jnp.max(jnp.abs(actual - predicted)))
        errors[scale] = departure / max(variation, 1e-300)

        # A departure smaller than the arithmetic's OWN noise floor is not
        # evidence of curvature; without this the relative measure explodes at
        # small probes, where the variation is vanishing but roundoff is not,
        # and rejects perfectly linear blocks. The floor is set by the magnitudes
        # actually being differenced AT THIS PROBE — not by a constant, and not
        # by the baseline alone. A constant floor would silently exempt every
        # model whose prediction is small in its own units, and a baseline-only
        # floor would let an unrelated bright component disable the check.
        floor = 1e4 * epsilon * max(
            float(jnp.max(jnp.abs(actual))), float(jnp.max(jnp.abs(baseline)))
        )
        # NaN must count as a FAILURE, not a pass: `nan > rtol` is False, so a
        # naive comparison treats an unusable probe as evidence of linearity.
        finite = np.isfinite(errors[scale]) and np.isfinite(departure)
        verdicts[scale] = (not finite) or (errors[scale] > rtol and departure > floor)

    failed = sorted(scale for scale, bad in verdicts.items() if bad)
    return errors, failed, rtol


def check_linearity(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
    key: jax.Array | None = None,
) -> dict[float, float]:
    """Verify that the prediction really is affine in one latent — or in a group.

    Compares the model against its own linearization at zero, at several
    magnitudes of probe. Costs one linearization plus one forward evaluation
    per scale.

    Args:
        space, pipeline, state_template: the model under test.
        name: which latent. Optional when exactly one is declared linear.
        names: several latents, checked **jointly** — the claim a grouped
            :func:`linear_operator` block makes. Mutually exclusive with
            ``name``. This is strictly stronger than checking each in turn, and
            the difference is the whole reason a bilinear model needs more than
            one block: a gain and an antenna temperature are each affine given
            the other, and their product is not affine in the pair, so a group
            holding both is refused here rather than solved as if it were
            linear.
        at: values for the latents OUTSIDE the block. Linearity is a claim
            *given* them, so check it where the sampler will actually be.
            Defaults to the declared initial values.
        scales: probe magnitudes, as multiples of the latent's own scale,
            taken from ``max|init|`` — per latent, for a group, since two
            latents in one block are routinely in different units. The default
            spans six orders of magnitude on purpose — see the module
            docstring. NOTE: an all-zero ``init`` has no scale to take, so it
            falls back to 1.0 and the probes become absolute. If your latent
            lives at 1e6 (sky alms in kelvin, say), give a representative
            ``init`` or pass ``scales`` explicitly — otherwise the sweep never
            reaches the regime the sampler will actually explore.
        rtol: tolerance on the relative departure from affinity. Default:
            ``1e4 * eps`` of the prediction dtype, which leaves room for
            accumulated roundoff in a long reduction without admitting real
            curvature.
        key: PRNG key for the probes. Fixed by default, so the check is
            reproducible. For a group the per-latent sub-keys are folded in by
            position in the SORTED names, so permuting ``names`` probes the
            model at the same points and returns the same verdict.

    Returns:
        ``{scale: relative error}`` — useful for reporting how linear a block
        is, not only whether it passes.

    Raises:
        ParameterSpaceError: if ``name`` and ``names`` are both given, or if any
            scale departs from affinity by more than ``rtol``.
    """
    if name is not None and names is not None:
        raise ParameterSpaceError(f"check_linearity() {_BOTH_SPELLINGS}")
    key = jax.random.key(0) if key is None else key

    if names is None:
        name = _resolve_name(space, name)
        g, zero = _isolate(space, pipeline, state_template, name, at)
        _require_inexact(space, (name,))
        latent = space.latent(name)
        magnitude = _magnitude(latent)

        def probe_at(index: int, scale: float) -> jax.Array:
            return magnitude * scale * jax.random.normal(
                jax.random.fold_in(key, index), latent.init.shape, dtype=latent.init.dtype
            )

        subject = (
            f"Latent {name!r} is declared linear=True, but the prediction is not affine "
            "in it"
        )
        scale_of = "the latent's scale"
        remedy = (
            "Either drop the declaration, or re-parameterize so the model really is "
            "linear in this block."
        )
    else:
        selected = _resolve_names(space, names)
        g, zero = _isolate_group(space, pipeline, state_template, selected, at)
        _require_inexact(space, selected)
        ordered = sorted(selected)

        def probe_at(index: int, scale: float) -> dict[str, jax.Array]:
            root = jax.random.fold_in(key, index)
            return {
                member: _magnitude(space.latent(member)) * scale * jax.random.normal(
                    jax.random.fold_in(root, position),
                    space.latent(member).init.shape,
                    dtype=space.latent(member).init.dtype,
                )
                for position, member in enumerate(ordered)
            }

        subject = (
            f"Latents {list(selected)} are each declared linear=True, but the prediction "
            "is not affine in them JOINTLY"
        )
        scale_of = "each latent's own scale"
        remedy = (
            "Each conditional of a bilinear model is affine on its own, which is why "
            "this is not caught one latent at a time — and why these two cannot share "
            "one linear block. Split them into separate blocks and alternate, or "
            "re-parameterize so the joint map really is affine. "
            "identifiability(space, pipeline, state) will tell you what the split "
            "costs before you choose it."
        )

    errors, failed, rtol = _affinity_errors(g, zero, probe_at, scales, rtol)
    if failed:
        detail = ", ".join(f"{scale:g}x -> {err:.2e}" for scale, err in errors.items())
        raise ParameterSpaceError(
            f"{subject}: departure from its own linearization exceeds rtol={rtol:.2e} "
            f"(above the per-probe roundoff floor) at {failed} times {scale_of} "
            f"({detail}). {remedy}"
        )
    return errors


def linear_operator(
    space: ParameterSpace,
    pipeline: AbstractOperator,
    state_template: State,
    name: str | None = None,
    *,
    names: Sequence[str] | str | None = None,
    at: dict[str, jax.Array] | None = None,
    check: bool = True,
    scales: Sequence[float] = DEFAULT_SCALES,
    rtol: float | None = None,
) -> LinearBlock:
    """Export ``A``, ``Aᵀ`` and the offset for a declared-linear latent — or a group.

    No matrix is ever formed: ``A`` comes from ``jax.linearize`` and ``Aᵀ``
    from ``jax.vjp``, so a 10⁶-dimensional block costs the same as one forward
    evaluation per application. That is what makes conjugate-Gaussian solves
    tractable here — see :func:`wiener_solve`.

    Args:
        space, pipeline, state_template: the model.
        name: which latent. Optional when exactly one is declared linear. The
            block's ``x`` is then one array, and so is the solve's answer.
        names: several latents, exported as ONE block. Mutually exclusive with
            ``name``. The block's ``x`` is then a ``{name: array}`` dict — and
            so is the answer, which is the point: the physical names survive the
            solve instead of the caller slicing an anonymous stacked vector.
            ``names=("gain",)`` is a legitimate group of one, and is how a
            partition can hold one-latent and many-latent blocks without the
            caller special-casing either.

            Solving a group JOINTLY is not the same as alternating over its
            members: two latents the data barely tells apart are resolved in one
            CG here, where alternation converges at the rate of their
            correlation while reporting a converged residual and a κ of ~1 at
            every step. The joint κ that
            :func:`condition_estimate` reports for this block is the honest one.
        at: values for the latents OUTSIDE the block, fixing where it is built.
            Defaults to the declared initial values — right exactly once, so a
            Gibbs sweep must pass the current values here every sweep.
        check: verify the linearity claim first (:func:`check_linearity`).
            Leave it on. Turning it off costs three forward evaluations less
            and buys a class of silent, confident errors. For a group the claim
            checked is JOINT affinity, which a bilinear pair fails.
        scales, rtol: forwarded to :func:`check_linearity`.

    Raises:
        ParameterSpaceError: if both ``name`` and ``names`` are given; if
            ``names`` is empty, repeats a latent, or names an undeclared or
            non-linear one; or if the linearity claim fails.
    """
    if name is not None and names is not None:
        raise ParameterSpaceError(f"linear_operator() {_BOTH_SPELLINGS}")

    if names is None:
        name = _resolve_name(space, name)
        if check:
            check_linearity(space, pipeline, state_template, name, at=at,
                            scales=scales, rtol=rtol)
        g, zero = _isolate(space, pipeline, state_template, name, at)
        latent = space.latent(name)

        offset, tangent = jax.linearize(g, zero)
        _, pullback = jax.vjp(g, zero)

        return LinearBlock(
            name=name,
            shape=latent.init.shape,
            dtype=latent.init.dtype,
            offset=offset,
            forward=tangent,
            adjoint=lambda y: pullback(y)[0],
            prior=latent.prior,
        )

    selected = _resolve_names(space, names)
    if check:
        check_linearity(space, pipeline, state_template, names=selected, at=at,
                        scales=scales, rtol=rtol)
    g, zero = _isolate_group(space, pipeline, state_template, selected, at)

    offset, tangent = jax.linearize(g, zero)
    _, pullback = jax.vjp(g, zero)

    return LinearBlock(
        name=selected,
        shape={member: space.latent(member).init.shape for member in selected},
        dtype={member: space.latent(member).init.dtype for member in selected},
        offset=offset,
        forward=tangent,
        adjoint=lambda y: pullback(y)[0],
        prior={member: space.latent(member).prior for member in selected},
    )


def _real_parts(block: LinearBlock) -> tuple[Callable, Callable]:
    """Convert between a latent and its real degrees of freedom.

    A complex latent is carried as ``(real, imag)``. This is not bookkeeping
    pedantry: ``prediction`` is real, so the map from complex coefficients to
    data is **ℝ-linear but not ℂ-linear**, and a Krylov method run over ℂ
    would be solving a different problem. Splitting makes the vector space the
    one the objective actually lives on.

    For a group the split is per member, so a block mixing a real latent with a
    complex one carries ``{"real_one": array, "complex_one": (re, im)}`` — the
    real one is NOT wrapped in a one-element tuple, because there is nothing to
    unwrap it back from and a uniform wrapper would only move the asymmetry
    somewhere less visible. The treedef is what every ``jax.tree.map`` in this
    module aligns against, so it is the one thing that must be exactly
    invertible; ``join(split(x)) == x`` is what makes it so.
    """
    if not block.grouped:
        if _is_complex(block.dtype):
            return (
                lambda x: (jnp.real(x), jnp.imag(x)),
                lambda parts: parts[0] + 1j * parts[1],
            )
        return (lambda x: x, lambda parts: parts)

    complexity = {member: _is_complex(block.dtype[member]) for member in block.names}

    def split(x):
        return {
            member: (jnp.real(x[member]), jnp.imag(x[member]))
            if complexity[member]
            else x[member]
            for member in block.names
        }

    def join(parts):
        return {
            member: (parts[member][0] + 1j * parts[member][1])
            if complexity[member]
            else parts[member]
            for member in block.names
        }

    return split, join


def _domain_zero(block: LinearBlock) -> Any:
    """A zero of the latent domain — an array, or ``{name: array}`` for a group."""
    if not block.grouped:
        return jnp.zeros(block.shape, dtype=block.dtype)
    return {
        member: jnp.zeros(block.shape[member], dtype=block.dtype[member])
        for member in block.names
    }


def _domain_centre(block: LinearBlock, prior_mean: Any) -> Any:
    """``prior_mean`` laid out over the latent domain, zero where it is ``None``."""

    def one(shape, dtype, mean):
        if mean is None:
            return jnp.zeros(shape, dtype=dtype)
        return jnp.broadcast_to(jnp.asarray(mean, dtype=dtype), shape)

    if not block.grouped:
        return one(block.shape, block.dtype, prior_mean)
    return {
        member: one(block.shape[member], block.dtype[member], prior_mean[member])
        for member in block.names
    }


def _variance_parts(block: LinearBlock, prior_std: Any) -> Any:
    """``S⁻¹``'s diagonal, shaped like :func:`_real_parts`' output.

    This is the block-diagonal assembly, and it is assembly by *placement*
    rather than by concatenation: each latent's variance lands on the leaf its
    own parameters live on, so ``x / variance`` in a ``jax.tree.map`` IS
    ``S⁻¹x`` with no indices to get wrong. A complex latent's variance is
    duplicated across its real and imaginary parts, which is what
    :func:`gcr_sample` documents ``prior_std`` to mean for one.
    """

    def one(std, is_complex):
        variance = jnp.asarray(std) ** 2
        return (variance, variance) if is_complex else variance

    if not block.grouped:
        return one(prior_std, _is_complex(block.dtype))
    return {
        member: one(prior_std[member], _is_complex(block.dtype[member]))
        for member in block.names
    }


def _largest_variance(prior_variance: Any) -> jax.Array:
    """The biggest prior variance anywhere in the block.

    ``1/λ`` of it floors ``λ_min`` of the normal operator: ``AᵀN⁻¹A`` is
    positive semi-definite, so the LOOSEST prior in the block is what bounds
    the operator from below. Taking the tightest instead would floor the
    estimate above the true ``λ_min`` and report a condition number smaller
    than the real one — an over-confident guard, which is the direction that
    costs something.
    """
    leaves = jax.tree.leaves(prior_variance)
    return jnp.max(jnp.stack([jnp.max(jnp.asarray(leaf)) for leaf in leaves]))


def _numpyro_distributions() -> Any:
    """numpyro's distribution module, or ``None`` when it is not installed.

    Imported here rather than at module scope because numpyro is an optional
    extra and this module is usable without it — a prior-free linear block
    solves from keywords alone.
    """
    try:
        import numpyro.distributions as distributions
    except ImportError:  # pragma: no cover - numpyro is an optional extra
        return None
    return distributions


def _gaussian_parameters(prior: Any) -> tuple[Any, Any] | None:
    """``(loc, scale)`` if ``prior`` is a Gaussian **on the latent itself**.

    ``None`` otherwise — including for distributions that merely look like one.
    Identification is by TYPE, never by attribute, and that is the whole point:
    ``numpyro.distributions.LogNormal`` carries ``.loc`` and ``.scale`` and even
    a ``.base_dist`` that *is* a ``Normal``, while being a Gaussian in ``log x``
    and not in ``x``. Duck-typing on ``.loc``/``.scale`` would read those two
    numbers off it and return a finite, confident posterior for a
    parameterization nobody declared, which is exactly the failure this module
    exists to refuse.

    ``Independent`` and ``ExpandedDistribution`` are unwrapped because both
    only re-shape a base distribution; ``TransformedDistribution`` and the
    truncations are not, because both change what the distribution *is*.
    """
    distributions = _numpyro_distributions()
    if distributions is None:  # pragma: no cover - numpyro is an optional extra
        return None
    if isinstance(prior, (distributions.Independent, distributions.ExpandedDistribution)):
        return _gaussian_parameters(prior.base_dist)
    if isinstance(prior, distributions.Normal):
        return prior.loc, prior.scale
    return None


def _holds_a_tracer(value: Any) -> bool:
    """Whether ``value`` carries a tracer anywhere inside it.

    Asked of the pytree *leaves* rather than of the object itself, so a tracer
    wrapped in a list or a tuple is still recognised as one. The alternative —
    letting the comparison run and reading the failure — cannot tell a tracer
    apart from a shape mismatch, because ``TracerArrayConversionError`` is a
    ``TypeError``; an unanswerable comparison would then be reported as a
    settled *disagreement*, which is the one verdict this must never invent.
    """
    return any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves(value))


def _agrees(supplied: Any, declared: Any) -> bool | None:
    """Whether two prior parameters are the same number. ``None``: undecidable.

    Only a genuine tracer is undecidable. Two concrete numbers are the same two
    numbers whether or not some enclosing ``jit`` or ``lax.while_loop`` happens
    to be tracing — so the comparison is evaluated *here*, on the constants in
    hand, rather than staged into that trace. Staged, ``bool()`` raises on the
    result, a settled ``True`` comes back as unanswerable, and
    :func:`_reconcile` refuses a correct call while blaming a tracer that does
    not exist. That is not hypothetical:
    :func:`~rheplicant.inference.gls.iterative_gls` resolves the prior once and
    re-passes it into :func:`wiener_solve` from *inside* its reweighting loop,
    so this guard meets a live trace on every iteration of the one function it
    was written to serve.

    The comparison itself stays in ``jnp``, which canonicalizes both sides to
    the working precision. Comparing in NumPy instead would widen a declared
    ``float32`` scale to ``float64`` and call ``prior_std=0.05`` a
    contradiction of ``Normal(jnp.asarray(1.0), jnp.asarray(0.05))``, whose
    scale reads ``0.05000000074505806`` once widened — the same false refusal,
    moved rather than removed.
    """
    if _holds_a_tracer(supplied) or _holds_a_tracer(declared):
        return None
    try:
        with _RIGHT_NOW():
            return bool(jnp.all(jnp.asarray(supplied) == jnp.asarray(declared)))
    except jax.errors.ConcretizationTypeError:
        # Unreachable given the check above, and kept regardless: an
        # undecidable comparison has to reach the caller as undecidable, never
        # as a verdict.
        return None
    except (TypeError, ValueError):
        # Shapes that do not even broadcast are a disagreement, not a crash.
        return False


def _reconcile(
    keyword: str, field: str, supplied: Any, declared: Any, name: str, prior: Any, caller: str
) -> Any:
    """The supplied keyword, or the declared value — never a silent choice."""
    if supplied is None:
        return declared
    verdict = _agrees(supplied, declared)
    if verdict is None:
        side = (
            f"the {keyword}= you passed is"
            if _holds_a_tracer(supplied)
            else f"latent {name!r}'s declared {field} is"
        )
        raise ParameterSpaceError(
            f"{caller} cannot check the {keyword}= it was given against the prior latent "
            f"{name!r} declares: {side} a traced value, so what the two are cannot be "
            "known until the trace runs. Pass one or the other, not both: whichever lost "
            "would still look like it was in force. (Two CONCRETE values are compared "
            "normally, jit or no jit — being inside a trace is not itself the problem.)"
        )
    if not verdict:
        raise ParameterSpaceError(
            f"{caller} was given {keyword}={supplied!r}, but latent {name!r} declares "
            f"prior={type(prior).__name__}(..., {field}={declared!r}) in its "
            "ParameterSpace. One of the two would silently win and the other would be a "
            "number you believed was in force — and that same declaration reaches "
            "to_numpyro_model unchanged, so this exit and NUTS would then target different "
            f"posteriors from one space. Drop {keyword}= and let the declaration drive the "
            "solve, or change the declaration."
        )
    return supplied


def _resolve_one_prior(
    name: str, prior: Any, prior_mean: Any, prior_std: Any, caller: str
) -> tuple[Any, Any]:
    """Fill ``prior_mean``/``prior_std`` from one latent's declaration.

    A latent with no declared prior passes straight through — that is the escape
    hatch for a prior-free latent, which the optimizers use and which
    ``prior_std=`` alone is enough for.
    """
    if prior is None:
        return prior_mean, prior_std
    gaussian = _gaussian_parameters(prior)
    if gaussian is None:
        raise ParameterSpaceError(
            f"{caller} is a conjugate-Gaussian solve, but latent {name!r} declares a "
            f"{type(prior).__name__} prior, which has no conjugate Gaussian form. "
            "These exits solve (AᵀN⁻¹A + S⁻¹)x = b, and S⁻¹ only exists as a matrix for a "
            "Gaussian S; substituting the distribution's mean and variance would return a "
            "finite, confident posterior for a prior you did not declare — narrower than "
            "the truth wherever the declared prior is skewed or bounded. Sample this space "
            "with to_numpyro_model + NUTS instead, which honours the prior as written, or "
            "declare a numpyro Normal here and keep the conjugate exits."
        )
    loc, scale = gaussian
    return (
        _reconcile("prior_mean", "loc", prior_mean, loc, name, prior, caller),
        _reconcile("prior_std", "scale", prior_std, scale, name, prior, caller),
    )


def _group_priors(block: LinearBlock) -> dict[str, Any]:
    """The declared prior of each member of a group.

    A hand-assembled grouped block may carry ``prior=None``, meaning no member
    declares one; anything else has to be a dict covering every member, because
    a single distribution object standing for a whole group is a statement about
    latents in different units that nobody made, and dropping it silently would
    solve at whatever ``prior_std=`` happened to say.
    """
    if block.prior is None:
        return dict.fromkeys(block.names)
    if isinstance(block.prior, dict) and set(block.prior) == set(block.names):
        return block.prior
    raise ParameterSpaceError(
        f"This block groups {list(block.names)}, so its `prior` must be a dict with one "
        f"entry per member (use None for a prior-free one); it holds "
        f"{type(block.prior).__name__}"
        + (f" keyed by {sorted(block.prior)}" if isinstance(block.prior, dict) else "")
        + ". S is block-diagonal over the group and each member contributes its own "
        "block, so there is no reading under which one declaration covers all of them."
    )


def _per_member(keyword: str, value: Any, block: LinearBlock, caller: str) -> dict[str, Any]:
    """A grouped keyword, split by member. ``None`` everywhere when not given."""
    if value is None:
        return dict.fromkeys(block.names)
    if not isinstance(value, dict):
        raise ParameterSpaceError(
            f"{caller} was given {keyword}={value!r} for a block grouping "
            f"{list(block.names)}, but a grouped block has one prior PER LATENT — S is "
            "block-diagonal, not a multiple of the identity. These latents are routinely "
            "in different units (a noise-wave temperature in kelvin, a gain of order one), "
            "so one number spread across all of them is a prior nobody declared, and it "
            "would come back as a finite, confidently wrong posterior. Pass a dict keyed "
            f"by latent name — {keyword}={{{block.names[0]!r}: ...}} — or omit it and let "
            "each latent's own Latent(prior=...) drive the solve."
        )
    unknown = [key for key in value if key not in block.names]
    if unknown:
        raise ParameterSpaceError(
            f"{caller} was given {keyword} for {unknown}, which this block does not group; "
            f"it holds {list(block.names)}. The entry would be silently dropped and the "
            "latent it names solved at some other prior entirely."
        )
    return {member: value.get(member) for member in block.names}


def _resolve_prior(
    block: LinearBlock, prior_mean: Any, prior_std: Any, caller: str
) -> tuple[Any, Any]:
    """Fill ``prior_mean``/``prior_std`` from the block's declaration(s).

    For a group the resolution is per member and independent — each one takes
    its keyword if it was given, its declaration if it was not, and raises if
    the two disagree — so a group mixing a declared latent with a prior-free one
    is honoured rather than refused wholesale, and it is
    :func:`_require_prior_std` that names any member left with nothing at all.
    """
    if not block.grouped:
        return _resolve_one_prior(block.name, block.prior, prior_mean, prior_std, caller)

    priors = _group_priors(block)
    means = _per_member("prior_mean", prior_mean, block, caller)
    stds = _per_member("prior_std", prior_std, block, caller)
    resolved_mean: dict[str, Any] = {}
    resolved_std: dict[str, Any] = {}
    for member in block.names:
        resolved_mean[member], resolved_std[member] = _resolve_one_prior(
            member, priors[member], means[member], stds[member], caller
        )
    return resolved_mean, resolved_std


def _require_prior_std(block: LinearBlock, prior_std: Any, caller: str) -> None:
    """No prior at all leaves AᵀN⁻¹A free to be singular."""
    if block.grouped:
        missing = [member for member in block.names if prior_std[member] is None]
        if not missing:
            return
        detail = (
            f"needs a prior_std for {missing} — the other members of this block have one, "
            "which does not help: "
        )
    else:
        if prior_std is not None:
            return
        detail = "needs prior_std: "
    raise ParameterSpaceError(
        f"{caller} {detail}with no prior the normal operator AᵀN⁻¹A can be "
        "singular, and CG would return a finite, arbitrary answer rather than fail. "
        "Pass a large prior_std for an effectively flat prior, or declare "
        "Latent(prior=dist.Normal(...)) and it will be read from there."
    )


def _refuse_a_noise_model_at_the_conjugate_seam(noise_std: Any, caller: str) -> None:
    """Say why a :class:`NoiseModel` does not belong here, instead of TypeError.

    ``check_noise_std_axis`` accepts a noise model -- it has to, since every
    other exit in the package passes one. This module does not: the conjugate
    solves take ``1 / sigma**2`` from a plain array. Without this refusal a
    model reaches ``jnp.asarray`` and comes back as
    ``TypeError: Value 'HomoscedasticNoise(sigma=weak_f32[])' with dtype
    object is not a valid JAX array type``, which names the wrong layer and
    reads like a bug in the package rather than a wrong argument.

    A prediction-dependent model gets the longer sentence because it is not a
    packaging problem: a conjugate solve has no prediction to evaluate one at,
    the prediction being what it solves for. Freezing sigma at some parameter
    tuple is a real choice with a statistical consequence -- see
    :mod:`rheplicant.inference.plan` -- and it belongs to the caller who knows
    which tuple, never to a silent unwrap here.
    """
    if not isinstance(noise_std, NoiseModel):
        return
    if getattr(noise_std, "depends_on_prediction", False):
        raise ParameterSpaceError(
            f"{caller} was given {type(noise_std).__name__}, whose sigma depends on "
            "the prediction — but a conjugate solve has no prediction to evaluate it "
            "at, because the prediction is what it solves for. Freeze it yourself at "
            "the parameter tuple you mean (`noise.std(prediction)`) and pass that "
            "array, which also makes explicit that the result is an exact draw at "
            "THAT covariance and not from the full model's conditional. A "
            "SamplingPlan does this per sweep."
        )
    raise ParameterSpaceError(
        f"{caller} takes a plain sigma array, not a {type(noise_std).__name__}. "
        "The conjugate solves compute 1/sigma**2 directly; pass `noise.std(...)`, "
        "or the sigma you built the model from."
    )


def _check_solve_arguments(
    block: LinearBlock,
    observed: jax.Array,
    prior_mean: Any,
    prior_std: Any,
    caller: str,
    *,
    noise_std: Any = None,
) -> tuple[Any, Any]:
    """Shared preconditions for the mean and the draw, plus the resolved prior.

    Returns the ``(prior_mean, prior_std)`` the solve should actually use: the
    keywords when they were given, the latent's declaration when they were not,
    and an exception when the two disagree.

    ``noise_std`` is checked here for the axis contract
    (:func:`~rheplicant.inference.noise.check_noise_std_axis`). Both exits pass
    it — :func:`wiener_solve` and :func:`gcr_sample` — and
    ``tests/inference/test_noise_std_axis.py`` asserts they refuse the same
    inputs, because a rule enforced on the mean and not on the draw is worse
    than no rule: it teaches that the argument is checked.

    **Two homes, and it stays that way.** Every other exit reaches that rule
    through :func:`~rheplicant.inference.uncertainty.as_noise_model`, the one
    place a ``noise_std`` argument is normalized. This module does not call it:
    :func:`_conjugate_solve` and :func:`condition_estimate` take the bare array
    to ``1 / sigma**2`` directly. Routing them through ``as_noise_model`` would
    leave one home, and it was assessed and rejected for two measured reasons.

    **The weight formulas disagree on NaN, in the dangerous direction.**
    ``1 / sigma**2`` and
    :func:`~rheplicant.inference.noise.inverse_variance` agree on every finite
    sigma, on ``inf`` (both give exactly ``0``), on ``0`` and on a negative
    sigma. They differ on ``nan``: this module propagates it, so the solution
    comes back NaN and the caller knows; ``inverse_variance`` maps it to weight
    ``0.0``, which *means* "unobserved". Switching would turn "your sigma array
    has a NaN in it" from a loud failure into a silently dropped sample — and
    in a conjugate solve a dropped sample moves the posterior WIDTH, not only
    the point, with nothing reporting how many went.

    **A conjugate solve has no prediction to give it.**
    ``inverse_variance(noise, prediction)`` needs one, and for a
    ``depends_on_prediction`` model it genuinely matters: a
    :class:`~rheplicant.inference.noise.RadiometerNoise`'s weights move 9x
    between a 100 K and a 300 K prediction. But the prediction is what the
    solve is *for*. So the solves would have to freeze ``N`` at some arbitrary
    point, and :mod:`rheplicant.inference.plan` already documents what freezing
    costs: an exact draw from a linear-Gaussian conditional *at that
    covariance*, which is not the full model's conditional. Accepting a
    ``NoiseModel`` at this seam would invite precisely that mistake by making
    it type-check.

    The duplication is therefore deliberate rather than owed.
    ``tests/inference/test_noise_std_axis.py::TestWhyTheRuleHasTwoHomes`` pins
    both measurements, so an author who unifies them anyway meets the two
    consequences rather than rediscovering them. The keyword stays optional so
    an internal caller that has already normalized need not pay for it twice.
    """
    check_observed_shape(jnp.shape(block.offset), observed)
    if noise_std is not None:
        check_noise_std_axis(noise_std, jnp.shape(block.offset), caller)
        _refuse_a_noise_model_at_the_conjugate_seam(noise_std, caller)
    prior_mean, prior_std = _resolve_prior(block, prior_mean, prior_std, caller)
    _require_prior_std(block, prior_std, caller)
    if jnp.issubdtype(jnp.asarray(block.offset).dtype, jnp.complexfloating):
        raise ParameterSpaceError(
            f"{caller} expects a real-valued prediction; this block's offset is complex."
        )
    return prior_mean, prior_std


def wiener_solve(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any = None,
    prior_mean: Any = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = 1e-3,
) -> tuple[Any, jax.Array]:
    """Posterior mean of a linear-Gaussian block — the Wiener filter, by CG.

    With ``d = A x + offset + n``, ``n ~ N(0, N)`` and ``x ~ N(m, S)``::

        x̂ = (AᵀN⁻¹A + S⁻¹)⁻¹ [AᵀN⁻¹ (d - offset) + S⁻¹m]

    solved with conjugate gradients, so the normal operator is only ever
    *applied*, never formed. Each iteration costs one JVP and one VJP through
    the forward model — which is why a block with 10⁶ degrees of freedom is
    tractable at all.

    The normal operator and the right-hand side are both obtained as gradients
    of the objective itself rather than assembled from ``A`` and ``Aᵀ`` by
    hand. That is not a shortcut: it makes the operator symmetric positive
    definite *by construction* over the real degrees of freedom, with no
    adjoint-convention arithmetic left to get wrong for complex latents.

    This is the posterior **mean**, not a sample. For a draw, see
    :func:`gcr_sample`, which adds a fluctuation term to this same right-hand
    side and costs exactly the same solve.

    Args:
        block: from :func:`linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise_std: noise standard deviation — a scalar, or an array whose
            SHAPE says which axis of the data it runs along: ``(n_time, 1)``
            for a per-time sigma, ``(1, n_freq)`` for a per-channel one. A bare
            1-D vector is accepted only where its length matches a single axis
            of the data; on a square grid it matches two, both readings are
            legitimate, and the one broadcasting picks is not the one most
            callers mean — so that case raises rather than being resolved by
            trailing-axis alignment. See
            :func:`~rheplicant.inference.noise.check_noise_std_axis`.

            **An array, never a** :class:`~rheplicant.inference.noise.NoiseModel`,
            and the keyword name is the signal rather than an accident of
            history. ``noise_std=`` is a sigma that has already been decided;
            ``noise=`` — on :func:`~rheplicant.inference.gls.iterative_gls` and
            on :class:`~rheplicant.inference.plan.SamplingPlan` — is the *rule*
            that decides one. A conjugate solve has no prediction to evaluate a
            rule at, the prediction being what it solves for, so a model is
            refused here by name rather than quietly frozen at some arbitrary
            point: see ``_refuse_a_noise_model_at_the_conjugate_seam`` for
            the message and ``_check_solve_arguments`` for the two measured
            reasons this seam is not routed through ``as_noise_model``. Freeze
            it yourself — ``noise.std(prediction)`` — and pass that array.
        prior_std: prior standard deviation on the latent — scalar or
            broadcastable to it. **Defaults to the latent's declared prior**;
            required only when there is none, because without a prior the
            normal operator can be singular and CG would return a finite,
            arbitrary answer instead of complaining. Passing a value that
            contradicts the declaration raises rather than one silently
            winning — see the note below.
        prior_mean: centre of the prior. Defaults to the declared prior's
            location, and to zero when nothing is declared — which is wrong for
            most physical quantities, a noise-wave temperature sitting near
            250 K. Equivalent to an affine binding that adds the same offset,
            but says what it means.
        tol: CG tolerance — a bound on the relative RESIDUAL, which is not the
            same as accuracy. See the note on conditioning below.
        maxiter: CG iteration cap. ``None`` lets JAX choose.
        require_convergence: raise unless the relative ERROR can be bounded by
            this. Defaults to ``1e-3``; ``None`` disables the guard and returns
            whatever CG produced. On by default because jax's ``cg`` reports no
            convergence status, so an unconverged solve otherwise comes back
            looking exactly like a converged one.

            The bound is ``κ · relative_residual``, with ``κ`` estimated by
            :func:`condition_estimate`. Guarding on the residual alone would
            certify nothing in the regime that matters — see below — so this
            costs ``2 · POWER_ITERATIONS`` extra operator applications. That is
            not free: on a well-conditioned block, where CG itself converges in
            a few iterations, it roughly DOUBLES the solve. In a Gibbs sweep,
            where the conditioning barely moves from sweep to sweep, call
            :func:`condition_estimate` once outside the loop, choose ``tol``
            from it, and pass ``require_convergence=None`` inside — the same
            bargain :func:`linear_operator`'s ``check`` offers.

    Returns:
        ``(x̂, relative_residual)``, the residual being ``‖M x̂ - b‖ / ‖b‖``
        over the real degrees of freedom. Note that this is the residual, not
        the error; multiply by :func:`condition_estimate` for the error bound.

        ``x̂`` is the block's own domain, so its shape follows the spelling that
        built the block: a ``{name: array}`` dict for ``names=``, and a **bare
        array** for ``name=``. The bare form is not what anything downstream
        reads — ``space.forward_fn``'s ``forward``,
        :meth:`~rheplicant.inference.parameters.ParameterSpace.bind`,
        :func:`~rheplicant.inference.uncertainty.fisher_information`,
        :func:`~rheplicant.inference.identifiability.identifiability`'s ``at=``,
        :func:`linear_operator`'s own ``at=`` and
        :func:`~rheplicant.inference.engines.conditional_potential` all index by
        latent name and all six raise on it. Wrap it as ``{block.name: x̂}``
        first; :meth:`LinearBlock.as_dict` is that call, and does nothing to the
        grouped form, so it is correct either way.

    Note:
        **Conditioning, and why ``tol`` is not accuracy.** Residual and error
        differ by the condition number of ``M = AᵀN⁻¹A + S⁻¹``::

            ‖x̂ - x*‖ / ‖x*‖  ≤  κ(M) · ‖M x̂ - b‖ / ‖b‖

        For a block the data does not fully identify — one calibration load
        against three unknowns, a flagged channel, a short integration — the
        prior is the only thing holding the blind directions down, so
        ``λ_min(M)`` is exactly ``1/prior_std²`` and ``κ ≈ ‖AᵀN⁻¹A‖ · prior_std²``
        runs to 1e6 and beyond. At κ=1e7 the default ``tol=1e-6`` bounds the
        relative error by 10: no digits at all. CG stops on a residual that
        looks converged, having left the prior-dominated directions at their
        starting value, and the draw comes back with far too little scatter.

        This is exactly the regime these solvers exist for, so the guard is on
        by default and the accuracy target is stated as an error, not a
        residual. To solve rather than refuse, pass ``tol ≈
        require_convergence / κ`` with a ``maxiter`` to match. Past ``κ · eps``
        no tolerance helps and only precision does; the guard says so in its
        own words.

    Note:
        **Where S comes from.** ``Latent(prior=dist.Normal(m, s))`` is the
        package's one statement of what a latent is a priori, and it is the
        statement ``to_numpyro_model`` reads. So it is the statement this solve
        reads too: declare it once and both exits target the same posterior.
        The keywords remain, for a prior-free latent and for overriding a
        declaration you are deliberately solving away from — but a keyword that
        *contradicts* a declaration raises, because the alternative is one of
        the two silently winning and the two exits quietly disagreeing. A
        declared prior with no conjugate Gaussian form (a Half-Normal, a
        Uniform) raises here as well; NUTS is where that space belongs.
    """
    prior_mean, prior_std = _check_solve_arguments(
        block, observed, prior_mean, prior_std, "wiener_solve", noise_std=noise_std
    )
    return _conjugate_solve(
        block, observed, noise_std=noise_std, prior_std=prior_std,
        prior_mean=prior_mean, tol=tol, maxiter=maxiter, key=None,
        require_convergence=require_convergence,
    )


def _normal_operator(block: LinearBlock, weight, prior_variance) -> Callable:
    """``x -> (AᵀN⁻¹A + S⁻¹) x`` over the block's real degrees of freedom.

    The curvature half is taken as a gradient rather than assembled from
    ``A`` and ``Aᵀ``, which makes it symmetric positive definite by
    construction with no adjoint convention left to get wrong — and, for a
    group, no cross-block bookkeeping either: ``jax.grad`` of the group's own
    ``χ²`` produces the full operator, off-diagonal blocks included, which is
    exactly the coupling an alternating solve throws away.

    ``prior_variance`` is a pytree of the same shape as ``parts``, so ``S⁻¹``
    enters leaf by leaf. See :func:`_variance_parts`.
    """
    split, join = _real_parts(block)

    def half_chi2(parts):
        return 0.5 * jnp.sum(weight * block.forward(join(parts)) ** 2)

    def normal(parts):
        curvature = jax.grad(half_chi2)(parts)
        return jax.tree.map(
            lambda c, p, v: c + p / v, curvature, parts, prior_variance
        )

    return normal


def _condition_number(
    block: LinearBlock, weight, prior_variance, key, iterations: int
) -> jax.Array:
    """Estimated ``κ`` of ``AᵀN⁻¹A + S⁻¹``.

    For a group this is the JOINT condition number, and it is the number a
    per-block guard cannot produce: two latents the data barely distinguishes
    give a well-conditioned operator each and a badly conditioned one together.
    """
    split, _ = _real_parts(block)
    template = split(_domain_zero(block))
    largest, smallest = extreme_eigenvalues(
        _normal_operator(block, weight, prior_variance), template, key, iterations
    )
    # AᵀN⁻¹A is positive semi-definite, so λ_min can never fall below the
    # prior's own curvature however rank-deficient the data is.
    floor = 1.0 / _largest_variance(prior_variance)
    return largest / jnp.maximum(smallest, floor)


def condition_estimate(
    block: LinearBlock,
    *,
    noise_std: Any,
    prior_std: Any = None,
    iterations: int = POWER_ITERATIONS,
    key: jax.Array | None = None,
) -> jax.Array:
    """Condition number of the normal operator this block would be solved with.

    ``κ(AᵀN⁻¹A + S⁻¹)`` is the number that says how much a solver's residual
    understates its error: for a solution ``x`` with relative residual ``r``,

        ‖x - x*‖ / ‖x*‖  ≤  κ · r

    so a residual of 1e-6 against κ=1e7 certifies nothing at all. Use it to
    pick ``tol`` for :func:`wiener_solve` and :func:`gcr_sample`: for a target
    relative accuracy ``a``, ask for roughly ``tol = a / κ``.

    Large κ is not a defect here, it is the design: for a block the data does
    not fully identify, ``λ_min`` is exactly ``1/prior_std²`` while ``λ_max``
    is set by the data, so κ grows with how much better the data constrains
    one direction than the prior constrains another.

    Costs ``2 · iterations`` applications of the normal operator — each the
    same JVP-plus-VJP a CG iteration costs — and no matrix is ever formed.

    Args:
        block: from :func:`linear_operator`.
        noise_std: the same decided sigma array those solves take, and a
            :class:`~rheplicant.inference.noise.NoiseModel` is as wrong here as
            it is there — a κ is the conditioning of one particular normal
            operator, so it needs the covariance settled, not a rule for
            producing one. Note that this exit does **not** run
            ``_check_solve_arguments``, so neither the seam refusal nor the
            1-D axis check fires: a model reaches ``jnp.asarray`` and comes back
            as ``TypeError: Value 'HomoscedasticNoise(...)' with dtype object is
            not a valid JAX array type``, which names the wrong layer. Pass
            ``noise.std(prediction)``.
        prior_std: as for :func:`wiener_solve` — it defaults to the latent's
            declared prior, so the κ reported here is the κ of the system those
            solves will build rather than of a system nobody solves.
        iterations: power-iteration steps per end of the spectrum. The default
            is comfortable; the estimate typically settles within three.
        key: PRNG key for the starting vectors. Fixed by default, so the
            estimate is reproducible.

    Returns:
        The estimated condition number, as a scalar array.
    """
    _, prior_std = _resolve_prior(block, None, prior_std, "condition_estimate")
    _require_prior_std(block, prior_std, "condition_estimate")
    return _condition_number(
        block,
        1.0 / jnp.asarray(noise_std) ** 2,
        _variance_parts(block, prior_std),
        jax.random.key(0) if key is None else key,
        iterations,
    )


def _conjugate_solve(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any,
    prior_mean: Any,
    tol: float,
    maxiter: int | None,
    key: jax.Array | None,
    require_convergence: float | None,
) -> tuple[Any, jax.Array]:
    """Shared machinery for the posterior mean and for a posterior draw.

    Both solve ``(AᵀN⁻¹A + S⁻¹) x = b`` by CG over the latent's real degrees of
    freedom. They differ only in ``b``: the mean uses ``AᵀN⁻¹(d - offset)``,
    a draw adds the two fluctuation terms. ``key=None`` selects the mean.
    """
    split, join = _real_parts(block)
    weight = 1.0 / jnp.asarray(noise_std) ** 2
    prior_variance = _variance_parts(block, prior_std)
    residual_data = observed - block.offset
    zero = split(_domain_zero(block))
    centre = split(_domain_centre(block, prior_mean))

    def pair_with(vector):
        """``Aᵀ vector`` in real coordinates, as the gradient of a real pairing.

        Taking it as a gradient rather than calling ``block.adjoint`` is what
        keeps the real/complex conventions from ever entering: ``jax.grad`` of a
        real scalar is by construction the adjoint of the real inner product,
        which is the pairing every term here lives in.
        """
        return jax.grad(lambda parts: jnp.sum(block.forward(join(parts)) * vector))(zero)

    normal = _normal_operator(block, weight, prior_variance)

    # S^-1 m: a zero-mean prior is wrong for most physical quantities (a
    # noise-wave temperature sits near 250 K, not near zero), and shifting the
    # prior is not the same act as shifting the model even though the two give
    # the same Gaussian.
    rhs = jax.tree.map(
        lambda base, m, v: base + m / v,
        pair_with(weight * residual_data),
        centre,
        prior_variance,
    )

    if key is not None:
        # Constrained realization: the two fluctuation terms whose covariances
        # sum to the normal operator itself, which is exactly why the solve
        # comes out distributed as the posterior rather than merely centred on
        # its mean.  b = AᵀN⁻¹(d-offset) + AᵀN⁻¹ᐟ²ω₁ + S⁻¹ᐟ²ω₂
        data_key, prior_key = jax.random.split(key)
        omega_data = jax.random.normal(
            data_key, jnp.shape(residual_data), dtype=jnp.result_type(residual_data)
        )
        omega_prior = jax.tree.map(
            lambda leaf, k: jax.random.normal(k, leaf.shape, dtype=leaf.dtype),
            zero,
            _split_like(prior_key, zero),
        )
        rhs = jax.tree.map(
            lambda base, from_data, from_prior, v: (
                base + from_data + from_prior / jnp.sqrt(v)
            ),
            rhs,
            pair_with(jnp.sqrt(weight) * omega_data),
            omega_prior,
            prior_variance,
        )

    solution, _ = jax.scipy.sparse.linalg.cg(normal, rhs, tol=tol, maxiter=maxiter)
    misfit = jax.tree.map(lambda a, b: a - b, normal(solution), rhs)
    residual = tree_norm(misfit) / jnp.maximum(tree_norm(rhs), 1e-30)
    if require_convergence is not None:
        # jax's cg reports no convergence status of its own, so an unconverged
        # solve otherwise comes back looking like any other answer. eqx.error_if
        # fires under jit, where a Python `if` on a traced value cannot.
        #
        # The residual ALONE cannot decide this. Error and residual differ by
        # the condition number, and for a block the data does not fully
        # identify κ is enormous by construction — λ_min is exactly the prior's
        # 1/prior_std² — so CG stops on a tiny residual with the prior-dominated
        # directions still at their starting value, and hands back a draw whose
        # posterior scatter there is orders of magnitude too small. Guarding on
        # the residual certifies precisely nothing in the one regime these
        # solvers exist to serve.
        kappa = _condition_number(
            block, weight, prior_variance, jax.random.key(0), POWER_ITERATIONS
        )
        error_bound = residual * kappa
        bad = jnp.logical_or(~jnp.isfinite(residual), error_bound > require_convergence)

        # Below κ·eps no tolerance can help: the arithmetic itself cannot
        # represent the answer that accurately. Worth its own message, because
        # the remedy is precision, and the natural response to the other
        # message — tighten tol, raise maxiter — burns a great many iterations
        # here to arrive at an equally wrong answer.
        epsilon = float(jnp.finfo(jnp.asarray(block.offset).dtype).eps)
        unreachable = kappa * epsilon > require_convergence

        solution = eqx.error_if(
            solution,
            jnp.logical_and(bad, unreachable),
            "wiener_solve/gcr_sample cannot reach require_convergence at this "
            "precision: the normal operator's condition number times the machine "
            "epsilon already exceeds it, so no tol or maxiter will help. This is "
            "the usual signature of a block the data does not identify. Enable "
            "jax_enable_x64, or strengthen the prior (prior_std bounds the "
            "conditioning: κ ≈ ‖AᵀN⁻¹A‖·prior_std²). condition_estimate() reports "
            "the number.",
        )
        solution = eqx.error_if(
            solution,
            jnp.logical_and(bad, ~unreachable),
            "wiener_solve/gcr_sample did not converge: the relative residual times "
            "the normal operator's condition number — the bound on the RELATIVE "
            "ERROR, which is what require_convergence limits — exceeds it. The "
            "residual alone looks converged; it is not, along the directions the "
            "prior dominates. Pass tol ≈ require_convergence/κ with a maxiter to "
            "match, or strengthen the prior. condition_estimate() reports κ.",
        )
    return join(solution), residual


def _split_like(key: jax.Array, template) -> Any:
    """One independent key per leaf of ``template``, same structure."""
    leaves, treedef = jax.tree.flatten(template)
    return jax.tree.unflatten(treedef, list(jax.random.split(key, len(leaves))))


def gcr_sample(
    block: LinearBlock,
    observed: jax.Array,
    *,
    noise_std: Any,
    prior_std: Any = None,
    key: jax.Array,
    prior_mean: Any = None,
    tol: float = 1e-6,
    maxiter: int | None = None,
    require_convergence: float | None = 1e-3,
) -> tuple[Any, jax.Array]:
    """Draw an EXACT posterior sample of a linear-Gaussian block.

    The constrained-realization (GCR) identity: solve the same system
    :func:`wiener_solve` does, but with two white-noise terms added to the
    right-hand side::

        (AᵀN⁻¹A + S⁻¹) x = AᵀN⁻¹(d - offset) + S⁻¹m + AᵀN⁻¹ᐟ² ω₁ + S⁻¹ᐟ² ω₂

    with ``ω₁``, ``ω₂`` standard normal on the data and on the latent. The
    right-hand side then has the posterior-mean numerator as its mean and
    covariance ``AᵀN⁻¹A + S⁻¹`` — the operator itself — so ``x = M⁻¹b`` has the posterior
    mean and covariance ``M⁻¹M M⁻¹ = M⁻¹`` exactly. Not an approximation and
    not a Markov chain: every call is an independent draw, with no burn-in and
    nothing to diagnose for convergence.

    This is what makes a 10⁶-dimensional block samplable at all. It costs one
    CG solve — the same as the mean — because the fluctuation enters the
    right-hand side, never the operator.

    In a Gibbs scheme, this draws the linear block conditional on the nonlinear
    parameters; rebuild the block with
    :func:`linear_operator(..., check=False)` each sweep, having checked the
    linearity claim once outside the loop. The conditioning guard is worth
    hoisting the same way: :func:`condition_estimate` once to fix ``tol``, then
    ``require_convergence=None`` in the loop. What you must NOT do is leave
    ``tol`` at its default and the guard off — that is the combination this
    module returned a silently over-confident posterior for.

    Args:
        block: from :func:`linear_operator`.
        observed: the data, shaped like ``block.offset``.
        noise_std: noise standard deviation, exactly as for
            :func:`wiener_solve` — the same axis contract on a 1-D sigma
            (both exits share ``_check_solve_arguments``, so a shape one refuses
            the other refuses), and the same refusal of a
            :class:`~rheplicant.inference.noise.NoiseModel` at this seam. The
            keyword is the signal: ``noise_std=`` takes a decided sigma,
            ``noise=`` takes the rule that decides one, and a draw has no
            prediction to evaluate a rule at any more than the mean does.
        prior_std: prior standard deviation on the latent. Defaults to the
            latent's declared prior, as for :func:`wiener_solve`, and required
            only when there is none. For a complex latent this is the width of
            the real and imaginary parts independently.
        key: PRNG key. ``vmap`` over split keys for many independent draws.
        prior_mean: centre of the prior; defaults to the declared prior's
            location, and to zero when nothing is declared. With uninformative
            data the draws fall back to ``N(prior_mean, prior_std²)``, which is
            the check that it is wired in correctly.
        tol: CG tolerance — a bound on the residual, not on the accuracy.
        maxiter: CG iteration cap.
        require_convergence: as for :func:`wiener_solve`, including the
            conditioning note there, which a draw is MORE exposed to than the
            mean. The fluctuation term ``S⁻¹ᐟ²ω₂`` puts weight on every
            direction of the latent by construction, including the ones the
            data is blind to — so a draw always has something to resolve where
            the operator is worst conditioned, whereas the mean does only when
            ``prior_mean`` is nonzero.

    Returns:
        ``(x, relative_residual)``. An unconverged CG returns a draw from the
        WRONG distribution — and a distribution that is too NARROW, since the
        directions left unresolved are the prior-dominated ones that should
        have carried the most scatter — so ``require_convergence`` is on by
        default here too.

        ``x`` carries the block's domain, dict or bare array, exactly as
        :func:`wiener_solve`'s does; see the note there, and
        :meth:`LinearBlock.as_dict` for the wrap.

    Note:
        ``S`` is read off ``Latent(prior=...)`` when the keywords are omitted;
        see the corresponding note on :func:`wiener_solve` for what that does
        and does not permit. It matters more here than for the mean: with a
        declared prior ignored, the fluctuation term ``S⁻¹ᐟ²ω₂`` is drawn at
        the wrong width, so every draw is wrong in the one direction the mean
        can be right in.
    """
    prior_mean, prior_std = _check_solve_arguments(
        block, observed, prior_mean, prior_std, "gcr_sample", noise_std=noise_std
    )
    return _conjugate_solve(
        block, observed, noise_std=noise_std, prior_std=prior_std,
        prior_mean=prior_mean, tol=tol, maxiter=maxiter, key=key,
        require_convergence=require_convergence,
    )
