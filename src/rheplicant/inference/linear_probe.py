"""Probing an operator to see whether the linearity claim holds.

Resolving parameter names to values, isolating one parameter or a group,
evaluating at the probe points and reporting the affinity errors. This is the
measurement; `check_linearity` beside it is the verdict.
"""

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from bayesmith.exact.linearity import DEFAULT_AT_POINTS as _DEFAULT_AT_POINTS
from bayesmith.exact.linearity import (
    RELATIVE_FLOOR_FACTOR,
    WEIGHTED_FLOOR_FACTOR,
    WEIGHTED_RTOL,
    Unresolved,
)

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State
from rheplicant.inference.parameters import ParameterSpace

from .linear_priors import (
    _gaussian_parameters,
    _holds_a_tracer,
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


def _resolve_names(
    space: ParameterSpace,
    names: Sequence[str] | str,
    *,
    require_linear: bool = True,
) -> tuple[str, ...]:
    """The latents to put in one block, in the caller's own order.

    A bare string is one name, the same "one or many" convention
    :class:`~rheplicant.inference.parameters.Bind` and
    :func:`~rheplicant.inference.identifiability.identifiability` use. Without
    it, ``names="gain"`` iterates into ``('g', 'a', 'i', 'n')``.

    ``require_linear=False`` keeps every structural check — non-empty, declared,
    no repeats — and drops only the ``linear=True`` requirement, for
    :mod:`rheplicant.inference.loglinear`, whose blocks are precisely the ones
    that must NOT carry that declaration. The checks are shared rather than
    copied so that a change to what a block may be made of reaches both.
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
    if require_linear:
        not_linear = [name for name in selected if not space.latent(name).linear]
        if not_linear:
            raise ParameterSpaceError(
                f"Latent(s) {not_linear} are not declared linear=True, so their linear "
                "operator is not meaningful. Declare them, and the claim will be checked "
                "— jointly, which is stricter than one at a time: a gain and an antenna "
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
    """The latent's own scale, with the documented fallback for an all-zero init.

    ``np`` and not ``jnp``, and the difference is not stylistic. Inside a trace
    a ``jnp`` call is staged into the jaxpr *even when its input is concrete* --
    JAX does not constant-fold at trace time -- so ``jnp.max`` hands back a
    tracer and the ``float`` below raises ``ConcretizationTypeError``. ``np``
    runs eagerly on the array and returns a Python float. Which is the honest
    spelling anyway: ``latent.init`` is a declaration, fixed when the program is
    built, and this is a step size chosen at build time rather than a quantity
    that varies with the data.

    (Closed-over arrays are *not* traced by ``eqx.filter_jit`` -- verified --
    so it really is the ``jnp`` call that introduces the tracer, not the
    closure.)
    """
    magnitude = float(np.max(np.abs(latent.init)))
    return magnitude if magnitude != 0.0 else 1.0


#: How many values of the OUTSIDE latents a check looks at (D16 axis 2, ruled
#: 2026-08-27). Imported from the sibling package rather than spelled again:
#: one statement of the number, both sides reading it.
DEFAULT_AT_POINTS: int = _DEFAULT_AT_POINTS


def _prior_at_points(
    space: ParameterSpace,
    selected: Sequence[str],
    at: dict[str, jax.Array],
    count: int,
    key: jax.Array,
) -> list[dict[str, jax.Array]]:
    """``at``, plus ``count - 1`` draws from the OUTSIDE latents' own priors.

    D16 axis 2. Checking at one point is how a check becomes a
    moderate-parameter probe -- the failure mode ``boundary-validation.md``
    exists to prevent -- and "affine GIVEN the outside latents" is a claim
    about the values they will actually take, not about the one they were
    declared at. Measured before the change: a model affine in ``u`` exactly
    when ``w`` sits at its declared init reported **0.0 at every scale** and
    was accepted; one step away it refuses.

    A latent with no prior to draw from keeps its ``at`` value in every point,
    and says nothing: a free parameter has no distribution to sample, and
    inventing a spread for it would probe somewhere the declaration never
    claimed anything about.
    """
    chosen = set(selected)
    drawable = [
        (n, _gaussian_parameters(space.latent(n).prior))
        for n in space.names
        # An INTEGER latent is not drawable either, and the check for it is
        # separate from "has no prior" on purpose: `_require_inexact` refuses
        # an integer dtype among the SELECTED latents, but an outside one is
        # perfectly legal (a channel count, a mode index) and simply has no
        # Gaussian draw. Measured: `jax.random.normal` refuses an int32 dtype
        # outright, so without this an entirely valid document dies here with
        # a message about dtypes rather than a verdict about linearity.
        if n not in chosen and jnp.issubdtype(jnp.asarray(space.latent(n).init).dtype, jnp.inexact)
    ]
    points = [dict(at)]
    for index in range(1, max(count, 1)):
        point = dict(at)
        for position, (member, parameters) in enumerate(drawable):
            if parameters is None:
                continue
            template = jnp.asarray(space.latent(member).init)
            sub_key = jax.random.fold_in(jax.random.fold_in(key, index), position)
            loc, scale = parameters
            point[member] = jnp.asarray(loc, dtype=template.dtype) + jnp.asarray(
                scale, dtype=template.dtype
            ) * jax.random.normal(sub_key, template.shape, dtype=template.dtype)
        points.append(point)
    return points


def _worse(current: float, value: float) -> float:
    """``max`` that PROPAGATES NaN, and keeps an ``Unresolved`` that wins.

    ``max`` alone drops NaN silently on one argument order and returns it on
    the other, and a NaN here means "this probe was unusable", which must
    survive a merge across at-points rather than depend on iteration order.
    """
    if np.isnan(current) or np.isnan(value):
        return float("nan")
    return current if current >= value else value


def _probe_anchor(latent: Any) -> float:
    """The magnitude one probe is measured in: the latent's PRIOR width.

    D16 axis 1, ruled 2026-08-27. This used to be ``max|init|``
    (:func:`_magnitude`), and the reason it moved is written into that
    function's own neighbourhood: an all-zero ``init`` has no scale to take,
    falls back to 1.0, and the probes become absolute -- so a sky declared at
    zero, which is the ordinary declaration, was probed at magnitudes that
    have nothing to do with where the sampler will go. The PRIOR is where it
    will go, so that is what a probe is measured in.

    Measured before the change: on ``signal + 1e-7 signal**2`` with
    ``max|init| = 1`` and a prior width of 100, the old anchor reached 1e3 and
    accepted (worst 8.24e-05 against rtol 1.19e-03) while the new one reaches
    1e5 and refuses. Same criteria, different reach.

    Falls back to ``max|init|`` when there is no Gaussian prior to read -- a
    prior-free latent, or one whose prior is not Gaussian in the latent
    itself. :func:`_gaussian_parameters` identifies that by TYPE and not by
    attribute, which matters here for the same reason it matters there:
    ``LogNormal`` carries a ``.scale`` that is a width in ``log x``.

    ``_magnitude`` itself is untouched: ``engines.py`` reads it to turn a
    gradient block's ``learning_rate`` into an absolute step, and that is a
    different question with a different right answer.
    """
    parameters = _gaussian_parameters(latent.prior)
    if parameters is not None and not _holds_a_tracer(parameters[1]):
        width = float(np.max(np.abs(np.asarray(parameters[1]))))
        if np.isfinite(width) and width > 0.0:
            return width
    return _magnitude(latent)


def _single_probe(
    space: ParameterSpace, name: str, key: jax.Array
) -> Callable[[int, float], jax.Array]:
    """Probes for a one-latent block: ``magnitude * scale * N(0, 1)``.

    Factored out because the log-space check
    (:func:`~rheplicant.inference.loglinear.check_log_linearity`) asks the same
    question of a different map and must ask it at the SAME points. A second
    copy of the probe scheme would let the two drift into probing different
    models while both reported on "linearity".
    """
    latent = space.latent(name)
    magnitude = _probe_anchor(latent)

    def probe_at(index: int, scale: float) -> jax.Array:
        return (
            magnitude
            * scale
            * jax.random.normal(
                jax.random.fold_in(key, index), latent.init.shape, dtype=latent.init.dtype
            )
        )

    return probe_at


def _group_probe(
    space: ParameterSpace, selected: Sequence[str], key: jax.Array
) -> Callable[[int, float], dict[str, jax.Array]]:
    """Probes for a grouped block, per latent, at each member's own scale.

    Sub-keys are folded in by position in the SORTED names, so permuting the
    caller's ``names`` probes the model at the same points.
    """
    ordered = sorted(selected)

    def probe_at(index: int, scale: float) -> dict[str, jax.Array]:
        root = jax.random.fold_in(key, index)
        return {
            member: _probe_anchor(space.latent(member))
            * scale
            * jax.random.normal(
                jax.random.fold_in(root, position),
                space.latent(member).init.shape,
                dtype=space.latent(member).init.dtype,
            )
            for position, member in enumerate(ordered)
        }

    return probe_at


def _reported(values: jax.Array, kept: jax.Array, departure: jax.Array, threshold: float) -> float:
    """The worst of ``values`` among the elements the roundoff floor kept.

    That is the number the criterion actually JUDGED, so it is the number a
    refusal quotes: reporting the raw maximum instead would print values above
    the threshold beside a verdict of "pass" and read as a broken guard.

    **A masked element with a real departure is reported, not zeroed.** The
    mask holds two very different populations. An element whose departure is
    exactly 0.0 is bitwise-affine, and a reported 0.0 says so truthfully. An
    element whose departure is non-zero, sits under the floor, and WOULD have
    breached ``threshold`` had it been judged is a question the arithmetic
    could not answer -- reporting 0.0 there states the model was measured and
    found exactly affine, which is the opposite of what happened. Those come
    back as :class:`~bayesmith.exact.linearity.Unresolved`, a float whose
    string says it was not judged.

    Adopted with D16 axis 5 rather than invented here, and the type is
    IMPORTED rather than redefined so that ``isinstance`` means the same thing
    on both sides of the seam.
    """
    judged = float(jnp.max(jnp.where(kept, values, 0.0)))
    declined = (departure > 0) & ~kept & (values > threshold)
    if not bool(jnp.any(declined)):
        return judged
    return Unresolved(max(judged, float(jnp.max(jnp.where(declined, values, 0.0)))))


def _affinity_errors(
    g: Callable[[Any], jax.Array],
    zero: Any,
    probe_at: Callable[[int, float], Any],
    scales: Sequence[float],
    rtol: float | None,
    noise: Any | None = None,
) -> tuple[dict[float, float], dict[float, float] | None, list[float], float]:
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
    # NOT the 1e-300 literal this used to carry. Measured on the other side of
    # the seam: 1e-300 underflows to 0.0 in float32, so an element the block
    # cannot move at all gives 0/0 = NaN and the finiteness branch reads that
    # as a refusal -- condemning an entirely honest model.
    tiny = float(jnp.finfo(baseline.dtype).tiny)
    # D16 axis 3 (owner ruled 2026-08-27). "Small" has to be small compared to
    # SOMETHING, and a departure can be far under the relative tolerance while
    # being many noise widths wide -- which is the regime a conjugate solve
    # gets wrong. Measured: on `signal + 1e-7 signal**2` the relative column
    # reads 0.000e+00 at every probe while the sigma-weighted one reads
    # 6.262e-02 against a threshold of 1e-3.
    #
    # `None` when the caller passed no noise, and then this criterion simply
    # does not apply. That is NOT the shape D17's log gate took, and the
    # difference is worth stating: there, a missing noise model made a
    # POSITIVE claim unsafe (a log block that would be refused downstream), so
    # the verdict went conservative. Here it only makes the check weaker, and
    # the fallback verdict is the one this function gave before the axis
    # moved -- so silence is the honest default rather than a refusal nobody
    # could act on.
    sigma = None if noise is None else jnp.abs(jnp.asarray(noise.std(baseline)))
    errors: dict[float, float] = {}
    weighted_errors: dict[float, float] | None = None if sigma is None else {}
    verdicts: dict[float, bool] = {}
    for index, scale in enumerate(scales):
        probe = probe_at(index, scale)
        actual = g(probe)
        predicted = baseline + tangent(probe)
        # PER ELEMENT, since D16 axis 5 (owner ruled 2026-08-27). A maximum
        # over the whole output lets a bright entry supply both the yardstick
        # and the roundoff floor for a faint one: measured, six channels at
        # 1e8 beside six carrying a 1e-2 quadratic reported 2.81e-14 -- a
        # false "perfectly affine" -- and the same curvature alone refuses.
        #
        # Measure against the VARIATION, not the total: a large constant
        # offset would otherwise hide a completely nonlinear response.
        variation = jnp.abs(actual - baseline)
        departure = jnp.abs(actual - predicted)
        magnitude = jnp.maximum(jnp.abs(actual), jnp.abs(baseline))
        relative = departure / jnp.maximum(variation, tiny)
        # A departure smaller than the arithmetic's OWN noise floor is not
        # evidence of curvature; without this the relative measure explodes at
        # small probes, where the variation is vanishing but roundoff is not,
        # and rejects perfectly linear blocks. The floor is per element and set
        # by the magnitudes actually being differenced AT THIS ELEMENT -- not
        # by a constant, which would exempt every model whose prediction is
        # small in its own units, and not by the output's maximum, which is
        # the dilution above.
        judged = departure > RELATIVE_FLOOR_FACTOR * epsilon * magnitude
        finite = bool(jnp.all(jnp.isfinite(relative)))
        errors[scale] = _reported(relative, judged, departure, rtol) if finite else (float("nan"))
        # NaN must count as a FAILURE, not a pass: `nan > rtol` is False, so a
        # naive comparison treats an unusable probe as evidence of linearity.
        refused = (not finite) or bool(jnp.any(judged & (relative > rtol)))

        if sigma is not None:
            # In the units the likelihood divides by, and gated by a floor of
            # its OWN -- four decades below the relative column's. Ungated it
            # measures DYNAMIC RANGE rather than curvature; gated at the
            # relative column's factor it could not fire on any model with
            # more signal than noise.
            in_sigma = departure / jnp.maximum(sigma, tiny)
            above = departure > WEIGHTED_FLOOR_FACTOR * epsilon * magnitude
            weighted_finite = bool(jnp.all(jnp.isfinite(in_sigma)))
            weighted_errors[scale] = (
                _reported(in_sigma, above, departure, WEIGHTED_RTOL)
                if weighted_finite
                else float("nan")
            )
            refused = (
                refused
                or (not weighted_finite)
                or bool(jnp.any(above & (in_sigma > WEIGHTED_RTOL)))
            )
        verdicts[scale] = refused

    failed = sorted(scale for scale, bad in verdicts.items() if bad)
    return errors, weighted_errors, failed, rtol
