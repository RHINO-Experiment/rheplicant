"""The cheap diagnostics: what a document asks before it pays for a fit.

``identifiability`` and ``score_directions`` need a ParameterSpace, the fit
twin and the state -- no observed data, no noise model, no seed.  That is
what makes them the checks a user is meant to run first, and it is why they
sweep so few keys.

The two package entry points have OPPOSITE calling conventions and the
difference is not cosmetic: ``identifiability`` puts ``names``/``at``/``rtol``
behind a bare ``*`` (identifiability.py:418-426), while ``score_directions``
takes ``names`` and ``at`` positional-or-keyword (reduced_basis.py:114-120).
Both are called by keyword here so that neither call shape can be copied onto
the other.

``at:`` is resolved through :func:`rheplicant.config.values.resolve_value`,
the same value grammar ``inference.observed.<name>.at`` reads, so
``{value: 1.0, unit: K}`` means here what it means there.

Both executors differentiate ``built.inference.fit_twin``, never
``built.twin``: the fit twin is the model twin with ``inference.twin:``
applied, and the repair that section exists for -- ``without: [noise]`` -- is
exactly what makes the model differentiable at all
(``refuse_stochastic_stages``, ``inference/parameters.py:125``).  The two are
the same object whenever a document declares no ``inference.twin:``, which is
why the tests build one that does.

``mmodes`` is the cheapest of the family and the odd one out: no latents,
no noise, no observed data, and nothing off ``built.inference`` at all.  It
reads two declared resources and ``built.state.coords``, which is why
:func:`resolve_reference` is imported at the head -- ``config.refs`` is a
config-layer module with no package dependencies, the way ``sections/model.py``
imports it.  The lazy-import rule in this layer is about
``rheplicant.inference`` and numpyro, not about ``config.refs``.

``gradient`` joins them at the foot: one ``jax.grad`` of a named objective,
no optimiser, and the point moved by the same ``at:``.  Its four objectives
are defined HERE -- ``_OBJECTIVES`` is a config-layer table, because the
schema's ``chi2``/``sum_squares``/``mean`` exist in no package module
(measured; ``mean_squared_error`` is the only shipped objective of any kind)
-- and ``chi2`` is where ``inference.noise.include_logdet`` acquires its
first consumer in ``src/``.

``predict`` closes the family and is the only exit here that reads nothing
from the document but a NAME: its input is an earlier run's product, reached
through the run-level ``reuse:``.  ``condition`` is NOT here: it rides
``_conjugate_block``, so Task 6 put it in ``conjugate.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from rheplicant.config.errors import ConfigError
from rheplicant.config.refs import resolve_reference
from rheplicant.config.sections.exit_support import (
    _PROBE,
    _binds,
    _noise,
    _number,
    _observed,
    _space,
    _sweep,
    register,
    reuse_of,
)
from rheplicant.config.sections.runs import RunSpec
from rheplicant.config.values import resolve_value

_IDENTIFIABILITY_KEYS = frozenset({"names", "at", "rtol"})
_SCORE_KEYS = frozenset({"names", "at"})


def _names(run: RunSpec) -> tuple[str, ...] | None:
    """``names:`` -> a tuple, or None when the document declares none.

    Four shapes are refused here, and for one reason: this single helper
    serves two entry points that DISAGREE about every one of them, so
    anything it lets through means one thing on ``identifiability`` and
    another on ``score_directions``.  Measured, with the guard bypassed:

    * a bare string -- ``identifiability`` reads it as a group of one, by an
      explicit ``isinstance(names, str)`` (``identifiability.py:180``, in
      prose at ``:174-176``), so ``names: gd`` asks for one latent called
      ``gd``; ``score_directions`` does a plain ``tuple(names)``
      (``reduced_basis.py:164``) and reads the same document as ``g`` and
      ``d``, two rows.  In YAML a bare ``names: g`` is far more often a typo
      than an intention, and ``[g]`` says one thing to both.
    * an empty list -- ``identifiability`` refuses it by name;
      ``score_directions`` reaches a concatenate of nothing and raises
      ``ValueError: at least one array or dtype is required``, which is not
      a ConfigError and names no run.
    * a non-string member -- ``names: [[g]]`` reaches a dict lookup on a list
      and raises ``TypeError: unhashable type: 'list'``.
    * a repeat -- ``identifiability`` refuses ``['g', 'g']`` because two
      copies of one latent are exactly degenerate with each other;
      ``score_directions`` silently returns ONE key for the two-name ask, so
      ``names: [g, d, g]`` hands back a 2-key product for a 3-name list and
      a caller zipping the two is off by one.  That is the permutation bug
      ``reduced_basis.py:171-180`` is named after, reached from the far side.

    Which latents the space actually declares is NOT checked here: that
    refusal is the package's own on both kinds, and it names the declared
    set.  ``at:`` is different -- see :func:`_at_values`.
    """
    if "names" not in run.options:
        return None
    names = run.options["names"]
    if not isinstance(names, list) or not names or not all(
            isinstance(name, str) for name in names):
        raise ConfigError(
            f"runs[{run.name!r}]: names: is a non-empty list of latent names "
            f"-- [g] for a block of one; got {names!r}."
        )
    repeated = sorted({name for name in names if names.count(name) > 1})
    if repeated:
        raise ConfigError(
            f"runs[{run.name!r}]: names: lists {repeated} more than once. Two "
            "copies of one latent are exactly degenerate with each other, so "
            "identifiability refuses the block outright while "
            "score_directions returns one row for the two asks -- a product "
            "with fewer keys than the document named, which a caller zipping "
            f"the two together reads off by one; got {names!r}."
        )
    return tuple(names)


def _at_values(run: RunSpec, built: Any, space: Any) -> dict[str, Any]:
    """``at:`` -> the declared overrides alone, ``{}`` when there are none.

    A latent is a local property of a nonlinear model, so ``at:`` is the
    question "what does the prediction look like *here*".  It moves the
    POINT, not the model: every latent it does not name stays at its declared
    ``init:``, and ``identifiability``/``score_directions`` merge the
    overrides over the space's initial values themselves.  So the overrides
    come back on their own, and ``{}`` -- not None -- is the right empty:
    both entry points accept ``at={}``.

    The values go through :func:`resolve_value`, exactly as
    ``inference.observed.<name>.at`` does, so ``{value: 1.0, unit: K}`` means
    here what it means there.  A name the space does not declare is refused
    HERE rather than downstream, because this layer can see the space and can
    name what it does declare.

    An ABSENT ``at:`` is ``{}``; a ``at:`` DECLARED with an empty YAML value
    is refused, because ``None`` is not a mapping.  The two are read apart by
    membership rather than by ``.get``, which would fold them together and
    make a key the user wrote into a silent no-op -- :func:`_names` already
    refuses ``names:`` that way, and one grammar cannot mean "you meant
    nothing" on one key and "you meant something" on its neighbour.

    ``gradient`` calls this too -- one helper, three callers, three
    positional arguments (plan section 3.1).  There is no second ``at:``
    resolver anywhere in this layer, which is what keeps a value node
    meaning one thing on all three kinds.
    """
    where = f"runs[{run.name!r}]"
    if "at" not in run.options:
        return {}
    at = run.options["at"]
    if not isinstance(at, Mapping):
        raise ConfigError(
            f"{where}: at: is a mapping of latent -> value, the point the "
            f"derivative is taken at; got {at!r}."
        )
    unknown = sorted(set(at) - set(space.names))
    if unknown:
        raise ConfigError(
            f"{where}: at: names {unknown}, which inference.parameters does "
            f"not declare; it declares {list(space.names)}."
        )
    # The dtype is the DOCUMENT's, as it is for a latent's own ``init:`` and
    # ``ref:`` (parameters.py).  Measured: under the repo's default x32 the
    # cast changes nothing -- an integer ``at: {g: 2}`` reaches the same
    # answer as float32 2.0, because jax promotes weakly -- so no test in
    # this layer can see it and none pretends to.  It is here so that a
    # float64 document gets float64 overrides rather than whatever the value
    # node happened to resolve to.
    context = built.context
    return {name: jnp.asarray(resolve_value(node, context).value,
                              dtype=context.dtype)
            for name, node in at.items()}


def _bounded_rtol(run: RunSpec) -> float:
    """``rtol:`` -> a float in ``[0, 1)``, which is where a rank lives.

    The cutoff is ``rtol * s_max``, so both ends of that interval are
    verdicts about arithmetic rather than about the model, and the package
    checks neither.  Below 0 every singular value is above the cutoff and
    nothing is null; at 1 the largest singular value is itself AT the cutoff
    and nothing is above it -- measured rank 0 / nullity 2 at ``rtol: 1.0``
    on a pair that is genuinely two directions and on one that is genuinely
    one, the same answer for two different models.  The interval is
    half-open because 0.999 still discriminates (measured rank 1, nullity 1
    on the identified pair), so the ceiling is where the arithmetic goes
    vacuous and not a round number chosen for looks.
    """
    rtol = _number(run, "rtol", run.options["rtol"], kind=float, minimum=0.0)
    if rtol >= 1.0:
        raise ConfigError(
            f"runs[{run.name!r}]: rtol: must be < 1; got {rtol!r}. The cutoff "
            "is rtol * s_max, so at 1 the largest singular value sits at the "
            "threshold and NOTHING is identified -- rank 0 over the whole "
            "block, whatever the model. It is the mirror of the floor: below "
            "0 nothing is null, at or above 1 nothing is identified, and "
            "neither verdict is about the latents."
        )
    return rtol


@register("identifiability")
def _run_identifiability(run: RunSpec, built: Any,
                         *, results: Any = None) -> Any:
    """``kind: identifiability`` -> the package's IdentifiabilityReport.

    The report holds numpy arrays and Python ints and cannot be jitted
    (identifiability.py:255-267), so this executor drives the call plainly.
    Nor does it need an x64 document: ``identifiability`` forces x64 on for
    its own duration and casts the selected latents (``:481``, ``:408``), so
    an ordinary float32 config run LANDS -- its "even with x64" refusal
    (``:501-509``) is for a model that pins its output dtype.
    """
    from rheplicant.inference import identifiability

    _sweep(run, _IDENTIFIABILITY_KEYS)
    space = _space(run, built)
    kwargs: dict[str, Any] = {}
    names = _names(run)
    if names is not None:
        kwargs["names"] = names
    at = _at_values(run, built, space)
    if at:
        kwargs["at"] = at
    if "rtol" in run.options:
        kwargs["rtol"] = _bounded_rtol(run)
    # Keyword-only, all three.  Positional is a TypeError here and legal one
    # module away, which is exactly why this call names every argument.
    return identifiability(space, built.inference.fit_twin, built.state,
                           **kwargs)


@register("score_directions")
def _run_score_directions(run: RunSpec, built: Any,
                          *, results: Any = None) -> Any:
    """``kind: score_directions`` -> ``{latent: (size, n_data)}``.

    Returned exactly as the package built it.  The order is the caller's,
    deliberately: jax rebuilds a dict from its flattened, sorted form, so
    re-keying or re-sorting this product hands back alphabetical names and
    reintroduces the bug reduced_basis.py:171-180 is named after.
    """
    from rheplicant.inference import score_directions

    _sweep(run, _SCORE_KEYS)
    space = _space(run, built)
    kwargs: dict[str, Any] = {}
    names = _names(run)
    if names is not None:
        kwargs["names"] = names
    at = _at_values(run, built, space)
    if at:
        kwargs["at"] = at
    return score_directions(space, built.inference.fit_twin, built.state,
                            **kwargs)


# --- kind: gradient --------------------------------------------------------

_GRADIENT_KEYS = frozenset({"objective", "of", "at"})


def _chi2(run: RunSpec, built: Any) -> Any:
    """``-2 log p`` under the document's noise model: ``sum r^2 / sigma^2``,
    plus the ``log 2 pi sigma^2`` term unless ``include_logdet: false``.

    The term is present by DEFAULT -- undeclared means nothing is passed and
    :class:`NoiseModelLikelihood`'s own ``include_logdet=True`` stands, so it
    is ``false`` that removes it, not silence.  This is spelled with the
    package's likelihood rather than open-coded: the log-determinant belongs
    to a sigma that may depend on the prediction, and
    inference/noise.py:382-383 is where its sign and factor are already
    right.

    ``inference.noise.include_logdet`` has three states.  Undeclared it is
    None and nothing is passed, so the package's own default stands; True and
    False are passed through.  The None state is reachable only under a sigma
    that does NOT depend on the prediction -- noise.py:156-159 requires the
    key for ``kind: radiometer`` and refuses it everywhere else -- and there
    the log-determinant is an additive constant no GRADIENT can see.  It is
    still not the same objective, so it is pinned by evaluating this closure
    directly rather than by differentiating it.

    ``radiometer_frozen`` decides a sigma ARRAY with no model behind it, so
    there is no likelihood to build -- and a frozen sigma's log-determinant is
    again an additive constant, which is why that route is the plain weighted
    sum.
    """
    from rheplicant.inference import NoiseModelLikelihood

    noise = _noise(run, built)
    observed = _observed(run, built)
    build = built.inference.noise
    if build.model is None:
        def frozen_chi2(prediction: Any) -> Any:
            return jnp.sum(((observed - prediction) / noise) ** 2)

        return frozen_chi2
    declared = ({} if build.include_logdet is None
                else {"include_logdet": bool(build.include_logdet)})
    likelihood = NoiseModelLikelihood(noise=noise, **declared)

    def chi2(prediction: Any) -> Any:
        return -2.0 * likelihood(prediction, observed)

    return chi2


def _sum_squares(run: RunSpec, built: Any) -> Any:
    """``sum prediction^2`` -- pure in the prediction, reading no data."""

    def sum_squares(prediction: Any) -> Any:
        return jnp.sum(prediction ** 2)

    return sum_squares


def _mean(run: RunSpec, built: Any) -> Any:
    """``mean(prediction)`` -- pure in the prediction, reading no data."""

    def mean(prediction: Any) -> Any:
        return jnp.mean(prediction)

    return mean


def _mse(run: RunSpec, built: Any) -> Any:
    """The package's own mean_squared_error -- what optimize minimises.

    It is here so that a gradient and a fit can be asked about the same
    objective under the same name; a future author who wants the two
    vocabularies kept apart drops this entry and one test.
    """
    from rheplicant.inference import mean_squared_error

    observed = _observed(run, built)

    def mse(prediction: Any) -> Any:
        return mean_squared_error(prediction, observed)

    return mse


# The schema's chi2/sum_squares/mean do not exist in the package -- measured,
# see this plan's Executor's note -- so this table IS the definition.  Each
# entry takes (run, built) and returns a closure over the prediction alone,
# so an objective that reads no data never reaches for inference.observed.
_OBJECTIVES = {"chi2": _chi2, "mean": _mean, "mse": _mse,
               "sum_squares": _sum_squares}




def _scores_a_pair(where: str, target: str, scoring: Any) -> None:
    """Refuse a ``python:`` objective the seam cannot CALL the way it calls.

    This is a contract check, not a restriction on the hatch (decision
    D-C11: the hatch is recorded, not restricted).  It forbids nothing a
    working objective can do -- it asks only whether the callable accepts the
    two arguments this seam passes, which every objective that runs must.

    Without it the natural mistake reaches the user as a raw
    ``TypeError: math.sqrt() takes exactly one argument (2 given)`` naming no
    run.  And the mistake is the LIKELY one, not an exotic one: three of the
    four named objectives read no data, so a user writing the obvious
    analogue ``def my_objective(prediction)`` lands here immediately.

    A callable ``inspect`` cannot describe -- some C builtins, some jax
    wrappers -- is passed through: the call itself is then the check, and
    guessing would refuse working objectives.
    """
    binds, signature = _binds(scoring, _PROBE, _PROBE)
    if not binds:
        raise ConfigError(
            f"{where}: objective: {target!r} cannot be called as "
            f"(prediction, observed) -- its signature is {signature}. An "
            "objective takes BOTH, the same shape optimize's loss: takes, "
            "even when it reads only the first: chi2 and mse weigh the "
            "residual, sum_squares and mean ignore observed and still "
            "accept it."
        )


def _objective(run: RunSpec, built: Any) -> Any:
    """``objective:`` -> ``prediction -> scalar``.

    The ``{python: 'mod:fn'}`` branch goes through the same
    :func:`~rheplicant.config.hatch.import_target` seam ``optimize``'s
    ``loss:`` uses, and takes the same ``(prediction, observed)`` shape --
    but it is resolved here rather than through ``exits._loss_fn``, because
    a loss a calibrator minimises and an objective a gradient
    differentiates are different vocabularies with different keys.
    """
    where = f"runs[{run.name!r}]"
    objective = run.options.get("objective")
    if objective is None:
        raise ConfigError(
            f"{where}: objective: is required -- one of "
            f"{sorted(_OBJECTIVES)}, or {{python: 'mod:fn'}}. The four "
            "differ in sign as well as in size on the same document, so "
            "there is no default worth guessing."
        )
    if isinstance(objective, str) and objective in _OBJECTIVES:
        return _OBJECTIVES[objective](run, built)
    if isinstance(objective, Mapping) and set(objective) == {"python"}:
        from rheplicant.config.hatch import import_target

        target = objective["python"]
        scoring = import_target(target)
        _scores_a_pair(where, target, scoring)
        observed = _observed(run, built)

        def imported(prediction: Any) -> Any:
            scored = scoring(prediction, observed)
            if jnp.ndim(scored) != 0:
                raise ConfigError(
                    f"{where}: objective: {target!r} returned shape "
                    f"{tuple(jnp.shape(scored))}; an objective is a SCALAR "
                    "score, because jax.grad is defined for a scalar output "
                    "alone. Reduce it -- jnp.sum or jnp.mean over the "
                    "residual is what the four named objectives do."
                )
            return scored

        return imported
    raise ConfigError(
        f"{where}: objective: is one of {sorted(_OBJECTIVES)} or "
        f"{{python: 'mod:fn'}}; got {objective!r}."
    )


def _of_paths(run: RunSpec) -> tuple[str, ...]:
    """``of:`` -> the declared paths, one or several, in DECLARED order.

    The order is the caller's and is kept, for the reason
    :func:`_run_score_directions` gives: a product re-keyed into JAX's sorted
    order is the bug ``reduced_basis.py:171-180`` is named after.

    A repeat is refused, for the reason :func:`_names` gives about
    ``names:``.  Measured before this guard: ``of: [gain.gain, gain.gain]``
    was accepted and handed back a product with ONE key for a two-path ask,
    because a dict keyed by path cannot hold the same path twice -- and a
    caller zipping that product against their own list reads it off by one.
    Two derivatives of one leaf are also exactly the same number, so the
    document can only have meant something it did not write.
    """
    where = f"runs[{run.name!r}]"
    of = run.options.get("of")
    if of is None:
        raise ConfigError(
            f"{where}: of: is required -- the path, or list of paths, whose "
            "leaves this gradient differentiates."
        )
    paths = [of] if isinstance(of, str) else of
    if (not isinstance(paths, list) or not paths
            or not all(isinstance(path, str) for path in paths)):
        raise ConfigError(
            f"{where}: of: is a path or a non-empty list of paths; got "
            f"{of!r}."
        )
    repeated = sorted({path for path in paths if paths.count(path) > 1})
    if repeated:
        raise ConfigError(
            f"{where}: of: lists {repeated} more than once. The product is "
            "keyed BY path, so a repeat comes back as one key for two asks "
            "-- fewer keys than the document named, which a caller zipping "
            "the two together reads off by one -- and the two derivatives "
            f"would be the same number anyway; got {of!r}."
        )
    return tuple(paths)


@register("gradient")
def _run_gradient(run: RunSpec, built: Any, *, results: Any = None) -> Any:
    """One differentiation of a named objective (schema section 4.7.9).

    The evaluation point is the document's own: where ``inference.parameters``
    exists the latents are bound at their declared ``init:``, overridden by
    whatever ``at:`` names, and where it does not the twin's own leaves
    stand.  ``of:`` then names leaves of THAT twin, in Plan 1B's path
    grammar -- resolved through ``resolve_path_on`` rather than synthesised
    into a key path, because a config path compiles to a selector that walks
    the object's own accessors and the written string never appears in the
    key path that comes back (config/paths.py's module docstring).

    The product is ``{declared path: gradient}``: a bare array would lose
    which leaf it belonged to as soon as ``of:`` named two.

    :func:`_space` is NOT called here on purpose -- ``gradient`` does not
    require latents, so the shared "fits latents" refusal would say the wrong
    thing.  The ``at:``-without-parameters refusal below is the one this kind
    needs.
    """
    import equinox as eqx
    import jax

    from rheplicant.config.paths import resolve_path_on
    from rheplicant.inference import build_forward_fn

    where = f"runs[{run.name!r}]"
    _sweep(run, _GRADIENT_KEYS)
    paths = _of_paths(run)
    objective = _objective(run, built)
    space = built.inference.space
    twin = built.inference.fit_twin
    if space is None:
        if "at" in run.options:
            raise ConfigError(
                f"{where}: at: overrides the declared init: of a latent, and "
                "this document declares no inference.parameters. Without "
                "them the gradient runs at the twin's own leaf values, which "
                "is what dropping at: asks for."
            )
    else:
        twin = space.bind(twin, {**space.initial_values(),
                                 **_at_values(run, built, space)})
    selectors = [resolve_path_on(path, twin).selector for path in paths]
    spec = jax.tree.map(lambda _: False, twin)
    for selector in selectors:
        spec = eqx.tree_at(selector, spec, replace=True)
    forward, params0 = build_forward_fn(twin, built.state, spec)
    grads = jax.grad(lambda params: objective(forward(params)))(params0)
    return {path: selector(grads)
            for path, selector in zip(paths, selectors, strict=True)}


# --- kind: mmodes ----------------------------------------------------------

_MMODES_KEYS = frozenset({"projector", "sky"})


def _mmodes_ref(run: RunSpec, built: Any, key: str) -> Any:
    """``projector:``/``sky:`` -> the LIVE object the ``{ref}`` names.

    One helper for both keys, so the two cannot drift into accepting
    different shapes of the same node.  ``resolve_reference`` returns the
    object itself rather than a copy (refs.py's module docstring), so this
    exit and the twin's own operator share one projector -- which is the
    point, and which is why nothing here assigns to it: Equinox Modules are
    frozen, and a variant would have to be built with ``eqx.tree_at``.
    """
    node = run.options.get(key)
    if not isinstance(node, Mapping) or set(node) != {"ref"}:
        raise ConfigError(
            f"runs[{run.name!r}]: {key}: is {{ref: resources.<kind>.<name>}} "
            "and is required -- kind: mmodes observes two DECLARED resources "
            f"and nothing inline; got {node!r}."
        )
    return resolve_reference(node["ref"], built.context)


def _evaluates_a_grid(where: str, sky_model: Any) -> None:
    """Refuse a ``sky:`` the exit cannot CALL the way it calls.

    The mirror of :func:`_scores_a_pair`, and for the same reason.
    ``callable()`` alone is a much weaker predicate than the ``projector:``
    side's ``hasattr(..., "mmodes")``: it admits any object with a
    ``__call__`` whatever its arity, and the value grammar hands one over by
    the front door -- ``resources.arrays.<name>: {python: 'operator:add'}``
    delivers the uncalled attribute itself (hatch.py's presence-of-the-key
    rule), and ``{ref}`` to it reaches ``sky_model(freq)`` as
    ``TypeError: add expected 2 arguments, got 1``, naming no run.  Measured.

    So the two routes are guarded to the same DEPTH: a projector must have
    the method this exit calls, and a sky must accept the one argument this
    exit passes.  It forbids nothing a working sky model can do -- all three
    shipped models bind ``(freq)`` -- and a callable ``inspect`` cannot
    describe is passed through, because guessing would refuse working ones.
    """
    binds, signature = _binds(sky_model, _PROBE)
    if not binds:
        raise ConfigError(
            f"{where}: sky: names a {type(sky_model).__name__} that cannot be "
            f"called as (freq) -- its signature is {signature}. A sky model "
            "is evaluated ON the run's own frequency grid, which is the one "
            "argument every AbstractSkyModel takes and the only one this exit "
            "has to give."
        )


@register("mmodes")
def _run_mmodes(run: RunSpec, built: Any, *, results: Any = None) -> Any:
    """``kind: mmodes`` -> ``DriftScanProjector.mmodes`` on the run's own grid.

    Two references and nothing else.  The beam is the projector's own traced
    ``beam_alms`` -- ``mmodes(sky, coords)`` has no ``beam=`` argument to give
    it (driftscan.py:663) -- and the coords come off ``built.state`` because
    ``mmodes`` reads ``coords.extra["lst_deg"]`` and cross-checks
    ``coords.pointing`` against the projector's fixed az/el itself
    (driftscan.py:387-436).  That cross-check's refusal is the package's and
    is left alone: it names the disagreement, and nothing this layer could
    say about it would be more specific.

    The product is the complex ``(n_freq, lmax + 1)`` spectrum, ``lmax``
    being the PROJECTOR's; ``(lmax+1)(lmax+2)//2`` is the sky's alm width and
    a different number.
    """
    _sweep(run, _MMODES_KEYS)
    where = f"runs[{run.name!r}]"
    projector = _mmodes_ref(run, built, "projector")
    if not hasattr(projector, "mmodes"):
        raise ConfigError(
            f"{where}: projector: names a {type(projector).__name__}, which "
            "has no mmodes(). The m-mode expansion is a drift scan's own "
            "spectrum, so engine: driftscan is the one engine that offers it."
        )
    # Off the BUILT object, never off the spec: a {ref} may name a projector
    # declared anywhere in resources:, and optimizations: [cache_beam_rotation]
    # REPLACES that object with to_reference_frame()'s return
    # (projectors.py:240-241), which keeps normalize_beam and changes the
    # frame.
    if projector.normalize_beam:
        raise ConfigError(
            f"{where}: kind: mmodes needs a normalize_beam: false projector. "
            "forward() additionally divides the TOD by the ones-map "
            "denominator, which the m-mode expansion cannot represent, so "
            "these coefficients would not be the spectrum of forward() "
            "(measured ~18x off). The package refuses the same pairing as a "
            "StateValidationError once the whole document is built; this is "
            "that refusal, before anything is traced."
        )
    sky_model = _mmodes_ref(run, built, "sky")
    if not callable(sky_model):
        raise ConfigError(
            f"{where}: sky: names a {type(sky_model).__name__}, which is not "
            "a sky model. A {ref: resources.sky_models.<name>} resolves to "
            "the MODEL, not to maps; this exit evaluates it on the run's own "
            "frequency grid to get the (n_freq, n_pix) maps mmodes takes."
        )
    _evaluates_a_grid(where, sky_model)
    coords = built.state.coords
    maps = sky_model(coords.freq)
    # `shape` and not the extents: `_validate_sky` (driftscan.py:551-557)
    # already names n_freq, n_pix AND the nside they follow from, and says it
    # better than this layer could.  What it cannot survive is an argument
    # with no `shape` at all, which is where it reaches first -- so this
    # guard is exactly the attribute the package is about to touch, and
    # nothing more.
    if not hasattr(maps, "shape"):
        raise ConfigError(
            f"{where}: sky: names a {type(sky_model).__name__} whose "
            f"__call__(freq) returned a {type(maps).__name__}, not maps. The "
            "contract is __call__(freq) -> (n_freq, n_pix) brightness "
            "temperatures (radio/sky/model.py's module docstring); the "
            "projector checks the two extents itself and cannot report on "
            "something with no shape."
        )
    return projector.mmodes(maps, coords)


# --- kind: predict ---------------------------------------------------------

_PREDICT_KEYS = frozenset({"n_draw"})

#: The kinds whose product IS a stack of draws, mapped to the clause that
#: says why the count they kept is a ceiling.  A dict rather than a tuple
#: because the three reach that ceiling for three different reasons and one
#: sentence for all of them would be false twice: ``nuts`` has a warmup and
#: ``get_samples()`` has already dropped it, ``npe`` has no warmup at all.
#: The dispatch is on the KIND -- never on ``hasattr(product, "samples")``,
#: which ``conjugate.wiener`` with ``width: fisher`` already defeats on the
#: covariance side (conjugate.py:195, :392) and which would let any future
#: kind carrying a ``.samples`` in past this table.
_DRAW_SOURCES = {
    "plan.sample": "plan.sample discards its warmup before returning",
    "nuts": "get_samples() returns the post-warmup draws alone -- "
            "num_samples x num_chains is the whole chain",
    "npe": "npe drew exactly the inference.npe.sample.n_draws: it was "
           "asked for and has no warmup to recover",
}


@register("predict")
def _run_predict(run: RunSpec, built: Any, *, results: Any = None) -> Any:
    """``kind: predict`` -- an earlier run's product, pushed through the model.

    TWO routes, over ``fisher`` plus every kind :data:`_DRAW_SOURCES` lists,
    chosen by the ``kind`` of the run ``reuse:`` names, because
    ``RunResult.product`` has no uniform type:

    * a ``fisher`` run carries a covariance under ``product["covariance"]``,
      and :func:`~rheplicant.inference.propagate_covariance` turns it into a
      per-sample prediction standard deviation by the delta method.  Nothing
      is drawn on this route, so ``n_draw:`` is refused rather than ignored.
    * a ``plan.sample``, ``nuts`` or ``npe`` run carries draws, and
      :func:`~rheplicant.inference.predict_from_samples` runs the pipeline
      over them.  **Those predictions are NOISELESS** -- the likelihood's own
      scatter is not added back (numpyro_bridge.py:337-338).  "Predictive"
      usually means the opposite, so it is said here in full.

    ``n_draw:`` keeps the LAST draws on all three, and on a multi-chain
    ``nuts`` product an ``n_draw:`` at or below ``num_samples`` reads ONE
    chain: ``get_samples()`` concatenates the chains in order (measured --
    the flat stack equals the grouped stack reshaped), so the flat tail is
    the last chain's tail.  That is the thinning the document asked for and
    not the pooled posterior; it is said here because the answer comes back
    finite, correctly shaped and silent about which chain it came from.

    :func:`~rheplicant.inference.push_forward` is neither route.  It is a
    ``jax.vmap`` over a SAMPLES pytree (uncertainty.py:577) and takes no
    covariance at all, so routing a fisher product to it would mean this layer
    inventing a Cholesky, an unflatten and a seed to manufacture draws the
    package never asked for; and on the samples side it is
    ``predict_from_samples`` that validates the stacks against ``space.names``
    before it vmaps.  Schema 4.7.9's ``predict`` row names the pair
    "push_forward / predict_from_samples" and mislabels it.

    The samples route needs numpyro; the covariance route does not.  Both
    imports therefore sit inside this body, which is also what keeps
    ``import rheplicant.config`` off ``rheplicant.inference``.

    Both routes re-derive their expansion point or their pipeline from THIS
    run's ``built``, so a ``predict`` on a different ``variant:`` from the
    run it reuses MIXES TWO BUILDS.  ``RunResult`` carries the variant its
    run was configured on, and the comparison below sits before the dispatch
    so it guards both routes at once.  The package catches only what moves
    the parameter LAYOUT (uncertainty.py:533, :544) and is silent about the
    rest: measured, a model-only mismatch returns a finite, correctly-shaped
    width off by 1.1 % -- a ratio of 0.98883 against the un-mixed answer on
    the SAME variant, 1.12 % at the worst channel -- because with one latent
    the whole error is the scalar sigma_g(base)/sigma_g(variant).  An error
    nobody would ever notice is the argument for refusing it.
    """
    where = f"runs[{run.name!r}]"
    if "from" in run.options:
        raise ConfigError(
            f"{where}: from: is schema 4.7.9's second spelling of the "
            "cross-run link, and reuse: is the one this layer reads -- "
            "runs[].reuse is already a member of the run grammar and from: "
            "is not. Rename from: to reuse:."
        )
    _sweep(run, _PREDICT_KEYS)
    earlier = reuse_of(run, results)
    if earlier.variant != run.variant:
        raise ConfigError(
            f"{where}: variant: {run.variant!r}, and reuse: "
            f"{run.reuse!r} ran on variant: {earlier.variant!r} -- pushing "
            "one build's product through another build's model MIXES TWO "
            "BUILDS. The numbers would come back finite, correctly shaped "
            "and plausible: measured, a model-only mismatch is 1.1 % wrong. "
            "Declare the same variant: on both runs."
        )
    space = _space(run, built)
    if earlier.kind == "fisher":
        from rheplicant.inference import propagate_covariance

        if "n_draw" in run.options:
            raise ConfigError(
                f"{where}: n_draw: thins an earlier run's draws, and "
                f"reuse: {run.reuse!r} names a kind: fisher run, whose "
                "product is a covariance -- the delta method draws nothing. "
                f"Drop n_draw:, or reuse one of {' / '.join(_DRAW_SOURCES)}."
            )
        forward, values = space.forward_fn(built.inference.fit_twin,
                                           built.state)
        return propagate_covariance(forward, values,
                                    earlier.product["covariance"])
    if earlier.kind in _DRAW_SOURCES:
        from rheplicant.inference import predict_from_samples

        draws = earlier.product
        available = draws.n_draw
        keep = available
        if "n_draw" in run.options:
            keep = _number(run, "n_draw", run.options["n_draw"], kind=int,
                           minimum=1)
            if keep > available:
                raise ConfigError(
                    f"{where}: n_draw: {keep} exceeds the {available} draws "
                    f"reuse: {run.reuse!r} kept -- "
                    f"{_DRAW_SOURCES[earlier.kind]}, so this is all there is."
                    f" Lower n_draw:, or make {run.reuse!r} draw more."
                )
        # The LAST draws, not the first: no source here still carries a
        # warmup -- two discarded theirs and npe never had one -- but the
        # leading kept draws are still the ones nearest the declared init,
        # and the tail is the part a thinning is meant to keep.
        samples = {name: stack[-keep:]
                   for name, stack in draws.samples.items()}
        return predict_from_samples(built.inference.fit_twin, built.state,
                                    space, samples)
    raise ConfigError(
        f"{where}: reuse: {run.reuse!r} names a kind: {earlier.kind} run, "
        "and predict pushes forward either a fisher run's covariance or the "
        f"draws of a {' / '.join(_DRAW_SOURCES)} run. Those are the "
        f"{len(_DRAW_SOURCES) + 1} products this exit knows how to propagate."
    )
