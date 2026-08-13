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

Tasks 8, 10 and 11 add ``gradient``, ``mmodes`` and ``predict`` to this
module.  ``condition`` is NOT here: it rides ``_conjugate_block``, so Task 6
put it in ``conjugate.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import (
    _number,
    _space,
    _sweep,
    register,
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

    Task 8's ``gradient`` calls this too -- one helper, three callers, three
    positional arguments (plan section 3.1).  Task 8 writes no second
    ``at:`` resolver.
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
