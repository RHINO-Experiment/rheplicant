"""What every conjugate exit opens with: the block, the prior, the knobs.

The conjugate family is four exits -- ``conjugate.wiener``, ``conjugate.gcr``,
``conjugate.gls`` and ``condition`` (schema §4.7.9) -- and every one of them
starts the same way: turn ``names:`` into a
:class:`~rheplicant.inference.linear.LinearBlock`, turn
``prior_std:``/``prior_mean:`` into the per-member mappings a grouped solve
takes, and coerce whichever solver knobs the document declared.  That shared
opening lives here; the executors that go on to do four different things with
it live in ``conjugate.py``.

The seam is the ``exit_support.py`` / ``exits.py`` one, one level down, and
for the same reason: ``conjugate.py`` was on its way past this repository's
800-line ceiling with two exits still to land.  **The import is one-way** --
``conjugate_support`` must not import ``conjugate`` -- so the split cannot
become a cycle, and neither name is re-exported from the other module,
because two importable spellings of one function is how the two drift apart.

Two rules this half exists to keep:

* **The block is always the GROUPED spelling.**  ``linear_operator`` takes
  ``name=`` OR ``names=`` and they are not interchangeable: the first returns
  a bare array, the second ``{latent: array}``, and six downstream consumers
  raise on the bare form (``linear.py:184-215``).  The config layer compiles
  to ``names=`` even for a block of one.
* **A grouped block's prior is per member.**  ``S`` is block-diagonal, not a
  multiple of the identity, so ``_per_member`` (``linear.py:963-973``) refuses
  a scalar outright rather than broadcasting it.  A scalar in a document is
  therefore broadcast HERE, and only when the block names exactly one latent
  -- check A51.

``check:`` belongs to ``linear_operator`` alone; none of the four solves takes
it.  It is passed only when the document declares it, so the package's own
default stands otherwise.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import (
    _decided_sigma,
    _number,
    _observed,
    _space,
)

# --- One block, so the executors have one place to compose from ------------

# The pieces each exit COMPOSES its own _sweep set from.  Not a wholesale
# union for all four: see _SOLVE_KEYS below for which exit takes which, and
# why `condition` takes only part of _PRIOR_KEYS.
_BLOCK_KEYS = frozenset({"names", "check"})
_PRIOR_KEYS = frozenset({"prior_std", "prior_mean"})
#: The two kinds that ALWAYS solve at a DECLARED sigma, so _conjugate_block
#: resolves it for them and check A27 fires there.  conjugate.gcr may take
#: GLSResult.noise_std instead (``noise_from: gls``) and conjugate.gls takes
#: the noise RULE through ``_decided_model``: resolving a decided sigma for
#: either would fire A27 on exactly the document that exit exists to serve.
_DECIDES_SIGMA_HERE = frozenset({"conjugate.wiener", "condition"})
#: key, coercion, floor, and whether the package spells "off" as None.
_SOLVER_KNOBS = (("tol", float, 0.0, False),
                 ("maxiter", int, 1, True),
                 ("require_convergence", float, 0.0, True))
#: The same three, as names: the CG knobs every conjugate SOLVE forwards.
#: Derived rather than retyped so the two can never drift.
_SOLVE_PASSTHROUGH = tuple(key for key, _cast, _floor, _null in _SOLVER_KNOBS)
#: What the three conjugate SOLVES take -- ``conjugate.wiener``,
#: ``conjugate.gcr`` and ``conjugate.gls``, each of which unions its own keys
#: onto this.
#:
#: NOT what every exit in the family takes.  ``condition`` is the fourth, and
#: measured against the venv its signature is
#: ``condition_estimate(block, *, noise_std, prior_std=None, iterations=12,
#: key=None)`` -- so of these seven it can use exactly three: ``names`` and
#: ``check`` (which build the block) and ``prior_std``.  It takes no
#: ``prior_mean`` and none of the three CG knobs, because it runs power
#: iteration rather than CG.  Task 6 therefore builds its own set from
#: ``_BLOCK_KEYS | {"prior_std", "iterations"}`` rather than unioning onto
#: this one, and it must NOT call :func:`_prior_kwargs`, which emits
#: ``prior_mean`` whenever the document declares it and would reach
#: ``condition_estimate`` as a TypeError.
_SOLVE_KEYS = _BLOCK_KEYS | _PRIOR_KEYS | frozenset(_SOLVE_PASSTHROUGH)


def _selected(run: Any, where: str) -> tuple[str, ...]:
    """``names:`` -> the latents this block groups, in the declared order."""
    names = run.options.get("names")
    if isinstance(names, str):
        return (names,)
    if (isinstance(names, list) and names
            and all(isinstance(one, str) for one in names)):
        return tuple(names)
    raise ConfigError(
        f"{where}: names: is required -- one latent name, or a list of "
        "them. The conjugate exits always build the GROUPED operator "
        "(linear_operator(names=)), whose solution comes back keyed by "
        f"latent, so which latents the block holds is not guessed; got "
        f"{names!r}."
    )


def _conjugate_block(run: Any, built: Any, where: str, *,
                     needs_observed: bool = True) -> tuple[Any, Any, Any]:
    """``(block, sigma, observed)`` -- everything a conjugate solve opens with.

    Three things come back together because no executor may hold one without
    having decided the others: the grouped ``LinearBlock``, the decided sigma
    (with check A27), and the observation.

    ``needs_observed`` says whether this exit's solve reads data:
    ``condition`` estimates kappa from the operator alone
    (``linear.py:1337`` takes no ``observed``), while the three solves do.
    Where it is True the missing-observation refusal fires BEFORE the
    operator is built, so a document with no ``inference.observed`` hears
    about the data it did not declare rather than about its latents; where it
    is False ``observed`` is None and ``_observed`` is never reached.

    ``sigma`` is :func:`_decided_sigma` for the kinds in
    :data:`_DECIDES_SIGMA_HERE` and None for the other two, which find their
    own -- see that constant's note.

    ``where`` is the ``exits.py`` spelling ``f"runs[{run.name!r}]"`` and is the
    THIRD POSITIONAL argument, never a keyword (plan section 3.1).
    """
    from rheplicant.inference import linear_operator

    space = _space(run, built)
    observed = _observed(run, built) if needs_observed else None
    sigma = (_decided_sigma(run, built)
             if run.kind in _DECIDES_SIGMA_HERE else None)
    knobs: dict[str, Any] = {}
    if "check" in run.options:
        check = run.options["check"]
        if not isinstance(check, bool):
            raise ConfigError(f"{where}: check: is a bool; got {check!r}.")
        knobs["check"] = check
    block = linear_operator(space, built.inference.fit_twin, built.state,
                            names=_selected(run, where), **knobs)
    return block, sigma, observed


def _one_prior(run: Any, where: str, key: str, value: Any, block: Any,
               space: Any) -> dict[str, Any]:
    """One of ``prior_std``/``prior_mean`` -> the per-member mapping."""
    minimum = 0.0 if key == "prior_std" else None
    if isinstance(value, Mapping):
        if set(value) != set(block.names):
            declared = [name for name in block.names
                        if space.latent(name).prior is not None]
            raise ConfigError(
                f"{where}: {key}: names {sorted(value)}, and this block "
                f"groups {list(block.names)}; S is block-diagonal, so a "
                "grouped block takes one entry per latent. Name every "
                f"member, or drop {key}: and let each latent's own prior: "
                f"drive the solve ({declared} declare one)."
            )
        return {name: _number(run, f"{key}.{name}", value[name], kind=float,
                              minimum=minimum)
                for name in block.names}
    number = _number(run, key, value, kind=float, minimum=minimum)
    if len(block.names) == 1:
        return {block.names[0]: number}
    raise ConfigError(
        f"{where}: {key}: {value!r} is one number for a block grouping "
        f"{list(block.names)}, and S is block-diagonal rather than a "
        "multiple of the identity: their widths differ by orders of "
        "magnitude and a block-diagonal S returns a finite, "
        "correctly-shaped, wrongly-regularised answer with no residual "
        f"signature (check A51). Write one entry per latent -- {key}: "
        f"{{{block.names[0]}: ...}} -- or drop the key and let each "
        "latent's own prior: drive the solve."
    )


def _prior_kwargs(run: Any, built: Any, block: Any,
                  where: str) -> dict[str, Any]:
    """The ``prior_std=``/``prior_mean=`` keywords the solve should take.

    Absent keys are absent from the result: the package then reads each
    latent's own ``Latent(prior=...)`` (``linear.py:928-931``), which is the
    standing decision that config never restates a package default.

    The ParameterSpace is derived HERE, from ``built``.  Callers pass
    ``built`` -- never a space, and never the ``where`` string, which binds
    silently and then breaks only inside the refusal branch that reads
    ``space.latent(name).prior`` (plan section 3.1).
    """
    space = _space(run, built)
    return {key: _one_prior(run, where, key, run.options[key], block, space)
            for key in ("prior_std", "prior_mean") if key in run.options}


def _knobs(run: Any, specs: tuple) -> dict[str, Any]:
    """The knobs among ``specs`` this document declared, coerced.

    A knob the document omits is omitted from the call, so the package's own
    default stands -- this plan's standing decision.  ``None`` passes through
    where the package spells "off" that way (``maxiter: null`` is no cap,
    ``require_convergence: null`` is no guard); everywhere else a non-number
    is a ConfigError here rather than a bare TypeError from inside a trace.

    Every conjugate exit calls this, with its own spec tuple, so that
    ``maxiter: "many"`` is refused identically whichever exit was asked for.
    """
    resolved: dict[str, Any] = {}
    for key, cast, floor, nullable in specs:
        if key not in run.options:
            continue                  # the package's own default stands
        value = run.options[key]
        if value is None and nullable:
            resolved[key] = None      # "no cap" / "no guard", as the package
            continue                  # spells them
        resolved[key] = _number(run, key, value, kind=cast, minimum=floor)
    return resolved
