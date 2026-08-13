"""The conjugate family's shared machine: the block, and its prior keywords.

Four exits ride this module -- ``conjugate.wiener``, ``conjugate.gcr``,
``conjugate.gls`` and ``condition`` (schema §4.7.9).  Every one of them starts
by turning ``names:`` into a :class:`~rheplicant.inference.linear.LinearBlock`
and ``prior_std:``/``prior_mean:`` into the per-member mappings the grouped
solves take, so both live here rather than four times over.

Two rules this module exists to keep:

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

# Tasks 3-5 union these into each conjugate exit's own _sweep set.
_BLOCK_KEYS = frozenset({"names", "check"})
_PRIOR_KEYS = frozenset({"prior_std", "prior_mean"})
#: The two kinds that ALWAYS solve at a DECLARED sigma, so _conjugate_block
#: resolves it for them and check A27 fires there.  conjugate.gcr may take
#: GLSResult.noise_std instead (``noise_from: gls``) and conjugate.gls takes
#: the noise RULE through ``_decided_model``: resolving a decided sigma for
#: either would fire A27 on exactly the document that exit exists to serve.
_DECIDES_SIGMA_HERE = frozenset({"conjugate.wiener", "condition"})


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
