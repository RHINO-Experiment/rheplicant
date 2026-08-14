"""What only ``kind: nuts`` and ``kind: npe`` share (schema §4.7.9, §4.7.10).

Three things, and each is here rather than in one of the two executors
because the other would otherwise write it again and the two copies would
drift:

* **The prior gate the two PACKAGE routes disagree about.**
  ``to_numpyro_model`` -> ``_require_priors`` accepts a latent with no
  ``prior:`` when the space's ``joint_prior`` covers it
  (``numpyro_bridge.py:61-79``, ``joint.covers(name)``); ``simulate_pairs``
  tests ``latent.prior is None`` alone and consults ``joint_prior`` not at
  all (``npe.py:111-118``).  So one document is a posterior on one route and
  a ``ParameterSpaceError`` naming no run on the other, and
  :func:`_sampled_space` is where that becomes a refusal in this layer's
  voice.
* **The unravel.**  ``NeuralPosterior.sample`` returns a flat
  ``(n_draws, n_params)`` array while 2C's shipped ``predict`` reads a
  samples product as a MAPPING (``diagnostics.py:763``).
  :func:`_unravel` is the inverse of the layout ``simulate_pairs`` documents
  at ``npe.py:100-102``, and getting its ORDER wrong returns finite,
  correctly-shaped, wrong draws.
* **The seed.**  ``draws.py``'s ``seed_for``/``_seed_name`` pair is the one
  place a ``{from: runtime.seeds.<name>}`` becomes a reportable integer, and
  :func:`_draw_key` is the one place this plan turns that integer into a key.
  The four hand-copied idioms already past ``draws.py:_key`` stay Plan 3's;
  2D adds no fifth and sixth.

**The import rule this module lives under.**  Nothing here imports
``numpyro`` or ``jax`` at module scope: ``exits.py``'s foot import reaches
``nuts.py``, which reaches this file, which is reached by
``import rheplicant.config`` -- and that import must leave ``numpyro`` out of
``sys.modules`` (measured after this commit: 0.295-0.327 s, ``False``).

**Ownership -- this file is COMPLETE and this is everything it binds.**
Written whole by Task 4, called by Tasks 4, 5, 6, 7 and 8.  No later task
appends to it and no task edits an existing body: :func:`_draw_key`'s
``spec=`` parameter is what lets ``npe``'s four named seeds reuse it, so no
per-call variant is ever needed, and there is no ``_bank_key``.  The
inventory is here rather than left to plan §3.1 because a drafter who reads
an authoritative list and does not find the name they need concludes the name
is free -- which is how ``npe.py`` came to have three names bound twice.

**Ten module-level names, in binding order.**  Six are imports --
``annotations`` (the ``__future__`` import), ``math``, ``Mapping``, ``Any``,
``ConfigError`` and ``_space`` -- and four are defined here: ``__all__``,
:func:`_sampled_space`, :func:`_unravel` and :func:`_draw_key`.  Nothing
else.  ``jax``, ``jax.numpy``, ``seed_for`` and ``_seed_name`` are bound
INSIDE function bodies and are deliberately not module-level -- see the
import rule above.

:func:`_unravel` takes an optional keyword-only ``where=`` that plan §3.1's
pinned signature does not show; the pinned two-argument call still binds, and
Task 8 should pass one.  See that function.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import _space

__all__: list[str] = []


def _sampled_space(run: Any, built: Any, *, route: str) -> Any:
    """The space, checked against what THIS route calls a prior.

    ``route`` is ``"nuts"`` or ``"npe"`` and is the whole difference.  Both
    legs refuse a latent that has no prior of any kind; only the ``npe`` leg
    refuses one the ``joint_prior`` covers, and its refusal says so, because
    a user looking at a document that a sibling exit runs is owed the reason
    rather than a package error naming no run.
    """
    space = _space(run, built)
    joint = space.joint_prior
    where = f"runs[{run.name!r}]"
    if route == "nuts":
        missing = [latent.name for latent in space.latents
                   if latent.prior is None
                   and not (joint is not None and joint.covers(latent.name))]
        if missing:
            raise ConfigError(
                f"{where}: kind: nuts draws a POSTERIOR, and "
                f"inference.parameters declares {missing} with no prior: and "
                "no inference.joint_prior covering them -- a prior-free "
                "latent is a free parameter, which the calibrator exits "
                "(kind: optimize, kind: plan.estimate) fit and a posterior "
                "cannot. Give each one a prior:, cover it with "
                "inference.joint_prior, or run one of those."
            )
        return space
    missing = [latent.name for latent in space.latents
               if latent.prior is None]
    if missing:
        covered = sorted(name for name in missing
                         if joint is not None and joint.covers(name))
        # BOTH clauses below are conditional on `covered`, and BOTH are
        # load-bearing.  The joint-prior branch of `instead` is ADVICE, and on
        # a document with no joint_prior at all it is FALSE advice: measured,
        # the nuts leg above refuses that same document with "declares ['g']
        # with no prior: and no inference.joint_prior covering them", so a
        # reader sent to the sibling exit is refused again for the same
        # reason.  A test that asserts a string is merely PRESENT locks that
        # in, so the branches are pinned by two tests that disagree with each
        # other: test_a_joint_prior_only_space_is_refused_for_npe reads the
        # opening clause of `because`, and
        # test_a_prior_free_space_gets_no_joint_prior_advice_from_npe asserts
        # that clause and the joint-prior branch of `instead` are BOTH ABSENT
        # while the no-coverage branch is present.  Making either clause
        # unconditional fails one of the two (measured, both directions).
        #
        # THE WORDING BELOW IS GREPPED, so two rules hold for anyone editing
        # it.  Task 8's Step 8.0 matches four fixed strings against this
        # function and expects four hits; one of them is the five words that
        # open the second fragment of `because`, and a grep finds them only
        # while they sit on ONE source line -- so that fragment is not
        # re-wrapped.  And no comment in this file repeats any of the four
        # verbatim, this one included: a comment hit holds the count at four
        # while the code being counted is gone, which is the failure the
        # tripwire exists to catch.
        because = (f"; inference.joint_prior covers {covered}, "
                   "which is why kind: nuts accepts this space and this "
                   "exit does not"
                   if covered else "")
        instead = (" -- or run kind: nuts, which takes joint-prior coverage"
                   if covered else
                   ". kind: nuts refuses this document too, for the same "
                   "missing prior")
        raise ConfigError(
            f"{where}: kind: npe SIMULATES a bank from each latent's OWN "
            f"prior, and inference.parameters declares {missing} with no "
            "prior: (npe.py:111-118 reads the latent alone and consults "
            f"inference.joint_prior not at all{because}). Declare "
            f"inference.parameters.<name>.prior:{instead}."
        )
    return space


def _unravel(space: Any, flat: Any, *, where: str | None = None
             ) -> dict[str, Any]:
    """``(n_draws, n_params)`` -> ``{latent name: (n_draws, *shape)}``.

    Ordered by ``space.names`` -- DECLARATION order, not sorted -- and sized
    by each latent's own ``initial_values()`` shape.  Both of those are
    silent when wrong: sorting the names hands every latent another latent's
    column, and assuming a scalar hands the first latent a slice of the
    second, and in both cases the draws that come back are finite and
    correctly shaped.

    ``where`` is an ADDITION to the signature plan §3.1 pins, and it is
    optional so that the pinned two-argument call still binds.  Task 8 has a
    run name to hand and should pass it: without one this is the only refusal
    in either new module that names no run, which in a layer where every
    other refusal opens with ``runs['<name>']:`` reads as a package error
    rather than as this layer's.

    The width is checked BEFORE the loop, and that ordering is the whole
    difference between the two failure directions.  With the check after it,
    a too-WIDE flat got this refusal while a too-NARROW one reached jax as
    ``TypeError: cannot reshape array of shape (2, 0) into shape (2,)``
    (measured) -- and narrow is the likelier bug, because it is what an
    estimator sized from a stale space produces.
    """
    import jax.numpy as jnp

    initial = space.initial_values()
    shapes = {name: tuple(int(size) for size in jnp.shape(initial[name]))
              for name in space.names}
    widths = {name: math.prod(shape) for name, shape in shapes.items()}
    n_draws, width = (int(size) for size in jnp.shape(flat))
    accounted = sum(widths.values())
    if accounted != width:
        prefix = "The draws" if where is None else f"{where}: the draws"
        raise ConfigError(
            f"{prefix} are {width} wide and inference.parameters accounts "
            f"for {accounted}: {shapes}. A flat draw array carries one "
            "column per latent VALUE, concatenated in space.names order "
            "(npe.py:100-102), so the space that laid the columns out is the "
            "one that has to unravel them -- pass the space the bank was "
            "simulated from, not one rebuilt since."
        )
    unravelled: dict[str, Any] = {}
    cut = 0
    for name in space.names:
        unravelled[name] = jnp.reshape(flat[:, cut:cut + widths[name]],
                                       (n_draws, *shapes[name]))
        cut += widths[name]
    return unravelled


def _draw_key(run: Any, where: str, built: Any,
              spec: Mapping | None = None) -> Any:
    """A jax PRNG key from a ``{from: runtime.seeds.<name>}`` declaration.

    ``spec`` defaults to ``dict(run.options)`` -- the run-level form
    ``kind: nuts`` uses.  ``kind: npe`` passes one of its own
    ``inference.npe:`` subsections instead, because it needs four
    independent named seeds and a run carries one.

    ``where`` is the prefix ``draws._seed_name``'s refusals wear, so a
    missing or literal seed is refused as ``runs['<name>']:`` from a run and
    as ``inference.npe.<sub>:`` from the section.
    """
    import jax

    from rheplicant.config.draws import _seed_name, seed_for

    declared = dict(run.options) if spec is None else dict(spec)
    return jax.random.key(seed_for(_seed_name(declared, where),
                                   built.context))
