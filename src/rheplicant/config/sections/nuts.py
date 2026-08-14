"""``kind: nuts``: numpyro's NUTS over ``to_numpyro_model`` (schema §4.7.9).

The first exit in this layer whose package entry point needs ``numpyro`` at
all, and therefore the module most able to break the invariant the whole
config layer is measured on::

    import rheplicant.config          # 0.30 s, and numpyro NOT in sys.modules

``exits.py``'s foot import reaches this module from Task 5 onward, so **every
numpyro import here sits inside a function body** -- the shipped pattern is
``predict``'s samples route (``diagnostics.py:745``).  A module-level
``import numpyro`` would fail nothing: the import would simply cost ~2 s and
drag numpyro into every process that reads a config.

Three facts about the seam, each of which the obvious code gets wrong while
still compiling:

* ``to_numpyro_model(...)`` returns a **callable** ``model(observed=None)``
  (``numpyro_bridge.py:241``) with exactly one parameter, and
  ``MCMC.run(self, rng_key, *args, **kwargs)`` forwards both straight to it --
  so ``mcmc.run(key, data)`` and ``mcmc.run(key, observed=data)`` bind the
  same argument.  The drive line below uses **the keyword** because it is
  the form that still binds correctly the day ``to_numpyro_model`` grows a
  second parameter, not because the positional form is broken today.
* ``init_strategy`` is **``NUTS``'s**, not ``MCMC``'s.  ``MCMC`` has no such
  parameter (measured: ``TypeError: MCMC.__init__() got an unexpected keyword
  argument 'init_strategy'``), and passing it to neither object is the silent
  failure: ``init_to_declared``'s own docstring measures ``r_hat = 840`` and
  ``n_eff = 2`` from numpyro's default against ``r_hat = 1.002`` and
  ``n_eff = 1327`` from the identical model started at the declaration.
* ``noise_std`` takes the NoiseModel **whole**.  Unlike the conjugate
  family, which needs a DECIDED array, this route wants the rule: a
  prediction-dependent sigma brings its log-determinant with it, because
  ``Normal(loc, scale).log_prob`` carries ``-log scale``
  (``numpyro_bridge.py:188``, the module's own Note).  So this executor calls
  ``_noise`` and must never call ``_decided_sigma``.

**The memory trap.**  ``mcmc.get_samples()`` also returns the deterministic
site ``"prediction"``, whose per-sample shape is the whole TOD -- measured on
this layer's own one-latent document, ``g (200,)`` against
``prediction (200, 16, 8)``, 128 times the latent's footprint on a toy.
:class:`NutsProduct` carries ``space.names`` and nothing else.

**Ownership -- this file is written by THREE tasks, so here is who binds
what.**  Plan §3.1 pins the shared shapes; this block is the full inventory,
because a name that section does not list reads as free and that is how three
names came to be bound twice in ``npe.py``.

*Fifteen module-level names bound by Task 4, in binding order.*  Ten are
imports -- ``annotations`` (the ``__future__`` import), ``Any``,
``NamedTuple``, ``ConfigError``, ``_noise``, ``_number``, ``_observed``,
``_sweep``, ``_draw_key``, ``_sampled_space`` -- and five are defined here:
``__all__``, :data:`_NUTS_KEYS`, ``_COUNTS``, :class:`NutsProduct` and
:func:`_run_nuts`.

*Two more bound by Task 5*, plus the ``register`` import it added to the
``exit_support`` list: ``_INITS`` (a tuple) and :func:`_init_strategy`.

*Reserved, and bound by NO task before its own.*  **Task 6:** ``_MCMC_KEYS``
(a TUPLE, and ``target_accept_prob`` is NOT in it -- that knob is ``NUTS``'s,
and a table carrying it would ``TypeError`` on the first document that
declared it) and ``_NUTS_KERNEL_KEYS``.  Those two are listed because a name
absent from an authoritative list reads as free -- which is the failure this
whole block exists to prevent, and which an earlier draft of the block
committed by reserving ``_MCMC_KEYS`` alone.

Tasks 5 and 6 GROW :data:`_NUTS_KEYS` and :class:`NutsProduct` rather than
rebinding them, and Task 5 added the ``@register("nuts")`` decorator to
:func:`_run_nuts`.
"""

from __future__ import annotations

from typing import Any, NamedTuple

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import (
    _noise,
    _number,
    _observed,
    _sweep,
    register,
)
from rheplicant.config.sections.posterior_support import (
    _draw_key,
    _sampled_space,
)

__all__ = ["NutsProduct"]

#: EVERY key ``kind: nuts`` sweeps.  Task 6 adds the last five.  ONE
#: frozenset, grown by that task.
_NUTS_KEYS = frozenset({"num_warmup", "num_samples", "seed", "init"})

_COUNTS = ("num_warmup", "num_samples")
#: What ``init:`` may say.  ``declared`` is the DEFAULT, and it is this
#: layer's own choice rather than a restatement of numpyro's -- see
#: :func:`_init_strategy`.
_INITS = ("declared", "ref")


class NutsProduct(NamedTuple):
    """What a ``kind: nuts`` run returns.

    ``samples`` and ``n_draw`` are not free choices: 2C's shipped ``predict``
    reads a samples product as ``product.n_draw`` (an int) and
    ``product.samples`` (a mapping of latent name -> stack with a leading
    draw axis), ``diagnostics.py:748`` and ``:763``.

    ``samples`` carries ``space.names`` AND NOTHING ELSE -- in particular not
    the deterministic ``"prediction"`` site.

    **Task 6 appends ``diagnostics`` and ``divergences``**, in §3.1's order.
    The three fields here are the first three of the five that section pins.
    """

    samples: dict[str, Any]
    n_draw: int
    n_chain: int


def _init_strategy(run: Any, built: Any, space: Any) -> Any:
    """``init:`` -> the ``init_strategy=`` the KERNEL takes.

    ``declared`` is this layer's default and is a deliberate override of
    numpyro's, not a restatement of it: numpyro's default is
    ``init_to_uniform``, which draws in the unconstrained space with no
    knowledge of the declaration, and on ``examples/tutorial_nuts.py``'s ring
    toy that is ``r_hat = 840`` and ``n_eff = 2`` against ``r_hat = 1.002``
    and ``n_eff = 1327`` from the identical model started at the declaration
    (``numpyro_bridge.py:296-310``).  The schema says ``init=declared`` for
    the same reason.

    ``ref`` starts at each latent's ``ref:`` instead -- the first consumer
    ``ParsedLatent.ref`` has had since it was parsed in Plan 2B.  A latent
    with no ``ref:`` is refused **by name**: falling back to its ``init:``
    would run a document that asked for one start from another, which is the
    invisible-wrong shape this whole effort is written against.

    ``ParsedLatent.ref`` reaches no ``ParameterSpace`` -- ``build_space``
    keeps ``entry.latent`` and drops the rest (``transforms.py:402``) -- so
    this reads :attr:`~rheplicant.config.sections.inference.InferenceBuild.refs`,
    populated by ``build_inference`` where the latents are ALREADY parsed.
    It does not re-parse the section: a second ``parse_latents`` per run is a
    second validator that can disagree with the first the day either grows a
    context-dependent branch, and that shape is already on Plan 3's ledger
    once.
    """
    where = f"runs[{run.name!r}]"
    asked = run.options.get("init", "declared")
    if asked not in _INITS:
        raise ConfigError(
            f"{where}: init: is one of {list(_INITS)} -- where the chain "
            f"STARTS, not how it moves; got {asked!r}."
        )
    if asked == "declared":
        from rheplicant.inference import init_to_declared

        return init_to_declared(space)
    import numpyro

    refs = built.inference.refs or {}
    without = [name for name in space.names if refs.get(name) is None]
    if without:
        raise ConfigError(
            f"{where}: init: ref starts the chain at each latent's own ref:, "
            f"and inference.parameters declares no ref: for {without}. "
            "Declare one for each, or say init: declared to start at init: "
            "-- starting the named ones at ref: and the rest at init: would "
            "be a third starting point no document asked for."
        )
    return numpyro.infer.init_to_value(
        values={name: refs[name] for name in space.names})


@register("nuts")
def _run_nuts(run: Any, built: Any, *, results: Any = None) -> Any:
    """One ``kind: nuts`` run -> a :class:`NutsProduct`."""
    import numpyro

    from rheplicant.inference import to_numpyro_model

    _sweep(run, _NUTS_KEYS)
    where = f"runs[{run.name!r}]"
    for key in _COUNTS:
        if key not in run.options:
            raise ConfigError(
                f"{where}: {key}: is required -- numpyro's MCMC declares "
                "num_warmup and num_samples keyword-only with NO default, so "
                "there is no package default for this layer to stand aside "
                "for, and a chain length invented here would be a number "
                "nobody wrote down."
            )
    counts = {key: _number(run, key, run.options[key], kind=int, minimum=1)
              for key in _COUNTS}
    space = _sampled_space(run, built, route="nuts")
    observed = _observed(run, built)
    # _noise, never _decided_sigma: see the module docstring.
    model = to_numpyro_model(built.inference.fit_twin, built.state, space,
                             _noise(run, built))
    # init_strategy on the KERNEL, and never on MCMC: passing it to neither
    # object is the difference between a posterior and noise, not a tuning
    # knob.  Which strategy the document asked for is _init_strategy's.
    kernel = numpyro.infer.NUTS(
        model, init_strategy=_init_strategy(run, built, space))
    mcmc = numpyro.infer.MCMC(kernel, **counts)
    # `observed=` by keyword.  The model is a one-parameter closure, so the
    # positional form binds identically today; the keyword is what survives a
    # to_numpyro_model that grows a second parameter.
    mcmc.run(_draw_key(run, where, built), observed=observed)
    drawn = mcmc.get_samples()
    samples = {name: drawn[name] for name in space.names}
    return NutsProduct(samples=samples,
                       n_draw=int(samples[space.names[0]].shape[0]),
                       n_chain=int(mcmc.num_chains))
