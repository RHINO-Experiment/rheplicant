"""``kind: nuts``: numpyro's NUTS over ``to_numpyro_model`` (schema §4.7.9).

The first exit in this layer whose package entry point needs ``numpyro`` at
all, and therefore the module most able to break the invariant the whole
config layer is measured on::

    import rheplicant.config          # 0.30 s, and numpyro NOT in sys.modules

``exits.py``'s foot import reaches this module from Task 5 onward, so **every
numpyro import here sits inside a function body** -- the shipped pattern is
``predict``'s samples route (``config/sections/diagnostics.py:771`` -- the
path is qualified because this repository has TWO ``diagnostics.py`` and the
sibling, ``inference/diagnostics.py``, is the one every other citation in
this module resolves to).  A module-level
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

*Two more bound by Task 6*, plus the ``warnings`` and ``_passthrough``
imports it added: :data:`_MCMC_KEYS` (a TUPLE, and ``target_accept_prob`` is
NOT in it -- that knob is ``NUTS``'s, and a table carrying it would
``TypeError`` on the first document that declared it) and
:data:`_NUTS_KERNEL_KEYS`.  Task 4 reserved exactly those two names here,
because a name absent from an authoritative list reads as free -- which is
the failure this whole block exists to prevent, and which an earlier draft of
the block committed by reserving ``_MCMC_KEYS`` alone.  **That is the whole
inventory: this module is written by Tasks 4, 5 and 6 and by nobody after
them.**  The three ``chain_method`` words numpyro takes are a LOCAL inside
:func:`_run_nuts` rather than a third module constant, so the reservation
above stays exhaustive.

Tasks 5 and 6 GROW :data:`_NUTS_KEYS` and :class:`NutsProduct` rather than
rebinding them, and Task 5 added the ``@register("nuts")`` decorator to
:func:`_run_nuts`.
"""

from __future__ import annotations

import warnings
from typing import Any, NamedTuple

from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import (
    _noise,
    _number,
    _observed,
    _passthrough,
    _sweep,
    register,
)
from rheplicant.config.sections.posterior_support import (
    _draw_key,
    _sampled_space,
)

__all__ = ["NutsProduct"]

#: EVERY key ``kind: nuts`` sweeps.  ONE frozenset, grown to this shape by
#: Tasks 4, 5 and 6 rather than three unioned at call time.
_NUTS_KEYS = frozenset({
    "num_warmup", "num_samples", "seed",
    "init",
    "num_chains", "chain_method", "thinning",
    "target_accept_prob", "progress_bar",
})

#: The knobs that ride on ``MCMC(...)``.  ``target_accept_prob`` is NOT here
#: -- it is ``NUTS``'s (plan §2.4) -- and a table that carried it would
#: ``TypeError: MCMC.__init__() got an unexpected keyword argument
#: 'target_accept_prob'`` on the first document that declared it.
_MCMC_KEYS = ("num_chains", "chain_method", "thinning", "progress_bar")

#: The knobs that ride on ``NUTS(...)``.  ``init_strategy`` does NOT travel
#: through this table, because ``init:`` is a WORD in the document and a
#: callable at the seam -- :func:`_init_strategy` is that translation, and it
#: lands BESIDE this passthrough on the kernel line, not around it.
_NUTS_KERNEL_KEYS = ("target_accept_prob",)

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
    draw axis), ``config/sections/diagnostics.py:774`` and ``:791`` -- NOT
    ``inference/diagnostics.py``, which contains no ``n_draw`` at all.

    ``samples`` carries ``space.names`` AND NOTHING ELSE -- in particular not
    the deterministic ``"prediction"`` site.

    ``diagnostics`` is ``{latent: {"r_hat": float, "n_eff": float}}`` and
    ``divergences`` the number of divergent transitions across every chain.
    A diverging chain returns finite, plausible, WRONG draws, so the count is
    carried here and warned about, never silently dropped.  It is NOT a
    refusal: the number at which it becomes fatal is a judgement this layer
    has no basis for (measured on one document at four seeds under a sloppy
    target: 200, 101, 77 and 52 out of 200), and ``expect: refuse`` discards
    the product (``exits.py:296``), which would make the count unreachable
    exactly when someone wanted to ask how bad it was.
    """

    samples: dict[str, Any]
    n_draw: int
    n_chain: int
    diagnostics: dict[str, Any]
    divergences: int


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
    from numpyro.diagnostics import summary

    from rheplicant.inference import to_numpyro_model

    _sweep(run, _NUTS_KEYS)
    where = f"runs[{run.name!r}]"
    # `_sweep` checks key NAMES.  Nothing checked VALUES, and every knob here
    # reached the package raw: `num_chains: 0` as `IndexError: tuple index out
    # of range`, `thinning: 1.5` as numpyro's own bare ValueError,
    # `target_accept_prob: "hi"` as a TypeError about a DynamicJaxprTracer --
    # and `target_accept_prob: 2.0` did not raise at all.  Meanwhile
    # `num_samples: 2.5` was already refused by `_number` a few lines below,
    # so half this exit's counts were guarded and half were not.
    for key in ("num_chains", "thinning"):
        if key in run.options:
            _number(run, key, run.options[key], kind=int, minimum=1)
    if "target_accept_prob" in run.options:
        # minimum=0.0 and NO upper bound: 1.0 as a ceiling would be this layer
        # restating numpyro's parametrisation, and 2.0 measurably just runs.
        _number(run, "target_accept_prob", run.options["target_accept_prob"],
                kind=float, minimum=0.0)
    if "progress_bar" in run.options and not isinstance(
            run.options["progress_bar"], bool):
        # The fifth knob of this commit, and the one `_number` cannot check.
        # Every non-empty string is truthy, so `progress_bar: "false"` printed
        # the bar the document asked to be rid of and said nothing (measured).
        raise ConfigError(
            f"{where}: progress_bar: is true or false; got "
            f"{run.options['progress_bar']!r}. Every non-empty string is "
            'truthy, so progress_bar: "false" would print the very bar the '
            "document asked to be rid of, and say nothing about it."
        )
    # numpyro's own three words, not this layer's invention and not a default
    # restated: the layer forwards whatever is declared and refuses only what
    # the package would refuse anyway, in its own voice instead of
    # `ValueError: Only supporting the following methods`, which names no run
    # and no document key.  A LOCAL rather than a module constant -- the
    # module docstring's reserved-name list is exhaustive and stays so.
    #
    # "Declared" here MUST mean PRESENT, the way `_passthrough` means it.  An
    # earlier form of this guard read `.get("chain_method")` and tested
    # `is not None`, which disagrees with the forwarder on exactly one input:
    # `chain_method: null` slipped through the gap and reached `MCMC(...)` as
    # the very ValueError quoted above.  Measured on that form, `num_chains`,
    # `thinning`, `target_accept_prob` and `init: null` all refused by name
    # and `chain_method` alone leaked -- a hole closed on one route and left
    # open on its twin, by the commit that closed it for the other four.
    methods = ("parallel", "sequential", "vectorized")
    if ("chain_method" in run.options
            and run.options["chain_method"] not in methods):
        raise ConfigError(
            f"{where}: chain_method: {run.options['chain_method']!r} is not "
            f"one of numpyro's {', '.join(methods)}. parallel needs one "
            "device per chain and falls back to sequential with a warning "
            "when it does not have them; vectorized runs them under one vmap."
        )
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
        model, init_strategy=_init_strategy(run, built, space),
        **_passthrough(run.options, _NUTS_KERNEL_KEYS))
    mcmc = numpyro.infer.MCMC(kernel, **counts,
                              **_passthrough(run.options, _MCMC_KEYS))
    # `observed=` by keyword.  The model is a one-parameter closure, so the
    # positional form binds identically today; the keyword is what survives a
    # to_numpyro_model that grows a second parameter.
    mcmc.run(_draw_key(run, where, built), observed=observed)
    drawn = mcmc.get_samples()
    samples = {name: drawn[name] for name in space.names}
    # `summary` needs the chain axis -- (1, 200) with one chain, (2, 200) with
    # two -- and it is restricted to the latents for the same reason `samples`
    # is: run over everything `get_samples` returns it would summarise the
    # deterministic "prediction" site too, 200 x 16 x 8 of it, and hand back a
    # row under a key no latent owns.  `r_hat` and `n_eff` are two of the
    # seven keys it returns, and both arrive as numpy scalars, which is why
    # `float(...)` sits on each below rather than after someone finds a numpy
    # scalar in a message.  No `prob=` argument: the plan's snippet passed
    # `prob=0.9`, which IS the package default and controls only the
    # '5.0%'/'95.0%' rows the comprehension below discards -- restating a
    # default two lines above the comment refusing to restate one.
    grouped = mcmc.get_samples(group_by_chain=True)
    table = summary({name: grouped[name] for name in space.names})
    # `diverging` is in get_extra_fields() with NO extra_fields= argument
    # (measured); passing one would restate a package default.
    diverging = mcmc.get_extra_fields()["diverging"]
    divergences = int(diverging.sum())
    if divergences:
        warnings.warn(
            f"{where}: kind: nuts recorded {divergences} divergent "
            f"transition(s) out of {int(diverging.size)}. A diverging chain "
            "returns finite, plausible, WRONG draws; the count is on the "
            "product as .divergences and is not a refusal, because the "
            "number at which it becomes fatal is a judgement this layer has "
            "no basis for.",
            UserWarning, stacklevel=2,
        )
    return NutsProduct(
        samples=samples,
        n_draw=int(samples[space.names[0]].shape[0]),
        n_chain=int(mcmc.num_chains),
        diagnostics={name: {"r_hat": float(row["r_hat"]),
                            "n_eff": float(row["n_eff"])}
                     for name, row in table.items()},
        divergences=divergences,
    )
