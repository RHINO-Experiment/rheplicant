"""The documents ``kind: nuts`` and ``kind: npe`` are measured on.

Split out of :mod:`tests.config.exit_helpers` by Task 7 (plan §2.5).  That
module was **787 lines** against this repository's 800-line ceiling when the
split was made, with Task 8's three further builders still to come and **no
automated guard** on file length -- so passing the ceiling would have been
silent.  What moved is exactly what this plan added for the two new kinds;
nothing that predates it moved, and nothing was added to ``exit_helpers`` in
its place.

**The import goes ONE WAY: this module imports ``exit_helpers``, and
``exit_helpers`` imports nothing back.**  The obvious alternative -- re-export
these names from ``exit_helpers`` so every caller keeps one import path --
does not work, and it is written down here because a reader will otherwise
propose it again.  This module needs several names *from* ``exit_helpers``, so
a re-export at that module's foot closes an import cycle: it survives
``import exit_helpers`` (the foot import runs after every name it needs is
bound) and dies on ``import posterior_helpers``, which pytest does the moment
one test module reaches for a posterior document directly --
``ImportError: cannot import name 'NEEDLE' from partially initialized module``.
A contract that holds only under one import order is not a contract.

So the 2C rule "ONE place builds a test document" is now spelled as **one
package, two modules**: the conjugate, diagnostic and fan documents stay in
``exit_helpers``, the posterior documents live here, and a test module imports
each name from the module that defines it.  Task 10's fixture-contract guard
walks BOTH -- a ``value.__module__ == exit_helpers.__name__`` filter would
silently drop every builder that moved and stay green while covering less.
"""

from rheplicant.config.document import load_document
from rheplicant.config.sections.runs import RunSpec
from tests.config.exit_helpers import (
    HOMOSCEDASTIC,
    MODEL_NOISE,
    ONE_LATENT,
    TRUTH_A,
    TRUTH_D,
    conjugate_document,
    run_product,
)

# --- What kind: nuts is measured on -----------------------------------------
#
# ``name: "chain"``, deliberately NOT the kind -- the same reason ``GLS`` in
# :mod:`tests.config.exit_helpers` gives: ``runs.py`` names an unnamed run
# after its kind, so a run named for its kind cannot tell a ``where`` built
# from ``run.name`` from one built out of ``run.kind``, and every refusal
# below reads that prefix.
#
# 200 warmup + 200 samples on the one-latent document is 1.3-2.7 s on the first
# call in a process and 0.3-0.6 s on the second (measured over several runs; a
# single pair of numbers does not reproduce, and the first call carries the
# numpyro import and the trace), and it is a CHAIN LENGTH rather than a
# statistical recommendation: it is the pair the plan's budget section pins,
# and this document's r_hat under it is 0.99527 with n_eff 80.2 (measured
# through ``numpyro.diagnostics.summary`` on
# ``get_samples(group_by_chain=True)``).  ``seeds={"chain": 3}`` is what the
# run's ``seed: {from: runtime.seeds.chain}`` resolves against; every number
# pinned in tests/config/test_config_exits_nuts.py was measured under it.
#
# ``progress_bar: false`` was a DEBT this constant carried from Task 4 to
# Task 6, and Task 6 has now repaid it.  It could not go in at Task 4:
# ``_NUTS_KEYS`` held three names there and ``_sweep`` refuses every option
# outside the table, so declaring it refused every run this constant drives --
# measured, 11 of the 12 tests in tests/config/test_config_exits_nuts.py
# failed, the 12th passing only because it is the one asserting that the sweep
# refuses an unknown key.  What its absence cost was real: numpyro's own
# default is True, so every chain test wrote a tqdm bar to captured stderr,
# and -- worse -- with NO document in the suite declaring the key,
# ``progress_bar`` could be deleted from both ``_NUTS_KEYS`` and
# ``_MCMC_KEYS`` and every test stayed green.  It is declared here now, in the
# commit that added the key to those two tables, which closes the SWEEP leg;
# test_config_exits_nuts.py's MCMC spy closes the FORWARD leg, and the same
# module's ``product({"drop": ("progress_bar",)})`` is how the silent document
# is still reachable.
NUTS = {"name": "chain", "kind": "nuts", "num_warmup": 200,
        "num_samples": 200, "seed": {"from": "runtime.seeds.chain"},
        "progress_bar": False}


def nuts_document(run=None, **kwargs):
    """:func:`conjugate_document` with a ``kind: nuts`` run merged over NUTS.

    ``run`` is merged over :data:`NUTS`, so a caller adds ``num_chains:`` or
    ``init:`` without restating ``seed:``; every other keyword goes to
    :func:`conjugate_document` unchanged.

    **It took a ``drop`` key inside ``run`` and no longer does.**  That branch
    took keys AWAY, for a document that must be missing a required one, and it
    never acquired a caller: Task 5 wrote it expecting Task 6's
    ``run_document`` legs to use it and they did not -- Task 6's silent
    document goes through :func:`nuts_spec`'s own ``drop=``, which builds the
    ``RunSpec`` the executor actually reads -- Task 7 kept it because its
    split's entire verification was an AST comparison asserting that ZERO
    bodies changed, and Task 9 declined it as outside its Files list.
    Measured once more before removing it: outside this module
    ``nuts_document`` is called exactly twice, at
    ``test_config_exits_nuts.py`` :355 and :360, and inside it twice, by
    :func:`nuts_built` and :func:`nuts_product`; no other file in ``src`` or
    ``tests`` names it at all, and not one of the four calls passes ``drop``.
    A parameter no test exercises is a parameter a later edit can break with
    every test still green, which is why :func:`npe_spec` lost its own unused
    ``**options`` at Task 8 and why this went the same way.  What replaces
    it: :func:`nuts_spec` for a refusal read straight off the executor, and
    ``del document["runs"][0][key]`` for one read through ``run_document`` --
    the idiom ``test_config_exits_gcr.py:215`` already uses for exactly this.
    """
    merged = {**NUTS, **(run or {})}
    kwargs.setdefault("seeds", {"chain": 3})
    return conjugate_document(merged, **kwargs)


def nuts_built(run=None, **kwargs):
    """:func:`nuts_document`, BUILT -- the ``built`` the executor receives."""
    return load_document(nuts_document(run, **kwargs))


def nuts_spec(drop=(), **options):
    """A ``kind: nuts`` RunSpec named ``chain``, straight to the executor.

    Task 4 drove ``_run_nuts`` through this rather than through
    ``run_document``, because ``parse_runs`` refused ``kind: nuts`` until
    Task 5 moved it out of ``_KINDS_2D``.  It stays for the reason its
    callers actually use: it returns a ``RunSpec``, so a test can call the
    executor directly and read what it raised or returned, without
    ``run_document``'s loop turning a refusal into a ``RunResult.error``.

    **Its ``drop=`` is now the only helper that takes a required key away.**
    An earlier draft of this docstring claimed it was the only way to reach a
    missing required key and was corrected to point at :func:`nuts_document`'s
    own ``drop``; Task 10 then removed that branch as never once called, so
    the correction outlived the thing it corrected to.  The route it named is
    still open and needs no parameter: a test that wants the DOCUMENT to be
    missing a key builds one and deletes the key, the way
    ``test_config_exits_gcr.py:215`` does.
    """
    body = {key: value for key, value in NUTS.items()
            if key not in ("name", "kind") and key not in drop}
    body.update(options)
    return RunSpec(name="chain", kind="nuts", variant=None, on="primary",
                   expect="ok", options=body)


def nuts_product(run=None, **kwargs):
    """:func:`nuts_document`, EXECUTED, and its NutsProduct.

    The ``run_document`` twin of :func:`nuts_spec`, for a test that wants the
    route a user takes rather than the executor called directly.
    """
    return run_product(nuts_document(run, **kwargs), "chain")


# --- What ``init:`` is measured on ------------------------------------------
#
#: The needle: a Gaussian's CENTRE, where the likelihood is flat everywhere
#: the line is not.  ``ref: 60 MHz`` is 18 MHz from the declared start and
#: 20 MHz from the truth, at the very bottom of this document's 60-85 MHz
#: band, far enough down the shoulder that a chain started there never finds
#: the line -- which is what makes ``init:`` OBSERVABLE.  On the one-latent
#: gain document it is not: measured here, the declared start gives g mean
#: 1.500021 / r_hat 0.99527 / n_eff 80.2 and numpyro's own ``init_to_uniform``
#: gives 1.499984 / 0.99922 / 106.4.  Those agree to five significant figures
#: and numpyro's own is marginally the better-mixed of the two, so no
#: assertion on that document can tell the two starts apart -- which is the
#: whole reason this one exists.
#:
#: Both the init and the ref carry ``unit: MHz``, and that is not decoration.
#: ``global_signal.centre`` is canonicalised to Hz, so a bare ``78.0`` is 78
#: Hz -- measured, that is what ``space.initial_values()`` then holds -- which
#: starts the chain sixty MHz below the bottom of the band and leaves it
#: there: measured on this document with the init written bare and everything
#: else unchanged, c comes back mean 2.93e7 std 1.60e7, against 8.00e7 and
#: 1.11e5 from the unit-carrying one.  A document that proves nothing looks
#: exactly like one that does.
NEEDLE_CENTRE = {"init": {"value": 78.0, "unit": "MHz"},
                 "into": "global_signal.centre",
                 "ref": {"value": 60.0, "unit": "MHz"},
                 "prior": {"normal": {"loc": {"value": 78.0, "unit": "MHz"},
                                      "scale": {"value": 30.0,
                                                "unit": "MHz"}}}}
NEEDLE = {"parameters": {"c": NEEDLE_CENTRE}, "noise": HOMOSCEDASTIC,
          "observed": {"from": "simulation",
                       "at": {"c": {"value": 80.0, "unit": "MHz"}}}}


# --- What kind: npe is measured on ------------------------------------------
#
# ``name: "amortized"``, deliberately NOT the kind -- the same reason
# :data:`NUTS` above gives: ``runs.py`` names an unnamed run after its kind, so
# a run named for its kind cannot tell a ``where`` built from ``run.name`` from
# one built out of ``run.kind``, and every refusal below reads that prefix.
#
# FOUR named seeds, because ``inference.npe:`` needs four and a run carries
# one: the bank draws theta from the priors, ``create`` initialises the
# network's weights, ``train`` shuffles the minibatches, and ``sample`` draws.
# Check A29 is what makes each of them required.
NPE_SEEDS = {"bank": 11, "create": 12, "train": 13, "sample": 14}

#: The sizes are this plan's own measured floor and none of them is a
#: statistical recommendation: 64 simulations is ~1.0 s on the 16 x 8 grid and
#: ``create`` + 50 training steps at width 16 / depth 2 is ~0.5 s more
#: (measured).  Raising one without measuring the cost is what the plan's §0.1
#: forbids.
#:
#: ``n_components: 1`` is DECLARED and not defaulted.  The package's default is
#: 4 and its own tuning table says 4 over-fits -- both
#: ``tests/inference/test_npe.py:145`` and the shipped example pass 1 -- and
#: the config layer restates no package default, so a document that wants 1
#: says 1.  ``min_scale`` is left unwritten precisely so that one key in this
#: section demonstrates the package's own default arriving untouched, and
#: ``test_an_undeclared_knob_gets_the_packages_own_default`` asserts it is
#: STILL unwritten, so a later edit adding it here cannot leave that test
#: passing while proving nothing.
#:
#: **That argument covers the DEFAULT leg only, and covering only that leg is
#: what let a passthrough with ``min_scale`` dropped from ``_CREATE_OPTIONS``
#: leave the whole of ``tests/config`` green** -- the dropped-forwarding
#: mutant produces exactly the package default this constant's silence
#: arranges for.  The declared leg is a document of its own, built inline by
#: ``test_a_declared_min_scale_is_forwarded_and_not_dropped``; do not answer
#: that gap by writing ``min_scale`` here, which would close the declared leg
#: by opening the default one.
NPE_SECTION = {
    "bank": {"n_simulations": 64, "seed": {"from": "runtime.seeds.bank"}},
    "create": {"n_components": 1, "width": 16, "depth": 2,
               "seed": {"from": "runtime.seeds.create"}},
    "train": {"n_steps": 50, "batch_size": 32,
              "seed": {"from": "runtime.seeds.train"}},
    "sample": {"n_draws": 100, "seed": {"from": "runtime.seeds.sample"}},
}

NPE = {"name": "amortized", "kind": "npe"}


def npe_document(run=None, *extra_runs, npe=None, **kwargs):
    """:func:`conjugate_document` with an ``inference.npe:`` section on it.

    ``extra_runs`` are passed straight through to
    :func:`conjugate_document` AFTER the npe run, for a document that needs a
    second run beside it -- Task 9's ``predict`` reusing this one.  They are
    NOT merged over :data:`NPE`; only ``run`` is.  The mirror of the way
    :func:`conjugate_document` already takes ``*runs``.

    ``npe`` is merged SUBSECTION BY SUBSECTION over :data:`NPE_SECTION`, so
    ``npe={"train": {"validation_fraction": 0.0}}`` keeps that subsection's
    seed and its n_steps.  A test that needs a key GONE -- the required-seed
    refusals -- builds the document and deletes from it, the way the gcr tests
    do.

    THE MERGE IS OVER THE UNION, not over :data:`NPE_SECTION`'s keys.  That
    matters for exactly one key: ``embed`` is a legal member of
    ``inference.npe:`` and :data:`NPE_SECTION` deliberately does NOT carry it
    (the parser's silent default IS the package's ``jnp.ravel``, and this
    layer restates no default).  A comprehension over ``NPE_SECTION.items()``
    would DISCARD ``npe={"embed": ...}`` silently -- measured, it keeps
    ``['bank', 'create', 'sample', 'train']`` and drops ``['embed']`` -- and
    ``test_the_embed_reaches_create_and_resizes_the_input_layer`` would then
    build a document with no embed at all, get ``jnp.ravel``, and assert
    ``in_size == 8`` against an actual 128.  ``embed`` is the only casualty;
    every other override names a subsection ``NPE_SECTION`` already has.

    ``run`` is merged over :data:`NPE`.  ``inference`` replaces the whole
    inference block (:data:`ONE_LATENT` by default) and the npe section is
    written onto whatever block results, so a caller never restates it.  Every
    other keyword goes to :func:`conjugate_document` unchanged -- ``at``,
    ``noise``, ``parameters``, ``prior``, ``model``, ``seeds``.

    The MODEL is deliberately left at :func:`conjugate_document`'s default
    (``CONJUGATE_MODEL``, which carries ``uniform_sky``): every number pinned
    in ``tests/config/test_config_exits_npe.py`` was measured on it, and
    ``WIENER_MODEL`` would move all of them.
    """
    block = dict(kwargs.pop("inference", None) or ONE_LATENT)
    supplied = npe or {}
    block["npe"] = {
        **{name: dict(sub) for name, sub in NPE_SECTION.items()},
        **{name: ({**NPE_SECTION[name], **value}
                  if name in NPE_SECTION else value)
           for name, value in supplied.items()},
    }
    kwargs.setdefault("seeds", NPE_SEEDS)
    return conjugate_document({**NPE, **(run or {})}, *extra_runs,
                              inference=block, **kwargs)


def npe_built(run=None, **kwargs):
    """:func:`npe_document`, BUILT -- the ``built`` the executor receives."""
    return load_document(npe_document(run, **kwargs))


def npe_spec():
    """A ``kind: npe`` RunSpec named ``amortized``, straight to a helper.

    Task 7 drove ``_simulate_bank`` and ``_estimator`` through this rather
    than ``run_document``, because ``parse_runs`` refused ``kind: npe`` until
    Task 8 promoted it -- the same reason :func:`nuts_spec` exists for Task 4.
    Those tests still use it: a test of the bank that does not pay for
    training is worth keeping, and it reads what a helper RAISED or RETURNED
    without ``run_document``'s loop turning a refusal into a
    ``RunResult.error``.

    **It took a ``**options`` and no longer does.**  Task 7 kept the parameter
    with no caller and wrote down the condition for removing it: the only
    run-level option a caller could write is one the exit must REFUSE, so the
    parameter had a future only if Task 8's sweep were tested through this
    helper.  It was not -- ``TestTheRunTakesNoKeysOfItsOwn`` drives both
    refusals through ``run_document``, which is where a user meets them --
    so the parameter is gone with the commit that decided it.
    """
    return RunSpec(name="amortized", kind="npe", variant=None, on="primary",
                   expect="ok", options={})


def npe_product(run=None, **kwargs):
    """:func:`npe_document`, EXECUTED, and the ``NpeProduct`` it hands back.

    Named for :data:`NPE`'s ``name:``, not for its kind -- see that constant.
    The tests that need the DOCUMENT alone, the ones that delete a key before
    executing it, reach for :func:`npe_document` directly.
    """
    return run_product(npe_document(run, **kwargs), "amortized")


#: :data:`~tests.config.exit_helpers.CONJUGATE_MODEL` with a PER-TIME gain
#: leaf.  A latent's shape must match the leaf it binds into --
#: ``ParameterSpace.validate`` refuses ``Bind for ('m',) produces shape (16,)
#: for `into` selector 0, but that leaf has shape ()`` -- so a document with a
#: NON-SCALAR latent needs a model with a non-scalar leaf, and this is the
#: smallest one this suite can build.  Kept apart from ``CONJUGATE_MODEL``
#: rather than replacing it: every number the conjugate family pins was
#: measured on the scalar gain.
VECTOR_GAIN_MODEL = {
    "global_signal": {"depth": {"value": 0.5, "unit": "K"},
                      "centre": {"value": 75.0, "unit": "MHz"},
                      "width": {"value": 5.0, "unit": "MHz"}},
    "uniform_sky": {"amplitude": {"value": 10.0, "unit": "K"}},
    "gain": {"gain": {"full": {"shape": ["n_time"], "value": 1.1},
                      "unit": "dimensionless"}},
    "noise": MODEL_NOISE,
}

#: THREE latents, declared ``d, a, m``.  Sorted is ``a, d, m`` and reversed is
#: ``m, d, a``, so the three orderings are pairwise different -- which two
#: latents can never be, because the reverse of a two-name sort IS one of the
#: two orders it is meant to be told apart from.  ``m`` is ``(16,)``
#: (measured through ``space.initial_values()`` on this document), so the flat
#: draws are 18 wide and an unravel assuming scalars consumes 3 of them.  The
#: three inits are 0.5, 10.0 and 1.1: unmistakable by magnitude, which is what
#: lets a mis-ordered unravel be read off a mean.
NPE_TRIO = {
    "parameters": {
        "d": {"init": 0.5, "linear": True, "into": "global_signal.depth",
              "prior": {"normal": {"loc": 0.5, "scale": 0.1}}},
        "a": {"init": 10.0, "linear": True, "into": "uniform_sky.amplitude",
              "prior": {"normal": {"loc": 10.0, "scale": 1.0}}},
        "m": {"init": {"full": {"shape": ["n_time"], "value": 1.1}},
              "linear": True, "into": "gain.gain",
              "prior": {"normal": {"loc": 1.1, "scale": 0.05}}},
    },
    "noise": HOMOSCEDASTIC,
    "observed": {"from": "simulation"},
}


def trio_npe_document():
    """The three-latent, one-vector document the unravel is measured on.

    32 simulations / 20 steps / 40 draws rather than :data:`NPE_SECTION`'s
    64 / 50 / 100: this document's estimator is 18-dimensional and the tests
    on it are about ORDER and SHAPE, not about a recovered posterior.

    It takes NO arguments, deliberately.  :func:`npe_document` carries ``run``
    and ``npe`` overrides because several tests need them; nothing needs one
    here, and Task 7's own ``npe_spec`` docstring records what an unexercised
    parameter costs -- a later edit can drop it while every test stays green.
    A test that needs a variant of this document calls
    :func:`npe_document` with these three keywords itself.
    """
    return npe_document(
        None, inference=NPE_TRIO, model=VECTOR_GAIN_MODEL,
        npe={"bank": {"n_simulations": 32},
             "train": {"n_steps": 20, "batch_size": 16},
             "sample": {"n_draws": 40}})


#: Two latents with NO ``prior:`` of their own, both covered by a joint prior.
#: ``to_numpyro_model`` accepts this (``joint.covers(name)``) and
#: ``simulate_pairs`` does not, which is the half of check A23 the schema does
#: not record.  ``{jeffreys: {over: [...]}}`` is the grammar;
#: ``{kind: jeffreys, names: [...]}`` is refused by ``build_space``.
#: It carries NO ``npe:`` of its own: :func:`joint_prior_document` writes one,
#: and a key here would be dead -- the builder overrides it -- while being the
#: last un-copied reference to the shared, module-level :data:`NPE_SECTION`,
#: which is the hazard that builder's copy exists to prevent.
JOINT_PRIOR_PAIR = {
    "parameters": {
        "d": {"init": 0.5, "linear": True, "into": "global_signal.depth"},
        "a": {"init": 10.0, "linear": True, "into": "uniform_sky.amplitude"},
    },
    "joint_prior": {"jeffreys": {"over": ["d", "a"]}},
    "noise": HOMOSCEDASTIC,
    "observed": {"from": "simulation", "at": {"d": TRUTH_D, "a": TRUTH_A}},
}


def joint_prior_document():
    """ONE document carrying BOTH exits: npe expecting a refusal, nuts running.

    The two are runs on the same document rather than two documents, because
    the claim is that ONE space is a posterior on one route and a refusal on
    the other.  Two documents that merely look alike cannot make that claim,
    and a refusal with no sibling demonstration cannot make the second half of
    it.  ``expect: refuse`` is ``runs.py``'s own mechanism for an exit that
    exists in order to be refused.

    :data:`NUTS` is passed whole rather than with ``progress_bar: False``
    written over it: that key has been part of the constant since Task 6
    repaid its debt, and restating it here would read as a claim that this
    document needs something :data:`NUTS` does not already carry.

    **This builder is what puts ``inference.npe:`` on the document**, and it
    does so DELIBERATELY: without it the npe run would be refused by
    :func:`~rheplicant.config.sections.npe._npe_spec` one branch earlier, and
    ``TestThePriorGate`` would be testing something else entirely.
    ``test_the_refusal_is_not_the_missing_section_one`` is what keeps that
    deliberate.

    The subsections are COPIED out of :data:`NPE_SECTION` for the same reason
    :func:`npe_document` copies them: the constant is module-level and shared,
    and a test that deletes a key from a built document would otherwise edit
    the template for every later document in the process.
    """
    return conjugate_document(
        {**NPE, "expect": "refuse"}, NUTS,
        inference={**JOINT_PRIOR_PAIR,
                   "npe": {name: dict(sub)
                           for name, sub in NPE_SECTION.items()}},
        seeds={**NPE_SEEDS, "chain": 3})
