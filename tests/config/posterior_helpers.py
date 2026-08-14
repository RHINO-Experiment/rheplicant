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
    ONE_LATENT,
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
    :func:`conjugate_document` unchanged.  A ``drop`` key inside ``run``
    takes keys AWAY, for a document that must be missing a required key.

    ``drop`` is NOT what the required-key refusals use -- those go through
    :func:`nuts_spec`'s own ``drop=``, which builds a RunSpec directly and is
    a different mechanism.  This branch exists for the refusals a LATER task
    reaches through ``run_document``.

    **Still no caller, and KEPT at Task 7's split, deliberately.**  Measured
    again here with ``grep``: the only occurrences of ``nuts_document({"drop":``
    in ``src`` or ``tests`` are the two inside this file's docstrings.  Task 5
    expected Task 6's ``run_document`` legs to be the callers and they were
    not -- Task 6's silent document goes through ``nuts_spec(drop=...)``, which
    builds the RunSpec the executor actually reads -- and Task 6 passed the
    keep-or-delete decision to whichever task made this split.  This task made
    it: **kept**, because the split's entire verification is an AST comparison
    asserting that ZERO bodies changed, and a deletion performed inside a pure
    move is a deletion nothing verified.  Task 9 is the next task to grow these
    builders (``predict`` reusing a chain, which is what would want a run
    beside this one); it deletes the branch if it still has no caller then.
    """
    merged = {**NUTS, **(run or {})}
    for key in merged.pop("drop", ()):
        merged.pop(key, None)
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

    It is NOT the only way to reach a missing required key -- an earlier
    draft of this docstring said so and was wrong.  ``nuts_document`` has its
    own ``drop``, and measured, ``nuts_document({"drop": ("num_samples",)})``
    reaches the identical refusal.
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


def npe_spec(**options):
    """A ``kind: npe`` RunSpec named ``amortized``, straight to a helper.

    Task 7 drives ``_simulate_bank`` and ``_estimator`` through this rather
    than ``run_document``, because ``parse_runs`` still refuses ``kind: npe``
    until Task 8 moves it out of ``_KINDS_2D`` -- the same reason
    :func:`nuts_spec` exists for Task 4.

    ``**options`` has NO caller: measured at Task 7, every one of the
    twenty-one call sites is a bare ``npe_spec()``.  It is kept, and the reason
    is not symmetry with :func:`nuts_spec` -- whose ``drop=``/``**options``
    carry the nuts run's own keys, which is a thing ``kind: nuts`` HAS.
    ``kind: npe`` takes no run-level keys at all (``_npe_spec``'s docstring
    says why: every knob is in ``inference.npe:``), so the only run-level
    option a caller can write is one the exit should REFUSE, and Task 8's
    sweep is the task that adds that refusal and its test.  If Task 8 lands
    that sweep through some other route, this parameter has no future caller
    and should go with the commit that decides so.
    """
    return RunSpec(name="amortized", kind="npe", variant=None, on="primary",
                   expect="ok", options=dict(options))
