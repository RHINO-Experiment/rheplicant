"""The documents the exit executors are measured on, and the one repair
that makes every one of them able to tell the two twins apart.

``tests/config/inference_helpers.py`` hoists a twin; this hoists whole
DOCUMENTS, because the conjugate family needs more than a twin: a latent whose
prediction really is affine, a noise kind whose sigma is decidable, and
observed data with a known truth to recover.  The diagnostics family
(``identifiability``, ``score_directions``, ``gradient``) needs less than
that and is built by :func:`diagnostic_document` at the foot.

Built by extending ``test_config_document.synthetic_document()``: the
stochastic ``noise`` node is REPAIRED AWAY rather than deleted (see
:func:`_repaired`), ``uniform_sky`` is added so a SECOND additive latent
exists, and the gain is left fixed at 1.1 so the two-latent block stays
affine -- ``gain`` times ``depth`` would be bilinear and ``check_linearity``
would say so.

Measured on this document: ``wiener_solve`` returns ``g = 1.4999994``
(truth 1.5) with a relative residual of 1.4e-07 and ``condition_estimate``
1.0; on the two-latent variant ``d = 1.19989``, ``a = 11.99995`` with
residual 8.1e-08 and kappa 12.1.

**The POSTERIOR documents are not here.**  ``kind: nuts`` and ``kind: npe``
are built by :mod:`tests.config.posterior_helpers`, which Task 7 split out
when this file reached 787 lines against a 800-line ceiling that nothing
guards automatically.  The import goes ONE WAY -- that module imports this
one, and this one imports nothing back, because a re-export here would close
a cycle that survives ``import exit_helpers`` and dies on
``import posterior_helpers``.  So "one place builds a test document" is now
one package spelled as two modules, and a test module imports each name from
whichever of the two defines it.  ``TWO_REFS`` and ``ONE_REF`` at the foot
below are the exception and stayed: they are built from :data:`TWO_LATENTS`,
which is this module's.
"""

from collections.abc import Mapping

from rheplicant.config.document import load_document
from rheplicant.config.sections.runs import RunSpec, run_document
from tests.config.test_config_document import synthetic_document

SIGMA_K = 0.05
TRUTH_G = 1.5
TRUTH_D = 1.2
TRUTH_A = 12.0
CHANNEL_WIDTH_HZ = 1.0e6
INTEGRATION_TIME_S = 2.0

#: ``synthetic_document``'s own stochastic node, KEPT in every model here and
#: repaired away in ``inference.twin.without:`` rather than omitted --
#: see :func:`_repaired` for why the difference is load-bearing.
#:
#: **Drawn at SIGMA_K, not its own 0.5** (D-10, Plan 3C Task 6).  Before this
#: edit this operator drew at 0.5 K while :data:`HOMOSCEDASTIC` below weighs
#: at ``SIGMA_K`` = 0.05 K -- D-C17's own disagreement row, and the base
#: fixture every ``preflight_document()`` delegates to.  A numeric C18
#: implemented to contract refuses the base document itself.  The twin repair
#: (``_repaired``, below) still applies on the ``twin: fit`` family, so on
#: every document but ``preflight_document()``'s own (which pins
#: ``observed.twin: full`` to dodge A42) this operator never touches the
#: data anyway -- moving its sigma costs nothing pinned.
MODEL_NOISE = {"type": "NoiseOperator", "sigma": {"value": SIGMA_K, "unit": "K"}}

# Named CONJUGATE_MODEL, not MODEL: tests/config/inference_helpers.py already
# exports a MODEL with different contents (no uniform_sky), and Tasks 3-6 all
# append to one test module.  The first of them to want inference_helpers'
# context or twin alongside these documents would have one `from ... import
# MODEL` silently win over the other.
CONJUGATE_MODEL = {
    "global_signal": {"depth": {"value": 0.5, "unit": "K"},
                      "centre": {"value": 75.0, "unit": "MHz"},
                      "width": {"value": 5.0, "unit": "MHz"}},
    "uniform_sky": {"amplitude": {"value": 10.0, "unit": "K"}},
    "gain": {"gain": {"value": 1.1, "unit": "dimensionless"}},
    "noise": MODEL_NOISE,
}

HOMOSCEDASTIC = {"kind": "homoscedastic",
                 "sigma": {"value": SIGMA_K, "unit": "K"}}
RADIOMETER = {"kind": "radiometer", "include_logdet": True,
              "channel_width": {"value": CHANNEL_WIDTH_HZ, "unit": "Hz"},
              "integration_time": {"value": INTEGRATION_TIME_S, "unit": "s"}}
FROZEN = {"kind": "radiometer_frozen", "source": "observed",
          "channel_width": {"value": CHANNEL_WIDTH_HZ, "unit": "Hz"},
          "integration_time": {"value": INTEGRATION_TIME_S, "unit": "s"}}

ONE_LATENT = {
    "parameters": {"g": {"init": 1.0, "linear": True, "into": "gain.gain",
                         "prior": {"normal": {"loc": 1.0, "scale": 0.5}}}},
    "noise": HOMOSCEDASTIC,
    "observed": {"from": "simulation", "at": {"g": TRUTH_G}},
}

TWO_LATENTS = {
    "parameters": {
        "d": {"init": 0.5, "linear": True, "into": "global_signal.depth",
              "prior": {"normal": {"loc": 0.5, "scale": 1.0}}},
        "a": {"init": 10.0, "linear": True, "into": "uniform_sky.amplitude",
              "prior": {"normal": {"loc": 10.0, "scale": 5.0}}},
    },
    "noise": HOMOSCEDASTIC,
    "observed": {"from": "simulation", "at": {"d": TRUTH_D, "a": TRUTH_A}},
}

NONLINEAR_LATENT = {
    "parameters": {"w": {"init": 5.0, "linear": True,
                         "into": "global_signal.width",
                         "prior": {"normal": {"loc": 5.0, "scale": 1.0}}}},
    "noise": HOMOSCEDASTIC,
    "observed": {"from": "simulation", "at": {"w": 6.0}},
}

NO_OBSERVED = {key: value for key, value in ONE_LATENT.items()
               if key != "observed"}

# The prior-FREE route, and the only one on which a compiled prior_std: can
# reach a solve.  Every latent above declares a prior:, and linear.py's
# _reconcile (:872) refuses a supplied prior_std= that DISAGREES with one --
# so `conjugate_document(prior=...)` cannot help here: it can set a prior,
# never clear one.  Swapping `parameters:` wholesale is what clears it:
#
#     conjugate_built(parameters=PRIOR_FREE)
#     conjugate_built(inference=TWO_LATENTS, parameters=PRIOR_FREE_TWO)
#
# With no prior anywhere, prior_std: is not merely accepted but REQUIRED --
# _require_prior_std (linear.py:1009) is what refuses the bare solve.
PRIOR_FREE = {"g": {"init": 1.0, "linear": True, "into": "gain.gain"}}

PRIOR_FREE_TWO = {
    "d": {"init": 0.5, "linear": True, "into": "global_signal.depth"},
    "a": {"init": 10.0, "linear": True, "into": "uniform_sky.amplitude"},
}


def _repaired(block):
    """An inference block with the fit-twin repair written into it.

    ``synthetic_document``'s stochastic ``noise`` node stays in ``model:``
    (:data:`MODEL_NOISE` puts it back into the two models here) and is
    repaired away in ``inference.twin.without:``.  Deleting it from ``model:``
    instead -- which is what every document in this file did before -- makes
    ``built.twin`` and ``built.inference.fit_twin`` the SAME OBJECT, and on
    such a document no test can tell an executor that reaches for one from an
    executor that reaches for the other.  Measured: with the repair, pointing
    ``_conjugate_block``'s ``linear_operator`` at ``built.twin`` fails 74
    tests -- 20 in ``test_config_conjugate_shared``, 22 in
    ``test_config_exits_conjugate``, 17 in ``test_config_exits_gcr``, 15 in
    ``test_config_exits_gls`` -- and without it, none.

    ``observed: {from: simulation}`` defaults to ``twin: full`` -- the FULL
    twin, NoiseOperator and all -- so the repair alone would turn every
    simulated observation into a noise realisation and move every pinned
    number in this suite (measured: ``g`` recovered at 1.5072 instead of
    1.5000).  Simulating from the FIT twin is what keeps the data exactly the
    deterministic forward it has always been: measured both ways, the
    observed arrays are BIT-IDENTICAL to the pre-repair documents.

    Both keys are DEFAULTS, not overrides.  A caller that declares its own
    ``twin:`` -- ``{replace: ...}``, or a ``without:`` naming another node --
    keeps it, and so does a caller that wants ``observed.twin: full`` because
    it is after noise-realised data.  Supplying silently and overwriting
    silently look identical until the day someone declares one of these and
    gets the other, with nothing said.

    Both edits are copies, never mutations: the caller's block is untouched.
    """
    repaired = dict(block)
    repaired.setdefault("twin", {"without": ["noise"]})
    observed = repaired.get("observed")
    if (isinstance(observed, Mapping) and "twin" not in observed
            and observed.get("from") == "simulation"):
        repaired["observed"] = {**observed, "twin": "fit"}
    return repaired


def conjugate_document(*runs, inference=None, parameters=None, noise=None,
                       prior=None, at=None, seeds=None, model=None):
    """The shared document, with the runs a test wants declared on it.

    This is the ONE document builder the conjugate tests use, across both
    ``tests/config/test_config_conjugate_shared.py`` and
    ``tests/config/test_config_exits_conjugate.py``, and Tasks 5-6 append to
    the second.  A second ``def conjugate_document`` in either module would
    SHADOW this import rather than sit beside it, and every earlier task's
    call would start raising TypeError from the later task's commit onward.

    ``model:`` replaces the whole model block (:data:`CONJUGATE_MODEL` by
    default) and ``inference:`` the whole inference block (:data:`ONE_LATENT`
    by default).
    The other four edit that inference block rather than replacing it, which is
    what most tests want: ``parameters`` swaps ``inference.parameters``,
    ``prior`` rewrites every latent's ``prior:`` to one mapping, ``noise`` swaps
    ``inference.noise``, and ``at`` swaps the values the observed data is
    simulated at.  ``seeds`` writes ``runtime.seeds``, which is what a
    ``seed: {from: runtime.seeds.<name>}`` on a run resolves against.

    ``prior`` can only SET a prior, never clear one; for a document with no
    prior at all -- the one a compiled ``prior_std:`` can actually solve on --
    pass ``parameters=PRIOR_FREE`` (or ``PRIOR_FREE_TWO``).

    Every block goes through :func:`_repaired` last, so a caller's
    ``inference=`` never has to remember the ``twin:`` repair; a caller that
    declares one of its own keeps it.  A ``model=`` of its own must carry a
    ``noise`` node (:data:`MODEL_NOISE`) unless it also declares the
    ``twin:`` it wants, because ``without: [noise]`` is what the default
    names.
    """
    doc = synthetic_document()
    doc["model"] = {key: dict(value)
                    for key, value in (CONJUGATE_MODEL if model is None
                                       else model).items()}
    block = dict(inference if inference is not None else ONE_LATENT)
    if parameters is not None:
        block["parameters"] = dict(parameters)
    if prior is not None:
        block["parameters"] = {name: {**latent, "prior": prior}
                               for name, latent in block["parameters"].items()}
    if noise is not None:
        block["noise"] = noise
    if at is not None:
        block["observed"] = {**block.get("observed", {"from": "simulation"}),
                             "at": at}
    doc["inference"] = _repaired(block)
    if seeds is not None:
        doc["runtime"] = {**doc["runtime"], "seeds": dict(seeds)}
    doc["runs"] = [dict(one) for one in runs] or [{"kind": "forward"}]
    return doc


def conjugate_built(*runs, **document):
    """The same document, built -- the ``built`` an executor receives.

    ``**document`` is forwarded to :func:`conjugate_document` unchanged, so
    every keyword above works here too.
    """
    return load_document(conjugate_document(*runs, **document))


def spec(kind="conjugate.wiener", *, on="primary", **options):
    """A RunSpec straight to a helper, without going through parse_runs.

    ``on`` is keyword-only and lands on the SPEC, not among the
    kind-specific options.  Before it existed, ``spec(on="night")`` built
    ``RunSpec(on="primary", options={"on": "night"})`` -- measured -- so a
    test written that way read the PRIMARY's sigma while its name claimed
    the secondary's, and passed either way.  No caller passes ``on`` as an
    option today (measured: zero), so making it keyword-only takes nothing
    away.
    """
    return RunSpec(name=kind, kind=kind, variant=None, on=on,
                   expect="ok", options=dict(options))


# --- The two documents the RUNNABLE conjugate exits are measured on --------
#
# The three above (ONE_LATENT, TWO_LATENTS, NONLINEAR_LATENT) drive the shared
# helpers directly.  These two go through `run_document`, so they are chosen
# for what an EXECUTOR has to demonstrate rather than for what a block builder
# does: a truth to recover, and a pair ill-conditioned enough that the
# package's own convergence guard fires at its default.

#: :data:`CONJUGATE_MODEL` without ``uniform_sky``.  Kept apart rather than
#: merged: every number the wiener exit pins was measured on this model, and
#: the second additive latent CONJUGATE_MODEL exists to provide would change
#: the conditioning of both documents below.
WIENER_MODEL = {
    "global_signal": {"depth": {"value": 0.5, "unit": "K"},
                      "centre": {"value": 75.0, "unit": "MHz"},
                      "width": {"value": 5.0, "unit": "MHz"}},
    "gain": {"gain": {"value": 1.1, "unit": "dimensionless"}},
    "noise": MODEL_NOISE,
}
# gain.gain is the LAST node of the synthetic twin, so g scales the whole
# prediction and the block is exactly affine in it.  scale 10.0 is a prior
# wide enough that the posterior mean IS the truth, to 1.3e-06 (measured).
GAIN_LATENT = {"init": 1.0, "linear": True, "into": "gain.gain",
               "prior": {"normal": {"loc": 1.0, "scale": 10.0}}}
GROUND_PICKUP = {"coupling": {"value": 0.01, "unit": "dimensionless"},
                 "t_ground": {"value": 290.0, "unit": "K"}}


def wiener_document(run, *, parameters=None, at=None, noise=None):
    """One linear latent ``g`` into ``gain.gain``; data simulated at g = 1.5.

    Measured on it: ``wiener_solve`` returns ``g = 1.4999987`` with a relative
    residual of 0.0.
    """
    return conjugate_document(
        run, model=WIENER_MODEL,
        inference={"parameters": parameters or {"g": GAIN_LATENT},
                   "noise": noise or HOMOSCEDASTIC,
                   "observed": {"from": "simulation",
                                "at": at or {"g": TRUTH_G}}})


def two_latent_document(run, *, noise=None):
    """Two latents that are affine JOINTLY -- which a pair including the gain
    never is.

    The twin runs ``global_signal -> ground_pickup -> gain``, so the gain
    multiplies everything upstream of it and ``(gain, anything)`` is bilinear;
    ``check_linearity`` says so in as many words.  ``depth`` and ``coupling``
    both sit upstream of the gain and add, so the pair passes the check.
    Data is simulated at ``depth = 1.0 K`` and ``coupling = 0.02``.

    The pair is also ill-conditioned enough that the package's own
    ``require_convergence=1e-3`` default FIRES on it -- measured, as an
    ``eqx.error_if`` from inside jit -- which is what makes a declared
    ``require_convergence: null`` observable.  With the guard off it recovers
    ``dep = 0.9999954`` and ``c = 0.0199999`` (residual 8.9e-08).

    ``noise`` swaps :data:`HOMOSCEDASTIC` out.  It exists for
    :func:`gls_pair_document`, which needs this pair's CONDITIONING under a
    prediction-dependent sigma: a one-latent block's normal operator is 1x1
    and CG reaches its answer in a single iteration, so ``maxiter:`` has no
    lever there at all -- and a knob with no lever is a knob no test can
    watch travel.
    """
    return conjugate_document(
        run, model={**WIENER_MODEL, "ground_pickup": GROUND_PICKUP},
        inference={
            "parameters": {
                "dep": {"init": 0.5, "linear": True,
                        "into": "global_signal.depth",
                        "prior": {"normal": {"loc": 0.5, "scale": 10.0}}},
                "c": {"init": 0.01, "linear": True,
                      "into": "ground_pickup.coupling",
                      "prior": {"normal": {"loc": 0.01, "scale": 10.0}}},
            },
            "noise": HOMOSCEDASTIC if noise is None else noise,
            "observed": {"from": "simulation",
                         "at": {"dep": 1.0, "c": 0.02}},
        })


# --- The runs those documents carry, and the one way to execute them -------
#
# These live HERE rather than in a test module because the conjugate tests are
# TWO modules now -- ``test_config_conjugate_shared.py`` for what never calls
# run_document, ``test_config_exits_conjugate.py`` for what always does -- and
# Tasks 5-6 append to the second.  A run template or a product helper copied
# per module is how the copies drift apart.
#
# names: is REQUIRED -- ``_conjugate_block``'s ``_selected`` raises without it,
# and ``test_names_is_required_and_says_why`` pins that refusal -- so both
# templates carry one.  ``["g"]`` is :func:`wiener_document`'s and
# :func:`gcr_document`'s own single latent; the :func:`two_latent_document`
# call sites override it with ``["dep", "c"]`` or a deliberate sub-block.
WIENER = {"kind": "conjugate.wiener", "width": "none", "names": ["g"]}
GCR = {"kind": "conjugate.gcr", "names": ["g"],
       "seed": {"from": "runtime.seeds.draws"}}

# GAIN_LATENT -- the document's default latent, above -- is what
# wiener_document() binds when a test passes no parameters=.  The same latent
# under a prior tight enough that the prior curvature is two thirds of the
# answer, which is what makes width: fisher's space= visible as a number
# rather than as a fifth decimal place:
TIGHT_GAIN = {"init": 1.0, "linear": True, "into": "gain.gain",
              "prior": {"normal": {"loc": 1.0, "scale": 0.005}}}
# Declared linear=True and demonstrably not: the prediction is a Gaussian in
# frequency, so its CENTRE is the knob check_linearity refuses.
CENTRE_LATENT = {"init": 75.0, "linear": True, "into": "global_signal.centre",
                 "prior": {"normal": {"loc": 75.0, "scale": 10.0}}}


def run_product(document, name="conjugate.wiener"):
    """The named run of ``document``, executed, and its product.

    ``name`` defaults to the run NAME an unnamed :data:`WIENER` gets: ``runs.py``
    names a run after its kind when the entry declares no ``name:``.
    """
    return run_document(document)[name].product


# --- What a DRAW is measured against ---------------------------------------
#
# A prior of 0.01 against a likelihood of 0.015872 makes the posterior
# 0.0084608 -- a width that is NEITHER of its parents, so a draw taken at the
# prior's width or at the likelihood's is a visibly different number.  All
# three were measured through run_document on gcr_document()'s own model and
# prior, via kind: fisher:
#     {"kind": "fisher"}               -> covariance.sigma("g") = 0.015872
#     {"kind": "fisher", "space": True} -> 0.0084608
# model=WIENER_MODEL is not optional: on CONJUGATE_MODEL, which adds
# uniform_sky, the same two come back 0.00045170 and 0.00045124 -- a fraction
# of a percent apart, which no scatter test could tell apart.
TIGHT = {"normal": {"loc": 1.0, "scale": 0.01}}
PRIOR_SIGMA = 0.01
LIKELIHOOD_SIGMA = 0.015872
POSTERIOR_SIGMA = 0.0084608

# 3.5714286 MHz over 8 channels is the synthetic document's own grid spacing,
# and 2 s its own sample step.  channel_width/integration_time are spelled out
# because their {from: observation} default reads
# observation.time.channel_width, which this document does not declare.  The
# name is GCR_RADIOMETER, not RADIOMETER: this module already exports a
# RADIOMETER at a different bandwidth, and every number measured under this
# one would move if the two were confused.
GCR_RADIOMETER = {"kind": "radiometer",
                  "channel_width": {"value": 3.5714286, "unit": "MHz"},
                  "integration_time": {"value": 2.0, "unit": "s"},
                  "include_logdet": False}


def gcr_document(run=None, **kwargs):
    """The shared one-latent document with a ``conjugate.gcr`` run on top.

    ``model=WIENER_MODEL``, ``prior=TIGHT`` and ``seeds={"draws": 11}`` are
    pinned rather than left to :func:`conjugate_document`'s defaults, which are
    :data:`CONJUGATE_MODEL` (it carries ``uniform_sky``), each latent's own
    ``scale: 0.5``, and no ``runtime.seeds`` at all.  Every number this
    document is measured at -- :data:`LIKELIHOOD_SIGMA`,
    :data:`POSTERIOR_SIGMA` and the draw scatter around them -- was measured
    without the first and with the second; a named seed is what
    :data:`GCR`'s ``seed: {from: runtime.seeds.draws}`` resolves against.

    ``run`` is merged over :data:`GCR`, so a caller adds ``n_draws:`` or
    ``noise_from:`` without restating ``names:`` or ``seed:``; every other
    keyword goes to :func:`conjugate_document` unchanged.
    """
    kwargs.setdefault("model", WIENER_MODEL)
    kwargs.setdefault("prior", TIGHT)
    kwargs.setdefault("seeds", {"draws": 11})
    return conjugate_document({**GCR, **(run or {})}, **kwargs)


def gcr_product(run=None, **kwargs):
    """:func:`gcr_document`, executed, and the draw product it hands back.

    The pair is one call because almost every gcr test wants both halves;
    the tests that need the DOCUMENT alone -- the ones that delete a key from
    the run before executing it -- reach for :func:`gcr_document` directly.
    """
    return run_product(gcr_document(run, **kwargs), "conjugate.gcr")


# --- What kind: conjugate.gls is measured on --------------------------------
#
# The reweighted route needs a PREDICTION-DEPENDENT sigma -- the one thing
# check A27 refuses everywhere else in the family -- so this document declares
# :data:`RADIOMETER` rather than :data:`HOMOSCEDASTIC`.  RADIOMETER and not
# GCR_RADIOMETER: the two differ in bandwidth (1 MHz against 3.5714286) and
# every number the gls tests pin was measured under the first.

#: ``name: gls``, deliberately NOT the kind.  ``runs.py`` names an unnamed run
#: after its kind, so a run named for its kind cannot tell a ``where`` built
#: from ``run.name`` -- what ``exits.py`` spells -- from one built out of
#: ``run.kind``.  Under this name the two are different strings, and the
#: refusal tests read the prefix.
GLS = {"name": "gls", "kind": "conjugate.gls", "names": ["g"]}


def gls_document(run=None, **kwargs):
    """The shared one-latent document with a ``conjugate.gls`` run on top.

    ``model=WIENER_MODEL`` and ``noise=RADIOMETER`` are pinned rather than
    left to :func:`conjugate_document`'s defaults, which are
    :data:`CONJUGATE_MODEL` (it carries ``uniform_sky``, which changes the
    conditioning of every solve) and :data:`HOMOSCEDASTIC` (whose sigma does
    not depend on the prediction, so ``iterative_gls`` would return after one
    step and there would be no reweighting to observe).

    The inference block is :data:`ONE_LATENT`'s: ``g`` into ``gain.gain``,
    which is the LAST node of the synthetic twin, so ``g`` scales the whole
    prediction and the block is exactly affine in it.  ``WIENER_MODEL`` is
    ``synthetic_document()``'s model with the stochastic ``noise`` node
    dropped, so ``observed`` is a deterministic forward at ``g = 1.5``: every
    number these tests pin is exact across runs, and the document carries no
    randomness at all.

    ``run`` is merged over :data:`GLS`, so a caller adds a reweight knob
    without restating ``names:``; every other keyword goes to
    :func:`conjugate_document` unchanged (``at=`` moves the value the data is
    simulated at, ``parameters=`` swaps the latents wholesale).
    """
    kwargs.setdefault("model", WIENER_MODEL)
    kwargs.setdefault("noise", RADIOMETER)
    return conjugate_document({**GLS, **(run or {})}, **kwargs)


def gls_product(run=None, **kwargs):
    """:func:`gls_document`, executed, and the ``GLSResult`` it hands back.

    Named for :data:`GLS`'s ``name:``, not for its kind -- see that constant.
    """
    return run_product(gls_document(run, **kwargs), "gls")


#: :func:`two_latent_document`'s pair, reweighted.  ``require_convergence:
#: null`` is part of the template rather than of each caller's run: this pair
#: is ill-conditioned enough that the package's own default guard fires on it
#: (it compares residual x kappa, linear.py:1493), so every test that is not
#: ABOUT the guard would otherwise spend its first line turning it off.
GLS_PAIR = {"name": "gls", "kind": "conjugate.gls", "names": ["dep", "c"],
            "require_convergence": None}


def gls_pair_document(run=None):
    """The two-latent pair under :data:`RADIOMETER`, as a ``conjugate.gls`` run.

    :func:`gls_document`'s block holds ONE latent, whose normal operator is
    1x1 -- CG reaches its answer in a single iteration there, so ``maxiter:``
    changes nothing on it whatever value it takes.  This pair is the document
    where the cap has a lever, and that is the whole reason it exists: a knob
    the config layer forwards and no test can watch arrive is exactly the
    defect this plan is written against.
    """
    return two_latent_document({**GLS_PAIR, **(run or {})}, noise=RADIOMETER)


def gls_pair_product(run=None):
    """:func:`gls_pair_document`, executed, and its ``GLSResult``."""
    return run_product(gls_pair_document(run), "gls")


# --- What the CHEAP diagnostics are measured on -----------------------------
#
# ``identifiability``, ``score_directions`` and ``gradient`` need no conjugate
# block at all: a ParameterSpace, the fit twin and the state are the whole
# input for the first two, and the third adds one objective.  So this family
# keeps ``synthetic_document``'s own model rather than swapping in
# :data:`CONJUGATE_MODEL`, and goes through the same :func:`_repaired` seam.
#
# It lives HERE rather than in a test module because THREE modules now build
# it -- ``test_config_exits_diagnostics.py`` (Task 7),
# ``test_config_diagnostics_guards.py`` and ``test_config_exits_gradient.py``
# (Task 8) -- and a fixture copied per module is how the repair above gets
# re-derived as the deletion form, after which every copy quietly stops
# discriminating.

#: ``g`` scales the whole prediction and ``d`` scales the signal the gain
#: multiplies, so ``d(data)/dg = d * gaussian`` and ``d(data)/dd = g *
#: gaussian`` are proportional -- the schema's A33 shape (bandpass and gain
#: both free) in the cheapest model this suite already builds.
DIAGNOSTIC_GAIN = {"init": 1.0, "linear": True, "into": "gain.gain"}
DIAGNOSTIC_DEPTH = {"init": 0.5, "into": "global_signal.depth"}
#: The width enters the gaussian's EXPONENT, so its column is not proportional
#: to the gain's: measured singular values 1.2572932 and 0.6474673, a ratio of
#: 0.514969.  That is the pair on which ``rtol`` can move the rank -- on the
#: degenerate pair the rank is 1 for every tolerance, and an executor that
#: dropped ``rtol:`` would pass every test built on it.
DIAGNOSTIC_WIDTH = {"init": {"value": 5.0, "unit": "MHz"},
                    "into": "global_signal.width"}

#: ``..._PAIR``, because ``DEGENERATE`` is already
#: ``test_config_exits_diagnostics.py``'s frequency basis for its condition
#: tests.
DEGENERATE_PAIR = {"g": DIAGNOSTIC_GAIN, "d": DIAGNOSTIC_DEPTH}
IDENTIFIED_PAIR = {"g": DIAGNOSTIC_GAIN, "w": DIAGNOSTIC_WIDTH}


def diagnostic_document(run, parameters=None, inference=None):
    """A document whose FIT twin is not its model twin.

    ``synthetic_document``'s stochastic ``noise`` node stays in ``model:`` and
    is repaired away in ``inference.twin.without:`` rather than being deleted
    from the model -- see :func:`_repaired` for what that buys, and for what
    it would cost if the observed data were simulated from the full twin.  The
    prediction is the same either way -- measured bit-identical reports,
    singular values 1.41421356 and 4.5236e-17 both ways -- but this way
    ``built.twin`` still carries NoiseOperator while
    ``built.inference.fit_twin`` does not, so an executor that differentiated
    ``built.twin`` raises ``refuse_stochastic_stages`` and every test built on
    this document fails.  Delete the ``twin:`` repair and reach for
    ``built.twin`` instead, and nothing notices: with no ``inference.twin:``
    the two are the same object.

    ``parameters`` names the latents alone (:data:`DEGENERATE_PAIR` by
    default), which is all ``identifiability`` and ``score_directions`` need;
    ``inference`` replaces the whole block, which is what ``gradient`` wants
    when it declares a noise model and observed data beside them.  The two are
    alternatives: ``inference`` wins, and ``parameters`` is then ignored.
    """
    doc = synthetic_document()
    block = (dict(inference) if inference is not None
             else {"parameters": dict(parameters or DEGENERATE_PAIR)})
    doc["inference"] = _repaired(block)
    doc["runs"] = [run]
    return doc


def diagnostic_report(run, parameters=None):
    """One ``identifiability`` exit -> its ``IdentifiabilityReport``.

    Keyed by ``"identifiability"``, so like :func:`run_product` above it
    serves only runs that leave ``name:`` unwritten -- ``runs.py`` then names
    a run after its kind.  The one test that needs a name of its own calls
    ``run_document`` directly.
    """
    return run_product(diagnostic_document(run, parameters),
                       "identifiability")


def diagnostic_rows(run, parameters=None):
    """One ``score_directions`` exit -> its ``{latent: (size, n_data)}``."""
    return run_product(diagnostic_document(run, parameters),
                       "score_directions")


# --- What the observation FAN is measured on --------------------------------
#
# `radiometer_frozen` with `source: observed` decides its sigma FROM the data,
# so a document with two observations has TWO sigmas -- and a run's `on:` says
# which of them weighs its residuals.  Every other noise kind is one model or
# one array for the whole document, and `source: prediction_at_init` reads the
# twin, so this is the one shape in the layer that fans at all.

#: Twice :data:`TRUTH_G`, so ``night``'s data is exactly twice ``primary``'s
#: and so is its frozen sigma.  A fixture whose two sigmas were EQUAL could
#: not tell the fan from the bug it replaces -- which is why the truths differ
#: rather than, say, the two entries' channel widths.
TRUTH_NIGHT = 3.0

TWO_OBSERVED = {
    "parameters": {"g": GAIN_LATENT},
    "noise": FROZEN,
    # `twin: fit` is spelled out on BOTH entries: `_repaired` supplies that
    # default only for the single-observation form (it looks for a top-level
    # `from:`), and a named mapping without it simulates from the FULL twin,
    # whose NoiseOperator makes the data a noise realisation and every exact
    # ratio below a near miss.
    "observed": {
        "primary": {"from": "simulation", "twin": "fit",
                    "at": {"g": TRUTH_G}},
        "night": {"from": "simulation", "twin": "fit",
                  "at": {"g": TRUTH_NIGHT}},
    },
}

#: 1/sqrt(channel_width * integration_time) for :data:`FROZEN` -- the factor
#: ``freeze_sigma`` multiplies |reference| by.  Bound here so a test can say
#: "this sigma is THIS observation's" rather than "this sigma is some array".
FROZEN_FRACTION = 1.0 / (CHANNEL_WIDTH_HZ * INTEGRATION_TIME_S) ** 0.5


def fanned_document(run, *, noise=None):
    """The two-observation document a run's ``on:`` can actually choose in.

    ``noise`` swaps :data:`FROZEN` out, which is how a test asks what the
    fan does NOT do: under a model kind there is one noise for the document
    and ``on:`` chooses nothing.
    """
    block = dict(TWO_OBSERVED)
    if noise is not None:
        block["noise"] = noise
    return conjugate_document(run, model=WIENER_MODEL, inference=block)


def fanned_built(run=None, *, noise=None):
    """:func:`fanned_document`, built -- the ``built`` an accessor receives."""
    return load_document(fanned_document(run or {"kind": "forward"},
                                         noise=noise))


#: :data:`TWO_LATENTS` with a ``ref:`` on EACH -- the document the name-to-ref
#: PAIRING is measured on, which one latent cannot show at all.  The two refs
#: are far from each other and from both inits (0.5, 10.0), so a swapped
#: pairing is a number no assertion can read as rounding; ``d`` is declared
#: before ``a``, which is NOT alphabetical, because ``space.names`` carries
#: declaration order and everything downstream inherits it.
TWO_REFS = {**TWO_LATENTS,
            "parameters": {
                "d": {**TWO_LATENTS["parameters"]["d"], "ref": 0.25},
                "a": {**TWO_LATENTS["parameters"]["a"], "ref": 40.0}}}

#: The same pair with a ``ref:`` on ONE of them: the MIXED document
#: ``_init_strategy``'s refusal is written about and that no document with
#: zero refs (:data:`ONE_LATENT`) or two (above) can stand in for.
ONE_REF = {**TWO_REFS,
           "parameters": {**TWO_REFS["parameters"],
                          "a": TWO_LATENTS["parameters"]["a"]}}
