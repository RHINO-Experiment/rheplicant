"""One linear-Gaussian document the conjugate exits actually converge on.

``tests/config/inference_helpers.py`` hoists a twin; this hoists a whole
DOCUMENT, because the conjugate family needs more than a twin: a latent whose
prediction really is affine, a noise kind whose sigma is decidable, and
observed data with a known truth to recover.

Built by extending ``test_config_document.synthetic_document()``: the
stochastic ``noise`` node is dropped (a conjugate solve fits a deterministic
twin), ``uniform_sky`` is added so a SECOND additive latent exists, and the
gain is left fixed at 1.1 so the two-latent block stays affine -- ``gain``
times ``depth`` would be bilinear and ``check_linearity`` would say so.

Measured on this document: ``wiener_solve`` returns ``g = 1.4999994``
(truth 1.5) with a relative residual of 1.4e-07 and ``condition_estimate``
1.0; on the two-latent variant ``d = 1.19989``, ``a = 11.99995`` with
residual 8.1e-08 and kappa 12.1.
"""

from rheplicant.config.document import load_document
from rheplicant.config.sections.runs import RunSpec, run_document
from tests.config.test_config_document import synthetic_document

SIGMA_K = 0.05
TRUTH_G = 1.5
TRUTH_D = 1.2
TRUTH_A = 12.0
CHANNEL_WIDTH_HZ = 1.0e6
INTEGRATION_TIME_S = 2.0

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
    doc["inference"] = block
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


def spec(kind="conjugate.wiener", **options):
    """A RunSpec straight to a helper, without going through parse_runs."""
    return RunSpec(name=kind, kind=kind, variant=None, on="primary",
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


def two_latent_document(run):
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
            "noise": HOMOSCEDASTIC,
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
