"""``kind: conjugate.wiener`` end to end, and what both conjugate kinds share.

The seam against ``test_config_conjugate_shared.py`` is mechanical rather than
thematic: **every test here is about the runs list** -- it parses one, asks
:data:`EXECUTORS` whether a kind is registered, or drives one through
:func:`run_document`, from the YAML mapping down to ``wiener_solve`` and back.
So each of them also depends on the kind being admitted by ``runs.py``, which
is what the four-step promotion of a kind turns green.

The shared machine those executors open with -- ``_conjugate_block``,
``_prior_kwargs``, ``_decided_sigma``, ``_decided_model`` -- is driven
directly in ``test_config_conjugate_shared.py``, which calls ``run_document``
nowhere.  ``kind: conjugate.gcr``'s own end-to-end tests are in
``test_config_exits_gcr.py``; what stays HERE is what the two kinds share
because one executor serves both: the registry, the parse, and the order
:func:`_run_conjugate` runs its checks in.
"""

import equinox as eqx
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections import exits
from rheplicant.config.sections.runs import parse_runs, run_document
from rheplicant.core.errors import ParameterSpaceError
from tests.config.exit_helpers import (
    CENTRE_LATENT,
    FROZEN,
    PRIOR_FREE,
    RADIOMETER,
    TIGHT_GAIN,
    TRUTH_G,
    WIENER,
    conjugate_document,
    gcr_document,
    run_product,
    two_latent_document,
    wiener_document,
)

#: :data:`WIENER` with the one key that has no default taken back out.
NO_WIDTH = {key: value for key, value in WIENER.items() if key != "width"}


LINEARITY_DECLINED = {"linearity": {"mode": "skip",
                      "reason": "this fixture declares linear: true on a "
                                "latent the prediction is not affine in, on "
                                "purpose, so that the exit-level check: has "
                                "a lever"}}


def _declined(document):
    document["inference"]["checks"] = LINEARITY_DECLINED
    return document


class TestTheKindIsRunnable:
    def test_conjugate_wiener_parses_rather_than_deferring(self):
        (run,) = parse_runs([{"kind": "conjugate.wiener", "width": "none"}])
        assert run.kind == "conjugate.wiener"
        assert run.options == {"width": "none"}

    def test_both_conjugate_kinds_have_an_executor(self):
        # Read out of ``exits`` and not out of ``exit_support``, where the
        # table itself lives: importing the ROOT is what pulls the leaf
        # modules in and runs their @register decorators, and a table read
        # straight from exit_support reflects only what this process has
        # happened to import -- which, in a single-file run of this module, is
        # nothing at all.
        assert "conjugate.wiener" in exits.EXECUTORS
        assert "conjugate.gcr" in exits.EXECUTORS

    def test_conjugate_gcr_parses_rather_than_deferring(self):
        (run,) = parse_runs([{"kind": "conjugate.gcr", "names": ["g"]}])
        assert run.kind == "conjugate.gcr"
        assert run.options == {"names": ["g"]}

    def test_the_kind_sweeps_keys_that_belong_to_its_siblings(self):
        # n_draws is conjugate.gcr's.  Accepting it here would take a key
        # that asks for a posterior and return a point.
        with pytest.raises(ConfigError, match="n_draws") as caught:
            run_document(wiener_document({**WIENER, "n_draws": 4}))
        message = str(caught.value)
        # The sweep set has to be THIS exit's, not the family's: _sweep names
        # what it does take, and width: is the key conjugate.wiener adds to
        # _SOLVE_KEYS.  Sweeping with the shared set instead would refuse
        # n_draws just the same and this match alone would not notice.
        assert "kind: conjugate.wiener does not take ['n_draws']" in message
        assert "'width'" in message


class TestTheSolve:
    def test_the_solve_recovers_the_injected_truth(self):
        # observed.at injects g = 1.5; measured |g_hat - 1.5| = 1.3e-06 in
        # the float32 default session, so abs=1e-4 is 75x the measured error
        # and still 4000x away from the init (1.0) that a solve which never
        # moved would hand back.
        product = run_product(wiener_document(WIENER))
        assert float(product["mean"]["g"]) == pytest.approx(TRUTH_G, abs=1e-4)

    def test_the_product_carries_the_relative_residual_as_a_number(self):
        # wiener_solve returns a 2-TUPLE, (x_hat, relative_residual), and the
        # residual is a scalar jax.Array.  ``type(...) is float`` is the
        # assertion: isinstance would also accept a numpy float64, and these
        # products end up in reports and refusal messages.
        product = run_product(wiener_document(WIENER))
        assert type(product["residual"]) is float
        assert product["residual"] < 1e-5

    def test_a_grouped_block_of_one_still_comes_back_a_mapping(self):
        # The schema always compiles to names=, even for a block of one,
        # because six downstream consumers raise on the bare-array form.
        # name='g' would put a bare array under "mean" and fail here.  (It
        # does NOT prove block.as_dict was called: over a grouped block the
        # solve already returns the mapping, and as_dict is the idempotent
        # wrap that makes the product's shape independent of that.)
        product = run_product(wiener_document(WIENER))
        assert sorted(product["mean"]) == ["g"]

    def test_a_two_latent_block_recovers_both(self):
        # Measured: dep = 0.9999954 against a truth of 1.0, c = 0.0199999
        # against 0.02.  Two latents whose scales differ by 50x, so a prior
        # or a solution filed under the wrong member shows up here.
        product = run_product(two_latent_document(
            {**WIENER, "names": ["dep", "c"], "require_convergence": None}))
        assert float(product["mean"]["dep"]) == pytest.approx(1.0, abs=1e-3)
        assert float(product["mean"]["c"]) == pytest.approx(0.02, abs=1e-4)

    def test_the_product_is_keyed_in_the_declared_order_not_the_spaces(self):
        # BOTH directions, because with two latents no single document can
        # separate the three candidate orderings.  inference.parameters
        # declares dep before c, so:
        #   names: ["dep", "c"] -- space order AGREES, sorted() does not, so
        #                          this half kills a product rebuilt through
        #                          sorted() and cannot see space.names;
        #   names: ["c", "dep"] -- sorted() AGREES, space order does not, so
        #                          this half kills a product rebuilt from
        #                          space.names and cannot see sorted().
        # Only the block's own order satisfies both.
        for names in (["dep", "c"], ["c", "dep"]):
            product = run_product(two_latent_document(
                {**WIENER, "names": names, "require_convergence": None}))
            assert list(product["mean"]) == names

    def test_names_selects_a_sub_block(self):
        # Holding c FIXED is a different question from fitting the pair, and
        # measurably so: dep comes back -2.9279 here against 0.9999954 when
        # the block holds both.  Asserting only the key set would pass
        # against an executor that solved the whole space and then sliced.
        product = run_product(two_latent_document(
            {**WIENER, "names": ["dep"], "require_convergence": None}))
        assert sorted(product["mean"]) == ["dep"]
        assert float(product["mean"]["dep"]) == pytest.approx(-2.9279,
                                                              rel=1e-3)

    def test_require_convergence_null_reaches_the_solve(self):
        # The two-latent block is ill-conditioned enough that the guard fires
        # once it is asked for -- as an eqx.error_if from inside jit, so an
        # EquinoxRuntimeError and NOT a ParameterSpaceError, and its text names
        # both exits whichever was called.  Declaring require_convergence: null
        # turns it off, so this pair pins that the declared key travelled: an
        # executor that dropped it raises on the second call, one that
        # hard-coded null never raises on the first.
        #
        # 1e-3 is DECLARED rather than defaulted: the shipped default became
        # null when kappa became a bound (inference/linear.py::_condition_bound),
        # so leaving it out would make both halves of this pair the same call.
        with pytest.raises(eqx.EquinoxRuntimeError,
                           match="wiener_solve/gcr_sample"):
            run_document(two_latent_document(
                {**WIENER, "names": ["dep", "c"], "require_convergence": 1e-3}))
        product = run_product(two_latent_document(
            {**WIENER, "names": ["dep", "c"], "require_convergence": None}))
        assert float(product["mean"]["dep"]) == pytest.approx(1.0, abs=1e-3)

    def test_maxiter_reaches_the_solve(self):
        # One CG iteration on the two-latent block leaves dep at -2.7e-05
        # where the converged answer is 0.9999954, so a maxiter: that never
        # arrived is a four-order-of-magnitude difference, not a rounding one.
        product = run_product(two_latent_document(
            {**WIENER, "names": ["dep", "c"], "require_convergence": None,
             "maxiter": 1}))
        assert abs(float(product["mean"]["dep"])) < 0.01
        # The residual's VALUE, not just its type.  Measured 8.36e-05 here
        # against 8.9e-08 converged, 1.4e-07 for the sub-block and 0.0 for
        # the one-latent document -- so a hard-coded constant that satisfied
        # `< 1e-5` everywhere else cannot satisfy this.  Task 11's predict
        # and the report layer both read this field.
        assert product["residual"] == pytest.approx(8.36e-05, rel=0.05)

    def test_tol_reaches_the_solve(self):
        # The one _SOLVER_KNOBS member whose forwarding nothing else can see:
        # maxiter and require_convergence each have a test that dies when
        # they are dropped from the call, and tol had only _number's
        # coercion, which runs whether or not the value travels.
        #
        # Measured on the two-latent block: tol: 2.0 is a tolerance CG meets
        # before its first iteration, so both latents come back at exactly
        # 0.0 -- not the init (dep 0.5, c 0.01), the CG zero start -- with a
        # relative residual of exactly 1.0.  Dropped, tol falls back to the
        # package's 1e-6 and dep converges to 0.9999954.
        #
        # 2.0, NOT 1.0, and the margin is the point.  cg stops when
        # ||r||^2 <= tol^2 ||b||^2; at the zero start r == b, so the rule is
        # 1 <= tol^2 and tol: 1.0 sits on EXACT equality.  Measured: tol 0.9
        # runs one iteration and returns numbers byte-identical to the
        # maxiter: 1 document above -- so on the threshold, a change from <=
        # to < in jax's stopping rule would drop this test into that branch,
        # where it fails AND stops being distinguishable from its neighbour.
        # boundary-validation.md: do not park a lever on the boundary.
        product = run_product(two_latent_document(
            {**WIENER, "names": ["dep", "c"], "require_convergence": None,
             "tol": 2.0}))
        assert float(product["mean"]["dep"]) == pytest.approx(0.0, abs=1e-6)
        assert float(product["mean"]["c"]) == pytest.approx(0.0, abs=1e-6)
        assert product["residual"] == pytest.approx(1.0, rel=1e-3)

    def test_a_non_numeric_solver_knob_is_a_config_refusal(self):
        # Without the coercion these reach jax and come back as bare
        # TypeErrors, breaking the layer's single-ConfigError contract.  All
        # THREE knobs, because _SOLVER_KNOBS is a table and a member dropped
        # from it forwards raw.
        for options, key in (({"maxiter": "many"}, "maxiter"),
                             ({"tol": True}, "tol"),
                             ({"require_convergence": "loose"},
                              "require_convergence")):
            with pytest.raises(ConfigError, match=f"{key}: is a number"):
                run_document(wiener_document({**WIENER, **options}))

    def test_a_solver_knob_below_its_floor_is_refused(self):
        # maxiter: 0 is not "no cap" -- the package spells that None -- it is
        # a solve that never iterates, and it returns the init in silence.
        with pytest.raises(ConfigError, match=r"maxiter: must be >= 1"):
            run_document(wiener_document({**WIENER, "maxiter": 0}))
        with pytest.raises(ConfigError, match=r"tol: must be >= 0"):
            run_document(wiener_document({**WIENER, "tol": -1.0}))

    def test_null_is_off_only_where_the_package_spells_it_that_way(self):
        # _SOLVER_KNOBS carries a nullable flag per key and nothing else
        # reads it: maxiter: null is "no cap" and require_convergence: null
        # is "no guard" (both `X | None = None` in the signature), but tol is
        # `tol: float = 1e-06` and has no off.  A tol: null waved through
        # reaches the CG loop as a comparison against None.
        with pytest.raises(ConfigError, match="tol: is a number"):
            run_document(wiener_document({**WIENER, "tol": None}))
        product = run_product(wiener_document({**WIENER, "maxiter": None}))
        assert float(product["mean"]["g"]) == pytest.approx(TRUTH_G, abs=1e-4)


class TestWidth:
    def test_width_is_required_by_name(self):
        run = {key: value for key, value in WIENER.items() if key != "width"}
        with pytest.raises(ConfigError, match="width") as caught:
            run_document(wiener_document(run))
        message = str(caught.value)
        # The refusal has to say the key is MISSING, not merely that some
        # width is wrong: a default quietly substituted upstream would make
        # this branch unreachable and the run would return a product.
        assert "width: is required and has no default" in message
        assert "declares None" in message

    def test_the_refusal_is_prefixed_with_the_runs_name(self):
        # `where` is the exits.py spelling, f"runs[{run.name!r}]" -- runs.py's
        # own _one uses the INDEX form for the same prefix, and the two must
        # never be confused: an index points at a position in a list a reader
        # has to count out, and stops matching as soon as a run is inserted
        # above it.  A run NAMED apart from its kind separates three
        # spellings that agree under WIENER: the name, the kind, and
        # "runs[0]".
        document = wiener_document({**WIENER, "name": "posterior",
                                    "width": "covariance"})
        with pytest.raises(ConfigError) as caught:
            run_document(document)
        assert str(caught.value).startswith("runs['posterior']: width:")

    def test_an_unknown_width_names_the_three(self):
        with pytest.raises(ConfigError, match="width") as caught:
            run_document(wiener_document({**WIENER, "width": "covariance"}))
        message = str(caught.value)
        # The whole vocabulary, including draws -- which is refused by the
        # branch below and must still be NAMED here, or a reader who wanted
        # draws is told it does not exist rather than where it lives.
        for name in ("'none'", "'draws'", "'fisher'"):
            assert name in message
        assert "'covariance'" in message

    def test_width_none_records_the_absence_rather_than_omitting_it(self):
        product = run_product(wiener_document(WIENER))
        assert product["width"] == "none"
        assert "covariance" not in product
        assert "fisher" not in product

    def test_width_draws_names_conjugate_gcr(self):
        # PERMANENT.  width: draws is refused for the whole of 2C (plan
        # section 0's revised decision): it would be a second spelling of
        # kind: conjugate.gcr, running the same solver at the same seed for
        # the same draws, and A29 would need a branch to survive it.  No
        # later task deletes this test.
        with pytest.raises(ConfigError, match="conjugate.gcr") as caught:
            run_document(wiener_document({**WIENER, "width": "draws"}))
        message = str(caught.value)
        # draws is a KNOWN width refused on purpose, not an unknown one:
        # dropping it from _WIDTHS would refuse it too, in a message that
        # tells the reader the word does not exist.
        assert "width: is required and has no default" not in message
        assert "n_draws" in message and "seed" in message

    def test_width_fisher_returns_the_posterior_covariance(self):
        # Measured under the TIGHT prior (scale 0.005): the posterior sigma
        # is 0.00476898, while the likelihood-only Fisher -- which is what
        # fisher_information returns when space= is dropped -- gives
        # 0.01587247 and reports kind "covariance".  Both assertions kill
        # "space= was never passed"; the sigma also kills "the covariance is
        # the Fisher un-inverted".
        product = run_product(wiener_document(
            {**WIENER, "width": "fisher"}, parameters={"g": TIGHT_GAIN}))
        assert product["width"] == "fisher"
        assert product["covariance"].kind == "posterior_covariance"
        assert float(product["covariance"].sigma("g")) == \
            pytest.approx(0.00476898, rel=1e-3)
        # The mean is still the product's: a width is bought BESIDE it, not
        # instead of it.  Under this prior the mean is pulled well off the
        # truth (1.0451 measured), which is the point of the tight prior.
        assert float(product["mean"]["g"]) == pytest.approx(1.0451, rel=1e-3)
        assert product["fisher"] is not None

    def test_width_fisher_over_a_partial_block_is_refused(self):
        # The Fisher is the whole space's.  Over a sub-block it would
        # marginalise the latents the solve held FIXED and report the answer
        # under this run's name -- a wider number that reads as this block's
        # own width.
        with pytest.raises(ConfigError, match="width: fisher") as caught:
            run_document(two_latent_document(
                {**WIENER, "width": "fisher", "names": ["dep"],
                 "require_convergence": None}))
        message = str(caught.value)
        # Which side is which: the difference taken the other way round is
        # empty here, so an implementation that computed it backwards would
        # return a product, and one that reported the BLOCK as the excess
        # would name the wrong latent to widen.
        assert "groups ['dep']" in message
        assert "['c']" in message
        # The second refusal built from `where`, so it is pinned here too.
        assert message.startswith("runs['conjugate.wiener']: width: fisher")

    def test_width_fisher_over_a_whole_multi_latent_space_is_allowed(self):
        # The passing counterpart of the refusal above: the same document and
        # the same key, with names: widened to the whole space.  Without it
        # the guard could refuse EVERY block of more than one and no test
        # would notice.  Measured: sigma(dep) = 0.0115564 and
        # sigma(c) = 2.29834e-05, five orders apart, so a covariance filed
        # against the wrong member is visible rather than plausible.
        product = run_product(two_latent_document(
            {**WIENER, "width": "fisher", "names": ["dep", "c"],
             "require_convergence": None}))
        covariance = product["covariance"]
        assert covariance.kind == "posterior_covariance"
        assert float(covariance.sigma("dep")) == pytest.approx(0.0115564,
                                                               rel=1e-3)
        assert float(covariance.sigma("c")) == pytest.approx(2.29834e-05,
                                                             rel=1e-3)

    def test_width_fisher_under_check_false_on_a_curved_latent(self):
        # The combination _gaussian_width's docstring argues about: check:
        # false is what lets a latent through that the prediction is NOT
        # affine in.  Measured: it runs, and reports sigma(c) = 10.0, which
        # is the prior scale to the digit -- at the linearization point the
        # data constrains a Gaussian's CENTRE not at all, so the posterior
        # width is the prior width.  A number that came back below the prior
        # would be the likelihood claiming information it has not got.
        #
        # What this does NOT pin, despite the neighbourhood: the CHOICE of
        # evaluation point.  Measured -- taking the Fisher at zeros instead
        # of at the solved mean gives the same 10.0, because the likelihood
        # contributes numerically nothing at EITHER point.  So this test dies
        # when space= is dropped (the prior is load-bearing) and lives when
        # the point moves.  _gaussian_width's docstring is right that no test
        # pins the point; a task that ever needs one must find a curved
        # latent whose Jacobian actually differs between init and mean.
        product = run_product(_declined(wiener_document(
            {**WIENER, "names": ["c"], "width": "fisher", "check": False},
            parameters={"c": CENTRE_LATENT}, at={"c": 76.0})))
        assert product["covariance"].kind == "posterior_covariance"
        assert float(product["covariance"].sigma("c")) == pytest.approx(
            10.0, rel=1e-3)

    def test_width_fisher_needs_the_prior_declared_not_supplied(self):
        # prior_std: drives the SOLVE; fisher_information reads the
        # ParameterSpace.  With no prior: declared the two cannot agree, and
        # the package refuses in its own voice rather than returning a matrix
        # that is part likelihood and part posterior.
        with pytest.raises(ParameterSpaceError, match="declares no prior"):
            run_document(wiener_document(
                {**WIENER, "width": "fisher", "prior_std": 10.0},
                parameters=PRIOR_FREE))


class TestTheNoiseSeam:
    def test_a_radiometer_model_names_both_routes(self):
        # Check A27, reached through the EXECUTOR: the helper's own test
        # above drives _decided_sigma directly, which passes just as well
        # against an executor that never routed through _conjugate_block.
        # The refusal must name BOTH ways out, because they are different
        # answers to different questions: keep the estimator and freeze the
        # sigma (radiometer_frozen), or keep the noise model and change the
        # estimator (conjugate.gls).
        with pytest.raises(ConfigError) as caught:
            run_document(wiener_document(WIENER, noise=RADIOMETER))
        message = str(caught.value)
        assert "conjugate.gls" in message
        assert "radiometer_frozen" in message
        assert "check A27" in message

    def test_radiometer_frozen_is_the_route_that_works(self):
        # Measured: g = 1.5 against the injected 1.5.  A frozen radiometer
        # sigma is per-sample, so this also proves an ARRAY reached
        # noise_std= and not a NoiseModel, which linear.py:1031 refuses.
        product = run_product(wiener_document(WIENER, noise=FROZEN))
        assert float(product["mean"]["g"]) == pytest.approx(TRUTH_G, abs=1e-4)

    def test_noise_kind_none_is_refused_naming_the_legal_exits(self):
        doc = wiener_document(WIENER)
        doc["inference"]["noise"] = {"kind": "none"}
        with pytest.raises(ConfigError, match="forward and optimize"):
            run_document(doc)


class TestThePrior:
    def test_a_scalar_prior_std_over_several_latents_is_check_a51(self):
        # A51, through the executor.  A ConfigError rather than the package's
        # own ParameterSpaceError is half the assertion: the config layer can
        # see the block's member count and must refuse first, because the
        # package only gets its turn after the whole document is built.
        with pytest.raises(ConfigError, match="prior_std") as caught:
            run_document(two_latent_document(
                {**WIENER, "names": ["dep", "c"], "prior_std": 0.1,
                 "require_convergence": None}))
        assert "check A51" in str(caught.value)

    def test_a_scalar_prior_std_over_a_block_of_one_is_broadcast(self):
        # Measured: g = 1.4999962.  One latent leaves no ambiguity about
        # which member a lone number belongs to, so the mapping form is not
        # demanded -- but the package requires the mapping for a grouped
        # block, so the broadcast has to happen somewhere, and this is where.
        # PRIOR_FREE is what makes the width OBSERVABLE: against a latent
        # that declares its own prior:, _reconcile (linear.py:872) refuses a
        # disagreeing keyword and accepts an agreeing one in silence.
        product = run_product(wiener_document(
            {**WIENER, "prior_std": 10.0}, parameters=PRIOR_FREE))
        assert float(product["mean"]["g"]) == pytest.approx(TRUTH_G, abs=1e-4)

    def test_no_prior_anywhere_is_the_packages_own_refusal(self):
        with pytest.raises(ParameterSpaceError, match="needs a prior_std"):
            run_document(wiener_document(WIENER, parameters=PRIOR_FREE))


class TestCheckReachesLinearOperator:
    """check: is linear_operator's alone, and it is not decoration.

    check=True runs check_linearity, which probes the prediction at
    [0.001, 1, 1000] times each latent's own scale and refuses when the
    departure from the linearization exceeds rtol.  global_signal.centre is
    declared linear=True here and is not linear at all, so the two settings
    of one key give a refusal and a product -- which is the only way to
    observe a knob whose entire effect is a check that otherwise passes in
    silence.
    """

    def test_a_latent_that_is_not_linear_is_refused_by_the_exits_own_default(self):
        # names: ["c"] rather than WIENER's ["g"]: this document's only
        # latent is c, and names: has no default to fall back on.
        #
        # Renamed from ...refused_by_default: the subject is the EXIT key
        # check:'s default (True), which the document-level linearity gate
        # does not touch (this fixture declines it via _declined()) -- but
        # the document is no longer "at defaults" in the sense a reader took
        # from the old name, now that C12 is refuse-by-default too.
        with pytest.raises(ParameterSpaceError, match="JOINTLY"):
            run_document(_declined(wiener_document(
                {**WIENER, "names": ["c"]},
                parameters={"c": CENTRE_LATENT}, at={"c": 76.0})))

    def test_check_false_reaches_linear_operator(self):
        # Measured: the same document with check: false solves and lands at
        # c = 75.000008.  An executor that dropped check: from its sweep set
        # refuses the key outright; one that sent it to wiener_solve -- which
        # has no such parameter -- raises TypeError.  Neither produces this.
        product = run_product(_declined(wiener_document(
            {**WIENER, "names": ["c"], "check": False},
            parameters={"c": CENTRE_LATENT}, at={"c": 76.0})))
        assert float(product["mean"]["c"]) == pytest.approx(75.0, abs=1e-3)


class TestTheCheapChecksComeFirst:
    """A document broken in its grammar AND its model hears about the grammar
    -- WITHIN ``_run_conjugate``, and no longer at the top of the pipeline.

    **Since Plan 3A, the pre-flight pass is in front of both.**  A27 and A28
    are decided from two words of text before any executor exists, so a
    document that is wrong in its NOISE and in its per-kind grammar now hears
    about the noise first -- ``test_the_undecidable_sigma_is_now_heard_
    before_the_missing_width`` is that case and it inverted deliberately.
    What this class still pins is the order WITHIN ``_run_conjugate``, for
    the documents the pass does not decide:

    ``_run_conjugate`` runs the per-kind grammar -- ``width:`` for wiener,
    ``seed:``/``noise_from:``/``n_draws:`` for gcr -- and the solver-knob
    coercion BEFORE ``_conjugate_block``, which builds the operator and runs
    ``check_linearity`` (three probe scales per latent).  Every other test in
    these modules uses a document broken exactly ONE way, so not one of them
    can see that order.

    Measured on a draft of Task 4 that hoisted ``_conjugate_block`` above the
    per-kind half: the third case below came back a ParameterSpaceError from
    inside the package instead of a ConfigError -- a config-layer refusal
    turned into a package exception, which is the seam this layer exists to
    close.  The other two changed which refusal a doubly-broken document got.

    **CORRECTION, Task 4 fix round.**  This class's title used to be true of
    the WHOLE pipeline; it no longer is.  Document-level C12 (this plan's own
    ``linearity`` gate, ``refuse`` by default) now runs during
    ``load_document``, ahead of any run's own execution -- so on a document
    that declares a ``linear: true`` latent and does NOT decline C12, the
    document-level check_linearity pays its full cost and can refuse the
    document BEFORE ``_run_conjugate``'s cheap per-kind grammar is ever
    reached.  Measured on the smallest fixture, warm (excluding the first
    JAX-compile call):
    ``test_a_missing_width_is_heard_before_check_linearity``'s own document
    costs 8.1-8.4 ms through ``run_document`` with C12 on against 3.3-3.7 ms
    with it declined via :func:`_declined` -- roughly 2x, and the qualitative
    point stands regardless of the exact ratio on a given machine.
    ``A29``'s sibling test is unaffected -- A29 is pre-flight, still in front
    of everything -- and this class's OWN per-kind-grammar-before-
    check_linearity claim is unaffected
    for the RUNS it covers, because every test that reaches a genuinely
    nonlinear ``linear: true`` latent now declines the document gate via
    :func:`_declined` so the exit-level ``check:`` knob, which is what this
    class is actually about, is what gets exercised.  This is a consequence
    of the plan's own §0.1 decision to default C12 to ``refuse``, not a
    defect in this class or in Task 4; Task 7 carries the correction into the
    plan record.
    """

    def test_the_undecidable_sigma_is_now_heard_before_the_missing_width(self):
        """The order this document gets REVERSED with Plan 3A, deliberately.

        Until the pre-flight pass existed, ``_run_conjugate`` refused the
        missing ``width:`` at ``conjugate.py:127`` and A27 was never reached
        on a doubly-wrong document -- measured, the refusal named the width
        and said nothing about the sigma.  Both are real, and the sigma is
        the one the user cannot fix by reading the message they were given:
        ``width: none`` is one word and leaves the document still broken.
        The pass decides A27 from two words before any executor exists, so it
        is now first.  The width refusal is unchanged and still fires on a
        document whose noise is fine -- ``test_a_missing_width_is_heard_
        before_a_bad_solver_knob`` below is what keeps it alive.
        """
        with pytest.raises(ConfigError, match="check A27") as caught:
            run_document(wiener_document(NO_WIDTH, noise=RADIOMETER))
        assert "width: is required" not in str(caught.value)

    def test_a_missing_width_is_heard_before_a_bad_solver_knob(self):
        # Both keys wrong at once: _width runs before _knobs, so the width is
        # what comes back and the message says nothing about maxiter.
        with pytest.raises(ConfigError, match="width:") as caught:
            run_document(wiener_document({**WIENER, "width": "covariance",
                                          "maxiter": "many"}))
        assert "maxiter" not in str(caught.value)

    def test_a_missing_width_is_heard_before_check_linearity(self):
        # THE material one.  global_signal.centre is declared linear=True and
        # is not, so check_linearity refuses this document -- but only once
        # the operator is built, and a run that never named a width has a
        # grammar error the layer can see without building anything.
        with pytest.raises(ConfigError, match="width: is required"):
            run_document(_declined(wiener_document(
                {**NO_WIDTH, "names": ["c"]},
                parameters={"c": CENTRE_LATENT}, at={"c": 76.0})))

    def test_a_missing_seed_is_heard_before_check_linearity(self):
        # The same guarantee for the draw, and the reason A29 lives in
        # _gcr_plan rather than in _gcr_product: the two halves of the
        # executor must be equally cheap to refuse.
        document = gcr_document({"names": ["c"]}, prior=None,
                                parameters={"c": CENTRE_LATENT},
                                at={"c": 76.0})
        del document["runs"][0]["seed"]
        with pytest.raises(ConfigError, match="A29"):
            run_document(document)


def _explode(*args, **kwargs):
    raise AssertionError(f"an operator ran during a grammar refusal: {args!r}")


class TestRefusalsPrecedeTheOperator:
    """Plan 4A Task 8: a grammar refusal used to wait for the operator."""

    def test_a_bad_prior_shape_speaks_before_the_operator_is_built(
            self, monkeypatch):
        import rheplicant.inference as inference

        monkeypatch.setattr(inference, "linear_operator", _explode)
        monkeypatch.setattr(inference, "wiener_solve", _explode)
        with pytest.raises(ConfigError, match="block-diagonal"):
            run_document(wiener_document({**WIENER,
                                         "prior_std": {"ghost": 1.0}}))


class TestConditionDefaultsAreThePackagesOwn:
    """``iterations: 12`` is the package's default, pinned as a RESULT
    equivalence -- config restates no package default in the parsed views."""

    def test_an_undeclared_iterations_is_the_packages_twelve(self):
        run = {"kind": "condition", "names": ["g"]}
        implicit = run_product(conjugate_document(run), "condition")
        explicit = run_product(conjugate_document({**run, "iterations": 12}),
                               "condition")
        assert float(implicit) == float(explicit)

    def test_the_seed_is_optional_and_reproducible_by_name(self):
        run = {"kind": "condition", "names": ["g"],
               "seed": {"from": "runtime.seeds.kappa"}}
        first = run_product(conjugate_document(run, seeds={"kappa": 5}),
                            "condition")
        again = run_product(conjugate_document(run, seeds={"kappa": 5}),
                            "condition")
        assert float(first) == float(again)
        # ...and the unseeded run runs at all: A29 makes no demand here.
        unseeded = run_product(
            conjugate_document({"kind": "condition", "names": ["g"]}),
            "condition")
        assert float(unseeded) > 0.0
