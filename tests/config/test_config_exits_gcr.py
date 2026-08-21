"""``kind: conjugate.gcr`` end to end: a document's draws, and what decides them.

The third of the conjugate test modules, and the seam is the same mechanical
one the other two split on -- everything here goes through
:func:`run_document`, so the shared machine driven directly lives in
``test_config_conjugate_shared.py`` and the ``conjugate.wiener`` half in
``test_config_exits_conjugate.py``, which also owns what both kinds share:
the registry, the parse, and the order the executor runs its checks in.

Every number below was measured on :func:`gcr_document`'s document, whose
three widths -- exit_helpers' PRIOR_SIGMA, LIKELIHOOD_SIGMA and
POSTERIOR_SIGMA -- are made to differ visibly on purpose: a sampler that
ignored the prior, ignored the seed, or returned the wiener mean n times
scores a different number on at least one of them.
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections.runs import run_document
from rheplicant.core.errors import ParameterSpaceError
from tests.config.exit_helpers import (
    GCR_RADIOMETER,
    LIKELIHOOD_SIGMA,
    POSTERIOR_SIGMA,
    PRIOR_FREE,
    PRIOR_SIGMA,
    SIGMA_K,
    WIENER,
    gcr_document,
    gcr_product,
    wiener_document,
)


class TestGcrDraws:
    def test_the_product_is_draws_a_residual_and_the_sigma_they_used(self):
        product = gcr_product({"n_draws": 8})
        # The two products share no key but "residual" -- a report reading one
        # must branch on RunResult.kind -- so the WHOLE key set is the
        # assertion, not merely that "draws" is present.
        assert set(product) == {"draws", "residual", "noise_std", "gls"}
        assert product["draws"]["g"].shape == (8,)
        # gcr_sample's second element is the relative CG residual, and there
        # is ONE PER DRAW: keeping it is what makes a badly conditioned block
        # visible in the product rather than only in an eqx.error_if.  Its
        # VALUE is pinned by test_the_solver_knobs_reach_the_draw below, which
        # drives it to exactly 1.0 -- a hard-coded zeros() satisfies the bound
        # here and cannot satisfy that.
        assert product["residual"].shape == (8,)
        assert float(jnp.max(product["residual"])) < 1.0e-5
        # Full-shaped rather than scalar: noise_std= is per sample, and a
        # scalar reported here would be a different array from the one the
        # solve was actually given.
        assert product["noise_std"].shape == (16, 8)
        assert float(jnp.mean(product["noise_std"])) == pytest.approx(
            SIGMA_K, rel=1e-4)
        assert product["gls"] is None

    def test_n_draws_reaches_the_shape_and_defaults_to_one(self):
        # gcr_sample returns ONE draw and has no n_draws parameter, so this
        # key is the config layer's own vocabulary rather than a package
        # default restated -- and the vmap over split keys is the whole of it.
        many = gcr_product({"n_draws": 40})
        assert many["draws"]["g"].shape == (40,)
        assert many["residual"].shape == (40,)
        one = gcr_product()
        assert one["draws"]["g"].shape == (1,)
        assert one["residual"].shape == (1,)

    def test_n_draws_is_a_positive_whole_number(self):
        # Without _number's coercion these reach jax.random.split as a bare
        # TypeError from inside a trace, and n_draws: 0 is a vmap over an
        # empty axis -- a product with no draws in it, reported as a success.
        with pytest.raises(ConfigError, match="n_draws: is a number"):
            gcr_product({"n_draws": "many"})
        with pytest.raises(ConfigError, match=r"n_draws: must be >= 1"):
            gcr_product({"n_draws": 0})

    def test_the_named_seed_decides_the_draws(self):
        # Same document twice: bitwise the same eight draws.  A different
        # runtime.seeds.draws: eight different ones.  The SECOND half is what
        # kills a jax.random.key(0) that never calls seed_for -- the
        # reproducibility pair alone passes against it.
        run = {"n_draws": 8}
        first = gcr_product(run)
        again = gcr_product(run)
        moved = gcr_product(run, seeds={"draws": 12})
        assert np.array_equal(np.asarray(first["draws"]["g"]),
                              np.asarray(again["draws"]["g"]))
        assert not np.array_equal(np.asarray(first["draws"]["g"]),
                                  np.asarray(moved["draws"]["g"]))
        # And the eight are eight DIFFERENT draws, not one draw repeated:
        # jax.random.split is what makes them so, and a vmap over one key
        # broadcast n times returns a shape (8,) of identical numbers that
        # both assertions above accept.
        assert len(set(np.asarray(first["draws"]["g"]).tolist())) == 8

    def test_the_seed_name_is_the_entry_that_is_looked_up(self):
        # The test above moves the VALUE under a fixed name; this one moves
        # the NAME.  Both are needed: gcr_document hard-codes
        # runtime.seeds.draws, so no document above ever names a different
        # entry, and an implementation that ran _seed_name for its refusals
        # and then discarded the result --
        #     seed_for("draws", built.context)
        # -- passes every one of them, including the bare-integer refusal.
        #
        # What that costs is the failure draws.py:_key's own docstring says
        # nothing downstream can catch: runtime.seeds: {draws: 11,
        # posterior: 99} with seed: {from: runtime.seeds.posterior} would draw
        # from 11 while provenance.json recorded 99.  Both arrays are finite,
        # and a different sky is still a plausible sky.
        #
        # So: a run NAMING posterior in a document where posterior is 99 must
        # give the same eight draws as a run naming draws in a document where
        # DRAWS is 99 -- and must differ from the shipped seeds.draws of 11.
        named = gcr_product({"n_draws": 8,
                             "seed": {"from": "runtime.seeds.posterior"}},
                            seeds={"draws": 11, "posterior": 99})
        by_value = gcr_product({"n_draws": 8}, seeds={"draws": 99})
        shipped = gcr_product({"n_draws": 8})
        assert np.array_equal(np.asarray(named["draws"]["g"]),
                              np.asarray(by_value["draws"]["g"]))
        assert not np.array_equal(np.asarray(named["draws"]["g"]),
                                  np.asarray(shipped["draws"]["g"]))

    def test_the_scatter_is_the_posterior_width_not_the_prior_or_the_data(self):
        # 400 draws at runtime.seeds.draws = 11: mean 1.141650, sd 0.0086774
        # (measured).  The posterior here is N(1.14205, 0.0084608) exactly --
        # a N(1.0, 0.01) prior times a likelihood of width 0.015872 centred on
        # 1.5.  So a sampler that drew:
        #   * at the PRIOR's width scores 0.01           (+18%, and >= it)
        #   * at the LIKELIHOOD's width scores 0.015872  (+88%)
        #   * with prior_mean and prior_std swapped centres on ~1.5, not 1.142
        #   * the wiener mean n times scores sd 0.
        product = gcr_product({"n_draws": 400})
        draws = product["draws"]["g"]
        assert float(jnp.mean(draws)) == pytest.approx(1.14205, abs=2.0e-3)
        assert float(jnp.std(draws)) == pytest.approx(POSTERIOR_SIGMA,
                                                      rel=0.10)
        assert float(jnp.std(draws)) < PRIOR_SIGMA
        assert float(jnp.std(draws)) < LIKELIHOOD_SIGMA

    def test_the_solver_knobs_reach_the_draw(self):
        # `solve` is the dict _run_conjugate builds once and threads into BOTH
        # gcr_sample here and iterative_gls under noise_from: gls -- the same
        # object, not two coercions of the same keys.  This test watches the
        # gcr_sample end; TestNoiseFromGls's own knob test watches the other,
        # and neither can see the other's call.  Dropped from the gcr_sample
        # call, nothing else in this module notices: the draws still have the
        # right shape, the right seed and roughly the right scatter.
        #
        # Measured: tol: 2.0 is a tolerance CG meets before its first
        # iteration -- it stops when ||r||^2 <= tol^2 ||b||^2 and at the zero
        # start r == b -- so every draw comes back at exactly 0.0 with a
        # relative residual of exactly 1.0.  That second number is also what
        # kills a product whose "residual" is a hard-coded zeros().
        #
        # 2.0, NOT 1.0, and the margin is the point: at tol 1.0 the rule is an
        # exact equality, so the test would sit on the dispatch boundary and a
        # change from <= to < in jax's stopping rule would move it into the
        # one-iteration branch (boundary-validation.md).
        #
        # The first half is the package's own convergence guard firing on the
        # residual that loose tol leaves -- an eqx.error_if from inside jit,
        # so an EquinoxRuntimeError whose text names BOTH exits whichever was
        # called.  It says tol arrived; the second half says
        # require_convergence: null did too, and pins what the draw becomes.
        with pytest.raises(eqx.EquinoxRuntimeError,
                           match="wiener_solve/gcr_sample"):
            gcr_product({"n_draws": 4, "tol": 2.0})
        product = gcr_product(
            {"n_draws": 4, "tol": 2.0, "require_convergence": None})
        assert float(jnp.max(jnp.abs(product["draws"]["g"]))) < 1.0e-6
        assert float(jnp.min(product["residual"])) == pytest.approx(1.0,
                                                                    rel=1e-3)

    def test_the_compiled_prior_reaches_the_draw(self):
        # `prior` is compiled once and handed to BOTH gcr_sample and (under
        # noise_from: gls) iterative_gls.  Every other document in this class
        # declares its prior on the LATENT, where _prior_kwargs returns {} and
        # dropping the keywords from the call changes nothing at all; on a
        # PRIOR_FREE document the package refuses the bare draw outright
        # (_require_prior_std, linear.py:1009), so this is the one test that
        # can watch the compiled mapping travel.
        #
        # prior_std: 0.01 / prior_mean: 1.0 is TIGHT written as run keys, and
        # measured it reproduces the declared-prior route to every digit:
        # mean 1.141650, sd 0.0086774.  Taken the other way round -- 1.0 wide
        # centred on 0.01 -- the same document scores 1.498 +- 0.0162, so a
        # swap between the two keywords is visible rather than plausible.
        with pytest.raises(ParameterSpaceError,
                           match="gcr_sample needs a prior_std"):
            gcr_product({"n_draws": 4}, parameters=PRIOR_FREE, prior=None)
        product = gcr_product(
            {"n_draws": 400, "prior_std": PRIOR_SIGMA, "prior_mean": 1.0},
            parameters=PRIOR_FREE, prior=None)
        draws = product["draws"]["g"]
        assert float(jnp.mean(draws)) == pytest.approx(1.14165, abs=1.0e-4)
        assert float(jnp.std(draws)) == pytest.approx(POSTERIOR_SIGMA,
                                                      rel=0.05)


class TestGcrGrammar:
    def test_the_seed_is_required_and_says_a29(self):
        # Matched on "A29", not on "seed": the key sweep refuses an unknown
        # key and _seed_name's own message says "seed" too, so only "A29"
        # tells the deliberate refusal from either fallback.  Built by
        # DELETING the key from gcr_document's own run rather than by
        # rebuilding the document, which is how the two would drift apart.
        document = gcr_document({"n_draws": 4})
        del document["runs"][0]["seed"]
        with pytest.raises(ConfigError, match="A29") as caught:
            run_document(document)
        message = str(caught.value)
        assert message.startswith("runs['conjugate.gcr']: ")
        # The refusal has to say the key is MISSING and name the exits that
        # refuse one instead; A29 is a statement about both halves.
        assert "seed: is required and has no default" in message
        assert "conjugate.wiener" in message

    def test_the_deterministic_sibling_refuses_the_seed(self):
        # A29's other half, and the reason `seed` is _GCR_KEYS' own rather
        # than the family's: conjugate.wiener returns the posterior MEAN, so a
        # seed reaching it would be a declared key that decides nothing.
        with pytest.raises(ConfigError, match=r"does not take \['seed'\]"):
            run_document(wiener_document(
                {**WIENER, "seed": {"from": "runtime.seeds.draws"}}))

    def test_a_bare_integer_seed_is_refused_by_name(self):
        # draws.py:_seed_name's own text -- a literal seed appears in one
        # value node and nowhere else, so provenance.json cannot record it.
        with pytest.raises(ConfigError, match="runtime.seeds"):
            gcr_product({"seed": 7})

    def test_noise_from_is_declared_or_gls(self):
        with pytest.raises(ConfigError,
                           match="noise_from: is declared or gls") as caught:
            gcr_product({"noise_from": "frozen"})
        # The value it got, so the refusal is this branch's and not the
        # neighbouring stale-knob one, which also interpolates "noise_from".
        assert "'frozen'" in str(caught.value)

    def test_the_sweep_is_this_exits_own_key_set(self):
        # width: is conjugate.wiener's, and _GCR_KEYS is built from the SHARED
        # _SOLVE_KEYS rather than from _WIENER_KEYS for exactly this reason: a
        # width: accepted here would ask a draw for an error bar it already
        # is.
        with pytest.raises(ConfigError,
                           match=r"does not take \['width'\]") as caught:
            gcr_product({"width": "none"})
        message = str(caught.value)
        assert "kind: conjugate.gcr" in message
        # And the message must offer THIS exit's vocabulary back.
        for key in ("'n_draws'", "'seed'", "'noise_from'", "'max_reweights'"):
            assert key in message

    def test_unknown_keys_are_swept(self):
        with pytest.raises(ConfigError, match=r"\['n_draw'\]"):
            gcr_product({"n_draw": 4})


class TestNoiseFromGls:
    def test_gls_closes_the_radiometer_dead_end(self):
        # The whole point of noise_from: gls -- a document A27 refuses on the
        # declared route, drawn at the covariance iterative_gls converges to.
        # Measured on this document: 5 reweights to a sigma of mean 1.22073e-4
        # and 64 draws landing at 1.499984 +- 4.774e-5, some 180x tighter than
        # the declared-sigma route above, because the radiometer sigma at a
        # 0.5 K signal is 400x smaller than the 0.05 K this document declares
        # when homoscedastic.  A run that quietly kept the first iterate, or
        # the prior, cannot produce a scatter this size around a mean this
        # close to the truth.
        product = gcr_product({"noise_from": "gls", "n_draws": 64},
                                      noise=GCR_RADIOMETER)
        assert float(jnp.mean(product["noise_std"])) == pytest.approx(
            1.22073e-4, rel=1.0e-3)
        assert float(jnp.mean(product["draws"]["g"])) == pytest.approx(
            1.5, abs=1.0e-3)
        assert 0.0 < float(jnp.std(product["draws"]["g"])) < 1.0e-4

    def test_the_gls_diagnostics_come_back_as_python_scalars(self):
        # GLSResult's iterations/delta/converged are jax.Arrays.  `is True` is
        # the assertion a raw jax.Array fails: bool(x) is truthy but
        # `jnp.asarray(True) is True` is False, and a report that has to
        # serialise the product needs the Python scalar.
        product = gcr_product({"noise_from": "gls", "n_draws": 4},
                                      noise=GCR_RADIOMETER)
        assert set(product["gls"]) == {"iterations", "delta", "converged"}
        assert product["gls"]["converged"] is True
        assert product["gls"]["iterations"] == 5
        # The VALUE too, not merely the type: measured 7.947e-08, which is the
        # relative step the fifth reweight took.  A record that reported
        # GLSResult's `residual` under this name -- a plausible mix-up, since
        # both are small relative numbers off the same result -- lands three
        # orders away.
        assert product["gls"]["delta"] == pytest.approx(7.947e-08, rel=1e-3)
        assert type(product["gls"]["iterations"]) is int
        assert type(product["gls"]["delta"]) is float

    def test_a_prediction_dependent_sigma_alone_names_the_way_out(self):
        # A27 through THIS exit: _decided_sigma's own refusal names
        # kind: conjugate.gls and radiometer_frozen, and neither is the route
        # a gcr run should take -- gcr has a third way out that costs it
        # nothing, and the message has to be the one that says so.
        with pytest.raises(ConfigError, match="noise_from: gls") as caught:
            gcr_product({"n_draws": 4}, noise=GCR_RADIOMETER)
        message = str(caught.value)
        assert "check A27" in message
        assert "radiometer_frozen" in message

    def test_an_unconverged_covariance_must_be_acknowledged(self):
        # min_reweights == max_reweights == 2 stops the loop three steps short
        # of the fixed point: measured iterations 2, delta 1.9868e-06,
        # converged False.
        with pytest.raises(ConfigError,
                           match="acknowledge_unconverged_covariance") as got:
            gcr_product(
                {"noise_from": "gls", "n_draws": 4, "min_reweights": 2,
                 "max_reweights": 2}, noise=GCR_RADIOMETER)
        message = str(got.value)
        # The refusal QUOTES what the loop reached, so a reader can tell a cap
        # that was too low from a tolerance that was too tight.  Task 5's own
        # test pins the same two fields on its own document, so the :.4g is
        # load-bearing in both.
        assert "stopped after 2 reweights" in message
        assert "1.987e-06" in message

    def test_acknowledging_it_draws_anyway_and_records_the_false(self):
        product = gcr_product(
            {"noise_from": "gls", "n_draws": 4, "min_reweights": 2,
             "max_reweights": 2, "acknowledge_unconverged_covariance": True},
            noise=GCR_RADIOMETER)
        assert product["gls"]["converged"] is False
        assert product["gls"]["iterations"] == 2

    def test_the_acknowledgement_is_a_bool(self):
        with pytest.raises(ConfigError,
                           match="acknowledge_unconverged_covariance: is a "
                                 "bool"):
            gcr_product(
                {"noise_from": "gls", "n_draws": 4,
                 "acknowledge_unconverged_covariance": "yes"},
                noise=GCR_RADIOMETER)

    def test_reweight_tol_reaches_the_loop(self):
        # max_reweights is pinned by the unconverged pair above; reweight_tol
        # is the third member of _GLS_KNOB_SPECS and nothing else can see it
        # travel.  It has to be measured against a CONTROL rather than against
        # the shipped run, because the default min_reweights of 5 hides it
        # entirely: reweight_tol: 0.1 with min_reweights left alone still
        # takes five steps, the floor binding before the tolerance is
        # consulted.
        #
        # Measured, both at min_reweights: 1 -- three steps to a delta of
        # 3.179e-07 at the package's own reweight_tol (max(8 * eps, tol)), and
        # two steps to 1.987e-06 at 0.1.  BOTH fields move, and the control's
        # own 3 (against the shipped 5) is what says min_reweights travelled.
        control = gcr_product(
            {"noise_from": "gls", "n_draws": 4, "min_reweights": 1},
            noise=GCR_RADIOMETER)
        assert control["gls"]["iterations"] == 3
        assert control["gls"]["delta"] == pytest.approx(3.179e-07, rel=1e-3)
        loosened = gcr_product(
            {"noise_from": "gls", "n_draws": 4, "min_reweights": 1,
             "reweight_tol": 0.1}, noise=GCR_RADIOMETER)
        assert loosened["gls"]["converged"] is True
        assert loosened["gls"]["iterations"] == 2
        assert loosened["gls"]["delta"] == pytest.approx(1.9868e-06, rel=1e-3)

    def test_the_gls_knobs_without_noise_from_gls_are_refused(self):
        # A declared key that reaches nothing is the defect this whole plan is
        # written against: under noise_from: declared there is no loop to cap,
        # no tolerance to meet and no unconvergence to acknowledge.  All four
        # keys, one at a time, because the stale list is built from a tuple
        # and a member dropped from it goes quiet.
        for key, value in (("reweight_tol", 0.1), ("min_reweights", 2),
                           ("max_reweights", 2),
                           ("acknowledge_unconverged_covariance", True)):
            with pytest.raises(ConfigError,
                               match="noise_from: declared") as caught:
                gcr_product({"n_draws": 4, key: value})
            assert f"['{key}']" in str(caught.value)

    def test_the_solver_knobs_reach_the_gls_loop(self):
        # The OTHER end of the `solve` dict.  Every other test in this class
        # runs the shipped tol/maxiter/require_convergence, and the one test
        # that drives tol: 2.0 runs under noise_from: declared, where
        # _gls_result is never called -- so dropping **solve from the
        # iterative_gls call was invisible.  That is Task 3's tol defect
        # exactly (validated, never observed to travel), one exit along, and
        # it is Task 5's inheritance: kind: conjugate.gls, maxiter: 1 would
        # run CG to convergence and report a converged solution from a key
        # that reached nothing.
        #
        # Measured: tol: 2.0 collapses every inner solve to the CG zero start,
        # so the fixed point iterates on a prediction of zero and the
        # covariance it converges to is IDENTICALLY 0.0 -- against 1.22073e-4
        # for the same document at the shipped tol.  delta goes to 0.0 too,
        # from 7.947e-08.  require_convergence: null is needed for the same
        # reason as on the draw, and pins that it travelled here as well.
        product = gcr_product({"noise_from": "gls", "n_draws": 4, "tol": 2.0,
                               "require_convergence": None},
                              noise=GCR_RADIOMETER)
        assert float(jnp.mean(product["noise_std"])) == 0.0
        assert product["gls"]["delta"] == 0.0
        assert float(jnp.max(jnp.abs(product["draws"]["g"]))) == 0.0

    def test_the_compiled_prior_reaches_the_gls_loop(self):
        # The other end of `prior`, and the same blind spot: every document in
        # this class declares its prior on the LATENT, so _prior_kwargs
        # returns {} and dropping **prior from the iterative_gls call changes
        # nothing.  What it would cost is not a crash: the fixed point would
        # be found with NO regularisation, converge to a different noise_std,
        # and the draw would then be taken at a covariance that ignored the
        # prior while the draw itself used it -- finite, right-shaped, wrong.
        #
        # The first half pins that PRIOR_FREE really is prior-free ON THIS
        # ROUTE, and the match is "iterative_gls needs a prior_std", not the
        # bare phrase: _require_prior_std names its caller, so gcr_sample's
        # own refusal carries the same words and would satisfy a looser match
        # while proving nothing about the loop.
        with pytest.raises(ParameterSpaceError,
                           match="iterative_gls needs a prior_std"):
            gcr_product({"noise_from": "gls", "n_draws": 4},
                        parameters=PRIOR_FREE, prior=None,
                        noise=GCR_RADIOMETER)
        # And with it declared the loop runs: 5 reweights to the same
        # 1.220735e-4 the latent-declared document converges to, and 8 draws
        # at 1.5000052 +- 5.773e-05.
        product = gcr_product(
            {"noise_from": "gls", "n_draws": 8, "prior_std": PRIOR_SIGMA,
             "prior_mean": 1.0},
            parameters=PRIOR_FREE, prior=None, noise=GCR_RADIOMETER)
        assert product["gls"]["iterations"] == 5
        assert float(jnp.mean(product["noise_std"])) == pytest.approx(
            1.220735e-4, rel=1.0e-3)
        assert float(jnp.mean(product["draws"]["g"])) == pytest.approx(
            1.5, abs=1.0e-3)


class TestTheExecutorReadsOnlyTheParsedView:
    """Plan 4A Task 8: poisoning the raw mapping after parse changes nothing."""

    def test_poisoning_the_raw_options_after_parse_changes_nothing(self):
        from _rheplicant_bootstrap.variants import LayerRef
        from rheplicant.config.document import load_document
        from rheplicant.config.sections.exit_support import (
            handler_for,
            parse_run,
        )
        from rheplicant.config.sections.runs import parse_runs

        doc = gcr_document({"n_draws": 2})
        built = load_document(doc)
        (spec,) = parse_runs(doc["runs"])
        parsed = parse_run(spec, built, index=0,
                           layer=LayerRef(kind="base", name=None, prefix="",
                                          document={}, declared_runs=None))
        spec.options["n_draws"] = 5  # poison AFTER the parse
        product = handler_for("conjugate.gcr").execute(parsed, built, {})
        assert product["draws"]["g"].shape[0] == 2
