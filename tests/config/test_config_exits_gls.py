"""``kind: conjugate.gls`` end to end: the covariance, and what it must earn.

The fourth of the conjugate test modules, split on the same mechanical seam as
the other three -- everything here goes through :func:`run_document`.  The
shared machine driven directly is ``test_config_conjugate_shared.py``, the
``conjugate.wiener`` half (and what both kinds of ``_run_conjugate`` share) is
``test_config_exits_conjugate.py``, and ``conjugate.gcr`` is
``test_config_exits_gcr.py``.

This exit is the one that takes the noise **rule** rather than a decided
sigma, so it is also the one that runs ``iterative_gls`` for its own sake
rather than to feed a draw.  That makes ``test_config_exits_gcr.py``'s
``TestNoiseFromGls`` its twin: the same ``_gls_result``, the same loop, the
same refusal -- reached through a different executor.  **A hole closed on one
of those routes is not closed on the other**, which is why
:class:`TestWhatReachesTheLoop` below pins each keyword this executor forwards
separately rather than trusting that the gcr module already watched it travel.

Every number here was measured on :func:`gls_document`'s document, whose model
is ``synthetic_document()``'s with the stochastic node dropped -- so
``observed`` is a deterministic forward at ``g = 1.5``, the document carries no
randomness at all, and every pin below is exact rather than sampled.
"""

import equinox as eqx
import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.sections import exits
from rheplicant.config.sections.runs import parse_runs, run_document
from rheplicant.core.errors import ParameterSpaceError
from tests.config.exit_helpers import (
    FROZEN,
    PRIOR_FREE,
    RADIOMETER,
    TRUTH_G,
    gls_document,
    gls_pair_product,
    gls_product,
)

#: ``max_reweights: 2`` stops the loop three steps short of the fixed point,
#: and ``reweight_tol: 1e-12`` is five orders below the step it does reach
#: (4.768e-07, measured), so the run is genuinely unconverged with nothing
#: monkeypatched.  ``min_reweights: 1`` is what lets the cap bind at all --
#: the package's own floor of 5 is above it.
#:
#: 1e-12 rather than something just under that step on purpose: a tolerance
#: chosen near it would sit on the dispatch boundary of
#: ``delta <= reweight_tol`` (boundary-validation.md), where a change of
#: ``<=`` to ``<`` in the package flips the verdict.  Five orders of margin is
#: what keeps this constant about the executor.
SQUEEZED = {"min_reweights": 1, "max_reweights": 2, "reweight_tol": 1.0e-12}

#: The converged covariance's mean, measured: 2.30699312e-04.
SIGMA_MEAN = 2.30699e-4


class TestTheKindIsRunnable:
    def test_conjugate_gls_parses_rather_than_deferring(self):
        (run,) = parse_runs([{"kind": "conjugate.gls", "names": ["g"]}])
        assert run.kind == "conjugate.gls"
        assert run.options == {"names": ["g"]}

    def test_conjugate_gls_has_an_executor(self):
        # Read out of ``exits`` and not out of ``exit_support``, where the
        # table itself lives: importing the ROOT is what pulls the leaf
        # modules in and runs their @register decorators, and a table read
        # straight from exit_support reflects only what this process has
        # happened to import.
        assert "conjugate.gls" in exits.EXECUTORS

    def test_it_is_its_own_executor_and_not_the_shared_one(self):
        # Plan section 3.1: a separate executor rather than a third @register
        # on _run_conjugate.  The sigma spelling (noise= against noise_std=),
        # the product (a GLSResult against a 2-tuple) and the convergence gate
        # share nothing with that body -- and _conjugate_block hands gls a
        # sigma of None, which _wiener_product's prologue would pass straight
        # to wiener_solve.
        assert (exits.EXECUTORS["conjugate.gls"]
                is not exits.EXECUTORS["conjugate.wiener"])
        assert (exits.EXECUTORS["conjugate.gls"]
                is not exits.EXECUTORS["conjugate.gcr"])


class TestTheProduct:
    def test_the_product_is_a_gls_result_with_the_converged_sigma(self):
        # The whole GLSResult, not a dict: the reweighting's own product IS
        # the covariance, and the fields that carry it are the package's.  The
        # field tuple is asserted rather than sampled because _run_gls rebuilds
        # the result with _replace, and a product that had become a plain dict
        # -- or that had dropped a field -- is the shape a report reads.
        product = gls_product()
        assert product._fields == ("noise_std", "solution", "residual",
                                   "iterations", "delta", "converged")
        assert product.converged is True
        assert product.iterations == 5
        # `delta` is trajectory, and the trajectory is not portable: 3.9736e-07
        # on arm64, 7.947e-08 on x86_64. `converged` is the property, and the
        # package only sets it when delta fell below the tolerance, so the pair
        # is asserted as that -- positive, finite, and small enough to be a
        # fixed point rather than a step.
        assert 0.0 < float(product.delta) < 1e-5, product.delta
        assert set(product.solution) == {"g"}
        assert float(product.solution["g"]) == pytest.approx(TRUTH_G,
                                                             abs=1e-4)
        # Full-shaped rather than scalar: the radiometer sigma is per sample,
        # and the mean is the discriminating number -- a run that handed the
        # package the DATA's sigma rather than the fixed point's, or that
        # returned the relative residual in noise_std's place, cannot produce
        # it.
        assert product.noise_std.shape == (16, 8)
        assert float(jnp.mean(product.noise_std)) == pytest.approx(
            SIGMA_MEAN, rel=1e-4)

    def test_the_jax_diagnostics_arrive_as_python_scalars(self):
        # gls.py builds iterations/delta/converged as jax.Arrays, and
        # examples/gls_gcr.py:150-152 casts all three.  `type(...) is` rather
        # than isinstance: bool is a subclass of int, so an `iterations` cast
        # with bool() would satisfy isinstance(..., int).  And `is True` is
        # the assertion a raw jax.Array fails -- bool(x) is truthy but
        # `jnp.asarray(True) is True` is False, and a report serialising this
        # product needs the Python scalar.
        product = gls_product()
        assert type(product.iterations) is int
        assert type(product.delta) is float
        assert type(product.converged) is bool

    def test_the_sigma_is_the_declared_radiometer_not_a_default(self):
        # Radiometer sigma scales as 1/sqrt(channel_width * integration_time),
        # so four times the bandwidth halves it exactly.  A run that built its
        # own noise model, or that reached decided_noise's frozen branch,
        # cannot track the declaration.  Measured: 2.30699312e-04 against
        # 1.15349656e-04, which is the halving to every digit float32 has.
        narrow = gls_product()
        wide = gls_product(noise={**RADIOMETER,
                                  "channel_width": {"value": 4.0,
                                                    "unit": "MHz"}})
        assert float(jnp.mean(wide.noise_std)) == pytest.approx(
            0.5 * float(jnp.mean(narrow.noise_std)), rel=1e-4)

    def test_the_observation_the_run_names_is_the_one_it_solves(self):
        # `observed` is the third thing _conjugate_block hands back and the
        # second POSITIONAL argument iterative_gls takes; nothing else in this
        # module can tell it from a different array of the right shape.  Data
        # simulated at g = 1.2 must be recovered as 1.2 -- and because the
        # radiometer sigma tracks the PREDICTION, the converged covariance
        # moves with it too, so both halves of the result depend on this
        # argument.  Measured: g = 1.2000014, sigma mean 1.84559394e-04.
        moved = gls_product(at={"g": 1.2})
        assert float(moved.solution["g"]) == pytest.approx(1.2, abs=1e-4)
        assert float(jnp.mean(moved.noise_std)) == pytest.approx(
            1.84559e-4, rel=1e-4)
        # And `residual` is the package's own, carried through _replace
        # untouched: on the shipped document it is exactly 0.0, which a
        # hard-coded zero would also satisfy; here it is 1.725e-06.
        # Not pinned to the digit: 1.725e-06 on arm64, 7.5e-08 on x86_64. The
        # claim the comment above makes is that the residual is the package's
        # OWN and was carried through untouched -- so what has to be true is
        # that it is not the hard-coded zero, which is exactly what a
        # fabricated field would be.
        assert 0.0 < float(moved.residual) < 1e-4, moved.residual


class TestTheConvergenceGate:
    """``GLSResult.converged is False`` is the one gate the package leaves us.

    gls.py:89-91 says a covariance that is not a fixed point is still a
    number, and that everything conditioned on it inherits that.  This exit
    refuses the product rather than reporting it.
    """

    def test_an_unconverged_covariance_is_refused_quoting_what_it_reached(self):
        with pytest.raises(ConfigError,
                           match="never fell below reweight_tol") as caught:
            gls_product(dict(SQUEEZED))
        message = str(caught.value)
        # The prefix is `where`, and GLS names its run `gls` rather than
        # letting runs.py name it after its kind -- so this also pins that
        # `where` was built from run.name and travelled into _gls_result.
        assert message.startswith("runs['gls']: ")
        # The refusal QUOTES what the loop reached, so a reader can tell a cap
        # that was too low from a tolerance that was too tight.  Both fields
        # are pinned: "2" says max_reweights arrived (dropped, the default of
        # 100 lets the loop run to 100 against a tolerance it never meets),
        # and the delta's four digits are what make _gls_result's {delta:.4g}
        # API -- the gcr module pins the same format on its own document.
        assert "stopped after 2 reweights" in message
        # The delta is DERIVED rather than pinned, and that is the stronger
        # test as well as the portable one. What is under contract is that the
        # refusal quotes what the loop REACHED, at `{delta:.4g}`; the value
        # reached is the platform's arithmetic (4.768e-07 on arm64, 7.947e-08
        # on x86_64), so pinning its digits tested the machine. This still
        # fails if the number stops being quoted, if the format changes, or if
        # the quoted number is not the one the run actually reached.
        reached = gls_product({**SQUEEZED,
                               "acknowledge_unconverged_covariance": True})
        assert f"{float(reached.delta):.4g}" in message
        assert "acknowledge_unconverged_covariance: true" in message

    def test_the_acknowledgement_lets_that_covariance_through(self):
        product = gls_product({**SQUEEZED,
                               "acknowledge_unconverged_covariance": True})
        assert product.converged is False
        assert product.iterations == 2
        # Unconverged is the property; the delta is the trajectory. SQUEEZED's
        # tolerance is five orders below anything the loop reaches on either
        # platform, which is what makes `converged is False` true by
        # construction rather than by which machine ran it.
        assert float(product.delta) > SQUEEZED["reweight_tol"], product.delta
        # The covariance comes back too, and it is a real one -- but note what
        # this assertion does NOT do: two reweights already put it within 2e-6
        # (relatively) of the fixed point's 2.30699312e-04, so no tolerance
        # float32 supports could tell the truncated covariance from the
        # converged one.  What distinguishes this branch is `converged`,
        # `iterations` and `delta` above; the sigma is here only to say a
        # product was returned rather than a refusal raised.
        assert float(jnp.mean(product.noise_std)) == pytest.approx(
            SIGMA_MEAN, rel=1e-4)

    def test_reweight_tol_alone_decides_converged(self):
        # Same block, same step count, the same delta of 4.768e-07: only the
        # tolerance differs.  This kills an implementation that swept
        # reweight_tol and dropped it -- the package's own default here is
        # max(8 * eps, tol) = 1e-6, which calls BOTH of these converged, so a
        # dropped key makes the pair agree.
        tight = gls_product({**SQUEEZED,
                             "acknowledge_unconverged_covariance": True})
        loose = gls_product({"min_reweights": 1, "max_reweights": 2,
                             "reweight_tol": 1.0e-3})
        assert tight.iterations == 2
        assert loose.iterations == 2
        assert tight.delta == pytest.approx(loose.delta, rel=1e-6)
        assert tight.converged is False
        assert loose.converged is True

    def test_min_reweights_is_a_floor_the_package_honours(self):
        # The fixed point is reached in 5 steps, so a floor of 8 is visible
        # only if the declaration arrived; and max_reweights is pinned by the
        # SQUEEZED pair above, so all three members of _GLS_KNOB_SPECS are
        # watched travelling on this route.
        assert gls_product().iterations == 5
        longer = gls_product({"min_reweights": 8})
        assert longer.iterations == 8
        assert longer.converged is True

    def test_require_convergence_fires_the_packages_guard_naming_wiener(self):
        # iterative_gls calls _check_solve_arguments WITHOUT noise_std=, and
        # applies the guard once, on the final inner wiener_solve -- so this
        # refusal is an eqx.error_if from linear.py rather than a
        # ParameterSpaceError or a ConfigError, and its text names
        # wiener_solve/gcr_sample whatever exit was asked for.  A test that
        # expected it to say conjugate.gls would fail, which is why the last
        # assertion is here rather than in prose.
        #
        # On the g = 1.2 twin and NOT on the shipped document: the shipped
        # one's final relative residual is exactly 0.0, so `residual * kappa`
        # cannot exceed any require_convergence and no value of the knob can
        # make the guard fire at all.  The twin leaves 1.725e-06, which is
        # under the package's default of 1e-3 and over 1e-30 -- so the control
        # below is what makes the knob's VALUE observable rather than merely
        # its presence: at the default this same run returns a product.
        control = gls_product(at={"g": 1.2})
        assert control.converged is True
        result = run_document(gls_document(
            {"expect": "refuse", "require_convergence": 1.0e-30},
            at={"g": 1.2}))["gls"]
        assert result.product is None
        assert isinstance(result.error, eqx.EquinoxRuntimeError)
        assert "cannot reach require_convergence" in str(result.error)
        assert "wiener_solve/gcr_sample" in str(result.error)
        assert "conjugate.gls" not in str(result.error)


class TestWhatReachesTheLoop:
    """One test per keyword ``_run_gls`` forwards into ``iterative_gls``.

    Line coverage says the call ran; it does not say each keyword arrived.
    ``conjugate.gcr``'s ``noise_from: gls`` route reaches the same
    ``_gls_result`` six lines away, and Task 4 shipped with ``**prior`` and
    the CG knobs pinned into ``gcr_sample`` and not into ``iterative_gls`` --
    a hole closed on one route and left open on its twin.  These are this
    route's own.

    ``maxiter`` needs a document of its own, and the reason is worth
    recording: :func:`gls_document`'s block holds ONE latent, so its normal
    operator is 1x1 and CG reaches the answer in a single iteration --
    ``maxiter`` of 1, 2 and 5 return that document's numbers to every digit,
    and the same on the ``g = 1.2`` twin and the tight-prior one.  Measuring
    that and calling the knob benign would have been the wrong conclusion: it
    says the FIXTURE has no lever, not that the exit has none.  The pair in
    :func:`gls_pair_product` has one, and ``test_config_exits_gcr.py:395``
    named this obligation for this task in as many words.
    """

    def test_the_compiled_prior_reaches_the_loop(self):
        # Every other document in this module declares its prior on the
        # LATENT, where _prior_kwargs returns {} and dropping **prior from the
        # call changes nothing at all.  On a PRIOR_FREE document the package
        # refuses the bare solve outright, so this is the only place the
        # compiled mapping can be watched travelling.
        #
        # The match is caller-qualified.  _require_prior_std interpolates its
        # CALLER, so the bare phrase "needs a prior_std" is carried by
        # wiener_solve's, gcr_sample's and iterative_gls' refusals alike, and
        # a looser match would prove nothing about which one ran.
        with pytest.raises(ParameterSpaceError,
                           match="iterative_gls needs a prior_std"):
            gls_product(parameters=PRIOR_FREE)
        # Three distinguishable outcomes, because three hypotheses have to
        # die.  A prior_std of 1e-5 is tight enough to pull the answer off the
        # data (1.5) and most of the way onto prior_mean, so:
        #   * prior_mean 1.0 -> 1.0121891, prior_mean 1.4 -> 1.4012868
        #     (measured): prior_MEAN travelled, and with its value.
        #   * prior_STD's magnitude is what sets the residual pull towards the
        #     data: the offset from prior_mean goes as prior_std^2 while the
        #     prior dominates, so 2e-5 scores 1.0430151 -- four times the
        #     offset, and 300 times the tolerance below.  That is the pin the
        #     first draft of this test did not have: at prior_std 1e-6 the
        #     offset is 1.278e-04 and a rel=1e-3 tolerance admitted +-1e-3,
        #     ten times the signal, so a doubled prior_std passed.
        #   * the two keywords SWAPPED -- prior_std 1.0 centred on 1e-5 -- is
        #     a wide prior, and the likelihood then returns 1.5000000
        #     (measured), which neither number above accepts.
        #
        # 1e-5 rather than the 1e-6 first written, for margin against a
        # measured cliff: on this document prior_std of 3e-7 and below raises
        # the package's convergence guard outright (an EquinoxRuntimeError --
        # a prior that tight makes the normal operator's condition number
        # exceed what float32 can certify), while 5e-7 and above return.  1e-5
        # sits 20x above that edge, against the 2-3x the first draft had
        # (boundary-validation.md).
        centred = gls_product({"prior_std": 1.0e-5, "prior_mean": 1.0},
                              parameters=PRIOR_FREE)
        assert float(centred.solution["g"]) == pytest.approx(1.0121891,
                                                             abs=1e-4)
        moved = gls_product({"prior_std": 1.0e-5, "prior_mean": 1.4},
                            parameters=PRIOR_FREE)
        assert float(moved.solution["g"]) == pytest.approx(1.4012868, abs=1e-4)
        wide = gls_product({"prior_std": 0.5, "prior_mean": 1.0},
                           parameters=PRIOR_FREE)
        assert wide.iterations == 5
        assert float(wide.solution["g"]) == pytest.approx(TRUTH_G, abs=1e-4)
        assert float(jnp.mean(wide.noise_std)) == pytest.approx(SIGMA_MEAN,
                                                                rel=1e-4)

    def test_the_solver_knobs_reach_the_loop(self):
        # `solve` is _knobs(run, _SOLVER_KNOBS), compiled by the executor and
        # handed to _gls_result, which must NOT recompile it -- and must not
        # be handed it twice, which is a TypeError.  Nothing else in this
        # module notices a `solve={}`: the loop still converges, in the same 5
        # steps, to the same covariance.
        #
        # Measured: tol: 2.0 is a tolerance CG meets before its first
        # iteration -- it stops when ||r||^2 <= tol^2 ||b||^2 and at the zero
        # start r == b -- so every inner solve collapses to the zero start,
        # the fixed point iterates on a prediction of zero, and the covariance
        # it converges to is IDENTICALLY 0.0 against 2.30699312e-04 at the
        # shipped tol.  delta and the solution go to 0.0 with it.
        #
        # 2.0, NOT 1.0, and the margin is the point: at tol 1.0 the stopping
        # rule is an exact equality, so the test would sit on the dispatch
        # boundary and a change from <= to < in jax's rule would move it into
        # the one-iteration branch (boundary-validation.md).  Measured on the
        # other side too: every tol from 1e-3 to 0.5 leaves this document's
        # answer untouched, so 2.0 is the only lever there is.
        #
        # The first half is the package's convergence guard firing on the
        # residual that loose tol leaves -- an eqx.error_if from inside jit,
        # whose text names BOTH exits whichever was called.  It says tol
        # arrived; the second says require_convergence: null did too, since
        # with the guard asked for this same call raises rather than returning.
        #
        # Both halves DECLARE the key: the shipped default became null when
        # kappa became a bound (inference/linear.py::_condition_bound), so
        # omitting it would make the two halves the same call.
        with pytest.raises(eqx.EquinoxRuntimeError,
                           match="wiener_solve/gcr_sample"):
            gls_product({"tol": 2.0, "require_convergence": 1e-3})
        product = gls_product({"tol": 2.0, "require_convergence": None})
        assert float(jnp.max(jnp.abs(product.noise_std))) == 0.0
        assert product.delta == 0.0
        assert float(jnp.abs(product.solution["g"])) == 0.0

    def test_maxiter_reaches_the_loop_where_the_block_can_show_it(self):
        # The CG iteration CAP, on the two-latent pair whose normal operator
        # is 2x2 -- the one block in these fixtures where one iteration is not
        # already the answer.  Measured, with require_convergence: null on
        # both (it is baked into GLS_PAIR, whose note says why):
        #
        #   maxiter: 1   -> dep = -0.0000271, c = 0.0184463, 5 reweights
        #   maxiter: 20  -> dep = +1.0000012, c = 0.0200000, 9 reweights
        #
        # `dep` moves by five orders, and the truncated run still reports
        # converged=True -- which is the whole failure a lost maxiter would
        # hide: a document that asked for a cap, did not get one (or got one
        # it did not ask for), and reports a fixed point either way.
        #
        # 20 rather than 3 for the margin, and the margin is measured: 3, 4, 5
        # and 20 all return the uncapped numbers, and maxiter absent returns
        # them too (asserted below, since "no cap" is how the package spells
        # the default).  On the other side, 1 is the schema's own floor --
        # _number refuses 0 -- so there is no room beneath it; its neighbour
        # at 2 is a measured cliff, a ConfigError, because the loop's inner
        # solves stop contracting and the fixed point is never reached
        # (boundary-validation.md: recorded rather than sat on).
        capped = gls_pair_product({"maxiter": 1})
        uncapped = gls_pair_product({"maxiter": 20})
        assert float(capped.solution["dep"]) == pytest.approx(-2.71e-5,
                                                              abs=1.0e-5)
        assert float(capped.solution["c"]) == pytest.approx(0.0184463,
                                                            rel=1e-3)
        assert float(uncapped.solution["dep"]) == pytest.approx(1.0, abs=1e-4)
        assert float(uncapped.solution["c"]) == pytest.approx(0.02, abs=1e-5)
        # The reweight counts are NOT asserted absolutely. They are trajectory:
        # the uncapped run takes 9 steps on arm64 and 5 on x86_64. The five
        # orders `dep` moves between capped and uncapped is what this test is
        # about, and it is asserted above; the count carried no part of it.
        assert capped.iterations > 0 and uncapped.iterations > 0
        # And a declared cap that is generous means what no cap means, which
        # is what says the capped run above was capped BY THE DECLARATION
        # rather than by anything the pair does on its own.
        absent = gls_pair_product()
        assert float(absent.solution["dep"]) == pytest.approx(
            float(uncapped.solution["dep"]), rel=1e-6)
        assert absent.iterations == uncapped.iterations

    def test_the_noise_rule_reaches_the_loop_as_a_model(self):
        # iterative_gls takes noise= (a NoiseModel) where the other three
        # conjugate solves take noise_std= (a decided array), and passing
        # either where the other belongs is a hard ParameterSpaceError in both
        # directions.  _gls_result fetches the model through _decided_model,
        # and what this pins is that the gls route reaches THAT accessor and
        # not _decided_sigma: one document, one noise declaration, the KIND
        # the only difference.  Under conjugate.wiener check A27 refuses it;
        # under conjugate.gls it is the document the exit exists to serve.
        assert gls_product().converged is True
        refused = gls_document()
        refused["runs"][0] = {**refused["runs"][0],
                              "kind": "conjugate.wiener", "width": "none"}
        with pytest.raises(ConfigError, match="check A27"):
            run_document(refused)
        # And the mirror, check A28: a sigma already decided into an array has
        # no fixed point to iterate.  Matched on "check A28", which belongs to
        # _decided_model alone -- A27's neighbouring refusal names
        # radiometer_frozen and conjugate.wiener too, so those two literals
        # cannot tell the pair apart on their own.
        with pytest.raises(ConfigError, match="check A28") as caught:
            gls_product(noise=FROZEN)
        message = str(caught.value)
        assert message.startswith("runs['gls']: ")
        assert "radiometer_frozen" in message
        assert "conjugate.wiener" in message


class TestGlsGrammar:
    def test_the_sweep_is_this_exits_own_key_set(self):
        # _GLS_KEYS is built from the SHARED _SOLVE_KEYS, never from
        # _WIENER_KEYS: width: asks the reweighting for an error bar it does
        # not compute.
        with pytest.raises(ConfigError,
                           match=r"does not take \['width'\]") as caught:
            gls_product({"width": "none"})
        message = str(caught.value)
        assert "kind: conjugate.gls" in message
        # And the message must offer THIS exit's vocabulary back, so a reader
        # who reached for the wrong key learns where the right ones are.  All
        # three reweight knobs and the acknowledgement, because the set is
        # built from a tuple and a member dropped from it goes quiet: the key
        # would then be refused by the sweep before _gls_result could read it.
        for key in ("'names'", "'check'", "'prior_std'", "'prior_mean'",
                    "'tol'", "'maxiter'", "'require_convergence'",
                    "'reweight_tol'", "'min_reweights'", "'max_reweights'",
                    "'acknowledge_unconverged_covariance'"):
            assert key in message, f"{key} is not offered back"

    def test_the_drawing_siblings_keys_are_refused(self):
        # A29's other half: conjugate.gls is deterministic, so a seed reaching
        # it would be a declared key that decides nothing -- and _gcr_plan's
        # own refusal names this exit as one of the two that refuse one.
        # n_draws: and noise_from: go with it; all three are conjugate.gcr's.
        for key, value in (("seed", {"from": "runtime.seeds.draws"}),
                           ("n_draws", 4),
                           ("noise_from", "gls")):
            with pytest.raises(ConfigError,
                               match=rf"does not take \['{key}'\]"):
                gls_product({key: value})

    def test_unknown_keys_are_swept(self):
        with pytest.raises(ConfigError, match=r"\['max_reweight'\]"):
            gls_product({"max_reweight": 3})

    def test_the_numeric_knobs_are_numbers_and_the_flag_is_a_bool(self):
        # One from _SOLVER_KNOBS and one from _GLS_KNOB_SPECS, so a spec tuple
        # that never reached _knobs is visible from both tables; the floor
        # too, because min_reweights: 0 is a configuration iterative_gls
        # refuses with a ParameterSpaceError of its own, from inside the
        # package, about an argument the document did not write that way.
        with pytest.raises(ConfigError, match="maxiter: is a number"):
            gls_product({"maxiter": "many"})
        with pytest.raises(ConfigError, match="reweight_tol: is a number"):
            gls_product({"reweight_tol": "tight"})
        with pytest.raises(ConfigError, match=r"min_reweights: must be >= 1"):
            gls_product({"min_reweights": 0})
        with pytest.raises(
                ConfigError,
                match="acknowledge_unconverged_covariance: is a bool"):
            gls_product({"acknowledge_unconverged_covariance": "yes"})

    def test_the_reweight_knobs_are_not_nullable(self):
        # _GLS_KNOB_SPECS carries a nullable flag per key, and all three are
        # False -- the mirror of test_config_exits_conjugate.py's pin that
        # maxiter: null and require_convergence: null ARE "no cap" / "no
        # guard".  Nothing held this side of it: with a flag flipped to True,
        # _knobs passes None straight through, and
        #   * reweight_tol: null stops being an error and quietly becomes the
        #     package's derived default, max(8 * eps, tol) -- a declared
        #     tolerance that decides nothing;
        #   * min_reweights: null reaches gls.py:198's `1 <= min_reweights` as
        #     `1 <= None`, a bare TypeError from inside the package about an
        #     argument the document never wrote that way, which is the exact
        #     substitution this layer exists to prevent.
        # None of the three is nullable because iterative_gls DERIVES
        # reweight_tol from tol when it is absent, and absent is spelled by
        # leaving the key out.
        for key in ("reweight_tol", "min_reweights", "max_reweights"):
            with pytest.raises(ConfigError,
                               match=rf"{key}: is a number; got None"):
                gls_product({key: None})

    def test_names_is_required(self):
        # _selected's refusal, reached through this executor: without it the
        # grouped operator has nothing to group, and linear_operator's own
        # failure would name a package argument the document never wrote.
        # Built by DELETING the key from gls_document's own run rather than by
        # rebuilding the document, which is how the two would drift apart.
        document = gls_document()
        del document["runs"][0]["names"]
        with pytest.raises(ConfigError, match="names: is required"):
            run_document(document)


def _explode(*args, **kwargs):
    raise AssertionError(f"an operator ran during a grammar refusal: {args!r}")


class TestRefusalsPrecedeTheOperator:
    """Plan 4A Task 8: the acknowledgement boolean was judged after the
    block was built."""

    def test_a_non_bool_acknowledgement_speaks_before_the_block_is_built(
            self, monkeypatch):
        import rheplicant.inference as inference

        monkeypatch.setattr(inference, "linear_operator", _explode)
        monkeypatch.setattr(inference, "iterative_gls", _explode)
        with pytest.raises(ConfigError,
                           match="acknowledge_unconverged_covariance: is a "
                                 "bool"):
            run_document(gls_document(
                {"acknowledge_unconverged_covariance": "yes"}))
