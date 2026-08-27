"""``kind: npe`` end to end: the bank, the estimator, and what they forward.

Task 7 writes the two halves that come before training and drives them
DIRECTLY -- ``parse_runs`` still refuses ``kind: npe`` at this commit, so
``run_document`` cannot reach them.  Task 8 registers the executor, promotes
the kind, and everything it adds below goes through ``run_document``.

Every number here was measured against ``npe_document()`` at 54617ce: the
16 x 8 synthetic grid, one latent ``g`` into ``gain.gain``, data simulated at
``g = 1.5``, and ``runtime.seeds: {bank: 11, create: 12, train: 13,
sample: 14}``.  Under those seeds ``seed_for("bank", context)`` is the DECLARED
11 rather than a digest of the name, which is what lets the bank's own first
draws be pinned to a specific subsection's seed.

Task 8 registers the executor and promotes the kind, so everything from
:class:`TestTheTrainedPosterior` down goes through ``run_document`` -- the
route a user takes.  The bank and estimator tests above keep driving the two
helpers directly, because a test of the bank that does not pay for training is
worth keeping, and because a guard reached one way and not the other is the
hole-closed-on-one-route shape this plan has already shipped twice.
"""

import math
from functools import cache

import jax
import jax.numpy as jnp
import pytest

from rheplicant.config.document import load_document
from rheplicant.config.draws import seed_for
from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.exit_support import _observed
from rheplicant.config.sections.npe import _estimator, _npe_spec, _simulate_bank
from rheplicant.config.sections.runs import _KINDS, run_document
from rheplicant.inference.npe import MIN_SCALE
from tests.config.exit_helpers import FROZEN, PRIOR_FREE
from tests.config.inflight_helpers import built_run
from tests.config.posterior_helpers import (
    NPE_SECTION,
    NPE_SEEDS,
    joint_prior_document,
    npe_built,
    npe_document,
    npe_product,
    npe_spec,
    trio_npe_document,
)


@cache
def trio_product():
    """The trio document, executed ONCE for the three unravel assertions.

    A module-level cache rather than a fixture: the three tests below make
    three DIFFERENT claims about one product, and re-running an 18-dimensional
    estimator three times would put this module over the plan's budget on its
    own.  ``functools.cache`` rather than a module-level list appended to on
    first call, so that the memo is not a mutable global this file can be
    reached into and edited through.
    """
    return run_document(trio_npe_document())["amortized"].product


@cache
def joint_results():
    """The joint-prior document, executed ONCE -- it carries a nuts run.

    **Run in float64, and the document is BUILT inside the block too.** A
    declared ``joint_prior`` is ``sqrt(det I)`` evaluated at every leapfrog
    step, and its half-log-determinant is refused at single precision (D25 in
    the migration ledger): measured on an exactly degenerate block, float32
    gives -27.52 where the block honestly gives -338.05, 310 nats, in a term
    the sampler exponentiates and with a converged chain on the other side of
    it. ``to_numpyro_model`` says so at construction time and names the route,
    which is this one.

    Inside the block and not merely around the run: ``jax_enable_x64`` is a
    tracing-time global, so a space declared outside it carries float32
    constants into a float64 session, and that is refused separately and by a
    different sentence.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        document = joint_prior_document()
        # Written HERE and not in the builder. `runtime.jax_enable_x64` is
        # VERIFIED against the process at build time, so a builder that
        # declared it would be unbuildable in this suite's float32 session --
        # and the fixture census drives every `*_document` builder with no
        # arguments, so it would take two unrelated census tests down with it.
        # Measured, and it is the shape this file already lives with: the
        # projector helper writes `acknowledge_float32_sky` for the same
        # reason, that a document's declarations are constrained by who builds
        # it and not only by what it means.
        document["runtime"] = {**document["runtime"], "jax_enable_x64": True}
        return run_document(document)
    finally:
        jax.config.update("jax_enable_x64", was)


class TestTheBank:
    def test_the_pairs_come_back_at_the_declared_size_and_shape(self):
        # thetas is (n_simulations, n_latent_values) and data is
        # (n_simulations, *data.shape) -- npe.py:100-102.  An executor that
        # forwarded n_simulations to the wrong argument, or dropped it, cannot
        # produce a leading 64; one that banked against a state other than
        # built.state cannot produce 16 x 8.  The space comes back with them
        # because the columns of thetas are laid out in ITS names order and
        # Task 8 unravels the draws through the same object.
        built = npe_built()
        space, thetas, data = _simulate_bank(npe_spec(), built,
                                             built.inference.npe)
        assert thetas.shape == (64, 1)
        assert data.shape == (64, 16, 8)
        assert thetas.dtype == jnp.float32 and data.dtype == jnp.float32
        assert space is built.inference.space

    def test_n_simulations_reaches_the_package_and_changes_the_bank(self):
        # Forward coverage, not line coverage: a mutant that ignores
        # n_simulations sits on a line every other test in this class
        # executes.  Two DIFFERENT declared sizes are what it cannot survive.
        eight = npe_built(npe={"bank": {"n_simulations": 8}})
        sixteen = npe_built(npe={"bank": {"n_simulations": 16}})
        _, thetas8, data8 = _simulate_bank(npe_spec(), eight,
                                           eight.inference.npe)
        _, thetas16, _ = _simulate_bank(npe_spec(), sixteen,
                                        sixteen.inference.npe)
        assert thetas8.shape[0] == 8 and data8.shape[0] == 8
        assert thetas16.shape[0] == 16

    def test_the_bank_key_is_runtime_seeds_bank_and_not_another_subsection(
            self):
        # The four seeds are the one place a copy-paste is invisible: wiring
        # spec.create's seed into the bank leaves every shape, dtype and
        # finiteness assertion above green.  These three numbers are the first
        # draws simulate_pairs makes from jax.random.key(11), and 11 is
        # runtime.seeds.bank alone -- create is 12, train 13, sample 14, and
        # none of them reproduces this triple.
        built = npe_built(npe={"bank": {"n_simulations": 8}})
        assert seed_for("bank", built.context) == 11
        _, thetas, _ = _simulate_bank(npe_spec(), built, built.inference.npe)
        assert float(thetas[0, 0]) == pytest.approx(0.8553414344787598,
                                                    rel=1e-6)
        assert float(thetas[1, 0]) == pytest.approx(0.3442210555076599,
                                                    rel=1e-6)
        assert float(thetas[2, 0]) == pytest.approx(1.096588134765625,
                                                    rel=1e-6)

    def test_the_bank_key_comes_from_the_document_and_is_not_a_constant(self):
        # The test above pins three floats from key(11) and so discriminates
        # among the four SIBLING subsections -- but not against a bank that
        # reads the document not at all: measured, replacing the whole
        # _draw_key call with a literal jax.random.key(11) leaves every one of
        # tests/config green, because 11 is what the document says.  Its
        # `seed_for("bank", ...) == 11` line does not close that either: it
        # exercises draws.seed_for, not _simulate_bank.
        #
        # This is the mirror of test_the_create_key_is_runtime_seeds_create,
        # and it is here because the hole was closed on create and open on
        # its twin.  Two documents differing ONLY in runtime.seeds.bank must
        # bank different thetas.
        built = npe_built(npe={"bank": {"n_simulations": 8}})
        other = npe_built(npe={"bank": {"n_simulations": 8}},
                          seeds={**NPE_SEEDS, "bank": 88})
        assert seed_for("bank", other.context) == 88
        _, thetas, _ = _simulate_bank(npe_spec(), built, built.inference.npe)
        _, others, _ = _simulate_bank(npe_spec(), other, other.inference.npe)
        assert not jnp.array_equal(thetas, others)

    def test_a_frozen_sigma_is_refused_before_the_bank_is_simulated(self):
        # simulate_pairs takes noise= -- the RULE -- so the accessor is
        # _decided_model and never _decided_sigma.  Handed a decided array,
        # simulate_pairs dies inside a vmap as
        # `jax.errors.ConcretizationTypeError: ... The axis argument must be
        # known statically` (measured), naming no run, no key and no exit.
        #
        # The match is on `radiometer_frozen`, which is the part of
        # _decided_model's message true for every caller.  Plan 3A's Task 10
        # gave the rest of it a caller-supplied clause: this run is now told
        # that npe SIMULATES a bank, and is no longer offered
        # kind: conjugate.wiener, which is not an alternative to an amortized
        # posterior.  The positive assertion for that wording is
        # TestTheDecidedModelAccessor in tests/config/test_preflight_fitting.py;
        # what is pinned here is that the executor's own route still reaches
        # the accessor and still names this run.
        #
        # THE DOCUMENT'S RUN IS `forward` AND THE SPEC IS STILL `npe`.  Task
        # 10's pass refuses `kind: npe` beside a decided sigma at P-1, so
        # `npe_built(noise=FROZEN)` -- which is a `load_document` -- now
        # raises on the line that BUILDS the fixture rather than inside the
        # `raises` below, and this test would error instead of asserting.
        # `npe_spec()` fabricates the RunSpec independently of the document's
        # `runs:` (that is what it is for), so swapping the document's run
        # kind leaves the subject exactly what it was: `_simulate_bank`
        # reaching `_decided_model` and not `_decided_sigma`.
        built = npe_built({"kind": "forward"}, noise=FROZEN)
        with pytest.raises(ConfigError, match="radiometer_frozen") as caught:
            _simulate_bank(npe_spec(), built, built.inference.npe)
        assert str(caught.value).startswith("runs['amortized']: ")
        assert "SIMULATES a bank" in str(caught.value)
        assert "conjugate.wiener" not in str(caught.value)

    def test_a_prior_free_latent_is_refused_naming_the_sibling_exit(self):
        # _sampled_space(route="npe") is Task 4's, and this is the FIRST
        # caller that can reach its npe leg.  Task 8 tests the same guard
        # through run_document, which is the route a user takes; both legs
        # matter, because a guard reached one way and not the other is 2C
        # shape 4.  The match is on SIMULATES, which the nuts leg's message
        # does not contain (it says "draws a POSTERIOR"), so a neighbouring
        # refusal cannot satisfy it -- which is what makes this test able to
        # tell route="npe" from route="nuts" rather than merely observing
        # that something was refused.
        #
        # `expect: refuse` is what keeps this document LOADABLE.  Plan 3A's
        # Task 8 hoists the same property to P-1 as check A23, so
        # `load_document` on a prior-free npe run now refuses before
        # `npe_built` can return -- and `expect: refuse` is the run's own way
        # of saying the refusal is the point (`exits.py:293-303` captures it;
        # `_prior_gates` stands down on it for that reason).  The subject of
        # this test is unchanged: `_simulate_bank`'s route to
        # `_sampled_space`, driven directly.
        built = npe_built({"expect": "refuse"},
                          parameters={"g": {"init": 1.0, "linear": True,
                                            "into": "gain.gain"}})
        with pytest.raises(ConfigError, match="SIMULATES a bank") as caught:
            _simulate_bank(npe_spec(), built, built.inference.npe)
        message = str(caught.value)
        assert message.startswith("runs['amortized']: ")
        assert "inference.parameters.<name>.prior:" in message
        # And the advice must be TRUE OF THIS DOCUMENT.  This one declares no
        # joint_prior at all, so `kind: nuts` refuses it too -- measured, with
        # "declares ['g'] with no prior: and no inference.joint_prior covering
        # them".  Task 4's message makes the "or run kind: nuts" clause
        # conditional on `covered` for exactly this reason, and an assertion
        # that the string IS there would have locked in false advice.
        assert "or run kind: nuts" not in message
        assert "kind: nuts refuses this document too" in message


class TestTheEstimator:
    @staticmethod
    def _pairs(built):
        return _simulate_bank(npe_spec(), built, built.inference.npe)

    def test_the_declared_knobs_land_on_the_network(self):
        # The whole point of forwarding: n_components, width and depth are
        # each observable on the returned object, and the package's own
        # defaults are 4 / 64 / 3 (measured), so a create that dropped the
        # passthrough would report those instead.  out_size is
        # n_components * (1 + 2 * n_params) = 1 * 3, which is the one of the
        # four that a wrong n_components moves twice over; n_params is 1 and
        # not 128, which is what a create called (data, thetas) would report.
        built = npe_built(npe={"bank": {"n_simulations": 8}})
        _, thetas, data = self._pairs(built)
        estimator = _estimator(npe_spec(), built, built.inference.npe,
                               thetas, data)
        assert estimator.n_components == 1
        assert estimator.net.width_size == 16
        assert estimator.net.depth == 2
        assert estimator.net.out_size == 3
        assert estimator.n_params == 1

    def test_width_is_width_and_not_width_size(self):
        # create's parameter is width=; width_size= is what it passes
        # INTERNALLY to eqx.nn.MLP at npe.py:216.  A caller that spelled it
        # width_size= raises TypeError, so this test dies at the call rather
        # than at the assertion -- which is the point of writing it, because
        # 2C's carry-forward note says the opposite and is wrong about the
        # caller.  9 and not 16: the value has to differ from NPE_SECTION's
        # own so that a passthrough reading the wrong subsection is visible.
        built = npe_built(npe={"bank": {"n_simulations": 8},
                               "create": {"width": 9}})
        _, thetas, data = self._pairs(built)
        assert _estimator(npe_spec(), built, built.inference.npe,
                          thetas, data).net.width_size == 9

    def test_an_undeclared_knob_is_materialized_as_the_package_default(self):
        built = npe_built(npe={"bank": {"n_simulations": 8}})
        _, thetas, data = self._pairs(built)
        estimator = _estimator(npe_spec(), built, built.inference.npe,
                               thetas, data)
        assert estimator.min_scale == MIN_SCALE
        assert built.inference.npe.create["min_scale"] == MIN_SCALE

    def test_a_declared_min_scale_is_forwarded_and_not_dropped(self):
        # The other leg, and it is not optional.  min_scale is the one
        # _CREATE_OPTIONS entry no document in the suite declared, so
        # `estimator.min_scale == MIN_SCALE` above is exactly what a
        # passthrough that dropped min_scale from the tuple produces --
        # measured, that mutation left all of tests/config green.  One
        # assertion cannot do the work of two: the default leg proves the
        # package's default arrives, and this one proves the document's does.
        # 0.5 rather than a near-MIN_SCALE value so the two cannot be
        # confused by a comparison at any tolerance.
        built = npe_built(npe={"bank": {"n_simulations": 8},
                               "create": {"min_scale": 0.5}})
        _, thetas, data = self._pairs(built)
        estimator = _estimator(npe_spec(), built, built.inference.npe,
                               thetas, data)
        assert estimator.min_scale == 0.5

    def test_the_embed_reaches_create_and_resizes_the_input_layer(self):
        # embed is resolved to a CALLABLE by parse_npe, and the ONLY place it
        # is observable is the network it sized: create does
        # `features = jax.vmap(embed)(data)` (inference/npe.py:209) and takes
        # `in_size=features.shape[1]` (:214) -- the plan cited 210-212, which
        # is n_params/theta_mean/data_mean, and the citation was transcribed
        # from it verbatim.  jnp.ravel on (16, 8) is 128; jnp.diagonal is 8.
        # An executor that dropped embed= entirely still gets 128, which is
        # why the assertion that matters is on the NON-default -- and the
        # second half pins the default so the two cannot be confused.
        embedded = npe_built(npe={"bank": {"n_simulations": 8},
                                  "embed": {"python": "jax.numpy:diagonal"}})
        _, thetas, data = self._pairs(embedded)
        assert _estimator(npe_spec(), embedded, embedded.inference.npe,
                          thetas, data).net.in_size == 8
        plain = npe_built(npe={"bank": {"n_simulations": 8}})
        _, pt, pd = self._pairs(plain)
        assert _estimator(npe_spec(), plain, plain.inference.npe,
                          pt, pd).net.in_size == 128

    def test_the_create_key_is_runtime_seeds_create(self):
        # Two documents differing ONLY in runtime.seeds.create must give
        # different initial weights.  This is the leg that catches the bank
        # seed being reused for the network, which nothing about shape, dtype
        # or in_size can see -- and, with the bank test above, it is what
        # stops a SWAP of the two seeds passing Task 8's four-seed test.  The
        # SAME pairs go into both calls, so the only difference between the
        # two estimators is the key create was handed.
        built = npe_built(npe={"bank": {"n_simulations": 8}})
        other = npe_built(npe={"bank": {"n_simulations": 8}},
                          seeds={**NPE_SEEDS, "create": 88})
        assert seed_for("create", other.context) == 88
        _, thetas, data = self._pairs(built)
        first = _estimator(npe_spec(), built, built.inference.npe,
                           thetas, data)
        second = _estimator(npe_spec(), other, other.inference.npe,
                            thetas, data)
        assert not jnp.array_equal(first.net.layers[0].weight,
                                   second.net.layers[0].weight)


class TestTheSectionIsRequired:
    def test_the_parsed_section_is_what_reaches_the_helpers(self):
        # The other two tests in this class read _npe_spec's REFUSAL, and a
        # refusal-only pair cannot tell `return spec` from `return None`:
        # both helpers above take the spec as an argument, so nothing else in
        # this module executes the returning branch at all.  Task 8's
        # executor is what would have found it -- one task later, in a commit
        # about training.
        built = npe_built()
        assert _npe_spec(npe_spec(), built) is built.inference.npe
        assert built.inference.npe.bank["n_simulations"] == 64

    def test_a_document_with_no_inference_npe_is_refused_by_name(self):
        # Every knob kind: npe has lives in inference.npe:, so a document
        # without it declares no bank size, no seeds and no draw count.  The
        # refusal names the SECTION and lists its five subsections rather
        # than naming whichever key happened to be looked up first.
        #
        # Two boundaries, one refusal: Task 10's orchestration handler-parses
        # the schedule at LOAD, so the document itself never loads; the
        # payload route (``built_run``, no handler parse) still hands the
        # helper a build, and the helper's own refusal is unchanged.
        doc = npe_document()
        del doc["inference"]["npe"]
        built = built_run(doc)
        assert built.inference.npe is None
        with pytest.raises(ConfigError,
                           match="declares no inference.npe:") as caught:
            _npe_spec(npe_spec(), built)
        assert str(caught.value).startswith("runs['amortized']: ")
        with pytest.raises(ConfigError,
                           match="declares no inference.npe:") as caught:
            load_document(doc)
        assert str(caught.value).startswith("runs['amortized']: ")

    def test_a_subsection_with_no_seed_is_refused_naming_that_subsection(self):
        # The four subsections must not share one `where`.  Matching on
        # `inference.npe.bank` rather than on "seed" is what stops this test
        # passing when the create, train or sample seed is the missing one.
        # Whether the refusal comes from parse_npe at load time or from
        # draws._seed_name at draw time is Task 3's choice; either way it
        # names the subsection, and that is the claim.
        doc = npe_document()
        del doc["inference"]["npe"]["bank"]["seed"]
        with pytest.raises(ConfigError, match=r"inference\.npe\.bank"):
            built = load_document(doc)
            _simulate_bank(npe_spec(), built, built.inference.npe)


class TestTheDocumentBuilder:
    def test_extra_runs_ride_beside_the_npe_run_and_are_not_merged_over_it(
            self):
        # ``npe_document`` takes ``*extra_runs`` for Task 9, whose predict
        # reuses this run -- and a parameter no test exercises is a parameter
        # a later edit can drop while every test stays green.  Both halves
        # are load-bearing: that the extra run ARRIVES (a builder ignoring
        # *extra_runs passes a one-run assertion) and that NPE's keys are not
        # merged into it (a builder spreading NPE over every run would give
        # the predict run kind: npe and name: amortized).
        doc = npe_document(None, {"kind": "predict", "reuse": "amortized"})
        assert [one["kind"] for one in doc["runs"]] == ["npe", "predict"]
        assert doc["runs"][0]["name"] == "amortized"
        assert "name" not in doc["runs"][1]

    def test_each_document_gets_its_own_copy_of_every_subsection(self):
        # NPE_SECTION is module-level and every document is built from it, so
        # handing out the dicts THEMSELVES rather than copies makes any test
        # that edits a built document edit the template.  Measured on the
        # `{name: sub}` form: after the missing-seed test below does
        # `del doc[...]["bank"]["seed"]`, NPE_SECTION["bank"] is permanently
        # {'n_simulations': 64} and every later document in the process is
        # built with no bank seed.  It stays green today only because that
        # test is second-to-last in the module, so Task 8 appending after it
        # is what turns this into an order-dependent failure.  The code is
        # already right; this is what keeps it right.
        doc = npe_document()
        del doc["inference"]["npe"]["bank"]["seed"]
        doc["inference"]["npe"]["create"]["width"] = 4096
        assert NPE_SECTION["bank"]["seed"] == {"from": "runtime.seeds.bank"}
        assert NPE_SECTION["create"]["width"] == 16
        fresh = npe_document()["inference"]["npe"]
        assert fresh["bank"]["seed"] == {"from": "runtime.seeds.bank"}
        assert fresh["create"]["width"] == 16


class TestTheTrainedPosterior:
    def test_the_product_carries_what_predict_reads(self):
        # samples is a MAPPING and n_draw an int, because that is what 2C's
        # shipped predict reads off a samples product (diagnostics.py:748 and
        # :763) and Task 9 makes npe one of its sources.  posterior is the
        # trained estimator, so a caller can log_prob against it -- the
        # amortized half of what NPE is for.  best_step == 50 rather than 49
        # is what kills a 0-based re-index: train_posterior's own is 1-based,
        # and the layer converts the TYPE without touching the value.
        product = npe_product()
        assert set(product.samples) == {"g"}
        assert product.samples["g"].shape == (100,)
        assert product.n_draw == 100
        assert type(product.best_step) is int
        assert product.best_step == 50
        assert product.train_loss.shape == (50,)
        assert product.validation_loss.shape == (50,)
        assert product.posterior.n_params == 1

    def test_the_posterior_field_is_the_TRAINED_estimator(self):
        # `posterior` promises the trained estimator, so a caller can log_prob
        # or re-sample against it -- the amortized half of what NPE is for --
        # and NOTHING else in this suite exercises that promise: every other
        # read of the field is `.n_params`, which the UNTRAINED estimator
        # reports identically.  Measured, returning `_estimator(...)` here
        # while `trained` still produced the draws left all thirty-three
        # tests green, and Task 9 consumes `samples` rather than `posterior`,
        # so nothing downstream would have closed it either.
        #
        # Re-drawing is the discriminating assertion: the same datum and the
        # same key against the same weights must reproduce the product's own
        # stack, bit for bit.  An untrained estimator, or any other estimator,
        # cannot.  key(14) is `runtime.seeds.sample` spelled out, which is
        # also why this test asserts what that seed resolves to first.
        product = npe_product()
        built = npe_built()
        assert seed_for("sample", built.context) == 14
        redrawn = product.posterior.sample(
            _observed(npe_spec(), built), jax.random.key(14), 100)
        assert redrawn.shape == (100, 1)
        assert jnp.array_equal(redrawn.reshape(100), product.samples["g"])

    def test_the_draws_follow_the_data_rather_than_the_declaration(self):
        # The one recovered number this exit asserts.  Both documents declare
        # the SAME init (g = 1.0) and the same prior (loc 1.0, scale 0.5);
        # only the value the observed data was simulated at moves.  So a run
        # that sampled the PRIOR, or that handed sample() something other than
        # this run's observed datum, gives one answer twice.  It also kills
        # the reversed sample(key, datum, n) order, which dies inside embed as
        # `TypeError: subtract does not accept dtypes key<fry>, float32` --
        # both orders bind, and only one runs.
        at_truth = npe_product()
        at_one = npe_product(at={"g": 1.0})
        assert float(jnp.mean(at_truth.samples["g"])) == pytest.approx(
            1.5614667, rel=1e-4)
        assert float(jnp.mean(at_one.samples["g"])) == pytest.approx(
            1.0128615, rel=1e-4)
        # and the honest band, which survives a jax or hardware change the two
        # pins above would not: the draws sit on the data, and the two answers
        # are half a unit apart against a draw sd of 0.22.
        assert abs(float(jnp.mean(at_truth.samples["g"])) - 1.5) < 0.25
        assert abs(float(jnp.mean(at_one.samples["g"])) - 1.0) < 0.25

    def test_n_draws_reaches_sample_and_sizes_the_product(self):
        product = npe_product(npe={"sample": {"n_draws": 25}})
        assert product.n_draw == 25
        assert product.samples["g"].shape == (25,)

    def test_the_train_knobs_reach_train_posterior(self):
        # n_steps is observable as the length of BOTH curves, and it is the
        # knob a dropped _passthrough would replace with the package's 3000 --
        # which would also take this test from a fraction of a second to about
        # a minute, so the regression is loud twice over.
        product = npe_product(npe={"train": {"n_steps": 12}})
        assert product.train_loss.shape == (12,)
        assert product.validation_loss.shape == (12,)
        assert 1 <= product.best_step <= 12

    def test_best_step_is_the_BEST_step_and_not_the_last_one(self):
        # THE DOCUMENT ABOVE CANNOT MAKE THIS CLAIM, and neither can the two
        # other places best_step is asserted.  At 50, 80 and 100 steps this
        # estimator's validation loss is still falling, so the best step IS
        # the last one, and `best_step == n_steps` is true of the honest
        # implementation and of `int(history.train.shape[0])` alike -- the two
        # candidates are sampled only where they agree.  Measured: that
        # mutation left all thirty-three tests green, and best_step is the one
        # field telling a caller WHICH estimator came back.
        #
        # At 150 steps the curve turns: best_step is 113 and the last step is
        # 150.  The assertion is against the curve's own argmin rather than
        # against 113, so it survives a jax or hardware change that moves the
        # turning point; `< 150` is what makes it discriminate at all.
        # 1-based, so argmin() + 1 -- which is also the off-by-one this pins.
        product = npe_product(npe={"train": {"n_steps": 150}})
        assert product.validation_loss.shape == (150,)
        assert product.best_step < 150
        assert product.best_step == int(product.validation_loss.argmin()) + 1

    def test_the_optimiser_knobs_reach_train_posterior_too(self):
        # n_steps, batch_size and validation_fraction are each pinned above by
        # a curve's length or emptiness; the four Adam knobs -- learning_rate,
        # beta1, beta2, eps -- were pinned by nothing, which is Task 7's
        # min_scale hole on the twin subsection.  All seven ride ONE
        # `_passthrough` off ONE tuple whose membership
        # test_config_section_npe.py pins by exact equality against
        # train_posterior's signature, so there is no per-knob wiring and one
        # of the four arriving is the tuple being forwarded whole.  Measured,
        # forwarding _TRAIN_OPTIONS minus the four left both npe modules green.
        #
        # eps is the cheap one to make visible: it divides the update by
        # sqrt(v) + eps, so eps = 1e6 freezes the fit in place, and the final
        # training loss comes back POSITIVE against a trained -1.04.  Both
        # legs, for the reason min_scale needed both: the declared value
        # arriving, and the package's own default arriving when nothing is
        # declared.
        frozen = npe_product(npe={"train": {"eps": 1.0e6}})
        assert float(frozen.train_loss[-1]) > 0.0
        assert float(npe_product().train_loss[-1]) < 0.0

    def test_each_of_the_four_seeds_moves_something_of_its_own(self):
        # Four calls, four named seeds, and wiring one subsection's seed into
        # all four is invisible to every shape, dtype and finiteness assertion
        # in this file.  Each leg below changes ONE entry of runtime.seeds and
        # nothing else.
        #
        # What NO leg below proves on its own: a SWAP of two seeds, because
        # both sites still move when either entry changes.  The bank half of
        # that is closed by TestTheBank's first draws pinned to key(11) and
        # the create half by that class's weight comparison.  The train/sample
        # swap is closed too, by ATTRIBUTION rather than by movement --
        # measured, exchanging spec.train and spec.sample at their two call
        # sites fails three tests: test_the_product_carries_what_predict_reads
        # (best_step 48, not 50), test_the_draws_follow_the_data_rather_than_
        # the_declaration (mean 1.5649859 against the pinned 1.5614667), and
        # the train-loss leg at the foot of THIS test.  An earlier draft of
        # this comment said that pair was separated by nothing in the suite;
        # it was a recorded measurement nobody re-measured, and it was wrong.
        base = npe_product()
        for name, value in (("bank", 77), ("create", 88), ("sample", 99)):
            moved = npe_product(seeds={**NPE_SEEDS, name: value})
            assert not jnp.array_equal(moved.samples["g"],
                                       base.samples["g"]), name
        # the train seed shuffles minibatches, so it moves the LOSS CURVE
        # rather than moving the draws by a margin worth a tolerance.
        shuffled = npe_product(seeds={**NPE_SEEDS, "train": 99})
        assert not jnp.array_equal(shuffled.train_loss, base.train_loss)

    def test_validation_fraction_zero_runs_and_reports_an_empty_curve(self):
        # THE TWIN ROUTE.  train_posterior returns validation as (0,) when
        # validation_fraction is 0.0, and history.validation[-1] on it raises
        # `IndexError: index -1 is out of bounds for axis 0 with size 0`.
        # best_step is STILL 50, so it is no signal that validation happened.
        # An executor that reads the final or the best validation loss -- the
        # obvious way to report training -- dies HERE and nowhere else: the
        # default document has a full curve and says nothing.
        product = npe_product(npe={"train": {"validation_fraction": 0.0}})
        assert product.validation_loss.shape == (0,)
        assert product.train_loss.shape == (50,)
        assert product.best_step == 50
        assert product.n_draw == 100


class TestTheUnravel:
    """The flat draws become per-latent stacks, in DECLARATION order.

    ``sample`` returns ``(n_draws, n_params)``; ``predict`` reads a mapping.
    THREE latents, because two cannot separate three orderings -- the reverse
    of a two-name sort is one of the two orders it is meant to be told apart
    from -- and one of the three is ``(16,)``, because a space of scalars
    cannot separate "sized by each latent's own shape" from "one column each".
    """

    def test_the_stacks_are_named_and_shaped_by_the_space(self):
        product = trio_product()
        assert list(product.samples) == ["d", "a", "m"]
        assert product.samples["d"].shape == (40,)
        assert product.samples["a"].shape == (40,)
        assert product.samples["m"].shape == (40, 16)

    def test_the_order_is_the_declared_one_and_not_a_sort(self):
        # d is declared at 0.5, a at 10.0, m at 1.1, so the three are
        # unmistakable by magnitude.  A sorted unravel (a, d, m) hands d the
        # column holding a and reads ~10.0 here; a reverse-sorted one
        # (m, d, a) hands d a gain column and reads ~1.1.
        product = trio_product()
        assert 0.2 < float(jnp.mean(product.samples["d"])) < 0.8
        assert 9.0 < float(jnp.mean(product.samples["a"])) < 11.0
        assert 1.0 < float(jnp.mean(product.samples["m"])) < 1.2

    def test_every_column_of_the_flat_draws_is_accounted_for(self):
        # 1 + 1 + 16 = 18, which is the estimator's own n_params.  An unravel
        # that assumed scalars consumes 3, and _unravel's own width check
        # refuses it before it reaches here (measured: that mutation dies as
        # "the draws are 18 wide and inference.parameters accounts for 3");
        # one that permuted the WIDTHS lands here instead.
        product = trio_product()
        consumed = sum(math.prod(stack.shape[1:])
                       for stack in product.samples.values())
        assert consumed == product.posterior.n_params == 18

    def test_the_unravel_is_told_which_run_it_is_serving(self, monkeypatch):
        """The width refusal names the run, and only a spy can say so.

        ``_unravel`` refuses a flat whose width the space does not account
        for, and its ``where=`` is what makes that refusal open
        ``runs['<name>']:`` the way every other refusal in this layer does --
        without one it is the only refusal in either new module that names no
        run, which reads as a package error rather than as this layer's.

        **No document can reach that refusal**: the estimator is sized from
        the same space the draws are unravelled through, so the widths always
        agree, and every test above that unravels anything unravels
        successfully.  Measured -- dropping ``where=where`` from the call left
        all thirty-two tests in this module green.  So the spy is the only
        test there is, and the alternative was shipping a decision that a
        revert leaves passing.

        It rides the ONE-latent document rather than the cached trio: a spy
        must be installed before the run it watches, and a cache that another
        test may already have filled would make this pass on an empty
        capture in one order and fail in another.
        """
        from rheplicant.config.sections import npe as npe_module

        seen = {}
        real = npe_module._unravel

        def spy(space, flat, **kwargs):
            seen["where"] = kwargs.get("where")
            seen["shape"] = tuple(flat.shape)
            return real(space, flat, **kwargs)

        monkeypatch.setattr(npe_module, "_unravel", spy)
        product = npe_product()
        assert seen["where"] == "runs['amortized']"
        # the RAW (n_draws, n_params) array the package handed back, not
        # something this layer reshaped on the way in
        assert seen["shape"] == (100, 1)
        assert product.samples["g"].shape == (100,)


class TestThePriorGate:
    """``joint_prior`` coverage is a prior to ``nuts`` and not to ``npe``.

    ONE document, two runs.  The npe run declares ``expect: refuse`` and the
    nuts run does not, so a single ``run_document`` proves both halves of the
    claim on the same space -- not on two documents that merely look alike.
    """

    def test_one_document_is_refused_by_npe_and_run_by_nuts(self):
        results = joint_results()
        refused = results["amortized"]
        assert refused.product is None
        message = str(refused.error)
        # SIMULATES appears in no other refusal in this layer (grepped over
        # nuts.py and exit_support.py: zero hits), so this cannot be satisfied
        # by the neighbouring nuts-leg message, by _npe_spec's "declares no
        # inference.npe:", or by _decided_model's.
        assert "kind: npe SIMULATES a bank" in message
        assert message.startswith("runs['amortized']: ")
        # both ways out, by name
        assert "inference.parameters.<name>.prior:" in message
        assert "run kind: nuts" in message
        # and WHY the sibling differs, which the package's own
        # ParameterSpaceError never says -- it names the latents, names no run
        # and mentions no other exit.
        assert "which is why kind: nuts accepts this space" in message

        # the sibling, on the SAME document, actually running
        chain = results["chain"]
        assert chain.error is None
        assert set(chain.product.samples) == {"d", "a"}
        # get_samples() also returns the deterministic "prediction" site,
        # whose per-sample shape is the whole TOD.  The SET assertion above is
        # the discriminating half; this line is the weaker, redundant one.
        # Measured 2026-08-27 (D26): renaming the site to `__mu__` leaves this
        # line trivially true while the set assertion still fires, because it
        # would see the new name. This comment used to claim the opposite and
        # it sent a reader the wrong way once.
        assert "prediction" not in chain.product.samples
        assert chain.product.n_draw == 200
        assert float(jnp.mean(chain.product.samples["d"])) == pytest.approx(
            1.2, abs=0.01)
        assert float(jnp.mean(chain.product.samples["a"])) == pytest.approx(
            12.0, abs=0.05)

    def test_the_refusal_is_not_the_missing_section_one(self):
        # joint_prior_document writes inference.npe: onto the document on
        # purpose.  Without it the run would be refused by _npe_spec one
        # branch earlier and this class would be testing something else
        # entirely.  The document assertion is what keeps that deliberate --
        # and it asserts of the BUILDER, which is the one place that section
        # comes from: JOINT_PRIOR_PAIR carries no npe: key, because one there
        # would be dead (the builder overrides it) while holding the last
        # un-copied reference to the shared NPE_SECTION.
        assert "npe" in joint_prior_document()["inference"]
        message = str(joint_results()["amortized"].error)
        assert "declares no inference.npe:" not in message

    def test_the_advice_the_gate_gives_depends_on_the_document(self):
        # The gate's two clauses are BOTH conditional on joint-prior coverage,
        # and on a document with no joint_prior at all the coverage advice
        # would be FALSE: kind: nuts refuses that document too, for the same
        # missing prior.
        #
        # Making either clause UNCONDITIONAL is caught by three tests -- this
        # one, TestTheBank's prior-free test, and
        # test_a_prior_free_space_gets_no_joint_prior_advice_from_npe -- so
        # that is not why this test exists.  What only a test holding BOTH
        # messages at once can catch is the clause APPENDED to the covered
        # branch as well: one document then gets "or run kind: nuts" and "kind:
        # nuts refuses this document too" in the same breath, advice that
        # contradicts itself, and every single-document test still passes
        # because the string it looks for is present.  Measured: that mutation
        # fails exactly one test in the suite, this one.
        #
        # It is also the route a user takes, which is where the run's own name
        # reaches the message.
        #
        # BOTH documents now declare `expect: refuse` -- the joint one always
        # did.  Plan 3A's Task 8 hoists this property to P-1 as check A23, so
        # a prior-free npe run makes the whole document unloadable unless the
        # run says the refusal is what it is for, and `run_document` would
        # otherwise raise A23's sentence rather than capture this gate's.  The
        # test is unchanged in what it reads: two messages from this gate, in
        # one test, so a clause appended to the covered branch as well is
        # still the mutation only this test catches.
        joint = str(joint_results()["amortized"].error)
        bare = str(run_document(npe_document({"expect": "refuse"},
                                             parameters=PRIOR_FREE))
                   ["amortized"].error)
        assert "SIMULATES a bank" in bare
        assert "which is why kind: nuts accepts this space" in joint
        assert "or run kind: nuts" in joint
        assert "kind: nuts refuses this document too" not in joint
        assert "which is why kind: nuts accepts this space" not in bare
        assert "or run kind: nuts" not in bare
        assert "kind: nuts refuses this document too" in bare
        assert bare.startswith("runs['amortized']: ")


class TestTheRunTakesNoKeysOfItsOwn:
    def test_a_run_level_seed_is_refused_naming_the_four_that_are_needed(self):
        # A user copying a kind: nuts run writes seed: on it, and check A29
        # does say npe needs a seed -- so the bare sweep message ("does not
        # take ['seed']; it takes []") would be true and useless.  The match
        # is on FOUR, which no other refusal in this layer contains.
        with pytest.raises(ConfigError, match="FOUR seeds"):
            run_document(npe_document({"seed": {"from": "runtime.seeds.x"}}))

    def test_any_other_run_level_key_is_swept(self):
        # n_draws is the plausible mistake: it IS a key this exit honours,
        # just in inference.npe.sample rather than on the run.  Matching the
        # sweep's own wording as well as the key is what stops this passing
        # on the seed refusal above.
        with pytest.raises(ConfigError, match="does not take") as caught:
            run_document(npe_document({"n_draws": 10}))
        assert "n_draws" in str(caught.value)
        assert "FOUR seeds" not in str(caught.value)


class TestTheKindIsRunnable:
    def test_npe_is_declared_and_the_deferral_tuple_has_gone(self):
        from rheplicant.config.sections import runs as runs_module

        assert "npe" in _KINDS
        assert not hasattr(runs_module, "_KINDS_2D"), (
            "runs._KINDS_2D still exists; the last task to move a kind out of "
            "it deletes the tuple and its refusal branch."
        )


class TestTheRunLevelParse:
    """Plan 4A Task 9: npe's parse is the empty sweep plus the section's
    presence -- no simulation, no training, no draw."""

    def test_parse_touches_no_science_and_projects_empty_views(
            self, monkeypatch):
        import rheplicant.inference as inference
        from _rheplicant_bootstrap.variants import LayerRef
        from rheplicant.config.sections.exit_support import parse_run

        def explode(*args, **kwargs):
            raise AssertionError("science ran during parse")

        monkeypatch.setattr(inference, "simulate_pairs", explode)
        monkeypatch.setattr(inference, "NeuralPosterior", explode)
        monkeypatch.setattr(inference, "train_posterior", explode)
        parsed = parse_run(npe_spec(), npe_built(), index=0,
                           layer=LayerRef(kind="base", name=None, prefix="",
                                          document={}, declared_runs=None))
        assert dict(parsed.parsed.execution) == {}
        assert dict(parsed.parsed.resolved) == {}
