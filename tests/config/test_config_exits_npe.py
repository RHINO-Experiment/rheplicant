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
"""

import jax.numpy as jnp
import pytest

from rheplicant.config.document import load_document
from rheplicant.config.draws import seed_for
from rheplicant.config.errors import ConfigError
from rheplicant.config.sections.npe import _estimator, _npe_spec, _simulate_bank
from rheplicant.inference.npe import MIN_SCALE
from tests.config.exit_helpers import FROZEN
from tests.config.posterior_helpers import (
    NPE_SECTION,
    NPE_SEEDS,
    npe_built,
    npe_document,
    npe_spec,
)


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
        # _decided_model's message true for every caller.  The rest of it is
        # conjugate prose -- measured here, it offers kind: conjugate.wiener
        # to a document that asked for a neural posterior -- and that drift is
        # recorded to the plan's §6 rather than fixed here, because
        # exit_support.py is Task 2's alone.
        built = npe_built(noise=FROZEN)
        with pytest.raises(ConfigError, match="radiometer_frozen") as caught:
            _simulate_bank(npe_spec(), built, built.inference.npe)
        assert str(caught.value).startswith("runs['amortized']: ")

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
        built = npe_built(parameters={"g": {"init": 1.0, "linear": True,
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

    def test_an_undeclared_knob_gets_the_packages_own_default(self):
        # min_scale is the key NPE_SECTION deliberately does not write, so
        # this is the standing rule -- config keys never restate package
        # defaults -- observed arriving rather than asserted in prose.  The
        # second assertion is what keeps it honest: if a later edit adds
        # min_scale to NPE_SECTION, this test would otherwise keep passing
        # while proving nothing.
        built = npe_built(npe={"bank": {"n_simulations": 8}})
        _, thetas, data = self._pairs(built)
        estimator = _estimator(npe_spec(), built, built.inference.npe,
                               thetas, data)
        assert estimator.min_scale == MIN_SCALE
        assert "min_scale" not in built.inference.npe.create

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
        doc = npe_document()
        del doc["inference"]["npe"]
        built = load_document(doc)
        assert built.inference.npe is None
        with pytest.raises(ConfigError,
                           match="declares no inference.npe:") as caught:
            _npe_spec(npe_spec(), built)
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
