"""The conjugate family: its shared machine, and the exits that ride it.

The first half drives the helpers -- ``_conjugate_block``, ``_prior_kwargs``,
``_decided_sigma``, ``_decided_model`` -- directly against a built document,
which is exactly the surface the four executors call.  The second half runs
``conjugate.wiener`` through ``run_document``, from the YAML mapping down to
``wiener_solve`` and back.
"""

import equinox as eqx
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.document import load_document
from rheplicant.config.sections import conjugate as conjugate_module
from rheplicant.config.sections.conjugate import _conjugate_block, _prior_kwargs
from rheplicant.config.sections.exit_support import (
    EXECUTORS,
    _decided_model,
    _decided_sigma,
)
from rheplicant.config.sections.runs import parse_runs, run_document
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import (
    HomoscedasticNoise,
    RadiometerNoise,
    condition_estimate,
    wiener_solve,
)
from tests.config.exit_helpers import (
    CHANNEL_WIDTH_HZ,
    FROZEN,
    INTEGRATION_TIME_S,
    NO_OBSERVED,
    NONLINEAR_LATENT,
    PRIOR_FREE,
    PRIOR_FREE_TWO,
    RADIOMETER,
    SIGMA_K,
    TRUTH_A,
    TRUTH_D,
    TRUTH_G,
    TWO_LATENTS,
    conjugate_built,
    conjugate_document,
    spec,
    two_latent_document,
    wiener_document,
)

# Two builds, shared: load_document is the expensive part and neither test
# mutates what it hands back.
ONE = conjugate_built()
TWO = conjugate_built(inference=TWO_LATENTS)
# `where` is the exits.py spelling (the quoted NAME, not the index) and is
# always the third POSITIONAL argument to the two helpers -- see plan section
# 3.1.  spec() names its runs after their kind, so this is what they carry.
WHERE = "runs['conjugate.wiener']"
CONDITION_WHERE = "runs['condition']"


class TestTheLeafIsWired:
    def test_exits_imports_the_conjugate_leaf_at_its_foot(self):
        """Importing exits must pull the leaf modules in.

        Tasks 3-5 register their executors inside this module; if the foot
        import goes missing they register nowhere and Task 1's completeness
        test is the only thing left to notice.
        """
        from rheplicant.config.sections import exits

        assert exits.conjugate is conjugate_module


class TestTheSharedDocument:
    def test_the_one_latent_document_is_solvable_at_the_truth(self):
        """Also pins the three-tuple: one call hands back all three inputs.

        ``_conjugate_block`` returns ``(block, sigma, observed)`` so that no
        executor has to remember to run the A27 gate and the observed lookup
        beside it -- an executor that called only the block builder would
        solve at whatever sigma it invented.
        """
        run = spec(names=["g"])
        block, sigma, observed = _conjugate_block(run, ONE, WHERE)
        assert observed is ONE.inference.observed.entries["primary"]
        assert float(sigma[0, 0]) == pytest.approx(SIGMA_K)
        solved, residual = wiener_solve(block, observed, noise_std=sigma)
        assert float(solved["g"]) == pytest.approx(TRUTH_G, abs=1e-4)
        assert float(residual) < 1e-5

    def test_the_two_latent_document_separates_its_latents(self):
        """Two latents in one block, recovered apart: the A51 fixture is real.

        A block whose members were degenerate would still solve, and would
        still return the right shape -- it would just return the wrong split.
        """
        run = spec(names=["d", "a"])
        block, sigma, observed = _conjugate_block(run, TWO, WHERE)
        solved, residual = wiener_solve(block, observed, noise_std=sigma)
        assert float(solved["d"]) == pytest.approx(TRUTH_D, abs=1e-3)
        assert float(solved["a"]) == pytest.approx(TRUTH_A, abs=1e-2)
        assert float(residual) < 1e-5


class TestTheConjugateBlock:
    def test_the_block_groups_even_a_single_latent(self):
        """names= even for one, so the solution comes back keyed by latent.

        The name= spelling would also build and also solve; its answer is a
        bare array, which six downstream consumers raise on.
        """
        block, _, _ = _conjugate_block(spec(names=["g"]), ONE, WHERE)
        assert block.grouped is True
        assert block.names == ("g",)

    def test_a_bare_string_names_a_group_of_one(self):
        """names: g is one latent, not four one-character ones."""
        block, _, _ = _conjugate_block(spec(names="g"), ONE, WHERE)
        assert block.names == ("g",)

    def test_names_is_required_and_says_why(self):
        with pytest.raises(ConfigError, match="names:") as caught:
            _conjugate_block(spec(), ONE, WHERE)
        assert str(caught.value).startswith(f"{WHERE}: names:")

    def test_names_is_a_name_or_a_list_of_them(self):
        for bad in (7, [], ["g", 3], {"g": 1}):
            with pytest.raises(ConfigError, match="names:") as caught:
                _conjugate_block(spec(names=bad), ONE, WHERE)
            assert repr(bad) in str(caught.value)

    def test_check_reaches_linear_operator_and_nothing_else(self):
        """check: is linear_operator's alone, and it is observable.

        w writes into global_signal.width, which the prediction is NOT affine
        in, so the declared linear=True is a false claim: check=True catches
        it and check=False builds the block anyway.
        """
        nonlinear = conjugate_built(inference=NONLINEAR_LATENT)
        with pytest.raises(ParameterSpaceError, match="not affine"):
            _conjugate_block(spec(names=["w"]), nonlinear, WHERE)
        block, _, _ = _conjugate_block(spec(names=["w"], check=False),
                                       nonlinear, WHERE)
        assert block.names == ("w",)

    def test_a_non_bool_check_is_refused(self):
        with pytest.raises(ConfigError, match="check:") as caught:
            _conjugate_block(spec(names=["g"], check="no"), ONE, WHERE)
        assert "'no'" in str(caught.value)

    def test_needs_observed_decides_whether_data_is_required(self):
        """condition estimates kappa from the operator alone; the solves do not.

        The same document must refuse under one flag and build under the
        other -- an implementation that ignored the flag passes neither half.
        """
        bare = load_document(conjugate_document(inference=NO_OBSERVED))
        with pytest.raises(ConfigError, match="inference.observed"):
            _conjugate_block(spec(names=["g"]), bare, WHERE)
        block, sigma, observed = _conjugate_block(spec(names=["g"]), bare,
                                                  WHERE, needs_observed=False)
        assert block.names == ("g",)
        assert observed is None          # the third element, suppressed
        assert sigma.shape == (16, 8)    # the sigma is still decided

    def test_the_two_kinds_that_find_their_own_sigma_get_none(self):
        """The False branch of _DECIDES_SIGMA_HERE, which nothing else here takes.

        Every other test in this module runs spec()'s default kind, so an
        implementation that dropped the gate and always called
        _decided_sigma would pass all of them.  gcr under `noise_from: gls`
        and gls itself take their sigma from the GLS solve, and resolving a
        DECLARED one eagerly would fire A27 on exactly the document those two
        exist to serve.
        """
        for kind in ("conjugate.gcr", "conjugate.gls"):
            _, sigma, _ = _conjugate_block(spec(kind=kind, names=["g"]),
                                           ONE, WHERE)
            assert sigma is None

    def test_condition_is_the_other_kind_that_decides_its_sigma_here(self):
        """The SECOND member of _DECIDES_SIGMA_HERE, run under its own kind.

        Every other test in this module -- including the needs_observed one,
        which exercises condition's MODE -- runs spec()'s default
        conjugate.wiener, so dropping "condition" from the set left the whole
        suite green.  What it costs downstream: condition_estimate's
        noise_std= is keyword-only with no default, so Task 6 would get a
        TypeError from inside the package instead of a sigma.
        """
        bare = conjugate_built(inference=NO_OBSERVED)
        block, sigma, observed = _conjugate_block(
            spec(kind="condition", names=["g"]), bare, CONDITION_WHERE,
            needs_observed=False)
        assert observed is None
        assert sigma.shape == (16, 8)
        assert float(sigma[0, 0]) == pytest.approx(SIGMA_K)
        assert float(condition_estimate(block, noise_std=sigma)) == (
            pytest.approx(1.0, abs=1e-3))

    def test_check_A27_fires_under_kind_condition_too(self):
        """A27 must reach all four kinds it names, not just the default one.

        Without condition in _DECIDES_SIGMA_HERE this document reaches
        condition_estimate with noise_std missing, and the user hears a
        package-level TypeError instead of the refusal that names the two
        routes -- the exact substitution A27 exists to prevent.
        """
        built = conjugate_built(inference={**NO_OBSERVED, "noise": RADIOMETER})
        with pytest.raises(ConfigError, match="check A27") as caught:
            _conjugate_block(spec(kind="condition", names=["g"]), built,
                             CONDITION_WHERE, needs_observed=False)
        assert "kind: condition takes a DECIDED sigma" in str(caught.value)
        # CONDITION_WHERE has to EARN its name.  _decided_sigma builds its own
        # prefix from run.name rather than taking `where`, so without this the
        # constant is documentary: it could drift to the runs.py index form,
        # or to any string at all, with the suite green.  WHERE is pinned the
        # same way by test_names_is_required_and_says_why.
        assert str(caught.value).startswith(f"{CONDITION_WHERE}: ")

    def test_the_missing_observation_is_heard_before_the_undecidable_sigma(self):
        """A document broken BOTH ways hears about its data first.

        The docstring promises this order; moving the _observed call after
        _decided_sigma changes which refusal a doubly-broken document gets
        and no other test can tell.  Data it never declared is the more
        actionable of the two.
        """
        doubly = conjugate_built(inference={**NO_OBSERVED,
                                            "noise": RADIOMETER})
        with pytest.raises(ConfigError,
                           match="compares against inference.observed") as got:
            _conjugate_block(spec(names=["g"]), doubly, WHERE)
        assert "check A27" not in str(got.value)


class TestThePriorKwargs:
    """Four positional arguments, in this order: run, built, block, where.

    ``_prior_kwargs`` derives the ParameterSpace itself, from ``built`` -- a
    draft of this plan had one caller passing the ``where`` STRING where a
    space belonged, which does not raise and quietly breaks the refusal
    branch that reads ``space.latent(name).prior``.  Every call site below
    passes ``built``, and never a space.
    """

    def test_an_undeclared_prior_is_not_passed_at_all(self):
        """Absent means absent: the latent's own prior: drives the solve."""
        run = spec(names=["g"])
        block, _, _ = _conjugate_block(run, ONE, WHERE)
        assert _prior_kwargs(run, ONE, block, WHERE) == {}

    def test_a_scalar_broadcasts_into_the_one_name_the_block_holds(self):
        run = spec(names=["g"], prior_std=0.25)
        block, _, _ = _conjugate_block(run, ONE, WHERE)
        assert _prior_kwargs(run, ONE, block, WHERE) == {
            "prior_std": {"g": 0.25}}

    def test_a_scalar_over_several_latents_is_check_A51(self):
        run = spec(names=["d", "a"], prior_std=0.25)
        block, _, _ = _conjugate_block(run, TWO, WHERE)
        with pytest.raises(ConfigError, match="wrongly-regularised") as caught:
            _prior_kwargs(run, TWO, block, WHERE)
        assert "check A51" in str(caught.value)

    def test_the_mapping_form_keeps_each_width_on_its_own_latent(self):
        """Distinct numbers per key and per keyword: nothing may be swapped."""
        run = spec(names=["d", "a"], prior_std={"d": 0.25, "a": 7.0},
                   prior_mean={"d": -1.0, "a": 3.0})
        block, _, _ = _conjugate_block(run, TWO, WHERE)
        assert _prior_kwargs(run, TWO, block, WHERE) == {
            "prior_std": {"d": 0.25, "a": 7.0},
            "prior_mean": {"d": -1.0, "a": 3.0}}

    def test_a_partial_mapping_names_the_members_that_declare_a_prior(self):
        run = spec(names=["d", "a"], prior_std={"d": 0.25})
        block, _, _ = _conjugate_block(run, TWO, WHERE)
        with pytest.raises(ConfigError, match=r"\['d', 'a'\] declare one"):
            _prior_kwargs(run, TWO, block, WHERE)

    def test_a_key_the_block_does_not_group_is_refused(self):
        """The refusal must NAME the keys given, not merely say S is block-diagonal.

        Every refusal in ``_one_prior`` carries "block-diagonal", so matching
        that alone would pass against an implementation that dropped this
        branch entirely and fell through to the scalar one.
        """
        run = spec(names=["d", "a"], prior_std={"d": 0.25, "a": 7.0, "g": 1.0})
        block, _, _ = _conjugate_block(run, TWO, WHERE)
        with pytest.raises(ConfigError,
                           match=r"names \['a', 'd', 'g'\], and this block "
                                 r"groups \['d', 'a'\]"):
            _prior_kwargs(run, TWO, block, WHERE)

    def test_a_negative_width_is_refused_and_a_negative_centre_is_not(self):
        """A width is non-negative; a centre is a location and may be any sign."""
        block, _, _ = _conjugate_block(spec(names=["g"]), ONE, WHERE)
        with pytest.raises(ConfigError, match="prior_std") as caught:
            _prior_kwargs(spec(names=["g"], prior_std=-1.0), ONE, block, WHERE)
        assert ">= 0" in str(caught.value)   # the FLOOR, not the A51 refusal
        assert _prior_kwargs(spec(names=["g"], prior_mean=-1.0), ONE, block,
                             WHERE) == {"prior_mean": {"g": -1.0}}

    def test_a_non_numeric_width_is_refused_naming_the_member(self):
        block, _, _ = _conjugate_block(spec(names=["g"]), ONE, WHERE)
        run = spec(names=["g"], prior_std={"g": "wide"})
        with pytest.raises(ConfigError, match=r"prior_std\.g"):
            _prior_kwargs(run, ONE, block, WHERE)


class TestThePriorKwargsReachTheSolve:
    """Compiled, then SOLVED WITH -- not merely compared against a literal.

    Every test above asserts the returned dict against a literal, and three
    differently-shaped compilations all satisfy that comparison: a non-dict
    Mapping, a numpy scalar and a JAX scalar each `== {"g": 0.25}`.  Measured,
    only the Mapping detonates at the solve -- `_per_member` (linear.py:963)
    tests `isinstance(value, dict)`, which a mappingproxy fails; the two
    scalar shapes solve to `g=1.4999948`, identical to the shipped float to
    the last digit.  So this class buys exactly one failure mode that the
    literal comparison cannot see, and that one is real.  These tests pass the
    compiled output to the real ``wiener_solve``.

    They need the PRIOR-FREE documents.  Every latent in ONE_LATENT and
    TWO_LATENTS declares a prior:, and ``_reconcile`` (linear.py:872) refuses
    a supplied prior_std= that disagrees with one -- a package refusal this
    layer deliberately does not duplicate, since it names both sides already.
    """

    def test_a_broadcast_scalar_is_a_width_the_solver_accepts(self):
        built = conjugate_built(parameters=PRIOR_FREE)
        run = spec(names=["g"], prior_std=0.25)
        block, sigma, observed = _conjugate_block(run, built, WHERE)
        kwargs = _prior_kwargs(run, built, block, WHERE)
        assert kwargs == {"prior_std": {"g": 0.25}}
        solved, residual = wiener_solve(block, observed, noise_std=sigma,
                                        **kwargs)
        assert float(solved["g"]) == pytest.approx(TRUTH_G, abs=1e-4)
        assert float(residual) < 1e-5

    def test_the_mapping_form_reaches_the_solve_on_the_right_latents(self):
        """Distinct widths and centres per latent, all four surviving the trip.

        Wide priors and centres away from the truth, so the data has to do
        the work: a swapped prior_mean would pull the answer off it.
        """
        built = conjugate_built(inference=TWO_LATENTS,
                                parameters=PRIOR_FREE_TWO)
        run = spec(names=["d", "a"], prior_std={"d": 5.0, "a": 50.0},
                   prior_mean={"d": 0.0, "a": 10.0})
        block, sigma, observed = _conjugate_block(run, built, WHERE)
        kwargs = _prior_kwargs(run, built, block, WHERE)
        assert kwargs == {"prior_std": {"d": 5.0, "a": 50.0},
                          "prior_mean": {"d": 0.0, "a": 10.0}}
        solved, residual = wiener_solve(block, observed, noise_std=sigma,
                                        **kwargs)
        assert float(solved["d"]) == pytest.approx(TRUTH_D, abs=1e-3)
        assert float(solved["a"]) == pytest.approx(TRUTH_A, abs=1e-2)
        assert float(residual) < 1e-5

    def test_a_prior_free_document_with_no_width_is_the_package_refusal(self):
        """Why the key is not optional here: the solve itself demands one.

        Pins that PRIOR_FREE really is prior-free, so the two tests above
        prove the keyword was ACCEPTED rather than quietly ignored.
        """
        built = conjugate_built(parameters=PRIOR_FREE)
        run = spec(names=["g"])
        block, sigma, observed = _conjugate_block(run, built, WHERE)
        assert _prior_kwargs(run, built, block, WHERE) == {}
        with pytest.raises(ParameterSpaceError, match="needs a prior_std"):
            wiener_solve(block, observed, noise_std=sigma)

    def test_the_two_latent_document_is_prior_free_as_well(self):
        """The same guard for PRIOR_FREE_TWO, which the grouped block uses.

        Without it, an AGREEING prior: creeping into PRIOR_FREE_TWO would let
        the mapping-form test pass with the compiled kwargs removed entirely:
        _reconcile accepts a keyword that agrees with a declaration, so the
        declaration would silently supply what the keyword was meant to prove.
        That is the tautology the one-latent test above rules out, and Tasks
        3-6 reach for the GROUPED constant far more often than the single.
        """
        built = conjugate_built(inference=TWO_LATENTS,
                                parameters=PRIOR_FREE_TWO)
        run = spec(names=["d", "a"])
        block, sigma, observed = _conjugate_block(run, built, WHERE)
        assert _prior_kwargs(run, built, block, WHERE) == {}
        with pytest.raises(ParameterSpaceError, match="needs a prior_std"):
            wiener_solve(block, observed, noise_std=sigma)


class TestTheDecidedSigma:
    def test_a_homoscedastic_document_decides_a_full_shaped_array(self):
        """decided_noise hands back the MODEL here; the seam takes an array."""
        sigma = _decided_sigma(spec(), ONE)
        assert sigma.shape == (16, 8)
        assert float(sigma[0, 0]) == pytest.approx(SIGMA_K)

    def test_a_radiometer_document_is_check_A27(self):
        """Both routes by name, and in the layer's own voice.

        The package refuses this too (linear.py:1031) but with a
        ParameterSpaceError, and only once a block exists; ConfigError here
        is the assertion.
        """
        built = conjugate_built(noise=RADIOMETER)
        with pytest.raises(ConfigError, match="conjugate.gls") as caught:
            _decided_sigma(spec(), built)
        assert "radiometer_frozen" in str(caught.value)
        assert "check A27" in str(caught.value)

    def test_a_frozen_sigma_passes_through_undisturbed(self):
        built = conjugate_built(noise=FROZEN)
        sigma = _decided_sigma(spec(), built)
        observed = built.inference.observed.entries["primary"]
        assert float(sigma[0, 0]) == pytest.approx(
            abs(float(observed[0, 0]))
            / (CHANNEL_WIDTH_HZ * INTEGRATION_TIME_S) ** 0.5, rel=1e-5)

    def test_noise_kind_none_keeps_the_shared_refusal(self):
        built = conjugate_built(inference={"parameters":
                                           NO_OBSERVED["parameters"]})
        with pytest.raises(ConfigError, match="forward and optimize"):
            _decided_sigma(spec(), built)


class TestTheDecidedModel:
    def test_a_homoscedastic_document_hands_back_the_model(self):
        assert isinstance(_decided_model(spec(), ONE), HomoscedasticNoise)

    def test_what_A27_refuses_is_exactly_what_gls_takes(self):
        """The mirror: the radiometer the array route refuses is gls's input."""
        built = conjugate_built(noise=RADIOMETER)
        assert isinstance(_decided_model(spec(), built), RadiometerNoise)

    def test_a_decided_array_is_refused_naming_conjugate_wiener(self):
        """Run it as gls, so the named alternative is not the interpolated kind.

        Under spec()'s default kind the message would carry
        "kind: conjugate.wiener" from the interpolation alone, and this match
        would pin nothing: gls is the kind that actually calls this.
        """
        built = conjugate_built(noise=FROZEN)
        with pytest.raises(ConfigError, match="conjugate.wiener") as caught:
            _decided_model(spec(kind="conjugate.gls"), built)
        assert "check A28" in str(caught.value)
        assert "radiometer_frozen" in str(caught.value)

    def test_noise_kind_none_keeps_the_shared_refusal(self):
        """The mirror of the sigma route's own kind: none test.

        Both accessors must go through _noise, not decided_noise: a document
        declaring NO noise gets the shared "legal only for forward and
        optimize" refusal.  Reading decided_noise directly would hand None
        to the isinstance and fall through to A28, which then says kind: none
        "decides its sigma into an array" -- of a document that declares no
        sigma at all.
        """
        built = conjugate_built(inference={"parameters":
                                           NO_OBSERVED["parameters"]})
        with pytest.raises(ConfigError, match="forward and optimize") as got:
            _decided_model(spec(kind="conjugate.gls"), built)
        assert "check A28" not in str(got.value)


# --- kind: conjugate.wiener, end to end ------------------------------------
#
# names: is REQUIRED -- _conjugate_block's _selected raises without it, and
# test_names_is_required_and_says_why above pins that refusal -- so every run
# dict below carries one.  ["g"] is wiener_document()'s own single latent; the
# two_latent_document() call sites override it with ["dep", "c"] or a
# deliberate sub-block.
WIENER = {"kind": "conjugate.wiener", "width": "none", "names": ["g"]}

# GAIN_LATENT -- the document's default latent, in exit_helpers -- is what
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


def _product(document):
    """The one run a Task-3 document declares, executed, its product."""
    return run_document(document)["conjugate.wiener"].product


class TestTheKindIsRunnable:
    def test_conjugate_wiener_parses_rather_than_deferring(self):
        (run,) = parse_runs([{"kind": "conjugate.wiener", "width": "none"}])
        assert run.kind == "conjugate.wiener"
        assert run.options == {"width": "none"}

    def test_the_kind_has_an_executor(self):
        assert "conjugate.wiener" in EXECUTORS

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
        product = _product(wiener_document(WIENER))
        assert float(product["mean"]["g"]) == pytest.approx(TRUTH_G, abs=1e-4)

    def test_the_product_carries_the_relative_residual_as_a_number(self):
        # wiener_solve returns a 2-TUPLE, (x_hat, relative_residual), and the
        # residual is a scalar jax.Array.  ``type(...) is float`` is the
        # assertion: isinstance would also accept a numpy float64, and these
        # products end up in reports and refusal messages.
        product = _product(wiener_document(WIENER))
        assert type(product["residual"]) is float
        assert product["residual"] < 1e-5

    def test_a_grouped_block_of_one_still_comes_back_a_mapping(self):
        # The schema always compiles to names=, even for a block of one,
        # because six downstream consumers raise on the bare-array form.
        # name='g' would put a bare array under "mean" and fail here.  (It
        # does NOT prove block.as_dict was called: over a grouped block the
        # solve already returns the mapping, and as_dict is the idempotent
        # wrap that makes the product's shape independent of that.)
        product = _product(wiener_document(WIENER))
        assert sorted(product["mean"]) == ["g"]

    def test_a_two_latent_block_recovers_both(self):
        # Measured: dep = 0.9999954 against a truth of 1.0, c = 0.0199999
        # against 0.02.  Two latents whose scales differ by 50x, so a prior
        # or a solution filed under the wrong member shows up here.
        product = _product(two_latent_document(
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
            product = _product(two_latent_document(
                {**WIENER, "names": names, "require_convergence": None}))
            assert list(product["mean"]) == names

    def test_names_selects_a_sub_block(self):
        # Holding c FIXED is a different question from fitting the pair, and
        # measurably so: dep comes back -2.9279 here against 0.9999954 when
        # the block holds both.  Asserting only the key set would pass
        # against an executor that solved the whole space and then sliced.
        product = _product(two_latent_document(
            {**WIENER, "names": ["dep"], "require_convergence": None}))
        assert sorted(product["mean"]) == ["dep"]
        assert float(product["mean"]["dep"]) == pytest.approx(-2.9279,
                                                              rel=1e-3)

    def test_require_convergence_null_reaches_the_solve(self):
        # The two-latent block is ill-conditioned enough that the package's
        # own guard (require_convergence=1e-3, its default) fires -- as an
        # eqx.error_if from inside jit, so an EquinoxRuntimeError and NOT a
        # ParameterSpaceError, and its text names both exits whichever was
        # called.  Declaring require_convergence: null turns the guard off,
        # so this pair pins that the declared key travelled: an executor that
        # dropped it raises on the second call, one that hard-coded null
        # never raises on the first.
        with pytest.raises(eqx.EquinoxRuntimeError,
                           match="wiener_solve/gcr_sample"):
            run_document(two_latent_document({**WIENER,
                                              "names": ["dep", "c"]}))
        product = _product(two_latent_document(
            {**WIENER, "names": ["dep", "c"], "require_convergence": None}))
        assert float(product["mean"]["dep"]) == pytest.approx(1.0, abs=1e-3)

    def test_maxiter_reaches_the_solve(self):
        # One CG iteration on the two-latent block leaves dep at -2.7e-05
        # where the converged answer is 0.9999954, so a maxiter: that never
        # arrived is a four-order-of-magnitude difference, not a rounding one.
        product = _product(two_latent_document(
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
        product = _product(two_latent_document(
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
        product = _product(wiener_document({**WIENER, "maxiter": None}))
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
        product = _product(wiener_document(WIENER))
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
        product = _product(wiener_document(
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
        product = _product(two_latent_document(
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
        product = _product(wiener_document(
            {**WIENER, "names": ["c"], "width": "fisher", "check": False},
            parameters={"c": CENTRE_LATENT}, at={"c": 76.0}))
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
        product = _product(wiener_document(WIENER, noise=FROZEN))
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
        product = _product(wiener_document(
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

    def test_a_latent_that_is_not_linear_is_refused_by_default(self):
        # names: ["c"] rather than WIENER's ["g"]: this document's only
        # latent is c, and names: has no default to fall back on.
        with pytest.raises(ParameterSpaceError, match="JOINTLY"):
            run_document(wiener_document(
                {**WIENER, "names": ["c"]},
                parameters={"c": CENTRE_LATENT}, at={"c": 76.0}))

    def test_check_false_reaches_linear_operator(self):
        # Measured: the same document with check: false solves and lands at
        # c = 75.000008.  An executor that dropped check: from its sweep set
        # refuses the key outright; one that sent it to wiener_solve -- which
        # has no such parameter -- raises TypeError.  Neither produces this.
        product = _product(wiener_document(
            {**WIENER, "names": ["c"], "check": False},
            parameters={"c": CENTRE_LATENT}, at={"c": 76.0}))
        assert float(product["mean"]["c"]) == pytest.approx(75.0, abs=1e-3)
