"""The conjugate family's shared machine, driven directly.

The seam against ``test_config_exits_conjugate.py`` is mechanical rather than
thematic: **nothing in this module calls** :func:`run_document`.  Every test
here reaches the shared helpers -- ``_conjugate_block`` and ``_prior_kwargs``
from ``sections/conjugate_support.py``, ``_decided_sigma`` and
``_decided_model`` from ``sections/exit_support.py`` -- against a document
built by hand, which is exactly the surface the executors open with (the
executors themselves are ``sections/conjugate.py``, and the split between
those two modules is the seam this module's own imports follow).  Some go on
to hand
what came back to the real ``wiener_solve``, and that is still this side of
the seam: what no test here does is ask a document's ``runs:`` to execute, so
none of them depends on which kinds ``runs.py`` currently admits or on which
executor is registered for them.

The other module is the end-to-end half, where a document's ``runs:`` runs.
"""

import pytest

from rheplicant.config import ConfigError
from rheplicant.config.document import load_document
from rheplicant.config.sections import conjugate as conjugate_module
from rheplicant.config.sections.conjugate import _A28_GLS_CLAUSES
from rheplicant.config.sections.conjugate_support import (
    _conjugate_block,
    _prior_kwargs,
)
from rheplicant.config.sections.exit_support import _decided_model, _decided_sigma
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
# `_decided_model` took two positional arguments until Plan 3A's Task 10 and
# now takes `wants` and `instead` as REQUIRED keyword-only clauses -- the fix
# for one sentence written for `conjugate.gls` and inherited by `npe.py`,
# which does neither of the two things it claims.  The four direct calls
# below spread `conjugate.py`'s OWN mapping rather than a copy of it: the
# only document that reaches that call site's raise is one the pre-flight
# pass now refuses at P-1, so a copied clause here would pin nothing and a
# reword of the production one would leave these four green.


LINEARITY_DECLINED = {"linearity": {"mode": "skip",
                      "reason": "this fixture declares linear: true on a "
                                "latent the prediction is not affine in, on "
                                "purpose, so that the exit-level check: has "
                                "a lever"}}


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
        nonlinear = conjugate_built(
            inference={**NONLINEAR_LATENT, "checks": LINEARITY_DECLINED})
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
        assert isinstance(_decided_model(spec(), ONE, **_A28_GLS_CLAUSES),
                          HomoscedasticNoise)

    def test_what_A27_refuses_is_exactly_what_gls_takes(self):
        """The mirror: the radiometer the array route refuses is gls's input."""
        built = conjugate_built(noise=RADIOMETER)
        assert isinstance(_decided_model(spec(), built, **_A28_GLS_CLAUSES),
                          RadiometerNoise)

    def test_a_decided_array_is_refused_naming_conjugate_wiener(self):
        """Run it as gls, so the named alternative is not the interpolated kind.

        Under spec()'s default kind the message would carry
        "kind: conjugate.wiener" from the interpolation alone, and this match
        would pin nothing: gls is the kind that actually calls this.

        **These three assertions are `be2027b`'s, restored.**  Plan 3A's Task
        10 replaced them with an equality on a REWORDED sentence -- *"as a
        RULE"* and *"is not a rule"* where this message has always said *"as
        a model"* and *"has no fixed point to iterate"* -- which is a fifth
        message changed where plan §2.3 designates four, and a test rewritten
        to agree with the change is a test that has stopped being evidence.
        The equality pin that belongs beside them is
        `test_the_gls_refusal_is_be2027b_verbatim`, below, and it is written
        against the pre-move text rather than against whatever the code says
        today.
        """
        built = conjugate_built(noise=FROZEN)
        with pytest.raises(ConfigError, match="conjugate.wiener") as caught:
            _decided_model(spec(kind="conjugate.gls"), built,
                           **_A28_GLS_CLAUSES)
        assert "check A28" in str(caught.value)
        assert "radiometer_frozen" in str(caught.value)

    def test_the_gls_refusal_is_be2027b_verbatim(self):
        """A28's gls sentence, to the character, against the PRE-MOVE text.

        This is the pin `test_config_preflight.py::TestNoMovedMessageWasReworded`
        forgives the three vanished `be2027b` literals for: the sentence is
        assembled from `conjugate._A28_GLS_CLAUSES` and
        `exit_support._decided_model`'s template now, rather than written out
        in one place, so a source-literal guard cannot see it and something
        has to.

        Four clauses are the caller's -- `wants`, `reads`, `because`,
        `instead` -- and this string is what they have to add up to.  Kills
        any of the four being reworded, the template between them being
        reworded, and a fifth caller's words leaking into this one.

        The expected text was read off `git show be2027b:src/rheplicant/
        config/sections/exit_support.py`, not off the code beside it.
        """
        built = conjugate_built(noise=FROZEN)
        with pytest.raises(ConfigError) as caught:
            _decided_model(spec(kind="conjugate.gls"), built,
                           **_A28_GLS_CLAUSES)
        assert str(caught.value) == (
            "runs['conjugate.gls']: kind: conjugate.gls solves for the "
            "covariance a PREDICTION-DEPENDENT sigma implies, so it reads "
            "inference.noise as a model; inference.noise.kind: "
            "radiometer_frozen decides its sigma into an array before any "
            "run sees it, and a decided array has no fixed point to iterate "
            "(check A28). Declare inference.noise.kind: radiometer to "
            "iterate the rule, or run kind: conjugate.wiener, which is what "
            "a decided sigma wants.")

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
            _decided_model(spec(kind="conjugate.gls"), built,
                           **_A28_GLS_CLAUSES)
        assert "check A28" not in str(got.value)
