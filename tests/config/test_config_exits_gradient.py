"""kind: gradient -- one differentiation of a named objective.

Every expected number here is a closed form in two measured constants, not a
recorded output.  The document's fit twin is ``g * s`` where ``s`` is the
global_signal trough alone; on the 16x8 grid at depth 0.5 that gives
``sum(s^2) = 9.9231749`` and ``sum(s) = -27.840679``, both measured against
this repo at ``c11270d`` through
:func:`~tests.config.exit_helpers.diagnostic_document`.  Writing each
derivative out in those terms is what makes a wrong objective, a wrong
evaluation point or a dropped key show up as a NUMBER rather than as a shape.

The document comes from ``exit_helpers`` rather than being built here, and
that is load-bearing rather than tidy: it keeps ``synthetic_document``'s
stochastic ``noise`` node in ``model:`` and repairs it away in
``inference.twin.without:``, so ``built.twin`` and
``built.inference.fit_twin`` are DIFFERENT objects and every assertion below
is a statement about the second.  Build the fit twin by deleting ``noise``
from ``model:`` instead and the two become the same object, at which point
nothing here can tell an executor that differentiates one from an executor
that differentiates the other.  :class:`TestTheSharedFixturesDiscriminate`
asserts that property for this family and for the conjugate one.
"""

import math

import jax.numpy as jnp
import pytest

from rheplicant.config import ConfigError, load_document
from rheplicant.config.sections.diagnostics import _chi2
from rheplicant.config.sections.runs import run_document
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import build_forward_fn, mean_squared_error
from tests.config.exit_helpers import (
    TRUTH_G,
    WIENER,
    _repaired,
    conjugate_built,
    diagnostic_document,
    spec,
)

SUM_SQ_SIGNAL = 9.9231749
SUM_SIGNAL = -27.840679
N_SAMPLES = 128
SIGMA = 0.05
TRUTH = 1.5
INIT = 1.0
MODEL_GAIN = 1.1
DEPTH = 0.5

#: ``d chi2 / dg`` at the declared ``init:`` -- the number three classes
#: below are anchored on.  Negative: the fit wants a larger gain.
CHI2_AT_INIT = -2.0 * (TRUTH - INIT) * SUM_SQ_SIGNAL / SIGMA ** 2  # -3969.27

PARAMETERS = {"g": {"init": INIT, "linear": True, "into": "gain.gain",
                    "prior": {"normal": {"loc": 1.0, "scale": 0.05}}}}
HOMOSCEDASTIC = {
    "parameters": PARAMETERS,
    "noise": {"kind": "homoscedastic", "sigma": {"value": SIGMA,
                                                 "unit": "K"}},
    "observed": {"from": "simulation", "at": {"g": TRUTH}},
}
# f = 1/sqrt(channel_width * integration_time) = 1 exactly, which is what
# puts the log-determinant term and the weighted sum on the same scale.
RADIOMETER_FACTS = {"channel_width": {"value": 0.5, "unit": "Hz"},
                    "integration_time": {"value": 2.0, "unit": "s"}}


def quarter_mse(prediction, observed):
    """A quarter of the package mse -- importable, and observably not it."""
    return 0.25 * mean_squared_error(prediction, observed)


def needs_three(prediction, observed, weight):
    """An importable objective the seam cannot call: too many, not too few."""
    return weight * mean_squared_error(prediction, observed)


def gradient_document(run, inference=None):
    """The shared diagnostics document, carrying one gradient run."""
    return diagnostic_document(
        run, inference=HOMOSCEDASTIC if inference is None else inference)


def gradient_of(options, inference=None, path="gain.gain"):
    """One gradient run, executed, and the named path's derivative."""
    results = run_document(gradient_document({"kind": "gradient", **options},
                                             inference))
    return float(results["gradient"].product[path])


class TestTheNumberIsTheDerivative:
    """The gradient of chi2 on a linear model, predicted then measured.

    chi2 = sum((obs - g s)^2) / sigma^2 with obs = 1.5 s, so
    d chi2 / dg = -2 (1.5 - g) sum(s^2) / sigma^2.  At the declared init
    g = 1.0 that is negative -- the fit wants a larger gain -- and its size
    is fixed to four figures.  A test asserting only a shape, or only that
    the result is not None, would pass against jax.grad of the wrong
    objective, the wrong sign convention and the wrong evaluation point.
    """

    def test_chi2_has_the_sign_and_the_magnitude_the_model_predicts(self):
        measured = gradient_of({"objective": "chi2", "of": "gain.gain"})
        assert measured < 0.0
        assert measured == pytest.approx(CHI2_AT_INIT, rel=1e-4)

    def test_the_point_is_the_declared_init_not_the_models_own_leaf(self):
        # The model writes gain: 1.1 and the latent declares init: 1.0.
        # Differentiating the unbound fit_twin gives -3175.42; binding the
        # latents at their inits gives -3969.27.  Both are finite, negative
        # and correctly shaped, so only the number tells them apart.
        wrong = -2.0 * (TRUTH - MODEL_GAIN) * SUM_SQ_SIGNAL / SIGMA ** 2
        measured = gradient_of({"objective": "chi2", "of": "gain.gain"})
        assert measured == pytest.approx(CHI2_AT_INIT, rel=1e-4)
        assert measured != pytest.approx(wrong, rel=1e-2)

    def test_at_moves_the_evaluation_point(self):
        # Two at: values, two gradients, and the closed form says they are
        # exact negatives of each other about the truth.  A run that drops
        # at: -- or that merges the overrides UNDER the inits rather than
        # over them -- returns the same number three times.
        below = gradient_of({"objective": "chi2", "of": "gain.gain",
                             "at": {"g": 1.0}})
        truth = gradient_of({"objective": "chi2", "of": "gain.gain",
                             "at": {"g": TRUTH}})
        above = gradient_of({"objective": "chi2", "of": "gain.gain",
                             "at": {"g": 2.0}})
        assert below == pytest.approx(CHI2_AT_INIT, rel=1e-4)
        assert above == pytest.approx(-CHI2_AT_INIT, rel=1e-4)
        assert truth == pytest.approx(0.0, abs=1e-2)
        assert below < truth < above

    def test_at_naming_no_latent_is_refused(self):
        """Matched on the CLAUSE, not on the section name.

        ``at:`` has two neighbouring refusals here and both spell
        ``inference.parameters``: this one, and the no-latents one below.  A
        ``match="inference.parameters"`` is satisfied by either, so it would
        keep passing with this branch deleted.
        """
        with pytest.raises(
                ConfigError,
                match=r"at: names \['h'\], which inference.parameters does "
                      r"not declare") as caught:
            gradient_of({"objective": "chi2", "of": "gain.gain",
                         "at": {"h": 1.0}})
        assert "it declares ['g']" in str(caught.value)

    def test_at_without_parameters_is_refused_naming_the_alternative(self):
        """The other half of the pair, matched on what only it says."""
        with pytest.raises(
                ConfigError,
                match="this document declares no inference.parameters"
                ) as caught:
            gradient_of({"objective": "sum_squares", "of": "gain.gain",
                         "at": {"g": 1.0}}, inference={})
        assert "what dropping at: asks for" in str(caught.value)
        assert "does not declare" not in str(caught.value)

    def test_without_parameters_the_twins_own_leaves_are_the_point(self):
        # No latents, so nothing is bound: the point is the model's own
        # gain: 1.1, and d/dg sum((g s)^2) = 2 g sum(s^2).
        assert gradient_of({"objective": "sum_squares", "of": "gain.gain"},
                           inference={}) == pytest.approx(
            2.0 * MODEL_GAIN * SUM_SQ_SIGNAL, rel=1e-4)


class TestEachObjectiveIsItsOwnNumber:
    """Four names, four derivatives, spanning four decades and both signs.

    Pinning each one separately is what kills a table that maps every name
    onto one closure, or that resolves ``objective:`` and then differentiates
    something else.
    """

    CHI2 = CHI2_AT_INIT
    SUM_SQUARES = 2.0 * INIT * SUM_SQ_SIGNAL
    MEAN = SUM_SIGNAL / N_SAMPLES
    MSE = 2.0 * (INIT - TRUTH) * SUM_SQ_SIGNAL / N_SAMPLES

    def test_the_four_objectives_disagree_on_one_document(self):
        measured = {name: gradient_of({"objective": name, "of": "gain.gain"})
                    for name in ("chi2", "sum_squares", "mean", "mse")}
        assert measured["chi2"] == pytest.approx(self.CHI2, rel=1e-4)
        assert measured["sum_squares"] == pytest.approx(self.SUM_SQUARES,
                                                        rel=1e-4)
        assert measured["mean"] == pytest.approx(self.MEAN, rel=1e-4)
        assert measured["mse"] == pytest.approx(self.MSE, rel=1e-4)

    def test_no_two_of_the_four_could_be_confused_on_this_document(self):
        """The premise the four pins rest on, asserted rather than assumed.

        Two objectives that happened to agree HERE would both pass their own
        pin while a table mapping one name onto the other's closure went
        unnoticed.  Measured: -3969.27, +19.85, -0.2175, -0.07752 -- four
        decades and both signs, and the closest pair (mean and mse) is 2.8x
        apart, which is far outside every rel=1e-4 above.
        """
        values = [gradient_of({"objective": name, "of": "gain.gain"})
                  for name in ("chi2", "sum_squares", "mean", "mse")]
        for index, one in enumerate(values):
            for other in values[index + 1:]:
                assert abs(one - other) > 0.5 * min(abs(one), abs(other))

    def test_sum_squares_and_mean_read_no_data(self):
        # Pure in the prediction: a document with no observed and no noise
        # still differentiates them.  An executor that resolves observed
        # for every objective refuses here instead.
        bare = {"parameters": PARAMETERS}
        assert gradient_of({"objective": "sum_squares", "of": "gain.gain"},
                           bare) == pytest.approx(self.SUM_SQUARES, rel=1e-4)
        assert gradient_of({"objective": "mean", "of": "gain.gain"},
                           bare) == pytest.approx(self.MEAN, rel=1e-4)

    def test_chi2_without_a_noise_declaration_is_refused(self):
        with pytest.raises(ConfigError,
                           match="weighs residuals with inference.noise"):
            gradient_of({"objective": "chi2", "of": "gain.gain"},
                        {"parameters": PARAMETERS})

    def test_mse_without_observed_is_refused(self):
        with pytest.raises(ConfigError,
                           match="compares against inference.observed"):
            gradient_of({"objective": "mse", "of": "gain.gain"},
                        {"parameters": PARAMETERS,
                         "noise": {"kind": "homoscedastic",
                                   "sigma": {"value": SIGMA, "unit": "K"}}})

    def test_a_python_objective_is_the_one_differentiated(self):
        # quarter_mse scores exactly a quarter of mse everywhere, so the
        # gradient is a quarter of mse's.  A run that quietly falls back to
        # a named objective cannot produce the ratio.
        quarter = gradient_of({"objective": {
            "python": "tests.config.test_config_exits_gradient:quarter_mse"},
            "of": "gain.gain"})
        assert quarter == pytest.approx(0.25 * self.MSE, rel=1e-4)
        assert quarter != pytest.approx(self.MSE, rel=1e-2)

    def test_an_unnamed_objective_is_refused(self):
        """Matched on ``is required``: the unknown-name refusal below names
        ``objective:`` too, so a bare ``match="objective:"`` is satisfied by
        either and pins neither."""
        with pytest.raises(ConfigError, match="objective: is required"):
            gradient_of({"of": "gain.gain"})

    def test_an_unknown_objective_names_the_four(self):
        with pytest.raises(ConfigError,
                           match="objective: is one of") as caught:
            gradient_of({"objective": "chisq", "of": "gain.gain"})
        message = str(caught.value)
        assert all(name in message
                   for name in ("chi2", "mean", "mse", "sum_squares"))
        assert "chisq" in message


class TestThePythonSeamRefusesInThisLayersCurrency:
    """What the ``{python: 'mod:fn'}`` hatch does when the callable is wrong.

    Decision D-C11 records that the hatch is "recorded, not restricted -- the
    hatch exists to do what the schema cannot express".  Asking whether a
    callable can be CALLED the way this seam calls it, and whether its output
    is the shape ``jax.grad`` requires, restricts nothing a working objective
    can do; both are contract checks and neither looks at what the function
    computes.

    Measured through ``run_document`` before the two guards, and both reached
    the user as raw Python naming no run:

        {python: "math:sqrt"}      TypeError: math.sqrt() takes exactly one
                                   argument (2 given)
        {python: "jax.numpy:add"}  TypeError: Gradient only defined for
                                   scalar-output functions. Output had
                                   shape: (16, 8).

    The first is the LIKELY mistake, not an exotic one: three of the four
    named objectives read no data, so ``def my_objective(prediction)`` is the
    obvious analogue to write.

    ``grep -rn "raise " src/rheplicant/config/ | grep -v ConfigError`` shows
    nothing here, and never would have: these escaped through a CALL, not a
    raise.  That grep is not sufficient evidence for this layer's
    single-ConfigError property.
    """

    def test_a_one_argument_objective_is_refused_naming_the_signature(self):
        with pytest.raises(ConfigError,
                           match="cannot be called as "
                                 r"\(prediction, observed\)") as caught:
            gradient_of({"objective": {"python": "math:sqrt"},
                         "of": "gain.gain"})
        message = str(caught.value)
        assert "runs['gradient']: " in message
        assert "math:sqrt" in message
        assert "sum_squares and mean ignore observed" in message

    def test_a_three_argument_objective_is_refused_by_the_same_clause(self):
        with pytest.raises(ConfigError, match="cannot be called as"):
            gradient_of({"objective": {
                "python": "tests.config.test_config_exits_gradient:"
                          "needs_three"},
                "of": "gain.gain"})

    def test_an_objective_that_does_not_reduce_is_refused(self):
        """The shape jax.grad needs, said in this layer's currency.

        ``jnp.add`` takes the pair happily and returns the (16, 8) grid, so
        the arity guard passes it and only the output check can refuse it.
        """
        with pytest.raises(ConfigError,
                           match=r"returned shape \(16, 8\)") as caught:
            gradient_of({"objective": {"python": "jax.numpy:add"},
                         "of": "gain.gain"})
        assert "SCALAR score" in str(caught.value)
        assert "runs['gradient']: " in str(caught.value)

    def test_a_two_argument_objective_that_reduces_is_accepted(self):
        """The other side of both guards, so neither can be a blanket no.

        ``quarter_mse`` above already proves the seam runs; this asserts that
        a callable whose signature is a plain ``(prediction, observed)``
        pair -- no ``/``, no defaults -- passes the arity check rather than
        being caught by an over-eager one.
        """
        assert gradient_of({"objective": {
            "python": "tests.config.test_config_exits_gradient:quarter_mse"},
            "of": "gain.gain"}) == pytest.approx(0.25 * (
                2.0 * (INIT - TRUTH) * SUM_SQ_SIGNAL / N_SAMPLES), rel=1e-4)

    def test_a_python_objective_declared_beside_another_key_is_refused(self):
        """``{python: ..., loss: ...}`` is not the hatch, and says so.

        ``_sweep`` guards the RUN's keys, not this nested mapping, so the
        exclusivity leg of ``set(objective) == {'python'}`` is the only thing
        watching -- and relaxing it to ``'python' in objective`` survived the
        whole suite until this test.
        """
        with pytest.raises(ConfigError, match="objective: is one of"):
            gradient_of({"objective": {
                "python": "tests.config.test_config_exits_gradient:"
                          "quarter_mse", "loss": "mse"},
                "of": "gain.gain"})

    def test_a_target_that_imports_nothing_is_still_the_hatch_s_own_refusal(
            self):
        with pytest.raises(ConfigError, match="no.such.module"):
            gradient_of({"objective": {"python": "no.such.module:fn"},
                         "of": "gain.gain"})


class TestOfNamesTheLeaves:
    def test_a_list_of_paths_gives_one_gradient_each(self):
        # The prediction is bilinear in gain and depth, so at gain 1.0 and
        # depth 0.5 the depth derivative is exactly (gain/depth) = 2 times
        # the gain derivative.  An executor that differentiates only the
        # first path hands back None for the second and raises in float().
        #
        # The two paths are declared in REVERSE alphabetical order on
        # purpose, and the product's key order is asserted with list() rather
        # than sorted(): _of_paths promises DECLARED order and _run_gradient
        # promises {declared path: gradient}, and a sorted() assertion is
        # satisfied by an executor that re-sorts either one -- which is the
        # bug diagnostics.py:248-250 warns about for score_directions, and
        # which survived the whole suite until this line stopped sorting.
        results = run_document(gradient_document(
            {"kind": "gradient", "objective": "chi2",
             "of": ["global_signal.depth", "gain.gain"]}))
        product = results["gradient"].product
        assert list(product) == ["global_signal.depth", "gain.gain"]
        assert float(product["gain.gain"]) == pytest.approx(CHI2_AT_INIT,
                                                            rel=1e-4)
        assert float(product["global_signal.depth"]) == pytest.approx(
            (INIT / DEPTH) * CHI2_AT_INIT, rel=1e-4)

    def test_a_repeated_path_is_refused_the_way_names_is(self):
        """One key for a two-path ask, and no refusal -- measured, before
        this guard: ``of: [gain.gain, gain.gain]`` was ACCEPTED and returned
        ``{'gain.gain': ...}``.  That is what ``_names`` refuses 116 lines
        earlier in the same module, under a paragraph about a product with
        fewer keys than the document named being read off by one; the hole
        was closed on ``names:`` and open on ``of:``.
        """
        with pytest.raises(
                ConfigError,
                match=r"of: lists \['gain.gain'\] more than once") as caught:
            gradient_of({"objective": "chi2",
                         "of": ["gain.gain", "global_signal.depth",
                                "gain.gain"]})
        assert "off by one" in str(caught.value)

    def test_of_alone_decides_which_leaf_a_single_run_differentiates(self):
        """``of:`` naming ONE leaf that is not the document's latent.

        The pair test above could pass against an executor that always
        differentiated both leaves and keyed the answer by whatever ``of:``
        listed.  This one names ``global_signal.depth`` alone, and the
        product carries that key and the depth number -- twice the gain's,
        and nothing else in this module.
        """
        product = run_document(gradient_document(
            {"kind": "gradient", "objective": "chi2",
             "of": "global_signal.depth"}))["gradient"].product
        assert list(product) == ["global_signal.depth"]
        assert float(product["global_signal.depth"]) == pytest.approx(
            (INIT / DEPTH) * CHI2_AT_INIT, rel=1e-4)

    def test_of_is_required(self):
        with pytest.raises(ConfigError, match="of: is required"):
            gradient_of({"objective": "chi2"})

    def test_of_is_a_path_or_a_list_of_paths(self):
        for bad in (7, [], ["gain.gain", 7], {"gain.gain": 1}):
            with pytest.raises(ConfigError,
                               match="of: is a path or a non-empty list"):
                gradient_of({"objective": "chi2", "of": bad})

    def test_a_path_that_stops_short_of_a_leaf_is_refused(self):
        with pytest.raises(ConfigError, match="which is not a leaf"):
            gradient_of({"objective": "chi2", "of": "gain"})

    def test_unknown_keys_are_swept(self):
        with pytest.raises(ConfigError,
                           match=r"does not take \['objectives'\]") as caught:
            gradient_of({"objective": "chi2", "of": "gain.gain",
                         "objectives": "chi2"})
        assert "it takes ['at', 'objective', 'of']" in str(caught.value)

    def test_gradient_takes_neither_of_optimize_s_two_required_knobs(self):
        """``optimize`` REQUIRES learning_rate and n_steps (exits.py:96-103).

        ``gradient`` differentiates once and runs no optimiser, so both are
        swept rather than accepted-and-ignored -- which is the failure an
        executor copied from ``optimize`` would produce.
        """
        for knob, value in (("learning_rate", 0.01), ("n_steps", 10)):
            with pytest.raises(ConfigError,
                               match=f"does not take \\['{knob}'\\]"):
                gradient_of({"objective": "chi2", "of": "gain.gain",
                             knob: value})


class TestIncludeLogdetGetsItsFirstConsumer:
    """chi2 under a prediction-dependent sigma, with and without the term.

    RadiometerNoise has sigma = |prediction| * f, and the facts above make
    f = 1 exactly.  Then, at g = 1 with obs = 1.5 s:

        sum r^2/sigma^2 = N (1.5 - g)^2 / g^2   ->  d/dg = -1.5 N = -192
        sum log 2 pi sigma^2 = N log(2 pi g^2 f^2) + const
                                                ->  d/dg = 2 N / g = +256

    so keeping the term does not merely shift the number, it flips the
    sign.  An executor that hard-codes include_logdet either way gets one
    of the two documents wrong.

    ``include_logdet`` has THREE states, and the third -- undeclared, which
    ``NoiseBuild`` records as None -- is reachable only under a sigma that
    does not depend on the prediction (noise.py:104-109 requires the key for
    ``kind: radiometer`` and refuses it everywhere else).  There the
    log-determinant is an additive constant and no gradient can see it, so
    the None state is pinned by EVALUATING the objective rather than by
    differentiating it.
    """

    def _radiometer(self, include_logdet):
        return {**HOMOSCEDASTIC,
                "noise": {"kind": "radiometer", **RADIOMETER_FACTS,
                          "include_logdet": include_logdet}}

    def test_the_declared_include_logdet_changes_the_number_and_its_sign(self):
        without = gradient_of({"objective": "chi2", "of": "gain.gain"},
                              self._radiometer(False))
        with_it = gradient_of({"objective": "chi2", "of": "gain.gain"},
                              self._radiometer(True))
        assert without == pytest.approx(-1.5 * N_SAMPLES, rel=1e-3)
        assert with_it == pytest.approx(-1.5 * N_SAMPLES + 2 * N_SAMPLES,
                                        rel=1e-3)
        assert without < 0.0 < with_it

    def test_the_undeclared_state_keeps_the_packages_own_term(self):
        """The third state, made observable by not differentiating it.

        Under this homoscedastic document ``include_logdet`` is None -- the
        key is refused for every kind but ``radiometer`` -- and the term it
        controls is then a constant, so the gradient below is the same
        number for all three states (MEASURED: -3969.272216796875 for None,
        True and False alike).  The objective is NOT the same, and this is
        where that shows: passing ``bool(None)`` instead of passing nothing
        drops 128 * log(2 pi sigma^2) = -531.66 from a weighted sum of
        992.32, which is a 2.2x change in the value the closure returns.
        """
        built = load_document(gradient_document(
            {"kind": "gradient", "objective": "chi2", "of": "gain.gain"}))
        assert built.inference.noise.include_logdet is None
        space = built.inference.space
        bound = space.bind(built.inference.fit_twin,
                           dict(space.initial_values()))
        prediction = bound(built.state).data
        observed = built.inference.observed.entries["primary"]
        weighted = float(jnp.sum(((observed - prediction) / SIGMA) ** 2))
        logdet = N_SAMPLES * math.log(2.0 * math.pi * SIGMA ** 2)
        measured = float(_chi2(spec("gradient"), built)(prediction))
        assert weighted == pytest.approx(992.318, rel=1e-4)
        assert measured == pytest.approx(weighted + logdet, rel=1e-4)
        assert measured != pytest.approx(weighted, rel=1e-2)

    def test_an_undeclared_include_logdet_still_lands_on_its_number(self):
        assert gradient_of({"objective": "chi2", "of": "gain.gain"}) == \
            pytest.approx(CHI2_AT_INIT, rel=1e-4)

    def test_a_frozen_sigma_weighs_without_a_noise_model(self):
        # radiometer_frozen decides an ARRAY, not a NoiseModel, so
        # NoiseModelLikelihood cannot be built at all.  sigma = 1.5|s| here
        # (frozen off the observed data with f = 1), which cancels the
        # signal out of the weighted sum entirely:
        #     chi2 = N (1.5 - g)^2 / 1.5^2  ->  d/dg = -2 N (1.5 - g)/2.25
        frozen = {**HOMOSCEDASTIC,
                  "noise": {"kind": "radiometer_frozen", "source": "observed",
                            **RADIOMETER_FACTS}}
        assert gradient_of({"objective": "chi2", "of": "gain.gain"},
                           frozen) == pytest.approx(
            -2.0 * N_SAMPLES * (TRUTH - INIT) / TRUTH ** 2, rel=1e-3)


class TestTheSharedFixturesDiscriminate:
    """Both hoisted documents keep ``twin`` and ``fit_twin`` apart.

    They live here rather than beside the fixtures they guard because this is
    the module whose commit made the repair: ``kind: gradient`` was its third
    caller and the reason the family moved into ``exit_helpers``.  What they
    pin is the reason the fixtures are shaped the way they are:
    ``synthetic_document``'s stochastic ``noise`` node stays in ``model:``
    and is repaired away in ``inference.twin.without:``, so ``built.twin``
    still carries NoiseOperator.  Build the fit twin by DELETING the node
    instead and the two become one object, at which point an executor
    reaching for the wrong one is invisible to every test in six modules.

    Measured: pointing ``_conjugate_block``'s ``linear_operator`` at
    ``built.twin`` fails 74 tests with the repair in place -- 20 shared, 22
    conjugate, 17 gcr, 15 gls -- and none without it.  And restoring the
    pre-repair fixture wholesale leaves every other number in ``tests/config``
    untouched and fails exactly four tests: T7's
    ``test_the_fit_twin_is_what_it_differentiates`` and three of the five
    below.
    """

    def test_the_diagnostics_document_differentiates_the_fit_twin(self):
        built = load_document(gradient_document(
            {"kind": "gradient", "objective": "chi2", "of": "gain.gain"}))
        assert built.twin is not built.inference.fit_twin
        with pytest.raises(ParameterSpaceError, match="NoiseOperator"):
            build_forward_fn(built.twin, built.state)
        assert gradient_of({"objective": "chi2", "of": "gain.gain"}) == \
            pytest.approx(CHI2_AT_INIT, rel=1e-4)

    def test_the_simulated_data_still_comes_off_the_fit_twin(self):
        """The other half of the repair, and the half that pins numbers.

        ``observed: {from: simulation}`` defaults to ``twin: full``, so a
        model that carries NoiseOperator would simulate a NOISE REALISATION
        and every closed form in this module would be wrong by a random
        amount.  The fixture declares ``twin: fit``; the data is therefore
        the exact deterministic forward at the truth, and obs/pred is 1.5 to
        float32 roundoff rather than scattered.
        """
        built = load_document(gradient_document(
            {"kind": "gradient", "objective": "chi2", "of": "gain.gain"}))
        space = built.inference.space
        bound = space.bind(built.inference.fit_twin,
                           dict(space.initial_values()))
        ratio = (built.inference.observed.entries["primary"]
                 / bound(built.state).data)
        assert float(jnp.max(jnp.abs(ratio - TRUTH))) < 1.0e-5

    def test_the_conjugate_document_keeps_the_two_twins_apart_too(self):
        """The same repair on ``conjugate_document``, which Tasks 3-6 use.

        Its own suite cannot state this: every conjugate assertion is about
        a solve, and a solve reaching for ``built.twin`` merely raises.
        """
        built = conjugate_built(WIENER)
        assert built.twin is not built.inference.fit_twin
        with pytest.raises(ParameterSpaceError, match="NoiseOperator"):
            built.inference.space.forward_fn(built.twin, built.state)
        space = built.inference.space
        truth = built.inference.observed.at["primary"]
        assert float(truth["g"]) == pytest.approx(TRUTH_G)
        at_truth = space.bind(built.inference.fit_twin,
                              {**space.initial_values(), **truth})
        assert float(jnp.max(jnp.abs(
            built.inference.observed.entries["primary"]
            - at_truth(built.state).data))) < 1.0e-5

    def test_the_repair_is_supplied_when_the_block_says_nothing(self):
        """Both keys default, which is what "cannot drop the repair" means."""
        repaired = _repaired({"parameters": {},
                              "observed": {"from": "simulation"}})
        assert repaired["twin"] == {"without": ["noise"]}
        assert repaired["observed"] == {"from": "simulation", "twin": "fit"}

    def test_a_block_that_declares_either_key_keeps_its_own(self):
        """A default, not an override -- and the difference is invisible
        until someone declares one of these and silently gets the other.

        Task 11's ``predict`` plausibly wants noise-realised data
        (``observed.twin: full``), and ``test_config_exits_estimators.py``
        already builds documents with ``twin: {replace: ...}``, so a later
        caller reaching for one of these is not hypothetical.  Measured
        before this change: both declarations were discarded in silence.
        """
        mine = _repaired({"twin": {"without": ["gain"]},
                          "observed": {"from": "simulation", "twin": "full"}})
        assert mine["twin"] == {"without": ["gain"]}
        assert mine["observed"]["twin"] == "full"

    def test_the_repair_never_mutates_the_block_it_is_handed(self):
        block = {"parameters": {}, "observed": {"from": "simulation"}}
        _repaired(block)
        assert block == {"parameters": {},
                         "observed": {"from": "simulation"}}
