"""``noise_std`` has to say which axis it runs along (defect A6).

Every docstring in the package used to describe ``noise_std`` as "scalar or
broadcastable to the data", under which a 1-D vector against a SQUARE grid has
two legitimate readings — one sigma per time sample, or one per frequency
channel — and no way to tell them apart. Both broadcast, both return the right
shape, and they weight completely different sets of samples.

Measured on an 8x8 grid with one ``(8,)`` per-time gain latent and
``sigma = linspace(0.01, 1.0, 8)``, before the rule existed::

    noise_std (8,)   -> sigma('gt') shape (8,)  [0.00010 .. 0.00010]
    noise_std (8,1)  -> sigma('gt') shape (8,)  [0.00004 .. 0.00354]
    noise_std (1,8)  -> sigma('gt') shape (8,)  [0.00010 .. 0.00010]

All three succeeded. The per-TIME vector was silently applied per-FREQUENCY,
flattening an error bar that genuinely spans ~90x. So the fixture here is
deliberately square and its sigma deliberately asymmetric: a symmetric fixture
cannot distinguish the two readings and would pass whichever one the code
happens to take.

The guard reads SHAPES only. That is not incidental — a NaN in ``noise_std``
defeats every comparison-based guard (``nan < 0`` is ``False``), so a value
check written here would be the one thing that sails through. Shapes are
integers; there is nothing for a NaN to defeat.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, Environment, State
from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference import (
    Bind,
    FlaggedNoise,
    HomoscedasticNoise,
    Latent,
    ParameterSpace,
    RadiometerNoise,
    condition_estimate,
    gcr_sample,
    linear_operator,
    wiener_solve,
)
from rheplicant.inference.noise import check_noise_std_axis, inverse_variance
from rheplicant.inference.uncertainty import (
    as_noise_model,
    fisher_information,
    parameter_covariance,
)
from rheplicant.radio import GainOperator, SkyOperator, assemble

N = 8
SKY = 100.0
# Asymmetric on purpose: a 90x spread from first to last, so the two readings
# of the same vector cannot produce the same number by accident.
SIGMA_VECTOR = jnp.linspace(0.01, 1.0, N)


@pytest.fixture
def square_state():
    """A deliberately SQUARE (n_time == n_freq) grid — where the bug lives."""
    return _square_state()


def _square_state():
    """The fixture's body, callable directly.

    One test has to build this INSIDE a ``jax.enable_x64`` block, and a
    fixture is built before the test body runs. Two spellings of one grid is
    the shape this codebase pays for most often, so the grid is written once
    and the fixture is a caller like any other.
    """
    return State(
        coords=Coordinates(
            time=jnp.linspace(0.0, 7.0, N),
            freq=jnp.linspace(60e6, 85e6, N),
        ),
        env=Environment(temperature=jnp.array(280.0)),
        key=jax.random.key(0),
        meta={"telescope": "RHINO"},
    )


@pytest.fixture
def oblong_state():
    """The same model on a NON-square grid: one reading, so no refusal."""
    return State(
        coords=Coordinates(
            time=jnp.linspace(0.0, 7.0, N),
            freq=jnp.linspace(60e6, 85e6, N + 3),
        ),
        env=Environment(temperature=jnp.array(280.0)),
        key=jax.random.key(0),
        meta={"telescope": "RHINO"},
    )


@pytest.fixture
def gain_model():
    """``forward({'gt': (8,)}) -> (n_time, n_freq)`` — a per-TIME gain."""
    return _gain_model()


def _gain_model():
    """The fixture's body, callable directly — see :func:`_square_state`."""
    twin = assemble(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.ones(N)),
    )
    space = ParameterSpace(
        latents=[Latent("gt", init=jnp.ones(N))],
        bindings=[Bind("gt", into=lambda p: p["gain"].gain)],
    )
    return space, twin


class TestTheRuleItself:
    """``check_noise_std_axis`` is the whole contract, in one function."""

    def test_ambiguous_vector_on_a_square_grid_is_refused(self):
        with pytest.raises(StateValidationError, match="more than one"):
            check_noise_std_axis(SIGMA_VECTOR, (N, N), "caller")

    def test_the_refusal_names_both_readings_and_the_way_out(self):
        with pytest.raises(StateValidationError) as excinfo:
            check_noise_std_axis(SIGMA_VECTOR, (N, N), "wiener_solve")
        message = str(excinfo.value)
        assert "wiener_solve" in message
        assert "(8, 1)" in message and "(1, 8)" in message
        assert "axis 0" in message and "axis 1" in message

    def test_a_length_matching_one_axis_only_is_unambiguous(self):
        """The other branch: nothing to disambiguate, so nothing to refuse."""
        check_noise_std_axis(SIGMA_VECTOR, (N, N + 3), "caller")

    def test_an_explicit_column_passes(self):
        check_noise_std_axis(SIGMA_VECTOR[:, None], (N, N), "caller")

    def test_an_explicit_row_passes(self):
        check_noise_std_axis(SIGMA_VECTOR[None, :], (N, N), "caller")

    def test_a_scalar_passes(self):
        check_noise_std_axis(0.3, (N, N), "caller")
        check_noise_std_axis(jnp.asarray(0.3), (N, N), "caller")

    def test_a_1d_prediction_has_only_one_axis_to_read(self):
        check_noise_std_axis(SIGMA_VECTOR, (N,), "caller")

    def test_a_one_element_vector_is_a_scalar_wearing_an_axis(self):
        """Found by sweeping the rule rather than by reading it: ``(1,)``
        against a ``(1, 1)`` prediction matches BOTH axes, but every reading
        broadcasts the same single number, so refusing would be a false
        positive on a shape nobody could have got wrong."""
        check_noise_std_axis(jnp.ones(1), (1, 1), "caller")
        check_noise_std_axis(jnp.ones(1), (1, 1, 1), "caller")

    def test_a_longer_vector_at_the_same_corner_is_still_refused(self):
        """The other side of that exemption: length 1 is the only exempt one."""
        with pytest.raises(StateValidationError, match="more than one"):
            check_noise_std_axis(jnp.ones(2), (2, 2), "caller")

    def test_three_axes_of_the_same_length_are_refused_and_all_named(self):
        with pytest.raises(StateValidationError) as excinfo:
            check_noise_std_axis(SIGMA_VECTOR, (N, N, N), "caller")
        message = str(excinfo.value)
        assert "(8, 1, 1)" in message and "(1, 1, 8)" in message

    def test_a_homoscedastic_model_carries_the_same_ambiguity(self):
        """Wrapping the vector in the noise model does not resolve it."""
        with pytest.raises(StateValidationError, match="more than one"):
            check_noise_std_axis(
                HomoscedasticNoise(SIGMA_VECTOR), (N, N), "caller"
            )

    def test_flagging_does_not_hide_it_either(self):
        noise = FlaggedNoise(HomoscedasticNoise(SIGMA_VECTOR), jnp.zeros((N, N), bool))
        with pytest.raises(StateValidationError, match="more than one"):
            check_noise_std_axis(noise, (N, N), "caller")

    def test_a_prediction_dependent_model_has_no_free_axis_to_read(self):
        """``RadiometerNoise`` derives sigma FROM the prediction, so its shape
        is the prediction's — there is no vector whose axis could be misread."""
        check_noise_std_axis(RadiometerNoise(50.0, 2.0), (N, N), "caller")


class TestNaNCannotSlipPast:
    """The trap: a value check would let NaN through; a shape check cannot."""

    def test_a_nan_sigma_is_still_refused_when_the_axis_is_ambiguous(self):
        poisoned = SIGMA_VECTOR.at[3].set(jnp.nan)
        with pytest.raises(StateValidationError, match="more than one"):
            check_noise_std_axis(poisoned, (N, N), "caller")

    def test_a_nan_sigma_is_not_refused_by_THIS_rule_when_unambiguous(self):
        """The rule is about axes, not about values — it must not grow a
        second job it would do badly."""
        poisoned = SIGMA_VECTOR.at[3].set(jnp.nan)
        check_noise_std_axis(poisoned, (N, N + 3), "caller")


class TestAsNoiseModel:
    def test_the_shape_is_optional_so_existing_callers_are_untouched(self):
        """``as_noise_model`` is called from places that have no prediction in
        hand yet; omitting the shape must keep the old behaviour exactly."""
        noise = as_noise_model(SIGMA_VECTOR)
        assert isinstance(noise, HomoscedasticNoise)

    def test_the_shape_turns_the_rule_on(self):
        with pytest.raises(StateValidationError, match="more than one"):
            as_noise_model(SIGMA_VECTOR, prediction_shape=(N, N))

    def test_flags_still_wrap_when_the_shape_is_unambiguous(self):
        noise = as_noise_model(
            SIGMA_VECTOR,
            jnp.zeros((N, N + 3), bool),
            prediction_shape=(N, N + 3),
        )
        assert isinstance(noise, FlaggedNoise)


class TestFisherInformation:
    def test_the_ambiguous_vector_is_refused(self, square_state, gain_model):
        space, twin = gain_model
        forward, values0 = space.forward_fn(twin, square_state)
        with pytest.raises(StateValidationError, match="more than one"):
            fisher_information(forward, values0, noise_std=SIGMA_VECTOR)

    def test_the_two_explicit_readings_give_visibly_different_answers(self):
        """This is what the refusal is protecting: the numbers really do differ.

        Per-time weighting tracks sigma across the latent (0.00004 .. 0.00354);
        per-frequency weighting averages it away into a flat 0.00010.

        **Float64, and the model is built inside the block.** ``sigma`` spans
        100x here on purpose, and ``F = J^T N^-1 J`` SQUARES that, so the
        inverse runs at ``kappa = 1.0e4`` -- past what float32 carries
        (2.90e+03), and refused since ``parameter_covariance`` took the far
        side's ceiling (D29). That is not a nuisance about this test: the
        conditioning and the subject are the SAME 100x spread, so a version of
        this test that stayed in float32 would be asserting a closed form to
        rtol 1e-4 against an inverse that had spent more than half its digits.

        Inside the block and not merely around the call: ``jax.enable_x64`` is
        a tracing-time global, so an array built outside stays float32 and
        widening only the inverse recovers nothing -- measured on the far side
        at 2.45e-02 relative error against 2.41e-02 for doing nothing.
        """
        with jax.enable_x64(True):
            space, twin = _gain_model()
            forward, values0 = space.forward_fn(twin, _square_state())
            sigma = jnp.asarray(SIGMA_VECTOR, jnp.float64)

            per_time = parameter_covariance(
                fisher_information(forward, values0, noise_std=sigma[:, None])
            ).sigma("gt")
            per_freq = parameter_covariance(
                fisher_information(forward, values0, noise_std=sigma[None, :])
            ).sigma("gt")

            assert per_time.shape == per_freq.shape == (N,)
            # Per-time: sigma_i = noise_i / (sqrt(n_freq) * SKY), a 100x spread.
            assert jnp.allclose(per_time, sigma / (jnp.sqrt(N) * SKY), rtol=1e-4)
            assert float(per_time.max() / per_time.min()) > 50.0
            # Per-frequency: every time sample sees the same set of channels, so
            # the error bar is flat.
            assert jnp.allclose(per_freq, per_freq[0], rtol=1e-5)
            assert not jnp.allclose(per_time, per_freq, rtol=0.1)

    def test_the_widened_run_really_is_in_double(self):
        """The sibling of the test above, and the reason it can fail.

        A widened test whose arrays are still float32 passes for the wrong
        reason: the ceiling it was moved to clear would be the float32 one
        again, and nothing in the numbers would say so. This asserts the
        arithmetic it runs in, so "the block did not take" is a separate red
        from "the answer moved".
        """
        with jax.enable_x64(True):
            space, twin = _gain_model()
            forward, values0 = space.forward_fn(twin, _square_state())
            sigma = jnp.asarray(SIGMA_VECTOR, jnp.float64)
            fisher = fisher_information(forward, values0, noise_std=sigma[:, None])
        assert fisher.matrix.dtype == jnp.float64
        assert values0["gt"].dtype == jnp.float64

    def test_an_unambiguous_vector_still_works(self, oblong_state, gain_model):
        space, twin = gain_model
        forward, values0 = space.forward_fn(twin, oblong_state)
        sigma = jnp.linspace(0.01, 1.0, N + 3)  # matches the freq axis alone
        cov = parameter_covariance(
            fisher_information(forward, values0, noise_std=sigma)
        )
        assert cov.sigma("gt").shape == (N,)

    def test_a_scalar_is_never_ambiguous(self, square_state, gain_model):
        space, twin = gain_model
        forward, values0 = space.forward_fn(twin, square_state)
        cov = parameter_covariance(
            fisher_information(forward, values0, noise_std=0.5)
        )
        assert jnp.allclose(cov.sigma("gt"), 0.5 / (jnp.sqrt(N) * SKY), rtol=1e-4)


class TestWienerSolve:
    """``linear.py`` does not route through ``as_noise_model``, so the rule
    needs its own home in the solve's argument check until that is unified."""

    @pytest.fixture
    def square_block(self, square_state):
        twin = assemble(
            SkyOperator(amplitude=jnp.array(SKY)),
            GainOperator(gain=jnp.ones(N)),
        )
        space = ParameterSpace(
            latents=[Latent("gt", init=jnp.ones(N), linear=True)],
            bindings=[Bind("gt", into=lambda p: p["gain"].gain)],
        )
        return linear_operator(space, twin, square_state)

    def test_the_ambiguous_vector_is_refused(self, square_block, square_state):
        observed = jnp.full((N, N), SKY)
        with pytest.raises(StateValidationError, match="more than one"):
            wiener_solve(
                square_block, observed, noise_std=SIGMA_VECTOR, prior_std=1.0
            )

    def test_the_refusal_names_the_solve(self, square_block):
        observed = jnp.full((N, N), SKY)
        with pytest.raises(StateValidationError, match="wiener_solve"):
            wiener_solve(
                square_block, observed, noise_std=SIGMA_VECTOR, prior_std=1.0
            )

    def test_an_explicit_column_solves(self, square_block):
        observed = jnp.full((N, N), SKY)
        solved, _ = wiener_solve(
            square_block, observed, noise_std=SIGMA_VECTOR[:, None], prior_std=1.0
        )
        assert solved.shape == (N,)
        assert jnp.all(jnp.isfinite(solved))

    def test_an_explicit_row_solves_and_disagrees(self, square_block):
        """Both readings solve; they are not the same answer, which is the
        entire reason the vector may not be left to broadcasting.

        The data has to vary along BOTH axes for the mean to notice. With
        structure along time alone every channel of a row carries the same
        number, the weights cancel out of the weighted mean, and the two
        readings agree to five digits while the error bars differ by 90x —
        a fixture that would have passed this test without testing anything.
        """
        observed = (
            jnp.full((N, N), SKY)
            + jnp.linspace(-3.0, 3.0, N)[:, None]
            + 30.0 * jnp.linspace(-1.0, 1.0, N)[None, :]
        )
        per_time, _ = wiener_solve(
            square_block, observed, noise_std=SIGMA_VECTOR[:, None], prior_std=1.0
        )
        per_freq, _ = wiener_solve(
            square_block, observed, noise_std=SIGMA_VECTOR[None, :], prior_std=1.0
        )
        assert not jnp.allclose(per_time, per_freq, rtol=1e-3)

    def test_a_scalar_still_solves(self, square_block):
        observed = jnp.full((N, N), SKY)
        solved, _ = wiener_solve(
            square_block, observed, noise_std=0.5, prior_std=1.0
        )
        assert solved.shape == (N,)


class TestBothExitsAgree:
    """The mean and the draw refuse the same inputs.

    ``gcr_sample`` shares ``_check_solve_arguments`` with ``wiener_solve`` but
    did not pass ``noise_std`` to it, so the draw accepted an ambiguous vector
    that the mean refused. That asymmetry is worse than having no rule: a user
    who meets the refusal on ``estimate`` learns the argument is checked, and
    then carries the same array into ``sample``.

    It also lands harder on the draw. The mean applies ``noise_std`` once, in
    the weights; the draw applies it again in the fluctuation term, so a
    misread axis puts the *width* of every draw on the wrong axis, not merely
    the point.

    Written as a symmetry test rather than a second copy of the wiener_solve
    cases on purpose -- a copied test drifts, and it was the absence of a
    both-exits assertion that let the gap exist in the first place.
    """

    @pytest.fixture
    def square_block(self, square_state):
        twin = assemble(
            SkyOperator(amplitude=jnp.array(SKY)),
            GainOperator(gain=jnp.ones(N)),
        )
        space = ParameterSpace(
            latents=[Latent("gt", init=jnp.ones(N), linear=True)],
            bindings=[Bind("gt", into=lambda p: p["gain"].gain)],
        )
        return linear_operator(space, twin, square_state)

    @staticmethod
    def _exits(block, observed, noise_std):
        """Both exits as zero-argument thunks, so the two rows are identical
        apart from the exit itself."""
        return {
            "wiener_solve": lambda: wiener_solve(
                block, observed, noise_std=noise_std, prior_std=1.0
            ),
            "gcr_sample": lambda: gcr_sample(
                block, observed, noise_std=noise_std, prior_std=1.0,
                key=jax.random.PRNGKey(0),
            ),
        }

    def test_the_ambiguous_vector_is_refused_by_both(self, square_block, square_state):
        observed = jnp.full((N, N), SKY)
        for name, run in self._exits(square_block, observed, SIGMA_VECTOR).items():
            with pytest.raises(StateValidationError, match="more than one"):
                run()
            # ...and each names itself, so the message tells the caller which
            # exit they were at rather than which one happened to check.
            with pytest.raises(StateValidationError, match=name):
                run()

    @pytest.mark.parametrize(
        "reading",
        [
            pytest.param(SIGMA_VECTOR[:, None], id="explicit-column"),
            pytest.param(SIGMA_VECTOR[None, :], id="explicit-row"),
            pytest.param(0.5, id="scalar"),
        ],
    )
    def test_an_unambiguous_noise_std_is_accepted_by_both(self, square_block, reading):
        observed = jnp.full((N, N), SKY)
        for run in self._exits(square_block, observed, reading).values():
            value, _ = run()
            assert value.shape == (N,)
            assert jnp.all(jnp.isfinite(value))


class TestWhyTheRuleHasTwoHomes:
    """The two measurements behind not unifying the noise_std path.

    ``inference/linear.py`` does not call ``as_noise_model``; it takes the bare
    array to ``1 / sigma**2``. That duplication looks like debt, and was
    recorded as OWED until it was assessed. It is not debt, and these tests
    exist so the next author to reach for the unification meets the two
    consequences instead of rediscovering them.

    If the unification IS made deliberately, delete this class along with the
    paragraphs it supports in ``_check_solve_arguments`` -- a test pinning a
    difference that no longer exists is worse than no test.
    """

    @pytest.fixture
    def square_block(self, square_state):
        twin = assemble(
            SkyOperator(amplitude=jnp.array(SKY)),
            GainOperator(gain=jnp.ones(N)),
        )
        space = ParameterSpace(
            latents=[Latent("gt", init=jnp.ones(N), linear=True)],
            bindings=[Bind("gt", into=lambda p: p["gain"].gain)],
        )
        return linear_operator(space, twin, square_state)

    @pytest.mark.parametrize(
        ("label", "sigma"),
        [
            pytest.param("finite", 0.5, id="finite"),
            pytest.param("inf", jnp.inf, id="inf"),
            pytest.param("zero", 0.0, id="zero"),
            pytest.param("negative", -0.5, id="negative"),
        ],
    )
    def test_the_two_weight_formulas_agree_everywhere_except_nan(self, label, sigma):
        """Everything the two paths have in common, asserted first.

        Without these rows the NaN row below would be indistinguishable from
        "the two formulas are simply different", which is not the claim.
        """
        sigma_array = jnp.full((3,), sigma)
        direct = 1.0 / jnp.asarray(sigma_array) ** 2
        via_model = inverse_variance(HomoscedasticNoise(sigma_array), jnp.ones((3,)))
        assert jnp.array_equal(direct, via_model), (label, direct, via_model)

    def test_a_nan_sigma_is_loud_in_the_solve_and_silent_through_the_noise_model(self):
        """The reason the solves keep their own arithmetic.

        ``inverse_variance`` reads a non-finite sigma as "not observed" and
        returns weight 0. That is right for a flagged sample and wrong for a
        corrupt one, and it cannot tell them apart. In a conjugate solve a
        dropped sample moves the posterior WIDTH, not only the point, so the
        difference is not cosmetic: NaN in, NaN out is the behaviour that gets
        noticed.
        """
        sigma = jnp.full((3,), jnp.nan)
        direct = 1.0 / jnp.asarray(sigma) ** 2
        via_model = inverse_variance(HomoscedasticNoise(sigma), jnp.ones((3,)))

        assert jnp.all(jnp.isnan(direct)), direct
        assert jnp.all(via_model == 0.0), via_model

    def test_a_prediction_dependent_model_has_no_fixed_weights_to_give_a_solve(self):
        """The second obstacle: the solve has no prediction to evaluate it at.

        Pinned as a ratio rather than as two values so the test says what
        matters -- that the weights genuinely move with the prediction, by a
        lot -- rather than pinning a radiometer constant that is not the point.
        """
        noise = RadiometerNoise(channel_width=1e6, integration_time=1.0)
        assert noise.depends_on_prediction

        cold = inverse_variance(noise, jnp.full((2, 2), 100.0))
        warm = inverse_variance(noise, jnp.full((2, 2), 300.0))
        assert float(cold[0, 0] / warm[0, 0]) == pytest.approx(9.0, rel=1e-3)

    def test_a_constant_noise_model_is_refused_by_name_not_by_TypeError(
        self, square_block
    ):
        """The seam this whole assessment is about, asserted rather than implied.

        ``check_noise_std_axis`` accepts a noise model, because every other
        exit passes one. Before the branded refusal, a model reached
        ``jnp.asarray`` here and came back as ``TypeError: Value
        'HomoscedasticNoise(sigma=weak_f32[])' with dtype object is not a valid
        JAX array type`` -- which names the wrong layer and reads like a bug in
        the package rather than a wrong argument.
        """
        observed = jnp.full((N, N), SKY)
        with pytest.raises(ParameterSpaceError, match="takes a plain sigma array"):
            wiener_solve(
                square_block,
                observed,
                noise_std=HomoscedasticNoise(jnp.asarray(0.5)),
                prior_std=1.0,
            )

    def test_a_prediction_dependent_model_gets_the_longer_refusal(self, square_block):
        """Two branches, two messages, because they are different mistakes.

        A constant model is a packaging problem -- unwrap it and pass the
        array. A prediction-dependent one cannot be accepted at all, and the
        message has to say why rather than suggesting an unwrap that would
        silently freeze sigma at a tuple nobody chose.
        """
        observed = jnp.full((N, N), SKY)
        noise = RadiometerNoise(channel_width=1e6, integration_time=1.0)
        with pytest.raises(ParameterSpaceError, match="no prediction to evaluate it"):
            wiener_solve(square_block, observed, noise_std=noise, prior_std=1.0)

    def test_both_exits_refuse_a_noise_model(self, square_block):
        """The same asymmetry this file already caught once, not reintroduced."""
        observed = jnp.full((N, N), SKY)
        model = HomoscedasticNoise(jnp.asarray(0.5))
        with pytest.raises(ParameterSpaceError, match="wiener_solve"):
            wiener_solve(square_block, observed, noise_std=model, prior_std=1.0)
        with pytest.raises(ParameterSpaceError, match="gcr_sample"):
            gcr_sample(
                square_block, observed, noise_std=model, prior_std=1.0,
                key=jax.random.PRNGKey(0),
            )


class TestConditionEstimateRunsTheSameRules:
    """The third exit, which had the same keyword and neither check.

    ``condition_estimate`` takes ``noise_std=`` like ``wiener_solve`` and
    ``gcr_sample``, but never called ``_check_solve_arguments`` -- so a
    ``NoiseModel`` reached ``jnp.asarray`` and came back as a bare ``TypeError``
    naming a jax dtype, and an ambiguous 1-D sigma was silently given whichever
    reading NumPy's trailing-axis rule produced.

    It matters more here than the message quality suggests: this is the
    function a caller is told to consult in order to pick ``tol`` for the
    guarded solves. A kappa computed under a different reading of the same
    array answers a different question than the solve it was computed for, and
    nothing would have said so.
    """

    @pytest.fixture
    def square_block(self, square_state):
        twin = assemble(
            SkyOperator(amplitude=jnp.array(SKY)),
            GainOperator(gain=jnp.ones(N)),
        )
        space = ParameterSpace(
            latents=[Latent("gt", init=jnp.ones(N), linear=True)],
            bindings=[Bind("gt", into=lambda p: p["gain"].gain)],
        )
        return linear_operator(space, twin, square_state)

    def test_the_ambiguous_vector_is_refused_and_names_this_exit(self, square_block):
        with pytest.raises(StateValidationError, match="condition_estimate"):
            condition_estimate(square_block, noise_std=SIGMA_VECTOR, prior_std=1.0)

    def test_a_noise_model_is_refused_by_name(self, square_block):
        with pytest.raises(ParameterSpaceError, match="takes a plain sigma array"):
            condition_estimate(
                square_block,
                noise_std=HomoscedasticNoise(jnp.asarray(0.5)),
                prior_std=1.0,
            )

    @pytest.mark.parametrize(
        "reading",
        [
            pytest.param(SIGMA_VECTOR[:, None], id="explicit-column"),
            pytest.param(SIGMA_VECTOR[None, :], id="explicit-row"),
            pytest.param(0.5, id="scalar"),
        ],
    )
    def test_an_unambiguous_noise_std_still_gives_a_number(self, square_block, reading):
        kappa = condition_estimate(square_block, noise_std=reading, prior_std=1.0)
        assert jnp.isfinite(kappa) and float(kappa) >= 1.0

    def test_the_two_explicit_readings_give_different_condition_numbers(
        self, square_block
    ):
        """Which is why leaving the axis to broadcasting was not harmless here.

        If the two readings agreed, the missing check would be a cosmetic
        message problem. They do not.
        """
        per_time = condition_estimate(
            square_block, noise_std=SIGMA_VECTOR[:, None], prior_std=1.0
        )
        per_freq = condition_estimate(
            square_block, noise_std=SIGMA_VECTOR[None, :], prior_std=1.0
        )
        assert not jnp.allclose(per_time, per_freq, rtol=1e-3), (per_time, per_freq)

    def test_all_three_exits_now_refuse_the_same_input(self, square_block):
        """The symmetry assertion, which is what stops the third one drifting again.

        ``gcr_sample`` was missing this rule once and was fixed; then
        ``condition_estimate`` turned out to be missing it too. Asserting the
        set rather than each member is what makes a fourth exit's omission
        visible.
        """
        observed = jnp.full((N, N), SKY)
        exits = {
            "wiener_solve": lambda: wiener_solve(
                square_block, observed, noise_std=SIGMA_VECTOR, prior_std=1.0
            ),
            "gcr_sample": lambda: gcr_sample(
                square_block, observed, noise_std=SIGMA_VECTOR, prior_std=1.0,
                key=jax.random.PRNGKey(0),
            ),
            "condition_estimate": lambda: condition_estimate(
                square_block, noise_std=SIGMA_VECTOR, prior_std=1.0
            ),
        }
        for name, run in exits.items():
            with pytest.raises(StateValidationError, match=name):
                run()
