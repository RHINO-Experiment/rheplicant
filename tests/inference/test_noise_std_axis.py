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
from rheplicant.core.errors import StateValidationError
from rheplicant.inference import (
    Bind,
    FlaggedNoise,
    HomoscedasticNoise,
    Latent,
    ParameterSpace,
    RadiometerNoise,
    linear_operator,
    wiener_solve,
)
from rheplicant.inference.noise import check_noise_std_axis
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

    def test_the_two_explicit_readings_give_visibly_different_answers(
        self, square_state, gain_model
    ):
        """This is what the refusal is protecting: the numbers really do differ.

        Per-time weighting tracks sigma across the latent (0.00004 .. 0.00354);
        per-frequency weighting averages it away into a flat 0.00010.
        """
        space, twin = gain_model
        forward, values0 = space.forward_fn(twin, square_state)

        per_time = parameter_covariance(
            fisher_information(forward, values0, noise_std=SIGMA_VECTOR[:, None])
        ).sigma("gt")
        per_freq = parameter_covariance(
            fisher_information(forward, values0, noise_std=SIGMA_VECTOR[None, :])
        ).sigma("gt")

        assert per_time.shape == per_freq.shape == (N,)
        # Per-time: sigma_i = noise_i / (sqrt(n_freq) * SKY), a 100x spread.
        assert jnp.allclose(
            per_time, SIGMA_VECTOR / (jnp.sqrt(N) * SKY), rtol=1e-4
        )
        assert float(per_time.max() / per_time.min()) > 50.0
        # Per-frequency: every time sample sees the same set of channels, so
        # the error bar is flat.
        assert jnp.allclose(per_freq, per_freq[0], rtol=1e-5)
        assert not jnp.allclose(per_time, per_freq, rtol=0.1)

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
