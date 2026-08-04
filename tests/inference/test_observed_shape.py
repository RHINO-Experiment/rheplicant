"""Every exit that consumes ``observed`` must refuse one shaped wrong.

The scenario is one honest slicing mistake, in five shapes. ``observed[0]``
drops a rank — ``(8,)`` where ``(24, 8)`` was meant. ``observed[:, :1]`` and
``observed[:1]`` keep it — ``(24, 1)`` and ``(1, 8)``, one channel or one time
sample standing in for the whole record. ``observed[:, None, :]`` gains a rank
instead of losing one — ``(24, 1, 8)``, a spurious axis nothing asked for.
``observed.mean()`` collapses the whole record to a single number — ``()``.
All five broadcast. Nothing about any of them is loud: the subtraction
succeeds, the loss is finite and small, and the calibrator reports convergence
while every recovered gain is wrong. The loss history is the only evidence the
user has, and it says the fit worked.

The two rank-preserving slices are the more insidious of the first three, and
they are here because a guard that compared only ``jnp.ndim`` would let them
through. Measured, with that relaxation in place: both calibrators converged
on ``(24, 1)`` and ``(1, 8)``, the NumPyro site conditioned without complaint,
and ``wiener_solve`` returned a perfectly well-shaped ``(24,)`` gain estimate
from a single column of data. A finite, correctly-shaped, wrong answer is the
failure these guards exist to prevent, so one scenario that happens to differ
in rank cannot be the whole of the evidence.

The rank-gained and scalar shapes close the other side of that same gap: a
relaxation that tolerates a singleton or an added axis — "squeeze before
comparing", or "``jnp.size(observed) == 1`` broadcasts everywhere, that's not
a mistake" — passes all three narrowing shapes above yet still admits
``observed.mean()``, ``(1,)``, and ``(24, 1, 8)`` at every exit. Measured, with
the size-1 relaxation in place: ``AdamCalibrator`` converges on the mean of the
record with ``loss[-1]=2.274e-11`` and every recovered gain still wrong by up
to ``0.23``.

:func:`~rheplicant.inference.linear.wiener_solve` already refused this, and its
message already owns the explanation. These tests pin the same refusal, with
the same wording, at the exits that used to broadcast: both calibrators and the
NumPyro observation site.

The fourth exit named by the audit, ``fisher_information``, is covered by
:class:`TestTheFisherExit` — see the class docstring for why it is a different
guard.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, Environment, State
from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference import (
    AdamCalibrator,
    GradientCalibrator,
    ParameterSpace,
    fisher_information,
    linear_operator,
    to_numpyro_model,
    wiener_solve,
)
from rheplicant.radio import GainOperator, SkyOperator, assemble

numpyro = pytest.importorskip("numpyro", reason="numpyro not installed")
import numpyro.distributions as dist  # noqa: E402
from numpyro.handlers import seed  # noqa: E402

N_TIME, N_FREQ = 24, 8
SKY = 100.0
SIGMA = 0.5


@pytest.fixture
def state():
    """A (24, 8) observation context — deliberately NOT square, so a mis-slice
    of either axis is a shape the other axis cannot silently absorb."""
    return State(
        coords=Coordinates(
            time=jnp.linspace(0.0, 23.0, N_TIME),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        env=Environment(temperature=jnp.array(280.0)),
        key=jax.random.key(0),
        meta={"telescope": "RHINO", "obs_id": "b4"},
    )


@pytest.fixture
def twin():
    """Per-time gain, so the latent is (24,) and the prediction is (24, 8)."""
    return assemble(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.ones(N_TIME)),
    )


@pytest.fixture
def space():
    return ParameterSpace.direct(
        "gain",
        init=jnp.ones(N_TIME),
        into=lambda p: p["gain"].gain,
        prior=dist.Normal(jnp.ones(N_TIME), 0.3),
    )


@pytest.fixture
def observed(state):
    truth = assemble(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=1.0 + 0.02 * jnp.arange(N_TIME)),
    )(state).data
    return truth + SIGMA * jax.random.normal(jax.random.key(99), truth.shape)


MIS_SLICES = [
    pytest.param(
        (lambda d: d[0], (N_FREQ,)),
        id="one_time_sample_rank_dropped",
    ),
    pytest.param(
        (lambda d: d[:, :1], (N_TIME, 1)),
        id="one_channel_rank_kept",
    ),
    pytest.param(
        (lambda d: d[:1], (1, N_FREQ)),
        id="one_time_sample_rank_kept",
    ),
    pytest.param(
        (lambda d: d[:, None, :], (N_TIME, 1, N_FREQ)),
        id="spurious_axis_rank_gained",
    ),
    pytest.param(
        (lambda d: d.mean(), ()),
        id="scalar_whole_record_collapsed",
    ),
]


@pytest.fixture(params=MIS_SLICES)
def mis_sliced(request, observed):
    """One slice of the record where the whole record was meant.

    Parametrized rather than fixed, so that every exit is asked about a
    rank-preserving mistake as well as a rank-dropping one — see the module
    docstring for why the distinction decides whether these tests have any
    force.
    """
    take, expected_shape = request.param
    sliced = take(observed)
    assert jnp.shape(sliced) == expected_shape, (
        f"the fixture does not slice what it claims: got {jnp.shape(sliced)}, "
        f"declared {expected_shape}"
    )
    return sliced


@pytest.fixture
def forward_and_start(space, twin, state):
    return space.forward_fn(twin, state)


def assert_names_which_shape_is_which(message: str, observed) -> None:
    """The message must say which shape belongs to which side.

    Asserting only that both shapes appear *somewhere* passes just as happily
    when they are swapped, and the swapped message — ``observed has shape
    (24, 8) but this forward model predicts (8,)`` — is worse than none: it
    sends the user to debug a forward model that is correct, and away from the
    slice that is not. So the fragments are pinned in order. They are pinned
    around the predictor name, not through it, because the three exits pass
    three different ones ("this block", "this model", "this forward model").
    """
    assert f"observed has shape {jnp.shape(observed)}" in message, message
    assert f"predicts {(N_TIME, N_FREQ)}" in message, message
    assert "Broadcasting these would solve a different problem" in message, message


class TestTheCalibratorExits:
    """``fit`` used to broadcast, converge, and report a small loss."""

    def test_the_gradient_calibrator_refuses_a_mis_shaped_observed(
        self, forward_and_start, mis_sliced
    ):
        forward, start = forward_and_start
        with pytest.raises(ParameterSpaceError) as excinfo:
            GradientCalibrator(learning_rate=1e-6, n_steps=10).fit(
                forward, start, mis_sliced
            )
        assert_names_which_shape_is_which(str(excinfo.value), mis_sliced)

    def test_the_adam_calibrator_refuses_a_mis_shaped_observed(
        self, forward_and_start, mis_sliced
    ):
        forward, start = forward_and_start
        with pytest.raises(ParameterSpaceError) as excinfo:
            AdamCalibrator(learning_rate=0.05, n_steps=10).fit(
                forward, start, mis_sliced
            )
        assert_names_which_shape_is_which(str(excinfo.value), mis_sliced)

    def test_the_refusal_precedes_any_optimization(
        self, forward_and_start, mis_sliced
    ):
        """It raises at entry, not after n_steps of a wrong fit: a 10^6-step
        calibrator must fail as fast as a 10-step one."""
        forward, start = forward_and_start
        with pytest.raises(ParameterSpaceError):
            AdamCalibrator(learning_rate=0.05, n_steps=1_000_000).fit(
                forward, start, mis_sliced
            )


class TestTheNumpyroExit:
    """The observation site used to broadcast, and NUTS converged on nonsense."""

    def test_the_model_refuses_a_mis_shaped_observed(
        self, twin, state, space, mis_sliced
    ):
        model = to_numpyro_model(twin, state, space, noise_std=SIGMA)
        with pytest.raises(ParameterSpaceError) as excinfo:
            seed(model, jax.random.key(0))(observed=mis_sliced)
        assert_names_which_shape_is_which(str(excinfo.value), mis_sliced)

    def test_nuts_refuses_a_mis_shaped_observed(self, twin, state, space, mis_sliced):
        model = to_numpyro_model(twin, state, space, noise_std=SIGMA)
        mcmc = numpyro.infer.MCMC(
            numpyro.infer.NUTS(model),
            num_warmup=2,
            num_samples=2,
            progress_bar=False,
        )
        with pytest.raises(ParameterSpaceError) as excinfo:
            mcmc.run(jax.random.key(0), observed=mis_sliced)
        assert_names_which_shape_is_which(str(excinfo.value), mis_sliced)

    def test_an_unobserved_model_still_runs(self, twin, state, space):
        """``observed=None`` is the prior-predictive call, not a mismatch."""
        model = to_numpyro_model(twin, state, space, noise_std=SIGMA)
        seed(model, jax.random.key(0))()  # must not raise

    def test_a_correctly_shaped_observed_still_runs(
        self, twin, state, space, observed
    ):
        model = to_numpyro_model(twin, state, space, noise_std=SIGMA)
        seed(model, jax.random.key(0))(observed=observed)  # must not raise


class TestTheLinearExit:
    """The guard that was already right — pinned so the shared seam keeps it."""

    def test_wiener_solve_refuses_a_mis_shaped_observed(
        self, twin, state, mis_sliced
    ):
        linear_space = ParameterSpace.direct(
            "gain",
            init=jnp.ones(N_TIME),
            into=lambda p: p["gain"].gain,
            prior=dist.Normal(jnp.ones(N_TIME), 0.3),
            linear=True,
        )
        block = linear_operator(linear_space, twin, state, "gain")
        with pytest.raises(ParameterSpaceError) as excinfo:
            wiener_solve(block, mis_sliced, noise_std=SIGMA, prior_std=1.0)
        assert_names_which_shape_is_which(str(excinfo.value), mis_sliced)


class TestTheFisherExit:
    """``fisher_information`` has no ``observed`` to check — it never sees data.

    The audit listed it as a fourth exit that "does not check that ``observed``
    matches the prediction". Its signature is
    ``fisher_information(forward, params, noise_std, flags=None)``: a Fisher
    forecast is a function of the model and the noise alone, so there is no
    ``observed`` argument to mis-shape and no guard to add. The first test below
    pins that, so a future ``observed=`` parameter cannot be added without
    revisiting this decision.

    The data-shaped argument it *does* take is ``flags``, and the same slicing
    mistake there is refused — by the guard already in
    :class:`~rheplicant.inference.noise.FlaggedNoise`, which names both shapes.
    """

    def test_fisher_information_takes_no_observed(self):
        import inspect

        parameters = inspect.signature(fisher_information).parameters
        assert "observed" not in parameters, (
            "fisher_information gained an `observed` argument; it must now be "
            "checked against the prediction like every other exit."
        )

    @pytest.mark.parametrize(
        "flag_shape",
        [(N_FREQ,), (N_TIME, 1), (1, N_FREQ)],
        ids=[
            "one_time_sample_rank_dropped",
            "one_channel_rank_kept",
            "one_time_sample_rank_kept",
        ],
    )
    def test_it_refuses_a_mis_shaped_flags(self, forward_and_start, flag_shape):
        """The same slicing mistake, on the only data-shaped argument this exit
        has — and swept over the same three shapes, because a ``(24, 1)`` flags
        array would otherwise broadcast and silently leave seven channels of
        eight unflagged while reporting a Fisher matrix of the right size."""
        forward, start = forward_and_start
        flags = jnp.zeros(flag_shape, bool)
        with pytest.raises(StateValidationError) as excinfo:
            fisher_information(forward, start, noise_std=SIGMA, flags=flags)
        message = str(excinfo.value)
        assert f"flags shape {flag_shape}" in message, message
        assert f"prediction shape {(N_TIME, N_FREQ)}" in message, message

    def test_a_correctly_shaped_flags_still_runs(self, forward_and_start):
        forward, start = forward_and_start
        flags = jnp.zeros((N_TIME, N_FREQ), bool).at[0].set(True)
        matrix = fisher_information(forward, start, noise_std=SIGMA, flags=flags)
        assert matrix.matrix.shape == (N_TIME, N_TIME)


class TestBroadcastingThatMustKeepWorking:
    """A scalar loss target and a per-channel sigma are not mistakes."""

    def test_a_correctly_shaped_observed_still_fits(
        self, forward_and_start, observed
    ):
        forward, start = forward_and_start
        fitted, losses = AdamCalibrator(learning_rate=0.05, n_steps=200).fit(
            forward, start, observed
        )
        assert jnp.all(jnp.isfinite(losses))
        assert fitted["gain"].shape == (N_TIME,)

    def test_a_per_channel_sigma_still_broadcasts(self, forward_and_start):
        """``noise_std`` is documented as broadcastable to the prediction and
        several callers rely on it; the observed guard must not reach it."""
        forward, start = forward_and_start
        sigma = jnp.linspace(0.2, 0.6, N_FREQ)
        matrix = fisher_information(forward, start, noise_std=sigma)
        assert matrix.matrix.shape == (N_TIME, N_TIME)
