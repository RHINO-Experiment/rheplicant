"""The bandpass/gain scalar degeneracy, measured, and the convention that ends it.

``b -> c*b, g -> g/c`` leaves every predicted sample unchanged, so a free
bandpass and a free gain are separately unidentifiable up to one scalar. This
is exact, it is not mentioned anywhere in the design record, and it is now
something the package measures rather than asserts:
:func:`~rheplicant.inference.identifiability.identifiability` names the null
direction by latent.

Everything is built in FLOAT64. ``identifiability()`` forces x64 for its own
arithmetic, but that does not help when the model's constants were rounded to
float32 before it was called — the error is already baked in, and a rank
verdict then measures the rounding. So the fixtures are constructed inside the
x64 context, not at import.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.identifiability import identifiability
from rheplicant.radio import (
    GainOperator,
    ReceiverOperator,
    unit_mean_bandpass,
    unit_mean_free,
)

# Non-square on purpose: 5 channels against 6 samples, so a b/g confusion is a
# shape error rather than a silently plausible number.
N_TIME, N_FREQ = 6, 5


@pytest.fixture(autouse=True)
def _x64():
    """Build the fixtures in double precision, and put the flag back."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", was)


class KnownSky(AbstractOperator):
    """A fixed, KNOWN antenna temperature — test double, not physics.

    Held known so that the only degeneracy in the model is the one under
    examination. A free-per-cell ``T_ant`` has much larger problems of its own
    (see ``tests/inference/test_identifiability.py``) and they would mask this.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    t_ant: jax.Array

    def __call__(self, state):
        return state.with_data(self.t_ant)


def _fixtures():
    time = jnp.arange(N_TIME, dtype=float)
    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    # Varies in BOTH axes and is not separable, so nothing about the sky can
    # stand in for a bandpass or for a gain.
    t_ant = 200.0 + 7.0 * time[:, None] + 3.0 * jnp.arange(N_FREQ)[None, :] ** 1.5
    bandpass = jnp.array([0.80, 0.95, 1.10, 1.05, 0.90])
    gain = 1.5 + 0.05 * time
    pipeline = Pipeline(
        KnownSky(t_ant=t_ant),
        ReceiverOperator(bandpass=bandpass),
        GainOperator(gain=gain),
        names=("t_ant", "bandpass", "gain"),
    )
    state = State(
        coords=Coordinates(time=time, freq=freq),
        meta={"telescope": "RHINO", "obs_id": "bg-degeneracy"},
    )
    return pipeline, state, bandpass, gain


def _free_space(bandpass, gain) -> ParameterSpace:
    """Both free, in their own units — the degenerate parameterization."""
    return ParameterSpace(
        latents=[Latent("bandpass", init=bandpass), Latent("gain", init=gain)],
        bindings=[
            Bind("bandpass", into=lambda p: p["bandpass"].bandpass),
            Bind("gain", into=lambda p: p["gain"].gain),
        ],
    )


def _convention_space(bandpass, gain) -> ParameterSpace:
    """The package convention: the bandpass is SHAPE, the gain carries level."""
    return ParameterSpace(
        latents=[
            Latent("bandpass_shape", init=unit_mean_free(bandpass)),
            Latent("gain", init=gain),
        ],
        bindings=[
            Bind(
                "bandpass_shape",
                into=lambda p: p["bandpass"].bandpass,
                fn=unit_mean_bandpass,
            ),
            Bind("gain", into=lambda p: p["gain"].gain),
        ],
    )


def _renormalising_space(bandpass, gain) -> ParameterSpace:
    """The trap: normalise in the binding, keep all n_freq parameters."""
    return ParameterSpace(
        latents=[Latent("bandpass", init=bandpass), Latent("gain", init=gain)],
        bindings=[
            Bind(
                "bandpass",
                into=lambda p: p["bandpass"].bandpass,
                fn=lambda b: b / jnp.mean(b),
            ),
            Bind("gain", into=lambda p: p["gain"].gain),
        ],
    )


class TestTheDegeneracyIsReal:
    def test_a_free_bandpass_and_a_free_gain_have_one_null_direction(self):
        pipeline, state, bandpass, gain = _fixtures()
        report = identifiability(_free_space(bandpass, gain), pipeline, state)
        assert (report.n_par, report.rank, report.nullity) == (11, 10, 1)
        # exact, not weakly constrained: 16 decades below the largest
        largest_null = report.singular_values[report.rank] / report.singular_values[0]
        assert largest_null < 1e-13
        assert report.weakest_identified > 1e-2

    def test_the_fixture_really_is_double_precision(self):
        """The discipline this file exists to keep. A float32 model reports its
        own rounding as a rank verdict."""
        pipeline, state, bandpass, gain = _fixtures()
        assert bandpass.dtype == jnp.float64 and gain.dtype == jnp.float64
        report = identifiability(_free_space(bandpass, gain), pipeline, state)
        assert report.jacobian.dtype == np.float64

    def test_the_null_direction_is_the_b_over_g_trade_by_name(self):
        """The evidence, in latent coordinates rather than as an assertion.

        A null direction proportional to ``+b`` in the bandpass and to ``-g``
        in the gain, with the SAME constant of proportionality, is exactly the
        statement ``b -> c*b, g -> g/c``.
        """
        pipeline, state, bandpass, gain = _fixtures()
        report = identifiability(_free_space(bandpass, gain), pipeline, state)
        direction = report.direction(0)

        ratio_b = np.asarray(direction["bandpass"]) / np.asarray(bandpass)
        ratio_g = np.asarray(direction["gain"]) / np.asarray(gain)
        # constant within each latent ...
        assert np.allclose(ratio_b, ratio_b[0], rtol=1e-9)
        assert np.allclose(ratio_g, ratio_g[0], rtol=1e-9)
        # ... equal and opposite across them (the SVD fixes the sign, not us)
        assert np.isclose(ratio_b[0], -ratio_g[0], rtol=1e-9)
        assert abs(ratio_b[0]) > 0.1  # both halves genuinely participate

    def test_both_latents_carry_half_of_it(self):
        pipeline, state, bandpass, gain = _fixtures()
        report = identifiability(_free_space(bandpass, gain), pipeline, state)
        share = report.participation(0)
        assert np.isclose(share["bandpass"], 0.5, atol=1e-9)
        assert np.isclose(share["gain"], 0.5, atol=1e-9)


class TestTheConventionRemovesIt:
    def test_nullity_goes_from_one_to_zero(self):
        """The measurement the convention is chosen on, stated as a comparison
        so that an implementation returning a constant could not pass it."""
        pipeline, state, bandpass, gain = _fixtures()
        before = identifiability(_free_space(bandpass, gain), pipeline, state)
        after = identifiability(_convention_space(bandpass, gain), pipeline, state)
        assert (before.nullity, after.nullity) == (1, 0)
        assert after.n_par == before.n_par - 1  # one parameter, not a reweighting
        assert after.rank == after.n_par
        assert after.weakest_identified > 1e-2

    def test_the_convention_does_not_change_the_prediction(self):
        """It re-coordinates the bandpass; it must not move the model."""
        pipeline, state, bandpass, gain = _fixtures()
        space = _convention_space(bandpass, gain)
        forward, values = space.forward_fn(pipeline, state)
        predicted = forward(values)
        expected = pipeline(state).data / jnp.mean(bandpass)
        assert jnp.allclose(predicted, expected, rtol=1e-12)

    def test_normalising_in_the_binding_alone_does_NOT_remove_it(self):
        """The trap, measured. It looks like the same convention and is not:
        the prediction becomes blind to the RAW latent's scale, so the null
        direction survives — it simply stops being a b/g trade and becomes the
        bandpass latent's own scale ray, which ``participation`` reports."""
        pipeline, state, bandpass, gain = _fixtures()
        report = identifiability(_renormalising_space(bandpass, gain), pipeline, state)
        assert (report.n_par, report.nullity) == (11, 1)
        share = report.participation(0)
        assert np.isclose(share["bandpass"], 1.0, atol=1e-9)
        assert np.isclose(share["gain"], 0.0, atol=1e-9)


class TestTheConventionHelpers:
    def test_the_expanded_bandpass_has_mean_exactly_one(self):
        free = jnp.array([0.8, 0.95, 1.1, 1.05])
        assert jnp.isclose(jnp.mean(unit_mean_bandpass(free)), 1.0, rtol=0, atol=1e-15)

    def test_it_adds_exactly_one_channel(self):
        free = jnp.array([0.8, 0.95, 1.1, 1.05])
        assert unit_mean_bandpass(free).shape == (free.size + 1,)

    def test_the_free_channels_pass_through_untouched(self):
        """Only the last channel is determined; the rest are the parameters."""
        free = jnp.array([0.8, 0.95, 1.1, 1.05])
        assert jnp.allclose(unit_mean_bandpass(free)[:-1], free)

    def test_round_trip_recovers_the_normalised_bandpass(self):
        bandpass = jnp.array([0.80, 0.95, 1.10, 1.05, 0.90])
        recovered = unit_mean_bandpass(unit_mean_free(bandpass))
        assert jnp.allclose(recovered, bandpass / jnp.mean(bandpass), rtol=1e-12)

    def test_unit_mean_free_drops_one_value(self):
        bandpass = jnp.array([0.80, 0.95, 1.10, 1.05, 0.90])
        assert unit_mean_free(bandpass).shape == (bandpass.size - 1,)

    def test_a_two_dimensional_free_vector_is_refused(self):
        """It would concatenate along the wrong axis and return a bandpass of
        the wrong length, which ReceiverOperator would then blame on the data."""
        with pytest.raises(StateValidationError, match="ndim=2"):
            unit_mean_bandpass(jnp.ones((1, 4)))

    def test_a_scalar_free_vector_is_refused(self):
        with pytest.raises(StateValidationError, match="ndim=0"):
            unit_mean_bandpass(jnp.array(1.0))

    def test_unit_mean_free_refuses_a_two_dimensional_bandpass(self):
        with pytest.raises(StateValidationError, match="ndim=2"):
            unit_mean_free(jnp.ones((2, 5)))

    def test_the_expansion_is_differentiable(self):
        """It is a binding ``fn``: everything downstream of it is a gradient."""
        free = jnp.array([0.8, 0.95, 1.1, 1.05])
        jac = jax.jacfwd(unit_mean_bandpass)(free)
        assert jac.shape == (5, 4)
        assert jnp.allclose(jac[-1], -1.0)  # the dependent channel absorbs the rest
