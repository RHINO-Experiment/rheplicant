"""AntennaLossOperator: the antenna's own ohmic loss, and where it belongs.

Two things are being tested. The arithmetic is one line, so most of the value is
in the *placement*: a loss applied to the wrong set of contributions, or on the
wrong side of the calibration switch, is a finite and correctly-shaped bias in
every noise-wave solution that follows.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import StateValidationError
from rheplicant.radio import (
    AntennaLossOperator,
    AtmosphericEmissionOperator,
    CalLoadOperator,
    GroundPickupOperator,
    SkyOperator,
    assemble,
)

N_TIME, N_FREQ = 6, 4
T_PHYS = 293.0


@pytest.fixture
def coords():
    return Coordinates(
        time=jnp.arange(float(N_TIME)),
        freq=jnp.linspace(60e6, 85e6, N_FREQ),
    )


def state_with(data, coords, switch=None):
    if switch is not None:
        coords = coords.replace(extra={"receiver_input": jnp.asarray(switch)})
    return State(data=data, coords=coords)


class TestThermodynamics:
    def test_a_lossless_antenna_is_the_identity(self, coords):
        data = jnp.full((N_TIME, N_FREQ), 1234.0)
        op = AntennaLossOperator(efficiency=jnp.array(1.0),
                                 t_physical=jnp.array(T_PHYS))
        assert jnp.array_equal(op(state_with(data, coords)).data, data)

    def test_a_totally_lossy_antenna_delivers_only_its_own_temperature(self, coords):
        """eta = 0: nothing collected survives, and what comes out is the
        structure's own thermal emission -- not zero."""
        data = jnp.full((N_TIME, N_FREQ), 1234.0)
        op = AntennaLossOperator(efficiency=jnp.array(0.0),
                                 t_physical=jnp.array(T_PHYS))
        assert jnp.allclose(op(state_with(data, coords)).data, T_PHYS)

    @pytest.mark.parametrize("efficiency", [0.0, 0.25, 0.5, 0.9, 1.0])
    def test_an_isothermal_enclosure_is_a_fixed_point(self, coords, efficiency):
        """The invariant that makes this Kirchhoff and not an arbitrary blend:
        an antenna at temperature T looking at a sky at the SAME temperature T
        must deliver T, whatever its efficiency. Any other coefficient pairing
        -- (eta, eta), (eta, 1), (1, 1 - eta) -- breaks this."""
        data = jnp.full((N_TIME, N_FREQ), T_PHYS)
        op = AntennaLossOperator(efficiency=jnp.array(efficiency),
                                 t_physical=jnp.array(T_PHYS))
        assert jnp.allclose(op(state_with(data, coords)).data, T_PHYS, atol=1e-4)

    def test_the_added_emission_is_not_a_pure_attenuation(self, coords):
        """Distinguishes ohmic loss from the noise-wave stage's mismatch loss
        c_s, which attenuates and adds nothing of its own. Getting these
        confused would fit an efficiency as a reflection and lose the
        (1 - eta) T_phys term entirely."""
        data = jnp.zeros((N_TIME, N_FREQ))  # a cold sky
        op = AntennaLossOperator(efficiency=jnp.array(0.8),
                                 t_physical=jnp.array(T_PHYS))
        out = op(state_with(data, coords)).data
        assert jnp.allclose(out, 0.2 * T_PHYS, atol=1e-4)
        assert not jnp.allclose(out, 0.0)

    def test_a_per_channel_efficiency_acts_per_channel(self, coords):
        eta = jnp.linspace(0.7, 1.0, N_FREQ)
        data = jnp.full((N_TIME, N_FREQ), 500.0)
        op = AntennaLossOperator(efficiency=eta, t_physical=jnp.array(T_PHYS))
        expected = eta * 500.0 + (1.0 - eta) * T_PHYS
        assert jnp.allclose(op(state_with(data, coords)).data, expected[None, :])


class TestRejections:
    def test_an_efficiency_with_the_wrong_channel_count_is_refused(self, coords):
        op = AntennaLossOperator(efficiency=jnp.ones(N_FREQ + 1),
                                 t_physical=jnp.array(T_PHYS))
        with pytest.raises(StateValidationError, match="channels"):
            op(state_with(jnp.zeros((N_TIME, N_FREQ)), coords))

    def test_a_physical_temperature_with_the_wrong_channel_count_is_refused(
        self, coords
    ):
        op = AntennaLossOperator(efficiency=jnp.array(0.9),
                                 t_physical=jnp.ones(N_FREQ + 2))
        with pytest.raises(StateValidationError, match="channels"):
            op(state_with(jnp.zeros((N_TIME, N_FREQ)), coords))

    def test_disagreeing_spectra_are_refused_at_construction(self):
        with pytest.raises(StateValidationError, match="channels"):
            AntennaLossOperator(efficiency=jnp.ones(4), t_physical=jnp.ones(5))

    def test_a_two_dimensional_efficiency_is_refused(self):
        with pytest.raises(StateValidationError, match="ndim"):
            AntennaLossOperator(efficiency=jnp.ones((2, 2)),
                                t_physical=jnp.array(T_PHYS))

    def test_one_dimensional_data_is_refused(self, coords):
        op = AntennaLossOperator(efficiency=jnp.array(0.9),
                                 t_physical=jnp.array(T_PHYS))
        with pytest.raises(StateValidationError, match="n_time, n_freq"):
            op(state_with(jnp.zeros(N_TIME), coords))


class TestPlacement:
    """Where the node sits IS the physics; these pin it."""

    def test_it_lands_between_the_antenna_sum_and_the_switch(self, coords):
        twin = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            AntennaLossOperator(efficiency=jnp.array(0.9),
                                t_physical=jnp.array(T_PHYS)),
        )
        assert "antenna_loss" in twin.lit
        assert list(twin.lit).index("antenna_loss") == len(twin.lit) - 1

    def test_the_calibration_loads_do_not_see_it(self, coords):
        """The loads connect at the receiver input, downstream of the antenna.
        A loss applied after the switch would attenuate them too and bias every
        noise-wave solution built on them."""
        switch = jnp.arange(N_TIME) % 2  # 0 antenna, 1 load
        twin = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            AntennaLossOperator(efficiency=jnp.array(0.5),
                                t_physical=jnp.array(T_PHYS)),
            CalLoadOperator(t_load=jnp.array(400.0)),
        )
        out = twin(State(coords=coords.replace(
            extra={"receiver_input": switch}))).data
        assert jnp.allclose(out[switch == 1], 400.0, atol=1e-3), (
            "the load must arrive unattenuated"
        )
        assert jnp.allclose(out[switch == 0], 0.5 * 100.0 + 0.5 * T_PHYS, atol=1e-3)

    def test_it_attenuates_every_antenna_temperature_contribution(self, coords):
        """Unlike atmospheric opacity (D13), ohmic loss acts AFTER collection,
        so it applies to the whole t_ant_sum -- sky, ground spill and
        atmospheric emission alike."""
        pieces = dict(
            sky=SkyOperator(amplitude=jnp.array(100.0)),
            ground=GroundPickupOperator(coupling=jnp.array(0.2),
                                        t_ground=jnp.array(290.0)),
            atmosphere=AtmosphericEmissionOperator(t_atm=jnp.array(20.0)),
        )
        collected = assemble(*pieces.values())(State(coords=coords)).data
        lossy = assemble(
            *pieces.values(),
            AntennaLossOperator(efficiency=jnp.array(0.8),
                                t_physical=jnp.array(T_PHYS)),
        )(State(coords=coords)).data
        assert jnp.allclose(lossy, 0.8 * collected + 0.2 * T_PHYS, atol=1e-3)
        assert float(collected.mean()) > 150.0, "all three branches must be live"


class TestTransforms:
    def test_it_jits_and_differentiates_in_both_leaves(self, coords):
        data = jnp.full((N_TIME, N_FREQ), 500.0)

        def loss(efficiency, t_physical):
            op = AntennaLossOperator(efficiency=efficiency, t_physical=t_physical)
            return jnp.sum(op(state_with(data, coords)).data ** 2)

        d_eta, d_t = jax.jit(jax.grad(loss, argnums=(0, 1)))(
            jnp.array(0.85), jnp.array(T_PHYS)
        )
        assert jnp.isfinite(d_eta) and d_eta != 0.0
        assert jnp.isfinite(d_t) and d_t != 0.0
