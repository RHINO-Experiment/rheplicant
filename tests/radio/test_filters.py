"""Tests for the filter family: sidereal, Fourier-band, sky-space, apply-cal."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from replicant import Pipeline, SnapshotOperator, State
from replicant.core.errors import StateValidationError
from replicant.radio import (
    ApplyCalibrationOperator,
    FourierBandFilter,
    GainOperator,
    MatrixProjector,
    SiderealFilter,
    SkySpaceFilter,
)

N_DAYS, N_LST, N_FREQ = 4, 6, 3
N_TIME = N_DAYS * N_LST


@pytest.fixture
def sidereal_data(coords):
    """day-repeating template + per-day distinct offsets."""
    lst_template = jnp.outer(jnp.arange(1.0, N_LST + 1.0), jnp.ones(N_FREQ))
    repeating = jnp.tile(lst_template, (N_DAYS, 1))
    per_day = jnp.repeat(jnp.arange(N_DAYS, dtype=jnp.float32), N_LST)[:, None] * 10.0
    state = State(data=repeating + per_day)
    return state, repeating, per_day


class TestSiderealFilter:
    def test_extract_recovers_repeating_structure(self, sidereal_data):
        state, repeating, per_day = sidereal_data
        out = SiderealFilter(n_days=N_DAYS, mode="extract")(state)
        # the day-mean of the per-day offsets leaks into the template (as it must:
        # a constant-in-LST offset IS day-repeating), so the expected template is
        # the LST profile plus the mean day offset
        template = repeating[:N_LST] + per_day.reshape(N_DAYS, N_LST, 1).mean(axis=0)
        assert jnp.allclose(out.data, jnp.tile(template, (N_DAYS, 1)))

    def test_remove_is_complement(self, sidereal_data):
        state, *_ = sidereal_data
        extracted = SiderealFilter(n_days=N_DAYS, mode="extract")(state)
        removed = SiderealFilter(n_days=N_DAYS, mode="remove")(state)
        assert jnp.allclose(extracted.data + removed.data, state.data, atol=1e-5)

    def test_remove_kills_pure_repetition(self):
        template = jnp.outer(jnp.arange(N_LST, dtype=jnp.float32), jnp.ones(N_FREQ))
        state = State(data=jnp.tile(template, (N_DAYS, 1)))
        out = SiderealFilter(n_days=N_DAYS, mode="remove")(state)
        assert jnp.allclose(out.data, 0.0, atol=1e-5)

    def test_indivisible_raises(self):
        with pytest.raises(StateValidationError, match="divisible"):
            SiderealFilter(n_days=5, mode="remove")(State(data=jnp.ones((N_TIME, N_FREQ))))

    def test_config_validation(self):
        with pytest.raises(StateValidationError, match="n_days"):
            SiderealFilter(n_days=1)
        with pytest.raises(StateValidationError, match="mode"):
            SiderealFilter(n_days=2, mode="banish")


class TestFourierBandFilter:
    def test_removes_band_keeps_rest(self):
        t = jnp.arange(32.0)
        slow = jnp.cos(2 * jnp.pi * (1 / 32) * t)   # on-grid bin: |f| ~ 0.031, outside band
        fast = jnp.cos(2 * jnp.pi * (8 / 32) * t)   # on-grid bin: |f| = 0.25, inside band
        data = (slow + fast)[:, None] * jnp.ones((1, 2))
        out = FourierBandFilter(axis=0, low=0.2, high=0.3, mode="remove")(State(data=data))
        assert jnp.allclose(out.data[:, 0], slow, atol=1e-4)

    def test_extract_plus_remove_is_identity(self):
        data = jax.random.normal(jax.random.key(1), (16, 4))
        ex = FourierBandFilter(axis=1, low=0.1, high=0.4, mode="extract")(State(data=data))
        rm = FourierBandFilter(axis=1, low=0.1, high=0.4, mode="remove")(State(data=data))
        assert jnp.allclose(ex.data + rm.data, data, atol=1e-5)

    def test_band_validation(self):
        with pytest.raises(StateValidationError, match="Band"):
            FourierBandFilter(axis=0, low=0.4, high=0.2)
        with pytest.raises(StateValidationError, match="axis"):
            FourierBandFilter(axis=2, low=0.1, high=0.2)


class TestSkySpaceFilter:
    N_PIX = 5

    @pytest.fixture
    def projector(self):
        key = jax.random.key(3)
        return MatrixProjector(matrix=jax.random.normal(key, (24, self.N_PIX)))

    def test_extract_recovers_sky_locked_data(self, projector):
        """Data that IS a projected sky must survive extraction (almost) intact."""
        sky_true = jnp.outer(jnp.ones(N_FREQ), jnp.arange(1.0, self.N_PIX + 1.0))
        data = projector.forward(sky_true, None)
        filt = SkySpaceFilter(
            projector=projector, regularization=jnp.array(1e-6),
            cg_maxiter=200, mode="extract",
        )
        out = filt(State(data=data))
        assert jnp.allclose(out.data, data, rtol=1e-3, atol=1e-3)

    def test_remove_leaves_small_residual_on_sky_data(self, projector):
        sky_true = jnp.ones((N_FREQ, self.N_PIX))
        data = projector.forward(sky_true, None)
        out = SkySpaceFilter(
            projector=projector, regularization=jnp.array(1e-6),
            cg_maxiter=200, mode="remove",
        )(State(data=data))
        assert jnp.max(jnp.abs(out.data)) < 1e-2 * jnp.max(jnp.abs(data))

    def test_flag_weighting_path(self, projector):
        data = projector.forward(jnp.ones((N_FREQ, self.N_PIX)), None)
        flags = jnp.zeros(data.shape, dtype=bool).at[0, 0].set(True)
        out = SkySpaceFilter(
            projector=projector, regularization=jnp.array(1e-4), mode="extract"
        )(State(data=data, aux={"flags": flags}))
        assert jnp.all(jnp.isfinite(out.data))

    def test_differentiable_through_cg(self, projector):
        data = projector.forward(jnp.ones((N_FREQ, self.N_PIX)), None)

        def loss(filt):
            return jnp.sum(filt(State(data=data)).data ** 2)

        filt = SkySpaceFilter(
            projector=projector, regularization=jnp.array(1e-3), mode="remove"
        )
        grads = eqx.filter_grad(loss)(filt)
        assert jnp.isfinite(grads.regularization)

    def test_projector_type_validation(self):
        with pytest.raises(StateValidationError, match="projector"):
            SkySpaceFilter(projector=jnp.ones(3), regularization=jnp.array(1e-3))


class TestApplyCalibration:
    def test_inverts_gain_operator(self, template_state):
        data_state = template_state.with_data(jnp.full((8, 4), 10.0))
        gain = jnp.linspace(0.9, 1.2, 8)
        corrupted = GainOperator(gain=gain)(data_state)
        recovered = ApplyCalibrationOperator(gain=gain)(corrupted)
        assert jnp.allclose(recovered.data, data_state.data, rtol=1e-6)

    def test_analysis_pipeline_shape(self, template_state):
        """The documented analysis pattern: snapshot -> apply-cal -> filter."""
        data_state = template_state.with_data(jnp.ones((8, 4)))
        analysis = Pipeline(
            SnapshotOperator(name="raw"),
            ApplyCalibrationOperator(gain=jnp.array(2.0)),
            SiderealFilter(n_days=2, mode="remove"),
            names=("snap", "cal", "sidereal"),
        )
        out = analysis(data_state)
        assert jnp.array_equal(out.aux["snapshot/raw"], data_state.data)
        assert jnp.allclose(out.data, 0.0, atol=1e-6)  # uniform data is all-repeating
