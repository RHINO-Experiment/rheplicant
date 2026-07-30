"""NoiseWaveOperator: draft Eq. 1 as a rheplicant operator."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("rhino_cal_jax", reason="rhino-cal-jax not installed")

from rheplicant.core.coordinates import Coordinates  # noqa: E402
from rheplicant.core.errors import StateValidationError  # noqa: E402
from rheplicant.core.state import State  # noqa: E402
from rheplicant.radio import NoiseWaveOperator  # noqa: E402

N_TIME, N_FREQ = 12, 4

# Three physically distinct loads, in the order the switch indexes them: a
# structured antenna reflection, a near-matched ambient load, and a short-like
# reflection of opposite sign. Three is the minimum that makes the per-channel
# noise-wave system full rank -- see TestIdentifiability.
G_SRC_RE = np.array([[0.30, 0.28, 0.26, 0.24],
                     [0.02, 0.01, 0.00, -0.01],
                     [-0.60, -0.62, -0.64, -0.66]])
G_SRC_IM = np.array([[0.10, 0.05, 0.00, -0.05],
                     [0.00, 0.03, 0.02, 0.01],
                     [0.15, -0.10, 0.05, 0.20]])
N_SOURCE = G_SRC_RE.shape[0]
G_REC_RE = np.full(N_FREQ, 0.08)
G_REC_IM = np.full(N_FREQ, -0.03)

# rheplicant's suite runs float32; jax_enable_x64 is process-global and cannot
# be flipped per-module, so the tolerance is read from the active precision.
# The same file then passes under both `pytest` and `JAX_ENABLE_X64=1 pytest`.
RTOL = 1e-13 if jax.config.read("jax_enable_x64") else 2e-6


def make_operator(**overrides):
    kwargs = dict(
        t_unc=jnp.array(250.0), t_cos=jnp.array(30.0),
        t_sin=jnp.array(-40.0), t_rx=jnp.array(290.0),
        gamma_src_re=jnp.asarray(G_SRC_RE), gamma_src_im=jnp.asarray(G_SRC_IM),
        gamma_rec_re=jnp.asarray(G_REC_RE), gamma_rec_im=jnp.asarray(G_REC_IM),
    )
    kwargs.update(overrides)
    return NoiseWaveOperator(**kwargs)


def make_state(switch=None, data=None):
    extra = {} if switch is None else {"receiver_input": jnp.asarray(switch)}
    return State(
        data=jnp.full((N_TIME, N_FREQ), 300.0) if data is None else data,
        coords=Coordinates(
            time=jnp.arange(float(N_TIME)),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
            extra=extra,
        ),
    )


class TestForward:
    def test_it_reproduces_rhino_cal_jax_directly(self):
        import rhino_cal_jax as rcj

        switch = np.arange(N_TIME) % N_SOURCE
        out = make_operator()(make_state(switch)).data

        cycle = rcj.SwitchCycle(
            source_index=jnp.asarray(switch),
            labels=tuple(str(i) for i in range(N_SOURCE)),
        )
        stacked = cycle.gather(
            rcj.couplings(
                jnp.asarray(G_SRC_RE + 1j * G_SRC_IM),
                jnp.asarray(G_REC_RE + 1j * G_REC_IM),
            ).stacked
        )
        expected = (
            300.0 * stacked[..., 0] + 250.0 * stacked[..., 1]
            + 30.0 * stacked[..., 2] - 40.0 * stacked[..., 3] + 290.0
        )
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=RTOL)

    def test_a_matched_single_source_reduces_to_t_src_plus_t_rx(self):
        op = make_operator(
            gamma_src_re=jnp.zeros((1, N_FREQ)), gamma_src_im=jnp.zeros((1, N_FREQ)),
            gamma_rec_re=jnp.zeros(N_FREQ), gamma_rec_im=jnp.zeros(N_FREQ),
        )
        out = op(make_state(np.zeros(N_TIME, dtype=int))).data
        np.testing.assert_allclose(np.asarray(out), 300.0 + 290.0, rtol=RTOL)

    def test_a_single_source_needs_no_switch_array(self):
        op = make_operator(
            gamma_src_re=jnp.zeros((1, N_FREQ)), gamma_src_im=jnp.zeros((1, N_FREQ)),
        )
        assert op(make_state()).data.shape == (N_TIME, N_FREQ)

    def test_per_channel_noise_wave_temperatures_are_accepted(self):
        op = make_operator(t_unc=jnp.linspace(240.0, 260.0, N_FREQ))
        out = op(make_state(np.arange(N_TIME) % N_SOURCE)).data
        assert out.shape == (N_TIME, N_FREQ)


class TestRejections:
    def test_several_sources_without_a_switch_array_is_refused(self):
        """Silently using source 0 for every sample is finite and wrong."""
        with pytest.raises(StateValidationError, match="receiver_input"):
            make_operator()(make_state())

    def test_a_switch_longer_than_the_data_is_refused(self):
        with pytest.raises(StateValidationError, match="n_time"):
            make_operator()(make_state(np.zeros(N_TIME + 1, dtype=int)))

    def test_mismatched_gamma_real_and_imaginary_shapes_are_refused(self):
        with pytest.raises(StateValidationError, match="gamma_src"):
            make_operator(gamma_src_im=jnp.zeros((N_SOURCE + 1, N_FREQ)))

    def test_a_gamma_whose_channels_do_not_match_the_receiver_is_refused(self):
        with pytest.raises(StateValidationError, match="n_freq"):
            make_operator(gamma_rec_re=jnp.zeros(N_FREQ + 1),
                          gamma_rec_im=jnp.zeros(N_FREQ + 1))

    def test_a_one_dimensional_gamma_src_is_refused(self):
        with pytest.raises(StateValidationError, match="2D"):
            make_operator(gamma_src_re=jnp.zeros(N_FREQ),
                          gamma_src_im=jnp.zeros(N_FREQ))

    def test_data_whose_channels_disagree_with_gamma_is_refused(self):
        state = make_state(np.arange(N_TIME) % N_SOURCE,
                           data=jnp.full((N_TIME, N_FREQ + 2), 300.0))
        with pytest.raises(StateValidationError, match="n_freq"):
            make_operator()(state)


class TestIdentifiability:
    """What switching buys, counted per frequency channel.

    Each switch position contributes exactly ONE equation per channel against
    the three unknowns living there, so rank is ``min(n_src, 3) * n_freq``.

    This deliberately does NOT use scalar temperatures: those are fully
    identified by a single load, so a scalar test passes with one source and
    demonstrates nothing about switching.
    """

    def _per_channel_jacobian(self, n_source: int) -> np.ndarray:
        switch = np.arange(N_TIME) % n_source

        def predict(flat):  # flat: (3, n_freq) -- t_unc, t_cos, t_sin per channel
            op = make_operator(
                t_unc=flat[0], t_cos=flat[1], t_sin=flat[2],
                gamma_src_re=jnp.asarray(G_SRC_RE[:n_source]),
                gamma_src_im=jnp.asarray(G_SRC_IM[:n_source]),
            )
            return op(make_state(switch)).data.ravel()

        jac = jax.jacobian(predict)(jnp.zeros((3, N_FREQ)))
        return np.asarray(jac).reshape(-1, 3 * N_FREQ)

    @pytest.mark.parametrize("n_source,expected_rank", [(1, N_FREQ), (2, 2 * N_FREQ)])
    def test_too_few_loads_leave_the_system_rank_deficient(self, n_source, expected_rank):
        jac = self._per_channel_jacobian(n_source)
        assert np.linalg.matrix_rank(jac) == expected_rank
        assert expected_rank < 3 * N_FREQ

    def test_three_loads_make_the_per_channel_system_full_rank(self):
        assert np.linalg.matrix_rank(self._per_channel_jacobian(3)) == 3 * N_FREQ

    def test_the_operator_reproduces_its_own_truth(self):
        """Sanity: the forward model is deterministic and Gamma is wired per source."""
        switch = np.arange(N_TIME) % N_SOURCE
        truth = make_operator()(make_state(switch)).data

        def predict(t_nw):
            return make_operator(t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2])(
                make_state(switch)
            ).data.ravel()

        np.testing.assert_allclose(
            np.asarray(predict(jnp.array([250.0, 30.0, -40.0]))),
            np.asarray(truth).ravel(), rtol=RTOL,
        )


class TestTransforms:
    def test_gradients_reach_every_temperature_and_gamma(self):
        switch = np.arange(N_TIME) % N_SOURCE
        op = make_operator()

        def loss(operator):
            return jnp.sum(operator(make_state(switch)).data)

        grads = jax.grad(loss)(op)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert np.all(np.isfinite(np.asarray(leaf)))

    def test_the_operator_jits(self):
        switch = np.arange(N_TIME) % N_SOURCE
        out = jax.jit(lambda o, s: o(s).data)(make_operator(), make_state(switch))
        assert out.shape == (N_TIME, N_FREQ)
