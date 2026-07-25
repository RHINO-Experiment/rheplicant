"""End-to-end: the full 8-stage radio digital-twin demo under jit/grad/vmap.

This is the headline guarantee of the framework: an entire heterogeneous
instrument pipeline is one differentiable, jit-compilable, vmappable function.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from replicant import Pipeline, State, SumOperator
from replicant.inference import build_forward_fn, mean_squared_error
from replicant.radio import (
    ADCOperator,
    AtmosphericEmissionOperator,
    BackendOperator,
    BeamOperator,
    GainOperator,
    NoiseOperator,
    ReceiverOperator,
    SkyOperator,
)

N_FREQ = 4  # matches tests/conftest.py


@pytest.fixture
def demo_pipeline():
    # Atmospheric emission is a source branch of the antenna-temperature sum,
    # mirroring the canonical graph (parallel to ground_pickup/t_sys_extra).
    t_ant = SumOperator(
        Pipeline(
            SkyOperator(amplitude=jnp.array(1.0e3)),
            BeamOperator(solid_angle=jnp.array(0.8)),
            names=("sky", "beam"),
        ),
        AtmosphericEmissionOperator(t_atm=jnp.array(150.0)),
        names=("observed_sky", "atmosphere"),
    )
    return Pipeline(
        t_ant,
        ReceiverOperator(bandpass=jnp.ones(N_FREQ)),
        GainOperator(gain=jnp.array(1.0)),
        NoiseOperator(sigma=jnp.array(0.1)),
        ADCOperator(scale=jnp.array(1.0), n_bits=14),
        BackendOperator(n_chunk=4),
        names=("t_ant", "receiver", "gain", "noise", "adc", "backend"),
    )


class TestEndToEnd:
    def test_runs_under_jit(self, demo_pipeline, template_state):
        out_eager = demo_pipeline(template_state)
        out_jit = eqx.filter_jit(demo_pipeline)(template_state)
        assert out_jit.data.shape == (2, N_FREQ)  # 8 samples / n_chunk=4
        assert jnp.allclose(out_jit.data, out_eager.data)
        # metadata survives the whole pipeline (bookkeeping requirement)
        assert out_jit.meta["telescope"] == "RHINO"
        assert out_jit.env is not None

    def test_grad_wrt_all_params(self, demo_pipeline, template_state):
        observed = demo_pipeline(template_state).data

        def loss(pipeline):
            return mean_squared_error(pipeline(template_state).data, observed * 1.05)

        grads = eqx.filter_grad(loss)(demo_pipeline)
        leaves = jax.tree.leaves(eqx.filter(grads, eqx.is_inexact_array))
        # amplitude, solid_angle, t_sys, bandpass, gain, sigma, adc-scale
        assert len(leaves) == 7
        assert all(jnp.all(jnp.isfinite(leaf)) for leaf in leaves)
        assert any(jnp.any(leaf != 0) for leaf in leaves)

    def test_jit_of_grad_compiles_once(self, demo_pipeline, template_state):
        observed = demo_pipeline(template_state).data
        traces = []

        @eqx.filter_jit
        def grad_step(pipeline):
            traces.append(1)
            return eqx.filter_grad(
                lambda p: mean_squared_error(p(template_state).data, observed)
            )(pipeline)

        grad_step(demo_pipeline)
        grad_step(demo_pipeline)
        assert len(traces) == 1

    def test_vmap_over_keys_gives_distinct_realisations(self, demo_pipeline, coords):
        keys = jax.random.split(jax.random.key(0), 3)

        def realise(key):
            return demo_pipeline(State(coords=coords, key=key)).data

        batch = eqx.filter_vmap(realise)(keys)
        assert batch.shape == (3, 2, N_FREQ)
        assert not jnp.allclose(batch[0], batch[1])  # different keys, different noise

    def test_forward_fn_composes_with_pipeline(self, demo_pipeline, template_state):
        forward, params0 = build_forward_fn(demo_pipeline, template_state)
        assert jnp.array_equal(forward(params0), demo_pipeline(template_state).data)
        grads = jax.grad(lambda p: jnp.sum(forward(p)))(params0)
        assert all(jnp.all(jnp.isfinite(g)) for g in jax.tree.leaves(grads))
