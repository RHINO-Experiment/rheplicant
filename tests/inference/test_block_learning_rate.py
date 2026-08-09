"""Block.learning_rate reaches gradient_estimate.

engines.gradient_estimate has accepted a learning_rate since it was written;
plan.py simply never passed one, so a gradient block's step size was a
documented default nobody could change.
"""

import pytest

from rheplicant.inference import Block
from rheplicant.inference.engines import DEFAULT_LEARNING_RATE


def _run_one_gradient_block_estimate(block):
    """Smallest possible SamplingPlan.estimate over one gradient block.

    A scalar gain on a two-operator twin: SkyOperator makes data, GainOperator
    scales it. `gain` is declared WITHOUT linear=True, so the plan derives a
    gradient block for it.
    """
    import jax.numpy as jnp

    from rheplicant import Coordinates, State
    from rheplicant.inference import (
        Bind,
        HomoscedasticNoise,
        Latent,
        ParameterSpace,
        SamplingPlan,
    )
    from rheplicant.radio import GainOperator, SkyOperator, assemble

    n_time, n_freq = 4, 3
    state = State(
        coords=Coordinates(
            time=jnp.arange(float(n_time)),
            freq=jnp.linspace(60e6, 85e6, n_freq),
        )
    )
    twin = assemble(SkyOperator(amplitude=jnp.array(10.0)),
                    GainOperator(gain=jnp.array(1.0)))
    observed = twin(state).data * 1.2

    space = ParameterSpace(
        latents=[Latent("gain", init=jnp.array(1.0))],
        bindings=[Bind("gain", into=lambda p: p["gain"].gain)],
    )
    plan = SamplingPlan(space, block)
    return plan.estimate(twin, state, observed,
                         noise=HomoscedasticNoise(sigma=jnp.array(1.0)))


def test_block_defaults_learning_rate_to_none():
    assert Block("gain").learning_rate is None


def test_block_accepts_a_learning_rate():
    assert Block("gain", learning_rate=3e-3).learning_rate == 3e-3


def test_block_refuses_a_non_positive_learning_rate():
    with pytest.raises(Exception, match="learning_rate must be > 0"):
        Block("gain", learning_rate=0.0)


def test_block_refuses_a_learning_rate_on_a_conjugate_block():
    """Same rule as `steps`: an argument a conjugate solve cannot mean is an
    error, not something silently ignored."""
    with pytest.raises(Exception, match="conjugate"):
        Block("t_nw", engine="conjugate", learning_rate=1e-3)


def test_the_declared_rate_reaches_gradient_estimate(monkeypatch):
    """The plumbing test: intercept the engine and read what it was handed."""
    import rheplicant.inference.plan as plan_mod

    seen = {}
    real = plan_mod.gradient_estimate

    def spy(cond, names, values, *, steps, learning_rate=DEFAULT_LEARNING_RATE, **kw):
        seen["learning_rate"] = learning_rate
        seen["steps"] = steps
        return real(cond, names, values, steps=steps, learning_rate=learning_rate, **kw)

    monkeypatch.setattr(plan_mod, "gradient_estimate", spy)
    _run_one_gradient_block_estimate(Block("gain", steps=2, learning_rate=7e-3))

    assert seen["learning_rate"] == 7e-3
    assert seen["steps"] == 2


def test_an_undeclared_rate_reaches_the_engine_default(monkeypatch):
    import rheplicant.inference.plan as plan_mod

    seen = {}
    real = plan_mod.gradient_estimate

    def spy(cond, names, values, *, steps, learning_rate=DEFAULT_LEARNING_RATE, **kw):
        seen["learning_rate"] = learning_rate
        return real(cond, names, values, steps=steps, learning_rate=learning_rate, **kw)

    monkeypatch.setattr(plan_mod, "gradient_estimate", spy)
    _run_one_gradient_block_estimate(Block("gain", steps=2))

    assert seen["learning_rate"] == DEFAULT_LEARNING_RATE
