"""Shared builders for the inference-section tests: one small twin, one state.

Mirrors ``tests/config/test_config_section_compose.py``'s fixtures, hoisted
because five 2B test files need the same twin. Everything here is float32 and
tiny: 16 times, 8 channels, a source (``global_signal``) plus ``gain`` -- and,
when asked, a stochastic ``noise`` node for the twin-repair tests.
"""

import jax
import jax.numpy as jnp

from rheplicant.config.context import ResolutionContext
from rheplicant.config.sections.compose import build_model
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.state import State

FREQ_HZ = jnp.linspace(6.0e7, 8.5e7, 8)
TIME_S = jnp.arange(16.0) * 2.0

MODEL = {
    "global_signal": {"depth": {"value": 0.5, "unit": "K"},
                      "centre": {"value": 75.0, "unit": "MHz"},
                      "width": {"value": 5.0, "unit": "MHz"}},
    "gain": {"gain": {"value": 1.1, "unit": "dimensionless"}},
}

NOISY_MODEL = {**MODEL,
               "noise": {"type": "NoiseOperator",
                         "sigma": {"value": 0.5, "unit": "K"}}}


def context(**overrides):
    base = dict(freq=FREQ_HZ, time=TIME_S, dtype="float32",
                seed=20260806, seeds={}, switch_order=())
    base.update(overrides)
    return ResolutionContext(**base)


def twin(model=None, ctx=None):
    return build_model(dict(model if model is not None else MODEL),
                       ctx if ctx is not None else context(),
                       switch_order=())


def state(with_key=True):
    return State(coords=Coordinates(time=TIME_S, freq=FREQ_HZ),
                 key=jax.random.key(20260806) if with_key else None)
