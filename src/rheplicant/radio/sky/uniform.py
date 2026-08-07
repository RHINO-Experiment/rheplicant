"""SkyOperator — PLACEHOLDER sky: one uniform brightness, no map and no beam.

**The sky-TOD port is not coming here.** This module used to promise it --
"sky maps with spectral models, observed along ``coords.pointing``" -- and that
port has since arrived as
:class:`~rheplicant.radio.sky.source.SkySourceOperator` composed with a
projector, which is real rather than a stand-in. Leaving the promise here would
have pointed at work that is finished somewhere else.

What stays placeholder is the *sky*: one brightness temperature over the whole
(time, frequency) grid stands in for a real sky model. That is deliberately the
cheap path, and it is what a test, a smoke run or an inference fixture usually
wants when the sky is not the subject. Reach for the projector when the beam
matters, and for this when it does not.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class SkyOperator(AbstractOperator):
    """Fill ``state.data`` with a uniform sky brightness [K] (placeholder).

    Attributes:
        amplitude: sky brightness temperature [K] — a differentiable scalar.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "uniform_sky"

    amplitude: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.time is None or state.coords.freq is None:
            raise StateValidationError(
                "SkyOperator requires state.coords with both time and freq axes."
            )
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        sky = self.amplitude * jnp.ones((n_time, n_freq))
        return state.with_data(sky)
