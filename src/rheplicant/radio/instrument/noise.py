"""NoiseOperator — PLACEHOLDER radiometric noise.

Real physics to come (radiometer equation, limTOD / hydra-tod noise models):
the noise level tracks the total power itself,
``sigma = T_total / sqrt(dt * dnu)``, plus correlated 1/f fluctuations.
This placeholder adds white Gaussian noise with a fixed sigma, demonstrating
the PRNG-consumption contract.
"""

from typing import ClassVar

import jax

from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class NoiseOperator(AbstractOperator):
    """Add white Gaussian noise to ``state.data`` (placeholder).

    Consumes randomness through the State PRNG protocol: the returned state
    carries an *advanced* key, so repeated application gives fresh draws while
    a single seed reproduces the whole pipeline.

    Attributes:
        sigma: noise standard deviation [K] — differentiable scalar.
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "key")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "noise"

    sigma: jax.Array

    def __call__(self, state: State) -> State:
        subkey, state = state.next_key()
        noise = self.sigma * jax.random.normal(subkey, jax.numpy.shape(state.data))
        return state.with_data(state.data + noise)
