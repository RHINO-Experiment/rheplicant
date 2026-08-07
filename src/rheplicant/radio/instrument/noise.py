"""NoiseOperator — PLACEHOLDER radiometric noise.

Real physics to come (limTOD / hydra-tod noise models): correlated 1/f
fluctuations, and a level that tracks the total power as it is *drawn*.

The radiometer equation itself is not future work on the inference side --
:class:`~rheplicant.inference.noise.RadiometerNoise` is
``sigma = |prediction| / sqrt(delta_nu * tau)`` and is this package's default
noise model. The asymmetry is real and worth knowing: a likelihood can scale
its sigma with the prediction it already has, while *drawing* a realisation
whose sigma depends on the total power needs that power first. So this
operator adds white Gaussian noise at a fixed sigma, and an example wanting a
radiometric draw scales it by hand.
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
