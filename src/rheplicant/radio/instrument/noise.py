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

import equinox as eqx
import jax

from rheplicant.core.errors import StateValidationError
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


class RadiometerNoiseOperator(AbstractOperator):
    """Draw radiometer noise on the signal path: ``d -> d (1 + f w)``.

    The GENERATOR half of the radiometer equation, ``f = 1/sqrt(dnu tau)`` --
    the same statics and the same multiplicative form as
    :meth:`rheplicant.inference.noise.RadiometerNoise.realise`, and for the
    same reason: ``sigma = |prediction| f`` takes an absolute value that a
    generator must not, and the two forms differ in sign wherever the
    prediction does. Sitting on the path, this operator HAS the total power
    the module docstring above says a radiometer draw needs.

    Decided as D-C17 (2026-08-09), for a twin that must be self-contained.
    The recorded cost: a run can now carry two sigmas -- this operator's and
    ``inference.noise``'s -- and nothing in the package keeps them equal, so
    the config layer's validation (Plan 3) must cross-check them and refuse a
    disagreement, naming both paths. That duty travels with this class.

    Attributes:
        channel_width: channel bandwidth [Hz] -- static, part of the jit key.
        integration_time: per-sample integration [s] -- static.
    """

    requires: ClassVar[tuple[str, ...]] = ("data", "key")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "noise"

    channel_width: float = eqx.field(static=True)
    integration_time: float = eqx.field(static=True)

    def __check_init__(self) -> None:
        for name, value in (
            ("channel_width", self.channel_width),
            ("integration_time", self.integration_time),
        ):
            if not value > 0.0:
                raise StateValidationError(
                    f"RadiometerNoiseOperator.{name} must be positive, got "
                    f"{value!r}."
                )

    @property
    def fractional(self) -> float:
        """``1/sqrt(dnu tau)`` -- the fractional radiometer scatter."""
        return 1.0 / (self.channel_width * self.integration_time) ** 0.5

    def __call__(self, state: State) -> State:
        subkey, state = state.next_key()
        draw = jax.random.normal(subkey, jax.numpy.shape(state.data))
        return state.with_data(state.data * (1.0 + self.fractional * draw))
