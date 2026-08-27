"""The instrument vocabulary the seam examples share, in one place.

A module rather than constants repeated in ``conftest.py`` and the test file:
this directory's whole subject is two implementations agreeing, and the fastest
way to fake that agreement is to let a fixture and its reference drift into
being two different models. Everything numerical the examples lean on is
spelled here once.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp

from rheplicant.core.combinators import SumOperator
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.radio import SkyOperator

#: The two sky amplitudes and the gain the instrument fixture is built at.
SKY_A = 100.0
SKY_B = 20.0
GAIN = 1.5

#: The template state's grid, restated here only so the examples can size an
#: array without importing the root conftest's private constants.
N_TIME = 8
N_FREQ = 4


class ComplexCoeffOperator(AbstractOperator):
    """A real observation linear in COMPLEX coefficients -- sky alms in miniature.

    Test double, not physics. It exists so the complex path through the adapter
    is exercised on a model whose imaginary half is NOT a null direction: the
    matrix is complex, so a solve that dropped the imaginary part would be
    visibly wrong rather than merely unconstrained.
    """

    provides: ClassVar[tuple[str, ...]] = ("data",)

    coeffs: jax.Array
    matrix: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        return state.with_data(jnp.real(self.matrix @ self.coeffs).reshape(n_time, n_freq))


class LogGainOperator(AbstractOperator):
    """A gain declared in LOG units: ``data -> data * exp(log_gain)``.

    Test double, and the shape example 9 needs rather than a convenience. A
    gain declared multiplicatively is affine in itself, so ``log(prediction)``
    is affine in ``log(gain)`` and NOT in ``gain`` -- measured: taking the
    ordinary ``GainOperator`` to log space refuses, because the block's offset
    is the prediction at ``gain = 0`` and ``log(0)`` is ``-inf``. A gain
    parameterised in log units is the model that genuinely has a log-linear
    route, and declaring one in dB-like units is what an instrument model does
    anyway.
    """

    provides: ClassVar[tuple[str, ...]] = ("data",)

    log_gain: jax.Array

    def __call__(self, state):
        return state.with_data(state.data * jnp.exp(self.log_gain)[:, None])


def log_gain_instrument():
    """The same two-term sky, through a gain declared in log units.

    A builder rather than a fixture so the test module and this one cannot end
    up holding two spellings of the same instrument.
    """
    return Pipeline(
        SumOperator(
            SkyOperator(amplitude=jnp.array(SKY_A)),
            SkyOperator(amplitude=jnp.array(SKY_B)),
            names=("sky_a", "sky_b"),
        ),
        LogGainOperator(log_gain=jnp.zeros(N_TIME)),
        names=("sum", "gain"),
    )
