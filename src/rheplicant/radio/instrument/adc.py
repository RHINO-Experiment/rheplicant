"""ADCOperator — PLACEHOLDER digitization.

Real physics to come: true quantization (round to 2**n_bits levels) has zero
gradient almost everywhere, so the differentiable version will need a
straight-through estimator (identity gradient through the rounding) or a
smooth surrogate. This placeholder applies scale + clip, which is
differentiable almost everywhere and preserves the saturation behaviour.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class ADCOperator(AbstractOperator):
    """Scale and clip ``state.data`` to the ADC dynamic range (placeholder).

    Note:
        **Two things a real ADC does that this one does not**, stated so the
        omission is not mistaken for the model:

        1. *No kelvin-to-counts link.* ``scale`` is a free differentiable
           number, not a calibrated conversion, so the output is in "counts"
           only by assertion. Nothing checks that ``scale`` corresponds to the
           front-end gain, and the clip limit ``2**(n_bits-1)`` is therefore
           compared against a quantity whose units are whatever ``scale`` made
           them.
        2. *No quantization noise.* The body clips and does not round, so the
           digitizer contributes no error at all. A real n-bit converter adds
           roughly ``LSB/sqrt(12)`` of it, and for a global-signal experiment
           quantization is one of the systematics the ADC exists to represent.

        What IS real here is the clipping, and it is the part worth having: it
        is where a mis-scaled model saturates rather than growing without
        bound, and the gradient through the clipped region is zero, which is
        visible in a fit.

    Attributes:
        scale: pre-digitization scaling — differentiable scalar. Not a
            calibrated K-to-counts conversion; see the note above.
        n_bits: ADC bit depth (static configuration; clip limit is 2**(n_bits-1)).
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "adc"

    scale: jax.Array
    n_bits: int = eqx.field(static=True)

    def __check_init__(self):
        if not isinstance(self.n_bits, int) or self.n_bits < 1:
            raise StateValidationError(f"n_bits must be a positive int, got {self.n_bits!r}.")

    def __call__(self, state: State) -> State:
        limit = 2.0 ** (self.n_bits - 1)
        return state.with_data(jnp.clip(state.data * self.scale, -limit, limit))
