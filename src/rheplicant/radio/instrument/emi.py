"""EMIOperator — PLACEHOLDER self-generated interference.

Element: "Self-generated EMI (e.g. due to power supply fluctuations,
switched-mode power sources and some control signals from the Arduino control
board, Odroid computer, or the SDR itself). Mostly looks like RFI."

Real physics to come: characterised spectral lines of the system's own
electronics (switching harmonics form comb-like structures). The placeholder
adds a constant-amplitude frequency comb.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


class EMIOperator(AbstractOperator):
    """Add a frequency comb of self-generated EMI lines (placeholder).

    Every ``period``-th channel receives an extra ``amplitude``.

    Note:
        **``period`` counts CHANNELS, not hertz, and real EMI does not.** A
        switching supply or a clock harmonic sits at a fixed frequency spacing;
        this comb sits at a fixed channel spacing, so the same operator models
        a different physical source at every channelisation. Re-bin the band
        and the lines move.

        The consequence to keep in mind while it is a stand-in: a fit that
        constrains ``amplitude`` on one channelisation says nothing about the
        same instrument read out at another. When the body becomes real,
        ``period`` should become a frequency spacing in Hz and the comb should
        be built from ``state.coords.freq`` rather than from ``arange``, which
        is also what makes it independent of ``n_freq``.

        The comb is also exactly periodic and infinitely sharp: no line width,
        no per-line amplitude, no drift. Real EMI has all three.

    Attributes:
        amplitude: line amplitude [K-equivalent] — differentiable scalar.
        period: channel spacing of the comb (static configuration). Channels,
            not hertz — see the note above.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "emi"

    amplitude: jax.Array
    period: int = eqx.field(static=True)

    def __check_init__(self):
        if not isinstance(self.period, int) or self.period < 1:
            raise StateValidationError(f"period must be a positive int, got {self.period!r}.")

    def __call__(self, state: State) -> State:
        n_freq = state.data.shape[-1]
        comb = (jnp.arange(n_freq) % self.period == 0).astype(state.data.dtype)
        return state.with_data(state.data + self.amplitude * comb[None, :])
