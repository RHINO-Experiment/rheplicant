"""NoiseWaveOperator — PLACEHOLDER receiver noise-wave / reflection terms.

Elements: "Reflections and bandpass shape ... includes the noise wave Gamma
parameters. Antenna also produces significant reflection term" and
"Noise-wave reflection terms (the noise-wave T parameters; expected to be
relatively smooth in LST and frequency)".

Real physics to come (noise-wave GCR draft, Eq. 1): the measured power of a
source with noise temperature ``T_src`` and reflection coefficient
``Gamma_src`` is::

    P = g * ( T_src (1-|G|^2) |F|^2 + T_unc |G|^2 |F|^2
              + T_cos Re(G F) + T_sin Im(G F) + T_0 )

with ``F = sqrt(1-|G_rec|^2) / (1 - G G_rec)``. The placeholder implements
the ``F -> 1`` limit with a real/imaginary reflection pair, preserving the
property that matters for calibration: the output is *linear* in the
noise-wave vector ``t_nw = (T_unc, T_cos, T_sin)`` — the ``d = H t_nw``
structure that Gaussian Constrained Realisation sampling relies on (draft
Eqs. 20, 24, 28; ``g`` and ``T_0`` are common to all PSD measurements and
cancel in the quotient construction, so ``T_0`` is not a column of ``H`` —
though the operator output happens to be linear in ``T_0`` as well).
"""

from typing import ClassVar

import jax

from replicant.core.operator import AbstractOperator
from replicant.core.state import State


class NoiseWaveOperator(AbstractOperator):
    """Apply reflection loss and add noise-wave terms (placeholder, F -> 1).

    ``data * (1 - |G|^2) + T_unc |G|^2 + T_cos Re(G) + T_sin Im(G) + T_0``

    All parameters are differentiable leaves; each may be a scalar or a
    per-frequency ``(n_freq,)`` array (the real terms are smooth in
    frequency and slowly varying in time).

    Attributes:
        t_unc: uncorrelated noise-wave temperature [K].
        t_cos: cosine (in-phase) noise-wave temperature [K].
        t_sin: sine (quadrature) noise-wave temperature [K].
        t_zero: offset temperature T_0 [K].
        gamma_re: Re(Gamma) of the source reflection coefficient.
        gamma_im: Im(Gamma) of the source reflection coefficient.
    """

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "noise_wave"

    t_unc: jax.Array
    t_cos: jax.Array
    t_sin: jax.Array
    t_zero: jax.Array
    gamma_re: jax.Array
    gamma_im: jax.Array

    def __call__(self, state: State) -> State:
        gamma_sq = self.gamma_re**2 + self.gamma_im**2
        data = (
            state.data * (1.0 - gamma_sq)
            + self.t_unc * gamma_sq
            + self.t_cos * self.gamma_re
            + self.t_sin * self.gamma_im
            + self.t_zero
        )
        return state.with_data(data)
