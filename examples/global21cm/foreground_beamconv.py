"""Demo A: the beam-convolved foreground as a per-LST moment expansion.

Each LST sample ``t`` gets its own spectrum,

    T_fg(nu, t) = sum_j a[t, j] b_j(nu),

over a fixed spectral basis ``b_j``. The basis is MERS's moment expansion
(``MERS/SEDfitting.py``, ``fg_moment_basis._single_pivot_basis``, lines
216-254),

    phi_k(nu) = (nu / nu_ref)^beta0 [ln(nu / nu_ref)]^k,   k = 0 .. K-1,

optionally multiplied, column by column, by the leading beam-chromaticity
spectra ``u_i(nu)`` of an SVD of the beam maps (MERS's ``BCSVD`` route,
``tensor_prod_basis``, lines 181-197), and every column scaled to unit l2
norm as MERS does. The coefficients ``a`` are the only free foreground
parameters and enter linearly with a flat prior, so they are integrated out
exactly (:func:`global21cm.collapse.per_lst_basis`).
:func:`global21cm.strategies.per_lst_moments` picks the number of moments
and beam spectra by the rule the document names (one of
``strategies.ORDER_RULES``), in the document's hook and in ``prepare.py``
alike.
:class:`MomentForeground` is registered by the plugin and tested, but no
document uses it.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State


def moment_columns(freqs_mhz, beta0: float, n_moments: int, nu_ref_mhz: float) -> np.ndarray:
    """``phi_k``, ``(n_freq, n_moments)``, before normalisation."""
    log_x = np.log(np.asarray(freqs_mhz, dtype=np.float64) / nu_ref_mhz)
    vander = np.vander(log_x, N=n_moments, increasing=True)
    return vander * np.exp(beta0 * log_x)[:, None]


def beam_svd_spectra(beam_maps, n_beam: int) -> np.ndarray:
    """Leading left singular vectors of the pixel-sum-normalised beam stack.

    ``beam_maps`` is ``(n_freq, n_pix)``; each row is divided by its sum so
    that the spectra describe the shape of the normalised beam, which is
    what the drift scan's ``normalize=True`` projects with.
    """
    maps = np.asarray(beam_maps, dtype=np.float64)
    maps = maps / maps.sum(axis=1, keepdims=True)
    u, _, _ = np.linalg.svd(maps, full_matrices=False)
    return u[:, :n_beam]


def moment_basis(
    freqs_mhz,
    beta0: float,
    n_moments: int,
    nu_ref_mhz: float,
    beam_spectra=None,
) -> np.ndarray:
    """The unit-norm basis ``(n_freq, n_basis)``; ``n_basis = n_moments * n_beam``.

    Column order follows MERS's ``tensor_prod_basis``: moment-major, beam
    spectrum minor.
    """
    columns = moment_columns(freqs_mhz, beta0, n_moments, nu_ref_mhz)
    if beam_spectra is not None:
        spectra = np.asarray(beam_spectra, dtype=np.float64)
        columns = (columns[:, :, None] * spectra[:, None, :]).reshape(columns.shape[0], -1)
    return columns / np.linalg.norm(columns, axis=0)[None, :]


def orthonormal_span(basis) -> np.ndarray:
    """An orthonormal basis of the same column space (thin QR).

    The moment columns are nearly collinear (on the 55-85 MHz band the Gram
    matrix's condition number was measured at 2.2e4 for seven plain moments
    and 1.4e14 for seven moments times four beam spectra). The span, and so
    every prediction the model can make, is unchanged.
    """
    q, _ = np.linalg.qr(np.asarray(basis, dtype=np.float64))
    return q


class MomentForeground(AbstractOperator):
    """Per-LST foreground spectra on a fixed basis, linear in ``coefficients``.

    Attributes:
        coefficients: ``(n_time, n_basis)`` amplitudes [K].
        basis: ``(n_freq, n_basis)`` unit-norm spectral basis
            [dimensionless], not a fitted quantity.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "observed_astro_sky"

    coefficients: jax.Array
    basis: jax.Array

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None or state.coords.time is None:
            raise StateValidationError("MomentForeground requires coords.time and coords.freq.")
        n_time, n_freq = state.coords.time.shape[0], state.coords.freq.shape[0]
        if self.basis.shape[0] != n_freq or self.coefficients.shape[0] != n_time:
            raise StateValidationError(
                f"MomentForeground: basis {self.basis.shape} and coefficients "
                f"{self.coefficients.shape} do not match the grid ({n_time}, {n_freq})."
            )
        return state.with_data(jnp.einsum("tj,fj->tf", self.coefficients, self.basis))

