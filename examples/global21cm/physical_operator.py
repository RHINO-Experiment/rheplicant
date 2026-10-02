"""Demo B's foreground: low-resolution sky maps seen through the drift scan.

The sky is the MERS synchrotron law (:mod:`global21cm.foreground_physical`)
on an NSIDE 16 HEALPix grid, expanded to first order in the spectral index
about a linearisation point ``beta_t``:

    T(p, nu) = E(p, nu) [A(p) + B(p) L(nu)],
    E(p, nu) = exp(beta_t(p) L(nu) + c (Q(nu) - Q(nu0))),   L = ln(nu / nu0).

``A`` is the amplitude map at ``nu0 = 70 MHz`` and ``B = A (beta - beta_t)``
carries the spectral-index map, ``beta = beta_t + B / A``. Here the
curvature ``c`` is fixed inside the spectra ``E``: :class:`PhysicalSkyForeground`
is the fixed-curvature operator the tests hold against the simulator
(registered by the plugin, used by no document). The documents' demo B adds
a global curvature column to :func:`moment_jacobian`'s ``(A, B)`` columns,
with its own Gaussian prior (:func:`global21cm.physical_inputs.jacobian`).
The foreground is linear in ``(A, B)``, which is what lets
:mod:`global21cm.collapse` integrate it out exactly; the expansion replaced
a nonlinear index map that did not sample (README, "Deviations").

The drift scan enters as ``response``, the ``(n_freq, n_time, n_pix)``
matrix that limtod_jax's m-mode projection is
(:func:`global21cm.instrument.drift_response`). :func:`moment_jacobian` and
:class:`PhysicalSkyForeground` are the same linear map written two ways; the
tests hold them equal, and the operator equal to the simulator where the
two models coincide.
"""

from __future__ import annotations

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
from rheplicant.core.errors import StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State

from global21cm import scenario


def moment_jacobian(response, spectra, log_ratio) -> np.ndarray:
    """Waterfall Jacobian with respect to ``(A, B)``, ``(n_freq * n_time, 2 n_pix)``.

    Rows are ordered frequency-major, matching ``waterfall.T.reshape(-1)``.
    """
    response = np.asarray(response, dtype=np.float64)
    spectra = np.asarray(spectra, dtype=np.float64)
    log_ratio = np.asarray(log_ratio, dtype=np.float64)
    n_pix = response.shape[-1]
    jac_a = (response * spectra[:, None, :]).reshape(-1, n_pix)
    jac_b = (response * (spectra * log_ratio[:, None])[:, None, :]).reshape(-1, n_pix)
    return np.hstack([jac_a, jac_b])


class PhysicalSkyForeground(AbstractOperator):
    """Beam-convolved synchrotron from the ``(A, B)`` maps, linear in ``coords``.

    Attributes:
        coords: ``(n_coord,)`` whitened coordinates, the fitted latents.
        offset: ``(2 n_pix,)`` K; ``(A, B)`` at ``coords = 0``.
        design: ``(2 n_pix, n_coord)`` K; ``(A, B) = offset + design @ coords``.
        spectra: ``(n_freq, n_pix)`` template spectral factors ``E``.
        response: ``(n_freq, n_time, n_pix)`` drift-scan matrix.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    graph_node: ClassVar[str] = "observed_astro_sky"

    coords: jax.Array
    offset: jax.Array
    design: jax.Array
    spectra: jax.Array
    response: jax.Array

    def maps(self) -> tuple[jax.Array, jax.Array]:
        """``(A, B)`` in K at ``scenario.NU0_MHZ``."""
        both = self.offset + self.design @ self.coords
        n_pix = self.response.shape[-1]
        return both[:n_pix], both[n_pix:]

    def sky(self, freqs_mhz) -> jax.Array:
        """``(n_freq, n_pix)`` sky in K."""
        amplitude, first = self.maps()
        log_ratio = jnp.log(jnp.asarray(freqs_mhz) / scenario.NU0_MHZ)
        return self.spectra * (amplitude[None, :] + first[None, :] * log_ratio[:, None])

    def __call__(self, state: State) -> State:
        if state.coords is None or state.coords.freq is None or state.coords.time is None:
            raise StateValidationError("PhysicalSkyForeground requires coords.time and coords.freq.")
        n_time, n_freq = state.coords.time.shape[0], state.coords.freq.shape[0]
        n_pix = self.response.shape[-1]
        if self.response.shape[:2] != (n_freq, n_time) or self.spectra.shape != (n_freq, n_pix):
            raise StateValidationError(
                f"PhysicalSkyForeground: response {self.response.shape} and spectra "
                f"{self.spectra.shape} do not match the grid ({n_time}, {n_freq})."
            )
        sky = self.sky(state.coords.freq / 1e6)
        return state.with_data(jnp.einsum("ftp,fp->tf", self.response, sky))
