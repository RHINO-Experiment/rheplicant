"""Sky models: parameters -> a sky representation on the frequency grid.

One half of the modular sky abstraction (the other half is
:mod:`~replicant.radio.sky.projection`):

    SkyModel  (what the sky IS)   ->  (n_freq, n_pix) brightness maps
    Projector (how the sky is SEEN) ->  (n_time, n_freq) antenna temperature

Keeping them separate means the same sky (e.g. moment-expanded foregrounds)
can be observed through different engines (limTOD beam convolution, m-mode
transfer matrices, ...) and the same engine can observe different skies.

Representation contract: ``__call__(freq) -> Array[(n_freq, n_pix)]`` of real
brightness temperatures [K]. Pixelization is HEALPix RING in the real
implementations; the placeholders are pixelization-agnostic (any ``n_pix``).
"""

import abc

import equinox as eqx
import jax
import jax.numpy as jnp

from replicant.core.errors import StateValidationError


class AbstractSkyModel(eqx.Module):
    """Parameters -> sky brightness maps ``(n_freq, n_pix)`` [K].

    Differentiable sky parameters (amplitudes, spectral indices, moment
    coefficients...) are ordinary array fields of the concrete model.
    """

    @abc.abstractmethod
    def __call__(self, freq: jax.Array) -> jax.Array:
        """Evaluate the sky on the ``(n_freq,)`` frequency grid [Hz]."""


class UniformSkyModel(AbstractSkyModel):
    """PLACEHOLDER: spatially and spectrally uniform sky.

    Attributes:
        amplitude: brightness temperature [K] — differentiable scalar.
        n_pix: number of sky pixels (static configuration).
    """

    amplitude: jax.Array
    n_pix: int = eqx.field(static=True)

    def __check_init__(self):
        if not isinstance(self.n_pix, int) or self.n_pix < 1:
            raise StateValidationError(f"n_pix must be a positive int, got {self.n_pix!r}.")

    def __call__(self, freq: jax.Array) -> jax.Array:
        return self.amplitude * jnp.ones((freq.shape[0], self.n_pix))


class PowerLawSkyModel(AbstractSkyModel):
    """PLACEHOLDER: power-law sky with a per-pixel amplitude map.

    ``T(freq, pix) = amplitude[pix] * (freq / ref_freq) ** (-spectral_index)``

    Real version: uncertain spectral-index maps / moment-expanded foregrounds
    (the identified foreground pain point) — same contract, more parameters.

    Attributes:
        amplitude: ``(n_pix,)`` amplitude map at ``ref_freq`` [K] (or scalar).
        spectral_index: power-law index — differentiable scalar.
        ref_freq: reference frequency [Hz] (static configuration).
        n_pix: number of sky pixels (static configuration).
    """

    amplitude: jax.Array
    spectral_index: jax.Array
    ref_freq: float = eqx.field(static=True)
    n_pix: int = eqx.field(static=True)

    def __check_init__(self):
        if self.ref_freq <= 0:
            raise StateValidationError(f"ref_freq must be > 0, got {self.ref_freq}.")
        if not isinstance(self.n_pix, int) or self.n_pix < 1:
            raise StateValidationError(f"n_pix must be a positive int, got {self.n_pix!r}.")

    def __call__(self, freq: jax.Array) -> jax.Array:
        spectrum = (freq / self.ref_freq) ** (-self.spectral_index)  # (n_freq,)
        amplitude = jnp.broadcast_to(self.amplitude, (self.n_pix,))
        return jnp.outer(spectrum, amplitude)
