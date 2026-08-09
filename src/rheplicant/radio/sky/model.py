"""Sky models: parameters -> a sky representation on the frequency grid.

One half of the modular sky abstraction (the other half is
:mod:`~rheplicant.radio.sky.projection`):

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

from rheplicant.core.errors import StateValidationError


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
        if not self.ref_freq > 0:  # `not >` so a NaN ref_freq is refused too
            raise StateValidationError(f"ref_freq must be > 0, got {self.ref_freq}.")
        if not isinstance(self.n_pix, int) or self.n_pix < 1:
            raise StateValidationError(f"n_pix must be a positive int, got {self.n_pix!r}.")

    def __call__(self, freq: jax.Array) -> jax.Array:
        spectrum = (freq / self.ref_freq) ** (-self.spectral_index)  # (n_freq,)
        amplitude = jnp.broadcast_to(self.amplitude, (self.n_pix,))
        return jnp.outer(spectrum, amplitude)


class MapSky(AbstractSkyModel):
    """Fixed brightness maps, and the frequency grid they were built on.

    The stand-in for a GSM / pyGDSM realisation, and the shape every worked
    example in this package reaches for. ``__call__`` returns the stored maps
    and **does not consult its ``freq`` argument** beyond checking that it has
    the same shape as the grid the maps were built on.

    **What that check does and does not catch.** A map built for 60-85 MHz and
    evaluated on a 60-85 MHz grid of a different length -- or of the same length
    and a different rank, such as ``(n_freq, 1)`` -- is refused. A map built
    for 60-85 MHz and evaluated on a 100-125 MHz grid of the SAME length is
    not, and cannot be under ``jit`` -- the values are traced, only the shape is
    static. That failure returns a smooth, plausible, wrong temperature, so
    ``freq`` is stored to give it a name and a place for a config layer to
    check it before tracing begins.

    Attributes:
        maps: ``(n_freq, n_pix)`` brightness temperatures [K] -- a
            differentiable leaf, so a sky can be inferred rather than assumed.
        freq: ``(n_freq,)`` the frequency grid the maps were built on [Hz].
    """

    maps: jax.Array
    freq: jax.Array

    def __check_init__(self):
        if jnp.ndim(self.maps) != 2:
            raise StateValidationError(
                f"maps must be (n_freq, n_pix), got ndim {jnp.ndim(self.maps)} "
                f"with shape {jnp.shape(self.maps)}. A single map is (1, n_pix), "
                "not (n_pix,) -- the frequency axis is not optional, because "
                "MapSky's whole contract is which grid the maps belong to."
            )
        if jnp.ndim(self.freq) != 1:
            raise StateValidationError(
                f"freq must be (n_freq,), got ndim {jnp.ndim(self.freq)} with "
                f"shape {jnp.shape(self.freq)}."
            )
        if jnp.shape(self.maps)[0] != jnp.shape(self.freq)[0]:
            raise StateValidationError(
                f"maps and freq disagree on the number of channels: maps is "
                f"{jnp.shape(self.maps)} and freq is {jnp.shape(self.freq)}. "
                "The maps' first axis IS the frequency axis."
            )

    def __call__(self, freq: jax.Array) -> jax.Array:
        if jnp.shape(freq) != jnp.shape(self.freq):
            raise StateValidationError(
                f"MapSky was built on a grid of shape {jnp.shape(self.freq)} "
                f"and asked for shape {jnp.shape(freq)}. The maps are not "
                "interpolated -- they are returned as stored -- so a grid of "
                "another shape is a modelling error, not a resampling "
                "request. Rebuild the maps on the grid you mean to observe on."
            )
        return self.maps
