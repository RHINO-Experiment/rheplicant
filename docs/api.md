# API reference

Generated from the source docstrings. Layering rule: `rheplicant.core` is
domain-agnostic; `rheplicant.radio` and `rheplicant.inference` build on it.

## rheplicant.core

```{eval-rst}
.. automodule:: rheplicant.core.state
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.coordinates
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.environment
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.frozen
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.operator
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.pipeline
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.combinators
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.graph
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.render
   :members:

.. automodule:: rheplicant.core.errors
   :members:
   :show-inheritance:
```

## rheplicant.radio

```{eval-rst}
.. automodule:: rheplicant.radio.graph
   :members:

.. automodule:: rheplicant.radio.sky.model
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.sky.projection
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.sky.general_pointing
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.sky.driftscan
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.sky.source
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.sky.uniform
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.sky.global_signal
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.sky.foregrounds
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.sky.point_sources
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.environment.ionosphere
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.environment.ground
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.environment.rfi
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.beams
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.beam
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.antenna_loss
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.beam_spill
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.receiver
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.noise_wave
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.calibration
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.gain
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.noise
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.emi
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.instrument.adc
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.backend.flagging
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.backend.averaging
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.filters.base
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.filters.sidereal
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.filters.skyspace
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.filters.fourier
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.surrogate
   :members:
   :show-inheritance:
```

## rheplicant.inference

```{eval-rst}
.. automodule:: rheplicant.inference.parameters
   :members:
```

```{eval-rst}
.. automodule:: rheplicant.inference.linear
   :members:

.. automodule:: rheplicant.inference.conditioning
   :members:
```

```{eval-rst}
.. automodule:: rheplicant.inference.forward
   :members:

.. automodule:: rheplicant.inference.calibrate
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.noise
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.likelihood
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.numpyro_bridge
   :members:

.. automodule:: rheplicant.inference.uncertainty
   :members:
   :show-inheritance:
```
