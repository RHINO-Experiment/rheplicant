# API reference

Generated from the source docstrings. Layering rule: `rheplicant.core` is
domain-agnostic; `rheplicant.radio` and `rheplicant.inference` build on it; and
`rheplicant.config` sits above all three, importing them and imported by none of
them.

For the prose behind these signatures: [the guided tour](tour.md) for the shape
of a twin, [the operator catalog](operators.md) for what lives at each graph
node, [inferring anything](inference.md) for the parameter-space machinery,
[contracts between stages](contracts.md) for the refusals, [values in a
config document](config-values.md) for the grammar the config layer resolves,
and [resources and paths](config-resources.md) for the loader built on top of
it.

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

.. automodule:: rheplicant.core.contract
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.graph
   :members:
   :show-inheritance:

.. automodule:: rheplicant.core.basis
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

.. automodule:: rheplicant.radio.environment.atmosphere
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

.. automodule:: rheplicant.radio.t_sys
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.protection
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

## Ingestion

Readers for the two file formats RHINO records: the spectrometer's HDF5
observations, and the Touchstone `.sNp` sweeps that supply the reflection
coefficients the noise-wave model consumes.

```{eval-rst}
.. automodule:: rheplicant.radio.rhino
   :members:
   :show-inheritance:

.. automodule:: rheplicant.radio.touchstone
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

.. automodule:: rheplicant.inference.identifiability
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

.. automodule:: rheplicant.inference.gls
   :members:

.. automodule:: rheplicant.inference.likelihood
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.numpyro_bridge
   :members:

.. automodule:: rheplicant.inference.npe
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.uncertainty
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.sensitivity
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.priors
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: rheplicant.inference.plan
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.engines
   :members:
```

```{eval-rst}
.. automodule:: rheplicant.inference.sqrtinfo
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.factorize
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.compressed
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.compress
   :members:

.. automodule:: rheplicant.inference.reduced_basis
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.memory
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.chain
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.diagnostics
   :members:
   :show-inheritance:

.. automodule:: rheplicant.inference.archive
   :members:
```

## rheplicant.config

The value grammar: what a fragment of a document may say, and how the resolved
value reaches a field. [Values in a config document](config-values.md) is the
prose; these are the signatures.

```{eval-rst}
.. automodule:: rheplicant.config.errors
   :members:
   :show-inheritance:

.. automodule:: rheplicant.config.registry
   :members:
   :show-inheritance:

.. automodule:: rheplicant.config.units
   :members:
   :show-inheritance:

.. automodule:: rheplicant.config.symbols
   :members:
   :show-inheritance:

.. automodule:: rheplicant.config.context
   :members:
   :show-inheritance:

.. automodule:: rheplicant.config.values
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: rheplicant.config.arrays
   :members:

.. automodule:: rheplicant.config.draws
   :members:

.. automodule:: rheplicant.config.files
   :members:

.. automodule:: rheplicant.config.derive
   :members:

.. automodule:: rheplicant.config.refs
   :members:

.. automodule:: rheplicant.config.hatch
   :members:

.. automodule:: rheplicant.config.modifiers
   :members:

.. automodule:: rheplicant.config.delivery
   :members:
   :show-inheritance:
```

## rheplicant.config, continued: resources and paths

The resource loader and the path grammar, built on top of the value grammar
above. [Resources and paths in a config document](config-resources.md) is the
prose; these are the signatures.

```{eval-rst}
.. automodule:: rheplicant.config.paths
   :members:
   :show-inheritance:

.. automodule:: rheplicant.config.resources
   :members:
   :show-inheritance:
```

```{eval-rst}
.. automodule:: rheplicant.config.kinds
   :members:

.. automodule:: rheplicant.config.kinds.arrays
   :members:

.. automodule:: rheplicant.config.kinds.bases
   :members:

.. automodule:: rheplicant.config.kinds.sky_models
   :members:

.. automodule:: rheplicant.config.kinds.beams
   :members:

.. automodule:: rheplicant.config.kinds.projectors
   :members:

.. automodule:: rheplicant.config.kinds.s_params
   :members:
```
