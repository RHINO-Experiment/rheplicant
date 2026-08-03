"""Generic single-antenna radio telescope operators (placeholder physics for now).

Organized by the element taxonomy (see ``DESIGN.md``):

- ``rheplicant.radio.sky`` — astrophysical: 21 cm global signal, foregrounds,
  point sources (+ the simplest uniform ``SkyOperator``).
- ``rheplicant.radio.environment`` — ionosphere, atmospheric emission, ground
  pickup, RFI.
- ``rheplicant.radio.instrument`` — beam, noise-wave / reflection terms, bandpass,
  gain, CW calibration tone, thermal noise, self-generated EMI, digitisation.
- ``rheplicant.radio.backend`` — flagging, averaging.

A forward model composes them with the two core combinators, following the
canonical signal-path graph (``rheplicant.radio.graph``; RFI enters as a
pre-beam field, ground pickup as a post-beam effective temperature)::

    astro = Pipeline(SumOperator(signal, foregrounds, point_sources), ionosphere)
    field = Pipeline(SumOperator(astro, rfi_field), beam)
    t_ant = SumOperator(field, ground_pickup, atmosphere)
    twin  = Pipeline(t_ant, noise_wave, cw_tone, bandpass, gain,
                     noise, emi, adc, flagging, averaging)

or, equivalently, by just providing the operators::

    twin = assemble(signal, foregrounds, point_sources, ionosphere, rfi_field,
                    beam, ground_pickup, atmosphere, noise_wave, cw_tone,
                    bandpass, gain, noise, emi, adc, flagging, averaging)

Every operator is a trivial-but-runnable placeholder that establishes the
contract. The real physics will be ported from limTOD (single-antenna TOD
simulation, itself to be rewritten in JAX + Equinox) and the related family —
see DESIGN.md for the roadmap. Instrument-specific parameters (e.g. RHINO's)
enter later as concrete operator configurations, never as framework
assumptions.
"""

from rheplicant.radio.backend import BackendOperator, FlaggingOperator, MomentRFIFlaggingOperator
from rheplicant.radio.beams import (
    cst_beam_maps,
    cst_frequency_table,
    horizon_truncated_beam,
    read_cst_farfield,
)
from rheplicant.radio.environment import (
    AtmosphericEmissionOperator,
    GroundPickupOperator,
    IonosphereOperator,
    RFIOperator,
)
from rheplicant.radio.filters import (
    AbstractLinearFilter,
    FourierBandFilter,
    SiderealFilter,
    SkySpaceFilter,
)
from rheplicant.radio.instrument import (
    ADCOperator,
    AntennaLossOperator,
    ApplyCalibrationOperator,
    BeamOperator,
    BeamSpillOperator,
    CalLoadOperator,
    CWCalibrationOperator,
    EMIOperator,
    GainOperator,
    NoiseOperator,
    NoiseWaveOperator,
    ReceiverOperator,
)
from rheplicant.radio.rhino import (
    RhinoObservation,
    read_rhino_observation,
)
from rheplicant.radio.rhino import (
    to_state as rhino_to_state,
)
from rheplicant.radio.sky import (
    AbstractSkyModel,
    AbstractSkyProjector,
    DriftScanProjector,
    ForegroundOperator,
    GlobalSignalOperator,
    MatrixProjector,
    PointSourceOperator,
    PowerLawSkyModel,
    SkyOperator,
    SkySourceOperator,
    UniformSkyModel,
)
from rheplicant.radio.surrogate import NeuralOperator
from rheplicant.radio.touchstone import Touchstone, interpolate_onto, read_touchstone

__all__ = [
    "ADCOperator",
    "AbstractLinearFilter",
    "AbstractSkyModel",
    "AbstractSkyProjector",
    "AntennaLossOperator",
    "ApplyCalibrationOperator",
    "AtmosphericEmissionOperator",
    "BackendOperator",
    "BeamOperator",
    "BeamSpillOperator",
    "CWCalibrationOperator",
    "CalLoadOperator",
    "DriftScanProjector",
    "EMIOperator",
    "FlaggingOperator",
    "ForegroundOperator",
    "FourierBandFilter",
    "GainOperator",
    "GlobalSignalOperator",
    "GroundPickupOperator",
    "IonosphereOperator",
    "MatrixProjector",
    "MomentRFIFlaggingOperator",
    "NeuralOperator",
    "NoiseOperator",
    "NoiseWaveOperator",
    "PointSourceOperator",
    "PowerLawSkyModel",
    "RFIOperator",
    "ReceiverOperator",
    "SiderealFilter",
    "SkyOperator",
    "SkySourceOperator",
    "SkySpaceFilter",
    "UniformSkyModel",
]

from rheplicant.radio.graph import RADIO_GRAPH, assemble  # noqa: E402  (needs operators above)

__all__ += ["RADIO_GRAPH", "assemble"]
__all__ += [
    "cst_beam_maps",
    "cst_frequency_table",
    "horizon_truncated_beam",
    "read_cst_farfield",
]
__all__ += [
    "Touchstone",
    "interpolate_onto",
    "read_touchstone",
]
__all__ += [
    "RhinoObservation",
    "read_rhino_observation",
    "rhino_to_state",
]

from rheplicant.radio.graph import _validate_registrations as _v  # noqa: E402

_v()
del _v
