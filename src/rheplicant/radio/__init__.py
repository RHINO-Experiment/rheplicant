"""Generic single-antenna radio telescope operators.

Organized by the element taxonomy (see ``DESIGN.md``). The operator packages:

- ``rheplicant.radio.sky`` — astrophysical: 21 cm global signal, foregrounds,
  point sources (+ the simplest uniform ``SkyOperator``), and the sky engines
  that convolve a model with a beam.
- ``rheplicant.radio.environment`` — ionosphere, atmospheric emission, ground
  pickup, RFI.
- ``rheplicant.radio.instrument`` — beam, horizon split, antenna ohmic loss,
  switched calibration loads, noise-wave / reflection terms, bandpass, gain,
  CW calibration tone, thermal noise, self-generated EMI, digitisation.
- ``rheplicant.radio.backend`` — flagging, averaging.
- ``rheplicant.radio.filters`` — sidereal / sky-space / Fourier linear filters.
- ``rheplicant.radio.t_sys`` — a separable ``(nu, t)`` basis as an effective
  temperature, which is how a smooth ``T_ant`` reaches the antenna sum.
- ``rheplicant.radio.surrogate`` — a learned stand-in for an expensive stage.

and the supporting modules, which ship no operators:

- ``rheplicant.radio.graph`` — the canonical signal-path graph and
  :func:`~rheplicant.radio.graph.assemble`.
- ``rheplicant.radio.protection`` — the ``aux`` contract that keeps a known
  calibrator (the CW tone) out of the flags it would otherwise trip.
- ``rheplicant.radio.touchstone`` / ``rheplicant.radio.rhino`` — readers for
  measured S-parameters and for RHINO's HDF5 observations.
- ``rheplicant.radio.beams`` — readers for CST far-field exports.

A forward model composes them with the two core combinators, following the
canonical signal-path graph (``rheplicant.radio.graph``; RFI enters as a
pre-beam field, ground pickup as a post-beam effective temperature)::

    astro = Pipeline(SumOperator(signal, foregrounds, point_sources), ionosphere)
    field = Pipeline(SumOperator(astro, rfi_field), beam)
    t_ant = SumOperator(field, ground_pickup, atmosphere)
    twin  = Pipeline(t_ant, noise_wave, cw_tone, bandpass, gain,
                     noise, emi, adc, flagging, averaging)

or by just providing the operators::

    twin = assemble(signal, foregrounds, point_sources, ionosphere, rfi_field,
                    beam, ground_pickup, atmosphere, noise_wave, cw_tone,
                    bandpass, gain, noise, emi, adc, flagging, averaging)

Both spellings compile to the same composition, and both now check the ordering
an operator declares (``must_precede``). They are still not interchangeable,
and the difference is what each one is *able* to ask:

* ``assemble`` tests **reachability** on the template — does this stage's
  output actually reach the node it must precede — and additionally refuses a
  ``must_precede`` naming a node the template does not have, because an
  unenforceable declaration is prose in a ClassVar. It raises ``AssemblyError``
  and can name the node the operator landed on.
* ``Pipeline`` tests **order among its own** ``names``: if a named stage is
  present in this sequence, it must come after the declaring stage. It raises
  ``PipelineError``. A stage that is not present is not a violation, it cannot
  see into a nested composite, and — having no node list — it cannot tell a
  typo from a legitimately absent stage.

Both errors derive from ``DirtError`` and from ``ValueError``, so one ``except``
catches either.

The practical consequence, because it is easy to be caught by: the sequence
route binds through stage NAMES, so auto-derived names bind a constraint only
where they happen to coincide with node ids. ``GainOperator`` auto-names to
``"gain"``, so the tone's ``"gain"`` half binds; ``ReceiverOperator``
auto-names to ``"receiver"``, **not** ``"bandpass"``, so that half does not
bind unless the caller passes ``names=``. Measured::

    Pipeline(receiver, tone)                              # accepted
    Pipeline(receiver, tone, names=("bandpass", "cw_tone"))
    # PipelineError: ... 'bandpass' ... runs BEFORE this one

The example above is in exactly that situation — its ``cw_tone`` precedes both
stages, so nothing is wrong with it, but only the ``gain`` half of the
constraint is actually being checked. Pass ``names=`` matching the graph's node
ids, or use ``assemble``, which has no such gap because placement is on the
node itself. See D27.

Physics is deliberately placeholder where the operator's own docstring says so
— 15 of the 28 concrete operator classes here, a count pinned by
``tests/radio/test_placeholder_census.py`` so that it moves when the physics
does. Real physics replaces function bodies, never interfaces: the sky
engines, the horizon split, the antenna's ohmic loss, the noise-wave
reflection terms, the CW tone and the separable-basis antenna temperature all
arrived that way, ported from limTOD (single-antenna TOD simulation, itself
rewritten in JAX + Equinox) and the related family — see ``DESIGN.md`` for the
roadmap and the README's Status section for the current split.
Instrument-specific parameters (e.g. RHINO's) enter as concrete operator
configurations, never as framework assumptions.
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
    BeamSpillOperator,
    CalLoadOperator,
    CWCalibrationOperator,
    EMIOperator,
    GainOperator,
    NoiseOperator,
    NoiseWaveOperator,
    ReceiverOperator,
    unit_mean_bandpass,
    unit_mean_free,
)
from rheplicant.radio.protection import (
    PROTECTED_KEY,
    protect,
    reduce_protection,
    unflag_protected,
)
from rheplicant.radio.rhino import (
    RhinoObservation,
    cal_load_operators,
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
    MapSky,
    MatrixProjector,
    PointSourceOperator,
    PowerLawSkyModel,
    SkyOperator,
    SkySourceOperator,
    UniformSkyModel,
)
from rheplicant.radio.surrogate import NeuralOperator
from rheplicant.radio.t_sys import BasisTemperatureOperator
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
    "BasisTemperatureOperator",
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
    "MapSky",
    "MatrixProjector",
    "MomentRFIFlaggingOperator",
    "NeuralOperator",
    "NoiseOperator",
    "NoiseWaveOperator",
    "PROTECTED_KEY",
    "PointSourceOperator",
    "PowerLawSkyModel",
    "RFIOperator",
    "ReceiverOperator",
    "SiderealFilter",
    "SkyOperator",
    "SkySourceOperator",
    "SkySpaceFilter",
    "UniformSkyModel",
    "protect",
    "reduce_protection",
    "unflag_protected",
    "unit_mean_bandpass",
    "unit_mean_free",
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
    "cal_load_operators",
    "read_rhino_observation",
    "rhino_to_state",
]

from rheplicant.radio.graph import _validate_registrations as _v  # noqa: E402

_v()
del _v
