"""Astrophysical sky components (elements taxonomy: "Astrophysical").

Compose additively with :class:`~rheplicant.core.combinators.SumOperator`::

    sky = SumOperator(
        GlobalSignalOperator(...), ForegroundOperator(...), PointSourceOperator(...),
        names=("signal", "foregrounds", "point_sources"),
    )

``SkyOperator`` (uniform brightness) remains as the simplest single-component
placeholder for quick tests and demos.
"""

from rheplicant.radio.sky.driftscan import DriftScanProjector
from rheplicant.radio.sky.foregrounds import ForegroundOperator
from rheplicant.radio.sky.general_pointing import GeneralPointingProjector
from rheplicant.radio.sky.global_signal import GlobalSignalOperator
from rheplicant.radio.sky.model import AbstractSkyModel, PowerLawSkyModel, UniformSkyModel
from rheplicant.radio.sky.point_sources import PointSourceOperator
from rheplicant.radio.sky.projection import (
    AbstractSkyProjector,
    LimTODProjector,
    MatrixProjector,
    MModeProjector,
)
from rheplicant.radio.sky.source import SkySourceOperator
from rheplicant.radio.sky.uniform import SkyOperator

__all__ = [
    "AbstractSkyModel",
    "AbstractSkyProjector",
    "DriftScanProjector",
    "ForegroundOperator",
    "GlobalSignalOperator",
    "LimTODProjector",
    "MModeProjector",
    "MatrixProjector",
    "GeneralPointingProjector",
    "PointSourceOperator",
    "PowerLawSkyModel",
    "SkyOperator",
    "SkySourceOperator",
    "UniformSkyModel",
]
