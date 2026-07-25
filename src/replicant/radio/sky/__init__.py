"""Astrophysical sky components (elements taxonomy: "Astrophysical").

Compose additively with :class:`~replicant.core.combinators.SumOperator`::

    sky = SumOperator(
        GlobalSignalOperator(...), ForegroundOperator(...), PointSourceOperator(...),
        names=("signal", "foregrounds", "point_sources"),
    )

``SkyOperator`` (uniform brightness) remains as the simplest single-component
placeholder for quick tests and demos.
"""

from replicant.radio.sky.foregrounds import ForegroundOperator
from replicant.radio.sky.global_signal import GlobalSignalOperator
from replicant.radio.sky.model import AbstractSkyModel, PowerLawSkyModel, UniformSkyModel
from replicant.radio.sky.native import NativeLimTODProjector
from replicant.radio.sky.point_sources import PointSourceOperator
from replicant.radio.sky.projection import (
    AbstractSkyProjector,
    LimTODProjector,
    MatrixProjector,
    MModeProjector,
)
from replicant.radio.sky.source import SkySourceOperator
from replicant.radio.sky.uniform import SkyOperator

__all__ = [
    "AbstractSkyModel",
    "AbstractSkyProjector",
    "ForegroundOperator",
    "GlobalSignalOperator",
    "LimTODProjector",
    "MModeProjector",
    "MatrixProjector",
    "NativeLimTODProjector",
    "PointSourceOperator",
    "PowerLawSkyModel",
    "SkyOperator",
    "SkySourceOperator",
    "UniformSkyModel",
]
