"""Environmental contributions (elements taxonomy: "Environmental").

Ionosphere distorts the astrophysical signal (apply after the sky sum);
ground pickup, atmospheric emission, and RFI are additive terrestrial/sky-side
contributions — ground pickup and atmospheric emission as branches of the
antenna-temperature :class:`~rheplicant.core.combinators.SumOperator`, RFI as a
pre-beam field.
"""

from rheplicant.radio.environment.atmosphere import AtmosphericEmissionOperator
from rheplicant.radio.environment.ground import GroundPickupOperator
from rheplicant.radio.environment.ionosphere import IonosphereOperator
from rheplicant.radio.environment.rfi import RFIOperator

__all__ = [
    "AtmosphericEmissionOperator",
    "GroundPickupOperator",
    "IonosphereOperator",
    "RFIOperator",
]
