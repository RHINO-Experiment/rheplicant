"""Environmental contributions (elements taxonomy: "Environmental").

Ionosphere distorts the astrophysical signal (apply after the sky sum);
ground pickup, atmospheric emission, and RFI are additive terrestrial/sky-side
contributions — ground pickup and atmospheric emission as branches of the
antenna-temperature :class:`~replicant.core.combinators.SumOperator`, RFI as a
pre-beam field.
"""

from replicant.radio.environment.atmosphere import AtmosphericEmissionOperator
from replicant.radio.environment.ground import GroundPickupOperator
from replicant.radio.environment.ionosphere import IonosphereOperator
from replicant.radio.environment.rfi import RFIOperator

__all__ = [
    "AtmosphericEmissionOperator",
    "GroundPickupOperator",
    "IonosphereOperator",
    "RFIOperator",
]
