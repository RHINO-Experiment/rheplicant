"""Instrumental response chain (elements taxonomy: "Instrumental").

Typical ordering in a forward model (RHINO paper Eq. 6:
``P_rec = g (T_ant + T_nw + T_cw) + T_n`` — the CW tone joins *before*
bandpass and gain so it tracks gain drift; the antenna/cal-load switch sits
between the sky-side temperatures and the receiver terms)::

    Beam -> (+ sky-side temperatures) -> [switch <- CalLoad] -> NoiseWave
         -> CWCalibration -> Receiver(bandpass) -> Gain -> Noise -> EMI -> ADC
"""

from replicant.radio.instrument.adc import ADCOperator
from replicant.radio.instrument.beam import BeamOperator
from replicant.radio.instrument.calibration import (
    ApplyCalibrationOperator,
    CalLoadOperator,
    CWCalibrationOperator,
)
from replicant.radio.instrument.emi import EMIOperator
from replicant.radio.instrument.gain import GainOperator
from replicant.radio.instrument.noise import NoiseOperator
from replicant.radio.instrument.noise_wave import NoiseWaveOperator
from replicant.radio.instrument.receiver import ReceiverOperator

__all__ = [
    "ADCOperator",
    "ApplyCalibrationOperator",
    "BeamOperator",
    "CalLoadOperator",
    "CWCalibrationOperator",
    "EMIOperator",
    "GainOperator",
    "NoiseOperator",
    "NoiseWaveOperator",
    "ReceiverOperator",
]
