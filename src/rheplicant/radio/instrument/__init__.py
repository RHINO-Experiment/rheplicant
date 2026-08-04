"""Instrumental response chain (elements taxonomy: "Instrumental").

Typical ordering in a forward model (RHINO paper Eq. 6:
``P_rec = g (T_ant + T_nw + T_cw) + T_n`` — the CW tone joins *before*
bandpass and gain so it tracks gain drift; the antenna/cal-load switch sits
between the sky-side temperatures and the receiver terms)::

    Beam -> BeamSpill -> (+ sky-side temperatures) -> AntennaLoss
         -> [switch <- CalLoad]
         -> NoiseWave -> CWCalibration -> Receiver(bandpass) -> Gain
         -> Noise -> EMI -> ADC
"""

from rheplicant.radio.instrument.adc import ADCOperator
from rheplicant.radio.instrument.antenna_loss import AntennaLossOperator
from rheplicant.radio.instrument.beam import BeamOperator
from rheplicant.radio.instrument.beam_spill import BeamSpillOperator
from rheplicant.radio.instrument.calibration import (
    ApplyCalibrationOperator,
    CalLoadOperator,
    CWCalibrationOperator,
)
from rheplicant.radio.instrument.emi import EMIOperator
from rheplicant.radio.instrument.gain import GainOperator
from rheplicant.radio.instrument.noise import NoiseOperator
from rheplicant.radio.instrument.noise_wave import NoiseWaveOperator
from rheplicant.radio.instrument.receiver import (
    ReceiverOperator,
    unit_mean_bandpass,
    unit_mean_free,
)

__all__ = [
    "ADCOperator",
    "AntennaLossOperator",
    "ApplyCalibrationOperator",
    "BeamOperator",
    "BeamSpillOperator",
    "CalLoadOperator",
    "CWCalibrationOperator",
    "EMIOperator",
    "GainOperator",
    "NoiseOperator",
    "NoiseWaveOperator",
    "ReceiverOperator",
    "unit_mean_bandpass",
    "unit_mean_free",
]
