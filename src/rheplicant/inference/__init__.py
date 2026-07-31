"""Inference / calibration layer: treats a Pipeline as data, never lives inside it."""

from rheplicant.inference.calibrate import AdamCalibrator, GradientCalibrator
from rheplicant.inference.forward import build_forward_fn
from rheplicant.inference.likelihood import (
    GaussianLikelihood,
    Likelihood,
    MaskedGaussianLikelihood,
    mean_squared_error,
)
from rheplicant.inference.linear import (
    LinearBlock,
    check_linearity,
    condition_estimate,
    gcr_sample,
    linear_operator,
    wiener_solve,
)
from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    NoiseModel,
    NoiseModelLikelihood,
    RadiometerNoise,
    inverse_variance,
)
from rheplicant.inference.numpyro_bridge import predict_from_samples, to_numpyro_model
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace
from rheplicant.inference.uncertainty import (
    as_noise_model,
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)

__all__ = [
    "AdamCalibrator",
    "Bind",
    "FlaggedNoise",
    "GaussianLikelihood",
    "GradientCalibrator",
    "HomoscedasticNoise",
    "Latent",
    "LinearBlock",
    "Likelihood",
    "MaskedGaussianLikelihood",
    "NoiseModel",
    "NoiseModelLikelihood",
    "ParameterSpace",
    "RadiometerNoise",
    "as_noise_model",
    "build_forward_fn",
    "check_linearity",
    "condition_estimate",
    "fisher_information",
    "gcr_sample",
    "inverse_variance",
    "linear_operator",
    "mean_squared_error",
    "parameter_covariance",
    "predict_from_samples",
    "propagate_covariance",
    "push_forward",
    "to_numpyro_model",
    "wiener_solve",
]
