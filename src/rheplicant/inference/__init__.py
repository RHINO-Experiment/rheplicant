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
    gcr_sample,
    linear_operator,
    wiener_solve,
)
from rheplicant.inference.numpyro_bridge import predict_from_samples, to_numpyro_model
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace
from rheplicant.inference.uncertainty import (
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)

__all__ = [
    "AdamCalibrator",
    "Bind",
    "GaussianLikelihood",
    "GradientCalibrator",
    "Latent",
    "LinearBlock",
    "Likelihood",
    "MaskedGaussianLikelihood",
    "ParameterSpace",
    "build_forward_fn",
    "check_linearity",
    "fisher_information",
    "gcr_sample",
    "linear_operator",
    "mean_squared_error",
    "parameter_covariance",
    "predict_from_samples",
    "propagate_covariance",
    "push_forward",
    "to_numpyro_model",
    "wiener_solve",
]
