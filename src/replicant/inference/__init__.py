"""Inference / calibration layer: treats a Pipeline as data, never lives inside it."""

from replicant.inference.calibrate import AdamCalibrator, GradientCalibrator
from replicant.inference.forward import build_forward_fn
from replicant.inference.likelihood import (
    GaussianLikelihood,
    Likelihood,
    MaskedGaussianLikelihood,
    mean_squared_error,
)
from replicant.inference.numpyro_bridge import (
    predict_from_samples,
    prior_template,
    set_prior,
    to_numpyro_model,
)
from replicant.inference.uncertainty import (
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)

__all__ = [
    "AdamCalibrator",
    "GaussianLikelihood",
    "GradientCalibrator",
    "Likelihood",
    "MaskedGaussianLikelihood",
    "build_forward_fn",
    "fisher_information",
    "mean_squared_error",
    "parameter_covariance",
    "predict_from_samples",
    "prior_template",
    "propagate_covariance",
    "push_forward",
    "set_prior",
    "to_numpyro_model",
]
