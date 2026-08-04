"""Inference / calibration layer: treats a Pipeline as data, never lives inside it."""

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.calibrate import AdamCalibrator, GradientCalibrator
from rheplicant.inference.forward import build_forward_fn
from rheplicant.inference.gls import GLSResult, iterative_gls
from rheplicant.inference.identifiability import (
    DEFAULT_RANK_RTOL,
    IdentifiabilityReport,
    identifiability,
)
from rheplicant.inference.likelihood import (
    GaussianLikelihood,
    Likelihood,
    MaskedGaussianLikelihood,
    check_observed_shape,
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
from rheplicant.inference.npe import (
    NeuralPosterior,
    simulate_pairs,
    train_posterior,
)
from rheplicant.inference.numpyro_bridge import (
    init_to_declared,
    predict_from_samples,
    to_numpyro_model,
)
from rheplicant.inference.parameters import (
    Bind,
    Latent,
    ParameterSpace,
    refuse_stochastic_stages,
)
from rheplicant.inference.plan import (
    Block,
    Draws,
    Estimate,
    PlanDiagnostics,
    PlanResult,
    SamplingPlan,
    split_rhat,
)
from rheplicant.inference.uncertainty import (
    as_noise_model,
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)

__all__ = [
    "DEFAULT_RANK_RTOL",
    "ParameterSpaceError",
    "AdamCalibrator",
    "Bind",
    "Block",
    "Draws",
    "Estimate",
    "FlaggedNoise",
    "GLSResult",
    "GaussianLikelihood",
    "GradientCalibrator",
    "HomoscedasticNoise",
    "IdentifiabilityReport",
    "Latent",
    "LinearBlock",
    "Likelihood",
    "MaskedGaussianLikelihood",
    "NeuralPosterior",
    "NoiseModel",
    "NoiseModelLikelihood",
    "ParameterSpace",
    "PlanDiagnostics",
    "PlanResult",
    "RadiometerNoise",
    "SamplingPlan",
    "as_noise_model",
    "build_forward_fn",
    "check_linearity",
    "check_observed_shape",
    "condition_estimate",
    "fisher_information",
    "gcr_sample",
    "identifiability",
    "init_to_declared",
    "inverse_variance",
    "iterative_gls",
    "linear_operator",
    "mean_squared_error",
    "parameter_covariance",
    "predict_from_samples",
    "propagate_covariance",
    "refuse_stochastic_stages",
    "simulate_pairs",
    "split_rhat",
    "push_forward",
    "to_numpyro_model",
    "train_posterior",
    "wiener_solve",
]
