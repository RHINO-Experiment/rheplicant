"""Inference / calibration layer: treats a Pipeline as data, never lives inside it."""

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.archive import load_memory, save_memory
from rheplicant.inference.calibrate import AdamCalibrator, GradientCalibrator
from rheplicant.inference.compress import compress_linear
from rheplicant.inference.compressed import CompressedLikelihood, QuadraticLikelihood
from rheplicant.inference.factorize import Factorization
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
from rheplicant.inference.memory import BayesMemory
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
    BROADCAST,
    DISTRIBUTE,
    AmbiguousFanWarning,
    Bind,
    Latent,
    ParameterSpace,
    refuse_stochastic_stages,
)
from rheplicant.inference.plan import (
    MIN_DRAWS,
    Block,
    Draws,
    Estimate,
    PlanDiagnostics,
    PlanResult,
    SamplingPlan,
    split_rhat,
)
from rheplicant.inference.sqrtinfo import SqrtInfo
from rheplicant.inference.uncertainty import (
    as_noise_model,
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)

__all__ = [
    "BROADCAST",
    "DEFAULT_RANK_RTOL",
    "MIN_DRAWS",
    "DISTRIBUTE",
    "ParameterSpaceError",
    "AdamCalibrator",
    "AmbiguousFanWarning",
    "BayesMemory",
    "Bind",
    "Block",
    "CompressedLikelihood",
    "Draws",
    "Estimate",
    "Factorization",
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
    "QuadraticLikelihood",
    "RadiometerNoise",
    "SamplingPlan",
    "SqrtInfo",
    "as_noise_model",
    "build_forward_fn",
    "check_linearity",
    "check_observed_shape",
    "compress_linear",
    "condition_estimate",
    "fisher_information",
    "gcr_sample",
    "identifiability",
    "init_to_declared",
    "inverse_variance",
    "iterative_gls",
    "linear_operator",
    "load_memory",
    "mean_squared_error",
    "parameter_covariance",
    "predict_from_samples",
    "propagate_covariance",
    "refuse_stochastic_stages",
    "save_memory",
    "simulate_pairs",
    "split_rhat",
    "push_forward",
    "to_numpyro_model",
    "train_posterior",
    "wiener_solve",
]
