"""Inference / calibration layer: treats a Pipeline as data, never lives inside it."""

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.archive import load_memory, save_memory
from rheplicant.inference.calibrate import AdamCalibrator, GradientCalibrator
from rheplicant.inference.chain import (
    ChainMemory,
    HyperTransition,
    LinearGaussianTransition,
    chain_log_likelihood,
    chain_marginal,
    ornstein_uhlenbeck,
    smooth,
)
from rheplicant.inference.compress import (
    compress,
    compress_linear,
    compress_reduced_basis,
)
from rheplicant.inference.compressed import (
    COEFFICIENTS,
    REQUIRED_TERM_MEMBERS,
    CompressedLikelihood,
    QuadraticLikelihood,
    RawLikelihood,
    ReducedBasisLikelihood,
)
from rheplicant.inference.diagnostics import (
    EpochResidual,
    HeldOut,
    coherent_mode,
    epoch_residuals,
    held_out_z,
    shrinkage_power,
    shrinkage_report,
    systematic_floor,
)
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
from rheplicant.inference.priors import JeffreysPrior
from rheplicant.inference.reduced_basis import (
    FidelityReport,
    ReducedBasis,
    basis_fidelity,
    build_reduced_basis,
    numerical_rank,
    orthonormal_transform,
    orthonormalise,
    score_directions,
    select_greedy,
    select_svd,
)
from rheplicant.inference.sensitivity import (
    CRITERION_SHIFT,
    PriorSensitivityReport,
    prior_sensitivity,
)
from rheplicant.inference.sqrtinfo import SqrtInfo, marginalise, marginalise_arrays
from rheplicant.inference.uncertainty import (
    as_noise_model,
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)

__all__ = [
    "BROADCAST",
    "COEFFICIENTS",
    "CRITERION_SHIFT",
    "DEFAULT_RANK_RTOL",
    "MIN_DRAWS",
    "DISTRIBUTE",
    "REQUIRED_TERM_MEMBERS",
    "ParameterSpaceError",
    "AdamCalibrator",
    "AmbiguousFanWarning",
    "BayesMemory",
    "Bind",
    "Block",
    "ChainMemory",
    "CompressedLikelihood",
    "Draws",
    "EpochResidual",
    "Estimate",
    "Factorization",
    "FidelityReport",
    "FlaggedNoise",
    "GLSResult",
    "GaussianLikelihood",
    "GradientCalibrator",
    "HeldOut",
    "HomoscedasticNoise",
    "HyperTransition",
    "IdentifiabilityReport",
    "JeffreysPrior",
    "Latent",
    "LinearBlock",
    "LinearGaussianTransition",
    "Likelihood",
    "MaskedGaussianLikelihood",
    "NeuralPosterior",
    "NoiseModel",
    "NoiseModelLikelihood",
    "ParameterSpace",
    "PlanDiagnostics",
    "PlanResult",
    "PriorSensitivityReport",
    "QuadraticLikelihood",
    "RadiometerNoise",
    "RawLikelihood",
    "ReducedBasis",
    "ReducedBasisLikelihood",
    "SamplingPlan",
    "SqrtInfo",
    "as_noise_model",
    "basis_fidelity",
    "build_forward_fn",
    "build_reduced_basis",
    "chain_log_likelihood",
    "chain_marginal",
    "check_linearity",
    "check_observed_shape",
    "coherent_mode",
    "compress",
    "compress_linear",
    "compress_reduced_basis",
    "condition_estimate",
    "epoch_residuals",
    "fisher_information",
    "gcr_sample",
    "held_out_z",
    "identifiability",
    "init_to_declared",
    "inverse_variance",
    "iterative_gls",
    "linear_operator",
    "load_memory",
    "marginalise",
    "marginalise_arrays",
    "mean_squared_error",
    "numerical_rank",
    "ornstein_uhlenbeck",
    "orthonormal_transform",
    "orthonormalise",
    "parameter_covariance",
    "predict_from_samples",
    "prior_sensitivity",
    "propagate_covariance",
    "refuse_stochastic_stages",
    "save_memory",
    "score_directions",
    "select_greedy",
    "select_svd",
    "shrinkage_power",
    "shrinkage_report",
    "simulate_pairs",
    "smooth",
    "split_rhat",
    "systematic_floor",
    "push_forward",
    "to_numpyro_model",
    "train_posterior",
    "wiener_solve",
]
