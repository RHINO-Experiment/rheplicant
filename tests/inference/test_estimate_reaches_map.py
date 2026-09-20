"""A point estimate that reports ``converged=True`` is the exact MAP.

``SamplingPlan.estimate`` describes its answer as a fixed point of the whole
model, and ``PlanDiagnostics.converged`` is documented as "False means the
answer is not what it looks like". A gradient block's estimate broke that
(T-002 A5-2, measured by an independent verifier): its Adam restarted every
sweep, and the first bias-corrected step is ``learning_rate * max|init|``
whatever the gradient, so the sweep map had a fixed point about 0.19 of that
step from the optimum. A spectral index over 4096 channels at 0.01 K landed
-2881 posterior sigma from its MAP, reported converged. And the stop rule
(A5-1) accepted any sweep whose JOINT chi-squared did not fall, so a rise
counted as convergence. A prior raises chi-squared while the posterior
improves, and runs certified convergence 1 to 15 posterior sigma from the
exact MAP, including plans of conjugate blocks with no gradient block.

Every case below asserts the same contract. A run that returns has
``converged=True`` and lands within :data:`MAHALANOBIS_MAX` of the exact MAP,
measured in posterior sigma with the exact posterior precision; a run that
cannot must raise the "did not converge" refusal. Cases flagged
``must_converge`` also require the return, so the file cannot pass by
refusing everything.

The exact MAP is closed form for the two linear models and a float64 Newton solve
for the power law. The module runs in float64 because 0.1 sigma of the
spectral index at 4096 channels and 0.01 K is 1.7e-7, finer than float32
resolves a number near 2.55.
"""

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Block,
    HomoscedasticNoise,
    Latent,
    ParameterSpace,
    SamplingPlan,
)
from rheplicant.inference.plan import (
    DEFAULT_CHI2_TOL,
    EARLIEST_CONVERGED_SWEEP,
    MIN_SWEEPS,
)
from rheplicant.radio import ForegroundOperator, SkyOperator

dist = pytest.importorskip("numpyro.distributions", reason="numpyro not installed")

#: The contract: a converged estimate is this close to the exact MAP, in
#: posterior sigma (Mahalanobis distance under the exact posterior precision).
MAHALANOBIS_MAX = 0.1


@pytest.fixture(scope="module", autouse=True)
def _float64():
    """Module-scoped and restored: the flag is process-global."""
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


def _estimate_or_refusal(plan, pipeline, state, observed, noise, **options):
    """The run's estimate, or ``None`` when it refused with "did not converge"."""
    try:
        estimate = plan.estimate(pipeline, state, observed, noise=noise, **options)
    except ParameterSpaceError as refusal:
        assert "did not converge" in str(refusal), refusal
        return None
    assert estimate.diagnostics.converged is True, estimate.diagnostics
    return estimate


def _mahalanobis(got, exact, precision) -> float:
    residual = np.asarray(got, dtype=np.float64) - np.asarray(exact, dtype=np.float64)
    return float(np.sqrt(residual @ precision @ residual))


def _check(estimate, got, exact, precision, must_converge, label):
    if estimate is None:
        assert not must_converge, f"{label}: refused, and this case must converge"
        return
    distance = _mahalanobis(got, exact, precision)
    assert distance < MAHALANOBIS_MAX, (
        f"{label}: converged=True after {estimate.diagnostics.sweeps} sweeps, "
        f"{distance:.3g} posterior sigma from the exact MAP (got {got}, MAP "
        f"{exact}). A converged estimate must be within {MAHALANOBIS_MAX}."
    )


# ----------------------------------------------- (i) one gradient block, 1-D --

SKY_SHAPE = (4, 8)
SKY_SIGMA = 1.0


def _grid_state(n_time: int, n_freq: int) -> State:
    return State(
        coords=Coordinates(time=jnp.arange(float(n_time)), freq=jnp.linspace(60e6, 85e6, n_freq))
    )


def _sky_case(ratio: float):
    """``data = a + noise`` with ``a ~ N(0, tau)``, ``tau = sigma_MLE / ratio``.

    Declared without ``linear=True``, so the plan derives a GRADIENT block. The
    seed puts the MLE 1.37 sigma_MLE below the prior mean, which is the
    verifier's case: at ratio 30, started at the MLE, the run used to report
    converged 11 posterior sigma from the MAP.
    """
    rng = np.random.default_rng(5)
    observed = SKY_SIGMA * rng.standard_normal(SKY_SHAPE)
    n = observed.size
    mle = float(observed.mean())
    tau = SKY_SIGMA / np.sqrt(n) / ratio
    precision = n / SKY_SIGMA**2 + 1.0 / tau**2
    exact = float(observed.sum()) / SKY_SIGMA**2 / precision
    return observed, mle, tau, exact, np.array([[precision]])


@pytest.mark.parametrize("ratio", [0.1, 3.0, 30.0])
@pytest.mark.parametrize("start", ["mle", "prior_mean", "far_side"])
def test_one_gradient_block_reaches_the_analytic_map(ratio, start):
    observed, mle, tau, exact, precision = _sky_case(ratio)
    init = {"mle": mle, "prior_mean": 0.0, "far_side": -mle}[start]
    space = ParameterSpace(
        latents=[Latent("a", init=jnp.array(init), prior=dist.Normal(0.0, tau))],
        bindings=[Bind("a", into=lambda p: p["sky"].amplitude)],
    )
    plan = SamplingPlan(space, Block("a"))
    pipeline = Pipeline(SkyOperator(amplitude=jnp.array(0.0)), names=("sky",))
    estimate = _estimate_or_refusal(
        plan,
        pipeline,
        _grid_state(*SKY_SHAPE),
        jnp.asarray(observed),
        HomoscedasticNoise(sigma=jnp.array(SKY_SIGMA)),
    )
    got = [float(estimate.values["a"])] if estimate is not None else None
    _check(estimate, got, [exact], precision, True, f"ratio {ratio}, start {start}")


def test_a_quadratic_gradient_block_stops_at_exactly_min_sweeps():
    """One block on a quadratic potential is solved in the first sweep.

    Adam's steps leave it short of the optimum and one Newton step lands on
    it. The verdict needs the objective settled over two consecutive sweeps
    and is consulted from ``MIN_SWEEPS`` on, so the run must return at
    exactly ``MIN_SWEEPS``, at the MAP.
    """
    observed, mle, tau, exact, precision = _sky_case(30.0)
    space = ParameterSpace(
        latents=[Latent("a", init=jnp.array(mle), prior=dist.Normal(0.0, tau))],
        bindings=[Bind("a", into=lambda p: p["sky"].amplitude)],
    )
    pipeline = Pipeline(SkyOperator(amplitude=jnp.array(0.0)), names=("sky",))
    estimate = SamplingPlan(space, Block("a")).estimate(
        pipeline,
        _grid_state(*SKY_SHAPE),
        jnp.asarray(observed),
        noise=HomoscedasticNoise(sigma=jnp.array(SKY_SIGMA)),
    )
    assert estimate.diagnostics.converged is True
    assert estimate.diagnostics.sweeps == MIN_SWEEPS
    got = [float(estimate.values["a"])]
    assert _mahalanobis(got, [exact], precision) < MAHALANOBIS_MAX


# ---------------------------------- (ii) two collinear CONJUGATE blocks only --

TEMPLATE_SHAPE = (4, 8)
TEMPLATE_SIGMA = 1.0
TEMPLATE_TRUTH = np.array([3.0, 2.0])


class _TwoTemplates(AbstractOperator):
    """``data[t, f] = a + b * (1 + eps * x_f)``: two templates, nearly collinear."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    a: jax.Array
    b: jax.Array
    eps: float = eqx.field(static=True)

    def __call__(self, state: State) -> State:
        n_time, n_freq = state.coords.time.shape[0], state.coords.freq.shape[0]
        x = jnp.linspace(-1.0, 1.0, n_freq)
        profile = self.a + self.b * (1.0 + self.eps * x)
        return state.with_data(jnp.broadcast_to(profile[None, :], (n_time, n_freq)))


def _template_design(eps: float, shape) -> np.ndarray:
    n_time, n_freq = shape
    x = np.linspace(-1.0, 1.0, n_freq)
    return np.tile(np.stack([np.ones(n_freq), 1.0 + eps * x], axis=1), (n_time, 1))


def _template_map(observed, eps: float, tau: float):
    """``(mle, map, precision)`` of ``observed`` exactly as given, in float64."""
    design = _template_design(eps, np.shape(observed))
    fisher = design.T @ design / TEMPLATE_SIGMA**2
    projected = design.T @ np.asarray(observed, np.float64).ravel() / TEMPLATE_SIGMA**2
    precision = fisher + np.eye(2) / tau**2
    return (np.linalg.solve(fisher, projected), np.linalg.solve(precision, projected), precision)


def _template_case(eps: float, tau: float, seed: int, shape=TEMPLATE_SHAPE):
    design = _template_design(eps, shape)
    noise = np.random.default_rng(seed).standard_normal(shape)
    observed = (design @ TEMPLATE_TRUTH).reshape(shape) + TEMPLATE_SIGMA * noise
    return (observed, *_template_map(observed, eps, tau))


#: Enough sweeps for every cell but the slowest pair to converge. At
#: ``eps = 0.01, tau = 3`` the two blocks' posterior correlation is 0.9965, a
#: sweep shrinks the error by 0.993, and a verdict takes 577 to 1166 sweeps
#: (measured), so those cells must refuse here. Before the repair seed 12 and
#: 13 of that pair reported converged at 34 and 40 sweeps, 12 and 15 sigma
#: from the MAP.
TEMPLATE_MAX_ITER = 300


@pytest.mark.parametrize("eps", [0.2, 0.01])
@pytest.mark.parametrize("tau", [1.0, 3.0])
@pytest.mark.parametrize("seed", [11, 12, 13])
@pytest.mark.parametrize("start", ["mle", "reflected", "map"])
def test_two_collinear_conjugate_blocks(eps, tau, seed, start):
    """No gradient block anywhere, so no Adam: the stop rule alone decides.

    ``reflected`` starts on the other side of the MAP from the MLE. ``map``
    starts at the exact answer, where every sweep reproduces it, so the run
    must return at exactly ``MIN_SWEEPS``.
    """
    slowest = eps == 0.01 and tau == 3.0
    observed, mle, exact, precision = _template_case(eps, tau, seed)
    init = {"mle": mle, "reflected": 2.0 * exact - mle, "map": exact}[start]
    space = ParameterSpace(
        latents=[
            Latent("a", init=jnp.array(init[0]), prior=dist.Normal(0.0, tau), linear=True),
            Latent("b", init=jnp.array(init[1]), prior=dist.Normal(0.0, tau), linear=True),
        ],
        bindings=[
            Bind("a", into=lambda p: p["tt"].a),
            Bind("b", into=lambda p: p["tt"].b),
        ],
    )
    pipeline = Pipeline(_TwoTemplates(a=jnp.array(0.0), b=jnp.array(0.0), eps=eps), names=("tt",))
    estimate = _estimate_or_refusal(
        SamplingPlan(space, Block("a"), Block("b")),
        pipeline,
        _grid_state(*TEMPLATE_SHAPE),
        jnp.asarray(observed),
        HomoscedasticNoise(sigma=jnp.array(TEMPLATE_SIGMA)),
        max_iter=TEMPLATE_MAX_ITER,
    )
    got = None if estimate is None else [float(estimate.values["a"]), float(estimate.values["b"])]
    label = f"eps {eps}, tau {tau}, seed {seed}, start {start}"
    _check(estimate, got, exact, precision, start == "map" or not slowest, label)
    if start == "map":
        assert estimate.diagnostics.sweeps == MIN_SWEEPS, label


def _template_plan(eps, tau, init, dtype=None):
    """Two one-latent conjugate blocks over :class:`_TwoTemplates`."""

    def array(value):
        return jnp.array(value) if dtype is None else jnp.array(value, dtype)

    space = ParameterSpace(
        latents=[
            Latent("a", init=array(init[0]), prior=dist.Normal(0.0, tau), linear=True),
            Latent("b", init=array(init[1]), prior=dist.Normal(0.0, tau), linear=True),
        ],
        bindings=[
            Bind("a", into=lambda p: p["tt"].a),
            Bind("b", into=lambda p: p["tt"].b),
        ],
    )
    pipeline = Pipeline(_TwoTemplates(a=array(0.0), b=array(0.0), eps=eps), names=("tt",))
    return SamplingPlan(space, Block("a"), Block("b")), pipeline


def test_the_earliest_stop_is_sweep_three_whatever_min_sweeps_says():
    """Started AT the MAP, no sweep moves the objective beyond its resolution.

    The pre-screen passes from sweep 2 on (a run that never moved uses a
    contraction of 0) and the decrement there is zero, but the change test's
    first change is sweep 2 against sweep 1, so the earliest stop is sweep 3
    for ``min_sweeps`` of 1, 2 and 3; ``min_sweeps`` above that is the floor;
    and a cap of two sweeps can never converge. Moved here from
    ``test_plan.py``'s float32 basis model, whose start at the answer is not
    a fixed point in float32: its conjugate solves move the objective by up
    to 2.3e-3 nats a sweep there, which the pre-screen reads as the rises
    they are.
    """
    observed, _, exact, _ = _template_case(0.2, 3.0, 11)
    plan, pipeline = _template_plan(0.2, 3.0, exact)
    common = {"noise": HomoscedasticNoise(sigma=jnp.array(TEMPLATE_SIGMA)), "max_iter": 30}
    state, data = _grid_state(*TEMPLATE_SHAPE), jnp.asarray(observed)
    for min_sweeps in (1, 2, 3):
        early = plan.estimate(pipeline, state, data, min_sweeps=min_sweeps, **common)
        assert early.diagnostics.converged is True
        assert early.diagnostics.sweeps == EARLIEST_CONVERGED_SWEEP == 3, min_sweeps
    floored = plan.estimate(pipeline, state, data, min_sweeps=8, **common)
    assert floored.diagnostics.sweeps == 8
    with pytest.raises(ParameterSpaceError) as capped:
        plan.estimate(pipeline, state, data, min_sweeps=1, **{**common, "max_iter": 2})
    assert "did not converge" in str(capped.value)


# ----------------------------------------- (iii) the certificate at large N --

#: ``(n_time, n_freq)``: 1e4 and 1e5 samples. The review's measurements went
#: to 1e6 (see the report); 1e5 keeps this file inside its runtime budget.
LARGE_N = [(100, 100), (250, 400)]


@pytest.mark.parametrize("shape", LARGE_N, ids=["N=1e4", "N=1e5"])
@pytest.mark.parametrize("eps", [1.0, 0.5, 0.2], ids=["r=0.86", "r=0.96", "r=0.993"])
def test_a_large_n_collinear_pair_is_certified_within_a_tenth_of_a_sigma(shape, eps):
    """The T-002 review's HIGH: the relative change test certified a distance
    that grows as ``sqrt(N)``. Measured by the reviewer on this model, float64,
    ``converged=True`` at 0.059 sigma (``N = 1e4``, r = 0.993) and 0.60 sigma
    (``N = 1e6``, r = 0.993); at ``N = 1e5`` it would be ~0.2. The Newton
    decrement is a Mahalanobis distance, so the same 0.1 sigma holds at every
    ``N``.

    Started 20 posterior sigma off along both axes. Every cell must converge
    in float64, within 0.1 sigma, and report a distance bound that covers it:
    the model is linear, so the objective is exactly quadratic and the
    decrement IS the distance to the MAP.
    """
    observed, _, exact, precision = _template_case(eps, 3.0, 21, shape=shape)
    sd = np.sqrt(np.diag(np.linalg.inv(precision)))
    plan, pipeline = _template_plan(eps, 3.0, exact + 20.0 * sd)
    estimate = plan.estimate(
        pipeline,
        _grid_state(*shape),
        jnp.asarray(observed),
        noise=HomoscedasticNoise(sigma=jnp.array(TEMPLATE_SIGMA)),
        max_iter=1000,
    )
    got = [float(estimate.values["a"]), float(estimate.values["b"])]
    distance = _mahalanobis(got, exact, precision)
    diagnostics = estimate.diagnostics
    assert diagnostics.converged is True
    assert distance < MAHALANOBIS_MAX, (distance, diagnostics.sweeps)
    assert distance * (1.0 - 1e-6) <= diagnostics.distance_bound <= MAHALANOBIS_MAX
    assert 0.0 <= diagnostics.contraction < 1.0


def test_the_monitor_terms_sum_to_the_objective():
    """The certificate reads the objective term by term; the terms must be the
    objective. Pinned against ``Conditioning.neg_log_posterior`` for an
    additive and a prediction-dependent noise, flagged and not, since the
    per-sample ``log sigma`` and the flag rule are restated there."""
    from rheplicant.inference.engines import Conditioning, _objective_terms
    from rheplicant.inference.noise import FlaggedNoise, RadiometerNoise

    observed, _, exact, _ = _template_case(0.5, 3.0, 7)
    plan, pipeline = _template_plan(0.5, 3.0, exact)
    forward, values = plan.space.forward_fn(pipeline, _grid_state(*TEMPLATE_SHAPE))
    flags = jnp.zeros(TEMPLATE_SHAPE, bool).at[1, 2].set(True)
    for noise in (
        HomoscedasticNoise(sigma=jnp.array(0.7)),
        RadiometerNoise(1e3, 1.0, 0.5),
        FlaggedNoise(RadiometerNoise(1e3, 1.0), flags),
    ):
        cond = Conditioning(
            space=plan.space,
            pipeline=pipeline,
            state_template=_grid_state(*TEMPLATE_SHAPE),
            observed=jnp.asarray(observed),
            noise=noise,
            forward=forward,
        )
        chi2, terms, _ = _objective_terms(cond, values)
        total = sum(float(jnp.sum(term)) for term in terms.values())
        assert total == pytest.approx(float(cond.neg_log_posterior(values)), rel=1e-12)
        assert float(chi2) == pytest.approx(float(cond.chi2(values)), rel=1e-12)


# ----------------------------- (iv) power law: conjugate A + gradient beta --

REF_FREQ = 70e6
A_TRUE, BETA_TRUE = 1000.0, 2.55
A_PRIOR, BETA_PRIOR = (1000.0, 1e3), (2.5, 1.0)


def _power_law_nlp(params, x, observed, sigma):
    amplitude, beta = params[0], params[1]
    residual = (observed - amplitude * x ** (-beta)) / sigma
    return (
        0.5 * jnp.sum(residual**2)
        + 0.5 * ((amplitude - A_PRIOR[0]) / A_PRIOR[1]) ** 2
        + 0.5 * ((beta - BETA_PRIOR[0]) / BETA_PRIOR[1]) ** 2
    )


def _power_law_case(n_freq: int, sigma: float):
    """Data, exact MAP (Newton in float64) and the posterior precision there."""
    freq = np.linspace(60e6, 85e6, n_freq)
    x = freq / REF_FREQ
    rng = np.random.default_rng(2026)
    observed = A_TRUE * x ** (-BETA_TRUE) + sigma * rng.standard_normal(n_freq)
    objective = jax.jit(lambda p: _power_law_nlp(p, jnp.asarray(x), jnp.asarray(observed), sigma))
    gradient, hessian = jax.jit(jax.grad(objective)), jax.jit(jax.hessian(objective))
    params = jnp.array([A_TRUE, BETA_TRUE])
    for _ in range(50):
        params = params - jnp.linalg.solve(hessian(params), gradient(params))
    precision = np.asarray(hessian(params))
    step = np.linalg.solve(precision, np.asarray(gradient(params)))
    assert np.sqrt(step @ precision @ step) < 1e-8, "the reference Newton solve stalled"
    return freq, observed[None, :], np.asarray(params), precision


@pytest.mark.parametrize("n_freq", [8, 4096])
@pytest.mark.parametrize("sigma", [100.0, 1.0, 0.01])
@pytest.mark.parametrize("learning_rate, steps", [(None, None), (1e-3, 100)])
def test_a_power_law_reaches_the_joint_map(n_freq, sigma, learning_rate, steps):
    """The verifier's A5-2 grid: -1.55 sigma at 8 channels and 1 K, -2881 sigma
    at 4096 channels and 0.01 K, both reported converged before the repair.
    ``A`` is conjugate, ``beta`` is a gradient block, and the two are correlated,
    so the sweep has to settle the pair and not only each block."""
    freq, observed, exact, precision = _power_law_case(n_freq, sigma)
    state = State(coords=Coordinates(time=jnp.arange(1.0), freq=jnp.asarray(freq)))
    pipeline = Pipeline(
        ForegroundOperator(
            amplitude=jnp.array(A_TRUE),
            spectral_index=jnp.array(BETA_TRUE),
            ref_freq=REF_FREQ,
        ),
        names=("fg",),
    )
    space = ParameterSpace(
        latents=[
            Latent("A", init=jnp.array(900.0), prior=dist.Normal(*A_PRIOR), linear=True),
            Latent("beta", init=jnp.array(2.5), prior=dist.Normal(*BETA_PRIOR)),
        ],
        bindings=[
            Bind("A", into=lambda p: p["fg"].amplitude),
            Bind("beta", into=lambda p: p["fg"].spectral_index),
        ],
    )
    options = {
        key: value
        for key, value in (("learning_rate", learning_rate), ("steps", steps))
        if value is not None
    }
    plan = SamplingPlan(space, Block("A"), Block("beta", **options))
    estimate = _estimate_or_refusal(
        plan,
        pipeline,
        state,
        jnp.asarray(observed),
        HomoscedasticNoise(sigma=jnp.array(sigma)),
    )
    got = (
        None if estimate is None else [float(estimate.values["A"]), float(estimate.values["beta"])]
    )
    label = f"n_freq {n_freq}, sigma {sigma}, lr {learning_rate}, steps {steps}"
    _check(estimate, got, exact, precision, True, label)


# ------------------- (vi) the second review's cases, as real plans in float64 --

#: The latents a :class:`_Dense` model can carry.
DENSE_LATENTS = ("x0", "x1", "x2", "x3", "x4")
DENSE_SHAPE = (100, 100)
DENSE_TAU = 1e3


class _Dense(AbstractOperator):
    """``data = design @ (x0, ..., x4)[:k]`` on the grid: a linear model of up
    to five scalar latents with any posterior precision."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    design: jax.Array
    x0: jax.Array
    x1: jax.Array
    x2: jax.Array
    x3: jax.Array
    x4: jax.Array

    def __call__(self, state: State) -> State:
        n_time, n_freq = state.coords.time.shape[0], state.coords.freq.shape[0]
        theta = jnp.stack([self.x0, self.x1, self.x2, self.x3, self.x4])
        signal = self.design @ theta[: self.design.shape[1]]
        return state.with_data(signal.reshape(n_time, n_freq))


def _dense_case(data_precision, seed):
    """A design whose data give ``data_precision`` (sigma 1), with data drawn
    from it; the posterior precision adds the ``N(0, DENSE_TAU)`` priors.
    Returns ``(design, observed, map, precision)``."""
    n, size = len(data_precision), int(np.prod(DENSE_SHAPE))
    rng = np.random.default_rng(seed)
    basis, _ = np.linalg.qr(rng.standard_normal((size, n)))
    design = basis @ np.linalg.cholesky(data_precision).T
    observed = design @ (3.0 * rng.standard_normal(n)) + rng.standard_normal(size)
    precision = data_precision + np.eye(n) / DENSE_TAU**2
    return design, observed, np.linalg.solve(precision, design.T @ observed), precision


def _dense_run(design, observed, init, **options):
    names = DENSE_LATENTS[: design.shape[1]]
    space = ParameterSpace(
        latents=[
            Latent(name, init=jnp.array(value), prior=dist.Normal(0.0, DENSE_TAU), linear=True)
            for name, value in zip(names, init, strict=True)
        ],
        bindings=[
            Bind(name, into=lambda p, name=name: getattr(p["dense"], name)) for name in names
        ],
    )
    pipeline = Pipeline(
        _Dense(design=jnp.asarray(design), **{name: jnp.array(0.0) for name in DENSE_LATENTS}),
        names=("dense",),
    )
    plan = SamplingPlan(space, *(Block(name) for name in names))
    estimate = plan.estimate(
        pipeline,
        _grid_state(*DENSE_SHAPE),
        jnp.asarray(observed.reshape(DENSE_SHAPE)),
        noise=HomoscedasticNoise(sigma=jnp.array(1.0)),
        **options,
    )
    return estimate, [float(estimate.values[name]) for name in names]


def _pairs(fast: float, slow: float) -> np.ndarray:
    """Two independent pairs of unit-variance latents, correlated ``fast`` and
    ``slow`` within each: the data precision is the correlation's inverse."""
    correlation = np.zeros((4, 4))
    correlation[:2, :2] = [[1.0, fast], [fast, 1.0]]
    correlation[2:, 2:] = [[1.0, slow], [slow, 1.0]]
    return np.linalg.inv(correlation)


@pytest.mark.parametrize(
    "slow, tol",
    [(0.9975, 1e-3), (0.9975, DEFAULT_CHI2_TOL), (0.99975, 1e-3)],
    ids=["r=0.9975-tol=1e-3", "r=0.9975-default-tol", "r=0.99975-tol=1e-3"],
)
def test_a_slow_pair_under_a_fast_one_is_certified_by_the_decrement(slow, tol):
    """The second review's HIGH. Four one-latent conjugate blocks, a fast pair
    (r = 0.6687) started 30 sigma off and a slow pair started 1 sigma off.
    The fast pair's decreases dominate every sweep, so a contraction read
    from them is the fast one, and the gap test passed with the slow pair
    0.5 to 1.0 sigma from the MAP (reviewer's measurements at 3a73590). The
    decrement sees the slow mode: every cell must converge, within 0.1 sigma,
    with a bound that covers the distance.

    It is also what keeps the decrement rare. At r = 0.99975 the run takes
    about 4600 sweeps and three decrements (measured); with the pre-screen
    removed it computed 14 and ran 8210 sweeps, because every refusal
    doubles the wait before the next candidate.
    """
    precision = _pairs(0.6687, slow)
    design, observed, exact, precision = _dense_case(precision, 0)
    init = exact + np.array([0.0, 30.0, 0.0, 1.0])
    estimate, got = _dense_run(design, observed, init, tol=tol, max_iter=20000)
    distance = _mahalanobis(got, exact, precision)
    diagnostics = estimate.diagnostics
    assert diagnostics.converged is True
    assert distance < MAHALANOBIS_MAX, (distance, diagnostics.sweeps)
    assert distance * (1.0 - 1e-6) <= diagnostics.distance_bound <= MAHALANOBIS_MAX
    if slow == 0.99975:
        assert diagnostics.certificate_attempts <= 4, diagnostics.certificate_attempts


#: The contraction of the Gauss-Seidel sweep a random precision must have, by
#: the kind of its leading eigenvalue. Real ones at 0.99 and above are the
#: slow runs the review's false certificates came from (the cap of 0.999
#: keeps a run to a few hundred sweeps); complex ones oscillate, and one in
#: several hundred reaches 0.9.
GAUSS_SEIDEL_CONTRACTION = {"real": (0.99, 0.999), "complex": (0.9, 0.999)}


def _gauss_seidel_precision(seed: int, kind: str) -> np.ndarray:
    """The first of the review's random dense precisions, 3 to 5 blocks with
    unit diagonal, whose sweep's leading eigenvalue is of ``kind`` and inside
    :data:`GAUSS_SEIDEL_CONTRACTION`."""
    rng = np.random.default_rng(seed)
    low, high = GAUSS_SEIDEL_CONTRACTION[kind]
    while True:
        n = int(rng.integers(3, 6))
        factor = rng.standard_normal((n, n + int(rng.integers(0, 3))))
        precision = factor @ factor.T + 1e-3 * np.eye(n)
        root = np.sqrt(np.diag(precision))
        precision = precision / np.outer(root, root)
        spectrum = np.linalg.eigvals(-np.linalg.solve(np.tril(precision), np.triu(precision, 1)))
        lead = spectrum[np.argmax(np.abs(spectrum))]
        found = "complex" if abs(lead.imag) > 1e-9 else "real"
        if found == kind and low <= abs(lead) <= high:
            return precision


@pytest.mark.parametrize("seed", [7, 8])
@pytest.mark.parametrize("kind", ["real", "complex"])
def test_a_random_dense_precision_converges_within_a_tenth_of_a_sigma(seed, kind):
    """The second review replayed the gap test's rule on 4618 such
    precisions and found false certificates up to 0.32 sigma at ``|f| =
    5e3``, about this ``N``, and 1.33 sigma at ``|f| = 5e5``. Here they run
    as real plans, one conjugate block per latent, started 30 sigma off: a
    verdict must be within 0.1 sigma. Measured: 0.074 to 0.082 sigma for the
    slow real ones, which stop just inside the threshold as they should."""
    data_precision = _gauss_seidel_precision(seed, kind)
    design, observed, exact, precision = _dense_case(data_precision, seed)
    start = np.random.default_rng(seed).standard_normal(len(exact))
    estimate = None
    try:
        estimate, got = _dense_run(design, observed, exact + 30.0 * start, max_iter=20000)
    except ParameterSpaceError as refusal:
        assert "did not converge" in str(refusal)
    _check(
        estimate,
        None if estimate is None else got,
        exact,
        precision,
        True,
        f"seed {seed}, {kind} leading eigenvalue",
    )


def _float64_basis_model():
    """``test_plan.py``'s bilinear basis model, built in float64.

    That module's constants are arrays made at import, and it is imported in
    the suite's float32; executing a fresh copy while this module's fixture
    holds x64 on makes every one of them float64.
    """
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "_basis_model_float64", Path(__file__).with_name("test_plan.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("noise", [0.30, 0.32])
def test_an_inexact_inner_solve_is_tightened_until_the_decrement_certifies(noise):
    """The second review's MEDIUM: at the default ``solve_tol = 1e-6`` the
    basis model's sweep has a fixed point 0.113 (noise 0.30) and 0.106 (0.32)
    posterior sigma from the MAP, measured by the reviewer in float64, so no
    stop there can be certified. The run tightens the conjugate solves when
    the objective rises or the decrement refuses, and converges; measured at
    ``solve_tol = 1e-8``, 0.002 sigma off. Without the tightening both runs
    refuse after 3000 sweeps."""
    basis = _float64_basis_model()
    space, pipeline = basis.basis_space(), basis.make_pipeline()
    observed = basis.observed_of(space, pipeline, basis.TRUTH)
    exact, precision = basis._basis_map(observed, sigma=noise)
    estimate = SamplingPlan(space, Block("gain"), Block("t_coeff")).estimate(
        pipeline,
        basis.make_state(),
        observed,
        noise=noise,
        max_iter=3000,
        solve_guard=None,
    )
    assert estimate.values["gain"].dtype == jnp.float64
    diagnostics = estimate.diagnostics
    assert diagnostics.converged is True
    distance = basis._posterior_sigmas_from(estimate, exact, precision)
    assert distance < MAHALANOBIS_MAX, (distance, diagnostics.sweeps)
    assert diagnostics.solve_tol < 1e-6


# ------------------- (vii) more latents than the decrement forms a Hessian for --

WIDE_COEFFICIENTS = 1200
WIDE_SHAPE = (40, 100)
WIDE_TAU = 10.0


class _Wide(AbstractOperator):
    """``data = design @ theta`` for one vector latent: a block whose size
    puts the decrement on its iterative path."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    design: jax.Array
    theta: jax.Array

    def __call__(self, state: State) -> State:
        n_time, n_freq = state.coords.time.shape[0], state.coords.freq.shape[0]
        return state.with_data((self.design @ self.theta).reshape(n_time, n_freq))


class _Split(AbstractOperator):
    """:class:`_Wide` with its coefficients in two latents, so a plan over it
    has two blocks and no claim to joint linearity from either."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    design: jax.Array
    theta: jax.Array
    phi: jax.Array

    def __call__(self, state: State) -> State:
        n_time, n_freq = state.coords.time.shape[0], state.coords.freq.shape[0]
        whole = jnp.concatenate([self.theta, self.phi])
        return state.with_data((self.design @ whole).reshape(n_time, n_freq))


class _Product(AbstractOperator):
    """``data = (design @ theta) * (1 + mean(phi))``: affine in each latent
    alone, bilinear in the pair. The shape that makes two blocks' separate
    linearity checks insufficient, and the reason a floor needs the joint
    one."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)
    design: jax.Array
    theta: jax.Array
    phi: jax.Array

    def __call__(self, state: State) -> State:
        n_time, n_freq = state.coords.time.shape[0], state.coords.freq.shape[0]
        columns = self.design[:, : self.theta.shape[0]]
        signal = (columns @ self.theta) * (1.0 + jnp.mean(self.phi))
        return state.with_data(signal.reshape(n_time, n_freq))


def _wide_plan(seed=3, blocks=1, scales=None):
    """A wide linear-Gaussian model, its data, and the exact MAP and posterior
    precision. ``blocks=2`` splits the coefficients in two, which is enough to
    cost the plan its claim to joint linearity and so its floor; ``scales``
    replaces the single prior width with one per coefficient."""
    rng = np.random.default_rng(seed)
    size = int(np.prod(WIDE_SHAPE))
    design = rng.standard_normal((size, WIDE_COEFFICIENTS)) / np.sqrt(WIDE_COEFFICIENTS)
    truth = rng.standard_normal(WIDE_COEFFICIENTS)
    observed = design @ truth + rng.standard_normal(size)
    width = np.full(WIDE_COEFFICIENTS, WIDE_TAU) if scales is None else np.asarray(scales)
    precision = design.T @ design + np.diag(1.0 / width**2)
    exact = np.linalg.solve(precision, design.T @ observed)
    half = WIDE_COEFFICIENTS // 2
    product = blocks == "bilinear"
    if product:
        blocks = 2
    # a scalar width where they are all the same: the linearity probe reads a
    # latent's magnitude off its declaration, and a 1200-vector of tens sends
    # it far enough out that the design's own matmul rounding trips the check

    def declared(start, stop):
        return WIDE_TAU if scales is None else jnp.asarray(width[start:stop])

    if blocks == 1:
        latents = [
            Latent(
                "theta",
                init=jnp.zeros(WIDE_COEFFICIENTS),
                prior=dist.Normal(jnp.zeros(WIDE_COEFFICIENTS), declared(0, WIDE_COEFFICIENTS)),
                linear=True,
            )
        ]
        bindings = [Bind("theta", into=lambda p: p["wide"].theta)]
        pipeline = Pipeline(
            _Wide(design=jnp.asarray(design), theta=jnp.zeros(WIDE_COEFFICIENTS)),
            names=("wide",),
        )
        plan = SamplingPlan(ParameterSpace(latents=latents, bindings=bindings), Block("theta"))
    else:
        latents = [
            Latent(
                "theta",
                init=jnp.zeros(half),
                prior=dist.Normal(jnp.zeros(half), declared(0, half)),
                linear=True,
            ),
            Latent(
                "phi",
                init=jnp.zeros(WIDE_COEFFICIENTS - half),
                prior=dist.Normal(
                    jnp.zeros(WIDE_COEFFICIENTS - half), declared(half, WIDE_COEFFICIENTS)
                ),
                linear=True,
            ),
        ]
        bindings = [
            Bind("theta", into=lambda p: p["wide"].theta),
            Bind("phi", into=lambda p: p["wide"].phi),
        ]
        operator = _Product if product else _Split
        pipeline = Pipeline(
            operator(
                design=jnp.asarray(design),
                theta=jnp.zeros(half),
                phi=jnp.zeros(WIDE_COEFFICIENTS - half),
            ),
            names=("wide",),
        )
        plan = SamplingPlan(
            ParameterSpace(latents=latents, bindings=bindings), Block("theta"), Block("phi")
        )
    return plan, pipeline, jnp.asarray(observed.reshape(WIDE_SHAPE)), exact, precision


def test_a_block_too_wide_to_form_a_hessian_certifies_on_its_prior_floor():
    """Above :data:`~bayesmith.optimize.certify.DENSE_MAX` latents the
    decrement cannot form the Hessian, so its bound needs a lower bound on
    the smallest eigenvalue. This plan can prove one: a single conjugate
    block is linear in every latent jointly, the noise does not depend on the
    prediction, and every prior is Normal, so ``H = J^T N^-1 J + P`` is
    bounded below by the prior precision. The run must certify with that, in
    a handful of Hessian-vector products rather than a probe of the
    spectrum."""
    plan, pipeline, observed, exact, precision = _wide_plan()
    state = _grid_state(*WIDE_SHAPE)
    noise = HomoscedasticNoise(sigma=jnp.array(1.0))
    cond, _ = plan._prepare(pipeline, state, observed, noise, False, "probe")
    assert plan._curvature_floor(cond) == pytest.approx(1.0 / WIDE_TAU**2)
    estimate = plan.estimate(
        pipeline,
        state,
        observed,
        noise=noise,
        max_iter=30,
        check_identifiability=False,
    )
    diagnostics = estimate.diagnostics
    got = np.asarray(estimate.values["theta"], np.float64)
    distance = _mahalanobis(got, exact, precision)
    assert diagnostics.converged is True
    assert distance < MAHALANOBIS_MAX, (distance, diagnostics.sweeps)
    assert distance * (1.0 - 1e-6) <= diagnostics.distance_bound <= MAHALANOBIS_MAX
    # Measured: 2 Hessian-vector products -- the iteration stops as soon as
    # its bound is inside the threshold -- against the 1201 that forming the
    # Hessian would take and the 34 that probing the spectrum does.
    assert diagnostics.certificate_iterations <= 20, diagnostics.certificate_iterations


def test_without_a_proof_that_block_refuses_and_says_what_would_prove_it():
    """The same size of model with no floor its plan can prove: the decrement
    falls back to a Lanczos probe, whose floor is an estimate, so nothing is
    certified however small the distance looks. The run refuses at max_iter,
    and the refusal names the three things that would make a proof."""
    from bayesmith.optimize import certify

    from rheplicant.inference.plan import _Attempt, _not_converged_message

    probed = certify.Decrement(
        estimate=0.01,
        distance=0.02,
        lambda2=1e-4,
        residual=1e-6,
        reach=1e-3,
        kappa=50.0,
        products=64,
        dense=False,
        floor_source=certify.FLOOR_PROBE,
        status=certify.CONVERGED,
    )
    message = _not_converged_message(
        max_iter=50,
        tol=1e-8,
        gap_tol=0.005,
        effective=1e-8,
        changed=True,
        objective=[1.0, 1.0],
        chi2=[1.0, 1.0],
        contraction=0.5,
        gap=1e-9,
        rise=None,
        attempt=_Attempt(50, probed, False),
        solve_tol=1e-6,
        dtype=np.float32,
        hidden="",
    )
    assert "cannot certify this estimate at this size and precision" in message
    assert "came from probe, not from a proof" in message
    assert f"more than {certify.DENSE_MAX} latents" in message  # form the Hessian
    assert "ONE conjugate block" in message  # prove the floor
    assert "JAX_ENABLE_X64=1" in message  # or change precision
    # a proven floor says none of it
    proven = dataclasses.replace(probed, floor_source=certify.FLOOR_DENSE)
    plain = _not_converged_message(
        max_iter=50,
        tol=1e-8,
        gap_tol=0.005,
        effective=1e-8,
        changed=True,
        objective=[1.0, 1.0],
        chi2=[1.0, 1.0],
        contraction=0.5,
        gap=1e-9,
        rise=None,
        attempt=_Attempt(50, proven, False),
        solve_tol=1e-6,
        dtype=np.float32,
        hidden="",
    )
    assert plain.startswith("SamplingPlan.estimate did not converge")
    assert "not from a proof" not in plain


def test_a_wide_block_without_a_proof_runs_to_max_iter_and_refuses():
    """The same at the level of a run: 1200 coefficients in two blocks over a
    BILINEAR model, which each block's own linearity check passes and the
    joint one does not, so the plan can prove no floor and the decrement has
    only a probe. The run refuses however small the probe's bound looks,
    which is the point: this package does not certify on an estimated
    floor."""
    # prior width 1 rather than 10: the same model, with the linearity probe's
    # 1000x excursion small enough that the design's matmul rounding stays
    # under its tolerance
    plan, pipeline, observed, exact, precision = _wide_plan(
        blocks="bilinear", scales=np.ones(WIDE_COEFFICIENTS)
    )
    state = _grid_state(*WIDE_SHAPE)
    noise = HomoscedasticNoise(sigma=jnp.array(1.0))
    cond, _ = plan._prepare(pipeline, state, observed, noise, False, "probe")
    assert plan._curvature_floor(cond) is None
    with pytest.raises(ParameterSpaceError) as refused:
        plan.estimate(
            pipeline, state, observed, noise=noise, max_iter=20, check_identifiability=False
        )
    message = str(refused.value)
    assert "cannot certify this estimate at this size and precision" in message
    assert "came from probe, not from a proof" in message


@pytest.mark.parametrize(
    "variation, floored",
    [
        ("plain", True),
        ("two blocks", True),
        ("gradient block", False),
        ("bilinear", False),
        ("no prior", False),
        ("sigma from the prediction", False),
    ],
)
def test_the_floor_is_claimed_only_where_the_model_proves_it(variation, floored):
    """``H >= P`` needs a prediction the model is affine in JOINTLY, declared
    so on every latent, a Normal prior on every latent, and a sigma that does
    not depend on the prediction. Two blocks are not evidence of joint
    affinity, so the check is asked of the union: the bilinear case here
    passes each block's own linearity check and fails that one. A latent that
    does not claim to be linear is not checked and so cannot be floored,
    whatever its model happens to be."""
    from rheplicant.inference.noise import RadiometerNoise

    size, width = 8, 4.0
    rng = np.random.default_rng(0)
    design = rng.standard_normal((int(np.prod(DENSE_SHAPE)), size))
    observed = jnp.asarray((design @ rng.standard_normal(size)).reshape(DENSE_SHAPE))
    prior = None if variation == "no prior" else dist.Normal(jnp.zeros(size), width)
    linear = variation != "gradient block"
    space = ParameterSpace(
        latents=[Latent("theta", init=jnp.zeros(size), prior=prior, linear=linear)],
        bindings=[Bind("theta", into=lambda p: p["wide"].theta)],
    )
    pipeline = Pipeline(_Wide(design=jnp.asarray(design), theta=jnp.zeros(size)), names=("wide",))
    blocks = (Block("theta"),)
    if variation in ("two blocks", "bilinear"):
        space = ParameterSpace(
            latents=[
                Latent(
                    name,
                    init=jnp.zeros(size // 2) + (0.0 if name == "theta" else 1.0),
                    prior=dist.Normal(jnp.zeros(size // 2), width),
                    linear=True,
                )
                for name in ("theta", "phi")
            ],
            bindings=[
                Bind("theta", into=lambda p: p["wide"].theta),
                Bind("phi", into=lambda p: p["wide"].phi),
            ],
        )
        operator = _Split if variation == "two blocks" else _Product
        pipeline = Pipeline(
            operator(
                design=jnp.asarray(design), theta=jnp.zeros(size // 2), phi=jnp.zeros(size // 2)
            ),
            names=("wide",),
        )
        blocks = (Block("theta"), Block("phi"))
    noise = (
        RadiometerNoise(1e3, 1.0)
        if variation == "sigma from the prediction"
        else HomoscedasticNoise(sigma=jnp.array(1.0))
    )
    plan = SamplingPlan(space, *blocks)
    cond, _ = plan._prepare(pipeline, _grid_state(*DENSE_SHAPE), observed, noise, False, "probe")
    floor = plan._curvature_floor(cond)
    assert (floor is not None) is floored, (variation, floor)
    if floored:
        assert floor == pytest.approx(1.0 / width**2)


@pytest.mark.parametrize(
    "scales",
    [
        "one width",
        "logspace(-2, 2)",
        "logspace(2, -2)",
        "one wide coefficient",
        "one narrow coefficient",
    ],
)
def test_the_claimed_floor_is_below_the_hessians_smallest_eigenvalue(scales):
    """``H >= P`` floors the curvature at the SMALLEST prior precision, which
    is the WIDEST prior scale — one character apart from the narrowest, and
    wrong by the square of the spread.

    The third review measured that: swapping the widest for the narrowest on
    a block whose scales span ``logspace(-2, 2)`` claims 1e4 against a true
    smallest eigenvalue of 5.4e-4, and certifies 2.4e-4 posterior sigma as
    5.8e-6. Here the claim is checked against the eigenvalue itself, on a
    model weak enough in the data that the prior sets the curvature, so a
    floor above it is the assertion that fails."""
    size = WIDE_COEFFICIENTS
    width = {
        "one width": np.full(size, 3.0),
        "logspace(-2, 2)": np.logspace(-2.0, 2.0, size),
        "logspace(2, -2)": np.logspace(2.0, -2.0, size),
        "one wide coefficient": np.concatenate([[50.0], np.full(size - 1, 0.1)]),
        "one narrow coefficient": np.concatenate([[0.02], np.full(size - 1, 5.0)]),
    }[scales]
    plan, pipeline, observed, _, precision = _wide_plan(scales=width, seed=11)
    cond, _ = plan._prepare(
        pipeline,
        _grid_state(*WIDE_SHAPE),
        observed,
        HomoscedasticNoise(sigma=jnp.array(1.0)),
        False,
        "probe",
    )
    floor = plan._curvature_floor(cond)
    smallest = float(np.linalg.eigvalsh(precision)[0])
    assert floor is not None
    assert floor <= smallest, (scales, floor, smallest)
    assert floor == pytest.approx(1.0 / np.max(width) ** 2)
