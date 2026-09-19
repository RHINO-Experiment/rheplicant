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
from rheplicant.inference.plan import EARLIEST_CONVERGED_SWEEP, MIN_SWEEPS
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
        coords=Coordinates(
            time=jnp.arange(float(n_time)), freq=jnp.linspace(60e6, 85e6, n_freq)
        )
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
        plan, pipeline, _grid_state(*SKY_SHAPE), jnp.asarray(observed),
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
        pipeline, _grid_state(*SKY_SHAPE), jnp.asarray(observed),
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
    return (np.linalg.solve(fisher, projected), np.linalg.solve(precision, projected),
            precision)


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
    pipeline = Pipeline(
        _TwoTemplates(a=jnp.array(0.0), b=jnp.array(0.0), eps=eps), names=("tt",)
    )
    estimate = _estimate_or_refusal(
        SamplingPlan(space, Block("a"), Block("b")), pipeline,
        _grid_state(*TEMPLATE_SHAPE), jnp.asarray(observed),
        HomoscedasticNoise(sigma=jnp.array(TEMPLATE_SIGMA)),
        max_iter=TEMPLATE_MAX_ITER,
    )
    got = (
        None if estimate is None
        else [float(estimate.values["a"]), float(estimate.values["b"])]
    )
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
    pipeline = Pipeline(
        _TwoTemplates(a=array(0.0), b=array(0.0), eps=eps), names=("tt",)
    )
    return SamplingPlan(space, Block("a"), Block("b")), pipeline


def test_the_earliest_stop_is_sweep_three_whatever_min_sweeps_says():
    """Started AT the MAP, no sweep moves the objective beyond its resolution.

    Both certificates hold from sweep 2 on (a run that never moved uses a
    contraction of 0), and the first certifiable decrease is sweep 2 against
    sweep 1, so the earliest stop is sweep 3 for ``min_sweeps`` of 1, 2 and 3;
    ``min_sweeps`` above that is the floor; and a cap of two sweeps can
    never converge. Moved here from ``test_plan.py``'s float32 basis model,
    whose start at the answer is not a fixed point in float32: its conjugate
    solves move the objective by up to 2.3e-3 nats a sweep there, which the
    gap certificate reads as the rises they are.
    """
    observed, _, exact, _ = _template_case(0.2, 3.0, 11)
    plan, pipeline = _template_plan(0.2, 3.0, exact)
    common = {"noise": HomoscedasticNoise(sigma=jnp.array(TEMPLATE_SIGMA)),
              "max_iter": 30}
    state, data = _grid_state(*TEMPLATE_SHAPE), jnp.asarray(observed)
    for min_sweeps in (1, 2, 3):
        early = plan.estimate(pipeline, state, data, min_sweeps=min_sweeps, **common)
        assert early.diagnostics.converged is True
        assert early.diagnostics.sweeps == EARLIEST_CONVERGED_SWEEP == 3, min_sweeps
    floored = plan.estimate(pipeline, state, data, min_sweeps=8, **common)
    assert floored.diagnostics.sweeps == 8
    with pytest.raises(ParameterSpaceError) as capped:
        plan.estimate(pipeline, state, data, min_sweeps=1,
                      **{**common, "max_iter": 2})
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
    (``N = 1e6``, r = 0.993); at ``N = 1e5`` it would be ~0.2. The gap
    certificate is in nats, so the same 0.1 sigma holds at every ``N``.

    Started 20 posterior sigma off along both axes. Every cell must converge
    in float64, within 0.1 sigma, and report a distance bound that covers it.
    """
    observed, _, exact, precision = _template_case(eps, 3.0, 21, shape=shape)
    sd = np.sqrt(np.diag(np.linalg.inv(precision)))
    plan, pipeline = _template_plan(eps, 3.0, exact + 20.0 * sd)
    estimate = plan.estimate(
        pipeline, _grid_state(*shape), jnp.asarray(observed),
        noise=HomoscedasticNoise(sigma=jnp.array(TEMPLATE_SIGMA)), max_iter=1000,
    )
    got = [float(estimate.values["a"]), float(estimate.values["b"])]
    distance = _mahalanobis(got, exact, precision)
    diagnostics = estimate.diagnostics
    assert diagnostics.converged is True
    assert distance < MAHALANOBIS_MAX, (distance, diagnostics.sweeps)
    assert diagnostics.distance_bound <= MAHALANOBIS_MAX
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
    for noise in (HomoscedasticNoise(sigma=jnp.array(0.7)),
                  RadiometerNoise(1e3, 1.0, 0.5),
                  FlaggedNoise(RadiometerNoise(1e3, 1.0), flags)):
        cond = Conditioning(space=plan.space, pipeline=pipeline,
                            state_template=_grid_state(*TEMPLATE_SHAPE),
                            observed=jnp.asarray(observed), noise=noise,
                            forward=forward)
        chi2, terms, _ = _objective_terms(cond, values)
        total = sum(float(jnp.sum(term)) for term in terms.values())
        assert total == pytest.approx(float(cond.neg_log_posterior(values)), rel=1e-12)
        assert float(chi2) == pytest.approx(float(cond.chi2(values)), rel=1e-12)


class TestTheGapCertificate:
    """``plan._gap_step`` on hand-made decreases, in nats, tol = 0.005."""

    @staticmethod
    def _run(decreases, resolution=1e-9, tol=0.005):
        from rheplicant.inference.plan import _gap_step, _GapState

        state, out = _GapState(), []
        for decrease in decreases:
            state, certified, gap, rho = _gap_step(state, decrease, resolution, tol)
            out.append((certified, gap, rho))
        return out

    def test_a_geometric_run_certifies_once_its_tail_is_inside_tol(self):
        # D[k] = 0.5**k: the gap after D is (D + resolution) / (1 - 0.5).
        out = self._run([0.5**k for k in range(1, 12)])
        for (certified, gap, rho), k in zip(out[1:], range(2, 12), strict=True):
            assert rho == pytest.approx(0.5)
            assert gap == pytest.approx(2 * (0.5**k + 1e-9), rel=1e-12)
            assert certified is (gap <= 0.005)
        assert [certified for certified, _, _ in out].index(True) == 8  # 2**-9

    def test_a_slow_contraction_is_not_certified_by_a_small_step(self):
        # D = 1e-4 per sweep at rho = 0.999: the tail is 0.1 nats, 20x tol.
        out = self._run([1e-4 * 0.999**k for k in range(10)])
        assert not any(certified for certified, _, _ in out)

    def test_a_rise_certifies_nothing_and_forgets_the_contraction(self):
        out = self._run([1e-3, 5e-4, -1e-3, 1e-6])
        assert out[2] == (False, None, None)
        assert out[3][0] is False and out[3][2] is None

    def test_a_stall_below_the_resolution_keeps_the_last_contraction(self):
        # A long valley stalled on rounding: resolvable decreases contracting
        # at 0.9999, then nothing the arithmetic can see.
        out = self._run([1e-3 * 0.9999**k for k in range(5)] + [0.0] * 5,
                        resolution=1e-8)
        assert not any(certified for certified, _, _ in out)
        assert out[-1][2] == pytest.approx(0.9999, rel=1e-6)
        # ... and reads the vanished decrease at the size that contraction
        # predicts: the tail is still ~10 nats, not the resolution.
        assert out[-1][1] > 1.0

    def test_a_run_that_never_moved_uses_a_contraction_of_zero(self):
        out = self._run([0.0, 0.0, 0.0], resolution=1e-9)
        assert [certified for certified, _, _ in out] == [True, True, True]
        assert out[-1][2] == 0.0


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
    objective = jax.jit(
        lambda p: _power_law_nlp(p, jnp.asarray(x), jnp.asarray(observed), sigma)
    )
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
            amplitude=jnp.array(A_TRUE), spectral_index=jnp.array(BETA_TRUE),
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
        plan, pipeline, state, jnp.asarray(observed),
        HomoscedasticNoise(sigma=jnp.array(sigma)),
    )
    got = (
        None if estimate is None
        else [float(estimate.values["A"]), float(estimate.values["beta"])]
    )
    label = f"n_freq {n_freq}, sigma {sigma}, lr {learning_rate}, steps {steps}"
    _check(estimate, got, exact, precision, True, label)


def test_a_newton_step_that_raises_the_potential_is_refused():
    """The Newton steps after Adam are kept only when the potential does not rise.

    ``-cos(x)`` has negative curvature at ``x = 2``, where the Newton step
    heads past the maximum at ``pi`` to 4.19 and the potential goes up from
    0.42 to 0.50: refused, so the point Adam left is returned unchanged. From
    ``x = 0.5``, inside the convex basin, the same three steps reach the
    minimum at 0.
    """
    from rheplicant.inference.engines import _newton_polish

    def potential(x):
        return -jnp.cos(x["x"])

    stuck = _newton_polish(potential, {"x": jnp.array(2.0)}, 3)
    assert float(stuck["x"]) == 2.0
    settled = _newton_polish(potential, {"x": jnp.array(0.5)}, 3)
    assert abs(float(settled["x"])) < 1e-6
