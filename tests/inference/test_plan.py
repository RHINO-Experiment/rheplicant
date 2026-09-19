"""Tests for SamplingPlan — one declared partition, two exits.

The motivating failure is a bilinear ``gain x T_ant`` model solved by
alternation. Every per-block guard the package ships reports green at every
sweep — CG residual ~1e-7, per-block condition number ~1.6, ``check_linearity``
passing because each conditional genuinely is affine — and the answer is
hundreds to thousands of kelvin wrong. Three things in this file are the point:

* the free-per-cell parameterization is REFUSED by the identifiability check,
  with the degenerate direction named as a combination of latents;
* the basis parameterization runs, and ``plan.estimate`` and ``plan.sample``
  agree with each other and with the truth;
* the JOINT chi-squared monitor detects non-convergence at a sweep where every
  per-block residual reads converged.

The fixture is deliberately asymmetric in every dimension a symmetric one would
blind: 6 times against 9 frequencies, a (3, 4) coefficient matrix that is
neither square nor symmetric, a 6-element gain against a 12-element temperature
block, and fewer data points (54) than parameters (60) in the free
parameterization — which is the case an SVD shortcut silently empties.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Block,
    Draws,
    Estimate,
    Latent,
    ParameterSpace,
    SamplingPlan,
    identifiability,
    split_rhat,
)
from rheplicant.inference.engines import CONJUGATE, GRADIENT
from rheplicant.inference.noise import (
    FlaggedNoise,
    HomoscedasticNoise,
    RadiometerNoise,
)
from rheplicant.inference.plan import (
    CHECK_EACH_SWEEP,
    CHECK_ONCE,
    MIN_DRAWS,
    MIN_SWEEPS,
    OBJECTIVE_FLOOR_EPS,
    _halves,
    _settled,
)
from rheplicant.radio import GainOperator

N_TIME, N_FREQ = 6, 9
TONE_CHANNEL, TONE_KELVIN = 4, 4000.0
NOISE = 1.0


# --------------------------------------------------------------- test doubles --


class AntennaTemperature(AbstractOperator):
    """Write a full ``(n_time, n_freq)`` antenna temperature as the data."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    t_ant: jax.Array

    def __call__(self, state):
        return state.with_data(self.t_ant)


class CalibrationTone(AbstractOperator):
    """A KNOWN per-channel signal ahead of the gain — the identifying tone."""

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    tone: jax.Array

    def __call__(self, state):
        return state.with_data(state.data + self.tone[None, :])


class GaussianLine(AbstractOperator):
    """``data[t, f] = amp[t] * exp(-((x[f] - centre)/width)^2 / 2)``.

    The fixture for the GRADIENT engine: ``amp`` enters linearly and ``centre``
    does not, so a plan over the two has one block of each kind. Deliberately a
    separate model from the bilinear one above — a non-linear latent bolted onto
    that one would be nearly degenerate with its polynomial basis, and a
    gradient-engine test running on an ill-conditioned model would be measuring
    the conditioning rather than the engine.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    amp: jax.Array
    centre: jax.Array
    width: float = 0.4

    def __call__(self, state):
        x = jnp.linspace(-1.0, 1.0, state.coords.freq.shape[0])
        profile = jnp.exp(-0.5 * ((x - self.centre) / self.width) ** 2)
        return state.with_data(self.amp[:, None] * profile[None, :])


# ------------------------------------------------------------------- fixtures --


def _poly(n: int, degree: int) -> jax.Array:
    x = jnp.linspace(-1.0, 1.0, n)
    return jnp.stack([x**k for k in range(degree)], axis=1)


#: Different degrees on purpose, so the coefficient matrix is (3, 4): neither
#: square nor symmetric, and not confusable with either basis.
TIME_BASIS = _poly(N_TIME, 3)
FREQ_BASIS = _poly(N_FREQ, 4)
COEFF0 = jnp.array(
    [
        [2900.0, -170.0, 45.0, -6.0],
        [110.0, 22.0, -9.0, 3.0],
        [-38.0, 7.0, 2.5, -1.0],
    ]
)
T_ANT0 = TIME_BASIS @ COEFF0 @ FREQ_BASIS.T
GAIN0 = 1.4 + 0.07 * jnp.arange(N_TIME, dtype=float)

#: Starting points: wrong, and wrong by DIFFERENT fractions in the two latents,
#: so a run that recovered one and left the other could not pass by symmetry.
GAIN_GUESS = 0.88 * GAIN0
COEFF_GUESS = 0.80 * COEFF0

GAIN_PRIOR = dist.Normal(jnp.ones(N_TIME), 10.0)
COEFF_PRIOR = dist.Normal(jnp.zeros((3, 4)), 1e4)
CELL_PRIOR = dist.Normal(jnp.zeros((N_TIME, N_FREQ)), 1e4)


def make_state() -> State:
    return State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        meta={"telescope": "RHINO", "obs_id": "plan-000"},
    )


STATE = make_state()


@pytest.fixture
def state():
    return make_state()


def make_pipeline(tone_kelvin: float = TONE_KELVIN) -> Pipeline:
    """``data[t, f] = gain[t] * (T_ant[t, f] + tone[f])``, tone known."""
    tone = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(tone_kelvin)
    return Pipeline(
        AntennaTemperature(t_ant=T_ANT0),
        CalibrationTone(tone=tone),
        GainOperator(gain=GAIN0),
        names=("t_ant", "tone", "gain"),
    )


def basis_space(gain_prior=GAIN_PRIOR, coeff_prior=COEFF_PRIOR) -> ParameterSpace:
    """The identified parameterization: a (3, 4) time x frequency basis."""
    return ParameterSpace(
        latents=[
            Latent("gain", init=GAIN_GUESS, prior=gain_prior, linear=True),
            Latent("t_coeff", init=COEFF_GUESS, prior=coeff_prior, linear=True),
        ],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind(
                "t_coeff",
                into=lambda p: p["t_ant"].t_ant,
                fn=lambda c: TIME_BASIS @ c @ FREQ_BASIS.T,
            ),
        ],
    )


def free_space() -> ParameterSpace:
    """The degenerate one: a free antenna temperature per (time, freq) cell."""
    return ParameterSpace(
        latents=[
            Latent("gain", init=GAIN_GUESS, prior=GAIN_PRIOR, linear=True),
            Latent("t_ant", init=0.8 * T_ANT0, prior=CELL_PRIOR, linear=True),
        ],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind("t_ant", into=lambda p: p["t_ant"].t_ant),
        ],
    )


def observed_of(space: ParameterSpace, pipeline, truth: dict) -> jax.Array:
    forward, _ = space.forward_fn(pipeline, STATE)
    return forward(truth)


TRUTH = {"gain": GAIN0, "t_coeff": COEFF0}


@pytest.fixture
def basis_setup(state):
    space, pipeline = basis_space(), make_pipeline()
    return space, pipeline, observed_of(space, pipeline, TRUTH)


CENTRE_PRIOR = dist.Normal(0.1, 0.5)
AMP_PRIOR = dist.Normal(jnp.zeros(N_TIME), 1e3)


def line_space(centre_prior=CENTRE_PRIOR) -> ParameterSpace:
    """``amp`` linear, ``centre`` not: one block of each engine."""
    return ParameterSpace(
        latents=[
            Latent(
                "amp",
                init=jnp.full((N_TIME,), 40.0),
                prior=AMP_PRIOR,
                linear=True,
            ),
            Latent("centre", init=jnp.array(0.10), prior=centre_prior),
        ],
        bindings=[
            Bind("amp", into=lambda p: p["line"].amp),
            Bind("centre", into=lambda p: p["line"].centre),
        ],
    )


LINE_TRUTH = {
    "amp": jnp.array([31.0, 44.0, 57.0, 25.0, 63.0, 38.0]),
    "centre": jnp.array(0.35),
}


def make_line_pipeline() -> Pipeline:
    return Pipeline(
        GaussianLine(amp=jnp.zeros(N_TIME), centre=jnp.array(0.0)), names=("line",)
    )


def _line_map(observed, *, sigma):
    """The line model's exact joint MAP and the posterior precision there.

    Newton on ``0.5 chi2 - log prior`` over ``(amp, centre)``, in NumPy float64
    with the derivatives of :class:`GaussianLine` written out by hand, so the
    reference shares no code with the estimate it judges and does not touch
    the process-global x64 flag. Returns ``({name: array}, precision)`` with
    the precision over ``amp`` then ``centre``, flattened.
    """
    data = np.asarray(observed, np.float64)
    x = np.linspace(-1.0, 1.0, N_FREQ)
    width = GaussianLine(amp=jnp.zeros(N_TIME), centre=jnp.array(0.0)).width
    amp_loc = np.broadcast_to(np.asarray(AMP_PRIOR.loc, np.float64), (N_TIME,))
    amp_scale = np.broadcast_to(np.asarray(AMP_PRIOR.scale, np.float64), (N_TIME,))
    centre_loc = float(CENTRE_PRIOR.loc)
    centre_scale = float(CENTRE_PRIOR.scale)

    def derivatives(amp, centre):
        u = (x - centre) / width**2
        profile = np.exp(-0.5 * ((x - centre) / width) ** 2)
        slope = profile * u                        # d profile / d centre
        bend = profile * (u**2 - 1.0 / width**2)   # d2 profile / d centre2
        residual = (data - amp[:, None] * profile[None, :]) / sigma
        gradient = np.concatenate([
            -residual @ profile / sigma + (amp - amp_loc) / amp_scale**2,
            [-np.sum(residual * amp[:, None] * slope[None, :]) / sigma
             + (centre - centre_loc) / centre_scale**2],
        ])
        hessian = np.zeros((N_TIME + 1, N_TIME + 1))
        hessian[:N_TIME, :N_TIME] = np.diag(
            np.sum(profile**2) / sigma**2 + 1.0 / amp_scale**2
        )
        cross = amp * np.sum(profile * slope) / sigma**2 - residual @ slope / sigma
        hessian[:N_TIME, N_TIME] = hessian[N_TIME, :N_TIME] = cross
        hessian[N_TIME, N_TIME] = (
            np.sum(amp**2) * np.sum(slope**2) / sigma**2
            - np.sum(residual * amp[:, None] * bend[None, :]) / sigma
            + 1.0 / centre_scale**2
        )
        return gradient, hessian

    flat = np.concatenate([np.asarray(LINE_TRUTH["amp"], np.float64),
                           [float(LINE_TRUTH["centre"])]])
    for _ in range(30):
        gradient, hessian = derivatives(flat[:N_TIME], flat[N_TIME])
        flat = flat - np.linalg.solve(hessian, gradient)
    gradient, precision = derivatives(flat[:N_TIME], flat[N_TIME])
    step = np.linalg.solve(precision, gradient)
    assert np.sqrt(step @ precision @ step) < 1e-8, "reference Newton stalled"
    return {"amp": flat[:N_TIME], "centre": flat[N_TIME]}, precision


def _basis_map(observed, *, sigma, gain_prior=GAIN_PRIOR, coeff_prior=COEFF_PRIOR):
    """The basis model's exact joint MAP and the posterior precision there.

    ``mu[t, f] = gain[t] * (T[t, f] + tone[f])`` with ``T = TIME_BASIS @ c @
    FREQ_BASIS.T``, bilinear, so its derivatives are written out by hand and
    Newton is run in NumPy float64 from the truth, with step halving. Shares no
    code with the plan and leaves the process-global x64 flag alone. Returns
    ``({name: array}, precision)``, the precision over ``gain`` then the
    row-major ``t_coeff``.
    """
    data = np.asarray(observed, np.float64)
    time_basis = np.asarray(TIME_BASIS, np.float64)
    freq_basis = np.asarray(FREQ_BASIS, np.float64)
    tone = np.zeros(N_FREQ)
    tone[TONE_CHANNEL] = TONE_KELVIN
    shape = np.shape(COEFF0)
    n_gain = N_TIME

    def normal(prior, size):
        loc = np.broadcast_to(np.asarray(prior.loc, np.float64), prior.batch_shape)
        scale = np.broadcast_to(np.asarray(prior.scale, np.float64), prior.batch_shape)
        return loc.reshape(size), scale.reshape(size)

    gain_loc, gain_scale = normal(gain_prior, n_gain)
    coeff_loc, coeff_scale = normal(coeff_prior, int(np.prod(shape)))
    loc = np.concatenate([gain_loc, coeff_loc])
    prior_precision = 1.0 / np.concatenate([gain_scale, coeff_scale]) ** 2
    # d T[t, f] / d c[i, j] = TIME_BASIS[t, i] * FREQ_BASIS[f, j], as (t, f, i*j)
    basis = np.einsum("ti,fj->tfij", time_basis, freq_basis).reshape(
        N_TIME, N_FREQ, -1
    )

    def objective(theta):
        gain, coeff = theta[:n_gain], theta[n_gain:].reshape(shape)
        signal = time_basis @ coeff @ freq_basis.T + tone[None, :]
        residual = (data - gain[:, None] * signal) / sigma
        return 0.5 * (np.sum(residual**2) + np.sum(prior_precision * (theta - loc) ** 2))

    def derivatives(theta):
        gain, coeff = theta[:n_gain], theta[n_gain:].reshape(shape)
        signal = time_basis @ coeff @ freq_basis.T + tone[None, :]
        residual = (data - gain[:, None] * signal) / sigma
        jacobian = np.zeros((N_TIME, N_FREQ, theta.size))
        jacobian[np.arange(N_TIME), :, np.arange(N_TIME)] = signal
        jacobian[:, :, n_gain:] = gain[:, None, None] * basis
        jacobian = jacobian.reshape(N_TIME * N_FREQ, theta.size)
        gradient = (
            -jacobian.T @ residual.ravel() / sigma + prior_precision * (theta - loc)
        )
        hessian = jacobian.T @ jacobian / sigma**2 + np.diag(prior_precision)
        # d2 mu[t, f] / d gain[t] d c[i, j] = basis[t, f, ij]
        cross = -np.einsum("tf,tfk->tk", residual, basis) / sigma
        hessian[:n_gain, n_gain:] += cross
        hessian[n_gain:, :n_gain] += cross.T
        return gradient, hessian

    theta = np.concatenate([np.asarray(GAIN0, np.float64),
                            np.asarray(COEFF0, np.float64).ravel()])
    for _ in range(100):
        gradient, hessian = derivatives(theta)
        step = -np.linalg.solve(hessian, gradient)
        length = 1.0
        while objective(theta + length * step) > objective(theta) and length > 1e-8:
            length /= 2.0
        theta = theta + length * step
    gradient, precision = derivatives(theta)
    step = np.linalg.solve(precision, gradient)
    assert np.sqrt(step @ precision @ step) < 1e-6, "reference Newton stalled"
    return {"gain": theta[:n_gain], "t_coeff": theta[n_gain:].reshape(shape)}, precision


def _posterior_sigmas_from(estimate, exact, precision):
    """Mahalanobis distance of ``estimate.values`` from ``exact``, in posterior sigma."""
    got = np.concatenate(
        [np.ravel(np.asarray(estimate.values[name], np.float64)) for name in exact]
    )
    want = np.concatenate([np.ravel(value) for value in exact.values()])
    residual = got - want
    return float(np.sqrt(residual @ precision @ residual))


# ------------------------------------------------------------ Block declaring --


class TestBlockDeclaration:
    def test_a_block_holds_names_in_the_callers_order(self):
        block = Block("t_nw", "t_ant")
        assert block.names == ("t_nw", "t_ant")
        assert block.steps is None and block.engine is None
        assert block.label == "('t_nw', 't_ant')"

    def test_an_empty_block_is_refused(self):
        """An empty block runs every sweep and changes nothing, so a plan
        holding one converges while its partition covers less than it claims."""
        with pytest.raises(ParameterSpaceError, match="at least one latent name"):
            Block()

    def test_a_non_string_member_is_refused(self):
        """Blocks are declared over NAMES. A Latent object here would fail much
        later, inside the partition check, blaming the space."""
        with pytest.raises(ParameterSpaceError, match="latent NAMES"):
            Block("gain", Latent("t_ant", init=jnp.zeros(3)))

    def test_a_repeated_member_is_refused(self):
        with pytest.raises(ParameterSpaceError, match="more than once"):
            Block("gain", "t_ant", "gain")

    def test_an_unknown_engine_is_refused(self):
        with pytest.raises(ParameterSpaceError, match="the engines are"):
            Block("gain", engine="nuts")

    @pytest.mark.parametrize("steps", [0, -3, 2.5, "many", True])
    def test_a_non_positive_step_count_is_refused(self, steps):
        """steps=0 leaves the block at its current value every sweep — a latent
        excluded from the inference while the partition still reports it
        covered. ``True`` is an int in Python and is not a step count."""
        with pytest.raises(ParameterSpaceError, match="positive int"):
            Block("beam_fwhm", steps=steps)


# ------------------------------------------------------------ the partition --


class TestPartition:
    def test_a_plan_with_no_blocks_is_refused(self):
        with pytest.raises(ParameterSpaceError, match="at least one Block"):
            SamplingPlan(basis_space())

    def test_a_block_naming_an_undeclared_latent_is_refused(self):
        with pytest.raises(ParameterSpaceError, match="does not declare"):
            SamplingPlan(basis_space(), Block("gain"), Block("t_coeff"), Block("nope"))

    def test_the_refusal_lists_what_the_space_does_declare(self):
        with pytest.raises(ParameterSpaceError, match=r"\['gain', 't_coeff'\]"):
            SamplingPlan(basis_space(), Block("gain", "nope"))

    def test_a_latent_in_two_blocks_is_refused_by_name(self):
        """Both blocks would run each sweep, and the second would be solving a
        conditional the first had just invalidated."""
        with pytest.raises(ParameterSpaceError, match="'gain' is in more than one block"):
            SamplingPlan(basis_space(), Block("gain"), Block("gain", "t_coeff"))

    def test_a_latent_in_no_block_is_refused_by_name(self):
        """The dangerous one: an omitted latent is frozen at its init for the
        whole run, the sweep converges, and nothing reports it."""
        with pytest.raises(ParameterSpaceError, match=r"does not cover latent\(s\) \['t_coeff'\]"):
            SamplingPlan(basis_space(), Block("gain"))

    def test_a_complete_partition_is_accepted_in_either_grouping(self):
        one_each = SamplingPlan(basis_space(), Block("gain"), Block("t_coeff"))
        assert set(one_each.engines) == {("gain",), ("t_coeff",)}
        # ... and the same latents in ONE block is also a complete partition,
        # though this particular pair is bilinear and refused later, at the
        # linearity check, which is a different question.
        together = SamplingPlan(basis_space(), Block("gain", "t_coeff"))
        assert set(together.engines) == {("gain", "t_coeff")}

    def test_the_repr_names_the_blocks_and_their_engines(self):
        plan = SamplingPlan(basis_space(), Block("gain"), Block("t_coeff"))
        assert "conjugate" in repr(plan)
        assert "'gain'" in repr(plan)


# ------------------------------------------------------ deriving the engine --


class TestEngineDerivation:
    """``Latent(..., linear=True)`` already says which exit a latent takes, so a
    Block does not restate it. An explicit engine is an override."""

    def test_an_all_linear_block_derives_the_conjugate_engine(self):
        plan = SamplingPlan(basis_space(), Block("gain"), Block("t_coeff"))
        assert plan.engines == {("gain",): CONJUGATE, ("t_coeff",): CONJUGATE}

    def test_a_block_with_no_linear_member_derives_the_gradient_engine(self):
        plan = SamplingPlan(line_space(), Block("amp"), Block("centre"))
        assert plan.engines == {("amp",): CONJUGATE, ("centre",): GRADIENT}

    def test_a_MIXED_block_cannot_be_derived_and_is_refused(self):
        """A conjugate solve needs the whole block affine; a gradient step
        throws away the linear members' structure entirely. Guessing either way
        is a decision the caller has to make."""
        with pytest.raises(ParameterSpaceError, match="mixes declared-linear"):
            SamplingPlan(line_space(), Block("amp", "centre"))

    def test_the_refusal_names_which_members_are_which(self):
        with pytest.raises(ParameterSpaceError, match=r"\['amp'\].*\['centre'\]"):
            SamplingPlan(line_space(), Block("amp", "centre"))

    def test_a_mixed_block_may_be_DOWNGRADED_to_gradient_explicitly(self):
        """The legitimate override, and the only reason engine= exists."""
        plan = SamplingPlan(line_space(), Block("amp", "centre", engine=GRADIENT))
        assert plan.engines == {("amp", "centre"): GRADIENT}

    def test_an_all_linear_block_may_also_be_downgraded(self):
        plan = SamplingPlan(
            basis_space(), Block("gain", engine=GRADIENT), Block("t_coeff")
        )
        assert plan.engines == {("gain",): GRADIENT, ("t_coeff",): CONJUGATE}

    def test_a_block_cannot_be_UPGRADED_to_conjugate(self):
        """The claim that the prediction is affine in a latent belongs in the
        Latent declaration, where check_linearity verifies it — not in a plan
        that asserts it."""
        with pytest.raises(ParameterSpaceError, match="not declared linear=True"):
            SamplingPlan(line_space(), Block("amp"), Block("centre", engine=CONJUGATE))

    def test_steps_on_a_conjugate_block_is_refused_rather_than_ignored(self):
        """A conjugate solve has no inner steps, so steps= would be silently
        dropped — and it looks exactly like a knob that did something."""
        with pytest.raises(ParameterSpaceError, match="no inner "):
            SamplingPlan(basis_space(), Block("gain", steps=20), Block("t_coeff"))

    def test_steps_is_accepted_once_the_block_is_downgraded(self):
        plan = SamplingPlan(
            basis_space(), Block("gain", steps=20, engine=GRADIENT), Block("t_coeff")
        )
        assert plan.engines[("gain",)] == GRADIENT


# ------------------------------------------------------------- the headline --


class TestTheMotivatingCase:
    """The three things this whole piece exists to demonstrate."""

    def test_the_free_per_cell_model_is_REFUSED_and_the_direction_is_NAMED(self, state):
        """Item one. Every per-block guard passes on this model — that is the
        measured starting point — and the plan refuses it before a sweep runs,
        naming the degenerate direction as a combination of latents rather than
        as an index into an anonymous vector.
        """
        space, pipeline = free_space(), make_pipeline()
        observed = observed_of(space, pipeline, {"gain": GAIN0, "t_ant": T_ANT0})
        plan = SamplingPlan(space, Block("gain"), Block("t_ant"))

        with pytest.raises(ParameterSpaceError) as caught:
            plan.estimate(pipeline, state, observed, noise=NOISE)
        message = str(caught.value)

        assert "nullity 6 of 60 parameters" in message, message
        # named, and named as BOTH latents the degeneracy mixes
        assert "direction 0:" in message
        assert "gain" in message and "t_ant" in message
        assert "0.50" in message, message
        # ... and it says how many it did not print
        assert "and 2 more" in message, message

        # the same refusal at the OTHER exit, which is the one people expect to
        # need it less and which needs it more
        with pytest.raises(ParameterSpaceError, match="nullity 6"):
            plan.sample(
                pipeline, state, observed, noise=NOISE,
                key=jax.random.key(0), n_sweeps=8,
            )

    def test_the_tone_buys_nothing_here_which_is_why_the_check_is_the_repair(self, state):
        """The free-per-cell cell at the tone's channel absorbs the gain sample
        by sample, so an identifying tone does not rescue this parameterization
        and only a re-parameterization does. Pinned so the refusal above cannot
        be mistaken for something a brighter tone would fix."""
        space = free_space()
        with_tone = identifiability(space, make_pipeline(TONE_KELVIN), state)
        without = identifiability(space, make_pipeline(0.0), state)
        assert with_tone.nullity == without.nullity == N_TIME

    def test_the_basis_model_runs_and_both_exits_agree_with_the_truth(
        self, basis_setup, state
    ):
        """Item two, and the whole thesis in one test: a point estimate and a
        posterior sample are two exits from ONE workflow. The same plan, the
        same partition, the same conditioning — and the mean of the draws lands
        on the point estimate, which lands on the truth.
        """
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))

        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=200, solve_guard=None
        )
        draws = plan.sample(
            pipeline, state, observed, noise=NOISE, key=jax.random.key(0),
            n_sweeps=200, warmup=100, solve_guard=None,
        )

        assert est.diagnostics.converged is True
        assert draws.diagnostics.rhat < 1.05, draws.diagnostics.rhat
        assert draws.diagnostics.converged is True

        # the estimate is on the truth
        gain_error = float(jnp.max(jnp.abs(est.values["gain"] - GAIN0)))
        assert gain_error < 1e-3, gain_error
        recovered = TIME_BASIS @ est.values["t_coeff"] @ FREQ_BASIS.T
        assert float(jnp.sqrt(jnp.mean((recovered - T_ANT0) ** 2))) < 1.0

        # ... and so is the posterior mean, to within its own scatter
        for name in ("gain", "t_coeff"):
            gap = jnp.abs(draws.mean[name] - TRUTH[name])
            assert jnp.all(gap < 5.0 * draws.std[name] + 1e-6), (name, gap)
            assert jnp.all(jnp.abs(draws.mean[name] - est.values[name])
                           < 5.0 * draws.std[name] + 1e-6), name

        # the posterior has real width — a draw that came back as the mean
        # would satisfy every assertion above and be wrong about everything
        assert float(jnp.min(draws.std["gain"])) > 0.0

    def test_the_JOINT_chi2_sees_what_every_per_block_residual_misses(
        self, basis_setup, state
    ):
        """Item three, and the evidence the piece is worth having.

        Three sweeps in, the joint chi-squared is still falling by tens of
        millions while EVERY block's own CG residual has been converged since
        sweep one. A per-block residual is computed from the block; it cannot
        see across the partition, and this is what that costs.

        The verdict is taken on the joint negative log posterior, the quantity
        the sweep minimises (T-002 A5-1); the refusal names it and still
        reports the joint chi-squared beside it.
        """
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))

        with pytest.raises(ParameterSpaceError) as caught:
            plan.estimate(
                pipeline, state, observed, noise=NOISE, max_iter=3, solve_guard=None
            )
        message = str(caught.value)
        assert "did not converge" in message
        assert "JOINT negative log posterior is still changing" in message, message
        assert "chi2 = " in message, message

        # and the counter-evidence, in the message itself: the per-block number
        # that reads converged the whole way down
        short = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=3, tol=None,
            solve_guard=None,
        )
        assert short.diagnostics.converged is None
        assert max(short.diagnostics.block_residuals.values()) < 1e-5, (
            short.diagnostics.block_residuals
        )
        # ... while the joint chi-squared has not remotely settled
        trace = short.diagnostics.chi2
        assert trace[-2] - trace[-1] > 1e3, trace

        # the same plan, given the sweeps it needs, does converge — so the
        # refusal above is about the SWEEP COUNT and not about the model
        full = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=200, solve_guard=None
        )
        assert full.diagnostics.converged is True
        assert full.diagnostics.sweeps > 3
        assert full.diagnostics.chi2[-1] < trace[-1]


# ------------------------------------------------------- identifiability knob --


class TestIdentifiabilityCadence:
    def test_an_unknown_cadence_is_refused_rather_than_guessed(self, basis_setup, state):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        with pytest.raises(ParameterSpaceError, match="check_identifiability"):
            plan.estimate(
                pipeline, state, observed, noise=NOISE, check_identifiability="sometimes"
            )
        with pytest.raises(ParameterSpaceError, match="check_identifiability"):
            plan.sample(
                pipeline, state, observed, noise=NOISE, key=jax.random.key(0),
                n_sweeps=8, check_identifiability=True,
            )

    def test_once_runs_the_check_and_reports_it(self, basis_setup, state):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=4, tol=None,
            check_identifiability=CHECK_ONCE, solve_guard=None,
        )
        assert est.diagnostics.identifiability is not None
        assert est.diagnostics.identifiability.nullity == 0
        assert est.diagnostics.identifiability.n_par == 18

    def test_False_skips_it_and_reports_nothing(self, basis_setup, state):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=2, tol=None,
            check_identifiability=False, solve_guard=None,
        )
        assert est.diagnostics.identifiability is None

    def test_False_is_how_a_degenerate_model_can_be_run_deliberately(self, state):
        """The escape hatch has to actually work, or the guard is a wall. It is
        also the only route for a complex latent (which the rank test cannot
        analyse) and for a block too large to form a Jacobian of."""
        space, pipeline = free_space(), make_pipeline()
        observed = observed_of(space, pipeline, {"gain": GAIN0, "t_ant": T_ANT0})
        plan = SamplingPlan(space, Block("gain"), Block("t_ant"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=2, tol=None,
            check_identifiability=False, solve_guard=None,
        )
        assert est.diagnostics.identifiability is None
        assert set(est.values) == {"gain", "t_ant"}

    def test_each_sweep_catches_a_degeneracy_that_ONCE_cannot(self, state):
        """Identifiability is a LOCAL property of a nonlinear model, so a check
        only at the starting values misses a degeneracy that opens up at the
        parameters the run actually reaches.

        Here the run starts at an identified point and walks to a gain of zero,
        where the temperature block stops reaching the data at all. ``"once"``
        signs the model off; ``"each_sweep"`` refuses it.
        """
        # A gain pinned to zero is where the degeneracy lives; a plan that
        # updates only the temperature walks straight into it.
        space = ParameterSpace(
            latents=[
                Latent("gain", init=GAIN_GUESS, prior=GAIN_PRIOR, linear=True),
                Latent("t_coeff", init=COEFF_GUESS, prior=COEFF_PRIOR, linear=True),
            ],
            bindings=[
                Bind("gain", into=lambda p: p["gain"].gain),
                Bind(
                    "t_coeff",
                    into=lambda p: p["t_ant"].t_ant,
                    fn=lambda c: TIME_BASIS @ c @ FREQ_BASIS.T,
                ),
            ],
        )
        pipeline = make_pipeline()
        here = identifiability(space, pipeline, STATE)
        there = identifiability(space, pipeline, STATE, at={"gain": jnp.zeros(N_TIME)})
        assert here.nullity == 0 and there.nullity == 12, (here.nullity, there.nullity)

        # The plan's own reading of the same two points: "once" looks only at
        # the first, "each_sweep" at every one.
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        assert plan._identifiable(  # the check itself, at the declared start
            plan._prepare(
                pipeline, STATE, observed_of(space, pipeline, TRUTH), NOISE,
                CHECK_ONCE, "test",
            )[0],
            space.initial_values(),
            "test",
        ).nullity == 0
        with pytest.raises(ParameterSpaceError, match="nullity 12"):
            plan._identifiable(
                plan._prepare(
                    pipeline, STATE, observed_of(space, pipeline, TRUTH), NOISE,
                    CHECK_EACH_SWEEP, "test",
                )[0],
                {**space.initial_values(), "gain": jnp.zeros(N_TIME)},
                "test",
            )

    def test_each_sweep_runs_the_check_every_sweep(self, basis_setup, state, monkeypatch):
        """Cheap for a small model and strictly more informative — but only if
        it really happens more than once."""
        import rheplicant.inference.plan as plan_module

        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        calls = []
        real = plan_module.identifiability
        monkeypatch.setattr(
            plan_module,
            "identifiability",
            lambda *a, **k: (calls.append(1), real(*a, **k))[1],
        )
        plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=4, tol=None,
            check_identifiability=CHECK_EACH_SWEEP, solve_guard=None,
        )
        assert len(calls) == 4, calls

        calls.clear()
        plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=4, tol=None,
            check_identifiability=CHECK_ONCE, solve_guard=None,
        )
        assert len(calls) == 1, calls


# --------------------------------------------------------- convergence knobs --


class TestConvergence:
    def test_tol_None_returns_an_answer_with_no_convergence_claim(
        self, basis_setup, state
    ):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=2, tol=None,
            solve_guard=None,
        )
        assert est.diagnostics.converged is None
        assert est.diagnostics.sweeps == 2
        assert est.diagnostics.chi2.shape == (3,)

    def test_an_increase_of_chi_squared_is_not_a_stop(self, state):
        """A chi-squared rise is not convergence, and the run goes on past it.

        Until T-002 the rule was a DECREASE of the joint chi-squared, so any
        sweep on which chi-squared rose counted as converged. Here a tight
        prior pulls the gain away from the truth, and chi-squared rises by
        half its value at sweep 58 while the objective is still falling: the
        old rule stopped there, 3.6 posterior sigma from the MAP (measured).

        The relative change test then stopped at sweep 73, 0.74 sigma off in
        this module's float32. The Newton decrement refuses that point and
        the run continues: measured, it stops at sweep 95, 0.069 sigma from
        the MAP, having tightened the conjugate solves twice along the way.
        """
        gain_prior = dist.Normal(jnp.ones(N_TIME), 0.01)
        space = basis_space(gain_prior=gain_prior)
        pipeline = make_pipeline()
        observed = observed_of(space, pipeline, TRUTH)
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        common = {"noise": NOISE, "max_iter": 200, "solve_guard": None}

        free = plan.estimate(pipeline, state, observed, tol=None, **common)
        chi2, objective = free.diagnostics.chi2, free.diagnostics.objective
        rise = np.diff(chi2) / np.maximum(np.abs(chi2[1:]), 1.0)
        rose = [sweep for sweep in range(MIN_SWEEPS, 200) if rise[sweep - 1] > 0.1]
        assert rose, "the fixture must make chi-squared rise, or this test is vacuous"
        # the objective was still falling where chi-squared first rose
        assert objective[rose[0] - 1] - objective[rose[0]] > 1.0, objective[rose[0]]

        estimate = plan.estimate(pipeline, state, observed, **common)
        diagnostics = estimate.diagnostics
        assert diagnostics.converged is True
        assert diagnostics.sweeps > rose[0], (diagnostics.sweeps, rose[0])
        exact, precision = _basis_map(observed, sigma=NOISE, gain_prior=gain_prior)
        distance = _posterior_sigmas_from(estimate, exact, precision)
        assert distance < 0.1, (distance, diagnostics.sweeps)

    def test_an_inexact_inner_solve_is_tightened_until_it_certifies(
        self, basis_setup, state
    ):
        """The second review's MEDIUM, in float32. At noise 0.30 the sweep's
        fixed point at the default ``solve_tol = 1e-6`` is 0.113 posterior
        sigma from the MAP (the reviewer's float64 measurement), so a stop
        there cannot be certified. The run tightens the closed-form blocks'
        CG when the objective rises beyond its resolution or the decrement
        refuses a candidate, down to the float32 floor of two machine
        epsilons; measured, it converges at sweep 111, 0.072 sigma off.
        Without the tightening it exhausts 3000 sweeps and refuses.
        """
        space, pipeline, _ = basis_setup
        observed = observed_of(space, pipeline, TRUTH)
        estimate = SamplingPlan(space, Block("gain"), Block("t_coeff")).estimate(
            pipeline, state, observed, noise=0.30, max_iter=3000, solve_guard=None
        )
        diagnostics = estimate.diagnostics
        assert diagnostics.converged is True
        assert diagnostics.solve_tol < 1e-6
        exact, precision = _basis_map(observed, sigma=0.30)
        distance = _posterior_sigmas_from(estimate, exact, precision)
        assert distance < 0.1, (distance, diagnostics.sweeps)

    def test_the_stop_rule_counts_changes_in_both_directions(self):
        """``_settled`` on hand-made traces: a rise beyond the tolerance is not
        settled, which the old one-sided test would have called settled."""
        tol = 1e-6
        assert _settled([10.0, 10.0, 10.0], tol)
        assert _settled([10.0, 10.0 + 5e-6, 10.0], tol)
        assert not _settled([10.0, 10.5, 11.0], tol)       # rising
        assert not _settled([11.0, 10.5, 10.0], tol)       # falling
        assert not _settled([10.0, 10.0, 11.0], tol)       # only one change settled
        assert not _settled([10.0, 10.0], tol)             # one change is not two

    def test_the_floor_is_recorded_and_scales_with_the_dtype(self, basis_setup, state):
        """The applied tolerance is ``max(tol, OBJECTIVE_FLOOR_EPS * eps)``.

        This module runs in float32, where the default ``tol = 1e-8`` is below
        the objective's epsilon; the floor is what lets a float32 run stop at
        all (see :data:`OBJECTIVE_FLOOR_EPS`). A ``tol`` above the floor is
        applied as given.
        """
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        common = {"noise": NOISE, "max_iter": 200, "solve_guard": None}
        floored = plan.estimate(pipeline, state, observed, **common)
        eps = float(np.finfo(np.float32).eps)
        assert floored.diagnostics.effective_tol == OBJECTIVE_FLOOR_EPS * eps
        loose = plan.estimate(pipeline, state, observed, tol=1e-3, **common)
        assert loose.diagnostics.effective_tol == 1e-3
        free = plan.estimate(pipeline, state, observed, tol=None, max_iter=2,
                             noise=NOISE, solve_guard=None)
        assert free.diagnostics.effective_tol is None

    def test_the_float32_basis_model_converges_onto_the_float64_map(
        self, basis_setup, state
    ):
        """The motivating model, in this module's float32, within 0.1 posterior
        sigma of its MAP computed in float64.

        Measured: the run stops at sweep 94, 0.079 sigma from the MAP. Without
        the floor it never stops (the objective's plateau moves by tens of
        float32 ulps a sweep); with the old chi-squared rule it stopped at
        sweep 95, 0.080 sigma.
        """
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=200, solve_guard=None
        )
        assert est.diagnostics.converged is True
        exact, precision = _basis_map(observed, sigma=NOISE)
        distance = _posterior_sigmas_from(est, exact, precision)
        assert distance < 0.1, (distance, est.diagnostics.sweeps)

    def test_a_min_sweeps_above_the_cap_is_refused(self, basis_setup, state):
        """It would make the test unreachable, so every run would exhaust
        max_iter and refuse — including one that converged at sweep two."""
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        with pytest.raises(ParameterSpaceError, match="min_sweeps <= max_iter"):
            plan.estimate(
                pipeline, state, observed, noise=NOISE, max_iter=5, min_sweeps=6
            )

    @pytest.mark.parametrize("max_iter", [0, -1, 2.0])
    def test_a_nonsense_sweep_cap_is_refused(self, basis_setup, state, max_iter):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        with pytest.raises(ParameterSpaceError, match="max_iter >= 1"):
            plan.estimate(pipeline, state, observed, noise=NOISE, max_iter=max_iter)


# --------------------------------------------------------------- sample knobs --


class TestSampleGuards:
    @pytest.fixture
    def plan_and_data(self, basis_setup):
        space, pipeline, observed = basis_setup
        return SamplingPlan(space, Block("gain"), Block("t_coeff")), pipeline, observed

    @pytest.mark.parametrize("n_sweeps", [0, -4, 3.5])
    def test_a_nonsense_sweep_count_is_refused(self, plan_and_data, state, n_sweeps):
        plan, pipeline, observed = plan_and_data
        with pytest.raises(ParameterSpaceError, match="n_sweeps >= 1"):
            plan.sample(
                pipeline, state, observed, noise=NOISE, key=jax.random.key(0),
                n_sweeps=n_sweeps,
            )

    def test_a_negative_warmup_is_refused(self, plan_and_data, state):
        """It would slice the chi-squared trace from the end and report r_hat
        over draws that were never kept."""
        plan, pipeline, observed = plan_and_data
        with pytest.raises(ParameterSpaceError, match="warmup >= 0"):
            plan.sample(
                pipeline, state, observed, noise=NOISE, key=jax.random.key(0),
                n_sweeps=10, warmup=-2,
            )

    def test_too_few_kept_draws_is_refused_because_r_hat_is_undefined(
        self, plan_and_data, state
    ):
        """A run whose only convergence evidence is undefined is exactly the
        silent answer this plan exists to refuse."""
        plan, pipeline, observed = plan_and_data
        with pytest.raises(ParameterSpaceError, match=f"at least {MIN_DRAWS}"):
            plan.sample(
                pipeline, state, observed, noise=NOISE, key=jax.random.key(0),
                n_sweeps=10, warmup=8,
            )

    def test_warmup_defaults_to_half_the_sweeps(self, plan_and_data, state):
        plan, pipeline, observed = plan_and_data
        draws = plan.sample(
            pipeline, state, observed, noise=NOISE, key=jax.random.key(0),
            n_sweeps=11, solve_guard=None,
        )
        assert draws.diagnostics.warmup == 5
        assert draws.n_draw == 6
        assert draws.diagnostics.chi2.shape == (11,)

    def test_a_gradient_block_with_no_declared_prior_cannot_be_SAMPLED(self, state):
        """The potential is flat in a prior-free latent, so the chain wanders
        off with no diagnostic reporting anything wrong. Same rule, and the same
        reason, as to_numpyro_model's."""
        space = line_space(centre_prior=None)
        pipeline = make_line_pipeline()
        observed = observed_of(space, pipeline, LINE_TRUTH)
        plan = SamplingPlan(space, Block("amp"), Block("centre"))
        with pytest.raises(ParameterSpaceError, match=r"\['centre'\] have none"):
            plan.sample(
                pipeline, state, observed, noise=NOISE, key=jax.random.key(0),
                n_sweeps=8,
            )
        # ... while the point estimate, for which a free parameter is
        # meaningful, is not refused.
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=2, tol=None,
            solve_guard=None,
        )
        assert set(est.values) == {"amp", "centre"}


# ------------------------------------------------------------- the two exits --


class TestSharedSeam:
    """What the two exits share is the implementation, not the signature."""

    def test_both_exits_refuse_a_mis_shaped_observed(self, basis_setup, state):
        """A BROADCASTABLE mismatch on purpose: (1, 9) against a (6, 9)
        prediction subtracts cleanly, minimizes a different problem, and reports
        a small converged chi-squared for it. A shape that merely fails to
        broadcast would be caught by jax anyway and would test nothing.

        Matched on the PLAN's own wording rather than on the shared phrase.
        wiener_solve refuses the same data one layer down, saying "this block",
        so a test that accepted either message passes with the plan's guard
        deleted — which is how it was written first, and what mutation caught.
        The next test is why the plan's guard is not redundant.
        """
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        wrong = observed[:1, :]
        assert jnp.shape(wrong - observed) == jnp.shape(observed), "must broadcast"
        with pytest.raises(ParameterSpaceError, match="this plan's model predicts"):
            plan.estimate(
                pipeline, state, wrong, noise=NOISE, max_iter=2, tol=None,
                solve_guard=None,
            )
        with pytest.raises(ParameterSpaceError, match="this plan's model predicts"):
            plan.sample(
                pipeline, state, wrong, noise=NOISE, key=jax.random.key(0), n_sweeps=8,
                solve_guard=None,
            )

    def test_an_ALL_GRADIENT_plan_has_no_other_shape_guard_at_all(self, state):
        """Why the plan checks ``observed`` itself instead of leaving it to the
        block solves.

        A conjugate block passes its data to wiener_solve, which refuses a
        mis-shaped one. A gradient block passes it to nothing: the joint
        chi-squared subtracts the two arrays directly, so a broadcastable
        mismatch is a finite, converged, wrong answer with no symptom anywhere.
        This is the plan's guard doing the only work being done.
        """
        space, pipeline = line_space(), make_line_pipeline()
        observed = observed_of(space, pipeline, LINE_TRUTH)
        plan = SamplingPlan(space, Block("amp", "centre", engine=GRADIENT, steps=2))
        assert set(plan.engines.values()) == {GRADIENT}, plan.engines

        wrong = observed[:1, :]
        assert jnp.shape(wrong - observed) == jnp.shape(observed), "must broadcast"
        with pytest.raises(ParameterSpaceError, match="this plan's model predicts"):
            plan.estimate(
                pipeline, state, wrong, noise=NOISE, max_iter=2, tol=None,
                check_identifiability=False,
            )
        with pytest.raises(ParameterSpaceError, match="this plan's model predicts"):
            plan.sample(
                pipeline, state, wrong, noise=NOISE, key=jax.random.key(0), n_sweeps=8,
                check_identifiability=False,
            )

    def test_both_exits_refuse_a_BILINEAR_group(self, basis_setup, state):
        """gain and t_coeff are each affine given the other and bilinear
        together, so one block over both is not a conjugate problem at all.
        Inherited from check_linearity, which probes the JOINT map — and which
        no per-latent check could have caught."""
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain", "t_coeff"))
        with pytest.raises(ParameterSpaceError, match="not affine in them JOINTLY"):
            plan.estimate(
                pipeline, state, observed, noise=NOISE, max_iter=2, tol=None,
                solve_guard=None,
            )
        with pytest.raises(ParameterSpaceError, match="not affine in them JOINTLY"):
            plan.sample(
                pipeline, state, observed, noise=NOISE, key=jax.random.key(0), n_sweeps=8,
                solve_guard=None,
            )

    def test_sample_cannot_be_called_without_a_key(self, basis_setup, state):
        """The invalid combination is unrepresentable rather than validated:
        'asked to sample and forgot the key' is a TypeError from Python itself,
        not a runtime check that could be forgotten."""
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        with pytest.raises(TypeError, match="key"):
            plan.sample(pipeline, state, observed, noise=NOISE, n_sweeps=8)

    def test_estimate_has_no_key_to_pass(self, basis_setup, state):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        with pytest.raises(TypeError):
            plan.estimate(
                pipeline, state, observed, noise=NOISE, key=jax.random.key(0)
            )

    def test_a_bare_sigma_and_a_noise_model_are_the_same_run(self, basis_setup, state):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        common = {"max_iter": 6, "tol": None, "solve_guard": None}
        bare = plan.estimate(pipeline, state, observed, noise=NOISE, **common)
        wrapped = plan.estimate(
            pipeline, state, observed,
            noise=HomoscedasticNoise(jnp.asarray(NOISE)), **common,
        )
        assert jnp.allclose(bare.values["gain"], wrapped.values["gain"])
        assert bare.diagnostics.noise_depends_on_prediction is False

    def test_a_flagged_sample_contributes_nothing_to_the_JOINT_chi2(
        self, basis_setup, state
    ):
        """A flagged sample was not observed, so it must inform nothing — and it
        arrives at the monitor as an infinite sigma, where the naive residual is
        ``0 * inf`` and the whole convergence trace becomes NaN.

        The evidence is a comparison, not a finiteness check: corrupting only
        the flagged cells must leave the chi-squared trace bitwise unchanged.
        A monitor that let them through would move.
        """
        space, pipeline, observed = basis_setup
        flags = jnp.zeros(jnp.shape(observed), dtype=bool).at[2, 5].set(True)
        flags = flags.at[4, 1].set(True)
        noise = FlaggedNoise(HomoscedasticNoise(jnp.asarray(NOISE)), flags)
        corrupted = observed.at[2, 5].set(1e9).at[4, 1].set(-1e9)
        common = {
            "noise": noise, "max_iter": 4, "tol": None, "solve_guard": None,
            "check_identifiability": False,
        }

        clean = plan_estimate = SamplingPlan(
            space, Block("gain"), Block("t_coeff")
        ).estimate(pipeline, state, observed, **common)
        dirty = SamplingPlan(space, Block("gain"), Block("t_coeff")).estimate(
            pipeline, state, corrupted, **common
        )
        assert np.all(np.isfinite(clean.diagnostics.chi2)), clean.diagnostics.chi2
        assert np.array_equal(plan_estimate.diagnostics.chi2, dirty.diagnostics.chi2), (
            clean.diagnostics.chi2, dirty.diagnostics.chi2
        )
        # ... and the run really did depend on the unflagged data, so the
        # comparison above is not two runs that both ignored everything.
        elsewhere = observed.at[0, 0].set(observed[0, 0] + 5e3)
        moved = SamplingPlan(space, Block("gain"), Block("t_coeff")).estimate(
            pipeline, state, elsewhere, **common
        )
        assert not np.array_equal(clean.diagnostics.chi2, moved.diagnostics.chi2)

    def test_a_prediction_dependent_noise_model_is_recorded_as_such(
        self, basis_setup, state
    ):
        """The sweep IS the reweighting for a RadiometerNoise, so the plan does
        not nest iterative_gls — and the statistical consequence of freezing
        sigma inside each solve is recorded rather than hidden."""
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed,
            noise=RadiometerNoise(channel_width=1e6, integration_time=1.0, floor=1.0),
            max_iter=6, tol=None, solve_guard=None,
        )
        assert est.diagnostics.noise_depends_on_prediction is True


# ---------------------------------------------------------- result currency --


class TestResults:
    def test_an_estimate_is_keyed_by_latent_name(self, basis_setup, state):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=4, tol=None,
            solve_guard=None,
        )
        assert isinstance(est, Estimate)
        assert est.names == ("gain", "t_coeff")
        assert est.values["gain"].shape == (N_TIME,)
        assert est.values["t_coeff"].shape == (3, 4)

    def test_draws_are_stacked_per_latent_with_warmup_already_gone(
        self, basis_setup, state
    ):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        draws = plan.sample(
            pipeline, state, observed, noise=NOISE, key=jax.random.key(1),
            n_sweeps=14, warmup=4, solve_guard=None,
        )
        assert isinstance(draws, Draws)
        assert draws.names == ("gain", "t_coeff")
        assert draws.n_draw == 10
        assert draws.samples["gain"].shape == (10, N_TIME)
        assert draws.samples["t_coeff"].shape == (10, 3, 4)
        assert draws.mean["gain"].shape == (N_TIME,)
        assert draws.std["t_coeff"].shape == (3, 4)
        # the two latents have different shapes AND different sizes, so a stack
        # that carried the wrong latent's draws cannot pass here
        assert draws.samples["gain"].size != draws.samples["t_coeff"].size

    def test_both_results_expose_the_same_diagnostics_protocol(
        self, basis_setup, state
    ):
        """Two types, one currency. A caller can log or assert on a run without
        knowing which exit produced it."""
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=4, tol=None,
            solve_guard=None,
        )
        draws = plan.sample(
            pipeline, state, observed, noise=NOISE, key=jax.random.key(2),
            n_sweeps=10, warmup=4, solve_guard=None,
        )
        for result in (est, draws):
            assert set(result.names) == {"gain", "t_coeff"}
            assert result.diagnostics.engines == {
                ("gain",): CONJUGATE, ("t_coeff",): CONJUGATE
            }
            assert set(result.diagnostics.block_residuals) == {("gain",), ("t_coeff",)}
            assert result.diagnostics.chi2.ndim == 1

        # ... and what they do NOT share is what an answer IS
        assert est.diagnostics.warmup is None and est.diagnostics.rhat is None
        assert draws.diagnostics.warmup == 4 and draws.diagnostics.rhat is not None


# ------------------------------------------------------------- split r_hat --


class TestSplitRhat:
    def test_a_constant_trace_has_nothing_to_mix(self):
        assert split_rhat(np.full(20, 3.5)) == 1.0

    @pytest.mark.parametrize("size", [0, 1, 2, 3])
    def test_a_trace_shorter_than_MIN_DRAWS_is_refused_by_name(self, size):
        """Every one of these used to be an accident rather than an answer.

        Measured before the guard: 0 raised a bare ``ZeroDivisionError``, 1 a
        bare ``ValueError: all input arrays must have the same shape``, and 2
        and 3 RETURNED ``nan`` carrying only a numpy ``RuntimeWarning`` that
        nothing in this package surfaces. The nan is the dangerous one — it
        defeats both directions of a comparison-based guard, so a caller
        testing ``rhat <= rhat_max`` and one testing ``rhat > rhat_max`` both
        read an undefined diagnostic as the answer they were hoping for.

        ``SamplingPlan.sample`` enforced this minimum already; ``split_rhat``
        is public and exported and did not.
        """
        trace = np.arange(size, dtype=np.float64) * 1.7 + 1.0
        with pytest.raises(ParameterSpaceError, match=f"at least {MIN_DRAWS}"):
            split_rhat(trace)

    def test_MIN_DRAWS_itself_is_accepted_and_the_value_is_pinned(self):
        """The other branch of the guard: the boundary is inclusive, and the
        answer at it is a number rather than a nan.

        Two traces, because ``r_hat`` is invariant under an affine map of the
        trace — every four-draw straight ramp gives the same 1.5*sqrt(2), so a
        ramp alone would pin an accident of the fixture rather than the
        formula. The second is asymmetric in its halves' spread as well as
        their means.
        """
        ramp = split_rhat(np.array([1.0, 2.7, 4.4, 6.1]))
        assert ramp == pytest.approx(2.1213203435596424)

        uneven = split_rhat(np.array([1.0, 2.0, 5.0, 11.0]))
        assert uneven == pytest.approx(1.6684674955730434)
        assert uneven != pytest.approx(ramp)

    @pytest.mark.parametrize("size", [0, 1, 2, 3, 4, 5, 6, 7])
    def test_the_two_halves_are_the_first_and_the_LAST_n_over_2(self, size):
        """The slice, checked at the lengths ``MIN_DRAWS`` forbids as well as
        the ones it allows — deliberately bypassing ``split_rhat``, because a
        guard that makes a bug unreachable has not fixed it.

        ``values[-half:]`` is ``values[0:]`` — the WHOLE trace — when ``half``
        is 0, which is how a one-draw trace came to be compared against
        itself and reported as mismatched shapes.
        """
        values = np.arange(size, dtype=np.float64) * 1.7 + 1.0
        halves = _halves(values)
        half = size // 2

        assert halves.shape == (2, half)
        if half:
            np.testing.assert_array_equal(halves[0], values[:half])
            assert halves[1][0] == values[size - half]
            assert halves[1][-1] == values[-1]
        if size % 2 and half:
            assert values[half] not in halves.ravel().tolist()

    def test_sample_at_its_own_minimum_hands_split_rhat_a_trace_it_accepts(
        self, basis_setup, state
    ):
        """The two guards are the same number, so the exit that computes an
        r_hat cannot trip the one that refuses to compute it. Run at exactly
        MIN_DRAWS kept draws — one fewer and ``sample`` refuses first.
        """
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        draws = plan.sample(
            pipeline, state, observed, noise=NOISE, key=jax.random.key(5),
            n_sweeps=2 * MIN_DRAWS, warmup=MIN_DRAWS, solve_guard=None,
        )
        assert draws.diagnostics.chi2[MIN_DRAWS:].size == MIN_DRAWS
        assert np.isfinite(draws.diagnostics.rhat)

    def test_two_constant_halves_at_different_values_are_infinitely_unmixed(self):
        """A chain that moved once and stopped. Reported as inf rather than as
        a division by zero, which is the honest reading."""
        trace = np.concatenate([np.zeros(10), np.ones(10)])
        assert split_rhat(trace) == float("inf")

    def test_white_noise_is_reported_as_mixed(self):
        rng = np.random.default_rng(0)
        assert split_rhat(rng.normal(size=400)) < 1.05

    def test_a_drifting_trace_is_reported_as_unmixed(self):
        rng = np.random.default_rng(0)
        drift = np.linspace(0.0, 12.0, 400) + rng.normal(size=400)
        assert split_rhat(drift) > 1.5, split_rhat(drift)

    def test_it_is_the_diagnostic_the_run_reports(self, basis_setup, state):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        draws = plan.sample(
            pipeline, state, observed, noise=NOISE, key=jax.random.key(3),
            n_sweeps=12, warmup=4, solve_guard=None,
        )
        expected = split_rhat(draws.diagnostics.chi2[4:])
        assert draws.diagnostics.rhat == pytest.approx(expected)
        assert draws.diagnostics.converged is (expected <= 1.05)

    def test_rhat_max_is_the_callers_threshold(self, basis_setup, state):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        common = {
            "noise": NOISE, "key": jax.random.key(4), "n_sweeps": 12,
            "warmup": 4, "solve_guard": None,
        }
        strict = plan.sample(pipeline, state, observed, rhat_max=1.0, **common)
        loose = plan.sample(pipeline, state, observed, rhat_max=1e6, **common)
        assert strict.diagnostics.rhat == loose.diagnostics.rhat
        assert loose.diagnostics.converged is True
        assert strict.diagnostics.converged is (strict.diagnostics.rhat <= 1.0)


# -------------------------------------------------------- the gradient engine --


class TestGradientEngine:
    """One conjugate block and one gradient block in the same plan."""

    @pytest.fixture
    def line_setup(self):
        space, pipeline = line_space(), make_line_pipeline()
        return space, pipeline, observed_of(space, pipeline, LINE_TRUTH)

    def test_the_conditional_potential_carries_the_DECLARED_prior(self, line_setup):
        """The gradient engine's two exits share one potential, so the prior has
        to be in it: descend the likelihood alone and the point estimate and the
        draw target different distributions.

        Asserted as an exact identity rather than as an outcome, because a
        loose prior barely moves either answer — which is how this branch would
        ship pinned on the wrong side.
        """
        from rheplicant.inference.engines import Conditioning, conditional_potential

        space, pipeline, observed = line_setup
        forward, values0 = space.forward_fn(pipeline, STATE)
        cond = Conditioning(
            space=space, pipeline=pipeline, state_template=STATE, observed=observed,
            noise=HomoscedasticNoise(jnp.asarray(0.5)), forward=forward,
        )
        free = ParameterSpace(
            latents=[
                Latent("amp", init=values0["amp"], prior=AMP_PRIOR, linear=True),
                Latent("centre", init=values0["centre"]),
            ],
            bindings=list(space.bindings),
        )
        bare = Conditioning(
            space=free, pipeline=pipeline, state_template=STATE, observed=observed,
            noise=HomoscedasticNoise(jnp.asarray(0.5)), forward=forward,
        )

        probe = {"centre": jnp.array(0.42)}
        declared = conditional_potential(cond, ("centre",), values0)(probe)
        without = conditional_potential(bare, ("centre",), values0)(probe)
        expected = -jnp.sum(CENTRE_PRIOR.log_prob(probe["centre"]))
        # rel=1e-3, not tighter: this is a difference of two chi-squareds of
        # order 1e3 taken in float32, so ~1e-4 of cancellation noise is the
        # arithmetic and not the identity.
        assert float(declared - without) == pytest.approx(float(expected), rel=1e-3)
        assert float(expected) != 0.0, "the fixture's prior must actually bite"

    @pytest.mark.parametrize("steps", [None, 200])
    def test_a_mixed_plan_estimates_both_blocks(self, line_setup, state, steps):
        """Both blocks land on the exact joint MAP, within 0.1 posterior sigma.

        This compared against the TRUTH with an absolute 5e-3 on ``centre``,
        whose posterior sigma is 1.5e-4, so neither T-002 estimate defect could
        fail it. It also ran only ``steps=200``, where Adam's restart floor is
        already small; at the default step count the same plan used to land
        0.66 posterior sigma from the MAP and report converged. The reference
        here is the MAP itself, by float64 Newton on the joint objective, and
        the distance is Mahalanobis under the posterior precision there.
        """
        space, pipeline, observed = line_setup
        block = Block("centre") if steps is None else Block("centre", steps=steps)
        plan = SamplingPlan(space, Block("amp"), block)
        est = plan.estimate(
            pipeline, state, observed, noise=0.05, max_iter=60, tol=1e-6,
            solve_guard=None,
        )
        assert plan.engines == {("amp",): CONJUGATE, ("centre",): GRADIENT}
        exact, precision = _line_map(observed, sigma=0.05)
        got = np.concatenate(
            [np.ravel(np.asarray(est.values[name], np.float64)) for name in exact]
        )
        residual = got - np.concatenate([np.ravel(value) for value in exact.values()])
        distance = float(np.sqrt(residual @ precision @ residual))
        assert distance < 0.1, (distance, est.values, exact)
        # the amps are all different from each other, so a solve that returned
        # one number broadcast across the block would fail here
        assert jnp.allclose(est.values["amp"], LINE_TRUTH["amp"], rtol=2e-2), (
            est.values["amp"]
        )

    def test_a_mixed_plan_samples_both_blocks(self, line_setup, state):
        """NUTS-within-Gibbs: the conjugate block is drawn exactly and the
        gradient block takes a finite number of NUTS steps, which is what makes
        the scheme Metropolis-within-Gibbs rather than exact."""
        space, pipeline, observed = line_setup
        plan = SamplingPlan(space, Block("amp"), Block("centre", steps=8))
        draws = plan.sample(
            pipeline, state, observed, noise=0.5, key=jax.random.key(0),
            n_sweeps=12, warmup=6, solve_guard=None,
        )
        assert draws.n_draw == 6
        assert draws.samples["centre"].shape == (6,)
        assert draws.samples["amp"].shape == (6, N_TIME)
        assert np.all(np.isfinite(np.asarray(draws.samples["centre"])))
        # the gradient block MOVED — a NUTS step that silently returned its
        # starting point would leave every draw identical
        assert float(jnp.std(draws.samples["centre"])) > 0.0

    def test_the_NUTS_tuning_is_frozen_once_warmup_ends(
        self, line_setup, state, monkeypatch
    ):
        """A kernel that keeps adapting from the states it visits is no longer a
        valid transition, so every sweep whose draws are KEPT must run frozen.
        Asserted on the argument itself rather than on an outcome: adaptive and
        frozen chains both produce finite draws, so nothing downstream can tell
        them apart, and this is exactly the sort of branch that ships pinned on
        one side.
        """
        import rheplicant.inference.plan as plan_module

        space, pipeline, observed = line_setup
        plan = SamplingPlan(space, Block("amp"), Block("centre", steps=4))
        seen = []
        real = plan_module.gradient_draw

        def spy(*args, **kwargs):
            seen.append(kwargs["adapt"])
            return real(*args, **kwargs)

        monkeypatch.setattr(plan_module, "gradient_draw", spy)
        plan.sample(
            pipeline, state, observed, noise=0.5, key=jax.random.key(0),
            n_sweeps=9, warmup=3, solve_guard=None,
        )
        assert seen == [True, True, True, False, False, False, False, False, False], seen

    def test_the_gradient_block_uses_its_declared_step_count(
        self, line_setup, state, monkeypatch
    ):
        """``steps`` is a statistical assumption for a draw and a budget of
        Adam steps for an estimate; either way it must reach the engine.

        This compared how far one Adam step and four hundred travelled. The
        Newton steps that now follow Adam (T-002 A5-2) put the block on its
        conditional optimum either way, so the distance no longer depends on
        the count, and the plumbing is read off the optimiser's own argument.
        The last assertion is that independence; before the repair the two
        counts ended more than twenty times apart in distance travelled.
        """
        import rheplicant.inference.engines as engines_module

        space, pipeline, observed = line_setup
        common = {
            "noise": 0.05, "max_iter": 3, "tol": None, "solve_guard": None,
            "check_identifiability": False,
        }
        seen: list[int] = []
        real = engines_module._adam

        def spy(potential, x0, steps, step_sizes):
            seen.append(steps)
            return real(potential, x0, steps, step_sizes)

        monkeypatch.setattr(engines_module, "_adam", spy)
        stingy = SamplingPlan(space, Block("amp"), Block("centre", steps=1)).estimate(
            pipeline, state, observed, **common
        )
        assert set(seen) == {1}, seen
        seen.clear()
        generous = SamplingPlan(
            space, Block("amp"), Block("centre", steps=400)
        ).estimate(pipeline, state, observed, **common)
        assert set(seen) == {400}, seen
        assert float(stingy.values["centre"]) == pytest.approx(
            float(generous.values["centre"]), rel=1e-6
        )


class TestTheDefaultsTheConfigLayerQUOTES:
    """The values, pinned where the constants live.

    ``tests/config/test_preflight_fitting.py`` builds A25's message out of
    these rather than writing ``3`` and ``100`` into it, and asserts the
    message quotes them -- which kills a restated default, and is what that
    test is for. What it cannot do is notice the CONSTANT changing: a derived
    message follows the new value and every assertion there still passes.

    That gap was covered by an ``assert (MIN_SWEEPS, DEFAULT_MAX_ITER) ==
    (3, 100)`` line in the config test -- **the only pin these two values
    had**, and it sat in the layer that consumes them rather than the one that
    declares them. Wave B moves ``plan`` behind the adapter, so a pin reaching
    across that seam is one the migration has to renegotiate; a pin here is
    one it does not (**D51**).
    """

    def test_the_sweep_defaults_are_the_numbers_the_message_quotes(self):
        from rheplicant.inference.plan import DEFAULT_MAX_ITER, MIN_SWEEPS

        assert (MIN_SWEEPS, DEFAULT_MAX_ITER) == (3, 100)

    def test_min_sweeps_is_below_the_cap_it_is_compared_against(self):
        """The relation the guard reads, not just the two numbers.

        ``plan`` refuses ``min_sweeps > max_iter``, so a default pair that
        violated its own guard would refuse every document that wrote neither
        key -- which no fixture exercises, because they all write at least
        one.
        """
        from rheplicant.inference.plan import DEFAULT_MAX_ITER, MIN_SWEEPS

        assert MIN_SWEEPS <= DEFAULT_MAX_ITER

    def test_the_earliest_verdict_is_the_sweep_the_A25_message_quotes(self):
        """Pinned beside its declaration, like the two above: pre-flight A25
        refuses a ``max_iter`` below it and quotes it, and derives nothing
        from it but the comparison. The stop rule counts two changes between
        sweep outputs, so it is 3; the default cap must leave room for it."""
        from rheplicant.inference.plan import (
            DEFAULT_MAX_ITER,
            EARLIEST_CONVERGED_SWEEP,
        )

        assert EARLIEST_CONVERGED_SWEEP == 3
        assert EARLIEST_CONVERGED_SWEEP <= DEFAULT_MAX_ITER

    def test_min_draws_is_the_smallest_a_split_rhat_is_defined_on(self):
        """``MIN_DRAWS`` is derived rather than chosen, so it is pinned that
        way: two halves of two is the smallest split a variance exists on.

        Asserted through the package's own splitter rather than as the literal
        4, so a change to how the halves are taken moves this with it.
        """
        import numpy as np

        from rheplicant.inference.plan import MIN_DRAWS, _halves

        halves = _halves(np.arange(float(MIN_DRAWS)))
        assert halves.shape == (2, 2)
        assert _halves(np.arange(float(MIN_DRAWS - 1))).shape[1] < 2
