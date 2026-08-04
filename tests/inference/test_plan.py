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
from rheplicant.inference.noise import HomoscedasticNoise, RadiometerNoise
from rheplicant.inference.plan import CHECK_EACH_SWEEP, CHECK_ONCE, MIN_DRAWS
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
        """
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))

        with pytest.raises(ParameterSpaceError) as caught:
            plan.estimate(
                pipeline, state, observed, noise=NOISE, max_iter=3, solve_guard=None
            )
        message = str(caught.value)
        assert "did not converge" in message
        assert "JOINT chi-squared is still falling" in message, message

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

    def test_the_test_is_a_DECREASE_not_a_CHANGE(self, basis_setup, state):
        """The trap iterative_gls documents for its own reweight_tol: at the
        fixed point consecutive sweeps differ by the inner solver's own noise,
        so |chi2[k] - chi2[k-1]| never falls below it and a converged run is
        refused forever. Measured here: the plateau's sweep-to-sweep jitter is
        far above the default tol, and the run still converges — which it could
        not if the test were on the absolute change.
        """
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=200, solve_guard=None
        )
        assert est.diagnostics.converged is True
        trace = est.diagnostics.chi2
        # the last step made no progress (that is why it stopped) ...
        assert trace[-2] - trace[-1] <= 1e-8 * max(abs(trace[-1]), 1.0)
        # ... while the plateau it stopped on is jittering by far more than that
        plateau = trace[-min(10, trace.size) :]
        assert float(np.max(np.abs(np.diff(plateau)))) > 1e-8, plateau

    def test_min_sweeps_keeps_a_stationary_first_step_from_ending_the_run(
        self, basis_setup, state
    ):
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=200, min_sweeps=8,
            solve_guard=None,
        )
        assert est.diagnostics.sweeps >= 8

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
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))
        wrong = observed[:, 0]
        with pytest.raises(ParameterSpaceError, match="observed has shape"):
            plan.estimate(pipeline, state, wrong, noise=NOISE)
        with pytest.raises(ParameterSpaceError, match="observed has shape"):
            plan.sample(
                pipeline, state, wrong, noise=NOISE, key=jax.random.key(0), n_sweeps=8
            )

    def test_both_exits_refuse_a_BILINEAR_group(self, basis_setup, state):
        """gain and t_coeff are each affine given the other and bilinear
        together, so one block over both is not a conjugate problem at all.
        Inherited from check_linearity, which probes the JOINT map — and which
        no per-latent check could have caught."""
        space, pipeline, observed = basis_setup
        plan = SamplingPlan(space, Block("gain", "t_coeff"))
        with pytest.raises(ParameterSpaceError, match="not affine in them JOINTLY"):
            plan.estimate(pipeline, state, observed, noise=NOISE)
        with pytest.raises(ParameterSpaceError, match="not affine in them JOINTLY"):
            plan.sample(
                pipeline, state, observed, noise=NOISE, key=jax.random.key(0), n_sweeps=8
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

    def test_a_mixed_plan_estimates_both_blocks(self, line_setup, state):
        space, pipeline, observed = line_setup
        plan = SamplingPlan(space, Block("amp"), Block("centre", steps=200))
        est = plan.estimate(
            pipeline, state, observed, noise=0.05, max_iter=60, tol=1e-6,
            solve_guard=None,
        )
        assert plan.engines == {("amp",): CONJUGATE, ("centre",): GRADIENT}
        assert float(jnp.abs(est.values["centre"] - LINE_TRUTH["centre"])) < 5e-3, (
            est.values["centre"]
        )
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

    def test_the_gradient_block_uses_its_declared_step_count(self, line_setup, state):
        """``steps`` is a statistical assumption for a draw and a real budget
        for an estimate; either way it must reach the engine. One Adam step
        cannot travel as far as two hundred."""
        space, pipeline, observed = line_setup
        common = {
            "noise": 0.05, "max_iter": 3, "tol": None, "solve_guard": None,
            "check_identifiability": False,
        }
        stingy = SamplingPlan(space, Block("amp"), Block("centre", steps=1)).estimate(
            pipeline, state, observed, **common
        )
        generous = SamplingPlan(
            space, Block("amp"), Block("centre", steps=400)
        ).estimate(pipeline, state, observed, **common)
        start = float(space.latent("centre").init)
        moved_little = abs(float(stingy.values["centre"]) - start)
        moved_far = abs(float(generous.values["centre"]) - start)
        assert moved_far > 20.0 * moved_little, (moved_little, moved_far)
