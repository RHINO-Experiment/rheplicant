"""Tests for ``auto_blocks`` — the partition derived instead of declared.

The headline case is multilinear: ``gain * (B_ant @ t_ant + B_nw @ t_nw +
tone)`` with the tone known. Three latents, all three genuinely
``linear=True``, and exactly one correct conjugate partition —
``{t_ant, t_nw}`` in one block and ``{gain}`` in another. Both of the
neighbouring answers are wrong in a way this file pins rather than assumes:

* sweeping all three into ONE block is refused by the plan's own joint
  linearity check, and ``test_one_block_holding_all_three_is_refused`` runs
  that refusal, so "auto_blocks avoided a real error" is measured here and not
  taken on trust;
* splitting ``t_ant`` from ``t_nw`` is not refused by anything — it is a legal,
  quietly worse partition, and it is the pair ``test_linear_groups.py``
  measures as recovered by one joint solve and missed by hundreds of kelvin by
  alternation. Nothing downstream would report it, which is why the grouping is
  asserted exactly and not merely for block count.

The tone is what makes the model identifiable: without it ``gain`` and the
temperatures trade off along ``gain -> c gain, T -> T / c`` and the plan
refuses the model before any of this is reached, which would leave the
end-to-end test measuring the identifiability guard instead of the partition.
"""

import warnings
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import LinearityRefused, ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Block,
    Latent,
    ParameterSpace,
    SamplingPlan,
    auto_blocks,
)
from rheplicant.inference.engines import CONJUGATE, GRADIENT, LOG_CONJUGATE
from rheplicant.inference.loglinear import (
    FIRST_ORDER_MAX_FRACTIONAL,
    log_route_refusal,
    to_log_space,
)
from rheplicant.inference.noise import HomoscedasticNoise, RadiometerNoise
from rheplicant.inference.partition import UncheckedLogRouteWarning
from rheplicant.radio import GainOperator


class _Multiplicative:
    """The smallest thing the log route reads as multiplicative noise.

    Only ``fractional`` and ``std`` are consulted, so this is the honest
    minimum rather than a stub pretending to be a RadiometerNoise. It exists
    to drive ``f`` to a chosen value -- reaching the ceiling with a real
    radiometer would need a channel width nobody observes with.
    """

    def __init__(self, fractional: float):
        self.fractional = fractional

    def std(self, prediction):
        return self.fractional * jnp.abs(prediction)


#: f = 4.05e-3 for a 61 kHz channel at 1 s -- two orders under the ceiling,
#: which is where a real instrument sits.
MULTIPLICATIVE = RadiometerNoise(channel_width=61e3, integration_time=1.0)

N_TIME, N_FREQ = 6, 9
TONE_CHANNEL = 2
TONE_KELVIN = 400.0


# ------------------------------------------------------------ test doubles --


class BasisOperator(AbstractOperator):
    """``data[t, f] = (basis @ coeff)[f]`` — a spectral basis, flat in time."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    coeff: jax.Array
    basis: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        row = self.basis @ self.coeff
        return state.with_data(jnp.broadcast_to(row, (n_time, row.shape[0])))


class AddBasisOperator(AbstractOperator):
    """The same, added to whatever is already on the signal path."""

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    coeff: jax.Array
    basis: jax.Array

    def __call__(self, state):
        return state.with_data(state.data + (self.basis @ self.coeff)[None, :])


class CalibrationTone(AbstractOperator):
    """A KNOWN per-channel signal ahead of the gain — the identifying tone."""

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    tone: jax.Array

    def __call__(self, state):
        return state.with_data(state.data + self.tone[None, :])


class GaussianLine(AbstractOperator):
    """``data[t, f] = amp[t] * exp(-((x[f] - centre)/width)^2 / 2)``.

    ``amp`` enters linearly and ``centre`` does not, so this is the fixture for
    a partition holding one block of each kind.
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


# ------------------------------------------------------------------ bases ---


def _orthonormal(n: int, powers: tuple[int, ...]) -> jax.Array:
    x = jnp.linspace(-1.0, 1.0, n)
    q, _ = jnp.linalg.qr(jnp.stack([x**k for k in powers], axis=1))
    return q


#: Split from ONE orthonormal set, so the two bases are exactly orthogonal and
#: the joint block is exactly well conditioned — a grouping test must not be
#: measuring conditioning.
_SHAPES = _orthonormal(N_FREQ, (0, 1, 2, 3, 4))
B_ANT = _SHAPES[:, :3]
B_NW = _SHAPES[:, 3:5]

#: Asymmetric in shape AND magnitude, so a member swap cannot pass unnoticed.
TRUE_ANT = jnp.array([2800.0, -160.0, 35.0])
TRUE_NW = jnp.array([250.0, -18.0])
GAIN0 = 1.5 + 0.05 * jnp.arange(N_TIME, dtype=float)

TONE = jnp.zeros(N_FREQ).at[TONE_CHANNEL].set(TONE_KELVIN)


@pytest.fixture
def state():
    return State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        meta={"telescope": "RHINO", "obs_id": "auto-partition-000"},
    )


# ------------------------------------------------------- multilinear model --


@pytest.fixture
def multilinear():
    """``data = gain * (B_ant @ t_ant + B_nw @ t_nw + tone)``, tone known."""
    return Pipeline(
        BasisOperator(coeff=jnp.zeros(3), basis=B_ANT),
        AddBasisOperator(coeff=jnp.zeros(2), basis=B_NW),
        CalibrationTone(tone=TONE),
        GainOperator(gain=GAIN0),
        names=("t_ant", "t_nw", "tone", "gain"),
    )


def multilinear_space() -> ParameterSpace:
    return ParameterSpace(
        latents=[
            Latent(
                "t_ant",
                init=TRUE_ANT,
                prior=dist.Normal(jnp.zeros(3), 1e4),
                linear=True,
            ),
            Latent(
                "t_nw",
                init=TRUE_NW,
                prior=dist.Normal(jnp.zeros(2), 1e3),
                linear=True,
            ),
            Latent(
                "gain",
                init=GAIN0,
                prior=dist.Normal(GAIN0, 0.5),
                linear=True,
            ),
        ],
        bindings=[
            Bind("t_ant", into=lambda p: p["t_ant"].coeff),
            Bind("t_nw", into=lambda p: p["t_nw"].coeff),
            Bind("gain", into=lambda p: p["gain"].gain),
        ],
    )


@pytest.fixture
def space():
    return multilinear_space()


@pytest.fixture
def observed(space, multilinear, state):
    forward, values0 = space.forward_fn(multilinear, state)
    return forward(
        {**values0, "t_ant": TRUE_ANT, "t_nw": TRUE_NW, "gain": GAIN0}
    )


class TestTheMultilinearSplit:
    def test_the_coupled_factor_gets_a_block_of_its_own(
        self, space, multilinear, state
    ):
        """The whole point: one block per factor, all of a factor's latents in it."""
        blocks = auto_blocks(space, multilinear, state)
        assert [block.names for block in blocks] == [("t_ant", "t_nw"), ("gain",)]

    def test_both_blocks_are_conjugate(self, space, multilinear, state):
        plan = SamplingPlan.automatic(space, multilinear, state)
        assert set(plan.engines.values()) == {CONJUGATE}

    def test_one_block_holding_all_three_is_refused(
        self, space, multilinear, state, observed
    ):
        """What auto_blocks avoided — run, not assumed.

        Each of the three is affine on its own, so nothing about the
        DECLARATION is wrong; the group is refused because the joint map is
        not affine. If this ever stops raising, the grouping test above stops
        being evidence of anything.
        """
        lumped = SamplingPlan(space, Block("t_ant", "t_nw", "gain"))
        with pytest.raises(LinearityRefused) as caught:
            lumped.estimate(
                multilinear,
                state,
                observed,
                noise=HomoscedasticNoise(sigma=1.0),
                check_identifiability=False,
            )
        assert "JOINTLY" in str(caught.value)

    def test_splitting_the_uncoupled_pair_is_NOT_refused(
        self, space, multilinear, state, observed
    ):
        """The other neighbour: legal, quietly worse, and nothing reports it.

        This is why the grouping assertion is on the exact partition rather
        than on the block count — a per-latent split would run clean.

        ``tol=None`` keeps the test on the one question it is asking. Whether
        the split converges is a separate claim, and asserting it here would
        make a convergence failure read as a linearity refusal.
        """
        split = SamplingPlan(space, Block("t_ant"), Block("t_nw"), Block("gain"))
        split.estimate(
            multilinear,
            state,
            observed,
            noise=HomoscedasticNoise(sigma=1.0),
            check_identifiability=False,
            max_iter=2,
            tol=None,
        )

    def test_the_derived_partition_recovers_the_truth(
        self, space, multilinear, state, observed
    ):
        """End to end: the derived blocks survive every check the plan runs."""
        plan = SamplingPlan.automatic(space, multilinear, state)
        estimate = plan.estimate(
            multilinear, state, observed, noise=HomoscedasticNoise(sigma=1.0)
        )
        assert jnp.allclose(estimate.values["t_ant"], TRUE_ANT, rtol=1e-2)
        assert jnp.allclose(estimate.values["t_nw"], TRUE_NW, rtol=1e-2)
        assert jnp.allclose(estimate.values["gain"], GAIN0, rtol=1e-2)

    def test_automatic_is_the_classmethod_for_the_same_blocks(
        self, space, multilinear, state
    ):
        blocks = auto_blocks(space, multilinear, state)
        assert SamplingPlan.automatic(space, multilinear, state).engines == {
            block.names: CONJUGATE for block in blocks
        }


# ------------------------------------------------------- uncoupled grouping --


class TestGroupingWithoutCoupling:
    def test_an_additive_model_puts_every_linear_latent_in_one_block(
        self, space, state
    ):
        """No gain, so nothing couples: the two temperatures share one solve.

        The complement of the headline test. Together they pin that the split
        tracks the model's coupling rather than the latent count.
        """
        additive = Pipeline(
            BasisOperator(coeff=jnp.zeros(3), basis=B_ANT),
            AddBasisOperator(coeff=jnp.zeros(2), basis=B_NW),
            names=("t_ant", "t_nw"),
        )
        additive_space = ParameterSpace(
            latents=[
                Latent(
                    "t_ant",
                    init=TRUE_ANT,
                    prior=dist.Normal(jnp.zeros(3), 1e4),
                    linear=True,
                ),
                Latent(
                    "t_nw",
                    init=TRUE_NW,
                    prior=dist.Normal(jnp.zeros(2), 1e3),
                    linear=True,
                ),
            ],
            bindings=[
                Bind("t_ant", into=lambda p: p["t_ant"].coeff),
                Bind("t_nw", into=lambda p: p["t_nw"].coeff),
            ],
        )
        blocks = auto_blocks(additive_space, additive, state)
        assert [block.names for block in blocks] == [("t_ant", "t_nw")]


# ----------------------------------------------- discovered, not declared ---


LOG_G = jnp.log(GAIN0)

#: The same two bases, oriented so the sky is POSITIVE. QR fixes each column
#: only up to sign, and at these coefficients the raw orientation gives a
#: spectrum of about -1100 K — which is not a temperature and has no log. Fixed
#: here so the fixture is physical; the refusal a genuinely negative prediction
#: earns is exercised deliberately in ``test_loglinear.py`` rather than met by
#: accident in a test about partitioning.
B_ANT_SKY = -B_ANT
B_NW_SKY = -B_NW


@pytest.fixture
def log_gain_model():
    """``d = exp(log_gain) (B_ant t_ant + B_nw t_nw + tone)``.

    The sky is affine given the gain and declares ``linear=True``; ``log_gain``
    declares NOTHING. There is no ``log_linear=True`` to write, deliberately —
    the classification is the probe's to make.
    """
    return Pipeline(
        BasisOperator(coeff=jnp.zeros(3), basis=B_ANT_SKY),
        AddBasisOperator(coeff=jnp.zeros(2), basis=B_NW_SKY),
        CalibrationTone(tone=TONE),
        GainOperator(gain=jnp.ones(N_TIME)),
        names=("t_ant", "t_nw", "tone", "gain"),
    )


def log_gain_space() -> ParameterSpace:
    return ParameterSpace(
        latents=[
            Latent(
                "t_ant", init=TRUE_ANT, prior=dist.Normal(jnp.zeros(3), 1e4), linear=True
            ),
            Latent(
                "t_nw", init=TRUE_NW, prior=dist.Normal(jnp.zeros(2), 1e3), linear=True
            ),
            Latent("log_gain", init=LOG_G, prior=dist.Normal(LOG_G, 0.5)),
        ],
        bindings=[
            Bind("t_ant", into=lambda p: p["t_ant"].coeff),
            Bind("t_nw", into=lambda p: p["t_nw"].coeff),
            Bind("log_gain", into=lambda p: p["gain"].gain, fn=jnp.exp),
        ],
    )


class TestLogLinearIsDiscovered:
    def test_the_log_gain_gets_a_log_conjugate_block_with_no_declaration(
        self, log_gain_model, state
    ):
        """The headline for the third engine: nobody wrote ``log_linear=True``.

        ``t_ant`` and ``t_nw`` are jointly affine given the gain and share one
        conjugate block; ``log_gain`` is affine only after the data is logged,
        which the probe finds. No gradient block is produced at all — every
        latent in this model is solved in closed form.
        """
        plan = SamplingPlan.automatic(
            log_gain_space(), log_gain_model, state, noise=MULTIPLICATIVE
        )
        assert plan.engines == {
            ("t_ant", "t_nw"): CONJUGATE,
            ("log_gain",): LOG_CONJUGATE,
        }

    def test_a_genuinely_nonlinear_latent_is_NOT_filed_as_log_linear(
        self, line, state
    ):
        """The complement: the probe must say "no" as readily as it says "yes".

        Without this, "discovers log-linearity" would be consistent with a
        probe that answers yes to everything and quietly routes a beam width
        into a conjugate solve.
        """
        blocks = auto_blocks(line_space(), line, state)
        assert [block.names for block in blocks] == [("amp",), ("centre",)]
        assert blocks[-1].engine is None  # derived as gradient, not log_conjugate

    def test_the_linear_probe_scales_are_not_used_for_the_log_probe(
        self, log_gain_model, state
    ):
        """A 1e3 probe through an exponential overflows and the check refuses.

        Passing the linear default down would file a genuinely log-linear
        latent as gradient — a misclassification costing a conjugate block,
        reported nowhere. Pinned by driving it deliberately.
        """
        space = log_gain_space()
        derived = auto_blocks(space, log_gain_model, state, noise=MULTIPLICATIVE)
        assert derived[-1].engine == LOG_CONJUGATE

        misprobed = auto_blocks(
            space, log_gain_model, state, noise=MULTIPLICATIVE,
            log_scales=(1e3,), steps=4,
        )
        assert misprobed[-1].names == ("log_gain",)
        assert misprobed[-1].engine is None  # fell through to the gradient block


class TestTheNoiseIsHalfTheLogQuestion:
    """A log-conjugate block is a claim about the LIKELIHOOD, not the prediction.

    Found by D17's dual-run protocol on 2026-08-27 and ruled by the owner the
    same day: ``auto_blocks`` took no noise model, so it could hand out a
    ``log_conjugate`` block that ``to_log_space`` then refused -- which this
    module's own docstring names as the failure it exists to prevent. The two
    refusals were already written, and on the same constant; they simply lived
    at solve time. These pin them at partition time.
    """

    def test_an_additive_noise_model_means_no_log_block(self, log_gain_model, state):
        """Taking logs of an already-additive noise does not simplify it -- it
        states a different likelihood from the one declared."""
        blocks = auto_blocks(
            log_gain_space(),
            log_gain_model,
            state,
            noise=HomoscedasticNoise(sigma=1.0),
        )
        assert blocks[-1].names == ("log_gain",)
        assert blocks[-1].engine is None  # derived as gradient

    def test_a_fractional_level_above_the_ceiling_means_no_log_block(
        self, log_gain_model, state
    ):
        """The same ceiling the transform enforces, applied before the
        partition rather than at the first sweep."""
        blocks = auto_blocks(
            log_gain_space(), log_gain_model, state, noise=_Multiplicative(0.3)
        )
        assert blocks[-1].names == ("log_gain",)
        assert blocks[-1].engine is None

    def test_omitting_the_noise_warns_rather_than_claiming_a_block(
        self, log_gain_model, state
    ):
        """Conservative AND loud.

        Gradient is always a sound verdict, so the omission cannot produce a
        wrong answer -- but saying nothing would turn the defect this whole
        change is about from wrong into silent, which is worse.
        """
        with pytest.warns(UncheckedLogRouteWarning, match="log_gain"):
            blocks = auto_blocks(log_gain_space(), log_gain_model, state)
        assert blocks[-1].names == ("log_gain",)
        assert blocks[-1].engine is None

    def test_a_model_with_no_log_candidate_does_not_warn(self, line, state):
        """The warning names a real missed opportunity or it is noise."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", UncheckedLogRouteWarning)
            auto_blocks(line_space(), line, state)

    @pytest.mark.parametrize(
        "noise",
        [
            pytest.param(HomoscedasticNoise(sigma=1.0), id="additive"),
            pytest.param(_Multiplicative(0.004), id="f-small"),
            pytest.param(_Multiplicative(FIRST_ORDER_MAX_FRACTIONAL), id="f-at-ceiling"),
            pytest.param(_Multiplicative(0.3), id="f-too-large"),
            pytest.param(
                RadiometerNoise(channel_width=61e3, integration_time=1.0),
                id="radiometer",
            ),
        ],
    )
    def test_the_two_consumers_of_the_predicate_cannot_disagree(self, noise):
        """The anti-drift guard, and the reason the predicate was extracted.

        ``log_route_refusal`` is asked at PARTITION time; ``to_log_space``
        raises at SOLVE time. If those two could ever answer differently, the
        defect D17 found comes straight back in a new shape -- a partition
        promising a route the transform refuses. So the equivalence is
        asserted per noise model rather than assumed from them sharing a
        function today.
        """
        data = jnp.abs(jnp.linspace(1.0, 3.0, N_FREQ)) + 1.0
        refused_at_partition = log_route_refusal(noise) is not None
        try:
            to_log_space(data, noise)
            refused_at_solve = False
        except ParameterSpaceError:
            refused_at_solve = True
        assert refused_at_partition == refused_at_solve


# ---------------------------------------------------- linear and non-linear --


def line_space(amp_linear: bool = True) -> ParameterSpace:
    return ParameterSpace(
        latents=[
            Latent(
                "amp",
                init=jnp.full((N_TIME,), 40.0),
                prior=dist.Normal(jnp.zeros(N_TIME), 1e2),
                linear=amp_linear,
            ),
            Latent("centre", init=jnp.array(0.10), prior=dist.Normal(0.0, 0.5)),
        ],
        bindings=[
            Bind("amp", into=lambda p: p["line"].amp),
            Bind("centre", into=lambda p: p["line"].centre),
        ],
    )


@pytest.fixture
def line():
    return Pipeline(
        GaussianLine(amp=jnp.full((N_TIME,), 40.0), centre=jnp.array(0.10)),
        names=("line",),
    )


class TestLinearAndNonlinear:
    def test_the_nonlinear_latent_gets_a_gradient_block(self, line, state):
        plan = SamplingPlan.automatic(line_space(), line, state)
        assert plan.engines == {("amp",): CONJUGATE, ("centre",): GRADIENT}

    def test_the_gradient_block_comes_last(self, line, state):
        """Sweep order: a gradient block visited first would take its NUTS
        steps against linear latents still at their declared init."""
        blocks = auto_blocks(line_space(), line, state)
        assert [block.names for block in blocks] == [("amp",), ("centre",)]

    def test_steps_and_learning_rate_reach_the_gradient_block(self, line, state):
        blocks = auto_blocks(line_space(), line, state, steps=7, learning_rate=0.5)
        gradient = blocks[-1]
        assert gradient.names == ("centre",)
        assert (gradient.steps, gradient.learning_rate) == (7, 0.5)

    def test_a_space_with_no_linear_latents_is_one_gradient_block(self, line, state):
        """``linear=`` is a declaration, not a discovered property: dropping it
        from ``amp`` moves it into the gradient block."""
        blocks = auto_blocks(line_space(amp_linear=False), line, state)
        assert [block.names for block in blocks] == [("amp", "centre")]


# -------------------------------------------------------------- refusals ----


class TestRefusals:
    @pytest.mark.parametrize(
        "knob", [{"steps": 7}, {"learning_rate": 0.5}], ids=["steps", "learning_rate"]
    )
    def test_a_gradient_knob_with_no_gradient_block_is_refused(
        self, space, multilinear, state, knob
    ):
        """A silently ignored tuning would read as a sampler that was tuned."""
        with pytest.raises(ParameterSpaceError) as caught:
            auto_blocks(space, multilinear, state, **knob)
        assert next(iter(knob)) in str(caught.value)

    def test_a_false_linear_declaration_is_refused_by_NAME(self, line, state):
        """Checked per latent BEFORE any pair, so the message is the single
        one — a broken declaration must not be reported as a coupling."""
        wrong = ParameterSpace(
            latents=[
                Latent(
                    "amp",
                    init=jnp.full((N_TIME,), 40.0),
                    prior=dist.Normal(jnp.zeros(N_TIME), 1e2),
                    linear=True,
                ),
                Latent(
                    "centre",
                    init=jnp.array(0.10),
                    prior=dist.Normal(0.0, 0.5),
                    linear=True,
                ),
            ],
            bindings=[
                Bind("amp", into=lambda p: p["line"].amp),
                Bind("centre", into=lambda p: p["line"].centre),
            ],
        )
        with pytest.raises(LinearityRefused) as caught:
            auto_blocks(wrong, line, state)
        message = str(caught.value)
        # The singleton list is the per-latent diagnosis: discovery probes
        # every latent ALONE (through the same group spelling the plan's
        # re-check uses) before any pair is tried, so a broken declaration
        # is named by itself and never as half of a pair.
        assert "['centre']" in message
