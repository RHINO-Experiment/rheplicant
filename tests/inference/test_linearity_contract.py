"""What ``check_linearity`` decides, pinned per axis (D16, ruled 2026-08-27).

Four of D16's five axes moved on that date, and each moved a verdict rather
than a number: the probe anchor, the number of outside values checked at, whether a second
criterion in units of the noise applies, and whether the departure is
aggregated over the whole output or per element. The
fixtures here are the ones the adjudication probe measured with
(``bayesmith/docs/probes/probe_12_d16_five_axes.py``), promoted to the suite
because a probe that lives only beside a decision decays as soon as the
decision is made -- which is what ``boundary-validation.md`` says to do with
one.

Each test states the OLD verdict as well as the new one. A test that only
knows the current answer cannot tell a deliberate change from a regression,
and every one of these three passed happily before the change while the
model under it was mis-classified.
"""

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import LinearityRefused
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference.linear import (
    DEFAULT_AT_POINTS,
    DEFAULT_SCALES,
    _magnitude,
    _probe_anchor,
    check_linearity,
)
from rheplicant.inference.noise import HomoscedasticNoise
from rheplicant.inference.parameters import Bind, Latent, ParameterSpace

N_TIME, N_FREQ = 4, 12
_X = jnp.linspace(-1.0, 1.0, N_FREQ)
_Q, _ = jnp.linalg.qr(jnp.stack([_X**k for k in range(5)], axis=1))
B1, B2 = _Q[:, :2], _Q[:, 2:4]


class Predict(AbstractOperator):
    """One prediction function, as an operator. The fixture's maths is in ``fn``."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    params: dict[str, jax.Array]
    # static: a function is not a JAX leaf, and eqx will try to trace it into
    # the pipeline's pytree otherwise -- the failure arrives as "returned a
    # value of type <class 'function'>", several frames from here.
    fn: Any = eqx.field(static=True)

    def __call__(self, state):
        row = self.fn(self.params)
        return state.with_data(jnp.broadcast_to(row, (N_TIME, N_FREQ)))


def _state() -> State:
    return State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        meta={"telescope": "RHINO", "obs_id": "d16-contract"},
    )


def _probe(fn, spec, *, names, **options):
    """``check_linearity`` over a one-operator pipeline. Verdict as a string."""
    operator = Predict(params={n: spec[n][0] for n in spec}, fn=fn)
    space = ParameterSpace(
        latents=[
            Latent(
                n,
                init=spec[n][0],
                prior=(
                    None
                    if spec[n][1] is None
                    else dist.Normal(spec[n][1], spec[n][2])
                ),
                linear=spec[n][3],
            )
            for n in spec
        ],
        bindings=[
            Bind(n, into=(lambda p, _n=n: p["predict"].params[_n])) for n in spec
        ],
    )
    try:
        return "accepted", check_linearity(
            space, Pipeline(operator, names=("predict",)), _state(),
            names=names, **options,
        )
    except LinearityRefused as refused:
        return "refused", refused


class TestTheAnchorIsThePriorWidth:
    """Axis 1. Probe magnitudes are multiples of the DECLARED uncertainty."""

    @staticmethod
    def _model(p):
        signal = B1 @ p["u"]
        return signal + 1e-7 * signal**2

    def test_a_wide_prior_reaches_curvature_a_narrow_init_does_not(self):
        """max|init| = 1 against a prior width of 100: a hundredfold reach.

        Measured before the anchor moved: accepted, worst departure 8.24e-05
        against an rtol of 1.19e-03. The model is the same either way -- what
        changed is where it was asked about.
        """
        verdict, _ = _probe(
            self._model,
            {"u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 100.0, True)},
            names=("u",),
        )
        assert verdict == "refused"

    def test_the_same_model_passes_when_the_declared_uncertainty_is_small(self):
        """The other half, without which the first is just "this refuses".

        Narrow the declared prior and the probes no longer reach the
        curvature, so the claim stands -- over the range the declaration
        actually spans, the prediction IS affine to within the tolerance.
        That is the anchor doing its job, not failing to.
        """
        verdict, _ = _probe(
            self._model,
            {"u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 0.1, True)},
            names=("u",),
        )
        assert verdict == "accepted"

    def test_it_reads_the_prior_and_not_the_init(self):
        latent = Latent(
            "u", init=jnp.asarray(75.0), prior=dist.Normal(75.0, 10.0), linear=True
        )
        assert _probe_anchor(latent) == pytest.approx(10.0)
        assert _magnitude(latent) == pytest.approx(75.0)

    def test_a_latent_with_no_prior_falls_back_to_its_init(self):
        """A free parameter has no declared uncertainty to anchor on.

        Falling back to ``max|init|`` is the old rule, kept exactly where the
        new one has nothing to read -- and to 1.0 from there when the init is
        all zeros, which is the documented last resort.
        """
        free = Latent("u", init=jnp.array([3.0, -1.0]), prior=None, linear=True)
        assert _probe_anchor(free) == pytest.approx(3.0)
        zeroed = Latent("u", init=jnp.zeros(2), prior=None, linear=True)
        assert _probe_anchor(zeroed) == pytest.approx(1.0)

    def test_a_lognormal_prior_is_not_read_as_a_width(self):
        """``LogNormal`` carries a ``.scale`` that is a width in ``log x``.

        Identification is by TYPE for the same reason it is in
        ``_gaussian_parameters``: duck-typing on ``.scale`` would anchor the
        probes on a number that is not a width in the latent at all.
        """
        latent = Latent(
            "u", init=jnp.asarray(4.0), prior=dist.LogNormal(0.0, 0.5), linear=True
        )
        assert _probe_anchor(latent) == pytest.approx(4.0)  # the init, not 0.5


class TestTheCheckLooksAtMoreThanOneOutsideValue:
    """Axis 2. "Affine GIVEN the outside latents" -- given which values?"""

    CENTRE = 2.0

    @classmethod
    def _model(cls, p):
        signal = B1 @ p["u"]
        # Exactly affine when w sits at its declared init, curved anywhere else.
        return signal + 1e-3 * (p["w"] - cls.CENTRE) * signal**2

    def _spec(self):
        return {
            "u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 1.0, True),
            "w": (jnp.asarray(self.CENTRE), jnp.asarray(self.CENTRE), 1.0, False),
        }

    def test_a_model_affine_only_at_the_declared_point_is_refused(self):
        """Measured before the default moved: 0.0 at every scale, accepted.

        Not "nearly affine" -- EXACTLY affine, because at that one point the
        quadratic term is multiplied by zero. A single-point check had no
        margin to be suspicious of.
        """
        verdict, _ = _probe(self._model, self._spec(), names=("u",))
        assert verdict == "refused"

    def test_pinning_the_outside_value_restores_the_old_verdict(self):
        """``at_points=`` is the escape, and it is deliberately explicit.

        A caller whose model really is used at exactly one outside value can
        say so; what they cannot do is get that by accident.
        """
        verdict, errors = _probe(
            self._model,
            self._spec(),
            names=("u",),
            at_points=[{"w": jnp.asarray(self.CENTRE)}],
        )
        assert verdict == "accepted"
        assert all(float(value) == 0.0 for value in errors.values())

    def test_the_default_point_count_is_the_shared_constant(self):
        assert DEFAULT_AT_POINTS == 3

    def test_an_integer_outside_latent_does_not_break_the_draw(self):
        """A channel index is a perfectly legal outside latent and has no
        Gaussian draw. Measured: without the dtype guard, ``jax.random.normal``
        refuses an int32 outright and a valid document dies with a message
        about dtypes rather than a verdict about linearity."""

        def model(p):
            return (B1 @ p["u"]) * (1.0 + 0.0 * p["k"])

        spec = {
            "u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 1.0, True),
            "k": (jnp.asarray(2, dtype=jnp.int32), None, None, False),
        }
        verdict, _ = _probe(model, spec, names=("u",))
        assert verdict == "accepted"


class TestTheDepartureIsAggregatedPerElement:
    """Axis 5. A bright channel must not supply a faint one's yardstick."""

    @staticmethod
    def _bright_and_faint(p):
        loud = 1e8 * (B1 @ p["u"])[:6]
        faint = (B2 @ p["u"])[:6]
        return jnp.concatenate([loud, faint + 1e-2 * faint**2])

    @staticmethod
    def _faint_alone(p):
        faint = B2 @ p["u"]
        return faint + 1e-2 * faint**2

    SPEC: ClassVar[dict] = {"u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 1.0, True)}

    def test_a_faint_lie_beside_a_bright_truth_is_refused(self):
        """Measured before the aggregation moved: accepted, and the worst
        departure reported as 2.81e-14 -- a 1e-2 curvature rendered as
        'bitwise affine' because the bright half set both the departure and
        the variation the ratio divides by."""
        verdict, _ = _probe(self._bright_and_faint, dict(self.SPEC), names=("u",))
        assert verdict == "refused"

    def test_the_same_lie_alone_was_always_caught(self):
        """The control. Without it, the test above is consistent with a check
        that refuses everything -- and the point is that the curvature was
        always visible, only diluted."""
        verdict, _ = _probe(self._faint_alone, dict(self.SPEC), names=("u",))
        assert verdict == "refused"

    def test_an_honestly_affine_model_still_passes_at_the_same_dynamic_range(self):
        """Per-element aggregation must not simply refuse wide dynamic range.

        Same 1e8 spread, no curvature anywhere: this has to pass, or the axis
        bought a stricter check by making it useless.
        """

        def honest(p):
            loud = 1e8 * (B1 @ p["u"])[:6]
            faint = (B2 @ p["u"])[:6]
            return jnp.concatenate([loud, faint])

        verdict, errors = _probe(honest, dict(self.SPEC), names=("u",))
        assert verdict == "accepted"
        assert set(errors) == set(DEFAULT_SCALES)


class TestSmallComparedToWhat:
    """Axis 3. A departure under the relative tolerance can still be many
    noise widths wide, and that is the regime a conjugate solve gets wrong."""

    @staticmethod
    def _model(p):
        signal = B1 @ p["u"]
        return signal + 1e-7 * signal**2

    SPEC: ClassVar[dict] = {"u": (jnp.array([1.0, 0.3]), jnp.zeros(2), 1.0, True)}

    def test_without_a_noise_model_the_verdict_cannot_move_with_the_noise(self):
        """The old contract, kept as the fallback and pinned as such.

        Not a bug and not a warning: with no noise passed there is no second
        criterion, and the answer is the one this check always gave. What
        would be wrong is for it to LOOK like it had considered the noise.
        """
        verdicts = {
            sigma: _probe(self._model, dict(self.SPEC), names=("u",))[0]
            for sigma in (1e2, 1.0, 1e-2, 1e-4)
        }
        assert set(verdicts.values()) == {"accepted"}

    @pytest.mark.parametrize(
        ("sigma", "expected"),
        [(1e2, "accepted"), (1.0, "refused"), (1e-2, "refused"), (1e-4, "refused")],
    )
    def test_with_one_the_verdict_moves_with_the_noise_alone(self, sigma, expected):
        """The model, the probes and the departure never move; sigma does.

        Measured at sigma=1: the relative column reads 0.000e+00 at every
        probe -- under its own roundoff floor -- while the sigma-weighted one
        reads 6.262e-02 against a threshold of 1e-3. So the refusal is the
        second criterion's alone, which is what makes this axis a separate
        question rather than a tighter tolerance on the first.
        """
        verdict, _ = _probe(
            self._model,
            dict(self.SPEC),
            names=("u",),
            noise=HomoscedasticNoise(sigma=sigma),
        )
        assert verdict == expected

    def test_the_refusal_carries_both_columns_and_says_which_threshold(self):
        """One number against one threshold is unreadable half the time.

        The guard is a DISJUNCTION, so a reader shown only the relative column
        would see a value under its tolerance printed beside a refusal and
        conclude the guard is broken.
        """
        _, refused = _probe(
            self._model,
            dict(self.SPEC),
            names=("u",),
            noise=HomoscedasticNoise(sigma=1.0),
        )
        assert refused.weighted is not None
        assert refused.weighted_rtol == pytest.approx(1e-3)
        assert set(refused.weighted) == set(refused.errors)
        assert max(float(v) for v in refused.weighted.values()) > refused.weighted_rtol
        assert "weighted_rtol" in str(refused)

    def test_no_noise_means_the_second_column_is_absent_not_empty(self):
        """"Not measured" and "measured as nothing" are different answers.

        The whole reason this payload exists is that the second was once
        reported for both.
        """
        curved = lambda p: (B1 @ p["u"]) + 0.5 * (B1 @ p["u"]) ** 2  # noqa: E731
        _, refused = _probe(curved, dict(self.SPEC), names=("u",))
        assert refused.weighted is None
        assert refused.weighted_rtol is None
        assert "weighted_rtol" not in str(refused)
