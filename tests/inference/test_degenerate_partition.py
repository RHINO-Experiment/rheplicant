"""The motivating failure, measured — not quoted.

Six docstrings in :mod:`rheplicant.inference` told the same story about a
hand-rolled alternating solve over a bilinear ``gain x T_ant`` model: every
per-block guard reads green, and the answer is thousands of kelvin from the
truth. They quoted it as a specific triple (a condition number, a CG residual,
a distance) — and they disagreed with each other, because nothing in the suite
produced the numbers and each site remembered a different run.

Measuring it settles the disagreement and replaces it with a stronger claim.
The distance from truth is not a property of the degeneracy at all: it is the
initial offset, carried along the null direction and left there. Start 1 % off
and land 56 K out; start 100 % off and land 6077 K out; start ON the truth and
stay. **The guards read the same in every one of those runs** — condition
number in a band 3 % wide, CG residual at 1e-7, ``check_linearity`` passing —
so they do not merely miss the error, they are blind to four decades of it,
including the difference between the run that is right and the run that is
catastrophically wrong.

That is the honest version of what the docstrings were reaching for, and it is
why the fix is a rank test over the joint Jacobian rather than a tighter
per-block tolerance. No tolerance separates these rows.

Both precisions: the errors below are identical to five figures under
``JAX_ENABLE_X64=1``, which is the point — the degeneracy is structural, not
numerical, so a suite that only held in one precision would be testing the
arithmetic instead of the model.
"""

import jax.numpy as jnp
import pytest

# No sys.path insert to reach the sibling: pytest's default (prepend) import
# mode already puts this directory on the path, because tests/inference/ has no
# __init__.py and is therefore the basedir of every module in it. The insert
# that used to be here prepended tests/ globally, at import time and for the
# rest of the session, which is the shape of leak that made an unrelated module
# visible to a later test and broke it -- see tests/test_tour_runs.py.
from test_identifiability import (  # noqa: E402
    GAIN0,
    N_FREQ,
    N_TIME,
    T_ANT0,
    TONE_KELVIN,
    make_pipeline,
)

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.identifiability import identifiability
from rheplicant.inference.linear import (
    check_linearity,
    condition_estimate,
    linear_operator,
    wiener_solve,
)

#: Flat enough that the prior does not resolve the degeneracy. A tight prior
#: WOULD pick a point on the null direction, which is a real remedy but a
#: different one -- and it would hide the failure being demonstrated here.
PRIOR_STD = 1e6
NOISE_STD = 1.0

#: Five sweeps, not fifty. Measured: the answer at 5 sweeps and at 200 agrees
#: to four figures (1446 K vs 1447 K), because the solve reaches the solution
#: manifold almost immediately and then has nowhere left to move. Iterating
#: longer is not a remedy, and the test should not imply it might be.
N_SWEEP = 5


def _template() -> State:
    return State(
        data=jnp.zeros((N_TIME, N_FREQ)),
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 80e6, N_FREQ),
        ),
    )


def _bilinear_space() -> ParameterSpace:
    """Both latents declared ``linear=True`` — the honest declaration.

    This is not a mistake the user makes. Each conditional genuinely IS
    affine, ``check_linearity`` confirms it, and the declaration is correct.
    The model is bilinear anyway. That gap is the whole subject.
    """
    return ParameterSpace(
        latents=[
            Latent("gain", init=GAIN0, linear=True),
            Latent("t_ant", init=T_ANT0, linear=True),
        ],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind("t_ant", into=lambda p: p["t_ant"].t_ant),
        ],
    )


def _alternating_solve(gain_scale: float, t_ant_scale: float) -> dict:
    """Run the hand-rolled sweep and report what each guard read.

    Deliberately hand-rolled rather than driven through ``SamplingPlan``: the
    plan refuses this space before the first sweep, and the failure only
    exists for someone who does not have the plan.
    """
    pipeline, template = make_pipeline(TONE_KELVIN), _template()
    truth = pipeline(template).data  # noiseless: nothing here is a noise effect
    space = _bilinear_space()

    params = {"gain": GAIN0 * gain_scale, "t_ant": T_ANT0 * t_ant_scale}
    residual = kappa = None
    linearity_refused = False

    for _ in range(N_SWEEP):
        for name in ("gain", "t_ant"):
            block = linear_operator(space, pipeline, template, names=(name,), at=params)
            try:
                check_linearity(space, pipeline, template, names=(name,), at=params)
            except ParameterSpaceError:
                linearity_refused = True
            prior_std = {name: jnp.full_like(params[name], PRIOR_STD)}
            value, residual = wiener_solve(
                block,
                truth,
                noise_std=NOISE_STD,
                prior_mean={name: jnp.zeros_like(params[name])},
                prior_std=prior_std,
                require_convergence=None,
            )
            params = {**params, name: value[name]}
            kappa = condition_estimate(block, noise_std=NOISE_STD, prior_std=prior_std)

    predicted = pipeline.at(
        {"t_ant": {"t_ant": params["t_ant"]}, "gain": {"gain": params["gain"]}}
    )(template).data if hasattr(pipeline, "at") else None
    return {
        "rms_error_kelvin": float(jnp.sqrt(jnp.mean((params["t_ant"] - T_ANT0) ** 2))),
        "max_error_kelvin": float(jnp.max(jnp.abs(params["t_ant"] - T_ANT0))),
        "residual": float(residual),
        "kappa": float(kappa),
        "linearity_refused": linearity_refused,
        "predicted": predicted,
        "params": params,
    }


#: ``(gain_scale, t_ant_scale, approximate rms error in kelvin)``. The last
#: row starts AT the truth and is the control: without it, a bug that simply
#: returned the initial values would reproduce every other row exactly.
STARTS = [
    pytest.param(0.5, 2.0, 2962.0, id="off-by-100pc"),
    pytest.param(0.8, 1.25, 703.6, id="off-by-25pc"),
    pytest.param(0.9, 1.1, 276.8, id="off-by-10pc"),
    pytest.param(0.99, 1.01, 27.35, id="off-by-1pc"),
    pytest.param(1.0, 1.0, 0.0, id="starts-at-the-truth"),
]


class TestTheGuardsCannotSeeIt:
    @pytest.mark.parametrize(("gain_scale", "t_ant_scale", "expected_rms"), STARTS)
    def test_error_is_the_initial_offset_carried_along_the_null_direction(
        self, gain_scale, t_ant_scale, expected_rms
    ):
        """The distance from truth is set by where you started, not by the data.

        Pinned to 5 % because the value is a deterministic function of the
        fixture, not a sample -- a loose tolerance here would let a genuine
        change in the degeneracy pass as rounding.

        The ``abs=1.0`` floor exists for one row only. Where the offset is
        zero there is nothing to carry, so what remains is float32 arithmetic
        rather than geometry: it reads 0.014 K here and 0.113 K after forty
        sweeps, and pinning either would be pinning rounding. Every other row
        is orders of magnitude above the floor and is held by ``rel``.
        """
        got = _alternating_solve(gain_scale, t_ant_scale)
        assert got["rms_error_kelvin"] == pytest.approx(
            expected_rms, rel=0.05, abs=1.0
        ), got

    @pytest.mark.parametrize(("gain_scale", "t_ant_scale", "expected_rms"), STARTS)
    def test_every_per_block_guard_reads_green_regardless(
        self, gain_scale, t_ant_scale, expected_rms
    ):
        """Including on the row that is 2962 K wrong, and on the row that is right."""
        got = _alternating_solve(gain_scale, t_ant_scale)
        assert not got["linearity_refused"], "check_linearity refused; it should not"
        assert got["residual"] < 1e-5, got["residual"]
        assert 1.0 < got["kappa"] < 2.0, got["kappa"]

    def test_the_guards_do_not_separate_the_best_run_from_the_worst(self):
        """The assertion the six docstrings were really making.

        Two runs whose errors differ by four decades produce condition numbers
        within a few percent of each other and residuals of the same order. No
        threshold on either number can be placed between them, which is why
        the remedy had to be a different measurement rather than a tighter one.
        """
        worst = _alternating_solve(0.5, 2.0)
        best = _alternating_solve(1.0, 1.0)

        decades = worst["rms_error_kelvin"] / best["rms_error_kelvin"]
        assert decades > 1e4, decades

        kappa_spread = abs(worst["kappa"] - best["kappa"]) / best["kappa"]
        assert kappa_spread < 0.05, (worst["kappa"], best["kappa"])
        assert worst["residual"] < 1e-5 and best["residual"] < 1e-5

    def test_the_wrong_answer_fits_the_data(self):
        """It is not a failed fit. That is what makes it silent.

        A solve that landed 2962 K from the truth AND fit the data badly would
        be caught by any residual check. This one reproduces the observation,
        because the direction it moved along changes no prediction.
        """
        got = _alternating_solve(0.5, 2.0)
        pipeline, template = make_pipeline(TONE_KELVIN), _template()
        truth = pipeline(template).data
        forward, _initial = _bilinear_space().forward_fn(pipeline, template)
        predicted = forward(got["params"])

        assert got["rms_error_kelvin"] > 1000.0
        mismatch = float(jnp.max(jnp.abs(predicted - truth)))
        assert mismatch < 1e-2 * float(jnp.max(jnp.abs(truth))), mismatch


class TestWhatDoesSeeIt:
    def test_the_rank_test_refuses_the_model_before_any_start_point_is_chosen(self):
        """The contrast that makes the diagnostic worth its dense SVD.

        Note what it does NOT depend on: no start point, no sweep, no data. The
        per-block numbers above are properties of a run; this is a property of
        the parameterization, which is the thing that is actually wrong.
        """
        report = identifiability(
            _bilinear_space(), make_pipeline(TONE_KELVIN), _template()
        )
        assert report.nullity > 0, report
        # And it names WHICH latents are degenerate, not merely that something
        # is: a verdict of "rank deficient" with no direction would leave the
        # caller exactly as stuck as the per-block numbers did.
        #
        # Measured 0.500/0.500, and that is the fingerprint rather than a
        # coincidence: the null direction is "scale the gain up, scale T_ant
        # down by the same factor", so once the Jacobian's columns are
        # normalised the two carry it equally. A lopsided split would mean the
        # degeneracy found here is NOT the gain-temperature trade the whole
        # module is about, so the tight window is the informative assertion.
        participation = report.participation(0)
        assert set(participation) == {"gain", "t_ant"}, participation
        assert participation["gain"] == pytest.approx(0.5, abs=0.01), participation
        assert participation["t_ant"] == pytest.approx(0.5, abs=0.01), participation
