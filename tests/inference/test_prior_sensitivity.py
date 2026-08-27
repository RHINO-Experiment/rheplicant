"""``prior_sensitivity`` — the question a chain cannot answer about itself.

The tour's nonlinear pair declares ``fg_beta ~ Normal(2.3, 0.3)`` and the data
put the mode at 2.5531. The prior therefore pulls, and the whole question is by
how much relative to the width being reported. Measured here: **0.0069 sigma**.
A NUTS run cannot answer that. The Monte Carlo standard error of a posterior
mean from ``n_eff`` draws is ``1/sqrt(n_eff)`` sigma, so resolving 0.0069 sigma
needs ``n_eff`` of order 2 x 10^4 *and* a second chain run without the prior to
difference against — two chains whose noise adds. The diagnostic under test is
deterministic and costs two Newton solves.

Every number below was measured twice by construction: a closed form (one
linear solve — ``H^-1 P (m - theta_hat)`` for the shift as reported, and
``(H + P_s)^-1 P_s (I + H^-1 P_d)(m - theta_hat)`` for the counterfactual
ladder, which is the same identity anchored at the likelihood mode and reduces
to the first when ``P_s = P_d``) and an independent Newton refit written in
this file, which shares no code with ``sensitivity.py``. Where the two
disagree, both are printed in the assertion.

Three pins deserve naming up front, because they are the ones that would move
if the implementation drifted:

* the tour's **likelihood-only** mode is ``(7.824320254, 2.553087147)`` and its
  MAP under the declared priors is ``(7.824320989, 2.553069844)``. Those differ
  in the fifth decimal of beta, and that difference IS the answer: 1.73e-5 in
  beta over a 2.499e-3 sigma is the 0.0069.
* the s-ladder, seven prior widths from 3.0 down to 0.01, against the column
  measured in the design phase. The refit reproduces it to 3.7e-5 relative at
  every row; the closed form to 3.6e-4 down to ``s = 0.025`` and 1.8e-3 at
  ``s = 0.01``, where the shift is six sigma and a quadratic expansion about
  the mode is being asked for more than it has.
* ``criterion_std = 0.0795`` — the prior width at which the tour's beta would
  move by 0.1 sigma. The declared 0.3 is 3.8x looser than that.
"""

import contextlib
import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from bayesmith.diagnose import sensitivity as bayesmith_sensitivity

from rheplicant import Coordinates, Environment, State
from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.inference import Bind, Latent, ParameterSpace, sensitivity
from rheplicant.inference.noise import NoiseModelLikelihood
from rheplicant.inference.sensitivity import (
    CRITERION_SHIFT,
    VERIFY_ATOL,
    VERIFY_RTOL,
    PriorSensitivityReport,
    prior_sensitivity,
)
from rheplicant.inference.uncertainty import as_noise_model

pytest.importorskip("numpyro")
import numpyro.distributions as dist  # noqa: E402

# The tour's own grid, switch cycle, seed and operator stack. Reproduced rather
# than imported: docs/_generate_tour_figures.py is a script with 60 s of
# plotting in it, and this file needs the first twenty lines of it.
N_TIME, N_FREQ = 64, 8
ADC_SCALE, SIGMA_POST_GAIN = 0.25, 2.0
NOISE_STD = ADC_SCALE * SIGMA_POST_GAIN

#: The design phase's s-ladder for ``fg_beta``: prior width -> shift in sigmas,
#: with every other latent's prior left at its declaration. One column for both
#: routes: an independent Newton refit reproduces it to 3.7e-5 at every row,
#: and :meth:`shift_at` to 3.6e-4 down to s = 0.025 — the two are the same
#: number to the six decimals quoted until the last row, which is why that row
#: has a test of its own. Quoted to six decimals, which is why the comparisons
#: below carry an absolute floor of half a unit in the last place alongside
#: their relative tolerance — at s = 3.0 the quoted 0.000033 is a rounding of
#: 0.000033371 (refit) and 0.000033370 (closed form), and nothing is wrong.
LADDER: tuple[tuple[float, float], ...] = (
    (3.0, +0.000033),
    (1.5, -0.000177),
    (0.3, -0.006924),
    (0.1, -0.063106),
    (0.05, -0.252247),
    (0.025, -1.001566),
    (0.01, -5.942289),
)

#: Half a unit in the last quoted decimal of :data:`LADDER`.
QUOTED_ATOL = 5e-7


@pytest.fixture(scope="module", autouse=True)
def _float64():
    """The tour is a float64 run; its ADC counts are float64 counts.

    ``prior_sensitivity`` forces x64 for its own arithmetic regardless, but
    ``observed`` is built out here, and a float32 waterfall through a 12-bit
    quantizer is not the same data.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


@pytest.fixture(scope="module")
def tour(_float64):
    """The tour's twin, its data, and its nonlinear two-latent space."""
    rcj = pytest.importorskip("rhino_cal_jax", reason="rhino-cal-jax not installed")
    from rheplicant.radio import (
        ADCOperator,
        AntennaLossOperator,
        BeamSpillOperator,
        CalLoadOperator,
        ForegroundOperator,
        GainOperator,
        GlobalSignalOperator,
        NoiseOperator,
        NoiseWaveOperator,
        ReceiverOperator,
        assemble,
    )

    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    time_s = jnp.arange(float(N_TIME)) * 2.0
    state = State(
        coords=Coordinates(
            time=time_s, freq=freq, extra={"receiver_input": jnp.arange(N_TIME) % 4}
        ),
        env=Environment(temperature=jnp.array(280.0)),
        key=jax.random.key(20260806),
        meta={"telescope": "RHINO", "obs_id": "tour-001"},
    )
    gamma_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)
    gamma_src = jnp.stack([
        rcj.cable_gamma(rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92),
        rcj.termination_gamma("resistive", N_FREQ, impedance=10.0),
        rcj.cable_gamma(rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98),
        rcj.cable_gamma(
            rcj.termination_gamma("resistive", N_FREQ, impedance=150.0),
            freq, length=1.1, loss=0.95,
        ),
    ])
    bandpass = 1.0 + 0.10 * jnp.cos(2 * jnp.pi * (freq - freq[0]) / (freq[-1] - freq[0]))
    twin = assemble(
        GlobalSignalOperator(
            depth=jnp.array(0.5), centre=jnp.array(75e6), width=jnp.array(5e6)
        ),
        ForegroundOperator(
            amplitude=jnp.array(2500.0), spectral_index=jnp.array(2.55), ref_freq=70e6
        ),
        BeamSpillOperator(sky_fraction=jnp.array(0.97), t_ground=jnp.array(290.0)),
        AntennaLossOperator(efficiency=jnp.array(0.97), t_physical=jnp.array(293.0)),
        CalLoadOperator(t_load=jnp.array(300.0)),
        CalLoadOperator(t_load=jnp.array(400.0)),
        CalLoadOperator(t_load=jnp.array(1200.0)),
        NoiseWaveOperator(
            t_unc=250.0 + 20.0 * jnp.linspace(-1.0, 1.0, N_FREQ),
            t_cos=30.0 * jnp.cos(jnp.linspace(0.0, 3.0, N_FREQ)),
            t_sin=-40.0 + 8.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 2,
            t_rx=290.0 + 5.0 * jnp.linspace(-1.0, 1.0, N_FREQ) ** 3,
            gamma_src_re=gamma_src.real, gamma_src_im=gamma_src.imag,
            gamma_rec_re=gamma_rec.real, gamma_rec_im=gamma_rec.imag,
        ),
        ReceiverOperator(bandpass=bandpass / jnp.mean(bandpass)),
        GainOperator(gain=1.0 + 0.02 * jnp.sin(2 * jnp.pi * time_s / 60.0)),
        NoiseOperator(sigma=jnp.array(SIGMA_POST_GAIN)),
        ADCOperator(scale=jnp.array(ADC_SCALE), n_bits=12),
    )
    observed = twin(state).data
    space = ParameterSpace(
        latents=[
            Latent("fg_log_amp", init=jnp.log(jnp.array(2000.0)),
                   prior=dist.Normal(jnp.log(2000.0), 0.5)),
            Latent("fg_beta", init=jnp.array(2.30), prior=dist.Normal(2.3, 0.3)),
        ],
        bindings=[
            Bind("fg_log_amp", into=lambda p: p["foregrounds"].amplitude, fn=jnp.exp),
            Bind("fg_beta", into=lambda p: p["foregrounds"].spectral_index),
        ],
    )
    return {
        "state": state,
        "twin": twin,
        "fit": twin.without("noise"),
        "observed": observed,
        "space": space,
    }


@pytest.fixture(scope="module")
def report(tour):
    return prior_sensitivity(
        tour["space"], tour["fit"], tour["state"], tour["observed"], NOISE_STD
    )


@pytest.fixture(scope="module")
def conditional(tour, report):
    """``fg_beta`` alone, with ``fg_log_amp`` pinned at the joint mode."""
    return prior_sensitivity(
        tour["space"], tour["fit"], tour["state"], tour["observed"], NOISE_STD,
        names=("fg_beta",),
        at={"fg_log_amp": jnp.asarray(report.mode_of("fg_log_amp"))},
    )


# ------------------------------------------------- an independent refit --


def _newton(objective, x0, steps=80):
    """Plain undamped Newton on a scalar objective. Shares nothing with the module.

    Deliberately the simplest thing that can be written: no backtracking, no
    trust region. If this and ``sensitivity.py``'s damped solver land on the
    same point, the point is the mode and not an artefact of either search.
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    for _ in range(steps):
        step = jnp.linalg.solve(jax.hessian(objective)(x), jax.grad(objective)(x))
        x = x - step
        if float(jnp.max(jnp.abs(step))) < 1e-14:
            break
    return x


def _refit_pair(tour, prior_scale):
    """``(mode under the priors, mode under the likelihood alone)`` for the tour."""
    forward, _ = tour["space"].forward_fn(tour["fit"], tour["state"])
    loglik = NoiseModelLikelihood(as_noise_model(NOISE_STD))
    loc = jnp.array([jnp.log(2000.0), 2.3])
    scale = jnp.asarray(prior_scale)

    def neg_log_lik(x):
        return -loglik(forward({"fg_log_amp": x[0], "fg_beta": x[1]}), tour["observed"])

    def neg_log_post(x):
        return neg_log_lik(x) + 0.5 * jnp.sum(((x - loc) / scale) ** 2)

    start = jnp.array([jnp.log(2000.0), 2.30])
    return _newton(neg_log_post, start), _newton(neg_log_lik, start)


def _conditional_refit(tour, log_amp, prior_std):
    """``fg_beta``'s displacement in K with ``fg_log_amp`` pinned, from two solves.

    The conditional twin of :func:`_refit_pair`. Shared by the two boundary
    tests below so that the closed form's fixed-``H`` counterfactual and the
    refit's moving one are differenced against the same solver rather than
    against two spellings of it.
    """
    forward, _ = tour["space"].forward_fn(tour["fit"], tour["state"])
    loglik = NoiseModelLikelihood(as_noise_model(NOISE_STD))

    def neg_log_lik(x):
        return -loglik(
            forward({"fg_log_amp": log_amp, "fg_beta": x[0]}), tour["observed"]
        )

    def neg_log_post(x):
        return neg_log_lik(x) + 0.5 * ((x[0] - 2.3) / prior_std) ** 2

    start = jnp.array([2.30])
    return float(_newton(neg_log_post, start)[0] - _newton(neg_log_lik, start)[0])


# ----------------------------------------------------------- the answer --


class TestTheTourPosterior:
    def test_the_mode_is_where_the_design_phase_measured_it(self, report, tour):
        mine, _ = _refit_pair(tour, [0.5, 0.3])
        assert report.mode_of("fg_log_amp") == pytest.approx(7.824320989, rel=1e-8)
        assert report.mode_of("fg_beta") == pytest.approx(2.553069844, rel=1e-8)
        assert np.asarray(mine) == pytest.approx(
            [report.mode_of("fg_log_amp"), report.mode_of("fg_beta")], rel=1e-9
        )

    def test_the_likelihood_only_mode_is_the_reference_the_shift_is_measured_from(
        self, report, tour
    ):
        """And it is the pair the design phase quoted, to every digit it quoted.

        Worth stating because the two modes agree to five decimals: any code
        that accidentally used one where it meant the other would still look
        right in a print, and would report a shift of exactly zero.
        """
        _, likelihood_only = _refit_pair(tour, [0.5, 0.3])
        assert float(likelihood_only[0]) == pytest.approx(7.824320254, abs=5e-10)
        assert float(likelihood_only[1]) == pytest.approx(2.553087147, abs=5e-10)
        assert float(likelihood_only[1]) != pytest.approx(report.mode_of("fg_beta"), rel=1e-7)

    def test_the_widths_and_the_offsets(self, report):
        assert report.for_latent("fg_log_amp")["sigma_post"] == pytest.approx(
            2.9775575e-04, rel=1e-6
        )
        assert report.for_latent("fg_beta")["sigma_post"] == pytest.approx(
            2.4990616e-03, rel=1e-6
        )
        # |m - theta_hat|: 2.3 is 101 sigma below the mode, in beta's own sigmas.
        assert report.for_latent("fg_beta")["mean_offset"] == pytest.approx(
            0.253069844, rel=1e-8
        )
        assert report.for_latent("fg_log_amp")["mean_offset"] == pytest.approx(
            0.223418530, rel=1e-8
        )


class TestTheShiftAtTheDeclaredPrior:
    def test_the_tour_prior_moves_beta_by_seven_thousandths_of_a_sigma(self, report):
        assert report.for_latent("fg_beta")["shift_sigma"] == pytest.approx(
            -0.0069239167, rel=1e-6
        )
        assert report.for_latent("fg_log_amp")["shift_sigma"] == pytest.approx(
            +0.0024711038, rel=1e-6
        )

    def test_the_sign_says_which_way_and_it_is_towards_the_prior_mean(self, report):
        """2.3 sits BELOW the mode, so beta is pulled down: the shift is negative.

        ``mean_offset`` is a magnitude by construction, so the sign lives only
        in ``shift_sigma`` — and a diagnostic that reported |shift| would say a
        prior pulling the wrong way and one pulling the right way are the same
        situation.
        """
        assert report.for_latent("fg_beta")["shift_sigma"] < 0.0
        assert report.for_latent("fg_beta")["mean_offset"] > 0.0

    def test_an_independent_refit_confirms_it_and_the_report_says_so(self, report, tour):
        mode, likelihood_only = _refit_pair(tour, [0.5, 0.3])
        refit = np.asarray(mode - likelihood_only) / np.asarray(report.sigma_post)
        assert refit == pytest.approx(np.asarray(report.shift_sigma_refit), rel=1e-7)
        # closed form vs refit: 9.6e-6 (log-amp) and 2.1e-6 (beta) relative,
        # measured. Both are second-order residues of the model's nonlinearity
        # over the displacement — the derivation itself is exact on a quadratic,
        # which is what TestTheClosedFormAgreesWithTheRefit pins on an affine
        # model to 2.4e-10.
        assert np.asarray(report.shift_sigma) == pytest.approx(refit, rel=1e-3)
        assert report.refit_converged
        assert bool(np.all(report.verified))

    @pytest.mark.parametrize(
        ("name", "declared_std", "diagonal", "cross"),
        [
            ("fg_log_amp", 0.5, -2.6609660e-04, +2.7372005e-03),
            ("fg_beta", 0.3, -7.0270792e-03, +1.0316250e-04),
        ],
    )
    def test_the_cross_term_from_the_other_prior_is_not_negligible(
        self, report, name, declared_std, diagonal, cross
    ):
        """The module docstring's table, executed rather than asserted in prose.

        ``sigma_post * (m - theta_hat) / s^2`` is the per-latent scalar law —
        this latent's own prior, pulling on this latent. For ``fg_log_amp`` it
        says **-2.66e-4** and the answer is **+2.47e-3**: ten times larger and
        the other way round, because beta's prior drags the amplitude up
        through a -0.39 correlation. A diagnostic built on the scalar law would
        report the direction of that bias wrongly, which is worse than
        reporting its size wrongly.
        """
        entry = report.for_latent(name)
        law = (
            float(entry["sigma_post"])
            * (float(entry["prior_loc"]) - float(entry["mode"]))
            / declared_std**2
        )
        assert law == pytest.approx(diagonal, rel=1e-4)
        assert float(entry["shift_sigma"]) - law == pytest.approx(cross, rel=1e-4)

    def test_the_worst_offender_is_named_not_indexed(self, report):
        name, index, value = report.worst
        assert (name, index) == ("fg_beta", 0)
        assert value == pytest.approx(-0.0069239167, rel=1e-6)


class TestTheLadder:
    """Seven prior widths across 2.5 decades, both routes, one column."""

    @pytest.mark.parametrize(("prior_std", "expected"), LADDER)
    def test_an_independent_refit_reproduces_the_measured_column(
        self, report, tour, prior_std, expected
    ):
        mode, likelihood_only = _refit_pair(tour, [0.5, prior_std])
        # Expressed in the DECLARED posterior sigma, so the rows of the ladder
        # are comparable with each other and with `criterion_std`. Dividing each
        # row by its own sigma instead would fold the prior's shrinking of the
        # error bar into a number that is supposed to report only the shift.
        refit = float(mode[1] - likelihood_only[1]) / float(
            report.for_latent("fg_beta")["sigma_post"]
        )
        assert refit == pytest.approx(expected, rel=1e-3, abs=QUOTED_ATOL)

    @pytest.mark.parametrize(("prior_std", "expected"), LADDER[:-1])
    def test_the_closed_form_tracks_it_down_to_a_prior_of_0_025(
        self, report, prior_std, expected
    ):
        assert float(report.shift_at("fg_beta", prior_std)) == pytest.approx(
            expected, rel=1e-3, abs=QUOTED_ATOL
        )

    def test_at_a_prior_of_0_01_the_closed_form_has_drifted_to_two_digits(
        self, report, tour
    ):
        """Six sigma of shift is past what one expansion about the mode covers.

        Stated as a measurement rather than left as a silently loosened
        tolerance: 1.82e-3 relative against a refit run here, and 3.2e-4 at
        s = 0.025 one row up. The refit is still good to 3.7e-5 there, which is
        why the report ships both routes and not just the cheap one.

        Nothing in the closed form is being approximated here except the model:
        the counterfactual is exact on a quadratic, so what is left at six
        sigma is the tour's own nonlinearity over that distance. Re-declaring
        the prior at 0.01 and solving again lands 1.8e-3 away too, which
        ``TestTheCounterfactualIsAnchoredWhereItCanBe`` pins from the other
        side.

        The refit is measured rather than read off :data:`LADDER`, so that this
        is a comparison of two computations and not of two pinned constants —
        the quoted -5.942289 is asserted separately, on the same solve.
        """
        closed = float(report.shift_at("fg_beta", 0.01))
        assert closed == pytest.approx(-5.953311, rel=1e-3)
        mode, likelihood_only = _refit_pair(tour, [0.5, 0.01])
        refit = float(mode[1] - likelihood_only[1]) / float(
            report.for_latent("fg_beta")["sigma_post"]
        )
        assert refit == pytest.approx(-5.942289, rel=1e-3)
        assert abs(closed - refit) / abs(refit) == pytest.approx(1.82e-3, rel=0.05)

    def test_the_declared_row_of_the_ladder_is_the_reported_shift(self, report):
        """To the last bit, and that is a statement about the derivation.

        The two are written differently — ``shift_sigma`` is
        ``H^-1 P (m - theta_hat)``, anchored at the mode it has; the ladder is
        ``(H + P_s)^-1 P_s (I + H^-1 P_d)(m - theta_hat)``, anchored at the
        likelihood mode it reconstructs. Put ``P_s = P_d`` and the second
        collapses onto the first algebraically, so agreement here is not a
        coincidence to be checked loosely: measured at 2.5e-16 on beta and
        1.8e-16 on log-amp, which is float64 saying "the same number".

        A counterfactual that dropped the ``(I + H^-1 P_d)`` factor would still
        look reasonable — 6.9e-5 from the reported shift, well inside anything
        a reader would call agreement — and this is the assertion that would
        not have it.
        """
        for name, declared in (("fg_beta", 0.3), ("fg_log_amp", 0.5)):
            assert float(report.shift_at(name, declared)) == pytest.approx(
                float(report.for_latent(name)["shift_sigma"]), rel=1e-12
            )


class TestTheOneOverSSquaredLawAtItsExtremes:
    """Boundary validation: the law is asserted where it must break, not in the middle.

    Run on the CONDITIONAL problem — ``fg_beta`` alone, ``fg_log_amp`` pinned at
    the joint mode — because the joint shift carries a cross term from the OTHER
    latent's prior that has no ``1/s^2`` in it at all. That term is what makes
    the joint ladder come back POSITIVE at s = 3.0 while beta's own prior is
    still pulling down, and testing the scaling law through it would be testing
    a sum of two laws.
    """

    def test_pinning_the_other_latent_narrows_beta_from_2_499e_3_to_2_302e_3(
        self, conditional, report
    ):
        """The conditional width, not the marginal one — and they are 8% apart."""
        assert conditional.for_latent("fg_beta")["sigma_post"] == pytest.approx(
            2.301707e-03, rel=1e-5
        )
        assert conditional.mode_of("fg_beta") == pytest.approx(
            report.mode_of("fg_beta"), rel=1e-9
        )

    @pytest.mark.parametrize(
        ("prior_std", "expected"),
        [
            (300.0, -6.47290275e-09),
            (3.0, -6.47289894e-05),
            (0.3, -6.47252172e-03),
            (0.03, -6.43502073e-01),
            (0.01, -5.53249179e00),
        ],
    )
    def test_the_closed_form_across_four_and_a_half_decades_of_prior_width(
        self, conditional, prior_std, expected
    ):
        """Four and a half decades of ``1/s^2``, and where it stops being that.

        With one latent selected there is no cross term, so the only structure
        left is the law and the curvature. The top three rows are ``1/s^2`` to
        the digits printed — 6.4729e-9, 6.4729e-5, 6.4725e-3 — and the bottom
        two are not: by s = 0.03 the hypothesised prior has entered ``H + P_s``
        and pulled the mantissa down to 6.4350, and by 0.01 to 5.5325. That
        departure is the subject of the two tests below.

        The third row is the declared width, so it is also this report's
        ``shift_sigma`` to the last bit — the ladder passes through the number
        it is a counterfactual around.
        """
        assert float(conditional.shift_at("fg_beta", prior_std)) == pytest.approx(
            expected, rel=1e-6
        )

    def test_at_the_loose_extreme_the_law_is_exact(self, conditional):
        """s = 300 against s = 3: two decades, ``shift * s^2`` constant to 6e-7.

        This is the regime ``criterion_std`` is derived in — it inverts the law
        — so it is the regime the law has to hold in. 5.9e-7 is measured, and
        it is not roundoff: it is ``P/(H + P)`` at s = 3, the same term that
        reaches 5% at the tight end below.
        """
        loose = float(conditional.shift_at("fg_beta", 300.0)) * 300.0**2
        mid = float(conditional.shift_at("fg_beta", 3.0)) * 3.0**2
        assert loose == pytest.approx(mid, rel=1e-6)
        assert loose == pytest.approx(-5.8256125e-04, rel=1e-6)

    def test_at_the_tight_extreme_the_law_bends_by_five_percent(
        self, conditional, tour, report
    ):
        """And that is not a bug — it is the prior becoming the curvature.

        At s = 0.01 the prior's own curvature 1/s^2 = 1e4 has reached 5.298% of
        the likelihood's 1.8874e5, so ``Sigma_s`` is no longer ``Sigma``: it is
        smaller by ``1 - h/(h + 1e4)`` = 5.032%, and the invariant
        ``shift * s^2`` falls by exactly that. The two numbers agreeing is the
        content — the departure from the law IS the prior entering the
        curvature, and nothing else. A sweep of moderate widths would report a
        clean 1/s^2 law and never reach the place the law is used near its
        limit.

        Those first two assertions are algebra: ``(H + P_s)^-1`` contains the
        share, so it reproduces it whatever the model does. The third is not —
        an actual REFIT's ``shift * s^2`` falls by 4.941% across the same span,
        which puts the closed form's 5.032% within 1.8% of a number no formula
        here supplied. The residue is the model's nonlinearity over the five
        and a half sigma the mode travels to get there.
        """
        loose = float(conditional.shift_at("fg_beta", 300.0)) * 300.0**2
        tight = float(conditional.shift_at("fg_beta", 0.01)) * 0.01**2
        assert 1.0 - tight / loose == pytest.approx(0.05032, rel=1e-3)
        # And it is the prior's share of the curvature, not a coincidence.
        curvature = 1.0 / float(conditional.for_latent("fg_beta")["sigma_post"]) ** 2
        likelihood_only = curvature - 1.0 / 0.3**2
        assert 1.0 - likelihood_only / (likelihood_only + 1.0 / 0.01**2) == pytest.approx(
            1.0 - tight / loose, rel=1e-3
        )

        log_amp = jnp.asarray(report.mode_of("fg_log_amp"))
        refit_loose = _conditional_refit(tour, log_amp, 300.0) * 300.0**2
        refit_tight = _conditional_refit(tour, log_amp, 0.01) * 0.01**2
        bend = 1.0 - refit_tight / refit_loose
        assert bend == pytest.approx(0.04941, rel=1e-3)
        assert 1.0 - tight / loose == pytest.approx(bend, rel=0.03)

    @pytest.mark.parametrize(
        ("prior_std", "expected_disagreement"), [(300.0, 2.83e-5), (0.01, 9.25e-4)]
    )
    def test_both_routes_agree_at_both_extremes(
        self, conditional, tour, report, prior_std, expected_disagreement
    ):
        """The boundary-validation check proper: closed form vs refit, at the ends.

        Not through the report's own dispatch — this refits the conditional
        problem from scratch, priors on and priors off, and differences the two
        modes. Both ends are checked, because a fix that improved one at the
        cost of the other would otherwise pass. That is not hypothetical: an
        anchor at ``theta_hat`` rather than at ``theta_L`` holds the loose end
        to 3.1e-5 and takes the tight one to 5.2e-2.

        **The discrimination lives at the TIGHT end, and only there.** Measured
        2026-08-26: the loose end's disagreement is 2.83e-5 on arm64 macOS and
        **5.81e-5** on x86_64 Linux -- a factor of 2.05 between two CORRECT
        runs, while the distance from the correct value to the wrong anchor's
        3.1e-5 is a factor of 1.1. A tolerance wide enough to admit the second
        machine necessarily admits the defect, so the loose end cannot separate
        them at any tolerance whatever. Pinning it to 5 % never made it
        discriminating; it made it pass on one laptop. It is asserted here as
        an order of magnitude, which is what it is worth.

        The tight end keeps its 5 %, passes on both platforms, and is what
        actually catches the anchor defect: 5.2e-2 against 9.25e-4 is a factor
        of 56, far outside any platform spread.
        """
        sigma = float(conditional.for_latent("fg_beta")["sigma_post"])
        log_amp = jnp.asarray(report.mode_of("fg_log_amp"))
        refit = _conditional_refit(tour, log_amp, prior_std) / sigma
        closed = float(conditional.shift_at("fg_beta", prior_std))
        disagreement = abs(closed - refit) / abs(refit)
        if prior_std == 0.01:
            assert disagreement == pytest.approx(expected_disagreement, rel=0.05)
        else:
            assert (
                0.2 * expected_disagreement
                < disagreement
                < 5 * expected_disagreement
            ), disagreement
        assert closed == pytest.approx(refit, rel=3e-3)


@pytest.fixture(scope="module")
def vector_report(tour):
    """One latent of EIGHT elements — the tour's receiver noise-wave spectrum.

    Every other fixture here is scalar, and a scalar latent hides three things
    at once: ``for_latent``'s reshape, ``worst``'s index-within-the-latent
    arithmetic, and ``shift_at``'s slice of the flat vector are all the
    identity when the span is one element wide.

    Declared ``linear=True`` WITH a prior, which is the configuration that is
    fine — what is refused is ``linear=True`` with the prior left to a
    ``prior_std=`` argument.
    """
    space = ParameterSpace(
        latents=[
            Latent(
                "t_rx",
                init=jnp.full((N_FREQ,), 280.0),
                prior=dist.Normal(jnp.full((N_FREQ,), 280.0), 40.0),
                linear=True,
            )
        ],
        bindings=[Bind("t_rx", into=lambda p: p["noise_wave"].t_rx)],
    )
    return prior_sensitivity(
        space, tour["fit"], tour["state"], tour["observed"], NOISE_STD
    )


class TestAVectorLatent:
    def test_every_per_element_column_is_shaped_like_the_latent(self, vector_report):
        assert vector_report.names == ("t_rx",)
        assert vector_report.spans == ((0, N_FREQ),)
        assert vector_report.n_par == N_FREQ
        entry = vector_report.for_latent("t_rx")
        for field, array in entry.items():
            assert array.shape == (N_FREQ,), field
        # The prior was declared with a vector loc and a scalar scale; both are
        # broadcast to the latent, so neither ends up covering one element.
        assert entry["prior_loc"] == pytest.approx(np.full(N_FREQ, 280.0))
        assert entry["prior_std"] == pytest.approx(np.full(N_FREQ, 40.0))

    def test_one_exact_newton_step_plus_one_to_confirm_it_did_not_move(
        self, vector_report
    ):
        """A noise-wave temperature enters the prediction affinely.

        The negative log-posterior is then exactly quadratic, one Newton step
        is exact, and a second confirms it did not move — which is what a
        converged solve on a quadratic costs and a useful check that the damped
        line search is not shortening steps it should not.
        """
        assert vector_report.newton_steps == 2
        assert vector_report.refit_steps == 2

    def test_worst_names_the_element_and_the_index_agrees_with_the_column(
        self, vector_report
    ):
        name, index, value = vector_report.worst
        assert name == "t_rx"
        column = vector_report.for_latent("t_rx")["shift_sigma"]
        assert index == int(np.argmax(np.abs(column)))
        # 7, the top channel. Non-zero is the load-bearing part: an index
        # returned as the span's START rather than the offset within it would
        # be 0 here and correct for every scalar latent in this file.
        assert index == N_FREQ - 1
        assert value == pytest.approx(column[index])

    def test_a_scalar_width_and_a_vector_of_that_width_are_the_same_question(
        self, vector_report
    ):
        scalar = vector_report.shift_at("t_rx", 4.0)
        vector = vector_report.shift_at("t_rx", np.full(N_FREQ, 4.0))
        assert scalar.shape == (N_FREQ,)
        assert scalar == pytest.approx(vector, rel=1e-14)
        # 40 -> 4 is a hundredfold tightening of 1/s^2, and this latent has no
        # other prior to cross-talk with, so the law holds here to 4.76e-3 --
        # the tightened prior's own share of (H + P_s), which is what keeps
        # this a comparison rather than an equality. 1e-2 and not the 2e-2 it
        # was: the share is a computed quantity, not slack.
        assert scalar == pytest.approx(
            100.0 * vector_report.for_latent("t_rx")["shift_sigma"], rel=1e-2
        )

    def test_a_per_element_width_moves_only_that_element_a_lot(self, vector_report):
        widths = np.full(N_FREQ, 40.0)
        widths[3] = 0.5
        shifted = vector_report.shift_at("t_rx", widths)
        declared = vector_report.for_latent("t_rx")["shift_sigma"]
        assert abs(shifted[3]) > 100.0 * abs(declared[3])
        # The others move too — the posterior is correlated across frequency —
        # but by nothing like that factor.
        others = [i for i in range(N_FREQ) if i != 3]
        assert np.all(np.abs(shifted[others]) < 100.0 * np.abs(declared[others]))

    def test_both_routes_agree_on_all_eight_and_the_floor_is_the_refit_s(
        self, vector_report
    ):
        """2.1e-11 to 2.4e-10 relative — and here it is the REFIT that is inexact.

        The prediction is affine in a noise-wave temperature and the noise is
        homoscedastic, so the negative log-posterior is exactly quadratic and
        the closed form is exact by construction. The refit is not: it recovers
        a 5e-4 K displacement by differencing two modes of about 290 K each, so
        roughly six digits are gone to cancellation before the linear solves'
        own roundoff is reached. On the tour's nonlinear pair the roles are the
        other way round. That is the case for shipping both and reporting
        ``verified`` rather than picking one and calling it the answer.

        That cancellation argument predicts ``290 * eps / 5e-4`` = 1.3e-10, and
        the measured worst element is 2.4e-10 — so it is now the whole story.
        It was not while this test read 4.8e-5: no amount of cancellation in a
        float64 difference of two 290 K numbers reaches five decimal places,
        and the 4.8e-5 was the ``(H + P)^-1`` closed form's own error wearing
        the refit's name. The floor is bounded within a decade of the
        prediction rather than pinned to a digit, because a quantity made of
        last-bit cancellation is not reproducible across LAPACK builds the way
        a derivation error is.
        """
        assert vector_report.refit_converged
        assert bool(np.all(vector_report.verified))
        assert vector_report.shift_sigma == pytest.approx(
            vector_report.shift_sigma_refit, rel=1e-8
        )
        disagreement = np.abs(
            vector_report.shift_sigma / vector_report.shift_sigma_refit - 1.0
        )
        cancellation = 290.0 * float(np.finfo(np.float64).eps) / 5e-4
        assert cancellation / 10.0 < disagreement.max() < cancellation * 10.0

    def test_this_prior_is_the_clean_bill_of_health_the_tour_s_beta_is_not(
        self, vector_report
    ):
        """Declared 40 K against a criterion of 3.4-5.9 K: a factor of 7 to 12.

        Worth one test because a diagnostic that only ever fires is not a
        diagnostic. Here every element's ``criterion_std`` sits an order of
        magnitude BELOW the declared width, so no element could be moved 0.1
        sigma by this prior, and the reported shifts are all under 0.0022 sigma.
        """
        entry = vector_report.for_latent("t_rx")
        assert np.all(entry["criterion_std"] < entry["prior_std"] / 6.0)
        assert np.abs(entry["shift_sigma"]).max() < 0.0022
        assert entry["criterion_std"].min() == pytest.approx(3.4337, rel=1e-3)
        assert entry["criterion_std"].max() == pytest.approx(5.8647, rel=1e-3)


class TestTheCriterion:
    def test_the_tour_beta_would_need_a_prior_of_0_0795_to_move_a_tenth_of_a_sigma(
        self, report
    ):
        assert report.for_latent("fg_beta")["criterion_std"] == pytest.approx(
            0.0795, rel=1e-3
        )
        assert report.for_latent("fg_log_amp")["criterion_std"] == pytest.approx(
            0.025792276, rel=1e-6
        )

    def test_the_declared_prior_is_almost_four_times_looser_than_the_criterion(
        self, report
    ):
        assert 0.3 / float(report.for_latent("fg_beta")["criterion_std"]) == pytest.approx(
            3.77, rel=0.01
        )

    def test_evaluating_the_shift_at_the_criterion_returns_a_tenth_of_a_sigma(
        self, report
    ):
        """To 0.19% on beta and 2.75% on log-amp — the cross term, not an error.

        ``criterion_std`` inverts the DIAGONAL law, which is the only part of
        the shift that has an ``s`` in it; the cross term from the other
        latent's prior rides along unchanged. So the criterion is a criterion
        and not an identity, and the size of the gap is the size of the
        correlation.
        """
        for name, gap in (("fg_beta", 0.0019), ("fg_log_amp", 0.0275)):
            criterion = float(report.for_latent(name)["criterion_std"])
            landed = abs(float(report.shift_at(name, criterion)))
            assert landed == pytest.approx(CRITERION_SHIFT, rel=0.05)
            assert 1.0 - landed / CRITERION_SHIFT == pytest.approx(gap, abs=0.004)


class TestTheReportItself:
    def test_it_is_reachable_from_the_package_namespace(self):
        """A diagnostic nobody can import is a diagnostic nobody runs."""
        import rheplicant.inference as inference

        assert inference.prior_sensitivity is prior_sensitivity
        assert inference.PriorSensitivityReport is PriorSensitivityReport
        assert inference.CRITERION_SHIFT == CRITERION_SHIFT
        for name in ("prior_sensitivity", "PriorSensitivityReport", "CRITERION_SHIFT"):
            assert name in inference.__all__

    def test_it_is_frozen_and_holds_numpy(self, report):
        assert dataclasses.is_dataclass(PriorSensitivityReport)
        with pytest.raises(dataclasses.FrozenInstanceError):
            report.shift_sigma = np.zeros(2)
        for field in ("shift_sigma", "sigma_post", "mean_offset", "criterion_std"):
            assert isinstance(getattr(report, field), np.ndarray)
            assert getattr(report, field).dtype == np.float64

    def test_the_rows_are_in_declaration_order_not_sorted_order(self, report):
        """``fisher_information`` flattens a dict in SORTED key order.

        Sorted, ``fg_beta`` comes FIRST — so a report that borrowed that
        ordering while naming its rows in declaration order would hand back
        beta's sigma under log-amp's name, with both numbers finite and the
        shapes all agreeing. The two orderings differ for this very space,
        which is why the check is here rather than on a fixture invented for it.
        """
        assert report.names == ("fg_log_amp", "fg_beta")
        assert sorted(report.names) != list(report.names)
        assert report.spans == ((0, 1), (1, 2))
        # sigma_post[0] belongs to fg_log_amp: 2.98e-4, an order below beta's.
        assert report.sigma_post[0] < report.sigma_post[1] / 5.0

    def test_a_latent_that_is_not_there_is_named_in_the_refusal(self, report):
        with pytest.raises(StateValidationError, match="fg_gamma"):
            report.for_latent("fg_gamma")
        with pytest.raises(StateValidationError, match="fg_gamma"):
            report.shift_at("fg_gamma", 0.1)

    def test_a_nonpositive_prior_width_is_refused_rather_than_dividing_by_zero(
        self, report
    ):
        with pytest.raises(StateValidationError, match="positive"):
            report.shift_at("fg_beta", 0.0)
        with pytest.raises(StateValidationError, match="positive"):
            report.shift_at("fg_beta", -0.3)

    def test_a_width_that_does_not_broadcast_is_refused(self, report):
        with pytest.raises(StateValidationError, match="broadcast"):
            report.shift_at("fg_beta", [0.1, 0.2, 0.3])

    def test_it_counts_the_newton_steps_it_took(self, report):
        """7 from the declared init, 3 for the refit from the mode.

        Pinned loosely, as a ceiling rather than an equality: the point is that
        the constant :data:`MAX_NEWTON_STEPS` is 100 against a measured need of
        7, so a solve that hits the ceiling has diverged rather than run out of
        budget.
        """
        assert report.newton_steps == 7
        assert report.refit_steps == 3


class TestWhenNewtonDoesNotGetThere:
    """Both solves can fail, and they are not the same kind of failure."""

    def test_a_mode_that_is_not_found_is_fatal_and_says_what_it_prevents(
        self, tour, monkeypatch
    ):
        # bayesmith's constant, not this package's: the Newton solve moved
        # there when the module became a facade, and patching the
        # re-exported name here would change nothing while reading as
        # though it had -- a test that cannot fail.
        monkeypatch.setattr(bayesmith_sensitivity, "MAX_NEWTON_STEPS", 2)
        with pytest.raises(StateValidationError) as excinfo:
            prior_sensitivity(
                tour["space"], tour["fit"], tour["state"], tour["observed"], NOISE_STD
            )
        message = str(excinfo.value)
        assert "did not converge" in message
        # Naming the alternatives is the point: a bad basin, float32, or a
        # posterior this method has no business approximating.
        assert "at=" in message and "float32" in message and "NUTS" in message

    def test_a_verification_that_could_not_run_is_reported_not_raised(
        self, tour, report, monkeypatch
    ):
        """Starting AT the mode, the MAP re-converges in one step; the refit needs 3.

        So a two-step budget separates the two solves cleanly, and what comes
        back is the closed form with ``verified`` false and the refit column
        NaN — never a NaN silently averaged into a verdict.
        """
        # bayesmith's constant, not this package's: the Newton solve moved
        # there when the module became a facade, and patching the
        # re-exported name here would change nothing while reading as
        # though it had -- a test that cannot fail.
        monkeypatch.setattr(bayesmith_sensitivity, "MAX_NEWTON_STEPS", 2)
        starved = prior_sensitivity(
            tour["space"], tour["fit"], tour["state"], tour["observed"], NOISE_STD,
            at={
                "fg_log_amp": jnp.asarray(report.mode_of("fg_log_amp")),
                "fg_beta": jnp.asarray(report.mode_of("fg_beta")),
            },
        )
        assert not starved.refit_converged
        assert not bool(np.any(starved.verified))
        assert bool(np.all(np.isnan(starved.shift_sigma_refit)))
        # The closed form is unaffected — it never needed the second solve.
        assert starved.for_latent("fg_beta")["shift_sigma"] == pytest.approx(
            report.for_latent("fg_beta")["shift_sigma"], rel=1e-6
        )


# ------------------------------------------------------------- refusals --


class TestRefusals:
    def test_a_prior_with_no_quadratic_form_is_refused_by_name(self, tour):
        space = ParameterSpace(
            latents=[
                Latent("fg_log_amp", init=jnp.log(jnp.array(2000.0)),
                       prior=dist.Normal(jnp.log(2000.0), 0.5)),
                Latent("fg_beta", init=jnp.array(2.30), prior=dist.Uniform(2.0, 3.0)),
            ],
            bindings=list(tour["space"].bindings),
        )
        with pytest.raises(ParameterSpaceError) as excinfo:
            prior_sensitivity(
                space, tour["fit"], tour["state"], tour["observed"], NOISE_STD
            )
        message = str(excinfo.value)
        assert "fg_beta" in message
        assert "Uniform" in message
        # The refusal has to say what it prevents, not merely that it refused.
        assert "quadratic" in message

    def test_a_uniform_prior_is_fine_as_long_as_it_is_not_asked_about(self, tour):
        """``names=`` is the escape hatch, and it has to actually work.

        A Uniform on another latent is no obstacle to asking whether THIS
        latent's Gaussian prior is biasing it — the same conditional reading
        ``identifiability(names=...)`` takes.
        """
        space = ParameterSpace(
            latents=[
                Latent("fg_log_amp", init=jnp.log(jnp.array(2000.0)),
                       prior=dist.Uniform(6.0, 9.0)),
                Latent("fg_beta", init=jnp.array(2.30), prior=dist.Normal(2.3, 0.3)),
            ],
            bindings=list(tour["space"].bindings),
        )
        report = prior_sensitivity(
            space, tour["fit"], tour["state"], tour["observed"], NOISE_STD,
            names=("fg_beta",),
        )
        assert report.names == ("fg_beta",)
        assert float(report.shift_sigma[0]) < 0.0

    def test_a_linear_latent_whose_prior_lives_at_the_call_site_is_refused(self, tour):
        """``linear=True`` with no ``Latent(prior=...)`` means ``prior_std=``.

        That argument belongs to ``wiener_solve``/``gcr_sample`` and is supplied
        where they are CALLED. Nothing in the space records it, so a sensitivity
        computed here would be a sensitivity to a prior the solve never reads —
        and the tour's own linear block is declared exactly this way, so this is
        not a hypothetical space.
        """
        space = ParameterSpace(
            latents=[Latent("t_rx", init=jnp.zeros((N_FREQ,)), linear=True)],
            bindings=[Bind("t_rx", into=lambda p: p["noise_wave"].t_rx)],
        )
        with pytest.raises(ParameterSpaceError) as excinfo:
            prior_sensitivity(
                space, tour["fit"], tour["state"], tour["observed"], NOISE_STD,
                names=("t_rx",),
            )
        message = str(excinfo.value)
        assert "t_rx" in message
        assert "prior_std" in message
        assert "linear" in message

    def test_a_prior_free_nonlinear_latent_is_refused_too(self, tour):
        space = ParameterSpace(
            latents=[Latent("fg_beta", init=jnp.array(2.30))],
            bindings=[Bind("fg_beta", into=lambda p: p["foregrounds"].spectral_index)],
        )
        with pytest.raises(ParameterSpaceError, match="fg_beta"):
            prior_sensitivity(
                space, tour["fit"], tour["state"], tour["observed"], NOISE_STD
            )

    def test_a_rank_deficient_selection_is_refused_and_identifiability_is_named(
        self, tour
    ):
        """Two latents whose sum is the only thing the data sees.

        The posterior is still proper — the declared priors make it so — and
        every number that comes back is finite. What is NOT defined is the
        reference: the shift is measured from the mode the likelihood alone
        would choose, and along the null direction that mode is a ray. A finite
        answer here would be the prior reporting on itself.
        """
        space = ParameterSpace(
            latents=[
                Latent("fg_a", init=jnp.array(3.8), prior=dist.Normal(3.8, 0.5)),
                Latent("fg_b", init=jnp.array(3.8), prior=dist.Normal(3.8, 0.5)),
            ],
            bindings=[
                Bind(("fg_a", "fg_b"), into=lambda p: p["foregrounds"].amplitude,
                     fn=lambda a, b: jnp.exp(a + b)),
            ],
        )
        with pytest.raises(ParameterSpaceError) as excinfo:
            prior_sensitivity(
                space, tour["fit"], tour["state"], tour["observed"], NOISE_STD
            )
        message = str(excinfo.value)
        assert "identifiability" in message
        assert "rank 1 of 2" in message
        assert "fg_a" in message and "fg_b" in message

    def test_observed_of_the_wrong_shape_is_refused_before_any_solve(self, tour):
        with pytest.raises(ParameterSpaceError, match="broadcast|shape"):
            prior_sensitivity(
                tour["space"], tour["fit"], tour["state"],
                tour["observed"][0], NOISE_STD,
            )

    def test_an_undeclared_name_is_refused(self, tour):
        with pytest.raises(ParameterSpaceError, match="fg_gamma"):
            prior_sensitivity(
                tour["space"], tour["fit"], tour["state"], tour["observed"], NOISE_STD,
                names=("fg_gamma",),
            )
        with pytest.raises(ParameterSpaceError, match="fg_gamma"):
            prior_sensitivity(
                tour["space"], tour["fit"], tour["state"], tour["observed"], NOISE_STD,
                at={"fg_gamma": jnp.array(1.0)},
            )


class TestTheClosedFormAgreesWithTheRefit:
    """The refit is the ground truth; the closed form has to reproduce it.

    This is the check that catches the derivation being wrong rather than the
    arithmetic. The displacement of the MAP from the likelihood mode is

        theta_hat - theta_L = H^-1 P (m - theta_hat)

    with ``H`` the LIKELIHOOD curvature. Using the posterior's ``(H + P)^-1``
    instead -- which is the natural mistake, because that is the matrix already
    in hand for ``sigma_post`` -- is wrong by exactly ``diag((H + P)^-1 P)``,
    the prior's share of the posterior precision. It is invisible where the
    prior is weak, which is exactly where a test written against the tour's own
    numbers would be — 6.9e-5 at the declared beta — and it grows without bound
    as the prior tightens: 5.9e-2 at the tour's beta with ``s = 0.01``.

    The tolerance is the report's own ``verified`` tolerance, so a change that
    loosens one loosens the other visibly.
    """

    def test_every_latent_agrees_with_its_own_refit(self, report) -> None:
        for name in ("fg_log_amp", "fg_beta"):
            row = report.for_latent(name)
            closed, refit = float(row["shift_sigma"]), float(row["shift_sigma_refit"])
            assert closed == pytest.approx(refit, rel=VERIFY_RTOL, abs=VERIFY_ATOL), (
                f"{name}: the closed form says {closed:+.9f} and an independent "
                f"Newton refit says {refit:+.9f}. The refit is the truth here."
            )

    def test_an_affine_model_is_where_the_derivation_shows(self, _float64) -> None:
        """On an exactly quadratic posterior the refit is exact, so any
        disagreement is the derivation.

        The tour's model cannot make this call: tighten its prior far enough to
        separate ``H^-1`` from ``(H + P)^-1`` and the MAP moves far enough that
        the quadratic approximation itself breaks -- the report says so, with
        ``verified`` false, which is that flag working. An affine model has no
        such confound. Here the refit is Newton-exact in one step and the two
        candidate matrices differ by ``diag((H + P)^-1 P)``, which this prior
        makes **0.277** -- ninety times ``VERIFY_RTOL``. Getting there needs a
        deliberately starved fit: eight channels, unit-amplitude foreground and
        unit noise, so the likelihood carries H = 65.1 against the prior's
        P = 25. On the tour's own numbers H is 1.6e10 and no prior anyone would
        write is visible against it, which is why this fixture is small and
        poor on purpose.
        """
        from rheplicant.radio import (
            ForegroundOperator,
            GainOperator,
            assemble,
        )

        n = 8
        freq = jnp.linspace(60e6, 85e6, n)
        state = State(coords=Coordinates(time=jnp.arange(float(n)), freq=freq),
                      key=jax.random.key(0), meta={"telescope": "affine"})
        twin = assemble(
            ForegroundOperator(amplitude=jnp.array(1.0),
                               spectral_index=jnp.array(2.5), ref_freq=70e6),
            GainOperator(gain=jnp.array(1.1)),
        )
        observed = twin(state).data
        space = ParameterSpace(
            latents=[Latent("gain", init=jnp.array(1.0),
                            prior=dist.Normal(1.4, 0.2))],
            bindings=[Bind("gain", into=lambda p: p["gain"].gain)],
        )
        report = prior_sensitivity(space, twin, state, observed, 1.0)
        row = report.for_latent("gain")
        closed = float(row["shift_sigma"])
        refit = float(row["shift_sigma_refit"])
        assert closed == pytest.approx(refit, rel=VERIFY_RTOL, abs=VERIFY_ATOL), (
            f"closed form {closed:+.9f} against an exact refit {refit:+.9f} on a "
            "model with no nonlinearity to blame. This is H^-1 versus (H + P)^-1."
        )
        assert bool(np.all(report.verified))


class TestTheCounterfactualIsAnchoredWhereItCanBe:
    """``shift_at`` uses a different identity from ``shift_sigma``, on purpose.

    Both are exact::

        theta_s - theta_L = H^-1 P_s (m - theta_s)          [anchored at theta_s]
                          = (H + P_s)^-1 P_s (m - theta_L)  [anchored at theta_L]

    ``shift_sigma`` reports the DECLARED prior, where ``theta_hat`` *is*
    ``theta_s``, so the first form is exact and is what it uses. A counterfactual
    has no ``theta_s`` to stand on -- finding it is the re-fit the method exists
    to avoid -- so it must use the second, and it does not hold that anchor
    either. It reconstructs it: ``m - theta_L = (I + H^-1 P_d)(m - theta_hat)``
    is the declared solve's own displacement run backwards, exact, so

        shift_at = (H + P_s)^-1 P_s (I + H^-1 P_d) (m - theta_hat)

    needs nothing it does not have. Setting ``P_s = P_d`` collapses it onto
    ``shift_sigma``, which is why the declared row of the ladder is the reported
    shift to the last bit rather than 6.9e-5 away from it.

    Applying ``shift_sigma``'s form here looks like consistency and is not: it
    substitutes ``theta_hat`` for a ``theta_s`` it is nowhere near. This test
    exists because that substitution was made, passed 62 tests, and was caught
    only by running the counterfactual for real.
    """

    @pytest.mark.parametrize(
        ("hypothetical", "tolerance"), [(0.1, 1e-4), (0.01, 5e-3)]
    )
    def test_it_matches_an_actual_run_at_that_width(
        self, tour, report, hypothetical, tolerance
    ) -> None:
        """Compare against re-declaring the prior and solving again.

        Both sides are expressed in the DECLARED posterior sigma. That matters:
        the re-run's own sigma is narrower -- 0.3 -> 0.01 shrinks beta's by 2.4x
        -- and dividing by it would compare two different quantities and hide
        the very error this test is for.

        Measured, against this truth: the shipped form is 0.0016 % out at
        s = 0.1 and 0.18 % at 0.01. Dropping the ``(I + H^-1 P_d)`` anchor
        gives 0.005 % and 0.17 %; using ``H^-1 P_s (m - theta_hat)`` instead
        gives 0.06 % and **6.43 %**.

        The two tolerances are set from those measurements and differ by a
        factor of 50 on purpose. At s = 0.1 the counterfactual is exact to
        1.6e-5 and 1e-4 holds it there. At 0.01 the mode has travelled six
        sigma and 1.8e-3 of the tour's own nonlinearity is left, which no
        rearrangement of an exactly-quadratic formula removes — the anchored
        form is marginally WORSE there than the unanchored one (1.82e-3 against
        1.75e-3), both being at that floor.
        """
        index = report.names.index("fg_beta")
        lo, hi = report.spans[index]
        sigma_declared = float(np.asarray(report.sigma_post)[lo:hi][0])
        likelihood_mode = float(np.asarray(report.mode)[lo:hi][0]) - (
            float(np.asarray(report.shift_sigma_refit)[lo:hi][0]) * sigma_declared
        )

        rewidened = eqx.tree_at(
            lambda s: s.latent("fg_beta").prior,
            tour["space"],
            dist.Normal(2.3, hypothetical),
        )
        rerun = prior_sensitivity(
            rewidened, tour["fit"], tour["state"], tour["observed"], NOISE_STD
        )
        j = rerun.names.index("fg_beta")
        a, b = rerun.spans[j]
        truth = (float(np.asarray(rerun.mode)[a:b][0]) - likelihood_mode) / sigma_declared

        assert float(report.shift_at("fg_beta", hypothetical)) == pytest.approx(
            truth, rel=tolerance
        )


# ------------------------------------------ declared in single precision --


@contextlib.contextmanager
def _ambient_float32():
    """Build things the way an ordinary rheplicant session builds them.

    This module's autouse ``_float64`` fixture is right for the tour and wrong
    for this question: it makes the twin, the state and the latents float64, so
    nothing below the facade is ever asked to survive single precision.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", False)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", was)


class TestAModelDeclaredInSinglePrecision:
    """The caller declares in float32; the verdict still comes back in double.

    ``_widened`` was removed as an experiment while ``sensitivity`` was being
    switched and all 64 tests in this file stayed green — recorded as the open
    item S6 in ``2026-08-27-wave-A-sensitivity.md``, with two opposite
    dispositions: dead code, or load-bearing with no fixture to exercise it.
    Measured, it is the second, and this class is the missing fixture. Every
    other fixture here is built under the module's ``_float64``, which is right
    for the tour — its ADC counts are float64 counts — and which removes the
    condition entirely.

    **The condition is the MODEL's own arrays, not the init's dtype.** Four
    cells of ``{init, model consts} x {float32, float64}``, ``_widened``
    against ``_widened`` replaced by the identity:

    ====================  ==================  ==============================
    model consts          init                without ``_widened``
    ====================  ==================  ==============================
    float64               float64             passes
    float64               float32             passes
    float32               float64 (weak)      **refused**
    float32               float32             **refused**
    ====================  ==================  ==============================

    The third row is the one that names the mechanism, and it is why the
    docstring on ``_widened`` was rewritten in this batch: ``jnp.array(1.0)``
    is **weakly** typed, and a weak float64 adopts a strong float32's dtype
    instead of promoting it. ``astype(float64)`` therefore does two things and
    only one of them is a widening — the other is stripping the weak type, so
    that the init wins the promotion against the model's float32 constants and
    the prediction reaches ``refuse_single_precision`` in double.
    """

    @pytest.fixture
    def affine32(self):
        """The affine fixture of :class:`TestTheClosedFormAgreesWithTheRefit`,
        declared the way a config-driven run declares it.

        Small and poor on purpose for the same reason it is there: eight
        channels and unit noise, so the prior is visible against the
        likelihood. Nothing here is float64.
        """
        from rheplicant.radio import ForegroundOperator, GainOperator, assemble

        with _ambient_float32():
            n = 8
            freq = jnp.linspace(60e6, 85e6, n)
            state = State(
                coords=Coordinates(time=jnp.arange(float(n)), freq=freq),
                key=jax.random.key(0),
                meta={"telescope": "affine-single"},
            )
            twin = assemble(
                ForegroundOperator(
                    amplitude=jnp.array(1.0),
                    spectral_index=jnp.array(2.5),
                    ref_freq=70e6,
                ),
                GainOperator(gain=jnp.array(1.1)),
            )
            observed = twin(state).data
            space = ParameterSpace(
                latents=[
                    Latent("gain", init=jnp.array(1.0), prior=dist.Normal(1.4, 0.2))
                ],
                bindings=[Bind("gain", into=lambda p: p["gain"].gain)],
            )
        return {"space": space, "twin": twin, "state": state, "observed": observed}

    def test_the_fixture_really_is_declared_in_single_precision(self, affine32):
        """Without this the class below is a second copy of the affine test.

        Everything the facade does not cast is checked: the init, the data, and
        one of the model's own coordinate arrays. If a later edit gives this
        fixture x64 — the way the rest of the module has it — this assertion is
        what says so, rather than three tests quietly passing for free.
        """
        assert affine32["space"].initial_values()["gain"].dtype == jnp.float32
        assert affine32["observed"].dtype == jnp.float32
        assert affine32["state"].coords.freq.dtype == jnp.float32

    def test_the_verdict_comes_back_in_double(self, affine32):
        """The whole point of forcing x64: a float32 model, a float64 answer.

        Without ``_widened`` this raises instead — bayesmith's
        ``refuse_single_precision`` on the prediction, which is the refusal
        doing its job and not a surprise. The report is checked for the
        property that made the affine model worth using, not merely for a
        dtype: on an exactly quadratic posterior the independent refit is
        Newton-exact, so the closed form has nothing to hide behind.
        """
        with _ambient_float32():
            report = prior_sensitivity(
                affine32["space"], affine32["twin"], affine32["state"],
                affine32["observed"], 1.0,
            )
        assert report.mode.dtype == jnp.float64
        assert report.precision.dtype == jnp.float64
        assert report.sigma_post.dtype == jnp.float64
        row = report.for_latent("gain")
        closed, refit = float(row["shift_sigma"]), float(row["shift_sigma_refit"])
        assert closed == pytest.approx(refit, rel=VERIFY_RTOL, abs=VERIFY_ATOL), (
            f"closed form {closed:+.9f} against an exact refit {refit:+.9f} on a "
            "model declared in single precision. A float32 Hessian is noise at "
            "the 1e-7 level, which is where these two would first disagree."
        )
        assert bool(np.all(report.verified))

    def test_a_weak_float64_init_does_not_carry_a_float32_model(self, affine32):
        """The half of ``_widened`` that is not a widening.

        The init here is **already float64** — declared under x64, like every
        other latent in this module — and the model around it is float32. A
        ``_widened`` that only cast where the dtype was float32 would leave
        this weakly typed, the weak float64 would adopt the model's float32,
        and the prediction would arrive in single precision anyway. Measured:
        ``forward`` returns float32 for the raw init and float64 for the
        widened one, from the same values.
        """
        space = ParameterSpace(
            latents=[Latent("gain", init=jnp.array(1.0), prior=dist.Normal(1.4, 0.2))],
            bindings=[Bind("gain", into=lambda p: p["gain"].gain)],
        )
        init = space.initial_values()["gain"]
        assert init.dtype == jnp.float64 and init.weak_type, (
            "this test is about a WEAKLY typed float64; if `initial_values` "
            "starts returning a strong one, the condition is gone and so is "
            "the test's subject."
        )
        with _ambient_float32():
            report = prior_sensitivity(
                space, affine32["twin"], affine32["state"], affine32["observed"], 1.0
            )
        assert report.mode.dtype == jnp.float64
        assert bool(np.all(report.verified))
