"""Validation item 12.11. The diagnostic that cannot fire, and the two that can.

The shrinkage check is kept -- section 9 says to keep it, "labelled as one that
cannot detect coherent bias" -- and this file is that label, in executable form.

Two of the plan's own claims about this fixture did not survive measurement and
are corrected here rather than asserted:

* the fitted power is ``-0.49991034`` for the campaign as a whole. The plan's
  ``-0.49989592`` is the *first coordinate's own* slope; the second is
  ``-0.49992476``, and a single float over two parameters is the pooled fit.
* the per-epoch chi-square scatter is **not** preserved by a common mode.
  Measured ``5.5467`` against ``sqrt(2 * 6) = 3.4641``, which is 60 % high and
  well outside the plan's ``rel=0.35``. What is preserved is the *template
  projection* scatter, and what explains the chi-square's is noncentrality --
  see :func:`test_the_per_epoch_chi_square_scatter_is_inflated_not_preserved`.
"""

import numpy as np
import pytest

from rheplicant.inference.diagnostics import (
    coherent_mode,
    held_out_z,
    shrinkage_power,
    shrinkage_report,
)
from tests.evidence import campaign_bank as camp

SIZES = (10, 40, 160, 640)


def _sigmas(biased):
    return {n: np.sqrt(np.diag(camp.posterior(camp.terms(n, biased))[1])) for n in SIZES}


def test_shrinkage_is_identical_in_the_clean_and_the_biased_campaign():
    """The self-refutation, measured. Not `approximately -0.5`: the SAME NUMBER.

    ``sigma_N = (sum_e F_e + F_prior)^-1/2`` does not read the data at all, and a
    deterministic common-mode error changes only ``b``. So the two campaigns
    cannot differ here, and a test that injected a systematic and watched sigma
    would be asserting that arithmetic is arithmetic. Measured: the per-N sigma
    arrays are equal element for element, and the fitted power is
    ``-0.49991034`` on both.
    """
    clean_sigmas, biased_sigmas = _sigmas(biased=False), _sigmas(biased=True)
    for n in SIZES:
        assert np.array_equal(clean_sigmas[n], biased_sigmas[n])
    clean = shrinkage_power(clean_sigmas)
    biased = shrinkage_power(biased_sigmas)
    assert clean == pytest.approx(biased, rel=1e-12)
    assert clean == pytest.approx(-0.49991034, abs=1e-7)


def test_shrinkage_power_reports_the_power_it_is_actually_given():
    """Otherwise a function that returned the constant -0.5 would pass this file.

    The measured campaign value sits within 1e-4 of -0.5, so every other
    assertion about it is also satisfied by a stub. This one is not.
    """
    for power in (-1.0, -0.25, 0.0, 0.5):
        synthetic = {n: 3.7 * np.array([n**power, 2.0 * n**power]) for n in SIZES}
        assert shrinkage_power(synthetic) == pytest.approx(power, abs=1e-12)


def test_a_uniform_rescaling_of_the_fisher_still_fits_exactly_minus_a_half():
    """Section 9's own verification: `F_e -> (1+c) F_e` for c in {0, +0.5, -0.3}.

    A uniform rescaling is a constant offset in ``log sigma``, and a fitted slope
    with a free intercept cannot see one. Measured: the same value to every digit
    float64 has, not merely to the tolerance below.
    """
    base = _sigmas(biased=False)
    for scale in (0.0, 0.5, -0.3):
        rescaled = {n: sigma / np.sqrt(1.0 + scale) for n, sigma in base.items()}
        assert shrinkage_power(rescaled) == pytest.approx(shrinkage_power(base), rel=1e-12)


def test_the_answer_is_wrong_and_the_error_grows_as_root_n():
    """Section 7's arithmetic: the bias is N-independent, the width is not.

    Measured ratios: 7.428 at N = 10, 14.724 at 40, 26.204 at 160, 52.568 at 640.
    """
    ratios = {}
    for n in SIZES:
        mean, covariance = camp.posterior(camp.terms(n, biased=True))
        ratios[n] = float(np.max(np.abs(mean - camp.TRUTH) / np.sqrt(np.diag(covariance))))
    assert ratios[640] > 20.0  # measured 52.568
    # sqrt(640/40) = 4; measured 3.570, because the prior still carries weight at
    # N = 40 and the bias is not exactly along the tightest direction.
    assert ratios[640] / ratios[40] == pytest.approx(4.0, rel=0.5)


def test_the_mean_level_diagnostics_fire_on_the_same_campaign():
    """12.11's actual demand, and the reason section 9.3 exists.

    Measured at N = 640: ``chi2_z`` +31.9231 biased against +0.4457 clean, and
    the named template +52.5498 against +1.7163.
    """
    biased = coherent_mode(camp.terms(640, biased=True))
    clean = coherent_mode(camp.terms(640, biased=False))
    assert biased["chi2_z"] > 10.0 and abs(clean["chi2_z"]) < 4.0
    assert biased["templates"]["gain_ripple"]["z"] > 20.0
    assert abs(clean["templates"]["gain_ripple"]["z"]) < 4.0


def test_the_held_out_score_is_the_one_that_does_not_fire_here():
    """Deviation 9. Named beside the two that do, so nobody reads three as one."""
    clean = np.array([r.z for r in held_out_z(camp.terms(640, False), camp.prior_fisher())])
    biased = np.array([r.z for r in held_out_z(camp.terms(640, True), camp.prior_fisher())])
    np.testing.assert_allclose(biased, clean, atol=5e-4)  # measured max 4.93e-05


def test_split_half_and_leave_one_out_are_blind_to_the_common_mode():
    """Two of section 9's list of what a common mode passes, checked not quoted.

    Both pass *identically* in the clean and the biased campaign, which is a much
    stronger statement than "within tolerance": the fault shifts both halves of a
    split by the same amount, and both sides of a leave-one-out residual by the
    same amount, so it cancels rather than being small.

    Measured: the difference between the two half-campaign posterior means is the
    same in the clean and the biased run to 7.8e-16 -- roundoff -- with a largest
    per-parameter discrepancy of 0.881 spreads. The leave-one-out mean is
    -0.049296 in both, against a 4-sigma bound of 0.158114.
    """
    biased = camp.terms(640, biased=True)
    clean = camp.terms(640, biased=False)

    def split(collected):
        return camp.posterior(collected[:320])[0] - camp.posterior(collected[320:])[0]

    spread = np.sqrt(np.diag(camp.posterior(biased[:320])[1]))
    np.testing.assert_allclose(split(biased), split(clean), atol=1e-14)
    assert np.max(np.abs(split(biased)) / spread) < 4.0

    scores = np.array([r.z for r in held_out_z(biased, camp.prior_fisher())])
    assert abs(scores.mean()) < 4.0 / np.sqrt(len(scores))


def test_the_per_epoch_chi_square_scatter_is_inflated_not_preserved():
    """The plan asserted this one passes cleanly. Measured, it does not.

    ``chi2.std() == approx(sqrt(2 * dof), rel=0.35)`` is what the plan wrote, and
    it fails: 5.5467 against sqrt(12) = 3.4641, 60 % high. The reason is not extra
    randomness -- the fault is deterministic and injects none -- but noncentrality.
    A shifted Gaussian's squared norm is noncentral chi-square with
    ``lambda = ||c_perp||^2 = 4.0376``, whose standard deviation is
    ``sqrt(2 (k + 2 lambda)) = 5.3057``; measured 5.5467, a ratio of 1.045, the
    same 4 % excess the clean campaign shows against ``sqrt(2 k)`` at this sample
    size.

    What *is* preserved, and is the honest form of "no variance was injected", is
    the **template projection** scatter: 1.00200 on the clean campaign and
    1.00200 on the biased one, because there the fault is a pure additive shift
    of 2.00937 -- exactly ``||c_perp||`` -- with an epoch-to-epoch spread of
    5.6e-16.
    """
    biased = camp.terms(640, biased=True)
    clean = camp.terms(640, biased=False)
    dof = biased[0].residual_dof
    chi2 = np.array([float(term.residual_chi2) for term in biased])
    _, perpendicular = camp.whitened_split()
    noncentrality = perpendicular**2

    assert chi2.std() > 1.4 * np.sqrt(2 * dof)
    assert chi2.std() == pytest.approx(np.sqrt(2 * (dof + 2 * noncentrality)), rel=0.1)

    projections = {
        label: np.array([float(term.template_projections[0]) for term in collected])
        for label, collected in (("biased", biased), ("clean", clean))
    }
    shift = projections["biased"] - projections["clean"]
    assert shift.std() < 1e-12
    assert shift.mean() == pytest.approx(perpendicular, rel=1e-9)
    assert projections["biased"].std() == pytest.approx(projections["clean"].std(), rel=1e-9)


def test_shrinkage_power_says_in_its_own_report_that_it_cannot_detect_this():
    """A number that cannot fire must not be returned looking like one that can."""
    report = shrinkage_report(_sigmas(biased=True))
    assert report["power"] == pytest.approx(-0.5, abs=1e-3)
    assert report["detects_coherent_bias"] is False
    assert "no variance" in report["caveat"]
    assert report["n_values"] == SIZES


def test_a_single_campaign_size_cannot_have_a_power_and_is_refused():
    """One point fits every slope. Returning -0.5 there would be an invention."""
    with pytest.raises(ValueError, match="at least two"):
        shrinkage_power({640: np.array([0.007, 0.005])})
    # The nearest legitimate case: two sizes, which is exactly determined.
    assert shrinkage_power({160: np.array([2.0]), 640: np.array([1.0])}) == pytest.approx(
        -0.5, rel=1e-12
    )


def test_a_non_positive_campaign_size_is_refused():
    """The fit is in log N. `log(0)` is `-inf` and `log(-40)` is `nan`.

    Either one leaves a slope that is nan, inf, or -- worse -- finite and
    plausible, depending on where the bad size falls in the grid.
    """
    base = _sigmas(biased=False)
    for size in (0, -40):
        broken = {size: base[10], 160: base[160], 640: base[640]}
        with pytest.raises(ValueError, match="not positive"):
            shrinkage_power(broken)
    # The nearest legitimate case: the smallest campaign there is.
    assert shrinkage_power(
        {1: np.array([2.0]), 4: np.array([1.0])}
    ) == pytest.approx(-0.5, rel=1e-12)


def test_a_non_finite_or_non_positive_sigma_is_refused_rather_than_logged():
    """`np.log(nan)` is `nan`, and a NaN power loses every comparison about it."""
    base = _sigmas(biased=False)
    for poison in (np.nan, np.inf, 0.0, -1.0):
        broken = {n: sigma.copy() for n, sigma in base.items()}
        broken[160][0] = poison
        with pytest.raises(ValueError, match="finite and strictly positive"):
            shrinkage_power(broken)
    # ...and a legitimately tiny sigma is not refused.
    tiny = {n: sigma * 1e-30 for n, sigma in base.items()}
    assert shrinkage_power(tiny) == pytest.approx(shrinkage_power(base), rel=1e-9)


def test_a_ragged_set_of_campaign_sizes_is_refused():
    """A pooled fit over 2 parameters at one N and 3 at another is not one fit."""
    base = _sigmas(biased=False)
    ragged = {n: sigma for n, sigma in base.items()}
    ragged[160] = np.append(ragged[160], 0.01)
    with pytest.raises(ValueError, match="same latents"):
        shrinkage_power(ragged)


class TestTheCaveatIsThisPackagesOwn:
    """`shrinkage_report` delegates its numbers and keeps its own caveat.

    The far side's ends *"Use template_modes(), coherent_mode() and the
    systematic floor"*. ``template_modes`` does not exist in rheplicant, so
    forwarding that text would hand a reader a function their package does not
    have -- the same trap the `calibrate` switch hit with a remedy sentence,
    recorded there.

    This caveat also carries a measurement the far side's does not: the twelve
    digits on a campaign biased by 52.6 sigma, which `test_provenance.py`
    computes independently. That is evidence, not phrasing, and it would be
    lost by forwarding.

    Both directions are asserted, because the substitution is a decision that
    can silently stop being made -- somebody delegating one more field, or
    upstream rewording, and the caveat quietly becomes the far side's again
    while every numeric assertion in this file stays green.
    """

    def _report(self):
        base = _sigmas(biased=False)
        return shrinkage_report(base)

    def test_it_names_a_route_this_package_has(self):
        caveat = self._report()["caveat"]
        assert "coherent_mode()" in caveat, caveat

    def test_it_does_not_name_one_this_package_lacks(self):
        """The half with teeth. `template_modes` is bayesmith's."""
        import rheplicant.inference as inference

        caveat = self._report()["caveat"]
        assert not hasattr(inference, "template_modes"), (
            "rheplicant now HAS template_modes, so this test is stale -- the "
            "caveat may name it and the substitution may no longer be needed"
        )
        assert "template_modes" not in caveat, (
            "the far side's caveat reached this package's output; the "
            "substitution in shrinkage_report stopped being applied"
        )

    def test_it_keeps_the_measurement_the_far_side_drops(self):
        assert "52.6 sigma" in self._report()["caveat"]

    def test_the_numbers_are_still_the_far_sides(self):
        """ANTI-VACUITY: prove the delegation is real, not a local rewrite.

        If `shrinkage_report` had quietly gone back to computing everything
        here, all three cases above would still pass.
        """
        from bayesmith.marginal.diagnostics import shrinkage_report as far

        base = _sigmas(biased=False)
        ours, theirs = shrinkage_report(base), far(base)
        assert ours["power"] == theirs["power"]
        assert ours["n_values"] == theirs["n_values"]
        assert ours["detects_coherent_bias"] == theirs["detects_coherent_bias"]
        assert ours["caveat"] != theirs["caveat"], "the caveat must NOT match"
