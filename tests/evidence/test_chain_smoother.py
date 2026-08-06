"""The smoothed drift, against the dense oracle's own conditional.

A smoother is where a transition applied in the wrong direction stops being
invisible: `phi zeta_e` and `phi^T zeta_e` give the same filter answer for a
scalar chain and different smoothed means for a wide one, so this file runs both
widths, and the wide one at three probes -- Task 3 measured the transposed
filter's error at 0.0078 nats at one of its probes and 0.97 at another, small
enough that a single-point test called it exact.

**The covariance does not depend on theta**, in a linear-Gaussian model or in
this one: the posterior spread of the drift is a property of the designs, the
noise and the transition alone. So it is pinned once, in full -- cross-epoch
blocks included -- while the mean is pinned at every probe. That split is the
whole test strategy here, and it is why a mean-only file would be weak: a filter
mistaken for a smoother returns the *right* means at the last epoch and at
every epoch after conditioning, and gets the interior variances wrong.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference import chain
from rheplicant.inference.chain import LinearGaussianTransition, smooth
from tests.evidence import chain_bank as bank


def _transition():
    return LinearGaussianTransition(
        phi=bank.PHI,
        process_std=bank.PROCESS_STD,
        initial_std=bank.INITIAL_STD,
        initial_mean=bank.INITIAL_MEAN,
    )


def _values(probe):
    return {"t_rx": jnp.asarray(probe[0]), "gain_slope": jnp.asarray(probe[1])}


def _smoothed(probe, blocks=None, transition=None):
    return smooth(
        bank.stacked() if blocks is None else blocks,
        _transition() if transition is None else transition,
        _values(probe),
        names=bank.THETA_NAMES,
        shapes=((), ()),
    )


@pytest.mark.parametrize("probe", bank.PROBES)
def test_the_smoothed_mean_is_the_oracles_conditional_mean(probe):
    """Measured: 5.2e-14 at worst over the four probes, against a band of 1e-8."""
    mean, _ = _smoothed(probe)
    expected, _ = bank.oracle_zeta_posterior(probe)
    np.testing.assert_allclose(
        np.asarray(mean).ravel(), expected, rtol=1e-8, atol=1e-10
    )


def test_the_smoothed_variance_is_the_oracles_diagonal():
    _, cov = _smoothed(bank.PROBES[1])
    _, expected = bank.oracle_zeta_posterior(bank.PROBES[1])
    np.testing.assert_allclose(
        np.asarray(cov).reshape(bank.N_EPOCHS), np.diag(expected), rtol=1e-8
    )


def test_the_whole_covariance_matches_including_the_cross_epoch_blocks():
    """The diagonal alone is a weaker claim than it looks.

    A joint solve that coupled the epochs in the wrong direction -- or coupled
    them twice -- can still reproduce six variances while getting
    ``cov(zeta_2, zeta_5)`` wrong, and every per-epoch diagnostic would go on
    reading correctly. So the full 6x6 is compared, off-diagonals included.
    Measured: maximum absolute difference from the oracle 2.2e-14, against a
    diagonal running 1.04e-2 to 2.22e-2 and a largest off-diagonal of 5.07e-4.
    Note what `atol` is doing: the far corner entries are 8.4e-11, so `rtol`
    alone would demand 8.4e-19 of them and fail on arithmetic noise.
    """
    covariance = np.asarray(
        chain._joint_covariance(
            bank.stacked(),
            _transition(),
            _values(bank.PROBES[1]),
            names=bank.THETA_NAMES,
            shapes=((), ()),
        )
    )
    _, expected = bank.oracle_zeta_posterior(bank.PROBES[1])
    np.testing.assert_allclose(covariance, expected, rtol=1e-8, atol=1e-12)
    # And it is not diagonal, so the previous test really was the weaker one:
    # measured 5.07e-4 off the diagonal against 1.04e-2 on it, or 4.9 %.
    assert np.max(np.abs(covariance - np.diag(np.diag(covariance)))) > 1e-4


def test_a_smoothed_epoch_is_sharper_than_the_same_epoch_filtered():
    """The name is the claim, and this is the failure it is named against.

    A filter mistaken for a smoother returns means that look right -- they are
    right, at the last epoch -- and variances that are too large everywhere
    before it, because the filtered marginal has seen only `d_1:e`. Running the
    same joint solve over the first `e` epochs' blocks IS the filtered marginal
    at epoch `e`, so the two are compared directly rather than through a claim.

    **The margin is small on this fixture, and that is measured, not assumed.**
    Ratios smoothed/filtered at epochs 1..6: 0.9872, 0.9745, 0.9922, 0.9765,
    0.9825, 1.0000. Four samples a night at `SIGMA = 0.3` already pin each
    epoch's drift to a variance of about 0.01 against a prior of 1.0, so there
    is little left for the neighbours to add. The comparison is therefore a
    strict `<` with the last epoch pinned to equality -- the equality is what
    makes the inequality mean something, because a smoother that had quietly
    returned the filtered marginal would satisfy neither.
    """
    factors, targets, offsets = bank.stacked()
    _, smoothed = _smoothed(bank.PROBES[1])
    smoothed = np.asarray(smoothed).ravel()
    for epochs in range(1, bank.N_EPOCHS + 1):
        _, filtered = _smoothed(
            bank.PROBES[1],
            blocks=(factors[:epochs], targets[:epochs], offsets[:epochs]),
        )
        last = float(np.asarray(filtered).ravel()[-1])
        if epochs == bank.N_EPOCHS:
            assert last == pytest.approx(smoothed[-1], rel=1e-10)
        else:
            assert smoothed[epochs - 1] < last, (
                f"epoch {epochs}: smoothed variance {smoothed[epochs - 1]} is not "
                f"below the filtered {last} -- the backward information is not "
                "reaching the interior epochs"
            )
            assert smoothed[epochs - 1] / last < 0.999


def test_a_late_epochs_data_moves_an_early_epochs_smoothed_mean():
    """"Neighbouring epochs pull on each other", as a number that cannot be 0.

    Adding 1.0 to the LAST epoch's target moves every earlier epoch's smoothed
    mean, by -5.41e-3, -1.63e-4, -1.66e-6 and -5.0e-8 going backwards -- the
    chain's memory, decaying by roughly `phi^2 var_filtered / var_prior` a step.
    Under `phi = 0` the same perturbation moves the last epoch by the same
    -0.2364 and every other epoch by **exactly zero**, to 1e-12, which is the
    control: this is a statement about the coupling and not about the solve.
    """
    factors, targets, offsets = bank.stacked()
    bumped = targets.at[-1].add(1.0)
    base, _ = _smoothed(bank.PROBES[1])
    moved, _ = _smoothed(bank.PROBES[1], blocks=(factors, bumped, offsets))
    shift = np.asarray(moved).ravel() - np.asarray(base).ravel()
    assert abs(shift[-1]) > 0.1
    assert abs(shift[-2]) > 1e-4
    assert abs(shift[-3]) > 1e-6

    loose = LinearGaussianTransition(
        phi=0.0, process_std=bank.PROCESS_STD, initial_std=bank.INITIAL_STD
    )
    base, _ = _smoothed(bank.PROBES[1], transition=loose)
    moved, _ = _smoothed(
        bank.PROBES[1], blocks=(factors, bumped, offsets), transition=loose
    )
    unlinked = np.asarray(moved).ravel() - np.asarray(base).ravel()
    assert abs(unlinked[-1]) > 0.1
    np.testing.assert_allclose(unlinked[:-1], 0.0, atol=1e-12)


def test_the_smoothed_variance_is_below_the_prior_variance_everywhere():
    """Data can only sharpen a Gaussian. A smoother that widened one would be wrong."""
    _, cov = _smoothed(bank.PROBES[1])
    prior = np.diag(bank.zeta_covariance())
    got = np.asarray(cov).reshape(bank.N_EPOCHS)
    assert np.all(got < prior)
    assert np.all(got > 0.0)


def test_the_smoothed_mean_is_smoother_than_a_per_epoch_fit():
    """The plan predicted a factor of about 2 here. It is 1.05, and that is real.

    Measured on this fixture: mean successive difference 0.2262 for the linked
    smoother against 0.2375 for independent per-epoch estimates, a ratio of
    0.9528. The prediction was wrong for the reason
    `test_a_smoothed_epoch_is_sharper_than_the_same_epoch_filtered` records --
    four samples a night at `SIGMA = 0.3` pin each epoch's drift to about 0.01
    against a prior of 1.0, so the sequence is nearly the per-epoch fit whatever
    the transition says. The assertion is kept because its direction is a real
    claim, and the number is written down because a reader who expected 2 should
    be able to see that the fixture, not the smoother, is what decides it. The
    load-bearing evidence for "neighbours pull" is
    `test_a_late_epochs_data_moves_an_early_epochs_smoothed_mean`, where the
    control is exactly zero.
    """
    mean, _ = _smoothed(bank.PROBES[1])
    linked = np.abs(np.diff(np.asarray(mean).ravel())).mean()
    loose = LinearGaussianTransition(
        phi=0.0, process_std=bank.INITIAL_STD, initial_std=bank.INITIAL_STD
    )
    independent, _ = _smoothed(bank.PROBES[1], transition=loose)
    assert linked < np.abs(np.diff(np.asarray(independent).ravel())).mean()


def _wide_transition():
    """Asymmetric `phi`, and two different process spreads. Both matter."""
    return LinearGaussianTransition(
        phi=jnp.asarray([[0.9, 0.3], [0.0, 0.5]]),
        process_std=jnp.asarray([0.4, 0.7]),
        initial_std=jnp.asarray([1.0, 1.0]),
    )


@pytest.mark.parametrize("probe", bank.PROBES[:3])
def test_a_wide_chain_applies_phi_in_the_direction_it_claims(probe):
    """A scalar chain cannot tell `diag(1/q) @ phi` from `phi @ diag(1/q)`.

    Neither can a diagonal one, and neither can a wide one whose `process_std`
    is the same in every component -- the two forms differ only where an
    off-diagonal of `phi` meets two different spreads. So this builds a
    deliberately asymmetric `phi` over a width-2 chain with `process_std =
    (0.4, 0.7)` and checks against a dense conditional that uses `phi` in the
    forward direction only.

    Verified discriminating by sabotage: writing the coupling rows as
    `-resolved.phi * inverse_process[None, :]` leaves **every scalar test in
    this file green** and breaks all three of these, the smoothed means moving
    by 0.0299, 0.0252 and 0.1653 at the three probes. Three probes rather than
    one for the reason Task 3 measured on the filter: the wrong ordering is a
    rotation, so what it costs depends on where in theta you stand -- here by a
    factor of 6.6 between the smallest and the largest.

    Measured agreement, unsabotaged: 3.3e-14 at worst over the three probes.
    """
    transition = _wide_transition()
    mean, _ = smooth(
        bank.wide_stacked(2),
        transition,
        _values(probe),
        names=bank.THETA_NAMES,
        shapes=((), ()),
    )
    expected, _ = bank.wide_oracle_zeta_posterior(probe, transition)
    np.testing.assert_allclose(np.asarray(mean), expected, rtol=1e-7, atol=1e-9)


def test_a_wide_chains_whole_covariance_matches_too():
    """Including the cross-component blocks, which the scalar fixture has none of.

    Measured: 7.9e-15 at worst against entries up to 0.0416.
    """
    transition = _wide_transition()
    covariance = np.asarray(
        chain._joint_covariance(
            bank.wide_stacked(2),
            transition,
            _values(bank.PROBES[1]),
            names=bank.THETA_NAMES,
            shapes=((), ()),
        )
    )
    _, expected = bank.wide_oracle_zeta_posterior(bank.PROBES[1], transition)
    np.testing.assert_allclose(covariance, expected, rtol=1e-7, atol=1e-10)


def test_the_blocks_must_be_as_wide_as_the_latents_they_claim_to_be_over():
    """The filter's refusal, at the smoother's call site.

    One implementation, two callers: a smoother that sliced `factors[e][:,
    n_theta:]` against a block of the wrong width would take theta's columns
    for the chain's and return a finite, plausible, wrong drift.
    """
    factors, targets, offsets = bank.stacked()
    with pytest.raises(
        chain.StateValidationError, match="a different one is not a rename"
    ):
        smooth(
            (factors[:, :, :2], targets, offsets),
            _transition(),
            _values(bank.PROBES[0]),
            names=bank.THETA_NAMES,
            shapes=((), ()),
        )
