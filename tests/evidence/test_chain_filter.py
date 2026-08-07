"""The section 6 recursion, against a dense oracle, absolutely.

Every number in this file was measured on this fixture, on the run that shipped
it, and written beside its assertion so a later reader can re-measure rather than
believe. The tolerance bands are wide enough to survive a different RNG draw of
the fixture and far too tight to survive a dropped term: every constant's gap
below is between 0.9 and 46 nats in magnitude, while the recursion's own
disagreement with the oracle is 9.1e-13 at worst -- so the smallest gap pinned is
1e12 times the noise it has to be told apart from.

Why absolute log-densities and nothing else. Section 6 is nothing but constants,
and the recursion's shape, its gradient and its curvature are all correct with
any of them dropped. A test on moments, or on a difference of densities, or on a
posterior's width, passes against a filter that is wrong by hundreds of nats over
a real campaign. So each constant gets its own test, deleted by `monkeypatch` on
a named module-level helper -- which also pins that the filter *calls* that
helper rather than carrying a second inlined copy of the same arithmetic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference import chain
from rheplicant.inference.chain import (
    LinearGaussianTransition,
    chain_log_likelihood,
    chain_marginal,
    ornstein_uhlenbeck,
)
from rheplicant.inference.compress import compress_linear
from rheplicant.inference.sqrtinfo import SqrtInfo
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


def _filtered(probe, blocks=None, transition=None):
    return float(
        chain_log_likelihood(
            bank.stacked() if blocks is None else blocks,
            _transition() if transition is None else transition,
            _values(probe),
            names=bank.THETA_NAMES,
            shapes=((), ()),
        )
    )


@pytest.mark.parametrize("probe", bank.PROBES)
def test_the_filter_is_the_marginal_likelihood_to_roundoff(probe):
    """The exactness pin.

    Measured gaps against the dense oracle, one per probe, at densities from -73
    to -2506 nats: 1.1e-13, 5.7e-14, 9.1e-13, 9.1e-13. The smallest constant
    pinned below is 0.9189 nats, so every band is at least 1e12 times the
    recursion's own error -- which is what makes a dropped constant
    distinguishable from float noise rather than a judgement call.
    """
    assert _filtered(probe) == pytest.approx(bank.oracle(probe), abs=1e-9)


def test_the_filter_is_exact_under_jit():
    probe = bank.PROBES[1]
    blocks, transition = bank.stacked(), _transition()

    def density(values):
        return chain_log_likelihood(
            blocks, transition, values, names=bank.THETA_NAMES, shapes=((), ())
        )

    assert float(jax.jit(density)(_values(probe))) == pytest.approx(
        bank.oracle(probe), abs=1e-9
    )


def test_the_filter_is_differentiable_in_the_transitions_own_parameters():
    """Section 6's `linked_hyper` in miniature: the whole point of the scan.

    `marginalise` cannot be here at all -- it raises ConcretizationTypeError
    under grad as well as under jit -- which is why `marginalise_arrays` exists.
    """
    blocks = bank.stacked()
    probe = bank.PROBES[1]

    def density(tau):
        return chain_log_likelihood(
            blocks,
            ornstein_uhlenbeck(tau=tau, sigma=1.0),
            _values(probe),
            names=bank.THETA_NAMES,
            shapes=((), ()),
        )

    analytic = float(jax.grad(density)(3.0))
    step = 1e-6
    numeric = (float(density(3.0 + step)) - float(density(3.0 - step))) / (2 * step)
    # Measured: analytic +0.485405251039, central difference +0.485405251993,
    # agreeing to 2.0e-9 relative -- the step's own truncation error, not the
    # gradient's.
    assert analytic == pytest.approx(numeric, rel=1e-6)
    # And it is not two ways of computing zero: the chain does respond to tau.
    assert abs(analytic) > 1e-3


def test_an_inferred_correlation_time_is_exact_at_several_values():
    """Not one tau: a filter that pinned the transition would agree at one point."""
    blocks = bank.stacked()
    probe = bank.PROBES[1]
    for tau in (0.5, 2.0, 8.0):
        transition = ornstein_uhlenbeck(tau=tau, sigma=1.0)
        phi = float(transition.phi[0, 0])
        process_var = float(transition.process_std[0]) ** 2
        expected = bank.oracle(probe, phi=phi, process_var=process_var, initial_var=1.0)
        assert _filtered(probe, blocks, transition) == pytest.approx(expected, abs=1e-8)


def test_the_marginal_is_a_sqrtinfo_over_the_global_latents_only():
    """theta is never marginalised; zeta always is.

    **The third assertion used to be that the marginal's Fisher is positive
    semi-definite.** ``fisher()`` is ``R.T @ R``, so ``eigvalsh(...) > -1e-12``
    holds for every array of real numbers this function could possibly return,
    including one from a filter that had dropped the chain entirely. It could
    not fail, so it said nothing.

    What it should have said is that the *value* is right, and the dense oracle
    already knows it: for the whole campaign as one Gaussian,
    ``F = A^T (sigma^2 I + C Sigma_zeta C^T)^-1 A``. Measured, that agrees with
    the filter's marginal to ``2.6e-13`` absolute, ``1.4e-15`` relative -- and a
    bag's ``A^T A / sigma^2``, which is what ignoring the chain gives, is 26 %
    larger in trace with an off-diagonal of ``+0.0498`` against the correct
    ``-10.383``. The sign is different, so this cannot pass on a filter that
    forgets what it is filtering.
    """
    marginal = chain_marginal(
        bank.stacked(),
        _transition(),
        _values(bank.PROBES[0]),
        names=bank.THETA_NAMES,
        shapes=((), ()),
    )
    assert marginal.names == bank.THETA_NAMES
    assert marginal.factor.shape[1] == bank.N_THETA

    designs, drifts, _ = bank.design(0)
    rows = bank.N_EPOCHS * bank.N_SAMPLES
    theta_design = np.zeros((rows, bank.N_THETA))
    zeta_design = np.zeros((rows, bank.N_EPOCHS))
    for epoch in range(bank.N_EPOCHS):
        span = slice(epoch * bank.N_SAMPLES, (epoch + 1) * bank.N_SAMPLES)
        theta_design[span, :] = designs[epoch]
        zeta_design[span, epoch] = drifts[epoch][:, 0]
    covariance = bank.SIGMA**2 * np.eye(rows) + (
        zeta_design @ bank.zeta_covariance() @ zeta_design.T
    )
    oracle = theta_design.T @ np.linalg.solve(covariance, theta_design)
    np.testing.assert_allclose(np.asarray(marginal.fisher()), oracle, atol=1e-11)
    # ...and the bag's answer, which the old assertion also accepted.
    bag = theta_design.T @ theta_design / bank.SIGMA**2
    assert np.sign(bag[0, 1]) != np.sign(oracle[0, 1])


def _wide_campaign(seed=7):
    """A width-2 chain over two epochs, and its own dense oracle.

    Returns ``(blocks, transition, oracle_of_theta)``.

    `chain_bank`'s chain is scalar, and for a scalar chain `diag(1/q) @ phi` and
    `phi @ diag(1/q)` are the same number -- so every other test in this file
    passes against a filter that scales the transition rows on the wrong side.
    Here `phi` is asymmetric with nonzero off-diagonals and `process_std`'s two
    components differ by a factor of 2.4, so the two orderings are different
    matrices and the answer separates.
    """
    n_theta, n_zeta, n_epochs, n_samples, sigma = 2, 2, 2, 3, 0.4
    phi = np.array([[0.8, 0.3], [-0.2, 0.5]])
    process_std = np.array([0.5, 1.2])
    initial_std = np.array([1.0, 0.7])
    initial_mean = np.array([0.2, -0.1])

    keys = jax.random.split(jax.random.key(seed), 3 * n_epochs)
    designs = [
        np.asarray(jax.random.normal(keys[e], (n_samples, n_theta)))
        for e in range(n_epochs)
    ]
    responses = [
        np.asarray(jax.random.normal(keys[n_epochs + e], (n_samples, n_zeta))) + 1.0
        for e in range(n_epochs)
    ]
    data = [
        np.asarray(jax.random.normal(keys[2 * n_epochs + e], (n_samples,)))
        for e in range(n_epochs)
    ]

    infos = []
    for epoch in range(n_epochs):
        term = compress_linear(
            design={
                "a": designs[epoch][:, :1],
                "b": designs[epoch][:, 1:],
                "z": responses[epoch],
            },
            observed=jnp.asarray(data[epoch]),
            noise_std=sigma,
            shapes={"a": (), "b": (), "z": (n_zeta,)},
            epoch_id=f"wide{epoch}",
        )
        infos.append(
            SqrtInfo.combine(
                SqrtInfo.null(term.info.names, term.info.shapes), term.info
            )
        )
    blocks = (
        jnp.stack([info.factor for info in infos]),
        jnp.stack([info.target for info in infos]),
        jnp.stack([jnp.asarray(info.offset) for info in infos]),
    )
    transition = LinearGaussianTransition(
        phi=phi,
        process_std=process_std,
        initial_std=initial_std,
        initial_mean=initial_mean,
    )

    prior_initial = np.diag(initial_std**2)
    process = np.diag(process_std**2)
    covariance_zeta = np.zeros((n_epochs * n_zeta, n_epochs * n_zeta))
    covariance_zeta[:n_zeta, :n_zeta] = prior_initial
    covariance_zeta[n_zeta:, :n_zeta] = phi @ prior_initial
    covariance_zeta[:n_zeta, n_zeta:] = prior_initial @ phi.T
    covariance_zeta[n_zeta:, n_zeta:] = phi @ prior_initial @ phi.T + process
    mean_zeta = np.concatenate([initial_mean, phi @ initial_mean])

    n_data = n_epochs * n_samples
    big_a = np.zeros((n_data, n_theta))
    big_c = np.zeros((n_data, n_epochs * n_zeta))
    for epoch in range(n_epochs):
        rows = slice(epoch * n_samples, (epoch + 1) * n_samples)
        big_a[rows, :] = designs[epoch]
        big_c[rows, epoch * n_zeta : (epoch + 1) * n_zeta] = responses[epoch]
    covariance = sigma**2 * np.eye(n_data) + big_c @ covariance_zeta @ big_c.T
    _, logdet = np.linalg.slogdet(covariance)

    def oracle(theta):
        resid = (
            np.concatenate(data) - big_a @ np.asarray(theta, float) - big_c @ mean_zeta
        )
        return float(
            -0.5 * resid @ np.linalg.solve(covariance, resid)
            - 0.5 * (logdet + n_data * np.log(2 * np.pi))
        )

    return blocks, transition, oracle


@pytest.mark.parametrize("theta", [(0.0, 0.0), (0.4, -1.1), (-2.0, 3.0)])
def test_a_wide_chain_is_exact_too_which_is_what_fixes_the_scaling_side(theta):
    """`diag(1/q) @ phi`, not `phi @ diag(1/q)`. The scalar fixture cannot tell.

    Measured gaps against this campaign's own dense oracle: 7.1e-15, 7.1e-15 and
    5.0e-14 nats at the three probes. Verified discriminating by sabotage:
    writing the transition rows as `-resolved.phi * inverse_process` -- which is
    `phi @ diag(1/q)` -- leaves **every other test in this file green** and
    breaks all three of these, by -0.074891, +0.007844 and -0.970236 nats.

    Three probes rather than one, and that is not padding: the middle probe's
    error is 0.0078 nats, ten times smaller than the first probe's and a hundred
    and twenty times smaller than the third's -- small enough that a single-point
    test at that theta with a generous band would have called the transposed
    filter exact. The wrong ordering is a rotation, so how much it costs depends
    on where in theta you stand.
    """
    blocks, transition, oracle = _wide_campaign()
    filtered = float(
        chain_log_likelihood(
            blocks,
            transition,
            {"a": jnp.asarray(theta[0]), "b": jnp.asarray(theta[1])},
            names=("a", "b"),
            shapes=((), ()),
        )
    )
    assert filtered == pytest.approx(oracle(theta), abs=1e-9)


def test_the_blocks_must_be_as_wide_as_the_latents_they_claim_to_be_over():
    """A stored block is a form in an ordered vector, not a labelled table."""
    factors, targets, offsets = bank.stacked()
    with pytest.raises(
        chain.StateValidationError, match="a different one is not a rename"
    ):
        chain_marginal(
            (factors[:, :, :2], targets, offsets),
            _transition(),
            _values(bank.PROBES[0]),
            names=bank.THETA_NAMES,
            shapes=((), ()),
        )


# ------------------------------------------------------------ the six constants --
#
# Each test deletes exactly one and pins the gap it opens. The bands are roughly
# 5 % of the measured value or one order below it, whichever is looser: tight
# enough that the constant cannot be half-present, loose enough to survive a
# different RNG draw of the fixture.


def test_the_initial_zeta_prior_normalisation_is_carried(monkeypatch):
    """New to the chain: nobody else could have added it.

    `-0.5 logdet(2 pi P0)` is the prior on zeta_1. It is not `compress`'s -- the
    epoch blocks know nothing about the chain -- and it is not `marginalise`'s.
    Measured gap: +0.9189 nats, which is exactly `0.5 log(2 pi)` for a scalar
    chain at `P0 = 1.0`, so a fixture with `INITIAL_VAR != 1` would separate the
    two halves further.
    """
    base = _filtered(bank.PROBES[1])
    monkeypatch.setattr(chain, "_initial_log_norm", lambda transition: 0.0)
    assert _filtered(bank.PROBES[1]) - base == pytest.approx(0.9189, abs=0.02)


def test_the_transition_normalisation_is_the_whole_density_not_half_of_it(
    monkeypatch,
):
    """Section 6 names `0.5 logdet Q^-1`; the density also carries `2 pi`.

    The `2 pi` half cancels against the marginalisation's `+0.5 n log(2 pi)`,
    which is why the shorthand reads plausible. Following it leaves `+0.5
    log(2 pi)` per transition: +0.9189 an epoch, +918 nats over a thousand-epoch
    campaign, and no effect on any posterior mean, width or gradient. Measured
    here: dropping the whole term costs +2.8618 nats over five transitions;
    keeping only the spec's half costs +4.5947, which is 0.9189 x 5 exactly.
    """
    base = _filtered(bank.PROBES[1])
    monkeypatch.setattr(chain, "_transition_log_norm", lambda transition: 0.0)
    whole = _filtered(bank.PROBES[1]) - base
    assert whole == pytest.approx(2.8618, abs=0.05)

    # ...and the spec's shorthand, which keeps only `0.5 logdet Q^-1`:
    monkeypatch.setattr(
        chain,
        "_transition_log_norm",
        lambda transition: -jnp.sum(jnp.log(transition.process_std)),
    )
    half = _filtered(bank.PROBES[1]) - base
    assert half == pytest.approx(4.5947, abs=0.05)
    per_transition = half / (bank.N_EPOCHS - 1)
    assert per_transition == pytest.approx(0.5 * float(np.log(2 * np.pi)), abs=1e-9)


def test_the_marginalisation_constant_is_carried(monkeypatch):
    """`+0.5 n log(2 pi) - sum log|U_bb,ii|`, at every step and at the last one.

    Deleted here by substituting a kernel that omits it, which also pins that
    the filter CALLS the kernel rather than inlining a second copy: an inlined
    copy would be untouched by this monkeypatch and the test would fail with a
    gap of zero, naming the right cause.

    Measured: +7.2619 nats. Split by site, by patching only the scan body and
    only the final call: 6.2764 from the five in-chain marginalisations and
    0.9855 from the final one -- so the final marginalisation's constant is
    nearly a nat on its own, and it is the one nobody else could have added.
    """
    base = _filtered(bank.PROBES[1])

    def without_the_constant(factor, target, offset, n_block):
        width = factor.shape[1]
        upper = jnp.linalg.qr(
            jnp.concatenate([factor, target[:, None]], axis=1), mode="r"
        )
        keep = min(upper.shape[0], width)
        corner = upper[keep:, width]
        return (
            upper[n_block:keep, n_block:width],
            upper[n_block:keep, width],
            offset - 0.5 * jnp.sum(corner**2),
            jnp.abs(jnp.diag(upper)),
        )

    monkeypatch.setattr(chain, "marginalise_arrays", without_the_constant)
    gap = _filtered(bank.PROBES[1]) - base
    assert gap == pytest.approx(7.2619, abs=0.2)  # 6.2764 in-chain + 0.9855 final
    assert gap != pytest.approx(0.0, abs=1e-6), (
        "the filter is not calling marginalise_arrays -- a second copy of the "
        "constant is exactly how one of them gets fixed and the other does not"
    )


def test_the_fold_corner_is_carried(monkeypatch):
    """Plan A's bug, one layer along, and by far the largest of the six.

    The corner is the part of two stacked residuals that no quadratic form in
    the retained columns can express. Measured: +45.9502 nats over six epochs,
    growing with the campaign, invisible in the posterior's shape.
    """
    base = _filtered(bank.PROBES[1])
    real = chain._fold

    def without_corner(factor, target, offset, block, width):
        new_factor, new_target, new_offset = real(factor, target, offset, block, width)
        # add the corner back on, which is the same as never subtracting it
        upper = jnp.linalg.qr(
            jnp.concatenate(
                [
                    jnp.concatenate([factor, target[:, None]], axis=1),
                    jnp.concatenate([block[0], block[1][:, None]], axis=1),
                ],
                axis=0,
            ),
            mode="r",
        )
        keep = min(upper.shape[0], width)
        return (
            new_factor,
            new_target,
            new_offset + 0.5 * jnp.sum(upper[keep:, width] ** 2),
        )

    monkeypatch.setattr(chain, "_fold", without_corner)
    assert _filtered(bank.PROBES[1]) - base == pytest.approx(45.9502, rel=0.05)


def test_the_marginalisation_corner_is_structurally_empty():
    """The one that does NOT matter, pinned so nobody writes a vacuous test for it.

    The marginalisation QR is square -- `rows == width` -- so `upper[keep:,
    width]` is a length-zero slice and `-0.5 rho^2` is exactly 0.0. A test that
    deleted it and asserted a nonzero gap would fail; a test that deleted it and
    asserted "the constants matter" would pass while proving nothing. The corner
    that does matter is the fold's, above, and it is +45.95 rather than 0.
    """
    blocks = bank.stacked()
    factor = blocks[0][0]
    width = factor.shape[1]
    upper = jnp.linalg.qr(
        jnp.concatenate([factor, blocks[1][0][:, None]], axis=1), mode="r"
    )
    assert upper.shape[0] == width
    assert upper[min(upper.shape[0], width) :, width].shape == (0,)


def test_deleting_the_marginalisation_corner_changes_nothing(monkeypatch):
    """The same fact, measured through the filter rather than argued from shapes.

    Exactly 0.0000 nats, bit for bit -- which is why the marginalisation's corner
    is not one of the six and why the fold's is.
    """
    base = _filtered(bank.PROBES[1])
    real = chain.marginalise_arrays

    def without_the_corner(factor, target, offset, n_block):
        width = factor.shape[1]
        upper = jnp.linalg.qr(
            jnp.concatenate([factor, target[:, None]], axis=1), mode="r"
        )
        keep = min(upper.shape[0], width)
        new_factor, new_target, new_offset, pivots = real(
            factor, target, offset, n_block
        )
        return (
            new_factor,
            new_target,
            new_offset + 0.5 * jnp.sum(upper[keep:, width] ** 2),
            pivots,
        )

    monkeypatch.setattr(chain, "marginalise_arrays", without_the_corner)
    assert _filtered(bank.PROBES[1]) - base == 0.0


def test_the_masked_data_normalisation_reaches_the_answer():
    """`compress`'s constant, which the chain must not drop on the way through.

    Measured -6.8408 nats over six epochs. Negative, unlike the other five, and
    that is not a sign error: `2 pi sigma^2` is 0.565 at this fixture's noise,
    so its log is negative and removing the term *lowers* the density.
    """
    base = _filtered(bank.PROBES[1])
    factors, targets, offsets = bank.stacked()
    stripped = (
        factors,
        targets,
        offsets + 0.5 * bank.N_SAMPLES * float(np.log(2 * np.pi * bank.SIGMA**2)),
    )
    assert _filtered(bank.PROBES[1], stripped) - base == pytest.approx(
        -6.8408, abs=0.05
    )


def test_the_final_marginalisation_happens_at_all():
    """The last zeta is integrated out, not evaluated at zero.

    Nobody else's constant: the recursion's last step has no successor to hand
    the chain to, and a filter that simply returned the carry would give a
    finite, smooth, correctly-shaped answer to a different question.
    """
    base = _filtered(bank.PROBES[1])
    marginal = chain_marginal(
        bank.stacked(),
        _transition(),
        _values(bank.PROBES[1]),
        names=bank.THETA_NAMES,
        shapes=((), ()),
    )
    assert marginal.factor.shape[1] == bank.N_THETA, (
        "zeta_N is still a column: the final marginalisation did not run"
    )
    assert base == pytest.approx(bank.oracle(bank.PROBES[1]), abs=1e-9)
