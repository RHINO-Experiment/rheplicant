"""Section 12.8. The square root is not a style choice; it is the reason float64 holds.

Both ends of every axis, because the failure mode is U-shaped: a probe over
moderate N and moderate tau passes while both corners are broken. That is the
boundary-validation rule, and this file is a dispatcher-free instance of it --
what is compared at each corner is two METHODS for the same quantity, evaluated
directly, not one method's output either side of a threshold.

**What this fixture could and could not show.** Section 3's claim is that
accumulating ``F`` directly and taking explicit Schur complements goes
indefinite in float64 on a near-degenerate campaign; the spec quotes
``lambda_min = -1.4e-8`` at ``cond = 7.9e13`` for its own fixture -- six globals
graded 1 to 1e-5 at ``sigma = 1e-3``. It does not reproduce here, and it was
measured rather than assumed: at every one of the six cells below -- and at
``N = 200`` beside them, probed and not shipped because it says nothing the
other three N values do not -- the two methods agree to all nine printed digits,
and the largest condition number this fixture reaches is ``7.66`` (at ``N = 1``,
where the campaign is one night).
Grading the two globals by 1e-5 and dropping sigma to 1e-3 was tried as well and
raised the condition number to ``5.4e9`` while the two methods still agreed to
``2.4e-15`` relative -- because the block being Schur-complemented here is one
``zeta`` component per epoch, a ``1 x 1`` solve, and a ``1 x 1`` solve does not
lose digits. What separates the two methods is a *wide* near-singular nuisance
block, which this chain does not have.

So the assertion that ships is the structural one -- the square-root form's
smallest eigenvalue is never below zero and never below the explicit form's --
and "the explicit form did not fail here either" is recorded as the honest
result rather than replaced by a fixture built to make it fail.

What the comparison does buy, and it is not nothing: ``_explicit_schur`` is an
independent NumPy implementation of the same marginal curvature, sharing no line
with the SRIF scan, and it agrees with it **matrix by matrix** to between
2.3e-17 and 4.5e-14 relative over the grid. That is a cross-check on the filter's
marginal, not merely on its spectrum -- see
``test_the_two_methods_agree_entry_by_entry_and_not_only_in_spectrum`` for why
the distinction is the whole test.
"""

from functools import cache

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference.chain import ChainMemory, ornstein_uhlenbeck
from rheplicant.inference.compress import compress_linear
from tests.evidence import chain_bank as bank

N_VALUES = (1, 10, 1000)
TAU_VALUES = (5.0, 5e4)

#: Where every campaign below is evaluated. A :class:`LinearGaussianTransition`
#: ignores it, but ``ChainMemory.fisher`` requires it rather than defaulting --
#: with a ``HyperTransition`` the marginal curvature is a function of theta, and
#: a default point would be a linearisation nobody declared.
AT = {"t_rx": jnp.asarray(0.0), "gain_slope": jnp.asarray(0.0)}

#: Declared column order ``("t_rx", "gain_slope")`` into flatten order
#: ``("gain_slope", "t_rx")``. See ``_explicit_schur``.
SWAP = np.array([[0.0, 1.0], [1.0, 0.0]])


@cache
def _campaign(n_epochs, tau, seed=0):
    """``n_epochs`` nights of a scalar OU chain, accumulated in order.

    Cached because the grid below asks for the same ``(N, tau)`` from three
    tests and the ``N = 1000`` cells dominate the file's runtime. Safe to share:
    a :class:`ChainMemory` is an ``eqx.Module``, ``remember`` returns a new one,
    and nothing here mutates. Uncached the file takes about 90 s; cached, 35.

    ``remember`` is O(N^2) in total over a campaign -- ``jnp.concatenate`` on a
    growing stack -- which is stated in ``ChainMemory``'s docstring and is not
    to be "fixed" by making ``remember`` mutate. At N = 1000 it is a few seconds
    and the recursion itself is one ``lax.scan``.
    """
    rng = np.random.default_rng(seed)
    memory = ChainMemory(bank.factorization(ornstein_uhlenbeck(tau=tau, sigma=1.0)))
    for e in range(n_epochs):
        a = rng.normal(size=(bank.N_SAMPLES, bank.N_THETA))
        c = rng.normal(size=(bank.N_SAMPLES, 1)) + 1.0
        d = rng.normal(size=(bank.N_SAMPLES,))
        memory = memory.remember(
            compress_linear(
                design={
                    "t_rx": a[:, :1],
                    "gain_slope": a[:, 1:],
                    bank.ZETA_NAME: c,
                },
                observed=jnp.asarray(d),
                noise_std=bank.SIGMA,
                shapes={"t_rx": (), "gain_slope": (), bank.ZETA_NAME: ()},
                epoch_id=f"e{e}",
            )
        )
    return memory


def _prior_fisher():
    """The globals' prior curvature, in **flatten** order.

    ``("gain_slope", "t_rx")``, matching what ``ChainMemory.fisher`` returns.
    Both entries happen to be 0.25 on this fixture, so a transposition here
    would be invisible -- which is why the order is written out rather than
    taken from ``bank.THETA_NAMES``, and why ``SWAP`` exists below.
    """
    return np.diag([1.0 / bank.PRIOR_STD[name] ** 2 for name in ("gain_slope", "t_rx")])


def _explicit_schur(memory, values):
    """The `(F, b)` recursion section 3 rejects, for comparison only.

    Written here rather than in `src/` on purpose: it is the method the design
    measured and discarded, and shipping it would give a caller a way to choose
    the one that goes indefinite.

    **The result is in declared order, ``("t_rx", "gain_slope")``, and
    ``ChainMemory.fisher`` returns flatten order.** The stored blocks are
    columned in declared order, so that is what falls out here;
    ``BayesMemory.fisher`` permutes into the order ``jax`` flattens a dict into,
    which is sorted. Measured on this fixture at ``N = 1``: the two matrices
    differ by ``15.58`` entry-wise and have **identical** eigenvalues, because a
    symmetric permutation is a similarity transform. Every caller below applies
    ``SWAP`` before comparing entries, and none of them compares raw.
    """
    transition = memory.transition.at(values)
    n_theta, n_zeta = bank.N_THETA, transition.width
    q = float(transition.process_std[0]) ** 2
    phi = float(transition.phi[0, 0])
    factors = np.asarray(memory.stacked[0])
    fisher = np.zeros((n_theta + n_zeta, n_theta + n_zeta))
    fisher[n_theta:, n_theta:] = 1.0 / float(transition.initial_std[0]) ** 2
    for e in range(factors.shape[0]):
        fisher = fisher + factors[e].T @ factors[e]
        if e == factors.shape[0] - 1:
            break
        widened = np.zeros((n_theta + 2 * n_zeta,) * 2)
        widened[: n_theta + n_zeta, : n_theta + n_zeta] = fisher
        widened[n_theta, n_theta] += phi**2 / q
        widened[n_theta, -1] += -phi / q
        widened[-1, n_theta] += -phi / q
        widened[-1, -1] += 1.0 / q
        keep = list(range(n_theta)) + [n_theta + 2 * n_zeta - 1]
        drop = [n_theta]
        f_bb = widened[np.ix_(drop, drop)]
        f_bk = widened[np.ix_(drop, keep)]
        fisher = widened[np.ix_(keep, keep)] - f_bk.T @ np.linalg.solve(f_bb, f_bk)
    f_bb = fisher[n_theta:, n_theta:]
    f_bk = fisher[n_theta:, :n_theta]
    return fisher[:n_theta, :n_theta] - f_bk.T @ np.linalg.solve(f_bb, f_bk)


@pytest.mark.parametrize("n_epochs", N_VALUES)
@pytest.mark.parametrize("tau", TAU_VALUES)
def test_the_accumulated_fisher_plus_the_prior_is_positive_definite(n_epochs, tau):
    """Section 12.8's first half, at both ends of N and of tau.

    Measured ``(lambda_min, cond)`` of ``sum_e F_e + F_prior``:

    ==========  ==========================  ==========================
    N           tau = 5                     tau = 5e4
    ==========  ==========================  ==========================
    1           2.627193817, 7.664604984    2.627193817, 7.664604984
    10          138.6465036, 2.813895090    221.6864547, 1.971082663
    1000        32547.90323, 1.075804205    42540.90114, 1.057722309
    ==========  ==========================  ==========================

    The two tau columns agree exactly at ``N = 1`` and that is arithmetic, not a
    coincidence to be explained away: a one-epoch campaign has no transition in
    it at all, and ``ornstein_uhlenbeck`` gives ``initial_std = sigma`` whatever
    ``tau`` is. They separate from ``N = 2`` on, and by ``N = 1000`` the slow
    chain leaves 31 % more information in theta -- a drift with a 5e4-epoch
    correlation time is nearly a constant, and a constant offset is cheaper to
    separate from theta than a fresh nuisance every night.

    The condition number falls toward 1 as the campaign grows, which is the
    direction section 3 predicts and the opposite of what the explicit form is
    accused of doing.
    """
    memory = _campaign(n_epochs, tau)
    total = np.asarray(memory.fisher(at=AT).matrix) + _prior_fisher()
    eigenvalues = np.linalg.eigvalsh(total)
    # `not <=`, never `>`: NaN loses every comparison, so `lambda_min > 0` waves
    # a poisoned campaign through while reporting the NaN in the same breath.
    assert not (eigenvalues[0] <= 0.0), f"lambda_min = {eigenvalues[0]:.3e}"
    # And the other end, because `inf > 0` is True: an infinite eigenvalue
    # passes the line above and is not a conditioned campaign either.
    assert np.all(np.isfinite(eigenvalues)), f"eigenvalues = {eigenvalues}"
    assert np.isfinite(eigenvalues[-1] / eigenvalues[0])


@pytest.mark.parametrize("n_epochs", N_VALUES)
@pytest.mark.parametrize("tau", TAU_VALUES)
def test_the_square_root_form_is_never_worse_than_the_explicit_schur_complement(
    n_epochs, tau
):
    """Section 3's measurement, on this fixture, with the digits recorded.

    The spec's numbers -- float64 ``lambda_min = -1.4e-8`` at ``tau = 5e4``,
    ``cond = 7.9e13`` -- are its own fixture's: six globals graded 1 to 1e-5 at
    ``sigma = 1e-3``, 1000 epochs. What is asserted here is the STRUCTURAL claim
    they were evidence for: the square-root form's smallest eigenvalue is never
    below the explicit form's, and never below zero.

    **Measured result: the explicit form does not fail here.** At all six cells
    the two ``lambda_min`` agree to every digit the table in
    ``test_the_accumulated_fisher_plus_the_prior_is_positive_definite`` prints,
    and the relative gap is at most 5e-16. The module docstring records what was
    tried to provoke a separation and why a scalar chain cannot show one. This is
    reported rather than engineered around; the assertion that matters is
    ``srif > 0``, and it is not relaxed.
    """
    memory = _campaign(n_epochs, tau)
    srif = np.linalg.eigvalsh(
        np.asarray(memory.fisher(at=AT).matrix) + _prior_fisher()
    )[0]
    # The prior is permuted INTO declared order here rather than the Schur
    # complement out of it, so that both spectra are of matrices in the same
    # basis. Both entries are 0.25 on this fixture, so the permutation changes
    # nothing arithmetically -- it is written because the next fixture's will not
    # be, and a spectrum is exactly the quantity that would not say so.
    explicit = np.linalg.eigvalsh(
        _explicit_schur(memory, AT) + SWAP @ _prior_fisher() @ SWAP.T
    )[0]
    assert not (srif <= 0.0), f"SRIF lambda_min = {srif:.6e}"
    assert srif >= explicit - 1e-12 * abs(srif), (
        f"SRIF {srif:.6e} vs explicit Schur {explicit:.6e} at N = {n_epochs}, "
        f"tau = {tau}"
    )


@pytest.mark.parametrize("n_epochs", N_VALUES)
@pytest.mark.parametrize("tau", TAU_VALUES)
def test_the_two_methods_agree_entry_by_entry_and_not_only_in_spectrum(n_epochs, tau):
    """The comparison the eigenvalue test above cannot make.

    A symmetric permutation is a similarity transform, so every eigenvalue test
    in this file is blind to one -- and there *is* one here: the stored blocks
    are columned in declared order ``("t_rx", "gain_slope")`` and
    ``ChainMemory.fisher`` returns flatten order ``("gain_slope", "t_rx")``.
    Measured at ``N = 1``: the raw matrices differ by ``15.58`` entry-wise while
    their spectra are bit-identical. This fixture declares its globals
    non-alphabetically for exactly this reason; on an alphabetical one the
    permutation is the identity and this test would pass with the ``SWAP``
    deleted.

    With the permutation applied, an independent NumPy accumulation of ``F``
    with explicit Schur complements reproduces the SRIF scan's marginal
    curvature to a relative ``2.3e-17`` at ``N = 1``, ``9.3e-16`` at ``N = 10``
    and ``4.5e-14`` at ``N = 1000`` -- growing as the campaign does, which is
    what accumulating a thousand rank-one updates in double precision costs.
    """
    memory = _campaign(n_epochs, tau)
    srif = np.asarray(memory.fisher(at=AT).matrix)
    assert memory.fisher(at=AT).names == ("gain_slope", "t_rx")
    explicit = SWAP @ _explicit_schur(memory, AT) @ SWAP.T
    relative = np.max(np.abs(srif - explicit)) / np.max(np.abs(srif))
    assert relative < 1e-12, (
        f"SRIF and explicit Schur disagree by {relative:.3e} relative at "
        f"N = {n_epochs}, tau = {tau}"
    )


def test_a_rank_deficient_night_is_not_an_error_and_the_prior_is_the_floor():
    """Section 2.2 -- but the plan's version of it asserted the wrong thing.

    The plan wrote this as "one night does not constrain all of theta" against a
    single epoch of ``_campaign``. On this fixture that is simply false: four
    samples over two globals and one ``zeta`` is a full-rank design, and the
    measured ``lambda_min`` of the one-epoch marginal is ``2.377``, not a
    near-zero. A test asserting it was below 1e-6 fails, and correctly.

    What is true, and is what section 2.2 is about, needs a night that really is
    degenerate. Here the two globals share one column up to a factor of two, so
    only their sum is measured. Measured on that night:

    * the accumulated Fisher's eigenvalues are ``-6.21e-16`` and ``13.26`` --
      the small one is **negative**, by roundoff, which is why every guard in
      this file is written ``not (x <= 0)`` and why the accumulator alone is
      never the thing asserted positive;
    * adding the prior puts ``lambda_min`` at exactly ``0.25``, which is
      ``1 / PRIOR_STD^2`` -- section 3's "stays at the prior floor", as a number;
    * the density is finite: ``-9.3676`` nats. A rank-deficient night is a short
      factor, not an error.
    """
    rng = np.random.default_rng(0)
    memory = ChainMemory(bank.factorization(ornstein_uhlenbeck(tau=5.0, sigma=1.0)))
    column = rng.normal(size=(bank.N_SAMPLES, 1))
    memory = memory.remember(
        compress_linear(
            design={
                "t_rx": column,
                "gain_slope": 2.0 * column,
                bank.ZETA_NAME: rng.normal(size=(bank.N_SAMPLES, 1)) + 1.0,
            },
            observed=jnp.asarray(rng.normal(size=(bank.N_SAMPLES,))),
            noise_std=bank.SIGMA,
            shapes={"t_rx": (), "gain_slope": (), bank.ZETA_NAME: ()},
            epoch_id="collinear",
        )
    )
    fisher = np.asarray(memory.fisher(at=AT).matrix)
    eigenvalues = np.linalg.eigvalsh(fisher)
    assert eigenvalues[0] == pytest.approx(0.0, abs=1e-12)
    assert eigenvalues[-1] == pytest.approx(13.262953, rel=1e-6)
    assert np.linalg.eigvalsh(fisher + _prior_fisher())[0] == pytest.approx(
        0.25, rel=1e-9
    )
    assert float(memory.log_likelihood(AT)) == pytest.approx(-9.367610, rel=1e-6)


def test_a_full_campaign_of_one_night_still_constrains_both_globals():
    """A guard that over-refuses is its own bug -- the nearest legitimate case.

    The test above is about a night that is genuinely degenerate. This one pins
    the other side of the same boundary: a *typical* single night on this fixture
    constrains both globals on its own, at ``lambda_min = 2.377``, so nothing
    here treats "N = 1" as a synonym for "rank deficient".
    """
    fisher = np.asarray(_campaign(1, 5.0).fisher(at=AT).matrix)
    assert np.linalg.eigvalsh(fisher)[0] == pytest.approx(2.377194, rel=1e-6)


def test_a_thousand_epochs_of_chain_stays_finite_end_to_end():
    """The far corner of both axes, through the density rather than the curvature.

    ``tau = 5e4`` over 1000 epochs is ``phi = 0.99998`` and
    ``process_std = 6.32e-3``: a chain that barely moves, which is the near-
    degenerate direction, and the one a filter that accumulates ``1 / q`` rows
    would blow up on.
    """
    memory = _campaign(1000, 5e4)
    values = {"t_rx": jnp.asarray(0.3), "gain_slope": jnp.asarray(-0.2)}
    assert np.isfinite(float(memory.log_likelihood(values)))
    assert np.isfinite(float(memory.log_posterior(values)))
