"""T0/T1/T2 agree at every threshold, at the extreme parameter values included.

Section 12.12. Each tier is called **directly**. Going through ``compress``
would route one input to one method and show only that the routed function is
continuous, which it is for anything analytic -- the claim is that two *methods*
give the same number at the same point, and a dispatcher cannot be asked that
question. Nothing in this file imports ``compress``.

The model is linear and its design columns are seeded into the span, which is
the one construction where all three tiers are exact simultaneously: T2 is a
sufficient statistic, T1's projection is the identity on ``mu`` because ``mu``
lies in the span, and T0 is the definition. Any disagreement here is a defect
and not a truncation, so the comparisons are absolute log-densities at a fixed
tolerance rather than moments -- a constant error is invisible in a posterior's
shape, and both constants T1 carries (the masked normalisation and the QR
corner) are pure offsets.

The extremes are the point, not decoration. ``sigma`` spans four decades, so
SNR runs from 1e2 to 1e-2 and the stored offsets change sign; ``n_S`` is tested
at the smallest value that can be exact and at the largest the data admits;
``theta`` is probed at the corner of the declared box and outside it; one epoch
is rank-deficient; one epoch is flagged, so the metric has exact zeros in it;
and the campaign length runs from 1 to 10,000.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.compress import compress_linear, compress_reduced_basis
from rheplicant.inference.compressed import QuadraticLikelihood, RawLikelihood
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory
from rheplicant.inference.noise import HomoscedasticNoise
from rheplicant.inference.reduced_basis import ReducedBasis, orthonormalise
from rheplicant.inference.sqrtinfo import SqrtInfo

N_DATA, N_THETA = 24, 3
BOX = (-2.0, 2.0)
TRUTH = jnp.array([0.4, -1.1, 0.7])

#: Samples the flagged epoch could not see. Their data is NaN, which is the
#: usual reason a sample is flagged in the first place, so an epoch built by
#: multiplying by a zero weight rather than selecting on the mask returns NaN
#: from every tier -- and `nan == pytest.approx(x)` is False in both
#: directions, which is why the flagged case is asserted finite as well.
FLAGGED = (3, 11, 20)


class _Normal:
    """Minimal duck-typed prior: only ``log_prob`` is read downstream."""

    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale

    def log_prob(self, x):
        return -0.5 * (
            ((x - self.loc) / self.scale) ** 2 + jnp.log(2 * jnp.pi * self.scale**2)
        )


def _design(rank=N_THETA, key=None):
    """``(N_DATA, N_THETA)``, optionally with a repeated column.

    A repeated column is a genuinely rank-deficient epoch: the data constrains a
    subspace, which section 2.2 says is the normal case and not an error.
    """
    columns = jax.random.normal(
        jax.random.key(70) if key is None else key, (N_DATA, rank)
    )
    if rank < N_THETA:
        columns = jnp.concatenate([columns, columns[:, :1]], axis=1)
    return columns


def _space():
    latent = Latent("x", init=jnp.zeros(N_THETA), prior=_Normal(0.0, 5.0), linear=True)
    return ParameterSpace(latents=(latent,), bindings=(Bind("x", into=lambda p: p.x),))


def _basis(design, n_basis=N_THETA, seed=0):
    """A dictionary whose first rows ARE the design columns.

    The design columns of a linear model are its score directions, so seeding
    them is what makes the span contain ``mu`` exactly at every ``theta`` --
    including outside the training box, where a truncated span would not.

    Built in the **reference** metric, a uniform ``1/sigma`` with no flags, so
    that a flagged epoch below is a real mismatch between the campaign's
    dictionary and the epoch's metric rather than a dictionary rebuilt to suit
    it.
    """
    weight = jnp.ones(N_DATA)
    filler = jax.random.normal(
        jax.random.key(seed + 99), (max(n_basis - N_THETA, 0), N_DATA)
    )
    candidates = jnp.concatenate([design.T, filler], axis=0)
    rows = orthonormalise(candidates * weight)[:n_basis] / weight
    return ReducedBasis(
        rows=rows,
        weight=weight,
        predict=lambda values: design @ values["x"],
        reference=jnp.zeros(N_DATA),
        support={"x": BOX},
        reference_values={"x": jnp.zeros(N_THETA)},
    )


def _epoch(design, sigma, seed=0, flagged=()):
    """``(data, sigma_row)`` for one epoch, NaN where the epoch saw nothing."""
    row = jnp.full(N_DATA, float(sigma))
    data = design @ TRUTH + row * jax.random.normal(jax.random.key(seed), (N_DATA,))
    for index in flagged:
        row = row.at[index].set(jnp.inf)
        data = data.at[index].set(jnp.nan)
    return data, row


def _raw(design, data, row):
    return RawLikelihood(
        predict=lambda values: design @ values["x"],
        observed=data,
        sigma=row,
        names=("x",),
        epoch_id="e0",
    )


def _quadratic(design, data, row):
    return compress_linear(
        design={"x": design},
        observed=data,
        noise_std=row,
        shapes={"x": (N_THETA,)},
        epoch_id="e0",
    )


def _reduced(design, data, row, n_basis=N_THETA, seed=0):
    return compress_reduced_basis(
        _basis(design, n_basis=n_basis, seed=seed),
        observed=data,
        noise=HomoscedasticNoise(row),
        epoch_id="e0",
    )


def _tiers(design, sigma, n_basis=N_THETA, seed=0, flagged=()):
    """T0, T1 and T2 over the same epoch, each built directly."""
    data, row = _epoch(design, sigma, seed=seed, flagged=flagged)
    return (
        _raw(design, data, row),
        _reduced(design, data, row, n_basis=n_basis, seed=seed),
        _quadratic(design, data, row),
    )


def _probes():
    return {
        "centre": {"x": jnp.zeros(N_THETA)},
        "interior": {"x": TRUTH},
        "corner": {"x": jnp.full(N_THETA, BOX[1])},
        "outside": {"x": jnp.full(N_THETA, 1.5 * BOX[1])},
    }


@pytest.mark.parametrize("sigma", [1e-2, 1.0, 1e2])
@pytest.mark.parametrize("where", ["centre", "interior", "corner", "outside"])
def test_the_three_tiers_agree_at_every_snr_and_every_corner(sigma, where):
    raw, reduced, quadratic = _tiers(_design(), sigma)
    probe = _probes()[where]
    expected = float(raw(probe))
    assert np.isfinite(expected)
    assert float(quadratic(probe)) == pytest.approx(expected, rel=1e-9, abs=1e-8)
    assert float(reduced(probe)) == pytest.approx(expected, rel=1e-9, abs=1e-8)


@pytest.mark.parametrize("n_basis", [N_THETA, N_DATA - 1])
def test_they_agree_at_both_ends_of_the_admissible_basis_size(n_basis):
    """``n_S`` at its minimum for exactness, and at the largest the data allows.

    One more row than samples is refused outright -- the last test in this file
    -- so ``N_DATA - 1`` is the nearest legitimate case to that refusal, which
    is the case a guard that over-refuses would take with it.
    """
    raw, reduced, _ = _tiers(_design(), 0.1, n_basis=n_basis)
    assert reduced.basis.n_basis == n_basis
    probe = _probes()["interior"]
    assert float(reduced(probe)) == pytest.approx(float(raw(probe)), rel=1e-9, abs=1e-8)


def test_a_rank_deficient_epoch_agrees_across_tiers():
    """Section 2.2: one epoch constrains a subspace, and that is not an error."""
    design = _design(rank=N_THETA - 1)
    raw, reduced, quadratic = _tiers(design, 0.1)
    assert int(
        np.linalg.matrix_rank(np.asarray(quadratic.info.fisher()), tol=1e-9)
    ) == (N_THETA - 1)
    # The dictionary is short by the same one direction, for the same reason.
    assert reduced.basis.n_basis == N_THETA - 1
    probe = _probes()["interior"]
    expected = float(raw(probe))
    assert float(quadratic(probe)) == pytest.approx(expected, rel=1e-9, abs=1e-8)
    assert float(reduced(probe)) == pytest.approx(expected, rel=1e-9, abs=1e-8)


@pytest.mark.parametrize("where", ["interior", "outside"])
def test_a_flagged_epoch_agrees_across_tiers(where):
    """The metric has exact zeros in it, and the data behind them is NaN.

    ``weight = 0`` is a threshold like any other, and the one this package has
    got wrong before: a masked path that multiplies by the zero instead of
    selecting on the mask returns ``nan``, because ``0.0 * nan`` is ``nan``.
    The finiteness assertion is not redundant with the agreement one -- NaN
    defeats every comparison, so ``nan == pytest.approx(nan)`` is False and a
    poisoned tier would fail loudly here but silently in any test that only
    checked a ratio or a difference against a tolerance.
    """
    design = _design()
    raw, reduced, quadratic = _tiers(design, 0.3, flagged=FLAGGED)
    assert raw.n_observed == N_DATA - len(FLAGGED)
    assert quadratic.n_observed == N_DATA - len(FLAGGED)
    assert reduced.n_observed == N_DATA - len(FLAGGED)
    probe = _probes()[where]
    expected = float(raw(probe))
    assert np.isfinite(expected)
    assert float(quadratic(probe)) == pytest.approx(expected, rel=1e-9, abs=1e-8)
    assert float(reduced(probe)) == pytest.approx(expected, rel=1e-9, abs=1e-8)


def _accumulate(design, sigma, n_epochs):
    """``(running SqrtInfo, sum of T0 densities)`` over ``n_epochs`` epochs.

    ``SqrtInfo.combine`` rather than ``BayesMemory.remember``, and the two are
    pinned equal by the test below rather than assumed so. The reason is cost,
    measured: ``remember`` rebuilds the memory on every call, and equinox
    re-flattens the whole module -- including an ``archive`` tuple that grows by
    one term per epoch -- so the accumulation is O(N^2). 2,000 epochs took 20.7 s
    with 4.0e6 calls to the generated ``QuadraticLikelihood`` flattener, which
    puts 10,000 near nine minutes. The arithmetic under test is one
    ``SqrtInfo.combine`` per epoch either way; the archive is bookkeeping this
    claim does not touch.
    """
    probe = _probes()["interior"]
    running = SqrtInfo.null(("x",), ((N_THETA,),))
    total = 0.0
    for index in range(n_epochs):
        data, row = _epoch(design, sigma, seed=index)
        running = SqrtInfo.combine(running, _quadratic(design, data, row).info)
        total += float(_raw(design, data, row)(probe))
    return running, total


@pytest.mark.parametrize("n_epochs", [1, 10_000])
def test_one_epoch_and_ten_thousand_agree_across_tiers(n_epochs):
    """The campaign length is the axis the claim is about, so it spans four decades.

    The accumulated factor is handed to a real ``BayesMemory`` and read through
    ``log_likelihood``, so what is compared is the memory's own density against
    the sum of ten thousand oracle densities -- not two spellings of the same
    QR.
    """
    running, total = _accumulate(_design(), 0.5, n_epochs)
    memory = BayesMemory(Factorization(_space()), accumulated=running)
    assert float(memory.log_likelihood(_probes()["interior"])) == pytest.approx(
        total, rel=1e-9
    )


def test_remember_and_combine_are_the_same_accumulation():
    """What the ten-thousand-epoch case substitutes, verified rather than assumed.

    Sixty-four epochs is short enough for the O(N^2) path above to be free and
    long enough that a per-term constant dropped by one route and kept by the
    other would show: the two offsets a term carries sum to order 1e2 nats each,
    so a single missing one is 64 times that.
    """
    design = _design()
    memory = BayesMemory(Factorization(_space()))
    running = SqrtInfo.null(("x",), ((N_THETA,),))
    for index in range(64):
        data, row = _epoch(design, 0.5, seed=index)
        term = _quadratic(design, data, row)
        memory = memory.remember(
            QuadraticLikelihood(
                info=term.info,
                epoch_id=f"n{index}",
                n_observed=term.n_observed,
                exact=True,
            )
        )
        running = SqrtInfo.combine(running, term.info)
    probe = _probes()["interior"]
    assert float(memory.log_likelihood(probe)) == pytest.approx(
        float(running.log_prob(probe)), rel=1e-12
    )


def test_an_over_rank_basis_is_refused_rather_than_returning_a_negative_form():
    """Above the numerical rank, ``c^T G c`` comes back finite and sometimes negative."""
    with pytest.raises(StateValidationError, match="more directions than"):
        ReducedBasis(
            rows=jnp.zeros((N_DATA + 1, N_DATA)),
            weight=jnp.ones(N_DATA),
            predict=lambda values: jnp.zeros(N_DATA),
            reference=jnp.zeros(N_DATA),
            support={"x": BOX},
        )
