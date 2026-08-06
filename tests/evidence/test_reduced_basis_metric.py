"""Section 5 requirement 1, split into the two claims it runs together.

Truncating in the wrong metric costs SENSITIVITY. Projecting in the wrong metric
costs BIAS. The spec attributes the bias to the truncation; the algebra says
otherwise, and this file measures both so the record is a number rather than a
sentence.

Write the whitened rows ``S_w`` and ``Pi = S_w^T (S_w S_w^T)^-1 S_w``. ``Pi`` is
the ``N^-1``-orthogonal projector onto the retained span *whatever metric chose
the rows*, so T1's score at the truth is
``((I - Pi) mu)^T N^-1 (Pi dmu/dtheta) = 0`` identically. Replace ``Pi`` by the
unweighted least-squares projector and the cross term stops vanishing, because
that operator is not self-adjoint in the likelihood's inner product.

The statistic is the score z at the truth on NOISELESS data:

    z = d/dtheta logL (theta*) / sqrt(-d2/dtheta2 logL (theta*))

which is exactly "the shift the compression induces, in units of this epoch's
sigma", with no Monte Carlo error. A coherent error repeats every epoch while
sigma falls as N^-1/2, so the campaign statistic is z sqrt(N) and the crossing
epoch is 1/z^2 -- section 7's sqrt(N) growth, computed rather than quoted.

**What this fixture measures, and it is not what the spec expects.**

At the shipped operating point -- ``n_S = 6``, score directions seeded --
neither projector is measurably biased:

    projector      z (per epoch)      N at which |z| sqrt(N) = 1
    N^-1            4.117e-12          5.9e+22
    unweighted     -1.394e-11          5.1e+21

Both are roundoff. The reason is the seeding, not the metric: with every score
direction in the span, ``mu(theta)`` near the truth lies in the span, so
``(I - Pi) mu = 0`` and the two projectors return the same vector. **The wrong
projector needs a truncation residual to act on.** That is the finding, it is
the opposite of "the truncation is the bias route", and it is why the mechanism
below has to be built deliberately rather than found at the operating point.

Build a basis that genuinely deletes the target -- ``n_S = 3``, plain SVD, no
seeding, measured ``r_t21 = 0.5618`` -- and the split appears cleanly:

    projector      z (per epoch)      N at which |z| sqrt(N) = 1
    N^-1            5.855e-12          2.9e+22
    unweighted      3.317e-02          909

So the mechanism is real and it is large once it has something to bite on: a
909-epoch campaign compressed with an unweighted projector onto a basis that has
deleted the 21 cm direction is biased by one sigma. The same basis with the
``N^-1`` projector stays centred at roundoff however badly it truncates -- it
simply measures less. Under a flat sigma the unweighted projector *is* the
``N^-1`` projector and the bias returns to ``1.6e-12``; the last test asserts
that, so a reader can see this file is capable of failing.

**For any campaign RHINO will actually run, the induced bias at the operating
point is not a concern** -- 5.1e+21 epochs is beyond any observing programme by
seventeen orders of magnitude. What is worth carrying forward is the conditional:
the number is small *because* the basis is seeded, and it degrades to a
thousand-epoch budget the moment a basis stops containing the science direction.

Seed sensitivity, checked over ``key(30)``, ``key(31)`` and ``key(32)``: the
``n_S = 3`` unweighted z agrees to 12 % across seeds and keeps its sign. At
``n_S = 2`` it does not -- 1.5e-2 against -3.8e-1 -- so the pin is at 3, where
the measurement is stable.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference.reduced_basis import (
    ReducedBasis,
    basis_fidelity,
    build_reduced_basis,
    orthonormalise,
    score_directions,
    select_svd,
)
from tests.evidence import rhino_bank

TARGET = "t21_depth"

#: Roundoff on this fixture. Every z that is "exactly zero" lands two to three
#: orders below this; every z that is not lands nine orders above it.
NOISE_FLOOR = 1e-9


@pytest.fixture(scope="module")
def bank():
    return rhino_bank.bank(jax.random.key(30), n_draws=400)


@pytest.fixture(scope="module")
def seeded(bank):
    """The shipped operating point: n_S = 6, every score direction in the span."""
    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=bank,
        n_basis=6,
        support=rhino_bank.support(),
    )


@pytest.fixture(scope="module")
def truncating(bank):
    """A basis that has deleted the target -- the only place the bias is visible.

    Plain SVD at n_S = 3 with seeding off. This is the failure
    `test_reduced_basis_build.py` pins; here it is the *instrument*, because a
    projector bias is a functional of the truncation residual and the operating
    point has none.
    """
    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=bank,
        n_basis=3,
        seed_scores=False,
        support=rhino_bank.support(),
    )


def _noiseless():
    predict, values = rhino_bank.forward()
    return predict(values), values


def _z_statistic(log_likelihood, values):
    """``(z, sigma_1)`` along TARGET at the truth: score over root information."""

    def along(shift):
        return log_likelihood({**values, TARGET: values[TARGET] + shift})

    zero = jnp.asarray(0.0)
    score = float(jax.grad(along)(zero))
    curvature = float(-jax.hessian(along)(zero))
    width = 1.0 / np.sqrt(curvature)
    return score * width, width


def _correct_projector_log_likelihood(basis, data, weight):
    """T1's own arithmetic, written out so the wrong variant differs by one line."""

    def differentiable(values):
        model = basis.rows.T @ basis.project(basis.predict(values))
        return -0.5 * jnp.sum(((data - model) * weight) ** 2)

    return differentiable


def _unweighted_projector_log_likelihood(basis, data, weight):
    """The same thing with the projection taken in unweighted L2 -- the mistake."""
    gram = basis.rows @ basis.rows.T

    def differentiable(values):
        prediction = jnp.ravel(basis.predict(values))
        coefficients = jnp.linalg.solve(gram, basis.rows @ prediction)
        model = basis.rows.T @ coefficients
        return -0.5 * jnp.sum(((data - model) * weight) ** 2)

    return differentiable


def test_the_band_has_the_sigma_spread_the_claim_depends_on():
    """Pinned, and it is not the number the spec quotes.

    Under RadiometerNoise sigma tracks the prediction, so the spread across
    60-85 MHz is the foreground's own: (85/60)^2.5 = 2.38872 for beta = -2.5,
    and the fixture measures 2.38896 because the trough perturbs the top of the
    band. Section 5 says "~6x", which would need beta near -5.1 over the same
    band; adding a receiver temperature compresses the ratio further rather than
    widening it. What matters for this file is only that the spread is not 1 --
    the last test shows what happens when it is.
    """
    predict, values = rhino_bank.forward()
    sigma = rhino_bank.noise().std(predict(values))
    assert float(jnp.max(sigma) / jnp.min(sigma)) == pytest.approx(2.38896, rel=1e-4)


def test_at_the_operating_point_neither_projector_is_measurably_biased(seeded):
    """The finding, and it is not the spec's claim.

    With every score direction seeded, mu(theta) near the truth lies in the span,
    so (I - Pi) mu = 0 and BOTH projectors return the same vector. The wrong
    projector has nothing to be wrong about. Its crossing epoch is 5.1e+21 --
    seventeen orders beyond any campaign RHINO will run.
    """
    data, values = _noiseless()
    weight = rhino_bank.weight()
    z_right, _ = _z_statistic(
        _correct_projector_log_likelihood(seeded, data, weight), values
    )
    z_wrong, _ = _z_statistic(
        _unweighted_projector_log_likelihood(seeded, data, weight), values
    )
    assert abs(z_right) < NOISE_FLOOR, z_right
    assert abs(z_wrong) < NOISE_FLOOR, z_wrong
    assert 1.0 / z_wrong**2 > 1e18


def test_the_truncating_basis_really_has_deleted_the_direction(truncating):
    """Otherwise the two tests below measure a projector against no residual."""
    scores = score_directions(
        rhino_bank.space(), rhino_bank.pipeline(), rhino_bank.state()
    )
    assert basis_fidelity(truncating, scores).residuals[TARGET] == pytest.approx(
        0.5618, rel=5e-3
    )


def test_the_correct_projector_has_zero_score_bias_however_badly_it_truncates(
    truncating,
):
    """(I - Pi) x is N^-1-orthogonal to the span, so the cross term vanishes.

    This basis has thrown away 56 % of the 21 cm signature. It still costs
    nothing in bias: measured 5.9e-12, which is roundoff. The truncation is paid
    for in width, not in centre -- sigma_1 is 1.584e-02 here against 1.310e-02
    at the operating point, a 21 % loss.
    """
    data, values = _noiseless()
    weight = rhino_bank.weight()
    z, width = _z_statistic(
        _correct_projector_log_likelihood(truncating, data, weight), values
    )
    assert abs(z) < NOISE_FLOOR, z
    assert width == pytest.approx(1.584e-02, rel=1e-2)


def test_the_unweighted_projector_does_not(truncating):
    """The genuine bias route, measured, with its own sqrt(N) crossing."""
    data, values = _noiseless()
    weight = rhino_bank.weight()
    z_right, _ = _z_statistic(
        _correct_projector_log_likelihood(truncating, data, weight), values
    )
    z_wrong, _ = _z_statistic(
        _unweighted_projector_log_likelihood(truncating, data, weight), values
    )
    assert abs(z_wrong) > 1e6 * max(abs(z_right), 1e-12), (
        f"z_right={z_right:.3e}, z_wrong={z_wrong:.3e}. If these are comparable "
        "the projector variants are not actually different -- check that the "
        "unweighted one really solves against `rows @ rows.T` and not the "
        "whitened Gram."
    )
    # Section 7: a coherent per-epoch bias grows as sqrt(N) against sigma_N.
    assert z_wrong == pytest.approx(3.317e-02, rel=1e-2)
    assert 1.0 / z_wrong**2 == pytest.approx(909.0, rel=2e-2)


def test_the_bias_vanishes_under_a_constant_sigma_which_is_why_the_fixture_is_not(
    truncating,
):
    """The proof that this file can fail.

    With a flat weight the unweighted projector IS the N^-1 projector, so the
    bias is identically zero -- measured 1.6e-12 against 3.3e-2 -- and every
    assertion above would hold for a reason that has nothing to do with the code
    under test. The same truncating rows are reused, so the *only* thing that
    changed is the metric.
    """
    data, values = _noiseless()
    flat = jnp.ones(rhino_bank.N_FREQ)
    basis = ReducedBasis(
        rows=truncating.rows,
        weight=flat,
        predict=rhino_bank.forward()[0],
        reference=data,
        support=rhino_bank.support(),
    )
    z_wrong, _ = _z_statistic(
        _unweighted_projector_log_likelihood(basis, data, flat), values
    )
    assert abs(z_wrong) < NOISE_FLOOR, z_wrong


def test_selecting_in_the_wrong_metric_costs_sensitivity_not_bias(bank):
    """The half of requirement 1 that IS about truncation, stated correctly.

    Both spans are unbiased, because the projector is N^-1-orthogonal either
    way. The cost is the width, and on this fixture it is small and real: 1.0035
    at n_S = 3, holding across three seeds at 1.0017-1.0035.

    It is a measurement and not a theorem. At n_S = 2 the ratio is 0.99976 on
    all three seeds -- the unweighted span happens to align slightly better with
    the trough there -- and from n_S = 4 the two agree to every digit, because
    four vectors span the tangent space whichever metric ordered them.
    """
    data, values = _noiseless()
    weight = rhino_bank.weight()
    predict, _ = rhino_bank.forward()

    right = ReducedBasis(
        rows=orthonormalise(select_svd(bank * weight, 3)) / weight,
        weight=weight,
        predict=predict,
        reference=data,
        support=rhino_bank.support(),
    )
    wrong = ReducedBasis(
        rows=orthonormalise(select_svd(bank, 3)),
        weight=weight,
        predict=predict,
        reference=data,
        support=rhino_bank.support(),
    )
    z_right, sigma_right = _z_statistic(
        _correct_projector_log_likelihood(right, data, weight), values
    )
    z_wrong, sigma_wrong = _z_statistic(
        _correct_projector_log_likelihood(wrong, data, weight), values
    )
    # Both unbiased: the projector is N^-1-orthogonal either way.
    assert abs(z_right) < NOISE_FLOOR and abs(z_wrong) < NOISE_FLOOR
    # The cost is the width, and it is a real cost.
    assert sigma_wrong > sigma_right
    assert sigma_wrong / sigma_right == pytest.approx(1.0035, rel=1e-3)
