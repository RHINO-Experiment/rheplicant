"""Section 7: the gradient of the compression error, not its magnitude.

v1 of the spec bounded ``|delta log L|``, which is the wrong functional twice
over. A constant offset has exactly zero effect on a posterior, so a bound on
the magnitude refuses answers that are perfectly good; and an arbitrarily small
theta-dependent tilt has unbounded effect, so no bound on the magnitude refuses
the answers that are bad. What the posterior sees is the theta-**gradient**.

The second half is the part that decides how the numbers below are read. One
basis, one instrument model and one estimator serve every epoch, so whatever
tilt the compression introduces is the *same* tilt each time. Summed over N
epochs the gradient grows as N while the Fisher does too, and the induced bias
``Delta_theta = -F^-1 grad(delta)`` is therefore **N-independent** -- while
``sigma_N`` falls as ``N^-1/2``. The ratio the campaign is actually exposed to
grows as ``sqrt(N)``, and that is measured here rather than asserted:
``test_the_ratio_grows_as_sqrt_n_for_a_coherent_error`` reads 1.571e-11 over
four epochs and 6.286e-11 over sixty-four, a factor 4.001 against the 4.000
that ``sqrt(64/4)`` predicts.

**Where the coherent tilt comes from, on this fixture.** Not from the
truncation: the basis is seeded, so every score direction is in the span and
the truncation's contribution to the gradient cancels to roundoff. It comes
from deviation 4 -- the coefficient map ``c(theta)`` is built once in the
reference metric while each epoch stores its statistics in its own. Here every
epoch freezes sigma at the same reference prediction, so that mismatch is
identical epoch to epoch, which is exactly what "coherent" means. Measured
against an **unseeded** ``n_S=3`` basis, where the truncation is large
(``r_t21 = 0.562``) but noise-driven, the ratio is a hundred times bigger and
does **not** grow: 0.0062, 0.0250, 0.0104 at N = 4, 16, 64. Bigger and
incoherent is the case a sqrt-N law does not describe, and the shipped
operating point is the one it does.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compress import compress_reduced_basis
from rheplicant.inference.compressed import ReducedBasisLikelihood
from rheplicant.inference.factorize import Factorization
from rheplicant.inference.memory import BayesMemory
from rheplicant.inference.noise import FlaggedNoise
from rheplicant.inference.reduced_basis import ReducedBasis, build_reduced_basis
from tests.evidence import rhino_bank

#: The whitened bank's numerical rank on this fixture, pinned in
#: `test_reduced_basis_build.py`. `build_reduced_basis` refuses anything above
#: it, so this is the largest honest dictionary rather than a chosen one.
N_BASIS = 6

#: Flatten order, which is NOT the order `rhino_bank` declares its latents in.
FLAT = ("amplitude", "index", "running", "t21_depth")


def _basis(n_basis=N_BASIS, seed_scores=True, key=50):
    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(key)),
        n_basis=n_basis,
        support=rhino_bank.support(),
        seed_scores=seed_scores,
    )


def _term(basis, index, **overrides):
    keywords = {
        "observed": rhino_bank.observed(jax.random.fold_in(jax.random.key(51), index)),
        "noise": rhino_bank.noise(),
        "epoch_id": f"n{index}",
        **overrides,
    }
    return compress_reduced_basis(basis, **keywords)


def _campaign(basis, n_epochs):
    memory = BayesMemory(Factorization(rhino_bank.space()))
    for index in range(n_epochs):
        memory = memory.remember(_term(basis, index))
    return memory


def test_the_stored_gradient_is_the_gradient_of_the_fidelity_residual():
    """Against the residual written out by hand, not against itself."""
    basis, data = _basis(), rhino_bank.observed(jax.random.key(52))
    term = compress_reduced_basis(
        basis, observed=data, noise=rhino_bank.noise(), epoch_id="e0"
    )
    predict, values = rhino_bank.forward()
    sigma = rhino_bank.noise().std(basis.rows.T @ basis.c_ref)

    def residual(chosen):
        probe = {**values, **chosen}
        raw = -0.5 * jnp.sum(((data - predict(probe)) / sigma) ** 2) - 0.5 * jnp.sum(
            jnp.log(2 * jnp.pi * sigma**2)
        )
        return term(probe) - raw

    direct = jax.grad(residual)(dict(values))
    ordered = jnp.concatenate([jnp.ravel(direct[name]) for name in sorted(values)])
    np.testing.assert_allclose(
        np.asarray(term.bias_gradient), np.asarray(ordered), rtol=1e-6, atol=1e-12
    )


def test_the_gradient_is_stored_in_flatten_order_not_declared_order():
    """The permutation Plan A's fisher() got wrong, one layer along.

    `rhino_bank` declares ("t21_depth", "index", "amplitude", "running") on
    purpose. A gradient stored in that order and combined with a Fisher in
    flatten order is wrong by a permutation that is the identity exactly when
    the latents happen to be alphabetical.
    """
    term = compress_reduced_basis(
        _basis(),
        observed=rhino_bank.observed(jax.random.key(53)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
    )
    assert term.bias_names == FLAT
    assert term.bias_names != rhino_bank.space().names


def test_audit_reports_the_ratio_per_named_direction():
    report = _campaign(_basis(), 8).audit()
    assert set(report["bias_over_sigma"]) == set(rhino_bank.TRUTH)
    assert all(np.isfinite(v) for v in report["bias_over_sigma"].values())
    assert report["unconstrained"] == ()


def test_the_ratio_grows_as_sqrt_n_for_a_coherent_error():
    """Section 7's whole point, measured rather than asserted.

    4 -> 64 epochs is a 16x campaign, so a coherent per-epoch tilt must show
    sqrt(16) = 4x. Growth near 1x would mean the stored gradients are
    cancelling, i.e. the error is being treated as noise -- and section 9 is
    explicit that a deterministic common-mode error contributes no variance and
    is invisible to every shrinkage diagnostic.
    """
    basis = _basis(4)
    small = _campaign(basis, 4).audit()["bias_over_sigma"]["t21_depth"]
    large = _campaign(basis, 64).audit()["bias_over_sigma"]["t21_depth"]
    assert large / small == pytest.approx(4.0, rel=0.05), (
        f"{small} -> {large} over a 16x campaign; sqrt(N) predicts 4.0x."
    )
    # And the absolute scale, so a regression that keeps the ratio but moves
    # the scale is visible. As a BAND, not a pin: measured 1.571e-11 on arm64
    # macOS and 1.639e-12 on x86_64 Linux, a factor of ten. These are
    # bias-over-sigma at the 1e-11 level, which is to say a coherent error
    # already indistinguishable from zero, so its absolute size is decided by
    # where the arithmetic rounds and not by the campaign. The RATIO above is
    # the section's claim and holds on both platforms to well inside 5 %.
    #
    # The band still does the job the pin was written for. What it is guarding
    # against is a regression that moves the scale by orders of magnitude --
    # a bias at 1e-6 would be a real one, and no platform spread reaches that.
    assert 1e-13 < small < 1e-9, small
    assert 1e-13 < large < 1e-9, large


def test_a_bigger_incoherent_error_does_not_show_the_sqrt_n_law():
    """The conditional Task 6 settled, stated where the budget is read.

    An unseeded n_S=3 basis loses the t21 score direction (r_t21 = 0.562), so
    its compression error is a hundred times larger -- and driven by each
    epoch's own noise projected onto the missing direction rather than by
    anything common. That sum grows as sqrt(N), the Fisher grows as N, and the
    ratio is flat-to-erratic instead of rising. Bigger is not the same as
    coherent, and only the coherent part is what section 7 budgets.
    """
    basis = _basis(3, seed_scores=False)
    ratios = [
        _campaign(basis, n).audit()["bias_over_sigma"]["t21_depth"]
        for n in (4, 16, 64)
    ]
    assert min(ratios) > 1e-4, ratios
    assert max(ratios) / min(ratios) < 5.0, ratios
    assert not (ratios[0] < ratios[1] < ratios[2]), ratios


def test_the_reported_ratio_is_the_marginalised_one_and_not_the_raw_gradient():
    """Why the report is ``|F^-1 grad| / sigma_N`` and not ``|grad|``.

    Against the formula written out by hand, on a fixture whose bias is a real
    bias: an unseeded ``n_basis=4`` basis, whose smallest reported ratio is
    ~4e-5 rather than the ~5e-12 a seeded basis gives.
    """
    memory = _campaign(_basis(n_basis=4, seed_scores=False), 8)
    reported = memory.audit()["bias_over_sigma"]

    _, names, spans, _, _ = memory._flatten_layout()
    # `at=` is the storage origin, which is what `audit()` defaults to and
    # what `_theta_fisher` refuses to guess: the gradients were taken there, so
    # a Fisher taken anywhere else makes the ratio a quotient of two different
    # linearisations.
    at = dict(memory.basis.reference_values)
    covariance = np.asarray(
        jnp.linalg.pinv(memory._permuted(memory._theta_fisher(at)))
    )
    gradient = np.sum(
        np.stack([np.asarray(term.bias_gradient) for term in memory.archive]), axis=0
    )
    bias = covariance @ gradient
    width = np.sqrt(np.diag(covariance))

    assert min(reported.values()) > 1e-6, reported
    for name, (start, stop) in zip(names, spans, strict=True):
        by_hand = float(np.max(np.abs(bias[start:stop]) / width[start:stop]))
        assert reported[name] == pytest.approx(by_hand, rel=1e-9), name


def test_the_marginalised_ratio_is_invariant_to_units_and_the_raw_gradient_is_not():
    """The property that makes one of them a scientific quantity.

    Rescale latent *i* by ``c`` -- the same model, different units. The stored
    gradient scales as ``c`` (it is a derivative with respect to whatever the
    latent carries), so ``|grad|`` reorders and can be made to name any
    direction as the worst by choosing units. The reported ratio does not move
    at all: the bias picks up ``c`` from ``F^-1`` and ``sigma_N`` picks up the
    same ``c`` from ``sqrt(diag(F^-1))``, and they cancel.

    Done on the campaign's own numbers rather than by rebuilding it in new
    units, so the arithmetic is the thing under test and not the fixture.

    **This replaced an assertion that a particular ORDERING was reversed
    (2026-08-28, D49), and the reason is worth keeping.** That test read
    ``raw["running"] > 3 * raw["t21_depth"]`` and
    ``ratios["t21_depth"] > 3 * ratios["running"]`` on a SEEDED basis, where
    both reported ratios are ~5e-12 -- the regime the sibling test above
    already calls "a coherent error already indistinguishable from zero, so its
    absolute size is decided by where the arithmetic rounds". A ratio between
    two such numbers is not a measurement, and CI (linux/x86-64) read 0.70
    where this machine reads 3.97.

    Swept over the fixtures where the bias IS real, the reversal turns out not
    to be a property of the method at all -- it follows the random key::

        n_basis  key   raw r/t   mrg t/r   min |mrg|
              5   50      3.09      2.68   3.468e-06   <- reversed
              5    7      7.33      0.81   7.183e-07   <- both name running
              5  123      0.56      9.47   2.694e-06   <- both name t21_depth
              6    7     57.76      0.59   1.025e-08   <- both name running

    So the old assertion was true of one draw in the roundoff regime, and the
    thing it was reaching for -- that the two rank directions differently, and
    only one of the two rankings means anything -- is the invariance below.
    """
    memory = _campaign(_basis(n_basis=4, seed_scores=False), 8)
    _, names, spans, _, _ = memory._flatten_layout()
    # `at=` is the storage origin, which is what `audit()` defaults to and
    # what `_theta_fisher` refuses to guess: the gradients were taken there, so
    # a Fisher taken anywhere else makes the ratio a quotient of two different
    # linearisations.
    at = dict(memory.basis.reference_values)
    covariance = np.asarray(
        jnp.linalg.pinv(memory._permuted(memory._theta_fisher(at)))
    )
    gradient = np.sum(
        np.stack([np.asarray(term.bias_gradient) for term in memory.archive]), axis=0
    )

    def report(cov, grad):
        bias, width = cov @ grad, np.sqrt(np.diag(cov))
        return {
            name: float(np.max(np.abs(bias[a:b]) / width[a:b]))
            for name, (a, b) in zip(names, spans, strict=True)
        }

    def raw(grad):
        return {
            name: float(np.max(np.abs(grad[a:b])))
            for name, (a, b) in zip(names, spans, strict=True)
        }

    raw_before, before = raw(gradient), report(covariance, gradient)

    # Rescale a direction the raw gradient does NOT currently name as worst,
    # by enough that it must. Chosen from the measurement rather than written
    # in: pinning "t21_depth" here would make the test's meaning depend on the
    # fixture's argmax, which is the kind of accident this test replaced.
    worst = max(raw_before, key=raw_before.__getitem__)
    promoted = min(raw_before, key=raw_before.__getitem__)
    assert promoted != worst, raw_before

    # theta_i -> theta_i / c_i: the same model in different units. The Fisher
    # transforms as D F D and the gradient as D g, with D = diag(c).
    scales = np.ones(covariance.shape[0])
    start, stop = dict(zip(names, spans, strict=True))[promoted]
    scales[start:stop] = 1e6
    diagonal = np.diag(scales)
    rescaled_cov = np.linalg.pinv(diagonal @ np.linalg.pinv(covariance) @ diagonal)
    rescaled_grad = diagonal @ gradient

    after = report(rescaled_cov, rescaled_grad)
    for name in names:
        assert after[name] == pytest.approx(before[name], rel=1e-6), (
            f"{name} moved under a pure change of units: {before} -> {after}"
        )
    assert max(after, key=after.__getitem__) == max(before, key=before.__getitem__)

    # And the raw gradient's answer changes completely: the direction it named
    # as least compromised is now the one it names as worst. A factor of 1e6,
    # asserted as an argmax flip rather than as a margin -- a large effect, not
    # a small difference, which is the failure mode this file has been bitten
    # by twice.
    raw_after = raw(rescaled_grad)
    assert max(raw_after, key=raw_after.__getitem__) == promoted, (
        f"the rescaling did not change the raw gradient's answer: "
        f"{raw_before} -> {raw_after}"
    )
    assert raw_after[worst] == pytest.approx(raw_before[worst], rel=1e-9)


def test_audit_refuses_above_a_declared_ratio():
    with pytest.raises(StateValidationError, match="exceeds the declared budget"):
        _campaign(_basis(4), 64).audit(bias_tolerance=1e-12)


def test_a_tolerance_a_good_basis_meets_does_not_fire():
    """The nearest legitimate case: a guard that over-refuses is its own bug."""
    report = _campaign(_basis(), 8).audit(bias_tolerance=10.0)
    assert max(report["bias_over_sigma"].values()) < 1e-9


def test_an_unconstrained_direction_is_named_rather_than_refused():
    """A night lost entirely to RFI constrains nothing, and 0/0 is not a failure.

    Every channel flagged is `sigma = inf` everywhere, so the epoch's stored
    factor is exactly zero, the pulled-back Fisher is exactly zero, and the
    pseudo-inverse gives every direction a width of exactly zero. Reporting
    that as a bias failure would refuse a campaign for having lost a night.
    """
    basis = _basis()
    dark = compress_reduced_basis(
        basis,
        observed=rhino_bank.observed(jax.random.key(54)),
        noise=FlaggedNoise(
            rhino_bank.noise(), jnp.ones((rhino_bank.N_FREQ,), dtype=bool)
        ),
        epoch_id="dark",
    )
    assert dark.n_observed == 0
    report = BayesMemory(Factorization(rhino_bank.space())).remember(dark).audit(
        bias_tolerance=1e-30
    )
    assert report["unconstrained"] == FLAT
    assert report["bias_over_sigma"] == {}


def test_a_memory_with_no_reduced_terms_still_audits():
    report = BayesMemory(Factorization(rhino_bank.space())).audit()
    assert report["n_epochs"] == 0
    assert report["bias_over_sigma"] == {}
    assert report["unconstrained"] == ()


def test_an_epoch_with_a_nuisance_block_gets_a_gradient_too():
    """The oracle there is the phi-MARGINAL raw density, not the raw density.

    A term whose nuisance was integrated out is a function of theta alone, and
    comparing it against a raw density that still carries phi would measure the
    marginalisation rather than the compression. Without this the whole
    nuisance path would silently store no gradient at all, and a campaign of
    such epochs would report an empty bias budget and read as clean.
    """
    columns = jnp.stack(
        [jnp.ones((rhino_bank.N_FREQ,)), jnp.linspace(-1.0, 1.0, rhino_bank.N_FREQ)],
        axis=1,
    )
    basis = build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(jax.random.key(55)),
        n_basis=N_BASIS,
        support=rhino_bank.support(),
        extra_directions=columns.T,
    )
    term = compress_reduced_basis(
        basis,
        observed=rhino_bank.observed(jax.random.key(56)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
        nuisance_design={"offset": columns},
        nuisance_prior_std={"offset": 1.0},
    )
    assert term.bias_names == FLAT
    assert np.all(np.isfinite(np.asarray(term.bias_gradient)))
    assert float(np.max(np.abs(np.asarray(term.bias_gradient)))) > 0.0


def test_a_hand_built_basis_is_refused_rather_than_given_a_gradient_about_nothing():
    """No reference_values means no storage origin to differentiate at."""
    predict, values = rhino_bank.forward()
    reference = predict(values)
    hand = ReducedBasis(
        rows=jnp.stack([reference, reference * jnp.linspace(0.5, 1.5, len(reference))]),
        weight=rhino_bank.weight(),
        predict=predict,
        reference=reference,
        support=rhino_bank.support(),
    )
    with pytest.raises(StateValidationError, match="records no reference_values"):
        compress_reduced_basis(
            hand,
            observed=rhino_bank.observed(jax.random.key(57)),
            noise=rhino_bank.noise(),
            epoch_id="e0",
        )


def test_fisher_needs_a_point_once_the_memory_holds_a_reduced_term():
    """`c(theta)` is nonlinear, so there is no single Jacobian to default to."""
    memory = _campaign(_basis(), 2)
    with pytest.raises(StateValidationError, match="Pass at="):
        memory.fisher()
    named = memory.fisher(at=dict(rhino_bank.forward()[1]))
    assert named.names == FLAT
    assert np.all(np.isfinite(np.asarray(named.matrix)))


def test_a_gradient_over_different_names_is_refused_rather_than_summed():
    """A permutation or a truncation here is silent: the shapes still match."""
    memory = _campaign(_basis(), 1)
    term = memory.archive[0]
    mangled = ReducedBasisLikelihood(
        basis=term.basis,
        info=term.info,
        joint=term.joint,
        epoch_id="n1",
        n_observed=term.n_observed,
        support=term.support,
        bias_gradient=term.bias_gradient,
        bias_names=tuple(reversed(FLAT)),
    )
    with pytest.raises(StateValidationError, match="bias_names"):
        memory.remember(mangled).audit()


def test_the_gradient_survives_a_round_trip_through_the_term():
    """`bias_gradient` is a dynamic leaf, so equinox moves it like any array."""
    term = _term(_basis(), 0)
    copied = eqx.tree_at(lambda t: t.bias_gradient, term, term.bias_gradient * 1.0)
    np.testing.assert_array_equal(
        np.asarray(copied.bias_gradient), np.asarray(term.bias_gradient)
    )
    assert copied.bias_names == term.bias_names
