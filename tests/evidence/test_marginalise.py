"""Integrating a named block out of a square-root information form.

Every number here is an ABSOLUTE log-density against a dense oracle. The
constant is the whole difficulty: Plan A shipped this same marginalisation with
`-sum(log(std))` missing, which is 0 when std == 1 and 27.5 nats when it is not,
and which no posterior's shape can show. Both `_joint` callers below therefore
use `std != 1`, and the two-route test at twenty-five nuisances is the one that
would have caught it.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compress import compress_linear
from rheplicant.inference.sqrtinfo import SqrtInfo, marginalise
from tests.evidence import rhino_bank

#: The whitened RHINO bank's numerical rank, measured. `build_reduced_basis`
#: refuses anything above it, so this is the dictionary size every T1 test in
#: this file uses -- and with two nuisance columns declared as extra directions
#: it is exactly four score rows plus two, with no room left for the bank.
BANK_RANK = 6


def _joint(design_x, design_p, data, sigma, prior_std, prior_mean=0.0):
    """The un-marginalised joint over (p, x), prior rows appended.

    The offset carries the prior's own normalisation, because that is the
    caller's to supply -- `marginalise` contributes only the integral.
    """
    n_p = design_p.shape[1]
    std = jnp.broadcast_to(jnp.asarray(prior_std), (n_p,))
    mean = jnp.broadcast_to(jnp.asarray(prior_mean), (n_p,))
    top = jnp.concatenate([design_p / sigma, design_x / sigma], axis=1)
    bottom = jnp.concatenate(
        [jnp.diag(1.0 / std), jnp.zeros((n_p, design_x.shape[1]))], axis=1
    )
    return SqrtInfo(
        factor=jnp.concatenate([top, bottom], axis=0),
        target=jnp.concatenate([data / sigma, mean / std]),
        offset=(
            -0.5 * len(data) * jnp.log(2 * jnp.pi * sigma**2)
            - jnp.sum(jnp.log(std))
            - 0.5 * n_p * jnp.log(2 * jnp.pi)
        ),
        names=("p", "x"),
        shapes=((n_p,), (design_x.shape[1],)),
    )


def _dense_marginal(design_x, design_p, data, sigma, prior_std, x):
    cov = np.diag(np.full(len(data), sigma**2)) + design_p @ np.diag(
        np.full(design_p.shape[1], prior_std**2)
    ) @ design_p.T
    resid = np.asarray(data) - np.asarray(design_x) @ np.asarray(x)
    sign, logdet = np.linalg.slogdet(cov)
    assert sign > 0
    return -0.5 * (
        resid @ np.linalg.solve(cov, resid) + logdet + len(data) * np.log(2 * np.pi)
    )


def _designs(key, n_data, n_x, n_p):
    ka, kb = jax.random.split(key)
    return jax.random.normal(ka, (n_data, n_x)), jax.random.normal(kb, (n_data, n_p))


def test_marginalise_matches_the_dense_gaussian_integral():
    design_x, design_p = _designs(jax.random.key(0), 40, 2, 3)
    data = jax.random.normal(jax.random.key(1), (40,))
    marginal = marginalise(_joint(design_x, design_p, data, 0.1, 0.7), ("p",))
    for seed in range(3):
        x = jax.random.normal(jax.random.key(seed + 5), (2,))
        assert float(marginal.log_prob({"x": x})) == pytest.approx(
            _dense_marginal(design_x, design_p, data, 0.1, 0.7, x), rel=1e-10
        )


def test_it_agrees_with_compress_linear_absolutely():
    """Two routes to the same constant. If they differ, one of them invented it."""
    design_x, design_p = _designs(jax.random.key(2), 40, 2, 3)
    data = jax.random.normal(jax.random.key(3), (40,))
    term = compress_linear(
        design={"x": design_x},
        nuisance_design={"p": design_p},
        nuisance_prior_std={"p": 0.7},
        observed=data,
        noise_std=0.1,
        shapes={"x": (2,)},
        nuisance_shapes={"p": (3,)},
        epoch_id="e0",
    )
    marginal = marginalise(_joint(design_x, design_p, data, 0.1, 0.7), ("p",))
    x = jnp.array([0.3, 0.9])
    assert float(marginal.log_prob({"x": x})) == pytest.approx(
        float(term({"x": x})), rel=1e-10
    )


def test_it_still_agrees_with_twenty_five_nuisances_at_std_three():
    """std = 1 makes sum(log(std)) exactly zero. This is the case that catches it."""
    design_x, design_p = _designs(jax.random.key(4), 80, 2, 25)
    data = jax.random.normal(jax.random.key(5), (80,))
    term = compress_linear(
        design={"x": design_x},
        nuisance_design={"p": design_p},
        nuisance_prior_std={"p": 3.0},
        observed=data,
        noise_std=0.1,
        shapes={"x": (2,)},
        nuisance_shapes={"p": (25,)},
        epoch_id="e0",
    )
    marginal = marginalise(_joint(design_x, design_p, data, 0.1, 3.0), ("p",))
    x = jnp.array([0.3, 0.9])
    assert float(marginal.log_prob({"x": x})) == pytest.approx(
        float(term({"x": x})), rel=1e-10
    )


def test_dropping_the_prior_normalisation_would_be_this_many_nats():
    """The size of the bug, said out loud, at both of the plan's two cases.

    `sum(log(std))` is what a marginalisation that forgets the prior's own
    normalisation is wrong by. It is measured here rather than asserted to be
    small, because at `std = 1` it is exactly zero and the test that would have
    caught it is the one that does not use unit priors.
    """
    assert 3 * float(np.log(1 / 0.7)) == pytest.approx(1.0700, abs=5e-4)
    assert 25 * float(np.log(3.0)) == pytest.approx(27.465, abs=5e-4)
    assert 3 * float(np.log(1.0)) == 0.0


def test_a_nonzero_prior_mean_travels():
    design_x, design_p = _designs(jax.random.key(6), 40, 2, 3)
    data = jax.random.normal(jax.random.key(7), (40,))
    shifted = marginalise(
        _joint(design_x, design_p, data, 0.1, 0.7, prior_mean=1.5), ("p",)
    )
    centred = marginalise(_joint(design_x, design_p, data, 0.1, 0.7), ("p",))
    x = jnp.array([0.3, 0.9])
    assert float(shifted.log_prob({"x": x})) != pytest.approx(
        float(centred.log_prob({"x": x})), rel=1e-6
    )


def test_marginalising_every_latent_leaves_a_zero_width_term():
    """Not a degenerate case to refuse: it is what T1 does at each theta."""
    design_x, design_p = _designs(jax.random.key(8), 20, 1, 2)
    data = jax.random.normal(jax.random.key(9), (20,))
    joint = _joint(design_x, design_p, data, 0.1, 0.7)
    everything = marginalise(joint, ("p", "x"))
    assert everything.names == ()
    assert everything.factor.shape == (0, 0)
    stepwise = marginalise(marginalise(joint, ("p",)), ("x",))
    assert float(everything.log_prob({})) == pytest.approx(
        float(stepwise.log_prob({})), rel=1e-10
    )


def test_marginalising_nothing_is_the_identity_the_no_nuisance_path_takes():
    """The nearest legitimate case to the two refusals below.

    A guard that over-refuses is its own bug, and an empty block is what the
    no-nuisance path passes when the two paths are one.
    """
    design_x, design_p = _designs(jax.random.key(20), 30, 2, 2)
    data = jax.random.normal(jax.random.key(21), (30,))
    joint = _joint(design_x, design_p, data, 0.1, 0.7)
    same = marginalise(joint, ())
    values = {"p": jnp.array([0.2, -0.4]), "x": jnp.array([0.3, 0.9])}
    assert same.names == ("p", "x")
    assert float(same.log_prob(values)) == pytest.approx(
        float(joint.log_prob(values)), rel=1e-10
    )


def test_an_unconstrained_block_is_refused_rather_than_returning_plus_infinity():
    """C3 made operational: an improper integral is not a likelihood."""
    joint = SqrtInfo(
        factor=jnp.array([[0.0, 1.0], [0.0, 2.0]]),
        target=jnp.zeros(2),
        offset=jnp.array(0.0),
        names=("p", "x"),
        shapes=((), ()),
    )
    with pytest.raises(StateValidationError, match="does not constrain"):
        marginalise(joint, ("p",))


def test_a_constrained_block_one_step_away_is_not_refused():
    """The same shape with a real prior row: the guard must not eat this."""
    joint = SqrtInfo(
        factor=jnp.array([[0.0, 1.0], [0.0, 2.0], [1.0 / 0.7, 0.0]]),
        target=jnp.zeros(3),
        offset=jnp.array(0.0),
        names=("p", "x"),
        shapes=((), ()),
    )
    assert np.isfinite(float(marginalise(joint, ("p",)).log_prob({"x": jnp.array(0.5)})))


def test_an_unknown_or_repeated_block_name_is_refused():
    joint = SqrtInfo(
        factor=jnp.eye(2),
        target=jnp.zeros(2),
        offset=jnp.array(0.0),
        names=("p", "x"),
        shapes=((), ()),
    )
    with pytest.raises(StateValidationError, match="not over"):
        marginalise(joint, ("q",))
    with pytest.raises(StateValidationError, match="twice"):
        marginalise(joint, ("p", "p"))


def _nuisance_columns():
    return jnp.stack(
        [jnp.ones(rhino_bank.N_FREQ), jnp.linspace(-1.0, 1.0, rhino_bank.N_FREQ)],
        axis=1,
    )


def _basis_with(nuisance, key):
    from rheplicant.inference.reduced_basis import build_reduced_basis

    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=rhino_bank.bank(key),
        n_basis=BANK_RANK,
        support=rhino_bank.support(),
        extra_directions=None if nuisance is None else nuisance.T,
    )


def test_t1_with_an_affine_nuisance_matches_the_analytic_marginal():
    """Section 4.2's pin, at the tier it was written for.

    An ABSOLUTE comparison against the dense ``N + A Sigma A^T`` Gaussian, at
    ``std = 0.5``. Built literally, T1 without the marginalisation stores
    ``p(d | theta, phi fixed)`` -- the same shape, the same gradient, an
    effective covariance of ``N`` rather than ``N + A Sigma A^T``, and every
    epoch of the campaign too tight by the same coherent factor. The gap is
    measured below rather than assumed small.
    """
    from rheplicant.inference.compress import compress_reduced_basis

    nuisance = _nuisance_columns()
    basis = _basis_with(nuisance, jax.random.key(10))
    data = rhino_bank.observed(jax.random.key(11))
    term = compress_reduced_basis(
        basis,
        observed=data,
        noise=rhino_bank.noise(),
        epoch_id="e0",
        nuisance_design={"offset": nuisance},
        nuisance_prior_std={"offset": 0.5},
        nuisance_shapes={"offset": (2,)},
    )
    tight = compress_reduced_basis(
        basis, observed=data, noise=rhino_bank.noise(), epoch_id="e0"
    )
    predict, values = rhino_bank.forward()

    sigma = np.asarray(rhino_bank.noise().std(basis.rows.T @ basis.c_ref))
    design = np.asarray(nuisance)
    covariance = np.diag(sigma**2) + 0.5**2 * design @ design.T
    model = np.asarray(
        basis.rows.T @ (basis.coefficients(values) - basis.c_ref)
        + basis.rows.T @ basis.c_ref
    )
    resid = np.asarray(data) - model
    sign, logdet = np.linalg.slogdet(covariance)
    assert sign > 0
    oracle = -0.5 * (
        resid @ np.linalg.solve(covariance, resid)
        + logdet
        + len(resid) * np.log(2 * np.pi)
    )
    assert float(term(values)) == pytest.approx(oracle, rel=1e-9)

    assert term.joint is not None
    assert term.nuisance_names == ("offset",)
    # Marginalising a nuisance can only widen the epoch's information, and it is
    # this much: two directions each carrying far more than the noise.
    assert float(term(values)) < float(tight(values))


def test_the_stored_joint_is_the_thing_the_marginal_came_from():
    """Section 4.2: store the un-marginalised block, or a mis-declaration is
    permanently unfalsifiable once the raw data is gone."""
    from rheplicant.inference.compress import compress_reduced_basis
    from rheplicant.inference.compressed import COEFFICIENTS

    nuisance = _nuisance_columns()
    basis = _basis_with(nuisance, jax.random.key(12))
    term = compress_reduced_basis(
        basis,
        observed=rhino_bank.observed(jax.random.key(13)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
        nuisance_design={"offset": nuisance},
        nuisance_prior_std={"offset": 0.5},
        nuisance_shapes={"offset": (2,)},
    )
    recomputed = marginalise(term.joint, term.nuisance_names)
    probe = {COEFFICIENTS: jnp.zeros(basis.n_basis)}
    assert float(recomputed.log_prob(probe)) == pytest.approx(
        float(term.info.log_prob(probe)), rel=1e-10
    )


def test_a_nuisance_column_outside_the_span_is_refused_not_quietly_projected():
    """4.2(b): a direction the basis cannot represent is one the marginalisation
    cannot remove, so it would come back as signal."""
    from rheplicant.inference.compress import compress_reduced_basis

    nuisance = _nuisance_columns()
    # Built WITHOUT the nuisance columns declared: the bank completes the span
    # with its own leading directions, which are not these.
    basis = _basis_with(None, jax.random.key(14))
    with pytest.raises(StateValidationError, match="outside"):
        compress_reduced_basis(
            basis,
            observed=rhino_bank.observed(jax.random.key(15)),
            noise=rhino_bank.noise(),
            epoch_id="e0",
            nuisance_design={"offset": nuisance},
            nuisance_prior_std={"offset": 0.5},
            nuisance_shapes={"offset": (2,)},
        )


def test_a_nuisance_without_a_prior_is_refused():
    """C3 again, one level up: the prior is part of the model here, not a knob."""
    from rheplicant.inference.compress import compress_reduced_basis

    nuisance = _nuisance_columns()
    basis = _basis_with(nuisance, jax.random.key(16))
    with pytest.raises(StateValidationError, match="nuisance_prior_std"):
        compress_reduced_basis(
            basis,
            observed=rhino_bank.observed(jax.random.key(17)),
            noise=rhino_bank.noise(),
            epoch_id="e0",
            nuisance_design={"offset": nuisance},
            nuisance_shapes={"offset": (2,)},
        )


def test_no_nuisance_leaves_no_joint_and_the_two_paths_are_one_term():
    from rheplicant.inference.compress import compress_reduced_basis

    basis = _basis_with(None, jax.random.key(18))
    term = compress_reduced_basis(
        basis,
        observed=rhino_bank.observed(jax.random.key(19)),
        noise=rhino_bank.noise(),
        epoch_id="e0",
    )
    assert term.joint is None
    assert term.nuisance_names == ()
