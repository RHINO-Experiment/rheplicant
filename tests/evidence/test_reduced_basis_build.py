"""Selection, numerical rank, and the reason the 21 cm direction goes missing.

The pre-planning probe measured this table on a RHINO-like bank (plain SVD,
residual fraction of the t21 score direction):

    n_S      3        5        8       13
    r_t21    0.3147   0.0289   0.0040  0.0000

**This fixture is not that bank, and its numbers are its own.** Measured here,
over `jax.random.key(11)` and `key(12)` and over 400 and 2000 draws:

    n_S      1       2       3       4         5        6        8        13
    r_t21    0.767   0.572   0.562   1.7e-4    8e-6     1.5e-7   ~1e-12   ~2e-14

The deletion is worse than the probe's at n_S = 3 -- 0.56 rather than 0.31 --
and the recovery is at n_S = 4 rather than 13. Both differences have the same
cause: this model has exactly four latents and enters near-linearly, so the
whitened bank is one mean direction plus a four-dimensional tangent space plus
curvature, and its numerical rank is 6, not the 13-38 the spec quotes for a
richer bank. Any n_S >= 4 must therefore contain the trough whatever the SVD
sorts it behind; at n_S = 3 there is nowhere for it to hide.

So the headline claim is pinned at n_S = 3, and the companion assertion below
says where it stops being true, so the first reading cannot be taken for a
general law. The bands are wide enough to survive a different RNG draw -- the
two seeds probed agree to two digits at n_S = 3 and to within a factor of two
below it -- and narrow enough that a basis which dropped a direction cannot
pass.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.reduced_basis import (
    ReducedBasis,
    build_reduced_basis,
    numerical_rank,
    orthonormalise,
    score_directions,
    select_greedy,
    select_svd,
)
from tests.evidence import rhino_bank

#: The whitened bank's numerical rank, measured: stable at 6 across both seeds
#: and both bank sizes probed. One mean direction, four tangent directions, and
#: one curvature direction still above the sqrt(eps) cut -- the sixth relative
#: singular value is 7.9e-8 against a cut of 1.49e-8, and the seventh is 5.4e-10.
BANK_RANK = 6


@pytest.fixture(scope="module")
def bank():
    return rhino_bank.bank(jax.random.key(11), n_draws=400)


@pytest.fixture(scope="module")
def scores():
    return score_directions(
        rhino_bank.space(), rhino_bank.pipeline(), rhino_bank.state()
    )


def _build(bank, n_basis, **kwargs):
    return build_reduced_basis(
        rhino_bank.space(),
        rhino_bank.pipeline(),
        rhino_bank.state(),
        noise=rhino_bank.noise(),
        bank=bank,
        n_basis=n_basis,
        **kwargs,
    )


def _wrap(rows, weight, predict, reference, seeded=()):
    """`rows` are RAW. Dividing the whitened rows back out is legitimate only
    because RadiometerNoise on a positive prediction leaves no zero weight."""
    return ReducedBasis(
        rows=rows,
        weight=weight,
        predict=predict,
        reference=reference,
        seeded=seeded,
    )


def _plain_svd_basis(bank, n_basis):
    """A basis built the way section 5 says everyone builds one, and warns about."""
    weight = rhino_bank.weight()
    predict, values = rhino_bank.forward()
    return _wrap(
        orthonormalise(select_svd(bank * weight, n_basis)) / weight,
        weight,
        predict,
        predict(values),
    )


def test_svd_selection_orders_the_bank_by_prior_induced_amplitude(bank):
    """The mechanism behind the deletion, stated as a fact about the numbers."""
    singular = jnp.linalg.svd(bank * rhino_bank.weight(), compute_uv=False)
    # Three or four orders between the foreground's leading mode and the trough's
    # amplitude is exactly why the science direction sorts last. Measured
    # s_0/s_2 = 2.1e3 here, and s_3 -- the trough -- is 7.0e3 below s_0.
    assert float(singular[0] / singular[2]) > 100.0


def test_plain_svd_at_three_vectors_loses_the_t21_direction(bank, scores):
    """The headline. Pinned at n_S = 3 on purpose -- see the module docstring."""
    residual = float(_plain_svd_basis(bank, 3).residual_fraction(scores["t21_depth"][0]))
    assert residual > 0.15, (
        f"r_t21 = {residual}; this fixture measures 0.562, and the pre-planning "
        "probe measured 0.3147 on its own bank. A value near zero means the bank "
        "is not RHINO-like -- check that the trough amplitude is three orders "
        "below the foreground, because with a flat hierarchy there is nothing "
        "for the SVD to sort last."
    )


def test_score_seeding_repairs_it_completely_at_the_same_three_vectors(bank, scores):
    """Like for like, at the n_S the claim above is pinned at.

    One seeded row and two bank rows against three bank rows -- the same storage,
    a different choice of what to spend it on. Measured 1.5e-16 seeded against
    0.562 unseeded: a complete repair, not an improvement.

    Only ``t21_depth`` is named, because seeding all four latents needs four rows
    and the comparison would stop being at the same n_S. That the argument exists
    at all is the point of ``names=``: a campaign may declare more latents than
    it can afford rows for, and which ones it protects is then a decision it
    makes rather than one truncation makes for it.
    """
    basis = _build(bank, 3, names=("t21_depth",))
    assert basis.seeded == ("t21_depth",)
    assert basis.n_basis == 3
    assert float(basis.residual_fraction(scores["t21_depth"][0])) < 1e-10


def test_seeding_every_latent_costs_one_row_each_and_keeps_declared_order(bank, scores):
    """Four latents, four seeded rows, in the order the space DECLARES them.

    Not the order jax rebuilds a dict in. The fixture declares
    ("t21_depth", "index", "amplitude", "running") and sorted order is
    ("amplitude", "index", "running", "t21_depth"), so a builder that read the
    jacobian's own keys would report a `seeded` tuple that is right by accident
    only for an alphabetical model.
    """
    basis = _build(bank, 4)
    assert basis.seeded == ("t21_depth", "index", "amplitude", "running")
    for name in basis.seeded:
        assert float(basis.residual_fraction(scores[name][0])) < 1e-10, name


def test_four_vectors_recover_it_unseeded_which_is_why_three_is_pinned(bank, scores):
    """The companion assertion. Without it the headline looks like a general law.

    The probe's bank needed 13; this one needs 4, because it varies in only four
    parameters and a plain SVD given four vectors cannot avoid spanning them. A
    test of the headline written at this fixture's operating point would assert
    nothing, and would go on passing after the seeding was deleted.
    """
    residual_at_four = float(
        _plain_svd_basis(bank, 4).residual_fraction(scores["t21_depth"][0])
    )
    assert residual_at_four < 1e-2
    residual_at_thirteen = float(
        _plain_svd_basis(bank, 13).residual_fraction(scores["t21_depth"][0])
    )
    assert residual_at_thirteen < 1e-3


def test_greedy_selection_spans_the_bank_to_a_declared_tolerance(bank):
    """Asking for eight and getting six is the rank, not a silent failure.

    ``orthonormalise`` drops the candidates whose residual falls under the
    sqrt(eps) cut, so a greedy request above the bank's numerical rank returns a
    basis of exactly that rank. The six that survive reproduce every draw probed
    to 2.2e-9; the band is 1e-6 because a basis that had dropped a real direction
    lands at 1e-3 or worse.
    """
    weight = rhino_bank.weight()
    predict, _ = rhino_bank.forward()
    chosen = select_greedy(bank * weight, 8)
    basis = _wrap(
        orthonormalise(chosen) / weight, weight, predict, jnp.zeros(rhino_bank.N_FREQ)
    )
    assert basis.n_basis == BANK_RANK
    worst = max(float(basis.residual_fraction(draw)) for draw in bank[:20])
    assert worst < 1e-6


def test_selection_and_basis_are_different_objects(bank):
    """Section 5 requirement 2, in executable form.

    The tolerance is 1e-8 and not machine precision, and that is a measurement
    rather than a concession: the sixth candidate is recovered from a residual
    7.9e-8 of the leading row norm, so its orthogonality is bounded by
    eps / 7.9e-8 = 2.8e-9. Measured 1.7e-10 on this seed and 6.1e-10 on the
    other. Against a candidate Gram matrix whose condition number is 3.5e13,
    1e-8 is still the difference between a basis and a pile of snapshots.
    """
    candidates = select_greedy(bank * rhino_bank.weight(), BANK_RANK)
    gram = candidates @ candidates.T
    assert float(jnp.linalg.cond(gram)) > 1e3  # candidates: not a basis
    rows = orthonormalise(candidates)
    np.testing.assert_allclose(np.asarray(rows @ rows.T), np.eye(len(rows)), atol=1e-8)


def test_numerical_rank_is_the_sqrt_eps_cut():
    values = jnp.array([1.0, 1e-3, 1e-6, 1e-12, 1e-20])
    matrix = jnp.diag(values) @ jax.random.normal(jax.random.key(0), (5, 40))
    # sqrt(eps) for float64 is 1.49e-8: the first three survive, the last two do not.
    assert numerical_rank(matrix) == 3


def test_the_whitened_rhino_bank_has_the_rank_its_four_latents_allow(bank):
    """Six, not the spec's 13-38 -- and the difference is the fixture, not a bug.

    A bank of predictions from a four-parameter model spans a mean direction,
    four tangent directions and whatever curvature stays above sqrt(eps). The
    spec's 13-38 is for a bank with more parameters in it. Pinning the spec's
    range here would have been pinning someone else's fixture.
    """
    rank = numerical_rank(bank * rhino_bank.weight())
    assert rank == BANK_RANK, (
        f"rank {rank}; this bank measured 6 over two seeds and two bank sizes. "
        "A larger value means the prior draws now move in more directions -- "
        "check n_draws and PRIOR_STD before widening this."
    )


def test_a_basis_above_the_numerical_rank_is_refused(bank):
    rank = numerical_rank(bank * rhino_bank.weight())
    with pytest.raises(StateValidationError, match="numerical rank"):
        _build(bank, rank + 1)


def test_a_basis_at_the_numerical_rank_is_not_refused(bank):
    """The nearest legitimate case to the refusal above. A guard that over-refuses
    is its own bug, and `> rank` versus `>= rank` is a one-character version of it."""
    assert _build(bank, numerical_rank(bank * rhino_bank.weight())).n_basis == BANK_RANK


def test_asking_for_fewer_vectors_than_there_are_scores_is_refused(bank):
    with pytest.raises(StateValidationError, match="score directions"):
        _build(bank, 2)


def test_an_unknown_selection_method_is_refused(bank):
    with pytest.raises(StateValidationError, match="'svd' or 'greedy'"):
        _build(bank, 5, method="qr")


def test_the_built_basis_reports_its_conditioning(bank):
    basis = _build(bank, BANK_RANK)
    assert basis.n_basis == BANK_RANK
    # Orthonormalised, so the Gram matrix is the identity and kappa is exactly 1.
    # The band is 10 rather than 1 + eps because what is being asserted is that
    # the rows were orthonormalised at all, not the roundoff of the measurement.
    assert basis.condition() < 10.0


def test_the_greedy_route_builds_the_same_span_as_the_svd_route(bank, scores):
    """`method=` chooses how the remainder is picked, not what is guaranteed.

    Both routes seed the same four score directions first, so both must hold
    every latent's signature exactly whatever the bank completion does.
    """
    basis = _build(bank, 5, method="greedy")
    assert basis.n_basis == 5
    for name in basis.seeded:
        assert float(basis.residual_fraction(scores[name][0])) < 1e-10, name


def test_the_reference_point_is_carried_as_arrays_not_as_treedef(bank, recwarn):
    """Arrays belong in the leaves; equinox says so and this repo has the scar.

    `reference_values` was declared static, which puts a dict of jax arrays into
    the TREEDEF, where `dict.__eq__` decides treedef equality. For this fixture's
    scalar latents that silently works; for a latent of shape (4,) -- which
    `score_directions` documents support for -- comparing two treedefs raises
    "the truth value of an array with more than one element is ambiguous",
    measured. `tests/core/test_basis.py` pins the identical rule for `Bind.fn`.
    """
    basis = _build(bank, 4)
    assert set(basis.reference_values) == set(rhino_bank.TRUTH)
    assert not [w for w in recwarn if "static" in str(w.message)]
    leaves = jax.tree_util.tree_leaves(basis)
    assert any(leaf is basis.reference_values["t21_depth"] for leaf in leaves)
