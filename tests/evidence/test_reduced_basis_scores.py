"""Section 5 requirement 3: every named latent's signature is in the basis."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference.reduced_basis import (
    ReducedBasis,
    orthonormalise,
    score_directions,
)
from tests.evidence import rhino_bank

#: Central-difference step, as a fraction of `max(|value|, 1)`. Measured, not
#: guessed: at 1e-5 the `t21_depth` column misses by up to 2.5e-9 in eight of
#: the 128 channels, because a ~1 K trough rides on a ~349 K foreground and the
#: difference of two evaluations loses ~1e-13 absolute, which a 2e-5 divisor
#: turns into ~5e-9. At 1e-4 no column violates, and the truncation error of
#: the most curved latent (`index`, 4e-8 absolute against values of order 77)
#: is still three orders inside the relative tolerance.
STEP = 1e-4


def _scores(**kwargs):
    return score_directions(
        rhino_bank.space(), rhino_bank.pipeline(), rhino_bank.state(), **kwargs
    )


def test_score_directions_match_a_central_difference():
    predict, values = rhino_bank.forward()
    scores = _scores()
    for name in values:
        step = STEP * max(abs(float(values[name])), 1.0)
        high = predict({**values, name: values[name] + step})
        low = predict({**values, name: values[name] - step})
        np.testing.assert_allclose(
            np.asarray(scores[name][0]),
            np.asarray((high - low) / (2 * step)),
            rtol=2e-5,
            atol=1e-9,
        )


def test_one_row_per_scalar_degree_of_freedom_in_declared_order():
    """Declared order, not jax's sorted rebuild -- see the note in the source.

    The fixture declares ("t21_depth", "index", "amplitude", "running"); jax
    flattens a dict in sorted order and rebuilds it that way, so a jacobian
    returned in its own order would be alphabetical and every downstream zip
    against a declared-order list would be wrong by a permutation.
    """
    scores = _scores()
    assert tuple(scores) == ("t21_depth", "index", "amplitude", "running")
    for rows in scores.values():
        assert rows.shape == (1, rhino_bank.N_FREQ)


def test_at_moves_the_point_and_a_nonlinear_score_moves_with_it():
    """`amplitude` enters linearly, so its own value cannot move any score.

    The move is made in `amplitude` rather than in `index`, which is what makes
    both halves true at once: the prediction is `A exp((beta + gamma L) L)`, so
    `d mu / d A` does not depend on `A` -- measured, exactly unchanged -- while
    `d mu / d beta` is proportional to it and moves by the full 30 % of the
    amplitude change. Moving `index` instead would shift BOTH by 6.9 %, since
    `d mu / d A` is a function of `beta`; the first assertion would then fail
    while claiming to state a linearity that does hold.
    """
    here = _scores()
    there = _scores(at={"amplitude": jnp.asarray(260.0)})
    np.testing.assert_allclose(
        np.asarray(here["amplitude"]), np.asarray(there["amplitude"]), rtol=1e-12
    )
    assert not np.allclose(
        np.asarray(here["index"]), np.asarray(there["index"]), rtol=1e-3
    )


def test_an_unknown_name_is_refused():
    with pytest.raises(ParameterSpaceError, match="not declared"):
        _scores(names=("t21_amplitude",))


def test_an_unknown_point_is_refused():
    """The nearest thing to the case above that the same guard must also catch."""
    with pytest.raises(ParameterSpaceError, match="not declared"):
        _scores(at={"t21_amplitude": jnp.asarray(1.0)})


def test_a_seeded_score_is_in_the_span_exactly():
    """Gram-Schmidt nests the spans, so seeding first is a guarantee."""
    weight = rhino_bank.weight()
    scores = _scores()
    seed = scores["t21_depth"]
    filler = jax.random.normal(jax.random.key(0), (4, rhino_bank.N_FREQ))
    # Orthonormalise in the whitened metric and divide back out, which is legal
    # here only because this reference weight is strictly positive (a radiometer
    # sigma tracks a foreground that never reaches zero in this band).
    rows = orthonormalise(jnp.concatenate([seed, filler], axis=0) * weight) / weight
    basis = ReducedBasis(
        rows=rows,
        weight=weight,
        predict=rhino_bank.forward()[0],
        reference=jnp.zeros(rhino_bank.N_FREQ),
        seeded=("t21_depth",),
    )
    assert float(basis.residual_fraction(scores["t21_depth"][0])) < 1e-12


def test_an_unseeded_science_direction_is_what_this_repairs():
    """The same four fillers without the seed do not contain the trough.

    Without this the test above pins nothing: if a span of random directions
    already held the score direction, `< 1e-12` would look like a strong claim
    about a weak one. Measured on this draw: r_t21 = 0.9921 unseeded against
    9.1e-17 seeded. The band is 0.5 rather than 0.99 because those digits are a
    property of one RNG seed -- four random directions in 128 dimensions retain
    of order sqrt(4/128) = 0.18 of a generic direction, so any seed lands far
    above 0.5, and a basis that had quietly kept the seed lands at 1e-16.
    """
    weight = rhino_bank.weight()
    scores = _scores()
    filler = jax.random.normal(jax.random.key(0), (4, rhino_bank.N_FREQ))
    rows = orthonormalise(filler * weight) / weight
    basis = ReducedBasis(
        rows=rows,
        weight=weight,
        predict=rhino_bank.forward()[0],
        reference=jnp.zeros(rhino_bank.N_FREQ),
    )
    assert float(basis.residual_fraction(scores["t21_depth"][0])) > 0.5
