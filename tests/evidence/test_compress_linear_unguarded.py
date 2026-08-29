"""Four behaviours of ``compress_linear`` that nothing was watching.

Each of these was found the same way: mutate the source, run the whole
``tests/evidence`` directory, and see whether anything goes red. All four came
back **542 passed, exit 0** — the mutation was invisible.

============================================  =========================
mutation                                      result before this file
============================================  =========================
drop the ``offset_prediction`` term           542 passed
zero the ``nuisance_prior_mean``              542 passed
replace both of this function's own refusals  542 passed
============================================  =========================

They are three different kinds of gap and it is worth keeping them apart.

**``offset_prediction`` had no test and no caller.** ``grep`` finds it three
times in ``src`` — all inside this module — and zero times under ``tests``. The
tiered ``compress()`` router does not even forward it. An argument reachable
only from outside the package is exactly the kind that rots quietly.

**``nuisance_prior_mean`` had no test.** Every fixture leaves the prior mean at
zero, where the term it contributes is identically zero, so the code path is
executed constantly and its *effect* never is. That is the harder shape to
notice: coverage is satisfied, behaviour is not.

**The two refusals were never executed at all.** An instrumented run over the
whole directory wrote 40542 entry records into ``compress_linear`` and **zero**
into either refusal branch. Replacing both with a foreign exception class
changed nothing. This is the reachability version of the same problem, and it
matters beyond tidiness: those two refusals are the ones whose exception class
would change if this function's arithmetic were ever delegated, and there was
nothing to report the change.

The fourth test below is not from a mutation. It pins the ``shapes``
normalisation, which differs between this package and ``bayesmith`` in a way
that makes terms from the two **refuse to combine** — a fact worth a guard
now rather than a debugging session later.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.compress import compress_linear


def _design(seed=0, n=40, width=3):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(n, width)))


def test_the_offset_prediction_is_subtracted_from_the_data():
    """It shifts the density, and by exactly the amount subtracting it would.

    Pinned as an identity against a second call rather than against a number:
    the claim is "``offset_prediction=c`` is compressing ``observed - c``", and
    a pinned number would also pass for an implementation that shifted by
    something else consistently.
    """
    design = _design()
    observed = jnp.asarray(np.random.default_rng(1).normal(size=(40,)))
    offset = jnp.asarray(np.random.default_rng(2).normal(size=(40,))) * 0.7
    probe = {"x": jnp.array([0.3, -0.2, 0.9])}

    with_offset = compress_linear(
        design={"x": design}, observed=observed, noise_std=0.1,
        shapes={"x": (3,)}, epoch_id="e0", offset_prediction=offset,
    )
    pre_subtracted = compress_linear(
        design={"x": design}, observed=observed - offset, noise_std=0.1,
        shapes={"x": (3,)}, epoch_id="e0",
    )
    assert float(with_offset(probe)) == pytest.approx(
        float(pre_subtracted(probe)), rel=1e-12
    )


def test_the_offset_prediction_actually_changes_the_answer():
    """The anti-vacuity half: without it the test above passes on a no-op.

    An implementation that ignored ``offset_prediction`` entirely would satisfy
    the identity above only if ``observed - offset`` also did nothing, which it
    does not. Said as its own test so the failure names the right thing.
    """
    design = _design()
    observed = jnp.asarray(np.random.default_rng(1).normal(size=(40,)))
    offset = jnp.asarray(np.random.default_rng(2).normal(size=(40,))) * 0.7
    probe = {"x": jnp.array([0.3, -0.2, 0.9])}
    common = dict(
        design={"x": design}, observed=observed, noise_std=0.1,
        shapes={"x": (3,)}, epoch_id="e0",
    )
    without = float(compress_linear(**common)(probe))
    with_it = float(compress_linear(**common, offset_prediction=offset)(probe))
    assert with_it != pytest.approx(without, rel=1e-9), (
        "offset_prediction moved the density by nothing, so the identity test "
        "above is passing vacuously"
    )


def test_a_non_zero_nuisance_prior_mean_reaches_the_density():
    """Every fixture leaves this at zero, where its contribution is zero too.

    Compared against the same model with the prior mean absorbed into the data:
    a prior ``N(m, s)`` on a nuisance with design ``A_p`` is the same model as a
    prior ``N(0, s)`` after subtracting ``A_p m`` from the observation. If the
    mean is being dropped, the two stop agreeing.
    """
    design, nuisance = _design(0, width=2), _design(3, width=3)
    observed = jnp.asarray(np.random.default_rng(4).normal(size=(40,)))
    mean = jnp.array([0.3, -0.2, 0.9])
    probe = {"x": jnp.array([0.4, -1.1])}
    common = dict(
        design={"x": design}, nuisance_design={"p": nuisance},
        nuisance_prior_std={"p": 0.7}, noise_std=0.1,
        shapes={"x": (2,)}, nuisance_shapes={"p": (3,)}, epoch_id="e0",
    )
    with_mean = compress_linear(
        observed=observed, nuisance_prior_mean={"p": mean}, **common
    )
    absorbed = compress_linear(observed=observed - nuisance @ mean, **common)
    # The two differ by the prior's own normalisation only, which is the same
    # constant in both, so the densities coincide.
    assert float(with_mean(probe)) == pytest.approx(float(absorbed(probe)), rel=1e-9)

    zero_mean = compress_linear(observed=observed, **common)
    assert float(with_mean(probe)) != pytest.approx(float(zero_mean(probe)), rel=1e-9), (
        "a non-zero prior mean moved the density by nothing, so this test "
        "cannot tell a dropped mean from a carried one"
    )


def test_the_two_refusals_this_function_raises_itself():
    """Never executed by the suite before this test: 0 hits in 40542 entries.

    Their class is the point as much as their existence. Both are
    ``StateValidationError``; ``bayesmith``'s counterparts are
    ``StructureError``, which shares only ``ValueError`` with it. If this
    function's arithmetic is ever delegated, these are the two whose class
    would change, and this is what would say so.
    """
    with pytest.raises(StateValidationError, match="at least one design block"):
        compress_linear(
            design={}, observed=jnp.zeros(4), noise_std=0.1,
            shapes={}, epoch_id="e0",
        )

    with pytest.raises(StateValidationError, match="nuisance_prior_std"):
        compress_linear(
            design={"x": _design(0, n=4, width=2)},
            nuisance_design={"p": _design(1, n=4, width=1)},
            observed=jnp.zeros(4), noise_std=0.1,
            shapes={"x": (2,)}, nuisance_shapes={"p": (1,)}, epoch_id="e0",
        )


def test_the_stored_shapes_are_the_callers_own_objects_not_normalised_tuples():
    """A difference from ``bayesmith`` that makes two terms refuse to combine.

    This package stores ``shapes[name]`` as given; ``bayesmith.marginal.compress``
    stores ``tuple(tuple(shapes[name]) ...)``. ``SqrtInfo.combine`` refuses when
    ``first.shapes != second.shapes``, so a term built here and a term built
    there over the same latents cannot be folded together when a caller passed
    a list.

    Pinned as the current behaviour, not as the desirable one. Whoever
    normalises this later should expect this test to fail and should delete it
    rather than widen it.
    """
    term = compress_linear(
        design={"x": _design(0, width=2)}, observed=jnp.zeros(40), noise_std=0.1,
        shapes={"x": [2]}, epoch_id="e0",
    )
    assert term.info.shapes == ([2],), (
        f"stored shapes are {term.info.shapes!r}. If this is now ((2,),) the "
        f"normalisation has been aligned with bayesmith -- delete this test."
    )
