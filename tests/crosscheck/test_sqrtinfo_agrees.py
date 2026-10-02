"""``rheplicant.inference.sqrtinfo`` against ``bayesmith.marginal.sqrtinfo``.

The square-root information form exists in both packages. These tests build
the same ``[R | z]`` as each package's own type, run the same operation on
both, and compare the results bit for bit, so a divergence between the two
copies is a failing test.

"near" is this package and "far" is bayesmith throughout.

What is compared, and why each still has two sides:

* ``combine`` and ``null``. Both packages own this arithmetic: the QR fold
  and its ``rho`` corner are written twice.
* ``marginalise``. The Schur complement and its Gaussian-integral constant
  are one kernel since this package's ``b87e44f`` delegated
  ``marginalise_arrays`` to bayesmith, so a kernel defect is the same on both
  sides and is not visible here. What is compared is the shell around it: the
  name-to-leading-block permutation, the offset carried into and out of the
  kernel, and the pivot reading the refusals stand on.

The exceptions are not compared. bayesmith raises ``StructureError`` where
this package raises ``StateValidationError``, and each suite owns its own
refusals.

Everything is float64, which is what the evidence layer runs at. A float32
comparison would agree to a tolerance that hides a difference in the
constant.

Moved from bayesmith's ``tests/crosscheck/`` at its ``d861220``, the last
revision that held it.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from bayesmith.marginal import SqrtInfo as FarSqrtInfo
from bayesmith.marginal import marginalise as far_marginalise

from rheplicant.inference.sqrtinfo import SqrtInfo as NearSqrtInfo
from rheplicant.inference.sqrtinfo import marginalise as near_marginalise


def _pair(rows, width, seed, offset=0.0, names=("x",), shapes=None):
    """The same ``[R | z]``, built as each package's own type: ``(near, far)``."""
    rng = np.random.default_rng(seed)
    factor = jnp.asarray(rng.normal(size=(rows, width)))
    target = jnp.asarray(rng.normal(size=rows))
    shapes = shapes or ((width,),)
    kwargs = {
        "factor": factor,
        "target": target,
        "offset": jnp.asarray(offset),
        "names": names,
        "shapes": shapes,
    }
    return NearSqrtInfo(**kwargs), FarSqrtInfo(**kwargs)


def _same(near, far, *, at):
    assert near.names == far.names
    assert near.shapes == far.shapes
    assert jnp.allclose(near.factor, far.factor, rtol=0, atol=0), "factor"
    assert jnp.allclose(near.target, far.target, rtol=0, atol=0), "target"
    assert float(near.offset) == float(far.offset), "offset"
    assert float(near.log_prob(at)) == float(far.log_prob(at)), "log_prob"


def test_combine_agrees_bitwise():
    """Same QR, same corner, same offset, to the bit.

    It is the same arithmetic in the same order on the same library, so
    bitwise is the bar. A tolerance would pass a different fold.
    """
    with jax.enable_x64(True):
        near_a, far_a = _pair(4, 3, seed=2)
        near_b, far_b = _pair(5, 3, seed=3, offset=-0.75)
        _same(
            NearSqrtInfo.combine(near_a, near_b),
            FarSqrtInfo.combine(far_a, far_b),
            at={"x": jnp.asarray([1.5, -0.5, 2.0])},
        )


def test_null_agrees_bitwise():
    with jax.enable_x64(True):
        names, shapes = ("a", "b"), ((2,), ())
        _same(
            NearSqrtInfo.null(names, shapes),
            FarSqrtInfo.null(names, shapes),
            at={"a": jnp.asarray([0.5, -1.0]), "b": jnp.asarray(2.0)},
        )


@pytest.mark.parametrize("prior_std", [0.7, 1.0, 3.0])
def test_marginalise_agrees_bitwise_including_the_constant(prior_std):
    """The two shells over the one kernel, at a non-unit prior.

    Both sides compute the Schur complement and its constant in bayesmith's
    kernel, so a kernel that dropped the constant would drop it for both and
    this would stay green. A red here is a difference between the shells: the
    name-to-leading-block permutation, the offset carried through the kernel,
    or the pivot reading.

    The prior is non-unit because the constant is zero at ``std = 1``. A
    non-zero offset has to pass through both shells, so a shell that mangled
    it cannot agree bitwise.
    """
    n_block, n_keep, rows = 2, 3, 9
    with jax.enable_x64(True):
        rng = np.random.default_rng(41)
        width = n_block + n_keep
        factor = jnp.concatenate(
            [
                jnp.asarray(rng.normal(size=(rows, width))),
                jnp.concatenate(
                    [jnp.eye(n_block) / prior_std, jnp.zeros((n_block, n_keep))],
                    axis=1,
                ),
            ],
            axis=0,
        )
        kwargs = {
            "factor": factor,
            "target": jnp.concatenate([jnp.asarray(rng.normal(size=rows)), jnp.zeros(n_block)]),
            "offset": jnp.asarray(
                -n_block * math.log(prior_std) - 0.5 * n_block * math.log(2.0 * math.pi)
            ),
            "names": ("b", "k"),
            "shapes": ((n_block,), (n_keep,)),
        }
        _same(
            near_marginalise(NearSqrtInfo(**kwargs), ["b"]),
            far_marginalise(FarSqrtInfo(**kwargs), ["b"]),
            at={"k": jnp.asarray([0.5, -0.25, 1.0])},
        )


def test_the_comparison_can_still_fail():
    """A comparison that compared nothing would pass silently.

    One entry of the far factor is perturbed by 1e-9 and the comparison has to
    notice, so a ``_same`` that stopped reading a field, or a ``_pair`` that
    stopped building both, fails here.
    """
    with jax.enable_x64(True):
        near, far = _pair(4, 3, seed=2)
        bent = FarSqrtInfo(
            factor=far.factor.at[0, 0].add(1e-9),
            target=far.target,
            offset=far.offset,
            names=far.names,
            shapes=far.shapes,
        )
        with pytest.raises(AssertionError, match="factor"):
            _same(near, bent, at={"x": jnp.asarray([1.0, 0.0, 0.0])})
