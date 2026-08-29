"""The smoother on a STIFF chain, which is where a square root earns its keep.

`smooth` is the one G3 name that did **not** move to ``bayesmith`` in Wave D,
and this file is the reason. Both implementations answer the same question and
agree to ~5e-16 on every fixture that existed before this file; they part
company when the chain gets stiff, and the far one parts company by returning a
plausible wrong number rather than by failing.

**The two arithmetics.** This package assembles the block-tridiagonal joint
**information square root** over ``zeta_1:N`` and does two triangular solves, so
it pays ``sqrt(kappa(F))``. ``bayesmith.marginal.chain.smooth`` assembles the
explicit precision and calls ``jnp.linalg.inv``, so it pays ``kappa(F)``. On a
chain whose ``process_std`` is small, ``kappa(F)`` grows like the square of the
stiffness ratio, and squaring it again is the whole difference.

**Why no oracle is needed to say which one is right.** Set ``phi = 1`` and let
``process_std -> 0``. The chain **freezes**: ``zeta_e`` is the same latent at
every epoch, so the smoothed posterior must *converge*, and its across-epoch
spread must go to zero. Measured on ``chain_bank.stacked()`` at
``theta = (0.4, -1.1)``:

====================  ==================  ==================
``process_std``       this package        ``bayesmith``
====================  ==================  ==================
1e-6                  0.454968749367      0.454968385497
1e-7                  0.454968749468      0.454928244813
1e-8                  0.454968748764      0.460387792656
1e-9                  0.454968747262      **0.931437422422**
1e-10                 0.454968730633      **nan**
====================  ==================  ==================

A limit exists and this package reaches it. The far column doubles and then
stops being a number. The ``0.93`` is the dangerous entry, not the ``nan``.

**``process_std=1e-9`` is a declared input, not a corner I invented.**
:class:`~rheplicant.inference.chain.LinearGaussianTransition`'s own docstring
says a chain that genuinely does not move is ``process_std=1e-9`` rather than
``0.0``, and ``test_transition.py`` constructs exactly that.

**Nothing here could fail before this file existed.** The stiffest chain
anywhere in ``tests/evidence`` was ``test_chain_conditioning.py``'s 6.32e-3,
which asserts only ``isfinite`` and exercises the *filter*; every smoother
fixture used ``bank.PROCESS_STD`` = 0.7071. Swapping this package's smoother for
the far one passed all 91 chain tests. Written per
``~/.claude/rules/common/boundary-validation.md``: sweep the parameter that
drives the failure, include the extreme values, and assert a property rather
than a pinned number.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.inference.chain import LinearGaussianTransition, smooth
from tests.evidence import chain_bank as bank

_NAMES = bank.THETA_NAMES
_SHAPES = ((), ())
_THETA = {name: jnp.array(v) for name, v in zip(_NAMES, [0.4, -1.1])}

#: Down to the value this package's own docstring recommends for a frozen chain.
_PROCESS_STD = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-7, 1e-8, 1e-9)


def _residual_drift(process_std):
    """How far a nearly-frozen chain may legitimately still drift.

    A chain with ``phi = 1`` is not *exactly* frozen at a finite ``process_std``;
    it retains a residual drift, and the bound has to scale with that rather
    than sit at a constant, or it is too loose to catch anything at 1e-9 and
    fails honest runs at 1e-4.

    **Measured** (x64, ``chain_bank``, widths 1-3, ``process_std`` = 1e-4 to
    1e-9): the across-epoch spread scales as ``process_std ** 2``, cleanly --
    2.007e-07, 2.007e-11, 2.165e-15 at width 1 for ``process_std`` 1e-4, 1e-6,
    1e-8. The constant is 20 at width 1, 208 at width 2, 149 at width 3, so
    ``1e4`` leaves roughly two decades of headroom; below about 1e-8 the spread
    stops falling and floors on roundoff at ~2e-16, which the additive term
    covers.

    The far implementation this guards against misses by far more than the
    headroom: at ``process_std = 1e-9`` it returns 0.9314 where the limit is
    0.4550, a walk of 0.48 against a bound of 1e-7.
    """
    return 1e4 * process_std**2 + 1e-7


def _blocks(n_epochs):
    """`chain_bank.stacked()` tiled to `n_epochs`, so length is a free variable."""
    factors, targets, offsets = bank.stacked()
    reps = int(np.ceil(n_epochs / factors.shape[0]))
    return (
        jnp.concatenate([factors] * reps)[:n_epochs],
        jnp.concatenate([targets] * reps)[:n_epochs],
        jnp.concatenate([offsets] * reps)[:n_epochs],
    )


@pytest.mark.parametrize("process_std", _PROCESS_STD)
@pytest.mark.parametrize("n_epochs", (6, 64, 256))
def test_every_smoothed_variance_is_positive(process_std, n_epochs):
    """A variance cannot be negative, and this is the assertion with no tolerance.

    It needs no reference value and no oracle, which is what makes it the one
    worth running at every cell: an implementation that has lost its
    conditioning announces itself here before any comparison is set up.
    """
    transition = LinearGaussianTransition(
        phi=jnp.array([[0.8]]),
        process_std=jnp.array([process_std]),
        initial_std=jnp.array([1.0]),
    )
    _, variance = smooth(_blocks(n_epochs), transition, _THETA, _NAMES, _SHAPES)
    variance = np.asarray(variance)
    assert np.all(np.isfinite(variance)), (
        f"process_std={process_std:g}, n_epochs={n_epochs}: "
        f"{int((~np.isfinite(variance)).sum())} of {variance.size} are not finite"
    )
    assert np.all(variance > 0.0), (
        f"process_std={process_std:g}, n_epochs={n_epochs}: minimum variance is "
        f"{float(variance.min()):.6e}. A negative variance is the signature of a "
        f"precision matrix inverted at a condition number the square-root form "
        f"would have absorbed."
    )


@pytest.mark.parametrize("width", (1, 2, 3))
def test_a_frozen_chain_converges_to_one_latent(width):
    """`phi = 1`, `process_std -> 0`: every epoch shares one latent, so converge.

    The property, not a pinned number -- the across-epoch spread of the smoothed
    mean must fall towards zero as the chain stiffens, and the mean itself must
    stop moving. An implementation that loses conditioning fails both: its
    spread stops falling and its mean walks away.
    """
    blocks = bank.stacked() if width == 1 else bank.wide_stacked(width)
    previous_mean = None
    previous_std = None
    for process_std in (1e-4, 1e-6, 1e-8, 1e-9):
        transition = LinearGaussianTransition(
            phi=jnp.eye(width),
            process_std=jnp.full((width,), process_std),
            initial_std=jnp.ones(width),
        )
        mean, variance = smooth(blocks, transition, _THETA, _NAMES, _SHAPES)
        mean = np.asarray(mean)
        assert np.all(np.isfinite(mean)), (
            f"width={width}, process_std={process_std:g}: the smoothed mean is "
            f"not finite"
        )
        assert np.all(np.asarray(variance) > 0.0)
        spread = float(np.max(np.ptp(mean, axis=0)))
        assert spread < _residual_drift(process_std), (
            f"width={width}, process_std={process_std:g}: the chain is frozen, so "
            f"every epoch holds the same latent, but the smoothed mean spreads "
            f"{spread:.3e} across epochs, against a bound of "
            f"{_residual_drift(process_std):.3e}."
        )
        if previous_mean is not None:
            walk = float(np.max(np.abs(mean - previous_mean)))
            assert walk < _residual_drift(previous_std), (
                f"width={width}, process_std={process_std:g}: stiffening the chain "
                f"from {previous_std:g} moved the smoothed mean by {walk:.3e}, "
                f"against a bound of {_residual_drift(previous_std):.3e}. A frozen "
                f"chain has a limit; an implementation that pays kappa(F) rather "
                f"than sqrt(kappa(F)) does not reach it."
            )
        previous_mean = mean
        previous_std = process_std


def test_the_smoother_still_agrees_with_the_dense_oracle_where_both_are_conditioned():
    """The anchor: at the bank's own stiffness the answer is the oracle's.

    Without this, the two tests above would be satisfied by a smoother that
    returned a well-conditioned constant.
    """
    transition = LinearGaussianTransition(
        phi=jnp.array([[bank.PHI]]),
        process_std=jnp.array([bank.PROCESS_STD]),
        initial_std=jnp.array([bank.INITIAL_STD]),
    )
    mean, variance = smooth(bank.stacked(), transition, _THETA, _NAMES, _SHAPES)
    expected_mean, expected_covariance = bank.oracle_zeta_posterior([0.4, -1.1])
    assert np.asarray(mean).ravel() == pytest.approx(
        np.asarray(expected_mean).ravel(), abs=1e-10
    )
    assert np.asarray(variance).ravel() == pytest.approx(
        np.diag(np.asarray(expected_covariance)), abs=1e-10
    )
