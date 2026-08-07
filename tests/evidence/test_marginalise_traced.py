"""The Schur complement on both sides of a trace, from one copy of the constant.

Every assertion below is on an ABSOLUTE log-density or on an exact array
equality. `marginalise`'s constant is the one Plan A shipped wrong -- 1.07 nats
for three nuisances at std=0.7, 27.47 for twenty-five at std=3, and exactly zero
at std=1 -- so a test that compared shapes, gradients or moments would have
passed against the bug.

The split this file pins is not tidiness. The checked path concretises, so it
cannot be differentiated; the chain filter differentiates the same arithmetic
once per epoch with respect to the transition's own parameters. The last two
tests are the honest half of that split: what the traced path does when the
block is rank-deficient (returns a finite, plausible, wrong number) and what it
hands back so that somebody who can look -- an eager caller, or a construction-
time precondition -- still can.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.sqrtinfo import SqrtInfo, marginalise, marginalise_arrays


def _info(seed=0, rows=6, width=4):
    rng = np.random.default_rng(seed)
    return SqrtInfo(
        factor=jnp.asarray(rng.normal(size=(rows, width))),
        target=jnp.asarray(rng.normal(size=(rows,))),
        offset=jnp.asarray(0.25),
        names=("block", "kept"),
        shapes=((2,), (2,)),
    )


def test_the_kernel_and_the_checked_path_return_the_same_numbers():
    """One copy of the constant, verified rather than asserted in a docstring."""
    info = _info()
    checked = marginalise(info, ("block",))
    factor, target, offset, _ = marginalise_arrays(
        info.factor, info.target, info.offset, 2
    )
    np.testing.assert_array_equal(np.asarray(checked.factor), np.asarray(factor))
    np.testing.assert_array_equal(np.asarray(checked.target), np.asarray(target))
    assert float(checked.offset) == float(offset)


def test_the_kernel_reproduces_a_dense_gaussian_integral_absolutely():
    """The whole point of the constant, against an integral done a second way."""
    info = _info(seed=3)
    fisher = np.asarray(info.fisher())
    b = np.asarray(info.factor).T @ np.asarray(info.target)
    n_block = 2
    # log int exp(-0.5 x^T F x + b^T x) dx_block, by the standard block formula.
    f_bb = fisher[:n_block, :n_block]
    f_bk = fisher[:n_block, n_block:]
    f_kk = fisher[n_block:, n_block:]
    schur = f_kk - f_bk.T @ np.linalg.solve(f_bb, f_bk)
    shift = b[n_block:] - f_bk.T @ np.linalg.solve(f_bb, b[:n_block])
    constant = (
        0.5 * n_block * np.log(2 * np.pi)
        - 0.5 * np.linalg.slogdet(f_bb)[1]
        + 0.5 * b[:n_block] @ np.linalg.solve(f_bb, b[:n_block])
    )
    kept = jnp.asarray([0.7, -1.3])
    expected = float(
        info.offset
        - 0.5 * float(np.asarray(info.target) @ np.asarray(info.target))
        + constant
        - 0.5 * kept @ schur @ kept
        + shift @ kept
    )
    factor, target, offset, _ = marginalise_arrays(
        info.factor, info.target, info.offset, n_block
    )
    got = float(offset - 0.5 * jnp.sum((factor @ kept - target) ** 2))
    assert got == pytest.approx(expected, abs=1e-9)


def test_the_kernel_survives_jit_grad_and_scan():
    """Section 6 needs all three; `marginalise` gives none of them."""
    info = _info(seed=5)

    def density(scale):
        _, _, offset, _ = marginalise_arrays(
            info.factor * scale, info.target, info.offset, 2
        )
        return offset

    assert np.isfinite(float(jax.jit(density)(1.0)))
    analytic = float(jax.grad(density)(1.3))
    step = 1e-6
    numeric = (float(density(1.3 + step)) - float(density(1.3 - step))) / (2 * step)
    assert analytic == pytest.approx(numeric, rel=1e-6)

    def scanned(scale):
        def body(carry, _):
            _, _, offset, _ = marginalise_arrays(
                info.factor * scale, info.target, carry, 2
            )
            return offset, None

        total, _ = jax.lax.scan(body, jnp.zeros(()), None, length=4)
        return total

    assert np.isfinite(float(jax.jit(scanned)(1.1)))
    assert np.isfinite(float(jax.grad(scanned)(1.1)))


def test_the_checked_path_refuses_under_a_trace_rather_than_skipping_its_guard():
    """A guard that cannot run under a trace must not silently vanish under one.

    Both halves are asserted because they fail for the same reason and only one
    of them was noticed. `grad` is the half that matters: the chain filter
    differentiates this arithmetic once per epoch with respect to the
    transition's own parameters, so a `marginalise` that could be jitted but not
    differentiated would still be unusable there.

    The refusal is JAX's `ConcretizationTypeError` from `float(...)`, not a
    message of ours -- it names the abstract value, not the remedy. The remedy
    is named where a reader will be when they hit this: the first paragraph of
    `marginalise`'s docstring points at `marginalise_arrays`. Turning this into
    our own message would mean a tracer check whose own sabotage test is worth
    less than the one line it would replace.
    """
    info = _info(seed=7)

    def density(scale):
        scaled = SqrtInfo(
            factor=info.factor * scale,
            target=info.target,
            offset=info.offset,
            names=info.names,
            shapes=info.shapes,
        )
        return marginalise(scaled, ("block",)).offset

    assert np.isfinite(float(density(1.0)))
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.jit(density)(1.0)
    with pytest.raises(jax.errors.ConcretizationTypeError):
        jax.grad(density)(1.0)


def test_the_kernel_returns_the_pivots_the_checked_path_judges():
    info = _info(seed=9)
    _, _, _, pivots = marginalise_arrays(info.factor, info.target, info.offset, 2)
    upper = jnp.linalg.qr(
        jnp.concatenate([info.factor, info.target[:, None]], axis=1), mode="r"
    )
    np.testing.assert_allclose(
        np.asarray(pivots), np.abs(np.asarray(jnp.diag(upper))), atol=0.0
    )


def _block_scaled(scale, seed=11):
    """The same term with its block column multiplied by `scale`.

    `scale = 1` is a healthy block; `1e-10` is a block that constrains itself
    only at the rounding scale; `0` is one that does not appear at all. All
    three share the rest of the matrix, so what differs between them is the
    block's pivot and nothing else.
    """
    rng = np.random.default_rng(seed)
    factor = jnp.asarray(rng.normal(size=(4, 3))).at[:, 0].multiply(scale)
    return SqrtInfo(
        factor=factor,
        target=jnp.asarray(rng.normal(size=(4,))),
        offset=jnp.zeros(()),
        names=("block", "kept"),
        shapes=((1,), (2,)),
    )


def test_an_unconstrained_block_is_still_refused_by_name():
    """The nearest legitimate case is a factor of 1e10 away, and it passes."""
    for scale in (0.0, 1e-10):
        with pytest.raises(
            StateValidationError, match="does not constrain one of its own"
        ):
            marginalise(_block_scaled(scale), ("block",))
    # ...and the same term with a healthy block column is accepted: measured
    # +0.2028 nats, against the +23.23 the 1e-10 version would have returned.
    healthy = float(marginalise(_block_scaled(1.0), ("block",)).offset)
    assert healthy == pytest.approx(0.2028, abs=1e-3)


@pytest.mark.parametrize("scale", [float("nan"), float("inf")])
def test_a_poisoned_block_is_refused_rather_than_marginalised_to_nan(scale):
    """The degeneracy guard was defeated from both ends, and only one was seen.

    ``scale = 0`` was where the table above stopped. Two scales past it the
    guard read ``pivots[:n_block] <= floor`` with ``floor`` computed from
    ``float(jnp.max(pivots))``:

    * ``nan`` makes every pivot ``nan``, so ``max`` is ``nan``, ``floor`` is
      ``nan``, and ``pivot <= nan`` is **False** -- accepted, returning a
      ``SqrtInfo`` whose offset is ``nan``.
    * ``inf`` puts ``inf`` in the block's pivot and ``nan`` in the maximum, with
      the same result.

    Measured, before the fix: ``scale=0`` refused; ``scale=nan`` ACCEPTED with
    ``offset=nan``; ``scale=inf`` ACCEPTED with ``offset=nan``.
    ``SqrtInfo.__check_init__`` validates shapes and nothing else, so nothing
    between here and a campaign total says a word about it, and ``marginalise``
    is the function whose ``Raises:`` block promises to catch exactly this.
    """
    with pytest.raises(StateValidationError, match="is not finite"):
        marginalise(_block_scaled(scale), ("block",))


def test_the_kernel_cannot_see_what_the_checked_path_refuses():
    """What the traced path does about rank deficiency, in executable form.

    It does nothing, and that is not an oversight to be tolerated quietly: the
    judgement needs a comparison against a value, and under a trace there is no
    value. So say what it costs, in numbers, rather than in a caveat.

    A block whose column is **identically** zero is the easy half -- the pivot
    is exactly 0.0, `-log|pivot|` is `+inf`, and the offset comes back `+inf`,
    which anything downstream that tests for finiteness will catch.

    The half that is actually dangerous is a block that constrains itself only
    at the rounding scale, which is what a real near-degenerate night looks
    like. Measured, on the same term with its block column scaled by 1e-10: the
    kernel returns **+23.23 nats** where the healthy term returns **+0.203**.
    Finite, the right sign, the right order of magnitude for a good night's
    evidence, and wrong -- and it grows as `-log(pivot)`, so it is 27.8 at 1e-12
    and unbounded in principle. Nothing downstream tests for that.

    What the kernel hands back instead is the evidence: `pivots`, as data.
    `pivots[0]` is 2e-10 against a largest pivot of 1.59, which is exactly what
    `marginalise` judges eagerly and what the chain's construction-time
    positivity precondition on the transition makes impossible by construction
    rather than by inspection at each of a thousand steps. A caller that has
    neither owns the gap.
    """
    zero = _block_scaled(0.0)
    _, _, offset, pivots = marginalise_arrays(
        zero.factor, zero.target, zero.offset, 1
    )
    assert float(offset) == np.inf
    assert float(pivots[0]) == 0.0

    tiny = _block_scaled(1e-10)
    _, _, offset, pivots = marginalise_arrays(
        tiny.factor, tiny.target, tiny.offset, 1
    )
    # Finite and plausible -- the whole complaint. Bands are wide because the
    # value is -log(pivot) and the pivot is a scaled column norm.
    assert np.isfinite(float(offset))
    assert 20.0 < float(offset) < 27.0  # measured 23.23
    assert float(pivots[0]) == pytest.approx(2.0e-10, rel=0.5)
    assert float(jnp.max(pivots)) == pytest.approx(1.59, abs=0.1)

    # Two scales past zero, which is where this test used to stop. The kernel
    # does nothing about these either -- it cannot -- and the pivots it hands
    # back are `nan` and `inf` respectively, which is the evidence the checked
    # path now reads. Measured: both offsets come back `nan`.
    for scale in (float("nan"), float("inf")):
        poisoned = _block_scaled(scale)
        _, _, offset, pivots = marginalise_arrays(
            poisoned.factor, poisoned.target, poisoned.offset, 1
        )
        assert not np.isfinite(float(offset))
        assert not np.isfinite(float(jnp.max(pivots)))


def test_marginalising_nothing_is_the_identity_on_the_density():
    info = _info(seed=13)
    values = {"block": jnp.asarray([0.3, -0.2]), "kept": jnp.asarray([1.1, 0.4])}
    assert float(marginalise(info, ()).log_prob(values)) == pytest.approx(
        float(info.log_prob(values)), abs=1e-10
    )
