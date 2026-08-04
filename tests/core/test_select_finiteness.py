"""``SelectOperator`` must SELECT, and must say what selecting cannot repair.

The class promises ``data[t] = contribution[switch[t]][t]``. It was implemented
as ``leaf * mask``, which is not that promise but an arithmetic identity that
holds only while every branch is finite everywhere: a switched-OFF sample of a
branch that returns ``inf`` there enters the sum as ``inf * 0 -> nan``.

Whether the nan actually appeared depended on the execution mode, because XLA
is free to rewrite ``x * convert(pred)`` into a select and does so in some
fusion contexts and not others. So the forward answer was correct by luck, and
the luck ran out under ``jax.disable_jit`` (and, in a bare-function
formulation, under ``jax.jit`` while eager was fine). That is the shape of
failure this file exists to prevent: every execution mode is checked against
the same analytic number.

The gradient is a different claim and this file is careful not to overstate
it. Reverse mode differentiates EVERY branch at EVERY sample, and a branch
whose output is non-finite at a switched-off sample produces the nan inside
its OWN backward pass (``cotangent 0 * residual inf``), upstream of anything
the selector does. No masking here can repair that — measured, not assumed —
so it is a documented precondition with a working remedy, and both the
limitation and the remedy are pinned below.

The fixture is deliberately asymmetric: the four samples take four different
values, so a selection that picks the wrong branch, or the wrong sample of the
right branch, cannot produce the expected array by accident.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.combinators import SelectOperator
from rheplicant.core.coordinates import Coordinates
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.state import State

# t[0] = 0 is the singular sample. The spacing is uneven so that a shifted
# selection lands on a different number rather than on a coincidence.
TIME = jnp.array([0.0, 1.0, 2.0, 4.0])
# Sample 0 selects branch 1 (the finite one); the rest select branch 0. So the
# reciprocal's singularity is never selected, and the answer is finite.
SWITCH = jnp.array([1, 0, 0, 0])
SCALE, LEVEL = 1.0, 5.0
#            t=0 -> level, then scale/t at t = 1, 2, 4
ANALYTIC = jnp.array([LEVEL, 1.0, 0.5, 0.25])
# d(sum data)/d(scale) over the SELECTED samples only, and d/d(level).
D_SCALE = 1.0 / 1.0 + 1.0 / 2.0 + 1.0 / 4.0  # 1.75
D_LEVEL = 1.0


class Reciprocal(AbstractOperator):
    """``scale / t`` — non-finite at ``t = 0``, and it does not know that."""

    scale: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(self.scale / state.coords.time)


class GuardedReciprocal(AbstractOperator):
    """The same physics, guarding its own singularity — the documented remedy.

    The guard is the standard JAX double-``where``: the division never sees the
    zero, so no infinity is created and nothing infinite is differentiated.
    """

    scale: jax.Array

    def __call__(self, state: State) -> State:
        t = state.coords.time
        safe_t = jnp.where(t == 0.0, 1.0, t)
        return state.with_data(jnp.where(t == 0.0, 0.0, self.scale / safe_t))


class Level(AbstractOperator):
    """A finite branch whose value differs from every selected value."""

    level: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(self.level * jnp.ones_like(state.coords.time))


def _state(switch=SWITCH) -> State:
    return State(
        coords=Coordinates(time=TIME, extra={"switch_state": jnp.asarray(switch)})
    )


def _select(singular: bool) -> SelectOperator:
    first = (
        Reciprocal(scale=jnp.array(SCALE))
        if singular
        else GuardedReciprocal(scale=jnp.array(SCALE))
    )
    return SelectOperator(first, Level(level=jnp.array(LEVEL)), names=("recip", "level"))


class TestTheForwardIsASelect:
    """The switched-off singularity must not reach the output in ANY mode."""

    @pytest.mark.parametrize("singular", [True, False])
    def test_eager(self, singular):
        out = _select(singular)(_state()).data
        assert jnp.all(jnp.isfinite(out)), out
        assert jnp.allclose(out, ANALYTIC)

    @pytest.mark.parametrize("singular", [True, False])
    def test_filter_jit(self, singular):
        out = eqx.filter_jit(_select(singular))(_state()).data
        assert jnp.all(jnp.isfinite(out)), out
        assert jnp.allclose(out, ANALYTIC)

    @pytest.mark.parametrize("singular", [True, False])
    def test_disable_jit(self, singular):
        """The mode the mask-multiply failed in: no fusion, no XLA rewrite."""
        with jax.disable_jit():
            out = _select(singular)(_state()).data
        assert jnp.all(jnp.isfinite(out)), out
        assert jnp.allclose(out, ANALYTIC)

    def test_every_mode_agrees_with_every_other(self):
        """Boundary check: the three modes are three methods for one answer."""
        op, state = _select(singular=True), _state()
        eager = op(state).data
        jitted = eqx.filter_jit(op)(state).data
        with jax.disable_jit():
            nojit = op(state).data
        for name, value in (("eager", eager), ("jit", jitted), ("disable_jit", nojit)):
            assert jnp.all(jnp.isfinite(value)), f"{name}: {value}"
            assert jnp.allclose(value, ANALYTIC), f"{name}: {value}"

    def test_out_of_range_switch_still_selects_nothing(self):
        """The documented zero-fill, re-pinned now that the arithmetic changed."""
        out = _select(singular=False)(_state([9, 0, 0, 9])).data
        assert jnp.allclose(out, jnp.array([0.0, 1.0, 0.5, 0.0]))

    def test_a_wrong_selection_would_be_visible(self):
        """The fixture's own guard: no two expected samples are equal."""
        assert len(set(ANALYTIC.tolist())) == len(ANALYTIC)


class TestTheGradient:
    """What selecting can and cannot repair, measured rather than assumed."""

    def test_finite_branches_differentiate_exactly(self):
        op, state = _select(singular=False), _state()

        def loss(o):
            return jnp.sum(o(state).data)

        grads = eqx.filter_grad(loss)(op)
        assert jnp.isfinite(grads.branches[0].scale)
        assert jnp.allclose(grads.branches[0].scale, D_SCALE)
        assert jnp.allclose(grads.branches[1].level, D_LEVEL)

    def test_finite_branches_differentiate_exactly_through_jax_grad(self):
        """``jax.grad`` on a closure, not only equinox's filtered wrapper."""
        state = _state()

        def loss(scale):
            op = SelectOperator(
                GuardedReciprocal(scale=scale),
                Level(level=jnp.array(LEVEL)),
                names=("recip", "level"),
            )
            return jnp.sum(op(state).data)

        g = jax.grad(loss)(jnp.array(SCALE))
        assert jnp.isfinite(g)
        assert jnp.allclose(g, D_SCALE)

    def test_an_unguarded_singular_branch_still_poisons_the_gradient(self):
        """The documented PRECONDITION, pinned as a fact rather than hidden.

        This is not a bug the selector can fix. ``d(scale/t)/d(scale) = 1/t``
        is the branch's own residual, infinite at ``t = 0``; reverse mode
        multiplies it by the selector's zero cotangent for that sample, and
        ``0 * inf`` is nan before the selector sees anything. Replacing the
        forward multiply with a select — which this file's other tests pin —
        does not and cannot change that.

        If this test ever fails, the precondition has been lifted: delete it,
        and delete the paragraph in ``SelectOperator``'s docstring that states
        it.
        """
        op = _select(singular=True)
        state = _state()

        def loss(o):
            return jnp.sum(o(state).data)

        grads = eqx.filter_grad(loss)(op)
        assert jnp.isnan(grads.branches[0].scale)
        # The finite branch's gradient survives, which is why the failure is
        # so quiet: only the offending branch's parameters go nan.
        assert jnp.allclose(grads.branches[1].level, D_LEVEL)
        # ... and the forward stays right, so nothing about the run looks ill.
        assert jnp.allclose(op(state).data, ANALYTIC)
