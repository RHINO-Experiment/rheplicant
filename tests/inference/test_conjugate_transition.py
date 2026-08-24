"""The jitted conjugate update solves what the unjitted one solved, and still refuses.

Stage 2 of ``docs/superpowers/plans/2026-08-07-samplingplan-compiles-once.md``.
``linear_operator`` returns an operator whose matvec closes over ``at=values``,
so every sweep handed CG a Python function it had never seen and XLA rebuilt the
program -- measured, 21 compilations over 20 sweeps against 61 over 60, exactly
one per sweep. Wrapping the update in a cached ``eqx.filter_jit`` takes that to
zero: 3 compilations at both 20 and 60 sweeps.

**The jit choice here is correctness, not taste.** ``gcr_sample``'s convergence
guard is an :func:`equinox.error_if`. Under ``eqx.filter_jit`` it surfaces as an
``EquinoxRuntimeError`` carrying the message that names the remedy; under
``jax.jit`` the same condition still fires but arrives as a bare
``JaxRuntimeError``. Both refuse, so a test that only checked "something raised"
would pass either way -- which is why the guard test below asserts the exception
TYPE, and why it is the test this stage was required to ship.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, State
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.engines import (
    Conditioning,
    conjugate_draw,
    conjugate_estimate,
)
from rheplicant.inference.linear import gcr_sample, linear_operator, wiener_solve
from rheplicant.inference.uncertainty import as_noise_model
from rheplicant.radio import (
    ForegroundOperator,
    GainOperator,
    NoiseOperator,
    assemble,
)

dist = pytest.importorskip("numpyro.distributions", reason="numpyro not installed")

N_TIME, N_FREQ = 16, 8

#: float32 across two paths that build the same graph by different routes. The
#: solve is a CG iteration, so this is iteration noise rather than one ulp.
CLOSE = 1e-5


@pytest.fixture(scope="module")
def block():
    """A gain the data identifies well: this fixture is about plumbing, not κ."""
    state = State(
        coords=Coordinates(
            time=jnp.arange(float(N_TIME)), freq=jnp.linspace(60e6, 85e6, N_FREQ)
        ),
        key=jax.random.key(0),
        meta={"telescope": "conjugate-transition"},
    )
    twin = assemble(
        ForegroundOperator(
            amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5), ref_freq=70e6
        ),
        GainOperator(gain=jnp.array(1.1)),
        NoiseOperator(sigma=jnp.array(0.5)),
    )
    observed = twin(state).data
    space = ParameterSpace(
        latents=[
            Latent("gain", init=jnp.array(1.0), linear=True, prior=dist.Normal(1.0, 0.3))
        ],
        bindings=[Bind("gain", into=lambda p: p["gain"].gain)],
    )
    fit = twin.without("noise")
    forward, _ = space.forward_fn(fit, state)
    cond = Conditioning(
        space=space,
        pipeline=fit,
        state_template=state,
        observed=observed,
        noise=as_noise_model(0.5),
        forward=forward,
    )
    return cond, {"gain": jnp.array(1.0)}


#: A ``require_convergence`` this fixture can actually meet at float32.
#:
#: Measured on this fixture: the bound is 4.69e+07, and CG lands at the
#: single-precision residual floor -- 1.12e-07 for the mean and 2.23e-07 for
#: the draw, whose right-hand side carries the two fluctuation terms. So the
#: error bounds are 5.24 and 10.47, and no target below the larger of them can
#: be certified here at float32.
#:
#: It used to be 1e-3 throughout this file, back when kappa was a measurement
#: biased low; since ``_condition_number`` became a rigorous upper bound, 1e-3
#: is unreachable on this block rather than merely demanding. See that
#: function's docstring for why the measurement had to go.
REACHABLE = 20.0


def _reference(cond, names, values, *, key):
    """The update as it read before the jit, rebuilt so the test owns the baseline."""
    operator = linear_operator(
        cond.space, cond.pipeline, cond.state_template,
        names=names, at=values, check=False,
    )
    solve = wiener_solve if key is None else gcr_sample
    extra = {} if key is None else {"key": key}
    solved, _ = solve(
        operator, cond.observed, noise_std=cond.sigma(values),
        tol=1e-8, maxiter=None, require_convergence=REACHABLE, **extra,
    )
    return solved


def test_the_mean_matches_the_unjitted_solve(block) -> None:
    cond, values = block
    got, _ = conjugate_estimate(
        cond, ("gain",), values, tol=1e-8, maxiter=None,
        require_convergence=REACHABLE,
    )
    want = _reference(cond, ("gain",), values, key=None)
    assert float(got["gain"]) == pytest.approx(float(want["gain"]), rel=CLOSE)


def test_the_draw_matches_the_unjitted_draw(block) -> None:
    """Same key, same draw -- a faster sampler that samples something else is not one."""
    cond, values = block
    key = jax.random.key(7)
    got, _ = conjugate_draw(
        cond, ("gain",), values, key=key, tol=1e-8, maxiter=None,
        require_convergence=REACHABLE,
    )
    want = _reference(cond, ("gain",), values, key=key)
    assert float(got["gain"]) == pytest.approx(float(want["gain"]), rel=CLOSE), (
        "The jitted conjugate draw is not the draw the unjitted path made from "
        "the same key. gcr_sample's randomness must come from `key` alone."
    )


def test_the_conjugate_convergence_guard_still_raises_equinox(block) -> None:
    """The guard must keep its TYPE, not merely keep refusing.

    ``maxiter=1`` against ``require_convergence=1e-8`` cannot converge. Both
    ``eqx.filter_jit`` and ``jax.jit`` refuse it -- the difference is that
    ``jax.jit`` yields a bare ``JaxRuntimeError`` with the diagnosis buried in a
    callback traceback. Asserting the type is what forces the right wrapper;
    asserting only the text would pass under either.
    """
    from equinox import EquinoxRuntimeError

    cond, values = block
    with pytest.raises(EquinoxRuntimeError) as caught:
        conjugate_draw(
            cond, ("gain",), values, key=jax.random.key(1), tol=1e-14, maxiter=1,
            require_convergence=1e-8,
        )
    assert "did not converge" in str(caught.value) or "cannot reach" in str(
        caught.value
    ), (
        "The guard fired but said something unrecognised; the message is the "
        f"half of it that tells the caller what to do. Got: {caught.value}"
    )


def test_a_reachable_target_does_not_trip_the_guard(block) -> None:
    """Otherwise the test above would pass on a guard that refuses everything.

    The counterweight still exists; what changed is what counts as reachable.
    At ``REACHABLE`` the bound certifies the solve and the guard is silent; at
    1e-3 it refuses, and on this fixture at float32 that refusal is CORRECT --
    the draw's error bound really is 10.47 and no tolerance moves it. Both halves are
    asserted, so this cannot pass on a guard that has stopped firing either.
    """
    cond, values = block
    got, _ = conjugate_draw(
        cond, ("gain",), values, key=jax.random.key(1), tol=1e-8, maxiter=None,
        require_convergence=REACHABLE,
    )
    assert jnp.isfinite(got["gain"])

    from equinox import EquinoxRuntimeError

    with pytest.raises(EquinoxRuntimeError):
        conjugate_draw(
            cond, ("gain",), values, key=jax.random.key(1), tol=1e-8,
            maxiter=None, require_convergence=1e-3,
        )


def test_the_cache_holds_one_program_per_branch(block) -> None:
    """Estimate and draw are different programs; four sweeps of each are not eight.

    The key carries ``tol``, ``maxiter`` and ``require_convergence`` because all
    three are baked into the compiled graph -- the first two reach CG as static
    arguments and the third decides whether the guard is traced in at all.
    """
    cond, values = block
    programs: dict = {}
    for sweep in range(4):
        conjugate_estimate(
            cond, ("gain",), values, tol=1e-8, maxiter=None,
            require_convergence=REACHABLE, programs=programs,
        )
        conjugate_draw(
            cond, ("gain",), values, key=jax.random.key(sweep), tol=1e-8,
            maxiter=None, require_convergence=REACHABLE, programs=programs,
        )
    assert sorted(programs) == sorted(
        [
            (("gain",), False, 1e-8, None, REACHABLE),
            (("gain",), True, 1e-8, None, REACHABLE),
        ]
    ), (
        f"Expected one program for the mean and one for the draw, got "
        f"{list(programs)}. A key that grows per sweep is a cache that never "
        "hits, which is the defect this change exists to remove."
    )
