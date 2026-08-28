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
#: biased low; since ``condition_bound`` became a rigorous upper bound, 1e-3
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
    # Swept rather than pinned to one key, and that is a correction. This used
    # to pass `key(1)` and read as deterministic; it never was. Whether the
    # guard fires depends on the fluctuation drawn -- measured across 20 keys,
    # 14 refused and 6 did not, and the Wave B switch moved the split to 10/10
    # by changing how the draw's whitening is spelled, without touching what
    # this test is about. A one-key pin on a key-dependent outcome is a guard
    # that can stop being able to fail without anything saying so.
    caught = None
    for seed in range(20):
        try:
            conjugate_draw(
                cond, ("gain",), values, key=jax.random.key(seed), tol=1e-14,
                maxiter=1, require_convergence=1e-8,
            )
        except EquinoxRuntimeError as refused:
            caught = refused
            break
        except Exception as wrong_type:  # noqa: BLE001 -- the TYPE is the claim
            raise AssertionError(
                "the guard refused, but as "
                f"{type(wrong_type).__name__} rather than EquinoxRuntimeError: "
                "under `jax.jit` the diagnosis is buried in a callback "
                "traceback, which is why the transition is built with "
                "`eqx.filter_jit`"
            ) from wrong_type

    assert caught is not None, (
        "no key in 20 was refused, so this test can no longer fail for the "
        "reason it exists -- `maxiter=1` no longer produces an unconverged "
        "solve on this block"
    )
    assert "did not converge" in str(caught) or "cannot reach" in str(caught), (
        "The guard fired but said something unrecognised; the message is the "
        f"half of it that tells the caller what to do. Got: {caught}"
    )


def test_a_reachable_target_does_not_trip_the_guard(block) -> None:
    """Otherwise the test above would pass on a guard that refuses everything.

    The counterweight still exists; what changed is what counts as reachable.
    At ``REACHABLE`` the bound certifies the solve and the guard is silent, and
    at a tolerance no arithmetic can reach it refuses. Both halves are
    asserted, so this cannot pass on a guard that has stopped firing either.

    **This test is now the SILENT half only, and the reason is measured.**
    It used to also assert that the guard refuses at a tight tolerance. That
    cannot be established on this fixture, on any tolerance, on every machine:
    the guard's condition is ``residual * kappa > require_convergence``, and on
    x86_64 this CG converges to a residual of exactly 0.0, so the bound is 0
    and no positive tolerance is ever exceeded. Measured on the runner with
    1e-30, which did not fire. The same architecture reaches an exact fixed
    point in the GLS reweighting next door, for the same reason: these float32
    iterations land exactly there and hover one ulp above zero here.

    Nothing is lost by dropping it. That the guard CAN fire is what
    ``test_the_conjugate_convergence_guard_still_raises_equinox`` establishes,
    with ``maxiter=1`` against a tolerance it cannot meet -- a refusal that
    comes from the iteration being cut short rather than from the fixture's
    conditioning, and which passes on both platforms. The pair is still a
    pair; the two halves simply live in the two tests.
    """
    cond, values = block
    got, _ = conjugate_draw(
        cond, ("gain",), values, key=jax.random.key(1), tol=1e-8, maxiter=None,
        require_convergence=REACHABLE,
    )
    assert jnp.isfinite(got["gain"])


def test_the_cache_holds_one_program_per_branch(block) -> None:
    """Estimate and draw are different programs; four sweeps of each are not eight.

    The key carries ``tol``, ``maxiter`` and ``require_convergence`` because all
    three are baked into the compiled graph -- the first two reach CG as static
    arguments and the third decides whether the guard is traced in at all.

    It carries the ENGINE for a different reason: a ``log_conjugate`` block
    solves the same names against ``log`` of the data, so serving it from a
    ``conjugate`` block's slot would run one space's program on the other's
    arrays -- a confident wrong answer with every guard still green. That the
    two get separate slots is asserted in
    ``tests/inference/test_loglinear.py``, where a log-linear fixture exists.
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
            (("gain",), False, 1e-8, None, REACHABLE, "conjugate"),
            (("gain",), True, 1e-8, None, REACHABLE, "conjugate"),
        ]
    ), (
        f"Expected one program for the mean and one for the draw, got "
        f"{list(programs)}. A key that grows per sweep is a cache that never "
        "hits, which is the defect this change exists to remove."
    )
