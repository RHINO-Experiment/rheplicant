"""A latent's declared scale is a build-time constant, so it must not enter the trace.

Stage 3 of ``docs/superpowers/plans/2026-08-07-samplingplan-compiles-once.md``,
and a one-line change: ``_magnitude`` computed ``float(jnp.max(jnp.abs(
latent.init)))``, which raises ``ConcretizationTypeError`` under any jit.

The reason is worth stating because it is the opposite of the obvious guess.
Closed-over arrays are **not** traced by ``eqx.filter_jit`` -- ``latent.init``
is a concrete ``ArrayImpl`` inside the jit, verified directly. What introduces
the tracer is the ``jnp`` call itself: inside an active trace, ``jnp.max`` is
staged into the jaxpr even when its input is concrete, because JAX does not
constant-fold at trace time. ``np.max`` runs eagerly and returns a Python float.

So ``jnp`` versus ``np`` here is a claim about *when a value is known*, not a
style preference, and ``latent.init`` is a declaration -- known when the program
is built. This blocks ``gradient_estimate`` from being jitted at all, which is
why it is on the path to stage 4.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, State
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.engines import Conditioning, gradient_estimate
from rheplicant.inference.linear import _magnitude
from rheplicant.inference.uncertainty import as_noise_model
from rheplicant.radio import (
    ForegroundOperator,
    GainOperator,
    NoiseOperator,
    assemble,
)

dist = pytest.importorskip("numpyro.distributions", reason="numpyro not installed")


@pytest.fixture(scope="module")
def block():
    state = State(
        coords=Coordinates(
            time=jnp.arange(16.0), freq=jnp.linspace(60e6, 85e6, 8)
        ),
        key=jax.random.key(0),
        meta={"telescope": "magnitude"},
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
        latents=[Latent("beta", init=jnp.array(2.3), prior=dist.Normal(2.3, 0.3))],
        bindings=[Bind("beta", into=lambda p: p["foregrounds"].spectral_index)],
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
    return cond, {"beta": jnp.array(2.30)}


def test_magnitude_returns_a_python_float_inside_a_trace() -> None:
    """The narrow claim, isolated from everything that calls it."""
    latent = Latent(
        "x", init=jnp.array([2.5, -4.0]), prior=dist.Normal(jnp.zeros(2), jnp.ones(2))
    )
    seen = {}

    @eqx.filter_jit
    def under_jit(x):
        seen["value"] = _magnitude(latent)
        seen["type"] = type(seen["value"]).__name__
        return x * 0

    under_jit(jnp.array(1.0))
    assert seen["type"] == "float", (
        f"_magnitude returned a {seen['type']} inside a trace. If it is a "
        "tracer, a jnp call staged it out: use np, which runs eagerly on the "
        "concrete init."
    )
    assert seen["value"] == pytest.approx(4.0)


def test_gradient_estimate_is_jittable(block) -> None:
    """The test the plan names, and the reason the one-liner is worth landing."""
    cond, values = block

    @eqx.filter_jit
    def estimate(v):
        return gradient_estimate(cond, ("beta",), v, steps=5)[0]

    got = estimate(values)
    want, _ = gradient_estimate(cond, ("beta",), values, steps=5)
    assert float(got["beta"]) == pytest.approx(float(want["beta"]), rel=1e-5), (
        "The jitted descent reached a different point from the eager one."
    )


def test_the_all_zero_fallback_survives(block) -> None:
    """A zero init would give a zero step size, so 1.0 stands in -- still, under jit.

    The fallback is a Python ``!=`` on the value. That works only because the
    value is a real float; on a tracer it would raise, so this asserts the
    branch is still reachable rather than merely present in the source.
    """
    zero = Latent(
        "z", init=jnp.zeros(3), prior=dist.Normal(jnp.zeros(3), jnp.ones(3))
    )
    seen = {}

    @eqx.filter_jit
    def under_jit(x):
        seen["value"] = _magnitude(zero)
        return x * 0

    under_jit(jnp.array(1.0))
    assert seen["value"] == 1.0
