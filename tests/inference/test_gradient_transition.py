"""The scanned gradient transition draws exactly what ``MCMC.run`` drew.

``gradient_draw`` used to build a fresh ``numpyro.infer.MCMC`` on every call,
which cost one XLA compilation per sweep -- 316 ms of work wrapped around 0.3 ms
of leapfrog. It now calls the documented ``MCMCKernel`` protocol directly and
scans ``kernel.sample`` under a cached ``eqx.filter_jit``.

**A faster loop that samples a different chain is not a faster loop**, so the
load-bearing test here compares the new transition, call for call, against an
``MCMC`` built locally in the test from the same kernel and the same key. Two
branches, because they differ in scan length: an adapting sweep is
``num_warmup=steps, num_samples=steps`` and therefore ``2 * steps`` kernel
applications, a frozen one is ``num_warmup=0`` and therefore ``steps``.

The fixture is **deliberately starved** -- amplitude 1, eight channels, unit
noise -- and that is the whole reason the test is worth running. The obvious
fixture (amplitude 1e3, 32 x 8, sigma 0.5) gives a posterior so sharp that NUTS
diverges on every step from a unit step size: the chain accepts once and then
sits still, so a stuck chain agrees bit-for-bit with a stuck chain and the test
passes no matter what the scan does. Measured on that fixture, scanning ``steps``
instead of ``2 * steps`` disagreed by **0.0e+00** -- a green test proving
nothing. On this one the same mutation disagrees by **1.7e-01**, at mean accept
probability 0.75 with no divergences.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, State
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.engines import (
    Conditioning,
    conditional_potential,
    gradient_draw,
)
from rheplicant.inference.uncertainty import as_noise_model
from rheplicant.radio import (
    ForegroundOperator,
    GainOperator,
    NoiseOperator,
    assemble,
)

dist = pytest.importorskip("numpyro.distributions", reason="numpyro not installed")

N_TIME, N_FREQ = 8, 8
STEPS = 20

#: Where to put the line. The two paths apply the same kernel to the same state
#: but drive it with different loop constructs -- ``lax.scan`` here against
#: numpyro's ``fori_collect`` -- and XLA fuses those differently, so at the
#: suite's float32 they agree to 3.0e-06 rather than exactly. (Under x64 the
#: adapting branch agrees to 0.0e+00 and the frozen one already does at float32.)
#: The mutation this test exists to catch is 1.7e-01 wide. 1e-04 sits 33x above
#: the roundoff and 1700x below the mutation, which is the whole point of
#: measuring both ends before choosing.
ROUNDOFF = 1e-4


#: No ``jax_enable_x64`` here, deliberately. The flag is process-global and the
#: suite's production dtype is float32 -- ``tests/test_evidence_session.py``
#: documents the eighteen tests that go red when it is on, and an earlier draft
#: of this file switched it on in a module fixture and took
#: ``test_linear_groups.py`` down with it. Nothing here needs the precision: the
#: agreement at float32 is 3.0e-06 against a 1.7e-01 mutation, which is all the
#: separation this test needs.


@pytest.fixture(scope="module")
def block():
    """A conditioning with one nonlinear latent, and a second held outside it.

    Amplitude 1 at unit noise over 64 samples: the likelihood is weak enough
    that NUTS mixes instead of diverging. See the module docstring for why that
    matters more than realism here.
    """
    freq = jnp.linspace(60e6, 85e6, N_FREQ)
    state = State(
        coords=Coordinates(time=jnp.arange(float(N_TIME)), freq=freq),
        key=jax.random.key(0),
        meta={"telescope": "transition"},
    )
    twin = assemble(
        ForegroundOperator(
            amplitude=jnp.array(1.0), spectral_index=jnp.array(2.5), ref_freq=70e6
        ),
        GainOperator(gain=jnp.array(1.1)),
        NoiseOperator(sigma=jnp.array(1.0)),
    )
    observed = twin(state).data
    space = ParameterSpace(
        latents=[
            Latent("beta", init=jnp.array(2.3), prior=dist.Normal(2.3, 0.3)),
            Latent("gain", init=jnp.array(1.0), prior=dist.Normal(1.0, 0.3)),
        ],
        bindings=[
            Bind("beta", into=lambda p: p["foregrounds"].spectral_index),
            Bind("gain", into=lambda p: p["gain"].gain),
        ],
    )
    fit = twin.without("noise")
    forward, _ = space.forward_fn(fit, state)
    cond = Conditioning(
        space=space,
        pipeline=fit,
        state_template=state,
        observed=observed,
        noise=as_noise_model(1.0),
        forward=forward,
    )
    values = {"beta": jnp.array(2.30), "gain": jnp.array(1.05)}
    return cond, values


def _mcmc_reference(cond, names, values, *, key, steps, tuning, adapt):
    """What ``gradient_draw`` used to do, rebuilt here so the test owns it."""
    from numpyro.infer import MCMC, NUTS

    potential = conditional_potential(cond, names, values)
    x0 = {name: values[name] for name in names}
    if adapt or tuning is None:
        kernel, warmup = NUTS(potential_fn=potential), steps
    else:
        step_size, inverse_mass_matrix = tuning
        kernel = NUTS(
            potential_fn=potential,
            step_size=step_size,
            inverse_mass_matrix=inverse_mass_matrix,
            adapt_step_size=False,
            adapt_mass_matrix=False,
        )
        warmup = 0
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=steps, progress_bar=False)
    mcmc.run(key, init_params=x0)
    drawn = mcmc.get_samples()
    adapted = mcmc.last_state.adapt_state
    return (
        {**values, **{name: drawn[name][-1] for name in names}},
        (adapted.step_size, adapted.inverse_mass_matrix),
    )


@pytest.mark.parametrize("adapt", [True, False], ids=["adapting", "frozen"])
def test_it_draws_what_mcmc_run_drew(block, adapt) -> None:
    """Same key, same kernel, same draw.

    The frozen branch needs a tuning to freeze, so the adapting branch is run
    first to supply one; that also exercises the handover the sweep loop does.
    """
    cond, values = block
    key = jax.random.key(11)
    tuning = None
    if not adapt:
        _, tuning = gradient_draw(
            cond,
            ("beta",),
            values,
            key=jax.random.key(2),
            steps=STEPS,
            tuning=None,
            adapt=True,
        )

    got, got_tuning = gradient_draw(
        cond, ("beta",), values, key=key, steps=STEPS, tuning=tuning, adapt=adapt
    )
    want, want_tuning = _mcmc_reference(
        cond, ("beta",), values, key=key, steps=STEPS, tuning=tuning, adapt=adapt
    )

    assert float(got["beta"]) == pytest.approx(float(want["beta"]), rel=ROUNDOFF), (
        "The scanned transition drew a different chain from MCMC.run. If the "
        "adapting branch is the one failing, check the scan length: it is "
        "2 * steps there (num_warmup=steps AND num_samples=steps), and steps "
        "alone disagrees at 1.7e-01."
    )
    assert float(got["gain"]) == float(values["gain"]), (
        "A latent OUTSIDE the block moved. The transition returns the block's "
        "names merged back into the full values dict; it must not write others."
    )
    assert float(got_tuning[0]) == pytest.approx(float(want_tuning[0]), rel=ROUNDOFF)
    assert jnp.allclose(got_tuning[1], want_tuning[1], rtol=ROUNDOFF)


def test_the_chain_actually_mixes(block) -> None:
    """The fixture must be one where a wrong scan length would show.

    Without this, every other assertion in the file is vacuous: a NUTS chain
    that diverges on every step returns its initial point regardless of how many
    times it is applied, so a broken scan and a correct one agree exactly. That
    is not hypothetical -- it is what the first draft of this fixture did.
    """
    cond, values = block
    a, _ = gradient_draw(
        cond, ("beta",), values, key=jax.random.key(11), steps=STEPS, adapt=True
    )
    b, _ = gradient_draw(
        cond, ("beta",), values, key=jax.random.key(11), steps=STEPS // 2, adapt=True
    )
    moved = abs(float(a["beta"]) - float(b["beta"])) / abs(float(a["beta"]))
    assert moved > 1e-3, (
        f"Halving the scan length moved the draw by {moved:.1e}, so this chain "
        "is not mixing and the scan-length assertions above prove nothing. "
        "Weaken the likelihood (lower amplitude, fewer channels, larger sigma) "
        "until NUTS stops diverging."
    )


def test_the_potential_sees_the_neighbours_at_their_current_values(block) -> None:
    """``others`` is a traced argument now; it must still be the current values.

    Lifting it out of the closure is what stops the recompilation, and it is
    also the one way to get this wrong silently: a transition that keeps
    stepping against a STALE neighbour still runs, still converges, and reports
    a chi-squared computed from the fresh values.
    """
    cond, values = block
    moved = {**values, "gain": jnp.array(1.30)}
    a, _ = gradient_draw(
        cond, ("beta",), values, key=jax.random.key(3), steps=STEPS, adapt=True
    )
    b, _ = gradient_draw(
        cond, ("beta",), moved, key=jax.random.key(3), steps=STEPS, adapt=True
    )
    assert float(a["beta"]) != float(b["beta"]), (
        "Moving a latent OUTSIDE the block left the block's draw unchanged, so "
        "the transition is conditioning on a stale neighbour."
    )


def test_the_cache_compiles_one_program_per_branch(block) -> None:
    """A caller-supplied ``programs`` dict is the thing that stops recompiling.

    Keyed on ``(names, steps, adapting)`` and nothing else -- notably not on the
    conditioning, whose id CPython reuses after collection.
    """
    cond, values = block
    programs: dict = {}
    for sweep in range(4):
        _, tuning = gradient_draw(
            cond,
            ("beta",),
            values,
            key=jax.random.key(sweep),
            steps=STEPS,
            adapt=True,
            programs=programs,
        )
    assert list(programs) == [(("beta",), STEPS, True)], (
        f"Expected one cached program for one block on one branch, got "
        f"{list(programs)}. A key that grows per sweep is a cache that never "
        "hits, which is the defect this whole change exists to remove."
    )
