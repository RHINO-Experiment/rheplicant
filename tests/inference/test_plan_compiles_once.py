"""A sweep is not a compilation unit: ``SamplingPlan`` must compile once per run.

``SamplingPlan.sample`` recompiles its transitions on EVERY sweep. Measured on
the fixture below: **45 XLA compilations over 20 sweeps, 121 over 60** -- 2.02
per sweep in steady state, not a warm-up artefact. The arithmetic the sampler
exists to do is 0.3 ms of a 315 ms sweep; the rest is the compiler.

The cause is closure identity. ``conditional_potential``
(``engines.py:163-178``) closes over ``others`` -- every latent outside the
block -- as concrete arrays, so each sweep mints a fresh Python function and
every ``jax.jit`` below it sees a cache it has never seen before. Reusing one
``MCMC`` object does not help: measured, the same object with identical inputs
still pays 240 ms of compile, because NumPyro's ``fori_collect`` defines
``_fori_collect_loop`` locally and jits *that*.

**Why a compile COUNT and not a wall clock.** A timing test on a shared machine
is a flaky test, and a slow sweep has many innocent explanations. A compilation
count has one: something the run should have reused was rebuilt. It is also the
measurement that survives a faster machine, which a threshold on seconds does
not.

The counting test is xfail(strict=True): it is the defect, written so it
can be fixed, and strict so that landing the repair turns it into a failing
xpass rather than a quietly mislabelled pass. See
``docs/superpowers/plans/2026-08-07-samplingplan-compiles-once.md`` has the
staged repair and its measured effect (386.7 -> 1.7 ms per sweep).
"""

import logging
import re

import jax
import jax.numpy as jnp
import pytest

from rheplicant import Coordinates, State
from rheplicant.inference import Bind, Block, Latent, ParameterSpace, SamplingPlan
from rheplicant.radio import (
    ForegroundOperator,
    GainOperator,
    NoiseOperator,
    assemble,
)

dist = pytest.importorskip("numpyro.distributions", reason="numpyro not installed")

#: What XLA logs when it has just built a program. Matching the message rather
#: than counting `jit` calls is deliberate: a cache HIT calls jit too.
_COMPILED = re.compile(r"Finished XLA compilation")

N_TIME, N_FREQ = 64, 8


class _CountCompiles(logging.Handler):
    """Counts XLA compilations while it is installed on the ``jax`` logger."""

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.n = 0

    def emit(self, record: logging.LogRecord) -> None:
        if _COMPILED.search(record.getMessage()):
            self.n += 1


def _count_compiles(fn) -> int:
    logger = logging.getLogger("jax")
    handler = _CountCompiles()
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    try:
        fn()
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate
    return handler.n


@pytest.fixture(scope="module")
def mixed():
    """One conjugate block and one gradient block, the case the plan is for.

    The spectral index rather than the amplitude, because a scalar gain and an
    amplitude are exactly degenerate and ``identifiability`` refuses the pair --
    correctly, and it is not what this test is about.
    """
    state = State(
        coords=Coordinates(time=jnp.arange(float(N_TIME)),
                           freq=jnp.linspace(60e6, 85e6, N_FREQ)),
        key=jax.random.key(0),
        meta={"telescope": "compile-count"},
    )
    twin = assemble(
        ForegroundOperator(amplitude=jnp.array(1e3), spectral_index=jnp.array(2.5),
                           ref_freq=70e6),
        GainOperator(gain=jnp.array(1.1)),
        NoiseOperator(sigma=jnp.array(0.5)),
    )
    observed = twin(state).data
    space = ParameterSpace(
        latents=[
            Latent("gain", init=jnp.array(1.0), linear=True,
                   prior=dist.Normal(1.0, 0.3)),
            Latent("beta", init=jnp.array(2.3), prior=dist.Normal(2.3, 0.3)),
        ],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind("beta", into=lambda p: p["foregrounds"].spectral_index),
        ],
    )
    plan = SamplingPlan(space, Block("gain"), Block("beta", steps=10))
    return plan, twin.without("noise"), state, observed


def _sample(mixed, n_sweeps):
    plan, fit, state, observed = mixed
    return lambda: plan.sample(
        fit, state, observed, noise=0.5, key=jax.random.key(1),
        n_sweeps=n_sweeps, warmup=n_sweeps // 2,
    )


def test_the_handler_sees_compilations_at_all(mixed) -> None:
    """A counter that counts nothing would make every assertion below pass.

    ``jax`` logs the message only at DEBUG, so a library or a test-runner
    setting that silences the logger turns this file into a vacuous pass.
    """
    assert _count_compiles(_sample(mixed, 8)) > 0, (
        "No 'Finished XLA compilation' records were seen at all. Either jax "
        "stopped logging under that message, or something is suppressing the "
        "'jax' logger -- either way every assertion in this file is vacuous."
    )


@pytest.mark.xfail(
    strict=True,
    reason="stage 0 of docs/superpowers/plans/2026-08-07-samplingplan-compiles-once.md: "
    "the defect is stated here first and repaired in stages 1-2. strict=True so "
    "that landing the repair turns this into a FAILING xpass, which is what forces "
    "the marker off rather than leaving a passing test quietly mislabelled.",
)
def test_the_transition_is_compiled_once_per_run(mixed) -> None:
    """Tripling the sweeps must not triple the compilations.

    Measured today: 45 at 20 sweeps and 121 at 60, so this fails by a factor of
    about 2.7. After the repair the count is flat -- the programs are built once
    and reused, which is what makes the run's wall clock flat in ``n_sweeps``
    (1.518 s at n=8 against 1.722 s at n=80, versus 3.5 s and 35.6 s today).
    """
    few = _count_compiles(_sample(mixed, 20))
    many = _count_compiles(_sample(mixed, 60))
    assert many <= few + 4, (
        f"{few} XLA compilations over 20 sweeps and {many} over 60: the sweep "
        f"loop is rebuilding its transitions instead of reusing them, so "
        f"{(many - few) / 40:.2f} compilations are paid per extra sweep. A "
        "compile is ~300 ms and the leapfrog work it wraps is ~0.3 ms. See "
        "docs/superpowers/plans/2026-08-07-samplingplan-compiles-once.md."
    )
