"""``tests/seam/`` runs only under ``jax_enable_x64``. This is the gate.

The adapter's acceptance suite has a deterministic tier, and that tier's whole
claim is "the number that comes back through the seam is the number the dense
normal equations give, to roundoff". At float32 that claim cannot be stated:
the agreement to check is at ``rtol <= 1e-12`` and float32 carries about seven
decimal digits, so the tier would be measuring the dtype rather than the seam.

The rest of the suite must NOT get x64. float32 is this package's production
dtype and a population of tests elsewhere assert refusals only float32 forces;
that population -- its size, its file-by-file breakdown, the date it was
measured and the command that reproduces it -- is recorded once in
``tests/test_evidence_session.py`` and deliberately not repeated here.

**This is the second x64-gated directory, and it makes a sentence in that file
false.** It used to say ``tests/evidence/`` was the only one, and drew a
conclusion from that about what "the two sessions merge" will mean at the end
of the migration. The sentence has been corrected there rather than here, so
the correction sits beside the reasoning it changes.

Why this file does NOT call ``jax.config.update("jax_enable_x64", True)``: a
conftest in a subdirectory is imported while the whole session is being
collected, so the update would land before any test ran and break every test
that needs float32. The flag arrives from the environment or it does not
arrive.

Skip, not deselect -- ``tests/test_readme_counts.py`` derives the README's test
count from a plain ``--collect-only``, and a deselected test is gone from that
count while a skipped one is still collected.
"""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest
from seam_models import GAIN, N_TIME, SKY_A, SKY_B

from rheplicant.core.combinators import SumOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import HomoscedasticNoise, ParameterSpace
from rheplicant.radio import GainOperator, SkyOperator

_DIRECTORY = Path(__file__).resolve().parent

_REASON = (
    "the adapter's deterministic tier needs float64, and jax_enable_x64 is "
    "process-global -- run `JAX_ENABLE_X64=1 .venv/bin/python -m pytest tests/seam`. "
    "This skip is not a hole in the default suite: tests/test_seam_session.py runs "
    "exactly that command as a subprocess and fails if it fails, or if it turns out "
    "to have run nothing."
)


def pytest_collection_modifyitems(config, items):
    """Defer this directory's tests when the process is not in x64."""
    if jax.config.read("jax_enable_x64"):
        return
    gate = pytest.mark.skip(reason=_REASON)
    for item in items:
        path = getattr(item, "path", None)
        # Filter by path rather than trusting pytest to scope a subdirectory
        # conftest's hook, because it does not: the hook is a SESSION hook and
        # is handed every item in the run. An unfiltered loop would skip the
        # entire suite whenever x64 is off, which in the default session is
        # always.
        if path is not None and Path(path).is_relative_to(_DIRECTORY):
            item.add_marker(gate)


# ------------------------------------------------------------- the fixtures --
#
# One instrument, several spaces over it. The pipeline is written once and each
# example declares a different ParameterSpace against it, so an example can
# never be comparing a different model from its own dense reference -- which is
# the failure that cost this programme a session: a graph fixture missing an
# additive offset read exactly like a solver bug.

@pytest.fixture
def instrument():
    """Two additive sky terms through a per-time gain.

    Per-TIME and not scalar, so the design matrix of a gain block has full
    column rank and a dense reference is a fair comparison rather than a
    rank-one problem where the dense solve is the unstable one.
    """
    scalar = Pipeline(
        SumOperator(
            SkyOperator(amplitude=jnp.array(SKY_A)),
            SkyOperator(amplitude=jnp.array(SKY_B)),
            names=("sky_a", "sky_b"),
        ),
        GainOperator(gain=jnp.array(GAIN)),
        names=("sum", "gain"),
    )
    return eqx.tree_at(lambda p: p["gain"].gain, scalar, jnp.full((N_TIME,), GAIN))


@pytest.fixture
def gain_space():
    """The per-time gain, declared linear, with a Gaussian prior."""
    return ParameterSpace.direct(
        "gains",
        init=jnp.full((N_TIME,), GAIN),
        into=lambda p: p["gain"].gain,
        linear=True,
        prior=dist.Normal(jnp.full((N_TIME,), GAIN), 5.0),
    )


@pytest.fixture
def gain_truth():
    return GAIN + 0.1 * jnp.arange(N_TIME, dtype=float)


@pytest.fixture
def quiet_noise():
    return HomoscedasticNoise(sigma=jnp.array(0.5))


@pytest.fixture
def observed(instrument, gain_space, gain_truth, quiet_noise, template_state):
    """Data realised at the truth through the noise model's OWN generator.

    ``noise.realise`` and not ``truth + sigma * normal``: the two agree for a
    homoscedastic model and differ for the radiometer, and a hand-written
    generator beside a likelihood carrying its own sigma is exactly the drift
    this package's noise module exists to prevent. Writing it once here means
    an example cannot accidentally realise data under one model and score it
    under another.
    """
    forward, _ = gain_space.forward_fn(instrument, template_state)
    return quiet_noise.realise(forward({"gains": gain_truth}), key=jax.random.key(1))
