"""D23: the fixture that can tell the two refusal criteria apart.

The migration ledger's D23 recorded a semantic difference that was registered,
unruled, and carried by **no test in either package**:

* this module used to refuse on the OBSERVED JACOBIAN's RANK -- along a
  direction the data cannot see there is no likelihood mode, only a ray;
* ``bayesmith.diagnose.sensitivity`` refuses on the REST TERM's own
  CURVATURE at the mode, with a condition-number ceiling of ``1/sqrt(eps)``.

Wave A switched ``prior_sensitivity`` to the remote, so **the curvature
criterion is what has been running ever since**. The two refusals were ported
with the same wording and the facade only translates the exception class, so
none of the sixty-four tests around it could tell. D23 said a ruling needed a
fixture that could; this file is that fixture.

The difference runs in **both** directions, and they are not equally
reachable:

* **the remote ACCEPTS what a rank test refuses** -- a selected latent held
  by a DOWNSTREAM density (``child ~ Normal(parent, s)`` with ``child``
  outside the selection). Measured in a bayesmith graph: observed-Jacobian
  rank 0 of 1, and ``prior_sensitivity`` returns a shift of -0.1096. But this
  package **cannot declare it**: a ``Latent``'s prior is a numpyro
  distribution built at declaration time, so its parameters are concrete
  arrays and there is no latent to point at. See
  :class:`TestTheDirectionThisPackageCannotReach`.
* **the remote REFUSES what a rank test accepts** -- a near-collinear design
  whose observed Jacobian is FULL RANK while the curvature has spent more
  than half the arithmetic's digits. That one is an ordinary two-parameter
  model, so it is reachable by anyone, and it is what
  :class:`TestTheDirectionAnyoneCanReach` pins.

Probe: ``bayesmith/docs/probes/probe_15_d23_two_criteria.py``.
"""

from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.inference import Bind, Latent, ParameterSpace
from rheplicant.inference.identifiability import identifiability
from rheplicant.inference.sensitivity import prior_sensitivity

N_FREQ = 16
NOISE_STD = 0.05


@pytest.fixture(scope="module", autouse=True)
def _float64():
    """The curvature ceiling is read from the dtype, so the dtype is the fixture.

    ``1/sqrt(eps)`` is 2.90e3 in float32 and 6.71e7 in float64, so a
    separation chosen to sit between the two verdicts in one precision sits
    on the wrong side in the other. Every number in this file is a float64
    number.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", was)


class NearCollinearDesign(AbstractOperator):
    """``data = a * g + b * (g + separation * g**2)``.

    A test double with one knob. At ``separation = 0`` the two columns are the
    same vector and the design is exactly rank one; as it grows the second
    column leaves the first one's span, and the curvature's condition number
    falls like ``separation**-2``.

    **Not ``g * (1 + separation)``**, which is what the first draft of the
    probe used: that is a scalar MULTIPLE of the first column, exactly
    collinear at every separation, so it reproduces the rank deficiency of the
    other direction while wearing this one's name.
    """

    requires: ClassVar[tuple[str, ...]] = ("coords.freq",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    a: jax.Array
    b: jax.Array
    separation: float

    def __call__(self, state):
        grid = jnp.linspace(0.0, 1.0, N_FREQ)
        return state.with_data(
            self.a * grid + self.b * (grid + self.separation * grid**2)
        )


def _problem(separation: float):
    pipeline = NearCollinearDesign(
        a=jnp.asarray(1.0), b=jnp.asarray(0.5), separation=separation
    )
    state = State(
        coords=Coordinates(freq=jnp.linspace(60e6, 85e6, N_FREQ)),
        meta={"telescope": "RHINO", "obs_id": "d23-000"},
    )
    space = ParameterSpace(
        latents=[
            Latent("a", init=jnp.asarray(1.0), prior=dist.Normal(0.0, 5.0)),
            Latent("b", init=jnp.asarray(0.5), prior=dist.Normal(0.0, 5.0)),
        ],
        bindings=[
            Bind("a", into=lambda p: p.a),
            Bind("b", into=lambda p: p.b),
        ],
    )
    return space, pipeline, state, pipeline(state).data


class TestTheDirectionAnyoneCanReach:
    """Full observed rank, and a curvature past the ceiling.

    Measured through this package's own public API:

    | separation | identifiability rank | prior_sensitivity |
    |------------|----------------------|-------------------|
    | 1e-1       | 2 of 2               | accepted          |
    | 1e-3       | 2 of 2               | **refused**       |
    | 1e-5       | 2 of 2               | **refused**       |

    The middle two rows are the whole of D23's reachable half: a rank test
    accepts them and the criterion actually running refuses them.
    """

    WELL_CONDITIONED = 1e-1
    DIGIT_STARVED = (1e-3, 1e-5)

    def test_a_well_conditioned_design_is_reported_on(self):
        """The baseline. Without it, every refusal below could be the fixture
        being unusable rather than the criterion firing."""
        report = prior_sensitivity(*_problem(self.WELL_CONDITIONED), NOISE_STD)
        assert report.names == ("a", "b")
        assert np.all(np.isfinite(np.asarray(report.shift_sigma)))

    @pytest.mark.parametrize("separation", DIGIT_STARVED)
    def test_the_observed_jacobian_is_full_rank_there(self, separation):
        """**The half that makes this a discriminating fixture rather than a
        degenerate one.**

        A rank test would accept these models. If the Jacobian were rank
        deficient the refusal below would be one both criteria agree on, and
        this file would pin nothing.
        """
        space, pipeline, state, _ = _problem(separation)
        report = identifiability(space, pipeline, state)
        assert report.rank == report.n_par == 2
        assert report.nullity == 0

    @pytest.mark.parametrize("separation", DIGIT_STARVED)
    def test_the_curvature_criterion_refuses_it_anyway(self, separation):
        with pytest.raises(ParameterSpaceError, match="curvature"):
            prior_sensitivity(*_problem(separation), NOISE_STD)

    @pytest.mark.parametrize("separation", DIGIT_STARVED)
    def test_the_refusal_says_the_jacobian_was_not_the_reason(self, separation):
        """A refusal that named a rank deficiency here would be a false
        statement about the model, and the remote takes care to say so: when
        ``identifiability`` disagrees with the curvature, the message reports
        what was measured instead of borrowing a verdict that does not hold.
        """
        with pytest.raises(ParameterSpaceError) as excinfo:
            prior_sensitivity(*_problem(separation), NOISE_STD)
        message = " ".join(str(excinfo.value).split())
        assert "full-rank" in message
        assert "spectrum" in message

    def test_the_verdict_moves_with_the_conditioning_and_not_with_something_else(self):
        """The knob reaches the answer.

        Two accepted rows and two refused ones separated only by
        ``separation`` -- so the refusal is a statement about the curvature
        rather than about this operator, this prior or this data.
        """
        accepted = prior_sensitivity(*_problem(self.WELL_CONDITIONED), NOISE_STD)
        assert np.all(np.isfinite(np.asarray(accepted.shift_sigma)))
        for separation in self.DIGIT_STARVED:
            with pytest.raises(ParameterSpaceError):
                prior_sensitivity(*_problem(separation), NOISE_STD)


class TestTheDirectionThisPackageCannotReach:
    """``child ~ Normal(parent, s)``: correct on the remote, undeclarable here.

    The remote's argument for the curvature criterion rests on this case, and
    it is a good argument -- a selected latent that reaches no observed node
    can still be held by a downstream latent's density, and its
    likelihood-only mode is perfectly well defined. Measured in a bayesmith
    graph: observed-Jacobian rank 0 of 1, ``prior_sensitivity`` returns
    -0.1096, and a rank test would have refused a legitimate question.

    **This package cannot state it**, and that is what keeps the difference
    one-sided in practice.
    """

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_a_prior_parameterised_by_another_latent_does_not_survive_use(self):
        """It CONSTRUCTS, which is why this test looks at what happens next.

        ``numpyro.distributions.Normal`` takes whatever it is handed, so
        ``dist.Normal(<a Latent>, 0.5)`` builds without complaint. The first
        version of the probe read that as "the shape is reachable" -- a check
        that could not fail. What decides it is using the declaration.
        """
        parent = Latent("parent", init=jnp.asarray(0.0), prior=dist.Normal(0.0, 3.0))
        child = Latent("child", init=jnp.asarray(0.0), prior=dist.Normal(parent, 0.5))
        assert isinstance(child.prior, dist.Normal), "it constructs"
        with pytest.raises(TypeError, match="unsupported operand"):
            child.prior.sample(jax.random.key(0))

    def test_a_declarable_prior_samples_fine(self):
        """The sibling. Without it the refusal above could be `sample` being
        broken rather than the declaration being unstateable."""
        ordinary = Latent("p", init=jnp.asarray(0.0), prior=dist.Normal(0.0, 3.0))
        drawn = ordinary.prior.sample(jax.random.key(0))
        assert bool(jnp.isfinite(drawn))
