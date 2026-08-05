"""A twin that draws its own noise is refused at every inference exit.

The defect this pins: inference closes the forward model over ONE template
state, so an operator consuming the PRNG draws one realisation and adds that
same frozen field to every prediction. The corruption is exactly affine, so
`check_linearity` sees residual 0.0 and `identifiability` reports full rank;
the fitted parameter moves by many sigma and the reported error bar does not
move at all. Both exits of the workflow — the point estimate and the posterior
— are wrong by the same amount.

Both sides of the guard are pinned here: the stochastic twin is refused at all
seven exits, the same twin with the stage dropped is accepted at all seven and
recovers the truth, and the frozen-draw mechanism itself is measured so the
reason survives in the suite rather than only in a commit message.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, Environment, Pipeline, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.inference import (
    Block,
    ParameterSpace,
    SamplingPlan,
    build_forward_fn,
    check_linearity,
    identifiability,
    linear_operator,
    to_numpyro_model,
    wiener_solve,
)
from rheplicant.inference.uncertainty import fisher_information, parameter_covariance
from rheplicant.radio import (
    GainOperator,
    NoiseOperator,
    RFIOperator,
    SkyOperator,
    assemble,
)

N = 8
SKY, TRUTH, SIGMA_MEAS, SIGMA_TWIN = 100.0, 1.1, 2.0, 20.0


@pytest.fixture(scope="module")
def state():
    return State(
        coords=Coordinates(
            time=jnp.linspace(0.0, 7.0, N), freq=jnp.linspace(60e6, 85e6, N)
        ),
        env=Environment(temperature=jnp.array(280.0)),
        key=jax.random.key(0),
    )


@pytest.fixture(scope="module")
def observed(state):
    """Data from the HONEST model at g = 1.1, plus independent 2 K scatter."""
    honest = Pipeline(
        SkyOperator(amplitude=jnp.array(SKY)), GainOperator(gain=jnp.array(TRUTH))
    )
    return honest(state).data + SIGMA_MEAS * jax.random.normal(jax.random.key(7), (N, N))


@pytest.fixture
def space():
    return ParameterSpace.direct(
        "g", init=jnp.array(1.0), into=lambda p: p["gain"].gain,
        prior=dist.Normal(1.0, 10.0), linear=True,
    )


@pytest.fixture
def clean():
    return assemble(
        SkyOperator(amplitude=jnp.array(SKY)), GainOperator(gain=jnp.array(1.0))
    )


@pytest.fixture
def stochastic():
    return assemble(
        SkyOperator(amplitude=jnp.array(SKY)),
        GainOperator(gain=jnp.array(1.0)),
        NoiseOperator(sigma=jnp.array(SIGMA_TWIN)),
    )


def exits(space, twin, state, observed):
    """The seven documented ways into an answer, each as a zero-argument call."""
    plan = SamplingPlan(space, Block("g", engine="conjugate"))
    return {
        "check_linearity": lambda: check_linearity(space, twin, state, "g"),
        "linear_operator": lambda: linear_operator(space, twin, state, "g"),
        "identifiability": lambda: identifiability(space, twin, state),
        "to_numpyro_model": lambda: to_numpyro_model(twin, state, space, SIGMA_MEAS),
        "wiener_solve": lambda: wiener_solve(
            linear_operator(space, twin, state, "g"),
            observed, noise_std=SIGMA_MEAS, prior_std=10.0,
        ),
        "plan.estimate": lambda: plan.estimate(twin, state, observed, noise=SIGMA_MEAS),
        "plan.sample": lambda: plan.sample(
            twin, state, observed, noise=SIGMA_MEAS, key=jax.random.key(1), n_sweeps=200
        ),
    }


class TestEveryExitRefuses:
    @pytest.mark.parametrize(
        "exit_name",
        [
            "check_linearity", "linear_operator", "identifiability", "to_numpyro_model",
            "wiener_solve", "plan.estimate", "plan.sample",
        ],
    )
    def test_exit_refuses_and_names_the_stage(
        self, space, stochastic, state, observed, exit_name
    ):
        with pytest.raises(ParameterSpaceError, match="NoiseOperator at 'noise'"):
            exits(space, stochastic, state, observed)[exit_name]()

    def test_build_forward_fn_refuses_too(self, stochastic, state):
        with pytest.raises(ParameterSpaceError, match="NoiseOperator at 'noise'"):
            build_forward_fn(stochastic, state)

    def test_the_refusal_points_at_the_supported_remedy(self, space, stochastic):
        with pytest.raises(ParameterSpaceError, match=r"Assembly\.without\(node_id\)"):
            space.validate(stochastic)

    def test_an_rfi_stage_is_caught_by_the_same_rule(self, space, state):
        """RFIOperator declares 'key' too and was never pushed through an exit."""
        twin = assemble(
            SkyOperator(amplitude=jnp.array(SKY)),
            RFIOperator(amplitude=jnp.array(50.0), occupancy=0.1),
            GainOperator(gain=jnp.array(1.0)),
        )
        # The label carries the fold path, so a branch source is still located.
        with pytest.raises(ParameterSpaceError, match="RFIOperator at 'field_sum/rfi_field'"):
            space.validate(twin)


class TestTheCleanTwinIsUntouched:
    @pytest.mark.parametrize(
        "exit_name",
        [
            "check_linearity", "linear_operator", "identifiability", "to_numpyro_model",
            "wiener_solve", "plan.estimate", "plan.sample",
        ],
    )
    def test_exit_still_runs(self, space, clean, state, observed, exit_name):
        exits(space, clean, state, observed)[exit_name]()

    def test_the_estimate_recovers_the_truth(self, space, clean, state, observed):
        plan = SamplingPlan(space, Block("g", engine="conjugate"))
        estimate = plan.estimate(clean, state, observed, noise=SIGMA_MEAS)
        assert float(estimate.values["g"]) == pytest.approx(TRUTH, abs=0.02)


class TestTheRefusalHasAnAnswer:
    def test_without_turns_the_refused_twin_into_a_working_one(
        self, space, stochastic, state, observed
    ):
        """The whole point of pairing A1's refusal with A12's `without`."""
        assert "noise" in stochastic.lit
        repaired = stochastic.without("noise")
        assert "noise" not in repaired.lit

        plan = SamplingPlan(space, Block("g", engine="conjugate"))
        estimate = plan.estimate(repaired, state, observed, noise=SIGMA_MEAS)
        assert float(estimate.values["g"]) == pytest.approx(TRUTH, abs=0.02)


class TestTheMechanismTheGuardExistsFor:
    def test_the_draw_is_frozen_across_calls_of_one_closure(self, stochastic, state):
        """Why a stochastic stage cannot be a fit target: `forward` is constant."""
        params, static = eqx.partition(stochastic, eqx.is_inexact_array)

        def forward(p):
            return eqx.combine(p, static)(state).data

        first, second = forward(params), forward(params)
        assert jnp.array_equal(first, second)  # same draw, every single call

        clean_prediction = SKY * 1.0 * jnp.ones((N, N))
        spurious = first - clean_prediction
        # A whole field of it, and its mean is what shifts the fitted gain.
        assert float(jnp.std(spurious)) == pytest.approx(SIGMA_TWIN, rel=0.25)
        assert abs(float(jnp.mean(spurious))) > 0.5

    def test_the_corruption_is_invisible_to_the_checks_that_would_see_a_bug(
        self, space, stochastic, state
    ):
        """Bypass the guard: linearity and rank both pass on the corrupt model.

        This is why the refusal had to be a declaration check — there is no
        numerical symptom for the other diagnostics to find.
        """
        params, static = eqx.partition(stochastic, eqx.is_inexact_array)

        def forward(values):
            model = eqx.combine(params, static)
            return eqx.tree_at(lambda m: m["gain"].gain, model, values["g"])(state).data

        # Exactly affine in g: the frozen field is an offset, not a curvature.
        base = forward({"g": jnp.array(0.0)})
        one = forward({"g": jnp.array(1.0)})
        two = forward({"g": jnp.array(2.0)})
        assert jnp.allclose(two - one, one - base, atol=1e-4)
        # ...and the offset is not zero, which is the whole problem.
        assert not jnp.allclose(base, 0.0)


class TestTheMagnitudeTheDocstringQuotes:
    """The numbers in ``refuse_stochastic_stages``' docstring, executed.

    This file already pinned the MECHANISM -- that every exit refuses a twin
    containing a drawing stage. It did not pin the magnitudes, and the
    docstring quoted 1.1015 -> 1.0824 with an error bar of 0.002451 and 7.8
    sigma, which two independent re-measurements could not reproduce. A number
    in a docstring that nothing executes is exactly the claim this package
    refuses to make anywhere else.

    So the digits live here now, and the docstring points at this class. The
    load-bearing assertion is not the bias -- that is one PRNG realisation --
    but that the two error bars are **equal to every digit**: the estimate
    moves and the diagnostic does not, which is the whole reason a refusal was
    needed rather than a warning.
    """

    @staticmethod
    def _solve(space, twin, state, observed, monkeypatch):
        """One exit, with the refusal lifted so the bias is measurable at all."""
        import rheplicant.inference.parameters as parameters_module

        monkeypatch.setattr(
            parameters_module, "refuse_stochastic_stages", lambda *a, **k: None
        )
        block = linear_operator(space, twin, state, names=("g",))
        value, _ = wiener_solve(block, observed, noise_std=SIGMA_MEAS)
        forward, _ = space.forward_fn(twin, state)
        covariance = parameter_covariance(
            fisher_information(forward, {"g": value["g"]}, noise_std=SIGMA_MEAS)
        )
        return float(value["g"]), float(jnp.sqrt(covariance.matrix[0, 0]))

    def test_the_clean_twin_recovers_the_truth(
        self, space, clean, state, observed, monkeypatch
    ):
        estimate, sigma = self._solve(space, clean, state, observed, monkeypatch)
        assert estimate == pytest.approx(1.100162, abs=5e-6)
        assert sigma == pytest.approx(0.0025000, abs=5e-8)

    def test_the_stochastic_twin_is_biased_by_ten_sigma(
        self, space, stochastic, state, observed, monkeypatch
    ):
        estimate, sigma = self._solve(space, stochastic, state, observed, monkeypatch)
        assert estimate == pytest.approx(1.073513, abs=5e-6)
        assert abs(estimate - TRUTH) / sigma == pytest.approx(10.6, abs=0.1)

    def test_both_error_bars_agree_to_every_digit(
        self, space, clean, stochastic, state, observed, monkeypatch
    ):
        """The assertion the refusal exists for.

        Asserted as exact equality, not `approx`: the point is that no
        diagnostic moves at all, and a tolerance would let a small movement --
        which WOULD be a usable signal -- pass as if it were none.
        """
        _, clean_sigma = self._solve(space, clean, state, observed, monkeypatch)
        _, noisy_sigma = self._solve(space, stochastic, state, observed, monkeypatch)
        assert clean_sigma == noisy_sigma

    def test_the_docstring_quotes_these_digits(self):
        """The two must not drift apart again, which is how this started."""
        import rheplicant.inference.parameters as parameters_module

        doc = parameters_module.refuse_stochastic_stages.__doc__ or ""
        for digits in ("1.100162", "1.073513", "0.0025000", "10.6 sigma"):
            assert digits in doc, digits

