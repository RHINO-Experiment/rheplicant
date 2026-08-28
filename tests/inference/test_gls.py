"""Iterative GLS: finding the covariance a GCR draw is conditioned on.

`gcr_sample` is a linear sampler *given* a covariance. Under the default
radiometer noise the covariance is not given — sigma tracks the prediction, so
the weights and the solution determine each other and have to be found
together. That is all `iterative_gls` does, and nothing about `gcr_sample`
changes: it takes the sigma this produces.

The reference in `_hydra_tod_iterative_gls` below is transcribed from
hydra-tod's `hydra_tod/linear_sampler.py` (`iterative_gls` and
`params_space_oper_and_data`), which is the numpy implementation this port
follows. It is inlined rather than imported because hydra-tod imports `mpi4py`
at module scope — too heavy a dependency to pull into a test for twenty lines
of reference.
"""

from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.core.state import State
from rheplicant.inference import (
    FlaggedNoise,
    HomoscedasticNoise,
    ParameterSpace,
    RadiometerNoise,
    gcr_sample,
    iterative_gls,
    linear_operator,
    wiener_solve,
)
from rheplicant.inference import gls as gls_module

N_DATA, N_PAR = 64, 4
OFFSET = 300.0  # "mu" in hydra-tod's d = (U p + mu)(1 + n)
CHANNEL_WIDTH, INTEGRATION_TIME = 1e4, 1.0  # fractional noise 1e-2
PRIOR = 1e3  # weak enough that the reference's prior-free solve is comparable


# --------------------------------------------------------------------------
# Reference: hydra-tod's iterative GLS, transcribed.
# --------------------------------------------------------------------------
def _hydra_tod_params_space(d, U, p, N_inv, mu):
    """`hydra_tod.linear_sampler.params_space_oper_and_data`, verbatim."""
    D_p_inv = 1.0 / (U @ p + mu)
    sigma_inv = N_inv * np.outer(D_p_inv, D_p_inv)
    aux = U.T @ sigma_inv
    return aux @ U, aux @ (d - mu)


def _hydra_tod_iterative_gls(d, U, N_inv, mu=0.0, tol=1e-12, min_iter=5, max_iter=100):
    """`hydra_tod.linear_sampler.iterative_gls`, transcribed (dense solve)."""
    p = np.linalg.lstsq(U, d - mu, rcond=None)[0]
    for iteration in range(1, max_iter + 1):
        A, b = _hydra_tod_params_space(d, U, p, N_inv, mu=mu)
        p_new = np.linalg.solve(A, b)
        if np.linalg.norm(p_new - p) < tol * np.linalg.norm(p) and iteration >= min_iter:
            break
        p = p_new
    D_p_inv = 1.0 / (U @ p_new + mu)
    return p_new, N_inv * np.outer(D_p_inv, D_p_inv)


# --------------------------------------------------------------------------
# A block whose design matrix is a genuine dense U, so the reference applies.
# --------------------------------------------------------------------------
class DesignOperator(AbstractOperator):
    """``data -> data + U @ coeffs`` — linear in ``coeffs`` with a real offset."""

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    design: jax.Array = eqx.field(static=False)
    coeffs: jax.Array

    def __call__(self, state: State) -> State:
        return state.with_data(state.data + self.design @ self.coeffs)


@pytest.fixture
def design():
    return jax.random.normal(jax.random.key(7), (N_DATA, N_PAR))


@pytest.fixture
def truth():
    return jnp.array([12.0, -5.0, 3.0, 8.0])


@pytest.fixture
def block(design):
    twin = Pipeline(DesignOperator(design=design, coeffs=jnp.zeros(N_PAR)),
                    names=("design",))
    space = ParameterSpace.direct(
        "coeffs", init=jnp.zeros(N_PAR),
        into=lambda p: p["design"].coeffs, linear=True,
    )
    template = State(data=jnp.full((N_DATA,), OFFSET))
    return linear_operator(space, twin, template)


@pytest.fixture
def noise():
    return RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)


@pytest.fixture
def observed(block, truth, noise, design):
    """Genuinely multiplicative: d = (U p + mu)(1 + w), w ~ N(0, f^2)."""
    clean = block.offset + block.forward(truth)
    w = noise.fractional * jax.random.normal(jax.random.key(2), (N_DATA,))
    return clean * (1.0 + w)


class TestAgainstHydraTod:
    def test_the_solution_matches_the_numpy_reference(
        self, block, observed, noise, design
    ):
        result = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        reference, _ = _hydra_tod_iterative_gls(
            np.asarray(observed, dtype=np.float64),
            np.asarray(design, dtype=np.float64),
            np.eye(N_DATA) / noise.fractional**2,
            mu=OFFSET,
        )
        np.testing.assert_allclose(
            np.asarray(result.solution), reference, rtol=2e-4
        )

    def test_the_covariance_matches_the_numpy_reference(
        self, block, observed, noise, design
    ):
        """The returned sigma IS the reference's Sigma, diagonally."""
        result = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        _, sigma_inv = _hydra_tod_iterative_gls(
            np.asarray(observed, dtype=np.float64),
            np.asarray(design, dtype=np.float64),
            np.eye(N_DATA) / noise.fractional**2,
            mu=OFFSET,
        )
        np.testing.assert_allclose(
            np.asarray(result.noise_std), 1.0 / np.sqrt(np.diag(sigma_inv)), rtol=2e-4
        )


class TestConvergence:
    def test_it_reports_converging(self, block, observed, noise):
        result = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        assert bool(result.converged)
        assert 1 < int(result.iterations) < 20
        assert float(result.delta) < 1e-6

    def test_the_result_is_a_fixed_point(self, block, observed, noise):
        """Re-weighting at the answer must return the answer."""
        result = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        again, _ = wiener_solve(
            block, observed, noise_std=result.noise_std, prior_std=PRIOR
        )
        assert jnp.allclose(again, result.solution, rtol=1e-4)

    def test_constant_noise_needs_no_reweighting_at_all(self, block, observed):
        """``depends_on_prediction`` is a claim a solver acts on, not a hint."""
        noise = HomoscedasticNoise(jnp.asarray(3.0))
        result = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        direct, _ = wiener_solve(block, observed, noise_std=3.0, prior_std=PRIOR)
        assert int(result.iterations) == 1
        assert jnp.allclose(result.solution, direct, rtol=1e-5)

    def test_stopping_early_is_reported_rather_than_hidden(
        self, block, observed, noise
    ):
        result = iterative_gls(
            block, observed, noise=noise, prior_std=PRIOR,
            min_reweights=1, max_reweights=2, reweight_tol=1e-15,
        )
        assert not bool(result.converged)

    def test_a_tolerance_below_the_epsilon_never_converges(
        self, block, observed, noise
    ):
        """Why the default cannot be a fixed number.

        float32's epsilon is 1.2e-7, so a relative step of 1e-8 is rounding
        rather than a measurement. A plausible-looking fixed default runs the
        loop to its cap and reports failure for a run that had settled — this
        test pins the behaviour so the derived default is not quietly undone.

        **What is asserted changed with the Wave B switch, and the claim did
        not.** This used to assert the exit — ``not converged``, ``iterations
        == 100`` — which is an artefact of where the float32 iterate stops
        relative to the fixed point, not a property of the tolerance. Measured:
        the same fixture settles at ``delta = 6.98e-08`` one side of the seam
        and at ``delta = 0.0`` exactly the other, and at exactly zero a 1e-12
        tolerance is trivially met, so the loop converges at step 8 instead of
        running to 100. Both numbers are below float32's epsilon, which is the
        thing that was ever true; the CI triage of 2026-08-28 met the same
        arm64/x86-64 split on a sibling fixture and reached the same reading.
        """
        stuck = iterative_gls(
            block, observed, noise=noise, prior_std=PRIOR, reweight_tol=1e-12
        )
        # The claim is that the REQUESTED step is below what the dtype can
        # measure, so asking for it tells you nothing. That is what is
        # asserted, and it holds however the loop then exits.
        epsilon = float(jnp.finfo(jnp.float32).eps)
        assert float(stuck.delta) < epsilon, (float(stuck.delta), epsilon)

        derived = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        assert bool(derived.converged)
        assert int(derived.iterations) < 20
        assert jnp.allclose(derived.solution, stuck.solution, rtol=1e-5)

    def test_it_is_jittable(self, block, observed, noise):
        run = jax.jit(
            lambda d: iterative_gls(block, d, noise=noise, prior_std=PRIOR).solution
        )
        assert jnp.allclose(
            run(observed),
            iterative_gls(block, observed, noise=noise, prior_std=PRIOR).solution,
            rtol=1e-5,
        )

    def test_a_prior_is_required(self, block, observed, noise):
        with pytest.raises(ParameterSpaceError, match="prior_std"):
            iterative_gls(block, observed, noise=noise, prior_std=None)

    def test_a_minimum_above_the_maximum_is_refused(self, block, observed, noise):
        """Written the other way round this configuration never terminates, and
        an infinite ``lax.while_loop`` under jit cannot be interrupted."""
        with pytest.raises(ParameterSpaceError, match="min_reweights"):
            iterative_gls(
                block, observed, noise=noise, prior_std=PRIOR,
                min_reweights=8, max_reweights=2,
            )


class TestFlagsSurviveTheReweighting:
    def test_a_flagged_sample_stays_weightless_throughout(
        self, block, observed, noise
    ):
        flags = jnp.zeros(N_DATA, bool).at[:8].set(True)
        ruined = observed.at[:8].set(1e6)
        wrapped = FlaggedNoise(noise, flags)
        result = iterative_gls(block, ruined, noise=wrapped, prior_std=PRIOR)
        clean = iterative_gls(
            block, observed, noise=wrapped, prior_std=PRIOR
        )
        # The flagged samples were replaced by garbage; the answer must not move.
        assert jnp.allclose(result.solution, clean.solution, rtol=1e-5)
        assert jnp.all(jnp.isinf(result.noise_std[:8]))


class TestTheDrawAtTheFoundCovariance:
    """`gcr_sample` is unchanged: it takes the sigma iterative_gls produced."""

    def test_the_sample_mean_is_the_gls_solution(self, block, observed, noise):
        result = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        keys = jax.random.split(jax.random.key(0), 600)
        draws = jax.vmap(
            lambda k: gcr_sample(
                block, observed, noise_std=result.noise_std,
                prior_std=PRIOR, key=k,
            )[0]
        )(keys)
        scatter = draws.std(axis=0) / jnp.sqrt(draws.shape[0])
        assert jnp.all(jnp.abs(draws.mean(axis=0) - result.solution) < 4.0 * scatter)

    def test_the_sample_covariance_is_the_posterior_covariance(
        self, block, observed, noise, design
    ):
        result = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        keys = jax.random.split(jax.random.key(1), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(
                block, observed, noise_std=result.noise_std,
                prior_std=PRIOR, key=k,
            )[0]
        )(keys)
        weight = 1.0 / np.asarray(result.noise_std, dtype=np.float64) ** 2
        U = np.asarray(design, dtype=np.float64)
        expected = np.linalg.inv(U.T @ (weight[:, None] * U) + np.eye(N_PAR) / PRIOR**2)
        empirical = np.cov(np.asarray(draws, dtype=np.float64).T)
        # Norm-relative, as elsewhere in the suite: a per-element rtol is a
        # tolerance on the SMALLEST off-diagonal, whose Monte Carlo error is
        # set by the LARGEST diagonal and so can never meet it.
        relative = np.linalg.norm(empirical - expected) / np.linalg.norm(expected)
        assert relative < 0.12, f"covariance off by {relative:.3f}"


class TestASquareSystemMovesOnlyTheWidth:
    """The mean can be weight-independent while the posterior width is not.

    A square linear system has one solution and the weights cancel out of it,
    so reweighting cannot move the estimate at all — this is exact, not a weak
    effect, and it is the situation in ``examples/gls_gcr.py`` (three switched
    loads against three per-channel noise-wave unknowns).

    The posterior covariance ``(A^T Sigma^-1 A + S^-1)^-1`` depends on Sigma
    regardless. Since a GCR draw is precisely a draw of that width, "the point
    estimate came out the same" is not evidence the covariance did not matter.
    """

    SIZE = 4

    @pytest.fixture
    def square(self):
        key = jax.random.key(5)
        # Positive and spread over a decade, so sigma genuinely varies.
        design = jnp.diag(jnp.logspace(0.0, 1.0, self.SIZE)) + 0.1 * jnp.abs(
            jax.random.normal(key, (self.SIZE, self.SIZE))
        )
        twin = Pipeline(
            DesignOperator(design=design, coeffs=jnp.zeros(self.SIZE)),
            names=("design",),
        )
        space = ParameterSpace.direct(
            "coeffs", init=jnp.zeros(self.SIZE),
            into=lambda p: p["design"].coeffs, linear=True,
        )
        template = State(data=jnp.full((self.SIZE,), 50.0))
        return linear_operator(space, twin, template), design

    @pytest.fixture
    def square_data(self, square):
        block, _ = square
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        clean = block.offset + block.forward(jnp.array([9.0, 7.0, 5.0, 3.0]))
        w = noise.fractional * jax.random.normal(jax.random.key(6), (self.SIZE,))
        return clean * (1.0 + w), noise

    def test_the_mean_does_not_depend_on_the_weighting(self, square, square_data):
        block, _ = square
        observed, noise = square_data
        found = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        flat, _ = wiener_solve(
            block, observed,
            noise_std=float(jnp.mean(observed) * noise.fractional), prior_std=PRIOR,
        )
        assert jnp.allclose(found.solution, flat, rtol=1e-3)

    def test_the_posterior_width_does(self, square, square_data):
        block, design = square
        observed, noise = square_data
        found = iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        U = np.asarray(design, dtype=np.float64)
        eye = np.eye(self.SIZE) / PRIOR**2

        def width(sigma):
            weight = 1.0 / np.asarray(sigma, dtype=np.float64) ** 2
            return np.sqrt(np.diag(np.linalg.inv(U.T @ (weight[:, None] * U) + eye)))

        reweighted = width(found.noise_std)
        frozen = width(
            np.full(self.SIZE, float(jnp.mean(observed) * noise.fractional))
        )
        assert np.max(np.abs(frozen / reweighted - 1.0)) > 0.05


class TestWhyItIsNeeded:
    """Freezing sigma at one constant is not a small error.

    Weighting by ``sigma = |prediction| * f`` means bright samples carry less
    weight. Replacing that with one constant throws the weighting away, and on
    a prediction spanning a decade — an ordinary foreground spectrum — the
    difference is a factor in the recovered error, not a rounding.

    The claim is stated over realizations, not on one draw. A single seed
    decides nothing here: the reweighted solve wins about 38 times in 40, so a
    one-draw assertion would be a coin flip five percent of the time and would
    eventually fail for a reason that has nothing to do with the code.
    """

    TRUTH = jnp.array([2.0, 1.0, 0.5, 1.5])
    N_REALIZATIONS = 24

    @pytest.fixture
    def steep(self):
        """A positive prediction spanning a decade, so the weights truly vary.

        ``abs`` on the design keeps every prediction positive: a temperature
        that crosses zero is not a physical model, and near a zero crossing the
        radiometer weight diverges, which would make this comparison look far
        more favourable than the physics warrants.
        """
        ramp = jnp.logspace(0.0, 2.0, N_DATA)
        design = ramp[:, None] * jnp.abs(
            jax.random.normal(jax.random.key(11), (N_DATA, N_PAR))
        )
        twin = Pipeline(
            DesignOperator(design=design, coeffs=jnp.zeros(N_PAR)), names=("design",)
        )
        space = ParameterSpace.direct(
            "coeffs", init=jnp.zeros(N_PAR),
            into=lambda p: p["design"].coeffs, linear=True,
        )
        template = State(data=jnp.full((N_DATA,), 10.0))
        return linear_operator(space, twin, template)

    def test_the_weights_really_do_vary(self, steep):
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        sigma = noise.std(steep.offset + steep.forward(self.TRUTH))
        assert float(sigma.max() / sigma.min()) > 20.0

    def test_freezing_sigma_costs_a_factor_in_the_recovered_error(self, steep):
        noise = RadiometerNoise(CHANNEL_WIDTH, INTEGRATION_TIME)
        clean = steep.offset + steep.forward(self.TRUTH)

        def errors(seed):
            w = noise.fractional * jax.random.normal(jax.random.key(seed), (N_DATA,))
            observed = clean * (1.0 + w)
            found = iterative_gls(steep, observed, noise=noise, prior_std=PRIOR)
            frozen, _ = wiener_solve(
                steep, observed,
                noise_std=jnp.mean(observed) * noise.fractional, prior_std=PRIOR,
            )
            return (
                jnp.linalg.norm(found.solution - self.TRUTH),
                jnp.linalg.norm(frozen - self.TRUTH),
            )

        pairs = [errors(seed) for seed in range(self.N_REALIZATIONS)]
        rms_gls = float(jnp.sqrt(jnp.mean(jnp.array([a for a, _ in pairs]) ** 2)))
        rms_frozen = float(jnp.sqrt(jnp.mean(jnp.array([b for _, b in pairs]) ** 2)))
        assert rms_frozen / rms_gls > 1.5, f"only {rms_frozen / rms_gls:.2f}x"


class TestABareSigmaIsRefusedByName:
    """``iterative_gls`` is the one exit that REQUIRES a model, not accepts one.

    A bare array used to reach ``noise.depends_on_prediction`` and come back as
    ``AttributeError: 'ArrayImpl' object has no attribute
    'depends_on_prediction'`` -- an attribute name the caller never wrote, from
    a layer they were not thinking about.

    The asymmetry is worth stating because it runs the opposite way to every
    other ``noise``/``noise_std`` argument in the package: the conjugate solves
    refuse a MODEL and want an array, this refuses an ARRAY and wants a model.
    Both refusals are the same rule seen from two sides -- whether the exit has
    a prediction at which a prediction-dependent sigma could be evaluated.
    Here the whole subject is the fixed point such a sigma implies, and a
    decided array leaves nothing to iterate.
    """

    def test_a_bare_array_is_refused(self, block, observed):
        with pytest.raises(ParameterSpaceError, match="needs a NoiseModel"):
            iterative_gls(block, observed, noise=0.5, prior_std=1.0)

    def test_the_message_names_the_wrapper_and_the_alternative(self, block, observed):
        """A refusal that names the defect without naming the way out sends the
        reader to the source, which is what the sentence exists to avoid."""
        with pytest.raises(ParameterSpaceError) as excinfo:
            iterative_gls(block, observed, noise=jnp.full((N_DATA,), 0.5), prior_std=1.0)
        message = str(excinfo.value)
        assert "HomoscedasticNoise" in message, message
        assert "wiener_solve" in message, message

    def test_the_wrapped_constant_is_accepted_and_converges_at_once(
        self, block, observed
    ):
        """The remedy the message offers has to work, and to do the right thing.

        A constant sigma has no fixed point to find, so the honest answer is
        one step and ``converged=True`` -- not an error, and not a loop.
        """
        result = iterative_gls(
            block, observed, noise=HomoscedasticNoise(jnp.asarray(0.5)), prior_std=1.0
        )
        assert result.converged


class TestTheKnobsReachTheFarSideExactlyAsWritten:
    """Added 2026-08-28 by the Wave B gls mutation set, which found the gap.

    D50 pinned ``reweight_tol`` one hop UP -- the config layer forwarding it
    into ``inference.iterative_gls`` -- and did so by watching the seam rather
    than a numerical consequence, because no fixture in that family can
    bracket the reweighting's own step. The Wave B switch then created a
    SECOND hop, this module forwarding into ``bayesmith.exact.gls``, and
    nothing watched it: deleting ``reweight_tol=reweight_tol`` from the
    delegation left every one of the eleven files that touch
    ``iterative_gls`` green.

    It survives for a reason worth stating, because it is not "no test
    exercises this". The far side computes its default from **the same
    formula** this module used to -- ``max(REWEIGHT_TOL_EPS * eps, tol)`` --
    so dropping the forward is exactly equivalent whenever the caller passes
    ``None``, and every fixture in this family does. The gap only opens for an
    EXPLICIT tolerance, and the numerical consequence of that is what D50
    already established cannot be bracketed here.

    So the seam is watched directly, the same remedy at the same shape of
    problem, one layer down.
    """

    @staticmethod
    def _spy(monkeypatch):
        seen: dict = {}
        real = gls_module._far_iterative_gls

        def spy(*args, **kwargs):
            seen.update(kwargs)
            return real(*args, **kwargs)

        monkeypatch.setattr(gls_module, "_far_iterative_gls", spy)
        return seen

    def test_an_explicit_reweight_tol_arrives_unchanged(
        self, monkeypatch, block, observed, noise
    ):
        seen = self._spy(monkeypatch)
        iterative_gls(
            block, observed, noise=noise, prior_std=PRIOR, reweight_tol=3.7e-09
        )
        assert seen["reweight_tol"] == 3.7e-09

    def test_no_declared_tolerance_forwards_none_rather_than_a_local_default(
        self, monkeypatch, block, observed, noise
    ):
        """The two sides derive the same default from the same formula, so the
        facade must hand the far side ``None`` and let it derive -- computing
        it here as well would be the second spelling that goes stale."""
        seen = self._spy(monkeypatch)
        iterative_gls(block, observed, noise=noise, prior_std=PRIOR)
        assert seen["reweight_tol"] is None

    @pytest.mark.parametrize(
        "knob,value",
        [("min_reweights", 3), ("max_reweights", 7), ("tol", 1e-5),
         ("maxiter", 11), ("require_convergence", 1e-1)],
    )
    def test_every_other_knob_arrives_unchanged(
        self, monkeypatch, block, observed, noise, knob, value
    ):
        """The mutation set killed each of these through a numerical
        consequence, so they are not unguarded -- but they are guarded
        INDIRECTLY, and a knob that stops being forwarded should say so here
        rather than through whichever fixture happens to notice."""
        seen = self._spy(monkeypatch)
        try:
            iterative_gls(
                block, observed, noise=noise, prior_std=PRIOR, **{knob: value}
            )
        except Exception:  # noqa: BLE001 -- see below
            # Deliberately tolerated. The spy records the keywords BEFORE the
            # far side runs, so a value the far side then refuses -- which
            # `require_convergence=1e-1` is on this block -- has still proved
            # it arrived, and arguably proved it harder: the far side acted on
            # it. Asserting a clean return here would mean choosing values
            # that change nothing, which is the opposite of what a forwarding
            # test wants.
            pass
        assert seen[knob] == value
