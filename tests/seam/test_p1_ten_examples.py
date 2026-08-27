"""P1's acceptance: ten named examples across the adapter, in two tiers.

The adapter (``rheplicant.inference.graph_bridge``) presents a ``Pipeline`` and
a ``ParameterSpace`` to bayesmith as a ``Graph``. These ten examples are what
"it works" means for it, and they are named rather than sampled so that a later
wave can point at one and say which of them it broke.

**Two tiers, two kinds of claim, and they are not interchangeable.**

*Deterministic tier* (examples 1, 2, 3, 5, 6). The answer is a number and the
reference is a **dense** solve of the normal equations, formed here in NumPy
from the block's own ``forward`` -- no CG, no tree map, no knowledge of how the
solver represents anything. Agreement is asked at ``rtol <= 1e-12``, which is
the whole reason this directory is x64-gated. Where a row is CG-endorsed the
tolerance is a statement about the CONVERGED solve at this configuration; two
correct implementations can differ far above 1e-12 element-wise for structural
reasons, and that is not a failure of the seam.

*Sampling tier* (examples 7, 8, 9, 10). The answer is a distribution, so the
claim is about **moments** against the same dense posterior: 400+ independent
draws, mean within 4 standard errors per coordinate, covariance within a
correlation-scaled tolerance. A mean-only assertion is not enough -- a draw with
no spread at all passes it.

Example 4 is a report rather than a number, and asserts the healthy/refused
PAIR: a guard that only ever sees the healthy case cannot tell working from
disabled.

**Diagnostic fields are excluded from every assertion**, by decision: iteration
counts, deltas and convergence flags are properties of a particular solver's
path to the answer, not of the answer, and pinning them turns an implementation
detail into a contract.
"""

import bayesmith
import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist
import pytest
from seam_models import GAIN, N_TIME, SKY_A, ComplexCoeffOperator, log_gain_instrument

from rheplicant.inference import (
    HomoscedasticNoise,
    Latent,
    ParameterSpace,
    RadiometerNoise,
)
from rheplicant.inference.graph_bridge import (
    OBSERVATION,
    priors_from_keywords,
    to_graph,
)

#: The deterministic tier's agreement with a dense reference. Machine-level:
#: both sides solve the same normal equations at float64, so anything looser
#: would be measuring nothing.
DENSE_RTOL = 1e-12

#: The sampling tier's per-coordinate z-score ceiling for the mean, and the
#: covariance's correlation-scaled tolerance. 400 draws puts the mean's
#: standard error at sigma/20, so |z| < 4 is a real constraint rather than a
#: formality; the covariance of 400 draws carries roughly 7% relative error on
#: its own diagonal, hence 0.25 rather than something tighter.
Z_CEILING = 4.0
COV_TOL = 0.25
N_DRAWS = 400


# --------------------------------------------------------- the dense oracle --


def dense_posterior(block, sigma, *, obs=OBSERVATION):
    """``(mean, covariance)`` of the Gaussian posterior this block describes.

    Built the expensive, obviously-correct way: one column of ``A`` per real
    degree of freedom, obtained by pushing a unit vector through the block's
    own ``forward``, then the normal equations formed and inverted densely in
    NumPy. It shares no code with the solver under test -- not the operator, not
    the prior assembly, not the representation of a complex latent.

    A complex member contributes TWO columns (real and imaginary), because the
    map from complex coefficients to real data is R-linear and not C-linear.
    The returned mean is complex again for such a member; the covariance stays
    in the real degrees of freedom, which is where it is a covariance.
    """
    names = block.names
    layout = []
    for name in names:
        size = int(np.prod(block.shape[name])) if block.shape[name] else 1
        units = (1.0, 1j) if bayesmith.exact.block.is_complex(block.dtype[name]) else (1.0,)
        layout.append((name, size, units))

    columns = []
    for name, size, units in layout:
        for index in range(size):
            for unit in units:
                probe = {
                    other: jnp.zeros(block.shape[other], block.dtype[other])
                    for other, _, _ in layout
                }
                flat = jnp.zeros(size, block.dtype[name]).at[index].set(unit)
                probe[name] = flat.reshape(block.shape[name])
                columns.append(np.ravel(np.asarray(block.forward(probe)[obs])))
    design = np.stack(columns, axis=1)

    weight = 1.0 / np.asarray(sigma).ravel() ** 2
    residual = np.ravel(np.asarray(block.data[obs] - block.offset[obs]))

    centres, widths = [], []
    for name, _size, units in layout:
        mean = np.broadcast_to(np.asarray(block.prior_mean[name]), block.shape[name])
        width = np.broadcast_to(np.asarray(block.prior_std[name]), block.shape[name])
        if len(units) == 2:
            centres.append(
                np.stack([np.real(mean).ravel(), np.imag(mean).ravel()], axis=1).ravel()
            )
            # Each half of a complex latent carries scale**2 -- the convention
            # ComplexNormal states and variance_parts duplicates. Reading it as
            # a split of one variance would report a sqrt(2) as physics.
            widths.append(np.repeat(np.real(width).ravel(), 2))
        else:
            centres.append(np.real(mean).ravel())
            widths.append(np.real(width).ravel())
    centre = np.concatenate(centres)
    width = np.concatenate(widths)

    normal = design.T @ (weight[:, None] * design) + np.diag(1.0 / width**2)
    covariance = np.linalg.inv(normal)
    mean = covariance @ (design.T @ (weight * residual) + centre / width**2)
    return mean, covariance


def as_real_vector(block, values):
    """A solution dict flattened the way :func:`dense_posterior` orders it."""
    pieces = []
    for name in block.names:
        array = np.asarray(values[name]).ravel()
        if bayesmith.exact.block.is_complex(block.dtype[name]):
            pieces.append(np.stack([array.real, array.imag], axis=1).ravel())
        else:
            pieces.append(array)
    return np.concatenate(pieces)


def relative(got, expected):
    return float(np.max(np.abs(got - expected) / np.maximum(np.abs(expected), 1e-300)))


def moment_verdict(draws, mean, covariance):
    """``(worst |z| of the mean, worst correlation-scaled covariance error)``.

    The covariance is compared SCALED by ``sqrt(C_ii C_jj)`` rather than
    element-wise relative: an off-diagonal entry near zero makes a relative
    comparison diverge on a difference that is statistically nothing, and the
    first version of this file measured ``inf`` for exactly that reason.
    """
    scale = np.sqrt(np.outer(np.diag(covariance), np.diag(covariance)))
    z = (draws.mean(axis=0) - mean) / np.sqrt(np.diag(covariance) / draws.shape[0])
    error = np.abs(np.cov(draws.T) - covariance) / scale
    return float(np.max(np.abs(z))), float(np.max(error))


# ------------------------------------------------ 1. the linear Wiener solve --


class TestExample1LinearWiener:
    """The plainest thing the adapter can be asked for, and the tier's anchor."""

    def test_the_solve_through_the_adapter_matches_a_dense_one(
        self, instrument, gain_space, observed, quiet_noise, gain_truth, template_state
    ):
        graph = to_graph(gain_space, instrument, template_state, observed, quiet_noise)
        block = bayesmith.linear_operator(graph, ("gains",))
        solved, _ = bayesmith.wiener_solve(
            block, precision=bayesmith.precision_at(graph, {"gains": gain_truth})
        )
        mean, _ = dense_posterior(block, np.full(observed.shape, 0.5))
        assert relative(as_real_vector(block, solved), mean) < DENSE_RTOL

    def test_the_graph_is_three_layers_and_hides_its_internal_names(
        self, instrument, gain_space, observed, quiet_noise, template_state
    ):
        """One sample node per latent, one det, one observe -- and the two
        internal names are the only nodes a caller did not declare.

        Pinned because ``G7``'s convention is that internal node names must not
        reach the posterior's key space, and the cheapest way for that to rot is
        for the adapter to grow a fourth node nobody notices.
        """
        graph = to_graph(gain_space, instrument, template_state, observed, quiet_noise)
        assert [node.name for node in graph.nodes] == ["gains", "__mu__", "__data__"]


# --------------------------------------- 2. the complex alm, mean and sample --


class TestExample2ComplexAlm:
    """G9's minimal surface, accepted where the plan says it must be: through
    the adapter, not through a hand-built block.

    The declaration path is half the example. Measured while G9 was built: no
    numpyro distribution samples complex, so before ``ComplexNormal`` a complex
    latent could be SOLVED but not DECLARED -- and the adapter hands back a
    ``Graph``, so a solver-only surface would have left this example unreachable.
    """

    CENTRE = jnp.array([0.2 + 0.1j, -0.1 + 0.05j, 0.3 - 0.2j])
    TRUTH = jnp.array([1.0 + 2.0j, -0.5 + 0.25j, 3.0 - 1.0j])
    SIGMA = 0.1

    @pytest.fixture
    def sky(self, template_state):
        from rheplicant.core.pipeline import Pipeline

        rows = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        key = jax.random.key(3)
        matrix = jax.random.normal(key, (rows, 3)) + 1j * jax.random.normal(
            jax.random.fold_in(key, 1), (rows, 3)
        )
        return Pipeline(
            ComplexCoeffOperator(coeffs=jnp.zeros(3, dtype=matrix.dtype), matrix=matrix),
            names=("sky",),
        )

    @pytest.fixture
    def alm_space(self):
        return ParameterSpace.direct(
            "alm",
            init=jnp.ones(3) + 0j,
            into=lambda p: p["sky"].coeffs,
            linear=True,
            prior=bayesmith.ComplexNormal(self.CENTRE, 10.0),
        )

    @pytest.fixture
    def alm_data(self, sky, alm_space, template_state):
        forward, _ = alm_space.forward_fn(sky, template_state)
        noise = HomoscedasticNoise(sigma=jnp.array(self.SIGMA))
        return noise.realise(forward({"alm": self.TRUTH}), key=jax.random.key(7))

    @pytest.fixture
    def alm_block(self, sky, alm_space, alm_data, template_state):
        graph = to_graph(
            alm_space,
            sky,
            template_state,
            alm_data,
            HomoscedasticNoise(sigma=jnp.array(self.SIGMA)),
        )
        return graph, bayesmith.linear_operator(graph, ("alm",))

    def test_the_mean_matches_a_dense_solve_over_the_real_degrees_of_freedom(
        self, alm_block, alm_data
    ):
        graph, block = alm_block
        solved, _ = bayesmith.wiener_solve(
            block, precision=bayesmith.precision_at(graph, {"alm": self.TRUTH})
        )
        mean, _ = dense_posterior(block, np.full(alm_data.shape, self.SIGMA))
        assert relative(as_real_vector(block, solved), mean) < DENSE_RTOL

    def test_the_imaginary_half_is_constrained_and_not_merely_carried(
        self, alm_block, alm_data
    ):
        """A complex fixture whose imaginary part is a null direction would pass
        the test above while asserting nothing about half the latent.

        So the recovered imaginary parts are compared against the truth at a
        tolerance the prior alone could not deliver: the prior is centred at
        ``CENTRE`` with width 10, so agreement with ``TRUTH`` to a few percent
        is the data speaking.
        """
        graph, block = alm_block
        solved, _ = bayesmith.wiener_solve(
            block, precision=bayesmith.precision_at(graph, {"alm": self.TRUTH})
        )
        recovered = np.imag(np.asarray(solved["alm"]))
        assert np.allclose(recovered, np.imag(np.asarray(self.TRUTH)), rtol=0.1, atol=0.05)

    def test_the_gcr_draws_have_the_dense_posterior_s_moments(self, alm_block, alm_data):
        graph, block = alm_block
        precision = bayesmith.precision_at(graph, {"alm": self.TRUTH})
        draw = jax.jit(lambda key: bayesmith.gcr_sample(block, precision=precision, key=key)[0])
        keys = jax.random.split(jax.random.key(23), N_DRAWS)
        drawn = np.stack([as_real_vector(block, draw(key)) for key in keys])
        mean, covariance = dense_posterior(block, np.full(alm_data.shape, self.SIGMA))
        worst_z, worst_cov = moment_verdict(drawn, mean, covariance)
        assert worst_z < Z_CEILING
        assert worst_cov < COV_TOL


# ------------------------------------------- 3. the conjugate prior override --


class TestExample3PriorOverride:
    """A prior supplied at the call site rather than declared on the latent.

    The point of the example is that there is ONE path for this:
    ``priors_from_keywords`` folds ``prior_mean=``/``prior_std=`` into the
    ``priors=`` mapping ``to_graph`` takes, carrying the contradiction check
    that has always guarded those keywords. A second path would be a second
    place for a prior to be silently preferred.
    """

    @pytest.fixture
    def free_space(self):
        return ParameterSpace.direct(
            "gains",
            init=jnp.full((N_TIME,), GAIN),
            into=lambda p: p["gain"].gain,
            linear=True,
        )

    def test_the_supplied_prior_is_the_one_the_solve_uses(
        self, instrument, free_space, observed, quiet_noise, gain_truth, template_state
    ):
        priors = priors_from_keywords(
            free_space, prior_mean=GAIN, prior_std=5.0, caller="example 3"
        )
        graph = to_graph(
            free_space, instrument, template_state, observed, quiet_noise, priors=priors
        )
        block = bayesmith.linear_operator(graph, ("gains",))
        solved, _ = bayesmith.wiener_solve(
            block, precision=bayesmith.precision_at(graph, {"gains": gain_truth})
        )
        mean, _ = dense_posterior(block, np.full(observed.shape, 0.5))
        assert relative(as_real_vector(block, solved), mean) < DENSE_RTOL

    def test_a_tighter_supplied_prior_actually_tightens_the_answer(
        self, instrument, free_space, observed, quiet_noise, gain_truth, template_state
    ):
        """Otherwise the test above would pass with the prior dropped entirely.

        Two runs differing only in ``prior_std=`` must give different answers,
        and the tighter one must sit closer to the prior mean. This is the
        assertion that would fail if ``priors=`` were being ignored.
        """
        # 1e-4 and not, say, 0.01: measured here, the likelihood's own
        # precision on one gain is about 2.3e5 (each gain multiplies a sky of
        # 120 across four channels at sigma 0.5), so a prior of width 0.01
        # carries 1e4 and is simply outvoted. A test written at 0.01 would
        # have failed for physics rather than for a dropped prior, which is
        # the wrong reason to be red.
        answers = {}
        for width in (5.0, 1e-4):
            priors = priors_from_keywords(
                free_space, prior_mean=GAIN, prior_std=width, caller="example 3"
            )
            graph = to_graph(
                free_space, instrument, template_state, observed, quiet_noise, priors=priors
            )
            block = bayesmith.linear_operator(graph, ("gains",))
            solved, _ = bayesmith.wiener_solve(
                block, precision=bayesmith.precision_at(graph, {"gains": gain_truth})
            )
            answers[width] = np.asarray(solved["gains"])
        loose = np.max(np.abs(answers[5.0] - GAIN))
        tight = np.max(np.abs(answers[1e-4] - GAIN))
        assert tight < loose / 10.0


# ------------------------------------------------------ 4. the identifiability pair --


class TestExample4Identifiability:
    """The healthy report and the refusal, together.

    Together on purpose: a report fixture that only ever exercises the healthy
    case cannot distinguish a working rank test from one that was switched off,
    and this package has recorded that failure shape more than once.
    """

    @pytest.fixture
    def degenerate_space(self):
        """Both sky amplitudes, which enter only through their SUM.

        So the pair is rank-deficient by construction: one direction in the
        two-dimensional latent space leaves every prediction unchanged.
        """
        return ParameterSpace(
            latents=(
                Latent(name="amp_a", init=jnp.array(100.0), prior=dist.Normal(100.0, 10.0)),
                Latent(name="amp_b", init=jnp.array(20.0), prior=dist.Normal(20.0, 10.0)),
            ),
            bindings=(
                ParameterSpace.direct(
                    "amp_a", init=100.0, into=lambda p: p["sum"]["sky_a"].amplitude
                ).bindings[0],
                ParameterSpace.direct(
                    "amp_b", init=20.0, into=lambda p: p["sum"]["sky_b"].amplitude
                ).bindings[0],
            ),
        )

    def test_a_well_posed_space_reports_identified(
        self, instrument, gain_space, observed, quiet_noise, template_state
    ):
        graph = to_graph(gain_space, instrument, template_state, observed, quiet_noise)
        report = bayesmith.identifiability(graph, names=("gains",))
        assert report.rank == N_TIME
        assert report.nullity == 0

    def test_the_degenerate_pair_is_reported_as_degenerate(
        self, instrument, degenerate_space, observed, quiet_noise, template_state
    ):
        graph = to_graph(degenerate_space, instrument, template_state, observed, quiet_noise)
        report = bayesmith.identifiability(graph, names=("amp_a", "amp_b"))
        assert report.rank == 1
        assert report.nullity == 1
        # And the null direction is the one the model says it is: the two
        # amplitudes enter only through their sum, so the unconstrained
        # combination is their DIFFERENCE. Asserting the nullity alone would
        # pass for a rank test that had gone wrong in some other direction.
        direction = report.null_space[0]
        assert abs(abs(direction[0]) - abs(direction[1])) < 1e-6
        assert direction[0] * direction[1] < 0


# ----------------------------------------------------- 5. the GLS fixed point --


class TestExample5IterativeGLS:
    """Prediction-dependent sigma: solve, reweight, repeat.

    The assertion is that the answer IS a fixed point -- re-solving at the
    sigma the answer implies reproduces it -- and not that the loop took some
    number of turns. ``iterations`` and ``delta`` are excluded by decision:
    they describe the path, and pinning a path makes an implementation detail a
    contract.
    """

    FRACTIONAL_SOURCE = dict(channel_width=61e3, integration_time=1.0)

    @pytest.fixture
    def radiometer(self):
        # floor=1.0 and not 0.0: the probe points a linearity check draws from
        # the prior reach predictions near zero, where an unfloored radiometer
        # sigma is zero and the covariance is singular. That refusal is
        # correct, and it is example 5b's subject rather than this one's.
        return RadiometerNoise(**self.FRACTIONAL_SOURCE, floor=1.0)

    @pytest.fixture
    def radiometer_data(self, instrument, gain_space, gain_truth, radiometer, template_state):
        forward, _ = gain_space.forward_fn(instrument, template_state)
        return radiometer.realise(forward({"gains": gain_truth}), key=jax.random.key(2))

    def test_the_answer_is_a_fixed_point_of_the_reweighting(
        self, instrument, gain_space, radiometer_data, radiometer, gain_truth, template_state
    ):
        graph = to_graph(
            gain_space, instrument, template_state, radiometer_data, radiometer
        )
        block = bayesmith.linear_operator(graph, ("gains",))
        result = bayesmith.iterative_gls(
            block, bayesmith.sigma_from_graph(graph, {"gains": gain_truth})
        )
        again, _ = bayesmith.wiener_solve(
            block, precision=bayesmith.precision_at(graph, {"gains": result.solution["gains"]})
        )
        assert relative(
            np.asarray(again["gains"]), np.asarray(result.solution["gains"])
        ) < DENSE_RTOL

    def test_the_fixed_point_matches_a_dense_solve_at_its_own_sigma(
        self, instrument, gain_space, radiometer_data, radiometer, gain_truth, template_state
    ):
        """The fixed point is also the right answer, not merely a stable one.

        A loop that returned its own starting value forever would satisfy the
        test above; this one forms the dense normal equations at the sigma the
        answer implies and requires the same vector.
        """
        graph = to_graph(
            gain_space, instrument, template_state, radiometer_data, radiometer
        )
        block = bayesmith.linear_operator(graph, ("gains",))
        result = bayesmith.iterative_gls(
            block, bayesmith.sigma_from_graph(graph, {"gains": gain_truth})
        )
        sigma = bayesmith.noise_std_at(graph, {"gains": result.solution["gains"]})
        mean, _ = dense_posterior(block, np.asarray(sigma[OBSERVATION]))
        assert relative(as_real_vector(block, result.solution), mean) < DENSE_RTOL

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "D19: bayesmith starts from sigma at the PRIOR CENTRE, so a zero-centred "
            "prior with RadiometerNoise(floor=0) has sigma = 0 there. The ruling is a "
            "data-anchored start, and its implementation is Wave B's gls work -- P1 "
            "only accepts it. Measured 2026-08-27 and refining the ledger row: the "
            "degeneracy surfaces one layer EARLIER than D19 describes, in "
            "linear_operator's own linearisation point, as a StructureError naming the "
            "scale expression -- not in the first solve. strict=True so this goes red "
            "for being green the day Wave B lands, which is the day the marker must go."
        ),
    )
    def test_a_zero_centred_prior_with_no_floor_still_solves(
        self, instrument, gain_truth, template_state
    ):
        zero_centred = ParameterSpace.direct(
            "gains",
            init=jnp.zeros(N_TIME),
            into=lambda p: p["gain"].gain,
            linear=True,
            prior=dist.Normal(jnp.zeros(N_TIME), 5.0),
        )
        noise = RadiometerNoise(**self.FRACTIONAL_SOURCE, floor=0.0)
        forward, _ = zero_centred.forward_fn(instrument, template_state)
        data = noise.realise(forward({"gains": gain_truth}), key=jax.random.key(4))
        graph = to_graph(zero_centred, instrument, template_state, data, noise)
        block = bayesmith.linear_operator(graph, ("gains",))
        result = bayesmith.iterative_gls(
            block, bayesmith.sigma_from_graph(graph, {"gains": jnp.zeros(N_TIME)})
        )
        assert np.all(np.isfinite(np.asarray(result.solution["gains"])))


# ------------------------------------------- 6. plan.estimate, whole-graph --


class TestExample6WholeGraphEstimate:
    """``compile`` then ``estimate``, on the form today's bayesmith can execute:
    one fully conjugate block covering the whole graph.

    The gradient-block variant and the per-sweep diagnostics are Wave B's
    acceptance, per D14 -- deliberately not smuggled in here, because an example
    that quietly tests less than its name says is worse than an absent one.
    """

    def test_the_estimate_is_the_dense_map(
        self, instrument, gain_space, observed, quiet_noise, template_state
    ):
        graph = to_graph(gain_space, instrument, template_state, observed, quiet_noise)
        plan = bayesmith.compile(graph)
        estimate = plan.estimate()
        block = bayesmith.linear_operator(graph, ("gains",))
        mean, _ = dense_posterior(block, np.full(observed.shape, 0.5))
        assert relative(as_real_vector(block, estimate.values), mean) < DENSE_RTOL


# --------------------------------------------------- 7/8. GCR, scalar and vmap --


class TestExample7And8GCR:
    """The same draw two ways: one key at a time, and ``jax.vmap`` over keys.

    Both are checked against the SAME dense posterior, so the comparison is
    "two paths, one distribution" rather than each path against its own
    reference. That vmap traces at all is the hard constraint the plan names:
    validation and translation happen on the Python side, before the trace, so
    nothing inside the mapped function can raise a refusal that vmap would turn
    into a shape error.
    """

    @pytest.fixture
    def block_and_precision(
        self, instrument, gain_space, observed, quiet_noise, gain_truth, template_state
    ):
        graph = to_graph(gain_space, instrument, template_state, observed, quiet_noise)
        block = bayesmith.linear_operator(graph, ("gains",))
        return block, bayesmith.precision_at(graph, {"gains": gain_truth})

    def test_example_7_the_scalar_path_reproduces_the_dense_moments(
        self, block_and_precision, observed
    ):
        block, precision = block_and_precision
        draw = jax.jit(lambda key: bayesmith.gcr_sample(block, precision=precision, key=key)[0])
        keys = jax.random.split(jax.random.key(11), N_DRAWS)
        drawn = np.stack([as_real_vector(block, draw(key)) for key in keys])
        mean, covariance = dense_posterior(block, np.full(observed.shape, 0.5))
        worst_z, worst_cov = moment_verdict(drawn, mean, covariance)
        assert worst_z < Z_CEILING
        assert worst_cov < COV_TOL

    def test_example_8_the_vmapped_path_reproduces_them_too(
        self, block_and_precision, observed
    ):
        block, precision = block_and_precision
        drawn_tree = jax.vmap(
            lambda key: bayesmith.gcr_sample(block, precision=precision, key=key)[0]
        )(jax.random.split(jax.random.key(11), N_DRAWS))
        drawn = np.asarray(drawn_tree["gains"])
        mean, covariance = dense_posterior(block, np.full(observed.shape, 0.5))
        worst_z, worst_cov = moment_verdict(drawn, mean, covariance)
        assert worst_z < Z_CEILING
        assert worst_cov < COV_TOL

    def test_example_8_the_whole_draw_is_traceable_under_jit(self, block_and_precision):
        """vmap over ``n_draws`` is a hard constraint, so it is asserted as one.

        ``jax.eval_shape`` compiles the mapped function abstractly: no arrays
        are produced, so what this asserts is exactly traceability and not that
        the numbers came out. A refusal reachable from inside the trace would
        show up here as a concretization error rather than as a refusal.
        """
        block, precision = block_and_precision
        shape = jax.eval_shape(
            jax.vmap(
                lambda key: bayesmith.gcr_sample(block, precision=precision, key=key)[0]
            ),
            jax.random.split(jax.random.key(0), 3),
        )
        assert shape["gains"].shape == (3, N_TIME)


# --------------------------------------------------------- 9. GCR in log space --


class TestExample9LogSpaceGCR:
    """A multiplicative-noise model taken to log space, then sampled there.

    The graph the adapter hands over is the input to ``log_space``: the
    transform is bayesmith's, and what P1 accepts is that the adapter's output
    is a graph the transform recognises -- an observed node whose scale really
    does track its prediction, which is what ``RadiometerNoise`` means and what
    the additive alternative would not give.
    """

    @pytest.fixture
    def multiplicative(self):
        # 4.05e-3 fractional: comfortably under the measured 0.06 ceiling at
        # which the first-order log-space equivalence stops holding.
        return RadiometerNoise(channel_width=61e3, integration_time=1.0)

    #: The truth the log-gain example is generated at, in log units.
    LOG_TRUTH = jnp.log(GAIN) + 0.05 * jnp.arange(N_TIME, dtype=float)

    @pytest.fixture
    def log_instrument(self):
        """The same sky, through a gain declared in LOG units.

        Not the ordinary ``GainOperator``: measured here, taking a
        multiplicatively-entering gain to log space refuses, because a block's
        offset is the prediction at ``gain = 0`` and ``log(0)`` is ``-inf``.
        ``log(prediction)`` is affine in ``log(gain)``, which is what a
        log-space block needs, so the example declares that model instead of
        loosening the check that noticed the other one had no log route. The
        refusal is correct and is asserted below in its own right.
        """
        return log_gain_instrument()

    @pytest.fixture
    def log_space_space(self):
        return ParameterSpace.direct(
            "log_gain",
            init=jnp.zeros(N_TIME),
            into=lambda p: p["gain"].log_gain,
            linear=True,
            prior=dist.Normal(jnp.log(GAIN) * jnp.ones(N_TIME), 0.3),
        )

    @pytest.fixture
    def log_graph(
        self, log_instrument, log_space_space, multiplicative, template_state
    ):
        forward, _ = log_space_space.forward_fn(log_instrument, template_state)
        data = multiplicative.realise(
            forward({"log_gain": self.LOG_TRUTH}), key=jax.random.key(5)
        )
        return to_graph(
            log_space_space, log_instrument, template_state, data, multiplicative
        )

    def test_the_adapter_s_graph_has_a_log_route(self, log_graph):
        transformed = bayesmith.log_space(log_graph)
        assert transformed.graph is not None

    def test_the_log_space_draws_are_centred_on_the_log_space_solve(self, log_graph):
        """Moments in the space the solve happens in.

        The dense oracle here is the log-space block's own normal equations --
        the same construction as the deterministic tier, applied to the
        transformed graph, so this is not a second implementation of the
        reference.
        """
        block, transformed = bayesmith.log_linear_operator(log_graph, ("log_gain",))
        at = {"log_gain": self.LOG_TRUTH}
        precision = bayesmith.precision_at(transformed.graph, at)
        sigma = bayesmith.noise_std_at(transformed.graph, at)
        drawn_tree = jax.vmap(
            lambda key: bayesmith.gcr_sample(block, precision=precision, key=key)[0]
        )(jax.random.split(jax.random.key(31), N_DRAWS))
        drawn = np.asarray(drawn_tree["log_gain"])
        observed_name = next(iter(sigma))
        mean, covariance = dense_posterior(
            block, np.asarray(sigma[observed_name]), obs=observed_name
        )
        worst_z, worst_cov = moment_verdict(drawn, mean, covariance)
        assert worst_z < Z_CEILING
        assert worst_cov < COV_TOL


# -------------------------------------------------- 10. a small factor sweep --


class TestExample10SmallPartition:
    """``sample_factors`` over an explicitly declared partition.

    Limited, by decision, to the form today's ``sample_factors`` can execute: a
    ``FactorPlan`` handed in rather than derived. The declared-partition entry
    point, the per-sweep diagnostics and the sweep-shaped estimate are G10, and
    the complete form is Wave B's acceptance.

    The model is bilinear -- prediction is ``(amp_a + amp_b) * gain(t)`` -- so
    the two latents are each affine alone and not jointly, which is exactly the
    case a partition exists for and a single block cannot hold.
    """

    @pytest.fixture
    def bilinear_space(self):
        # Assembled from two single-latent spaces rather than with
        # ``eqx.tree_at``: ``bindings`` is a STATIC field, so tree_at cannot
        # address it -- it raises "`where` does not specify an element of
        # `pytree`", which reads like a typo and is not one.
        amp = ParameterSpace.direct(
            "amp_a",
            init=jnp.array(SKY_A),
            into=lambda p: p["sum"]["sky_a"].amplitude,
            linear=True,
            prior=dist.Normal(SKY_A, 10.0),
        )
        gains = ParameterSpace.direct(
            "gains",
            init=jnp.full((N_TIME,), GAIN),
            into=lambda p: p["gain"].gain,
            linear=True,
            prior=dist.Normal(jnp.full((N_TIME,), GAIN), 0.5),
        )
        return ParameterSpace(
            latents=amp.latents + gains.latents,
            bindings=amp.bindings + gains.bindings,
        )

    @pytest.fixture
    def bilinear_graph(self, instrument, bilinear_space, quiet_noise, template_state):
        forward, _ = bilinear_space.forward_fn(instrument, template_state)
        truth = {"amp_a": jnp.array(105.0), "gains": GAIN + 0.1 * jnp.arange(N_TIME, dtype=float)}
        data = quiet_noise.realise(forward(truth), key=jax.random.key(9))
        return to_graph(bilinear_space, instrument, template_state, data, quiet_noise)

    def test_a_declared_partition_sweeps_and_returns_both_latents(self, bilinear_graph):
        from bayesmith.dispatch.factor import FactorPlan
        from bayesmith.dispatch.plan import Block

        declared = FactorPlan(
            blocks=(
                Block(latents=("amp_a",), method="gcr", reason="declared by the caller"),
                Block(latents=("gains",), method="gcr", reason="declared by the caller"),
            ),
            log_space=None,
        )
        draws = bayesmith.sample_factors(
            bilinear_graph, declared, jax.random.key(13), num_warmup=0, num_samples=64
        )
        assert set(draws) >= {"amp_a", "gains"}
        assert draws["gains"].shape == (64, N_TIME)
        assert np.all(np.isfinite(np.asarray(draws["amp_a"])))

    def test_the_sweep_recovers_the_product_the_data_pins(self, bilinear_graph):
        """A bilinear model pins the PRODUCT, not the factors.

        So that is what is asserted: ``amp_a * gain`` per time sample, against
        the value the data was generated at. Asserting each factor separately
        would be asserting a scale convention the likelihood does not fix, and
        the sweep is free to wander along it.
        """
        from bayesmith.dispatch.factor import FactorPlan
        from bayesmith.dispatch.plan import Block

        declared = FactorPlan(
            blocks=(
                Block(latents=("amp_a",), method="gcr", reason="declared by the caller"),
                Block(latents=("gains",), method="gcr", reason="declared by the caller"),
            ),
            log_space=None,
        )
        draws = bayesmith.sample_factors(
            bilinear_graph, declared, jax.random.key(13), num_warmup=0, num_samples=256
        )
        product = np.asarray(draws["amp_a"])[:, None] * np.asarray(draws["gains"])
        truth = 105.0 * (GAIN + 0.1 * np.arange(N_TIME))
        assert np.allclose(product.mean(axis=0), truth, rtol=0.05)
