"""Tests for uncertainty propagation: Fisher, delta method, pushforward."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.core.errors import StateValidationError
from rheplicant.inference.noise import HomoscedasticNoise, RadiometerNoise
from rheplicant.inference.uncertainty import (
    FlatMatrix,
    fisher_information,
    parameter_covariance,
    propagate_covariance,
    push_forward,
)

N_DATA, N_PAR = 12, 3


@pytest.fixture
def linear_problem():
    """forward(theta) = A @ theta — Fisher and delta method are EXACT here."""
    key = jax.random.key(0)
    A = jax.random.normal(key, (N_DATA, N_PAR))
    theta0 = jnp.array([1.0, -2.0, 0.5])

    def forward(theta):
        return A @ theta

    return forward, theta0, A


class TestFisher:
    def test_matches_analytic_linear(self, linear_problem):
        forward, theta0, A = linear_problem
        sigma = 0.3
        F = fisher_information(forward, theta0, noise_std=sigma)
        expected = A.T @ A / sigma**2
        assert jnp.allclose(F.matrix, expected, rtol=1e-5)

    def test_flags_zero_out_samples(self, linear_problem):
        forward, theta0, A = linear_problem
        flags = jnp.zeros(N_DATA, bool).at[:4].set(True)
        F = fisher_information(forward, theta0, noise_std=1.0, flags=flags)
        expected = A[4:].T @ A[4:]
        assert jnp.allclose(F.matrix, expected, rtol=1e-5)

    def test_flags_shape_mismatch_raises(self, linear_problem):
        forward, theta0, _ = linear_problem
        with pytest.raises(StateValidationError, match="flags"):
            fisher_information(forward, theta0, 1.0, flags=jnp.zeros(3, bool))

    def test_a_noise_model_may_replace_the_bare_sigma(self, linear_problem):
        forward, theta0, A = linear_problem
        bare = fisher_information(forward, theta0, noise_std=0.3)
        model = fisher_information(
            forward, theta0, noise_std=HomoscedasticNoise(jnp.asarray(0.3))
        )
        assert jnp.allclose(bare.matrix, model.matrix, rtol=1e-5)

    def test_an_array_sigma_is_not_mistaken_for_a_noise_model(self, linear_problem):
        """jax arrays have a ``.std`` method; only the protocol's data member
        tells the two apart, so a per-sample sigma must still be a sigma."""
        forward, theta0, A = linear_problem
        sigma = jnp.linspace(0.2, 0.6, N_DATA)
        F = fisher_information(forward, theta0, noise_std=sigma)
        assert jnp.allclose(F.matrix, A.T @ (A / sigma[:, None] ** 2), rtol=1e-5)

    def test_prediction_dependent_noise_adds_the_variance_term(self):
        """For the radiometer model the correction is a closed-form factor.

        ``sigma = |mu| f`` gives ``d log sigma/d theta = A/mu``, so the
        variance term is ``2 sum A A / mu^2`` while ``J^T N^-1 J`` is
        ``(1/f^2) sum A A / mu^2``. The full Fisher is therefore exactly
        ``(1 + 2 f^2)`` times the naive one — and a forecast built on the naive
        one alone reports error bars too wide by ``sqrt(1 + 2 f^2)``.
        """
        key = jax.random.key(4)
        A = jax.random.normal(key, (N_DATA, N_PAR))
        theta0 = jnp.array([1.0, -2.0, 0.5])
        # Offset well clear of zero: sigma is proportional to the prediction,
        # so a prediction near zero is a weight near infinity, not a bug.
        def forward(theta):
            return A @ theta + 300.0

        noise = RadiometerNoise(50.0, 2.0)  # f = 0.1
        full = fisher_information(forward, theta0, noise_std=noise)
        naive = fisher_information(
            forward, theta0, noise_std=noise.std(forward(theta0))
        )
        factor = 1.0 + 2.0 * noise.fractional**2
        assert jnp.allclose(full.matrix, factor * naive.matrix, rtol=1e-4)

    def test_flagged_samples_carry_no_variance_information_either(self):
        key = jax.random.key(4)
        A = jax.random.normal(key, (N_DATA, N_PAR))
        theta0 = jnp.array([1.0, -2.0, 0.5])

        def forward(theta):
            return A @ theta + 300.0

        flags = jnp.zeros(N_DATA, bool).at[:4].set(True)
        noise = RadiometerNoise(50.0, 2.0)
        flagged = fisher_information(forward, theta0, noise_std=noise, flags=flags)
        kept = fisher_information(
            lambda t: forward(t)[4:], theta0, noise_std=noise
        )
        assert jnp.allclose(flagged.matrix, kept.matrix, rtol=1e-4)

    def test_pytree_params(self):
        """Fisher works on structured (pytree) parameter sets."""

        def forward(p):
            return p["gain"] * jnp.arange(1.0, 5.0) + p["offset"]

        params = {"gain": jnp.array(2.0), "offset": jnp.array(0.1)}
        F = fisher_information(forward, params, noise_std=1.0)
        assert F.matrix.shape == (2, 2)
        assert jnp.all(jnp.linalg.eigvalsh(F.matrix) > 0)

    def test_empty_params_rejected(self):
        with pytest.raises(StateValidationError, match="no trainable"):
            fisher_information(lambda p: jnp.zeros(3), {}, 1.0)


class TestCovariancePropagation:
    def test_cramer_rao_roundtrip(self, linear_problem):
        forward, theta0, A = linear_problem
        F = fisher_information(forward, theta0, noise_std=0.5)
        cov = parameter_covariance(F)
        assert jnp.allclose(F.matrix @ cov.matrix, jnp.eye(N_PAR), atol=1e-4)

    def test_delta_method_exact_for_linear(self, linear_problem):
        """For y = A theta: std(y) = sqrt(diag(A Sigma A^T)) exactly."""
        forward, theta0, A = linear_problem
        cov = jnp.diag(jnp.array([0.04, 0.01, 0.09]))
        std = propagate_covariance(forward, theta0, cov)
        expected = jnp.sqrt(jnp.diag(A @ cov @ A.T))
        assert jnp.allclose(std, expected, rtol=1e-5)

    def test_delta_method_matches_monte_carlo(self, linear_problem):
        forward, theta0, _ = linear_problem
        cov = 0.02 * jnp.eye(N_PAR)
        std_delta = propagate_covariance(forward, theta0, cov)

        chol = jnp.linalg.cholesky(cov)
        draws = theta0 + jax.random.normal(jax.random.key(1), (20000, N_PAR)) @ chol.T
        std_mc = push_forward(forward, draws).std(axis=0)
        assert jnp.allclose(std_delta, std_mc, rtol=0.05)

    def test_cov_shape_mismatch_raises(self, linear_problem):
        forward, theta0, _ = linear_problem
        with pytest.raises(StateValidationError, match="param_cov"):
            propagate_covariance(forward, theta0, jnp.eye(N_PAR + 1))

    def test_structure_mismatch_rejected(self, linear_problem):
        """Regression: a covariance from a DIFFERENT parameterization must
        not be silently applied — same size, different flattening order."""
        forward, theta0, _ = linear_problem
        other_params = {"alpha": jnp.zeros(2), "beta": jnp.zeros(1)}
        F_other = fisher_information(
            lambda p: jnp.concatenate([p["alpha"], p["beta"]]) * jnp.ones(3),
            other_params, noise_std=1.0,
        )
        cov_other = parameter_covariance(F_other, jitter=1e-6)
        with pytest.raises(StateValidationError, match="structure"):
            propagate_covariance(forward, theta0, cov_other)

    def test_a_precision_is_refused_rather_than_propagated(self, linear_problem):
        """A new refusal, and the same table :meth:`FlatMatrix.sigma` uses.

        A Fisher matrix and a covariance are the same shape, so this used to
        return a finite, correctly shaped error bar wrong by the square of
        everything. Measured cost of the refusal: across ``tests/inference`` +
        ``tests/config`` + the two x64 sessions, all 23 covariances that reach
        this function are ``kind='covariance'``; none is a precision.
        """
        forward, theta0, _ = linear_problem
        fisher = fisher_information(forward, theta0, noise_std=0.5)
        with pytest.raises(StateValidationError, match="not a covariance"):
            propagate_covariance(forward, theta0, fisher)

    def test_a_posterior_covariance_still_propagates(self, linear_problem):
        """The other side of that rule, because the far side does not have
        this kind at all.

        ``'posterior_covariance'`` IS a covariance; a translation table that
        forgot it would send it across under a name the far side refuses, and
        the config layer's ``predict`` route reads exactly this product from a
        ``width: fisher`` run.
        """
        forward, theta0, _ = linear_problem
        cov = parameter_covariance(
            fisher_information(forward, theta0, noise_std=0.5)
        )
        posterior = FlatMatrix(
            matrix=cov.matrix,
            structure=cov.structure,
            kind="posterior_covariance",
            names=cov.names,
            spans=cov.spans,
            shapes=cov.shapes,
        )
        assert jnp.allclose(
            propagate_covariance(forward, theta0, posterior),
            propagate_covariance(forward, theta0, cov.matrix),
            rtol=1e-6,
        )


class TestTheSynthesisedGraphCannotReachTheAnswer:
    """``propagate_covariance`` gives the far side a graph, and a graph needs
    a noise model and data that the delta method has no use for.

    The legality of synthesising them is one sentence -- the delta method
    reads the Jacobian of the prediction and the covariance, and neither the
    residual nor the weighting appears in ``sqrt(diag(J Sigma J^T))``. But a
    sentence is not a measurement, and this is the assumption the whole
    delegation rests on, so it is built three ways and the reports are
    compared bitwise. D22 made the same argument for the rank test and put the
    same class beside it.
    """

    @staticmethod
    def _report(sigma, data_scale, linear_problem):
        """The delta-method report, with the synthesised pieces overridden."""
        import rheplicant.inference.graph_bridge as bridge

        forward, theta0, _ = linear_problem
        cov = jnp.diag(jnp.array([0.04, 0.01, 0.09]))
        real = bridge.graph_for_information

        def spied(fn, values, noise, *, priors=None, caller="JeffreysPrior"):
            # The two synthesised things, replaced. `graph_for_information`
            # makes the data itself (zeros of the prediction's shape), so the
            # data is moved by moving what the graph is built to observe --
            # which is what a wrong synthetic residual would look like.
            return real(
                lambda v: fn(v) + data_scale,
                values,
                HomoscedasticNoise(jnp.asarray(sigma)),
                priors=priors,
                caller=caller,
            )

        bridge.graph_for_information = spied
        try:
            return propagate_covariance(forward, theta0, cov)
        finally:
            bridge.graph_for_information = real

    def test_the_synthetic_sigma_and_data_do_not_move_the_report(
        self, linear_problem
    ):
        base = self._report(1.0, 0.0, linear_problem)
        wider = self._report(1e4, 0.0, linear_problem)
        offset = self._report(1.0, 1e3, linear_problem)
        assert jnp.array_equal(base, wider)
        assert jnp.array_equal(base, offset)

    def test_the_baseline_is_not_degenerate(self, linear_problem):
        """The sibling of the test above, and it is what makes it mean
        something: three identical reports would also be three zeros, or three
        NaNs, and the comparison would pass on all of them."""
        base = self._report(1.0, 0.0, linear_problem)
        assert jnp.all(jnp.isfinite(base))
        assert float(base.min()) > 0.0


class TestPushForward:
    def test_matches_python_loop(self, linear_problem):
        forward, theta0, _ = linear_problem
        samples = theta0 + 0.1 * jax.random.normal(jax.random.key(2), (5, N_PAR))
        stacked = push_forward(forward, samples)
        assert stacked.shape == (5, N_DATA)
        for i in range(5):
            assert jnp.allclose(stacked[i], forward(samples[i]))

    def test_pytree_samples(self):
        def forward(p):
            return p["a"] * jnp.ones(3)

        samples = {"a": jnp.arange(4.0)}
        out = push_forward(forward, samples)
        assert out.shape == (4, 3)
        assert jnp.array_equal(out[:, 0], jnp.arange(4.0))


class TestEndToEndWithPipeline:
    def test_fisher_through_real_forward_fn(self, template_state):
        """The seam works: Fisher of the gain through a real assembled twin."""
        import equinox as eqx

        from rheplicant.inference import build_forward_fn
        from rheplicant.radio import GainOperator, SkyOperator, assemble

        twin = assemble(
            SkyOperator(amplitude=jnp.array(100.0)),
            GainOperator(gain=jnp.array(1.1)),
        )
        spec = jax.tree.map(lambda _: False, twin)
        spec = eqx.tree_at(lambda p: p["gain"].gain, spec, replace=True)
        forward, params0 = build_forward_fn(twin, template_state, filter_spec=spec)

        F = fisher_information(forward, params0, noise_std=0.5)
        # d(pred)/d(gain) = 100 per sample; F = n_samples * 100^2 / 0.25
        n_samples = template_state.coords.time.shape[0] * template_state.coords.freq.shape[0]
        assert jnp.allclose(F.matrix[0, 0], n_samples * 1e4 / 0.25, rtol=1e-4)
        sigma_gain = jnp.sqrt(parameter_covariance(F).matrix[0, 0])
        assert sigma_gain < 1e-2  # sub-percent gain forecast


class TestNamedParameters:
    """A Fisher matrix over a ParameterSpace has rows that mean something."""

    @pytest.fixture
    def named_problem(self):
        """forward({'a': scalar, 'b': (2,)}) — deliberately NOT alphabetical in
        declaration order, so the name->slice mapping is doing real work."""
        key = jax.random.key(1)
        A = jax.random.normal(key, (N_DATA, 3))

        def forward(values):
            return A @ jnp.concatenate([jnp.atleast_1d(values["z_scalar"]),
                                        values["a_vector"]])

        return forward, {"z_scalar": jnp.array(1.0), "a_vector": jnp.array([2.0, 3.0])}

    def test_slices_follow_the_actual_flattening(self, named_problem):
        from jax.flatten_util import ravel_pytree

        forward, params = named_problem
        cov = parameter_covariance(fisher_information(forward, params, noise_std=1.0))
        flat, _ = ravel_pytree(params)
        for name in cov.names:
            start, stop = cov.span(name)
            assert jnp.allclose(flat[start:stop], jnp.ravel(params[name]))

    def test_sigma_is_shaped_like_the_parameter(self, named_problem):
        forward, params = named_problem
        cov = parameter_covariance(fisher_information(forward, params, noise_std=1.0))
        assert cov.sigma("z_scalar").shape == ()
        assert cov.sigma("a_vector").shape == (2,)

    def test_sigma_matches_the_diagonal(self, named_problem):
        forward, params = named_problem
        cov = parameter_covariance(fisher_information(forward, params, noise_std=1.0))
        start, stop = cov.span("a_vector")
        assert jnp.allclose(
            cov.sigma("a_vector"), jnp.sqrt(jnp.diag(cov.matrix)[start:stop])
        )

    def test_block_extracts_a_cross_covariance(self, named_problem):
        forward, params = named_problem
        cov = parameter_covariance(fisher_information(forward, params, noise_std=1.0))
        assert cov.block("z_scalar", "a_vector").shape == (1, 2)
        assert cov.block("a_vector").shape == (2, 2)

    def test_sigma_of_a_fisher_matrix_is_refused(self, named_problem):
        """sqrt(diag(F)) is not an error bar — inverting is the whole point."""
        forward, params = named_problem
        fisher = fisher_information(forward, params, noise_std=1.0)
        with pytest.raises(StateValidationError, match="parameter_covariance"):
            fisher.sigma("z_scalar")

    def test_unknown_name_is_refused(self, named_problem):
        forward, params = named_problem
        cov = parameter_covariance(fisher_information(forward, params, noise_std=1.0))
        with pytest.raises(StateValidationError, match="no parameter named"):
            cov.sigma("nope")

    def test_unnamed_parameters_still_work(self, linear_problem):
        """A plain array pytree from build_forward_fn has no names, and says so."""
        forward, theta0, _ = linear_problem
        cov = parameter_covariance(fisher_information(forward, theta0, noise_std=1.0))
        assert cov.names is None
        with pytest.raises(StateValidationError, match="not named"):
            cov.sigma("anything")

    def test_a_complex_parameter_is_refused_with_guidance(self):
        """jax.jacfwd cannot differentiate a real output wrt a complex input, so
        the Fisher routines have to say so rather than die inside JAX."""
        key = jax.random.key(2)
        A = jax.random.normal(key, (N_DATA, 3)) + 1j * jax.random.normal(
            jax.random.fold_in(key, 1), (N_DATA, 3)
        )

        def forward(values):
            return jnp.real(A @ values["coeffs"])

        with pytest.raises(StateValidationError, match="Complex parameters"):
            fisher_information(forward, {"coeffs": jnp.ones(3) + 0j}, noise_std=1.0)


class TestTheConditionCeiling:
    """``parameter_covariance`` gates on conditioning, and the gate is the far
    side's (D29).

    ``F = J^T N^-1 J`` SQUARES the design's condition number, so an ordinary
    model reaches the arithmetic's limit; the ceiling is ``1/sqrt(eps)`` read
    from the values' own dtype -- float32 2.90e+03, float64 6.71e+07 -- which
    is where inverting has spent half the digits available. This package used
    to return the number anyway: measured at ``kappa(J) = 1e3``, the float32
    covariance is 2.4 % wrong and the float64 one 1.08e-12 wrong, and neither
    said which it was. A Cramer-Rao bound that is wrong without saying so is
    worse than no bound, which is why this arrives as a correction rather than
    as a regression.
    """

    @staticmethod
    def _lopsided():
        """A model whose two columns differ in scale by 1e3, and nothing else.

        The Jacobian is ``diag(1, 1e-3)``, so ``F`` is ``diag(1, 1e-6)`` and
        the inverse runs at ``kappa = 1e6`` -- outside float32's ceiling and
        inside float64's, which is what lets ONE model show both sides of the
        rule rather than two models showing one side each.
        """

        def forward(params):
            return jnp.stack([params["a"], 1e-3 * params["b"]])

        return forward, {"a": jnp.array(1.0), "b": jnp.array(1.0)}

    def test_a_float32_inverse_past_the_ceiling_is_refused(self):
        forward, params = self._lopsided()
        fisher = fisher_information(forward, params, noise_std=1.0)
        with pytest.raises(StateValidationError, match="condition number"):
            parameter_covariance(fisher)

    def test_the_refusal_wears_this_packages_class_and_keeps_the_original(self):
        """Both halves, because either alone is satisfiable the wrong way.

        The class is this package's promise -- every other refusal this module
        raises is a ``StateValidationError``, and a bare ``ValueError`` in the
        far side's vocabulary arriving at a rheplicant exit is the shape
        ``numpyro_bridge`` already had to put a sentence in front of. The
        ``__cause__`` is the other half: a facade that re-raised without
        chaining would satisfy the class assertion while throwing away the
        measurement, and the message it quotes is the one that names the
        remedy.
        """
        forward, params = self._lopsided()
        fisher = fisher_information(forward, params, noise_std=1.0)
        with pytest.raises(StateValidationError) as caught:
            parameter_covariance(fisher)
        assert caught.value.__cause__ is not None
        assert type(caught.value.__cause__) is ValueError
        assert "parameter_covariance" in str(caught.value)

    def test_the_refusal_names_the_remedy_and_says_where_it_does_not_work(self):
        """The remedy has to be reachable, and the wrong one has to be named.

        Widening only the inverse is the natural fix and it recovers nothing
        -- the digits were spent forming ``F`` -- so a message that said only
        "use float64" would send a reader to the one place it does not help.
        """
        forward, params = self._lopsided()
        fisher = fisher_information(forward, params, noise_std=1.0)
        with pytest.raises(StateValidationError) as caught:
            parameter_covariance(fisher)
        message = str(caught.value)
        assert "jax.enable_x64" in message
        assert "not around this call" in message

    def test_the_same_model_in_double_is_allowed(self):
        """Anti-vacuity: the rule is about the arithmetic, not about the model.

        Without this the ceiling could be refusing every multi-scale model and
        nothing here would notice.
        """
        with jax.enable_x64(True):
            forward, params = self._lopsided()
            params = {k: jnp.asarray(v, jnp.float64) for k, v in params.items()}
            cov = parameter_covariance(
                fisher_information(forward, params, noise_std=1.0)
            )
            assert cov.matrix.dtype == jnp.float64
            # diag(1, 1e6): the honest inverse of diag(1, 1e-6).
            assert float(cov.matrix[0, 0]) == pytest.approx(1.0, rel=1e-9)
            assert float(cov.matrix[1, 1]) == pytest.approx(1e6, rel=1e-9)

    def test_jitter_is_measured_after_it_is_applied(self):
        """``jitter`` is the one remedy this call itself offers, so the
        condition has to be read off the matrix it was added to.

        Measuring first would refuse a caller who has already fixed the
        problem, and the two orderings are indistinguishable on a
        well-conditioned matrix -- which is every other test in this file.
        """
        forward, params = self._lopsided()
        fisher = fisher_information(forward, params, noise_std=1.0)
        cov = parameter_covariance(fisher, jitter=1e-3)
        assert jnp.all(jnp.isfinite(cov.matrix))

    def test_a_covariance_is_not_inverted_a_second_time(self):
        """A new refusal, and its cost was measured at zero.

        Inverting a covariance gives a precision back, which is a legitimate
        operation and not what this function's name promises -- and the
        result used to come back labelled ``kind='covariance'`` on a matrix
        that was not one. Across ``tests/inference`` + ``tests/config`` the 65
        calls to this function are 49 ``'fisher'`` and 16
        ``'posterior_precision'``; not one hands it a covariance.
        """
        forward, params = self._lopsided()
        with jax.enable_x64(True):
            params = {k: jnp.asarray(v, jnp.float64) for k, v in params.items()}
            cov = parameter_covariance(
                fisher_information(forward, params, noise_std=1.0)
            )
            with pytest.raises(StateValidationError, match="covariance"):
                parameter_covariance(cov)

    def test_a_posterior_covariance_is_refused_by_the_same_rule(self):
        """The second covariance spelling, which the far side does not have.

        This package needs four kinds where the far side has three, so the
        translation table is the only thing standing between
        ``'posterior_covariance'`` and a silent second inversion. A table that
        dropped it would send an unknown kind across as a precision and invert
        a covariance without a word.
        """
        forward, params = self._lopsided()
        with jax.enable_x64(True):
            params = {k: jnp.asarray(v, jnp.float64) for k, v in params.items()}
            cov = parameter_covariance(
                fisher_information(forward, params, noise_std=1.0)
            )
            # Rebuilt rather than `tree_at`-ed: `kind` is a static field, so
            # it is part of the treedef and not a leaf to swap.
            posterior = FlatMatrix(
                matrix=cov.matrix,
                structure=cov.structure,
                kind="posterior_covariance",
                names=cov.names,
                spans=cov.spans,
                shapes=cov.shapes,
            )
            with pytest.raises(StateValidationError, match="covariance"):
                parameter_covariance(posterior)


class TestTheFlatLayoutIsSortedByName:
    """The invariant the seam's block order rests on, and it is JAX's, not ours.

    ``fisher_information`` asks the far side for the block in
    ``sorted(names)`` order so that nothing needs permuting afterwards -- the
    far side lays a block out in the order it is given, and this package's flat
    layout has always been sorted-by-key because ``ravel_pytree`` orders a dict
    that way.

    **That ``sorted()`` is therefore a no-op today**, which a mutation set found
    by removing it and killing nothing. It is kept because the two orders must
    not be free to drift, and the drift would be silent: every number would be
    right and every row would be in the wrong place. What was missing is an
    assertion on the invariant itself, since it belongs to JAX's dict
    flattening rather than to anything here.
    """

    def test_named_spans_orders_by_name_not_by_declaration(self):
        from rheplicant.inference.uncertainty import _named_spans

        names, spans, shapes = _named_spans(
            {"zeta": jnp.zeros(3), "mu": jnp.array(1.0), "alpha": jnp.zeros(2)}
        )
        assert names == ("alpha", "mu", "zeta"), (
            "the flat layout is no longer sorted by name, so fisher_information's "
            "block order and its reported spans have come apart"
        )
        # The spans follow the same order and are contiguous from zero, which is
        # the half that says they describe THIS ordering rather than some other.
        assert spans == ((0, 2), (2, 3), (3, 6))
        assert shapes == ((2,), (), (3,))

    def test_the_matrix_rows_follow_that_order(self):
        """End to end, on a model where the two orders visibly differ.

        ``alpha`` is declared last and sorts first, and it is the only latent
        the prediction is sensitive to at this point -- so row 0 carries the
        information and row 1 is the one that is nearly empty. A block asked
        for in declaration order would put them the other way round with every
        value still correct.
        """
        def forward(p):
            return jnp.stack([p["zulu"] * 0.0 + 1.0, p["alpha"] * 3.0])

        matrix = fisher_information(
            forward, {"zulu": jnp.array(1.0), "alpha": jnp.array(1.0)}, 1.0
        )
        assert matrix.names == ("alpha", "zulu")
        values = np.asarray(matrix.matrix)
        assert values[0, 0] == pytest.approx(9.0), "row 0 is not alpha's"
        assert values[1, 1] == pytest.approx(0.0, abs=1e-12), "row 1 is not zulu's"
