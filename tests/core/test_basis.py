"""Tests for the (time, frequency) basis expansion.

The fixture is non-square in every dimension a square one would blind: 7 time
samples against 5 frequency channels, 3 time coefficients against 2 frequency
ones. That is not decoration. ``expand`` is ``time @ coeff @ freq.T``, and with
``n_time == n_freq`` and ``n_k == n_j`` a swapped pair of design matrices is
shape-legal and returns a finite, correctly-shaped, transposed answer — the
same ambiguity :mod:`rheplicant.radio.instrument.noise_wave` states its own
``__check_init__`` cannot catch. Here it is catchable, and only because nothing
in the fixture is square.

:class:`TestTheBindPattern` reaches up into :mod:`rheplicant.inference` from a
``tests/core`` file, which is backwards for everything else in this directory.
It is deliberate: the module under test exists to be handed to ``Bind``, and two
of its design decisions — a plain dataclass rather than an ``eqx.Module``, and
``eq=False`` — are load-bearing only at that call site. A test that never made
it would leave both looking arbitrary.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.basis import BASIS_KINDS, SeparableBasis, basis_matrix
from rheplicant.core.errors import DirtError, StateValidationError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Latent,
    ParameterSpace,
    check_linearity,
    identifiability,
)

N_TIME, N_FREQ = 7, 5
N_K, N_J = 3, 2


@pytest.fixture
def basis() -> SeparableBasis:
    return SeparableBasis(
        time=basis_matrix("legendre", n=N_TIME, n_basis=N_K),
        freq=basis_matrix("legendre", n=N_FREQ, n_basis=N_J),
    )


# ------------------------------------------------------------- basis_matrix --


class TestBasisMatrix:
    def test_every_kind_is_a_design_matrix_with_one_column_per_function(self):
        """``(n, n_basis)``: rows are samples, columns are basis functions.

        The orientation ``numpy.polynomial``'s own ``*vander`` helpers use, and
        the one that makes ``time @ coeff @ freq.T`` read left to right.
        """
        for kind in BASIS_KINDS:
            design = basis_matrix(kind, n=N_TIME, n_basis=N_K)
            assert design.shape == (N_TIME, N_K), kind

    def test_the_first_column_is_the_constant_for_every_kind(self):
        """So coefficient ``[0, 0]`` is the mean level, whatever the kind.

        A basis whose constant sat somewhere else would still fit; it would
        make every ``prior_mean`` and every printed coefficient mean something
        different per kind, silently.
        """
        for kind in BASIS_KINDS:
            design = basis_matrix(kind, n=N_FREQ, n_basis=N_J)
            assert np.allclose(np.asarray(design[:, 0]), 1.0), kind

    def test_legendre_is_the_legendre_vandermonde_on_a_symmetric_grid(self):
        design = np.asarray(basis_matrix("legendre", n=N_TIME, n_basis=N_K))
        expected = np.polynomial.legendre.legvander(
            np.linspace(-1.0, 1.0, N_TIME), N_K - 1
        )
        assert np.allclose(design, expected, atol=1e-6)

    def test_polynomial_column_k_is_x_to_the_k_on_the_same_grid(self):
        design = np.asarray(basis_matrix("polynomial", n=N_TIME, n_basis=N_K))
        x = np.linspace(-1.0, 1.0, N_TIME)
        for k in range(N_K):
            assert np.allclose(design[:, k], x**k, atol=1e-6), k

    def test_fourier_uses_the_endpoint_EXCLUDED_grid_and_is_orthogonal_on_it(self):
        """The grid is ``i/n``, not ``linspace(-1, 1, n)``.

        On the endpoint-INCLUDED grid the first and last sample are one period
        apart, so they carry the same phase, and the harmonics stop being
        orthogonal: the pair ``(cos, sin)`` at a given ``k`` then has a nonzero
        inner product and the coefficients acquire a correlation nobody
        declared. Pinned because it is a one-character difference in the
        implementation and invisible in every shape.
        """
        design = np.asarray(basis_matrix("fourier", n=N_FREQ, n_basis=N_FREQ))
        gram = design.T @ design
        off_diagonal = gram - np.diag(np.diag(gram))
        assert np.max(np.abs(off_diagonal)) < 1e-6, gram

    def test_fourier_alternates_cosine_then_sine_at_each_harmonic(self):
        design = np.asarray(basis_matrix("fourier", n=N_TIME, n_basis=5))
        x = np.arange(N_TIME) / N_TIME
        for column, expected in enumerate(
            [
                np.ones(N_TIME),
                np.cos(2 * np.pi * x),
                np.sin(2 * np.pi * x),
                np.cos(4 * np.pi * x),
                np.sin(4 * np.pi * x),
            ]
        ):
            assert np.allclose(design[:, column], expected, atol=1e-6), column

    def test_an_unknown_kind_is_refused_and_the_known_ones_are_named(self):
        with pytest.raises(StateValidationError) as caught:
            basis_matrix("chebyshev", n=N_TIME, n_basis=N_K)
        message = str(caught.value)
        assert "chebyshev" in message
        for kind in BASIS_KINDS:
            assert kind in message, message

    @pytest.mark.parametrize("n", [0, -1])
    def test_a_grid_with_no_samples_is_refused(self, n):
        """Matched on the phrase only THIS guard uses. ``n < 1`` also satisfies
        ``n_basis > n``, so a looser pattern is answered by the wrong guard and
        the test passes with this one deleted — measured, not hypothetical."""
        with pytest.raises(StateValidationError, match="An axis with no samples"):
            basis_matrix("legendre", n=n, n_basis=1)

    @pytest.mark.parametrize("n_basis", [0, -2])
    def test_a_basis_with_no_functions_is_refused(self, n_basis):
        """An empty basis expands every coefficient to exactly zero, so the
        quantity it parameterizes is pinned at zero while the fit reports a
        converged answer for the parameters that remain."""
        with pytest.raises(StateValidationError, match="n_basis="):
            basis_matrix("legendre", n=N_TIME, n_basis=n_basis)

    def test_more_functions_than_samples_is_refused_by_counting_alone(self):
        with pytest.raises(StateValidationError) as caught:
            basis_matrix("legendre", n=N_J, n_basis=N_J + 1)
        assert "3" in str(caught.value) and "2" in str(caught.value)

    def test_legendre_is_far_better_conditioned_than_the_same_span_in_monomials(self):
        """The only reason to prefer one over the other: they span exactly the
        same functions, and one of them does it with a design matrix four and a
        half decades worse conditioned. That number lands on the block's normal
        operator, which is what ``wiener_solve``'s error guard divides by.

        Pinned loosely (a factor of 100 at n=16, of 1000 at n=32) so a change of
        implementation has room, and tightly enough that swapping the two kinds
        fails here rather than in someone's posterior."""
        for n, n_basis, ratio in [(16, 10, 100.0), (32, 16, 1000.0)]:
            smooth = np.linalg.cond(
                np.asarray(basis_matrix("legendre", n=n, n_basis=n_basis), np.float64)
            )
            raw = np.linalg.cond(
                np.asarray(basis_matrix("polynomial", n=n, n_basis=n_basis), np.float64)
            )
            assert raw > ratio * smooth, (n, n_basis, smooth, raw)

        # ... and the same span: each reproduces the other exactly, so the
        # difference really is conditioning and nothing else.
        legendre = np.asarray(basis_matrix("legendre", n=N_TIME, n_basis=N_K), np.float64)
        monomial = np.asarray(basis_matrix("polynomial", n=N_TIME, n_basis=N_K), np.float64)
        assert np.allclose(legendre @ np.linalg.pinv(legendre) @ monomial, monomial)

    def test_a_COMPLETE_basis_is_allowed_and_spans_every_function_on_its_axis(self):
        """``n_basis == n`` is legal, and the docstring's warning about it is
        a statement about identifiability rather than about the matrix: the
        matrix is perfectly well conditioned and invertible."""
        for kind in BASIS_KINDS:
            design = np.asarray(basis_matrix(kind, n=N_FREQ, n_basis=N_FREQ))
            assert np.linalg.matrix_rank(design) == N_FREQ, kind


# ---------------------------------------------------------- SeparableBasis --


class TestSeparableBasis:
    def test_it_reports_the_grid_it_covers_and_the_coefficients_it_takes(self, basis):
        assert basis.shape == (N_TIME, N_FREQ)
        assert basis.coeff_shape == (N_K, N_J)

    def test_expand_is_time_coeff_freqT_and_lands_on_the_full_grid(self, basis):
        coeff = jnp.arange(N_K * N_J, dtype=float).reshape(N_K, N_J) + 1.0
        expanded = basis.expand(coeff)
        assert expanded.shape == (N_TIME, N_FREQ)
        assert jnp.allclose(expanded, basis.time @ coeff @ basis.freq.T)

    def test_expand_always_returns_2D_which_is_the_unambiguous_temperature_shape(
        self, basis
    ):
        """:mod:`rheplicant.radio.instrument.noise_wave` accepts ``()``,
        ``(n_freq,)``, ``(n_time, 1)`` and ``(n_time, n_freq)``, and states that
        it cannot tell a bare ``(n,)`` per-time vector from a spectrum when
        ``n_time == n_freq``. An expansion never produces a bare vector: even a
        one-coefficient basis on each axis comes back on the full grid, so the
        ambiguity that guard cannot resolve is one this route cannot create.
        """
        flat = SeparableBasis(
            time=basis_matrix("legendre", n=N_TIME, n_basis=1),
            freq=basis_matrix("legendre", n=N_FREQ, n_basis=1),
        )
        expanded = flat.expand(jnp.asarray([[300.0]]))
        assert expanded.shape == (N_TIME, N_FREQ)
        assert jnp.allclose(expanded, 300.0)

    def test_fit_recovers_the_coefficients_of_a_field_inside_the_span(self, basis):
        coeff = jnp.asarray([[2800.0, -120.0], [90.0, 17.0], [-30.0, 6.0]])
        recovered = basis.fit(basis.expand(coeff))
        assert jnp.allclose(recovered, coeff, rtol=1e-4, atol=1e-3), recovered

    def test_fit_of_a_field_OUTSIDE_the_span_is_the_least_squares_projection(
        self, basis
    ):
        """Not an error, and not the field: ``fit`` answers "the closest thing
        this basis can say", which is what makes it usable for a starting
        value. The residual is orthogonal to the span, which is the check that
        it really is the projection and not merely something close."""
        rough = jax.random.normal(jax.random.key(11), (N_TIME, N_FREQ)) * 100.0
        residual = rough - basis.expand(basis.fit(rough))
        assert float(jnp.max(jnp.abs(residual))) > 1.0
        overlap = basis.time.T @ residual @ basis.freq
        assert float(jnp.max(jnp.abs(overlap))) < 1e-2, overlap

    @pytest.mark.parametrize("axis", ["time", "freq"])
    def test_a_design_matrix_that_is_not_2D_is_refused_per_axis(self, axis):
        good = {
            "time": basis_matrix("legendre", n=N_TIME, n_basis=N_K),
            "freq": basis_matrix("legendre", n=N_FREQ, n_basis=N_J),
        }
        with pytest.raises(StateValidationError) as caught:
            SeparableBasis(**{**good, axis: jnp.ones(N_TIME)})
        assert axis in str(caught.value)

    @pytest.mark.parametrize("axis", ["time", "freq"])
    def test_an_empty_design_matrix_is_refused_per_axis(self, axis):
        good = {
            "time": basis_matrix("legendre", n=N_TIME, n_basis=N_K),
            "freq": basis_matrix("legendre", n=N_FREQ, n_basis=N_J),
        }
        with pytest.raises(StateValidationError) as caught:
            SeparableBasis(**{**good, axis: jnp.ones((N_TIME, 0))})
        assert axis in str(caught.value)

    @pytest.mark.parametrize("axis", ["time", "freq"])
    def test_more_functions_than_samples_is_refused_per_axis(self, axis):
        good = {
            "time": basis_matrix("legendre", n=N_TIME, n_basis=N_K),
            "freq": basis_matrix("legendre", n=N_FREQ, n_basis=N_J),
        }
        with pytest.raises(StateValidationError) as caught:
            SeparableBasis(**{**good, axis: jnp.ones((2, 3))})
        assert axis in str(caught.value)

    def test_the_two_axes_are_not_interchangeable_and_the_shapes_say_so(self, basis):
        """The whole reason this fixture is non-square. Swapping the two design
        matrices is refused rather than transposing the answer."""
        with pytest.raises(StateValidationError):
            SeparableBasis(time=basis.freq, freq=basis.time).expand(
                jnp.zeros((N_K, N_J))
            )

    def test_expand_refuses_a_coefficient_of_the_wrong_shape(self, basis):
        with pytest.raises(StateValidationError) as caught:
            basis.expand(jnp.zeros((N_J, N_K)))
        message = str(caught.value)
        assert "(3, 2)" in message and "(2, 3)" in message
        assert "transpose" in message.lower(), message

    def test_fit_refuses_a_field_that_is_not_on_this_grid(self, basis):
        with pytest.raises(StateValidationError) as caught:
            basis.fit(jnp.zeros((N_FREQ, N_TIME)))
        assert "(7, 5)" in str(caught.value)


# --------------------------------------------------------- the Bind pattern --


class _FullGridTemperature(AbstractOperator):
    """An operator holding a full ``(n_time, n_freq)`` temperature leaf.

    Stands in for every operator whose temperature leaf is the whole grid —
    :class:`~rheplicant.radio.instrument.noise_wave.NoiseWaveOperator`'s four,
    above all. Those cannot hold coefficients themselves, which is exactly why
    the expansion has to be expressible as a ``Bind``.
    """

    requires = ("coords.time", "coords.freq")
    provides = ("data",)

    temperature: jax.Array

    def __call__(self, state):
        return state.with_data(self.temperature)


def _state() -> State:
    return State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        )
    )


class TestTheBindPattern:
    """``Bind(..., fn=basis.expand)`` — the whole of what the framework needs.

    :mod:`rheplicant.inference.linear`'s premise is that a basis expansion is
    already expressible: a linear latent bound through a matrix product. These
    tests are that claim, run.
    """

    def test_a_coefficient_latent_drives_a_full_grid_leaf_through_expand(self, basis):
        pipeline = Pipeline(
            _FullGridTemperature(temperature=jnp.zeros((N_TIME, N_FREQ))),
            names=("t_ant",),
        )
        space = ParameterSpace(
            latents=[Latent("t_coeff", init=jnp.zeros((N_K, N_J)), linear=True)],
            bindings=[
                Bind("t_coeff", into=lambda p: p["t_ant"].temperature, fn=basis.expand)
            ],
        )
        forward, _ = space.forward_fn(pipeline, _state())
        coeff = jnp.asarray([[2800.0, -120.0], [90.0, 17.0], [-30.0, 6.0]])
        assert jnp.allclose(forward({"t_coeff": coeff}), basis.expand(coeff))

    def test_expand_is_a_drop_in_for_the_lambda_it_replaces(self, basis, recwarn):
        """``Bind.fn`` is a STATIC field, and an ``eqx.Module`` here would be
        wrong in a way no shape shows: equinox wraps a Module's bound methods as
        pytrees, so ``fn=basis.expand`` would carry the design matrices INTO the
        treedef, where equinox warns and where array ``__eq__`` would decide
        treedef equality."""
        Bind("t_coeff", into=lambda p: p["t_ant"].temperature, fn=basis.expand)
        static_array_warnings = [w for w in recwarn if "static" in str(w.message)]
        assert not static_array_warnings, [str(w.message) for w in static_array_warnings]

    def test_two_bases_can_be_compared_and_hashed_like_any_other_object(self, basis):
        """``eq=False``, and this is the whole of what it buys — measured, not
        assumed. A frozen dataclass defaults to ``eq=True``, which compares the
        design matrices elementwise: ``a == b`` then raises "the truth value of
        an array with more than one element is ambiguous" instead of answering,
        and ``__hash__`` becomes None.

        It is NOT what makes ``fn=basis.expand`` work as a static field. A bound
        method compares and hashes its ``__self__`` by pointer, so that route
        survives ``eq=True`` untouched — which is exactly why this needs its own
        test rather than riding on the one above.
        """
        twin = SeparableBasis(time=basis.time, freq=basis.freq)
        assert (basis == twin) is False
        assert (basis == basis) is True
        assert hash(basis) == hash(basis)

    def test_the_expansion_is_affine_so_the_conjugate_exits_accept_it(self, basis):
        """``linear=True`` is a claim, and this is the claim being checked
        against the model the basis actually produces."""
        pipeline = Pipeline(
            _FullGridTemperature(temperature=jnp.zeros((N_TIME, N_FREQ))),
            names=("t_ant",),
        )
        space = ParameterSpace(
            latents=[Latent("t_coeff", init=jnp.ones((N_K, N_J)), linear=True)],
            bindings=[
                Bind("t_coeff", into=lambda p: p["t_ant"].temperature, fn=basis.expand)
            ],
        )
        errors = check_linearity(space, pipeline, _state(), "t_coeff")
        assert max(errors.values()) < 1e-4, errors

    def test_the_coefficients_of_a_lone_smooth_temperature_are_all_identified(
        self, basis
    ):
        pipeline = Pipeline(
            _FullGridTemperature(temperature=jnp.zeros((N_TIME, N_FREQ))),
            names=("t_ant",),
        )
        space = ParameterSpace(
            latents=[Latent("t_coeff", init=jnp.ones((N_K, N_J)), linear=True)],
            bindings=[
                Bind("t_coeff", into=lambda p: p["t_ant"].temperature, fn=basis.expand)
            ],
        )
        report = identifiability(space, pipeline, _state())
        assert report.n_par == N_K * N_J
        assert report.nullity == 0, report.nullity


class TestWhichErrorTheRefusalsRaise:
    """Structural refusals, so :class:`StateValidationError` — core's own class
    for "constructed with invalid contents", and the one every operator in this
    package already raises.

    ``ParameterSpaceError`` would read well for the ``Bind`` route and is
    deliberately absent from ``rheplicant.core``'s public surface; a core module
    raising it would make ``rheplicant.core`` inconsistent about its own error
    vocabulary.
    """

    def test_refusals_are_state_validation_errors_and_stay_in_the_family(self):
        import rheplicant.core as core

        assert "ParameterSpaceError" not in core.__all__
        with pytest.raises(StateValidationError):
            basis_matrix("nope", n=4, n_basis=2)
        # ... and still catchable with the one clause that catches them all
        with pytest.raises(DirtError):
            SeparableBasis(time=jnp.ones((2, 3)), freq=jnp.ones((N_FREQ, N_J)))
