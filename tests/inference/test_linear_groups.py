"""Tests for GROUPED linear blocks — one block holding several latents.

``linear_operator(..., names=("t_nw", "t_ant"))`` exports the joint operator
over a group, whose ``x`` is a ``{name: array}`` dict. The reason it exists is
measurable and is pinned at the bottom of this file: two latents the data
barely tells apart are recovered by one joint solve and missed by hundreds of
kelvin by an alternating one — while every per-block guard the package ships
reports a condition number of ~1 and a converged residual at every step of the
alternation, because both numbers are computed FROM THE BLOCK and neither can
see across the partition.

The fixtures here are deliberately asymmetric. ``t_ant`` has three coefficients
and ``t_nw`` two; their truths sit at ~1e3 K and ~1e2 K; the two bases are
different functions of frequency. Two latents of the same shape cannot show
that the wrong one's answer was carried, and a symmetric split cannot show an
inversion — both have shipped in this project before.

Verified under BOTH float32 and JAX_ENABLE_X64=1.
"""

import dataclasses
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import ParameterSpaceError
from rheplicant.core.operator import AbstractOperator
from rheplicant.core.pipeline import Pipeline
from rheplicant.inference import (
    Bind,
    Latent,
    LinearBlock,
    ParameterSpace,
    check_linearity,
    condition_estimate,
    gcr_sample,
    identifiability,
    linear_operator,
    wiener_solve,
)
from rheplicant.radio import GainOperator

N_TIME, N_FREQ = 8, 8


# ------------------------------------------------------------ test doubles --


class BasisOperator(AbstractOperator):
    """``data[t, f] = (basis @ coeff)[f]`` — a spectral basis, flat in time."""

    requires: ClassVar[tuple[str, ...]] = ("coords.time", "coords.freq")
    provides: ClassVar[tuple[str, ...]] = ("data",)

    coeff: jax.Array
    basis: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        row = self.basis @ self.coeff
        return state.with_data(jnp.broadcast_to(row, (n_time, row.shape[0])))


class AddBasisOperator(AbstractOperator):
    """The same, added to whatever is already on the signal path."""

    requires: ClassVar[tuple[str, ...]] = ("data",)
    provides: ClassVar[tuple[str, ...]] = ("data",)

    coeff: jax.Array
    basis: jax.Array

    def __call__(self, state):
        return state.with_data(state.data + (self.basis @ self.coeff)[None, :])


class MixedDTypeOperator(AbstractOperator):
    """A real observation linear in a REAL latent and a COMPLEX one at once.

    The case the group's real/complex split is most likely to get wrong: a
    concatenation over "real degrees of freedom" has to know that one member
    contributes ``n`` of them and the other ``2n``, in the right order.
    """

    provides: ClassVar[tuple[str, ...]] = ("data",)

    amp: jax.Array
    coeffs: jax.Array
    real_basis: jax.Array
    cplx_matrix: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        rows = self.real_basis @ self.amp + jnp.real(self.cplx_matrix @ self.coeffs)
        return state.with_data(rows.reshape(n_time, n_freq))


class SeenAndBlindOperator(AbstractOperator):
    """One latent the data sees, one it is completely blind to.

    ``blind`` is bound and differentiable and has an exactly zero Jacobian, so
    the normal operator's ``λ_min`` is exactly ``1/prior_std['blind']²`` — which
    is what makes this the fixture for the floor the condition estimate applies.

    ``seen`` is a SCALAR on purpose. With a vector there, the normal operator
    carries a second eigenvalue close to ``λ_max`` and the power iteration for
    ``λ_max - λ_min`` stalls between the two, so the floor never gets consulted
    and the fixture tests nothing about it.
    """

    provides: ClassVar[tuple[str, ...]] = ("data",)

    seen: jax.Array
    blind: jax.Array

    def __call__(self, state):
        n_time = state.coords.time.shape[0]
        n_freq = state.coords.freq.shape[0]
        value = self.seen + 0.0 * jnp.sum(self.blind)
        return state.with_data(jnp.full((n_time, n_freq), value))


# ------------------------------------------------------------------ bases ---


def _orthonormal(n: int, powers: tuple[int, ...]) -> jax.Array:
    x = jnp.linspace(-1.0, 1.0, n)
    q, _ = jnp.linalg.qr(jnp.stack([x**k for k in powers], axis=1))
    return q


#: Five orthonormal spectral shapes, split 3 + 2. Split from ONE orthonormal
#: set rather than built as two independent ones, so the well-conditioned
#: fixture is exactly well-conditioned (κ = 1 jointly and per block) and the
#: near-degenerate fixture below is the only thing that is not.
_SHAPES = _orthonormal(N_FREQ, (0, 1, 2, 3, 4))
B_ANT = _SHAPES[:, :3]
B_NW = _SHAPES[:, 3:5]

#: The near-degenerate pair: ``t_nw``'s shapes are 85% ``t_ant``'s. Chosen so
#: the JOINT condition number is ~130 while each block's is 1, and so that
#: block-coordinate descent contracts by only 0.97 per sweep.
TILT = 0.15
B_NW_TILTED = (1.0 - TILT) * _SHAPES[:, :2] + TILT * _SHAPES[:, 3:5]

GAIN0 = 1.5 + 0.05 * jnp.arange(N_TIME, dtype=float)

#: An order of magnitude apart, and neither a multiple of the other, so a
#: member swap cannot pass.
TRUE_ANT = jnp.array([2800.0, -160.0, 35.0])
TRUE_NW = jnp.array([250.0, -18.0])

PRIOR = {"t_nw": 1e3, "t_ant": 1e4}


@pytest.fixture
def state():
    return State(
        coords=Coordinates(
            time=jnp.arange(N_TIME, dtype=float),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
        ),
        meta={"telescope": "RHINO", "obs_id": "linear-groups-000"},
    )


@pytest.fixture
def three_linear():
    """``data = gain * (B_ant @ t_ant + B_nw @ t_nw)`` — the tranche, in miniature.

    Three latents, each linear given the other two; ``t_nw`` and ``t_ant``
    jointly linear given the gain, and the gain NOT jointly linear with either.
    """
    return Pipeline(
        BasisOperator(coeff=jnp.zeros(3), basis=B_ANT),
        AddBasisOperator(coeff=jnp.zeros(2), basis=B_NW),
        GainOperator(gain=GAIN0),
        names=("t_ant", "t_nw", "gain"),
    )


def three_linear_space(priors=None):
    priors = priors or {}
    return ParameterSpace(
        latents=[
            Latent("t_ant", init=TRUE_ANT, prior=priors.get("t_ant"), linear=True),
            Latent("t_nw", init=TRUE_NW, prior=priors.get("t_nw"), linear=True),
            Latent("gain", init=GAIN0, prior=priors.get("gain"), linear=True),
        ],
        bindings=[
            Bind("t_ant", into=lambda p: p["t_ant"].coeff),
            Bind("t_nw", into=lambda p: p["t_nw"].coeff),
            Bind("gain", into=lambda p: p["gain"].gain),
        ],
    )


@pytest.fixture
def space():
    return three_linear_space()


@pytest.fixture
def pair(space, three_linear, state):
    """The grouped block over ``("t_nw", "t_ant")``, at the declared gain."""
    return linear_operator(space, three_linear, state, names=("t_nw", "t_ant"))


@pytest.fixture
def observed(pair):
    return pair.offset + pair.forward({"t_nw": TRUE_NW, "t_ant": TRUE_ANT})


# ------------------------------------------------------------ the operator --


class TestGroupedOperator:
    def test_offset_is_the_model_with_the_WHOLE_group_at_zero(
        self, space, three_linear, state
    ):
        block = linear_operator(space, three_linear, state, names=("t_nw", "t_ant"))
        # both members zeroed leaves nothing on the signal path
        assert jnp.allclose(block.offset, 0.0, atol=1e-3)
        # ... whereas a block over t_nw alone keeps t_ant's contribution
        alone = linear_operator(space, three_linear, state, "t_nw")
        assert not jnp.allclose(alone.offset, 0.0, atol=1e-3)

    def test_offset_plus_forward_reproduces_the_model(
        self, pair, space, three_linear, state
    ):
        forward, values0 = space.forward_fn(three_linear, state)
        x = {"t_nw": TRUE_NW, "t_ant": TRUE_ANT}
        assert jnp.allclose(
            pair.offset + pair.forward(x), forward({**values0, **x}), rtol=1e-5
        )

    def test_shape_and_dtype_are_keyed_by_name(self, pair):
        assert pair.grouped
        assert pair.names == ("t_nw", "t_ant")
        assert pair.shape == {"t_nw": (2,), "t_ant": (3,)}
        assert set(pair.dtype) == {"t_nw", "t_ant"}
        assert pair.prior == {"t_nw": None, "t_ant": None}

    def test_a_single_block_is_still_a_single_block(self, space, three_linear, state):
        block = linear_operator(space, three_linear, state, "t_nw")
        assert not block.grouped
        assert block.names == ("t_nw",)
        assert block.shape == (2,)

    def test_a_group_of_ONE_is_legal_and_answers_in_a_dict(
        self, space, three_linear, state
    ):
        """How a partition holds one-latent and many-latent blocks uniformly."""
        block = linear_operator(space, three_linear, state, names=("t_nw",))
        assert block.grouped and block.names == ("t_nw",)
        solved, _ = wiener_solve(
            block, block.offset + block.forward({"t_nw": TRUE_NW}),
            noise_std=1e-2, prior_std={"t_nw": 1e3},
        )
        assert set(solved) == {"t_nw"}
        assert jnp.allclose(solved["t_nw"], TRUE_NW, rtol=1e-3)

    def test_a_bare_string_is_ONE_name_not_four(self, space, three_linear, state):
        """`names="t_nw"` must not iterate into ('t', '_', 'n', 'w')."""
        block = linear_operator(space, three_linear, state, names="t_nw")
        assert block.names == ("t_nw",)

    def test_name_and_names_together_are_refused(self, space, three_linear, state):
        with pytest.raises(ParameterSpaceError, match="name= OR names="):
            linear_operator(space, three_linear, state, "t_nw", names=("t_nw", "t_ant"))

    def test_empty_names_is_refused(self, space, three_linear, state):
        with pytest.raises(ParameterSpaceError, match="at least one latent name"):
            linear_operator(space, three_linear, state, names=())

    def test_an_unknown_name_is_refused(self, space, three_linear, state):
        with pytest.raises(ParameterSpaceError, match="not a latent of this space"):
            linear_operator(space, three_linear, state, names=("t_nw", "nope"))

    def test_a_repeated_name_is_refused(self, space, three_linear, state):
        with pytest.raises(ParameterSpaceError, match="more than once"):
            linear_operator(space, three_linear, state,
                            names=("t_nw", "t_ant", "t_nw"))

    def test_a_member_not_declared_linear_is_refused(self, three_linear, state):
        undeclared = ParameterSpace(
            latents=[
                Latent("t_ant", init=TRUE_ANT, linear=True),
                Latent("t_nw", init=TRUE_NW),          # not declared
                Latent("gain", init=GAIN0, linear=True),
            ],
            bindings=[
                Bind("t_ant", into=lambda p: p["t_ant"].coeff),
                Bind("t_nw", into=lambda p: p["t_nw"].coeff),
                Bind("gain", into=lambda p: p["gain"].gain),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="not declared linear=True"):
            linear_operator(undeclared, three_linear, state, names=("t_nw", "t_ant"))

    def test_at_rebuilds_the_group_at_another_gain(self, space, three_linear, state):
        doubled = linear_operator(space, three_linear, state, names=("t_nw", "t_ant"),
                                  at={"gain": 2.0 * GAIN0})
        base = linear_operator(space, three_linear, state, names=("t_nw", "t_ant"))
        x = {"t_nw": TRUE_NW, "t_ant": TRUE_ANT}
        assert jnp.allclose(doubled.forward(x), 2.0 * base.forward(x), rtol=1e-4)

    def test_at_rejects_an_unknown_name(self, space, three_linear, state):
        with pytest.raises(ParameterSpaceError, match="not a latent"):
            linear_operator(space, three_linear, state, names=("t_nw", "t_ant"),
                            at={"nope": jnp.array(1.0)})


# --------------------------------------------------------------- adjoints ---


@pytest.fixture
def mixed_pipeline(state):
    n_row = N_TIME * N_FREQ
    key = jax.random.key(0)
    return Pipeline(
        MixedDTypeOperator(
            amp=jnp.zeros(2),
            coeffs=jnp.zeros(3, dtype=jnp.complex64),
            # Deliberately unlike each other, and neither a permutation of the
            # other: a symmetric pair could not show that the two members'
            # cotangents were swapped.
            real_basis=jax.random.normal(key, (n_row, 2)),
            cplx_matrix=(
                jax.random.normal(jax.random.fold_in(key, 1), (n_row, 3))
                + 1j * jax.random.normal(jax.random.fold_in(key, 2), (n_row, 3))
            ),
        ),
        names=("mixed",),
    )


@pytest.fixture
def mixed_space():
    return ParameterSpace(
        latents=[
            Latent("amp", init=jnp.array([7.0, -2.0]), linear=True),
            Latent("coeffs", init=jnp.ones(3) + 0j, linear=True),
        ],
        bindings=[
            Bind("amp", into=lambda p: p["mixed"].amp),
            Bind("coeffs", into=lambda p: p["mixed"].coeffs),
        ],
    )


@pytest.fixture
def mixed_block(mixed_space, mixed_pipeline, state):
    return linear_operator(mixed_space, mixed_pipeline, state, names=("amp", "coeffs"))


class TestGroupedAdjoint:
    """The convention that has to survive a group mixing real and complex.

    ``adjoint`` is ``jax.vjp``, which returns the CONJUGATE gradient for complex
    inputs, so the identity that holds is the one over the REAL inner product::

        Re sum_members sum(x * adjoint(y))  ==  sum(forward(x) * y)

    and NOT the sesquilinear ``sum(conj(x) * adjoint(y))``. A group is where a
    concatenation over real degrees of freedom is most likely to get this wrong,
    because the two members contribute a different number of them.
    """

    @pytest.fixture
    def probe(self):
        key = jax.random.key(7)
        return {
            "amp": jax.random.normal(key, (2,)),
            "coeffs": jax.random.normal(jax.random.fold_in(key, 1), (3,))
            + 1j * jax.random.normal(jax.random.fold_in(key, 2), (3,)),
        }

    def test_the_real_pairing_holds_and_the_sesquilinear_one_does_not(
        self, mixed_block, probe
    ):
        y = jax.random.normal(jax.random.key(11), mixed_block.offset.shape)
        cotangent = mixed_block.adjoint(y)
        paired = float(jnp.sum(mixed_block.forward(probe) * y))

        real_form = float(
            sum(
                jnp.real(jnp.sum(probe[member] * cotangent[member]))
                for member in ("amp", "coeffs")
            )
        )
        assert real_form == pytest.approx(paired, rel=1e-4)

        sesquilinear = float(
            sum(
                jnp.real(jnp.sum(jnp.conj(probe[member]) * cotangent[member]))
                for member in ("amp", "coeffs")
            )
        )
        assert sesquilinear != pytest.approx(paired, rel=1e-3)

    def test_the_cotangent_lands_on_the_right_member(self, mixed_block, probe):
        """Each member's half of the pairing must match that member alone.

        Swapping the two cotangents can leave the TOTAL pairing nearly
        unchanged, so the total is not enough.
        """
        y = jax.random.normal(jax.random.key(13), mixed_block.offset.shape)
        cotangent = mixed_block.adjoint(y)
        for member in ("amp", "coeffs"):
            only = {
                key: value if key == member else jnp.zeros_like(value)
                for key, value in probe.items()
            }
            assert float(jnp.real(jnp.sum(probe[member] * cotangent[member]))) == (
                pytest.approx(float(jnp.sum(mixed_block.forward(only) * y)), rel=1e-4)
            )

    def test_the_cotangent_is_shaped_and_typed_like_the_latents(self, mixed_block):
        cotangent = mixed_block.adjoint(jnp.ones_like(mixed_block.offset))
        assert cotangent["amp"].shape == (2,)
        assert cotangent["coeffs"].shape == (3,)
        assert jnp.issubdtype(cotangent["coeffs"].dtype, jnp.complexfloating)
        assert not jnp.issubdtype(cotangent["amp"].dtype, jnp.complexfloating)

    def test_a_real_only_group_keeps_real_cotangents(self, pair):
        cotangent = pair.adjoint(jnp.ones_like(pair.offset))
        assert not jnp.issubdtype(cotangent["t_nw"].dtype, jnp.complexfloating)
        assert not jnp.issubdtype(cotangent["t_ant"].dtype, jnp.complexfloating)


# ------------------------------------------------------------------ solves --


def _dense_group(block, order):
    """``(A, spans)`` over the group's real degrees of freedom, built by columns.

    The expensive, obviously-correct reference. ``order`` states the column
    layout explicitly so the reference cannot inherit a mistake from the code
    under test; a complex member contributes two columns per coefficient.
    """
    columns, spans = [], {}
    for member in order:
        start = len(columns)
        size = int(jnp.prod(jnp.array(block.shape[member]))) if block.shape[member] else 1
        is_complex = jnp.issubdtype(block.dtype[member], jnp.complexfloating)
        for index in range(size):
            for unit in (1.0, 1j) if is_complex else (1.0,):
                basis = {
                    name: jnp.zeros(block.shape[name], dtype=block.dtype[name])
                    for name in block.names
                }
                flat = jnp.zeros(size, dtype=block.dtype[member]).at[index].set(unit)
                basis[member] = flat.reshape(block.shape[member])
                columns.append(jnp.ravel(block.forward(basis)))
        spans[member] = (start, len(columns))
    return jnp.stack(columns, axis=1), spans


def _dense_normal(block, noise_std, prior_std, order):
    A, spans = _dense_group(block, order)
    inverse_variance = jnp.concatenate([
        jnp.full(stop - start, 1.0 / jnp.asarray(prior_std[member]) ** 2)
        for member, (start, stop) in spans.items()
    ])
    weight = 1.0 / jnp.asarray(noise_std) ** 2
    return A, spans, A.T @ (weight * A) + jnp.diag(inverse_variance)


def _dense_solve(block, observed, noise_std, prior_std, order):
    A, spans, normal = _dense_normal(block, noise_std, prior_std, order)
    weight = 1.0 / jnp.asarray(noise_std) ** 2
    flat = jnp.linalg.solve(normal, A.T @ (weight * jnp.ravel(observed - block.offset)))
    solved = {}
    for member, (start, stop) in spans.items():
        piece = flat[start:stop]
        if jnp.issubdtype(block.dtype[member], jnp.complexfloating):
            piece = piece[0::2] + 1j * piece[1::2]
        solved[member] = piece.reshape(block.shape[member])
    return solved


class TestGroupedSolve:
    def test_matches_a_dense_solve(self, pair, observed):
        solved, _ = wiener_solve(pair, observed, noise_std=1.0, prior_std=PRIOR)
        expected = _dense_solve(pair, observed, 1.0, PRIOR, ("t_nw", "t_ant"))
        for member in ("t_nw", "t_ant"):
            assert jnp.allclose(solved[member], expected[member], rtol=1e-3, atol=1e-2)

    def test_recovers_a_noiseless_signal_under_a_weak_prior(self, pair, observed):
        solved, _ = wiener_solve(pair, observed, noise_std=1e-2,
                                 prior_std={"t_nw": 1e5, "t_ant": 1e5})
        assert jnp.allclose(solved["t_nw"], TRUE_NW, rtol=2e-3)
        assert jnp.allclose(solved["t_ant"], TRUE_ANT, rtol=2e-3)

    def test_the_answer_is_keyed_and_shaped_like_the_latents(self, pair, observed):
        solved, residual = wiener_solve(pair, observed, noise_std=1.0, prior_std=PRIOR)
        assert set(solved) == {"t_nw", "t_ant"}
        assert solved["t_nw"].shape == (2,)
        assert solved["t_ant"].shape == (3,)
        assert float(residual) < 1e-3

    def test_permuting_names_gives_the_SAME_solution(self, space, three_linear, state):
        """Ordering is deterministic, so the caller's order cannot change the answer.

        Asserted as EXACT equality, not `allclose`: the group's domain is a
        pytree whose treedef JAX derives from the sorted keys, so a permutation
        must reach bit-for-bit the same CG. Anything looser would pass on an
        implementation that concatenated in caller order and happened to be
        close.
        """
        forward_order = linear_operator(space, three_linear, state,
                                        names=("t_nw", "t_ant"))
        reverse_order = linear_operator(space, three_linear, state,
                                        names=("t_ant", "t_nw"))
        assert reverse_order.names == ("t_ant", "t_nw")
        data = forward_order.offset + forward_order.forward(
            {"t_nw": TRUE_NW, "t_ant": TRUE_ANT}
        )
        first, _ = wiener_solve(forward_order, data, noise_std=0.5, prior_std=PRIOR)
        second, _ = wiener_solve(reverse_order, data, noise_std=0.5, prior_std=PRIOR)
        for member in ("t_nw", "t_ant"):
            assert jnp.array_equal(first[member], second[member]), member

    def test_the_per_member_prior_std_reaches_the_member_that_names_it(self, pair,
                                                                       observed):
        """A prior tight on one member must shrink THAT member and leave the
        other where it was.

        The two bases are orthogonal, so the untightened member genuinely cannot
        absorb the shrunk one's signal — which makes "leave the other alone" a
        real assertion rather than an accident, and makes a block-diagonal S
        assembled onto the wrong leaves impossible to miss.
        """
        wide = {"t_nw": 1e2, "t_ant": 1e2}
        loose, _ = wiener_solve(pair, observed, noise_std=1.0, prior_std=wide)
        tight_nw, _ = wiener_solve(pair, observed, noise_std=1.0,
                                   prior_std={**wide, "t_nw": 1e-2})
        tight_ant, _ = wiener_solve(pair, observed, noise_std=1.0,
                                    prior_std={**wide, "t_ant": 1e-2})

        def size(values, member):
            return float(jnp.linalg.norm(values[member]))

        assert size(tight_nw, "t_nw") < 0.05 * size(loose, "t_nw")
        assert size(tight_nw, "t_ant") == pytest.approx(size(loose, "t_ant"), rel=0.02)
        assert size(tight_ant, "t_ant") < 0.05 * size(loose, "t_ant")
        assert size(tight_ant, "t_nw") == pytest.approx(size(loose, "t_nw"), rel=0.02)

    def test_the_per_member_prior_mean_reaches_the_member_that_names_it(self, pair):
        """With uninformative data the mean IS the prior mean — per member."""
        mean, _ = wiener_solve(
            pair, pair.offset, noise_std=1e8,
            prior_std={"t_nw": 1.0, "t_ant": 1.0},
            prior_mean={"t_nw": 3.0, "t_ant": -7.0},
        )
        assert jnp.allclose(mean["t_nw"], 3.0, rtol=1e-3)
        assert jnp.allclose(mean["t_ant"], -7.0, rtol=1e-3)

    def test_an_omitted_prior_mean_is_zero_for_that_member_only(self, pair):
        mean, _ = wiener_solve(
            pair, pair.offset, noise_std=1e8,
            prior_std={"t_nw": 1.0, "t_ant": 1.0}, prior_mean={"t_ant": -7.0},
        )
        assert jnp.allclose(mean["t_nw"], 0.0, atol=1e-5)
        assert jnp.allclose(mean["t_ant"], -7.0, rtol=1e-3)

    def test_a_draw_with_no_data_follows_each_members_OWN_prior(self, pair):
        """Both moments, per member — the check that S is block-diagonal AND on
        the right leaves. The two widths differ by 5x, so a single width spread
        across the group cannot pass this."""
        keys = jax.random.split(jax.random.key(4), 4000)
        draws = jax.vmap(
            lambda k: gcr_sample(pair, pair.offset, noise_std=1e8,
                                 prior_std={"t_nw": 0.4, "t_ant": 2.0},
                                 prior_mean={"t_nw": 1.0, "t_ant": -5.0}, key=k)[0]
        )(keys)
        assert float(draws["t_nw"].std()) == pytest.approx(0.4, rel=0.06)
        assert float(draws["t_ant"].std()) == pytest.approx(2.0, rel=0.06)
        assert float(draws["t_nw"].mean()) == pytest.approx(1.0, abs=0.05)
        assert float(draws["t_ant"].mean()) == pytest.approx(-5.0, abs=0.2)

    def test_a_draw_is_not_the_mean_and_is_reproducible(self, pair, observed):
        kwargs = dict(noise_std=1.0, prior_std=PRIOR, key=jax.random.key(3))
        mean, _ = wiener_solve(pair, observed, noise_std=1.0, prior_std=PRIOR)
        first, _ = gcr_sample(pair, observed, **kwargs)
        second, _ = gcr_sample(pair, observed, **kwargs)
        for member in ("t_nw", "t_ant"):
            assert jnp.allclose(first[member], second[member])
        assert not jnp.allclose(first["t_ant"], mean["t_ant"], atol=1e-3)

    def test_a_mixed_real_complex_group_matches_a_dense_solve(self, mixed_block):
        truth = {
            "amp": jnp.array([7.0, -2.0]),
            "coeffs": jnp.array([1.0 + 2.0j, -0.5 - 1.25j, 3.0 + 0.75j]),
        }
        data = mixed_block.offset + mixed_block.forward(truth)
        priors = {"amp": 50.0, "coeffs": 20.0}
        solved, _ = wiener_solve(mixed_block, data, noise_std=0.1, prior_std=priors)
        expected = _dense_solve(mixed_block, data, 0.1, priors, ("amp", "coeffs"))
        assert jnp.allclose(solved["amp"], expected["amp"], rtol=1e-3, atol=1e-3)
        assert jnp.allclose(solved["coeffs"], expected["coeffs"], rtol=1e-3, atol=1e-3)
        # the imaginary half is what the R-linear/C-linear split exists for
        assert jnp.allclose(jnp.imag(solved["coeffs"]), jnp.imag(truth["coeffs"]),
                            atol=1e-1)

    def test_solving_a_group_is_jittable(self, pair, observed):
        run = jax.jit(lambda d: wiener_solve(pair, d, noise_std=1.0, prior_std=PRIOR)[0])
        direct, _ = wiener_solve(pair, observed, noise_std=1.0, prior_std=PRIOR)
        for member in ("t_nw", "t_ant"):
            assert jnp.allclose(run(observed)[member], direct[member], rtol=1e-4)

    def test_a_mismatched_observation_is_still_refused(self, pair):
        with pytest.raises(ParameterSpaceError, match="different"):
            wiener_solve(pair, jnp.zeros(pair.offset.shape[0]), noise_std=1.0,
                         prior_std=PRIOR)


# ------------------------------------------------------------- the priors ---


class TestGroupedPrior:
    def test_a_scalar_prior_std_is_refused(self, pair, observed):
        """One number for latents in kelvin and in dimensionless gain is a
        prior nobody declared."""
        with pytest.raises(ParameterSpaceError, match="one prior PER LATENT"):
            wiener_solve(pair, observed, noise_std=1.0, prior_std=5.0)

    def test_a_scalar_prior_mean_is_refused_too(self, pair, observed):
        with pytest.raises(ParameterSpaceError, match="one prior PER LATENT"):
            wiener_solve(pair, observed, noise_std=1.0, prior_std=PRIOR, prior_mean=1.0)

    def test_condition_estimate_refuses_a_scalar_prior_std_too(self, pair):
        with pytest.raises(ParameterSpaceError, match="one prior PER LATENT"):
            condition_estimate(pair, noise_std=1.0, prior_std=5.0)

    def test_a_prior_std_naming_a_non_member_is_refused(self, pair, observed):
        with pytest.raises(ParameterSpaceError, match="does not group"):
            wiener_solve(pair, observed, noise_std=1.0,
                         prior_std={**PRIOR, "gain": 0.1})

    def test_a_member_left_with_no_prior_at_all_is_NAMED(self, pair, observed):
        with pytest.raises(ParameterSpaceError, match=r"prior_std for \['t_ant'\]"):
            wiener_solve(pair, observed, noise_std=1.0, prior_std={"t_nw": 1e3})

    def test_a_declaration_fills_the_member_the_keyword_omits(
        self, three_linear, state, observed
    ):
        """A group mixing a declared latent with a prior-free one is honoured,
        not refused: the resolution is per member and independent."""
        declared = three_linear_space(
            {"t_ant": dist.Normal(jnp.zeros(3), jnp.full((3,), 1e4))}
        )
        block = linear_operator(declared, three_linear, state, names=("t_nw", "t_ant"))
        solved, _ = wiener_solve(block, observed, noise_std=1e-2,
                                 prior_std={"t_nw": 1e5})
        assert jnp.allclose(solved["t_ant"], TRUE_ANT, rtol=5e-3)
        assert jnp.allclose(solved["t_nw"], TRUE_NW, rtol=5e-3)

    def test_a_non_conjugate_member_is_named(self, three_linear, state, observed):
        declared = three_linear_space({"t_nw": dist.Uniform(jnp.zeros(2), jnp.ones(2))})
        block = linear_operator(declared, three_linear, state, names=("t_nw", "t_ant"))
        with pytest.raises(ParameterSpaceError, match="latent 't_nw' declares a Uniform"):
            wiener_solve(block, observed, noise_std=1.0, prior_std={"t_ant": 1e4})

    def test_a_contradiction_is_measured_against_the_member_that_declared_it(
        self, three_linear, state, observed
    ):
        """Both members declare and only ONE keyword contradicts, so the message
        has to name that one. Measured against the other member's declaration it
        would raise on the agreeing keyword and pass the contradicting one."""
        declared = three_linear_space({
            "t_nw": dist.Normal(jnp.zeros(2), jnp.full((2,), 0.5)),
            "t_ant": dist.Normal(jnp.zeros(3), jnp.full((3,), 7.0)),
        })
        block = linear_operator(declared, three_linear, state, names=("t_nw", "t_ant"))
        with pytest.raises(ParameterSpaceError, match="latent 't_ant' declares"):
            wiener_solve(block, observed, noise_std=1.0,
                         prior_std={"t_nw": jnp.full((2,), 0.5),
                                    "t_ant": jnp.full((3,), 99.0)})

    def test_an_agreeing_keyword_is_accepted_for_both_members(
        self, three_linear, state, observed
    ):
        declared = three_linear_space({
            "t_nw": dist.Normal(jnp.zeros(2), jnp.full((2,), 1e3)),
            "t_ant": dist.Normal(jnp.zeros(3), jnp.full((3,), 1e4)),
        })
        block = linear_operator(declared, three_linear, state, names=("t_nw", "t_ant"))
        solved, _ = wiener_solve(block, observed, noise_std=1.0,
                                 prior_std={"t_nw": jnp.full((2,), 1e3),
                                            "t_ant": jnp.full((3,), 1e4)})
        assert jnp.all(jnp.isfinite(solved["t_nw"]))

    def test_a_declared_group_solves_with_no_keywords_at_all(
        self, three_linear, state, observed
    ):
        declared = three_linear_space({
            "t_nw": dist.Normal(jnp.zeros(2), jnp.full((2,), 1e4)),
            "t_ant": dist.Normal(jnp.zeros(3), jnp.full((3,), 1e4)),
        })
        block = linear_operator(declared, three_linear, state, names=("t_nw", "t_ant"))
        solved, _ = wiener_solve(block, observed, noise_std=1e-2)
        assert jnp.allclose(solved["t_ant"], TRUE_ANT, rtol=5e-3)

    def test_a_hand_built_group_carrying_a_BARE_prior_is_refused(self, pair, observed):
        """A single distribution standing for a whole group is a statement about
        latents in different units that nobody made."""
        smuggled = dataclasses.replace(pair, prior=dist.Normal(0.0, 1.0))
        with pytest.raises(ParameterSpaceError, match="one entry per member"):
            wiener_solve(smuggled, observed, noise_std=1.0, prior_std=PRIOR)

    def test_a_hand_built_group_with_prior_None_solves_from_keywords(self, pair,
                                                                     observed):
        plain = dataclasses.replace(pair, prior=None)
        solved, _ = wiener_solve(plain, observed, noise_std=1e-2,
                                 prior_std={"t_nw": 1e5, "t_ant": 1e5})
        assert jnp.allclose(solved["t_ant"], TRUE_ANT, rtol=2e-3)

    def test_a_hand_built_group_whose_prior_dict_misses_a_member_is_refused(
        self, pair, observed
    ):
        partial = dataclasses.replace(pair, prior={"t_nw": None})
        with pytest.raises(ParameterSpaceError, match="one entry per member"):
            wiener_solve(partial, observed, noise_std=1.0, prior_std=PRIOR)


# -------------------------------------------------------------- the guards --


class TestGroupedLinearityCheck:
    def test_a_jointly_linear_group_passes(self, space, three_linear, state):
        errors = check_linearity(space, three_linear, state, names=("t_nw", "t_ant"))
        assert all(err < 1e-4 for err in errors.values())

    def test_a_BILINEAR_pair_is_refused_although_each_half_passes(
        self, space, three_linear, state
    ):
        """The guard grouping makes possible.

        ``gain`` and ``t_ant`` are each affine given the other — both pass the
        one-latent check, which is why no per-block diagnostic ever complains
        about the partition — and their product is not affine in the pair. Only
        the JOINT probe can say so.
        """
        check_linearity(space, three_linear, state, "gain")
        check_linearity(space, three_linear, state, "t_ant")
        with pytest.raises(ParameterSpaceError, match="not affine in them JOINTLY"):
            check_linearity(space, three_linear, state, names=("gain", "t_ant"))

    def test_linear_operator_refuses_the_bilinear_group_at_export(
        self, space, three_linear, state
    ):
        with pytest.raises(ParameterSpaceError, match="not affine"):
            linear_operator(space, three_linear, state, names=("gain", "t_ant"))

    def test_check_false_lets_the_bilinear_group_through(self, space, three_linear,
                                                         state):
        """The documented bargain, so the refusal above is the CHECK's doing and
        not an accident of the export path."""
        block = linear_operator(space, three_linear, state, names=("gain", "t_ant"),
                                check=False)
        assert block.names == ("gain", "t_ant")

    def test_permuting_names_gives_the_same_linearity_errors(self, space, three_linear,
                                                             state):
        assert check_linearity(
            space, three_linear, state, names=("t_nw", "t_ant")
        ) == check_linearity(space, three_linear, state, names=("t_ant", "t_nw"))

    def test_an_integer_member_is_refused(self, three_linear, state):
        integral = ParameterSpace(
            latents=[
                Latent("t_ant", init=TRUE_ANT, linear=True),
                Latent("t_nw", init=jnp.array([2, 3]), linear=True),
                Latent("gain", init=GAIN0, linear=True),
            ],
            bindings=[
                Bind("t_ant", into=lambda p: p["t_ant"].coeff),
                Bind("t_nw", into=lambda p: p["t_nw"].coeff, fn=lambda c: c * 1.0),
                Bind("gain", into=lambda p: p["gain"].gain),
            ],
        )
        with pytest.raises(ParameterSpaceError, match="floating-point or complex"):
            check_linearity(integral, three_linear, state, names=("t_nw", "t_ant"))

    def test_an_integer_LATENT_is_refused_alone_too(self, three_linear, state):
        """The same helper serves the one-latent path, which had no test of its
        own — so the group's version would have been the only thing holding it."""
        integral = ParameterSpace.direct(
            "t_nw", init=jnp.array([2, 3]), into=lambda p: p["t_nw"].coeff,
            fn=lambda c: c * 1.0, linear=True,
        )
        with pytest.raises(ParameterSpaceError, match="floating-point or complex"):
            check_linearity(integral, three_linear, state, "t_nw")

    def test_name_and_names_together_are_refused(self, space, three_linear, state):
        with pytest.raises(ParameterSpaceError, match="name= OR names="):
            check_linearity(space, three_linear, state, "t_nw",
                            names=("t_nw", "t_ant"))

    def test_check_linearity_honours_at_for_a_group(self, space, three_linear, state):
        errors = check_linearity(space, three_linear, state, names=("t_nw", "t_ant"),
                                 at={"gain": 3.0 * GAIN0})
        assert all(err < 1e-4 for err in errors.values())


@pytest.fixture
def seen_and_blind(state):
    """A group whose ``λ_min`` is EXACTLY the loosest prior's curvature."""
    pipeline = Pipeline(
        SeenAndBlindOperator(seen=jnp.zeros(()), blind=jnp.zeros(2)),
        names=("load",),
    )
    space = ParameterSpace(
        latents=[
            Latent("seen", init=jnp.array(250.0), linear=True),
            Latent("blind", init=jnp.array([1.0, -2.0]), linear=True),
        ],
        bindings=[
            Bind("seen", into=lambda p: p["load"].seen),
            Bind("blind", into=lambda p: p["load"].blind),
        ],
    )
    return linear_operator(space, pipeline, state, names=("seen", "blind"))


class TestGroupedConditioning:
    def test_the_condition_estimate_matches_a_dense_eigenvalue_computation(self, pair):
        priors = {"t_nw": 3.0, "t_ant": 40.0}
        _, _, dense = _dense_normal(pair, 0.5, priors, ("t_nw", "t_ant"))
        estimated = condition_estimate(pair, noise_std=0.5, prior_std=priors)
        assert float(estimated) == pytest.approx(float(jnp.linalg.cond(dense)), rel=0.1)

    def test_the_floor_uses_the_LOOSEST_prior_in_the_group(self, seen_and_blind):
        """``λ_min`` can never fall below ``1/max(prior_variance)`` over the
        group, and taking the TIGHTEST prior instead would floor the estimate
        above the true ``λ_min`` and report a condition number orders of
        magnitude too small — an over-confident guard, which is the direction
        that costs something.

        ``blind`` has an exactly zero Jacobian, so ``λ_min`` IS
        ``1/prior_std['blind']²`` = 1e-6, which is below what ``λ_max - spread``
        can resolve against a ``λ_max`` of 6404 in any precision this suite runs
        in. The floor is therefore what decides. Taking the tightest prior it
        would decide ``1/0.5²`` = 4 and report κ = 1600 for an operator whose κ
        is 6.4e9, so the assertion is bracketed rather than pinned: anything
        above 1e5 could not have come from the wrong floor, and the analytic κ
        is an upper bound the estimate cannot exceed.
        """
        kappa = float(condition_estimate(seen_and_blind, noise_std=0.1,
                                         prior_std={"seen": 0.5, "blind": 1e3}))
        largest = (N_TIME * N_FREQ) / 0.1**2 + 1.0 / 0.5**2
        assert 1e5 < kappa <= 1.01 * largest * 1e3**2, kappa


class TestGroupedVsAlternating:
    """The measured failure, reproduced under control — and fixed.

    ``t_nw``'s spectral shapes are 85% ``t_ant``'s, which is the regime the
    whole feature is about: the joint model is IDENTIFIED and the PARTITION is
    what fails. Alternation contracts at the rate of the correlation between
    the blocks while reporting a converged residual and a condition number of
    ~1 the entire way down.
    """

    TRUTH: ClassVar[dict] = {
        "t_ant": jnp.array([2800.0, -160.0]),
        "t_nw": jnp.array([250.0, -18.0]),
    }
    WIDE: ClassVar[dict] = {"t_ant": 1e5, "t_nw": 1e5}

    @pytest.fixture
    def near_degenerate(self):
        pipeline = Pipeline(
            BasisOperator(coeff=jnp.zeros(2), basis=_SHAPES[:, :2]),
            AddBasisOperator(coeff=jnp.zeros(2), basis=B_NW_TILTED),
            names=("t_ant", "t_nw"),
        )
        space = ParameterSpace(
            latents=[
                Latent("t_ant", init=jnp.zeros(2), linear=True),
                Latent("t_nw", init=jnp.zeros(2), linear=True),
            ],
            bindings=[
                Bind("t_ant", into=lambda p: p["t_ant"].coeff),
                Bind("t_nw", into=lambda p: p["t_nw"].coeff),
            ],
        )
        return space, pipeline

    def test_one_grouped_solve_beats_many_alternating_ones(self, near_degenerate,
                                                           state):
        space, pipeline = near_degenerate
        group = linear_operator(space, pipeline, state, names=("t_nw", "t_ant"))
        data = group.offset + group.forward(self.TRUTH)

        joint, _ = wiener_solve(
            group, data, noise_std=1e-2, prior_std=self.WIDE,
            tol=1e-10, maxiter=2000, require_convergence=None,
        )
        joint_error = max(
            float(jnp.max(jnp.abs(joint[m] - self.TRUTH[m]))) for m in self.TRUTH
        )

        values = {"t_ant": jnp.zeros(2), "t_nw": jnp.zeros(2)}
        kappas, residuals = {}, {}
        for _ in range(20):
            for member in ("t_nw", "t_ant"):
                block = linear_operator(space, pipeline, state, member, at=values,
                                        check=False)
                kappas[member] = float(
                    condition_estimate(block, noise_std=1e-2,
                                       prior_std=self.WIDE[member])
                )
                solved, residual = wiener_solve(
                    block, data, noise_std=1e-2, prior_std=self.WIDE[member],
                    tol=1e-10, maxiter=2000, require_convergence=None,
                )
                residuals[member] = float(residual)
                values = {**values, member: solved}
        alternating_error = max(
            float(jnp.max(jnp.abs(values[m] - self.TRUTH[m]))) for m in self.TRUTH
        )

        # Every per-block guard reports green the whole way down...
        assert max(kappas.values()) < 10.0, kappas
        assert max(residuals.values()) < 1e-4, residuals
        # ... and the alternating answer is hundreds of kelvin out, while the
        # single grouped solve lands on the truth.
        assert joint_error < 1.0, joint_error
        assert alternating_error > 100.0, (joint_error, alternating_error)

    def test_the_grouped_block_reports_the_conditioning_the_blocks_hide(
        self, near_degenerate, state
    ):
        space, pipeline = near_degenerate
        group = linear_operator(space, pipeline, state, names=("t_nw", "t_ant"))
        joint = float(condition_estimate(group, noise_std=1e-2, prior_std=self.WIDE))
        singles = [
            float(condition_estimate(
                linear_operator(space, pipeline, state, member),
                noise_std=1e-2, prior_std=self.WIDE[member],
            ))
            for member in ("t_nw", "t_ant")
        ]
        assert max(singles) < 10.0, singles
        assert joint > 30.0 * max(singles), (joint, singles)

    def test_identifiability_agrees_that_the_JOINT_set_is_the_weak_one(
        self, near_degenerate, state
    ):
        """The cross-check from the instrument that was built for exactly this.

        Each block on its own is perfectly identified — its two shapes are
        orthonormal, so ``weakest_identified`` is 1. The joint set is identified
        too, ``nullity`` 0, and more than ten times less well. That is the
        honest statement: the model is fine and the partition is not.
        """
        space, pipeline = near_degenerate
        reports = {
            names: identifiability(space, pipeline, state, names=names)
            for names in (("t_nw",), ("t_ant",), ("t_nw", "t_ant"))
        }
        weakest = {names: report.weakest_identified for names, report in reports.items()}
        assert all(report.nullity == 0 for report in reports.values())
        assert reports[("t_nw", "t_ant")].n_par == 4
        assert weakest[("t_nw",)] > 0.9 and weakest[("t_ant",)] > 0.9, weakest
        assert weakest[("t_nw", "t_ant")] < 0.1, weakest


def test_a_hand_assembled_group_needs_no_ParameterSpace():
    """The block is a linear-algebra handle; nothing about a group requires it
    to have come from a declaration. Pinned because the x64 regression script
    for single blocks builds one by hand and the grouped one must be able to."""
    matrix = jnp.array([[1.0, 2.0], [0.5, -1.0], [3.0, 0.25]])

    def forward(x):
        return matrix @ x["a"] + jnp.array([1.0, -2.0, 0.5]) * x["b"]

    zero = {"a": jnp.zeros(2), "b": jnp.zeros(())}
    block = LinearBlock(
        name=("a", "b"),
        shape={"a": (2,), "b": ()},
        dtype={"a": jnp.float32, "b": jnp.float32},
        offset=jnp.zeros(3),
        forward=forward,
        adjoint=lambda y: jax.vjp(forward, zero)[1](y)[0],
    )
    truth = {"a": jnp.array([4.0, -1.5]), "b": jnp.array(9.0)}
    solved, _ = wiener_solve(block, forward(truth), noise_std=1e-3,
                             prior_std={"a": 1e4, "b": 1e4})
    assert jnp.allclose(solved["a"], truth["a"], rtol=1e-2)
    assert jnp.allclose(solved["b"], truth["b"], rtol=1e-2)
