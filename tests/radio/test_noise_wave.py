"""NoiseWaveOperator: the noise-wave system temperature as a rheplicant operator."""

import contextlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("rhino_cal_jax", reason="rhino-cal-jax not installed")

from rheplicant.core.coordinates import Coordinates  # noqa: E402
from rheplicant.core.errors import StateValidationError  # noqa: E402
from rheplicant.core.pipeline import Pipeline  # noqa: E402
from rheplicant.core.state import State  # noqa: E402
from rheplicant.inference import Bind, Latent, ParameterSpace  # noqa: E402
from rheplicant.inference.identifiability import (  # noqa: E402
    DEFAULT_RANK_RTOL,
    identifiability,
)
from rheplicant.radio import NoiseWaveOperator  # noqa: E402

# n_time != n_freq != n_source, and no two of them equal. Deliberate: a square
# grid makes a per-time vector indistinguishable in SHAPE from a per-frequency
# spectrum, which is precisely the confusion TestTemperatureShapes exists to
# catch — and this project has twice shipped a test blinded by a symmetric
# fixture.
N_TIME, N_FREQ = 12, 4

# Three physically distinct loads, in the order the switch indexes them: a
# structured antenna reflection, a near-matched ambient load, and a short-like
# reflection of opposite sign. Three is the minimum that makes the per-channel
# noise-wave system full rank -- see TestIdentifiability.
G_SRC_RE = np.array([[0.30, 0.28, 0.26, 0.24],
                     [0.02, 0.01, 0.00, -0.01],
                     [-0.60, -0.62, -0.64, -0.66]])
G_SRC_IM = np.array([[0.10, 0.05, 0.00, -0.05],
                     [0.00, 0.03, 0.02, 0.01],
                     [0.15, -0.10, 0.05, 0.20]])
N_SOURCE = G_SRC_RE.shape[0]
G_REC_RE = np.full(N_FREQ, 0.08)
G_REC_IM = np.full(N_FREQ, -0.03)

# rheplicant's suite runs float32; jax_enable_x64 is process-global and cannot
# be flipped per-module, so the tolerance is read from the active precision.
# The same file then passes under both `pytest` and `JAX_ENABLE_X64=1 pytest`.
RTOL = 1e-13 if jax.config.read("jax_enable_x64") else 2e-6


def make_operator(**overrides):
    kwargs = dict(
        t_unc=jnp.array(250.0), t_cos=jnp.array(30.0),
        t_sin=jnp.array(-40.0), t_rx=jnp.array(290.0),
        gamma_src_re=jnp.asarray(G_SRC_RE), gamma_src_im=jnp.asarray(G_SRC_IM),
        gamma_rec_re=jnp.asarray(G_REC_RE), gamma_rec_im=jnp.asarray(G_REC_IM),
    )
    kwargs.update(overrides)
    return NoiseWaveOperator(**kwargs)


def make_state(switch=None, data=None):
    extra = {} if switch is None else {"receiver_input": jnp.asarray(switch)}
    return State(
        data=jnp.full((N_TIME, N_FREQ), 300.0) if data is None else data,
        coords=Coordinates(
            time=jnp.arange(float(N_TIME)),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
            extra=extra,
        ),
    )


class TestForward:
    def test_it_reproduces_rhino_cal_jax_directly(self):
        import rhino_cal_jax as rcj

        switch = np.arange(N_TIME) % N_SOURCE
        out = make_operator()(make_state(switch)).data

        cycle = rcj.SwitchCycle(
            source_index=jnp.asarray(switch),
            labels=tuple(str(i) for i in range(N_SOURCE)),
        )
        stacked = cycle.gather(
            rcj.couplings(
                jnp.asarray(G_SRC_RE + 1j * G_SRC_IM),
                jnp.asarray(G_REC_RE + 1j * G_REC_IM),
            ).stacked
        )
        expected = (
            300.0 * stacked[..., 0] + 250.0 * stacked[..., 1]
            + 30.0 * stacked[..., 2] - 40.0 * stacked[..., 3] + 290.0
        )
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected), rtol=RTOL)

    def test_a_matched_single_source_reduces_to_t_src_plus_t_rx(self):
        op = make_operator(
            gamma_src_re=jnp.zeros((1, N_FREQ)), gamma_src_im=jnp.zeros((1, N_FREQ)),
            gamma_rec_re=jnp.zeros(N_FREQ), gamma_rec_im=jnp.zeros(N_FREQ),
        )
        out = op(make_state(np.zeros(N_TIME, dtype=int))).data
        np.testing.assert_allclose(np.asarray(out), 300.0 + 290.0, rtol=RTOL)

    def test_a_single_source_needs_no_switch_array(self):
        op = make_operator(
            gamma_src_re=jnp.zeros((1, N_FREQ)), gamma_src_im=jnp.zeros((1, N_FREQ)),
        )
        assert op(make_state()).data.shape == (N_TIME, N_FREQ)

    def test_per_channel_noise_wave_temperatures_are_accepted(self):
        op = make_operator(t_unc=jnp.linspace(240.0, 260.0, N_FREQ))
        out = op(make_state(np.arange(N_TIME) % N_SOURCE)).data
        assert out.shape == (N_TIME, N_FREQ)


class TestRejections:
    def test_several_sources_without_a_switch_array_is_refused(self):
        """Silently using source 0 for every sample is finite and wrong."""
        with pytest.raises(StateValidationError, match="receiver_input"):
            make_operator()(make_state())

    def test_a_switch_longer_than_the_data_is_refused(self):
        with pytest.raises(StateValidationError, match="n_time"):
            make_operator()(make_state(np.zeros(N_TIME + 1, dtype=int)))

    def test_mismatched_gamma_real_and_imaginary_shapes_are_refused(self):
        with pytest.raises(StateValidationError, match="gamma_src"):
            make_operator(gamma_src_im=jnp.zeros((N_SOURCE + 1, N_FREQ)))

    def test_a_gamma_whose_channels_do_not_match_the_receiver_is_refused(self):
        with pytest.raises(StateValidationError, match="n_freq"):
            make_operator(gamma_rec_re=jnp.zeros(N_FREQ + 1),
                          gamma_rec_im=jnp.zeros(N_FREQ + 1))

    def test_a_one_dimensional_gamma_src_is_refused(self):
        with pytest.raises(StateValidationError, match="2D"):
            make_operator(gamma_src_re=jnp.zeros(N_FREQ),
                          gamma_src_im=jnp.zeros(N_FREQ))

    def test_mismatched_gamma_rec_real_and_imaginary_shapes_are_refused(self):
        """The receiver's counterpart of the ``gamma_src`` check above.

        Both halves of the reflection coefficient are separate leaves so that
        each stays real and differentiable; nothing in the type system ties
        their shapes together, and a mismatch here would otherwise broadcast
        into a complex gamma of the wrong length.

        ``N_FREQ + 1`` rather than a second axis: this must fail on LENGTH
        while both parts are still 1-D, or it would be indistinguishable from
        the rank check below.
        """
        with pytest.raises(StateValidationError, match="gamma_rec real/imaginary"):
            make_operator(gamma_rec_im=jnp.zeros(N_FREQ + 1))

    def test_a_two_dimensional_gamma_rec_is_refused(self):
        """Reaching this needs BOTH parts reshaped -- see the order test."""
        with pytest.raises(StateValidationError, match="gamma_rec_re must be 1D"):
            make_operator(gamma_rec_re=jnp.zeros((2, N_FREQ)),
                          gamma_rec_im=jnp.zeros((2, N_FREQ)))

    def test_the_shape_agreement_check_precedes_the_rank_check(self):
        """A raise-order dependency the two tests above are built on.

        With only ``gamma_rec_re`` made 2-D, the parts disagree AND the rank
        is wrong. Which sentence comes back is decided by the order of the
        two ``if``s in ``__check_init__``, not by which fault is worse. If
        that order is ever swapped, the rank test above stops reaching the
        line it was written for and starts passing for the wrong reason --
        silently, because both raises are ``StateValidationError`` and both
        mention ``gamma_rec``.
        """
        with pytest.raises(StateValidationError) as excinfo:
            make_operator(gamma_rec_re=jnp.zeros((2, N_FREQ)))
        assert "real/imaginary" in str(excinfo.value), str(excinfo.value)
        assert "must be 1D" not in str(excinfo.value), str(excinfo.value)

    def test_a_well_formed_receiver_gamma_is_accepted(self):
        """The arm the three refusals above cannot distinguish themselves from.

        Per-channel and non-constant: a flat gamma would survive an operator
        that collapsed the spectrum to its first channel.
        """
        operator = make_operator(
            gamma_rec_re=jnp.linspace(0.08, 0.05, N_FREQ),
            gamma_rec_im=jnp.linspace(-0.03, -0.01, N_FREQ),
        )
        out = operator(make_state(np.arange(N_TIME) % N_SOURCE))
        assert out.data.shape == (N_TIME, N_FREQ)
        assert bool(jnp.all(jnp.isfinite(out.data)))

    def test_data_whose_channels_disagree_with_gamma_is_refused(self):
        state = make_state(np.arange(N_TIME) % N_SOURCE,
                           data=jnp.full((N_TIME, N_FREQ + 2), 300.0))
        with pytest.raises(StateValidationError, match="n_freq"):
            make_operator()(state)


class TestTemperatureShapes:
    """The shape contract of the four temperature leaves.

    ``rhino_cal_jax.system_temperature`` broadcasts each temperature against a
    ``(n_time, n_freq)`` coupling, so four shapes are legal and everything else
    used to arrive as a raw broadcasting ``ValueError`` at CALL time, naming
    neither the leaf nor the convention. These pin the named refusal.

    Every fixture here is deliberately NON-SQUARE — ``n_time=12``,
    ``n_freq=4``, ``n_source=3``, no two equal. On a square grid a per-time
    vector and a per-frequency spectrum have the identical shape, so the exact
    confusion this guard exists to catch would be invisible to the test.
    ``test_the_square_grid_ambiguity_is_real_and_cannot_be_guarded`` is the one
    place a square grid appears, and it is there to show what survives.
    """

    LEAVES = ("t_unc", "t_cos", "t_sin", "t_rx")

    @pytest.mark.parametrize("leaf", LEAVES)
    @pytest.mark.parametrize(
        "shape",
        [(), (N_FREQ,), (N_TIME, 1), (N_TIME, N_FREQ), (1, N_FREQ), (1, 1)],
        ids=["scalar", "per-freq", "time-column", "per-cell", "one-row", "one-cell"],
    )
    def test_every_documented_shape_is_accepted(self, leaf, shape):
        op = make_operator(**{leaf: jnp.full(shape, 250.0)})
        assert op(make_state(np.arange(N_TIME) % N_SOURCE)).data.shape == (N_TIME, N_FREQ)

    @pytest.mark.parametrize("leaf", LEAVES)
    def test_a_bare_per_time_vector_is_refused_by_name(self, leaf):
        """The live hazard: time-varying temperatures become the norm next.

        A length-``n_time`` vector is not a spectrum, and read as one it
        broadcasts along the wrong axis. Parametrised over all four leaves
        because each one needs its own guard — three checked leaves and one
        unchecked is the state this test closes.
        """
        with pytest.raises(StateValidationError, match=leaf):
            make_operator(**{leaf: jnp.linspace(240.0, 260.0, N_TIME)})

    def test_the_refusal_names_the_column_convention_and_the_residual_ambiguity(self):
        """A guard that says 'no' without saying 'do this instead' is half a guard."""
        with pytest.raises(StateValidationError) as excinfo:
            make_operator(t_unc=jnp.linspace(240.0, 260.0, N_TIME))
        message = str(excinfo.value)
        assert "(n_time, 1)" in message
        assert "n_time == n_freq" in message
        assert f"n_freq={N_FREQ}" in message

    def test_a_three_dimensional_temperature_is_refused(self):
        with pytest.raises(StateValidationError, match="3-D"):
            make_operator(t_unc=jnp.zeros((2, N_TIME, N_FREQ)))

    def test_a_one_dimensional_temperature_of_the_wrong_length_is_refused(self):
        with pytest.raises(StateValidationError, match="per-FREQUENCY"):
            make_operator(t_cos=jnp.zeros(N_FREQ + 1))

    @pytest.mark.parametrize(
        "shape", [(N_TIME, 2), (N_FREQ, N_TIME)], ids=["wrong-width", "transposed"]
    )
    def test_a_two_dimensional_temperature_whose_trailing_axis_is_not_frequency(self, shape):
        with pytest.raises(StateValidationError, match="trailing axis"):
            make_operator(t_sin=jnp.zeros(shape))

    def test_a_bare_vector_is_read_along_frequency_and_a_column_along_time(self):
        """The convention as an equality, not as prose.

        A ``(n_freq,)`` spectrum must equal its own ``(n_time, n_freq)`` tiling,
        and a ``(n_time, 1)`` column must equal ITS tiling — which is the whole
        content of "1-D means per-frequency; use a column for per-time".
        """
        state = make_state(np.arange(N_TIME) % N_SOURCE)

        spectrum = jnp.linspace(240.0, 260.0, N_FREQ)
        np.testing.assert_allclose(
            np.asarray(make_operator(t_unc=spectrum)(state).data),
            np.asarray(
                make_operator(t_unc=jnp.broadcast_to(spectrum, (N_TIME, N_FREQ)))(state).data
            ),
            rtol=RTOL,
        )

        column = jnp.linspace(240.0, 260.0, N_TIME)[:, None]
        np.testing.assert_allclose(
            np.asarray(make_operator(t_unc=column)(state).data),
            np.asarray(
                make_operator(t_unc=jnp.broadcast_to(column, (N_TIME, N_FREQ)))(state).data
            ),
            rtol=RTOL,
        )

    def test_the_square_grid_ambiguity_is_real_and_cannot_be_guarded(self):
        """What the guard does NOT catch, so nobody assumes it does.

        The ONLY square fixture in this file, and it is here to be a
        counter-example. When ``n_time == n_freq`` a per-time vector and a
        per-frequency spectrum are the same shape; the guard accepts both, reads
        both as spectra, and the caller who meant per-time gets a finite,
        correctly-shaped, wrong ``T_sys``. Nothing short of a units system can
        tell them apart, which is exactly why the refusal message points at the
        ``(n_time, 1)`` column rather than promising to catch this.
        """
        n = N_FREQ
        op_kwargs = dict(
            t_cos=jnp.array(30.0), t_sin=jnp.array(-40.0), t_rx=jnp.array(290.0),
            gamma_src_re=jnp.asarray(G_SRC_RE[:1]), gamma_src_im=jnp.asarray(G_SRC_IM[:1]),
            gamma_rec_re=jnp.asarray(G_REC_RE), gamma_rec_im=jnp.asarray(G_REC_IM),
        )
        square = State(
            data=jnp.full((n, n), 300.0),
            coords=Coordinates(time=jnp.arange(float(n)),
                               freq=jnp.linspace(60e6, 85e6, n),
                               extra={"receiver_input": jnp.zeros(n, dtype=int)}),
        )
        meant_per_time = jnp.linspace(240.0, 260.0, n)

        # Accepted — the guard cannot refuse it, and does not pretend to.
        as_spectrum = NoiseWaveOperator(t_unc=meant_per_time, **op_kwargs)(square).data
        as_column = NoiseWaveOperator(t_unc=meant_per_time[:, None], **op_kwargs)(square).data

        assert as_spectrum.shape == as_column.shape == (n, n)
        assert not np.allclose(np.asarray(as_spectrum), np.asarray(as_column))


class TestIdentifiability:
    """What switching buys, counted per frequency channel, for k = 3.

    Each switch position contributes exactly ONE equation per channel, so with
    ``T_rx`` HELD KNOWN — the ``k = 3`` case, which is what these three tests
    fix by varying only ``t_unc, t_cos, t_sin`` — rank is
    ``min(n_src, 3) * n_freq``.

    The general rule is ``min(n_src, k) * n_freq`` over the ``k`` free
    temperature families, and it holds only while the temperatures are free per
    channel. :class:`TestPerChannelRankRule` and
    :class:`TestBasisRegimeBreaksTheRule` measure both halves of that with
    :func:`~rheplicant.inference.identifiability.identifiability`; these three
    are the hand-rolled Jacobian that predates it and is kept as an independent
    check of the same number.

    This deliberately does NOT use scalar temperatures: those are fully
    identified by a single load, so a scalar test passes with one source and
    demonstrates nothing about switching.
    """

    def _per_channel_jacobian(self, n_source: int) -> np.ndarray:
        switch = np.arange(N_TIME) % n_source

        def predict(flat):  # flat: (3, n_freq) -- t_unc, t_cos, t_sin per channel
            op = make_operator(
                t_unc=flat[0], t_cos=flat[1], t_sin=flat[2],
                gamma_src_re=jnp.asarray(G_SRC_RE[:n_source]),
                gamma_src_im=jnp.asarray(G_SRC_IM[:n_source]),
            )
            return op(make_state(switch)).data.ravel()

        jac = jax.jacobian(predict)(jnp.zeros((3, N_FREQ)))
        return np.asarray(jac).reshape(-1, 3 * N_FREQ)

    @pytest.mark.parametrize("n_source,expected_rank", [(1, N_FREQ), (2, 2 * N_FREQ)])
    def test_too_few_loads_leave_the_system_rank_deficient(self, n_source, expected_rank):
        jac = self._per_channel_jacobian(n_source)
        assert np.linalg.matrix_rank(jac) == expected_rank
        assert expected_rank < 3 * N_FREQ

    def test_three_loads_make_the_per_channel_system_full_rank(self):
        assert np.linalg.matrix_rank(self._per_channel_jacobian(3)) == 3 * N_FREQ

    def test_the_operator_reproduces_its_own_truth(self):
        """Sanity: the forward model is deterministic and Gamma is wired per source."""
        switch = np.arange(N_TIME) % N_SOURCE
        truth = make_operator()(make_state(switch)).data

        def predict(t_nw):
            return make_operator(t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2])(
                make_state(switch)
            ).data.ravel()

        np.testing.assert_allclose(
            np.asarray(predict(jnp.array([250.0, 30.0, -40.0]))),
            np.asarray(truth).ravel(), rtol=RTOL,
        )


class TestTransforms:
    def test_gradients_reach_every_temperature_and_gamma(self):
        switch = np.arange(N_TIME) % N_SOURCE
        op = make_operator()

        def loss(operator):
            return jnp.sum(operator(make_state(switch)).data)

        grads = jax.grad(loss)(op)
        for leaf in jax.tree_util.tree_leaves(grads):
            assert np.all(np.isfinite(np.asarray(leaf)))

    def test_the_operator_jits(self):
        switch = np.arange(N_TIME) % N_SOURCE
        out = jax.jit(lambda o, s: o(s).data)(make_operator(), make_state(switch))
        assert out.shape == (N_TIME, N_FREQ)


# ------------------------------------------------------- the rank rule, measured --
#
# Everything below establishes the module docstring's rank claim by MEASUREMENT,
# through `identifiability()`, rather than by argument. The claim is the number a
# real experiment picks its switching cadence from, and the previous version of it
# was wrong twice: it hard-coded three free temperature families when `t_rx` is a
# leaf like the other three, and it was stated as though it survived a change of
# parameterisation, which it does not.

N_TIME_RANK = 11   # prime, and equal to no other dimension here
N_BASIS = 3
N_SRC_MAX = 5


@contextlib.contextmanager
def _float64_leaves():
    """Build the fixture's OWN arrays in double precision, then restore.

    ``identifiability()`` already forces float64 for the Jacobian and the SVD.
    It cannot restore digits that were thrown away earlier, and the operator's
    ``Gamma`` leaves are earlier: built in the suite's default float32 they
    carry ~1e-7 relative noise into a diagnostic whose verdict turns on 1e-8.

    That is not hypothetical here — it changes an answer in this very file. The
    one-load basis model below measures rank 5 with a comfortable
    ``weakest_identified`` of 1.3e-3 in double precision, and rank **6** with
    ``weakest_identified`` of 3.0e-8 in single, three times the default
    tolerance and therefore a coin-flip.
    ``test_float32_gamma_leaves_make_the_basis_verdict_untrustworthy`` pins that
    gap; this context manager is what keeps every other test on the right side
    of it.

    ``jax.config`` is process-global, so the ``finally`` is load-bearing.
    """
    was = jax.config.read("jax_enable_x64")
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", was)


def _gamma_bank(n_freq: int) -> tuple[np.ndarray, np.ndarray]:
    """Five loads whose reflection spectra differ in SHAPE, not just in level.

    Row 0 is deliberately linear in channel index and row 2 likewise: a
    low-order ``Gamma`` is what makes the basis-regime bound overstate, and
    ``test_a_low_order_gamma_falls_short_of_the_bound`` needs one to exist.
    Rows 1, 3 and 4 carry genuine curvature.
    """
    x = np.linspace(-1.0, 1.0, n_freq)
    ramp = np.arange(n_freq) / n_freq
    re = np.stack([
        0.30 - 0.06 * ramp,
        0.55 * np.cos(np.linspace(0.0, 2.0, n_freq)),
        -0.60 - 0.06 * x,
        0.20 * np.sin(np.linspace(0.5, 3.5, n_freq)),
        -0.35 + 0.25 * x**2,
    ])
    im = np.stack([
        0.10 - 0.05 * ramp,
        0.05 * np.sin(np.linspace(0.0, 4.0, n_freq)),
        0.15 - 0.10 * x,
        -0.30 * np.cos(np.linspace(0.2, 2.2, n_freq)),
        0.22 * x,
    ])
    return re, im


FAMILIES = ("t_unc", "t_cos", "t_sin", "t_rx")


def _into(name: str):
    return (lambda p, _n=name: getattr(p["noise_wave"], _n),)


def _rank_report(*, loads, k, n_freq, basis=None, dtype=None):
    """``identifiability()`` of a noise-wave model with ``k`` free families.

    ``basis=None`` is the per-CELL regime: each free family is its own
    ``(n_freq,)`` latent. A ``(n_basis, n_freq)`` ``basis`` puts each family's
    ``(n_basis,)`` coefficients in front of it instead — the regime the next
    tranche makes ordinary, and the one the per-channel counting does not
    survive.
    """
    n_src = len(loads)
    g_re, g_im = _gamma_bank(n_freq)
    cast = (lambda a: jnp.asarray(a, dtype=dtype)) if dtype else jnp.asarray
    op = NoiseWaveOperator(
        t_unc=jnp.zeros(n_freq), t_cos=jnp.zeros(n_freq),
        t_sin=jnp.zeros(n_freq), t_rx=jnp.zeros(n_freq),
        gamma_src_re=cast(g_re[list(loads)]), gamma_src_im=cast(g_im[list(loads)]),
        gamma_rec_re=cast(np.full(n_freq, 0.08)),
        gamma_rec_im=cast(np.full(n_freq, -0.03)),
    )
    state = State(
        data=jnp.full((N_TIME_RANK, n_freq), 300.0),
        coords=Coordinates(
            time=jnp.arange(float(N_TIME_RANK)),
            freq=jnp.linspace(60e6, 85e6, n_freq),
            extra={"receiver_input": jnp.asarray(np.arange(N_TIME_RANK) % n_src)},
        ),
    )
    free = FAMILIES[:k]
    if basis is None:
        space = ParameterSpace(
            latents=[Latent(n, init=jnp.zeros(n_freq), linear=True) for n in free],
            bindings=[Bind(n, into=_into(n)) for n in free],
        )
    else:
        design = jnp.asarray(basis)
        space = ParameterSpace(
            latents=[Latent(n, init=jnp.zeros(design.shape[0]), linear=True) for n in free],
            bindings=[Bind(n, into=_into(n), fn=lambda v, _d=design: v @ _d) for n in free],
        )
    return identifiability(space, Pipeline(op, names=("noise_wave",)), state, names=free)


def _legendre(n_basis: int, n_freq: int) -> np.ndarray:
    """``(n_basis, n_freq)`` — a smooth spectral basis that is not ill-conditioned."""
    return np.polynomial.legendre.legvander(np.linspace(-1.0, 1.0, n_freq), n_basis - 1).T


class TestPerChannelRankRule:
    """``rank == min(n_src, k) * n_freq`` — measured, over k as well as n_src.

    The rule the module docstring states, and the one a switching cadence gets
    chosen from. Two things about it were previously wrong in the source:

    * ``k`` is the number of FREE temperature families, not the literal 3.
      ``t_rx`` is a leaf of this operator exactly like ``t_unc, t_cos, t_sin``;
      its coupling is 1 rather than absent. Fit it and four loads are needed to
      make the per-channel system square, not three.
    * the rule holds only while the temperatures are free per channel —
      see :class:`TestBasisRegimeBreaksTheRule`.

    Swept over ``n_freq`` as well, because a rule stated as a product of two
    factors is not established by fixing one of them.
    """

    @pytest.mark.parametrize("n_src", range(1, N_SRC_MAX + 1))
    @pytest.mark.parametrize("k", [3, 4])
    @pytest.mark.parametrize("n_freq", [3, 5, 7])
    def test_rank_is_min_n_src_k_times_n_freq(self, n_freq, k, n_src):
        with _float64_leaves():
            report = _rank_report(loads=range(n_src), k=k, n_freq=n_freq)
        assert report.n_par == k * n_freq
        assert report.rank == min(n_src, k) * n_freq
        assert report.nullity == (k - min(n_src, k)) * n_freq
        # Not a marginal call in either direction: the identified directions sit
        # decades above the tolerance, so the verdict is the model's and not the
        # arithmetic's.
        assert report.weakest_identified > 1e3 * DEFAULT_RANK_RTOL

    def test_a_fourth_free_family_costs_a_fourth_load(self):
        """The headline correction, as the two numbers that differ.

        Three loads make a three-family per-channel fit square and leave a
        four-family one short by exactly ``n_freq``. A team reading
        ``min(n_src, 3)`` off the old docstring would have switched between
        three calibrators and fitted ``T_rx`` anyway.
        """
        n_freq = 5
        with _float64_leaves():
            three = _rank_report(loads=range(3), k=3, n_freq=n_freq)
            four = _rank_report(loads=range(3), k=4, n_freq=n_freq)
        assert three.nullity == 0
        assert four.nullity == n_freq
        with _float64_leaves():
            fixed = _rank_report(loads=range(4), k=4, n_freq=n_freq)
        assert fixed.nullity == 0

    def test_the_blind_direction_of_a_three_load_four_family_fit_mixes_t_rx(self):
        """Naming the degeneracy, which is what makes the report actionable."""
        with _float64_leaves():
            report = _rank_report(loads=range(3), k=4, n_freq=5)
        share = report.participation(0)
        assert set(share) == set(FAMILIES)
        assert share["t_rx"] > 0.01
        assert sum(share.values()) == pytest.approx(1.0)

    def test_one_gamma_shared_across_the_cycle_collapses_the_rank(self):
        """Three switch positions with the same load are one switch position.

        The docstring's other claim: sharing a single ``Gamma`` gives up
        switching entirely, and the fit comes back finite, correctly shaped and
        wholly prior-driven.
        """
        n_freq = 5
        with _float64_leaves():
            shared = _rank_report(loads=(0, 0, 0), k=3, n_freq=n_freq)
            distinct = _rank_report(loads=(0, 1, 2), k=3, n_freq=n_freq)
        assert shared.rank == n_freq
        assert distinct.rank == 3 * n_freq


class TestBasisRegimeBreaksTheRule:
    """Per-channel counting does not survive a frequency basis, in BOTH directions.

    Once each temperature is ``coeffs @ basis`` the basis ties channels
    together, one equation per channel is no longer one equation per unknown,
    and ``min(n_src, k) * n_basis`` is simply not the rank. What survives is a
    bound, ``rank <= min(n_src * n_freq, k * n_basis)``, and it binds loosely
    enough at both ends that the only honest instruction is to measure.
    """

    def test_two_loads_identify_a_basis_model_the_per_channel_count_calls_deficient(self):
        """The rule UNDERSTATES: 12 of 12, where per-channel counting says 6."""
        n_freq = 7
        basis = _legendre(N_BASIS, n_freq)
        with _float64_leaves():
            report = _rank_report(loads=(0, 1), k=4, n_freq=n_freq, basis=basis)
        assert report.n_par == 4 * N_BASIS
        assert report.rank == 4 * N_BASIS
        assert report.nullity == 0
        assert min(2, 4) * N_BASIS == 6  # what the per-channel rule would have said
        assert report.weakest_identified > 10 * DEFAULT_RANK_RTOL

    def test_a_low_order_gamma_falls_short_of_the_bound(self):
        """The bound OVERSTATES, and ``n_src`` does not tell you when.

        Load 0's ``Gamma`` is linear in frequency, so its couplings are
        low-order too; a degree-<=2 basis function times a low-order coupling is
        another low-order function, and the ``k * n_basis`` products span only
        five dimensions rather than the seven the bound allows. Adding ``t_rx``
        as a fourth family buys nothing at all here — three more parameters,
        the same rank 5.
        """
        n_freq = 7
        basis = _legendre(N_BASIS, n_freq)
        with _float64_leaves():
            three = _rank_report(loads=(0,), k=3, n_freq=n_freq, basis=basis)
            four = _rank_report(loads=(0,), k=4, n_freq=n_freq, basis=basis)
        assert three.rank == 5
        assert three.rank < min(1 * n_freq, 3 * N_BASIS)
        assert four.n_par == three.n_par + N_BASIS
        assert four.rank == 5
        assert three.weakest_identified > 1e4 * DEFAULT_RANK_RTOL

    @pytest.mark.parametrize("n_src", range(1, N_SRC_MAX + 1))
    @pytest.mark.parametrize("k", [3, 4])
    def test_the_bound_holds_everywhere_even_where_the_counting_rule_does_not(self, k, n_src):
        n_freq = 7
        basis = _legendre(N_BASIS, n_freq)
        with _float64_leaves():
            report = _rank_report(loads=range(n_src), k=k, n_freq=n_freq, basis=basis)
        assert report.rank <= min(n_src * n_freq, k * N_BASIS)

    def test_scalar_temperatures_are_the_one_basis_corner_a_single_load_identifies(self):
        """``n_basis == 1``: all four families identified from ONE load.

        The docstring's long-standing parenthetical, restated as the corner of
        the basis regime that it is — and the reason a scalar demonstration
        proves nothing about switching.
        """
        n_freq = 7
        with _float64_leaves():
            report = _rank_report(loads=(1,), k=4, n_freq=n_freq, basis=_legendre(1, n_freq))
        assert report.n_par == 4
        assert report.nullity == 0

    def test_float32_gamma_leaves_make_the_basis_verdict_untrustworthy(self):
        """Why ``_float64_leaves`` exists, measured on the model it changes.

        ``identifiability()`` forces float64 for the Jacobian, and its own
        docstring is explicit that this cannot rescue a model whose
        INTERMEDIATES were already rounded. Single-precision ``Gamma`` leaves
        are exactly that: the same one-load basis model's weakest identified
        direction lands within a decade or two of the tolerance instead of five,
        and the rank it reports is roundoff's opinion rather than the model's.
        """
        n_freq = 7
        basis = _legendre(N_BASIS, n_freq)
        with _float64_leaves():
            good = _rank_report(loads=(0,), k=3, n_freq=n_freq, basis=basis)
            bad = _rank_report(loads=(0,), k=3, n_freq=n_freq, basis=basis,
                               dtype=jnp.float32)
        assert good.weakest_identified > 1e2 * DEFAULT_RANK_RTOL
        assert bad.weakest_identified < 1e2 * DEFAULT_RANK_RTOL
