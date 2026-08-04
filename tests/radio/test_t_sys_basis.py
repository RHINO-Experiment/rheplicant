"""BasisTemperatureOperator on ``t_sys_extra``, and what the basis buys.

This file carries the measurement the whole piece turns on. A known 5000 K CW
tone against a gain free per time sample, on the assembled canonical graph::

    free-per-cell T_ant,  tone ON  (5000 K)   n_par=42 rank=35 nullity=7
    free-per-cell T_ant,  tone OFF            n_par=42 rank=35 nullity=7
    (3,2)-basis T_ant,    tone ON  (5000 K)   n_par=13 rank=13 nullity=0
    (3,2)-basis T_ant,    tone OFF            n_par=13 rank=12 nullity=1

Against a free-per-cell antenna temperature the tone buys exactly nothing —
nullity is ``n_time`` either way, because the free cell at the tone's channel
absorbs ``g[t] * A`` sample by sample. It earns its keep only once ``T_ant`` is
frequency-smooth. That is why the basis has to reach the antenna temperature
and not only the noise waves.

The fixture is non-square in every dimension: 7 time samples against 5
frequency channels, a (3, 2) coefficient matrix, a 7-element gain against a
6-element coefficient block. A square one could not tell a transposed basis
pair from the intended one, could not tell the time axis from the frequency
axis in any refusal, and — for the sweep in
:class:`TestWhichAxisTheToneArguesFor` — could not tell "complete in time" from
"complete in frequency", which is the whole result.

Identifiability is a LOCAL property, so every report here is taken at a GENERIC
coefficient point. At a rank-one coefficient (all ones, say) the same (3, 2)
model reports nullity 3 rather than 1: the field is then an outer product and
a whole family of gain rescalings closes. That is not a different verdict about
the basis, it is a different point.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.basis import SeparableBasis, basis_matrix
from rheplicant.core.errors import ParameterSpaceError, StateValidationError
from rheplicant.core.graph import AmbiguousNodeError
from rheplicant.inference import (
    Bind,
    Block,
    Latent,
    ParameterSpace,
    SamplingPlan,
    check_linearity,
    identifiability,
)
from rheplicant.radio import (
    BasisTemperatureOperator,
    CWCalibrationOperator,
    GainOperator,
)
from rheplicant.radio.graph import assemble

N_TIME, N_FREQ = 7, 5
N_K, N_J = 3, 2
TONE_CHANNEL, TONE_KELVIN = 3, 5000.0
NOISE = 1.0

#: Generic on purpose: not rank one, not symmetric, and every entry a different
#: order of magnitude from its neighbours. A rank-one coefficient makes the
#: expanded field an outer product and opens degeneracies that belong to the
#: point, not to the basis.
COEFF0 = jnp.asarray([[2800.0, -120.0], [90.0, 17.0], [-30.0, 6.0]])
GAIN0 = 1.4 + 0.07 * jnp.arange(N_TIME, dtype=float)

#: Wrong by DIFFERENT fractions in the two latents, so a run that recovered one
#: and left the other could not pass by symmetry.
GAIN_GUESS = 0.88 * GAIN0
COEFF_GUESS = 0.80 * COEFF0

FREQS = jnp.linspace(60e6, 85e6, N_FREQ)
TONE_FREQ = float(FREQS[TONE_CHANNEL])

#: The critically-sampled unwindowed-FFT width — one channel spacing — which is
#: the narrowest ``sinc2`` this grid may carry and the one whose nulls fall
#: exactly on the neighbouring channels. Centred on a channel it therefore
#: injects the whole amplitude into that one channel, which is the sharpest
#: feature a real spectrometer can present and the hardest one for a smooth
#: basis to reabsorb. A wider line only makes the tone MORE like something a
#: frequency basis can absorb, so this is the favourable end of the argument
#: and the numbers below should be read as such.
CHANNEL_WIDTH = float(FREQS[1] - FREQS[0])

GAIN_PRIOR = dist.Normal(jnp.ones(N_TIME), 10.0)
COEFF_PRIOR = dist.Normal(jnp.zeros((N_K, N_J)), 1e4)


def make_state() -> State:
    return State(
        coords=Coordinates(time=jnp.arange(N_TIME, dtype=float), freq=FREQS),
        meta={"telescope": "RHINO", "obs_id": "tsys-000"},
    )


STATE = make_state()


@pytest.fixture
def state():
    return make_state()


@pytest.fixture
def basis() -> SeparableBasis:
    return SeparableBasis(
        time=basis_matrix("legendre", n=N_TIME, n_basis=N_K),
        freq=basis_matrix("legendre", n=N_FREQ, n_basis=N_J),
    )


def cell_basis() -> SeparableBasis:
    """The free-per-cell parameterization, written as a basis: two identities.

    Not a straw man and not a different operator — the SAME operator, with the
    complete basis on each axis. That is what makes the comparison below a
    statement about the parameterization and about nothing else.
    """
    return SeparableBasis(time=jnp.eye(N_TIME), freq=jnp.eye(N_FREQ))


def twin(basis: SeparableBasis, coeff, tone_kelvin: float = TONE_KELVIN):
    """``data[t, f] = gain[t] * (T_ant[t, f] + tone[f])`` on the real graph."""
    return assemble(
        BasisTemperatureOperator.from_basis(basis, coeff),
        CWCalibrationOperator(
            amplitude=tone_kelvin, tone_freq=TONE_FREQ, line_width=CHANNEL_WIDTH
        ),
        GainOperator(gain=GAIN0),
    )


def space_over(init, coeff_prior=None) -> ParameterSpace:
    """Gain and the coefficient block, both declared linear."""
    return ParameterSpace(
        latents=[
            Latent("gain", init=GAIN_GUESS, prior=GAIN_PRIOR, linear=True),
            Latent(
                "t_coeff",
                init=init,
                prior=coeff_prior
                if coeff_prior is not None
                else dist.Normal(jnp.zeros(jnp.shape(init)), 1e4),
                linear=True,
            ),
        ],
        bindings=[
            Bind("gain", into=lambda p: p["gain"].gain),
            Bind("t_coeff", into=lambda p: p["t_sys_extra"].coeff),
        ],
    )


# ------------------------------------------------------------- the operator --


class TestTheOperator:
    def test_it_writes_the_expansion_onto_the_full_grid(self, basis, state):
        operator = BasisTemperatureOperator.from_basis(basis, COEFF0)
        produced = operator(state).data
        assert produced.shape == (N_TIME, N_FREQ)
        assert jnp.allclose(produced, basis.expand(COEFF0))

    def test_it_sits_on_the_reserved_t_sys_extra_node(self, basis):
        assert BasisTemperatureOperator.graph_node == "t_sys_extra"
        assembly = assemble(BasisTemperatureOperator.from_basis(basis, COEFF0))
        assert "t_sys_extra" in assembly.lit

    def test_the_basis_property_round_trips_the_leaves(self, basis):
        operator = BasisTemperatureOperator.from_basis(basis, COEFF0)
        assert operator.basis.shape == (N_TIME, N_FREQ)
        assert operator.basis.coeff_shape == (N_K, N_J)
        assert jnp.allclose(operator.basis.expand(COEFF0), basis.expand(COEFF0))

    def test_the_leaves_are_the_parameterization_not_the_cells(self, basis):
        """The whole point of Part B: what is inferable here is ``(n_k, n_j)``
        coefficients, never ``(n_time, n_freq)`` cells. Six numbers, not
        thirty-five."""
        operator = BasisTemperatureOperator.from_basis(basis, COEFF0)
        assert operator.coeff.shape == (N_K, N_J)
        assert operator.coeff.size < N_TIME * N_FREQ

    def test_a_1D_coefficient_is_refused(self, basis):
        with pytest.raises(StateValidationError, match="2-D"):
            BasisTemperatureOperator(
                coeff=jnp.zeros(N_K), time_basis=basis.time, freq_basis=basis.freq
            )

    def test_a_coefficient_that_does_not_match_the_bases_is_refused(self, basis):
        with pytest.raises(StateValidationError) as caught:
            BasisTemperatureOperator(
                coeff=jnp.zeros((N_J, N_K)),
                time_basis=basis.time,
                freq_basis=basis.freq,
            )
        message = str(caught.value)
        assert "(3, 2)" in message and "(2, 3)" in message

    def test_the_operator_runs_the_same_design_checks_as_the_basis(self, basis):
        """Not a separate rule: the operator holds its design matrices as its
        own leaves, so it re-runs the check rather than trusting that a
        SeparableBasis was involved at all.

        Matched on ``design matrix`` — wording only ``_check_design`` uses. The
        obvious pattern ``freq`` is also matched by the coefficient-shape
        refusal two lines below it ("...by 2 frequency ones"), which passes with
        the design check deleted entirely; measured, not hypothetical.
        """
        with pytest.raises(StateValidationError, match="design matrix"):
            BasisTemperatureOperator(
                coeff=jnp.zeros((N_K, N_J)),
                time_basis=basis.time,
                freq_basis=jnp.ones(N_FREQ),
            )
        # ... and an over-complete design matrix, which without the check would
        # construct cleanly and be caught much later, if at all.
        with pytest.raises(StateValidationError, match="design matrix"):
            BasisTemperatureOperator(
                coeff=jnp.zeros((N_K, N_J)),
                time_basis=jnp.ones((2, N_K)),
                freq_basis=basis.freq,
            )

    def test_a_state_without_a_time_or_frequency_axis_is_refused(self, basis):
        operator = BasisTemperatureOperator.from_basis(basis, COEFF0)
        with pytest.raises(StateValidationError, match="coords"):
            operator(State(coords=Coordinates(time=jnp.arange(N_TIME, dtype=float))))

    def test_a_basis_built_for_another_grid_is_refused_per_axis(self, basis):
        """The transposition catch, and the reason this fixture is non-square:
        with 7 times against 5 channels a swapped pair of design matrices lands
        here instead of returning the transpose of the intended field."""
        swapped = BasisTemperatureOperator(
            coeff=jnp.zeros((N_J, N_K)), time_basis=basis.freq, freq_basis=basis.time
        )
        with pytest.raises(StateValidationError) as caught:
            swapped(make_state())
        assert "time" in str(caught.value)

        wrong_freq = BasisTemperatureOperator(
            coeff=jnp.zeros((N_K, N_K)),
            time_basis=basis.time,
            freq_basis=basis_matrix("legendre", n=N_FREQ + 2, n_basis=N_K),
        )
        with pytest.raises(StateValidationError) as caught:
            wrong_freq(make_state())
        assert "freq" in str(caught.value)


# ------------------------------------------------------- a second instance --


class TestASecondContribution:
    """``t_sys_extra`` is ``many=True`` and this operator will not be the only
    thing on it forever.

    Before the stable-id work landed, a second contribution silently made
    ``p["t_sys_extra"]`` resolve to a ``SumOperator``, invalidating any
    ParameterSpace written for the first. It is now a named refusal, and the
    per-instance ids are what a binding should use.
    """

    def test_two_contributions_sum_and_each_keeps_its_own_id(self, basis, state):
        second = SeparableBasis(
            time=basis_matrix("legendre", n=N_TIME, n_basis=1),
            freq=basis_matrix("legendre", n=N_FREQ, n_basis=1),
        )
        first_op = BasisTemperatureOperator.from_basis(basis, COEFF0)
        second_op = BasisTemperatureOperator.from_basis(second, jnp.asarray([[17.0]]))

        assembly = assemble(first_op, second_op, GainOperator(gain=GAIN0))
        assert dict(assembly.instances)["t_sys_extra"] == (
            "t_sys_extra_1",
            "t_sys_extra_2",
        )
        assert jnp.allclose(
            assembly(state).data,
            GAIN0[:, None] * (basis.expand(COEFF0) + 17.0),
        )

    def test_the_bare_node_id_is_refused_once_there_are_two(self, basis):
        second = SeparableBasis(
            time=basis_matrix("legendre", n=N_TIME, n_basis=1),
            freq=basis_matrix("legendre", n=N_FREQ, n_basis=1),
        )
        assembly = assemble(
            BasisTemperatureOperator.from_basis(basis, COEFF0),
            BasisTemperatureOperator.from_basis(second, jnp.asarray([[17.0]])),
            GainOperator(gain=GAIN0),
        )
        with pytest.raises(AmbiguousNodeError) as caught:
            assembly["t_sys_extra"]
        assert "t_sys_extra_1" in str(caught.value)
        assert isinstance(assembly["t_sys_extra_1"], BasisTemperatureOperator)

    def test_a_space_binds_to_the_per_instance_id_and_reaches_only_that_one(
        self, basis, state
    ):
        """The id is what makes a sibling contribution survivable: a binding
        written for the first instance keeps reaching the first instance."""
        second = SeparableBasis(
            time=basis_matrix("legendre", n=N_TIME, n_basis=1),
            freq=basis_matrix("legendre", n=N_FREQ, n_basis=1),
        )
        assembly = assemble(
            BasisTemperatureOperator.from_basis(basis, COEFF0),
            BasisTemperatureOperator.from_basis(second, jnp.asarray([[17.0]])),
        )
        space = ParameterSpace(
            latents=[Latent("t_coeff", init=COEFF0, linear=True)],
            bindings=[Bind("t_coeff", into=lambda p: p["t_sys_extra_1"].coeff)],
        )
        forward, _ = space.forward_fn(assembly, state)
        moved = forward({"t_coeff": COEFF0 * 2.0})
        assert jnp.allclose(moved, basis.expand(COEFF0 * 2.0) + 17.0)


# -------------------------------------------------------- the measured table --


def _report(basis: SeparableBasis, coeff, tone_kelvin: float):
    return identifiability(space_over(coeff), twin(basis, coeff, tone_kelvin), STATE)


class TestWhatTheToneBuys:
    """The table in the module docstring, measured through this operator."""

    def test_the_free_per_cell_model_is_blind_in_n_time_directions_either_way(self):
        cells = cell_basis()
        truth = cells.expand(SeparableBasis(
            time=basis_matrix("legendre", n=N_TIME, n_basis=N_K),
            freq=basis_matrix("legendre", n=N_FREQ, n_basis=N_J),
        ).expand(COEFF0))
        on = _report(cells, truth, TONE_KELVIN)
        off = _report(cells, truth, 0.0)

        assert on.n_par == off.n_par == N_TIME * N_FREQ + N_TIME
        assert on.rank == off.rank == N_TIME * N_FREQ
        assert on.nullity == off.nullity == N_TIME
        # ... and the tone changes nothing about how well the rest is seen
        assert abs(on.weakest_identified - off.weakest_identified) < 1e-6

    def test_the_basis_model_is_fully_identified_WITH_the_tone(self, basis):
        report = _report(basis, COEFF0, TONE_KELVIN)
        assert report.n_par == N_K * N_J + N_TIME == 13
        assert report.rank == 13
        assert report.nullity == 0
        assert report.weakest_identified > 1e-2, report.weakest_identified

    def test_the_basis_model_keeps_ONE_blind_direction_without_the_tone(self, basis):
        """The overall scale: ``g -> c g``, ``T -> T/c``. One direction, not
        ``n_time`` — that is what the smooth basis removed — and the tone is
        what removes the last one."""
        report = _report(basis, COEFF0, 0.0)
        assert report.n_par == 13
        assert report.rank == 12
        assert report.nullity == 1
        share = report.participation(0)
        assert share["gain"] > 0.1 and share["t_coeff"] > 0.1, share

    def test_the_degenerate_direction_is_named_as_both_latents(self):
        cells = cell_basis()
        report = _report(cells, jnp.full((N_TIME, N_FREQ), 2500.0), TONE_KELVIN)
        share = report.participation(0)
        assert set(share) == {"gain", "t_coeff"}
        assert abs(share["gain"] - 0.5) < 0.01, share


class TestWhichAxisTheToneArguesFor:
    """Which axis has to be restricted, measured rather than asserted.

    A basis complete in FREQUENCY makes the tone worth nothing — the tone's
    delta is then inside the span and is reabsorbed. A basis complete in TIME
    is fine as long as frequency is restricted. So "frequency-smooth" is the
    condition, and ``n_j < n_freq`` is what it means; a square fixture could
    not tell the two apart at all.
    """

    @staticmethod
    def _nullity(n_k: int, n_j: int, tone_kelvin: float) -> int:
        basis = SeparableBasis(
            time=basis_matrix("legendre", n=N_TIME, n_basis=n_k),
            freq=basis_matrix("legendre", n=N_FREQ, n_basis=n_j),
        )
        coeff = 100.0 + 500.0 * jax.random.normal(jax.random.key(3), (n_k, n_j))
        return _report(basis, coeff, tone_kelvin).nullity

    def test_a_complete_TIME_basis_is_still_rescued_by_the_tone(self):
        assert self._nullity(N_TIME, N_J, TONE_KELVIN) == 0
        assert self._nullity(N_TIME, N_J, 0.0) == N_TIME

    def test_a_complete_FREQUENCY_basis_makes_the_tone_worth_nothing(self):
        assert self._nullity(N_K, N_FREQ, TONE_KELVIN) == 1
        assert self._nullity(N_K, N_FREQ, 0.0) == 1

    def test_complete_in_both_is_the_free_per_cell_row_of_the_table(self):
        assert self._nullity(N_TIME, N_FREQ, TONE_KELVIN) == N_TIME
        assert self._nullity(N_TIME, N_FREQ, 0.0) == N_TIME


# ----------------------------------------------------- blocks and bilinearity --


class TestTheGainAndTheTemperatureCannotShareABlock:
    def test_the_JOINT_linearity_claim_is_refused_as_bilinear(self, basis, state):
        """Verified rather than assumed. ``check_linearity(names=...)`` probes
        the joint map, and ``g * T`` is bilinear, not affine — so these two
        cannot be one conjugate block however linear each is on its own."""
        space, pipeline = space_over(COEFF_GUESS, COEFF_PRIOR), twin(basis, COEFF0)
        with pytest.raises(ParameterSpaceError) as caught:
            check_linearity(space, pipeline, state, names=("gain", "t_coeff"))
        message = str(caught.value)
        assert "JOINTLY" in message
        assert "Split them into separate blocks" in message

    def test_each_one_alone_IS_affine_which_is_why_the_joint_check_is_needed(
        self, basis, state
    ):
        space, pipeline = space_over(COEFF_GUESS, COEFF_PRIOR), twin(basis, COEFF0)
        assert max(check_linearity(space, pipeline, state, "gain").values()) < 1e-4
        assert max(check_linearity(space, pipeline, state, "t_coeff").values()) < 1e-4

    def test_two_blocks_both_derive_the_conjugate_engine(self, basis):
        plan = SamplingPlan(
            space_over(COEFF_GUESS, COEFF_PRIOR), Block("gain"), Block("t_coeff")
        )
        assert plan.engines == {("gain",): "conjugate", ("t_coeff",): "conjugate"}


# ------------------------------------------------------------ end to end --


class TestBothExitsOfOnePlan:
    def test_the_free_per_cell_model_is_REFUSED_and_the_direction_is_NAMED(self, state):
        cells = cell_basis()
        truth = jnp.full((N_TIME, N_FREQ), 2500.0)
        space, pipeline = space_over(truth), twin(cells, truth)
        forward, _ = space.forward_fn(pipeline, STATE)
        observed = forward({"gain": GAIN0, "t_coeff": truth})
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))

        with pytest.raises(ParameterSpaceError) as caught:
            plan.estimate(pipeline, state, observed, noise=NOISE)
        message = str(caught.value)
        assert f"nullity {N_TIME} of {N_TIME * N_FREQ + N_TIME} parameters" in message
        assert "direction 0:" in message
        assert "gain" in message and "t_coeff" in message

        with pytest.raises(ParameterSpaceError, match=f"nullity {N_TIME}"):
            plan.sample(
                pipeline, state, observed, noise=NOISE,
                key=jax.random.key(0), n_sweeps=8,
            )

    def test_the_basis_model_runs_and_both_exits_agree_with_the_truth(
        self, basis, state
    ):
        space, pipeline = space_over(COEFF_GUESS, COEFF_PRIOR), twin(basis, COEFF0)
        forward, _ = space.forward_fn(pipeline, STATE)
        observed = forward({"gain": GAIN0, "t_coeff": COEFF0})
        plan = SamplingPlan(space, Block("gain"), Block("t_coeff"))

        est = plan.estimate(
            pipeline, state, observed, noise=NOISE, max_iter=300, solve_guard=None
        )
        draws = plan.sample(
            pipeline, state, observed, noise=NOISE, key=jax.random.key(0),
            n_sweeps=300, warmup=150, solve_guard=None,
        )

        assert est.diagnostics.converged is True
        assert draws.diagnostics.converged is True
        assert draws.diagnostics.rhat < 1.05, draws.diagnostics.rhat

        assert float(jnp.max(jnp.abs(est.values["gain"] - GAIN0))) < 1e-3
        recovered = basis.expand(est.values["t_coeff"])
        truth_field = basis.expand(COEFF0)
        assert float(jnp.sqrt(jnp.mean((recovered - truth_field) ** 2))) < 1.0

        truth = {"gain": GAIN0, "t_coeff": COEFF0}
        for name in ("gain", "t_coeff"):
            gap = jnp.abs(draws.mean[name] - truth[name])
            assert jnp.all(gap < 5.0 * draws.std[name] + 1e-6), (name, gap)
            assert jnp.all(
                jnp.abs(draws.mean[name] - est.values[name])
                < 5.0 * draws.std[name] + 1e-6
            ), name

        # the posterior has real width — a draw that came back as the mean
        # would satisfy every assertion above and be wrong about everything
        assert float(jnp.min(draws.std["gain"])) > 0.0
        assert float(jnp.min(draws.std["t_coeff"])) > 0.0


# ------------------------------------------------- the noise-wave leaf contract --


class TestTheShapeTheNoiseWaveLeavesAccept:
    def test_an_expansion_is_a_legal_noise_wave_temperature(self, basis):
        """``NoiseWaveOperator``'s ``__check_init__`` accepts ``()``,
        ``(n_freq,)``, ``(n_time, 1)`` and ``(n_time, n_freq)``, and cannot
        distinguish a bare ``(n,)`` per-time vector from a spectrum when the two
        axes are the same length. An expansion is always the full 2-D grid, so
        the shape this route produces is the one shape that is never ambiguous.
        """
        noise_wave = pytest.importorskip("rheplicant.radio.instrument.noise_wave")
        expanded = basis.expand(COEFF0)
        assert expanded.shape == (N_TIME, N_FREQ)
        operator = noise_wave.NoiseWaveOperator(
            t_unc=expanded,
            t_cos=jnp.zeros((N_TIME, N_FREQ)),
            t_sin=jnp.zeros((N_TIME, N_FREQ)),
            t_rx=jnp.asarray(250.0),
            gamma_src_re=jnp.zeros((1, N_FREQ)),
            gamma_src_im=jnp.zeros((1, N_FREQ)),
            gamma_rec_re=jnp.zeros(N_FREQ),
            gamma_rec_im=jnp.zeros(N_FREQ),
        )
        assert operator.t_unc.shape == (N_TIME, N_FREQ)
