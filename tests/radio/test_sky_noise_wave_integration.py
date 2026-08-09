"""The beam-convolved sky as the noise-wave model's ``T_src``.

limTOD produces an antenna temperature; the noise-wave data model wants
a source temperature. This suite is the claim that they are the same quantity,
and that the graph delivers one to the other without rescaling it, leaking it
across the switch, or folding it into the wrong term.

Three hazards get their own tests because none of them is a shape error, so
none can be caught by a structural guard:

* ``normalize_beam=False`` (the default, matching numpy limTOD) returns
  ``int(B T)``, not ``int(B T)/int(B)``. Fed to the receiver it is a temperature
  scaled by the beam solid angle -- a few percent off even for a beam the user
  normalized by hand, because the band-limit truncates the denominator too.
* The selector's branch order and ``NoiseWaveOperator``'s ``gamma_src`` rows
  are two independent orderings that MUST agree; nothing checks that they do.
* A switch value with no matching branch selects nothing (``T_src = 0``) while
  the coupling lookup clamps to the last row, so the sample survives with a
  receiver contribution and no source.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("limtod_jax", reason="limTOD[jax] not installed")
pytest.importorskip("rhino_cal_jax", reason="rhino-cal-jax not installed")

import equinox as eqx  # noqa: E402
import rhino_cal_jax as rcj  # noqa: E402

from rheplicant import (  # noqa: E402
    Coordinates,
    Pipeline,
    SelectOperator,
    State,
    SumOperator,
)
from rheplicant.radio import (  # noqa: E402
    AntennaLossOperator,
    AtmosphericEmissionOperator,
    CalLoadOperator,
    GroundPickupOperator,
    MapSky,
    NoiseWaveOperator,
    SkySourceOperator,
    assemble,
)
from rheplicant.radio.sky import DriftScanProjector  # noqa: E402

NSIDE, LMAX, N_FREQ, N_TIME = 4, 11, 2, 8
N_PIX = 12 * NSIDE**2
LAT_DEG, AZ_DEG, EL_DEG = 53.2, 0.0, 90.0

X64 = jax.config.read("jax_enable_x64")
RTOL = 1e-12 if X64 else 3e-5  # the suite runs f32; x64 is process-global


def beam_maps(sigma: float = 0.35) -> jax.Array:
    """A zenith-centred Gaussian, normalized to unit HEALPix pixel sum."""
    theta = jnp.arccos(1.0 - 2.0 * (jnp.arange(N_PIX) + 0.5) / N_PIX)
    raw = jnp.stack([jnp.exp(-0.5 * (theta / (sigma + 0.02 * f)) ** 2)
                     for f in range(N_FREQ)])
    return raw / raw.sum(axis=1, keepdims=True)


def projector(normalize_beam: bool = True, sigma: float = 0.35) -> DriftScanProjector:
    return DriftScanProjector.from_beam_maps(
        beam_maps(sigma), lat_deg=LAT_DEG, az_deg=AZ_DEG, el_deg=EL_DEG,
        lmax=LMAX, normalize_beam=normalize_beam,
    )


@pytest.fixture
def sky_maps():
    return 200.0 + 40.0 * jax.random.normal(jax.random.key(0), (N_FREQ, N_PIX))


@pytest.fixture
def freq():
    return jnp.linspace(60e6, 85e6, N_FREQ)


def make_coords(freq, switch):
    return Coordinates(
        time=jnp.arange(float(N_TIME)),
        freq=freq,
        extra={
            "lst_deg": DriftScanProjector.uniform_lst_grid(N_TIME),
            "receiver_input": jnp.asarray(switch),
        },
    )


def make_gammas(freq, *, matched: bool = False):
    """``(gamma_src (2, n_freq), gamma_rec (n_freq,))`` -- antenna then load."""
    if matched:
        return jnp.zeros((2, N_FREQ), dtype=complex), jnp.zeros(N_FREQ, dtype=complex)
    antenna = rcj.cable_gamma(
        rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92
    )
    load = rcj.termination_gamma("resistive", N_FREQ, impedance=10.0)
    receiver = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)
    return jnp.stack([antenna, load]), receiver


def make_noise_wave(freq, *, matched=False, zero_temps=False, gamma_src=None):
    g_src, g_rec = make_gammas(freq, matched=matched)
    if gamma_src is not None:
        g_src = gamma_src
    zero = jnp.zeros(())
    return NoiseWaveOperator(
        t_unc=zero if zero_temps else jnp.array(250.0),
        t_cos=zero if zero_temps else jnp.array(30.0),
        t_sin=zero if zero_temps else jnp.array(-40.0),
        t_rx=zero if zero_temps else jnp.array(290.0),
        gamma_src_re=g_src.real, gamma_src_im=g_src.imag,
        gamma_rec_re=g_rec.real, gamma_rec_im=g_rec.imag,
    )


def make_twin(sky_maps, freq, *, t_load=300.0, normalize_beam=True, **nw_kwargs):
    """The assembled two-branch twin: sky | load -> switch -> noise wave."""
    return assemble(
        SkySourceOperator(sky_model=MapSky(maps=sky_maps, freq=freq),
                          projector=projector(normalize_beam)),
        CalLoadOperator(t_load=jnp.array(t_load)),
        make_noise_wave(freq, **nw_kwargs),
    )


ALTERNATING = jnp.arange(N_TIME) % 2  # 0 = antenna, 1 = load


class TestTheSkyIsTheSourceTemperature:
    """What ``T_src`` means when the source is an antenna looking at the sky."""

    def test_a_matched_antenna_passes_the_beam_convolved_sky_through(
        self, sky_maps, freq
    ):
        """Gamma = 0 everywhere and no receiver terms: the model collapses to
        ``T_sys = T_ant``, so the noise-wave stage must return the projector's
        own output unchanged. This is the whole claim in one assertion -- the
        sky enters as T_src and nothing rescales it on the way."""
        coords = make_coords(freq, jnp.zeros(N_TIME, dtype=int))  # antenna always
        twin = make_twin(sky_maps, freq, matched=True, zero_temps=True)
        out = twin(State(coords=coords)).data
        t_ant = projector().forward(sky_maps, coords)
        assert jnp.allclose(out, t_ant, rtol=RTOL, atol=0.0)

    def test_a_mismatched_antenna_attenuates_the_sky_by_exactly_c_src(
        self, sky_maps, freq
    ):
        """With the noise-wave temperatures zeroed, the only thing left acting
        on the sky is the mismatch loss ``c_src = (1-|G_s|^2)|F|^2``. It must
        appear as an exact multiplicative factor, and it must be < 1: a real
        antenna delivers less than it collects."""
        coords = make_coords(freq, jnp.zeros(N_TIME, dtype=int))
        twin = make_twin(sky_maps, freq, zero_temps=True)
        out = twin(State(coords=coords)).data

        t_ant = projector().forward(sky_maps, coords)
        g_src, g_rec = make_gammas(freq)
        c_src = rcj.couplings(g_src, g_rec).c_src[0]  # antenna row
        assert jnp.all(c_src < 1.0)
        assert jnp.max(c_src) < 0.9, "pick a Gamma with a visible mismatch loss"
        assert jnp.allclose(out, c_src[None, :] * t_ant, rtol=RTOL, atol=0.0)

    def test_the_receiver_terms_do_not_scale_with_the_sky(self, sky_maps, freq):
        """T_unc/T_cos/T_sin/T_rx are receiver properties. Changing the sky may
        move the output only through ``c_src * dT_ant`` -- if any receiver term
        picked up a sky factor (or the sky landed in the wrong coupling column),
        this difference would not close."""
        coords = make_coords(freq, ALTERNATING)
        out_a = make_twin(sky_maps, freq)(State(coords=coords)).data
        out_b = make_twin(2.0 * sky_maps, freq)(State(coords=coords)).data

        t_ant = projector().forward(sky_maps, coords)
        g_src, g_rec = make_gammas(freq)
        c_src = rcj.couplings(g_src, g_rec).c_src[0]
        expected = jnp.where(
            (ALTERNATING == 0)[:, None], c_src[None, :] * t_ant, 0.0
        )
        assert jnp.allclose(out_b - out_a, expected, rtol=RTOL, atol=1e-4)

    def test_the_assembled_twin_matches_a_hand_built_reference(self, sky_maps, freq):
        """End to end against the system temperature spelled out by hand: project, switch the
        source temperature, gather the couplings, evaluate. Nothing in the
        assembled path may deviate from that."""
        coords = make_coords(freq, ALTERNATING)
        out = eqx.filter_jit(make_twin(sky_maps, freq))(State(coords=coords)).data

        t_ant = projector().forward(sky_maps, coords)
        t_src = jnp.where((ALTERNATING == 0)[:, None], t_ant, 300.0)
        g_src, g_rec = make_gammas(freq)
        gathered = rcj.Couplings.from_stacked(
            rcj.couplings(g_src, g_rec).stacked[ALTERNATING]
        )
        reference = rcj.system_temperature(
            gathered, t_src=t_src, t_unc=jnp.array(250.0), t_cos=jnp.array(30.0),
            t_sin=jnp.array(-40.0), t_rx=jnp.array(290.0),
        )
        assert jnp.allclose(out, reference, rtol=RTOL, atol=1e-4)


class TestTheSwitch:
    """The selector must REPLACE the antenna, and take Gamma with it."""

    def test_load_samples_never_see_the_sky(self, sky_maps, freq):
        """A different sky must leave every load sample bit-identical. A
        SumOperator in the selector's place would pass every other test in this
        file and fail this one."""
        coords = make_coords(freq, ALTERNATING)
        out_a = make_twin(sky_maps, freq)(State(coords=coords)).data
        out_b = make_twin(sky_maps * 3.0 + 17.0, freq)(State(coords=coords)).data
        load = ALTERNATING == 1
        assert jnp.array_equal(out_a[load], out_b[load])
        assert not jnp.allclose(out_a[~load], out_b[~load])

    def test_antenna_samples_never_see_the_load(self, sky_maps, freq):
        coords = make_coords(freq, ALTERNATING)
        out_a = make_twin(sky_maps, freq, t_load=300.0)(State(coords=coords)).data
        out_b = make_twin(sky_maps, freq, t_load=900.0)(State(coords=coords)).data
        antenna = ALTERNATING == 0
        assert jnp.array_equal(out_a[antenna], out_b[antenna])
        assert not jnp.allclose(out_a[~antenna], out_b[~antenna])

    def test_the_switch_picks_the_source_temperature_and_its_gamma_together(
        self, sky_maps, freq
    ):
        """HAZARD, deliberately pinned. The selector orders its branches by the
        graph's in-edge declaration (antenna, then load); ``gamma_src`` orders
        its rows by however the caller stacked them. Nothing checks that the two
        agree -- both are ``(2, n_freq)``, so a transposition is shape-legal.

        The test measures what that costs. Every sample stays finite and
        correctly shaped, and the answer moves by tens of kelvin -- ~46 K peak,
        ~28 K mean on a ~545 K signal (8% relative) for these two loads. If a
        future guard makes the transposition raise, this test should be
        rewritten to assert the raise, not relaxed."""
        coords = make_coords(freq, ALTERNATING)
        g_src, _ = make_gammas(freq)
        right = make_twin(sky_maps, freq)(State(coords=coords)).data
        wrong = make_twin(
            sky_maps, freq, gamma_src=g_src[::-1]
        )(State(coords=coords)).data

        assert jnp.all(jnp.isfinite(wrong))
        assert wrong.shape == right.shape
        assert float(jnp.max(jnp.abs(wrong - right))) > 10.0, (
            "a swapped Gamma ordering should be numerically obvious; if it is "
            "not, the loads are too similar for this test to mean anything"
        )

    def test_an_out_of_range_switch_value_is_refused_eagerly(self, sky_maps, freq):
        """``SwitchCycle`` range-checks the switch against its label count, so
        a switch that names a source the operator does not carry is caught --
        eagerly."""
        switch = jnp.where(jnp.arange(N_TIME) < 4, 0, 2)  # only 2 branches exist
        coords = make_coords(freq, switch)
        with pytest.raises(Exception, match="out of range"):
            make_twin(sky_maps, freq)(State(coords=coords))

    def test_an_out_of_range_switch_value_is_nan_under_jit_not_clamped(
        self, sky_maps, freq
    ):
        """The eager check needs concrete values and is skipped under tracing,
        which is the production path. JAX's gather semantics would then CLAMP
        the coupling lookup to the last row while ``SelectOperator`` selected no
        branch at all -- a sample with a receiver contribution, no source, and
        another load's Gamma, finite and correctly shaped throughout.

        ``SwitchCycle.gather`` fills out-of-range samples with NaN instead, so
        the disagreement announces itself. In-range samples are untouched."""
        switch = jnp.where(jnp.arange(N_TIME) < 4, 0, 2)
        coords = make_coords(freq, switch)
        out = eqx.filter_jit(make_twin(sky_maps, freq))(State(coords=coords)).data

        assert jnp.all(jnp.isnan(out[switch == 2]))
        assert jnp.all(jnp.isfinite(out[switch == 0]))
        reference = eqx.filter_jit(make_twin(sky_maps, freq))(
            State(coords=make_coords(freq, jnp.zeros(N_TIME, dtype=int)))
        ).data
        assert jnp.allclose(out[switch == 0], reference[switch == 0],
                            rtol=RTOL, atol=1e-4)


class TestBeamNormalization:
    """``normalize_beam`` decides whether T_src is a temperature at all."""

    @pytest.fixture
    def constant_sky(self):
        """A uniform 200 K sky: any honest beam average must return 200 K."""
        return jnp.full((N_FREQ, N_PIX), 200.0)

    def test_a_normalized_beam_returns_the_sky_temperature_exactly(
        self, constant_sky, freq
    ):
        coords = make_coords(freq, jnp.zeros(N_TIME, dtype=int))
        t_ant = projector(normalize_beam=True).forward(constant_sky, coords)
        assert jnp.allclose(t_ant, 200.0, rtol=1e-5 if not X64 else 1e-12)

    def test_an_unnormalized_beam_biases_t_src_by_percent(self, constant_sky, freq):
        """The default ``normalize_beam=False`` returns ``int(B T)``. Even with
        a beam whose PIXEL SUM is 1 the answer is biased, because the band-limit
        truncates ``int(B)`` away from 1 as well -- and the bias grows as the
        beam narrows towards the pixel scale. Percent-level, finite, correctly
        shaped, and wrong."""
        coords = make_coords(freq, jnp.zeros(N_TIME, dtype=int))
        biased = projector(normalize_beam=False).forward(constant_sky, coords)
        bias = float(jnp.mean(biased) / 200.0 - 1.0)
        assert abs(bias) > 0.01, (
            "expected a percent-level normalization bias at this nside/lmax; "
            f"got {bias:.2e} -- the demonstration has stopped demonstrating"
        )

    def test_the_normalization_bias_propagates_into_the_noise_wave_output(
        self, constant_sky, freq
    ):
        """The bias is not absorbed downstream: it reaches the receiver output
        multiplied by ``c_src``, which is exactly how a mis-normalized beam
        would corrupt a noise-wave calibration."""
        coords = make_coords(freq, jnp.zeros(N_TIME, dtype=int))
        good = make_twin(constant_sky, freq, zero_temps=True)(State(coords=coords)).data
        bad = make_twin(constant_sky, freq, zero_temps=True,
                        normalize_beam=False)(State(coords=coords)).data

        g_src, g_rec = make_gammas(freq)
        c_src = rcj.couplings(g_src, g_rec).c_src[0]
        t_ant_bias = (
            projector(normalize_beam=False).forward(constant_sky, coords)
            - projector(normalize_beam=True).forward(constant_sky, coords)
        )
        assert jnp.allclose(bad - good, c_src[None, :] * t_ant_bias,
                            rtol=1e-4, atol=1e-4)


class TestAssembly:
    def test_assemble_lights_the_sky_switch_and_receiver(self, sky_maps, freq):
        twin = make_twin(sky_maps, freq)
        assert twin.lit == ("observed_astro_sky", "cal_loads", "noise_wave")
        assert "t_ant_sum" in twin.skipped

    def test_the_selector_branch_order_is_antenna_then_load(self, sky_maps, freq):
        """The order the switch values index, read off the assembled twin
        rather than assumed -- it is the ordering ``gamma_src`` must match."""
        selector = make_twin(sky_maps, freq)["receiver_input"]
        assert isinstance(selector, SelectOperator)
        assert selector.switch_key == "receiver_input"
        assert selector.names == ("observed_astro_sky", "cal_loads")
        assert isinstance(selector.branches[0], SkySourceOperator)
        assert isinstance(selector.branches[1], CalLoadOperator)

    def test_each_extra_cal_load_becomes_its_own_switch_position(
        self, sky_maps, freq
    ):
        """``cal_loads`` is ``many=True`` and feeds only the selector, so its
        instances fan out into sibling SELECTOR branches rather than being
        summed. Three sources -- the minimum for an identifiable per-channel
        noise-wave fit -- therefore come straight out of ``assemble()``, in the
        order they were provided."""
        g_src, g_rec = make_gammas(freq)
        three = jnp.stack([g_src[0], g_src[1], g_src[1] * 0.5])
        twin = assemble(
            SkySourceOperator(sky_model=MapSky(maps=sky_maps, freq=freq), projector=projector()),
            CalLoadOperator(t_load=jnp.array(300.0)),
            CalLoadOperator(t_load=jnp.array(400.0)),
            make_noise_wave(freq, gamma_src=three),
        )
        selector = twin["receiver_input"]
        assert selector.names == ("observed_astro_sky", "cal_loads_1", "cal_loads_2")
        assert [float(b.t_load) for b in selector.branches[1:]] == [300.0, 400.0]
        # and the bare node id now addresses neither of them, by name
        assert dict(twin.instances)["cal_loads"] == ("cal_loads_1", "cal_loads_2")

    def test_the_loads_are_switched_between_not_added_together(
        self, sky_maps, freq
    ):
        """The distinction ``many`` now makes: at a junction the instances would
        sum to 700 K, at a selector each takes its own samples. Summing them
        would be finite, correctly shaped, and a different instrument."""
        g_src, g_rec = make_gammas(freq)
        three = jnp.stack([g_src[0], g_src[1], g_src[1]])
        switch = jnp.arange(N_TIME) % 3
        twin = assemble(
            SkySourceOperator(sky_model=MapSky(maps=sky_maps, freq=freq), projector=projector()),
            CalLoadOperator(t_load=jnp.array(300.0)),
            CalLoadOperator(t_load=jnp.array(400.0)),
            make_noise_wave(freq, gamma_src=three, zero_temps=True),
        )
        out = twin(State(coords=make_coords(freq, switch))).data
        c_src = rcj.couplings(three, g_rec).c_src
        assert jnp.allclose(out[switch == 1], 300.0 * c_src[1][None, :],
                            rtol=RTOL, atol=1e-3)
        assert jnp.allclose(out[switch == 2], 400.0 * c_src[2][None, :],
                            rtol=RTOL, atol=1e-3)


class TestTheAntennaChain:
    """The sky is not the only thing at the antenna terminals."""

    def test_the_ohmic_loss_reaches_the_receiver_as_a_changed_t_src(
        self, sky_maps, freq
    ):
        """eta acts on the sky BEFORE the receiver stage, so the receiver sees
        ``eta T_sky + (1 - eta) T_phys`` in T_src's place -- attenuated AND
        offset, which the mismatch loss c_s alone can never produce."""
        coords = make_coords(freq, jnp.zeros(N_TIME, dtype=int))
        eta, t_phys = 0.9, 293.0
        lossless = make_twin(sky_maps, freq, zero_temps=True)
        lossy = assemble(
            SkySourceOperator(sky_model=MapSky(maps=sky_maps, freq=freq), projector=projector()),
            AntennaLossOperator(efficiency=jnp.array(eta),
                                t_physical=jnp.array(t_phys)),
            CalLoadOperator(t_load=jnp.array(300.0)),
            make_noise_wave(freq, zero_temps=True),
        )
        out_lossless = lossless(State(coords=coords)).data
        out_lossy = lossy(State(coords=coords)).data

        g_src, g_rec = make_gammas(freq)
        c_src = rcj.couplings(g_src, g_rec).c_src[0]
        expected = eta * out_lossless + (1.0 - eta) * t_phys * c_src[None, :]
        assert jnp.allclose(out_lossy, expected, rtol=RTOL, atol=1e-3)

    def test_a_hand_built_branch_must_reproduce_what_assemble_builds(
        self, sky_maps, freq
    ):
        """Hand-wiring is no longer forced on anyone -- ``cal_loads`` is
        ``many=True`` and ``assemble()`` builds the multi-load selector -- but
        anyone who does it still meets this: a Pipeline of SOURCE operators
        REPLACES the data at each stage rather than summing it, so
        ``Pipeline(sky, ground, atmosphere)`` silently keeps only the
        atmosphere. Finite, correctly shaped, and missing the sky entirely. The
        graph's t_ant_sum junction is what makes them add; this is the check
        that a hand-wired branch actually agrees with it."""
        antenna_only = jnp.zeros(N_TIME, dtype=int)
        coords = make_coords(freq, antenna_only)
        sources = (
            SkySourceOperator(sky_model=MapSky(maps=sky_maps, freq=freq), projector=projector()),
            GroundPickupOperator(coupling=jnp.array(0.02),
                                 t_ground=jnp.array(290.0)),
            AtmosphericEmissionOperator(t_atm=jnp.array(3.0)),
        )
        loss = AntennaLossOperator(efficiency=jnp.array(0.97),
                                   t_physical=jnp.array(293.0))
        assembled = assemble(
            *sources, loss, CalLoadOperator(t_load=jnp.array(300.0)),
            make_noise_wave(freq),
        )(State(coords=coords)).data

        by_hand = Pipeline(
            SelectOperator(
                Pipeline(SumOperator(*sources, names=("sky", "ground", "atm")),
                         loss, names=("t_ant_sum", "antenna_loss")),
                CalLoadOperator(t_load=jnp.array(300.0)),
                names=("antenna", "load"), switch_key="receiver_input",
            ),
            make_noise_wave(freq),
            names=("receiver_input", "noise_wave"),
        )(State(coords=coords)).data
        assert jnp.allclose(by_hand, assembled, rtol=RTOL, atol=1e-4)

        # And name what the wrong wiring actually does, rather than measuring how
        # far off it happens to land: the sky is GONE. Only the last source
        # survives, so scaling the sky maps changes nothing at all.
        def wrongly_wired(maps):
            replaced = (
                SkySourceOperator(sky_model=MapSky(maps=maps, freq=freq), projector=projector()),
            ) + sources[1:]
            return Pipeline(
                SelectOperator(
                    Pipeline(*replaced, loss,
                             names=("sky", "ground", "atm", "antenna_loss")),
                    CalLoadOperator(t_load=jnp.array(300.0)),
                    names=("antenna", "load"), switch_key="receiver_input",
                ),
                make_noise_wave(freq),
                names=("receiver_input", "noise_wave"),
            )(State(coords=coords)).data

        assert jnp.all(jnp.isfinite(wrongly_wired(sky_maps)))
        assert jnp.array_equal(wrongly_wired(sky_maps), wrongly_wired(3.0 * sky_maps))


class TestCalibrationClosure:
    """The point of all of it: solve for the noise waves with the sky known."""

    N_SOURCE = 3  # antenna + two loads -- the minimum for a per-channel fit

    def three_branch_twin(self, sky_maps, freq, t_nw):
        """Straight out of ``assemble()`` -- no hand-wired selector."""
        g_ant = rcj.cable_gamma(
            rcj.termination_gamma("open", N_FREQ), freq, length=2.0, loss=0.92
        )
        g_ambient = rcj.termination_gamma("resistive", N_FREQ, impedance=10.0)
        g_short = rcj.cable_gamma(
            rcj.termination_gamma("short", N_FREQ), freq, length=0.4, loss=0.98
        )
        g_src = jnp.stack([g_ant, g_ambient, g_short])
        g_rec = rcj.termination_gamma("resistive", N_FREQ, impedance=45.0)
        return assemble(
            SkySourceOperator(sky_model=MapSky(maps=sky_maps, freq=freq), projector=projector()),
            CalLoadOperator(t_load=jnp.array(300.0)),
            CalLoadOperator(t_load=jnp.array(400.0)),
            NoiseWaveOperator(
                t_unc=t_nw[0], t_cos=t_nw[1], t_sin=t_nw[2], t_rx=jnp.array(290.0),
                gamma_src_re=g_src.real, gamma_src_im=g_src.imag,
                gamma_rec_re=g_rec.real, gamma_rec_im=g_rec.imag,
            ),
        )

    def test_three_sources_make_the_per_channel_system_full_rank(
        self, sky_maps, freq
    ):
        """Each switch position gives one equation per channel; with the sky
        known, the antenna counts as a source like any other. Three distinct
        Gamma therefore make the per-channel 3x3 system square -- the same
        counting as ``examples/noise_wave_gcr.py``, now with a real sky in the
        T_src column.

        Three because ``T_rx`` is held known here and only ``t_unc, t_cos,
        t_sin`` are varied below. The general rule is ``min(n_src, k) * n_freq``
        over the ``k`` FREE temperature families, measured in
        ``tests/radio/test_noise_wave.py::TestPerChannelRankRule``."""
        switch = jnp.arange(N_TIME) % self.N_SOURCE
        coords = make_coords(freq, switch)
        state = State(coords=coords)

        def predict(flat):
            t_nw = flat.reshape(3, N_FREQ)
            return self.three_branch_twin(sky_maps, freq, t_nw)(state).data

        jac = np.asarray(jax.jacfwd(predict)(jnp.zeros(3 * N_FREQ)))
        jac = jac.reshape(N_TIME * N_FREQ, 3 * N_FREQ)
        assert np.linalg.matrix_rank(jac, tol=1e-4) == 3 * N_FREQ

    def test_the_noise_waves_are_recovered_with_the_sky_known(self, sky_maps, freq):
        """Simulate with a truth, solve in closed form, compare. The sky is not
        a nuisance here -- it is data, supplied by limTOD, and its only job is
        to be right."""
        from rheplicant.inference import (
            Bind,
            Latent,
            ParameterSpace,
            check_linearity,
            linear_operator,
            wiener_solve,
        )

        truth = jnp.stack([
            jnp.linspace(240.0, 260.0, N_FREQ),
            jnp.linspace(20.0, 40.0, N_FREQ),
            jnp.linspace(-45.0, -35.0, N_FREQ),
        ])
        switch = jnp.arange(N_TIME * 4) % self.N_SOURCE
        coords = Coordinates(
            time=jnp.arange(float(N_TIME * 4)),
            freq=freq,
            extra={
                "lst_deg": DriftScanProjector.uniform_lst_grid(N_TIME * 4),
                "receiver_input": switch,
            },
        )
        state = State(coords=coords)
        observed = self.three_branch_twin(sky_maps, freq, truth)(state).data

        # A representative init, not zeros: check_linearity takes its probe
        # magnitude from max|init|, and an all-zero init falls back to 1.0 --
        # which in float32 probes the model 0.001 K away from a ~500 K
        # prediction and measures nothing but roundoff.
        space = ParameterSpace(
            latents=[Latent("t_nw", init=jnp.full((3, N_FREQ), 100.0), linear=True)],
            bindings=[
                Bind("t_nw", into=(lambda p: p["noise_wave"].t_unc,
                                   lambda p: p["noise_wave"].t_cos,
                                   lambda p: p["noise_wave"].t_sin),
                     fn=lambda v: (v[0], v[1], v[2])),
            ],
        )
        start = self.three_branch_twin(sky_maps, freq, jnp.zeros((3, N_FREQ)))
        errors = check_linearity(space, start, state)  # raises if not affine
        assert errors[1.0] < (1e-12 if X64 else 1e-4)

        block = linear_operator(space, start, state)
        solved, _ = wiener_solve(block, observed, noise_std=0.5, prior_std=100.0,
                                 tol=1e-8, maxiter=2000)
        assert jnp.allclose(solved, truth, rtol=0.0, atol=1.0)


class TestTransforms:
    def test_the_composite_jits(self, sky_maps, freq):
        coords = make_coords(freq, ALTERNATING)
        twin = make_twin(sky_maps, freq)
        out = eqx.filter_jit(twin)(State(coords=coords)).data
        assert out.shape == (N_TIME, N_FREQ)
        assert jnp.all(jnp.isfinite(out))

    def test_gradients_reach_the_sky_maps_and_the_noise_wave_temperatures(
        self, sky_maps, freq
    ):
        """One differentiable object from the HEALPix sky to the receiver
        output -- the reason the port was worth doing at all."""
        coords = make_coords(freq, ALTERNATING)
        state = State(coords=coords)

        def loss(maps, t_unc):
            twin = assemble(
                SkySourceOperator(sky_model=MapSky(maps=maps, freq=freq), projector=projector()),
                CalLoadOperator(t_load=jnp.array(300.0)),
                eqx.tree_at(lambda o: o.t_unc, make_noise_wave(freq), t_unc),
            )
            return jnp.sum(twin(state).data ** 2)

        d_maps, d_t_unc = jax.grad(loss, argnums=(0, 1))(sky_maps, jnp.array(250.0))
        assert jnp.all(jnp.isfinite(d_maps)) and jnp.any(d_maps != 0.0)
        assert jnp.isfinite(d_t_unc) and d_t_unc != 0.0
