"""kind: mmodes -- the m-mode spectrum of a drift scan (schema section 4.7.9).

The exit takes two ``{ref}``s and nothing else, so almost every way of
getting it wrong still returns a finite, correctly-shaped complex array.
This module is written against that: the document below is built so that
every plausible width of the second axis is a DIFFERENT number, and every
property test pins a value that moves when the thing it is about moves.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant.config import ConfigError
from rheplicant.config.document import load_document
from rheplicant.config.refs import resolve_reference
from rheplicant.config.sections.runs import run_document
from tests.config.test_config_document import synthetic_document

pytest.importorskip("limtod_jax", reason="limTOD[jax] not installed")
pytest.importorskip("healpy", reason="healpy comes with limTOD")

LMAX = 8
HIGH_LMAX = 11                 # a second projector, everything else equal
NSIDE = 4
N_PIX = 12 * NSIDE**2          # 192, what _validate_sky demands at nside 4
N_FREQ = 8                     # synthetic_document's own freq grid
N_TIME = 16                    # synthetic_document's own time grid
N_ALM = (LMAX + 1) * (LMAX + 2) // 2          # 45 -- the SKY's alm width
HIGH_N_ALM = (HIGH_LMAX + 1) * (HIGH_LMAX + 2) // 2   # 78
SPECTRAL_INDEX = 4.0
FREQ_LO_MHZ = 60.0             # synthetic_document's linspace endpoints
FREQ_HI_MHZ = 85.0
LST0_DEG = 30.0                # NOT zero: see TestTheProduct's anchor test

RUN = {"kind": "mmodes",
       "projector": {"ref": "resources.projectors.drift"},
       "sky": {"ref": "resources.sky_models.fg"}}


def _projector(beam, lmax, **overrides):
    spec = {
        "engine": "driftscan",
        "beam": {"ref": f"resources.beams.{beam}"},
        "lmax": lmax,
        "lat_deg": {"value": 53.2367, "unit": "deg"},
        "az_deg": {"value": 0.0, "unit": "deg"},
        "el_deg": {"value": 90.0, "unit": "deg"},
        "normalize_beam": False,
        "acknowledge_float32_sky": True,
    }
    spec.update(overrides)
    return spec


def _gaussian(fwhm_deg):
    return {"format": "gaussian",
            "fwhm_deg": {"value": fwhm_deg, "unit": "deg"},
            "nside": NSIDE, "normalize": "pixel_sum", "frame": "beam_local"}


def _power_law(amplitude):
    return {"kind": "power_law", "amplitude": amplitude,
            "spectral_index": SPECTRAL_INDEX,
            "ref_freq": {"value": FREQ_LO_MHZ, "unit": "MHz"},
            "n_pix": N_PIX}


def document(run, *, normalize_beam=False, optimizations=None,
             lst0_deg=LST0_DEG, el_deg=90.0, materialise=("pointing",),
             lst=True):
    """The synthetic document plus a drift scan it can actually m-mode.

    ``pointing.mode: drift`` is what writes ``coords.extra["lst_deg"]``, and
    its az/el are the projector's own so ``_reject_disagreeing_pointing``
    (driftscan.py:398-436, 1e-3 deg) has nothing to object to.  ``el_deg``,
    ``materialise`` and ``lst`` are parameters only so the pointing tests
    below can make the track disagree, make it absent, and take the LST grid
    away entirely.

    Four projectors and two skies, so that the tests can vary ONE thing:

    * ``drift`` -- the horn beam at lmax 8.
    * ``narrow_beam`` -- the SAME lmax, a different beam.
    * ``high_lmax`` -- the SAME beam, lmax 11.
    * ``general`` -- the engine that has no ``mmodes()``.
    * ``fg`` -- a spatially flat power law, whose m=0 coefficient carries the
      spectrum exactly.
    * ``mottled`` -- the same power law with a per-pixel amplitude, so the
      m > 0 coefficients are signal rather than pixelisation noise.
    """
    doc = synthetic_document()
    doc["observation"]["pointing"] = {
        "mode": "drift",
        "az_deg": {"value": 0.0, "unit": "deg"},
        "el_deg": {"value": el_deg, "unit": "deg"},
        "materialise": list(materialise),
    }
    if lst:
        doc["observation"]["pointing"]["lst"] = {
            "mode": "uniform_turn", "n_time": "n_time",
            "lst0_deg": {"value": lst0_deg, "unit": "deg"}}
    drift = _projector("horn", LMAX, normalize_beam=normalize_beam)
    if optimizations is not None:
        # cache_beam_rotation is refused without lst_ref_deg
        # (projectors.py:178-185): to_reference_frame() raises without one.
        drift["optimizations"] = optimizations
        drift["lst_ref_deg"] = {"value": 0.0, "unit": "deg"}
    doc["resources"] = {
        "beams": {"horn": _gaussian(60.0), "narrow": _gaussian(20.0)},
        "projectors": {
            "drift": drift,
            "narrow_beam": _projector("narrow", LMAX),
            "high_lmax": _projector("horn", HIGH_LMAX),
            "general": {"engine": "general_pointing",
                        "beam": {"ref": "resources.beams.horn"},
                        "lmax": LMAX, "nside": NSIDE,
                        "lat_deg": {"value": 53.2367, "unit": "deg"},
                        "normalize_beam": False,
                        "acknowledge_float32_sky": True},
        },
        # Two callables that are not sky models, delivered by the front door:
        # a `python:` value node with no `args:` hands over the UNCALLED
        # attribute (hatch.py's presence-of-the-key rule), so `{ref}` to one
        # of these reaches the exit as an object that passes `callable()`.
        "arrays": {"two_argument": {"python": "operator:add"},
                   "shapeless": {"python": "builtins:repr"}},
        "sky_models": {
            "fg": _power_law({"value": 300.0, "unit": "K"}),
            "mottled": _power_law({"linspace": {"start": 100.0, "stop": 400.0,
                                                "num": N_PIX,
                                                "endpoint": True},
                                   "unit": "K"}),
        },
    }
    doc["runs"] = [run]
    return doc


def product(run=RUN, **kwargs):
    """The one run's product, as a numpy array."""
    return np.asarray(run_document(document(run, **kwargs))["mmodes"].product)


def _by_hand(doc, projector="resources.projectors.drift",
             sky="resources.sky_models.fg"):
    """The two calls the exit is supposed to make, spelled out."""
    built = load_document(doc)
    return (resolve_reference(projector, built.context),
            resolve_reference(sky, built.context),
            built.state.coords)


class TestTheProduct:
    def test_the_second_axis_is_lmax_plus_one_and_the_dtype_is_complex64(self):
        """The shape is the whole point of the row.

        At lmax 8 the m-mode axis is 9.  Every other width this document can
        offer is a different number -- the SKY's alm width at the same lmax
        is (lmax+1)(lmax+2)//2 = 45, n_freq is 8, n_time is 16, n_pix is 192
        -- and the assertion below says so, so that a later edit which
        collapsed two of them would fail HERE rather than quietly making the
        shape check undiscriminating.  forward() on the same document
        returns the real (n_time, n_freq) = (16, 8) waterfall.  The dtype pin
        is the second half: under the default x32 mode mmodes comes back
        complex64, so an executor that returned jnp.abs() of it -- float32,
        same shape -- fails here rather than downstream.
        """
        assert len({LMAX + 1, N_ALM, N_FREQ, N_TIME, N_PIX}) == 5
        result = run_document(document(RUN))["mmodes"]
        assert result.kind == "mmodes"
        assert result.error is None
        assert result.product.shape == (N_FREQ, LMAX + 1) == (8, 9)
        assert result.product.dtype == jnp.complex64
        assert bool(jnp.all(jnp.isfinite(result.product)))

    def test_the_second_axis_follows_the_projector_the_ref_names(self):
        """lmax is the PROJECTOR's, and only a second projector can say so.

        ``high_lmax`` differs from ``drift`` in exactly one key.  Its m-mode
        axis is 12 and its own alm width is 78; the sky, the coords, the
        beam and the frequency grid are untouched.  An executor that read
        the axis off anything else -- the sky's alm width, n_freq, a
        constant -- returns (8, 9) for both refs and passes the test above.
        """
        assert len({HIGH_LMAX + 1, HIGH_N_ALM, N_FREQ, N_TIME, N_PIX,
                    LMAX + 1}) == 6
        high = product({**RUN,
                        "projector": {"ref": "resources.projectors.high_lmax"}})
        assert high.shape == (N_FREQ, HIGH_LMAX + 1) == (8, 12)

    def test_the_sky_model_is_evaluated_on_the_runs_own_frequency_grid(self):
        """A power law makes the grid observable in the product.

        mmodes is linear in the sky and this sky is spatially flat, so
        |V_0(f)| carries the model's spectrum exactly: the ratio between the
        band edges is (85/60)^-4 = 0.2483.  An executor that evaluated the
        model on ANY constant or index-valued grid returns a flat spectrum
        and a ratio of 1.0 -- finite, correctly shaped, and wrong by a
        factor of four.  MEASURED here: 0.24827304.
        """
        magnitude = np.abs(product())
        assert magnitude[-1, 0] / magnitude[0, 0] == pytest.approx(
            (FREQ_HI_MHZ / FREQ_LO_MHZ) ** -SPECTRAL_INDEX, rel=1e-4)

    def test_the_product_is_the_projectors_own_mmodes_of_the_models_maps(self):
        """Element for element, against the two calls spelled out by hand.

        Everything else here pins a property; this pins the composition.  A
        swapped pair of {ref}s, a sky handed over un-evaluated, or coords
        rebuilt rather than taken off built.state all survive a shape check
        and die here.
        """
        doc = document(RUN)
        projector, sky, coords = _by_hand(doc)
        expected = projector.mmodes(sky(coords.freq), coords)
        assert jnp.array_equal(run_document(doc)["mmodes"].product, expected)

    def test_the_beam_comes_from_the_projectors_own_state(self):
        """mmodes(sky, coords) has no beam argument -- and it uses one anyway.

        ``narrow_beam`` is ``drift`` with a 20 deg beam instead of 60, and
        nothing else: same lmax, same sky, same coords, same grid.  So the
        only route by which the two products can differ is the traced
        ``beam_alms`` the projector carries.  The sky is ``mottled``, so the
        m > 0 coefficients are signal too.  MEASURED at the first channel:
        |V_0| 146.44 against 167.53 and |V_1| 2.6176 against 3.5932 -- 14%
        and 37% apart, far outside float32 roundoff.  A shape-and-finiteness
        test cannot tell whether the beam was consulted at all; this can.
        """
        run = {**RUN, "sky": {"ref": "resources.sky_models.mottled"}}
        wide = np.abs(product(run))
        narrow = np.abs(product({**run,
                                 "projector": {"ref":
                                               "resources.projectors."
                                               "narrow_beam"}}))
        assert wide.shape == narrow.shape
        assert wide[0, 0] == pytest.approx(146.44, rel=1e-3)
        assert narrow[0, 0] == pytest.approx(167.53, rel=1e-3)
        assert wide[0, 1] == pytest.approx(2.6176, rel=1e-3)
        assert narrow[0, 1] == pytest.approx(3.5932, rel=1e-3)

    def test_the_coefficients_are_the_spectrum_of_the_drift_scan_tod(self):
        """What the kind is NAMED after, and the one thing shape cannot say.

        An m-mode is the Fourier coefficient of the sidereal-day-periodic
        TOD (driftscan.py:663-671).  So the product must equal
        ``rfft(forward(same sky, same coords), axis=0) / n_time``, bin for
        bin, and it does -- on the ``mottled`` sky, whose per-pixel
        amplitude gives the m > 0 bins real signal instead of pixelisation
        noise.  MEASURED relative agreement at channel 0: 3e-18 at m=0,
        7.5e-7 at m=1, 4.1e-2 at m=7 (float32, on a coefficient of
        magnitude 1e-4).

        m = lmax is excluded because this document has n_time = 2 * lmax, so
        bin 8 of a 16-sample rfft is the NYQUIST bin -- it folds +8 and -8
        together and is 0.52 away from the m-mode, which is a statement
        about sampling and not about this exit (measured; at n_time 64 the
        same comparison agrees to 1e-3 over every bin including m = lmax).
        """
        run = {**RUN, "sky": {"ref": "resources.sky_models.mottled"}}
        doc = document(run)
        projector, sky, coords = _by_hand(doc, sky="resources.sky_models."
                                                   "mottled")
        tod = np.asarray(projector.forward(sky(coords.freq), coords))
        assert tod.shape == (N_TIME, N_FREQ)
        spectrum = np.fft.rfft(tod, axis=0) / N_TIME
        assert np.allclose(product(run)[:, :LMAX], spectrum[:LMAX].T,
                           rtol=1e-2, atol=1e-4)

    def test_the_phases_are_anchored_on_the_documents_own_lst_grid(self):
        """coords come off built.state, and the anchor is visible in them.

        ``mmodes`` measures its phases from the reference LST, which with no
        ``lst_ref_deg`` is the grid's FIRST sample (driftscan.py:476-479).
        This document starts its turn at 30 deg rather than 0, and that is
        deliberate: MEASURED on the mottled sky, m=1 at channel 0 moves from
        -0.1344+2.6142j to -1.4235+2.1968j -- same magnitude, a rotated
        phase -- so an executor that synthesised its own LST grid, or read a
        zero-anchored one, fails every element-wise assertion in this class
        instead of passing them by accident.
        """
        run = {**RUN, "sky": {"ref": "resources.sky_models.mottled"}}
        anchored = product(run)
        at_zero = product(run, lst0_deg=0.0)
        assert anchored[0, 1] == pytest.approx(-1.4235 + 2.1968j, rel=1e-3)
        assert at_zero[0, 1] == pytest.approx(-0.13437 + 2.6142j, rel=1e-3)
        assert np.allclose(np.abs(anchored), np.abs(at_zero), rtol=1e-4)


class TestThePackagesOwnPointingRefusal:
    """coords.pointing is not this layer's business, in both directions."""

    def test_a_disagreeing_pointing_is_the_packages_refusal_not_this_ones(self):
        """The trap: an mmodes run can fail for a reason mmodes is not about.

        ``pointing.el_deg: 45`` writes a per-sample track the projector
        would silently ignore, and DriftScanProjector refuses it at 1e-3 deg
        (driftscan.py:398-436).  That refusal is the package's -- it names
        the disagreement and this layer neither duplicates nor swallows it,
        so it must arrive as a StateValidationError and NOT as a ConfigError.
        """
        from rheplicant.core.errors import StateValidationError

        with pytest.raises(StateValidationError,
                           match="pointing this projector would ignore") as e:
            run_document(document(RUN, el_deg=45.0))
        assert not isinstance(e.value, ConfigError)

    def test_no_materialised_pointing_at_all_still_runs(self):
        """The other leg: the check is about DISAGREEMENT, not presence.

        With ``materialise: []`` there is no coords.pointing to compare, and
        mmodes reads only coords.extra["lst_deg"].  A layer that had decided
        to require a materialised pointing would refuse a document the
        package accepts.
        """
        assert product(materialise=()).shape == (N_FREQ, LMAX + 1)

    def test_a_document_with_no_lst_grid_gets_the_packages_own_words(self):
        """The half of the trap that is about what mmodes DOES read.

        Drop ``pointing.lst:`` and coords.extra carries no "lst_deg", which
        is the one entry mmodes requires (driftscan.py:387-393).  This layer
        adds no check of its own there either: the package's message names
        the missing key AND says in the same breath that coords.pointing is
        ignored, which is the whole confusion a config-layer paraphrase
        would have to reproduce.
        """
        from rheplicant.core.errors import StateValidationError

        with pytest.raises(StateValidationError,
                           match=r'requires coords\.extra\["lst_deg"\]') as e:
            run_document(document(RUN, lst=False))
        assert not isinstance(e.value, ConfigError)


class TestTheNormalizeBeamRefusal:
    def test_it_speaks_in_the_config_layers_voice_quoting_the_source(self):
        """ConfigError, not the package's StateValidationError.

        The package refuses this pairing too (driftscan.py:687-697), but as
        a StateValidationError -- a SIBLING of ConfigError under DirtError,
        not a subclass -- and only once the beam has been read and analysed.
        pytest.raises(ConfigError) therefore does not catch the package's
        version, and the type assertion says so out loud.  The regex is the
        SOURCE's spelling of the measurement (driftscan.py:676): ASCII
        lowercase x, no space.
        """
        with pytest.raises(ConfigError, match=r"measured ~18x off") as caught:
            run_document(document(RUN, normalize_beam=True))
        assert type(caught.value) is ConfigError
        assert "runs['mmodes']" in str(caught.value)

    def test_the_packages_refusal_could_not_have_satisfied_that_match(self):
        """What makes the assertion above capable of failing.

        Deleting this layer's refusal leaves the package's, which fires a
        few lines deeper.  MEASURED, so that the test above is known to be
        about the earlier one and not merely about "something raised":
        StateValidationError is not a subclass of ConfigError, and its
        message does not contain the source's own 'measured ~18x off' --
        that phrase lives in the mmodes() DOCSTRING (driftscan.py:676), not
        in the exception text.
        """
        from rheplicant.core.errors import StateValidationError

        assert not issubclass(StateValidationError, ConfigError)
        assert not issubclass(ConfigError, StateValidationError)
        doc = document(RUN, normalize_beam=True)
        projector, sky, coords = _by_hand(doc)
        assert projector.normalize_beam is True
        with pytest.raises(StateValidationError) as caught:
            projector.mmodes(sky(coords.freq), coords)
        assert "measured ~18x off" not in str(caught.value)

    def test_it_survives_the_projector_being_replaced_by_the_optimisation(self):
        """normalize_beam is read off the BUILT object, not off the spec.

        optimizations: [cache_beam_rotation] swaps the projector for
        to_reference_frame()'s return (projectors.py:240-241), which is a
        different instance carrying beam_frame="reference".  MEASURED: it
        keeps normalize_beam=True and it is NOT the object the spec
        described, so the refusal must still fire -- an executor that keyed
        the check on the frame, or on a spec read before the swap, passes
        the test above and fails this one.
        """
        doc = document(RUN, normalize_beam=True,
                       optimizations=["cache_beam_rotation"])
        projector, _, _ = _by_hand(doc)
        assert projector.beam_frame == "reference"
        assert projector.normalize_beam is True
        with pytest.raises(ConfigError, match=r"measured ~18x off"):
            run_document(doc)

    def test_the_optimisation_alone_does_not_refuse_anything(self):
        """The other leg: a cached rotation on a normalize_beam: false
        projector is an ordinary run, so the refusal above is about the flag
        and not about the frame."""
        doc = document(RUN, optimizations=["cache_beam_rotation"])
        assert np.asarray(
            run_document(doc)["mmodes"].product).shape == (N_FREQ, LMAX + 1)


class TestTheGrammar:
    def test_a_beam_key_is_swept_because_mmodes_has_no_beam_argument(self):
        """mmodes(sky, coords) takes the beam from projector.beam_alms.

        A beam: key would name an argument that does not exist, so it must
        not be quietly accepted and dropped.
        """
        with pytest.raises(ConfigError, match=r"does not take \['beam'\]"):
            run_document(document({**RUN,
                                   "beam": {"ref": "resources.beams.horn"}}))

    def test_coords_are_not_an_exit_option_either(self):
        """They come off built.state; a coords: key names nothing."""
        with pytest.raises(ConfigError, match=r"does not take \['coords'\]"):
            run_document(document({**RUN, "coords": {"ref": "resources"}}))

    @pytest.mark.parametrize("key", ["projector", "sky"])
    def test_both_refs_are_required(self, key):
        """And the refusal names the RUN, not just the key."""
        run = {name: value for name, value in RUN.items() if name != key}
        with pytest.raises(ConfigError,
                           match=rf"runs\['mmodes'\]: {key}: is \{{ref"):
            run_document(document(run))

    @pytest.mark.parametrize("key", ["projector", "sky"])
    @pytest.mark.parametrize("form", ["dotted", "extra_key", "null"])
    def test_neither_ref_may_be_written_any_other_way(self, key, form):
        """One helper serves both keys, so both must refuse the same shapes.

        A dotted string, a {ref} carrying a second key, and a declared-empty
        value are the three near-misses; each is refused by name rather than
        reaching resolve_reference or the projector.  Parametrised over both
        keys because a guard closed on one route and open on its twin is the
        shape this layer keeps rediscovering.
        """
        written = {"dotted": RUN[key]["ref"],
                   "extra_key": {**RUN[key], "unit": "K"},
                   "null": None}[form]
        with pytest.raises(ConfigError,
                           match=rf"runs\['mmodes'\]: {key}: is \{{ref"):
            run_document(document({**RUN, key: written}))

    def test_a_general_pointing_projector_names_the_engine_that_has_mmodes(
            self):
        """grep 'def mmodes' src/ returns two hits, both on
        DriftScanProjector.

        GeneralPointingProjector has no mmodes(); without this check the
        refusal is an AttributeError from inside the executor, which breaks
        the layer's single-ConfigError contract.
        """
        with pytest.raises(ConfigError,
                           match=r"GeneralPointingProjector, which has no "
                                 r"mmodes\(\)"):
            run_document(document(
                {**RUN,
                 "projector": {"ref": "resources.projectors.general"}}))

    def test_a_projector_ref_that_is_not_a_projector_names_the_same_thing(
            self):
        """resources.beams.<name> resolves; it simply has no mmodes()."""
        with pytest.raises(ConfigError,
                           match=r"Beam, which has no mmodes\(\)"):
            run_document(document(
                {**RUN, "projector": {"ref": "resources.beams.horn"}}))

    def test_a_sky_ref_that_is_not_a_model_names_the_kind_that_is(self):
        """resources.beams.<name> resolves to a Beam, which is not callable.

        The exit's whole sky contract is __call__(freq) -> (n_freq, n_pix);
        anything else must be refused by name rather than raising TypeError
        halfway into a trace.
        """
        with pytest.raises(ConfigError,
                           match=r"Beam, which is not a sky model") as caught:
            run_document(document({**RUN,
                                   "sky": {"ref": "resources.beams.horn"}}))
        assert "resources.sky_models.<name>" in str(caught.value)
        assert "runs['mmodes']" in str(caught.value)

    def test_a_sky_that_takes_two_arguments_is_refused_by_arity(self):
        """callable() is weaker than the projector side's own check.

        ``hasattr(projector, "mmodes")`` asks for the exact method this exit
        calls; ``callable(sky_model)`` asks only that SOMETHING can be
        called.  A ``{python: 'operator:add'}`` array resource is callable,
        reaches ``sky_model(freq)``, and without this guard escapes as
        ``TypeError: add expected 2 arguments, got 1`` naming no run -- the
        same hole ``_scores_a_pair`` was written to close on
        ``objective: {python: ...}``.  Measured before the guard existed.
        """
        with pytest.raises(ConfigError,
                           match=r"cannot be called as \(freq\)") as caught:
            run_document(document(
                {**RUN, "sky": {"ref": "resources.arrays.two_argument"}}))
        assert "runs['mmodes']" in str(caught.value)

    def test_a_sky_whose_call_returns_no_maps_is_refused_before_the_package(
            self):
        """The other end of the same callable: right arity, wrong product.

        ``repr`` takes one argument and returns a str.  _validate_sky
        (driftscan.py:551-557) reaches for ``.shape`` first and would raise
        a bare AttributeError; the extents themselves are still left to it,
        because its message names the nside they follow from.
        """
        with pytest.raises(ConfigError,
                           match=r"returned a str, not maps"):
            run_document(document(
                {**RUN, "sky": {"ref": "resources.arrays.shapeless"}}))

    def test_a_sky_ref_naming_a_projector_is_refused_before_it_is_traced(self):
        """The twin of the case above: a projector INSTANCE is not callable
        either (measured), so the same guard catches the swap of the two
        refs -- which is otherwise a run that traces and then dies inside
        jax."""
        with pytest.raises(ConfigError,
                           match=r"DriftScanProjector, which is not a sky "
                                 r"model"):
            run_document(document(
                {**RUN, "sky": {"ref": "resources.projectors.drift"}}))
