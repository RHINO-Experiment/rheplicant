"""DriftScanProjector refuses a band where one of its s2fft transforms cannot run.

s2fft's HEALPix ANALYSIS (map to alm) needs ``nside >= 2`` and
``lmax + 1 >= 2 * nside``; its SYNTHESIS (alm to map) needs only the second.
Below either edge the transform fails inside s2fft with no message of its own:
``ValueError: Need at least one array to stack`` from the analysis at nside 1,
a ``TypeError`` from the analysis and a bare ``AssertionError`` from the
synthesis for ``lmax < 2 * nside - 1``. Measured on the baseline for nside 1-4,
8, 16, 32 and 64: every transform path failed below its edge and ran at and
above it, with no upper edge up to ``lmax = 64`` at nside 2.

``forward_alms`` and ``mmodes_alms`` on a projector with ``normalize_beam`` and
``horizon_mask`` both off run NO HEALPix transform, and returned finite values
everywhere in that domain on the baseline (nside 1 to 64, lmax 0 to 126),
including through ``to_reference_frame`` and ``uniform_sampling``. So the
refusal sits where s2fft is called -- ``sky_to_alms`` (and ``forward`` /
``mmodes``, which call it), ``adjoint``, ``horizon_fraction``,
``from_beam_maps`` -- and at construction only for ``normalize_beam`` or
``horizon_mask``, which put an analysis on every call. The config layer's
``beam_analysis`` transform refuses the analysis domain too
(``tests/config/test_config_transforms.py``).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates
from rheplicant.core.errors import StateValidationError
from rheplicant.radio.sky import DriftScanProjector

pytest.importorskip("limtod_jax")

N_TIME = 9
LAT, AZ, EL, SELFROT = 53.2, 30.0, 50.0, 10.0


def _alms(lmax):
    n_alm = (lmax + 1) * (lmax + 2) // 2
    re, im = jax.random.normal(jax.random.key(0), (2, 1, n_alm))
    return re + 1j * im.at[:, : lmax + 1].set(0.0)


def _projector(nside, lmax, **kwargs):
    return DriftScanProjector(
        beam_alms=_alms(lmax), lat_deg=LAT, az_deg=AZ, el_deg=EL,
        lmax=lmax, nside=nside, selfrot_deg=SELFROT, **kwargs,
    )


def _coords():
    return Coordinates(
        time=jnp.arange(float(N_TIME)), freq=jnp.asarray([60e6]),
        extra={"lst_deg": jnp.linspace(0.0, 40.0, N_TIME)},
    )


def _sky(nside):
    return jax.random.uniform(jax.random.key(1), (1, 12 * nside**2))


def _finite(value):
    return bool(np.isfinite(np.asarray(value)).all())


#: One under the lmax edge, ``lmax = 2 * nside - 2``, at nside 1, 2, 4 and 8:
#: every path's crash domain, and where the alm-only paths are RUN (their cost
#: is the O(lmax^3) Wigner rotation, which is not what the band limits).
BELOW_THE_EDGE = [(1, 0), (2, 2), (4, 6), (8, 14)]
#: The analysis's crash domain: the cells above, nside 1 AT its lmax edge
#: (the analysis refuses nside 1 at every lmax), and two far cells, which are
#: refused before any transform runs and so cost nothing.
CRASH = [(1, 0), (1, 1), (2, 2), (4, 6), (8, 14), (1, 40), (64, 126)]
#: The synthesis's crash domain is the same minus nside 1 at lmax >= 1.
SYNTHESIS_CRASH = [cell for cell in CRASH if cell[1] < 2 * cell[0] - 1]
#: At the edge, ``lmax = 2 * nside - 1`` (nside 1's is the synthesis test's),
#: and one cell far above it.
ACCEPTED = [(2, 3), (4, 7), (8, 15), (2, 64)]


@pytest.mark.parametrize(("nside", "lmax"), BELOW_THE_EDGE)
def test_the_alm_only_paths_are_accepted_in_the_crash_domain(nside, lmax):
    """No HEALPix transform on these paths, so no refusal either."""
    projector = _projector(nside, lmax)
    coords = _coords()
    sky_alms = _alms(lmax)
    assert _finite(projector.forward_alms(sky_alms, coords))
    assert _finite(projector.mmodes_alms(sky_alms, coords))


def test_a_cached_rotation_is_alm_only_too():
    """``to_reference_frame`` rotates the beam and runs no HEALPix transform
    either; once, at the smallest cell below the edge."""
    projector = _projector(2, 2)
    cached = projector.to_reference_frame(lst_ref_deg=0.0)
    assert _finite(cached.forward_alms(_alms(2), _coords()))


@pytest.mark.parametrize(("nside", "lmax"), CRASH)
class TestTheCrashDomain:
    def test_a_plain_projector_constructs(self, nside, lmax):
        assert _projector(nside, lmax).lmax == lmax

    @pytest.mark.parametrize("call", ["sky_to_alms", "forward", "mmodes"])
    def test_the_analysis_paths_are_refused_by_name(self, nside, lmax, call):
        projector = _projector(nside, lmax)
        args = (_sky(nside),) if call == "sky_to_alms" else (_sky(nside), _coords())
        with pytest.raises(StateValidationError) as excinfo:
            getattr(projector, call)(*args)
        message = str(excinfo.value)
        assert "sky_to_alms()" in message and "analysis" in message
        assert "nside >= 2 and lmax >= 2 * nside - 1" in message
        assert f"nside={nside}, lmax={lmax}" in message
        assert "forward_alms() and mmodes_alms()" in message  # what does run

    @pytest.mark.parametrize("flag", ["normalize_beam", "horizon_mask"])
    def test_a_flag_that_analyses_on_every_call_is_refused_at_construction(
        self, nside, lmax, flag
    ):
        """Early: with either flag, every forward and adjoint runs an analysis."""
        with pytest.raises(StateValidationError) as excinfo:
            _projector(nside, lmax, **{flag: True})
        message = str(excinfo.value)
        assert f"{flag}=True" in message
        assert f"nside={nside}, lmax={lmax}" in message


@pytest.mark.parametrize(("nside", "lmax"), SYNTHESIS_CRASH)
@pytest.mark.parametrize("call", ["adjoint", "horizon_fraction"])
def test_the_synthesis_paths_are_refused_below_their_edge(nside, lmax, call):
    projector = _projector(nside, lmax)
    args = (jnp.ones((N_TIME, 1)), _coords()) if call == "adjoint" else ()
    with pytest.raises(StateValidationError) as excinfo:
        getattr(projector, call)(*args)
    message = str(excinfo.value)
    assert f"{call}()" in message and "synthesis" in message
    assert "lmax >= 2 * nside - 1" in message and "nside >= 2" not in message


@pytest.mark.parametrize(("nside", "lmax"), [(1, 1)])
def test_the_synthesis_paths_run_at_nside_1(nside, lmax):
    """The synthesis's own domain: nside 1 is refused only by the analysis."""
    projector = _projector(nside, lmax)
    assert _finite(projector.adjoint(jnp.ones((N_TIME, 1)), _coords()))
    assert _finite(projector.horizon_fraction())


@pytest.mark.parametrize(("nside", "lmax"), ACCEPTED)
def test_the_valid_side_constructs_and_both_s2fft_transforms_run(nside, lmax):
    """The other side of the edge, through the two s2fft calls that crashed:
    the analysis ``sky_to_alms`` and the synthesis the adjoint ends in."""
    import limtod_jax as ltj

    alms = _projector(nside, lmax, normalize_beam=True).sky_to_alms(_sky(nside))
    synthesised = ltj.alm2map(alms[0], nside=nside, lmax=lmax)
    assert _finite(alms) and _finite(synthesised)


@pytest.mark.parametrize(("nside", "lmax"), [(2, 3)])
def test_at_the_edge_the_projector_itself_runs(nside, lmax):
    """The whole forward and adjoint at the edge, normalised, not only the
    transforms -- at the smallest nside, because the Wigner rotation
    dominates the cost and is not what fails."""
    coords = _coords()
    projector = _projector(nside, lmax, normalize_beam=True)
    assert _finite(projector.forward(_sky(nside), coords))
    assert _finite(projector.adjoint(jnp.ones((N_TIME, 1)), coords))


@pytest.mark.parametrize(("nside", "lmax"), [(1, 1), (4, 6)])
def test_from_beam_maps_refuses_before_the_transform(nside, lmax):
    """``from_beam_maps`` runs ``map2alm_iter`` before it constructs, so the
    refusal has to come first there too, or s2fft's error arrives instead."""
    maps = jnp.ones((1, 12 * nside**2))
    with pytest.raises(StateValidationError, match="from_beam_maps\\(\\)"):
        DriftScanProjector.from_beam_maps(maps, lat_deg=LAT, az_deg=AZ, el_deg=EL, lmax=lmax)


def test_from_beam_maps_accepts_the_edge():
    maps = jnp.ones((1, 12 * 4**2))
    built = DriftScanProjector.from_beam_maps(maps, lat_deg=LAT, az_deg=AZ, el_deg=EL, lmax=7)
    assert built.lmax == 7 and built.nside == 4


def test_a_field_edit_into_the_crash_domain_is_judged_again():
    """``dataclasses.replace`` re-runs the check, as every functional update
    does: refused with a flag that analyses on every call, accepted without."""
    with pytest.raises(StateValidationError, match="normalize_beam=True"):
        dataclasses.replace(_projector(4, 11, normalize_beam=True), nside=8)
    assert dataclasses.replace(_projector(4, 11), nside=8).nside == 8
