"""GeneralPointingProjector's three refusals, and the one it does not make.

The projector validates its inputs at three places -- the pointing coordinates
it reads, the sky it is asked to project, and the TOD it is asked to
back-project -- and a full-suite coverage run found none of the three
``raise`` lines had ever executed. Its sibling ``DriftScanProjector`` has
tests; this one had the geometry nobody wrote a fixture for.

The shapes here are ``n_time=5``, ``n_freq=3``, ``nside=2`` (``n_pix=48``),
``lmax=3`` (``n_alm=10``). Every one of those numbers is different from every
other, on purpose: the ``sky`` and ``tod`` guards are both about WHICH AXIS IS
WHICH, and a fixture with any two of them equal cannot tell a correct check
from a transposed one. The ``tod`` test below passes ``(n_freq, n_time)`` --
the exact transpose of the valid shape -- which is only a distinguishable
input because ``5 != 3``.

The fourth section records what the coordinate guard does NOT do. It is a
PRESENCE check, not a value check: it asks whether ``lst_deg`` and
``pointing`` are there, never whether they are finite. NaN passes it, and the
two directions then behave very differently -- see ``TestNaNPassesTheGuard``.
That is reported, not fixed; the fix is a source change.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import StateValidationError
from rheplicant.radio.sky import GeneralPointingProjector

N_TIME, N_FREQ = 5, 3
NSIDE, LMAX = 2, 3
N_PIX = 12 * NSIDE**2          # 48
N_ALM = (LMAX + 1) * (LMAX + 2) // 2   # 10

assert len({N_TIME, N_FREQ, N_PIX, N_ALM}) == 4, "the fixture must be unambiguous"


@pytest.fixture
def projector():
    # Structured, not flat: a zero or constant beam makes the forward map
    # degenerate, so an adjoint that dropped a term would still return zeros
    # and match. The m=0 coefficients are forced real, which is what
    # healpy-packed alms of a real field satisfy and what the projector
    # documents as its validity condition.
    key = jax.random.key(0)
    beam = jax.random.normal(key, (N_FREQ, N_ALM)) + 1j * jax.random.normal(
        jax.random.key(1), (N_FREQ, N_ALM)
    )
    beam = beam.at[:, : LMAX + 1].set(beam[:, : LMAX + 1].real)
    return GeneralPointingProjector(
        beam_alms=beam, lat_deg=-30.7, lmax=LMAX, nside=NSIDE
    )


def _coords(*, pointing=..., lst=...):
    """Valid drift-like coordinates, with either field overridable."""
    extra = {}
    if lst is not ...:
        extra["lst_deg"] = lst
    else:
        extra["lst_deg"] = jnp.linspace(0.0, 40.0, N_TIME)
    if pointing is ...:
        # Az and el differ from each other and vary in time: a fixed or
        # symmetric pointing would not exercise the per-sample rotation the
        # guard is protecting.
        pointing = jnp.stack(
            [jnp.linspace(10.0, 50.0, N_TIME), jnp.linspace(60.0, 80.0, N_TIME)],
            axis=-1,
        )
    return Coordinates(
        time=jnp.arange(float(N_TIME)),
        freq=jnp.linspace(60e6, 85e6, N_FREQ),
        pointing=pointing,
        extra=extra,
    )


def _sky():
    return jax.random.normal(jax.random.key(2), (N_FREQ, N_PIX))


def _tod():
    return jax.random.normal(jax.random.key(3), (N_TIME, N_FREQ))


class TestCoordinateGuard:
    """Both arms of ``_validate_coords``, on both entry points.

    Parametrized over ``forward``/``adjoint`` because the guard is called
    from each separately -- deleting either call site leaves the other test
    green, which is how a one-sided fix passes review.
    """

    @pytest.mark.parametrize("direction", ["forward", "adjoint"])
    def test_absent_pointing_is_refused(self, projector, direction):
        coords = _coords(pointing=None)
        payload = _sky() if direction == "forward" else _tod()
        with pytest.raises(StateValidationError, match="coords.pointing"):
            getattr(projector, direction)(payload, coords)

    @pytest.mark.parametrize("direction", ["forward", "adjoint"])
    def test_absent_lst_is_refused(self, projector, direction):
        coords = Coordinates(
            time=jnp.arange(float(N_TIME)),
            freq=jnp.linspace(60e6, 85e6, N_FREQ),
            pointing=jnp.zeros((N_TIME, 2)),
            extra={},
        )
        payload = _sky() if direction == "forward" else _tod()
        with pytest.raises(StateValidationError, match="lst_deg"):
            getattr(projector, direction)(payload, coords)

    def test_the_two_refusals_say_which_field_is_missing(self, projector):
        """Both name ``GeneralPointingProjector``; only the field differs.

        Whichever one a user hits, the actionable half of the message is the
        field name -- so a copy that reported the wrong one would still name
        the right class and still be a ``StateValidationError``.
        """
        with pytest.raises(StateValidationError) as no_pointing:
            projector.forward(_sky(), _coords(pointing=None))
        with pytest.raises(StateValidationError) as no_lst:
            projector.forward(
                _sky(),
                Coordinates(
                    time=jnp.arange(float(N_TIME)),
                    freq=jnp.linspace(60e6, 85e6, N_FREQ),
                    pointing=jnp.zeros((N_TIME, 2)),
                    extra={},
                ),
            )
        assert "lst_deg" not in str(no_pointing.value), str(no_pointing.value)
        assert "lst_deg" in str(no_lst.value), str(no_lst.value)
        assert all(
            "GeneralPointingProjector" in str(e.value) for e in (no_pointing, no_lst)
        )

    @pytest.mark.parametrize("direction", ["forward", "adjoint"])
    def test_complete_coordinates_get_past_the_guard(self, projector, direction):
        """The other branch. The guard's condition is an ``or`` of two
        clauses; collapsing it to ``True`` is a one-character edit that every
        refusal test above would survive."""
        pytest.importorskip("limtod_jax", reason="limTOD[jax] not installed")
        payload = _sky() if direction == "forward" else _tod()
        out = getattr(projector, direction)(payload, _coords())
        assert out.shape == ((N_TIME, N_FREQ) if direction == "forward" else (N_FREQ, N_PIX))
        assert bool(jnp.all(jnp.isfinite(out)))


class TestSkyShapeGuard:
    def test_a_sky_with_the_wrong_pixel_count_is_refused(self, projector):
        with pytest.raises(StateValidationError, match=f"n_pix={N_PIX}"):
            projector.forward(jnp.ones((N_FREQ, N_PIX + 1)), _coords())

    def test_a_sky_with_the_wrong_channel_count_is_refused(self, projector):
        """``n_freq`` must match the BEAM, not the coords: the beam alms are
        the per-channel object being projected through."""
        with pytest.raises(StateValidationError, match=f"n_freq={N_FREQ}"):
            projector.forward(jnp.ones((N_FREQ + 1, N_PIX)), _coords())

    def test_a_transposed_sky_is_refused(self, projector):
        """``(n_pix, n_freq)`` instead of ``(n_freq, n_pix)``.

        Distinguishable only because ``N_PIX != N_FREQ``. This is the mistake
        the guard exists for -- both axes present, wrong order -- and the one
        a square fixture would wave through.
        """
        with pytest.raises(StateValidationError) as excinfo:
            projector.forward(jnp.ones((N_PIX, N_FREQ)), _coords())
        assert f"got ({N_PIX}, {N_FREQ})" in str(excinfo.value), str(excinfo.value)

    def test_the_refusal_reports_the_shape_it_got(self, projector):
        with pytest.raises(StateValidationError) as excinfo:
            projector.forward(jnp.ones((N_FREQ, N_PIX + 1)), _coords())
        assert f"got ({N_FREQ}, {N_PIX + 1})" in str(excinfo.value), str(excinfo.value)


class TestTodShapeGuard:
    @pytest.mark.parametrize(
        "shape",
        [
            (N_FREQ, N_TIME),          # transposed -- the real-world mistake
            (N_TIME, N_FREQ + 1),      # wrong channel count
            (N_TIME + 1, N_FREQ),      # wrong sample count
            (N_TIME,),                 # rank 1
            (N_TIME, N_FREQ, 1),       # rank 3
        ],
    )
    def test_tod_that_is_not_the_expected_waterfall_is_refused(self, projector, shape):
        with pytest.raises(StateValidationError, match="tod must be"):
            projector.adjoint(jnp.ones(shape), _coords())

    def test_the_expected_shape_is_stated_and_taken_from_the_coordinates(
        self, projector
    ):
        """``n_time`` is read from ``coords.pointing``, ``n_freq`` from the
        beam. A message that transposed the two would still be a plausible
        sentence, so both numbers are pinned against a fixture where they
        differ."""
        with pytest.raises(StateValidationError) as excinfo:
            projector.adjoint(jnp.ones((N_FREQ, N_TIME)), _coords())
        message = str(excinfo.value)
        assert f"n_time={N_TIME}" in message, message
        assert f"n_freq={N_FREQ}" in message, message
        assert f"got ({N_FREQ}, {N_TIME})" in message, message


class TestNaNPassesTheGuard:
    """A finding: the coordinate guard checks presence, never values.

    ``_validate_coords`` asks ``is None``. It never asks ``isfinite``, so a
    NaN pointing or a NaN ``lst_deg`` -- the plausible output of a failed
    ephemeris lookup or a gap in a telemetry stream -- goes straight through
    into the Wigner rotation.

    What happens next is not symmetric, and the asymmetry is the point:

    * ``forward`` returns an all-NaN TOD. Loud. A user notices.
    * ``adjoint`` returns an **identically zero, entirely finite** map. Silent.
      Map-making on a corrupted pointing yields a clean-looking empty map, and
      every ``isfinite`` assertion downstream passes.

    These tests pin the current behaviour so the asymmetry is on the record.
    They are characterization, not endorsement: adding a finiteness check to
    ``_validate_coords`` is a source change and out of scope here, and it
    would turn both of these into refusals.
    """

    @pytest.fixture(autouse=True)
    def _needs_limtod(self):
        pytest.importorskip("limtod_jax", reason="limTOD[jax] not installed")

    @pytest.mark.parametrize("field", ["pointing", "lst"])
    def test_nan_coordinates_are_not_refused(self, projector, field):
        coords = (
            _coords(pointing=jnp.full((N_TIME, 2), jnp.nan))
            if field == "pointing"
            else _coords(lst=jnp.full((N_TIME,), jnp.nan))
        )
        # No raise: the guard is about presence, and NaN is present.
        projector.forward(_sky(), coords)

    @pytest.mark.parametrize("field", ["pointing", "lst"])
    def test_nan_coordinates_make_the_forward_tod_all_nan(self, projector, field):
        coords = (
            _coords(pointing=jnp.full((N_TIME, 2), jnp.nan))
            if field == "pointing"
            else _coords(lst=jnp.full((N_TIME,), jnp.nan))
        )
        out = projector.forward(_sky(), coords)
        assert bool(jnp.all(jnp.isnan(out))), "the loud direction stopped being loud"

    @pytest.mark.parametrize("field", ["pointing", "lst"])
    def test_nan_coordinates_make_the_adjoint_map_silently_empty(
        self, projector, field
    ):
        """The dangerous half. Finite, correctly shaped, and all zero.

        Compared against the valid-pointing adjoint so that "zero" is shown
        to be wrong rather than merely asserted -- on this fixture the honest
        answer is O(10), not O(0).
        """
        coords = (
            _coords(pointing=jnp.full((N_TIME, 2), jnp.nan))
            if field == "pointing"
            else _coords(lst=jnp.full((N_TIME,), jnp.nan))
        )
        corrupted = projector.adjoint(_tod(), coords)
        honest = projector.adjoint(_tod(), _coords())

        assert bool(jnp.all(jnp.isfinite(corrupted))), "no NaN survives to warn anyone"
        assert float(jnp.max(jnp.abs(corrupted))) == 0.0
        assert float(jnp.max(jnp.abs(honest))) > 1.0, "the fixture must be non-trivial"
