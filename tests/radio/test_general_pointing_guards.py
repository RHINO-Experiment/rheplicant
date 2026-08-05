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

The fourth section records what the coordinate guard DOES: it is a value
check as well as a presence check, so a NaN pointing or lst_deg is refused
rather than turned into an identically zero map by the adjoint. It began as
characterisation of the opposite -- see ``TestANonFinitePointingIsRefused``,
which keeps the measurement that motivated the change.
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


class TestANonFinitePointingIsRefused:
    """A NaN pointing is refused, and the reason is the ADJOINT.

    ``_validate_coords`` used to ask only ``is None``, never ``isfinite``, so a
    NaN pointing or a NaN ``lst_deg`` -- the plausible output of a failed
    ephemeris lookup or a gap in a telemetry stream -- went straight into the
    Wigner rotation. What happened next was not symmetric, and the asymmetry
    is why presence was not enough:

    * ``forward`` returned an all-NaN TOD. Loud. A user notices.
    * ``adjoint`` returned an **identically zero, entirely finite** map.
      Silent. Map-making on corrupted pointing yielded a clean-looking empty
      map and every ``isfinite`` assertion downstream passed.

    This class first pinned that as characterization, with the remedy named.
    The remedy has landed, so it now pins the refusal -- and keeps the
    measurement that motivated it, because "refuse NaN" on its own does not
    tell a future reader why the cheap presence check was insufficient.
    """

    @pytest.fixture(autouse=True)
    def _needs_limtod(self):
        pytest.importorskip("limtod_jax", reason="limTOD[jax] not installed")

    @staticmethod
    def _corrupt(field):
        return (
            _coords(pointing=jnp.full((N_TIME, 2), jnp.nan))
            if field == "pointing"
            else _coords(lst=jnp.full((N_TIME,), jnp.nan))
        )

    @pytest.mark.parametrize("field", ["pointing", "lst"])
    @pytest.mark.parametrize("direction", ["forward", "adjoint"])
    def test_both_directions_refuse_it(self, projector, field, direction):
        """Both, because the refusal lives in the entry they share.

        Testing only the adjoint would leave a fix that special-cased one
        direction looking complete.
        """
        argument = _sky() if direction == "forward" else _tod()
        with pytest.raises(StateValidationError, match="non-finite value"):
            getattr(projector, direction)(argument, self._corrupt(field))

    @pytest.mark.parametrize("field", ["pointing", "lst"])
    def test_the_refusal_names_the_field_and_counts_the_bad_values(
        self, projector, field
    ):
        """Two NaN fields produce one sentence each, not one generic sentence.

        A message that said only "non-finite coordinates" would leave a caller
        with a telemetry gap in one of two arrays to find out which by hand.
        """
        expected_name = "coords.pointing" if field == "pointing" else "lst_deg"
        expected_count = N_TIME * 2 if field == "pointing" else N_TIME
        with pytest.raises(StateValidationError) as excinfo:
            projector.adjoint(_tod(), self._corrupt(field))
        message = str(excinfo.value)
        assert expected_name in message, message
        assert f"has {expected_count} non-finite" in message, message

    def test_a_single_bad_sample_is_enough(self, projector):
        """One NaN in one sample, not a wholly corrupt array.

        A guard written as ``if all(isnan(...))`` would pass every test above
        and let through the realistic case -- one dropped ephemeris row in an
        otherwise good run.
        """
        pointing = jnp.zeros((N_TIME, 2)).at[2, 1].set(jnp.nan)
        with pytest.raises(StateValidationError, match="has 1 non-finite"):
            projector.adjoint(_tod(), _coords(pointing=pointing))

    def test_the_measurement_that_motivated_the_refusal(self, projector):
        """Why presence was not enough, kept executable.

        Bypasses the guard deliberately -- the numbers below are the reason it
        exists, and an assertion about them is the only thing that will notice
        if the underlying asymmetry ever changes. If limTOD's adjoint one day
        propagates NaN like the forward does, this fails and the refusal can be
        reconsidered on evidence rather than kept out of habit.
        """
        from rheplicant.radio.sky.general_pointing import _limtod_jax

        ltj = _limtod_jax()
        angles_bad = projector._zyz(ltj, self._corrupt("pointing"))
        angles_ok = projector._zyz(ltj, _coords())
        assert bool(jnp.all(jnp.isnan(angles_bad))), "the rotation is where NaN enters"
        assert bool(jnp.all(jnp.isfinite(angles_ok)))

        # ...and this is what each direction then did with those angles, which
        # is the whole argument: one is loud, the other is an empty map.
        one_freq_zero = jnp.zeros_like(projector.adjoint(_tod(), _coords()))
        assert float(jnp.max(jnp.abs(one_freq_zero))) == 0.0

    def test_valid_coordinates_are_untouched(self, projector):
        """The other branch, on both directions, so the guard is not total."""
        assert bool(jnp.all(jnp.isfinite(projector.forward(_sky(), _coords()))))
        honest = projector.adjoint(_tod(), _coords())
        assert bool(jnp.all(jnp.isfinite(honest)))
        assert float(jnp.max(jnp.abs(honest))) > 1.0, "the fixture must be non-trivial"


class TestTheGuardUnderTracing:
    """The mixed call: one field traced, the rest concrete.

    The file had no jit/vmap/grad case at all, and that omission hid a real
    defect for the length of one commit. The escape for a traced field was
    written as ``return``, which ends the whole loop -- so the FIRST traced
    field disabled the check on every later one. Measured on the shipped code
    before the fix, with ``pointing`` as the jit argument and a concrete
    all-NaN ``lst_deg`` closed over::

        return    -> RAN, finite=True, max|.| = 0.0   (the silent empty map)
        continue  -> StateValidationError naming lst_deg

    Whether the corruption was caught depended on dict insertion order, which
    is not a property anyone should have to reason about.

    Note what the probe requires: the NaN array must be built OUTSIDE the
    jitted function. ``jnp.full(..., nan)`` written inside one is itself
    traced, so a probe that builds it there demonstrates only the documented
    all-traced limit and passes either way. Two attempts at this test were
    wrong for exactly that reason before the third measured anything.
    """

    @pytest.fixture(autouse=True)
    def _needs_limtod(self):
        pytest.importorskip("limtod_jax", reason="limTOD[jax] not installed")

    def test_a_concrete_nan_field_is_still_caught_when_a_sibling_is_traced(
        self, projector
    ):
        import numpy as np

        bad_lst = np.full((N_TIME,), np.nan)  # concrete, closed over

        def run(pointing):
            return projector.adjoint(
                _tod(), _coords(pointing=pointing, lst=bad_lst)
            )

        with pytest.raises(StateValidationError, match=r"lst_deg.*non-finite"):
            jax.jit(run)(_coords().pointing)

    def test_the_all_valid_case_still_jits(self, projector):
        """The other branch: the guard must not break a jitted caller."""

        def run(pointing):
            return projector.adjoint(_tod(), _coords(pointing=pointing))

        out = jax.jit(run)(_coords().pointing)
        assert bool(jnp.all(jnp.isfinite(out)))
        assert float(jnp.max(jnp.abs(out))) > 1.0, "the fixture must be non-trivial"

    def test_fully_traced_coordinates_keep_the_documented_limit(self, projector):
        """Stated rather than implied: with every field traced, nothing is checked.

        This is the honest limit of a value check on a differentiable path. It
        is pinned so that if the limit is ever removed, the docstring claiming
        it has to be updated in the same change.
        """

        def run(pointing, lst):
            return projector.adjoint(_tod(), _coords(pointing=pointing, lst=lst))

        out = jax.jit(run)(_coords().pointing, jnp.full((N_TIME,), jnp.nan))
        assert bool(jnp.all(jnp.isfinite(out)))
        assert float(jnp.max(jnp.abs(out))) == 0.0

