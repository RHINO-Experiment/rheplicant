"""The residue of the raise audit: refusals with nothing in common but being unpinned.

``tools/raise_audit.py`` cross-references every ``ast.Raise`` in ``src/``
against a coverage run. Most of what it found fell into families -- one guard
copy-pasted across nine operators, a dozen ``__check_init__`` argument checks --
and those are covered by tests that derive their population from the source, so
a tenth copy cannot arrive untested.

These are what was left: single guards, each about a different thing. There is
no family to derive, so they are written out, and each says what would go wrong
if it were removed. A test that only says "this raises" would not be worth the
line; a guard that is never executed is not merely untested, it is a claim
nobody has checked, and the cheapest way for it to be wrong is to name the
wrong thing or to fire on the wrong side of its condition.
"""

import jax
import jax.numpy as jnp
import pytest

from rheplicant.core.coordinates import Coordinates
from rheplicant.core.errors import DataIngestionError, StateValidationError
from rheplicant.core.state import State
from rheplicant.radio import GroundPickupOperator, IonosphereOperator, read_touchstone


def _write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestTouchstonePortReversal:
    """``flipped=True`` on a 1-port file.

    Port reversal exchanges two ports. A 1-port file has one, so the request is
    not merely unusual -- it has no meaning, and the array slice that would
    implement it (``s[:, ::-1, ::-1]``) is a silent no-op on a 1x1 matrix. A
    caller who passed ``flipped=True`` believing their file was 2-port would
    get exactly the numbers they asked for and none of the reversal.
    """

    #: Two frequencies, all four numbers distinct, so a reversal that DID
    #: happen would be visible rather than hidden by symmetry.
    ONE_PORT = "# MHZ S RI R 50\n60.0  0.1 0.2\n70.0  0.3 0.4\n"

    def test_flipped_on_a_one_port_file_is_refused(self, tmp_path):
        with pytest.raises(DataIngestionError, match="flipped=True on a 1-port file"):
            read_touchstone(_write(tmp_path, "one.s1p", self.ONE_PORT), flipped=True)

    def test_the_same_file_reads_fine_unflipped(self, tmp_path):
        """The other branch: the refusal is about ``flipped``, not about the file."""
        touchstone = read_touchstone(_write(tmp_path, "one.s1p", self.ONE_PORT))
        assert touchstone.s11.shape == (2,)
        assert touchstone.s11[0] == pytest.approx(0.1 + 0.2j)
        assert touchstone.s11[1] == pytest.approx(0.3 + 0.4j)


class TestTheGraphRegistrationSelfCheck:
    """``_validate_registrations`` runs at import and asserts the registry is honest.

    Every operator declaring ``graph_node`` must name a node the radio graph
    actually has. It fires at import time, so in a correct package it never
    fires at all -- which is exactly why it was never executed, and exactly why
    it is worth one test: an import-time assertion that has never once run is
    indistinguishable from an import-time assertion with a typo in it.
    """

    def test_an_operator_naming_a_node_the_graph_lacks_is_refused(self, monkeypatch):
        import rheplicant.radio as radio
        from rheplicant.radio.graph import _validate_registrations

        class Impostor:
            graph_node = "no_such_node"

        monkeypatch.setattr(radio, "Impostor", Impostor, raising=False)
        monkeypatch.setattr(radio, "__all__", [*radio.__all__, "Impostor"])

        with pytest.raises(AssertionError, match=r"Impostor.graph_node = 'no_such_node'"):
            _validate_registrations()

    def test_it_passes_on_the_package_as_shipped(self):
        """The branch that runs on every import, asserted rather than assumed.

        Without this, a mutation making the check vacuous -- inverting the
        ``in`` test, or dropping the loop body -- would be caught by nothing,
        since the test above only proves it *can* raise.
        """
        from rheplicant.radio.graph import _validate_registrations

        _validate_registrations()


class TestGroundPickupAmbientTemperature:
    """``env.temperature`` is read per time sample, and a length mismatch is refused.

    The operator falls back to its own ``t_ground`` when the environment
    carries no temperature, so the interesting case is an environment that
    carries the WRONG one: a temperature log of a different length than the
    time axis. Broadcasting would not save it -- lengths 1 and ``n_time`` are
    the two legal cases and anything else is a log from another run.
    """

    N_TIME, N_FREQ = 4, 6  # not square: a transposed read cannot pass

    def _state(self, temperature):
        from rheplicant.core.environment import Environment

        return State(
            data=jnp.zeros((self.N_TIME, self.N_FREQ)),
            coords=Coordinates(
                time=jnp.arange(float(self.N_TIME)),
                freq=jnp.linspace(60e6, 80e6, self.N_FREQ),
            ),
            env=Environment(temperature=temperature),
        )

    def test_a_temperature_log_of_the_wrong_length_is_refused(self):
        operator = GroundPickupOperator(
            coupling=jnp.array(0.05), t_ground=jnp.array(290.0)
        )
        with pytest.raises(StateValidationError, match=r"has 5 samples but coords.time has 4"):
            operator(self._state(jnp.linspace(280.0, 300.0, 5)))

    @pytest.mark.parametrize("n", [1, 4], ids=["scalar-broadcast", "per-sample"])
    def test_the_two_legal_lengths_pass(self, n):
        """Both accepted branches, because the guard's condition is a 2-tuple
        membership test and dropping either element is a one-character edit."""
        operator = GroundPickupOperator(
            coupling=jnp.array(0.05), t_ground=jnp.array(290.0)
        )
        out = operator(self._state(jnp.linspace(280.0, 300.0, n)))
        assert out.data.shape == (self.N_TIME, self.N_FREQ)
        assert jnp.all(jnp.isfinite(out.data))


class TestIonosphereNeedsAFrequencyAxis:
    """A near-miss member of the coords-guard family, and it reads differently.

    Nine operators say *"X requires state.coords with time and freq axes"* and
    are covered as a family in ``test_coords_guard_family.py``. This one needs
    only ``freq``, so its sentence differs and the family's source-derived
    population does not include it. That is correct -- it is a different
    claim -- but it is also how it came to be the one left uncovered, so the
    difference is asserted here rather than left to be rediscovered.
    """

    def test_a_state_without_a_frequency_axis_is_refused(self):
        operator = IonosphereOperator(delta=jnp.array(1e-3), ref_freq=70e6)
        state = State(
            data=jnp.zeros((4, 6)),
            coords=Coordinates(time=jnp.arange(4.0), freq=None),
        )
        with pytest.raises(StateValidationError, match="requires state.coords.freq"):
            operator(state)

    def test_a_state_with_only_a_frequency_axis_is_enough(self):
        """The distinguishing branch: it does NOT need ``time``.

        If this operator ever acquires the nine-operator sentence, this test
        fails and the family test starts covering it -- which is the outcome
        we want, arrived at loudly.
        """
        operator = IonosphereOperator(delta=jnp.array(1e-3), ref_freq=70e6)
        state = State(
            data=jnp.ones((4, 6)),
            coords=Coordinates(time=None, freq=jnp.linspace(60e6, 80e6, 6)),
        )
        out = operator(state)
        assert out.data.shape == (4, 6)
        # ... and it did something: the factor is chromatic, so no two columns
        # of a constant input come back equal.
        assert len(set(float(v) for v in out.data[0])) == 6


class TestDriftScanHorizonFractionVersionGate:
    """A dependency-version gate, which no test run can reach by installing things.

    ``horizon_fraction()`` needs ``limTOD >= 1.9``. The suite runs against one
    installed version, so one side of this branch is unreachable by
    construction -- but only one, and which one depends on the machine. Faking
    the module attribute is the only way to test the refusal without pinning
    the suite to an old dependency, and it is worth testing because a version
    gate that has never fired is a message nobody has read.
    """

    def test_an_old_limtod_is_refused_by_name(self, monkeypatch):
        from rheplicant.radio.sky import driftscan

        class OldLimtod:
            __version__ = "1.8.3"
            # deliberately no horizon_beam_fraction

        monkeypatch.setattr(driftscan, "_limtod_jax", lambda *a, **k: OldLimtod())

        projector = _drift_scan_projector()
        with pytest.raises(StateValidationError, match=r"needs limTOD >= 1\.9"):
            projector.horizon_fraction()

    def test_the_message_quotes_the_version_it_found(self, monkeypatch):
        """Because "needs >= 1.9" without saying what is installed sends the
        reader to check by hand, which is the whole cost of a bad message."""
        from rheplicant.radio.sky import driftscan

        class OldLimtod:
            __version__ = "1.8.3"

        monkeypatch.setattr(driftscan, "_limtod_jax", lambda *a, **k: OldLimtod())
        with pytest.raises(StateValidationError, match="1.8.3"):
            _drift_scan_projector().horizon_fraction()


#: Smallest well-formed projector geometry. Every angle distinct, so a
#: constructor that transposed two of them would not go unnoticed if this
#: fixture is ever reused for something that reads them.
_LMAX, _NSIDE = 3, 2
_N_ALM = (_LMAX + 1) * (_LMAX + 2) // 2


def _drift_scan_projector():
    """Smallest projector that reaches ``horizon_fraction``'s version gate."""
    from rheplicant.radio import DriftScanProjector

    return DriftScanProjector(
        beam_alms=jnp.zeros((2, _N_ALM), dtype=jnp.complex64),
        lat_deg=53.2,
        az_deg=30.0,
        el_deg=50.0,
        lmax=_LMAX,
        nside=_NSIDE,
    )


def test_the_suite_still_imports_the_radio_package_cleanly():
    """``_validate_registrations`` runs at import; monkeypatching it above must
    not have left the package's ``__all__`` mutated for anyone else."""
    import importlib

    import rheplicant.radio as radio

    importlib.reload(radio)
    assert "Impostor" not in radio.__all__
    assert jax is not None  # the import above is what is under test
