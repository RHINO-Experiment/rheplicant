"""Agreement with rhino-cal's numpy readers, where both can read the same file.

Skipped unless the rhino-cal checkout is importable. Nothing here re-tests
rheplicant's own rejections -- those live in test_touchstone.py and
test_rhino.py. What this file establishes is that where the reference produces
an answer, so do we, and it is the same one.

Every fixture below is deliberately valid under *both* implementations, which
is a real constraint: this port is strictly the stricter of the two. It
requires a declared frequency unit and checks it, refuses Touchstone's unstated
GHz default, rejects a second option line, non-S parameter types, non-ascending
or non-finite frequencies and times, shape mismatches and non-finite thermistor
readings, and refuses to extrapolate where ``np.interp`` would clamp. On every
one of those the reference happily returns a number. That set is the
behavioural delta the port introduced; it is documented in the plan and
exercised by the two test files named above, and it is *not* something this
file can assert on, because agreement is only defined where both answer.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

RHINO_CAL = Path("~/projects/rhino-cal").expanduser()
if RHINO_CAL.is_dir() and str(RHINO_CAL) not in sys.path:
    sys.path.insert(0, str(RHINO_CAL))

reference = pytest.importorskip("utils.utils", reason=f"rhino-cal not present at {RHINO_CAL}")
h5py = pytest.importorskip("h5py", reason="h5py comes with rheplicant[rhino]")

from rheplicant.radio.rhino import read_rhino_observation  # noqa: E402
from rheplicant.radio.touchstone import interpolate_onto, read_touchstone  # noqa: E402


def _import_data_handler():
    """rhino-cal's ``DataHandler``, the reference HDF5 reader.

    Its module reaches ``gcr.data_processing -> gcr.data_and_noise_covariance
    -> rfi_flagging.rfi_flagging -> MomentRFI.MomentRFI``, and that last hop
    is a module path MomentRFI has since renamed: the installed package
    exports ``IterativeSurfaceFitter`` from its top level, not from a
    same-named submodule. Nothing on ``DataHandler.__init__``'s path -- the
    only thing this file calls -- touches MomentRFI at all; the dependency is
    a transitive import and no more. So when that is the *only* thing in the
    way, the renamed path is bridged to the genuine installed package for the
    duration of the import and then removed again, rather than losing the
    single most valuable comparison in this file to an unrelated packaging
    drift. No behaviour is stubbed: the aliased module is the real one. Any
    other import failure skips.
    """
    try:
        from gcr.data_processing import DataHandler

        return DataHandler
    except ModuleNotFoundError as first:
        if first.name != "MomentRFI.MomentRFI":
            pytest.skip(f"rhino-cal's gcr.data_processing is not importable: {first}")

    momentrfi = pytest.importorskip("MomentRFI", reason="rhino-cal's gcr imports MomentRFI")
    if not hasattr(momentrfi, "IterativeSurfaceFitter"):
        pytest.skip("the installed MomentRFI exports no IterativeSurfaceFitter to bridge to")

    sys.modules["MomentRFI.MomentRFI"] = momentrfi
    try:
        from gcr.data_processing import DataHandler
    except ImportError as second:
        pytest.skip(f"rhino-cal's gcr.data_processing is not importable: {second}")
    finally:
        del sys.modules["MomentRFI.MomentRFI"]
    return DataHandler


@pytest.fixture(scope="module")
def data_handler():
    """Behind a fixture, not at module scope, so that a failure to import the
    reference *HDF5* reader does not also skip the Touchstone comparisons --
    those need only ``utils.utils``, which has no dependencies beyond numpy
    and astropy."""
    return _import_data_handler()


# The HDF5 fixture is rebuilt here rather than imported from test_rhino.py:
# tests/radio/ has no __init__.py, so there is no package for a relative import
# to resolve against, and adding one to make two test modules share twenty
# lines is the wrong trade.
FREQ_MHZ = np.array([60.0, 70.0, 80.0])
TIME_S = np.arange(0.0, 12.0, 1.0) + 1000.0
SWITCH_TIMES = np.array([1000.0, 1004.0, 1008.0])
SWITCH_STATES = [b"antenna", b"internal_load", b"heated_load"]
#: Column 0 = ambient, column 1 = hot -- the same assignment the reference
#: hard-codes as ambient_load_index=0, heated_load_index=1, so that the two
#: readers are describing the same file rather than two different ones.
COLUMNS = {"antenna": 0, "internal_load": 0, "heated_load": 1}


def make_file(path):
    """A recording both readers accept.

    Three constraints make it readable by *this* package, none of which the
    reference would have enforced, and all of which matter for the comparison
    to be like-for-like: the first sample coincides with the first switch
    transition (so no leading sample is dropped and both time axes have the
    same length), the thermistor log shares the SDR's time axis (so resampling
    it onto that axis is the identity, and the reference's un-resampled
    columns are directly comparable), and the frequencies are in MHz, within
    the reader's plausible band, matching the reference's own ``freq_unit``
    default.
    """
    n_time, n_freq = len(TIME_S), len(FREQ_MHZ)
    temps = np.stack([np.full(n_time, 20.0), np.full(n_time, 100.0)], axis=1)
    with h5py.File(path, "w") as f:
        sdr = f.create_group("sdr")
        sdr.create_dataset("sdr_freqs", data=FREQ_MHZ)
        sdr.create_dataset("sdr_times", data=TIME_S)
        sdr.create_dataset(
            "sdr_waterfall",
            data=np.arange(n_time * n_freq, dtype=float).reshape(n_time, n_freq),
        )
        sw = f.create_group("switches")
        sw.create_dataset("switch_times", data=SWITCH_TIMES)
        sw.create_dataset("switch_states", data=np.array(SWITCH_STATES, dtype="S16"))
        tg = f.create_group("temperatures")
        tg.create_dataset("temperature_times", data=TIME_S)
        tg.create_dataset("temperatures", data=temps)
    return path


#: Asymmetric and sign-varied in all four S-parameters on purpose. A fixture
#: whose off-diagonal terms were equal, or whose diagonal terms were, could not
#: distinguish the Touchstone column order (freq S11 S21 S12 S22 -- the second
#: pair is S21) from the row-major reading it is so easily mistaken for. The
#: distinctness is asserted below rather than left to inspection.
TWO_PORT = """\
# MHZ S RI R 50
60.0   0.10  0.20   0.30  0.40   0.50  0.60   0.70  0.80
70.0   0.11 -0.21   0.31  0.41  -0.51  0.61   0.71  0.81
80.0  -0.12  0.22   0.32 -0.42   0.52  0.62   0.72 -0.82
"""


def write_two_port(tmp_path):
    path = tmp_path / "cal.s2p"
    path.write_text(TWO_PORT)
    return path


def assert_same_number(name, ours, theirs):
    """Exact agreement, not approximate.

    Both readers parse this fixture's decimal literals with ``float()`` and
    scale the frequency by the same power of ten, so every value here is
    reachable bit-for-bit by both. Any drift at all is therefore a finding
    about the port, not float noise to be absorbed by a tolerance -- hence
    rtol=atol=0, which keeps assert_allclose's element-by-element complex-plane
    diff while asserting equality.
    """
    np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), rtol=0, atol=0, err_msg=name)


def assert_pairwise_distinct(**arrays):
    """Guard that a comparison over these arrays could see them swapped.

    Without this, a fixture edit that made (say) s12 and s21 equal would leave
    every assertion in the test passing while the test stopped being able to
    detect a transposed off-diagonal at all.
    """
    items = list(arrays.items())
    for i, (name_a, a) in enumerate(items):
        for name_b, b in items[i + 1 :]:
            assert not np.array_equal(np.asarray(a), np.asarray(b)), (
                f"{name_a} and {name_b} are equal in this fixture, so no assertion over "
                "them can detect the two being swapped"
            )


def test_touchstone_agrees_with_read_s2p(tmp_path):
    path = write_two_port(tmp_path)

    # Note the tuple order: read_s2p returns (s11, s12, s21, s22, freq), which
    # is neither the file's column order nor this package's array order.
    s11, s12, s21, s22, freq = reference.read_s2p(str(path))
    ts = read_touchstone(path)

    assert_pairwise_distinct(s11=s11, s12=s12, s21=s21, s22=s22)

    assert_same_number("freq_hz", ts.freq_hz, freq)
    assert_same_number("s11", ts.s11, s11)
    assert_same_number("s12", ts.s12, s12)
    assert_same_number("s21", ts.s21, s21)
    assert_same_number("s22", ts.s22, s22)


def test_flipped_agrees_with_read_s2p_flipped(tmp_path):
    path = write_two_port(tmp_path)

    # Under flipped_measurement=True the reference returns (s22, s21, s12,
    # s11, freq) -- i.e. positionally the port-reversed equivalents of s11,
    # s12, s21, s22, in that order, since reversing the ports maps the old s21
    # onto the new s12 and vice versa. The names below are therefore the
    # *reversed* network's, not the file's.
    flipped_s11, flipped_s12, flipped_s21, flipped_s22, _ = reference.read_s2p(
        str(path), flipped_measurement=True
    )
    ts = read_touchstone(path, flipped=True)

    assert_pairwise_distinct(
        s11=flipped_s11, s12=flipped_s12, s21=flipped_s21, s22=flipped_s22
    )

    # The reversal actually moved something: without this, both sides agreeing
    # would be consistent with neither having flipped anything.
    unflipped = read_touchstone(path)
    assert_same_number("flipped s11 is the unflipped s22", flipped_s11, unflipped.s22)
    assert_same_number("flipped s12 is the unflipped s21", flipped_s12, unflipped.s21)

    assert_same_number("s11", ts.s11, flipped_s11)
    assert_same_number("s12", ts.s12, flipped_s12)
    assert_same_number("s21", ts.s21, flipped_s21)
    assert_same_number("s22", ts.s22, flipped_s22)


def test_hdf5_waterfall_times_and_frequencies_agree_with_datahandler(tmp_path, data_handler):
    path = make_file(tmp_path / "obs.hd5f")

    ours = read_rhino_observation(
        path, freq_unit="MHz", thermistor_columns=COLUMNS, settle_seconds=0.0
    )
    theirs = data_handler(filepath=str(path), gamma_src_dict={}, gamma_rec=None)

    # The reference carries astropy Quantities and defaults freq_unit to MHz;
    # this reader carries plain numpy and was told MHz. Same declaration, so
    # the Hz values must match exactly.
    assert_same_number("freq_hz", ours.freq_hz, theirs.freqs.to("Hz").value)
    assert_same_number("time_s", ours.time_s, theirs.times.to("s").value)
    assert_same_number("waterfall", ours.waterfall, theirs.waterfall)
    # Nothing was dropped, so the two time axes really are the same samples
    # rather than merely the same length.
    assert ours.n_leading_dropped == 0


def test_hdf5_thermistors_agree_with_datahandler(tmp_path, data_handler):
    """The Celsius->Kelvin conversion and the load-to-column mapping.

    The reference derives the mapping from two magic indices
    (``heated_load_index=1``, ``ambient_load_index=0``, everything that is not
    ``heated_load`` routed to ambient); this reader is told it. For a file
    written in that same column order the two must produce identical arrays --
    which is the whole point of checking, since it is the one case where the
    reference's undeclared convention happens to be right and so the only case
    where the numbers are comparable at all.
    """
    path = make_file(tmp_path / "obs.hd5f")

    ours = read_rhino_observation(
        path, freq_unit="MHz", thermistor_columns=COLUMNS, settle_seconds=0.0
    )
    theirs = data_handler(filepath=str(path), gamma_src_dict={}, gamma_rec=None)

    assert set(ours.thermistor_k) == {str(k) for k in theirs.temperature_dict}
    for label in ours.thermistor_k:
        assert_same_number(
            f"thermistor {label!r}", ours.thermistor_k[label], theirs.temperature_dict[label]
        )
    # Not vacuous: the hot column must differ from the ambient one, or a
    # transposed column map would compare equal.
    assert_pairwise_distinct(
        heated_load=ours.thermistor_k["heated_load"],
        internal_load=ours.thermistor_k["internal_load"],
    )


def test_gamma_interpolated_onto_the_band_agrees_with_datahandler(tmp_path, data_handler):
    """Both readers, composed: a Touchstone file resampled onto the recording.

    This is the pairing the reference actually performs -- ``DataHandler``
    hands a ``.s2p`` path's s11 to ``interp_vals_to_new_freq`` -- so it checks
    the ported ``_interp_strict`` against ``interp_vals_to_new_freq`` as well
    as the two file parsers, over the interior of the sweep where the strict
    version's refusal to extrapolate does not bite.
    """
    obs_path = make_file(tmp_path / "obs.hd5f")
    s2p_path = write_two_port(tmp_path)

    ours = read_rhino_observation(
        obs_path, freq_unit="MHz", thermistor_columns=COLUMNS, settle_seconds=0.0
    )
    gamma = interpolate_onto(ours.freq_hz, read_touchstone(s2p_path), component="s11")

    theirs = data_handler(
        filepath=str(obs_path),
        gamma_src_dict={"antenna": str(s2p_path)},
        gamma_rec=str(s2p_path),
    )

    assert_same_number("gamma_src", gamma, theirs.gamma_src_dict["antenna"])
    assert_same_number("gamma_rec", gamma, theirs.gamma_rec)
