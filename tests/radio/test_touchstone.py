"""The Touchstone adapter's own surface.

What an S-parameter *means* is rhino_cal_jax's subject. What belongs here is
the seam: that a Touchstone file's bytes become the complex array the rest of
the package expects, in Hz, with every way of getting that wrong named rather
than absorbed.
"""

import numpy as np
import pytest

from rheplicant.core.errors import DataIngestionError

# _interp_strict is imported directly, not only exercised through
# interpolate_onto, for the tests below that need a state interpolate_onto's
# public surface cannot express at all: an x axis at unix-epoch magnitude,
# rather than frequency. It is *not* because Touchstone's other branches
# (NaN, non-ascending, empty) are unreachable through interpolate_onto --
# Touchstone is a frozen dataclass with no __post_init__, so a hand-built one
# reaches those through interpolate_onto too, and two tests below do exactly
# that. The reason to prefer the direct route where either is possible is
# that it isolates _interp_strict's own contract from Touchstone's
# *incidental* lack of construction-time validation: if Touchstone ever grows
# a __post_init__ that enforces its own invariants, a test that only reaches
# a branch by constructing an invalid Touchstone would silently stop
# covering it, while a direct call to _interp_strict would not.
from rheplicant.radio.touchstone import (
    Touchstone,
    _interp_strict,
    interpolate_onto,
    read_touchstone,
)

RI_2PORT = """\
! a two-port, real/imaginary
# MHZ S RI R 50
60.0   0.10  0.20   0.30  0.40   0.50  0.60   0.70  0.80
70.0   0.11  0.21   0.31  0.41   0.51  0.61   0.71  0.81
"""


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return path


def test_ri_two_port_parses_to_hz_and_touchstone_column_order(tmp_path):
    ts = read_touchstone(write(tmp_path, "cal.s2p", RI_2PORT))

    assert isinstance(ts, Touchstone)
    np.testing.assert_allclose(ts.freq_hz, [60e6, 70e6])
    assert ts.s.shape == (2, 2, 2)
    # Touchstone 2-port column order is S11 S21 S12 S22 -- the SECOND pair is
    # S21. A row-major reading transposes the off-diagonal silently.
    np.testing.assert_allclose(ts.s11, [0.10 + 0.20j, 0.11 + 0.21j])
    np.testing.assert_allclose(ts.s21, [0.30 + 0.40j, 0.31 + 0.41j])
    np.testing.assert_allclose(ts.s12, [0.50 + 0.60j, 0.51 + 0.61j])
    np.testing.assert_allclose(ts.s22, [0.70 + 0.80j, 0.71 + 0.81j])
    assert ts.z0 == 50.0
    assert ts.n_port == 2


def test_option_line_ending_in_r_with_no_value_raises(tmp_path):
    # "R" as the option line's last token used to index one past the end of
    # `parts` and leak a bare IndexError, with no path/line context.
    path = write(tmp_path, "bad_r.s1p", "# MHZ S RI R\n")
    with pytest.raises(DataIngestionError, match="impedance") as excinfo:
        read_touchstone(path)
    assert f"{path}:1" in str(excinfo.value)


def test_option_line_non_numeric_impedance_raises(tmp_path):
    # A non-numeric value after "R" used to leak a bare ValueError from
    # float(), again with no path/line context.
    path = write(tmp_path, "bad_r2.s1p", "# MHZ S RI R FOO\n")
    with pytest.raises(DataIngestionError, match="not a number") as excinfo:
        read_touchstone(path)
    assert f"{path}:1" in str(excinfo.value)


def test_option_line_missing_frequency_unit_raises_rather_than_defaulting_to_ghz(tmp_path):
    # Touchstone v1 defaults an omitted unit to GHz. Applying that default
    # unstated would silently rescale the whole sweep by 1e9; this reader
    # raises instead of guessing.
    path = write(tmp_path, "no_unit.s1p", "# S RI R 50\n")
    with pytest.raises(DataIngestionError, match="GHz") as excinfo:
        read_touchstone(path)
    assert f"{path}:1" in str(excinfo.value)


def test_second_option_line_raises(tmp_path):
    # A second '#' line used to silently re-specify the unit/format/impedance
    # for the rows that follow it, leaving one Touchstone.s array built from
    # two different interpretations.
    text = "# MHZ S RI R 50\n60.0  0.1 0.2\n# MHZ S RI R 50\n70.0  0.3 0.4\n"
    path = write(tmp_path, "two_opts.s1p", text)
    with pytest.raises(DataIngestionError, match="second") as excinfo:
        read_touchstone(path)
    assert f"{path}:3" in str(excinfo.value)


def test_option_line_non_s_parameter_type_raises(tmp_path):
    # The network-parameter type token was never checked, so a Y-parameter
    # file would parse and land in Touchstone.s mislabelled as S-parameters.
    path = write(tmp_path, "yparam.s1p", "# MHZ Y RI R 50\n")
    with pytest.raises(DataIngestionError, match="S-parameters") as excinfo:
        read_touchstone(path)
    assert "'Y'" in str(excinfo.value)


def test_ri_ma_and_db_render_the_same_physical_s(tmp_path):
    # 37 degrees, deliberately off the 0/45/90/... grid: real and imag stay
    # unequal and nonzero, so an RI real/imag swap or a degree-vs-radian slip
    # in MA/DB cannot hide behind a coincidental symmetry.
    amplitude, angle_deg = 0.5, 37.0
    z = amplitude * np.exp(1j * np.deg2rad(angle_deg))
    # float(...) before !r: with numpy>=2.0, repr(np.float64(x)) is the
    # unambiguous "np.float64(x)" (NEP 51), which float() cannot parse back
    # out of the generated file text. Converting to a plain Python float first
    # is lossless (same IEEE-754 double) and only changes the string written,
    # not the value.
    ri = f"# HZ S RI R 50\n1.0e8   {float(z.real)!r}  {float(z.imag)!r}\n"
    ma = f"# HZ S MA R 50\n1.0e8   {amplitude!r}  {angle_deg!r}\n"
    db = f"# HZ S DB R 50\n1.0e8   {float(20 * np.log10(amplitude))!r}  {angle_deg!r}\n"

    got = [
        (tag, read_touchstone(write(tmp_path, f"one_{tag}.s1p", text)).s11[0])
        for tag, text in (("ri", ri), ("ma", ma), ("db", db))
    ]
    for tag, value in got:
        assert value == pytest.approx(z, rel=1e-12), f"{tag} format disagreed with RI"


def test_a_short_row_raises_instead_of_vanishing(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.1 0.2  0.3 0.4  0.5 0.6  0.7 0.8\n70.0  0.1 0.2\n"
    with pytest.raises(DataIngestionError, match="columns"):
        read_touchstone(write(tmp_path, "short.s2p", text))


def test_data_before_the_option_line_raises(tmp_path):
    with pytest.raises(DataIngestionError, match="option line"):
        read_touchstone(write(tmp_path, "no_opt.s1p", "60.0  0.1  0.2\n"))


def test_non_ascending_frequencies_raise(tmp_path):
    text = "# MHZ S RI R 50\n70.0  0.1 0.2\n60.0  0.3 0.4\n"
    with pytest.raises(DataIngestionError, match="ascending"):
        read_touchstone(write(tmp_path, "back.s1p", text))


def test_suffix_disagreeing_with_the_port_count_raises(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.1 0.2\n"
    with pytest.raises(DataIngestionError, match="name says"):
        read_touchstone(write(tmp_path, "mislabelled.s2p", text))


def test_comments_and_blank_lines_do_not_change_the_result(tmp_path):
    noisy = (
        "! leading comment\n"
        "\n"
        "# MHZ S RI R 50\n"
        "!\n"
        "60.0  0.1 0.2   ! trailing comment on a real row\n"
        "   \n"
        "70.0  0.3 0.4\n"
    )
    clean = "# MHZ S RI R 50\n60.0  0.1 0.2\n70.0  0.3 0.4\n"
    a = read_touchstone(write(tmp_path, "noisy.s1p", noisy))
    b = read_touchstone(write(tmp_path, "clean.s1p", clean))
    np.testing.assert_allclose(a.freq_hz, b.freq_hz)
    np.testing.assert_allclose(a.s11, b.s11)


def test_s1p_has_no_transmission_term(tmp_path):
    ts = read_touchstone(write(tmp_path, "one.s1p", "# MHZ S RI R 50\n60.0  0.1 0.2\n"))
    with pytest.raises(DataIngestionError, match="1-port"):
        _ = ts.s21  # bare `ts.s21` trips ruff B018 ("useless expression"); the
        # property access itself is what raises, so the value is discarded on
        # purpose -- assigning to `_` says so instead of looking like a typo.


def test_interpolation_lands_on_the_target_grid(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.0 0.0\n80.0  2.0 4.0\n"
    ts = read_touchstone(write(tmp_path, "ramp.s1p", text))
    got = interpolate_onto(np.array([60e6, 70e6, 80e6]), ts)
    np.testing.assert_allclose(got, [0.0 + 0.0j, 1.0 + 2.0j, 2.0 + 4.0j])


def test_extrapolation_is_refused_by_default_and_names_both_bands(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.0 0.0\n80.0  2.0 4.0\n"
    ts = read_touchstone(write(tmp_path, "ramp.s1p", text))
    with pytest.raises(DataIngestionError, match="outside"):
        interpolate_onto(np.array([50e6, 70e6]), ts)


def test_extrapolation_clamps_when_explicitly_allowed(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.0 0.0\n80.0  2.0 4.0\n"
    ts = read_touchstone(write(tmp_path, "ramp.s1p", text))
    got = interpolate_onto(np.array([50e6]), ts, allow_extrapolation=True)
    np.testing.assert_allclose(got, [0.0 + 0.0j])


def test_an_unknown_component_raises(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.1 0.2\n70.0  0.3 0.4\n"
    ts = read_touchstone(write(tmp_path, "one.s1p", text))
    with pytest.raises(DataIngestionError, match="component"):
        interpolate_onto(np.array([65e6]), ts, component="s99")


def test_interpolate_onto_a_1port_component_defers_to_touchstones_own_error(tmp_path):
    # component="s12" is a recognised S-parameter name -- interpolate_onto's own
    # check is only about the string. Whether this particular source actually
    # carries it is Touchstone.s12's job, and its error already names the
    # reason ("1-port"); duplicating that check here would just be a second
    # place for the message to drift out of sync with the first.
    text = "# MHZ S RI R 50\n60.0  0.1 0.2\n70.0  0.3 0.4\n"
    ts = read_touchstone(write(tmp_path, "one.s1p", text))
    with pytest.raises(DataIngestionError, match="1-port"):
        interpolate_onto(np.array([65e6]), ts, component="s12")


def test_interpolate_onto_rejects_a_non_ascending_source_axis():
    # Touchstone.freq_hz is only guaranteed ascending by read_touchstone's own
    # check (_check_strictly_ascending) -- the dataclass itself enforces
    # nothing. _interp_strict is shared with a caller (the thermistor time
    # axis interpolated in Task 9) that has no such upstream guarantee at all,
    # so it re-checks ascending order itself rather than trusting the caller;
    # np.interp assumes ascending order and does not check it.
    ts = Touchstone(
        freq_hz=np.array([80e6, 60e6]),
        s=np.array([[[1 + 0j]], [[2 + 0j]]]),
        z0=50.0,
    )
    with pytest.raises(DataIngestionError, match="ascending") as excinfo:
        interpolate_onto(np.array([70e6]), ts)
    # Mirrors _check_strictly_ascending's own "first offender" naming (Minor
    # 2 of the touchstone.py review): index 1 (0-based) is freq_hz[1] = 60e6,
    # the later element of the one disordered pair.
    assert "index 1" in str(excinfo.value)


def test_interpolate_onto_rejects_an_empty_source_axis():
    # x[0] on an empty array raises a bare IndexError with no path/context.
    # read_touchstone can never produce an empty freq_hz (it raises "no data
    # rows" first), but _interp_strict is a general helper that does not get
    # to assume its caller already checked that -- so it checks it directly.
    ts = Touchstone(freq_hz=np.array([]), s=np.zeros((0, 1, 1), dtype=complex), z0=50.0)
    with pytest.raises(DataIngestionError, match="empty"):
        interpolate_onto(np.array([70e6]), ts)


def test_interpolate_onto_an_empty_target_grid_does_not_crash(tmp_path):
    # x_new.min()/.max() raise a bare ValueError ("zero-size array to
    # reduction operation...") on an empty target grid, with no context. An
    # empty target is vacuously inside any range, so there is nothing to
    # refuse -- this should return an empty result, not crash.
    text = "# MHZ S RI R 50\n60.0  0.0 0.0\n80.0  2.0 4.0\n"
    ts = read_touchstone(write(tmp_path, "ramp.s1p", text))
    got = interpolate_onto(np.array([]), ts)
    assert got.shape == (0,)


def test_the_public_names_are_reachable_from_the_subpackage():
    from rheplicant import radio
    from rheplicant.radio import Touchstone as T
    from rheplicant.radio import interpolate_onto as i
    from rheplicant.radio import read_touchstone as r

    assert (T, r, i) == (Touchstone, read_touchstone, interpolate_onto)
    # Reachability alone is not enough: `from pkg import name` resolves through
    # the module namespace, not through __all__, so deleting the `__all__ +=`
    # block leaves the three asserts above passing. __all__ is what
    # _validate_registrations() iterates at import time and what `import *`
    # honours -- and this file's own convention puts the import line and the
    # `__all__ +=` block in two separate places, which is what invites the
    # omission in the first place.
    assert {"Touchstone", "interpolate_onto", "read_touchstone"}.issubset(radio.__all__)


@pytest.mark.parametrize(
    ("freqs", "expected_row"),
    [
        (["nan", "70.0", "80.0"], 1),
        (["60.0", "nan", "80.0"], 2),
        (["60.0", "70.0", "nan"], 3),
    ],
    ids=["leading", "middle", "trailing"],
)
def test_a_nan_frequency_names_its_own_row_not_the_next_one(tmp_path, freqs, expected_row):
    # A NaN frequency spreads into np.diff's cells on both sides of it, and
    # the ordering check's "first offender" arithmetic always blames the
    # later element of a diff pair -- right for genuine disorder, wrong for a
    # leading NaN, which has no earlier row to blame. All three positions are
    # pinned so that misattribution cannot come back unnoticed.
    text = "# MHZ S RI R 50\n" + "".join(f"{f}  0.1 0.2\n" for f in freqs)
    with pytest.raises(DataIngestionError, match="NaN") as excinfo:
        read_touchstone(write(tmp_path, "nan_freq.s1p", text))
    assert f"data row {expected_row}" in str(excinfo.value)


@pytest.mark.parametrize(
    "name",
    ["no_suffix", "data.txt", "data.s3p", "foo.s2p.bak"],
)
def test_an_unrecognised_suffix_passes_through_unchecked(tmp_path, name):
    # None of these are ".s1p"/".s2p" as pathlib's `.suffix` sees them: no
    # extension at all, an unrelated one, a Touchstone-shaped extension this
    # reader does not support at any port count (3-port), and ".s2p" that is
    # not the *final* suffix. The suffix check has no second opinion to offer
    # in any of these cases and must not invent one.
    text = "# MHZ S RI R 50\n60.0  0.1 0.2\n"
    ts = read_touchstone(write(tmp_path, name, text))
    assert ts.n_port == 1


def test_suffix_check_case_insensitively_accepts_a_match(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.1 0.2  0.3 0.4  0.5 0.6  0.7 0.8\n"
    ts = read_touchstone(write(tmp_path, "cal.S2P", text))
    assert ts.n_port == 2


def test_suffix_check_case_insensitively_still_catches_a_mismatch(tmp_path):
    text = "# MHZ S RI R 50\n60.0  0.1 0.2  0.3 0.4  0.5 0.6  0.7 0.8\n"
    with pytest.raises(DataIngestionError, match="name says"):
        read_touchstone(write(tmp_path, "cal.S1P", text))


def test_interp_strict_names_a_nan_in_the_source_axis_rather_than_calling_it_unsorted():
    # Mirrors _check_strictly_ascending's NaN-first structure: NaN compares
    # False to everything, including "<= 0", so a bare
    # `np.any(np.diff(x) <= 0)` check does not see a NaN axis as disordered
    # at all -- np.interp would return NaN silently rather than this
    # function raising. NaN gets its own check and its own message ahead of
    # the ordering check, exactly as the file-reading sibling check does.
    x = np.array([0.0, 1.0, np.nan, 3.0])
    y = np.array([0.0, 1.0, 2.0, 3.0])
    with pytest.raises(DataIngestionError, match="NaN"):
        _interp_strict(np.array([0.5]), x, y, what="nan axis test")


def test_frequency_scale_extrapolation_is_still_refused_under_the_new_tolerance(tmp_path):
    # Regime check alongside the unix-epoch regression below: the combined
    # span + ULP tolerance must not accidentally loosen the ordinary
    # megahertz-scale case the old span-dominated term already covered.
    text = "# MHZ S RI R 50\n60.0  0.0 0.0\n80.0  2.0 4.0\n"
    ts = read_touchstone(write(tmp_path, "ramp.s1p", text))
    with pytest.raises(DataIngestionError, match="outside") as excinfo:
        interpolate_onto(np.array([80e6 + 1e4]), ts)  # 10 kHz past the edge
    # The message states the overshoot itself (here 1e4 Hz, verified
    # numerically before writing this test), not just the word "outside" --
    # see the epoch-scale test below for why that distinction matters.
    assert "10000" in str(excinfo.value)


def test_a_target_past_a_short_unix_epoch_window_is_still_refused():
    # The regression the old tolerance missed: on a unix-epoch axis
    # (~1.75e9 s) `1e-9 * max(span, abs(hi))` is dominated by abs(hi)
    # whenever the span is short, giving ~1.75 s of slack *regardless of
    # window length* -- so a target 1.5 s past a 1 s-cadence, 11 s-wide
    # window was silently accepted and clamped. The new tolerance here is
    # ~1e-6 s (verified numerically before writing this test), far below 1.5 s.
    epoch = 1_750_000_000.0
    time_s = epoch + np.arange(12.0)  # 12 samples, 1 s cadence
    values = np.arange(12.0)
    target = np.array([time_s[-1] + 1.5])
    with pytest.raises(DataIngestionError, match="outside") as excinfo:
        _interp_strict(target, time_s, values, what="thermistor time axis")
    message = str(excinfo.value)
    # %.6g on the four raw bounds alone renders all of them as "1.75e+09" at
    # this magnitude -- six significant figures cannot resolve a 1.5 s gap
    # against a 1.7e9 base -- so the message must state the shortfall itself
    # (verified numerically before writing this assertion: exactly 1.5,
    # bit-exact, not just approximately), not merely the word "outside".
    assert "1.5" in message
    # And for a mismatch this small (a log a couple of samples short), the
    # remedy is not "pass allow_extrapolation=True" -- that clamps and hides
    # the very problem being reported. The message must name a real fix.
    assert "trim" in message


def test_a_target_one_ulp_past_a_short_unix_epoch_window_still_passes():
    # The fix must not overcorrect: a target that agrees with the sampled
    # range up to float64 roundoff (one ULP past the last sample) must still
    # be accepted and clamped, not rejected as "outside". Dropping the ULP
    # term and keeping only 1e-9 * span would fail this, since float64
    # spacing at epoch magnitude (~2.4e-7 s) exceeds 1e-9 * an 11 s span
    # (~1.1e-8 s).
    epoch = 1_750_000_000.0
    time_s = epoch + np.arange(12.0)
    values = np.arange(12.0)
    target = np.array([time_s[-1] + np.spacing(time_s[-1])])
    got = _interp_strict(target, time_s, values, what="thermistor time axis")
    np.testing.assert_allclose(got, [11.0])


def test_a_single_row_touchstone_still_only_tolerates_a_few_ulps(tmp_path):
    # A single-row Touchstone file is a legitimate, if unusual, input --
    # _check_strictly_ascending does not reject it (np.diff on a length-1
    # array is trivially empty), so this state, unlike the epoch-scale ones
    # above, IS reachable through the public surface without hand-building a
    # Touchstone: goes through read_touchstone + interpolate_onto rather than
    # calling _interp_strict directly (see the note by the import above).
    # Single-sample x means hi == lo, so the span term of tol is zero and the
    # tolerance comes entirely from the ULP term -- confirms the
    # "effectively exact match required" behaviour a single-sample axis
    # relies on (there is nothing to interpolate between, only a point to
    # match) survives the tolerance formula change; verified numerically
    # before writing this test (tol ~= 3e-8 Hz at 60 MHz).
    text = "# MHZ S RI R 50\n60.0  42.0 0.0\n"
    ts = read_touchstone(write(tmp_path, "single.s1p", text))
    got = interpolate_onto(np.array([60e6]), ts)
    np.testing.assert_allclose(got, [42.0 + 0.0j])
    with pytest.raises(DataIngestionError, match="outside"):
        interpolate_onto(np.array([60e6 + 1.0]), ts)  # 1 Hz past the edge


def test_interp_strict_names_a_length_mismatch_between_x_and_y():
    # np.interp's own error here is unbranded and context-free: "fp and xp
    # are not of the same length." No `what`, no indication of which call
    # site or which arrays were involved -- the exact failure class this
    # module's docstring says it exists to eliminate. No current caller can
    # trigger this (Touchstone.freq_hz and each sXX are co-indexed by
    # construction), but Task 9 calls this with x and y read from two
    # different HDF5 datasets, where a partial write or schema drift makes a
    # mismatch an ordinary ingestion bug rather than a programming error.
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    with pytest.raises(DataIngestionError, match="length") as excinfo:
        _interp_strict(np.array([0.5]), x, y, what="mismatched-length test")
    assert "mismatched-length test" in str(excinfo.value)


def test_interp_strict_handles_real_valued_y_not_just_complex():
    # interpolate_onto only ever hands _interp_strict complex S-parameters,
    # so the real-valued branch (np.iscomplexobj(y) is False) has no coverage
    # through the public API -- yet it is exactly Task 9's thermistor path
    # (temperatures are real), the one branch the general-helper contract
    # exists to serve.
    x = np.array([0.0, 1.0, 2.0])
    y = np.array([10.0, 20.0, 30.0])
    got = _interp_strict(np.array([0.5, 1.5]), x, y, what="real-valued test")
    assert not np.iscomplexobj(got)
    np.testing.assert_allclose(got, [15.0, 25.0])


class TestTheOptionLineAndRowShapeGuards:
    """Three refusals that fire, are well phrased, and were asserted nowhere.

    Named in an audit of ``raise`` statements the full suite never executes:
    of 375 in ``src/``, these are ingestion-layer guards a user meets by
    handing the reader a real file from a real instrument, which is the
    likeliest place in the package to meet one at all.

    What being unpinned costs is not that the guard breaks -- it is that a
    refactor downgrades it without anything noticing. Measured by neutralising
    each guard on a copy of the tree, and the two results differ in kind:

    * the row-shape check degrades to ``ValueError: zip() argument 2 is
      shorter than argument 1``, which still raises but names nothing a user
      could act on;
    * the option-line format check degrades to **no error at all**. The file
      parses, and a row ``0.1 0.2`` that meant nothing in particular comes
      back as ``s11 = 1.0116+0.0035j`` -- a fall-through to the DB reading of
      numbers whose format was never stated. Under RI the same row is
      ``0.1+0.2j``. An order of magnitude wrong, finite, correctly shaped, and
      silent.

    So a test asserting only ``pytest.raises(DataIngestionError)`` would have
    kept passing through the first degradation and could not have existed for
    the second. Each test below pins the sentence, and the last pins that the
    three sentences differ from one another.
    """

    def test_an_option_line_with_no_data_format_is_refused(self, tmp_path):
        """``# MHZ S R 50`` -- a unit and a parameter type, no RI/MA/DB.

        Distinct from ``test_data_before_the_option_line_raises``, which has
        no option line at all: here the line is present and parses, and it is
        the *format* token that is missing. The two share the substring
        "option line", which is why this asserts the specific sentence.
        """
        with pytest.raises(DataIngestionError, match="names no data format"):
            read_touchstone(write(tmp_path, "nofmt.s1p", "# MHZ S R 50\n60.0  0.1 0.2\n"))

    def test_a_first_data_row_with_an_impossible_column_count_is_refused(self, tmp_path):
        """The ``if`` branch above the one the suite already pins.

        ``test_a_short_row_raises_instead_of_vanishing`` covers the ``elif``:
        a LATER row disagreeing with the first. The branch above it -- the
        FIRST row having a count that is neither 3 nor 9 -- had no test, and
        it is the one that decides ``n_column`` for the whole file.

        Four columns, deliberately: it is one more than a legal 1-port row, so
        a reader that silently truncated to 3 would produce a perfectly valid
        Touchstone from a corrupt file.
        """
        with pytest.raises(DataIngestionError, match=r"a Touchstone data row has 3"):
            read_touchstone(
                write(tmp_path, "wide.s1p", "# MHZ S RI R 50\n60.0  0.1 0.2 0.3\n")
            )

    def test_a_file_with_no_data_rows_at_all_is_refused(self, tmp_path):
        """Header and comments, no numbers.

        The empty-file case that would otherwise reach ``_build_touchstone``
        with ``n_column=None``, whose parameter is typed ``int``. Comments and
        a valid option line are included on purpose: a guard keyed on file
        length rather than on rows parsed would pass this.
        """
        with pytest.raises(DataIngestionError, match="no data rows"):
            read_touchstone(
                write(tmp_path, "empty.s1p", "! a comment\n# MHZ S RI R 50\n! another\n")
            )

    def test_the_three_refusals_are_distinguishable_from_each_other(self, tmp_path):
        """Each names its own cause, so a user is told which of the three it is.

        Without this, three tests each matching a substring could all be
        satisfied by one over-broad message -- and the file would read as
        thoroughly covered while telling a user nothing.
        """
        messages = []
        for name, text in [
            ("nofmt.s1p", "# MHZ S R 50\n60.0  0.1 0.2\n"),
            ("wide.s1p", "# MHZ S RI R 50\n60.0  0.1 0.2 0.3\n"),
            ("empty.s1p", "! c\n# MHZ S RI R 50\n"),
        ]:
            with pytest.raises(DataIngestionError) as excinfo:
                read_touchstone(write(tmp_path, name, text))
            messages.append(str(excinfo.value))
        assert len(set(messages)) == 3, messages
