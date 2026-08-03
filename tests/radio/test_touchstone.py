"""The Touchstone adapter's own surface.

What an S-parameter *means* is rhino_cal_jax's subject. What belongs here is
the seam: that a Touchstone file's bytes become the complex array the rest of
the package expects, in Hz, with every way of getting that wrong named rather
than absorbed.
"""

import numpy as np
import pytest

from rheplicant.core.errors import DataIngestionError
from rheplicant.radio.touchstone import Touchstone, interpolate_onto, read_touchstone

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
    with pytest.raises(DataIngestionError, match="ascending"):
        interpolate_onto(np.array([70e6]), ts)


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
    from rheplicant.radio import Touchstone as T
    from rheplicant.radio import interpolate_onto as i
    from rheplicant.radio import read_touchstone as r

    assert (T, r, i) == (Touchstone, read_touchstone, interpolate_onto)


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
