"""The Touchstone adapter's own surface.

What an S-parameter *means* is rhino_cal_jax's subject. What belongs here is
the seam: that a Touchstone file's bytes become the complex array the rest of
the package expects, in Hz, with every way of getting that wrong named rather
than absorbed.
"""

import numpy as np
import pytest

from rheplicant.core.errors import DataIngestionError
from rheplicant.radio.touchstone import Touchstone, read_touchstone

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
    z = 0.5 * np.exp(1j * np.deg2rad(37.0))
    # float(...) before !r: with numpy>=2.0, repr(np.float64(x)) is the
    # unambiguous "np.float64(x)" (NEP 51), which float() cannot parse back
    # out of the generated file text. Converting to a plain Python float first
    # is lossless (same IEEE-754 double) and only changes the string written,
    # not the value.
    ri = f"# HZ S RI R 50\n1.0e8   {float(z.real)!r}  {float(z.imag)!r}\n"
    ma = "# HZ S MA R 50\n1.0e8   0.5  37.0\n"
    db = f"# HZ S DB R 50\n1.0e8   {float(20 * np.log10(0.5))!r}  37.0\n"

    got = [
        read_touchstone(write(tmp_path, f"one_{tag}.s1p", text)).s11[0]
        for tag, text in (("ri", ri), ("ma", ma), ("db", db))
    ]
    for value in got:
        assert value == pytest.approx(z, rel=1e-12)
