"""The Touchstone adapter's own surface.

What an S-parameter *means* is rhino_cal_jax's subject. What belongs here is
the seam: that a Touchstone file's bytes become the complex array the rest of
the package expects, in Hz, with every way of getting that wrong named rather
than absorbed.
"""

import numpy as np

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
