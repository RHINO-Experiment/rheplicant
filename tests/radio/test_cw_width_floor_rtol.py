"""The width floor's slack against the rounding of the channel grid (A5-5).

``CWCalibrationOperator`` refuses a ``line_width`` below
``MIN_WIDTH_IN_CHANNELS[lineshape]`` channels, measured as the median gap of
the stored grid. The slack on that comparison was a fixed ``1e-5``, tighter
than a float32 grid resolves its own spacing: on ``jnp.linspace(60e6, 85e6, N)``
the median gap differs from ``25e6 / (N - 1)`` by up to 7.9e-5 of a channel for
N <= 599 and 1.1e-3 by N = 8192, because every stored channel is rounded to an
8 Hz grid at 85 MHz. Measured before the fix, for N in 3..599: the ideal
one-channel width refused for 146 of 597 N, ``freq[1] - freq[0]`` for 120, and
the RHINO channel width 61035.15625 Hz on every float32 60-85 MHz grid; float64
refused none.

The slack is now ``width_floor_rtol(freq, spacing) = max(1e-5,
4 eps max|freq| / spacing)``. Measured over N in 3..8192, the ideal width sits
at most 0.35 ``eps max|freq|`` below the median gap and ``freq[1] - freq[0]``
at most 0.79, so the factor 4 has a margin of five.

The check is called as ``_validate_over_the_run(freq, time)`` with numpy arrays
so the float64 half runs in this float32 session: the method reads the dtype
of what it is handed.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from rheplicant import Coordinates, State
from rheplicant.core.errors import StateValidationError
from rheplicant.radio import CWCalibrationOperator
from rheplicant.radio.instrument.calibration import (
    WIDTH_FLOOR_RTOL,
    WIDTH_FLOOR_RTOL_MAX,
    WIDTH_FLOOR_ULPS,
    width_floor_rtol,
)

LOW, HIGH = 60e6, 85e6
TIME = np.arange(4.0)
#: Sampled N: both ends, the N the audit found refused (84, 100, 599), the
#: float32 cut for a 1e-3 margin (617 / 618), the worst measured deviations
#: (3096, 5153), and powers of two with their neighbours.
SAMPLED_N = [3, 4, 5, 84, 100, 257, 599, 617, 618, 1000, 3096, 4097, 5153, 8191, 8192]
DTYPES = [pytest.param(np.float32, id="float32"), pytest.param(np.float64, id="float64")]
#: The RHINO channel width, 250 MHz / 4096, exactly representable.
RHINO_WIDTH = 250e6 / 4096


def _grid(n, dtype):
    if dtype is np.float32:
        return np.asarray(jnp.linspace(LOW, HIGH, n))  # the production path
    return np.linspace(LOW, HIGH, n, dtype=np.float64)


def _spacing(freq):
    return float(np.median(np.abs(np.diff(freq))))


def _accepts(freq, width, lineshape="sinc2"):
    op = CWCalibrationOperator(
        amplitude=1.0, tone_freq=float(freq[len(freq) // 2]), line_width=width,
        lineshape=lineshape,
    )
    try:
        op._validate_over_the_run(freq, TIME)
    except StateValidationError as error:
        assert "narrower than the channel" in str(error)
        return False
    return True


def test_the_grids_have_the_dtypes_the_test_claims():
    assert _grid(8, np.float32).dtype == np.float32
    assert _grid(8, np.float64).dtype == np.float64


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n", SAMPLED_N)
class TestEachSampledGrid:
    def test_the_ideal_width_is_accepted(self, n, dtype):
        assert _accepts(_grid(n, dtype), (HIGH - LOW) / (n - 1))

    def test_the_first_gap_is_accepted(self, n, dtype):
        """The width the docstring calls the natural spelling of one channel."""
        freq = _grid(n, dtype)
        assert _accepts(freq, float(freq[1] - freq[0]))

    def test_the_gaussian_floor_at_a_quarter_channel_is_accepted(self, n, dtype):
        assert _accepts(_grid(n, dtype), 0.25 * (HIGH - LOW) / (n - 1), "gaussian")

    def test_a_width_just_inside_the_slack_is_accepted(self, n, dtype):
        """The slack is read, not dropped: half of it below the floor passes."""
        freq = _grid(n, dtype)
        spacing = _spacing(freq)
        assert _accepts(freq, spacing * (1.0 - 0.5 * width_floor_rtol(freq, spacing)))

    def test_a_width_a_thousandth_under_the_channel_is_refused_where_resolved(
        self, n, dtype
    ):
        """``(1 - 1e-3)`` of the spacing is refused wherever the grid resolves a
        thousandth of a channel -- float64 at every N, float32 below N = 618.
        Above that a float32 grid does not carry 1e-3 of its own spacing (8 Hz
        rounding against 3052 Hz at N = 8192), and the floor is held to 2 %."""
        freq = _grid(n, dtype)
        spacing = _spacing(freq)
        margin = 1e-3 if width_floor_rtol(freq, spacing) < 1e-3 else 2e-2
        assert not _accepts(freq, (1.0 - margin) * spacing)


def test_the_float32_cut_for_a_thousandth_is_between_617_and_618():
    """Where the margin in the test above switches, pinned rather than derived
    silently: ``4 eps 85e6 / spacing`` crosses 1e-3 between these two N."""
    below, above = _grid(617, np.float32), _grid(618, np.float32)
    assert width_floor_rtol(below, _spacing(below)) < 1e-3
    assert width_floor_rtol(above, _spacing(above)) >= 1e-3


def _verdict(freq, width, lineshape="sinc2"):
    """``"accepted"``, ``"narrow"`` or ``"unresolved"``, each checked by its text."""
    op = CWCalibrationOperator(
        amplitude=1.0, tone_freq=float(freq[len(freq) // 2]), line_width=width,
        lineshape=lineshape,
    )
    try:
        op._validate_over_the_run(freq, TIME)
    except StateValidationError as error:
        message = str(error)
        if "line_width cannot be checked" in message:
            assert "float64" in message and "relative" in message
            return "unresolved"
        assert "narrower than the channel" in message
        return "narrow"
    return "accepted"


def _raw_slack(freq):
    """The uncapped slack, ``WIDTH_FLOOR_ULPS eps max|freq| / spacing``."""
    eps = float(np.finfo(freq.dtype).eps)
    return WIDTH_FLOOR_ULPS * eps * float(np.abs(freq).max()) / _spacing(freq)


#: Float32 60-85 MHz grids past the cap: the raw slack ``4 eps 85e6 / spacing``
#: passes 2e-2 between N = 12331 and 12332 and reaches 1 near N = 617000.
PAST_THE_CAP = [12332, 16384, 20000, 200000, 617000]


@pytest.mark.parametrize("n", PAST_THE_CAP)
def test_the_slack_is_capped(n):
    freq = _grid(n, np.float32)
    assert width_floor_rtol(freq, _spacing(freq)) == WIDTH_FLOOR_RTOL_MAX


@pytest.mark.parametrize("n", PAST_THE_CAP)
def test_past_the_cap_a_width_the_rounding_cannot_place_is_unresolved(n):
    """Uncapped, a width 2 % under one channel was accepted from N = 12332
    and a 1e-6 Hz line at N = 617000. Capped alone, the ideal width would be
    refused as narrow at N = 200000, where it sits 2.3 % below the median gap.
    So a width inside ``floor (1 +/- raw)`` -- the ideal width, and the median
    gap itself -- is refused as unresolved, with the remedy."""
    freq = _grid(n, np.float32)
    for width in ((HIGH - LOW) / (n - 1), _spacing(freq)):
        assert _verdict(freq, width) == "unresolved"


@pytest.mark.parametrize("n", PAST_THE_CAP)
def test_past_the_cap_a_width_clear_of_the_band_is_judged_as_before(n):
    """The band ``floor (1 +/- raw)`` is all the refusal covers: a width
    above it passes the floor, a width below it is refused as narrow."""
    freq = _grid(n, np.float32)
    raw = _raw_slack(freq)
    spacing = _spacing(freq)
    assert _verdict(freq, spacing * (1.0 + raw) * 1.01) == "accepted"
    if raw < 0.9:
        assert _verdict(freq, spacing * (1.0 - raw) * 0.99) == "narrow"


class TestAtN16384:
    """The review's grid: float32 60-85 MHz, N = 16384, raw slack 2.7e-2.
    The width-independent refusal refused every width here; base accepted
    every width at or above one channel."""

    N = 16384

    @pytest.mark.parametrize("channels", [2, 5, 20])
    def test_a_line_several_channels_wide_is_accepted(self, channels):
        freq = _grid(self.N, np.float32)
        assert _verdict(freq, channels * (HIGH - LOW) / (self.N - 1)) == "accepted"

    def test_a_one_channel_gaussian_four_times_its_floor_is_accepted(self):
        freq = _grid(self.N, np.float32)
        assert _verdict(freq, (HIGH - LOW) / (self.N - 1), "gaussian") == "accepted"

    def test_a_width_inside_the_band_is_unresolved(self):
        freq = _grid(self.N, np.float32)
        assert _verdict(freq, 1.01 * _spacing(freq)) == "unresolved"

    def test_a_width_far_below_is_narrow(self):
        freq = _grid(self.N, np.float32)
        assert _verdict(freq, 0.5 * _spacing(freq)) == "narrow"


def test_at_a_million_channels_a_1e_6_hz_line_is_still_refused():
    """raw slack 1.6: ``floor (1 - raw)`` is below zero, so no positive width
    is clearly narrow and a 1e-6 Hz line is inside the band."""
    freq = _grid(1_000_000, np.float32)
    assert _raw_slack(freq) > 1.0
    assert _verdict(freq, 1e-6) == "unresolved"


def test_the_same_fine_grid_in_float64_is_judged_normally():
    """The remedy the refusal names, run at N = 200000."""
    freq = _grid(200000, np.float64)
    assert _verdict(freq, (HIGH - LOW) / 199999) == "accepted"


@pytest.mark.parametrize("n", [12000, 12331])
def test_a_grid_inside_the_cap_is_judged_normally(n):
    """N = 12000 (raw slack 1.9e-2) and 12331, the last N under the cap: the
    ideal width is accepted, and a width 2.5 % under the channel is still
    refused as narrow, not as unresolved."""
    freq = _grid(n, np.float32)
    spacing = _spacing(freq)
    assert width_floor_rtol(freq, spacing) < WIDTH_FLOOR_RTOL_MAX
    assert _verdict(freq, (HIGH - LOW) / (n - 1)) == "accepted"
    assert _verdict(freq, 0.975 * spacing) == "narrow"


def test_the_rounding_factor_is_pinned_by_a_verdict():
    """The same cut, read off the operator's verdict rather than the function:
    a width 1e-3 under one channel is refused at N = 617 and accepted at 618.
    With ``WIDTH_FLOOR_ULPS`` at 1 the cut would sit near N = 2470 and 618
    would refuse; at 8, near 309 and 617 would accept."""
    below, above = _grid(617, np.float32), _grid(618, np.float32)
    assert not _accepts(below, (1.0 - 1e-3) * _spacing(below))
    assert _accepts(above, (1.0 - 1e-3) * _spacing(above))


@pytest.mark.parametrize(
    ("dtype", "last"),
    [pytest.param(np.float32, 599, id="float32"), pytest.param(np.float64, 8192, id="float64")],
)
def test_every_n_accepts_the_ideal_width(dtype, last):
    """The audited sweep over every N rather than a sample. float32 stops at
    the audited 599 because ``jnp.linspace`` compiles once per N (10 s for
    these 597); the sampled N above carry it to 8192."""
    refused = [n for n in range(3, last + 1)
               if not _accepts(_grid(n, dtype), (HIGH - LOW) / (n - 1))]
    assert refused == []


@pytest.mark.parametrize("start", [LOW, LOW + 0.5 * RHINO_WIDTH, 983 * RHINO_WIDTH])
@pytest.mark.parametrize("n", [16, 64, 410])
def test_the_rhino_width_is_accepted_on_a_float32_grid(start, n):
    """410 channels of 61035.15625 Hz span 60-85 MHz; each channel is stored on
    a 4-8 Hz float32 grid. Refused on every such grid before the fix."""
    freq = np.asarray(jnp.asarray(start + RHINO_WIDTH * np.arange(n), dtype=jnp.float32))
    assert freq.dtype == np.float32
    assert _accepts(freq, RHINO_WIDTH)


def test_the_rhino_width_is_accepted_through_the_operator_call():
    """The same verdict through ``__call__``, which builds the grid from a State."""
    freq = jnp.asarray(LOW + RHINO_WIDTH * np.arange(410), dtype=jnp.float32)
    state = State(
        data=jnp.ones((4, 410)), coords=Coordinates(time=jnp.asarray(TIME), freq=freq),
        meta={"telescope": "RHINO", "obs_id": "a5-5"},
    )
    op = CWCalibrationOperator(amplitude=1.0, tone_freq=72.5e6, line_width=RHINO_WIDTH)
    assert np.isfinite(np.asarray(op(state).data)).all()


class TestTheDefinition:
    def test_float64_keeps_the_lower_bound(self):
        freq = _grid(8192, np.float64)
        assert width_floor_rtol(freq, _spacing(freq)) == WIDTH_FLOOR_RTOL

    def test_float32_scales_with_the_grid_resolution(self):
        freq = _grid(8192, np.float32)
        spacing = _spacing(freq)
        expected = WIDTH_FLOOR_ULPS * float(np.finfo(np.float32).eps) * HIGH / spacing
        assert expected > WIDTH_FLOOR_RTOL
        assert width_floor_rtol(freq, spacing) == pytest.approx(expected, rel=1e-6)

    def test_a_coarse_float32_grid_keeps_the_lower_bound(self):
        """Three channels 12.5 MHz apart: 4 eps max / spacing is 3.2e-6."""
        freq = _grid(3, np.float32)
        assert width_floor_rtol(freq, _spacing(freq)) == WIDTH_FLOOR_RTOL

    def test_an_integer_grid_keeps_the_lower_bound(self):
        freq = np.arange(60_000_000, 60_000_100, 10, dtype=np.int64)
        assert width_floor_rtol(freq, 10.0) == WIDTH_FLOOR_RTOL
