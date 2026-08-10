"""The site adapter: unix epoch + seconds-from-start in, LST degrees out."""

import numpy as np
import pytest

from rheplicant.radio.site import _utc_string, lst_grid_deg

SITE = dict(lat_deg=53.2367, lon_deg=-2.3085, alt_m=78.0)
EPOCH = 1785312000.0
SIDEREAL_S = 86164.0905


def test_the_export_is_the_function():
    import rheplicant.radio as radio

    assert radio.lst_grid_deg is lst_grid_deg
    assert "lst_grid_deg" in radio.__all__


class TestTheEpochSeam:
    """The adapter's whole contribution: unix seconds -> the UTC string limTOD wants."""

    def test_epoch_zero_is_the_unix_origin_in_every_timezone(self, monkeypatch):
        import time

        monkeypatch.setenv("TZ", "America/New_York")
        time.tzset()
        try:
            assert _utc_string(0.0) == "1970-01-01 00:00:00.000000"
        finally:
            monkeypatch.undo()
            time.tzset()

    def test_fractional_seconds_survive(self):
        assert _utc_string(0.25).endswith("00.250000")


class TestTheGrid:
    def test_shape_and_range(self):
        time_s = np.arange(0.0, 600.0, 60.0)
        lst = lst_grid_deg(**SITE, time_s=time_s, epoch_unix_s=EPOCH)
        assert lst.shape == time_s.shape
        assert np.all((lst >= 0.0) & (lst < 360.0))

    def test_half_a_sidereal_day_advances_lst_by_180_degrees(self):
        lst = lst_grid_deg(**SITE, time_s=np.array([0.0, SIDEREAL_S / 2.0]),
                           epoch_unix_s=EPOCH)
        delta = (lst[1] - lst[0]) % 360.0
        assert delta == pytest.approx(180.0, abs=0.05)

    def test_the_rate_is_sidereal_not_solar(self):
        """One SOLAR day advances LST by ~0.9856 deg; a solar-rate
        implementation would return exactly 0 here."""
        lst = lst_grid_deg(**SITE, time_s=np.array([0.0, 86400.0]),
                           epoch_unix_s=EPOCH)
        delta = (lst[1] - lst[0]) % 360.0
        assert delta == pytest.approx(360.0 / 365.2422, abs=0.05)

    def test_the_absolute_lst_is_pinned(self):
        """A latitude/longitude swap shifts LST by a constant and slips past
        every differential assertion; one absolute pin kills it."""
        lst = lst_grid_deg(**SITE, time_s=np.array([0.0]), epoch_unix_s=EPOCH)
        assert float(lst[0]) == pytest.approx(64.6836685570699, abs=0.01)
