"""The figure of merit on synthetic posteriors whose answers are known exactly."""

import numpy as np
import pytest
from scipy import stats

from global21cm import fom

TRUTH = np.array([-0.05, -0.15, -0.10, -0.02])  # ||t||^2 = 0.0354


def _symmetric(center, offsets):
    """Draws ``center +/- offset`` for each row of ``offsets``: mean is ``center`` exactly."""
    offsets = np.asarray(offsets, dtype=float)
    return np.concatenate([center + offsets, center - offsets])


class TestAnalyticCases:
    def test_perfect_recovery(self):
        draws = np.tile(TRUTH, (50, 1))
        scores = fom.extraction(draws, TRUTH)
        assert scores.mse == 0.0
        assert scores.ser == float("inf")
        assert scores.bias_fraction == 0.0
        assert scores.coverage68 == 1.0 and scores.coverage95 == 1.0

    def test_pure_bias(self):
        shift = np.array([0.01, -0.02, 0.0, 0.02])
        draws = np.tile(TRUTH + shift, (20, 1))
        mse, bias2, variance = fom.mse_decomposition(draws, TRUTH)
        assert variance == 0.0
        assert bias2 == pytest.approx(np.sum(shift**2), rel=1e-12)
        assert fom.signal_to_error(draws, TRUTH) == pytest.approx(
            np.sqrt(np.sum(TRUTH**2) / np.sum(shift**2)), rel=1e-12
        )
        scores = fom.extraction(draws, TRUTH)
        assert scores.bias_fraction == 1.0

    def test_pure_variance(self):
        offsets = np.diag([0.03, 0.01, 0.02, 0.04])
        draws = _symmetric(TRUTH, offsets)
        mse, bias2, variance = fom.mse_decomposition(draws, TRUTH)
        assert bias2 == pytest.approx(0.0, abs=1e-30)
        # 2 of 8 draws differ from the mean in each coordinate, by +/- a.
        assert variance == pytest.approx(np.sum(np.diag(offsets) ** 2) * 2 / 8, rel=1e-12)
        assert mse == pytest.approx(bias2 + variance, rel=1e-15)

    def test_prior_only(self):
        # Draws centred on zero, knowing nothing of the truth: SER < 1.
        offsets = 0.1 * np.eye(4)
        draws = _symmetric(np.zeros(4), offsets)
        expected = np.sqrt(np.sum(TRUTH**2) / (np.sum(TRUTH**2) + 4 * 0.01 * 2 / 8))
        assert fom.signal_to_error(draws, TRUTH) == pytest.approx(expected, rel=1e-12)
        assert fom.signal_to_error(draws, TRUTH) < 1.0

    def test_efficiency_is_the_ratio_of_spreads(self):
        oracle = _symmetric(TRUTH, 0.01 * np.eye(4))
        joint = _symmetric(TRUTH, 0.04 * np.eye(4))
        assert fom.efficiency(oracle, joint) == pytest.approx(0.25, rel=1e-12)
        assert fom.efficiency(joint, joint) == pytest.approx(1.0, rel=1e-12)


class TestBoundaries:
    def test_a_single_draw_is_pure_bias(self):
        draw = TRUTH + np.array([0.0, 0.01, 0.0, 0.0])
        scores = fom.extraction(draw[None, :], TRUTH)
        assert scores.variance == 0.0
        assert scores.bias2 == pytest.approx(1e-4, rel=1e-12)
        assert scores.coverage68 == 0.75  # three of four channels match exactly

    def test_a_single_draw_on_the_truth(self):
        scores = fom.extraction(TRUTH[None, :], TRUTH)
        assert scores.ser == float("inf")

    def test_zero_variance_posterior_off_the_truth(self):
        draws = np.tile(TRUTH + 0.001, (100, 1))
        assert np.isnan(fom.efficiency(draws, draws))

    def test_truth_zero(self):
        zero = np.zeros(4)
        wide = _symmetric(zero, 0.1 * np.eye(4))
        assert fom.signal_to_error(wide, zero) == 0.0
        assert np.isnan(fom.signal_to_error(np.zeros((3, 4)), zero))
        assert np.isnan(fom.perpendicular_fraction(np.eye(4), zero))

    def test_efficiency_of_a_zero_spread_joint(self):
        spread = _symmetric(TRUTH, 0.01 * np.eye(4))
        assert fom.efficiency(spread, np.tile(TRUTH, (5, 1))) == float("inf")

    def test_shapes_and_values_are_checked(self):
        with pytest.raises(ValueError):
            fom.extraction(np.zeros((3, 5)), TRUTH)
        with pytest.raises(ValueError):
            fom.extraction(np.full((3, 4), np.nan), TRUTH)
        with pytest.raises(ValueError):
            fom.coverage(np.zeros((3, 4)), TRUTH, 1.0)


class TestPerpendicularFraction:
    def test_signal_inside_the_span_is_zero(self):
        columns = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        assert fom.perpendicular_fraction(columns, np.array([2.0, -1.0, 0.0, 0.0])) == 0.0

    def test_signal_orthogonal_to_the_span_is_one(self):
        columns = np.array([[1.0], [1.0], [0.0], [0.0]])
        signal = np.array([1.0, -1.0, 0.0, 0.0])
        assert fom.perpendicular_fraction(columns, signal) == pytest.approx(1.0, rel=1e-14)

    def test_a_half_angle(self):
        columns = np.array([[1.0], [0.0]])
        assert fom.perpendicular_fraction(columns, np.array([1.0, 1.0])) == pytest.approx(
            np.sqrt(0.5), rel=1e-14
        )

    def test_degenerate_columns_are_not_counted_twice(self):
        columns = np.array([[1.0, 1.0], [0.0, 0.0], [0.0, 0.0]])
        signal = np.array([0.0, 3.0, 4.0])
        assert fom.perpendicular_fraction(columns, signal) == pytest.approx(1.0, rel=1e-14)

    def test_no_columns_leave_everything(self):
        assert fom.perpendicular_fraction(np.zeros((3, 2)), np.array([1.0, 0.0, 0.0])) == 1.0

    def test_the_fraction_is_unit_free(self):
        # The rank cut is relative to the largest singular value: rescaling the
        # foreground columns (a change of units) must not change the fraction.
        columns, signal = np.array([[1.0], [0.0]]), np.array([1.0, 1.0])
        for scale in (1e-13, 1.0, 1e8):
            assert fom.perpendicular_fraction(scale * columns, signal) == pytest.approx(np.sqrt(0.5), rel=1e-12)


class TestCalibration:
    def test_parameter_z2_of_a_gaussian_offset(self):
        # Six draws, +/- sqrt(3) e_i about the mean: unit covariance, so z^2 = |offset|^2.
        u = _symmetric(np.array([0.3, -0.4, 0.0]), np.sqrt(3) * np.eye(3))
        z2, tail = fom.parameter_z2(u, np.zeros(3))
        assert z2 == pytest.approx(0.25, rel=1e-12)
        assert tail == pytest.approx(stats.chi2.sf(0.25, 3), rel=1e-12)

    def test_parameter_z2_degenerate(self):
        assert fom.parameter_z2(np.zeros((1, 7)), np.zeros(7)) == (0.0, 1.0)
        z2, tail = fom.parameter_z2(np.ones((5, 7)), np.zeros(7))
        assert z2 == float("inf") and tail == 0.0

    def test_quantile_and_trough(self):
        freqs = np.array([50.0, 60.0, 70.0, 80.0])
        curves = np.array([[0.0, -1.0, -2.0, 0.0], [0.0, -3.0, -1.0, 0.0]])
        depth, where = fom.trough(curves, freqs)
        np.testing.assert_array_equal(depth, [-2.0, -3.0])
        np.testing.assert_array_equal(where, [70.0, 60.0])
        assert fom.quantile_of(-2.5, depth) == 0.5
        assert fom.quantile_of(-2.0, depth) == 0.75  # a tie counts half
        assert fom.quantile_of(-9.0, depth) == 0.0 and fom.quantile_of(9.0, depth) == 1.0

    def test_calibrated_thresholds(self):
        assert fom.calibrated(0.5, 0.5, 0.5)
        assert not fom.calibrated(0.001, 0.5, 0.5)
        assert not fom.calibrated(0.5, 0.0, 0.5)
        assert not fom.calibrated(0.5, 0.5, 1.0)

    def test_calibrated_edges(self):
        # Each gate at its own edge: the tail "at least ALPHA", the quantiles in
        # [ALPHA/2, 1 - ALPHA/2], so a quantile between ALPHA/2 and ALPHA passes.
        assert fom.calibrated(fom.ALPHA, 0.5, 0.5)
        assert fom.calibrated(0.5, 0.75 * fom.ALPHA, 0.5)
        assert fom.calibrated(0.5, 0.5, 1.0 - 0.75 * fom.ALPHA)
        assert not fom.calibrated(0.5, 0.4 * fom.ALPHA, 0.5)
        assert not fom.calibrated(0.5, 0.5, 1.0 - 0.4 * fom.ALPHA)

    def test_alpha_is_one_percent(self):
        # Literals, not multiples of ALPHA: the value of the level is what is pinned.
        assert not fom.calibrated(0.009, 0.5, 0.5)
        assert fom.calibrated(0.5, 0.005, 0.995)
        assert not fom.calibrated(0.5, 0.0049, 0.5)

    def test_prior_relative_scores(self):
        prior = _symmetric(np.zeros(4), 0.1 * np.eye(4))
        scores = fom.relative_to_prior(prior, prior, TRUTH)
        assert scores == {"ser_over_prior": 1.0, "variance_over_prior": 1.0}
        tight = _symmetric(TRUTH, 0.01 * np.eye(4))
        assert fom.relative_to_prior(tight, prior, TRUTH)["variance_over_prior"] == pytest.approx(0.01)


class TestCoverage:
    def test_coverage_counts_channels_inside_the_central_band(self):
        # 101 draws per channel at 0, 0.01, ..., 1: the p-th percentile is p / 100.
        draws = np.tile(np.linspace(0.0, 1.0, 101)[:, None], (1, 4))
        truth = np.array([0.02, 0.20, 0.50, 0.90])
        assert fom.coverage(draws, truth, 0.68) == 0.5  # band [0.16, 0.84]: 0.20 and 0.50
        assert fom.coverage(draws, truth, 0.95) == 0.75  # band [0.025, 0.975]: all but 0.02
        scores = fom.extraction(draws, truth)
        assert (scores.coverage68, scores.coverage95) == (0.5, 0.75)


class TestSpread:
    def test_spread_is_the_population_trace(self):
        draws = np.array([[0.0, 1.0], [2.0, 3.0]])
        assert fom.spread(draws) == 2.0  # per column: variance 1 about the mean, divided by S = 2

    def test_spread_is_the_variance_term_of_the_mse(self):
        # Two spellings of tr Cov: they must agree, whatever the draw count.
        draws = np.random.default_rng(4).normal(size=(5, 4))
        _, _, variance = fom.mse_decomposition(draws, TRUTH)
        assert fom.spread(draws) == pytest.approx(variance, rel=1e-14)
