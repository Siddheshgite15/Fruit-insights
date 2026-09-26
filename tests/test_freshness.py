"""
Unit tests for the Freshness Index calculation.
No models or images needed.
"""
import pytest
from app.utils.preprocessing import compute_freshness_index


class TestFreshnessIndexFresh:

    def test_fresh_high_confidence_in_range(self):
        idx = compute_freshness_index("Fresh", 94.0)
        assert 10.0 <= idx <= 100.0

    def test_fresh_max_confidence(self):
        idx = compute_freshness_index("Fresh", 100.0)
        assert idx == 100.0

    def test_fresh_min_confidence(self):
        idx = compute_freshness_index("Fresh", 0.0)
        assert idx == pytest.approx(10.0, abs=0.01)

    def test_fresh_always_geq_10(self):
        for conf in range(0, 101, 5):
            idx = compute_freshness_index("Fresh", float(conf))
            assert idx >= 10.0, f"Fresh index {idx} < 10 at confidence {conf}"

    def test_fresh_never_exceeds_100(self):
        for conf in [99.0, 100.0, 101.0]:
            idx = compute_freshness_index("Fresh", conf)
            assert idx <= 100.0

    def test_fresh_formula(self):
        """10 + (conf/100) * 90"""
        idx = compute_freshness_index("Fresh", 80.0)
        assert idx == pytest.approx(10.0 + 0.80 * 90.0, abs=0.01)


class TestFreshnessIndexRotten:

    def test_rotten_always_below_10(self):
        for conf in range(0, 101, 5):
            idx = compute_freshness_index("Rotten", float(conf))
            assert idx < 10.0, f"Rotten index {idx} >= 10 at confidence {conf}"

    def test_rotten_high_confidence_near_zero(self):
        """High rotten confidence = very low freshness."""
        idx = compute_freshness_index("Rotten", 99.0)
        assert idx < 1.0

    def test_rotten_low_confidence_near_nine(self):
        idx = compute_freshness_index("Rotten", 0.0)
        assert idx == pytest.approx(9.99, abs=0.01)

    def test_rotten_never_negative(self):
        idx = compute_freshness_index("Rotten", 100.0)
        assert idx >= 0.0


class TestFreshnessIndexEdgeCases:

    def test_case_insensitive_fresh(self):
        idx_upper = compute_freshness_index("FRESH", 70.0)
        idx_lower = compute_freshness_index("fresh", 70.0)
        assert idx_upper == idx_lower

    def test_case_insensitive_rotten(self):
        idx = compute_freshness_index("ROTTEN", 70.0)
        assert idx < 10.0

    def test_fresh_and_rotten_same_confidence_gap(self):
        """Fresh index must be strictly greater than rotten index."""
        fresh  = compute_freshness_index("Fresh",  80.0)
        rotten = compute_freshness_index("Rotten", 80.0)
        assert fresh > rotten

    def test_result_rounded_to_2dp(self):
        idx = compute_freshness_index("Fresh", 73.456789)
        assert idx == round(idx, 2)
