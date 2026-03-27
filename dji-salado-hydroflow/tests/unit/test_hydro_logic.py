"""
test_hydro_logic.py — Unit tests for the core hydrodynamic algorithms.

Tests are deliberately free of ROS 2 or PSDK dependencies so they can run
in any standard Python environment (CI, developer laptop, Manifold 3).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# Adjust sys.path so that the src package is importable without installation
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hydro_logic import apply_gaussian_filter, calculate_flow_vector


# ---------------------------------------------------------------------------
# apply_gaussian_filter
# ---------------------------------------------------------------------------

class TestApplyGaussianFilter:
    """Tests for apply_gaussian_filter."""

    def test_output_shape_matches_input(self):
        """Filtered array must have the same shape as the input."""
        matrix = np.random.rand(50, 60)
        result = apply_gaussian_filter(matrix, sigma=2.0)
        assert result.shape == matrix.shape

    def test_output_dtype_is_float(self):
        """Output must be floating-point regardless of input dtype."""
        matrix = np.ones((10, 10), dtype=np.int32)
        result = apply_gaussian_filter(matrix)
        assert np.issubdtype(result.dtype, np.floating)

    def test_uniform_matrix_unchanged(self):
        """A spatially uniform DTM should be unaffected by smoothing."""
        matrix = np.full((20, 20), 42.0)
        result = apply_gaussian_filter(matrix, sigma=2.0)
        np.testing.assert_allclose(result, matrix, atol=1e-10)

    def test_smoothing_reduces_variance(self):
        """Gaussian smoothing must reduce spatial variance."""
        rng = np.random.default_rng(seed=0)
        matrix = rng.standard_normal((100, 100))
        result = apply_gaussian_filter(matrix, sigma=3.0)
        assert result.var() < matrix.var()

    def test_large_sigma_approaches_mean(self):
        """Very large σ should push every pixel toward the global mean."""
        rng = np.random.default_rng(seed=42)
        matrix = rng.standard_normal((50, 50))
        result = apply_gaussian_filter(matrix, sigma=50.0)
        # All values should be close to the global mean
        np.testing.assert_allclose(result, matrix.mean(), atol=1e-1)

    def test_raises_on_1d_input(self):
        """1-D array must raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            apply_gaussian_filter(np.ones(10), sigma=1.0)

    def test_raises_on_3d_input(self):
        """3-D array must raise ValueError."""
        with pytest.raises(ValueError, match="2-D"):
            apply_gaussian_filter(np.ones((5, 5, 3)), sigma=1.0)

    def test_raises_on_nonpositive_sigma(self):
        """Non-positive sigma must raise ValueError."""
        with pytest.raises(ValueError, match="sigma"):
            apply_gaussian_filter(np.ones((5, 5)), sigma=0.0)
        with pytest.raises(ValueError, match="sigma"):
            apply_gaussian_filter(np.ones((5, 5)), sigma=-1.0)


# ---------------------------------------------------------------------------
# calculate_flow_vector
# ---------------------------------------------------------------------------

class TestCalculateFlowVector:
    """Tests for calculate_flow_vector."""

    def _slope_east(self, rows: int = 20, cols: int = 20) -> np.ndarray:
        """Create a DTM with a uniform eastward slope (elevation increases west→east)."""
        dtm = np.zeros((rows, cols))
        for c in range(cols):
            dtm[:, c] = float(c)
        return dtm

    def _slope_north(self, rows: int = 20, cols: int = 20) -> np.ndarray:
        """Create a DTM with a uniform northward slope (elevation increases south→north)."""
        dtm = np.zeros((rows, cols))
        for r in range(rows):
            dtm[r, :] = float(rows - r)
        return dtm

    def test_output_is_unit_vector(self):
        """Flow vector must be a unit vector (or zero)."""
        dtm = self._slope_east()
        mask = np.ones(dtm.shape, dtype=bool)
        vec = calculate_flow_vector(dtm, mask)
        if not np.allclose(vec, 0.0):
            np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-9)

    def test_east_slope_produces_west_flow(self):
        """Uniform eastward slope → flow should point west (v_east < 0)."""
        dtm = self._slope_east()
        mask = np.ones(dtm.shape, dtype=bool)
        vec = calculate_flow_vector(dtm, mask)
        # Flow is downhill → west direction → v_east (index 0) should be negative
        assert vec[0] < 0, f"Expected westward flow (v_east < 0), got {vec}"

    def test_north_slope_produces_south_flow(self):
        """Uniform northward slope → flow should point south (v_north < 0)."""
        dtm = self._slope_north()
        mask = np.ones(dtm.shape, dtype=bool)
        vec = calculate_flow_vector(dtm, mask)
        # Flow is downhill → south direction → v_north (index 1) should be negative
        assert vec[1] < 0, f"Expected southward flow (v_north < 0), got {vec}"

    def test_empty_water_mask_returns_zero(self):
        """No water pixels → zero vector (no navigable flow direction)."""
        dtm = self._slope_east()
        mask = np.zeros(dtm.shape, dtype=bool)
        vec = calculate_flow_vector(dtm, mask)
        np.testing.assert_array_equal(vec, [0.0, 0.0])

    def test_flat_terrain_returns_zero(self):
        """Perfectly flat terrain has no gradient → zero vector."""
        dtm = np.zeros((20, 20))
        mask = np.ones(dtm.shape, dtype=bool)
        vec = calculate_flow_vector(dtm, mask)
        np.testing.assert_array_equal(vec, [0.0, 0.0])

    def test_shape_mismatch_raises(self):
        """Mismatched dtm and water_mask shapes must raise ValueError."""
        dtm = np.ones((10, 10))
        mask = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            calculate_flow_vector(dtm, mask)

    def test_output_length_is_two(self):
        """Return value must always be a 1-D array with exactly 2 elements."""
        dtm = self._slope_east()
        mask = np.ones(dtm.shape, dtype=bool)
        vec = calculate_flow_vector(dtm, mask)
        assert vec.shape == (2,)

    def test_partial_water_mask(self):
        """Only the water-covered region should influence the flow direction."""
        dtm = self._slope_east(20, 20)
        # Mask only the right half (high elevation) — flow should still be westward
        mask = np.zeros((20, 20), dtype=bool)
        mask[:, 10:] = True
        vec = calculate_flow_vector(dtm, mask)
        assert vec[0] < 0, f"Expected westward flow over high-elevation water, got {vec}"

    def test_gaussian_then_flow_pipeline(self):
        """End-to-end: Gaussian filter → flow vector gives a valid unit vector."""
        rng = np.random.default_rng(seed=7)
        noisy_dtm = self._slope_east() + rng.standard_normal((20, 20)) * 0.1
        smoothed = apply_gaussian_filter(noisy_dtm, sigma=2.0)
        mask = np.ones((20, 20), dtype=bool)
        vec = calculate_flow_vector(smoothed, mask)
        norm = float(np.linalg.norm(vec))
        assert norm > 0, "Pipeline produced a zero vector for a non-flat DTM"
        np.testing.assert_allclose(norm, 1.0, atol=1e-9)
