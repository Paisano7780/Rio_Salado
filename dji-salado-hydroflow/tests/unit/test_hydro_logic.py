"""
test_hydro_logic.py
===================
Unit tests for the hydrodynamic processing functions in hydro_logic.py.

Validates:
  - Gaussian filter output shape, dtype, and smoothing behaviour.
  - Flow-vector normalisation (unit length for non-flat water cells).
  - Correct handling of edge cases: flat terrain, empty water mask, NaN cells.
  - RTK node selection correctness in RTKManager.

Run with::

    pytest tests/unit/test_hydro_logic.py -v
"""

import math
import sys
import os

import numpy as np
import pytest

# Make src/ importable without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from hydro_logic import apply_gaussian_filter, calculate_flow_vector
from utils.rtk_manager import RTKManager, haversine_distance


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_dtm() -> np.ndarray:
    """10×10 ramp DTM: elevation increases linearly along the row axis."""
    rows, cols = 10, 10
    dtm = np.zeros((rows, cols), dtype=np.float64)
    for r in range(rows):
        dtm[r, :] = r * 0.1  # 10 cm/row slope
    return dtm


@pytest.fixture
def flat_dtm() -> np.ndarray:
    """10×10 flat DTM: all zeros."""
    return np.zeros((10, 10), dtype=np.float64)


@pytest.fixture
def full_water_mask() -> np.ndarray:
    """10×10 mask: all cells are water."""
    return np.ones((10, 10), dtype=bool)


@pytest.fixture
def empty_water_mask() -> np.ndarray:
    """10×10 mask: no water cells."""
    return np.zeros((10, 10), dtype=bool)


# =============================================================================
# apply_gaussian_filter tests
# =============================================================================


class TestApplyGaussianFilter:
    def test_output_shape_matches_input(self, synthetic_dtm):
        result = apply_gaussian_filter(synthetic_dtm, sigma=2.0)
        assert result.shape == synthetic_dtm.shape

    def test_output_dtype_is_float64(self, synthetic_dtm):
        result = apply_gaussian_filter(synthetic_dtm, sigma=2.0)
        assert result.dtype == np.float64

    def test_smoothed_values_within_input_range(self, synthetic_dtm):
        result = apply_gaussian_filter(synthetic_dtm, sigma=2.0)
        assert result.min() >= synthetic_dtm.min() - 1e-9
        assert result.max() <= synthetic_dtm.max() + 1e-9

    def test_smoothing_reduces_variance(self):
        """A noisy raster should have lower variance after filtering."""
        rng = np.random.default_rng(42)
        noisy = rng.normal(loc=5.0, scale=2.0, size=(50, 50))
        smoothed = apply_gaussian_filter(noisy, sigma=3.0)
        assert smoothed.var() < noisy.var()

    def test_sigma_zero_raises(self, synthetic_dtm):
        with pytest.raises(ValueError, match="sigma must be positive"):
            apply_gaussian_filter(synthetic_dtm, sigma=0.0)

    def test_negative_sigma_raises(self, synthetic_dtm):
        with pytest.raises(ValueError, match="sigma must be positive"):
            apply_gaussian_filter(synthetic_dtm, sigma=-1.0)

    def test_non_2d_input_raises(self):
        with pytest.raises(ValueError, match="2-D"):
            apply_gaussian_filter(np.ones((3, 3, 3)), sigma=1.0)

    def test_nan_cells_preserved(self):
        dtm = np.array([[1.0, 2.0, 3.0],
                         [4.0, np.nan, 6.0],
                         [7.0, 8.0, 9.0]])
        result = apply_gaussian_filter(dtm, sigma=1.0)
        assert np.isnan(result[1, 1]), "NaN cell should remain NaN after filtering"

    def test_integer_input_converted_to_float64(self):
        int_dtm = np.arange(16, dtype=np.int32).reshape(4, 4)
        result = apply_gaussian_filter(int_dtm, sigma=1.0)
        assert result.dtype == np.float64

    def test_linear_ramp_preserved_at_low_sigma(self, synthetic_dtm):
        """With very small sigma the smoothed surface should closely match input."""
        result = apply_gaussian_filter(synthetic_dtm, sigma=0.1)
        np.testing.assert_allclose(result, synthetic_dtm, atol=1e-3)


# =============================================================================
# calculate_flow_vector tests
# =============================================================================


class TestCalculateFlowVector:
    def test_output_shape(self, synthetic_dtm, full_water_mask):
        result = calculate_flow_vector(synthetic_dtm, full_water_mask)
        assert result.shape == (2, *synthetic_dtm.shape)

    def test_unit_length_on_water_cells_with_slope(self, synthetic_dtm, full_water_mask):
        """Every non-flat water cell must produce a unit-length vector."""
        result = calculate_flow_vector(synthetic_dtm, full_water_mask)
        v_row = result[0]
        v_col = result[1]
        lengths = np.sqrt(v_row**2 + v_col**2)

        # Interior cells definitely have non-zero gradient on the ramp
        interior = lengths[1:-1, 1:-1]
        np.testing.assert_allclose(interior, 1.0, atol=1e-9)

    def test_zero_vectors_outside_water_mask(self, synthetic_dtm, empty_water_mask):
        result = calculate_flow_vector(synthetic_dtm, empty_water_mask)
        assert np.all(result == 0.0)

    def test_flat_dtm_returns_zero_vectors(self, flat_dtm, full_water_mask):
        result = calculate_flow_vector(flat_dtm, full_water_mask)
        assert np.all(result == 0.0)

    def test_flow_direction_downhill(self, synthetic_dtm, full_water_mask):
        """On a ramp DTM (elevation = row * 0.1) flow must point toward row 0.

        In the synthetic fixture, row 0 has elevation 0 (lowest) and row 9 has
        elevation 0.9 (highest).  Downhill = decreasing row index, so the row
        component of the flow vector must be negative (−1 after normalisation).
        """
        result = calculate_flow_vector(synthetic_dtm, full_water_mask)
        # Row component should be negative (downhill = decreasing row index)
        interior_row = result[0, 1:-1, 1:-1]
        assert np.all(interior_row < 0)

    def test_shape_mismatch_raises(self, synthetic_dtm):
        bad_mask = np.ones((5, 5), dtype=bool)
        with pytest.raises(ValueError, match="shape"):
            calculate_flow_vector(synthetic_dtm, bad_mask)

    def test_partial_water_mask(self, synthetic_dtm):
        """Cells outside the mask are zero; cells inside have unit vectors."""
        mask = np.zeros((10, 10), dtype=bool)
        mask[3:7, 3:7] = True  # 4×4 sub-region
        result = calculate_flow_vector(synthetic_dtm, mask)

        # Outside mask: all zeros
        outside = result[:, ~mask]
        assert np.all(outside == 0.0)

        # Inside mask (interior cells): unit length
        interior_mask = np.zeros_like(mask)
        interior_mask[4:6, 4:6] = True
        lengths = np.sqrt(result[0]**2 + result[1]**2)
        np.testing.assert_allclose(lengths[interior_mask], 1.0, atol=1e-9)


# =============================================================================
# RTKManager tests
# =============================================================================


class TestRTKManager:
    def test_select_nearest_simple(self):
        nodes = [(-35.50, -60.20), (-35.60, -60.30), (-35.55, -60.25)]
        mgr = RTKManager(nodes=nodes)
        current = (-35.51, -60.21)
        selected = mgr.select_nearest_node(current)
        assert selected == (-35.50, -60.20), "Should select the closest node"

    def test_empty_manager_raises(self):
        mgr = RTKManager()
        with pytest.raises(RuntimeError, match="no registered nodes"):
            mgr.select_nearest_node((-35.5, -60.2))

    def test_add_node(self):
        mgr = RTKManager()
        mgr.add_node((-35.5, -60.2))
        assert len(mgr.nodes) == 1

    def test_add_nodes(self):
        mgr = RTKManager()
        mgr.add_nodes([(-35.5, -60.2), (-35.6, -60.3)])
        assert len(mgr.nodes) == 2

    def test_wind_bias_selects_downwind_node(self):
        """A southward wind should favour the southern (downwind/tailwind) node.

        wind = (-10, 0) → wind vector points south.
        Flying south to south_node = tailwind → lower cost.
        Flying north to north_node = headwind → higher cost.
        """
        current = (0.0, 0.0)
        north_node = (0.1, 0.0)   # roughly 11 km north
        south_node = (-0.1, 0.0)  # roughly 11 km south

        # Wind vector pointing south (negative north component)
        # → south_node is downwind (tailwind) → should be preferred
        wind = (-10.0, 0.0)  # 10 m/s southward wind

        mgr = RTKManager(nodes=[north_node, south_node])
        selected = mgr.select_nearest_node(current, wind_vector=wind, wind_bias_factor=0.05)
        assert selected == south_node, "Should prefer downwind (south) node with southward wind"

    def test_haversine_distance_known_value(self):
        """Buenos Aires → Montevideo ≈ 210 km."""
        buenos_aires = (-34.6037, -58.3816)
        montevideo = (-34.9011, -56.1645)
        dist = haversine_distance(buenos_aires, montevideo)
        assert 190_000 < dist < 220_000, f"Expected ~210 km, got {dist/1000:.1f} km"

    def test_haversine_same_point_is_zero(self):
        pt = (-35.5, -60.2)
        assert haversine_distance(pt, pt) == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Integration-style smoke test (hydro pipeline end-to-end)
# =============================================================================


class TestHydroPipelineSmoke:
    def test_full_pipeline_with_synthetic_data(self):
        """Verify the two-step pipeline produces valid output for a synthetic DTM."""
        rng = np.random.default_rng(0)
        # 50×50 DTM: linear ramp plus random noise
        base = np.linspace(0, 5, 50)
        dtm = np.tile(base, (50, 1)).T + rng.normal(0, 0.05, (50, 50))

        # Checkerboard water mask (50 % coverage)
        mask = np.zeros((50, 50), dtype=bool)
        mask[::2, ::2] = True
        mask[1::2, 1::2] = True

        smoothed = apply_gaussian_filter(dtm, sigma=2.0)
        vectors = calculate_flow_vector(smoothed, mask)

        # Shape check
        assert vectors.shape == (2, 50, 50)

        # All non-zero vectors have unit length
        lengths = np.sqrt(vectors[0]**2 + vectors[1]**2)
        non_zero_lengths = lengths[lengths > 0]
        np.testing.assert_allclose(non_zero_lengths, 1.0, atol=1e-9)

        # No NaN values in output
        assert not np.any(np.isnan(vectors))
