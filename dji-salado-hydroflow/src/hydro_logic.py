"""
hydro_logic.py
==============
Core hydrodynamic processing module for the AI-HydroFlow Salado project.

Provides:
  - apply_gaussian_filter : smooth a Digital Terrain Model (DTM) raster to
    suppress LiDAR noise while preserving macro-hydrological features.
  - calculate_flow_vector : derive a unit-length flow-direction vector from
    the smoothed DTM and a binary water mask.

All operations are pure NumPy/SciPy functions (no ROS 2 dependency) so that
they can be unit-tested independently of the drone runtime.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def apply_gaussian_filter(matrix: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Apply a Gaussian low-pass filter to a 2-D DTM raster.

    The filter smooths micro-relief artefacts caused by LiDAR range noise
    (≈ 2 cm at 50 m), while preserving hydraulically significant features
    such as natural channels and micro-depressions larger than 5 cm.

    This is especially important for the near-flat terrain of the Argentine
    Pampas (slope < 0.1 %) where raw gradients are dominated by noise rather
    than true topographic signal.

    Parameters
    ----------
    matrix : np.ndarray
        2-D array of elevation values (metres).  NaN values are accepted; the
        filter is applied after temporarily replacing NaN with the local mean,
        and NaN positions are restored afterwards.
    sigma : float, optional
        Standard deviation of the Gaussian kernel (grid cells).  Default is
        2.0, which corresponds to a 10 m spatial scale at 5 m/cell resolution.

    Returns
    -------
    np.ndarray
        Smoothed elevation array of the same shape and dtype (float64).

    Raises
    ------
    ValueError
        If *matrix* is not a 2-D array or *sigma* is not positive.
    """
    if matrix.ndim != 2:
        raise ValueError(
            f"apply_gaussian_filter expects a 2-D array; got shape {matrix.shape}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive; got {sigma}")

    matrix = np.asarray(matrix, dtype=np.float64)

    nan_mask = np.isnan(matrix)
    if nan_mask.any():
        fill_value = np.nanmean(matrix)
        matrix = np.where(nan_mask, fill_value, matrix)
        smoothed = gaussian_filter(matrix, sigma=sigma)
        smoothed = np.where(nan_mask, np.nan, smoothed)
    else:
        smoothed = gaussian_filter(matrix, sigma=sigma)

    return smoothed


def calculate_flow_vector(
    smoothed_dtm: np.ndarray,
    water_mask: np.ndarray,
) -> np.ndarray:
    """Compute a unit-length hydraulic flow-direction vector field.

    The flow direction at each cell is the steepest-descent direction of the
    smoothed DTM, masked to cells where surface water has been detected by the
    thermal camera.  Cells outside the water mask return a zero vector.

    The gradient (∇z) is computed with NumPy's central-difference scheme.
    The negative gradient points *downhill* (hydraulic flow direction).

    Parameters
    ----------
    smoothed_dtm : np.ndarray
        2-D float64 array of smoothed elevation values (metres), as returned
        by :func:`apply_gaussian_filter`.
    water_mask : np.ndarray
        2-D boolean (or 0/1 integer) array of the same shape as
        *smoothed_dtm*.  True / 1 indicates a water-covered cell.

    Returns
    -------
    np.ndarray
        3-D array of shape ``(2, rows, cols)`` where:
          - ``result[0]`` is the unit-vector component along the row axis
            (north–south for georeferenced rasters).
          - ``result[1]`` is the unit-vector component along the column axis
            (east–west for georeferenced rasters).
        Cells outside the water mask are zero.  Flat cells within the mask
        (|∇z| == 0) also return zero vectors to avoid division by zero.

    Raises
    ------
    ValueError
        If *smoothed_dtm* and *water_mask* shapes do not match.
    """
    smoothed_dtm = np.asarray(smoothed_dtm, dtype=np.float64)
    water_mask = np.asarray(water_mask, dtype=bool)

    if smoothed_dtm.shape != water_mask.shape:
        raise ValueError(
            f"smoothed_dtm shape {smoothed_dtm.shape} does not match "
            f"water_mask shape {water_mask.shape}"
        )

    # Central-difference gradient: returns (grad_row, grad_col)
    grad_row, grad_col = np.gradient(smoothed_dtm)

    # Hydraulic flow follows the negative gradient (downhill)
    flow_row = -grad_row
    flow_col = -grad_col

    # Magnitude of the gradient vector at each cell
    magnitude = np.sqrt(flow_row**2 + flow_col**2)

    # Avoid division by zero on perfectly flat cells
    safe_magnitude = np.where(magnitude > 0, magnitude, 1.0)

    # Normalise to unit length
    unit_row = flow_row / safe_magnitude
    unit_col = flow_col / safe_magnitude

    # Apply water mask: zero out non-water cells
    unit_row = np.where(water_mask, unit_row, 0.0)
    unit_col = np.where(water_mask, unit_col, 0.0)
    # Also zero flat cells inside the mask
    unit_row = np.where((magnitude > 0) | ~water_mask, unit_row, 0.0)
    unit_col = np.where((magnitude > 0) | ~water_mask, unit_col, 0.0)

    return np.stack([unit_row, unit_col], axis=0)
