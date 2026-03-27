"""
hydro_logic.py — Core hydrodynamic computations for AI-HydroFlow Salado.

Provides:
  - apply_gaussian_filter : smooths a DTM raster to remove micro-relief noise.
  - calculate_flow_vector  : derives a unit flow-direction vector from the
                             smoothed DTM constrained to a water mask.

All functions operate on 2-D NumPy arrays and have no side-effects, making them
straightforward to unit-test in isolation from the ROS 2 runtime.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter


def apply_gaussian_filter(
    matrix: NDArray[np.floating],
    sigma: float = 2.0,
) -> NDArray[np.floating]:
    """Apply a 2-D Gaussian filter to a DTM raster.

    The filter suppresses high-frequency micro-relief noise inherent to the
    flat pampean terrain of the Cuenca del Salado while preserving the
    macro-drainage gradient used for flow navigation.

    Parameters
    ----------
    matrix:
        2-D array representing raw DTM elevation values (metres).  Any finite
        floating-point dtype is accepted.
    sigma:
        Standard deviation of the Gaussian kernel in *pixel* units.  A value
        of 2.0 corresponds to ≈ 0.5 m when the raster resolution is 0.25 m/px.

    Returns
    -------
    NDArray[np.floating]
        Smoothed DTM with the same shape and dtype as *matrix*.

    Raises
    ------
    ValueError
        If *matrix* is not exactly 2-D, or if *sigma* is not positive.
    """
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError(
            f"matrix must be 2-D, got shape {matrix.shape}"
        )
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")

    return gaussian_filter(matrix, sigma=sigma)


def calculate_flow_vector(
    smoothed_dtm: NDArray[np.floating],
    water_mask: NDArray[np.bool_],
) -> NDArray[np.floating]:
    """Compute the dominant flow-direction unit vector from a smoothed DTM.

    The algorithm:
    1. Computes the numerical gradient ∇z = (∂z/∂x, ∂z/∂y) of the DTM.
    2. Negates the gradient so the vector points *downhill* (flow direction).
    3. Gates the gradient with *water_mask* to consider only water pixels.
    4. Averages the masked gradient components and normalises to a unit vector.

    Parameters
    ----------
    smoothed_dtm:
        2-D array of smoothed elevation values (metres), typically the output
        of :func:`apply_gaussian_filter`.
    water_mask:
        2-D boolean array with the same shape as *smoothed_dtm*. ``True``
        pixels indicate confirmed water surface (from thermal segmentation).

    Returns
    -------
    NDArray[np.floating]
        1-D array ``[v_x, v_y]`` — unit vector in ENU frame (East, North)
        pointing in the direction of net downhill flow over the water-covered
        area.  Returns ``[0.0, 0.0]`` when no water pixels are present or
        when the mean gradient magnitude is zero (perfectly flat terrain).

    Raises
    ------
    ValueError
        If *smoothed_dtm* and *water_mask* have different shapes.
    """
    smoothed_dtm = np.asarray(smoothed_dtm, dtype=float)
    water_mask = np.asarray(water_mask, dtype=bool)

    if smoothed_dtm.shape != water_mask.shape:
        raise ValueError(
            f"smoothed_dtm shape {smoothed_dtm.shape} does not match "
            f"water_mask shape {water_mask.shape}"
        )

    # np.gradient returns [grad_row (axis-0, south-positive), grad_col (axis-1, east-positive)]
    grad_row, grad_col = np.gradient(smoothed_dtm)

    # Flow is downhill → negate the gradient
    flow_row = -grad_row
    flow_col = -grad_col

    # Gate with water mask
    if not np.any(water_mask):
        return np.array([0.0, 0.0])

    mean_col = float(np.mean(flow_col[water_mask]))
    mean_row = float(np.mean(flow_row[water_mask]))

    magnitude = np.sqrt(mean_col**2 + mean_row**2)
    if magnitude < 1e-9:
        return np.array([0.0, 0.0])

    # In NumPy array convention increasing row index → south; negate to get
    # the geographic North component (positive = north, negative = south).
    # Column index maps naturally to East (positive = east, negative = west).
    v_east = mean_col / magnitude
    v_north = -mean_row / magnitude  # negate: row↓ = south ↔ north component negative
    return np.array([v_east, v_north])
