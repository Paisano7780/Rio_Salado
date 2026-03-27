"""
rtk_manager.py — Management of RTK rescue antenna network.

Provides :class:`RTKManager` which stores a list of pre-surveyed D-RTK 2
base-station coordinates and selects the optimal landing node based on
proximity and wind-vector alignment.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# Type alias: WGS-84 coordinate (latitude°, longitude°, altitude_m)
Coordinate = Tuple[float, float, float]

# ---------------------------------------------------------------------------
# Default rescue nodes — replace with surveyed coordinates for each mission.
# Coordinates are in WGS-84 decimal degrees; altitude in metres (orthometric).
# ---------------------------------------------------------------------------
DEFAULT_RTK_NODES: List[Coordinate] = [
    (-36.2741, -60.9308, 32.5),  # Node 1 — Base camp
    (-36.3105, -60.8876, 31.8),  # Node 2 — North sector
    (-36.2500, -61.0150, 33.1),  # Node 3 — West sector
    (-36.3400, -61.1000, 30.9),  # Node 4 — South-west sector
]

# Earth mean radius (metres) for Haversine formula
_EARTH_RADIUS_M: float = 6_371_000.0


def _haversine(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Return the great-circle distance in metres between two WGS-84 points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * _EARTH_RADIUS_M * math.asin(math.sqrt(a))


def _bearing(
    lat1: float, lon1: float,
    lat2: float, lon2: float,
) -> float:
    """Return the initial bearing (radians, 0 = North, clockwise) from point 1 to point 2."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return math.atan2(x, y)


class RTKManager:
    """Manages a network of RTK rescue landing nodes.

    Parameters
    ----------
    nodes:
        List of pre-surveyed WGS-84 coordinates ``(lat, lon, alt_m)``.
        Defaults to :data:`DEFAULT_RTK_NODES` if not provided.
    """

    def __init__(self, nodes: Optional[List[Coordinate]] = None) -> None:
        self._nodes: List[Coordinate] = nodes if nodes is not None else list(DEFAULT_RTK_NODES)
        if not self._nodes:
            raise ValueError("RTKManager requires at least one rescue node.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def nodes(self) -> List[Coordinate]:
        """Read-only view of the registered rescue nodes."""
        return list(self._nodes)

    def add_node(self, lat: float, lon: float, alt: float = 0.0) -> None:
        """Register an additional RTK rescue node at runtime."""
        self._nodes.append((lat, lon, alt))

    def select_nearest_node(
        self,
        current_position: NDArray[np.floating],
        wind_vector: Optional[NDArray[np.floating]] = None,
    ) -> Coordinate:
        """Select the most efficient RTK rescue node to land at.

        The scoring function maximises both proximity and tailwind advantage:

        .. code-block:: text

            score(n) = (1 / distance) × (1 + cos(θ_wind, bearing→n))

        where ``θ_wind`` is the angle between the wind vector and the bearing
        from the current position to node *n*.  When *wind_vector* is ``None``
        the wind term is omitted and the nearest node is selected.

        Parameters
        ----------
        current_position:
            1-D array ``[lat, lon, alt_m]`` of the UAV's current WGS-84
            position as reported by the RTK subsystem.
        wind_vector:
            Optional 1-D array ``[v_east, v_north]`` in m/s.  When supplied,
            nodes downwind of the UAV are scored higher to reduce battery
            consumption during the return flight.

        Returns
        -------
        Coordinate
            The ``(lat, lon, alt_m)`` tuple of the selected rescue node.
        """
        lat0 = float(current_position[0])
        lon0 = float(current_position[1])

        best_node: Optional[Coordinate] = None
        best_score: float = -1.0

        for node in self._nodes:
            lat_n, lon_n, _ = node
            dist = _haversine(lat0, lon0, lat_n, lon_n)
            if dist < 1e-3:
                # Already at this node
                return node

            proximity_score = 1.0 / dist

            if wind_vector is not None and np.linalg.norm(wind_vector) > 0.1:
                node_bearing = _bearing(lat0, lon0, lat_n, lon_n)
                wind_bearing = math.atan2(
                    float(wind_vector[0]),  # east component → x
                    float(wind_vector[1]),  # north component → y
                )
                angle_diff = node_bearing - wind_bearing
                wind_term = 1.0 + math.cos(angle_diff)
            else:
                wind_term = 1.0

            score = proximity_score * wind_term

            if score > best_score:
                best_score = score
                best_node = node

        assert best_node is not None  # nodes list is guaranteed non-empty
        return best_node
