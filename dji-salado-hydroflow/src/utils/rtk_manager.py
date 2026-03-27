"""
rtk_manager.py
==============
Manages a list of RTK rescue-antenna nodes and selects the optimal landing
site for the battery-failsafe scenario.

Selection criteria
------------------
1. **Distance** from the drone's current position (primary criterion).
2. **Wind vector** bias: the effective cost of reaching a node is penalised by
   the headwind component, rewarding downwind or cross-wind nodes that require
   less remaining battery to reach.

All coordinates are in (latitude, longitude) WGS-84 decimal degrees.
"""

import math
from typing import Sequence

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Coordinate = tuple[float, float]  # (latitude, longitude)
WindVector = tuple[float, float]  # (north_m_s, east_m_s)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EARTH_RADIUS_M: float = 6_371_000.0  # mean Earth radius in metres


class RTKManager:
    """Manages RTK rescue-antenna node coordinates and selects the best node.

    Parameters
    ----------
    nodes : list of Coordinate, optional
        Pre-loaded list of RTK antenna coordinates.  If omitted, the manager
        initialises with an empty list; nodes can be added later via
        :meth:`add_node`.

    Examples
    --------
    >>> mgr = RTKManager(nodes=[(−35.5, −60.2), (−35.6, −60.3)])
    >>> mgr.select_nearest_node(current_position=(−35.55, −60.25))
    (−35.5, −60.2)
    """

    def __init__(self, nodes: list[Coordinate] | None = None) -> None:
        self._nodes: list[Coordinate] = list(nodes) if nodes else []

    # =========================================================================
    # Public interface
    # =========================================================================

    def add_node(self, coordinate: Coordinate) -> None:
        """Add a single RTK antenna coordinate to the managed list.

        Parameters
        ----------
        coordinate : Coordinate
            (latitude, longitude) in decimal degrees.
        """
        self._nodes.append(coordinate)

    def add_nodes(self, coordinates: Sequence[Coordinate]) -> None:
        """Add multiple RTK antenna coordinates.

        Parameters
        ----------
        coordinates : sequence of Coordinate
        """
        self._nodes.extend(coordinates)

    def select_nearest_node(
        self,
        current_position: Coordinate,
        wind_vector: WindVector = (0.0, 0.0),
        wind_bias_factor: float = 0.05,
    ) -> Coordinate:
        """Return the most energy-efficient RTK antenna node to land at.

        The effective cost of reaching each node is::

            cost = distance_m * (1 + wind_bias_factor * headwind_component)

        where ``headwind_component`` is the projection of the wind vector onto
        the direction from the drone to the node (positive = headwind,
        negative = tailwind).

        Parameters
        ----------
        current_position : Coordinate
            Current drone position (latitude, longitude).
        wind_vector : WindVector, optional
            Current wind (north_m_s, east_m_s).  Defaults to calm (0, 0).
        wind_bias_factor : float, optional
            Scaling factor for the wind-heading penalty.  Default is 0.05,
            meaning a 10 m/s headwind over 1 km adds ≈ 5 % to cost.

        Returns
        -------
        Coordinate
            The (latitude, longitude) of the selected node.

        Raises
        ------
        RuntimeError
            If no nodes have been registered.
        """
        if not self._nodes:
            raise RuntimeError(
                "RTKManager has no registered nodes. "
                "Add at least one node with add_node() before calling select_nearest_node()."
            )

        best_node: Coordinate = self._nodes[0]
        best_cost: float = float("inf")

        wind_n, wind_e = wind_vector

        for node in self._nodes:
            distance = haversine_distance(current_position, node)

            # Direction from current position toward node (unit vector, NE frame)
            direction_n, direction_e = bearing_unit_vector(current_position, node)

            # Headwind = wind component opposing the desired direction
            headwind = -(wind_n * direction_n + wind_e * direction_e)

            cost = distance * (1.0 + wind_bias_factor * headwind)

            if cost < best_cost:
                best_cost = cost
                best_node = node

        return best_node

    @property
    def nodes(self) -> list[Coordinate]:
        """Read-only view of registered node coordinates."""
        return list(self._nodes)


# =============================================================================
# Geodetic utility functions
# =============================================================================


def haversine_distance(coord_a: Coordinate, coord_b: Coordinate) -> float:
    """Compute the great-circle distance (metres) between two WGS-84 points.

    Parameters
    ----------
    coord_a, coord_b : Coordinate
        (latitude, longitude) in decimal degrees.

    Returns
    -------
    float
        Distance in metres.
    """
    lat1, lon1 = math.radians(coord_a[0]), math.radians(coord_a[1])
    lat2, lon2 = math.radians(coord_b[0]), math.radians(coord_b[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_M * c


def bearing_unit_vector(
    origin: Coordinate,
    destination: Coordinate,
) -> tuple[float, float]:
    """Return the unit vector (north, east) pointing from *origin* to *destination*.

    Uses the forward-azimuth formula on the sphere for simplicity.  For the
    short distances involved in a failsafe scenario (< 5 km), accuracy is
    better than 0.1 %.

    Parameters
    ----------
    origin, destination : Coordinate
        (latitude, longitude) in decimal degrees.

    Returns
    -------
    tuple[float, float]
        (north_component, east_component) of the unit vector.
        Returns (0, 0) when origin and destination are coincident.
    """
    lat1, lon1 = math.radians(origin[0]), math.radians(origin[1])
    lat2, lon2 = math.radians(destination[0]), math.radians(destination[1])

    dlon = lon2 - lon1

    # Forward azimuth components (standard spherical formula)
    # east  = X = sin(Δλ)·cos(φ₂)
    # north = Y = cos(φ₁)·sin(φ₂) − sin(φ₁)·cos(φ₂)·cos(Δλ)
    east = math.sin(dlon) * math.cos(lat2)
    north = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)

    magnitude = math.sqrt(north**2 + east**2)
    if magnitude < 1e-12:
        return 0.0, 0.0

    return north / magnitude, east / magnitude
