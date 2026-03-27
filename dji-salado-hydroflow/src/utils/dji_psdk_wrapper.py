"""
dji_psdk_wrapper.py
===================
Thin abstraction layer over the DJI PSDK V3 Python bindings.

In a real deployment on the Manifold 3 this module wraps the ``dji_sdk``
(or equivalent) Python package that communicates with the DJI Onboard SDK
daemon.  Here we provide a well-typed stub that:

  - Exposes the same public interface expected by :mod:`main`.
  - Raises :class:`NotImplementedError` for operations that require actual
    hardware, making the gap obvious during simulation and unit tests.
  - Can be monkey-patched in tests to return synthetic telemetry.

Public interface
----------------
.. code-block:: python

    wrapper = DJIPSDKWrapper(node=ros2_node)

    battery_pct  = wrapper.get_battery_percentage()   # float 0–100
    altitude_agl = wrapper.get_altitude_agl()          # float metres
    wind_vec     = wrapper.get_wind_vector()            # (north, east) m/s

    wrapper.goto_rtk_and_land(target_lat, target_lon)  # RTK-guided land
    wrapper.emergency_land()                            # immediate forced land
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


class DJIPSDKWrapper:
    """Interface to DJI PSDK V3 telemetry and flight-control commands.

    Parameters
    ----------
    node : rclpy.node.Node
        The parent ROS 2 node, used for logging.

    Notes
    -----
    All methods that access hardware raise :class:`NotImplementedError` when
    running without actual PSDK bindings.  Replace or mock these in
    integration tests.
    """

    def __init__(self, node: "Node") -> None:
        self._node = node
        self._logger = node.get_logger()

    # =========================================================================
    # Telemetry accessors
    # =========================================================================

    def get_battery_percentage(self) -> float:
        """Return the main battery level as a percentage (0–100).

        Reads from the PSDK ``BatteryInfo`` data subscription.

        Returns
        -------
        float
            Battery percentage.  Returns 100.0 when PSDK bindings are absent
            (simulation / unit-test mode) to avoid spurious failsafe triggers.
        """
        try:
            from dji_sdk import battery  # type: ignore[import]

            return float(battery.get_percentage())
        except ImportError:
            self._logger.debug(
                "dji_sdk not available — returning synthetic battery level 100 %"
            )
            return 100.0

    def get_altitude_agl(self) -> float:
        """Return the current altitude above ground level (metres).

        Derived from the PSDK ``AltitudeFusedWithGPS`` topic combined with
        the onboard terrain database for AGL correction.

        Returns
        -------
        float
            Altitude in metres AGL.  Returns 70.0 in simulation mode.
        """
        try:
            from dji_sdk import altitude  # type: ignore[import]

            return float(altitude.get_agl())
        except ImportError:
            return 70.0

    def get_wind_vector(self) -> tuple[float, float]:
        """Return the estimated wind vector (north_m_s, east_m_s).

        Estimated from IMU accelerations and GPS ground-speed minus air-speed
        (requires pitot tube or wind-estimation algorithm in PSDK).

        Returns
        -------
        tuple[float, float]
            ``(north_component, east_component)`` in m/s.
            Returns ``(0.0, 0.0)`` in simulation mode (calm wind assumption).
        """
        try:
            from dji_sdk import wind  # type: ignore[import]

            n, e = wind.get_vector()
            return float(n), float(e)
        except ImportError:
            return 0.0, 0.0

    # =========================================================================
    # Flight commands
    # =========================================================================

    def goto_rtk_and_land(self, target_lat: float, target_lon: float) -> None:
        """Navigate to a waypoint using RTK guidance and perform precision landing.

        Uses PSDK ``WaypointV3Mission`` with RTK position control enabled.
        The drone descends vertically at the target coordinates and lands with
        centimetric accuracy.

        Parameters
        ----------
        target_lat : float
            Target latitude in decimal degrees (WGS-84).
        target_lon : float
            Target longitude in decimal degrees (WGS-84).

        Raises
        ------
        NotImplementedError
            When PSDK hardware bindings are not available.
        """
        self._logger.info(
            f"RTK-guided landing initiated → target ({target_lat:.7f}, {target_lon:.7f})"
        )
        try:
            from dji_sdk import mission  # type: ignore[import]

            mission.goto_and_land_rtk(lat=target_lat, lon=target_lon)
        except ImportError as exc:
            raise NotImplementedError(
                "goto_rtk_and_land requires DJI PSDK hardware bindings."
            ) from exc

    def emergency_land(self) -> None:
        """Immediately initiate an emergency landing at the current position.

        Bypasses all mission logic and commands the flight controller to
        descend and land as fast as safely possible.

        Raises
        ------
        NotImplementedError
            When PSDK hardware bindings are not available.
        """
        self._logger.warning("Emergency landing commanded at current position.")
        try:
            from dji_sdk import flight_control  # type: ignore[import]

            flight_control.emergency_land()
        except ImportError as exc:
            raise NotImplementedError(
                "emergency_land requires DJI PSDK hardware bindings."
            ) from exc
