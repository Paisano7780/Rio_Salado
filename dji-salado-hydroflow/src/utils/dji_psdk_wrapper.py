"""
dji_psdk_wrapper.py — Thin Python abstraction over the DJI PSDK V3 C++ API.

This module wraps the PSDK V3 flight-control primitives exposed via the
DJI OSDK ROS 2 bridge.  In a production deployment the low-level calls are
forwarded to the C++ PSDK shared library through ctypes or a ROS 2 service
interface.  When the PSDK library is unavailable (e.g. during CI/unit tests)
the wrapper operates in *simulation mode* and logs commands without executing
them.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    # Attempt to import the PSDK bridge (available only on Manifold 3)
    import dji_psdk_bridge as _psdk_lib  # type: ignore[import]
    _PSDK_AVAILABLE = True
except ImportError:
    _psdk_lib = None  # type: ignore[assignment]
    _PSDK_AVAILABLE = False
    logger.warning(
        "dji_psdk_bridge not found — DJIPSDKWrapper running in simulation mode."
    )


class DJIPSDKWrapper:
    """High-level interface to DJI PSDK V3 flight-control functions.

    All public methods are safe to call even when the PSDK library is absent;
    in that case they log the intended command and return immediately.
    """

    def __init__(self) -> None:
        self._simulation = not _PSDK_AVAILABLE
        if self._simulation:
            logger.info("DJIPSDKWrapper: simulation mode active.")

    # ------------------------------------------------------------------
    # Flight control primitives
    # ------------------------------------------------------------------

    def set_velocity(
        self,
        v_x: float,
        v_y: float,
        v_z: float,
        yaw_rate: float = 0.0,
    ) -> None:
        """Command body-frame velocity (m/s) and yaw rate (rad/s).

        Parameters
        ----------
        v_x:
            East velocity component (m/s).
        v_y:
            North velocity component (m/s).
        v_z:
            Vertical velocity component, positive = up (m/s).
        yaw_rate:
            Yaw angular rate (rad/s).  Defaults to 0 (heading hold).
        """
        if self._simulation:
            logger.debug(
                "PSDK set_velocity: vx=%.3f vy=%.3f vz=%.3f yaw=%.4f",
                v_x, v_y, v_z, yaw_rate,
            )
            return
        _psdk_lib.flight_controller_set_velocity(v_x, v_y, v_z, yaw_rate)  # type: ignore[union-attr]

    def go_to_position(
        self,
        lat: float,
        lon: float,
        alt: float,
        speed: float = 5.0,
    ) -> None:
        """Navigate to an absolute WGS-84 position at the specified altitude.

        Parameters
        ----------
        lat:
            Target latitude in decimal degrees.
        lon:
            Target longitude in decimal degrees.
        alt:
            Target altitude in metres (orthometric / MSL).
        speed:
            Approach speed in m/s (default 5 m/s).
        """
        if self._simulation:
            logger.info(
                "PSDK go_to_position: lat=%.7f lon=%.7f alt=%.2f speed=%.1f",
                lat, lon, alt, speed,
            )
            return
        _psdk_lib.waypoint_v3_go_to(lat, lon, alt, speed)  # type: ignore[union-attr]

    def land(self, verify_ground: bool = True) -> None:
        """Initiate autonomous precision landing at the current position.

        Parameters
        ----------
        verify_ground:
            When ``True`` (default) the PSDK will engage downward vision
            sensors for centimetre-level touchdown accuracy.
        """
        if self._simulation:
            logger.info("PSDK land: verify_ground=%s", verify_ground)
            return
        _psdk_lib.flight_controller_land(verify_ground)  # type: ignore[union-attr]

    def return_to_home(self) -> None:
        """Trigger return-to-home sequence using stored home point."""
        if self._simulation:
            logger.info("PSDK return_to_home triggered.")
            return
        _psdk_lib.flight_controller_return_to_home()  # type: ignore[union-attr]

    def get_battery_percentage(self) -> Optional[float]:
        """Query current battery state of charge in percent (0–100).

        Returns
        -------
        float or None
            Battery percentage, or ``None`` in simulation mode / on error.
        """
        if self._simulation:
            logger.debug("PSDK get_battery_percentage: simulation mode, returning None.")
            return None
        try:
            return float(_psdk_lib.battery_get_percentage())  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to read battery percentage: %s", exc)
            return None

    def set_cruise_altitude(self, altitude_agl: float) -> None:
        """Set the autopilot's target altitude above ground level.

        Parameters
        ----------
        altitude_agl:
            Desired altitude in metres AGL.
        """
        if self._simulation:
            logger.info("PSDK set_cruise_altitude: %.2f m AGL", altitude_agl)
            return
        _psdk_lib.flight_controller_set_altitude(altitude_agl)  # type: ignore[union-attr]

    @property
    def is_simulation(self) -> bool:
        """``True`` when running without a physical PSDK connection."""
        return self._simulation
