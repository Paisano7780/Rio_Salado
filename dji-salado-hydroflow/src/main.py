"""
main.py — ROS 2 HydroFlowPilot node for AI-HydroFlow Salado.

Responsibilities
----------------
* Subscribe to RTK position, LiDAR point-cloud and thermal image topics.
* On each sensor cycle: smooth the DTM, compute flow direction, publish
  velocity commands via the DJI PSDK wrapper.
* Maintain constant cruise altitude of 70 m AGL.
* Trigger autonomous centimetre-precision landing when battery ≤ 20 %.
"""

from __future__ import annotations

import sys
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# ROS 2 message types (standard + DJI OSDK bridge)
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import PointCloud2, Image
from std_msgs.msg import Float32, String

from hydro_logic import apply_gaussian_filter, calculate_flow_vector
from utils.rtk_manager import RTKManager
from utils.dji_psdk_wrapper import DJIPSDKWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CRUISE_ALTITUDE_AGL: float = 70.0   # metres above ground level
BATTERY_FAILSAFE_PCT: float = 20.0  # percent — trigger autonomous landing
DTM_SIGMA: float = 2.0              # Gaussian smoothing σ for DTM
DTM_ROWS: int = 200                 # synthetic DTM grid rows (placeholder)
DTM_COLS: int = 200                 # synthetic DTM grid cols (placeholder)


class HydroFlowPilot(Node):
    """ROS 2 node that fuses LiDAR, thermal and RTK data to navigate
    along the dominant water-runoff direction in the Cuenca del Salado."""

    def __init__(self) -> None:
        super().__init__("hydro_flow_pilot")

        # -- QoS profile for sensor topics (best-effort, keep-last 1) -------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # -- Subscribers ----------------------------------------------------
        self._rtk_sub = self.create_subscription(
            Vector3,
            "/dji_osdk/rtk_position",
            self._rtk_callback,
            sensor_qos,
        )
        self._lidar_sub = self.create_subscription(
            PointCloud2,
            "/dji_osdk/lidar_pointcloud",
            self._lidar_callback,
            sensor_qos,
        )
        self._thermal_sub = self.create_subscription(
            Image,
            "/dji_osdk/thermal_image",
            self._thermal_callback,
            sensor_qos,
        )
        self._battery_sub = self.create_subscription(
            Float32,
            "/dji_osdk/battery_percentage",
            self._battery_callback,
            sensor_qos,
        )

        # -- Publisher ------------------------------------------------------
        self._vel_pub = self.create_publisher(
            Vector3,
            "/dji_osdk/velocity_cmd",
            10,
        )

        # -- Internal state -------------------------------------------------
        self._rtk_position: Optional[np.ndarray] = None   # [lat, lon, alt]
        self._dtm: Optional[np.ndarray] = None            # raw DTM grid
        self._water_mask: Optional[np.ndarray] = None     # binary water mask
        self._battery_pct: float = 100.0
        self._failsafe_triggered: bool = False

        # -- Helper objects -------------------------------------------------
        self._rtk_manager = RTKManager()
        self._psdk = DJIPSDKWrapper()

        # -- Main control loop at 10 Hz ------------------------------------
        self._timer = self.create_timer(0.1, self._control_loop)

        self.get_logger().info("HydroFlowPilot node initialised.")

    # -----------------------------------------------------------------------
    # Subscriber callbacks
    # -----------------------------------------------------------------------

    def _rtk_callback(self, msg: Vector3) -> None:
        """Store latest RTK position [lat°, lon°, alt_m]."""
        self._rtk_position = np.array([msg.x, msg.y, msg.z])

    def _lidar_callback(self, msg: PointCloud2) -> None:
        """Convert point-cloud to a simplified DTM grid.

        In production this step runs the Progressive Morphological Filter and
        Kriging interpolation.  Here we decode the raw bytes into a flat
        elevation array and reshape to the configured grid dimensions.
        """
        # Decode raw float32 z-values from the PointCloud2 message.
        raw = np.frombuffer(msg.data, dtype=np.float32)
        if raw.size == 0:
            return

        # Tile/trim to DTM_ROWS × DTM_COLS for processing.
        total = DTM_ROWS * DTM_COLS
        if raw.size < total:
            raw = np.tile(raw, int(np.ceil(total / raw.size)))
        self._dtm = raw[:total].reshape(DTM_ROWS, DTM_COLS).astype(float)

    def _thermal_callback(self, msg: Image) -> None:
        """Derive binary water mask from thermal image.

        Pixels whose 16-bit radiometric value exceeds the empirical water
        emissivity threshold are flagged as water (True).
        """
        raw = np.frombuffer(msg.data, dtype=np.uint16)
        if raw.size == 0:
            return

        total = DTM_ROWS * DTM_COLS
        if raw.size < total:
            raw = np.tile(raw, int(np.ceil(total / raw.size)))
        thermal = raw[:total].reshape(DTM_ROWS, DTM_COLS).astype(float)

        # Adaptive threshold: water pixels are cooler than surrounding terrain
        threshold = float(np.median(thermal)) - float(np.std(thermal)) * 0.5
        self._water_mask = thermal < threshold

    def _battery_callback(self, msg: Float32) -> None:
        """Update battery state and arm failsafe when below threshold."""
        self._battery_pct = float(msg.data)
        if self._battery_pct <= BATTERY_FAILSAFE_PCT and not self._failsafe_triggered:
            self.get_logger().warning(
                f"Battery at {self._battery_pct:.1f}% — initiating failsafe landing."
            )
            self._trigger_failsafe()

    # -----------------------------------------------------------------------
    # Failsafe
    # -----------------------------------------------------------------------

    def _trigger_failsafe(self) -> None:
        """Select nearest RTK rescue node and execute autonomous landing."""
        self._failsafe_triggered = True
        if self._rtk_position is None:
            self.get_logger().error("RTK position unavailable; cannot perform safe landing.")
            return

        landing_node = self._rtk_manager.select_nearest_node(
            current_position=self._rtk_position,
        )
        self.get_logger().info(
            f"Landing at RTK node: lat={landing_node[0]:.7f}, lon={landing_node[1]:.7f}"
        )
        self._psdk.go_to_position(
            lat=landing_node[0],
            lon=landing_node[1],
            alt=landing_node[2],  # use the surveyed orthometric altitude of the node
        )
        self._psdk.land()

    # -----------------------------------------------------------------------
    # Main control loop
    # -----------------------------------------------------------------------

    def _control_loop(self) -> None:
        """10 Hz guidance cycle: compute flow vector → publish velocity cmd."""
        if self._failsafe_triggered:
            return

        if self._dtm is None or self._water_mask is None:
            self.get_logger().debug("Waiting for sensor data …")
            return

        # 1. Smooth DTM with Gaussian filter
        smoothed = apply_gaussian_filter(self._dtm, sigma=DTM_SIGMA)

        # 2. Compute flow direction unit vector [v_east, v_north]
        flow_vec = calculate_flow_vector(smoothed, self._water_mask)

        # 3. Scale to a safe cruise speed (m/s) — adjust as needed
        cruise_speed: float = 3.0  # m/s horizontal
        v_x = float(flow_vec[0]) * cruise_speed
        v_y = float(flow_vec[1]) * cruise_speed

        # 4. Altitude hold: compute Δz to maintain CRUISE_ALTITUDE_AGL
        current_alt = float(self._rtk_position[2]) if self._rtk_position is not None else CRUISE_ALTITUDE_AGL
        v_z = float(np.clip(CRUISE_ALTITUDE_AGL - current_alt, -1.0, 1.0))

        # 5. Publish velocity command
        cmd = Vector3()
        cmd.x = v_x   # East
        cmd.y = v_y   # North
        cmd.z = v_z   # Up (altitude correction)
        self._vel_pub.publish(cmd)

        self.get_logger().debug(
            f"Flow cmd: v_x={v_x:.3f} v_y={v_y:.3f} v_z={v_z:.3f} "
            f"bat={self._battery_pct:.1f}%"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None) -> None:
    rclpy.init(args=args)
    node = HydroFlowPilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutdown requested.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
