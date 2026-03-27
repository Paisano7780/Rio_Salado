"""
main.py
=======
ROS 2 core node — HydroFlowPilot.

Orchestrates:
  - Sensor subscription (RTK position, LiDAR point cloud, thermal image).
  - Hydrodynamic processing via hydro_logic.py.
  - Velocity command publication to the DJI PSDK interface.
  - Battery failsafe with centimetric RTK-guided autonomous landing.

Execution
---------
Run as a standard ROS 2 Python node::

    ros2 run dji_salado_hydroflow hydroflow_pilot

or with explicit parameters::

    ros2 run dji_salado_hydroflow hydroflow_pilot \
        --ros-args -p target_altitude_agl:=70.0 -p failsafe_battery_pct:=20.0
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

# ROS 2 message types
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, NavSatFix, PointCloud2
from std_msgs.msg import String

# Local modules
from hydro_logic import apply_gaussian_filter, calculate_flow_vector
from utils.dji_psdk_wrapper import DJIPSDKWrapper
from utils.rtk_manager import RTKManager

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_ALTITUDE_AGL: float = 70.0   # metres above ground level
FAILSAFE_BATTERY_PCT: float = 20.0  # trigger autonomous return/land
LOOP_RATE_HZ: int = 10              # main processing loop frequency
DTM_ROWS: int = 100                 # raster grid height (cells)
DTM_COLS: int = 100                 # raster grid width  (cells)
GAUSSIAN_SIGMA: float = 2.0         # spatial smoothing parameter


class HydroFlowPilot(Node):
    """ROS 2 node that guides the DJI Matrice 350 RTK along hydraulic flow paths.

    The node fuses LiDAR-derived terrain data with thermal water detection to
    compute the steepest-descent flow direction and publishes corresponding
    velocity commands to the PSDK interface.  A built-in battery failsafe
    triggers centimetric RTK-guided autonomous landing at the nearest rescue
    antenna when battery drops to or below ``failsafe_battery_pct``.
    """

    def __init__(self) -> None:
        super().__init__("hydroflow_pilot")

        # ---- ROS 2 parameters ------------------------------------------------
        self.declare_parameter("target_altitude_agl", TARGET_ALTITUDE_AGL)
        self.declare_parameter("failsafe_battery_pct", FAILSAFE_BATTERY_PCT)
        self.declare_parameter("gaussian_sigma", GAUSSIAN_SIGMA)

        self._target_alt: float = self.get_parameter("target_altitude_agl").value
        self._failsafe_pct: float = self.get_parameter("failsafe_battery_pct").value
        self._sigma: float = self.get_parameter("gaussian_sigma").value

        # ---- Internal state --------------------------------------------------
        self._latest_rtk: NavSatFix | None = None
        self._latest_cloud: PointCloud2 | None = None
        self._latest_thermal: Image | None = None
        self._battery_pct: float = 100.0
        self._failsafe_active: bool = False

        # ---- External helpers ------------------------------------------------
        self._psdk = DJIPSDKWrapper(node=self)
        self._rtk_manager = RTKManager()

        # ---- QoS profile for sensor data ------------------------------------
        sensor_qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # ---- Subscribers -----------------------------------------------------
        self.create_subscription(
            NavSatFix,
            "/dji_osdk/rtk_position",
            self._rtk_callback,
            sensor_qos,
        )
        self.create_subscription(
            PointCloud2,
            "/dji_osdk/lidar_pointcloud",
            self._lidar_callback,
            sensor_qos,
        )
        self.create_subscription(
            Image,
            "/dji_osdk/thermal_image",
            self._thermal_callback,
            sensor_qos,
        )

        # ---- Publishers ------------------------------------------------------
        self._vel_pub = self.create_publisher(Twist, "/hydroflow/velocity_cmd", 10)
        self._status_pub = self.create_publisher(String, "/hydroflow/status", 10)

        # ---- Main processing timer -------------------------------------------
        self.create_timer(1.0 / LOOP_RATE_HZ, self._processing_loop)

        self.get_logger().info(
            f"HydroFlowPilot started — target AGL: {self._target_alt} m, "
            f"failsafe at: {self._failsafe_pct} %"
        )

    # =========================================================================
    # Subscriber callbacks
    # =========================================================================

    def _rtk_callback(self, msg: NavSatFix) -> None:
        self._latest_rtk = msg

    def _lidar_callback(self, msg: PointCloud2) -> None:
        self._latest_cloud = msg

    def _thermal_callback(self, msg: Image) -> None:
        self._latest_thermal = msg

    # =========================================================================
    # Main processing loop (10 Hz)
    # =========================================================================

    def _processing_loop(self) -> None:
        """Execute one cycle of the hydrodynamic navigation pipeline."""
        # Update battery state from PSDK
        self._battery_pct = self._psdk.get_battery_percentage()

        # Check failsafe condition first
        if self._battery_pct <= self._failsafe_pct and not self._failsafe_active:
            self._trigger_failsafe()
            return

        if self._failsafe_active:
            return  # Failsafe handler owns the vehicle now

        # Require all sensor data before processing
        if (
            self._latest_rtk is None
            or self._latest_cloud is None
            or self._latest_thermal is None
        ):
            self.get_logger().warn("Waiting for sensor data...", throttle_duration_sec=5)
            return

        # --- 1. Build DTM from LiDAR point cloud ----------------------------
        dtm = self._build_dtm(self._latest_cloud)

        # --- 2. Smooth DTM with Gaussian filter -----------------------------
        smoothed_dtm = apply_gaussian_filter(dtm, sigma=self._sigma)

        # --- 3. Segment water mask from thermal image -----------------------
        water_mask = self._segment_water_mask(self._latest_thermal)

        # --- 4. Compute flow vector field -----------------------------------
        flow_vectors = calculate_flow_vector(smoothed_dtm, water_mask)

        # --- 5. Extract dominant flow direction at drone's grid position ----
        v_x, v_y = self._extract_dominant_vector(flow_vectors)

        # --- 6. Maintain altitude and publish velocity command --------------
        altitude_error = self._target_alt - self._psdk.get_altitude_agl()
        v_z = float(np.clip(altitude_error * 0.5, -2.0, 2.0))

        twist = Twist()
        twist.linear.x = float(v_x)
        twist.linear.y = float(v_y)
        twist.linear.z = v_z
        self._vel_pub.publish(twist)

        # --- 7. Publish status ----------------------------------------------
        status_msg = String()
        status_msg.data = (
            f"OK | bat={self._battery_pct:.1f}% | "
            f"v_x={v_x:.3f} v_y={v_y:.3f} v_z={v_z:.3f}"
        )
        self._status_pub.publish(status_msg)

    # =========================================================================
    # Sensor processing helpers
    # =========================================================================

    def _build_dtm(self, cloud_msg: PointCloud2) -> np.ndarray:
        """Convert a PointCloud2 message to a 2-D DTM raster (stub).

        In production this would use ``ros2_numpy`` or a custom C extension to
        extract Z values and interpolate them onto the raster grid.  Here we
        return a synthetic ramp for compilation / unit-test purposes.
        """
        dtm = np.zeros((DTM_ROWS, DTM_COLS), dtype=np.float64)
        for r in range(DTM_ROWS):
            for c in range(DTM_COLS):
                dtm[r, c] = r * 0.01 + c * 0.005  # gentle synthetic slope
        return dtm

    def _segment_water_mask(self, thermal_msg: Image) -> np.ndarray:
        """Segment water pixels from a thermal Image message (stub).

        Production implementation: convert LWIR 8-bit frame to NumPy array via
        ``cv_bridge``, apply adaptive Otsu thresholding, and morphologically
        close the resulting binary mask.
        """
        mask = np.ones((DTM_ROWS, DTM_COLS), dtype=bool)
        mask[: DTM_ROWS // 4, :] = False  # synthetic: no water in upper quarter
        return mask

    def _extract_dominant_vector(self, flow_vectors: np.ndarray) -> tuple[float, float]:
        """Return the mean non-zero flow vector across all water cells."""
        v_row = flow_vectors[0]
        v_col = flow_vectors[1]

        nonzero = (v_row != 0) | (v_col != 0)
        if not nonzero.any():
            return 0.0, 0.0

        mean_row = float(np.mean(v_row[nonzero]))
        mean_col = float(np.mean(v_col[nonzero]))

        mag = math.sqrt(mean_row**2 + mean_col**2)
        if mag < 1e-9:
            return 0.0, 0.0

        return mean_row / mag, mean_col / mag

    # =========================================================================
    # Failsafe
    # =========================================================================

    def _trigger_failsafe(self) -> None:
        """Initiate battery-low failsafe: navigate to nearest RTK node and land."""
        self._failsafe_active = True
        self.get_logger().warn(
            f"Battery at {self._battery_pct:.1f}% — initiating failsafe landing."
        )

        if self._latest_rtk is None:
            self.get_logger().error("No RTK fix — forcing immediate landing.")
            self._psdk.emergency_land()
            return

        current_pos = (self._latest_rtk.latitude, self._latest_rtk.longitude)
        wind_vector = self._psdk.get_wind_vector()

        nearest_node = self._rtk_manager.select_nearest_node(
            current_position=current_pos,
            wind_vector=wind_vector,
        )

        status_msg = String()
        status_msg.data = (
            f"FAILSAFE | Returning to RTK node at "
            f"{nearest_node[0]:.6f}, {nearest_node[1]:.6f}"
        )
        self._status_pub.publish(status_msg)
        self.get_logger().info(status_msg.data)

        self._psdk.goto_rtk_and_land(
            target_lat=nearest_node[0],
            target_lon=nearest_node[1],
        )


# =============================================================================
# Entry point
# =============================================================================


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HydroFlowPilot()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down HydroFlowPilot.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
