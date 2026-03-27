# AI-HydroFlow Salado — Engineering Specifications

## Objective

Real-time tracking of surface water runoff in the **Cuenca del Salado** (Buenos Aires Province, Argentina).  
The system uses a DJI Matrice 350 RTK drone equipped with a DJI Manifold 3 onboard computer to autonomously
follow the path of surface-water flow across flat floodplains, delivering centimetric-precision georeferenced
data to field operators.

---

## 1. Hardware Platform

| Component             | Model / Specification                                 |
|-----------------------|-------------------------------------------------------|
| Airframe              | DJI Matrice 350 RTK                                   |
| Onboard Computer      | DJI Manifold 3 (NVIDIA Orin NX, ARM64, 16 GB LPDDR5) |
| LiDAR                 | DJI Zenmuse L2 (Multi-return, 240,000 pts/s)          |
| Thermal / Optical     | DJI Zenmuse H30T (640×512 LWIR + 48 MP RGB)           |
| RTK Reference Network | D-RTK 2 Mobile Station + NTRIP correction feed        |
| Obstacle Detection    | DJI CSM Radar (360° azimuth, 30 m range)              |
| GNSS                  | L1/L2/L5 multi-constellation RTK (± 1 cm + 1 ppm)    |

---

## 2. Sensor Fusion Architecture

### 2.1 LiDAR L2 → Digital Terrain Model (DTM)

1. Raw point cloud acquired at 10 Hz via PSDK V3 `LidarPointCloud` topic.
2. Ground points classified with a Cloth Simulation Filter (CSF).
3. Interpolated to a 0.5 m × 0.5 m raster (IDW).
4. **Gaussian filter** (σ = 2.0, kernel 5×5) applied to smooth micro-relief artefacts typical of flat
   Argentine pampas terrain, exposing the true hydraulic slope.

### 2.2 Thermal Camera H30T → Water Mask

1. 8-bit LWIR frames acquired at 30 Hz.
2. Water pixels segmented via adaptive Otsu thresholding (water ≈ 2–4 °C colder than surrounding soil).
3. Binary mask morphologically closed (3×3 kernel) to remove salt-and-pepper noise.
4. Mask co-registered to DTM raster using RTK timestamp synchronisation.

### 2.3 Thermo-Topographic Flow Gradient

The navigation direction vector **v** is computed as:

```
∇z  = gradient(smoothed_DTM)          # steepest descent
w   = water_mask ∈ {0, 1}             # binary water presence
v   = -normalise(∇z) * w              # follow gravity where water exists
```

The negative gradient ensures the drone moves toward lower elevations (hydraulic flow direction).

---

## 3. Key Innovation: Gaussian Smoothing for Flat-Terrain Navigation

Standard gradient-based flow algorithms diverge on near-flat surfaces (slope < 0.1 %) due to
sensor noise.  The Gaussian pre-filter effectively acts as a low-pass spatial filter:

- Suppresses LiDAR range noise (σ_range ≈ 2 cm at 50 m).
- Preserves macro-hydrological features (natural channels, micro-depressions > 5 cm).
- Eliminates numerical artefacts that would cause oscillatory waypoint generation.

---

## 4. Flight Parameters

| Parameter              | Value              |
|------------------------|--------------------|
| Cruise altitude (AGL)  | 70 m               |
| Cruise speed           | 8 m/s              |
| RTK mode               | Fix (< 2 cm CEP)   |
| LiDAR scan angle       | ±30° (nadir)       |
| Thermal frame rate     | 30 Hz              |
| Main loop rate         | 10 Hz              |

---

## 5. Safety & Failsafe Systems

### 5.1 Active Obstacle Avoidance
- **Optical sensors** (6-direction ToF + stereo): obstacles < 5 m trigger emergency stop.
- **CSM Radar**: dedicated detection of thin obstacles (power lines, cables) in a 30 m radius.
  Firmware integration via `OmnidirectionalObstacleAvoidance` PSDK interface.

### 5.2 Battery Failsafe
| Battery Level | Action                                                                          |
|---------------|---------------------------------------------------------------------------------|
| 30 %          | Warning published on `/hydroflow/status`; operator notified via data-link.      |
| 20 %          | `RTKManager.select_nearest_node()` invoked; autonomous RTK-guided return/land.  |
| 15 %          | Emergency forced landing at current position.                                   |

### 5.3 RTK Integrity Monitor
- Continuous PDOP check; if PDOP > 3.0, switch to visual-inertial odometry (VIO) fallback.
- Minimum 6 satellites required for RTK Fix; below this, hover-in-place until signal restored.

---

## 6. Data Products

| Product             | Format          | Resolution  | Update Rate |
|---------------------|-----------------|-------------|-------------|
| Georeferenced DTM   | GeoTIFF (EPSG:22174) | 0.5 m  | Per flight  |
| Flow Direction Map  | GeoTIFF + Shapefile  | 0.5 m  | Per flight  |
| Thermal Ortho       | GeoTIFF         | 5 cm        | Per flight  |
| Telemetry Log       | ROS 2 bag       | —           | Continuous  |

---

## 7. Software Architecture

```
HydroFlowPilot (ROS 2 Node)
  │
  ├── Subscribers
  │     ├── /dji_osdk/rtk_position       → sensor_msgs/NavSatFix
  │     ├── /dji_osdk/lidar_pointcloud   → sensor_msgs/PointCloud2
  │     └── /dji_osdk/thermal_image      → sensor_msgs/Image
  │
  ├── Processing Pipeline (10 Hz)
  │     ├── build_dtm()                  ← point cloud → raster
  │     ├── apply_gaussian_filter()      ← smooth DTM
  │     ├── segment_water_mask()         ← thermal → binary mask
  │     └── calculate_flow_vector()      ← gradient → unit vector
  │
  └── Publishers / Commands
        ├── /hydroflow/velocity_cmd      → geometry_msgs/Twist
        └── /hydroflow/status            → std_msgs/String
```

---

## 8. CI/CD Pipeline

A GitHub Actions workflow (`.github/workflows/ci_cd_manifold.yml`) uses `docker/setup-qemu-action`
to emulate the ARM64 target architecture and:

1. Builds the Docker image based on `nvcr.io/nvidia/l4t-base:r35.2.1`.
2. Executes `pytest` against unit tests to validate:
   - Gaussian filter output shape and value range.
   - Flow-vector normalisation (unit length).
   - RTK node selection correctness.

---

*Document version: 1.0.0 — AI-HydroFlow Salado Project*
