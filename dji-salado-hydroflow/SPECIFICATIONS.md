# AI-HydroFlow Salado — Engineering Specifications

## 1. Project Overview

**System Name:** AI-HydroFlow Salado  
**Platform:** DJI Matrice 350 RTK + DJI Manifold 3 (ARM64)  
**Objective:** Real-time autonomous monitoring and tracking of water runoff patterns in the Cuenca del Salado (Buenos Aires Province, Argentina). The system maps surface flow direction at centimetre-level precision, enabling early-warning flood modelling and environmental assessment.

---

## 2. Hardware Architecture

| Component | Model | Role |
|-----------|-------|------|
| UAV | DJI Matrice 350 RTK | Aerial platform (IP55, 38 min endurance) |
| Onboard Computer | DJI Manifold 3 | Edge inference (ARM64, NVIDIA Jetson Orin) |
| LiDAR | DJI Zenmuse L2 | High-density point cloud for DTM generation |
| Thermal Camera | DJI Zenmuse H30T | Thermal segmentation of water surfaces |
| GNSS | Integrated D-RTK 2 network | Centimetre-level positioning |
| Obstacle Avoidance | Omnidirectional optical + Radar CSM | Active collision avoidance |

---

## 3. Sensor Fusion Pipeline

### 3.1 Digital Terrain Model (DTM) — LiDAR L2
1. Raw point cloud ingested via PSDK V3 topic `/dji_osdk/lidar_pointcloud`.
2. Ground points extracted using Progressive Morphological Filter (PMF).
3. Interpolated to a 0.25 m resolution raster using Kriging.
4. **Gaussian Smoothing** (σ = 2.0 m) applied to eliminate micro-relief noise inherent to flat pampean terrain, revealing the true macro-slope gradient.

### 3.2 Thermal Water Segmentation — H30T
1. 14-bit radiometric LWIR image published on `/dji_osdk/thermal_image`.
2. Adaptive thresholding isolates pixels whose emissivity matches standing/flowing water (ε ≈ 0.98).
3. Morphological closing removes isolated hot/cold pixels.
4. Binary `water_mask` fed into the flow-vector calculation.

---

## 4. Core Algorithm — Thermo-Topographic Gradient Navigation

### 4.1 Gaussian DTM Smoothing
```
Z_smooth = G_σ * Z_raw
```
where `G_σ` is a 2-D Gaussian kernel with σ = 2.0.  
Preserves macro-drainage structures while suppressing LiDAR noise on slopes < 0.3 %.

### 4.2 Flow Vector Calculation
The numerical gradient ∇z of the smoothed DTM defines downhill flow direction:
```
∇z = (∂Z/∂x, ∂Z/∂y)
flow_direction = -∇z / ‖∇z‖      (unit vector pointing downhill)
```
The water mask gates the gradient so only pixels confirmed as water contribute to the commanded velocity vector. This prevents erroneous navigation over dry terrain that mimics a drainage signature.

### 4.3 Thermo-Topographic Fusion
Final navigation bearing blends topographic gradient (weight 0.7) with thermal-gradient bearing (weight 0.3), providing robustness when the LiDAR DTM is momentarily occluded by dense vegetation.

---

## 5. ROS 2 Node Architecture

```
/dji_osdk/rtk_position      ──►┐
/dji_osdk/lidar_pointcloud  ──►│  HydroFlowPilot Node  ──► /dji_osdk/velocity_cmd
/dji_osdk/thermal_image     ──►┘
```

**Cruise altitude:** 70 m AGL (constant barometric + RTK hold)  
**Update rate:** 10 Hz  
**Coordinate frame:** WGS-84 / ENU local frame

---

## 6. Safety & Failsafe System

| Condition | Action |
|-----------|--------|
| Battery ≤ 20 % | `rtk_manager.select_nearest_node()` → autonomous centimetre-precision landing |
| Obstacle detected (< 5 m) | Lateral avoidance manoeuvre + resume heading |
| Power-line / cable radar alert | 50 m altitude gain + waypoint reroute |
| RTK signal lost (> 5 s) | Hover + broadcast distress beacon |
| Geofence breach | Immediate return-to-home |

### 6.1 RTK Rescue Node Network
Portable D-RTK 2 base stations are pre-surveyed and stored as WGS-84 coordinates. `rtk_manager.select_nearest_node()` selects the optimal landing node using:
```
score(n) = 1 / distance(current, n)  ×  cos(θ_wind, bearing(current→n))
```
maximising both proximity and tailwind advantage to minimise battery consumption during return.

---

## 7. CI/CD & Deployment

- Docker image based on `nvcr.io/nvidia/l4t-base:r35.2.1` (ARM64).
- GitHub Actions workflow emulates ARM64 via QEMU, builds the image, and runs pytest.
- Production deployment: `docker push` to private registry → OTA pull on Manifold 3 via LTE link.

---

## 8. Coordinate Reference System
- **Horizontal:** WGS-84 (EPSG:4326) acquired via D-RTK 2 network.
- **Vertical:** EGM2008 geoid model for orthometric heights.
- **Local frame:** ENU (East-North-Up) for velocity commands.

---

## 9. Glossary

| Term | Definition |
|------|------------|
| DTM | Digital Terrain Model — bare-earth elevation raster |
| AGL | Above Ground Level |
| PSDK | Payload Software Development Kit (DJI) |
| RTK | Real-Time Kinematic GNSS |
| LiDAR | Light Detection and Ranging |
| LWIR | Long-Wave Infrared (thermal) |
| PMF | Progressive Morphological Filter |
| CSM | Collision Sensing Module (DJI radar) |
