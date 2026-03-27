# AI-HydroFlow Salado — ARM64 Docker image for DJI Manifold 3
#
# Base: NVIDIA L4T (Linux for Tegra) r35.2.1 — compatible with Manifold 3
# Target: ARM64 / aarch64

FROM nvcr.io/nvidia/l4t-base:r35.2.1

LABEL maintainer="AI-HydroFlow Salado Team"
LABEL description="ROS 2 Foxy + PSDK V3 environment for DJI Manifold 3 (ARM64)"

# ---------------------------------------------------------------------------
# System dependencies
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gnupg2 \
        lsb-release \
        locales \
        build-essential \
        cmake \
        git \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        libssl-dev \
        libusb-1.0-0-dev \
        udev \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# ROS 2 Foxy (ARM64 apt repository)
# ---------------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu focal main" \
        > /etc/apt/sources.list.d/ros2-latest.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-foxy-ros-base \
        ros-foxy-sensor-msgs \
        ros-foxy-geometry-msgs \
        ros-foxy-std-msgs \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# Python scientific stack and project dependencies
# ---------------------------------------------------------------------------
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

# ---------------------------------------------------------------------------
# Application source
# ---------------------------------------------------------------------------
WORKDIR /ros2_ws/src/rio_salado
COPY src/ /ros2_ws/src/rio_salado/src/
COPY tests/ /ros2_ws/src/rio_salado/tests/

# Ensure the src package is importable
ENV PYTHONPATH="/ros2_ws/src/rio_salado/src:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# ROS 2 environment
# ---------------------------------------------------------------------------
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc

# ---------------------------------------------------------------------------
# Default command — run unit tests
# ---------------------------------------------------------------------------
CMD ["pytest", "/ros2_ws/src/rio_salado/tests/unit/test_hydro_logic.py", "--tb=short", "-v"]
