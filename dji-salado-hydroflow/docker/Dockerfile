# syntax=docker/dockerfile:1
# =============================================================================
# AI-HydroFlow Salado — Docker image for DJI Manifold 3 (ARM64 / Jetson Orin)
# Base: NVIDIA L4T r35.2.1  →  JetPack 5.1 / CUDA 11.4
# =============================================================================

FROM nvcr.io/nvidia/l4t-base:r35.2.1

LABEL maintainer="AI-HydroFlow Salado <hydro@salado.ar>" \
      description="ROS 2 Foxy + OpenCV CUDA + PSDK V3 runtime for DJI Manifold 3" \
      version="1.0.0"

# ---------------------------------------------------------------------------
# 0. Environment
# ---------------------------------------------------------------------------
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8 \
    ROS_DISTRO=foxy \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ---------------------------------------------------------------------------
# 1. System prerequisites
# ---------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        locales \
        curl \
        gnupg2 \
        lsb-release \
        ca-certificates \
        software-properties-common \
        build-essential \
        cmake \
        git \
        pkg-config \
        wget \
        unzip \
        # Python 3.8 (default on Ubuntu 20.04 / L4T r35)
        python3 \
        python3-pip \
        python3-dev \
        python3-setuptools \
        python3-wheel \
        # Networking / time sync
        ntp \
        # YAML / JSON parsing for ROS 2 params
        libyaml-dev \
    && locale-gen en_US en_US.UTF-8 \
    && update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 2. ROS 2 Foxy (Jetson / ARM64 binary packages from OSRF)
# ---------------------------------------------------------------------------
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=arm64 signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu focal main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
        ros-foxy-ros-base \
        ros-foxy-sensor-msgs \
        ros-foxy-geometry-msgs \
        ros-foxy-std-msgs \
        ros-foxy-nav-msgs \
        python3-colcon-common-extensions \
        python3-rosdep \
    && rosdep init \
    && rosdep update \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 3. OpenCV 4.x with CUDA support (pre-built for L4T r35 from JetPack)
# ---------------------------------------------------------------------------
# The NVIDIA L4T base image already ships libopencv with CUDA via the
# nvidia-l4t-jetson-multimedia-api package.  We install the Python bindings.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopencv-dev \
        python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# ---------------------------------------------------------------------------
# 4. Python dependencies (requirements.txt)
# ---------------------------------------------------------------------------
WORKDIR /opt/hydroflow

COPY requirements.txt .

RUN pip3 install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------------
# 5. DJI PSDK V3 Python bindings (installed from local SDK tarball)
#    Place the extracted SDK under docker/psdk_v3/ before building the image.
#    The COPY line is intentionally commented-out for CI where the SDK is not
#    available; uncomment for real deployment builds.
# ---------------------------------------------------------------------------
# COPY docker/psdk_v3 /opt/psdk_v3
# RUN cd /opt/psdk_v3 && pip3 install --no-cache-dir .

# ---------------------------------------------------------------------------
# 6. Application source
# ---------------------------------------------------------------------------
COPY src/ /opt/hydroflow/src/
COPY tests/ /opt/hydroflow/tests/

# ---------------------------------------------------------------------------
# 7. ROS 2 environment sourcing
# ---------------------------------------------------------------------------
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/foxy/setup.bash" >> /etc/bash.bashrc

ENV PYTHONPATH="/opt/hydroflow/src:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# 8. Health check: validate Python + NumPy + SciPy import
# ---------------------------------------------------------------------------
RUN python3 -c "import numpy; import scipy; print('NumPy', numpy.__version__, '/ SciPy', scipy.__version__, '— OK')"

# ---------------------------------------------------------------------------
# 9. Default command: run unit tests (useful for CI)
# ---------------------------------------------------------------------------
CMD ["bash", "-c", "source /opt/ros/foxy/setup.bash && pytest /opt/hydroflow/tests/unit -v"]
