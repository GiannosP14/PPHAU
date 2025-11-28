#!/usr/bin/env bash
# ros_colima_run.sh
# Usage: ./ros_colima_run.sh your_file.bag
# Automates: start colima, ensure XQuartz access, find Colima IP, run ROS container with DISPLAY set.
# Works with ROS Noetic (ROS1). For ROS2, change image to osrf/ros:humble-desktop and ros commands accordingly.

set -euo pipefail

BAGFILE="${1:-}"
if [[ -z "$BAGFILE" ]]; then
  echo "Usage: $0 path/to/your_file.bag"
  exit 2
fi

if [[ ! -f "$BAGFILE" ]]; then
  echo "File not found: $BAGFILE"
  exit 3
fi

# Display bag file information if found
echo "========================================="
echo "Bag file found: $BAGFILE"
echo "========================================="
echo "File size: $(du -h "$BAGFILE" | cut -f1)"
echo "Full path: $(cd "$(dirname "$BAGFILE")" && pwd)/$(basename "$BAGFILE")"
echo "========================================="

# 1) Ensure Colima & docker CLI installed
if ! command -v colima >/dev/null 2>&1; then
  echo "colima not found. Install with: brew install colima docker"
  exit 4
fi

# 2) Start Colima (adjust resources as needed)
#echo "Starting Colima (may take a few seconds)..."
#colima start --cpu 4 --memory 6 --disk 40 --vm-type=qemu --network-address

# 3) Ensure XQuartz is running and X access allowed
echo "========================================="
echo "Setting up XQuartz for GUI display..."
echo "========================================="
echo "Make sure XQuartz is installed and open. If not installed: https://www.xquartz.org/"
open -a XQuartz || true
# Wait briefly for XQuartz to initialize
sleep 3
echo "Allowing X connections from localhost (xhost +localhost)"
xhost +localhost
echo "========================================="

# 4) Get Colima VM IP for DISPLAY
COLIMA_DISPLAY="192.168.5.1:0"
echo "Using COLIMA_DISPLAY=${COLIMA_DISPLAY}"

# 5) Prepare mounts
HOST_BAG_ABS="$(cd "$(dirname "$BAGFILE")" && pwd)/$(basename "$BAGFILE")"
BAGFILE_NAME="$(basename "$BAGFILE")"
CONTAINER_BAG_PATH="/data/${BAGFILE_NAME}"
HOST_X11="/tmp/.X11-unix"
if [[ ! -d "$HOST_X11" ]]; then
  echo "Warning: $HOST_X11 not found. GUI forwarding may fail."
fi

# 6) Pull ROS image if not present and run container
ROS_IMAGE="osrf/ros:noetic-desktop-full"
echo "Pulling ROS image (if needed): ${ROS_IMAGE}"
docker pull "${ROS_IMAGE}"

CONTAINER_NAME="ros_colima_$$"

echo "========================================="
echo "Launching container '${CONTAINER_NAME}'"
echo "Bag file will be mounted at: ${CONTAINER_BAG_PATH}"
echo "DISPLAY=${COLIMA_DISPLAY}"
echo "========================================="

docker run --rm -it \
  --name "${CONTAINER_NAME}" \
  -e DISPLAY="${COLIMA_DISPLAY}" \
  -e QT_X11_NO_MITSHM=1 \
  -v "${HOST_BAG_ABS}":${CONTAINER_BAG_PATH}:ro \
  -v "${HOST_X11}":/tmp/.X11-unix:rw \
  --network bridge \
  "${ROS_IMAGE}" /bin/bash -lc "\
    echo '========================================='; \
    echo 'ROS Container Started'; \
    echo 'X display: \$DISPLAY'; \
    echo 'Bag file: ${CONTAINER_BAG_PATH}'; \
    echo '========================================='; \
    echo 'Initializing ROS environment...'; \
    source /opt/ros/noetic/setup.bash; \
    echo 'Verifying bag file exists...'; \
    if [ -f '${CONTAINER_BAG_PATH}' ]; then \
      echo '✓ Bag file found at ${CONTAINER_BAG_PATH}'; \
      echo 'Bag file size:' \$(du -h '${CONTAINER_BAG_PATH}' | cut -f1); \
      echo ''; \
      echo 'Starting roscore in background...'; \
      roscore > /tmp/roscore.log 2>&1 & \
      sleep 2; \
      echo 'Starting rosbag play...'; \
      rosbag play '${CONTAINER_BAG_PATH}' --clock & \
      ROSBAG_PID=\$!; \
      echo \"✓ rosbag play started (PID: \$ROSBAG_PID)\"; \
      echo ''; \
      echo '========================================='; \
      echo 'Bag file is now playing!'; \
      echo 'You can now run:'; \
      echo '  - rqt_bag ${CONTAINER_BAG_PATH} &'; \
      echo '  - rviz &'; \
      echo '  - rostopic list'; \
      echo '  - rostopic echo /topic_name'; \
      echo '========================================='; \
    else \
      echo '✗ ERROR: Bag file not found at ${CONTAINER_BAG_PATH}'; \
      echo 'Please check the mount path.'; \
    fi; \
    bash"

# Note: when the container exits, Colima remains running.
