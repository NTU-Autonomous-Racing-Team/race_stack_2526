#!/usr/bin/env bash
set -e

# Source ROS first so Python can resolve ament modules during colcon build.
if [ -f /opt/ros/foxy/setup.bash ]; then
	source /opt/ros/foxy/setup.bash
elif [ -f /opt/ros/humble/setup.bash ]; then
	source /opt/ros/humble/setup.bash
else
	echo "No ROS setup found under /opt/ros" >&2
	exit 1
fi

# Build from workspace root
# cd /ros2_ws
colcon build

# Source workspace
source ./install/local_setup.bash

echo "Ensure the F1Tenth simulator is already running."
echo "Launching evaluation..."
ros2 launch state_machine state_machine.py
