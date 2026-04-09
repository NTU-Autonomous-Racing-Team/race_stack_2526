#!/usr/bin/env bash

# Build from workspace root
# cd /ros2_ws
colcon build

# Source workspace
source /opt/ros/foxy/setup.bash
source ./install/local_setup.bash

echo "Ensure the F1Tenth simulator is already running."
echo "Launching evaluation..."
ros2 launch state_machine state_machine_launch.py
