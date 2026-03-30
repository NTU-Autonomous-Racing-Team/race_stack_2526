#!/usr/bin/env bash

# Build from workspace root
# cd /ros2_ws
colcon build

# Source workspace
source /opt/ros/foxy/setup.bash
source install/local_setup.bash

# Ensure the F1Tenth simulator is already running.
echo "Ensure the F1Tenth simulator is already running."
echo "Launching evaluation..."
ros2 launch f1tenth_gym_ros gym_bridge_launch.py
