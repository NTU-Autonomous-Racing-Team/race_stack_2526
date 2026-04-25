#!/bin/bash
set -e

colcon build
# Setup Development Environment
source /opt/ros/humble/setup.bash
source /home/autodrive_devkit/install/setup.bash

# AutoDRIVE Devkit Workspace
cd /home/autodrive_devkit

# Launch AutoDRIVE Devkit with GUI
ros2 launch autodrive_roboracer bringup_graphics.launch.py

# Launch AutoDRIVE Devkit Headless
# ros2 launch autodrive_roboracer bringup_headless.launch.py

# echo "Ensure the F1Tenth simulator is already running."
# echo "Launching evaluation..."
# ros2 launch state_machine state_machine_launch.py
