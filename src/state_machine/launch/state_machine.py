#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception',
            executable='detect',
            name='detect',
            output='screen',
        ),
        Node(
            package='state_machine',
            executable='state_machine',
            name='state_machine',
            output='screen',
        ),
        Node(
            package='gap_finder',
            executable='gap_finder_node',
            name='gap_finder_node',
            output='screen',
        ),
        Node(
            package='pure_pursuit',
            executable='controller_manager_node',
            name='controller_manager_node',
            output='screen',
        ),
    ])
