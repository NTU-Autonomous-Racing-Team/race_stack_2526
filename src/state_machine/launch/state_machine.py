#!/usr/bin/env python3

from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='state_machine',
            executable='state_machine',
            name='state_machine',
            output='screen',
        )
    ])
