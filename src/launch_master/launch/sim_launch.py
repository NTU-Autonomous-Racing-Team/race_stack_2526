#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    pp_pkg = get_package_share_directory('pure_pursuit')

    config_arg = DeclareLaunchArgument(
        name='config',
        default_value=os.path.join(pp_pkg, 'config', 'controller_manager.yaml'),
        description='Full path to the controller_manager YAML config file'
    )

    return LaunchDescription([
        config_arg,
        Node(
            package='state_machine',
            executable='state_machine',
            name='state_machine',
            output='screen',
        ),
        Node(
            package='pure_pursuit',
            executable='controller_manager_node',
            name='controller_manager_node',
            parameters=[
                LaunchConfiguration('config')
            ],
            output='screen',
        ),
        Node(
            package='perception',
            executable='detect',
            name='detect',
            parameters=[{'use_sim_time': True}],
            output='screen',
        ),

    ])