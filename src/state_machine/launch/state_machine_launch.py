#!/usr/bin/env python3
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pp_share = get_package_share_directory('pure_pursuit')
    pp_raceline = os.path.join(pp_share, 'racelines', 'korea_mintime_sparse.csv')

    config_arg = DeclareLaunchArgument(
        name='config',
        default_value=os.path.join(pp_share, 'config', 'controller_manager.yaml'),
        description='Full path to the controller_manager YAML config file',
    )
    waypoints_arg = DeclareLaunchArgument(
        name='waypoints',
        default_value=pp_raceline,
        description='Global raceline CSV used by detector/planner/controller',
    )

    return LaunchDescription([
        config_arg,
        waypoints_arg,
        Node(
            package='state_machine',
            executable='obstacle_detector',
            name='obstacle_detector',
            parameters=[{
                'waypoints_path': LaunchConfiguration('waypoints'),
                'use_track_filter': True,
                'track_dist_thresh': 0.35,
                'detect_dist': 4.5,
                'min_cluster_points': 1,
                'max_cluster_points': 60,
                'size_min_x': 0.02,
                'size_min_y': 0.02,
                'size_max_x': 0.60,
                'size_max_y': 0.60,
                'min_persist_frames': 1,
                'hold_time': 0.75,
            }],
            output='screen',
        ),
        Node(
            package='local_planner',
            executable='spliner',
            name='spliner',
            parameters=[{
                'waypoints_path': LaunchConfiguration('waypoints'),
                'static_trigger_distance': 3.20,
                'opponent_trigger_distance': 0.0,
                'planner_lookahead_horizon': 12.0,
                'min_side_clearance': 1.00,
                'lane_half_width': 1.20,
                'boundary_margin': 0.38,
                'apex_lateral_margin': 0.45,
                'obstacle_half_width': 0.18,
                'overtake_lateral_buffer': 0.22,
                'static_line_d_threshold': 0.30,
                'static_obs_alpha': 0.20,
                'pre_apex_points': [3.0, 4.2, 5.4],
                'post_apex_points': [5.0, 6.5, 8.0],
                'path_hold_sec': 1.60,
                'side_lock_release_sec': 1.60,
                'lateral_smoothing_window': 9,
                'prefer_right_overtake': True,
            }],
            output='screen',
        ),
        Node(
            package='state_machine',
            executable='state_machine',
            name='state_machine',
            parameters=[{
                'static_trigger_distance': 3.20,
                'opponent_trigger_distance': 0.0,
                'hard_safety_distance': 0.10,
                'front_window': 24,
                'clear_distance': 1.50,
                'hazard_confirm_cycles': 2,
                'overtake_timeout_sec': 8.0,
                'overtake_min_commit_sec': 1.50,
                'planner_hold_sec': 3.00,
                'required_clear_cycles': 10,
                'overtake_lost_cycles_to_fallback': 12,
                'emergency_persist_cycles': 3,
                'ignore_opponent': True,
            }],
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
    ])
