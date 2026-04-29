import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # Get the slam_toolbox package share directory
    pkgdir = get_package_share_directory('autodrive_roboracer')
    
    # Declare launch arguments
    slam_params_file = DeclareLaunchArgument(
        'slam_params_file',
        default_value=os.path.join(pkgdir, 'config', 'mapper_params_online_async.yaml'),
        description='Full path to the SLAM params file to use'
    )
    
    use_sim_time = DeclareLaunchArgument(
        'use_sim_time',
        default_value='False',
        description='Use /clock if true'
    )

    # SLAM Toolbox node
    slam_toolbox_node = Node(
        package='slam_toolbox',
        executable='async_slam_toolbox_node',
        name='slam_toolbox',
        output='screen',
        parameters=[
            LaunchConfiguration('slam_params_file'),
            {'use_sim_time': LaunchConfiguration('use_sim_time')}
        ],
    )

    return LaunchDescription([
        slam_params_file,
        use_sim_time,
        slam_toolbox_node,
    ])