from launch import LaunchDescription
from launch_ros.actions import Node
import os
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    actuator_pkg = 'ucsd_robocar_actuator2_pkg'
    vesc_launch = os.path.join(
        get_package_share_directory(actuator_pkg),
        'launch',
        'vesc_twist.launch.py'
    )

    # Optional: your local publisher node (replace package/executable as needed)
    drive_publisher = Node(
        package='golfball',
        executable='drive_publisher', 
        name='drive_publisher',
        output='screen'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(vesc_launch),
            # Optional remap / override launch args:
            # launch_arguments={'topic_name': '/cmd_vel'}.items()
        ),
        drive_publisher,  # remove or comment this line if you don't have a local publisher yet
    ])
