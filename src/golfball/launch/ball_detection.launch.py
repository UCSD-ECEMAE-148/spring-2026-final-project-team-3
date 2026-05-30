from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration

def generate_launch_description():
    display_mode_arg = DeclareLaunchArgument(
        'display_mode',
        default_value='false',
        description='Enable display window for debugging'
    )
    
    fps_arg = DeclareLaunchArgument(
        'fps',
        default_value='30',
        description='Camera FPS'
    )

    ball_detection = Node(
        package='golfball',
        executable='ball_detection_node',
        name='ball_detection_node',
        parameters=[
            {'display_mode': LaunchConfiguration('display_mode')},
            {'fps': LaunchConfiguration('fps')}
        ],
        output='screen'
    )

    return LaunchDescription([
        display_mode_arg,
        fps_arg,
        ball_detection
    ])