import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import tf_transformations
from custom_interfaces.msg import OdomDrive  # Assuming you have a custom message for odometry info
import math

class OdomSubscriber(Node):
    """ROS2 node that subscribes to odometry data and tracks distance traveled."""
    
    def __init__(self):
        super().__init__('odom_subscriber')
        
        # Subscribe to the odometry topic published by the VESC or other source
        self.odom_sub = self.create_subscription(
            Odometry,
            '/odom',
            self.odom_callback,
            10
        )

        self.publisher = self.create_publisher(OdomDrive, '/odom_drive', 10)
        
        # State for tracking distance
        self.distance_from_origin = 0.0
        self.outer_boundary = 3.0 # meters

        self.pose = {
            'x': 0.0,
            'y': 0.0,
            'theta': 0.0
        }
    
    def odom_callback(self, msg):
        # Extract current position from odometry message
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        self.distance_from_origin = (x**2 + y**2) ** 0.5
        
        # Extract orientation (yaw) from quaternion
        self.pose['theta'] = tf_transformations.euler_from_quaternion([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])[2]  # Yaw angle in radians
        
        # Log the current distance from origin
        self.get_logger().info(f'Distance from origin: {self.distance_from_origin:.2f} meters')

        odom_msg = OdomDrive()
        if self.distance_from_origin > self.outer_boundary:
            desired_yaw = math.atan2(-self.pose['y'], -self.pose['x'])  # Point back towards origin
            yaw_error = desired_yaw - self.pose['theta']
            
            if abs(yaw_error) > 0.1:  # If the yaw error is significant, turn towards the origin
                if yaw_error > 0:
                    odom_msg.action = "right"
                else:
                    odom_msg.action = "left"
            else:
                odom_msg.action = "forward"

            odom_msg.priority = 2

        else:
            odom_msg.action = "none"
            odom_msg.priority = 0

        self.publisher.publish(odom_msg)


def main(args=None):
    rclpy.init(args=args)
    odom_subscriber = OdomSubscriber()
    rclpy.spin(odom_subscriber)
    odom_subscriber.destroy_node()
    rclpy.shutdown()

