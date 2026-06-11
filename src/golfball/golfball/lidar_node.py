import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from custom_interfaces.msg import LidarRanges

class LidarNode(Node):
    def __init__(self):
        super().__init__('lidar_node')
        self.scan_sub = self.create_subscription(
            LaserScan, 
            '/scan', 
            self.scan_callback, 
            10
        )

        self.lidar_pub = self.create_publisher(
            LidarRanges, 
            "/lidar_ranges",
            10
        )

        self.obstacle_threshold = 2.0  # meters

    def scan_callback(self, msg):
        # The SICK TIM 5xx outputs ~811 points for a 270 degree sweep.
        # Index 0 is far right-rear (-135 deg). Index 405 is straight ahead (0 deg).
        
        sectors = {
            "Right_Rear": (0, 134),      # -135° to -90°
            "Right": (135, 269),         # -90° to -45°
            "Front_Right": (270, 404),   # -45° to 0°
            "Front_Left": (405, 539),    # 0° to +45°
            "Left": (540, 674),          # +45° to +90°
            "Left_Rear": (675, 810)      # +90° to +135°
        }

        min_distances = {key: float('inf') for key in sectors.keys()}

        for sector, (start_idx, end_idx) in sectors.items():
            # Check bounds just in case the sensor drops a packet
            if start_idx < len(msg.ranges) and end_idx < len(msg.ranges):
                sector_ranges = msg.ranges[start_idx:end_idx + 1]
                
                # Filter out 'inf' and 'nan' values which the sensor returns for "out of range"
                # Filter out inf, NaN, and any distance under 5cm (0.05m) which includes the 0.0 error code
                valid_ranges = [r for r in sector_ranges if r != float('inf') and r == r and r > 0.05]
                
                # self.get_logger().info(f'{sector} ranges: {valid_ranges}')

                if valid_ranges:
                    min_distances[sector] = min(valid_ranges)
                    
        # self.get_logger().info(f'Min distances: {min_distances}')

        detections = {sector: min_distance < self.obstacle_threshold for sector, min_distance in min_distances.items()}

        priority = 0
        if detections["Front_Left"] and detections["Front_Right"]:
            priority = 3
            dir = 'right'
            arbitrary_direction = 'left' if min_distances["Front_Left"] > min_distances["Front_Right"] else 'right'
            action = "Selected Turn Arbitrary Direction "+arbitrary_direction        
        elif detections["Front_Left"] and not detections["Front_Right"]:
            priority = 3
            dir = 'right'
            action = "Turn Right to avoid obstacle on the front-left"
        elif detections["Front_Right"] and not detections["Front_Left"]:
            priority = 3
            dir = 'left'
            action = "Turn Left to avoid obstacle on the front-right"
        # Priority 2
        elif detections["Left"]:
            priority = 2
            dir = 'slight-right'
            action = "Go Forwards turning slightly right to avoid obstacle on the left"
        elif detections["Right"]:
            priority = 2
            dir = 'slight-left'
            action = "Go Forwards turning slightly left to avoid obstacle on the right"
        # Priority 3
        elif detections["Right_Rear"]:
            priority = 1
            dir = 'forward'
            action = "Go Forwards, BUT DONT reverse Right"
        elif detections["Left_Rear"]:   
            priority = 1
            dir = 'forward'
            action = "Go Forwards, BUT DONT reverse left"
        else:
            dir = 'forward'
            action = "Go Forwards"

        # Log the suggested action
        self.get_logger().info(f'Suggested action: {action}')

        lidar = LidarRanges()
        lidar.action = dir
        lidar.priority = priority

        self.lidar_pub.publish(
            lidar
        )


def main(args=None):
    rclpy.init(args=args)
    node = LidarNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()