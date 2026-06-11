import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from custom_interfaces.msg import DetectedBalls, ControllerInput, LidarRanges


class SteeringPIDController:
    """PID controller for steering to center the ball on screen."""
    
    def __init__(self, kp=0.5, kd=0.1, max_steering=1.0):
        """
        Args:
            kp: Proportional gain
            kd: Derivative gain
            max_steering: Maximum steering angle (radians)
        """
        self.kp = kp
        self.kd = kd
        self.max_steering = max_steering
        
        self.previous_error = 0.0
        self.dt = 0.05  # 10 Hz timer
    
    def update(self, error):
        """
        Calculate steering command from error.
        
        Args:
            error: Lateral error (radians)
        
        Returns:
            Steering command (radians), clamped to [-max_steering, max_steering]
        """
        # Proportional term
        p_term = self.kp * error
        
        # Derivative term
        error_derivative = (error - self.previous_error) / self.dt
        d_term = self.kd * error_derivative
        
        # PID output
        steering_raw = p_term + d_term
        
        # Clamp to valid steering range
        steering = max(-self.max_steering, min(self.max_steering, steering_raw))
        
        # Update state
        self.previous_error = error
        
        return steering


class Driver(Node):
    """ROS2 node that drives the robot toward a detected ball using steering PID."""
    
    # OAK-D Lite camera parameters
    IMAGE_WIDTH = 640
    IMAGE_CENTER_X = IMAGE_WIDTH / 2.0  # 320
    FOV_HORIZONTAL_DEG = 127.0  # Degrees
    FOV_HORIZONTAL_RAD = FOV_HORIZONTAL_DEG * math.pi / 180.0
    PIXELS_PER_RADIAN = (IMAGE_CENTER_X) / (FOV_HORIZONTAL_RAD / 2.0)
    
    # Control parameters
    FORWARD_SPEED = 0.35  # m/s
    MIN_CONFIDENCE = 0.5  # Ignore detections below this

    MANUAL_DRIVE_MODE = True

    TEMP_LIDAR_MODE = True
    
    def __init__(self):
        super().__init__('driver')
        
        # Publisher for steering commands
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Subscriber for ball detection
        self.ball_subscription = self.create_subscription(
            DetectedBalls,
            '/detected_balls',
            self.ball_callback,
            10
        )

        self.controller_subscription = self.create_subscription(
            ControllerInput,
            '/controller_input',
            self.manual_driving,
            10
        )

        self.lidar_subscription = self.create_subscription(
            LidarRanges,
            '/lidar_ranges',
            self.lidar_callback,
            10
        )
        
        # PID controller for steering
        self.pid_controller = SteeringPIDController(kp=0.6, kd=0.05)
        
        # State tracking
        self.last_ball_position = None  # (x, y) tuple for ball center
        self.last_ball_time = None
        self.frames_without_ball = 0
        
        # Timer for control loop
        timer_period = 0.1  # 10 Hz
        self.timer = self.create_timer(timer_period, self.control_callback)
        
        self.get_logger().info(
            f"Driver initialized. FOV: {self.FOV_HORIZONTAL_DEG}° horizontal, "
            f"Image center: ({self.IMAGE_CENTER_X}, 320)"
        )

        self.lidar_vel = {
            'linear': 0.0, 
            'angular': 0.0
        }

        self.MANUAL_SPEED = {
            'X': 0.0,
            'Y': 0.0
        }

    def manual_driving(self, msg):
        """
        Callback for manual controller input.
        
        Args:
            msg: ControllerInput message containing button and stick states
        """
        if msg.button_y:
            self.MANUAL_DRIVE_MODE = not self.MANUAL_DRIVE_MODE
            self.get_logger().info(f"Drive mode toggled: {not self.MANUAL_DRIVE_MODE}")
        
    
    def ball_callback(self, msg):
        """
        Callback for ball detection messages.
        
        Args:
            msg: DetectedBalls message containing detected balls
        """
        if self.TEMP_LIDAR_MODE:
            return
        
        if not msg.balls:
            # No balls detected
            self.frames_without_ball += 1
            return
        
        # Select primary ball (highest confidence)
        primary_ball = max(
            msg.balls,
            key=lambda ball: ball.confidence
        )
        
        # Skip if confidence too low
        if primary_ball.confidence < self.MIN_CONFIDENCE:
            self.frames_without_ball += 1
            return
        
        # Calculate ball center
        ball_center_x = (primary_ball.x1 + primary_ball.x2) / 2.0
        ball_center_y = (primary_ball.y1 + primary_ball.y2) / 2.0
        
        self.last_ball_position = (ball_center_x, ball_center_y)
        self.frames_without_ball = 0
        
        self.get_logger().debug(
            f"Ball detected at pixel ({ball_center_x:.1f}, {ball_center_y:.1f}), "
            f"confidence: {primary_ball.confidence:.2f}"
        )

    def lidar_callback(self, msg):
        """
        Callback for LidarRanges messages.
        
        Args:
            msg: LidarRanges message containing obstacle information
        """
        # for (testing) just ignore the priorities
        self.get_logger().info(f"Lidar message received with priority: {msg.priority}")
        action = msg.action
        if action == "forward":
            self.lidar_vel['linear'] = self.FORWARD_SPEED
            self.lidar_vel['angular'] = 0.0
        elif action == "slight-left":
            self.lidar_vel['linear'] = self.FORWARD_SPEED
            self.lidar_vel['angular'] = 0.3
        elif action == "slight-right":
            self.lidar_vel['linear'] = self.FORWARD_SPEED
            self.lidar_vel['angular'] = -0.3
        elif action == "left":
            self.lidar_vel['linear'] = 0.0
            self.lidar_vel['angular'] = 0.5
        elif action == "right":
            self.lidar_vel['linear'] = 0.0
            self.lidar_vel['angular'] = -0.5

    
    def pixel_error_to_angle_error(self, pixel_error_x):
        """
        Convert pixel error to angle error in radians.
        
        Args:
            pixel_error_x: Horizontal pixel offset from image center
        
        Returns:
            Angle error in radians
        """
        angle_error = pixel_error_x / self.PIXELS_PER_RADIAN
        return angle_error
    
    def control_callback(self):
        """
        Main control loop. Called at 10 Hz.
        Calculates steering command based on ball position.
        """

        msg = Twist()

        if not self.TEMP_LIDAR_MODE:
            msg.linear.x = self.FORWARD_SPEED
            
            if self.last_ball_position is not None and self.frames_without_ball < 3:
                # Ball detected recently - use steering PID
                ball_x, ball_y = self.last_ball_position
                
                # Calculate lateral error (pixel offset from image center)
                pixel_error_x = ball_x - self.IMAGE_CENTER_X
                
                # Convert to angle error
                angle_error = self.pixel_error_to_angle_error(pixel_error_x)
                
                # Calculate steering command using PID
                steering_command = self.pid_controller.update(angle_error)
                
                msg.angular.z = steering_command
                
                self.get_logger().debug(
                    f"Ball at pixel x={ball_x:.1f}, error={pixel_error_x:.1f}px "
                    f"({angle_error:.4f}rad), steering={steering_command:.4f}rad"
                )
            else:
                # No ball detected - hold steady
                msg.angular.z = 0.0
                if self.frames_without_ball == 0:
                    self.get_logger().info("No ball detected, steering neutral")
        
        else:
            # msg.linear.x = self.lidar_vel['linear']
            msg.linear.x = self.FORWARD_SPEED
            msg.angular.z = -self.lidar_vel['angular']

            self.get_logger().info(f'Linear speed: {self.lidar_vel["linear"]:.2f} m/s, Angular speed: {self.lidar_vel["angular"]:.2f} rad/s')

        if self.MANUAL_DRIVE_MODE:
            msg.linear.x = self.MANUAL_SPEED['X'] / 255.0
            msg.angular.z = self.MANUAL_SPEED['Y'] / -255.0

            self.get_logger().info(f"Manual drive mode: linear={msg.linear.x:.2f} m/s, angular={msg.angular.z:.2f} rad/s")

        self.publisher_.publish(msg)
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        self.get_logger().info("Driver node shutting down")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    driver = Driver()
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        driver.get_logger().info("Interrupted by user")
    finally:
        driver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()