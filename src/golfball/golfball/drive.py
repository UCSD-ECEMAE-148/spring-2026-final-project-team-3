import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class Driver(Node):
    def __init__(self):
        super().__init__('driver')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Twist()
        msg.linear.x = 0.5
        msg.angular.z = 1.0
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    driver = Driver()
    try:
        rclpy.spin(driver)
    except KeyboardInterrupt:
        driver.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
