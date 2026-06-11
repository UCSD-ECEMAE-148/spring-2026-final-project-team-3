import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32, Bool
from evdev import InputDevice, ecodes, categorize
from custom_interfaces.msg import ControllerInput

class ControllerNode(Node):
    def __init__(self):
        super().__init__('controller_node')
        self.gamepad = InputDevice('/dev/input/event1') # Might have to change this depending on what event the controller maps to
        # python3 -m evdev.evtest
        
        self.state = {
            'Y': False,  # Button Y
            'LX': 0.0,   # Left stick X-axis
            'LY': 0.0,   # Left stick Y-axis
        }

        # Publishers
        self.controller_pub = self.create_publisher(ControllerInput, '/controller_input', 10)
        
        # Timer to read events (prevents blocking the ROS loop)
        self.create_timer(0.01, self.read_controller)
        
    def read_controller(self):
        # Read all events currently available in the device buffer
        try:
            # .read() returns an iterator of events
            for event in self.gamepad.read():
                if event.type == ecodes.EV_KEY and event.code == 308:
                    self.state['Y'] = event.value == 1
                
                elif event.type == ecodes.EV_ABS and event.code == 1:
                    self.state['LY'] = (event.value - 128) / 128.0  # Normalize to [-1, 1]

                elif event.type == ecodes.EV_ABS and event.code == 0:
                    self.state['LX'] = (event.value - 128) / 128.0  # Normalize to [-1, 1]
                
            # Publish the current state
            msg = ControllerInput()
            msg.button_y = self.state['Y']
            msg.left_stick_x = self.state['LX']
            msg.left_stick_y = self.state['LY']
            self.controller_pub.publish(msg)

        except BlockingIOError:
            # This is expected when no events are in the buffer
            pass

def main(args=None):
    rclpy.init(args=args)
    node = ControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()