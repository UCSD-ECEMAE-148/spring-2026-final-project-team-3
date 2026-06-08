#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import threading
import time
from flask import Flask, Response

app = Flask(__name__)
latest_web_jpeg = b''
frame_lock = threading.Lock()

class Stream(Node):
    def __init__(self):
        super().__init__('stream')
        
        # Subscribe to the compressed image topic emitted from the vision node
        self.sub = self.create_subscription(
            CompressedImage,
            'camera/image_raw/compressed',
            self.image_callback,
            10
        )
        self.get_logger().info('Stream Node initialized. Listening on port 5000...')

    def image_callback(self, msg):
        global latest_web_jpeg
        with frame_lock:
            latest_web_jpeg = bytes(msg.data)


# ===== FLASK WEB SERVER ROUTINES =====

def generate_web_stream():
    global latest_web_jpeg
    while True:
        with frame_lock:
            if not latest_web_jpeg:
                time.sleep(0.01)
                continue
            frame_data = latest_web_jpeg
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n')
        time.sleep(0.03)  # Limits streaming frame output rate slightly to save host bandwidth


@app.route('/')
def index():
    return "<h1>Golf ball detection stream</h1><img src='/video_feed' width='640' height='640'>"


@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    # Runs the server locally accessible at http://localhost:5000
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


def main(args=None):
    rclpy.init(args=args)
    
    # Start Flask Web engine completely decoupled in its own background worker thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    node = Stream()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()