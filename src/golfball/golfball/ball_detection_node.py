#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
import cv2
import depthai as dai
import numpy as np
from custom_interfaces.msg import DetectedBall, DetectedBalls
from std_msgs.msg import Header
import os

# Config constants (from detection2.py)
BLOB_PATH = "/home/projects/ros2_ws/src/golfball/detection_model/best.blob"
CONFIDENCE_THRESHOLD = 0.3
ALPHA = 1.0  # Lower = smoother, Higher = snappier
MAX_BALLS = 3
DIST_THRESH = 80 # Max pixel distance to match a ball to its previous position

def create_pipeline():
    """Create OAK pipeline (from detection2.py)."""
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    # manip = pipeline.create(dai.node.ImageManip)
    # xout_rgb = pipeline.create(dai.node.XLinkOut)
    # xout_nn = pipeline.create(dai.node.XLinkOut)
    # xout_rgb.setStreamName("rgb")
    # xout_nn.setStreamName("nn")
    
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setPreviewKeepAspectRatio(True)
    
    manip.setMaxOutputFrameSize(640 * 640 * 3)
    # manip.initialConfig.setCropRect(0.0, 0.25, 1.0, 1.0)
    # manip.initialConfig.setResize(640, 640)
    # manip.initialConfig.setCropRect(0.0, 0.25, 1.0, 1.0) // this crops the top 25% of the frame, which may help if the camera is angled downwards and we want to ignore the ceiling
    manip.setResize(640, 640)  # top-level call instead
    
    detection_nn.setBlobPath(BLOB_PATH)
    detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detection_nn.setNumClasses(1)
    detection_nn.setCoordinateSize(4)
    detection_nn.setIouThreshold(0.5)
    
    cam_rgb.preview.link(manip.inputImage)
    manip.out.link(detection_nn.input)
    manip.out.link(xout_rgb.input)
    detection_nn.out.link(xout_nn.input)
    
    return pipeline

def update_tracked_balls(raw_detections, tracked_balls, next_ball_id, width, height):
    """Update tracked balls with new detections (from detection2.py)."""
    current_frame_matches = {}
    
    for detection in raw_detections:
        rx1 = int(detection.xmin * width)
        ry1 = int(detection.ymin * height)
        rx2 = int(detection.xmax * width)
        ry2 = int(detection.ymax * height)
        conf = detection.confidence
        rcx, rcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2
        
        matched_id = None
        best_dist = DIST_THRESH
        
        for b_id, data in tracked_balls.items():
            prev_cx = (data[0] + data[2]) // 2
            prev_cy = (data[1] + data[3]) // 2
            distance = np.hypot(rcx - prev_cx, rcy - prev_cy)
            if distance < best_dist:
                best_dist = distance
                matched_id = b_id
        
        if matched_id is None:
            matched_id = next_ball_id
            next_ball_id += 1
            current_frame_matches[matched_id] = [rx1, ry1, rx2, ry2, 0, conf]
        else:
            prev = tracked_balls[matched_id]
            sm_x1 = int(ALPHA * rx1 + (1 - ALPHA) * prev[0])
            sm_y1 = int(ALPHA * ry1 + (1 - ALPHA) * prev[1])
            sm_x2 = int(ALPHA * rx2 + (1 - ALPHA) * prev[2])
            sm_y2 = int(ALPHA * ry2 + (1 - ALPHA) * prev[3])
            sm_conf = ALPHA * conf + (1 - ALPHA) * prev[5]
            current_frame_matches[matched_id] = [sm_x1, sm_y1, sm_x2, sm_y2, 0, sm_conf]
    
    for b_id in list(tracked_balls.keys()):
        if b_id not in current_frame_matches:
            tracked_balls[b_id][4] += 1
            if tracked_balls[b_id][4] > 5:
                del tracked_balls[b_id]
        else:
            tracked_balls[b_id] = current_frame_matches[b_id]
    
    for b_id, data in current_frame_matches.items():
        tracked_balls[b_id] = data
    
    return tracked_balls, next_ball_id

def display_detections(frame, tracked_balls):
    """Draw bounding boxes (from detection2.py)."""
    for b_id, data in tracked_balls.items():
        x1, y1, x2, y2, _, conf = data
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Ball {b_id}: {conf * 100:.1f}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

class BallDetectionNode(Node):
    def __init__(self):
        super().__init__('ball_detection_node')
        
        # Declare parameters
        self.declare_parameter('display_mode', False)
        self.declare_parameter('fps', 30)

        self.declare_parameter('save_frames', True)  # Optional: Save frames for debugging
        self.declare_parameter('save_interval', 30)  # save every 30 frames = ~1/sec at 30fps
        self.declare_parameter('save_path', './saved_frames')
        
        self.display_mode = self.get_parameter('display_mode').value
        self.fps = self.get_parameter('fps').value

        self.save_frames = self.get_parameter('save_frames').value
        self.save_interval = self.get_parameter('save_interval').value
        self.save_path = self.get_parameter('save_path').value
        self.frame_count = 0
        
        self.get_logger().info(f"Ball Detection Node starting (display: {self.display_mode}, fps: {self.fps})")
        
        # Initialize OAK device
        self.pipeline = create_pipeline()
        self.device = dai.Device(self.pipeline)
        
        # Setup queues
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_nn = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)
        
        # Publisher
        self.publisher = self.create_publisher(DetectedBalls, 'detected_balls', 10)
        
        # Tracked balls
        self.tracked_balls = {}
        self.next_ball_id = 0
        
        # Timer (30 Hz by default)
        self.create_timer(1.0 / self.fps, self.timer_callback)

        if self.save_frames:
            os.makedirs(self.save_path, exist_ok=True)
        
        self.get_logger().info('Ball Detection Node initialized')
    
    def timer_callback(self):
        """Poll OAK and publish detections."""
        try:
            in_rgb = self.q_rgb.tryGet()
            in_nn = self.q_nn.tryGet()
            
            if in_rgb is None:
                return
            
            frame = in_rgb.getCvFrame()
            height, width = frame.shape[:2]
            
            if in_nn is not None:
                raw_detections = sorted(in_nn.detections, key=lambda x: x.confidence, reverse=True)[:MAX_BALLS]
                self.tracked_balls, self.next_ball_id = update_tracked_balls(
                    raw_detections, self.tracked_balls, self.next_ball_id, width, height
                )
            
            # Publish message
            msg = DetectedBalls()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_optical_frame'
            
            for b_id, data in self.tracked_balls.items():
                ball = DetectedBall()
                ball.id = b_id
                ball.x1 = float(data[0])
                ball.y1 = float(data[1])
                ball.x2 = float(data[2])
                ball.y2 = float(data[3])
                ball.confidence = float(data[5])
                msg.balls.append(ball)
            
            self.publisher.publish(msg)
            
            # Optional display
            if self.display_mode:
                display_detections(frame, self.tracked_balls)
                cv2.imshow("OAK Detection", frame)
                cv2.waitKey(1)
            
            # Optional: Save frames for debugging
            if self.save_frames:
                self.frame_count += 1
                if self.frame_count % self.save_interval == 0:
                    display_detections(frame, self.tracked_balls)
                    filename = os.path.join(self.save_path, f"frame_{self.frame_count:06d}.jpg")
                    cv2.imwrite(filename, frame)
                    self.get_logger().info(f"Saved {filename}")

            # DEBUG: Delete this later
            if in_nn is not None:
                for d in in_nn.detections:
                    print(f"RAW: xmin={d.xmin:.3f} ymin={d.ymin:.3f} xmax={d.xmax:.3f} ymax={d.ymax:.3f}")
        
        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")
    
    def destroy_node(self):
        """Cleanup on shutdown."""
        self.device.close()
        if self.display_mode:
            cv2.destroyAllWindows()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = BallDetectionNode()
    
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    
    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()