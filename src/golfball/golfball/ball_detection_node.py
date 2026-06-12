#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import depthai as dai
import numpy as np
from custom_interfaces.msg import DetectedBall, DetectedBalls
from sensor_msgs.msg import CompressedImage
import threading
import time  # Imported for the retry delay

# ====== CONFIGURATION ======
BLOB_PATH = "/home/projects/ros2_ws/src/golfball/detection_model/best.blob"
CONFIDENCE_THRESHOLD = 0.3
ANNOTATE_STREAM = True  # True = Draw bounding boxes on the web stream, False = Pure raw video
STREAM = False
# ===========================

def create_pipeline():
    pipeline = dai.Pipeline()
    
    cam = pipeline.create(dai.node.ColorCamera)
    nn = pipeline.create(dai.node.NeuralNetwork)
    
    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    
    xout_video.setStreamName("video")
    xout_nn.setStreamName("nn")

    cam.setPreviewSize(640, 640)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_12_MP)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)

    nn.setBlobPath(BLOB_PATH)

    cam.preview.link(nn.input)
    cam.preview.link(xout_video.input)
    nn.out.link(xout_nn.input)
    
    return pipeline


class BallDetectionNode(Node):
    def __init__(self):
        super().__init__('ball_detection_node')
        
        # Publishers
        self.ball_pub = self.create_publisher(DetectedBalls, 'detected_balls', 10)
        if STREAM:
            self.image_pub = self.create_publisher(CompressedImage, 'camera/image_raw/compressed', 10)
        
        # ====== DEPTHAI HARDWARE RETRY SETUP ======
        self.device = None
        max_retries = 5
        retry_delay = 2.0  # seconds
        pipeline = create_pipeline()
        
        for attempt in range(1, max_retries + 1):
            try:
                self.get_logger().info(f'Connecting to OAK-D Camera... (Attempt {attempt}/{max_retries})')
                self.device = dai.Device(pipeline)
                break
            except RuntimeError as e:
                self.get_logger().warn(f'Camera connection failed on attempt {attempt}: {e}')
                if attempt < max_retries:
                    self.get_logger().info(f'Retrying in {retry_delay} seconds...')
                    time.sleep(retry_delay)
                else:
                    self.get_logger().error('Max retries reached. Could not connect to OAK-D hardware.')
                    raise e
        
        self.q_video = self.device.getOutputQueue("video", 1, False)
        self.q_nn = self.device.getOutputQueue("nn", 1, False)
        # ==========================================
        
        # Persistent target memory tracking coordinates
        self.tracked_ball_center = None 
        
        self.running = True
        self.thread = threading.Thread(target=self.device_loop, daemon=True)
        self.thread.start()
        self.get_logger().info('Ball Detection Node initialized.')

    def parse_detections(self, nn_data, width, height):
        raw_data = nn_data.getLayerFp16('output0') if hasattr(nn_data, 'getLayerFp16') else nn_data.getFirstTensor()
        if raw_data is None: return []
        raw = np.array(raw_data).reshape(5, 8400)
        
        dets = []
        for i in range(8400):
            c = float(raw[4, i])
            if c < CONFIDENCE_THRESHOLD: continue
            cx, cy, w, h = raw[0,i]/640, raw[1,i]/640, raw[2,i]/640, raw[3,i]/640
            dets.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, c))

        if not dets: return []
        boxes = [[d[0]*width, d[1]*height, (d[2]-d[0])*width, (d[3]-d[1])*height] for d in dets]
        idx = cv2.dnn.NMSBoxes(boxes, [d[4] for d in dets], CONFIDENCE_THRESHOLD, 0.5)
        if len(idx) == 0: return []
        return [(max(0,int(dets[i][0]*width)), max(0,int(dets[i][1]*height)),
                 min(width,int(dets[i][2]*width)), min(height,int(dets[i][3]*height)),
                 dets[i][4]) for i in idx.flatten()]

    def device_loop(self):
        while self.running and rclpy.ok():
            in_video = self.q_video.get()
            in_nn = self.q_nn.get()

            img = in_video.getCvFrame()
            H, W = img.shape[:2]

            dets = []
            if in_nn is not None:
                dets = self.parse_detections(in_nn, W, H)

            balls_msg = DetectedBalls()
            balls_msg.header.stamp = self.get_clock().now().to_msg()
            balls_msg.header.frame_id = 'camera_optical_frame'
            
            target_idx = None
            
            if len(dets) > 0:
                # If we were tracking a ball previously, find the one closest to its last known center point
                if self.tracked_ball_center is not None:
                    distances = []
                    for (x1, y1, x2, y2, _) in dets:
                        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                        dist = np.hypot(cx - self.tracked_ball_center[0], cy - self.tracked_ball_center[1])
                        distances.append(dist)
                    
                    # If the closest matching ball is reasonably nearby, keep tracking it
                    best_match = np.argmin(distances)
                    if distances[best_match] < (W * 0.25):  # Gated boundary window limits jumping
                        target_idx = best_match
                
                # Default option: If no previous track, lock onto the physically closest ball 
                # (The ball with the maximum y2 coordinate sits lowest on the image frame)
                if target_idx == None:
                    target_idx = np.argmax([d[3] for d in dets])
                
                # Save the center position of our locked target frame
                tx1, ty1, tx2, ty2, tconf = dets[target_idx]
                self.tracked_ball_center = ((tx1 + tx2) / 2, (ty1 + ty2) / 2)
            else:
                # Clear memory tracking cache if no balls are visible at all
                self.tracked_ball_center = None

            # Process array for publishing
            for i, (x1, y1, x2, y2, conf) in enumerate(dets):
                b = DetectedBall()
                
                # Assign ID 0 explicitly to our prioritized locked target ball, 1+ to background distractors
                if i == target_idx:
                    b.id = 0
                    color = (0, 0, 255)  # Red box for locked target
                    label = f"LOCKED TARGET {conf*100:.1f}%"
                else:
                    b.id = i + 1
                    color = (0, 255, 0)  # Green box for passive background obstacles
                    label = f"Queued {conf*100:.1f}%"
                
                b.x1, b.y1, b.x2, b.y2, b.confidence = float(x1), float(y1), float(x2), float(y2), float(conf)
                
                # Append only the locked target, or keep background entries with distinct IDs
                # We'll publish all entries but flag the active steering target as id=0
                balls_msg.balls.append(b)
                
                if ANNOTATE_STREAM:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(img, label, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            self.ball_pub.publish(balls_msg)

            if STREAM:
                # Compress and Publish Image over ROS2 topic
                _, encoded_img = cv2.imencode('.jpg', img)
                img_msg = CompressedImage()
                img_msg.header = balls_msg.header
                img_msg.format = "jpeg"
                img_msg.data = encoded_img.tobytes()
                self.image_pub.publish(img_msg)

    def destroy_node(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join()
        if hasattr(self, 'device') and self.device is not None:
            self.device.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = BallDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()