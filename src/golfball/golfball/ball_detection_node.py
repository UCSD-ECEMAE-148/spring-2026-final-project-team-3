#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import depthai as dai
import numpy as np
from custom_interfaces.msg import DetectedBall, DetectedBalls
from sensor_msgs.msg import CompressedImage
import threading

# ====== CONFIGURATION ======
BLOB_PATH = "/home/projects/ros2_ws/src/golfball/detection_model/best.blob"
CONFIDENCE_THRESHOLD = 0.3
ANNOTATE_STREAM = True  # True = Draw bounding boxes on the web stream, False = Pure raw video
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
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
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
        self.image_pub = self.create_publisher(CompressedImage, 'camera/image_raw/compressed', 10)
        
        # DepthAI Hardware Setup
        self.device = dai.Device(create_pipeline())
        self.q_video = self.device.getOutputQueue("video", 1, False)
        self.q_nn = self.device.getOutputQueue("nn", 1, False)
        
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
        import threading
        while self.running and rclpy.ok():
            in_video = self.q_video.get()
            in_nn = self.q_nn.get()

            img = in_video.getCvFrame()
            H, W = img.shape[:2]

            dets = []
            if in_nn is not None:
                dets = self.parse_detections(in_nn, W, H)

            # 1. Build and Publish Coordinates Message
            balls_msg = DetectedBalls()
            balls_msg.header.stamp = self.get_clock().now().to_msg()
            balls_msg.header.frame_id = 'camera_optical_frame'
            
            for i, (x1, y1, x2, y2, conf) in enumerate(dets):
                b = DetectedBall()
                b.id = i
                b.x1, b.y1, b.x2, b.y2, b.confidence = float(x1), float(y1), float(x2), float(y2), float(conf)
                balls_msg.balls.append(b)
                
                if ANNOTATE_STREAM:
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, f"Raw {conf*100:.1f}%", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            self.ball_pub.publish(balls_msg)

            # 2. Compress and Publish Image over ROS2 topic
            _, encoded_img = cv2.imencode('.jpg', img)
            img_msg = CompressedImage()
            img_msg.header = balls_msg.header
            img_msg.format = "jpeg"
            img_msg.data = encoded_img.tobytes()
            self.image_pub.publish(img_msg)

    def destroy_node(self):
        self.running = False
        self.thread.join()
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