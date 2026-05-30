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

BLOB_PATH = "/home/projects/ros2_ws/src/golfball/detection_model/best.blob"
CONFIDENCE_THRESHOLD = 0.5
MAX_BALLS = 3
DIST_THRESH = 80
SAVE_PATH = "./saved_frames"


def create_pipeline():
    pipeline = dai.Pipeline()
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    manip = pipeline.create(dai.node.ImageManip)
    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("nn")

    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)
    cam_rgb.setPreviewKeepAspectRatio(True)

    manip.setMaxOutputFrameSize(640 * 640 * 3)
    manip.initialConfig.setCropRect(0.0, 0.25, 1.0, 1.0)
    manip.setResize(640, 640)

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


def make_kalman():
    kf = cv2.KalmanFilter(6, 4)
    kf.transitionMatrix = np.array([
        [1, 0, 1, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.5
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf


class Track:
    def __init__(self, track_id, x1, y1, x2, y2, conf):
        self.id = track_id
        self.conf = conf
        self.age = 0
        self.kf = make_kalman()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = float(x2 - x1)
        h = float(y2 - y1)
        self.kf.statePost = np.array([[cx], [cy], [0], [0], [w], [h]], dtype=np.float32)

    def predict(self):
        self.kf.predict()

    def update(self, x1, y1, x2, y2, conf):
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        w = float(x2 - x1)
        h = float(y2 - y1)
        meas = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        self.kf.correct(meas)
        self.conf = conf
        self.age = 0

    def get_box(self):
        s = self.kf.statePost
        cx, cy, w, h = s[0, 0], s[1, 0], s[4, 0], s[5, 0]
        return int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)


def update_tracks(detections, tracks, next_id, width, height):
    for t in tracks.values():
        t.predict()

    det_boxes = []
    for d in detections:
        x1 = max(0, int(d.xmin * width))
        y1 = max(0, int(d.ymin * height))
        x2 = min(width, int(d.xmax * width))
        y2 = min(height, int(d.ymax * height))
        det_boxes.append((x1, y1, x2, y2, d.confidence))

    matched_track_ids = set()
    matched_det_ids = set()

    for di, (x1, y1, x2, y2, conf) in enumerate(det_boxes):
        dcx, dcy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        best_dist = DIST_THRESH
        best_tid = None
        for tid, t in tracks.items():
            if tid in matched_track_ids:
                continue
            s = t.kf.statePost
            dist = np.hypot(dcx - s[0, 0], dcy - s[1, 0])
            if dist < best_dist:
                best_dist = dist
                best_tid = tid
        if best_tid is not None:
            tracks[best_tid].update(x1, y1, x2, y2, conf)
            matched_track_ids.add(best_tid)
            matched_det_ids.add(di)

    for di, (x1, y1, x2, y2, conf) in enumerate(det_boxes):
        if di not in matched_det_ids:
            tracks[next_id] = Track(next_id, x1, y1, x2, y2, conf)
            next_id += 1

    for tid in list(tracks.keys()):
        if tid not in matched_track_ids:
            tracks[tid].age += 1
            if tracks[tid].age > 5:
                del tracks[tid]

    if len(tracks) > MAX_BALLS:
        sorted_tracks = sorted(tracks.items(), key=lambda x: x[1].conf, reverse=True)
        tracks = dict(sorted_tracks[:MAX_BALLS])

    return tracks, next_id


def draw_tracks(frame, tracks, width, height):
    for tid, t in tracks.items():
        x1, y1, x2, y2 = t.get_box()
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width, x2), min(height, y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Ball {tid}: {t.conf * 100:.1f}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


class BallDetectionNode(Node):
    def __init__(self):
        super().__init__('ball_detection_node')

        self.declare_parameter('fps', 30)
        self.declare_parameter('save_path', SAVE_PATH)

        self.fps = self.get_parameter('fps').value
        self.save_path = self.get_parameter('save_path').value
        self.frame_count = 0
        self.tracks = {}
        self.next_id = 0

        os.makedirs(self.save_path, exist_ok=True)

        self.pipeline = create_pipeline()
        self.device = dai.Device(self.pipeline)
        self.q_rgb = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        self.q_nn = self.device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        self.publisher = self.create_publisher(DetectedBalls, 'detected_balls', 10)
        self.create_timer(1.0 / self.fps, self.timer_callback)

        self.get_logger().info('Ball Detection Node initialized')

    def timer_callback(self):
        try:
            in_rgb = self.q_rgb.tryGet()
            in_nn = self.q_nn.tryGet()

            if in_rgb is None:
                return

            frame = in_rgb.getCvFrame()
            height, width = frame.shape[:2]

            if in_nn is not None:
                raw = sorted(in_nn.detections, key=lambda x: x.confidence, reverse=True)[:MAX_BALLS]
                self.tracks, self.next_id = update_tracks(raw, self.tracks, self.next_id, width, height)

            # Publish
            msg = DetectedBalls()
            msg.header = Header()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_optical_frame'
            for tid, t in self.tracks.items():
                x1, y1, x2, y2 = t.get_box()
                ball = DetectedBall()
                ball.id = tid
                ball.x1 = float(max(0, x1))
                ball.y1 = float(max(0, y1))
                ball.x2 = float(min(width, x2))
                ball.y2 = float(min(height, y2))
                ball.confidence = float(t.conf)
                msg.balls.append(ball)
            self.publisher.publish(msg)

            # Save frame every 30
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                draw_tracks(frame, self.tracks, width, height)
                filename = os.path.join(self.save_path, f"frame_{self.frame_count:06d}.jpg")
                cv2.imwrite(filename, frame)
                self.get_logger().info(f"Saved {filename}")

        except Exception as e:
            self.get_logger().error(f"Error: {str(e)}")

    def destroy_node(self):
        self.device.close()
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