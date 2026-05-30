#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import depthai as dai
import numpy as np
from custom_interfaces.msg import DetectedBall, DetectedBalls
from std_msgs.msg import Header
import os
import threading
import time
from flask import Flask, Response

# ====== CONFIGURATION ======
BLOB_PATH = "/home/projects/ros2_ws/src/golfball/detection_model/best.blob"
CONFIDENCE_THRESHOLD = 0.3
MAX_BALLS = 3
ALPHA = 1.5
DIST_THRESH = 80
SHOW_DISPLAY = False   # TOGGLE: True = Process video overlay for Web, False = Fast coordinates only
# ===========================

app = Flask(__name__)
latest_web_jpeg = b''
frame_lock = threading.Lock()


def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    nn = pipeline.create(dai.node.NeuralNetwork)
    
    manip_bgr = pipeline.create(dai.node.ImageManip)
    manip_nv12 = pipeline.create(dai.node.ImageManip)
    video_enc = pipeline.create(dai.node.VideoEncoder)

    xout_mjpeg = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)
    
    xout_mjpeg.setStreamName("mjpeg")
    xout_nn.setStreamName("nn")

    # ===== CAMERA CONFIG =====
    cam.setPreviewSize(640, 640)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setFps(30)
    cam.setPreviewKeepAspectRatio(True)

    # ===== MANIP BGR CONFIG (For Neural Network) =====
    manip_bgr.setMaxOutputFrameSize(640 * 640 * 3)
    manip_bgr.setResize(640, 640)
    manip_bgr.setFrameType(dai.RawImgFrame.Type.BGR888p)

    # ===== MANIP NV12 CONFIG (For Video Encoder) =====
    manip_nv12.setMaxOutputFrameSize(640 * 640 * 3)
    manip_nv12.setResize(640, 640)
    manip_nv12.setFrameType(dai.RawImgFrame.Type.NV12)

    # ===== VIDEO ENCODER CONFIG =====
    video_enc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
    video_enc.setQuality(80)
    
    nn.setBlobPath(BLOB_PATH)

    # ===== PIPELINE LINKS =====
    cam.preview.link(manip_bgr.inputImage)
    cam.preview.link(manip_nv12.inputImage)
    
    manip_bgr.out.link(nn.input)
    manip_nv12.out.link(video_enc.input)

    video_enc.bitstream.link(xout_mjpeg.input)
    nn.out.link(xout_nn.input)
    return pipeline


def make_kalman():
    kf = cv2.KalmanFilter(6, 4)
    kf.transitionMatrix = np.array([[1,0,1,0,0,0],[0,1,0,1,0,0],[0,0,1,0,0,0],[0,0,0,1,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.float32)
    kf.measurementMatrix = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,1,0],[0,0,0,0,0,1]], dtype=np.float32)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.5
    kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 0.1
    kf.errorCovPost = np.eye(6, dtype=np.float32)
    return kf


class Track:
    def __init__(self, tid, x1, y1, x2, y2, conf):
        self.id, self.conf, self.age = tid, conf, 0
        self.kf = make_kalman()
        
        # Save the actual width and height measured by YOLO on frame 1
        self.w = int(x2 - x1)
        self.h = int(y2 - y1)
        
        # If the box is accidentally tiny, enforce a minimum 30x30 pixel size
        if self.w < 10: self.w = 30
        if self.h < 10: self.h = 30
        
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        self.kf.statePost = np.array([[cx], [cy], [0], [0], [float(self.w)], [float(self.h)]], dtype=np.float32)

    def predict(self): 
        self.kf.predict()

    def update(self, x1, y1, x2, y2, conf):
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        
        # Keep updating the box dimensions with the latest real sensor readings
        self.w = int(x2 - x1)
        self.h = int(y2 - y1)
        
        self.kf.correct(np.array([[cx], [cy], [float(self.w)], [float(self.h)]], dtype=np.float32))
        self.kf.statePost[0,0], self.kf.statePost[1,0] = cx, cy
        self.conf, self.age = conf, 0

    def get_box(self):
        s = self.kf.statePost
        cx, cy = s[0,0], s[1,0]
        
        # Natively draw the box around the tracked center using our stable width/height variables
        return int(cx - self.w / 2), int(cy - self.h / 2), int(cx + self.w / 2), int(cy + self.h / 2)


class BallDetectionNode(Node):
    def __init__(self):
        super().__init__('ball_detection_node')
        
        self.pub = self.create_publisher(DetectedBalls, 'detected_balls', 10)
        self.tracks, self.next_id = {}, 0
        
        self.device = dai.Device(create_pipeline())
        self.q_mjpeg = self.device.getOutputQueue("mjpeg", 1, False)
        self.q_nn = self.device.getOutputQueue("nn", 1, False)
        
        self.running = True
        self.thread = threading.Thread(target=self.device_loop, daemon=True)
        self.thread.start()
        self.get_logger().info(f'Ball Detection Node initialized. Show Display = {SHOW_DISPLAY}')

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
        global latest_web_jpeg
        while self.running and rclpy.ok():
            in_mjpeg = self.q_mjpeg.get()
            in_nn = self.q_nn.get()

            W, H = 640, 640

            if in_nn is not None:
                dets = self.parse_detections(in_nn, W, H)
                current_frame_matches = {}

                for (rx1, ry1, rx2, ry2, conf) in dets:
                    rcx, rcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2 
                    matched_id = None
                    best_dist = DIST_THRESH

                    for b_id, data_ball in self.tracks.items():
                        prev_cx = (data_ball.kf.statePost[0,0])
                        prev_cy = (data_ball.kf.statePost[1,0])
                        distance = np.hypot(rcx - prev_cx, rcy - prev_cy)
                        if distance < best_dist:
                            best_dist = distance
                            matched_id = b_id

                    if matched_id is None:
                        matched_id = self.next_id
                        self.next_id += 1
                        current_frame_matches[matched_id] = [rx1, ry1, rx2, ry2, 0, conf]
                    else:
                        prev = self.tracks[matched_id]
                        # Extract previous box coords for geometric calculations
                        p_box = prev.get_box()
                        sm_x1 = int(ALPHA * rx1 + (1 - ALPHA) * p_box[0])
                        sm_y1 = int(ALPHA * ry1 + (1 - ALPHA) * p_box[1])
                        sm_x2 = int(ALPHA * rx2 + (1 - ALPHA) * p_box[2])
                        sm_y2 = int(ALPHA * ry2 + (1 - ALPHA) * p_box[3]) 
                        sm_conf = ALPHA * conf + (1 - ALPHA) * prev.conf
                        current_frame_matches[matched_id] = [sm_x1, sm_y1, sm_x2, sm_y2, 0, sm_conf]

                for b_id in list(self.tracks.keys()):
                    if b_id not in current_frame_matches:
                        self.tracks[b_id].age += 1
                        if self.tracks[b_id].age > 5: del self.tracks[b_id]
                    else:
                        m = current_frame_matches[b_id]
                        self.tracks[b_id].update(m[0], m[1], m[2], m[3], m[5])

                for b_id, data_ball in current_frame_matches.items():
                    if b_id not in self.tracks:
                        self.tracks[b_id] = Track(b_id, data_ball[0], data_ball[1], data_ball[2], data_ball[3], data_ball[5])

            # Publish the fast numeric coordinates to driving nodes
            msg = DetectedBalls()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_optical_frame'
            for tid, t in self.tracks.items():
                x1, y1, x2, y2 = t.get_box()
                b = DetectedBall()
                b.id, b.x1, b.y1, b.x2, b.y2, b.confidence = tid, float(max(0,x1)), float(max(0,y1)), float(min(W,x2)), float(min(H,y2)), float(t.conf)
                msg.balls.append(b)
            self.pub.publish(msg)

            if not SHOW_DISPLAY:
                # If display is off, wipe the Flask buffer clear and skip frame processing entirely!
                with frame_lock:
                    latest_web_jpeg = b''
                continue

            # Check our global variable toggle
            jpeg_bytes = in_mjpeg.getData()
            
            if self.tracks:
                img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                for tid, t in self.tracks.items():
                    x1, y1, x2, y2 = t.get_box()
                    cv2.rectangle(img, (max(0,x1), max(0,y1)), (min(W,x2), min(H,y2)), (0, 255, 0), 2)
                    cv2.putText(img, f"Ball {tid}: {t.conf*100:.1f}%", (max(0,x1), max(0,y1)-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                _, encoded_img = cv2.imencode('.jpg', img)
                jpeg_bytes = encoded_img.tobytes()
            else:
                if isinstance(jpeg_bytes, np.ndarray):
                    jpeg_bytes = jpeg_bytes.tobytes()

            with frame_lock:
                latest_web_jpeg = jpeg_bytes

    def destroy_node(self):
        self.running = False
        self.thread.join()
        self.device.close()
        super().destroy_node()


# ===== FLASK WEB ENGINE Worker =====

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
        time.sleep(0.03)


@app.route('/')
def index():
    return "<h1>ECE 148 Telemetry Stream</h1><img src='/video_feed' width='640' height='640'>"


@app.route('/video_feed')
def video_feed():
    return Response(generate_web_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


def run_flask():
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)


def main(args=None):
    rclpy.init(args=args)
    
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    node = BallDetectionNode()
    from rclpy.executors import MultiThreadedExecutor
    ex = MultiThreadedExecutor(num_threads=2)
    ex.add_node(node)
    
    try: 
        ex.spin()
    except KeyboardInterrupt: 
        pass
    finally: 
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()