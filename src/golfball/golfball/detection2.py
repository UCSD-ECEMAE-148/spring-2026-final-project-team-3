import cv2
import depthai as pipeline_module  
import depthai as dai
import numpy as np
import signal
import sys

# ====== CONFIGURATION ======
BLOB_PATH = "../detection_model/good.blob"
CONFIDENCE_THRESHOLD = 0.5
ALPHA = 0.75          # Lower = smoother, Higher = snappier
MAX_BALLS = 3         # Cap total detections to prevent noise saturation
DIST_THRESH = 80      # Max pixel distance to match a ball to its previous position
DISPLAY_MODE = False   # True = display frames; False = extract data only
# ===========================

def create_pipeline():
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_nn = pipeline.create(dai.node.YoloDetectionNetwork)
    manip = pipeline.create(dai.node.ImageManip)

    xout_rgb = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)

    xout_rgb.setStreamName("rgb")
    xout_nn.setStreamName("nn")

    # ===== CAMERA CONFIG =====
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    if DISPLAY_MODE:
        cam_rgb.setFps(5)
    else:
        cam_rgb.setFps(30)

    # IMPORTANT: prevents squished golf balls
    # cam_rgb.setPreviewKeepAspectRatio(False)
    cam_rgb.setPreviewKeepAspectRatio(True)

    # ===== IMAGE MANIP (crop + safety buffer) =====
    manip.setMaxOutputFrameSize(640 * 640 * 3)

    # Optional: black out top 25% (your original idea)
    manip.initialConfig.setCropRect(
        0.0,  # left
        0.25, # top (cuts off top 25%)
        1.0,  # right
        1.0   # bottom
    )

    manip.initialConfig.setResize(640, 640)

    # ===== YOLO CONFIG =====
    detection_nn.setBlobPath(BLOB_PATH)
    detection_nn.setConfidenceThreshold(CONFIDENCE_THRESHOLD)
    detection_nn.setNumClasses(1)
    detection_nn.setCoordinateSize(4)
    detection_nn.setIouThreshold(0.5)

    # ===== PIPELINE LINKS =====
    cam_rgb.preview.link(manip.inputImage)
    manip.out.link(detection_nn.input)

    manip.out.link(xout_rgb.input)
    detection_nn.out.link(xout_nn.input)

    return pipeline


def update_tracked_balls(raw_detections, tracked_balls, next_ball_id, width, height):
    """Update tracked balls with new detections. Returns (tracked_balls, next_ball_id)."""
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

        # Match to existing tracked balls
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

    # Age tracking
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


def get_ball_data(tracked_balls):
    """Return minimal ball data: {ball_id: {x1, y1, x2, y2, confidence}}."""
    return {b_id: {"x1": data[0], "y1": data[1], "x2": data[2], "y2": data[3], "confidence": data[5]} 
            for b_id, data in tracked_balls.items()}


def display_detections(frame, tracked_balls):
    """Draw bounding boxes and labels on frame."""
    for b_id, data in tracked_balls.items():
        x1, y1, x2, y2, _, conf = data
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Ball {b_id}: {conf * 100:.1f}%"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def signal_handler(sig, frame):
    print("\nShutting down...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    pipeline = create_pipeline()
    tracked_balls = {}
    next_ball_id = 0

    with dai.Device(pipeline) as device:
        print("OAK Device connected successfully. Launching live window...")

        q_rgb = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        i = 0

        while True:
            in_rgb = q_rgb.get()  # Non-blocking; returns None if no frame
            in_nn = q_nn.tryGet()

            # Display frame if available (even if detections aren't)
            if in_rgb is not None:
                frame = in_rgb.getCvFrame()
                height, width = frame.shape[:2]

                # Process detections if available
                if in_nn is not None:
                    raw_detections = sorted(in_nn.detections, key=lambda x: x.confidence, reverse=True)[:MAX_BALLS]
                    tracked_balls, next_ball_id = update_tracked_balls(raw_detections, tracked_balls, next_ball_id, width, height)

                if DISPLAY_MODE:
                    display_detections(frame, tracked_balls)
                    cv2.imshow("OAK Detection", frame)
                    if cv2.waitKey(1) == ord('q'):
                        break
                else:
                    ball_data = get_ball_data(tracked_balls)
                    if i % 1 == 0:
                        print(ball_data)

                i += 1


if __name__ == "__main__":
    main()