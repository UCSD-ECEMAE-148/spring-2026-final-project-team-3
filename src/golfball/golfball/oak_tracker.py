import cv2
import depthai as dai
import numpy as np

# ====== CONFIGURATION ======
BLOB_PATH = "../detection_model/best.blob" 
CONFIDENCE_THRESHOLD = 0.5
ALPHA = 1.0    # smoothing factor for box coordinates and confidence         
MAX_BALLS = 3         
DIST_THRESH = 80      
# ===========================

def create_pipeline():
    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    
    manip_bgr = pipeline.create(dai.node.ImageManip)
    manip_nv12 = pipeline.create(dai.node.ImageManip)
    
    video_enc = pipeline.create(dai.node.VideoEncoder)

    xout_mjpeg = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)

    xout_mjpeg.setStreamName("mjpeg")
    xout_nn.setStreamName("nn")

    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)
    cam_rgb.setPreviewKeepAspectRatio(True)

    manip_bgr.setMaxOutputFrameSize(640 * 640 * 3)
    manip_bgr.initialConfig.setCropRect(0.0, 0.25, 1.0, 1.0) 
    manip_bgr.setResize(640, 640)
    manip_bgr.setFrameType(dai.RawImgFrame.Type.BGR888p) 

    manip_nv12.setMaxOutputFrameSize(640 * 640 * 3)
    manip_nv12.initialConfig.setCropRect(0.0, 0.25, 1.0, 1.0)
    manip_nv12.setResize(640, 640)
    manip_nv12.setFrameType(dai.RawImgFrame.Type.NV12) 

    video_enc.setDefaultProfilePreset(30, dai.VideoEncoderProperties.Profile.MJPEG)
    video_enc.setQuality(80)

    detection_nn.setBlobPath(BLOB_PATH)

    cam_rgb.preview.link(manip_bgr.inputImage)
    cam_rgb.preview.link(manip_nv12.inputImage)
    
    manip_bgr.out.link(detection_nn.input)
    manip_nv12.out.link(video_enc.input)

    video_enc.bitstream.link(xout_mjpeg.input)
    detection_nn.out.link(xout_nn.input)

    return pipeline


def parse_detections(nn_data, width, height):
    raw_data = nn_data.getLayerFp16('output0') if hasattr(nn_data, 'getLayerFp16') else nn_data.getFirstTensor()
    if raw_data is None: return []
        
    raw = np.array(raw_data).reshape(5, 8400)
    dets = []
    for i in range(8400):
        conf = float(raw[4, i])
        if conf < CONFIDENCE_THRESHOLD: continue
        cx, cy, w, h = raw[0,i]/640, raw[1,i]/640, raw[2,i]/640, raw[3,i]/640
        dets.append((cx-w/2, cy-h/2, cx+w/2, cy+h/2, conf))
        
    if not dets: return []
    boxes = [[d[0]*width, d[1]*height, (d[2]-d[0])*width, (d[3]-d[1])*height] for d in dets]
    idx = cv2.dnn.NMSBoxes(boxes, [d[4] for d in dets], CONFIDENCE_THRESHOLD, 0.5)
    if len(idx) == 0: return []
    return [(max(0,int(dets[i][0]*width)), max(0,int(dets[i][1]*height)),
             min(width,int(dets[i][2]*width)), min(height,int(dets[i][3]*height)),
             dets[i][4]) for i in idx.flatten()]


def generate_frames():
    """Generator loop that initializes the OAK camera and yields tracked JPEG frames."""
    pipeline = create_pipeline()
    tracked_balls = {}
    next_ball_id = 0

    with dai.Device(pipeline) as device:
        print("OAK Device calculation thread started.")
        q_mjpeg = device.getOutputQueue(name="mjpeg", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        while True:
            in_mjpeg = q_mjpeg.get() 
            in_nn = q_nn.tryGet()

            if in_mjpeg is not None:
                width, height = 640, 640 

                if in_nn is not None:
                    dets = parse_detections(in_nn, width, height)
                    current_frame_matches = {}

                    for (rx1, ry1, rx2, ry2, conf) in dets:
                        rcx, rcy = (rx1 + rx2) // 2, (ry1 + ry2) // 2 
                        matched_id = None
                        best_dist = DIST_THRESH

                        for b_id, data_ball in tracked_balls.items():
                            prev_cx = (data_ball[0] + data_ball[2]) // 2
                            prev_cy = (data_ball[1] + data_ball[3]) // 2
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
                            if tracked_balls[b_id][4] > 5: del tracked_balls[b_id]
                        else:
                            tracked_balls[b_id] = current_frame_matches[b_id]

                    for b_id, data_ball in current_frame_matches.items():
                        tracked_balls[b_id] = data_ball

                jpeg_bytes = in_mjpeg.getData()

                # Overlay box drawings if balls are detected
                if tracked_balls and in_nn is not None:
                    img = cv2.imdecode(np.frombuffer(jpeg_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
                    for b_id, data_ball in tracked_balls.items():
                        x1, y1, x2, y2, _, conf = data_ball
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"Ball {b_id}: {conf*100:.1f}%", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    _, encoded_img = cv2.imencode('.jpg', img)
                    jpeg_bytes = encoded_img.tobytes()

                
                # Ensure the JPEG data is converted to standard Python bytes before concatenation
                if isinstance(jpeg_bytes, np.ndarray):
                    raw_bytes = jpeg_bytes.tobytes()
                else:
                    raw_bytes = bytes(jpeg_bytes)

                # Yield back standard MJPEG formatted blocks safely
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + raw_bytes + b'\r\n')


    