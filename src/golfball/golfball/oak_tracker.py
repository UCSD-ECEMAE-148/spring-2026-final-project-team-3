import cv2
import depthai as dai
import numpy as np

# ====== CONFIGURATION ======
BLOB_PATH = "../detection_model/best.blob" 
CONFIDENCE_THRESHOLD = 0.5
# ===========================

def create_pipeline():
    pipeline = dai.Pipeline()

    # 1. Define Nodes
    cam_rgb = pipeline.create(dai.node.ColorCamera)
    detection_nn = pipeline.create(dai.node.NeuralNetwork)
    
    xout_video = pipeline.create(dai.node.XLinkOut)
    xout_nn = pipeline.create(dai.node.XLinkOut)

    xout_video.setStreamName("video")
    xout_nn.setStreamName("nn")

    # 2. Camera Setup (Pre-configured to 640x640 for the NN)
    cam_rgb.setPreviewSize(640, 640)
    cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(30)

    # 3. NN Setup
    detection_nn.setBlobPath(BLOB_PATH)

    # 4. Link Nodes (Send same 640x640 frame to both NN and Host)
    cam_rgb.preview.link(detection_nn.input)
    cam_rgb.preview.link(xout_video.input)
    detection_nn.out.link(xout_nn.input)

    return pipeline


def parse_detections(nn_data, width, height):
    """Mandatory raw tensor math for un-decoded YOLO exports."""
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

    with dai.Device(pipeline) as device:
        print("OAK Device calculation thread started.")
        q_video = device.getOutputQueue(name="video", maxSize=4, blocking=False)
        q_nn = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

        while True:
            in_video = q_video.get() 
            in_nn = q_nn.tryGet()

            if in_video is not None:
                # Get the raw OpenCV frame directly (No decoding needed)
                img = in_video.getCvFrame()
                height, width = img.shape[:2]

                if in_nn is not None:
                    dets = parse_detections(in_nn, width, height)

                    # Draw the raw bounding boxes
                    for (x1, y1, x2, y2, conf) in dets:
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(img, f"{conf*100:.1f}%", (x1, y1 - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Encode to JPEG once and yield to the web server
                _, encoded_img = cv2.imencode('.jpg', img)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + encoded_img.tobytes() + b'\r\n')


                       