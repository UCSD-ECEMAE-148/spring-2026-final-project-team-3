#!/usr/bin/env python3
from flask import Flask, Response
# Import our frame generator function directly from our tracking module
from oak_tracker import generate_frames

app = Flask(__name__)

@app.route('/')
def index():
    """Main route. Serves up the streaming stream headers directly."""
    return "<h1>ECE 148 Golf Ball Tracking Server</h1><img src='/video_feed' width='640' height='640'>"

@app.route('/video_feed')
def video_feed():
    """Feeds the camera generator data stream blocks into the browser response context."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    print("Launching isolated Flask server on port 5000...")
    # Launch server. It handles spinning up the imported device context natively
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)