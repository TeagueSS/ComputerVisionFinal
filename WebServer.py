#!/usr/bin/env python3
"""
Simple Raspberry Pi Camera Web Server using PiCamera
Works with the native Raspberry Pi camera module
"""
import socket
import time
import sys
import io
import threading
from flask import Flask, jsonify, request, Response
import os
from pathlib import Path

# Debug information about Python paths
print("Python version:", sys.version)
print("Python path:", sys.path)

# Try importing PiCamera with error handling
try:
    from picamera import PiCamera

    print("PiCamera successfully imported.")
except ImportError as e:
    print("ERROR: Could not import PiCamera:", e)
    print("\nTry installing PiCamera with:")
    print("sudo apt install python3-picamera")
    sys.exit(1)


# Initialize camera with PiCamera
class PiCameraStream:
    def __init__(self):
        self.camera = None
        self.stream_info = {
            'bitrate': '0 kb/s',
            'latency': '0 ms'
        }
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.is_running = False
        self.lock = threading.Lock()

    def start(self):
        """Start the camera capture"""
        if self.camera is None:
            self.camera = PiCamera()
            self.camera.resolution = (640, 480)
            self.camera.framerate = 24
            # Give camera time to warm up
            time.sleep(2)

        self.is_running = True
        # Start background thread for calculating stats
        threading.Thread(target=self._update_stats, daemon=True).start()

    def _update_stats(self):
        """Update streaming statistics"""
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time

            if elapsed >= 1.0:  # Update stats every second
                with self.lock:
                    estimated_bitrate = self.camera.framerate * 640 * 480 * 3 * 8 / 1000  # kbps
                    self.stream_info['bitrate'] = "{:.2f} kb/s".format(estimated_bitrate)
                    self.stream_info['latency'] = "{:.2f} ms".format(1000.0 / self.camera.framerate)
                    self.last_frame_time = current_time

            time.sleep(0.5)

    def capture_feed(self):
        """Generator function for streaming frames"""
        while True:
            # Create in-memory stream
            stream = io.BytesIO()
            self.camera.capture(stream, 'jpeg', use_video_port=True)
            # Reset stream position
            stream.seek(0)
            # Get the frame
            frame = stream.read()

            # Yield the frame in the correct format for HTTP streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

            # Add a small delay to control frame rate
            time.sleep(1.0 / self.camera.framerate)

    def stop(self):
        """Stop the camera capture"""
        self.is_running = False
        if self.camera is not None:
            self.camera.close()
            self.camera = None


# Create Flask app
app = Flask(__name__)

# Instantiate the camera
camera = PiCameraStream()

# Start the camera when the app starts
camera.start()


def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.gaierror as e:
        return "Error getting IP address: {}".format(e)


@app.route('/')
def index():
    """Main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raspberry Pi Camera Feed</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
            h1 { color: #333; }
            .video-container { margin: 20px auto; max-width: 800px; }
            img { width: 100%; border: 1px solid #ddd; }
            .stats { margin: 20px auto; max-width: 400px; background: #f0f0f0; padding: 10px; border-radius: 5px; }
        </style>
    </head>
    <body>
        <h1>Raspberry Pi Camera Feed (PiCamera)</h1>
        <div class="video-container">
            <img src="/video_feed" alt="Video Feed">
        </div>
        <div class="stats">
            <h3>Stream Statistics</h3>
            <p>Bitrate: <span id="bitrate">0 kb/s</span></p>
            <p>Latency: <span id="latency">0 ms</span></p>
        </div>
        <script>
            function updateStats() {
                fetch('/stream_stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('bitrate').textContent = data.bitrate;
                        document.getElementById('latency').textContent = data.latency;
                    });
            }

            // Update stats every second
            setInterval(updateStats, 1000);
        </script>
    </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    """Video streaming route for <img> tag src attribute"""
    return Response(camera.capture_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')



# Creating a new subpath to hold our image path ->
@app.route('/take_picture')
def take_picture():
    # Getting our photo information from the API request
    # Get the filename from request parameters
    fileName = request.args.get('fileName', 'image.jpg')  # Default filename if none provided

    # Take the file name and save the image in the local directory under that name
    # Check if the directory is created
    # If not create the /Pictures Directory
    # if it is then save our picture at the maximum size possible

    # Getting our current directory
    current_directory = Path.cwd()
    # Printing our current direcotry
    print(current_directory)

#TODO
# Creat a take picture method that calls the currently active camera and saves the photo locally



def button():


@app.route('/stream_stats')
def stream_stats():
    """Return the current bitrate and latency as JSON"""
    return jsonify(camera.stream_info)


if __name__ == '__main__':
    try:
        ip = get_ip_address()
        print("Current IP address: {}".format(ip))
        print("Server starting. Access at: http://{}:5000".format(ip))
        print("Press Ctrl+C to quit")
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        # Make sure camera is properly released when server stops
        camera.stop()