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
        self.picture_number = 1
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
            self.camera.resolution = (2592, 1944)
            self.camera.framerate = 30
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
                    # Convert fractions to float before formatting
                    framerate_float = float(self.camera.framerate)
                    estimated_bitrate = framerate_float * 640 * 480 * 3 * 8 / 1000  # kbps
                    self.stream_info['bitrate'] = "{:.2f} kb/s".format(estimated_bitrate)
                    self.stream_info['latency'] = "{:.2f} ms".format(1000.0 / framerate_float)
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

    def take_picture(self, fileName="image.jpg"):
        """
        Capture and save a picture with the given filename

        Args:
            fileName: Name of the file to save (default: image.jpg)

        Returns:
            Path to the saved image
        """
        try:
            # Getting our current directory
            current_directory = Path.cwd()
            print(f"Current directory: {current_directory}")

            # Create Pictures directory if it doesn't exist
            pictures_dir = current_directory / 'Pictures'
            if not pictures_dir.exists():
                pictures_dir.mkdir()
                print(f"Created directory: {pictures_dir}")

            # Create a filename that includes the picture number
            numbered_filename = f"{self.picture_number}_{fileName}"

            # Complete path for saving the image
            file_path = pictures_dir / numbered_filename
            self.picture_number += 1

            # Capture the image
            self.camera.capture(str(file_path))
            print(f"Image saved to: {file_path}")

            return file_path

        except Exception as e:
            print(f"Error saving image: {e}")
            # Return None or raise the exception to indicate failure
            return None

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
            .controls { margin: 20px auto; max-width: 400px; background: #e8f4ff; padding: 15px; border-radius: 5px; }
            input[type="text"] { padding: 8px; width: 200px; margin-right: 10px; border-radius: 4px; border: 1px solid #ccc; }
            button { padding: 8px 15px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
            button:hover { background-color: #45a049; }
            .message { margin-top: 10px; color: #333; }
        </style>
    </head>
    <body>
        <h1>Raspberry Pi Camera Feed (PiCamera)</h1>
        <div class="video-container">
            <img src="/video_feed" alt="Video Feed">
        </div>
        <div class="controls">
            <h3>Take a Picture</h3>
            <form action="/save_picture" method="post">
                <input type="text" name="filename" placeholder="Enter image name" required>
                <button type="submit">Save Picture</button>
            </form>
            <div id="picture-message" class="message"></div>
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

            // Check for message parameter in URL
            const urlParams = new URLSearchParams(window.location.search);
            const message = urlParams.get('message');
            if (message) {
                document.getElementById('picture-message').textContent = decodeURIComponent(message);
            }
        </script>
    </body>
    </html>
    """


@app.route('/video_feed')
def video_feed():
    """Video streaming route for <img> tag src attribute"""
    return Response(camera.capture_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Original take_picture route
@app.route('/take_picture')
def take_picture():
    camera.take_picture()
    return f"Picture taken and saved as Took Picture!"


# New route to handle the form submission
@app.route('/save_picture', methods=['POST'])
def save_picture():
    """Save a picture with a custom filename"""
    if request.method == 'POST':
        # Get the filename from the form
        filename = request.form.get('filename', 'image.jpg')

        # Make sure the filename ends with .jpg
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            filename += '.jpg'

        # Take the picture with the custom filename
        file_path = camera.take_picture(fileName=filename)

        if file_path:
            # Redirect back to the main page with a success message
            return f"Picture taken and saved as {file_path}", 200
        else:
            # Redirect back to the main page with an error message
            return "Error saving picture", 500


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