#!/usr/bin/env python3
"""
Simplified Laptop Webcam Server
Features:
- Webcam streaming
- Countdown timer for photos (displayed outside the camera feed)
- Metadata collection (distance, object type, notes)
"""
import socket
import time
import sys
import threading
import csv
import os
from pathlib import Path
import cv2
from flask import Flask, jsonify, request, Response, redirect, url_for


class WebcamStream:
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
        self.current_frame = None
        self.countdown_active = False
        self.countdown_time = 0
        self.countdown_start = 0

    def start(self):
        """Start the camera capture"""
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)  # 0 is usually the default webcam
            # Give camera time to warm up
            time.sleep(1)

            # Check if camera opened successfully
            if not self.camera.isOpened():
                print("Error: Could not open webcam")
                return False

        self.is_running = True
        # Start background thread for capturing and updating stats
        threading.Thread(target=self._update_frame, daemon=True).start()
        return True

    def _update_frame(self):
        """Continuously update the current frame and stats"""
        last_fps_time = time.time()
        fps = 0
        frame_count = 0

        while self.is_running:
            success, frame = self.camera.read()
            if not success:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue

            # Update current frame (thread-safe)
            with self.lock:
                self.current_frame = frame

            # Calculate FPS
            frame_count += 1
            current_time = time.time()
            if current_time - last_fps_time >= 1.0:  # Update stats every second
                fps = frame_count
                frame_count = 0
                last_fps_time = current_time

                # Update stream info
                with self.lock:
                    height, width = frame.shape[:2]
                    estimated_bitrate = fps * width * height * 3 * 8 / 1000  # kbps
                    self.stream_info['bitrate'] = "{:.2f} kb/s".format(estimated_bitrate)
                    self.stream_info['latency'] = "{:.2f} ms".format(1000.0 / max(1, fps))
                    # Add FPS to the stream info
                    self.stream_info['fps'] = "{} fps".format(fps)

            # Small delay to control CPU usage
            time.sleep(0.01)

    def capture_feed(self):
        """Generator function for streaming frames"""
        while True:
            # Wait until we have a frame
            if self.current_frame is None:
                time.sleep(0.1)
                continue

            # Get the current frame (thread-safe)
            with self.lock:
                if self.current_frame is None:
                    continue
                frame = self.current_frame.copy()

            # Encode frame as JPEG
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()

            # Yield the frame in the correct format for HTTP streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

            # Add a small delay
            time.sleep(0.03)  # About 30 FPS

    def start_countdown(self, seconds=3):
        """Start a countdown before taking a picture"""
        self.countdown_active = True
        self.countdown_time = seconds
        self.countdown_start = time.time()

    def get_countdown_status(self):
        """Get the current countdown status"""
        if not self.countdown_active:
            return {"active": False, "remaining": 0}

        remaining = max(0, self.countdown_time - int(time.time() - self.countdown_start))
        if remaining == 0:
            self.countdown_active = False
            # Take picture automatically when countdown reaches zero
            return {"active": False, "remaining": 0, "complete": True}

        return {"active": True, "remaining": remaining, "complete": False}

    def take_picture(self, fileName="image.jpg", metadata=None):
        """
        Capture and save a picture with the given filename and metadata

        Args:
            fileName: Name of the file to save (default: image.jpg)
            metadata: Dictionary containing distance, object_type, and notes

        Returns:
            Path to the saved image
        """
        try:
            if self.current_frame is None:
                print("No frame available")
                return None

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

            # Copy the current frame and save it
            with self.lock:
                if self.current_frame is None:
                    return None
                img_to_save = self.current_frame.copy()

            # Save the image
            cv2.imwrite(str(file_path), img_to_save)
            print(f"Image saved to: {file_path}")

            # Save metadata to CSV if provided
            if metadata:
                self.save_metadata(file_path, metadata)

            return file_path

        except Exception as e:
            print(f"Error saving image: {e}")
            return None

    def save_metadata(self, file_path, metadata):
        """Save metadata to a CSV file"""
        try:
            # Create metadata directory if it doesn't exist
            current_directory = Path.cwd()
            metadata_dir = current_directory / 'Metadata'
            if not metadata_dir.exists():
                metadata_dir.mkdir()
                print(f"Created directory: {metadata_dir}")

            # Metadata CSV path
            csv_path = metadata_dir / 'image_metadata.csv'

            # Check if file exists to write headers
            file_exists = csv_path.exists()

            # Write to CSV
            with open(csv_path, mode='a', newline='') as file:
                fieldnames = ['file_path', 'distance', 'object_type', 'notes', 'timestamp']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write header if file is new
                if not file_exists:
                    writer.writeheader()

                # Add timestamp and file path to metadata
                metadata['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
                metadata['file_path'] = str(file_path)

                # Write row
                writer.writerow(metadata)

            print(f"Metadata saved to: {csv_path}")

        except Exception as e:
            print(f"Error saving metadata: {e}")

    def stop(self):
        """Stop the camera capture"""
        self.is_running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None


# Create Flask app
app = Flask(__name__)

# Instantiate the camera
camera = WebcamStream()


def get_ip_address():
    """Get the non-loopback IP address of the computer"""
    try:
        # This approach gets the IP address that would be used to connect to an external host
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # This doesn't actually establish a connection
        s.connect(("8.8.8.8", 80))
        ip_address = s.getsockname()[0]
        s.close()
        return ip_address
    except Exception as e:
        print(f"Error getting IP address: {e}")
        # Fallback to hostname method
        try:
            hostname = socket.gethostname()
            ip_address = socket.gethostbyname(hostname)
            return ip_address
        except socket.gaierror as e:
            return "127.0.0.1"  # Default fallback


@app.route('/')
def index():
    """Main page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Laptop Computer Vision System</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; background-color: #f8f9fa; }
            h1 { color: #333; margin-bottom: 5px; }
            h2 { color: #666; font-size: 1.2em; margin-top: 0; font-weight: normal; }
            .video-container { margin: 20px auto; max-width: 800px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }
            img { width: 100%; display: block; }
            .stats { margin: 20px auto; max-width: 400px; background: #fff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .controls { margin: 20px auto; max-width: 600px; background: #e3f2fd; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .metadata { margin: 20px auto; max-width: 600px; background: #e8f5e9; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .panel-title { background: #f5f5f5; margin: -20px -20px 15px -20px; padding: 12px; border-radius: 8px 8px 0 0; font-weight: bold; color: #424242; border-bottom: 1px solid #ddd; }
            .form-group { margin-bottom: 12px; text-align: left; }
            input[type="text"], input[type="number"], select { padding: 10px; width: 200px; margin: 5px; border-radius: 4px; border: 1px solid #ccc; box-sizing: border-box; }
            button { padding: 10px 20px; background-color: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; margin: 5px; font-weight: bold; transition: background-color 0.3s; }
            button:hover { background-color: #45a049; }
            .message { margin-top: 10px; color: #333; background: #f1f8e9; padding: 8px; border-radius: 4px; display: inline-block; }
            .countdown-controls { margin: 15px 0; background: #fff; padding: 10px; border-radius: 4px; }
            .status-container { display: flex; justify-content: space-between; }
            .status-box { flex: 1; margin: 0 5px; padding: 10px; background: #f5f5f5; border-radius: 4px; }
            .status-label { font-size: 0.9em; color: #666; }
            .status-value { font-weight: bold; color: #333; margin-top: 5px; }

            /* External countdown styles */
            .countdown-display {
                display: none;
                position: fixed;
                top: 20px;
                right: 20px;
                width: 100px;
                height: 100px;
                background-color: rgba(0, 0, 0, 0.7);
                color: white;
                font-size: 48px;
                border-radius: 50%;
                line-height: 100px;
                text-align: center;
                box-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
                z-index: 1000;
                animation: pulse 1s infinite;
            }
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.05); }
                100% { transform: scale(1); }
            }
        </style>
    </head>
    <body>
        <h1>Computer Vision Capture System</h1>
        <h2>Laptop Webcam</h2>

        <!-- External countdown display -->
        <div id="countdownDisplay" class="countdown-display"></div>

        <div class="video-container">
            <img src="/video_feed" alt="Video Feed">
        </div>

        <div class="controls">
            <div class="panel-title">Camera Controls</div>
            <div class="countdown-controls">
                <label for="countdown">Countdown Timer: </label>
                <select id="countdown" name="countdown">
                    <option value="0">No Countdown</option>
                    <option value="1">1 Second</option>
                    <option value="2">2 Seconds</option>
                    <option value="3" selected>3 Seconds</option>
                    <option value="5">5 Seconds</option>
                    <option value="10">10 Seconds</option>
                </select>
            </div>
            <form id="photoForm" action="/save_picture" method="post">
                <div class="form-group">
                    <label for="filename">Image Name: </label>
                    <input type="text" name="filename" id="filename" placeholder="Enter image name" required>
                </div>
                <input type="hidden" name="countdown" id="countdownHidden" value="3">
                <button type="submit">Capture Image</button>
            </form>
            <div id="picture-message" class="message"></div>
        </div>

        <div class="metadata">
            <div class="panel-title">Image Metadata</div>
            <form id="metadataForm" action="/save_metadata" method="post">
                <div class="form-group">
                    <label for="distance">Distance (meters): </label>
                    <input type="number" id="distance" name="distance" step="0.1" min="0" placeholder="Distance in meters">
                </div>
                <div class="form-group">
                    <label for="object_type">Object Type: </label>
                    <input type="text" id="object_type" name="object_type" placeholder="e.g., Cat, Car, Person">
                </div>
                <div class="form-group">
                    <label for="notes">Notes: </label>
                    <input type="text" id="notes" name="notes" placeholder="Additional notes">
                </div>
                <button type="button" onclick="updateMetadata()">Save Metadata</button>
            </form>
        </div>

        <div class="stats">
            <div class="panel-title">Stream Information</div>
            <div class="status-container">
                <div class="status-box">
                    <div class="status-label">Bitrate</div>
                    <div id="bitrate" class="status-value">0 kb/s</div>
                </div>
                <div class="status-box">
                    <div class="status-label">Latency</div>
                    <div id="latency" class="status-value">0 ms</div>
                </div>
                <div class="status-box">
                    <div class="status-label">FPS</div>
                    <div id="fps" class="status-value">0 fps</div>
                </div>
            </div>
        </div>

        <script>
            // Update the hidden countdown field when dropdown changes
            document.getElementById('countdown').addEventListener('change', function() {
                document.getElementById('countdownHidden').value = this.value;
            });

            // Handle the photo form submission
            document.getElementById('photoForm').addEventListener('submit', function(e) {
                e.preventDefault();

                const formData = new FormData(this);
                const countdownValue = parseInt(document.getElementById('countdown').value);

                // If countdown is selected, show it on screen
                if (countdownValue > 0) {
                    fetch('/start_countdown', {
                        method: 'POST',
                        body: formData
                    });
                    document.getElementById('picture-message').textContent = "Countdown started...";

                    // Start the external countdown
                    startExternalCountdown(countdownValue);
                } else {
                    // Take picture immediately if no countdown
                    fetch('/save_picture', {
                        method: 'POST',
                        body: formData
                    })
                    .then(response => response.text())
                    .then(data => {
                        document.getElementById('picture-message').textContent = data;
                    });
                }
            });

            // External countdown function
            function startExternalCountdown(seconds) {
                const countdownDisplay = document.getElementById('countdownDisplay');
                countdownDisplay.style.display = 'block';
                countdownDisplay.textContent = seconds;

                let remainingSeconds = seconds;
                const countdownInterval = setInterval(() => {
                    remainingSeconds--;

                    if (remainingSeconds <= 0) {
                        clearInterval(countdownInterval);
                        countdownDisplay.style.display = 'none';

                        // Check if picture was taken after countdown
                        setTimeout(checkCountdownCompleted, 500);
                    } else {
                        countdownDisplay.textContent = remainingSeconds;
                    }
                }, 1000);
            }

            // Check if countdown completed and picture was taken
            function checkCountdownCompleted() {
                fetch('/countdown_status')
                    .then(response => response.json())
                    .then(data => {
                        if (data.complete) {
                            document.getElementById('picture-message').textContent = "Picture taken after countdown!";
                        }
                    });
            }

            // Update metadata function
            function updateMetadata() {
                const formData = new FormData(document.getElementById('metadataForm'));

                fetch('/update_metadata', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.text())
                .then(data => {
                    document.getElementById('picture-message').textContent = data;
                });
            }

            // Update stats function
            function updateStats() {
                fetch('/stream_stats')
                    .then(response => response.json())
                    .then(data => {
                        document.getElementById('bitrate').textContent = data.bitrate;
                        document.getElementById('latency').textContent = data.latency;
                        if (data.fps) {
                            document.getElementById('fps').textContent = data.fps;
                        }
                    });
            }

            // Update stats every second
            setInterval(updateStats, 1000);

            // Check countdown status periodically
            setInterval(() => {
                if (document.getElementById('countdownDisplay').style.display === 'block') {
                    fetch('/countdown_status')
                        .then(response => response.json())
                        .then(data => {
                            if (!data.active && document.getElementById('countdownDisplay').style.display === 'block') {
                                document.getElementById('countdownDisplay').style.display = 'none';
                            }
                        });
                }
            }, 500);

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


@app.route('/start_countdown', methods=['POST'])
def start_countdown():
    """Start a countdown before taking a picture"""
    if request.method == 'POST':
        # Get the countdown time from the form
        countdown = int(request.form.get('countdown', 3))
        filename = request.form.get('filename', 'image.jpg')

        # Make sure the filename ends with .jpg
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            filename += '.jpg'

        # Start the countdown
        camera.start_countdown(countdown)

        # Store the filename to use when the countdown finishes
        app.config['pending_filename'] = filename

        return "Countdown started"


@app.route('/countdown_status')
def countdown_status():
    """Return the current countdown status"""
    status = camera.get_countdown_status()

    # If countdown completed, take a picture
    if status.get('complete', False):
        # Get the pending filename
        filename = app.config.get('pending_filename', 'image.jpg')

        # Get current metadata if available
        metadata = {
            'distance': app.config.get('distance', ''),
            'object_type': app.config.get('object_type', ''),
            'notes': app.config.get('notes', '')
        }

        # Take the picture
        camera.take_picture(fileName=filename, metadata=metadata)

    return jsonify(status)


@app.route('/save_picture', methods=['POST'])
def save_picture():
    """Save a picture with a custom filename"""
    if request.method == 'POST':
        # Get the filename from the form
        filename = request.form.get('filename', 'image.jpg')

        # Make sure the filename ends with .jpg
        if not filename.lower().endswith(('.jpg', '.jpeg')):
            filename += '.jpg'

        # Get current metadata if available
        metadata = {
            'distance': app.config.get('distance', ''),
            'object_type': app.config.get('object_type', ''),
            'notes': app.config.get('notes', '')
        }

        # Take the picture with the custom filename and metadata
        file_path = camera.take_picture(fileName=filename, metadata=metadata)

        if file_path:
            # Return success message
            return f"Picture taken and saved as {file_path}"
        else:
            # Return error message
            return "Error saving picture"


@app.route('/update_metadata', methods=['POST'])
def update_metadata():
    """Update the metadata to be saved with the next picture"""
    if request.method == 'POST':
        # Store the metadata in the app config for the next picture
        app.config['distance'] = request.form.get('distance', '')
        app.config['object_type'] = request.form.get('object_type', '')
        app.config['notes'] = request.form.get('notes', '')

        return "Metadata updated and will be saved with the next picture"


@app.route('/stream_stats')
def stream_stats():
    """Return the current stream statistics as JSON"""
    return jsonify(camera.stream_info)


def check_camera_available():
    """Check if any camera is available"""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("ERROR: No camera detected! Please connect a webcam.")
        return False
    cam.release()
    return True


if __name__ == '__main__':
    try:
        print("=" * 80)
        print("Computer Vision Capture System")
        print("=" * 80)

        # Check for camera
        if not check_camera_available():
            sys.exit(1)

        # Start the camera
        camera = WebcamStream()
        if not camera.start():
            print("Failed to start camera")
            sys.exit(1)

        # Initialize app config for metadata
        app.config['distance'] = ''
        app.config['object_type'] = ''
        app.config['notes'] = ''

        # Start the server
        ip = get_ip_address()
        print("Current IP address: {}".format(ip))
        print("Server starting. Access at: http://{}:5000".format(ip))
        print("Press Ctrl+C to quit")
        app.run(host='0.0.0.0', port=5000, threaded=True, debug=True)
    finally:
        # Make sure camera is properly released when server stops
        if 'camera' in locals():
            camera.stop()