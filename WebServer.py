# For Python 2.7 and 3 compatibility
from __future__ import print_function

import socket
import time
import cv2
import threading

try:
    # Python 3
    from flask import Flask, jsonify, request, Response
except ImportError:
    # Try with a more basic approach for old systems
    print("Warning: Using fallback imports. Consider upgrading to Python 3.")
    try:
        from flask import Flask, jsonify, request, Response
    except ImportError:
        print("Error: Flask not installed. Install with: sudo apt install python3-flask")
        exit(1)


# Initialize camera with OpenCV
class SimpleCamera:
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        self.stream_info = {
            'bitrate': '0 kb/s',
            'latency': '0 ms'
        }
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.is_running = False
        self.lock = threading.Lock()
        self.current_frame = None

    def start(self):
        """Start the camera capture"""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
            # Set resolution (adjust as needed)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.is_running = True
        # Start background thread for frame capture
        threading.Thread(target=self._update_frame, daemon=True).start()
        # Start background thread for calculating stats
        threading.Thread(target=self._update_stats, daemon=True).start()

    def _update_frame(self):
        """Continuously update the current frame"""
        while self.is_running:
            success, frame = self.cap.read()
            if success:
                with self.lock:
                    self.current_frame = frame
                    self.frame_count += 1
            else:
                # If camera read fails, try to reconnect
                self.cap.release()
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.camera_id)

            # Avoid high CPU usage
            time.sleep(0.01)

    def _update_stats(self):
        """Update streaming statistics"""
        while self.is_running:
            current_time = time.time()
            elapsed = current_time - self.last_frame_time

            if elapsed >= 1.0:  # Update stats every second
                with self.lock:
                    fps = self.frame_count / elapsed
                    estimated_bitrate = (self.frame_count * 640 * 480 * 3 * 8) / (elapsed * 1000)  # kbps
                    self.stream_info['bitrate'] = f"{estimated_bitrate:.2f} kb/s"
                    self.stream_info['latency'] = f"{(elapsed * 1000 / max(1, self.frame_count)):.2f} ms"
                    self.frame_count = 0
                    self.last_frame_time = current_time

            time.sleep(0.5)

    def get_frame(self):
        """Get the current frame"""
        with self.lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
            return None

    def capture_feed(self):
        """Generator function for streaming frames"""
        while True:
            frame = self.get_frame()
            if frame is not None:
                # Encode frame as JPEG
                ret, jpeg = cv2.imencode('.jpg', frame)
                if ret:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

            # Add a small delay to control frame rate
            time.sleep(0.03)  # ~30 fps

    def stop(self):
        """Stop the camera capture"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# Simple joystick controller
class SimpleController:
    def __init__(self):
        self.state = {
            'direction': 'none',
            'speed': 0,
            'buttons': {}
        }

    def process_commands(self, commands):
        """Process commands from web interface"""
        if commands:
            self.state.update(commands)
            print(f"Controller state updated: {self.state}")
            # You can add code here to control your robot/motors
            # based on the received commands


# Create Flask app
app = Flask(__name__)

# Instantiate the camera and controller
camera = SimpleCamera(0)  # Use 0 for default camera
controller = SimpleController()

# Start the camera when the app starts
camera.start()


def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.gaierror as e:
        return f"Error getting IP address: {e}"


@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route for <img> tag src attribute"""
    return Response(camera.capture_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_stats')
def stream_stats():
    """Return the current bitrate and latency as JSON"""
    return jsonify(camera.stream_info)


@app.route('/controller')
def controller_page():
    """Controller interface page"""
    return render_template('controller.html')


@app.route('/controller_input', methods=['POST'])
def controller_input():
    """Handle controller input from the web interface"""
    data = request.get_json()
    print("Received controller input:", data)
    controller.process_commands(data)
    return jsonify({"status": "ok", "received": data})


@app.route('/ping/<string:host>', methods=['GET'])
def ping(host):
    """Simple ping endpoint to test connectivity"""
    try:
        if request.accept_mimetypes.accept_html:
            return "<p>Hello, World! You are connected to the server!</p>"
        else:
            # Simple response for non-HTML requests
            return jsonify({'result': 'Connection successful'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# HTML templates needed for the app
# Using string formatting that works with Python 2.7
@app.route('/templates/<template_name>')
def get_template(template_name):
    """Create basic templates dynamically"""
    if template_name == 'index.html':
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Raspberry Pi Camera Server</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
                h1 { color: #333; }
                .video-container { margin: 20px auto; max-width: 800px; }
                img { width: 100%; border: 1px solid #ddd; }
                .stats { margin: 20px auto; max-width: 400px; background: #f0f0f0; padding: 10px; border-radius: 5px; }
                .btn { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; margin: 5px; }
                .btn:hover { background: #45a049; }
            </style>
        </head>
        <body>
            <h1>Raspberry Pi Camera Feed</h1>
            <div class="video-container">
                <img src="/video_feed" alt="Video Feed">
            </div>
            <div class="stats">
                <h3>Stream Statistics</h3>
                <p>Bitrate: <span id="bitrate">0 kb/s</span></p>
                <p>Latency: <span id="latency">0 ms</span></p>
            </div>
            <div>
                <a href="/controller"><button class="btn">Open Controller</button></a>
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
    elif template_name == 'controller.html':
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Pi Camera Controller</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; text-align: center; }
                h1 { color: #333; }
                .controls { margin: 20px auto; max-width: 400px; background: #f0f0f0; padding: 20px; border-radius: 5px; }
                .d-pad { display: grid; grid-template-columns: repeat(3, 1fr); grid-gap: 10px; margin: 20px auto; }
                .btn { padding: 20px; background: #4CAF50; color: white; border: none; cursor: pointer; font-size: 20px; border-radius: 5px; }
                .btn:hover { background: #45a049; }
                .btn:active { background: #3e8e41; }
                .spacer { visibility: hidden; }
            </style>
        </head>
        <body>
            <h1>Camera Controller</h1>
            <div class="controls">
                <div class="d-pad">
                    <div class="spacer"></div>
                    <button class="btn" id="up">↑</button>
                    <div class="spacer"></div>
                    <button class="btn" id="left">←</button>
                    <button class="btn" id="stop">■</button>
                    <button class="btn" id="right">→</button>
                    <div class="spacer"></div>
                    <button class="btn" id="down">↓</button>
                    <div class="spacer"></div>
                </div>
                <div>
                    <a href="/"><button class="btn">Back to Camera Feed</button></a>
                </div>
            </div>
            <script>
                const buttons = document.querySelectorAll('.btn');

                buttons.forEach(button => {
                    button.addEventListener('click', function() {
                        const command = this.id;
                        let directionData = {};

                        switch(command) {
                            case 'up':
                                directionData = {direction: 'forward', speed: 50};
                                break;
                            case 'down':
                                directionData = {direction: 'backward', speed: 50};
                                break;
                            case 'left':
                                directionData = {direction: 'left', speed: 50};
                                break;
                            case 'right':
                                directionData = {direction: 'right', speed: 50};
                                break;
                            case 'stop':
                                directionData = {direction: 'none', speed: 0};
                                break;
                        }

                        fetch('/controller_input', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(directionData)
                        })
                        .then(response => response.json())
                        .then(data => {
                            console.log('Success:', data);
                        })
                        .catch(error => {
                            console.error('Error:', error);
                        });
                    });
                });
            </script>
        </body>
        </html>
        """
    else:
        return "Template not found", 404


if __name__ == '__main__':
    try:
        ip = get_ip_address()
        print("Current IP address: {}".format(ip))
        print("Server starting. Access at: http://{}:5000".format(ip))
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        # Make sure camera is properly released when server stops
        camera.stop()