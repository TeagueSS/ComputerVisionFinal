import socket
import subprocess
import time
import traceback

from Wrappers.Camera import Camera
from Wrappers.Remote import Controller
import speedtest
from flask import Flask, jsonify, request, render_template, Response

#pip install speedtest-cli
# pip install Flask
# Instantiate the camera once when the app starts
camera = Camera(0)
# Create an Instance of our controller
controller = Controller()
# Creating a flask app we can build off of for our communications ->
app = Flask(__name__)
# Checking if we have a connection
@app.route('/ping/<string:host>', methods=['GET'])
def ping(host):
    # putting all of our communications in a try block ->
    try:
        # Here we have two Seperate way of letting them connect
        # if it was through a browser with a static IP i want to tell them they're
        # Connected correctly
        output = "<p>Hello, World! You are connected to the server!</p>"
        if request.accept_mimetypes.accept_html:
            return output  # returns HTML
        else:
            # If the care doing a curl command or a ping we can return a version
            # they can use ->
            output = subprocess.check_output(['ping', '-c', '4', host], universal_newlines=True)
            return jsonify({'result': output})
    except subprocess.CalledProcessError as e:
       return jsonify({'error': str(e)}), 500

@app.route('/speedtest', methods=['GET'])
def speed_test():
   try:
        # Running a Speed test between our two connections ->
        st = speedtest.Speedtest()
        st.download()
        st.upload()
        results = st.results.dict()
        # Converting our results so they can be shown on a WebPage
        download_speed = round(results["download"] / 1_000_000, 2)  # Convert to Mbps
        upload_speed = round(results["upload"] / 1_000_000, 2)  # Convert to Mbps
        ping = results["ping"]

        # If they can accept a web page return it in that form ->
        if request.accept_mimetypes.accept_html:
            return render_template('speedtest.html', download_speed=download_speed, upload_speed=upload_speed,
                                   ping=ping)
        else:
            # Returning it in a format they can use ->
            results = st.results.dict()
            return jsonify(results)

   except Exception as e:
       return jsonify({'error': str(e)}), 500

def get_ip_address():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.gaierror as e:
        return f"Error getting IP address: {e}"

# >-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<
# >-<*>-<*>-<*>-<*>-<*>-<*>-FFMPEG for accelerating Video Feed-<*>-<*>-<*>-<*>-<
# >-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<
# >-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<

# Global dictionary to hold live stream stats
#
stream_info = {'bitrate': '0 kb/s', 'latency': '0 ms'}

def update_ffmpeg_stats(stderr_pipe):
    """
    Reads FFmpeg progress info from stderr and updates the stream_info dictionary.
    Expected progress lines are like:
      bitrate=  500.0kbits/s
      out_time_ms=12345678
    """
    for line in iter(stderr_pipe.readline, b''):
        try:
            decoded_line = line.decode('utf-8').strip()
            # Parse bitrate info
            if decoded_line.startswith("bitrate="):
                bitrate = decoded_line.split('=')[1].strip()
                stream_info['bitrate'] = bitrate
            # Parse out_time_ms for an estimated latency (if available)
            if decoded_line.startswith("out_time_ms="):
                out_time_ms = int(decoded_line.split('=')[1].strip())
                current_time_ms = int(time.time() * 1000)
                latency = current_time_ms - out_time_ms
                stream_info['latency'] = f"{latency} ms"
        except Exception:
            continue

# Global dictionary for stream stats
stream_info = {
    'bitrate': '0 kb/s',
    'latency': '0 ms'
}

# Global variables for bitrate calculation
last_stat_time = time.time()
accumulated_bytes = 0


@app.route('/video_feed')
def video_feed():

    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(camera.capture_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_stats')
def stream_stats():
    """Return the current bitrate and latency as JSON."""
    return jsonify(camera.stream_info)

@app.route('/webcam')
def webcam():
    """Render the page with the webcam stream and live stats."""
    return render_template('webcam.html')

# >-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<
# >-<*>-<*>-<*>-<*>-<*>-<*>-Getting controller inputs from the browser -<*>-<*>-<*>-<*>-<
# >-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<
# >-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<*>-<


@app.route('/controller')
def controller_page():
    return render_template('controller.html')
@app.route('/controller_input', methods=['POST'])
def controller_input():
    # Get JSON data from the POST request
    data = request.get_json()
    # Here we simply log the data and echo it back.
    # In a real application, you might process the data further.
    print("Received controller input:", data)
    controller.process_commands(data)
    return jsonify({"status": "ok", "received": data})




# Methods for showing control surfaces
@app.route("/remote_update", methods=["POST"])
def update():
    try:
        # Get the posted JSON data
        commands = request.get_json()
        print("Received commands:", commands)

        # Pass the commands to your controller instance
        controller.process_commands(commands)

        # Return a JSON response indicating success
        return jsonify({"status": "ok"})
    except Exception as e:
        # Log the full error traceback in the console
        traceback.print_exc()
        # Return a JSON error response with a 500 status code
        return jsonify({"status": "error", "error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)







if __name__ == '__main__':
   app.run(host='0.0.0.0', port=5000)
   ip = get_ip_address()
   print(f"Current IP address: {ip}")
   print(f"Connect at: {ip}/5000 and make a get request")

