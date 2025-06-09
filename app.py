from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from transformer_setup import FRAME_WINDOW,buffer, get_frame_landmarks, predict_sign
from collections import deque
import base64
import numpy as np
import cv2


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

@app.route("/")
def index():
    return render_template("index.html")

@socketio.on('connect')
def handle_connect():
    print(f"Client connected")
    buffer = deque(maxlen=FRAME_WINDOW)

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected")
    buffer.clear()

@socketio.on('frame')
def handle_frame(frame):
    sid = request.sid
    
    # Decode base64 image
    try:
        frame = frame['image']
        # frame = cv2.flip(frame)
        frame = np.frombuffer(base64.b64decode(frame), dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is None or frame.shape == ():
            print("Error: Invalid frame received")
            return None, None, None, None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        
    except Exception as e:
        print(f"Decoding error: {e}")
        return

    # Extract landmarks
    landmarks, results_hands, results_pose, results_face = get_frame_landmarks(frame)
    
    if landmarks is not None:
        buffer.append(landmarks)

        # Run inference when buffer is full
        if len(buffer) == FRAME_WINDOW:
            
            predicted_label, confidence = predict_sign(buffer)
            
            # Emit asynchronously to prevent blocking
            socketio.emit('prediction', {
                'label': predicted_label,
                'confidence': confidence
            })


if __name__ == '__main__':
    socketio.run(app, port=2000, debug=True)