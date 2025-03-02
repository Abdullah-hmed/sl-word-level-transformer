import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import torch
from model import SignLanguageTransformer

# Load the trained model
model = SignLanguageTransformer(num_classes=30)
model.load_state_dict(torch.load("sl_transformer_30.pth", map_location=torch.device('cpu')))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize MediaPipe models
mp_hands = mp.solutions.hands.Hands()
mp_pose = mp.solutions.pose.Pose()
mp_face = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
mp_drawing = mp.solutions.drawing_utils  # For visualization

# Define landmark indices
filtered_hand = list(range(21))
filtered_pose = [11, 12, 13, 14, 15, 16]
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
                 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
                 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154,
                 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191,
                 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
                 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
                 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
                 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
                 415, 454, 466, 468, 473]

# Number of landmarks per type
HAND_NUM = len(filtered_hand)
POSE_NUM = len(filtered_pose)
FACE_NUM = len(filtered_face)
TOTAL = HAND_NUM*2+POSE_NUM+FACE_NUM
print(f"Total number of landmarks: Hands: {HAND_NUM}, Pose: {POSE_NUM}, Face: {FACE_NUM}, Total: {TOTAL}")
# Buffer to store past frames
FRAME_WINDOW = 10
buffer = deque(maxlen=FRAME_WINDOW)

label_map = {
    0: 'apple', 1: 'backpack', 2: 'capture', 3: 'circuswheel', 4: 'count',
    5: 'downhill', 6: 'easy to do', 7: 'gym', 8: 'hug', 9: 'impossible',
    10: 'influence', 11: 'joke', 12: 'juice', 13: 'nice', 14: 'nose',
    15: 'park', 16: 'peacock', 17: 'ponder', 18: 'powder', 19: 'pull convince',
    20: 'rose', 21: 'scold', 22: 'smooth', 23: 'soccer', 24: 'social',
    25: 'society', 26: 'stink', 27: 'tube', 28: 'we', 29: 'weave'
}

def get_frame_landmarks(frame):
    """Extract hand, pose, and face landmarks from a frame."""
    all_landmarks = np.zeros((HAND_NUM * 2 + POSE_NUM + FACE_NUM, 3))
    
    results_hands = mp_hands.process(frame)
    results_pose = mp_pose.process(frame)
    results_face = mp_face.process(frame)

    if results_hands.multi_hand_landmarks:
        for i, hand_landmarks in enumerate(results_hands.multi_hand_landmarks):
            if results_hands.multi_handedness[i].classification[0].index == 0:
                all_landmarks[:HAND_NUM, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])
            else:
                all_landmarks[HAND_NUM:HAND_NUM * 2, :] = np.array([(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark])

    if results_pose.pose_landmarks:
        all_landmarks[HAND_NUM * 2:HAND_NUM * 2 + POSE_NUM, :] = np.array(
            [(lm.x, lm.y, lm.z) for lm in results_pose.pose_landmarks.landmark])[filtered_pose]

    if results_face.multi_face_landmarks:
        all_landmarks[HAND_NUM * 2 + POSE_NUM:, :] = np.array(
            [(lm.x, lm.y, lm.z) for lm in results_face.multi_face_landmarks[0].landmark])[filtered_face]

    return all_landmarks, results_hands, results_pose, results_face

def draw_landmarks(frame, results_hands, results_pose, results_face):
    """Draw hand, pose, and face landmarks on the webcam feed."""
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS, 
                                      mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1))

    if results_pose.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results_pose.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=1))

    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            mp_drawing.draw_landmarks(frame, face_landmarks, None, 
                                      mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1, circle_radius=1))

def predict_sign(buffer):
    """Pass collected frames through the model for prediction."""
    if len(buffer) < FRAME_WINDOW:
        return "Waiting..."
    
    input_sequence = np.array(buffer)  # Convert buffer to NumPy array
    input_tensor = torch.tensor(input_sequence, dtype=torch.float32).to(device)
    num_frames = input_tensor.shape[0]

    expected_shape = (1, num_frames, 180, 3)  # Correct expected shape
    expected_total_size = np.prod(expected_shape)  # 1 * T * 180 * 3

    if input_tensor.numel() != expected_total_size:
        return f"Error: Shape Mismatch {expected_total_size} vs {input_tensor.numel()}"

    # Correct reshape
    input_tensor = input_tensor.view(expected_shape)

    with torch.no_grad():
        lengths = torch.tensor([num_frames], dtype=torch.long).to(device)
        prediction = model(input_tensor, lengths)  # Pass through model
        predicted_class = torch.argmax(prediction, dim=1).item()

    predicted_label = label_map.get(predicted_class, "Unknown")
    return predicted_label

frame_num = 0

# OpenCV Webcam Capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Flip frame to act as a mirror
    frame = cv2.flip(frame, 1)

    # Convert to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Extract landmarks
    landmarks, results_hands, results_pose, results_face = get_frame_landmarks(rgb_frame)

    # Append to buffer
    buffer.append(landmarks)

    if len(buffer) > FRAME_WINDOW:
        buffer.pop(0)  # Remove the oldest frame

    
    # Draw landmarks on frame
    # draw_landmarks(frame, results_hands, results_pose, results_face)

    predicted_label = predict_sign(buffer)

    # Display buffer count
    cv2.putText(frame, f"Prediction: {predicted_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    # Show video feed
    cv2.imshow("Live Landmark Extraction", frame)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
