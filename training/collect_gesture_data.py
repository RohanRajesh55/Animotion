import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

gesture_sequences = []

# Start video capture
cap = cv2.VideoCapture(0)

while len(gesture_sequences) < 200:  # Reduced data collection size for efficiency
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for lm in face_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])
            gesture_sequences.append(landmarks)

cap.release()
cv2.destroyAllWindows()

# Convert to NumPy array and save in compressed format
gesture_sequences = np.array(gesture_sequences)
np.savez_compressed("data/gesture_sequences.npz", X=gesture_sequences)

print("Gesture dataset saved in compressed format!")
