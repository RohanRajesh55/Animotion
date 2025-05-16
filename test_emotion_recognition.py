import cv2
from emotion_detector import predict_emotion

def main():
    # Open the default webcam (device 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Load Haar Cascade for face detection (using OpenCV's built-in cascades)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    print("Starting realtime emotion recognition. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Optionally, flip the frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert frame to grayscale for faster face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            # Extract the face ROI from the original color frame
            face_roi = frame[y:y+h, x:x+w]
            # Predict emotion using the face ROI
            emotion = predict_emotion(face_roi)
            # Draw a rectangle around the face and overlay the predicted emotion
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, emotion, (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Display the live video with soul overlay
        cv2.imshow("Live Emotion Recognition", frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()