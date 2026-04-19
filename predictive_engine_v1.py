import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time

# --- Kalman Filter Setup ---
def create_kalman_filter():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    return kalman

def main():
    # 1. Initialize MediaPipe Task
    base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
    detector = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(0)
    kf = create_kalman_filter()
    
    print("[KineticFPS] Predictive Engine V1 (Tasks API) Started.")
    print(" - RED: Raw AI Tracking")
    print(" - GREEN: Neural Prediction")

    while cap.isOpened():
        success, frame = cap.read()
        if not success: break

        frame = cv2.flip(frame, 1)
        rgb_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # 2. Detect landmarks
        detection_result = detector.detect(rgb_frame)

        if detection_result.hand_landmarks:
            # MediaPipe Tasks returns a list of normalized landmarks
            # Landmark 8 is the index finger tip
            index_tip = detection_result.hand_landmarks[0][8]
            h, w, _ = frame.shape
            cx, cy = int(index_tip.x * w), int(index_tip.y * h)

            # 3. Kalman Update & Predict
            measured = np.array([[np.float32(cx)], [np.float32(cy)]])
            kf.correct(measured)
            
            prediction = kf.predict()
            px, py = int(prediction[0][0]), int(prediction[1][0])

            # Draw visuals
            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), -1) # Raw
            cv2.circle(frame, (px, py), 15, (0, 255, 0), 2)  # Predicted
            cv2.line(frame, (cx, cy), (px, py), (255, 255, 0), 2)

        cv2.putText(frame, "KineticFPS Prototype", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('KineticFPS', frame)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break

    detector.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
