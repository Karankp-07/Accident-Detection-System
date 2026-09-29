import sys
import cv2
import numpy as np

import config
from detection import AccidentDetectionModel


def run_camera_feed(camera_index: int = 0, fallback_demo: str = "Demo.gif"):
    print("Loading Accident Detection Model...")
    model = AccidentDetectionModel(config.MODEL_JSON_PATH, config.MODEL_WEIGHTS_PATH)
    print("Model initialized.")

    print(f"Connecting to camera index {camera_index}...")
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened() or not cap.read()[0]:
        print(f"Webcam not available on index {camera_index}. Switching to demo clip: {fallback_demo}")
        cap.release()
        cap = cv2.VideoCapture(fallback_demo)

    font = cv2.FONT_HERSHEY_SIMPLEX
    print("Monitoring stream. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Stream completed or disconnected.")
            break

        # Convert to RGB and resize to match model input dimensions
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        roi = cv2.resize(rgb_frame, (250, 250))

        pred, prob = model.predict_accident(roi[np.newaxis, :, :])

        if pred == "Accident":
            confidence = round(float(prob[0][0]) * 100, 2)
            cv2.rectangle(frame, (0, 0), (320, 45), (0, 0, 0), -1)
            cv2.putText(frame, f"{pred}: {confidence}%", (15, 32), font, 0.9, (0, 0, 255), 2)
            if confidence >= 90:
                print(f"[ALERT] Accident event detected with {confidence}% confidence!")
        else:
            confidence = round(float(prob[0][1]) * 100, 2)
            cv2.rectangle(frame, (0, 0), (280, 40), (0, 0, 0), -1)
            cv2.putText(frame, f"Normal: {confidence}%", (15, 30), font, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Accident Monitor", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    run_camera_feed()