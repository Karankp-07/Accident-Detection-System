import sys
import time
import argparse
import logging
import cv2
import numpy as np

import config
from detection import AccidentDetectionModel

# Cross-platform sound handling
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

# Setup structured logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("AccidentDetector")


def play_alarm(beeps: int = 2):
    if not config.ENABLE_SOUND_ALERT:
        return
    for _ in range(beeps):
        if HAS_WINSOUND:
            winsound.Beep(1500, 1000)
            time.sleep(0.3)
        else:
            print("\a", end="", flush=True)
            time.sleep(0.5)


def send_emergency_alert(phone_number: str, message: str):
    if not config.ENABLE_WHATSAPP_ALERT:
        logger.info("WhatsApp alerting disabled by configuration.")
        return

    try:
        import pywhatkit as pw
        logger.info(f"Dispatching emergency alert to {phone_number}...")
        pw.sendwhatmsg_instantly(phone_number, message, wait_time=10, tab_close=True)
        logger.info("Alert message dispatched successfully.")
    except Exception as err:
        logger.error(f"Failed to dispatch emergency WhatsApp alert: {err}")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Real-Time Accident Detection System with Automated Emergency Alerts"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=config.DEFAULT_VIDEO_SOURCE,
        help="Video source: '0' for primary webcam, or path to video/gif file (default: %(default)s)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.ACCIDENT_THRESHOLD,
        help="Confidence threshold (0-100) to trigger emergency alert (default: %(default)s)"
    )
    parser.add_argument(
        "--phone",
        type=str,
        default=config.ALERT_PHONE_NUMBER,
        help="Emergency recipient phone number with country code (default: %(default)s)"
    )
    parser.add_argument(
        "--no-whatsapp",
        action="store_true",
        help="Disable WhatsApp emergency message dispatch"
    )
    parser.add_argument(
        "--no-sound",
        action="store_true",
        help="Disable audio siren alarm"
    )
    return parser.parse_args()


def run_pipeline(source: str, threshold: float, phone: str):
    video_source = int(source) if source.isdigit() else source

    logger.info("Initializing Accident Detection Model...")
    model = AccidentDetectionModel(config.MODEL_JSON_PATH, config.MODEL_WEIGHTS_PATH)
    logger.info("Model loaded successfully.")

    logger.info(f"Opening video feed from: {source}")
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        logger.error(f"Could not open video source: {source}")
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    prev_time = time.time()
    accident_triggered = False

    logger.info(f"Monitoring feed. Emergency alert threshold: {threshold}%. Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream reached.")
                break

            # Calculate FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
            prev_time = curr_time

            # Preprocess frame for CNN inference (250x250 RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            roi = cv2.resize(rgb_frame, (250, 250))
            roi_batch = roi[np.newaxis, :, :]

            # Model prediction
            label, probabilities = model.predict_accident(roi_batch)
            confidence = round(float(probabilities[0][0]) * 100, 2)

            # Visual overlay configuration
            if label == "Accident":
                status_color = (0, 0, 255)  # Red (BGR)
                status_text = f"CRITICAL: Accident ({confidence:.1f}%)"
            else:
                status_color = (0, 255, 0)  # Green (BGR)
                safe_conf = round(float(probabilities[0][1]) * 100, 2)
                status_text = f"STATUS: Normal ({safe_conf:.1f}%)"

            # Draw top overlay HUD bar
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], 55), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

            # Draw status text and FPS
            cv2.putText(frame, status_text, (20, 36), font, 0.9, status_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"FPS: {int(fps)}", (frame.shape[1] - 120, 36), font, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

            # Trigger emergency response if accident threshold is met
            if label == "Accident" and confidence >= threshold and not accident_triggered:
                accident_triggered = True
                logger.warning(f"ACCIDENT CONFIRMED with {confidence}% confidence! Triggering emergency response...")

                # Display emergency alert banner across frame
                cv2.rectangle(frame, (0, frame.shape[0] // 2 - 40), (frame.shape[1], frame.shape[0] // 2 + 40), (0, 0, 200), -1)
                cv2.putText(frame, "EMERGENCY: ACCIDENT DETECTED!", (40, frame.shape[0] // 2 + 15), font, 1.1, (255, 255, 255), 3, cv2.LINE_AA)
                cv2.imshow("Accident Detection System", frame)
                cv2.waitKey(1)

                # Alarm and dispatch
                play_alarm(beeps=2)
                send_emergency_alert(phone, config.EMERGENCY_ALERT_MESSAGE)
                play_alarm(beeps=2)
                break

            # Display feed
            cv2.imshow("Accident Detection System", frame)

            # Check keyboard input (exit on 'q' or ESC)
            key = cv2.waitKey(30) & 0xFF
            if key in (ord('q'), 27):
                logger.info("Termination key received. Exiting...")
                break

    except KeyboardInterrupt:
        logger.info("Process interrupted by user. Cleaning up...")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Video feed released and windows closed.")


def main():
    args = parse_arguments()
    if args.no_whatsapp:
        config.ENABLE_WHATSAPP_ALERT = False
    if args.no_sound:
        config.ENABLE_SOUND_ALERT = False

    run_pipeline(
        source=args.source,
        threshold=args.threshold,
        phone=args.phone
    )


if __name__ == "__main__":
    main()
