import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Model configuration paths
MODEL_JSON_PATH = str(BASE_DIR / "model.json")
MODEL_WEIGHTS_PATH = str(BASE_DIR / "model_weights.h5")

# Alert configuration
ALERT_PHONE_NUMBER = os.getenv("ALERT_PHONE_NUMBER", "+918709368334")
ACCIDENT_THRESHOLD = float(os.getenv("ACCIDENT_THRESHOLD", 95.0))
ENABLE_WHATSAPP_ALERT = os.getenv("ENABLE_WHATSAPP_ALERT", "True").lower() in ("true", "1", "yes")
ENABLE_SOUND_ALERT = os.getenv("ENABLE_SOUND_ALERT", "True").lower() in ("true", "1", "yes")

# Default Video Source ('0' for primary webcam, or file path)
DEFAULT_VIDEO_SOURCE = os.getenv("DEFAULT_VIDEO_SOURCE", "test1.gif")

# Emergency message payload
EMERGENCY_ALERT_MESSAGE = "EMERGENCY: Traffic accident detected! Immediate medical and roadside assistance required."
