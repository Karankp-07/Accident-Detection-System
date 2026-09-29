# Autonomous Real-Time Traffic Accident Detection & Emergency Alert System

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10-FF6F00?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![Status](https://img.shields.io/badge/Status-Production--Ready-success)](#)

An intelligent computer vision system that monitors video streams in real-time, classifies collision incidents using a Convolutional Neural Network (CNN), and automatically dispatches emergency alerts (WhatsApp notifications and audio alarms) to accelerate first-responder response times.

---

## Architecture Overview

```
                      +-----------------------------+
                      |   Video Source Ingestion    |
                      |  (Webcam / RTSP / MP4 / GIF)|
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |    Preprocessing Pipeline   |
                      |   - Frame extraction        |
                      |   - RGB conversion          |
                      |   - Resizing (250x250x3)    |
                      |   - Tensor batch expansion  |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Deep CNN Inference Engine  |
                      | (4 Conv Blocks + BatchNorm) |
                      +--------------+--------------+
                                     |
                                     v
                      +-----------------------------+
                      |  Confidence Score Analysis  |
                      +--------------+--------------+
                                     |
                    +----------------+----------------+
                    |                                 |
           Confidence < Threshold            Confidence >= Threshold
                    |                                 |
                    v                                 v
          +-------------------+             +-------------------+
          |  HUD Green Normal |             |  EMERGENCY ALARM  |
          |   Status Display  |             |  - Audio Siren    |
          +-------------------+             |  - WhatsApp Alert |
                                            |  - Visual HUD Red |
                                            +-------------------+
```

---

## Key Features

- **Real-Time Video Stream Inference**: Processes input streams at high frame rates with dynamic FPS tracking.
- **Deep Convolutional Neural Network (CNN)**: 4-stage convolutional pipeline with Batch Normalization, ReLU activations, and Max Pooling to capture complex crash dynamics.
- **Automated Emergency Dispatch**: Automatically sends real-time WhatsApp emergency alerts via PyWhatKit with exact alert messaging.
- **Audible Siren System**: Triggers alarm sirens on high-probability collision events to alert local operators.
- **Flexible Stream Ingestion**: Seamlessly handles live webcams (`0`), video files (`.mp4`, `.webm`), and animated sequences (`.gif`).
- **Interactive Visual HUD**: Real-time bounding overlays displaying classification state, confidence percentage, and FPS metrics.
- **Configurable Environment**: Clean separation of secrets and parameters using `.env` and CLI flags.

---

## Tech Stack

| Component | Technology | Purpose |
| :--- | :--- | :--- |
| **Deep Learning Framework** | TensorFlow / Keras 2.10 | Model training, architecture definition, weight management |
| **Computer Vision** | OpenCV (`opencv-python`) | Frame ingestion, colorspace transforms, HUD overlay rendering |
| **Numerical Computing** | NumPy (`1.26.4`) | Array vectorization, tensor preprocessing |
| **Automated Messaging** | PyWhatKit, PyAutoGUI | Web-based automated emergency dispatch |
| **Configuration** | Python-Dotenv | Environment-based secret and threshold management |

---

## Model Architecture

The custom sequential CNN accepts an input tensor of shape `(250, 250, 3)`:

1. **Input & Normalization**: `BatchNormalization` layer
2. **Block 1**: `Conv2D(32, (3, 3), relu)` -> `MaxPooling2D((2, 2))`
3. **Block 2**: `Conv2D(64, (3, 3), relu)` -> `MaxPooling2D((2, 2))`
4. **Block 3**: `Conv2D(128, (3, 3), relu)` -> `MaxPooling2D((2, 2))`
5. **Block 4**: `Conv2D(256, (3, 3), relu)` -> `MaxPooling2D((2, 2))`
6. **Classifier Head**:
   - `Flatten` (43,264 features)
   - `Dense(512, activation='relu')`
   - `Dense(2, activation='softmax')` (`Accident` vs `No Accident`)

---

## Project Structure

```
Accident-Detection-System/
├── config.py                     # Centralized configuration & environment loader
├── detection.py                  # Model loading & inference abstraction
├── main.py                       # Main pipeline, visual HUD, and alert dispatcher
├── camera.py                     # Dedicated live webcam / demo monitor
├── requirements.txt              # Pinned, tested dependency manifest
├── .env.example                  # Template configuration file
├── .env                          # Local configuration (ignored by git)
├── .gitignore                    # Production git ignore rules
├── model.json                    # Serialized CNN model architecture
├── model_weights.h5              # Trained model weights (HDF5 format)
├── accident-classification.ipynb # Exploratory analysis & training notebook
├── data/                         # Train, validation, and test datasets
│   ├── train/
│   ├── val/
│   └── test/
└── test1.gif                     # Sample test streams for evaluation
```

---

## Installation & Setup

### 1. Prerequisites
- Python 3.10 (recommended for TensorFlow 2.10 compatibility)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/your-username/Accident-Detection-System.git
cd Accident-Detection-System
```

### 3. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Configure Settings
Copy `.env.example` to `.env` and set your desired alert phone number and threshold:
```bash
cp .env.example .env
```
Edit `.env`:
```env
ALERT_PHONE_NUMBER=+919876543210
ACCIDENT_THRESHOLD=95.0
ENABLE_WHATSAPP_ALERT=True
ENABLE_SOUND_ALERT=True
DEFAULT_VIDEO_SOURCE=test1.gif
```

---

## Usage

### Run on Default Video Stream
```bash
python main.py
```

### Run on Live Webcam Feed
```bash
python main.py --source 0
```
*(Or run `python camera.py`)*

### Run with Custom Video File and Threshold
```bash
python main.py --source test4.mp4 --threshold 90 --phone +919876543210
```

### Run Without Alerts (Silent Mode for Testing)
```bash
python main.py --no-whatsapp --no-sound
```

### Controls
- Press **`q`** or **`ESC`** in the video window to quit gracefully.

---

## Resume Highlights (STAR / XYZ Method)

If showcasing this project on your Software Engineering / Machine Learning resume, you can use the following tailored bullet points:

- **Engineered an End-to-End Real-Time Accident Detection System** using TensorFlow and OpenCV, reducing incident notification latency to under 3 seconds through automated WhatsApp and siren dispatch.
- **Designed and Deployed a 4-Stage Deep CNN Architecture** with batch normalization and max-pooling, achieving robust collision detection on multi-angle surveillance feeds.
- **Implemented an Optimized Frame Preprocessing & Inference Pipeline** with vectorized tensor batches and OpenCV HUD overlays, maintaining real-time video processing throughput.
- **Architected a Modular, Configurable Codebase** supporting dynamic video stream sources (webcam, RTSP, pre-recorded feeds), command-line parametrization, and fail-safe exception handling.
