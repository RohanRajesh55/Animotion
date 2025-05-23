# Animotion - Real-Time Facial Tracker for Avatar Animation

Animotion is a **real-time facial tracking system** designed to capture facial expressions and head movements to drive avatar animation. Built with **OpenCV** and **MediaPipe**, Animotion detects facial landmarks and transmits data via **WebSocket** to external applications like **VTube Studio**. Features also include basic lip sync detection using in-house modules, configurable parameters, and a modular design for easy expansion.

---

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Customization](#customization)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)


## Features

- **Real-Time Facial Expression Detection** – Capture eye blinks, mouth movements, and eyebrow raises.
- **Head Pose Estimation** – Track yaw, pitch, and roll of the head.
- **WebSocket Integration** – Send real-time facial tracking data to applications (e.g., VTube Studio).
- **Configurable Parameters** – Easily adjust system settings via the `config.ini` file.
- **Logging and Error Handling** – Comprehensive logging for robust operation.
- **Modular Design** – Well-organized code for simple maintenance and future additions.


## Prerequisites

- **OS:** Windows, macOS, or Linux
- **Python:** 3.7+
- **Webcam:** Built-in or external
- **Virtual Environment:** Recommended to avoid package conflicts

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/RohanRajesh55/Animotion.git
cd Animotion
```

````

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

### 3. Activate the Virtual Environment

**On Windows:**

```bash
venv\Scripts\activate
```

**On macOS/Linux:**

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

_Note: Ensure that packages like `opencv-python`, `mediapipe`, `numpy`, and `websockets` are installed properly._


## Configuration

Customize the settings in `config.ini` to fine-tune the behavior of Animotion. For example:

```ini
[Camera]
DROIDCAM_URL = 0  # Use 0 for the default webcam

[Thresholds]
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.5
EBR_THRESHOLD = 1.5
EMOTION_THRESHOLD = 0.4

[WebSocket]
VTS_WS_URL = ws://localhost:8001
AUTH_TOKEN = your_auth_token

[Logging]
LOG_LEVEL = INFO  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL

[Animation]
SMOOTHING_FACTOR = 0.8
INTERPOLATION_TYPE = linear

[Advanced]
LIP_SYNC_SKIP_FRAMES = 3
```


## Usage

### Run the Application

```bash
python main.py
```

### Interacting with Animotion

- **Blink Detection:** Blink your eyes.
- **Mouth Open Detection:** Open your mouth.
- **Head Pose Estimation:** Rotate your head to test yaw, pitch, and roll.
- **Lip Sync Testing:** Speak or move your lips (with microphone, if enabled).
- **Quit:** Press **`q`** in the application window to exit.


## Customization

- **Adjust Thresholds:** Modify settings like `EAR_THRESHOLD`, `MAR_THRESHOLD`, and others in `config.ini` to optimize detection accuracy.
- **Extend Functionality:** Add or update modules in the `detectors/` directory to customize facial feature tracking or modify the WebSocket logic in `websocket_client.py` for integration with other platforms.


## Project Structure

```
Animotion/
├── .gitignore                         # Git ignore file for excluding build artifacts, virtual envs, etc.
├── README.md                          # Main project documentation: overview, installation, usage, etc.
├── project_root.txt                   # (Optional) Human-readable summary of the project structure
├── requirements.txt                   # List of pinned dependencies for reproducible environments
├── config.ini                         # Configuration file storing user settings (e.g., thresholds, API keys)
├── calibration.py                     # Tool for camera calibration and computing EAR/MAR thresholds
├── emotion_recognition.py             # DeepFace‑based emotion recognition module (with GPU support if available)
├── emotion_integration.py             # Module for integrating emotion data into the processing pipeline
├── performance_monitor.py             # Monitors runtime performance (FPS, resource usage, etc.)
├── main.py                            # Main application: orchestrates video processing and landmark detection
├── websocket_client.py                # Module for sending facial tracking data via WebSocket to an external API
├── detectors/                         # Package for facial feature detection modules
│   ├── __init__.py                    # Marks the detectors folder as a package
│   ├── eye_detector.py                # Computes the Eye Aspect Ratio (EAR) for blink detection
│   ├── eyebrow_detector.py            # Computes the Eyebrow Raise Ratio (EBR) for eyebrow movement detection
│   ├── head_pose_estimator.py         # Estimates head pose (yaw, pitch, roll) via methods like solvePnP
│   ├── lip_sync.py                   # Calculates lip-sync values based on mouth landmark analysis
│   ├── mouth_detector.py              # Computes the Mouth Aspect Ratio (MAR) for evaluating mouth openness
│   └── smile_detector.py              # Detects smiling by analyzing facial landmarks and cues
├── filter/                            # Package for filtering and smoothing signals
│   ├── __init__.py                    # Marks the filter folder as a package
│   └── kalman_filter.py               # Implements a 1D Kalman Filter for smoothing noisy signals
└── utils/                             # Utility modules and shared resources
    ├── __init__.py                    # Marks the utils folder as a package
    ├── calculations.py                # Provides helper functions (e.g., distance calculations, FPS estimation)
    ├── config_manager.py              # Manages configuration files (parsing, updating config.ini values)
    ├── shared_variables.py            # Container for inter-module data exchange (shared variables)
    └── vtube_mapper.py                # Handles mapping of virtual tube (or avatar) settings as needed
```


## Contributing

We welcome contributions! To contribute:

1. **Fork the Repository**

   [Fork this Repo](https://github.com/RohanRajesh55/Animotion)

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/your-username/Animotion.git
   cd Animotion
   ```

3. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Commit Your Changes**

   ```bash
   git commit -am "Describe your changes"
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**
   Submit a pull request from your feature branch to the main repository.


## License

This project is open source and available under the **MIT License**.


## Acknowledgments

- **MediaPipe** – For real-time face mesh models.
- **OpenCV** – For computer vision and image processing.
- **VTube Studio** – For inspiring live avatar animation.
- Thanks to the community and contributors for their support!


## Contact

For questions or support, please open an issue on the [GitHub Repository](https://github.com/RohanRajesh55/Animotion).

