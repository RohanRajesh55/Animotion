# Facial Tracker for Avatar Animation

A real-time facial tracking system designed to capture facial expressions and head movements, enabling the animation of a virtual avatar. This project utilizes OpenCV and MediaPipe to detect facial landmarks and expressions, sending the data via WebSocket to animate avatars in applications like VTube Studio.

![Facial Tracker Demo](assets/demo.gif)

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Features

- **Real-Time Facial Expression Detection**: Captures eye blinks, mouth movements, eyebrow raises, and lip synchronization.
- **Head Pose Estimation**: Calculates yaw, pitch, and roll to track head movements.
- **WebSocket Integration**: Transmits facial data to avatar applications for real-time animation.
- **Configurable Parameters**: Adjustable thresholds and settings via a `config.ini` file.
- **Logging and Error Handling**: Comprehensive logging and exception management for robust performance.
- **Modular Design**: Clean code structure with modular components for easy extension and maintenance.

## Prerequisites

- **Python 3.7 or higher**
- **Operating System**: Windows, macOS, or Linux
- **Webcam**: Integrated or external webcam, or an IP camera feed
- **Virtual Environment**: Recommended to avoid package conflicts

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/facial_tracker.git
   cd facial_tracker
   ```

2. **Create a Virtual Environment**

   ```bash
   python -m venv venv
   ```

3. **Activate the Virtual Environment**

   - **Windows**

     ```bash
     venv\Scripts\activate
     ```

   - **macOS/Linux**

     ```bash
     source venv/bin/activate
     ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   **Note**: Ensure that you have the required packages listed in `requirements.txt`, such as:

   - `opencv-python`
   - `mediapipe`
   - `numpy`
   - `websockets`

## Configuration

Edit the `config.ini` file to adjust settings according to your setup.

```ini
# config.ini

[Camera]
# Use 0 for default webcam or provide the URL for an external camera feed
DROIDCAM_URL = 0

[Thresholds]
EAR_THRESHOLD = 0.22
MAR_THRESHOLD = 0.5
EBR_THRESHOLD = 1.5

[WebSocket]
VTS_WS_URL = ws://localhost:8001
AUTH_TOKEN = your_auth_token

[Logging]
# Set logging level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = INFO
```

- **Camera Settings**
  - `DROIDCAM_URL`: Set to `0` for the default webcam or provide the IP camera URL.
- **Thresholds**
  - Adjust the thresholds (`EAR_THRESHOLD`, `MAR_THRESHOLD`, `EBR_THRESHOLD`) based on calibration for accurate detection.
- **WebSocket**
  - `VTS_WS_URL`: WebSocket URL of your avatar application (e.g., VTube Studio).
  - `AUTH_TOKEN`: Authentication token for the avatar application API.
- **Logging**
  - `LOG_LEVEL`: Set the desired logging level for the application.

## Usage

1. **Run the Application**

   ```bash
   python main.py
   ```

2. **Interact with the Facial Tracker**

   - **Expressions to Try**:
     - Blink your eyes to test eye blink detection.
     - Open your mouth to test mouth open detection.
     - Raise your eyebrows to test eyebrow detection.
     - Speak or move your lips to test lip synchronization.
     - Move your head around to test head pose estimation.

3. **Quit the Application**

   - Press the `q` key in the application window to exit.

## Customization

### Adjusting Thresholds

- **Calibration**: Adjust the thresholds in `config.ini` to calibrate the sensitivity of the detectors.
  - **Eye Aspect Ratio (EAR)**: Lower values make blink detection more sensitive.
  - **Mouth Aspect Ratio (MAR)**: Adjust to detect mouth opening accurately.
  - **Eyebrow Raise Ratio (EBR)**: Modify to capture eyebrow movements effectively.

### Modifying Landmarks

- **Landmark Indices**: If you wish to modify which facial landmarks are used, refer to the [MediaPipe Face Mesh landmark map](https://google.github.io/mediapipe/solutions/face_mesh.html) and update the indices in `main.py` accordingly.

### Extending Functionality

- **Additional Detectors**: Implement new detectors by creating modules in the `detectors/` directory.
- **Data Mapping**: Customize how the calculated metrics map to your avatar application's parameters in `websocket_client.py`.

## Project Structure

```
facial_tracker/
├── main.py
├── config.ini
├── requirements.txt
├── websocket_client.py
├── detectors/
│   ├── eye_detector.py
│   ├── mouth_detector.py
│   ├── eyebrow_detector.py
│   ├── lip_sync.py
│   └── head_pose_estimator.py
├── utils/
│   ├── calculations.py
│   └── shared_variables.py
├── assets/
│   └── demo.gif
└── README.md
```

- **main.py**: Main application script for processing video frames and detecting expressions.
- **config.ini**: Configuration file for adjustable settings.
- **requirements.txt**: List of required Python packages.
- **websocket_client.py**: Handles communication with the avatar application via WebSocket.
- **detectors/**: Contains modules for each facial feature detector.
- **utils/**: Utility modules for calculations and shared variables.
- **assets/**: Contains media assets like images or GIFs for the README.

## Contributing

Contributions are welcome! Please follow these steps:

1. **Fork the Repository**

   Click the "Fork" button at the top right of this page to create a copy of this repository to your GitHub account.

2. **Clone Your Fork**

   ```bash
   git clone https://github.com/yourusername/facial_tracker.git
   ```

3. **Create a Feature Branch**

   ```bash
   git checkout -b feature/your-feature-name
   ```

4. **Commit Your Changes**

   ```bash
   git commit -am 'Add a new feature'
   ```

5. **Push to Your Fork**

   ```bash
   git push origin feature/your-feature-name
   ```

6. **Open a Pull Request**

   Create a pull request from your forked repository's feature branch to the main repository's `main` branch.

## License

## This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- **MediaPipe**: For providing powerful, real-time face mesh models.
- **OpenCV**: For comprehensive computer vision tools.
- **VTube Studio**: For enabling live 2D avatar animation.
- **Community**: Thanks to all contributors and users who have supported this project.

## Contact

_Note: Replace `yourusername`, `your.email@example.com`, and other placeholder information with your actual details._
