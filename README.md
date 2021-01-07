# Pose Prediction on webcam
Using OpenPifPaf to run multi-person pose estimation on the webcam feed
Uses opencv to capture the web camera stream and overlays the pose predictions before display
Currently displays only the skeleton


## Setup
- Create a virtual environment with python version 3.7
- Install all dependencies as given in requirements.txt

## Execution
- Via command line: `python3 pose_display.py`

### To dos:
- Add functionality to display and highlight the joints and show the confidence scores
- Increase throughput, decrease latency via multi threading