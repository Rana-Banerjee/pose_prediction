# Pose prediction on webcam
Using OpenPifPaf to run multi-person pose estimation on the webcam feed. 
Uses opencv to capture the web camera stream and overlays the pose predictions before display. 
Currently displays only the skeleton overlay.

## Project file details
- `pose_display.py` : Main code that takes webcam feed, performs pose detections and diplays it
- `requirements.txt`: Dependencies to be installed
- `images` : Sample images of demo run

## Setup
- Create a virtual environment with python version 3.7
- Activate the environment
- Install all dependencies as given in requirements.txt

## Execution
```
git clone https://github.com/Rana-Banerjee/pose_prediction.git
cd pose_prediction
python3 pose_display.py
```
Press 'q' key to exit

### To dos:
- Add functionality to display and highlight the joints and show the confidence scores
- Increase throughput, decrease latency via multi threading

### Sample demo images:
![alt text](https://github.com/Rana-Banerjee/pose_prediction/blob/main/images/1.jpg?raw=true)
![alt text](https://github.com/Rana-Banerjee/pose_prediction/blob/main/images/2.jpg?raw=true)
