# Tracking with YoloV5 & DeepSORT
## Introduction
[DeepSORT](https://arxiv.org/abs/1703.07402) basically is an improvement based on [SORT](https://arxiv.org/abs/1602.00763) which integrated a CNN feature extractor that helps reduce ID-switch problem in SORT. This CNN model acts as a ReID model which extracts features for each object that has been detected by an object detection model, then those features are used for assignment problems. This indeed helps reduce ID-switch problems a lot!
 
For each object detected in the previous frame, Kalman Filter will predict its new position in the adjacent frame. Then for each object that has been detected by the Object Detection Model, it will be assigned to its prediction by the similarity between its CNN feature.
 
For occlusion objects, DeepSORT counts the number of frames it disappears. If it is less than 30 frames, the trajectory is still keeping, otherwise delete that trajectory.
 
This repository contains an implementation of DeepSORT. Using [Yolov5](https://github.com/ultralytics/yolov5) as Object Detector and OpenCV to process video. The input is a video or webcam by default, the output is video which has the detected objects and its relative ID.

## Usage
1. Clone this repository
```
git clone https://github.com/tien02/TrackingwYolov5DeepSORT.git
```
2. Create virtual environment (venv), then activate it.
```
python -m venv your venv
```
3. Install dependencies
```
pip install -r requirements.txt
```
4. Run demo, run --help for more details
```
python tracking.py [-h]
                    [--config]  
                    [--weight]  
                    [--source]  
                    [--display]
                    [--save]
```

## References 
- Paper: [Simple Online and Realtime Tracking with a Deep Association Metric](https://arxiv.org/abs/1703.07402)
- DeepSORT code: [ZQPei/deep_sort_pytorch](https://github.com/ZQPei/deep_sort_pytorch) 
- Yolov5 code: [ultralytics/yolov5](https://github.com/ultralytics/yolov5)


Approaching Pedestrian Tracking problem on surveillance camera with YoloV5 for pedestrian detection and DeepSORT for tracking.
