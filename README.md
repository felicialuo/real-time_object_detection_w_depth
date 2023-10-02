# Real-time Object Dectection with Depth
Real-time object detection and visualization using YOLOv3 and extract depth information via Intel RealSense D435i.

Original code: [dev_realsense_yolo_v3 by Tony](https://github.com/dovanhuong/dev_realsense_yolo_v3_2d#dev_realsense_yolo_v3-by-tony)


## Installation instructions
```
conda create --name realsense
conda activate realsense
conda install pip
pip install pyrealsense2
pip install opencv-python
```

## Run the script
- Download [weight file of yolo]( https://pjreddie.com/media/files/yolov3.weights)
- Run the script from terminal `python script.py`. Press `Ctrl+C` or `q` to quit.
- Visualization example:
<p float="left">
   <img src="preview.PNG" width="100%">
</p>

## Troubleshoot
- wait_for_frames(): "RuntimeError: Frame didn't arrive within 5000"
  - Disable auto exposure from your Intel RealSense Viewer. You can re-enable it later if the issue does not persist.
  - or skip the first several frames as suggested [here](https://github.com/IntelRealSense/librealsense/issues/9417#issuecomment-880762163).