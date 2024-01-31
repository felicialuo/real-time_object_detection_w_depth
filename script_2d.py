"""
Real-time Inference yolov3 with Intel Realsense D435i camera
Creator: Felicia Luo
Date: 12/09/2023
Based on Tony Do (vanhuong.robotics@gmail.com)'s Inference yolov3 in Realsense D435 camera

Added save recorded color, depth, and annotated detection images
"""
import pyrealsense2 as rs
import numpy as np
import cv2
import sys
import json
from os.path import exists, join, abspath
import os
from datetime import datetime

# Path Configs
OUTPUT_FOLDER = '../output/rs_recording/'
PATH_DEPTH  = OUTPUT_FOLDER + 'depth/'
PATH_COLOR = OUTPUT_FOLDER + 'color'
PATH_DET = OUTPUT_FOLDER + 'detection'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PATH_DEPTH, exist_ok=True)
os.makedirs(PATH_COLOR, exist_ok=True)
os.makedirs(PATH_DET, exist_ok=True)

# Initialize the parameters
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416
classesFile = "coco.names"

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))
depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()
found_rgb = False

for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        print("36. found rgb camera")
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    sys.exit()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
    print("46. config 960*540 color")
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    print("49. config 640*480 color")
# Start streaming
pipeline.start(config)
print("52. start streaming")

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    out = []
    for layer in net.getUnconnectedOutLayers():
        out.append(layersNames[layer -1])
    return out

def drawPredicted(classId, conf, left, top, right, bottom, frame,x ,y):
    # print(classId, conf, x, y)
    cv2.rectangle(frame, (left,top), (right,bottom), (255,178,50),3)
    dpt_frame = pipeline.wait_for_frames().get_depth_frame().as_depth_frame()
    distance = dpt_frame.get_distance(x,y)
    cv2.circle(frame,(x,y),radius=1,color=(0,0,254), thickness=5)
    label = '%.2f' % conf

    if classes:
        assert(classId < len(classes))
        label = '%s' %(classes[classId])
    print('label', label)
    
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(frame, label,(left,top-5), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,0),2)
    distance_string = "Dist: " + str(round(distance,2)) + " meter away"
    cv2.putText(frame,distance_string,(left,top+30), cv2.FONT_HERSHEY_SIMPLEX,0.75,(255,255,0),2)

def process_detection(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0]*frameWidth)
                center_y = int(detection[1]*frameHeight)
                width = int(detection[2]*frameWidth)
                height = int(detection[3]*frameHeight)
                left = int(center_x - width/2)
                top = int(center_y - height/2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left,top,width,height])
    indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    print('----------')
    print('detections:', len(indices))

    # only save detection if person is detected
    # if 0 in indices:
    #     print("Person detected")
        # person_i = np.argwhere(classIds==0)
        # person_i = 0
        # person_x = int(boxes[person_i][0] + boxes[person_i][2] / 2)
        # person_y = int(boxes[person_i][1] + boxes[person_i][3] / 2)
    for i in indices:
        # i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        right = min(left+width, frameWidth-1)
        bottom = min(top+height, frameHeight-1)
        x = int(left+width/2)
        y = int(top+ height/2)
        # # only save objects in short distance from person
        # if np.linalg.norm(np.array([x, y]) - np.array([person_x, person_y])) <= 300:
        drawPredicted(classIds[i], confidences[i],  left, top, left+width, top+height, frame, x, y)


def save_intrinsic_as_json(filename, frame):
    intrinsics = frame.profile.as_video_stream_profile().intrinsics
    with open(filename, 'w') as outfile:
        obj = json.dump(
            {
                'width':
                    intrinsics.width,
                'height':
                    intrinsics.height,
                'intrinsic_matrix': [
                    intrinsics.fx, 0, 0, 0, intrinsics.fy, 0, intrinsics.ppx,
                    intrinsics.ppy, 1
                ]
            },
            outfile,
            indent=4)

if __name__ == "__main__":
    classes = None
    with open(classesFile, "rt") as f:
        classes = f.read().rstrip('\n').split('\n')
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    # print("118. net.getLayerNames()", net.getLayerNames())

    # Streaming loop
    frame_count = 0
    try:
        while True:
            dt0 = datetime.now()
            
            # Wait for a coherent pair of frames: depth and color
            # print("122. waiting for frames")
            frames = pipeline.wait_for_frames()
            # print("124. got frames")
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            blob = cv2.dnn.blobFromImage(color_image, 1/255, (inpWidth, inpHeight), [0,0,0],1,crop=False)
            net.setInput(blob)
            detection = net.forward(getOutputsNames(net))

            # Record color and depth images
            if frame_count == 0:
                save_intrinsic_as_json(
                    join(OUTPUT_FOLDER, "camera_intrinsic.json"),
                    color_frame)
            cv2.imwrite("%s/%06d.png" % \
                    (PATH_DEPTH, frame_count), depth_image)
            cv2.imwrite("%s/%06d.jpg" % \
                    (PATH_COLOR, frame_count), color_image)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            process_detection(color_image, detection)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_image.shape

            # Save the annotated image with detection
            cv2.imwrite("%s/%06d.jpg" % \
                    (PATH_DET, frame_count), color_image)
            print("Saved color + depth + detection %06d" % frame_count)
            frame_count += 1

            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))

            # Show images
            cv2.imshow('Real-time YOLO with Depth', images)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # Print fps
            process_time = datetime.now() - dt0
            print("FPS: "+str(1/process_time.total_seconds()))

    finally:
        # Stop streaming
        pipeline.stop()
