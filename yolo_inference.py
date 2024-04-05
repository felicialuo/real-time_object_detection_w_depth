"""
YOLOv3 Inference
Creator: Felicia Luo
Date: 03/21/2024
"""

import numpy as np
import cv2
from tqdm import tqdm
import os
from datetime import datetime
import csv

## SPECIFY THESE ##
DATASET_FOLDER = '../datasets/20240404_18_33_38_fps1_clip_1_0/'

# Path Configs
PATH_DEPTH  = DATASET_FOLDER + 'depth/'
PATH_COLOR = DATASET_FOLDER + 'color'
PATH_DET = DATASET_FOLDER + 'object_detection'
PATH_DET_CSV = DATASET_FOLDER + 'object_detection_csv'
os.makedirs(PATH_DET, exist_ok=True)
os.makedirs(PATH_DET_CSV, exist_ok=True)

# Initialize the parameters
confThreshold = 0.5
nmsThreshold = 0.4
inpWidth = 416
inpHeight = 416
classesFile = "coco.names"

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    out = []
    for layer in net.getUnconnectedOutLayers():
        out.append(layersNames[layer -1])
    return out

def drawPredicted(classId, conf, left, top, right, bottom, color_frame, dpt_frame, cx, cy):
    # print(classId, conf, x, y)
    cv2.rectangle(color_frame, (left,top), (right,bottom), (255,178,50),1)
    distance = dpt_frame[cy][cx] * 0.001
    cv2.circle(color_frame,(cx, cy),radius=1,color=(0,0,254), thickness=5)
    label = '%.2f' % conf

    if classes:
        assert(classId < len(classes))
        label = '%s' %(classes[classId])
    
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.putText(color_frame, label,(left,top-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
    distance_string = "Dist: " + str(distance) + " meter away"
    cv2.putText(color_frame,distance_string,(left,top+30), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255,255,0), 1)

def process_detection(color_frame, depth_frame, outs, filename):
    frameHeight = color_frame.shape[0]
    frameWidth = color_frame.shape[1]
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

    # only save detection if person is detected
    # if 0 in indices:
    #     print("Person detected")
        # person_i = np.argwhere(classIds==0)
        # person_i = 0
        # person_x = int(boxes[person_i][0] + boxes[person_i][2] / 2)
        # person_y = int(boxes[person_i][1] + boxes[person_i][3] / 2)
    
    # save the results to csv
    csv_path = os.path.join(PATH_DET_CSV, filename)[:-3] + 'csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # Write the header row
        writer.writerow(['Object Class Label', 'confidence', 'left', 'top', 'right', 'bottom', 'center_dist'])

        for i in indices:
            # i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            right = min(left+width, frameWidth-1)
            bottom = min(top+height, frameHeight-1)
            cx = int(left+width/2)
            cy = int(top+ height/2)
            # print('Dist', depth_frame[cy][cx] * 0.001)

            # # only save objects in short distance from person
            # if np.linalg.norm(np.array([x, y]) - np.array([person_x, person_y])) <= 300:
            drawPredicted(classIds[i], confidences[i],  left, top, left+width, top+height, color_frame, depth_frame, cx, cy)

            # Write the class labels, confidence, bbox to the CSV file
            writer.writerow([classes[classIds[i]], confidences[i], left, top, right, bottom, depth_frame[cy][cx] * 0.001])



if __name__ == "__main__":
    classes = None
    with open(classesFile, "rt") as f:
        classes = f.read().rstrip('\n').split('\n')
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    for filename in tqdm(os.listdir(PATH_COLOR)):
        dt0 = datetime.now()

        # read image sequence
        color_image = cv2.imread(os.path.join(PATH_COLOR, filename))
        depth_image = np.load(os.path.join(PATH_DEPTH, filename)[:-3] + 'npz')['arr_0'] # shape (480, 640) unit mm
        # depth_image = np.zeros(color_image.shape) ## placeholder, should use above

        # Get YOLO detections
        blob = cv2.dnn.blobFromImage(color_image, 1/255, (inpWidth, inpHeight), [0,0,0],1,crop=False)
        net.setInput(blob)
        detection = net.forward(getOutputsNames(net))

        # Save detections to csv
        process_detection(color_image, depth_image, detection, filename)

        # Save the annotated image with detection
        cv2.imwrite(os.path.join(PATH_DET, filename), color_image)