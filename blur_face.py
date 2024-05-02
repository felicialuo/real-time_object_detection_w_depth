import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm import tqdm 



source_dir = "../datasets/event_20240405_18_06_48_fps1_clip_1_0/color_original"
target_dir = "../datasets/event_20240405_18_06_48_fps1_clip_1_0/color"
os.makedirs(target_dir, exist_ok=True)

color_files = sorted([f for f in os.listdir(source_dir)])

# https://stackoverflow.com/questions/18064914/how-to-use-opencv-and-haar-cascades-to-blur-faces

# Load the pre-trained model
model_file = "deploy.prototxt"
model_weights = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
net = cv2.dnn.readNetFromCaffe(model_file, model_weights)

from typing import Tuple, Union
import math
import cv2
import numpy as np

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red



for imgname in tqdm(color_files):

    file_path = os.path.join(source_dir, imgname)
    img = cv2.imread(file_path)
    h, w = img.shape[:2]

    # Prepare the image for the neural network
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)

    # Perform face detection
    detections = net.forward()

    # Apply a blur to each face detected
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.25:  # Confidence threshold
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x2, y2) = box.astype("int")
            # cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
            face_roi = img[y:y2, x:x2]
            blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
            img[y:y2, x:x2] = blurred_face

    # Save the output image
    output_path = os.path.join(target_dir, imgname)
    cv2.imwrite(output_path, img)

    

