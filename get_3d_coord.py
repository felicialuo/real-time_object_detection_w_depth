import pyrealsense2 as rs
import json
import numpy as np
import matplotlib.pyplot as plt
import csv
import cv2
import os
from tqdm import tqdm

## SPECIFY THESE ##
DATASET_FOLDER = '../datasets/event_20240405_18_06_48_fps1_clip_1_0/'

# Path Configs
PATH_DEPTH  = DATASET_FOLDER + 'depth/'
PATH_COLOR = DATASET_FOLDER + 'color'
PATH_DET = DATASET_FOLDER + 'object_detection'
PATH_DET_CSV = DATASET_FOLDER + 'object_detection_csv'
PATH_OBJ3D_CSV = DATASET_FOLDER + 'object_3Dcoord_csv'
os.makedirs(PATH_OBJ3D_CSV, exist_ok=True)

# read camera intrinsics
json_path = os.path.join(DATASET_FOLDER, 'camera_intrinsic.json')
with open(json_path, 'r') as file:
    data = json.load(file)
intrinsics = rs.pyrealsense2.intrinsics()
intrinsics.width = data['width']
intrinsics.height = data['height']
intrinsics.ppx = data['intrinsic_matrix'][6]
intrinsics.ppy = data['intrinsic_matrix'][7]
intrinsics.fx = data['intrinsic_matrix'][0]
intrinsics.fy = data['intrinsic_matrix'][4]
intrinsics.model = rs.pyrealsense2.distortion.brown_conrady
intrinsics.coeffs = [0.0, 0.0, 0.0, 0.0, 0.0]


all_det_csv = sorted([f for f in os.listdir(PATH_DET_CSV)])
for det_csv in tqdm(all_det_csv):
    OBJ3D_csv_path = os.path.join(PATH_OBJ3D_CSV, det_csv)
    det_csv_path = os.path.join(PATH_DET_CSV, det_csv)
    with open(OBJ3D_csv_path, 'w', newline='') as obj3d_file:
        with open(det_csv_path,'r' ) as det_file:
            reader = csv.reader(det_file)
            next(reader)
            writer = csv.writer(obj3d_file)
            writer.writerow(["Object Class Label","confidence", "x", "y", "z"])

            for row in reader:
                label, confidence, left, top, right, bottom, center_dist = row
                left, top, right, bottom, center_dist = int(left), int(top), int(right), int(bottom), float(center_dist) * 1000
                x, y, z =  rs.rs2_deproject_pixel_to_point(intrinsics, [(left+right)/2, (top+bottom)/2], center_dist)
                writer.writerow([label, confidence, x, y, z])




# '''get whole scene 2.5D'''
# # read depth npz
# color = cv2.imread("../datasets/event_20240405_18_06_48_fps1_clip_1_0/color/18_07_00_009.jpg")
# depth = np.load("../datasets/event_20240405_18_06_48_fps1_clip_1_0/depth/18_07_00_009.npz")["arr_0"]
# print(color.shape) # (480, 640, 3)
# print(depth.shape) # (480, 640)

# # save the results to csv
# csv_path = "../datasets/event_20240405_18_06_48_fps1_clip_1_0/test.csv"
# with open(csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)

#     points = []
#     for h in range(intrinsics.height):
#         for w in range(intrinsics.width):
#             point3d = rs.rs2_deproject_pixel_to_point(intrinsics, [w, h], depth[h, w])
#             if point3d[2] > 8000: continue
#             points.append(point3d)
#             rgb = color[h, w, :]
            
#             writer.writerow(point3d + list(rgb))
#             # print(point3d)

# def viz_pts_3d(pts,xrange=None,yrange=None,zrange=None,title=None):
#     # viz the 3D points
#     fig = plt.figure(figsize=(10,10))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=1)
#     ax.set_xlabel('X [m]')
#     ax.set_ylabel('Y [m]')
#     ax.set_zlabel('Z [m]')

#     if xrange is not None:
#         ax.set_xlim(xrange)
#     if yrange is not None:
#         ax.set_ylim(yrange)
#     if zrange is not None:
#         ax.set_zlim(zrange)

#     if title is not None:
#         ax.set_title(title)
#     plt.show()



# csv_path = "../datasets/event_20240405_18_06_48_fps1_clip_1_0/test_person.csv"
# with open(csv_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     person_point = rs.rs2_deproject_pixel_to_point(intrinsics, [94, 268], 2123)
#     writer.writerow(person_point)
