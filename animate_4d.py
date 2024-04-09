import pyvista as pv
import numpy as np
import os
import json
import pyrealsense2 as rs
import cv2
from tqdm import tqdm

## SPECIFY THESE ##
DATASET_FOLDER = '../datasets/event_20240405_18_06_48_fps1_clip_1_0/'
room_pcd_path = '../datasets/CoDe_Lab-poly/point_cloud_aligned_sidecropped.ply'

# Path Configs
PATH_DEPTH  = DATASET_FOLDER + 'depth/'
PATH_COLOR = DATASET_FOLDER + 'color'
PATH_DET = DATASET_FOLDER + 'object_detection'
PATH_DET_CSV = DATASET_FOLDER + 'object_detection_csv'
PATH_OBJ3D_CSV = DATASET_FOLDER + 'object_3Dcoord_csv'
PATH_WHOLESCENE_IMG = DATASET_FOLDER + 'whole_scene_image'
os.makedirs(PATH_WHOLESCENE_IMG, exist_ok=True)

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

# Load the room point cloud 
room_pcd = pv.read(room_pcd_path)

# Create a plotter and add the point cloud
plotter = pv.Plotter(window_size=(640, 480), off_screen=True) 
plotter.add_points(room_pcd, point_size=1, scalars='RGB', rgb=True, show_scalar_bar=False, name='room')
plotter.camera_position = [(-638.3734874053291, 749.6318610092106, -4816.520935364866),
                           (-177.22672827222857, -937.5148101347535, 3190.4936848128286),
                           (0.010822758270575782, -0.9783305712435747, -0.20676595772427683)]

'''get whole scene 2.5D'''
all_color_frame = sorted([f for f in os.listdir(PATH_COLOR)])
for color_file in tqdm(all_color_frame):
    color_path = os.path.join(PATH_COLOR, color_file)
    depth_path = os.path.join(PATH_DEPTH, color_file.replace(".jpg", ".npz"))
    color = cv2.imread(color_path)
    depth = np.load(depth_path)["arr_0"]

    points = []
    colors = []
    for h in range(intrinsics.height):
        for w in range(intrinsics.width):
            if h % 2 == 0 and w % 2 == 0: # downsample
                point3d = rs.rs2_deproject_pixel_to_point(intrinsics, [w, h], depth[h, w])
                if point3d[2] > 8000: continue
                points.append(point3d)
                colors.append(color[h, w, [2, 1, 0]]) # bgr to rgb

    wholescene_pcd= pv.PolyData(np.asarray(points))
    wholescene_pcd['colors'] = np.asarray(colors).astype(np.uint8)
    plotter.add_points(wholescene_pcd, rgb=True, scalars='colors', point_size=3, name='wholescene')

    # # Show the plotter
    # plotter.show()

    # Save screenshot of current frame
    # plotter.show(auto_close=True)
    screenshot_path = os.path.join(PATH_WHOLESCENE_IMG, color_file)
    plotter.screenshot(screenshot_path)

    plotter.remove_actor('wholescene')
    
    # break


# Get the current camera position
camera_position = plotter.camera_position
print("Camera Position:", camera_position)