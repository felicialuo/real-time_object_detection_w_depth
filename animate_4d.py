import pyvista as pv
import numpy as np
import os
import json
import pyrealsense2 as rs
import cv2
from tqdm import tqdm
import csv
from collections import deque



# read camera intrinsics
def read_intrinsics(DATASET_FOLDER):
    json_path=os.path.join(DATASET_FOLDER, 'camera_intrinsic.json')
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
    return intrinsics

def read_colors(csv_path='unique_label2color.csv'):
    label2color = {}
    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            id, label, r, g, b = row
            label2color[label] = (int(r), int(g), int(b))
    return label2color


def animate(DATASET_FOLDER, room_pcd_path, PATH_DEPTH, PATH_COLOR, PATH_OBJ3D_CSV, PATH_WHOLESCENE_IMG, PATH_WHOLESCENE_OBJDET, ifObjDet=True):
    # Load the room point cloud 
    room_pcd = pv.read(room_pcd_path)

    # Create a plotter and add the point cloud
    plotter = pv.Plotter(window_size=(640, 480), off_screen=True) 
    plotter.add_points(room_pcd, point_size=1, scalars='RGB', rgb=True, show_scalar_bar=False, name='room')
    plotter.camera_position =  [(236.3686914087349, 768.3376358358289, -5433.626816010387),
                                (-207.1093402337325, -597.2907623405931, 3076.0675274607047),
                                (0.006662591897705766, -0.9873990694511892, -0.1581097325155246)]


    label2color = read_colors()
    past_obj = deque()
    past_obj_colors = deque()

    '''get whole scene 2.5D'''
    all_color_frame = sorted([f for f in os.listdir(PATH_COLOR)])
    intrinsics = read_intrinsics(DATASET_FOLDER)
    for color_file in tqdm(all_color_frame):
        ###### the whole scene
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

        wholescene_pcd = pv.PolyData(np.asarray(points))
        wholescene_pcd['colors'] = np.asarray(colors).astype(np.uint8)
        plotter.add_points(wholescene_pcd, rgb=True, scalars='colors', point_size=3, name='wholescene', render_points_as_spheres=True)

        
        if ifObjDet:
            objdet_path = os.path.join(PATH_OBJ3D_CSV, color_file.replace(".jpg", ".csv"))
            with open(objdet_path,'r' ) as det_file:
                reader = csv.reader(det_file)
                next(reader)
                
                obj_points = []
                obj_colors = []
                for row in reader:
                    label, confidence, x, y, z = row
                    obj_points.append([(x), y, z])
                    obj_colors.append(label2color[label])
            
            ###### + past object detection locs
            past_obj.append(np.asarray(obj_points).astype(np.float64))
            past_obj_colors.append(np.asarray(obj_colors).astype(np.uint8))
            if len(past_obj) > 300: 
                past_obj.popleft()
                past_obj_colors.popleft()
            past_pcd = pv.PolyData(np.vstack(past_obj))
            past_pcd['colors'] = np.vstack(past_obj_colors)
            plotter.add_points(past_pcd, rgb=True, scalars='colors', point_size=5, name='past_obj', render_points_as_spheres=True)
            
            ###### + curr object detection locs
            objdet_pcd = pv.PolyData(np.asarray(obj_points).astype(np.float64))
            objdet_pcd['colors'] = np.asarray(obj_colors).astype(np.uint8)
            plotter.add_points(objdet_pcd, rgb=True, scalars='colors', point_size=20, name='objdet', render_points_as_spheres=True)


        # Show the plotter
        # plotter.show(auto_close=False) # plotter = pv.Plotter(window_size=(640, 480), off_screen=True) 
        # break


        # Save screenshot of current frame
        # just scene
        screenshot_path = os.path.join(PATH_WHOLESCENE_IMG, color_file)
        # with object detection loc
        if ifObjDet: screenshot_path = os.path.join(PATH_WHOLESCENE_OBJDET, color_file)
        plotter.screenshot(screenshot_path)

        plotter.remove_actor('wholescene')
        if ifObjDet: 
            plotter.remove_actor('objdet')
            plotter.remove_actor('past_obj')
        
        
    # # Get the current camera position
    camera_position = plotter.camera_position
    print("Camera Position:", camera_position)


if __name__ == "__main__":
    ## SPECIFY THESE ##
    DATASET_FOLDER = '../datasets/event_20240405_18_06_48_fps1_clip_1_0/'
    room_pcd_path = '../datasets/CoDe_Lab-poly/point_cloud_aligned_outside_cropped.ply'

    # Path Configs
    PATH_DEPTH  = DATASET_FOLDER + 'depth/'
    PATH_COLOR = DATASET_FOLDER + 'color'
    PATH_DET = DATASET_FOLDER + 'object_detection'
    PATH_DET_CSV = DATASET_FOLDER + 'object_detection_csv'
    PATH_OBJ3D_CSV = DATASET_FOLDER + 'object_3Dcoord_csv'
    PATH_WHOLESCENE_IMG = DATASET_FOLDER + 'whole_scene_image'
    PATH_WHOLESCENE_OBJDET = DATASET_FOLDER + 'whole_scene_objdet'
    os.makedirs(PATH_WHOLESCENE_IMG, exist_ok=True)
    os.makedirs(PATH_WHOLESCENE_OBJDET, exist_ok=True)

    animate(DATASET_FOLDER, room_pcd_path, PATH_DEPTH, PATH_COLOR, PATH_OBJ3D_CSV, PATH_WHOLESCENE_IMG, 
            PATH_WHOLESCENE_OBJDET, ifObjDet=False)