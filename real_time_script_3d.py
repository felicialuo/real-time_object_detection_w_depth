# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

# Modified by Felicia Luo to include YOLO object detection and visualize the detectd bounding boxes
# 01/30/2024

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color source
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import math
import time
import cv2
import numpy as np
import csv
import pyrealsense2 as rs

import sys
import json
from os.path import exists, join, abspath
import os
from datetime import datetime

class AppState:

    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.pitch, self.yaw = math.radians(-10), math.radians(-15)
        self.translation = np.array([0, 0, -1], dtype=np.float32)
        self.distance = 2
        self.prev_mouse = 0, 0
        self.mouse_btns = [False, False, False]
        self.paused = False
        self.scale = True
        self.color = True

    def reset(self):
        self.pitch, self.yaw, self.distance = 0, 0, 2
        self.translation[:] = 0, 0, -1

    @property
    def rotation(self):
        Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
        Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
        return np.dot(Ry, Rx).astype(np.float32)

    @property
    def pivot(self):
        return self.translation + np.array((0, 0, self.distance), dtype=np.float32)

state = AppState()

# Path Configs
OUTPUT_FOLDER = '../output/rs_recording/'
PATH_DEPTH  = OUTPUT_FOLDER + 'depth/'
PATH_COLOR = OUTPUT_FOLDER + 'color'
PATH_DET = OUTPUT_FOLDER + 'detection'
PATH_BBOX = OUTPUT_FOLDER + 'det_3d_bbox'
PATH_MESH = OUTPUT_FOLDER + 'mesh'
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(PATH_DEPTH, exist_ok=True)
os.makedirs(PATH_COLOR, exist_ok=True)
os.makedirs(PATH_DET, exist_ok=True)
os.makedirs(PATH_BBOX, exist_ok=True)
os.makedirs(PATH_MESH, exist_ok=True)

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
        print("--found rgb camera")
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    sys.exit()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 6)#30
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 6) #30

# Start streaming
pipeline.start(config)
print("start streaming")

# Get stream profile and camera intrinsics
profile = pipeline.get_active_profile()
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()
w, h = depth_intrinsics.width, depth_intrinsics.height

# Processing blocks
pc = rs.pointcloud()
colorizer = rs.colorizer()


def mouse_cb(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        state.mouse_btns[0] = True

    if event == cv2.EVENT_LBUTTONUP:
        state.mouse_btns[0] = False

    if event == cv2.EVENT_RBUTTONDOWN:
        state.mouse_btns[1] = True

    if event == cv2.EVENT_RBUTTONUP:
        state.mouse_btns[1] = False

    if event == cv2.EVENT_MBUTTONDOWN:
        state.mouse_btns[2] = True

    if event == cv2.EVENT_MBUTTONUP:
        state.mouse_btns[2] = False

    if event == cv2.EVENT_MOUSEMOVE:

        h, w = out.shape[:2]
        dx, dy = x - state.prev_mouse[0], y - state.prev_mouse[1]

        if state.mouse_btns[0]:
            state.yaw += float(dx) / w * 2
            state.pitch -= float(dy) / h * 2

        elif state.mouse_btns[1]:
            dp = np.array((dx / w, dy / h, 0), dtype=np.float32)
            state.translation -= np.dot(state.rotation, dp)

        elif state.mouse_btns[2]:
            dz = math.sqrt(dx**2 + dy**2) * math.copysign(0.01, -dy)
            state.translation[2] += dz
            state.distance -= dz

    if event == cv2.EVENT_MOUSEWHEEL:
        dz = math.copysign(0.1, flags)
        state.translation[2] += dz
        state.distance -= dz

    state.prev_mouse = (x, y)


cv2.namedWindow(state.WIN_NAME, cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow(state.WIN_NAME, w, h)
cv2.setMouseCallback(state.WIN_NAME, mouse_cb)


def project(v):
    """project 3d vector array to 2d"""
    h, w = out.shape[:2]
    view_aspect = float(h)/w

    # ignore divide by zero for invalid depth
    with np.errstate(divide='ignore', invalid='ignore'):
        proj = v[:, :-1] / v[:, -1, np.newaxis] * \
            (w*view_aspect, h) + (w/2.0, h/2.0)

    # near clipping
    znear = 0.03
    proj[v[:, 2] < znear] = np.nan
    return proj


def view(v):
    """apply view transformation on vector array"""
    return np.dot(v - state.pivot, state.rotation) + state.pivot - state.translation


def line3d(out, pt1, pt2, color=(0x80, 0x80, 0x80), thickness=1):
    """draw a 3d line from pt1 to pt2"""
    p0 = project(pt1.reshape(-1, 3))[0]
    p1 = project(pt2.reshape(-1, 3))[0]
    if np.isnan(p0).any() or np.isnan(p1).any():
        return
    p0 = tuple(p0.astype(int))
    p1 = tuple(p1.astype(int))
    rect = (0, 0, out.shape[1], out.shape[0])
    inside, p0, p1 = cv2.clipLine(rect, p0, p1)
    if inside:
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)


def grid(out, pos, rotation=np.eye(3), size=1, n=10, color=(0x80, 0x80, 0x80)):
    """draw a grid on xz plane"""
    pos = np.array(pos)
    s = size / float(n)
    s2 = 0.5 * size
    for i in range(0, n+1):
        x = -s2 + i*s
        line3d(out, view(pos + np.dot((x, 0, -s2), rotation)),
               view(pos + np.dot((x, 0, s2), rotation)), color)
    for i in range(0, n+1):
        z = -s2 + i*s
        line3d(out, view(pos + np.dot((-s2, 0, z), rotation)),
               view(pos + np.dot((s2, 0, z), rotation)), color)


def axes(out, pos, rotation=np.eye(3), size=0.075, thickness=2):
    """draw 3d axes"""
    line3d(out, pos, pos +
           np.dot((0, 0, size), rotation), (0xff, 0, 0), thickness)
    line3d(out, pos, pos +
           np.dot((0, size, 0), rotation), (0, 0xff, 0), thickness)
    line3d(out, pos, pos +
           np.dot((size, 0, 0), rotation), (0, 0, 0xff), thickness)


def frustum(out, intrinsics, color=(0x40, 0x40, 0x40)):
    """draw camera's frustum"""
    orig = view([0, 0, 0])
    w, h = intrinsics.width, intrinsics.height

    for d in range(1, 6, 2):
        def get_point(x, y):
            p = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], d)
            line3d(out, orig, view(p), color)
            return p

        top_left = get_point(0, 0)
        # print('frustum top left', top_left)
        top_right = get_point(w, 0)
        bottom_right = get_point(w, h)
        # print('w', w, 'h', h)
        # print('frustum bottom right', bottom_right)
        bottom_left = get_point(0, h)

        line3d(out, view(top_left), view(top_right), color)
        line3d(out, view(top_right), view(bottom_right), color)
        line3d(out, view(bottom_right), view(bottom_left), color)
        line3d(out, view(bottom_left), view(top_left), color)


def pointcloud(out, verts, texcoords, color, painter=True):
    """draw point cloud with optional painter's algorithm"""
    if painter:
        # Painter's algo, sort points from back to front

        # get reverse sorted indices by z (in view-space)
        # https://gist.github.com/stevenvo/e3dad127598842459b68
        v = view(verts)
        s = v[:, 2].argsort()[::-1]
        proj = project(v[s])
    else:
        proj = project(view(verts))


    h, w = out.shape[:2]

    # proj now contains 2d image coordinates
    j, i = proj.astype(np.uint32).T

    # create a mask to ignore out-of-bound indices
    im = (i >= 0) & (i < h)
    jm = (j >= 0) & (j < w)
    m = im & jm

    cw, ch = color.shape[:2][::-1]
    if painter:
        # sort texcoord with same indices as above
        # texcoords are [0..1] and relative to top-left pixel corner,
        # multiply by size and add 0.5 to center
        v, u = (texcoords[s] * (cw, ch) + 0.5).astype(np.uint32).T
    else:
        v, u = (texcoords * (cw, ch) + 0.5).astype(np.uint32).T
    # clip texcoords to image
    np.clip(u, 0, ch-1, out=u)
    np.clip(v, 0, cw-1, out=v)

    # perform uv-mapping
    out[i[m], j[m]] = color[u[m], v[m]]

def getOutputsNames(net):
    layersNames = net.getLayerNames()
    out = []
    for layer in net.getUnconnectedOutLayers():
        out.append(layersNames[layer -1])
    return out

def img2cam(points, K, depths=None):
    # project the points from image coordinates to camera coordinates
    # (1) Use the intrinsic matrix K to convert the points from image coordinates to a point cloud in the camera frame.
    # (2) Normalize the points to a plane with z=1.
    # (3) Use depths to scale the points to be at the correct distance from the camera.

    # points shape (N, 2)
    # homo image frame coord for points
    ones = np.ones((points.shape[0], 1))
    homo_img_coord = np.hstack((points, ones)) # shape (N, 3)

    # convert to 3d camera frame coord
    cam_3d = homo_img_coord @ np.linalg.inv(K.T) # shape (N, 3)

    # normalize to z=1
    cam_3d = cam_3d / cam_3d[:, [2]]

    # use depth to scale
    # print("cam_3d[:, 2]", cam_3d[:, 2].shape) # shape (1802,)
    # print("depths", depths.flatten().shape) # shape (1802,)
    if depths is not None:
      cam_3d = cam_3d * depths.reshape(depths.shape[0], 1).repeat(3, axis=1)

    return cam_3d

# draw the 3D bbox of detected object
def drawPredicted(classId, conf, left, top, right, bottom, intrinsics):
    # print('original', left, top, right, bottom)
    # left, top, right, bottom = left//2, top//2, right//2, bottom//2 # view w 320 h 240
    # print('after', left, top, right, bottom)
    cx = (left + right) // 2
    cy = (top + bottom) // 2
    dpt_frame = pipeline.wait_for_frames().get_depth_frame().as_depth_frame()

    label = '%.2f' % conf
    if classes:
        assert(classId < len(classes))
        label = '%s' %(classes[classId])

    # print('label', label)

    dist_tl = dpt_frame.get_distance(left, top)
    dist_tr = dpt_frame.get_distance(right, top)
    dist_bl = dpt_frame.get_distance(left, bottom)
    dist_br = dpt_frame.get_distance(right, bottom)
    dist_center = dpt_frame.get_distance(cx, cy)
    dist_min = min(dist_tl, dist_tr, dist_bl, dist_br)
    dist_max = max(dist_tl, dist_tr, dist_bl, dist_br)
    # print('dist', dist_tl, dist_tr, dist_bl, dist_br)

    out_bbox = np.zeros((27))
    if dist_min and dist_max: # only draw if valid distance
        # get 3d bbox vertices
        print(type(intrinsics))
        print(intrinsics)
        print(rs.rs2_deproject_pixel_to_point(intrinsics, [94, 268], 2123))
        print('width', intrinsics.width)
        print('height', intrinsics.height)
        print('ppx', intrinsics.ppx)
        print('ppy', intrinsics.ppy)
        print('fx', intrinsics.fx)
        print('fy', intrinsics.fy)
        print('model', intrinsics.model)
        print('coeffs', intrinsics.coeffs)
        tl_front = rs.rs2_deproject_pixel_to_point(intrinsics, [left, top], dist_min)
        tr_front = rs.rs2_deproject_pixel_to_point(intrinsics, [right, top], dist_min)
        bl_front = rs.rs2_deproject_pixel_to_point(intrinsics, [left, bottom], dist_min)
        br_front = rs.rs2_deproject_pixel_to_point(intrinsics, [right, bottom], dist_min)

        tl_back = rs.rs2_deproject_pixel_to_point(intrinsics, [left, top], dist_max)
        tr_back = rs.rs2_deproject_pixel_to_point(intrinsics, [right, top], dist_max)
        bl_back = rs.rs2_deproject_pixel_to_point(intrinsics, [left, bottom], dist_max)
        br_back = rs.rs2_deproject_pixel_to_point(intrinsics, [right, bottom], dist_max)

        center =  rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist_center)

        # draw 3d bbox
        line3d(out, view(tl_front), view(tr_front), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(tr_front), view(br_front), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(br_front), view(bl_front), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(bl_front), view(tl_front), color=(0x8B, 0x0, 0x0), thickness=3)

        line3d(out, view(tl_back), view(tr_back), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(tr_back), view(br_back), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(br_back), view(bl_back), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(bl_back), view(tl_back), color=(0x8B, 0x0, 0x0), thickness=3)

        line3d(out, view(tl_front), view(tl_back), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(tr_front), view(tr_back), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(bl_front), view(bl_back), color=(0x8B, 0x0, 0x0), thickness=3)
        line3d(out, view(br_front), view(br_back), color=(0x8B, 0x0, 0x0), thickness=3)
    
        out_bbox = np.array([tl_front, tr_front, bl_front, br_front, tl_back, tr_back, bl_back, br_back, center]).flatten()

    return classId, out_bbox

# gather detected object and call drawPredicted
def process_detection(frame, outs, intrinsics, out):
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
    # print('----------')
    # print('detections:', len(indices))

    labels_bbox = dict()
    for i in indices:
        # i = i[0]
        box = boxes[i]
        left = max(0, box[0])
        top = max(0, box[1])
        width = box[2]
        height = box[3]
        right = min(left+width, frameWidth-1)
        bottom = min(top+height, frameHeight-1)
        # x = int(left+width/2)
        # y = int(top+ height/2)
        # drawPredicted(classIds[i], confidences[i], x, y, intrinsics)
        classId, bbox = drawPredicted(classIds[i], confidences[i], left, top, right, bottom, intrinsics)
        labels_bbox[classId] = bbox
        # print(classId, classes[classId], bbox.shape)
    return labels_bbox


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
    # Yolo detection
    classes = None
    with open(classesFile, "rt") as f:
        classes = f.read().rstrip('\n').split('\n')
    modelConfiguration = "yolov3.cfg"
    modelWeights = "yolov3.weights"
    net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("set up yolo")

    out = np.empty((h, w, 3), dtype=np.uint8)
    frame_count = 0
    detection_npz = []
    prev_time = time.time()
    while True:
        
        now = time.time()
        time_elapsed = now - prev_time+ + 1e-8

        # print fps
        print('fps:', str(1/time_elapsed))
        prev_time = now

        # # run once every 60 seconds 
        # if time_elapsed < 60:
        #     continue

        # Grab camera data
        if not state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            # Grab new intrinsics
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            ''' ------------------------ '''
            ''' Get yolo detection '''
            color_YOLO = color_image.copy()
            depth_YOLO = depth_image.copy()
            blob = cv2.dnn.blobFromImage(color_YOLO, 1/255, (inpWidth, inpHeight), [0,0,0],1,crop=False)
            net.setInput(blob)
            detection = net.forward(getOutputsNames(net))
            # print('----yolo detection')s

            # # Record color and depth images
            # cv2.imwrite("%s/%06d.png" % \
            #         (PATH_DEPTH, frame_count), depth_YOLO)
            # cv2.imwrite("%s/%06d.jpg" % \
            #         (PATH_COLOR, frame_count), color_YOLO)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_YOLO, alpha=0.03), cv2.COLORMAP_JET)
            # process_YOLO(color_YOLO, detection)
            depth_colormap_dim = depth_colormap.shape
            color_colormap_dim = color_YOLO.shape

            # # Save the annotated image with detection
            # cv2.imwrite("%s/%06d.jpg" % \
            #         (PATH_DET, frame_count), color_YOLO)
            # print("Saved color + depth + detection %06d" % frame_count)
            # ''' ------------------------ '''

            depth_colormap = np.asanyarray(
                colorizer.colorize(depth_frame).get_data())

            if state.color:
                mapped_frame, color_source = color_frame, color_image
            else:
                mapped_frame, color_source = depth_frame, depth_colormap

            points = pc.calculate(depth_frame)
            pc.map_to(mapped_frame)

            # Pointcloud data to arrays
            v, t = points.get_vertices(), points.get_texture_coordinates()
            verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv

            # save camera intrinsic
            if frame_count == 0:
                save_intrinsic_as_json(
                    join(OUTPUT_FOLDER, "camera_intrinsic.json"),
                    color_frame)
            frame_count += 1


        # Render
        out.fill(0)

        # grid(out, (0, 0.5, 1), size=1, n=10)
        frustum(out, depth_intrinsics)
        axes(out, view([0, 0, 0]), state.rotation, size=0.1, thickness=1)

        if not state.scale or out.shape[:2] == (h, w):
            pointcloud(out, verts, texcoords, color_source)
        else:
            tmp = np.zeros((h, w, 3), dtype=np.uint8)
            pointcloud(tmp, verts, texcoords, color_source)
            tmp = cv2.resize(
                tmp, out.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
            np.putmask(out, tmp > 0, tmp)

        if any(state.mouse_btns):
            axes(out, view(state.pivot), state.rotation, thickness=4)


        # show 3d detection bbox
        labels = ''
        labels_bbox = process_detection(color_image, detection, depth_intrinsics, out)
        # all curr det in 1 string
        for key in labels_bbox.keys():
            labels = labels + classes[key] + ' '

        # # save to csv
        # csv_data = [[key] + value.tolist() for key, value in labels_bbox.items()]
        # # Specify the CSV file name
        # csv_file = "%s/%06d.csv" % (PATH_BBOX, frame_count)
        # # Writing to CSV
        # with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        #     writer = csv.writer(file)
        #     # Writing header (optional, adjust as needed)
        #     writer.writerow(['class', \
        #                      'left',' top', 'right', 'bottom', \
        #                      'dist_min',' dist_max'])
        #     writer.writerows(csv_data)

        # # Save mesh.ply
        # points.export_to_ply("%s/%06d.ply" % (PATH_MESH, frame_count), mapped_frame)

        cv2.setWindowTitle(
            state.WIN_NAME, "%s (%dx%d) %06d.jpg %s" %
            (labels, w, h, frame_count, "PAUSED" if state.paused else ""))
        
        cv2.imshow(state.WIN_NAME, out)
        key = cv2.waitKey(1)


        if key == ord("r"):
            state.reset()

        if key == ord("p"):
            state.paused ^= True

        if key == ord("z"):
            state.scale ^= True

        if key == ord("c"):
            state.color ^= True

        if key == ord("s"):
            cv2.imwrite(join(OUTPUT_FOLDER, "out.png"), out)
            print("saved screenshot in", join(OUTPUT_FOLDER, "out.png"))

        if key == ord("e"):
            points.export_to_ply(join(OUTPUT_FOLDER, "out.ply"), mapped_frame)
            print("saved mesh in", join(OUTPUT_FOLDER, "out.ply"))

        if key in (27, ord("q")) or cv2.getWindowProperty(state.WIN_NAME, cv2.WND_PROP_AUTOSIZE) < 0:
            # save last mesh ply
            points.export_to_ply(join(OUTPUT_FOLDER, "out.ply"), mapped_frame)
            print("saved mesh in", join(OUTPUT_FOLDER, "out.ply"))
            break

        

    # Stop streaming
    pipeline.stop()
