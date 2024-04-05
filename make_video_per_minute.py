import cv2
import os

def make_videos_from_frames(frame_folder, output_folder, frame_rate=30, num_frames=60):
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.endswith('.jpg') or f.endswith('.png')])
    frame_count = len(frame_files)
    video_count = frame_count // num_frames # discard last incomplete minute

    for i in range(video_count):
        video_name =  frame_files[i * num_frames][:8] + '.avi'
        print(video_name)
        video_path = os.path.join(output_folder, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(video_path, fourcc, frame_rate, (640, 480))

        for j in range(num_frames):
            frame_path = os.path.join(frame_folder, frame_files[i * num_frames + j])
            frame = cv2.imread(frame_path)
            out.write(frame)

        out.release()

if __name__ == '__main__':
    DATASET_FOLDER = '../datasets/20240404_18_33_38_fps1_clip_1_0/'
    NUM_FRAMES = 60
    fps = 10

    PATH_COLOR = DATASET_FOLDER + 'color'
    PATH_VIDEO = DATASET_FOLDER + 'color_video'
    os.makedirs(PATH_VIDEO, exist_ok=True)
    make_videos_from_frames(PATH_COLOR, PATH_VIDEO, frame_rate=fps, num_frames=NUM_FRAMES)
