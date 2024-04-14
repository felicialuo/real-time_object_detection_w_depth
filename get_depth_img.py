import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

depth_images = sorted([f for f in os.listdir("../figma_pres_imgs/exp2/depth")])

# video_path = "../figma_pres_imgs/exp2/depth_vidoe.avi"
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter(video_path, fourcc, 10, (640, 480))

# Read the npz files and store the depth images in a list
for img in depth_images:

    file_path = os.path.join("../figma_pres_imgs/exp2/depth", img)
    data = np.load(file_path)
    depth_image = data['arr_0']  # Assuming the array is stored with the default key 'arr_0'

    min_value = depth_image.min()
    max_value = depth_image.max()
    normalized_array = ((depth_image - min_value) / (max_value - min_value)) * 255
    normalized_array_int = normalized_array.astype(np.uint8)
    print(normalized_array_int.shape)

    cv2.imwrite(os.path.join("../figma_pres_imgs/exp2/depth_img", img.replace("npz", "jpg")), normalized_array_int)

    # plt.imshow(normalized_array_int, cmap='gray')
    # plt.show()
    # break

#     out.write(normalized_array_int)

# out.release()