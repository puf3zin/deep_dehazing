from PIL import Image
import h5py
import numpy as np
import os

IMAGE_PATH = '/home/nautec/Downloads/reside/images/'
DEPTH_PATH = '/home/nautec/Downloads/reside/depth_png/'
depths = os.listdir(DEPTH_PATH)

count = 0
for depth in depths:
    img_path = IMAGE_PATH + depth.split('.')[0] + ".jpg"
    if not os.path.isfile(img_path):
        print(depth)
        os.remove(DEPTH_PATH + depth)
        count += 1
print (count)