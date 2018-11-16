from PIL import Image
import os
import numpy as np


DEPTH_PATH = '/home/nautec/Downloads/reside/depth_png/'
NEW_PATH = '/home/nautec/Downloads/reside/depth_fixed/'

depths = sorted([DEPTH_PATH + f for f in os.listdir(DEPTH_PATH)])

for depth_path in depths:
    depth_name = depth_path.split("/")[-1]
    depth = np.array(Image.open(depth_path)) #* (255.0/65535.0)
    depth = 255 - depth
    depth = 1 + (depth // 10)
    depth_im = Image.fromarray(depth)
    new_path = NEW_PATH + depth_name
    depth_im.save(new_path)
