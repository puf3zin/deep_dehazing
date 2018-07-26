from PIL import Image
import h5py
import numpy as np
import os
IMAGE_PATH = '/home/nautec/Downloads/reside/images/'
DEPTH_PATH = '/home/nautec/Downloads/reside/depth/'
paths = [DEPTH_PATH + f for f in os.listdir(DEPTH_PATH)]

count = 0
for path in paths:
    #img = np.array(Image.open(path))
    f = h5py.File(path,'r')
    data = f.get('depth')
    depth = np.transpose(np.array(data))
    depth = depth.astype(np.uint8)
    img_path = path.replace("depth", "images")
    img_path = img_path.replace("mat", "jpg")
    if (depth.shape[0] < 224 or depth.shape[1] < 224):
        print (depth.shape)
        print (path)
        print (img_path)
        #os.remove(path)
        #os.remove(depth_path)
        count += 1
        print (count)