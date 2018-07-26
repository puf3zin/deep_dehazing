import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

data_path = "Datasets/Data/reside/tfrecords/train/reside.tfrecords"

reconstructed_images = []
reconstructed_depths = []

record_iterator = tf.python_io.tf_record_iterator(path=data_path)

count = 0
for string_record in record_iterator:
    if count > 2:
        break
    count += 1
    example = tf.train.Example()
    example.ParseFromString(string_record)
    
    img_string = (example.features.feature['image_raw']
                                  .bytes_list
                                  .value[0])
    
    depth_string = (example.features.feature['depth_raw']
                                .bytes_list
                                .value[0])
    
    img_1d = np.fromstring(img_string, dtype=np.uint8)
    depth_1d = np.fromstring(depth_string, dtype=np.uint8)
    reconstructed_img = img_1d.reshape((224, 224, 3))
    reconstructed_depth = depth_1d.reshape((224, 224))
    
    reconstructed_images.append(reconstructed_img)
    reconstructed_depths.append(reconstructed_depth)

for image, depth in zip(reconstructed_images, reconstructed_depths):
    img = Image.fromarray(image, 'RGB')
    dph = Image.fromarray(depth, 'L')
    #img.save('my.png')
    #img.show()
    dph.show()
    print(depth)