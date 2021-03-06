from PIL import Image
import os
import numpy as np
import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def resize_image(img_arr, depth=False):
    img = Image.fromarray(img_arr)
    base_size = 224
    if img.size[0] < img.size[1]: #if width < height
        wpercent = (base_size / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_size, hsize), Image.ANTIALIAS)
    else:
        hpercent = (base_size / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, base_size), Image.ANTIALIAS)
    if not depth:
        return np.array(img)[:base_size, :base_size, :3]
    else:
        return np.array(img)[:base_size, :base_size]

TF_RECORDS_TRAIN_FILENAME = 'Datasets/Data/reside/tfrecords/train/reside.tfrecords'
TF_RECORDS_VALIDATION_FILENAME = 'Datasets/Data/reside/tfrecords/test/reside.tfrecords'

WRITER_TRAIN = tf.python_io.TFRecordWriter(TF_RECORDS_TRAIN_FILENAME)
WRITER_VALIDATION = tf.python_io.TFRecordWriter(TF_RECORDS_VALIDATION_FILENAME)

IMAGE_PATH = '/home/nautec/Downloads/reside/images/'
DEPTH_PATH = '/home/nautec/Downloads/reside/depth_fixed/'

IMAGES = sorted([IMAGE_PATH + f for f in os.listdir(IMAGE_PATH)])
DEPTHS = sorted([DEPTH_PATH + f for f in os.listdir(DEPTH_PATH)])

train_length = int(0.9 * len(IMAGES))
validation_length = len(IMAGES) - train_length
images_train = IMAGES[:train_length]
depths_train = DEPTHS[:train_length]

images_validation = IMAGES[-validation_length:]
depths_validation = DEPTHS[-validation_length:]

FILENAME_TRAIN_PAIRS = zip(images_train, depths_train)
FILENAME_VALIDATION_PAIRS = zip(images_validation, depths_validation)

for img_path, depth_path in FILENAME_TRAIN_PAIRS:
    print(img_path, depth_path)
    img = np.array(Image.open(img_path))
    if img.shape[2] > 3:
        continue
    depth = np.array(Image.open(depth_path)) #* (255.0/65535.0)
    depth = depth.astype(np.uint8)
    #print (img.shape[2], depth.shape)
    resized_image = resize_image(img)[:224, :224, :3]
    resized_depth = resize_image(depth, True)[:224, :224]
    image_raw = resized_image.tostring()
    depth_raw = resized_depth.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'depth_raw': _bytes_feature(depth_raw)}))
    WRITER_TRAIN.write(example.SerializeToString())

WRITER_TRAIN.close()


for img_path , depth_path in FILENAME_VALIDATION_PAIRS:
    print(img_path, depth_path)
    img = np.array(Image.open(img_path))[:224, :224, :]
    depth = np.array(Image.open(depth_path)) #* (255.0/65535.0)
    depth = depth.astype(np.uint8)[:224, :224]
    resized_image = resize_image(img)
    resized_depth = resize_image(depth, True)
    image_raw = resized_image.tostring()
    depth_raw = resized_depth.tostring()

    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'depth_raw': _bytes_feature(depth_raw)}))
    WRITER_VALIDATION.write(example.SerializeToString())

WRITER_VALIDATION.close()