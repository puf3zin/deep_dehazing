import PIL
from PIL import Image
import os
import numpy as np
import tensorflow as tf
import h5py

def cut_into_four(image):
    l = []
    l.append(image[:224, :224])
    l.append(image[-224:, :224])
    l.append(image[:224, -224:])
    l.append(image[-224:, -224:])
    return l

def cut_into_one(image):
    return image[:224, :224]

def resize_image(img_arr):
    img = Image.fromarray(img_arr)
    base_size = 224
    if img.size[0] < img.size[1]: #if width < height
        wpercent = (base_size / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_size, hsize), PIL.Image.ANTIALIAS)
    else:
        hpercent = (base_size / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, base_size), PIL.Image.ANTIALIAS)
    return np.array(img)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


TF_RECORDS_TRAIN_FILENAME = 'Datasets/Data/reside/tfrecords/train/reside.tfrecords'
TF_RECORDS_VALIDATION_FILENAME = 'Datasets/Data/reside/tfrecords/test/reside.tfrecords'

WRITER_TRAIN = tf.python_io.TFRecordWriter(TF_RECORDS_TRAIN_FILENAME)
WRITER_VALIDATION = tf.python_io.TFRecordWriter(TF_RECORDS_VALIDATION_FILENAME)

IMAGE_PATH = '/home/nautec/Downloads/reside/images/'
DEPTH_PATH = '/home/nautec/Downloads/reside/depth/'

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
    #print (img_path, depth_path)
    im = Image.open(img_path)
    img = resize_image(np.array(im))
    f = h5py.File(depth_path,'r')
    data = f.get('depth')
    depth = np.transpose(np.array(data))
    depth = resize_image(depth.astype(np.uint8))
    cut_img, cut_depth = cut_into_one(img), cut_into_one(depth)
    if (cut_img.shape == (224,224,3) and cut_depth.shape == (224,224)):
        #print ((cut_img.shape), (cut_depth.shape))
        image_raw = cut_img.tostring()
        depth_raw = cut_depth.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'depth_raw': _bytes_feature(depth_raw)}))
        WRITER_TRAIN.write(example.SerializeToString())
    else:
        print (img_path)
        print (cut_img.shape, cut_depth.shape)

WRITER_TRAIN.close()

print ("--------validation-------------")

for img_path, depth_path in FILENAME_VALIDATION_PAIRS:
    #print (img_path, depth_path)
    im = Image.open(img_path)
    img = resize_image(np.array(im))
    f = h5py.File(depth_path,'r')
    data = f.get('depth')
    depth = np.transpose(np.array(data))
    depth = resize_image(depth.astype(np.uint8))
    cut_img, cut_depth = cut_into_one(img), cut_into_one(depth)
    if (cut_img.shape == (224,224,3) and cut_depth.shape == (224,224)):
        image_raw = cut_img.tostring()
        depth_raw = cut_depth.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_raw': _bytes_feature(image_raw),
            'depth_raw': _bytes_feature(depth_raw)}))
        WRITER_VALIDATION.write(example.SerializeToString())
    else:
        print (img_path)
        print (cut_img.shape, cut_depth.shape)

WRITER_VALIDATION.close()
