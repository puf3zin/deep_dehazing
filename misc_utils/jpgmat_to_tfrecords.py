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
    img = np.array(Image.open(img_path))
    f = h5py.File(depth_path,'r')
    data = f.get('depth')
    depth = np.transpose(np.array(data))
    depth = depth.astype(np.uint8)
    pair = zip(cut_into_four(img), cut_into_four(depth))
    for img2, depth2 in pair:
        if (img2.shape == (224,224,3) and depth2.shape == (224,224)):
            #print ((img2.shape), (depth2.shape))
            image_raw = img2.tostring()
            depth_raw = depth2.tostring()

            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw),
                'depth_raw': _bytes_feature(depth_raw)}))
            WRITER_TRAIN.write(example.SerializeToString())
        else:
            print (img_path)
            print (img2.shape, depth2.shape)

WRITER_TRAIN.close()

print ("--------validation-------------")

for img_path, depth_path in FILENAME_VALIDATION_PAIRS:
    #print (img_path, depth_path)
    img = np.array(Image.open(img_path))
    f = h5py.File(depth_path,'r')
    data = f.get('depth')
    depth = np.transpose(np.array(data))
    depth = depth.astype(np.uint8)
    pair = zip(cut_into_four(img), cut_into_four(depth))
    for img2, depth2 in pair:
        if (img2.shape == (224,224,3) and depth2.shape == (224,224)):
            image_raw = img2.tostring()
            depth_raw = depth2.tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'image_raw': _bytes_feature(image_raw),
                'depth_raw': _bytes_feature(depth_raw)}))
            WRITER_VALIDATION.write(example.SerializeToString())
        else:
            print (img_path)
            print (img2.shape, depth2.shape)

WRITER_VALIDATION.close()
