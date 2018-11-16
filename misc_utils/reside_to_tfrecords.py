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

def resize_image(img_arr):
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
    return np.array(img)[:base_size, :base_size, :3]

TF_RECORDS_TRAIN_FILENAME = 'Datasets/Data/reside/tfrecords/train/reside.tfrecords'
TF_RECORDS_VALIDATION_FILENAME = 'Datasets/Data/reside/tfrecords/test/reside.tfrecords'

WRITER_TRAIN = tf.python_io.TFRecordWriter(TF_RECORDS_TRAIN_FILENAME)
WRITER_VALIDATION = tf.python_io.TFRecordWriter(TF_RECORDS_VALIDATION_FILENAME)

IMAGE_PATH = '/home/nautec/Downloads/reside/images/'
HAZE_PATH = '/home/nautec/Downloads/reside/haze/'

IMAGES = sorted([IMAGE_PATH + f for f in os.listdir(IMAGE_PATH)])
HAZES = sorted([HAZE_PATH + f for f in os.listdir(HAZE_PATH)])

train_length = int(0.9 * len(IMAGES))
validation_length = len(IMAGES) - train_length
images_train = IMAGES[:train_length]
hazes_train = HAZES[:train_length]

images_validation = IMAGES[-validation_length:]
hazes_validation = HAZES[-validation_length:]

FILENAME_TRAIN_PAIRS = zip(images_train, hazes_train)
FILENAME_VALIDATION_PAIRS = zip(images_validation, hazes_validation)

for img_path, haze_path in FILENAME_TRAIN_PAIRS:
    print(img_path, haze_path)
    img = np.array(Image.open(img_path))
    img = resize_image(img)
    if img.shape[2] > 3:
        continue
    haze = np.array(Image.open(haze_path)) #* (255.0/65535.0)
    haze = resize_image(haze)
    image_raw = img.tostring()
    haze_raw = haze.tostring()
    print (img.shape, haze.shape)
    print (len(image_raw), len(haze_raw))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'haze_raw': _bytes_feature(haze_raw)}))
    WRITER_TRAIN.write(example.SerializeToString())

WRITER_TRAIN.close()


for img_path , haze_path in FILENAME_VALIDATION_PAIRS:
    print(img_path, haze_path)
    img = np.array(Image.open(img_path))
    img = resize_image(img)
    if img.shape[2] > 3:
        continue
    haze = np.array(Image.open(haze_path)) #* (255.0/65535.0)
    haze = resize_image(haze)
    image_raw = img.tostring()
    haze_raw = haze.tostring()
    print (img.shape, haze.shape)
    print (len(image_raw), len(haze_raw))
    example = tf.train.Example(features=tf.train.Features(feature={
        'image_raw': _bytes_feature(image_raw),
        'haze_raw': _bytes_feature(haze_raw)}))
    WRITER_VALIDATION.write(example.SerializeToString())

WRITER_VALIDATION.close()