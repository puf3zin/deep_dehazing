
import time

import numpy as np
import tensorflow as tf

from skimage import img_as_float
from skimage.io import imread, imsave

from guided_filter_tf.guided_filter import guided_filter

## GuidedFilter
print('GuidedFilter:')
## check forward
# forward on img
rgb = img_as_float(imread('tests/7in.jpg'))
gt  = img_as_float(imread('tests/7.jpg'))
x, y = [tf.constant(i.transpose((0, 1, 2))[None]) for i in [rgb, gt]]
print (x, y)
output = guided_filter(x, y, 64, 0, nhwc=True)

with tf.Session() as sess:
    start_time = time.time()
    r = sess.run(output)
    end_time = time.time()
print('\tForward on img ...')
print('\t\tTime: {}'.format(end_time - start_time))

r = r.squeeze().transpose(0, 1, 2)
r = np.asarray(r.clip(0, 1) * 255, dtype=np.uint8)
print (r.shape)
imsave('tests/r.jpg', r)