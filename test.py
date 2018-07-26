import tensorflow as tf
zeros = tf.zeros([224, 224, 3])
hr_shape = tf.shape(zeros)
first = tf.cast(hr_shape[0] / 4, tf.int32)
second = tf.cast(hr_shape[1] / 4, tf.int32)
lr_shape = [first, second, hr_shape[2]]

sess = tf.Session()

print(sess.run(lr_shape))