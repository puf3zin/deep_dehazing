import architecture
import tensorflow as tf
#import Architectures.Layers.guided_filter as gf
from guided_filter_tf.guided_filter import fast_guided_filter


def get_low_resolution(hr_x, ssr=4):
    three = tf.constant([3.], dtype=tf.float32)
    hr_shape = tf.shape(hr_x)
    first = tf.cast(hr_shape[1] / ssr, tf.int32)
    second = tf.cast(hr_shape[2] / ssr, tf.int32)
    lr_shape = [first, second]
    lr_x = tf.image.resize_images(hr_x, lr_shape)
    return lr_x



class GuidedNet(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        " Coarse-scale Network"
        subsampling_ratio = 4
        lr_sample = get_low_resolution(sample, subsampling_ratio)
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        nc = 16
        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=nc, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=None,
                                         activation_fn=tf.nn.relu)

        encode1 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=2*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encode1)

        encode2 = tf.contrib.layers.conv2d(inputs=encode1, num_outputs=4*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encode2)                                 
        encode3 = tf.contrib.layers.conv2d(inputs=encode2, num_outputs=8*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encode3)
        encode4 = tf.contrib.layers.conv2d(inputs=encode3, num_outputs=8*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encode4)
        
        decode1 = tf.contrib.layers.conv2d_transpose(encode4, num_outputs=8*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)

        skip1 = tf.concat([encode3, decode1], 3)

        print(skip1)

        decode2 = tf.contrib.layers.conv2d_transpose(skip1, num_outputs=4*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        skip2 = tf.concat([encode2, decode2], 3)

        print(skip2)

        decode3 = tf.contrib.layers.conv2d_transpose(skip2, num_outputs=2*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        skip3 = tf.concat([encode1, decode3], 3)

        print(skip3)

        decode4 = tf.contrib.layers.conv2d_transpose(skip3, num_outputs=nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
                                                    
        skip4 = tf.concat([conv1, decode4], 3)

        conv4_1 = tf.contrib.layers.conv2d(inputs=skip4, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        lr_conv4_1 = get_low_resolution(conv4_1, subsampling_ratio)
        
        guided4_1 = fast_guided_filter(lr_sample, lr_conv4_1, sample,
                                       r=5, eps=10**-4, nhwc=True)

        conv4_2 = tf.contrib.layers.conv2d(inputs=guided4_1, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)

        lr_conv4_2 = get_low_resolution(conv4_2, subsampling_ratio)
        guided4_2 = fast_guided_filter(lr_sample, lr_conv4_2, sample,
                                       r=5, eps=10**-4, nhwc=True)
        
        guided4 = tf.concat([guided4_2*sample,guided4_1,sample],3)

        conv5 = tf.contrib.layers.conv2d(inputs=guided4, num_outputs=3, kernel_size=[1, 1],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=None,
                                         activation_fn=tf.nn.relu)

        brelu = tf.minimum(conv5,1)

        tf.summary.image("architecture_output", brelu)
        return brelu


    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
