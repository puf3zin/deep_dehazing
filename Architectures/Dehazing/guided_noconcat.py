import architecture
import tensorflow as tf
from guided_filter_tf.guided_filter import guided_filter

class GuidedNoConcat(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def prediction(self, sample, training=False):
        " Coarse-scale Network"
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        nc = 64
        conv1 = tf.contrib.layers.conv2d(inputs=sample, num_outputs=nc, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=None,
                                         activation_fn=tf.nn.relu)

        encod1 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=2*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encod1)

        encod2 = tf.contrib.layers.conv2d(inputs=encod1, num_outputs=4*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encod2)                                 
        encod3 = tf.contrib.layers.conv2d(inputs=encod2, num_outputs=8*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encod3)
        encod4 = tf.contrib.layers.conv2d(inputs=encod3, num_outputs=8*nc, kernel_size=[3, 3],
                                          stride=[2, 2], padding='SAME',
                                          normalizer_fn=tf.contrib.layers.batch_norm,
                                          normalizer_params=normalizer_params,
                                          activation_fn=tf.nn.relu)
        print(encod4)
        
        decode1 = tf.contrib.layers.conv2d_transpose(encod4, num_outputs=8*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)

        decode1 = tf.concat([encod3, decode1], 3)

        print(decode1)

        decode2 = tf.contrib.layers.conv2d_transpose(decode1, num_outputs=4*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        decode2 = tf.concat([encod2, decode2], 3)

        print(decode2)

        decode3 = tf.contrib.layers.conv2d_transpose(decode2, num_outputs=2*nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        decode3 = tf.concat([encod1, decode3], 3)

        print(decode3)

        decode4 = tf.contrib.layers.conv2d_transpose(decode3, num_outputs=nc,
                                                    kernel_size=[4,4],stride=[2, 2],
                                                    padding='SAME',
                                                    normalizer_fn=tf.contrib.layers.batch_norm,
                                                    normalizer_params=normalizer_params,
                                                    activation_fn=tf.nn.relu)
        decode4 = tf.concat([conv1, decode4], 3)
        
        print(decode4)

        conv4 = tf.contrib.layers.conv2d(inputs=decode4, num_outputs=3, kernel_size=[3, 3],
                                         stride=[1, 1], padding='SAME',
                                         normalizer_fn=tf.contrib.layers.batch_norm,
                                         normalizer_params=normalizer_params,
                                         activation_fn=tf.nn.relu)


        tf.summary.image("pre_guided", conv4)

        guided = guided_filter(sample, conv4, 8, 0, nhwc=True)

        tf.summary.image("post_guided", guided)

        # conv5 = tf.contrib.layers.conv2d(inputs=conv4, num_outputs=3, kernel_size=[1, 1],
        #                                  stride=[1, 1], padding='SAME',
        #                                  normalizer_fn=tf.contrib.layers.batch_norm,
        #                                  normalizer_params=normalizer_params,
        #                                  activation_fn=tf.nn.relu)

        brelu = tf.minimum(guided,1)
        tf.summary.image("architecture_output", brelu)
        return brelu



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
