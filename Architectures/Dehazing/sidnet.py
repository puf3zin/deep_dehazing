import architecture
import tensorflow as tf

class SIDNet(architecture.Architecture):
    def __init__(self):
        parameters_list = ['input_size', 'summary_writing_period',
                           "validation_period", "model_saving_period"]

        self.config_dict = self.open_config(parameters_list)
        self.input_size = self.config_dict["input_size"][0:2]

    def crop_and_concat(self, x1, x2):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2,
                      (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        reshaped = tf.reshape(x1_crop, x2_shape)
        return tf.concat([reshaped, x2], 3)

    def two_convs(self, channel, nc, padding='SAME'):
        conv1 = tf.contrib.layers.conv2d(inputs=channel, num_outputs=nc,
                                         kernel_size=[3, 3], stride=[1, 1],
                                         padding=padding, normalizer_fn=None,
                                         activation_fn=tf.nn.relu)

        conv2 = tf.contrib.layers.conv2d(inputs=conv1, num_outputs=nc,
                                         kernel_size=[3, 3], stride=[1, 1],
                                         padding=padding, normalizer_fn=None,
                                         activation_fn=tf.nn.relu)
        return conv2

    def downsample(self, channel):
        max_pool = tf.nn.max_pool(value=channel, ksize=[1, 2, 2, 1],
                                   strides=[1, 2, 2, 1], padding='SAME')

        return max_pool

    def upsample_and_concat(self, channel, to_concat, nc, normalizer_params):
        deconv1 = tf.contrib.layers.conv2d_transpose(channel,
                    num_outputs=nc, kernel_size=[2,2],stride=[2, 2],
                    padding='SAME', normalizer_fn=tf.contrib.layers.batch_norm,
                    normalizer_params=normalizer_params,
                    activation_fn=tf.nn.relu)
        concat = self.crop_and_concat(deconv1, to_concat)
        return concat

    def print_tensor(self, tensor, text):
        print(text, tensor)

    def process_single_channel(self, channel, nc, normalizer_params): # 224²x1
        conv1 = self.two_convs(channel, nc) # 220²x64
        encode1 = self.downsample(conv1) # 110²x64
        self.print_tensor(encode1, "encode1")
                            
        conv2 = self.two_convs(encode1, nc*2) # 106²x128
        encode2 = self.downsample(conv2) # 53²x128
        self.print_tensor(encode2, "encode2")

        conv3 = self.two_convs(encode2, nc*4) # 49²x256
        encode3 = self.downsample(conv3) # 25²x256
        self.print_tensor(encode3, "encode3")

        conv4 = self.two_convs(encode3, nc*8) # 21²x512

        decode1 = self.upsample_and_concat(conv4, conv3, nc*4,
                                           normalizer_params) # 42²x512
        self.print_tensor(decode1, "decode1")
        conv5 = self.two_convs(decode1, nc*4) # 38²x256
        self.print_tensor(conv5, "conv5")
        
        decode2 = self.upsample_and_concat(conv5, conv2, nc*2,
                                           normalizer_params) # 76²x256
        conv6 = self.two_convs(decode2, nc*2) # 72²x128
        self.print_tensor(conv6, "conv6")

        decode3 = self.upsample_and_concat(conv6, conv1, nc,
                                           normalizer_params) # 144²x128
        conv7 = self.two_convs(decode3, nc) # 140²x64
        self.print_tensor(conv7, "conv7")

        conv8 = tf.contrib.layers.conv2d(inputs=conv7, num_outputs=1,
                                         kernel_size=[1, 1], stride=[1, 1],
                                         padding='SAME', normalizer_fn=None,
                                         activation_fn=tf.nn.relu)
        self.print_tensor(conv8, "conv8")

        return conv8
        
    def prediction(self, sample, training=False):
        " Coarse-scale Network"
        normalizer_params = {'is_training':training, 'center':True,
                             'updates_collections':None, 'scale':True}
        
        nc = 32
        
        red, green, blue = tf.split(sample, num_or_size_splits=3, axis=3)
        #print (tf.shape(red), tf.shape(green), tf.shape(blue))

        processed_r = self.process_single_channel(red, nc, normalizer_params)
        self.print_tensor(processed_r, "processed_r")
        processed_g = self.process_single_channel(green, nc, normalizer_params)
        self.print_tensor(processed_g, "processed_g")
        processed_b = self.process_single_channel(blue, nc, normalizer_params)
        self.print_tensor(processed_b, "processed_b")

        processed_total = tf.concat([processed_r,
                                     processed_g,
                                     processed_b], axis=3)

        self.print_tensor(processed_total, "processed_total")
        brelu = tf.minimum(processed_total,1)
        self.print_tensor(brelu, "brelu")

        tf.summary.image("architecture_output", brelu)
        return brelu



    def get_validation_period(self):
        return self.config_dict["validation_period"]

    def get_model_saving_period(self):
        return self.config_dict["model_saving_period"]

    def get_summary_writing_period(self):
        return self.config_dict["summary_writing_period"]
