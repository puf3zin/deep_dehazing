import loss
import tensorflow as tf
class CrossEntropy(loss.Loss):
    def __init__(self):
        parameters_list = []
        self.config_dict = self.open_config(parameters_list)
    def evaluate(self, architecture_output, target_output):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=target_output,
                                                          logits=architecture_output))
