import tensorflow as tf
import loss

from misc_utils import ms_ssim

class MSSSIM(loss.Loss):
    def __init__(self):
        parameters_list = []
        self.config_dict = self.open_config(parameters_list)

    def evaluate(self, architecture_input, architecture_output, target_output):
        print ("MSSSIM")
        ssim1 = ms_ssim.MultiScaleSSIM(architecture_output, target_output)
        return ssim1