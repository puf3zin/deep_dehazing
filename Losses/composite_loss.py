import numpy as np
import tensorflow as tf

import Losses.discriminator as discriminator
import Losses.ssim as ssim
import Losses.l1_distance as l1_distance
import Losses.feature_loss as feature_loss

import loss

class CompositeLoss(loss.Loss):
    """This class is responsible for creating the DeepDive loss network, which is a
    mixture of SSIM with the Discriminator Loss.

    The class allows the usage of the DeepDive loss.
    """

    def __init__(self):
        """This constructor initializes a new DeepLoss object.

        The function loads the parameters from the deep_loss json and creates a new 
        object based on them.

        Returns:
            Nothing.
        """
        parameters_list = ["ssim_weight", "discriminator_weight", "l1_weight", "feature_weight"]
        self.open_config(parameters_list)
        print (self.config_dict)
        self.l1_weight = self.config_dict["l1_weight"]
        self.ssim_weight = self.config_dict["ssim_weight"]
        self.discriminator_weight = self.config_dict["discriminator_weight"]
        self.feature_weight = self.config_dict["feature_weight"]
        
        # Initialize Losses
        self.discriminator_loss = discriminator.DiscriminatorLoss()
        self.ssim_loss = ssim.SSIM()
        self.l1_distance = l1_distance.L1Distance()
        self.feature_loss = feature_loss.FeatureLoss()

    def evaluate(self, architecture_input, architecture_output, target_output):
        """This method evaluates the loss for the given image and it's ground-truth.

        The method models a discriminator neural network mixed with Feature Loss or MSE.

        Args:
            architecture_input: The image that's input in the generator network.

            architecture_output: The image to input in the deep loss.

            target_output: The ground-truth image to input in the deep loss.

        Returns:
            The value of the deep loss.
        """
        print ("composite loss")
        ssim_value = self.ssim_weight * \
                     self.ssim_loss.evaluate(architecture_input,
                                             architecture_output,
                                             target_output)
        gan_value  = self.discriminator_weight * \
                     self.discriminator_loss.evaluate(architecture_input,
                                                      architecture_output,
                                                      target_output)
        l1_value   = self.l1_weight * \
                     self.l1_distance.evaluate(architecture_input,
                                               architecture_output,
                                               target_output)
        feature_value = self.feature_weight * \
                        self.feature_loss.evaluate(architecture_input,
                                                   architecture_output,
                                                   target_output)

        print (ssim_value, gan_value, l1_value)
        print ("ssim, gan, l1")
        return ssim_value + gan_value + l1_value

    def train(self, optimizer_imp):
        """This method returns the training operation of the network.

        This method returns the training operation that is to be runned by tensorflow
        to minimize the deep dive network in relation to it's own error.

        Args:
            optimizer_imp: The implementation of the optimizer to use.

        Returns:
            The operation to run to optimize the deep dive network.
        """
        return self.discriminator_loss.train(optimizer_imp)

    def trainable(self):
        """This method tells whether this network is trainable or not.

        This method overrides the parent default method to make this network be trained on
        the main loop of the project.

        Returns:
            True, as the network is trainable.
        """
        return True
