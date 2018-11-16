import os

import numpy as np
import matplotlib.pyplot as plt

from skimage import io
from skimage import img_as_float
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr



hazy_path = '/home/nautec/deep_dehazing/Evaluates/evaluate_input/'
dehazed_path = '/home/nautec/Documents/results/architecture/Unguided64/'

hazy_images = sorted([hazy_path + f for f in os.listdir(hazy_path)])
dehazed_images = sorted([dehazed_path + f for f in os.listdir(dehazed_path)])

for hazy, dehazed in zip(hazy_images, dehazed_images):
    print (hazy)
    hazy_im = img_as_float(io.imread(hazy))
    dehazed_im = img_as_float(io.imread(dehazed))
    result_ssim = ssim(hazy_im, dehazed_im, multichannel=True)
    result_psnr = psnr(hazy_im, dehazed_im)
    print ("SSIM:", result_ssim)
    #print ("PSNR:", result_psnr)