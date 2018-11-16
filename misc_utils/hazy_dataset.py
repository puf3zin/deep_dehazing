import PIL
import cv2
import math
from PIL import Image
import numpy as np
import os

original_path = '/home/nautec/Downloads/UnannotatedHazyImages/UnannotatedHazyImages/'
output_path = '/home/nautec/Downloads/UnannotatedHazyImages/small_processed/'

SIDE = 224

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def resize_image(img_arr):
    img = Image.fromarray(img_arr)
    base_size = SIDE
    if img.size[0] < img.size[1]: #if width < height
        wpercent = (base_size / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((base_size, hsize), PIL.Image.ANTIALIAS)
    else:
        hpercent = (base_size / float(img.size[1]))
        wsize = int((float(img.size[0]) * float(hpercent)))
        img = img.resize((wsize, base_size), PIL.Image.ANTIALIAS)
    return np.array(img)

def cut(image):
    return image[:SIDE, :SIDE, :3]

images = [original_path + f for f in os.listdir(original_path)]
rotations = [0, 90, 180, 270]

for img_path in images:
    name = img_path.split("/")[-1]
    im = Image.open(img_path)
    w, h, d = np.array(im).shape
    print (w, h, d)
    for rotation in rotations:
        rotated = np.array(im.rotate(rotation, expand=True))
        new_w, new_h = largest_rotated_rect(w, h, math.radians(rotation))
        cropped = crop_around_center(rotated, new_w, new_h)
        resized = resize_image(cropped)
        cut_img = cut(resized)
        flipped1 = np.flipud(cut_img)
        flipped2 = np.fliplr(cut_img)
        if cut_img.shape == (SIDE, SIDE, 3):
            final = Image.fromarray(cut_img)
            final.save(output_path + str(rotation) + name)
            final_flipped1 = Image.fromarray(flipped1)
            final_flipped1.save(output_path + str(rotation) + "UD" + name)
            final_flipped2 = Image.fromarray(flipped2)
            final_flipped2.save(output_path + str(rotation) + "LR" + name)
        else:
            print(img_path)
            print(cut_img.shape)