# imports 
import numpy as np
from PIL import Image, ImageFilter
import random


def rotate_filter(image, angle=90):
    """ Rotates the image by the given angle in degrees.
    :param image: PIL image
    :param angle: angle in degrees
    
    """
    return image.rotate(angle)

def blur_filter(image, radius=2):
    """ Blurs the image.
    :param image: PIL image
    :param radius: radius of the blur kernel
    
    """ 
    return image.filter(ImageFilter.GaussianBlur(radius=radius))

def gaussian_noise_filter(image):
    """ Adds gaussian noise to the image.
    :param image: PIL image
    """

    # convert to numpy array
    image = np.asarray(image)
    row, col, ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.9
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    # convert image to float in range 0-1
    noise = gauss * 255

    # add noise to image and round if down or up if final value is 0 or 255
    noise = image + noise
    noise = np.where(noise < 0, 0, noise)
    noise = np.where(noise > 255, 255, noise)

    # convert back to PIL image
    return Image.fromarray(noise.astype('uint8'), 'RGB')

def salt_and_pepper_filter(image, prob=0.2):
    """ Adds salt and pepper noise to the image.
    :param image: PIL image
    :param prob: probability of pixel being set to 0 or 255
    """
    # convert image to numpy array
    image = np.asarray(image).copy()

    # set random pixels to 0 or 255 of random channels
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            
            if random.random() < prob/2:
                chanel = random.randint(0, image.shape[2] - 1)
                image[i, j, chanel] = 0
            elif random.random() < prob:
                chanel = random.randint(0, image.shape[2] - 1)
                image[i, j, chanel] = 255

    # convert back to PIL image
    image = Image.fromarray(image)
    return image

def mix_color_filter(image, mode=1):
    """ Mixes the color channels of the image.
    :param image: PIL image
    :param mode: mode of mixing
    """

    # Split the color channels
    r, g, b = image.split()

    # Mix the color channels
    if mode == 1:
        return Image.merge("RGB", (b, g, r))
    elif mode == 2:
        return Image.merge("RGB", (r, b, g))
    elif mode == 3:
        return Image.merge("RGB", (g, r, b))

