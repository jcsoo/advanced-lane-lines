# Convolution Tests

import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from calibrate import ImageLoader

def load(path):
    loader = ImageLoader('calibration.p')
    return loader.load_bgr(path)

def show(name, img):
    if img.dtype == np.float32:
        img = (127 + (img * 127.0)).astype(np.uint8)

    cv2.imshow(name, img)
    cv2.waitKey()

def process_image(img):
    img = gray_f32(img)
    img -= img.mean()
    img = filter_vline(img, (2, 8)) # range [-1.0,1.0]
    thresh = 0.05 
    img[img <= thresh] = -1.0
    img[img > thresh] = 1.0
    # img = np.uint8(255 * img / np.max(img))
    return img

def identity(img):
    return img

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,1]

def gray_f32(img):
    return (gray(img).astype(np.float32) / 128.0) - 1.0

def filter_vline(img, size):
    kernel = (np.concatenate([
        np.zeros(size, np.float32),
        np.ones(size, np.float32),
        np.ones(size, np.float32),
        np.zeros(size, np.float32),
    ], axis=1) - 0.5) / (size[0] * size[1] * 4)
    return cv2.filter2D(img, -1, kernel)
    

def filter_avg(img, size=(5, 5)):
    kernel = np.ones(size, np.float32) / (size[0] * size[1])
    return cv2.filter2D(img,-1,kernel)

def main(args):
    for arg in args:
        show(arg, process_image(load(arg)))


if __name__ == '__main__':
    main(sys.argv[1:])