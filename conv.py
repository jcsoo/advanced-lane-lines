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
    print(img.dtype)
    if img.dtype == np.float32 or img.dtype == np.float64:
        print("to uint8")
        img = (img * 255.0).astype(np.uint8)

    cv2.imshow(name, img)
    cv2.waitKey()

def equalize_hist(img):
    if False:
        # Global Histogram Equalization
        img[:,:,1] = cv2.equalizeHist(img[:,:,1])
        img[:,:,2] = cv2.equalizeHist(img[:,:,2])

    if False:
        # Striped Histogram Equalization
        steps = 4
        step = int(img.shape[0] / steps)
        for i in range(steps):
            img_row = img[(i * step):((i + 1) * step),:,:]
            img_row[:,:,1] = cv2.equalizeHist(img_row[:,:,1])
            img_row[:,:,2] = cv2.equalizeHist(img_row[:,:,2])
    return img

def filter_thresh(img, thresh):
    img[img < thresh[0]] = -1.0
    img[img > thresh[1]] = -1.0
    img[img > -1.0] = 1.0
    return img

def filter_stripes(img, bias=0.05, mul=5.0, wmul=1):
    img = np.concatenate([
        np.zeros((420, img.shape[1])),
        filter_vline(img[420:500,:], (2, 4 * wmul)),
        filter_vline(img[500:570,:], (2, 6 * wmul)),
        filter_vline(img[570:640,:], (4, 8 * wmul)),
        filter_vline(img[640:720,:], (8, 10 * wmul)),
    ])

    img -= bias
    img *= mul

    img[img < 0] = 0.0
    img[img > 1.0] = 1.0
    # img[img > 0] = 1.0
    # img = img ** 2
    # img[:,:,0] = -1.0
    return img

def process_image(img):
    orig = img.copy()
    img = equalize_hist(img)
    img = np.log(hsv_f32(img) / 2.0 + 1.0)
    # img = np.log(hls_f32(img) / 2.0 + 1.0)
    # img[:,:,0] = filter_thresh(img[:,:,0],[0.032, 0.044])
    img[:,:,0] = filter_stripes(1 - np.abs(img[:,:,0] - 0.035), 0.005, 30.0, wmul=2)
    img[:,:,1] = filter_stripes(img[:,:,1], 0.005, 20.0, wmul=2)
    img[:,:,2] = filter_stripes(img[:,:,2], 0.002, 20.0)

    # img[:,:,0] = 0
    # img[:,:,1] = 0
    # img[:,:,2] = 0
    # img = np.uint8(255 * img / np.max(img))
    return merge_images(orig, 0.5, img, 0.5)

def merge_images(im1, a1, im2, v2):
    if len(im2.shape) < 3:
        im2 = np.dstack([im2, im2, im2])
    if im2.dtype != np.uint8:
        im2 = (im2 * 255.0).astype(np.uint8)
    # print(im1.shape, im1.dtype, im2.shape, im2.dtype)
    return cv2.addWeighted(im1, a1, im2, v2, 0)

def identity(img):
    return img

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)[:,:,1]

def gray_f32(img):
    return (gray(img).astype(np.float32) / 255.0)

def hsv(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def hsv_f32(img):
    return (hsv(img).astype(np.float32) / 255.0)

def hls(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

def hls_f32(img):
    return (hls(img).astype(np.float32) / 255.0)


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