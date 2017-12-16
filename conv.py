# Convolution Tests

import sys, os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from perspective import *
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

def filter_stripes(img, bias, mul, wmul=1):
    img = np.concatenate([
        filter_vline(img[0:80,:], (2, 4 * wmul)),
        filter_vline(img[80:150,:], (2, 6 * wmul)),
        filter_vline(img[150:220,:], (2, 6 * wmul)),
        # filter_vline(img[220:,:], (8, 10 * wmul)),
    ])
    img += bias
    img[img < 0] = 0.0
    # img[img > 1.0] = 1.0
    # img[img > 0] = 1.0
    # img = img ** 2
    # img[:,:,0] = -1.0

    if mul is None:
        img /= img.max()
    else:
        img *= mul

    return img


def filter_vline(img, size):
    kernel = (np.concatenate([
        np.zeros(size, np.float32),
        np.ones(size, np.float32),
        np.ones(size, np.float32),
        np.zeros(size, np.float32),
    ], axis=1) - 0.5) / (size[0] * size[1] * 4)

    kernel = cv2.GaussianBlur(kernel, (5, 5), 0, 0)    
    k2 = np.array([[-0.5, -0.5, 0.5, 0.5]])
    kernel = cv2.filter2D(kernel, -1, k2)
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)

def filter_vxline(img, size):
    kernel = (np.concatenate([
        np.zeros(size, np.float32),
        np.ones(size, np.float32),
        np.ones(size, np.float32),
        np.zeros(size, np.float32),
    ], axis=1) - 0.5) / (size[0] * size[1] * 4)
    # kernel = [-0.5, 0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, -0.5]
    # kernel = cv2.GaussianBlur(kernel, (3, 3), 0, 0)    
    # kernel2 = cv2.filter2D(np.array([kernel1]), -1, np.array([-1.0, 1.0]))
    # print(kernel1)
    # print(kernel2)
    


def midpoint_adjust(img):
    # Sample patch

    mid_x = int(img.shape[1] / 2)
    margin = 16
    mid = img[:, (mid_x - margin):(mid_x + margin),:]
    mid_mean = mid.mean(axis=1)

    img_mid = img.copy()
    for i in range(img.shape[1]):
        img_mid[:,i,:] = np.abs(img_mid[:,i,:] - mid_mean)
    return img_mid

def score(*args):
    print(args)
    return 0

def process_image(img):
    img = hsv_f32(equalize_hist(img))

    # img[:,:,0] = filter_stripes(1 - np.abs(img[:,:,0] - 0.035), 0.005, 30.0, wmul=2)

    h = filter_thresh(img[:,:,0], (0.080, 0.095))
    s = filter_stripes(img[:,:,1], -0.005, 10)
    v_s = filter_stripes(img[:,:,2], -0.005, 20)
    v_t = filter_thresh(img[:,:,2], (0.625, 1.0))

    h[h > 0] = 1.0
    s[s > 0] = 1.0
    v_s[v_s > 0] = 1.0
    v_t[v_t > 0] = 1.0

    kernel = np.ones((3, 3),np.uint8)

    # a = h * s 
    a_o =  cv2.dilate(h, kernel, iterations=1) * s

    b = s * v_s
    c = v_s * v_t
    # out[h > 0 and s > 0 and v_s > 0 and v_t > 0] = 1.0

    # show('out', v_s)


    # img[:,:,1] = filter_thresh(img[:,:,1], (0.35, 0.55))
    # img[:,:,2] = filter_thresh(img[:,:,2], (0.8, 1.0))

    # img[:,:,0] = 0
    # img[:,:,1] = 0
    # img[:,:,2] = 0

    # show('h', img[:,:,0])    
    return np.dstack([a_o, b, c])

def merge(im1, a1, im2, v2):
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


def filter_avg(img, size=(5, 5)):
    kernel = np.ones(size, np.float32) / (size[0] * size[1])
    return cv2.filter2D(img,-1,kernel)

def main(args):
    for arg in args:
        img = load(arg)
        show(arg, merge(img, 0.5, process_image(img), 0.5))


if __name__ == '__main__':
    main(sys.argv[1:])