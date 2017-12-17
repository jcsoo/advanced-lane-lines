import cv2
import numpy as np
import sys, os, pickle

# Camera Calibration Helpers

NX = 9
NY = 6

class ImageLoader:
    def __init__(self, calibration_path):
        self.path = calibration_path
        self.mat, self.dist = load_calibration(calibration_path)

    def load_bgr(self, path):
        return undistort(cv2.imread(path), self.mat, self.dist)

    def load_rgb(self, path):
        return cv2.cvtColor(self.load_bgr(path), cv2.COLOR_BGR2RGB)

    def load_hsv(self, path):
        return cv2.cvtColor(self.load_bgr(path), cv2.COLOR_BGR2HSV)

    def load_hsl(self, path):
        return cv2.cvtColor(self.load_bgr(path), cv2.COLOR_BGR2HSL)

    def load_gray(self, path):
        return cv2.cvtColor(self.load_bgr(path), cv2.COLOR_BGR2GRAY)

    def undistort(self, img):
        return undistort(img, self.mat, self.dist)

def calibration_images():
    base = './camera_cal'
    for name in os.listdir(base):
        path = os.path.join(base, name) 
        gray = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
        yield (name, gray)

def calibration_corners(nx, ny):
    for (name, gray) in calibration_images():
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        if ret:
            yield (name, gray, ret, corners)

def calibrate(nx=NX, ny=NY, interactive=False):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((NX*NY,3), np.float32)
    objp[:,:2] = np.mgrid[0:NX,0:NY].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    for (name, img, ret, corners) in calibration_corners(nx, ny):
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1, -1), criteria)
        imgpoints.append(corners)
        if interactive:
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(5000)        

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    if ret:
        return (mtx, dist)
    else:
        return None

def undistort(img, mtx, dist):
    return cv2.undistort(img, mtx, dist, None, mtx)

def view_calibration(mtx, dist):
    for (name, img) in calibration_images():
        dst = undistort(img, mtx, dist)
        cv2.imshow('img', dst)
        cv2.waitKey(5000)            

def save_calibration(path, mtx, dist):    
    with open(path, 'wb') as f:
        pickle.dump((mtx, dist), f)

def load_calibration(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def main(args):
    path = 'calibration.p'
    if args[0] == 'save':
        mtx, dist = calibrate()
        save_calibration(path, mtx, dist)
    elif args[0] == 'view':
        mtx, dist = load_calibration(path)
        view_calibration(mtx, dist)

if __name__ == '__main__':
    main(sys.argv[1:])