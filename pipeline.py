import cv2
import numpy as np
import sys, os, pickle

from calibrate import ImageLoader

PIPELINE_CFG = {
    'calibration_path': 'calibration.p',
    'mask' : (0.08, 0.40),
    'font' : cv2.FONT_HERSHEY_SIMPLEX,
}

WHITE = (255, 255, 255)

class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.loader = ImageLoader(cfg['calibration_path'])

    def load_image(self, path):
        return self.loader.load_bgr(path)

    def mask_image(self, img, mask):        
        """
        Applies an image mask.
        
        Only keeps the region of the image defined by the polygon
        formed from `vertices`. The rest of the image is set to black.
        """
        if mask is None:
            return img

        imshape = img.shape

        mask_w, mask_h = mask

        vertices = np.array([[
            (0,imshape[0]),
            (imshape[1] * (0.5 - mask_w), imshape[0] * (1.0 - mask_h)), 
            (imshape[1] * (0.5 + mask_w), imshape[0] * (1.0 - mask_h)), 
            (imshape[1], imshape[0])
        ]], dtype=np.int32)


        #defining a blank mask to start with
        mask = np.zeros_like(img)   
        
        #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
        if len(img.shape) > 2:
            channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
            ignore_mask_color = (255,) * channel_count
        else:
            ignore_mask_color = 255
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image        

    def sobel(self, img):
        k_size=7
        sx_thresh = None #(20, 100), 
        sy_thresh = None #(0, 10), 
        mag_thresh = (50, 100)
        dir_thresh = (0.7, 1.3)

        img = np.copy(img)
        # Convert to HLS color space and separate the L channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        l_channel = hls[:,:,1]
        
        # Sobel x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=k_size) # Take the derivative in x
        sobely = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=k_size) # Take the derivative in 1

        abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
        abs_sobely = np.absolute(sobely) # Absolute x derivative to accentuate lines away from horizontal
        mag_sobel = np.sqrt(sobelx**2 + sobely**2) # Magnitude of gradient
        dir_sobel = np.arctan2(abs_sobely, abs_sobelx) # Direction of gradient
        
        scaled_sobelx = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
        scaled_sobely = np.uint8(255*abs_sobely/np.max(abs_sobely))
        scaled_mag = np.uint8(255*mag_sobel/np.max(mag_sobel))

        gradx= np.zeros_like(scaled_sobelx)
        if sx_thresh:
            gradx[(scaled_sobelx >= sx_thresh[0]) & (scaled_sobelx <= sx_thresh[1])] = 1

        grady = np.zeros_like(scaled_sobely)
        if sy_thresh:
            grady[(scaled_sobely >= sy_thresh[0]) & (scaled_sobely <= sy_thresh[1])] = 1

        mag_binary = np.zeros_like(mag_sobel)
        if mag_thresh:
            mag_binary[(scaled_mag >= mag_thresh[0]) & (scaled_mag <= mag_thresh[1])] = 1

        dir_binary = np.zeros_like(dir_sobel)
        if dir_thresh:
            dir_binary[(dir_sobel >= dir_thresh[0]) & (dir_sobel <= dir_thresh[1])] = 1
        
        combined = np.zeros_like(dir_sobel)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1    
        #combined[(mag_binary == 1)] = 1
        return combined
        
    def color_threshold(self, img):
        s_thresh=(170, 255)

        img = np.copy(img)
        # Convert to HLS color space and separate the S channel
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS).astype(np.float)
        s_channel = hls[:,:,2]
        # Threshold color channel
        s_binary = np.zeros_like(s_channel)
        if s_thresh:
            s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
        return s_binary

    def combined(self, img):
        sobel = self.sobel(img)
        s_binary = self.color_threshold(img)

        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sobel)
        combined_binary[(s_binary == 1) | (sobel == 1)] = 1        
        return combined_binary

    def warp(self, img, M, img_size):
        return cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)

    def draw_text(self, img, text, org, scale, color):      
        org = [org[0], org[1]]
        shape = img.shape
        if org[0] < 0:
            org[0] = shape[1] + org[0]
        if org[1] < 0:
            org[1] = shape[0] + org[1]
        org = (org[0], org[1])
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)  

    def draw_poly(self, img, points, closed, color, thickness=1):
        cv2.polylines(img, [np.array(points, np.int32)], closed, color, thickness)

    def run(self, path):
        img = self.load_image(path)
        img_orig = img.copy()

        img = self.combined(img)

        # img = self.mask_image(img, self.cfg.get('mask', None))

        # Center = 640, 420
        # [600, 450]
        src = np.array([(575, 450), (705, 450), (100, 680), (1180, 680)], np.float32)
        dst = np.array([(0, 0), (400, 0), (0, 500), (400, 500)], np.float32)

        M = cv2.getPerspectiveTransform(src, dst)

        #img = self.warp(img, M, (500, 500))

        #self.draw_text(img, 'Hello, World', (-500, 100), 2.0, WHITE)
        #self.draw_poly(img, [(0,0), (0, 100), (100, 100), (100, 0)], False, WHITE)
        return img

    def view(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()   


def main(args):
    pipeline = Pipeline(PIPELINE_CFG)

    for arg in args:
        pipeline.view(pipeline.run(arg))


if __name__ == '__main__':
    main(sys.argv[1:])