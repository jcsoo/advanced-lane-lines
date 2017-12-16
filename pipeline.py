import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys, os, pickle
from moviepy.editor import VideoFileClip

from calibrate import ImageLoader
import conv

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
        self.img_shape = None
        self.M = None
        self.Minv = None
        self.fit_left = None
        self.fit_right = None
        self.fit_arr_left = []
        self.fit_arr_right = []
        self.num_left = 0
        self.num_right = 0
        self.frames = 0


    def load_image(self, path):
        return self.loader.load_bgr(path)

    def find_base_lines(self, img):
        # Histogram of Warped Image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find midpoint
        midpoint = np.int(histogram.shape[0]/2)
        # Find left peak
        leftx_base = np.argmax(histogram[:midpoint])
        # Find right peak
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        return leftx_base, rightx_base

    def find_line(self, img, x_base, nwindows=24, margin=30, minpix=20):
        binary_warped = img

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)

        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Current positions to be updated for each window
        x_current = x_base

        # Create empty lists to receive left and right lane pixel indices
        lane_inds = []


        last_delta = 0

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_x_low = x_current - margin
            win_x_high = x_current + margin

            # Identify the nonzero pixels in x and y within the window
            good_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_x_low) &  (nonzerox < win_x_high)).nonzero()[0]

            # Append these indices to the lists
            lane_inds.append(good_inds)

            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_inds) > minpix:
                x_last = x_current
                x_current = np.int(np.mean(nonzerox[good_inds]))
                last_delta = x_current - x_last
            else:
                x_current += last_delta

        # Concatenate the arrays of indices
        lane_inds = np.concatenate(lane_inds)

        # Extract left and right line pixel positions
        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds] 

        num_x = np.count_nonzero(x)

        if num_x > 0 :
            fit = np.polyfit(y, x, 2)
        else:
            fit = None
        
        return fit, num_x

    def find_lines(self, img):
        left_x, right_x = self.find_base_lines(img)

        left_fit, left_num = self.find_line(img, left_x)
        right_fit, right_num = self.find_line(img, right_x)

        return left_fit, right_fit, left_num, right_num

    def find_line_with_priors(self, img, fit, margin=50):
        nonzero = img.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        lane_inds = (
            (nonzerox > (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] - margin)) &
            (nonzerox < (fit[0] * (nonzeroy ** 2) + fit[1] * nonzeroy + fit[2] + margin))
        )         

        x = nonzerox[lane_inds]
        y = nonzeroy[lane_inds]        

        num = np.count_nonzero(x)

        if num > 0:
            fit = np.polyfit(y, x, 2)
        else:
            fit = None

        return fit, num

    def find_lines_with_priors(self, img, left_fit, right_fit, margin=50, min_points=500, plot=False):
        binary_warped = img
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!

        lfit, lnum = self.find_line_with_priors(img, left_fit, margin)
        rfit, rnum = self.find_line_with_priors(img, right_fit, margin)

        if lnum < min_points:
            lfit = left_fit
        
        if rnum < min_points:
            rfit = right_fit

        return lfit, rfit, lnum, rnum

    def warp(self, img):
        return cv2.warpPerspective(img, self.M, self.img_shape, flags=cv2.INTER_LINEAR)

    def unwarp(self, img):
        return cv2.warpPerspective(img, self.Minv, self.img_shape, flags=cv2.INTER_LINEAR)

    def draw_unwarped(self, undist, warped, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            print("No lines")
            return undist
        # if self.num_left == 0 or self.num_right == 0:
        #     return undist

        image = undist
        binary_warped = warped

        ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0] )
        left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


        # Create an image to draw the lines on
        warp_zero = np.zeros_like(warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # print(left_fitx.shape, ploty.shape)

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

        newwarp = self.unwarp(color_warp)

        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

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

    def perspective_transform(self, vp, tl, bl, out):
        # Symmetric perspective transform based on vanishing point, top left, bottom left

        src = np.array([
            (tl[0], tl[1]), 
            (vp[0] + (vp[0] - tl[0]), tl[1]), 
            (bl[0], bl[1]), 
            (vp[0] + (vp[0] - bl[0]), bl[1])],
        np.float32)
        dst = np.array([(0, 0), (out[0], 0), (0, out[1]), (out[0], out[1])], np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        Minv = cv2.getPerspectiveTransform(dst, src)

        return M, Minv

    def process_video(self, img):
        return cv2.cvtColor(self.process(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)), cv2.COLOR_BGR2RGB)

    def process(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Image is BGR Colorspace
        img_h, img_w = img.shape[:2]        

        self.base_lines = None          

        if self.img_shape is None:
            self.img_shape = (img_w, img_h)

        if self.M is None:
            self.M, self.Minv = self.perspective_transform(
                vp = (640, 420),
                tl = (500, 480),
                bl = (0, 700),
                out = self.img_shape,
            )

        # self.view(self.warp(img))


        img_conv = conv.process_image(img)
        img_combined = np.dot(img_conv, np.array([0.0, 1.0, 1.0]).transpose() / 2.0)
        img_combined[img_combined > 0.1] = 1.0
     
        if False:
            # img_out = np.dstack((img_conv, img_combined, img_combined)) * 255 # making the original road pixels 3 color channels
            img_out = cv2.addWeighted(img, 0.5, img_conv.astype(np.uint8), 0.5, 0)
            self.view(img_out)        


        img_warped = self.warp(img_combined)
        # self.view(img_warped)

        min_points = 500

        fit_left, fit_right = self.fit_left, self.fit_right       
        num_left, num_right = 0, 0 
        # fit_left, num_left = None, 0
        # fit_right, num_right = None, 0

        # Reset every N rames

        if True or self.frames % 10 == 0:
            # Reset priors so find_line_with_priors is not used
            fit_left, fit_right = None, None


        if fit_left is not None:                
            fit_left, num_left = self.find_line_with_priors(img_warped, fit_left, margin=10)
            # print('left', fit_left, num_left)
            if num_left < min_points:
                print("Lost left lane")
                left_fit, num_left = None, 0
            # else:
            #     self.fit_left = fit_left
            #     self.num_left = num_left

        if fit_right is not None:                
            fit_right, num_right = self.find_line_with_priors(img_warped, fit_right, margin=10)
            # print("right", fit_right, num_right)
            if num_right < min_points:
                print("Lost right lane")
                right_fit, num_right = None, 0
            # else:
            #     self.fit_right = fit_right
            #     self.num_right = num_right

        if fit_left is not None and fit_right is not None:
            if fit_left[0] == fit_right[0]:
                print("Both lanes the same")
                fit_left, num_left = None, 0
                fit_right, num_right = None, 0


        if fit_left is None or fit_right is None:
            left_x, right_x = self.find_base_lines(img_warped)
            # print(left_x, right_x)
            if fit_left is None:
                fit_left, num_left = self.find_line(img_warped, left_x)

            if fit_right is None:
                fit_right, num_right = self.find_line(img_warped, right_x)        

        n_fit = 8

        if num_left < min_points:
            print("Lost left, using last fit")
            # print(fit_left, num_left)
            fit_left = self.fit_left
            num_right = self.num_right
        else:
            if False:
                self.fit_left = fit_left
                self.num_left = num_left
            elif fit_left is not None:
                self.fit_arr_left.append(fit_left)
                self.fit_arr_left = self.fit_arr_left[-n_fit:]
                self.fit_left = np.concatenate([self.fit_arr_left]).mean(axis=0)
                self.num_left = num_left

        if num_right < min_points:
            print("Lost right, using last fit")
            # print(fit_right, num_right)
            fit_right = self.fit_right
            num_right = self.num_right
        else:
            if False:
                self.fit_right = fit_right
                self.num_right = num_right
            elif fit_right is not None:              
                self.fit_arr_right.append(fit_right)
                self.fit_arr_right = self.fit_arr_right[-n_fit:]
                self.fit_right = np.concatenate([self.fit_arr_right]).mean(axis=0)
                self.num_right = num_right

        # print(fit_left, fit_right)
        # print(self.fit_left, self.fit_right)


        img_out = self.draw_unwarped(img, img_warped, self.fit_left, self.fit_right)
        if False:
            # Overlay img_combined
            z = np.zeros_like(img_combined)
            tmp = np.dstack((img_combined, img_combined, img_combined)) * 255 # making the original road pixels 3 color channels
            # print(img_out.dtype, tmp.dtype)
            img_out = cv2.addWeighted(img_out, 0.5, tmp.astype(np.uint8), 1.0, 0)
            # self.view(img_out)        
        if True:
            # Overlay img_combined
            img_out = cv2.addWeighted(img_out, 0.5, (255 * img_conv).astype(np.uint8), 1, 0)
            # self.view(img_out)        
        self.frames += 1
        return img_out

    def run(self, path):
        return self.process(self.load_image(path))

    def view(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()           

    def process_movie(self, path, out_path):
        #out_path = os.path.join('out_videos', path);
        print(path, out_path)
        if out_path.index('-1'):
            clip = VideoFileClip(path).subclip(0, 1)
        elif out_path.index('-5'):
            clip = VideoFileClip(path).subclip(0, 5)
        else:
            clip = VideoFileClip(path)
        out_clip = clip.fl_image(self.process_video)
        out_clip.write_videofile(out_path, audio=False)

def main(args):
    pipeline = Pipeline(PIPELINE_CFG)

    arg = args[0]
    if os.path.splitext(arg)[1] == '.jpg':
        pipeline.view(pipeline.run(arg))
    elif os.path.splitext(arg)[1] == '.mp4':
        pipeline.process_movie(arg, args[1])

if __name__ == '__main__':
    main(sys.argv[1:])