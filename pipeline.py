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

VCROP = (420, -80)

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
        self.reset_frames = 1000
        self.n_avg = 1
        self.base_lines = None          
        self.stage = None

    def process(self, img):
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # Image is BGR Colorspace
        img = img[VCROP[0]:VCROP[1],:]

        img_h, img_w = img.shape[:2]        


        if self.img_shape is None:
            self.img_shape = (img_w, img_h)

        self.warp_shape = (1280, 800)


        if self.M is None:
            self.M, self.Minv = self.perspective_transform(
                vp = (640, 0),
                tl = (520, 40),
                bl = (0, 220),
                out = self.warp_shape
            )

        # From warped image measurements
        #
        #   Lane Width = ~640px
        #   Dash Height = ~120px
        #
        #   Actual Width = 3.7m
        #   Actual Height = 3m
        #
        #   Xpx / Xm = 640 / 3.7
        #   Ypx / Ym = 120 / 3

        self.xpx_per_m = 640 / 3.7
        self.ypx_per_m = 120 / 3.0

        self.xm_per_pix = 3.7 / 640
        # self.ym_per_pix = 3.0 / 120.0

        # Add fudge factor to bring curvature closer to nominal
        # Perspective transform may need to be adjusted
        self.ym_per_pix = 1.5 * 3.0 / 120.0

        # print(self.img_shape)

        # return self.warp(img, self.warp_shape)
        # self.view(self.warp(img))

        # print(img.shape)
        img_conv = conv.process_image(img)

        if self.stage == 'CONV':
            return cv2.addWeighted(img, 0.25, (img_conv * 255.0).astype(np.uint8), 1, 0)

        # img_combined = np.dot(img_conv, np.array([1.0, 0.0, 0.0]).transpose() / 1.0)
        img_combined = img_conv[:,:,0] + img_conv[:,:,1] + img_conv[:,:,2]
        # print(img_combined.min(), img_combined.mean(), img_combined.max())
        img_combined[img_combined < 0.0] = 0
        img_combined[img_combined > 0.0] = 1.0       
        img_combined = img_combined.astype(np.uint8)
        if self.stage == 'COMB':
            img_out = np.dstack((img_combined, img_combined, img_combined)) * 255 # making the original road pixels 3 color channels
            return cv2.addWeighted(img, 0.5, img_out.astype(np.uint8), 1, 0)            


        if False:
            # img_out = np.dstack((img_conv, img_combined, img_combined)) * 255 # making the original road pixels 3 color channels
            img_out = cv2.addWeighted(img, 0.5, img_conv.astype(np.uint8), 0.5, 0)
            self.view(img_out)        


        img_warped = self.warp(img_combined)
        if self.stage == 'WARPED':
            # self.view(img_warped)
            # return img_warped
            return (np.dstack((img_warped, img_warped, img_warped)) * 255).astype(np.uint8)

        min_bins = 3

        fit_left, fit_right = self.fit_left, self.fit_right       
        num_left, num_right = 0, 0 
        # fit_left, num_left = None, 0
        # fit_right, num_right = None, 0

        # Reset every N rames

        reset = False

        if self.frames % self.reset_frames == 0:
            print('force reset frames')
            # Reset priors so find_line_with_priors is not used
            # fit_left, fit_right = None, None
            reset = True


        if fit_left is None:                
            print('fit_left is None')
        else:
            fit_left, num_left = self.find_line_with_priors(img_warped, fit_left, margin=20)
            if fit_left is None:
                print('find_line_with_priors lost left')

        if fit_right is None:                
            print('fit_right is None')
        else:
            fit_right, num_right = self.find_line_with_priors(img_warped, fit_right, margin=20)
            if fit_right is None:
                print('find_line_with_priors lost right')


        if fit_left is not None and fit_left[0] is not None and fit_right is not None and fit_right[0] is not None:
            if fit_left[0][0] == fit_right[0][0]:
                print("Both lanes the same")
                reset = True

        if reset:
            if self.base_lines is None:
                self.base_lines = self.find_base_lines(img_warped)
            
            left_x, right_x = self.base_lines
            if fit_left is None:
                print('search for left')
                fit_left, num_left = self.find_line(img_warped, left_x)

            if fit_right is None:
                print('search for right')
                fit_right, num_right = self.find_line(img_warped, right_x)        

        n_avg = self.n_avg

        if fit_left is None or num_left < min_bins:
            print("Lost left, using last fit", num_left)
            # print(fit_left, num_left)
            fit_left = self.fit_left
            num_right = self.num_right
        else:
            self.fit_arr_left.append(fit_left)
            self.fit_arr_left = self.fit_arr_left[-n_avg:]
            self.fit_left = np.concatenate([self.fit_arr_left]).mean(axis=0)
            if self.fit_left is None:
                print("fit_arr_left is None")
            self.num_left = num_left

        if fit_right is None or num_right < min_bins:
            print("Lost right, using last fit", num_right)
            # print(fit_right, num_right)
            fit_right = self.fit_right
            num_right = self.num_right
        else:         
            self.fit_arr_right.append(fit_right)
            self.fit_arr_right = self.fit_arr_right[-n_avg:]
            self.fit_right = np.concatenate([self.fit_arr_right]).mean(axis=0)
            if self.fit_right is None:
                print("fit_arr_right is None")
            self.num_right = num_right

        # print(fit_left, fit_right)
        # print(self.fit_left, self.fit_right)


        if False:
            # Overlay img_combined
            z = np.zeros_like(img_combined)
            tmp = np.dstack((img_combined, img_combined, img_combined)) * 255 # making the original road pixels 3 color channels
            # print(img_out.dtype, tmp.dtype)
            img = cv2.addWeighted(img_out, 0.5, tmp.astype(np.uint8), 1.0, 0)
            # self.view(img_out)        
        if False:
            # Overlay img_combined
            img_out = cv2.addWeighted(img_out, 0.5, (255 * img_conv).astype(np.uint8), 1, 0)
            # self.view(img_out)    

        img_out = self.draw_unwarped(img, img_warped, self.fit_left, self.fit_right)

        # Curve Radius

        # Evaluate radius at warped image bottom
        yeval_px = img_warped.shape[0]
        yeval_m = yeval_px * self.ym_per_pix

        left_radius_px, left_px = 0, 0
        right_radius_px, right_px = 0, 0
        left_radius_m, left_m = 0, 0
        right_radius_m, right_m = 0, 0

        # print(fit_left, fit_right)

        if fit_left is not None and fit_left[0] is not None:
            left_radius_px = self.curve_radius(fit_left[0], yeval_px)
            left_px = self.x_at(fit_left[0], yeval_px)

        if fit_right is not None and fit_right[0] is not None:
            right_radius_px = self.curve_radius(fit_right[0], yeval_px)
            right_px = self.x_at(fit_right[0], yeval_px)


        if fit_left is not None and fit_left[1] is not None:
            left_radius_m = self.curve_radius(fit_left[1], yeval_m)
            left_m = self.x_at(fit_left[1], yeval_m)
    
        if fit_right is not None and fit_right[1] is not None:
            right_radius_m = self.curve_radius(fit_right[1], yeval_m)
            right_m = self.x_at(fit_right[1], yeval_m)

        # Draw Info

        line = ''
        line += 'Bins: %d %d ' % (num_left, num_right)
        line += '| Radius: %dpx %dpx / %dm %dm ' % (left_radius_px, right_radius_px, left_radius_m, right_radius_m)
        # line += 'Pos: %dpx %dpx / %2.2fm %2.2fm ' % (left_px, right_px, left_m, right_m)
        if left_px and right_px and left_m and right_m:
            width_px = right_px - left_px
            width_m = right_m - left_m
            line += '| Width: %dpx / %2.2fm ' % (width_px, width_m)

            avg_px = (left_px + right_px) / 2
            avg_m = (left_m + right_m) / 2

            mid_px = img_warped.shape[1] // 2
            mid_m = mid_px * self.xm_per_pix

            pos_px = mid_px - avg_px
            pos_m = mid_m - avg_m

            line += '| Pos: %dpx / %2.2fm ' % (pos_px, pos_m)

        line += ''
        print(line)
        self.draw_text(img_out, line, (10, 20), 0.5, WHITE)


        self.frames += 1
        return img_out

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

    def find_line(self, img, x_base, nwindows=18, margin=20, minpix=300):
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

        h = np.histogram(y, bins=8, range=(0, img.shape[0]))
        bins = np.count_nonzero(h[0])
        

        if num_x > 0 :
            fit_px = np.polyfit(y, x, 2)
            fit_m = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
            fit = (fit_px, fit_m)
        else:
            fit = None
        
        return fit, bins

    def find_lines(self, img):
        left_x, right_x = self.find_base_lines(img)

        left_fit, left_num = self.find_line(img, left_x)
        right_fit, right_num = self.find_line(img, right_x)

        return left_fit, right_fit, left_num, right_num

    def find_line_with_priors(self, img, fit, margin=100):
        fit_px, fit_m = fit
        fit = fit_px

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

        h = np.histogram(y, bins=8, range=(0, img.shape[0]))
        bins = np.count_nonzero(h[0])

        if num > 0 :
            fit_px = np.polyfit(y, x, 2)
            fit_m = np.polyfit(y * self.ym_per_pix, x * self.xm_per_pix, 2)
            fit = (fit_px, fit_m)
        else:
            fit = None
        
        return fit, bins

    def find_lines_with_priors(self, img, left_fit, right_fit, margin=50, min_points=200, plot=False):
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

    def warp(self, img, size=None):
        if size is None:
            size = self.warp_shape
        return cv2.warpPerspective(img, self.M, size, flags=cv2.INTER_LINEAR)

    def unwarp(self, img, size=None):
        if size is None:
            size = self.img_shape
        return cv2.warpPerspective(img, self.Minv, size, flags=cv2.INTER_LINEAR)

    def draw_unwarped(self, undist, warped, left_fit, right_fit):
        if left_fit is None or right_fit is None:
            print("No lines")
            return undist
        # if self.num_left == 0 or self.num_right == 0:
        #     return undist

        left_fit = left_fit[0]
        right_fit = right_fit[0]

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

        # return color_warp.astype(np.uint8)

        newwarp = self.unwarp(color_warp)

        return cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

    def x_at(self, fit, y_eval):
        return fit[0] * y_eval ** 2 + fit[1] * y_eval + fit[2]

    def curve_radius(self, fit, y_eval):
        curverad = ((1 + (2 * fit[0] * y_eval + fit[1]) ** 2) ** 1.5) / np.absolute(2 * fit[0])
        # Example values - px: 1926.74 1908.48
        return curverad

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
        img_in = self.loader.undistort(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        img_out = self.process(img_in)
        return cv2.cvtColor(img_out, cv2.COLOR_BGR2RGB)

    def run(self, path):
        return self.process(self.load_image(path))

    def view(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()           

    def process_movie(self, path, out_path):
        #out_path = os.path.join('out_videos', path);
        print(path, out_path)
        if out_path.find('-1') > 0:
            clip = VideoFileClip(path).subclip(0, 1)
        elif out_path.find('-5') > 0:
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