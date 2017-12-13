import cv2
import numpy as np
import matplotlib.pyplot as plt
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

    def mask_image_binary(self, img, mask):        
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
        ignore_mask_color = 1
            
        #filling pixels inside the polygon defined by "vertices" with the fill color    
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        
        #returning the image only where mask pixels are nonzero
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image            

    def sobel(self, img):
        k_size = 5
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

    def find_lines(self, img):
        ### From Class Notes

        # Choose the number of sliding windows
        nwindows = 9
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 20

        binary_warped = img
        # Find Lane Lines
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

        # Histogram of Warped Image
        histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
        # Find midpoint
        midpoint = np.int(histogram.shape[0]/2)
        # Find left peak
        leftx_base = np.argmax(histogram[:midpoint])
        # Find right peak
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # Set height of windows
        window_height = np.int(binary_warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window+1)*window_height
            win_y_high = binary_warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds] 

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)    
        plt.show()
        return (left_fit, right_fit, out_img)    

    def find_lines_with_priors(self, img, left_fit, right_fit):        
        binary_warped = img
        # Assume you now have a new warped binary image 
        # from the next frame of video (also called "binary_warped")
        # It's now much easier to find line pixels!

        margin = 50


        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + 
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + 
        left_fit[1]*nonzeroy + left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + 
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + 
        right_fit[1]*nonzeroy + right_fit[2] + margin)))  

        # Again, extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds] 
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

        # Create an image to draw on and an image to show the selection window
        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        # Color in left and right line pixels
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # Generate a polygon to illustrate the search window area
        # And recast the x and y points into usable format for cv2.fillPoly()
        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin, 
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, 
                                    ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0,255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0,255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)
        plt.imshow(result)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0) 
        plt.show()
        return (left_fit, right_fit, out_img)    
               

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
        # img = self.sobel(img)
        # img = self.color_threshold(img)
        img = self.mask_image_binary(img, self.cfg.get('mask', None))
        img_masked = img.copy()


        # Center = 640, 420
        # [600, 450]
        # src = np.array([(575, 450), (705, 450), (100, 680), (1180, 680)], np.float32)

        vp = (640, 420)
        tl = (520, 450)
        bl = (30, 615)

        src = np.array([
            (tl[0], tl[1]), 
            (vp[0] + (vp[0] - tl[0]), tl[1]), 
            (bl[0], bl[1]), 
            (vp[0] + (vp[0] - bl[0]), bl[1])],
        np.float32)
        dst = np.array([(0, 0), (1280, 0), (0, 720), (1280, 720)], np.float32)

        M = cv2.getPerspectiveTransform(src, dst)

        img = self.warp(img, M, (1280, 720))

        (fitx, fity, out_img) = self.find_lines(img)
        (fitx, fity, out_img) = self.find_lines_with_priors(img, fitx, fity)
        # pipeline.view(out_img)
        # self.display_lanes(img, fit)

        # plt.plot(histogram)
        # plt.show()

        #self.draw_text(img, 'Hello, World', (-500, 100), 2.0, WHITE)
        #self.draw_poly(img, [(0,0), (0, 100), (100, 100), (100, 0)], False, WHITE)
        return out_img

    def view(self, img):
        cv2.imshow('img', img)
        cv2.waitKey()   


def main(args):
    pipeline = Pipeline(PIPELINE_CFG)

    for arg in args:
        pipeline.view(pipeline.run(arg))


if __name__ == '__main__':
    main(sys.argv[1:])